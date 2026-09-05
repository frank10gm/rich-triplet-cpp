#include "rt/t5.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "rt/conv2d.hpp"
#include "rt/gguf.hpp"
#include "rt/mat.hpp"
#include "rt/qlinear.hpp"
#include "rt/result.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

T5Config T5Config::xxl() {
    T5Config c;
    c.vocab_size = 32128;
    c.d_model = 4096;
    c.d_ff = 10240;
    c.n_layers = 24;
    c.n_heads = 64;
    c.d_kv = 64;
    return c;
}

T5Config T5Config::xl() {
    T5Config c;
    c.vocab_size = 32128;
    c.d_model = 2048;
    c.d_ff = 5120;
    c.n_layers = 24;
    c.n_heads = 32;
    c.d_kv = 64;
    return c;
}

// =============================================================================
// Norm
// =============================================================================

Mat t5_rms_norm(const Mat& x, std::span<const float> weight, float eps) {
    assert(weight.size() == x.cols && "t5_rms_norm: weight length != d_model");
    Mat out = Mat::zeros(x.rows, x.cols);
    for (std::size_t r = 0; r < x.rows; ++r) {
        const float* src = x.row(r).data();
        // f64 accumulation: at d_model 4096 the sum of squares of ordinary
        // activations is large enough that f32 loses digits off the end.
        double sq = 0.0;
        for (std::size_t c = 0; c < x.cols; ++c) {
            sq += static_cast<double>(src[c]) * src[c];
        }
        const auto inv = static_cast<float>(
            1.0 / std::sqrt(sq / static_cast<double>(x.cols) + static_cast<double>(eps)));
        float* dst = out.row_mut(r).data();
        for (std::size_t c = 0; c < x.cols; ++c) {
            dst[c] = src[c] * inv * weight[c];
        }
    }
    return out;
}

// =============================================================================
// Relative position bias
// =============================================================================

std::size_t t5_relative_bucket(long relative_position, std::size_t num_buckets,
                               std::size_t max_distance) {
    // Bidirectional: the sign selects the half of the table.
    std::size_t bucket = 0;
    num_buckets /= 2;
    if (relative_position > 0) {
        bucket = num_buckets;
    }
    const auto n = static_cast<std::size_t>(std::labs(relative_position));

    const std::size_t max_exact = num_buckets / 2;
    if (n < max_exact) {
        return bucket + n;
    }
    // Logarithmic beyond `max_exact`, saturating at the last bucket.
    const double ratio = std::log(static_cast<double>(n) / static_cast<double>(max_exact)) /
                         std::log(static_cast<double>(max_distance) /
                                  static_cast<double>(max_exact));
    const auto scaled = static_cast<std::size_t>(
        static_cast<double>(max_exact) + ratio * static_cast<double>(num_buckets - max_exact));
    return bucket + std::min(scaled, num_buckets - 1);
}

Mat t5_position_bias(const Mat& table, std::size_t seq_len, std::size_t n_heads,
                     std::size_t num_buckets, std::size_t max_distance) {
    assert(table.cols == n_heads && "t5_position_bias: table.cols != n_heads");
    Mat bias = Mat::zeros(n_heads * seq_len, seq_len);
    for (std::size_t q = 0; q < seq_len; ++q) {
        for (std::size_t k = 0; k < seq_len; ++k) {
            // `memory_position - query_position`, in that order. The other
            // order mirrors the bias and makes the encoder read backwards.
            const long rel = static_cast<long>(k) - static_cast<long>(q);
            const std::size_t bucket = t5_relative_bucket(rel, num_buckets, max_distance);
            for (std::size_t h = 0; h < n_heads; ++h) {
                bias.at_mut(h * seq_len + q, k) = table.at(bucket, h);
            }
        }
    }
    return bias;
}

// =============================================================================
// Layers
// =============================================================================

Mat T5Attention::forward(const Mat& x, const Mat& bias) const {
    const std::size_t t = x.rows;
    const std::size_t inner = n_heads * d_kv;

    const Mat q = this->q.forward(x);
    const Mat k = this->k.forward(x);
    const Mat v = this->v.forward(x);
    assert(q.cols == inner && "t5 attention: projection width != n_heads * d_kv");

    Mat context = Mat::zeros(t, inner);
    std::vector<float> scores(t);

    for (std::size_t h = 0; h < n_heads; ++h) {
        const std::size_t off = h * d_kv;
        for (std::size_t i = 0; i < t; ++i) {
            const float* qi = q.row(i).data() + off;
            float max_score = -std::numeric_limits<float>::infinity();
            for (std::size_t j = 0; j < t; ++j) {
                const float* kj = k.row(j).data() + off;
                float acc = 0.0f;
                for (std::size_t c = 0; c < d_kv; ++c) {
                    acc += qi[c] * kj[c];
                }
                // No 1/sqrt(d_kv). T5 folds it into the query initialisation,
                // and dividing here flattens every softmax in the model.
                acc += bias.at(h * t + i, j);
                scores[j] = acc;
                max_score = std::max(max_score, acc);
            }
            float denom = 0.0f;
            for (std::size_t j = 0; j < t; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                denom += scores[j];
            }
            const float inv = 1.0f / denom;
            float* out = context.row_mut(i).data() + off;
            for (std::size_t j = 0; j < t; ++j) {
                const float weight = scores[j] * inv;
                const float* vj = v.row(j).data() + off;
                for (std::size_t c = 0; c < d_kv; ++c) {
                    out[c] += weight * vj[c];
                }
            }
        }
    }
    return o.forward(context);
}

Mat T5FeedForward::forward(const Mat& x) const {
    Mat gate = wi_0.forward(x);
    gelu_tanh_inplace(gate);
    const Mat up = wi_1.forward(x);
    for (std::size_t i = 0; i < gate.data.size(); ++i) {
        gate.data[i] *= up.data[i];
    }
    return wo.forward(gate);
}

Mat T5Block::forward(const Mat& x, const Mat& bias, float eps) const {
    Mat h = attn.forward(t5_rms_norm(x, norm1, eps), bias);
    for (std::size_t i = 0; i < h.data.size(); ++i) {
        h.data[i] += x.data[i];
    }
    Mat f = ff.forward(t5_rms_norm(h, norm2, eps));
    for (std::size_t i = 0; i < f.data.size(); ++i) {
        f.data[i] += h.data[i];
    }
    return f;
}

// =============================================================================
// Loading
// =============================================================================

namespace {

/// GGUF stores a 2-D tensor's dimensions fastest-varying first, so a
/// `[out, in]` PyTorch weight has `ne = {in, out}`. Everything here wants the
/// PyTorch reading.
[[nodiscard]] Result<QLinear> load_linear(const GgufFile& gguf, const std::string& name,
                                          std::size_t out_features, std::size_t in_features) {
    const auto idx = gguf.find_tensor(name);
    if (!idx) {
        return err("t5: missing tensor " + name);
    }
    const GgufTensorInfo& info = gguf.tensor_info[*idx];
    if (info.n_elements() != out_features * in_features) {
        return err("t5: " + name + " has " + std::to_string(info.n_elements()) +
                   " elements, expected " + std::to_string(out_features * in_features));
    }

    switch (info.gguf_type) {
        case GgufType::Q4K: {
            // The only format kept packed. `decode_q4k_to_q4kmat` also flips
            // GGUF's [in, out] listing to the [out, in] this wants.
            RT_TRY(q, gguf.decode_q4k_to_q4kmat(*idx));
            return QLinear::from_q4k(std::move(q), out_features, in_features);
        }
        case GgufType::Bf16: {
            RT_TRY(bits, gguf.decode_bf16(*idx));
            return QLinear::from_bf16(MatBf16(std::move(bits), out_features, in_features));
        }
        default: {
            // Q6_K, Q8_0, Q5_K, F16 and F32 all land here: decode to f32 and
            // fold to BF16, which halves the resident cost and throws away
            // less than the source format already did.
            RT_TRY(f, gguf_tensor_to_f32(gguf, *idx));
            const Mat m(std::move(f), out_features, in_features);
            return QLinear::from_bf16(mat_to_bf16(m));
        }
    }
}

[[nodiscard]] Result<std::vector<float>> load_vec(const GgufFile& gguf, const std::string& name,
                                                  std::size_t len) {
    const auto idx = gguf.find_tensor(name);
    if (!idx) {
        return err("t5: missing tensor " + name);
    }
    RT_TRY(v, gguf_tensor_to_f32(gguf, *idx));
    if (v.size() != len) {
        return err("t5: " + name + " has " + std::to_string(v.size()) + " elements, expected " +
                   std::to_string(len));
    }
    return v;
}

[[nodiscard]] Result<Mat> load_mat(const GgufFile& gguf, const std::string& name,
                                   std::size_t rows, std::size_t cols) {
    const auto idx = gguf.find_tensor(name);
    if (!idx) {
        return err("t5: missing tensor " + name);
    }
    RT_TRY(v, gguf_tensor_to_f32(gguf, *idx));
    if (v.size() != rows * cols) {
        return err("t5: " + name + " has " + std::to_string(v.size()) + " elements, expected " +
                   std::to_string(rows * cols));
    }
    return Mat(std::move(v), rows, cols);
}

/// The encoder GGUFs keep the original T5 parameter names. A block's prefix.
[[nodiscard]] std::string block_prefix(std::size_t i) {
    return "encoder.block." + std::to_string(i) + ".layer.";
}

}  // namespace

Result<T5Encoder> T5Encoder::load_gguf(const std::string& path, T5Config cfg) {
    RT_TRY(gguf, GgufFile::open(path));

    T5Encoder e;
    e.cfg = cfg;

    RT_TRY(emb, load_mat(gguf, "shared.weight", cfg.vocab_size, cfg.d_model));
    e.token_embedding = std::move(emb);

    RT_TRY(rel, load_mat(gguf,
                         "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                         cfg.rel_attn_buckets, cfg.n_heads));
    e.rel_bias_table = std::move(rel);

    const std::size_t inner = cfg.n_heads * cfg.d_kv;
    e.blocks.reserve(cfg.n_layers);
    for (std::size_t i = 0; i < cfg.n_layers; ++i) {
        const std::string p = block_prefix(i);
        T5Block b;

        RT_TRY(n1, load_vec(gguf, p + "0.layer_norm.weight", cfg.d_model));
        b.norm1 = std::move(n1);

        RT_TRY(q, load_linear(gguf, p + "0.SelfAttention.q.weight", inner, cfg.d_model));
        RT_TRY(k, load_linear(gguf, p + "0.SelfAttention.k.weight", inner, cfg.d_model));
        RT_TRY(v, load_linear(gguf, p + "0.SelfAttention.v.weight", inner, cfg.d_model));
        RT_TRY(o, load_linear(gguf, p + "0.SelfAttention.o.weight", cfg.d_model, inner));
        b.attn.q = std::move(q);
        b.attn.k = std::move(k);
        b.attn.v = std::move(v);
        b.attn.o = std::move(o);
        b.attn.n_heads = cfg.n_heads;
        b.attn.d_kv = cfg.d_kv;

        RT_TRY(n2, load_vec(gguf, p + "1.layer_norm.weight", cfg.d_model));
        b.norm2 = std::move(n2);

        // v1.1's gated pair. A v1.0 checkpoint has a single `DenseReluDense.wi`
        // and fails here, which is the outcome worth having.
        RT_TRY(wi0, load_linear(gguf, p + "1.DenseReluDense.wi_0.weight", cfg.d_ff, cfg.d_model));
        RT_TRY(wi1, load_linear(gguf, p + "1.DenseReluDense.wi_1.weight", cfg.d_ff, cfg.d_model));
        RT_TRY(wo, load_linear(gguf, p + "1.DenseReluDense.wo.weight", cfg.d_model, cfg.d_ff));
        b.ff.wi_0 = std::move(wi0);
        b.ff.wi_1 = std::move(wi1);
        b.ff.wo = std::move(wo);

        e.blocks.push_back(std::move(b));
    }

    RT_TRY(fn, load_vec(gguf, "encoder.final_layer_norm.weight", cfg.d_model));
    e.final_norm = std::move(fn);

    return e;
}

// =============================================================================
// Forward
// =============================================================================

Result<Mat> T5Encoder::forward(const std::vector<std::uint32_t>& tokens) const {
    if (tokens.empty()) {
        return err("t5 forward: empty token sequence");
    }
    if (token_embedding.rows != cfg.vocab_size) {
        return err("t5 forward: embedding table not loaded");
    }

    const std::size_t t = tokens.size();
    Mat x = Mat::zeros(t, cfg.d_model);
    for (std::size_t i = 0; i < t; ++i) {
        if (tokens[i] >= cfg.vocab_size) {
            return err("t5 forward: token id " + std::to_string(tokens[i]) + " out of range");
        }
        const float* row = token_embedding.row(tokens[i]).data();
        std::copy(row, row + cfg.d_model, x.row_mut(i).data());
    }

    // Computed once from layer 0's table and reused by every layer. This is the
    // model's only source of positional information.
    const Mat bias = t5_position_bias(rel_bias_table, t, cfg.n_heads, cfg.rel_attn_buckets,
                                      cfg.rel_attn_max_distance);

    for (const T5Block& b : blocks) {
        x = b.forward(x, bias, cfg.rms_norm_eps);
    }
    return t5_rms_norm(x, final_norm, cfg.rms_norm_eps);
}

std::size_t T5Encoder::parameter_count() const {
    std::size_t n = token_embedding.numel() + rel_bias_table.numel() + final_norm.size();
    for (const T5Block& b : blocks) {
        n += b.norm1.size() + b.norm2.size();
        n += b.attn.q.out_features * b.attn.q.in_features;
        n += b.attn.k.out_features * b.attn.k.in_features;
        n += b.attn.v.out_features * b.attn.v.in_features;
        n += b.attn.o.out_features * b.attn.o.in_features;
        n += b.ff.wi_0.out_features * b.ff.wi_0.in_features;
        n += b.ff.wi_1.out_features * b.ff.wi_1.in_features;
        n += b.ff.wo.out_features * b.ff.wo.in_features;
    }
    return n;
}

void T5Encoder::free_weights() {
    token_embedding = Mat();
    rel_bias_table = Mat();
    for (T5Block& b : blocks) {
        b.attn.q.free_weight();
        b.attn.k.free_weight();
        b.attn.v.free_weight();
        b.attn.o.free_weight();
        b.ff.wi_0.free_weight();
        b.ff.wi_1.free_weight();
        b.ff.wo.free_weight();
    }
    blocks.clear();
}

}  // namespace rt
