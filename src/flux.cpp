#include "rt/flux.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <cstddef>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "rt/conv2d.hpp"
#include "rt/gguf.hpp"
#include "rt/init_rng.hpp"
#include "rt/mat.hpp"
#include "rt/qlinear.hpp"
#include "rt/result.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

FluxConfig FluxConfig::schnell() {
    FluxConfig c;
    c.guidance_embed = false;
    return c;
}

FluxConfig FluxConfig::dev() {
    FluxConfig c;
    c.guidance_embed = true;
    return c;
}

std::size_t FluxConfig::patch_dim() const { return in_channels * patch_size * patch_size; }

std::size_t FluxConfig::mlp_hidden() const {
    return static_cast<std::size_t>(static_cast<float>(hidden_size) * mlp_ratio);
}

// =============================================================================
// Patching
// =============================================================================

Mat flux_patchify(const Mat& z, std::size_t lat_h, std::size_t lat_w, std::size_t patch) {
    assert(z.rows == lat_h * lat_w && "flux_patchify: rows != lat_h * lat_w");
    assert(lat_h % patch == 0 && lat_w % patch == 0 && "flux_patchify: latent not divisible");

    const std::size_t channels = z.cols;
    const std::size_t gh = lat_h / patch;
    const std::size_t gw = lat_w / patch;
    Mat out = Mat::zeros(gh * gw, channels * patch * patch);

    for (std::size_t py = 0; py < gh; ++py) {
        for (std::size_t px = 0; px < gw; ++px) {
            float* dst = out.row_mut(py * gw + px).data();
            for (std::size_t c = 0; c < channels; ++c) {
                for (std::size_t ky = 0; ky < patch; ++ky) {
                    for (std::size_t kx = 0; kx < patch; ++kx) {
                        // Channel-major within the patch: index
                        // `c * patch^2 + ky * patch + kx`.
                        const std::size_t src = (py * patch + ky) * lat_w + (px * patch + kx);
                        dst[c * patch * patch + ky * patch + kx] = z.at(src, c);
                    }
                }
            }
        }
    }
    return out;
}

Mat flux_unpatchify(const Mat& tokens, std::size_t lat_h, std::size_t lat_w,
                    std::size_t channels, std::size_t patch) {
    const std::size_t gh = lat_h / patch;
    const std::size_t gw = lat_w / patch;
    assert(tokens.rows == gh * gw && "flux_unpatchify: token count != patch grid");
    assert(tokens.cols == channels * patch * patch && "flux_unpatchify: token width mismatch");

    Mat out = Mat::zeros(lat_h * lat_w, channels);
    for (std::size_t py = 0; py < gh; ++py) {
        for (std::size_t px = 0; px < gw; ++px) {
            const float* src = tokens.row(py * gw + px).data();
            for (std::size_t c = 0; c < channels; ++c) {
                for (std::size_t ky = 0; ky < patch; ++ky) {
                    for (std::size_t kx = 0; kx < patch; ++kx) {
                        const std::size_t dst = (py * patch + ky) * lat_w + (px * patch + kx);
                        out.at_mut(dst, c) = src[c * patch * patch + ky * patch + kx];
                    }
                }
            }
        }
    }
    return out;
}

Mat flux_image_ids(std::size_t lat_h, std::size_t lat_w, std::size_t patch) {
    const std::size_t gh = lat_h / patch;
    const std::size_t gw = lat_w / patch;
    Mat ids = Mat::zeros(gh * gw, 3);
    for (std::size_t y = 0; y < gh; ++y) {
        for (std::size_t x = 0; x < gw; ++x) {
            // Axis 0 stays zero: it exists for video, where it is the frame.
            ids.at_mut(y * gw + x, 1) = static_cast<float>(y);
            ids.at_mut(y * gw + x, 2) = static_cast<float>(x);
        }
    }
    return ids;
}

// =============================================================================
// Embeddings
// =============================================================================

std::vector<float> flux_timestep_embedding(float t, std::size_t dim, float max_period,
                                           float time_factor) {
    const std::size_t half = dim / 2;
    std::vector<float> out(dim, 0.0f);
    const float scaled = t * time_factor;
    for (std::size_t i = 0; i < half; ++i) {
        const float freq =
            std::exp(-std::log(max_period) * static_cast<float>(i) / static_cast<float>(half));
        const float arg = scaled * freq;
        // Cosines first. The usual convention is the other way round.
        out[i] = std::cos(arg);
        out[half + i] = std::sin(arg);
    }
    return out;
}

void flux_apply_rope(Mat& x, const Mat& ids, std::size_t n_heads, std::size_t head_dim,
                     const std::vector<std::size_t>& axes_dim, float theta) {
    assert(x.rows == ids.rows && "flux_apply_rope: token count mismatch");
    assert(ids.cols == axes_dim.size() && "flux_apply_rope: id width != axis count");
    assert(x.cols == n_heads * head_dim && "flux_apply_rope: width != n_heads * head_dim");

    // Precompute one angle per (token, pair). Every head applies the same
    // rotation, so this is computed once and reused 24 times.
    std::size_t pairs = 0;
    for (const std::size_t d : axes_dim) {
        pairs += d / 2;
    }
    assert(pairs * 2 == head_dim && "flux_apply_rope: axes_dim must sum to head_dim");

    std::vector<float> cos_tab(x.rows * pairs);
    std::vector<float> sin_tab(x.rows * pairs);
    for (std::size_t t = 0; t < x.rows; ++t) {
        std::size_t p = 0;
        for (std::size_t a = 0; a < axes_dim.size(); ++a) {
            const float pos = ids.at(t, a);
            const std::size_t dim = axes_dim[a];
            for (std::size_t i = 0; i < dim / 2; ++i, ++p) {
                // theta^(-2i/dim), matching `rope()`'s `arange(0, dim, 2)/dim`.
                //
                // Computed in f64. The reference does the same, and it is not
                // fussiness: `theta` is 10000 and the exponent is a ratio, so a
                // single-precision `pow` puts a relative error into every
                // frequency, which the position then multiplies up.
                const double exponent =
                    static_cast<double>(2 * i) / static_cast<double>(dim);
                const double omega = 1.0 / std::pow(static_cast<double>(theta), exponent);
                const double angle = static_cast<double>(pos) * omega;
                cos_tab[t * pairs + p] = static_cast<float>(std::cos(angle));
                sin_tab[t * pairs + p] = static_cast<float>(std::sin(angle));
            }
        }
    }

    for (std::size_t t = 0; t < x.rows; ++t) {
        float* row = x.row_mut(t).data();
        for (std::size_t h = 0; h < n_heads; ++h) {
            float* head = row + h * head_dim;
            for (std::size_t p = 0; p < pairs; ++p) {
                // Adjacent pairs, not split halves.
                const float a = head[2 * p];
                const float b = head[2 * p + 1];
                const float c = cos_tab[t * pairs + p];
                const float s = sin_tab[t * pairs + p];
                head[2 * p] = a * c - b * s;
                head[2 * p + 1] = a * s + b * c;
            }
        }
    }
}

// =============================================================================
// Internals
// =============================================================================

namespace {

/// LayerNorm with no affine parameters. Every norm in FLUX is one of these:
/// the scale and shift arrive from the modulation path instead.
[[nodiscard]] Mat layer_norm_noaffine(const Mat& x, float eps) {
    Mat out = Mat::zeros(x.rows, x.cols);
    const auto n = static_cast<double>(x.cols);
    for (std::size_t r = 0; r < x.rows; ++r) {
        const float* src = x.row(r).data();
        double sum = 0.0;
        for (std::size_t c = 0; c < x.cols; ++c) {
            sum += src[c];
        }
        const double mean = sum / n;
        double var = 0.0;
        for (std::size_t c = 0; c < x.cols; ++c) {
            const double d = static_cast<double>(src[c]) - mean;
            var += d * d;
        }
        const auto inv = static_cast<float>(1.0 / std::sqrt(var / n + static_cast<double>(eps)));
        const auto mean_f = static_cast<float>(mean);
        float* dst = out.row_mut(r).data();
        for (std::size_t c = 0; c < x.cols; ++c) {
            dst[c] = (src[c] - mean_f) * inv;
        }
    }
    return out;
}

/// `(1 + scale) * x + shift`, broadcast over rows.
void modulate_inplace(Mat& x, std::span<const float> shift, std::span<const float> scale) {
    assert(shift.size() == x.cols && scale.size() == x.cols && "modulate: width mismatch");
    for (std::size_t r = 0; r < x.rows; ++r) {
        float* row = x.row_mut(r).data();
        for (std::size_t c = 0; c < x.cols; ++c) {
            row[c] = (1.0f + scale[c]) * row[c] + shift[c];
        }
    }
}

/// `dst += gate * src`, broadcast over rows.
void gated_add_inplace(Mat& dst, const Mat& src, std::span<const float> gate) {
    assert(dst.rows == src.rows && dst.cols == src.cols && "gated_add: shape mismatch");
    for (std::size_t r = 0; r < dst.rows; ++r) {
        float* d = dst.row_mut(r).data();
        const float* s = src.row(r).data();
        for (std::size_t c = 0; c < dst.cols; ++c) {
            d[c] += gate[c] * s[c];
        }
    }
}

/// Split a `[1, k * hidden]` modulation projection into its triples.
[[nodiscard]] std::vector<FluxModulation> split_modulation(const Mat& m, std::size_t hidden,
                                                           std::size_t triples) {
    assert(m.rows == 1 && m.cols == triples * 3 * hidden && "split_modulation: width mismatch");
    std::vector<FluxModulation> out(triples);
    const float* src = m.row(0).data();
    for (std::size_t t = 0; t < triples; ++t) {
        const std::size_t base = t * 3 * hidden;
        // Order within a triple is shift, scale, gate.
        out[t].shift.assign(src + base, src + base + hidden);
        out[t].scale.assign(src + base + hidden, src + base + 2 * hidden);
        out[t].gate.assign(src + base + 2 * hidden, src + base + 3 * hidden);
    }
    return out;
}

/// Per-head RMSNorm over the head dimension, in place.
void head_rms_norm_inplace(Mat& x, std::size_t n_heads, std::size_t head_dim,
                           std::span<const float> scale, float eps) {
    assert(scale.size() == head_dim && "head_rms_norm: scale length != head_dim");
    for (std::size_t r = 0; r < x.rows; ++r) {
        float* row = x.row_mut(r).data();
        for (std::size_t h = 0; h < n_heads; ++h) {
            float* head = row + h * head_dim;
            double sq = 0.0;
            for (std::size_t c = 0; c < head_dim; ++c) {
                sq += static_cast<double>(head[c]) * head[c];
            }
            const auto inv = static_cast<float>(
                1.0 / std::sqrt(sq / static_cast<double>(head_dim) + static_cast<double>(eps)));
            for (std::size_t c = 0; c < head_dim; ++c) {
                head[c] = head[c] * inv * scale[c];
            }
        }
    }
}

/// Multi-head scaled dot-product attention with no mask.
///
/// Every token attends to every other one -- there is nothing causal about a
/// diffusion transformer, and the text tokens are as visible to the image as
/// the image is to itself.
[[nodiscard]] Mat attention(const Mat& q, const Mat& k, const Mat& v, std::size_t n_heads,
                            std::size_t head_dim) {
    const std::size_t t = q.rows;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Mat out = Mat::zeros(t, n_heads * head_dim);
    std::vector<float> scores(t);

    for (std::size_t h = 0; h < n_heads; ++h) {
        const std::size_t off = h * head_dim;
        for (std::size_t i = 0; i < t; ++i) {
            const float* qi = q.row(i).data() + off;
            float max_score = -std::numeric_limits<float>::infinity();
            for (std::size_t j = 0; j < t; ++j) {
                const float* kj = k.row(j).data() + off;
                float acc = 0.0f;
                for (std::size_t c = 0; c < head_dim; ++c) {
                    acc += qi[c] * kj[c];
                }
                scores[j] = acc * scale;
                max_score = std::max(max_score, scores[j]);
            }
            float denom = 0.0f;
            for (std::size_t j = 0; j < t; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                denom += scores[j];
            }
            const float inv = 1.0f / denom;
            float* dst = out.row_mut(i).data() + off;
            for (std::size_t j = 0; j < t; ++j) {
                const float weight = scores[j] * inv;
                const float* vj = v.row(j).data() + off;
                for (std::size_t c = 0; c < head_dim; ++c) {
                    dst[c] += weight * vj[c];
                }
            }
        }
    }
    return out;
}

/// Take columns `[from, to)` of every row.
[[nodiscard]] Mat slice_cols(const Mat& x, std::size_t from, std::size_t to) {
    Mat out = Mat::zeros(x.rows, to - from);
    for (std::size_t r = 0; r < x.rows; ++r) {
        const float* src = x.row(r).data();
        std::copy(src + from, src + to, out.row_mut(r).data());
    }
    return out;
}

/// Stack `a` on top of `b`. Both must be the same width.
[[nodiscard]] Mat concat_rows(const Mat& a, const Mat& b) {
    assert(a.cols == b.cols && "concat_rows: width mismatch");
    Mat out = Mat::zeros(a.rows + b.rows, a.cols);
    std::copy(a.data.begin(), a.data.end(), out.data.begin());
    std::copy(b.data.begin(), b.data.end(),
              out.data.begin() + static_cast<std::ptrdiff_t>(a.data.size()));
    return out;
}

/// Take rows `[from, to)`.
[[nodiscard]] Mat slice_rows(const Mat& x, std::size_t from, std::size_t to) {
    Mat out = Mat::zeros(to - from, x.cols);
    std::copy(x.data.begin() + static_cast<std::ptrdiff_t>(from * x.cols),
              x.data.begin() + static_cast<std::ptrdiff_t>(to * x.cols), out.data.begin());
    return out;
}

/// Join two matrices side by side.
[[nodiscard]] Mat concat_cols(const Mat& a, const Mat& b) {
    assert(a.rows == b.rows && "concat_cols: row mismatch");
    Mat out = Mat::zeros(a.rows, a.cols + b.cols);
    for (std::size_t r = 0; r < a.rows; ++r) {
        float* dst = out.row_mut(r).data();
        std::copy(a.row(r).begin(), a.row(r).end(), dst);
        std::copy(b.row(r).begin(), b.row(r).end(), dst + a.cols);
    }
    return out;
}

}  // namespace

// =============================================================================
// Loading
// =============================================================================

namespace {

[[nodiscard]] Result<QLinear> load_linear(const GgufFile& gguf, const std::string& name,
                                          std::size_t out_features, std::size_t in_features,
                                          bool with_bias) {
    const auto idx = gguf.find_tensor(name + ".weight");
    if (!idx) {
        return err("flux: missing tensor " + name + ".weight");
    }
    const GgufTensorInfo& info = gguf.tensor_info[*idx];
    if (info.n_elements() != out_features * in_features) {
        return err("flux: " + name + ".weight has " + std::to_string(info.n_elements()) +
                   " elements, expected " + std::to_string(out_features * in_features));
    }

    std::vector<float> bias;
    if (with_bias) {
        RT_TRY(b, load_gguf_vector(gguf, name + ".bias"));
        if (b.size() != out_features) {
            return err("flux: " + name + ".bias has " + std::to_string(b.size()) +
                       " elements, expected " + std::to_string(out_features));
        }
        bias = std::move(b);
    }

    switch (info.gguf_type) {
        case GgufType::Q4K: {
            RT_TRY(q, gguf.decode_q4k_to_q4kmat(*idx));
            return QLinear::from_q4k(std::move(q), out_features, in_features, std::move(bias));
        }
        case GgufType::Bf16: {
            RT_TRY(bits, gguf.decode_bf16(*idx));
            return QLinear::from_bf16(MatBf16(std::move(bits), out_features, in_features),
                                      std::move(bias));
        }
        case GgufType::F32:
        case GgufType::F16: {
            // Kept at full precision. Widening f16 to f32 is exact, and folding
            // it down to bfloat instead would throw away two mantissa bits for
            // no saving worth having -- there are four such tensors in FLUX,
            // 100 MB between them, and one of them is the final layer's
            // modulation, which sets the scale of every output channel.
            RT_TRY(f, gguf_tensor_to_f32(gguf, *idx));
            return QLinear::from_f32(Mat(std::move(f), out_features, in_features),
                                     std::move(bias));
        }
        default: {
            // Q6_K, Q8_0 and Q5_K land here: decode to f32 and fold to BF16,
            // which halves the resident cost and throws away less than the
            // source format already did.
            RT_TRY(f, gguf_tensor_to_f32(gguf, *idx));
            const Mat m(std::move(f), out_features, in_features);
            return QLinear::from_bf16(mat_to_bf16(m), std::move(bias));
        }
    }
}

[[nodiscard]] Result<std::vector<float>> load_scale(const GgufFile& gguf, const std::string& name,
                                                    std::size_t len) {
    RT_TRY(v, load_gguf_vector(gguf, name));
    if (v.size() != len) {
        return err("flux: " + name + " has " + std::to_string(v.size()) + " elements, expected " +
                   std::to_string(len));
    }
    return v;
}

}  // namespace

Result<void> flux_check_axes(const FluxConfig& cfg) {
    std::size_t axes_sum = 0;
    for (const std::size_t d : cfg.axes_dim) {
        // Each axis rotates adjacent pairs within its own slice, so an odd
        // width would leave one dimension of that slice unrotated and shift
        // every later axis off its own frequencies. The published FLUX axes --
        // 16, 56, 56 -- are all even for exactly this reason.
        if (d % 2 != 0) {
            return err("flux: RoPE axis dimension " + std::to_string(d) + " is not even");
        }
        axes_sum += d;
    }
    if (axes_sum != cfg.head_dim()) {
        return err("flux: axes_dim sums to " + std::to_string(axes_sum) + ", expected head_dim " +
                   std::to_string(cfg.head_dim()));
    }
    return {};
}

Result<FluxModel> FluxModel::load_gguf(const std::string& path, FluxConfig cfg) {
    RT_TRY_VOID(flux_check_axes(cfg));

    RT_TRY(gguf, GgufFile::open(path));

    FluxModel m;
    m.cfg = cfg;
    const std::size_t hidden = cfg.hidden_size;
    const std::size_t head_dim = cfg.head_dim();
    const std::size_t mlp_hidden = cfg.mlp_hidden();

    RT_TRY(img_in, load_linear(gguf, "img_in", hidden, cfg.patch_dim(), true));
    RT_TRY(txt_in, load_linear(gguf, "txt_in", hidden, cfg.context_dim, true));
    RT_TRY(t1, load_linear(gguf, "time_in.in_layer", hidden, 256, true));
    RT_TRY(t2, load_linear(gguf, "time_in.out_layer", hidden, hidden, true));
    RT_TRY(v1, load_linear(gguf, "vector_in.in_layer", hidden, cfg.pooled_dim, true));
    RT_TRY(v2, load_linear(gguf, "vector_in.out_layer", hidden, hidden, true));
    m.img_in = std::move(img_in);
    m.txt_in = std::move(txt_in);
    m.time_in_1 = std::move(t1);
    m.time_in_2 = std::move(t2);
    m.vector_in_1 = std::move(v1);
    m.vector_in_2 = std::move(v2);

    if (cfg.guidance_embed) {
        RT_TRY(g1, load_linear(gguf, "guidance_in.in_layer", hidden, 256, true));
        RT_TRY(g2, load_linear(gguf, "guidance_in.out_layer", hidden, hidden, true));
        m.guidance_in_1 = std::move(g1);
        m.guidance_in_2 = std::move(g2);
    }

    m.double_blocks.reserve(cfg.n_double_blocks);
    for (std::size_t i = 0; i < cfg.n_double_blocks; ++i) {
        const std::string p = "double_blocks." + std::to_string(i) + ".";
        FluxDoubleBlock b;
        RT_TRY(im, load_linear(gguf, p + "img_mod.lin", 6 * hidden, hidden, true));
        RT_TRY(iq, load_linear(gguf, p + "img_attn.qkv", 3 * hidden, hidden, true));
        RT_TRY(ip, load_linear(gguf, p + "img_attn.proj", hidden, hidden, true));
        RT_TRY(i1, load_linear(gguf, p + "img_mlp.0", mlp_hidden, hidden, true));
        RT_TRY(i2, load_linear(gguf, p + "img_mlp.2", hidden, mlp_hidden, true));
        RT_TRY(iqn, load_scale(gguf, p + "img_attn.norm.query_norm.scale", head_dim));
        RT_TRY(ikn, load_scale(gguf, p + "img_attn.norm.key_norm.scale", head_dim));
        b.img_mod = std::move(im);
        b.img_qkv = std::move(iq);
        b.img_proj = std::move(ip);
        b.img_mlp_in = std::move(i1);
        b.img_mlp_out = std::move(i2);
        b.img_norm.query_scale = std::move(iqn);
        b.img_norm.key_scale = std::move(ikn);

        RT_TRY(tm, load_linear(gguf, p + "txt_mod.lin", 6 * hidden, hidden, true));
        RT_TRY(tq, load_linear(gguf, p + "txt_attn.qkv", 3 * hidden, hidden, true));
        RT_TRY(tp, load_linear(gguf, p + "txt_attn.proj", hidden, hidden, true));
        RT_TRY(m1, load_linear(gguf, p + "txt_mlp.0", mlp_hidden, hidden, true));
        RT_TRY(m2, load_linear(gguf, p + "txt_mlp.2", hidden, mlp_hidden, true));
        RT_TRY(tqn, load_scale(gguf, p + "txt_attn.norm.query_norm.scale", head_dim));
        RT_TRY(tkn, load_scale(gguf, p + "txt_attn.norm.key_norm.scale", head_dim));
        b.txt_mod = std::move(tm);
        b.txt_qkv = std::move(tq);
        b.txt_proj = std::move(tp);
        b.txt_mlp_in = std::move(m1);
        b.txt_mlp_out = std::move(m2);
        b.txt_norm.query_scale = std::move(tqn);
        b.txt_norm.key_scale = std::move(tkn);

        m.double_blocks.push_back(std::move(b));
    }

    m.single_blocks.reserve(cfg.n_single_blocks);
    for (std::size_t i = 0; i < cfg.n_single_blocks; ++i) {
        const std::string p = "single_blocks." + std::to_string(i) + ".";
        FluxSingleBlock b;
        RT_TRY(mod, load_linear(gguf, p + "modulation.lin", 3 * hidden, hidden, true));
        RT_TRY(l1, load_linear(gguf, p + "linear1", 3 * hidden + mlp_hidden, hidden, true));
        RT_TRY(l2, load_linear(gguf, p + "linear2", hidden, hidden + mlp_hidden, true));
        RT_TRY(qn, load_scale(gguf, p + "norm.query_norm.scale", head_dim));
        RT_TRY(kn, load_scale(gguf, p + "norm.key_norm.scale", head_dim));
        b.modulation = std::move(mod);
        b.linear1 = std::move(l1);
        b.linear2 = std::move(l2);
        b.norm.query_scale = std::move(qn);
        b.norm.key_scale = std::move(kn);
        m.single_blocks.push_back(std::move(b));
    }

    RT_TRY(fm, load_linear(gguf, "final_layer.adaLN_modulation.1", 2 * hidden, hidden, true));
    RT_TRY(fl, load_linear(gguf, "final_layer.linear", cfg.patch_dim(), hidden, true));
    m.final_mod = std::move(fm);
    m.final_linear = std::move(fl);

    return m;
}

// =============================================================================
// Forward
// =============================================================================

Result<Mat> FluxModel::forward(const Mat& latent, std::size_t lat_h, std::size_t lat_w,
                               const Mat& context, std::span<const float> pooled, float timestep,
                               float guidance) const {
    const std::size_t hidden = cfg.hidden_size;
    const std::size_t head_dim = cfg.head_dim();
    const std::size_t patch = cfg.patch_size;

    if (latent.cols != cfg.in_channels) {
        return err("flux forward: latent has " + std::to_string(latent.cols) +
                   " channels, expected " + std::to_string(cfg.in_channels));
    }
    if (latent.rows != lat_h * lat_w) {
        return err("flux forward: latent rows != lat_h * lat_w");
    }
    if (lat_h % patch != 0 || lat_w % patch != 0) {
        return err("flux forward: latent dimensions must be multiples of the patch size");
    }
    if (context.cols != cfg.context_dim) {
        return err("flux forward: context width " + std::to_string(context.cols) +
                   " != context_dim " + std::to_string(cfg.context_dim));
    }
    if (pooled.size() != cfg.pooled_dim) {
        return err("flux forward: pooled vector has " + std::to_string(pooled.size()) +
                   " entries, expected " + std::to_string(cfg.pooled_dim));
    }
    RT_TRY_VOID(flux_check_axes(cfg));

    // --- Conditioning vector -------------------------------------------------
    const std::vector<float> t_emb = flux_timestep_embedding(timestep, 256);
    Mat vec = time_in_2.forward([&] {
        Mat h = time_in_1.forward(Mat(t_emb, 1, 256));
        silu_inplace(h);
        return h;
    }());

    if (cfg.guidance_embed) {
        const std::vector<float> g_emb = flux_timestep_embedding(guidance, 256);
        const Mat g = guidance_in_2.forward([&] {
            Mat h = guidance_in_1.forward(Mat(g_emb, 1, 256));
            silu_inplace(h);
            return h;
        }());
        for (std::size_t c = 0; c < hidden; ++c) {
            vec.at_mut(0, c) += g.at(0, c);
        }
    }

    {
        const Mat y(std::vector<float>(pooled.begin(), pooled.end()), 1, cfg.pooled_dim);
        const Mat p = vector_in_2.forward([&] {
            Mat h = vector_in_1.forward(y);
            silu_inplace(h);
            return h;
        }());
        for (std::size_t c = 0; c < hidden; ++c) {
            vec.at_mut(0, c) += p.at(0, c);
        }
    }

    // The modulation projections all read `silu(vec)`, not `vec`.
    Mat mod_input = vec;
    silu_inplace(mod_input);

    // Debug hook: dump intermediates for the parity harness. Off unless the
    // environment asks, and compiled out of nothing -- the branch costs a
    // getenv per forward pass, which is beneath measurement next to 12B
    // parameters.
    const char* dump_dir = std::getenv("RT_FLUX_DUMP");
    const auto dump = [&](const char* tag, const Mat& m) {
        if (dump_dir == nullptr) {
            return;
        }
        std::ofstream f(std::string(dump_dir) + "/" + tag + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(m.data.data()),
                static_cast<std::streamsize>(m.data.size() * sizeof(float)));
    };
    dump("vec", vec);

    // --- Streams -------------------------------------------------------------
    Mat img = img_in.forward(flux_patchify(latent, lat_h, lat_w, patch));
    Mat txt = txt_in.forward(context);
    dump("img_in", img);
    dump("txt_in", txt);

    const Mat img_ids = flux_image_ids(lat_h, lat_w, patch);
    // Text positions are all zero, so their rotation is the identity. That is
    // deliberate: the prompt has no place in the image's coordinate system.
    const Mat txt_ids = Mat::zeros(txt.rows, 3);
    const Mat ids = concat_rows(txt_ids, img_ids);
    const std::size_t n_txt = txt.rows;

    // --- Double-stream blocks ------------------------------------------------
    for (const FluxDoubleBlock& b : double_blocks) {
        const std::vector<FluxModulation> img_mod =
            split_modulation(b.img_mod.forward(mod_input), hidden, 2);
        const std::vector<FluxModulation> txt_mod =
            split_modulation(b.txt_mod.forward(mod_input), hidden, 2);

        Mat img_m = layer_norm_noaffine(img, cfg.layer_norm_eps);
        modulate_inplace(img_m, img_mod[0].shift, img_mod[0].scale);
        const Mat img_qkv = b.img_qkv.forward(img_m);

        Mat txt_m = layer_norm_noaffine(txt, cfg.layer_norm_eps);
        modulate_inplace(txt_m, txt_mod[0].shift, txt_mod[0].scale);
        const Mat txt_qkv = b.txt_qkv.forward(txt_m);

        // The fused projection is laid out [q | k | v], each head-major.
        Mat img_q = slice_cols(img_qkv, 0, hidden);
        Mat img_k = slice_cols(img_qkv, hidden, 2 * hidden);
        const Mat img_v = slice_cols(img_qkv, 2 * hidden, 3 * hidden);
        head_rms_norm_inplace(img_q, cfg.n_heads, head_dim, b.img_norm.query_scale,
                              cfg.qk_norm_eps);
        head_rms_norm_inplace(img_k, cfg.n_heads, head_dim, b.img_norm.key_scale,
                              cfg.qk_norm_eps);

        Mat q_txt = slice_cols(txt_qkv, 0, hidden);
        Mat k_txt = slice_cols(txt_qkv, hidden, 2 * hidden);
        const Mat v_txt = slice_cols(txt_qkv, 2 * hidden, 3 * hidden);
        head_rms_norm_inplace(q_txt, cfg.n_heads, head_dim, b.txt_norm.query_scale,
                              cfg.qk_norm_eps);
        head_rms_norm_inplace(k_txt, cfg.n_heads, head_dim, b.txt_norm.key_scale,
                              cfg.qk_norm_eps);

        // Text first, then image -- the order the position ids were built in.
        Mat q = concat_rows(q_txt, img_q);
        Mat k = concat_rows(k_txt, img_k);
        const Mat v = concat_rows(v_txt, img_v);
        flux_apply_rope(q, ids, cfg.n_heads, head_dim, cfg.axes_dim, cfg.rope_theta);
        flux_apply_rope(k, ids, cfg.n_heads, head_dim, cfg.axes_dim, cfg.rope_theta);

        const Mat attn = attention(q, k, v, cfg.n_heads, head_dim);
        const Mat txt_attn = slice_rows(attn, 0, n_txt);
        const Mat img_attn = slice_rows(attn, n_txt, attn.rows);


        gated_add_inplace(img, b.img_proj.forward(img_attn), img_mod[0].gate);
        {
            Mat h = layer_norm_noaffine(img, cfg.layer_norm_eps);
            modulate_inplace(h, img_mod[1].shift, img_mod[1].scale);
            Mat f = b.img_mlp_in.forward(h);
            gelu_tanh_inplace(f);
            gated_add_inplace(img, b.img_mlp_out.forward(f), img_mod[1].gate);
        }

        gated_add_inplace(txt, b.txt_proj.forward(txt_attn), txt_mod[0].gate);
        {
            Mat h = layer_norm_noaffine(txt, cfg.layer_norm_eps);
            modulate_inplace(h, txt_mod[1].shift, txt_mod[1].scale);
            Mat f = b.txt_mlp_in.forward(h);
            gelu_tanh_inplace(f);
            gated_add_inplace(txt, b.txt_mlp_out.forward(f), txt_mod[1].gate);
        }
        if (dump_dir != nullptr && &b == &double_blocks.front()) {
            dump("block0_img", img);
            dump("block0_txt", txt);
        }
    }

    if (dump_dir != nullptr) {
        dump("after_double", concat_rows(txt, img));
    }

    // --- Single-stream blocks ------------------------------------------------
    Mat x = concat_rows(txt, img);
    const std::size_t mlp_hidden = cfg.mlp_hidden();

    for (const FluxSingleBlock& b : single_blocks) {
        const std::vector<FluxModulation> mod =
            split_modulation(b.modulation.forward(mod_input), hidden, 1);

        Mat x_mod = layer_norm_noaffine(x, cfg.layer_norm_eps);
        modulate_inplace(x_mod, mod[0].shift, mod[0].scale);

        // One projection produces QKV and the MLP's up-projection together.
        const Mat fused = b.linear1.forward(x_mod);
        Mat q = slice_cols(fused, 0, hidden);
        Mat k = slice_cols(fused, hidden, 2 * hidden);
        const Mat v = slice_cols(fused, 2 * hidden, 3 * hidden);
        Mat mlp = slice_cols(fused, 3 * hidden, 3 * hidden + mlp_hidden);

        head_rms_norm_inplace(q, cfg.n_heads, head_dim, b.norm.query_scale, cfg.qk_norm_eps);
        head_rms_norm_inplace(k, cfg.n_heads, head_dim, b.norm.key_scale, cfg.qk_norm_eps);
        flux_apply_rope(q, ids, cfg.n_heads, head_dim, cfg.axes_dim, cfg.rope_theta);
        flux_apply_rope(k, ids, cfg.n_heads, head_dim, cfg.axes_dim, cfg.rope_theta);

        const Mat attn = attention(q, k, v, cfg.n_heads, head_dim);
        gelu_tanh_inplace(mlp);

        // Attention and MLP run in parallel from the same input and are joined
        // before a single output projection, rather than in sequence.
        const Mat joined = concat_cols(attn, mlp);
        gated_add_inplace(x, b.linear2.forward(joined), mod[0].gate);
    }

    // --- Final layer ---------------------------------------------------------
    Mat out_img = slice_rows(x, n_txt, x.rows);
    const Mat fm = final_mod.forward(mod_input);
    // Two values here, not three, and in the order shift then scale.
    const std::span<const float> shift(fm.row(0).data(), hidden);
    const std::span<const float> scale(fm.row(0).data() + hidden, hidden);

    Mat normed = layer_norm_noaffine(out_img, cfg.layer_norm_eps);
    modulate_inplace(normed, shift, scale);
    const Mat tokens = final_linear.forward(normed);

    return flux_unpatchify(tokens, lat_h, lat_w, cfg.in_channels, patch);
}

std::size_t FluxModel::parameter_count() const {
    const auto count = [](const QLinear& l) {
        return l.out_features * l.in_features + l.bias.size();
    };
    std::size_t n = count(img_in) + count(txt_in) + count(time_in_1) + count(time_in_2) +
                    count(vector_in_1) + count(vector_in_2) + count(guidance_in_1) +
                    count(guidance_in_2) + count(final_mod) + count(final_linear);
    for (const FluxDoubleBlock& b : double_blocks) {
        n += count(b.img_mod) + count(b.img_qkv) + count(b.img_proj) + count(b.img_mlp_in) +
             count(b.img_mlp_out) + b.img_norm.query_scale.size() + b.img_norm.key_scale.size();
        n += count(b.txt_mod) + count(b.txt_qkv) + count(b.txt_proj) + count(b.txt_mlp_in) +
             count(b.txt_mlp_out) + b.txt_norm.query_scale.size() + b.txt_norm.key_scale.size();
    }
    for (const FluxSingleBlock& b : single_blocks) {
        n += count(b.modulation) + count(b.linear1) + count(b.linear2) +
             b.norm.query_scale.size() + b.norm.key_scale.size();
    }
    return n;
}

std::size_t FluxModel::weight_bytes() const {
    std::size_t n = img_in.size_bytes() + txt_in.size_bytes() + time_in_1.size_bytes() +
                    time_in_2.size_bytes() + vector_in_1.size_bytes() +
                    vector_in_2.size_bytes() + guidance_in_1.size_bytes() +
                    guidance_in_2.size_bytes() + final_mod.size_bytes() +
                    final_linear.size_bytes();
    for (const FluxDoubleBlock& b : double_blocks) {
        n += b.img_mod.size_bytes() + b.img_qkv.size_bytes() + b.img_proj.size_bytes() +
             b.img_mlp_in.size_bytes() + b.img_mlp_out.size_bytes();
        n += b.txt_mod.size_bytes() + b.txt_qkv.size_bytes() + b.txt_proj.size_bytes() +
             b.txt_mlp_in.size_bytes() + b.txt_mlp_out.size_bytes();
    }
    for (const FluxSingleBlock& b : single_blocks) {
        n += b.modulation.size_bytes() + b.linear1.size_bytes() + b.linear2.size_bytes();
    }
    return n;
}

void FluxModel::free_weights() {
    for (QLinear* l : {&img_in, &txt_in, &time_in_1, &time_in_2, &vector_in_1, &vector_in_2,
                       &guidance_in_1, &guidance_in_2, &final_mod, &final_linear}) {
        l->free_weight();
    }
    double_blocks.clear();
    single_blocks.clear();
}

// =============================================================================
// Sampling
// =============================================================================

std::vector<float> flux_schedule(std::size_t steps, float shift) {
    std::vector<float> sigmas(steps + 1);
    for (std::size_t i = 0; i <= steps; ++i) {
        const float t = 1.0f - static_cast<float>(i) / static_cast<float>(steps);
        // The shift reparameterises the schedule toward the noisy end. At
        // shift == 1 it is the identity, which is what schnell wants.
        sigmas[i] = shift == 1.0f ? t : (shift * t) / (1.0f + (shift - 1.0f) * t);
    }
    return sigmas;
}

Result<Mat> flux_sample(const FluxModel& model, const Mat& context, std::span<const float> pooled,
                        const FluxSampleParams& params) {
    const FluxConfig& cfg = model.cfg;
    const std::size_t factor = 8;  // the VAE's spatial compression
    if (params.width % (factor * cfg.patch_size) != 0 ||
        params.height % (factor * cfg.patch_size) != 0) {
        return err("flux sample: width and height must be multiples of " +
                   std::to_string(factor * cfg.patch_size));
    }
    if (params.steps == 0) {
        return err("flux sample: steps must be > 0");
    }

    const std::size_t lat_h = params.height / factor;
    const std::size_t lat_w = params.width / factor;

    // Gaussian noise at sigma 1, which is where the schedule starts.
    InitRng rng(params.seed);
    Mat x = Mat::zeros(lat_h * lat_w, cfg.in_channels);
    for (float& v : x.data) {
        v = rng.next_normal();
    }

    const std::vector<float> sigmas = flux_schedule(params.steps, params.shift);
    for (std::size_t i = 0; i < params.steps; ++i) {
        RT_TRY(velocity, model.forward(x, lat_h, lat_w, context, pooled, sigmas[i],
                                       params.guidance));
        // Euler: one step of `dx = v dt` along a schedule that runs downward,
        // so the increment is negative.
        const float dt = sigmas[i + 1] - sigmas[i];
        for (std::size_t k = 0; k < x.data.size(); ++k) {
            x.data[k] += dt * velocity.data[k];
        }
    }
    return x;
}

}  // namespace rt
