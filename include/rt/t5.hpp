#pragma once

// =============================================================================
// T5 v1.1 -- encoder only
// =============================================================================
//
// FLUX conditions on two text encoders. This is the big one: T5-XXL, 4.7B
// parameters of encoder, producing a [256, 4096] sequence embedding that
// carries essentially all of the prompt's meaning. (The CLIP encoder next door
// contributes a single pooled vector.)
//
// It is an ordinary pre-norm transformer with three things that are not
// ordinary, all of them silent when wrong:
//
// ## 1. Attention scores are not scaled
//
// There is no `1 / sqrt(d_kv)`. T5 folds that factor into the initialisation
// of the query projection instead, so applying it here divides every score by
// 8 and the softmax comes out far too flat. The embeddings stay finite,
// smooth, and plausible; the image they condition just comes out generic --
// prompt-shaped but not prompt-specific. It is the single most expensive
// mistake in this file.
//
// ## 2. Relative position bias lives in layer 0 only
//
// T5 has no positional embedding and no RoPE. Instead the first layer owns a
// learned `[n_heads, 32]` table, indexed by a logarithmic bucketing of
// `key_pos - query_pos`, and the resulting `[n_heads, T, T]` bias is added to
// the scores of **every** layer. Recomputing it per layer from each layer's
// own weights fails loudly (the tensors do not exist); computing it once and
// forgetting to add it to the later layers does not.
//
// ## 3. The FFN is gated, and the gate uses the tanh GELU
//
//   h = gelu_tanh(x @ wi_0^T) * (x @ wi_1^T)
//   y = h @ wo^T
//
// T5 v1.0 has a single `wi` and a ReLU. v1.1 -- which is what the XXL
// checkpoint FLUX uses is -- has the gated pair. The two are not
// interchangeable and the tensor names differ, so this fails loudly.
//
// ## Norms
//
// RMSNorm, no bias, no mean subtraction, and the reciprocal square root is
// computed in f32 even when the weights are BF16 -- at `d_model = 4096` the
// sum of squares overflows half precision on ordinary activations.
//
// ## Tokenizer
//
// SentencePiece unigram, 32128 pieces, `</s>` appended and no BOS.
// `tokenizer.cpp` already has the Viterbi decoder this needs.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rt/mat.hpp"
#include "rt/qlinear.hpp"
#include "rt/result.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

struct T5Config {
    std::size_t vocab_size = 32128;
    std::size_t d_model = 4096;
    std::size_t d_ff = 10240;
    std::size_t n_layers = 24;
    std::size_t n_heads = 64;
    std::size_t d_kv = 64;
    /// Buckets in the relative position table. Half are exact, half logarithmic.
    std::size_t rel_attn_buckets = 32;
    /// Beyond this distance every offset falls in the last bucket.
    std::size_t rel_attn_max_distance = 128;
    float rms_norm_eps = 1e-6f;

    /// google/t5-v1_1-xxl, which is what FLUX and SD3 both use.
    [[nodiscard]] static T5Config xxl();
    /// google/t5-v1_1-xl, for a smaller test path.
    [[nodiscard]] static T5Config xl();
};

// =============================================================================
// Relative position bias
// =============================================================================

/// Map `key_pos - query_pos` onto a bucket index.
///
/// Bidirectional: the sign picks the half of the table, so `num_buckets` is
/// halved first. Within a half, the first quarter of the range is exact and
/// the rest is logarithmic out to `max_distance`.
///
/// The encoder is bidirectional, which is the case that uses both halves. The
/// unidirectional variant clamps negatives to zero instead, and using it here
/// makes every backwards offset collide onto bucket 0 -- the prompt still
/// encodes, it just stops distinguishing word order at range.
[[nodiscard]] std::size_t t5_relative_bucket(long relative_position, std::size_t num_buckets,
                                             std::size_t max_distance);

/// Build the `[n_heads, T, T]` bias, flattened to `[n_heads * T, T]`.
///
/// `table` is the layer-0 `relative_attention_bias.weight`, [num_buckets,
/// n_heads] as stored -- an nn.Embedding, so buckets are the rows.
[[nodiscard]] Mat t5_position_bias(const Mat& table, std::size_t seq_len, std::size_t n_heads,
                                   std::size_t num_buckets, std::size_t max_distance);

// =============================================================================
// Layers
// =============================================================================

struct T5Attention {
    QLinear q;
    QLinear k;
    QLinear v;
    QLinear o;
    std::size_t n_heads = 0;
    std::size_t d_kv = 0;

    /// `bias` is [n_heads * T, T] from `t5_position_bias`, shared by every
    /// layer. Scores are **not** divided by sqrt(d_kv).
    [[nodiscard]] Mat forward(const Mat& x, const Mat& bias) const;
};

struct T5FeedForward {
    QLinear wi_0;  // gate
    QLinear wi_1;  // up
    QLinear wo;

    [[nodiscard]] Mat forward(const Mat& x) const;
};

struct T5Block {
    std::vector<float> norm1;  // RMSNorm weight, [d_model]
    T5Attention attn;
    std::vector<float> norm2;
    T5FeedForward ff;

    [[nodiscard]] Mat forward(const Mat& x, const Mat& bias, float eps) const;
};

// =============================================================================
// Encoder
// =============================================================================

class T5Encoder {
   public:
    T5Config cfg;

    /// [vocab_size, d_model]. Held as BF16 or f32; at XXL this is 132 M
    /// parameters and only `T` of its rows are ever read, so it is gathered on
    /// the CPU and never quantized.
    Mat token_embedding;
    /// Layer 0's relative attention bias table, [num_buckets, n_heads].
    Mat rel_bias_table;
    std::vector<T5Block> blocks;
    std::vector<float> final_norm;

    /// Load from a GGUF encoder checkpoint (`city96/t5-v1_1-xxl-encoder-gguf`
    /// and friends), which stores the weights under the original T5 names.
    [[nodiscard]] static Result<T5Encoder> load_gguf(const std::string& path, T5Config cfg);

    /// Encode token ids to `[T, d_model]`.
    ///
    /// The caller pads or truncates to the length the diffusion model expects:
    /// 256 for FLUX.1-schnell, 512 for dev. T5 itself has no length limit --
    /// it has no positional embedding to run out of.
    [[nodiscard]] Result<Mat> forward(const std::vector<std::uint32_t>& tokens) const;

    [[nodiscard]] std::size_t parameter_count() const;

    /// Release every weight. The prompt embedding is computed once, and 2.8 GB
    /// is worth reclaiming before a 12B transformer starts.
    void free_weights();
};

/// RMSNorm with no bias and no mean subtraction, accumulating in f64.
[[nodiscard]] Mat t5_rms_norm(const Mat& x, std::span<const float> weight, float eps);

}  // namespace rt
