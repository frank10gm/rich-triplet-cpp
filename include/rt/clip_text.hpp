#pragma once

// =============================================================================
// CLIP-L text encoder
// =============================================================================
//
// The smaller of FLUX's two text encoders, and the one whose output is a single
// vector rather than a sequence. T5 carries the prompt's content; this carries
// something closer to its overall register, and it enters the transformer
// through the same modulation path as the timestep.
//
// Twelve layers, width 768, twelve heads, learned positional embeddings, 77
// tokens. Ordinary except in three places:
//
// ## The mask is causal
//
// A text *encoder* that reads left to right looks like a mistake and is not.
// CLIP trains its text tower autoregressively-masked and pools from the last
// token, so bidirectional attention here changes every embedding it produces.
//
// ## The activation is `quick_gelu`
//
//   x * sigmoid(1.702 * x)
//
// A pre-tanh approximation of GELU that OpenAI trained with and that survives
// in the checkpoint. Substituting exact GELU or the tanh approximation shifts
// the pooled vector by a few percent -- enough to change which image a prompt
// produces, not enough to look broken.
//
// ## Pooling reads the end-of-text position
//
// The pooled vector is the final-layer-norm hidden state at the position of the
// EOT token, found by taking the **argmax over the token ids**. That works
// because EOT (49407) is the highest id in the vocabulary, and it is how
// diffusers does it. Taking the last position instead picks up padding, since
// the sequence is padded to 77 with EOT itself -- so the two agree only when
// the prompt is exactly 77 tokens long.
//
// FLUX uses the pooled output only. The per-token sequence is computed anyway
// because it is what the pooling reads.

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

struct ClipTextConfig {
    std::size_t vocab_size = 49408;
    std::size_t hidden_size = 768;
    std::size_t intermediate_size = 3072;
    std::size_t n_layers = 12;
    std::size_t n_heads = 12;
    std::size_t max_position_embeddings = 77;
    /// The highest id in the vocabulary, and the one pooling searches for.
    std::uint32_t eot_token_id = 49407;
    float layer_norm_eps = 1e-5f;

    /// openai/clip-vit-large-patch14, which is what FLUX and SD3 both use.
    [[nodiscard]] static ClipTextConfig large();
};

// =============================================================================
// Layers
// =============================================================================

struct ClipAttention {
    QLinear q;
    QLinear k;
    QLinear v;
    QLinear out;
    std::size_t n_heads = 0;
    std::size_t head_dim = 0;

    /// Causal, and scaled by `1/sqrt(head_dim)` -- unlike T5 next door, which
    /// is neither.
    [[nodiscard]] Mat forward(const Mat& x) const;
};

struct ClipMlp {
    QLinear fc1;
    QLinear fc2;

    [[nodiscard]] Mat forward(const Mat& x) const;
};

struct ClipLayer {
    std::vector<float> norm1_weight;
    std::vector<float> norm1_bias;
    ClipAttention attn;
    std::vector<float> norm2_weight;
    std::vector<float> norm2_bias;
    ClipMlp mlp;

    [[nodiscard]] Mat forward(const Mat& x, float eps) const;
};

// =============================================================================
// Encoder
// =============================================================================

struct ClipTextOutput {
    /// [T, hidden_size] after the final layer norm.
    Mat sequence;
    /// The row of `sequence` at the EOT position. This is what FLUX consumes.
    std::vector<float> pooled;
    /// Where the EOT token was found.
    std::size_t eot_index = 0;
};

class ClipTextEncoder {
   public:
    ClipTextConfig cfg;

    Mat token_embedding;     // [vocab_size, hidden]
    Mat position_embedding;  // [max_positions, hidden]
    std::vector<ClipLayer> layers;
    std::vector<float> final_norm_weight;
    std::vector<float> final_norm_bias;

    /// Load from a `clip_l.safetensors` in either the HuggingFace
    /// `text_model.*` layout or the flat ComfyUI one.
    [[nodiscard]] static Result<ClipTextEncoder> load(const std::string& path,
                                                      ClipTextConfig cfg);

    /// Encode token ids.
    ///
    /// The sequence is used as given: CLIP's positional embedding table has
    /// exactly `max_position_embeddings` rows, so anything longer is an error
    /// rather than something to wrap around.
    [[nodiscard]] Result<ClipTextOutput> forward(const std::vector<std::uint32_t>& tokens) const;

    [[nodiscard]] std::size_t parameter_count() const;

    void free_weights();
};

/// LayerNorm with affine parameters, accumulating in f64.
[[nodiscard]] Mat clip_layer_norm(const Mat& x, std::span<const float> weight,
                                  std::span<const float> bias, float eps);

}  // namespace rt
