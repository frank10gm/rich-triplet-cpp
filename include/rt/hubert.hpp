#pragma once

// =============================================================================
// HuBERT -- the codec's semantic encoder
// =============================================================================
//
// The eighth transformer here, and the first that is not a language model.
// HuBERT reads a raw waveform and produces one frame of features every 20 ms;
// OmniVoice's codec uses it as half of its analysis path, so that the codes it
// quantizes carry *what was said* and not only how it sounded.
//
// It is a wav2vec2-shaped encoder, which differs from every other transformer
// in this project in ways that all matter:
//
//   * **The input is audio, not tokens.** Seven strided convolutions turn
//     16 kHz samples into 768-wide frames at 50 Hz -- a 320x reduction, done
//     with no padding at all, so the length arithmetic is exact and unforgiving.
//   * **Position is a convolution, not a rotation.** There is no RoPE and no
//     learned table. A single depth-16 grouped convolution with a 128-wide
//     kernel is added to the hidden states once, before the first layer.
//   * **Attention is bidirectional and unmasked**, like the OmniVoice LM and
//     unlike everything else, so it reuses that kernel.
//   * **Post-layer-norm.** The norm comes *after* each residual add, which is
//     the original Transformer arrangement and the opposite of every other
//     model here. Swapping it does not crash; it quietly changes the features.
//   * **LayerNorm, not RMSNorm** -- mean subtraction included.
//   * **GELU is the erf form**, not the tanh approximation Gemma uses.
//
// ## What the codec actually asks for
//
// Not the last hidden state: the **mean of all thirteen**, the embedding output
// plus one per layer. That is what `mean_hidden_states` returns, and it is why
// this cannot be a normal `forward` that keeps only its final output.
//
// ## The length chain
//
// Every stage is a bare convolution with no padding, so lengths only shrink,
// and by exactly `floor((T - K) / S) + 1` each time:
//
//   16 kHz samples -> /5 -> /2 -> /2 -> /2 -> /2 -> /2 -> /2  = 320x
//
// which is 50 Hz. The codec then keeps every other frame to reach its own
// 25 Hz. Feeding it `n * 640 + 320` samples -- one second of 24 kHz audio
// resampled and padded -- yields exactly `2n` frames, and that identity is
// what the caller checks rather than trusting the chain.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "rt/gguf.hpp"
#include "rt/mat.hpp"
#include "rt/result.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

struct HubertConfig {
    std::size_t hidden_size = 768;
    std::size_t num_hidden_layers = 12;
    std::size_t num_attention_heads = 12;
    std::size_t intermediate_size = 3072;

    /// The convolutional feature extractor, one entry per layer.
    std::vector<std::size_t> conv_dim{512, 512, 512, 512, 512, 512, 512};
    std::vector<std::size_t> conv_kernel{10, 3, 3, 3, 3, 2, 2};
    std::vector<std::size_t> conv_stride{5, 2, 2, 2, 2, 2, 2};

    /// The positional convolution: kernel width and group count.
    std::size_t num_conv_pos_embeddings = 128;
    std::size_t num_conv_pos_embedding_groups = 16;

    float layer_norm_eps = 1e-5f;
    /// nn.GroupNorm's default, and separate from the above -- they are
    /// different layers and PyTorch does not give them the same epsilon.
    float group_norm_eps = 1e-5f;

    [[nodiscard]] static HubertConfig omnivoice_semantic();

    [[nodiscard]] std::size_t head_dim() const { return hidden_size / num_attention_heads; }

    /// Product of the convolution strides -- 320, so 16 kHz in, 50 Hz out.
    [[nodiscard]] std::size_t downsample_factor() const;

    /// How many frames `samples` 16 kHz samples produce, or 0 if the stack
    /// cannot consume them.
    [[nodiscard]] std::size_t feature_frames(std::size_t samples) const;
};

// =============================================================================
// Layers
// =============================================================================

/// One convolution of the feature extractor: convolve, normalise, GELU.
///
/// Only the first layer normalises. `feat_extract_norm="group"` puts a
/// GroupNorm with one group per channel -- so, per-channel over time -- on
/// layer 0 and nothing on the other six. A checkpoint carrying norms on every
/// layer would be the `"layer"` variant, which this is not.
struct HubertFeatureLayer {
    /// [Cout, Cin * K]
    Mat weight;
    std::size_t kernel = 1;
    std::size_t stride = 1;
    bool group_norm = false;
    std::vector<float> norm_weight;
    std::vector<float> norm_bias;

    [[nodiscard]] Mat forward(const Mat& x, float eps) const;
};

/// Post-layer-norm encoder block: attention, add, norm, feed-forward, add,
/// norm.
struct HubertEncoderLayer {
    /// All four are [hidden, hidden] and all four carry a bias, unlike the
    /// language models here.
    Mat q_weight, k_weight, v_weight, o_weight;
    std::vector<float> q_bias, k_bias, v_bias, o_bias;

    std::vector<float> attn_norm_weight, attn_norm_bias;

    /// [intermediate, hidden] then [hidden, intermediate].
    Mat fc1_weight, fc2_weight;
    std::vector<float> fc1_bias, fc2_bias;

    std::vector<float> final_norm_weight, final_norm_bias;

    [[nodiscard]] Mat forward(const Mat& x, const HubertConfig& cfg) const;
};

// =============================================================================
// Model
// =============================================================================

class HubertModel {
   public:
    HubertConfig config;

    std::vector<HubertFeatureLayer> feature_layers;

    /// `feature_projection`: normalise the 512-wide convolution output, then
    /// widen it to the model's 768.
    std::vector<float> proj_norm_weight, proj_norm_bias;
    Mat proj_weight;
    std::vector<float> proj_bias;

    /// `encoder.pos_conv_embed.conv`: [hidden, (hidden / groups) * K].
    Mat pos_conv_weight;
    std::vector<float> pos_conv_bias;

    std::vector<float> encoder_norm_weight, encoder_norm_bias;
    std::vector<HubertEncoderLayer> layers;

    /// Load from an open GGUF under `prefix` (`"semantic_model"` in the
    /// OmniVoice tokenizer file).
    [[nodiscard]] static Result<HubertModel> load(const GgufFile& gguf, HubertConfig cfg,
                                                  const std::string& prefix);

    /// Run the convolutional front end: 16 kHz samples to [T, conv_dim.back()].
    [[nodiscard]] Result<Mat> extract_features(std::span<const float> samples) const;

    /// The mean of all `num_hidden_layers + 1` hidden states, [T, hidden_size].
    ///
    /// Averaging is the codec's choice, not HuBERT's: the early layers carry
    /// acoustic detail and the late ones carry phonetic identity, and the
    /// quantizer was fit on the average of both.
    [[nodiscard]] Result<Mat> mean_hidden_states(std::span<const float> samples) const;

    [[nodiscard]] std::size_t parameter_count() const;
};

// =============================================================================
// Helpers (exposed for testing)
// =============================================================================

/// LayerNorm each row of `x` in place: subtract the mean, divide by the
/// standard deviation, scale and shift.
void layer_norm_rows(Mat& x, std::span<const float> weight, std::span<const float> bias,
                     float eps);

/// GroupNorm with one group per channel, over the time axis of a [T, C]
/// matrix. Equivalent to normalising each column on its own.
void group_norm_channels(Mat& x, std::span<const float> weight, std::span<const float> bias,
                         float eps);

/// The exact GELU, `x * 0.5 * (1 + erf(x / sqrt(2)))`.
///
/// Not `gelu_tanh`. The two agree to about 1e-3, which is invisible in a
/// language model's logits and is not what this checkpoint was trained with.
void gelu_erf_inplace(Mat& x);

}  // namespace rt
