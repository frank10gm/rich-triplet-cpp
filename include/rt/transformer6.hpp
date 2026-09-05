#pragma once

// =============================================================================
// OmniVoice LM -- a Qwen3 backbone used as a masked diffusion model
// =============================================================================
//
// Every other transformer in this project is autoregressive: sample a token,
// append it to a KV cache, run one more position. This one is not, and the
// difference reaches all the way down into the attention kernel.
//
// OmniVoice starts with every audio position masked, runs the **whole sequence**
// through the model, unmasks the positions it is most confident about, and
// repeats -- 32 times by default. So:
//
//   * **Attention is bidirectional.** A masked position has to see the
//     positions after it, or there would be nothing to condition on. Every
//     attention path elsewhere in this project is causal, so this needs its
//     own.
//   * **There is no KV cache.** Unmasking a position changes the hidden states
//     of every position that attends to it, which under a full mask is all of
//     them. Nothing carries between steps.
//   * **The head predicts eight codebooks at once.** Logits are
//     `[T, 8, 1025]`, produced by a single `[1024 -> 8200]` matmul and
//     reshaped, not by eight separate heads.
//
// The backbone itself is an ordinary Qwen3 0.6B: 28 blocks, hidden 1024, 16
// query heads over 8 KV heads, head_dim 128, SwiGLU, and per-head RMSNorm on Q
// and K -- the same `apply_per_head_norm` shape Gemma 3 uses.
//
// ## Audio tokens
//
// Eight codebooks of 1024 entries share one embedding table of 8200 rows, where
// row `i * 1025 + c` is code `c` of codebook `i`. The extra entry per codebook
// is the mask token, id 1024. A position's embedding is the **sum** across all
// eight codebooks, so a fully masked position still has a well-defined
// embedding -- the sum of the eight mask rows.
//
// ## RoPE pairing
//
// The same fork that made Orpheus emit noise for its first few tokens applies
// here: rotary embeddings can pair `i` with `i + head_dim/2` (half-split) or
// `2i` with `2i+1` (interleaved), and which is right depends on whether the
// GGUF converter permuted the Q and K weight rows. llama.cpp permutes for the
// `llama` architecture and not for Qwen3, which predicts half-split here.
//
// **Half-split is confirmed by listening.** Both settings produce output that
// passes every automated check -- finite, in range, speech-like statistics --
// because the two conventions rotate by the same angles and differ only in
// which pairs receive them. Only a human could separate them: half-split is
// intelligible speech, interleaved is unintelligible. The one metric that did
// hint was the interleaved output's DC offset of -0.03 against half-split's
// -0.0001, with a zero-crossing rate of 0.009 -- rumble rather than voice.
//
// The flag stays because it is the first thing to try when a new checkpoint
// sounds wrong, not because the answer is open.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rt/gguf.hpp"
#include "rt/mat.hpp"
#include "rt/nn2.hpp"
#include "rt/result.hpp"
#include "rt/tensor_node.hpp"
#include "rt/tokenizer.hpp"
#include "rt/transformer5.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

struct Config6 {
    std::size_t text_vocab_size = 151676;
    std::size_t hidden_size = 1024;
    std::size_t num_hidden_layers = 28;
    std::size_t num_attention_heads = 16;
    std::size_t num_key_value_heads = 8;
    std::size_t intermediate_size = 3072;
    std::size_t head_dim = 128;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;

    /// Codebooks the head predicts in parallel.
    std::size_t num_audio_codebook = 8;
    /// 1024 codes plus the mask token.
    std::size_t audio_vocab_size = 1025;
    /// The id standing for "not yet decided".
    std::size_t audio_mask_id = 1024;

    /// Special marker ids, read from the checkpoint metadata.
    std::size_t text_start = 151674;
    std::size_t text_end = 151675;
    std::size_t lang_start = 151670;
    std::size_t lang_end = 151671;
    std::size_t instruct_start = 151672;
    std::size_t instruct_end = 151673;
    std::size_t denoise = 151669;

    /// See the header. Half-split, confirmed by listening; interleaved is
    /// unintelligible despite passing every automated check.
    RopePairing rope_pairing = RopePairing::HalfSplit;

    [[nodiscard]] static Config6 omnivoice();

    /// Rows in the shared audio embedding table.
    [[nodiscard]] std::size_t audio_table_size() const {
        return num_audio_codebook * audio_vocab_size;
    }

    /// Where codebook `i`'s block starts in that table.
    [[nodiscard]] std::size_t audio_row(std::size_t codebook, std::size_t code) const {
        return codebook * audio_vocab_size + code;
    }

    [[nodiscard]] std::vector<float> inv_freq() const;
};

// =============================================================================
// Layers
// =============================================================================

/// Bidirectional grouped-query attention with per-head Q/K normalization.
///
/// No cache and no causal mask: every position attends to every other.
class OmniAttention {
   public:
    Linear2 q_proj;
    Linear2 k_proj;
    Linear2 v_proj;
    Linear2 o_proj;
    /// Per-head RMSNorm over `head_dim`, applied before RoPE.
    RmsNorm2 q_norm;
    RmsNorm2 k_norm;
    std::size_t n_q_heads = 0;
    std::size_t n_kv_heads = 0;
    std::size_t head_dim = 0;
    float attn_scale = 0.0f;
    RopePairing rope_pairing = RopePairing::HalfSplit;

    explicit OmniAttention(const Config6& cfg);

    /// `inv_freq` is passed in rather than held.
    ///
    /// An earlier version cached a `const std::vector<float>*` into the owning
    /// model. Returning the model by value left every layer pointing at a
    /// moved-from vector, and the resulting garbage frequencies produced
    /// denormals that crawled through 28 layers of BLAS -- a hang, not a crash,
    /// which is a miserable thing to chase. Passing the reference down removes
    /// the lifetime question entirely.
    [[nodiscard]] Mat forward(const Mat& x, const std::vector<float>& inv_freq) const;
};

class OmniMlp {
   public:
    Linear2 gate_proj;
    Linear2 up_proj;
    Linear2 down_proj;

    explicit OmniMlp(const Config6& cfg);

    [[nodiscard]] Mat forward(const Mat& x) const;
};

class OmniBlock {
   public:
    RmsNorm2 input_layernorm;
    OmniAttention self_attn;
    RmsNorm2 post_attention_layernorm;
    OmniMlp mlp;

    explicit OmniBlock(const Config6& cfg);

    [[nodiscard]] Mat forward(const Mat& x, const std::vector<float>& inv_freq) const;
};

// =============================================================================
// Model
// =============================================================================

/// One position of the model's input.
///
/// A position is either a text token or a stack of `num_audio_codebook` audio
/// codes; the audio codes may be the mask id. Keeping both in one struct means
/// the sequence is a single flat vector and the audio mask is derived rather
/// than tracked separately.
struct OmniToken {
    /// Text token id, used when `audio` is empty.
    std::size_t text_id = 0;
    /// One code per codebook, or empty for a text position.
    std::vector<std::uint32_t> audio;

    [[nodiscard]] bool is_audio() const { return !audio.empty(); }

    [[nodiscard]] static OmniToken text(std::size_t id) { return OmniToken{id, {}}; }
    [[nodiscard]] static OmniToken masked(std::size_t codebooks, std::size_t mask_id) {
        return OmniToken{0, std::vector<std::uint32_t>(codebooks,
                                                       static_cast<std::uint32_t>(mask_id))};
    }
};

// =============================================================================
// Backends
// =============================================================================

/// Anything that can run one full-sequence forward pass.
///
/// `OmniLm` is one; `MetalOmniContext` is the other, and the diffusion loop
/// does not need to know which it has. The interface is deliberately the whole
/// pass rather than a layer at a time: a masked diffusion model caches nothing
/// between steps, so there is no state for a backend to hold across calls and
/// nothing finer worth abstracting.
class OmniForward {
   public:
    virtual ~OmniForward() = default;

    /// Audio logits for every position, `[T, codebooks * vocab]`.
    [[nodiscard]] virtual Result<Mat> forward(const std::vector<OmniToken>& tokens) const = 0;
};

class OmniLm final : public OmniForward {
   public:
    Config6 config;
    /// [text_vocab_size, hidden_size]
    std::optional<MatBf16> text_embed;
    /// [audio_table_size, hidden_size]
    std::optional<MatBf16> audio_embed;
    std::vector<OmniBlock> layers;
    RmsNorm2 norm;
    /// [audio_table_size, hidden_size]; logits reshape to [T, codebooks, vocab].
    Linear2 audio_head;
    std::vector<float> inv_freq_cache;

    explicit OmniLm(Config6 cfg);

    [[nodiscard]] static Result<OmniLm> load(const std::string& path, Config6 cfg);

    /// Recompute the cached inverse frequencies from the config.
    void refresh_inv_freq();

    /// Embed a sequence: text rows straight from the table, audio rows summed
    /// across all eight codebooks.
    [[nodiscard]] Result<Mat> embed(const std::vector<OmniToken>& tokens) const;

    /// Run the whole sequence and return audio logits, `[T, codebooks * vocab]`.
    ///
    /// Column `i * audio_vocab_size + c` is the logit for code `c` of codebook
    /// `i` at that position.
    [[nodiscard]] Result<Mat> forward(const std::vector<OmniToken>& tokens) const override;

    [[nodiscard]] std::size_t weight_bytes() const;

    /// Requantize every BF16 projection to Q4_K.
    std::size_t quantize_projections_to_q4k();
};

// =============================================================================
// Helpers (exposed for testing)
// =============================================================================

/// Full bidirectional grouped-query attention.
///
/// `q` is [T, n_q_heads * d], `k` and `v` are [T, n_kv_heads * d]. No mask of
/// any kind: every query attends to every key. That is the whole point -- a
/// masked diffusion model conditions each position on both sides.
[[nodiscard]] Mat bidirectional_gqa_attention(const Mat& q, const Mat& k, const Mat& v,
                                              std::size_t n_q_heads, std::size_t n_kv_heads,
                                              std::size_t head_dim, float scale);

/// Apply an RMSNorm independently to each head's slice of a
/// [T, n_heads * head_dim] matrix.
[[nodiscard]] Mat apply_head_norm(const Mat& x, const RmsNorm2& norm, std::size_t n_heads,
                                  std::size_t head_dim);

}  // namespace rt
