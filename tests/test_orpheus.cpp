#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <algorithm>
#include <cmath>

#include "rt/orpheus.hpp"
#include "rt/transformer5.hpp"
#include "rt/wav.hpp"

using namespace rt;

namespace {

/// An Orpheus checkpoint found on disk, with text it can actually speak.
///
/// The published fine-tunes share an architecture and a vocabulary but not a
/// language, and which one is present depends on what has been downloaded. So
/// the integration tests discover the checkpoint rather than naming it, and
/// take their prompt from whatever `general.languages` it declares -- an
/// English sentence in an Italian voice would test very little.
struct FoundModel {
    std::string path;
    std::string language;
    std::string prompt;
    std::string voice;
};

[[nodiscard]] std::optional<FoundModel> find_orpheus_model() {
    if (!std::filesystem::is_directory("models")) {
        return std::nullopt;
    }
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator("models")) {
        if (entry.path().extension() != ".gguf") {
            continue;
        }
        const Result<GgufFile> gguf = GgufFile::open(entry.path().string());
        if (!gguf) {
            continue;
        }
        const auto arch = gguf->metadata.find("general.architecture");
        if (arch == gguf->metadata.end() || arch->second.as_str() != "llama") {
            continue;
        }
        // The audio-token arithmetic depends on this exact vocabulary.
        const auto vocab = gguf->metadata.find("llama.vocab_size");
        if (vocab == gguf->metadata.end() || vocab->second.as_u64() != 156940u) {
            continue;
        }

        FoundModel found;
        found.path = entry.path().string();
        found.language = "en";
        if (const auto langs = gguf->metadata.find("general.languages");
            langs != gguf->metadata.end()) {
            if (const std::vector<GgufMetaValue>* array = langs->second.as_array()) {
                for (const GgufMetaValue& v : *array) {
                    if (const std::optional<std::string_view> code = v.as_str()) {
                        // Prefer a language this build has voices for.
                        if (!orpheus_voices_for(*code).empty()) {
                            found.language = std::string(*code);
                            break;
                        }
                    }
                }
            }
        }

        if (found.language == "it") {
            found.prompt = "Ciao, oggi e una bella giornata.";
        } else if (found.language == "es") {
            found.prompt = "Hola, hoy hace un dia estupendo.";
        } else {
            found.prompt = "Hello, my name is Tara.";
        }
        const std::vector<std::string> voices = orpheus_voices_for(found.language);
        found.voice = voices.empty() ? "tara" : voices.front();
        return found;
    }
    return std::nullopt;
}

/// The id of the code-`code` token for slot `slot`.
[[nodiscard]] std::size_t audio_id(const OrpheusConfig& cfg, std::size_t slot, std::size_t code) {
    return cfg.audio_token_base + slot * cfg.codebook_size + code;
}

/// Feed one complete group whose codes are 0..6 in slot order.
void push_group(OrpheusCodeStream& stream, const OrpheusConfig& cfg, std::size_t first = 0) {
    for (std::size_t slot = 0; slot < cfg.codes_per_frame; ++slot) {
        REQUIRE(stream.push(audio_id(cfg, slot, first + slot)));
    }
}

}  // namespace

// =============================================================================
// Token protocol
// =============================================================================

TEST_CASE("audio token range matches the custom_token layout", "[orpheus]") {
    const OrpheusConfig cfg = OrpheusConfig::defaults();
    // <custom_token_0> is id 128256, and the reference offsets codes by 10,
    // so slot 0 code 0 is <custom_token_10> at 128266.
    REQUIRE(cfg.audio_token_base == 128266);
    // Seven codebooks of 4096: the last audio id is 156937.
    REQUIRE(cfg.audio_token_limit() == 156938);
    REQUIRE(cfg.audio_token_limit() - cfg.audio_token_base == 7 * 4096);

    REQUIRE(!cfg.is_audio_token(128265));
    REQUIRE(cfg.is_audio_token(128266));
    REQUIRE(cfg.is_audio_token(156937));
    REQUIRE(!cfg.is_audio_token(156938));
    // The markers are all below the audio range.
    REQUIRE(!cfg.is_audio_token(cfg.prompt_start));
    for (const std::size_t id : cfg.stop_tokens) {
        REQUIRE(!cfg.is_audio_token(id));
    }
}

TEST_CASE("the audio range fits inside the Orpheus vocabulary", "[orpheus]") {
    const OrpheusConfig cfg = OrpheusConfig::defaults();
    REQUIRE(cfg.audio_token_limit() <= Config5::orpheus_3b().vocab_size);
}

TEST_CASE("voices carry the language of their checkpoint", "[orpheus]") {
    REQUIRE(orpheus_voice_known("tara"));
    REQUIRE(orpheus_voice_known("giulia"));
    REQUIRE(!orpheus_voice_known("nobody"));

    REQUIRE(orpheus_voice_language("tara") == "en");
    REQUIRE(orpheus_voice_language("giulia") == "it");
    REQUIRE(orpheus_voice_language("javi") == "es");
    REQUIRE(orpheus_voice_language("nobody").empty());

    // The English fine-tune has eight speakers; the Spanish/Italian research
    // release has three each.
    REQUIRE(orpheus_voices_for("en").size() == 8);
    REQUIRE(orpheus_voices_for("it") == std::vector<std::string>{"pietro", "giulia", "carlo"});
    REQUIRE(orpheus_voices_for("es") == std::vector<std::string>{"javi", "sergio", "maria"});
    REQUIRE(orpheus_voices_for("xx").empty());
}

TEST_CASE("default sampling matches the reference engine", "[orpheus]") {
    const SamplingParams p = orpheus_default_sampling(7);
    REQUIRE(p.temperature == 0.6f);
    REQUIRE(p.top_p == 0.8f);
    REQUIRE(p.repetition_penalty == 1.3f);
    REQUIRE(p.seed == 7);
}

// =============================================================================
// Code stream
// =============================================================================

TEST_CASE("OrpheusCodeStream decodes each slot at its own offset", "[orpheus]") {
    const OrpheusConfig cfg = OrpheusConfig::defaults();
    OrpheusCodeStream stream(cfg);

    // Slot i's code c sits at base + i*4096 + c.
    for (std::size_t slot = 0; slot < 7; ++slot) {
        REQUIRE(stream.push(audio_id(cfg, slot, 100 + slot)));
    }
    REQUIRE(stream.accepted() == 7);
    REQUIRE(stream.rejected() == 0);
    REQUIRE(stream.complete_groups() == 1);

    const std::vector<std::vector<std::uint32_t>> codes = stream.snac_codes();
    REQUIRE(codes.size() == 3);
    // slot 0 -> level 0
    REQUIRE(codes[0] == std::vector<std::uint32_t>{100});
    // slots 1 and 4 -> level 1
    REQUIRE(codes[1] == std::vector<std::uint32_t>{101, 104});
    // slots 2, 3, 5, 6 -> level 2
    REQUIRE(codes[2] == std::vector<std::uint32_t>{102, 103, 105, 106});
}

TEST_CASE("OrpheusCodeStream keeps code 0", "[orpheus]") {
    // The reference accepts a code only when strictly positive, discarding
    // every legitimate code 0. Since a rejected token does not advance the
    // slot counter, dropping one shifts every code after it by a slot and
    // corrupts the rest of the utterance.
    const OrpheusConfig cfg = OrpheusConfig::defaults();
    OrpheusCodeStream stream(cfg);
    for (std::size_t slot = 0; slot < 7; ++slot) {
        REQUIRE(stream.push(audio_id(cfg, slot, 0)));
    }
    REQUIRE(stream.accepted() == 7);
    REQUIRE(stream.rejected() == 0);
    const std::vector<std::vector<std::uint32_t>> codes = stream.snac_codes();
    REQUIRE(codes[0] == std::vector<std::uint32_t>{0});
    REQUIRE(codes[1] == std::vector<std::uint32_t>{0, 0});
    REQUIRE(codes[2] == std::vector<std::uint32_t>{0, 0, 0, 0});
}

TEST_CASE("OrpheusCodeStream ignores non-audio tokens", "[orpheus]") {
    const OrpheusConfig cfg = OrpheusConfig::defaults();
    OrpheusCodeStream stream(cfg);
    REQUIRE(!stream.push(0));
    REQUIRE(!stream.push(1000));
    REQUIRE(!stream.push(128265));  // one below the audio range
    REQUIRE(!stream.push(200000));  // above the vocabulary
    REQUIRE(stream.accepted() == 0);
    REQUIRE(stream.rejected() == 4);
    REQUIRE(stream.snac_codes().empty());
}

TEST_CASE("OrpheusCodeStream rejects without advancing the slot", "[orpheus]") {
    // This is the resynchronisation property. A token for the wrong slot
    // computes a code outside [0, 4096), and refusing it while leaving the
    // counter alone means the next correct token still lands in the right slot.
    const OrpheusConfig cfg = OrpheusConfig::defaults();
    OrpheusCodeStream stream(cfg);

    // Expecting slot 0, but hand it a slot-3 token: code would be 3*4096+5.
    REQUIRE(!stream.push(audio_id(cfg, 3, 5)));
    REQUIRE(stream.accepted() == 0);
    REQUIRE(stream.rejected() == 1);

    // The stream is still waiting for slot 0, so a genuine slot-0 token lands.
    REQUIRE(stream.push(audio_id(cfg, 0, 42)));
    REQUIRE(stream.accepted() == 1);

    // Now expecting slot 1; a slot-0 token computes a negative code.
    REQUIRE(!stream.push(audio_id(cfg, 0, 1)));
    REQUIRE(stream.rejected() == 2);
    REQUIRE(stream.push(audio_id(cfg, 1, 7)));
    REQUIRE(stream.accepted() == 2);
}

TEST_CASE("OrpheusCodeStream truncates to complete groups", "[orpheus]") {
    const OrpheusConfig cfg = OrpheusConfig::defaults();
    OrpheusCodeStream stream(cfg);
    push_group(stream, cfg);
    // Three extra codes: not a group, so they must not reach the codes.
    for (std::size_t slot = 0; slot < 3; ++slot) {
        REQUIRE(stream.push(audio_id(cfg, slot, 200 + slot)));
    }
    REQUIRE(stream.accepted() == 10);
    REQUIRE(stream.complete_groups() == 1);

    const std::vector<std::vector<std::uint32_t>> codes = stream.snac_codes();
    REQUIRE(codes[0].size() == 1);
    REQUIRE(codes[1].size() == 2);
    REQUIRE(codes[2].size() == 4);
}

TEST_CASE("OrpheusCodeStream produces SNAC's 1:2:4 code ratio", "[orpheus]") {
    const OrpheusConfig cfg = OrpheusConfig::defaults();
    OrpheusCodeStream stream(cfg);
    constexpr std::size_t groups = 5;
    for (std::size_t g = 0; g < groups; ++g) {
        push_group(stream, cfg, g * 7);
    }
    REQUIRE(stream.complete_groups() == groups);
    // Each group covers 4 frames at the finest rate.
    REQUIRE(stream.frames() == groups * 4);

    const std::vector<std::vector<std::uint32_t>> codes = stream.snac_codes();
    // The strides the SNAC quantizer will check: 4, 2, 1 against frames.
    REQUIRE(codes[0].size() * 4 == stream.frames());
    REQUIRE(codes[1].size() * 2 == stream.frames());
    REQUIRE(codes[2].size() * 1 == stream.frames());
}

TEST_CASE("OrpheusCodeStream output is accepted by the SNAC quantizer", "[orpheus]") {
    // The two halves of the pipeline agree on frame counts. This is the seam
    // where a stride or interleave mistake would show up, and it needs no
    // weights to check.
    const OrpheusConfig cfg = OrpheusConfig::defaults();
    OrpheusCodeStream stream(cfg);
    for (std::size_t g = 0; g < 3; ++g) {
        push_group(stream, cfg, g);
    }
    const std::vector<std::vector<std::uint32_t>> codes = stream.snac_codes();

    SnacQuantizer q;
    q.latent_dim = 4;
    for (const std::size_t stride : SnacConfig::snac_24khz().vq_strides) {
        SnacQuantizer::Level level;
        level.codebook = Mat::zeros(cfg.codebook_size, 2);
        level.out_proj_weight = Mat::zeros(4, 2);
        level.stride = stride;
        q.levels.push_back(std::move(level));
    }

    const Result<Mat> z = q.from_codes(codes);
    REQUIRE(z.has_value());
    REQUIRE(z->rows == stream.frames());
    REQUIRE(z->rows == 12);  // 3 groups x 4 frames
}

// =============================================================================
// Prompt framing
// =============================================================================

TEST_CASE("orpheus_build_prompt frames the encoded text", "[orpheus][.integration]") {
    const std::optional<FoundModel> model = find_orpheus_model();
    if (!model) {
        SKIP("no Orpheus GGUF in models/");
    }
    const Result<GgufFile> gguf = GgufFile::open(model->path);
    REQUIRE(gguf.has_value());
    const Result<HfBpeTokenizer> tok = load_gguf_tokenizer(*gguf);
    if (!tok.has_value()) {
        FAIL(tok.error());
    }

    const OrpheusConfig cfg = OrpheusConfig::defaults();
    const Result<std::vector<std::size_t>> ids =
        orpheus_build_prompt(*tok, cfg, "tara", "Hello there.");
    REQUIRE(ids.has_value());

    // [BOS, start marker, BOS] + text + end markers. Both BOS tokens are
    // deliberate; see the orpheus.hpp header.
    REQUIRE(cfg.leading_bos);
    REQUIRE((*ids)[0] == cfg.bos_token);
    REQUIRE((*ids)[1] == cfg.prompt_start);
    REQUIRE((*ids)[2] == cfg.bos_token);
    REQUIRE(ids->size() > 3 + cfg.prompt_end.size());
    for (std::size_t i = 0; i < cfg.prompt_end.size(); ++i) {
        REQUIRE((*ids)[ids->size() - cfg.prompt_end.size() + i] == cfg.prompt_end[i]);
    }
    // The body carries no audio tokens.
    for (const std::size_t id : *ids) {
        REQUIRE(!cfg.is_audio_token(id));
    }
}

TEST_CASE("orpheus_build_prompt can drop the leading BOS", "[orpheus][.integration]") {
    const std::optional<FoundModel> model = find_orpheus_model();
    if (!model) {
        SKIP("no Orpheus GGUF in models/");
    }
    const Result<GgufFile> gguf = GgufFile::open(model->path);
    REQUIRE(gguf.has_value());
    const Result<HfBpeTokenizer> tok = load_gguf_tokenizer(*gguf);
    REQUIRE(tok.has_value());

    OrpheusConfig cfg = OrpheusConfig::defaults();
    cfg.leading_bos = false;
    const Result<std::vector<std::size_t>> ids =
        orpheus_build_prompt(*tok, cfg, "tara", "Hello there.");
    REQUIRE(ids.has_value());
    REQUIRE((*ids)[0] == cfg.prompt_start);
    REQUIRE((*ids)[1] == cfg.bos_token);
}

TEST_CASE("128009 is a prompt separator, not a stop token", "[orpheus]") {
    // It appears inside prompt_end, and the model emits it freely. Treating it
    // as a stop ends generation within a few tokens.
    const OrpheusConfig cfg = OrpheusConfig::defaults();
    REQUIRE(std::find(cfg.prompt_end.begin(), cfg.prompt_end.end(), 128009) !=
            cfg.prompt_end.end());
    REQUIRE(std::find(cfg.stop_tokens.begin(), cfg.stop_tokens.end(), 128009) ==
            cfg.stop_tokens.end());
    REQUIRE(cfg.stop_tokens == std::vector<std::size_t>{128258});
}

TEST_CASE("orpheus_build_prompt rejects empty text", "[orpheus]") {
    // Needs no tokenizer: the check comes first.
    const OrpheusConfig cfg = OrpheusConfig::defaults();
    const Result<HfBpeTokenizer> tok =
        HfBpeTokenizer::from_vocab_and_merges({"a", "b"}, {}, true,
                                              HfBpeTokenizer::PreTokenizer::Llama3);
    REQUIRE(tok.has_value());
    const Result<std::vector<std::size_t>> ids = orpheus_build_prompt(*tok, cfg, "tara", "");
    REQUIRE(!ids.has_value());
}

// =============================================================================
// The embedded tokenizer
// =============================================================================

TEST_CASE("load_gguf_tokenizer reads the embedded vocabulary", "[orpheus][.integration]") {
    const std::optional<FoundModel> model = find_orpheus_model();
    if (!model) {
        SKIP("no Orpheus GGUF in models/");
    }
    const Result<GgufFile> gguf = GgufFile::open(model->path);
    REQUIRE(gguf.has_value());
    const Result<HfBpeTokenizer> tok = load_gguf_tokenizer(*gguf);
    if (!tok.has_value()) {
        FAIL(tok.error());
    }

    REQUIRE(tok->vocab_size() == 156940);
    REQUIRE(tok->byte_level());
    // tokenizer.ggml.pre is "llama-bpe", which groups digits.
    REQUIRE(tok->pre_tokenizer() == HfBpeTokenizer::PreTokenizer::Llama3);

    // The audio tokens are where the code arithmetic assumes.
    REQUIRE(tok->token_text(128256) == "<custom_token_0>");
    REQUIRE(tok->token_text(128266) == "<custom_token_10>");
    REQUIRE(tok->token_text(OrpheusConfig::defaults().audio_token_base) == "<custom_token_10>");

    // Round-trip ordinary text.
    const std::string text = "tara: Hello there, friend.";
    const std::vector<std::uint32_t> ids = tok->encode(text);
    REQUIRE(!ids.empty());
    REQUIRE(tok->decode(ids) == text);
}

TEST_CASE("the GGUF tokenizer groups digits the Llama 3 way", "[orpheus][.integration]") {
    const std::optional<FoundModel> model = find_orpheus_model();
    if (!model) {
        SKIP("no Orpheus GGUF in models/");
    }
    const Result<GgufFile> gguf = GgufFile::open(model->path);
    REQUIRE(gguf.has_value());
    const Result<HfBpeTokenizer> tok = load_gguf_tokenizer(*gguf);
    REQUIRE(tok.has_value());

    // GPT-2 splits every digit; Llama 3's `\p{N}{1,3}` takes runs of three, so
    // "2024" is "202" + "4". Getting this wrong mispronounces every number.
    const std::vector<std::string> llama_split =
        HfBpeTokenizer::pretokenize("2024", HfBpeTokenizer::PreTokenizer::Llama3);
    REQUIRE(llama_split == std::vector<std::string>{"202", "4"});

    const std::vector<std::string> gpt2_split =
        HfBpeTokenizer::pretokenize("2024", HfBpeTokenizer::PreTokenizer::Gpt2);
    REQUIRE(gpt2_split == std::vector<std::string>{"2", "0", "2", "4"});

    // And the instance actually uses the Llama 3 clause, so the two encodings
    // differ.
    REQUIRE(tok->encode("2024").size() < 4);
    // Round-tripping still recovers the text either way.
    REQUIRE(tok->decode(tok->encode("in 2024 and 7")) == "in 2024 and 7");
}

// =============================================================================
// RoPE scaling
// =============================================================================

TEST_CASE("rope_freqs divisors stretch the low-frequency bands", "[orpheus][.integration]") {
    const std::optional<FoundModel> found = find_orpheus_model();
    if (!found) {
        SKIP("no Orpheus GGUF in models/");
    }
    LlamaModel model = LlamaModel::new_for_inference(Config5::orpheus_3b());
    // The unscaled frequencies, before the file is read.
    const std::vector<float> plain = model.config.inv_freq();

    const Result<GgufFile> gguf = GgufFile::open(found->path);
    REQUIRE(gguf.has_value());
    const std::optional<std::size_t> idx = gguf->find_tensor("rope_freqs.weight");
    REQUIRE(idx.has_value());
    const Result<std::vector<float>> divisors = gguf->decode_f32(*idx);
    REQUIRE(divisors.has_value());
    REQUIRE(divisors->size() == 64);

    // Divisors, not multipliers: they run from 1.0 up to 32.0.
    REQUIRE(divisors->front() == 1.0f);
    REQUIRE(divisors->back() == 32.0f);

    model.config.rope_freq_divisors = *divisors;
    const std::vector<float> scaled = model.config.inv_freq();
    REQUIRE(scaled.size() == plain.size());

    // A divisor of 1 leaves the band alone; 32 divides it. Applying these as
    // multipliers would raise the last band by 32x instead -- a factor of 1024
    // the wrong way, on exactly the dimensions carrying long-range position.
    REQUIRE(scaled.front() == plain.front());
    REQUIRE(scaled.back() < plain.back());
    REQUIRE(std::fabs(scaled.back() - plain.back() / 32.0f) < 1e-12f);
}

// =============================================================================
// End to end
// =============================================================================

TEST_CASE("Orpheus synthesises speech-shaped audio", "[orpheus][.e2e]") {
    // Tagged separately from [.integration]: this one loads 2.4 GB of weights
    // and generates, so it is a minute of work rather than a second.
    const std::optional<FoundModel> found = find_orpheus_model();
    if (!found || !std::filesystem::exists("models/snac_24khz.bin")) {
        SKIP("Orpheus or SNAC weights not present");
    }
    INFO("using " << found->path << " (" << found->language << ", voice " << found->voice << ")");

    const Result<GgufFile> gguf = GgufFile::open(found->path);
    REQUIRE(gguf.has_value());
    const Result<HfBpeTokenizer> tok = load_gguf_tokenizer(*gguf);
    REQUIRE(tok.has_value());

    LlamaModel model = LlamaModel::new_for_inference(Config5::orpheus_3b());
    const Result<void> loaded = model.load_weights_from_gguf(found->path);
    if (!loaded.has_value()) {
        FAIL(loaded.error());
    }
    model.quantize_lm_head();

    const Result<SnacDecoder> snac =
        SnacDecoder::load("models/snac_24khz.bin", SnacConfig::snac_24khz());
    REQUIRE(snac.has_value());

    OrpheusRequest request;
    request.text = found->prompt;
    request.voice = found->voice;
    request.max_new = 210;  // 30 groups, about 2.5 s
    request.sampling = orpheus_default_sampling(1234);

    const Result<OrpheusResult> result =
        orpheus_synthesize(model, *snac, *tok, request, OrpheusConfig::defaults());
    if (!result.has_value()) {
        FAIL(result.error());
    }

    // The backbone should be emitting audio tokens almost exclusively.
    INFO("tokens " << result->tokens_generated << " kept " << result->codes_accepted
                   << " rejected " << result->codes_rejected);
    REQUIRE(result->codes_accepted > 0);
    REQUIRE(result->groups > 0);
    REQUIRE(result->codes_rejected * 10 < result->codes_accepted);

    // Length is fixed by the frame accounting, with no slack.
    REQUIRE(result->samples.size() == result->groups * 4 * 512);

    const WaveStats stats = wave_stats(result->samples);
    INFO(stats.describe());
    REQUIRE(stats.in_range());
    REQUIRE(stats.looks_like_speech());
}

// =============================================================================
// Decode-path self-consistency
// =============================================================================

TEST_CASE("prefill and incremental decode agree", "[orpheus][.e2e]") {
    // No oracle needed: running N tokens through prefill must give the same
    // final-position logits as running N-1 through prefill and the last one
    // through the incremental decode path. Any disagreement is a KV cache or
    // RoPE-offset bug, which otherwise shows up only as audio that starts
    // plausible and degrades.
    const std::optional<FoundModel> found = find_orpheus_model();
    if (!found) {
        SKIP("no Orpheus GGUF in models/");
    }
    LlamaModel model = LlamaModel::new_for_inference(Config5::orpheus_3b());
    const Result<void> loaded = model.load_weights_from_gguf(found->path);
    if (!loaded.has_value()) {
        FAIL(loaded.error());
    }

    const std::vector<std::size_t> ids{128000, 128259, 128000, 83, 5169, 25, 22691, 11, 13};

    LlamaKvCache full(model.config, 64);
    const Mat all_at_once = model.forward_cached(ids, full);

    LlamaKvCache split(model.config, 64);
    const std::vector<std::size_t> head(ids.begin(), ids.end() - 1);
    model.forward_cached(head, split);
    const Mat incremental = model.forward_cached({ids.back()}, split);

    REQUIRE(all_at_once.cols == incremental.cols);

    // The two paths are not bit-identical and are not meant to be:
    // `gqa_attention_cached` runs per-head sgemm for prefill and sdot/saxpy for
    // a single decode query, so the reductions sum in different orders. Over 28
    // layers of 3072-wide f32 reductions that drifts by a few hundredths of a
    // logit. What has to hold is the ranking -- a wrong RoPE offset or a
    // misaligned cache moves logits by whole units and reorders the top of the
    // distribution.
    const auto top5 = [](const Mat& logits) {
        std::vector<std::size_t> idx(logits.cols);
        for (std::size_t c = 0; c < logits.cols; ++c) {
            idx[c] = c;
        }
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&logits](std::size_t a, std::size_t b) {
                              return logits.at(0, a) > logits.at(0, b);
                          });
        idx.resize(5);
        return idx;
    };

    float max_delta = 0.0f;
    for (std::size_t c = 0; c < all_at_once.cols; ++c) {
        max_delta = std::max(max_delta, std::fabs(all_at_once.at(0, c) - incremental.at(0, c)));
    }

    const std::vector<std::size_t> a = top5(all_at_once);
    const std::vector<std::size_t> b = top5(incremental);
    INFO("prefill top1 " << a.front() << " logit " << all_at_once.at(0, a.front())
                         << ", decode top1 " << b.front() << " logit "
                         << incremental.at(0, b.front()) << ", max |delta| " << max_delta);
    REQUIRE(a == b);
    // Loose enough for accumulation order, far tighter than any real bug.
    REQUIRE(max_delta < 0.5f);
}
