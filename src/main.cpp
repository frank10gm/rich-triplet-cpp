// =============================================================================
// rich-triplet -- CLI
// =============================================================================
//
// One binary over every model in the project:
//
//   --benchmark            scalar autograd vs tensor autodiff, side by side
//   --prompt TEXT          train the small GPT on the built-in corpus, generate
//   --prompt + --weights   load real weights (GPT-OSS, Gemma 3 or Qwen 3.5)
//   --pretokenize SRC DST  tokenize a text file into a binary corpus

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "rt/dataset.hpp"
#include "rt/init_rng.hpp"
#include "rt/tensor_node.hpp"
#include "rt/tokenizer.hpp"
#include "rt/train.hpp"
#include "rt/train2.hpp"
#include "rt/transformer.hpp"
#include "rt/transformer2.hpp"
#include "rt/transformer3.hpp"
#include "rt/transformer4.hpp"
#include "rt/omnivoice.hpp"
#if RT_FEATURE_METAL
#include "rt/metal_omnivoice.hpp"
#endif
#include "rt/orpheus.hpp"
#include "rt/snac.hpp"
#include "rt/transformer5.hpp"
#include "rt/transformer_qwen35.hpp"
#include "rt/wav.hpp"

using namespace rt;

namespace {

// =============================================================================
// Bilingual training corpus -- Italian and English
// =============================================================================

constexpr const char* kCorpus = R"RT_CORPUS(
Il cielo sopra Milano era grigio come sempre. Giovanni guardava dalla finestra
del suo appartamento al quinto piano, pensando alla riunione del mattino.
La sua collega Chiara gli aveva detto che il progetto era in ritardo di due
settimane. Bisognava trovare una soluzione prima di venerdì.

The sky above London was the same color as a television tuned to a dead channel.
Thomas looked out from his office on the fifth floor, thinking about the morning
meeting. His colleague Sarah had told him the project was two weeks behind
schedule. They needed to find a solution before Friday.

La lingua è il vestito del pensiero. Ogni parola porta con sé il peso della
storia, la memoria di chi l'ha usata prima di noi. Imparare una lingua straniera
significa aprire una finestra su un altro mondo, un altro modo di vedere le cose.

Language is the dress of thought. Every word carries with it the weight of
history, the memory of those who used it before us. Learning a foreign language
means opening a window onto another world, another way of seeing things.

Il modello linguistico non capisce davvero le parole. Calcola le probabilità
delle sequenze di caratteri basandosi sui pattern nel testo di addestramento.
Eppure, da questi semplici calcoli, emerge qualcosa che assomiglia alla comprensione.

The language model does not truly understand words. It calculates probabilities
of character sequences based on patterns in the training text. Yet from these
simple calculations something emerges that resembles understanding.

Buongiorno, come stai? Sto bene, grazie. E tu? Anch'io sto bene.
Hello, how are you? I am well, thank you. And you? I am well too.

Milano, Roma, Firenze, Venezia, Napoli, Torino, Bologna, Palermo.
London, Paris, Berlin, Madrid, Rome, Amsterdam, Vienna, Prague.

uno due tre quattro cinque sei sette otto nove dieci
one two three four five six seven eight nine ten

il lo la i gli le un una dello della degli delle
the a an of in on at to for with from by

essere avere fare dire andare venire sapere potere volere
to be to have to do to say to go to come to know to can to want

bello brutto grande piccolo vecchio nuovo buono cattivo
beautiful ugly big small old new good bad

oggi ieri domani adesso sempre mai spesso raramente
today yesterday tomorrow now always never often rarely
)RT_CORPUS";

// =============================================================================
// CLI argument parsing
// =============================================================================

struct CliArgs {
    /// --prompt TEXT     : text to complete (triggers generation mode)
    std::optional<std::string> prompt;
    /// --weights DIR     : directory with .safetensors shards, or a .gguf file
    std::optional<std::string> weights;
    /// --vocab PATH      : BPE vocab.json (required with --weights for GPT-OSS)
    std::optional<std::string> vocab;
    /// --merges PATH     : BPE merges.txt (required with --weights for GPT-OSS)
    std::optional<std::string> merges;
    /// --tokenizer-model PATH : SentencePiece .model (Gemma 3 with --weights)
    std::optional<std::string> tokenizer_model;
    /// --tokenizer-dir DIR : directory holding tokenizer.json, for GGUF files
    /// where the tokenizer is not bundled
    std::optional<std::string> tokenizer_dir;
    /// --model NAME      : architecture (gpt-oss | gemma3-1b | gemma3-4b | ...)
    std::optional<std::string> model;
    /// --max-new N       : tokens to generate
    std::size_t max_new = 200;
    /// --temp T          : sampling temperature
    float temperature = 0.8f;
    /// --top-k K         : top-k cutoff, 0 disables
    std::size_t top_k = 40;
    /// --top-p P         : nucleus probability
    float top_p = 0.95f;
    /// --rep-penalty R   : repetition penalty, 1.0 disables
    float rep_penalty = 1.1f;
    /// --seed S          : RNG seed
    std::uint64_t seed = 42;
    /// --train-steps N   : steps for on-the-fly training
    std::size_t train_steps = 200;
    /// --checkpoint PATH : load a saved .ckpt instead of training
    std::optional<std::string> checkpoint;
    /// --pretokenize SRC DST : tokenize SRC into DST, then exit
    std::optional<std::pair<std::string, std::string>> pretokenize;
    /// --benchmark : run the scalar-vs-tensor autograd benchmark
    bool benchmark = false;
    /// --quantize : quantize weights to Q4 after loading
    bool quantize = false;
    /// --debug : per-step diagnostics (h_rms, logit gaps, top-5 tokens)
    bool debug = false;
    /// --draft-len N : max speculative draft tokens per step, 0 disables
    std::size_t draft_len = 0;
    /// --voice NAME : Orpheus speaker
    std::optional<std::string> voice;
    /// --snac PATH : SNAC codec checkpoint (pytorch_model.bin)
    std::optional<std::string> snac;
    /// --out PATH : where to write the synthesised WAV
    std::optional<std::string> out;
    /// --no-audio-mask : let Orpheus sample outside the audio token range
    bool no_audio_mask = false;
    /// --no-leading-bos : drop the BOS that vLLM's re-tokenization prepends
    bool no_leading_bos = false;
    /// --language NAME : OmniVoice language hint
    std::optional<std::string> language;
    /// --instruct TEXT : OmniVoice free-text voice description
    std::optional<std::string> instruct;
    /// --ref-audio PATH : WAV of the voice to clone
    std::optional<std::string> ref_audio;
    /// --ref-text TEXT : what that WAV says
    std::optional<std::string> ref_text;
    /// --duration S : audio seconds to generate; 0 uses the length heuristic
    float duration = 0.0f;
    /// --steps N : OmniVoice unmasking steps
    std::size_t steps = 12;
    /// --guidance G : classifier-free guidance scale; 0 disables it
    float guidance = 2.0f;
    /// --rope-interleaved : pair 2i with 2i+1 instead of i with i+head_dim/2
    bool rope_interleaved = false;
};

void print_help();

/// Parse `text`, falling back to `fallback` on anything unparseable.
template <typename T>
[[nodiscard]] T parse_or(const std::string& text, T fallback) {
    try {
        if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>(std::stof(text));
        } else {
            return static_cast<T>(std::stoull(text));
        }
    } catch (const std::exception&) {
        return fallback;
    }
}

[[nodiscard]] CliArgs parse_args(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    CliArgs a;

    // Every value-taking flag consumes the next argument; a flag at the end of
    // the line with nothing after it is simply ignored.
    const auto take = [&args](std::size_t& i) -> std::optional<std::string> {
        ++i;
        if (i < args.size()) {
            return args[i];
        }
        return std::nullopt;
    };

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--prompt") {
            a.prompt = take(i);
        } else if (arg == "--weights") {
            a.weights = take(i);
        } else if (arg == "--vocab") {
            a.vocab = take(i);
        } else if (arg == "--merges") {
            a.merges = take(i);
        } else if (arg == "--tokenizer-model") {
            a.tokenizer_model = take(i);
        } else if (arg == "--tokenizer-dir") {
            a.tokenizer_dir = take(i);
        } else if (arg == "--model") {
            a.model = take(i);
        } else if (arg == "--max-new") {
            if (const auto v = take(i)) a.max_new = parse_or<std::size_t>(*v, 200);
        } else if (arg == "--temp") {
            if (const auto v = take(i)) a.temperature = parse_or<float>(*v, 0.8f);
        } else if (arg == "--top-k") {
            if (const auto v = take(i)) a.top_k = parse_or<std::size_t>(*v, 40);
        } else if (arg == "--top-p") {
            if (const auto v = take(i)) a.top_p = parse_or<float>(*v, 0.95f);
        } else if (arg == "--rep-penalty") {
            if (const auto v = take(i)) a.rep_penalty = parse_or<float>(*v, 1.1f);
        } else if (arg == "--seed") {
            if (const auto v = take(i)) a.seed = parse_or<std::uint64_t>(*v, 42);
        } else if (arg == "--train-steps") {
            if (const auto v = take(i)) a.train_steps = parse_or<std::size_t>(*v, 200);
        } else if (arg == "--checkpoint") {
            a.checkpoint = take(i);
        } else if (arg == "--pretokenize") {
            const std::string src = take(i).value_or(std::string{});
            const std::string dst = take(i).value_or(std::string{});
            a.pretokenize = std::pair{src, dst};
        } else if (arg == "--benchmark") {
            a.benchmark = true;
        } else if (arg == "--quantize") {
            a.quantize = true;
        } else if (arg == "--debug") {
            a.debug = true;
        } else if (arg == "--voice") {
            a.voice = take(i);
        } else if (arg == "--snac") {
            a.snac = take(i);
        } else if (arg == "--out") {
            a.out = take(i);
        } else if (arg == "--no-audio-mask") {
            a.no_audio_mask = true;
        } else if (arg == "--no-leading-bos") {
            a.no_leading_bos = true;
        } else if (arg == "--language") {
            a.language = take(i);
        } else if (arg == "--instruct") {
            a.instruct = take(i);
        } else if (arg == "--ref-audio") {
            if (const auto v = take(i)) a.ref_audio = *v;
        } else if (arg == "--ref-text") {
            if (const auto v = take(i)) a.ref_text = *v;
        } else if (arg == "--duration") {
            if (const auto v = take(i)) a.duration = parse_or<float>(*v, 0.0f);
        } else if (arg == "--steps") {
            if (const auto v = take(i)) a.steps = parse_or<std::size_t>(*v, 12);
        } else if (arg == "--guidance") {
            if (const auto v = take(i)) a.guidance = parse_or<float>(*v, 2.0f);
        } else if (arg == "--rope-interleaved") {
            a.rope_interleaved = true;
        } else if (arg == "--draft-len") {
            if (const auto v = take(i)) a.draft_len = parse_or<std::size_t>(*v, 4);
        } else if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_help();
            std::exit(1);
        }
    }
    return a;
}

void print_help() {
    std::printf("rich-triplet — LLM from scratch in C++\n");
    std::printf("\n");
    std::printf("USAGE:\n");
    std::printf("  rich-triplet --benchmark              Run scalar-vs-tensor autograd benchmark\n");
    std::printf("  rich-triplet --prompt TEXT            Train on corpus, then generate\n");
    std::printf("  rich-triplet --prompt TEXT \\\n");
    std::printf("               --weights DIR \\\n");
    std::printf("               --vocab vocab.json \\\n");
    std::printf("               --merges merges.txt      Load GPT-OSS weights, generate\n");
    std::printf("  rich-triplet --prompt TEXT \\\n");
    std::printf("               --weights DIR \\\n");
    std::printf("               --tokenizer-model tokenizer.model \\\n");
    std::printf("               --model gemma3-1b           Load Gemma 3 weights, generate\n");
    std::printf("  rich-triplet --pretokenize SRC DST    Tokenize SRC text file → DST .bin\n");
    std::printf("\n");
    std::printf("OPTIONS:\n");
    std::printf("  --prompt TEXT            Prompt text to complete\n");
    std::printf("  --weights DIR            Directory with .safetensors shards\n");
    std::printf("  --vocab PATH             BPE vocab.json      (GPT-OSS)\n");
    std::printf("  --merges PATH            BPE merges.txt      (GPT-OSS)\n");
    std::printf("  --tokenizer-model PATH   SentencePiece .model (Gemma 3)\n");
    std::printf(
        "  --tokenizer-dir DIR      Dir with tokenizer.json (for GGUF, where tokenizer is "
        "separate)\n");
    std::printf(
        "  --model NAME             Architecture: gpt-oss | gemma3-1b | gemma3-4b | qwen35-0.8b |\n"
        "                           qwen35-4b | qwen35-9b | orpheus-3b | omnivoice\n");
    std::printf("  --max-new N              Tokens to generate          [default: 200]\n");
    std::printf("  --temp T                 Sampling temperature        [default: 0.8]\n");
    std::printf("  --top-k K                Top-K cutoff (0=disabled)   [default: 40]\n");
    std::printf("  --top-p P                Nucleus probability         [default: 0.95]\n");
    std::printf("  --rep-penalty R          Repetition penalty          [default: 1.1]\n");
    std::printf("  --seed S                 RNG seed                    [default: 42]\n");
    std::printf("  --train-steps N          Training steps (no-weights) [default: 200]\n");
    std::printf("  --checkpoint PATH        Load saved .ckpt instead of training\n");
    std::printf("  --benchmark              Run scalar-vs-tensor autograd benchmark\n");
    std::printf("  --debug                  Enable per-step diagnostic logging\n");
    std::printf("\nText to speech (--model orpheus-3b):\n");
    std::printf("  --voice NAME             en: tara leah jess leo dan mia zac zoe [default: tara]\n");
    std::printf("                           es: javi sergio maria   it: pietro giulia carlo\n");
    std::printf("                           (which work depends on the checkpoint loaded)\n");
    std::printf("  --snac PATH              SNAC 24 kHz checkpoint (pytorch_model.bin)\n");
    std::printf("  --out PATH               Output WAV                  [default: out.wav]\n");
    std::printf("  --no-audio-mask          Allow sampling outside the audio token range\n");
    std::printf("  --no-leading-bos         Drop the leading BOS token from the prompt\n");
    std::printf("\nOmniVoice (--model omnivoice):\n");
    std::printf("  --language NAME          Language hint, e.g. Italian    [default: None]\n");
    std::printf("  --instruct TEXT          Voice description              [default: None]\n");
    std::printf("  --ref-audio PATH         WAV of a voice to clone\n");
    std::printf("  --ref-text TEXT          What that WAV says (required with it)\n");
    std::printf("  --duration S             Audio seconds (0 = estimate)   [default: 0]\n");
    std::printf("  --steps N                Unmasking steps                [default: 12]\n");
    std::printf("  --guidance G             Guidance scale (0 = off)       [default: 2.0]\n");
    std::printf("  --rope-interleaved       Use interleaved RoPE pairing (debugging; the\n");
    std::printf("                           default half-split is the correct one)\n");
    std::printf("  --weights PATH           omnivoice-base GGUF\n");
    std::printf("  --snac PATH              omnivoice-tokenizer GGUF\n");
    std::printf("\n");
    std::printf("  --pretokenize S D        Tokenize text file S, write binary D.bin\n");
    std::printf("                           Uses char tokenizer built from S.\n");
    std::printf("                           For BPE: also pass --vocab and --merges.\n");
}

// =============================================================================
// Shared helpers
// =============================================================================

/// Write `text` to stdout immediately, so streamed tokens appear as they land.
void emit(const std::string& text) {
    std::fputs(text.c_str(), stdout);
    std::fflush(stdout);
}

[[noreturn]] void die(const std::string& message) {
    std::fprintf(stderr, "%s\n", message.c_str());
    std::exit(1);
}

/// Whether `path` names a GGUF file.
///
/// The extension is the usual signal, but Ollama stores blobs under
/// extensionless names, so a regular file is also sniffed for the magic.
[[nodiscard]] bool looks_like_gguf(const std::string& path) {
    if (path.ends_with(".gguf")) {
        return true;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    char magic[4] = {};
    return f.read(magic, 4) && std::memcmp(magic, "GGUF", 4) == 0;
}

/// Where the tokenizer and weight cache live, given `--weights`.
///
/// For a GGUF file that is its parent directory; for safetensors it is the
/// directory itself, with any trailing slash removed.
[[nodiscard]] std::string weights_dir_of(const std::string& weights_path, bool is_gguf) {
    if (is_gguf) {
        const std::filesystem::path parent = std::filesystem::path(weights_path).parent_path();
        return parent.empty() ? "." : parent.string();
    }
    std::string dir = weights_path;
    while (!dir.empty() && dir.back() == '/') {
        dir.pop_back();
    }
    return dir;
}

/// Load tokenizer.json from `--tokenizer-dir`, else from the weights directory.
[[nodiscard]] HfBpeTokenizer load_hf_tokenizer(const CliArgs& args, const std::string& weights_dir,
                                               bool is_gguf, const char* tag) {
    std::string tok_dir = args.tokenizer_dir.value_or(weights_dir);
    while (!tok_dir.empty() && tok_dir.back() == '/') {
        tok_dir.pop_back();
    }
    const std::string tok_json_path = tok_dir + "/tokenizer.json";
    std::fprintf(stderr, "[ %s ] Loading tokenizer from %s...\n", tag, tok_json_path.c_str());

    Result<HfBpeTokenizer> tok = HfBpeTokenizer::from_json_file(tok_json_path);
    if (!tok) {
        std::fprintf(stderr, "Error: failed to load tokenizer from %s: %s\n",
                     tok_json_path.c_str(), tok.error().c_str());
        if (is_gguf && !args.tokenizer_dir) {
            std::fprintf(stderr, "Hint: GGUF files do not include a tokenizer.\n");
            std::fprintf(stderr,
                         "      Pass --tokenizer-dir pointing to your safetensors directory,\n");
            std::fprintf(stderr, "      e.g.: --tokenizer-dir /path/to/gemma-3-4b-it/\n");
        }
        std::exit(1);
    }
    std::fprintf(stderr, "[ %s ] Vocab size: %zu\n", tag, tok->vocab_size());
    return std::move(*tok);
}

/// Log every prompt token with its decoded text -- the fastest way to spot a
/// chat template that tokenized into subword pieces instead of special ids.
void dump_prompt_tokens(const Tokenizer& tok, const std::vector<std::size_t>& token_ids) {
    for (std::size_t i = 0; i < token_ids.size(); ++i) {
        const std::string text = tok.decode({static_cast<std::uint32_t>(token_ids[i])});
        std::fprintf(stderr, "  [%zu] id=%zu text=\"%s\"\n", i, token_ids[i], text.c_str());
    }
}

/// Append `text`'s token ids to `out`.
void extend_encoded(std::vector<std::size_t>& out, const Tokenizer& tok, std::string_view text) {
    for (std::uint32_t id : tok.encode(text)) {
        out.push_back(static_cast<std::size_t>(id));
    }
}

// =============================================================================
// Generation mode -- GPT-OSS with loaded weights
// =============================================================================

void run_gpt_oss(const CliArgs& args, const std::string& prompt) {
    const std::string& weights_dir = *args.weights;
    if (!args.vocab) {
        die("--vocab required with --weights (path to vocab.json)");
    }
    if (!args.merges) {
        die("--merges required with --weights (path to merges.txt)");
    }

    std::fprintf(stderr, "[ GPT-OSS ] Loading tokenizer...\n");
    Result<BpeTokenizer> tok = BpeTokenizer::from_files(*args.vocab, *args.merges);
    if (!tok) {
        die("failed to load BPE tokenizer: " + tok.error());
    }

    std::fprintf(stderr, "[ GPT-OSS ] Building model (gpt-oss-20b config)...\n");
    InitRng rng(0);
    GptOssModel model(Config3::gpt_oss_20b(), rng);

    std::fprintf(stderr, "[ GPT-OSS ] Loading weights from %s...\n", weights_dir.c_str());
    if (const Result<void> loaded = model.load_weights_from_dir(weights_dir); !loaded) {
        die("failed to load weights: " + loaded.error());
    }

    std::vector<std::size_t> token_ids;
    extend_encoded(token_ids, *tok, prompt);
    if (token_ids.empty()) {
        die("Error: prompt encodes to zero tokens");
    }

    SamplingParams params = SamplingParams::creative(args.seed);
    params.temperature = args.temperature;
    params.top_k = args.top_k;
    params.top_p = args.top_p;
    params.seed = args.seed;

    // Echo the prompt first, then stream the continuation.
    emit(prompt);
    model.generate_with_params_streaming(token_ids, args.max_new, params, [&](std::size_t tok_id) {
        emit(tok->decode({static_cast<std::uint32_t>(tok_id)}));
    });
    std::printf("\n");
}

// =============================================================================
// Generation mode -- OmniVoice masked-diffusion text to speech
// =============================================================================

void run_omnivoice(const CliArgs& args, const std::string& prompt) {
    const std::string lm_path = args.weights.value_or("models/omnivoice-base-Q8_0.gguf");
    const std::string codec_path = args.snac.value_or("models/omnivoice-tokenizer-Q8_0.gguf");
    const std::string out_path = args.out.value_or("out.wav");

    for (const std::string& p : {lm_path, codec_path}) {
        if (!std::filesystem::exists(p)) {
            die("OmniVoice weights not found at " + p +
                "\n       Fetch both halves with:\n"
                "         curl -L -o models/omnivoice-base-Q8_0.gguf \\\n"
                "           https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/"
                "omnivoice-base-Q8_0.gguf\n"
                "         curl -L -o models/omnivoice-tokenizer-Q8_0.gguf \\\n"
                "           https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/"
                "omnivoice-tokenizer-Q8_0.gguf");
        }
    }

    // The LM's GGUF carries its own byte-level vocabulary, so no separate
    // tokenizer file is needed.
    const Result<GgufFile> gguf = GgufFile::open(lm_path);
    if (!gguf) {
        die("failed to open GGUF: " + gguf.error());
    }
    const Result<HfBpeTokenizer> tok = load_gguf_tokenizer(*gguf);
    if (!tok) {
        die("failed to build the tokenizer from GGUF: " + tok.error());
    }

    Config6 cfg = Config6::omnivoice();
    cfg.rope_pairing =
        args.rope_interleaved ? RopePairing::Interleaved : RopePairing::HalfSplit;

    std::fprintf(stderr, "[ OmniVoice ] Loading the language model from %s...\n",
                 lm_path.c_str());
    Result<OmniLm> lm = OmniLm::load(lm_path, cfg);
    if (!lm) {
        die("failed to load the language model: " + lm.error());
    }
    if (args.quantize) {
        const std::size_t n = lm->quantize_projections_to_q4k();
        std::fprintf(stderr, "[ OmniVoice ] Requantized %zu projections -> %.2f GB\n", n,
                     static_cast<double>(lm->weight_bytes()) / 1e9);
    }
    release_memory_to_os();
    print_rss("after weight load");

    std::fprintf(stderr, "[ OmniVoice ] Loading the codec from %s...\n", codec_path.c_str());
    const Result<OmniCodecDecoder> codec =
        OmniCodecDecoder::load(codec_path, OmniCodecConfig::defaults());
    if (!codec) {
        die("failed to load the codec: " + codec.error());
    }
    std::fprintf(stderr, "[ OmniVoice ] Codec decoder: %zu parameters\n",
                 codec->parameter_count());

    OmniRequest request;
    request.text = prompt;
    request.language = args.language.value_or("");
    request.instruct = args.instruct.value_or("");
    request.duration_seconds = args.duration;

    // Voice cloning: read the reference, encode it, and hand the codes over as
    // decided positions. The analysis half of the codec is only loaded when
    // there is something to analyse -- it is bigger than the synthesis half.
    if (args.ref_audio) {
        if (!args.ref_text || args.ref_text->empty()) {
            die("--ref-audio needs --ref-text: the model has to know which part of the "
                "prompt it has already heard");
        }
        const Result<WavFile> wav = read_wav(*args.ref_audio);
        if (!wav) {
            die("failed to read the reference audio: " + wav.error());
        }
        const Result<OmniReference> ref =
            omni_prepare_reference(wav->mono(), wav->sample_rate, codec->config);
        if (!ref) {
            die("failed to prepare the reference audio: " + ref.error());
        }
        std::fprintf(stderr,
                     "[ OmniVoice ] Reference: %.2f s, %zu Hz, %zu ch, rms %.4f\n",
                     ref->seconds(codec->config), wav->sample_rate, wav->channels,
                     static_cast<double>(ref->rms));
        if (ref->seconds(codec->config) > 20.0) {
            std::fprintf(stderr,
                         "[ OmniVoice ] Warning: reference clips over 20 s slow generation "
                         "down and clone no better; 3-10 s is the useful range\n");
        }

        std::fprintf(stderr, "[ OmniVoice ] Loading the codec encoder from %s...\n",
                     codec_path.c_str());
        const Result<OmniCodecEncoder> encoder =
            OmniCodecEncoder::load(codec_path, OmniCodecConfig::defaults());
        if (!encoder) {
            die("failed to load the codec encoder: " + encoder.error());
        }
        const Result<std::vector<std::vector<std::uint32_t>>> codes =
            encoder->encode(ref->samples);
        if (!codes) {
            die("failed to encode the reference audio: " + codes.error());
        }
        std::fprintf(stderr, "[ OmniVoice ] Encoded the reference to %zu frames\n",
                     (*codes)[0].size());
        request.ref_codes = *codes;
        request.ref_text = *args.ref_text;
        request.ref_rms = ref->rms;
    }
    request.debug = args.debug;
    request.gen.num_step = args.steps;
    request.gen.guidance_scale = args.guidance;
    request.gen.seed = args.seed;

    char duration_field[32];
    if (request.duration_seconds > 0.0f) {
        std::snprintf(duration_field, sizeof(duration_field), "%.1fs",
                      static_cast<double>(request.duration_seconds));
    } else {
        std::snprintf(duration_field, sizeof(duration_field), "estimated");
    }
    std::fprintf(stderr,
                 "[ OmniVoice ] lang=%s steps=%zu guidance=%.2f duration=%s rope=%s clone=%s\n",
                 request.language.empty() ? "None" : request.language.c_str(),
                 request.gen.num_step, static_cast<double>(request.gen.guidance_scale),
                 duration_field, args.rope_interleaved ? "interleaved" : "half-split",
                 request.ref_codes.empty() ? "off" : "on");
    std::fprintf(stderr, "[ OmniVoice ] Synthesising: \"%s\"\n", prompt.c_str());

    const OmniForward* accel = nullptr;
#if RT_FEATURE_METAL
    // The whole forward pass in one command buffer. Unlike the decode engines,
    // this exists for the GEMMs rather than despite them: a diffusion step is a
    // full-sequence pass over a few hundred positions, which is the shape the
    // matrix units are for.
    std::fprintf(stderr, "[ Metal ] Uploading weights...\n");
    const auto t_upload = std::chrono::steady_clock::now();
    std::unique_ptr<MetalOmniContext> metal_ctx =
        MetalOmniContext::create(*lm, MetalOmniContext::kMaxTokens);
    if (metal_ctx) {
        std::fprintf(stderr, "[ Metal ] %.2f GB in GPU buffers, uploaded in %.1f s\n",
                     static_cast<double>(metal_ctx->buffer_bytes()) / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t_upload)
                         .count());
        release_memory_to_os();
        print_rss("after the GPU upload");
        accel = metal_ctx.get();
    } else {
        // create() frees the CPU weights as it goes, so a partial failure
        // leaves nothing to fall back to.
        die("the Metal engine failed to initialise");
    }
#endif

    const Result<OmniResult> result = omni_synthesize(*lm, *codec, *tok, request, accel);
    if (!result) {
        die("synthesis failed: " + result.error());
    }

    const WaveStats stats = wave_stats(result->samples);
    std::fprintf(stderr,
                 "[ OmniVoice ] %zu frames, %zu prompt tokens, %zu forward passes -> %zu samples\n",
                 result->frames, result->prompt_tokens, result->forward_passes,
                 result->samples.size());
    std::fprintf(stderr,
                 "[ OmniVoice ] %.2f s audio in %.2f s generate + %.2f s decode (RTF %.2f)\n",
                 result->audio_seconds(), result->generate_seconds, result->decode_seconds,
                 result->realtime_factor());
    std::fprintf(stderr, "[ OmniVoice ] waveform: %s\n", stats.describe().c_str());
    if (!stats.looks_like_speech()) {
        std::fprintf(stderr,
                     "[ OmniVoice ] Warning: the waveform statistics do not look like speech\n");
    }

    if (const Result<void> ok = write_wav(out_path, result->samples, result->sample_rate, 1);
        !ok) {
        die("failed to write the WAV: " + ok.error());
    }
    std::fprintf(stderr, "[ OmniVoice ] Wrote %s\n", out_path.c_str());
}

// =============================================================================
// Generation mode -- Orpheus text to speech
// =============================================================================

void run_orpheus(const CliArgs& args, const std::string& prompt) {
    const std::string& weights_path = *args.weights;
    const std::string snac_path = args.snac.value_or("models/snac_24khz.bin");
    const std::string out_path = args.out.value_or("out.wav");
    const std::string voice = args.voice.value_or("tara");

    if (!std::filesystem::exists(snac_path)) {
        die("SNAC codec weights not found at " + snac_path +
            "\n       Fetch them with:\n         curl -L -o " + snac_path +
            " https://huggingface.co/hubertsiuzdak/snac_24khz/resolve/main/pytorch_model.bin\n"
            "       or point --snac at an existing copy.");
    }

    // The GGUF carries its own vocabulary and merges, so no --tokenizer-dir is
    // needed -- which matters here, because the Orpheus repository is gated.
    std::fprintf(stderr, "[ Orpheus ] Reading %s...\n", weights_path.c_str());
    const Result<GgufFile> gguf = GgufFile::open(weights_path);
    if (!gguf) {
        die("failed to open GGUF: " + gguf.error());
    }
    const Result<HfBpeTokenizer> tok = load_gguf_tokenizer(*gguf);
    if (!tok) {
        die("failed to build the tokenizer from GGUF: " + tok.error());
    }

    LlamaModel model = LlamaModel::new_for_inference(Config5::orpheus_3b());
    if (const Result<void> ok = model.load_weights_from_gguf(weights_path); !ok) {
        die("failed to load weights: " + ok.error());
    }

    // How the file was quantized decides what to do next.
    //
    // A Q4_K_M checkpoint keeps most projections native and lifts only
    // attn_v, ffn_down and output to Q6_K -- those are the quality-sensitive
    // ones, chosen deliberately. Flattening them would throw that away, so
    // only the lm_head is requantized: it is the largest single read per
    // token, and at 156 940 entries it costs 964 MB as BF16 against 271 MB as
    // Q4_K.
    //
    // A uniformly higher-precision file -- Q8_0, F16 -- has no Q4_K tensors at
    // all, so every projection widens to BF16 and the model lands near 6.6 GB
    // with roughly 3.5x the per-token memory traffic. There is no deliberate
    // choice to preserve there, so requantizing all of them is the right call.
    const std::size_t projections = model.projection_count();
    const std::size_t widened = model.bf16_projection_count();
    if (widened * 2 > projections) {
        std::fprintf(stderr,
                     "[ Orpheus ] %zu of %zu projections were widened to BF16 (%.2f GB); "
                     "requantizing to Q4_K...\n",
                     widened, projections, static_cast<double>(model.weight_bytes()) / 1e9);
        const std::size_t converted = model.quantize_projections_to_q4k();
        std::fprintf(stderr, "[ Orpheus ] Requantized %zu projections, now %.2f GB\n", converted,
                     static_cast<double>(model.weight_bytes()) / 1e9);
    } else {
        std::fprintf(stderr, "[ Orpheus ] Quantizing lm_head to Q4_K...\n");
        model.quantize_lm_head();
    }
    release_memory_to_os();
    print_rss("after weight load");

    std::fprintf(stderr, "[ Orpheus ] Loading SNAC codec from %s...\n", snac_path.c_str());
    const Result<SnacDecoder> snac = SnacDecoder::load(snac_path, SnacConfig::snac_24khz());
    if (!snac) {
        die("failed to load the SNAC codec: " + snac.error());
    }
    std::fprintf(stderr, "[ Orpheus ] SNAC decoder: %zu parameters\n", snac->parameter_count());

    OrpheusRequest request;
    request.text = prompt;
    request.voice = voice;
    request.max_new = args.max_new;
    request.debug = args.debug;
    request.mask_to_audio = !args.no_audio_mask;
    request.sampling = orpheus_default_sampling(args.seed);
    // Explicit flags win over the reference defaults.
    request.sampling.temperature = args.temperature;
    request.sampling.top_p = args.top_p;
    request.sampling.top_k = args.top_k;
    request.sampling.repetition_penalty = args.rep_penalty;

    std::fprintf(stderr, "[ Orpheus ] voice=%s max_new=%zu temp=%.2f top_p=%.2f rep=%.2f\n",
                 voice.c_str(), request.max_new, static_cast<double>(request.sampling.temperature),
                 static_cast<double>(request.sampling.top_p),
                 static_cast<double>(request.sampling.repetition_penalty));
    std::fprintf(stderr, "[ Orpheus ] Synthesising: \"%s\"\n", prompt.c_str());

    OrpheusConfig cfg = OrpheusConfig::defaults();
    cfg.leading_bos = !args.no_leading_bos;
    const Result<OrpheusResult> result = orpheus_synthesize(model, *snac, *tok, request, cfg);
    if (!result) {
        die("synthesis failed: " + result.error());
    }

    const WaveStats stats = wave_stats(result->samples);
    std::fprintf(stderr,
                 "[ Orpheus ] %zu tokens -> %zu codes (%zu rejected) -> %zu groups -> %zu samples\n",
                 result->tokens_generated, result->codes_accepted, result->codes_rejected,
                 result->groups, result->samples.size());
    std::fprintf(stderr,
                 "[ Orpheus ] %.2f s audio in %.2f s generate + %.2f s decode (RTF %.2f)\n",
                 result->audio_seconds(), result->generate_seconds, result->decode_seconds,
                 result->realtime_factor());
    std::fprintf(stderr, "[ Orpheus ] waveform: %s\n", stats.describe().c_str());
    if (!stats.looks_like_speech()) {
        // Not fatal -- a short or quiet clip fails this legitimately -- but a
        // pipeline fault shows up here first, and every fault in this pipeline
        // sounds the same.
        std::fprintf(stderr,
                     "[ Orpheus ] Warning: the waveform statistics do not look like speech\n");
    }

    if (const Result<void> ok = write_wav(out_path, result->samples, result->sample_rate, 1);
        !ok) {
        die("failed to write the WAV: " + ok.error());
    }
    std::fprintf(stderr, "[ Orpheus ] Wrote %s\n", out_path.c_str());
}

// =============================================================================
// Generation mode -- Gemma 3 with loaded weights
// =============================================================================

void run_gemma3(const CliArgs& args, const std::string& prompt) {
    const std::string& weights_path = *args.weights;
    const std::string model_name = args.model.value_or("gemma3-1b");

    const bool is_gguf = looks_like_gguf(weights_path);
    const std::string weights_dir = weights_dir_of(weights_path, is_gguf);
    const HfBpeTokenizer tok = load_hf_tokenizer(args, weights_dir, is_gguf, "Gemma3");

    const Config4 config = model_name == "gemma3-4b" ? Config4::gemma3_4b() : Config4::gemma3_1b();
    std::fprintf(stderr, "[ Gemma3 ] Building %s model (%zu layers, hidden=%zu)...\n",
                 model_name.c_str(), config.num_hidden_layers, config.hidden_size);

    Gemma3Model model = Gemma3Model::new_for_inference(config);

    if (is_gguf) {
        std::fprintf(stderr, "[ Gemma3 ] Loading weights from GGUF: %s...\n",
                     weights_path.c_str());
        if (const Result<void> ok = model.load_weights_from_gguf(weights_path); !ok) {
            die("failed to load GGUF weights: " + ok.error());
        }
    } else {
        // Safetensors: try the binary cache first, and write one if it is
        // missing, so later runs skip the JSON parse and BF16 conversion.
        const std::string cache_path = weights_dir + "/gemma3-" + model_name + ".cache";
        const Result<bool> cached = model.load_cache(cache_path);
        if (cached && *cached) {
            std::fprintf(stderr, "[ Gemma3 ] Loaded weights from cache (%s).\n",
                         cache_path.c_str());
        } else {
            std::fprintf(stderr, "[ Gemma3 ] Loading weights from %s...\n", weights_dir.c_str());
            if (const Result<void> ok = model.load_weights_from_dir(weights_dir); !ok) {
                die("failed to load weights: " + ok.error());
            }
            std::fprintf(stderr, "[ Gemma3 ] Saving weight cache to %s...\n", cache_path.c_str());
            if (const Result<void> ok = model.save_cache(cache_path); !ok) {
                die("failed to save cache: " + ok.error());
            }
            std::fprintf(stderr, "[ Gemma3 ] Cache saved.\n");
        }
    }
    release_memory_to_os();
    print_rss("after weight load");

    if (args.quantize) {
        std::fprintf(stderr, "[ Gemma3 ] Quantizing weights to Q4...\n");
        model.quantize_all_weights();
        std::fprintf(stderr, "[ Gemma3 ] Quantization complete.\n");
    }

    // Gemma 3-IT needs its chat template:
    //   <bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
    // The special ids are injected directly -- encoding the literal text
    // "<start_of_turn>" would split it into ordinary subword pieces.
    // ids: bos=2, start_of_turn=105, end_of_turn=106, \n=107, user=2364,
    //      model=4368
    std::vector<std::size_t> token_ids{2, 105, 2364, 107};
    extend_encoded(token_ids, tok, prompt);
    token_ids.insert(token_ids.end(), {106, 107, 105, 4368, 107});

    dump_prompt_tokens(tok, token_ids);

    emit(prompt);
    model.generate_cached_streaming(token_ids, args.max_new, args.temperature, args.top_k,
                                    args.top_p, args.rep_penalty, args.seed, args.debug,
                                    args.draft_len, [&](std::size_t tok_id) {
                                        emit(tok.decode({static_cast<std::uint32_t>(tok_id)}));
                                    });
    std::printf("\n");
}

// =============================================================================
// Generation mode -- Qwen 3.5 with loaded weights
// =============================================================================

void run_qwen35(const CliArgs& args, const std::string& prompt) {
    const std::string& weights_path = *args.weights;
    const std::string model_name = args.model.value_or("qwen35-4b");

    const bool is_gguf = looks_like_gguf(weights_path);
    const std::string weights_dir = weights_dir_of(weights_path, is_gguf);
    const HfBpeTokenizer tok = load_hf_tokenizer(args, weights_dir, is_gguf, "Qwen3.5");

    ConfigQwen35 config = ConfigQwen35::qwen35_4b();
    if (model_name == "qwen35-9b") {
        config = ConfigQwen35::qwen35_9b();
    } else if (model_name == "qwen35-0.8b" || model_name == "qwen35-0_8b") {
        config = ConfigQwen35::qwen35_0_8b();
    }
    std::fprintf(stderr, "[ Qwen3.5 ] Building %s model (%zu layers, hidden=%zu)...\n",
                 model_name.c_str(), config.num_hidden_layers, config.hidden_size);

    Qwen35Model model = Qwen35Model::new_for_inference(config);

    if (is_gguf) {
        std::fprintf(stderr, "[ Qwen3.5 ] Loading weights from GGUF: %s...\n",
                     weights_path.c_str());
        if (const Result<void> ok = model.load_weights_from_gguf(weights_path); !ok) {
            die("failed to load GGUF weights: " + ok.error());
        }
    } else {
        std::fprintf(stderr, "[ Qwen3.5 ] Loading weights from %s...\n", weights_path.c_str());
        if (const Result<void> ok = model.load_weights_from_dir(weights_path); !ok) {
            die("failed to load safetensors weights: " + ok.error());
        }
    }
    release_memory_to_os();
    print_rss("after weight load");

    // Qwen 3.5 uses the ChatML template:
    //   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    // ids: <|im_start|> = 248045, <|im_end|> = 248046, \n = 198
    constexpr std::size_t kImStart = 248045;
    constexpr std::size_t kImEnd = 248046;
    constexpr std::size_t kNewline = 198;
    std::vector<std::size_t> token_ids{kImStart};
    extend_encoded(token_ids, tok, "user");
    token_ids.push_back(kNewline);
    extend_encoded(token_ids, tok, prompt);
    token_ids.push_back(kImEnd);
    token_ids.push_back(kNewline);
    token_ids.push_back(kImStart);
    extend_encoded(token_ids, tok, "assistant");
    token_ids.push_back(kNewline);

    dump_prompt_tokens(tok, token_ids);

    emit(prompt);
    model.generate_cached_streaming(token_ids, args.max_new, args.temperature, args.top_k,
                                    args.top_p, args.rep_penalty, args.seed, args.debug,
                                    [&](std::size_t tok_id) {
                                        emit(tok.decode({static_cast<std::uint32_t>(tok_id)}));
                                    });
    std::printf("\n");
}

// =============================================================================
// Generation mode -- Gpt2 trained on the built-in corpus
// =============================================================================

void run_gpt2_generate(const CliArgs& args, const std::string& prompt) {
    const CharTokenizer tokenizer = CharTokenizer::from_text(kCorpus);
    const std::size_t vocab_size = tokenizer.vocab_size();
    constexpr std::size_t context_length = 64;

    const auto [train_data, val_data] =
        TextDataset::train_val_split(kCorpus, tokenizer, context_length);

    Config model_config;
    model_config.vocab_size = vocab_size;
    model_config.context_length = context_length;
    model_config.d_model = 64;
    model_config.n_layers = 4;
    model_config.n_heads = 4;

    InitRng rng(42);
    const Gpt2 model(model_config, rng);

    if (args.checkpoint) {
        std::fprintf(stderr, "[ Load ] Restoring from checkpoint: %s\n", args.checkpoint->c_str());
        if (const Result<void> ok = restore_checkpoint(*args.checkpoint, model.parameters()); !ok) {
            die("failed to restore checkpoint: " + ok.error());
        }
    } else {
        std::fprintf(stderr, "[ Train ] vocab=%zu context=%zu steps=%zu\n", vocab_size,
                     context_length, args.train_steps);
        TrainConfig2 cfg;
        cfg.max_steps = args.train_steps;
        cfg.eval_interval = args.train_steps / 5;
        cfg.learning_rate = 3e-3f;
        cfg.grad_clip = 1.0f;
        (void)train2(model, train_data, val_data, cfg);
    }

    // Stream through the KV cache: O(T) per step instead of O(T^2).
    std::vector<std::size_t> token_ids;
    extend_encoded(token_ids, tokenizer, prompt);
    if (token_ids.empty()) {
        token_ids.push_back(0);
    }

    emit(prompt);
    model.generate_cached_streaming(token_ids, args.max_new, args.temperature, args.top_k,
                                    [&](std::size_t tok_id) {
                                        emit(tokenizer.decode(
                                            {static_cast<std::uint32_t>(tok_id)}));
                                    });
    std::printf("\n");
}

// =============================================================================
// Benchmark mode -- scalar autograd vs tensor autodiff
// =============================================================================

void run_benchmark(std::size_t train_steps) {
    using Clock = std::chrono::steady_clock;

    std::printf("╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║        Rich Triplet — LLM from scratch in C++              ║\n");
    std::printf("║        Scalar autograd  vs  Tensor autodiff benchmark       ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");
    std::printf("\n");

    const CharTokenizer tokenizer = CharTokenizer::from_text(kCorpus);
    const std::size_t vocab_size = tokenizer.vocab_size();
    constexpr std::size_t context_length = 16;

    const auto [train_data, val_data] =
        TextDataset::train_val_split(kCorpus, tokenizer, context_length);

    std::printf("[ Setup ] Building tokenizer and dataset...\n");
    std::printf("         Vocabulary:     %zu unique characters\n", vocab_size);
    std::printf("         Context window: %zu tokens\n", context_length);
    std::printf("         Random loss:    %.4f  (= ln(%zu))\n",
                std::log(static_cast<double>(vocab_size)), vocab_size);

    Config model_config;
    model_config.vocab_size = vocab_size;
    model_config.context_length = context_length;
    model_config.d_model = 32;
    model_config.n_layers = 2;
    model_config.n_heads = 2;

    const std::size_t eval_interval = std::max<std::size_t>(train_steps / 5, 1);

    // -------------------------------------------------------------------------
    // Phase A -- scalar autograd
    // -------------------------------------------------------------------------
    std::printf("\n═══════════════════════════════════════════════════════════════\n");
    std::printf(" PHASE A: Scalar autograd  (one Value node per weight element)\n");
    std::printf("═══════════════════════════════════════════════════════════════\n");

    InitRng rng_a(42);
    const Gpt scalar_model(model_config, rng_a);
    {
        const std::size_t n = scalar_model.parameters().size();
        std::printf(" Model:  %.1fK scalar nodes  (%d param matrices × ~%zu elements avg)\n",
                    static_cast<double>(n) / 1000.0, 0, n);
    }

    TrainConfig scalar_cfg;
    scalar_cfg.max_steps = train_steps;
    scalar_cfg.eval_interval = eval_interval;
    scalar_cfg.learning_rate = 1e-3f;
    scalar_cfg.grad_clip = 1.0f;

    const Clock::time_point t_scalar_start = Clock::now();
    (void)train(scalar_model, tokenizer, train_data, val_data, scalar_cfg);
    const double t_scalar = std::chrono::duration<double>(Clock::now() - t_scalar_start).count();

    std::printf("\n[ Generation — scalar model ]\n");
    (void)generate(scalar_model, tokenizer, "Il ", 80, 0.8f, 5);
    (void)generate(scalar_model, tokenizer, "The ", 80, 0.8f, 5);

    // -------------------------------------------------------------------------
    // Phase B -- tensor autodiff
    // -------------------------------------------------------------------------
    std::printf("\n═══════════════════════════════════════════════════════════════\n");
    std::printf(" PHASE B: Tensor autodiff  (one TensorNode per weight matrix)\n");
    std::printf("═══════════════════════════════════════════════════════════════\n");

    InitRng rng_b(42);
    const Gpt2 tensor_model(model_config, rng_b);
    {
        const std::vector<TensorNode> params = tensor_model.parameters();
        std::size_t total_elems = 0;
        for (const TensorNode& p : params) {
            total_elems += p.data().rows * p.data().cols;
        }
        std::printf(" Model:  %zu tensor nodes  (%zu elements total)\n", params.size(),
                    total_elems);
    }

    TrainConfig2 tensor_cfg;
    tensor_cfg.max_steps = train_steps;
    tensor_cfg.eval_interval = eval_interval;
    tensor_cfg.learning_rate = 1e-3f;
    tensor_cfg.grad_clip = 1.0f;

    const Clock::time_point t_tensor_start = Clock::now();
    (void)train2(tensor_model, train_data, val_data, tensor_cfg);
    const double t_tensor = std::chrono::duration<double>(Clock::now() - t_tensor_start).count();

    std::printf("\n[ Generation — tensor model ]\n");
    for (const char* prompt_str : {"Il ", "The "}) {
        std::vector<std::size_t> ids;
        extend_encoded(ids, tokenizer, prompt_str);
        emit(prompt_str);
        tensor_model.generate_cached_streaming(ids, 80, 0.8f, 5, [&](std::size_t tok_id) {
            emit(tokenizer.decode({static_cast<std::uint32_t>(tok_id)}));
        });
        std::printf("\n");
    }

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    std::printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║                   Benchmark Summary                         ║\n");
    std::printf("╠══════════════════════════════════════════════════════════════╣\n");
    std::printf("║  Steps: %4zu                                                ║\n", train_steps);
    std::printf("║                                                              ║\n");
    std::printf("║  Scalar autograd:   %8.2fs   (%5.0f ms/step)           ║\n", t_scalar,
                t_scalar * 1000.0 / static_cast<double>(train_steps));
    std::printf("║  Tensor autodiff:   %8.2fs   (%5.0f ms/step)           ║\n", t_tensor,
                t_tensor * 1000.0 / static_cast<double>(train_steps));
    std::printf("║                                                              ║\n");
    std::printf("║  Speedup:           %7.1fx                                ║\n",
                t_scalar / t_tensor);
    std::printf("║                                                              ║\n");
    std::printf("║  Why faster?                                                 ║\n");
    std::printf("║  • Scalar: ~400K nodes in graph → 400K backward visits      ║\n");
    std::printf("║  • Tensor:    ~60 nodes in graph →   60 backward visits     ║\n");
    std::printf("║  • Each tensor backward does a SIMD-able matmul instead     ║\n");
    std::printf("║    of millions of individual scalar multiply-accumulate ops  ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");

    std::printf("\nDone.\n");
}

}  // namespace

int main(int argc, char** argv) {
    const CliArgs args = parse_args(argc, argv);

    // -------------------------------------------------------------------------
    // Pretokenize mode: a text file -> a binary corpus
    // -------------------------------------------------------------------------
    if (args.pretokenize) {
        const auto& [src, dst] = *args.pretokenize;
        std::fprintf(stderr, "[pretokenize] Reading: %s\n", src.c_str());
        std::fprintf(stderr, "[pretokenize] Output:  %s\n", dst.c_str());

        Result<std::size_t> n_tokens = err(std::string{});
        if (args.vocab && args.merges) {
            Result<BpeTokenizer> tok = BpeTokenizer::from_files(*args.vocab, *args.merges);
            if (!tok) {
                die("failed to load BPE tokenizer: " + tok.error());
            }
            n_tokens = TokenizedDataset::write_bin_from_file(dst, src, *tok);
        } else {
            // The char vocabulary has to come from the whole file, so this path
            // reads it all rather than streaming.
            std::ifstream in(src);
            if (!in) {
                die("cannot read source file");
            }
            const std::string text((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
            const CharTokenizer tok = CharTokenizer::from_text(text);
            std::fprintf(stderr, "[pretokenize] Char vocab size: %zu\n", tok.vocab_size());
            n_tokens = TokenizedDataset::write_bin(dst, text, tok);
        }
        if (!n_tokens) {
            die("pretokenize failed: " + n_tokens.error());
        }
        std::fprintf(stderr, "[pretokenize] Done: %zu tokens → %s\n", *n_tokens, dst.c_str());
        return 0;
    }

    // -------------------------------------------------------------------------
    // Generation mode
    // -------------------------------------------------------------------------
    if (args.prompt) {
        const bool is_omnivoice = args.model && args.model->starts_with("omnivoice");
        const bool is_orpheus = args.model && args.model->starts_with("orpheus");
        const bool is_qwen35 = args.model && args.model->starts_with("qwen35");
        // A .gguf file with no --model is assumed to be Gemma 3, which is the
        // only architecture this CLI ever loaded from GGUF first.
        const bool is_gemma3 = args.tokenizer_model.has_value() ||
                               (args.model && args.model->starts_with("gemma3")) ||
                               (!is_qwen35 && !is_orpheus && !is_omnivoice && args.weights &&
                                args.weights->ends_with(".gguf"));
        if (is_omnivoice) {
            run_omnivoice(args, *args.prompt);
        } else if (is_orpheus) {
            if (!args.weights) {
                die("--model orpheus-3b needs --weights pointing at the GGUF file");
            }
            run_orpheus(args, *args.prompt);
        } else if (is_qwen35) {
            run_qwen35(args, *args.prompt);
        } else if (is_gemma3) {
            run_gemma3(args, *args.prompt);
        } else if (args.weights) {
            run_gpt_oss(args, *args.prompt);
        } else {
            run_gpt2_generate(args, *args.prompt);
        }
        return 0;
    }

    // -------------------------------------------------------------------------
    // Benchmark mode
    // -------------------------------------------------------------------------
    if (args.benchmark) {
        run_benchmark(args.train_steps);
        return 0;
    }

    print_help();
    return 0;
}
