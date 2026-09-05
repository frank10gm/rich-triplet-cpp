# Rich Triplet — LLM from scratch in C++

A complete LLM stack built from first principles in C++23, **with no ML dependencies**. Every component — matrix math, automatic differentiation, attention, quantization, Metal GPU kernels, tokenizer — is written from scratch.

This is a port of [the Rust original](../rich-triplet), kept numerically identical: the parity work behind it is described in [Porting notes](#porting-notes).

Supports **Gemma 3** and **Qwen 3.5** text inference on Apple Silicon with Q4_K_M / Q4_0 GGUF weights and full-graph Metal GPU decode, plus two complete text-to-speech stacks — text in, WAV out, neural audio codecs included: **Orpheus**, an autoregressive Llama 3.2 emitting SNAC codes, and **OmniVoice**, a masked-diffusion Qwen3 that unmasks eight codebooks in parallel and clones a voice from a few seconds of reference audio.

It also carries a text-to-image stack: **FLUX.1-schnell**, a 12B rectified-flow transformer with its T5-XXL and CLIP-L text encoders and a 16-channel autoencoder, text in and PNG out, with a full-graph Metal engine that keeps the transformer quantized on the GPU and widens each weight only as it is used. The design and its build log are in [docs/text-to-image.md](docs/text-to-image.md) — including [what is not yet verified](docs/text-to-image.md#what-is-not-verified), which is that it has never been run against the real checkpoints.

---

## What this project is

Eleven transformer implementations (GPT-2 scalar, GPT-2 tensor, GPT-OSS, Gemma 3, Qwen 3.5, Llama 3.2, a bidirectional Qwen3 used as a diffusion model, HuBERT reading raw audio, a T5 encoder, a CLIP text tower, and FLUX's two-stream MMDiT), two neural audio codec decoders and one encoder, an image autoencoder, a tensor autodiff engine, Apple Metal GPU acceleration, Flash Attention, GGUF / safetensors / torch-pickle weight loading, and a CLI for inference, training, speech synthesis and image generation.

| File | What you understand after writing it |
|---|---|
| `tensor.cpp` | Row-major storage, matmul, softmax |
| `autograd.cpp` | The chain rule as a computation graph, shared-pointer ownership |
| `mat.cpp` / `quant_*.cpp` | Why PyTorch is fast: matrix VJPs, Q4_K/BF16 GEMV, ARM NEON + SDOT |
| `nn.cpp` / `nn2.cpp` | Linear layers, GELU, RMSNorm, SwiGLU from first principles |
| `transformer.cpp` / `transformer2.cpp` | Q/K/V attention, causal masking, Flash Attention |
| `transformer3.cpp` | GPT-OSS: RoPE, GQA, Mixture of Experts, safetensors |
| `transformer4.cpp` | Gemma 3: NeoX RoPE, sliding window, 4-norm blocks, GGUF |
| `transformer_qwen35.cpp` | Qwen 3.5: Gated DeltaNet + softmax attention hybrid |
| `transformer5.cpp` | Llama 3.2: the two RoPE pair conventions, and why GGUF needs the other one |
| `transformer6.cpp` | Bidirectional attention, and what a model without a KV cache costs |
| `hubert.cpp` | A transformer whose input is a waveform, and whose position is a convolution |
| `conv1d.cpp` | Dilated, grouped and transposed 1-D convolution; Snake; weight norm |
| `snac.cpp` | Multi-scale residual vector quantization, and a codec decoder |
| `orpheus.cpp` | Audio tokens: slot offsets, frame de-interleaving, resynchronisation |
| `omnivoice.cpp` | Masked diffusion decoding: confidence unmasking, classifier-free guidance |
| `omnivoice_codec.cpp` | Dense residual convolution, and why im2col earns its memory |
| `conv2d.cpp` | The same operator one dimension up, and why the layout follows from it |
| `vae.cpp` | GroupNorm's statistics span space, and what a latent is scaled by |
| `t5.cpp` | An encoder with no positional embedding, and no attention scaling either |
| `clip_text.cpp` | Why a text encoder is causal, and where its pooled vector comes from |
| `flux.cpp` | Two stream types, adaptive layer norm, and three-axis RoPE |
| `duration.cpp` | That Unicode has opinions about how long a character takes to say |
| `resample.cpp` | Polyphase rate conversion, and why the filter has to be *that* filter |
| `torch_pickle.cpp` | ZIP central directories, and just enough pickle to be safe |
| `wav.cpp` | RIFF in both directions, and how to tell speech from noise without listening |
| `png.cpp` | That DEFLATE has a mode needing no compressor, and CRC-32 either way |
| `gguf.cpp` | GGUF file format: Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, BF16, F16, F32 |
| `tokenizer.cpp` | Character, BPE, SentencePiece and HuggingFace BPE tokenizers |
| `metal_ops.mm` | Metal GPU tiled matmul, Q4_K/BF16 GEMV kernels |
| `metal_decode.mm` | Full-graph Metal decode: ~717 dispatches per token in one command buffer |
| `metal_omnivoice.mm` | A GPU GEMM worth writing, and how to know a second implementation agrees |
| `metal_flux.mm` | Keeping 12B quantized at rest, and streaming softmax past the threadgroup limit |
| `train.cpp` / `train2.cpp` | AdamW, gradient clipping, autoregressive generation |

---

## Building

Requires CMake 3.24+ and a C++23 compiler. Apple clang 21 (Xcode 26) works out of the box; the Metal backend needs macOS.

```bash
# CPU build (Accelerate BLAS + NEON)
cmake -S . -B build
cmake --build build -j

# Metal GPU build (Apple Silicon)
cmake -S . -B build-metal -DRT_METAL=ON
cmake --build build-metal -j
```

The binary lands at `build/rich-triplet`, the test runner at `build/tests/rt_tests`.

### CMake options

| Option | Default | Effect |
|---|---|---|
| `RT_BLAS` | ON | Use Apple Accelerate `cblas_sgemm` for large matmuls |
| `RT_METAL` | OFF | Build the Metal GPU kernels and decode engines |
| `RT_PARALLEL` | ON | Multi-threaded matmul and GEMV |
| `RT_WEIGHT_LOADING` | ON | safetensors and GGUF loaders |
| `RT_MMAP_LOADING` | ON | mmap-based shard loading for GPT-OSS |
| `RT_Q4K_LM_HEAD` | OFF | Quantize the lm_head to Q4_K |
| `RT_BUILD_TESTS` | ON | Build the Catch2 test suite |

Floating-point contraction is forced off (`-ffp-contract=off -fno-fast-math`) across the project. Letting the compiler fuse `a*b+c` into an FMA changes results in the last bits and would drift from the reference.

---

## Gemma 3 inference

```bash
./build-metal/rich-triplet \
  --model gemma3-4b \
  --weights ./models/gemma-3-4b-it-q4_0.gguf \
  --tokenizer-dir ./models/gemma-3-4b-it/ \
  --prompt "What is the capital of France?" \
  --max-new 200 \
  --temp 0.8
```

Both Q4_K_M and Q4_0 GGUF files work. Q4_0 weights are repacked to Q4_K at load time, which is both smaller (0.56 vs 0.63 bytes/element) and the only quantized form with a Metal GEMV and an SDOT CPU kernel.

GGUF files do not carry a tokenizer, so `--tokenizer-dir` has to point at a directory holding `tokenizer.json` — usually the original HuggingFace checkout.

### How it works

1. **CPU prefill** runs the prompt through all 34 layers, filling the KV cache.
2. **Metal decode** encodes the entire forward pass — RMSNorm, RoPE, GEMV, attention, GELU, residuals — as ~717 dispatches in a single command buffer, so there are no CPU round-trips per layer.
3. **KV cache** lives in GPU `MTLBuffer`s (`StorageModeShared`) and is synced once from the CPU after prefill, after which the CPU weights and cache are freed.
4. **Speculative decoding** (`--draft-len N`) drafts continuations from an n-gram match over the generation history and verifies several at once in a batched dispatch.

### GGUF weight types

| Type | How it is stored in memory |
|---|---|
| Q4_K | Native Q4_K blocks (0.56 bytes/element) — fastest path |
| Q4_0 | Repacked to Q4_K at load time |
| Q5_K, Q6_K, Q8_0 | Dequantized to BF16 (2 bytes/element) |
| BF16, F16, F32 | BF16 in memory (2 bytes/element) |

---

## Qwen 3.5 inference

```bash
./build-metal/rich-triplet \
  --model qwen35-0.8b \
  --weights ./models/qwen3.5-0.8b-q4_k_m.gguf \
  --tokenizer-dir ./models/Qwen3.5-0.8B/ \
  --prompt "Explain attention in one paragraph." \
  --max-new 200
```

Qwen 3.5 is a hybrid: three quarters of its layers use Gated DeltaNet (linear attention carrying a recurrent state, with a causal conv1d and a delta-rule update) and every fourth uses softmax GQA with a sigmoid output gate and partial RoPE. Both kinds have their own Metal encode path, and both live in one command buffer.

---

## Orpheus text to speech

```bash
# The codec weights (80 MB) -- fetched once, shared by every language
curl -L -o models/snac_24khz.bin \
  https://huggingface.co/hubertsiuzdak/snac_24khz/resolve/main/pytorch_model.bin

./build/rich-triplet \
  --model orpheus-3b \
  --weights ./models/Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf \
  --snac ./models/snac_24khz.bin \
  --prompt "Ciao, mi chiamo Giulia. Oggi è una bella giornata a Roma." \
  --voice giulia \
  --out giulia.wav
```

Emotion tags like `<laugh>` and `<sigh>` are ordinary text — the tokenizer
merges them like any other word, so they need no special handling.

### Checkpoints and voices

The published fine-tunes share an architecture, a vocabulary and a codec, and
differ only in language. Any of them loads with no code changes.

| Checkpoint | Languages | Voices |
|---|---|---|
| `canopylabs/orpheus-3b-0.1-ft` | en | tara, leah, jess, leo, dan, mia, zac, zoe |
| `lex-au/Orpheus-3b-Italian_Spanish-FT-Q8_0` | it, es | pietro, giulia, carlo · javi, sergio, maria |

A voice is only a prompt prefix, so naming one the checkpoint was not trained
on still synthesises — it just will not sound like a consistent speaker. The
CLI warns.

The Italian and Spanish weights are a **research release**, and it shows in the
download counts: a few hundred against a quarter of a million for the English
fine-tune. Judge the output by ear before building anything on it.

No `--tokenizer-dir` is needed. The GGUF carries its own 156 940-entry
vocabulary and 280 147 merges, which matters because the Orpheus repositories
are gated on HuggingFace — the weights are freely mirrored as GGUF but
`tokenizer.json` is not.

Nothing in the pipeline is language-specific. The tokenizer is byte-level BPE,
so accented text encodes and round-trips without special handling, and the
codec is phonetically neutral. The language lives entirely in the weights.

### How it works

1. **The backbone is a Llama 3.2 3B** whose vocabulary has been extended by
   28 672 audio codes. It does not emit audio; it emits codec tokens.
2. **Seven tokens make a frame.** `<custom_token_0>` is id 128256, and a code
   is `id - 128266 - (slot * 4096)`, so audio ids run 128266–156937 across
   seven codebook slots of 4096.
3. **A frame is 2048 samples** — 85.33 ms at 24 kHz. Realtime therefore needs
   about 82 tokens a second.
4. **SNAC reconstructs the waveform** from three codebooks running at 1/4, 1/2
   and 1/1 of the frame rate, then four transposed-convolution blocks upsample
   by 8, 8, 4 and 2 for 512 samples per frame.

### How the file was quantized decides what happens at load

The two published checkpoints are packaged differently, and each needs
something the other does not. Both cases are handled automatically; the CLI
prints which branch it took.

| | English Q4_K_M | Italian/Spanish Q8_0 |
|---|---|---|
| On disk | 2.36 GB | 3.52 GB |
| Tensors | 256 | 255 — **lm_head is weight-tied** |
| Widened to BF16 | 29 of 197 projections | **197 of 197** |
| After load | 4.08 GB | 7.57 GB |
| Action taken | requantize lm_head only | requantize every projection |
| Resident | **3.5 GB** | **2.8 GB** |

A Q4_K_M file keeps most projections native and lifts only `attn_v`,
`ffn_down` and `output` to Q6_K — those are the quality-sensitive ones, chosen
deliberately by the quantizer. Flattening them to Q4_K would throw that choice
away, so only the lm_head is requantized: at 156 940 entries it costs 964 MB as
BF16 against 271 MB as Q4_K, and it is the largest single read per token.

A uniformly higher-precision file has no Q4_K tensors at all, so every
projection widens and there is no deliberate choice to preserve. Requantizing
all of them is the right call, and lands *below* the Q4_K_M model.

The decision is made on the fraction of projections widened, not on a byte
threshold, because that fraction is what actually distinguishes the two cases.

Weight tying is free: `MatBf16` is reference-counted, so a checkpoint with no
`output.weight` has its lm_head adopt the embedding's bits rather than copy
964 MB of them. The orientations already agree — the table is `[vocab, hidden]`
and `Linear2` computes `input @ weight.T`.

### Measured on an M3 Pro (18 GB, CPU build)

| | |
|---|---|
| Resident weights | 2.8–3.5 GB depending on the checkpoint |
| Decode | ~16 tokens/s |
| Realtime factor | ~5.3 |
| SNAC decode | ~0.1 s per second of audio |

So a 4 s clip takes about 21 s. Decode runs on the CPU and the Metal build
measures the same, for a duller reason than it looks: with `RT_BLAS` on --
which is the default -- `Mat::matmul` dispatches to Accelerate and never
reaches the GPU at all, so a Metal build without a dedicated engine is a CPU
build. Two things would close most of the gap — a `metal_decode_llama.mm`
trimmed from the Gemma engine, and speculative decoding, which is already
implemented for Gemma. OmniVoice has such an engine; Orpheus does not yet.

### Diagnosing it

Every fault in a speech pipeline sounds the same — a wrong RoPE convention, a
wrong codebook stride and a wrong convolution padding all produce noise. Two
things make that tractable without a reference implementation to diff against.

**Exact length invariants, asserted in code.** Each upsampling block
multiplies its input length by exactly its stride, and each residual unit
preserves length exactly, so `n_frames * 512` has no slack. Every padding or
`output_padding` mistake breaks the multiple and trips an assertion instead of
degrading the audio.

**Waveform statistics, printed every run.** Speech at 24 kHz sits near an RMS
of 0.03–0.2 with negligible DC offset and a low zero-crossing rate; a decoder
fed bad latents saturates its output `tanh` and lands near 0.5 with most
samples at the rails. `wave_stats` reports both and the CLI warns when the
numbers do not look like speech.

`--debug` adds a per-token dump of which codebook slot each id actually falls
in against the slot the stream expected. A healthy stream walks 0, 1, 2, 3, 4,
5, 6 and repeats; anything else is the frame structure breaking down, which is
invisible in the audio itself.

---

## OmniVoice text to speech

```bash
# Both halves of the model -- the LM and its codec (0.94 GB together)
curl -L -o models/omnivoice-base-Q8_0.gguf \
  https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-base-Q8_0.gguf
curl -L -o models/omnivoice-tokenizer-Q8_0.gguf \
  https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-tokenizer-Q8_0.gguf

./build/rich-triplet \
  --model omnivoice \
  --prompt "Ciao, mi chiamo Giulia. Oggi è una bella giornata a Roma." \
  --language Italian \
  --steps 12 \
  --out giulia.wav
```

OmniVoice is a **masked diffusion** language model, not an autoregressive one,
and that single fact reaches all the way down into the attention kernel. It is
also a much smaller model than Orpheus — a Qwen3 0.6B backbone against a Llama
3.2 3B — so both halves together are 0.94 GB on disk against Orpheus's 2.4–3.5.
Speed depends on `--steps`: at 12 a 4 s clip takes about 8 s, at the reference
default of 32 about 21, which is where Orpheus lands too.

`--language` and `--instruct` are free text, not enumerations: there is no
voice list, because a voice is described rather than named. `--instruct
"a calm young woman"` is a valid request, and so is leaving it out.

### Two GGUF files, and what is in each

| | `omnivoice-base` | `omnivoice-tokenizer` |
|---|---|---|
| On disk | 0.66 GB | 0.29 GB |
| Holds | the diffusion LM | the audio codec |
| Loaded | 312 tensors, 1.23 GB resident | 21.6 M to synthesise, 163 M to analyse |
| Also carries | its own 151 676-entry vocabulary | both directions of the codec |

The LM's GGUF embeds its tokenizer, so there is no `--tokenizer-dir`. The
codec's file holds both halves: `acoustic_decoder` and the codebooks for
synthesis, and `acoustic_encoder`, `encoder_semantic` and a 94 M-parameter
`semantic_model` for analysis. Plain synthesis loads only the first, about 190
of its 486 tensors; `--ref-audio` loads the rest.

### How it works

1. **Every audio position starts masked.** The sequence is built once — style
   markers, language, instruction, text, then `N` masked frames — and its
   length is fixed before a single sample exists.
2. **Each step runs the whole sequence**, scores every `(codebook, position)`
   pair by confidence, and unmasks the `k` most confident, where `k` comes from
   a warped timestep schedule. After `--steps` steps everything is decided.
3. **Classifier-free guidance means two passes per step**: one conditioned on
   the text, one on the masked frames alone. `log_probs = c + g * (c - u)`.
4. **A layer penalty makes decoding coarse to fine.** Codebook 0 is unpenalised
   and is decided first; the later codebooks condition on what it chose.
5. **The codec reconstructs the waveform** from all eight codebooks at one
   rate, upsampling by 8, 5, 4, 2 and 3 — exactly 960 samples per frame, so
   25 Hz at 24 kHz.

### Why there is no KV cache

Unmasking a position changes the hidden state of every position that attends to
it, and under a bidirectional mask that is all of them. Nothing computed at
step *n* is still valid at step *n+1*, so there is nothing to cache.

The same reasoning forces bidirectional attention: a masked position has to see
the positions *after* it, or the first step would have nothing to condition on.
Every other attention path in this project is causal, so this one needed its own
kernel, `bidirectional_gqa_attention`.

The head is likewise not the usual one. Eight codebooks are predicted at once by
a single `[1024 -> 8200]` matmul reshaped to `[T, 8, 1025]` — the extra entry
per codebook is its mask token. A position's *input* embedding is the sum across
all eight codebooks, so a fully masked position still has a well-defined
embedding: the sum of the eight mask rows.

### `--steps` is the quality/speed dial

Cost is `steps * 2` full-sequence forward passes and nothing else, so it is
almost exactly linear. Measured on an M3 Pro (18 GB, CPU build), synthesising
4 s of Italian:

| `--steps` | RTF, CPU | RTF, Metal | |
|---|---|---|---|
| 8 | 1.30 | 0.38 | |
| 12 | 1.96 | **0.54** | the default |
| 16 | 2.46 | 0.69 | |
| 32 | 5.17 | 1.31 | the reference's |

On the GPU it synthesises faster than the audio plays at every setting up to
16. A reference clip adds its own frames to every forward pass, so cloning
costs more: a 4 s reference roughly doubles the sequence and takes `--steps 12`
from 0.54 to 1.06.

The default is 12 rather than the reference's 32, which is three times faster
and was judged indistinguishable by ear on Italian. That is a listening call
and could not have been anything else: the waveform statistics look like speech
across the whole range and only stop at the extreme, where a single step
produces something the check flags and warns about. `--steps 32` restores the
reference's setting.

### The Metal path

```bash
cmake -S . -B build-metal -DRT_METAL=ON && cmake --build build-metal -j
./build-metal/rich-triplet --model omnivoice --prompt "..." --language Italian
```

This is the first Metal engine here for a text-to-speech model and the first
for a *prefill* shape rather than a decode one. The Gemma and Qwen engines
exist because a decode step is a chain of GEMVs, memory-bound and too small to
dispatch one at a time. This one exists for the opposite reason: OmniVoice has
no KV cache, so every step is a full-sequence pass over a few hundred
positions — large GEMMs, compute-bound, exactly what the matrix units are for.

**It had to clear a high bar.** Accelerate's sgemm reaches about 1.0 TFLOP/s on
an M3 Pro for these shapes. The tiled matmul in `metal_ops.mm` manages 250
GF/s, so routing the GEMMs through it would have been a 4× regression. What
clears the bar is `simdgroup_matrix`: a 64×64 tile per threadgroup built from
sixteen 8×8 accumulators, which reaches 1.2–2.3 TFLOP/s depending on shape.
Every 8×8 operand loaded from memory feeds four multiply-accumulates instead of
one, and that register blocking is the whole difference.

The GEMMs are only 40% of the CPU runtime, though, so moving them alone would
have capped the win near 1.3×. Profiling said where the rest goes, and all of
it is work the GPU does not have to do:

| | Share of CPU runtime | On the GPU |
|---|---|---|
| sgemm | 40% | the matrix units |
| BF16 dequantization | 12% | **gone** — `bfloat` is an operand type |
| `bzero` in `Mat::zeros` | 8% | **gone** — scratch is allocated once |
| attention, `expf`, norms, SiLU, RoPE | ~33% | kernels |

The BF16 line is the one worth naming. The CPU path dequantizes every weight
into an f32 scratch buffer on *every call* — 437 million conversions per
forward pass, 28 billion for a four-second clip. A Metal kernel takes `bfloat`
straight as a matrix operand, so the checkpoint's own storage format is read
with no conversion pass at all.

Measured on an M3 Pro, four seconds of Italian at the default 12 steps:

| | CPU | Metal |
|---|---|---|
| Generate | 7.53 s | **1.88 s** |
| Realtime factor | 1.96 | **0.54** |
| Resident | 2.0 GB | **1.8 GB** |

Four times faster, and *less* memory: the CPU weights are freed as they are
uploaded, so nothing holds both copies. The remaining 0.26 s is the codec
decode, which is still on the CPU and is now a fifth of the total.

### Trusting a second implementation of the same model

A GPU engine is a rewrite of the forward pass, and a wrong one produces audio
that sounds plausible — the same trap as every other part of this pipeline.
What makes it checkable is that the CPU path is still there.

The test loads the checkpoint twice, uploads one copy, and compares. Logits
agree to a relative 1e-5, which is f32 epsilon — BLAS chunks the weight while
the GPU accumulates 8×8 tiles, so the two sum in different orders and could not
agree further. The check that matters is not the logits but the **argmax per
(position, codebook)**, which is what the sampler actually reads: those agree
**exactly**, on every one of them.

Over a full generation the two are not bit-identical, and the reason is worth
knowing. Which positions get unmasked at each step comes from sorting
confidence scores, and scores that differ in their last bits can sort
differently. One such flip early on changes every decision after it. The
waveforms still correlate at 0.9995 — the same utterance, decided in a
marginally different order.

Sequence length is padded to a multiple of 64 so the GEMM tiles divide evenly
and the inner loop needs no bounds checks. Those padding rows are zeroed and
carry through harmlessly, because every operator except attention is
row-independent — and attention is dispatched over the true length, never the
padded one. There is a test for exactly the lengths that are not multiples of
64.

### Length has to be decided in advance

A diffusion model cannot stop early: every frame exists, masked, from the first
step. Get the length wrong and the failure is not graceful — measured on one
sentence, asking for 2 s of a 4 s phrase truncated it mid-sentence, and asking
for 10 s collapsed into 98% silence at a peak of 0.001.

So `duration.cpp` ports the reference's `RuleDurationEstimator`, which is not a
neural model but a lookup table. Every character gets a phonetic weight
relative to one Latin letter — a CJK ideograph is a whole syllable at 3.0, a
combining accent is silent at 0.0, a digit is 3.5 because "2024" is four
characters and fourteen letters' worth of speech — and the sum is scaled
against "Nice to meet you." at 25 frames.

Unicode general category is consulted *before* the script block, which is the
part that is easy to get backwards: a Devanagari vowel sign sits inside the
Devanagari block but is a combining mark, and charging it 1.8 would inflate
every Hindi estimate. Below 50 frames the result is pulled up a cube-root
curve, because the fixed costs of an utterance — onset, final lengthening, the
breath at the end — do not shrink with the text.

The port was checked exhaustively rather than by sampling: all 1 112 064
Unicode code points were classified by both implementations and compared, and
they agree everywhere. That is a one-off harness, not a test in the suite —
running it needs Python's `unicodedata`, which is the thing being replaced.

`--duration S` overrides the whole thing.

### Text longer than a breath

A diffusion model decides its length up front, and OmniVoice was not trained to
hold a voice across a monologue. Past roughly half a minute a single pass stops
sounding like speech at all — the frames are there and the model runs out of
things to put in them. Measured on 40 s of Italian asked for in one pass: RMS
0.023 against the 0.03–0.2 speech band, and a zero-crossing rate of 0.012.
Rumble, not voice.

So above `--chunk-threshold` seconds (30 by default) the text is split into
pieces of about `--chunk-seconds` (15), generated one at a time, and joined.
The same 40 s comes back at RMS 0.102 and a zero-crossing rate of 0.051 —
indistinguishable from a four-second clip.

**Splitting happens at punctuation**, never mid-sentence, because a seam in the
middle of a rising phrase is audible where a seam at a full stop is not.
Sentences merge greedily up to the target, so a chunk overruns only when a
single sentence already does. A full stop that ends a known abbreviation —
`Dr.`, `e.g.`, `No.` — is not a break, and a closing quote or bracket stays
with the sentence it closes rather than opening the next.

**The voice is held by the first chunk.** Every piece after it takes chunk one
as a reference clip, through exactly the machinery voice cloning uses — its
codes go into the prompt as decided frames and its text joins the prompt. Take
that away and each piece invents its own speaker, which is the thing that makes
naive chunking sound like a relay race. With `--ref-audio` the user's clip is
the reference for every chunk instead, and chunk one is no longer special.

**The pieces are joined with a gap, not an overlap.** They are separate
utterances rather than one signal cut in two, so cross-fading them onto each
other would sound like two people talking over one another.

Two things about that join are worth more than they look, and both are
departures from the reference, made after measuring what it produces and then
**confirmed by ear against the reference's own behaviour** — the same standard
the RoPE pairing was settled by, and the only one that can settle a question
about how something sounds.

*Each piece is trimmed to what it actually says.* A chunk's length comes from a
duration estimate, so it ends with however much silence the estimate overshot
by and begins with however much the model took to start. Measured across four
joins of one clip, the pause came out at 113, 127, 191 and **649** ms for the
same intended gap — a hole in the middle of a paragraph. Trimming first makes
every join exactly `--chunk-gap`.

The trim measures RMS over 10 ms windows, not a per-sample peak. That is not a
detail: a chunk's quiet head crosses -50 dBFS on the odd sample while averaging
far below it, so a peak test keeps everything from the first crossing and
leaves half a second of near-silence in the join. A genuinely loud transient is
still kept — that is content.

*The fade is 8 ms, not 100.* The reference ramps a tenth of a second to nothing
at every chunk edge. When the duration estimate is tight — and it usually is —
that ramp lands on speech. Measured on the last 100 ms before four joins, the
level fell 0.176 → 0.011 across the fade: the final syllable of every chunk was
being faded out. At 8 ms the level holds to the edge and the fade only does
what it is for, which is stopping a click.

Measured on an M3 Pro, Metal build:

| text | chunks | audio | wall clock | RMS |
|---|---|---|---|---|
| 683 chars | 3 | 40.6 s | 47 s | 0.102 |
| 1024 chars | 5 | 61.8 s | 64 s | 0.108 |

Roughly linear, and roughly realtime. Nothing caps the total length: 40 000
characters is about 190 chunks and 47 minutes of audio, at 47 minutes of work.

What *is* capped is a single chunk. The GPU engine holds one query's attention
scores in threadgroup memory, which limits a sequence to 2048 positions — and
each chunk after the first carries the reference's frames on top of its own, so
`--chunk-seconds` much past 25 will exceed it. The error says so and names the
number.

### Voice cloning

```bash
./build/rich-triplet \
  --model omnivoice \
  --ref-audio giulia.wav \
  --ref-text "Ciao, mi chiamo Giulia. Oggi è una bella giornata a Roma." \
  --prompt "Domani andrò al mercato con mia sorella." \
  --language Italian \
  --out clone.wav
```

A reference clip is not a second mode. It is a prefix:

```
<|denoise|><|lang_start|>...<|instruct_end|>
<|text_start|>{ref_text} {text}<|text_end|>
[ reference frames, decided ][ target frames, masked ]
```

The clip is encoded to codes and those codes go into the sequence as
*already-decided* audio positions, with its transcript joined to the prompt.
So the model is continuing a recording it can see rather than imitating one it
cannot, and every masked position attends to the reference through the same
bidirectional attention it uses for everything else. The unconditional branch
is unchanged — still the masked frames alone — which is what makes guidance
push *towards* the reference voice.

The unmasking loop needed no change: it finds the target frames by counting
back from the end of the sequence, so anything before them is transparent to
it.

`--ref-text` is required. Without it the model cannot tell which part of the
text it has already heard, and would try to say the whole thing again. 3–10 s
of reference is the useful range; the CLI warns past 20.

Two details are the reference implementation's and are not obvious:

**`<|denoise|>` leads the prompt** whenever there is a recording, and only
then. It asks the model to clean the reference up rather than reproduce the
room it was recorded in.

**A quiet clip is levelled before encoding** — brought up to 0.1 RMS — and the
original loudness restored on the way out. The codec was fit on speech at a
particular level, and a quiet recording otherwise encodes into a part of the
codebook space that carries a quiet voice rather than that voice quietly.

Duration estimation switches reference too: with a clip, the estimator
calibrates on *this speaker's* rate rather than on the built-in phrase, which
is strictly better when one is available.

### Reading audio in

Cloning is the first thing here that needs analysis rather than synthesis, and
it pulled in three pieces the project did not have.

**A WAV reader.** `wav.cpp` could write a file and not open one. It now takes
8, 16, 24 and 32-bit PCM plus 32 and 64-bit float and WAVE_FORMAT_EXTENSIBLE,
and walks the chunk list rather than assuming the 44-byte header it writes —
almost nothing else writes that, and `LIST`/`INFO` metadata between `fmt ` and
`data` would otherwise be read as samples.

**A resampler.** The codec's two analysis paths run at different rates, so
something has to convert 24 kHz to 16 kHz. It matters that it is *that* filter
and not merely a good one: the features that come out feed a quantizer, so a
different transition band puts the latents somewhere the codebooks were never
fit. `resample.cpp` is a port of `torchaudio.functional.resample` at its
defaults, in polyphase form — 24 kHz to 16 kHz is 3 to 2, so two filter phases
of 23 taps and one pass over the input.

**HuBERT.** The codec quantizes the concatenation of an *acoustic* path and a
*semantic* one, so that a code carries what was said and not only how it
sounded. The semantic path is a 94 M-parameter HuBERT, which is the eighth
transformer here and the first that is not a language model:

| | Every other model here | HuBERT |
|---|---|---|
| Input | token ids | a raw waveform |
| Position | RoPE | a grouped convolution, added once |
| Norm placement | before each sublayer | **after each residual add** |
| Norm | RMSNorm | **LayerNorm**, mean subtraction included |
| GELU | tanh approximation | **the erf form** |
| What is used | the last hidden state | **the mean of all thirteen** |

Seven strided convolutions with no padding anywhere reduce 16 kHz audio by
exactly 320, to 50 Hz; the codec keeps every other frame to reach its own 25.
That the two paths arrive at the same frame rate by different arithmetic — 24
kHz over 960 against 16 kHz over 320 and then halved — is the invariant the
whole analysis half rests on, and both ends assert it.

### Checking an encoder with no reference implementation

The decoder could be checked by ear. The encoder cannot: its output is a
thousand integers.

What makes it checkable is that the two halves invert each other. Decode a set
of codes, encode the waveform back, and compare. On real speech codebook 0
recovers **90%** of its indices, falling to 47% by codebook 7 — which is
exactly the shape residual quantization should give, because each codebook only
ever sees the error the ones before it could not represent, and by the eighth
that error is nearly noise. Chance is one in 1024. The waveform correlates at
0.92 with the original and its statistics match to three decimals.

Nothing subtly wrong survives that. A normalisation applied on the wrong axis,
the two paths concatenated in the wrong order, a padding off by one — all of
them land at chance, not near it.

### Diagnosing it

The same rule as Orpheus applies, and harder: every fault sounds identical.
`wave_stats` prints on every run and the exact length invariants hold here too —
each upsampling block multiplies its input length by exactly its stride, and
`frames * 960` has no slack.

One trap is worth naming because no automated check catches it. RoPE can pair
dimension `i` with `i + head_dim/2` (half-split) or `2i` with `2i+1`
(interleaved), and the right answer depends on whether the GGUF converter
permuted the Q and K weight rows — llama.cpp permutes for the `llama`
architecture and not for Qwen3. So Orpheus needs interleaved and OmniVoice needs
half-split, off the same file format.

**Half-split here is confirmed by listening, not by measurement.** Both settings
produce finite, in-range, speech-shaped output that passes every check in the
suite, because the two conventions rotate by the same angles and differ only in
which pairs receive them. Half-split is intelligible Italian; interleaved is
not. The only numeric hint was interleaved's DC offset of -0.03 against
half-split's -0.0001, with a zero-crossing rate of 0.009 — rumble rather than
voice — and that is too weak to have trusted alone. `--rope-interleaved` stays
as the first thing to try when a *new* checkpoint sounds wrong.

The model and its codec are `k2-fsa/OmniVoice` (Apache 2.0, Xiaomi Corp.),
re-uploaded as GGUF by a third party; the backbone is Qwen3-0.6B, also
Apache 2.0.

---

## FLUX text to image

A 12B rectified-flow transformer, its two text encoders and a 16-channel
autoencoder. Text in, PNG out.

```bash
./build-metal/rich-triplet \
  --model flux-schnell \
  --prompt "a photograph of a harbour at dawn, long exposure" \
  --weights models/flux1-schnell-Q4_K_M.gguf \
  --t5 models/t5-v1_1-xxl-encoder-Q4_K.gguf \
  --t5-tokenizer models/spiece.model \
  --clip models/clip_l.safetensors \
  --clip-tokenizer models/clip_tokenizer.json \
  --vae models/ae.safetensors \
  --width 1024 --height 1024 --steps 4 --seed 42 \
  --out harbour.png
```

### What it is

FLUX is not a UNet. It is 19 **double-stream** blocks, where image and text are
separate residual streams with separate weights but a single joint attention
over the concatenation of both, followed by 38 **single-stream** blocks over
that concatenation — with attention and MLP computed in parallel from the same
modulated input and joined before one output projection, rather than run in
sequence.

There is no cross-attention anywhere. The prompt enters as tokens in the text
stream; the timestep and the pooled CLIP vector enter through adaptive layer
norm, which is why every LayerNorm in the model is affine-free. Position is
three-axis RoPE over `(t, h, w)` with per-axis head dimensions 16, 56 and 56 —
image tokens carry their patch-grid coordinates and text tokens carry zeros,
which makes their rotation the identity.

schnell is distilled to four steps **and** guidance-distilled, so there is no
classifier-free guidance and one forward pass per step.

### Memory, on an 18 GB M3 Pro

```
FLUX transformer   11.9B   Q4_K_M    ~6.7 GB
T5-XXL encoder      4.7B   Q4_K      ~2.8 GB   released after encoding
CLIP-L text          123M  F16       ~0.25 GB
VAE decoder           84M  F32       ~0.34 GB
```

The two text encoders run first and are released before the transformer loads;
holding T5 and the transformer at once is the difference between fitting and
not. The autoencoder's last level runs 128 channels at full resolution — half a
gigabyte per activation at 1024x1024 — so `--tile` decodes in overlapping tiles
and blends the seams.

### The Metal engine

Weights stay quantized at rest and are widened per use: before each GEMM, one
dispatch dequantizes that weight into a shared bfloat scratch buffer. This
sounds wasteful and is not — the largest weight is a single-stream block's
fused `linear1` at 3072 → 21504, which is 66 M elements to widen against
575 GFLOP of GEMM to follow, and the scratch buffer is 132 MB reused by every
projection in the model. Widening all 12B at rest would need 24 GB.

The attention kernel is a streaming online softmax rather than the
score-row-in-threadgroup-memory approach the OmniVoice engine uses, which caps
out at 2048 keys; FLUX runs 4352 at 1024x1024.

GPU and CPU agree to **0.5–0.7% RMS-relative**, flat in the number of blocks —
the flatness being what identifies the residual as bfloat rounding in the
matrix unit rather than a structural disagreement, which would compound with
depth.

`--cpu` runs the transformer on the CPU instead, which is correct and slow.

### Weights

| Component | Source |
|---|---|
| Transformer | `city96/FLUX.1-schnell-gguf` (Q4_K_M) |
| T5-XXL encoder | `city96/t5-v1_1-xxl-encoder-gguf` (Q4_K) |
| CLIP-L | `comfyanonymous/flux_text_encoders` |
| VAE, tokenizers | `black-forest-labs/FLUX.1-schnell` |

FLUX.1-schnell is Apache 2.0. `--model flux-dev` runs the dev config — same
architecture plus a distilled-guidance embedding, 28 steps, non-commercial
licence.

### Status

**This has not been run against the real checkpoints.** All 112 tests across the
image stack build their weights synthetically: they verify that each piece
agrees with its own reference implementation and that the GPU path agrees with
the CPU path. They do not verify that the tensor names in the GGUF are the ones
the loader asks for, or that the result is a picture.
[docs/text-to-image.md](docs/text-to-image.md) has the full design, the build
log, and the list of what remains unverified.

---

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--prompt TEXT` | — | Text to complete |
| `--model NAME` | — | `gpt-oss`, `gemma3-1b`, `gemma3-4b`, `qwen35-0.8b`, `qwen35-4b`, `qwen35-9b`, `orpheus-3b`, `omnivoice`, `flux-schnell`, `flux-dev` |
| `--weights PATH` | — | GGUF file or safetensors directory |
| `--tokenizer-dir DIR` | — | Directory containing `tokenizer.json` |
| `--vocab PATH` / `--merges PATH` | — | BPE files, for GPT-OSS |
| `--max-new N` | 200 | Tokens to generate |
| `--temp T` | 0.8 | Sampling temperature (0 = greedy) |
| `--top-k K` | 40 | Top-K cutoff (0 = disabled) |
| `--top-p P` | 0.95 | Nucleus probability (1.0 = disabled) |
| `--rep-penalty R` | 1.1 | Repetition penalty (1.0 = disabled) |
| `--seed S` | 42 | RNG seed |
| `--draft-len N` | 0 | Speculative draft tokens per step (Metal only) |
| `--quantize` | off | Quantize weights to Q4 after loading |
| `--debug` | off | Per-step diagnostics: `h_rms`, logit gaps, top-5 tokens |
| `--train-steps N` | 200 | Training steps when no weights are given |
| `--checkpoint PATH` | — | Load a saved checkpoint instead of training |
| `--pretokenize S D` | — | Tokenize text file `S` into binary `D`, then exit |
| `--benchmark` | off | Scalar autograd vs tensor autodiff benchmark |
| `--voice NAME` | `tara` | Orpheus speaker |
| `--snac PATH` | `models/snac_24khz.bin` | Codec checkpoint — SNAC for Orpheus, the tokenizer GGUF for OmniVoice |
| `--out PATH` | `out.wav` | Where to write the synthesised audio |
| `--no-audio-mask` | off | Let Orpheus sample outside the audio token range |
| `--no-leading-bos` | off | Drop the leading BOS from the Orpheus prompt |
| `--language NAME` | `None` | OmniVoice language hint, e.g. `Italian` |
| `--instruct TEXT` | `None` | OmniVoice voice description, e.g. `a calm young woman` |
| `--duration S` | 0 | OmniVoice audio seconds; 0 estimates from the text |
| `--steps N` | 12 | OmniVoice unmasking steps — the quality/speed dial |
| `--chunk-seconds S` | 15 | Audio per chunk when splitting long text; 0 never splits |
| `--chunk-threshold S` | 30 | Split only when the estimate exceeds this |
| `--chunk-gap S` | 0.3 | Pause between chunks, in seconds |
| `--guidance G` | 2.0 | OmniVoice classifier-free guidance; 0 halves the work |
| `--ref-audio PATH` | — | WAV of a voice for OmniVoice to clone |
| `--ref-text TEXT` | — | What that WAV says; required alongside it |
| `--rope-interleaved` | off | Use interleaved RoPE pairing (debugging only) |

---

## Running tests

```bash
./build/tests/rt_tests          # 623 cases
./build-metal/tests/rt_tests    # 642 cases, including the GPU kernels
```

Covers matrix ops, gradient correctness against finite differences, attention
shapes, Flash Attention, Q4_K/BF16 quantization, GGUF, safetensors and
torch-pickle parsing, all four tokenizers, every architecture, 1-D convolution
against reference loops, both codec decoders and the OmniVoice encoder, RIFF in
both directions, resampling against the reference filter, the duration
estimator's character classes, the Metal kernels, and the weight-loading paths.

The image stack adds 112: 2-D convolution and GroupNorm against index-by-index
references, the autoencoder's block algebra and tile blending, PNG containers
checked chunk by chunk with a stored-block inflater, the T5 relative-position
bucketing and its unscaled attention, CLIP's causal mask and argmax pooling,
FLUX's patch order and three-axis RoPE, and the Metal engine against the CPU
forward pass.

A further 32 cases are hidden by default because they need downloaded weights
(36 on the Metal build, which also checks the GPU engine against the CPU one):

```bash
./build/tests/rt_tests '[.integration]'   # real checkpoints, seconds
./build/tests/rt_tests '[.e2e]'           # loads 2.4 GB and generates, minutes
```

They skip cleanly when `models/` is empty, and they *discover* whichever
Orpheus checkpoint is present rather than naming one, taking their prompt from
the `general.languages` it declares — an English sentence in an Italian voice
would test very little. The `[.e2e]` set includes a
consistency check worth calling out: running N tokens through prefill must rank
its logits identically to running N-1 through prefill and the last through the
incremental decode path. It needs no reference implementation, and it separates
a KV-cache or RoPE-offset bug from an architecture one — which otherwise
present the same way.

The OmniVoice half is checked mostly *without* weights, because its invariants
are arithmetic: each decoder block multiplies its input length by exactly its
stride, the full chain by exactly 960, the unmask schedule accounts for every
`(codebook, position)` pair with none left masked, and the duration estimator
puts each script, category and boundary code point in the class the reference
does. The cloning prompt is checked the same way, against a twenty-word
tokenizer built inside the test, since what is under test is the layout and not
the merges. What needs the checkpoint is the codec round trip, the Metal
engine's agreement with the CPU one, and that everything loads and is
deterministic.

---

## Training from scratch

```bash
./build/rich-triplet --prompt "Once upon a time" --train-steps 2000
```

Trains a small GPT-2 on the built-in bilingual corpus, prints train and validation loss, then streams a completion.

```bash
./build/rich-triplet --benchmark
```

Runs the same tiny model through both autodiff engines and prints the speedup. The scalar engine allocates one node per number; the tensor engine allocates one per matrix operation, so a step goes from hundreds of thousands of backward visits to a few dozen matmuls.

---

## Porting notes

The port is intended to be behaviourally identical to the Rust original, which needed more care than a mechanical translation. Each module was checked with a parity harness: a temporary test appended to the Rust source printed FNV-1a hashes of its outputs, an equivalent C++ snippet printed the same hashes, and the two were diffed. Every stage came out bit-exact.

The end-to-end paths were checked the same way against the built binaries. `--help` output, `--pretokenize` output bytes, the trained-GPT-2 generation run (stdout and stderr) and the benchmark output are byte-identical apart from the language name in the banner and wall-clock timings.

Semantic mismatches the harness caught, each of which would have passed a shape-and-finiteness test:

- **`f32 as u8` saturates in Rust** (NaN maps to 0, out-of-range clamps) but is undefined behaviour in C++. Q4_K's `quantize_block` produces negative candidates, so this needed an explicit `sat_u8`.
- **`Q4KMat::f32_to_f16` truncates** the mantissa despite a doc comment claiming round-to-nearest-even. Every Q4_K block header passes through it, so "fixing" the rounding would shift every dequantized weight.
- **Member initialization order is declaration order in C++**, not initializer-list order. Three classes drew RNG values in the wrong sequence until the members were reordered.
- **Float association matters**: `x * sigmoid(x)` is not `x / (1 + exp(-x))` in the last ulp, and summing squared gradients per parameter is not the same as one flat running sum.
- **`Objective-C` reference parameters default to `__autoreleasing`** and will not bind to `__strong` struct members; the Metal helpers need `__strong id<MTLBuffer>&`.

---

## Project structure

```
include/rt/             Public headers, one per module
src/
├── mat.cpp             Mat / MatBf16, matmul, NEON GEMV, chunked sgemm
├── quant_q4.cpp        Q4_0 and Q8 quantization
├── quant_q4k.cpp       Q4_K blocks, SDOT GEMV, dequantization
├── tensor.cpp          Basic tensor math
├── tokenizer.cpp       Char, BPE, SentencePiece and HuggingFace BPE tokenizers
├── utf8.cpp            UTF-8 decoding for the tokenizers
├── dataset.cpp         Sliding-window dataset, train/val split, binary corpora
│
├── autograd.cpp        Scalar automatic differentiation (Value nodes)
├── tensor_node.cpp     Tensor autodiff — Mat-level VJPs, 20 ops, checkpoints
├── nn.cpp / nn2.cpp    Layers (Linear, RMSNorm, SwiGLU, LayerNorm)
│
├── transformer.cpp     Scalar GPT model
├── transformer2.cpp    GPT-2 architecture, trains end to end
├── transformer3.cpp    GPT-OSS: RoPE, GQA, MoE, safetensors
├── transformer4.cpp    Gemma 3 architecture
├── transformer4_load.cpp        Gemma 3 weight loading, cache, generation
├── transformer_qwen35.cpp       Qwen 3.5 architecture
├── transformer_qwen35_load.cpp  Qwen 3.5 weight loading and generation
├── transformer5.cpp             Llama 3.2 architecture, RoPE, large-vocab sampling
├── transformer5_load.cpp        Llama 3.2 GGUF loading, embedded tokenizer
├── transformer6.cpp             Bidirectional Qwen3, multi-codebook head
├── transformer6_load.cpp        OmniVoice GGUF loading
├── hubert.cpp                   HuBERT: waveform in, semantic features out
│
├── conv1d.cpp          1-D convolution: dilated, grouped, transposed; Snake, im2col
├── snac.cpp            SNAC 24 kHz codec decoder and residual vector quantizer
├── orpheus.cpp         Audio-token protocol, frame de-interleaving, synthesis
├── omnivoice.cpp       Masked-diffusion sampler: schedules, guidance, unmasking
├── omnivoice_codec.cpp OmniVoice codec, both directions — 8 codebooks, 960x
├── duration.cpp        Rule-based duration estimation from character weights
├── resample.cpp        Polyphase windowed-sinc sample rate conversion
├── torch_pickle.cpp    PyTorch .bin reader: ZIP container + pickle manifest
├── wav.cpp             RIFF reading and writing, and waveform statistics
│
├── conv2d.cpp          2-D convolution, nearest upsampling, GroupNorm
├── vae.cpp             16-channel AutoencoderKL decoder, whole and tiled
├── qlinear.cpp         Inference-only linear over f32 / BF16 / Q4_K weights
├── t5.cpp              T5 v1.1 encoder: relative position bias, gated GELU
├── clip_text.cpp       CLIP-L text tower and its pooled output
├── flux.cpp            FLUX MMDiT, three-axis RoPE, rectified-flow sampler
├── png.cpp             PNG writing with stored DEFLATE blocks, image statistics
│
├── gguf.cpp            GGUF parser and every quantized tensor decoder
├── metal_ops.mm        Metal GPU kernels (tiled matmul, per-dispatch GEMV)
├── metal_decode.mm     Full-graph Metal decode for Gemma 3
├── metal_decode_qwen35.mm       Full-graph Metal decode for Qwen 3.5
├── metal_omnivoice.mm           Full-graph Metal forward pass for OmniVoice
├── metal_flux.mm                Full-graph Metal forward pass for FLUX
├── shaders/*.msl       The MSL kernel sources, embedded at build time
│
├── train.cpp           Scalar AdamW + generation
├── train2.cpp          Tensor AdamW + streaming generation
└── main.cpp            CLI entry point
tests/                  Catch2 suite, one file per module
```

---

## Key concepts implemented

### Automatic differentiation

```
Forward:  loss = f(weights)   — build the computation graph
Backward: d(loss)/d(weights)  — walk it in reverse, applying the chain rule
```

The scalar engine (`autograd.cpp`) creates one node per number; the tensor engine (`tensor_node.cpp`) creates one per matrix operation. Both topologically sort and traverse in reverse. Backward closures capture their own node, so the graph holds reference cycles — `free_graph()` breaks them iteratively when a graph is no longer needed.

### Flash Attention

Standard attention materializes a `T x T` score matrix. Flash Attention tiles Q in blocks of 64 and accumulates the output with an online softmax, using O(T) memory. The backward pass recomputes the softmax weights from the stored `(l, m)` vectors.

### Quantization

- **Q4_K**: 4-bit with two levels of scale (super-block and sub-block), 256 elements in 144 bytes, dequantized during the GEMV through ARM's SDOT instruction.
- **Q4_0 to Q4_K repacking** at load time, so there is one quantized path to optimize rather than two.
- **INT4 dynamic quantization** for GPT-OSS weights, roughly a 4x memory reduction.

### RoPE

Gemma 3 uses the NeoX half-split pattern — `i` pairs with `i + head_dim/2`, not with `i + 1`:

```
angle       = (pos * freq_scale) / theta^(2i / d_head)
x'_i        = x_i * cos(angle) - x_{i+half} * sin(angle)
x'_{i+half} = x_{i+half} * cos(angle) + x_i * sin(angle)
```

Local layers use theta = 10000 and freq_scale = 1.0; global layers (every sixth) use theta = 1e6 and freq_scale = 1/8. Qwen 3.5 rotates only the first quarter of each head.

### Two RoPE pair conventions

Rotary embeddings can pair dimension `i` with `i + head_dim/2` (half-split) or
`2i` with `2i + 1` (interleaved). HuggingFace's Llama uses half-split;
llama.cpp uses interleaved and reconciles the two by permuting the Q and K
weight rows during conversion. So Gemma 3, Llama 3.2 and Qwen3 need *different*
conventions off the same file format — `convert_hf_to_gguf.py` permutes for
llama and not for the other two. Orpheus is a Llama, so it takes interleaved;
OmniVoice is a Qwen3, so it takes half-split.

Both conventions rotate by the same angles and differ only in which pairs those
angles apply to, so at low positions — where every rotation is near identity —
they agree to several decimals. They separate as position grows. In an
autoregressive speech model that means the first three or four tokens come out
right and everything after is noise. In a diffusion one there is no such tell:
the whole sequence is wrong at once, and the waveform statistics stay
speech-shaped either way. Only listening separates them.

### Llama 3 RoPE scaling

Llama 3.2 does not scale RoPE by one factor. It divides each frequency band by
a different amount: high-frequency dimensions untouched so local structure
survives, low-frequency ones divided by 32 so positions stretch, with a smooth
ramp between. GGUF ships the resulting per-dimension values in
`rope_freqs.weight`, running from 1.0 up to 32.0 — they are **divisors**, not
multipliers.

### Residual vector quantization, in both directions

Reconstruction from codes is a sum, and the codebooks can be read in any order.
Going the other way they cannot: quantization is greedy and sequential.

```
residual = latents
for each codebook:
    code     = nearest entry to project_in(residual)
    residual = residual - project_out(codebook[code])
```

So codebook `i` only ever sees the error codebooks `0..i-1` could not
represent. That is why the coarse ones are robust and the fine ones are nearly
coding noise — measured over a decode-and-re-encode cycle, codebook 0 recovers
90% of its indices and codebook 7 recovers 47% — and it is also why OmniVoice's
diffusion loop penalises the later codebooks and decides them last. The order
is not a heuristic; it is the structure of the code.

The nearest-neighbour search does not need distances. `||x||^2` is the same for
every candidate, so `argmin ||x - e||^2` is `argmin(||e||^2 - 2 x·e)`, which is
one gemm against the codebook plus a precomputed norm.

### Multi-scale residual vector quantization

A plain codec quantizes every frame at one rate. SNAC runs three codebooks at
1/4, 1/2 and 1/1 of the frame rate, each coding what the previous one left
behind, so coarse structure gets cheap slow codes and detail gets fast ones —
7 codes for 4 frames:

```
z = sum_i out_proj_i(codebook_i[code_i]) repeated by stride_i
```

The upsampling is a repeat, not a tile: stride 4 turns `[a, b]` into
`[a, a, a, a, b, b, b, b]`.

### Weight normalization, and which axis

`weight_norm` stores a magnitude `g` and a direction `v`, reconstructing
`g * v / ||v||` with the norm over every axis except axis 0 of the *stored*
tensor. Which axis that is depends on the layer: `Conv1d` stores
`[out, in, k]`, so the magnitude is per output channel, while
`ConvTranspose1d` stores `[in, out, k]`, so it is per **input** channel.
Deriving the group count from `g`'s own length gets both right with no special
case.

### Quantization is a set of choices, not a single knob

A Q4_K_M checkpoint is not uniformly Q4_K. The quantizer keeps `attn_v`,
`ffn_down` and the output projection at Q6_K because those carry most of the
quality, and reads back as a mix. A Q8_0 checkpoint is uniform, so a loader
that widens anything non-native to BF16 lands at 7.57 GB for the same model
that fits in 2.8 GB requantized.

So "should I requantize?" has no single answer: yes for the uniform file,
no for the mixed one, where it would discard exactly the tensors the format
went out of its way to protect.

### Grouped Multi-Query Attention

Gemma 3 4B has 8 query heads and 4 KV heads, each KV head shared by two query heads, halving the KV cache.

### Mixture of Experts

GPT-OSS routes each token to 4 of 32 experts: eight times the parameters at the same compute per token.

### Gated DeltaNet

Qwen 3.5's linear-attention layers carry a per-head state matrix instead of a growing KV cache:

```
S   <- S * exp(g)
S   <- S + outer(k, (v - S^T k) * beta)
out = S^T q
```

so decode is O(1) in sequence length rather than O(T).

### Two shapes of GPU work

A decode step and a diffusion step want opposite things from a GPU, and this
project now has an engine for each.

An autoregressive decode is one token wide: every weight is read once and
multiplied by a single vector. That is memory-bound, the arithmetic is trivial,
and the enemy is dispatch overhead — hence `metal_decode.mm`, which encodes
~717 GEMVs into one command buffer so the GPU is never idle between them.

A masked diffusion step is the whole sequence wide, because there is no cache
to shorten it. Every weight is read once and multiplied by a few hundred
vectors, which is compute-bound and hands the matrix units exactly the shape
they want. There the enemy is the kernel itself: a scalar tiled matmul loses to
Accelerate by 4×, and only register blocking with `simdgroup_matrix` wins.

Same hardware, same model family, opposite bottleneck.

### Masked diffusion decoding

An autoregressive model spends its compute on one new position at a time and
caches the rest. A masked diffusion model does the opposite: every position
exists from the start, all of them masked, and each step re-runs the entire
sequence to decide which few to commit.

```
for step in 0..num_step:
    c = model(text + frames)          # conditional
    u = model(frames)                 # unconditional
    log_probs = log_softmax(c + g * (c - u))
    score     = max(log_probs) - codebook_index * layer_penalty
    unmask the top k of score, ignoring what is already decided
```

That trades a memory-bound GEMV per token for a compute-bound matmul over the
whole sequence — the shape BLAS is good at — but it forfeits the KV cache
entirely, because unmasking one position invalidates every hidden state that
attends to it. Two passes per step is the price of classifier-free guidance,
and the layer penalty is what makes decoding coarse to fine: codebook 0 is
unpenalised, so it commits first and the rest condition on it.

### How long is this text, in frames?

A diffusion model has to know its output length before it generates anything,
so OmniVoice ships a rule-based estimator: a phonetic weight per character,
summed and scaled against a reference phrase. The weights say something real
about writing systems — a CJK ideograph is a syllable (3.0), a Hangul block is
close (2.5), an abugida consonant carries its vowel (1.8), a Latin letter is
the 1.0 baseline, a combining mark is silent (0.0), and a digit is 3.5 because
it is read as a word.

The ordering is the subtle part. Unicode *category* is checked before Unicode
*block*, so a Devanagari vowel sign is a mark rather than an Indic letter and a
full stop inside a CJK run is a pause rather than an ideograph. Getting that
backwards costs nothing visible and inflates every estimate for half the
world's scripts.

---

## Stats

- ~39,000 lines of C++, Objective-C++ and MSL, plus ~15,000 of tests
- 642 test cases with Metal, 623 without, plus 32 that need downloaded weights
  (36 with Metal)
- Zero ML dependencies (Accelerate and Metal are system frameworks)
- Every published weight format read from scratch: GGUF, safetensors, and
  PyTorch's ZIP-plus-pickle `.bin`
- The ported modules are bit-exact against the Rust reference, module by
  module; both text-to-speech stacks and the text-to-image stack have no Rust
  counterpart and are new here
