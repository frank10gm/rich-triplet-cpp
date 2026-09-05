# Rich Triplet — LLM from scratch in C++

A complete LLM stack built from first principles in C++23, **with no ML dependencies**. Every component — matrix math, automatic differentiation, attention, quantization, Metal GPU kernels, tokenizer — is written from scratch.

This is a port of [the Rust original](../rich-triplet), kept numerically identical: the parity work behind it is described in [Porting notes](#porting-notes).

Supports **Gemma 3** and **Qwen 3.5** text inference on Apple Silicon with Q4_K_M / Q4_0 GGUF weights and full-graph Metal GPU decode, plus **Orpheus** text-to-speech synthesis end to end in English, Italian and Spanish — text in, WAV out, including the SNAC neural audio codec.

---

## What this project is

Six transformer implementations (GPT-2 scalar, GPT-2 tensor, GPT-OSS, Gemma 3, Qwen 3.5, Llama 3.2), a neural audio codec decoder, a tensor autodiff engine, Apple Metal GPU acceleration, Flash Attention, GGUF / safetensors / torch-pickle weight loading, and a CLI for inference, training and speech synthesis.

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
| `conv1d.cpp` | Dilated, grouped and transposed 1-D convolution; Snake; weight norm |
| `snac.cpp` | Multi-scale residual vector quantization, and a codec decoder |
| `orpheus.cpp` | Audio tokens: slot offsets, frame de-interleaving, resynchronisation |
| `torch_pickle.cpp` | ZIP central directories, and just enough pickle to be safe |
| `wav.cpp` | RIFF, and how to tell speech from noise without listening |
| `gguf.cpp` | GGUF file format: Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, BF16, F16, F32 |
| `tokenizer.cpp` | Character, BPE, SentencePiece and HuggingFace BPE tokenizers |
| `metal_ops.mm` | Metal GPU tiled matmul, Q4_K/BF16 GEMV kernels |
| `metal_decode.mm` | Full-graph Metal decode: ~717 dispatches per token in one command buffer |
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

So a 4 s clip takes about 21 s. Decode runs on the CPU: there is no full-graph
Metal path for Llama yet, and the Metal build measures the same, because
per-token cost is Q4_K GEMV rather than the large matmuls `Mat::matmul` sends
to the GPU. Two things would close most of the gap — a `metal_decode_llama.mm`
trimmed from the Gemma engine, and speculative decoding, which is already
implemented for Gemma.

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

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--prompt TEXT` | — | Text to complete |
| `--model NAME` | — | `gpt-oss`, `gemma3-1b`, `gemma3-4b`, `qwen35-0.8b`, `qwen35-4b`, `qwen35-9b`, `orpheus-3b` |
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
| `--snac PATH` | `models/snac_24khz.bin` | SNAC 24 kHz codec checkpoint |
| `--out PATH` | `out.wav` | Where to write the synthesised audio |
| `--no-audio-mask` | off | Let Orpheus sample outside the audio token range |
| `--no-leading-bos` | off | Drop the leading BOS from the Orpheus prompt |

---

## Running tests

```bash
./build/tests/rt_tests          # 401 cases
./build-metal/tests/rt_tests    # 413 cases, including the GPU kernels
```

Covers matrix ops, gradient correctness against finite differences, attention
shapes, Flash Attention, Q4_K/BF16 quantization, GGUF, safetensors and
torch-pickle parsing, all four tokenizers, every architecture, 1-D convolution
against reference loops, the SNAC decoder, RIFF output, the Metal kernels, and
the weight-loading paths.

A further 12 cases are hidden by default because they need downloaded weights:

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
│
├── conv1d.cpp          1-D convolution: dilated, grouped, transposed; Snake
├── snac.cpp            SNAC 24 kHz codec decoder and residual vector quantizer
├── orpheus.cpp         Audio-token protocol, frame de-interleaving, synthesis
├── torch_pickle.cpp    PyTorch .bin reader: ZIP container + pickle manifest
├── wav.cpp             16-bit PCM WAV output and waveform statistics
│
├── gguf.cpp            GGUF parser and every quantized tensor decoder
├── metal_ops.mm        Metal GPU kernels (tiled matmul, per-dispatch GEMV)
├── metal_decode.mm     Full-graph Metal decode for Gemma 3
├── metal_decode_qwen35.mm       Full-graph Metal decode for Qwen 3.5
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
weight rows during conversion. So Gemma 3 and Llama 3.2 need *different*
conventions off the same file format — `convert_hf_to_gguf.py` permutes for
llama and not for Gemma.

Both conventions rotate by the same angles and differ only in which pairs those
angles apply to, so at low positions — where every rotation is near identity —
they agree to several decimals. They separate as position grows. In a speech
model that means the first three or four tokens come out right and everything
after is noise.

### Llama 3 RoPE scaling

Llama 3.2 does not scale RoPE by one factor. It divides each frequency band by
a different amount: high-frequency dimensions untouched so local structure
survives, low-frequency ones divided by 32 so positions stretch, with a smooth
ramp between. GGUF ships the resulting per-dimension values in
`rope_freqs.weight`, running from 1.0 up to 32.0 — they are **divisors**, not
multipliers.

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

---

## Stats

- ~27,100 lines of C++, Objective-C++ and MSL, plus ~9,700 of tests
- 413 test cases with Metal, 401 without, plus 12 that need downloaded weights
- Zero ML dependencies (Accelerate and Metal are system frameworks)
- Every published weight format read from scratch: GGUF, safetensors, and
  PyTorch's ZIP-plus-pickle `.bin`
- The ported modules are bit-exact against the Rust reference, module by
  module; the text-to-speech stack has no Rust counterpart and is new here
