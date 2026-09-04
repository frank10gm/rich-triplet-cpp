# Rich Triplet — LLM from scratch in C++

A complete LLM stack built from first principles in C++23, **with no ML dependencies**. Every component — matrix math, automatic differentiation, attention, quantization, Metal GPU kernels, tokenizer — is written from scratch.

This is a port of [the Rust original](../rich-triplet), kept numerically identical: the parity work behind it is described in [Porting notes](#porting-notes).

Supports **Gemma 3** and **Qwen 3.5** inference on Apple Silicon with Q4_K_M / Q4_0 GGUF weights and full-graph Metal GPU decode.

---

## What this project is

Four transformer implementations (GPT-2 scalar, GPT-2 tensor, GPT-OSS, Gemma 3, Qwen 3.5), a tensor autodiff engine, Apple Metal GPU acceleration, Flash Attention, GGUF and safetensors weight loading, and a CLI for inference and training.

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

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--prompt TEXT` | — | Text to complete |
| `--model NAME` | — | `gpt-oss`, `gemma3-1b`, `gemma3-4b`, `qwen35-0.8b`, `qwen35-4b`, `qwen35-9b` |
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

---

## Running tests

```bash
./build/tests/rt_tests          # 292 cases
./build-metal/tests/rt_tests    # 304 cases, including the GPU kernels
```

Covers matrix ops, gradient correctness against finite differences, attention shapes, Flash Attention, Q4_K/BF16 quantization, GGUF and safetensors parsing, all four tokenizers, every architecture, the Metal kernels, and the weight-loading paths.

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

- ~22,700 lines of C++, Objective-C++ and MSL, plus ~6,900 of tests
- 304 test cases with Metal, 292 without
- Zero ML dependencies (Accelerate and Metal are system frameworks)
- Bit-exact against the Rust reference, module by module
