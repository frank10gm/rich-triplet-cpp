# Text-to-image: FLUX.1-schnell

A design note in the same spirit as the rest of the project -- what you would
understand after writing it, what the machine can actually run, and which of
the plausible-looking mistakes are the expensive ones -- kept up to date as the
code landed.

## Status

Working end to end against the real checkpoints. `--model flux-schnell` takes a
prompt and writes a PNG, on the GPU or the CPU.

Everything below is implemented: `conv2d.cpp`, `png.cpp`, `vae.cpp`, `t5.cpp`,
`clip_text.cpp`, `flux.cpp`, `metal_flux.mm` and `shaders/flux.msl`, with 120
test cases across them.

Getting from "the tests pass" to "it draws a harbour" took six more bugs, all
of them in the seam between this code and the real files rather than in the
architecture. They are recorded under
[Meeting the real checkpoints](#meeting-the-real-checkpoints), because that
seam is where the next port will lose its time too.

---

## Verdict

Yes, and the shape of the work is familiar. A diffusion transformer is a
transformer with a different conditioning story, and this repo is already a
transformer factory: RoPE, RMSNorm, QK-norm, GQA attention, SwiGLU, GGUF
Q4_K/Q6_K/Q8_0 loading, safetensors, SentencePiece unigram, HuggingFace BPE,
and a Metal backend that has twice been driven to a full-graph forward pass.

The genuinely new material is two-dimensional convolution and a VAE decoder.
Everything else is a re-spelling of code that exists.

Target: **FLUX.1-schnell**, Q4_K quantized, on an M3 Pro with 18 GB of unified
memory.

---

## Why schnell and not the alternatives

| Candidate | Params | Why not |
|---|---|---|
| **FLUX.1-schnell** | 12B DiT | — chosen |
| FLUX.1-dev | 12B DiT | Same code, but 28 steps instead of 4 and a non-commercial licence |
| SD 3.5 Medium | 2.5B MMDiT | 5x less compute per step but 28-40 steps, so no net win, and it still drags T5-XXL along |
| SDXL | 2.6B UNet | Most new code of any option: resblock/skip tree, spatial transformers, up/down sampler ladder. Oldest quality. |
| SANA 1.6B | 1.6B DiT | Genuinely faster (see below) but the deep-compression autoencoder is a bigger convolution stack than FLUX's VAE, and quality is a step down |
| Qwen-Image | 20B | Does not fit |

The three properties that decide it:

1. **Four steps.** schnell is timestep-distilled, and distilled to
   `guidance_scale = 0` — so no classifier-free guidance either. One forward
   pass per step, four passes per image. A 28-step CFG model is 14x the work.
2. **It is a transformer.** 19 double-stream blocks and 38 single-stream
   blocks, all attention and MLP. No UNet zoo.
3. **Apache 2.0**, and community GGUFs already exist in exactly the quant
   formats `gguf.cpp` reads.

### The fast alternative, if 3 minutes an image is too slow

SANA's deep-compression autoencoder is 32x rather than 8x, so 1024x1024 is
**1024 tokens instead of 4096** — and its text encoder is Gemma-2, which
`transformer4.cpp` is most of the way to already. Expect 30-60 s per image
against FLUX's 3-4 minutes. It is a real option, and the conv2d work below is
shared. It is not the "high quality one".

---

## Budget on this machine

M3 Pro, 11 cores, 18 GB unified, 144 GB free disk.

### Memory

```
FLUX transformer   11.9B   Q4_K_M    ~6.7 GB
T5-XXL encoder      4.7B   Q4_K      ~2.8 GB   released after encoding
CLIP-L text          123M  F16       ~0.25 GB
VAE decoder           84M  F32       ~0.34 GB
latents + activations                ~1.5 GB   at 1024x1024, DiT only
```

Peak is about **9 GB** if T5 is unloaded before the denoise loop starts, which
it can be — the text embedding is computed once and is 256 x 4096 floats.

The VAE decoder is the surprise. Its last upsampling level runs 128 channels at
1024x1024, which is 537 MB for a single activation; a residual unit holds three
at once. Decoding a 1024x1024 image whole costs roughly 2-3 GB on top of
everything else. Tiled decode with overlap is the standard answer and should be
assumed for 1024x1024. At 512x512 it is a non-issue.

### Speed

Per denoise step at 1024x1024: 4096 image tokens + 256 text tokens, 11.9B
parameters, so `2 * 11.9e9 * 4352` ~ **104 TFLOP**. The estimate below that was
~45 s/step, from assuming 2-3 TFLOPS through the GEMM. Measured, on an M3 Pro:

| Resolution | Tokens (img + txt) | Per step | 4 steps | Total incl. VAE |
|---|---|---|---|---|
| 256x256 | 256 + 256 | 6.2 s | 25 s | 27 s |
| 512x512 | 1024 + 256 | 20.2 s | 81 s | 92 s |
| 1024x1024 | 4096 + 256 | 156 s | 624 s | 688 s |

So the estimate was low by 3.5x at full resolution, and the scaling is worse
than the parameter count predicts: from 256x256 to 1024x1024 the token count
grows 8.5x and the time 25x. That gap is attention, which is `O(T^2)` while
everything else is `O(T)` -- at 4352 tokens it stops being a rounding error and
starts being the bill.

The process sits at 13% CPU throughout, so this is GPU-bound, which is the
right place for it to be. What it is *not* is optimised: the attention kernel
streams keys in tiles of 64 with a single thread doing each tile's softmax
bookkeeping, and that is the obvious thing to fix first.

T5 encoding is 6.7 s. VAE decode at 1024x1024 is ~60 s tiled, on the CPU.

Text-to-image on this Mac is a ten-minute-per-image experience at full
resolution and a half-minute one at 256x256.

---

## What the repo already provides

| Need | Where it already lives |
|---|---|
| Q4_K / Q6_K / Q8_0 / BF16 tensor loading | `gguf.cpp` |
| safetensors reading | `transformer3.cpp` |
| T5's tokenizer (SentencePiece unigram, Viterbi) | `tokenizer.cpp` |
| CLIP's tokenizer (byte-level BPE) | `tokenizer.cpp` |
| RMSNorm, LayerNorm, SiLU, GELU, gated FFN | `nn2.hpp` |
| Multi-head attention, RoPE, QK-norm | `transformer5.cpp`, `transformer6.cpp` |
| Bidirectional (non-causal) attention | `transformer6.cpp` — the diffusion Qwen3 |
| im2col + gemm convolution, grouped and strided | `conv1d.cpp` |
| Metal tiled GEMM, per-layer weight upload, full-graph command buffer | `metal_omnivoice.mm` |

`transformer6.cpp` is worth calling out. It is already a bidirectional
transformer used as a diffusion model, with no KV cache — structurally the
closest thing in the repo to a DiT.

---

## What is missing

New files, named to match the existing scheme:

| File | Contents | Est. |
|---|---|---|
| `conv2d.cpp` / `.hpp` | 2-D convolution: dense im2col, 1x1 fast path, stride, padding, nearest upsample | ~500 |
| `groupnorm` (into `nn2.cpp`) | GroupNorm over (group channels, H, W) | ~80 |
| `vae.cpp` / `.hpp` | 16-channel AutoencoderKL decoder: conv_in, mid block with attention, 4 upsampling levels, conv_out | ~600 |
| `t5.cpp` / `.hpp` | T5-XXL encoder: 24 layers, relative position bias, gated GELU | ~700 |
| `clip_text.cpp` / `.hpp` | CLIP-L text encoder: 12 causal layers, pooled output | ~400 |
| `flux.cpp` / `.hpp` | MMDiT: patchify, 3-axis RoPE, AdaLN modulation, double- and single-stream blocks | ~1000 |
| `flow.cpp` (into `flux.cpp`) | Flow-matching Euler sampler | ~150 |
| `png.cpp` / `.hpp` | PNG writer, the counterpart of `wav.cpp` | ~250 |
| `metal_flux.mm` + `flux.msl` | Q4_K dequant-to-half, half GEMM, fused attention | ~1200 |

About 4900 lines. The OmniVoice stack that landed last week is comparable.

---

## Activation layout

`conv1d.cpp` stores activations time-major, `[T, channels]`, one row per frame,
because a `Mat` row is always one token. The two-dimensional analogue is
**spatial-major**: `[H * W, channels]`, one row per pixel, with `h` and `w`
passed alongside since `Mat` is flat.

This is not merely consistent, it is the layout that makes the rest cheap:

- A 1x1 convolution is `matmul_bt` against the weight, unchanged.
- The VAE mid-block's self-attention wants `[H * W, C]` — it is already in
  token-major form, so `transformer2.cpp`'s attention applies directly.
- DiT patchify is a row regrouping, not a transpose.
- im2col gathers `Cin * K * K` per output pixel into a row, exactly as the 1-D
  version gathers `Cin * K`.

The one place it costs something is GroupNorm, which reduces over channels
**and** space at once, so the reduction is strided rather than contiguous. That
is a handful of extra lines, paid once.

Weight layout follows the same rule as `conv1d.hpp`: the `Mat` row-major bytes
are exactly the PyTorch flat buffer, so loading is a wrap and not a repack.

| PyTorch parameter | shape | `Mat` |
|---|---|---|
| `Conv2d.weight` | [Cout, Cin, KH, KW] | [Cout, Cin * KH * KW] |
| `Conv2d.weight` (1x1) | [Cout, Cin, 1, 1] | [Cout, Cin] |

---

## Build order

Each milestone is independently verifiable, and none of them requires the next
one to exist. That matters: a wrong sign in the VAE and a wrong sign in the DiT
produce the same symptom (noise), and debugging them together is miserable.

All five landed. What each one actually cost, and what it caught, is recorded
under [Build log](#build-log).

### M0 — conv2d and GroupNorm

No weights, no downloads. Tests only, in the style of `test_conv1d.cpp`: a
reference implementation written out index by index, compared against the
im2col path on deterministic ramp-filled inputs.

**Evidence:** `rt_tests` green.

### M1 — the VAE decoder, alone

Dump one latent from diffusers (`torch.save` of a `[1, 16, 64, 64]` tensor),
decode it, write a PNG. This proves half the pixel stack with no transformer in
sight, and the output is a recognisable image or it is not — the cheapest
possible oracle.

**Evidence:** a PNG that matches the diffusers decode of the same latent to
within a few LSBs.

### M2 — the text encoders

CLIP-L first: 12 layers, small, and its pooled output is a single 768-vector
that is trivial to diff against a dumped reference. Then T5-XXL, checked the
same way against dumped `[256, 4096]` embeddings.

**Evidence:** max absolute deviation from the reference embeddings, printed.

### M3 — the DiT and the sampler, on CPU

Correctness before speed. Run at 256x256 (256 tokens) where a CPU pass is
tolerable, with a fixed seed, and compare the denoised latent against diffusers
step by step. A DiT that is subtly wrong still produces something
image-shaped, so per-step latent comparison is the only honest check.

**Evidence:** per-step latent agreement, then a 256x256 image.

### M4 — Metal

Only once M3 is right. See below.

---

## Milestone 0 in detail: conv2d and GroupNorm

### `include/rt/conv2d.hpp`

```cpp
/// Output size of one axis of a 2-D convolution.
[[nodiscard]] std::size_t conv2d_out_size(std::size_t in, std::size_t kernel,
                                          std::size_t dilation, std::size_t padding,
                                          std::size_t stride = 1);

/// Kernel-size-1 convolution: `out[p] = weight @ x[p] + bias`.
/// `x` is [H*W, Cin], `weight` is [Cout, Cin]. A plain `matmul_bt`.
[[nodiscard]] Mat conv2d_pointwise(const Mat& x, const Mat& weight,
                                   std::span<const float> bias);

/// Dense 2-D convolution.
/// `x` is [H*W, Cin], `weight` is [Cout, Cin*KH*KW], output is [H_out*W_out, Cout].
[[nodiscard]] Mat conv2d_dense(const Mat& x, std::size_t h, std::size_t w,
                               const Mat& weight, std::size_t out_channels,
                               std::size_t kernel_h, std::size_t kernel_w,
                               std::span<const float> bias,
                               std::size_t stride, std::size_t padding);

/// Nearest-neighbour 2x upsample. The VAE's only upsampling operator --
/// it upsamples then convolves, rather than using a transposed convolution.
[[nodiscard]] Mat upsample_nearest2d(const Mat& x, std::size_t h, std::size_t w,
                                     std::size_t factor);
```

Three paths inside `conv2d_dense`, mirroring `conv1d_dense`: a 1x1 fast path
that delegates to `conv2d_pointwise`, a direct accumulation loop for small
problems, and tiled im2col + `matmul_bt` for everything else.

Tiling is not optional here. A full im2col at the VAE's last level is
`1024*1024` rows by `128*9` columns — 4.8 GB. `conv1d.cpp` already materialises
im2col in row tiles for the same reason; the tile-size logic carries over.

### GroupNorm, into `nn2.hpp`

```cpp
/// GroupNorm over spatial-major activations.
/// `x` is [H*W, C]; channels are split into `groups` contiguous blocks and each
/// block is normalized over its channels *and* all spatial positions together.
[[nodiscard]] Mat group_norm(const Mat& x, std::size_t groups,
                             std::span<const float> weight,
                             std::span<const float> bias, float eps = 1e-6f);
```

The trap: GroupNorm's statistics span space as well as channels. Normalizing
each row independently is LayerNorm, produces a plausible-looking image, and is
wrong. The test should compare against an explicit two-pass reference over both
axes.

### Tests, in `tests/test_conv2d.cpp`

Following `test_conv1d.cpp`: `ramp()` inputs, `approx_rel` at 1e-5 because
im2col and the direct loop sum in different orders.

- 1x1 against `matmul_bt`, exactly
- 3x3 pad 1 preserves size, all three paths agree
- stride 2 halves size, agrees with the direct loop
- asymmetric H != W (catches every transposed index)
- non-square kernel
- padding larger than the kernel
- nearest upsample is a pure repeat in both axes
- GroupNorm with `groups == C` reduces to per-channel normalization over space
- GroupNorm with `groups == 1` normalizes everything together
- GroupNorm statistics span space: a tensor constant per row but varying across
  rows must **not** come out as zeros

---

## Milestone 1 in detail: the VAE decoder

FLUX uses the SD3-family 16-channel `AutoencoderKL`. Decoder only — encoding
needs the analysis stack, and text-to-image never runs it.

```
z [16, H/8, W/8]
  -> conv_in   16 -> 512, 3x3
  -> mid: ResnetBlock(512), AttnBlock(512), ResnetBlock(512)
  -> up level 3: 3x ResnetBlock(512),        upsample 2x
  -> up level 2: 3x ResnetBlock(512 -> 512), upsample 2x
  -> up level 1: 3x ResnetBlock(512 -> 256), upsample 2x
  -> up level 0: 3x ResnetBlock(256 -> 128)
  -> GroupNorm, SiLU, conv_out 128 -> 3, 3x3
```

`ResnetBlock` is GroupNorm, SiLU, conv3x3, GroupNorm, SiLU, conv3x3, plus a 1x1
shortcut convolution only when the channel count changes.

`AttnBlock` is GroupNorm, then three 1x1 convolutions for q/k/v, single-head
attention over the `H*W` positions with scale `1/sqrt(C)`, then a 1x1 output
projection and a residual add. In spatial-major layout this is just
`transformer2.cpp`'s attention with one head.

### The traps

**Latent scaling.** FLUX's VAE has both a scaling factor and a shift factor:

```
z = (z_model / 0.3611) + 0.1159
```

Forgetting the shift gives a washed-out image with a colour cast — clearly
wrong, but not obviously *this*.

**Upsampling is nearest-then-convolve**, not transposed convolution. A
transposed convolution with the same weights produces checkerboard artefacts
that read as JPEG-ish texture.

**GroupNorm is 32 groups, eps 1e-6.** Not 1e-5. The default in most frameworks
is 1e-5 and it visibly shifts contrast at this depth.

**Output range.** The decoder emits roughly [-1, 1]; the image is `x / 2 + 0.5`
clamped to [0, 1]. Clamping before the shift instead of after crushes the
shadows.

### PNG writing

`png.cpp`, the counterpart of `wav.cpp`. PNG's IDAT stream is zlib, which means
a deflate encoder — but deflate has a *stored* (uncompressed) block mode, and a
stored-block stream with a correct zlib header, Adler-32 and per-chunk CRC-32 is
a completely valid PNG that any decoder reads. About 100 lines and no
dependency. A 1024x1024 RGB image lands at ~3 MB instead of ~1.5 MB, which is
the right trade for a project whose point is that nothing is imported.

---

## Milestones 2-3, in brief

### T5-XXL encoder

24 layers, `d_model` 4096, `d_ff` 10240, 64 heads of 64, no biases anywhere.
Two traps, both silent:

- **T5 does not scale attention scores.** There is no `1/sqrt(d_k)`; the
  factor is folded into the initialisation. Adding it produces embeddings that
  are smooth and wrong, and the image comes out generic rather than broken.
- **Relative position bias is computed in layer 0 only** and added to the
  scores of *every* layer. 32 buckets, bidirectional, max distance 128.

Gated GELU FFN: `gelu_tanh(wi_0(x)) * wi_1(x)`, then `wo`. RMSNorm computed in
f32. schnell truncates or pads the prompt to 256 tokens; T5 appends `</s>` and
has no BOS.

### CLIP-L text encoder

12 layers, width 768, 12 heads, LayerNorm, **causal** mask, and `quick_gelu`
(`x * sigmoid(1.702 * x)`) rather than the usual GELU. 77 tokens. The pooled
vector FLUX wants is the final-layer-norm hidden state at the end-of-text
token's position — found by `argmax` over token ids, because the EOT id is the
highest in the vocabulary.

### FLUX MMDiT

```
latent [16, h, w] --patchify 2x2--> [h/2 * w/2, 64] --x_embedder--> [N, 3072]
text  [256, 4096] --context_embedder--> [256, 3072]
pooled CLIP [768] + sinusoidal timestep [256] --> modulation vector [3072]

19 x DoubleStreamBlock:  separate img and txt streams, joint attention over
                         concat(txt, img), 6 AdaLN parameters per stream
38 x SingleStreamBlock:  one stream over concat(txt, img), attention and MLP
                         computed in parallel from a shared input, 3 AdaLN
                         parameters
--> AdaLN final layer --> Linear(3072 -> 64) --> unpatchify --> velocity
```

- Hidden 3072, 24 heads of 128. QK-norm is RMSNorm per head — `transformer5.cpp`
  has this.
- **RoPE is 3-axis**, dims `[16, 56, 56]` summing to 128, over `(t, h, w)`
  position ids. Text tokens get all-zero ids; image tokens get their patch-grid
  coordinates. Theta 10000.
- **Patchify packs channel-major within the patch**: `c (h ph) (w pw) -> (h w)
  (c ph pw)`. The other order gives a 2x2-scrambled image that still looks
  structured.
- Norms are LayerNorm **without** affine parameters, eps 1e-6 — the scale and
  shift come from AdaLN instead.
- schnell has **no** guidance embedding. dev does. Loading dev weights into a
  schnell graph fails loudly on a missing tensor, which is the good case.

### Sampler

Rectified flow, Euler, and for schnell there is no timestep shifting
(`shift = 1.0`, dynamic shifting off):

```
sigmas = linspace(1.0, 1.0 / N, N), then 0.0 appended
for i in 0..N:
    v = model(x, sigma_i, text, pooled)
    x = x + (sigma_{i+1} - sigma_i) * v
```

Four lines, and one forward pass per step because guidance is distilled away.

---

## Milestone 4: Metal

The existing quantized Metal kernels are **GEMV only** — `metal_gemv_q4k_t`
computes one output row, which is exactly right for autoregressive decode and
exactly wrong here. FLUX runs 4352 tokens through every projection.

The obvious answer is a Q4_K x dense GEMM kernel that dequantizes weight
super-blocks into threadgroup memory. The better answer is simpler:

**Dequantize each weight to half once per layer per step, then run a plain half
GEMM.** The largest single FLUX weight is a single-stream block's fused
`linear1` at 3072 -> 21504, which is 66M parameters and 132 MB as half — an
affordable scratch buffer. The dequantisation touches 66M elements; the GEMM
that follows is `2 * 66e6 * 4352` = 575 GFLOP. The dequant is noise, and it
reuses the block-decode logic already written for the GEMV kernels.

So `flux.msl` needs:

- `dequant_q4k_to_half` — per super-block, one threadgroup
- `gemm_half` — `simdgroup_matrix` tiles, the one kernel actually worth tuning
- `flash_attention` over 4352 tokens, 24 heads — the existing OmniVoice
  attention kernel keeps a full row of scores in threadgroup memory, which does
  not fit at this length, so this one is a genuine rewrite in tiled/streaming
  form
- `adaln_modulate`, `layernorm_noaffine`, `rope_3axis`, `gelu_tanh`, `add`

The full-graph pattern from `metal_decode.mm` and `metal_omnivoice.mm` applies:
upload weights once, encode all 57 blocks into a single command buffer per
step, and never round-trip to the CPU inside a step.

VAE decode can stay on the CPU at first. It is a few hundred GFLOP and BLAS
handles im2col + gemm at maybe 20 s for 1024x1024 — annoying next to a 3 minute
denoise, not blocking.

---

## Weights to fetch

| Component | Source | Format | Size |
|---|---|---|---|
| FLUX transformer | `city96/FLUX.1-schnell-gguf` | GGUF Q4_K_M | 6.7 GB |
| T5-XXL encoder | `city96/t5-v1_1-xxl-encoder-gguf` | GGUF Q4_K | 2.8 GB |
| CLIP-L | `comfyanonymous/flux_text_encoders` | safetensors F16 | 246 MB |
| VAE | `black-forest-labs/FLUX.1-schnell` | safetensors | 168 MB |
| T5 tokenizer | same repo | `spiece.model` | 792 KB |
| CLIP tokenizer | same repo | vocab + merges | 1.7 MB |

Both GGUFs are in formats `gguf.cpp` already reads, and both safetensors files
are in a format `transformer3.cpp` already parses. Nothing new on the loading
side except tensor-name mapping.

---

## CLI

Following the existing `--model` prefix dispatch in `main.cpp`:

```
rich-triplet --model flux-schnell \
             --prompt "a photograph of ..." \
             --weights models/flux1-schnell-Q4_K_M.gguf \
             --t5 models/t5-v1_1-xxl-encoder-Q4_K.gguf \
             --clip models/clip_l.safetensors \
             --vae models/ae.safetensors \
             --width 1024 --height 1024 \
             --steps 4 --seed 42 \
             --out image.png
```

---

## Build log

What each milestone cost and what it caught. The bugs are the interesting part:
three of the four were the kind that produce finite, plausible numbers.

| Milestone | Files | Tests |
|---|---|---|
| M0 conv2d, GroupNorm | `conv2d.{hpp,cpp}` | 23 |
| M1 VAE, PNG | `vae.{hpp,cpp}`, `png.{hpp,cpp}` | 32 |
| M2 text encoders | `t5.{hpp,cpp}`, `clip_text.{hpp,cpp}`, `qlinear.{hpp,cpp}` | 21 |
| M3 MMDiT, sampler | `flux.{hpp,cpp}` | 29 |
| M4 Metal | `metal_flux.{hpp,mm}`, `shaders/flux.msl` | 8 |
| Real checkpoints | tokenizer, loader and overflow fixes | 8 |
| | | **120** |

### Four bugs worth recording

**Infinity became black.** `f32_to_rgb8` guarded the cast with `isfinite`, which
sends `+inf` to 0 rather than to 255. NaN does have to be caught by name --
it compares false against both bounds, so `std::clamp` passes it through and the
cast that follows is undefined -- but an infinity needs no special case at all,
because clamping already saturates it to the right end.

**The GEMM was fed bfloat as though it were f32.** Non-Q4_K weights are widened
once at upload, but `gemm_into` ran every weight through the dequantization
step regardless, reinterpreting pairs of bfloat as single floats. The output
was finite and plausibly scaled -- roughly the right magnitude, entirely the
wrong numbers -- which is why it took a stage-by-stage dump of the CPU and GPU
intermediates to find. The `tmp(patch)` staging matched and everything
downstream of the first projection did not, which localised it in one read.

**Odd RoPE axes.** The tiny test config used `axes_dim = {2, 3, 3}`: it sums to
the head dimension, which is the condition the loader checked, but each axis
rotates *adjacent pairs* within its slice, so an odd width leaves one dimension
unrotated and pushes every later axis off its own frequencies. FLUX's real axes
-- 16, 56, 56 -- are all even for this reason. `flux_check_axes` now rejects
the odd case, and it is checked on the CPU and the GPU path alike.

**Shared staging buffers across one command buffer.** The timestep embedding,
the guidance embedding and the pooled CLIP vector are all narrow rows written
by the host, and reusing one buffer for all three is the obvious economy. It is
also wrong: the host writes them all *before* the command buffer is submitted,
so every reader sees the last value written rather than the one encoded
alongside it. Three buffers, 196 KB each.

### One shape constraint that had to be enforced rather than assumed

`flux_gemm_bt` is a 64x64 register-blocked tile with no bounds checks, so every
projection width must be a multiple of 64 and every contracted dimension a
multiple of 8. FLUX satisfies all of them naturally -- 3072, 9216, 12288,
15360, 18432, 21504 -- but a config that violates one corrupts the last partial
tile silently rather than faulting, so `MetalFluxContext::create` checks.

---

## Meeting the real checkpoints

The 112 tests passed and the model still produced a black image. Everything
that stood between the two was in the seam: naming, packaging, and two
performance cliffs. None of it was the architecture.

### 1. The T5 checkpoint is named by llama.cpp, not by HuggingFace

The loader asked for `encoder.block.3.layer.0.SelfAttention.q.weight`. The file
that people actually distribute is converted by llama.cpp, which normalises
everything to its own scheme: `enc.blk.3.attn_q.weight`, `ffn_gate`, `ffn_up`,
`ffn_down`, `enc.output_norm`. Both conventions are now accepted, and the error
names every candidate it tried.

`ffn_gate` is `wi_0` and `ffn_up` is `wi_1`, in that order -- and the order
matters, because only the gate takes the GELU.

### 2. The T5 checkpoint carries its own tokenizer

`tokenizer.ggml.tokens` and `tokenizer.ggml.scores` are a complete
SentencePiece unigram vocabulary. `--t5-tokenizer` is now optional: without it,
the tokenizer is built from the checkpoint itself, which is one fewer download
and one fewer way to pair a checkpoint with the wrong vocabulary.

### 3. The autoencoder ships under two conventions that number levels backwards

`black-forest-labs` ships `ae.safetensors` under the original LDM names --
`decoder.mid.block_1`, `decoder.up.0.block.0`, `nin_shortcut` -- and diffusers
ships `vae/diffusion_pytorch_model.safetensors` under its own. Repackaged
mirrors put either naming under either path, so the convention has to be
detected from the tensors present.

The spelling is the easy half. **The two number the upsampling levels in
opposite directions**: diffusers' `up_blocks.0` is the coarsest level and the
original's `up.0` is the finest. Loading one as the other lines up for the two
512-channel levels and then fails the channel count on the third -- which is
the good case, because reversing the resnets without reversing the upsamplers
would have produced a blurred image and no error at all.

### 4. CLIP's tokenizer is not GPT-2's

This one produced correct-looking nonsense. `HfBpeTokenizer` implements
byte-level BPE the way GPT-2 and Llama 3 do it, and it happily loaded CLIP's
49408-entry vocabulary and returned ids. They were wrong:

```
a photograph of a harbour at dawn
  as GPT-2:  64 220 1688 220 684 220 64 220 35430 220 527 220 30590
  as CLIP:   320 8853 539 320 10011 536 7689
```

Token 220 is a space. CLIP is a different family: the text is lowercased and
its whitespace collapsed first, the split *drops* whitespace rather than
attaching it to the following word, and each word carries an explicit `</w>`
marker instead of a leading space. `PreTokenizer::Clip` and an
`end_of_word_suffix` that merges onto the last symbol of each word fix it; the
suffix's presence in `tokenizer.json` is what selects the mode.

Nothing about this fails loudly. The prompt still encodes, the model still
draws, and the image is merely unrelated to what was asked for.

### 5. `matmul_q4k_t` is a scalar triple loop above batch one

The first real T5 encode did not finish in ten minutes. `Q4KMat::matmul_q4k_t`
has a fused NEON path for the single row a decode step asks for and a plain
triple loop for everything else -- the right trade for a language model and the
wrong one for a stack that never has a batch of one. Switching `QLinear` to
`matmul_q4k_t_blas`, which dequantizes a chunk of weight rows at a time and
hands each chunk to sgemm:

| | before | after |
|---|---|---|
| T5-XXL, 256 tokens | > 10 min | **6.7 s** |

### 6. Metal's fast-math `tanh` overflows, and one NaN is enough

With everything else fixed, the GPU produced a uniformly NaN image while the
CPU produced a picture. Instrumenting the forward pass stage by stage put the
first NaN in the *text* stream of the *first* double block -- the image stream
beside it was clean -- and then in its GELU specifically: 277 non-finite values
out of 3.1 M, from a finite input whose largest magnitude was 27.

The cubic term is why. At `v = 27` the tanh argument is near 724, and Metal
compiles `tanh` under fast-math into a form that evaluates `exp(2x)`. At 724
that is `inf`, and `inf/inf` is `NaN`. Clamping the argument to ±10 is exact --
tanh is saturated to within a float's resolution long before there.

The failure mode is worth remembering: 277 poisoned values out of three million
became a fully black image, because a NaN in a residual stream reaches
everything downstream of it and there are 56 blocks downstream.

Two regression tests now cover it, one on each side, both driving activations
past the point where the synthetic weights never went.

---

## What is not verified

Shorter than it was, but not empty:

1. **Numerical parity with diffusers.** The output is a photograph of what was
   asked for, which rules out the whole class of bugs that produce noise or the
   wrong subject. It does not prove that a given seed produces the *same*
   photograph the reference implementation would. Checking that needs a
   step-by-step latent comparison against a diffusers run.
2. **`--model flux-dev`.** The config, the guidance embedding and the shifted
   schedule are implemented and tested against the CPU path, but dev's weights
   are behind a gated repository and have never been loaded.
3. **Prompts with combining marks.** CLIP's normalizer includes NFC
   composition, which `normalize` does not do -- it lowercases and collapses
   whitespace only. It costs a token boundary on text carrying combining
   characters, and nothing at all on the Latin text prompts are usually in.
4. **Resolutions other than the three that were run.** 256x256, 512x512 and
   1024x1024 all work. Non-square sizes are handled in the code and covered by
   the shape tests, but no non-square image has been generated.

---

## Open questions

1. **Tiled VAE decode at 1024x1024** — implemented and covered for coverage,
   but the overlap width wants measuring against a real image rather than
   guessing.
2. **Whether to write the Q4_K GEMM after all.** The dequant-per-use path costs
   132 MB of scratch and a little bandwidth; a fused kernel would avoid both.
   Worth revisiting only if profiling says the dequant matters.
3. **Half-precision accumulation.** FLUX is trained in bf16 and tolerant, but
   the VAE decoder is not — it should stay f32.
4. **SANA as a second model.** Shares conv2d, GroupNorm, the sampler and the
   PNG writer; adds a linear-attention DiT and a Gemma-2 encoder that
   `transformer4.cpp` mostly covers. Cheap to add once FLUX is done, and it is
   the one that runs in under a minute.
