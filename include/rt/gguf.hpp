#pragma once

// =============================================================================
// GGUF file format reader
// =============================================================================
//
// Implements the subset of the GGUF spec (v2/v3) needed to load Gemma 3 and
// Qwen 3.5 weights from HuggingFace GGUF files.
//
// ## Binary layout
//
//   [magic: 4 bytes "GGUF"]
//   [version: u32le]
//   [n_tensors: u64le]
//   [n_kv: u64le]
//   [kv pairs: n_kv x (key_str, value_type, value)]
//   [tensor_infos: n_tensors x (name_str, n_dims, dims[], type, offset)]
//   [padding to alignment]
//   [tensor data: raw bytes at tensor_info offsets]
//
// ## Metadata value types
//   0=u8, 1=i8, 2=u16, 3=i16, 4=u32, 5=i32, 6=f32, 7=bool,
//   8=string, 9=array, 10=u64, 11=i64, 12=f64

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "rt/mat.hpp"
#include "rt/quant.hpp"
#include "rt/result.hpp"

namespace rt {

/// Convert an IEEE 754 half-precision float (f16) bit pattern to f32.
/// Re-exported from `bf16.hpp` so GGUF users need only this header.
using rt::f16_to_f32;

// ---------------------------------------------------------------------------
// Tensor type enum
// ---------------------------------------------------------------------------

enum class GgufType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    Unknown = 0xffffffff,
};

[[nodiscard]] GgufType gguf_type_from_u32(std::uint32_t v);

/// Human-readable name, for error messages.
[[nodiscard]] const char* gguf_type_name(GgufType t);

/// Bytes per block and elements per block, for computing total tensor bytes.
/// `nullopt` for types this reader cannot size.
[[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>> gguf_block_info(GgufType t);

/// Total bytes for `n_elements` elements of this type.
[[nodiscard]] std::optional<std::size_t> gguf_byte_size(GgufType t, std::size_t n_elements);

// ---------------------------------------------------------------------------
// Metadata values
// ---------------------------------------------------------------------------

struct GgufMetaValue;

using GgufMetaVariant = std::variant<std::uint8_t,                            // 0  U8
                                     std::int8_t,                             // 1  I8
                                     std::uint16_t,                           // 2  U16
                                     std::int16_t,                            // 3  I16
                                     std::uint32_t,                           // 4  U32
                                     std::int32_t,                            // 5  I32
                                     float,                                   // 6  F32
                                     bool,                                    // 7  Bool
                                     std::string,                             // 8  Str
                                     std::vector<GgufMetaValue>,              // 9  Array
                                     std::uint64_t,                           // 10 U64
                                     std::int64_t,                            // 11 I64
                                     double>;                                 // 12 F64

struct GgufMetaValue {
    GgufMetaVariant value;

    [[nodiscard]] std::optional<std::uint64_t> as_u64() const;
    [[nodiscard]] std::optional<std::string_view> as_str() const;
    [[nodiscard]] std::optional<float> as_f32() const;
    [[nodiscard]] const std::vector<GgufMetaValue>* as_array() const;
};

// ---------------------------------------------------------------------------
// Tensor info (from the header section)
// ---------------------------------------------------------------------------

struct GgufTensorInfo {
    std::string name;
    std::vector<std::size_t> shape;  // e.g. [in, out] in GGUF's convention
    GgufType gguf_type = GgufType::Unknown;
    std::uint64_t data_offset = 0;   // byte offset from the start of the data section

    [[nodiscard]] std::size_t n_elements() const;
};

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

class GgufFile {
   public:
    std::map<std::string, GgufMetaValue> metadata;
    std::vector<GgufTensorInfo> tensor_info;
    /// Byte offset in the file where the tensor data section starts.
    std::uint64_t data_start = 0;
    std::string file_path;

    /// Open and parse the header (metadata + tensor info).
    /// Does NOT read tensor data into memory.
    [[nodiscard]] static Result<GgufFile> open(const std::string& path);

    /// Read raw bytes for a tensor by its header index.
    [[nodiscard]] Result<std::vector<std::uint8_t>> read_tensor_bytes(std::size_t idx) const;

    /// Look up a tensor by name; returns its index in `tensor_info`.
    [[nodiscard]] std::optional<std::size_t> find_tensor(std::string_view name) const;

    /// Decode a Q4_0 tensor into our internal `Q4Mat`.
    ///
    /// GGUF Q4_0 blocks are 18 bytes per 32 elements: an f16 scale followed by
    /// 16 bytes of nibbles in *split* order --
    ///   qs[k] low nibble  = element k
    ///   qs[k] high nibble = element k + 16
    /// with unsigned values whose weight is `(nibble - 8) * scale`.
    ///
    /// `Q4Mat` instead interleaves (packed[k] holds elements 2k and 2k+1) and
    /// stores 4-bit two's complement. Conversion reorders the nibbles and
    /// remaps `v -> (v - 8) & 0x0F`; the scale carries over unchanged, since
    /// `(nibble - 8) * scale` and `signed_nibble * scale` agree.
    ///
    /// GGUF stores weight matrices as [in_features, out_features]; we flip to
    /// [out_features, in_features] to match `input @ weight.T`.
    [[nodiscard]] Result<Q4Mat> decode_q4_0_to_q4mat(std::size_t idx) const;

    /// Load a Q4_K tensor into a `Q4KMat` by copying the raw block bytes.
    ///
    /// The 144-byte/256-element GGUF layout is preserved verbatim;
    /// dequantization happens on the fly in `Q4KMat::dequantize_row_into`.
    [[nodiscard]] Result<Q4KMat> decode_q4k_to_q4kmat(std::size_t idx) const;

    /// Decode an F32 tensor.
    [[nodiscard]] Result<std::vector<float>> decode_f32(std::size_t idx) const;

    /// Decode a BF16 tensor into raw bf16 bits.
    [[nodiscard]] Result<std::vector<std::uint16_t>> decode_bf16(std::size_t idx) const;

    /// Decode an F16 tensor, converting to f32 on load.
    [[nodiscard]] Result<std::vector<float>> decode_f16_to_f32(std::size_t idx) const;

    /// Decode a Q4_0 tensor directly to flat f32, in GGUF memory order.
    [[nodiscard]] Result<std::vector<float>> decode_q4_0_to_f32(std::size_t idx) const;

    /// Decode a Q6_K tensor to flat f32.
    ///
    /// Q6_K blocks are 210 bytes per 256 elements:
    ///   ql[128]    -- lower 4 bits of each 6-bit quant
    ///   qh[64]     -- upper 2 bits
    ///   scales[16] -- int8 scales, one per 16-element group
    ///   d          -- fp16 super-block scale
    /// The 256 elements are two halves of 128; within each half, four lanes of
    /// 32 are interleaved (see llama.cpp `dequantize_row_q6_K`).
    [[nodiscard]] Result<std::vector<float>> decode_q6k_to_f32(std::size_t idx) const;

    /// Decode a Q4_K tensor to flat f32.
    ///
    /// Q4_K blocks are 144 bytes per 256 elements: f16 `d`, f16 `dmin`, 12
    /// bytes of packed 6-bit scale/min pairs, then 128 bytes of 4-bit quants.
    [[nodiscard]] Result<std::vector<float>> decode_q4k_to_f32(std::size_t idx) const;

    /// Decode a Q8_0 tensor to flat f32.
    /// Blocks are 34 bytes per 32 elements: f16 scale, then 32 int8 quants.
    [[nodiscard]] Result<std::vector<float>> decode_q8_0_to_f32(std::size_t idx) const;

    /// Decode a Q5_K tensor to flat f32.
    ///
    /// Q5_K blocks are 176 bytes per 256 elements: f16 `d`, f16 `dmin`, 12
    /// bytes of packed scales, 32 bytes of high bits, 128 bytes of low nibbles.
    [[nodiscard]] Result<std::vector<float>> decode_q5k_to_f32(std::size_t idx) const;
};

/// Extract the 6-bit scale and min for pair `j` (0..8) from a 12-byte packed
/// scales array. This is llama.cpp's `get_scale_min_k4`, shared by Q4_K and Q5_K.
void gguf_get_scale_min(const std::uint8_t* sc, std::size_t j, float& scale, float& min);

// =============================================================================
// Weight loading helpers
// =============================================================================

/// Read a convolution weight out of GGUF into `conv1d.hpp`'s layout.
///
/// GGUF *lists* dimensions in reverse of PyTorch but stores the same flat
/// buffer, so no transpose is needed -- a `Conv1d` weight listed as
/// `[K, Cin, Cout]` is still `[Cout][Cin][K]` row-major in memory, which is
/// exactly the `[Cout, Cin * K]` matrix the convolution routines want. Only
/// the row and column counts have to be worked out.
///
/// Those are derived from `out_channels` rather than from the shape, because
/// GGUF elides trailing dimensions of length 1: the decoder's final
/// `[1, 32, 7]` projection is listed as a two-dimensional `[7, 32]`, and there
/// is no way to tell that from a genuinely two-dimensional weight. Passing the
/// expected output width sidesteps the ambiguity and validates the tensor at
/// the same time.
[[nodiscard]] Result<Mat> load_gguf_conv_weight(const GgufFile& gguf, const std::string& name,
                                                std::size_t out_channels);

/// Read a `[1, C]` Snake alpha as a flat vector.
[[nodiscard]] Result<std::vector<float>> load_gguf_alpha(const GgufFile& gguf,
                                                         const std::string& name);

/// Read an f32/f16 tensor as a flat vector, for biases.
[[nodiscard]] Result<std::vector<float>> load_gguf_vector(const GgufFile& gguf,
                                                          const std::string& name);
}  // namespace rt
