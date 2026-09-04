#include "rt/gguf.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace rt {

// ---------------------------------------------------------------------------
// Tensor type enum
// ---------------------------------------------------------------------------

GgufType gguf_type_from_u32(std::uint32_t v) {
    switch (v) {
        case 0: return GgufType::F32;
        case 1: return GgufType::F16;
        case 2: return GgufType::Q4_0;
        case 3: return GgufType::Q4_1;
        case 6: return GgufType::Q5_0;
        case 7: return GgufType::Q5_1;
        case 8: return GgufType::Q8_0;
        case 9: return GgufType::Q8_1;
        case 10: return GgufType::Q2K;
        case 11: return GgufType::Q3K;
        case 12: return GgufType::Q4K;
        case 13: return GgufType::Q5K;
        case 14: return GgufType::Q6K;
        case 15: return GgufType::Q8K;
        case 16: return GgufType::Iq2Xxs;
        case 17: return GgufType::Iq2Xs;
        case 18: return GgufType::Iq3Xxs;
        case 19: return GgufType::Iq1S;
        case 20: return GgufType::Iq4Nl;
        case 21: return GgufType::Iq3S;
        case 22: return GgufType::Iq2S;
        case 23: return GgufType::Iq4Xs;
        case 24: return GgufType::I8;
        case 25: return GgufType::I16;
        case 26: return GgufType::I32;
        case 27: return GgufType::I64;
        case 28: return GgufType::F64;
        case 29: return GgufType::Iq1M;
        case 30: return GgufType::Bf16;
        default: return GgufType::Unknown;
    }
}

const char* gguf_type_name(GgufType t) {
    switch (t) {
        case GgufType::F32: return "F32";
        case GgufType::F16: return "F16";
        case GgufType::Q4_0: return "Q4_0";
        case GgufType::Q4_1: return "Q4_1";
        case GgufType::Q5_0: return "Q5_0";
        case GgufType::Q5_1: return "Q5_1";
        case GgufType::Q8_0: return "Q8_0";
        case GgufType::Q8_1: return "Q8_1";
        case GgufType::Q2K: return "Q2K";
        case GgufType::Q3K: return "Q3K";
        case GgufType::Q4K: return "Q4K";
        case GgufType::Q5K: return "Q5K";
        case GgufType::Q6K: return "Q6K";
        case GgufType::Q8K: return "Q8K";
        case GgufType::I8: return "I8";
        case GgufType::I16: return "I16";
        case GgufType::I32: return "I32";
        case GgufType::I64: return "I64";
        case GgufType::F64: return "F64";
        case GgufType::Bf16: return "Bf16";
        default: return "Unknown";
    }
}

std::optional<std::pair<std::size_t, std::size_t>> gguf_block_info(GgufType t) {
    switch (t) {
        case GgufType::F32: return std::pair{std::size_t{4}, std::size_t{1}};
        case GgufType::F16: return std::pair{std::size_t{2}, std::size_t{1}};
        case GgufType::Bf16: return std::pair{std::size_t{2}, std::size_t{1}};
        case GgufType::Q4_0: return std::pair{std::size_t{18}, std::size_t{32}};  // f16 scale + 16B nibbles
        case GgufType::Q8_0: return std::pair{std::size_t{34}, std::size_t{32}};  // f16 scale + 32B i8
        case GgufType::Q4K: return std::pair{std::size_t{144}, std::size_t{256}};
        case GgufType::Q5K: return std::pair{std::size_t{176}, std::size_t{256}};
        case GgufType::Q6K: return std::pair{std::size_t{210}, std::size_t{256}};
        default: return std::nullopt;
    }
}

std::optional<std::size_t> gguf_byte_size(GgufType t, std::size_t n_elements) {
    const auto info = gguf_block_info(t);
    if (!info) {
        return std::nullopt;
    }
    const auto [bpb, epb] = *info;
    return ((n_elements + epb - 1) / epb) * bpb;
}

std::size_t GgufTensorInfo::n_elements() const {
    std::size_t n = 1;
    for (std::size_t d : shape) {
        n *= d;
    }
    return n;
}

// ---------------------------------------------------------------------------
// Metadata values
// ---------------------------------------------------------------------------

std::optional<std::uint64_t> GgufMetaValue::as_u64() const {
    if (const auto* v = std::get_if<std::uint8_t>(&value)) return *v;
    if (const auto* v = std::get_if<std::uint16_t>(&value)) return *v;
    if (const auto* v = std::get_if<std::uint32_t>(&value)) return *v;
    if (const auto* v = std::get_if<std::uint64_t>(&value)) return *v;
    if (const auto* v = std::get_if<std::int32_t>(&value)) return static_cast<std::uint64_t>(*v);
    if (const auto* v = std::get_if<std::int64_t>(&value)) return static_cast<std::uint64_t>(*v);
    return std::nullopt;
}

std::optional<std::string_view> GgufMetaValue::as_str() const {
    if (const auto* v = std::get_if<std::string>(&value)) return std::string_view(*v);
    return std::nullopt;
}

std::optional<float> GgufMetaValue::as_f32() const {
    if (const auto* v = std::get_if<float>(&value)) return *v;
    if (const auto* v = std::get_if<double>(&value)) return static_cast<float>(*v);
    return std::nullopt;
}

const std::vector<GgufMetaValue>* GgufMetaValue::as_array() const {
    return std::get_if<std::vector<GgufMetaValue>>(&value);
}

// ---------------------------------------------------------------------------
// Low-level I/O helpers
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] Result<void> read_exact(std::istream& f, void* dst, std::size_t n) {
    f.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (!f || static_cast<std::size_t>(f.gcount()) != n) {
        return err("unexpected end of file");
    }
    return {};
}

[[nodiscard]] Result<std::uint8_t> read_u8(std::istream& f) {
    std::uint8_t b = 0;
    RT_TRY_VOID(read_exact(f, &b, 1));
    return b;
}

[[nodiscard]] Result<std::uint16_t> read_u16le(std::istream& f) {
    std::uint8_t b[2];
    RT_TRY_VOID(read_exact(f, b, 2));
    return static_cast<std::uint16_t>(b[0] | (b[1] << 8));
}

[[nodiscard]] Result<std::uint32_t> read_u32le(std::istream& f) {
    std::uint8_t b[4];
    RT_TRY_VOID(read_exact(f, b, 4));
    return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
           (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

[[nodiscard]] Result<std::uint64_t> read_u64le(std::istream& f) {
    std::uint8_t b[8];
    RT_TRY_VOID(read_exact(f, b, 8));
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | b[i];
    }
    return v;
}

[[nodiscard]] Result<std::string> read_string(std::istream& f) {
    RT_TRY(len, read_u64le(f));
    std::string s(static_cast<std::size_t>(len), '\0');
    if (len > 0) {
        RT_TRY_VOID(read_exact(f, s.data(), static_cast<std::size_t>(len)));
    }
    return s;
}

[[nodiscard]] constexpr std::uint64_t align_up(std::uint64_t x, std::uint64_t align) {
    return (x + align - 1) / align * align;
}

[[nodiscard]] Result<GgufMetaValue> read_meta_typed(std::istream& f, std::uint32_t vtype);

[[nodiscard]] Result<GgufMetaValue> read_meta(std::istream& f) {
    RT_TRY(vtype, read_u32le(f));
    return read_meta_typed(f, vtype);
}

Result<GgufMetaValue> read_meta_typed(std::istream& f, std::uint32_t vtype) {
    switch (vtype) {
        case 0: {
            RT_TRY(v, read_u8(f));
            return GgufMetaValue{v};
        }
        case 1: {
            RT_TRY(v, read_u8(f));
            return GgufMetaValue{static_cast<std::int8_t>(v)};
        }
        case 2: {
            RT_TRY(v, read_u16le(f));
            return GgufMetaValue{v};
        }
        case 3: {
            RT_TRY(v, read_u16le(f));
            return GgufMetaValue{static_cast<std::int16_t>(v)};
        }
        case 4: {
            RT_TRY(v, read_u32le(f));
            return GgufMetaValue{v};
        }
        case 5: {
            RT_TRY(v, read_u32le(f));
            return GgufMetaValue{static_cast<std::int32_t>(v)};
        }
        case 6: {
            RT_TRY(v, read_u32le(f));
            return GgufMetaValue{std::bit_cast<float>(v)};
        }
        case 7: {
            RT_TRY(v, read_u8(f));
            return GgufMetaValue{v != 0};
        }
        case 8: {
            RT_TRY(v, read_string(f));
            return GgufMetaValue{std::move(v)};
        }
        case 9: {
            // array: elem_type (u32), count (u64), elements
            RT_TRY(elem_type, read_u32le(f));
            RT_TRY(count, read_u64le(f));
            std::vector<GgufMetaValue> arr;
            arr.reserve(std::min<std::uint64_t>(count, 1024));
            for (std::uint64_t i = 0; i < count; ++i) {
                RT_TRY(item, read_meta_typed(f, elem_type));
                arr.push_back(std::move(item));
            }
            return GgufMetaValue{std::move(arr)};
        }
        case 10: {
            RT_TRY(v, read_u64le(f));
            return GgufMetaValue{v};
        }
        case 11: {
            RT_TRY(v, read_u64le(f));
            return GgufMetaValue{static_cast<std::int64_t>(v)};
        }
        case 12: {
            RT_TRY(v, read_u64le(f));
            return GgufMetaValue{std::bit_cast<double>(v)};
        }
        default:
            return err("unknown metadata value type " + std::to_string(vtype));
    }
}

[[nodiscard]] std::uint16_t read_u16_at(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

}  // namespace

void gguf_get_scale_min(const std::uint8_t* sc, std::size_t j, float& scale, float& min) {
    std::uint8_t sc_val, min_val;
    if (j < 4) {
        sc_val = sc[j] & 0x3f;
        min_val = sc[j + 4] & 0x3f;
    } else {
        sc_val = static_cast<std::uint8_t>((sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4));
        min_val = static_cast<std::uint8_t>((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
    }
    scale = static_cast<float>(sc_val);
    min = static_cast<float>(min_val);
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

Result<GgufFile> GgufFile::open(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return err("cannot open GGUF file: " + path);
    }

    char magic[4];
    RT_TRY_VOID(read_exact(f, magic, 4));
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        return err("not a GGUF file: " + path);
    }

    RT_TRY(version, read_u32le(f));
    if (version < 2 || version > 3) {
        return err("unsupported GGUF version " + std::to_string(version));
    }

    RT_TRY(n_tensors, read_u64le(f));
    RT_TRY(n_kv, read_u64le(f));

    GgufFile out;
    out.file_path = path;

    for (std::uint64_t i = 0; i < n_kv; ++i) {
        RT_TRY(key, read_string(f));
        RT_TRY(val, read_meta(f));
        out.metadata.emplace(std::move(key), std::move(val));
    }

    out.tensor_info.reserve(static_cast<std::size_t>(n_tensors));
    for (std::uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensorInfo info;
        RT_TRY(name, read_string(f));
        info.name = std::move(name);
        RT_TRY(n_dims, read_u32le(f));
        info.shape.resize(n_dims);
        for (std::uint32_t d = 0; d < n_dims; ++d) {
            RT_TRY(dim, read_u64le(f));
            info.shape[d] = static_cast<std::size_t>(dim);
        }
        RT_TRY(type_id, read_u32le(f));
        info.gguf_type = gguf_type_from_u32(type_id);
        RT_TRY(offset, read_u64le(f));
        info.data_offset = offset;
        out.tensor_info.push_back(std::move(info));
    }

    // Alignment for the data section (default 32 when unset).
    std::uint64_t alignment = 32;
    if (auto it = out.metadata.find("general.alignment"); it != out.metadata.end()) {
        if (auto a = it->second.as_u64()) {
            alignment = *a;
        }
    }

    // Tensor data starts after the header, padded to alignment.
    const auto current_pos = static_cast<std::uint64_t>(f.tellg());
    out.data_start = align_up(current_pos, alignment);

    return out;
}

Result<std::vector<std::uint8_t>> GgufFile::read_tensor_bytes(std::size_t idx) const {
    const GgufTensorInfo& info = tensor_info[idx];
    const auto n_bytes = gguf_byte_size(info.gguf_type, info.n_elements());
    if (!n_bytes) {
        return err(std::string("unsupported gguf type ") + gguf_type_name(info.gguf_type) +
                   " for tensor " + info.name);
    }

    std::ifstream f(file_path, std::ios::binary);
    if (!f) {
        return err("cannot reopen GGUF file: " + file_path);
    }
    f.seekg(static_cast<std::streamoff>(data_start + info.data_offset));
    std::vector<std::uint8_t> buf(*n_bytes);
    RT_TRY_VOID(read_exact(f, buf.data(), buf.size()));
    return buf;
}

std::optional<std::size_t> GgufFile::find_tensor(std::string_view name) const {
    for (std::size_t i = 0; i < tensor_info.size(); ++i) {
        if (tensor_info[i].name == name) {
            return i;
        }
    }
    return std::nullopt;
}

namespace {
/// GGUF stores weight matrices as [in_features, out_features]; we store
/// [out_features, in_features] so `input @ weight.T` works directly.
[[nodiscard]] Result<std::pair<std::size_t, std::size_t>> transposed_shape(
    const GgufTensorInfo& info) {
    switch (info.shape.size()) {
        case 1: return std::pair{std::size_t{1}, info.shape[0]};
        case 2: return std::pair{info.shape[1], info.shape[0]};
        default:
            return err("unexpected shape rank " + std::to_string(info.shape.size()) + " for " +
                       info.name);
    }
}
}  // namespace

Result<Q4Mat> GgufFile::decode_q4_0_to_q4mat(std::size_t idx) const {
    const GgufTensorInfo& info = tensor_info[idx];
    assert(info.gguf_type == GgufType::Q4_0 &&
           "decode_q4_0_to_q4mat called on non-Q4_0 tensor");

    RT_TRY(shape, transposed_shape(info));
    RT_TRY(bytes, read_tensor_bytes(idx));

    const std::size_t n_elem = info.n_elements();
    const std::size_t n_blocks = (n_elem + 31) / 32;

    Q4Mat q;
    q.rows = shape.first;
    q.cols = shape.second;
    q.packed.assign((n_elem + 1) / 2, 0);
    q.scales.reserve(n_blocks);

    // Write nibble `v` for absolute element index `elem`: low nibble of
    // packed[elem/2] when even, high nibble when odd.
    const auto set_nibble = [&q](std::size_t elem, std::uint8_t v) {
        const std::size_t byte_idx = elem / 2;
        if (elem % 2 == 0) {
            q.packed[byte_idx] = static_cast<std::uint8_t>((q.packed[byte_idx] & 0xf0) | (v & 0x0f));
        } else {
            q.packed[byte_idx] =
                static_cast<std::uint8_t>((q.packed[byte_idx] & 0x0f) | ((v & 0x0f) << 4));
        }
    };

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::size_t block_off = b * 18;

        // GGUF dequant is (nibble - 8) * scale; after remapping the nibble to
        // two's complement our dequant is signed_nibble * scale, so the scale
        // carries over unchanged.
        q.scales.push_back(f16_to_f32(read_u16_at(bytes.data() + block_off)));

        const std::size_t nibble_off = block_off + 2;
        const std::size_t start_elem = b * 32;
        const std::size_t end_elem = std::min(start_elem + 32, n_elem);

        // Split layout: qs[k] low -> element start+k, high -> element start+k+16.
        for (std::size_t k = 0; k < 16; ++k) {
            const std::uint8_t src = bytes[nibble_off + k];

            const std::size_t e0 = start_elem + k;
            if (e0 < end_elem) {
                set_nibble(e0, static_cast<std::uint8_t>((src & 0x0f) - 8) & 0x0f);
            }

            const std::size_t e1 = start_elem + k + 16;
            if (e1 < end_elem) {
                set_nibble(e1, static_cast<std::uint8_t>(((src >> 4) & 0x0f) - 8) & 0x0f);
            }
        }
    }

    return q;
}

Result<Q4KMat> GgufFile::decode_q4k_to_q4kmat(std::size_t idx) const {
    const GgufTensorInfo& info = tensor_info[idx];
    assert(info.gguf_type == GgufType::Q4K && "decode_q4k_to_q4kmat called on non-Q4K tensor");

    RT_TRY(shape, transposed_shape(info));
    RT_TRY(bytes, read_tensor_bytes(idx));

    Q4KMat q;
    q.rows = shape.first;
    q.cols = shape.second;
    const std::size_t n_blocks = (q.rows * q.cols + 255) / 256;
    if (bytes.size() != n_blocks * 144) {
        return err("Q4K tensor " + info.name + " expected " + std::to_string(n_blocks * 144) +
                   " bytes, got " + std::to_string(bytes.size()));
    }
    q.blocks = std::move(bytes);
    return q;
}

Result<std::vector<float>> GgufFile::decode_f32(std::size_t idx) const {
    RT_TRY(bytes, read_tensor_bytes(idx));
    std::vector<float> out(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

Result<std::vector<std::uint16_t>> GgufFile::decode_bf16(std::size_t idx) const {
    RT_TRY(bytes, read_tensor_bytes(idx));
    std::vector<std::uint16_t> out(bytes.size() / 2);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

Result<std::vector<float>> GgufFile::decode_f16_to_f32(std::size_t idx) const {
    RT_TRY(bytes, read_tensor_bytes(idx));
    const std::size_t n = bytes.size() / 2;
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = f16_to_f32(read_u16_at(bytes.data() + i * 2));
    }
    return out;
}

Result<std::vector<float>> GgufFile::decode_q4_0_to_f32(std::size_t idx) const {
    const GgufTensorInfo& info = tensor_info[idx];
    assert(info.gguf_type == GgufType::Q4_0 && "decode_q4_0_to_f32 called on non-Q4_0 tensor");

    RT_TRY(bytes, read_tensor_bytes(idx));
    const std::size_t n_elem = info.n_elements();
    const std::size_t n_blocks = (n_elem + 31) / 32;
    std::vector<float> out(n_elem, 0.0f);

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::size_t block_off = b * 18;
        const float scale = f16_to_f32(read_u16_at(bytes.data() + block_off));

        const std::size_t nibble_off = block_off + 2;
        const std::size_t start_elem = b * 32;
        const std::size_t end_elem = std::min(start_elem + 32, n_elem);

        // Split layout: qs[k] low -> elem(start+k), high -> elem(start+k+16).
        for (std::size_t k = 0; k < 16; ++k) {
            const std::uint8_t src = bytes[nibble_off + k];

            const std::size_t e0 = start_elem + k;
            if (e0 < end_elem) {
                out[e0] = static_cast<float>(static_cast<int>(src & 0x0f) - 8) * scale;
            }

            const std::size_t e1 = start_elem + k + 16;
            if (e1 < end_elem) {
                out[e1] = static_cast<float>(static_cast<int>((src >> 4) & 0x0f) - 8) * scale;
            }
        }
    }

    return out;
}

Result<std::vector<float>> GgufFile::decode_q6k_to_f32(std::size_t idx) const {
    const GgufTensorInfo& info = tensor_info[idx];
    assert(info.gguf_type == GgufType::Q6K && "decode_q6k_to_f32 called on non-Q6K tensor");

    RT_TRY(bytes, read_tensor_bytes(idx));
    const std::size_t n_elem = info.n_elements();
    const std::size_t n_blocks = (n_elem + 255) / 256;
    std::vector<float> out(n_elem, 0.0f);

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::size_t block_off = b * 210;
        const std::uint8_t* ql_all = bytes.data() + block_off;
        const std::uint8_t* qh_all = bytes.data() + block_off + 128;
        const auto* sc_all = reinterpret_cast<const std::int8_t*>(bytes.data() + block_off + 192);
        const float d = f16_to_f32(read_u16_at(bytes.data() + block_off + 208));

        const std::size_t base = b * 256;
        const std::size_t end = std::min(base + 256, n_elem);

        // Two halves of 128 elements each.
        for (std::size_t j = 0; j < 2; ++j) {
            const std::uint8_t* ql = ql_all + j * 64;
            const std::uint8_t* qh = qh_all + j * 32;
            const std::int8_t* sc = sc_all + j * 8;
            const std::size_t y_base = base + j * 128;

            // Each l produces 4 outputs, at l, l+32, l+64, l+96.
            for (std::size_t l = 0; l < 32; ++l) {
                const std::size_t is = l / 16;  // 0 for l=0..15, 1 for l=16..31
                const int q1 = static_cast<int>((ql[l] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int>((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                // Scale pairs: is=0 -> sc[0],sc[2],sc[4],sc[6]; is=1 -> sc[1],sc[3],sc[5],sc[7].
                const float s0 = d * static_cast<float>(sc[is]);
                const float s1 = d * static_cast<float>(sc[is + 2]);
                const float s2 = d * static_cast<float>(sc[is + 4]);
                const float s3 = d * static_cast<float>(sc[is + 6]);

                if (y_base + l < end) out[y_base + l] = s0 * static_cast<float>(q1);
                if (y_base + l + 32 < end) out[y_base + l + 32] = s1 * static_cast<float>(q2);
                if (y_base + l + 64 < end) out[y_base + l + 64] = s2 * static_cast<float>(q3);
                if (y_base + l + 96 < end) out[y_base + l + 96] = s3 * static_cast<float>(q4);
            }
        }
    }

    return out;
}

Result<std::vector<float>> GgufFile::decode_q4k_to_f32(std::size_t idx) const {
    const GgufTensorInfo& info = tensor_info[idx];
    assert(info.gguf_type == GgufType::Q4K && "decode_q4k_to_f32 called on non-Q4K tensor");

    RT_TRY(bytes, read_tensor_bytes(idx));
    const std::size_t n_elem = info.n_elements();
    const std::size_t n_blocks = (n_elem + 255) / 256;
    std::vector<float> out(n_elem, 0.0f);

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::size_t block_off = b * 144;
        const float d = f16_to_f32(read_u16_at(bytes.data() + block_off));
        const float dmin = f16_to_f32(read_u16_at(bytes.data() + block_off + 2));
        const std::uint8_t* sc = bytes.data() + block_off + 4;
        const std::uint8_t* qs = bytes.data() + block_off + 16;

        const std::size_t base = b * 256;
        const std::size_t end = std::min(base + 256, n_elem);

        // Four chunks of 64 elements; each advances qs by 32 bytes and uses
        // two scale/min pairs.
        std::size_t q_off = 0, is = 0, elem = 0;
        for (std::size_t chunk = 0; chunk < 4; ++chunk) {
            float d1, m1, d2, m2;
            gguf_get_scale_min(sc, is, d1, m1);
            gguf_get_scale_min(sc, is + 1, d2, m2);
            const float scale1 = d * d1, min1 = dmin * m1;
            const float scale2 = d * d2, min2 = dmin * m2;

            for (std::size_t l = 0; l < 32; ++l, ++elem) {
                if (base + elem < end) {
                    out[base + elem] = scale1 * static_cast<float>(qs[q_off + l] & 0x0f) - min1;
                }
            }
            for (std::size_t l = 0; l < 32; ++l, ++elem) {
                if (base + elem < end) {
                    out[base + elem] = scale2 * static_cast<float>(qs[q_off + l] >> 4) - min2;
                }
            }

            q_off += 32;
            is += 2;
        }
    }

    return out;
}

Result<std::vector<float>> GgufFile::decode_q8_0_to_f32(std::size_t idx) const {
    const GgufTensorInfo& info = tensor_info[idx];
    assert(info.gguf_type == GgufType::Q8_0 && "decode_q8_0_to_f32 called on non-Q8_0 tensor");

    RT_TRY(bytes, read_tensor_bytes(idx));
    const std::size_t n_elem = info.n_elements();
    const std::size_t n_blocks = (n_elem + 31) / 32;
    std::vector<float> out(n_elem, 0.0f);

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::size_t block_off = b * 34;
        const float d = f16_to_f32(read_u16_at(bytes.data() + block_off));

        const std::size_t base = b * 32;
        const std::size_t end = std::min(base + 32, n_elem);
        for (std::size_t i = 0; i < 32; ++i) {
            if (base + i < end) {
                const auto q = static_cast<std::int8_t>(bytes[block_off + 2 + i]);
                out[base + i] = d * static_cast<float>(q);
            }
        }
    }

    return out;
}

Result<std::vector<float>> GgufFile::decode_q5k_to_f32(std::size_t idx) const {
    const GgufTensorInfo& info = tensor_info[idx];
    assert(info.gguf_type == GgufType::Q5K && "decode_q5k_to_f32 called on non-Q5K tensor");

    RT_TRY(bytes, read_tensor_bytes(idx));
    const std::size_t n_elem = info.n_elements();
    const std::size_t n_blocks = (n_elem + 255) / 256;
    std::vector<float> out(n_elem, 0.0f);

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::size_t block_off = b * 176;
        const float d = f16_to_f32(read_u16_at(bytes.data() + block_off));
        const float dmin = f16_to_f32(read_u16_at(bytes.data() + block_off + 2));
        const std::uint8_t* sc = bytes.data() + block_off + 4;
        const std::uint8_t* qh = bytes.data() + block_off + 16;
        const std::uint8_t* qs = bytes.data() + block_off + 48;

        const std::size_t base = b * 256;
        const std::size_t end = std::min(base + 256, n_elem);

        std::size_t q_off = 0, is = 0, elem = 0;
        for (std::size_t chunk = 0; chunk < 4; ++chunk) {
            float d1, m1, d2, m2;
            gguf_get_scale_min(sc, is, d1, m1);
            gguf_get_scale_min(sc, is + 1, d2, m2);
            const float scale1 = d * d1, min1 = dmin * m1;
            const float scale2 = d * d2, min2 = dmin * m2;

            // First 32: low nibbles plus the high bit from qh.
            for (std::size_t l = 0; l < 32; ++l, ++elem) {
                if (base + elem < end) {
                    const std::uint32_t lo = qs[q_off + l] & 0x0f;
                    const std::uint32_t hi = (qh[l] >> is) & 1;
                    out[base + elem] = scale1 * static_cast<float>(lo | (hi << 4)) - min1;
                }
            }
            // Next 32: high nibbles plus the high bit.
            for (std::size_t l = 0; l < 32; ++l, ++elem) {
                if (base + elem < end) {
                    const std::uint32_t lo = qs[q_off + l] >> 4;
                    const std::uint32_t hi = (qh[l] >> (is + 1)) & 1;
                    out[base + elem] = scale2 * static_cast<float>(lo | (hi << 4)) - min2;
                }
            }

            q_off += 32;
            is += 2;
        }
    }

    return out;
}

}  // namespace rt
