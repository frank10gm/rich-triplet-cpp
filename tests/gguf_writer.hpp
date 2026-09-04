#pragma once

// Minimal GGUF *writer*, used only by the tests to synthesize files the reader
// can be pointed at. Mirrors the layout documented in `rt/gguf.hpp`.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "rt/gguf.hpp"

namespace rt::testing {

class GgufWriter {
   public:
    struct Tensor {
        std::string name;
        std::vector<std::size_t> shape;  // GGUF order: [in, out]
        GgufType type;
        std::vector<std::uint8_t> data;
    };

    void add_meta_u32(std::string key, std::uint32_t v) {
        meta_.push_back({std::move(key), 4, encode_u32(v)});
    }
    void add_meta_u64(std::string key, std::uint64_t v) {
        meta_.push_back({std::move(key), 10, encode_u64(v)});
    }
    void add_meta_f32(std::string key, float v) {
        std::uint32_t bits;
        std::memcpy(&bits, &v, 4);
        meta_.push_back({std::move(key), 6, encode_u32(bits)});
    }
    void add_meta_str(std::string key, const std::string& v) {
        std::vector<std::uint8_t> payload = encode_u64(v.size());
        payload.insert(payload.end(), v.begin(), v.end());
        meta_.push_back({std::move(key), 8, std::move(payload)});
    }
    /// A GGUF array of strings (`elem_type = 8`), as used for tokenizer vocabs.
    void add_meta_str_array(std::string key, const std::vector<std::string>& items) {
        std::vector<std::uint8_t> payload = encode_u32(8);
        append(payload, encode_u64(items.size()));
        for (const auto& s : items) {
            append(payload, encode_u64(s.size()));
            payload.insert(payload.end(), s.begin(), s.end());
        }
        meta_.push_back({std::move(key), 9, std::move(payload)});
    }

    void add_tensor(std::string name, std::vector<std::size_t> shape, GgufType type,
                    std::vector<std::uint8_t> data) {
        tensors_.push_back({std::move(name), std::move(shape), type, std::move(data)});
    }

    /// Serialize to `path`. Returns false if the file could not be written.
    bool write(const std::string& path) const {
        std::vector<std::uint8_t> header;
        const char magic[4] = {'G', 'G', 'U', 'F'};
        header.insert(header.end(), magic, magic + 4);
        append(header, encode_u32(3));  // version
        append(header, encode_u64(tensors_.size()));
        append(header, encode_u64(meta_.size()));

        for (const auto& kv : meta_) {
            append(header, encode_u64(kv.key.size()));
            header.insert(header.end(), kv.key.begin(), kv.key.end());
            append(header, encode_u32(kv.type));
            append(header, kv.payload);
        }

        // Tensor infos carry offsets into the data section, so lay the data out
        // first (each tensor aligned to `kAlignment`).
        std::vector<std::uint64_t> offsets;
        std::uint64_t cursor = 0;
        for (const auto& t : tensors_) {
            offsets.push_back(cursor);
            cursor = align_up(cursor + t.data.size(), kAlignment);
        }

        for (std::size_t i = 0; i < tensors_.size(); ++i) {
            const auto& t = tensors_[i];
            append(header, encode_u64(t.name.size()));
            header.insert(header.end(), t.name.begin(), t.name.end());
            append(header, encode_u32(static_cast<std::uint32_t>(t.shape.size())));
            for (std::size_t d : t.shape) {
                append(header, encode_u64(d));
            }
            append(header, encode_u32(static_cast<std::uint32_t>(t.type)));
            append(header, encode_u64(offsets[i]));
        }

        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) {
            return false;
        }
        f.write(reinterpret_cast<const char*>(header.data()),
                static_cast<std::streamsize>(header.size()));

        // Pad to the data-section alignment.
        for (std::uint64_t pos = header.size(); pos < align_up(header.size(), kAlignment); ++pos) {
            f.put('\0');
        }

        for (std::size_t i = 0; i < tensors_.size(); ++i) {
            const auto& t = tensors_[i];
            f.write(reinterpret_cast<const char*>(t.data.data()),
                    static_cast<std::streamsize>(t.data.size()));
            const std::uint64_t padded = align_up(t.data.size(), kAlignment);
            for (std::uint64_t p = t.data.size(); p < padded; ++p) {
                f.put('\0');
            }
        }
        return static_cast<bool>(f);
    }

   private:
    struct Meta {
        std::string key;
        std::uint32_t type;
        std::vector<std::uint8_t> payload;
    };

    static constexpr std::uint64_t kAlignment = 32;

    static std::uint64_t align_up(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

    static void append(std::vector<std::uint8_t>& dst, const std::vector<std::uint8_t>& src) {
        dst.insert(dst.end(), src.begin(), src.end());
    }

    static std::vector<std::uint8_t> encode_u32(std::uint32_t v) {
        return {static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(v >> 8),
                static_cast<std::uint8_t>(v >> 16), static_cast<std::uint8_t>(v >> 24)};
    }

    static std::vector<std::uint8_t> encode_u64(std::uint64_t v) {
        std::vector<std::uint8_t> b(8);
        for (int i = 0; i < 8; ++i) {
            b[i] = static_cast<std::uint8_t>(v >> (8 * i));
        }
        return b;
    }

    std::vector<Meta> meta_;
    std::vector<Tensor> tensors_;
};

}  // namespace rt::testing
