#pragma once

// Builds `torch.save`-shaped archives in memory, so the reader can be tested
// without a Python interpreter in the loop. Mirrors `gguf_writer.hpp`.
//
// Only what the reader has to cope with is modelled: stored ZIP entries, a
// pickle manifest of `_rebuild_tensor_v2` calls, and the knobs needed to
// produce the malformed cases (compression, bad strides, unknown globals,
// unsupported opcodes).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace rt::testing {

// ---------------------------------------------------------------------------
// Pickle emission
// ---------------------------------------------------------------------------

class PickleBuilder {
   public:
    void proto(std::uint8_t version) {
        put(0x80);
        put(version);
    }
    void stop() { put('.'); }
    void mark() { put('('); }
    void empty_tuple() { put(')'); }
    void empty_dict() { put('}'); }
    void tuple() { put('t'); }
    void tuple1() { put(0x85); }
    void tuple2() { put(0x86); }
    void tuple3() { put(0x87); }
    void setitems() { put('u'); }
    void setitem() { put('s'); }
    void reduce() { put('R'); }
    void persid() { put('Q'); }
    void build() { put('b'); }
    void newfalse() { put(0x89); }
    void newtrue() { put(0x88); }
    void none() { put('N'); }

    void global(const std::string& module, const std::string& name) {
        put('c');
        raw(module);
        put('\n');
        raw(name);
        put('\n');
    }

    void unicode(const std::string& s) {
        put('X');
        u32(static_cast<std::uint32_t>(s.size()));
        raw(s);
    }

    /// Smallest integer encoding that fits, like the real pickler.
    void integer(long long v) {
        if (v >= 0 && v < 256) {
            put('K');
            put(static_cast<std::uint8_t>(v));
        } else if (v >= 0 && v < 65536) {
            put('M');
            put(static_cast<std::uint8_t>(v & 0xff));
            put(static_cast<std::uint8_t>((v >> 8) & 0xff));
        } else {
            put('J');
            u32(static_cast<std::uint32_t>(static_cast<std::int32_t>(v)));
        }
    }

    void binput(std::uint32_t index) {
        if (index < 256) {
            put('q');
            put(static_cast<std::uint8_t>(index));
        } else {
            put('r');
            u32(index);
        }
    }

    void binget(std::uint32_t index) {
        if (index < 256) {
            put('h');
            put(static_cast<std::uint8_t>(index));
        } else {
            put('j');
            u32(index);
        }
    }

    /// Raw opcode escape hatch, for the unsupported-opcode path.
    void opcode(std::uint8_t op) { put(op); }

    [[nodiscard]] const std::vector<std::uint8_t>& bytes() const { return out_; }

   private:
    void put(std::uint8_t b) { out_.push_back(b); }
    void raw(const std::string& s) { out_.insert(out_.end(), s.begin(), s.end()); }
    void u32(std::uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            put(static_cast<std::uint8_t>((v >> (8 * i)) & 0xff));
        }
    }

    std::vector<std::uint8_t> out_;
};

// ---------------------------------------------------------------------------
// Tensor description
// ---------------------------------------------------------------------------

struct FakeTensor {
    std::string name;
    std::string storage_key;
    std::vector<std::size_t> shape;
    /// Number of elements in the backing storage entry.
    std::size_t storage_numel = 0;
    /// Element offset into that storage.
    std::size_t storage_offset = 0;
    /// Override the emitted strides; empty means emit contiguous ones.
    std::vector<long long> strides;
    /// "FloatStorage", "HalfStorage", "BFloat16Storage", "DoubleStorage".
    std::string storage_type = "FloatStorage";
    /// Use TUPLE1/2/3 rather than MARK ... TUPLE for the size and stride.
    bool short_tuples = false;
};

/// Emit a manifest for a list of tensors.
///
/// `rebuild_global` is the name called for each tensor; overriding it exercises
/// the reader's refusal to invoke globals it does not recognise. When
/// `with_metadata` is set, a `_metadata` entry is attached through BUILD, the
/// way a real state dict carries it.
[[nodiscard]] inline std::vector<std::uint8_t> build_manifest(
    const std::vector<FakeTensor>& tensors, bool with_metadata = true,
    const std::string& rebuild_global = "_rebuild_tensor_v2") {
    PickleBuilder p;
    p.proto(2);
    p.global("collections", "OrderedDict");
    p.binput(0);
    p.empty_tuple();
    p.reduce();
    p.binput(1);
    p.mark();

    for (const FakeTensor& t : tensors) {
        p.unicode(t.name);
        p.global("torch._utils", rebuild_global);

        p.mark();  // arguments to _rebuild_tensor_v2

        // storage: BINPERSID over ('storage', <Type>, key, 'cpu', numel)
        p.mark();
        p.unicode("storage");
        p.global("torch", t.storage_type);
        p.unicode(t.storage_key);
        p.unicode("cpu");
        p.integer(static_cast<long long>(t.storage_numel));
        p.tuple();
        p.persid();

        p.integer(static_cast<long long>(t.storage_offset));

        const auto emit_tuple = [&](const std::vector<long long>& values) {
            if (t.short_tuples && !values.empty() && values.size() <= 3) {
                for (const long long v : values) {
                    p.integer(v);
                }
                if (values.size() == 1) {
                    p.tuple1();
                } else if (values.size() == 2) {
                    p.tuple2();
                } else {
                    p.tuple3();
                }
            } else {
                p.mark();
                for (const long long v : values) {
                    p.integer(v);
                }
                p.tuple();
            }
        };

        std::vector<long long> size;
        size.reserve(t.shape.size());
        for (const std::size_t d : t.shape) {
            size.push_back(static_cast<long long>(d));
        }
        emit_tuple(size);

        std::vector<long long> stride = t.strides;
        if (stride.empty()) {
            stride.assign(t.shape.size(), 1);
            for (std::size_t i = t.shape.size(); i-- > 1;) {
                stride[i - 1] = stride[i] * static_cast<long long>(t.shape[i]);
            }
        }
        emit_tuple(stride);

        p.newfalse();   // requires_grad
        p.empty_dict(); // backward_hooks
        p.tuple();      // close the argument tuple
        p.reduce();     // -> tensor
    }

    p.setitems();

    if (with_metadata) {
        // A real state dict attaches `{'_metadata': {...}}` via BUILD. Nothing
        // in the reader needs it, so it must be skipped rather than parsed.
        p.empty_dict();
        p.unicode("_metadata");
        p.empty_dict();
        p.setitem();
        p.build();
    }

    p.stop();
    return p.bytes();
}

// ---------------------------------------------------------------------------
// ZIP emission
// ---------------------------------------------------------------------------

struct ZipMember {
    std::string name;
    std::vector<std::uint8_t> data;
    /// Non-zero claims a compression method the reader must reject.
    std::uint16_t method = 0;
};

/// Assemble a ZIP archive from stored members.
[[nodiscard]] inline std::vector<std::uint8_t> build_zip(const std::vector<ZipMember>& members) {
    std::vector<std::uint8_t> out;
    const auto u16 = [&out](std::uint16_t v) {
        out.push_back(static_cast<std::uint8_t>(v & 0xff));
        out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xff));
    };
    const auto u32 = [&out](std::uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xff));
        }
    };

    std::vector<std::uint32_t> local_offsets;
    for (const ZipMember& m : members) {
        local_offsets.push_back(static_cast<std::uint32_t>(out.size()));
        u32(0x04034b50);  // local header signature
        u16(20);          // version needed
        u16(0);           // flags
        u16(m.method);
        u16(0);  // mod time
        u16(0);  // mod date
        u32(0);  // crc32 -- not verified by the reader
        u32(static_cast<std::uint32_t>(m.data.size()));
        u32(static_cast<std::uint32_t>(m.data.size()));
        u16(static_cast<std::uint16_t>(m.name.size()));
        u16(0);  // extra length
        out.insert(out.end(), m.name.begin(), m.name.end());
        out.insert(out.end(), m.data.begin(), m.data.end());
    }

    const std::uint32_t cd_start = static_cast<std::uint32_t>(out.size());
    for (std::size_t i = 0; i < members.size(); ++i) {
        const ZipMember& m = members[i];
        u32(0x02014b50);  // central directory signature
        u16(20);          // version made by
        u16(20);          // version needed
        u16(0);           // flags
        u16(m.method);
        u16(0);  // mod time
        u16(0);  // mod date
        u32(0);  // crc32
        u32(static_cast<std::uint32_t>(m.data.size()));
        u32(static_cast<std::uint32_t>(m.data.size()));
        u16(static_cast<std::uint16_t>(m.name.size()));
        u16(0);  // extra length
        u16(0);  // comment length
        u16(0);  // disk number start
        u16(0);  // internal attributes
        u32(0);  // external attributes
        u32(local_offsets[i]);
        out.insert(out.end(), m.name.begin(), m.name.end());
    }
    const std::uint32_t cd_size = static_cast<std::uint32_t>(out.size()) - cd_start;

    u32(0x06054b50);  // end of central directory
    u16(0);           // this disk
    u16(0);           // disk with the central directory
    u16(static_cast<std::uint16_t>(members.size()));
    u16(static_cast<std::uint16_t>(members.size()));
    u32(cd_size);
    u32(cd_start);
    u16(0);  // comment length
    return out;
}

/// Raw little-endian f32 bytes.
[[nodiscard]] inline std::vector<std::uint8_t> f32_bytes(const std::vector<float>& values) {
    std::vector<std::uint8_t> out(values.size() * 4);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[i], 4);
        for (int b = 0; b < 4; ++b) {
            out[i * 4 + static_cast<std::size_t>(b)] =
                static_cast<std::uint8_t>((bits >> (8 * b)) & 0xff);
        }
    }
    return out;
}

/// Raw little-endian u16 bytes, for f16 and bf16 storages.
[[nodiscard]] inline std::vector<std::uint8_t> u16_bytes(const std::vector<std::uint16_t>& values) {
    std::vector<std::uint8_t> out(values.size() * 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i * 2] = static_cast<std::uint8_t>(values[i] & 0xff);
        out[i * 2 + 1] = static_cast<std::uint8_t>((values[i] >> 8) & 0xff);
    }
    return out;
}

/// Raw little-endian f64 bytes.
[[nodiscard]] inline std::vector<std::uint8_t> f64_bytes(const std::vector<double>& values) {
    std::vector<std::uint8_t> out(values.size() * 8);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &values[i], 8);
        for (int b = 0; b < 8; ++b) {
            out[i * 8 + static_cast<std::size_t>(b)] =
                static_cast<std::uint8_t>((bits >> (8 * b)) & 0xff);
        }
    }
    return out;
}

/// Write bytes to `path`, returning false on any I/O failure.
[[nodiscard]] inline bool write_file(const std::string& path,
                                     const std::vector<std::uint8_t>& bytes) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = bytes.empty() || std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    std::fclose(f);
    return ok;
}

}  // namespace rt::testing
