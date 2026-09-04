#include "rt/torch_pickle.hpp"

#include <cstdio>
#include <cstring>
#include <optional>
#include <span>
#include <utility>

#include "rt/bf16.hpp"

namespace rt {

namespace {

// ---------------------------------------------------------------------------
// Byte reading
// ---------------------------------------------------------------------------

[[nodiscard]] std::uint16_t rd_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) | static_cast<std::uint16_t>(p[1] << 8);
}

[[nodiscard]] std::uint32_t rd_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

[[nodiscard]] std::uint64_t rd_u64(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | p[static_cast<std::size_t>(i)];
    }
    return v;
}

/// RAII wrapper so every error path closes the file.
class File {
   public:
    explicit File(std::FILE* f) : f_(f) {}
    ~File() {
        if (f_ != nullptr) {
            std::fclose(f_);
        }
    }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    File(File&&) = delete;
    File& operator=(File&&) = delete;

    [[nodiscard]] std::FILE* get() const { return f_; }

   private:
    std::FILE* f_;
};

[[nodiscard]] Result<std::vector<std::uint8_t>> read_at(std::FILE* f, std::uint64_t offset,
                                                        std::size_t len) {
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0) {
        return err("torch_pickle: seek to " + std::to_string(offset) + " failed");
    }
    std::vector<std::uint8_t> buf(len);
    if (len > 0 && std::fread(buf.data(), 1, len, f) != len) {
        return err("torch_pickle: short read of " + std::to_string(len) + " bytes at " +
                   std::to_string(offset));
    }
    return buf;
}

[[nodiscard]] Result<std::uint64_t> file_size(std::FILE* f) {
    if (std::fseek(f, 0, SEEK_END) != 0) {
        return err("torch_pickle: seek to end failed");
    }
    const long n = std::ftell(f);
    if (n < 0) {
        return err("torch_pickle: ftell failed");
    }
    return static_cast<std::uint64_t>(n);
}

// ---------------------------------------------------------------------------
// ZIP central directory
// ---------------------------------------------------------------------------

constexpr std::uint32_t kEocdSig = 0x06054b50;    // "PK\5\6"
constexpr std::uint32_t kEocd64Sig = 0x06064b50;  // "PK\6\6"
constexpr std::uint32_t kCentralSig = 0x02014b50; // "PK\1\2"
constexpr std::uint32_t kLocalSig = 0x04034b50;   // "PK\3\4"
constexpr std::uint16_t kMethodStored = 0;

/// Where the central directory lives, and how many entries it has.
struct CentralDirInfo {
    std::uint64_t offset = 0;
    std::uint64_t count = 0;
};

/// Find the End Of Central Directory record by scanning backwards.
///
/// The EOCD sits at the end of the file but is followed by a variable-length
/// comment, so its position is not fixed and has to be searched for. 64 KiB is
/// the maximum a comment can be, so that bounds the scan.
[[nodiscard]] Result<CentralDirInfo> find_central_dir(std::FILE* f) {
    RT_TRY(total, file_size(f));
    constexpr std::uint64_t kMaxComment = 65535 + 22;
    const std::uint64_t window = total < kMaxComment ? total : kMaxComment;
    if (window < 22) {
        return err("torch_pickle: file too small to be a ZIP archive");
    }
    RT_TRY(tail, read_at(f, total - window, static_cast<std::size_t>(window)));

    std::optional<std::size_t> eocd;
    for (std::size_t back = 22; back <= tail.size(); ++back) {
        const std::size_t pos = tail.size() - back;
        if (rd_u32(tail.data() + pos) == kEocdSig) {
            eocd = pos;
            break;
        }
    }
    if (!eocd) {
        return err("torch_pickle: no ZIP end-of-central-directory record found");
    }

    const std::uint8_t* e = tail.data() + *eocd;
    CentralDirInfo info;
    info.count = rd_u16(e + 10);
    info.offset = rd_u32(e + 16);

    // A ZIP64 archive parks 0xffff / 0xffffffff in those fields and puts the
    // real values in a separate record. Checkpoints reach this size easily.
    if (info.count == 0xffff || info.offset == 0xffffffffu) {
        std::optional<std::size_t> z64;
        for (std::size_t pos = 0; pos + 56 <= tail.size(); ++pos) {
            if (rd_u32(tail.data() + pos) == kEocd64Sig) {
                z64 = pos;
                break;
            }
        }
        if (!z64) {
            return err("torch_pickle: ZIP64 archive without a ZIP64 EOCD record");
        }
        const std::uint8_t* z = tail.data() + *z64;
        info.count = rd_u64(z + 32);
        info.offset = rd_u64(z + 48);
    }
    return info;
}

/// Resolve an entry's data offset by reading its local header.
[[nodiscard]] Result<std::uint64_t> local_data_offset(std::FILE* f, std::uint64_t header_offset,
                                                      const std::string& name) {
    RT_TRY(hdr, read_at(f, header_offset, 30));
    if (rd_u32(hdr.data()) != kLocalSig) {
        return err("torch_pickle: bad local header signature for '" + name + "'");
    }
    const std::uint16_t method = rd_u16(hdr.data() + 8);
    if (method != kMethodStored) {
        return err("torch_pickle: entry '" + name + "' uses compression method " +
                   std::to_string(method) + "; only stored (0) is supported");
    }
    const std::uint16_t name_len = rd_u16(hdr.data() + 26);
    const std::uint16_t extra_len = rd_u16(hdr.data() + 28);
    return header_offset + 30 + name_len + extra_len;
}

}  // namespace

Result<std::vector<TorchZipEntry>> torch_zip_entries(const std::string& path) {
    const File file(std::fopen(path.c_str(), "rb"));
    if (file.get() == nullptr) {
        return err("torch_pickle: cannot open " + path);
    }
    std::FILE* f = file.get();

    RT_TRY(dir, find_central_dir(f));

    std::vector<TorchZipEntry> entries;
    entries.reserve(static_cast<std::size_t>(dir.count));

    std::uint64_t cursor = dir.offset;
    for (std::uint64_t i = 0; i < dir.count; ++i) {
        RT_TRY(head, read_at(f, cursor, 46));
        if (rd_u32(head.data()) != kCentralSig) {
            return err("torch_pickle: bad central directory signature at entry " +
                       std::to_string(i));
        }
        const std::uint16_t method = rd_u16(head.data() + 10);
        std::uint64_t comp_size = rd_u32(head.data() + 20);
        std::uint64_t uncomp_size = rd_u32(head.data() + 24);
        const std::uint16_t name_len = rd_u16(head.data() + 28);
        const std::uint16_t extra_len = rd_u16(head.data() + 30);
        const std::uint16_t comment_len = rd_u16(head.data() + 32);
        std::uint64_t local_offset = rd_u32(head.data() + 42);

        RT_TRY(name_bytes, read_at(f, cursor + 46, name_len));
        std::string name(reinterpret_cast<const char*>(name_bytes.data()), name_len);

        if (method != kMethodStored) {
            return err("torch_pickle: entry '" + name + "' uses compression method " +
                       std::to_string(method) + "; only stored (0) is supported");
        }

        // ZIP64 extra field (id 0x0001) carries the real sizes and offset when
        // the 32-bit fields are saturated. Values appear in a fixed order, but
        // only those that overflowed are present.
        if (uncomp_size == 0xffffffffu || comp_size == 0xffffffffu ||
            local_offset == 0xffffffffu) {
            RT_TRY(extra, read_at(f, cursor + 46 + name_len, extra_len));
            std::size_t pos = 0;
            bool found = false;
            while (pos + 4 <= extra.size()) {
                const std::uint16_t id = rd_u16(extra.data() + pos);
                const std::uint16_t len = rd_u16(extra.data() + pos + 2);
                if (id == 0x0001) {
                    std::size_t at = pos + 4;
                    const std::size_t end = at + len;
                    if (uncomp_size == 0xffffffffu && at + 8 <= end && at + 8 <= extra.size()) {
                        uncomp_size = rd_u64(extra.data() + at);
                        at += 8;
                    }
                    if (comp_size == 0xffffffffu && at + 8 <= end && at + 8 <= extra.size()) {
                        comp_size = rd_u64(extra.data() + at);
                        at += 8;
                    }
                    if (local_offset == 0xffffffffu && at + 8 <= end && at + 8 <= extra.size()) {
                        local_offset = rd_u64(extra.data() + at);
                    }
                    found = true;
                    break;
                }
                pos += 4 + len;
            }
            if (!found) {
                return err("torch_pickle: entry '" + name +
                           "' needs a ZIP64 extra field but has none");
            }
        }

        if (comp_size != uncomp_size) {
            return err("torch_pickle: entry '" + name + "' is not stored verbatim");
        }

        RT_TRY(data_off, local_data_offset(f, local_offset, name));

        TorchZipEntry entry;
        entry.name = std::move(name);
        entry.data_offset = data_off;
        entry.size = uncomp_size;
        entries.push_back(std::move(entry));

        cursor += 46 + name_len + extra_len + comment_len;
    }

    return entries;
}

// =============================================================================
// Pickle interpreter
// =============================================================================

namespace {

/// Storage element types this reader understands, all widened to f32.
enum class StorageDtype { F32, F16, Bf16, F64 };

[[nodiscard]] std::optional<StorageDtype> dtype_from_global(std::string_view name) {
    if (name == "torch FloatStorage") {
        return StorageDtype::F32;
    }
    if (name == "torch HalfStorage") {
        return StorageDtype::F16;
    }
    if (name == "torch BFloat16Storage") {
        return StorageDtype::Bf16;
    }
    if (name == "torch DoubleStorage") {
        return StorageDtype::F64;
    }
    return std::nullopt;
}

[[nodiscard]] std::size_t dtype_bytes(StorageDtype d) {
    switch (d) {
        case StorageDtype::F32:
            return 4;
        case StorageDtype::F16:
        case StorageDtype::Bf16:
            return 2;
        case StorageDtype::F64:
            return 8;
    }
    return 0;
}

/// One value on the interpreter's stack.
///
/// A flat tagged struct rather than a variant: the set of shapes is closed and
/// small, and every field is cheap to leave empty.
struct Value {
    enum class Kind {
        None,
        Mark,
        Int,
        Bool,
        Str,
        Tuple,
        Dict,
        Global,
        Storage,
        Tensor,
    };

    Kind kind = Kind::None;
    long long integer = 0;
    bool boolean = false;
    /// Str payload, or a global's "module name" pair joined by a space.
    std::string text;
    std::vector<Value> items;
    std::vector<std::pair<Value, Value>> entries;

    // Storage
    StorageDtype dtype = StorageDtype::F32;
    std::string storage_key;
    std::size_t storage_numel = 0;

    // Tensor
    std::vector<std::size_t> shape;
    std::size_t storage_offset = 0;
};

/// Interprets the closed subset of pickle that `torch.save` emits.
class PickleMachine {
   public:
    PickleMachine(std::span<const std::uint8_t> bytes) : b_(bytes) {}

    /// Run to STOP and return the final value.
    [[nodiscard]] Result<Value> run();

   private:
    [[nodiscard]] Result<std::uint8_t> byte() {
        if (pos_ >= b_.size()) {
            return err("torch_pickle: pickle stream ended mid-opcode");
        }
        return b_[pos_++];
    }

    [[nodiscard]] Result<std::uint32_t> u32() {
        if (pos_ + 4 > b_.size()) {
            return err("torch_pickle: pickle stream ended mid-u32");
        }
        const std::uint32_t v = rd_u32(b_.data() + pos_);
        pos_ += 4;
        return v;
    }

    [[nodiscard]] Result<std::string> line() {
        const std::size_t start = pos_;
        while (pos_ < b_.size() && b_[pos_] != '\n') {
            ++pos_;
        }
        if (pos_ >= b_.size()) {
            return err("torch_pickle: unterminated pickle text field");
        }
        std::string s(reinterpret_cast<const char*>(b_.data() + start), pos_ - start);
        ++pos_;  // consume '\n'
        return s;
    }

    [[nodiscard]] Result<Value> pop() {
        if (stack_.empty()) {
            return err("torch_pickle: pickle stack underflow");
        }
        Value v = std::move(stack_.back());
        stack_.pop_back();
        return v;
    }

    void push(Value v) { stack_.push_back(std::move(v)); }

    [[nodiscard]] Result<void> memo_put(std::size_t index) {
        if (stack_.empty()) {
            return err("torch_pickle: memo put with an empty stack");
        }
        if (index >= memo_.size()) {
            memo_.resize(index + 1);
        }
        memo_[index] = stack_.back();
        return {};
    }

    [[nodiscard]] Result<void> memo_get(std::size_t index) {
        if (index >= memo_.size()) {
            return err("torch_pickle: memo get of unset slot " + std::to_string(index));
        }
        stack_.push_back(memo_[index]);
        return {};
    }

    /// Pop everything above the most recent MARK, dropping the MARK itself.
    [[nodiscard]] Result<std::vector<Value>> pop_to_mark() {
        std::size_t at = stack_.size();
        while (at > 0 && stack_[at - 1].kind != Value::Kind::Mark) {
            --at;
        }
        if (at == 0) {
            return err("torch_pickle: no MARK on the pickle stack");
        }
        std::vector<Value> out(std::make_move_iterator(stack_.begin() + static_cast<long>(at)),
                               std::make_move_iterator(stack_.end()));
        stack_.resize(at - 1);  // also drops the MARK
        return out;
    }

    [[nodiscard]] Result<void> do_reduce();
    [[nodiscard]] Result<void> do_persid();
    [[nodiscard]] Result<Value> build_tensor(const std::vector<Value>& args);

    std::span<const std::uint8_t> b_;
    std::size_t pos_ = 0;
    std::vector<Value> stack_;
    std::vector<Value> memo_;
};

Result<Value> PickleMachine::build_tensor(const std::vector<Value>& args) {
    // _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad,
    //                    backward_hooks, metadata=None)
    if (args.size() < 4) {
        return err("torch_pickle: _rebuild_tensor_v2 needs at least 4 arguments, got " +
                   std::to_string(args.size()));
    }
    if (args[0].kind != Value::Kind::Storage) {
        return err("torch_pickle: _rebuild_tensor_v2 argument 0 is not a storage");
    }
    if (args[1].kind != Value::Kind::Int) {
        return err("torch_pickle: _rebuild_tensor_v2 storage_offset is not an integer");
    }
    if (args[2].kind != Value::Kind::Tuple || args[3].kind != Value::Kind::Tuple) {
        return err("torch_pickle: _rebuild_tensor_v2 size/stride are not tuples");
    }
    if (args[2].items.size() != args[3].items.size()) {
        return err("torch_pickle: _rebuild_tensor_v2 size and stride rank differ");
    }

    Value t = args[0];  // carry storage key, dtype, numel
    t.kind = Value::Kind::Tensor;
    t.storage_offset = static_cast<std::size_t>(args[1].integer);

    for (const Value& d : args[2].items) {
        if (d.kind != Value::Kind::Int || d.integer < 0) {
            return err("torch_pickle: _rebuild_tensor_v2 has a non-integer dimension");
        }
        t.shape.push_back(static_cast<std::size_t>(d.integer));
    }

    // Only contiguous tensors. A non-contiguous view would need its strides
    // honoured on read; silently ignoring them would scramble the weight.
    std::size_t expect = 1;
    for (std::size_t i = t.shape.size(); i-- > 0;) {
        const Value& s = args[3].items[i];
        if (s.kind != Value::Kind::Int) {
            return err("torch_pickle: _rebuild_tensor_v2 has a non-integer stride");
        }
        // A dimension of length 1 can carry any stride without changing the
        // layout, so it is not worth rejecting.
        if (t.shape[i] != 1 && static_cast<std::size_t>(s.integer) != expect) {
            return err("torch_pickle: non-contiguous tensor (stride " +
                       std::to_string(s.integer) + " at dim " + std::to_string(i) +
                       ", expected " + std::to_string(expect) + ")");
        }
        expect *= t.shape[i];
    }

    return t;
}

Result<void> PickleMachine::do_reduce() {
    RT_TRY(args, pop());
    RT_TRY(callable, pop());
    if (callable.kind != Value::Kind::Global) {
        return err("torch_pickle: REDUCE on a non-global callable");
    }
    if (args.kind != Value::Kind::Tuple) {
        return err("torch_pickle: REDUCE argument is not a tuple");
    }

    if (callable.text == "collections OrderedDict") {
        Value d;
        d.kind = Value::Kind::Dict;
        push(std::move(d));
        return {};
    }
    if (callable.text == "torch._utils _rebuild_tensor_v2") {
        RT_TRY(t, build_tensor(args.items));
        push(std::move(t));
        return {};
    }
    // Deliberately closed: a general unpickler would call whatever the file
    // names, which is the whole reason untrusted pickles are unsafe.
    return err("torch_pickle: refusing to call unrecognised global '" + callable.text + "'");
}

Result<void> PickleMachine::do_persid() {
    RT_TRY(pid, pop());
    if (pid.kind != Value::Kind::Tuple || pid.items.size() != 5) {
        return err("torch_pickle: persistent id is not a 5-tuple");
    }
    if (pid.items[0].kind != Value::Kind::Str || pid.items[0].text != "storage") {
        return err("torch_pickle: persistent id is not a storage reference");
    }
    if (pid.items[1].kind != Value::Kind::Global) {
        return err("torch_pickle: storage type is not a global");
    }
    const std::optional<StorageDtype> dtype = dtype_from_global(pid.items[1].text);
    if (!dtype) {
        return err("torch_pickle: unsupported storage type '" + pid.items[1].text +
                   "' (only Float, Half, BFloat16 and Double are handled)");
    }
    if (pid.items[2].kind != Value::Kind::Str) {
        return err("torch_pickle: storage key is not a string");
    }
    if (pid.items[4].kind != Value::Kind::Int) {
        return err("torch_pickle: storage element count is not an integer");
    }

    Value s;
    s.kind = Value::Kind::Storage;
    s.dtype = *dtype;
    s.storage_key = pid.items[2].text;
    s.storage_numel = static_cast<std::size_t>(pid.items[4].integer);
    push(std::move(s));
    return {};
}

Result<Value> PickleMachine::run() {
    while (true) {
        RT_TRY(op, byte());
        switch (op) {
            case 0x80: {  // PROTO
                RT_TRY(proto, byte());
                (void)proto;
                break;
            }
            case '.': {  // STOP
                RT_TRY(top, pop());
                return top;
            }
            case '(': {  // MARK
                Value m;
                m.kind = Value::Kind::Mark;
                push(std::move(m));
                break;
            }
            case 'N': {  // NONE
                push(Value{});
                break;
            }
            case 0x88:    // NEWTRUE
            case 0x89: {  // NEWFALSE
                Value v;
                v.kind = Value::Kind::Bool;
                v.boolean = (op == 0x88);
                push(std::move(v));
                break;
            }
            case 'K': {  // BININT1
                RT_TRY(n, byte());
                Value v;
                v.kind = Value::Kind::Int;
                v.integer = n;
                push(std::move(v));
                break;
            }
            case 'M': {  // BININT2
                RT_TRY(lo, byte());
                RT_TRY(hi, byte());
                Value v;
                v.kind = Value::Kind::Int;
                v.integer = static_cast<long long>(lo) | (static_cast<long long>(hi) << 8);
                push(std::move(v));
                break;
            }
            case 'J': {  // BININT (signed)
                RT_TRY(n, u32());
                Value v;
                v.kind = Value::Kind::Int;
                v.integer = static_cast<std::int32_t>(n);
                push(std::move(v));
                break;
            }
            case 'X': {  // BINUNICODE
                RT_TRY(len, u32());
                if (pos_ + len > b_.size()) {
                    return err("torch_pickle: BINUNICODE runs past the end of the stream");
                }
                Value v;
                v.kind = Value::Kind::Str;
                v.text.assign(reinterpret_cast<const char*>(b_.data() + pos_), len);
                pos_ += len;
                push(std::move(v));
                break;
            }
            case 'c': {  // GLOBAL
                RT_TRY(module, line());
                RT_TRY(name, line());
                Value v;
                v.kind = Value::Kind::Global;
                v.text = module + " " + name;
                push(std::move(v));
                break;
            }
            case 'q': {  // BINPUT
                RT_TRY(i, byte());
                RT_TRY_VOID(memo_put(i));
                break;
            }
            case 'r': {  // LONG_BINPUT
                RT_TRY(i, u32());
                RT_TRY_VOID(memo_put(i));
                break;
            }
            case 'h': {  // BINGET
                RT_TRY(i, byte());
                RT_TRY_VOID(memo_get(i));
                break;
            }
            case 'j': {  // LONG_BINGET
                RT_TRY(i, u32());
                RT_TRY_VOID(memo_get(i));
                break;
            }
            case ')': {  // EMPTY_TUPLE
                Value v;
                v.kind = Value::Kind::Tuple;
                push(std::move(v));
                break;
            }
            case '}': {  // EMPTY_DICT
                Value v;
                v.kind = Value::Kind::Dict;
                push(std::move(v));
                break;
            }
            case 't': {  // TUPLE
                RT_TRY(items, pop_to_mark());
                Value v;
                v.kind = Value::Kind::Tuple;
                v.items = std::move(items);
                push(std::move(v));
                break;
            }
            case 0x85:    // TUPLE1
            case 0x86:    // TUPLE2
            case 0x87: {  // TUPLE3
                const std::size_t n = static_cast<std::size_t>(op) - 0x84;
                if (stack_.size() < n) {
                    return err("torch_pickle: TUPLE" + std::to_string(n) + " stack underflow");
                }
                Value v;
                v.kind = Value::Kind::Tuple;
                v.items.assign(std::make_move_iterator(stack_.end() - static_cast<long>(n)),
                               std::make_move_iterator(stack_.end()));
                stack_.resize(stack_.size() - n);
                push(std::move(v));
                break;
            }
            case 's': {  // SETITEM
                RT_TRY(value, pop());
                RT_TRY(key, pop());
                if (stack_.empty() || stack_.back().kind != Value::Kind::Dict) {
                    return err("torch_pickle: SETITEM with no dict beneath it");
                }
                stack_.back().entries.emplace_back(std::move(key), std::move(value));
                break;
            }
            case 'u': {  // SETITEMS
                RT_TRY(flat, pop_to_mark());
                if (flat.size() % 2 != 0) {
                    return err("torch_pickle: SETITEMS with an odd number of values");
                }
                if (stack_.empty() || stack_.back().kind != Value::Kind::Dict) {
                    return err("torch_pickle: SETITEMS with no dict beneath it");
                }
                for (std::size_t i = 0; i < flat.size(); i += 2) {
                    stack_.back().entries.emplace_back(std::move(flat[i]), std::move(flat[i + 1]));
                }
                break;
            }
            case 'R': {  // REDUCE
                RT_TRY_VOID(do_reduce());
                break;
            }
            case 'Q': {  // BINPERSID
                RT_TRY_VOID(do_persid());
                break;
            }
            case 'b': {  // BUILD
                // The state dict's `_metadata` arrives this way. Nothing here
                // needs it; drop the state and leave the object.
                RT_TRY(state, pop());
                (void)state;
                if (stack_.empty()) {
                    return err("torch_pickle: BUILD with no object beneath it");
                }
                break;
            }
            default: {
                constexpr char kHex[] = "0123456789abcdef";
                const std::string code{kHex[op >> 4], kHex[op & 0x0f]};
                return err("torch_pickle: unsupported pickle opcode 0x" + code + " at offset " +
                           std::to_string(pos_ - 1));
            }
        }
    }
}

}  // namespace

// =============================================================================
// Public loader
// =============================================================================

std::size_t TorchTensor::inner_size() const {
    if (shape.size() < 2) {
        return 1;
    }
    std::size_t n = 1;
    for (std::size_t i = 1; i < shape.size(); ++i) {
        n *= shape[i];
    }
    return n;
}

const TorchTensor* TorchStateDict::find(std::string_view name) const {
    const auto it = tensors.find(std::string(name));
    return it == tensors.end() ? nullptr : &it->second;
}

Result<const TorchTensor*> TorchStateDict::require(std::string_view name) const {
    const TorchTensor* t = find(name);
    if (t == nullptr) {
        return err("torch_pickle: checkpoint has no tensor '" + std::string(name) + "'");
    }
    return t;
}

Result<const TorchTensor*> TorchStateDict::require(std::string_view name,
                                                   const std::vector<std::size_t>& shape) const {
    RT_TRY(t, require(name));
    if (t->shape != shape) {
        const auto fmt = [](const std::vector<std::size_t>& s) {
            std::string out = "[";
            for (std::size_t i = 0; i < s.size(); ++i) {
                out += (i > 0 ? ", " : "") + std::to_string(s[i]);
            }
            return out + "]";
        };
        return err("torch_pickle: tensor '" + std::string(name) + "' has shape " + fmt(t->shape) +
                   ", expected " + fmt(shape));
    }
    return t;
}

Result<TorchStateDict> load_torch_state_dict(const std::string& path) {
    RT_TRY(entries, torch_zip_entries(path));

    // The archive prefix comes from the name torch.save was given, so find it
    // rather than assuming "pytorch_model/".
    std::string prefix;
    const TorchZipEntry* manifest = nullptr;
    for (const TorchZipEntry& e : entries) {
        if (e.name.size() >= 8 && e.name.compare(e.name.size() - 8, 8, "data.pkl") == 0) {
            prefix = e.name.substr(0, e.name.size() - 8);
            manifest = &e;
            break;
        }
    }
    if (manifest == nullptr) {
        return err("torch_pickle: archive has no data.pkl manifest");
    }

    std::map<std::string, const TorchZipEntry*> by_name;
    for (const TorchZipEntry& e : entries) {
        by_name.emplace(e.name, &e);
    }

    const File file(std::fopen(path.c_str(), "rb"));
    if (file.get() == nullptr) {
        return err("torch_pickle: cannot reopen " + path);
    }
    std::FILE* f = file.get();

    RT_TRY(pkl, read_at(f, manifest->data_offset, static_cast<std::size_t>(manifest->size)));

    PickleMachine machine(pkl);
    RT_TRY(root, machine.run());
    if (root.kind != Value::Kind::Dict) {
        return err("torch_pickle: manifest root is not a dict");
    }

    // One storage can back several tensors, so cache the decoded bytes.
    std::map<std::string, std::vector<float>> storage_cache;

    TorchStateDict out;
    for (const auto& [key, value] : root.entries) {
        if (key.kind != Value::Kind::Str || value.kind != Value::Kind::Tensor) {
            continue;  // `_metadata` and friends
        }

        std::size_t want = 1;
        for (const std::size_t d : value.shape) {
            want *= d;
        }

        auto cached = storage_cache.find(value.storage_key);
        if (cached == storage_cache.end()) {
            const std::string entry_name = prefix + "data/" + value.storage_key;
            const auto it = by_name.find(entry_name);
            if (it == by_name.end()) {
                return err("torch_pickle: tensor '" + key.text + "' references missing entry '" +
                           entry_name + "'");
            }
            const std::size_t elem = dtype_bytes(value.dtype);
            const std::uint64_t need = static_cast<std::uint64_t>(value.storage_numel) * elem;
            if (it->second->size < need) {
                return err("torch_pickle: entry '" + entry_name + "' holds " +
                           std::to_string(it->second->size) + " bytes, needs " +
                           std::to_string(need));
            }
            RT_TRY(raw, read_at(f, it->second->data_offset, static_cast<std::size_t>(need)));

            std::vector<float> floats(value.storage_numel);
            for (std::size_t i = 0; i < value.storage_numel; ++i) {
                const std::uint8_t* p = raw.data() + i * elem;
                switch (value.dtype) {
                    case StorageDtype::F32: {
                        std::uint32_t bits = rd_u32(p);
                        float v = 0.0f;
                        std::memcpy(&v, &bits, 4);
                        floats[i] = v;
                        break;
                    }
                    case StorageDtype::F16:
                        floats[i] = f16_to_f32(rd_u16(p));
                        break;
                    case StorageDtype::Bf16:
                        floats[i] = bf16_to_f32(rd_u16(p));
                        break;
                    case StorageDtype::F64: {
                        std::uint64_t bits = rd_u64(p);
                        double v = 0.0;
                        std::memcpy(&v, &bits, 8);
                        floats[i] = static_cast<float>(v);
                        break;
                    }
                }
            }
            cached = storage_cache.emplace(value.storage_key, std::move(floats)).first;
        }

        const std::vector<float>& storage = cached->second;
        if (value.storage_offset + want > storage.size()) {
            return err("torch_pickle: tensor '" + key.text + "' runs past its storage (offset " +
                       std::to_string(value.storage_offset) + " + " + std::to_string(want) +
                       " > " + std::to_string(storage.size()) + ")");
        }

        TorchTensor tensor;
        tensor.shape = value.shape;
        tensor.data.assign(storage.begin() + static_cast<long>(value.storage_offset),
                           storage.begin() + static_cast<long>(value.storage_offset + want));
        out.tensors.emplace(key.text, std::move(tensor));
    }

    if (out.tensors.empty()) {
        return err("torch_pickle: manifest contained no tensors");
    }
    return out;
}

}  // namespace rt
