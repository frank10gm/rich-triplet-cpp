#pragma once

// =============================================================================
// PyTorch `.bin` checkpoint reader -- ZIP container + pickle state dict
// =============================================================================
//
// Every other weight format this project reads is a flat binary with a
// self-describing header: GGUF, safetensors, and the internal cache files all
// say where each tensor is and stop there. `torch.save` instead writes a ZIP
// archive whose manifest is a Python pickle, so reading it means implementing
// two formats.
//
// Some published weights only exist in this format. SNAC's 24 kHz codec, which
// Orpheus decodes its audio tokens with, ships `pytorch_model.bin` and nothing
// else -- no safetensors mirror. Hence this file.
//
// ## Container layout
//
//   pytorch_model/data.pkl     the manifest: a pickled OrderedDict
//   pytorch_model/byteorder    b"little"
//   pytorch_model/data/0       raw tensor bytes, one entry per storage
//   pytorch_model/data/1
//   ...
//
// The archive directory prefix comes from whatever name `torch.save` was
// given, so it is discovered from the `data.pkl` entry rather than assumed.
// Entries are **stored**, never deflated -- tensor bytes do not compress, so
// PyTorch does not try. This reader rejects a compressed entry instead of
// pulling in an inflate implementation for a case that does not arise.
//
// ## Manifest layout
//
// The pickle is small and closed. It builds one `collections.OrderedDict`,
// fills it with `(name, tensor)` pairs, and attaches a `_metadata` dict that
// nothing here needs. Each tensor is a call to
//
//   torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride,
//                                   requires_grad, backward_hooks, metadata=None)
//
// where `storage` arrives as a persistent id -- `('storage', <StorageType>,
// <key>, <device>, <numel>)` -- naming the `data/<key>` archive entry.
//
// So the interpreter needs a value stack, a memo, three recognised globals
// (`collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, and a storage
// type), and the 24 opcodes those constructs emit. It is not a general
// unpickler and must never become one: a general unpickler executes arbitrary
// constructors named by the file, which is exactly the property that makes
// loading untrusted pickles unsafe. Anything outside the recognised set is an
// error, so a hostile or merely unusual file fails to parse rather than
// reaching for a constructor nobody vetted.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "rt/result.hpp"

namespace rt {

// ---------------------------------------------------------------------------
// ZIP container
// ---------------------------------------------------------------------------

/// One stored entry in the archive.
struct TorchZipEntry {
    std::string name;
    /// Byte offset of the entry's data in the file, past its local header.
    std::uint64_t data_offset = 0;
    /// Uncompressed length in bytes; equals the stored length.
    std::uint64_t size = 0;
};

/// List every entry in a ZIP archive, resolving each one's true data offset.
///
/// The central directory records the offset of a *local header*, whose name and
/// extra fields can be longer than the central copy's, so the data offset has
/// to be computed by reading that header rather than by adding a fixed size.
///
/// Fails when any entry is compressed.
[[nodiscard]] Result<std::vector<TorchZipEntry>> torch_zip_entries(const std::string& path);

// ---------------------------------------------------------------------------
// Tensors
// ---------------------------------------------------------------------------

/// One tensor from a checkpoint, always converted to f32.
///
/// `data` is in row-major order, which is what a contiguous PyTorch tensor
/// already is -- non-contiguous tensors are rejected at load rather than
/// silently reordered.
struct TorchTensor {
    std::vector<std::size_t> shape;
    std::vector<float> data;

    [[nodiscard]] std::size_t numel() const { return data.size(); }

    /// Product of the trailing dimensions after the first, or 1 when the shape
    /// has fewer than two dimensions.
    ///
    /// This is the group size `weight_norm_combine` normalizes over, and the
    /// row length when a `[out, in, k]` conv weight is viewed as `[out, in*k]`.
    [[nodiscard]] std::size_t inner_size() const;
};

/// A loaded state dict, keyed by parameter name.
struct TorchStateDict {
    std::map<std::string, TorchTensor> tensors;

    /// Look up a parameter, or nullptr when it is absent.
    [[nodiscard]] const TorchTensor* find(std::string_view name) const;

    /// Look up a parameter, requiring an exact shape. Returns an error naming
    /// the parameter and both shapes on a mismatch -- a wrong shape in a codec
    /// decoder produces noise rather than a crash, so it is worth failing
    /// loudly at load.
    [[nodiscard]] Result<const TorchTensor*> require(
        std::string_view name, const std::vector<std::size_t>& shape) const;

    /// Look up a parameter of any shape, erroring when absent.
    [[nodiscard]] Result<const TorchTensor*> require(std::string_view name) const;
};

/// Read a `torch.save`d state dict.
///
/// Handles f32, f16, bf16 and f64 storages, converting every one to f32.
/// Rejects: compressed archive entries, integer and boolean storages,
/// non-contiguous tensors, and any pickle opcode or global outside the closed
/// set the format needs.
[[nodiscard]] Result<TorchStateDict> load_torch_state_dict(const std::string& path);

}  // namespace rt
