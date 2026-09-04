#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "rt/bf16.hpp"
#include "rt/torch_pickle.hpp"
#include "test_helpers.hpp"
#include "torch_pickle_writer.hpp"

using namespace rt;
using rt::testing::approx;
using rt::testing::build_manifest;
using rt::testing::build_zip;
using rt::testing::f32_bytes;
using rt::testing::f64_bytes;
using rt::testing::FakeTensor;
using rt::testing::PickleBuilder;
using rt::testing::u16_bytes;
using rt::testing::write_file;
using rt::testing::ZipMember;

namespace {

/// A checkpoint written to a temp path, removed when the test finishes.
class TempCheckpoint {
   public:
    explicit TempCheckpoint(const std::string& tag)
        : path_("/tmp/rt_torch_pickle_" + tag + ".bin") {}
    ~TempCheckpoint() { std::remove(path_.c_str()); }
    TempCheckpoint(const TempCheckpoint&) = delete;
    TempCheckpoint& operator=(const TempCheckpoint&) = delete;

    [[nodiscard]] const std::string& path() const { return path_; }

    [[nodiscard]] bool write(const std::vector<ZipMember>& members) const {
        return write_file(path_, build_zip(members));
    }

   private:
    std::string path_;
};

/// A two-tensor checkpoint: a [2,3] matrix and a [4] vector.
[[nodiscard]] std::vector<ZipMember> simple_members() {
    std::vector<FakeTensor> tensors;
    tensors.push_back(FakeTensor{.name = "layer.weight",
                                 .storage_key = "0",
                                 .shape = {2, 3},
                                 .storage_numel = 6});
    tensors.push_back(
        FakeTensor{.name = "layer.bias", .storage_key = "1", .shape = {4}, .storage_numel = 4});
    return {
        ZipMember{.name = "archive/data.pkl", .data = build_manifest(tensors)},
        ZipMember{.name = "archive/data/0",
                  .data = f32_bytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})},
        ZipMember{.name = "archive/data/1", .data = f32_bytes({-1.0f, -2.0f, -3.0f, -4.0f})},
    };
}

}  // namespace

// =============================================================================
// ZIP container
// =============================================================================

TEST_CASE("torch_zip_entries lists names, sizes and data offsets", "[torch_pickle]") {
    const TempCheckpoint ckpt("zip_list");
    const std::vector<ZipMember> members = simple_members();
    REQUIRE(ckpt.write(members));

    const Result<std::vector<TorchZipEntry>> entries = torch_zip_entries(ckpt.path());
    REQUIRE(entries.has_value());
    REQUIRE(entries->size() == 3);
    REQUIRE((*entries)[0].name == "archive/data.pkl");
    REQUIRE((*entries)[1].name == "archive/data/0");
    REQUIRE((*entries)[1].size == 24);
    REQUIRE((*entries)[2].size == 16);

    // The data offset must point at the payload, not the local header, so the
    // first four bytes of entry 1 are the f32 1.0 pattern.
    std::FILE* f = std::fopen(ckpt.path().c_str(), "rb");
    REQUIRE(f != nullptr);
    REQUIRE(std::fseek(f, static_cast<long>((*entries)[1].data_offset), SEEK_SET) == 0);
    float first = 0.0f;
    REQUIRE(std::fread(&first, 4, 1, f) == 1);
    std::fclose(f);
    REQUIRE(approx(first, 1.0f));
}

TEST_CASE("torch_zip_entries rejects a compressed entry", "[torch_pickle]") {
    const TempCheckpoint ckpt("compressed");
    std::vector<ZipMember> members = simple_members();
    members[1].method = 8;  // deflate
    REQUIRE(ckpt.write(members));

    const Result<std::vector<TorchZipEntry>> entries = torch_zip_entries(ckpt.path());
    REQUIRE(!entries.has_value());
    REQUIRE(entries.error().find("compression method 8") != std::string::npos);
}

TEST_CASE("torch_zip_entries fails cleanly on a non-archive", "[torch_pickle]") {
    const TempCheckpoint ckpt("garbage");
    REQUIRE(write_file(ckpt.path(), std::vector<std::uint8_t>(512, 0x41)));
    const Result<std::vector<TorchZipEntry>> entries = torch_zip_entries(ckpt.path());
    REQUIRE(!entries.has_value());
}

TEST_CASE("torch_zip_entries fails cleanly on a missing file", "[torch_pickle]") {
    const Result<std::vector<TorchZipEntry>> entries =
        torch_zip_entries("/tmp/rt_definitely_not_here.bin");
    REQUIRE(!entries.has_value());
    REQUIRE(entries.error().find("cannot open") != std::string::npos);
}

// =============================================================================
// State dict
// =============================================================================

TEST_CASE("load_torch_state_dict reads shapes and values", "[torch_pickle]") {
    const TempCheckpoint ckpt("simple");
    REQUIRE(ckpt.write(simple_members()));

    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(sd.has_value());
    REQUIRE(sd->tensors.size() == 2);

    const TorchTensor* w = sd->find("layer.weight");
    REQUIRE(w != nullptr);
    REQUIRE(w->shape == std::vector<std::size_t>{2, 3});
    REQUIRE(w->numel() == 6);
    for (std::size_t i = 0; i < 6; ++i) {
        REQUIRE(approx(w->data[i], static_cast<float>(i + 1)));
    }

    const TorchTensor* b = sd->find("layer.bias");
    REQUIRE(b != nullptr);
    REQUIRE(b->shape == std::vector<std::size_t>{4});
    REQUIRE(approx(b->data[3], -4.0f));

    REQUIRE(sd->find("layer.absent") == nullptr);
}

TEST_CASE("load_torch_state_dict discovers the archive prefix", "[torch_pickle]") {
    // The directory name comes from whatever torch.save was called with, so a
    // hardcoded "pytorch_model/" would fail here.
    const TempCheckpoint ckpt("prefix");
    std::vector<FakeTensor> tensors;
    tensors.push_back(
        FakeTensor{.name = "w", .storage_key = "0", .shape = {2}, .storage_numel = 2});
    REQUIRE(ckpt.write({
        ZipMember{.name = "some_other_name/data.pkl", .data = build_manifest(tensors)},
        ZipMember{.name = "some_other_name/data/0", .data = f32_bytes({7.0f, 8.0f})},
    }));

    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(sd.has_value());
    REQUIRE(approx(sd->find("w")->data[1], 8.0f));
}

TEST_CASE("load_torch_state_dict shares one storage between tensors", "[torch_pickle]") {
    // Two views into the same storage at different offsets -- how weight-tied
    // or sliced parameters get saved.
    const TempCheckpoint ckpt("shared");
    std::vector<FakeTensor> tensors;
    tensors.push_back(FakeTensor{
        .name = "first", .storage_key = "0", .shape = {3}, .storage_numel = 6, .storage_offset = 0});
    tensors.push_back(FakeTensor{
        .name = "second", .storage_key = "0", .shape = {3}, .storage_numel = 6, .storage_offset = 3});
    REQUIRE(ckpt.write({
        ZipMember{.name = "a/data.pkl", .data = build_manifest(tensors)},
        ZipMember{.name = "a/data/0",
                  .data = f32_bytes({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f})},
    }));

    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(sd.has_value());
    REQUIRE(approx(sd->find("first")->data[0], 10.0f));
    REQUIRE(approx(sd->find("second")->data[0], 40.0f));
    REQUIRE(approx(sd->find("second")->data[2], 60.0f));
}

TEST_CASE("load_torch_state_dict accepts TUPLE1/2/3 shapes", "[torch_pickle]") {
    // The real pickler uses the short tuple opcodes for ranks 1-3, which is
    // every conv weight in a codec decoder.
    const TempCheckpoint ckpt("short_tuples");
    std::vector<FakeTensor> tensors;
    tensors.push_back(FakeTensor{.name = "rank1",
                                 .storage_key = "0",
                                 .shape = {2},
                                 .storage_numel = 2,
                                 .short_tuples = true});
    tensors.push_back(FakeTensor{.name = "rank3",
                                 .storage_key = "1",
                                 .shape = {2, 2, 2},
                                 .storage_numel = 8,
                                 .short_tuples = true});
    REQUIRE(ckpt.write({
        ZipMember{.name = "a/data.pkl", .data = build_manifest(tensors)},
        ZipMember{.name = "a/data/0", .data = f32_bytes({1.0f, 2.0f})},
        ZipMember{.name = "a/data/1",
                  .data = f32_bytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f})},
    }));

    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(sd.has_value());
    REQUIRE(sd->find("rank1")->shape == std::vector<std::size_t>{2});
    REQUIRE(sd->find("rank3")->shape == std::vector<std::size_t>{2, 2, 2});
    REQUIRE(approx(sd->find("rank3")->data[7], 8.0f));
}

TEST_CASE("load_torch_state_dict skips the _metadata entry", "[torch_pickle]") {
    const TempCheckpoint ckpt("metadata");
    REQUIRE(ckpt.write(simple_members()));
    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(sd.has_value());
    // build_manifest attaches `_metadata` by default; only real tensors land.
    REQUIRE(sd->tensors.size() == 2);
    REQUIRE(sd->find("_metadata") == nullptr);
}

TEST_CASE("load_torch_state_dict works without a _metadata entry", "[torch_pickle]") {
    const TempCheckpoint ckpt("no_metadata");
    std::vector<FakeTensor> tensors;
    tensors.push_back(
        FakeTensor{.name = "w", .storage_key = "0", .shape = {2}, .storage_numel = 2});
    REQUIRE(ckpt.write({
        ZipMember{.name = "a/data.pkl", .data = build_manifest(tensors, /*with_metadata=*/false)},
        ZipMember{.name = "a/data/0", .data = f32_bytes({1.0f, 2.0f})},
    }));
    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(sd.has_value());
    REQUIRE(sd->tensors.size() == 1);
}

// =============================================================================
// Storage dtypes
// =============================================================================

TEST_CASE("load_torch_state_dict widens f16, bf16 and f64 to f32", "[torch_pickle]") {
    const TempCheckpoint ckpt("dtypes");
    std::vector<FakeTensor> tensors;
    tensors.push_back(FakeTensor{.name = "half",
                                 .storage_key = "0",
                                 .shape = {2},
                                 .storage_numel = 2,
                                 .storage_type = "HalfStorage"});
    tensors.push_back(FakeTensor{.name = "bfloat",
                                 .storage_key = "1",
                                 .shape = {2},
                                 .storage_numel = 2,
                                 .storage_type = "BFloat16Storage"});
    tensors.push_back(FakeTensor{.name = "double",
                                 .storage_key = "2",
                                 .shape = {2},
                                 .storage_numel = 2,
                                 .storage_type = "DoubleStorage"});

    // 1.0 and -2.0 in each encoding.
    const std::vector<std::uint16_t> halves{0x3c00, 0xc000};
    const std::vector<std::uint16_t> bfloats{0x3f80, 0xc000};

    REQUIRE(ckpt.write({
        ZipMember{.name = "a/data.pkl", .data = build_manifest(tensors)},
        ZipMember{.name = "a/data/0", .data = u16_bytes(halves)},
        ZipMember{.name = "a/data/1", .data = u16_bytes(bfloats)},
        ZipMember{.name = "a/data/2", .data = f64_bytes({1.0, -2.0})},
    }));

    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(sd.has_value());
    for (const char* name : {"half", "bfloat", "double"}) {
        const TorchTensor* t = sd->find(name);
        REQUIRE(t != nullptr);
        REQUIRE(approx(t->data[0], 1.0f));
        REQUIRE(approx(t->data[1], -2.0f));
    }
}

TEST_CASE("load_torch_state_dict rejects an integer storage", "[torch_pickle]") {
    const TempCheckpoint ckpt("int_storage");
    std::vector<FakeTensor> tensors;
    tensors.push_back(FakeTensor{.name = "counts",
                                 .storage_key = "0",
                                 .shape = {2},
                                 .storage_numel = 2,
                                 .storage_type = "LongStorage"});
    REQUIRE(ckpt.write({
        ZipMember{.name = "a/data.pkl", .data = build_manifest(tensors)},
        ZipMember{.name = "a/data/0", .data = std::vector<std::uint8_t>(16, 0)},
    }));

    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(!sd.has_value());
    REQUIRE(sd.error().find("unsupported storage type") != std::string::npos);
}

// =============================================================================
// Rejections
// =============================================================================

TEST_CASE("load_torch_state_dict rejects a non-contiguous tensor", "[torch_pickle]") {
    // A transposed view: shape [2,3] with strides [1,2]. Ignoring the strides
    // would scramble the weight, so this has to fail rather than load.
    const TempCheckpoint ckpt("strided");
    std::vector<FakeTensor> tensors;
    tensors.push_back(FakeTensor{.name = "w",
                                 .storage_key = "0",
                                 .shape = {2, 3},
                                 .storage_numel = 6,
                                 .strides = {1, 2}});
    REQUIRE(ckpt.write({
        ZipMember{.name = "a/data.pkl", .data = build_manifest(tensors)},
        ZipMember{.name = "a/data/0",
                  .data = f32_bytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})},
    }));

    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(!sd.has_value());
    REQUIRE(sd.error().find("non-contiguous") != std::string::npos);
}

TEST_CASE("load_torch_state_dict tolerates any stride on a length-1 dim", "[torch_pickle]") {
    // A dimension of length 1 can carry any stride without changing the
    // layout, which is how [C, 1, K] depthwise conv weights are often saved.
    const TempCheckpoint ckpt("unit_dim");
    std::vector<FakeTensor> tensors;
    tensors.push_back(FakeTensor{.name = "w",
                                 .storage_key = "0",
                                 .shape = {2, 1, 3},
                                 .storage_numel = 6,
                                 .strides = {3, 999, 1}});
    REQUIRE(ckpt.write({
        ZipMember{.name = "a/data.pkl", .data = build_manifest(tensors)},
        ZipMember{.name = "a/data/0",
                  .data = f32_bytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})},
    }));

    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(sd.has_value());
    REQUIRE(sd->find("w")->shape == std::vector<std::size_t>{2, 1, 3});
}

TEST_CASE("load_torch_state_dict refuses an unrecognised global", "[torch_pickle]") {
    // The security property: a general unpickler calls whatever constructor the
    // file names. This reader knows three globals and rejects everything else,
    // so a hostile file fails to parse instead of reaching for code nobody
    // vetted.
    const TempCheckpoint ckpt("bad_global");
    std::vector<FakeTensor> tensors;
    tensors.push_back(
        FakeTensor{.name = "w", .storage_key = "0", .shape = {2}, .storage_numel = 2});
    REQUIRE(ckpt.write({
        ZipMember{.name = "a/data.pkl",
                  .data = build_manifest(tensors, true, "definitely_not_rebuild_tensor")},
        ZipMember{.name = "a/data/0", .data = f32_bytes({1.0f, 2.0f})},
    }));

    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(!sd.has_value());
    REQUIRE(sd.error().find("refusing to call unrecognised global") != std::string::npos);
}

TEST_CASE("load_torch_state_dict rejects an unsupported opcode", "[torch_pickle]") {
    const TempCheckpoint ckpt("bad_opcode");
    PickleBuilder p;
    p.proto(2);
    p.opcode('I');  // INT -- text-mode integer, outside the supported set
    const std::vector<std::uint8_t> pkl = p.bytes();
    REQUIRE(ckpt.write({ZipMember{.name = "a/data.pkl", .data = pkl}}));

    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(!sd.has_value());
    REQUIRE(sd.error().find("unsupported pickle opcode 0x49") != std::string::npos);
}

TEST_CASE("load_torch_state_dict reports a missing storage entry", "[torch_pickle]") {
    const TempCheckpoint ckpt("missing_storage");
    std::vector<FakeTensor> tensors;
    tensors.push_back(
        FakeTensor{.name = "w", .storage_key = "7", .shape = {2}, .storage_numel = 2});
    REQUIRE(ckpt.write({
        ZipMember{.name = "a/data.pkl", .data = build_manifest(tensors)},
        ZipMember{.name = "a/data/0", .data = f32_bytes({1.0f, 2.0f})},
    }));

    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(!sd.has_value());
    REQUIRE(sd.error().find("missing entry") != std::string::npos);
}

TEST_CASE("load_torch_state_dict reports a storage that is too small", "[torch_pickle]") {
    const TempCheckpoint ckpt("short_storage");
    std::vector<FakeTensor> tensors;
    tensors.push_back(
        FakeTensor{.name = "w", .storage_key = "0", .shape = {8}, .storage_numel = 8});
    REQUIRE(ckpt.write({
        ZipMember{.name = "a/data.pkl", .data = build_manifest(tensors)},
        ZipMember{.name = "a/data/0", .data = f32_bytes({1.0f, 2.0f})},
    }));

    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(!sd.has_value());
    REQUIRE(sd.error().find("needs") != std::string::npos);
}

TEST_CASE("load_torch_state_dict reports a manifest with no tensors", "[torch_pickle]") {
    const TempCheckpoint ckpt("empty");
    REQUIRE(ckpt.write(
        {ZipMember{.name = "a/data.pkl", .data = build_manifest({}, /*with_metadata=*/true)}}));
    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(!sd.has_value());
    REQUIRE(sd.error().find("no tensors") != std::string::npos);
}

TEST_CASE("load_torch_state_dict reports an archive with no manifest", "[torch_pickle]") {
    const TempCheckpoint ckpt("no_manifest");
    REQUIRE(ckpt.write({ZipMember{.name = "a/data/0", .data = f32_bytes({1.0f})}}));
    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(!sd.has_value());
    REQUIRE(sd.error().find("no data.pkl") != std::string::npos);
}

// =============================================================================
// Lookup helpers
// =============================================================================

TEST_CASE("TorchStateDict::require names both shapes on a mismatch", "[torch_pickle]") {
    const TempCheckpoint ckpt("require");
    REQUIRE(ckpt.write(simple_members()));
    const Result<TorchStateDict> sd = load_torch_state_dict(ckpt.path());
    REQUIRE(sd.has_value());

    const Result<const TorchTensor*> ok = sd->require("layer.weight", {2, 3});
    REQUIRE(ok.has_value());

    const Result<const TorchTensor*> bad = sd->require("layer.weight", {3, 2});
    REQUIRE(!bad.has_value());
    REQUIRE(bad.error().find("[2, 3]") != std::string::npos);
    REQUIRE(bad.error().find("[3, 2]") != std::string::npos);

    const Result<const TorchTensor*> absent = sd->require("nope");
    REQUIRE(!absent.has_value());
    REQUIRE(absent.error().find("no tensor 'nope'") != std::string::npos);
}

TEST_CASE("TorchTensor::inner_size is the product past the first axis", "[torch_pickle]") {
    TorchTensor t;
    t.shape = {4};
    REQUIRE(t.inner_size() == 1);
    t.shape = {4, 5};
    REQUIRE(t.inner_size() == 5);
    t.shape = {1024, 512, 16};
    REQUIRE(t.inner_size() == 512 * 16);
}

// =============================================================================
// The real checkpoint
// =============================================================================

TEST_CASE("load_torch_state_dict reads the SNAC 24 kHz checkpoint", "[torch_pickle][.integration]") {
    // Skipped unless the weights have been fetched. Shapes are the ones the
    // decoder depends on, including the transposed convolution whose
    // weight-norm magnitude is indexed by *input* channel.
    const std::string path = "models/snac_24khz.bin";
    if (!std::filesystem::exists(path)) {
        SKIP("models/snac_24khz.bin not present");
    }

    const Result<TorchStateDict> sd = load_torch_state_dict(path);
    REQUIRE(sd.has_value());
    REQUIRE(sd->tensors.size() == 269);

    REQUIRE(sd->require("decoder.model.0.parametrizations.weight.original1", {768, 1, 7}));
    REQUIRE(sd->require("decoder.model.1.parametrizations.weight.original1", {1024, 768, 1}));
    REQUIRE(sd->require("decoder.model.2.block.1.parametrizations.weight.original1",
                        {1024, 512, 16}));
    REQUIRE(sd->require("decoder.model.2.block.1.parametrizations.weight.original0", {1024, 1, 1}));
    REQUIRE(sd->require("decoder.model.2.block.1.bias", {512}));
    REQUIRE(sd->require("decoder.model.2.block.0.alpha", {1, 1024, 1}));
    REQUIRE(sd->require("decoder.model.7.parametrizations.weight.original1", {1, 64, 7}));
    REQUIRE(sd->require("quantizer.quantizers.0.codebook.weight", {4096, 8}));
    REQUIRE(sd->require("quantizer.quantizers.2.out_proj.parametrizations.weight.original1",
                        {768, 8, 1}));

    // The NoiseBlock convolution has no bias, unlike every other conv here.
    REQUIRE(sd->find("decoder.model.2.block.2.linear.bias") == nullptr);
    REQUIRE(sd->require("decoder.model.2.block.2.linear.parametrizations.weight.original1",
                        {512, 512, 1}));

    // Weights should be finite and not all zero.
    const TorchTensor* alpha = sd->find("decoder.model.6.alpha");
    REQUIRE(alpha != nullptr);
    REQUIRE(alpha->numel() == 64);
    float sum_abs = 0.0f;
    for (const float v : alpha->data) {
        REQUIRE(std::isfinite(v));
        sum_abs += std::fabs(v);
    }
    REQUIRE(sum_abs > 0.0f);
}
