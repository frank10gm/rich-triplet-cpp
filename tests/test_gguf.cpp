#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <unistd.h>

#include "gguf_writer.hpp"
#include "rt/gguf.hpp"

using namespace rt;
using rt::testing::GgufWriter;

namespace {

/// A temp file that removes itself when the test scope ends.
struct TempFile {
    std::filesystem::path path;
    explicit TempFile(const char* stem)
        : path(std::filesystem::temp_directory_path() /
               (std::string(stem) + std::to_string(::getpid()) + ".gguf")) {}
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    [[nodiscard]] std::string str() const { return path.string(); }
};

std::vector<std::uint8_t> f32_bytes(const std::vector<float>& v) {
    std::vector<std::uint8_t> b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

void put_u16(std::vector<std::uint8_t>& b, std::uint16_t v) {
    b.push_back(static_cast<std::uint8_t>(v & 0xff));
    b.push_back(static_cast<std::uint8_t>(v >> 8));
}

/// Encode one Q4_0 block (18 bytes) from 32 values, using GGUF's split nibble
/// order and `(nibble - 8) * scale` dequantization.
std::vector<std::uint8_t> make_q4_0_block(const std::vector<float>& vals, float scale) {
    std::vector<std::uint8_t> b;
    put_u16(b, f32_to_f16(scale));
    for (std::size_t k = 0; k < 16; ++k) {
        const auto q = [&](std::size_t i) {
            const int n = static_cast<int>(std::lround(vals[i] / scale)) + 8;
            return static_cast<std::uint8_t>(std::clamp(n, 0, 15));
        };
        b.push_back(static_cast<std::uint8_t>(q(k) | (q(k + 16) << 4)));
    }
    return b;
}

/// Encode one Q8_0 block (34 bytes) from 32 values.
std::vector<std::uint8_t> make_q8_0_block(const std::vector<float>& vals, float scale) {
    std::vector<std::uint8_t> b;
    put_u16(b, f32_to_f16(scale));
    for (float v : vals) {
        const int q = std::clamp(static_cast<int>(std::lround(v / scale)), -127, 127);
        b.push_back(static_cast<std::uint8_t>(static_cast<std::int8_t>(q)));
    }
    return b;
}

}  // namespace

TEST_CASE("gguf header, metadata and tensor info round-trip", "[gguf]") {
    TempFile tmp("rt_gguf_header_");

    GgufWriter w;
    w.add_meta_u32("general.file_type", 15);
    w.add_meta_u64("general.alignment", 32);
    w.add_meta_str("general.architecture", "gemma3");
    w.add_meta_f32("gemma3.attention.layer_norm_rms_epsilon", 1e-6f);
    w.add_meta_str_array("tokenizer.ggml.tokens", {"<pad>", "hello", "world"});

    const std::vector<float> vals{1.0f, -2.0f, 3.5f, 0.25f, 8.0f, -0.5f};
    w.add_tensor("blk.0.attn_norm.weight", {3, 2}, GgufType::F32, f32_bytes(vals));
    REQUIRE(w.write(tmp.str()));

    const auto file = GgufFile::open(tmp.str());
    REQUIRE(file.has_value());

    REQUIRE(file->tensor_info.size() == 1);
    REQUIRE(file->metadata.at("general.architecture").as_str() == "gemma3");
    REQUIRE(file->metadata.at("general.file_type").as_u64() == 15);
    REQUIRE(file->metadata.at("gemma3.attention.layer_norm_rms_epsilon").as_f32() == 1e-6f);

    const auto* toks = file->metadata.at("tokenizer.ggml.tokens").as_array();
    REQUIRE(toks != nullptr);
    REQUIRE(toks->size() == 3);
    REQUIRE((*toks)[1].as_str() == "hello");

    const auto idx = file->find_tensor("blk.0.attn_norm.weight");
    REQUIRE(idx.has_value());
    const auto& info = file->tensor_info[*idx];
    REQUIRE(info.shape == std::vector<std::size_t>{3, 2});
    REQUIRE(info.gguf_type == GgufType::F32);
    REQUIRE(info.n_elements() == 6);

    const auto decoded = file->decode_f32(*idx);
    REQUIRE(decoded.has_value());
    REQUIRE(*decoded == vals);

    REQUIRE_FALSE(file->find_tensor("nope").has_value());
}

TEST_CASE("gguf rejects bad magic and version", "[gguf]") {
    TempFile tmp("rt_gguf_bad_");
    {
        std::ofstream f(tmp.str(), std::ios::binary);
        f << "NOPE";
        for (int i = 0; i < 64; ++i) f.put('\0');
    }
    const auto file = GgufFile::open(tmp.str());
    REQUIRE_FALSE(file.has_value());
    REQUIRE(file.error().find("not a GGUF file") != std::string::npos);

    const auto missing = GgufFile::open("/definitely/not/here.gguf");
    REQUIRE_FALSE(missing.has_value());
}

TEST_CASE("gguf type sizes", "[gguf]") {
    REQUIRE(gguf_byte_size(GgufType::F32, 10) == 40);
    REQUIRE(gguf_byte_size(GgufType::F16, 10) == 20);
    REQUIRE(gguf_byte_size(GgufType::Bf16, 10) == 20);
    REQUIRE(gguf_byte_size(GgufType::Q4_0, 32) == 18);
    REQUIRE(gguf_byte_size(GgufType::Q8_0, 32) == 34);
    REQUIRE(gguf_byte_size(GgufType::Q4K, 256) == 144);
    REQUIRE(gguf_byte_size(GgufType::Q5K, 256) == 176);
    REQUIRE(gguf_byte_size(GgufType::Q6K, 256) == 210);
    // Partial blocks round up.
    REQUIRE(gguf_byte_size(GgufType::Q4K, 257) == 288);
    // Types the reader cannot size.
    REQUIRE_FALSE(gguf_byte_size(GgufType::Q2K, 256).has_value());
    REQUIRE(gguf_type_from_u32(9999) == GgufType::Unknown);
}

TEST_CASE("gguf decodes F16 and BF16 tensors", "[gguf]") {
    TempFile tmp("rt_gguf_halves_");

    const std::vector<float> vals{0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -0.25f, 100.0f, -8.0f};
    std::vector<std::uint8_t> f16_data, bf16_data;
    for (float v : vals) {
        put_u16(f16_data, f32_to_f16(v));
        put_u16(bf16_data, f32_to_bf16(v));
    }

    GgufWriter w;
    w.add_tensor("h", {8}, GgufType::F16, f16_data);
    w.add_tensor("b", {8}, GgufType::Bf16, bf16_data);
    REQUIRE(w.write(tmp.str()));

    const auto file = GgufFile::open(tmp.str());
    REQUIRE(file.has_value());

    const auto h = file->decode_f16_to_f32(*file->find_tensor("h"));
    REQUIRE(h.has_value());
    for (std::size_t i = 0; i < vals.size(); ++i) {
        INFO("f16 element " << i);
        REQUIRE(std::fabs((*h)[i] - vals[i]) < std::fabs(vals[i]) * 0.002f + 1e-6f);
    }

    const auto b = file->decode_bf16(*file->find_tensor("b"));
    REQUIRE(b.has_value());
    REQUIRE(b->size() == vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i) {
        INFO("bf16 element " << i);
        REQUIRE(std::fabs(bf16_to_f32((*b)[i]) - vals[i]) < std::fabs(vals[i]) * 0.01f + 1e-6f);
    }
}

TEST_CASE("gguf decodes Q4_0 to f32 and to Q4Mat consistently", "[gguf]") {
    TempFile tmp("rt_gguf_q40_");

    // One block of 32 values, symmetric around zero so the (nibble-8) encoding
    // covers the full range.
    const float scale = 0.25f;
    std::vector<float> vals(32);
    for (std::size_t i = 0; i < 32; ++i) {
        vals[i] = (static_cast<float>(i) - 8.0f) * scale;
    }
    // Clamp to the representable range so the reference matches exactly.
    for (float& v : vals) {
        v = std::clamp(v, -8.0f * scale, 7.0f * scale);
    }

    GgufWriter w;
    // GGUF shape is [in, out]; 32x1 keeps a single block.
    w.add_tensor("w", {32, 1}, GgufType::Q4_0, make_q4_0_block(vals, scale));
    REQUIRE(w.write(tmp.str()));

    const auto file = GgufFile::open(tmp.str());
    REQUIRE(file.has_value());
    const std::size_t idx = *file->find_tensor("w");

    const auto flat = file->decode_q4_0_to_f32(idx);
    REQUIRE(flat.has_value());
    REQUIRE(flat->size() == 32);
    for (std::size_t i = 0; i < 32; ++i) {
        INFO("q4_0 flat element " << i);
        REQUIRE(std::fabs((*flat)[i] - vals[i]) <= scale * 0.5f + 1e-5f);
    }

    // The Q4Mat path re-packs the nibbles; dequantizing it must agree with the
    // flat decode element for element.
    const auto q4 = file->decode_q4_0_to_q4mat(idx);
    REQUIRE(q4.has_value());
    REQUIRE(q4->rows == 1);  // GGUF [32, 1] -> ours [1, 32]
    REQUIRE(q4->cols == 32);
    const Mat dq = q4->dequantize();
    for (std::size_t i = 0; i < 32; ++i) {
        INFO("q4_0 Q4Mat element " << i);
        REQUIRE(std::fabs(dq.data[i] - (*flat)[i]) < 1e-6f);
    }
}

TEST_CASE("gguf decodes Q8_0 to f32", "[gguf]") {
    TempFile tmp("rt_gguf_q80_");

    const float scale = 0.01f;
    std::vector<float> vals(32);
    for (std::size_t i = 0; i < 32; ++i) {
        vals[i] = (static_cast<float>(i) - 16.0f) * scale;
    }

    GgufWriter w;
    w.add_tensor("w", {32, 1}, GgufType::Q8_0, make_q8_0_block(vals, scale));
    REQUIRE(w.write(tmp.str()));

    const auto file = GgufFile::open(tmp.str());
    REQUIRE(file.has_value());
    const auto out = file->decode_q8_0_to_f32(*file->find_tensor("w"));
    REQUIRE(out.has_value());
    REQUIRE(out->size() == 32);
    // f16 rounding of the scale is the only error source here.
    for (std::size_t i = 0; i < 32; ++i) {
        INFO("q8_0 element " << i);
        REQUIRE(std::fabs((*out)[i] - vals[i]) < 1e-4f);
    }
}

TEST_CASE("gguf Q4K flat decode matches Q4KMat row decode", "[gguf]") {
    TempFile tmp("rt_gguf_q4k_");

    // Produce genuine Q4_K blocks with the quantizer, then read them back
    // through both GGUF decode paths -- they must agree bit for bit.
    const Mat src = Mat::from_fn(4, 256, [](std::size_t r, std::size_t c) {
        return std::sin(static_cast<float>(r * 256 + c) * 0.01f) * 0.8f;
    });
    const Q4KMat q = Q4KMat::quantize(src);

    GgufWriter w;
    // Ours is [rows=4, cols=256]; GGUF stores the transpose, [in=256, out=4].
    w.add_tensor("w", {256, 4}, GgufType::Q4K, q.blocks);
    REQUIRE(w.write(tmp.str()));

    const auto file = GgufFile::open(tmp.str());
    REQUIRE(file.has_value());
    const std::size_t idx = *file->find_tensor("w");

    const auto loaded = file->decode_q4k_to_q4kmat(idx);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->rows == 4);
    REQUIRE(loaded->cols == 256);
    REQUIRE(loaded->blocks == q.blocks);

    const auto flat = file->decode_q4k_to_f32(idx);
    REQUIRE(flat.has_value());
    REQUIRE(flat->size() == 4 * 256);

    std::vector<float> row(256);
    for (std::size_t r = 0; r < 4; ++r) {
        loaded->dequantize_row_into(r, row);
        for (std::size_t c = 0; c < 256; ++c) {
            INFO("q4k [" << r << "," << c << "]");
            REQUIRE((*flat)[r * 256 + c] == row[c]);
        }
    }

    // And the values still track the original matrix.
    float max_err = 0.0f;
    for (std::size_t i = 0; i < flat->size(); ++i) {
        max_err = std::max(max_err, std::fabs((*flat)[i] - src.data[i]));
    }
    INFO("q4k GGUF round-trip max error " << max_err);
    REQUIRE(max_err < 0.15f);
}

TEST_CASE("gguf Q6K decode reproduces a known block", "[gguf]") {
    TempFile tmp("rt_gguf_q6k_");

    // Build a Q6_K block by hand and check the decoder against the same
    // formula llama.cpp uses, so the lane interleaving is pinned down.
    std::vector<std::uint8_t> block(210, 0);
    std::mt19937 rng(1234);
    for (std::size_t i = 0; i < 192; ++i) {
        block[i] = static_cast<std::uint8_t>(rng() & 0xff);
    }
    for (std::size_t i = 0; i < 16; ++i) {
        block[192 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>((i * 7) % 61) - 30);
    }
    const float d = 0.0125f;
    block[208] = static_cast<std::uint8_t>(f32_to_f16(d) & 0xff);
    block[209] = static_cast<std::uint8_t>(f32_to_f16(d) >> 8);

    GgufWriter w;
    w.add_tensor("w", {256, 1}, GgufType::Q6K, block);
    REQUIRE(w.write(tmp.str()));

    const auto file = GgufFile::open(tmp.str());
    REQUIRE(file.has_value());
    const auto out = file->decode_q6k_to_f32(*file->find_tensor("w"));
    REQUIRE(out.has_value());
    REQUIRE(out->size() == 256);

    const float dd = f16_to_f32(static_cast<std::uint16_t>(block[208] | (block[209] << 8)));
    for (std::size_t j = 0; j < 2; ++j) {
        const std::uint8_t* ql = block.data() + j * 64;
        const std::uint8_t* qh = block.data() + 128 + j * 32;
        const auto* sc = reinterpret_cast<const std::int8_t*>(block.data() + 192 + j * 8);
        for (std::size_t l = 0; l < 32; ++l) {
            const std::size_t is = l / 16;
            const int q1 = static_cast<int>((ql[l] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const std::size_t base = j * 128;
            INFO("q6k lane l=" << l << " half=" << j);
            REQUIRE((*out)[base + l] == dd * static_cast<float>(sc[is]) * static_cast<float>(q1));
            REQUIRE((*out)[base + l + 64] ==
                    dd * static_cast<float>(sc[is + 4]) * static_cast<float>(q3));
        }
    }
}

TEST_CASE("gguf Q5K decode adds the high bit from qh", "[gguf]") {
    TempFile tmp("rt_gguf_q5k_");

    // scales/mins are all zero except the first pair, so the expected values
    // are easy to state: out = scale1 * (lo | hi<<4) - min1.
    std::vector<std::uint8_t> block(176, 0);
    const float d = 0.5f, dmin = 0.25f;
    block[0] = static_cast<std::uint8_t>(f32_to_f16(d) & 0xff);
    block[1] = static_cast<std::uint8_t>(f32_to_f16(d) >> 8);
    block[2] = static_cast<std::uint8_t>(f32_to_f16(dmin) & 0xff);
    block[3] = static_cast<std::uint8_t>(f32_to_f16(dmin) >> 8);
    block[4] = 3;   // scale for pair 0 = 3
    block[8] = 2;   // min for pair 0 = 2
    // qh bit 0 set for even l -> those elements get +16.
    for (std::size_t l = 0; l < 32; ++l) {
        block[16 + l] = static_cast<std::uint8_t>((l % 2 == 0) ? 0x01 : 0x00);
    }
    for (std::size_t l = 0; l < 32; ++l) {
        block[48 + l] = static_cast<std::uint8_t>(l % 16);  // low nibble varies, high nibble 0
    }

    GgufWriter w;
    w.add_tensor("w", {256, 1}, GgufType::Q5K, block);
    REQUIRE(w.write(tmp.str()));

    const auto file = GgufFile::open(tmp.str());
    REQUIRE(file.has_value());
    const auto out = file->decode_q5k_to_f32(*file->find_tensor("w"));
    REQUIRE(out.has_value());
    REQUIRE(out->size() == 256);

    const float scale1 = f16_to_f32(static_cast<std::uint16_t>(block[0] | (block[1] << 8))) * 3.0f;
    const float min1 = f16_to_f32(static_cast<std::uint16_t>(block[2] | (block[3] << 8))) * 2.0f;
    for (std::size_t l = 0; l < 32; ++l) {
        const std::uint32_t lo = block[48 + l] & 0x0f;
        const std::uint32_t hi = (l % 2 == 0) ? 1u : 0u;
        const float want = scale1 * static_cast<float>(lo | (hi << 4)) - min1;
        INFO("q5k element " << l);
        REQUIRE((*out)[l] == want);
    }
}

TEST_CASE("gguf get_scale_min matches the packed layout", "[gguf]") {
    // Round-trip: pack known 6-bit values the way Q4KMat does, then unpack.
    const std::uint8_t sv[8] = {5, 17, 33, 62, 9, 40, 55, 63};
    const std::uint8_t mv[8] = {1, 22, 44, 60, 3, 31, 50, 61};

    std::uint8_t sc[12] = {};
    for (std::size_t j = 0; j < 4; ++j) {
        sc[j] = static_cast<std::uint8_t>((sv[j] & 0x3f) | ((sv[j + 4] & 0x30) << 2));
        sc[j + 4] = static_cast<std::uint8_t>((mv[j] & 0x3f) | ((mv[j + 4] & 0x30) << 2));
        sc[8 + j] = static_cast<std::uint8_t>((sv[j + 4] & 0x0f) | ((mv[j + 4] & 0x0f) << 4));
    }

    for (std::size_t j = 0; j < 8; ++j) {
        float scale, min;
        gguf_get_scale_min(sc, j, scale, min);
        INFO("scale/min pair " << j);
        REQUIRE(scale == static_cast<float>(sv[j]));
        REQUIRE(min == static_cast<float>(mv[j]));

        // Q4KMat::scale_min is the same extraction; keep the two in lockstep.
        float s2, m2;
        Q4KMat::scale_min(sc, j, s2, m2);
        REQUIRE(s2 == scale);
        REQUIRE(m2 == min);
    }
}
