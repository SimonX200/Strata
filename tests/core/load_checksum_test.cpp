// tests/core/load_checksum_test.cpp - the expert arena reader's bytes, and its optional per-layer FNV-1a.
//
//   1. `load_experts_ranges` writes every byte of every range, and returns NO checksums: hashing what was read
//      is a diagnostic, and a serial byte loop over the whole arena costs about half the readers' time
//      (include/strata/core/pinned.hpp).  A build that hashed by default loaded 8 GiB at 4.9 GiB/s where the
//      same loop without the hash ran 8.4 GiB/s.
//   2. With `checksums = true` it returns one FNV-1a per layer, equal to the same hash taken independently over
//      the bytes that landed in the destination, and equal from run to run.
// No model, no GPU: the reader is host code, and the destination here is plain malloc'd memory.
#include "strata/core/pinned.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// The reference: written out again on purpose, so the test does not compare the loader with itself.
uint64_t ref_fnv(const uint8_t* p, size_t n, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

uint8_t pattern(size_t i) { return (uint8_t) ((i * 1103515245ull + 12345ull) >> 11); }

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-load-checksum-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void run(const fs::path& file, const std::vector<uint64_t>& off, const std::vector<uint64_t>& len,
         uint64_t bytes) {
    std::vector<uint8_t> dst((size_t) bytes, 0);
    const strata::core::LoadStats plain =
        strata::core::load_experts_ranges(file.string(), dst.data(), off, len, 3, 1u << 20, nullptr, false);
    check(plain.ok && plain.bytes == bytes, "the reader reads every range");
    check(plain.layer_checksums.empty(), "and returns no checksums: the default load does not hash");
    bool exact = true;
    for (uint64_t b = 0; b < bytes && exact; ++b)
        if (dst[(size_t) b] != pattern((size_t) b)) exact = false;
    check(exact, "every byte of the destination is the byte of the file");

    std::fill(dst.begin(), dst.end(), 0);
    const strata::core::LoadStats hashed =
        strata::core::load_experts_ranges(file.string(), dst.data(), off, len, 3, 1u << 20, nullptr, true);
    check(hashed.ok && hashed.layer_checksums.size() == off.size(),
          "checksums = true returns one per layer (" + std::to_string(hashed.layer_checksums.size()) + " of " +
              std::to_string(off.size()) + ")");
    bool same_bytes = true;
    for (uint64_t b = 0; b < bytes && same_bytes; ++b)
        if (dst[(size_t) b] != pattern((size_t) b)) same_bytes = false;
    check(same_bytes, "hashing changes nothing about the bytes that arrive");
    bool match = hashed.layer_checksums.size() == off.size();
    for (size_t L = 0; match && L < off.size(); ++L)
        if (hashed.layer_checksums[L] != ref_fnv(dst.data() + off[L], (size_t) len[L])) match = false;
    check(match, "each checksum is the FNV-1a of its layer's bytes");

    std::fill(dst.begin(), dst.end(), 0);
    const strata::core::LoadStats again =
        strata::core::load_experts_ranges(file.string(), dst.data(), off, len, 3, 1u << 20, nullptr, true);
    check(again.layer_checksums == hashed.layer_checksums, "and the same pack gives the same checksums twice");
}
}  // namespace

int main() {
    // The reader also honours STRATA_LOAD_CHECKSUM; this test is about the parameter and the default.
#if defined(_WIN32)
    _unsetenv("STRATA_LOAD_CHECKSUM");
#else
    unsetenv("STRATA_LOAD_CHECKSUM");
#endif
    std::printf("load_checksum_test\n");
    const size_t kLayer = 4u << 20, kLayers = 3;
    const size_t bytes = kLayer * kLayers;
    TempDir dir;
    std::vector<uint8_t> all(bytes);
    for (size_t b = 0; b < bytes; ++b) all[b] = pattern(b);
    const fs::path file = dir.path / "experts.bin";
    std::ofstream(file, std::ios::binary).write((const char*) all.data(), (std::streamsize) bytes);
    const std::vector<uint64_t> off = {0, kLayer, 2 * kLayer}, len = {kLayer, kLayer, kLayer};
    run(file, off, len, bytes);
    std::printf(g_fail ? "load_checksum_test: %d FAILED\n" : "load_checksum_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
