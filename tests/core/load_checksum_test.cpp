// tests/core/load_checksum_test.cpp - the expert arena reader's bytes, and its per-layer FNV-1a.
//
//   1. `load_experts_ranges` writes every byte of every range and, as it has done since it existed, returns
//      one FNV-1a per layer: equal to the same hash taken independently over the bytes that landed in the
//      destination, and equal from run to run.
//   2. `checksums = false` returns none, and so does STRATA_NO_LOAD_CHECKSUM=1 - checked in a child process,
//      because that variable is read once per process.  Skipping the hash is worth about half the readers'
//      time (a serial byte loop over everything the reader copies, include/strata/core/pinned.hpp), so it has
//      to move a single byte: both halves of this test compare the destination against the file.
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

bool every_byte(const std::vector<uint8_t>& dst) {
    for (size_t b = 0; b < dst.size(); ++b)
        if (dst[b] != pattern(b)) return false;
    return true;
}

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

// The default load: the hash runs, and its numbers are the FNV-1a of the bytes that arrived.
void run_default(const fs::path& file, const std::vector<uint64_t>& off, const std::vector<uint64_t>& len,
                 uint64_t bytes) {
    std::vector<uint8_t> dst((size_t) bytes, 0);
    const strata::core::LoadStats hashed =
        strata::core::load_experts_ranges(file.string(), dst.data(), off, len, 3, 1u << 20);
    check(hashed.ok && hashed.bytes == bytes, "the reader reads every range");
    check(hashed.layer_checksums.size() == off.size(),
          "and hashes by default: one FNV-1a per layer (" + std::to_string(hashed.layer_checksums.size()) +
              " of " + std::to_string(off.size()) + ")");
    check(every_byte(dst), "every byte of the destination is the byte of the file");
    bool match = hashed.layer_checksums.size() == off.size();
    for (size_t L = 0; match && L < off.size(); ++L)
        if (hashed.layer_checksums[L] != ref_fnv(dst.data() + off[L], (size_t) len[L])) match = false;
    check(match, "each checksum is the FNV-1a of its layer's bytes");

    std::fill(dst.begin(), dst.end(), 0);
    const strata::core::LoadStats again =
        strata::core::load_experts_ranges(file.string(), dst.data(), off, len, 3, 1u << 20, nullptr, true);
    check(again.layer_checksums == hashed.layer_checksums, "and the same pack gives the same checksums twice");
}

// The load with the hash turned off: no checksums, byte for byte the same destination.
void run_without_hash(const fs::path& file, const std::vector<uint64_t>& off, const std::vector<uint64_t>& len,
                      uint64_t bytes) {
    std::vector<uint8_t> dst((size_t) bytes, 0);
    const strata::core::LoadStats st =
        strata::core::load_experts_ranges(file.string(), dst.data(), off, len, 3, 1u << 20, nullptr, false);
    check(st.ok && st.bytes == bytes, "the reader still reads every range");
    check(st.layer_checksums.empty(), "and returns no checksums: the hash was skipped");
    check(every_byte(dst), "the bytes that arrive are the same with the hash off");
}
}  // namespace

int main(int argc, char** argv) {
    // STRATA_NO_LOAD_CHECKSUM is read once, so the off case runs as a child of this very test.
    const bool env_off = argc > 1 && std::strcmp(argv[1], "--env-off") == 0;
    if (env_off) {
#if defined(_WIN32)
        _putenv_s("STRATA_NO_LOAD_CHECKSUM", "1");
#else
        ::setenv("STRATA_NO_LOAD_CHECKSUM", "1", 1);
#endif
    } else {
#if defined(_WIN32)
        _unsetenv("STRATA_NO_LOAD_CHECKSUM");
#else
        ::unsetenv("STRATA_NO_LOAD_CHECKSUM");
#endif
    }
    std::printf(env_off ? "load_checksum_test, STRATA_NO_LOAD_CHECKSUM=1\n" : "load_checksum_test\n");
    const size_t kLayer = 4u << 20, kLayers = 3;
    const size_t bytes = kLayer * kLayers;
    TempDir dir;
    std::vector<uint8_t> all(bytes);
    for (size_t b = 0; b < bytes; ++b) all[b] = pattern(b);
    const fs::path file = dir.path / "experts.bin";
    std::ofstream(file, std::ios::binary).write((const char*) all.data(), (std::streamsize) bytes);
    const std::vector<uint64_t> off = {0, kLayer, 2 * kLayer}, len = {kLayer, kLayer, kLayer};
    if (env_off) {
        run_without_hash(file, off, len, bytes);
    } else {
        run_default(file, off, len, bytes);
        run_without_hash(file, off, len, bytes);
        std::string self;
        std::error_code ec;
        const fs::path exe = fs::canonical("/proc/self/exe", ec);   // Linux: this binary, resolved
        self = ec ? std::string(argv[0]) : exe.string();
        check(!self.empty() && std::system(("\"" + self + "\" --env-off").c_str()) == 0,
              "STRATA_NO_LOAD_CHECKSUM=1 does what checksums = false does (child process)");
    }
    std::printf(g_fail ? "load_checksum_test: %d FAILED\n" : "load_checksum_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
