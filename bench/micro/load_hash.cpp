// bench/micro/load_hash.cpp - what the per-layer FNV-1a of `load_experts_ranges` costs, on your machine.
//
// It reproduces that reader's loop exactly (per-thread `FILE*`, one staging buffer, `fread` -> `memcpy` ->
// optional FNV-1a) against a real file and runs it with the hash ON and OFF, so the difference IS the hash.
// This is the measurement behind `STRATA_NO_LOAD_CHECKSUM=1` (docs/DETAILS.md, "The expert arena read can skip
// its FNV-1a"); the loop is the one in src/core/pinned.cu, so a machine that is not copy-bound shows it here.
//
// build:  cmake --build <build dir> --target load_hash      (host code only: no GPU, no model, no downloads)
//         g++ -O2 -std=c++17 -D_GNU_SOURCE -pthread -o /tmp/load_hash bench/micro/load_hash.cpp
// usage:  dd if=/dev/zero of=/tmp/load_hash.bin bs=1M count=8192      # a file to read; delete it afterwards
//         /tmp/load_hash --file /tmp/load_hash.bin --threads 8 --chunk-mb 8 --GiB 8 --hash 1 --rounds 3
//         /tmp/load_hash --file /tmp/load_hash.bin --threads 8 --chunk-mb 8 --GiB 8 --hash 0 --rounds 3
//
// Round 1 reads cold, rounds 2+ warm (page cache): the warm rounds are the CPU-side cost, the cold ones what
// the drive adds on top.  The printed `read/copy/hash` seconds are SUMMED over the threads, like
// `LoadStats::read_seconds`, so a round of 8 GiB on 8 threads can print 8 x its own wall clock.  Measured
// 2026-10-16 on a Ryzen 9 5950X (no AVX-512) with a Samsung PM9A1, 8 readers, 8 MiB chunks, warm: 4.4-4.9
// GiB/s with the hash and 8.2-8.4 GiB/s without it - the hash is 6.1 s of the 8.7 s of reader thread time a
// 4 GiB round costs, about the documented ~1 byte per cycle of a serial byte loop.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

static uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t h = 1469598103934665603ull) {
    for (uint64_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    std::string path;
    int threads = 8;
    uint64_t chunk = 8ull << 20;
    uint64_t bytes = 8ull << 30;   // bytes to move per round
    int hash = 1;
    int rounds = 2;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--file") path = val();
        else if (a == "--threads") threads = std::atoi(val());
        else if (a == "--chunk-mb") chunk = std::strtoull(val(), nullptr, 10) << 20;
        else if (a == "--GiB") bytes = std::strtoull(val(), nullptr, 10) << 30;
        else if (a == "--hash") hash = std::atoi(val());
        else if (a == "--rounds") rounds = std::atoi(val());
    }
    if (path.empty()) { std::fprintf(stderr, "--file required\n"); return 2; }
    FILE* probe = std::fopen(path.c_str(), "rb");
    if (!probe) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 1; }
    std::fseek(probe, 0, SEEK_END);
    const uint64_t file_bytes = (uint64_t) ::ftello(probe);
    std::fclose(probe);
    if (bytes > file_bytes) bytes = file_bytes;
    uint8_t* dst = (uint8_t*) std::malloc(bytes);
    if (dst == nullptr) { std::fprintf(stderr, "cannot allocate %llu B\n", (unsigned long long) bytes); return 1; }
    std::memset(dst, 0, 1);   // touch one page; the rest is filled by the load itself

    std::printf("file %s  %.1f GiB read per round  chunk %llu MiB  hash %s\n", path.c_str(),
                (double) bytes / (1ull << 30), (unsigned long long) (chunk >> 20), hash ? "ON" : "off");
    for (int r = 0; r < rounds; ++r) {
        // the layers of a pack: the bytes split over the threads the way load_experts_ranges hands them out
        const uint64_t per = bytes / (uint64_t) threads;
        std::atomic<uint64_t> next{0};
        std::atomic<uint64_t> read_ns{0}, copy_ns{0}, hash_ns{0};
        uint64_t out_hash = 0;
        const auto t0 = std::chrono::steady_clock::now();
        auto worker = [&]() {
            std::vector<uint8_t> buf((size_t) chunk);
            FILE* f = std::fopen(path.c_str(), "rb");
            uint64_t h = 1469598103934665603ull, rn = 0, cn = 0, hn = 0;
            for (;;) {
                const uint64_t i = next.fetch_add(1);
                if (i * chunk >= bytes) break;
                const uint64_t off = i * chunk;
                
                if (::fseeko(f, (off_t) off, SEEK_SET) != 0) break;
                const auto tr = std::chrono::steady_clock::now();
                const size_t got = std::fread(buf.data(), 1, (size_t) chunk, f);
                rn += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - tr).count();
                if (got != (size_t) chunk) break;
                const auto tc = std::chrono::steady_clock::now();
                std::memcpy(dst + off, buf.data(), (size_t) chunk);
                cn += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - tc).count();
                if (hash) {
                    const auto th = std::chrono::steady_clock::now();
                    h = fnv1a64(buf.data(), chunk, h);
                    hn += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - th).count();
                }
            }
            std::fclose(f);
            read_ns.fetch_add(rn); copy_ns.fetch_add(cn); hash_ns.fetch_add(hn);
            if (hash) out_hash ^= h;
            (void) per;
        };
        std::vector<std::thread> pool;
        for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double gib = (double) bytes / (1ull << 30);
        std::printf("  round %d: %6.2f s  %5.2f GiB/s wall | summed over threads: read %5.2f s, copy %5.2f s, hash %5.2f s%s\n",
                    r, sec, gib / sec, (double) read_ns / 1e9, (double) copy_ns / 1e9, (double) hash_ns / 1e9,
                    hash ? "" : " (hash off)");
    }
    std::free(dst);
    return 0;
}
