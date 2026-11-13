#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "config.hpp"
#include "verifier.hpp"
#include "sparse_run.hpp"

namespace {

/// Parses an unsigned count in full decimal ("4000000000000000000") or
/// e-notation ("4e18", "250e6"). The previous strtoull(..., 0) silently
/// stopped at the first non-digit ("1e9" parsed as 1, "010" as octal 8),
/// so a typo could produce a zero-work run that still reported PASS.
/// Exact integer math only: mantissa * 10^exp with overflow checks.
/// @param s   text to parse (nullptr / empty rejected); no sign, no spaces.
/// @param ok  set to true only when the whole string parsed without overflow.
/// @return the value, or 0 when @p ok is false.
uint64_t parse_count(const char* s, bool& ok) {
    ok = false;
    if (s == nullptr || *s == '\0') return 0;
    uint64_t m = 0;
    uint64_t exp10 = 0;
    int digits = 0;
    const char* p = s;
    for (; *p >= '0' && *p <= '9'; ++p) {
        const uint64_t d = uint64_t(*p - '0');
        if (m > (~0ull - d) / 10) return 0;      // mantissa overflow
        m = m * 10 + d;
        ++digits;
    }
    if (digits == 0) return 0;
    if (*p == 'e' || *p == 'E') {
        ++p;
        if (*p < '0' || *p > '9') return 0;
        for (; *p >= '0' && *p <= '9'; ++p) {
            if (exp10 > 19) return 0;            // 10^exp cannot fit u64 anyway
            exp10 = exp10 * 10 + uint64_t(*p - '0');
        }
    }
    if (*p != '\0') return 0;                    // trailing junk
    for (uint64_t i = 0; i < exp10; ++i) {
        if (m > ~0ull / 10) return 0;            // shift overflow
        m *= 10;
    }
    ok = true;
    return m;
}

/// Prints the command-line help (with the build's defaults) to stdout.
void usage() {
    std::printf(
        "goldbach-l0 — Goldbach verifier for Intel Arc B580 (Level Zero)\n"
        "\n"
        "usage: goldbach_l0 [--limit N] [--start N] [--segment N] [--split N]\n"
        "                   [--sparse] [--spv FILE] [--prime-cache FILE]\n"
        "                   [--cpu-check N] [--json] [--csv] [--per-seg]\n"
        "                   [--info] [--verbose]\n"
        "\n"
        "  N accepts full decimal or e-notation: 4000000000000000000 = 4e18.\n"
        "\n"
        "  --limit N      verify every even n in [start, N]  (default %llu)\n"
        "  --start N      first even n to verify (scale tests; default 4)\n"
        "  --segment N    even integer span per GPU segment  (default %llu)\n"
        "  --split N      base-prime split index (0 = auto)  (default 0)\n"
        "  --sparse       sparse class map (v2.0, PROJECT-REFERENCE.md §9.2k; also\n"
        "                 GB_SPARSE=1): ~17x faster at 4e18, needs start >> sqrt(limit)\n"
        "  --spv FILE     SPIR-V module path                (default kernels/goldbach_bmg.spv)\n"
        "  --prime-cache FILE  disk cache for the base-prime list\n"
        "  --cpu-check N  run the independent CPU check up to N and exit\n"
        "  --json         machine-readable JSON report on stdout\n"
        "  --csv          one summary line: key=value pairs (grep/awk friendly)\n"
        "  --per-seg      print a per-segment timing table (implies collection)\n"
        "  --info         print device/driver/config info and exit\n"
        "  --verbose      per-segment progress output\n",
        (unsigned long long)gb::DEFAULT_LIMIT,
        (unsigned long long)gb::DEFAULT_SEGMENT_EVEN);
}

// ---- machine-readable output modes ----------------------------------------

/// @return @p s with '"' and '\\' backslash-escaped and control characters
///         replaced by spaces (nullptr yields ""), for a JSON string body.
std::string json_escape(const char* s) {
    std::string out;
    for (const char* p = s ? s : ""; *p; ++p) {
        if (*p == '"' || *p == '\\') { out += '\\'; out += *p; }
        else if (static_cast<unsigned char>(*p) < 0x20) out += ' ';
        else out += *p;
    }
    return out;
}

/// Prints the run result as one JSON object on stdout (per-segment array
/// included when s.per_segment is non-empty).
/// @pre   the run passed (the object always reports "result": "PASS").
void print_json(const gb::RunStats& s, uint64_t segment, uint32_t split,
                const std::string& spv) {
    std::printf("{\n");
    std::printf("  \"version\": \"%s\",\n", gb::VERSION);
    std::printf("  \"limit\": %llu,\n", (unsigned long long)s.limit);
    std::printf("  \"start\": %llu,\n", (unsigned long long)s.start);
    std::printf("  \"segment\": %llu,\n", (unsigned long long)segment);
    std::printf("  \"split\": %u,\n", split);
    if (s.split_used) std::printf("  \"split_used\": %u,\n", s.split_used);
    std::printf("  \"spv\": \"%s\",\n", json_escape(spv.c_str()).c_str());
    std::printf("  \"segments\": %llu,\n", (unsigned long long)s.segments);
    std::printf("  \"leftovers\": %llu,\n", (unsigned long long)s.leftovers);
    std::printf("  \"result\": \"PASS\",\n");
    std::printf("  \"init_seconds\": %.4f,\n", s.init_seconds);
    std::printf("  \"gpu_seconds\": %.4f,\n", s.gpu_seconds);
    std::printf("  \"total_seconds\": %.4f,\n", s.total_seconds);
    std::printf("  \"sieve_seconds\": %.4f,\n", s.sieve_seconds);
    std::printf("  \"large_seconds\": %.4f,\n", s.large_seconds);
    std::printf("  \"verify_seconds\": %.4f,\n", s.verify_seconds);
    std::printf("  \"offset_seconds\": %.4f,\n", s.offset_seconds);
    std::printf("  \"table_seconds\": %.4f,\n", s.table_seconds);
    std::printf("  \"sync_seconds\": %.4f,\n", s.sync_seconds);
    std::printf("  \"evens_per_second\": %.6g,\n", s.evens_per_second);
    std::printf("  \"total_tiles\": %llu,\n", (unsigned long long)s.total_tiles);
    std::printf("  \"total_large_items\": %llu,\n", (unsigned long long)s.total_large_items);
    std::printf("  \"ridge_min\": %u,\n", (unsigned)s.ridge_min);
    std::printf("  \"ridge_found\": %llu,\n", (unsigned long long)s.ridge_found);
    std::printf("  \"ridge_dropped\": %llu,\n", (unsigned long long)s.ridge_dropped);
    std::printf("  \"ridge_top\": [");
    for (size_t i = 0; i < s.ridge_top.size(); ++i) {
        const gb::RidgeEntry& e = s.ridge_top[i];
        std::printf("%s\n    {\"rank\": %zu, \"ridge\": %llu, \"even\": %llu, "
                    "\"p\": %llu, \"q\": %llu, \"segment\": %llu}",
                    i ? "," : "", i + 1,
                    (unsigned long long)e.p, (unsigned long long)e.n,
                    (unsigned long long)e.p, (unsigned long long)e.q,
                    (unsigned long long)e.seg);
    }
    std::printf("%s]", s.ridge_top.empty() ? "" : "\n  ");
    if (s.stats_on)
        std::printf(",\n  \"stats\": {\"tail_evens\": %llu, \"sampled_evens\": %llu, "
                    "\"pmax\": %llu, \"primes\": %llu, \"twins\": %llu}",
                    (unsigned long long)s.stats_tail, (unsigned long long)s.stats_sampled,
                    (unsigned long long)s.stats_pmax, (unsigned long long)s.stats_primes,
                    (unsigned long long)s.stats_twins);
    if (!s.per_segment.empty()) {
        std::printf(",\n  \"per_segment\": [\n");
        for (size_t i = 0; i < s.per_segment.size(); ++i) {
            const gb::SegmentStats& g = s.per_segment[i];
            std::printf(
                "    {\"seg\": %zu, \"start\": %llu, \"end\": %llu, "
                "\"evens\": %llu, \"tiles\": %llu, \"large_items\": %llu, "
                "\"split\": %u, \"mask_count\": %u, \"leftovers\": %u, "
                "\"sieve_s\": %.4f, \"large_s\": %.4f, \"check_s\": %.4f, "
                "\"offset_s\": %.4f, \"wall_s\": %.4f}%s\n",
                i + 1,
                (unsigned long long)g.seg_start, (unsigned long long)g.seg_end,
                (unsigned long long)g.evens, (unsigned long long)g.tiles,
                (unsigned long long)g.large_items, g.sieve_split, g.mask_count,
                g.leftovers, g.sieve_seconds, g.large_seconds,
                g.check_seconds, g.offset_seconds, g.wall_seconds,
                i + 1 < s.per_segment.size() ? "," : "");
        }
        std::printf("  ]");
    }
    std::printf("\n}\n");
}

/// Prints the run result as one line of key=value pairs on stdout.
/// @pre   the run passed (the line always reports result=PASS).
void print_csv(const gb::RunStats& s, uint64_t segment, uint32_t split) {
    // One line of key=value pairs — trivially parsed by awk/grep/Python.
    std::printf(
        "version=%s limit=%llu start=%llu segment=%llu split=%u "
        "segments=%llu leftovers=%llu result=PASS "
        "init_s=%.4f init_cache_s=%.4f init_module_s=%.4f init_upload_s=%.4f "
        "gpu_s=%.4f total_s=%.4f sieve_s=%.4f large_s=%.4f "
        "check_s=%.4f offsets_s=%.4f table_s=%.4f sync_s=%.4f evens_per_sec=%.6g "
        "tiles=%llu large_items=%llu "
        "ridge_min=%u ridge_found=%llu ridge_dropped=%llu ridge_max=%llu "
        "stats=%d stats_tail=%llu stats_sampled=%llu stats_pmax=%llu primes=%llu twins=%llu "
        "split_used=%u\n",
        gb::VERSION,
        (unsigned long long)s.limit, (unsigned long long)s.start,
        (unsigned long long)segment, split,
        (unsigned long long)s.segments, (unsigned long long)s.leftovers,
        s.init_seconds, s.init_cache_s, s.init_module_s, s.init_upload_s,
        s.gpu_seconds, s.total_seconds,
        s.sieve_seconds, s.large_seconds, s.verify_seconds,
        s.offset_seconds, s.table_seconds, s.sync_seconds, s.evens_per_second,
        (unsigned long long)s.total_tiles, (unsigned long long)s.total_large_items,
        (unsigned)s.ridge_min, (unsigned long long)s.ridge_found,
        (unsigned long long)s.ridge_dropped,
        (unsigned long long)(s.ridge_top.empty() ? 0 : s.ridge_top.front().p),
        s.stats_on ? 1 : 0, (unsigned long long)s.stats_tail, (unsigned long long)s.stats_sampled,
        (unsigned long long)s.stats_pmax, (unsigned long long)s.stats_primes,
        (unsigned long long)s.stats_twins, s.split_used);
}

/// Prints the per-segment table (header + one row per s.per_segment entry).
void print_per_segment(const gb::RunStats& s) {
    std::printf(
        "#seg  evens  [start, end]      tiles  large_items  split  "
        "sieve_s  large_s  check_s  offset_s  wall_s  left\n");
    for (size_t i = 0; i < s.per_segment.size(); ++i) {
        const gb::SegmentStats& g = s.per_segment[i];
        std::printf(
            "%4zu  %9llu  [%13llu, %-13llu] %6llu %11llu  %6u  "
            "%6.3f  %7.3f  %7.3f  %7.3f  %6.3f  %5u\n",
            i + 1,
            (unsigned long long)g.evens,
            (unsigned long long)g.seg_start, (unsigned long long)g.seg_end,
            (unsigned long long)g.tiles, (unsigned long long)g.large_items,
            g.sieve_split,
            g.sieve_seconds, g.large_seconds, g.check_seconds,
            g.offset_seconds, g.wall_seconds, g.leftovers);
    }
}

/// Prints the persistent Goldbach ridge leaderboard (include/ridge.hpp):
/// the RIDGE_KEEP largest recorded ridges, ridge size descending.
void print_ridges(const gb::RunStats& s) {
    if (!s.ridge_min) return;
    std::printf("\nMy Top 30 Largest Goldbach Ridges\n");
    if (s.ridge_top.empty()) {
        std::printf("  (none recorded at or above the floor %u yet)\n",
                    (unsigned)s.ridge_min);
        return;
    }
    for (size_t i = 0; i < s.ridge_top.size(); ++i) {
        const gb::RidgeEntry& e = s.ridge_top[i];
        std::printf("  %2zu. ridge %-5llu  %llu = %llu + %llu\n",
                    i + 1,
                    (unsigned long long)e.p,
                    (unsigned long long)e.n,
                    (unsigned long long)e.q,
                    (unsigned long long)e.p);
    }
}

} // namespace

/// Command-line entry point (see usage()): runs --info, --cpu-check, or a GPU
/// verification (dense or --sparse) and prints the result as text,
/// --json or --csv.
/// @return 0 on PASS (or --info / --help), 1 on a verification failure or
///         any runtime error, 2 on a command-line error.
int main(int argc, char** argv) {
    uint64_t limit       = gb::DEFAULT_LIMIT;
    uint64_t start       = 4;
    uint64_t segment     = gb::DEFAULT_SEGMENT_EVEN;
    uint64_t split       = 0;
    uint64_t cpu_check   = 0;
    bool cpu_check_set   = false;
    std::string spv      = "kernels/goldbach_bmg.spv";
    bool sparse          = false;
    bool spv_set         = false;
    std::string prime_cache = "base_primes.cache";
    bool verbose         = false;
    bool json            = false;
    bool csv             = false;
    bool per_seg         = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](uint64_t& out) {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            bool ok = false;
            out = parse_count(argv[++i], ok);
            if (!ok) {
                std::fprintf(stderr, "error: invalid number '%s'\n", argv[i]);
                std::exit(2);
            }
        };
        auto next_str = [&]() -> const char* {        // string value (flag must not be last)
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if      (a == "--limit")     next(limit);
        else if (a == "--start")     next(start);
        else if (a == "--segment")   next(segment);
        else if (a == "--split")     next(split);
        else if (a == "--cpu-check") { next(cpu_check); cpu_check_set = true; }
        else if (a == "--sparse")    sparse = true;
        else if (a == "--spv")       { spv = next_str(); spv_set = true; }
        else if (a == "--prime-cache") prime_cache = next_str();
        else if (a == "--verbose")   verbose = true;
        else if (a == "--json")      json = true;
        else if (a == "--csv")       csv = true;
        else if (a == "--per-seg")   per_seg = true;
        else if (a == "--info") {
            try {
                gb::print_device_info();
            } catch (const std::exception& e) {
                std::fprintf(stderr, "error: %s\n", e.what());
                return 1;
            }
            return 0;
        }
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(); return 2; }
    }

    // Range contract (GPU paths): verify every even n with max(start, 4) <= n
    // <= limit. An empty or reversed range is a usage error, never a PASS
    // over evens that were not requested.
    // --cpu-check 0 used to mean "flag absent" and silently ran a full GPU
    // verification of the default range instead.
    if (cpu_check_set && cpu_check < 4) {
        std::fprintf(stderr, "error: --cpu-check needs N >= 4\n");
        return 2;
    }
    if (!cpu_check_set) {
        const uint64_t first_even = start < 4 ? 4 : start + (start & 1);
        if (start > limit || first_even > limit || first_even < start) {
            std::fprintf(stderr, "error: no even n >= 4 in [start=%llu, limit=%llu]\n",
                         (unsigned long long)start, (unsigned long long)limit);
            return 2;
        }
        if (split > 0xFFFFFFFFull) {
            std::fprintf(stderr, "error: --split %llu exceeds 2^32 - 1\n", (unsigned long long)split);
            return 2;
        }
    }

    try {
        if (cpu_check_set) {
            const bool ok = gb::verify_cpu(cpu_check, verbose);
            std::printf("CPU check up to %llu: %s\n",
                        (unsigned long long)cpu_check, ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }

        // GB_SPARSE=1 selects the sparse path too (benchmark harness arms)
        if (const char* e = std::getenv("GB_SPARSE")) sparse = sparse || std::string(e) == "1";
        if (sparse && !spv_set) spv.clear();          // the geometry's module
        const gb::RunStats s = sparse
            ? gb::run_gpu_sparse(limit, segment, spv, verbose, start, uint32_t(split),
                                 prime_cache.c_str(), per_seg || json)
            : gb::run_gpu(limit, segment, spv, verbose, start,
                          uint32_t(split), prime_cache.c_str(),
                          per_seg || json);
        if (json) {
            print_json(s, s.segment_used ? s.segment_used : segment, uint32_t(split), spv);
        } else if (csv) {
            print_csv(s, s.segment_used ? s.segment_used : segment, uint32_t(split));
        } else {
            std::printf(
                "goldbach-l0 v%s%s\n"
                "  limit          : %llu\n"
                "  start          : %llu\n"
                "  segments       : %llu\n"
                "  leftovers      : %llu\n"
                "  init time      : %.3f s\n"
                "  gpu time       : %.3f s\n"
                "    sieve        : %.3f s\n"
                "    large sieve  : %.3f s\n"
                "    verify       : %.3f s\n"
                "    offsets      : %.3f s\n"
                "    sync/other   : %.3f s\n"
                "  total          : %.3f s\n"
                "  evens/second   : %.3g\n"
                "  result         : PASS\n",
                gb::VERSION, sparse ? " (sparse)" : "",
                (unsigned long long)s.limit,
                (unsigned long long)s.start,
                (unsigned long long)s.segments,
                (unsigned long long)s.leftovers,
                s.init_seconds, s.gpu_seconds,
                s.sieve_seconds, s.large_seconds, s.verify_seconds,
                s.offset_seconds, s.sync_seconds,
                s.total_seconds,
                s.evens_per_second);
        }
        if (s.stats_on && !csv && !json)
            std::printf("\nGoldbach statistics (exact; log %s, state %s)\n"
                        "  largest p(n)            : %llu\n"
                        "  evens with p(n) >= tail : %llu\n"
                        "  evens with full p(n)    : %llu (sampled segments)\n"
                        "  primes in [start, limit]: %llu\n"
                        "  twin-prime pairs        : %llu\n",
                        std::getenv("GB_STATS_LOG") ? std::getenv("GB_STATS_LOG") : "goldbach_stats.log",
                        std::getenv("GB_STATS_STATE") ? std::getenv("GB_STATS_STATE") : "goldbach_stats.state",
                        (unsigned long long)s.stats_pmax, (unsigned long long)s.stats_tail,
                        (unsigned long long)s.stats_sampled, (unsigned long long)s.stats_primes,
                        (unsigned long long)s.stats_twins);
        if (!csv && !json) print_ridges(s);   // --json / --csv carry ridge_top / ridge_max
        if (per_seg && !json) print_per_segment(s);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
