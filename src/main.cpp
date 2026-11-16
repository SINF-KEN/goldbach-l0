#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <print>
#include <string>
#include <vector>

#include "config.hpp"
#include "l0_context.hpp"
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
    std::print(
        "goldbach-l0 — Goldbach verifier for Intel Arc B580 (Level Zero)\n"
        "\n"
        "usage: goldbach_l0 [--limit N] [--start N] [--segment N] [--split N]\n"
        "                   [--sparse] [--spv FILE] [--prime-cache FILE]\n"
        "                   [--cpu-check N] [--json] [--csv] [--per-seg]\n"
        "                   [--info] [--verbose]\n"
        "\n"
        "  N accepts full decimal or e-notation: 4000000000000000000 = 4e18.\n"
        "\n"
        "  --limit N      verify every even n in [start, N]  (default {})\n"
        "  --start N      first even n to verify (scale tests; default 4)\n"
        "  --segment N    even integer span per GPU segment  (default {})\n"
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
        gb::DEFAULT_LIMIT, gb::DEFAULT_SEGMENT_EVEN);
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
    // literal JSON braces are doubled ({{ }}) in std::format strings
    std::println("{{");
    std::println("  \"version\": \"{}\",", gb::VERSION);
    std::println("  \"limit\": {},", s.limit);
    std::println("  \"start\": {},", s.start);
    std::println("  \"segment\": {},", segment);
    std::println("  \"split\": {},", split);
    if (s.split_used) std::println("  \"split_used\": {},", s.split_used);
    std::println("  \"sched\": \"{}\",", gb::sched_policy_name());
    std::println("  \"spv\": \"{}\",", json_escape(spv.c_str()));
    std::println("  \"segments\": {},", s.segments);
    std::println("  \"leftovers\": {},", s.leftovers);
    std::println("  \"result\": \"PASS\",");
    std::println("  \"init_seconds\": {:.4f},", s.init_seconds);
    std::println("  \"gpu_seconds\": {:.4f},", s.gpu_seconds);
    std::println("  \"total_seconds\": {:.4f},", s.total_seconds);
    std::println("  \"sieve_seconds\": {:.4f},", s.sieve_seconds);
    std::println("  \"large_seconds\": {:.4f},", s.large_seconds);
    std::println("  \"verify_seconds\": {:.4f},", s.verify_seconds);
    std::println("  \"offset_seconds\": {:.4f},", s.offset_seconds);
    std::println("  \"table_seconds\": {:.4f},", s.table_seconds);
    std::println("  \"sync_seconds\": {:.4f},", s.sync_seconds);
    std::println("  \"evens_per_second\": {:.6g},", s.evens_per_second);
    std::println("  \"total_tiles\": {},", s.total_tiles);
    std::println("  \"total_large_items\": {},", s.total_large_items);
    std::println("  \"ridge_min\": {},", s.ridge_min);
    std::println("  \"ridge_found\": {},", s.ridge_found);
    std::println("  \"ridge_dropped\": {},", s.ridge_dropped);
    std::print("  \"ridge_top\": [");
    for (size_t i = 0; i < s.ridge_top.size(); ++i) {
        const gb::RidgeEntry& e = s.ridge_top[i];
        std::print("{}\n    {{\"rank\": {}, \"ridge\": {}, \"even\": {}, "
                   "\"p\": {}, \"q\": {}, \"segment\": {}}}",
                   i ? "," : "", i + 1, e.p, e.n, e.p, e.q, e.seg);
    }
    std::print("{}]", s.ridge_top.empty() ? "" : "\n  ");
    if (s.stats_on)
        std::print(",\n  \"stats\": {{\"tail_evens\": {}, \"sampled_evens\": {}, "
                   "\"pmax\": {}, \"primes\": {}, \"twins\": {}}}",
                   s.stats_tail, s.stats_sampled, s.stats_pmax, s.stats_primes, s.stats_twins);
    if (!s.per_segment.empty()) {
        std::println(",\n  \"per_segment\": [");
        for (size_t i = 0; i < s.per_segment.size(); ++i) {
            const gb::SegmentStats& g = s.per_segment[i];
            std::println(
                "    {{\"seg\": {}, \"start\": {}, \"end\": {}, "
                "\"evens\": {}, \"tiles\": {}, \"large_items\": {}, "
                "\"split\": {}, \"mask_count\": {}, \"leftovers\": {}, "
                "\"sieve_s\": {:.4f}, \"large_s\": {:.4f}, \"check_s\": {:.4f}, "
                "\"offset_s\": {:.4f}, \"wall_s\": {:.4f}}}{}",
                i + 1, g.seg_start, g.seg_end, g.evens, g.tiles, g.large_items,
                g.sieve_split, g.mask_count, g.leftovers,
                g.sieve_seconds, g.large_seconds, g.check_seconds,
                g.offset_seconds, g.wall_seconds,
                i + 1 < s.per_segment.size() ? "," : "");
        }
        std::print("  ]");
    }
    std::print("\n}}\n");
}

/// Prints the run result as one line of key=value pairs on stdout.
/// @pre   the run passed (the line always reports result=PASS).
void print_csv(const gb::RunStats& s, uint64_t segment, uint32_t split) {
    // One line of key=value pairs — trivially parsed by awk/grep/Python.
    std::println(
        "version={} limit={} start={} segment={} split={} "
        "segments={} leftovers={} result=PASS "
        "init_s={:.4f} init_cache_s={:.4f} init_module_s={:.4f} init_upload_s={:.4f} "
        "gpu_s={:.4f} total_s={:.4f} sieve_s={:.4f} large_s={:.4f} "
        "check_s={:.4f} offsets_s={:.4f} table_s={:.4f} sync_s={:.4f} evens_per_sec={:.6g} "
        "tiles={} large_items={} "
        "ridge_min={} ridge_found={} ridge_dropped={} ridge_max={} "
        "stats={} stats_tail={} stats_sampled={} stats_pmax={} primes={} twins={} "
        "split_used={} sched={}",
        gb::VERSION, s.limit, s.start, segment, split, s.segments, s.leftovers,
        s.init_seconds, s.init_cache_s, s.init_module_s, s.init_upload_s,
        s.gpu_seconds, s.total_seconds,
        s.sieve_seconds, s.large_seconds, s.verify_seconds,
        s.offset_seconds, s.table_seconds, s.sync_seconds, s.evens_per_second,
        s.total_tiles, s.total_large_items,
        s.ridge_min, s.ridge_found, s.ridge_dropped,
        s.ridge_top.empty() ? 0 : s.ridge_top.front().p,
        s.stats_on ? 1 : 0, s.stats_tail, s.stats_sampled,
        s.stats_pmax, s.stats_primes, s.stats_twins, s.split_used, gb::sched_policy_name());
}

/// Prints the per-segment table (header + one row per s.per_segment entry).
void print_per_segment(const gb::RunStats& s) {
    std::println(
        "#seg  evens  [start, end]      tiles  large_items  split  "
        "sieve_s  large_s  check_s  offset_s  wall_s  left");
    for (size_t i = 0; i < s.per_segment.size(); ++i) {
        const gb::SegmentStats& g = s.per_segment[i];
        std::println(
            "{:4}  {:9}  [{:13}, {:<13}] {:6} {:11}  {:6}  "
            "{:6.3f}  {:7.3f}  {:7.3f}  {:7.3f}  {:6.3f}  {:5}",
            i + 1, g.evens, g.seg_start, g.seg_end, g.tiles, g.large_items,
            g.sieve_split,
            g.sieve_seconds, g.large_seconds, g.check_seconds,
            g.offset_seconds, g.wall_seconds, g.leftovers);
    }
}

/// Prints the persistent Goldbach ridge leaderboard (include/ridge.hpp):
/// the RIDGE_KEEP largest recorded ridges, ridge size descending.
void print_ridges(const gb::RunStats& s) {
    if (!s.ridge_min) return;
    std::println("\nMy Top 30 Largest Goldbach Ridges");
    if (s.ridge_top.empty()) {
        std::println("  (none recorded at or above the floor {} yet)", s.ridge_min);
        return;
    }
    for (size_t i = 0; i < s.ridge_top.size(); ++i) {
        const gb::RidgeEntry& e = s.ridge_top[i];
        std::println("  {:2}. ridge {:<5}  {} = {} + {}", i + 1, e.p, e.n, e.q, e.p);
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
                std::println(stderr, "error: invalid number '{}'", argv[i]);
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
                std::println(stderr, "error: {}", e.what());
                return 1;
            }
            return 0;
        }
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::println(stderr, "unknown option: {}", a); usage(); return 2; }
    }

    // Range contract (GPU paths): verify every even n with max(start, 4) <= n
    // <= limit. An empty or reversed range is a usage error, never a PASS
    // over evens that were not requested.
    // --cpu-check 0 used to mean "flag absent" and silently ran a full GPU
    // verification of the default range instead.
    if (cpu_check_set && cpu_check < 4) {
        std::println(stderr, "error: --cpu-check needs N >= 4");
        return 2;
    }
    if (!cpu_check_set) {
        const uint64_t first_even = start < 4 ? 4 : start + (start & 1);
        if (start > limit || first_even > limit || first_even < start) {
            std::println(stderr, "error: no even n >= 4 in [start={}, limit={}]", start, limit);
            return 2;
        }
        if (split > 0xFFFFFFFFull) {
            std::println(stderr, "error: --split {} exceeds 2^32 - 1", split);
            return 2;
        }
    }

    try {
        if (cpu_check_set) {
            const bool ok = gb::verify_cpu(cpu_check, verbose);
            std::println("CPU check up to {}: {}", cpu_check, ok ? "PASS" : "FAIL");
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
            std::print(
                "goldbach-l0 v{}{}\n"
                "  limit          : {}\n"
                "  start          : {}\n"
                "  segments       : {}\n"
                "  leftovers      : {}\n"
                "  init time      : {:.3f} s\n"
                "  gpu time       : {:.3f} s\n"
                "    sieve        : {:.3f} s\n"
                "    large sieve  : {:.3f} s\n"
                "    verify       : {:.3f} s\n"
                "    offsets      : {:.3f} s\n"
                "    sync/other   : {:.3f} s\n"
                "  total          : {:.3f} s\n"
                "  evens/second   : {:.3g}\n"
                "  EU arbitration : {}\n"
                "  result         : PASS\n",
                gb::VERSION, sparse ? " (sparse)" : "",
                s.limit, s.start, s.segments, s.leftovers,
                s.init_seconds, s.gpu_seconds,
                s.sieve_seconds, s.large_seconds, s.verify_seconds,
                s.offset_seconds, s.sync_seconds,
                s.total_seconds,
                s.evens_per_second, gb::sched_policy_name());
        }
        if (s.stats_on && !csv && !json)
            std::print("\nGoldbach statistics (exact; log {}, state {})\n"
                       "  largest p(n)            : {}\n"
                       "  evens with p(n) >= tail : {}\n"
                       "  evens with full p(n)    : {} (sampled segments)\n"
                       "  primes in [start, limit]: {}\n"
                       "  twin-prime pairs        : {}\n",
                       std::getenv("GB_STATS_LOG") ? std::getenv("GB_STATS_LOG") : "goldbach_stats.log",
                       std::getenv("GB_STATS_STATE") ? std::getenv("GB_STATS_STATE") : "goldbach_stats.state",
                       s.stats_pmax, s.stats_tail, s.stats_sampled, s.stats_primes, s.stats_twins);
        if (!csv && !json) print_ridges(s);   // --json / --csv carry ridge_top / ridge_max
        if (per_seg && !json) print_per_segment(s);
        return 0;
    } catch (const std::exception& e) {
        std::println(stderr, "error: {}", e.what());
        return 1;
    }
}
