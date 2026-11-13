#pragma once
// GPU verification driver: segmented prime-map construction + Goldbach check.

#include <cstdint>
#include <string>
#include <vector>

#include "ridge.hpp"

namespace gb {

// Per-segment measurements (populated when collect_segments is set).
// GPU times come from device kernel timestamps; wall time is host-side.
struct SegmentStats {
    uint64_t seg_start      = 0;   // first even n of the segment
    uint64_t seg_end        = 0;   // last even n of the segment (inclusive)
    uint64_t evens          = 0;
    uint64_t tiles          = 0;   // SLM tile work-groups in the tile sieve
    uint64_t large_items    = 0;   // large-prime kernel work-items
    uint32_t sieve_split    = 0;   // split index in force for this segment
    uint32_t mask_count     = 0;   // small-prime mask entries
    uint32_t leftovers      = 0;   // unresolved evens in this segment
    double   sieve_seconds  = 0;   // make_prime_map GPU time
    double   large_seconds  = 0;   // large-prime kernel GPU time
    double   check_seconds  = 0;   // checker GPU time
    double   offset_seconds = 0;   // host offset generation for this segment
    double   wall_seconds   = 0;   // host wall time inside the loop iteration
};

struct RunStats {
    uint64_t limit            = 0;   // evens in [4, limit] verified
    uint64_t start            = 0;   // first even n verified (scale testing)
    uint64_t segment_used     = 0;   // segment span actually used when it differs from
                                     // --segment (sparse path: M * dlt); 0 = the argument
    uint64_t segments         = 0;
    uint64_t leftovers        = 0;   // numbers the GPU could not resolve (should be 0)
    double   init_seconds     = 0.0; // prime generation + L0 setup + uploads
    double   init_cache_s     = 0.0; // init: base-prime cache load/generation
    double   init_module_s    = 0.0; // init: L0 context + module (JIT) + kernels
    double   init_upload_s    = 0.0; // init: staging uploads incl. completion sync
    double   gpu_seconds      = 0.0; // segment loop only
    double   total_seconds    = 0.0;
    double   evens_per_second = 0.0;
    double   sieve_seconds    = 0.0; // GPU time in make_prime_map (event-timed)
    double   large_seconds    = 0.0; // GPU time in the large-prime kernel
    double   verify_seconds   = 0.0; // GPU time in the checker (event-timed)
    double   sync_seconds     = 0.0; // residual wall time (may overlap kernels)
    double   offset_seconds   = 0.0; // host large-prime offset preparation
    double   table_seconds   = 0.0; // host per-tile first-mark table build
    uint64_t total_tiles      = 0;   // sum of tile work-groups over all segments
    uint64_t total_large_items = 0;  // sum of large-prime kernel work-items
    uint32_t split_used       = 0;   // dense path: base-prime split index in force
                                     // (--split 0 = auto resolved); 0 = not reported
    // Goldbach ridge tracking (include/ridge.hpp): dense path (kernel capture
    // + CPU fallback) and sparse CPU fallback. ridge_min = 0: tracking off.
    uint32_t ridge_min    = 0;       // effective recording floor
    uint64_t ridge_found  = 0;       // records the kernels / fallbacks appended
    uint64_t ridge_dropped = 0;      // records rejected by exact validation
    std::vector<RidgeEntry> ridge_top;  // the kept top list, ridge descending
    // Exact statistics (include/gstats.hpp), dense path.
    bool     stats_on     = false;
    uint64_t stats_pmax   = 0;       // largest p(n) of the run
    uint64_t stats_primes = 0;       // primes in [start, limit]
    uint64_t stats_twins  = 0;       // twin pairs (q, q + 2) in [start, limit]
    uint64_t stats_sampled = 0;      // evens with a full p(n) record (sampled segments)
    uint64_t stats_tail   = 0;       // evens with p(n) >= the tail bound (all segments)
    std::vector<SegmentStats> per_segment; // filled only if collect_segments
};

/// Verifies Goldbach for every even n in [max(start_even, 4) rounded up to
/// even, limit] on the GPU (mod-30 class map, hit lists / packed / tile
/// sieves, mod-30 checker; legacy variants selected by GB_* env). The range
/// below start is assumed verified elsewhere (slices of a campaign).
///
/// @param limit             last n to verify (inclusive).
/// @param segment_even      integer span per pipeline segment (rounded up to
///                          even; shrunk so the run has at least two segments).
/// @param spv_path          SPIR-V module built from kernels/goldbach.cl.
/// @param verbose           per-segment progress on stdout.
/// @param start_even        first n (rounded up to even, at least 4).
/// @param sieve_split_arg   base primes below this index use the SLM tile walk,
///                          the rest the large-prime kernels; 0 = auto.
/// @param prime_cache_path  disk cache for the base-prime list (nullptr: default name).
/// @param collect_segments  fill RunStats::per_segment.
/// @pre   a Level Zero GPU is present and spv_path is the module of this
///        source; 4 <= limit <= MAX_SUPPORTED_LIMIT (2^64 - 2^40); some even
///        n >= start in [start, limit]; the segment's odd count fits the
///        32-bit kernel offsets (else std::runtime_error before GPU work).
/// @post  on return, every even n in the range has a verified partition
///        n = p + q: p from the candidate list (primes < MAX_GB_PRIME) with
///        the map bit of q set (the map holds exactly the primes of its
///        classes), or p, q both prime by deterministic Miller-Rabin in the
///        CPU fallback (which continues past MAX_GB_PRIME). With statistics
///        on (GB_STATS, mod-30 checker, start >= 1e6), the state / log /
///        records files hold this run's exact counts (merged once).
/// @return timings and counters of the run.
/// @throws std::runtime_error on any Level Zero error, device error flag,
///         pending / leftover / hit-list overflow, map mismatch under
///         GB_BUCKET_VERIFY, a statistics state that cannot be read or
///         written, or a Goldbach counterexample (none is expected; it is
///         reported, never passed). Nothing is reported as PASS then.
RunStats run_gpu(uint64_t limit,
                 uint64_t segment_even,
                 const std::string& spv_path,
                 bool verbose,
                 uint64_t start_even = 4,
                 uint32_t sieve_split_arg = 0,
                 const char* prime_cache_path = nullptr,
                 bool collect_segments = false);

/// Prints device / driver / runtime information and the build's tuning
/// constants to stdout, one block per Level Zero GPU.
/// @throws std::runtime_error if zeInit fails or no GPU device is reported.
void print_device_info();

/// Independent CPU reference check by trial division (no GPU, no sieve);
/// intended for small limits only.
/// @param limit    last even n checked (inclusive).
/// @param verbose  print progress every 1e6 and the first failure.
/// @return true when every even n in [4, limit] has a Goldbach partition.
bool verify_cpu(uint64_t limit, bool verbose);

} // namespace gb
