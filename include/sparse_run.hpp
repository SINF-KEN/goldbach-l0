#pragma once
// Sparse class map pipeline driver (src/sparse.cpp). ScmEngine is also used
// directly by tools/debug_sparse.cpp (per-segment map / checker validation).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sparse.hpp"
#include "verifier.hpp"

namespace gb {

struct ScmOptions {
    std::string spv;            // empty: the geometry's module (kernels/sparse*_bmg.spv)
    uint32_t M = 0;             // modulus (0: GB_SCM_M or SCM_M_DEFAULT)
    const char* prime_cache = nullptr;
    uint32_t pmax = 0;          // 0: GB_SCM_PMAX or the geometry's default
    uint32_t dlt = 0;           // evens per class per segment (0: GB_SCM_DELTA or the geometry's)
    uint32_t split_p = 0;       // tile / large prime bound (0: GB_SCM_SPLIT_P or 131072)
    double target = 64.0;       // plan: phase-A tests per even class (M = 30030: 60..64 flat, pending ~0.02%)
    bool write_checked = false; // validation: phase A / B write checked[]
};

struct ScmSegResult {
    uint64_t N = 0;             // segment base (first even of class 0)
    uint32_t dl = 0, T = 0;
    uint32_t pending = 0, leftovers = 0;
    double tile_s = 0, large_s = 0, check_a_s = 0, check_b_s = 0;
    std::vector<uint64_t> left; // cr << 32 | i
};

/// Two-slot GPU pipeline of the sparse class map: segment s covers the evens
/// [N0 + s D, N0 + (s + 1) D), D = M * dlt, and runs on slot s & 1.
/// Segments must be enqueued in order s = 0, 1, 2, ... (the host first-hit
/// state advances one segment per enqueue), each slot collected before reuse.
class ScmEngine {
public:
    /// Loads base primes, builds (or loads + validates) the checker plan,
    /// loads the kernel module, checks its geometry handshake, uploads the
    /// tables and allocates both slots.
    /// @param start_even  first even to verify; N0 = start_even rounded down to M.
    /// @param limit       last even to verify (inclusive).
    /// @param o           options; zero fields fall back to GB_SCM_* env or defaults.
    /// @pre   N0 well above the candidate range and sqrt(limit) (sparse maps
    ///        do not handle small ranges).
    /// @throws std::runtime_error on out-of-range options, a small start, a
    ///         rejected plan, a module/host geometry mismatch or any GPU failure.
    ScmEngine(uint64_t start_even, uint64_t limit, const ScmOptions& o);
    ~ScmEngine();
    ScmEngine(const ScmEngine&) = delete;
    ScmEngine& operator=(const ScmEngine&) = delete;

    /// Enqueues sieve + checker of segment @p s on slot s & 1 (asynchronous).
    /// @pre   segments 0..s-1 were enqueued before; slot s & 1 is idle (collected).
    /// @post  the slot is live; the first-hit state is advanced to segment s + 1.
    /// @throws std::runtime_error if N_s > limit or a GPU call fails.
    void enqueue(uint64_t s);
    /// Validation: like enqueue, but skips the sieve and runs only the checker
    /// on a caller-supplied map.
    /// @param map  K x stride32 u32 words, the layout read_map returns.
    /// @throws std::runtime_error on a wrongly sized map, or as enqueue.
    void enqueue_check_only(uint64_t s, const std::vector<uint32_t>& map);
    /// Waits for @p slot, reads kernel times and checks the fail-closed
    /// counters; copies the leftovers (evens the GPU could not settle).
    /// @pre   the slot is live.
    /// @post  the slot is idle. Leftovers are NOT yet verified: call resolve().
    /// @return the segment's result.
    /// @throws std::runtime_error on an idle slot, pending or leftover overflow.
    ScmSegResult collect(uint32_t slot);
    /// CPU fallback: verifies every leftover of @p r by Miller-Rabin (candidate
    /// primes first, then all odd p <= n / 2). The ascending scan finds each
    /// leftover's true smallest witness, so this is the sparse path's only
    /// ridge-recording point (the class map cannot compute smallest-prime
    /// witnesses; @p ridges may be null).
    /// @post  returns normally only if every leftover has a Goldbach partition.
    /// @throws std::runtime_error ("Goldbach verification FAILED ...") on a counterexample.
    void resolve(const ScmSegResult& r, RidgeTracker* ridges = nullptr) const;

    // Validation access; valid after collect(slot), before the slot's next enqueue.

    /// Copies the slot's class map (K x stride32 u32 words) into @p out.
    void read_map(uint32_t slot, std::vector<uint32_t>& out);
    /// Copies the slot's checked[] bitmap (CR x chk_stride u64 words).
    /// @pre   constructed with ScmOptions::write_checked.
    /// @throws std::runtime_error if checked[] was not allocated.
    void read_checked(uint32_t slot, std::vector<uint64_t>& out);
    /// Copies the slot's phase-A pending entries (at most pending_cap).
    void read_pending(uint32_t slot, std::vector<uint64_t>& out);
    /// Ridge sampling (GB_RIDGE_SCAN): waits for the slot's segment, copies
    /// the head of its phase-A pending list (small D2H copy) and expands the
    /// miss masks into up to @p max_evens even values n = N + 2cr + M i.
    /// The map cannot tell whether such an n has a small witness, but the
    /// caller can re-derive n's exact smallest prime by an ascending
    /// Miller-Rabin scan (RidgeTracker::consider then validates it again).
    /// @pre   the slot is live (its check_b may still be running; waited here).
    /// @post  the segment is complete; the command list carries one extra copy.
    /// @return the segment start N the evens belong to (0 if nothing sampled).
    uint64_t sample_pending(uint32_t slot, uint32_t max_evens, std::vector<uint64_t>& out);

    const PlanSCM& plan() const;
    const ScmGeo& geo() const;
    uint32_t pshift() const;                  // pending entry: cr << pshift | word << 32 | mask
    uint32_t dlt() const;
    uint32_t tb() const;
    uint64_t n0() const;
    uint32_t split_index() const;
    uint32_t stride32() const;
    uint32_t chk_stride() const;
    const uint32_t* base_primes() const;
    uint32_t base_count() const;
    const std::vector<uint32_t>& cand_primes() const;

    struct Impl;
private:
    std::unique_ptr<Impl> im;   // owned: also freed when the constructor throws
};

/// Verifies Goldbach for every even n in [start_even, limit] with the sparse
/// class map pipeline (ScmEngine, two slots overlapped).
/// @param segment_even  if set and != DEFAULT_SEGMENT_EVEN, selects dlt =
///                      max(segment_even / M, 4096); otherwise the geometry default.
/// @param split_p       tile / large prime bound (0: GB_SCM_SPLIT_P or 131072).
/// Other parameters as run_gpu (verifier.hpp).
/// @pre   start_even (rounded to even, >= 4) <= limit, and high enough for
///        ScmEngine.
/// @post  on return every even in the range has a verified partition.
/// @return timings and counters; segment_used = M * dlt.
/// @throws std::runtime_error on start > limit or as ScmEngine.
RunStats run_gpu_sparse(uint64_t limit, uint64_t segment_even, const std::string& spv_path,
                        bool verbose, uint64_t start_even, uint32_t split_p,
                        const char* prime_cache_path, bool collect_segments);

} // namespace gb
