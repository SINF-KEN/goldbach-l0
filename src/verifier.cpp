#include "verifier.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "config.hpp"
#include "env.hpp"
#include "l0_context.hpp"
#include "primes.hpp"
#include "split_check.hpp"
#include "check30.hpp"
#include "gstats.hpp"
#include "host_math.hpp"

namespace gb {
namespace {

using clock_ = std::chrono::steady_clock;
/// @return seconds elapsed since @p t0.
double seconds_since(clock_::time_point t0) {
    return std::chrono::duration<double>(clock_::now() - t0).count();
}

// GPU buffer layout (all allocated once, reused for every segment):
//
//   d_prime_map   : odd-prime bitset for the current q segment
//                   bit i  <->  q = q_first + 2*i
//   d_checked     : bit-packed verification state, bit i <-> even n index i
//   d_gb_primes   : uint16 candidate primes p <= 9781 (uploaded once)
//   d_sieve_primes: uint32 base primes (uploaded once)
//   d_leftovers   : uint32 indices of unresolved even numbers
//   d_leftover_n  : uint32 count of leftovers (device counter)
struct GpuBuffers {
    // Read-only tables shared by both pipeline slots (uploaded once).
    uint16_t* gb_primes    = nullptr;
    uint32_t* sieve_primes = nullptr;
    uint64_t* magic        = nullptr;  // per-prime ceil(2^64 / p) reciprocal
    uint32_t* shifts       = nullptr;  // per-prime shift for div-by-p

    // Per-slot buffers: the segment pipeline alternates between slot 0 and
    // slot 1 so segment k's GPU work overlaps segment k-1's host work.
    uint64_t* prime_map[2]  = {};  // odd-prime bitset for the q segment
    uint64_t* checked[2]    = {};  // bit i <-> even n index i
    uint64_t* first_bits[2] = {};  // per-prime first markable bit (segment-global)
    uint64_t* mask_table[2] = {};  // small-prime word masks (per segment: q_first-dependent)
    uint32_t* leftovers[2]  = {};  // compact exception list
    uint32_t* leftover_n[2] = {};  // leftover count (shared: host-readable)
    uint32_t* tile_first[2] = {};  // [tile * sieve_split + j] per-tile first mark
    uint64_t* group_masks[2] = {}; // fused mask stage: per-group mask tables
    uint32_t* group_y[2] = {};     // class modes: per class x group canonical-table word offsets
    uint32_t* pending[2]    = {};  // two-phase checker: evens left by phase A
    uint32_t* pending_n[2]  = {};  // their count (shared: host-checked for overflow)
    uint64_t* ridge[2]      = {};  // Goldbach ridge appends (shared; count + (n, p) pairs)
    // Exact statistics (gstats.hpp): per-slot p(n) histogram / first-even
    // index per candidate-list entry, count30 limits + output (all shared).
    uint32_t* x_hist[2]     = {};      // sampled segments: counts per stats-plan entry
    uint32_t* p_hist[2]     = {};      // tail p(n) counts by p >> 1 (check30_b)
    uint32_t* p_first[2]    = {};      // tail first even index by p >> 1
    uint32_t* x_lim[2]      = {};
    uint32_t* x_cnt[2]      = {};
    uint64_t* g_part[2]     = {};      // count30 walk: largest gap per work-group (g << 32 | ~position)
    uint32_t* g_tail[2]     = {};      // count30 gap tail: [0] entries, [1] errors, (upper position, g) pairs
    uint32_t* g_hist[2]     = {};      // count30 walk: histogram by g / 2 (bin 0: primes without predecessor)
    uint32_t* want_p        = nullptr; // tail bins without a campaign first occurrence
    uint16_t* x_all         = nullptr;
    uint32_t* x_tab         = nullptr;
    uint32_t* x_ch          = nullptr;
    uint32_t* x_j           = nullptr;
    uint16_t* twin_primes   = nullptr; // lower members of twin pairs <= MAX_GB_PRIME
    uint16_t* rest_primes   = nullptr; // twin checker phase B: candidates phase A did not cover
    uint64_t* ms32          = nullptr; // 32-bit reciprocals for the table path
    // Class-split checker (v1.11.0): per-slot split maps, shared lists.
    uint32_t* split_s1[2]   = {};      // q = 1 (mod 6) bits of the map
    uint32_t* split_s5[2]   = {};      // q = 5 (mod 6) bits of the map
    uint16_t* split_cands   = nullptr; // phase-A funnels (L1, L5)
    uint32_t* split_meta    = nullptr; // stage table
    uint16_t* split_rest    = nullptr; // phase-B probes per class
    uint16_t* c30_cands     = nullptr; // mod-30 checker (check30.hpp)
    uint32_t* c30_meta      = nullptr;
    uint16_t* c30_rest      = nullptr;
    uint32_t* c30_rtab      = nullptr;
};

// Mask-group bound shared with the kernel (check_constants.py checks it).
inline constexpr uint32_t MAX_MASK_GROUPS_HOST = 8;   // kernel MAX_MASK_GROUPS

/// Copies host data to a device buffer through a shared staging allocation.
/// (Device memory is not host-accessible.)
/// @param dst  device buffer of at least @p bytes.
/// @pre   data readable for @p bytes; dst a device allocation of ctx.
/// @post  the copy is enqueued, not complete.
/// @return the staging buffer. The caller must host_sync() before freeing it
///         — the completion sync is LOAD-BEARING: without it the first
///         segment's kernels race the uploads and the device is lost
///         (observed on small multi-segment runs even though all commands
///         share one list).
/// @throws std::runtime_error on allocation or append failure.
void* stage_upload(Context& ctx, const void* data, size_t bytes, void* dst) {
    void* sh = ctx.alloc_shared(bytes);
    std::memcpy(sh, data, bytes);
    ZE_CHECK(zeCommandListAppendMemoryCopy(ctx.list(), dst, sh, bytes,
                                           nullptr, 0, nullptr));
    return sh;
}

// Per-slot kernel-timestamp events: sieve/large/check for each of the two
// pipeline slots. One pool per event so each can be reset independently.
struct SlotEvents {
    ze_event_pool_handle_t pool[20] = {};
    ze_event_handle_t sieve[2] = {};
    ze_event_handle_t large[2] = {};
    ze_event_handle_t check[2] = {};   // last checker kernel (host waits on it)
    ze_event_handle_t checka[2] = {};  // two-phase checker: phase A (timing only)
    ze_event_handle_t bucket[2] = {};  // hl_apply / bucket_process (timing only; counted in large_s)
    ze_event_handle_t splitk[2] = {};  // split_map (timing only; counted in check_s)
    ze_event_handle_t xv[3][2] = {};   // check30_xs8 / 6 / 4 (timing only; counted in check_s)
    ze_event_handle_t hl[2] = {};      // hl_harvest (timing only; counted in large_s)

    /// Creates the twenty pools and events.
    /// @throws std::runtime_error on failure; destroy() is safe afterwards.
    void create(ze_context_handle_t ctx) {
        for (int i = 0; i < 20; ++i) {
            ze_event_pool_desc_t pd{};
            pd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
            pd.count = 1;
            pd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
            ZE_CHECK(zeEventPoolCreate(ctx, &pd, 0, nullptr, &pool[i]));
            ze_event_desc_t ed{};
            ed.stype  = ZE_STRUCTURE_TYPE_EVENT_DESC;
            ed.index  = 0;
            ed.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
            ed.wait   = ZE_EVENT_SCOPE_FLAG_DEVICE;
            ze_event_handle_t* e = (i < 2) ? &sieve[i]
                                 : (i < 4) ? &large[i - 2]
                                 : (i < 6) ? &check[i - 4]
                                 : (i < 8) ? &checka[i - 6]
                                 : (i < 10) ? &bucket[i - 8]
                                 : (i < 12) ? &splitk[i - 10]
                                 : (i < 18) ? &xv[(i - 12) / 2][(i - 12) % 2]
                                            : &hl[i - 18];
            ZE_CHECK(zeEventCreate(pool[i], &ed, e));
        }
    }
    /// Destroys whatever create() made; safe after a partial create.
    void destroy() {
        for (int i = 0; i < 2; ++i) {
            if (sieve[i]) zeEventDestroy(sieve[i]);
            if (large[i]) zeEventDestroy(large[i]);
            if (check[i]) zeEventDestroy(check[i]);
            if (checka[i]) zeEventDestroy(checka[i]);
            if (bucket[i]) zeEventDestroy(bucket[i]);
            if (splitk[i]) zeEventDestroy(splitk[i]);
            for (int v = 0; v < 3; ++v) if (xv[v][i]) zeEventDestroy(xv[v][i]);
            if (hl[i]) zeEventDestroy(hl[i]);
        }
        for (int i = 0; i < 20; ++i)
            if (pool[i]) zeEventPoolDestroy(pool[i]);
    }
};

/// Fetches and resolves the leftover (unresolved) even numbers of a completed
/// segment: CPU fallback with Miller-Rabin, first over @p gb_primes, then over
/// every odd p <= n / 2 (a p_min above MAX_GB_PRIME is reported, not failed).
/// @param leftover_buf  device list of even indices i (n = seg_start + 2i).
/// @param n_left        entries in the list (device counter value).
/// @param seg_evens     evens in the segment: every valid index is < seg_evens.
/// @param ridges        Goldbach ridge tracker (may be null); the ascending
///                      scan finds each leftover's true smallest witness.
/// @param gstats        exact statistics (may be null): records that p(n).
/// @param sampled       the segment is a full-histogram sample (gstats).
/// @pre   the segment's check kernels have completed.
/// @post  a normal return means every leftover n was verified.
/// @throws std::runtime_error on n_left > MAX_LEFTOVERS, an index outside
///         the segment (a corrupted list would otherwise verify the wrong n
///         while the unresolved even goes unchecked), or a genuine Goldbach
///         counterexample.
void resolve_leftovers(Context& ctx, uint32_t* leftover_buf, uint32_t n_left,
                       uint64_t seg_start, uint64_t seg_evens,
                       const std::vector<uint16_t>& gb_primes,
                       bool verbose, uint64_t seg_index, RidgeTracker* ridges,
                       GoldbachStats* gstats, bool sampled) {
    if (verbose) std::printf("  seg %llu: n_left=%u\n",
                             (unsigned long long)seg_index, n_left);
    if (n_left > MAX_LEFTOVERS)
        throw std::runtime_error("leftover overflow (should be impossible)");
    std::vector<uint32_t> idx(n_left);
    ZE_CHECK(zeCommandListAppendMemoryCopy(ctx.list(), idx.data(), leftover_buf,
                                           size_t(n_left) * 4, nullptr, 0, nullptr));
    ctx.host_sync();
    for (uint32_t i : idx) {
        if (i >= seg_evens)
            throw std::runtime_error("leftover index " + std::to_string(i) + " outside the segment (" +
                                     std::to_string(seg_evens) + " evens)");
        const uint64_t n = seg_start + 2ull * i;
        bool ok = (n == 4); // 4 = 2 + 2 (the only even+even partition)
        for (uint32_t p : gb_primes) {
            const uint64_t q = n - p;
            if (p > q) break;
            if (is_prime_u64(q)) {
                ok = true;
                if (ridges) ridges->consider(n, p, seg_start);
                if (gstats) gstats->add_cpu(n, p, sampled);
                break;
            }
        }
        // MAX_GB_PRIME is only the p_min bound PROVEN below 4e18; above it an
        // even may legitimately need a larger p (a new p_min record). Keep
        // scanning odd primes instead of reporting a false counterexample.
        if (!ok && n > 4) {
            for (uint64_t p = MAX_GB_PRIME + 2; p <= n / 2; p += 2) {
                if (!is_prime_u64(p)) continue;
                if (is_prime_u64(n - p)) {
                    ok = true;
                    if (ridges) ridges->consider(n, p, seg_start);
                    if (gstats) gstats->add_cpu(n, p, sampled);
                    std::fprintf(stderr, "note: n=%llu needs p=%llu > MAX_GB_PRIME "
                                 "(new p_min record candidate)\n",
                                 (unsigned long long)n, (unsigned long long)p);
                    break;
                }
            }
        }
        if (!ok)
            throw std::runtime_error("Goldbach verification FAILED for n=" +
                                     std::to_string(n));
    }
}

// ---------------------------------------------------------------------------
// run_gpu, split into phases.
//
// RunSetup holds the arguments and the host-only setup (no Level Zero: range
// clamping, base primes); DenseRun adds the Level Zero state and runs the
// phases in the order run_gpu always ran them. Every object a phase creates
// is a member declared in the order run_gpu declared it as a local, so the
// destructors (Kernel, Module, Context) run in the same order on return and
// on an exception.
// ---------------------------------------------------------------------------
struct RunSetup {
    // Arguments of run_gpu (limit / start_even / segment_even are clamped
    // in place, as the parameters were).
    uint64_t limit;
    uint64_t segment_even;
    const std::string& spv_path;
    bool verbose;
    uint64_t start_even;
    uint32_t sieve_split_arg;
    const char* prime_cache_path;
    bool collect_segments;

    RunSetup(uint64_t limit_, uint64_t segment_even_, const std::string& spv_path_,
             bool verbose_, uint64_t start_even_, uint32_t sieve_split_arg_,
             const char* prime_cache_path_, bool collect_segments_)
        : limit(limit_), segment_even(segment_even_), spv_path(spv_path_),
          verbose(verbose_), start_even(start_even_), sieve_split_arg(sieve_split_arg_),
          prime_cache_path(prime_cache_path_), collect_segments(collect_segments_) {
        resolve_range();
        load_base_primes();
        // ---- Level Zero setup starts here: DenseRun's members follow.
        t_module = clock_::now();
        init_mark("pre-module");
    }

    // ---- resolve_range ----------------------------------------------------
    uint32_t phase_b_groups = 0;     // GB_CHECK_BGROUPS
    uint32_t stats_b_groups = 0;     // GB_STATS_BGROUPS
    uint32_t bucket_groups_env = 0;  // GB_BUCKET_GROUPS (0: unset, mode default)
    RunStats stats;
    clock_::time_point t_start;
    uint64_t total_evens = 0;        // evens in [start_even, limit]

    // ---- load_base_primes -------------------------------------------------
    gb::PrimeCacheView prime_view;
    const uint32_t* sieve_primes = nullptr;  // mmap-backed (or owned) by prime_view
    double cache_seconds = 0.0;
    uint32_t mask_count = 0;         // primes < 64: the small-prime mask path
    uint32_t sieve_count = 0;        // base primes
    uint32_t sieve_split = 0;        // [0, split): tile walk; [split, count): large kernels
    uint32_t large_count = 0;
    bool init_trace = false;         // GB_INIT_TRACE
    clock_::time_point t_mark;       // init_mark's previous mark
    std::vector<uint32_t> gb_all;    // odd candidate primes <= MAX_GB_PRIME
    std::vector<uint16_t> gb_primes;
    std::vector<uint16_t> twin_primes;  // lower members of the twin pairs among them
    clock_::time_point t_module;     // start of the Level Zero setup

    void resolve_range();
    void load_base_primes();
    void init_mark(const char* what);
};

class DenseRun : RunSetup {
public:
    DenseRun(uint64_t limit_, uint64_t segment_even_, const std::string& spv_path_,
             bool verbose_, uint64_t start_even_, uint32_t sieve_split_arg_,
             const char* prime_cache_path_, bool collect_segments_)
        : RunSetup(limit_, segment_even_, spv_path_, verbose_, start_even_, sieve_split_arg_,
                   prime_cache_path_, collect_segments_),
          ctx(true),   // unordered list, barrier per launch (§9.2v)
          mod(ctx, spv_path),
          k_sieve(mod, "make_prime_map"),
          k_large(mod, "sieve_large_packed"),
          k_check(mod, "check_phase_a"),
          k_check_b(mod, "check_phase_b") {}

    RunStats run();

private:
    // Level Zero objects, created in this order by the constructor (and
    // destroyed in reverse order after every member declared below).
    Context ctx;
    Module  mod;
    Kernel  k_sieve;
    Kernel  k_large;
    // Two-phase checker (twin form, the fallback of the class-split and
    // mod-30 checkers): check_phase_a (32-bit register-window word checker
    // over twin pairs) + check_phase_b (per-even sparse continuation).
    Kernel  k_check;
    Kernel  k_check_b;

    // ---- configure_checkers ---------------------------------------------------
    uint32_t a_w64 = 0;              // phase-A item width (CHECK_PHASE_A_W64)
    uint32_t twin_cap = 0;           // GB_TWIN_CAP
    std::vector<uint16_t> rest_primes;  // twin phase B: candidates not covered
    bool split_on = false;           // GB_SPLIT: class-split checker
    SplitPlan split_plan;
    uint32_t split_write_checked = 0;
    std::unique_ptr<Kernel> k_split, k_sa, k_sb, k_merge;
    bool class_mode = false;         // GB_CLASSMAP != 0: class map
    bool c30 = false;                // mod-30 class map
    uint32_t ncls_map = 0;           // class arrays of the map (8 / 2)
    bool c30chk = false;             // GB_C30: mod-30 checker
    Plan30 plan30;
    std::unique_ptr<Kernel> k_ca, k_cb;
    uint32_t cm_u32 = 0;             // kernel class-mode argument (0 / 1 / 2)

    // ---- configure_ridges_and_stats ---------------------------------------------
    // optional: constructed (it reads GB_RIDGE*) where run_gpu constructed it
    std::optional<RidgeTracker> ridges;
    std::string ridge_path;          // GB_RIDGE_FILE
    const char* ridge_file = nullptr;  // ridge_path.c_str()
    uint32_t ceil_fallback = 0;      // phase-A ceilings per checker (ridge floor)
    uint32_t ceil_split = 0;
    uint32_t ceil_c30 = 0;
    bool stats_on = false;           // exact statistics (GB_STATS, mod-30 checker)
    bool gaps_on = false;            // prime gaps (GB_GAPS)
    bool gaps_exact = false;         // count30 walk on every segment
    uint32_t gaps_groups = 0;        // GB_GAPS_GROUPS
    StatsPlan splan;
    std::unique_ptr<GoldbachStats> gstats;
    std::unique_ptr<Kernel> k_cxs[4], k_cxb, k_cnt;
    uint32_t stats_sample = 0;       // GB_STATS_SAMPLE

    // ---- set_group_sizes ------------------------------------------------------
    uint32_t own_end = 0;            // primes [0, own_end): thread-owned tile walk
    double module_seconds = 0.0;
    uint32_t large_wg = 0;           // GB_LARGE_WG

    void configure_checkers();
    void configure_ridges_and_stats();
    // ---- size_buffers ---------------------------------------------------------
    GpuBuffers b;                    // every device / shared buffer (allocated below)
    uint64_t map_words = 0;          // prime map per slot (whole tiles)
    uint32_t chk_stride = 0;         // class-split checker: words per class of checked[]
    uint32_t chk_stride30 = 0;       // mod-30 checker: the same
    uint64_t chk_words = 0;          // checked[] per slot
    uint32_t split_words = 0;        // split maps S1 / S5 per slot
    uint32_t pending_cap = 0;        // two-phase checker: pending list capacity
    uint64_t table_bytes = 0;        // per-tile first-mark table per slot
    bool use_table = false;
    uint32_t use_table_u32 = 0;
    uint32_t mask_period = 0;        // GB_MASK_PERIOD
    uint32_t mask_groups_n = 0;      // fused mask groups (0: per-prime mask loop)
    size_t mask_group_words = 0;
    uint32_t* d_group_desc = nullptr;

    void set_group_sizes();
    void size_buffers();
};

/// Phase 1: validates and clamps the range and segment span (run_gpu's
/// preconditions; the n = 4 fail-closed rule) and reads the grid-stride
/// work-group counts. No Level Zero call.
void RunSetup::resolve_range() {
    if (limit < 4) throw std::runtime_error("limit < 4: no even n >= 4 to verify");
    gb::require_supported_limit(limit);
    // The last even n is limit rounded down (as run_gpu_sparse does). An odd
    // limit left the segment and count30 ranges ending on an even q, so the
    // statistics missed a prime at limit - 2 (and the twin pair below it)
    // while the gap sum still included it.
    limit &= ~uint64_t(1);
    // Grid-stride work-group counts, validated before any GPU work: env_grid
    // refuses 0, which would skip phase B (pending evens never checked) or the
    // hit-list / bucket marks (composites left "prime") and still PASS.
    phase_b_groups = env_grid("GB_CHECK_BGROUPS", CHECK_PHASE_B_GROUPS);
    stats_b_groups = env_grid("GB_STATS_BGROUPS", 160u);
    bucket_groups_env = env_grid("GB_BUCKET_GROUPS", 0u);   // 0: unset (mode default)
    stats.limit = limit;
    t_start = clock_::now();

    // Scale-test entry point: verify [start, limit] instead of [4, limit].
    // Clamp to an even value >= 4 and below limit.
    if (start_even < 4) start_even = 4;
    if (start_even & 1) start_even += 1;
    if (start_even > limit) throw std::runtime_error("no even n in [start, limit]");
    if (segment_even < 2) segment_even = 2;
    if (segment_even & 1) segment_even += 1;

    // Guarantee at least two segments so the two-slot pipeline overlaps host
    // enqueue with GPU execution even for small runs (one giant segment gives
    // the driver a single huge launch and no overlap at all).
    total_evens = (limit - start_even) / 2 + 1;
    // segment_even is an integer span, so it covers segment_even/2 evens.
    if (segment_even > total_evens)
        segment_even = std::max<uint64_t>(total_evens & ~uint64_t(1), 2);
    // n = 4 = 2 + 2 has no odd-prime partition, so a segment must not hold it
    // alone (no odd q to sieve; segment_geometry throws): the first segment
    // spans at least [4, 6], and [4, 4] is not a GPU range at all.
    if (start_even == 4) {
        if (limit < 6)
            throw std::runtime_error("the GPU path needs limit >= 6 (n = 4 = 2 + 2 has no "
                                     "odd-prime partition; --cpu-check 4 verifies it)");
        segment_even = std::max<uint64_t>(segment_even, 4);
    }
}

/// Phase 2: base primes (cache load), the tile-walk / large-prime split and
/// the Goldbach candidate lists. No Level Zero call.
void RunSetup::load_base_primes() {
    // ------------------------------------------------ CPU-side setup ------
    // Base primes for the segmented sieve: need p <= sqrt(max q) = sqrt(limit).
    // Segmented generator: O(block) memory instead of O(sqrt(limit)) — the
    // flat sieve would need a 2 GB bitmap at limit=4e18.
    const uint32_t sqrt_limit = gb::base_prime_bound(limit);
    const auto t_cache = clock_::now();
    prime_view = gb::PrimeCacheView::load(
        std::max<uint64_t>(sqrt_limit, 64),
        prime_cache_path ? prime_cache_path : "base_primes.cache");
    // mmap-backed (or owned) read-only prime list; no intermediate copy.
    sieve_primes = prime_view.data();
    cache_seconds = seconds_since(t_cache);

    // Primes handled by the kernel's small-prime mask path (stage 3a): the
    // first mask_count entries of sieve_primes (ascending) are all < 64.
    // Keep in sync with MASK_MAX_P in kernels/goldbach.cl (table indexing
    // requires p < 64 so the phase fits the 64-entry per-prime table).
    mask_count = uint32_t(
        std::lower_bound(sieve_primes, sieve_primes + prime_view.size(), 64u) -
        sieve_primes);

    // Split point between the SLM tile walk (primes < split) and the flat
    // one-work-item-per-prime large kernel (primes >= split). The tile walk
    // costs O(pi(split)) per thread per tile; the large kernel costs
    // O(segment_odds/p) per prime. At N=4e18, pi(sqrt(N)) ~ 98M — walking
    // ALL base primes per tile blew the driver watchdog (engine reset), so
    // the split is what makes 4e18 feasible at all.
    // 0 = auto: keep the tile walk's per-thread prime loop around 64K
    // iterations (pi(1e6) ~ 78K) and hand everything above to the large
    // kernel. Tunable via --split for the sweep tool.
    sieve_count = uint32_t(prime_view.size());
    sieve_split = sieve_split_arg;
    // 0 = auto: 30000 (~350K) with 16 KB tiles (v1.8.1): moving primes out
    // of the global-atomic large kernel into the SLM medium stage pays until
    // its per-prime-per-tile setup dominates. 2e10 window at 4e18 (16 KB
    // tiles): 20K 21.10G, 25K 21.15G, 30K 21.18G, 35K 21.17G, 40K 21.14G.
    // (64 KB tiles: 40K optimum.) Keep explicit --split for sweeps.
    // v1.11.0 class map, 350M spans, per 1e11 at 4e18: 30K 945.6, 32K 939.7,
    // 34K 940.6, 36K 942.4, 40K 945.7 ms kernel time -> 32K.
    // v1.13.0 (mod-30 checker, 425M spans), gpu per 1e11 vs 32K: 28K +2.6%,
    // 34K -0.11%, 36K -0.11% (long-end -0.06%), 38K 0.0%, 42K +0.27% -> 36K.
    if (sieve_split == 0 || sieve_split > sieve_count)
        sieve_split = std::min<uint32_t>(sieve_count, 36000);
    sieve_split = std::max(sieve_split, mask_count);
    large_count = sieve_count - sieve_split;

    // Goldbach candidate primes: p <= 9781 (see config.hpp).
    init_trace = env::present("GB_INIT_TRACE");
    t_mark = clock_::now();
    init_mark("cache+setup");
    gb_all = odd_primes_up_to(MAX_GB_PRIME);
    gb_primes.reserve(gb_all.size());
    for (uint32_t p : gb_all) gb_primes.push_back(uint16_t(p));
    // Lower members of the twin pairs among the candidates (twin checker).
    for (size_t i = 0; i + 1 < gb_all.size(); ++i)
        if (gb_all[i + 1] == gb_all[i] + 2) twin_primes.push_back(uint16_t(gb_all[i]));
}

/// GB_INIT_TRACE: init time since the previous mark, to stderr.
void RunSetup::init_mark(const char* what) {
    if (!init_trace) return;
    std::fprintf(stderr, "  init %-34s %7.1f ms\n", what, seconds_since(t_mark) * 1e3);
    t_mark = clock_::now();
}

/// Phase 3a: the checker family (twin / class-split / mod-30) and the map
/// layout (dense, mod-6 or mod-30 class map) from GB_*; creates their kernels.
void DenseRun::configure_checkers() {
    // Phase A item width (64-bit words): the kernel's compiled-in A_W64
    // (tests/check_constants.py). Deliberately not overridable at run time:
    // a host-only change sizes the phase-A launch for a different width and
    // leaves evens unchecked.
    a_w64 = CHECK_PHASE_A_W64;
    // Twin checker (v1.10.0): phase A reads the map as a twin map with the lower
    // members of twin pairs (two candidates per funnel); phase B then probes
    // ALL candidates of the pending evens. GB_TWIN_CAP caps the twin funnels.
    twin_cap = env::u32("GB_TWIN_CAP", CHECK_TWIN_CANDS);
    // Phase B of the twin checker probes only the candidates the first
    // twin_cap twin pairs did not cover (ascending): every pending even has
    // already failed all of those.
    {
        std::vector<bool> covered(MAX_GB_PRIME + 3, false);
        for (uint32_t i = 0; i < std::min<size_t>(twin_cap, twin_primes.size()); ++i) {
            covered[twin_primes[i]] = true;
            covered[twin_primes[i] + 2] = true;
        }
        for (uint16_t p : gb_primes)
            if (!covered[p]) rest_primes.push_back(p);
    }
    // Class-split checker (v1.11.0, production): evens grouped by n mod 3,
    // map split by q mod 6, funnels over sexy-prime triplets then pairs
    // (include/split_check.hpp, kernels/goldbach.cl). GB_SPLIT=0 restores the
    // twin checker; GB_SPLIT_CAP = funnels per class list, GB_SPLIT_STAGES =
    // stage masks ("7/3": p, p+6, p+12 then p, p+6). Used per segment only
    // where the geometry allows it (split_segment().ok), else twin.
    split_on = env::onoff("GB_SPLIT", true);
    // checked[] is validation output only (tools/debug_check_hi runs the
    // kernels itself): the production checkers do not write it.
    split_write_checked = 0u;
    if (split_on) {
        const uint32_t cap = env::u32("GB_SPLIT_CAP", CHECK_SPLIT_CANDS);
        const char* st = env::raw("GB_SPLIT_STAGES");   // parse_split_stages validates it
        split_plan = make_split_plan(gb_all, cap, parse_split_stages(st ? st : CHECK_SPLIT_STAGES));
        k_split = std::make_unique<Kernel>(mod, "split_map");
        k_merge = std::make_unique<Kernel>(mod, "merge30");
        k_sa = std::make_unique<Kernel>(mod, "check_split_a");
        k_sb = std::make_unique<Kernel>(mod, "check_split_b");
    }
    // Class-map sieve (v1.11.0): the map is built as the q = 1 / 5 (mod 6)
    // arrays the split checker reads (kernels/goldbach.cl map_clear): 2/3 of
    // the tiles, no split_map pass. GB_CLASSMAP=0: dense map + split_map.
    const std::string classmap_env = env::choice("GB_CLASSMAP", "30", {"30", "6", "0"}, true);
    class_mode = split_on && classmap_env != "0";
    // Class map mod 30 (v1.12.0 default): 8 class arrays for the residues
    // coprime to 30; merge30 rebuilds the checker's S1 / S5. GB_CLASSMAP=6:
    // the v1.11.0 mod-6 class map (the checker reads it directly).
    c30 = class_mode && classmap_env != "6";
    ncls_map = c30 ? 8u : 2u;
    // Mod-30 checker (v1.13.0, include/check30.hpp): evens grouped by n mod
    // 30 read the class arrays directly (no merge30). GB_C30=0: the v1.12.0
    // merge30 + class-split checker; GB_C30_PLAN overrides the plan.
    c30chk = c30 && env::onoff("GB_C30", true);
    if (c30chk) {
        const char* pe = env::raw("GB_C30_PLAN");   // make_plan30 validates it
        plan30 = make_plan30(gb_all, pe ? pe : CHECK30_PLAN);
        k_ca = std::make_unique<Kernel>(mod, "check30_a");
        k_cb = std::make_unique<Kernel>(mod, "check30_b");
    }
    cm_u32 = class_mode ? (c30 ? 2u : 1u) : 0u;
}

/// Phase 3b: Goldbach ridge tracking (floor clamp, store load) and the exact
/// statistics (GB_STATS / GB_GAPS: plan, state, count30 / check30_x kernels).
void DenseRun::configure_ridges_and_stats() {
    // ---- Goldbach ridge tracking (include/ridge.hpp) -----------------------
    // The phase-B checkers record the smallest witness of every pending even
    // they resolve (their rest lists are ascending, so the first hit is the
    // ridge; the mod-30 family probes useful residues before the futile ones,
    // exact for n > MAX_GB_PRIME + 5 — the host's exact validation drops the
    // rare tiny-n records where that ordering could differ). Since v2.2 every
    // phase-A funnel member is below the plan's bound (3000), so every even
    // with a ridge >= 3000 reaches phase B: the top list is complete there
    // (below the bound phase A may resolve an even first; kept entries are
    // exact either way).
    ridges.emplace(&gb_all);
    ridge_path = env::path("GB_RIDGE_FILE", "goldbach_ridges.csv");
    ridge_file = ridge_path.c_str();
    // Phase-A ceilings: every candidate a checker's phase A tests is below
    // its ceiling, so an even whose ridge reaches the ceiling fails phase A
    // and phase B records its exact ridge. Below it phase A may resolve the
    // even through a larger candidate first (unrecorded), so the floor is
    // clamped to the production checker's ceiling (complete above it; a
    // GB_C30_PLAN without a bound prefix gets 9782, i.e. no kernel ridges)
    // and segments on a fallback checker record from that one's ceiling.
    const size_t twin_n = std::min<size_t>(twin_cap, twin_primes.size());
    ceil_fallback = twin_n ? twin_primes[twin_n - 1] + 3u : 0u;
    ceil_split = split_on ? split_plan.max_member + 1u : 0u;
    ceil_c30 = c30chk ? plan30.member_lt : 0u;
    const uint32_t a_ceiling = c30chk ? ceil_c30 : split_on ? ceil_split : ceil_fallback;
    if (ridges->active() && a_ceiling) ridges->clamp_floor_above(a_ceiling - 1u);
    // Load the store only after the clamp: load() filters by the floor, and
    // entries below the clamped floor (a low GB_RIDGE_MIN) would otherwise
    // stay in the top list and set the kernel threshold from a region the
    // kernels no longer record completely.
    if (ridges->active()) ridges->load(ridge_file);
    // ---- Exact statistics (include/gstats.hpp) -----------------------------
    // p(n) for every even, prime / twin counts. Needs the mod-30 checker and
    // a start far above MAX_GB_PRIME (futile-last candidate order); GB_STATS=0
    // turns it off (plain verification, check30_a / check30_b).
    const bool stats_env = env::onoff("GB_STATS", true);
    stats_on = c30chk && start_even >= STATS_MIN_START && stats_env;
    // prime gaps (count30's tail and walk): with the statistics unless GB_GAPS=0
    const std::string gaps_env = env::choice("GB_GAPS", "exact", {"exact", "1", "sampled", "0"}, true);
    gaps_on = stats_on && gaps_env != "0";
    // default: the full gap walk (count30) on every segment -- exact N(g) and
    // moments everywhere. GB_GAPS=sampled walks only the
    // sampled segments (count30's exact count / tail / maximum everywhere).
    gaps_exact = gaps_on && gaps_env != "sampled";
    // count30 grid on walked segments (work-groups striding over the segment;
    // GB_GAPS_GROUPS tunes it). The default keeps the B580-tuned share of the
    // device: GAPS30_GROUPS per GAPS30_REF_EUS EUs, scaled by this device's
    // EU count (the walk / phase-A overlap optimum moves with it).
    uint32_t gaps_default = GAPS30_GROUPS;
    {
        ze_device_properties_t dp{};
        dp.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        if (zeDeviceGetProperties(ctx.device(), &dp) == ZE_RESULT_SUCCESS) {
            const uint64_t eus = uint64_t(dp.numSlices) * dp.numSubslicesPerSlice * dp.numEUsPerSubslice;
            if (eus) gaps_default = uint32_t(std::clamp<uint64_t>(
                (uint64_t(GAPS30_GROUPS) * eus + GAPS30_REF_EUS / 2) / GAPS30_REF_EUS, 1, GAPS30_GROUPS_MAX));
        }
    }
    gaps_groups = env::u32("GB_GAPS_GROUPS", gaps_default, 1u, GAPS30_GROUPS_MAX);
    const uint32_t stats_k = env::u32("GB_STATS_K", STATS_K_DEFAULT, 1u, STATS_KMAX);
    stats_sample = env::u32("GB_STATS_SAMPLE", STATS_SAMPLE_DEFAULT, 1u);
    if (stats_on) {
        splan = make_stats_plan(gb_all, stats_k, plan30.member_lt);
        gstats = std::make_unique<GoldbachStats>(gb_all, splan, plan30.member_lt);
        gstats->set_run_start(start_even);   // before want_p reaches the device
        for (int v = 0; v < 4; ++v) k_cxs[v] = std::make_unique<Kernel>(mod, XS_KERNELS[v]);
        k_cxb = std::make_unique<Kernel>(mod, "check30_xb");
        k_cnt = std::make_unique<Kernel>(mod, "count30");
        if (gaps_on) gstats->begin_gaps(start_even);
    } else if (c30chk && start_even < STATS_MIN_START && !env::present("GB_STATS")) {
        std::fprintf(stderr, "note: Goldbach statistics off below start %llu\n",
                     (unsigned long long)STATS_MIN_START);
    }
}

/// Phase 3c: the tile walk's thread-owned prime range (own_end), the end of
/// the module timing, and the work-group size of every kernel created so far.
void DenseRun::set_group_sizes() {
    // Tile-walk split between the thread-owned walk (primes < MED_MIN_P,
    // several marks per thread range) and the prime-parallel SLM-atomic walk
    // (primes >= MED_MIN_P, a few marks per tile). GB_MED_P overrides the
    // bound for sweeps; a bound >= the split prime disables the medium stage.
    const uint32_t med_min_p = env::u32("GB_MED_P", class_mode ? MED_MIN_P_CLASS : MED_MIN_P);
    own_end = std::clamp<uint32_t>(uint32_t(
        std::lower_bound(sieve_primes, sieve_primes + sieve_split, med_min_p) -
        sieve_primes), mask_count, sieve_split);
    module_seconds = seconds_since(t_module);
    init_mark("module");

    k_sieve.set_group_size(WG_SIZE);
    // Large-kernel work-group size; GB_LARGE_WG=N overrides (tuning only: the
    // kernel indexes by global id). N must be a non-zero multiple of the
    // SIMD32 subgroup: 0 would divide the launch by zero, anything else
    // splits a subgroup.
    large_wg = env::wg32("GB_LARGE_WG", LARGE_WG_SIZE);
    k_large.set_group_size(large_wg);
    k_check.set_group_size(WG_SIZE);
    k_check_b.set_group_size(WG_SIZE);
    if (split_on) {
        k_split->set_group_size(WG_SIZE);
        k_merge->set_group_size(WG_SIZE);
        k_sa->set_group_size(WG_SIZE);
        k_sb->set_group_size(WG_SIZE);
    }
    if (c30chk) {
        k_ca->set_group_size(WG_SIZE);
        k_cb->set_group_size(WG_SIZE);
    }
    if (stats_on) {
        for (auto& k : k_cxs) k->set_group_size(WG_SIZE);
        k_cxb->set_group_size(WG_SIZE);
        k_cnt->set_group_size(WG_SIZE);
    }
}

/// Phase 4: sizes of the per-slot buffers (map, checked, split maps, pending
/// list, tile table) and the fused small-prime mask grouping (allocates and
/// fills its shared descriptor table).
void DenseRun::size_buffers() {
    // ------------------------------------------------- Persistent buffers -
    // q range spans [seg_start - 9781, seg_end - 3]: that is the segment plus a
    // 9781 back-extension, i.e. segment_even + ~4891 ODD values.
    // The sieve writes WHOLE tiles (one work-group = 16,384 words), so the map
    // must be sized in whole tiles — otherwise the last tile writes past the
    // end of the buffer and hangs the GPU (device lost).
    // segment_even is an integer span, not an even-count. Account for the
    // one-even tail that is merged into the preceding segment below.
    const uint64_t max_segment_evens = segment_even / 2 + 1;
    const uint64_t max_odds   = max_segment_evens + (MAX_GB_PRIME + 1) / 2 + 2;
    if (max_odds >= uint64_t(std::numeric_limits<uint32_t>::max()))
        throw std::runtime_error("segment span too large for 32-bit large-prime offsets");
    const uint64_t dense_tiles = (max_odds + BITS_PER_TILE - 1) / BITS_PER_TILE;
    // Class mode: two class arrays of tpc_max whole tiles each (<= dense + 2).
    const uint64_t tpc_max = (max_odds / (c30 ? 15 : 3) + 2 + BITS_PER_TILE - 1) / BITS_PER_TILE;
    const uint64_t map_tiles  = class_mode ? std::max(dense_tiles, ncls_map * tpc_max) : dense_tiles;
    map_words  = map_tiles * TILE_WORDS;
    // Class-split checker: checked[] holds the three classes at chk_stride
    // words each; the split maps cover the whole map buffer.
    chk_stride = uint32_t((max_segment_evens / 3 + 1 + 63) / 64 + 1);
    chk_stride30 = uint32_t((max_segment_evens / 15 + 1 + 63) / 64 + 1);
    chk_words  = std::max<uint64_t>(std::max<uint64_t>((max_segment_evens + 63) / 64,
                                                   split_on ? 3ull * chk_stride : 0ull),
                                                   c30chk ? 15ull * chk_stride30 : 0ull);
    split_words = uint32_t((2 * map_words + 2) / 3 + 1);
    // Two-phase checker: evens left unresolved by phase A (~0.6% at 4e18
    // with 128 candidates, ~2% with 96). Sized at 1/8 of the segment; the
    // host fails closed if a segment ever overflows it.
    pending_cap = uint32_t(std::min<uint64_t>(
        std::max<uint64_t>(max_segment_evens / 8, 1u << 16), UINT32_MAX / 2));

    // Per-tile first-mark table (tile-major, u32): removes all 64-bit
    // reciprocal math from the tile kernel's prime walk. The kernel needs one
    // u32 load per prime instead of a 64-bit mul_hi chain. Kept under a host
    // memory budget; beyond it the kernel falls back to the legacy in-kernel
    // reciprocal path (huge segments only).
    table_bytes = uint64_t(own_end) * map_tiles * 4;
    use_table = own_end > mask_count &&
                           sieve_split > 0 &&
                           table_bytes * 2 <= uint64_t(1) << 30;
    use_table_u32 = use_table ? 1u : 0u;

    // Fused small-prime mask stage (make_prime_map): grouped mask tables,
    // rebuilt per segment (q_first-dependent); the grouping itself is fixed.
    // GB_MASK_GROUPS=0 selects the per-prime mask loop instead (A/B).
    const bool mask_groups_on = mask_count > 0 && env::onoff("GB_MASK_GROUPS", true);
    // Max group period in words. 4096 -> 8 groups, 75 KB of tables (primes
    // 3..61): measured sieve_s per 2e10 window 0.1086 vs 0.1161 at 16384
    // (7 groups, 120 KB) and 0.1143 at 8192 — the tables must stay
    // cache-resident. Must yield <= MAX_MASK_GROUPS (8) groups, else the
    // per-prime path is used. GB_MASK_PERIOD overrides.
    mask_period = env::u32("GB_MASK_PERIOD", 4096);   // = small_prime_group_masks default (debug_map validates it)
    SmallPrimeGroups mask_groups0;
    if (mask_groups_on)
        mask_groups0 = small_prime_group_masks(sieve_primes, mask_count, 3, mask_period);   // sizes only
    mask_groups_n = (mask_groups_on && mask_groups0.count() <= 8)
                                 ? mask_groups0.count() : 0u;
    mask_group_words = mask_groups0.words.size();
    d_group_desc = nullptr;
    if (mask_groups_n > 0) {
        d_group_desc = static_cast<uint32_t*>(ctx.alloc_shared(mask_groups0.desc.size() * 4));
        std::memcpy(d_group_desc, mask_groups0.desc.data(), mask_groups0.desc.size() * 4);
    }
}

RunStats DenseRun::run() {
    // ------------------------------------------------ Level Zero setup ----
    // (ctx, mod and the first four kernels: DenseRun's member initializers)
    configure_checkers();
    configure_ridges_and_stats();
    set_group_sizes();

    size_buffers();
    // count30's gap outputs as the kernel expects them before a segment
    const auto reset_gaps = [&](int s) {
        b.g_tail[s][0] = b.g_tail[s][1] = 0;
        b.x_cnt[s][11] = ~0u;                   // count30's first prime position (atomic_min)
        b.x_cnt[s][12] = 0;
        std::memset(b.g_hist[s], 0, GAPS30_HBINS * 4);
    };
    for (int s = 0; s < 2; ++s) {
        b.prime_map[s]  = static_cast<uint64_t*>(ctx.alloc_device(map_words * 8));
        b.checked[s]    = static_cast<uint64_t*>(ctx.alloc_device(chk_words * 8));
        // With the mod-30 checker (c30chk) nothing reads S1 / S5, but they are
        // still allocated: they keep v1.12.0's device-memory layout. The
        // large-prime kernels are sensitive to where the two slot maps land
        // (bucket_process 0.55 vs 0.63 ms per 425M segment at 4e18 with the
        // same code; packed kernel up to +5% in some layouts; not alignment or
        // page size — map_words-sized pads and a joint allocation moved it
        // between fast and slow irregularly), and v1.12.0's layout measures
        // fast on this driver. ~24 MB of VRAM.
        if (split_on && (!class_mode || c30)) {
            b.split_s1[s] = static_cast<uint32_t*>(ctx.alloc_device(size_t(split_words) * 4));
            b.split_s5[s] = static_cast<uint32_t*>(ctx.alloc_device(size_t(split_words) * 4));
        }
        // class mode: per-class tables back to back (x2)
        const size_t ncls = class_mode ? ncls_map : 1;
        b.first_bits[s] = static_cast<uint64_t*>(ctx.alloc_shared(ncls * sieve_split * 8));
        b.mask_table[s] = static_cast<uint64_t*>(ctx.alloc_shared(ncls * mask_count * 64 * 8));
        if (mask_groups_n > 0)
            b.group_masks[s] = static_cast<uint64_t*>(ctx.alloc_shared(ncls * mask_group_words * 8));
        b.group_y[s] = static_cast<uint32_t*>(ctx.alloc_shared(8uz * MAX_MASK_GROUPS_HOST * 4));
        if (class_mode && mask_groups_n > 0) {   // canonical group tables: segment-independent
            const SmallPrimeGroups cg = small_prime_group_masks_canonical(
                sieve_primes, mask_count, c30 ? 15u : 3u, mask_period);
            if (cg.words.size() != mask_group_words)
                throw std::runtime_error("class masks: canonical group layout mismatch");
            std::memcpy(b.group_masks[s], cg.words.data(), cg.words.size() * 8);
        }
        b.leftovers[s]  = static_cast<uint32_t*>(ctx.alloc_device(size_t(MAX_LEFTOVERS) * 4));
        // Shared allocation so the host can read the counter directly after a
        // sync — no D2H copy command per segment.
        b.leftover_n[s] = static_cast<uint32_t*>(ctx.alloc_shared(4));
        b.pending[s]   = static_cast<uint32_t*>(ctx.alloc_device(size_t(pending_cap) * 4));
        b.pending_n[s] = static_cast<uint32_t*>(ctx.alloc_shared(4));
        // Ridge append buffer: u32 count in word 0 (low half) + RIDGE_CAP
        // (n, p) pairs; shared so the host merges it without a D2H copy.
        b.ridge[s]     = static_cast<uint64_t*>(ctx.alloc_shared((1 + 2 * size_t(RIDGE_CAP)) * 8));
        b.p_hist[s]  = static_cast<uint32_t*>(ctx.alloc_shared((MAX_GB_PRIME / 2 + 1) * 4));
        b.p_first[s] = static_cast<uint32_t*>(ctx.alloc_shared((MAX_GB_PRIME / 2 + 1) * 4));
        std::memset(b.p_hist[s], 0, (MAX_GB_PRIME / 2 + 1) * 4);
        std::memset(b.p_first[s], 0xFF, (MAX_GB_PRIME / 2 + 1) * 4);
        if (stats_on) {
            b.x_hist[s]  = static_cast<uint32_t*>(ctx.alloc_shared(splan.xall.size() * 4));
            b.x_lim[s]   = static_cast<uint32_t*>(ctx.alloc_shared(32 * 4));
            b.x_cnt[s]   = static_cast<uint32_t*>(ctx.alloc_shared(16 * 4));
            std::memset(b.x_cnt[s], 0, 16 * 4);
            b.g_part[s]  = static_cast<uint64_t*>(ctx.alloc_shared(size_t(GAPS30_GROUPS_MAX) * 8));
            b.g_tail[s]  = static_cast<uint32_t*>(ctx.alloc_shared((2 + 2 * GAPS30_TCAP) * 4));
            b.g_hist[s]  = static_cast<uint32_t*>(ctx.alloc_shared(GAPS30_HBINS * 4));
            reset_gaps(s);
        }
        b.tile_first[s] = use_table
            ? static_cast<uint32_t*>(ctx.alloc_shared(size_t(table_bytes)))
            : nullptr;
    }
    b.gb_primes    = static_cast<uint16_t*>(ctx.alloc_device(
        gb_primes.size() * 2));
    b.twin_primes = static_cast<uint16_t*>(ctx.alloc_device(twin_primes.size() * 2));
    if (!rest_primes.empty())
        b.rest_primes = static_cast<uint16_t*>(ctx.alloc_device(rest_primes.size() * 2));
    if (split_on) {
        b.split_cands = static_cast<uint16_t*>(ctx.alloc_device(std::max<size_t>(split_plan.cands.size(), 1) * 2));
        b.split_meta  = static_cast<uint32_t*>(ctx.alloc_device(split_plan.meta.size() * 4));
        b.split_rest  = static_cast<uint16_t*>(ctx.alloc_device(std::max<size_t>(split_plan.rest.size(), 1) * 2));
    }
    if (stats_on) {
        b.x_all  = static_cast<uint16_t*>(ctx.alloc_device(splan.xall.size() * 2));
        b.x_tab  = static_cast<uint32_t*>(ctx.alloc_device(splan.xtab.size() * 4));
        b.x_ch   = static_cast<uint32_t*>(ctx.alloc_device(splan.xch.size() * 4));
        b.x_j    = static_cast<uint32_t*>(ctx.alloc_device(splan.xj.size() * 4));
    }
    b.want_p = static_cast<uint32_t*>(ctx.alloc_shared((MAX_GB_PRIME / 64 + 1) * 4));
    std::memset(b.want_p, 0, (MAX_GB_PRIME / 64 + 1) * 4);
    if (stats_on) std::memcpy(b.want_p, gstats->want_p().data(), gstats->want_p().size() * 4);
    const uint32_t tail_t = stats_on ? gstats->tail_t() : 0xFFFFFFFFu;   // check30_b tail recording
    bool sampled_slot[2] = {false, false}; // slot's segment is a full-histogram sample
    uint64_t slot_qfirst[2] = {0, 0};      // slot's segment position origin (gap positions)
    uint32_t slot_ggroups[2] = {0, 0};     // count30 work-groups launched for the slot's segment
    bool walked_slot[2] = {false, false};  // count30 walked the slot's gaps
    uint64_t sampled_evens = 0;            // evens of the sampled segments
    uint64_t slot_evens[2] = {0, 0};       // evens of the slot's segment
    uint64_t evens_collected = 0;          // live progress (goldbach_live.json)
    const auto t_live0 = clock_::now();
    // Merges a completed slot's statistics buffers (after its check event).
    const auto collect_stats = [&](uint32_t sl, uint64_t seg_start_sl) {
        if (sampled_slot[sl]) gstats->add_full(b.x_hist[sl]);
        else gstats->add_tail(b.p_hist[sl]);
        gstats->add_first(seg_start_sl, b.p_first[sl]);
        if (gaps_on) {
            uint64_t wmax = 0;                  // count30's work-group maxima -> the segment's
            const bool w = walked_slot[sl];
            if (w)
                for (uint32_t g = 0; g < slot_ggroups[sl]; ++g) wmax = std::max(wmax, b.g_part[sl][g]);
            gstats->add_gap_segment(slot_qfirst[sl], slot_evens[sl], b.x_cnt[sl], b.g_tail[sl],
                                    w ? &wmax : nullptr, w ? b.g_hist[sl] : nullptr);
        }
        gstats->add_counts(b.x_cnt[sl]);
        // ready for the slot's next segment (enqueued after this collect)
        std::memset(b.p_hist[sl], 0, (MAX_GB_PRIME / 2 + 1) * 4);
        std::memset(b.p_first[sl], 0xFF, (MAX_GB_PRIME / 2 + 1) * 4);
        std::memset(b.x_cnt[sl], 0, 16 * 4);
        reset_gaps(sl);                         // after x_cnt: sets its first-prime sentinel
        evens_collected += slot_evens[sl];
        gstats->write_live(start_even, limit, evens_collected, total_evens, seconds_since(t_live0));
        // monotone (bins only leave the want set): safe while the other
        // slot's kernels read it
        std::memcpy(b.want_p, gstats->want_p().data(), gstats->want_p().size() * 4);
    };
    if (c30chk) {
        b.c30_cands = static_cast<uint16_t*>(ctx.alloc_device(std::max<size_t>(plan30.cands.size(), 1) * 2));
        b.c30_meta  = static_cast<uint32_t*>(ctx.alloc_device(plan30.meta.size() * 4));
        b.c30_rest  = static_cast<uint16_t*>(ctx.alloc_device(plan30.rest.size() * 2));
        b.c30_rtab  = static_cast<uint32_t*>(ctx.alloc_device(plan30.rtab.size() * 4));
    }
    // The base-prime list is uploaded to DEVICE memory: the large kernel
    // streams it (~393 MB per segment) and device-local reads are fastest.
    // The payload comes straight from the mmap'd cache into one shared
    // staging buffer — the old fread pass is gone — and the DMA copy doubles
    // as the clock warm-up before the segment loop.
    b.sieve_primes = static_cast<uint32_t*>(ctx.alloc_device(size_t(sieve_count) * 4));
    // ---- Bucket sieve geometry (kernels/goldbach.cl, "Bucket sieve") ------
    init_mark("buffers");
    // Primes p > q_odds_full hit a segment at most once; they leave the
    // per-segment large kernel for the hit lists (default) or, with
    // GB_HITLIST=0, a ring of R buckets.
    // Requires the uniform dense geometry: every segment's q_first is
    // seg_start - MAX_GB_PRIME and consecutive q ranges advance by exactly
    // delta = segment_even / 2 bits. GB_BUCKET=0 disables (A/B).
    const uint64_t delta = segment_even / 2;
    const uint64_t q_odds_full = (segment_even - 5 + MAX_GB_PRIME) / 2 + 1;
    const SegmentGeometry g_first = segment_geometry(start_even, limit, segment_even, 0);
    const uint64_t q_first0 = g_first.q_first;
    const bool bucket_env_off = !env::onoff("GB_BUCKET", true);
    uint32_t bucket_begin = sieve_count;   // B: first bucketed prime index
    // Multiplier wheel of the bucket kernel (WHEEL_W = 30030, the only wheel
    // the kernel module has): max step (in p) and kept fraction of odd
    // multipliers.
    static_assert(WHEEL_W == 30030, "kernels/goldbach.cl only has the 30030 wheel tables");
    constexpr uint64_t wheel_max_step = 11;
    constexpr double wheel_keep = 5760.0 / 15015;
    uint32_t bucket_R = 0;
    uint32_t bucket_cap = 0;
    double bucket_lam = 0.0;   // expected bucketed hits per segment
    bool bucket_on = !bucket_env_off && large_count > 0 &&
                     start_even >= MAX_GB_PRIME + 2 &&
                     g_first.seg_end != limit &&                  // >= 2 segments
                     g_first.q_odds == q_odds_full &&
                     q_first0 == start_even - MAX_GB_PRIME;
    if (bucket_on) {
        // Bucket only primes above max(q_odds_full, GB_BUCKET_MINP): below
        // ~9 x q_odds_full a prime's per-segment packed setup (~22 ps) is
        // cheaper than filing its ~q_odds/p entries (~200 ps each).
        uint64_t minp = env::u64("GB_BUCKET_MINP", BUCKET_MIN_P_FACTOR * q_odds_full);
        minp = std::max<uint64_t>(minp, q_odds_full);   // correctness floor: <= 1 hit/segment
        bucket_begin = uint32_t(std::upper_bound(sieve_primes + sieve_split,
                                                 sieve_primes + sieve_count,
                                                 uint32_t(std::min<uint64_t>(minp, UINT32_MAX))) -
                                sieve_primes);
        const uint64_t pmax = sieve_primes[sieve_count - 1];
        // Farthest next hit: the wheel step is up to wheel_max_step * p.
        const uint64_t x = q_odds_full - 1 + wheel_max_step * pmax;
        const uint64_t max_rel = (x < q_odds_full) ? 0 : (x - q_odds_full) / delta + 1;
        bucket_R = uint32_t(max_rel + 2);
        // bucket_process does its filing arithmetic in 32 bits: needs
        // off + p < 2^32 for every entry (off < q_odds_full).
        if (bucket_begin >= sieve_count || bucket_R > 256 ||
            pmax + q_odds_full >= (uint64_t(1) << 32))
            bucket_on = false;
    }
    if (bucket_on) {
        // Expected hits per segment (= live entries per bucket), plus slack;
        // overflow is detected on the device and fails the run closed.
        // lambda = sum over bucketed p of q_odds_full / p. Mertens' second
        // theorem, sum_{p<=x} 1/p = ln ln x + M + O(1/ln^2 x), gives it in
        // closed form (error ~1e-4 relative here, far inside the slack);
        // the exact 48M-term loop cost ~30 ms of init. Overflow is detected
        // on the device anyway (fail closed).
        const double lo = double(sieve_primes[bucket_begin]);
        const double hi = double(sieve_primes[sieve_count - 1]);
        const double lambda = double(q_odds_full) *
            std::max(0.0, std::log(std::log(hi)) - std::log(std::log(lo))) + 1.0;
        // with the wheel only multipliers coprime to 30 are hits: 8/15
        const double lam = lambda * wheel_keep;
        bucket_lam = lam;
        bucket_cap = uint32_t(lam * 1.10 + 8.0 * std::sqrt(lam) + 65536.0);
        // Round the bucket stride to 64 entries (512 B): bucket_process reads
        // each bucket as SIMD16 x 8 B = 128 B loads, and a stride that is not a
        // multiple of 128 B left half the buckets straddling two lines on
        // every load (bucket_process 0.633 -> 0.549 ms/segment when aligned;
        // it also made the threshold sweeps jump by +-2% with cap % 16).
        bucket_cap = (bucket_cap + 63u) & ~63u;
    }
    // Hit lists (kernels/goldbach.cl hl_harvest / hl_apply) instead of the
    // bucket ring for the same primes [bucket_begin, sieve_count): every
    // hl_K segments one harvest files all hits of the next hl_K segments as
    // u32 offsets; each segment applies its list. A list holds the same
    // hits as a bucket. The harvest walks in
    // (segment, offset) form: needs delta + p_max < 2^32 (the bucket's own
    // pmax + q_odds_full < 2^32 guard) and q_odds_full < 2 delta.
    constexpr uint32_t HL_MAX_K_HOST = 128;  // kernels/goldbach.cl HL_MAX_K (SLM counters)
    static_assert(WG_SIZE >= HL_MAX_K_HOST, "hl_harvest: one thread per SLM batch counter");
    // GB_HITLIST=0 restores the bucket ring (A/B); GB_HITLIST_K sets the batch.
    const bool hitlist = bucket_on && env::onoff("GB_HITLIST", true);
    uint32_t hl_K = env::u32("GB_HITLIST_K", 128, 1u, HL_MAX_K_HOST);
    // A segment's list is one region of hl_capg entries per harvest
    // work-group (hl_G groups, each harvesting a fixed share of the primes:
    // mean m = lam / hl_G hits per region). Capacity m * 1.10 + 10 sqrt(m) +
    // 64 (a region's count is a sum of independent hit indicators, variance
    // <= mean), rounded to a power of two; overflow fails the run closed.
    uint32_t hl_G = 1, hl_capg = 64;
    {
        const auto need = [](double m) { return m * 1.10 + 10.0 * std::sqrt(m) + 64.0; };
        const auto mmax = [&](double cg) {   // largest mean whose capacity fits cg
            double lo = 0.0, hi = cg;
            for (int it = 0; it < 60; ++it) {
                const double m = 0.5 * (lo + hi);
                (need(m) <= cg ? lo : hi) = m;
            }
            return lo;
        };
        while (std::ceil(bucket_lam / mmax(hl_capg)) > 4096.0) hl_capg *= 2;
        hl_G = uint32_t(std::max(1.0, std::ceil(bucket_lam / mmax(hl_capg))));
    }
    // one list allocation below 4 GiB (32-bit buffer offsets in the kernels)
    while (hl_K > 1 && uint64_t(hl_K) * hl_G * hl_capg * 4 >= (uint64_t(1) << 32)) --hl_K;
    // Segments of the run (the loop below: seg_start += segment_even until
    // seg_end == limit) and the last one's q_odds (the only short one).
    uint64_t run_segments = 0, run_last_q_odds = 0;
    // hl_harvest's 32-bit walk: off + p (off < delta), off + delta and the
    // segment index all stay below 2^32. 2 delta < 2^32 already follows from
    // bucket_on (a bucketed p > q_odds_full > delta with pmax + q_odds_full
    // < 2^32); it is checked here so the kernel does not depend on that.
    if (hitlist && !(q_odds_full < 2 * delta && 2 * delta < (uint64_t(1) << 32) &&
                     delta + sieve_primes[sieve_count - 1] < (uint64_t(1) << 32)))
        throw std::runtime_error("hit lists: geometry outside the harvest's 32-bit bounds");
    if (hitlist) {
        for (uint64_t s0 = start_even; s0 <= limit; s0 += segment_even) {
            const SegmentGeometry gs = segment_geometry(s0, limit, segment_even, run_segments);
            ++run_segments;
            if (gs.seg_end == limit) { run_last_q_odds = gs.q_odds; break; }
            if (gs.q_odds != q_odds_full || gs.q_first != q_first0 + 2 * (run_segments - 1) * delta)
                throw std::runtime_error("hit lists: non-uniform segment geometry");
        }
    }
    const uint32_t packed_end = bucket_on ? bucket_begin : sieve_count;
    init_mark("bucket-geom+lambda");
    const uint32_t packed_count = packed_end - sieve_split;
    // hl_apply fused into the packed kernel (sieve_large_packed_hl): the two
    // mark sets commute, and alone hl_apply was latency-bound. hl_apply runs
    // as its own launch only when no prime is packed.
    const bool hl_fuse = hitlist && packed_count > 0;
    PackedLarge packed;
    uint32_t* d_large_bases = nullptr;
    uint32_t* d_large_gaps = nullptr;
    if (packed_count > 0) {
        packed = pack_large_primes(sieve_primes, sieve_split, packed_end);
        init_mark("  pack (host)");
        d_large_bases = static_cast<uint32_t*>(ctx.alloc_device(packed.bases.size() * 4));
        d_large_gaps  = static_cast<uint32_t*>(ctx.alloc_device(packed.gaps.size()));
    }
    init_mark("  packed alloc");
    uint64_t* d_bkt_entries = nullptr;
    uint64_t* d_bkt_entries1 = nullptr;
    uint32_t bkt_split = 0;
    uint32_t* d_bkt_counts = nullptr;
    uint32_t* bkt_err = nullptr;
    uint32_t* d_hl_lists = nullptr;
    uint32_t* d_hl_counts = nullptr;
    if (hitlist) {
        d_hl_lists  = static_cast<uint32_t*>(ctx.alloc_device(size_t(hl_K) * hl_G * hl_capg * 4));
        d_hl_counts = static_cast<uint32_t*>(ctx.alloc_device(size_t(HL_MAX_K_HOST) * hl_G * 4));
    }
    if (bucket_on && !hitlist) {
        // Two allocations (buckets [0, split) and [split, R)): a single one
        // above ~5 GiB cost ~35-45 ms of host time at its first submission
        // on this driver (kernels/goldbach.cl, bkt_bucket).
        bkt_split = (bucket_R + 1) / 2;
        d_bkt_entries  = static_cast<uint64_t*>(ctx.alloc_device(size_t(bkt_split) * bucket_cap * 8));
        d_bkt_entries1 = static_cast<uint64_t*>(ctx.alloc_device(
            size_t(std::max<uint32_t>(bucket_R - bkt_split, 1)) * bucket_cap * 8));
        init_mark("  bucket entries alloc");
    }
    if (bucket_on) {
        d_bkt_counts  = static_cast<uint32_t*>(ctx.alloc_device(256 * 4));
        bkt_err       = static_cast<uint32_t*>(ctx.alloc_shared(4));
        *bkt_err = 0;
        const uint32_t zero = 0;
        ZE_CHECK(zeCommandListAppendMemoryFill(ctx.list(), d_bkt_counts, &zero, 4, 256 * 4,
                                               nullptr, 0, nullptr));
    }
    init_mark("  counts/err alloc+fill");
    // Bucketed primes whose first hit is p*p (p*p >= q_first0) activate only
    // once that hit is within the ring horizon; host-side pointer.
    const auto bkt_first_seg = [&](uint64_t G) -> uint64_t {
        return G < q_odds_full ? 0 : (G - q_odds_full) / delta + 1;
    };
    uint32_t act_next = bucket_begin;   // primes [bucket_begin, act_next) are active
    const uint32_t act_p2_begin = bucket_on
        ? uint32_t(std::lower_bound(sieve_primes + bucket_begin, sieve_primes + sieve_count,
                                    q_first0, [](uint32_t p, uint64_t q) {
                                        return uint64_t(p) * p < q; }) - sieve_primes)
        : sieve_count;
    std::unique_ptr<Kernel> k_bproc, k_bact, k_cmp, k_large_full, k_hlh, k_hla;
    // Class map: 2048 groups (per 1e11: -0.2% at 4e18, -0.4% at 5e18); the
    // dense map measured 2048 no better than 1024.
    const uint32_t bucket_groups = bucket_groups_env ? bucket_groups_env
                                                     : class_mode ? 2 * BUCKET_GROUPS : BUCKET_GROUPS;
    if (hitlist) {
        k_hlh = std::make_unique<Kernel>(mod, "hl_harvest");
        k_hla = std::make_unique<Kernel>(mod, hl_fuse ? "sieve_large_packed_hl" : "hl_apply");
        k_hlh->set_group_size(WG_SIZE);
        k_hla->set_group_size(hl_fuse ? large_wg : WG_SIZE);
    } else if (bucket_on) {
        k_bproc = std::make_unique<Kernel>(mod, "bucket_process");
        k_bact  = std::make_unique<Kernel>(mod, "bucket_activate");
        k_bproc->set_group_size(WG_SIZE);
        k_bact->set_group_size(WG_SIZE);
    }
    init_mark("  p2 search + kernel create");
    // GB_BUCKET_VERIFY=1: rebuild every segment's map WITHOUT buckets (tile +
    // packed kernel over ALL large primes) into a scratch map and compare it
    // word for word with the production map on the GPU. Validation mode.
    // GB_BUCKET_VERIFY=N checks every Nth segment (1 = all; e.g. 100 costs
    // ~1% and is a cheap production safety net: an end-to-end PASS cannot
    // reveal a lost bucket mark — other candidates still resolve the evens).
    const uint32_t bucket_verify_every = env::present("GB_BUCKET_VERIFY")
        ? env::u32("GB_BUCKET_VERIFY", 0u, 1u) : 0u;
    const bool bucket_verify = bucket_on && bucket_verify_every > 0;
    PackedLarge packed_full;
    uint32_t* d_full_bases = nullptr;
    uint32_t* d_full_gaps = nullptr;
    uint64_t* d_verify_map = nullptr;
    uint32_t* verify_diff = nullptr;
    if (bucket_verify) {
        packed_full = pack_large_primes(sieve_primes, sieve_split, sieve_count);
        d_full_bases = static_cast<uint32_t*>(ctx.alloc_device(packed_full.bases.size() * 4));
        d_full_gaps  = static_cast<uint32_t*>(ctx.alloc_device(packed_full.gaps.size()));
        verify_diff  = static_cast<uint32_t*>(ctx.alloc_shared(4));
        *verify_diff = 0;
        d_verify_map = static_cast<uint64_t*>(ctx.alloc_device(map_words * 8));
        k_cmp = std::make_unique<Kernel>(mod, "map_compare");
        k_large_full = std::make_unique<Kernel>(mod, "sieve_large_packed");
        k_cmp->set_group_size(WG_SIZE);
        k_large_full->set_group_size(large_wg);
    }
    // Reciprocal tables are consumed only by the tile kernel, whose loop ends
    // at sieve_split. At 4e18 the complete prime list is ~98M entries, while
    // the default split is only 78K; sizing these to the full list wasted
    // gigabytes and over a billion setup operations per isolated run.
    b.magic        = static_cast<uint64_t*>(ctx.alloc_device(size_t(sieve_split) * 8));
    b.shifts       = static_cast<uint32_t*>(ctx.alloc_device(size_t(sieve_split) * 4));
    b.ms32         = static_cast<uint64_t*>(ctx.alloc_device(size_t(sieve_split) * 8));

    // Precompute the per-prime magic reciprocal (ceil(2^64 / p)) and shift so
    // the kernel can do division-free jumps to each thread's owned range.
    // magic/shift are segment-independent; first_bits is per-segment.
    init_mark("pack+bucket-alloc(before magic)");
    const auto [magic, shift] = build_magic_reciprocals(sieve_primes, sieve_split);

    // 32-bit reciprocals for the tile table path. The kernel jumps from the
    // tile's first mark to a thread's 2048-bit range: d = my_bit0 - tf is
    // always < BITS_PER_TILE = 2^17, so floor(d/p) is computable with a 32-bit
    // reciprocal: with s = bit_width(BITS_PER_TILE * p) we have 2^s > BITS_PER_TILE*p,
    // which makes floor(d*m/2^s) = floor(d/p) exact for every d < BITS_PER_TILE
    // (for d divisible by p the product is k*p*(2^s+e)/2^s = k + k*e/2^s with
    // k*e < BITS_PER_TILE < 2^s; otherwise d*e < (p - r)*2^s follows from
    // d*e < BITS_PER_TILE*p < 2^s). m = ceil(2^s/p) <= 2*BITS_PER_TILE fits u32.
    // Primes >= BITS_PER_TILE never use it (single mark per tile, range check).
    std::vector<uint64_t> ms32v(sieve_split, 0ull);
    if (use_table) {
        for (size_t i = 0; i < sieve_split; ++i) {
            const uint64_t p = sieve_primes[i];
            if (p < BITS_PER_TILE) {
                uint32_t s = 0;
                while ((uint64_t(1) << s) <= BITS_PER_TILE * p) ++s;
                const uint64_t m = ((uint64_t(1) << s) + p - 1) / p;
                ms32v[i] = m | (uint64_t(s) << 32);
            }
        }
    }

    // Upload the read-only tables once; never touched again per segment.
    // The DMA copies also double as the clock warm-up before the segment loop.
    init_mark("magic+ms32");
    const auto t_upload = clock_::now();
    {
        void* stage1 = stage_upload(ctx,
            gb_primes.data(), gb_primes.size() * 2, b.gb_primes);
        void* stage_tw = stage_upload(ctx, twin_primes.data(), twin_primes.size() * 2, b.twin_primes);
        void* stage_rest = b.rest_primes ? stage_upload(ctx, rest_primes.data(),
                                                        rest_primes.size() * 2, b.rest_primes) : nullptr;
        void* stage_sc = (split_on && !split_plan.cands.empty())
            ? stage_upload(ctx, split_plan.cands.data(), split_plan.cands.size() * 2, b.split_cands) : nullptr;
        void* stage_sm = split_on
            ? stage_upload(ctx, split_plan.meta.data(), split_plan.meta.size() * 4, b.split_meta) : nullptr;
        void* stage_sr = (split_on && !split_plan.rest.empty())
            ? stage_upload(ctx, split_plan.rest.data(), split_plan.rest.size() * 2, b.split_rest) : nullptr;
        void* stage_c30[4] = {};
        if (c30chk) {
            if (!plan30.cands.empty())
                stage_c30[0] = stage_upload(ctx, plan30.cands.data(), plan30.cands.size() * 2, b.c30_cands);
            stage_c30[1] = stage_upload(ctx, plan30.meta.data(), plan30.meta.size() * 4, b.c30_meta);
            stage_c30[2] = stage_upload(ctx, plan30.rest.data(), plan30.rest.size() * 2, b.c30_rest);
            stage_c30[3] = stage_upload(ctx, plan30.rtab.data(), plan30.rtab.size() * 4, b.c30_rtab);
        }
        void* stage_x[4] = {};
        if (stats_on) {
            stage_x[3] = stage_upload(ctx, splan.xj.data(), splan.xj.size() * 4, b.x_j);
            stage_x[2] = stage_upload(ctx, splan.xch.data(), splan.xch.size() * 4, b.x_ch);
            stage_x[0] = stage_upload(ctx, splan.xall.data(), splan.xall.size() * 2, b.x_all);
            stage_x[1] = stage_upload(ctx, splan.xtab.data(), splan.xtab.size() * 4, b.x_tab);
        }
        // The base-prime list on the device is read only by the tile kernel
        // ([0, split)), bucket_activate ([bucket_begin, count)) and the
        // verify kernels (everything). Copy only the ranges that are read —
        // at 4e18 that is ~192 MB instead of 437 MB — and DMA them straight
        // from the mmap'd cache (host pages stay valid until the sync below),
        // skipping the staging memcpy.
        const auto copy_primes = [&](uint32_t lo, uint32_t hi) {
            if (hi <= lo) return;
            ZE_CHECK(zeCommandListAppendMemoryCopy(ctx.list(), b.sieve_primes + lo,
                                                   sieve_primes + lo, size_t(hi - lo) * 4,
                                                   nullptr, 0, nullptr));
        };
        if (bucket_verify) {
            copy_primes(0, sieve_count);
        } else {
            copy_primes(0, sieve_split);
            if (bucket_on) copy_primes(bucket_begin, sieve_count);
        }
        void* stage6 = d_large_bases ? stage_upload(ctx, packed.bases.data(),
                                                    packed.bases.size() * 4, d_large_bases) : nullptr;
        void* stage7 = d_large_gaps ? stage_upload(ctx, packed.gaps.data(),
                                                   packed.gaps.size(), d_large_gaps) : nullptr;
        void* stage8 = d_full_bases ? stage_upload(ctx, packed_full.bases.data(),
                                                   packed_full.bases.size() * 4, d_full_bases) : nullptr;
        void* stage9 = d_full_gaps ? stage_upload(ctx, packed_full.gaps.data(),
                                                  packed_full.gaps.size(), d_full_gaps) : nullptr;
        void* stage3 = stage_upload(ctx, magic.data(), size_t(sieve_split) * 8, b.magic);
        void* stage4 = stage_upload(ctx, shift.data(), size_t(sieve_split) * 4, b.shifts);
        void* stage5 = stage_upload(ctx, ms32v.data(), size_t(sieve_split) * 8, b.ms32);
        ctx.host_sync();
        Context::free_mem(ctx.context(), stage1);
        Context::free_mem(ctx.context(), stage_tw);
        if (stage_rest) Context::free_mem(ctx.context(), stage_rest);
        if (stage_sc) Context::free_mem(ctx.context(), stage_sc);
        if (stage_sm) Context::free_mem(ctx.context(), stage_sm);
        if (stage_sr) Context::free_mem(ctx.context(), stage_sr);
        for (void* p : stage_c30) if (p) Context::free_mem(ctx.context(), p);
        for (void* p : stage_x) if (p) Context::free_mem(ctx.context(), p);
        Context::free_mem(ctx.context(), stage3);
        Context::free_mem(ctx.context(), stage4);
        Context::free_mem(ctx.context(), stage5);
        if (stage6) Context::free_mem(ctx.context(), stage6);
        if (stage7) Context::free_mem(ctx.context(), stage7);
        if (stage8) Context::free_mem(ctx.context(), stage8);
        if (stage9) Context::free_mem(ctx.context(), stage9);
    }

    // Statistics buffers: one shared allocation per slot instead of seven.
    // collect_stats reads and resets them on the host once per segment; as
    // separate shared allocations each one paid its own migration to the host
    // and its own migration back at the next kernel append (a fixed cost per
    // allocation, not per byte). Re-pointed into one block they migrate once
    // each way per segment.
    //
    // Why this is safe: the host touches a slot's block only between that
    // slot's collect (after its check event) and the first append that
    // references it. count30's launch is that first append, and x_lim — the
    // one buffer here the host writes during enqueue — is written before it.
    // Nothing on the host reads or writes the block again until the slot's
    // next collect, so a queued kernel never sees the block leave the device.
    //
    // The seven original allocations stay, unused: the large-prime kernels
    // are sensitive to where the device allocations land (see the split_s1 /
    // split_s5 note above), so the block is added after every other buffer
    // and nothing that exists today moves.
    void* stats_block[2] = {nullptr, nullptr};
    std::vector<void*> stats_unpacked;   // the original allocations (freed at the end)
    if (stats_on) {
        const size_t ph_bytes = (size_t(MAX_GB_PRIME) / 2 + 1) * 4;
        size_t block_bytes = 0;
        const auto take = [&block_bytes](size_t bytes) {   // 64-byte aligned sub-buffer offset
            const size_t at = block_bytes;
            block_bytes += (bytes + 63) & ~63uz;
            return at;
        };
        const size_t at_p_hist  = take(ph_bytes);
        const size_t at_p_first = take(ph_bytes);
        const size_t at_x_lim   = take(32 * 4);
        const size_t at_x_cnt   = take(16 * 4);
        const size_t at_g_part  = take(size_t(GAPS30_GROUPS_MAX) * 8);
        const size_t at_g_tail  = take((2 + 2 * size_t(GAPS30_TCAP)) * 4);
        const size_t at_g_hist  = take(size_t(GAPS30_HBINS) * 4);
        for (int s = 0; s < 2; ++s) {
            unsigned char* base = static_cast<unsigned char*>(ctx.alloc_shared(block_bytes));
            stats_block[s] = base;
            std::memset(base, 0, block_bytes);
            for (void* p : {static_cast<void*>(b.p_hist[s]), static_cast<void*>(b.p_first[s]),
                            static_cast<void*>(b.x_lim[s]), static_cast<void*>(b.x_cnt[s]),
                            static_cast<void*>(b.g_part[s]), static_cast<void*>(b.g_tail[s]),
                            static_cast<void*>(b.g_hist[s])})
                stats_unpacked.push_back(p);
            b.p_hist[s]  = static_cast<uint32_t*>(static_cast<void*>(base + at_p_hist));
            b.p_first[s] = static_cast<uint32_t*>(static_cast<void*>(base + at_p_first));
            b.x_lim[s]   = static_cast<uint32_t*>(static_cast<void*>(base + at_x_lim));
            b.x_cnt[s]   = static_cast<uint32_t*>(static_cast<void*>(base + at_x_cnt));
            b.g_part[s]  = static_cast<uint64_t*>(static_cast<void*>(base + at_g_part));
            b.g_tail[s]  = static_cast<uint32_t*>(static_cast<void*>(base + at_g_tail));
            b.g_hist[s]  = static_cast<uint32_t*>(static_cast<void*>(base + at_g_hist));
            // the state the kernels expect before a slot's first segment (as
            // the original buffers had it): p_hist / x_cnt zero, p_first all
            // ones, count30's gap outputs reset
            std::memset(b.p_first[s], 0xFF, ph_bytes);
            reset_gaps(s);
        }
    }

    const uint32_t gb_count = uint32_t(gb_primes.size());
    // The mask-table kernel arg is set per segment (the table CONTENT is
    // q_first-dependent and rewritten into the slot's shared buffer each
    // iteration, alongside first_bits).

    // Events for per-kernel GPU timing and GPU-side dependencies.
    // Per pipeline slot: sieve signals after the tile-sieve launch, large
    // after the large-prime launch (the check kernel waits on BOTH GPU-side),
    // check signals after the check launch (the host waits on it once per
    // segment). Pools are per-event so each can be reset independently.
    init_mark("upload");
    SlotEvents ev;
    ev.create(ctx.context());

    stats.init_seconds = seconds_since(t_start);
    stats.init_cache_s  = cache_seconds;
    stats.init_module_s = module_seconds;
    stats.init_upload_s = seconds_since(t_upload);

    // ------------------------------------------------------ Segment loop --
    const auto t_gpu = clock_::now();
    uint64_t total_leftovers = 0;
    double sieve_seconds = 0.0, large_seconds = 0.0;
    double verify_seconds = 0.0, offset_seconds = 0.0, table_seconds = 0.0;
    uint64_t prev_seg_start = 0;  // first even n of the previously enqueued segment
    uint64_t prev_seg_evens = 0;  // its even count (leftover index bound)
    // Carried from the last ENQUEUED segment to the iteration that collects
    // it (one slot behind): host wall time inside that iteration and its
    // host offset-generation share. Stored here because collection happens
    // one pipeline slot after enqueue.
    double prev_seg_wall = 0.0, prev_seg_offsets = 0.0;

    // Device timer resolution (1.2 stype; base struct returns garbage).
    ze_device_properties_t dev_props{};
    dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
    zeDeviceGetProperties(ctx.device(), &dev_props);
    const double timer_hz = double(dev_props.timerResolution);

    // Per-segment record for the segment currently being enqueued, plus the
    // completed record of the segment enqueued one iteration earlier (its
    // GPU timestamps become readable only when the NEXT iteration collects
    // it, by which time seg_stat has been reused for the newer segment).
    SegmentStats seg_stat, prev_stat;
    uint64_t sum_tiles = 0, sum_large_items = 0;
    if (collect_segments) stats.per_segment.reserve(64);

    // Env-gated host-phase trace (GB_PHASE=1): per-segment wall time split
    // into prep (first-bits / table / masks), enqueue (args + resets +
    // launches), wait (ev_check host sync) and collect (timestamp queries +
    // leftover read). Printed to stderr; overhead is a few clock reads per
    // segment and the gate defaults to off.
    const bool phase_trace = env::present("GB_PHASE");
    bool hl_used[2] = {false, false};   // slot's segment launched a harvest (timing)
    std::vector<uint64_t> fb_host(class_mode ? size_t(ncls_map) * sieve_split : 0);   // per-segment first bits
    double sum_prep = 0.0, sum_enq = 0.0, sum_wait = 0.0, sum_coll = 0.0;
    double sum_args = 0.0, sum_launch = 0.0;
    bool split_used[2] = {false, false};   // slot's segment ran the split checker
    bool xs_used[2] = {false, false};      // slot's segment ran check30_xs8..3 (4 launches)

    // Query the per-kernel GPU timestamps of the given slot: the sieve, large
    // and check kernels SET ds / dl / dv, the extra kernels of a phase are
    // ADDED to them. Returns true when one of the three yielded a valid
    // timestamp. A kernel is skipped when the device clock reports
    // end <= start (kernel_seconds: anomalies seen on some driver builds).
    // With GB_PHASE=1, also log raw start/end ticks per segment so kernel
    // overlap (or its absence) can be verified directly.
    const auto query_slot = [&](uint32_t slot, double& ds, double& dl, double& dv) {
        bool any = false;
        if (const auto t = kernel_seconds(ev.sieve[slot], timer_hz)) { ds = *t; any = true; }
        if (const auto t = kernel_seconds(ev.large[slot], timer_hz)) { dl = *t; any = true; }
        if (const auto t = kernel_seconds(ev.check[slot], timer_hz)) { dv = *t; any = true; }
        if (hl_used[slot])   // large_s also counts the hit-list harvest
            if (const auto t = kernel_seconds(ev.hl[slot], timer_hz)) {
                dl += *t;
                if (phase_trace) std::fprintf(stderr, "    hl_harvest %.3f ms\n", *t * 1e3);
            }
        if (bucket_on && !hl_fuse)   // large_s = packed kernel + bucket_process
            if (const auto t = kernel_seconds(ev.bucket[slot], timer_hz)) {
                dl += *t;
                if (phase_trace)
                    std::fprintf(stderr, "    bucket_process %.3f ms (large total %.3f ms)\n",
                                 *t * 1e3, dl * 1e3);
            }
        // check_s = phase A + phase B (+ split_map)
        if (const auto t = kernel_seconds(ev.checka[slot], timer_hz)) {
            dv += *t;
            if (phase_trace) std::fprintf(stderr, "    phase_a %.3f ms\n", *t * 1e3);
        }
        for (int v = 0; v < 3 && xs_used[slot]; ++v)
            if (const auto t = kernel_seconds(ev.xv[v][slot], timer_hz)) {
                dv += *t;
                if (phase_trace) std::fprintf(stderr, "    phase_a variant %d %.3f ms\n", v, *t * 1e3);
            }
        if (split_used[slot])
            if (const auto t = kernel_seconds(ev.splitk[slot], timer_hz)) {
                dv += *t;
                if (phase_trace) std::fprintf(stderr, "    split_map %.3f ms\n", *t * 1e3);
            }
        if (phase_trace)
            if (const auto s = kernel_ticks(ev.sieve[slot]), l = kernel_ticks(ev.large[slot]),
                           c = kernel_ticks(ev.check[slot]);
                s && l && c)
                std::fprintf(stderr,
                    "    ticks: sieve[%lld..%lld] large[%lld..%lld] check[%lld..%lld] "
                    "gap_s->l %+.1f ms gap_l->c %+.1f ms\n",
                    (long long)s->kernelStart, (long long)s->kernelEnd,
                    (long long)l->kernelStart, (long long)l->kernelEnd,
                    (long long)c->kernelStart, (long long)c->kernelEnd,
                    (double(int64_t(l->kernelStart) - int64_t(s->kernelEnd)) / timer_hz) * 1e3,
                    (double(int64_t(c->kernelStart) - int64_t(l->kernelEnd)) / timer_hz) * 1e3);
        return any;
    };

    for (uint64_t seg_start = start_even; seg_start <= limit; seg_start += segment_even) {
        const SegmentGeometry g = segment_geometry(seg_start, limit, segment_even,
                                                   stats.segments);
        const uint64_t seg_end   = g.seg_end;
        const uint64_t seg_evens = g.seg_evens;
        const uint64_t q_first   = g.q_first;
        const uint64_t q_last    = g.q_last;
        const uint64_t q_odds    = g.q_odds;
        const uint64_t tiles     = g.tiles;
        const uint32_t slot      = g.slot;
        // Class mode: the map is two class arrays of cgeo.tpc tiles each.
        const ClassGeo cgeo = class_mode ? class_geometry(q_first, q_odds, ncls_map) : ClassGeo{};
        const uint64_t tiles_launch = class_mode ? uint64_t(cgeo.K) * cgeo.tpc : tiles;
        const uint32_t c5w = cgeo.tpc * TILE_WORDS;                 // class stride (u64 words)
        const uint32_t off5 = c5w * 64u;                            // class stride (bits)
        uint32_t phis = cgeo.phi[0] | (cgeo.phi[1] << 8);
        // phi1 / phi5 kernel args: K = 2 the two phases; K = 8 the dense
        // phase (mod 15) -> class nibble table (15 = not stored)
        uint32_t ga1 = cgeo.phi[0], ga2 = cgeo.phi[1];
        if (c30) {
            phis = 0;
            uint64_t ph2k = ~0ull;
            for (uint32_t c = 0; c < 8; ++c) {
                phis |= cgeo.phi[c] << (4 * c);
                ph2k &= ~(0xFull << (4 * cgeo.phi[c]));
                ph2k |= uint64_t(c) << (4 * cgeo.phi[c]);
            }
            ga1 = uint32_t(ph2k);
            ga2 = uint32_t(ph2k >> 32);
        }
        const uint32_t mask_coff = class_mode ? 0u : uint32_t(mask_group_words);   // class: one canonical set

        // Per-segment telemetry: start the wall clock. The record itself is
        // filled after mask_count_seg/tiles are computed below, and finalized
        // when this segment's GPU work is collected (next iteration or drain).
        const auto t_seg_wall = clock_::now();
        const double offset_before = offset_seconds;

        // The small-prime mask table is q_first-dependent; the kernel's stage
        // 3.5 (restore q = p) is range-sensitive and skipped when q_first >
        // MASK_MAX_P, so masks stay valid at every range (disabling them at
        // high ranges forced Xe2 into thousands of scalar SLM RMWs per lane
        // for p=3..61 instead of 18 x 32 whole-word ANDs).
        const uint32_t mask_count_seg = mask_count;

        // Per-segment telemetry record (collected after the NEXT iteration's
        // sync, once this segment's kernel timestamps are readable).
        seg_stat = SegmentStats{};
        seg_stat.seg_start   = seg_start;
        seg_stat.seg_end     = seg_end;
        seg_stat.evens       = seg_evens;
        seg_stat.sieve_split = sieve_split;
        seg_stat.mask_count  = mask_count_seg;
        seg_stat.tiles       = tiles_launch;
        // work-items of the packed large-prime kernel as launched below
        // (LARGE_PACK_PPI primes each; the bucketed / hit-listed primes are
        // not work-items of it). The legacy LARGE_PRIMES_PER_ITEM count over
        // every large prime reported ~4x the real launch.
        seg_stat.large_items = (uint64_t(packed_count) + LARGE_PACK_PPI - 1) / LARGE_PACK_PPI;
        sum_tiles += tiles_launch;
        sum_large_items += seg_stat.large_items;

        // Host prep for this segment (first-bits, tile table, masks).
        //
        // PIPELINE NOTE: this host work runs while the GPU is still executing
        // the PREVIOUS segment (other slot). It is safe to write the slot's
        // shared buffers here because that slot was last read by the segment
        // two iterations ago, whose check kernel completed before the host
        // collected it (host-synced on ev.check of that slot).
        const auto t_prep = clock_::now();
        const double table_before = table_seconds;
        {
            // first_bits is a shared (host-mapped) allocation: write it
            // directly, no copy command or sync needed. The kernel launch is
            // queued after this host write completes.
            // Class mode: the first bits are computed into a persistent host
            // buffer (a fresh 2.3 MB vector per segment paid its page faults
            // every time) by parallel workers over disjoint prime ranges, each
            // copying its own slices: 1.65 -> ~0.3 ms of per-segment host
            // prep, which had become the pipeline's limit once the GPU
            // segment time dropped below the host loop's.
            if (class_mode) {
                parallel_for(0, sieve_split, 6, [&](uint32_t j0, uint32_t j1) {
                    first_bits_class_range(fb_host.data(), sieve_primes, sieve_split, j0, j1,
                                           q_first, q_last, cgeo);
                    for (uint32_t c = 0; c < cgeo.K; ++c)
                        std::memcpy(b.first_bits[slot] + size_t(c) * sieve_split + j0,
                                    fb_host.data() + size_t(c) * sieve_split + j0, size_t(j1 - j0) * 8);
                });
            } else {
                fb_host = compute_first_bits(sieve_primes, sieve_split, q_first, q_last);
                std::memcpy(b.first_bits[slot], fb_host.data(), fb_host.size() * 8);
            }
            const std::vector<uint64_t>& fb = fb_host;

            if (use_table) {
                const auto t_table = clock_::now();
                build_tile_first_table(b.tile_first[slot], uint32_t(tiles_launch),
                                       sieve_split, own_end, fb.data(), magic.data(),
                                       shift.data(), sieve_primes, class_mode ? cgeo.tpc : 0u);
                table_seconds += seconds_since(t_table);
            }

            // Large-prime first bits are computed INLINE by
            // sieve_large_primes (one u64 division per prime — the former
            // host/98M-prime pass and then a separate GPU pre-pass were both
            // measured and removed; see tools/debug_fb.cpp for the bit-exact
            // validation of that formula).

            // Small-prime mask table for THIS segment (q_first-dependent).
            // Same shared-buffer pipeline-safety argument as first_bits:
            // this slot was last read two segments ago, already synced.
            if (class_mode) {
                for (uint32_t c = 0; c < cgeo.K; ++c) {
                    if (mask_groups_n > 0) {
                        // canonical tables (uploaded once): per class only the
                        // word offset of each group (primes.cpp)
                        const std::vector<uint32_t> ys = small_prime_group_offsets(
                            sieve_primes, mask_count_seg, cgeo.Q[c], cgeo.qmod, mask_period);
                        if (ys.size() != mask_groups_n)
                            throw std::runtime_error("class masks: group layout mismatch");
                        std::memcpy(b.group_y[slot] + c * MAX_MASK_GROUPS_HOST, ys.data(), ys.size() * 4);
                        continue;
                    }
                    const auto mt = small_prime_mask_table_step(sieve_primes, mask_count_seg, cgeo.Q[c], cgeo.qmod);
                    if (mask_count_seg > 0) {
                        std::memcpy(b.mask_table[slot] + size_t(c) * mask_count_seg * 64, mt.data(),
                                    mt.size() * 8);
                    }
                }
            } else if (mask_groups_n > 0) {
                const SmallPrimeGroups gm =
                    small_prime_group_masks(sieve_primes, mask_count_seg, q_first, mask_period);
                std::memcpy(b.group_masks[slot], gm.words.data(), gm.words.size() * 8);
            } else if (mask_count_seg > 0) {
                const auto mt = small_prime_mask_table(sieve_primes, mask_count_seg, q_first);
                std::memcpy(b.mask_table[slot], mt.data(), mt.size() * 8);
            }
        }
        const double prep_s = seconds_since(t_prep);
        const auto t_enq = clock_::now();

        // --- make_prime_map args (single launch covers ALL tiles)
        // Kernel arg order: 0=map 1=primes 2=fb_global 3=magic 4=shifts
        //                   5=count 6=q_odds 7=SLM 8=mask_count 9=mask_table
        //                   10=q_first 11=sieve_split
        const auto t_args0 = clock_::now();
        k_sieve.set_arg(0, sizeof(b.prime_map[slot]),    &b.prime_map[slot]);
        k_sieve.set_arg(1, sizeof(b.sieve_primes), &b.sieve_primes);
        k_sieve.set_arg(2, sizeof(b.first_bits[slot]),   &b.first_bits[slot]);
        k_sieve.set_arg(3, sizeof(b.magic),        &b.magic);
        k_sieve.set_arg(4, sizeof(b.shifts),       &b.shifts);
        k_sieve.set_arg(5, sizeof(sieve_split),    &sieve_split);
        k_sieve.set_arg(6, sizeof(q_odds),         &q_odds);
        k_sieve.set_local_arg(7, size_t(TILE_WORDS) * 8); // 16 KB SLM
        k_sieve.set_arg(8, sizeof(mask_count_seg), &mask_count_seg);
        k_sieve.set_arg(9, sizeof(b.mask_table[slot]), &b.mask_table[slot]);
        k_sieve.set_arg(10, sizeof(q_first),       &q_first);
        k_sieve.set_arg(11, sizeof(sieve_split),   &sieve_split);
        k_sieve.set_arg(12, sizeof(b.tile_first[slot]), &b.tile_first[slot]);
        k_sieve.set_arg(13, sizeof(use_table_u32), &use_table_u32);
        k_sieve.set_arg(14, sizeof(b.ms32),        &b.ms32);
        k_sieve.set_arg(15, sizeof(own_end),       &own_end);
        k_sieve.set_arg(16, sizeof(b.group_masks[slot]), &b.group_masks[slot]);
        k_sieve.set_arg(17, sizeof(d_group_desc),  &d_group_desc);
        k_sieve.set_arg(18, sizeof(mask_groups_n), &mask_groups_n);
        k_sieve.set_arg(19, 4, &cm_u32);
        k_sieve.set_arg(20, 4, &cgeo.tpc);
        k_sieve.set_arg(21, 4, &phis);
        k_sieve.set_arg(22, 8, &cgeo.n[0]);
        k_sieve.set_arg(23, 8, &cgeo.n[1]);
        k_sieve.set_arg(24, 4, &mask_coff);
        {
            uint32_t* gy = (class_mode && mask_groups_n > 0) ? b.group_y[slot] : nullptr;
            k_sieve.set_arg(26, sizeof(gy), &gy);
        }
        k_sieve.set_arg(25, 4, &sieve_split);

        // Reset per-segment state BEFORE enqueueing the kernels that signal
        // these events. zeEventHostReset after its signaling launch has been
        // enqueued is a race: if the GPU completes the kernel before the host
        // reset executes, the signal is lost and the later WaitOnEvents
        // blocks forever (this caused size-dependent hangs). All events are
        // quiescent here: their previous signals belonged to segment k-2,
        // whose check kernel was host-synchronized during the last iteration.
        const uint32_t zero = 0;
        ZE_CHECK(zeCommandListAppendMemoryFill(ctx.list(), b.leftover_n[slot], &zero, 1, 4,
                                               nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListAppendMemoryFill(ctx.list(), b.pending_n[slot], &zero, 1, 4,
                                               nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListAppendMemoryFill(ctx.list(), b.ridge[slot], &zero, 1, 4,
                                               nullptr, 0, nullptr));
        // statistics: p_hist / p_first / x_cnt are reset by the host after
        // each collect (collect_stats); x_hist is used by sampled segments only
        if (stats_on && stats.segments % stats_sample == 0)
            ZE_CHECK(zeCommandListAppendMemoryFill(ctx.list(), b.x_hist[slot], &zero, 4,
                                                   splan.xall.size() * 4, nullptr, 0, nullptr));
        ZE_CHECK(zeEventHostReset(ev.sieve[slot]));
        ZE_CHECK(zeEventHostReset(ev.large[slot]));
        ZE_CHECK(zeEventHostReset(ev.check[slot]));
        ZE_CHECK(zeEventHostReset(ev.checka[slot]));
        ZE_CHECK(zeEventHostReset(ev.bucket[slot]));
        ZE_CHECK(zeEventHostReset(ev.hl[slot]));
        ZE_CHECK(zeEventHostReset(ev.splitk[slot]));
        for (int v = 0; v < 3; ++v) ZE_CHECK(zeEventHostReset(ev.xv[v][slot]));
        const double args_s = seconds_since(t_args0);

        const auto t_launch0 = clock_::now();
        // launch_overlap: may overlap segment k-1's checker (this slot's map,
        // tables and counters were last used by segment k-2, host-synchronized)
        k_sieve.launch_overlap(ctx.list(), uint32_t(tiles_launch), ev.sieve[slot]); // all tiles, one launch
        const double l1_s = seconds_since(t_launch0);

        // --- sieve_large_primes: one work-item per large base prime.
        // The explicit event wait is LOAD-BEARING even though the list is
        // created in-order: the large kernel's atomic ANDs must never overlap
        // the tile kernel's whole-word map stores (a store landing after an
        // atomic clear would restore a composite as "prime" and silently
        // corrupt the verification; observed as nondeterministic map
        // corruption when only implicit ordering was relied on).
        // Bucket sieve FIRST, then the packed kernel (v1.10.0, +1.9%): the
        // bucket drain streams ~70 MB of entries through L3; run before the
        // packed kernel, that eviction no longer lands between the packed
        // kernel's ~23M atomics and the map. Both only AND-clear map bits
        // (monotone), so their relative order does not matter for the result;
        // the barrier after bucket_activate orders the tile kernel's stores
        // before bucket_process, and the packed block keeps its own barrier.
        uint32_t hl_jb = 0;   // this segment's list (hit lists)
        if (hitlist) {
            // Hit lists for primes [bucket_begin, sieve_count): at the first
            // segment of each batch of hl_K, harvest the whole batch; then
            // mark this segment's list. The barrier before hl_apply orders
            // the harvest's list stores and the tile kernel's map stores.
            const uint32_t k_idx = uint32_t(stats.segments);
            if (k_idx >= run_segments)
                throw std::runtime_error("hit lists: segment index beyond the run");
            const uint32_t jb = k_idx % hl_K;
            hl_used[slot] = false;
            if (jb == 0) {
                const uint32_t nseg = uint32_t(std::min<uint64_t>(hl_K, run_segments - k_idx));
                const uint32_t qtail = uint32_t((k_idx + nseg == run_segments) ? run_last_q_odds
                                                                              : q_odds_full);
                const uint32_t delta32 = uint32_t(delta), qof32 = uint32_t(q_odds_full);
                const uint32_t a = bucket_begin, bnd = sieve_count;
                const float inv_delta = 1.0f / float(delta);
                // unordered list: the harvest may overlap this segment's tile
                // sieve and segment k-1's checker (it touches only the lists,
                // their counts and the error flag; the barrier before
                // segment k-1's checker ordered it after the last apply)
                if (!g_unordered) ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
                k_hlh->set_arg(0, sizeof(b.sieve_primes), &b.sieve_primes);
                k_hlh->set_arg(1, sizeof(a), &a);
                k_hlh->set_arg(2, sizeof(bnd), &bnd);
                k_hlh->set_arg(3, sizeof(q_first), &q_first);
                k_hlh->set_arg(4, sizeof(delta32), &delta32);
                k_hlh->set_arg(5, sizeof(qof32), &qof32);
                k_hlh->set_arg(6, sizeof(nseg), &nseg);
                k_hlh->set_arg(7, sizeof(qtail), &qtail);
                k_hlh->set_arg(8, sizeof(inv_delta), &inv_delta);
                k_hlh->set_arg(9, sizeof(d_hl_lists), &d_hl_lists);
                k_hlh->set_arg(10, sizeof(hl_capg), &hl_capg);
                k_hlh->set_arg(11, sizeof(d_hl_counts), &d_hl_counts);
                k_hlh->set_arg(12, sizeof(bkt_err), &bkt_err);
                k_hlh->launch_overlap(ctx.list(), hl_G, ev.hl[slot]);   // exactly hl_G: the region layout
                hl_used[slot] = true;
            }
            hl_jb = jb;
            if (!hl_fuse) {
            ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
            const uint32_t q_odds32 = uint32_t(q_odds);
            k_hla->set_arg(0, sizeof(b.prime_map[slot]), &b.prime_map[slot]);
            k_hla->set_arg(1, sizeof(d_hl_lists), &d_hl_lists);
            k_hla->set_arg(2, sizeof(hl_capg), &hl_capg);
            k_hla->set_arg(3, sizeof(d_hl_counts), &d_hl_counts);
            k_hla->set_arg(4, sizeof(jb), &jb);
            k_hla->set_arg(5, sizeof(hl_G), &hl_G);
            k_hla->set_arg(6, sizeof(q_odds32), &q_odds32);
            k_hla->set_arg(7, sizeof(bkt_err), &bkt_err);
            k_hla->set_arg(8, 4, &cm_u32);
            k_hla->set_arg(9, 4, &ga1);
            k_hla->set_arg(10, 4, &ga2);
            k_hla->set_arg(11, 4, &off5);
            k_hla->launch(ctx.list(), bucket_groups, ev.bucket[slot]);
            }
        } else if (bucket_on) {
            // Bucket sieve for primes [bucket_begin, sieve_count): activate
            // the primes whose first hit entered the ring horizon, then drain
            // this segment's bucket (marks + re-filing), then reset its count.
            const uint32_t k_idx = uint32_t(stats.segments);
            const uint64_t horizon = uint64_t(k_idx) + bucket_R - 1;
            uint32_t act_to = act_next;
            if (k_idx == 0) act_to = std::max(act_to, act_p2_begin);   // p*p < q_first0: all at once
            while (act_to < sieve_count) {
                const uint64_t p = sieve_primes[act_to];
                if (bkt_first_seg((p * p - q_first0) >> 1) > horizon) break;
                ++act_to;
            }
            if (act_to > act_next) {
                const uint32_t a = act_next, bnd = act_to;
                k_bact->set_arg(0, sizeof(b.sieve_primes), &b.sieve_primes);
                k_bact->set_arg(1, sizeof(a), &a);
                k_bact->set_arg(2, sizeof(bnd), &bnd);
                k_bact->set_arg(3, sizeof(d_bkt_entries), &d_bkt_entries);
                k_bact->set_arg(4, sizeof(d_bkt_counts), &d_bkt_counts);
                k_bact->set_arg(5, sizeof(bucket_cap), &bucket_cap);
                k_bact->set_arg(6, sizeof(bucket_R), &bucket_R);
                k_bact->set_arg(7, sizeof(k_idx), &k_idx);
                k_bact->set_arg(8, sizeof(q_first0), &q_first0);
                k_bact->set_arg(9, sizeof(q_odds_full), &q_odds_full);
                k_bact->set_arg(10, sizeof(delta), &delta);
                k_bact->set_arg(11, sizeof(bkt_err), &bkt_err);
                k_bact->set_arg(12, sizeof(d_bkt_entries1), &d_bkt_entries1);
                k_bact->set_arg(13, sizeof(bkt_split), &bkt_split);
                const uint64_t chunk = uint64_t(WG_SIZE) * 8;   // BKT_E
                k_bact->launch(ctx.list(), uint32_t(std::min<uint64_t>(
                    (uint64_t(bnd - a) + chunk - 1) / chunk, 4096)));
                act_next = act_to;
            }
            ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
            const uint32_t bslot = k_idx % bucket_R;
            const uint32_t is_last = (seg_end == limit) ? 1u : 0u;
            const float inv_delta = 1.0f / float(delta);
            k_bproc->set_arg(0, sizeof(b.prime_map[slot]), &b.prime_map[slot]);
            k_bproc->set_arg(1, sizeof(d_bkt_entries), &d_bkt_entries);
            k_bproc->set_arg(2, sizeof(d_bkt_counts), &d_bkt_counts);
            k_bproc->set_arg(3, sizeof(bucket_cap), &bucket_cap);
            k_bproc->set_arg(4, sizeof(bucket_R), &bucket_R);
            k_bproc->set_arg(5, sizeof(bslot), &bslot);
            k_bproc->set_arg(6, sizeof(q_odds), &q_odds);
            k_bproc->set_arg(7, sizeof(q_odds_full), &q_odds_full);
            k_bproc->set_arg(8, sizeof(delta), &delta);
            k_bproc->set_arg(9, sizeof(inv_delta), &inv_delta);
            k_bproc->set_arg(10, sizeof(is_last), &is_last);
            k_bproc->set_arg(11, sizeof(bkt_err), &bkt_err);
            const uint32_t qfw = uint32_t(q_first % WHEEL_W);
            k_bproc->set_arg(12, sizeof(qfw), &qfw);
            k_bproc->set_arg(13, sizeof(d_bkt_entries1), &d_bkt_entries1);
            k_bproc->set_arg(14, sizeof(bkt_split), &bkt_split);
            k_bproc->set_arg(15, 4, &cm_u32);
            k_bproc->set_arg(16, 4, &ga1);
            k_bproc->set_arg(17, 4, &ga2);
            k_bproc->set_arg(18, 4, &off5);
            k_bproc->launch(ctx.list(), bucket_groups, ev.bucket[slot]);
            // The drained bucket is refilled only from segment k+1's
            // activation on (horizon k+R); the next segment's barriers order
            // this reset before any such push. The reset is a write after
            // bucket_process's read of the same counter; the in-order list
            // serializes them, so no barrier is needed there. The unordered
            // list does not: there the barrier is required.
            if (g_unordered)
                ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
            const uint32_t zero_count = 0;
            ZE_CHECK(zeCommandListAppendMemoryFill(ctx.list(), d_bkt_counts + bslot, &zero_count, 4, 4,
                                                   nullptr, 0, nullptr));
        }
        if (large_count > 0) {
            // BARRIER, not just an event wait: the large kernel's atomic ANDs
            // must see the tile kernel's whole-word stores. On this driver an
            // event wait alone does not make those stores visible, which
            // manifested as nondeterministic lost composite marks
            // (tools/debug_map.cpp: the map vs Miller-Rabin diff varied run
            // to run until the barrier was added).
            ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
            Kernel& kl = hl_fuse ? *k_hla : k_large;
            kl.set_arg(0, sizeof(b.prime_map[slot]), &b.prime_map[slot]);
            kl.set_arg(1, sizeof(d_large_bases),  &d_large_bases);
            kl.set_arg(2, sizeof(d_large_gaps),   &d_large_gaps);
            kl.set_arg(3, sizeof(packed_count),   &packed_count);
            kl.set_arg(4, sizeof(q_odds),         &q_odds);
            kl.set_arg(5, sizeof(q_first),        &q_first);
            kl.set_arg(6, sizeof(q_last),         &q_last);
            kl.set_arg(7, 4, &cm_u32);
            kl.set_arg(8, 4, &ga1);
            kl.set_arg(9, 4, &ga2);
            kl.set_arg(10, 4, &off5);
            if (hl_fuse) {
                kl.set_arg(11, sizeof(d_hl_lists), &d_hl_lists);
                kl.set_arg(12, sizeof(hl_capg), &hl_capg);
                kl.set_arg(13, sizeof(d_hl_counts), &d_hl_counts);
                kl.set_arg(14, sizeof(hl_jb), &hl_jb);
                kl.set_arg(15, sizeof(hl_G), &hl_G);
                kl.set_arg(16, sizeof(bkt_err), &bkt_err);
            }
            const uint64_t large_items = (uint64_t(packed_count) + LARGE_PACK_PPI - 1) / LARGE_PACK_PPI;
            if (large_items > 0)
                kl.launch(ctx.list(), uint32_t((large_items + large_wg - 1) / large_wg),
                               ev.large[slot]);
        }
        if (bucket_verify && stats.segments % bucket_verify_every == 0) {
            // Rebuild this segment's map without buckets into the scratch map
            // and count differing words (checked at collection).
            ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
            k_sieve.set_arg(0, sizeof(d_verify_map), &d_verify_map);
            k_sieve.launch(ctx.list(), uint32_t(tiles_launch));
            ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
            k_large_full->set_arg(0, sizeof(d_verify_map), &d_verify_map);
            k_large_full->set_arg(1, sizeof(d_full_bases), &d_full_bases);
            k_large_full->set_arg(2, sizeof(d_full_gaps), &d_full_gaps);
            k_large_full->set_arg(3, sizeof(large_count), &large_count);
            k_large_full->set_arg(4, sizeof(q_odds), &q_odds);
            k_large_full->set_arg(5, sizeof(q_first), &q_first);
            k_large_full->set_arg(6, sizeof(q_last), &q_last);
            k_large_full->set_arg(7, 4, &cm_u32);
            k_large_full->set_arg(8, 4, &ga1);
            k_large_full->set_arg(9, 4, &ga2);
            k_large_full->set_arg(10, 4, &off5);
            const uint64_t items_full = (uint64_t(large_count) + LARGE_PACK_PPI - 1) / LARGE_PACK_PPI;
            k_large_full->launch(ctx.list(), uint32_t((items_full + large_wg - 1) / large_wg));
            ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
            const uint64_t nwords = class_mode ? uint64_t(cgeo.K) * c5w : (q_odds + 63) / 64;
            k_cmp->set_arg(0, sizeof(b.prime_map[slot]), &b.prime_map[slot]);
            k_cmp->set_arg(1, sizeof(d_verify_map), &d_verify_map);
            k_cmp->set_arg(2, sizeof(nwords), &nwords);
            k_cmp->set_arg(3, sizeof(verify_diff), &verify_diff);
            k_cmp->launch(ctx.list(), 512);
            if (env::present("GB_BUCKET_VERIFY_DUMP")) {
                // Diagnostics: on the first differing segment, show which map
                // is wrong by checking the differing bits with Miller-Rabin.
                ctx.host_sync();
                if (*verify_diff) {
                    std::vector<uint64_t> a(nwords), bref(nwords);
                    ZE_CHECK(zeCommandListAppendMemoryCopy(ctx.list(), a.data(), b.prime_map[slot], nwords * 8, nullptr, 0, nullptr));
                    ZE_CHECK(zeCommandListAppendMemoryCopy(ctx.list(), bref.data(), d_verify_map, nwords * 8, nullptr, 0, nullptr));
                    ctx.host_sync();
                    int shown = 0;
                    for (uint64_t w = 0; w < nwords && shown < 12; ++w) {
                        uint64_t x = a[w] ^ bref[w];
                        while (x && shown < 12) {
                            const int bb = __builtin_ctzll(x); x &= x - 1;
                            const uint64_t bit = w * 64 + bb;
                            const uint64_t q = !class_mode ? q_first + 2 * bit
                                             : cgeo.Q[w / c5w] + cgeo.qmod * (bit - uint64_t(w / c5w) * c5w * 64);
                            std::fprintf(stderr, "seg %llu bit %llu q=%llu bucket_map=%d reference=%d prime(MR)=%d\n",
                                         (unsigned long long)stats.segments, (unsigned long long)bit,
                                         (unsigned long long)q, int((a[w] >> bb) & 1), int((bref[w] >> bb) & 1),
                                         int(is_prime_u64(q)));
                            ++shown;
                        }
                    }
                    throw std::runtime_error("GB_BUCKET_VERIFY_DUMP: mismatch dumped");
                }
            }
        }
        const double l2_s = seconds_since(t_launch0);

        // --- check_numbers args
        // Twin checker: phase A reads the twin map with the twin list.
        const uint32_t twin_count = uint32_t(twin_primes.size());
        k_check.set_arg(0, sizeof(b.prime_map[slot]),  &b.prime_map[slot]);
        k_check.set_arg(1, sizeof(b.checked[slot]),    &b.checked[slot]);
        k_check.set_arg(2, sizeof(b.twin_primes),  &b.twin_primes);
        k_check.set_arg(3, sizeof(twin_count),     &twin_count);
        k_check.set_arg(4, sizeof(b.pending[slot]),   &b.pending[slot]);
        k_check.set_arg(5, sizeof(b.pending_n[slot]), &b.pending_n[slot]);
        k_check.set_arg(6, sizeof(q_first),        &q_first);
        k_check.set_arg(7, sizeof(q_odds),         &q_odds);
        k_check.set_arg(8, sizeof(seg_start),      &seg_start);
        k_check.set_arg(9, sizeof(seg_evens),      &seg_evens);
        k_check.set_arg(10, sizeof(twin_cap),    &twin_cap);
        k_check.set_arg(11, sizeof(pending_cap), &pending_cap);
        const uint32_t twin_u32 = 1u;
        k_check.set_arg(12, sizeof(twin_u32),    &twin_u32);

        // The check kernel must SEE the completed map from BOTH sieve
        // kernels (memory visibility, not just execution ordering — see the
        // barrier note above). Same for the leftover_n fill at the top of
        // the segment: the check kernel's atomic increments must not race
        // it. Each check work-item covers 4 x 64 = 256 evens.
        ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
        const uint64_t evens_per_a_group = uint64_t(WG_SIZE) * a_w64 * 64;
        const uint32_t a_groups =
            uint32_t((seg_evens + evens_per_a_group - 1) / evens_per_a_group);
        const SplitSeg sseg = split_on ? split_segment(seg_start, seg_evens, q_first, split_plan)
                                       : SplitSeg{};
        const Seg30 s30 = c30chk ? seg30(seg_start, seg_evens, q_first) : Seg30{};
        // Recording threshold: the current Kth largest ridge (ratchets up as
        // the top list fills; ~0u disables every append).
        const uint32_t seg_ceiling = s30.ok ? ceil_c30 : sseg.ok ? ceil_split : ceil_fallback;
        const uint32_t ridge_thresh_u32 =
            ridges->active() ? std::max(ridges->thresh(), seg_ceiling) : 0xFFFFFFFFu;
        split_used[slot] = sseg.ok && (!class_mode || c30) && !s30.ok;  // split_map / merge30 timing
        if (class_mode && (!sseg.ok || (!c30 && (sseg.phi1 != cgeo.phi[0] || sseg.phi5 != cgeo.phi[1]))))
            throw std::runtime_error("class map: split checker geometry rejected");
        if (stats_on && !s30.ok)
            throw std::runtime_error("statistics: segment geometry outside the mod-30 checker");
        // Every stats_sample-th segment of the run (segment 0 always) gets the
        // full exact p(n) histogram; the others record the tail in check30_b.
        const bool sampled = stats_on && stats.segments % stats_sample == 0;
        sampled_slot[slot] = sampled;
        slot_evens[slot] = seg_evens;
        if (sampled) sampled_evens += seg_evens;
        xs_used[slot] = sampled;
        if (stats_on) {
            // count30 (prime / twin counts; fusing it into check30_a measured
            // no faster: the larger kernel slows phase A) is timed under the
            // splitk event (added to check_s).
            const uint32_t cw32 = 2 * c5w;
            const std::vector<uint32_t> lim = count30_limits(seg_start, seg_end, cgeo.Q);
            std::memcpy(b.x_lim[slot], lim.data(), lim.size() * 4);
            // class order by phase (gap kernels): ord / phis in ascending phi,
            // phic by class; the smallest counted position
            uint32_t g_ord = 0, g_phis = 0, g_phic = 0, g_poslo = ~0u;
            {
                uint32_t cls[8] = {0, 1, 2, 3, 4, 5, 6, 7};
                std::sort(cls, cls + 8, [&](uint32_t x, uint32_t y) { return cgeo.phi[x] < cgeo.phi[y]; });
                for (uint32_t k = 0; k < 8; ++k) {
                    g_ord |= cls[k] << (4 * k);
                    g_phis |= cgeo.phi[cls[k]] << (4 * k);
                    g_phic |= cgeo.phi[k] << (4 * k);
                    if (lim[2 * cls[k]] < lim[2 * cls[k] + 1])
                        g_poslo = std::min(g_poslo, 30 * lim[2 * cls[k]] + 2 * cgeo.phi[cls[k]]);
                }
            }
            // count30 also finds every gap >= GAPS30_TAIL (bit-parallel) and the
            // segment's first / last prime when the gap statistics are on, and
            // on the walked segments walks every prime (the gap histogram) on a
            // fixed grid of work-groups (one histogram flush each)
            const uint32_t ctcap = gaps_on ? GAPS30_TCAP : 0u;
            walked_slot[slot] = gaps_on && (sampled || gaps_exact);
            const uint32_t cwalk = walked_slot[slot] ? 1u : 0u;
            if (cw32 % COUNT30_CW) throw std::runtime_error("count30: class stride not a multiple of its lane width");
            const uint32_t cgroups = (cw32 / COUNT30_CW + WG_SIZE - 1) / WG_SIZE;
            slot_ggroups[slot] = cwalk ? std::min(gaps_groups, cgroups) : cgroups;
            slot_qfirst[slot] = q_first;
            k_cnt->set_arg(0, sizeof(b.prime_map[slot]), &b.prime_map[slot]);
            k_cnt->set_arg(1, 4, &cw32);
            k_cnt->set_arg(2, sizeof(b.x_lim[slot]), &b.x_lim[slot]);
            k_cnt->set_arg(3, sizeof(b.x_cnt[slot]), &b.x_cnt[slot]);
            k_cnt->set_arg(4, 4, &g_ord);
            k_cnt->set_arg(5, 4, &g_phis);
            k_cnt->set_arg(6, 4, &g_phic);
            k_cnt->set_arg(7, 4, &g_poslo);
            k_cnt->set_arg(8, sizeof(b.g_tail[slot]), &b.g_tail[slot]);
            k_cnt->set_arg(9, 4, &ctcap);
            k_cnt->set_arg(10, 4, &cwalk);
            k_cnt->set_arg(11, sizeof(b.g_part[slot]), &b.g_part[slot]);
            k_cnt->set_arg(12, sizeof(b.g_hist[slot]), &b.g_hist[slot]);
            k_cnt->launch(ctx.list(), slot_ggroups[slot], ev.splitk[slot]);
            split_used[slot] = true;
        }
        if (sampled) {
            // Full histogram: check30_xs8/6/4/3 (ascending prefix) -> barrier
            // -> check30_xb (ascending continuation).
            const uint32_t cw32 = 2 * c5w;
            const uint32_t se32 = uint32_t(seg_evens);
            int first_xs = 0;
            while (first_xs < 4 && !splan.vn[first_xs]) ++first_xs;
            for (int v = 0; v < 4; ++v) {      // one launch per array-count variant
                if (!splan.vn[v]) continue;
                Kernel& kv = *k_cxs[v];
                const uint32_t per_item = XS_W_HOST[v] * 32;
                const uint32_t items_pad = ((s30.ec_max + per_item - 1) / per_item + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
                kv.set_arg(0, sizeof(b.prime_map[slot]), &b.prime_map[slot]);
                kv.set_arg(1, 4, &cw32);
                kv.set_arg(2, sizeof(b.x_all), &b.x_all);
                kv.set_arg(3, sizeof(b.x_tab), &b.x_tab);
                kv.set_arg(4, sizeof(b.x_j), &b.x_j);
                kv.set_arg(5, sizeof(b.x_ch), &b.x_ch);
                kv.set_arg(6, sizeof(b.pending[slot]), &b.pending[slot]);
                kv.set_arg(7, sizeof(b.pending_n[slot]), &b.pending_n[slot]);
                kv.set_arg(8, 4, &pending_cap);
                kv.set_arg(9, 4, &items_pad);
                kv.set_arg(10, 4, &se32);
                kv.set_arg(11, 4, &s30.sm30);
                kv.set_arg(12, 4, &s30.tb);
                kv.set_arg(13, 4, &s30.phik);
                kv.set_arg(14, sizeof(b.x_hist[slot]), &b.x_hist[slot]);
                kv.set_arg(15, 8, &splan.vcls[v]);
                // the first variant may overlap count30 (see phase A below)
                const ze_event_handle_t xe = v == 3 ? ev.checka[slot] : ev.xv[v][slot];
                if (v == first_xs) kv.launch_overlap(ctx.list(), splan.vn[v] * items_pad / WG_SIZE, xe);
                else kv.launch(ctx.list(), splan.vn[v] * items_pad / WG_SIZE, xe);
            }
            ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
            k_cxb->set_arg(0, sizeof(b.prime_map[slot]), &b.prime_map[slot]);
            k_cxb->set_arg(1, sizeof(b.x_all), &b.x_all);
            k_cxb->set_arg(2, sizeof(b.x_tab), &b.x_tab);
            k_cxb->set_arg(3, sizeof(b.pending[slot]), &b.pending[slot]);
            k_cxb->set_arg(4, sizeof(b.pending_n[slot]), &b.pending_n[slot]);
            k_cxb->set_arg(5, sizeof(b.leftovers[slot]), &b.leftovers[slot]);
            k_cxb->set_arg(6, sizeof(b.leftover_n[slot]), &b.leftover_n[slot]);
            k_cxb->set_arg(7, sizeof(q_first), &q_first);
            k_cxb->set_arg(8, sizeof(q_odds), &q_odds);
            k_cxb->set_arg(9, sizeof(seg_start), &seg_start);
            k_cxb->set_arg(10, 4, &pending_cap);
            k_cxb->set_arg(11, 4, &ga1);
            k_cxb->set_arg(12, 4, &ga2);
            k_cxb->set_arg(13, 4, &off5);
            k_cxb->set_arg(14, sizeof(b.ridge[slot]), &b.ridge[slot]);
            k_cxb->set_arg(15, 4, &ridge_thresh_u32);
            k_cxb->set_arg(16, sizeof(b.x_hist[slot]), &b.x_hist[slot]);
            k_cxb->set_arg(17, sizeof(b.p_first[slot]), &b.p_first[slot]);
            k_cxb->set_arg(18, sizeof(b.want_p), &b.want_p);
            k_cxb->launch(ctx.list(), stats_b_groups, ev.check[slot]);  // grid-stride
        } else if (s30.ok) {
            // Mod-30 checker: phase A on the class arrays -> barrier -> phase B.
            const uint32_t cw32 = 2 * c5w;
            const uint32_t per_item = a_w64 * 64;
            const uint32_t items_pad = ((s30.ec_max + per_item - 1) / per_item + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
            const uint32_t se32 = uint32_t(seg_evens);
            k_ca->set_arg(0, sizeof(b.prime_map[slot]), &b.prime_map[slot]);
            k_ca->set_arg(1, 4, &cw32);
            k_ca->set_arg(2, sizeof(b.checked[slot]), &b.checked[slot]);
            k_ca->set_arg(3, 4, &chk_stride30);
            k_ca->set_arg(4, sizeof(b.c30_cands), &b.c30_cands);
            k_ca->set_arg(5, sizeof(b.c30_meta), &b.c30_meta);
            k_ca->set_arg(6, sizeof(b.pending[slot]), &b.pending[slot]);
            k_ca->set_arg(7, sizeof(b.pending_n[slot]), &b.pending_n[slot]);
            k_ca->set_arg(8, 4, &pending_cap);
            k_ca->set_arg(9, 4, &items_pad);
            k_ca->set_arg(10, 4, &se32);
            k_ca->set_arg(11, 4, &s30.sm30);
            k_ca->set_arg(12, 4, &s30.tb);
            k_ca->set_arg(13, 4, &s30.phik);
            k_ca->set_arg(14, 4, &split_write_checked);
            // launch_overlap: phase A may run alongside count30 (both only
            // read the finished map; the barrier above follows the marks)
            k_ca->launch_overlap(ctx.list(), 15 * items_pad / WG_SIZE, ev.checka[slot]);
            ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
            k_cb->set_arg(0, sizeof(b.prime_map[slot]), &b.prime_map[slot]);
            k_cb->set_arg(1, sizeof(b.checked[slot]), &b.checked[slot]);
            k_cb->set_arg(2, 4, &chk_stride30);
            k_cb->set_arg(3, sizeof(b.c30_rest), &b.c30_rest);
            k_cb->set_arg(4, sizeof(b.c30_rtab), &b.c30_rtab);
            k_cb->set_arg(5, sizeof(b.pending[slot]), &b.pending[slot]);
            k_cb->set_arg(6, sizeof(b.pending_n[slot]), &b.pending_n[slot]);
            k_cb->set_arg(7, sizeof(b.leftovers[slot]), &b.leftovers[slot]);
            k_cb->set_arg(8, sizeof(b.leftover_n[slot]), &b.leftover_n[slot]);
            k_cb->set_arg(9, sizeof(q_first), &q_first);
            k_cb->set_arg(10, sizeof(q_odds), &q_odds);
            k_cb->set_arg(11, sizeof(seg_start), &seg_start);
            k_cb->set_arg(12, 4, &pending_cap);
            k_cb->set_arg(13, 4, &split_write_checked);
            k_cb->set_arg(14, 4, &ga1);
            k_cb->set_arg(15, 4, &ga2);
            k_cb->set_arg(16, 4, &off5);
            k_cb->set_arg(17, sizeof(b.ridge[slot]), &b.ridge[slot]);
            k_cb->set_arg(18, 4, &ridge_thresh_u32);
            k_cb->set_arg(19, sizeof(b.p_hist[slot]), &b.p_hist[slot]);
            k_cb->set_arg(20, sizeof(b.p_first[slot]), &b.p_first[slot]);
            k_cb->set_arg(21, sizeof(b.want_p), &b.want_p);
            k_cb->set_arg(22, 4, &tail_t);
            k_cb->launch(ctx.list(), phase_b_groups, ev.check[slot]);  // grid-stride
        } else if (sseg.ok) {
            // Class-split checker: [split_map -> barrier ->] phase A -> barrier
            // -> phase B (same pending / leftover buffers as the twin path).
            // Class mode: the map IS S1 / S5 (no split_map pass).
            const uint32_t map32_words = uint32_t(2 * ((q_odds + 63) / 64));
            uint32_t s_words = std::min<uint32_t>(split_words, (map32_words + 2) / 3 + 1);
            uint32_t* s1p = b.split_s1[slot];
            uint32_t* s5p = b.split_s5[slot];
            if (c30) {
                // merge30: S1 / S5 from the mod-30 class arrays. j -> class
                // of residue (Q6 + 6j) mod 30, Q6 = first q of the mod-6 class
                uint32_t kmap[2] = {0, 0};
                for (int c = 0; c < 2; ++c) {
                    const uint64_t Q6 = q_first + 2 * uint64_t(c ? sseg.phi5 : sseg.phi1);
                    for (uint32_t j = 0; j < 5; ++j) {
                        const uint32_t r = uint32_t((Q6 + 6 * j) % 30);
                        uint32_t k = 15;
                        for (uint32_t cc = 0; cc < 8; ++cc) if (CLASS_RES30[cc] == r) k = cc;
                        kmap[c] |= k << (4 * j);
                    }
                }
                const uint32_t cw32 = 2 * c5w;
                k_merge->set_arg(0, sizeof(b.prime_map[slot]), &b.prime_map[slot]);
                k_merge->set_arg(1, 4, &cw32);
                k_merge->set_arg(2, sizeof(b.split_s1[slot]), &b.split_s1[slot]);
                k_merge->set_arg(3, sizeof(b.split_s5[slot]), &b.split_s5[slot]);
                k_merge->set_arg(4, 4, &s_words);
                k_merge->set_arg(5, 4, &kmap[0]);
                k_merge->set_arg(6, 4, &kmap[1]);
                // q = 5 is prime but not stored mod 30: keep it in S5 (range bottom)
                const uint64_t Q65 = q_first + 2 * uint64_t(sseg.phi5);
                const uint32_t q5_t = (q_first <= 5 && Q65 <= 5) ? uint32_t((5 - Q65) / 6) : ~0u;
                k_merge->set_arg(7, 4, &q5_t);
                k_merge->launch(ctx.list(), (2 * ((s_words + 4) / 5) + WG_SIZE - 1) / WG_SIZE, ev.splitk[slot]);
                ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
            } else if (class_mode) {
                s_words = 2 * c5w;
                s1p = reinterpret_cast<uint32_t*>(b.prime_map[slot]);
                s5p = s1p + 2 * size_t(c5w);
            } else {
                k_split->set_arg(0, sizeof(b.prime_map[slot]), &b.prime_map[slot]);
                k_split->set_arg(1, sizeof(map32_words), &map32_words);
                k_split->set_arg(2, sizeof(b.split_s1[slot]), &b.split_s1[slot]);
                k_split->set_arg(3, sizeof(b.split_s5[slot]), &b.split_s5[slot]);
                k_split->set_arg(4, sizeof(s_words), &s_words);
                k_split->set_arg(5, sizeof(sseg.phi1), &sseg.phi1);
                k_split->set_arg(6, sizeof(sseg.phi5), &sseg.phi5);
                k_split->launch(ctx.list(), (s_words + WG_SIZE - 1) / WG_SIZE, ev.splitk[slot]);
                ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
            }

            const uint32_t per_item = a_w64 * 64;
            const uint32_t ec_max = std::max({sseg.ec[0], sseg.ec[1], sseg.ec[2]});
            const uint32_t items_pad = ((ec_max + per_item - 1) / per_item + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
            k_sa->set_arg(0, sizeof(s1p), &s1p);
            k_sa->set_arg(1, sizeof(s5p), &s5p);
            k_sa->set_arg(2, sizeof(s_words), &s_words);
            k_sa->set_arg(3, sizeof(b.checked[slot]), &b.checked[slot]);
            k_sa->set_arg(4, sizeof(chk_stride), &chk_stride);
            k_sa->set_arg(5, sizeof(b.split_cands), &b.split_cands);
            k_sa->set_arg(6, sizeof(b.split_meta), &b.split_meta);
            k_sa->set_arg(7, sizeof(b.pending[slot]), &b.pending[slot]);
            k_sa->set_arg(8, sizeof(b.pending_n[slot]), &b.pending_n[slot]);
            k_sa->set_arg(9, sizeof(pending_cap), &pending_cap);
            k_sa->set_arg(10, sizeof(items_pad), &items_pad);
            for (uint32_t c = 0; c < 3; ++c) {
                k_sa->set_arg(11 + c, 4, &sseg.ec[c]);
                k_sa->set_arg(14 + c, 4, &sseg.e0[c]);
                k_sa->set_arg(17 + c, 4, &sseg.v[c]);
            }
            k_sa->set_arg(20, 4, &split_write_checked);
            k_sa->launch(ctx.list(), 3 * items_pad / WG_SIZE, ev.checka[slot]);
            ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));

            k_sb->set_arg(0, sizeof(b.prime_map[slot]), &b.prime_map[slot]);
            k_sb->set_arg(1, sizeof(b.checked[slot]), &b.checked[slot]);
            k_sb->set_arg(2, sizeof(chk_stride), &chk_stride);
            k_sb->set_arg(3, sizeof(b.split_rest), &b.split_rest);
            for (uint32_t c = 0; c < 3; ++c) {
                k_sb->set_arg(4 + c, 4, &split_plan.roff[c]);
                k_sb->set_arg(7 + c, 4, &split_plan.rcnt[c]);
            }
            k_sb->set_arg(10, sizeof(b.pending[slot]), &b.pending[slot]);
            k_sb->set_arg(11, sizeof(b.pending_n[slot]), &b.pending_n[slot]);
            k_sb->set_arg(12, sizeof(b.leftovers[slot]), &b.leftovers[slot]);
            k_sb->set_arg(13, sizeof(b.leftover_n[slot]), &b.leftover_n[slot]);
            k_sb->set_arg(14, sizeof(q_first), &q_first);
            k_sb->set_arg(15, sizeof(q_odds), &q_odds);
            k_sb->set_arg(16, sizeof(seg_start), &seg_start);
            k_sb->set_arg(17, sizeof(pending_cap), &pending_cap);
            for (uint32_t c = 0; c < 3; ++c) k_sb->set_arg(18 + c, 4, &sseg.e0[c]);
            k_sb->set_arg(21, 4, &split_write_checked);
            k_sb->set_arg(22, 4, &cm_u32);
            k_sb->set_arg(23, 4, &ga1);
            k_sb->set_arg(24, 4, &ga2);
            k_sb->set_arg(25, 4, &off5);
            k_sb->set_arg(26, sizeof(b.ridge[slot]), &b.ridge[slot]);
            k_sb->set_arg(27, 4, &ridge_thresh_u32);
            k_sb->launch(ctx.list(), phase_b_groups, ev.check[slot]);  // grid-stride
        } else {
            k_check.launch(ctx.list(), a_groups, ev.checka[slot]);
            // Phase B consumes phase A's pending list (visibility: barrier).
            ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
            k_check_b.set_arg(0, sizeof(b.prime_map[slot]),  &b.prime_map[slot]);
            k_check_b.set_arg(1, sizeof(b.checked[slot]),    &b.checked[slot]);
            // Twin checker: phase B probes the uncovered candidates only.
            const uint32_t rest_count = uint32_t(rest_primes.size());
            const bool use_rest = b.rest_primes != nullptr;
            k_check_b.set_arg(2, sizeof(b.gb_primes),        use_rest ? &b.rest_primes : &b.gb_primes);
            k_check_b.set_arg(3, sizeof(gb_count),           use_rest ? &rest_count : &gb_count);
            k_check_b.set_arg(4, sizeof(b.pending[slot]),    &b.pending[slot]);
            k_check_b.set_arg(5, sizeof(b.pending_n[slot]),  &b.pending_n[slot]);
            k_check_b.set_arg(6, sizeof(b.leftovers[slot]),  &b.leftovers[slot]);
            k_check_b.set_arg(7, sizeof(b.leftover_n[slot]), &b.leftover_n[slot]);
            k_check_b.set_arg(8, sizeof(q_first),            &q_first);
            k_check_b.set_arg(9, sizeof(q_odds),             &q_odds);
            k_check_b.set_arg(10, sizeof(seg_start),         &seg_start);
            const uint32_t b_from = 0u;   // twin: probe all candidates
            k_check_b.set_arg(11, sizeof(b_from),            &b_from);
            k_check_b.set_arg(12, sizeof(pending_cap),       &pending_cap);
            k_check_b.set_arg(13, sizeof(b.ridge[slot]),     &b.ridge[slot]);
            k_check_b.set_arg(14, 4, &ridge_thresh_u32);
            k_check_b.launch(ctx.list(), phase_b_groups, ev.check[slot]);  // grid-stride
        }
        const double enq_s = seconds_since(t_enq);
        const double launch_s = seconds_since(t_launch0);
        const double l3_s = seconds_since(t_launch0);

        // ---- Collect the PREVIOUS segment's results (other slot) ----------
        // The host blocks here only after the next segment's work is already
        // queued, so the GPU transitions straight from segment k-1's check to
        // segment k's sieve without an idle gap.
        double wait_s = 0.0, coll_s = 0.0;
        if (stats.segments > 0) {
            const uint32_t prev = uint32_t((stats.segments - 1) & 1);
            const auto t_wait = clock_::now();
            ZE_CHECK(zeEventHostSynchronize(ev.check[prev], UINT64_MAX));
            wait_s = seconds_since(t_wait);
            const auto t_coll = clock_::now();

            // Per-kernel GPU times from event timestamps.
            double ds = 0.0, dl = 0.0, dv = 0.0;
            query_slot(prev, ds, dl, dv);
            sieve_seconds  += ds;
            large_seconds  += dl;
            verify_seconds += dv;

            if (bkt_err && *bkt_err)
                throw std::runtime_error("bucket sieve: device error flags " + std::to_string(*bkt_err));
            if (verify_diff && *verify_diff)
                throw std::runtime_error("GB_BUCKET_VERIFY: bucket map differs from the reference map in " +
                                         std::to_string(*verify_diff) + " words");
            // Two-phase checker: fail closed if phase A's pending list
            // overflowed (evens beyond the cap were never checked).
            if (*b.pending_n[prev] > pending_cap)
                throw std::runtime_error("two-phase checker: pending list overflow");
            if (phase_trace)
                std::fprintf(stderr, "    pending %u\n", *b.pending_n[prev]);
            // Read the 4-byte counter directly from shared memory (no D2H copy).
            const uint32_t n_left = *b.leftover_n[prev];
            total_leftovers += n_left;
            ridges->merge(b.ridge[prev], prev_seg_start);
            if (stats_on) collect_stats(prev, prev_seg_start);

            if (collect_segments) {
                // prev_stat is the record of the segment enqueued one
                // iteration ago (seg_stat already holds the newer one).
                prev_stat.sieve_seconds  = ds;
                prev_stat.large_seconds  = dl;
                prev_stat.check_seconds  = dv;
                prev_stat.leftovers      = n_left;
                prev_stat.offset_seconds = prev_seg_offsets;
                prev_stat.wall_seconds   = prev_seg_wall;
                stats.per_segment.push_back(prev_stat);
            }

            if (n_left > 0)
                resolve_leftovers(ctx, b.leftovers[prev], n_left, prev_seg_start,
                                  prev_seg_evens, gb_primes, verbose, stats.segments - 1, &*ridges,
                                  gstats.get(), sampled_slot[prev]);
            ridges->save(ridge_file);
            coll_s = seconds_since(t_coll);
        }
        sum_prep += prep_s;
        sum_enq += enq_s;
        sum_wait += wait_s;
        sum_coll += coll_s;
        sum_args += args_s;
        sum_launch += launch_s;
        if (phase_trace) {
            std::fprintf(stderr,
                "  phase seg %llu: prep %.1f ms (off %.1f tab %.1f)  enq %.1f ms "
                "(args %.1f l1 %.2f l2 %.2f l3 %.2f)  wait %.1f ms  coll %.1f ms\n",
                (unsigned long long)stats.segments,
                prep_s * 1e3, offset_seconds - offset_before,
                (table_seconds - table_before) * 1e3,
                enq_s * 1e3, args_s * 1e3, l1_s * 1e3, (l2_s - l1_s) * 1e3,
                (l3_s - l2_s) * 1e3, wait_s * 1e3, coll_s * 1e3);
        }

        prev_seg_start = seg_start;
        prev_seg_evens = seg_evens;

        // Finalize the carry-over for this segment: its GPU work will be
        // collected in the NEXT loop iteration (or the drain) — that is the
        // point at which prev_seg_wall / prev_seg_offsets are consumed.
        prev_seg_wall    = seconds_since(t_seg_wall);
        prev_seg_offsets = offset_seconds - offset_before;
        prev_stat = seg_stat;

        ++stats.segments;

        if (verbose && (stats.segments % 10 == 0 || seg_end == limit)) {
            std::printf("  segment %llu: evens [%llu, %llu]  leftovers_total=%llu\n",
                        (unsigned long long)stats.segments,
                        (unsigned long long)seg_start,
                        (unsigned long long)seg_end,
                        (unsigned long long)total_leftovers);
        }

        // The merged final segment already includes limit. Without this
        // break, the for-loop increment lands exactly on limit and schedules
        // it again as a duplicate one-even segment, recreating the expensive
        // high-range tail pathology that the merge is intended to remove.
        if (seg_end == limit) break;
    }

    // ---- Drain: collect the final segment --------------------------------
    {
        const uint32_t last = uint32_t((stats.segments - 1) & 1);
        ZE_CHECK(zeEventHostSynchronize(ev.check[last], UINT64_MAX));

        double ds = 0.0, dl = 0.0, dv = 0.0;
        query_slot(last, ds, dl, dv);
        sieve_seconds  += ds;
        large_seconds  += dl;
        verify_seconds += dv;

        if (bkt_err && *bkt_err)
            throw std::runtime_error("bucket sieve: device error flags " + std::to_string(*bkt_err));
        if (verify_diff && *verify_diff)
            throw std::runtime_error("GB_BUCKET_VERIFY: bucket map differs from the reference map in " +
                             std::to_string(*verify_diff) + " words");
        if (*b.pending_n[last] > pending_cap)
            throw std::runtime_error("two-phase checker: pending list overflow");
        const uint32_t n_left = *b.leftover_n[last];
        total_leftovers += n_left;
        ridges->merge(b.ridge[last], prev_seg_start);
        if (stats_on) collect_stats(last, prev_seg_start);

        if (collect_segments) {
            // seg_stat still holds the FINAL segment here: it was never
            // overwritten after its enqueue, and its timestamps are now
            // readable — this is the one segment collected without delay.
            seg_stat.sieve_seconds  = ds;
            seg_stat.large_seconds  = dl;
            seg_stat.check_seconds  = dv;
            seg_stat.leftovers      = n_left;
            seg_stat.offset_seconds = prev_seg_offsets;
            seg_stat.wall_seconds   = prev_seg_wall;
            stats.per_segment.push_back(seg_stat);
        }

        if (n_left > 0)
            resolve_leftovers(ctx, b.leftovers[last], n_left, prev_seg_start,
                              prev_seg_evens, gb_primes, verbose, stats.segments - 1, &*ridges,
                              gstats.get(), sampled_slot[last]);
        ridges->save(ridge_file);
    }

    stats.gpu_seconds = seconds_since(t_gpu);
    if (phase_trace) {
        std::fprintf(stderr,
            "phase sums over %llu segments: prep %.1f ms  enq %.1f ms "
            "(args %.1f launch %.1f)  wait %.1f ms  coll %.1f ms (gpu wall %.1f ms)\n",
            (unsigned long long)stats.segments,
            sum_prep * 1e3, sum_enq * 1e3, sum_args * 1e3, sum_launch * 1e3,
            sum_wait * 1e3, sum_coll * 1e3,
            stats.gpu_seconds * 1e3);
    }
    stats.total_seconds = stats.init_seconds + stats.gpu_seconds;
    stats.evens_per_second = double(total_evens) / stats.gpu_seconds;
    stats.leftovers = total_leftovers;
    stats.sieve_seconds  = sieve_seconds;
    stats.large_seconds  = large_seconds;
    stats.verify_seconds = verify_seconds;
    // Host offset preparation is now GPU-side (prep_large_fb); the remaining
    // host work per segment is the tile table + masks, overlapped with GPU
    // execution. Report a non-negative residual rather than a misleading
    // negative sync.
    stats.sync_seconds   = std::max(0.0, stats.gpu_seconds - sieve_seconds -
                                         large_seconds - verify_seconds);
    stats.offset_seconds = offset_seconds;
    stats.table_seconds  = table_seconds;
    stats.start          = start_even;
    stats.split_used     = sieve_split;
    stats.total_tiles       = sum_tiles;
    stats.total_large_items = sum_large_items;
    stats.ridge_min    = ridges->active() ? ridges->floor() : 0;
    stats.ridge_found  = ridges->appends();
    stats.ridge_dropped = ridges->dropped();
    stats.ridge_top    = ridges->top();
    if (stats_on) {
        gstats->finish_run(start_even, limit, total_evens);
        stats.stats_on     = true;
        stats.stats_pmax   = gstats->run_pmax();
        stats.stats_primes = gstats->run_primes();
        stats.stats_twins  = gstats->run_twins();
        gstats->write_live(start_even, limit, total_evens, total_evens, seconds_since(t_live0), true);
        stats.stats_sampled = gstats->evens_sampled();
        stats.stats_tail    = gstats->run_tail();
        gstats->commit(VERSION, sampled_evens);
    }
    if (ridges->overflowed())
        std::fprintf(stderr, "warning: ridge append buffer over capacity; excess appends were "
                             "dropped by the kernel (%u per segment; should not happen at the "
                             "default floor once the top list fills)\n", RIDGE_CAP);

    ev.destroy();

    // Free GPU memory.
    for (int s = 0; s < 2; ++s) {
        Context::free_mem(ctx.context(), b.prime_map[s]);
        Context::free_mem(ctx.context(), b.checked[s]);
        Context::free_mem(ctx.context(), b.first_bits[s]);
        Context::free_mem(ctx.context(), b.mask_table[s]);
        if (b.group_masks[s]) Context::free_mem(ctx.context(), b.group_masks[s]);
        if (b.group_y[s]) Context::free_mem(ctx.context(), b.group_y[s]);
        Context::free_mem(ctx.context(), b.leftovers[s]);
        Context::free_mem(ctx.context(), b.leftover_n[s]);
        Context::free_mem(ctx.context(), b.tile_first[s]);
        if (b.pending[s]) Context::free_mem(ctx.context(), b.pending[s]);
        if (b.pending_n[s]) Context::free_mem(ctx.context(), b.pending_n[s]);
        if (b.ridge[s]) Context::free_mem(ctx.context(), b.ridge[s]);
        if (b.split_s1[s]) Context::free_mem(ctx.context(), b.split_s1[s]);
        if (b.split_s5[s]) Context::free_mem(ctx.context(), b.split_s5[s]);
        Context::free_mem(ctx.context(), stats_block[s]);
    }
    for (void* p : stats_unpacked) Context::free_mem(ctx.context(), p);
    if (b.split_cands) Context::free_mem(ctx.context(), b.split_cands);
    if (b.split_meta) Context::free_mem(ctx.context(), b.split_meta);
    if (b.split_rest) Context::free_mem(ctx.context(), b.split_rest);
    for (void* p : {(void*)b.c30_cands, (void*)b.c30_meta, (void*)b.c30_rest, (void*)b.c30_rtab})
        if (p) Context::free_mem(ctx.context(), p);
    Context::free_mem(ctx.context(), d_group_desc);
    Context::free_mem(ctx.context(), b.gb_primes);
    if (b.twin_primes) Context::free_mem(ctx.context(), b.twin_primes);
    if (b.rest_primes) Context::free_mem(ctx.context(), b.rest_primes);
    Context::free_mem(ctx.context(), b.sieve_primes);
    Context::free_mem(ctx.context(), d_large_bases);
    Context::free_mem(ctx.context(), d_bkt_entries);
    if (d_bkt_entries1) Context::free_mem(ctx.context(), d_bkt_entries1);
    Context::free_mem(ctx.context(), d_bkt_counts);
    if (d_hl_lists) Context::free_mem(ctx.context(), d_hl_lists);
    if (d_hl_counts) Context::free_mem(ctx.context(), d_hl_counts);
    Context::free_mem(ctx.context(), bkt_err);
    Context::free_mem(ctx.context(), d_full_bases);
    Context::free_mem(ctx.context(), d_full_gaps);
    Context::free_mem(ctx.context(), d_verify_map);
    Context::free_mem(ctx.context(), verify_diff);
    Context::free_mem(ctx.context(), d_large_gaps);
    Context::free_mem(ctx.context(), b.magic);
    Context::free_mem(ctx.context(), b.shifts);
    Context::free_mem(ctx.context(), b.ms32);

    return std::move(stats);   // a member now: move, not copy
}

} // namespace

RunStats run_gpu(uint64_t limit,
                 uint64_t segment_even,
                 const std::string& spv_path,
                 bool verbose,
                 uint64_t start_even,
                 uint32_t sieve_split_arg,
                 const char* prime_cache_path,
                 bool collect_segments) {
    DenseRun run(limit, segment_even, spv_path, verbose, start_even, sieve_split_arg,
                 prime_cache_path, collect_segments);
    return run.run();
}

// ------------------------------------------------------- CPU reference ----

bool verify_cpu(uint64_t limit, bool verbose) {
    for (uint64_t n = 4; n <= limit; n += 2) {
        bool ok = (n == 4); // 4 = 2 + 2 (the only even+even partition)
        for (uint64_t p = 3; p <= n / 2; p += 2) {
            if (is_prime_trial(p) && is_prime_trial(n - p)) { ok = true; break; }
        }
        if (!ok) {
            if (verbose)
                std::printf("CPU check FAILED at n=%llu\n", (unsigned long long)n);
            return false;
        }
        if (verbose && n % 1000000 == 0)
            std::printf("  cpu: verified up to %llu\n", (unsigned long long)n);
    }
    return true;
}

// ------------------------------------------------------- device info -------

void print_device_info() {
    ze_result_t r = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (r != ZE_RESULT_SUCCESS)
        throw std::runtime_error("zeInit failed: " + std::to_string(std::to_underlying(r)));

    uint32_t nd = 0;
    zeDriverGet(&nd, nullptr);
    if (nd == 0) throw std::runtime_error("no Level Zero driver found");
    std::vector<ze_driver_handle_t> drivers(nd);
    zeDriverGet(&nd, drivers.data());

    bool found = false;
    for (ze_driver_handle_t d : drivers) {
        uint32_t nv = 0;
        zeDeviceGet(d, &nv, nullptr);
        std::vector<ze_device_handle_t> devs(nv);
        zeDeviceGet(d, &nv, devs.data());

        for (ze_device_handle_t dev : devs) {
            ze_device_properties_t p{};
            p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            if (zeDeviceGetProperties(dev, &p) != ZE_RESULT_SUCCESS) continue;
            if (p.type != ZE_DEVICE_TYPE_GPU) continue;
            found = true;

            ze_device_properties_t p12{};   // timerResolution lives here (v1.17+)
            p12.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
            const bool have12 =
                zeDeviceGetProperties(dev, &p12) == ZE_RESULT_SUCCESS;
            const double timer_hz = have12 ? double(p12.timerResolution) : 0.0;

            ze_device_memory_properties_t m{};
            m.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
            uint32_t nm = 1;
            zeDeviceGetMemoryProperties(dev, &nm, &m);

            ze_device_compute_properties_t c{};
            c.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
            zeDeviceGetComputeProperties(dev, &c);

            ze_api_version_t api = ZE_API_VERSION_1_0;
            zeDriverGetApiVersion(d, &api);

            std::printf("device           : %s\n", p.name);
            std::printf("vendor id        : 0x%04x  (Intel = 0x8086)\n", p.vendorId);
            std::printf("device id        : 0x%04x\n", p.deviceId);
            std::printf("numEUs           : %u (threads/EU %u)\n",
                        p.numEUsPerSubslice * p.numSubslicesPerSlice * p.numSlices,
                        p12.numThreadsPerEU);
            std::printf("slices/subslices : %u / %u  (EUs per subslice %u)\n",
                        p.numSlices, p.numSubslicesPerSlice, p.numEUsPerSubslice);
            std::printf("core clock       : %u MHz\n", p.coreClockRate);
            std::printf("timer resolution : %.0f ns/tick\n",
                        timer_hz > 0 ? 1e9 / timer_hz : 0.0);
            std::printf("memory           : %.1f GB (%u MHz, %u-bit)\n",
                        double(m.totalSize) / double(1ull << 30),
                        m.maxClockRate, m.maxBusWidth);
            std::printf("work-group size  : max %u\n", c.maxGroupSizeX);
            std::printf("subgroup sizes   :");
            for (uint32_t i = 0; i < c.numSubGroupSizes; ++i)
                std::printf(" %u", c.subGroupSizes[i]);
            std::printf("\n");
            std::printf("L0 API version   : %u.%u\n",
                        ZE_MAJOR_VERSION(api), ZE_MINOR_VERSION(api));
            ze_driver_properties_t dp{};
            dp.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
            if (zeDriverGetProperties(d, &dp) == ZE_RESULT_SUCCESS)
                std::printf("driver version   : %u\n", dp.driverVersion);
            std::printf("\nconfig           : WG %u | TILE_WORDS %u (%.0f KB SLM) | "
                        "WPI(large) %u | check words/item %u | MAX_GB_PRIME %u | "
                        "seg default %llu\n",
                        WG_SIZE, TILE_WORDS, double(TILE_WORDS * 8) / 1024.0,
                        LARGE_PRIMES_PER_ITEM, CHECK_WORDS_PER_ITEM,
                        MAX_GB_PRIME,
                        (unsigned long long)DEFAULT_SEGMENT_EVEN);
        }
    }
    if (!found) throw std::runtime_error("no GPU device reported by Level Zero");
}

} // namespace gb
