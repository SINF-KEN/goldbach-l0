#include "verifier.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "config.hpp"
#include "env.hpp"
#include "l0_context.hpp"
#include "primes.hpp"
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
//   d_gb_primes   : uint16 candidate primes p <= 9781 (uploaded once)
//   d_sieve_primes: uint32 base primes (uploaded once)
//   d_leftovers   : uint32 indices of unresolved even numbers
//   d_leftover_n  : uint32 count of leftovers (device counter)
struct GpuBuffers {
    // Read-only tables shared by both pipeline slots (uploaded once).
    uint32_t* sieve_primes = nullptr;
    uint64_t* magic        = nullptr;  // per-prime ceil(2^64 / p) reciprocal
    uint32_t* shifts       = nullptr;  // per-prime shift for div-by-p

    // Per-slot buffers: the segment pipeline alternates between slot 0 and
    // slot 1 so segment k's GPU work overlaps segment k-1's host work.
    uint64_t* prime_map[2]  = {};  // odd-prime bitset for the q segment
    uint64_t* first_bits[2] = {};  // per-prime first markable bit (segment-global)
    uint64_t* mask_table[2] = {};  // small-prime word masks (per segment: q_first-dependent)
    uint32_t* leftovers[2]  = {};  // compact exception list
    uint32_t* leftover_n[2] = {};  // leftover count (shared: host-readable)
    uint32_t* tile_first[2] = {};  // [tile * sieve_split + j] per-tile first mark
    uint64_t* group_masks[2] = {}; // fused mask stage: per-group mask tables
    uint32_t* group_y[2] = {};     // class modes: per class x group canonical-table word offsets
    uint32_t* pending[2]    = {};  // two-phase checker: evens left by phase A
    uint32_t* pending_n[2]  = {};  // their count (shared: host-checked for overflow)
    // Positive coverage (check30_a / check30_b): [0] evens phase A resolved,
    // [1] pending evens phase B resolved (shared). The host requires
    // [0] + pending == segment evens and [1] + leftovers == pending, so a
    // checker launch that did not run, or covered too few evens, fails the
    // run instead of passing it.
    uint32_t* acct[2]       = {};
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
    uint64_t* ms32          = nullptr; // 32-bit reciprocals for the table path
    uint16_t* c30_cands     = nullptr; // mod-30 checker (check30.hpp)
    uint32_t* c30_meta      = nullptr;
    uint16_t* c30_rest      = nullptr;
    uint32_t* c30_rtab      = nullptr;
};

// Mask-group bound shared with the kernel (check_constants.py checks it).
inline constexpr uint32_t MAX_MASK_GROUPS_HOST = 8;   // kernel MAX_MASK_GROUPS
inline constexpr uint32_t HL_MAX_K_HOST = 256;  // kernels/goldbach.cl HL_MAX_K (SLM counters)
// hl_harvest work-group size (2026-11-18: the scattered list stores are L2
// write-combined; a larger group fills its regions' lines faster, harvest
// -6 ms per 1e11 vs 256). The kernel needs >= HL_MAX_K_HOST work-items.
inline constexpr uint32_t HL_HARVEST_WG = 1024;
static_assert(HL_HARVEST_WG >= HL_MAX_K_HOST && HL_HARVEST_WG <= 1024, "hl_harvest: group size");
// Segments per harvest batch (GB_HITLIST_K). 2026-11-18 with the 1024 groups,
// harvest ms per 1e11: K 32 47.9, 48 41.4, 64 41.8, 80 41.3, 96 44.7, 128 44.1.
// 2026-11-18 (r7): K up to 256 within the list budget below: a batch has a
// ~7 ms fixed part (per prime: list read, setup, first table word), so the
// largest K that fits pays (K 128 -> 147 at 4e18: harvest 0.173 -> 0.166
// ms per segment).
inline constexpr uint32_t HL_K_DEFAULT = 256;
// List memory cap (bytes, GB_HL_LIST_MAX): K is lowered until K x regions
// fit. 3.5e9 = the r7 list size at 5e18 (K 128 x 3358 x 2048 entries),
// now that the exact hit estimate and balanced regions need ~15 % fewer.
inline constexpr uint64_t HL_LIST_MAX_DEFAULT = 3500000000ull;
static_assert(WG_SIZE >= HL_MAX_K_HOST, "hl_harvest: one thread per SLM batch counter");

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
// pipeline slots. One pool holds all of them; each event is still reset on
// its own (zeEventHostReset acts on one event, not on its pool), as the
// sparse path's SparseEvents always did.
struct SlotEvents {
    static constexpr uint32_t kEvents = 22;
    ze_event_pool_handle_t pool = nullptr;
    ze_event_handle_t sieve[2] = {};
    ze_event_handle_t large[2] = {};
    ze_event_handle_t check[2] = {};   // last checker kernel (host waits on it)
    ze_event_handle_t checka[2] = {};  // two-phase checker: phase A (timing only)
    ze_event_handle_t bucket[2] = {};  // hl_apply / bucket_process (timing only; counted in large_s)
    ze_event_handle_t splitk[2] = {};  // split_map (timing only; counted in check_s)
    ze_event_handle_t xv[3][2] = {};   // check30_xs8 / 6 / 4 (timing only; counted in check_s)
    ze_event_handle_t hl[2] = {};      // hl_harvest (timing only; counted in large_s)
    ze_event_handle_t mw[2] = {};      // mid_window (timing only; counted in sieve_s)

    /// Creates the pool and its twenty-two events.
    /// @throws std::runtime_error on failure; destroy() is safe afterwards.
    void create(ze_context_handle_t ctx) {
        // KERNEL_TIMESTAMP is load-bearing beyond timing: timestamp events
        // flush on signal (PROJECT-REFERENCE.md §14).
        ze_event_pool_desc_t pd{};
        pd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        pd.count = kEvents;
        pd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
        ZE_CHECK(zeEventPoolCreate(ctx, &pd, 0, nullptr, &pool));
        for (uint32_t i = 0; i < kEvents; ++i) {
            ze_event_desc_t ed{};
            ed.stype  = ZE_STRUCTURE_TYPE_EVENT_DESC;
            ed.index  = i;
            ed.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
            ed.wait   = ZE_EVENT_SCOPE_FLAG_DEVICE;
            ze_event_handle_t* e = (i < 2) ? &sieve[i]
                                 : (i < 4) ? &large[i - 2]
                                 : (i < 6) ? &check[i - 4]
                                 : (i < 8) ? &checka[i - 6]
                                 : (i < 10) ? &bucket[i - 8]
                                 : (i < 12) ? &splitk[i - 10]
                                 : (i < 18) ? &xv[(i - 12) / 2][(i - 12) % 2]
                                 : (i < 20) ? &hl[i - 18]
                                            : &mw[i - 20];
            ZE_CHECK(zeEventCreate(pool, &ed, e));
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
            if (mw[i]) zeEventDestroy(mw[i]);
        }
        if (pool) zeEventPoolDestroy(pool);
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
        const uint64_t p = cpu_min_goldbach_prime(n, gb_primes.data(), gb_primes.size(), MAX_GB_PRIME);
        const bool ok = p != 0;
        if (p > 2) {                             // 4 = 2 + 2 has no odd witness to record
            if (ridges) ridges->consider(n, p, seg_start);
            if (gstats) gstats->add_cpu(n, p, sampled);
        }
        if (p > MAX_GB_PRIME)
            std::fprintf(stderr, "note: n=%llu needs p=%llu > MAX_GB_PRIME "
                         "(new p_min record candidate)\n",
                         (unsigned long long)n, (unsigned long long)p);
        if (!ok)
            throw std::runtime_error("Goldbach verification FAILED for n=" +
                                     std::to_string(n));
    }
}

/// One segment of the loop (stack object, one per iteration): geometry, the
/// class-map kernel arguments derived from it, and the host timings the
/// iteration's steps record. Kernel arguments are passed by address and
/// copied by set_arg, so the fields serve exactly as run_gpu's locals did.
struct Segment {
    uint64_t seg_start = 0;          // first even n
    uint64_t seg_end   = 0;          // last even n (inclusive)
    uint64_t seg_evens = 0;
    uint64_t q_first   = 0;
    uint64_t q_last    = 0;
    uint64_t q_odds    = 0;
    uint64_t tiles     = 0;          // dense-map tiles
    uint32_t slot      = 0;          // pipeline slot
    ClassGeo cgeo;                   // class mode: class arrays of cgeo.tpc tiles
    uint64_t tiles_launch = 0;       // tile-kernel work-groups
    uint32_t c5w  = 0;               // class stride (u64 words)
    uint32_t off5 = 0;               // class stride (bits)
    uint32_t phis = 0;               // class phases (kernel arg)
    uint32_t ga1 = 0, ga2 = 0;       // phi1 / phi5 kernel args
    uint32_t mask_coff = 0;          // group-mask table offset per class
    uint32_t mask_count_seg = 0;     // small-prime mask entries
    // host timings (GB_PHASE trace and telemetry)
    clock_::time_point t_enq, t_launch0;
    double args_s = 0.0;
    double l1_s = 0.0;
    double l2_s = 0.0;
    double enq_s = 0.0, launch_s = 0.0, l3_s = 0.0;
    double wait_s = 0.0, coll_s = 0.0;   // collection of the previous segment
    double table_before = 0.0;
    double prep_s = 0.0;
    clock_::time_point t_seg_wall;
    double offset_before = 0.0;
};

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
        (void)locate_module(spv_path);   // a missing module fails before the base primes
        load_base_primes();
        // ---- Level Zero setup starts here: DenseRun's members follow.
        t_module = clock_::now();
        init_mark("pre-module");
    }

    // ---- resolve_range ----------------------------------------------------
    uint32_t phase_b_groups = 0;     // GB_CHECK_BGROUPS (check30_b grid)
    uint32_t stats_b_groups = 0;     // GB_STATS_BGROUPS
    uint32_t bucket_groups_env = 0;  // GB_BUCKET_GROUPS (0: unset, mode default)
    RunStats stats;
    clock_::time_point t_start;
    uint64_t total_evens = 0;        // evens in [start_even, limit]

    // ---- load_base_primes -------------------------------------------------
    gb::PrimeCacheView prime_view;
    // its payload check (hash + content, ~0.12 s), run beside the GPU init
    // (DenseRun::run) and joined at the end of init (create_events); declared
    // after prime_view, so an early exit joins it before the mapping goes away
    std::future<const char*> cache_check;
    const uint32_t* sieve_primes = nullptr;  // mmap-backed (or owned) by prime_view
    double cache_seconds = 0.0;
    uint32_t mask_count = 0;         // primes < 64: the small-prime mask path
    // The mask stage clears every multiple of 7..29: cop_marks (medium items,
    // mid windows) skips those marks (docs/EXACTNESS.md rule 2b).
    bool mask_through_29() const { return mask_count > 0 && sieve_primes[mask_count - 1] >= 29; }
    uint32_t sieve_count = 0;        // base primes
    uint32_t sieve_split = 0;        // [0, split): tile walk; [split, count): large kernels
    uint64_t midwin_p2 = 0;          // GB_MIDWIN: mid-window bound (0 = off)
    bool midwin_pre = false;         // its preconditions hold (sets the auto split)
    uint32_t large_count = 0;
    bool init_trace = false;         // GB_INIT_TRACE
    clock_::time_point t_mark;       // init_mark's previous mark
    std::vector<uint32_t> gb_all;    // odd candidate primes <= MAX_GB_PRIME
    std::vector<uint16_t> gb_primes;
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
          k_large(mod, "sieve_large_packed") {}

    RunStats run();

private:
    // Phases, in run() order
    void configure_checkers();
    void configure_ridges_and_stats();
    void set_group_sizes();
    void size_buffers();
    void allocate_slot_buffers();
    void size_bucket_ring_and_hitlists();
    void allocate_large_prime_buffers();
    void upload_tables();
    void build_mtab();
    void pack_stats_buffers();
    void create_events();
    void segment_loop();
    void assemble_run_stats();
    void release_device_objects();

    // segment_loop: run-wide state, then the steps of each iteration, then the drain
    void begin_segment_loop();
    void begin_segment(Segment& sg);
    void prepare_segment_tables(Segment& sg);
    void enqueue_sieve(Segment& sg);
    void enqueue_large_primes(Segment& sg);
    void enqueue_midwin(const Segment& sg, uint64_t* map, ze_event_handle_t done);
    void enqueue_hitlists(const Segment& sg, uint32_t& hl_jb);
    void enqueue_packed_large(const Segment& sg, uint32_t hl_jb);
    void enqueue_bucket_verify(const Segment& sg);
    void enqueue_check(Segment& sg);
    void enqueue_count30(const Segment& sg, bool sampled);
    void enqueue_check_sampled(const Segment& sg, const Seg30& s30, uint32_t ridge_thresh_u32);
    void enqueue_check30(const Segment& sg, const Seg30& s30, uint32_t ridge_thresh_u32);
    void collect_previous_segment(Segment& sg);
    void end_iteration(const Segment& sg);
    void drain_final_segment();
    void collect_slot(uint32_t sl, SegmentStats& rec);
    void reset_slot_counters(uint32_t sl);
    void check_coverage(uint32_t sl, uint32_t pend, uint32_t n_left);

    // Helpers (former lambdas of run_gpu)
    void reset_gaps(int s);
    void collect_stats(uint32_t sl, uint64_t seg_start_sl);
    bool query_slot(uint32_t slot, double& ds, double& dl, double& dv);

    // Level Zero objects, created in this order by the constructor (and
    // destroyed in reverse order after every member declared below).
    Context ctx;
    Module  mod;
    Kernel  k_sieve;
    Kernel  k_large;

    // ---- configure_checkers ---------------------------------------------------
    // The map is the mod-30 class map (8 class arrays, kernel class mode 2)
    // and the checker the mod-30 one (check30_a / check30_b): v2.6 removed
    // the dense and mod-6 maps, the twin and class-split checkers and the
    // bucket ring, which only A/B knobs still reached.
    static constexpr uint32_t ncls_map = 8;
    const uint32_t cm_u32 = 2;       // kernel class-mode argument
    bool med_cop = false;            // first bits carry med_prime_cop residues (items)
    uint32_t a_w64 = 0;              // phase-A item width (CHECK_PHASE_A_W64)
    // checked[] (bit per even) is validation output of the tools only: the
    // verifier passes write_checked = 0 and no buffer (the kernels never
    // touch it then; tools/debug_check_hi.cpp allocates its own)
    const uint32_t write_checked = 0;
    uint64_t* const no_checked = nullptr;
    Plan30 plan30;
    std::unique_ptr<Kernel> k_ca, k_cb;

    // ---- configure_ridges_and_stats ---------------------------------------------
    // optional: constructed (it reads GB_RIDGE*) where run_gpu constructed it
    std::optional<RidgeTracker> ridges;
    std::string ridge_path;          // GB_RIDGE_FILE
    const char* ridge_file = nullptr;  // ridge_path.c_str()
    uint32_t ceil_c30 = 0;           // phase-A ceiling (ridge floor)
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
    std::vector<uint32_t> med_items_host;   // medium items (prime, word range lo | hi << 16)
    uint32_t med_nit = 0;            // items uploaded (0: none)
    double module_seconds = 0.0;
    uint32_t large_wg = 0;           // GB_LARGE_WG

    // ---- size_buffers ---------------------------------------------------------
    GpuBuffers b;                    // every device / shared buffer (allocated below)
    uint64_t map_words = 0;          // prime map per slot (whole tiles)
    uint32_t chk_stride30 = 0;       // mod-30 checker: words per class of checked[] (unused: no buffer)
    uint32_t pending_cap = 0;        // two-phase checker: pending list capacity
    uint64_t table_bytes = 0;        // per-tile first-mark table per slot
    bool use_table = false;
    uint32_t use_table_u32 = 0;
    uint32_t mask_period = 0;        // GB_MASK_PERIOD
    uint32_t mask_xp = 0;            // owned primes folded into the mask groups (MASK_XP_DEFAULT)
    uint32_t mask_groups_n = 0;      // fused mask groups (0: per-prime mask loop)
    size_t mask_group_words = 0;
    uint32_t* d_group_desc = nullptr;

    // ---- allocate_slot_buffers ---------------------------------------------------
    uint32_t tail_t = 0;             // check30_b tail recording bound (stats)
    bool sampled_slot[2] = {false, false}; // slot's segment is a full-histogram sample
    uint64_t slot_qfirst[2] = {0, 0};      // slot's segment position origin (gap positions)
    uint32_t slot_ggroups[2] = {0, 0};     // count30 work-groups launched for the slot's segment
    bool walked_slot[2] = {false, false};  // count30 walked the slot's gaps
    uint64_t sampled_evens = 0;            // evens of the sampled segments
    uint64_t slot_evens[2] = {0, 0};       // evens of the slot's segment
    uint64_t evens_collected = 0;          // live progress (goldbach_live.json)
    clock_::time_point t_live0;

    // ---- size_bucket_ring_and_hitlists -------------------------------------------
    uint64_t delta = 0;              // q-bit advance per segment (segment_even / 2)
    uint64_t q_odds_full = 0;        // map bits of a full segment
    uint64_t q_first0 = 0;           // first segment's q_first
    uint32_t bucket_begin = 0;       // first hit-listed prime index
    bool bucket_on = false;          // primes [bucket_begin, count) leave the packed kernel
    bool hitlist = false;            // ... into hit lists (== bucket_on)
    uint32_t hl_K = 0;               // hit lists: segments per harvest
    uint32_t hl_G = 1, hl_capg = 64; // harvest regions, entries per region
    uint64_t run_segments = 0, run_last_q_odds = 0;  // hit lists: segments of the run
    bool hl_mt = false;              // HL_MT: harvest files only hits p m with m prime
    bool mt_try = false;             // ... planned (the hit-list threshold follows it)
    /// HL_MT feasibility for hit-listed primes >= pmin: multipliers m of
    /// every hit in [m_lo, m_hi]; needs p^3 > limit and p^2 > m_hi (a
    /// composite m then has a factor < p), the table's sieving primes
    /// (<= sqrt(m_hi)) far below m_lo, m_hi < 2^38 (hl_harvest_mt's
    /// quotient words: hi < 64) and a table below 3.5 GiB.
    bool mt_geometry(uint64_t pmin, uint64_t& m_lo, uint64_t& m_hi) const {
        const uint64_t pmax = sieve_primes[sieve_count - 1];
        m_lo = q_first0 / pmax;
        m_hi = limit / pmin + 2;
        const uint64_t s_max = uint64_t(std::sqrt(double(m_hi))) + 2;
        return pmin > 1 && pmin * pmin > limit / pmin + 1 && pmin * pmin > m_hi &&
               2 * s_max < m_lo && m_hi < (uint64_t(1) << 38) &&
               (m_hi - m_lo) / 30030 * 720 < (uint64_t(7) << 29);   // table < 3.5 GiB (32-bit offsets)
    }
    uint32_t mt_k_lo = 0, mt_blocks = 0;   // multiplier table: blocks [k_lo, k_lo + blocks) of 30030
    uint32_t mt_s_begin = 0, mt_s_end = 0; // its sieving primes (indices into sieve_primes)
    uint32_t* d_mtab = nullptr;
    uint32_t packed_end = 0;         // packed kernel primes [sieve_split, packed_end)

    // ---- allocate_large_prime_buffers ---------------------------------------------
    uint32_t packed_count = 0;       // primes of the packed large kernel
    bool midwin_on = false;          // the mid-window pass runs (size_bucket_ring_and_hitlists)
    uint32_t mid_end = 0;            // mid windows: primes [sieve_split, mid_end)
    uint32_t mid_n = 0;
    uint32_t midw_bits = 0, midw_wg = 0;  // window bits, work-group size
    std::unique_ptr<Kernel> k_mids, k_midw;
    uint32_t* d_mid_i0[2] = {};      // per slot: first class bit per class and mid prime
    double midw_seconds = 0.0;       // window pass (counted in sieve_s)
    bool hl_fuse = false;            // hl_apply fused into sieve_large_packed_hl
    PackedLarge packed;
    uint32_t* d_large_bases = nullptr;
    uint32_t* d_large_gaps = nullptr;
    uint32_t* bkt_err = nullptr;     // shared device error flags (hit lists)
    uint32_t* d_hl_lists = nullptr;
    uint32_t* d_hl_counts = nullptr;
    std::unique_ptr<Kernel> k_cmp, k_large_full, k_hlh, k_hla;
    uint32_t bucket_groups = 0;      // GB_BUCKET_GROUPS or the mode default
    uint32_t bucket_verify_every = 0;  // GB_BUCKET_VERIFY (0: off)
    bool bucket_verify = false;
    PackedLarge packed_full;
    uint32_t verify_lo = 0;          // GB_BUCKET_VERIFY: packed reference primes [verify_lo, count)
    bool verify_mid = false;         // GB_BUCKET_VERIFY: the reference runs mid_window (not _FULL)
    uint32_t* d_full_bases = nullptr;
    uint32_t* d_full_gaps = nullptr;
    uint64_t* d_verify_map = nullptr;
    uint32_t* verify_diff = nullptr; // shared: differing words (map_compare)

    // ---- upload_tables --------------------------------------------------------
    std::vector<uint64_t> magic;     // per-prime magic reciprocals, [0, sieve_split)
    std::vector<uint32_t> shift;
    clock_::time_point t_upload;

    // ---- pack_stats_buffers ---------------------------------------------------
    void* stats_block[2] = {nullptr, nullptr};

    // ---- create_events ----------------------------------------------------------
    uint32_t gb_count = 0;           // candidate primes
    SlotEvents ev;

    // ---- begin_segment_loop: run-wide loop state -----------------------------------
    clock_::time_point t_gpu;
    uint64_t total_leftovers = 0;
    double sieve_seconds = 0.0, large_seconds = 0.0;
    double verify_seconds = 0.0, offset_seconds = 0.0, table_seconds = 0.0;
    uint64_t prev_seg_start = 0;  // first even n of the previously enqueued segment
    uint64_t prev_seg_evens = 0;  // its even count (leftover index bound)
    // Carried from the last ENQUEUED segment to the iteration that collects
    // it (one slot behind): host wall time inside that iteration and its
    // host offset-generation share.
    double prev_seg_wall = 0.0, prev_seg_offsets = 0.0;
    double timer_hz = 0.0;           // device timer resolution
    // Per-segment record for the segment currently being enqueued, plus the
    // completed record of the segment enqueued one iteration earlier.
    SegmentStats seg_stat, prev_stat;
    uint64_t sum_tiles = 0, sum_large_items = 0;
    bool phase_trace = false;        // GB_PHASE
    bool hl_used[2] = {false, false};   // slot's segment launched a harvest (timing)
    std::vector<uint64_t> fb_host;   // per-segment first bits (class mode: persistent)
    double sum_prep = 0.0, sum_enq = 0.0, sum_wait = 0.0, sum_coll = 0.0;
    double sum_args = 0.0, sum_launch = 0.0;
    bool split_used[2] = {false, false};   // slot's segment ran count30 (timed on ev.splitk)
    bool xs_used[2] = {false, false};      // slot's segment ran check30_xs8..3 (4 launches)
    bool acct_slot[2] = {false, false};    // slot's segment ran check30_a / _b (b.acct counts)
    uint64_t slot_hist_n[2] = {0, 0};      // sampled slot: evens its exact histogram counted
};

/// Phase 1: validates and clamps the range and segment span (run_gpu's
/// preconditions; the n = 4 fail-closed rule) and reads the grid-stride
/// work-group counts. No Level Zero call.
void RunSetup::resolve_range() {
    if (limit < 4) throw std::runtime_error("limit < 4: no even n >= 4 to verify");
    // Knobs of the checkers and map layouts removed in v2.6 are refused, not
    // ignored: a run asking for one would otherwise silently run something else.
    for (const char* gone : {"GB_SPLIT", "GB_SPLIT_CAP", "GB_SPLIT_STAGES", "GB_TWIN", "GB_TWIN_CAP", "GB_C30",
                             "GB_CLASSMAP", "GB_HITLIST", "GB_MASK_GROUPS"})
        if (env::present(gone))
            throw std::runtime_error(std::string(gone) + " was removed in v2.6 (the mod-30 map and "
                                     "checker with hit lists are the only pipeline)");
    // The speed rounds' A/B switches (2026-11-19): refused too, so an A/B arm
    // that sets one does not silently measure the default twice.
    for (const char* gone : {"GB_C30_LATE", "GB_C30_PREB", "GB_ML_OVERLAP", "GB_HL_LAM2", "GB_HL_CAPG",
                             "GB_HLH_WG", "GB_HL_UNCOMP", "GB_MT_UNCOMP", "GB_MASK_XP", "GB_MED_ITEMS",
                             "GB_MINP_FLOOR", "GB_HL_FILLDBG"})
        if (env::present(gone))
            throw std::runtime_error(std::string(gone) + " was removed on 2026-11-19 (its default is "
                                     "fixed; docs/HISTORY.md 9.2ac)");
    gb::require_supported_limit(limit);
    // The last even n is limit rounded down (as run_gpu_sparse does). An odd
    // limit left the segment and count30 ranges ending on an even q, so the
    // statistics missed a prime at limit - 2 (and the twin pair below it)
    // while the gap sum still included it.
    limit &= ~uint64_t(1);
    // Grid-stride work-group counts, validated before any GPU work: env_grid
    // refuses 0, which would skip phase B (pending evens never checked) or the
    // hit-list / bucket marks (composites left "prime") and still PASS.
    phase_b_groups = env_grid("GB_CHECK_BGROUPS", CHECK30_B_GROUPS);
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
        prime_cache_path ? prime_cache_path : "base_primes.cache", true);
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
    // v2.6 mid-prime SLM windows (HISTORY.md §9.2x): primes [split, GB_MIDWIN)
    // go to mid_window, and the auto split drops to MIDWIN_SPLIT (2026-11-17
    // grid: split 800..3500 x bound 2..4M within 0.3%). They need
    // q >= 2^60 (mod_ff) and class arrays < 2^24 bits (exact float
    // remainders); otherwise the old split. GB_MIDWIN=0 disables.
    midwin_p2 = env::u64("GB_MIDWIN", MIDWIN_P2_DEFAULT, 0, UINT32_MAX);
    {
        const uint64_t qof_est = (segment_even - 5 + MAX_GB_PRIME) / 2 + 1;
        midwin_pre = midwin_p2 > 0 && start_even >= (uint64_t(1) << 60) + MAX_GB_PRIME + 2 &&
                     (qof_est + 14) / 15 < (uint64_t(1) << 24);
    }
    if (sieve_split == 0 || sieve_split > sieve_count)
        sieve_split = std::min<uint32_t>(sieve_count, midwin_pre ? MIDWIN_SPLIT : 36000);
    sieve_split = std::max(sieve_split, mask_count);
    // The packed kernel (and the GB_BUCKET_VERIFY reference) take the first
    // odd multiple with mod_ff above q = 2^60, which is wrong for p < 512
    // (HISTORY.md §9.2z): keep every prime below MOD_FF_MIN_P in the tile walk.
    sieve_split = std::max(sieve_split, std::min(sieve_count, uint32_t(
        std::lower_bound(sieve_primes, sieve_primes + sieve_count, MOD_FF_MIN_P) - sieve_primes)));
    if (sieve_split < sieve_count && sieve_primes[sieve_split] < MOD_FF_MIN_P)
        throw std::runtime_error("split prime below MOD_FF_MIN_P (docs/EXACTNESS.md)");
    large_count = sieve_count - sieve_split;

    // Goldbach candidate primes: p <= 9781 (see config.hpp).
    init_trace = env::present("GB_INIT_TRACE");
    t_mark = clock_::now();
    init_mark("cache+setup");
    gb_all = odd_primes_up_to(MAX_GB_PRIME);
    gb_primes.reserve(gb_all.size());
    for (uint32_t p : gb_all) gb_primes.push_back(uint16_t(p));
}

/// GB_INIT_TRACE: init time since the previous mark, to stderr.
void RunSetup::init_mark(const char* what) {
    if (!init_trace) return;
    std::fprintf(stderr, "  init %-34s %7.1f ms\n", what, seconds_since(t_mark) * 1e3);
    t_mark = clock_::now();
}

/// Phase 3a: the mod-30 checker's plan and kernels.
void DenseRun::configure_checkers() {
    // Phase A item width (64-bit words): the kernel's compiled-in A_W64
    // (tests/check_constants.py). Deliberately not overridable at run time:
    // a host-only change sizes the phase-A launch for a different width and
    // leaves evens unchecked.
    a_w64 = CHECK_PHASE_A_W64;
    // Mod-30 checker (v1.13.0, include/check30.hpp): evens grouped by n mod
    // 30 read the class arrays directly. GB_C30_PLAN overrides the plan.
    const char* pe = env::raw("GB_C30_PLAN");   // make_plan30 validates it
    plan30 = make_plan30(gb_all, pe ? pe : CHECK30_PLAN);
    k_ca = std::make_unique<Kernel>(mod, "check30_a");
    k_cb = std::make_unique<Kernel>(mod, "check30_b");
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
    // Phase-A ceiling: every candidate phase A tests is below it, so an
    // even whose ridge reaches it fails phase A and phase B records its exact
    // ridge. Below it phase A may resolve the even through a larger candidate
    // first (unrecorded), so the floor is clamped to the ceiling (complete
    // above it; a GB_C30_PLAN without a bound prefix gets 9782, i.e. no
    // kernel ridges).
    ceil_c30 = plan30.member_lt;
    if (ridges->active() && ceil_c30) ridges->clamp_floor_above(ceil_c30 - 1u);
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
    stats_on = start_even >= STATS_MIN_START && stats_env;
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
    } else if (start_even < STATS_MIN_START && !env::present("GB_STATS")) {
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
    // Medium items (2026-11-18): with the mid windows the medium stage takes
    // every tile prime after the mask groups (the owned walk goes) as items of
    // about T marks (+0.6%; T 90..110 flat, 80 / 130 worse). Without them
    // (low starts) the owned walk stays.
    const uint32_t med_items_t = midwin_pre ? MED_ITEMS_MIDWIN : 0u;
    const uint32_t med_min_p = env::u32("GB_MED_P", midwin_pre ? (med_items_t ? MED_MIN_P_ITEMS : MED_MIN_P_MIDWIN)
                                                               : MED_MIN_P);
    own_end = std::clamp<uint32_t>(uint32_t(
        std::lower_bound(sieve_primes, sieve_primes + sieve_split, med_min_p) -
        sieve_primes), mask_count, sieve_split);
    module_seconds = seconds_since(t_module);
    init_mark("module");
    // Medium items (MED_ITEMS_MIDWIN marks per item, with the mid windows):
    // every medium prime split into n = round(marks
    // per tile / T) >= 1 equal tile word ranges, the items sorted by marks
    // (the lanes of a sub-group run equally long loops). Uploaded behind the
    // mask group descriptors (size_buffers).
    med_items_host.clear();
    if (const uint32_t T = med_items_t; T > 0 && own_end < sieve_split) {
        struct It { double m; uint32_t j, w; };
        std::vector<It> v;
        const uint32_t tw = uint32_t(BITS_PER_TILE / 32);
        for (uint32_t j = own_end; j < sieve_split; ++j) {
            const double m = double(BITS_PER_TILE) / sieve_primes[j];
            const uint32_t n = std::clamp<uint32_t>(uint32_t(std::lround(m / T)), 1u, tw);
            for (uint32_t i = 0; i < n; ++i)
                v.push_back({m / n, j, (i * tw / n) | (((i + 1) * tw / n) << 16)});
        }
        std::stable_sort(v.begin(), v.end(), [](const It& x, const It& y) { return x.m > y.m; });
        if (v.size() < 65536)
            for (const It& t : v) { med_items_host.push_back(t.j); med_items_host.push_back(t.w); }
    }
    // The items run med_prime_cop: the first bits of their primes carry the
    // multiplier residue, and its skips rely on the mask stage clearing
    // every multiple of 7..29 (docs/EXACTNESS.md).
    med_cop = !med_items_host.empty();
    if (med_cop && !mask_through_29())
        throw std::runtime_error("medium items need the mask stage through 29");
    k_sieve.set_group_size(WG_SIZE);
    // Large-kernel work-group size; GB_LARGE_WG=N overrides (tuning only: the
    // kernel indexes by global id). N must be a non-zero multiple of the
    // SIMD32 subgroup: 0 would divide the launch by zero, anything else
    // splits a subgroup.
    large_wg = env::wg32("GB_LARGE_WG", LARGE_WG_SIZE);
    k_large.set_group_size(large_wg);
    k_ca->set_group_size(WG_SIZE);
    k_cb->set_group_size(WG_SIZE);
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
    // 9781 back-extension, i.e. segment_even / 2 + ~4891 ODD values.
    // The sieve writes WHOLE tiles (one work-group = TILE_WORDS words), so each
    // class array must be sized in whole tiles — otherwise the last tile
    // writes past the end of the buffer and hangs the GPU (device lost).
    // segment_even is an integer span, not an even-count. Account for the
    // one-even tail that is merged into the preceding segment below.
    const uint64_t max_segment_evens = segment_even / 2 + 1;
    const uint64_t max_odds   = max_segment_evens + (MAX_GB_PRIME + 1) / 2 + 2;
    if (max_odds >= uint64_t(std::numeric_limits<uint32_t>::max()))
        throw std::runtime_error("segment span too large for 32-bit large-prime offsets");
    const uint64_t dense_tiles = (max_odds + BITS_PER_TILE - 1) / BITS_PER_TILE;
    // Eight class arrays of tpc_max whole tiles each.
    const uint64_t tpc_max = (max_odds / 15 + 2 + BITS_PER_TILE - 1) / BITS_PER_TILE;
    const uint64_t map_tiles  = std::max(dense_tiles, ncls_map * tpc_max);
    map_words  = map_tiles * TILE_WORDS;
    // checked[] (tools only) would hold the 15 even classes at chk_stride30
    // words each; the kernels still take the stride.
    chk_stride30 = uint32_t((max_segment_evens / 15 + 1 + 63) / 64 + 1);
    // Evens phase A leaves to phase B: ~0.045% of a segment at 4e18 (106K
    // of 233.75M), ~0.25% on a sampled segment (586K). Sized at 1/32 of the
    // segment (12x the sampled peak); the host fails closed if a segment
    // ever overflows it. (1/8 reserved 234 MB of VRAM for ~1 MB of use.)
    pending_cap = uint32_t(std::min<uint64_t>(
        std::max<uint64_t>(max_segment_evens / 32, 1u << 16), UINT32_MAX / 2));

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
    // More than MAX_MASK_GROUPS_HOST groups (a GB_MASK_PERIOD sweep) fall back
    // to the kernel's per-prime mask loop.
    const bool mask_groups_on = mask_count > 0;
    // Max group period in words. 4096 -> 8 groups, 75 KB of tables (primes
    // 3..61): measured sieve_s per 2e10 window 0.1086 vs 0.1161 at 16384
    // (7 groups, 120 KB) and 0.1143 at 8192 — the tables must stay
    // cache-resident. Must yield <= MAX_MASK_GROUPS (8) groups, else the
    // per-prime path is used. GB_MASK_PERIOD overrides.
    mask_period = env::u32("GB_MASK_PERIOD", 4096);   // = small_prime_group_masks default (debug_map validates it)
    // 2026-11-18: the first MASK_XP_DEFAULT owned primes (67, 71, 73, 79) are folded
    // into the mod-30 class map's canonical mask groups, each into the group
    // of smallest period (tables 75 KB -> 1.9 MB, no extra loads): the owned
    // walk loses its densest primes. Bench +0.6% (long, long-end, 1e14);
    // 3 and 5 measured slightly less, a new group of their own worse.
    mask_xp = (mask_count > 0)
        ? std::min<uint32_t>(MASK_XP_DEFAULT, own_end - mask_count) : 0u;
    SmallPrimeGroups mask_groups0;
    if (mask_groups_on)
        mask_groups0 = small_prime_group_masks_canonical(   // sizes and descriptors
            sieve_primes, mask_count, 15u, mask_period, mask_xp);   // (canonical grouping)
    mask_groups_n = (mask_groups_on && mask_groups0.count() <= 8)
                                 ? mask_groups0.count() : 0u;
    mask_group_words = mask_groups0.words.size();
    d_group_desc = nullptr;
    if (mask_groups_n == 0) mask_xp = 0;   // per-prime mask path: the owned walk takes them
    if (mask_groups_n > 0) {
        // the medium items follow the descriptors at 2 * MAX_MASK_GROUPS words
        const size_t dw = std::max<size_t>(mask_groups0.desc.size(), 2 * MAX_MASK_GROUPS_HOST);
        d_group_desc = static_cast<uint32_t*>(ctx.alloc_shared((dw + med_items_host.size()) * 4));
        std::memset(d_group_desc, 0, dw * 4);
        std::memcpy(d_group_desc, mask_groups0.desc.data(), mask_groups0.desc.size() * 4);
        std::memcpy(d_group_desc + 2 * MAX_MASK_GROUPS_HOST, med_items_host.data(), med_items_host.size() * 4);
        med_nit = uint32_t(med_items_host.size() / 2);
    }
}

/// Phase 5: per-slot buffers (map, checked, tables, counters, ridge and
/// statistics buffers), the shared read-only tables (candidates, plans,
/// want_p) and the device base-prime list. Allocation only, plus the host
/// initialisation of shared buffers; the uploads come in upload_tables.
void DenseRun::allocate_slot_buffers() {
    for (int s = 0; s < 2; ++s) {
        b.prime_map[s]  = static_cast<uint64_t*>(ctx.alloc_device(map_words * 8));
        // (v2.6: the unused S1 / S5 split maps that were kept "for v1.12.0's
        // device-memory layout" are gone: 1e12 at 4e18, 6 ABBA rounds, 0.03%.)
        // per-class tables back to back
        const size_t ncls = ncls_map;
        b.first_bits[s] = static_cast<uint64_t*>(ctx.alloc_shared(ncls * sieve_split * 8));
        b.mask_table[s] = static_cast<uint64_t*>(ctx.alloc_shared(ncls * mask_count * 64 * 8));
        if (mask_groups_n > 0)
            b.group_masks[s] = static_cast<uint64_t*>(ctx.alloc_shared(ncls * mask_group_words * 8));
        b.group_y[s] = static_cast<uint32_t*>(ctx.alloc_shared(8uz * MAX_MASK_GROUPS_HOST * 4));
        if (mask_groups_n > 0) {   // canonical group tables: segment-independent
            const SmallPrimeGroups cg = small_prime_group_masks_canonical(
                sieve_primes, mask_count, 15u, mask_period, mask_xp);
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
        b.acct[s]      = static_cast<uint32_t*>(ctx.alloc_shared(8));
        // Ridge append buffer: u32 count in word 0 (low half) + RIDGE_CAP
        // (n, p) pairs; shared so the host merges it without a D2H copy.
        b.ridge[s]     = static_cast<uint64_t*>(ctx.alloc_shared((1 + 2 * size_t(RIDGE_CAP)) * 8));
        reset_slot_counters(s);
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
            b.g_hist[s]  = static_cast<uint32_t*>(ctx.alloc_shared((GAPS30_HBINS + GAPS30_PBINS) * 4));
            reset_gaps(s);
        }
        b.tile_first[s] = use_table
            ? static_cast<uint32_t*>(ctx.alloc_shared(size_t(table_bytes)))
            : nullptr;
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
    tail_t = stats_on ? gstats->tail_t() : 0xFFFFFFFFu;   // check30_b tail recording
    t_live0 = clock_::now();
    b.c30_cands = static_cast<uint16_t*>(ctx.alloc_device(std::max<size_t>(plan30.cands.size(), 1) * 2));
    b.c30_meta  = static_cast<uint32_t*>(ctx.alloc_device(plan30.meta.size() * 4));
    b.c30_rest  = static_cast<uint16_t*>(ctx.alloc_device(plan30.rest.size() * 2));
    b.c30_rtab  = static_cast<uint32_t*>(ctx.alloc_device(plan30.rtab.size() * 4));
    // The base-prime list is uploaded to DEVICE memory: the large kernel
    // streams it (~393 MB per segment) and device-local reads are fastest.
    // The payload comes straight from the mmap'd cache into one shared
    // staging buffer — the old fread pass is gone — and the DMA copy doubles
    // as the clock warm-up before the segment loop.
    b.sieve_primes = static_cast<uint32_t*>(ctx.alloc_device(size_t(sieve_count) * 4));
}

/// count30's gap outputs as the kernel expects them before a segment.
void DenseRun::reset_gaps(int s) {
    b.g_tail[s][0] = b.g_tail[s][1] = 0;
    b.x_cnt[s][11] = ~0u;                   // count30's first prime position (atomic_min)
    b.x_cnt[s][12] = 0;
    std::memset(b.g_hist[s], 0, (GAPS30_HBINS + GAPS30_PBINS) * 4);
}

/// Merges a completed slot's statistics buffers (after its check event).
void DenseRun::collect_stats(uint32_t sl, uint64_t seg_start_sl) {
    if (sampled_slot[sl]) slot_hist_n[sl] = gstats->add_full(b.x_hist[sl]);
    else gstats->add_tail(b.p_hist[sl]);
    gstats->add_first(seg_start_sl, b.p_first[sl]);
    if (gaps_on) {
        uint64_t wmax = 0;                  // count30's work-group maxima -> the segment's
        const bool w = walked_slot[sl];
        if (w)
            for (uint32_t g = 0; g < slot_ggroups[sl]; ++g) wmax = std::max(wmax, b.g_part[sl][g]);
        // count30's block walk: the internal gaps of the blocks of 30
        // (bins g / 2 after the histogram)
        uint32_t hx[GAPS30_HBINS];
        if (w) {
            std::memcpy(hx, b.g_hist[sl], sizeof(hx));
            const uint32_t dint = add_inner_gaps(hx, b.g_hist[sl] + GAPS30_HBINS);
            // count30's maximum covers the gaps into blocks only
            if (dint && (wmax >> 32) <= 2ull * dint && b.x_cnt[sl][11] != ~0u)
                wmax = walk_max_exact(slot_qfirst[sl], b.x_cnt[sl][11], b.x_cnt[sl][12]);
        }
        gstats->add_gap_segment(slot_qfirst[sl], slot_evens[sl], b.x_cnt[sl], b.g_tail[sl],
                                w ? &wmax : nullptr, w ? hx : nullptr);
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
}

/// Phase 6: which large primes leave the packed kernel (bucket ring or hit
/// lists), the ring size and bucket capacity, the hit-list batch / regions,
/// and the run's segment count for the harvest.
void DenseRun::size_bucket_ring_and_hitlists() {
    init_mark("buffers");
    // Primes p > q_odds_full hit a segment at most once; they leave the
    // per-segment packed kernel for the hit lists. Requires the uniform
    // dense geometry: every segment's q_first is seg_start - MAX_GB_PRIME and
    // consecutive q ranges advance by exactly delta = segment_even / 2 bits.
    // GB_BUCKET=0 keeps every large prime in the packed kernel (A/B).
    delta = segment_even / 2;
    q_odds_full = (segment_even - 5 + MAX_GB_PRIME) / 2 + 1;
    const SegmentGeometry g_first = segment_geometry(start_even, limit, segment_even, 0);
    q_first0 = g_first.q_first;
    const bool bucket_env_off = !env::onoff("GB_BUCKET", true);
    bucket_begin = sieve_count;   // first hit-listed prime index
    static_assert(WHEEL_W == 30030, "kernels/goldbach.cl only has the 30030 wheel tables");
    constexpr uint64_t wheel_max_step = 11;     // largest wheel-30030 step, in units of p
    constexpr double wheel_keep = 5760.0 / 15015;
    double bucket_lam = 0.0;   // expected hit-listed hits per segment
    bucket_on = !bucket_env_off && large_count > 0 &&
                     start_even >= MAX_GB_PRIME + 2 &&
                     g_first.seg_end != limit &&                  // >= 2 segments
                     g_first.q_odds == q_odds_full &&
                     q_first0 == start_even - MAX_GB_PRIME;
    // the mid-window pass runs (allocate_large_prime_buffers): its
    // preconditions on the run's geometry
    midwin_on = midwin_pre && q_first0 >= (uint64_t(1) << 60) &&
                (q_odds_full + 14) / 15 < (uint64_t(1) << 24);
    // mid_window walks primes below MIDW_COP_MAXP with cop_marks
    if (midwin_on && !mask_through_29())
        throw std::runtime_error("mid windows need the mask stage through 29");
    if (bucket_on) {
        // Hit-list only primes above max(q_odds_full, GB_BUCKET_MINP): below
        // ~9 x q_odds_full a prime's per-segment packed setup (~22 ps) is
        // cheaper than filing its ~q_odds/p entries (~200 ps each).
        const uint64_t minp_default = midwin_on
            ? q_odds_full * BUCKET_MIN_P_FACTOR_MIDWIN_NUM / BUCKET_MIN_P_FACTOR_MIDWIN_DEN
            : BUCKET_MIN_P_FACTOR * q_odds_full;
        // HL_MT (hl_harvest_mt files only hits with a prime multiplier) makes
        // a hit-listed prime cheap enough to go down to HL_MT_MINP_NUM / _DEN
        // x q_odds_full (multi-hit primes included; the harvest handles any
        // number of hits per segment), when the multiplier table is feasible.
        const uint64_t mt_minp = q_odds_full * HL_MT_MINP_NUM / HL_MT_MINP_DEN;
        uint64_t m_lo_t = 0, m_hi_t = 0;
        // the table limits (size, m < 2^38) raise the threshold where needed
        uint64_t mt_p = std::max<uint64_t>(env::u64("GB_BUCKET_MINP", mt_minp), 1);
        if (!mt_geometry(mt_p, m_lo_t, m_hi_t)) {
            uint64_t lo = mt_p, hi = std::max<uint64_t>(mt_p, minp_default);
            if (mt_geometry(hi, m_lo_t, m_hi_t)) {
                while (hi - lo > 1) {
                    const uint64_t mid = lo + (hi - lo) / 2;
                    (mt_geometry(mid, m_lo_t, m_hi_t) ? hi : lo) = mid;
                }
                mt_p = hi;
            }
        }
        // GB_HL_MT=0: the old hit-list path (hl_harvest, primes > q_odds_full
        // only), kept as a fallback
        mt_try = env::onoff("GB_HL_MT", true) && mt_geometry(mt_p, m_lo_t, m_hi_t);
        uint64_t minp = mt_try ? mt_p : env::u64("GB_BUCKET_MINP", minp_default);
        if (!mt_try)   // hl_harvest files at most one hit per prime and segment
            minp = std::max<uint64_t>(minp, q_odds_full);
        bucket_begin = uint32_t(std::upper_bound(sieve_primes + sieve_split,
                                                 sieve_primes + sieve_count,
                                                 uint32_t(std::min<uint64_t>(minp, UINT32_MAX))) -
                                sieve_primes);
        const uint64_t pmax = sieve_primes[sieve_count - 1];
        // Engagement range: the one the bucket ring had (its horizon, the
        // farthest next hit in segments, within 256), where hit lists were
        // measured and validated; small spans keep the packed kernel. The
        // 32-bit bound is the harvest's (re-checked below).
        const uint64_t x = q_odds_full - 1 + wheel_max_step * pmax;
        const uint64_t horizon = ((x < q_odds_full) ? 0 : (x - q_odds_full) / delta + 1) + 2;
        if (bucket_begin >= sieve_count || horizon > 256 ||
            pmax + q_odds_full >= (uint64_t(1) << 32))
            bucket_on = false;
    }
    if (bucket_on) {
        // Expected hits per segment: lambda = sum over hit-listed p of
        // q_odds_full / p, by Mertens' second theorem (sum_{p<=x} 1/p =
        // ln ln x + M + O(1/ln^2 x); ~1e-4 relative here, far inside the
        // slack). With the wheel only 5760 / 15015 of the odd multipliers hit.
        const double lo = double(sieve_primes[bucket_begin]);
        const double hi = double(sieve_primes[sieve_count - 1]);
        const double lambda = double(q_odds_full) *
            std::max(0.0, std::log(std::log(hi)) - std::log(std::log(lo))) + 1.0;
        bucket_lam = lambda * wheel_keep;
    }
    // Hit lists (kernels/goldbach.cl hl_harvest / hl_apply) for primes
    // [bucket_begin, sieve_count): every hl_K segments one harvest files all
    // hits of the next hl_K segments as u32 offsets; each segment applies its
    // list. The harvest walks in (segment, offset) form: needs delta + p_max
    // < 2^32 and q_odds_full < 2 delta. GB_HITLIST_K sets the batch.
    hitlist = bucket_on;
    hl_K = env::u32("GB_HITLIST_K", HL_K_DEFAULT, 1u, HL_MAX_K_HOST);
    // HL_MT (kernels/goldbach.cl hl_harvest_mt): the multipliers m of the
    // hit-listed primes' hits lie in [q_first0 / pmax, limit / pmin]; a hit
    // is filed only when m is prime. Needs every hit-listed p above the
    // cube root (m < p^2: composite m has a factor < p) and the table's
    // sieving primes (<= sqrt(m_hi)) on the device and below m_lo.
    hl_mt = false;
    if (hitlist && mt_try) {
        const uint64_t pmin = sieve_primes[bucket_begin];
        uint64_t m_lo = 0, m_hi = 0;
        if (!mt_geometry(pmin, m_lo, m_hi))
            throw std::runtime_error("hl_mt: multiplier table geometry");
        const uint64_t k_lo = m_lo / 30030 > 0 ? m_lo / 30030 - 1 : 0;
        const uint64_t k_hi = m_hi / 30030 + 2;
        const uint64_t s_max = uint64_t(std::sqrt(double(m_hi))) + 2;
        const uint32_t sb = uint32_t(std::lower_bound(sieve_primes, sieve_primes + sieve_count, 67u) -
                                     sieve_primes);
        const uint32_t se = uint32_t(std::upper_bound(sieve_primes, sieve_primes + sieve_count,
                                                      uint32_t(std::min<uint64_t>(s_max, UINT32_MAX))) -
                                     sieve_primes);
        if (se < sieve_count && sieve_primes[se] > s_max - 2 && s_max < k_lo * 30030) {
            hl_mt = true;
            mt_k_lo = uint32_t(k_lo);
            mt_blocks = uint32_t(k_hi - k_lo);
            mt_s_begin = sb;
            mt_s_end = se;
        }
        else
            throw std::runtime_error("hl_mt: multiplier table sieving primes");
        if (verbose)
            std::fprintf(stderr, "hl_mt: %s m [%llu, %llu] blocks %llu sieve primes [%u, %u)\n",
                         hl_mt ? "on" : "off", (unsigned long long)m_lo, (unsigned long long)m_hi,
                         (unsigned long long)(k_hi - k_lo), sb, se);
    }
    if (hl_mt) {
        // filed hits per segment: prime p files the prime m among its
        // q_odds / p odd multipliers, density 2 / ln m, m ~ q / p:
        // lam = int_lo^hi 2 q_odds / (p ln p ln(q_first0 / p)) dp (prime
        // density 1 / ln p), by the midpoint rule in ln p. Measured at 4e18:
        // 4171752 filed per segment vs 4171805 (the old 1 / ln m_lo
        // estimate: 4796437). q_first0 is the run's smallest q (highest
        // density); the region capacity adds 10 % + 10 sigma on top.
        const double a = std::log(double(sieve_primes[bucket_begin]));
        const double bnd = std::log(double(sieve_primes[sieve_count - 1]));
        const double lq = std::log(double(q_first0));
        constexpr int N = 4096;
        const double h = (bnd - a) / N;
        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            const double u = a + (i + 0.5) * h;
            sum += 2.0 * double(q_odds_full) / (u * (lq - u)) * h;
        }
        bucket_lam = sum + 1.0;
    }
    // A segment's list is one region of hl_capg entries per harvest
    // work-group (hl_G groups, each harvesting a fixed share of the primes:
    // mean m = lam / hl_G hits per region). Capacity m * 1.10 + 10 sqrt(m) +
    // 64 (a region's count is a sum of independent hit indicators, variance
    // <= mean), rounded to a power of two; overflow fails the run closed.
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
    // and within the list budget (GB_HL_LIST_MAX bytes)
    {
        const uint64_t budget = env::u64("GB_HL_LIST_MAX", HL_LIST_MAX_DEFAULT);
        while (hl_K > 1 && uint64_t(hl_K) * hl_G * hl_capg * 4 > budget) --hl_K;
    }
    // K = 1 is the floor of both loops: fail closed rather than let a list
    // offset wrap (never near: K = 1 is ~30 MB at 4e18)
    if (hitlist && uint64_t(hl_K) * hl_G * hl_capg * 4 >= (uint64_t(1) << 32))
        throw std::runtime_error("hit lists: one batch's lists exceed 4 GiB");
    // Segments of the run (the loop below: seg_start += segment_even until
    // seg_end == limit) and the last one's q_odds (the only short one).
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
    packed_end = bucket_on ? bucket_begin : sieve_count;
    if (verbose && hitlist)
        std::fprintf(stderr, "hit lists: primes >= %u, lam %.0f, K %u, groups %u x %u entries\n",
                     sieve_primes[bucket_begin], bucket_lam, hl_K, hl_G, hl_capg);
    init_mark("bucket-geom+lambda");
}

/// Phase 7: the packed large-prime lists (host packing + device buffers),
/// the bucket ring or hit-list buffers and their kernels, and the
/// GB_BUCKET_VERIFY reference path (full packed list, scratch map, kernels).
void DenseRun::allocate_large_prime_buffers() {
    // Mid-prime SLM windows (§9.2x): [sieve_split, mid_end) leave the packed
    // kernel; the packed list starts at mid_end.
    mid_end = sieve_split;
    // <= 128 KB = 2^20 bits: ceil_div_f's spans stay <= 2^21 (rule 2b)
    midw_bits = env::u32("GB_MIDWIN_KB", MIDWIN_KB, 1, 128) * 8192u;
    static_assert(128u * 8192u <= (1u << 21) && BITS_PER_TILE <= (1u << 21),
                  "ceil_div_f is exact for spans <= 2^21 only");
    midw_wg = env::wg32("GB_MIDWIN_WG", MIDWIN_WG);
    if (midwin_pre) {
        if (midwin_on)
            mid_end = uint32_t(std::lower_bound(sieve_primes + sieve_split, sieve_primes + packed_end,
                                                uint32_t(midwin_p2)) - sieve_primes);
        else
            std::fprintf(stderr, "note: mid-window pass off (needs the mod-30 class map, "
                                 "q >= 2^60 and class arrays < 2^24 bits)\n");
    }
    mid_n = mid_end - sieve_split;
    if (mid_n > 0) {
        k_mids = std::make_unique<Kernel>(mod, "mid_setup");
        k_midw = std::make_unique<Kernel>(mod, "mid_window");
        k_mids->set_group_size(WG_SIZE);
        k_midw->set_group_size(midw_wg);
        for (int s = 0; s < 2; ++s)
            d_mid_i0[s] = static_cast<uint32_t*>(ctx.alloc_device(size_t(mid_n) * 16 * 4));  // i0 + residues
        if (verbose)
            std::fprintf(stderr, "mid windows: primes [%u, %u], %u primes, %u-bit windows, wg %u\n",
                         sieve_primes[sieve_split], sieve_primes[mid_end - 1], mid_n,
                         midw_bits, midw_wg);
    }
    packed_count = packed_end - mid_end;
    // hl_apply fused into the packed kernel (sieve_large_packed_hl): the two
    // mark sets commute, and alone hl_apply was latency-bound. hl_apply runs
    // as its own launch only when no prime is packed.
    hl_fuse = hitlist && packed_count > 0;
    d_large_bases = nullptr;
    d_large_gaps = nullptr;
    if (packed_count > 0) {
        packed = pack_large_primes(sieve_primes, mid_end, packed_end);
        init_mark("  pack (host)");
        d_large_bases = static_cast<uint32_t*>(ctx.alloc_device(packed.bases.size() * 4));
        d_large_gaps  = static_cast<uint32_t*>(ctx.alloc_device(packed.gaps.size()));
    }
    init_mark("  packed alloc");
    bkt_err = nullptr;
    d_hl_lists = nullptr;
    d_hl_counts = nullptr;
    if (hitlist) {
        // compressed (driver default): uncompressed lists measured +17 %
        d_hl_lists  = static_cast<uint32_t*>(ctx.alloc_device(size_t(hl_K) * hl_G * hl_capg * 4));
        d_hl_counts = static_cast<uint32_t*>(ctx.alloc_device(size_t(HL_MAX_K_HOST) * hl_G * 4));
    }
    if (hitlist) {
        bkt_err = static_cast<uint32_t*>(ctx.alloc_shared(4));
        *bkt_err = 0;
    }
    init_mark("  err alloc");
    // grid-stride work-groups of a separate hl_apply (when nothing is packed)
    bucket_groups = bucket_groups_env ? bucket_groups_env : 2 * BUCKET_GROUPS;
    if (hitlist) {
        k_hlh = std::make_unique<Kernel>(mod, hl_mt ? "hl_harvest_mt" : "hl_harvest");
        k_hla = std::make_unique<Kernel>(mod, hl_fuse ? "sieve_large_packed_hl" : "hl_apply");
        k_hlh->set_group_size(HL_HARVEST_WG);
        k_hla->set_group_size(hl_fuse ? large_wg : WG_SIZE);
    }
    init_mark("  kernel create");
    // GB_BUCKET_VERIFY=1: rebuild every segment's map WITHOUT hit lists (tile
    // + packed kernel over ALL large primes) into a scratch map and compare it
    // word for word with the production map on the GPU. Validation mode.
    // GB_BUCKET_VERIFY=N checks every Nth segment (1 = all; e.g. 100 costs
    // ~1% and is a cheap production safety net: an end-to-end PASS cannot
    // reveal a lost hit-list mark — other candidates still resolve the evens).
    bucket_verify_every = env::present("GB_BUCKET_VERIFY")
        ? env::u32("GB_BUCKET_VERIFY", 0u, 1u) : 0u;
    bucket_verify = bucket_on && bucket_verify_every > 0;
    d_full_bases = nullptr;
    d_full_gaps = nullptr;
    d_verify_map = nullptr;
    verify_diff = nullptr;
    if (bucket_verify) {
        // The reference rebuilds the map without hit lists: tile, mid windows
        // (when on) and the packed kernel over every larger prime. It checks
        // the hit-list path (harvest, multiplier table, lists, apply) against
        // the plain packed marks and re-runs every stage. (Until v2.8 the mid
        // primes went through the packed kernel too: global atomics at ~29 ms
        // per verified segment, ~3.5 % of production at GB_BUCKET_VERIFY=500.)
        // GB_BUCKET_VERIFY_FULL: the old, slower reference (mid primes in the
        // packed kernel too: also checks mid_window against other code).
        verify_mid = mid_n > 0 && !env::present("GB_BUCKET_VERIFY_FULL");
        verify_lo = verify_mid ? mid_end : sieve_split;
        packed_full = pack_large_primes(sieve_primes, verify_lo, sieve_count);
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
}

/// Phase 8: the tile kernel's reciprocal tables (magic / shift / ms32) and
/// every read-only upload: staged copies of the candidate lists, plans and
/// packed lists, direct DMA of the base-prime ranges the kernels read, one
/// completion sync, then the staging buffers are freed.
void DenseRun::upload_tables() {
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
    std::tie(magic, shift) = build_magic_reciprocals(sieve_primes, sieve_split);

    // 32-bit reciprocals for the tile table path (host_math.hpp); all zero
    // when the table path is off (the kernel does not read them then).
    std::vector<uint64_t> ms32v(sieve_split, 0ull);
    if (use_table) ms32v = build_tile_reciprocals32(sieve_primes, sieve_split);

    // Upload the read-only tables once; never touched again per segment.
    // The DMA copies also double as the clock warm-up before the segment loop.
    init_mark("magic+ms32");
    t_upload = clock_::now();
    {
        void* stage_c30[4] = {};
        if (!plan30.cands.empty())
            stage_c30[0] = stage_upload(ctx, plan30.cands.data(), plan30.cands.size() * 2, b.c30_cands);
        stage_c30[1] = stage_upload(ctx, plan30.meta.data(), plan30.meta.size() * 4, b.c30_meta);
        stage_c30[2] = stage_upload(ctx, plan30.rest.data(), plan30.rest.size() * 2, b.c30_rest);
        stage_c30[3] = stage_upload(ctx, plan30.rtab.data(), plan30.rtab.size() * 4, b.c30_rtab);
        void* stage_x[4] = {};
        if (stats_on) {
            stage_x[3] = stage_upload(ctx, splan.xj.data(), splan.xj.size() * 4, b.x_j);
            stage_x[2] = stage_upload(ctx, splan.xch.data(), splan.xch.size() * 4, b.x_ch);
            stage_x[0] = stage_upload(ctx, splan.xall.data(), splan.xall.size() * 2, b.x_all);
            stage_x[1] = stage_upload(ctx, splan.xtab.data(), splan.xtab.size() * 4, b.x_tab);
        }
        // The base-prime list on the device is read only by the tile kernel
        // ([0, split)), the mid windows ([split, mid_end)), hl_harvest
        // ([bucket_begin, count)) and the verify kernels (everything). Copy
        // only the ranges that are read —
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
            copy_primes(0, std::max(mid_end, hl_mt ? mt_s_end : 0u));   // mid_end >= sieve_split
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
    if (hl_mt) build_mtab();
}

/// HL_MT: builds the multiplier primality table on the device (mt_build)
/// and, with GB_MT_CHECK=N, checks N random table bits on the CPU.
void DenseRun::build_mtab() {
    Kernel kb(mod, "mt_build");
    kb.set_group_size(1024);
    const size_t words = size_t(mt_blocks) * 180;
    // uncompressed (2026-11-18 r7): the harvest's scattered word reads skip
    // the compression metadata, harvest -3.4 % (the lists themselves stay
    // compressed: uncompressed they are +17 %)
    d_mtab = static_cast<uint32_t*>(ctx.alloc_device(words * 4, 64, true));
    kb.set_arg(0, sizeof(d_mtab), &d_mtab);
    kb.set_arg(1, sizeof(mt_blocks), &mt_blocks);
    kb.set_arg(2, sizeof(mt_k_lo), &mt_k_lo);
    kb.set_arg(3, sizeof(b.sieve_primes), &b.sieve_primes);
    kb.set_arg(4, sizeof(mt_s_begin), &mt_s_begin);
    kb.set_arg(5, sizeof(mt_s_end), &mt_s_end);
    const uint32_t n_small = uint32_t(std::lower_bound(sieve_primes + mt_s_begin, sieve_primes + mt_s_end,
                                                       1024u) - (sieve_primes + mt_s_begin));
    kb.set_arg(6, sizeof(n_small), &n_small);
    // presieve patterns for 17..61 (kernels/goldbach.cl mt_build): prime s,
    // row k mod s, 180 words: bit w set iff s does not divide 30030 k + M[w]
    std::vector<uint32_t> M;
    for (uint32_t m = 1; m < 30030; m += 2)
        if (std::gcd(m, 30030u) == 1) M.push_back(m);
    std::vector<uint32_t> pat;
    for (const uint32_t s : {17u, 19u, 23u, 29u, 31u, 37u, 41u, 43u, 47u, 53u, 59u, 61u})
        for (uint32_t k = 0; k < s; ++k)
            for (uint32_t wd = 0; wd < 180; ++wd) {
                uint32_t x = 0;
                for (uint32_t bt = 0; bt < 32; ++bt)
                    if ((30030u * k + M[wd * 32 + bt]) % s != 0) x |= 1u << bt;
                pat.push_back(x);
            }
    uint32_t* d_pat = static_cast<uint32_t*>(ctx.alloc_device(pat.size() * 4));
    ZE_CHECK(zeCommandListAppendMemoryCopy(ctx.list(), d_pat, pat.data(), pat.size() * 4, nullptr, 0, nullptr));
    kb.set_arg(7, sizeof(d_pat), &d_pat);
    const auto t0 = clock_::now();
    kb.launch(ctx.list(), (mt_blocks + MT_CB_HOST - 1) / MT_CB_HOST);   // MT_CB blocks per group
    ctx.host_sync();
    Context::free_mem(ctx.context(), d_pat);
    if (verbose)
        std::fprintf(stderr, "hl_mt: table %.1f MB built in %.1f ms\n", double(words) * 4 / 1e6,
                     std::chrono::duration<double, std::milli>(clock_::now() - t0).count());
    const uint32_t nchk = env::u32("GB_MT_CHECK", 0, 0, 100000000u);
    if (nchk > 0) {
        uint32_t* h = static_cast<uint32_t*>(ctx.alloc_shared(words * 4));
        ZE_CHECK(zeCommandListAppendMemoryCopy(ctx.list(), h, d_mtab, words * 4, nullptr, 0, nullptr));
        ctx.host_sync();
        uint64_t x = 0x9E3779B97F4A7C15ull, bad = 0, ones = 0;
        for (uint32_t i = 0; i < nchk; ++i) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            const uint64_t bit = (i < 2 * 5760) ? i : x % (uint64_t(words) * 32);
            const uint64_t m = (mt_k_lo + bit / 5760) * 30030ull + M[bit % 5760];
            const bool t = (h[bit >> 5] >> (bit & 31)) & 1u;
            ones += t;
            if (t != is_prime_u64(m)) {
                if (bad < 10) std::fprintf(stderr, "hl_mt: table bit %llu (m = %llu) = %d, wrong\n",
                                           (unsigned long long)bit, (unsigned long long)m, int(t));
                ++bad;
            }
        }
        Context::free_mem(ctx.context(), h);
        std::fprintf(stderr, "hl_mt: table check %u bits, %llu prime, %llu wrong\n", nchk,
                     (unsigned long long)ones, (unsigned long long)bad);
        if (bad) throw std::runtime_error("hl_mt: multiplier table wrong");
    }
}

/// Phase 9: re-points each slot's seven statistics buffers into one shared
/// block and frees the original allocations.
void DenseRun::pack_stats_buffers() {
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
    // (Until v2.6 the seven original allocations stayed allocated, unused, so
    // "nothing that exists today moves" in device memory; the same layout
    // belief kept the unused S1 / S5 maps, which measured 0.03% (noise).)
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
        const size_t at_g_hist  = take(size_t(GAPS30_HBINS + GAPS30_PBINS) * 4);
        for (int s = 0; s < 2; ++s) {
            unsigned char* base = static_cast<unsigned char*>(ctx.alloc_shared(block_bytes));
            stats_block[s] = base;
            std::memset(base, 0, block_bytes);
            for (void* p : {static_cast<void*>(b.p_hist[s]), static_cast<void*>(b.p_first[s]),
                            static_cast<void*>(b.x_lim[s]), static_cast<void*>(b.x_cnt[s]),
                            static_cast<void*>(b.g_part[s]), static_cast<void*>(b.g_tail[s]),
                            static_cast<void*>(b.g_hist[s])})
                Context::free_mem(ctx.context(), p);
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
}

/// Phase 10: the per-slot timestamp events; closes the init timing.
void DenseRun::create_events() {
    gb_count = uint32_t(gb_primes.size());
    // The mask-table kernel arg is set per segment (the table CONTENT is
    // q_first-dependent and rewritten into the slot's shared buffer each
    // iteration, alongside first_bits).

    // Events for per-kernel GPU timing and GPU-side dependencies.
    // Per pipeline slot: sieve signals after the tile-sieve launch, large
    // after the large-prime launch (the check kernel waits on BOTH GPU-side),
    // check signals after the check launch (the host waits on it once per
    // segment). One pool; every event is reset on its own.
    init_mark("upload");
    ev.create(ctx.context());
    // nothing computed from the base primes is trusted before their cache passed
    if (const char* why = cache_check.get())
        throw std::runtime_error(std::string("base-prime cache damaged (") + why + "): moved aside, rerun");
    init_mark("cache check (joined)");

    stats.init_seconds = seconds_since(t_start);
    stats.init_cache_s  = cache_seconds;
    stats.init_module_s = module_seconds;
    stats.init_upload_s = seconds_since(t_upload);
}

/// Phase 11: the segment loop's run-wide state (timers, sums, carry-over of
/// the previous segment, device timer resolution, GB_PHASE).
void DenseRun::begin_segment_loop() {
    // ------------------------------------------------------ Segment loop --
    t_gpu = clock_::now();
    total_leftovers = 0;
    prev_seg_start = 0;  // first even n of the previously enqueued segment
    prev_seg_evens = 0;  // its even count (leftover index bound)
    // Carried from the last ENQUEUED segment to the iteration that collects
    // it (one slot behind): host wall time inside that iteration and its
    // host offset-generation share. Stored here because collection happens
    // one pipeline slot after enqueue.

    // Device timer resolution (1.2 stype; base struct returns garbage).
    ze_device_properties_t dev_props{};
    dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
    zeDeviceGetProperties(ctx.device(), &dev_props);
    timer_hz = double(dev_props.timerResolution);

    // Per-segment record for the segment currently being enqueued, plus the
    // completed record of the segment enqueued one iteration earlier (its
    // GPU timestamps become readable only when the NEXT iteration collects
    // it, by which time seg_stat has been reused for the newer segment).
    if (collect_segments) stats.per_segment.reserve(64);

    // Env-gated host-phase trace (GB_PHASE=1): per-segment wall time split
    // into prep (first-bits / table / masks), enqueue (args + resets +
    // launches), wait (ev_check host sync) and collect (timestamp queries +
    // leftover read). Printed to stderr; overhead is a few clock reads per
    // segment and the gate defaults to off.
    phase_trace = env::present("GB_PHASE");
    fb_host.resize(size_t(ncls_map) * sieve_split);   // per-segment first bits
}

/// Query the per-kernel GPU timestamps of the given slot: the sieve, large
/// and check kernels SET ds / dl / dv, the extra kernels of a phase are
/// ADDED to them. Returns true when one of the three yielded a valid
/// timestamp. A kernel is skipped when the device clock reports
/// end <= start (kernel_seconds: anomalies seen on some driver builds).
/// With GB_PHASE=1, also log raw start/end ticks per segment so kernel
/// overlap (or its absence) can be verified directly.
bool DenseRun::query_slot(uint32_t slot, double& ds, double& dl, double& dv) {
    bool any = false;
    if (const auto t = kernel_seconds(ev.sieve[slot], timer_hz)) { ds = *t; any = true; }
    if (const auto t = kernel_seconds(ev.large[slot], timer_hz)) { dl = *t; any = true; }
    if (const auto t = kernel_seconds(ev.check[slot], timer_hz)) { dv = *t; any = true; }
    if (mid_n > 0)       // sieve_s also counts the mid-window pass
        if (const auto t = kernel_seconds(ev.mw[slot], timer_hz)) {
            ds += *t;
            midw_seconds += *t;
            if (phase_trace) std::fprintf(stderr, "    mid_window %.3f ms\n", *t * 1e3);
        }
    if (hl_used[slot])   // large_s also counts the hit-list harvest
        if (const auto t = kernel_seconds(ev.hl[slot], timer_hz)) {
            dl += *t;
            if (phase_trace) std::fprintf(stderr, "    hl_harvest %.3f ms\n", *t * 1e3);
        }
    if (hitlist && !hl_fuse)   // large_s = packed kernel + a separate hl_apply
        if (const auto t = kernel_seconds(ev.bucket[slot], timer_hz)) {
            dl += *t;
            if (phase_trace)
                std::fprintf(stderr, "    hl_apply %.3f ms (large total %.3f ms)\n",
                             *t * 1e3, dl * 1e3);
        }
    // check_s = phase A + phase B (+ count30)
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
            if (phase_trace) std::fprintf(stderr, "    count30 %.3f ms\n", *t * 1e3);
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
}

/// Per-segment step 1: geometry of the segment at sg.seg_start, the class-map
/// kernel arguments derived from it, and its telemetry record (seg_stat).
void DenseRun::begin_segment(Segment& sg) {
    const SegmentGeometry g = segment_geometry(sg.seg_start, limit, segment_even,
                                               stats.segments);
    sg.seg_end   = g.seg_end;
    sg.seg_evens = g.seg_evens;
    sg.q_first   = g.q_first;
    sg.q_last    = g.q_last;
    sg.q_odds    = g.q_odds;
    sg.tiles     = g.tiles;
    sg.slot      = g.slot;
    // The map is eight class arrays of cgeo.tpc tiles each.
    sg.cgeo = class_geometry(sg.q_first, sg.q_odds, ncls_map);
    sg.tiles_launch = uint64_t(sg.cgeo.K) * sg.cgeo.tpc;
    sg.c5w = sg.cgeo.tpc * TILE_WORDS;                 // class stride (u64 words)
    sg.off5 = sg.c5w * 64u;                            // class stride (bits)
    // phis: the classes' dense phases (nibbles); ga1 / ga2 (the kernels'
    // phi1 / phi5): the dense phase (mod 15) -> class nibble table (15 = q
    // divisible by 3 or 5: not stored)
    sg.phis = 0;
    uint64_t ph2k = ~0ull;
    for (uint32_t c = 0; c < 8; ++c) {
        sg.phis |= sg.cgeo.phi[c] << (4 * c);
        ph2k &= ~(0xFull << (4 * sg.cgeo.phi[c]));
        ph2k |= uint64_t(c) << (4 * sg.cgeo.phi[c]);
    }
    sg.ga1 = uint32_t(ph2k);
    sg.ga2 = uint32_t(ph2k >> 32);
    sg.mask_coff = 0u;                                 // one canonical mask set

    // Per-segment telemetry: start the wall clock. The record itself is
    // filled after mask_count_seg/tiles are computed below, and finalized
    // when this segment's GPU work is collected (next iteration or drain).
    sg.t_seg_wall = clock_::now();
    sg.offset_before = offset_seconds;

    // The small-prime mask table is q_first-dependent; the kernel's stage
    // 3.5 (restore q = p) is range-sensitive and skipped when q_first >
    // MASK_MAX_P, so masks stay valid at every range (disabling them at
    // high ranges forced Xe2 into thousands of scalar SLM RMWs per lane
    // for p=3..61 instead of 18 x 32 whole-word ANDs).
    sg.mask_count_seg = mask_count;

    // Per-segment telemetry record (collected after the NEXT iteration's
    // sync, once this segment's kernel timestamps are readable).
    seg_stat = SegmentStats{};
    seg_stat.seg_start   = sg.seg_start;
    seg_stat.seg_end     = sg.seg_end;
    seg_stat.evens       = sg.seg_evens;
    seg_stat.sieve_split = sieve_split;
    seg_stat.mask_count  = sg.mask_count_seg;
    seg_stat.tiles       = sg.tiles_launch;
    // work-items of the packed large-prime kernel as launched below
    // (LARGE_PACK_PPI primes each; the bucketed / hit-listed primes are
    // not work-items of it). The legacy LARGE_PRIMES_PER_ITEM count over
    // every large prime reported ~4x the real launch.
    seg_stat.large_items = (uint64_t(packed_count) + LARGE_PACK_PPI - 1) / LARGE_PACK_PPI;
    sum_tiles += sg.tiles_launch;
    sum_large_items += seg_stat.large_items;
}

/// Per-segment step 2 (host prep, overlapping the GPU's previous segment):
/// first bits, the per-tile first-mark table and the small-prime masks,
/// written into this slot's shared buffers.
void DenseRun::prepare_segment_tables(Segment& sg) {
    // Host prep for this segment (first-bits, tile table, masks).
    //
    // PIPELINE NOTE: this host work runs while the GPU is still executing
    // the PREVIOUS segment (other slot). It is safe to write the slot's
    // shared buffers here because that slot was last read by the segment
    // two iterations ago, whose check kernel completed before the host
    // collected it (host-synced on ev.check of that slot).
    const auto t_prep = clock_::now();
    sg.table_before = table_seconds;
    {
        // first_bits is a shared (host-mapped) allocation: write it
        // directly, no copy command or sync needed. The kernel launch is
        // queued after this host write completes.
        // The first bits are computed into a persistent host buffer (a
        // fresh 2.3 MB vector per segment paid its page faults every time)
        // by parallel workers over disjoint prime ranges, each copying its
        // own slices: 1.65 -> ~0.3 ms of per-segment host prep, which had
        // become the pipeline's limit once the GPU segment time dropped below
        // the host loop's.
        parallel_for(0, sieve_split, 6, [&](uint32_t j0, uint32_t j1) {
            first_bits_class_range(fb_host.data(), sieve_primes, sieve_split, j0, j1,
                                   sg.q_first, sg.q_last, sg.cgeo, med_cop ? own_end : UINT32_MAX);
            for (uint32_t c = 0; c < sg.cgeo.K; ++c)
                std::memcpy(b.first_bits[sg.slot] + size_t(c) * sieve_split + j0,
                            fb_host.data() + size_t(c) * sieve_split + j0, size_t(j1 - j0) * 8);
        });
        const std::vector<uint64_t>& fb = fb_host;

        if (use_table) {
            const auto t_table = clock_::now();
            build_tile_first_table(b.tile_first[sg.slot], uint32_t(sg.tiles_launch),
                                   sieve_split, own_end, fb.data(), magic.data(),
                                   shift.data(), sieve_primes, sg.cgeo.tpc);
            table_seconds += seconds_since(t_table);
        }

        // Large-prime first bits are computed inline by the packed kernel
        // (tools/debug_fb.cpp validates that formula bit for bit).

        // Small-prime mask table for THIS segment (q_first-dependent).
        // Same shared-buffer pipeline-safety argument as first_bits:
        // this slot was last read two segments ago, already synced.
        for (uint32_t c = 0; c < sg.cgeo.K; ++c) {
            if (mask_groups_n > 0) {
                // canonical tables (uploaded once): per class only the
                // word offset of each group (primes.cpp)
                const std::vector<uint32_t> ys = small_prime_group_offsets(
                    sieve_primes, sg.mask_count_seg, sg.cgeo.Q[c], sg.cgeo.qmod, mask_period, mask_xp);
                if (ys.size() != mask_groups_n)
                    throw std::runtime_error("class masks: group layout mismatch");
                std::memcpy(b.group_y[sg.slot] + c * MAX_MASK_GROUPS_HOST, ys.data(), ys.size() * 4);
                continue;
            }
            const auto mt = small_prime_mask_table_step(sieve_primes, sg.mask_count_seg, sg.cgeo.Q[c], sg.cgeo.qmod);
            if (sg.mask_count_seg > 0) {
                std::memcpy(b.mask_table[sg.slot] + size_t(c) * sg.mask_count_seg * 64, mt.data(),
                            mt.size() * 8);
            }
        }
    }
    sg.prep_s = seconds_since(t_prep);
}

/// Per-segment step 3: make_prime_map's arguments, the slot's counter
/// fills and event resets, and the tile-sieve launch (all tiles, one launch).
void DenseRun::enqueue_sieve(Segment& sg) {
    sg.t_enq = clock_::now();

    // --- make_prime_map args (single launch covers ALL tiles)
    // Kernel arg order: 0=map 1=primes 2=fb_global 3=magic 4=shifts
    //                   5=count 6=q_odds 7=SLM 8=mask_count 9=mask_table
    //                   10=q_first 11=sieve_split
    const auto t_args0 = clock_::now();
    k_sieve.set_arg(0, sizeof(b.prime_map[sg.slot]),    &b.prime_map[sg.slot]);
    k_sieve.set_arg(1, sizeof(b.sieve_primes), &b.sieve_primes);
    k_sieve.set_arg(2, sizeof(b.first_bits[sg.slot]),   &b.first_bits[sg.slot]);
    k_sieve.set_arg(3, sizeof(b.magic),        &b.magic);
    k_sieve.set_arg(4, sizeof(b.shifts),       &b.shifts);
    k_sieve.set_arg(5, sizeof(sieve_split),    &sieve_split);
    k_sieve.set_arg(6, sizeof(sg.q_odds),         &sg.q_odds);
    k_sieve.set_local_arg(7, size_t(TILE_WORDS) * 8); // 16 KB SLM
    k_sieve.set_arg(8, sizeof(sg.mask_count_seg), &sg.mask_count_seg);
    k_sieve.set_arg(9, sizeof(b.mask_table[sg.slot]), &b.mask_table[sg.slot]);
    k_sieve.set_arg(10, sizeof(sg.q_first),       &sg.q_first);
    k_sieve.set_arg(11, sizeof(sieve_split),   &sieve_split);
    k_sieve.set_arg(12, sizeof(b.tile_first[sg.slot]), &b.tile_first[sg.slot]);
    k_sieve.set_arg(13, sizeof(use_table_u32), &use_table_u32);
    k_sieve.set_arg(14, sizeof(b.ms32),        &b.ms32);
    k_sieve.set_arg(15, sizeof(own_end),       &own_end);
    k_sieve.set_arg(16, sizeof(b.group_masks[sg.slot]), &b.group_masks[sg.slot]);
    k_sieve.set_arg(17, sizeof(d_group_desc),  &d_group_desc);
    const uint32_t groups_xp = mask_groups_n | mask_xp << 8 | med_nit << 16;   // make_prime_map: n_groups_xp
    k_sieve.set_arg(18, sizeof(groups_xp), &groups_xp);
    k_sieve.set_arg(19, 4, &cm_u32);
    k_sieve.set_arg(20, 4, &sg.cgeo.tpc);
    k_sieve.set_arg(21, 4, &sg.phis);
    k_sieve.set_arg(22, 8, &sg.cgeo.n[0]);
    k_sieve.set_arg(23, 8, &sg.cgeo.n[1]);
    k_sieve.set_arg(24, 4, &sg.mask_coff);
    {
        uint32_t* gy = mask_groups_n > 0 ? b.group_y[sg.slot] : nullptr;
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
    // leftover_n / pending_n / acct / ridge's count are zeroed by the host
    // after each collect (reset_slot_counters) instead of three 1-byte-
    // pattern MemoryFills per segment (builtin fill kernels; long6 +0.14%).
    // Under unitrace those fills appear to stall until count30 ends (67 us
    // average): an instrumentation artefact, not a bubble of the real run.
    const uint32_t zero = 0;
    // statistics: p_hist / p_first / x_cnt are reset by the host after
    // each collect (collect_stats); x_hist is used by sampled segments only
    if (stats_on && stats.segments % stats_sample == 0)
        ZE_CHECK(zeCommandListAppendMemoryFill(ctx.list(), b.x_hist[sg.slot], &zero, 4,
                                               splan.xall.size() * 4, nullptr, 0, nullptr));
    ZE_CHECK(zeEventHostReset(ev.sieve[sg.slot]));
    ZE_CHECK(zeEventHostReset(ev.large[sg.slot]));
    ZE_CHECK(zeEventHostReset(ev.check[sg.slot]));
    ZE_CHECK(zeEventHostReset(ev.checka[sg.slot]));
    ZE_CHECK(zeEventHostReset(ev.bucket[sg.slot]));
    ZE_CHECK(zeEventHostReset(ev.hl[sg.slot]));
    ZE_CHECK(zeEventHostReset(ev.mw[sg.slot]));
    ZE_CHECK(zeEventHostReset(ev.splitk[sg.slot]));
    for (int v = 0; v < 3; ++v) ZE_CHECK(zeEventHostReset(ev.xv[v][sg.slot]));
    sg.args_s = seconds_since(t_args0);

    sg.t_launch0 = clock_::now();
    // launch_overlap: may overlap segment k-1's checker (this slot's map,
    // tables and counters were last used by segment k-2, host-synchronized)
    k_sieve.launch_overlap(ctx.list(), uint32_t(sg.tiles_launch), ev.sieve[sg.slot]); // all tiles, one launch
    if (mid_n > 0) {
        // mid_setup reads only the primes and q_first; its table slot was
        // last read by segment k-2's mid_window (host-synchronized)
        k_mids->set_arg(0, sizeof(b.sieve_primes), &b.sieve_primes);
        k_mids->set_arg(1, 4, &sieve_split);
        k_mids->set_arg(2, 4, &mid_n);
        k_mids->set_arg(3, 8, &sg.q_first);
        k_mids->set_arg(4, 4, &sg.phis);
        k_mids->set_arg(5, sizeof(d_mid_i0[sg.slot]), &d_mid_i0[sg.slot]);
        k_mids->launch_overlap(ctx.list(), (mid_n + WG_SIZE - 1) / WG_SIZE);
    }
    sg.l1_s = seconds_since(sg.t_launch0);
}

/// Per-segment step 4: the large primes after the tile sieve: hit lists, the
/// packed large-prime kernel, and the GB_BUCKET_VERIFY reference map.
void DenseRun::enqueue_large_primes(Segment& sg) {
    // The large-prime marks are atomic ANDs and must never overlap the tile
    // kernel's whole-word map stores (a store landing after an atomic clear
    // would restore a composite as "prime"): a barrier precedes each mark
    // kernel (enqueue_packed_large, hl_apply). Hit lists and packed primes
    // only AND-clear map bits, so their relative order does not matter.
    uint32_t hl_jb = 0;   // this segment's list (hit lists)
    if (hitlist) enqueue_hitlists(sg, hl_jb);
    if (mid_n > 0) {
        enqueue_midwin(sg, b.prime_map[sg.slot], ev.mw[sg.slot]);
    }
    if (large_count > 0) {
        enqueue_packed_large(sg, hl_jb);
    }
    if (bucket_verify && stats.segments % bucket_verify_every == 0) {
        enqueue_bucket_verify(sg);
    }
    sg.l2_s = seconds_since(sg.t_launch0);
}

/// Step 4b' (mid windows, §9.2x): after a barrier (the tile kernel's stores
/// and mid_setup complete), mid_window rebuilds each 128 KB window of the
/// class map with the mid primes' marks and ANDs it into the map with
/// 64-bit atomics, so the packed kernel may run beside it (without HL_MT;
/// else its barrier orders its atomics after them).
void DenseRun::enqueue_midwin(const Segment& sg, uint64_t* map, ze_event_handle_t done) {
    ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
    const uint32_t c5w32 = sg.c5w * 2u;
    const uint64_t ncmax = (sg.q_odds + 14) / 15;
    const uint32_t wpc = uint32_t((ncmax + midw_bits - 1) / midw_bits);
    k_midw->set_arg(0, sizeof(map), &map);
    k_midw->set_arg(1, sizeof(b.sieve_primes), &b.sieve_primes);
    k_midw->set_arg(2, 4, &sieve_split);
    k_midw->set_arg(3, 4, &mid_n);
    k_midw->set_arg(4, sizeof(d_mid_i0[sg.slot]), &d_mid_i0[sg.slot]);
    k_midw->set_arg(5, 8, &sg.q_odds);
    k_midw->set_arg(6, 4, &sg.phis);
    k_midw->set_arg(7, 4, &c5w32);
    k_midw->set_arg(8, 4, &wpc);
    k_midw->set_arg(9, 4, &midw_bits);
    k_midw->set_local_arg(10, size_t(midw_bits) / 8);
    k_midw->set_arg(11, 8, &sg.q_first);
    k_midw->launch_overlap(ctx.list(), uint32_t(ncls_map) * wpc, done);
}

/// Step 4a: hit lists for primes [bucket_begin, sieve_count): the batch
/// harvest at the first segment of each batch, then (unless fused into the
/// packed kernel) hl_apply of this segment's list. Sets hl_jb.
void DenseRun::enqueue_hitlists(const Segment& sg, uint32_t& hl_jb) {
    // Hit lists for primes [bucket_begin, sieve_count): at the first
    // segment of each batch of hl_K, harvest the whole batch; then
    // mark this segment's list. The barrier before hl_apply orders
    // the harvest's list stores and the tile kernel's map stores.
    const uint32_t k_idx = uint32_t(stats.segments);
    if (k_idx >= run_segments)
        throw std::runtime_error("hit lists: segment index beyond the run");
    const uint32_t jb = k_idx % hl_K;
    hl_used[sg.slot] = false;
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
        k_hlh->set_arg(3, sizeof(sg.q_first), &sg.q_first);
        k_hlh->set_arg(4, sizeof(delta32), &delta32);
        k_hlh->set_arg(5, sizeof(qof32), &qof32);
        k_hlh->set_arg(6, sizeof(nseg), &nseg);
        k_hlh->set_arg(7, sizeof(qtail), &qtail);
        k_hlh->set_arg(8, sizeof(inv_delta), &inv_delta);
        k_hlh->set_arg(9, sizeof(d_hl_lists), &d_hl_lists);
        k_hlh->set_arg(10, sizeof(hl_capg), &hl_capg);
        k_hlh->set_arg(11, sizeof(d_hl_counts), &d_hl_counts);
        k_hlh->set_arg(12, sizeof(bkt_err), &bkt_err);
        if (hl_mt) {
            const uint32_t mt_words = mt_blocks * 180u;
            k_hlh->set_arg(13, sizeof(d_mtab), &d_mtab);
            k_hlh->set_arg(14, sizeof(mt_k_lo), &mt_k_lo);
            k_hlh->set_arg(15, sizeof(mt_words), &mt_words);
        }
        k_hlh->launch_overlap(ctx.list(), hl_G, ev.hl[sg.slot]);   // exactly hl_G: the region layout
        hl_used[sg.slot] = true;
    }
    hl_jb = jb;
    if (!hl_fuse) {
        ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
        const uint32_t q_odds32 = uint32_t(sg.q_odds);
        k_hla->set_arg(0, sizeof(b.prime_map[sg.slot]), &b.prime_map[sg.slot]);
        k_hla->set_arg(1, sizeof(d_hl_lists), &d_hl_lists);
        k_hla->set_arg(2, sizeof(hl_capg), &hl_capg);
        k_hla->set_arg(3, sizeof(d_hl_counts), &d_hl_counts);
        k_hla->set_arg(4, sizeof(jb), &jb);
        k_hla->set_arg(5, sizeof(hl_G), &hl_G);
        k_hla->set_arg(6, sizeof(q_odds32), &q_odds32);
        k_hla->set_arg(7, sizeof(bkt_err), &bkt_err);
        k_hla->set_arg(8, 4, &cm_u32);
        k_hla->set_arg(9, 4, &sg.ga1);
        k_hla->set_arg(10, 4, &sg.ga2);
        k_hla->set_arg(11, 4, &sg.off5);
        k_hla->launch(ctx.list(), bucket_groups, ev.bucket[sg.slot]);
    }
}

/// Step 4c: the packed large-prime kernel (with the fused hl_apply when
/// hl_fuse; hl_jb is this segment's list).
void DenseRun::enqueue_packed_large(const Segment& sg, uint32_t hl_jb) {
    // BARRIER, not just an event wait: the large kernel's atomic ANDs
    // must see the tile kernel's whole-word stores. On this driver an
    // event wait alone does not make those stores visible, which
    // manifested as nondeterministic lost composite marks
    // (tools/debug_map.cpp: the map vs Miller-Rabin diff varied run
    // to run until the barrier was added).
    // mid_window ANDs its windows into the map with atomics, so the packed
    // kernel may run beside it: both only AND-clear bits. The barrier
    // enqueue_midwin appends before mid_window already follows the tile
    // sieve's stores, mid_setup and the hit-list harvest (enqueued before it),
    // which is all this kernel needs. With HL_MT the packed kernel waits for
    // mid_window instead (barrier): it then shortens with every prime moved
    // to the hit lists, while beside mid_window its span did not (2026-11-18:
    // 35M gpu -2 % only without the overlap); GB_HL_MT=0 keeps the overlap
    // (+0.5 % on that path).
    const bool ml_ovl = !hl_mt && mid_n > 0;
    if (!ml_ovl) ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
    Kernel& kl = hl_fuse ? *k_hla : k_large;
    kl.set_arg(0, sizeof(b.prime_map[sg.slot]), &b.prime_map[sg.slot]);
    kl.set_arg(1, sizeof(d_large_bases),  &d_large_bases);
    kl.set_arg(2, sizeof(d_large_gaps),   &d_large_gaps);
    kl.set_arg(3, sizeof(packed_count),   &packed_count);
    kl.set_arg(4, sizeof(sg.q_odds),         &sg.q_odds);
    kl.set_arg(5, sizeof(sg.q_first),        &sg.q_first);
    kl.set_arg(6, sizeof(sg.q_last),         &sg.q_last);
    kl.set_arg(7, 4, &cm_u32);
    kl.set_arg(8, 4, &sg.ga1);
    kl.set_arg(9, 4, &sg.ga2);
    kl.set_arg(10, 4, &sg.off5);
    if (hl_fuse) {
        kl.set_arg(11, sizeof(d_hl_lists), &d_hl_lists);
        kl.set_arg(12, sizeof(hl_capg), &hl_capg);
        kl.set_arg(13, sizeof(d_hl_counts), &d_hl_counts);
        kl.set_arg(14, sizeof(hl_jb), &hl_jb);
        kl.set_arg(15, sizeof(hl_G), &hl_G);
        kl.set_arg(16, sizeof(bkt_err), &bkt_err);
    }
    const uint64_t large_items = (uint64_t(packed_count) + LARGE_PACK_PPI - 1) / LARGE_PACK_PPI;
    if (large_items > 0) {
        if (ml_ovl) kl.launch_overlap(ctx.list(), uint32_t((large_items + large_wg - 1) / large_wg),
                                      ev.large[sg.slot]);
        else kl.launch(ctx.list(), uint32_t((large_items + large_wg - 1) / large_wg),
                       ev.large[sg.slot]);
    }
}

/// Step 4d (GB_BUCKET_VERIFY): rebuild the map without hit lists into the
/// scratch map and count differing words (checked at collection).
void DenseRun::enqueue_bucket_verify(const Segment& sg) {
    ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
    k_sieve.set_arg(0, sizeof(d_verify_map), &d_verify_map);
    k_sieve.launch(ctx.list(), uint32_t(sg.tiles_launch));
    if (verify_mid) enqueue_midwin(sg, d_verify_map, nullptr);   // barrier, mid windows (mid_setup's table)
    else ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
    k_large_full->set_arg(0, sizeof(d_verify_map), &d_verify_map);
    k_large_full->set_arg(1, sizeof(d_full_bases), &d_full_bases);
    k_large_full->set_arg(2, sizeof(d_full_gaps), &d_full_gaps);
    const uint32_t full_count = sieve_count - verify_lo;
    k_large_full->set_arg(3, sizeof(full_count), &full_count);
    k_large_full->set_arg(4, sizeof(sg.q_odds), &sg.q_odds);
    k_large_full->set_arg(5, sizeof(sg.q_first), &sg.q_first);
    k_large_full->set_arg(6, sizeof(sg.q_last), &sg.q_last);
    k_large_full->set_arg(7, 4, &cm_u32);
    k_large_full->set_arg(8, 4, &sg.ga1);
    k_large_full->set_arg(9, 4, &sg.ga2);
    k_large_full->set_arg(10, 4, &sg.off5);
    const uint64_t items_full = (uint64_t(full_count) + LARGE_PACK_PPI - 1) / LARGE_PACK_PPI;
    k_large_full->launch(ctx.list(), uint32_t((items_full + large_wg - 1) / large_wg));
    ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
    const uint64_t nwords = uint64_t(sg.cgeo.K) * sg.c5w;
    k_cmp->set_arg(0, sizeof(b.prime_map[sg.slot]), &b.prime_map[sg.slot]);
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
            ZE_CHECK(zeCommandListAppendMemoryCopy(ctx.list(), a.data(), b.prime_map[sg.slot], nwords * 8, nullptr, 0, nullptr));
            ZE_CHECK(zeCommandListAppendMemoryCopy(ctx.list(), bref.data(), d_verify_map, nwords * 8, nullptr, 0, nullptr));
            ctx.host_sync();
            int shown = 0;
            for (uint64_t w = 0; w < nwords && shown < 12; ++w) {
                uint64_t x = a[w] ^ bref[w];
                while (x && shown < 12) {
                    const int bb = __builtin_ctzll(x); x &= x - 1;
                    const uint64_t bit = w * 64 + bb;
                    const uint64_t q = sg.cgeo.Q[w / sg.c5w] + sg.cgeo.qmod * (bit - uint64_t(w / sg.c5w) * sg.c5w * 64);
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

/// Per-segment step 5: the checker: this segment's ridge threshold and
/// statistics sampling, then count30 (statistics) and the mod-30 checker's
/// phases (the exact-histogram variant on sampled segments).
void DenseRun::enqueue_check(Segment& sg) {
    // The checker must SEE the completed map from both sieve kernels
    // (memory visibility, not just execution ordering — see the barrier note
    // in enqueue_packed_large). Same for the counter fills at the top of the
    // segment: the checker's atomics must not race them.
    ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
    const Seg30 s30 = seg30(sg.seg_start, sg.seg_evens, sg.q_first);
    if (!s30.ok)
        throw std::runtime_error("mod-30 checker: segment geometry rejected");
    // Recording threshold: the current Kth largest ridge (ratchets up as
    // the top list fills; ~0u disables every append).
    const uint32_t ridge_thresh_u32 =
        ridges->active() ? std::max(ridges->thresh(), ceil_c30) : 0xFFFFFFFFu;
    split_used[sg.slot] = false;                         // count30 sets it
    // Every stats_sample-th segment of the run (segment 0 always) gets the
    // full exact p(n) histogram; the others record the tail in check30_b.
    const bool sampled = stats_on && stats.segments % stats_sample == 0;
    sampled_slot[sg.slot] = sampled;
    slot_evens[sg.slot] = sg.seg_evens;
    if (sampled) sampled_evens += sg.seg_evens;
    xs_used[sg.slot] = sampled;
    acct_slot[sg.slot] = !sampled;
    // count30 (statistics) is enqueued by the checker step, after phase A's
    // barrier and before phase B: see enqueue_check30
    if (sampled) enqueue_check_sampled(sg, s30, ridge_thresh_u32);
    else enqueue_check30(sg, s30, ridge_thresh_u32);
    sg.enq_s = seconds_since(sg.t_enq);
    sg.launch_s = seconds_since(sg.t_launch0);
    sg.l3_s = seconds_since(sg.t_launch0);
}

/// Step 5a (statistics): count30 - prime / twin counts, the gap tail and,
/// on walked segments, the gap histogram (timed under the splitk event).
void DenseRun::enqueue_count30(const Segment& sg, bool sampled) {
    // count30 (prime / twin counts; fusing it into check30_a measured
    // no faster: the larger kernel slows phase A) is timed under the
    // splitk event (added to check_s).
    const uint32_t cw32 = 2 * sg.c5w;
    const std::vector<uint32_t> lim = count30_limits(sg.seg_start, sg.seg_end, sg.cgeo.Q);
    std::memcpy(b.x_lim[sg.slot], lim.data(), lim.size() * 4);
    // class order by phase (gap kernels): ord / phis in ascending phi,
    // phic by class; the smallest counted position
    uint32_t g_ord = 0, g_phis = 0, g_phic = 0, g_poslo = ~0u;
    {
        uint32_t cls[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        std::sort(cls, cls + 8, [&](uint32_t x, uint32_t y) { return sg.cgeo.phi[x] < sg.cgeo.phi[y]; });
        // count30's gap walk takes the phase order as a rotation of the
        // class order by ord & 15 (the residues sorted by (u - q0) mod 30)
        for (uint32_t k = 0; k < 8; ++k)
            if (cls[k] != (cls[0] + k) % 8)
                throw std::runtime_error("count30: classes in phase order are not a rotation");
        for (uint32_t k = 0; k < 8; ++k) {
            g_ord |= cls[k] << (4 * k);
            g_phis |= sg.cgeo.phi[cls[k]] << (4 * k);
            g_phic |= sg.cgeo.phi[k] << (4 * k);
            if (lim[2 * cls[k]] < lim[2 * cls[k] + 1])
                g_poslo = std::min(g_poslo, 30 * lim[2 * cls[k]] + 2 * sg.cgeo.phi[cls[k]]);
        }
    }
    // count30 also finds every gap >= GAPS30_TAIL (bit-parallel) and the
    // segment's first / last prime when the gap statistics are on, and
    // on the walked segments walks every prime (the gap histogram) on a
    // fixed grid of work-groups (one histogram flush each)
    const uint32_t ctcap = gaps_on ? GAPS30_TCAP : 0u;
    walked_slot[sg.slot] = gaps_on && (sampled || gaps_exact);
    const uint32_t cwalk = walked_slot[sg.slot] ? 1u : 0u;
    if (cw32 % COUNT30_CW) throw std::runtime_error("count30: class stride not a multiple of its lane width");
    const uint32_t cgroups = (cw32 / COUNT30_CW + WG_SIZE - 1) / WG_SIZE;
    slot_ggroups[sg.slot] = cwalk ? std::min(gaps_groups, cgroups) : cgroups;
    slot_qfirst[sg.slot] = sg.q_first;
    k_cnt->set_arg(0, sizeof(b.prime_map[sg.slot]), &b.prime_map[sg.slot]);
    k_cnt->set_arg(1, 4, &cw32);
    k_cnt->set_arg(2, sizeof(b.x_lim[sg.slot]), &b.x_lim[sg.slot]);
    k_cnt->set_arg(3, sizeof(b.x_cnt[sg.slot]), &b.x_cnt[sg.slot]);
    k_cnt->set_arg(4, 4, &g_ord);
    k_cnt->set_arg(5, 4, &g_phis);
    k_cnt->set_arg(6, 4, &g_phic);
    k_cnt->set_arg(7, 4, &g_poslo);
    k_cnt->set_arg(8, sizeof(b.g_tail[sg.slot]), &b.g_tail[sg.slot]);
    k_cnt->set_arg(9, 4, &ctcap);
    k_cnt->set_arg(10, 4, &cwalk);
    k_cnt->set_arg(11, sizeof(b.g_part[sg.slot]), &b.g_part[sg.slot]);
    k_cnt->set_arg(12, sizeof(b.g_hist[sg.slot]), &b.g_hist[sg.slot]);
    k_cnt->launch_overlap(ctx.list(), slot_ggroups[sg.slot], ev.splitk[sg.slot]);
    split_used[sg.slot] = true;
}

/// Step 5b (statistics, sampled segment): the full p(n) histogram checker,
/// check30_xs8/6/4/3 (ascending prefix) -> barrier -> check30_xb.
void DenseRun::enqueue_check_sampled(const Segment& sg, const Seg30& s30, uint32_t ridge_thresh_u32) {
    // Full histogram: check30_xs8/6/4/3 (ascending prefix) -> barrier
    // -> check30_xb (ascending continuation).
    const uint32_t cw32 = 2 * sg.c5w;
    const uint32_t se32 = uint32_t(sg.seg_evens);
    int first_xs = 0;
    while (first_xs < 4 && !splan.vn[first_xs]) ++first_xs;
    for (int v = 0; v < 4; ++v) {      // one launch per array-count variant
        if (!splan.vn[v]) continue;
        Kernel& kv = *k_cxs[v];
        const uint32_t per_item = XS_W_HOST[v] * 32;
        const uint32_t items_pad = ((s30.ec_max + per_item - 1) / per_item + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
        kv.set_arg(0, sizeof(b.prime_map[sg.slot]), &b.prime_map[sg.slot]);
        kv.set_arg(1, 4, &cw32);
        kv.set_arg(2, sizeof(b.x_all), &b.x_all);
        kv.set_arg(3, sizeof(b.x_tab), &b.x_tab);
        kv.set_arg(4, sizeof(b.x_j), &b.x_j);
        kv.set_arg(5, sizeof(b.x_ch), &b.x_ch);
        kv.set_arg(6, sizeof(b.pending[sg.slot]), &b.pending[sg.slot]);
        kv.set_arg(7, sizeof(b.pending_n[sg.slot]), &b.pending_n[sg.slot]);
        kv.set_arg(8, 4, &pending_cap);
        kv.set_arg(9, 4, &items_pad);
        kv.set_arg(10, 4, &se32);
        kv.set_arg(11, 4, &s30.sm30);
        kv.set_arg(12, 4, &s30.tb);
        kv.set_arg(13, 4, &s30.phik);
        kv.set_arg(14, sizeof(b.x_hist[sg.slot]), &b.x_hist[sg.slot]);
        kv.set_arg(15, 8, &splan.vcls[v]);
        // the first variant may overlap count30 (see phase A below)
        const ze_event_handle_t xe = v == 3 ? ev.checka[sg.slot] : ev.xv[v][sg.slot];
        if (v == first_xs) kv.launch_overlap(ctx.list(), splan.vn[v] * items_pad / WG_SIZE, xe);
        else kv.launch(ctx.list(), splan.vn[v] * items_pad / WG_SIZE, xe);
    }
    ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
    enqueue_count30(sg, true);   // before phase B: see enqueue_check30 (sampled => stats_on)
    k_cxb->set_arg(0, sizeof(b.prime_map[sg.slot]), &b.prime_map[sg.slot]);
    k_cxb->set_arg(1, sizeof(b.x_all), &b.x_all);
    k_cxb->set_arg(2, sizeof(b.x_tab), &b.x_tab);
    k_cxb->set_arg(3, sizeof(b.pending[sg.slot]), &b.pending[sg.slot]);
    k_cxb->set_arg(4, sizeof(b.pending_n[sg.slot]), &b.pending_n[sg.slot]);
    k_cxb->set_arg(5, sizeof(b.leftovers[sg.slot]), &b.leftovers[sg.slot]);
    k_cxb->set_arg(6, sizeof(b.leftover_n[sg.slot]), &b.leftover_n[sg.slot]);
    k_cxb->set_arg(7, sizeof(sg.q_first), &sg.q_first);
    k_cxb->set_arg(8, sizeof(sg.q_odds), &sg.q_odds);
    k_cxb->set_arg(9, sizeof(sg.seg_start), &sg.seg_start);
    k_cxb->set_arg(10, 4, &pending_cap);
    k_cxb->set_arg(11, 4, &sg.ga1);
    k_cxb->set_arg(12, 4, &sg.ga2);
    k_cxb->set_arg(13, 4, &sg.off5);
    k_cxb->set_arg(14, sizeof(b.ridge[sg.slot]), &b.ridge[sg.slot]);
    k_cxb->set_arg(15, 4, &ridge_thresh_u32);
    k_cxb->set_arg(16, sizeof(b.x_hist[sg.slot]), &b.x_hist[sg.slot]);
    k_cxb->set_arg(17, sizeof(b.p_first[sg.slot]), &b.p_first[sg.slot]);
    k_cxb->set_arg(18, sizeof(b.want_p), &b.want_p);
    // grid-stride; the barrier above already orders it after the variants
    k_cxb->launch_overlap(ctx.list(), stats_b_groups, ev.check[sg.slot]);
}

/// Step 5c: the mod-30 checker, check30_a -> barrier -> check30_b.
void DenseRun::enqueue_check30(const Segment& sg, const Seg30& s30, uint32_t ridge_thresh_u32) {
    // Mod-30 checker: phase A on the class arrays -> barrier -> phase B.
    const uint32_t cw32 = 2 * sg.c5w;
    const uint32_t per_item = a_w64 * 64;
    const uint32_t items_pad = ((s30.ec_max + per_item - 1) / per_item + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
    const uint32_t se32 = uint32_t(sg.seg_evens);
    k_ca->set_arg(0, sizeof(b.prime_map[sg.slot]), &b.prime_map[sg.slot]);
    k_ca->set_arg(1, 4, &cw32);
    k_ca->set_arg(2, sizeof(no_checked), &no_checked);
    k_ca->set_arg(3, 4, &chk_stride30);
    k_ca->set_arg(4, sizeof(b.c30_cands), &b.c30_cands);
    k_ca->set_arg(5, sizeof(b.c30_meta), &b.c30_meta);
    k_ca->set_arg(6, sizeof(b.pending[sg.slot]), &b.pending[sg.slot]);
    k_ca->set_arg(7, sizeof(b.pending_n[sg.slot]), &b.pending_n[sg.slot]);
    k_ca->set_arg(8, 4, &pending_cap);
    k_ca->set_arg(9, 4, &items_pad);
    k_ca->set_arg(10, 4, &se32);
    k_ca->set_arg(11, 4, &s30.sm30);
    k_ca->set_arg(12, 4, &s30.tb);
    k_ca->set_arg(13, 4, &s30.phik);
    k_ca->set_arg(14, 4, &write_checked);
    k_ca->set_arg(15, sizeof(b.acct[sg.slot]), &b.acct[sg.slot]);
    // launch_overlap: phase A only reads the finished map (the barrier above
    // follows the marks)
    k_ca->launch_overlap(ctx.list(), 15 * items_pad / WG_SIZE, ev.checka[sg.slot]);
    ZE_CHECK(zeCommandListAppendBarrier(ctx.list(), nullptr, 0, nullptr));
    // count30 (statistics; reads the finished map only) between phase A's
    // barrier and phase B, neither behind a further barrier: launched first,
    // its work-groups take their thread slots first and it ends inside the
    // next segment's tile sieve (2026-11-18: tile kernel 0.567 -> 0.495 ms per
    // segment, gpu -0.9%; beside phase A, which is ALU-bound, it cost ~1.5x
    // as much). The next mid_window barrier and the host's wait on its event
    // (collection) order everything after it.
    if (stats_on) enqueue_count30(sg, false);
    k_cb->set_arg(0, sizeof(b.prime_map[sg.slot]), &b.prime_map[sg.slot]);
    k_cb->set_arg(1, sizeof(no_checked), &no_checked);
    k_cb->set_arg(2, 4, &chk_stride30);
    k_cb->set_arg(3, sizeof(b.c30_rest), &b.c30_rest);
    k_cb->set_arg(4, sizeof(b.c30_rtab), &b.c30_rtab);
    k_cb->set_arg(5, sizeof(b.pending[sg.slot]), &b.pending[sg.slot]);
    k_cb->set_arg(6, sizeof(b.pending_n[sg.slot]), &b.pending_n[sg.slot]);
    k_cb->set_arg(7, sizeof(b.leftovers[sg.slot]), &b.leftovers[sg.slot]);
    k_cb->set_arg(8, sizeof(b.leftover_n[sg.slot]), &b.leftover_n[sg.slot]);
    k_cb->set_arg(9, sizeof(sg.q_first), &sg.q_first);
    k_cb->set_arg(10, sizeof(sg.q_odds), &sg.q_odds);
    k_cb->set_arg(11, sizeof(sg.seg_start), &sg.seg_start);
    k_cb->set_arg(12, 4, &pending_cap);
    k_cb->set_arg(13, 4, &write_checked);
    k_cb->set_arg(14, 4, &sg.ga1);
    k_cb->set_arg(15, 4, &sg.ga2);
    k_cb->set_arg(16, 4, &sg.off5);
    k_cb->set_arg(17, sizeof(b.ridge[sg.slot]), &b.ridge[sg.slot]);
    k_cb->set_arg(18, 4, &ridge_thresh_u32);
    k_cb->set_arg(19, sizeof(b.p_hist[sg.slot]), &b.p_hist[sg.slot]);
    k_cb->set_arg(20, sizeof(b.p_first[sg.slot]), &b.p_first[sg.slot]);
    k_cb->set_arg(21, sizeof(b.want_p), &b.want_p);
    k_cb->set_arg(22, 4, &tail_t);
    k_cb->set_arg(23, sizeof(b.acct[sg.slot]), &b.acct[sg.slot]);
    if (stats_on) k_cb->launch_overlap(ctx.list(), phase_b_groups, ev.check[sg.slot]);
    else k_cb->launch(ctx.list(), phase_b_groups, ev.check[sg.slot]);  // grid-stride
}

/// Per-segment step 6: collects the PREVIOUS segment (other slot) once this
/// one is queued: waits on its check event, adds its kernel times, fails
/// closed on device error flags / verify mismatch / pending overflow, merges
/// ridges and statistics, records its telemetry, resolves its leftovers on
/// the CPU and saves the ridge store.
void DenseRun::collect_previous_segment(Segment& sg) {
    const uint32_t prev = uint32_t((stats.segments - 1) & 1);
    const auto t_wait = clock_::now();
    ZE_CHECK(zeEventHostSynchronize(ev.check[prev], UINT64_MAX));
    if (stats_on) ZE_CHECK(zeEventHostSynchronize(ev.splitk[prev], UINT64_MAX));
    sg.wait_s = seconds_since(t_wait);
    const auto t_coll = clock_::now();

    collect_slot(prev, prev_stat);
    sg.coll_s = seconds_since(t_coll);
}

/// Per-segment step 7: phase sums and the GB_PHASE line, the carry-over of
/// this segment to the iteration (or drain) that collects it, the segment
/// count and the verbose progress line.
void DenseRun::end_iteration(const Segment& sg) {
    sum_prep += sg.prep_s;
    sum_enq += sg.enq_s;
    sum_wait += sg.wait_s;
    sum_coll += sg.coll_s;
    sum_args += sg.args_s;
    sum_launch += sg.launch_s;
    if (phase_trace) {
        std::fprintf(stderr,
            "  phase seg %llu: prep %.1f ms (off %.1f tab %.1f)  enq %.1f ms "
            "(args %.1f l1 %.2f l2 %.2f l3 %.2f)  wait %.1f ms  coll %.1f ms\n",
            (unsigned long long)stats.segments,
            sg.prep_s * 1e3, offset_seconds - sg.offset_before,
            (table_seconds - sg.table_before) * 1e3,
            sg.enq_s * 1e3, sg.args_s * 1e3, sg.l1_s * 1e3, (sg.l2_s - sg.l1_s) * 1e3,
            (sg.l3_s - sg.l2_s) * 1e3, sg.wait_s * 1e3, sg.coll_s * 1e3);
    }

    prev_seg_start = sg.seg_start;
    prev_seg_evens = sg.seg_evens;

    // Finalize the carry-over for this segment: its GPU work will be
    // collected in the NEXT loop iteration (or the drain) — that is the
    // point at which prev_seg_wall / prev_seg_offsets are consumed.
    prev_seg_wall    = seconds_since(sg.t_seg_wall);
    prev_seg_offsets = offset_seconds - sg.offset_before;
    prev_stat = seg_stat;

    ++stats.segments;

    if (verbose && (stats.segments % 10 == 0 || sg.seg_end == limit)) {
        std::printf("  segment %llu: evens [%llu, %llu]  leftovers_total=%llu\n",
                    (unsigned long long)stats.segments,
                    (unsigned long long)sg.seg_start,
                    (unsigned long long)sg.seg_end,
                    (unsigned long long)total_leftovers);
    }
}

/// After the loop: collects the final segment (its record is seg_stat: it was
/// never overwritten after its enqueue).
void DenseRun::drain_final_segment() {
    const uint32_t last = uint32_t((stats.segments - 1) & 1);
    ZE_CHECK(zeEventHostSynchronize(ev.check[last], UINT64_MAX));
    if (stats_on) ZE_CHECK(zeEventHostSynchronize(ev.splitk[last], UINT64_MAX));
    collect_slot(last, seg_stat);
}

/// Collects a completed slot (its check event has fired; prev_seg_start /
/// prev_seg_evens describe its segment): kernel times, device error flags,
/// the pending overflow, ridges and statistics, the coverage accounting,
/// the telemetry record @p rec, the CPU fallback for its leftovers and the
/// ridge store.
/// @throws std::runtime_error on any device error flag, a pending overflow,
///         a coverage mismatch or a genuine counterexample (fail closed).
void DenseRun::collect_slot(uint32_t sl, SegmentStats& rec) {
    double ds = 0.0, dl = 0.0, dv = 0.0;
    query_slot(sl, ds, dl, dv);
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
    const uint32_t pend = *b.pending_n[sl];
    if (pend > pending_cap)
        throw std::runtime_error("two-phase checker: pending list overflow");
    if (phase_trace)
        std::fprintf(stderr, "    pending %u\n", pend);
    // Read the 4-byte counter directly from shared memory (no D2H copy).
    const uint32_t n_left = *b.leftover_n[sl];
    total_leftovers += n_left;
    ridges->merge(b.ridge[sl], prev_seg_start);
    if (stats_on) collect_stats(sl, prev_seg_start);
    check_coverage(sl, pend, n_left);

    if (collect_segments) {
        rec.sieve_seconds  = ds;
        rec.large_seconds  = dl;
        rec.check_seconds  = dv;
        rec.leftovers      = n_left;
        rec.offset_seconds = prev_seg_offsets;
        rec.wall_seconds   = prev_seg_wall;
        stats.per_segment.push_back(rec);
    }

    if (n_left > 0)
        resolve_leftovers(ctx, b.leftovers[sl], n_left, prev_seg_start,
                          prev_seg_evens, gb_primes, verbose, stats.segments - 1, &*ridges,
                          gstats.get(), sampled_slot[sl]);
    ridges->save(ridge_file);
    reset_slot_counters(sl);
}

/// Zeroes slot @p sl's checker counters (leftover_n, pending_n, acct and the
/// ridge count) from the host, as collect_stats does for p_hist / x_cnt.
/// @pre no command that touches the slot is in flight (its check and count30
///      events have fired, or the slot was never used).
void DenseRun::reset_slot_counters(uint32_t sl) {
    *b.leftover_n[sl] = 0;
    *b.pending_n[sl] = 0;
    b.acct[sl][0] = b.acct[sl][1] = 0;
    *reinterpret_cast<uint32_t*>(b.ridge[sl]) = 0;
}

/// Positive coverage of a collected slot: every even of its segment must be
/// accounted for as resolved on the GPU or listed for the CPU fallback. A
/// PASS used to mean only "no failure was reported": a checker launch that
/// did not run (or covered too few evens) left nothing pending and passed.
/// mod-30 segments: check30_a's resolved count + pending == segment evens and
/// check30_b's resolved count + leftovers == pending. Sampled segments: the
/// exact histogram's count of the segment + leftovers == segment evens.
/// @throws std::runtime_error on a mismatch (the run fails, nothing is merged).
void DenseRun::check_coverage(uint32_t sl, uint32_t pend, uint32_t n_left) {
    const std::string seg = " (segment starting " + std::to_string(prev_seg_start) + ", " +
                            std::to_string(prev_seg_evens) + " evens)";
    if (sampled_slot[sl]) {
        if (slot_hist_n[sl] + n_left != prev_seg_evens)
            throw std::runtime_error("coverage: exact histogram " + std::to_string(slot_hist_n[sl]) +
                                     " + leftovers " + std::to_string(n_left) + " != segment evens" + seg);
        return;
    }
    if (!acct_slot[sl]) return;          // a fallback checker without accounting
    const uint64_t res_a = b.acct[sl][0], res_b = b.acct[sl][1];
    if (res_a + pend != prev_seg_evens)
        throw std::runtime_error("coverage: phase A resolved " + std::to_string(res_a) + " + pending " +
                                 std::to_string(pend) + " != segment evens" + seg);
    if (res_b + n_left != pend)
        throw std::runtime_error("coverage: phase B resolved " + std::to_string(res_b) + " + leftovers " +
                                 std::to_string(n_left) + " != pending " + std::to_string(pend) + seg);
}

/// Phase 12: the segment loop. Each iteration enqueues segment k on its slot,
/// then collects segment k - 1 from the other slot; the drain collects the
/// last one.
void DenseRun::segment_loop() {
    begin_segment_loop();

    for (uint64_t seg_start = start_even; seg_start <= limit; seg_start += segment_even) {
        Segment sg;
        sg.seg_start = seg_start;
        begin_segment(sg);
        prepare_segment_tables(sg);
        enqueue_sieve(sg);
        enqueue_large_primes(sg);
        enqueue_check(sg);

        // ---- Collect the PREVIOUS segment's results (other slot) ----------
        // The host blocks here only after the next segment's work is already
        // queued, so the GPU transitions straight from segment k-1's check to
        // segment k's sieve without an idle gap.
        if (stats.segments > 0) {
            collect_previous_segment(sg);
        }
        end_iteration(sg);

        // The merged final segment already includes limit. Without this
        // break, the for-loop increment lands exactly on limit and schedules
        // it again as a duplicate one-even segment, recreating the expensive
        // high-range tail pathology that the merge is intended to remove.
        if (sg.seg_end == limit) break;
    }

    // ---- Drain: collect the final segment --------------------------------
    drain_final_segment();
}

/// Phase 13: RunStats from the loop's sums; the statistics run is finished
/// and committed; the ridge overflow warning.
void DenseRun::assemble_run_stats() {
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
}

/// Phase 14 (success path only, as before): destroys the events and frees the
/// device buffers in run_gpu's order. On an exception nothing here runs; the
/// kernels, module and context are destroyed by ~DenseRun (as the locals were).
void DenseRun::release_device_objects() {
    ev.destroy();

    // Free GPU memory.
    for (int s = 0; s < 2; ++s) {
        Context::free_mem(ctx.context(), b.prime_map[s]);
        Context::free_mem(ctx.context(), b.first_bits[s]);
        Context::free_mem(ctx.context(), b.mask_table[s]);
        if (b.group_masks[s]) Context::free_mem(ctx.context(), b.group_masks[s]);
        if (b.group_y[s]) Context::free_mem(ctx.context(), b.group_y[s]);
        Context::free_mem(ctx.context(), b.leftovers[s]);
        Context::free_mem(ctx.context(), b.leftover_n[s]);
        Context::free_mem(ctx.context(), b.tile_first[s]);
        if (d_mid_i0[s]) Context::free_mem(ctx.context(), d_mid_i0[s]);
        if (b.pending[s]) Context::free_mem(ctx.context(), b.pending[s]);
        if (b.pending_n[s]) Context::free_mem(ctx.context(), b.pending_n[s]);
        if (b.acct[s]) Context::free_mem(ctx.context(), b.acct[s]);
        if (b.ridge[s]) Context::free_mem(ctx.context(), b.ridge[s]);
        Context::free_mem(ctx.context(), stats_block[s]);
    }
    for (void* p : {(void*)b.c30_cands, (void*)b.c30_meta, (void*)b.c30_rest, (void*)b.c30_rtab})
        if (p) Context::free_mem(ctx.context(), p);
    Context::free_mem(ctx.context(), d_group_desc);
    Context::free_mem(ctx.context(), b.sieve_primes);
    Context::free_mem(ctx.context(), d_large_bases);
    if (d_hl_lists) Context::free_mem(ctx.context(), d_hl_lists);
    if (d_hl_counts) Context::free_mem(ctx.context(), d_hl_counts);
    if (d_mtab) Context::free_mem(ctx.context(), d_mtab);
    Context::free_mem(ctx.context(), bkt_err);
    Context::free_mem(ctx.context(), d_full_bases);
    Context::free_mem(ctx.context(), d_full_gaps);
    Context::free_mem(ctx.context(), d_verify_map);
    Context::free_mem(ctx.context(), verify_diff);
    Context::free_mem(ctx.context(), d_large_gaps);
    Context::free_mem(ctx.context(), b.magic);
    Context::free_mem(ctx.context(), b.shifts);
    Context::free_mem(ctx.context(), b.ms32);
}

/// Runs the phases after construction (phases 1-2 and the Level Zero objects
/// are done by then), in run_gpu's order.
RunStats DenseRun::run() {
    // ---- Level Zero setup (ctx, mod and the first four kernels: DenseRun's
    // member initializers) -----------------------------------------------------
    // The base-prime cache check runs beside the rest of init (buffers,
    // uploads, mt_build: ~0.5 s). Started only now: a second thread alive
    // during zeInit / module creation made those ~60 ms slower.
    cache_check = std::async(std::launch::async, [this] { return prime_view.deferred_check(); });
    configure_checkers();
    configure_ridges_and_stats();
    set_group_sizes();

    // ---- buffers and uploads -------------------------------------------------
    size_buffers();
    allocate_slot_buffers();
    size_bucket_ring_and_hitlists();
    allocate_large_prime_buffers();
    upload_tables();
    pack_stats_buffers();
    create_events();

    // ---- segment loop ------------------------------------------------------
    segment_loop();

    // ---- teardown ------------------------------------------------------------
    assemble_run_stats();
    release_device_objects();
    return std::move(stats);   // a member: move, not copy
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
    for (const auto& [d, dev] : gpu_devices()) {
        ze_device_properties_t p{};
        p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        ZE_CHECK(zeDeviceGetProperties(dev, &p));

        ze_device_properties_t p12{};   // timerResolution lives here (v1.17+)
        p12.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
        const bool have12 =
            zeDeviceGetProperties(dev, &p12) == ZE_RESULT_SUCCESS;
        const double timer_hz = have12 ? double(p12.timerResolution) : 0.0;

        ze_device_memory_properties_t m{};
        m.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
        uint32_t nm = 1;
        ZE_CHECK(zeDeviceGetMemoryProperties(dev, &nm, &m));

        ze_device_compute_properties_t c{};
        c.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
        ZE_CHECK(zeDeviceGetComputeProperties(dev, &c));

        ze_api_version_t api = ZE_API_VERSION_1_0;
        ZE_CHECK(zeDriverGetApiVersion(d, &api));

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
                    "phase-A words %u | MAX_GB_PRIME %u | "
                    "seg default %llu\n",
                    WG_SIZE, TILE_WORDS, double(TILE_WORDS * 8) / 1024.0,
                    CHECK_PHASE_A_W64,
                    MAX_GB_PRIME,
                    (unsigned long long)DEFAULT_SEGMENT_EVEN);
    }
}

} // namespace gb
