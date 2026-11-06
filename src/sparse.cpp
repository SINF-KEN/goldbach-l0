// Sparse class map pipeline (v2.0, `--sparse`). See include/sparse.hpp for
// the math and kernels/sparse.cl for the kernels.
//
// Per segment (slot = s & 1, double-buffered like run_gpu):
//   host : first-hit table of the tile primes (incremental: t -= dlt mod p),
//          mask-group word offsets, valid even range per class
//   GPU  : scm_tile -> barrier -> scm_large -> barrier -> scm_check_a ->
//          barrier -> scm_check_b
//   host : collect the other slot (leftovers -> Miller-Rabin fallback)
#include "sparse_run.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include "config.hpp"
#include "l0_context.hpp"
#include "primes.hpp"
#include "sparse.hpp"

namespace gb {
namespace {

using clock_ = std::chrono::steady_clock;
/// @return seconds elapsed since @p t0.
double secs(clock_::time_point t0) {
    return std::chrono::duration<double>(clock_::now() - t0).count();
}

/// @return environment variable @p name parsed by strtoull (base auto-detected;
///         unparsable text yields 0), or @p dflt when unset.
uint64_t env_u64(const char* name, uint64_t dflt) {
    const char* e = std::getenv(name);
    return e ? std::strtoull(e, nullptr, 0) : dflt;
}
/// @return environment variable @p name parsed by strtod, or @p dflt when unset.
double env_f(const char* name, double dflt) {
    const char* e = std::getenv(name);
    return e ? std::strtod(e, nullptr) : dflt;
}

/// Splits [begin, end) into at most min(hardware threads, max_workers)
/// contiguous chunks and calls fn(first, last) for each on its own thread
/// (inline when one worker suffices).
/// @pre   fn is safe to run concurrently on disjoint ranges and does not throw.
/// @post  every index was covered exactly once; all threads joined.
template <typename Fn>
void par_for(uint32_t begin, uint32_t end, uint32_t max_workers, Fn&& fn) {
    const uint32_t count = end - begin;
    if (count == 0) return;
    const uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
    const uint32_t workers = std::max(1u, std::min({hw, max_workers, count}));
    if (workers <= 1) { fn(begin, end); return; }
    std::vector<std::thread> th;
    for (uint32_t t = 0; t < workers; ++t) {
        const uint32_t a = begin + uint32_t((uint64_t(count) * t) / workers);
        const uint32_t b = begin + uint32_t((uint64_t(count) * (t + 1)) / workers);
        th.emplace_back(fn, a, b);
    }
    for (auto& x : th) x.join();
}

// One byte per large prime (half-gap), first prime of each packing group:
// the sieve_large_packed format (kernels/sparse.cl decode_large).
struct Packed { std::vector<uint32_t> bases; std::vector<uint8_t> gaps; };
/// Packs primes[split, count) into the Packed format (one base per
/// LARGE_PACK_GROUP primes, half-gaps elsewhere, padded to whole items).
/// @pre   primes ascending odd, split <= count.
/// @return bases and gaps (both non-empty even when no prime is packed).
/// @throws std::runtime_error if a gap is odd or its half exceeds 255 (fail closed).
Packed pack_primes(const uint32_t* primes, uint32_t split, uint32_t count) {
    Packed pk;
    const uint64_t n = count - split;
    const uint64_t items = (n + LARGE_PACK_PPI - 1) / LARGE_PACK_PPI;
    pk.gaps.assign(std::max<uint64_t>(items, 1) * LARGE_PACK_PPI, 0);
    pk.bases.assign(std::max<uint64_t>((n + LARGE_PACK_GROUP - 1) / LARGE_PACK_GROUP, 1), 0);
    const uint64_t groups = (n + LARGE_PACK_GROUP - 1) / LARGE_PACK_GROUP;
    std::atomic<bool> bad{false};
    par_for(0, uint32_t(groups), 16, [&](uint32_t g0, uint32_t g1) {
        for (uint64_t i = uint64_t(g0) * LARGE_PACK_GROUP;
             i < std::min<uint64_t>(n, uint64_t(g1) * LARGE_PACK_GROUP); ++i) {
            const uint32_t p = primes[split + i];
            if (i % LARGE_PACK_GROUP == 0) { pk.bases[i / LARGE_PACK_GROUP] = p; continue; }
            const uint32_t gap = p - primes[split + i - 1];
            if ((gap & 1) || gap / 2 > 0xFF) { bad = true; return; }
            pk.gaps[i] = uint8_t(gap / 2);
        }
    });
    if (bad) throw std::runtime_error("sparse: prime gap does not fit the packed format");
    return pk;
}

// ---- checker plan disk cache (keyed by every plan input; validated on load) ----
/// FNV-1a 64-bit hash of @p n bytes, continuing from @p h (chainable).
/// @return the updated hash.
uint64_t fnv64(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
/// Appends @p v to @p out as a u64 element count followed by the raw elements.
template <typename T> void put_vec(std::string& out, const std::vector<T>& v) {
    const uint64_t n = v.size();
    out.append(reinterpret_cast<const char*>(&n), 8);
    out.append(reinterpret_cast<const char*>(v.data()), n * sizeof(T));
}
/// Reads a put_vec record from @p in at @p pos into @p v.
/// @post  on success pos is advanced past the record.
/// @return false if the record is truncated (v and pos then unspecified).
template <typename T> bool get_vec(const std::string& in, size_t& pos, std::vector<T>& v) {
    uint64_t n = 0;
    if (pos + 8 > in.size()) return false;
    std::memcpy(&n, in.data() + pos, 8); pos += 8;
    if (n > (in.size() - pos) / sizeof(T)) return false;
    v.resize(n);
    std::memcpy(v.data(), in.data() + pos, n * sizeof(T)); pos += n * sizeof(T);
    return true;
}
/// Loads a cached plan written by save_plan.
/// @param key  the full input description; must equal the stored key.
/// @post  on success @p plan holds the cached tables and statistics. The
///        caller must still run validate_plan_scm (the checksum only guards
///        against corruption, not against a wrong plan).
/// @return false if the file is missing, truncated, fails its checksum or
///         has another key (plan then partially overwritten: rebuild it).
bool load_plan(const std::string& path, const std::string& key, PlanSCM& plan) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::string in;
    char buf[1 << 16];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) in.append(buf, n);
    std::fclose(f);
    if (in.size() < 16) return false;
    uint64_t sum = 0;
    std::memcpy(&sum, in.data() + in.size() - 8, 8);
    if (fnv64(in.data(), in.size() - 8) != sum) return false;
    size_t pos = 0;
    std::vector<char> k;
    if (!get_vec(in, pos, k) || std::string(k.begin(), k.end()) != key) return false;
    std::vector<double> st;
    std::vector<uint32_t> sc;
    if (!get_vec(in, pos, plan.cands) || !get_vec(in, pos, plan.meta) || !get_vec(in, pos, plan.rest) ||
        !get_vec(in, pos, plan.rtab) || !get_vec(in, pos, st) || !get_vec(in, pos, sc) ||
        st.size() != 4 || sc.size() != 2)
        return false;
    plan.mean_tests = st[0]; plan.mean_cost = st[1]; plan.mean_funnels = st[2]; plan.mean_segs = st[3];
    plan.max_member = sc[0]; plan.pmax = sc[1];
    return true;
}
/// Writes @p plan under @p key with a trailing FNV-1a checksum, atomically
/// (temp file + rename). Best effort: any I/O failure leaves no file behind
/// and is otherwise ignored (the cache is optional).
void save_plan(const std::string& path, const std::string& key, const PlanSCM& plan) {
    std::string out;
    put_vec(out, std::vector<char>(key.begin(), key.end()));
    put_vec(out, plan.cands); put_vec(out, plan.meta); put_vec(out, plan.rest); put_vec(out, plan.rtab);
    put_vec(out, std::vector<double>{plan.mean_tests, plan.mean_cost, plan.mean_funnels, plan.mean_segs});
    put_vec(out, std::vector<uint32_t>{plan.max_member, plan.pmax});
    const uint64_t sum = fnv64(out.data(), out.size());
    out.append(reinterpret_cast<const char*>(&sum), 8);
    const std::string tmp = path + ".tmp" + std::to_string(::getpid());
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return;                                   // cache is optional
    const bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
    if (std::fclose(f) == 0 && ok) std::rename(tmp.c_str(), path.c_str());
    else std::remove(tmp.c_str());
}

/// Host-visible kernel-timestamp events, four per slot.
struct Events {
    ze_event_pool_handle_t pool = nullptr;
    ze_event_handle_t ev[2][4] = {};     // tile, large, check_a, check_b
    /// Creates the pool and all eight events.
    /// @throws std::runtime_error on failure; destroy() is safe afterwards.
    void create(ze_context_handle_t ctx) {
        ze_event_pool_desc_t pd{};
        pd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        pd.count = 8;
        pd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
        ZE_CHECK(zeEventPoolCreate(ctx, &pd, 0, nullptr, &pool));
        for (uint32_t i = 0; i < 8; ++i) {
            ze_event_desc_t ed{};
            ed.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
            ed.index = i;
            ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
            ed.wait = ZE_EVENT_SCOPE_FLAG_HOST;
            ZE_CHECK(zeEventCreate(pool, &ed, &ev[i / 4][i % 4]));
        }
    }
    /// Destroys whatever create() made; safe after a partial create.
    void destroy() {
        for (auto& s : ev) for (auto& e : s) if (e) zeEventDestroy(e);
        if (pool) zeEventPoolDestroy(pool);
    }
};

} // namespace

// ---------------------------------------------------------------------------
struct ScmEngine::Impl {
    Context ctx;
    std::unique_ptr<Module> mod;
    std::unique_ptr<Kernel> k_tile, k_large, k_ca, k_cb;
    Events ev;
    double timer_hz = 0;

    // configuration
    ScmGeo geo;
    uint32_t pshift = 0;                   // pending entry: cr << pshift | word << 32 | miss mask
    uint32_t wpl = SCM_A_WPL, span = SCM_A3_SPAN;   // phase-A item width / plan span (module-checked)
    uint64_t limit = 0, start = 0, N0 = 0;
    uint32_t pmax = 0, TB = 0, dlt = 0;
    uint32_t split = 0, own_end = 0, j_own0 = 0;
    uint32_t stride64 = 0, stride32 = 0, chk_stride = 0;
    uint32_t items_pad_max = 0, pending_cap = 0, leftover_cap = 0;
    uint32_t cb_groups = 640;
    bool write_checked = false;
    const std::vector<uint32_t>* check_only = nullptr;   // validation: checker on this map

    // host tables
    PrimeCacheView pv;
    const uint32_t* primes = nullptr;
    uint32_t count = 0;
    std::vector<uint32_t> cand_primes;     // odd primes <= pmax + margin (CPU fallback too)
    PlanSCM plan;
    GroupsSCM groups;
    std::vector<uint32_t> fstate;          // [k * split + j] first hit of the CURRENT segment
    std::vector<uint32_t> dltmod;          // dlt mod p (tile primes)
    std::vector<uint32_t> gidx;            // index of each mask-group prime in primes[]

    // device buffers
    uint32_t* d_primes = nullptr;
    uint32_t* d_bases = nullptr;
    uint32_t* d_gaps = nullptr;
    uint32_t large_count = 0;
    uint16_t* d_inv = nullptr;
    uint64_t* d_gmask = nullptr;
    uint32_t* d_gdesc = nullptr;
    uint16_t* d_cands = nullptr;
    uint32_t* d_meta = nullptr;
    uint32_t* d_rest = nullptr;
    uint32_t* d_rtab = nullptr;
    uint64_t* d_map[2] = {};
    uint32_t* h_fbt[2] = {};               // shared
    uint32_t* h_gy[2] = {};                // shared
    uint32_t* h_ivalid[2] = {};            // shared
    uint64_t* d_pending[2] = {};
    uint32_t* h_pending_n[2] = {};         // shared
    uint64_t* d_left[2] = {};
    uint32_t* h_left_n[2] = {};            // shared
    uint64_t* d_checked[2] = {};

    // per-slot segment record
    struct Seg { uint64_t N = 0; uint32_t dl = 0, T = 0, tpc = 0; bool live = false, check_only = false; } seg[2];

    ~Impl() {
        ev.destroy();
        void* bufs[] = {d_primes, d_bases, d_gaps, d_inv, d_gmask, d_gdesc, d_cands, d_meta,
                        d_rest, d_rtab};
        for (void* p : bufs) Context::free_mem(ctx.context(), p);
        for (int s = 0; s < 2; ++s) {
            void* sb[] = {d_map[s], h_fbt[s], h_gy[s], h_ivalid[s], d_pending[s], h_pending_n[s],
                          d_left[s], h_left_n[s], d_checked[s]};
            for (void* p : sb) Context::free_mem(ctx.context(), p);
        }
    }

    /// Copies @p v into a new device buffer (at least 64 bytes) through a
    /// shared staging buffer, synchronously.
    /// @return the device buffer; the caller stores it in a d_* member,
    ///         which ~Impl frees.
    /// @throws std::runtime_error on allocation or copy failure.
    template <typename T>
    T* upload(const std::vector<T>& v) {
        const size_t bytes = std::max<size_t>(v.size() * sizeof(T), 64);
        T* d = static_cast<T*>(ctx.alloc_device(bytes));
        void* sh = ctx.alloc_shared(bytes);
        std::memcpy(sh, v.data(), v.size() * sizeof(T));
        ZE_CHECK(zeCommandListAppendMemoryCopy(ctx.list(), d, sh, bytes, nullptr, 0, nullptr));
        ctx.host_sync();
        Context::free_mem(ctx.context(), sh);
        return d;
    }
};

ScmEngine::ScmEngine(uint64_t start_even, uint64_t limit, const ScmOptions& o)
    : im(new Impl)
{
    Impl& I = *im;
    const bool itrace = std::getenv("GB_INIT_TRACE") != nullptr;
    auto tmark = clock_::now();
    auto mark = [&](const char* what) {
        if (!itrace) return;
        std::fprintf(stderr, "  init %-24s %7.1f ms\n", what, secs(tmark) * 1e3);
        tmark = clock_::now();
    };
    I.geo = scm_geo(o.M ? o.M : uint32_t(env_u64("GB_SCM_M", SCM_M_DEFAULT)));
    const ScmGeo& G = I.geo;
    I.limit = limit;
    I.start = start_even;
    I.pmax = o.pmax ? o.pmax : uint32_t(env_u64("GB_SCM_PMAX", G.pmax));
    I.TB = I.pmax / G.M + SCM_MAXD + 2;
    I.dlt = o.dlt ? o.dlt : uint32_t(env_u64("GB_SCM_DELTA", G.dlt));
    if (I.dlt < 4096 || I.dlt > (1u << 26)) throw std::runtime_error("sparse: segment delta out of range");
    {   // pending entries: cr << pshift | word << 32 | miss mask must fit 64 bits
        uint32_t wb = 0, cb = 0;
        while ((1u << wb) < (I.dlt + 31) / 32) ++wb;
        while ((1u << cb) < G.CR()) ++cb;
        if (32 + wb + cb > 64) throw std::runtime_error("sparse: segment delta too large for the pending format");
        I.pshift = 32 + wb;
    }
    I.N0 = start_even - start_even % G.M;
    I.write_checked = o.write_checked;
    require_supported_limit(limit);
    const uint32_t sqrt_limit = base_prime_bound(limit);
    if (I.N0 < uint64_t(G.M) * I.TB + 2ull * sqrt_limit + 1000000ull)
        throw std::runtime_error("sparse map needs start well above the candidate range "
                                 "and sqrt(limit) (use the dense path for small ranges)");

    // ---- base primes ----
    I.pv = PrimeCacheView::load(std::max<uint64_t>(sqrt_limit, 64),
                                o.prime_cache ? o.prime_cache : "base_primes.cache");
    I.primes = I.pv.data();
    I.count = uint32_t(I.pv.size());
    mark("prime cache");
    auto idx_of = [&](uint32_t p) {
        return uint32_t(std::lower_bound(I.primes, I.primes + I.count, p) - I.primes);
    };
    I.j_own0 = idx_of(64);
    I.own_end = std::max(I.j_own0, idx_of(uint32_t(env_u64("GB_SCM_OWN_P", 512))));
    I.split = o.split_p ? idx_of(o.split_p)
                        : idx_of(uint32_t(env_u64("GB_SCM_SPLIT_P", 131072)));
    I.split = std::max(I.split, I.own_end);
    I.split = std::min(I.split, I.count);

    // ---- candidates + plan ----
    I.wpl = uint32_t(env_u64("GB_SCM_WPL", SCM_A_WPL));
    I.span = uint32_t(env_u64("GB_SCM_SPAN", G.span));
    I.cand_primes = odd_primes_up_to(I.pmax + (SCM_MAXD + 2) * G.M);
    mark("candidate primes");
    {   // checker plan: disk cache keyed by every input, validated either way
        const double target = env_f("GB_SCM_TARGET", o.target);
        const double w0 = env_f("GB_SCM_SEGW0", 3.0), w1 = env_f("GB_SCM_SEGW1", 1.0);
        // phase-B candidates per class: an even that failed ~64 phase-A tests
        // passes 384 more with probability ~0.878^384 ~ 2e-22 (beyond: CPU)
        const uint32_t rcap = uint32_t(env_u64("GB_SCM_RESTCAP", 384));
        std::string key = "scm-plan-v1 M=" + std::to_string(G.M) + " res=";
        for (uint32_t c : G.res) key += std::to_string(c) + ",";
        char kb[256];
        std::snprintf(kb, sizeof kb, " pmax=%u TB=%u target=%.17g w0=%.17g w1=%.17g rcap=%u span=%u maxd=%u meta=%u",
                      I.pmax, I.TB, target, w0, w1, rcap, I.span, SCM_MAXD, SCM_META);
        key += kb;
        const std::string cp = o.prime_cache ? o.prime_cache : "base_primes.cache";
        const size_t sl = cp.find_last_of('/');
        char hx[32];
        std::snprintf(hx, sizeof hx, "%016llx", (unsigned long long)fnv64(key.data(), key.size()));
        const std::string path = (sl == std::string::npos ? std::string() : cp.substr(0, sl + 1)) +
                                 "scm_plan." + std::to_string(G.M) + "." + hx + ".bin";
        const bool use_cache = env_u64("GB_SCM_PLAN_CACHE", 1) != 0;
        bool cached = use_cache && load_plan(path, key, I.plan);
        if (!cached)
            I.plan = make_plan_scm(G, I.cand_primes, I.pmax, I.TB, target, w0, w1, rcap, I.span);
        mark(cached ? "plan (cache)" : "plan (built)");
        std::vector<uint8_t> bits(I.pmax / 16 + 1, 0);    // odd primes <= pmax
        for (uint32_t p : I.cand_primes) if (p <= I.pmax) bits[p >> 4] |= uint8_t(1u << ((p >> 1) & 7u));
        validate_plan_scm(G, I.plan, I.TB, I.pmax, [&](uint32_t x) {
            return (x & 1u) && ((bits[x >> 4] >> ((x >> 1) & 7u)) & 1u);
        });
        mark("plan validation");
        if (!cached && use_cache) save_plan(path, key, I.plan);
    }
    I.groups = make_groups_scm(G.M);

    // ---- geometry ----
    const uint32_t Tmax = I.dlt + I.TB;
    {   // scm_large does its first-hit and mark arithmetic in 32 bits:
        // rho + b * delta < p + M^2 and t + p < T + p must not wrap (fail closed;
        // p < ~3.39e9 for M = 30030, i.e. limit < ~1.15e19)
        const uint64_t p_top = I.split < I.count ? I.primes[I.count - 1] : 0;
        const uint64_t head = std::max<uint64_t>(uint64_t(G.M) * G.M, Tmax);
        if (p_top + head >= (uint64_t(1) << 32))
            throw std::runtime_error("sparse: largest base prime too large for the 32-bit "
                                     "large-prime arithmetic (limit too high)");
    }
    const uint32_t tpc_max = (Tmax + SCM_TILE_BITS - 1) / SCM_TILE_BITS;
    I.stride64 = tpc_max * SCM_TILE_WORDS + 64;         // + zeroed pad (window reads)
    I.stride32 = 2 * I.stride64;
    I.chk_stride = I.dlt / 64 + 16;
    const uint32_t per_item = I.wpl * 32;
    I.items_pad_max = ((I.dlt + per_item - 1) / per_item + 255) / 256 * 256;
    // pending words: ~0.02% of the evens at the default plan target (64 tests,
    // M = 30030; 0.03% for M = 2310); 0.125% leaves a 4-6x margin (an overflow
    // fails closed) at 1.2 GB per slot for M = 30030
    const uint64_t seg_evens = uint64_t(I.dlt) * G.CR();
    I.pending_cap = uint32_t(std::min<uint64_t>(std::max<uint64_t>(seg_evens / 800, 1u << 20),
                                                0x7FFFFFFFu));
    I.pending_cap = uint32_t(env_u64("GB_SCM_PCAP", I.pending_cap));
    I.leftover_cap = uint32_t(env_u64("GB_SCM_LCAP", 1u << 20));
    I.cb_groups = uint32_t(env_u64("GB_SCM_BGROUPS", 640));

    // ---- L0 ----
    I.mod.reset(new Module(I.ctx, std::getenv("GB_SCM_SPV") ? std::string(std::getenv("GB_SCM_SPV"))
                                  : !o.spv.empty() ? o.spv : std::string(G.spv)));
    {   // geometry handshake (fail closed on a module built for another M / K / residues)
        Kernel kg(*I.mod, "scm_geometry");
        kg.set_group_size(1);
        uint32_t* out = static_cast<uint32_t*>(I.ctx.alloc_shared(64 * 4));
        std::memset(out, 0xFF, 64 * 4);
        kg.set_arg(0, sizeof(out), &out);
        kg.launch(I.ctx.list(), 1);
        I.ctx.host_sync();
        bool ok = out[0] == G.M && out[1] == G.K && G.K <= 48;
        for (uint32_t k = 0; ok && k < G.K; ++k) ok = out[2 + k] == G.res[k];
        const uint32_t* c = out + 2 + G.K;
        const bool layout = ok && c[0] == I.wpl && c[1] == I.span && c[2] == SCM_MAXD &&
                            c[3] == SCM_TILE_WORDS && c[4] == LARGE_PACK_PPI && c[5] == 32 &&
                            c[6] == SCM_MAX_GROUPS && c[7] == SCM_META;
        if (ok && !layout) {
            Context::free_mem(I.ctx.context(), out);
            throw std::runtime_error("sparse: kernel module layout constants (item width, span, "
                                     "pattern shift, tile, packing, groups, meta) do not match the host");
        }
        const uint32_t mm = out[0], mk = out[1];
        Context::free_mem(I.ctx.context(), out);
        if (!ok)
            throw std::runtime_error("sparse: kernel module geometry (M=" + std::to_string(mm) +
                                     ", K=" + std::to_string(mk) + ") does not match the host (M=" +
                                     std::to_string(G.M) + ", K=" + std::to_string(G.K) + ")");
    }
    mark("module + handshake");
    I.k_tile.reset(new Kernel(*I.mod, "scm_tile"));
    I.k_large.reset(new Kernel(*I.mod, "scm_large"));
    I.k_ca.reset(new Kernel(*I.mod, "scm_check_a"));
    I.k_cb.reset(new Kernel(*I.mod, "scm_check_b"));
    I.k_tile->set_group_size(256);
    I.k_large->set_group_size(uint32_t(env_u64("GB_SCM_LARGE_WG", 128)));
    I.k_ca->set_group_size(256);
    I.k_cb->set_group_size(256);
    I.ev.create(I.ctx.context());
    ze_device_properties_t dp{};
    dp.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
    zeDeviceGetProperties(I.ctx.device(), &dp);
    I.timer_hz = double(dp.timerResolution);

    // ---- uploads ----
    {
        std::vector<uint32_t> tp(I.primes, I.primes + I.split);
        I.d_primes = I.upload(tp);
        const Packed pk = pack_primes(I.primes, I.split, I.count);
        I.large_count = I.count - I.split;
        std::vector<uint32_t> gaps4((pk.gaps.size() + 3) / 4, 0);
        std::memcpy(gaps4.data(), pk.gaps.data(), pk.gaps.size());
        I.d_gaps = I.upload(gaps4);
        I.d_bases = I.upload(pk.bases);
        std::vector<uint16_t> inv(G.M, 0);
        for (uint32_t b = 1; b < G.M; ++b)
            if (G.unit(b)) inv[b] = uint16_t(scm_invmod(b, G.M));
        I.d_inv = I.upload(inv);
        I.d_gmask = I.upload(I.groups.words);
        I.d_gdesc = I.upload(I.groups.desc);
        std::vector<uint16_t> c = I.plan.cands; if (c.empty()) c.push_back(0);
        I.d_cands = I.upload(c);
        I.d_meta = I.upload(I.plan.meta);
        std::vector<uint32_t> rs = I.plan.rest; if (rs.empty()) rs.push_back(0);
        I.d_rest = I.upload(rs);
        I.d_rtab = I.upload(I.plan.rtab);
    }
    mark("uploads");
    const size_t map_bytes = size_t(G.K) * I.stride64 * 8;
    for (int s = 0; s < 2; ++s) {
        I.d_map[s] = static_cast<uint64_t*>(I.ctx.alloc_device(map_bytes));
        const uint8_t z = 0;
        ZE_CHECK(zeCommandListAppendMemoryFill(I.ctx.list(), I.d_map[s], &z, 1, map_bytes,
                                               nullptr, 0, nullptr));
        I.h_fbt[s] = static_cast<uint32_t*>(I.ctx.alloc_shared(size_t(G.K) * std::max(I.split, 1u) * 4));
        I.h_gy[s] = static_cast<uint32_t*>(I.ctx.alloc_shared(G.K * SCM_MAX_GROUPS * 4));
        I.h_ivalid[s] = static_cast<uint32_t*>(I.ctx.alloc_shared(2 * G.CR() * 4));
        I.d_pending[s] = static_cast<uint64_t*>(I.ctx.alloc_device(size_t(I.pending_cap) * 8));
        I.h_pending_n[s] = static_cast<uint32_t*>(I.ctx.alloc_shared(64));
        I.d_left[s] = static_cast<uint64_t*>(I.ctx.alloc_device(size_t(I.leftover_cap) * 8));
        I.h_left_n[s] = static_cast<uint32_t*>(I.ctx.alloc_shared(64));
        if (I.write_checked) {
            const size_t cb = size_t(G.CR()) * I.chk_stride * 8;
            I.d_checked[s] = static_cast<uint64_t*>(I.ctx.alloc_device(cb));
        }
    }
    I.ctx.host_sync();

    mark("slot buffers");
    // ---- first-hit state of the tile primes at segment 0 ----
    I.fstate.assign(size_t(G.K) * std::max(I.split, 1u), 0);
    I.dltmod.assign(std::max(I.split, 1u), 0);
    const uint64_t Q0 = I.N0 - uint64_t(G.M) * I.TB;
    par_for(0, I.split, 16, [&](uint32_t a, uint32_t b) {
        for (uint32_t j = a; j < b; ++j) {
            const uint32_t p = I.primes[j];
            if (G.M % p == 0) {                // primes of M: no multiples in any class
                for (uint32_t k = 0; k < G.K; ++k) I.fstate[size_t(k) * I.split + j] = 0xFFFFFFFFu;
                continue;
            }
            const uint64_t minv = scm_invmod(G.M % p, p);
            for (uint32_t k = 0; k < G.K; ++k)
                I.fstate[size_t(k) * I.split + j] = scm_first_hit(Q0, G.res[k], p, minv);
            I.dltmod[j] = I.dlt % p;
        }
    });
    mark("first hits");
    for (const auto& grp : I.groups.primes)
        for (uint32_t p : grp) {
            const uint32_t j = idx_of(p);
            if (j >= I.split || I.primes[j] != p)
                throw std::runtime_error("sparse: mask-group prime outside the tile primes");
        }
}

ScmEngine::~ScmEngine() = default;

const PlanSCM& ScmEngine::plan() const { return im->plan; }
const ScmGeo& ScmEngine::geo() const { return im->geo; }
uint32_t ScmEngine::pshift() const { return im->pshift; }
uint32_t ScmEngine::dlt() const { return im->dlt; }
uint32_t ScmEngine::tb() const { return im->TB; }
uint64_t ScmEngine::n0() const { return im->N0; }
uint32_t ScmEngine::split_index() const { return im->split; }
uint32_t ScmEngine::stride32() const { return im->stride32; }
uint32_t ScmEngine::chk_stride() const { return im->chk_stride; }
const uint32_t* ScmEngine::base_primes() const { return im->primes; }
uint32_t ScmEngine::base_count() const { return im->count; }
const std::vector<uint32_t>& ScmEngine::cand_primes() const { return im->cand_primes; }

void ScmEngine::enqueue_check_only(uint64_t s, const std::vector<uint32_t>& map)
{
    im->check_only = &map;
    try { enqueue(s); } catch (...) { im->check_only = nullptr; throw; }
    im->check_only = nullptr;
}

// Enqueue segment s (N_s = N0 + s D) on slot s & 1. The slot must be idle
// (collected). Advances the tile first-hit state by one segment.
void ScmEngine::enqueue(uint64_t s)
{
    Impl& I = *im;
    const uint32_t slot = uint32_t(s & 1);
    const uint64_t D = uint64_t(I.geo.M) * I.dlt;
    const uint64_t N = I.N0 + s * D;
    if (N > I.limit) throw std::runtime_error("sparse: segment beyond limit");
    uint32_t dl = I.dlt;
    if ((I.limit - N) / I.geo.M + 1 < dl) dl = uint32_t((I.limit - N) / I.geo.M + 1);
    const uint32_t T = dl + I.TB;
    const uint32_t tpc = (T + SCM_TILE_BITS - 1) / SCM_TILE_BITS;
    const uint64_t Q = N - uint64_t(I.geo.M) * I.TB;
    Impl::Seg& g = I.seg[slot];
    g.N = N; g.dl = dl; g.T = T; g.tpc = tpc; g.live = true; g.check_only = I.check_only != nullptr;

    // ---- host tables (slot idle: last read two segments ago, collected) ----
    std::memcpy(I.h_fbt[slot], I.fstate.data(), size_t(I.geo.K) * I.split * 4);
    for (uint32_t k = 0; k < I.geo.K; ++k) {
        const uint32_t* f = &I.fstate[size_t(k) * I.split];
        scm_group_offsets(I.groups, [&](uint32_t p) {
            const uint32_t j = uint32_t(std::lower_bound(I.primes, I.primes + I.split, p) - I.primes);
            return f[j];
        }, I.h_gy[slot] + k * SCM_MAX_GROUPS);
    }
    for (uint32_t cr = 0; cr < I.geo.CR(); ++cr) {
        const uint64_t n0 = N + 2 * cr;              // i = 0
        uint64_t ilo = 0, ihi = 0;
        if (I.limit >= n0) ihi = std::min<uint64_t>(dl, (I.limit - n0) / I.geo.M + 1);
        if (I.start > n0) ilo = (I.start - n0 + I.geo.M - 1) / I.geo.M;
        if (ilo > ihi) ilo = ihi;
        I.h_ivalid[slot][2 * cr] = uint32_t(ilo);
        I.h_ivalid[slot][2 * cr + 1] = uint32_t(ihi);
    }
    *I.h_pending_n[slot] = 0;
    *I.h_left_n[slot] = 0;
    // advance the first-hit state to segment s + 1 (class index shift by dlt;
    // K x split ~ 150K updates: cheaper inline than spawning threads)
    for (uint32_t k = 0; k < I.geo.K; ++k) {
        uint32_t* f = &I.fstate[size_t(k) * I.split];
        for (uint32_t j = 0; j < I.split; ++j) {
            if (f[j] == 0xFFFFFFFFu) continue;
            const uint32_t p = I.primes[j], d = I.dltmod[j];
            f[j] = (f[j] >= d) ? f[j] - d : f[j] + p - d;
        }
    }

    for (int i = 0; i < 4; ++i) ZE_CHECK(zeEventHostReset(I.ev.ev[slot][i]));

    if (I.check_only) {
        if (I.check_only->size() != size_t(I.geo.K) * I.stride32)
            throw std::runtime_error("sparse: check-only map has the wrong size");
        const size_t bytes = I.check_only->size() * 4;
        void* sh = I.ctx.alloc_shared(bytes);
        std::memcpy(sh, I.check_only->data(), bytes);
        ZE_CHECK(zeCommandListAppendMemoryCopy(I.ctx.list(), I.d_map[slot], sh, bytes, nullptr, 0, nullptr));
        I.ctx.host_sync();
        Context::free_mem(I.ctx.context(), sh);
    } else {
    // ---- tile sieve ----
    Kernel& kt = *I.k_tile;
    uint32_t ngr = uint32_t(I.groups.primes.size());
    kt.set_arg(0, sizeof(I.d_map[slot]), &I.d_map[slot]);
    kt.set_arg(1, 4, &I.stride64);
    kt.set_arg(2, 4, &tpc);
    kt.set_arg(3, 4, &T);
    kt.set_arg(4, sizeof(I.d_primes), &I.d_primes);
    kt.set_arg(5, sizeof(I.h_fbt[slot]), &I.h_fbt[slot]);
    kt.set_arg(6, 4, &I.j_own0);
    kt.set_arg(7, 4, &I.own_end);
    kt.set_arg(8, 4, &I.split);
    kt.set_arg(9, sizeof(I.d_gmask), &I.d_gmask);
    kt.set_arg(10, sizeof(I.d_gdesc), &I.d_gdesc);
    kt.set_arg(11, 4, &ngr);
    kt.set_arg(12, sizeof(I.h_gy[slot]), &I.h_gy[slot]);
    kt.set_local_arg(13, size_t(SCM_TILE_WORDS) * 8);
    kt.launch(I.ctx.list(), I.geo.K * tpc, I.ev.ev[slot][0]);
    ZE_CHECK(zeCommandListAppendBarrier(I.ctx.list(), nullptr, 0, nullptr));

    // ---- large primes ----
    if (I.large_count > 0) {
        Kernel& kl = *I.k_large;
        kl.set_arg(0, sizeof(I.d_map[slot]), &I.d_map[slot]);
        kl.set_arg(1, 4, &I.stride32);
        kl.set_arg(2, sizeof(I.d_bases), &I.d_bases);
        kl.set_arg(3, sizeof(I.d_gaps), &I.d_gaps);
        kl.set_arg(4, 4, &I.large_count);
        kl.set_arg(5, 8, &Q);
        kl.set_arg(6, 4, &T);
        kl.set_arg(7, sizeof(I.d_inv), &I.d_inv);
        const uint32_t wg = uint32_t(env_u64("GB_SCM_LARGE_WG", 128));
        const uint64_t items = (uint64_t(I.large_count) + LARGE_PACK_PPI - 1) / LARGE_PACK_PPI;
        kl.launch(I.ctx.list(), uint32_t((items + wg - 1) / wg), I.ev.ev[slot][1]);
        ZE_CHECK(zeCommandListAppendBarrier(I.ctx.list(), nullptr, 0, nullptr));
    }
    }

    // ---- checker ----
    const uint32_t per_item = I.wpl * 32;
    const uint32_t items_pad = ((dl + per_item - 1) / per_item + 255) / 256 * 256;
    const uint32_t wc = I.write_checked ? 1u : 0u;
    Kernel& ka = *I.k_ca;
    ka.set_arg(0, sizeof(I.d_map[slot]), &I.d_map[slot]);
    ka.set_arg(1, 4, &I.stride32);
    ka.set_arg(2, sizeof(I.d_checked[slot]), &I.d_checked[slot]);
    ka.set_arg(3, 4, &I.chk_stride);
    ka.set_arg(4, sizeof(I.d_cands), &I.d_cands);
    ka.set_arg(5, sizeof(I.d_meta), &I.d_meta);
    ka.set_arg(6, sizeof(I.d_pending[slot]), &I.d_pending[slot]);
    ka.set_arg(7, sizeof(I.h_pending_n[slot]), &I.h_pending_n[slot]);
    ka.set_arg(8, 4, &I.pending_cap);
    ka.set_arg(9, 4, &items_pad);
    ka.set_arg(10, sizeof(I.h_ivalid[slot]), &I.h_ivalid[slot]);
    ka.set_arg(11, 4, &wc);
    ka.set_arg(12, 4, &I.pshift);
    ka.launch(I.ctx.list(), uint32_t(uint64_t(I.geo.CR()) * items_pad / 256), I.ev.ev[slot][2]);
    ZE_CHECK(zeCommandListAppendBarrier(I.ctx.list(), nullptr, 0, nullptr));
    Kernel& kb = *I.k_cb;
    kb.set_arg(0, sizeof(I.d_map[slot]), &I.d_map[slot]);
    kb.set_arg(1, 4, &I.stride32);
    kb.set_arg(2, sizeof(I.d_checked[slot]), &I.d_checked[slot]);
    kb.set_arg(3, 4, &I.chk_stride);
    kb.set_arg(4, sizeof(I.d_rest), &I.d_rest);
    kb.set_arg(5, sizeof(I.d_rtab), &I.d_rtab);
    kb.set_arg(6, sizeof(I.d_pending[slot]), &I.d_pending[slot]);
    kb.set_arg(7, sizeof(I.h_pending_n[slot]), &I.h_pending_n[slot]);
    kb.set_arg(8, 4, &I.pending_cap);
    kb.set_arg(9, sizeof(I.d_left[slot]), &I.d_left[slot]);
    kb.set_arg(10, sizeof(I.h_left_n[slot]), &I.h_left_n[slot]);
    kb.set_arg(11, 4, &I.leftover_cap);
    kb.set_arg(12, 4, &wc);
    kb.set_arg(13, 4, &I.pshift);
    kb.launch(I.ctx.list(), I.cb_groups, I.ev.ev[slot][3]);
}

// Wait for slot's segment, check the fail-closed counters, fetch its
// leftovers (resolve() verifies them on the CPU).
ScmSegResult ScmEngine::collect(uint32_t slot)
{
    Impl& I = *im;
    Impl::Seg& g = I.seg[slot];
    if (!g.live) throw std::runtime_error("sparse: collect of an idle slot");
    ZE_CHECK(zeEventHostSynchronize(I.ev.ev[slot][3], UINT64_MAX));
    ScmSegResult r;
    r.N = g.N; r.dl = g.dl; r.T = g.T;
    double t[4] = {};
    for (int i = 0; i < 4; ++i) {
        ze_kernel_timestamp_result_t ts{};
        if ((i == 1 && I.large_count == 0) || (i < 2 && g.check_only)) continue;
        if (zeEventQueryKernelTimestamp(I.ev.ev[slot][i], &ts) == ZE_RESULT_SUCCESS && I.timer_hz > 0) {
            const int64_t d = int64_t(ts.context.kernelEnd) - int64_t(ts.context.kernelStart);
            if (d > 0) t[i] = double(d) / I.timer_hz;
        }
    }
    r.tile_s = t[0]; r.large_s = t[1]; r.check_a_s = t[2]; r.check_b_s = t[3];
    r.pending = *I.h_pending_n[slot];
    if (r.pending > I.pending_cap && !std::getenv("GB_SCM_TIMING_ONLY"))   // timing experiments only
        throw std::runtime_error("sparse: phase-A pending overflow (" + std::to_string(r.pending) + ")");
    const uint32_t nl = *I.h_left_n[slot];
    if (nl > I.leftover_cap) throw std::runtime_error("sparse: leftover overflow");
    r.leftovers = nl;
    if (nl) {
        r.left.resize(nl);
        ZE_CHECK(zeCommandListAppendMemoryCopy(I.ctx.list(), r.left.data(), I.d_left[slot],
                                               size_t(nl) * 8, nullptr, 0, nullptr));
        I.ctx.host_sync();
    }
    g.live = false;
    return r;
}

void ScmEngine::resolve(const ScmSegResult& r, RidgeTracker* ridges) const
{
    for (uint64_t e : r.left) {
        const uint32_t cr = uint32_t(e >> 32), i = uint32_t(e);
        const uint64_t n = r.N + 2ull * cr + uint64_t(im->geo.M) * i;
        bool ok = false;
        for (uint32_t p : im->cand_primes) {
            if (p >= n) break;
            if (is_prime_u64(n - p)) {
                ok = true;
                if (ridges) ridges->consider(n, p, r.N);   // true smallest witness
                break;
            }
        }
        if (!ok) {
            for (uint64_t p = im->cand_primes.back() + 2; p <= n / 2; p += 2) {
                if (!is_prime_u64(p)) continue;
                if (is_prime_u64(n - p)) {
                    ok = true;
                    if (ridges) ridges->consider(n, p, r.N);
                    break;
                }
            }
        }
        if (!ok) throw std::runtime_error("Goldbach verification FAILED for n=" + std::to_string(n));
    }
}

void ScmEngine::read_map(uint32_t slot, std::vector<uint32_t>& out)
{
    Impl& I = *im;
    out.resize(size_t(I.geo.K) * I.stride32);
    ZE_CHECK(zeCommandListAppendMemoryCopy(I.ctx.list(), out.data(), I.d_map[slot],
                                           out.size() * 4, nullptr, 0, nullptr));
    I.ctx.host_sync();
}
void ScmEngine::read_checked(uint32_t slot, std::vector<uint64_t>& out)
{
    Impl& I = *im;
    if (!I.d_checked[slot]) throw std::runtime_error("sparse: checked[] not allocated");
    out.resize(size_t(I.geo.CR()) * I.chk_stride);
    ZE_CHECK(zeCommandListAppendMemoryCopy(I.ctx.list(), out.data(), I.d_checked[slot],
                                           out.size() * 8, nullptr, 0, nullptr));
    I.ctx.host_sync();
}
void ScmEngine::read_pending(uint32_t slot, std::vector<uint64_t>& out)
{
    Impl& I = *im;
    const uint32_t n = std::min(*I.h_pending_n[slot], I.pending_cap);
    out.resize(n);
    if (n) {
        ZE_CHECK(zeCommandListAppendMemoryCopy(I.ctx.list(), out.data(), I.d_pending[slot],
                                               size_t(n) * 8, nullptr, 0, nullptr));
        I.ctx.host_sync();
    }
}

uint64_t ScmEngine::sample_pending(uint32_t slot, uint32_t max_evens,
                                   std::vector<uint64_t>& out)
{
    Impl& I = *im;
    ZE_CHECK(zeEventHostSynchronize(I.ev.ev[slot][3], UINT64_MAX));   // count + data stable
    out.clear();
    const uint64_t N = I.seg[slot].N;
    const uint32_t n = std::min(*I.h_pending_n[slot], I.pending_cap);
    if (!n || !max_evens) return N;
    std::vector<uint64_t> ent(std::min(n, max_evens));   // >= 1 pending even each
    ZE_CHECK(zeCommandListAppendMemoryCopy(I.ctx.list(), ent.data(), I.d_pending[slot],
                                           ent.size() * 8, nullptr, 0, nullptr));
    I.ctx.host_sync();
    out.reserve(ent.size());
    for (uint64_t e : ent) {                 // decode: cr << pshift | word << 32 | miss mask
        const uint32_t cr = uint32_t(e >> I.pshift);
        const uint32_t w = uint32_t(e >> 32) & ((1u << (I.pshift - 32u)) - 1u);
        uint32_t miss = uint32_t(e);         // bit b of word w = even index 32w + b
        while (miss) {
            const uint32_t i = 32u * w + uint32_t(__builtin_ctz(miss));
            miss &= miss - 1u;
            out.push_back(N + 2ull * cr + uint64_t(I.geo.M) * i);
            if (out.size() >= max_evens) return N;
        }
    }
    return N;
}

// ---------------------------------------------------------------------------
RunStats run_gpu_sparse(uint64_t limit, uint64_t segment_even, const std::string& spv_path,
                        bool verbose, uint64_t start_even, uint32_t split_p,
                        const char* prime_cache_path, bool collect_segments)
{
    require_supported_limit(limit);   // before Level Zero: fail closed with no GPU work
    RunStats stats;
    const auto t0 = clock_::now();
    if (start_even < 4) start_even = 4;
    if (start_even & 1) ++start_even;
    if (limit & 1) --limit;
    if (start_even > limit) throw std::runtime_error("sparse: start > limit");
    stats.limit = limit;
    stats.start = start_even;
    ScmOptions o;
    o.spv = spv_path;
    o.prime_cache = prime_cache_path;
    o.split_p = split_p;
    if (segment_even && segment_even != DEFAULT_SEGMENT_EVEN)
        o.dlt = uint32_t(std::max<uint64_t>(
            segment_even / scm_geo(uint32_t(env_u64("GB_SCM_M", SCM_M_DEFAULT))).M, 4096));
    ScmEngine eng(start_even, limit, o);
    stats.init_seconds = secs(t0);
    if (verbose || std::getenv("GB_PHASE")) {
        const PlanSCM& p = eng.plan();
        std::fprintf(stderr, "sparse: M=%u K=%u pmax=%u TB=%u dlt=%u (D=%llu) split=%u "
                     "plan tests %.1f funnels %.1f segs %.2f cost %.1f max member %u, rest %zu\n",
                     eng.geo().M, eng.geo().K, p.pmax, eng.tb(), eng.dlt(),
                     (unsigned long long)(uint64_t(eng.geo().M) * eng.dlt()), eng.split_index(),
                     p.mean_tests, p.mean_funnels, p.mean_segs, p.mean_cost, p.max_member,
                     p.rest.size());
    }
    const uint64_t D = uint64_t(eng.geo().M) * eng.dlt();
    stats.segment_used = D;                       // reported as --csv / --json segment
    // Ridge tracking on the sparse path: the class map stores only K of the
    // phi(M) prime-bearing residue classes, so the checker's own witnesses
    // are usable-class witnesses (~4 decades larger than the true smallest
    // prime) — NOT ridges. True ridges are recorded from two CPU-side
    // ascending Miller-Rabin scans: the leftover fallback (normally empty)
    // and, per GB_RIDGE_SCAN, a sampled head of each segment's phase-A
    // pending list (the scan runs between segments; the GPU idles for it,
    // ~0.5% at the default 64 evens). Dense runs track every pending even.
    RidgeTracker ridges(&eng.cand_primes());
    const char* ridge_file = std::getenv("GB_RIDGE_FILE");
    if (!ridge_file || !*ridge_file) ridge_file = "goldbach_ridges.csv";
    uint32_t ridge_scan = 0;
    if (ridges.active()) {
        ridges.load(ridge_file);
        ridge_scan = uint32_t(std::min<uint64_t>(env_u64("GB_RIDGE_SCAN", 64), 1u << 22));
        if (verbose)
            std::fprintf(stderr, "ridge tracking: %u pending evens/segment sampled for "
                         "exact ridges (GB_RIDGE_SCAN, 0 = off) + CPU leftovers\n",
                         ridge_scan);
    }
    const auto tg = clock_::now();
    uint64_t nseg = 0;
    for (uint64_t N = eng.n0(); N <= limit; N += D) ++nseg;
    const bool trace = verbose || std::getenv("GB_PHASE") != nullptr;
    auto take = [&](const ScmSegResult& r) {
        eng.resolve(r, &ridges);
        ridges.save(ridge_file);
        stats.sieve_seconds += r.tile_s;
        stats.large_seconds += r.large_s;
        stats.verify_seconds += r.check_a_s + r.check_b_s;
        stats.leftovers += r.leftovers;
        if (trace)
            std::fprintf(stderr, "  seg N=%llu dl=%u: tile %.3f large %.3f A %.3f B %.3f ms, "
                         "pending %u (%.4f%%) left %u\n",
                         (unsigned long long)r.N, r.dl, r.tile_s * 1e3, r.large_s * 1e3,
                         r.check_a_s * 1e3, r.check_b_s * 1e3, r.pending,
                         100.0 * r.pending / (double(r.dl) * eng.geo().CR()), r.leftovers);
        if (collect_segments) {
            SegmentStats s;
            s.seg_start = r.N; s.seg_end = std::min<uint64_t>(r.N + D - 2, limit);
            s.evens = uint64_t(r.dl) * eng.geo().CR();
            s.sieve_seconds = r.tile_s; s.large_seconds = r.large_s;
            s.check_seconds = r.check_a_s + r.check_b_s;
            s.leftovers = r.leftovers;
            stats.per_segment.push_back(s);
        }
    };
    // Exact ridge for sampled pending evens: ascending scan over the
    // candidate primes (first prime q = n - p wins = true smallest witness).
    // A small-prime prefilter kills most composite q cheaply. Never throws:
    // these evens are already verified, the scan only hunts ridges, and
    // consider() re-validates every record exactly before storing it.
    std::vector<uint64_t> scan_evens;
    auto ridge_scan_seg = [&](uint32_t slot) {
        const uint64_t N = eng.sample_pending(slot, ridge_scan, scan_evens);
        for (uint64_t n : scan_evens) {
            for (uint32_t p : eng.cand_primes()) {
                if (p >= n) break;
                const uint64_t q = n - p;
                if ((q % 3u) && (q % 5u) && (q % 7u) && (q % 11u) && (q % 13u) &&
                    is_prime_u64(q)) {
                    ridges.consider(n, p, N);
                    break;
                }
            }
        }
    };
    for (uint64_t s = 0; s < nseg; ++s) {
        // Sample segment s-1 BEFORE enqueueing s: the queue holds nothing new,
        // so sample_pending's copy + host_sync waits only for already-finished
        // work and the scan cost becomes a bounded GPU idle gap, never a
        // pipeline stall.
        if (ridge_scan && s >= 1) ridge_scan_seg(uint32_t((s - 1) & 1));
        eng.enqueue(s);
        if (s >= 1) take(eng.collect(uint32_t((s - 1) & 1)));
        if (verbose && s % 10 == 0)
            std::printf("  segment %llu / %llu\n", (unsigned long long)s, (unsigned long long)nseg);
    }
    if (nseg) {
        take(eng.collect(uint32_t((nseg - 1) & 1)));
        if (ridge_scan) {                          // last segment: queue drained
            ridge_scan_seg(uint32_t((nseg - 1) & 1));
            ridges.save(ridge_file);               // no further take() to save it
        }
    }
    stats.segments = nseg;
    stats.gpu_seconds = secs(tg);
    stats.total_seconds = secs(t0);
    const uint64_t evens = (limit - start_even) / 2 + 1;
    stats.evens_per_second = stats.gpu_seconds > 0 ? double(evens) / stats.gpu_seconds : 0;
    stats.sync_seconds = std::max(0.0, stats.gpu_seconds - stats.sieve_seconds -
                                       stats.large_seconds - stats.verify_seconds);
    stats.ridge_min    = ridges.active() ? ridges.floor() : 0;
    stats.ridge_found  = ridges.appends();
    stats.ridge_dropped = ridges.dropped();
    stats.ridge_top    = ridges.top();
    return stats;
}

} // namespace gb
