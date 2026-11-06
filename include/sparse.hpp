#pragma once
// Sparse class map (v2.0, `--sparse`; kernels/sparse.cl, src/sparse.cpp).
//
// Every even n needs ONE certified prime q = n - p, so the map does not have
// to cover every odd q: it holds only the q in K residue classes mod M (the
// geometry: M = 2310 with K = 8 classes, 0.35% of the integers, or M = 30030
// with K = 12, 0.04%, instead of the 26.7% of the mod-30 map). The classes are
// chosen so that every even class r = n mod M can use at least one of them
// (q = n - p with p = r - c (mod M) a unit, i.e. p can be prime): for every
// residue vector of n mod the odd primes of M there is a class whose residues
// all differ from n's. The candidate side then draws p from a wider range
// (p <= pmax, ~300-420 candidates per residue class, instead of p <= 9781): each
// test still succeeds with probability ~ (M / phi(M)) / ln n, so the number of
// tests an even needs is the same as with the full map (CPU simulation on an
// exact map at 4e18, M = 2310: mean 8.8 tests vs 11.2, max 130, none
// unresolved over 1e7 evens) — but the sieve marks 77x (2310) or 670x
// (30030) fewer numbers.
//
// Soundness does not depend on any of this: an even is verified only by a
// candidate prime p and a map bit q = n - p that is set, and a set bit is a
// prime (the map is an exact sieve of its classes by every base prime); evens
// the GPU cannot settle go to the CPU fallback (deterministic Miller-Rabin
// over all p), and a q that is prime but cleared (a base prime inside the q
// range at low ranges) only costs a candidate.
//
// Geometry of segment s (all in integers):
//   evens n in [N_s, N_s + D), N_s = N_0 + s D, N_0 = 0 (mod M), D = M * dlt
//   class k array bit t  <->  q = Q_s + c_k + M t,  Q_s = N_s - M * TB
//   t in [0, T), T = dlt + TB (TB covers the candidate range p <= pmax)
//   even class r (r even, cr = r / 2): n = N_s + r + M i, i in [0, dlt)
//   candidate p = s + M a (s = r - c_k mod M): q = n - p lies in class k at
//   t = i + U, U = TB - a - [r < c_k] (>= 1 for a <= TB - 2): no division.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace gb {

inline constexpr uint32_t SCM_MAXD = 4;                // largest pattern shift: MUST match the kernel
inline constexpr uint32_t SCM_MAXSEG = 15;             // plan segments per class
inline constexpr uint32_t SCM_META = 48;               // meta u32 per class: n_seg, 3 per segment
inline constexpr uint32_t SCM_TILE_WORDS = 2048;       // u64 words per SLM tile (16 KB)
inline constexpr uint32_t SCM_TILE_BITS = SCM_TILE_WORDS * 64;
inline constexpr uint32_t SCM_MAX_GROUPS = 8;          // kernel MAX_GROUPS
inline constexpr uint32_t SCM_A_WPL = 32;              // phase-A item width (u32 words): MUST match A3_WPL
inline constexpr uint32_t SCM_A3_SPAN = 12;            // phase-A segment U span (words): MUST match A3_SPAN

// Modulus, class residues (sorted) and the defaults that go with them. The
// kernel module is compiled per modulus (-DSCM_M / SCM_K / SCM_RES_LIST).
struct ScmGeo {
    uint32_t M = 0, K = 0;
    std::vector<uint32_t> res;
    uint32_t pmax = 0;          // candidate bound (30030: 4e7, ~420 per residue class; +1.5% over 2.6e7)
    uint32_t dlt = 0;           // evens per class per segment (map K * dlt bits: L3-resident)
    const char* spv = nullptr;  // default kernel module
    uint32_t span = SCM_A3_SPAN;  // phase-A segment U span (words): MUST match the module's A3_SPAN
    /// @return number of even classes r = n mod M (M / 2).
    uint32_t CR() const { return M / 2; }
    /// @return true iff @p x is coprime to M (a residue that can hold primes).
    bool unit(uint32_t x) const { return std::gcd(x % M, M) == 1; }
    /// Usable map classes of even class @p r (r even, < M).
    /// @return bit k set iff q = n - p in class k allows p to be a unit.
    uint32_t usable(uint32_t r) const {
        uint32_t u = 0;
        for (uint32_t k = 0; k < K; ++k)
            if (unit(r + M - res[k])) u |= 1u << k;
        return u;
    }
    /// Checks that every even class has at least one usable map class.
    /// @throws std::runtime_error naming the first uncovered class.
    void check_cover() const {
        for (uint32_t r = 0; r < M; r += 2)
            if (!usable(r))
                throw std::runtime_error("sparse map: even class " + std::to_string(r) +
                                         " has no usable class (residues not covering)");
    }
};

inline constexpr uint32_t SCM_M_DEFAULT = 30030;         // 30030: ~1.0 T evens/s vs 2310: ~0.78 T (2e12 windows at 4e18)

/// Geometry (residues and defaults) for a supported modulus.
/// @param M  2310 or 30030.
/// @post  the returned geometry passed check_cover().
/// @return the geometry for @p M.
/// @throws std::runtime_error for any other modulus.
inline ScmGeo scm_geo(uint32_t M)
{
    ScmGeo g;
    if (M == 2310) {
        // 4.2% of the evens have exactly one usable class
        g = {2310, 8, {17, 29, 229, 487, 631, 793, 1181, 1763}, 2000000, 14000000,
             "kernels/sparse_bmg.spv", 12};
    } else if (M == 30030) {
        // 6 residue tuples per residue mod 3 (a residue mod 5 repeated at
        // most twice): 0.6% of the even classes have exactly one usable class
        g = {30030, 12, {2377, 7979, 9731, 11267, 13799, 14611, 18427, 18649, 21467, 21523,
                         22823, 23911}, 40000000, 8000000, "kernels/sparse30030_bmg.spv", 8};
    } else {
        throw std::runtime_error("sparse map: modulus must be 2310 or 30030");
    }
    g.check_cover();
    return g;
}

// ---------------------------------------------------------------------------
// Checker plan. Per even class cr: segments (k, mask, funnels) for phase A;
// every funnel candidate p (its lowest member) is stored as its window offset
// U = TB - a - [r < c_k] (u16); members p + M d (mask bit d) are all prime
// and not tested before for this class. Phase B (rest) probes, per class, all
// remaining candidates p <= pmax of the usable classes in ascending p, stored
// as U | k << 28. Correctness does not depend on the plan: phase A + B test a
// subset of the candidates, and a pending even the plan cannot settle goes to
// the CPU fallback.
//
// The greedy is analytic (no map needed): every test succeeds with about the
// same probability, so it maximizes tests per cost until tests >= target. A
// segment costs its window (load + pattern OR: seg_w0 + seg_w1 * (members - 1)
// funnel units) once, and any funnel within span words of U is free of slides
// (scm_check_a reads it through a switch on the word offset), so each step
// takes the (class, mask, U window) with the most tests per cost and all of
// its funnels. Classes are independent: built in parallel.
struct PlanSCM {
    std::vector<uint16_t> cands;   // U of phase-A funnels, per segment ascending p
    std::vector<uint32_t> meta;    // CR x SCM_META: n_seg, (begin, end, k | mask << 8) x n_seg
    std::vector<uint32_t> rest;    // phase B: U | k << 28
    std::vector<uint32_t> rtab;    // [0, CR): rest offset, [CR, 2 CR): count
    uint32_t max_member = 0;       // largest p tested in phase A
    uint32_t pmax = 0;             // candidate bound (phase B)
    double mean_tests = 0, mean_cost = 0, mean_funnels = 0, mean_segs = 0;
};

/// Builds the sparse checker plan with the greedy described above, classes in
/// parallel (up to 16 threads). Deterministic for fixed inputs.
///
/// @param g            geometry (K <= 16).
/// @param primes       odd primes ascending, covering at least pmax + SCM_MAXD * M.
/// @param pmax         candidate bound: every tested p (and member) is <= pmax.
/// @param TB           window offset base; must satisfy pmax / M + 2 <= TB.
/// @param target       phase-A tests per even class to reach.
/// @param seg_w0,seg_w1  segment cost model (seg_w0 + seg_w1 * (members - 1)).
/// @param rest_cap     phase-B probes kept per class (ascending p).
/// @param span         window span in words within which funnels are free.
/// @param max_members  largest pattern popcount considered.
/// @post   the plan passes validate_plan_scm for the same g, TB and pmax.
/// @return the plan, including mean cost statistics.
/// @throws std::runtime_error if the prime list is too short, TB is too small
///         or K > 16 (or rethrows a worker exception).
inline PlanSCM make_plan_scm(const ScmGeo& g, const std::vector<uint32_t>& primes, uint32_t pmax,
                             uint32_t TB, double target = 64.0, double seg_w0 = 3.0,
                             double seg_w1 = 1.0, uint32_t rest_cap = 0xFFFFFFFFu,
                             uint32_t span = SCM_A3_SPAN, uint32_t max_members = 5)
{
    const uint32_t M = g.M, K = g.K, CR = g.CR();
    const uint32_t bound = primes.empty() ? 0 : primes.back();
    if (bound < pmax + SCM_MAXD * M) throw std::runtime_error("plan: candidate list too short");
    if (pmax / M + 2 > TB) throw std::runtime_error("plan: TB too small for pmax");
    if (K > 16) throw std::runtime_error("plan: at most 16 classes (rest entries k << 28)");
    std::vector<char> isp(bound + 1, 0);
    for (uint32_t p : primes) isp[p] = 1;
    std::vector<std::vector<uint32_t>> by_s(M);    // candidates per residue class, ascending
    for (uint32_t p : primes) if (p <= pmax && g.unit(p)) by_s[p % M].push_back(p);
    // per candidate: bit d set when p + M d is a prime <= pmax (pattern members)
    std::vector<std::vector<uint8_t>> pm_s(M);
    for (uint32_t s0 = 0; s0 < M; ++s0) {
        pm_s[s0].resize(by_s[s0].size());
        for (size_t i = 0; i < by_s[s0].size(); ++i) {
            uint8_t m = 0;
            for (uint32_t d = 0; d <= SCM_MAXD; ++d) {
                const uint64_t q = uint64_t(by_s[s0][i]) + uint64_t(M) * d;
                if (q <= pmax && isp[q]) m |= uint8_t(1u << d);
            }
            pm_s[s0][i] = m;
        }
    }
    std::vector<uint32_t> MASKS;                   // bit 0 = the funnel candidate
    for (uint32_t m = 1; m < (2u << SCM_MAXD); m += 2)
        if (uint32_t(__builtin_popcount(m)) <= max_members) MASKS.push_back(m);

    struct Local {
        std::vector<uint16_t> cands;
        std::vector<uint32_t> seg;                 // (begin, end, k | mask << 8), local indices
        std::vector<uint32_t> rest;
        uint32_t max_member = 0;
        double tests = 0, cost = 0;
        uint32_t nf = 0;
    };
    std::vector<Local> outs(CR);
    auto build_class = [&](uint32_t cr, std::vector<char>& cov, std::vector<uint32_t>& touched) {
        Local& loc = outs[cr];
        const uint32_t r = 2 * cr;
        const uint32_t use = g.usable(r);
        auto U_of = [&](uint32_t p, uint32_t k) -> uint32_t {
            return TB - p / M - (r < g.res[k] ? 1u : 0u);
        };
        auto members = [](uint32_t mask) { return double(__builtin_popcount(mask)); };
        auto seg_cost = [&](uint32_t mask) { return seg_w0 + seg_w1 * (members(mask) - 1); };
        auto ok_at = [&](uint32_t p, uint32_t mask) {
            for (uint32_t d = 0; d <= SCM_MAXD; ++d)
                if ((mask >> d) & 1u) {
                    const uint64_t q = uint64_t(p) + uint64_t(M) * d;
                    if (q > pmax || !isp[q] || cov[q]) return false;
                }
            return true;
        };
        std::vector<size_t> avail, bsel;
        double tests = 0;
        while (tests < target && loc.seg.size() / 3 < SCM_MAXSEG) {
            double best = -1; int bk = -1; uint32_t bm = 0;
            for (uint32_t k = 0; k < K; ++k) {
                if (!((use >> k) & 1u)) continue;
                const auto& L = by_s[(r + M - g.res[k]) % M];
                const auto& PM = pm_s[(r + M - g.res[k]) % M];
                for (uint32_t mask : MASKS) {
                    avail.clear();
                    for (size_t pos = 0; pos < L.size(); ++pos)
                        if ((PM[pos] & mask) == mask && ok_at(L[pos], mask)) avail.push_back(pos);
                    if (avail.empty()) continue;
                    const double mem = members(mask), sc = seg_cost(mask);
                    size_t i = 0, bi = 0, bj = 0;
                    double bv = -1;
                    for (size_t j = 0; j < avail.size(); ++j) {
                        const uint32_t uj = U_of(L[avail[j]], k) >> 5;
                        while ((U_of(L[avail[i]], k) >> 5) - uj > span) ++i;
                        const double F = double(j - i + 1);
                        // tests beyond the target are worth nothing
                        const double t = std::min(F * mem, std::max(target - tests, mem));
                        const double v = t / (sc + F);
                        if (v > bv) { bv = v; bi = i; bj = j; }
                    }
                    if (bv > best) {                   // copy the window once per (class, mask)
                        best = bv; bk = int(k); bm = mask;
                        bsel.assign(avail.begin() + bi, avail.begin() + bj + 1);
                    }
                }
            }
            if (bk < 0) break;
            const auto& L = by_s[(r + M - g.res[bk]) % M];
            const uint32_t b = uint32_t(loc.cands.size());
            for (size_t pos : bsel) {
                if (tests >= target) break;
                const uint32_t p = L[pos];
                if (!ok_at(p, bm)) continue;           // overlapping members within a run
                loc.cands.push_back(uint16_t(U_of(p, uint32_t(bk))));
                for (uint32_t d = 0; d <= SCM_MAXD; ++d)
                    if ((bm >> d) & 1u) {
                        const uint32_t q = p + M * d;
                        cov[q] = 1; touched.push_back(q);
                        loc.max_member = std::max(loc.max_member, q);
                    }
                tests += members(bm);
                loc.cost += 1.0;
                ++loc.nf;
            }
            loc.cost += seg_cost(bm);
            if (loc.cands.size() > b) {
                loc.seg.push_back(b);
                loc.seg.push_back(uint32_t(loc.cands.size()));
                loc.seg.push_back(uint32_t(bk) | (bm << 8));
            }
        }
        loc.tests = tests;
        // Phase B: untested candidates of the usable classes, ascending p.
        std::vector<std::pair<uint32_t, uint32_t>> rs;
        for (uint32_t k = 0; k < K; ++k)
            if ((use >> k) & 1u)
                for (uint32_t p : by_s[(r + M - g.res[k]) % M])
                    if (!cov[p]) rs.push_back({p, k});
        std::sort(rs.begin(), rs.end());
        for (const auto& [p, k] : rs) {
            if (loc.rest.size() >= rest_cap) break;
            loc.rest.push_back(U_of(p, k) | (k << 28));
        }
        for (uint32_t q : touched) cov[q] = 0;
        touched.clear();
    };
    {
        const uint32_t T = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        std::vector<std::thread> th;
        std::atomic<uint32_t> next{0};
        std::vector<std::exception_ptr> err(T);
        for (uint32_t t = 0; t < T; ++t)
            th.emplace_back([&, t] {
                try {
                    std::vector<char> cov(bound + 1, 0);
                    std::vector<uint32_t> touched;
                    for (uint32_t cr; (cr = next++) < CR;) build_class(cr, cov, touched);
                } catch (...) { err[t] = std::current_exception(); }
            });
        for (auto& x : th) x.join();
        for (auto& e : err) if (e) std::rethrow_exception(e);
    }
    PlanSCM plan;
    plan.pmax = pmax;
    plan.meta.assign(size_t(CR) * SCM_META, 0);
    plan.rtab.assign(size_t(2) * CR, 0);
    double sum_tests = 0, sum_cost = 0, sum_f = 0, sum_s = 0;
    for (uint32_t cr = 0; cr < CR; ++cr) {
        const Local& loc = outs[cr];
        const uint32_t off = uint32_t(plan.cands.size());
        const uint32_t ns = uint32_t(loc.seg.size() / 3);
        uint32_t* m = &plan.meta[size_t(cr) * SCM_META];
        m[0] = ns;
        for (uint32_t sg = 0; sg < ns; ++sg) {
            m[1 + 3 * sg] = loc.seg[3 * sg] + off;
            m[2 + 3 * sg] = loc.seg[3 * sg + 1] + off;
            m[3 + 3 * sg] = loc.seg[3 * sg + 2];
        }
        plan.cands.insert(plan.cands.end(), loc.cands.begin(), loc.cands.end());
        plan.rtab[cr] = uint32_t(plan.rest.size());
        plan.rtab[CR + cr] = uint32_t(loc.rest.size());
        plan.rest.insert(plan.rest.end(), loc.rest.begin(), loc.rest.end());
        plan.max_member = std::max(plan.max_member, loc.max_member);
        sum_tests += loc.tests; sum_cost += loc.cost; sum_f += loc.nf; sum_s += ns;
    }
    plan.mean_tests = sum_tests / CR;
    plan.mean_cost = sum_cost / CR;
    plan.mean_funnels = sum_f / CR;
    plan.mean_segs = sum_s / CR;
    return plan;
}

// ---------------------------------------------------------------------------
/// Plan validation: every phase-A funnel member and every phase-B probe is
/// decoded back to its candidate p (p = s + M a, s = r - c_k mod M,
/// a = TB - U - [r < c_k]) and must be a prime <= @p bound with a usable
/// class and U - d >= 1 — so a plan (fresh or loaded from the cache) can only
/// ever test real (prime p, class-k q) pairs. Runs on up to 16 threads.
///
/// @param g      geometry the plan was built for.
/// @param plan   plan to check (table sizes must match the geometry).
/// @param TB     window offset base used by the kernels.
/// @param bound  largest admissible candidate (pmax).
/// @param is_p   callable, is_p(x) == primality of x for 3 <= x <= bound;
///               must be thread-safe.
/// @post   returns normally only if every test in the plan is a genuine
///         (prime p, class-k q) pair.
/// @throws std::runtime_error ("sparse plan rejected: ...") on any violation.
template <typename IsPrime>
inline void validate_plan_scm(const ScmGeo& g, const PlanSCM& plan, uint32_t TB,
                              uint32_t bound, IsPrime&& is_p)
{
    const uint32_t M = g.M, K = g.K, CR = g.CR();
    if (plan.meta.size() != size_t(CR) * SCM_META || plan.rtab.size() != size_t(2) * CR)
        throw std::runtime_error("plan: table sizes do not match the geometry");
    std::atomic<bool> bad{false};
    std::string why;
    auto fail = [&](const std::string& w) { if (!bad.exchange(true)) why = w; };
    auto cand = [&](uint32_t cr, uint32_t use, uint32_t k, uint32_t U, uint32_t d) -> bool {
        if (k >= K || !((use >> k) & 1u)) return false;
        const uint32_t r = 2 * cr, w = r < g.res[k] ? 1u : 0u;
        if (U < d + 1 || uint64_t(U) + w > TB) return false;
        const uint64_t a = TB - U - w + d;                   // member p + M d
        const uint64_t p = (r + M - g.res[k]) % M + uint64_t(M) * a;
        return p >= 3 && p <= bound && is_p(uint32_t(p));
    };
    const uint32_t T = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (uint32_t t = 0; t < T; ++t)
        th.emplace_back([&, t] {
            for (uint32_t cr = t; cr < CR && !bad; cr += T) {
                const uint32_t use = g.usable(2 * cr);
                const uint32_t* m = &plan.meta[size_t(cr) * SCM_META];
                if (m[0] > SCM_MAXSEG) { fail("segment count"); return; }
                for (uint32_t sg = 0; sg < m[0]; ++sg) {
                    const uint32_t jb = m[1 + 3 * sg], je = m[2 + 3 * sg], km = m[3 + 3 * sg];
                    const uint32_t k = km & 0xFFu, mask = (km >> 8) & 0xFFu;
                    if (jb >= je || je > plan.cands.size() || !(mask & 1u) || (mask >> (SCM_MAXD + 1)) ||
                        (km >> 16)) { fail("segment entry"); return; }
                    for (uint32_t j = jb; j < je; ++j)
                        for (uint32_t d = 0; d <= SCM_MAXD; ++d)
                            if (((mask >> d) & 1u) && !cand(cr, use, k, plan.cands[j], d)) {
                                fail("phase-A funnel of class " + std::to_string(cr) + " is not a prime candidate");
                                return;
                            }
                }
                const uint64_t ro = plan.rtab[cr], rc = plan.rtab[CR + cr];
                if (ro + rc > plan.rest.size()) { fail("rest table"); return; }
                for (uint64_t j = ro; j < ro + rc; ++j) {
                    const uint32_t v = plan.rest[j];
                    if (!cand(cr, use, v >> 28, v & 0x0FFFFFFFu, 0)) {
                        fail("phase-B probe of class " + std::to_string(cr) + " is not a prime candidate");
                        return;
                    }
                }
            }
        });
    for (auto& x : th) x.join();
    if (bad) throw std::runtime_error("sparse plan rejected: " + why);
}

// ---------------------------------------------------------------------------
// Small-prime mask groups (the primes 13..61 not dividing M, as periodic
// whole-word masks). The canonical table of a group clears, in word w (bits
// 64 w .. 64 w + 63), every bit b with b = 0 (mod p) for a prime p of the
// group; the pattern of a class array whose marks sit at t = f_p (mod p) is
// the canonical one shifted by X = CRT(f_p) bits, i.e. by Y = X * 64^-1 (mod P)
// whole words (P = product of the group's primes, odd): word x of the class
// array = table word (x - Y) mod P.
struct GroupsSCM {
    std::vector<uint64_t> words;               // concatenated canonical tables
    std::vector<uint32_t> desc;                // (offset, period) per group
    std::vector<std::vector<uint32_t>> primes; // primes of each group
    std::vector<uint32_t> inv64;               // 64^-1 mod P per group
};

/// Builds the canonical small-prime mask groups for modulus @p M: primes
/// 13..61 not dividing M, grouped greedily (ascending) with product <=
/// @p max_period, plus 64^-1 mod each group period.
/// @return the groups (at most SCM_MAX_GROUPS).
/// @throws std::runtime_error if more than SCM_MAX_GROUPS groups result.
inline GroupsSCM make_groups_scm(uint32_t M, uint32_t max_period = 4200)
{
    static const uint32_t P[] = {13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61};
    GroupsSCM g;
    uint64_t prod = 0;
    for (uint32_t p : P) {
        if (M % p == 0) continue;               // no multiples in any class
        if (g.primes.empty() || prod * p > max_period) { g.primes.push_back({}); prod = 1; }
        g.primes.back().push_back(p);
        prod *= p;
    }
    if (g.primes.size() > SCM_MAX_GROUPS) throw std::runtime_error("sparse map: too many mask groups");
    for (const auto& grp : g.primes) {
        uint64_t per = 1;
        for (uint32_t p : grp) per *= p;
        g.desc.push_back(uint32_t(g.words.size()));
        g.desc.push_back(uint32_t(per));
        for (uint64_t w = 0; w < per; ++w) {
            uint64_t m = ~0ull;
            for (uint32_t b = 0; b < 64; ++b)
                for (uint32_t p : grp)
                    if ((64 * w + b) % p == 0) { m &= ~(1ull << b); break; }
            g.words.push_back(m);
        }
        uint64_t inv = 1;
        while ((inv * 64) % per != 1) ++inv;
        g.inv64.push_back(uint32_t(inv));
    }
    return g;
}

/// Word offset Y of each group for a class array whose prime p has its first
/// mark at t = f(p) (mod p): Y = CRT(f(p)) * 64^-1 mod P.
/// @param g   groups from make_groups_scm.
/// @param f   callable, f(p) = the first-mark class index of p (any value;
///            reduced mod p).
/// @param ys  output, g.primes.size() entries.
template <typename F>
inline void scm_group_offsets(const GroupsSCM& g, F&& f, uint32_t* ys)
{
    for (size_t gi = 0; gi < g.primes.size(); ++gi) {
        uint64_t X = 0, Pm = 1;                 // CRT: X mod Pm
        for (uint32_t p : g.primes[gi]) {
            const uint64_t r = f(p) % p;
            while (X % p != r) X += Pm;
            Pm *= p;
        }
        ys[gi] = uint32_t((X * g.inv64[gi]) % Pm);
    }
}

// Modular helpers for the host first-hit tables (64-bit exact).

/// @pre    m > 0.
/// @return a * b mod m, exact via 128-bit intermediate.
inline uint64_t scm_mulmod(uint64_t a, uint64_t b, uint64_t m)
{
    return uint64_t((unsigned __int128)a * b % m);
}
/// Modular inverse by extended Euclid.
/// @pre    0 < m < 2^63.
/// @return a^-1 mod m in [0, m).
/// @throws std::runtime_error if gcd(a, m) != 1.
inline uint64_t scm_invmod(uint64_t a, uint64_t m)
{
    int64_t t = 0, nt = 1;
    int64_t rr = int64_t(m), nr = int64_t(a % m);
    while (nr) {
        const int64_t q = rr / nr;
        int64_t tmp = t - q * nt; t = nt; nt = tmp;
        tmp = rr - q * nr; rr = nr; nr = tmp;
    }
    if (rr != 1) throw std::runtime_error("scm_invmod: not invertible");
    return uint64_t(t < 0 ? t + int64_t(m) : t);
}
/// First class index of a multiple of p in the class array q = Q + c + M t.
/// @param Q      segment base of the class arrays.
/// @param c      class residue.
/// @param p      prime not dividing M.
/// @param minv_p M^-1 mod p (scm_invmod(M % p, p)).
/// @return the unique t in [0, p) with p | (Q + c + M t).
inline uint32_t scm_first_hit(uint64_t Q, uint32_t c, uint32_t p, uint64_t minv_p)
{
    const uint64_t x = (Q % p + c) % p;               // (Q + c) mod p
    return uint32_t(scm_mulmod((p - x) % p, minv_p, p));
}

} // namespace gb
