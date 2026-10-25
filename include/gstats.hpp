#pragma once
// Exact Goldbach statistics of the dense path (v2.2; kernels check30_b,
// check30_xs8/6/4/3, check30_xb, count30 in kernels/goldbach.cl). On by
// default for runs that use the mod-30 checker far above MAX_GB_PRIME;
// GB_STATS=0 turns it off.
//
// p(n) = the smallest prime p with n - p prime ("minimal Goldbach prime").
// Everything recorded is exact (no estimate anywhere):
//   * tail, EVERY even: p(n) for every even with p(n) >= CHECK30_TAIL (3000;
//     the funnel plan keeps all phase-A members below it, so these evens all
//     reach check30_b, whose ascending probe yields p(n)); plus the first even
//     of the campaign with p(n) = p for every such p (n_min in range);
//   * full histogram, SAMPLED segments: every GB_STATS_SAMPLE-th segment of a
//     run (default 256; segment 0 always) gets p(n) for all of its evens
//     (check30_xs* bit-parallel ascending prefix + check30_xb); the counts are
//     exact for exactly those evens (evens_sampled says how many);
//   * prime and twin-prime counts in [start, limit] (every segment; count30
//     popcounts of the exact class map), primes by residue mod 30;
//   * leftovers (p(n) > MAX_GB_PRIME, never seen so far) by CPU Miller-Rabin.
//   * prime gaps (v2.4; v2.5: the walk fused into count30; after Oliveira e
//     Silva, Herzog and Pardi 2014, section 2.2): every gap q - prev(q) with
//     q in [start + 1, limit - 1], exactly: count, sums of g^k (k = 1..4,
//     128-bit), the largest gap, every gap >= GAPS30_TAIL (700) with its
//     first lower prime; the full histogram N(g) on the walked segments.
//     Per segment the host checks gaps + 1 = primes (popcounts), sum g =
//     last - first prime and the histogram's tail = the bit-parallel tail.
//
// Outputs, written only after a run PASSES (an interrupted or failed slice
// leaves nothing):
//   * one line per run appended to GB_STATS_LOG (default goldbach_stats.log),
//     ~2 KB (tail histogram, counts, new first occurrences: ~10 MB for all of
//     [4e18, 5e18] in 2e14 slices); scripts/gstats.py summarises it;
//   * GB_STATS_STATE (default goldbach_stats.state), rewritten atomically:
//     campaign totals per class n mod 30, n_min per tail p, covered ranges. A
//     run overlapping a covered range is not merged again (no double count).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <ctime>
#include <map>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include "primes.hpp"

namespace gb {

inline constexpr uint32_t STATS_K_DEFAULT = 64;    // sampled exact phase-A candidates per class
inline constexpr uint32_t STATS_SAMPLE_DEFAULT = 256;  // every 256th segment: full p(n) histogram (~0.12%)
inline constexpr uint32_t STATS_KMAX = 128;        // MUST match X_KMAX in goldbach.cl
inline constexpr uint32_t STATS_XB_SPAN = 32;      // MUST match XB_SPAN in goldbach.cl
inline constexpr uint32_t COUNT30_CW = 4;          // MUST match C30_CW in goldbach.cl (count30 words per lane)
// prime gaps (count30's tail and walk, the live paper's section 2.2)
inline constexpr uint32_t GAPS30_TAIL = 700;       // MUST match G30T_TAIL in goldbach.cl
inline constexpr uint32_t GAPS30_HBINS = 1024;     // MUST match G30_HBINS in goldbach.cl
inline constexpr uint32_t GAPS30_TCAP = 4096;      // tail entries per segment (~1 expected)
inline constexpr uint32_t GAPS30_GROUPS = 320;     // default count30 grid on walked segments (work-groups, grid-stride)
inline constexpr uint32_t GAPS30_GROUPS_MAX = 4096;
inline constexpr uint32_t GAPS30_MAX_SG = 32;      // MUST match G30_MAX_SG in goldbach.cl
// check30_xs8 / 6 / 4 / 3 item widths: MUST match XS_W8 / 6 / 4 / 3 in goldbach.cl
inline constexpr uint32_t XS_W_HOST[4] = {8, 9, 12, 14};
inline constexpr const char* XS_KERNELS[4] = {"check30_xs8", "check30_xs6", "check30_xs4", "check30_xs3"};
inline constexpr uint64_t STATS_MIN_START = 1000000;   // futile-last order exact far above 9786

/// Primes p <= 9781 for which no even n <= 4e18 has p(n) = p: their S(p)
/// (least even with p(n) = p) is unknown and exceeds 4e18 (T. Oliveira e
/// Silva, "Goldbach conjecture verification", table t0, 2012; '?' entries).
/// The first such even of a campaign that covers [4e18, n] contiguously IS
/// S(p). 9781 itself is the largest p(n) known (S(9781) = 3325581707333960528).
inline constexpr uint16_t S_UNKNOWN_4E18[] = {
    8423, 8429, 8537, 8627, 8669, 8713, 8741, 8753, 8783, 8803, 8807, 8837, 8839, 8849, 8861,
    8863, 8867, 8887, 8893, 8923, 8963, 8969, 8999, 9007, 9011, 9013, 9029, 9041, 9043, 9049,
    9059, 9067, 9103, 9109, 9127, 9133, 9137, 9151, 9157, 9173, 9181, 9187, 9199, 9209, 9221,
    9227, 9239, 9241, 9257, 9277, 9281, 9283, 9293, 9311, 9319, 9323, 9337, 9343, 9349, 9371,
    9377, 9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, 9461, 9463, 9467, 9473,
    9479, 9491, 9497, 9511, 9521, 9533, 9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623, 9631,
    9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733, 9739, 9743, 9749, 9767, 9769};
inline constexpr uint64_t RECORD_P = 9781;          // largest p(n) known below 4e18

/// p(n) by an ascending Miller-Rabin scan (independent of the GPU path):
/// the smallest odd prime p with n - p prime.
/// @pre    n even, n > 4 (cost: O(p(n)) Miller-Rabin tests).
/// @return p(n), or 0 if no odd prime p <= n / 2 works (a counterexample).
inline uint64_t pn_exact(uint64_t n)
{
    for (uint64_t p = 3; p <= n / 2; p += 2)
        if (is_prime_u64(p) && is_prime_u64(n - p)) return p;
    return 0;                                       // a Goldbach counterexample
}

/// Candidate lists of the statistics checker.
struct StatsPlan {
    std::vector<uint16_t> xall;   // per class r = 2 cr: useful candidates ascending, then futile
    std::vector<uint32_t> xtab;   // [0,15) offset, [15,30) K_r, [30,45) list length
    std::vector<uint16_t> pidx;   // per xall entry: index of its p in the odd-prime list
    std::vector<uint32_t> xj;     // check30_xs per xall entry: slot | d << 8
    std::vector<uint32_t> xch;    // check30_xs per class (stride 64): n | nd << 8 | NA << 16,
                                  // slot -> array nibbles, (jb | je << 16, pmax[8]) x n
    uint64_t vcls[4] = {};        // classes per variant (NA = 8, 6, 4, 3), nibble-packed
    uint32_t vn[4] = {};          // class count per variant
};

/// @param primes  odd candidate primes ascending (3 .. MAX_GB_PRIME).
/// @param K       exact phase-A prefix per class (1 .. STATS_KMAX).
/// @param p_lt    prefix candidates stay below this (the tail bound: tail
///                first occurrences are only tracked in check30_b / _xb).
/// @pre    primes = the odd primes 3 .. MAX_GB_PRIME ascending; 1 <= K <= STATS_KMAX.
/// @return per class r = 2 cr: xall = every candidate once, the useful ones
///         (r - p coprime to 30) ascending, then the futile ones ascending;
///         the first xtab[15 + cr] <= K entries are useful and < p_lt; the
///         chunk tables tile the prefix (host_tests checks all of it).
/// @throws std::runtime_error on an impossible class layout.
inline StatsPlan make_stats_plan(const std::vector<uint32_t>& primes, uint32_t K, uint32_t p_lt)
{
    if (K < 1 || K > STATS_KMAX) throw std::runtime_error("stats: K out of range");
    StatsPlan s;
    s.xtab.assign(45, 0);
    for (uint32_t cr = 0; cr < 15; ++cr) {
        const uint32_t r = 2 * cr;
        s.xtab[cr] = uint32_t(s.xall.size());
        std::vector<std::pair<uint16_t, uint16_t>> futile;
        uint32_t useful = 0;
        for (size_t i = 0; i < primes.size(); ++i) {
            const uint32_t p = primes[i];
            const bool u = std::gcd((r + 30 - p % 30) % 30, 30u) == 1;
            if (u) { s.xall.push_back(uint16_t(p)); s.pidx.push_back(uint16_t(i)); useful += p < p_lt; }
            else futile.push_back({uint16_t(p), uint16_t(i)});
        }
        for (auto& f : futile) { s.xall.push_back(f.first); s.pidx.push_back(f.second); }
        s.xtab[15 + cr] = std::min(K, useful);
        s.xtab[30 + cr] = uint32_t(s.xall.size()) - s.xtab[cr];
    }
    // check30_xs tables. Per class: the candidates p >= 7 of the prefix live
    // in NA "main" arrays (residue r - p coprime to 30), mapped to slots
    // 0..NA-1; p = 3 / 5 (first, when useful) are direct. Chunks: every
    // slot's candidates in a chunk span < 32 window bits (p range <= 930).
    s.xj.assign(s.xall.size(), 0);
    s.xch.assign(15 * 64, 0);
    static const uint32_t units[8] = {1, 7, 11, 13, 17, 19, 23, 29};
    for (uint32_t cr = 0; cr < 15; ++cr) {
        const uint32_t r = 2 * cr, o = s.xtab[cr], kx = s.xtab[15 + cr];
        uint32_t* t = &s.xch[cr * 64];
        uint32_t slot_of[8], na = 0, amap = 0;
        for (uint32_t k = 0; k < 8; ++k) {
            slot_of[k] = 99;
            if (std::gcd((r + 30 - units[k]) % 30, 30u) == 1) {   // p = r - c_k (mod 30) usable
                slot_of[k] = na;
                amap |= k << (4 * na++);
            }
        }
        uint32_t nd = 0;
        while (nd < kx && s.xall[o + nd] <= 5) ++nd;
        uint32_t pmin[8] = {}, nch = 0, jb = nd;
        uint32_t* ch = nullptr;
        for (uint32_t j = nd; j <= kx; ++j) {
            uint32_t a = 0, p = 0;
            if (j < kx) {
                p = s.xall[o + j];
                uint32_t k = 8;
                for (uint32_t u = 0; u < 8; ++u) if (units[u] == (r + 30 - p % 30) % 30) k = u;
                if (k > 7 || slot_of[k] > 7) throw std::runtime_error("stats: prefix candidate outside the main arrays");
                a = slot_of[k];
            }
            if (j == kx || (ch && pmin[a] && p - pmin[a] > 30 * 31)) {   // close the chunk
                if (ch) ch[0] = jb | (j << 16);
                if (j == kx) break;
                ch = nullptr;
            }
            if (!ch) {
                if (2 + 9 * (nch + 1) > 64) throw std::runtime_error("stats: too many check30_xs chunks");
                ch = t + 2 + 9 * nch++;
                jb = j;
                std::fill(pmin, pmin + 8, 0u);
            }
            if (!pmin[a]) pmin[a] = p;
            ch[1 + a] = p;                               // largest p of the slot so far
        }
        // d = (pmax of the slot in its chunk - p) / 30
        for (uint32_t c = 0; c < nch; ++c) {
            const uint32_t* h = t + 2 + 9 * c;
            for (uint32_t j = h[0] & 0xFFFF; j < (h[0] >> 16); ++j) {
                const uint32_t p = s.xall[o + j];
                uint32_t k = 0;
                for (uint32_t u = 0; u < 8; ++u) if (units[u] == (r + 30 - p % 30) % 30) k = u;
                const uint32_t a = slot_of[k], d = (h[1 + a] - p) / 30;
                if (d > 31 || (h[1 + a] - p) % 30) throw std::runtime_error("stats: chunk offset");
                s.xj[o + j] = a | (d << 8);
            }
        }
        t[0] = nch | (nd << 8) | (na << 16);
        t[1] = amap;
        const uint32_t v = na == 8 ? 0 : na == 6 ? 1 : na == 4 ? 2 : na == 3 ? 3 : 99;
        if (v > 3) throw std::runtime_error("stats: unexpected array count");
        s.vcls[v] |= uint64_t(cr) << (4 * s.vn[v]++);
    }
    return s;
}

/// count30 kernel limits of one segment: class c counts its bits t in
/// [lim[2c], lim[2c+1]); twin pair j (16 + 5j): class k, partner class k2,
/// partner offset delta, bit range [lo, hi) of class k.
/// Counted: primes q in [seg_start - 3, seg_end - 3], twin pairs (q, q + 2)
/// with q in [seg_start - 5, seg_end - 5] (contiguous across segments).
/// @param Q  first q of each mod-30 class array (class_geometry Q[]).
/// @pre    the segment's class geometry (K = 8); seg_start >= 6; the class
///         bit indices fit u32 (segment-span guard).
/// @return lim as above; consecutive segments' ranges tile the q axis with
///         no overlap or gap, so summed counts are exact for the run.
/// @throws std::runtime_error if a twin partner class is not at offset 0 / 1.
inline std::vector<uint32_t> count30_limits(uint64_t seg_start, uint64_t seg_end,
                                            const uint64_t Q[8])
{
    std::vector<uint32_t> lim(32, 0);
    const auto range = [&](uint32_t c, uint64_t lo, uint64_t hi, uint32_t& tl, uint32_t& th) {
        // t with lo <= Q[c] + 30 t <= hi
        tl = lo <= Q[c] ? 0u : uint32_t((lo - Q[c] + 29) / 30);
        th = hi < Q[c] ? 0u : uint32_t((hi - Q[c]) / 30 + 1);
        if (th < tl) th = tl;
    };
    for (uint32_t c = 0; c < 8; ++c)
        range(c, seg_start - 3, seg_end - 3, lim[2 * c], lim[2 * c + 1]);
    // residues 1 7 11 13 17 19 23 29: twins 11/13, 17/19, 29/31
    const uint32_t pk[3][2] = {{2, 3}, {4, 5}, {7, 0}};
    for (uint32_t j = 0; j < 3; ++j) {
        const uint32_t k = pk[j][0], k2 = pk[j][1];
        const int64_t d = int64_t(Q[k]) + 2 - int64_t(Q[k2]);
        if (d != 0 && d != 30) throw std::runtime_error("count30: twin class offset");
        lim[16 + 5 * j] = k;
        lim[17 + 5 * j] = k2;
        lim[18 + 5 * j] = uint32_t(d / 30);
        range(k, seg_start - 5, seg_end - 5, lim[19 + 5 * j], lim[20 + 5 * j]);
    }
    return lim;
}

/// Largest prime < x (x > 3; Miller-Rabin descending scan).
inline uint64_t prev_prime(uint64_t x)
{
    uint64_t q = (x - 1) | 1;
    if (q >= x) q -= 2;
    while (!is_prime_u64(q)) q -= 2;
    return q;
}
using u128 = unsigned __int128;
inline std::string u128_str(u128 v)
{
    std::string s;
    do { s.insert(s.begin(), char('0' + int(v % 10))); v /= 10; } while (v);
    return s;
}
inline u128 u128_parse(const char* s)
{
    u128 v = 0;
    while (*s >= '0' && *s <= '9') v = v * 10 + u128(*s++ - '0');
    return v;
}

/// Prime gap statistics of a run or campaign. A gap g = q - prev(q) belongs to
/// the run whose prime range [start + 1, limit - 1] holds its upper prime q
/// (as the prime counts). Exact, every segment: count (= the primes), D1 =
/// sum g, every gap >= GAPS30_TAIL with count and first lower prime (count30's
/// bit-parallel tail test), the largest gap (exact when >= GAPS30_TAIL or
/// every segment was walked). Walked segments (the sampled ones; every one
/// with GB_GAPS=exact): count30's full walk, the histogram N(g) by g / 2 and
/// the sums of g^k (k = 1..4) of those gaps.
struct GapStats {
    uint64_t n = 0;                                            // exact
    u128 d1 = 0;
    std::map<uint32_t, std::pair<uint64_t, uint64_t>> tail;   // g -> (count, first lower prime)
    uint64_t gmax = 0, gmax_lo = 0;                           // largest gap seen, its first lower prime
    bool all_walked = true;                                    // gmax is exact below the tail too
    uint64_t wn = 0;                                           // walked
    u128 wd[4] = {};
    std::vector<uint64_t> samp = std::vector<uint64_t>(GAPS30_HBINS, 0);

    /// A gap of the walked segments (sign -1: remove one counted before; a
    /// boundary fix of a gap below GAPS30_TAIL).
    void add_walk(uint64_t g, int sign = 1)
    {
        const u128 g1 = g, g2 = g1 * g1, m[4] = {g1, g2, g2 * g1, g2 * g2};
        if (sign > 0) {
            ++wn;
            for (int k = 0; k < 4; ++k) wd[k] += m[k];
            samp.at(g / 2) += 1;
        } else {
            if (g >= GAPS30_TAIL || !wn || !samp.at(g / 2)) throw std::runtime_error("gaps: bad walk removal");
            --wn;
            for (int k = 0; k < 4; ++k) wd[k] -= m[k];
            samp.at(g / 2) -= 1;
        }
    }
    /// A gap found by an exact path (tail list, segment / run boundary).
    void note_exact(uint64_t g, uint64_t lower)
    {
        if (g >= GAPS30_TAIL) note_tail(uint32_t(g), 1, lower);
        note_max(g, lower);
    }
    /// The largest gap; among equal ones the first (smallest lower prime).
    void note_max(uint64_t g, uint64_t lower)
    {
        if (g > gmax || (g == gmax && g && lower < gmax_lo)) { gmax = g; gmax_lo = lower; }
    }
    void note_tail(uint32_t g, uint64_t c, uint64_t lower)
    {
        auto& t = tail[g];
        if (!t.first || lower < t.second) t.second = lower;
        t.first += c;
    }
    bool gmax_exact() const { return gmax >= GAPS30_TAIL || all_walked; }
    void merge(const GapStats& o)
    {
        n += o.n;
        d1 += o.d1;
        for (auto& t : o.tail) note_tail(t.first, t.second.first, t.second.second);
        note_max(o.gmax, o.gmax_lo);
        all_walked = all_walked && o.all_walked;
        wn += o.wn;
        for (int k = 0; k < 4; ++k) wd[k] += o.wd[k];
        for (size_t i = 0; i < samp.size(); ++i) samp[i] += o.samp[i];
    }
    uint64_t sampled() const { return std::accumulate(samp.begin(), samp.end(), uint64_t(0)); }
    void clear() { *this = GapStats{}; }
};

/// Run accumulator + campaign state.
class GoldbachStats {
public:
    /// @param primes  odd candidate primes ascending (3 .. MAX_GB_PRIME).
    /// @param tail_t  every even with p(n) >= tail_t is recorded (plan bound).
    GoldbachStats(const std::vector<uint32_t>& primes, const StatsPlan& plan, uint32_t tail_t)
        : primes_(primes), plan_(plan), tail_t_(tail_t)
    {
        const size_t np = primes.size();
        pi_of_.assign(primes.back() / 2 + 1, 0xFFFF);
        for (size_t i = 0; i < np; ++i) pi_of_[primes[i] / 2] = uint16_t(i);
        samp_.assign(15 * np, 0);
        tail_.assign(15 * np, 0);
        cum_samp_.assign(15 * np, 0);
        cum_tail_.assign(15 * np, 0);
        first_.assign(np, 0);
        run_first_.assign(np, 0);
        want_p_.assign((primes.back() / 2 + 32) / 32, 0);
        const char* s = std::getenv("GB_STATS_STATE");
        state_path_ = (s && *s) ? s : "goldbach_stats.state";
        const char* l = std::getenv("GB_STATS_LOG");
        log_path_ = (l && *l) ? l : "goldbach_stats.log";
        const char* r = std::getenv("GB_RECORDS_FILE");
        rec_path_ = (r && *r) ? r : "goldbach_records.txt";
        const char* v = std::getenv("GB_STATS_LIVE");
        live_path_ = (v && *v) ? v : "goldbach_live.json";
        for (uint16_t p : S_UNKNOWN_4E18) unknown_.push_back(p);
        load_state();
        (void)load_gaps();                         // a damaged gap state fails before the run, not after
        for (size_t i = 0; i < np; ++i)
            if (primes[i] >= tail_t_ && !first_[i]) want_p_[primes[i] / 64] |= 1u << ((primes[i] / 2) & 31);
    }

    /// Bitset by p >> 1: tail bins without a campaign first occurrence yet.
    const std::vector<uint32_t>& want_p() const { return want_p_; }
    uint32_t tail_t() const { return tail_t_; }
    size_t pbins() const { return primes_.back() / 2 + 1; }

    /// Sampled segment: check30_xs* + check30_xb counts per xall entry.
    void add_full(const uint32_t* hist)
    {
        uint64_t n = 0;
        for (uint32_t cr = 0; cr < 15; ++cr) {
            const uint32_t o = plan_.xtab[cr], len = plan_.xtab[30 + cr];
            for (uint32_t j = 0; j < len; ++j) {
                const uint32_t c = hist[o + j];
                if (!c) continue;
                const uint32_t pi = plan_.pidx[o + j];
                samp_[cr * primes_.size() + pi] += c;     // kernels index by n's class: cr = (n mod 30) / 2
                if (primes_[pi] >= tail_t_) tail_[cr * primes_.size() + pi] += c;
                n += c;
            }
        }
        evens_sampled_ += n;
    }
    /// Normal segment: check30_b tail counts by p >> 1 (class-aggregated:
    /// stored under class 15 -> folded into the per-class arrays as class 0
    /// would be wrong, so they go to a separate aggregate).
    void add_tail(const uint32_t* hist_p)
    {
        for (size_t b = 0; b < pbins(); ++b)
            if (hist_p[b]) tail_agg_[b] += hist_p[b];
    }
    /// Either path: first-occurrence indices by p >> 1 (~0u = none).
    void add_first(uint64_t seg_start, const uint32_t* first_p)
    {
        for (size_t b = 0; b < pbins(); ++b)
            if (first_p[b] != ~0u && pi_of_[b] != 0xFFFF) note_first(pi_of_[b], seg_start + 2ull * first_p[b]);
    }
    /// count30 output: [0, 8) primes per class, [8, 11) twins per pair.
    void add_counts(const uint32_t* cnt)
    {
        for (int i = 0; i < 8; ++i) primes_res_[i] += cnt[i];
        for (int i = 0; i < 3; ++i) twins_[i] += cnt[8 + i];
    }

    /// Run start for the gaps: the prime before the counting range [start - 3, ...].
    void begin_gaps(uint64_t start) { gap_last_ = prev_prime(start - 3); gap_start_ = start; }
    /// One segment's gap outputs, segments in ascending order.
    /// @param q_first  the segment's position origin (positions are q - q_first).
    /// @param cnt      count30's output: [0, 8) primes per class, [11] / [12]
    ///                 the first / last counted prime's position (~0 / 0: none).
    /// @param gt       count30's gap tail: [0] entries, [1] error flags, then
    ///                 (upper position, g) pairs (gaps >= GAPS30_TAIL but the
    ///                 one into the segment's first prime, stitched here).
    /// @param wmax     count30 walked the segment: its largest gap as g << 32
    ///                 | ~(upper position) (0: no gap); else nullptr.
    /// @param hist     the walk's histogram by g / 2; bin 0 counts the primes
    ///                 without a counted predecessor (else nullptr).
    /// @throws std::runtime_error on a kernel error flag or a failed
    ///         cross-check: one prime without a predecessor, walk gaps + 1 =
    ///         primes, sum g = last - first, the histogram's bins >= GAPS30_TAIL
    ///         = the bit-parallel tail list, no gap in the overflow bin.
    void add_gap_segment(uint64_t q_first, uint64_t seg_evens, const uint32_t* cnt, const uint32_t* gt,
                         const uint64_t* wmax, const uint32_t* hist)
    {
        uint64_t primes = 0;
        for (int i = 0; i < 8; ++i) primes += cnt[i];
        const bool walked = hist != nullptr;
        gap_last_walked_ = walked;                 // limit - 1's gap belongs to the run's last segment
        GapStats& r = gaps_;
        r.all_walked = r.all_walked && walked;
        if (walked) walk_evens_ += seg_evens;
        if (gt[1]) throw std::runtime_error("count30: gap error flags " + std::to_string(gt[1]));
        if (gt[0] > GAPS30_TCAP) throw std::runtime_error("count30: gap tail list overflow");
        for (uint32_t i = 0; i < gt[0]; ++i) {
            const uint64_t up = q_first + gt[2 + 2 * i], g = gt[3 + 2 * i];
            if (up >= gap_start_ + 1) r.note_exact(g, up - g);
        }
        if (!primes) {
            if (cnt[11] != ~0u || (walked && (*wmax || std::any_of(hist, hist + GAPS30_HBINS, [](uint32_t c) { return c != 0; }))))
                throw std::runtime_error("gaps: a prime position in a segment without primes");
            return;
        }
        const uint64_t fq = q_first + cnt[11], lq = q_first + cnt[12];
        if (walked) {
            std::map<uint32_t, uint64_t> tl;       // the bit-parallel tail by g
            for (uint32_t i = 0; i < gt[0]; ++i) tl[gt[3 + 2 * i]] += 1;
            u128 hm[4] = {};
            uint64_t hs = 0, ht = 0;
            bool tail_ok = true;
            for (uint32_t i = 1; i + 1 < GAPS30_HBINS; ++i) {
                const u128 c = hist[i], g = 2 * i, g2 = g * g;
                hs += hist[i];
                hm[0] += c * g; hm[1] += c * g2; hm[2] += c * g2 * g; hm[3] += c * g2 * g2;
                if (2 * i >= GAPS30_TAIL) {
                    ht += hist[i];
                    const auto it = tl.find(2 * i);
                    tail_ok = tail_ok && hist[i] == (it == tl.end() ? 0 : it->second);
                }
            }
            const bool ok = hist[0] == 1 && hist[GAPS30_HBINS - 1] == 0 && hs + 1 == primes && hm[0] == lq - fq &&
                            tail_ok && ht == gt[0];
            if (!ok)
                throw std::runtime_error("gaps: cross-check failed (walk gaps " + std::to_string(hs) + ", primes " +
                                         std::to_string(primes) + ", sum " + std::to_string(uint64_t(hm[0])) + ", span " +
                                         std::to_string(lq - fq) + ", starts " + std::to_string(hist[0]) +
                                         ", overflow " + std::to_string(hist[GAPS30_HBINS - 1]) + ", tails " +
                                         std::to_string(ht) + "/" + std::to_string(gt[0]) + (tail_ok ? "" : " by g") + ")");
            r.wn += hs;
            for (int k = 0; k < 4; ++k) r.wd[k] += hm[k];
            for (uint32_t i = 1; i < GAPS30_HBINS; ++i) r.samp[i] += hist[i];
            if (hs) {
                const uint64_t g = *wmax >> 32, up = q_first + uint32_t(~uint32_t(*wmax));
                if (!g || g >= 2 * GAPS30_HBINS) throw std::runtime_error("gaps: walk maximum out of range");
                r.note_max(g, up - g);
            }
        }
        // the gap into the segment's first prime (not one of the run's while
        // its upper prime is below start + 1)
        if (fq >= gap_start_ + 1) {
            r.note_exact(fq - gap_last_, gap_last_);
            if (walked) r.add_walk(fq - gap_last_);
        }
        gap_last_ = lq;
    }

    /// CPU-resolved even (leftover fallback); p may exceed MAX_GB_PRIME.
    /// @param sampled  the even's segment is a sampled one.
    void add_cpu(uint64_t n, uint64_t p, bool sampled)
    {
        const uint32_t cr = uint32_t(n % 30) / 2;
        if (p <= primes_.back() && pi_of_[p / 2] != 0xFFFF) {
            const uint32_t pi = pi_of_[p / 2];
            if (sampled) { samp_[cr * primes_.size() + pi] += 1; ++evens_sampled_; }
            if (p >= tail_t_) {
                if (sampled) tail_[cr * primes_.size() + pi] += 1;
                else tail_agg_[p / 2] += 1;
                note_first(pi, n);
            }
        } else {
            beyond_.push_back({n, p});
            if (sampled) ++evens_sampled_;
            record(n, p, p > RECORD_P ? "RECORD" : "BEYOND", p > RECORD_P ? "NEW RECORD: p(n) > 9781, the largest p(n) known below 4e18"
                                      : "p(n) beyond the candidate list");
        }
    }

    /// Exact boundary corrections and the run's even count (once, at the end).
    void finish_run(uint64_t start, uint64_t limit, uint64_t evens)
    {
        start_ = start; limit_ = limit; evens_ = evens;
        // primes counted over [start - 3, limit - 3]; wanted [start + 1, limit - 1]
        for (uint64_t q : {start - 3, start - 1}) if (is_prime_u64(q)) primes_res_[res_idx(q)] -= 1;
        if (is_prime_u64(limit - 1)) primes_res_[res_idx(limit - 1)] += 1;
        // twins counted for q in [start - 5, limit - 5]; wanted q in [start + 1, limit - 3]
        for (uint64_t q : {start - 5, start - 3, start - 1})
            if (is_prime_u64(q) && is_prime_u64(q + 2)) twins_[twin_idx(q)] -= 1;
        if (is_prime_u64(limit - 3) && is_prime_u64(limit - 1)) twins_[twin_idx(limit - 3)] += 1;
        if (gap_start_ == start) {
            gaps_.n = run_primes();                // one gap per prime of [start + 1, limit - 1]
            if (limit >= start + 2) {
                // the walk's gap (start - 3, start - 1) has its upper prime below
                // start + 1 (segment 0 is always walked); limit - 1 is prime and
                // past the counting range: its gap from the last counted prime
                if (is_prime_u64(start - 3) && is_prime_u64(start - 1)) gaps_.add_walk(2, -1);
                if (is_prime_u64(limit - 1)) {
                    gaps_.note_exact(limit - 1 - gap_last_, gap_last_);
                    if (gap_last_walked_) gaps_.add_walk(limit - 1 - gap_last_);
                }
                // D1: the last prime of the range minus the first one's predecessor
                uint64_t f = (start + 1) | 1;
                while (f <= limit - 1 && !is_prime_u64(f)) f += 2;
                if (f <= limit - 1) gaps_.d1 = prev_prime(limit) - prev_prime(f);
            }
        }
    }
    const GapStats& run_gaps() const { return gaps_; }

    uint64_t run_primes() const { return std::accumulate(primes_res_, primes_res_ + 8, uint64_t(0)); }
    uint64_t run_twins() const { return twins_[0] + twins_[1] + twins_[2]; }
    uint64_t evens_sampled() const { return evens_sampled_; }
    /// Tail evens of the run (p(n) >= tail_t, all segments).
    uint64_t run_tail() const
    {
        uint64_t t = beyond_.size();
        for (uint64_t v : tail_) t += v;
        for (uint64_t v : tail_agg_) t += v;
        return t;
    }
    /// Largest p(n) of the run if it is >= tail_t (0 otherwise).
    uint64_t run_pmax() const
    {
        uint64_t m = 0;
        for (auto& b : beyond_) m = std::max(m, b.second);
        if (m) return m;
        for (size_t b = tail_agg_.size(); b-- > 0;)
            if (tail_agg_[b]) { m = 2 * b + 1; break; }
        for (size_t i = primes_.size(); i-- > 0 && primes_[i] > m;)
            for (uint32_t cr = 0; cr < 15; ++cr)
                if (tail_[cr * primes_.size() + i]) return primes_[i];
        return m;
    }

    /// Merges this run into the campaign state and appends its log line.
    /// @pre   the run PASSED; every segment's statistics were added.
    /// @param sampled_total  evens of the sampled segments (cross-check).
    /// @post  either [start, limit] overlaps a range the state file covers NOW
    ///        (nothing written: a re-run of covered evens is never counted
    ///        twice), or the state file holds the old totals plus this run's,
    ///        durably (fsync) and atomically (rename), and the log has the
    ///        run's line. The file is re-read under an exclusive lock, so
    ///        concurrent runs sharing a state file cannot lose updates.
    /// @throws std::runtime_error when the sampled-evens cross-check fails,
    ///         the state file is unreadable / incomplete, or it cannot be
    ///         written (the state is then unchanged).
    void commit(const char* version, uint64_t sampled_total)
    {
        if (evens_sampled_ != sampled_total)
            throw std::runtime_error("stats: " + std::to_string(evens_sampled_) +
                                     " sampled evens recorded, " + std::to_string(sampled_total) +
                                     " in the sampled segments (statistics not written)");
        const StateLock lock(state_path_ + ".lock");
        reset_campaign();
        load_state();                                  // the state as it is now
        for (auto& c : covered_)
            if (start_ <= c.second && c.first <= limit_) {
                std::fprintf(stderr, "stats: [%llu, %llu] overlaps the covered range [%llu, %llu] "
                             "of %s -- not recorded again\n",
                             (unsigned long long)start_, (unsigned long long)limit_,
                             (unsigned long long)c.first, (unsigned long long)c.second,
                             state_path_.c_str());
                // a gap backfill: the p(n) statistics of the range are in, its
                // gaps may not be (the gap state keeps its own ranges). The row also
                // carries the run's sampled p(n) histogram (psev evens, psamp counts by
                // prime index, classes summed): the state holds it only cumulatively,
                // so these rows give the p(n) history (live paper Figure 4) of a range
                // the backfill re-runs.
                if (gap_start_ == start_) {
                    const std::string row = "GBGAPS v1 version=" + std::string(version) + " start=" +
                                            std::to_string(start_) + " limit=" + std::to_string(limit_) +
                                            " evens=" + std::to_string(evens_) + gap_fields() +
                                            " psev=" + std::to_string(evens_sampled_) +
                                            " psamp=" + dense_list(by_prime(samp_, nullptr)) + "\n";
                    if (commit_gaps())
                        append_log(row);
                    else if (gaps_cover(start_, limit_) && !log_has_gaps_in(start_, limit_)) {
                        // merged, but its row never reached the log (a crash before
                        // the append hit the disk): the row depends on the run only
                        std::fprintf(stderr, "stats: %s has no gap row for [%llu, %llu] -- written now\n",
                                     log_path_.c_str(), (unsigned long long)start_, (unsigned long long)limit_);
                        append_log(row);
                    }
                }
                return;
            }
        const std::string line = log_line(version);    // before first_ changes (new_first)
        for (size_t i = 0; i < samp_.size(); ++i) { cum_samp_[i] += samp_[i]; cum_tail_[i] += tail_[i]; }
        for (size_t b = 0; b < tail_agg_.size(); ++b) cum_tail_agg_[b] += tail_agg_[b];
        for (size_t i = 0; i < primes_.size(); ++i)
            if (run_first_[i] && (!first_[i] || run_first_[i] < first_[i])) first_[i] = run_first_[i];
        for (int i = 0; i < 8; ++i) tot_primes_[i] += primes_res_[i];
        for (int i = 0; i < 3; ++i) tot_twins_[i] += twins_[i];
        tot_evens_ += evens_;
        tot_sampled_ += evens_sampled_;
        if (gap_start_ == start_) commit_gaps();
        for (auto& b : beyond_) tot_beyond_.push_back(b);
        covered_.push_back({start_, limit_});
        merge_covered();
        save_state();
        append_log(line);
    }

    const std::string& log_path() const { return log_path_; }

    /// Live snapshot for tools/goldbach_live.py, at most every 2 s (atomic
    /// rewrite, ~1 KB): run progress, largest p(n) so far, tail counts,
    /// running prime / twin counts, the run's record events.
    void write_live(uint64_t start, uint64_t limit, uint64_t evens_done, uint64_t evens_total,
                    double elapsed, bool force = false)
    {
        const std::time_t now = std::time(nullptr);
        if (!force && now - live_t_ < 2) return;
        live_t_ = now;
        uint64_t pm = 0, pmn = 0, ge[3] = {};
        for (size_t i = 0; i < primes_.size(); ++i)
            if (run_first_[i] && primes_[i] > pm) { pm = primes_[i]; pmn = run_first_[i]; }
        for (auto& b : beyond_) if (b.second > pm) { pm = b.second; pmn = b.first; }
        const uint64_t lims[3] = {8000, 9000, RECORD_P + 1};
        for (size_t b = 0; b < tail_agg_.size(); ++b)
            for (int k = 0; k < 3; ++k) if (2 * b + 1 >= lims[k]) ge[k] += tail_agg_[b];
        for (uint32_t cr = 0; cr < 15; ++cr)
            for (size_t i = 0; i < primes_.size(); ++i)
                for (int k = 0; k < 3; ++k)
                    if (primes_[i] >= lims[k]) ge[k] += tail_[cr * primes_.size() + i];
        ge[2] += beyond_.size();
        std::string o;
        char b[512];
        std::snprintf(b, sizeof b,
            "{\"time\": %lld, \"start\": %llu, \"limit\": %llu, \"evens_done\": %llu, "
            "\"evens_total\": %llu, \"elapsed_s\": %.1f, \"tail_t\": %u, \"tail_evens\": %llu, "
            "\"pmax\": %llu, \"pmax_n\": %llu, \"ge8000\": %llu, \"ge9000\": %llu, "
            "\"gt9781\": %llu, \"primes\": %llu, \"twins\": %llu, \"events\": [",
            (long long)now, (unsigned long long)start, (unsigned long long)limit,
            (unsigned long long)evens_done, (unsigned long long)evens_total, elapsed, tail_t_,
            (unsigned long long)run_tail(), (unsigned long long)pm, (unsigned long long)pmn,
            (unsigned long long)ge[0], (unsigned long long)ge[1], (unsigned long long)ge[2],
            (unsigned long long)run_primes(), (unsigned long long)run_twins());
        o += b;
        for (size_t i = 0; i < events_.size(); ++i) {
            std::snprintf(b, sizeof b, "%s[%llu, \"%llu\"]", i ? ", " : "",
                          (unsigned long long)events_[i].second, (unsigned long long)events_[i].first);
            o += b;
        }
        o += "]}\n";
        const std::string tmp = live_path_ + ".tmp";
        if (std::FILE* f = std::fopen(tmp.c_str(), "wb")) {
            const bool ok = std::fwrite(o.data(), 1, o.size(), f) == o.size();
            if (std::fclose(f) == 0 && ok) std::rename(tmp.c_str(), live_path_.c_str());
        }
    }

private:
    void note_first(uint32_t pi, uint64_t n)
    {
        const bool fresh = !run_first_[pi];
        if (fresh || n < run_first_[pi]) run_first_[pi] = n;
        const uint32_t p = primes_[pi];
        if (fresh && !first_[pi] && std::binary_search(unknown_.begin(), unknown_.end(), uint16_t(p)))
            record(n, p, "SP", "S(p) candidate: first even in this campaign with p(n) = p; "
                         "S(p) was unknown (> 4e18) -- this IS S(p) if the campaign "
                         "covers [4e18, n] without a gap");
    }
    /// Re-proves p(n) on the CPU, then appends the event to the records file
    /// (immediately: no waiting for the slice to finish) and to stderr, and
    /// runs the notification hook in the background: GB_RECORD_HOOK if set
    /// (empty: none), else scripts/notify_record.sh when present, called as
    /// `hook KIND P N Q CHECK` (KIND = RECORD / SP / BEYOND; digits and fixed
    /// words only, so the shell line needs no quoting).
    void record(uint64_t n, uint64_t p, const char* kind, const char* what)
    {
        const uint64_t ex = pn_exact(n);
        char line[512];
        const std::time_t t = std::time(nullptr);
        char ts[32];
        std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
        std::snprintf(line, sizeof line, "%s p(n)=%llu n=%llu q=%llu cpu_check=%s  %s\n", ts,
                      (unsigned long long)p, (unsigned long long)n, (unsigned long long)(n - p),
                      ex == p ? "OK" : "MISMATCH", what);
        std::fprintf(stderr, "\n*** %s", line);
        if (std::FILE* f = std::fopen(rec_path_.c_str(), "ab")) { std::fputs(line, f); std::fclose(f); }
        events_.push_back({n, p});
        const char* hk = std::getenv("GB_RECORD_HOOK");
        std::string hook = hk ? hk : "";
        if (!hk) {
            if (std::FILE* sf = std::fopen("scripts/notify_record.sh", "rb")) {
                std::fclose(sf);
                hook = "bash scripts/notify_record.sh";
            }
        }
        if (!hook.empty()) {
            char cmd[1024];
            std::snprintf(cmd, sizeof cmd, "%s %s %llu %llu %llu %s >/dev/null 2>&1 &", hook.c_str(), kind,
                          (unsigned long long)p, (unsigned long long)n, (unsigned long long)(n - p),
                          ex == p ? "OK" : "MISMATCH");
            if (std::system(cmd) != 0) std::fprintf(stderr, "record hook failed to start: %s\n", hook.c_str());
        }
    }
    static int res_idx(uint64_t q)
    {
        static const int idx[30] = {-1, 0, -1, -1, -1, -1, -1, 1, -1, -1, -1, 2, -1, 3, -1,
                                    -1, -1, 4, -1, 5, -1, -1, -1, 6, -1, -1, -1, -1, -1, 7};
        return idx[q % 30];
    }
    static int twin_idx(uint64_t q) { const uint32_t m = uint32_t(q % 30); return m == 11 ? 0 : m == 17 ? 1 : 2; }
    void merge_covered() { merge_ranges(covered_); }
    static void merge_ranges(std::vector<std::pair<uint64_t, uint64_t>>& r)
    {
        std::sort(r.begin(), r.end());
        std::vector<std::pair<uint64_t, uint64_t>> m;
        for (auto& c : r) {
            if (!m.empty() && c.first <= m.back().second + 2) m.back().second = std::max(m.back().second, c.second);
            else m.push_back(c);
        }
        r = m;
    }
    /// Per prime index, summed over the classes (+ the class-less tail).
    std::vector<uint64_t> by_prime(const std::vector<uint64_t>& cls, const std::vector<uint64_t>* agg) const
    {
        std::vector<uint64_t> v(primes_.size(), 0);
        for (uint32_t cr = 0; cr < 15; ++cr)
            for (size_t i = 0; i < primes_.size(); ++i) v[i] += cls[cr * primes_.size() + i];
        if (agg)
            for (size_t i = 0; i < primes_.size(); ++i) v[i] += (*agg)[primes_[i] / 2];
        return v;
    }
    // Dense list: counts by odd-prime index (3, 5, 7, ...), zeros empty, trailing zeros cut.
    static std::string dense_list(const std::vector<uint64_t>& v, size_t from = 0)
    {
        size_t last = 0;
        for (size_t i = from; i < v.size(); ++i) if (v[i]) last = i + 1;
        std::string s;
        char b[32];
        for (size_t i = from; i < last; ++i) {
            if (i > from) s += ',';
            if (v[i]) { std::snprintf(b, sizeof b, "%llu", (unsigned long long)v[i]); s += b; }
        }
        return s;
    }

    /// @return the run's GBSTATS log line (call before the campaign
    ///         first_ is updated: new_first compares against it).
    std::string log_line(const char* version) const
    {
        std::string line = "GBSTATS v2 version=" + std::string(version);
        char b[320];
        std::snprintf(b, sizeof b, " start=%llu limit=%llu evens=%llu evens_sampled=%llu tail_t=%u "
                      "tail_evens=%llu pmax=%llu primes=%llu twins=%llu",
                      (unsigned long long)start_, (unsigned long long)limit_,
                      (unsigned long long)evens_, (unsigned long long)evens_sampled_, tail_t_,
                      (unsigned long long)run_tail(), (unsigned long long)run_pmax(),
                      (unsigned long long)run_primes(), (unsigned long long)run_twins());
        line += b;
        line += " primes30=";
        for (int i = 0; i < 8; ++i) { std::snprintf(b, sizeof b, "%s%llu", i ? "," : "", (unsigned long long)primes_res_[i]); line += b; }
        line += " twins30=";
        for (int i = 0; i < 3; ++i) { std::snprintf(b, sizeof b, "%s%llu", i ? "," : "", (unsigned long long)twins_[i]); line += b; }
        std::string nf;
        for (size_t i = 0; i < primes_.size(); ++i)
            if (run_first_[i] && (!first_[i] || run_first_[i] < first_[i])) {
                std::snprintf(b, sizeof b, "%s%u:%llu", nf.empty() ? "" : ";", primes_[i],
                              (unsigned long long)run_first_[i]);
                nf += b;
            }
        line += " new_first=" + (nf.empty() ? std::string("-") : nf);
        std::string bey;
        for (auto& e : beyond_) {
            std::snprintf(b, sizeof b, "%s%llu:%llu", bey.empty() ? "" : ";",
                          (unsigned long long)e.second, (unsigned long long)e.first);
            bey += b;
        }
        line += " beyond=" + (bey.empty() ? std::string("-") : bey);
        if (gap_start_ == start_) line += gap_fields();
        // tail: counts by consecutive odd primes from the first p >= tail_t
        // ("3001:c,c,,c"); the sampled histogram is kept in the state only
        size_t t0 = 0;
        while (t0 < primes_.size() && primes_[t0] < tail_t_) ++t0;
        const std::string tl = t0 < primes_.size() ? dense_list(by_prime(tail_, &tail_agg_), t0) : "";
        line += " tail=" + (tl.empty() ? std::string("-") : std::to_string(primes_[t0]) + ":" + tl) + "\n";
        return line;
    }
    /// Log fields of a run's gaps: gaps=count gd1=sum g gmax=g:lower (- when
    /// not exact) gtail=g:count:lower;... gwalk=walked evens:walked gaps
    /// gsmom=walked sums of g^1..g^4 gsamp=<walked histogram by g / 2, dense>.
    std::string gap_fields() const
    {
        const GapStats& g = gaps_;
        std::string o = " gaps=" + std::to_string(g.n) + " gd1=" + u128_str(g.d1) + " gmax=" +
                        (g.gmax_exact() && g.gmax ? std::to_string(g.gmax) + ":" + std::to_string(g.gmax_lo) : "-") + " gtail=";
        std::string t;
        for (auto& e : g.tail)
            t += (t.empty() ? "" : ";") + std::to_string(e.first) + ":" + std::to_string(e.second.first) + ":" +
                 std::to_string(e.second.second);
        o += t.empty() ? "-" : t;
        o += " gwalk=" + std::to_string(walk_evens_) + ":" + std::to_string(g.wn) + " gsmom=";
        for (int k = 0; k < 4; ++k) o += (k ? "," : "") + u128_str(g.wd[k]);
        const std::string h = dense_list(g.samp);
        return o + " gsamp=" + (h.empty() ? std::string("-") : h);
    }
    /// Appends @p line to the run log. The log is a human-readable history;
    /// the state file is authoritative, so a failed append only warns. The
    /// append is durable (fsync): the state is already on disk, and a line
    /// left in the page cache is lost by a crash (the gap history is rebuilt
    /// from these lines).
    void append_log(const std::string& line) const
    {
        std::FILE* f = std::fopen(log_path_.c_str(), "ab");
        if (!f) { std::fprintf(stderr, "stats: cannot append to %s\n", log_path_.c_str()); return; }
        bool ok = std::fwrite(line.data(), 1, line.size(), f) == line.size();
        ok = (std::fflush(f) == 0) && ok;
        ok = (::fsync(::fileno(f)) == 0) && ok;
        if (std::fclose(f) != 0 || !ok)
            std::fprintf(stderr, "stats: incomplete append to %s\n", log_path_.c_str());
    }

    /// Whether the log has a line with gap fields (a GBGAPS row, or the
    /// GBSTATS line of a run that recorded its gaps) for evens in [a, b].
    bool log_has_gaps_in(uint64_t a, uint64_t b) const
    {
        std::FILE* f = std::fopen(log_path_.c_str(), "rb");
        if (!f) return false;
        char* ln = nullptr;
        size_t cap = 0;
        bool hit = false;
        while (!hit && ::getline(&ln, &cap, f) > 0) {
            if (std::strncmp(ln, "GB", 2) != 0 || !std::strstr(ln, " gd1=")) continue;
            const char* s = std::strstr(ln, " start=");
            const char* l = std::strstr(ln, " limit=");
            if (s && l)
                hit = std::strtoull(s + 7, nullptr, 10) <= b && a <= std::strtoull(l + 7, nullptr, 10);
        }
        std::free(ln);
        std::fclose(f);
        return hit;
    }

    static std::vector<uint64_t> parse_dense(const char* s, size_t n)
    {
        std::vector<uint64_t> v(n, 0);
        size_t i = 0;
        while (s && *s && *s != '\n' && i < n) {
            v[i++] = std::strtoull(s, nullptr, 10);
            s = std::strchr(s, ',');
            if (s) ++s;
        }
        return v;
    }

    /// Merges this run's gaps into the gap state (GAPS file: the state path +
    /// ".gaps", so isolating GB_STATS_STATE isolates it too), written BEFORE
    /// the main state: a crash in between leaves the run in the gap state
    /// only, and its re-run is then not merged into the gaps twice. A binary
    /// without gap statistics never reads or rewrites this file.
    /// @pre   under the state lock; the run passed the main overlap check.
    /// @return the run was merged (false: its range is already in the gap state).
    bool commit_gaps()
    {
        GapCampaign gc = load_gaps();
        for (auto& c : gc.covered)
            if (start_ <= c.second && c.first <= limit_) {
                std::fprintf(stderr, "stats: gaps of [%llu, %llu] already in %s -- not recorded again\n",
                             (unsigned long long)start_, (unsigned long long)limit_, gap_path().c_str());
                return false;
            }
        gc.g.merge(gaps_);
        gc.evens += evens_;
        gc.walked += walk_evens_;
        gc.covered.push_back({start_, limit_});
        merge_ranges(gc.covered);
        std::string out =
            "# goldbach-l0 prime gap state v1 (exact; include/gstats.hpp GapStats)\n"
            "# gaps_range: lo hi with prime gaps recorded (upper prime in [lo + 1, hi - 1]);\n"
            "# gaps_evens <evens> <walked evens>; gaps <count> <sum g> (exact);\n"
            "# gaps_walk <count> <sum g> <sum g^2> <sum g^3> <sum g^4> and gaps_samp (histogram by g / 2): walked segments;\n"
            "# gaps_max <g> <lower> <1: exact, 0: largest seen>; gaps_tail <g> <count> <first lower> (g >= 700, exact)\n";
        for (auto& c : gc.covered)
            out += "gaps_range " + std::to_string(c.first) + " " + std::to_string(c.second) + "\n";
        out += "gaps_evens " + std::to_string(gc.evens) + " " + std::to_string(gc.walked) + "\n";
        out += "gaps " + std::to_string(gc.g.n) + " " + u128_str(gc.g.d1) + "\ngaps_walk " + std::to_string(gc.g.wn);
        for (int k = 0; k < 4; ++k) out += " " + u128_str(gc.g.wd[k]);
        out += "\ngaps_samp " + dense_list(gc.g.samp) + "\n";
        out += "gaps_max " + std::to_string(gc.g.gmax) + " " + std::to_string(gc.g.gmax_lo) + " " +
               (gc.g.gmax_exact() ? "1" : "0") + "\n";
        for (auto& t : gc.g.tail)
            out += "gaps_tail " + std::to_string(t.first) + " " + std::to_string(t.second.first) + " " +
                   std::to_string(t.second.second) + "\n";
        out += "end\n";
        write_durably(gap_path(), out);
        return true;
    }
    struct GapCampaign {
        GapStats g;
        uint64_t evens = 0, walked = 0;
        std::vector<std::pair<uint64_t, uint64_t>> covered;
    };
    std::string gap_path() const { return state_path_ + ".gaps"; }
    /// Whether the gap state holds all of [a, b].
    bool gaps_cover(uint64_t a, uint64_t b) const
    {
        for (auto& c : load_gaps().covered)
            if (c.first <= a && b <= c.second) return true;
        return false;
    }
    /// The gap state (absent = empty).
    /// @throws std::runtime_error if it exists but is cut off or misses a
    ///         record (save writes every one, "end" last).
    GapCampaign load_gaps() const
    {
        GapCampaign gc;
        std::FILE* f = std::fopen(gap_path().c_str(), "rb");
        if (!f) return gc;
        std::string all;
        char chunk[1 << 14];
        size_t r;
        while ((r = std::fread(chunk, 1, sizeof chunk, f)) > 0) all.append(chunk, r);
        std::fclose(f);
        unsigned seen = 0;
        size_t pos = 0;
        while (pos < all.size()) {
            size_t e = all.find('\n', pos);
            if (e == std::string::npos) break;
            const std::string buf = all.substr(pos, e - pos);
            pos = e + 1;
            unsigned long long a = 0, c = 0;
            unsigned p = 0;
            if (buf.empty() || buf[0] == '#') continue;
            unsigned ex = 0;
            if (std::sscanf(buf.c_str(), "gaps_range %llu %llu", &a, &c) == 2) { gc.covered.push_back({a, c}); seen |= 1; }
            else if (std::sscanf(buf.c_str(), "gaps_evens %llu %llu", &a, &c) == 2) { gc.evens = a; gc.walked = c; seen |= 2; }
            else if (buf.compare(0, 10, "gaps_walk ") == 0) {
                const char* g = buf.c_str() + 10;
                gc.g.wn = std::strtoull(g, nullptr, 10);
                bool ok = true;
                for (int k = 0; k < 4 && ok; ++k) {
                    g = std::strchr(g, ' ');
                    ok = g != nullptr;
                    if (ok) gc.g.wd[k] = u128_parse(++g);
                }
                if (ok) seen |= 64;
            } else if (buf.compare(0, 5, "gaps ") == 0) {
                const char* g = buf.c_str() + 5;
                gc.g.n = std::strtoull(g, nullptr, 10);
                g = std::strchr(g, ' ');
                if (g) { gc.g.d1 = u128_parse(g + 1); seen |= 4; }
            } else if (buf.compare(0, 10, "gaps_samp ") == 0) {
                const auto v = parse_dense(buf.c_str() + 10, GAPS30_HBINS);
                std::copy(v.begin(), v.end(), gc.g.samp.begin());
                seen |= 8;
            } else if (buf == "gaps_samp") seen |= 8;
            else if (std::sscanf(buf.c_str(), "gaps_max %llu %llu %u", &a, &c, &ex) == 3) {
                gc.g.gmax = a; gc.g.gmax_lo = c; gc.g.all_walked = ex && a < GAPS30_TAIL; seen |= 16;
            } else if (std::sscanf(buf.c_str(), "gaps_tail %u %llu %llu", &p, &a, &c) == 3) gc.g.note_tail(p, a, c);
            else if (buf == "end") seen |= 32;
        }
        if (seen != 127)
            throw std::runtime_error("stats: gap state " + gap_path() + " is cut off or missing records; "
                                     "refusing to merge into it");
        merge_ranges(gc.covered);
        return gc;
    }

    /// Exclusive flock on a lock file next to the state (released on scope exit).
    struct StateLock {
        int fd = -1;
        explicit StateLock(const std::string& path)
        {
            fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
            if (fd < 0 || ::flock(fd, LOCK_EX) != 0)
                throw std::runtime_error("stats: cannot lock " + path);
        }
        ~StateLock() { if (fd >= 0) ::close(fd); }
        StateLock(const StateLock&) = delete;
        StateLock& operator=(const StateLock&) = delete;
    };

    /// Clears the campaign totals (before re-reading the state file).
    void reset_campaign()
    {
        std::fill(cum_samp_.begin(), cum_samp_.end(), 0);
        std::fill(cum_tail_.begin(), cum_tail_.end(), 0);
        std::fill(cum_tail_agg_.begin(), cum_tail_agg_.end(), 0);
        std::fill(first_.begin(), first_.end(), 0);
        covered_.clear();
        tot_beyond_.clear();
        tot_evens_ = tot_sampled_ = 0;
        for (auto& v : tot_primes_) v = 0;
        for (auto& v : tot_twins_) v = 0;
    }

    /// Loads the campaign totals from the state file (absent = empty campaign).
    /// @pre   campaign totals cleared (constructor or reset_campaign).
    /// @post  the totals equal the file's.
    /// @throws std::runtime_error if the file exists but is not a complete
    ///         state written by save_state (empty, cut-off last line, or a
    ///         record missing: save_state writes every record, tailagg last),
    ///         or was recorded with another tail threshold — merging into a
    ///         damaged or incompatible state would silently corrupt it.
    void load_state()
    {
        std::FILE* f = std::fopen(state_path_.c_str(), "rb");
        if (!f) return;
        std::string buf;
        char chunk[1 << 14];
        std::string all;
        size_t r;
        while ((r = std::fread(chunk, 1, sizeof chunk, f)) > 0) all.append(chunk, r);
        std::fclose(f);
        if (all.empty() || all.back() != '\n')
            throw std::runtime_error("stats: state " + state_path_ + " is empty or cut off "
                                     "(truncated write?); refusing to merge into it");
        uint32_t seen_samp = 0, seen_tail = 0;
        bool seen_t = false, seen_ev = false, seen_p = false, seen_tw = false, seen_agg = false;
        size_t pos = 0;
        const size_t np = primes_.size();
        while (pos < all.size()) {
            size_t e = all.find('\n', pos);
            if (e == std::string::npos) e = all.size();
            buf = all.substr(pos, e - pos);
            pos = e + 1;
            if (buf.empty() || buf[0] == '#') continue;
            unsigned long long a = 0, c = 0;
            unsigned p = 0, cr = 0;
            if (std::sscanf(buf.c_str(), "range %llu %llu", &a, &c) == 2) covered_.push_back({a, c});
            else if (std::sscanf(buf.c_str(), "tail_t %u", &p) == 1) {
                if (p != tail_t_)
                    throw std::runtime_error("stats: state " + state_path_ + " was recorded with tail_t " +
                                             std::to_string(p) + ", this run uses " + std::to_string(tail_t_));
                seen_t = true;
            }
            else if (std::sscanf(buf.c_str(), "evens %llu %llu", &a, &c) == 2) { tot_evens_ = a; tot_sampled_ = c; seen_ev = true; }
            else if (std::sscanf(buf.c_str(), "first %u %llu", &p, &a) == 2) {
                if (p <= primes_.back() && pi_of_[p / 2] != 0xFFFF) first_[pi_of_[p / 2]] = a;
            } else if (std::sscanf(buf.c_str(), "beyond %llu %llu", &c, &a) == 2) tot_beyond_.push_back({a, c});
            else if (buf.compare(0, 9, "primes30 ") == 0) {
                const char* s = buf.c_str() + 9;
                for (int i = 0; i < 8 && s; ++i) { tot_primes_[i] = std::strtoull(s, nullptr, 10); s = std::strchr(s + 1, ' '); }
                seen_p = true;
            } else if (buf.compare(0, 8, "twins30 ") == 0) {
                const char* s = buf.c_str() + 8;
                for (int i = 0; i < 3 && s; ++i) { tot_twins_[i] = std::strtoull(s, nullptr, 10); s = std::strchr(s + 1, ' '); }
                seen_tw = true;
            } else if (std::sscanf(buf.c_str(), "sampled %u ", &cr) == 1 && cr < 15) {
                const auto v = parse_dense(std::strchr(buf.c_str() + 8, ' ') + 1, np);
                std::copy(v.begin(), v.end(), cum_samp_.begin() + cr * np);
                seen_samp |= 1u << cr;
            } else if (std::sscanf(buf.c_str(), "tailcls %u ", &cr) == 1 && cr < 15) {
                const auto v = parse_dense(std::strchr(buf.c_str() + 8, ' ') + 1, np);
                std::copy(v.begin(), v.end(), cum_tail_.begin() + cr * np);
                seen_tail |= 1u << cr;
            } else if (buf.compare(0, 8, "tailagg ") == 0) {
                const auto v = parse_dense(buf.c_str() + 8, np);
                for (size_t i = 0; i < np; ++i) cum_tail_agg_[primes_[i] / 2] = v[i];
                seen_agg = true;
            }
        }
        if (!(seen_t && seen_ev && seen_p && seen_tw && seen_agg &&
              seen_samp == 0x7FFFu && seen_tail == 0x7FFFu))
            throw std::runtime_error("stats: state " + state_path_ + " is missing records "
                                     "(truncated or damaged); refusing to merge into it");
        merge_covered();
    }

    void save_state()
    {
        const size_t np = primes_.size();
        std::string out =
            "# goldbach-l0 statistics state v2 (exact; include/gstats.hpp)\n"
            "# lists: counts by odd-prime index (3, 5, 7, 11, ...), empty = 0\n"
            "# sampled <n mod 30 / 2>: p(n) histogram of the sampled segments\n"
            "# tailcls <n mod 30 / 2> / tailagg: p(n) >= tail_t, all evens (tailagg: class not kept)\n"
            "# first <p> <n>: first even of the campaign with p(n) = p (tail p only)\n";
        char b[160];
        std::snprintf(b, sizeof b, "tail_t %u\n", tail_t_);
        out += b;
        for (auto& c : covered_) {
            std::snprintf(b, sizeof b, "range %llu %llu\n", (unsigned long long)c.first, (unsigned long long)c.second);
            out += b;
        }
        std::snprintf(b, sizeof b, "evens %llu %llu\n", (unsigned long long)tot_evens_, (unsigned long long)tot_sampled_);
        out += b;
        out += "primes30";
        for (int i = 0; i < 8; ++i) { std::snprintf(b, sizeof b, " %llu", (unsigned long long)tot_primes_[i]); out += b; }
        out += "\ntwins30";
        for (int i = 0; i < 3; ++i) { std::snprintf(b, sizeof b, " %llu", (unsigned long long)tot_twins_[i]); out += b; }
        out += "\n";
        for (size_t i = 0; i < np; ++i)
            if (first_[i]) {
                std::snprintf(b, sizeof b, "first %u %llu\n", primes_[i], (unsigned long long)first_[i]);
                out += b;
            }
        for (auto& e : tot_beyond_) {
            std::snprintf(b, sizeof b, "beyond %llu %llu\n", (unsigned long long)e.second, (unsigned long long)e.first);
            out += b;
        }
        for (uint32_t cr = 0; cr < 15; ++cr) {
            std::vector<uint64_t> v(cum_samp_.begin() + cr * np, cum_samp_.begin() + (cr + 1) * np);
            std::snprintf(b, sizeof b, "sampled %u ", cr);
            out += b + dense_list(v) + "\n";
        }
        for (uint32_t cr = 0; cr < 15; ++cr) {
            std::vector<uint64_t> v(cum_tail_.begin() + cr * np, cum_tail_.begin() + (cr + 1) * np);
            std::snprintf(b, sizeof b, "tailcls %u ", cr);
            out += b + dense_list(v) + "\n";
        }
        std::vector<uint64_t> agg(np, 0);
        for (size_t i = 0; i < np; ++i) agg[i] = cum_tail_agg_[primes_[i] / 2];
        out += "tailagg " + dense_list(agg) + "\n";
        // The caller (run_range) journals the slice only after this process
        // exits, so a journaled PASS never outlives its statistics across a crash.
        write_durably(state_path_, out);
    }

    /// Durable atomic replace: write + fsync a temp file, rename it over
    /// @p path, fsync the directory.
    /// @throws std::runtime_error on any failure (@p path is then unchanged).
    static void write_durably(const std::string& path, const std::string& out)
    {
        const std::string tmp = path + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) throw std::runtime_error("stats: cannot write " + tmp);
        bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
        ok = (std::fflush(f) == 0) && ok;
        ok = (::fsync(::fileno(f)) == 0) && ok;
        ok = (std::fclose(f) == 0) && ok;
        if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
            std::remove(tmp.c_str());
            throw std::runtime_error("stats: cannot write " + path + " (state unchanged)");
        }
        const size_t sl = path.find_last_of('/');
        const std::string dir = sl == std::string::npos ? "." : (sl == 0 ? "/" : path.substr(0, sl));
        const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd >= 0) { ::fsync(dfd); ::close(dfd); }
    }

    const std::vector<uint32_t>& primes_;
    const StatsPlan& plan_;
    const uint32_t tail_t_;
    std::vector<uint16_t> pi_of_;              // p >> 1 -> prime index (0xFFFF: not a candidate)
    std::string state_path_, log_path_;
    // campaign (state file)
    std::vector<uint64_t> cum_samp_, cum_tail_;   // [class * primes + prime index]
    std::vector<uint64_t> cum_tail_agg_ = std::vector<uint64_t>(9783 / 2 + 1, 0);
    std::vector<uint64_t> first_;              // n_min per prime index (0 = unseen; tail p only)
    std::vector<std::pair<uint64_t, uint64_t>> covered_;
    std::vector<std::pair<uint64_t, uint64_t>> tot_beyond_;   // (n, p > MAX_GB_PRIME)
    uint64_t tot_evens_ = 0, tot_sampled_ = 0, tot_primes_[8] = {}, tot_twins_[3] = {};
    // this run
    std::vector<uint64_t> samp_, tail_;        // [class * primes + prime index]
    std::vector<uint64_t> tail_agg_ = std::vector<uint64_t>(9783 / 2 + 1, 0);   // by p >> 1
    std::vector<uint64_t> run_first_;
    std::vector<std::pair<uint64_t, uint64_t>> beyond_;
    uint64_t primes_res_[8] = {}, twins_[3] = {};
    GapStats gaps_;
    uint64_t gap_last_ = 0, gap_start_ = ~0ull;  // last prime stitched; the run start (begin_gaps)
    bool gap_last_walked_ = false;
    uint64_t walk_evens_ = 0;                  // evens of the walked segments
    uint64_t start_ = 0, limit_ = 0, evens_ = 0, evens_sampled_ = 0;
    std::vector<uint32_t> want_p_;
    std::string rec_path_, live_path_;
    std::vector<uint16_t> unknown_;            // S_UNKNOWN_4E18, sorted
    std::vector<std::pair<uint64_t, uint64_t>> events_;   // (n, p) recorded this run
    std::time_t live_t_ = 0;
};

} // namespace gb
