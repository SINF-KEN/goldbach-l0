#pragma once
// Host side of the mod-30 checker (v1.13.0; kernels check30_a / check30_b in
// kernels/goldbach.cl). Shared by the verifier and tools/debug_check_hi.
//
// Evens are grouped by n mod 30 (15 classes r = 0, 2, ..., 28) and read the
// mod-30 class map directly (no merge30): for a candidate p = s (mod 30),
// q = n - p lies in the class array of residue r - s, and consecutive evens
// of class r (n, n + 30, ...) are consecutive bits of that array. Only the
// candidate classes s with r - s coprime to 30 are used, so every phase-A test
// can succeed (the mod-6 checker spent 1/4 of its tests on q = 0 (mod 5) for
// 4/5 of the evens). A funnel ORs pattern shifts: mask bit d = candidate
// p + 30d (same class, same array), used only where every p + 30d is prime;
// 30 = 2 (mod 7) allows up to six members, and 4-5 member patterns are
// plentiful among small primes.
//
// The plan is a list of segments per class r, each (s, mask, funnels):
// phase A takes the first `funnels` candidates p = s (mod 30) in ascending
// order whose members are all prime and not yet covered for (r, s). Chosen
// by a greedy on a real map at 4e18 (Miller-Rabin-exact sieve of 2e6 evens):
// add funnels to a segment or open a new one, weighing a segment start (window
// reload + pattern OR) at ~14 funnels as measured, until 0.1% of the evens
// are left for phase B. Plans from reload weights 10-16 and targets
// 0.07-0.15% all measured within 1% of each other (phase A + B per 425M
// segment ~0.435-0.445 ms). Correctness does not depend on the plan: phase B
// probes every candidate phase A did not test.

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace gb {

inline constexpr uint32_t C30_MAXD = 4;       // largest pattern shift: MUST match the kernel
inline constexpr uint32_t C30_MAXSEG = 15;    // segments per class (meta stride 32)
inline constexpr uint32_t C30_UNITS[8] = {1, 7, 11, 13, 17, 19, 23, 29};

// Default plan (r = 0, 2, ..., 28; "s/mask x funnels"), v2.2: every funnel
// member below CHECK30_TAIL = 3000 (the "3000:" prefix), so every even with
// p(n) >= 3000 fails phase A and phase B finds its exact p(n) (complete tail
// statistics and ridges, include/gstats.hpp). Greedy plan (window reload
// weighed as 10 funnels, 84 members per class) measured at 4e18 per 425M
// segment: phase A + B 0.383 ms vs 0.402 ms for the v1.13.0 plan (members up
// to 9749), 0.046% of the evens left for phase B.
inline constexpr uint32_t CHECK30_TAIL = 3000;
inline constexpr const char* CHECK30_PLAN =
    "3000:"
    "7/7x9,11/7x9,17/7x9,1/31x1;"
    "1/7x8,13/7x8,19/3x17,1/3x1;"
    "11/7x9,17/7x9,23/7x7,11/3x5;"
    "7/7x9,17/7x9,13/7x8,19/15x2;"
    "7/7x9,1/7x8,19/7x7,7/3x6;"
    "11/7x9,17/7x9,23/7x7,29/31x2;"
    "11/7x9,29/3x21,1/31x3;"
    "7/7x9,1/7x8,13/7x8,1/3x5;"
    "17/7x9,29/3x21,23/7x5;"
    "7/7x9,11/7x9,17/7x9,1/31x1;"
    "7/7x9,1/7x8,13/7x8,19/15x3;"
    "11/7x9,29/3x21,23/7x5;"
    "7/7x9,11/7x9,17/7x9,1/31x1;"
    "7/7x9,13/7x8,19/7x7,7/3x6;"
    "11/7x9,17/7x9,29/7x7,11/3x5";

/// @return true iff @p x is coprime to 30 (a unit mod 30).
inline bool c30_unit(uint32_t x) { return std::gcd(x % 30u, 30u) == 1; }

struct Plan30 {
    std::vector<uint16_t> cands;          // phase-A funnel candidates (lowest member)
    std::vector<uint32_t> meta;           // 15 x 32: n_seg, (begin | end << 16, s | mask << 8) x n_seg
    std::vector<uint16_t> rest;           // phase-B probes per class, concatenated
    std::vector<uint32_t> rtab;           // [0, 15): rest offset per class, [15, 30): count
    uint32_t max_member = 0;              // largest p + 30d tested in phase A
    uint32_t member_lt = 0;               // spec bound: every member < member_lt (tail bound)
};

/// Builds the mod-30 checker plan from a spec such as CHECK30_PLAN: 15 ';'-
/// separated classes (r = 0, 2, ..., 28), each a ','-separated list of
/// "s/mask x funnels" segments. Each segment takes the first `funnels`
/// candidates p = s (mod 30), ascending, whose members p + 30d (mask bit d)
/// are all prime and not yet covered for the class. Phase-B probes per class
/// are every candidate not tested in phase A (futile ones last).
///
/// @param primes  odd candidate primes, ascending (the bound is the last).
/// @param spec    plan string (see CHECK30_PLAN).
/// @pre    every prime fits u16.
/// @post   phase A + phase B together test every candidate of every class.
/// @return the plan (cands, meta, rest, rtab, max_member).
/// @throws std::runtime_error on a malformed or inconsistent spec (unusable
///         s for the class, bad mask, > C30_MAXSEG segments, > 15 classes,
///         >= 65536 candidates).
inline Plan30 make_plan30(const std::vector<uint32_t>& primes, const std::string& spec_in)
{
    const uint32_t bound = primes.empty() ? 0 : primes.back();
    // Optional "T:" prefix: every phase-A member must be < T, so every even
    // with p(n) >= T fails phase A and gets its exact p(n) in phase B.
    std::string spec = spec_in;
    uint32_t member_lt = bound + 1;
    if (const size_t c = spec.find(':'); c != std::string::npos) {
        member_lt = uint32_t(std::stoul(spec.substr(0, c)));
        spec = spec.substr(c + 1);
    }
    std::vector<char> isp(bound + 1, 0);
    for (uint32_t p : primes) isp[p] = 1;
    Plan30 plan;
    plan.meta.assign(15 * 32, 0);
    plan.rtab.assign(30, 0);
    size_t pos = 0;
    for (uint32_t cr = 0; cr < 15; ++cr) {
        const uint32_t r = 2 * cr;
        size_t end = spec.find(';', pos);
        if (end == std::string::npos) end = spec.size();
        const std::string cls = spec.substr(pos, end - pos);
        pos = end + 1;
        std::vector<char> cov(bound + 1, 0);      // members tested for this class
        uint32_t ns = 0;
        size_t i = 0;
        while (i < cls.size()) {
            size_t j = cls.find(',', i);
            if (j == std::string::npos) j = cls.size();
            const std::string sg = cls.substr(i, j - i);
            i = j + 1;
            const size_t sl = sg.find('/'), x = sg.find('x');
            if (sl == std::string::npos || x == std::string::npos || x < sl)
                throw std::runtime_error("check30 plan: segment '" + sg + "' is not s/mask x n");
            const uint32_t s = uint32_t(std::stoul(sg.substr(0, sl)));
            const uint32_t mask = uint32_t(std::stoul(sg.substr(sl + 1, x - sl - 1), nullptr, 0));
            const uint32_t nf = uint32_t(std::stoul(sg.substr(x + 1)));
            if (!c30_unit(s) || s >= 30 || !c30_unit(r + 30 - s))
                throw std::runtime_error("check30 plan: class " + std::to_string(r) +
                                         " cannot use candidates = " + std::to_string(s) + " (mod 30)");
            if (!(mask & 1u) || (mask >> (C30_MAXD + 1)))
                throw std::runtime_error("check30 plan: mask needs bit 0 and shifts <= C30_MAXD");
            if (ns >= C30_MAXSEG) throw std::runtime_error("check30 plan: too many segments");
            const uint32_t b = uint32_t(plan.cands.size());
            for (uint32_t p : primes) {
                if (plan.cands.size() - b >= nf) break;
                if (p % 30 != s) continue;
                bool ok = true;
                for (uint32_t d = 0; d <= C30_MAXD && ok; ++d)
                    if ((mask >> d) & 1u) {
                        const uint32_t q = p + 30 * d;
                        ok = q <= bound && q < member_lt && isp[q] && !cov[q];
                    }
                if (!ok) continue;
                for (uint32_t d = 0; d <= C30_MAXD; ++d)
                    if ((mask >> d) & 1u) {
                        cov[p + 30 * d] = 1;
                        plan.max_member = std::max(plan.max_member, p + 30 * d);
                    }
                plan.cands.push_back(uint16_t(p));
            }
            const uint32_t e = uint32_t(plan.cands.size());
            if (e > b) {
                if (e >= 65536) throw std::runtime_error("check30 plan: too many candidates");
                plan.meta[cr * 32 + 1 + 2 * ns] = b | (e << 16);
                plan.meta[cr * 32 + 2 + 2 * ns] = s | (mask << 8);
                ++ns;
            }
        }
        plan.meta[cr * 32] = ns;
        // Phase B: untested useful candidates ascending, then the futile ones
        // (q divisible by 3 or 5: prime only for q = 3, 5), so phase A + B test
        // every candidate. p = 3, 5 are never in phase A.
        plan.rtab[cr] = uint32_t(plan.rest.size());
        std::vector<uint16_t> futile;
        for (uint32_t p : primes) {
            if (cov[p] && c30_unit(r + 30 - p % 30)) continue;   // tested in phase A
            if (c30_unit(r + 30 - p % 30)) plan.rest.push_back(uint16_t(p));
            else futile.push_back(uint16_t(p));
        }
        plan.rest.insert(plan.rest.end(), futile.begin(), futile.end());
        plan.rtab[15 + cr] = uint32_t(plan.rest.size()) - plan.rtab[cr];
    }
    if (pos < spec.size()) throw std::runtime_error("check30 plan: more than 15 classes");
    plan.member_lt = std::min(member_lt, plan.max_member + 1);
    return plan;
}

// Per-segment kernel parameters.
struct Seg30 {
    bool ok = false;
    uint32_t sm30 = 0;       // seg_start mod 30
    uint32_t qf30 = 0;       // q_first mod 30
    uint32_t phik = 0;       // class k's dense phase (b = 15t + phi_k), nibble k
    int32_t tb = 0;          // seg_start - q_first
    uint32_t ec_max = 0;     // evens of the largest class
};

/// Per-segment parameters of the mod-30 checker. Class k of the mod-30 map
/// holds q = q_first + 2 phi_k + 30 t, residue C30_UNITS[k] (the verifier's
/// class_geometry with K = 8).
/// @param seg_start  first even n of the segment (even).
/// @param seg_evens  number of evens in the segment.
/// @param q_first    q of dense map bit 0 (odd).
/// @pre    none (inputs outside the kernels' ranges give ok = false).
/// @return parameters with ok = true, or ok = false when the inputs have the
///         wrong parity or exceed the kernels' ranges (seg_start - q_first >
///         2^20, >= 2^32 evens).
inline Seg30 seg30(uint64_t seg_start, uint64_t seg_evens, uint64_t q_first)
{
    Seg30 s;
    if (seg_start < q_first || seg_start - q_first > (1u << 20) || seg_evens >= (1ull << 32) ||
        seg_start % 2 || q_first % 2 == 0)
        return s;
    s.sm30 = uint32_t(seg_start % 30);
    s.qf30 = uint32_t(q_first % 30);
    for (uint32_t k = 0; k < 8; ++k)
        s.phik |= (((C30_UNITS[k] + 30 - s.qf30) % 30) / 2) << (4 * k);
    s.tb = int32_t(seg_start - q_first);
    s.ec_max = uint32_t((seg_evens + 14) / 15);
    s.ok = true;
    return s;
}

} // namespace gb
