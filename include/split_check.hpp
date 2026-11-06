#pragma once
// Host side of the class-split checker (v1.11.0; kernels split_map,
// check_split_a, check_split_b in kernels/goldbach.cl — the header comment
// there explains the method). Shared by the verifier and tools/debug_check_hi.
//
// Evens are grouped by n mod 3 ("class" c) and the map by q mod 6:
//   c = 0: candidates p = 1 (mod 6) on S5      c = 2: p = 1 (mod 6) on S1
//   c = 1: candidates p = 5 (mod 6) on S5
// Phase A funnels test several candidates at once: a stage mask bit d means
// "also p + 6d", used only where every p + 6d is prime (and <= the
// candidate bound), so each OR'd map bit is a genuine witness.

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gb {

inline constexpr uint32_t SPLIT_MAXD = 2;   // largest pattern shift: MUST match the kernel's SPLIT_MAXD

struct SplitPlan {
    std::vector<uint16_t> cands;          // phase-A funnels: list L1, then L5
    std::array<uint32_t, 32> meta{};      // per list (L1 at 0, L5 at 16): n_stages, (begin, end, mask) x n
    std::vector<uint16_t> rest;           // phase-B probes for classes 0, 1, 2 (concatenated)
    uint32_t roff[3] = {}, rcnt[3] = {};
    uint32_t max_member = 0;              // largest candidate p + 6d tested in phase A
};

/// Builds the class-split checker plan: greedy phase-A funnels per list (L1:
/// p = 1 mod 6, L5: p = 5 mod 6), stage by stage, taking candidates whose
/// every pattern member p + 6d is a prime not yet covered; then per class the
/// phase-B probes (all candidates phase A did not test, futile ones last).
///
/// @param primes          odd candidate primes, ascending (the bound is the last).
/// @param cap             funnels per list (over all stages).
/// @param stage_masks_in  one mask per stage (bit 0 set, shifts <= SPLIT_MAXD);
///                        bits 16+ optionally hold a per-stage funnel limit
///                        (parse_split_stages "mask:limit").
/// @pre    every prime fits u16.
/// @post   phase A + phase B together test every candidate of every class.
/// @return the plan (cands, meta, rest, roff/rcnt, max_member).
/// @throws std::runtime_error on 0 or > 5 stages or an invalid stage mask.
inline SplitPlan make_split_plan(const std::vector<uint32_t>& primes, uint32_t cap,
                                 const std::vector<uint32_t>& stage_masks_in)
{
    std::vector<uint32_t> stage_masks, stage_lims;
    for (uint32_t v : stage_masks_in) {
        stage_masks.push_back(v & 0xFFFFu);
        stage_lims.push_back((v >> 16) ? (v >> 16) : 0xFFFFu);
    }
    if (stage_masks.empty() || stage_masks.size() > 5)
        throw std::runtime_error("split checker: 1..5 stages");
    const uint32_t bound = primes.empty() ? 0 : primes.back();
    std::vector<char> isp(bound + 1, 0);
    for (uint32_t p : primes) isp[p] = 1;
    SplitPlan plan;
    std::vector<char> cov1(bound + 1, 0), cov5(bound + 1, 0);
    for (int li = 0; li < 2; ++li) {
        const uint32_t r = li == 0 ? 1u : 5u;
        std::vector<char>& cov = li == 0 ? cov1 : cov5;
        const uint32_t mo = li == 0 ? 0u : 16u;
        const uint32_t list_begin = uint32_t(plan.cands.size());
        uint32_t ns = 0;
        for (size_t si = 0; si < stage_masks.size(); ++si) {
            const uint32_t mask = stage_masks[si];
            if (!(mask & 1u) || (mask >> (SPLIT_MAXD + 1)))
                throw std::runtime_error("split checker: stage mask needs bit 0 and shifts <= SPLIT_MAXD");
            const uint32_t b = uint32_t(plan.cands.size());
            for (uint32_t p : primes) {
                if (plan.cands.size() - list_begin >= cap) break;
                if (plan.cands.size() - b >= stage_lims[si]) break;
                if (p % 6 != r) continue;
                bool ok = true;
                for (uint32_t d = 0; d <= 12 && ok; ++d) {
                    if (!((mask >> d) & 1u)) continue;
                    const uint32_t x = p + 6 * d;
                    ok = x <= bound && isp[x] && !cov[x];
                }
                if (!ok) continue;
                for (uint32_t d = 0; d <= 12; ++d)
                    if ((mask >> d) & 1u) {
                        cov[p + 6 * d] = 1;
                        plan.max_member = std::max(plan.max_member, p + 6 * d);
                    }
                plan.cands.push_back(uint16_t(p));
            }
            const uint32_t e = uint32_t(plan.cands.size());
            if (e > b) {
                plan.meta[mo + 1 + 3 * ns] = b;
                plan.meta[mo + 2 + 3 * ns] = e;
                plan.meta[mo + 3 + 3 * ns] = mask;
                ++ns;
            }
        }
        plan.meta[mo] = ns;
    }
    // Phase B per class: useful untested candidates ascending, then the
    // futile ones (q divisible by 3: prime only for q = 3), so phase A + B
    // test every candidate.
    for (uint32_t c = 0; c < 3; ++c) {
        plan.roff[c] = uint32_t(plan.rest.size());
        std::vector<uint16_t> futile;
        for (uint32_t p : primes) {
            bool useful, tested;
            if (c == 0) { useful = p != 3; tested = p % 6 == 1 && cov1[p]; }
            else if (c == 2) { useful = p == 3 || p % 6 == 1; tested = p % 6 == 1 && cov1[p]; }
            else { useful = p == 3 || p % 6 == 5; tested = p % 6 == 5 && cov5[p]; }
            if (tested) continue;
            if (useful) plan.rest.push_back(uint16_t(p));
            else futile.push_back(uint16_t(p));
        }
        plan.rest.insert(plan.rest.end(), futile.begin(), futile.end());
        plan.rcnt[c] = uint32_t(plan.rest.size()) - plan.roff[c];
    }
    return plan;
}

/// Parses a stage spec: '/'-separated masks (decimal or 0x), each with an
/// optional ":limit" on that stage's funnels, e.g. "7:23/3" -> {7 | 23 << 16, 3}.
/// @return one encoded stage per field, in order (not validated here; see
///         make_split_plan).
/// @throws std::invalid_argument / std::out_of_range from std::stoul on a
///         malformed number.
inline std::vector<uint32_t> parse_split_stages(const std::string& s)
{
    std::vector<uint32_t> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find('/', i);
        if (j == std::string::npos) j = s.size();
        const std::string t = s.substr(i, j - i);
        const size_t c = t.find(':');
        uint32_t m = uint32_t(std::stoul(t.substr(0, c), nullptr, 0));
        if (c != std::string::npos) m |= uint32_t(std::stoul(t.substr(c + 1), nullptr, 0)) << 16;
        v.push_back(m);
        i = j + 1;
    }
    return v;
}

// Per-segment parameters of the split kernels.
struct SplitSeg {
    bool ok = false;                // geometry supports the split checker
    uint32_t phi1 = 0, phi5 = 0;    // dense bit phase of q = 1 / 5 (mod 6)
    uint32_t e0[3] = {}, ec[3] = {};// per class: first even index, even count
    int32_t v[3] = {};              // per class: U(p) = (v - p) / 6
};

/// Computes the split kernels' per-segment parameters (bit phases, per-class
/// even ranges and funnel base offsets).
/// @param seg_start  first even n of the segment.
/// @param seg_evens  number of evens in the segment.
/// @param q_first    q of dense map bit 0 (odd).
/// @param plan       unused by the math; kept for the call signature.
/// @return parameters with ok = true, or ok = false when the geometry is
///         outside the kernels' ranges (seg_start - q_first > 2^20, >= 2^32
///         evens, offset not divisible by 6 or > 2^24) — the caller must then
///         use another checker.
inline SplitSeg split_segment(uint64_t seg_start, uint64_t seg_evens, uint64_t q_first,
                              const SplitPlan& /*plan*/)
{
    SplitSeg s;
    if (seg_start < q_first || seg_start - q_first > (1u << 20) || seg_evens >= (1ull << 32))
        return s;
    const int64_t tb = int64_t(seg_start - q_first);
    const uint32_t qf6 = uint32_t(q_first % 6);          // odd
    s.phi1 = ((1 + 6 - qf6) % 6) / 2;
    s.phi5 = ((5 + 6 - qf6) % 6) / 2;
    const uint32_t sm3 = uint32_t(seg_start % 3);
    // At the bottom of the range (tb < max_member) funnel offsets can be
    // negative: the kernel floors them and reads split words below 0 as 0
    // (q < q_first cannot be a witness there: q < 3).
    for (uint32_t c = 0; c < 3; ++c) {
        uint32_t e0 = 0;
        while ((sm3 + 2 * e0) % 3 != c) ++e0;
        s.e0[c] = e0;
        s.ec[c] = seg_evens > e0 ? uint32_t((seg_evens - e0 + 2) / 3) : 0u;
        const uint32_t phi = (c == 2) ? s.phi1 : s.phi5;
        const int64_t v = tb + 2 * (int64_t(e0) - int64_t(phi));
        const int64_t r = (c == 1) ? 5 : 1;
        if (((v - r) % 6 + 6) % 6 != 0 || v > (1 << 24))   // exact division by 6
            return s;
        s.v[c] = int32_t(v);
    }
    s.ok = true;
    return s;
}

} // namespace gb
