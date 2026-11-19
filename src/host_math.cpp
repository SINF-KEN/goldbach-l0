// Pure host math of the dense pipeline (see include/host_math.hpp for the
// contracts). Moved verbatim from src/verifier.cpp so host_tests can check
// it against brute force; no Level Zero dependency.

#include "host_math.hpp"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <string>

namespace gb {

SegmentGeometry segment_geometry(uint64_t seg_start, uint64_t limit,
                                 uint64_t segment_even, uint64_t segment_index) {
    SegmentGeometry g;
    g.seg_start = seg_start;
    g.slot = uint32_t(segment_index & 1);
    // Merge the usual one-even inclusive tail into the preceding segment: a
    // tiny final segment would otherwise repeat the entire ~98M-prime setup
    // and dispatch merely to verify the target endpoint.
    const uint64_t remaining_span = limit - seg_start;
    g.seg_end = remaining_span <= segment_even
              ? limit
              : seg_start + segment_even - 2;  // inclusive even
    g.seg_evens = (g.seg_end - g.seg_start) / 2 + 1;
    // q_first must be ODD so the kernel's shift T = (seg_start - p - q_first)/2
    // is an exact integer for every odd p. seg_start is even and MAX_GB_PRIME
    // is odd, so seg_start - MAX_GB_PRIME is odd already; subtract 1 more only
    // if needed to stay strictly below seg_start - MAX_GB_PRIME.
    uint64_t qf = seg_start - MAX_GB_PRIME;      // even - odd = odd
    if (qf >= seg_start) qf -= 1;                // safety (never triggers)
    g.q_first = (seg_start >= MAX_GB_PRIME + 2) ? qf : 3;
    g.q_last  = g.seg_end - 3;
    // A segment holding n = 4 alone has no odd q (4 = 2 + 2): q_last = 1 <
    // q_first = 3 would wrap q_odds to 2^63 and size every launch from it.
    // run_gpu never builds one (segment span >= 4 at start 4, limit >= 6).
    if (g.q_last < g.q_first)
        throw std::runtime_error("segment [" + std::to_string(seg_start) + ", " +
                                 std::to_string(g.seg_end) + "] has no odd q to sieve");
    g.q_odds  = (g.q_last - g.q_first) / 2 + 1;
    g.tiles   = (g.q_odds + BITS_PER_TILE - 1) / BITS_PER_TILE;
    return g;
}

std::vector<uint64_t> compute_first_bits(const uint32_t* sieve_primes,
                                         uint32_t sieve_split,
                                         uint64_t q_first, uint64_t q_last) {
    std::vector<uint64_t> fb(sieve_split);
    for (uint32_t j = 0; j < sieve_split; ++j) {
        const uint64_t p = sieve_primes[j];
        if (p > q_last / p) { fb[j] = ~0ull; continue; }  // p*p > q_last
        uint64_t first = p * p;
        if (first < q_first) {
            first = ((q_first + p - 1) / p) * p;
            if ((first & 1) == 0) first += p;
        }
        if (first > q_last) { fb[j] = ~0ull; continue; }
        fb[j] = (first - q_first) >> 1;
    }
    return fb;
}

ClassGeo class_geometry(uint64_t q_first, uint64_t q_odds, uint32_t K) {
    ClassGeo g;
    g.K = K;
    g.qmod = K == 8 ? 30 : 6;
    const uint32_t dm = g.qmod / 2;                           // dense bits per class bit
    const uint32_t r0 = uint32_t(q_first % g.qmod);           // odd
    uint64_t nmax = 0;
    for (uint32_t c = 0; c < K; ++c) {
        g.phi[c] = ((g.res(c) + g.qmod - r0) % g.qmod) / 2;
        g.Q[c] = q_first + 2 * uint64_t(g.phi[c]);
        g.n[c] = q_odds > g.phi[c] ? (q_odds - g.phi[c] + dm - 1) / dm : 0;
        nmax = std::max(nmax, g.n[c]);
    }
    g.tpc = uint32_t(std::max<uint64_t>((nmax + BITS_PER_TILE - 1) / BITS_PER_TILE, 1));
    return g;
}

void first_bits_class_range(uint64_t* fb, const uint32_t* sieve_primes,
                            uint32_t sieve_split, uint32_t j0, uint32_t j1,
                            uint64_t q_first, uint64_t q_last, const ClassGeo& g,
                            uint32_t cop_j0) {
    static_assert(30 * COP_INV30 % COP_M == 1);
    for (uint32_t c = 0; c < g.K; ++c)
        std::fill(fb + size_t(c) * sieve_split + j0, fb + size_t(c) * sieve_split + j1, ~0ull);
    const uint64_t M = g.qmod;
    // the residues exist for the mod-30 map only: a K = 2 map has no
    // med_prime_cop, so a residue column there is a caller bug, not c = 0
    if (cop_j0 < j1 && M != 30)
        throw std::runtime_error("first_bits_class_range: multiplier residues need K = 8");
    for (uint32_t j = j0; j < j1; ++j) {
        const uint64_t p = sieve_primes[j];
        if (p % 3 == 0 || (M == 30 && p % 5 == 0) || p > q_last / p) continue;
        uint64_t pinv = 1;                                // p^-1 mod M: p^(phi(M) - 1)
        for (uint32_t e = 0; e + 1 < (M == 30 ? 8u : 2u); ++e) pinv = pinv * (p % M) % M;
        const uint64_t first = std::max(p * p, q_first);
        const uint64_t m0 = (first + p - 1) / p;
        for (uint32_t c = 0; c < g.K; ++c) {
            const uint64_t mt = (g.res(c) * pinv) % M;
            const uint64_t m = m0 + (mt + M - m0 % M) % M;
            const uint64_t q = m * p;
            if (q > q_last) continue;
            const uint64_t bit = (q - g.Q[c]) / M;
            const bool has_cop = j >= cop_j0;
            const uint64_t cop = has_cop ? m % COP_M * COP_INV30 % COP_M : 0;
            // checked for every residue column (c = 0 is a valid residue)
            if (has_cop && bit >> 32)
                throw std::runtime_error("first_bits_class_range: class bit >= 2^32 with a residue");
            fb[size_t(c) * sieve_split + j] = bit | cop << 32;
        }
    }
}

void build_tile_first_table(uint32_t* tbl, uint32_t tiles, uint32_t sieve_split,
                            uint32_t own_end,
                            const uint64_t* fb_all, const uint64_t* magic,
                            const uint32_t* shift, const uint32_t* sieve_primes,
                            uint32_t tpc) {   // class mode: rows [tpc, 2 tpc) = M5 tiles
    // Only the thread-owned columns [0, own_end) are read by the kernel
    // (medium primes derive their tile start from fb), so rows hold own_end.
    parallel_for(0, tiles, 16, [&](uint32_t t0, uint32_t t1) {
        for (uint32_t t = t0; t < t1; ++t) {
            uint32_t* row = tbl + size_t(t) * own_end;
            const uint32_t cls = tpc ? t / tpc : 0u;
            const uint64_t* fb = fb_all + size_t(cls) * sieve_split;
            const uint64_t tb0 = uint64_t(t - cls * tpc) * BITS_PER_TILE;
            for (uint32_t j = 0; j < own_end; ++j) {
                const uint64_t f = fb[j];
                if (f == ~0ull) { row[j] = UINT32_MAX; continue; }
                if (f >= tb0) {
                    row[j] = (f - tb0 < BITS_PER_TILE)
                           ? uint32_t(f - tb0) : UINT32_MAX;
                    continue;
                }
                const uint64_t p = sieve_primes[j];
                const uint64_t d = tb0 - f;
                uint64_t k = uint64_t(((__uint128_t)d * magic[j]) >> (64 + shift[j]));
                k += (k * p < d) ? 1ull : 0ull;
                const uint64_t bit = f + k * p - tb0;
                row[j] = (bit < BITS_PER_TILE) ? uint32_t(bit) : UINT32_MAX;
            }
        }
    });
}

std::pair<std::vector<uint64_t>, std::vector<uint32_t>>
build_magic_reciprocals(const uint32_t* sieve_primes, uint32_t sieve_split) {
    std::vector<uint64_t> magic(sieve_split);
    std::vector<uint32_t> shift(sieve_split);
    for (size_t i = 0; i < sieve_split; ++i) {
        const uint64_t p = sieve_primes[i];
        const uint64_t m = (uint64_t)(((__uint128_t)1 << 64) / p) + 1;
        magic[i] = m;
        uint32_t s = 0;
        for (;;) {
            bool ok = true;
            for (uint64_t x : {1ull, 2ull, 63ull, 4095ull, 4096ull, 65535ull,
                               1048575ull, 1048576ull, (1ull << 32) - 1, 1ull << 32,
                               (1ull << 33) - 1}) {
                const uint64_t got = (uint64_t)(((__uint128_t)x * m) >> (64 + s));
                if (got != x / p) { ok = false; break; }
            }
            if (ok) break;
            ++s;
            if (s > 32) { s = 0; break; }  // fallback: kernel correction handles it
        }
        shift[i] = s;
    }
    return {magic, shift};
}

std::vector<uint64_t> build_tile_reciprocals32(const uint32_t* sieve_primes, uint32_t sieve_split) {
    std::vector<uint64_t> ms32(sieve_split, 0ull);
    for (size_t i = 0; i < sieve_split; ++i) {
        const uint64_t p = sieve_primes[i];
        if (p < BITS_PER_TILE) {
            uint32_t s = 0;
            while ((uint64_t(1) << s) <= BITS_PER_TILE * p) ++s;
            const uint64_t m = ((uint64_t(1) << s) + p - 1) / p;
            ms32[i] = m | (uint64_t(s) << 32);
        }
    }
    return ms32;
}

PackedLarge pack_large_primes(const uint32_t* primes, uint32_t split, uint32_t count) {
    PackedLarge pk;
    const uint64_t n = count - split;
    const uint64_t items = (n + LARGE_PACK_PPI - 1) / LARGE_PACK_PPI;
    pk.gaps.assign(std::max<uint64_t>(items, 1) * LARGE_PACK_PPI, 0);
    pk.bases.assign(std::max<uint64_t>((n + LARGE_PACK_GROUP - 1) / LARGE_PACK_GROUP, 1), 0);
    // Groups are independent: pack them in parallel (51M primes at 4e18).
    const uint64_t groups = (n + LARGE_PACK_GROUP - 1) / LARGE_PACK_GROUP;
    std::atomic<bool> bad{false};
    parallel_for(0, uint32_t(groups), 16, [&](uint32_t g0, uint32_t g1) {
        for (uint64_t i = uint64_t(g0) * LARGE_PACK_GROUP;
             i < std::min<uint64_t>(n, uint64_t(g1) * LARGE_PACK_GROUP); ++i) {
            const uint32_t p = primes[split + i];
            if (i % LARGE_PACK_GROUP == 0) { pk.bases[i / LARGE_PACK_GROUP] = p; continue; }
            const uint32_t gap = p - primes[split + i - 1];
            if ((gap & 1) || gap / 2 > 0xFF) { bad = true; return; }
            pk.gaps[i] = uint8_t(gap / 2);
        }
    });
    if (bad)
        throw std::runtime_error("prime gap does not fit the packed large-prime format");
    return pk;
}

} // namespace gb
