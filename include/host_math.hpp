#pragma once
// Pure host math of the dense pipeline (run_gpu): segment and class-map
// geometry, the per-prime first-bit and per-tile first-mark tables, the
// division-free reciprocals and the packed large-prime list. No Level Zero
// dependency, so tests/host_tests.cpp checks every function here against a
// brute-force reference. The kernels (kernels/goldbach.cl) trust these
// tables bit for bit: a wrong entry leaves a composite marked prime.

#include <algorithm>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "config.hpp"

namespace gb {

/// Splits [begin, end) into at most min(hardware threads, max_workers)
/// contiguous chunks and calls fn(first, last) for each on its own thread
/// (inline when one worker suffices). Meant for short uniform work items
/// (e.g. one per tile).
/// @pre   fn is safe to run concurrently on disjoint ranges and does not throw.
/// @post  every index was covered exactly once; all threads joined.
template <typename Fn>
void parallel_for(uint32_t begin, uint32_t end, uint32_t max_workers, Fn&& fn) {
    const uint32_t count = end - begin;
    if (count == 0) return;
    const uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
    const uint32_t workers = std::max(1u, std::min({hw, max_workers, count}));
    if (workers <= 1) {
        fn(begin, end);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (uint32_t t = 0; t < workers; ++t) {
        const uint32_t first = begin + uint32_t((uint64_t(count) * t) / workers);
        const uint32_t last = begin + uint32_t((uint64_t(count) * (t + 1)) / workers);
        threads.emplace_back(fn, first, last);
    }
    for (auto& thread : threads) thread.join();
}

// ---------------------------------------------------------------------------
// Pure geometry of one pipeline segment (no side effects).
//
// A segment verifies every even n in [seg_start, seg_end] (inclusive). The
// candidate side q = n - p for odd primes p <= MAX_GB_PRIME spans
// [seg_start - MAX_GB_PRIME, seg_end - 3]; only odd q are represented in the
// prime map (bit i <-> q = q_first + 2*i).
// ---------------------------------------------------------------------------
struct SegmentGeometry {
    uint64_t seg_start = 0;  // first even n of the segment
    uint64_t seg_end   = 0;  // last even n of the segment (inclusive)
    uint64_t seg_evens = 0;  // number of even n in [seg_start, seg_end]
    uint64_t q_first = 0;    // first odd q (odd by construction, see below)
    uint64_t q_last  = 0;    // last odd q (= seg_end - 3)
    uint64_t q_odds  = 0;    // odd values in [q_first, q_last] = map bits
    uint64_t tiles = 0;      // SLM tile work-groups (ceil(q_odds / tile bits))
    uint32_t slot  = 0;      // pipeline slot (alternates 0/1 per segment)
};

/// Computes the geometry of the segment starting at @p seg_start.
/// @param seg_start      first even n of the segment.
/// @param limit          last even n of the run (inclusive).
/// @param segment_even   integer span per segment (even).
/// @param segment_index  0-based ordinal; selects the pipeline slot (index & 1).
/// @pre   seg_start even, 4 <= seg_start <= limit.
/// @post  a remainder <= segment_even is merged into this segment
///        (seg_end = limit); q_first is odd (3 at the bottom of the range).
/// @return the segment geometry.
SegmentGeometry segment_geometry(uint64_t seg_start, uint64_t limit,
                                 uint64_t segment_even, uint64_t segment_index);

/// Per-prime first markable bit of the dense map, SEGMENT-GLOBAL (one
/// division per prime per segment, not per tile):
///   fb[j] = bit of max(p*p, first odd multiple of p >= q_first)
/// Each tile derives its own start from it, so all tiles run in one launch.
/// @param sieve_primes  ascending odd primes; the first @p sieve_split are used.
/// @param q_first,q_last  odd q range of the dense map (bit i <-> q_first + 2i).
/// @pre   q_first odd, q_first <= q_last.
/// @return sieve_split entries; ~0ull for primes with no mark in [q_first, q_last].
std::vector<uint64_t> compute_first_bits(const uint32_t* sieve_primes,
                                         uint32_t sieve_split,
                                         uint64_t q_first, uint64_t q_last);

// Class-map geometry of one segment (kernels/goldbach.cl map_clear). K = 8
// (production): arrays for the residues mod 30 coprime to 30, dense bit
// b = 15t + phi[c]. K = 2 (host tests only): arrays for q = 1 / 5 (mod 6),
// dense bit b = 3t + phi[c]. Class c holds q = Q[c] + qmod t.
inline constexpr uint32_t CLASS_RES6[2]  = {1, 5};
inline constexpr uint32_t CLASS_RES30[8] = {1, 7, 11, 13, 17, 19, 23, 29};
struct ClassGeo {
    uint32_t K = 0, qmod = 0;   // classes; q modulus (6 / 30)
    uint32_t phi[8] = {};       // dense bit phase (mod qmod / 2) of each class
    uint64_t Q[8] = {};         // first q of each class
    uint64_t n[8] = {};         // valid bits per class
    uint32_t tpc = 0;           // tiles per class (class c at word c * tpc * TILE_WORDS)
    uint32_t res(uint32_t c) const { return K == 8 ? CLASS_RES30[c] : CLASS_RES6[c]; }
};
/// Class-map geometry of a segment whose dense map starts at @p q_first.
/// @param q_odds  dense map bits (odd q in the segment).
/// @param K       2 (mod-6 map) or 8 (mod-30 map).
/// @pre   q_first odd.
/// @return phases, first q and bit count per class, and tiles per class (>= 1).
ClassGeo class_geometry(uint64_t q_first, uint64_t q_odds, uint32_t K = 2);

/// compute_first_bits for the class map: fb[c * split + j] = class-c index of
/// the first multiple of p in class c that is >= max(p*p, q_first) (~0 if none
/// <= q_last). Multiples of p in class c are q = m p with m = res_c * p^-1
/// (mod qmod); primes dividing qmod (3, and 5 for K = 8) have none.
/// Computed for the primes [j0, j1) only (every entry of those columns is
/// written), so disjoint ranges can run concurrently (parallel_for).
/// @param fb  output, K * sieve_split entries, class-major.
/// @param g   geometry from class_geometry for the same q_first.
/// @pre   j0 <= j1 <= sieve_split; sieve_primes odd primes ascending;
///        q_first odd <= q_last <= MAX_SUPPORTED_LIMIT (so first + p - 1 and
///        m p <= first + qmod p stay below 2^64).
/// @post  for j in [j0, j1), every class c: fb[c * sieve_split + j] = (q - Q[c]) / qmod
///        for the smallest q >= max(p^2, q_first), q = 0 mod p, q in class c,
///        q <= q_last; ~0 if none (or p | qmod). Other entries untouched.
///        For j >= cop_j0 a found entry also carries, in its high word,
///        c = (m mod COP_M) * 30^-1 mod COP_M for its q = m p (the medium
///        stage's med_prime_cop).
/// @throws std::runtime_error if cop_j0 < j1 and K != 8, or a residue entry's
///         class bit is >= 2^32 (it would not fit the low word).
void first_bits_class_range(uint64_t* fb, const uint32_t* sieve_primes,
                            uint32_t sieve_split, uint32_t j0, uint32_t j1,
                            uint64_t q_first, uint64_t q_last, const ClassGeo& g,
                            uint32_t cop_j0 = UINT32_MAX);

/// COP_M in kernels/goldbach.cl: 7 * 11 * 13 * 17 * 19 * 23 * 29 (the mask
/// primes whose multiples med_prime_cop skips), and 30^-1 mod COP_M.
inline constexpr uint64_t COP_M = 215656441;
inline constexpr uint64_t COP_INV30 = 208467893;

/// Per-tile first-mark table for the table path (u32, tile-major: row t holds
/// the tile-local first bit of every prime in tile t, UINT32_MAX = none).
/// Mirrors the kernel's reciprocal path exactly: tile-local bit of
/// max(fb, ceil jump from tile start); the jump to a thread's own range
/// happens on the GPU with 32-bit math. Parallelized over tiles (uniform
/// short rows).
/// @param tbl          output, tiles * own_end entries.
/// @param own_end      row width: the thread-owned primes [0, own_end).
/// @param fb_all       first bits from compute_first_bits (class map:
///                     first_bits_class_range, K * sieve_split entries).
/// @param magic,shift  reciprocals from build_magic_reciprocals.
/// @param tpc          class map: tiles per class (tile t is in class t / tpc);
///                     0 for the dense map.
/// @pre   own_end <= sieve_split; tiles cover the map (dense) or K * tpc (class);
///        tbl has tiles * own_end entries; fb_all from the same segment.
/// @post  tbl[t * own_end + j] = tile-local bit of prime j's first mark in
///        tile t (< BITS_PER_TILE), or UINT32_MAX when the tile has none.
void build_tile_first_table(uint32_t* tbl, uint32_t tiles, uint32_t sieve_split,
                            uint32_t own_end,
                            const uint64_t* fb_all, const uint64_t* magic,
                            const uint32_t* shift, const uint32_t* sieve_primes,
                            uint32_t tpc = 0);

/// Magic reciprocal (floor(2^64 / p) + 1) and the smallest shift s such that
/// floor(x*magic >> (64+s)) == x/p on a set of probe values up to 2^33 (our
/// d = my_bit0 - first_bit < 2^33 always). The kernel computes ceil(d/p) =
/// floor(d/p) + (d%p != 0) with the correction step, so floor is all it needs.
/// @pre   primes >= 3 (p = 1 would overflow the magic).
/// @return {magic, shift}, sieve_split entries each; shift 0 when no s <= 32
///         passes (the kernel's correction step then handles it).
std::pair<std::vector<uint64_t>, std::vector<uint32_t>>
build_magic_reciprocals(const uint32_t* sieve_primes, uint32_t sieve_split);

/// 32-bit reciprocals for the tile table path. The kernel jumps from the
/// tile's first mark to a thread's own bit range: d = my_bit0 - tf is
/// always < BITS_PER_TILE = 2^17, so floor(d/p) is computable with a 32-bit
/// reciprocal: with s = bit_width(BITS_PER_TILE * p) we have 2^s > BITS_PER_TILE*p,
/// which makes floor(d*m/2^s) = floor(d/p) exact for every d < BITS_PER_TILE
/// (for d divisible by p the product is k*p*(2^s+e)/2^s = k + k*e/2^s with
/// k*e < BITS_PER_TILE < 2^s; otherwise d*e < (p - r)*2^s follows from
/// d*e < BITS_PER_TILE*p < 2^s). m = ceil(2^s/p) <= 2*BITS_PER_TILE fits u32.
/// Primes >= BITS_PER_TILE never use it (single mark per tile, range check).
/// @pre   primes >= 3.
/// @return sieve_split entries: m | (s << 32) for p < BITS_PER_TILE, 0 for
///         the larger primes. The kernel computes floor(d/p) as
///         (u32)((u64(d) * m) >> s), then ceil with k += (k * p < d).
std::vector<uint64_t> build_tile_reciprocals32(const uint32_t* sieve_primes, uint32_t sieve_split);

struct PackedLarge { std::vector<uint32_t> bases; std::vector<uint8_t> gaps; };
/// Packs the large primes [split, count) for sieve_large_packed: one byte per
/// prime (half-gap to the previous prime; 0 at every LARGE_PACK_GROUP-prime
/// group start) plus the first prime of each group.
/// @pre   primes ascending odd, split <= count.
/// @return bases and gaps (both non-empty even when no prime is packed).
/// @throws std::runtime_error (fail closed) if a half-gap does not fit a byte
///         — never happens below 2^32 (max gap 292 at 2.2e9), but the
///         kernel's correctness depends on it.
PackedLarge pack_large_primes(const uint32_t* primes, uint32_t split, uint32_t count);

} // namespace gb
