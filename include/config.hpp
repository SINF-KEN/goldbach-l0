#pragma once
// Global tuning constants for the B580 Goldbach verifier.
//
// B580 (Xe2 / Battlemage) assumptions:
//   20 Xe-cores, 160 vector engines
//   128 KB SLM per Xe-core, 128 KB max SLM per work-group
//   256 KB L1 per Xe-core, 18 MB shared L3
//   supported subgroup sizes: 16 and 32
//   max work-group size: 1024

#include <cstdint>

namespace gb {

// Algorithm version. v2.0.0 adds the sparse class map pipeline (--sparse,
// HISTORY.md §9.2k); the dense path is unchanged since v1.13.0.
// v2.0.1: correctness fixes only (--wheel low-range witnesses, §15; CLI and
// ScmEngine fail-closed paths); dense and --sparse kernels byte-identical.
// v2.1.0: cleanup release — concluded experiment kernels/tools and their
// A/B-only env knobs removed; production paths unchanged (spv byte-identical).
// PROJECT-REFERENCE.md §7 lists every GB_* variable.
// v2.2.0: exact Goldbach statistics on the dense path (include/gstats.hpp,
// HISTORY.md §9.2l) and a tail-bounded mod-30 checker plan
// (every phase-A member < 3000: complete ridges and p(n) >= 3000 tail).
// v2.2.1: fail closed above 2^64 - 2^40 (the base-prime bound wrapped to ~16
// above ~1.8446744e19 and sieved with primes < 64 only: false PASS); results
// below that unchanged, kernels byte-identical.
// v2.3.0: hit lists replace the bucket ring for the largest primes (+8.5% /
// +9.2%, §9.2t) plus packed-kernel setup trims (§9.2s); hardening: campaign
// statistics state merged under a lock, durably, refusing damaged state;
// atomic, content-checked prime cache; leftover indices range-checked;
// empty / reversed ranges are usage errors; the --wheel mode is removed.
// Results unchanged (statistics byte-identical, maps verified word for word).
// v2.4.0: prime gap statistics (the live paper's section 2.2), exact: every
// gap's count, sums of g^k (k <= 4), N(g), the largest gap, every gap >= 700
// with its first occurrence (gaps30's full walk on every segment, cross-checked
// against count30's bit-parallel tail and first / last primes; ~11% GPU time).
// GB_GAPS=sampled walks only the sampled segments, GB_GAPS=0 disables.
// New GBSTATS fields, a .gaps state file; every earlier field unchanged.
// v2.5.0: the gap walk fused into count30 (one read of the class map, the
// range masks and look-back shared; per prime only g / 2 into the histogram
// and the maximum: moments, count and sum come from the histogram, its tail
// is cross-checked against the bit-parallel tail). Results unchanged.
// v2.5.1: the packed kernel's gap-stream loads bypass L3 (LSC L1S_L3UC): the
// ~33 MB per segment stream no longer evicts the class map under the
// atomics (+0.4%). Results unchanged.
// v2.6.0 (audit 2026-11-17, CORRECTNESS.md §6): positive coverage accounting
// (every even of a segment counted as resolved or left for the CPU; a checker
// that did not run fails the run, dense and --sparse); the prime cache carries
// a payload hash ("GBP2"); the twin / class-split checkers, the dense / mod-6
// maps and the bucket ring are removed (only A/B knobs reached them; those
// knobs are refused). Results unchanged.
inline constexpr const char* VERSION = "2.6.0";

// ---- make_prime_map geometry --------------------------------------------
// One work-group = one tile. SLM holds 64-bit words; one bit represents one
// ODD number.
inline constexpr uint32_t WG_SIZE           = 256;                       // threads per work-group
// Tile = 2048 words = 16 KB SLM (v1.8.1; see TILE_WORDS in goldbach.cl for
// the measurements). GB_TILE_WORDS overrides at build time for experiments
// and MUST match the kernel's TILE_WORDS (-DTILE_WORDS=... to ocloc).
#ifndef GB_TILE_WORDS
#define GB_TILE_WORDS 2048
#endif
inline constexpr uint32_t TILE_WORDS        = GB_TILE_WORDS;             // words of 64 odd numbers
inline constexpr uint64_t BITS_PER_TILE     = uint64_t(TILE_WORDS) * 64; // 131,072 odd numbers (16 KB)
// sieve_large_primes (the plain reference kernel of the validators): primes
// per work-item. Keep this mirrored in kernels/goldbach.cl.
inline constexpr uint32_t LARGE_PRIMES_PER_ITEM = 4;
// sieve_large_packed: primes per packing group (one base u32 each) = 4 x the
// kernel's subgroup width. MUST match LARGE_PACK_GROUP in kernels/goldbach.cl.
#ifndef GB_LARGE_PACK_PPI
#define GB_LARGE_PACK_PPI 16
#endif
inline constexpr uint32_t LARGE_PACK_PPI   = GB_LARGE_PACK_PPI;   // primes per work-item
inline constexpr uint32_t LARGE_PACK_GROUP = 32 * LARGE_PACK_PPI; // SIMD32 subgroup
// Work-group size of the large-prime kernel (a multiple of its SIMD32
// subgroup). v1.10.0 (32-bit setup, buckets from 3 x q_odds): 128 beats 64 by
// +0.8% (steady) / +1.3% (end); 32 -1.8%, 256 ties.
inline constexpr uint32_t LARGE_WG_SIZE = 128;
// Multiplier wheel of the large-prime / bucket marking (v1.9.0; 30030 since
// v1.10.1): a prime p > 61 only marks p*m for m coprime to WHEEL_W; every
// other multiple is divisible by a prime <= 13 and cleared by the small-prime
// mask stage. MUST match WHEEL_W in kernels/goldbach.cl (30030 only).
inline constexpr uint64_t WHEEL_W = 30030;
// A separate hl_apply's grid-stride work-groups are 2 x this (only when no
// prime is packed; GB_BUCKET_GROUPS overrides).
inline constexpr uint32_t BUCKET_GROUPS = 1024;
// Primes p > BUCKET_MIN_P_FACTOR * q_odds_full go to the hit lists
// (GB_BUCKET_MINP=<p> overrides; floor q_odds_full: at most one hit per
// segment). The crossover against the packed kernel's per-segment setup was
// measured for the bucket ring that preceded the hit lists (HISTORY.md
// §9.2t): 3 x q_odds_full.
inline constexpr uint64_t BUCKET_MIN_P_FACTOR = 3;
// Phase-A item width in 64-bit words (768 evens per item) of every word
// checker. MUST match A_W64 in kernels/goldbach.cl. v1.13.0: 16 -> 14 (the
// mod-30 checker: 12 -1.0%, 14 -1.3%; 8: +0.3%). 2026-10-05: 14 -> 12 after
// the rotate funnel (12 spill-free; +0.18% / +0.20% long / long-end).
inline constexpr uint32_t CHECK_PHASE_A_W64    = 12;
inline constexpr uint32_t CHECK_PHASE_B_GROUPS = 640;
// ---- Goldbach candidate primes ------------------------------------------
// Largest "smallest prime" needed by any minimal Goldbach partition in the
// published 4e18 verification (Oliveira e Silva et al.) is 9781.
// So the candidate list p <= 9781 (~1,200 primes, uint16_t) suffices.
inline constexpr uint32_t MAX_GB_PRIME = 9781;

// ---- Goldbach ridge tracking (dense path; include/ridge.hpp) --------------
// A ridge is the smallest prime p of an even's Goldbach partition (no smaller
// prime works). The phase-B checkers append (n, p) of every pending even they
// resolve to a device-shared buffer; the host validates each entry with exact
// integer arithmetic and keeps the RIDGE_KEEP largest. RIDGE_CAP is the
// per-segment append capacity and MUST match RIDGE_CAP in kernels/goldbach.cl
// (tests/check_constants.py). The floor filters kernel appends (every pending
// even with a ridge >= floor); once the top list fills, the recording
// threshold ratchets to the current 30th ridge and the volume collapses to a
// handful per segment. The default floor is the mod-30 plan's phase-A bound
// (CHECK30_TAIL = 3000): every even with a ridge >= 3000 reaches phase B, so
// the list is complete above the floor; the verifier raises a lower
// GB_RIDGE_MIN to the active plan's bound (RidgeTracker::clamp_floor_above).
// 3000 x 64K also keeps the first segments well under the buffer in the
// two-segment enqueue pipeline blind spot. GB_RIDGE=0 / GB_RIDGE_MIN /
// GB_RIDGE_FILE override.
inline constexpr uint32_t RIDGE_CAP          = 65536;     // 64K pairs = 1 MB
inline constexpr uint32_t RIDGE_KEEP         = 30;
inline constexpr uint32_t RIDGE_MIN_DEFAULT  = 3000;

// Tile-kernel primes >= MED_MIN_P (and < the split) are walked prime-parallel
// with SLM atomics instead of by every thread (make_prime_map stage 3m;
// GB_MED_P overrides). Class map, per 1e11 window: sieve_s 261.7 (256) /
// 256.2 (384) / 253.0 (512) / 261.8 (640) ms.
inline constexpr uint32_t MED_MIN_P = 512;

// Small-prime mask sieve bound (must match MASK_MAX_P in kernels/goldbach.cl):
// primes below this use whole-word AND masks in make_prime_map stage 3a.
inline constexpr uint32_t MASK_MAX_P = 64;

// ---- Defaults ------------------------------------------------------------
inline constexpr uint64_t DEFAULT_LIMIT          = 100'000'000ULL;  // verify evens in [4, limit]
// Segment span: the largest span whose mod-30 class map (8/15 of the odd
// numbers, ~15.6 MB at 467.5M) stays in the B580's 18 MB L3, where the
// large-prime kernels' scattered atomic marks are cheap; past it the map
// falls out of L3 and the large-prime time multiplies (510M: -34%). v2.4
// (fused hit lists, one-walk harvest): 467.5M — gpu per 1e11 vs 425M +1.16% /
// +1.29% (450M +0.6 / +1.0%, 480M +0.85 / +1.1%, 491.5M +1.1 / +1.3%).
// Tile-exact spans do not matter (424.66M, 108 full tiles per class: +-0%).
inline constexpr uint64_t DEFAULT_SEGMENT_EVEN   = 467'500'000ULL;
inline constexpr uint32_t MAX_LEFTOVERS          = 1u << 20;        // compact exception list cap

} // namespace gb
