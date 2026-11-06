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
// PROJECT-REFERENCE.md §9.2k); the dense path is unchanged since v1.13.0.
// v2.0.1: correctness fixes only (--wheel low-range witnesses, §15; CLI and
// ScmEngine fail-closed paths); dense and --sparse kernels byte-identical.
// v2.1.0: cleanup release — concluded experiment kernels/tools and the
// A/B-only env knobs removed; production paths unchanged (spv byte-identical).
// v2.2.0: exact Goldbach statistics on the dense path (include/gstats.hpp,
// PROJECT-REFERENCE.md §9.2l) and a tail-bounded mod-30 checker plan
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
inline constexpr const char* VERSION = "2.5.1";

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
inline constexpr uint64_t ODDS_PER_TILE     = BITS_PER_TILE;             // odd values per tile
inline constexpr uint64_t INTEGERS_PER_TILE = BITS_PER_TILE * 2;         // ~262K integer range
inline constexpr uint32_t WORDS_PER_THREAD  = TILE_WORDS / WG_SIZE;      // 8 words per thread
// The large-prime kernel amortizes dispatch across a small, bounded batch.
// Keep this mirrored in kernels/goldbach.cl. Four preserves short work-items
// while reducing ~98M work-items at 4e18 to ~25M.
inline constexpr uint32_t LARGE_PRIMES_PER_ITEM = 4;
// check_numbers processes this many 64-even words per work-item. This must
// match WORDS_PER_ITEM in kernels/goldbach.cl and is used for launch coverage.
inline constexpr uint32_t CHECK_WORDS_PER_ITEM = 4;
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
// mask stage. MUST match WHEEL_W in kernels/goldbach.cl (GB_WHEEL_W overrides
// the host).
inline constexpr uint64_t WHEEL_W = 30030;
// bucket_process grid-stride work-groups (GB_BUCKET_GROUPS overrides).
inline constexpr uint32_t BUCKET_GROUPS = 1024;
// Primes p > BUCKET_MIN_P_FACTOR * q_odds_full go to the bucket sieve
// (GB_BUCKET_MINP=<p> overrides; floor q_odds_full). A bucket entry costs
// ~200 ps per hit (read + mark + re-file), a packed prime ~40 ps of setup per
// segment. The multiplier wheel (v1.9.0) cut the hits per prime to 480/1155
// but not the packed setup, which moved the crossover from ~8 to ~3 x q_odds
// (v1.10.0, 2e10 windows vs 1e9: 7e8 +9.4%, 5e8 +9.3%, 3.5e8 +9.6% steady;
// +8.2 / +10.7 / +11.6% end; 2.5e8 flat then -3.4% after the packed setup got
// cheaper). v1.10.1 (128-B-aligned buckets, wheel 30030), 1e11 windows at
// 4.0/4.3/4.7/5.0e18 vs 2310 at 4 x: 30030 at 4 x +0.8/+0.9/+0.8/+0.8%,
// 3 x +1.6/+1.6/+1.4/+1.2%, 2.5 x +1.3/+1.4/+1.1/+1.1%. Cost: the ring is
// R x cap entries, ~6.9 GiB of VRAM at 3 x with the 30030 wheel (R ~199).
inline constexpr uint64_t BUCKET_MIN_P_FACTOR = 3;
// Two-phase checker: candidates handled by the word-level phase A before the
// remaining evens go to the per-even phase B (GB_CHECK_CAP overrides), and
// phase B's grid-stride work-group count.
inline constexpr uint32_t CHECK_PHASE_A_CANDS  = 160;
// Phase-A item width in 64-bit words (768 evens per item) of every word
// checker. MUST match A_W64 in kernels/goldbach.cl. v1.13.0: 16 -> 14 (the
// mod-30 checker: 12 -1.0%, 14 -1.3%; 8: +0.3%). 2026-10-05: 14 -> 12 after
// the rotate funnel (12 spill-free; +0.18% / +0.20% long / long-end).
inline constexpr uint32_t CHECK_PHASE_A_W64    = 12;
inline constexpr uint32_t CHECK_PHASE_B_GROUPS = 640;
// Twin checker (v1.10.0): phase A funnels over the lower members of twin
// pairs (two candidates per funnel, kernels/goldbach.cl tw32), then phase B
// probes the candidates those pairs did not cover (GB_TWIN_CAP; GB_TWIN=0 =
// the v1.9.0 checker). 2e10 window: 80 +0.4%, 90 +1.4%, 100 +1.1%, 120 +0.3%
// over the plain checker with the rest list; plain cap 160 re-checked.
inline constexpr uint32_t CHECK_TWIN_CANDS = 100;

// Class-split checker (v1.11.0, include/split_check.hpp): funnels per class
// list and the stage masks (bit d = candidate p + 6d; "7/3" = sexy-prime
// triplets p, p+6, p+12, then pairs p, p+6 of the remaining primes).
// Class map, 350M spans, check_s per 1e11 at 4e18: pairs "3" cap 60: 123.1 ms;
// "7/3" cap 33/35/37/38/39: 125.9/120.0/116.8/116.6/116.3, cap 40/41/42:
// 123.9/129.3/137.3 (the triplet list's 3-5-word gaps start forcing full
// window reloads); -0.7% total at 4e18, 4.5e18 and 5e18. Stage limits
// ("7:20/3") measured no better.
inline constexpr uint32_t CHECK_SPLIT_CANDS = 38;
inline constexpr const char* CHECK_SPLIT_STAGES = "7/3";

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
// with SLM atomics instead of by every thread (make_prime_map stage 3m).
// Measured at 4e18 with 64 KB tiles (split 40000, sieve_s over the 2e9 window): 128 -> 0.0152,
// 256 -> 0.0138, 512 -> 0.0137, 1024 -> 0.0143, 2048 -> 0.0156, 4096 -> 0.0184.
// v1.10.0 (batched medium-stage loads): 256 +0.6% over 512 (8 paired rounds).
inline constexpr uint32_t MED_MIN_P = 256;
// Class mode (v1.11.0): per 1e11 window sieve_s 261.7 (256) / 256.2 (384) /
// 253.0 (512) / 261.8 (640) ms — fewer, longer class tiles favour the
// thread-owned walk for a few more primes.
inline constexpr uint32_t MED_MIN_P_CLASS = 512;

// Small-prime mask sieve bound (must match MASK_MAX_P in kernels/goldbach.cl):
// primes below this use whole-word AND masks in make_prime_map stage 3a.
inline constexpr uint32_t MASK_MAX_P = 64;

// ---- Sieve primes --------------------------------------------------------
// Base primes for the segmented sieve must reach sqrt(max q).
// 1e6 => supports q up to ~1e12. Raise this (and rebuild) for larger runs.
inline constexpr uint32_t SIEVE_PRIME_LIMIT = 1'000'000;

// ---- Defaults ------------------------------------------------------------
inline constexpr uint64_t DEFAULT_LIMIT          = 100'000'000ULL;  // verify evens in [4, limit]
// Segment span rationale: with the host prep overlapped and the prime map
// L3-resident, 200-250M spans are the measured optimum — at this span the
// 12-16 MB prime map fits the 18 MB L3, making the large-prime kernel's
// scattered atomic marks far cheaper than when they hit GDDR (400M spans
// fall off sharply). The matching split optimum is ~2400; see the split
// selection comment in src/verifier.cpp.
// v1.11.0 (mod-6 class map, 2/3 the bits): 350M spans — steady-state kernel
// time per 1e11 at 4e18 / 5e18: -1.8% / -2.5% vs 250M (300M -0.8%, 325M
// -1.4%, 375M 0%, 400M +0.1%, 500M: map leaves L3, large_s x5).
// v1.12.0 (mod-30 class map, 8/15 of the bits): 425M — kernel time per 1e11
// vs 350M: -1.5% / -1.3% (400M +0.1%, 450M -1.1%, 475M +0.8%, 500M +0.2%).
// v2.4 (fused hit lists, one-walk harvest): 467.5M — gpu per 1e11 vs 425M
// +1.16% / +1.29% faster (450M +0.6 / +1.0%, 480M +0.85 / +1.1%, 491.5M
// +1.1 / +1.3%; 510M: map leaves L3, -34%). Tile-exact spans do not matter
// (424.66M, 108 full tiles per class: +-0%).
inline constexpr uint64_t DEFAULT_SEGMENT_EVEN   = 467'500'000ULL;
inline constexpr uint32_t MAX_LEFTOVERS          = 1u << 20;        // compact exception list cap

} // namespace gb
