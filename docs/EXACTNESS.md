# Exactness rules of the fast paths

The verifier's output (PASS / FAIL, prime and twin counts, p(n) statistics,
gaps, ridges) must be bit-identical to a plain sieve. The speed rounds of
2026-10 .. 2026-11 (HISTORY.md §9.2r .. §9.2ab) added shortcuts that are exact
only under the conditions below. Each entry gives the condition, where it is
enforced, and what goes wrong if it is broken.

A broken rule does not always fail the run. A lost composite mark leaves a
composite "prime" in the map. The checker then accepts fake Goldbach
partitions, so the run still PASSes, but the prime counts and statistics are
wrong. Only `GB_BUCKET_VERIFY` (hit lists vs a full packed rebuild) and the
prime counts against `stats_ref` / a base build catch that class of bug. Any
change to these paths needs a `GB_BUCKET_VERIFY=1` run at 4e18 and 5e18 and
the low-height windows (1e12, 1e14, 1e18) compared against the previous build
(see "How to check a change" at the end).

Line numbers are as of the cleanup after the speed rounds (2026-11-19).

## 1. Prime-multiplier rule (hit lists with HL_MT)

**Rule.** For a hit-listed prime p with p³ > limit, a mark p·m only matters
when m is prime. A composite m has a prime factor r ≤ √m < p, so p·m is a
multiple of r < p and r already cleared it.

**Conditions** (`DenseRun::mt_geometry`, src/verifier.cpp:484):
- p_min · p_min > limit / p_min + 1, i.e. p_min³ > limit;
- p_min² > m_hi, so every multiplier m ≤ m_hi below p² is either prime or has
  a factor < p;
- the table's sieving primes (≤ √m_hi) lie far below m_lo (2·s_max < m_lo),
  so mt_build never clears a table bit for m = s itself;
- m_hi < 2³⁸ (hl_harvest_mt keeps the high quotient word below 64);
- the table is below 3.5 GiB (32-bit word offsets).

**Enforced.** `mt_geometry` picks the threshold (binary search,
src/verifier.cpp:1094) and is re-checked for the actual first hit-listed prime
(src/verifier.cpp:1151, throws `hl_mt: multiplier table geometry`). The
sieving-prime bound is checked at src/verifier.cpp:1160 (throws `hl_mt:
multiplier table sieving primes`). In the kernel, `hl_harvest_mt`
(kernels/goldbach.cl:4853) files a hit only when the table bit of m is set; a
multiplier below the table's first block sets error flag 32 (fails closed at
collection, src/verifier.cpp:2405). The table itself (`mt_build`,
kernels/goldbach.cl:4784) presieves 17..61 by patterns and marks the sieving
primes; `GB_MT_CHECK=N` compares N table bits with Miller-Rabin at init.

**If violated.** A composite m with no factor below p (p³ ≤ limit) would have
its mark dropped: p·m stays "prime". `GB_HL_MT=0` runs the old harvest
(`hl_harvest`, every multiplier, primes > q_odds_full only), which does not
rely on this rule.

## 2. Skip set: m divisible by 17..43

**Rule.** The packed large kernel and the hit-list harvest skip a mark p·m
(m coprime to 30030) when m is divisible by 17, 19, 23 or 29 (p < SKR_MAXP =
60M), or by 31, 37, 41 or 43 (p < SKR2_MAXP = 10M). Such a p·m is a multiple of
a prime r ≤ 43, and the mask stage of `make_prime_map` clears every multiple of
3..61 in every segment.

**Conditions.**
- The mask stage covers every prime ≤ 61: MASK_MAX_P = 64 in
  kernels/goldbach.cl:62 and include/config.hpp, which has
  `static_assert(MASK_MAX_P > 61)`. tests/check_constants.py checks the
  kernel's value too.
- SKR2_MAXP ≤ SKR_MAXP (the second residue set is only applied with the first;
  check_constants).
- SKR_R = 17·19·23·29 (check_constants). The residue fields are 6 bits wide
  (r ≤ 29 < 64) and 7 bits wide (r ≤ 43 < 128).
- p·m ≠ r (p is a large prime), so the skip never drops the prime r itself.
  The mask stage's own "restore q = p" step only applies to q ≤ MASK_MAX_P.

**Where.** kernels/goldbach.cl:540 (SKR_MAXP), :1156 (SKR2_MAXP), the use
in `sieve_large_packed` at :3108, the harvest skip at :4344.

**If violated.** A mark of a composite that the mask stage did not clear is
dropped, so the map holds a composite as prime.

## 2b. SLM skip set: m divisible by 7..29 (tile items, mid windows)

**Rule.** The tile sieve's medium items (`med_prime_cop`) and the mid windows
for p < MIDW_COP_MAXP (`mid_window` via `cop_marks`) skip a mark p·m when m
is divisible by 7, 11, 13, 17, 19, 23 or 29: the mask stage cleared every
multiple of these primes, and p·m ≠ r (p ≥ 83). A prime's marks along a class
array have multipliers m, m + 30, m + 60, ...; with c = m · 30⁻¹ mod COP_M
(COP_M = 7·11·…·29) mark t has m ≡ 30 (c + t), so r | m iff r | c + t.

**Conditions.**
- The mask stage covers 7..29 (MASK_MAX_P > 61, as rule 2; the verifier
  throws if the medium items run without it).
- c is right: the host writes it into the high word of the items' first bits
  (`first_bits_class_range` with cop_j0 = own_end; the class bit itself stays
  < 2³²), mid_setup into the second half of the i0 table. host_tests checks
  30 c ≡ m (mod COP_M); check_constants checks COP_M and the inverses.
- The per-word walk relies on no 32 consecutive c + t all being redundant
  (the longest run for 7..29 is 8).

**If violated.** A needed mark is skipped (wrong c) and the map holds a
composite as prime: prime counts change, the run may still PASS.

## 3. Read-before-mark in the packed kernel

**Rule.** `map_clear_chk` (kernels/goldbach.cl:336) reads the map word
(L1-uncached, L2-cached load) and issues the atomic AND only when the bit is
still 1. This is exact because while the packed kernel runs, every concurrent
writer only AND-clears bits. A stale 1 just costs a redundant atomic. A 0 is a
real clear: nothing sets map bits back to 1 during the kernel.

**Conditions.**
- The tile sieve's whole-word stores (which do set bits) are finished and
  visible before the packed kernel starts. The barrier in `enqueue_midwin`
  (src/verifier.cpp:1918) follows the tile sieve. Without mid windows, or with
  HL_MT, `enqueue_packed_large` appends its own barrier
  (src/verifier.cpp:2023-2024).
- mid_window, which may run beside the packed kernel when HL_MT is off
  (`ml_ovl`, src/verifier.cpp:2023), writes its windows back with 64-bit atomic
  ANDs only (`atom_and`, kernels/goldbach.cl:5229). It never stores whole
  words.
- The fused hit-list apply in `sieve_large_packed_hl` (kernels/goldbach.cl:5078)
  only AND-clears.

**If violated.** A writer that stores whole words (or sets bits) during the
packed kernel can make a read see a 0 that is later overwritten by a 1. The
skipped atomic leaves a composite marked prime. The 2026-10-19 note in the
kernel shows the same filter was a loss before the mid windows existed; it
was never inexact.

## 4. mod_ff_q: p ≥ 512, quotient path p ≥ 2²¹

**Rule.** The packed kernel takes q_first mod p from a float-float quotient
(`mod_ff_q`, kernels/goldbach.cl:155). A GPU self-test against the exact
remainder (HISTORY.md §9.2z) found wrong results for every prime 67..251 at
q ≈ 4e18 and up to 491 at q ≈ 9.2e18, and none for 512..2³¹ over 1.2e9
(x, p) pairs.

**Conditions.**
- `mod_ff_q` only for q_first ≥ 2⁶⁰ (`hi_range`, kernels/goldbach.cl:3075)
  and p < 2³⁰.
- Every prime the packed kernel (and the `GB_BUCKET_VERIFY` reference) gets is
  ≥ MOD_FF_MIN_P = 4096: the split is clamped to the first prime ≥ 4096
  (src/verifier.cpp:685) and re-checked (src/verifier.cpp:687, throws). The
  margin above 512 is deliberate. include/config.hpp has
  `static_assert(MOD_FF_MIN_P >= 512)`; check_constants checks it too.
  mid_setup uses the exact 64-bit `%`.
- The first multiplier from the quotient (m0 for the 17..43 skip residues)
  only for p in [2²¹, MQ_MAXP) (kernels/goldbach.cl:3099). p ≥ 2²¹ keeps
  the quotient's high word below 2⁴²/2³² = 1024 (q_first < 2⁶³), so
  qhi · 1919493 < 2³² in the residue formula. Below 2²¹ the kernel computes
  the residues the old way.

**If violated.** A wrong remainder puts every mark of that prime at the wrong
place. Composites stay prime and real primes are cleared. The second effect
can turn a PASS into a FAIL or change counts. The p < 512 bug of 2026-11-17
only showed in prime counts, not in `GB_BUCKET_VERIFY`, because the reference
path used the same remainder.

## 5. Gap walk by blocks of 30: host Miller-Rabin fallback

**Rule.** count30's gap walk steps once per non-empty block of 30 and adds the
gaps inside a block from a 256-entry pattern histogram. Its maximum only covers
gaps into blocks. When a segment's largest such gap is not larger than the
largest internal one (all gaps ≤ 28, which only happens on tiny ranges), the
host recomputes the maximum with Miller-Rabin over the counted range
(`walk_max_exact`, include/gstats.hpp:111, called at src/verifier.cpp:1030).

**Condition.** The CPU walk is bounded at 10⁷ numbers. Above that it throws
(`gaps: block-walk maximum needs a CPU walk over ...`), so the run fails closed
instead of reporting a wrong maximum gap. A range of 10⁷ always holds gaps
≥ 30 in practice, so the fallback never runs in production.

**If violated** (bound removed): a very slow CPU walk; (fallback removed): a
wrong maximum gap / position in the statistics on tiny ranges.

## 6. Hit-list memory budget

**Rule.** A segment's hit list is hl_G regions of hl_capg u32 entries, one
region per harvest work-group. The capacity is mean·1.10 + 10·√mean + 64,
rounded up to a power of two (src/verifier.cpp:1200-1210). The mean comes from
the exact hit estimate (src/verifier.cpp:1174). The batch length K is
lowered until K·hl_G·hl_capg·4 bytes is below 4 GiB (32-bit offsets in the
kernels) and below `GB_HL_LIST_MAX` (default HL_LIST_MAX_DEFAULT = 3.5e9 B,
src/verifier.cpp:111).

**Enforced.**
- A region overflow sets BKT_ERR_OVERFLOW (kernels/goldbach.cl:4384 /
  :5021). The host fails the run closed (src/verifier.cpp:2405). A wrong
  estimate can therefore only stop a run, never drop a mark.
- Since the cleanup, a batch that is still ≥ 4 GiB at K = 1 throws at init
  (src/verifier.cpp:1222). It used to continue silently; at 4e18 one batch is
  about 30 MB, so this never fired.
- The harvest's 32-bit walk needs q_odds_full < 2·delta < 2³² and
  delta + p_max < 2³² (src/verifier.cpp:1229, throws).

**Device memory.** At 4e18..5e18 the run takes ~6.5-7.2 GB of the B580's 12 GB:
lists ~3.5 GB, multiplier table 2.7-3.4 GB, maps and the rest. Do not add more
than a few hundred MB without measuring at 5e18.

## 7. Coverage accounting (v2.6)

**Rule.** A PASS means that every even of every segment was resolved on the
GPU or handed to the CPU fallback, not just that no failure was reported.
`check30_a` adds the evens it resolved to acct[0] (kernels/goldbach.cl:3491).
`check30_b` adds the pending evens it resolved to acct[1] (:3621). On sampled
segments, the exact histogram counts every even.

**Enforced** in `DenseRun::check_coverage` (src/verifier.cpp:2449):
- acct[0] + pending == segment evens (src/verifier.cpp:2460);
- acct[1] + leftovers == pending (src/verifier.cpp:2463);
- sampled: histogram count + leftovers == segment evens (src/verifier.cpp:2453).

A mismatch throws before anything is merged. The pending list overflow
(src/verifier.cpp:2413) and the leftover cap fail closed too.

**If removed.** A checker launch that did not run, or covered too few evens
(wrong grid, wrong items_pad, a zero group count), would leave nothing pending
and PASS without checking anything. `GB_CHECK_BGROUPS` / `GB_STATS_BGROUPS` /
`GB_BUCKET_GROUPS` refuse 0 for the same reason.

The atomic costs ~0.7 % (PR #33). It is kept on purpose.

## 8. Ordering rules the fast paths depend on

- **Tile stores vs. atomics.** The tile sieve stores whole words. Every kernel
  that AND-clears the same map (mid_window, the packed kernel, hl_apply, the
  fused apply) runs after a barrier that follows it (HISTORY.md §9.2v,
  PROJECT-REFERENCE.md §11).
- **count30 placement.** count30 only reads the finished map. It is launched
  after phase A's barrier and before phase B with no further barrier, and the
  host waits on its event before collecting the slot (src/verifier.cpp,
  `enqueue_check30` / `collect_previous_segment`).

## How to check a change

1. `./build.sh` with the pinned ocloc; commit the fresh `.spv` (CI's kernels
   gate). `ctest -LE gpu` (check_constants covers the constants above).
2. Exactness against the previous build with val.sh on 4e18 +1e11 and 5e18
   −1e11 with `GB_BUCKET_VERIFY=1`, on 1e12, 1e14 and 1e18 (+1e10), and on a
   5e8 window at 4e18. stats.state, stats.state.gaps, ridges and stats.log
   (except times) must be byte-identical.
3. `GB_HL_MT=0` at 4e18 must stay exact (the fallback).
4. `tests/gpu_validate.sh build quick`.
