// ============================================================================
// goldbach.cl — B580 (Xe2) Level Zero kernels
//
// make_prime_map : segmented odd-only sieve, one work-group per SLM tile
//                  (16 KB since v1.8.1; covers base primes p < sieve_split only)
// sieve_large_primes : one work-item per LARGE_PRIMES_PER_ITEM (4) base
//                  primes p >= sieve_split, marks its multiples straight
//                  into the global map (atomic AND).
//                  This is what makes N=4e18 feasible: the tile walk's cost
//                  no longer grows with pi(sqrt(N)) (~98M primes at 4e18 —
//                  walking all of them per tile blew the driver watchdog).
// check_numbers  : word-level Goldbach verification, WORDS_PER_ITEM*64 =
//                  256 evens per work-item (4 disjoint 64-even words)
//
// Geometry (see include/config.hpp; defaults):
//   work-group        = 256 threads
//   SLM per work-group= 16 KB = 2048 uint64 words = 131,072 odd numbers
//   words per thread  = 8   (exactly ONE owner per word -> no atomics)
//   subgroup size     = selected by the Xe2 compiler (16 or 32)
//   one bit           = one odd number; bit i <-> q = q_first + 2*i
// ============================================================================

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

// Tunables: overridable via `ocloc ... -options "-DNAME=value"` for
// systematic benchmarking. Defaults = measured optimum on B580 @ 1e10
// (see experiment log at the bottom of this file).
#ifndef WG_SIZE
#define WG_SIZE           256
#endif
// 2048 words = 16 KB SLM per tile (v1.8.1). With the prime-parallel medium
// stage, more resident work-groups beat fewer per-tile setups: sieve_s per
// 2e10 window 0.109 (64 KB) -> 0.100 (32 KB) -> 0.087 (16 KB); 8 KB and
// 128 KB lose. MUST match GB_TILE_WORDS in include/config.hpp.
#ifndef TILE_WORDS
#define TILE_WORDS        2048
#endif
#define TILE_BITS         ((u64)TILE_WORDS * 64UL)   // odd numbers per tile
#define WORDS_PER_THREAD  (TILE_WORDS / WG_SIZE)
#define BITS_PER_THREAD   (WORDS_PER_THREAD * 64)
#define MAX_LEFTOVERS     1048576     // compact exception list cap
// Goldbach ridge buffer of the phase-B checkers (host: RidgeTracker): word 0
// low half = u32 append count (atomic), then RIDGE_CAP pairs
// (even n, smallest witness p). MUST match gb::RIDGE_CAP (include/config.hpp).
#define RIDGE_CAP         65536
#ifndef WORDS_PER_ITEM
#define WORDS_PER_ITEM    4
#endif
#ifndef LARGE_PRIMES_PER_ITEM
#define LARGE_PRIMES_PER_ITEM 4
#endif
#ifndef EARLY_EXIT
#define EARLY_EXIT        1
#endif
#ifndef WRITE_CHECKED
#define WRITE_CHECKED     1
#endif
#ifndef REPORT_LEFTOVERS
#define REPORT_LEFTOVERS  1
#endif
// Small-prime mask sieve: primes < MASK_MAX_P are applied as whole-word ANDs
// from a precomputed host table (mask[p][phase]) instead of per-bit RMWs.
// 0 disables the path (falls back to the scalar loop for ALL primes).
#ifndef SMALL_PRIME_MASK
#define SMALL_PRIME_MASK  1
#endif
// Primes below this bound use the mask table. 64 -> 17 odd primes
// (3..61; 2 is not in the list), 17*64 = 1088 mask words of 8 B =
// 8.5 KB device memory. Tunable.
#ifndef MASK_MAX_P
#define MASK_MAX_P        64
#endif
// Primes per load batch in the per-tile-table path. The prime walk is
// latency-bound on its broadcast loads (table entry, prime, magic); the batch
// feeds a two-stage software pipeline (two live batches), so 16 independent
// loads are in flight per work-item.
#ifndef TABLE_BATCH
#define TABLE_BATCH       8
#endif
// Upper bound on small-prime mask groups of the fused mask stage (host
// small_prime_group_masks yields 8 for primes 3..61 with period <= 4096;
// a 16-group limit measured ~2% slower from the longer unrolled loop).
#define MAX_MASK_GROUPS   8

typedef unsigned long  u64;
typedef unsigned int   u32;
typedef unsigned short u16;



// ----------------------------------------------------------------------------
// x mod p without a 64-bit division (x < 2^63, odd p < 2^32).
// Xe2 has no integer divider: a u64 '/' compiles to a long emulated sequence,
// and sieve_large_primes did one per prime per segment (~98M at 4e18) — that
// setup, not the marking, was ~60% of the kernel. Here the quotient is
// estimated with a float reciprocal in two refinement steps and the
// remainder is then brought into [0, p) by exact integer fix-ups, so the
// RESULT IS EXACT for any float rounding: precision only bounds how many
// fix-up iterations run (for p >= 2^19: |r| <= ~2p after step 2).
// Validated bit-exact by tools/debug_fb (prep_large_fb) and tools/debug_map.
// ----------------------------------------------------------------------------
#ifndef FAST_MOD
#define FAST_MOD 1
#endif
#ifndef FF_MOD
#define FF_MOD 1
#endif
#ifndef LARGE_FIRST32
#define LARGE_FIRST32 1          // 32-bit first-multiple setup in sieve_large_packed
#endif
#ifndef LARGE_WHEEL30
#define LARGE_WHEEL30 1          // large-prime marks only for multipliers coprime to 30
#endif
#ifndef TILE_WHEEL30
#define TILE_WHEEL30 1           // same for the tile kernel's medium stage
#endif
#ifndef BUCKET_WHEEL30
#define BUCKET_WHEEL30 1         // bucket sieve: next hit = next multiplier coprime to 30
#endif
#ifndef TILE_WHEEL_MAXP
#define TILE_WHEEL_MAXP 16384u   // medium-stage wheel only below this prime: primes
                                 // with ~1 mark per 16 KB tile lose to the wheel's
                                 // setup (sieve_s: all 0.0966, <65536 0.0835,
                                 // <32768 0.0815, <16384 0.0810, <4096 0.0821, off 0.0848)
#endif
// (A wheel in the thread-owned stage, primes < MED_MIN_P, measured slower:
// 1-8 marks per thread range cannot pay for the residue setup.)

// Low 32 bits of the integer value of a non-negative, integer-valued float
// (|v| >= 2^23 or exactly integral), i.e. (u64)v mod 2^32, without a
// float -> u64 conversion.
/// @pre    v >= 0 and integer-valued (|v| >= 2^23 or exactly integral).
/// @return the low 32 bits of the integer value of v.
inline u32 float_int_low32(const float v)
{
    const u32 b = as_uint(v);
    const int e = (int)((b >> 23) & 0xFFu) - 150;   // v = m * 2^e, m < 2^24
    const u32 m = (b & 0x7FFFFFu) | 0x800000u;
    if (e >= 32 || v == 0.0f) return 0u;
    return (e >= 0) ? (m << e) : (m >> (-e));
}

// Float-float split of a (uniform) dividend x for mod_ff: computed once per
// work-item instead of once per prime.
typedef struct { float xh; float xl; u32 x32; } ff_x;
/// @pre    x < 2^63 (x - (u64)xh then fits a signed 64-bit integer).
/// @return {xh = float(x), xl = float(x - xh), x32 = x mod 2^32}: x = xh + xl
///         up to xl's rounding (|error| <= 2^-24 |x - xh| <= 2^(e(x) - 48)).
inline ff_x ff_split(const u64 x)
{
    ff_x r;
    r.xh = (float)x;
    r.xl = (float)(long)(x - (u64)r.xh);
    r.x32 = (u32)x;
    return r;
}
#ifndef FF_I32
#define FF_I32 1                 // mod_ff: p - float(p) in 32 bits (|.| < 2^6; +0.5%)
#endif
#ifndef FF_FIX
#define FF_FIX 2                 // mod_ff fix-ups: 1 = one guard test, 2 = selects + guard (+0.1%)
#endif
#ifndef FF_BIG
#define FF_BIG 1                 // mod_ff: quotient >= 2^23 (caller: x >= 2^60), no zero branch (+1.0%)
#endif
#if FF_BIG
// float_int_low32 for v >= 2^23 (exponent e >= 0, v != 0): no zero /
// negative-exponent branch.
/// @pre    v >= 2^23, integer-valued.
/// @return float_int_low32(v) (the low 32 bits of v's integer value).
inline u32 float_int_low32_big(const float v)
{
    const u32 b = as_uint(v);
    const u32 e = ((b >> 23) & 0xFFu) - 150u;       // >= 0 for v >= 2^23
    const u32 m = (b & 0x7FFFFFu) | 0x800000u;
    return (e < 32u) ? (m << e) : 0u;
}
#endif
// x mod p for p < 2^30 from a pre-split x (the FF_MOD math of mod_u64_u32).
// The float-float quotient q = qh + rint(ql) carries ~46 bits, so
// |q - floor(x / p)| <= (x / p) 2^-46 + 1: at most 1 for p >= 2^20 and a few
// units below (measured: 2), i.e. the wrap-around remainder r = x - q p
// satisfies |r| < 2^31 for every p < 2^30 (|r| <= 2.2 p < 2^21 for p < 2^20,
// < 2 p < 2^31 above). Hence r is exact in int, the selects fix the usual
// +-1 error and the guarded loops the rest. (Measured by CPU emulation over
// 1.4e7 (x, p) pairs: max |r| = 5.4e8.)
/// @pre    x from ff_split of a value < 2^63; 1 <= p < 2^30. For FF_BIG:
///         x / p >= 2^23 (callers: x >= 2^60), so qh >= 2^23.
/// @return x mod p, exact.
inline u32 mod_ff(const ff_x x, const u32 p)
{
    const float pf = (float)p;
#if FF_I32
    const int pr = (int)p - (int)pf;                 // p, pf <= 2^30: exact in int
#else
    const long pr = (long)p - (long)pf;
#endif
    const float ih = 1.0f / pf;
    const float e1 = fma(-pf, ih, 1.0f);
    const float il = ih * (e1 - (float)pr * ih);
    const float qh = rint(x.xh * ih);
    const float ql = fma(x.xh, ih, -qh) + x.xh * il + x.xl * ih;
#if FF_BIG
    const u32 q32 = float_int_low32_big(qh) + (u32)(int)rint(ql);
#else
    const u32 q32 = float_int_low32(qh) + (u32)(int)rint(ql);
#endif
    int r = (int)(x.x32 - q32 * p);
#if FF_FIX == 2
    r += (r < 0) ? (int)p : 0;                       // the usual +-1 quotient errors
    r -= (r >= (int)p) ? (int)p : 0;
#endif
#if FF_FIX
    if ((u32)r >= p) {                               // one test for both directions
        while (r < 0) r += (int)p;
        while (r >= (int)p) r -= (int)p;
    }
#else
    while (r < 0) r += (int)p;
    while (r >= (int)p) r -= (int)p;
#endif
    return (u32)r;
}

/// x mod p without a 64-bit division (see above; FF_MOD / FAST_MOD select
/// the method, both exact).
/// @pre    x < 2^63, 1 <= p < 2^32.
/// @return x mod p.
inline u32 mod_u64_u32(const u64 x, const u32 p)
{
#if FF_MOD
    // p < 2^30: float-float quotient + 32-bit remainder.
    // x = xh + xl (xh = float(x), xl = float(x - xh), exact enough); 1/p =
    // ih + il with il from the FMA residual. q ~= x/p is formed as
    // qh + ql, qh = round(xh*ih), ql = fma residual + cross terms, giving
    // ~46 bits: |q - x/p| < 1 for x < 2^63. Only q mod 2^32 is needed:
    // r = (u32)x - q32*p in wrap-around arithmetic is exact because the
    // true remainder lies in (-p, 2p) and fits 31 bits (p < 2^30). The
    // integer fix-ups make the result exact regardless of float rounding
    // as long as that bound holds; debug_fb checks all primes bit-exact.
    if (p < (1u << 30)) {
        const float xh = (float)x;
        const long xr = (long)(x - (u64)xh);        // |xr| <= ulp(xh)/2 < 2^39
        const float xl = (float)xr;
        const float pf = (float)p;                   // exact: p < 2^24? no — see il
        const long pr = (long)p - (long)pf;          // p - float(p), |pr| < 2^6
        const float ih = 1.0f / pf;
        // residual of pf*ih vs 1, then correct for p != pf as well
        const float e1 = fma(-pf, ih, 1.0f);
        const float il = ih * (e1 - (float)pr * ih);
        const float qh = rint(xh * ih);
        const float ql = fma(xh, ih, -qh) + xh * il + xl * ih;
        const u32 q32 = float_int_low32(qh) + (u32)(int)rint(ql);
        int r = (int)((u32)x - q32 * p);
        while (r < 0) r += (int)p;
        while (r >= (int)p) r -= (int)p;
        return (u32)r;
    }
#endif
#if FAST_MOD
    const float f = 1.0f / (float)p;
    const u64 qt = (u64)((float)x * f);             // |x/p - qt| <~ 2^-22 x/p
    long r = (long)(x - qt * (u64)p);                // |r| <~ 2^-22 x, exact
    r -= (long)((float)r * f) * (long)p;             // |r| <~ 2p (p >= 2^19)
    // (a third refinement step measured slower; small p just take more
    // fix-up iterations below — still exact)
    while (r < 0) r += (long)p;
    while (r >= (long)p) r -= (long)p;
    return (u32)r;
#else
    return (u32)(x % (u64)p);
#endif
}

// First markable ODD multiple of p inside [q_first, ...): max(p*p, the
// smallest odd multiple of p >= q_first). Shared by sieve_large_primes and
// prep_large_fb (so debug_fb validates exactly what production runs).
/// @pre    p odd prime, p < 2^32; q_first odd.
/// @return max(p*p, smallest odd multiple of p >= q_first). May exceed the
///         segment: the caller compares it with q_last.
inline u64 first_odd_multiple(const u64 p, const u64 q_first)
{
    u64 first = p * p;
    if (first < q_first) {
        const u32 r = mod_u64_u32(q_first, (u32)p);
        first = q_first + (r ? (p - r) : 0UL);       // smallest multiple >= q_first
        if ((first & 1) == 0) first += p;            // even multiple -> next (odd)
    }
    return first;
}


// ----------------------------------------------------------------------------
// Class-map layout (v1.11.0, "class mode"). The checker only reads q = 1 and
// q = 5 (mod 6) (every other odd q is divisible by 3), so the map is stored
// as two arrays: dense bit b (q = q_first + 2b) with b = 3t + phi1 is bit t of
// M1 (q = 1 mod 6), b = 3t + phi5 is bit t of M5 at bit offset off5; bits with
// q = 3 (mod 6) are not stored. Mark loops keep dense coordinates and convert
// per mark (+0.9% on the atomic kernels, measured); cm = 0 is the dense map.
// ----------------------------------------------------------------------------
// cm = 2 (mod-30 class map): 8 arrays for q = 1, 7, 11, ..., 29 (mod 30),
// dense bit b = 15t + phi_k is bit t of array k (at bit k * off5); phi1 /
// phi5 then carry the 15-entry phase -> class nibble table (b mod 15 ->
// k, 15 = q divisible by 3 or 5: not stored), low 8 nibbles / high 7.
#ifndef MARK_ROL
#define MARK_ROL 1               // clear masks by one rol (+0.45%)
#endif
// AND mask clearing bit (b mod 32) of a u32 word: ~(1 << (b & 31)) is and +
// shl + not; rotate(~1, b) is one native rol (the count is taken mod 32).
#if MARK_ROL
#define CLR_MASK(b) rotate(0xFFFFFFFEu, (u32)(b))
#else
#define CLR_MASK(b) (~(1u << ((b) & 31u)))
#endif
typedef struct { u32 cm, phi1, phi5, off5; } cmap;
// LSC cache-control operands of the IGC load / store builtins (IGC enum
// values: LDCC 1 = L1UC_L3UC, 2 = L1UC_L3C, 5 = L1S_L3UC).
enum LSC_STCC { LSC_STCC_DEFAULT = 0 };
enum LSC_LDCC { LSC_LDCC_DEFAULT = 0 };
/// @param ph  dense bit phase mod 15 (cm = 2 only).
/// @pre    ph < 15; g.phi1 / g.phi5 = the host's phase -> class nibble table
///         (low 8 / high 7 phases; 15 = no class), built for this segment.
/// @return class array k in [0, 8) of that phase, or 15 when q is divisible
///         by 3 or 5 (not stored).
inline u32 c30_class(const u32 ph, const cmap g)
{
    return ((ph < 8u) ? (g.phi1 >> (4u * ph)) : (g.phi5 >> (4u * (ph - 8u)))) & 15u;
}
/// Clears dense map bit @p bit (q = q_first + 2 bit) with a 32-bit atomic,
/// converting to class coordinates when g.cm != 0; bits of q not stored in
/// the class map (q = 0 mod 3, or mod 5 for cm = 2) are skipped.
/// @pre    bit < 2^32 and its class address lies inside the map; concurrent
///         callers allowed (atomic AND: clears commute).
/// @post   the bit of q = q_first + 2 bit is cleared if q is stored, nothing
///         else changes.
inline void map_clear(volatile __global u32* m32, const u32 bit, const cmap g)
{
    u32 ad = bit;
    if (g.cm == 2u) {                                 // uniform
        const u32 t = bit / 15u;
        const u32 k = c30_class(bit - 15u * t, g);
        if (k == 15u) return;                         // q divisible by 3 or 5
        ad = k * g.off5 + t;
    } else if (g.cm) {                                // uniform
        const u32 t = bit / 3u;
        const u32 ph = bit - 3u * t;
        if (ph == g.phi1) ad = t;
        else if (ph == g.phi5) ad = g.off5 + t;
        else return;                                  // q = 3 (mod 6): not stored
    }
    atomic_and(m32 + (ad >> 5), CLR_MASK(ad));
}
#ifndef MAPCLR_W
#define MAPCLR_W 1
#endif
// map_clear for multiplier-wheel marks (m coprime to 6, p > 3): q = p m is
// never 3 (mod 6), so the phase is phi1 or phi5 and the class conversion is
// branch-free.
/// map_clear for multiplier-wheel marks: branch-free class conversion.
/// @pre    q = q_first + 2 bit is coprime to 30 for cm = 2 (q = p m with m
///         coprime to 30030 and p > 13 gives that; so does gcd(q, 15015) = 1
///         for odd q), coprime to 6 for cm = 1; bit < 2^32; concurrent
///         callers allowed (atomic AND).
/// @post   the bit of q is cleared; nothing else changes. (A q violating
///         the PRE has class 15 and would address 15 * off5 + t, outside
///         the eight class arrays: every caller marks only q coprime to
///         15015 — multiplier wheel, CP15015 test — so this cannot occur.)
inline void map_clear_w(volatile __global u32* m32, const u32 bit, const cmap g)
{
#if MAPCLR_W
    u32 ad = bit;
    if (g.cm == 2u) {                                 // multiplier coprime to 30030: k valid
        const u32 t = bit / 15u;
        ad = c30_class(bit - 15u * t, g) * g.off5 + t;
    } else if (g.cm) {                                // uniform
        const u32 t = bit / 3u;
        ad = t + ((bit - 3u * t == g.phi5) ? g.off5 : 0u);
    }
    atomic_and(m32 + (ad >> 5), CLR_MASK(ad));
#else
    map_clear(m32, bit, g);
#endif
}

// ----------------------------------------------------------------------------
// Clear the odd multiples bit0, bit0 + p, ... (< q_odds) with 32-bit atomics.
// All arithmetic is 32-bit: Xe2 has no native 64-bit integer ALU, and the
// former u64 loop (bit < q_odds; bit += p) compiled to ~39 instructions per
// mark (emulated 64-bit add/compare + register moves) around one atomic.
// q_odds < 2^32 is guaranteed by the host; the loop condition n - bit > p
// is the overflow-free form of bit + p < n, so no headroom is assumed.
// The bit lives in exactly one u32 word; AND-clearing is monotone, so
// concurrent clears converge under any interleaving.
// ----------------------------------------------------------------------------
/// @param bit0    first map bit to clear (odd multiple of p).
/// @param q_odds  valid map bits; bits >= q_odds are never touched.
/// @pre    q_odds < 2^32, p < 2^32.
/// @post   bits bit0, bit0 + p, ... < q_odds are cleared (map_clear).
inline void mark_multiples_u32(volatile __global u64* prime_map, const u64 bit0,
                               const u64 p, const u64 q_odds, const cmap g)
{
    volatile __global u32* m32 = (volatile __global u32*)prime_map;
    const u32 n = (u32)q_odds;
    const u32 step = (u32)p;
    u32 bit = (u32)bit0;
    if (bit >= n) return;
    for (;;) {
        map_clear(m32, bit, g);
        if (n - bit <= step) break;
        bit += step;
    }
}


// ----------------------------------------------------------------------------
// Thread-owned tile marks: clear bit0, bit0 + p, ... (< end) in SLM with
// 32-bit words and 32-bit arithmetic (the u64 form — 64-bit shift of 1UL,
// u64 add/compare — compiled to ~37 instructions per mark on Xe2). The
// calling thread owns every word in [bit0, end), so plain RMWs are safe.
// Overflow-free loop condition (end - bit > p), no headroom assumed.
// ----------------------------------------------------------------------------
/// @pre    the calling thread owns every SLM word of [bit0, end); end < 2^32.
/// @post   tile bits bit0, bit0 + p, ... < end are cleared.
inline void mark_owned_u32(__local u64* tile, const u64 bit0, const u64 end, const u64 p)
{
    if (bit0 >= end) return;
    __local u32* t32 = (__local u32*)tile;
    const u32 e = (u32)end;
    const u32 step = (u32)p;
    u32 bit = (u32)bit0;
    for (;;) {
        t32[bit >> 5] &= CLR_MASK(bit);
        if (e - bit <= step) break;
        bit += step;
    }
}


// ----------------------------------------------------------------------------
// Wheel-30 marking (v1.9.0). A composite q whose smallest prime factor is
// s >= 7 is q = s*m with every prime factor of m >= s, so m is coprime to 30:
// prime s only has to clear s*m for m coprime to 30. Every other multiple
// s*m (3 | m or 5 | m) is divisible by 3 or 5 and is cleared by the
// small-prime stage, which marks EVERY multiple of 3 and 5 (masks, or the
// thread-owned walk for p <= 5 without the wheel). So only 8 of every 15
// odd multipliers are marked: 47% fewer marks. Valid for p > 5 only.
// The odd multipliers coprime to 30 step by 6,4,2,4,2,4,6,2 — in map bits
// (one bit per odd number) by 3p,2p,p,2p,p,2p,3p,p: nibble-packed below.
// W30_IDX[m] = position of residue m in the cycle (0xFF: not coprime);
// W30_INV[r] = r^-1 mod 30 (to get m0 mod 30 from first = m0 * p).
// ----------------------------------------------------------------------------
// Multiplier wheel of the large-prime and bucket marking: only p*m with m
// coprime to WHEEL_W (MUST match the host's WHEEL_W). Measured large_s per
// 2e10 window (v1.9.0): none 0.273, 30 0.212, 210 0.202, 2310 0.191, 30030
// 0.189. v1.10.1: 30030 (5760 of 15015 odd multipliers, -7.7% marks vs 2310)
// once the mark loops, not the setup, dominated: packed kernel +0.7%, and the
// bucket on it (ring up to ~199 buckets, BKT_MAX_R 256, ~6.9 GiB VRAM at
// 3 x q_odds) +1.2..1.6% across the range. Needs the mask stage to clear every
// multiple of 3..13 (it does: 3..61) and p > 13 (callers: p > 61).
#ifndef WHEEL_W
#define WHEEL_W 30030
#endif
// ---- generated by the v1.9.0 wheel generator (odd multipliers coprime to W) ----
#if WHEEL_W == 30
#define WHL_N 8
#define WHL_MISS 0xFFu
__constant uchar WHL_IDX[30] = { 0xFF, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1, 0xFF, 0xFF, 0xFF, 2, 0xFF, 3, 0xFF, 0xFF, 0xFF, 4, 0xFF, 5, 0xFF, 0xFF, 0xFF, 6, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 7 };
__constant ushort WHL_INV[30] = { 0, 1, 0, 0, 0, 0, 0, 13, 0, 0, 0, 11, 0, 7, 0, 0, 0, 23, 0, 19, 0, 0, 0, 17, 0, 0, 0, 0, 0, 29 };
__constant uchar WHL_STEP[8] = { 3, 2, 1, 2, 1, 2, 3, 1 };
#endif
#if WHEEL_W == 210
#define WHL_N 48
#define WHL_MISS 0xFFu
__constant uchar WHL_IDX[210] = { 0xFF, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1, 0xFF, 2, 0xFF, 0xFF, 0xFF, 3, 0xFF, 4, 0xFF, 0xFF, 0xFF, 5, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 6, 0xFF, 7, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 8, 0xFF, 0xFF, 0xFF, 9, 0xFF, 10, 0xFF, 0xFF, 0xFF, 11, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 12, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 13, 0xFF, 14, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 15, 0xFF, 0xFF, 0xFF, 16, 0xFF, 17, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 18, 0xFF, 0xFF, 0xFF, 19, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 20, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 21, 0xFF, 0xFF, 0xFF, 22, 0xFF, 23, 0xFF, 0xFF, 0xFF, 24, 0xFF, 25, 0xFF, 0xFF, 0xFF, 26, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 27, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 28, 0xFF, 0xFF, 0xFF, 29, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 30, 0xFF, 31, 0xFF, 0xFF, 0xFF, 32, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 33, 0xFF, 34, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 35, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 36, 0xFF, 0xFF, 0xFF, 37, 0xFF, 38, 0xFF, 0xFF, 0xFF, 39, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 40, 0xFF, 41, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 42, 0xFF, 0xFF, 0xFF, 43, 0xFF, 44, 0xFF, 0xFF, 0xFF, 45, 0xFF, 46, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 47 };
__constant ushort WHL_INV[210] = { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 191, 0, 97, 0, 0, 0, 173, 0, 199, 0, 0, 0, 137, 0, 0, 0, 0, 0, 29, 0, 61, 0, 0, 0, 0, 0, 193, 0, 0, 0, 41, 0, 127, 0, 0, 0, 143, 0, 0, 0, 0, 0, 107, 0, 0, 0, 0, 0, 89, 0, 31, 0, 0, 0, 0, 0, 163, 0, 0, 0, 71, 0, 187, 0, 0, 0, 0, 0, 109, 0, 0, 0, 167, 0, 0, 0, 0, 0, 59, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 131, 0, 157, 0, 0, 0, 53, 0, 79, 0, 0, 0, 197, 0, 0, 0, 0, 0, 0, 0, 151, 0, 0, 0, 0, 0, 43, 0, 0, 0, 101, 0, 0, 0, 0, 0, 23, 0, 139, 0, 0, 0, 47, 0, 0, 0, 0, 0, 179, 0, 121, 0, 0, 0, 0, 0, 103, 0, 0, 0, 0, 0, 67, 0, 0, 0, 83, 0, 169, 0, 0, 0, 17, 0, 0, 0, 0, 0, 149, 0, 181, 0, 0, 0, 0, 0, 73, 0, 0, 0, 11, 0, 37, 0, 0, 0, 113, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209 };
__constant uchar WHL_STEP[48] = { 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1 };
#endif
#if WHEEL_W == 2310
#define WHL_N 480
#define WHL_MISS 0xFFFFu
__constant ushort WHL_IDX[2310] = { 0xFFFF, 0, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 1, 0xFFFF, 0xFFFF, 0xFFFF, 2, 0xFFFF, 3, 0xFFFF, 0xFFFF, 0xFFFF, 4, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 5, 0xFFFF, 6, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 7, 0xFFFF, 0xFFFF, 0xFFFF, 8, 0xFFFF, 9, 0xFFFF, 0xFFFF, 0xFFFF, 10, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 11, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 12, 0xFFFF, 13, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 14, 0xFFFF, 0xFFFF, 0xFFFF, 15, 0xFFFF, 16, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 17, 0xFFFF, 0xFFFF, 0xFFFF, 18, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 19, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 20, 0xFFFF, 0xFFFF, 0xFFFF, 21, 0xFFFF, 22, 0xFFFF, 0xFFFF, 0xFFFF, 23, 0xFFFF, 24, 0xFFFF, 0xFFFF, 0xFFFF, 25, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 26, 0xFFFF, 0xFFFF, 0xFFFF, 27, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 28, 0xFFFF, 29, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 30, 0xFFFF, 31, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 32, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 33, 0xFFFF, 0xFFFF, 0xFFFF, 34, 0xFFFF, 35, 0xFFFF, 0xFFFF, 0xFFFF, 36, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 37, 0xFFFF, 38, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 39, 0xFFFF, 40, 0xFFFF, 0xFFFF, 0xFFFF, 41, 0xFFFF, 42, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 43, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 44, 0xFFFF, 45, 0xFFFF, 0xFFFF, 0xFFFF, 46, 0xFFFF, 47, 0xFFFF, 0xFFFF, 0xFFFF, 48, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 49, 0xFFFF, 50, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 51, 0xFFFF, 0xFFFF, 0xFFFF, 52, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 53, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 54, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 55, 0xFFFF, 56, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 57, 0xFFFF, 0xFFFF, 0xFFFF, 58, 0xFFFF, 59, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 60, 0xFFFF, 0xFFFF, 0xFFFF, 61, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 62, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 63, 0xFFFF, 0xFFFF, 0xFFFF, 64, 0xFFFF, 65, 0xFFFF, 0xFFFF, 0xFFFF, 66, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 67, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 68, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 69, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 70, 0xFFFF, 71, 0xFFFF, 0xFFFF, 0xFFFF, 72, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 73, 0xFFFF, 74, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 75, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 76, 0xFFFF, 0xFFFF, 0xFFFF, 77, 0xFFFF, 78, 0xFFFF, 0xFFFF, 0xFFFF, 79, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 80, 0xFFFF, 81, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 82, 0xFFFF, 0xFFFF, 0xFFFF, 83, 0xFFFF, 84, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 85, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 86, 0xFFFF, 87, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 88, 0xFFFF, 89, 0xFFFF, 0xFFFF, 0xFFFF, 90, 0xFFFF, 91, 0xFFFF, 0xFFFF, 0xFFFF, 92, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 93, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 94, 0xFFFF, 0xFFFF, 0xFFFF, 95, 0xFFFF, 96, 0xFFFF, 0xFFFF, 0xFFFF, 97, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 98, 0xFFFF, 99, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 100, 0xFFFF, 0xFFFF, 0xFFFF, 101, 0xFFFF, 102, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 103, 0xFFFF, 0xFFFF, 0xFFFF, 104, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 105, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 106, 0xFFFF, 107, 0xFFFF, 0xFFFF, 0xFFFF, 108, 0xFFFF, 109, 0xFFFF, 0xFFFF, 0xFFFF, 110, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 111, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 112, 0xFFFF, 0xFFFF, 0xFFFF, 113, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 114, 0xFFFF, 115, 0xFFFF, 0xFFFF, 0xFFFF, 116, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 117, 0xFFFF, 118, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 119, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 120, 0xFFFF, 121, 0xFFFF, 0xFFFF, 0xFFFF, 122, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 123, 0xFFFF, 124, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 125, 0xFFFF, 0xFFFF, 0xFFFF, 126, 0xFFFF, 127, 0xFFFF, 0xFFFF, 0xFFFF, 128, 0xFFFF, 129, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 130, 0xFFFF, 131, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 132, 0xFFFF, 133, 0xFFFF, 0xFFFF, 0xFFFF, 134, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 135, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 136, 0xFFFF, 137, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 138, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 139, 0xFFFF, 0xFFFF, 0xFFFF, 140, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 141, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 142, 0xFFFF, 143, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 144, 0xFFFF, 0xFFFF, 0xFFFF, 145, 0xFFFF, 146, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 147, 0xFFFF, 0xFFFF, 0xFFFF, 148, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 149, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 150, 0xFFFF, 0xFFFF, 0xFFFF, 151, 0xFFFF, 152, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 153, 0xFFFF, 0xFFFF, 0xFFFF, 154, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 155, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 156, 0xFFFF, 0xFFFF, 0xFFFF, 157, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 158, 0xFFFF, 159, 0xFFFF, 0xFFFF, 0xFFFF, 160, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 161, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 162, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 163, 0xFFFF, 0xFFFF, 0xFFFF, 164, 0xFFFF, 165, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 166, 0xFFFF, 167, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 168, 0xFFFF, 0xFFFF, 0xFFFF, 169, 0xFFFF, 170, 0xFFFF, 0xFFFF, 0xFFFF, 171, 0xFFFF, 172, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 173, 0xFFFF, 174, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 175, 0xFFFF, 176, 0xFFFF, 0xFFFF, 0xFFFF, 177, 0xFFFF, 178, 0xFFFF, 0xFFFF, 0xFFFF, 179, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 180, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 181, 0xFFFF, 0xFFFF, 0xFFFF, 182, 0xFFFF, 183, 0xFFFF, 0xFFFF, 0xFFFF, 184, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 185, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 186, 0xFFFF, 187, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 188, 0xFFFF, 0xFFFF, 0xFFFF, 189, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 190, 0xFFFF, 0xFFFF, 0xFFFF, 191, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 192, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 193, 0xFFFF, 0xFFFF, 0xFFFF, 194, 0xFFFF, 195, 0xFFFF, 0xFFFF, 0xFFFF, 196, 0xFFFF, 197, 0xFFFF, 0xFFFF, 0xFFFF, 198, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 199, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 200, 0xFFFF, 0xFFFF, 0xFFFF, 201, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 202, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 203, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 204, 0xFFFF, 205, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 206, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 207, 0xFFFF, 0xFFFF, 0xFFFF, 208, 0xFFFF, 209, 0xFFFF, 0xFFFF, 0xFFFF, 210, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 211, 0xFFFF, 212, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 213, 0xFFFF, 0xFFFF, 0xFFFF, 214, 0xFFFF, 215, 0xFFFF, 0xFFFF, 0xFFFF, 216, 0xFFFF, 217, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 218, 0xFFFF, 219, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 220, 0xFFFF, 221, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 222, 0xFFFF, 0xFFFF, 0xFFFF, 223, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 224, 0xFFFF, 225, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 226, 0xFFFF, 0xFFFF, 0xFFFF, 227, 0xFFFF, 228, 0xFFFF, 0xFFFF, 0xFFFF, 229, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 230, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 231, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 232, 0xFFFF, 0xFFFF, 0xFFFF, 233, 0xFFFF, 234, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 235, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 236, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 237, 0xFFFF, 0xFFFF, 0xFFFF, 238, 0xFFFF, 239, 0xFFFF, 0xFFFF, 0xFFFF, 240, 0xFFFF, 241, 0xFFFF, 0xFFFF, 0xFFFF, 242, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 243, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 244, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 245, 0xFFFF, 246, 0xFFFF, 0xFFFF, 0xFFFF, 247, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 248, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 249, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 250, 0xFFFF, 0xFFFF, 0xFFFF, 251, 0xFFFF, 252, 0xFFFF, 0xFFFF, 0xFFFF, 253, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 254, 0xFFFF, 255, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 256, 0xFFFF, 0xFFFF, 0xFFFF, 257, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 258, 0xFFFF, 259, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 260, 0xFFFF, 261, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 262, 0xFFFF, 263, 0xFFFF, 0xFFFF, 0xFFFF, 264, 0xFFFF, 265, 0xFFFF, 0xFFFF, 0xFFFF, 266, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 267, 0xFFFF, 268, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 269, 0xFFFF, 0xFFFF, 0xFFFF, 270, 0xFFFF, 271, 0xFFFF, 0xFFFF, 0xFFFF, 272, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 273, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 274, 0xFFFF, 275, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 276, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 277, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 278, 0xFFFF, 0xFFFF, 0xFFFF, 279, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 280, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 281, 0xFFFF, 0xFFFF, 0xFFFF, 282, 0xFFFF, 283, 0xFFFF, 0xFFFF, 0xFFFF, 284, 0xFFFF, 285, 0xFFFF, 0xFFFF, 0xFFFF, 286, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 287, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 288, 0xFFFF, 0xFFFF, 0xFFFF, 289, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 290, 0xFFFF, 0xFFFF, 0xFFFF, 291, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 292, 0xFFFF, 293, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 294, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 295, 0xFFFF, 0xFFFF, 0xFFFF, 296, 0xFFFF, 297, 0xFFFF, 0xFFFF, 0xFFFF, 298, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 299, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 300, 0xFFFF, 0xFFFF, 0xFFFF, 301, 0xFFFF, 302, 0xFFFF, 0xFFFF, 0xFFFF, 303, 0xFFFF, 304, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 305, 0xFFFF, 306, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 307, 0xFFFF, 308, 0xFFFF, 0xFFFF, 0xFFFF, 309, 0xFFFF, 310, 0xFFFF, 0xFFFF, 0xFFFF, 311, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 312, 0xFFFF, 313, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 314, 0xFFFF, 315, 0xFFFF, 0xFFFF, 0xFFFF, 316, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 317, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 318, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 319, 0xFFFF, 0xFFFF, 0xFFFF, 320, 0xFFFF, 321, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 322, 0xFFFF, 0xFFFF, 0xFFFF, 323, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 324, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 325, 0xFFFF, 0xFFFF, 0xFFFF, 326, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 327, 0xFFFF, 328, 0xFFFF, 0xFFFF, 0xFFFF, 329, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 330, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 331, 0xFFFF, 0xFFFF, 0xFFFF, 332, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 333, 0xFFFF, 334, 0xFFFF, 0xFFFF, 0xFFFF, 335, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 336, 0xFFFF, 337, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 338, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 339, 0xFFFF, 0xFFFF, 0xFFFF, 340, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 341, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 342, 0xFFFF, 343, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 344, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 345, 0xFFFF, 0xFFFF, 0xFFFF, 346, 0xFFFF, 347, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 348, 0xFFFF, 349, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 350, 0xFFFF, 351, 0xFFFF, 0xFFFF, 0xFFFF, 352, 0xFFFF, 353, 0xFFFF, 0xFFFF, 0xFFFF, 354, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 355, 0xFFFF, 356, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 357, 0xFFFF, 0xFFFF, 0xFFFF, 358, 0xFFFF, 359, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 360, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 361, 0xFFFF, 362, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 363, 0xFFFF, 0xFFFF, 0xFFFF, 364, 0xFFFF, 365, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 366, 0xFFFF, 0xFFFF, 0xFFFF, 367, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 368, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 369, 0xFFFF, 0xFFFF, 0xFFFF, 370, 0xFFFF, 371, 0xFFFF, 0xFFFF, 0xFFFF, 372, 0xFFFF, 373, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 374, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 375, 0xFFFF, 0xFFFF, 0xFFFF, 376, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 377, 0xFFFF, 378, 0xFFFF, 0xFFFF, 0xFFFF, 379, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 380, 0xFFFF, 381, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 382, 0xFFFF, 0xFFFF, 0xFFFF, 383, 0xFFFF, 384, 0xFFFF, 0xFFFF, 0xFFFF, 385, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 386, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 387, 0xFFFF, 0xFFFF, 0xFFFF, 388, 0xFFFF, 389, 0xFFFF, 0xFFFF, 0xFFFF, 390, 0xFFFF, 391, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 392, 0xFFFF, 393, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 394, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 395, 0xFFFF, 396, 0xFFFF, 0xFFFF, 0xFFFF, 397, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 398, 0xFFFF, 399, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 400, 0xFFFF, 0xFFFF, 0xFFFF, 401, 0xFFFF, 402, 0xFFFF, 0xFFFF, 0xFFFF, 403, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 404, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 405, 0xFFFF, 406, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 407, 0xFFFF, 0xFFFF, 0xFFFF, 408, 0xFFFF, 409, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 410, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 411, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 412, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 413, 0xFFFF, 0xFFFF, 0xFFFF, 414, 0xFFFF, 415, 0xFFFF, 0xFFFF, 0xFFFF, 416, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 417, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 418, 0xFFFF, 0xFFFF, 0xFFFF, 419, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 420, 0xFFFF, 421, 0xFFFF, 0xFFFF, 0xFFFF, 422, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 423, 0xFFFF, 424, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 425, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 426, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 427, 0xFFFF, 0xFFFF, 0xFFFF, 428, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 429, 0xFFFF, 430, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 431, 0xFFFF, 0xFFFF, 0xFFFF, 432, 0xFFFF, 433, 0xFFFF, 0xFFFF, 0xFFFF, 434, 0xFFFF, 435, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 436, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 437, 0xFFFF, 438, 0xFFFF, 0xFFFF, 0xFFFF, 439, 0xFFFF, 440, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 441, 0xFFFF, 442, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 443, 0xFFFF, 0xFFFF, 0xFFFF, 444, 0xFFFF, 445, 0xFFFF, 0xFFFF, 0xFFFF, 446, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 447, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 448, 0xFFFF, 449, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 450, 0xFFFF, 451, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 452, 0xFFFF, 0xFFFF, 0xFFFF, 453, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 454, 0xFFFF, 0xFFFF, 0xFFFF, 455, 0xFFFF, 456, 0xFFFF, 0xFFFF, 0xFFFF, 457, 0xFFFF, 458, 0xFFFF, 0xFFFF, 0xFFFF, 459, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 460, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 461, 0xFFFF, 0xFFFF, 0xFFFF, 462, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 463, 0xFFFF, 464, 0xFFFF, 0xFFFF, 0xFFFF, 465, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 466, 0xFFFF, 467, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 468, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 469, 0xFFFF, 0xFFFF, 0xFFFF, 470, 0xFFFF, 471, 0xFFFF, 0xFFFF, 0xFFFF, 472, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 473, 0xFFFF, 474, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 475, 0xFFFF, 0xFFFF, 0xFFFF, 476, 0xFFFF, 477, 0xFFFF, 0xFFFF, 0xFFFF, 478, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 479 };
__constant ushort WHL_INV[2310] = { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1777, 0, 0, 0, 1223, 0, 1459, 0, 0, 0, 1607, 0, 0, 0, 0, 0, 239, 0, 2161, 0, 0, 0, 0, 0, 1873, 0, 0, 0, 2141, 0, 967, 0, 0, 0, 983, 0, 0, 0, 0, 0, 1787, 0, 0, 0, 0, 0, 509, 0, 871, 0, 0, 0, 0, 0, 793, 0, 0, 0, 911, 0, 1867, 0, 0, 0, 0, 0, 1579, 0, 0, 0, 167, 0, 0, 0, 0, 0, 1739, 0, 0, 0, 0, 0, 0, 0, 643, 0, 0, 0, 1601, 0, 157, 0, 0, 0, 1943, 0, 1759, 0, 0, 0, 1247, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 673, 0, 0, 0, 1781, 0, 0, 0, 0, 0, 1703, 0, 349, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2279, 0, 1591, 0, 0, 0, 0, 0, 103, 0, 0, 0, 0, 0, 907, 0, 0, 0, 83, 0, 2269, 0, 0, 0, 227, 0, 0, 0, 0, 0, 2039, 0, 1021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 641, 0, 1927, 0, 0, 0, 1583, 0, 859, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1051, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1871, 0, 1357, 0, 0, 0, 173, 0, 1039, 0, 0, 0, 347, 0, 0, 0, 0, 0, 29, 0, 901, 0, 0, 0, 0, 0, 823, 0, 0, 0, 1721, 0, 0, 0, 0, 0, 773, 0, 0, 0, 0, 0, 527, 0, 0, 0, 0, 0, 1769, 0, 2131, 0, 0, 0, 0, 0, 1843, 0, 0, 0, 1751, 0, 1657, 0, 0, 0, 0, 0, 1159, 0, 0, 0, 1427, 0, 0, 0, 0, 0, 479, 0, 0, 0, 0, 0, 0, 0, 1693, 0, 0, 0, 1181, 0, 1417, 0, 0, 0, 1523, 0, 0, 0, 0, 0, 1037, 0, 0, 0, 0, 0, 0, 0, 991, 0, 0, 0, 0, 0, 1933, 0, 0, 0, 0, 0, 0, 0, 0, 0, 233, 0, 139, 0, 0, 0, 1937, 0, 0, 0, 0, 0, 1229, 0, 1171, 0, 0, 0, 0, 0, 2203, 0, 0, 0, 0, 0, 1957, 0, 0, 0, 1973, 0, 1219, 0, 0, 0, 2117, 0, 0, 0, 0, 0, 1829, 0, 1861, 0, 0, 0, 0, 0, 1123, 0, 0, 0, 1901, 0, 877, 0, 0, 0, 0, 0, 1909, 0, 0, 0, 0, 0, 0, 0, 0, 0, 419, 0, 631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 611, 0, 1147, 0, 0, 0, 2273, 0, 2089, 0, 0, 0, 2237, 0, 0, 0, 0, 0, 1919, 0, 0, 0, 0, 0, 0, 0, 1663, 0, 0, 0, 461, 0, 1387, 0, 0, 0, 2033, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 299, 0, 1921, 0, 0, 0, 0, 0, 1423, 0, 0, 0, 701, 0, 1237, 0, 0, 0, 0, 0, 949, 0, 0, 0, 1217, 0, 0, 0, 0, 0, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 971, 0, 2257, 0, 0, 0, 263, 0, 2179, 0, 0, 0, 2297, 0, 0, 0, 0, 0, 0, 0, 2041, 0, 0, 0, 0, 0, 1723, 0, 0, 0, 2201, 0, 0, 0, 0, 0, 1493, 0, 2029, 0, 0, 0, 677, 0, 0, 0, 0, 0, 1019, 0, 2221, 0, 0, 0, 0, 0, 1153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1763, 0, 2059, 0, 0, 0, 857, 0, 0, 0, 0, 0, 779, 0, 811, 0, 0, 0, 0, 0, 2173, 0, 0, 0, 431, 0, 667, 0, 0, 0, 2003, 0, 1489, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1469, 0, 421, 0, 0, 0, 0, 0, 0, 0, 0, 0, 191, 0, 97, 0, 0, 0, 1853, 0, 0, 0, 0, 0, 2027, 0, 0, 0, 0, 0, 659, 0, 1321, 0, 0, 0, 0, 0, 613, 0, 0, 0, 0, 0, 127, 0, 0, 0, 563, 0, 0, 0, 0, 0, 947, 0, 0, 0, 0, 0, 1559, 0, 1501, 0, 0, 0, 0, 0, 1213, 0, 0, 0, 491, 0, 2287, 0, 0, 0, 0, 0, 2209, 0, 0, 0, 797, 0, 0, 0, 0, 0, 2159, 0, 0, 0, 0, 0, 0, 0, 2113, 0, 0, 0, 2231, 0, 1207, 0, 0, 0, 0, 0, 919, 0, 0, 0, 827, 0, 0, 0, 0, 0, 0, 0, 1621, 0, 0, 0, 0, 0, 1303, 0, 0, 0, 941, 0, 0, 0, 0, 0, 1283, 0, 769, 0, 0, 0, 257, 0, 0, 0, 0, 0, 599, 0, 0, 0, 0, 0, 0, 0, 1993, 0, 0, 0, 0, 0, 67, 0, 0, 0, 713, 0, 1009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1619, 0, 601, 0, 0, 0, 0, 0, 1753, 0, 0, 0, 1691, 0, 247, 0, 0, 0, 743, 0, 1279, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1049, 0, 1681, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2291, 0, 937, 0, 0, 0, 593, 0, 199, 0, 0, 0, 977, 0, 0, 0, 0, 0, 0, 0, 61, 0, 0, 0, 0, 0, 403, 0, 0, 0, 881, 0, 2017, 0, 0, 0, 1823, 0, 0, 0, 0, 0, 1997, 0, 0, 0, 0, 0, 1349, 0, 241, 0, 0, 0, 0, 0, 163, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 739, 0, 0, 0, 1847, 0, 0, 0, 0, 0, 1109, 0, 0, 0, 0, 0, 0, 0, 853, 0, 0, 0, 761, 0, 997, 0, 0, 0, 683, 0, 499, 0, 0, 0, 2087, 0, 0, 0, 0, 0, 0, 0, 1411, 0, 0, 0, 0, 0, 43, 0, 0, 0, 521, 0, 0, 0, 0, 0, 863, 0, 0, 0, 0, 0, 47, 0, 0, 0, 0, 0, 1649, 0, 331, 0, 0, 0, 0, 0, 943, 0, 0, 0, 0, 0, 1117, 0, 0, 0, 1553, 0, 799, 0, 0, 0, 1277, 0, 0, 0, 0, 0, 569, 0, 181, 0, 0, 0, 0, 0, 1543, 0, 0, 0, 1481, 0, 1297, 0, 0, 0, 323, 0, 229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 839, 0, 211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1241, 0, 2197, 0, 0, 0, 0, 0, 1249, 0, 0, 0, 1817, 0, 0, 0, 0, 0, 1079, 0, 1951, 0, 0, 0, 0, 0, 2293, 0, 0, 0, 1931, 0, 1807, 0, 0, 0, 1613, 0, 0, 0, 0, 0, 1577, 0, 0, 0, 0, 0, 929, 0, 0, 0, 0, 0, 0, 0, 1003, 0, 0, 0, 1121, 0, 397, 0, 0, 0, 0, 0, 1999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1949, 0, 0, 0, 0, 0, 0, 0, 433, 0, 0, 0, 2021, 0, 577, 0, 0, 0, 1733, 0, 289, 0, 0, 0, 1877, 0, 0, 0, 0, 0, 0, 0, 361, 0, 0, 0, 0, 0, 0, 0, 0, 0, 311, 0, 0, 0, 0, 0, 1913, 0, 1189, 0, 0, 0, 1307, 0, 0, 0, 0, 0, 0, 0, 1381, 0, 0, 0, 0, 0, 733, 0, 0, 0, 0, 0, 697, 0, 0, 0, 503, 0, 379, 0, 0, 0, 17, 0, 0, 0, 0, 0, 359, 0, 1231, 0, 0, 0, 0, 0, 493, 0, 0, 0, 1061, 0, 0, 0, 0, 0, 113, 0, 1069, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2099, 0, 1471, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2081, 0, 1987, 0, 0, 0, 1013, 0, 829, 0, 0, 0, 767, 0, 0, 0, 0, 0, 2129, 0, 1741, 0, 0, 0, 0, 0, 1033, 0, 0, 0, 1511, 0, 757, 0, 0, 0, 1193, 0, 0, 0, 0, 0, 1367, 0, 0, 0, 0, 0, 1979, 0, 661, 0, 0, 0, 0, 0, 2263, 0, 0, 0, 0, 0, 1447, 0, 0, 0, 0, 0, 1789, 0, 0, 0, 2267, 0, 0, 0, 0, 0, 899, 0, 0, 0, 0, 0, 0, 0, 223, 0, 0, 0, 1811, 0, 1627, 0, 0, 0, 1313, 0, 1549, 0, 0, 0, 1457, 0, 0, 0, 0, 0, 0, 0, 1201, 0, 0, 0, 0, 0, 463, 0, 0, 0, 1571, 0, 0, 0, 0, 0, 0, 0, 2239, 0, 0, 0, 2147, 0, 0, 0, 0, 0, 2069, 0, 961, 0, 0, 0, 0, 0, 313, 0, 0, 0, 0, 0, 487, 0, 0, 0, 293, 0, 1429, 0, 0, 0, 1907, 0, 0, 0, 0, 0, 2249, 0, 0, 0, 0, 0, 0, 0, 1333, 0, 0, 0, 2111, 0, 1717, 0, 0, 0, 1373, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 629, 0, 1261, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1031, 0, 1567, 0, 0, 0, 2063, 0, 619, 0, 0, 0, 557, 0, 0, 0, 0, 0, 1709, 0, 691, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1301, 0, 1597, 0, 0, 0, 2243, 0, 0, 0, 0, 0, 317, 0, 0, 0, 0, 0, 0, 0, 1711, 0, 0, 0, 0, 0, 2053, 0, 0, 0, 1541, 0, 1027, 0, 0, 0, 0, 0, 1369, 0, 0, 0, 1007, 0, 0, 0, 0, 0, 689, 0, 0, 0, 0, 0, 0, 0, 1483, 0, 0, 0, 1391, 0, 0, 0, 0, 0, 1103, 0, 79, 0, 0, 0, 197, 0, 0, 0, 0, 0, 0, 0, 151, 0, 0, 0, 0, 0, 1513, 0, 0, 0, 101, 0, 0, 0, 0, 0, 23, 0, 1819, 0, 0, 0, 1097, 0, 0, 0, 0, 0, 809, 0, 751, 0, 0, 0, 0, 0, 1363, 0, 0, 0, 0, 0, 1747, 0, 0, 0, 2183, 0, 0, 0, 0, 0, 1697, 0, 0, 0, 0, 0, 989, 0, 1651, 0, 0, 0, 0, 0, 283, 0, 0, 0, 0, 0, 457, 0, 0, 0, 2213, 0, 2119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1889, 0, 841, 0, 0, 0, 0, 0, 0, 0, 0, 0, 821, 0, 307, 0, 0, 0, 1643, 0, 1879, 0, 0, 0, 137, 0, 0, 0, 0, 0, 1499, 0, 1531, 0, 0, 0, 0, 0, 1453, 0, 0, 0, 251, 0, 547, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1157, 0, 0, 0, 0, 0, 89, 0, 1291, 0, 0, 0, 0, 0, 1633, 0, 0, 0, 281, 0, 817, 0, 0, 0, 0, 0, 109, 0, 0, 0, 587, 0, 0, 0, 0, 0, 269, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 131, 0, 2047, 0, 0, 0, 53, 0, 1339, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2251, 0, 0, 0, 0, 0, 1093, 0, 0, 0, 1361, 0, 0, 0, 0, 0, 1073, 0, 1609, 0, 0, 0, 887, 0, 0, 0, 0, 0, 389, 0, 2011, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 277, 0, 0, 0, 923, 0, 1849, 0, 0, 0, 647, 0, 0, 0, 0, 0, 0, 0, 391, 0, 0, 0, 0, 0, 73, 0, 0, 0, 221, 0, 37, 0, 0, 0, 1163, 0, 1699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1679, 0, 1891, 0, 0, 0, 0, 0, 0, 0, 0, 0, 401, 0, 0, 0, 0, 0, 1433, 0, 409, 0, 0, 0, 1187, 0, 0, 0, 0, 0, 449, 0, 481, 0, 0, 0, 0, 0, 193, 0, 0, 0, 1091, 0, 337, 0, 0, 0, 353, 0, 0, 0, 0, 0, 107, 0, 0, 0, 0, 0, 1139, 0, 1081, 0, 0, 0, 0, 0, 373, 0, 0, 0, 2171, 0, 2077, 0, 0, 0, 0, 0, 0, 0, 0, 0, 377, 0, 0, 0, 0, 0, 1319, 0, 0, 0, 0, 0, 0, 0, 1273, 0, 0, 0, 0, 0, 787, 0, 0, 0, 893, 0, 1129, 0, 0, 0, 617, 0, 0, 0, 0, 0, 0, 0, 1831, 0, 0, 0, 0, 0, 883, 0, 0, 0, 1151, 0, 0, 0, 0, 0, 653, 0, 559, 0, 0, 0, 467, 0, 0, 0, 0, 0, 179, 0, 541, 0, 0, 0, 0, 0, 1783, 0, 0, 0, 0, 0, 1537, 0, 0, 0, 0, 0, 589, 0, 0, 0, 1487, 0, 0, 0, 0, 0, 1409, 0, 2281, 0, 0, 0, 0, 0, 1963, 0, 0, 0, 1271, 0, 2137, 0, 0, 0, 953, 0, 439, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1259, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1451, 0, 727, 0, 0, 0, 383, 0, 1669, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1289, 0, 271, 0, 0, 0, 0, 0, 2083, 0, 0, 0, 41, 0, 2227, 0, 0, 0, 1403, 0, 0, 0, 0, 0, 2207, 0, 0, 0, 0, 0, 719, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1961, 0, 607, 0, 0, 0, 0, 0, 529, 0, 0, 0, 1637, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1063, 0, 0, 0, 551, 0, 367, 0, 0, 0, 2153, 0, 709, 0, 0, 0, 1667, 0, 0, 0, 0, 0, 0, 0, 571, 0, 0, 0, 0, 0, 2143, 0, 0, 0, 731, 0, 0, 0, 0, 0, 443, 0, 1399, 0, 0, 0, 1517, 0, 0, 0, 0, 0, 1439, 0, 1801, 0, 0, 0, 0, 0, 523, 0, 0, 0, 0, 0, 1327, 0, 0, 0, 1343, 0, 169, 0, 0, 0, 437, 0, 0, 0, 0, 0, 149, 0, 2071, 0, 0, 0, 0, 0, 703, 0, 0, 0, 851, 0, 1087, 0, 0, 0, 533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2309 };
__constant uchar WHL_STEP[480] = { 6, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 3, 1, 5, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 2, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 3, 4, 3, 5, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 2, 6, 1, 3, 2, 1, 3, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 2, 1, 2, 1, 2, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 3, 1, 2, 3, 1, 6, 2, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 5, 3, 4, 3, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 2, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 5, 1, 3, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1 };
#endif
#if WHEEL_W == 30030
#define WHL_N 5760
#define WHL_MISS 0xFFFFu
__constant ushort WHL_IDX[30030] = { 65535, 0, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1, 65535, 2, 65535, 65535, 65535, 3, 65535, 65535, 65535, 65535, 65535, 4, 65535, 5, 65535, 65535, 65535, 65535, 65535, 6, 65535, 65535, 65535, 7, 65535, 8, 65535, 65535, 65535, 9, 65535, 65535, 65535, 65535, 65535, 10, 65535, 65535, 65535, 65535, 65535, 11, 65535, 12, 65535, 65535, 65535, 65535, 65535, 13, 65535, 65535, 65535, 14, 65535, 15, 65535, 65535, 65535, 65535, 65535, 16, 65535, 65535, 65535, 17, 65535, 65535, 65535, 65535, 65535, 18, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 19, 65535, 65535, 65535, 20, 65535, 21, 65535, 65535, 65535, 22, 65535, 23, 65535, 65535, 65535, 24, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 25, 65535, 65535, 65535, 26, 65535, 65535, 65535, 65535, 65535, 27, 65535, 28, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 29, 65535, 30, 65535, 65535, 65535, 65535, 65535, 31, 65535, 65535, 65535, 65535, 65535, 32, 65535, 65535, 65535, 33, 65535, 65535, 65535, 65535, 65535, 34, 65535, 65535, 65535, 65535, 65535, 35, 65535, 36, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 37, 65535, 38, 65535, 65535, 65535, 39, 65535, 40, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 41, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 42, 65535, 65535, 65535, 43, 65535, 44, 65535, 65535, 65535, 45, 65535, 65535, 65535, 65535, 65535, 46, 65535, 47, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 48, 65535, 65535, 65535, 65535, 65535, 49, 65535, 65535, 65535, 65535, 65535, 50, 65535, 65535, 65535, 65535, 65535, 51, 65535, 52, 65535, 65535, 65535, 65535, 65535, 53, 65535, 65535, 65535, 54, 65535, 55, 65535, 65535, 65535, 65535, 65535, 56, 65535, 65535, 65535, 57, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 58, 65535, 65535, 65535, 59, 65535, 60, 65535, 65535, 65535, 61, 65535, 65535, 65535, 65535, 65535, 62, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 63, 65535, 65535, 65535, 65535, 65535, 64, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65, 65535, 66, 65535, 65535, 65535, 67, 65535, 65535, 65535, 65535, 65535, 68, 65535, 69, 65535, 65535, 65535, 65535, 65535, 70, 65535, 65535, 65535, 65535, 65535, 71, 65535, 65535, 65535, 65535, 65535, 72, 65535, 65535, 65535, 73, 65535, 65535, 65535, 65535, 65535, 74, 65535, 75, 65535, 65535, 65535, 65535, 65535, 76, 65535, 65535, 65535, 77, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 78, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 79, 65535, 80, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 81, 65535, 82, 65535, 65535, 65535, 83, 65535, 84, 65535, 65535, 65535, 85, 65535, 65535, 65535, 65535, 65535, 86, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 87, 65535, 65535, 65535, 88, 65535, 89, 65535, 65535, 65535, 90, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 91, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 92, 65535, 65535, 65535, 93, 65535, 94, 65535, 65535, 65535, 65535, 65535, 95, 65535, 65535, 65535, 96, 65535, 65535, 65535, 65535, 65535, 97, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 98, 65535, 99, 65535, 65535, 65535, 100, 65535, 101, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 102, 65535, 65535, 65535, 65535, 65535, 103, 65535, 65535, 65535, 104, 65535, 65535, 65535, 65535, 65535, 105, 65535, 65535, 65535, 65535, 65535, 106, 65535, 65535, 65535, 65535, 65535, 107, 65535, 108, 65535, 65535, 65535, 65535, 65535, 109, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 110, 65535, 111, 65535, 65535, 65535, 112, 65535, 65535, 65535, 65535, 65535, 113, 65535, 114, 65535, 65535, 65535, 65535, 65535, 115, 65535, 65535, 65535, 65535, 65535, 116, 65535, 65535, 65535, 117, 65535, 118, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 119, 65535, 120, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 121, 65535, 122, 65535, 65535, 65535, 123, 65535, 65535, 65535, 65535, 65535, 124, 65535, 65535, 65535, 65535, 65535, 125, 65535, 126, 65535, 65535, 65535, 65535, 65535, 127, 65535, 65535, 65535, 65535, 65535, 128, 65535, 65535, 65535, 129, 65535, 65535, 65535, 65535, 65535, 130, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 131, 65535, 65535, 65535, 65535, 65535, 132, 65535, 65535, 65535, 133, 65535, 134, 65535, 65535, 65535, 65535, 65535, 135, 65535, 65535, 65535, 136, 65535, 65535, 65535, 65535, 65535, 137, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 138, 65535, 65535, 65535, 139, 65535, 140, 65535, 65535, 65535, 65535, 65535, 141, 65535, 65535, 65535, 142, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 143, 65535, 65535, 65535, 65535, 65535, 144, 65535, 65535, 65535, 145, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 146, 65535, 65535, 65535, 147, 65535, 65535, 65535, 65535, 65535, 148, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 149, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 150, 65535, 151, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 152, 65535, 153, 65535, 65535, 65535, 65535, 65535, 154, 65535, 65535, 65535, 155, 65535, 156, 65535, 65535, 65535, 157, 65535, 158, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 159, 65535, 160, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 161, 65535, 162, 65535, 65535, 65535, 163, 65535, 164, 65535, 65535, 65535, 165, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 166, 65535, 65535, 65535, 167, 65535, 168, 65535, 65535, 65535, 169, 65535, 65535, 65535, 65535, 65535, 170, 65535, 65535, 65535, 65535, 65535, 171, 65535, 172, 65535, 65535, 65535, 65535, 65535, 173, 65535, 65535, 65535, 174, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 175, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 176, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 177, 65535, 65535, 65535, 178, 65535, 179, 65535, 65535, 65535, 180, 65535, 65535, 65535, 65535, 65535, 181, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 182, 65535, 65535, 65535, 65535, 65535, 183, 65535, 65535, 65535, 184, 65535, 65535, 65535, 65535, 65535, 185, 65535, 65535, 65535, 65535, 65535, 186, 65535, 65535, 65535, 65535, 65535, 187, 65535, 188, 65535, 65535, 65535, 65535, 65535, 189, 65535, 65535, 65535, 65535, 65535, 190, 65535, 65535, 65535, 191, 65535, 192, 65535, 65535, 65535, 193, 65535, 65535, 65535, 65535, 65535, 194, 65535, 195, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 196, 65535, 197, 65535, 65535, 65535, 198, 65535, 199, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 200, 65535, 201, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 202, 65535, 203, 65535, 65535, 65535, 65535, 65535, 204, 65535, 65535, 65535, 205, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 206, 65535, 65535, 65535, 65535, 65535, 207, 65535, 65535, 65535, 208, 65535, 209, 65535, 65535, 65535, 210, 65535, 65535, 65535, 65535, 65535, 211, 65535, 65535, 65535, 65535, 65535, 212, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 213, 65535, 65535, 65535, 214, 65535, 215, 65535, 65535, 65535, 65535, 65535, 216, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 217, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 218, 65535, 65535, 65535, 219, 65535, 220, 65535, 65535, 65535, 65535, 65535, 221, 65535, 65535, 65535, 222, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 223, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 224, 65535, 65535, 65535, 65535, 65535, 225, 65535, 226, 65535, 65535, 65535, 227, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 228, 65535, 65535, 65535, 65535, 65535, 229, 65535, 65535, 65535, 65535, 65535, 230, 65535, 65535, 65535, 231, 65535, 232, 65535, 65535, 65535, 233, 65535, 65535, 65535, 65535, 65535, 234, 65535, 235, 65535, 65535, 65535, 65535, 65535, 236, 65535, 65535, 65535, 237, 65535, 65535, 65535, 65535, 65535, 238, 65535, 239, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 240, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 241, 65535, 242, 65535, 65535, 65535, 243, 65535, 244, 65535, 65535, 65535, 245, 65535, 65535, 65535, 65535, 65535, 246, 65535, 247, 65535, 65535, 65535, 65535, 65535, 248, 65535, 65535, 65535, 249, 65535, 250, 65535, 65535, 65535, 251, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 252, 65535, 253, 65535, 65535, 65535, 65535, 65535, 254, 65535, 65535, 65535, 65535, 65535, 255, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 256, 65535, 65535, 65535, 65535, 65535, 257, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 258, 65535, 65535, 65535, 259, 65535, 260, 65535, 65535, 65535, 261, 65535, 262, 65535, 65535, 65535, 263, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 264, 65535, 65535, 65535, 65535, 65535, 265, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 266, 65535, 65535, 65535, 267, 65535, 65535, 65535, 65535, 65535, 268, 65535, 269, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 270, 65535, 65535, 65535, 271, 65535, 272, 65535, 65535, 65535, 273, 65535, 65535, 65535, 65535, 65535, 274, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 275, 65535, 65535, 65535, 276, 65535, 277, 65535, 65535, 65535, 278, 65535, 279, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 280, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 281, 65535, 282, 65535, 65535, 65535, 283, 65535, 284, 65535, 65535, 65535, 285, 65535, 65535, 65535, 65535, 65535, 286, 65535, 287, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 288, 65535, 289, 65535, 65535, 65535, 290, 65535, 65535, 65535, 65535, 65535, 291, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 292, 65535, 65535, 65535, 65535, 65535, 293, 65535, 65535, 65535, 294, 65535, 295, 65535, 65535, 65535, 65535, 65535, 296, 65535, 65535, 65535, 297, 65535, 65535, 65535, 65535, 65535, 298, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 299, 65535, 65535, 65535, 300, 65535, 65535, 65535, 65535, 65535, 301, 65535, 302, 65535, 65535, 65535, 303, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 304, 65535, 65535, 65535, 65535, 65535, 305, 65535, 65535, 65535, 306, 65535, 65535, 65535, 65535, 65535, 307, 65535, 308, 65535, 65535, 65535, 309, 65535, 65535, 65535, 65535, 65535, 310, 65535, 311, 65535, 65535, 65535, 65535, 65535, 312, 65535, 65535, 65535, 65535, 65535, 313, 65535, 65535, 65535, 314, 65535, 65535, 65535, 65535, 65535, 315, 65535, 65535, 65535, 65535, 65535, 316, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 317, 65535, 65535, 65535, 65535, 65535, 318, 65535, 65535, 65535, 319, 65535, 320, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 321, 65535, 322, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 323, 65535, 324, 65535, 65535, 65535, 325, 65535, 326, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 327, 65535, 328, 65535, 65535, 65535, 65535, 65535, 329, 65535, 65535, 65535, 330, 65535, 331, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 332, 65535, 65535, 65535, 65535, 65535, 333, 65535, 334, 65535, 65535, 65535, 65535, 65535, 335, 65535, 65535, 65535, 336, 65535, 337, 65535, 65535, 65535, 65535, 65535, 338, 65535, 65535, 65535, 339, 65535, 65535, 65535, 65535, 65535, 340, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 341, 65535, 65535, 65535, 65535, 65535, 342, 65535, 65535, 65535, 343, 65535, 344, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 345, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 346, 65535, 65535, 65535, 65535, 65535, 347, 65535, 348, 65535, 65535, 65535, 349, 65535, 65535, 65535, 65535, 65535, 350, 65535, 351, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 352, 65535, 65535, 65535, 353, 65535, 354, 65535, 65535, 65535, 355, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 356, 65535, 65535, 65535, 65535, 65535, 357, 65535, 65535, 65535, 358, 65535, 359, 65535, 65535, 65535, 360, 65535, 361, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 362, 65535, 363, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 364, 65535, 65535, 65535, 65535, 65535, 365, 65535, 366, 65535, 65535, 65535, 367, 65535, 65535, 65535, 65535, 65535, 368, 65535, 369, 65535, 65535, 65535, 65535, 65535, 370, 65535, 65535, 65535, 371, 65535, 372, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 373, 65535, 65535, 65535, 65535, 65535, 374, 65535, 375, 65535, 65535, 65535, 65535, 65535, 376, 65535, 65535, 65535, 377, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 378, 65535, 65535, 65535, 65535, 65535, 379, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 380, 65535, 65535, 65535, 65535, 65535, 381, 65535, 65535, 65535, 382, 65535, 383, 65535, 65535, 65535, 384, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 385, 65535, 65535, 65535, 65535, 65535, 386, 65535, 65535, 65535, 387, 65535, 65535, 65535, 65535, 65535, 388, 65535, 389, 65535, 65535, 65535, 390, 65535, 65535, 65535, 65535, 65535, 391, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 392, 65535, 65535, 65535, 65535, 65535, 393, 65535, 65535, 65535, 65535, 65535, 394, 65535, 65535, 65535, 395, 65535, 65535, 65535, 65535, 65535, 396, 65535, 397, 65535, 65535, 65535, 65535, 65535, 398, 65535, 65535, 65535, 399, 65535, 400, 65535, 65535, 65535, 401, 65535, 402, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 403, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 404, 65535, 405, 65535, 65535, 65535, 406, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 407, 65535, 408, 65535, 65535, 65535, 65535, 65535, 409, 65535, 65535, 65535, 410, 65535, 411, 65535, 65535, 65535, 412, 65535, 65535, 65535, 65535, 65535, 413, 65535, 65535, 65535, 65535, 65535, 414, 65535, 415, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 416, 65535, 65535, 65535, 65535, 65535, 417, 65535, 65535, 65535, 418, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 419, 65535, 420, 65535, 65535, 65535, 421, 65535, 422, 65535, 65535, 65535, 423, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 424, 65535, 65535, 65535, 65535, 65535, 425, 65535, 65535, 65535, 426, 65535, 65535, 65535, 65535, 65535, 427, 65535, 428, 65535, 65535, 65535, 429, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 430, 65535, 65535, 65535, 65535, 65535, 431, 65535, 65535, 65535, 65535, 65535, 432, 65535, 65535, 65535, 433, 65535, 434, 65535, 65535, 65535, 435, 65535, 65535, 65535, 65535, 65535, 436, 65535, 437, 65535, 65535, 65535, 65535, 65535, 438, 65535, 65535, 65535, 439, 65535, 440, 65535, 65535, 65535, 441, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 442, 65535, 443, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 444, 65535, 65535, 65535, 65535, 65535, 445, 65535, 65535, 65535, 446, 65535, 65535, 65535, 65535, 65535, 447, 65535, 448, 65535, 65535, 65535, 65535, 65535, 449, 65535, 65535, 65535, 450, 65535, 65535, 65535, 65535, 65535, 451, 65535, 65535, 65535, 65535, 65535, 452, 65535, 65535, 65535, 65535, 65535, 453, 65535, 454, 65535, 65535, 65535, 65535, 65535, 455, 65535, 65535, 65535, 456, 65535, 457, 65535, 65535, 65535, 65535, 65535, 458, 65535, 65535, 65535, 459, 65535, 65535, 65535, 65535, 65535, 460, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 461, 65535, 65535, 65535, 462, 65535, 463, 65535, 65535, 65535, 464, 65535, 465, 65535, 65535, 65535, 466, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 467, 65535, 65535, 65535, 468, 65535, 65535, 65535, 65535, 65535, 469, 65535, 470, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 471, 65535, 472, 65535, 65535, 65535, 65535, 65535, 473, 65535, 65535, 65535, 65535, 65535, 474, 65535, 65535, 65535, 475, 65535, 476, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 477, 65535, 478, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 479, 65535, 480, 65535, 65535, 65535, 481, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 482, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 483, 65535, 484, 65535, 65535, 65535, 485, 65535, 486, 65535, 65535, 65535, 487, 65535, 65535, 65535, 65535, 65535, 488, 65535, 489, 65535, 65535, 65535, 65535, 65535, 490, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 491, 65535, 65535, 65535, 65535, 65535, 492, 65535, 65535, 65535, 65535, 65535, 493, 65535, 494, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 495, 65535, 496, 65535, 65535, 65535, 65535, 65535, 497, 65535, 65535, 65535, 498, 65535, 65535, 65535, 65535, 65535, 499, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 500, 65535, 65535, 65535, 501, 65535, 502, 65535, 65535, 65535, 503, 65535, 65535, 65535, 65535, 65535, 504, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 505, 65535, 65535, 65535, 65535, 65535, 506, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 507, 65535, 508, 65535, 65535, 65535, 509, 65535, 65535, 65535, 65535, 65535, 510, 65535, 511, 65535, 65535, 65535, 65535, 65535, 512, 65535, 65535, 65535, 65535, 65535, 513, 65535, 65535, 65535, 514, 65535, 515, 65535, 65535, 65535, 516, 65535, 65535, 65535, 65535, 65535, 517, 65535, 518, 65535, 65535, 65535, 65535, 65535, 519, 65535, 65535, 65535, 520, 65535, 521, 65535, 65535, 65535, 65535, 65535, 522, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 523, 65535, 524, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 525, 65535, 65535, 65535, 65535, 65535, 526, 65535, 527, 65535, 65535, 65535, 528, 65535, 65535, 65535, 65535, 65535, 529, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 530, 65535, 65535, 65535, 531, 65535, 532, 65535, 65535, 65535, 533, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 534, 65535, 535, 65535, 65535, 65535, 65535, 65535, 536, 65535, 65535, 65535, 537, 65535, 538, 65535, 65535, 65535, 65535, 65535, 539, 65535, 65535, 65535, 540, 65535, 65535, 65535, 65535, 65535, 541, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 542, 65535, 543, 65535, 65535, 65535, 544, 65535, 545, 65535, 65535, 65535, 546, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 547, 65535, 65535, 65535, 65535, 65535, 548, 65535, 65535, 65535, 549, 65535, 65535, 65535, 65535, 65535, 550, 65535, 551, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 552, 65535, 553, 65535, 65535, 65535, 65535, 65535, 554, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 555, 65535, 65535, 65535, 65535, 65535, 556, 65535, 65535, 65535, 65535, 65535, 557, 65535, 558, 65535, 65535, 65535, 65535, 65535, 559, 65535, 65535, 65535, 560, 65535, 561, 65535, 65535, 65535, 562, 65535, 563, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 564, 65535, 565, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 566, 65535, 65535, 65535, 567, 65535, 65535, 65535, 65535, 65535, 568, 65535, 65535, 65535, 65535, 65535, 569, 65535, 570, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 571, 65535, 65535, 65535, 572, 65535, 65535, 65535, 65535, 65535, 573, 65535, 65535, 65535, 65535, 65535, 574, 65535, 575, 65535, 65535, 65535, 65535, 65535, 576, 65535, 65535, 65535, 577, 65535, 578, 65535, 65535, 65535, 65535, 65535, 579, 65535, 65535, 65535, 580, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 581, 65535, 65535, 65535, 582, 65535, 583, 65535, 65535, 65535, 65535, 65535, 584, 65535, 65535, 65535, 585, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 586, 65535, 65535, 65535, 65535, 65535, 587, 65535, 65535, 65535, 588, 65535, 65535, 65535, 65535, 65535, 589, 65535, 590, 65535, 65535, 65535, 591, 65535, 65535, 65535, 65535, 65535, 592, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 593, 65535, 65535, 65535, 65535, 65535, 594, 65535, 65535, 65535, 65535, 65535, 595, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 596, 65535, 597, 65535, 65535, 65535, 65535, 65535, 598, 65535, 65535, 65535, 599, 65535, 65535, 65535, 65535, 65535, 600, 65535, 601, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 602, 65535, 603, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 604, 65535, 605, 65535, 65535, 65535, 606, 65535, 607, 65535, 65535, 65535, 608, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 609, 65535, 65535, 65535, 65535, 65535, 610, 65535, 65535, 65535, 611, 65535, 612, 65535, 65535, 65535, 613, 65535, 65535, 65535, 65535, 65535, 614, 65535, 65535, 65535, 65535, 65535, 615, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 616, 65535, 65535, 65535, 617, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 618, 65535, 65535, 65535, 619, 65535, 65535, 65535, 65535, 65535, 620, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 621, 65535, 65535, 65535, 622, 65535, 623, 65535, 65535, 65535, 624, 65535, 625, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 626, 65535, 65535, 65535, 65535, 65535, 627, 65535, 65535, 65535, 628, 65535, 65535, 65535, 65535, 65535, 629, 65535, 65535, 65535, 65535, 65535, 630, 65535, 65535, 65535, 65535, 65535, 631, 65535, 632, 65535, 65535, 65535, 65535, 65535, 633, 65535, 65535, 65535, 65535, 65535, 634, 65535, 65535, 65535, 635, 65535, 636, 65535, 65535, 65535, 637, 65535, 65535, 65535, 65535, 65535, 638, 65535, 639, 65535, 65535, 65535, 65535, 65535, 640, 65535, 65535, 65535, 65535, 65535, 641, 65535, 65535, 65535, 642, 65535, 643, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 644, 65535, 645, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 646, 65535, 647, 65535, 65535, 65535, 65535, 65535, 648, 65535, 65535, 65535, 649, 65535, 65535, 65535, 65535, 65535, 650, 65535, 651, 65535, 65535, 65535, 65535, 65535, 652, 65535, 65535, 65535, 653, 65535, 654, 65535, 65535, 65535, 655, 65535, 65535, 65535, 65535, 65535, 656, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 657, 65535, 65535, 65535, 658, 65535, 659, 65535, 65535, 65535, 65535, 65535, 660, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 661, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 662, 65535, 65535, 65535, 663, 65535, 664, 65535, 65535, 65535, 665, 65535, 666, 65535, 65535, 65535, 667, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 668, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 669, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 670, 65535, 65535, 65535, 671, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 672, 65535, 65535, 65535, 65535, 65535, 673, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 674, 65535, 675, 65535, 65535, 65535, 676, 65535, 65535, 65535, 65535, 65535, 677, 65535, 678, 65535, 65535, 65535, 65535, 65535, 679, 65535, 65535, 65535, 680, 65535, 65535, 65535, 65535, 65535, 681, 65535, 682, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 683, 65535, 684, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 685, 65535, 686, 65535, 65535, 65535, 687, 65535, 688, 65535, 65535, 65535, 689, 65535, 65535, 65535, 65535, 65535, 690, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 691, 65535, 65535, 65535, 692, 65535, 693, 65535, 65535, 65535, 694, 65535, 65535, 65535, 65535, 65535, 695, 65535, 65535, 65535, 65535, 65535, 696, 65535, 697, 65535, 65535, 65535, 65535, 65535, 698, 65535, 65535, 65535, 65535, 65535, 699, 65535, 65535, 65535, 65535, 65535, 700, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 701, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 702, 65535, 65535, 65535, 703, 65535, 704, 65535, 65535, 65535, 705, 65535, 65535, 65535, 65535, 65535, 706, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 707, 65535, 65535, 65535, 65535, 65535, 708, 65535, 65535, 65535, 709, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 710, 65535, 65535, 65535, 711, 65535, 65535, 65535, 65535, 65535, 712, 65535, 713, 65535, 65535, 65535, 65535, 65535, 714, 65535, 65535, 65535, 65535, 65535, 715, 65535, 65535, 65535, 716, 65535, 717, 65535, 65535, 65535, 718, 65535, 65535, 65535, 65535, 65535, 719, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 720, 65535, 721, 65535, 65535, 65535, 722, 65535, 723, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 724, 65535, 725, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 726, 65535, 727, 65535, 65535, 65535, 728, 65535, 729, 65535, 65535, 65535, 730, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 731, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 732, 65535, 733, 65535, 65535, 65535, 734, 65535, 65535, 65535, 65535, 65535, 735, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 736, 65535, 65535, 65535, 65535, 65535, 737, 65535, 65535, 65535, 738, 65535, 739, 65535, 65535, 65535, 65535, 65535, 740, 65535, 65535, 65535, 741, 65535, 65535, 65535, 65535, 65535, 742, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 743, 65535, 65535, 65535, 744, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 745, 65535, 65535, 65535, 746, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 747, 65535, 65535, 65535, 65535, 65535, 748, 65535, 65535, 65535, 749, 65535, 65535, 65535, 65535, 65535, 750, 65535, 751, 65535, 65535, 65535, 752, 65535, 65535, 65535, 65535, 65535, 753, 65535, 754, 65535, 65535, 65535, 65535, 65535, 755, 65535, 65535, 65535, 65535, 65535, 756, 65535, 65535, 65535, 757, 65535, 65535, 65535, 65535, 65535, 758, 65535, 65535, 65535, 65535, 65535, 759, 65535, 760, 65535, 65535, 65535, 65535, 65535, 761, 65535, 65535, 65535, 65535, 65535, 762, 65535, 65535, 65535, 763, 65535, 764, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 765, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 766, 65535, 767, 65535, 65535, 65535, 768, 65535, 769, 65535, 65535, 65535, 770, 65535, 65535, 65535, 65535, 65535, 771, 65535, 772, 65535, 65535, 65535, 65535, 65535, 773, 65535, 65535, 65535, 774, 65535, 775, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 776, 65535, 777, 65535, 65535, 65535, 65535, 65535, 778, 65535, 65535, 65535, 779, 65535, 780, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 781, 65535, 65535, 65535, 65535, 65535, 782, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 783, 65535, 65535, 65535, 784, 65535, 785, 65535, 65535, 65535, 786, 65535, 787, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 788, 65535, 65535, 65535, 65535, 65535, 789, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 790, 65535, 791, 65535, 65535, 65535, 792, 65535, 65535, 65535, 65535, 65535, 793, 65535, 794, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 795, 65535, 65535, 65535, 796, 65535, 797, 65535, 65535, 65535, 798, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 799, 65535, 65535, 65535, 65535, 65535, 800, 65535, 65535, 65535, 801, 65535, 802, 65535, 65535, 65535, 803, 65535, 804, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 805, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 806, 65535, 65535, 65535, 65535, 65535, 807, 65535, 808, 65535, 65535, 65535, 809, 65535, 65535, 65535, 65535, 65535, 810, 65535, 811, 65535, 65535, 65535, 65535, 65535, 812, 65535, 65535, 65535, 813, 65535, 814, 65535, 65535, 65535, 815, 65535, 65535, 65535, 65535, 65535, 816, 65535, 65535, 65535, 65535, 65535, 817, 65535, 818, 65535, 65535, 65535, 65535, 65535, 819, 65535, 65535, 65535, 820, 65535, 821, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 822, 65535, 65535, 65535, 65535, 65535, 823, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 824, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 825, 65535, 826, 65535, 65535, 65535, 827, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 828, 65535, 65535, 65535, 65535, 65535, 829, 65535, 65535, 65535, 830, 65535, 65535, 65535, 65535, 65535, 831, 65535, 832, 65535, 65535, 65535, 833, 65535, 65535, 65535, 65535, 65535, 834, 65535, 835, 65535, 65535, 65535, 65535, 65535, 836, 65535, 65535, 65535, 65535, 65535, 837, 65535, 65535, 65535, 65535, 65535, 838, 65535, 65535, 65535, 839, 65535, 65535, 65535, 65535, 65535, 840, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 841, 65535, 65535, 65535, 842, 65535, 843, 65535, 65535, 65535, 844, 65535, 845, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 846, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 847, 65535, 848, 65535, 65535, 65535, 849, 65535, 850, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 851, 65535, 852, 65535, 65535, 65535, 65535, 65535, 853, 65535, 65535, 65535, 854, 65535, 855, 65535, 65535, 65535, 856, 65535, 65535, 65535, 65535, 65535, 857, 65535, 65535, 65535, 65535, 65535, 858, 65535, 859, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 860, 65535, 861, 65535, 65535, 65535, 65535, 65535, 862, 65535, 65535, 65535, 863, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 864, 65535, 65535, 65535, 65535, 65535, 865, 65535, 65535, 65535, 866, 65535, 867, 65535, 65535, 65535, 868, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 869, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 870, 65535, 65535, 65535, 65535, 65535, 871, 65535, 872, 65535, 65535, 65535, 873, 65535, 65535, 65535, 65535, 65535, 874, 65535, 875, 65535, 65535, 65535, 65535, 65535, 876, 65535, 65535, 65535, 65535, 65535, 877, 65535, 65535, 65535, 878, 65535, 879, 65535, 65535, 65535, 880, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 881, 65535, 65535, 65535, 65535, 65535, 882, 65535, 65535, 65535, 883, 65535, 884, 65535, 65535, 65535, 885, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 886, 65535, 887, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 888, 65535, 65535, 65535, 889, 65535, 890, 65535, 65535, 65535, 891, 65535, 65535, 65535, 65535, 65535, 892, 65535, 893, 65535, 65535, 65535, 65535, 65535, 894, 65535, 65535, 65535, 895, 65535, 896, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 897, 65535, 65535, 65535, 65535, 65535, 898, 65535, 899, 65535, 65535, 65535, 65535, 65535, 900, 65535, 65535, 65535, 901, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 902, 65535, 65535, 65535, 903, 65535, 65535, 65535, 65535, 65535, 904, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 905, 65535, 65535, 65535, 906, 65535, 907, 65535, 65535, 65535, 908, 65535, 909, 65535, 65535, 65535, 910, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 911, 65535, 65535, 65535, 912, 65535, 65535, 65535, 65535, 65535, 913, 65535, 914, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 915, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 916, 65535, 65535, 65535, 65535, 65535, 917, 65535, 65535, 65535, 918, 65535, 919, 65535, 65535, 65535, 920, 65535, 65535, 65535, 65535, 65535, 921, 65535, 922, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 923, 65535, 924, 65535, 65535, 65535, 925, 65535, 926, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 927, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 928, 65535, 929, 65535, 65535, 65535, 930, 65535, 65535, 65535, 65535, 65535, 931, 65535, 65535, 65535, 65535, 65535, 932, 65535, 933, 65535, 65535, 65535, 65535, 65535, 934, 65535, 65535, 65535, 935, 65535, 65535, 65535, 65535, 65535, 936, 65535, 65535, 65535, 65535, 65535, 937, 65535, 65535, 65535, 65535, 65535, 938, 65535, 939, 65535, 65535, 65535, 65535, 65535, 940, 65535, 65535, 65535, 65535, 65535, 941, 65535, 65535, 65535, 65535, 65535, 942, 65535, 65535, 65535, 943, 65535, 65535, 65535, 65535, 65535, 944, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 945, 65535, 946, 65535, 65535, 65535, 947, 65535, 65535, 65535, 65535, 65535, 948, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 949, 65535, 65535, 65535, 65535, 65535, 950, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 951, 65535, 952, 65535, 65535, 65535, 953, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 954, 65535, 65535, 65535, 65535, 65535, 955, 65535, 65535, 65535, 65535, 65535, 956, 65535, 65535, 65535, 957, 65535, 958, 65535, 65535, 65535, 959, 65535, 65535, 65535, 65535, 65535, 960, 65535, 961, 65535, 65535, 65535, 65535, 65535, 962, 65535, 65535, 65535, 963, 65535, 964, 65535, 65535, 65535, 65535, 65535, 965, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 966, 65535, 967, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 968, 65535, 969, 65535, 65535, 65535, 65535, 65535, 970, 65535, 65535, 65535, 971, 65535, 65535, 65535, 65535, 65535, 972, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 973, 65535, 65535, 65535, 974, 65535, 65535, 65535, 65535, 65535, 975, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 976, 65535, 977, 65535, 65535, 65535, 65535, 65535, 978, 65535, 65535, 65535, 979, 65535, 980, 65535, 65535, 65535, 65535, 65535, 981, 65535, 65535, 65535, 982, 65535, 65535, 65535, 65535, 65535, 983, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 984, 65535, 985, 65535, 65535, 65535, 986, 65535, 987, 65535, 65535, 65535, 988, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 989, 65535, 65535, 65535, 990, 65535, 65535, 65535, 65535, 65535, 991, 65535, 992, 65535, 65535, 65535, 993, 65535, 65535, 65535, 65535, 65535, 994, 65535, 995, 65535, 65535, 65535, 65535, 65535, 996, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 997, 65535, 998, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 999, 65535, 1000, 65535, 65535, 65535, 65535, 65535, 1001, 65535, 65535, 65535, 1002, 65535, 1003, 65535, 65535, 65535, 1004, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1005, 65535, 1006, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1007, 65535, 1008, 65535, 65535, 65535, 1009, 65535, 65535, 65535, 65535, 65535, 1010, 65535, 65535, 65535, 65535, 65535, 1011, 65535, 1012, 65535, 65535, 65535, 65535, 65535, 1013, 65535, 65535, 65535, 65535, 65535, 1014, 65535, 65535, 65535, 1015, 65535, 65535, 65535, 65535, 65535, 1016, 65535, 65535, 65535, 65535, 65535, 1017, 65535, 1018, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1019, 65535, 1020, 65535, 65535, 65535, 65535, 65535, 1021, 65535, 65535, 65535, 1022, 65535, 65535, 65535, 65535, 65535, 1023, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1024, 65535, 65535, 65535, 1025, 65535, 1026, 65535, 65535, 65535, 65535, 65535, 1027, 65535, 65535, 65535, 1028, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1029, 65535, 65535, 65535, 65535, 65535, 1030, 65535, 65535, 65535, 1031, 65535, 65535, 65535, 65535, 65535, 1032, 65535, 1033, 65535, 65535, 65535, 1034, 65535, 65535, 65535, 65535, 65535, 1035, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1036, 65535, 65535, 65535, 65535, 65535, 1037, 65535, 65535, 65535, 1038, 65535, 1039, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1040, 65535, 1041, 65535, 65535, 65535, 65535, 65535, 1042, 65535, 65535, 65535, 1043, 65535, 1044, 65535, 65535, 65535, 65535, 65535, 1045, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1046, 65535, 1047, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1048, 65535, 65535, 65535, 65535, 65535, 1049, 65535, 1050, 65535, 65535, 65535, 1051, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1052, 65535, 65535, 65535, 65535, 65535, 1053, 65535, 65535, 65535, 1054, 65535, 1055, 65535, 65535, 65535, 1056, 65535, 65535, 65535, 65535, 65535, 1057, 65535, 65535, 65535, 65535, 65535, 1058, 65535, 1059, 65535, 65535, 65535, 65535, 65535, 1060, 65535, 65535, 65535, 1061, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1062, 65535, 65535, 65535, 1063, 65535, 65535, 65535, 65535, 65535, 1064, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1065, 65535, 65535, 65535, 1066, 65535, 1067, 65535, 65535, 65535, 1068, 65535, 1069, 65535, 65535, 65535, 1070, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1071, 65535, 65535, 65535, 65535, 65535, 1072, 65535, 65535, 65535, 1073, 65535, 65535, 65535, 65535, 65535, 1074, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1075, 65535, 1076, 65535, 65535, 65535, 65535, 65535, 1077, 65535, 65535, 65535, 65535, 65535, 1078, 65535, 65535, 65535, 1079, 65535, 65535, 65535, 65535, 65535, 1080, 65535, 65535, 65535, 65535, 65535, 1081, 65535, 1082, 65535, 65535, 65535, 65535, 65535, 1083, 65535, 65535, 65535, 1084, 65535, 1085, 65535, 65535, 65535, 1086, 65535, 1087, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1088, 65535, 1089, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1090, 65535, 65535, 65535, 65535, 65535, 1091, 65535, 65535, 65535, 1092, 65535, 65535, 65535, 65535, 65535, 1093, 65535, 1094, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1095, 65535, 1096, 65535, 65535, 65535, 1097, 65535, 65535, 65535, 65535, 65535, 1098, 65535, 65535, 65535, 65535, 65535, 1099, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1100, 65535, 65535, 65535, 1101, 65535, 1102, 65535, 65535, 65535, 65535, 65535, 1103, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1104, 65535, 65535, 65535, 1105, 65535, 1106, 65535, 65535, 65535, 1107, 65535, 1108, 65535, 65535, 65535, 1109, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1110, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1111, 65535, 65535, 65535, 65535, 65535, 1112, 65535, 1113, 65535, 65535, 65535, 1114, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1115, 65535, 65535, 65535, 65535, 65535, 1116, 65535, 65535, 65535, 65535, 65535, 1117, 65535, 65535, 65535, 65535, 65535, 1118, 65535, 65535, 65535, 1119, 65535, 65535, 65535, 65535, 65535, 1120, 65535, 1121, 65535, 65535, 65535, 65535, 65535, 1122, 65535, 65535, 65535, 1123, 65535, 65535, 65535, 65535, 65535, 1124, 65535, 1125, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1126, 65535, 1127, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1128, 65535, 1129, 65535, 65535, 65535, 1130, 65535, 1131, 65535, 65535, 65535, 1132, 65535, 65535, 65535, 65535, 65535, 1133, 65535, 1134, 65535, 65535, 65535, 65535, 65535, 1135, 65535, 65535, 65535, 1136, 65535, 1137, 65535, 65535, 65535, 1138, 65535, 65535, 65535, 65535, 65535, 1139, 65535, 65535, 65535, 65535, 65535, 1140, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1141, 65535, 65535, 65535, 65535, 65535, 1142, 65535, 65535, 65535, 65535, 65535, 1143, 65535, 65535, 65535, 1144, 65535, 65535, 65535, 65535, 65535, 1145, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1146, 65535, 65535, 65535, 1147, 65535, 1148, 65535, 65535, 65535, 1149, 65535, 1150, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1151, 65535, 65535, 65535, 65535, 65535, 1152, 65535, 65535, 65535, 1153, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1154, 65535, 65535, 65535, 65535, 65535, 1155, 65535, 1156, 65535, 65535, 65535, 65535, 65535, 1157, 65535, 65535, 65535, 65535, 65535, 1158, 65535, 65535, 65535, 1159, 65535, 1160, 65535, 65535, 65535, 1161, 65535, 65535, 65535, 65535, 65535, 1162, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1163, 65535, 65535, 65535, 65535, 65535, 1164, 65535, 65535, 65535, 1165, 65535, 1166, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1167, 65535, 1168, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1169, 65535, 1170, 65535, 65535, 65535, 1171, 65535, 1172, 65535, 65535, 65535, 1173, 65535, 65535, 65535, 65535, 65535, 1174, 65535, 1175, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1176, 65535, 1177, 65535, 65535, 65535, 1178, 65535, 65535, 65535, 65535, 65535, 1179, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1180, 65535, 65535, 65535, 65535, 65535, 1181, 65535, 65535, 65535, 1182, 65535, 1183, 65535, 65535, 65535, 65535, 65535, 1184, 65535, 65535, 65535, 1185, 65535, 65535, 65535, 65535, 65535, 1186, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1187, 65535, 65535, 65535, 1188, 65535, 65535, 65535, 65535, 65535, 1189, 65535, 1190, 65535, 65535, 65535, 1191, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1192, 65535, 65535, 65535, 65535, 65535, 1193, 65535, 65535, 65535, 1194, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1195, 65535, 65535, 65535, 1196, 65535, 65535, 65535, 65535, 65535, 1197, 65535, 1198, 65535, 65535, 65535, 65535, 65535, 1199, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1200, 65535, 65535, 65535, 65535, 65535, 1201, 65535, 65535, 65535, 65535, 65535, 1202, 65535, 1203, 65535, 65535, 65535, 65535, 65535, 1204, 65535, 65535, 65535, 65535, 65535, 1205, 65535, 65535, 65535, 1206, 65535, 1207, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1208, 65535, 1209, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1210, 65535, 1211, 65535, 65535, 65535, 1212, 65535, 1213, 65535, 65535, 65535, 1214, 65535, 65535, 65535, 65535, 65535, 1215, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1216, 65535, 65535, 65535, 1217, 65535, 1218, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1219, 65535, 65535, 65535, 65535, 65535, 1220, 65535, 1221, 65535, 65535, 65535, 65535, 65535, 1222, 65535, 65535, 65535, 1223, 65535, 1224, 65535, 65535, 65535, 65535, 65535, 1225, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1226, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1227, 65535, 65535, 65535, 1228, 65535, 1229, 65535, 65535, 65535, 1230, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1231, 65535, 65535, 65535, 65535, 65535, 1232, 65535, 65535, 65535, 1233, 65535, 65535, 65535, 65535, 65535, 1234, 65535, 1235, 65535, 65535, 65535, 1236, 65535, 65535, 65535, 65535, 65535, 1237, 65535, 1238, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1239, 65535, 65535, 65535, 1240, 65535, 1241, 65535, 65535, 65535, 1242, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1243, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1244, 65535, 1245, 65535, 65535, 65535, 1246, 65535, 1247, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1248, 65535, 1249, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1250, 65535, 65535, 65535, 65535, 65535, 1251, 65535, 1252, 65535, 65535, 65535, 1253, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1254, 65535, 65535, 65535, 65535, 65535, 1255, 65535, 65535, 65535, 1256, 65535, 1257, 65535, 65535, 65535, 1258, 65535, 65535, 65535, 65535, 65535, 1259, 65535, 65535, 65535, 65535, 65535, 1260, 65535, 1261, 65535, 65535, 65535, 65535, 65535, 1262, 65535, 65535, 65535, 1263, 65535, 1264, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1265, 65535, 65535, 65535, 65535, 65535, 1266, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1267, 65535, 65535, 65535, 65535, 65535, 1268, 65535, 65535, 65535, 65535, 65535, 1269, 65535, 65535, 65535, 1270, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1271, 65535, 65535, 65535, 65535, 65535, 1272, 65535, 65535, 65535, 1273, 65535, 65535, 65535, 65535, 65535, 1274, 65535, 1275, 65535, 65535, 65535, 1276, 65535, 65535, 65535, 65535, 65535, 1277, 65535, 1278, 65535, 65535, 65535, 65535, 65535, 1279, 65535, 65535, 65535, 65535, 65535, 1280, 65535, 65535, 65535, 65535, 65535, 1281, 65535, 65535, 65535, 1282, 65535, 65535, 65535, 65535, 65535, 1283, 65535, 1284, 65535, 65535, 65535, 65535, 65535, 1285, 65535, 65535, 65535, 1286, 65535, 1287, 65535, 65535, 65535, 1288, 65535, 1289, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1290, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1291, 65535, 1292, 65535, 65535, 65535, 1293, 65535, 1294, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1295, 65535, 1296, 65535, 65535, 65535, 65535, 65535, 1297, 65535, 65535, 65535, 1298, 65535, 1299, 65535, 65535, 65535, 1300, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1301, 65535, 1302, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1303, 65535, 1304, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1305, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1306, 65535, 65535, 65535, 1307, 65535, 1308, 65535, 65535, 65535, 1309, 65535, 1310, 65535, 65535, 65535, 1311, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1312, 65535, 65535, 65535, 65535, 65535, 1313, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1314, 65535, 1315, 65535, 65535, 65535, 1316, 65535, 65535, 65535, 65535, 65535, 1317, 65535, 1318, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1319, 65535, 65535, 65535, 1320, 65535, 1321, 65535, 65535, 65535, 1322, 65535, 65535, 65535, 65535, 65535, 1323, 65535, 1324, 65535, 65535, 65535, 65535, 65535, 1325, 65535, 65535, 65535, 1326, 65535, 1327, 65535, 65535, 65535, 1328, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1329, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1330, 65535, 65535, 65535, 1331, 65535, 1332, 65535, 65535, 65535, 1333, 65535, 65535, 65535, 65535, 65535, 1334, 65535, 1335, 65535, 65535, 65535, 65535, 65535, 1336, 65535, 65535, 65535, 1337, 65535, 1338, 65535, 65535, 65535, 1339, 65535, 65535, 65535, 65535, 65535, 1340, 65535, 65535, 65535, 65535, 65535, 1341, 65535, 1342, 65535, 65535, 65535, 65535, 65535, 1343, 65535, 65535, 65535, 1344, 65535, 1345, 65535, 65535, 65535, 65535, 65535, 1346, 65535, 65535, 65535, 1347, 65535, 65535, 65535, 65535, 65535, 1348, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1349, 65535, 65535, 65535, 1350, 65535, 65535, 65535, 65535, 65535, 1351, 65535, 1352, 65535, 65535, 65535, 1353, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1354, 65535, 65535, 65535, 1355, 65535, 65535, 65535, 65535, 65535, 1356, 65535, 1357, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1358, 65535, 1359, 65535, 65535, 65535, 65535, 65535, 1360, 65535, 65535, 65535, 65535, 65535, 1361, 65535, 65535, 65535, 1362, 65535, 1363, 65535, 65535, 65535, 1364, 65535, 65535, 65535, 65535, 65535, 1365, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1366, 65535, 1367, 65535, 65535, 65535, 1368, 65535, 1369, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1370, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1371, 65535, 1372, 65535, 65535, 65535, 1373, 65535, 1374, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1375, 65535, 1376, 65535, 65535, 65535, 65535, 65535, 1377, 65535, 65535, 65535, 1378, 65535, 65535, 65535, 65535, 65535, 1379, 65535, 65535, 65535, 65535, 65535, 1380, 65535, 65535, 65535, 65535, 65535, 1381, 65535, 1382, 65535, 65535, 65535, 65535, 65535, 1383, 65535, 65535, 65535, 1384, 65535, 1385, 65535, 65535, 65535, 65535, 65535, 1386, 65535, 65535, 65535, 1387, 65535, 65535, 65535, 65535, 65535, 1388, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1389, 65535, 65535, 65535, 65535, 65535, 1390, 65535, 65535, 65535, 1391, 65535, 65535, 65535, 65535, 65535, 1392, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1393, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1394, 65535, 1395, 65535, 65535, 65535, 1396, 65535, 65535, 65535, 65535, 65535, 1397, 65535, 1398, 65535, 65535, 65535, 65535, 65535, 1399, 65535, 65535, 65535, 65535, 65535, 1400, 65535, 65535, 65535, 1401, 65535, 1402, 65535, 65535, 65535, 1403, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1404, 65535, 65535, 65535, 65535, 65535, 1405, 65535, 65535, 65535, 1406, 65535, 1407, 65535, 65535, 65535, 65535, 65535, 1408, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1409, 65535, 1410, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1411, 65535, 1412, 65535, 65535, 65535, 1413, 65535, 1414, 65535, 65535, 65535, 1415, 65535, 65535, 65535, 65535, 65535, 1416, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1417, 65535, 65535, 65535, 1418, 65535, 1419, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1420, 65535, 1421, 65535, 65535, 65535, 65535, 65535, 1422, 65535, 65535, 65535, 1423, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1424, 65535, 65535, 65535, 1425, 65535, 65535, 65535, 65535, 65535, 1426, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1427, 65535, 1428, 65535, 65535, 65535, 1429, 65535, 1430, 65535, 65535, 65535, 1431, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1432, 65535, 65535, 65535, 65535, 65535, 1433, 65535, 65535, 65535, 1434, 65535, 65535, 65535, 65535, 65535, 1435, 65535, 1436, 65535, 65535, 65535, 1437, 65535, 65535, 65535, 65535, 65535, 1438, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1439, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1440, 65535, 1441, 65535, 65535, 65535, 1442, 65535, 65535, 65535, 65535, 65535, 1443, 65535, 1444, 65535, 65535, 65535, 65535, 65535, 1445, 65535, 65535, 65535, 1446, 65535, 1447, 65535, 65535, 65535, 1448, 65535, 1449, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1450, 65535, 1451, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1452, 65535, 1453, 65535, 65535, 65535, 1454, 65535, 65535, 65535, 65535, 65535, 1455, 65535, 65535, 65535, 65535, 65535, 1456, 65535, 1457, 65535, 65535, 65535, 65535, 65535, 1458, 65535, 65535, 65535, 65535, 65535, 1459, 65535, 65535, 65535, 1460, 65535, 65535, 65535, 65535, 65535, 1461, 65535, 65535, 65535, 65535, 65535, 1462, 65535, 1463, 65535, 65535, 65535, 65535, 65535, 1464, 65535, 65535, 65535, 65535, 65535, 1465, 65535, 65535, 65535, 65535, 65535, 1466, 65535, 65535, 65535, 1467, 65535, 65535, 65535, 65535, 65535, 1468, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1469, 65535, 1470, 65535, 65535, 65535, 65535, 65535, 1471, 65535, 65535, 65535, 1472, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1473, 65535, 65535, 65535, 65535, 65535, 1474, 65535, 65535, 65535, 1475, 65535, 65535, 65535, 65535, 65535, 1476, 65535, 1477, 65535, 65535, 65535, 1478, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1479, 65535, 65535, 65535, 65535, 65535, 1480, 65535, 65535, 65535, 1481, 65535, 1482, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1483, 65535, 1484, 65535, 65535, 65535, 65535, 65535, 1485, 65535, 65535, 65535, 1486, 65535, 1487, 65535, 65535, 65535, 1488, 65535, 1489, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1490, 65535, 1491, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1492, 65535, 1493, 65535, 65535, 65535, 65535, 65535, 1494, 65535, 65535, 65535, 1495, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1496, 65535, 65535, 65535, 65535, 65535, 1497, 65535, 65535, 65535, 1498, 65535, 65535, 65535, 65535, 65535, 1499, 65535, 65535, 65535, 65535, 65535, 1500, 65535, 65535, 65535, 65535, 65535, 1501, 65535, 1502, 65535, 65535, 65535, 65535, 65535, 1503, 65535, 65535, 65535, 1504, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1505, 65535, 65535, 65535, 1506, 65535, 65535, 65535, 65535, 65535, 1507, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1508, 65535, 65535, 65535, 1509, 65535, 1510, 65535, 65535, 65535, 1511, 65535, 1512, 65535, 65535, 65535, 1513, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1514, 65535, 65535, 65535, 1515, 65535, 65535, 65535, 65535, 65535, 1516, 65535, 65535, 65535, 65535, 65535, 1517, 65535, 65535, 65535, 65535, 65535, 1518, 65535, 1519, 65535, 65535, 65535, 65535, 65535, 1520, 65535, 65535, 65535, 65535, 65535, 1521, 65535, 65535, 65535, 1522, 65535, 1523, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1524, 65535, 1525, 65535, 65535, 65535, 65535, 65535, 1526, 65535, 65535, 65535, 1527, 65535, 1528, 65535, 65535, 65535, 1529, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1530, 65535, 1531, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1532, 65535, 1533, 65535, 65535, 65535, 65535, 65535, 1534, 65535, 65535, 65535, 1535, 65535, 65535, 65535, 65535, 65535, 1536, 65535, 1537, 65535, 65535, 65535, 65535, 65535, 1538, 65535, 65535, 65535, 65535, 65535, 1539, 65535, 65535, 65535, 1540, 65535, 65535, 65535, 65535, 65535, 1541, 65535, 65535, 65535, 65535, 65535, 1542, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1543, 65535, 1544, 65535, 65535, 65535, 65535, 65535, 1545, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1546, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1547, 65535, 65535, 65535, 1548, 65535, 1549, 65535, 65535, 65535, 1550, 65535, 1551, 65535, 65535, 65535, 1552, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1553, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1554, 65535, 65535, 65535, 65535, 65535, 1555, 65535, 1556, 65535, 65535, 65535, 1557, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1558, 65535, 65535, 65535, 65535, 65535, 1559, 65535, 65535, 65535, 65535, 65535, 1560, 65535, 65535, 65535, 1561, 65535, 1562, 65535, 65535, 65535, 1563, 65535, 65535, 65535, 65535, 65535, 1564, 65535, 1565, 65535, 65535, 65535, 65535, 65535, 1566, 65535, 65535, 65535, 1567, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1568, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1569, 65535, 1570, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1571, 65535, 65535, 65535, 65535, 65535, 1572, 65535, 1573, 65535, 65535, 65535, 1574, 65535, 65535, 65535, 65535, 65535, 1575, 65535, 1576, 65535, 65535, 65535, 65535, 65535, 1577, 65535, 65535, 65535, 1578, 65535, 1579, 65535, 65535, 65535, 1580, 65535, 65535, 65535, 65535, 65535, 1581, 65535, 65535, 65535, 65535, 65535, 1582, 65535, 1583, 65535, 65535, 65535, 65535, 65535, 1584, 65535, 65535, 65535, 65535, 65535, 1585, 65535, 65535, 65535, 65535, 65535, 1586, 65535, 65535, 65535, 1587, 65535, 65535, 65535, 65535, 65535, 1588, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1589, 65535, 65535, 65535, 1590, 65535, 1591, 65535, 65535, 65535, 1592, 65535, 1593, 65535, 65535, 65535, 1594, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1595, 65535, 65535, 65535, 65535, 65535, 1596, 65535, 65535, 65535, 1597, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1598, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1599, 65535, 1600, 65535, 65535, 65535, 65535, 65535, 1601, 65535, 65535, 65535, 65535, 65535, 1602, 65535, 65535, 65535, 1603, 65535, 65535, 65535, 65535, 65535, 1604, 65535, 65535, 65535, 65535, 65535, 1605, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1606, 65535, 65535, 65535, 1607, 65535, 1608, 65535, 65535, 65535, 1609, 65535, 1610, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1611, 65535, 1612, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1613, 65535, 65535, 65535, 1614, 65535, 1615, 65535, 65535, 65535, 1616, 65535, 65535, 65535, 65535, 65535, 1617, 65535, 1618, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1619, 65535, 1620, 65535, 65535, 65535, 1621, 65535, 65535, 65535, 65535, 65535, 1622, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1623, 65535, 65535, 65535, 65535, 65535, 1624, 65535, 65535, 65535, 1625, 65535, 1626, 65535, 65535, 65535, 65535, 65535, 1627, 65535, 65535, 65535, 1628, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1629, 65535, 65535, 65535, 1630, 65535, 65535, 65535, 65535, 65535, 1631, 65535, 1632, 65535, 65535, 65535, 1633, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1634, 65535, 65535, 65535, 65535, 65535, 1635, 65535, 65535, 65535, 1636, 65535, 65535, 65535, 65535, 65535, 1637, 65535, 1638, 65535, 65535, 65535, 1639, 65535, 65535, 65535, 65535, 65535, 1640, 65535, 1641, 65535, 65535, 65535, 65535, 65535, 1642, 65535, 65535, 65535, 65535, 65535, 1643, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1644, 65535, 65535, 65535, 65535, 65535, 1645, 65535, 1646, 65535, 65535, 65535, 65535, 65535, 1647, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1648, 65535, 1649, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1650, 65535, 1651, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1652, 65535, 1653, 65535, 65535, 65535, 1654, 65535, 1655, 65535, 65535, 65535, 1656, 65535, 65535, 65535, 65535, 65535, 1657, 65535, 1658, 65535, 65535, 65535, 65535, 65535, 1659, 65535, 65535, 65535, 1660, 65535, 1661, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1662, 65535, 65535, 65535, 65535, 65535, 1663, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1664, 65535, 65535, 65535, 1665, 65535, 1666, 65535, 65535, 65535, 65535, 65535, 1667, 65535, 65535, 65535, 1668, 65535, 65535, 65535, 65535, 65535, 1669, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1670, 65535, 65535, 65535, 1671, 65535, 1672, 65535, 65535, 65535, 1673, 65535, 1674, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1675, 65535, 65535, 65535, 65535, 65535, 1676, 65535, 65535, 65535, 1677, 65535, 65535, 65535, 65535, 65535, 1678, 65535, 65535, 65535, 65535, 65535, 1679, 65535, 65535, 65535, 65535, 65535, 1680, 65535, 1681, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1682, 65535, 65535, 65535, 1683, 65535, 1684, 65535, 65535, 65535, 1685, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1686, 65535, 65535, 65535, 65535, 65535, 1687, 65535, 65535, 65535, 65535, 65535, 1688, 65535, 65535, 65535, 1689, 65535, 1690, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1691, 65535, 1692, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1693, 65535, 65535, 65535, 65535, 65535, 1694, 65535, 1695, 65535, 65535, 65535, 1696, 65535, 65535, 65535, 65535, 65535, 1697, 65535, 1698, 65535, 65535, 65535, 65535, 65535, 1699, 65535, 65535, 65535, 1700, 65535, 1701, 65535, 65535, 65535, 1702, 65535, 65535, 65535, 65535, 65535, 1703, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1704, 65535, 65535, 65535, 65535, 65535, 1705, 65535, 65535, 65535, 1706, 65535, 1707, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1708, 65535, 65535, 65535, 65535, 65535, 1709, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1710, 65535, 65535, 65535, 65535, 65535, 1711, 65535, 65535, 65535, 1712, 65535, 1713, 65535, 65535, 65535, 1714, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1715, 65535, 65535, 65535, 65535, 65535, 1716, 65535, 65535, 65535, 1717, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1718, 65535, 65535, 65535, 1719, 65535, 65535, 65535, 65535, 65535, 1720, 65535, 1721, 65535, 65535, 65535, 65535, 65535, 1722, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1723, 65535, 65535, 65535, 1724, 65535, 65535, 65535, 65535, 65535, 1725, 65535, 1726, 65535, 65535, 65535, 65535, 65535, 1727, 65535, 65535, 65535, 1728, 65535, 1729, 65535, 65535, 65535, 1730, 65535, 1731, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1732, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1733, 65535, 1734, 65535, 65535, 65535, 1735, 65535, 1736, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1737, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1738, 65535, 65535, 65535, 1739, 65535, 1740, 65535, 65535, 65535, 1741, 65535, 65535, 65535, 65535, 65535, 1742, 65535, 65535, 65535, 65535, 65535, 1743, 65535, 1744, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1745, 65535, 1746, 65535, 65535, 65535, 65535, 65535, 1747, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1748, 65535, 65535, 65535, 1749, 65535, 1750, 65535, 65535, 65535, 1751, 65535, 65535, 65535, 65535, 65535, 1752, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1753, 65535, 65535, 65535, 65535, 65535, 1754, 65535, 65535, 65535, 1755, 65535, 65535, 65535, 65535, 65535, 1756, 65535, 1757, 65535, 65535, 65535, 1758, 65535, 65535, 65535, 65535, 65535, 1759, 65535, 1760, 65535, 65535, 65535, 65535, 65535, 1761, 65535, 65535, 65535, 65535, 65535, 1762, 65535, 65535, 65535, 1763, 65535, 1764, 65535, 65535, 65535, 1765, 65535, 65535, 65535, 65535, 65535, 1766, 65535, 1767, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1768, 65535, 1769, 65535, 65535, 65535, 1770, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1771, 65535, 1772, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1773, 65535, 65535, 65535, 1774, 65535, 1775, 65535, 65535, 65535, 1776, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1777, 65535, 65535, 65535, 65535, 65535, 1778, 65535, 65535, 65535, 1779, 65535, 1780, 65535, 65535, 65535, 1781, 65535, 65535, 65535, 65535, 65535, 1782, 65535, 65535, 65535, 65535, 65535, 1783, 65535, 1784, 65535, 65535, 65535, 65535, 65535, 1785, 65535, 65535, 65535, 1786, 65535, 1787, 65535, 65535, 65535, 65535, 65535, 1788, 65535, 65535, 65535, 1789, 65535, 65535, 65535, 65535, 65535, 1790, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1791, 65535, 65535, 65535, 1792, 65535, 1793, 65535, 65535, 65535, 65535, 65535, 1794, 65535, 65535, 65535, 1795, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1796, 65535, 65535, 65535, 1797, 65535, 65535, 65535, 65535, 65535, 1798, 65535, 1799, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1800, 65535, 1801, 65535, 65535, 65535, 65535, 65535, 1802, 65535, 65535, 65535, 65535, 65535, 1803, 65535, 65535, 65535, 1804, 65535, 1805, 65535, 65535, 65535, 1806, 65535, 65535, 65535, 65535, 65535, 1807, 65535, 1808, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1809, 65535, 1810, 65535, 65535, 65535, 1811, 65535, 1812, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1813, 65535, 1814, 65535, 65535, 65535, 1815, 65535, 1816, 65535, 65535, 65535, 1817, 65535, 65535, 65535, 65535, 65535, 1818, 65535, 1819, 65535, 65535, 65535, 65535, 65535, 1820, 65535, 65535, 65535, 1821, 65535, 65535, 65535, 65535, 65535, 1822, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1823, 65535, 1824, 65535, 65535, 65535, 65535, 65535, 1825, 65535, 65535, 65535, 1826, 65535, 1827, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1828, 65535, 65535, 65535, 65535, 65535, 1829, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1830, 65535, 65535, 65535, 1831, 65535, 1832, 65535, 65535, 65535, 1833, 65535, 65535, 65535, 65535, 65535, 1834, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1835, 65535, 65535, 65535, 65535, 65535, 1836, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1837, 65535, 1838, 65535, 65535, 65535, 1839, 65535, 65535, 65535, 65535, 65535, 1840, 65535, 1841, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1842, 65535, 65535, 65535, 1843, 65535, 1844, 65535, 65535, 65535, 1845, 65535, 65535, 65535, 65535, 65535, 1846, 65535, 1847, 65535, 65535, 65535, 65535, 65535, 1848, 65535, 65535, 65535, 1849, 65535, 1850, 65535, 65535, 65535, 65535, 65535, 1851, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1852, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1853, 65535, 1854, 65535, 65535, 65535, 1855, 65535, 1856, 65535, 65535, 65535, 1857, 65535, 65535, 65535, 65535, 65535, 1858, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1859, 65535, 65535, 65535, 1860, 65535, 1861, 65535, 65535, 65535, 1862, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1863, 65535, 1864, 65535, 65535, 65535, 65535, 65535, 1865, 65535, 65535, 65535, 1866, 65535, 1867, 65535, 65535, 65535, 65535, 65535, 1868, 65535, 65535, 65535, 1869, 65535, 65535, 65535, 65535, 65535, 1870, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1871, 65535, 65535, 65535, 65535, 65535, 1872, 65535, 1873, 65535, 65535, 65535, 1874, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1875, 65535, 65535, 65535, 65535, 65535, 1876, 65535, 65535, 65535, 1877, 65535, 65535, 65535, 65535, 65535, 1878, 65535, 1879, 65535, 65535, 65535, 1880, 65535, 65535, 65535, 65535, 65535, 1881, 65535, 1882, 65535, 65535, 65535, 65535, 65535, 1883, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1884, 65535, 1885, 65535, 65535, 65535, 1886, 65535, 65535, 65535, 65535, 65535, 1887, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1888, 65535, 65535, 65535, 1889, 65535, 1890, 65535, 65535, 65535, 1891, 65535, 1892, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1893, 65535, 1894, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1895, 65535, 1896, 65535, 65535, 65535, 1897, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1898, 65535, 1899, 65535, 65535, 65535, 65535, 65535, 1900, 65535, 65535, 65535, 65535, 65535, 1901, 65535, 65535, 65535, 1902, 65535, 65535, 65535, 65535, 65535, 1903, 65535, 65535, 65535, 65535, 65535, 1904, 65535, 1905, 65535, 65535, 65535, 65535, 65535, 1906, 65535, 65535, 65535, 1907, 65535, 1908, 65535, 65535, 65535, 65535, 65535, 1909, 65535, 65535, 65535, 1910, 65535, 65535, 65535, 65535, 65535, 1911, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1912, 65535, 65535, 65535, 65535, 65535, 1913, 65535, 65535, 65535, 65535, 65535, 1914, 65535, 65535, 65535, 1915, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1916, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1917, 65535, 65535, 65535, 65535, 65535, 1918, 65535, 1919, 65535, 65535, 65535, 1920, 65535, 65535, 65535, 65535, 65535, 1921, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1922, 65535, 65535, 65535, 65535, 65535, 1923, 65535, 65535, 65535, 1924, 65535, 1925, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1926, 65535, 65535, 65535, 65535, 65535, 1927, 65535, 65535, 65535, 1928, 65535, 1929, 65535, 65535, 65535, 1930, 65535, 1931, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1932, 65535, 1933, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1934, 65535, 1935, 65535, 65535, 65535, 1936, 65535, 1937, 65535, 65535, 65535, 1938, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1939, 65535, 65535, 65535, 65535, 65535, 1940, 65535, 65535, 65535, 1941, 65535, 1942, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1943, 65535, 65535, 65535, 65535, 65535, 1944, 65535, 1945, 65535, 65535, 65535, 65535, 65535, 1946, 65535, 65535, 65535, 1947, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1948, 65535, 65535, 65535, 1949, 65535, 65535, 65535, 65535, 65535, 1950, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1951, 65535, 65535, 65535, 1952, 65535, 1953, 65535, 65535, 65535, 1954, 65535, 1955, 65535, 65535, 65535, 1956, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1957, 65535, 65535, 65535, 65535, 65535, 1958, 65535, 65535, 65535, 1959, 65535, 65535, 65535, 65535, 65535, 1960, 65535, 65535, 65535, 65535, 65535, 1961, 65535, 65535, 65535, 65535, 65535, 1962, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1963, 65535, 65535, 65535, 65535, 65535, 1964, 65535, 65535, 65535, 1965, 65535, 1966, 65535, 65535, 65535, 1967, 65535, 65535, 65535, 65535, 65535, 1968, 65535, 1969, 65535, 65535, 65535, 65535, 65535, 1970, 65535, 65535, 65535, 1971, 65535, 1972, 65535, 65535, 65535, 1973, 65535, 1974, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1975, 65535, 1976, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1977, 65535, 1978, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1979, 65535, 65535, 65535, 65535, 65535, 1980, 65535, 1981, 65535, 65535, 65535, 65535, 65535, 1982, 65535, 65535, 65535, 1983, 65535, 1984, 65535, 65535, 65535, 1985, 65535, 65535, 65535, 65535, 65535, 1986, 65535, 65535, 65535, 65535, 65535, 1987, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1988, 65535, 65535, 65535, 65535, 65535, 1989, 65535, 65535, 65535, 65535, 65535, 1990, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1991, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1992, 65535, 1993, 65535, 65535, 65535, 1994, 65535, 1995, 65535, 65535, 65535, 1996, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1997, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1998, 65535, 65535, 65535, 65535, 65535, 1999, 65535, 2000, 65535, 65535, 65535, 2001, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2002, 65535, 65535, 65535, 65535, 65535, 2003, 65535, 65535, 65535, 65535, 65535, 2004, 65535, 65535, 65535, 2005, 65535, 2006, 65535, 65535, 65535, 2007, 65535, 65535, 65535, 65535, 65535, 2008, 65535, 2009, 65535, 65535, 65535, 65535, 65535, 2010, 65535, 65535, 65535, 2011, 65535, 65535, 65535, 65535, 65535, 2012, 65535, 2013, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2014, 65535, 2015, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2016, 65535, 2017, 65535, 65535, 65535, 65535, 65535, 2018, 65535, 65535, 65535, 2019, 65535, 65535, 65535, 65535, 65535, 2020, 65535, 2021, 65535, 65535, 65535, 65535, 65535, 2022, 65535, 65535, 65535, 2023, 65535, 65535, 65535, 65535, 65535, 2024, 65535, 65535, 65535, 65535, 65535, 2025, 65535, 65535, 65535, 65535, 65535, 2026, 65535, 2027, 65535, 65535, 65535, 65535, 65535, 2028, 65535, 65535, 65535, 65535, 65535, 2029, 65535, 65535, 65535, 65535, 65535, 2030, 65535, 65535, 65535, 2031, 65535, 65535, 65535, 65535, 65535, 2032, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2033, 65535, 65535, 65535, 2034, 65535, 2035, 65535, 65535, 65535, 2036, 65535, 2037, 65535, 65535, 65535, 2038, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2039, 65535, 65535, 65535, 2040, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2041, 65535, 65535, 65535, 2042, 65535, 65535, 65535, 65535, 65535, 2043, 65535, 2044, 65535, 65535, 65535, 65535, 65535, 2045, 65535, 65535, 65535, 65535, 65535, 2046, 65535, 65535, 65535, 2047, 65535, 2048, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2049, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2050, 65535, 65535, 65535, 2051, 65535, 2052, 65535, 65535, 65535, 2053, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2054, 65535, 2055, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2056, 65535, 2057, 65535, 65535, 65535, 2058, 65535, 2059, 65535, 65535, 65535, 2060, 65535, 65535, 65535, 65535, 65535, 2061, 65535, 2062, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2063, 65535, 65535, 65535, 2064, 65535, 65535, 65535, 65535, 65535, 2065, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2066, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2067, 65535, 2068, 65535, 65535, 65535, 65535, 65535, 2069, 65535, 65535, 65535, 2070, 65535, 65535, 65535, 65535, 65535, 2071, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2072, 65535, 65535, 65535, 2073, 65535, 65535, 65535, 65535, 65535, 2074, 65535, 2075, 65535, 65535, 65535, 2076, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2077, 65535, 65535, 65535, 65535, 65535, 2078, 65535, 65535, 65535, 2079, 65535, 65535, 65535, 65535, 65535, 2080, 65535, 2081, 65535, 65535, 65535, 2082, 65535, 65535, 65535, 65535, 65535, 2083, 65535, 2084, 65535, 65535, 65535, 65535, 65535, 2085, 65535, 65535, 65535, 65535, 65535, 2086, 65535, 65535, 65535, 2087, 65535, 65535, 65535, 65535, 65535, 2088, 65535, 65535, 65535, 65535, 65535, 2089, 65535, 2090, 65535, 65535, 65535, 65535, 65535, 2091, 65535, 65535, 65535, 65535, 65535, 2092, 65535, 65535, 65535, 65535, 65535, 2093, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2094, 65535, 2095, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2096, 65535, 65535, 65535, 65535, 65535, 2097, 65535, 2098, 65535, 65535, 65535, 2099, 65535, 65535, 65535, 65535, 65535, 2100, 65535, 2101, 65535, 65535, 65535, 65535, 65535, 2102, 65535, 65535, 65535, 2103, 65535, 2104, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2105, 65535, 65535, 65535, 65535, 65535, 2106, 65535, 2107, 65535, 65535, 65535, 65535, 65535, 2108, 65535, 65535, 65535, 2109, 65535, 2110, 65535, 65535, 65535, 65535, 65535, 2111, 65535, 65535, 65535, 2112, 65535, 65535, 65535, 65535, 65535, 2113, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2114, 65535, 65535, 65535, 2115, 65535, 2116, 65535, 65535, 65535, 2117, 65535, 2118, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2119, 65535, 65535, 65535, 65535, 65535, 2120, 65535, 65535, 65535, 2121, 65535, 65535, 65535, 65535, 65535, 2122, 65535, 2123, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2124, 65535, 2125, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2126, 65535, 65535, 65535, 2127, 65535, 65535, 65535, 65535, 65535, 2128, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2129, 65535, 65535, 65535, 65535, 65535, 2130, 65535, 65535, 65535, 2131, 65535, 2132, 65535, 65535, 65535, 2133, 65535, 2134, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2135, 65535, 2136, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2137, 65535, 2138, 65535, 65535, 65535, 2139, 65535, 65535, 65535, 65535, 65535, 2140, 65535, 2141, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2142, 65535, 2143, 65535, 65535, 65535, 2144, 65535, 65535, 65535, 65535, 65535, 2145, 65535, 65535, 65535, 65535, 65535, 2146, 65535, 2147, 65535, 65535, 65535, 65535, 65535, 2148, 65535, 65535, 65535, 2149, 65535, 2150, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2151, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2152, 65535, 65535, 65535, 65535, 65535, 2153, 65535, 65535, 65535, 2154, 65535, 2155, 65535, 65535, 65535, 2156, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2157, 65535, 65535, 65535, 65535, 65535, 2158, 65535, 65535, 65535, 2159, 65535, 65535, 65535, 65535, 65535, 2160, 65535, 2161, 65535, 65535, 65535, 2162, 65535, 65535, 65535, 65535, 65535, 2163, 65535, 2164, 65535, 65535, 65535, 65535, 65535, 2165, 65535, 65535, 65535, 65535, 65535, 2166, 65535, 65535, 65535, 65535, 65535, 2167, 65535, 65535, 65535, 2168, 65535, 65535, 65535, 65535, 65535, 2169, 65535, 2170, 65535, 65535, 65535, 65535, 65535, 2171, 65535, 65535, 65535, 2172, 65535, 65535, 65535, 65535, 65535, 2173, 65535, 2174, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2175, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2176, 65535, 2177, 65535, 65535, 65535, 2178, 65535, 2179, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2180, 65535, 2181, 65535, 65535, 65535, 65535, 65535, 2182, 65535, 65535, 65535, 2183, 65535, 2184, 65535, 65535, 65535, 2185, 65535, 65535, 65535, 65535, 65535, 2186, 65535, 65535, 65535, 65535, 65535, 2187, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2188, 65535, 2189, 65535, 65535, 65535, 65535, 65535, 2190, 65535, 65535, 65535, 2191, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2192, 65535, 65535, 65535, 2193, 65535, 2194, 65535, 65535, 65535, 2195, 65535, 2196, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2197, 65535, 65535, 65535, 65535, 65535, 2198, 65535, 65535, 65535, 2199, 65535, 65535, 65535, 65535, 65535, 2200, 65535, 65535, 65535, 65535, 65535, 2201, 65535, 65535, 65535, 65535, 65535, 2202, 65535, 2203, 65535, 65535, 65535, 65535, 65535, 2204, 65535, 65535, 65535, 65535, 65535, 2205, 65535, 65535, 65535, 2206, 65535, 2207, 65535, 65535, 65535, 2208, 65535, 65535, 65535, 65535, 65535, 2209, 65535, 2210, 65535, 65535, 65535, 65535, 65535, 2211, 65535, 65535, 65535, 65535, 65535, 2212, 65535, 65535, 65535, 2213, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2214, 65535, 2215, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2216, 65535, 65535, 65535, 2217, 65535, 2218, 65535, 65535, 65535, 2219, 65535, 65535, 65535, 65535, 65535, 2220, 65535, 2221, 65535, 65535, 65535, 65535, 65535, 2222, 65535, 65535, 65535, 2223, 65535, 2224, 65535, 65535, 65535, 2225, 65535, 65535, 65535, 65535, 65535, 2226, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2227, 65535, 65535, 65535, 65535, 65535, 2228, 65535, 65535, 65535, 2229, 65535, 2230, 65535, 65535, 65535, 65535, 65535, 2231, 65535, 65535, 65535, 2232, 65535, 65535, 65535, 65535, 65535, 2233, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2234, 65535, 65535, 65535, 2235, 65535, 2236, 65535, 65535, 65535, 2237, 65535, 2238, 65535, 65535, 65535, 2239, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2240, 65535, 65535, 65535, 2241, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2242, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2243, 65535, 2244, 65535, 65535, 65535, 65535, 65535, 2245, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2246, 65535, 2247, 65535, 65535, 65535, 2248, 65535, 65535, 65535, 65535, 65535, 2249, 65535, 2250, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2251, 65535, 2252, 65535, 65535, 65535, 2253, 65535, 2254, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2255, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2256, 65535, 2257, 65535, 65535, 65535, 2258, 65535, 2259, 65535, 65535, 65535, 2260, 65535, 65535, 65535, 65535, 65535, 2261, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2262, 65535, 65535, 65535, 2263, 65535, 65535, 65535, 65535, 65535, 2264, 65535, 65535, 65535, 65535, 65535, 2265, 65535, 65535, 65535, 65535, 65535, 2266, 65535, 2267, 65535, 65535, 65535, 65535, 65535, 2268, 65535, 65535, 65535, 2269, 65535, 2270, 65535, 65535, 65535, 65535, 65535, 2271, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2272, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2273, 65535, 65535, 65535, 2274, 65535, 2275, 65535, 65535, 65535, 2276, 65535, 65535, 65535, 65535, 65535, 2277, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2278, 65535, 65535, 65535, 65535, 65535, 2279, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2280, 65535, 2281, 65535, 65535, 65535, 2282, 65535, 65535, 65535, 65535, 65535, 2283, 65535, 2284, 65535, 65535, 65535, 65535, 65535, 2285, 65535, 65535, 65535, 65535, 65535, 2286, 65535, 65535, 65535, 2287, 65535, 2288, 65535, 65535, 65535, 2289, 65535, 65535, 65535, 65535, 65535, 2290, 65535, 2291, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2292, 65535, 2293, 65535, 65535, 65535, 65535, 65535, 2294, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2295, 65535, 2296, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2297, 65535, 2298, 65535, 65535, 65535, 2299, 65535, 2300, 65535, 65535, 65535, 2301, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2302, 65535, 65535, 65535, 2303, 65535, 2304, 65535, 65535, 65535, 2305, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2306, 65535, 2307, 65535, 65535, 65535, 65535, 65535, 2308, 65535, 65535, 65535, 2309, 65535, 2310, 65535, 65535, 65535, 65535, 65535, 2311, 65535, 65535, 65535, 2312, 65535, 65535, 65535, 65535, 65535, 2313, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2314, 65535, 2315, 65535, 65535, 65535, 65535, 65535, 2316, 65535, 65535, 65535, 2317, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2318, 65535, 65535, 65535, 65535, 65535, 2319, 65535, 65535, 65535, 2320, 65535, 65535, 65535, 65535, 65535, 2321, 65535, 2322, 65535, 65535, 65535, 2323, 65535, 65535, 65535, 65535, 65535, 2324, 65535, 2325, 65535, 65535, 65535, 65535, 65535, 2326, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2327, 65535, 2328, 65535, 65535, 65535, 2329, 65535, 65535, 65535, 65535, 65535, 2330, 65535, 2331, 65535, 65535, 65535, 65535, 65535, 2332, 65535, 65535, 65535, 2333, 65535, 2334, 65535, 65535, 65535, 2335, 65535, 2336, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2337, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2338, 65535, 2339, 65535, 65535, 65535, 2340, 65535, 65535, 65535, 65535, 65535, 2341, 65535, 65535, 65535, 65535, 65535, 2342, 65535, 2343, 65535, 65535, 65535, 65535, 65535, 2344, 65535, 65535, 65535, 65535, 65535, 2345, 65535, 65535, 65535, 2346, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2347, 65535, 2348, 65535, 65535, 65535, 65535, 65535, 2349, 65535, 65535, 65535, 2350, 65535, 2351, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2352, 65535, 65535, 65535, 65535, 65535, 2353, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2354, 65535, 65535, 65535, 2355, 65535, 2356, 65535, 65535, 65535, 65535, 65535, 2357, 65535, 65535, 65535, 2358, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2359, 65535, 65535, 65535, 65535, 65535, 2360, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2361, 65535, 2362, 65535, 65535, 65535, 2363, 65535, 65535, 65535, 65535, 65535, 2364, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2365, 65535, 65535, 65535, 2366, 65535, 2367, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2368, 65535, 2369, 65535, 65535, 65535, 65535, 65535, 2370, 65535, 65535, 65535, 2371, 65535, 2372, 65535, 65535, 65535, 2373, 65535, 2374, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2375, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2376, 65535, 2377, 65535, 65535, 65535, 2378, 65535, 2379, 65535, 65535, 65535, 2380, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2381, 65535, 65535, 65535, 65535, 65535, 2382, 65535, 65535, 65535, 2383, 65535, 2384, 65535, 65535, 65535, 2385, 65535, 65535, 65535, 65535, 65535, 2386, 65535, 65535, 65535, 65535, 65535, 2387, 65535, 2388, 65535, 65535, 65535, 65535, 65535, 2389, 65535, 65535, 65535, 2390, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2391, 65535, 65535, 65535, 2392, 65535, 65535, 65535, 65535, 65535, 2393, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2394, 65535, 65535, 65535, 2395, 65535, 65535, 65535, 65535, 65535, 2396, 65535, 2397, 65535, 65535, 65535, 2398, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2399, 65535, 65535, 65535, 65535, 65535, 2400, 65535, 65535, 65535, 2401, 65535, 65535, 65535, 65535, 65535, 2402, 65535, 65535, 65535, 65535, 65535, 2403, 65535, 65535, 65535, 65535, 65535, 2404, 65535, 2405, 65535, 65535, 65535, 65535, 65535, 2406, 65535, 65535, 65535, 65535, 65535, 2407, 65535, 65535, 65535, 2408, 65535, 2409, 65535, 65535, 65535, 2410, 65535, 65535, 65535, 65535, 65535, 2411, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2412, 65535, 65535, 65535, 2413, 65535, 2414, 65535, 65535, 65535, 2415, 65535, 2416, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2417, 65535, 2418, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2419, 65535, 2420, 65535, 65535, 65535, 65535, 65535, 2421, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2422, 65535, 2423, 65535, 65535, 65535, 65535, 65535, 2424, 65535, 65535, 65535, 2425, 65535, 2426, 65535, 65535, 65535, 2427, 65535, 65535, 65535, 65535, 65535, 2428, 65535, 65535, 65535, 65535, 65535, 2429, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2430, 65535, 65535, 65535, 2431, 65535, 2432, 65535, 65535, 65535, 65535, 65535, 2433, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2434, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2435, 65535, 65535, 65535, 65535, 65535, 2436, 65535, 65535, 65535, 2437, 65535, 2438, 65535, 65535, 65535, 2439, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2440, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2441, 65535, 65535, 65535, 65535, 65535, 2442, 65535, 2443, 65535, 65535, 65535, 2444, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2445, 65535, 65535, 65535, 65535, 65535, 2446, 65535, 65535, 65535, 65535, 65535, 2447, 65535, 65535, 65535, 2448, 65535, 2449, 65535, 65535, 65535, 2450, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2451, 65535, 65535, 65535, 65535, 65535, 2452, 65535, 65535, 65535, 2453, 65535, 65535, 65535, 65535, 65535, 2454, 65535, 2455, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2456, 65535, 2457, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2458, 65535, 2459, 65535, 65535, 65535, 2460, 65535, 2461, 65535, 65535, 65535, 2462, 65535, 65535, 65535, 65535, 65535, 2463, 65535, 2464, 65535, 65535, 65535, 65535, 65535, 2465, 65535, 65535, 65535, 2466, 65535, 2467, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2468, 65535, 65535, 65535, 65535, 65535, 2469, 65535, 2470, 65535, 65535, 65535, 65535, 65535, 2471, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2472, 65535, 65535, 65535, 2473, 65535, 65535, 65535, 65535, 65535, 2474, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2475, 65535, 65535, 65535, 2476, 65535, 2477, 65535, 65535, 65535, 2478, 65535, 2479, 65535, 65535, 65535, 2480, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2481, 65535, 65535, 65535, 65535, 65535, 2482, 65535, 65535, 65535, 2483, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2484, 65535, 65535, 65535, 2485, 65535, 65535, 65535, 65535, 65535, 2486, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2487, 65535, 65535, 65535, 65535, 65535, 2488, 65535, 65535, 65535, 2489, 65535, 2490, 65535, 65535, 65535, 2491, 65535, 65535, 65535, 65535, 65535, 2492, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2493, 65535, 65535, 65535, 2494, 65535, 2495, 65535, 65535, 65535, 2496, 65535, 2497, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2498, 65535, 2499, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2500, 65535, 2501, 65535, 65535, 65535, 2502, 65535, 65535, 65535, 65535, 65535, 2503, 65535, 65535, 65535, 65535, 65535, 2504, 65535, 2505, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2506, 65535, 2507, 65535, 65535, 65535, 2508, 65535, 65535, 65535, 65535, 65535, 2509, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2510, 65535, 65535, 65535, 65535, 65535, 2511, 65535, 65535, 65535, 65535, 65535, 2512, 65535, 65535, 65535, 65535, 65535, 2513, 65535, 65535, 65535, 2514, 65535, 65535, 65535, 65535, 65535, 2515, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2516, 65535, 65535, 65535, 65535, 65535, 2517, 65535, 2518, 65535, 65535, 65535, 2519, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2520, 65535, 65535, 65535, 65535, 65535, 2521, 65535, 65535, 65535, 2522, 65535, 65535, 65535, 65535, 65535, 2523, 65535, 2524, 65535, 65535, 65535, 2525, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2526, 65535, 65535, 65535, 65535, 65535, 2527, 65535, 65535, 65535, 65535, 65535, 2528, 65535, 65535, 65535, 2529, 65535, 65535, 65535, 65535, 65535, 2530, 65535, 65535, 65535, 65535, 65535, 2531, 65535, 2532, 65535, 65535, 65535, 65535, 65535, 2533, 65535, 65535, 65535, 65535, 65535, 2534, 65535, 65535, 65535, 2535, 65535, 2536, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2537, 65535, 2538, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2539, 65535, 2540, 65535, 65535, 65535, 65535, 65535, 2541, 65535, 65535, 65535, 2542, 65535, 65535, 65535, 65535, 65535, 2543, 65535, 2544, 65535, 65535, 65535, 65535, 65535, 2545, 65535, 65535, 65535, 2546, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2547, 65535, 65535, 65535, 65535, 65535, 2548, 65535, 2549, 65535, 65535, 65535, 65535, 65535, 2550, 65535, 65535, 65535, 2551, 65535, 2552, 65535, 65535, 65535, 65535, 65535, 2553, 65535, 65535, 65535, 2554, 65535, 65535, 65535, 65535, 65535, 2555, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2556, 65535, 65535, 65535, 2557, 65535, 2558, 65535, 65535, 65535, 2559, 65535, 2560, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2561, 65535, 65535, 65535, 2562, 65535, 65535, 65535, 65535, 65535, 2563, 65535, 2564, 65535, 65535, 65535, 2565, 65535, 65535, 65535, 65535, 65535, 2566, 65535, 2567, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2568, 65535, 65535, 65535, 2569, 65535, 2570, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2571, 65535, 65535, 65535, 65535, 65535, 2572, 65535, 65535, 65535, 2573, 65535, 2574, 65535, 65535, 65535, 2575, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2576, 65535, 2577, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2578, 65535, 65535, 65535, 65535, 65535, 2579, 65535, 2580, 65535, 65535, 65535, 2581, 65535, 65535, 65535, 65535, 65535, 2582, 65535, 2583, 65535, 65535, 65535, 65535, 65535, 2584, 65535, 65535, 65535, 65535, 65535, 2585, 65535, 65535, 65535, 2586, 65535, 65535, 65535, 65535, 65535, 2587, 65535, 65535, 65535, 65535, 65535, 2588, 65535, 2589, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2590, 65535, 2591, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2592, 65535, 65535, 65535, 65535, 65535, 2593, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2594, 65535, 65535, 65535, 65535, 65535, 2595, 65535, 65535, 65535, 2596, 65535, 2597, 65535, 65535, 65535, 2598, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2599, 65535, 65535, 65535, 65535, 65535, 2600, 65535, 65535, 65535, 2601, 65535, 65535, 65535, 65535, 65535, 2602, 65535, 2603, 65535, 65535, 65535, 2604, 65535, 65535, 65535, 65535, 65535, 2605, 65535, 2606, 65535, 65535, 65535, 65535, 65535, 2607, 65535, 65535, 65535, 65535, 65535, 2608, 65535, 65535, 65535, 65535, 65535, 2609, 65535, 65535, 65535, 2610, 65535, 65535, 65535, 65535, 65535, 2611, 65535, 2612, 65535, 65535, 65535, 65535, 65535, 2613, 65535, 65535, 65535, 2614, 65535, 2615, 65535, 65535, 65535, 65535, 65535, 2616, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2617, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2618, 65535, 65535, 65535, 65535, 65535, 2619, 65535, 2620, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2621, 65535, 2622, 65535, 65535, 65535, 65535, 65535, 2623, 65535, 65535, 65535, 2624, 65535, 2625, 65535, 65535, 65535, 2626, 65535, 65535, 65535, 65535, 65535, 2627, 65535, 65535, 65535, 65535, 65535, 2628, 65535, 2629, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2630, 65535, 2631, 65535, 65535, 65535, 65535, 65535, 2632, 65535, 65535, 65535, 2633, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2634, 65535, 65535, 65535, 2635, 65535, 2636, 65535, 65535, 65535, 2637, 65535, 2638, 65535, 65535, 65535, 2639, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2640, 65535, 65535, 65535, 65535, 65535, 2641, 65535, 65535, 65535, 2642, 65535, 65535, 65535, 65535, 65535, 2643, 65535, 2644, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2645, 65535, 2646, 65535, 65535, 65535, 65535, 65535, 2647, 65535, 65535, 65535, 65535, 65535, 2648, 65535, 65535, 65535, 2649, 65535, 65535, 65535, 65535, 65535, 2650, 65535, 65535, 65535, 65535, 65535, 2651, 65535, 2652, 65535, 65535, 65535, 65535, 65535, 2653, 65535, 65535, 65535, 2654, 65535, 2655, 65535, 65535, 65535, 2656, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2657, 65535, 2658, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2659, 65535, 65535, 65535, 2660, 65535, 2661, 65535, 65535, 65535, 2662, 65535, 65535, 65535, 65535, 65535, 2663, 65535, 2664, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2665, 65535, 2666, 65535, 65535, 65535, 2667, 65535, 65535, 65535, 65535, 65535, 2668, 65535, 65535, 65535, 65535, 65535, 2669, 65535, 2670, 65535, 65535, 65535, 65535, 65535, 2671, 65535, 65535, 65535, 2672, 65535, 2673, 65535, 65535, 65535, 65535, 65535, 2674, 65535, 65535, 65535, 2675, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2676, 65535, 65535, 65535, 2677, 65535, 2678, 65535, 65535, 65535, 2679, 65535, 2680, 65535, 65535, 65535, 2681, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2682, 65535, 65535, 65535, 2683, 65535, 65535, 65535, 65535, 65535, 2684, 65535, 2685, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2686, 65535, 2687, 65535, 65535, 65535, 65535, 65535, 2688, 65535, 65535, 65535, 65535, 65535, 2689, 65535, 65535, 65535, 65535, 65535, 2690, 65535, 65535, 65535, 2691, 65535, 65535, 65535, 65535, 65535, 2692, 65535, 2693, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2694, 65535, 65535, 65535, 65535, 65535, 2695, 65535, 2696, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2697, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2698, 65535, 2699, 65535, 65535, 65535, 2700, 65535, 2701, 65535, 65535, 65535, 2702, 65535, 65535, 65535, 65535, 65535, 2703, 65535, 2704, 65535, 65535, 65535, 65535, 65535, 2705, 65535, 65535, 65535, 2706, 65535, 65535, 65535, 65535, 65535, 2707, 65535, 65535, 65535, 65535, 65535, 2708, 65535, 65535, 65535, 65535, 65535, 2709, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2710, 65535, 65535, 65535, 2711, 65535, 2712, 65535, 65535, 65535, 65535, 65535, 2713, 65535, 65535, 65535, 2714, 65535, 65535, 65535, 65535, 65535, 2715, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2716, 65535, 65535, 65535, 2717, 65535, 2718, 65535, 65535, 65535, 2719, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2720, 65535, 65535, 65535, 65535, 65535, 2721, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2722, 65535, 65535, 65535, 65535, 65535, 2723, 65535, 65535, 65535, 65535, 65535, 2724, 65535, 2725, 65535, 65535, 65535, 65535, 65535, 2726, 65535, 65535, 65535, 65535, 65535, 2727, 65535, 65535, 65535, 2728, 65535, 2729, 65535, 65535, 65535, 2730, 65535, 65535, 65535, 65535, 65535, 2731, 65535, 2732, 65535, 65535, 65535, 65535, 65535, 2733, 65535, 65535, 65535, 65535, 65535, 2734, 65535, 65535, 65535, 65535, 65535, 2735, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2736, 65535, 2737, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2738, 65535, 2739, 65535, 65535, 65535, 2740, 65535, 2741, 65535, 65535, 65535, 2742, 65535, 65535, 65535, 65535, 65535, 2743, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2744, 65535, 65535, 65535, 2745, 65535, 2746, 65535, 65535, 65535, 2747, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2748, 65535, 65535, 65535, 65535, 65535, 2749, 65535, 65535, 65535, 2750, 65535, 2751, 65535, 65535, 65535, 65535, 65535, 2752, 65535, 65535, 65535, 2753, 65535, 65535, 65535, 65535, 65535, 2754, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2755, 65535, 2756, 65535, 65535, 65535, 2757, 65535, 2758, 65535, 65535, 65535, 2759, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2760, 65535, 65535, 65535, 65535, 65535, 2761, 65535, 65535, 65535, 2762, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2763, 65535, 65535, 65535, 2764, 65535, 65535, 65535, 65535, 65535, 2765, 65535, 2766, 65535, 65535, 65535, 65535, 65535, 2767, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2768, 65535, 2769, 65535, 65535, 65535, 2770, 65535, 65535, 65535, 65535, 65535, 2771, 65535, 2772, 65535, 65535, 65535, 65535, 65535, 2773, 65535, 65535, 65535, 2774, 65535, 2775, 65535, 65535, 65535, 2776, 65535, 2777, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2778, 65535, 2779, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2780, 65535, 2781, 65535, 65535, 65535, 2782, 65535, 65535, 65535, 65535, 65535, 2783, 65535, 65535, 65535, 65535, 65535, 2784, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2785, 65535, 65535, 65535, 65535, 65535, 2786, 65535, 65535, 65535, 2787, 65535, 65535, 65535, 65535, 65535, 2788, 65535, 65535, 65535, 65535, 65535, 2789, 65535, 2790, 65535, 65535, 65535, 65535, 65535, 2791, 65535, 65535, 65535, 2792, 65535, 2793, 65535, 65535, 65535, 65535, 65535, 2794, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2795, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2796, 65535, 65535, 65535, 2797, 65535, 2798, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2799, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2800, 65535, 65535, 65535, 65535, 65535, 2801, 65535, 65535, 65535, 2802, 65535, 65535, 65535, 65535, 65535, 2803, 65535, 2804, 65535, 65535, 65535, 2805, 65535, 65535, 65535, 65535, 65535, 2806, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2807, 65535, 65535, 65535, 65535, 65535, 2808, 65535, 65535, 65535, 2809, 65535, 2810, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2811, 65535, 2812, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2813, 65535, 2814, 65535, 65535, 65535, 2815, 65535, 2816, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2817, 65535, 2818, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2819, 65535, 2820, 65535, 65535, 65535, 2821, 65535, 2822, 65535, 65535, 65535, 2823, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2824, 65535, 65535, 65535, 65535, 65535, 2825, 65535, 65535, 65535, 2826, 65535, 2827, 65535, 65535, 65535, 2828, 65535, 65535, 65535, 65535, 65535, 2829, 65535, 65535, 65535, 65535, 65535, 2830, 65535, 2831, 65535, 65535, 65535, 65535, 65535, 2832, 65535, 65535, 65535, 2833, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2834, 65535, 65535, 65535, 2835, 65535, 65535, 65535, 65535, 65535, 2836, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2837, 65535, 65535, 65535, 2838, 65535, 2839, 65535, 65535, 65535, 65535, 65535, 2840, 65535, 65535, 65535, 2841, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2842, 65535, 65535, 65535, 65535, 65535, 2843, 65535, 65535, 65535, 2844, 65535, 65535, 65535, 65535, 65535, 2845, 65535, 65535, 65535, 65535, 65535, 2846, 65535, 65535, 65535, 65535, 65535, 2847, 65535, 2848, 65535, 65535, 65535, 65535, 65535, 2849, 65535, 65535, 65535, 65535, 65535, 2850, 65535, 65535, 65535, 2851, 65535, 2852, 65535, 65535, 65535, 2853, 65535, 65535, 65535, 65535, 65535, 2854, 65535, 2855, 65535, 65535, 65535, 65535, 65535, 2856, 65535, 65535, 65535, 2857, 65535, 2858, 65535, 65535, 65535, 2859, 65535, 2860, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2861, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2862, 65535, 2863, 65535, 65535, 65535, 65535, 65535, 2864, 65535, 65535, 65535, 2865, 65535, 65535, 65535, 65535, 65535, 2866, 65535, 2867, 65535, 65535, 65535, 65535, 65535, 2868, 65535, 65535, 65535, 2869, 65535, 2870, 65535, 65535, 65535, 2871, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2872, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2873, 65535, 65535, 65535, 2874, 65535, 2875, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2876, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2877, 65535, 65535, 65535, 2878, 65535, 2879, 65535, 65535, 65535, 2880, 65535, 2881, 65535, 65535, 65535, 2882, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2883, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2884, 65535, 2885, 65535, 65535, 65535, 2886, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2887, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2888, 65535, 65535, 65535, 2889, 65535, 2890, 65535, 65535, 65535, 2891, 65535, 65535, 65535, 65535, 65535, 2892, 65535, 2893, 65535, 65535, 65535, 65535, 65535, 2894, 65535, 65535, 65535, 2895, 65535, 65535, 65535, 65535, 65535, 2896, 65535, 2897, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2898, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2899, 65535, 2900, 65535, 65535, 65535, 2901, 65535, 2902, 65535, 65535, 65535, 2903, 65535, 65535, 65535, 65535, 65535, 2904, 65535, 2905, 65535, 65535, 65535, 65535, 65535, 2906, 65535, 65535, 65535, 2907, 65535, 2908, 65535, 65535, 65535, 2909, 65535, 65535, 65535, 65535, 65535, 2910, 65535, 65535, 65535, 65535, 65535, 2911, 65535, 2912, 65535, 65535, 65535, 65535, 65535, 2913, 65535, 65535, 65535, 65535, 65535, 2914, 65535, 65535, 65535, 65535, 65535, 2915, 65535, 65535, 65535, 2916, 65535, 65535, 65535, 65535, 65535, 2917, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2918, 65535, 65535, 65535, 2919, 65535, 65535, 65535, 65535, 65535, 2920, 65535, 2921, 65535, 65535, 65535, 2922, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2923, 65535, 65535, 65535, 65535, 65535, 2924, 65535, 65535, 65535, 2925, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2926, 65535, 65535, 65535, 2927, 65535, 65535, 65535, 65535, 65535, 2928, 65535, 2929, 65535, 65535, 65535, 65535, 65535, 2930, 65535, 65535, 65535, 65535, 65535, 2931, 65535, 65535, 65535, 2932, 65535, 2933, 65535, 65535, 65535, 2934, 65535, 65535, 65535, 65535, 65535, 2935, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2936, 65535, 65535, 65535, 2937, 65535, 2938, 65535, 65535, 65535, 2939, 65535, 2940, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2941, 65535, 2942, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2943, 65535, 2944, 65535, 65535, 65535, 2945, 65535, 2946, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2947, 65535, 2948, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2949, 65535, 2950, 65535, 65535, 65535, 2951, 65535, 65535, 65535, 65535, 65535, 2952, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2953, 65535, 65535, 65535, 65535, 65535, 2954, 65535, 65535, 65535, 2955, 65535, 2956, 65535, 65535, 65535, 65535, 65535, 2957, 65535, 65535, 65535, 2958, 65535, 65535, 65535, 65535, 65535, 2959, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2960, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2961, 65535, 2962, 65535, 65535, 65535, 2963, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2964, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2965, 65535, 65535, 65535, 65535, 65535, 2966, 65535, 2967, 65535, 65535, 65535, 2968, 65535, 65535, 65535, 65535, 65535, 2969, 65535, 2970, 65535, 65535, 65535, 65535, 65535, 2971, 65535, 65535, 65535, 65535, 65535, 2972, 65535, 65535, 65535, 2973, 65535, 65535, 65535, 65535, 65535, 2974, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2975, 65535, 65535, 65535, 65535, 65535, 2976, 65535, 65535, 65535, 65535, 65535, 2977, 65535, 65535, 65535, 2978, 65535, 2979, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2980, 65535, 2981, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2982, 65535, 2983, 65535, 65535, 65535, 2984, 65535, 2985, 65535, 65535, 65535, 2986, 65535, 65535, 65535, 65535, 65535, 2987, 65535, 2988, 65535, 65535, 65535, 65535, 65535, 2989, 65535, 65535, 65535, 2990, 65535, 2991, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2992, 65535, 65535, 65535, 65535, 65535, 2993, 65535, 2994, 65535, 65535, 65535, 65535, 65535, 2995, 65535, 65535, 65535, 2996, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2997, 65535, 65535, 65535, 2998, 65535, 65535, 65535, 65535, 65535, 2999, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3000, 65535, 65535, 65535, 3001, 65535, 3002, 65535, 65535, 65535, 3003, 65535, 3004, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3005, 65535, 65535, 65535, 65535, 65535, 3006, 65535, 65535, 65535, 3007, 65535, 65535, 65535, 65535, 65535, 3008, 65535, 3009, 65535, 65535, 65535, 3010, 65535, 65535, 65535, 65535, 65535, 3011, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3012, 65535, 65535, 65535, 3013, 65535, 3014, 65535, 65535, 65535, 3015, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3016, 65535, 65535, 65535, 65535, 65535, 3017, 65535, 65535, 65535, 3018, 65535, 3019, 65535, 65535, 65535, 3020, 65535, 3021, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3022, 65535, 3023, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3024, 65535, 65535, 65535, 65535, 65535, 3025, 65535, 65535, 65535, 65535, 65535, 3026, 65535, 65535, 65535, 65535, 65535, 3027, 65535, 3028, 65535, 65535, 65535, 65535, 65535, 3029, 65535, 65535, 65535, 3030, 65535, 3031, 65535, 65535, 65535, 3032, 65535, 65535, 65535, 65535, 65535, 3033, 65535, 65535, 65535, 65535, 65535, 3034, 65535, 3035, 65535, 65535, 65535, 65535, 65535, 3036, 65535, 65535, 65535, 65535, 65535, 3037, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3038, 65535, 65535, 65535, 65535, 65535, 3039, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3040, 65535, 65535, 65535, 3041, 65535, 3042, 65535, 65535, 65535, 3043, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3044, 65535, 65535, 65535, 65535, 65535, 3045, 65535, 65535, 65535, 3046, 65535, 65535, 65535, 65535, 65535, 3047, 65535, 3048, 65535, 65535, 65535, 3049, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3050, 65535, 65535, 65535, 65535, 65535, 3051, 65535, 65535, 65535, 65535, 65535, 3052, 65535, 65535, 65535, 65535, 65535, 3053, 65535, 65535, 65535, 3054, 65535, 65535, 65535, 65535, 65535, 3055, 65535, 3056, 65535, 65535, 65535, 65535, 65535, 3057, 65535, 65535, 65535, 3058, 65535, 3059, 65535, 65535, 65535, 3060, 65535, 3061, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3062, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3063, 65535, 3064, 65535, 65535, 65535, 65535, 65535, 3065, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3066, 65535, 3067, 65535, 65535, 65535, 65535, 65535, 3068, 65535, 65535, 65535, 3069, 65535, 65535, 65535, 65535, 65535, 3070, 65535, 65535, 65535, 65535, 65535, 3071, 65535, 65535, 65535, 65535, 65535, 3072, 65535, 3073, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3074, 65535, 3075, 65535, 65535, 65535, 65535, 65535, 3076, 65535, 65535, 65535, 3077, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3078, 65535, 65535, 65535, 3079, 65535, 3080, 65535, 65535, 65535, 3081, 65535, 3082, 65535, 65535, 65535, 3083, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3084, 65535, 65535, 65535, 3085, 65535, 65535, 65535, 65535, 65535, 3086, 65535, 3087, 65535, 65535, 65535, 3088, 65535, 65535, 65535, 65535, 65535, 3089, 65535, 3090, 65535, 65535, 65535, 65535, 65535, 3091, 65535, 65535, 65535, 65535, 65535, 3092, 65535, 65535, 65535, 3093, 65535, 3094, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3095, 65535, 3096, 65535, 65535, 65535, 65535, 65535, 3097, 65535, 65535, 65535, 3098, 65535, 3099, 65535, 65535, 65535, 3100, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3101, 65535, 3102, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3103, 65535, 65535, 65535, 3104, 65535, 3105, 65535, 65535, 65535, 3106, 65535, 65535, 65535, 65535, 65535, 3107, 65535, 3108, 65535, 65535, 65535, 65535, 65535, 3109, 65535, 65535, 65535, 65535, 65535, 3110, 65535, 65535, 65535, 3111, 65535, 65535, 65535, 65535, 65535, 3112, 65535, 65535, 65535, 65535, 65535, 3113, 65535, 3114, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3115, 65535, 3116, 65535, 65535, 65535, 65535, 65535, 3117, 65535, 65535, 65535, 3118, 65535, 65535, 65535, 65535, 65535, 3119, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3120, 65535, 65535, 65535, 3121, 65535, 3122, 65535, 65535, 65535, 3123, 65535, 3124, 65535, 65535, 65535, 3125, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3126, 65535, 65535, 65535, 3127, 65535, 65535, 65535, 65535, 65535, 3128, 65535, 3129, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3130, 65535, 3131, 65535, 65535, 65535, 65535, 65535, 3132, 65535, 65535, 65535, 65535, 65535, 3133, 65535, 65535, 65535, 3134, 65535, 3135, 65535, 65535, 65535, 3136, 65535, 65535, 65535, 65535, 65535, 3137, 65535, 3138, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3139, 65535, 3140, 65535, 65535, 65535, 65535, 65535, 3141, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3142, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3143, 65535, 65535, 65535, 65535, 65535, 3144, 65535, 3145, 65535, 65535, 65535, 3146, 65535, 65535, 65535, 65535, 65535, 3147, 65535, 3148, 65535, 65535, 65535, 65535, 65535, 3149, 65535, 65535, 65535, 3150, 65535, 65535, 65535, 65535, 65535, 3151, 65535, 65535, 65535, 65535, 65535, 3152, 65535, 65535, 65535, 65535, 65535, 3153, 65535, 3154, 65535, 65535, 65535, 65535, 65535, 3155, 65535, 65535, 65535, 3156, 65535, 3157, 65535, 65535, 65535, 65535, 65535, 3158, 65535, 65535, 65535, 3159, 65535, 65535, 65535, 65535, 65535, 3160, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3161, 65535, 65535, 65535, 3162, 65535, 3163, 65535, 65535, 65535, 3164, 65535, 65535, 65535, 65535, 65535, 3165, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3166, 65535, 65535, 65535, 65535, 65535, 3167, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3168, 65535, 3169, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3170, 65535, 3171, 65535, 65535, 65535, 65535, 65535, 3172, 65535, 65535, 65535, 65535, 65535, 3173, 65535, 65535, 65535, 3174, 65535, 65535, 65535, 65535, 65535, 3175, 65535, 65535, 65535, 65535, 65535, 3176, 65535, 3177, 65535, 65535, 65535, 65535, 65535, 3178, 65535, 65535, 65535, 3179, 65535, 3180, 65535, 65535, 65535, 65535, 65535, 3181, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3182, 65535, 3183, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3184, 65535, 65535, 65535, 3185, 65535, 3186, 65535, 65535, 65535, 3187, 65535, 65535, 65535, 65535, 65535, 3188, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3189, 65535, 3190, 65535, 65535, 65535, 3191, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3192, 65535, 3193, 65535, 65535, 65535, 65535, 65535, 3194, 65535, 65535, 65535, 3195, 65535, 3196, 65535, 65535, 65535, 65535, 65535, 3197, 65535, 65535, 65535, 3198, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3199, 65535, 3200, 65535, 65535, 65535, 3201, 65535, 3202, 65535, 65535, 65535, 3203, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3204, 65535, 65535, 65535, 65535, 65535, 3205, 65535, 65535, 65535, 3206, 65535, 65535, 65535, 65535, 65535, 3207, 65535, 3208, 65535, 65535, 65535, 3209, 65535, 65535, 65535, 65535, 65535, 3210, 65535, 3211, 65535, 65535, 65535, 65535, 65535, 3212, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3213, 65535, 65535, 65535, 3214, 65535, 65535, 65535, 65535, 65535, 3215, 65535, 3216, 65535, 65535, 65535, 65535, 65535, 3217, 65535, 65535, 65535, 3218, 65535, 65535, 65535, 65535, 65535, 3219, 65535, 3220, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3221, 65535, 3222, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3223, 65535, 3224, 65535, 65535, 65535, 3225, 65535, 65535, 65535, 65535, 65535, 3226, 65535, 65535, 65535, 65535, 65535, 3227, 65535, 3228, 65535, 65535, 65535, 65535, 65535, 3229, 65535, 65535, 65535, 65535, 65535, 3230, 65535, 65535, 65535, 3231, 65535, 65535, 65535, 65535, 65535, 3232, 65535, 65535, 65535, 65535, 65535, 3233, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3234, 65535, 65535, 65535, 3235, 65535, 3236, 65535, 65535, 65535, 65535, 65535, 3237, 65535, 65535, 65535, 3238, 65535, 65535, 65535, 65535, 65535, 3239, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3240, 65535, 65535, 65535, 3241, 65535, 3242, 65535, 65535, 65535, 65535, 65535, 3243, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3244, 65535, 65535, 65535, 65535, 65535, 3245, 65535, 65535, 65535, 3246, 65535, 65535, 65535, 65535, 65535, 3247, 65535, 65535, 65535, 65535, 65535, 3248, 65535, 65535, 65535, 65535, 65535, 3249, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3250, 65535, 65535, 65535, 65535, 65535, 3251, 65535, 65535, 65535, 3252, 65535, 3253, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3254, 65535, 3255, 65535, 65535, 65535, 65535, 65535, 3256, 65535, 65535, 65535, 65535, 65535, 3257, 65535, 65535, 65535, 3258, 65535, 3259, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3260, 65535, 3261, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3262, 65535, 3263, 65535, 65535, 65535, 3264, 65535, 3265, 65535, 65535, 65535, 3266, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3267, 65535, 65535, 65535, 65535, 65535, 3268, 65535, 65535, 65535, 3269, 65535, 3270, 65535, 65535, 65535, 3271, 65535, 65535, 65535, 65535, 65535, 3272, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3273, 65535, 65535, 65535, 65535, 65535, 3274, 65535, 65535, 65535, 3275, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3276, 65535, 65535, 65535, 3277, 65535, 65535, 65535, 65535, 65535, 3278, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3279, 65535, 65535, 65535, 3280, 65535, 3281, 65535, 65535, 65535, 3282, 65535, 3283, 65535, 65535, 65535, 3284, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3285, 65535, 65535, 65535, 65535, 65535, 3286, 65535, 65535, 65535, 3287, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3288, 65535, 65535, 65535, 65535, 65535, 3289, 65535, 3290, 65535, 65535, 65535, 65535, 65535, 3291, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3292, 65535, 3293, 65535, 65535, 65535, 3294, 65535, 65535, 65535, 65535, 65535, 3295, 65535, 3296, 65535, 65535, 65535, 65535, 65535, 3297, 65535, 65535, 65535, 3298, 65535, 3299, 65535, 65535, 65535, 3300, 65535, 3301, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3302, 65535, 3303, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3304, 65535, 3305, 65535, 65535, 65535, 65535, 65535, 3306, 65535, 65535, 65535, 3307, 65535, 65535, 65535, 65535, 65535, 3308, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3309, 65535, 65535, 65535, 3310, 65535, 3311, 65535, 65535, 65535, 3312, 65535, 65535, 65535, 65535, 65535, 3313, 65535, 65535, 65535, 65535, 65535, 3314, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3315, 65535, 65535, 65535, 3316, 65535, 3317, 65535, 65535, 65535, 65535, 65535, 3318, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3319, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3320, 65535, 65535, 65535, 3321, 65535, 3322, 65535, 65535, 65535, 3323, 65535, 65535, 65535, 65535, 65535, 3324, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3325, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3326, 65535, 65535, 65535, 65535, 65535, 3327, 65535, 3328, 65535, 65535, 65535, 3329, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3330, 65535, 65535, 65535, 65535, 65535, 3331, 65535, 65535, 65535, 65535, 65535, 3332, 65535, 65535, 65535, 3333, 65535, 3334, 65535, 65535, 65535, 3335, 65535, 65535, 65535, 65535, 65535, 3336, 65535, 3337, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3338, 65535, 65535, 65535, 65535, 65535, 3339, 65535, 3340, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3341, 65535, 3342, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3343, 65535, 3344, 65535, 65535, 65535, 3345, 65535, 3346, 65535, 65535, 65535, 3347, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3348, 65535, 65535, 65535, 65535, 65535, 3349, 65535, 65535, 65535, 3350, 65535, 3351, 65535, 65535, 65535, 3352, 65535, 65535, 65535, 65535, 65535, 3353, 65535, 65535, 65535, 65535, 65535, 3354, 65535, 3355, 65535, 65535, 65535, 65535, 65535, 3356, 65535, 65535, 65535, 65535, 65535, 3357, 65535, 65535, 65535, 65535, 65535, 3358, 65535, 65535, 65535, 3359, 65535, 65535, 65535, 65535, 65535, 3360, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3361, 65535, 65535, 65535, 3362, 65535, 3363, 65535, 65535, 65535, 65535, 65535, 3364, 65535, 65535, 65535, 3365, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3366, 65535, 65535, 65535, 65535, 65535, 3367, 65535, 65535, 65535, 3368, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3369, 65535, 65535, 65535, 3370, 65535, 65535, 65535, 65535, 65535, 3371, 65535, 3372, 65535, 65535, 65535, 65535, 65535, 3373, 65535, 65535, 65535, 65535, 65535, 3374, 65535, 65535, 65535, 3375, 65535, 3376, 65535, 65535, 65535, 3377, 65535, 65535, 65535, 65535, 65535, 3378, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3379, 65535, 65535, 65535, 3380, 65535, 3381, 65535, 65535, 65535, 3382, 65535, 3383, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3384, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3385, 65535, 3386, 65535, 65535, 65535, 3387, 65535, 3388, 65535, 65535, 65535, 3389, 65535, 65535, 65535, 65535, 65535, 3390, 65535, 3391, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3392, 65535, 3393, 65535, 65535, 65535, 3394, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3395, 65535, 65535, 65535, 65535, 65535, 3396, 65535, 65535, 65535, 3397, 65535, 3398, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3399, 65535, 65535, 65535, 65535, 65535, 3400, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3401, 65535, 65535, 65535, 3402, 65535, 65535, 65535, 65535, 65535, 3403, 65535, 3404, 65535, 65535, 65535, 3405, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3406, 65535, 65535, 65535, 65535, 65535, 3407, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3408, 65535, 3409, 65535, 65535, 65535, 3410, 65535, 65535, 65535, 65535, 65535, 3411, 65535, 3412, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3413, 65535, 65535, 65535, 3414, 65535, 65535, 65535, 65535, 65535, 3415, 65535, 65535, 65535, 65535, 65535, 3416, 65535, 3417, 65535, 65535, 65535, 65535, 65535, 3418, 65535, 65535, 65535, 65535, 65535, 3419, 65535, 65535, 65535, 3420, 65535, 3421, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3422, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3423, 65535, 3424, 65535, 65535, 65535, 3425, 65535, 3426, 65535, 65535, 65535, 3427, 65535, 65535, 65535, 65535, 65535, 3428, 65535, 3429, 65535, 65535, 65535, 65535, 65535, 3430, 65535, 65535, 65535, 3431, 65535, 3432, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3433, 65535, 65535, 65535, 65535, 65535, 3434, 65535, 3435, 65535, 65535, 65535, 65535, 65535, 3436, 65535, 65535, 65535, 3437, 65535, 3438, 65535, 65535, 65535, 65535, 65535, 3439, 65535, 65535, 65535, 3440, 65535, 65535, 65535, 65535, 65535, 3441, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3442, 65535, 65535, 65535, 3443, 65535, 65535, 65535, 65535, 65535, 3444, 65535, 3445, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3446, 65535, 65535, 65535, 65535, 65535, 3447, 65535, 65535, 65535, 3448, 65535, 65535, 65535, 65535, 65535, 3449, 65535, 3450, 65535, 65535, 65535, 3451, 65535, 65535, 65535, 65535, 65535, 3452, 65535, 3453, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3454, 65535, 65535, 65535, 3455, 65535, 3456, 65535, 65535, 65535, 3457, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3458, 65535, 65535, 65535, 3459, 65535, 3460, 65535, 65535, 65535, 3461, 65535, 3462, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3463, 65535, 3464, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3465, 65535, 65535, 65535, 65535, 65535, 3466, 65535, 3467, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3468, 65535, 3469, 65535, 65535, 65535, 65535, 65535, 3470, 65535, 65535, 65535, 3471, 65535, 3472, 65535, 65535, 65535, 3473, 65535, 65535, 65535, 65535, 65535, 3474, 65535, 65535, 65535, 65535, 65535, 3475, 65535, 3476, 65535, 65535, 65535, 65535, 65535, 3477, 65535, 65535, 65535, 3478, 65535, 3479, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3480, 65535, 65535, 65535, 65535, 65535, 3481, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3482, 65535, 65535, 65535, 65535, 65535, 3483, 65535, 65535, 65535, 3484, 65535, 3485, 65535, 65535, 65535, 3486, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3487, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3488, 65535, 65535, 65535, 65535, 65535, 3489, 65535, 3490, 65535, 65535, 65535, 3491, 65535, 65535, 65535, 65535, 65535, 3492, 65535, 3493, 65535, 65535, 65535, 65535, 65535, 3494, 65535, 65535, 65535, 65535, 65535, 3495, 65535, 65535, 65535, 65535, 65535, 3496, 65535, 65535, 65535, 3497, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3498, 65535, 65535, 65535, 65535, 65535, 3499, 65535, 65535, 65535, 3500, 65535, 3501, 65535, 65535, 65535, 3502, 65535, 3503, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3504, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3505, 65535, 3506, 65535, 65535, 65535, 3507, 65535, 3508, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3509, 65535, 3510, 65535, 65535, 65535, 65535, 65535, 3511, 65535, 65535, 65535, 3512, 65535, 3513, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3514, 65535, 65535, 65535, 65535, 65535, 3515, 65535, 3516, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3517, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3518, 65535, 65535, 65535, 3519, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3520, 65535, 65535, 65535, 3521, 65535, 3522, 65535, 65535, 65535, 3523, 65535, 3524, 65535, 65535, 65535, 3525, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3526, 65535, 65535, 65535, 65535, 65535, 3527, 65535, 65535, 65535, 3528, 65535, 65535, 65535, 65535, 65535, 3529, 65535, 3530, 65535, 65535, 65535, 3531, 65535, 65535, 65535, 65535, 65535, 3532, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3533, 65535, 65535, 65535, 65535, 65535, 3534, 65535, 65535, 65535, 3535, 65535, 3536, 65535, 65535, 65535, 3537, 65535, 65535, 65535, 65535, 65535, 3538, 65535, 3539, 65535, 65535, 65535, 65535, 65535, 3540, 65535, 65535, 65535, 3541, 65535, 3542, 65535, 65535, 65535, 3543, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3544, 65535, 3545, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3546, 65535, 65535, 65535, 3547, 65535, 65535, 65535, 65535, 65535, 3548, 65535, 65535, 65535, 65535, 65535, 3549, 65535, 3550, 65535, 65535, 65535, 65535, 65535, 3551, 65535, 65535, 65535, 3552, 65535, 3553, 65535, 65535, 65535, 3554, 65535, 65535, 65535, 65535, 65535, 3555, 65535, 65535, 65535, 65535, 65535, 3556, 65535, 3557, 65535, 65535, 65535, 65535, 65535, 3558, 65535, 65535, 65535, 65535, 65535, 3559, 65535, 65535, 65535, 65535, 65535, 3560, 65535, 65535, 65535, 3561, 65535, 65535, 65535, 65535, 65535, 3562, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3563, 65535, 3564, 65535, 65535, 65535, 3565, 65535, 3566, 65535, 65535, 65535, 3567, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3568, 65535, 65535, 65535, 3569, 65535, 65535, 65535, 65535, 65535, 3570, 65535, 3571, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3572, 65535, 65535, 65535, 65535, 65535, 3573, 65535, 65535, 65535, 65535, 65535, 3574, 65535, 65535, 65535, 3575, 65535, 3576, 65535, 65535, 65535, 3577, 65535, 65535, 65535, 65535, 65535, 3578, 65535, 3579, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3580, 65535, 3581, 65535, 65535, 65535, 3582, 65535, 3583, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3584, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3585, 65535, 3586, 65535, 65535, 65535, 65535, 65535, 3587, 65535, 65535, 65535, 3588, 65535, 65535, 65535, 65535, 65535, 3589, 65535, 3590, 65535, 65535, 65535, 65535, 65535, 3591, 65535, 65535, 65535, 3592, 65535, 65535, 65535, 65535, 65535, 3593, 65535, 65535, 65535, 65535, 65535, 3594, 65535, 65535, 65535, 65535, 65535, 3595, 65535, 3596, 65535, 65535, 65535, 65535, 65535, 3597, 65535, 65535, 65535, 3598, 65535, 3599, 65535, 65535, 65535, 65535, 65535, 3600, 65535, 65535, 65535, 3601, 65535, 65535, 65535, 65535, 65535, 3602, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3603, 65535, 65535, 65535, 3604, 65535, 3605, 65535, 65535, 65535, 3606, 65535, 65535, 65535, 65535, 65535, 3607, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3608, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3609, 65535, 3610, 65535, 65535, 65535, 3611, 65535, 65535, 65535, 65535, 65535, 3612, 65535, 3613, 65535, 65535, 65535, 65535, 65535, 3614, 65535, 65535, 65535, 65535, 65535, 3615, 65535, 65535, 65535, 3616, 65535, 3617, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3618, 65535, 3619, 65535, 65535, 65535, 65535, 65535, 3620, 65535, 65535, 65535, 3621, 65535, 3622, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3623, 65535, 3624, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3625, 65535, 3626, 65535, 65535, 65535, 3627, 65535, 3628, 65535, 65535, 65535, 3629, 65535, 65535, 65535, 65535, 65535, 3630, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3631, 65535, 65535, 65535, 65535, 65535, 3632, 65535, 65535, 65535, 3633, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3634, 65535, 3635, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3636, 65535, 3637, 65535, 65535, 65535, 65535, 65535, 3638, 65535, 65535, 65535, 3639, 65535, 65535, 65535, 65535, 65535, 3640, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3641, 65535, 3642, 65535, 65535, 65535, 3643, 65535, 3644, 65535, 65535, 65535, 3645, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3646, 65535, 65535, 65535, 65535, 65535, 3647, 65535, 65535, 65535, 3648, 65535, 65535, 65535, 65535, 65535, 3649, 65535, 3650, 65535, 65535, 65535, 3651, 65535, 65535, 65535, 65535, 65535, 3652, 65535, 3653, 65535, 65535, 65535, 65535, 65535, 3654, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3655, 65535, 3656, 65535, 65535, 65535, 3657, 65535, 65535, 65535, 65535, 65535, 3658, 65535, 3659, 65535, 65535, 65535, 65535, 65535, 3660, 65535, 65535, 65535, 3661, 65535, 3662, 65535, 65535, 65535, 65535, 65535, 3663, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3664, 65535, 3665, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3666, 65535, 65535, 65535, 65535, 65535, 3667, 65535, 65535, 65535, 65535, 65535, 3668, 65535, 65535, 65535, 65535, 65535, 3669, 65535, 3670, 65535, 65535, 65535, 65535, 65535, 3671, 65535, 65535, 65535, 65535, 65535, 3672, 65535, 65535, 65535, 3673, 65535, 65535, 65535, 65535, 65535, 3674, 65535, 65535, 65535, 65535, 65535, 3675, 65535, 3676, 65535, 65535, 65535, 65535, 65535, 3677, 65535, 65535, 65535, 3678, 65535, 3679, 65535, 65535, 65535, 65535, 65535, 3680, 65535, 65535, 65535, 3681, 65535, 65535, 65535, 65535, 65535, 3682, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3683, 65535, 65535, 65535, 3684, 65535, 3685, 65535, 65535, 65535, 65535, 65535, 3686, 65535, 65535, 65535, 3687, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3688, 65535, 65535, 65535, 65535, 65535, 3689, 65535, 65535, 65535, 3690, 65535, 65535, 65535, 65535, 65535, 3691, 65535, 3692, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3693, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3694, 65535, 65535, 65535, 65535, 65535, 3695, 65535, 65535, 65535, 3696, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3697, 65535, 3698, 65535, 65535, 65535, 65535, 65535, 3699, 65535, 65535, 65535, 3700, 65535, 3701, 65535, 65535, 65535, 3702, 65535, 3703, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3704, 65535, 3705, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3706, 65535, 65535, 65535, 3707, 65535, 3708, 65535, 65535, 65535, 3709, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3710, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3711, 65535, 3712, 65535, 65535, 65535, 3713, 65535, 65535, 65535, 65535, 65535, 3714, 65535, 65535, 65535, 65535, 65535, 3715, 65535, 3716, 65535, 65535, 65535, 65535, 65535, 3717, 65535, 65535, 65535, 3718, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3719, 65535, 65535, 65535, 3720, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3721, 65535, 65535, 65535, 3722, 65535, 3723, 65535, 65535, 65535, 3724, 65535, 3725, 65535, 65535, 65535, 3726, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3727, 65535, 65535, 65535, 65535, 65535, 3728, 65535, 65535, 65535, 3729, 65535, 65535, 65535, 65535, 65535, 3730, 65535, 65535, 65535, 65535, 65535, 3731, 65535, 65535, 65535, 65535, 65535, 3732, 65535, 3733, 65535, 65535, 65535, 65535, 65535, 3734, 65535, 65535, 65535, 65535, 65535, 3735, 65535, 65535, 65535, 65535, 65535, 3736, 65535, 65535, 65535, 3737, 65535, 65535, 65535, 65535, 65535, 3738, 65535, 3739, 65535, 65535, 65535, 65535, 65535, 3740, 65535, 65535, 65535, 3741, 65535, 65535, 65535, 65535, 65535, 3742, 65535, 3743, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3744, 65535, 3745, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3746, 65535, 3747, 65535, 65535, 65535, 65535, 65535, 3748, 65535, 65535, 65535, 3749, 65535, 65535, 65535, 65535, 65535, 3750, 65535, 3751, 65535, 65535, 65535, 65535, 65535, 3752, 65535, 65535, 65535, 3753, 65535, 3754, 65535, 65535, 65535, 3755, 65535, 65535, 65535, 65535, 65535, 3756, 65535, 65535, 65535, 65535, 65535, 3757, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3758, 65535, 65535, 65535, 3759, 65535, 3760, 65535, 65535, 65535, 65535, 65535, 3761, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3762, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3763, 65535, 65535, 65535, 3764, 65535, 3765, 65535, 65535, 65535, 3766, 65535, 3767, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3768, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3769, 65535, 65535, 65535, 65535, 65535, 3770, 65535, 65535, 65535, 65535, 65535, 3771, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3772, 65535, 65535, 65535, 65535, 65535, 3773, 65535, 65535, 65535, 65535, 65535, 3774, 65535, 65535, 65535, 3775, 65535, 3776, 65535, 65535, 65535, 3777, 65535, 65535, 65535, 65535, 65535, 3778, 65535, 3779, 65535, 65535, 65535, 65535, 65535, 3780, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3781, 65535, 3782, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3783, 65535, 3784, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3785, 65535, 3786, 65535, 65535, 65535, 3787, 65535, 3788, 65535, 65535, 65535, 3789, 65535, 65535, 65535, 65535, 65535, 3790, 65535, 3791, 65535, 65535, 65535, 65535, 65535, 3792, 65535, 65535, 65535, 3793, 65535, 3794, 65535, 65535, 65535, 3795, 65535, 65535, 65535, 65535, 65535, 3796, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3797, 65535, 65535, 65535, 65535, 65535, 3798, 65535, 65535, 65535, 65535, 65535, 3799, 65535, 65535, 65535, 65535, 65535, 3800, 65535, 65535, 65535, 3801, 65535, 65535, 65535, 65535, 65535, 3802, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3803, 65535, 65535, 65535, 3804, 65535, 3805, 65535, 65535, 65535, 3806, 65535, 3807, 65535, 65535, 65535, 3808, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3809, 65535, 65535, 65535, 65535, 65535, 3810, 65535, 65535, 65535, 3811, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3812, 65535, 65535, 65535, 3813, 65535, 65535, 65535, 65535, 65535, 3814, 65535, 3815, 65535, 65535, 65535, 65535, 65535, 3816, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3817, 65535, 3818, 65535, 65535, 65535, 3819, 65535, 65535, 65535, 65535, 65535, 3820, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3821, 65535, 65535, 65535, 3822, 65535, 3823, 65535, 65535, 65535, 3824, 65535, 3825, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3826, 65535, 3827, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3828, 65535, 3829, 65535, 65535, 65535, 3830, 65535, 3831, 65535, 65535, 65535, 3832, 65535, 65535, 65535, 65535, 65535, 3833, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3834, 65535, 3835, 65535, 65535, 65535, 3836, 65535, 65535, 65535, 65535, 65535, 3837, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3838, 65535, 65535, 65535, 65535, 65535, 3839, 65535, 65535, 65535, 3840, 65535, 3841, 65535, 65535, 65535, 65535, 65535, 3842, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3843, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3844, 65535, 65535, 65535, 3845, 65535, 65535, 65535, 65535, 65535, 3846, 65535, 65535, 65535, 65535, 65535, 3847, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3848, 65535, 65535, 65535, 65535, 65535, 3849, 65535, 65535, 65535, 3850, 65535, 65535, 65535, 65535, 65535, 3851, 65535, 3852, 65535, 65535, 65535, 3853, 65535, 65535, 65535, 65535, 65535, 3854, 65535, 3855, 65535, 65535, 65535, 65535, 65535, 3856, 65535, 65535, 65535, 65535, 65535, 3857, 65535, 65535, 65535, 3858, 65535, 65535, 65535, 65535, 65535, 3859, 65535, 65535, 65535, 65535, 65535, 3860, 65535, 3861, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3862, 65535, 65535, 65535, 3863, 65535, 3864, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3865, 65535, 3866, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3867, 65535, 3868, 65535, 65535, 65535, 3869, 65535, 3870, 65535, 65535, 65535, 3871, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3872, 65535, 65535, 65535, 65535, 65535, 3873, 65535, 65535, 65535, 3874, 65535, 3875, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3876, 65535, 65535, 65535, 65535, 65535, 3877, 65535, 3878, 65535, 65535, 65535, 65535, 65535, 3879, 65535, 65535, 65535, 3880, 65535, 3881, 65535, 65535, 65535, 65535, 65535, 3882, 65535, 65535, 65535, 3883, 65535, 65535, 65535, 65535, 65535, 3884, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3885, 65535, 65535, 65535, 3886, 65535, 3887, 65535, 65535, 65535, 65535, 65535, 3888, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3889, 65535, 65535, 65535, 65535, 65535, 3890, 65535, 65535, 65535, 3891, 65535, 65535, 65535, 65535, 65535, 3892, 65535, 3893, 65535, 65535, 65535, 3894, 65535, 65535, 65535, 65535, 65535, 3895, 65535, 3896, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3897, 65535, 65535, 65535, 3898, 65535, 3899, 65535, 65535, 65535, 3900, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3901, 65535, 65535, 65535, 65535, 65535, 3902, 65535, 65535, 65535, 3903, 65535, 3904, 65535, 65535, 65535, 3905, 65535, 3906, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3907, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3908, 65535, 65535, 65535, 65535, 65535, 3909, 65535, 3910, 65535, 65535, 65535, 3911, 65535, 65535, 65535, 65535, 65535, 3912, 65535, 3913, 65535, 65535, 65535, 65535, 65535, 3914, 65535, 65535, 65535, 3915, 65535, 3916, 65535, 65535, 65535, 3917, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3918, 65535, 3919, 65535, 65535, 65535, 65535, 65535, 3920, 65535, 65535, 65535, 3921, 65535, 3922, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3923, 65535, 65535, 65535, 65535, 65535, 3924, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3925, 65535, 65535, 65535, 65535, 65535, 3926, 65535, 65535, 65535, 3927, 65535, 3928, 65535, 65535, 65535, 3929, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3930, 65535, 65535, 65535, 65535, 65535, 3931, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3932, 65535, 3933, 65535, 65535, 65535, 3934, 65535, 65535, 65535, 65535, 65535, 3935, 65535, 3936, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3937, 65535, 65535, 65535, 65535, 65535, 3938, 65535, 65535, 65535, 3939, 65535, 65535, 65535, 65535, 65535, 3940, 65535, 3941, 65535, 65535, 65535, 65535, 65535, 3942, 65535, 65535, 65535, 3943, 65535, 3944, 65535, 65535, 65535, 3945, 65535, 3946, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3947, 65535, 3948, 65535, 65535, 65535, 3949, 65535, 3950, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3951, 65535, 3952, 65535, 65535, 65535, 65535, 65535, 3953, 65535, 65535, 65535, 3954, 65535, 3955, 65535, 65535, 65535, 3956, 65535, 65535, 65535, 65535, 65535, 3957, 65535, 65535, 65535, 65535, 65535, 3958, 65535, 3959, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3960, 65535, 3961, 65535, 65535, 65535, 65535, 65535, 3962, 65535, 65535, 65535, 3963, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3964, 65535, 65535, 65535, 3965, 65535, 65535, 65535, 65535, 65535, 3966, 65535, 3967, 65535, 65535, 65535, 3968, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3969, 65535, 65535, 65535, 65535, 65535, 3970, 65535, 65535, 65535, 3971, 65535, 65535, 65535, 65535, 65535, 3972, 65535, 3973, 65535, 65535, 65535, 3974, 65535, 65535, 65535, 65535, 65535, 3975, 65535, 3976, 65535, 65535, 65535, 65535, 65535, 3977, 65535, 65535, 65535, 65535, 65535, 3978, 65535, 65535, 65535, 3979, 65535, 3980, 65535, 65535, 65535, 3981, 65535, 65535, 65535, 65535, 65535, 3982, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3983, 65535, 65535, 65535, 3984, 65535, 3985, 65535, 65535, 65535, 3986, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3987, 65535, 3988, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3989, 65535, 65535, 65535, 3990, 65535, 3991, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3992, 65535, 3993, 65535, 65535, 65535, 65535, 65535, 3994, 65535, 65535, 65535, 3995, 65535, 3996, 65535, 65535, 65535, 3997, 65535, 65535, 65535, 65535, 65535, 3998, 65535, 65535, 65535, 65535, 65535, 3999, 65535, 4000, 65535, 65535, 65535, 65535, 65535, 4001, 65535, 65535, 65535, 4002, 65535, 4003, 65535, 65535, 65535, 65535, 65535, 4004, 65535, 65535, 65535, 4005, 65535, 65535, 65535, 65535, 65535, 4006, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4007, 65535, 65535, 65535, 65535, 65535, 4008, 65535, 65535, 65535, 4009, 65535, 4010, 65535, 65535, 65535, 4011, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4012, 65535, 65535, 65535, 65535, 65535, 4013, 65535, 4014, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4015, 65535, 4016, 65535, 65535, 65535, 65535, 65535, 4017, 65535, 65535, 65535, 65535, 65535, 4018, 65535, 65535, 65535, 4019, 65535, 4020, 65535, 65535, 65535, 4021, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4022, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4023, 65535, 4024, 65535, 65535, 65535, 4025, 65535, 4026, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4027, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4028, 65535, 4029, 65535, 65535, 65535, 4030, 65535, 4031, 65535, 65535, 65535, 4032, 65535, 65535, 65535, 65535, 65535, 4033, 65535, 4034, 65535, 65535, 65535, 65535, 65535, 4035, 65535, 65535, 65535, 4036, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4037, 65535, 65535, 65535, 65535, 65535, 4038, 65535, 4039, 65535, 65535, 65535, 65535, 65535, 4040, 65535, 65535, 65535, 4041, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4042, 65535, 65535, 65535, 4043, 65535, 65535, 65535, 65535, 65535, 4044, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4045, 65535, 65535, 65535, 4046, 65535, 4047, 65535, 65535, 65535, 4048, 65535, 65535, 65535, 65535, 65535, 4049, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4050, 65535, 65535, 65535, 65535, 65535, 4051, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4052, 65535, 4053, 65535, 65535, 65535, 4054, 65535, 65535, 65535, 65535, 65535, 4055, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4056, 65535, 65535, 65535, 65535, 65535, 4057, 65535, 65535, 65535, 4058, 65535, 4059, 65535, 65535, 65535, 4060, 65535, 65535, 65535, 65535, 65535, 4061, 65535, 4062, 65535, 65535, 65535, 65535, 65535, 4063, 65535, 65535, 65535, 4064, 65535, 4065, 65535, 65535, 65535, 65535, 65535, 4066, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4067, 65535, 4068, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4069, 65535, 4070, 65535, 65535, 65535, 4071, 65535, 65535, 65535, 65535, 65535, 4072, 65535, 65535, 65535, 65535, 65535, 4073, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4074, 65535, 65535, 65535, 4075, 65535, 4076, 65535, 65535, 65535, 4077, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4078, 65535, 4079, 65535, 65535, 65535, 65535, 65535, 4080, 65535, 65535, 65535, 65535, 65535, 4081, 65535, 65535, 65535, 65535, 65535, 4082, 65535, 65535, 65535, 4083, 65535, 65535, 65535, 65535, 65535, 4084, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4085, 65535, 4086, 65535, 65535, 65535, 4087, 65535, 4088, 65535, 65535, 65535, 4089, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4090, 65535, 65535, 65535, 65535, 65535, 4091, 65535, 65535, 65535, 4092, 65535, 65535, 65535, 65535, 65535, 4093, 65535, 4094, 65535, 65535, 65535, 4095, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4096, 65535, 65535, 65535, 65535, 65535, 4097, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4098, 65535, 4099, 65535, 65535, 65535, 4100, 65535, 65535, 65535, 65535, 65535, 4101, 65535, 4102, 65535, 65535, 65535, 65535, 65535, 4103, 65535, 65535, 65535, 4104, 65535, 4105, 65535, 65535, 65535, 4106, 65535, 4107, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4108, 65535, 4109, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4110, 65535, 4111, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4112, 65535, 65535, 65535, 65535, 65535, 4113, 65535, 4114, 65535, 65535, 65535, 65535, 65535, 4115, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4116, 65535, 65535, 65535, 65535, 65535, 4117, 65535, 65535, 65535, 65535, 65535, 4118, 65535, 4119, 65535, 65535, 65535, 65535, 65535, 4120, 65535, 65535, 65535, 4121, 65535, 4122, 65535, 65535, 65535, 65535, 65535, 4123, 65535, 65535, 65535, 4124, 65535, 65535, 65535, 65535, 65535, 4125, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4126, 65535, 65535, 65535, 4127, 65535, 4128, 65535, 65535, 65535, 65535, 65535, 4129, 65535, 65535, 65535, 4130, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4131, 65535, 65535, 65535, 4132, 65535, 65535, 65535, 65535, 65535, 4133, 65535, 4134, 65535, 65535, 65535, 4135, 65535, 65535, 65535, 65535, 65535, 4136, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4137, 65535, 65535, 65535, 65535, 65535, 4138, 65535, 65535, 65535, 4139, 65535, 4140, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4141, 65535, 4142, 65535, 65535, 65535, 65535, 65535, 4143, 65535, 65535, 65535, 4144, 65535, 4145, 65535, 65535, 65535, 4146, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4147, 65535, 4148, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4149, 65535, 4150, 65535, 65535, 65535, 4151, 65535, 4152, 65535, 65535, 65535, 4153, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4154, 65535, 65535, 65535, 65535, 65535, 4155, 65535, 65535, 65535, 65535, 65535, 4156, 65535, 65535, 65535, 4157, 65535, 65535, 65535, 65535, 65535, 4158, 65535, 65535, 65535, 65535, 65535, 4159, 65535, 4160, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4161, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4162, 65535, 65535, 65535, 4163, 65535, 65535, 65535, 65535, 65535, 4164, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4165, 65535, 65535, 65535, 4166, 65535, 4167, 65535, 65535, 65535, 4168, 65535, 4169, 65535, 65535, 65535, 4170, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4171, 65535, 65535, 65535, 65535, 65535, 4172, 65535, 65535, 65535, 4173, 65535, 65535, 65535, 65535, 65535, 4174, 65535, 65535, 65535, 65535, 65535, 4175, 65535, 65535, 65535, 65535, 65535, 4176, 65535, 4177, 65535, 65535, 65535, 65535, 65535, 4178, 65535, 65535, 65535, 65535, 65535, 4179, 65535, 65535, 65535, 4180, 65535, 4181, 65535, 65535, 65535, 4182, 65535, 65535, 65535, 65535, 65535, 4183, 65535, 4184, 65535, 65535, 65535, 65535, 65535, 4185, 65535, 65535, 65535, 4186, 65535, 4187, 65535, 65535, 65535, 65535, 65535, 4188, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4189, 65535, 4190, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4191, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4192, 65535, 65535, 65535, 4193, 65535, 65535, 65535, 65535, 65535, 4194, 65535, 4195, 65535, 65535, 65535, 65535, 65535, 4196, 65535, 65535, 65535, 4197, 65535, 4198, 65535, 65535, 65535, 4199, 65535, 65535, 65535, 65535, 65535, 4200, 65535, 65535, 65535, 65535, 65535, 4201, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4202, 65535, 65535, 65535, 4203, 65535, 4204, 65535, 65535, 65535, 65535, 65535, 4205, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4206, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4207, 65535, 65535, 65535, 4208, 65535, 4209, 65535, 65535, 65535, 4210, 65535, 4211, 65535, 65535, 65535, 4212, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4213, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4214, 65535, 65535, 65535, 65535, 65535, 4215, 65535, 4216, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4217, 65535, 65535, 65535, 65535, 65535, 4218, 65535, 65535, 65535, 65535, 65535, 4219, 65535, 65535, 65535, 4220, 65535, 65535, 65535, 65535, 65535, 4221, 65535, 65535, 65535, 65535, 65535, 4222, 65535, 4223, 65535, 65535, 65535, 65535, 65535, 4224, 65535, 65535, 65535, 4225, 65535, 65535, 65535, 65535, 65535, 4226, 65535, 4227, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4228, 65535, 4229, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4230, 65535, 65535, 65535, 4231, 65535, 4232, 65535, 65535, 65535, 4233, 65535, 65535, 65535, 65535, 65535, 4234, 65535, 4235, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4236, 65535, 4237, 65535, 65535, 65535, 4238, 65535, 65535, 65535, 65535, 65535, 4239, 65535, 65535, 65535, 65535, 65535, 4240, 65535, 4241, 65535, 65535, 65535, 65535, 65535, 4242, 65535, 65535, 65535, 65535, 65535, 4243, 65535, 65535, 65535, 65535, 65535, 4244, 65535, 65535, 65535, 4245, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4246, 65535, 65535, 65535, 4247, 65535, 4248, 65535, 65535, 65535, 4249, 65535, 4250, 65535, 65535, 65535, 4251, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4252, 65535, 65535, 65535, 65535, 65535, 4253, 65535, 65535, 65535, 4254, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4255, 65535, 65535, 65535, 4256, 65535, 65535, 65535, 65535, 65535, 4257, 65535, 4258, 65535, 65535, 65535, 65535, 65535, 4259, 65535, 65535, 65535, 65535, 65535, 4260, 65535, 65535, 65535, 65535, 65535, 4261, 65535, 65535, 65535, 4262, 65535, 65535, 65535, 65535, 65535, 4263, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4264, 65535, 65535, 65535, 4265, 65535, 65535, 65535, 65535, 65535, 4266, 65535, 4267, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4268, 65535, 4269, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4270, 65535, 4271, 65535, 65535, 65535, 4272, 65535, 4273, 65535, 65535, 65535, 4274, 65535, 65535, 65535, 65535, 65535, 4275, 65535, 4276, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4277, 65535, 4278, 65535, 65535, 65535, 4279, 65535, 65535, 65535, 65535, 65535, 4280, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4281, 65535, 65535, 65535, 4282, 65535, 4283, 65535, 65535, 65535, 65535, 65535, 4284, 65535, 65535, 65535, 4285, 65535, 65535, 65535, 65535, 65535, 4286, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4287, 65535, 65535, 65535, 4288, 65535, 65535, 65535, 65535, 65535, 4289, 65535, 4290, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4291, 65535, 65535, 65535, 65535, 65535, 4292, 65535, 65535, 65535, 4293, 65535, 65535, 65535, 65535, 65535, 4294, 65535, 65535, 65535, 65535, 65535, 4295, 65535, 65535, 65535, 65535, 65535, 4296, 65535, 4297, 65535, 65535, 65535, 65535, 65535, 4298, 65535, 65535, 65535, 65535, 65535, 4299, 65535, 65535, 65535, 4300, 65535, 65535, 65535, 65535, 65535, 4301, 65535, 65535, 65535, 65535, 65535, 4302, 65535, 4303, 65535, 65535, 65535, 65535, 65535, 4304, 65535, 65535, 65535, 65535, 65535, 4305, 65535, 65535, 65535, 4306, 65535, 4307, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4308, 65535, 4309, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4310, 65535, 4311, 65535, 65535, 65535, 4312, 65535, 4313, 65535, 65535, 65535, 4314, 65535, 65535, 65535, 65535, 65535, 4315, 65535, 4316, 65535, 65535, 65535, 65535, 65535, 4317, 65535, 65535, 65535, 4318, 65535, 4319, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4320, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4321, 65535, 65535, 65535, 65535, 65535, 4322, 65535, 65535, 65535, 4323, 65535, 4324, 65535, 65535, 65535, 65535, 65535, 4325, 65535, 65535, 65535, 4326, 65535, 65535, 65535, 65535, 65535, 4327, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4328, 65535, 65535, 65535, 4329, 65535, 4330, 65535, 65535, 65535, 4331, 65535, 4332, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4333, 65535, 65535, 65535, 65535, 65535, 4334, 65535, 65535, 65535, 4335, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4336, 65535, 65535, 65535, 4337, 65535, 65535, 65535, 65535, 65535, 4338, 65535, 4339, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4340, 65535, 4341, 65535, 65535, 65535, 4342, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4343, 65535, 65535, 65535, 65535, 65535, 4344, 65535, 65535, 65535, 4345, 65535, 4346, 65535, 65535, 65535, 4347, 65535, 4348, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4349, 65535, 4350, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4351, 65535, 65535, 65535, 65535, 65535, 4352, 65535, 4353, 65535, 65535, 65535, 4354, 65535, 65535, 65535, 65535, 65535, 4355, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4356, 65535, 65535, 65535, 4357, 65535, 4358, 65535, 65535, 65535, 4359, 65535, 65535, 65535, 65535, 65535, 4360, 65535, 65535, 65535, 65535, 65535, 4361, 65535, 4362, 65535, 65535, 65535, 65535, 65535, 4363, 65535, 65535, 65535, 4364, 65535, 4365, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4366, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4367, 65535, 65535, 65535, 65535, 65535, 4368, 65535, 65535, 65535, 4369, 65535, 65535, 65535, 65535, 65535, 4370, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4371, 65535, 65535, 65535, 65535, 65535, 4372, 65535, 65535, 65535, 4373, 65535, 65535, 65535, 65535, 65535, 4374, 65535, 4375, 65535, 65535, 65535, 4376, 65535, 65535, 65535, 65535, 65535, 4377, 65535, 4378, 65535, 65535, 65535, 65535, 65535, 4379, 65535, 65535, 65535, 65535, 65535, 4380, 65535, 65535, 65535, 65535, 65535, 4381, 65535, 65535, 65535, 4382, 65535, 65535, 65535, 65535, 65535, 4383, 65535, 4384, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4385, 65535, 4386, 65535, 65535, 65535, 4387, 65535, 4388, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4389, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4390, 65535, 4391, 65535, 65535, 65535, 4392, 65535, 4393, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4394, 65535, 65535, 65535, 65535, 65535, 4395, 65535, 65535, 65535, 4396, 65535, 4397, 65535, 65535, 65535, 4398, 65535, 65535, 65535, 65535, 65535, 4399, 65535, 65535, 65535, 65535, 65535, 4400, 65535, 4401, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4402, 65535, 4403, 65535, 65535, 65535, 65535, 65535, 4404, 65535, 65535, 65535, 4405, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4406, 65535, 65535, 65535, 4407, 65535, 4408, 65535, 65535, 65535, 65535, 65535, 4409, 65535, 65535, 65535, 4410, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4411, 65535, 65535, 65535, 65535, 65535, 4412, 65535, 65535, 65535, 4413, 65535, 65535, 65535, 65535, 65535, 4414, 65535, 4415, 65535, 65535, 65535, 4416, 65535, 65535, 65535, 65535, 65535, 4417, 65535, 4418, 65535, 65535, 65535, 65535, 65535, 4419, 65535, 65535, 65535, 65535, 65535, 4420, 65535, 65535, 65535, 4421, 65535, 4422, 65535, 65535, 65535, 4423, 65535, 65535, 65535, 65535, 65535, 4424, 65535, 4425, 65535, 65535, 65535, 65535, 65535, 4426, 65535, 65535, 65535, 4427, 65535, 4428, 65535, 65535, 65535, 4429, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4430, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4431, 65535, 65535, 65535, 4432, 65535, 4433, 65535, 65535, 65535, 4434, 65535, 65535, 65535, 65535, 65535, 4435, 65535, 4436, 65535, 65535, 65535, 65535, 65535, 4437, 65535, 65535, 65535, 4438, 65535, 4439, 65535, 65535, 65535, 4440, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4441, 65535, 4442, 65535, 65535, 65535, 65535, 65535, 4443, 65535, 65535, 65535, 4444, 65535, 4445, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4446, 65535, 65535, 65535, 65535, 65535, 4447, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4448, 65535, 65535, 65535, 4449, 65535, 4450, 65535, 65535, 65535, 4451, 65535, 4452, 65535, 65535, 65535, 4453, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4454, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4455, 65535, 4456, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4457, 65535, 4458, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4459, 65535, 65535, 65535, 4460, 65535, 4461, 65535, 65535, 65535, 4462, 65535, 65535, 65535, 65535, 65535, 4463, 65535, 4464, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4465, 65535, 4466, 65535, 65535, 65535, 4467, 65535, 4468, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4469, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4470, 65535, 4471, 65535, 65535, 65535, 4472, 65535, 4473, 65535, 65535, 65535, 4474, 65535, 65535, 65535, 65535, 65535, 4475, 65535, 4476, 65535, 65535, 65535, 65535, 65535, 4477, 65535, 65535, 65535, 4478, 65535, 65535, 65535, 65535, 65535, 4479, 65535, 65535, 65535, 65535, 65535, 4480, 65535, 65535, 65535, 65535, 65535, 4481, 65535, 4482, 65535, 65535, 65535, 65535, 65535, 4483, 65535, 65535, 65535, 4484, 65535, 4485, 65535, 65535, 65535, 65535, 65535, 4486, 65535, 65535, 65535, 4487, 65535, 65535, 65535, 65535, 65535, 4488, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4489, 65535, 65535, 65535, 4490, 65535, 65535, 65535, 65535, 65535, 4491, 65535, 65535, 65535, 65535, 65535, 4492, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4493, 65535, 65535, 65535, 65535, 65535, 4494, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4495, 65535, 4496, 65535, 65535, 65535, 4497, 65535, 65535, 65535, 65535, 65535, 4498, 65535, 4499, 65535, 65535, 65535, 65535, 65535, 4500, 65535, 65535, 65535, 65535, 65535, 4501, 65535, 65535, 65535, 4502, 65535, 4503, 65535, 65535, 65535, 4504, 65535, 65535, 65535, 65535, 65535, 4505, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4506, 65535, 65535, 65535, 4507, 65535, 4508, 65535, 65535, 65535, 65535, 65535, 4509, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4510, 65535, 4511, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4512, 65535, 4513, 65535, 65535, 65535, 4514, 65535, 4515, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4516, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4517, 65535, 65535, 65535, 4518, 65535, 4519, 65535, 65535, 65535, 4520, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4521, 65535, 4522, 65535, 65535, 65535, 65535, 65535, 4523, 65535, 65535, 65535, 4524, 65535, 4525, 65535, 65535, 65535, 65535, 65535, 4526, 65535, 65535, 65535, 4527, 65535, 65535, 65535, 65535, 65535, 4528, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4529, 65535, 65535, 65535, 4530, 65535, 4531, 65535, 65535, 65535, 4532, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4533, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4534, 65535, 65535, 65535, 65535, 65535, 4535, 65535, 4536, 65535, 65535, 65535, 4537, 65535, 65535, 65535, 65535, 65535, 4538, 65535, 4539, 65535, 65535, 65535, 65535, 65535, 4540, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4541, 65535, 4542, 65535, 65535, 65535, 4543, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4544, 65535, 65535, 65535, 65535, 65535, 4545, 65535, 65535, 65535, 4546, 65535, 4547, 65535, 65535, 65535, 4548, 65535, 4549, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4550, 65535, 4551, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4552, 65535, 4553, 65535, 65535, 65535, 4554, 65535, 65535, 65535, 65535, 65535, 4555, 65535, 65535, 65535, 65535, 65535, 4556, 65535, 4557, 65535, 65535, 65535, 65535, 65535, 4558, 65535, 65535, 65535, 65535, 65535, 4559, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4560, 65535, 65535, 65535, 65535, 65535, 4561, 65535, 4562, 65535, 65535, 65535, 65535, 65535, 4563, 65535, 65535, 65535, 4564, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4565, 65535, 65535, 65535, 4566, 65535, 65535, 65535, 65535, 65535, 4567, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4568, 65535, 65535, 65535, 4569, 65535, 4570, 65535, 65535, 65535, 65535, 65535, 4571, 65535, 65535, 65535, 4572, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4573, 65535, 65535, 65535, 65535, 65535, 4574, 65535, 65535, 65535, 4575, 65535, 65535, 65535, 65535, 65535, 4576, 65535, 4577, 65535, 65535, 65535, 4578, 65535, 65535, 65535, 65535, 65535, 4579, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4580, 65535, 65535, 65535, 65535, 65535, 4581, 65535, 65535, 65535, 4582, 65535, 4583, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4584, 65535, 4585, 65535, 65535, 65535, 65535, 65535, 4586, 65535, 65535, 65535, 4587, 65535, 4588, 65535, 65535, 65535, 4589, 65535, 4590, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4591, 65535, 4592, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4593, 65535, 4594, 65535, 65535, 65535, 4595, 65535, 65535, 65535, 65535, 65535, 4596, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4597, 65535, 65535, 65535, 65535, 65535, 4598, 65535, 65535, 65535, 4599, 65535, 4600, 65535, 65535, 65535, 4601, 65535, 65535, 65535, 65535, 65535, 4602, 65535, 65535, 65535, 65535, 65535, 4603, 65535, 4604, 65535, 65535, 65535, 65535, 65535, 4605, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4606, 65535, 65535, 65535, 4607, 65535, 65535, 65535, 65535, 65535, 4608, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4609, 65535, 4610, 65535, 65535, 65535, 4611, 65535, 4612, 65535, 65535, 65535, 4613, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4614, 65535, 65535, 65535, 65535, 65535, 4615, 65535, 65535, 65535, 4616, 65535, 65535, 65535, 65535, 65535, 4617, 65535, 65535, 65535, 65535, 65535, 4618, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4619, 65535, 65535, 65535, 65535, 65535, 4620, 65535, 65535, 65535, 65535, 65535, 4621, 65535, 65535, 65535, 4622, 65535, 4623, 65535, 65535, 65535, 4624, 65535, 65535, 65535, 65535, 65535, 4625, 65535, 4626, 65535, 65535, 65535, 65535, 65535, 4627, 65535, 65535, 65535, 4628, 65535, 4629, 65535, 65535, 65535, 4630, 65535, 4631, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4632, 65535, 4633, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4634, 65535, 4635, 65535, 65535, 65535, 65535, 65535, 4636, 65535, 65535, 65535, 4637, 65535, 65535, 65535, 65535, 65535, 4638, 65535, 4639, 65535, 65535, 65535, 65535, 65535, 4640, 65535, 65535, 65535, 4641, 65535, 65535, 65535, 65535, 65535, 4642, 65535, 65535, 65535, 65535, 65535, 4643, 65535, 65535, 65535, 65535, 65535, 4644, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4645, 65535, 65535, 65535, 4646, 65535, 4647, 65535, 65535, 65535, 65535, 65535, 4648, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4649, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4650, 65535, 65535, 65535, 4651, 65535, 4652, 65535, 65535, 65535, 4653, 65535, 4654, 65535, 65535, 65535, 4655, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4656, 65535, 65535, 65535, 65535, 65535, 4657, 65535, 4658, 65535, 65535, 65535, 4659, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4660, 65535, 65535, 65535, 65535, 65535, 4661, 65535, 65535, 65535, 65535, 65535, 4662, 65535, 65535, 65535, 4663, 65535, 4664, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4665, 65535, 4666, 65535, 65535, 65535, 65535, 65535, 4667, 65535, 65535, 65535, 4668, 65535, 65535, 65535, 65535, 65535, 4669, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4670, 65535, 4671, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4672, 65535, 4673, 65535, 65535, 65535, 4674, 65535, 4675, 65535, 65535, 65535, 4676, 65535, 65535, 65535, 65535, 65535, 4677, 65535, 4678, 65535, 65535, 65535, 65535, 65535, 4679, 65535, 65535, 65535, 65535, 65535, 4680, 65535, 65535, 65535, 4681, 65535, 65535, 65535, 65535, 65535, 4682, 65535, 65535, 65535, 65535, 65535, 4683, 65535, 4684, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4685, 65535, 65535, 65535, 65535, 65535, 4686, 65535, 65535, 65535, 4687, 65535, 65535, 65535, 65535, 65535, 4688, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4689, 65535, 65535, 65535, 4690, 65535, 4691, 65535, 65535, 65535, 4692, 65535, 4693, 65535, 65535, 65535, 4694, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4695, 65535, 65535, 65535, 65535, 65535, 4696, 65535, 65535, 65535, 4697, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4698, 65535, 65535, 65535, 4699, 65535, 65535, 65535, 65535, 65535, 4700, 65535, 4701, 65535, 65535, 65535, 65535, 65535, 4702, 65535, 65535, 65535, 65535, 65535, 4703, 65535, 65535, 65535, 4704, 65535, 4705, 65535, 65535, 65535, 4706, 65535, 65535, 65535, 65535, 65535, 4707, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4708, 65535, 65535, 65535, 4709, 65535, 4710, 65535, 65535, 65535, 65535, 65535, 4711, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4712, 65535, 4713, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4714, 65535, 65535, 65535, 65535, 65535, 4715, 65535, 4716, 65535, 65535, 65535, 4717, 65535, 65535, 65535, 65535, 65535, 4718, 65535, 4719, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4720, 65535, 4721, 65535, 65535, 65535, 4722, 65535, 65535, 65535, 65535, 65535, 4723, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4724, 65535, 65535, 65535, 65535, 65535, 4725, 65535, 65535, 65535, 4726, 65535, 4727, 65535, 65535, 65535, 65535, 65535, 4728, 65535, 65535, 65535, 4729, 65535, 65535, 65535, 65535, 65535, 4730, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4731, 65535, 65535, 65535, 4732, 65535, 65535, 65535, 65535, 65535, 4733, 65535, 4734, 65535, 65535, 65535, 4735, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4736, 65535, 65535, 65535, 65535, 65535, 4737, 65535, 65535, 65535, 4738, 65535, 65535, 65535, 65535, 65535, 4739, 65535, 4740, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4741, 65535, 4742, 65535, 65535, 65535, 65535, 65535, 4743, 65535, 65535, 65535, 65535, 65535, 4744, 65535, 65535, 65535, 4745, 65535, 65535, 65535, 65535, 65535, 4746, 65535, 65535, 65535, 65535, 65535, 4747, 65535, 4748, 65535, 65535, 65535, 65535, 65535, 4749, 65535, 65535, 65535, 65535, 65535, 4750, 65535, 65535, 65535, 4751, 65535, 4752, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4753, 65535, 4754, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4755, 65535, 65535, 65535, 4756, 65535, 4757, 65535, 65535, 65535, 4758, 65535, 65535, 65535, 65535, 65535, 4759, 65535, 4760, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4761, 65535, 4762, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4763, 65535, 65535, 65535, 65535, 65535, 4764, 65535, 4765, 65535, 65535, 65535, 65535, 65535, 4766, 65535, 65535, 65535, 4767, 65535, 4768, 65535, 65535, 65535, 65535, 65535, 4769, 65535, 65535, 65535, 4770, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4771, 65535, 65535, 65535, 4772, 65535, 4773, 65535, 65535, 65535, 4774, 65535, 4775, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4776, 65535, 65535, 65535, 65535, 65535, 4777, 65535, 65535, 65535, 4778, 65535, 65535, 65535, 65535, 65535, 4779, 65535, 4780, 65535, 65535, 65535, 4781, 65535, 65535, 65535, 65535, 65535, 4782, 65535, 4783, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4784, 65535, 65535, 65535, 65535, 65535, 4785, 65535, 65535, 65535, 4786, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4787, 65535, 65535, 65535, 65535, 65535, 4788, 65535, 65535, 65535, 4789, 65535, 65535, 65535, 65535, 65535, 4790, 65535, 4791, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4792, 65535, 4793, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4794, 65535, 65535, 65535, 65535, 65535, 4795, 65535, 4796, 65535, 65535, 65535, 4797, 65535, 65535, 65535, 65535, 65535, 4798, 65535, 4799, 65535, 65535, 65535, 65535, 65535, 4800, 65535, 65535, 65535, 4801, 65535, 4802, 65535, 65535, 65535, 4803, 65535, 65535, 65535, 65535, 65535, 4804, 65535, 65535, 65535, 65535, 65535, 4805, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4806, 65535, 65535, 65535, 4807, 65535, 4808, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4809, 65535, 65535, 65535, 65535, 65535, 4810, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4811, 65535, 65535, 65535, 65535, 65535, 4812, 65535, 65535, 65535, 4813, 65535, 4814, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4815, 65535, 65535, 65535, 65535, 65535, 4816, 65535, 65535, 65535, 4817, 65535, 65535, 65535, 65535, 65535, 4818, 65535, 65535, 65535, 65535, 65535, 4819, 65535, 65535, 65535, 65535, 65535, 4820, 65535, 4821, 65535, 65535, 65535, 65535, 65535, 4822, 65535, 65535, 65535, 65535, 65535, 4823, 65535, 65535, 65535, 65535, 65535, 4824, 65535, 65535, 65535, 4825, 65535, 65535, 65535, 65535, 65535, 4826, 65535, 4827, 65535, 65535, 65535, 65535, 65535, 4828, 65535, 65535, 65535, 65535, 65535, 4829, 65535, 65535, 65535, 4830, 65535, 4831, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4832, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4833, 65535, 4834, 65535, 65535, 65535, 4835, 65535, 4836, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4837, 65535, 4838, 65535, 65535, 65535, 65535, 65535, 4839, 65535, 65535, 65535, 4840, 65535, 4841, 65535, 65535, 65535, 4842, 65535, 65535, 65535, 65535, 65535, 4843, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4844, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4845, 65535, 4846, 65535, 65535, 65535, 65535, 65535, 4847, 65535, 65535, 65535, 4848, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4849, 65535, 65535, 65535, 4850, 65535, 4851, 65535, 65535, 65535, 4852, 65535, 4853, 65535, 65535, 65535, 4854, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4855, 65535, 65535, 65535, 65535, 65535, 4856, 65535, 65535, 65535, 4857, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4858, 65535, 65535, 65535, 4859, 65535, 65535, 65535, 65535, 65535, 4860, 65535, 4861, 65535, 65535, 65535, 65535, 65535, 4862, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4863, 65535, 4864, 65535, 65535, 65535, 4865, 65535, 65535, 65535, 65535, 65535, 4866, 65535, 4867, 65535, 65535, 65535, 65535, 65535, 4868, 65535, 65535, 65535, 4869, 65535, 4870, 65535, 65535, 65535, 4871, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4872, 65535, 4873, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4874, 65535, 65535, 65535, 4875, 65535, 4876, 65535, 65535, 65535, 4877, 65535, 65535, 65535, 65535, 65535, 4878, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4879, 65535, 65535, 65535, 4880, 65535, 4881, 65535, 65535, 65535, 4882, 65535, 65535, 65535, 65535, 65535, 4883, 65535, 65535, 65535, 65535, 65535, 4884, 65535, 4885, 65535, 65535, 65535, 65535, 65535, 4886, 65535, 65535, 65535, 4887, 65535, 4888, 65535, 65535, 65535, 65535, 65535, 4889, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4890, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4891, 65535, 65535, 65535, 4892, 65535, 4893, 65535, 65535, 65535, 4894, 65535, 65535, 65535, 65535, 65535, 4895, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4896, 65535, 65535, 65535, 4897, 65535, 65535, 65535, 65535, 65535, 4898, 65535, 4899, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4900, 65535, 4901, 65535, 65535, 65535, 65535, 65535, 4902, 65535, 65535, 65535, 65535, 65535, 4903, 65535, 65535, 65535, 4904, 65535, 4905, 65535, 65535, 65535, 4906, 65535, 65535, 65535, 65535, 65535, 4907, 65535, 4908, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4909, 65535, 4910, 65535, 65535, 65535, 4911, 65535, 4912, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4913, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4914, 65535, 4915, 65535, 65535, 65535, 4916, 65535, 4917, 65535, 65535, 65535, 4918, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4919, 65535, 65535, 65535, 65535, 65535, 4920, 65535, 65535, 65535, 4921, 65535, 65535, 65535, 65535, 65535, 4922, 65535, 65535, 65535, 65535, 65535, 4923, 65535, 65535, 65535, 65535, 65535, 4924, 65535, 4925, 65535, 65535, 65535, 65535, 65535, 4926, 65535, 65535, 65535, 4927, 65535, 4928, 65535, 65535, 65535, 65535, 65535, 4929, 65535, 65535, 65535, 4930, 65535, 65535, 65535, 65535, 65535, 4931, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4932, 65535, 65535, 65535, 4933, 65535, 4934, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4935, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4936, 65535, 65535, 65535, 65535, 65535, 4937, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4938, 65535, 4939, 65535, 65535, 65535, 4940, 65535, 65535, 65535, 65535, 65535, 4941, 65535, 4942, 65535, 65535, 65535, 65535, 65535, 4943, 65535, 65535, 65535, 65535, 65535, 4944, 65535, 65535, 65535, 4945, 65535, 4946, 65535, 65535, 65535, 4947, 65535, 65535, 65535, 65535, 65535, 4948, 65535, 4949, 65535, 65535, 65535, 65535, 65535, 4950, 65535, 65535, 65535, 4951, 65535, 4952, 65535, 65535, 65535, 65535, 65535, 4953, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4954, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4955, 65535, 4956, 65535, 65535, 65535, 4957, 65535, 4958, 65535, 65535, 65535, 4959, 65535, 65535, 65535, 65535, 65535, 4960, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4961, 65535, 65535, 65535, 4962, 65535, 4963, 65535, 65535, 65535, 4964, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4965, 65535, 4966, 65535, 65535, 65535, 65535, 65535, 4967, 65535, 65535, 65535, 4968, 65535, 4969, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4970, 65535, 65535, 65535, 65535, 65535, 4971, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4972, 65535, 4973, 65535, 65535, 65535, 4974, 65535, 4975, 65535, 65535, 65535, 4976, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4977, 65535, 65535, 65535, 65535, 65535, 4978, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4979, 65535, 4980, 65535, 65535, 65535, 4981, 65535, 65535, 65535, 65535, 65535, 4982, 65535, 4983, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4984, 65535, 4985, 65535, 65535, 65535, 4986, 65535, 65535, 65535, 65535, 65535, 4987, 65535, 4988, 65535, 65535, 65535, 65535, 65535, 4989, 65535, 65535, 65535, 4990, 65535, 4991, 65535, 65535, 65535, 4992, 65535, 4993, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4994, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4995, 65535, 4996, 65535, 65535, 65535, 4997, 65535, 65535, 65535, 65535, 65535, 4998, 65535, 65535, 65535, 65535, 65535, 4999, 65535, 5000, 65535, 65535, 65535, 65535, 65535, 5001, 65535, 65535, 65535, 65535, 65535, 5002, 65535, 65535, 65535, 5003, 65535, 65535, 65535, 65535, 65535, 5004, 65535, 65535, 65535, 65535, 65535, 5005, 65535, 5006, 65535, 65535, 65535, 65535, 65535, 5007, 65535, 65535, 65535, 5008, 65535, 5009, 65535, 65535, 65535, 65535, 65535, 5010, 65535, 65535, 65535, 5011, 65535, 65535, 65535, 65535, 65535, 5012, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5013, 65535, 65535, 65535, 5014, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5015, 65535, 65535, 65535, 5016, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5017, 65535, 65535, 65535, 65535, 65535, 5018, 65535, 65535, 65535, 5019, 65535, 65535, 65535, 65535, 65535, 5020, 65535, 5021, 65535, 65535, 65535, 5022, 65535, 65535, 65535, 65535, 65535, 5023, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5024, 65535, 65535, 65535, 65535, 65535, 5025, 65535, 65535, 65535, 5026, 65535, 5027, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5028, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5029, 65535, 65535, 65535, 5030, 65535, 5031, 65535, 65535, 65535, 5032, 65535, 5033, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5034, 65535, 5035, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5036, 65535, 5037, 65535, 65535, 65535, 5038, 65535, 5039, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5040, 65535, 65535, 65535, 65535, 65535, 5041, 65535, 65535, 65535, 5042, 65535, 5043, 65535, 65535, 65535, 5044, 65535, 65535, 65535, 65535, 65535, 5045, 65535, 65535, 65535, 65535, 65535, 5046, 65535, 5047, 65535, 65535, 65535, 65535, 65535, 5048, 65535, 65535, 65535, 5049, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5050, 65535, 65535, 65535, 5051, 65535, 65535, 65535, 65535, 65535, 5052, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5053, 65535, 65535, 65535, 65535, 65535, 5054, 65535, 65535, 65535, 5055, 65535, 5056, 65535, 65535, 65535, 5057, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5058, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5059, 65535, 65535, 65535, 65535, 65535, 5060, 65535, 65535, 65535, 65535, 65535, 5061, 65535, 65535, 65535, 65535, 65535, 5062, 65535, 5063, 65535, 65535, 65535, 65535, 65535, 5064, 65535, 65535, 65535, 65535, 65535, 5065, 65535, 65535, 65535, 5066, 65535, 5067, 65535, 65535, 65535, 5068, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5069, 65535, 65535, 65535, 65535, 65535, 5070, 65535, 65535, 65535, 5071, 65535, 5072, 65535, 65535, 65535, 5073, 65535, 5074, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5075, 65535, 5076, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5077, 65535, 5078, 65535, 65535, 65535, 65535, 65535, 5079, 65535, 65535, 65535, 5080, 65535, 65535, 65535, 65535, 65535, 5081, 65535, 5082, 65535, 65535, 65535, 65535, 65535, 5083, 65535, 65535, 65535, 5084, 65535, 5085, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5086, 65535, 65535, 65535, 65535, 65535, 5087, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5088, 65535, 65535, 65535, 5089, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5090, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5091, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5092, 65535, 65535, 65535, 5093, 65535, 5094, 65535, 65535, 65535, 5095, 65535, 5096, 65535, 65535, 65535, 5097, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5098, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5099, 65535, 65535, 65535, 65535, 65535, 5100, 65535, 5101, 65535, 65535, 65535, 5102, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5103, 65535, 65535, 65535, 65535, 65535, 5104, 65535, 65535, 65535, 5105, 65535, 5106, 65535, 65535, 65535, 5107, 65535, 65535, 65535, 65535, 65535, 5108, 65535, 5109, 65535, 65535, 65535, 65535, 65535, 5110, 65535, 65535, 65535, 5111, 65535, 65535, 65535, 65535, 65535, 5112, 65535, 5113, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5114, 65535, 5115, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5116, 65535, 5117, 65535, 65535, 65535, 5118, 65535, 65535, 65535, 65535, 65535, 5119, 65535, 65535, 65535, 65535, 65535, 5120, 65535, 5121, 65535, 65535, 65535, 65535, 65535, 5122, 65535, 65535, 65535, 5123, 65535, 5124, 65535, 65535, 65535, 5125, 65535, 65535, 65535, 65535, 65535, 5126, 65535, 65535, 65535, 65535, 65535, 5127, 65535, 5128, 65535, 65535, 65535, 65535, 65535, 5129, 65535, 65535, 65535, 65535, 65535, 5130, 65535, 65535, 65535, 65535, 65535, 5131, 65535, 65535, 65535, 5132, 65535, 65535, 65535, 65535, 65535, 5133, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5134, 65535, 5135, 65535, 65535, 65535, 5136, 65535, 5137, 65535, 65535, 65535, 5138, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5139, 65535, 65535, 65535, 65535, 65535, 5140, 65535, 65535, 65535, 5141, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5142, 65535, 65535, 65535, 5143, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5144, 65535, 65535, 65535, 65535, 65535, 5145, 65535, 65535, 65535, 65535, 65535, 5146, 65535, 65535, 65535, 5147, 65535, 5148, 65535, 65535, 65535, 5149, 65535, 65535, 65535, 65535, 65535, 5150, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5151, 65535, 65535, 65535, 5152, 65535, 5153, 65535, 65535, 65535, 5154, 65535, 5155, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5156, 65535, 5157, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5158, 65535, 5159, 65535, 65535, 65535, 65535, 65535, 5160, 65535, 65535, 65535, 5161, 65535, 65535, 65535, 65535, 65535, 5162, 65535, 5163, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5164, 65535, 65535, 65535, 65535, 65535, 5165, 65535, 65535, 65535, 65535, 65535, 5166, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5167, 65535, 65535, 65535, 65535, 65535, 5168, 65535, 65535, 65535, 5169, 65535, 5170, 65535, 65535, 65535, 65535, 65535, 5171, 65535, 65535, 65535, 5172, 65535, 65535, 65535, 65535, 65535, 5173, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5174, 65535, 65535, 65535, 5175, 65535, 65535, 65535, 65535, 65535, 5176, 65535, 5177, 65535, 65535, 65535, 5178, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5179, 65535, 65535, 65535, 5180, 65535, 65535, 65535, 65535, 65535, 5181, 65535, 5182, 65535, 65535, 65535, 5183, 65535, 65535, 65535, 65535, 65535, 5184, 65535, 5185, 65535, 65535, 65535, 65535, 65535, 5186, 65535, 65535, 65535, 65535, 65535, 5187, 65535, 65535, 65535, 5188, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5189, 65535, 5190, 65535, 65535, 65535, 65535, 65535, 5191, 65535, 65535, 65535, 65535, 65535, 5192, 65535, 65535, 65535, 5193, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5194, 65535, 5195, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5196, 65535, 5197, 65535, 65535, 65535, 5198, 65535, 5199, 65535, 65535, 65535, 5200, 65535, 65535, 65535, 65535, 65535, 5201, 65535, 5202, 65535, 65535, 65535, 65535, 65535, 5203, 65535, 65535, 65535, 65535, 65535, 5204, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5205, 65535, 65535, 65535, 65535, 65535, 5206, 65535, 5207, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5208, 65535, 5209, 65535, 65535, 65535, 65535, 65535, 5210, 65535, 65535, 65535, 5211, 65535, 65535, 65535, 65535, 65535, 5212, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5213, 65535, 65535, 65535, 5214, 65535, 5215, 65535, 65535, 65535, 5216, 65535, 5217, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5218, 65535, 65535, 65535, 65535, 65535, 5219, 65535, 65535, 65535, 5220, 65535, 65535, 65535, 65535, 65535, 5221, 65535, 5222, 65535, 65535, 65535, 5223, 65535, 65535, 65535, 65535, 65535, 5224, 65535, 5225, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5226, 65535, 65535, 65535, 5227, 65535, 5228, 65535, 65535, 65535, 5229, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5230, 65535, 65535, 65535, 65535, 65535, 5231, 65535, 65535, 65535, 5232, 65535, 5233, 65535, 65535, 65535, 65535, 65535, 5234, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5235, 65535, 5236, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5237, 65535, 65535, 65535, 65535, 65535, 5238, 65535, 5239, 65535, 65535, 65535, 5240, 65535, 65535, 65535, 65535, 65535, 5241, 65535, 5242, 65535, 65535, 65535, 65535, 65535, 5243, 65535, 65535, 65535, 5244, 65535, 5245, 65535, 65535, 65535, 5246, 65535, 65535, 65535, 65535, 65535, 5247, 65535, 65535, 65535, 65535, 65535, 5248, 65535, 5249, 65535, 65535, 65535, 65535, 65535, 5250, 65535, 65535, 65535, 5251, 65535, 5252, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5253, 65535, 65535, 65535, 65535, 65535, 5254, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5255, 65535, 65535, 65535, 65535, 65535, 5256, 65535, 65535, 65535, 5257, 65535, 5258, 65535, 65535, 65535, 5259, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5260, 65535, 65535, 65535, 65535, 65535, 5261, 65535, 65535, 65535, 5262, 65535, 65535, 65535, 65535, 65535, 5263, 65535, 5264, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5265, 65535, 5266, 65535, 65535, 65535, 65535, 65535, 5267, 65535, 65535, 65535, 65535, 65535, 5268, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5269, 65535, 65535, 65535, 65535, 65535, 5270, 65535, 5271, 65535, 65535, 65535, 65535, 65535, 5272, 65535, 65535, 65535, 5273, 65535, 5274, 65535, 65535, 65535, 5275, 65535, 5276, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5277, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5278, 65535, 65535, 65535, 5279, 65535, 5280, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5281, 65535, 5282, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5283, 65535, 5284, 65535, 65535, 65535, 5285, 65535, 65535, 65535, 65535, 65535, 5286, 65535, 65535, 65535, 65535, 65535, 5287, 65535, 5288, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5289, 65535, 5290, 65535, 65535, 65535, 65535, 65535, 5291, 65535, 65535, 65535, 5292, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5293, 65535, 65535, 65535, 5294, 65535, 5295, 65535, 65535, 65535, 5296, 65535, 5297, 65535, 65535, 65535, 5298, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5299, 65535, 65535, 65535, 65535, 65535, 5300, 65535, 65535, 65535, 5301, 65535, 65535, 65535, 65535, 65535, 5302, 65535, 5303, 65535, 65535, 65535, 5304, 65535, 65535, 65535, 65535, 65535, 5305, 65535, 5306, 65535, 65535, 65535, 65535, 65535, 5307, 65535, 65535, 65535, 65535, 65535, 5308, 65535, 65535, 65535, 65535, 65535, 5309, 65535, 65535, 65535, 5310, 65535, 65535, 65535, 65535, 65535, 5311, 65535, 5312, 65535, 65535, 65535, 65535, 65535, 5313, 65535, 65535, 65535, 5314, 65535, 65535, 65535, 65535, 65535, 5315, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5316, 65535, 5317, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5318, 65535, 65535, 65535, 5319, 65535, 5320, 65535, 65535, 65535, 5321, 65535, 65535, 65535, 65535, 65535, 5322, 65535, 5323, 65535, 65535, 65535, 65535, 65535, 5324, 65535, 65535, 65535, 5325, 65535, 5326, 65535, 65535, 65535, 5327, 65535, 65535, 65535, 65535, 65535, 5328, 65535, 65535, 65535, 65535, 65535, 5329, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5330, 65535, 65535, 65535, 5331, 65535, 5332, 65535, 65535, 65535, 65535, 65535, 5333, 65535, 65535, 65535, 5334, 65535, 65535, 65535, 65535, 65535, 5335, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5336, 65535, 65535, 65535, 5337, 65535, 5338, 65535, 65535, 65535, 5339, 65535, 5340, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5341, 65535, 65535, 65535, 5342, 65535, 65535, 65535, 65535, 65535, 5343, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5344, 65535, 5345, 65535, 65535, 65535, 65535, 65535, 5346, 65535, 65535, 65535, 65535, 65535, 5347, 65535, 65535, 65535, 5348, 65535, 5349, 65535, 65535, 65535, 5350, 65535, 65535, 65535, 65535, 65535, 5351, 65535, 5352, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5353, 65535, 65535, 65535, 5354, 65535, 5355, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5356, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5357, 65535, 5358, 65535, 65535, 65535, 5359, 65535, 5360, 65535, 65535, 65535, 5361, 65535, 65535, 65535, 65535, 65535, 5362, 65535, 5363, 65535, 65535, 65535, 65535, 65535, 5364, 65535, 65535, 65535, 5365, 65535, 65535, 65535, 65535, 65535, 5366, 65535, 65535, 65535, 65535, 65535, 5367, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5368, 65535, 65535, 65535, 65535, 65535, 5369, 65535, 65535, 65535, 5370, 65535, 5371, 65535, 65535, 65535, 65535, 65535, 5372, 65535, 65535, 65535, 5373, 65535, 65535, 65535, 65535, 65535, 5374, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5375, 65535, 65535, 65535, 5376, 65535, 5377, 65535, 65535, 65535, 5378, 65535, 65535, 65535, 65535, 65535, 5379, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5380, 65535, 65535, 65535, 65535, 65535, 5381, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5382, 65535, 65535, 65535, 5383, 65535, 65535, 65535, 65535, 65535, 5384, 65535, 5385, 65535, 65535, 65535, 65535, 65535, 5386, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5387, 65535, 5388, 65535, 65535, 65535, 5389, 65535, 65535, 65535, 65535, 65535, 5390, 65535, 5391, 65535, 65535, 65535, 65535, 65535, 5392, 65535, 65535, 65535, 5393, 65535, 5394, 65535, 65535, 65535, 65535, 65535, 5395, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5396, 65535, 5397, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5398, 65535, 5399, 65535, 65535, 65535, 5400, 65535, 5401, 65535, 65535, 65535, 5402, 65535, 65535, 65535, 65535, 65535, 5403, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5404, 65535, 65535, 65535, 5405, 65535, 5406, 65535, 65535, 65535, 5407, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5408, 65535, 5409, 65535, 65535, 65535, 65535, 65535, 5410, 65535, 65535, 65535, 5411, 65535, 5412, 65535, 65535, 65535, 65535, 65535, 5413, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5414, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5415, 65535, 5416, 65535, 65535, 65535, 5417, 65535, 65535, 65535, 65535, 65535, 5418, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5419, 65535, 65535, 65535, 65535, 65535, 5420, 65535, 65535, 65535, 5421, 65535, 65535, 65535, 65535, 65535, 5422, 65535, 5423, 65535, 65535, 65535, 5424, 65535, 65535, 65535, 65535, 65535, 5425, 65535, 5426, 65535, 65535, 65535, 65535, 65535, 5427, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5428, 65535, 5429, 65535, 65535, 65535, 5430, 65535, 65535, 65535, 65535, 65535, 5431, 65535, 5432, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5433, 65535, 5434, 65535, 65535, 65535, 5435, 65535, 5436, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5437, 65535, 5438, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5439, 65535, 5440, 65535, 65535, 65535, 5441, 65535, 65535, 65535, 65535, 65535, 5442, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5443, 65535, 65535, 65535, 65535, 65535, 5444, 65535, 65535, 65535, 65535, 65535, 5445, 65535, 65535, 65535, 5446, 65535, 65535, 65535, 65535, 65535, 5447, 65535, 65535, 65535, 65535, 65535, 5448, 65535, 5449, 65535, 65535, 65535, 65535, 65535, 5450, 65535, 65535, 65535, 5451, 65535, 5452, 65535, 65535, 65535, 65535, 65535, 5453, 65535, 65535, 65535, 5454, 65535, 65535, 65535, 65535, 65535, 5455, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5456, 65535, 65535, 65535, 5457, 65535, 5458, 65535, 65535, 65535, 65535, 65535, 5459, 65535, 65535, 65535, 5460, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5461, 65535, 65535, 65535, 65535, 65535, 5462, 65535, 65535, 65535, 5463, 65535, 65535, 65535, 65535, 65535, 5464, 65535, 5465, 65535, 65535, 65535, 5466, 65535, 65535, 65535, 65535, 65535, 5467, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5468, 65535, 65535, 65535, 65535, 65535, 5469, 65535, 65535, 65535, 5470, 65535, 5471, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5472, 65535, 5473, 65535, 65535, 65535, 65535, 65535, 5474, 65535, 65535, 65535, 5475, 65535, 5476, 65535, 65535, 65535, 5477, 65535, 5478, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5479, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5480, 65535, 5481, 65535, 65535, 65535, 5482, 65535, 5483, 65535, 65535, 65535, 5484, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5485, 65535, 65535, 65535, 65535, 65535, 5486, 65535, 65535, 65535, 5487, 65535, 5488, 65535, 65535, 65535, 5489, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5490, 65535, 5491, 65535, 65535, 65535, 65535, 65535, 5492, 65535, 65535, 65535, 5493, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5494, 65535, 65535, 65535, 65535, 65535, 5495, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5496, 65535, 65535, 65535, 5497, 65535, 5498, 65535, 65535, 65535, 5499, 65535, 5500, 65535, 65535, 65535, 5501, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5502, 65535, 65535, 65535, 65535, 65535, 5503, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5504, 65535, 65535, 65535, 65535, 65535, 5505, 65535, 65535, 65535, 65535, 65535, 5506, 65535, 5507, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5508, 65535, 65535, 65535, 5509, 65535, 5510, 65535, 65535, 65535, 5511, 65535, 65535, 65535, 65535, 65535, 5512, 65535, 5513, 65535, 65535, 65535, 65535, 65535, 5514, 65535, 65535, 65535, 5515, 65535, 5516, 65535, 65535, 65535, 5517, 65535, 5518, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5519, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5520, 65535, 5521, 65535, 65535, 65535, 65535, 65535, 5522, 65535, 65535, 65535, 5523, 65535, 65535, 65535, 65535, 65535, 5524, 65535, 5525, 65535, 65535, 65535, 65535, 65535, 5526, 65535, 65535, 65535, 5527, 65535, 5528, 65535, 65535, 65535, 5529, 65535, 65535, 65535, 65535, 65535, 5530, 65535, 65535, 65535, 65535, 65535, 5531, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5532, 65535, 65535, 65535, 5533, 65535, 5534, 65535, 65535, 65535, 65535, 65535, 5535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5536, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5537, 65535, 65535, 65535, 5538, 65535, 65535, 65535, 65535, 65535, 5539, 65535, 5540, 65535, 65535, 65535, 5541, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5542, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5543, 65535, 65535, 65535, 65535, 65535, 5544, 65535, 5545, 65535, 65535, 65535, 5546, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5547, 65535, 65535, 65535, 65535, 65535, 5548, 65535, 65535, 65535, 65535, 65535, 5549, 65535, 65535, 65535, 5550, 65535, 5551, 65535, 65535, 65535, 5552, 65535, 65535, 65535, 65535, 65535, 5553, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5554, 65535, 65535, 65535, 5555, 65535, 65535, 65535, 65535, 65535, 5556, 65535, 5557, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5558, 65535, 5559, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5560, 65535, 5561, 65535, 65535, 65535, 5562, 65535, 5563, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5564, 65535, 5565, 65535, 65535, 65535, 65535, 65535, 5566, 65535, 65535, 65535, 5567, 65535, 5568, 65535, 65535, 65535, 5569, 65535, 65535, 65535, 65535, 65535, 5570, 65535, 65535, 65535, 65535, 65535, 5571, 65535, 5572, 65535, 65535, 65535, 65535, 65535, 5573, 65535, 65535, 65535, 65535, 65535, 5574, 65535, 65535, 65535, 65535, 65535, 5575, 65535, 65535, 65535, 5576, 65535, 65535, 65535, 65535, 65535, 5577, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5578, 65535, 65535, 65535, 65535, 65535, 5579, 65535, 65535, 65535, 5580, 65535, 5581, 65535, 65535, 65535, 5582, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5583, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5584, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5585, 65535, 65535, 65535, 5586, 65535, 65535, 65535, 65535, 65535, 5587, 65535, 5588, 65535, 65535, 65535, 65535, 65535, 5589, 65535, 65535, 65535, 65535, 65535, 5590, 65535, 65535, 65535, 5591, 65535, 5592, 65535, 65535, 65535, 5593, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5594, 65535, 65535, 65535, 5595, 65535, 5596, 65535, 65535, 65535, 5597, 65535, 5598, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5599, 65535, 5600, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5601, 65535, 5602, 65535, 65535, 65535, 5603, 65535, 5604, 65535, 65535, 65535, 5605, 65535, 65535, 65535, 65535, 65535, 5606, 65535, 5607, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5608, 65535, 5609, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5610, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5611, 65535, 65535, 65535, 65535, 65535, 5612, 65535, 65535, 65535, 5613, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5614, 65535, 65535, 65535, 5615, 65535, 65535, 65535, 65535, 65535, 5616, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5617, 65535, 65535, 65535, 5618, 65535, 65535, 65535, 65535, 65535, 5619, 65535, 5620, 65535, 65535, 65535, 5621, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5622, 65535, 65535, 65535, 65535, 65535, 5623, 65535, 65535, 65535, 5624, 65535, 65535, 65535, 65535, 65535, 5625, 65535, 5626, 65535, 65535, 65535, 5627, 65535, 65535, 65535, 65535, 65535, 5628, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5629, 65535, 65535, 65535, 65535, 65535, 5630, 65535, 65535, 65535, 5631, 65535, 65535, 65535, 65535, 65535, 5632, 65535, 65535, 65535, 65535, 65535, 5633, 65535, 5634, 65535, 65535, 65535, 65535, 65535, 5635, 65535, 65535, 65535, 65535, 65535, 5636, 65535, 65535, 65535, 5637, 65535, 5638, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5639, 65535, 5640, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5641, 65535, 5642, 65535, 65535, 65535, 5643, 65535, 65535, 65535, 65535, 65535, 5644, 65535, 65535, 65535, 65535, 65535, 5645, 65535, 5646, 65535, 65535, 65535, 65535, 65535, 5647, 65535, 65535, 65535, 5648, 65535, 5649, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5650, 65535, 65535, 65535, 65535, 65535, 5651, 65535, 5652, 65535, 65535, 65535, 65535, 65535, 5653, 65535, 65535, 65535, 65535, 65535, 5654, 65535, 65535, 65535, 65535, 65535, 5655, 65535, 65535, 65535, 5656, 65535, 65535, 65535, 65535, 65535, 5657, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5658, 65535, 5659, 65535, 65535, 65535, 5660, 65535, 5661, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5662, 65535, 65535, 65535, 65535, 65535, 5663, 65535, 65535, 65535, 5664, 65535, 65535, 65535, 65535, 65535, 5665, 65535, 5666, 65535, 65535, 65535, 5667, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5668, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5669, 65535, 65535, 65535, 5670, 65535, 5671, 65535, 65535, 65535, 5672, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5673, 65535, 65535, 65535, 65535, 65535, 5674, 65535, 65535, 65535, 5675, 65535, 5676, 65535, 65535, 65535, 5677, 65535, 5678, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5679, 65535, 5680, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5681, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5682, 65535, 65535, 65535, 5683, 65535, 65535, 65535, 65535, 65535, 5684, 65535, 5685, 65535, 65535, 65535, 65535, 65535, 5686, 65535, 65535, 65535, 5687, 65535, 65535, 65535, 65535, 65535, 5688, 65535, 65535, 65535, 65535, 65535, 5689, 65535, 65535, 65535, 65535, 65535, 5690, 65535, 5691, 65535, 65535, 65535, 65535, 65535, 5692, 65535, 65535, 65535, 5693, 65535, 5694, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5695, 65535, 65535, 65535, 65535, 65535, 5696, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5697, 65535, 65535, 65535, 65535, 65535, 5698, 65535, 65535, 65535, 5699, 65535, 5700, 65535, 65535, 65535, 5701, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5702, 65535, 65535, 65535, 5703, 65535, 65535, 65535, 65535, 65535, 5704, 65535, 5705, 65535, 65535, 65535, 5706, 65535, 65535, 65535, 65535, 65535, 5707, 65535, 5708, 65535, 65535, 65535, 65535, 65535, 5709, 65535, 65535, 65535, 65535, 65535, 5710, 65535, 65535, 65535, 65535, 65535, 5711, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5712, 65535, 5713, 65535, 65535, 65535, 65535, 65535, 5714, 65535, 65535, 65535, 5715, 65535, 5716, 65535, 65535, 65535, 5717, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5718, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5719, 65535, 5720, 65535, 65535, 65535, 5721, 65535, 5722, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5723, 65535, 5724, 65535, 65535, 65535, 65535, 65535, 5725, 65535, 65535, 65535, 65535, 65535, 5726, 65535, 65535, 65535, 5727, 65535, 65535, 65535, 65535, 65535, 5728, 65535, 65535, 65535, 65535, 65535, 5729, 65535, 5730, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5731, 65535, 5732, 65535, 65535, 65535, 65535, 65535, 5733, 65535, 65535, 65535, 5734, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5735, 65535, 65535, 65535, 5736, 65535, 5737, 65535, 65535, 65535, 5738, 65535, 5739, 65535, 65535, 65535, 5740, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5741, 65535, 65535, 65535, 65535, 65535, 5742, 65535, 65535, 65535, 5743, 65535, 65535, 65535, 65535, 65535, 5744, 65535, 5745, 65535, 65535, 65535, 5746, 65535, 65535, 65535, 65535, 65535, 5747, 65535, 5748, 65535, 65535, 65535, 65535, 65535, 5749, 65535, 65535, 65535, 65535, 65535, 5750, 65535, 65535, 65535, 5751, 65535, 5752, 65535, 65535, 65535, 5753, 65535, 65535, 65535, 65535, 65535, 5754, 65535, 5755, 65535, 65535, 65535, 65535, 65535, 5756, 65535, 65535, 65535, 5757, 65535, 5758, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5759 };
__constant ushort WHL_INV[30030] = { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3533, 0, 26869, 0, 0, 0, 3917, 0, 0, 0, 0, 0, 27959, 0, 6781, 0, 0, 0, 0, 0, 6493, 0, 0, 0, 18311, 0, 5587, 0, 0, 0, 10223, 0, 0, 0, 0, 0, 27197, 0, 0, 0, 0, 0, 509, 0, 21661, 0, 0, 0, 0, 0, 19273, 0, 0, 0, 10151, 0, 11107, 0, 0, 0, 0, 0, 26989, 0, 0, 0, 9407, 0, 0, 0, 0, 0, 4049, 0, 0, 0, 0, 0, 0, 0, 5263, 0, 0, 0, 15461, 0, 16327, 0, 0, 0, 22733, 0, 29479, 0, 0, 0, 28967, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19153, 0, 0, 0, 8711, 0, 0, 0, 0, 0, 15563, 0, 4969, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20759, 0, 1591, 0, 0, 0, 0, 0, 13963, 0, 0, 0, 0, 0, 5527, 0, 0, 0, 7013, 0, 0, 0, 0, 0, 27947, 0, 0, 0, 0, 0, 13589, 0, 5641, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23741, 0, 22717, 0, 0, 0, 10823, 0, 3169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17221, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15217, 0, 0, 0, 18653, 0, 12589, 0, 0, 0, 21137, 0, 0, 0, 0, 0, 4649, 0, 17071, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6341, 0, 0, 0, 0, 0, 16943, 0, 0, 0, 0, 0, 14387, 0, 0, 0, 0, 0, 15629, 0, 20611, 0, 0, 0, 0, 0, 1843, 0, 0, 0, 4061, 0, 17827, 0, 0, 0, 0, 0, 19639, 0, 0, 0, 6047, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15553, 0, 0, 0, 8111, 0, 8347, 0, 0, 0, 13073, 0, 0, 0, 0, 0, 3347, 0, 0, 0, 0, 0, 0, 0, 26401, 0, 0, 0, 0, 0, 22723, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27953, 0, 9379, 0, 0, 0, 20417, 0, 0, 0, 0, 0, 17399, 0, 21961, 0, 0, 0, 0, 0, 22993, 0, 0, 0, 0, 0, 4267, 0, 0, 0, 0, 0, 10459, 0, 0, 0, 2117, 0, 0, 0, 0, 0, 22619, 0, 24961, 0, 0, 0, 0, 0, 10363, 0, 0, 0, 22691, 0, 0, 0, 0, 0, 0, 0, 1909, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11969, 0, 7561, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19091, 0, 19627, 0, 0, 0, 20753, 0, 27499, 0, 0, 0, 2237, 0, 0, 0, 0, 0, 20399, 0, 0, 0, 0, 0, 0, 0, 27073, 0, 0, 0, 5081, 0, 3697, 0, 0, 0, 20513, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11849, 0, 0, 0, 0, 0, 0, 0, 6043, 0, 0, 0, 5321, 0, 10477, 0, 0, 0, 0, 0, 26359, 0, 0, 0, 26627, 0, 0, 0, 0, 0, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5591, 0, 9187, 0, 0, 0, 23363, 0, 27589, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6661, 0, 0, 0, 0, 0, 20203, 0, 0, 0, 29921, 0, 0, 0, 0, 0, 26903, 0, 0, 0, 0, 0, 2987, 0, 0, 0, 0, 0, 24119, 0, 9151, 0, 0, 0, 0, 0, 24253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24863, 0, 6679, 0, 0, 0, 12407, 0, 0, 0, 0, 0, 14639, 0, 28531, 0, 0, 0, 0, 0, 9103, 0, 0, 0, 0, 0, 21457, 0, 0, 0, 11243, 0, 17659, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26879, 0, 9661, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7121, 0, 4717, 0, 0, 0, 11093, 0, 0, 0, 0, 0, 11267, 0, 0, 0, 0, 0, 19139, 0, 17491, 0, 0, 0, 0, 0, 26023, 0, 0, 0, 0, 0, 18607, 0, 0, 0, 9803, 0, 0, 0, 0, 0, 28667, 0, 0, 0, 0, 0, 0, 0, 8431, 0, 0, 0, 0, 0, 8143, 0, 0, 0, 14351, 0, 16147, 0, 0, 0, 0, 0, 22999, 0, 0, 0, 14657, 0, 0, 0, 0, 0, 4469, 0, 0, 0, 0, 0, 0, 0, 6733, 0, 0, 0, 9161, 0, 10447, 0, 0, 0, 0, 0, 10159, 0, 0, 0, 23927, 0, 0, 0, 0, 0, 0, 0, 27031, 0, 0, 0, 0, 0, 12853, 0, 0, 0, 10181, 0, 0, 0, 0, 0, 0, 0, 5389, 0, 0, 0, 7187, 0, 0, 0, 0, 0, 16769, 0, 0, 0, 0, 0, 0, 0, 18163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23813, 0, 21799, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15479, 0, 5221, 0, 0, 0, 0, 0, 27163, 0, 0, 0, 13241, 0, 9487, 0, 0, 0, 21533, 0, 24379, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12599, 0, 24781, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27701, 0, 28657, 0, 0, 0, 28313, 0, 9439, 0, 0, 0, 12527, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16573, 0, 0, 0, 19361, 0, 11257, 0, 0, 0, 13373, 0, 0, 0, 0, 0, 27407, 0, 0, 0, 0, 0, 10589, 0, 21031, 0, 0, 0, 0, 0, 9403, 0, 0, 0, 25481, 0, 0, 0, 0, 0, 0, 0, 5359, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19589, 0, 0, 0, 0, 0, 0, 0, 12403, 0, 0, 0, 7691, 0, 12547, 0, 0, 0, 23783, 0, 0, 0, 0, 0, 22877, 0, 0, 0, 0, 0, 0, 0, 6031, 0, 0, 0, 0, 0, 27763, 0, 0, 0, 14381, 0, 0, 0, 0, 0, 14723, 0, 0, 0, 0, 0, 9287, 0, 0, 0, 0, 0, 22439, 0, 21121, 0, 0, 0, 0, 0, 3253, 0, 0, 0, 0, 0, 26527, 0, 0, 0, 6173, 0, 23899, 0, 0, 0, 3587, 0, 0, 0, 0, 0, 16739, 0, 20971, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17651, 0, 8227, 0, 0, 0, 11873, 0, 27949, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3149, 0, 21001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24341, 0, 29917, 0, 0, 0, 0, 0, 26659, 0, 0, 0, 6437, 0, 0, 0, 0, 0, 0, 0, 13501, 0, 0, 0, 0, 0, 2293, 0, 0, 0, 6551, 0, 8737, 0, 0, 0, 20093, 0, 0, 0, 0, 0, 15437, 0, 0, 0, 0, 0, 28649, 0, 0, 0, 0, 0, 0, 0, 10243, 0, 0, 0, 12671, 0, 7327, 0, 0, 0, 0, 0, 4309, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13499, 0, 0, 0, 0, 0, 0, 0, 5053, 0, 0, 0, 4331, 0, 16747, 0, 0, 0, 0, 0, 28009, 0, 0, 0, 13427, 0, 0, 0, 0, 0, 0, 0, 28081, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18791, 0, 0, 0, 0, 0, 27323, 0, 5809, 0, 0, 0, 10547, 0, 0, 0, 0, 0, 0, 0, 6001, 0, 0, 0, 0, 0, 7663, 0, 0, 0, 0, 0, 14557, 0, 0, 0, 2813, 0, 14239, 0, 0, 0, 9257, 0, 0, 0, 0, 0, 18839, 0, 19711, 0, 0, 0, 0, 0, 25903, 0, 0, 0, 21851, 0, 0, 0, 0, 0, 20903, 0, 17239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2099, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22871, 0, 8917, 0, 0, 0, 10253, 0, 16999, 0, 0, 0, 21557, 0, 0, 0, 0, 0, 11369, 0, 17911, 0, 0, 0, 0, 0, 17203, 0, 0, 0, 17681, 0, 12307, 0, 0, 0, 5813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27389, 0, 19141, 0, 0, 0, 0, 0, 2263, 0, 0, 0, 0, 0, 17617, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6887, 0, 0, 0, 0, 0, 14759, 0, 0, 0, 0, 0, 0, 0, 11773, 0, 0, 0, 24911, 0, 29347, 0, 0, 0, 19793, 0, 26959, 0, 0, 0, 29177, 0, 0, 0, 0, 0, 0, 0, 28921, 0, 0, 0, 0, 0, 25873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16099, 0, 0, 0, 11387, 0, 0, 0, 0, 0, 13619, 0, 21751, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5107, 0, 0, 0, 9533, 0, 1429, 0, 0, 0, 1907, 0, 0, 0, 0, 0, 23039, 0, 0, 0, 0, 0, 0, 0, 22123, 0, 0, 0, 2111, 0, 22507, 0, 0, 0, 12923, 0, 20809, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19741, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26441, 0, 20047, 0, 0, 0, 25163, 0, 19099, 0, 0, 0, 25967, 0, 0, 0, 0, 0, 29429, 0, 3001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26711, 0, 10837, 0, 0, 0, 4553, 0, 0, 0, 0, 0, 14177, 0, 0, 0, 0, 0, 0, 0, 4021, 0, 0, 0, 0, 0, 11293, 0, 0, 0, 26951, 0, 21817, 0, 0, 0, 0, 0, 22159, 0, 0, 0, 5627, 0, 0, 0, 0, 0, 23789, 0, 0, 0, 0, 0, 0, 0, 8413, 0, 0, 0, 12941, 0, 0, 0, 0, 0, 24203, 0, 9319, 0, 0, 0, 14057, 0, 0, 0, 0, 0, 0, 0, 151, 0, 0, 0, 0, 0, 8443, 0, 0, 0, 9341, 0, 0, 0, 0, 0, 25433, 0, 6439, 0, 0, 0, 3407, 0, 0, 0, 0, 0, 23909, 0, 12301, 0, 0, 0, 0, 0, 3673, 0, 0, 0, 0, 0, 1747, 0, 0, 0, 2183, 0, 0, 0, 0, 0, 8627, 0, 0, 0, 0, 0, 5609, 0, 0, 0, 0, 0, 0, 0, 7213, 0, 0, 0, 0, 0, 9697, 0, 0, 0, 13763, 0, 27529, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22679, 0, 7771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23921, 0, 7237, 0, 0, 0, 10883, 0, 4189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19979, 0, 26941, 0, 0, 0, 0, 0, 29173, 0, 0, 0, 27971, 0, 7477, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10397, 0, 0, 0, 0, 0, 11639, 0, 26701, 0, 0, 0, 0, 0, 1633, 0, 0, 0, 25691, 0, 26227, 0, 0, 0, 0, 0, 4729, 0, 0, 0, 21377, 0, 0, 0, 0, 0, 4889, 0, 0, 0, 0, 0, 0, 0, 20803, 0, 0, 0, 0, 0, 8977, 0, 0, 0, 9293, 0, 5959, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8291, 0, 0, 0, 0, 0, 10313, 0, 15469, 0, 0, 0, 12437, 0, 0, 0, 0, 0, 28109, 0, 11251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 277, 0, 0, 0, 7853, 0, 13399, 0, 0, 0, 5267, 0, 0, 0, 0, 0, 0, 0, 11941, 0, 0, 0, 0, 0, 9313, 0, 0, 0, 23321, 0, 20827, 0, 0, 0, 15023, 0, 13249, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27089, 0, 6511, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16571, 0, 0, 0, 0, 0, 1433, 0, 409, 0, 0, 0, 21977, 0, 0, 0, 0, 0, 16619, 0, 28201, 0, 0, 0, 0, 0, 16363, 0, 0, 0, 21881, 0, 28057, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25517, 0, 0, 0, 0, 0, 28859, 0, 24181, 0, 0, 0, 0, 0, 11923, 0, 0, 0, 13721, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28097, 0, 0, 0, 0, 0, 22109, 0, 0, 0, 0, 0, 0, 0, 19753, 0, 0, 0, 0, 0, 21577, 0, 0, 0, 3203, 0, 12679, 0, 0, 0, 26027, 0, 0, 0, 0, 0, 0, 0, 6451, 0, 0, 0, 0, 0, 26293, 0, 0, 0, 28871, 0, 0, 0, 0, 0, 2963, 0, 7489, 0, 0, 0, 2777, 0, 0, 0, 0, 0, 2489, 0, 0, 0, 0, 0, 0, 0, 4093, 0, 0, 0, 0, 0, 3847, 0, 0, 0, 0, 0, 28309, 0, 0, 0, 17657, 0, 0, 0, 0, 0, 15269, 0, 30001, 0, 0, 0, 0, 0, 29683, 0, 0, 0, 28991, 0, 29857, 0, 0, 0, 17123, 0, 11989, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1259, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1451, 0, 21517, 0, 0, 0, 383, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15149, 0, 7201, 0, 0, 0, 0, 0, 6703, 0, 0, 0, 6971, 0, 16087, 0, 0, 0, 26813, 0, 0, 0, 0, 0, 27617, 0, 0, 0, 0, 0, 9959, 0, 27751, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16777, 0, 0, 0, 0, 0, 2839, 0, 0, 0, 1637, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5171, 0, 25777, 0, 0, 0, 4463, 0, 5329, 0, 0, 0, 13217, 0, 0, 0, 0, 0, 0, 0, 19051, 0, 0, 0, 0, 0, 25243, 0, 0, 0, 14591, 0, 0, 0, 0, 0, 443, 0, 8329, 0, 0, 0, 13067, 0, 0, 0, 0, 0, 0, 0, 1801, 0, 0, 0, 0, 0, 14383, 0, 0, 0, 0, 0, 1327, 0, 0, 0, 29063, 0, 14029, 0, 0, 0, 16607, 0, 0, 0, 0, 0, 27869, 0, 20551, 0, 0, 0, 0, 0, 16873, 0, 0, 0, 21641, 0, 1087, 0, 0, 0, 21323, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11549, 0, 20791, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20257, 0, 0, 0, 0, 0, 29179, 0, 0, 0, 22397, 0, 0, 0, 0, 0, 18719, 0, 18331, 0, 0, 0, 0, 0, 8803, 0, 0, 0, 11381, 0, 0, 0, 0, 0, 14843, 0, 0, 0, 0, 0, 17957, 0, 0, 0, 0, 0, 18989, 0, 26281, 0, 0, 0, 0, 0, 12343, 0, 0, 0, 12461, 0, 15727, 0, 0, 0, 0, 0, 17749, 0, 0, 0, 20957, 0, 0, 0, 0, 0, 6359, 0, 0, 0, 0, 0, 0, 0, 7573, 0, 0, 0, 3911, 0, 27877, 0, 0, 0, 13493, 0, 10999, 0, 0, 0, 3557, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7603, 0, 0, 0, 29501, 0, 0, 0, 0, 0, 4013, 0, 9589, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23069, 0, 6211, 0, 0, 0, 0, 0, 4723, 0, 0, 0, 0, 0, 24007, 0, 0, 0, 9323, 0, 4579, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2039, 0, 17191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28361, 0, 25027, 0, 0, 0, 3893, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7981, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29591, 0, 8287, 0, 0, 0, 20963, 0, 17209, 0, 0, 0, 2657, 0, 0, 0, 0, 0, 16199, 0, 5521, 0, 0, 0, 0, 0, 23923, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5393, 0, 0, 0, 0, 0, 5147, 0, 0, 0, 0, 0, 20249, 0, 22921, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8681, 0, 6277, 0, 0, 0, 0, 0, 10399, 0, 0, 0, 24527, 0, 0, 0, 0, 0, 5099, 0, 0, 0, 0, 0, 0, 0, 20173, 0, 0, 0, 19661, 0, 29137, 0, 0, 0, 24623, 0, 0, 0, 0, 0, 5657, 0, 0, 0, 0, 0, 0, 0, 28711, 0, 0, 0, 0, 0, 4243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2543, 0, 11689, 0, 0, 0, 13487, 0, 0, 0, 0, 0, 22019, 0, 10411, 0, 0, 0, 0, 0, 13753, 0, 0, 0, 0, 0, 8887, 0, 0, 0, 13523, 0, 3529, 0, 0, 0, 4427, 0, 0, 0, 0, 0, 4139, 0, 15721, 0, 0, 0, 0, 0, 28843, 0, 0, 0, 25001, 0, 10117, 0, 0, 0, 0, 0, 4219, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2729, 0, 19111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12161, 0, 0, 0, 0, 0, 25373, 0, 15949, 0, 0, 0, 23027, 0, 0, 0, 0, 0, 8849, 0, 0, 0, 0, 0, 0, 0, 20143, 0, 0, 0, 7391, 0, 8317, 0, 0, 0, 2033, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14159, 0, 13471, 0, 0, 0, 0, 0, 8353, 0, 0, 0, 23801, 0, 22027, 0, 0, 0, 0, 0, 7879, 0, 0, 0, 1217, 0, 0, 0, 0, 0, 23159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26381, 0, 29977, 0, 0, 0, 11813, 0, 2179, 0, 0, 0, 20777, 0, 0, 0, 0, 0, 0, 0, 11281, 0, 0, 0, 0, 0, 10963, 0, 0, 0, 11441, 0, 0, 0, 0, 0, 29213, 0, 15889, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12569, 0, 20701, 0, 0, 0, 0, 0, 5773, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17933, 0, 0, 0, 0, 0, 17027, 0, 0, 0, 0, 0, 5399, 0, 19291, 0, 0, 0, 0, 0, 13723, 0, 0, 0, 7361, 0, 14527, 0, 0, 0, 13553, 0, 22279, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8399, 0, 28141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7027, 0, 0, 0, 29573, 0, 0, 0, 0, 0, 2027, 0, 0, 0, 0, 0, 23759, 0, 19801, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7057, 0, 0, 0, 563, 0, 0, 0, 0, 0, 17117, 0, 0, 0, 0, 0, 29279, 0, 1501, 0, 0, 0, 0, 0, 12763, 0, 0, 0, 25901, 0, 6907, 0, 0, 0, 0, 0, 11449, 0, 0, 0, 16967, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18283, 0, 0, 0, 29951, 0, 21997, 0, 0, 0, 0, 0, 12469, 0, 0, 0, 16997, 0, 0, 0, 0, 0, 0, 0, 15481, 0, 0, 0, 0, 0, 3613, 0, 0, 0, 28661, 0, 0, 0, 0, 0, 19763, 0, 28489, 0, 0, 0, 9497, 0, 0, 0, 0, 0, 28319, 0, 0, 0, 0, 0, 0, 0, 6613, 0, 0, 0, 0, 0, 6997, 0, 0, 0, 0, 0, 24109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6239, 0, 16771, 0, 0, 0, 0, 0, 29473, 0, 0, 0, 6311, 0, 0, 0, 0, 0, 26153, 0, 12829, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1049, 0, 29401, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30011, 0, 3247, 0, 0, 0, 9833, 0, 199, 0, 0, 0, 24077, 0, 0, 0, 0, 0, 0, 0, 18541, 0, 0, 0, 0, 0, 18883, 0, 0, 0, 7811, 0, 22807, 0, 0, 0, 4133, 0, 0, 0, 0, 0, 1997, 0, 0, 0, 0, 0, 3659, 0, 0, 0, 0, 0, 0, 0, 27883, 0, 0, 0, 16241, 0, 0, 0, 0, 0, 0, 0, 9979, 0, 0, 0, 15707, 0, 0, 0, 0, 0, 21899, 0, 0, 0, 0, 0, 0, 0, 3163, 0, 0, 0, 19241, 0, 997, 0, 0, 0, 26093, 0, 16669, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17581, 0, 0, 0, 0, 0, 9283, 0, 0, 0, 19001, 0, 0, 0, 0, 0, 7793, 0, 0, 0, 0, 0, 13907, 0, 0, 0, 0, 0, 13199, 0, 11881, 0, 0, 0, 0, 0, 7873, 0, 0, 0, 0, 0, 19597, 0, 0, 0, 8483, 0, 28519, 0, 0, 0, 15137, 0, 0, 0, 0, 0, 28289, 0, 9421, 0, 0, 0, 0, 0, 20023, 0, 0, 0, 0, 0, 10537, 0, 0, 0, 323, 0, 9469, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7769, 0, 23311, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28961, 0, 18367, 0, 0, 0, 0, 0, 17419, 0, 0, 0, 24917, 0, 0, 0, 0, 0, 21869, 0, 6571, 0, 0, 0, 0, 0, 6913, 0, 0, 0, 18101, 0, 29527, 0, 0, 0, 1613, 0, 0, 0, 0, 0, 17747, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21793, 0, 0, 0, 3431, 0, 18877, 0, 0, 0, 0, 0, 6619, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18119, 0, 0, 0, 0, 0, 0, 0, 25843, 0, 0, 0, 22811, 0, 21367, 0, 0, 0, 24833, 0, 21079, 0, 0, 0, 15737, 0, 0, 0, 0, 0, 0, 0, 18841, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21101, 0, 0, 0, 0, 0, 0, 0, 8119, 0, 0, 0, 29027, 0, 0, 0, 0, 0, 0, 0, 17551, 0, 0, 0, 0, 0, 9973, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7433, 0, 2689, 0, 0, 0, 17, 0, 0, 0, 0, 0, 7289, 0, 24331, 0, 0, 0, 0, 0, 18973, 0, 0, 0, 24161, 0, 0, 0, 0, 0, 2423, 0, 7999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4409, 0, 8401, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11321, 0, 20467, 0, 0, 0, 1013, 0, 28549, 0, 0, 0, 26177, 0, 0, 0, 0, 0, 4439, 0, 0, 0, 0, 0, 0, 0, 5653, 0, 0, 0, 8441, 0, 3067, 0, 0, 0, 24293, 0, 0, 0, 0, 0, 8297, 0, 0, 0, 0, 0, 29699, 0, 23761, 0, 0, 0, 0, 0, 23053, 0, 0, 0, 0, 0, 6067, 0, 0, 0, 0, 0, 27199, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3209, 0, 0, 0, 0, 0, 0, 0, 23323, 0, 0, 0, 29531, 0, 1627, 0, 0, 0, 12863, 0, 0, 0, 0, 0, 3767, 0, 0, 0, 0, 0, 0, 0, 19681, 0, 0, 0, 0, 0, 463, 0, 0, 0, 6191, 0, 0, 0, 0, 0, 0, 0, 20719, 0, 0, 0, 22937, 0, 0, 0, 0, 0, 25169, 0, 10201, 0, 0, 0, 0, 0, 14173, 0, 0, 0, 0, 0, 7417, 0, 0, 0, 28013, 0, 12979, 0, 0, 0, 22697, 0, 0, 0, 0, 0, 27659, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6731, 0, 10957, 0, 0, 0, 3683, 0, 11569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16799, 0, 12811, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7961, 0, 10807, 0, 0, 0, 6683, 0, 7549, 0, 0, 0, 28277, 0, 0, 0, 0, 0, 0, 0, 5311, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17471, 0, 22387, 0, 0, 0, 9173, 0, 0, 0, 0, 0, 7247, 0, 0, 0, 0, 0, 0, 0, 22501, 0, 0, 0, 0, 0, 2053, 0, 0, 0, 15401, 0, 26437, 0, 0, 0, 0, 0, 15229, 0, 0, 0, 7937, 0, 0, 0, 0, 0, 5309, 0, 0, 0, 0, 0, 0, 0, 26893, 0, 0, 0, 15251, 0, 0, 0, 0, 0, 0, 0, 11629, 0, 0, 0, 2507, 0, 0, 0, 0, 0, 0, 0, 11701, 0, 0, 0, 0, 0, 10753, 0, 0, 0, 2411, 0, 0, 0, 0, 0, 23, 0, 24919, 0, 0, 0, 24197, 0, 0, 0, 0, 0, 12359, 0, 16921, 0, 0, 0, 0, 0, 26773, 0, 0, 0, 0, 0, 6367, 0, 0, 0, 13733, 0, 0, 0, 0, 0, 20177, 0, 0, 0, 0, 0, 7919, 0, 22441, 0, 0, 0, 0, 0, 9523, 0, 0, 0, 0, 0, 21247, 0, 0, 0, 4523, 0, 9049, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15749, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14681, 0, 28027, 0, 0, 0, 29363, 0, 8809, 0, 0, 0, 2447, 0, 0, 0, 0, 0, 22289, 0, 1531, 0, 0, 0, 0, 0, 19933, 0, 0, 0, 9491, 0, 25957, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 89, 0, 8221, 0, 0, 0, 0, 0, 13183, 0, 0, 0, 281, 0, 28537, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25997, 0, 0, 0, 0, 0, 25679, 0, 0, 0, 0, 0, 0, 0, 25423, 0, 0, 0, 13991, 0, 2047, 0, 0, 0, 11603, 0, 10579, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20731, 0, 0, 0, 0, 0, 14953, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28793, 0, 27019, 0, 0, 0, 3197, 0, 0, 0, 0, 0, 2699, 0, 13561, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18757, 0, 0, 0, 28643, 0, 4159, 0, 0, 0, 23747, 0, 0, 0, 0, 0, 0, 0, 5011, 0, 0, 0, 0, 0, 13933, 0, 0, 0, 4841, 0, 11587, 0, 0, 0, 26573, 0, 1699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8821, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7331, 0, 0, 0, 0, 0, 6053, 0, 2719, 0, 0, 0, 15047, 0, 0, 0, 0, 0, 21239, 0, 16651, 0, 0, 0, 0, 0, 7123, 0, 0, 0, 10331, 0, 2647, 0, 0, 0, 7283, 0, 0, 0, 0, 0, 27827, 0, 0, 0, 0, 0, 10379, 0, 14941, 0, 0, 0, 0, 0, 373, 0, 0, 0, 16031, 0, 13627, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16547, 0, 0, 0, 0, 0, 12869, 0, 0, 0, 0, 0, 0, 0, 22063, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7823, 0, 1129, 0, 0, 0, 16787, 0, 0, 0, 0, 0, 0, 0, 11071, 0, 0, 0, 0, 0, 19363, 0, 0, 0, 1151, 0, 0, 0, 0, 0, 14513, 0, 28279, 0, 0, 0, 14327, 0, 0, 0, 0, 0, 4799, 0, 25951, 0, 0, 0, 0, 0, 6403, 0, 0, 0, 0, 0, 15397, 0, 0, 0, 0, 0, 9829, 0, 0, 0, 22277, 0, 0, 0, 0, 0, 8339, 0, 0, 0, 0, 0, 0, 0, 18133, 0, 0, 0, 19751, 0, 20617, 0, 0, 0, 5573, 0, 16609, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13001, 0, 9967, 0, 0, 0, 2693, 0, 17839, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3599, 0, 18751, 0, 0, 0, 0, 0, 18253, 0, 0, 0, 11591, 0, 18397, 0, 0, 0, 19883, 0, 0, 0, 0, 0, 2207, 0, 0, 0, 0, 0, 719, 0, 18511, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6581, 0, 9847, 0, 0, 0, 0, 0, 7459, 0, 0, 0, 13187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12613, 0, 0, 0, 0, 0, 28087, 0, 0, 0, 22943, 0, 16879, 0, 0, 0, 3977, 0, 0, 0, 0, 0, 0, 0, 21361, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19211, 0, 0, 0, 0, 0, 21233, 0, 29119, 0, 0, 0, 1517, 0, 0, 0, 0, 0, 19919, 0, 24901, 0, 0, 0, 0, 0, 19003, 0, 0, 0, 0, 0, 22117, 0, 0, 0, 10583, 0, 2479, 0, 0, 0, 18917, 0, 0, 0, 0, 0, 0, 0, 22861, 0, 0, 0, 0, 0, 28423, 0, 0, 0, 12401, 0, 12637, 0, 0, 0, 25943, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16169, 0, 9241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24877, 0, 0, 0, 15083, 0, 22249, 0, 0, 0, 24707, 0, 0, 0, 0, 0, 239, 0, 9091, 0, 0, 0, 0, 0, 27283, 0, 0, 0, 13691, 0, 17137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6407, 0, 0, 0, 0, 0, 9749, 0, 7801, 0, 0, 0, 0, 0, 14653, 0, 0, 0, 5531, 0, 0, 0, 0, 0, 0, 0, 6199, 0, 0, 0, 11717, 0, 0, 0, 0, 0, 24839, 0, 0, 0, 0, 0, 0, 0, 643, 0, 0, 0, 6221, 0, 2467, 0, 0, 0, 25043, 0, 1759, 0, 0, 0, 15107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9913, 0, 0, 0, 17951, 0, 0, 0, 0, 0, 24803, 0, 21139, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16139, 0, 0, 0, 0, 0, 0, 0, 23203, 0, 0, 0, 0, 0, 14767, 0, 0, 0, 27803, 0, 9199, 0, 0, 0, 9467, 0, 0, 0, 0, 0, 4349, 0, 21811, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9881, 0, 13477, 0, 0, 0, 6203, 0, 14719, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19531, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4181, 0, 10597, 0, 0, 0, 14033, 0, 0, 0, 0, 0, 7277, 0, 0, 0, 0, 0, 6959, 0, 26311, 0, 0, 0, 0, 0, 28543, 0, 0, 0, 17891, 0, 0, 0, 0, 0, 7703, 0, 0, 0, 0, 0, 16697, 0, 0, 0, 0, 0, 1769, 0, 11371, 0, 0, 0, 0, 0, 13393, 0, 0, 0, 0, 0, 8587, 0, 0, 0, 0, 0, 21949, 0, 0, 0, 15287, 0, 0, 0, 0, 0, 9719, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24281, 0, 17587, 0, 0, 0, 15383, 0, 0, 0, 0, 0, 24137, 0, 0, 0, 0, 0, 0, 0, 21781, 0, 0, 0, 0, 0, 8863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14093, 0, 139, 0, 0, 0, 15797, 0, 0, 0, 0, 0, 0, 0, 12721, 0, 0, 0, 0, 0, 25303, 0, 0, 0, 0, 0, 20437, 0, 0, 0, 18143, 0, 5839, 0, 0, 0, 27527, 0, 0, 0, 0, 0, 8759, 0, 4171, 0, 0, 0, 0, 0, 19603, 0, 0, 0, 13451, 0, 14737, 0, 0, 0, 0, 0, 27319, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14279, 0, 9871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14471, 0, 1147, 0, 0, 0, 0, 0, 18259, 0, 0, 0, 11477, 0, 0, 0, 0, 0, 29639, 0, 0, 0, 0, 0, 0, 0, 22453, 0, 0, 0, 461, 0, 0, 0, 0, 0, 6653, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2609, 0, 18091, 0, 0, 0, 0, 0, 1423, 0, 0, 0, 26111, 0, 26647, 0, 0, 0, 0, 0, 28669, 0, 0, 0, 12767, 0, 0, 0, 0, 0, 25469, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14831, 0, 11497, 0, 0, 0, 2573, 0, 13729, 0, 0, 0, 25397, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29443, 0, 0, 0, 2201, 0, 0, 0, 0, 0, 17663, 0, 20509, 0, 0, 0, 14537, 0, 0, 0, 0, 0, 14879, 0, 25321, 0, 0, 0, 0, 0, 26563, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20243, 0, 18229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23879, 0, 811, 0, 0, 0, 0, 0, 25273, 0, 0, 0, 11981, 0, 16837, 0, 0, 0, 6623, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29189, 0, 18901, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18671, 0, 97, 0, 0, 0, 1853, 0, 0, 0, 0, 0, 13577, 0, 0, 0, 0, 0, 5279, 0, 8251, 0, 0, 0, 0, 0, 7543, 0, 0, 0, 0, 0, 9367, 0, 0, 0, 19043, 0, 0, 0, 0, 0, 7877, 0, 0, 0, 0, 0, 3869, 0, 3811, 0, 0, 0, 0, 0, 0, 0, 0, 0, 491, 0, 25387, 0, 0, 0, 0, 0, 2209, 0, 0, 0, 5417, 0, 0, 0, 0, 0, 16019, 0, 0, 0, 0, 0, 0, 0, 22903, 0, 0, 0, 11471, 0, 12757, 0, 0, 0, 0, 0, 919, 0, 0, 0, 19307, 0, 0, 0, 0, 0, 0, 0, 17791, 0, 0, 0, 0, 0, 15163, 0, 0, 0, 19421, 0, 0, 0, 0, 0, 24383, 0, 769, 0, 0, 0, 2567, 0, 0, 0, 0, 0, 2909, 0, 0, 0, 0, 0, 0, 0, 27403, 0, 0, 0, 0, 0, 11617, 0, 0, 0, 5333, 0, 17179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17789, 0, 7531, 0, 0, 0, 0, 0, 17923, 0, 0, 0, 8621, 0, 21037, 0, 0, 0, 0, 0, 15139, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21839, 0, 10921, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23081, 0, 0, 0, 0, 0, 14453, 0, 18679, 0, 0, 0, 14837, 0, 0, 0, 0, 0, 0, 0, 23161, 0, 0, 0, 0, 0, 11953, 0, 0, 0, 10121, 0, 27427, 0, 0, 0, 15683, 0, 0, 0, 0, 0, 13547, 0, 0, 0, 0, 0, 5969, 0, 2551, 0, 0, 0, 0, 0, 163, 0, 0, 0, 4691, 0, 0, 0, 0, 0, 0, 0, 21529, 0, 0, 0, 20327, 0, 0, 0, 0, 0, 14969, 0, 0, 0, 0, 0, 0, 0, 21643, 0, 0, 0, 10001, 0, 21787, 0, 0, 0, 14543, 0, 21289, 0, 0, 0, 4397, 0, 0, 0, 0, 0, 0, 0, 22201, 0, 0, 0, 0, 0, 43, 0, 0, 0, 521, 0, 0, 0, 0, 0, 10103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1649, 0, 23431, 0, 0, 0, 0, 0, 19423, 0, 0, 0, 0, 0, 21907, 0, 0, 0, 1553, 0, 0, 0, 0, 0, 19757, 0, 0, 0, 0, 0, 19049, 0, 181, 0, 0, 0, 0, 0, 24643, 0, 0, 0, 29201, 0, 3607, 0, 0, 0, 2633, 0, 14089, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19319, 0, 11761, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20677, 0, 0, 0, 0, 0, 28969, 0, 0, 0, 15677, 0, 0, 0, 0, 0, 26489, 0, 8881, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22721, 0, 17977, 0, 0, 0, 22403, 0, 0, 0, 0, 0, 6197, 0, 0, 0, 0, 0, 10169, 0, 0, 0, 0, 0, 0, 0, 26413, 0, 0, 0, 14981, 0, 9637, 0, 0, 0, 0, 0, 25099, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7363, 0, 0, 0, 13571, 0, 2887, 0, 0, 0, 29453, 0, 23389, 0, 0, 0, 8807, 0, 0, 0, 0, 0, 0, 0, 7291, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9551, 0, 0, 0, 0, 0, 8843, 0, 1189, 0, 0, 0, 1307, 0, 0, 0, 0, 0, 0, 0, 8311, 0, 0, 0, 0, 0, 28453, 0, 0, 0, 0, 0, 26107, 0, 0, 0, 0, 0, 4999, 0, 0, 0, 18497, 0, 0, 0, 0, 0, 28079, 0, 5851, 0, 0, 0, 0, 0, 21283, 0, 0, 0, 17231, 0, 0, 0, 0, 0, 7043, 0, 26479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22889, 0, 13021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13631, 0, 25087, 0, 0, 0, 12563, 0, 19309, 0, 0, 0, 7697, 0, 0, 0, 0, 0, 6749, 0, 29461, 0, 0, 0, 0, 0, 7963, 0, 0, 0, 26921, 0, 14617, 0, 0, 0, 15053, 0, 0, 0, 0, 0, 12917, 0, 0, 0, 0, 0, 22769, 0, 0, 0, 0, 0, 0, 0, 11503, 0, 0, 0, 0, 0, 26857, 0, 0, 0, 0, 0, 1789, 0, 0, 0, 18437, 0, 0, 0, 0, 0, 5519, 0, 0, 0, 0, 0, 0, 0, 14083, 0, 0, 0, 11051, 0, 20107, 0, 0, 0, 15173, 0, 8479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1201, 0, 0, 0, 0, 0, 12013, 0, 0, 0, 10811, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27557, 0, 0, 0, 0, 0, 15929, 0, 961, 0, 0, 0, 0, 0, 18793, 0, 0, 0, 0, 0, 487, 0, 0, 0, 293, 0, 17599, 0, 0, 0, 4217, 0, 0, 0, 0, 0, 9179, 0, 0, 0, 0, 0, 0, 0, 3643, 0, 0, 0, 0, 0, 13267, 0, 0, 0, 22163, 0, 23119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21419, 0, 15121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12581, 0, 29287, 0, 0, 0, 27473, 0, 28339, 0, 0, 0, 16727, 0, 0, 0, 0, 0, 10949, 0, 28411, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29021, 0, 13147, 0, 0, 0, 20723, 0, 0, 0, 0, 0, 9557, 0, 0, 0, 0, 0, 0, 0, 24811, 0, 0, 0, 0, 0, 13603, 0, 0, 0, 6161, 0, 7957, 0, 0, 0, 0, 0, 17539, 0, 0, 0, 1007, 0, 0, 0, 0, 0, 9929, 0, 0, 0, 0, 0, 0, 0, 17653, 0, 0, 0, 3701, 0, 0, 0, 0, 0, 5723, 0, 4699, 0, 0, 0, 4817, 0, 0, 0, 0, 0, 0, 0, 2461, 0, 0, 0, 0, 0, 29233, 0, 0, 0, 4721, 0, 0, 0, 0, 0, 0, 0, 27229, 0, 0, 0, 12647, 0, 0, 0, 0, 0, 3119, 0, 28471, 0, 0, 0, 0, 0, 29083, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18353, 0, 0, 0, 0, 0, 10937, 0, 0, 0, 0, 0, 26399, 0, 27061, 0, 0, 0, 0, 0, 2593, 0, 0, 0, 0, 0, 25867, 0, 0, 0, 16073, 0, 29839, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18059, 0, 19321, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3131, 0, 9547, 0, 0, 0, 20123, 0, 20359, 0, 0, 0, 7067, 0, 0, 0, 0, 0, 15359, 0, 0, 0, 0, 0, 0, 0, 8383, 0, 0, 0, 251, 0, 16717, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21947, 0, 0, 0, 0, 0, 2399, 0, 12841, 0, 0, 0, 0, 0, 3943, 0, 0, 0, 11831, 0, 16987, 0, 0, 0, 0, 0, 16279, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14129, 0, 0, 0, 0, 0, 0, 0, 6943, 0, 0, 0, 18611, 0, 4357, 0, 0, 0, 4673, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11491, 0, 0, 0, 0, 0, 19573, 0, 0, 0, 19841, 0, 0, 0, 0, 0, 1073, 0, 1609, 0, 0, 0, 14747, 0, 0, 0, 0, 0, 14249, 0, 2011, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21067, 0, 0, 0, 17093, 0, 15709, 0, 0, 0, 14507, 0, 0, 0, 0, 0, 0, 0, 7321, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9461, 0, 37, 0, 0, 0, 17333, 0, 22489, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8609, 0, 1891, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18881, 0, 0, 0, 0, 0, 17603, 0, 21199, 0, 0, 0, 17357, 0, 0, 0, 0, 0, 0, 0, 18961, 0, 0, 0, 0, 0, 18673, 0, 0, 0, 1091, 0, 14197, 0, 0, 0, 11903, 0, 0, 0, 0, 0, 20897, 0, 0, 0, 0, 0, 14999, 0, 3391, 0, 0, 0, 0, 0, 21163, 0, 0, 0, 4481, 0, 18247, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18857, 0, 0, 0, 0, 0, 24419, 0, 0, 0, 0, 0, 0, 0, 10513, 0, 0, 0, 0, 0, 3097, 0, 0, 0, 0, 0, 3439, 0, 0, 0, 5237, 0, 0, 0, 0, 0, 0, 0, 22621, 0, 0, 0, 0, 0, 21673, 0, 0, 0, 24251, 0, 0, 0, 0, 0, 19133, 0, 16729, 0, 0, 0, 5087, 0, 0, 0, 0, 0, 23279, 0, 541, 0, 0, 0, 0, 0, 29503, 0, 0, 0, 0, 0, 20017, 0, 0, 0, 0, 0, 589, 0, 0, 0, 3797, 0, 0, 0, 0, 0, 10649, 0, 11521, 0, 0, 0, 0, 0, 20443, 0, 0, 0, 8201, 0, 2137, 0, 0, 0, 26363, 0, 28159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26669, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3761, 0, 727, 0, 0, 0, 21173, 0, 22459, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5909, 0, 23371, 0, 0, 0, 0, 0, 9013, 0, 0, 0, 23141, 0, 6847, 0, 0, 0, 22193, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19199, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11201, 0, 12157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17807, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17233, 0, 0, 0, 16721, 0, 21157, 0, 0, 0, 25253, 0, 21499, 0, 0, 0, 15527, 0, 0, 0, 0, 0, 0, 0, 9811, 0, 0, 0, 0, 0, 6763, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9683, 0, 10639, 0, 0, 0, 22307, 0, 0, 0, 0, 0, 24539, 0, 27211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10567, 0, 0, 0, 1343, 0, 23269, 0, 0, 0, 7367, 0, 0, 0, 0, 0, 9389, 0, 15931, 0, 0, 0, 0, 0, 3013, 0, 0, 0, 23951, 0, 3397, 0, 0, 0, 7463, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11551, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6397, 0, 0, 0, 19703, 0, 24559, 0, 0, 0, 17777, 0, 0, 0, 0, 0, 4859, 0, 27571, 0, 0, 0, 0, 0, 18043, 0, 0, 0, 2141, 0, 21757, 0, 0, 0, 26393, 0, 0, 0, 0, 0, 8717, 0, 0, 0, 0, 0, 21299, 0, 28591, 0, 0, 0, 0, 0, 3103, 0, 0, 0, 7841, 0, 27277, 0, 0, 0, 0, 0, 8509, 0, 0, 0, 167, 0, 0, 0, 0, 0, 15599, 0, 0, 0, 0, 0, 0, 0, 2953, 0, 0, 0, 29321, 0, 0, 0, 0, 0, 29663, 0, 20239, 0, 0, 0, 5867, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2983, 0, 0, 0, 20261, 0, 0, 0, 0, 0, 6323, 0, 11899, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18449, 0, 17761, 0, 0, 0, 0, 0, 25513, 0, 0, 0, 0, 0, 26317, 0, 0, 0, 18563, 0, 20749, 0, 0, 0, 14087, 0, 0, 0, 0, 0, 27449, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 641, 0, 4237, 0, 0, 0, 24683, 0, 19339, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24151, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15731, 0, 29077, 0, 0, 0, 16343, 0, 28759, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25439, 0, 7831, 0, 0, 0, 0, 0, 10063, 0, 0, 0, 22511, 0, 0, 0, 0, 0, 773, 0, 0, 0, 0, 0, 21317, 0, 0, 0, 0, 0, 22559, 0, 2131, 0, 0, 0, 0, 0, 18013, 0, 0, 0, 20231, 0, 1657, 0, 0, 0, 0, 0, 26569, 0, 0, 0, 26837, 0, 0, 0, 0, 0, 21269, 0, 0, 0, 0, 0, 0, 0, 1693, 0, 0, 0, 0, 0, 19897, 0, 0, 0, 3833, 0, 0, 0, 0, 0, 14897, 0, 0, 0, 0, 0, 0, 0, 24091, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4853, 0, 20929, 0, 0, 0, 4247, 0, 0, 0, 0, 0, 3539, 0, 5791, 0, 0, 0, 0, 0, 29923, 0, 0, 0, 0, 0, 11197, 0, 0, 0, 29693, 0, 24319, 0, 0, 0, 29837, 0, 0, 0, 0, 0, 0, 0, 6481, 0, 0, 0, 0, 0, 1123, 0, 0, 0, 4211, 0, 26287, 0, 0, 0, 0, 0, 29629, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18899, 0, 28351, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2921, 0, 5767, 0, 0, 0, 6893, 0, 11329, 0, 0, 0, 13787, 0, 0, 0, 0, 0, 11159, 0, 0, 0, 0, 0, 0, 0, 10903, 0, 0, 0, 2771, 0, 19867, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23399, 0, 29641, 0, 0, 0, 0, 0, 3733, 0, 0, 0, 19181, 0, 0, 0, 0, 0, 0, 0, 17119, 0, 0, 0, 3527, 0, 0, 0, 0, 0, 13919, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17141, 0, 16117, 0, 0, 0, 14123, 0, 4489, 0, 0, 0, 6917, 0, 0, 0, 0, 0, 0, 0, 22831, 0, 0, 0, 0, 0, 1723, 0, 0, 0, 20681, 0, 0, 0, 0, 0, 8423, 0, 2029, 0, 0, 0, 19157, 0, 0, 0, 0, 0, 7949, 0, 0, 0, 0, 0, 0, 0, 15013, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8693, 0, 22849, 0, 0, 0, 28577, 0, 0, 0, 0, 0, 26189, 0, 5431, 0, 0, 0, 0, 0, 16033, 0, 0, 0, 23531, 0, 5287, 0, 0, 0, 8933, 0, 3799, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17639, 0, 421, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23291, 0, 2407, 0, 0, 0, 24953, 0, 0, 0, 0, 0, 18197, 0, 0, 0, 0, 0, 26069, 0, 29041, 0, 0, 0, 0, 0, 12163, 0, 0, 0, 0, 0, 2437, 0, 0, 0, 21353, 0, 0, 0, 0, 0, 19427, 0, 0, 0, 0, 0, 15419, 0, 22291, 0, 0, 0, 0, 0, 24313, 0, 0, 0, 0, 0, 27697, 0, 0, 0, 0, 0, 13759, 0, 0, 0, 26207, 0, 0, 0, 0, 0, 20639, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16091, 0, 1207, 0, 0, 0, 0, 0, 21709, 0, 0, 0, 7757, 0, 0, 0, 0, 0, 0, 0, 10861, 0, 0, 0, 0, 0, 19783, 0, 0, 0, 941, 0, 0, 0, 0, 0, 5903, 0, 19249, 0, 0, 0, 4877, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8923, 0, 0, 0, 0, 0, 23167, 0, 0, 0, 9953, 0, 19489, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22409, 0, 26011, 0, 0, 0, 0, 0, 8683, 0, 0, 0, 27101, 0, 25657, 0, 0, 0, 7673, 0, 8209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3359, 0, 1681, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25391, 0, 14797, 0, 0, 0, 0, 0, 20989, 0, 0, 0, 3287, 0, 0, 0, 0, 0, 0, 0, 4681, 0, 0, 0, 0, 0, 14263, 0, 0, 0, 3191, 0, 0, 0, 0, 0, 20303, 0, 0, 0, 0, 0, 4307, 0, 0, 0, 0, 0, 24449, 0, 7171, 0, 0, 0, 0, 0, 23263, 0, 0, 0, 7001, 0, 0, 0, 0, 0, 0, 0, 12289, 0, 0, 0, 1847, 0, 0, 0, 0, 0, 17279, 0, 0, 0, 0, 0, 0, 0, 23953, 0, 0, 0, 28481, 0, 3307, 0, 0, 0, 5303, 0, 2809, 0, 0, 0, 9017, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18523, 0, 0, 0, 21311, 0, 0, 0, 0, 0, 28583, 0, 0, 0, 0, 0, 25457, 0, 0, 0, 0, 0, 3959, 0, 28051, 0, 0, 0, 0, 0, 10183, 0, 0, 0, 0, 0, 10357, 0, 0, 0, 3863, 0, 10039, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7499, 0, 11731, 0, 0, 0, 0, 0, 6163, 0, 0, 0, 3791, 0, 5917, 0, 0, 0, 25733, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10079, 0, 2521, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10481, 0, 13747, 0, 0, 0, 0, 0, 3559, 0, 0, 0, 27227, 0, 0, 0, 0, 0, 8009, 0, 27361, 0, 0, 0, 0, 0, 18463, 0, 0, 0, 0, 0, 20287, 0, 0, 0, 10853, 0, 0, 0, 0, 0, 26987, 0, 0, 0, 0, 0, 14789, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19601, 0, 28117, 0, 0, 0, 0, 0, 15859, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29669, 0, 0, 0, 0, 0, 0, 0, 11983, 0, 0, 0, 25121, 0, 23677, 0, 0, 0, 10973, 0, 11839, 0, 0, 0, 11117, 0, 0, 0, 0, 0, 0, 0, 9601, 0, 0, 0, 0, 0, 0, 0, 0, 0, 311, 0, 0, 0, 0, 0, 13463, 0, 3499, 0, 0, 0, 24407, 0, 0, 0, 0, 0, 0, 0, 26791, 0, 0, 0, 0, 0, 19213, 0, 0, 0, 0, 0, 697, 0, 0, 0, 18983, 0, 28099, 0, 0, 0, 20807, 0, 0, 0, 0, 0, 9599, 0, 26641, 0, 0, 0, 0, 0, 9733, 0, 0, 0, 19541, 0, 0, 0, 0, 0, 0, 0, 28789, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13649, 0, 24571, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6701, 0, 0, 0, 0, 0, 17183, 0, 7759, 0, 0, 0, 28487, 0, 0, 0, 0, 0, 25229, 0, 4051, 0, 0, 0, 0, 0, 1033, 0, 0, 0, 29231, 0, 19237, 0, 0, 0, 26603, 0, 0, 0, 0, 0, 24467, 0, 0, 0, 0, 0, 25079, 0, 5281, 0, 0, 0, 0, 0, 13813, 0, 0, 0, 0, 0, 8377, 0, 0, 0, 0, 0, 13339, 0, 0, 0, 23057, 0, 0, 0, 0, 0, 28619, 0, 0, 0, 0, 0, 0, 0, 2533, 0, 0, 0, 1811, 0, 10867, 0, 0, 0, 3623, 0, 13099, 0, 0, 0, 15317, 0, 0, 0, 0, 0, 0, 0, 5821, 0, 0, 0, 0, 0, 2773, 0, 0, 0, 22361, 0, 0, 0, 0, 0, 0, 0, 2239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4379, 0, 12511, 0, 0, 0, 0, 0, 313, 0, 0, 0, 0, 0, 2797, 0, 0, 0, 23393, 0, 0, 0, 0, 0, 8837, 0, 0, 0, 0, 0, 29969, 0, 0, 0, 0, 0, 0, 0, 8263, 0, 0, 0, 18281, 0, 6337, 0, 0, 0, 24473, 0, 27739, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2939, 0, 3571, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1567, 0, 0, 0, 15923, 0, 9859, 0, 0, 0, 7487, 0, 0, 0, 0, 0, 15569, 0, 691, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3611, 0, 1597, 0, 0, 0, 11483, 0, 0, 0, 0, 0, 28037, 0, 0, 0, 0, 0, 0, 0, 17881, 0, 0, 0, 0, 0, 18223, 0, 0, 0, 17711, 0, 28747, 0, 0, 0, 0, 0, 5989, 0, 0, 0, 3317, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29203, 0, 0, 0, 24491, 0, 0, 0, 0, 0, 10343, 0, 7009, 0, 0, 0, 27917, 0, 0, 0, 0, 0, 0, 0, 20941, 0, 0, 0, 0, 0, 19993, 0, 0, 0, 23201, 0, 0, 0, 0, 0, 11573, 0, 20299, 0, 0, 0, 14957, 0, 0, 0, 0, 0, 14669, 0, 19231, 0, 0, 0, 0, 0, 17533, 0, 0, 0, 0, 0, 17917, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29417, 0, 0, 0, 0, 0, 17159, 0, 8581, 0, 0, 0, 0, 0, 4903, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20693, 0, 18289, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6509, 0, 23941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5441, 0, 14167, 0, 0, 0, 1643, 0, 11119, 0, 0, 0, 18617, 0, 0, 0, 0, 0, 17669, 0, 13081, 0, 0, 0, 0, 0, 10693, 0, 0, 0, 18731, 0, 28267, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26567, 0, 0, 0, 0, 0, 25499, 0, 0, 0, 0, 0, 0, 0, 22423, 0, 0, 0, 2591, 0, 7747, 0, 0, 0, 0, 0, 20899, 0, 0, 0, 7517, 0, 0, 0, 0, 0, 16439, 0, 0, 0, 0, 0, 0, 0, 27733, 0, 0, 0, 131, 0, 22837, 0, 0, 0, 6983, 0, 22129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23041, 0, 0, 0, 0, 0, 1093, 0, 0, 0, 24461, 0, 0, 0, 0, 0, 24173, 0, 0, 0, 0, 0, 19367, 0, 0, 0, 0, 0, 5009, 0, 22801, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14137, 0, 0, 0, 19403, 0, 20329, 0, 0, 0, 26057, 0, 0, 0, 0, 0, 0, 0, 25801, 0, 0, 0, 0, 0, 25483, 0, 0, 0, 0, 0, 2347, 0, 0, 0, 5783, 0, 4009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13229, 0, 4201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23501, 0, 0, 0, 0, 0, 8363, 0, 11959, 0, 0, 0, 5807, 0, 0, 0, 0, 0, 2759, 0, 12031, 0, 0, 0, 0, 0, 23293, 0, 0, 0, 12641, 0, 4957, 0, 0, 0, 23453, 0, 0, 0, 0, 0, 23207, 0, 0, 0, 0, 0, 0, 0, 5701, 0, 0, 0, 0, 0, 2683, 0, 0, 0, 25271, 0, 29797, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11927, 0, 0, 0, 0, 0, 29039, 0, 0, 0, 0, 0, 0, 0, 1273, 0, 0, 0, 0, 0, 7717, 0, 0, 0, 19373, 0, 26539, 0, 0, 0, 7547, 0, 0, 0, 0, 0, 0, 0, 13381, 0, 0, 0, 0, 0, 10123, 0, 0, 0, 26561, 0, 0, 0, 0, 0, 0, 0, 19039, 0, 0, 0, 23567, 0, 0, 0, 0, 0, 14039, 0, 12091, 0, 0, 0, 0, 0, 1783, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19069, 0, 0, 0, 24587, 0, 0, 0, 0, 0, 29129, 0, 16141, 0, 0, 0, 0, 0, 13513, 0, 0, 0, 10511, 0, 6757, 0, 0, 0, 7883, 0, 18919, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28979, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22241, 0, 12277, 0, 0, 0, 11933, 0, 3979, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29009, 0, 0, 0, 0, 0, 0, 0, 27493, 0, 0, 0, 13901, 0, 27637, 0, 0, 0, 10643, 0, 0, 0, 0, 0, 13757, 0, 0, 0, 0, 0, 21509, 0, 4651, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22751, 0, 607, 0, 0, 0, 0, 0, 19009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28783, 0, 0, 0, 21341, 0, 23467, 0, 0, 0, 18323, 0, 0, 0, 0, 0, 20147, 0, 0, 0, 0, 0, 0, 0, 571, 0, 0, 0, 0, 0, 11383, 0, 0, 0, 731, 0, 0, 0, 0, 0, 11993, 0, 15259, 0, 0, 0, 3827, 0, 0, 0, 0, 0, 6059, 0, 15661, 0, 0, 0, 0, 0, 523, 0, 0, 0, 0, 0, 12877, 0, 0, 0, 19823, 0, 4789, 0, 0, 0, 28157, 0, 0, 0, 0, 0, 14009, 0, 18241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28571, 0, 21877, 0, 0, 0, 28253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27719, 0, 4621, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27187, 0, 0, 0, 1223, 0, 13009, 0, 0, 0, 20087, 0, 0, 0, 0, 0, 0, 0, 29881, 0, 0, 0, 0, 0, 29593, 0, 0, 0, 22931, 0, 3277, 0, 0, 0, 983, 0, 0, 0, 0, 0, 1787, 0, 0, 0, 0, 0, 25919, 0, 17041, 0, 0, 0, 0, 0, 23893, 0, 0, 0, 26321, 0, 1867, 0, 0, 0, 0, 0, 1579, 0, 0, 0, 2477, 0, 0, 0, 0, 0, 27149, 0, 0, 0, 0, 0, 0, 0, 21433, 0, 0, 0, 1601, 0, 14017, 0, 0, 0, 0, 0, 22549, 0, 0, 0, 24347, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5293, 0, 0, 0, 13331, 0, 0, 0, 0, 0, 10943, 0, 349, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6899, 0, 22381, 0, 0, 0, 0, 0, 18583, 0, 0, 0, 0, 0, 907, 0, 0, 0, 83, 0, 11509, 0, 0, 0, 25637, 0, 0, 0, 0, 0, 29759, 0, 3331, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19121, 0, 15787, 0, 0, 0, 15443, 0, 859, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6491, 0, 19837, 0, 0, 0, 4793, 0, 3349, 0, 0, 0, 18827, 0, 0, 0, 0, 0, 27749, 0, 12451, 0, 0, 0, 0, 0, 823, 0, 0, 0, 4031, 0, 0, 0, 0, 0, 3083, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11009, 0, 13681, 0, 0, 0, 0, 0, 29563, 0, 0, 0, 24851, 0, 3967, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1427, 0, 0, 0, 0, 0, 12029, 0, 0, 0, 0, 0, 0, 0, 6313, 0, 0, 0, 5801, 0, 12967, 0, 0, 0, 6143, 0, 0, 0, 0, 0, 26447, 0, 0, 0, 0, 0, 0, 0, 12541, 0, 0, 0, 0, 0, 20413, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23333, 0, 2449, 0, 0, 0, 25037, 0, 0, 0, 0, 0, 8159, 0, 8101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29677, 0, 0, 0, 20453, 0, 15079, 0, 0, 0, 18287, 0, 0, 0, 0, 0, 20309, 0, 29581, 0, 0, 0, 0, 0, 5743, 0, 0, 0, 15761, 0, 17047, 0, 0, 0, 0, 0, 18079, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23711, 0, 17317, 0, 0, 0, 11513, 0, 13639, 0, 0, 0, 6857, 0, 0, 0, 0, 0, 15779, 0, 0, 0, 0, 0, 0, 0, 1663, 0, 0, 0, 21251, 0, 24487, 0, 0, 0, 18203, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4919, 0, 20401, 0, 0, 0, 0, 0, 22213, 0, 0, 0, 21491, 0, 8167, 0, 0, 0, 0, 0, 19429, 0, 0, 0, 22007, 0, 0, 0, 0, 0, 4679, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10211, 0, 0, 0, 0, 0, 18743, 0, 22969, 0, 0, 0, 27707, 0, 0, 0, 0, 0, 0, 0, 27451, 0, 0, 0, 0, 0, 24823, 0, 0, 0, 22991, 0, 0, 0, 0, 0, 19973, 0, 22819, 0, 0, 0, 677, 0, 0, 0, 0, 0, 10259, 0, 6841, 0, 0, 0, 0, 0, 17323, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29483, 0, 4369, 0, 0, 0, 3167, 0, 0, 0, 0, 0, 19259, 0, 0, 0, 0, 0, 0, 0, 4483, 0, 0, 0, 14291, 0, 26077, 0, 0, 0, 27413, 0, 8419, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19949, 0, 5041, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4811, 0, 20887, 0, 0, 0, 27263, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14519, 0, 10561, 0, 0, 0, 0, 0, 23713, 0, 0, 0, 0, 0, 4747, 0, 0, 0, 14423, 0, 0, 0, 0, 0, 24047, 0, 0, 0, 0, 0, 6179, 0, 13051, 0, 0, 0, 0, 0, 28933, 0, 0, 0, 12041, 0, 20767, 0, 0, 0, 0, 0, 18379, 0, 0, 0, 7727, 0, 0, 0, 0, 0, 2159, 0, 0, 0, 0, 0, 0, 0, 4423, 0, 0, 0, 0, 0, 3517, 0, 0, 0, 0, 0, 3229, 0, 0, 0, 28547, 0, 0, 0, 0, 0, 0, 0, 13171, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5561, 0, 0, 0, 0, 0, 26693, 0, 10009, 0, 0, 0, 23357, 0, 0, 0, 0, 0, 14459, 0, 0, 0, 0, 0, 0, 0, 13543, 0, 0, 0, 0, 0, 13927, 0, 0, 0, 21503, 0, 7939, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28321, 0, 0, 0, 0, 0, 20233, 0, 0, 0, 17861, 0, 7177, 0, 0, 0, 12293, 0, 10519, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7979, 0, 20161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13841, 0, 19417, 0, 0, 0, 26003, 0, 14059, 0, 0, 0, 5597, 0, 0, 0, 0, 0, 0, 0, 25471, 0, 0, 0, 0, 0, 2713, 0, 0, 0, 5501, 0, 8947, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22787, 0, 0, 0, 0, 0, 15209, 0, 18721, 0, 0, 0, 0, 0, 25573, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 739, 0, 0, 0, 22637, 0, 0, 0, 0, 0, 5729, 0, 0, 0, 0, 0, 0, 0, 17023, 0, 0, 0, 761, 0, 7927, 0, 0, 0, 16853, 0, 23599, 0, 0, 0, 20567, 0, 0, 0, 0, 0, 0, 0, 3721, 0, 0, 0, 0, 0, 20833, 0, 0, 0, 9761, 0, 0, 0, 0, 0, 19343, 0, 0, 0, 0, 0, 47, 0, 0, 0, 0, 0, 27059, 0, 0, 0, 0, 0, 0, 0, 28663, 0, 0, 0, 0, 0, 1117, 0, 0, 0, 22343, 0, 14659, 0, 0, 0, 1277, 0, 0, 0, 0, 0, 9809, 0, 16351, 0, 0, 0, 0, 0, 26953, 0, 0, 0, 15341, 0, 24397, 0, 0, 0, 28043, 0, 25639, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28559, 0, 14071, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15101, 0, 16057, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1817, 0, 0, 0, 0, 0, 28799, 0, 18121, 0, 0, 0, 0, 0, 23083, 0, 0, 0, 4241, 0, 13357, 0, 0, 0, 13163, 0, 0, 0, 0, 0, 8507, 0, 0, 0, 0, 0, 26339, 0, 0, 0, 0, 0, 0, 0, 7933, 0, 0, 0, 0, 0, 397, 0, 0, 0, 0, 0, 27409, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4259, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29741, 0, 12127, 0, 0, 0, 1733, 0, 2599, 0, 0, 0, 29597, 0, 0, 0, 0, 0, 0, 0, 2671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11861, 0, 0, 0, 0, 0, 25013, 0, 21979, 0, 0, 0, 26717, 0, 0, 0, 0, 0, 0, 0, 29101, 0, 0, 0, 0, 0, 733, 0, 0, 0, 0, 0, 12247, 0, 0, 0, 23603, 0, 379, 0, 0, 0, 13877, 0, 0, 0, 0, 0, 14219, 0, 15091, 0, 0, 0, 0, 0, 493, 0, 0, 0, 7991, 0, 0, 0, 0, 0, 18593, 0, 21859, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25199, 0, 15331, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9011, 0, 6607, 0, 0, 0, 0, 0, 10069, 0, 0, 0, 16937, 0, 0, 0, 0, 0, 15989, 0, 15601, 0, 0, 0, 0, 0, 3343, 0, 0, 0, 22301, 0, 0, 0, 0, 0, 1193, 0, 0, 0, 0, 0, 15227, 0, 0, 0, 0, 0, 13529, 0, 9901, 0, 0, 0, 0, 0, 6883, 0, 0, 0, 0, 0, 12997, 0, 0, 0, 0, 0, 4099, 0, 0, 0, 4577, 0, 0, 0, 0, 0, 899, 0, 0, 0, 0, 0, 0, 0, 4843, 0, 0, 0, 20291, 0, 22417, 0, 0, 0, 24413, 0, 24649, 0, 0, 0, 19937, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21253, 0, 0, 0, 13121, 0, 0, 0, 0, 0, 0, 0, 6859, 0, 0, 0, 9077, 0, 0, 0, 0, 0, 6689, 0, 17131, 0, 0, 0, 0, 0, 21103, 0, 0, 0, 0, 0, 21277, 0, 0, 0, 25703, 0, 29149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18419, 0, 0, 0, 0, 0, 0, 0, 19813, 0, 0, 0, 22901, 0, 8647, 0, 0, 0, 17543, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23729, 0, 24361, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24131, 0, 24667, 0, 0, 0, 18233, 0, 14479, 0, 0, 0, 19037, 0, 0, 0, 0, 0, 27119, 0, 19171, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3907, 0, 0, 0, 29963, 0, 0, 0, 0, 0, 18797, 0, 0, 0, 0, 0, 0, 0, 20191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22331, 0, 17197, 0, 0, 0, 0, 0, 26779, 0, 0, 0, 21797, 0, 0, 0, 0, 0, 21479, 0, 0, 0, 0, 0, 0, 0, 3793, 0, 0, 0, 6011, 0, 0, 0, 0, 0, 21893, 0, 25489, 0, 0, 0, 197, 0, 0, 0, 0, 0, 0, 0, 23251, 0, 0, 0, 0, 0, 1513, 0, 0, 0, 13961, 0, 0, 0, 0, 0, 16193, 0, 22609, 0, 0, 0, 8027, 0, 0, 0, 0, 0, 19289, 0, 7681, 0, 0, 0, 0, 0, 8293, 0, 0, 0, 0, 0, 22537, 0, 0, 0, 29903, 0, 0, 0, 0, 0, 1697, 0, 0, 0, 0, 0, 28709, 0, 29371, 0, 0, 0, 0, 0, 23383, 0, 0, 0, 0, 0, 7387, 0, 0, 0, 0, 0, 20599, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27299, 0, 5461, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28541, 0, 0, 0, 0, 0, 6263, 0, 29599, 0, 0, 0, 9377, 0, 0, 0, 0, 0, 6119, 0, 17701, 0, 0, 0, 0, 0, 3763, 0, 0, 0, 21041, 0, 2857, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8087, 0, 0, 0, 0, 0, 27809, 0, 24391, 0, 0, 0, 0, 0, 24733, 0, 0, 0, 21071, 0, 19297, 0, 0, 0, 0, 0, 2419, 0, 0, 0, 12137, 0, 0, 0, 0, 0, 9509, 0, 0, 0, 0, 0, 0, 0, 16183, 0, 0, 0, 20921, 0, 13597, 0, 0, 0, 25463, 0, 26749, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27661, 0, 0, 0, 0, 0, 21883, 0, 0, 0, 5981, 0, 0, 0, 0, 0, 26483, 0, 13159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23489, 0, 4321, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16447, 0, 0, 0, 12473, 0, 0, 0, 0, 0, 647, 0, 0, 0, 0, 0, 0, 0, 16561, 0, 0, 0, 0, 0, 73, 0, 0, 0, 21011, 0, 25447, 0, 0, 0, 8093, 0, 8629, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24779, 0, 22681, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26843, 0, 23509, 0, 0, 0, 26597, 0, 0, 0, 0, 0, 7379, 0, 14341, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17261, 0, 23437, 0, 0, 0, 14213, 0, 0, 0, 0, 0, 11657, 0, 0, 0, 0, 0, 26549, 0, 28801, 0, 0, 0, 0, 0, 7303, 0, 0, 0, 6791, 0, 20557, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12823, 0, 0, 0, 0, 0, 19267, 0, 0, 0, 23993, 0, 28849, 0, 0, 0, 617, 0, 0, 0, 0, 0, 0, 0, 1831, 0, 0, 0, 0, 0, 883, 0, 0, 0, 15011, 0, 0, 0, 0, 0, 653, 0, 12109, 0, 0, 0, 25877, 0, 0, 0, 0, 0, 25589, 0, 2851, 0, 0, 0, 0, 0, 20263, 0, 0, 0, 0, 0, 1537, 0, 0, 0, 0, 0, 21379, 0, 0, 0, 13037, 0, 0, 0, 0, 0, 19889, 0, 27691, 0, 0, 0, 0, 0, 15823, 0, 0, 0, 3581, 0, 0, 0, 0, 0, 12503, 0, 7369, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17429, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24551, 0, 16897, 0, 0, 0, 23483, 0, 24769, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1289, 0, 4891, 0, 0, 0, 0, 0, 29803, 0, 0, 0, 2351, 0, 9157, 0, 0, 0, 1403, 0, 0, 0, 0, 0, 18377, 0, 0, 0, 0, 0, 14579, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13511, 0, 21397, 0, 0, 0, 0, 0, 23629, 0, 0, 0, 29357, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19543, 0, 0, 0, 2861, 0, 11917, 0, 0, 0, 20633, 0, 3019, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12121, 0, 0, 0, 0, 0, 22933, 0, 0, 0, 5351, 0, 0, 0, 0, 0, 5063, 0, 0, 0, 0, 0, 8447, 0, 0, 0, 0, 0, 26849, 0, 6421, 0, 0, 0, 0, 0, 5143, 0, 0, 0, 0, 0, 5947, 0, 0, 0, 22133, 0, 9409, 0, 0, 0, 9677, 0, 0, 0, 0, 0, 25559, 0, 6691, 0, 0, 0, 0, 0, 14563, 0, 0, 0, 0, 0, 24187, 0, 0, 0, 16703, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2309, 0, 6931, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15637, 0, 0, 0, 22013, 0, 3769, 0, 0, 0, 8537, 0, 0, 0, 0, 0, 16409, 0, 22951, 0, 0, 0, 0, 0, 4183, 0, 0, 0, 4451, 0, 24067, 0, 0, 0, 12533, 0, 0, 0, 0, 0, 4097, 0, 0, 0, 0, 0, 0, 0, 19351, 0, 0, 0, 0, 0, 5413, 0, 0, 0, 17081, 0, 13417, 0, 0, 0, 0, 0, 3889, 0, 0, 0, 25577, 0, 0, 0, 0, 0, 1739, 0, 0, 0, 0, 0, 0, 0, 12193, 0, 0, 0, 20081, 0, 18637, 0, 0, 0, 11183, 0, 15619, 0, 0, 0, 26657, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23773, 0, 0, 0, 15641, 0, 0, 0, 0, 0, 0, 0, 2659, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27689, 0, 3901, 0, 0, 0, 0, 0, 20893, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4703, 0, 29989, 0, 0, 0, 16397, 0, 0, 0, 0, 0, 18209, 0, 7951, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21431, 0, 20407, 0, 0, 0, 26993, 0, 21649, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24971, 0, 1357, 0, 0, 0, 25583, 0, 14899, 0, 0, 0, 23447, 0, 0, 0, 0, 0, 20819, 0, 0, 0, 0, 0, 0, 0, 19303, 0, 0, 0, 24821, 0, 0, 0, 0, 0, 21563, 0, 0, 0, 0, 0, 2837, 0, 0, 0, 0, 0, 13319, 0, 18301, 0, 0, 0, 0, 0, 20323, 0, 0, 0, 6371, 0, 22447, 0, 0, 0, 0, 0, 8089, 0, 0, 0, 0, 0, 0, 0, 0, 0, 479, 0, 0, 0, 0, 0, 0, 0, 17863, 0, 0, 0, 10421, 0, 15277, 0, 0, 0, 29243, 0, 0, 0, 0, 0, 1037, 0, 0, 0, 0, 0, 0, 0, 3301, 0, 0, 0, 0, 0, 25033, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25643, 0, 7069, 0, 0, 0, 6557, 0, 0, 0, 0, 0, 19709, 0, 26581, 0, 0, 0, 0, 0, 11443, 0, 0, 0, 0, 0, 1957, 0, 0, 0, 8903, 0, 26629, 0, 0, 0, 9047, 0, 0, 0, 0, 0, 24929, 0, 1861, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20381, 0, 5497, 0, 0, 0, 0, 0, 8839, 0, 0, 0, 0, 0, 0, 0, 0, 0, 419, 0, 23731, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5231, 0, 8077, 0, 0, 0, 23063, 0, 2089, 0, 0, 0, 9167, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13213, 0, 0, 0, 12011, 0, 6007, 0, 0, 0, 22823, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9539, 0, 8851, 0, 0, 0, 0, 0, 12973, 0, 0, 0, 9941, 0, 12787, 0, 0, 0, 0, 0, 12499, 0, 0, 0, 24317, 0, 0, 0, 0, 0, 16229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12521, 0, 27667, 0, 0, 0, 0, 0, 25279, 0, 0, 0, 16157, 0, 0, 0, 0, 0, 0, 0, 8971, 0, 0, 0, 0, 0, 27133, 0, 0, 0, 16061, 0, 0, 0, 0, 0, 24593, 0, 11269, 0, 0, 0, 21467, 0, 0, 0, 0, 0, 28739, 0, 11461, 0, 0, 0, 0, 0, 10393, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11003, 0, 25159, 0, 0, 0, 14717, 0, 0, 0, 0, 0, 21569, 0, 16981, 0, 0, 0, 0, 0, 6793, 0, 0, 0, 2741, 0, 7597, 0, 0, 0, 18173, 0, 19969, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13019, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25601, 0, 11647, 0, 0, 0, 15713, 0, 0, 0, 0, 0, 29747, 0, 0, 0, 0, 0, 16829, 0, 15181, 0, 0, 0, 0, 0, 14473, 0, 0, 0, 0, 0, 23227, 0, 0, 0, 16733, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24659, 0, 24601, 0, 0, 0, 0, 0, 10453, 0, 0, 0, 16661, 0, 23077, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12347, 0, 0, 0, 0, 0, 22949, 0, 0, 0, 0, 0, 0, 0, 9043, 0, 0, 0, 27641, 0, 26617, 0, 0, 0, 0, 0, 7849, 0, 0, 0, 10067, 0, 0, 0, 0, 0, 0, 0, 1621, 0, 0, 0, 0, 0, 1303, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15143, 0, 21559, 0, 0, 0, 14117, 0, 0, 0, 0, 0, 19079, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2377, 0, 0, 0, 12263, 0, 28729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3929, 0, 21391, 0, 0, 0, 0, 0, 24853, 0, 0, 0, 29411, 0, 27967, 0, 0, 0, 23843, 0, 28999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22471, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4601, 0, 937, 0, 0, 0, 593, 0, 16369, 0, 0, 0, 28697, 0, 0, 0, 0, 0, 0, 0, 13921, 0, 0, 0, 0, 0, 23503, 0, 0, 0, 23981, 0, 13567, 0, 0, 0, 1823, 0, 0, 0, 0, 0, 25097, 0, 0, 0, 0, 0, 26759, 0, 9481, 0, 0, 0, 0, 0, 14023, 0, 0, 0, 2381, 0, 0, 0, 0, 0, 0, 0, 3049, 0, 0, 0, 11087, 0, 0, 0, 0, 0, 26519, 0, 0, 0, 0, 0, 0, 0, 19333, 0, 0, 0, 23861, 0, 0, 0, 0, 0, 21473, 0, 12049, 0, 0, 0, 11327, 0, 0, 0, 0, 0, 0, 0, 8341, 0, 0, 0, 0, 0, 13903, 0, 0, 0, 12071, 0, 0, 0, 0, 0, 863, 0, 0, 0, 0, 0, 11597, 0, 0, 0, 0, 0, 29369, 0, 9571, 0, 0, 0, 0, 0, 943, 0, 0, 0, 0, 0, 12667, 0, 0, 0, 13103, 0, 26209, 0, 0, 0, 5897, 0, 0, 0, 0, 0, 2879, 0, 0, 0, 0, 0, 0, 0, 15403, 0, 0, 0, 6101, 0, 15157, 0, 0, 0, 16493, 0, 229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 839, 0, 18691, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26651, 0, 4507, 0, 0, 0, 0, 0, 15109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17249, 0, 29671, 0, 0, 0, 0, 0, 4603, 0, 0, 0, 8861, 0, 15667, 0, 0, 0, 6233, 0, 0, 0, 0, 0, 13127, 0, 0, 0, 0, 0, 17099, 0, 0, 0, 0, 0, 0, 0, 12553, 0, 0, 0, 1121, 0, 23497, 0, 0, 0, 0, 0, 1999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15809, 0, 0, 0, 0, 0, 0, 0, 23533, 0, 0, 0, 0, 0, 14437, 0, 0, 0, 20213, 0, 14149, 0, 0, 0, 20357, 0, 0, 0, 0, 0, 0, 0, 4981, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16481, 0, 0, 0, 0, 0, 15773, 0, 12739, 0, 0, 0, 15167, 0, 0, 0, 0, 0, 0, 0, 22171, 0, 0, 0, 0, 0, 5353, 0, 0, 0, 0, 0, 3007, 0, 0, 0, 5123, 0, 18859, 0, 0, 0, 16187, 0, 0, 0, 0, 0, 0, 0, 17401, 0, 0, 0, 0, 0, 12043, 0, 0, 0, 28781, 0, 0, 0, 0, 0, 23213, 0, 24169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29819, 0, 3781, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27491, 0, 11227, 0, 0, 0, 28733, 0, 3139, 0, 0, 0, 19247, 0, 0, 0, 0, 0, 27539, 0, 6361, 0, 0, 0, 0, 0, 21823, 0, 0, 0, 24611, 0, 757, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3677, 0, 0, 0, 0, 0, 4289, 0, 21451, 0, 0, 0, 0, 0, 9193, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22579, 0, 0, 0, 25367, 0, 0, 0, 0, 0, 19379, 0, 0, 0, 0, 0, 0, 0, 27943, 0, 0, 0, 22601, 0, 27037, 0, 0, 0, 5933, 0, 15409, 0, 0, 0, 1457, 0, 0, 0, 0, 0, 0, 0, 17371, 0, 0, 0, 0, 0, 23563, 0, 0, 0, 1571, 0, 0, 0, 0, 0, 0, 0, 18409, 0, 0, 0, 13697, 0, 0, 0, 0, 0, 29789, 0, 0, 0, 0, 0, 0, 0, 9553, 0, 0, 0, 0, 0, 12037, 0, 0, 0, 14153, 0, 3739, 0, 0, 0, 20387, 0, 0, 0, 0, 0, 20729, 0, 0, 0, 0, 0, 0, 0, 10573, 0, 0, 0, 4421, 0, 27127, 0, 0, 0, 19853, 0, 9259, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12179, 0, 5881, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28751, 0, 26977, 0, 0, 0, 11303, 0, 0, 0, 0, 0, 23657, 0, 0, 0, 0, 0, 17879, 0, 9931, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15161, 0, 27007, 0, 0, 0, 2243, 0, 0, 0, 0, 0, 317, 0, 0, 0, 0, 0, 0, 0, 8641, 0, 0, 0, 0, 0, 29773, 0, 0, 0, 0, 0, 19507, 0, 0, 0, 0, 0, 8299, 0, 0, 0, 12557, 0, 0, 0, 0, 0, 26099, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10631, 0, 0, 0, 0, 0, 12653, 0, 16249, 0, 0, 0, 18677, 0, 0, 0, 0, 0, 0, 0, 16321, 0, 0, 0, 0, 0, 6133, 0, 0, 0, 25511, 0, 0, 0, 0, 0, 27743, 0, 11059, 0, 0, 0, 10337, 0, 0, 0, 0, 0, 0, 0, 9991, 0, 0, 0, 0, 0, 19843, 0, 0, 0, 0, 0, 4057, 0, 0, 0, 4493, 0, 0, 0, 0, 0, 24797, 0, 0, 0, 0, 0, 3299, 0, 17821, 0, 0, 0, 0, 0, 14143, 0, 0, 0, 0, 0, 12007, 0, 0, 0, 2213, 0, 13669, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8819, 0, 26251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 821, 0, 25717, 0, 0, 0, 0, 0, 1879, 0, 0, 0, 27857, 0, 0, 0, 0, 0, 26909, 0, 29251, 0, 0, 0, 0, 0, 6073, 0, 0, 0, 14111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28877, 0, 0, 0, 0, 0, 16259, 0, 29011, 0, 0, 0, 0, 0, 17803, 0, 0, 0, 23381, 0, 23917, 0, 0, 0, 0, 0, 23209, 0, 0, 0, 23687, 0, 0, 0, 0, 0, 11819, 0, 0, 0, 0, 0, 0, 0, 18493, 0, 0, 0, 9371, 0, 25147, 0, 0, 0, 16223, 0, 8269, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10333, 0, 0, 0, 26771, 0, 0, 0, 0, 0, 14933, 0, 17779, 0, 0, 0, 887, 0, 0, 0, 0, 0, 25799, 0, 8941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4897, 0, 0, 0, 14783, 0, 1849, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28111, 0, 0, 0, 0, 0, 11623, 0, 0, 0, 25631, 0, 27757, 0, 0, 0, 1163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15539, 0, 13441, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5021, 0, 0, 0, 0, 0, 29153, 0, 28129, 0, 0, 0, 8117, 0, 0, 0, 0, 0, 18929, 0, 2791, 0, 0, 0, 0, 0, 4813, 0, 0, 0, 0, 0, 25747, 0, 0, 0, 2663, 0, 0, 0, 0, 0, 2417, 0, 0, 0, 0, 0, 1139, 0, 1081, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11411, 0, 9007, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2687, 0, 0, 0, 0, 0, 10559, 0, 0, 0, 0, 0, 0, 0, 17443, 0, 0, 0, 0, 0, 10027, 0, 0, 0, 5513, 0, 17299, 0, 0, 0, 2927, 0, 0, 0, 0, 0, 0, 0, 4141, 0, 0, 0, 0, 0, 12433, 0, 0, 0, 5771, 0, 0, 0, 0, 0, 5273, 0, 14419, 0, 0, 0, 18947, 0, 0, 0, 0, 0, 179, 0, 21331, 0, 0, 0, 0, 0, 11023, 0, 0, 0, 0, 0, 6157, 0, 0, 0, 0, 0, 14449, 0, 0, 0, 15347, 0, 0, 0, 0, 0, 1409, 0, 18451, 0, 0, 0, 0, 0, 4273, 0, 0, 0, 5891, 0, 18307, 0, 0, 0, 0, 0, 9679, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17621, 0, 0, 0, 0, 0, 28103, 0, 13219, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19769, 0, 9511, 0, 0, 0, 0, 0, 22873, 0, 0, 0, 4661, 0, 13777, 0, 0, 0, 12953, 0, 0, 0, 0, 0, 29927, 0, 0, 0, 0, 0, 16889, 0, 16201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1961, 0, 2917, 0, 0, 0, 0, 0, 5149, 0, 0, 0, 3947, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7993, 0, 0, 0, 23651, 0, 2677, 0, 0, 0, 9083, 0, 7639, 0, 0, 0, 1667, 0, 0, 0, 0, 0, 0, 0, 16741, 0, 0, 0, 0, 0, 13693, 0, 0, 0, 16901, 0, 0, 0, 0, 0, 7373, 0, 26809, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15299, 0, 17971, 0, 0, 0, 0, 0, 16693, 0, 0, 0, 0, 0, 8257, 0, 0, 0, 15203, 0, 0, 0, 0, 0, 14297, 0, 0, 0, 0, 0, 16319, 0, 27481, 0, 0, 0, 0, 0, 19183, 0, 0, 0, 10091, 0, 17257, 0, 0, 0, 19013, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13859, 0, 25411, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17947, 0, 0, 0, 10463, 0, 15319, 0, 0, 0, 29327, 0, 0, 0, 0, 0, 21029, 0, 25261, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9071, 0, 12517, 0, 0, 0, 3293, 0, 0, 0, 0, 0, 22577, 0, 0, 0, 0, 0, 7439, 0, 12421, 0, 0, 0, 0, 0, 10033, 0, 0, 0, 28631, 0, 4177, 0, 0, 0, 0, 0, 22369, 0, 0, 0, 27887, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23743, 0, 0, 0, 10841, 0, 157, 0, 0, 0, 15803, 0, 17929, 0, 0, 0, 19727, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14533, 0, 0, 0, 4091, 0, 0, 0, 0, 0, 22493, 0, 25759, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9209, 0, 24691, 0, 0, 0, 0, 0, 9343, 0, 0, 0, 0, 0, 12457, 0, 0, 0, 0, 0, 2269, 0, 0, 0, 4847, 0, 0, 0, 0, 0, 8969, 0, 19501, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14501, 0, 0, 0, 0, 0, 1583, 0, 10099, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10291, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27281, 0, 5977, 0, 0, 0, 7103, 0, 5659, 0, 0, 0, 4967, 0, 0, 0, 0, 0, 23129, 0, 24001, 0, 0, 0, 0, 0, 21613, 0, 0, 0, 13271, 0, 0, 0, 0, 0, 12323, 0, 0, 0, 0, 0, 7457, 0, 0, 0, 0, 0, 6389, 0, 0, 0, 0, 0, 0, 0, 8773, 0, 0, 0, 27161, 0, 13207, 0, 0, 0, 0, 0, 12709, 0, 0, 0, 12977, 0, 0, 0, 0, 0, 2789, 0, 0, 0, 0, 0, 0, 0, 8623, 0, 0, 0, 21971, 0, 3727, 0, 0, 0, 1523, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14851, 0, 0, 0, 0, 0, 6553, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18713, 0, 0, 0, 0, 0, 11177, 0, 0, 0, 0, 0, 10469, 0, 17341, 0, 0, 0, 0, 0, 16063, 0, 0, 0, 0, 0, 25057, 0, 0, 0, 11213, 0, 1219, 0, 0, 0, 20597, 0, 0, 0, 0, 0, 6449, 0, 20341, 0, 0, 0, 0, 0, 17293, 0, 0, 0, 0, 0, 7807, 0, 0, 0, 0, 0, 20389, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5039, 0, 26041, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9851, 0, 26557, 0, 0, 0, 13823, 0, 22879, 0, 0, 0, 27647, 0, 0, 0, 0, 0, 27329, 0, 0, 0, 0, 0, 0, 0, 17833, 0, 0, 0, 23561, 0, 26797, 0, 0, 0, 4343, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11161, 0, 0, 0, 0, 0, 24523, 0, 0, 0, 701, 0, 24337, 0, 0, 0, 0, 0, 14809, 0, 0, 0, 17387, 0, 0, 0, 0, 0, 20849, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 971, 0, 2257, 0, 0, 0, 263, 0, 18349, 0, 0, 0, 18467, 0, 0, 0, 0, 0, 0, 0, 29761, 0, 0, 0, 0, 0, 15583, 0, 0, 0, 18371, 0, 0, 0, 0, 0, 0, 0, 13579, 0, 0, 0, 9917, 0, 0, 0, 0, 0, 19499, 0, 23011, 0, 0, 0, 0, 0, 12703, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15623, 0, 13609, 0, 0, 0, 5477, 0, 0, 0, 0, 0, 10019, 0, 21601, 0, 0, 0, 0, 0, 29893, 0, 0, 0, 5051, 0, 12217, 0, 0, 0, 29723, 0, 10729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15329, 0, 16591, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14051, 0, 23197, 0, 0, 0, 6473, 0, 0, 0, 0, 0, 4337, 0, 0, 0, 0, 0, 9899, 0, 0, 0, 0, 0, 0, 0, 2923, 0, 0, 0, 0, 0, 13987, 0, 0, 0, 5183, 0, 0, 0, 0, 0, 5567, 0, 0, 0, 0, 0, 26969, 0, 29221, 0, 0, 0, 0, 0, 1213, 0, 0, 0, 28211, 0, 11527, 0, 0, 0, 0, 0, 29929, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11399, 0, 0, 0, 0, 0, 0, 0, 20593, 0, 0, 0, 2231, 0, 28927, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14687, 0, 0, 0, 0, 0, 0, 0, 22411, 0, 0, 0, 0, 0, 5923, 0, 0, 0, 17111, 0, 0, 0, 0, 0, 17453, 0, 26179, 0, 0, 0, 25667, 0, 0, 0, 0, 0, 599, 0, 0, 0, 0, 0, 0, 0, 25093, 0, 0, 0, 0, 0, 4687, 0, 0, 0, 713, 0, 10249, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8549, 0, 23701, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4001, 0, 16417, 0, 0, 0, 14603, 0, 19759, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19529, 0, 15541, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16151, 0, 21727, 0, 0, 0, 12143, 0, 4819, 0, 0, 0, 977, 0, 0, 0, 0, 0, 0, 0, 16231, 0, 0, 0, 0, 0, 5023, 0, 0, 0, 14741, 0, 25117, 0, 0, 0, 6443, 0, 0, 0, 0, 0, 18167, 0, 0, 0, 0, 0, 1349, 0, 27961, 0, 0, 0, 0, 0, 4783, 0, 0, 0, 20861, 0, 0, 0, 0, 0, 0, 0, 26149, 0, 0, 0, 13397, 0, 0, 0, 0, 0, 8039, 0, 0, 0, 0, 0, 0, 0, 7783, 0, 0, 0, 26171, 0, 19477, 0, 0, 0, 0, 0, 14359, 0, 0, 0, 29807, 0, 0, 0, 0, 0, 0, 0, 19891, 0, 0, 0, 0, 0, 16213, 0, 0, 0, 5141, 0, 0, 0, 0, 0, 5483, 0, 0, 0, 0, 0, 2357, 0, 0, 0, 0, 0, 17819, 0, 14191, 0, 0, 0, 0, 0, 24043, 0, 0, 0, 0, 0, 17287, 0, 0, 0, 24653, 0, 16969, 0, 0, 0, 17447, 0, 0, 0, 0, 0, 5189, 0, 27901, 0, 0, 0, 0, 0, 17713, 0, 0, 0, 24581, 0, 26707, 0, 0, 0, 7253, 0, 11779, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23939, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17411, 0, 25297, 0, 0, 0, 0, 0, 19729, 0, 0, 0, 13367, 0, 0, 0, 0, 0, 19559, 0, 4261, 0, 0, 0, 0, 0, 25393, 0, 0, 0, 20411, 0, 4117, 0, 0, 0, 8543, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5549, 0, 0, 0, 0, 0, 0, 0, 24103, 0, 0, 0, 5741, 0, 25807, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6569, 0, 0, 0, 0, 0, 0, 0, 28153, 0, 0, 0, 11261, 0, 7507, 0, 0, 0, 22523, 0, 18769, 0, 0, 0, 1877, 0, 0, 0, 0, 0, 0, 0, 23461, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4223, 0, 24289, 0, 0, 0, 5927, 0, 0, 0, 0, 0, 0, 0, 24481, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21487, 0, 0, 0, 25913, 0, 9619, 0, 0, 0, 4637, 0, 0, 0, 0, 0, 25769, 0, 10471, 0, 0, 0, 0, 0, 16663, 0, 0, 0, 10301, 0, 0, 0, 0, 0, 4733, 0, 12619, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6091, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18251, 0, 22777, 0, 0, 0, 3323, 0, 5449, 0, 0, 0, 12317, 0, 0, 0, 0, 0, 2129, 0, 24841, 0, 0, 0, 0, 0, 12583, 0, 0, 0, 13061, 0, 5377, 0, 0, 0, 12743, 0, 0, 0, 0, 0, 5987, 0, 0, 0, 0, 0, 15839, 0, 12211, 0, 0, 0, 0, 0, 27673, 0, 0, 0, 0, 0, 24547, 0, 0, 0, 0, 0, 24889, 0, 0, 0, 13817, 0, 0, 0, 0, 0, 10139, 0, 0, 0, 0, 0, 0, 0, 223, 0, 0, 0, 15671, 0, 0, 0, 0, 0, 10553, 0, 3859, 0, 0, 0, 22247, 0, 0, 0, 0, 0, 0, 0, 21991, 0, 0, 0, 0, 0, 16633, 0, 0, 0, 3881, 0, 0, 0, 0, 0, 0, 0, 9169, 0, 0, 0, 25247, 0, 0, 0, 0, 0, 2069, 0, 28681, 0, 0, 0, 0, 0, 11863, 0, 0, 0, 0, 0, 23587, 0, 0, 0, 4913, 0, 15289, 0, 0, 0, 25007, 0, 0, 0, 0, 0, 13799, 0, 0, 0, 0, 0, 0, 0, 29053, 0, 0, 0, 25211, 0, 17887, 0, 0, 0, 8303, 0, 13879, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14489, 0, 10501, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10271, 0, 15427, 0, 0, 0, 13613, 0, 26029, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6329, 0, 21481, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19781, 0, 29317, 0, 0, 0, 25343, 0, 0, 0, 0, 0, 4937, 0, 0, 0, 0, 0, 0, 0, 29431, 0, 0, 0, 0, 0, 4363, 0, 0, 0, 3851, 0, 12577, 0, 0, 0, 0, 0, 12919, 0, 0, 0, 24107, 0, 0, 0, 0, 0, 7619, 0, 0, 0, 0, 0, 0, 0, 15343, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1103, 0, 27799, 0, 0, 0, 9437, 0, 0, 0, 0, 0, 0, 0, 18631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 0, 0, 0, 0, 0, 18503, 0, 1819, 0, 0, 0, 28817, 0, 0, 0, 0, 0, 809, 0, 3061, 0, 0, 0, 0, 0, 24463, 0, 0, 0, 0, 0, 24847, 0, 0, 0, 16043, 0, 0, 0, 0, 0, 27107, 0, 0, 0, 0, 0, 0, 0, 20131, 0, 0, 0, 0, 0, 25693, 0, 0, 0, 0, 0, 23557, 0, 0, 0, 6833, 0, 15979, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13439, 0, 14701, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19301, 0, 307, 0, 0, 0, 17813, 0, 24979, 0, 0, 0, 137, 0, 0, 0, 0, 0, 8429, 0, 20011, 0, 0, 0, 0, 0, 24553, 0, 0, 0, 16421, 0, 14407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17327, 0, 0, 0, 0, 0, 7019, 0, 10531, 0, 0, 0, 0, 0, 20113, 0, 0, 0, 16451, 0, 0, 0, 0, 0, 0, 0, 11659, 0, 0, 0, 14447, 0, 0, 0, 0, 0, 269, 0, 0, 0, 0, 0, 0, 0, 11563, 0, 0, 0, 11681, 0, 29767, 0, 0, 0, 27773, 0, 29059, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9181, 0, 0, 0, 0, 0, 12643, 0, 0, 0, 15221, 0, 0, 0, 0, 0, 5693, 0, 29329, 0, 0, 0, 5507, 0, 0, 0, 0, 0, 18869, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25687, 0, 0, 0, 3233, 0, 6469, 0, 0, 0, 12197, 0, 0, 0, 0, 0, 0, 0, 2701, 0, 0, 0, 0, 0, 2383, 0, 0, 0, 7151, 0, 16207, 0, 0, 0, 3473, 0, 20179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3989, 0, 24991, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9641, 0, 0, 0, 0, 0, 22223, 0, 0, 0, 0, 0, 12737, 0, 0, 0, 0, 0, 9689, 0, 23581, 0, 0, 0, 0, 0, 9433, 0, 0, 0, 28811, 0, 18817, 0, 0, 0, 4973, 0, 0, 0, 0, 0, 13967, 0, 0, 0, 0, 0, 12689, 0, 19561, 0, 0, 0, 0, 0, 18853, 0, 0, 0, 0, 0, 11317, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23477, 0, 0, 0, 0, 0, 15179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28507, 0, 0, 0, 26303, 0, 8059, 0, 0, 0, 21407, 0, 0, 0, 0, 0, 0, 0, 27241, 0, 0, 0, 0, 0, 17053, 0, 0, 0, 17321, 0, 0, 0, 0, 0, 16823, 0, 2869, 0, 0, 0, 21257, 0, 0, 0, 0, 0, 0, 0, 23641, 0, 0, 0, 0, 0, 22573, 0, 0, 0, 0, 0, 17707, 0, 0, 0, 0, 0, 16759, 0, 0, 0, 8417, 0, 0, 0, 0, 0, 6029, 0, 6901, 0, 0, 0, 0, 0, 25063, 0, 0, 0, 24371, 0, 22927, 0, 0, 0, 24053, 0, 2749, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19739, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19931, 0, 28447, 0, 0, 0, 0, 0, 15529, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10529, 0, 21061, 0, 0, 0, 0, 0, 25183, 0, 0, 0, 27761, 0, 0, 0, 0, 0, 17573, 0, 0, 0, 0, 0, 20687, 0, 0, 0, 0, 0, 5339, 0, 20821, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4271, 0, 7537, 0, 0, 0, 0, 0, 25939, 0, 0, 0, 15497, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10303, 0, 0, 0, 12101, 0, 14227, 0, 0, 0, 29873, 0, 19189, 0, 0, 0, 6287, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2143, 0, 0, 0, 7661, 0, 0, 0, 0, 0, 25853, 0, 1399, 0, 0, 0, 19997, 0, 0, 0, 0, 0, 17609, 0, 22591, 0, 0, 0, 0, 0, 7453, 0, 0, 0, 0, 0, 26737, 0, 0, 0, 17513, 0, 20959, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4769, 0, 9001, 0, 0, 0, 0, 0, 703, 0, 0, 0, 14711, 0, 19567, 0, 0, 0, 12083, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4619, 0, 16171, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11017, 0, 0, 0, 12773, 0, 19939, 0, 0, 0, 10847, 0, 0, 0, 0, 0, 2549, 0, 13711, 0, 0, 0, 0, 0, 15733, 0, 0, 0, 0, 0, 14827, 0, 0, 0, 21773, 0, 0, 0, 0, 0, 13337, 0, 0, 0, 0, 0, 12059, 0, 14731, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3221, 0, 22657, 0, 0, 0, 0, 0, 13129, 0, 0, 0, 16337, 0, 0, 0, 0, 0, 13289, 0, 0, 0, 0, 0, 0, 0, 28363, 0, 0, 0, 22391, 0, 20947, 0, 0, 0, 27353, 0, 6379, 0, 0, 0, 22037, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26083, 0, 0, 0, 24881, 0, 0, 0, 0, 0, 27113, 0, 28069, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13829, 0, 13141, 0, 0, 0, 0, 0, 103, 0, 0, 0, 0, 0, 17077, 0, 0, 0, 16253, 0, 25369, 0, 0, 0, 7157, 0, 0, 0, 0, 0, 20519, 0, 10261, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16811, 0, 1927, 0, 0, 0, 0, 0, 12409, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21841, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20351, 0, 0, 0, 0, 0, 11723, 0, 24139, 0, 0, 0, 25757, 0, 0, 0, 0, 0, 11579, 0, 28621, 0, 0, 0, 0, 0, 14683, 0, 0, 0, 15581, 0, 0, 0, 0, 0, 23873, 0, 0, 0, 0, 0, 19007, 0, 0, 0, 0, 0, 8699, 0, 29851, 0, 0, 0, 0, 0, 11083, 0, 0, 0, 15611, 0, 24757, 0, 0, 0, 0, 0, 24259, 0, 0, 0, 17597, 0, 0, 0, 0, 0, 25889, 0, 0, 0, 0, 0, 0, 0, 27103, 0, 0, 0, 12731, 0, 24517, 0, 0, 0, 20003, 0, 0, 0, 0, 0, 12587, 0, 0, 0, 0, 0, 0, 0, 19471, 0, 0, 0, 0, 0, 27343, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21023, 0, 18619, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28949, 0, 28891, 0, 0, 0, 0, 0, 27613, 0, 0, 0, 0, 0, 27367, 0, 0, 0, 4283, 0, 0, 0, 0, 0, 25217, 0, 0, 0, 0, 0, 27239, 0, 11101, 0, 0, 0, 0, 0, 21913, 0, 0, 0, 1901, 0, 877, 0, 0, 0, 0, 0, 25009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16589, 0, 14491, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28867, 0, 0, 0, 2273, 0, 4399, 0, 0, 0, 18407, 0, 0, 0, 0, 0, 1919, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28181, 0, 15247, 0, 0, 0, 25133, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21089, 0, 4231, 0, 0, 0, 0, 0, 29143, 0, 0, 0, 12251, 0, 15097, 0, 0, 0, 0, 0, 3259, 0, 0, 0, 19697, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21761, 0, 13807, 0, 0, 0, 4883, 0, 20659, 0, 0, 0, 11537, 0, 0, 0, 0, 0, 0, 0, 18211, 0, 0, 0, 0, 0, 6343, 0, 0, 0, 6821, 0, 0, 0, 0, 0, 6113, 0, 6649, 0, 0, 0, 12227, 0, 0, 0, 0, 0, 1019, 0, 13771, 0, 0, 0, 0, 0, 1153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15919, 0, 0, 0, 23957, 0, 0, 0, 0, 0, 779, 0, 3121, 0, 0, 0, 0, 0, 2173, 0, 0, 0, 28151, 0, 0, 0, 0, 0, 4313, 0, 29209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3779, 0, 21211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16361, 0, 27817, 0, 0, 0, 18023, 0, 0, 0, 0, 0, 15887, 0, 0, 0, 0, 0, 12209, 0, 26731, 0, 0, 0, 0, 0, 5233, 0, 0, 0, 0, 0, 25537, 0, 0, 0, 25973, 0, 0, 0, 0, 0, 10187, 0, 0, 0, 0, 0, 20039, 0, 0, 0, 0, 0, 0, 0, 19693, 0, 0, 0, 18971, 0, 2287, 0, 0, 0, 0, 0, 4519, 0, 0, 0, 23897, 0, 0, 0, 0, 0, 13709, 0, 0, 0, 0, 0, 0, 0, 11353, 0, 0, 0, 13781, 0, 17377, 0, 0, 0, 0, 0, 19399, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3931, 0, 0, 0, 0, 0, 17473, 0, 0, 0, 21731, 0, 0, 0, 0, 0, 10523, 0, 0, 0, 0, 0, 257, 0, 0, 0, 0, 0, 21389, 0, 0, 0, 0, 0, 0, 0, 29713, 0, 0, 0, 0, 0, 27787, 0, 0, 0, 3023, 0, 14869, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20099, 0, 12151, 0, 0, 0, 0, 0, 6373, 0, 0, 0, 0, 0, 18727, 0, 0, 0, 3053, 0, 1279, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24149, 0, 17851, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20771, 0, 10177, 0, 0, 0, 2903, 0, 25609, 0, 0, 0, 19457, 0, 0, 0, 0, 0, 0, 0, 9301, 0, 0, 0, 0, 0, 9643, 0, 0, 0, 26291, 0, 15877, 0, 0, 0, 17993, 0, 0, 0, 0, 0, 20477, 0, 0, 0, 0, 0, 0, 0, 241, 0, 0, 0, 0, 0, 16333, 0, 0, 0, 11621, 0, 0, 0, 0, 0, 0, 0, 28459, 0, 0, 0, 6467, 0, 0, 0, 0, 0, 12659, 0, 0, 0, 0, 0, 0, 0, 28573, 0, 0, 0, 14621, 0, 24097, 0, 0, 0, 2993, 0, 7429, 0, 0, 0, 2087, 0, 0, 0, 0, 0, 0, 0, 10651, 0, 0, 0, 0, 0, 4663, 0, 0, 0, 7451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20837, 0, 0, 0, 0, 0, 8579, 0, 25741, 0, 0, 0, 0, 0, 26353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29273, 0, 5419, 0, 0, 0, 8207, 0, 0, 0, 0, 0, 23669, 0, 2491, 0, 0, 0, 0, 0, 10783, 0, 0, 0, 26891, 0, 1297, 0, 0, 0, 18803, 0, 2539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26249, 0, 211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5861, 0, 6817, 0, 0, 0, 0, 0, 1249, 0, 0, 0, 17987, 0, 0, 0, 0, 0, 12629, 0, 0, 0, 0, 0, 0, 0, 13843, 0, 0, 0, 11171, 0, 24907, 0, 0, 0, 27023, 0, 0, 0, 0, 0, 24677, 0, 0, 0, 0, 0, 7859, 0, 0, 0, 0, 0, 0, 0, 14863, 0, 0, 0, 17291, 0, 14257, 0, 0, 0, 0, 0, 13549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25049, 0, 0, 0, 0, 0, 0, 0, 9673, 0, 0, 0, 15881, 0, 9817, 0, 0, 0, 15593, 0, 0, 0, 0, 0, 6497, 0, 0, 0, 0, 0, 0, 0, 14221, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28031, 0, 0, 0, 0, 0, 6533, 0, 28909, 0, 0, 0, 17477, 0, 0, 0, 0, 0, 0, 0, 12931, 0, 0, 0, 0, 0, 16903, 0, 0, 0, 0, 0, 23797, 0, 0, 0, 14363, 0, 21169, 0, 0, 0, 25427, 0, 0, 0, 0, 0, 359, 0, 12781, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14921, 0, 0, 0, 0, 0, 25523, 0, 3379, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11339, 0, 29191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29801, 0, 13537, 0, 0, 0, 14873, 0, 23929, 0, 0, 0, 14627, 0, 0, 0, 0, 0, 0, 0, 27151, 0, 0, 0, 0, 0, 24133, 0, 0, 0, 3821, 0, 16927, 0, 0, 0, 17363, 0, 0, 0, 0, 0, 29087, 0, 0, 0, 0, 0, 20459, 0, 661, 0, 0, 0, 0, 0, 18433, 0, 0, 0, 0, 0, 29167, 0, 0, 0, 0, 0, 17959, 0, 0, 0, 16127, 0, 0, 0, 0, 0, 21689, 0, 0, 0, 0, 0, 0, 0, 18703, 0, 0, 0, 17981, 0, 8557, 0, 0, 0, 0, 0, 6169, 0, 0, 0, 10697, 0, 0, 0, 0, 0, 0, 0, 3511, 0, 0, 0, 0, 0, 18943, 0, 0, 0, 26981, 0, 0, 0, 0, 0, 0, 0, 27649, 0, 0, 0, 16007, 0, 0, 0, 0, 0, 20549, 0, 3271, 0, 0, 0, 0, 0, 4933, 0, 0, 0, 0, 0, 28207, 0, 0, 0, 16463, 0, 6049, 0, 0, 0, 6527, 0, 0, 0, 0, 0, 16109, 0, 0, 0, 0, 0, 0, 0, 1333, 0, 0, 0, 13661, 0, 29437, 0, 0, 0, 29093, 0, 25429, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7559, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1031, 0, 6187, 0, 0, 0, 2063, 0, 619, 0, 0, 0, 5177, 0, 0, 0, 0, 0, 8639, 0, 26101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1301, 0, 17767, 0, 0, 0, 27653, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10951, 0, 0, 0, 0, 0, 15913, 0, 0, 0, 8471, 0, 14887, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28727, 0, 0, 0, 0, 0, 28409, 0, 0, 0, 0, 0, 0, 0, 19963, 0, 0, 0, 22181, 0, 0, 0, 0, 0, 3413, 0, 2389, 0, 0, 0, 20987, 0, 0, 0, 0, 0, 0, 0, 7081, 0, 0, 0, 0, 0, 17683, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6953, 0, 13369, 0, 0, 0, 19577, 0, 0, 0, 0, 0, 5429, 0, 5371, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13297, 0, 0, 0, 6803, 0, 0, 0, 0, 0, 15557, 0, 0, 0, 0, 0, 14849, 0, 13201, 0, 0, 0, 0, 0, 283, 0, 0, 0, 0, 0, 14317, 0, 0, 0, 18383, 0, 4429, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17011, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10061, 0, 11857, 0, 0, 0, 22433, 0, 27289, 0, 0, 0, 23237, 0, 0, 0, 0, 0, 13049, 0, 8461, 0, 0, 0, 0, 0, 15313, 0, 0, 0, 4871, 0, 19027, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19637, 0, 0, 0, 0, 0, 18569, 0, 1291, 0, 0, 0, 0, 0, 8563, 0, 0, 0, 18761, 0, 5437, 0, 0, 0, 0, 0, 13969, 0, 0, 0, 2897, 0, 0, 0, 0, 0, 21059, 0, 0, 0, 0, 0, 0, 0, 13873, 0, 0, 0, 4751, 0, 0, 0, 0, 0, 2363, 0, 17509, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13801, 0, 0, 0, 0, 0, 5713, 0, 0, 0, 17531, 0, 0, 0, 0, 0, 17243, 0, 20089, 0, 0, 0, 17057, 0, 0, 0, 0, 0, 21179, 0, 20491, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7207, 0, 0, 0, 24023, 0, 18019, 0, 0, 0, 16817, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20863, 0, 0, 0, 27941, 0, 6967, 0, 0, 0, 21953, 0, 24799, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6299, 0, 29611, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21191, 0, 0, 0, 0, 0, 24533, 0, 9649, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28169, 0, 5101, 0, 0, 0, 0, 0, 20983, 0, 0, 0, 3401, 0, 21127, 0, 0, 0, 28073, 0, 0, 0, 0, 0, 18587, 0, 0, 0, 0, 0, 3449, 0, 10321, 0, 0, 0, 0, 0, 23473, 0, 0, 0, 22961, 0, 4387, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4997, 0, 0, 0, 0, 0, 26729, 0, 0, 0, 0, 0, 0, 0, 28993, 0, 0, 0, 0, 0, 787, 0, 0, 0, 14753, 0, 19609, 0, 0, 0, 12167, 0, 0, 0, 0, 0, 0, 0, 29551, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21941, 0, 0, 0, 0, 0, 7583, 0, 23659, 0, 0, 0, 9707, 0, 0, 0, 0, 0, 11729, 0, 16711, 0, 0, 0, 0, 0, 27193, 0, 0, 0, 0, 0, 8467, 0, 0, 0, 0, 0, 5209, 0, 0, 0, 10727, 0, 0, 0, 0, 0, 0, 0, 9211, 0, 0, 0, 0, 0, 6583, 0, 0, 0, 15131, 0, 4447, 0, 0, 0, 28673, 0, 5059, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24359, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8381, 0, 3037, 0, 0, 0, 9623, 0, 8599, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22079, 0, 11821, 0, 0, 0, 0, 0, 13633, 0, 0, 0, 41, 0, 25327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9137, 0, 0, 0, 0, 0, 26129, 0, 2341, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27371, 0, 0, 0, 0, 0, 0, 0, 14389, 0, 0, 0, 6257, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3373, 0, 0, 0, 14411, 0, 18847, 0, 0, 0, 11393, 0, 9949, 0, 0, 0, 17837, 0, 0, 0, 0, 0, 0, 0, 28291, 0, 0, 0, 0, 0, 4453, 0, 0, 0, 26141, 0, 0, 0, 0, 0, 16613, 0, 12949, 0, 0, 0, 24617, 0, 0, 0, 0, 0, 10679, 0, 0, 0, 0, 0, 0, 0, 25933, 0, 0, 0, 0, 0, 17497, 0, 0, 0, 5963, 0, 25579, 0, 0, 0, 25847, 0, 0, 0, 0, 0, 7079, 0, 13621, 0, 0, 0, 0, 0, 21493, 0, 0, 0, 26261, 0, 8017, 0, 0, 0, 14393, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23099, 0, 27721, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13327, 0, 0, 0, 5843, 0, 0, 0, 0, 0, 15467, 0, 0, 0, 0, 0, 23339, 0, 4471, 0, 0, 0, 0, 0, 20353, 0, 0, 0, 20621, 0, 7897, 0, 0, 0, 24083, 0, 0, 0, 0, 0, 24887, 0, 0, 0, 0, 0, 23609, 0, 3181, 0, 0, 0, 0, 0, 21583, 0, 0, 0, 0, 0, 24967, 0, 0, 0, 0, 0, 24679, 0, 0, 0, 7097, 0, 0, 0, 0, 0, 17909, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27011, 0, 9397, 0, 0, 0, 18113, 0, 27169, 0, 0, 0, 10487, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 673, 0, 0, 0, 6401, 0, 0, 0, 0, 0, 8633, 0, 16519, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15451, 0, 0, 0, 0, 0, 11653, 0, 0, 0, 0, 0, 28627, 0, 0, 0, 20873, 0, 27679, 0, 0, 0, 227, 0, 0, 0, 0, 0, 25139, 0, 28741, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5261, 0, 6547, 0, 0, 0, 13133, 0, 5479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12601, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22661, 0, 17527, 0, 0, 0, 0, 0, 26449, 0, 0, 0, 14207, 0, 0, 0, 0, 0, 2339, 0, 10141, 0, 0, 0, 0, 0, 16993, 0, 0, 0, 8651, 0, 0, 0, 0, 0, 28493, 0, 0, 0, 0, 0, 9767, 0, 0, 0, 0, 0, 27179, 0, 4441, 0, 0, 0, 0, 0, 4153, 0, 0, 0, 17921, 0, 29377, 0, 0, 0, 0, 0, 15019, 0, 0, 0, 29147, 0, 0, 0, 0, 0, 28199, 0, 0, 0, 0, 0, 0, 0, 29413, 0, 0, 0, 1181, 0, 6037, 0, 0, 0, 10763, 0, 0, 0, 0, 0, 17207, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15793, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9473, 0, 23239, 0, 0, 0, 22727, 0, 0, 0, 0, 0, 1229, 0, 3481, 0, 0, 0, 0, 0, 18373, 0, 0, 0, 0, 0, 15817, 0, 0, 0, 6593, 0, 12769, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15689, 0, 22651, 0, 0, 0, 0, 0, 3433, 0, 0, 0, 6521, 0, 3187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7349, 0, 5251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21401, 0, 21937, 0, 0, 0, 4583, 0, 9019, 0, 0, 0, 29957, 0, 0, 0, 0, 0, 13469, 0, 0, 0, 0, 0, 0, 0, 29383, 0, 0, 0, 0, 0, 17557, 0, 0, 0, 13583, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25709, 0, 6541, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16871, 0, 3547, 0, 0, 0, 0, 0, 24049, 0, 0, 0, 8147, 0, 0, 0, 0, 0, 2369, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3281, 0, 4567, 0, 0, 0, 16433, 0, 9109, 0, 0, 0, 13847, 0, 0, 0, 0, 0, 0, 0, 20521, 0, 0, 0, 0, 0, 17893, 0, 0, 0, 27611, 0, 0, 0, 0, 0, 10733, 0, 8959, 0, 0, 0, 5297, 0, 0, 0, 0, 0, 5639, 0, 2221, 0, 0, 0, 0, 0, 21943, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27173, 0, 8989, 0, 0, 0, 26267, 0, 0, 0, 0, 0, 12329, 0, 23911, 0, 0, 0, 0, 0, 20653, 0, 0, 0, 431, 0, 23767, 0, 0, 0, 0, 0, 1489, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24569, 0, 2731, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9431, 0, 0, 0, 0, 0, 22643, 0, 0, 0, 0, 0, 6647, 0, 0, 0, 0, 0, 659, 0, 1321, 0, 0, 0, 0, 0, 28333, 0, 0, 0, 0, 0, 127, 0, 0, 0, 7493, 0, 0, 0, 0, 0, 21737, 0, 0, 0, 0, 0, 22349, 0, 10741, 0, 0, 0, 0, 0, 22003, 0, 0, 0, 7421, 0, 13837, 0, 0, 0, 0, 0, 16069, 0, 0, 0, 28517, 0, 0, 0, 0, 0, 6779, 0, 0, 0, 0, 0, 0, 0, 29833, 0, 0, 0, 4541, 0, 8137, 0, 0, 0, 0, 0, 24019, 0, 0, 0, 26237, 0, 0, 0, 0, 0, 0, 0, 8551, 0, 0, 0, 0, 0, 8233, 0, 0, 0, 3251, 0, 0, 0, 0, 0, 12833, 0, 7699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9839, 0, 0, 0, 0, 0, 0, 0, 11233, 0, 0, 0, 0, 0, 67, 0, 0, 0, 26123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10859, 0, 2911, 0, 0, 0, 0, 0, 10993, 0, 0, 0, 15551, 0, 11797, 0, 0, 0, 5363, 0, 5899, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5669, 0, 6301, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12487, 0, 0, 0, 21383, 0, 7129, 0, 0, 0, 10217, 0, 0, 0, 0, 0, 0, 0, 11611, 0, 0, 0, 0, 0, 0, 0, 0, 0, 881, 0, 4327, 0, 0, 0, 8753, 0, 0, 0, 0, 0, 8927, 0, 0, 0, 0, 0, 12899, 0, 23341, 0, 0, 0, 0, 0, 20953, 0, 0, 0, 23171, 0, 0, 0, 0, 0, 0, 0, 16909, 0, 0, 0, 8777, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10093, 0, 0, 0, 5381, 0, 5617, 0, 0, 0, 7613, 0, 9739, 0, 0, 0, 25187, 0, 0, 0, 0, 0, 0, 0, 29131, 0, 0, 0, 0, 0, 25453, 0, 0, 0, 25931, 0, 0, 0, 0, 0, 17033, 0, 0, 0, 0, 0, 23147, 0, 0, 0, 0, 0, 20129, 0, 16501, 0, 0, 0, 0, 0, 14803, 0, 0, 0, 0, 0, 28837, 0, 0, 0, 0, 0, 7729, 0, 0, 0, 26687, 0, 0, 0, 0, 0, 14429, 0, 14041, 0, 0, 0, 0, 0, 13093, 0, 0, 0, 19961, 0, 0, 0, 0, 0, 23423, 0, 21019, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14699, 0, 4831, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8171, 0, 11437, 0, 0, 0, 0, 0, 22039, 0, 0, 0, 29537, 0, 0, 0, 0, 0, 14939, 0, 15811, 0, 0, 0, 0, 0, 16153, 0, 0, 0, 29651, 0, 6427, 0, 0, 0, 17783, 0, 0, 0, 0, 0, 29297, 0, 0, 0, 0, 0, 929, 0, 0, 0, 0, 0, 0, 0, 3313, 0, 0, 0, 8051, 0, 5017, 0, 0, 0, 0, 0, 18169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27359, 0, 0, 0, 0, 0, 0, 0, 433, 0, 0, 0, 27431, 0, 28297, 0, 0, 0, 17903, 0, 289, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2621, 0, 0, 0, 0, 0, 29633, 0, 0, 0, 0, 0, 22097, 0, 0, 0, 0, 0, 0, 0, 3691, 0, 0, 0, 0, 0, 21523, 0, 0, 0, 0, 0, 16867, 0, 0, 0, 16673, 0, 25789, 0, 0, 0, 6947, 0, 0, 0, 0, 0, 11909, 0, 1231, 0, 0, 0, 0, 0, 28213, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13973, 0, 14929, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15959, 0, 1471, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4391, 0, 1987, 0, 0, 0, 5633, 0, 14689, 0, 0, 0, 3077, 0, 0, 0, 0, 0, 13679, 0, 20221, 0, 0, 0, 0, 0, 28753, 0, 0, 0, 15371, 0, 7687, 0, 0, 0, 28913, 0, 0, 0, 0, 0, 1367, 0, 0, 0, 0, 0, 0, 0, 2971, 0, 0, 0, 0, 0, 29983, 0, 0, 0, 0, 0, 10687, 0, 0, 0, 0, 0, 20269, 0, 0, 0, 9197, 0, 0, 0, 0, 0, 26309, 0, 0, 0, 0, 0, 0, 0, 9463, 0, 0, 0, 6431, 0, 13177, 0, 0, 0, 22103, 0, 29269, 0, 0, 0, 13007, 0, 0, 0, 0, 0, 0, 0, 24301, 0, 0, 0, 0, 0, 7393, 0, 0, 0, 29291, 0, 0, 0, 0, 0, 0, 0, 29959, 0, 0, 0, 4457, 0, 0, 0, 0, 0, 11309, 0, 14821, 0, 0, 0, 0, 0, 7243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21083, 0, 24529, 0, 0, 0, 27317, 0, 0, 0, 0, 0, 4559, 0, 0, 0, 0, 0, 0, 0, 24433, 0, 0, 0, 15971, 0, 4027, 0, 0, 0, 10613, 0, 16189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9869, 0, 22051, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19511, 0, 17737, 0, 0, 0, 22853, 0, 12169, 0, 0, 0, 9797, 0, 0, 0, 0, 0, 1709, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22091, 0, 8527, 0, 0, 0, 16103, 0, 0, 0, 0, 0, 16487, 0, 0, 0, 0, 0, 0, 0, 15571, 0, 0, 0, 0, 0, 6673, 0, 0, 0, 20021, 0, 3337, 0, 0, 0, 0, 0, 24469, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16859, 0, 0, 0, 0, 0, 0, 0, 1483, 0, 0, 0, 26801, 0, 0, 0, 0, 0, 26513, 0, 0, 0, 0, 0, 25607, 0, 0, 0, 0, 0, 0, 0, 27871, 0, 0, 0, 0, 0, 22303, 0, 0, 0, 11651, 0, 0, 0, 0, 0, 9263, 0, 17989, 0, 0, 0, 1097, 0, 0, 0, 0, 0, 16979, 0, 23851, 0, 0, 0, 0, 0, 5983, 0, 0, 0, 0, 0, 15607, 0, 0, 0, 25283, 0, 0, 0, 0, 0, 6317, 0, 0, 0, 0, 0, 19469, 0, 15511, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2767, 0, 0, 0, 9143, 0, 25219, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24989, 0, 10081, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21611, 0, 2617, 0, 0, 0, 3953, 0, 15739, 0, 0, 0, 25547, 0, 0, 0, 0, 0, 0, 0, 10771, 0, 0, 0, 0, 0, 26863, 0, 0, 0, 25661, 0, 547, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12707, 0, 0, 0, 0, 0, 23189, 0, 19771, 0, 0, 0, 0, 0, 29353, 0, 0, 0, 7211, 0, 10057, 0, 0, 0, 0, 0, 7039, 0, 0, 0, 5207, 0, 0, 0, 0, 0, 2579, 0, 0, 0, 0, 0, 0, 0, 2323, 0, 0, 0, 7061, 0, 11287, 0, 0, 0, 0, 0, 19819, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25351, 0, 0, 0, 0, 0, 8023, 0, 0, 0, 10601, 0, 0, 0, 0, 0, 21863, 0, 8539, 0, 0, 0, 7817, 0, 0, 0, 0, 0, 9629, 0, 25111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11827, 0, 0, 0, 5543, 0, 8779, 0, 0, 0, 28367, 0, 0, 0, 0, 0, 0, 0, 14251, 0, 0, 0, 0, 0, 23173, 0, 0, 0, 16391, 0, 18517, 0, 0, 0, 12713, 0, 6319, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29399, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11951, 0, 0, 0, 0, 0, 12983, 0, 14269, 0, 0, 0, 24287, 0, 0, 0, 0, 0, 449, 0, 9721, 0, 0, 0, 0, 0, 11743, 0, 0, 0, 14951, 0, 9577, 0, 0, 0, 353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21929, 0, 21871, 0, 0, 0, 0, 0, 4993, 0, 0, 0, 27581, 0, 6697, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9617, 0, 0, 0, 0, 0, 17489, 0, 0, 0, 0, 0, 0, 0, 3583, 0, 0, 0, 0, 0, 23887, 0, 0, 0, 17063, 0, 24229, 0, 0, 0, 23717, 0, 0, 0, 0, 0, 0, 0, 18001, 0, 0, 0, 0, 0, 28603, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26063, 0, 5179, 0, 0, 0, 467, 0, 0, 0, 0, 0, 16349, 0, 19021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26947, 0, 0, 0, 0, 0, 25999, 0, 0, 0, 29207, 0, 0, 0, 0, 0, 17579, 0, 2281, 0, 0, 0, 0, 0, 11203, 0, 0, 0, 26681, 0, 25237, 0, 0, 0, 10193, 0, 23539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29171, 0, 14587, 0, 0, 0, 14243, 0, 10909, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26699, 0, 271, 0, 0, 0, 0, 0, 4393, 0, 0, 0, 18521, 0, 29947, 0, 0, 0, 29123, 0, 0, 0, 0, 0, 11447, 0, 0, 0, 0, 0, 7649, 0, 23131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29681, 0, 19087, 0, 0, 0, 0, 0, 16699, 0, 0, 0, 24737, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5683, 0, 0, 0, 7481, 0, 0, 0, 0, 0, 16013, 0, 28429, 0, 0, 0, 8597, 0, 0, 0, 0, 0, 0, 0, 2881, 0, 0, 0, 0, 0, 27553, 0, 0, 0, 28451, 0, 0, 0, 0, 0, 28163, 0, 3709, 0, 0, 0, 6137, 0, 0, 0, 0, 0, 12989, 0, 4111, 0, 0, 0, 0, 0, 28243, 0, 0, 0, 0, 0, 29047, 0, 0, 0, 26753, 0, 7099, 0, 0, 0, 437, 0, 0, 0, 0, 0, 149, 0, 0, 0, 0, 0, 0, 0, 9943, 0, 0, 0, 17021, 0, 28807, 0, 0, 0, 2843, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25409, 0, 2311, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1777, 0, 0, 0, 8153, 0, 1459, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11789, 0, 16021, 0, 0, 0, 0, 0, 1873, 0, 0, 0, 25241, 0, 10207, 0, 0, 0, 17153, 0, 0, 0, 0, 0, 29507, 0, 0, 0, 0, 0, 14369, 0, 23971, 0, 0, 0, 0, 0, 26203, 0, 0, 0, 14771, 0, 18037, 0, 0, 0, 0, 0, 29299, 0, 0, 0, 18647, 0, 0, 0, 0, 0, 29459, 0, 0, 0, 0, 0, 0, 0, 9883, 0, 0, 0, 0, 0, 11707, 0, 0, 0, 6563, 0, 8689, 0, 0, 0, 1247, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11021, 0, 0, 0, 0, 0, 29423, 0, 7279, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25379, 0, 8521, 0, 0, 0, 0, 0, 16273, 0, 0, 0, 0, 0, 19387, 0, 0, 0, 2393, 0, 16129, 0, 0, 0, 2537, 0, 0, 0, 0, 0, 0, 0, 1021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26051, 0, 18097, 0, 0, 0, 17753, 0, 7789, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1051, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11111, 0, 22147, 0, 0, 0, 23273, 0, 19519, 0, 0, 0, 16517, 0, 0, 0, 0, 0, 13889, 0, 901, 0, 0, 0, 0, 0, 5443, 0, 0, 0, 10961, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28247, 0, 0, 0, 0, 0, 17939, 0, 15991, 0, 0, 0, 0, 0, 6463, 0, 0, 0, 10991, 0, 0, 0, 0, 0, 0, 0, 3469, 0, 0, 0, 19907, 0, 0, 0, 0, 0, 16649, 0, 0, 0, 0, 0, 0, 0, 22483, 0, 0, 0, 3491, 0, 10657, 0, 0, 0, 22313, 0, 0, 0, 0, 0, 28757, 0, 0, 0, 0, 0, 0, 0, 991, 0, 0, 0, 0, 0, 18103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 233, 0, 4759, 0, 0, 0, 27347, 0, 0, 0, 0, 0, 24329, 0, 0, 0, 0, 0, 0, 0, 6823, 0, 0, 0, 0, 0, 6577, 0, 0, 0, 25073, 0, 17389, 0, 0, 0, 6737, 0, 0, 0, 0, 0, 17999, 0, 27271, 0, 0, 0, 0, 0, 24223, 0, 0, 0, 18071, 0, 21667, 0, 0, 0, 0, 0, 6529, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25829, 0, 16801, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26021, 0, 24247, 0, 0, 0, 27683, 0, 0, 0, 0, 0, 4547, 0, 0, 0, 0, 0, 4229, 0, 0, 0, 0, 0, 0, 0, 3973, 0, 0, 0, 9701, 0, 10627, 0, 0, 0, 15893, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7229, 0, 25021, 0, 0, 0, 0, 0, 10663, 0, 0, 0, 0, 0, 5857, 0, 0, 0, 0, 0, 5569, 0, 0, 0, 28937, 0, 0, 0, 0, 0, 6989, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7901, 0, 23047, 0, 0, 0, 7193, 0, 29899, 0, 0, 0, 2297, 0, 0, 0, 0, 0, 0, 0, 13591, 0, 0, 0, 0, 0, 22513, 0, 0, 0, 9131, 0, 0, 0, 0, 0, 22283, 0, 27439, 0, 0, 0, 7607, 0, 0, 0, 0, 0, 0, 0, 4531, 0, 0, 0, 0, 0, 3463, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1763, 0, 11299, 0, 0, 0, 19337, 0, 0, 0, 0, 0, 16949, 0, 12361, 0, 0, 0, 0, 0, 11413, 0, 0, 0, 18911, 0, 28387, 0, 0, 0, 15863, 0, 24589, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6089, 0, 23521, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11741, 0, 9337, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25127, 0, 0, 0, 0, 0, 21449, 0, 12871, 0, 0, 0, 0, 0, 613, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12113, 0, 0, 0, 0, 0, 12497, 0, 0, 0, 0, 0, 10799, 0, 15361, 0, 0, 0, 0, 0, 15073, 0, 0, 0, 9731, 0, 18457, 0, 0, 0, 0, 0, 6829, 0, 0, 0, 10037, 0, 0, 0, 0, 0, 9089, 0, 0, 0, 0, 0, 0, 0, 2113, 0, 0, 0, 23021, 0, 19687, 0, 0, 0, 0, 0, 5539, 0, 0, 0, 827, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26713, 0, 0, 0, 24041, 0, 0, 0, 0, 0, 1283, 0, 12319, 0, 0, 0, 11807, 0, 0, 0, 0, 0, 12149, 0, 0, 0, 0, 0, 0, 0, 1993, 0, 0, 0, 0, 0, 18547, 0, 0, 0, 28433, 0, 26419, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29339, 0, 14461, 0, 0, 0, 0, 0, 22543, 0, 0, 0, 20171, 0, 14107, 0, 0, 0, 28463, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26459, 0, 27091, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2291, 0, 5557, 0, 0, 0, 23693, 0, 11749, 0, 0, 0, 21767, 0, 0, 0, 0, 0, 0, 0, 61, 0, 0, 0, 0, 0, 21193, 0, 0, 0, 0, 0, 6637, 0, 0, 0, 27233, 0, 0, 0, 0, 0, 29717, 0, 0, 0, 0, 0, 17519, 0, 25651, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27791, 0, 0, 0, 0, 0, 0, 0, 7669, 0, 0, 0, 27257, 0, 0, 0, 0, 0, 24209, 0, 0, 0, 0, 0, 0, 0, 14713, 0, 0, 0, 16931, 0, 26407, 0, 0, 0, 19163, 0, 28219, 0, 0, 0, 27497, 0, 0, 0, 0, 0, 0, 0, 1411, 0, 0, 0, 0, 0, 6973, 0, 0, 0, 16691, 0, 0, 0, 0, 0, 21653, 0, 0, 0, 0, 0, 16217, 0, 0, 0, 0, 0, 24749, 0, 4951, 0, 0, 0, 0, 0, 5563, 0, 0, 0, 0, 0, 3427, 0, 0, 0, 10793, 0, 799, 0, 0, 0, 28997, 0, 0, 0, 0, 0, 25979, 0, 4801, 0, 0, 0, 0, 0, 1543, 0, 0, 0, 22271, 0, 12847, 0, 0, 0, 0, 0, 23329, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5459, 0, 16381, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1241, 0, 0, 0, 0, 0, 0, 0, 10489, 0, 0, 0, 20297, 0, 0, 0, 0, 0, 3389, 0, 20431, 0, 0, 0, 0, 0, 9223, 0, 0, 0, 1931, 0, 11047, 0, 0, 0, 29333, 0, 0, 0, 0, 0, 10817, 0, 0, 0, 0, 0, 3239, 0, 0, 0, 0, 0, 0, 0, 5623, 0, 0, 0, 26531, 0, 16567, 0, 0, 0, 0, 0, 29719, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20429, 0, 0, 0, 0, 0, 0, 0, 18913, 0, 0, 0, 18191, 0, 19057, 0, 0, 0, 6353, 0, 4909, 0, 0, 0, 18047, 0, 0, 0, 0, 0, 0, 0, 361, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14171, 0, 0, 0, 0, 0, 1913, 0, 10429, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15241, 0, 0, 0, 0, 0, 3043, 0, 0, 0, 0, 0, 19177, 0, 0, 0, 9743, 0, 0, 0, 0, 0, 11567, 0, 0, 0, 0, 0, 2669, 0, 22021, 0, 0, 0, 0, 0, 2803, 0, 0, 0, 26471, 0, 0, 0, 0, 0, 16283, 0, 19549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27509, 0, 19951, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4297, 0, 0, 0, 24113, 0, 26239, 0, 0, 0, 23867, 0, 0, 0, 0, 0, 18299, 0, 22531, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19991, 0, 26167, 0, 0, 0, 19673, 0, 0, 0, 0, 0, 19847, 0, 0, 0, 0, 0, 1979, 0, 26071, 0, 0, 0, 0, 0, 4573, 0, 0, 0, 0, 0, 1447, 0, 0, 0, 0, 0, 8719, 0, 0, 0, 11507, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21013, 0, 0, 0, 27221, 0, 24727, 0, 0, 0, 26723, 0, 1549, 0, 0, 0, 6077, 0, 0, 0, 0, 0, 0, 0, 12751, 0, 0, 0, 0, 0, 28183, 0, 0, 0, 17741, 0, 0, 0, 0, 0, 0, 0, 23029, 0, 0, 0, 6767, 0, 0, 0, 0, 0, 22859, 0, 5581, 0, 0, 0, 0, 0, 25723, 0, 0, 0, 0, 0, 9727, 0, 0, 0, 0, 0, 26839, 0, 0, 0, 15767, 0, 0, 0, 0, 0, 25349, 0, 0, 0, 0, 0, 0, 0, 26743, 0, 0, 0, 9041, 0, 0, 0, 0, 0, 15233, 0, 4639, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28349, 0, 26671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21821, 0, 22357, 0, 0, 0, 4373, 0, 2929, 0, 0, 0, 21347, 0, 0, 0, 0, 0, 4019, 0, 7621, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10541, 0, 20077, 0, 0, 0, 6863, 0, 0, 0, 0, 0, 21107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25153, 0, 0, 0, 10781, 0, 24127, 0, 0, 0, 0, 0, 29089, 0, 0, 0, 10247, 0, 0, 0, 0, 0, 19169, 0, 0, 0, 0, 0, 0, 0, 22273, 0, 0, 0, 8321, 0, 0, 0, 0, 0, 28823, 0, 13939, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9391, 0, 0, 0, 0, 0, 3823, 0, 0, 0, 16271, 0, 0, 0, 0, 0, 2333, 0, 0, 0, 0, 0, 5717, 0, 0, 0, 0, 0, 7739, 0, 14611, 0, 0, 0, 0, 0, 10603, 0, 0, 0, 0, 0, 8677, 0, 0, 0, 27593, 0, 0, 0, 0, 0, 17867, 0, 0, 0, 0, 0, 989, 0, 3961, 0, 0, 0, 0, 0, 11833, 0, 0, 0, 0, 0, 5077, 0, 0, 0, 27623, 0, 6739, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29609, 0, 12391, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26231, 0, 21097, 0, 0, 0, 24743, 0, 6499, 0, 0, 0, 13997, 0, 0, 0, 0, 0, 24599, 0, 3841, 0, 0, 0, 0, 0, 1453, 0, 0, 0, 7181, 0, 21337, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15017, 0, 0, 0, 0, 0, 0, 0, 22081, 0, 0, 0, 0, 0, 10873, 0, 0, 0, 28001, 0, 21607, 0, 0, 0, 0, 0, 9349, 0, 0, 0, 28307, 0, 0, 0, 0, 0, 7199, 0, 0, 0, 0, 0, 0, 0, 23113, 0, 0, 0, 25541, 0, 15907, 0, 0, 0, 13913, 0, 12889, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16111, 0, 0, 0, 0, 0, 26503, 0, 0, 0, 12911, 0, 0, 0, 0, 0, 0, 0, 10849, 0, 0, 0, 26297, 0, 0, 0, 0, 0, 389, 0, 6631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10163, 0, 27259, 0, 0, 0, 19127, 0, 0, 0, 0, 0, 0, 0, 18871, 0, 0, 0, 0, 0, 16243, 0, 0, 0, 18701, 0, 23137, 0, 0, 0, 24263, 0, 27109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1679, 0, 11131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 401, 0, 0, 0, 0, 0, 3743, 0, 25819, 0, 0, 0, 28907, 0, 0, 0, 0, 0, 23549, 0, 0, 0, 0, 0, 0, 0, 193, 0, 0, 0, 5711, 0, 337, 0, 0, 0, 18833, 0, 0, 0, 0, 0, 107, 0, 0, 0, 0, 0, 24239, 0, 26491, 0, 0, 0, 0, 0, 25783, 0, 0, 0, 9101, 0, 25177, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5939, 0, 0, 0, 0, 0, 0, 0, 15133, 0, 0, 0, 0, 0, 26197, 0, 0, 0, 10133, 0, 0, 0, 0, 0, 28337, 0, 0, 0, 0, 0, 0, 0, 8761, 0, 0, 0, 0, 0, 3193, 0, 0, 0, 3461, 0, 0, 0, 0, 0, 28373, 0, 9799, 0, 0, 0, 12017, 0, 0, 0, 0, 0, 27899, 0, 7471, 0, 0, 0, 0, 0, 8713, 0, 0, 0, 0, 0, 29257, 0, 0, 0, 0, 0, 7519, 0, 0, 0, 19967, 0, 0, 0, 0, 0, 22199, 0, 4591, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1271, 0, 13687, 0, 0, 0, 953, 0, 14299, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5879, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10691, 0, 5347, 0, 0, 0, 25793, 0, 29389, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2581, 0, 0, 0, 0, 0, 15943, 0, 0, 0, 9281, 0, 11467, 0, 0, 0, 3713, 0, 0, 0, 0, 0, 4517, 0, 0, 0, 0, 0, 12269, 0, 11581, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18131, 0, 23707, 0, 0, 0, 0, 0, 9769, 0, 0, 0, 27047, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24163, 0, 0, 0, 9791, 0, 367, 0, 0, 0, 0, 0, 709, 0, 0, 0, 27077, 0, 0, 0, 0, 0, 0, 0, 14431, 0, 0, 0, 0, 0, 29863, 0, 0, 0, 21521, 0, 0, 0, 0, 0, 2753, 0, 22189, 0, 0, 0, 26927, 0, 0, 0, 0, 0, 1439, 0, 8731, 0, 0, 0, 0, 0, 21313, 0, 0, 0, 0, 0, 3637, 0, 0, 0, 8273, 0, 27889, 0, 0, 0, 11987, 0, 0, 0, 0, 0, 2459, 0, 25171, 0, 0, 0, 0, 0, 12253, 0, 0, 0, 5471, 0, 10327, 0, 0, 0, 23633, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22567, 0, 0, 0, 26633, 0, 6079, 0, 0, 0, 27017, 0, 0, 0, 0, 0, 14099, 0, 20641, 0, 0, 0, 0, 0, 22663, 0, 0, 0, 6761, 0, 28687, 0, 0, 0, 19463, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2819, 0, 5491, 0, 0, 0, 0, 0, 7723, 0, 0, 0, 19391, 0, 20347, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23267, 0, 0, 0, 0, 0, 20219, 0, 0, 0, 0, 0, 0, 0, 14503, 0, 0, 0, 8531, 0, 4777, 0, 0, 0, 8873, 0, 13309, 0, 0, 0, 12797, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12223, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17873, 0, 18829, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29999, 0, 10831, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7837, 0, 0, 0, 23183, 0, 6889, 0, 0, 0, 21017, 0, 0, 0, 0, 0, 6659, 0, 24121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7571, 0, 8857, 0, 0, 0, 29303, 0, 26269, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3361, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1871, 0, 3667, 0, 0, 0, 27893, 0, 21829, 0, 0, 0, 9587, 0, 0, 0, 0, 0, 18509, 0, 19381, 0, 0, 0, 0, 0, 26233, 0, 0, 0, 29441, 0, 0, 0, 0, 0, 10013, 0, 0, 0, 0, 0, 527, 0, 0, 0, 0, 0, 29489, 0, 6751, 0, 0, 0, 0, 0, 24943, 0, 0, 0, 13301, 0, 10897, 0, 0, 0, 0, 0, 5779, 0, 0, 0, 8357, 0, 0, 0, 0, 0, 7409, 0, 0, 0, 0, 0, 0, 0, 24793, 0, 0, 0, 26591, 0, 0, 0, 0, 0, 26933, 0, 0, 0, 0, 0, 19517, 0, 0, 0, 0, 0, 0, 0, 5611, 0, 0, 0, 0, 0, 11173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11783, 0, 25549, 0, 0, 0, 8867, 0, 0, 0, 0, 0, 26639, 0, 15031, 0, 0, 0, 0, 0, 9133, 0, 0, 0, 0, 0, 18127, 0, 0, 0, 15833, 0, 28939, 0, 0, 0, 11357, 0, 0, 0, 0, 0, 11069, 0, 0, 0, 0, 0, 0, 0, 12673, 0, 0, 0, 8831, 0, 12427, 0, 0, 0, 0, 0, 11149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28139, 0, 21421, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7541, 0, 12697, 0, 0, 0, 29993, 0, 20569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22709, 0, 0, 0, 0, 0, 0, 0, 15523, 0, 0, 0, 14321, 0, 12937, 0, 0, 0, 8963, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28019, 0, 15781, 0, 0, 0, 0, 0, 15283, 0, 0, 0, 28421, 0, 28957, 0, 0, 0, 0, 0, 10189, 0, 0, 0, 10457, 0, 0, 0, 0, 0, 18539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25357, 0, 0, 0, 25673, 0, 11419, 0, 0, 0, 23087, 0, 0, 0, 0, 0, 0, 0, 15901, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13751, 0, 0, 0, 0, 0, 13043, 0, 18199, 0, 0, 0, 26087, 0, 0, 0, 0, 0, 17189, 0, 27631, 0, 0, 0, 0, 0, 8083, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13313, 0, 29779, 0, 0, 0, 21647, 0, 0, 0, 0, 0, 0, 0, 14671, 0, 0, 0, 0, 0, 22963, 0, 0, 0, 9671, 0, 9907, 0, 0, 0, 20483, 0, 26899, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10709, 0, 11971, 0, 0, 0, 0, 0, 0, 0, 0, 0, 191, 0, 13957, 0, 0, 0, 4163, 0, 0, 0, 0, 0, 27437, 0, 0, 0, 0, 0, 2969, 0, 3631, 0, 0, 0, 0, 0, 19093, 0, 0, 0, 0, 0, 11677, 0, 0, 0, 0, 0, 0, 0, 0, 0, 947, 0, 0, 0, 0, 0, 1559, 0, 26911, 0, 0, 0, 0, 0, 17383, 0, 0, 0, 2801, 0, 0, 0, 0, 0, 0, 0, 25309, 0, 0, 0, 797, 0, 0, 0, 0, 0, 27569, 0, 0, 0, 0, 0, 0, 0, 25213, 0, 0, 0, 25331, 0, 24307, 0, 0, 0, 0, 0, 26329, 0, 0, 0, 12377, 0, 0, 0, 0, 0, 0, 0, 20101, 0, 0, 0, 0, 0, 29023, 0, 0, 0, 12491, 0, 0, 0, 0, 0, 22073, 0, 23869, 0, 0, 0, 16427, 0, 0, 0, 0, 0, 5219, 0, 0, 0, 0, 0, 0, 0, 20473, 0, 0, 0, 0, 0, 9307, 0, 0, 0, 16883, 0, 1009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1619, 0, 19081, 0, 0, 0, 0, 0, 13303, 0, 0, 0, 1691, 0, 2557, 0, 0, 0, 743, 0, 17449, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14909, 0, 8611, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6911, 0, 7867, 0, 0, 0, 16763, 0, 0, 0, 0, 0, 26387, 0, 0, 0, 0, 0, 0, 0, 20851, 0, 0, 0, 0, 0, 25813, 0, 0, 0, 12431, 0, 29737, 0, 0, 0, 29543, 0, 0, 0, 0, 0, 11237, 0, 0, 0, 0, 0, 29069, 0, 14101, 0, 0, 0, 0, 0, 2473, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19219, 0, 0, 0, 18017, 0, 0, 0, 0, 0, 28829, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21551, 0, 14857, 0, 0, 0, 9923, 0, 18979, 0, 0, 0, 15947, 0, 0, 0, 0, 0, 0, 0, 24511, 0, 0, 0, 0, 0, 11593, 0, 0, 0, 28241, 0, 0, 0, 0, 0, 3173, 0, 0, 0, 0, 0, 18527, 0, 0, 0, 0, 0, 0, 0, 7261, 0, 0, 0, 0, 0, 17113, 0, 0, 0, 0, 0, 14977, 0, 0, 0, 15413, 0, 3109, 0, 0, 0, 22067, 0, 0, 0, 0, 0, 569, 0, 23281, 0, 0, 0, 0, 0, 22333, 0, 0, 0, 10721, 0, 17467, 0, 0, 0, 4943, 0, 16399, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17009, 0, 7141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3551, 0, 22987, 0, 0, 0, 0, 0, 12799, 0, 0, 0, 8747, 0, 0, 0, 0, 0, 24179, 0, 1951, 0, 0, 0, 0, 0, 11533, 0, 0, 0, 25031, 0, 0, 0, 0, 0, 3923, 0, 0, 0, 0, 0, 1577, 0, 0, 0, 0, 0, 21719, 0, 0, 0, 0, 0, 0, 0, 28723, 0, 0, 0, 28841, 0, 21187, 0, 0, 0, 0, 0, 20479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22739, 0, 0, 0, 0, 0, 0, 0, 21223, 0, 0, 0, 6641, 0, 577, 0, 0, 0, 27143, 0, 16459, 0, 0, 0, 22667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4931, 0, 0, 0, 0, 0, 20393, 0, 15049, 0, 0, 0, 3617, 0, 0, 0, 0, 0, 0, 0, 19861, 0, 0, 0, 0, 0, 23833, 0, 0, 0, 0, 0, 7627, 0, 0, 0, 12053, 0, 7309, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21149, 0, 3541, 0, 0, 0, 0, 0, 14353, 0, 0, 0, 1061, 0, 0, 0, 0, 0, 9353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18269, 0, 10711, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15941, 0, 27397, 0, 0, 0, 26423, 0, 829, 0, 0, 0, 5387, 0, 0, 0, 0, 0, 29849, 0, 10981, 0, 0, 0, 0, 0, 10273, 0, 0, 0, 0, 0, 28477, 0, 0, 0, 8123, 0, 0, 0, 0, 0, 10607, 0, 0, 0, 0, 0, 6599, 0, 28381, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19927, 0, 0, 0, 0, 0, 29509, 0, 0, 0, 29987, 0, 0, 0, 0, 0, 7829, 0, 0, 0, 0, 0, 0, 0, 25633, 0, 0, 0, 8741, 0, 15487, 0, 0, 0, 8243, 0, 20029, 0, 0, 0, 8387, 0, 0, 0, 0, 0, 0, 0, 15061, 0, 0, 0, 0, 0, 9703, 0, 0, 0, 8501, 0, 0, 0, 0, 0, 0, 0, 25339, 0, 0, 0, 29867, 0, 0, 0, 0, 0, 27479, 0, 24061, 0, 0, 0, 0, 0, 16483, 0, 0, 0, 0, 0, 14347, 0, 0, 0, 2603, 0, 19909, 0, 0, 0, 18077, 0, 0, 0, 0, 0, 6869, 0, 0, 0, 0, 0, 0, 0, 15193, 0, 0, 0, 11351, 0, 15577, 0, 0, 0, 0, 0, 6949, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19109, 0, 8191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14891, 0, 0, 0, 0, 0, 8993, 0, 21409, 0, 0, 0, 12107, 0, 0, 0, 0, 0, 22499, 0, 12241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12851, 0, 24697, 0, 0, 0, 18413, 0, 0, 0, 0, 0, 2627, 0, 0, 0, 0, 0, 0, 0, 27121, 0, 0, 0, 0, 0, 27463, 0, 0, 0, 29261, 0, 5647, 0, 0, 0, 0, 0, 10609, 0, 0, 0, 14867, 0, 0, 0, 0, 0, 12239, 0, 0, 0, 0, 0, 0, 0, 10723, 0, 0, 0, 29111, 0, 0, 0, 0, 0, 17273, 0, 18559, 0, 0, 0, 7127, 0, 0, 0, 0, 0, 0, 0, 14011, 0, 0, 0, 0, 0, 24613, 0, 0, 0, 27821, 0, 0, 0, 0, 0, 4643, 0, 29539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26219, 0, 26161, 0, 0, 0, 0, 0, 22153, 0, 0, 0, 0, 0, 10987, 0, 0, 0, 20663, 0, 0, 0, 0, 0, 22487, 0, 0, 0, 0, 0, 21779, 0, 24751, 0, 0, 0, 0, 0, 16453, 0, 0, 0, 0, 0, 28177, 0, 0, 0, 29933, 0, 11359, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11129, 0, 841, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23407, 0, 0, 0, 13193, 0, 18049, 0, 0, 0, 4757, 0, 0, 0, 0, 0, 29219, 0, 6151, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11801, 0, 9787, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3467, 0, 0, 0, 0, 0, 4709, 0, 15151, 0, 0, 0, 0, 0, 15493, 0, 0, 0, 9521, 0, 12367, 0, 0, 0, 0, 0, 27829, 0, 0, 0, 587, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4633, 0, 0, 0, 16301, 0, 27457, 0, 0, 0, 18533, 0, 15199, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4561, 0, 0, 0, 0, 0, 17263, 0, 0, 0, 1361, 0, 0, 0, 0, 0, 3383, 0, 3919, 0, 0, 0, 28607, 0, 0, 0, 0, 0, 11939, 0, 27421, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23377, 0, 0, 0, 0, 0, 29569, 0, 0, 0, 7577, 0, 0, 0, 0, 0, 0, 0, 391, 0, 0, 0, 0, 0, 18553, 0, 0, 0, 11771, 0, 0, 0, 0, 0, 28883, 0, 15559, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20159, 0, 15751, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2711, 0, 0, 0, 0, 0, 15293, 0, 16579, 0, 0, 0, 10427, 0, 0, 0, 0, 0, 25859, 0, 21271, 0, 0, 0, 0, 0, 2503, 0, 0, 0, 24191, 0, 11887, 0, 0, 0, 9593, 0, 0, 0, 0, 0, 4727, 0, 0, 0, 0, 0, 17309, 0, 0, 0, 0, 0, 0, 0, 14233, 0, 0, 0, 29891, 0, 15937, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21167, 0, 0, 0, 0, 0, 8249, 0, 0, 0, 0, 0, 0, 0, 5893, 0, 0, 0, 0, 0, 14647, 0, 0, 0, 12443, 0, 5749, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20311, 0, 0, 0, 0, 0, 14743, 0, 0, 0, 8081, 0, 0, 0, 0, 0, 21443, 0, 0, 0, 0, 0, 16637, 0, 0, 0, 0, 0, 18659, 0, 28261, 0, 0, 0, 0, 0, 13333, 0, 0, 0, 0, 0, 22327, 0, 0, 0, 0, 0, 12139, 0, 0, 0, 1487, 0, 0, 0, 0, 0, 3719, 0, 23071, 0, 0, 0, 0, 0, 22753, 0, 0, 0, 0, 0, 15997, 0, 0, 0, 19433, 0, 25849, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10499, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15311, 0, 23827, 0, 0, 0, 16553, 0, 20149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8219, 0, 25681, 0, 0, 0, 0, 0, 20563, 0, 0, 0, 20831, 0, 2227, 0, 0, 0, 15263, 0, 0, 0, 0, 0, 6827, 0, 0, 0, 0, 0, 0, 0, 13891, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8891, 0, 5227, 0, 0, 0, 0, 0, 12079, 0, 0, 0, 20117, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14923, 0, 0, 0, 28271, 0, 4987, 0, 0, 0, 27563, 0, 23809, 0, 0, 0, 29387, 0, 0, 0, 0, 0, 0, 0, 5191, 0, 0, 0, 0, 0, 18313, 0, 0, 0, 23831, 0, 0, 0, 0, 0, 0, 0, 24499, 0, 0, 0, 15377, 0, 0, 0, 0, 0, 22229, 0, 20281, 0, 0, 0, 0, 0, 23623, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12893, 0, 16339, 0, 0, 0, 2747, 0, 0, 0, 0, 0, 20939, 0, 29791, 0, 0, 0, 0, 0, 5323, 0, 0, 0, 7781, 0, 14947, 0, 0, 0, 5153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20789, 0, 13861, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4087, 0, 0, 0, 17393, 0, 17629, 0, 0, 0, 1607, 0, 0, 0, 0, 0, 7169, 0, 0, 0, 0, 0, 0, 0, 11113, 0, 0, 0, 27551, 0, 19447, 0, 0, 0, 7913, 0, 0, 0, 0, 0, 11027, 0, 0, 0, 0, 0, 5129, 0, 10111, 0, 0, 0, 0, 0, 28513, 0, 0, 0, 911, 0, 8797, 0, 0, 0, 0, 0, 10819, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8669, 0, 0, 0, 0, 0, 0, 0, 26053, 0, 0, 0, 13151, 0, 7087, 0, 0, 0, 1943, 0, 0, 0, 0, 0, 17417, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16843, 0, 0, 0, 22571, 0, 0, 0, 0, 0, 20183, 0, 23449, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11519, 0, 29311, 0, 0, 0, 0, 0, 27823, 0, 0, 0, 0, 0, 10147, 0, 0, 0, 11633, 0, 18439, 0, 0, 0, 11777, 0, 0, 0, 0, 0, 11279, 0, 26431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12191, 0, 27337, 0, 0, 0, 20063, 0, 17029, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26461, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13421, 0, 24457, 0, 0, 0, 9413, 0, 10279, 0, 0, 0, 11897, 0, 0, 0, 0, 0, 0, 0, 21691, 0, 0, 0, 0, 0, 7753, 0, 0, 0, 20201, 0, 0, 0, 0, 0, 14633, 0, 0, 0, 0, 0, 23627, 0, 0, 0, 0, 0, 4079, 0, 25231, 0, 0, 0, 0, 0, 15703, 0, 0, 0, 1751, 0, 15517, 0, 0, 0, 0, 0, 28879, 0, 0, 0, 10667, 0, 0, 0, 0, 0, 18959, 0, 0, 0, 0, 0, 0, 0, 13243, 0, 0, 0, 28901, 0, 22207, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7967, 0, 0, 0, 0, 0, 0, 0, 17161, 0, 0, 0, 0, 0, 13483, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16403, 0, 13999, 0, 0, 0, 29657, 0, 0, 0, 0, 0, 15089, 0, 19651, 0, 0, 0, 0, 0, 2203, 0, 0, 0, 0, 0, 22747, 0, 0, 0, 27383, 0, 19699, 0, 0, 0, 22907, 0, 0, 0, 0, 0, 13379, 0, 8791, 0, 0, 0, 0, 0, 14983, 0, 0, 0, 27311, 0, 23977, 0, 0, 0, 0, 0, 22699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28331, 0, 3457, 0, 0, 0, 18443, 0, 25189, 0, 0, 0, 16097, 0, 0, 0, 0, 0, 25019, 0, 0, 0, 0, 0, 0, 0, 6283, 0, 0, 0, 25871, 0, 1387, 0, 0, 0, 11273, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16469, 0, 27331, 0, 0, 0, 0, 0, 26833, 0, 0, 0, 3011, 0, 1237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15077, 0, 0, 0, 0, 0, 9299, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19451, 0, 18427, 0, 0, 0, 27983, 0, 16039, 0, 0, 0, 4607, 0, 0, 0, 0, 0, 0, 0, 4351, 0, 0, 0, 0, 0, 4033, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1493, 0, 29749, 0, 0, 0, 16847, 0, 0, 0, 0, 0, 21809, 0, 29941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4073, 0, 20539, 0, 0, 0, 10097, 0, 0, 0, 0, 0, 28499, 0, 7741, 0, 0, 0, 0, 0, 27583, 0, 0, 0, 21221, 0, 667, 0, 0, 0, 2003, 0, 15349, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14281, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20981, 0, 25507, 0, 0, 0, 8783, 0, 0, 0, 0, 0, 20507, 0, 0, 0, 0, 0, 7589, 0, 22111, 0, 0, 0, 0, 0, 9853, 0, 0, 0, 0, 0, 16297, 0, 0, 0, 23663, 0, 0, 0, 0, 0, 3257, 0, 0, 0, 0, 0, 13109, 0, 17671, 0, 0, 0, 0, 0, 5833, 0, 0, 0, 5111, 0, 30007, 0, 0, 0, 0, 0, 27619, 0, 0, 0, 19277, 0, 0, 0, 0, 0, 18329, 0, 0, 0, 0, 0, 0, 0, 27523, 0, 0, 0, 18401, 0, 0, 0, 0, 0, 0, 0, 14779, 0, 0, 0, 3137, 0, 0, 0, 0, 0, 0, 0, 24721, 0, 0, 0, 0, 0, 22093, 0, 0, 0, 14801, 0, 0, 0, 0, 0, 3593, 0, 14629, 0, 0, 0, 27977, 0, 0, 0, 0, 0, 7529, 0, 0, 0, 0, 0, 0, 0, 22783, 0, 0, 0, 0, 0, 20857, 0, 0, 0, 7643, 0, 12559, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24719, 0, 0, 0, 0, 0, 0, 0, 1753, 0, 0, 0, 22481, 0, 23347, 0, 0, 0, 19223, 0, 22069, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17219, 0, 13231, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18461, 0, 26347, 0, 0, 0, 19073, 0, 23299, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2371, 0, 0, 0, 0, 0, 7333, 0, 0, 0, 17051, 0, 2017, 0, 0, 0, 22613, 0, 0, 0, 0, 0, 15857, 0, 0, 0, 0, 0, 19829, 0, 4861, 0, 0, 0, 0, 0, 7093, 0, 0, 0, 9311, 0, 0, 0, 0, 0, 0, 0, 23839, 0, 0, 0, 29567, 0, 0, 0, 0, 0, 10349, 0, 0, 0, 0, 0, 0, 0, 26263, 0, 0, 0, 0, 0, 17167, 0, 0, 0, 28403, 0, 499, 0, 0, 0, 6707, 0, 0, 0, 0, 0, 0, 0, 26821, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2831, 0, 0, 0, 0, 0, 23963, 0, 0, 0, 0, 0, 6977, 0, 0, 0, 0, 0, 6269, 0, 331, 0, 0, 0, 0, 0, 21733, 0, 0, 0, 0, 0, 5737, 0, 0, 0, 26963, 0, 21589, 0, 0, 0, 24377, 0, 0, 0, 0, 0, 0, 0, 25591, 0, 0, 0, 0, 0, 3853, 0, 0, 0, 1481, 0, 29017, 0, 0, 0, 9563, 0, 18709, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21629, 0, 25621, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22031, 0, 27607, 0, 0, 0, 0, 0, 5869, 0, 0, 0, 11057, 0, 0, 0, 0, 0, 5699, 0, 22741, 0, 0, 0, 0, 0, 30013, 0, 0, 0, 27341, 0, 22597, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20057, 0, 0, 0, 0, 0, 12479, 0, 0, 0, 0, 0, 0, 0, 1003, 0, 0, 0, 21911, 0, 0, 0, 0, 0, 0, 0, 8929, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11189, 0, 0, 0, 0, 0, 0, 0, 14293, 0, 0, 0, 8951, 0, 5197, 0, 0, 0, 8663, 0, 7219, 0, 0, 0, 4187, 0, 0, 0, 0, 0, 0, 0, 11911, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23411, 0, 0, 0, 0, 0, 11153, 0, 26599, 0, 0, 0, 8237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12283, 0, 0, 0, 0, 0, 28417, 0, 0, 0, 503, 0, 11929, 0, 0, 0, 23117, 0, 0, 0, 0, 0, 23459, 0, 8161, 0, 0, 0, 0, 0, 5113, 0, 0, 0, 12611, 0, 0, 0, 0, 0, 11663, 0, 1069, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6719, 0, 22261, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20561, 0, 29707, 0, 0, 0, 19493, 0, 0, 0, 0, 0, 10007, 0, 0, 0, 0, 0, 20609, 0, 1741, 0, 0, 0, 0, 0, 14893, 0, 0, 0, 1511, 0, 21547, 0, 0, 0, 10433, 0, 0, 0, 0, 0, 22157, 0, 0, 0, 0, 0, 18149, 0, 16831, 0, 0, 0, 0, 0, 16123, 0, 0, 0, 0, 0, 22237, 0, 0, 0, 0, 0, 11029, 0, 0, 0, 20747, 0, 0, 0, 0, 0, 12449, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13361, 0, 3937, 0, 0, 0, 29033, 0, 10789, 0, 0, 0, 26867, 0, 0, 0, 0, 0, 0, 0, 8131, 0, 0, 0, 0, 0, 14323, 0, 0, 0, 20051, 0, 0, 0, 0, 0, 0, 0, 13789, 0, 0, 0, 2147, 0, 0, 0, 0, 0, 0, 0, 26371, 0, 0, 0, 0, 0, 28033, 0, 0, 0, 0, 0, 25897, 0, 0, 0, 7223, 0, 22219, 0, 0, 0, 11147, 0, 0, 0, 0, 0, 11489, 0, 0, 0, 0, 0, 0, 0, 5953, 0, 0, 0, 29831, 0, 20197, 0, 0, 0, 26783, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 629, 0, 28981, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17201, 0, 3877, 0, 0, 0, 0, 0, 23719, 0, 0, 0, 557, 0, 0, 0, 0, 0, 13259, 0, 23791, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5921, 0, 0, 0, 0, 0, 23033, 0, 0, 0, 0, 0, 23417, 0, 0, 0, 0, 0, 0, 0, 1711, 0, 0, 0, 0, 0, 20533, 0, 0, 0, 1541, 0, 10267, 0, 0, 0, 0, 0, 1369, 0, 0, 0, 26417, 0, 0, 0, 0, 0, 14549, 0, 0, 0, 0, 0, 0, 0, 13033, 0, 0, 0, 17561, 0, 0, 0, 0, 0, 8033, 0, 79, 0, 0, 0, 11747, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13063, 0, 0, 0, 18581, 0, 0, 0, 0, 0, 23123, 0, 4129, 0, 0, 0, 17267, 0, 0, 0, 0, 0, 28529, 0, 751, 0, 0, 0, 0, 0, 12913, 0, 0, 0, 0, 0, 29467, 0, 0, 0, 22973, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10229, 0, 6271, 0, 0, 0, 0, 0, 28003, 0, 0, 0, 0, 0, 457, 0, 0, 0, 23003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1889, 0, 21631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7751, 0, 16477, 0, 0, 0, 15503, 0, 22669, 0, 0, 0, 16307, 0, 0, 0, 0, 0, 10739, 0, 24631, 0, 0, 0, 0, 0, 13003, 0, 0, 0, 0, 0, 12097, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24257, 0, 0, 0, 0, 0, 9329, 0, 17461, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14141, 0, 817, 0, 0, 0, 0, 0, 18589, 0, 0, 0, 19067, 0, 0, 0, 0, 0, 18749, 0, 0, 0, 0, 0, 0, 0, 9253, 0, 0, 0, 27851, 0, 18217, 0, 0, 0, 53, 0, 3649, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6871, 0, 0, 0, 0, 0, 28813, 0, 0, 0, 22151, 0, 0, 0, 0, 0, 8003, 0, 6229, 0, 0, 0, 21677, 0, 0, 0, 0, 0, 16559, 0, 15871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27997, 0, 0, 0, 21713, 0, 22639, 0, 0, 0, 9887, 0, 0, 0, 0, 0, 0, 0, 21181, 0, 0, 0, 0, 0, 7003, 0, 0, 0, 14081, 0, 4657, 0, 0, 0, 0, 0, 17869, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10919, 0, 27301, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25811, 0, 0, 0, 0, 0, 19913, 0, 5029, 0, 0, 0, 1187, 0, 0, 0, 0, 0, 14309, 0, 25891, 0, 0, 0, 0, 0, 25603, 0, 0, 0, 26501, 0, 16507, 0, 0, 0, 21143, 0, 0, 0, 0, 0, 16277, 0, 0, 0, 0, 0, 19619, 0, 8011, 0, 0, 0, 0, 0, 16543, 0, 0, 0, 18341, 0, 27487, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25787, 0, 0, 0, 0, 0, 1319, 0, 0, 0, 0, 0, 0, 0, 24373, 0, 0, 0, 0, 0, 5407, 0, 0, 0, 893, 0, 10369, 0, 0, 0, 9857, 0, 0, 0, 0, 0, 0, 0, 24931, 0, 0, 0, 0, 0, 5503, 0, 0, 0, 19631, 0, 0, 0, 0, 0, 23753, 0, 21349, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7109, 0, 9781, 0, 0, 0, 0, 0, 24883, 0, 0, 0, 0, 0, 24637, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6107, 0, 0, 0, 0, 0, 24509, 0, 13831, 0, 0, 0, 0, 0, 27373, 0, 0, 0, 12821, 0, 9067, 0, 0, 0, 21743, 0, 439, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22049, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26137, 0, 0, 0, 5003, 0, 1669, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12839, 0, 27991, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25451, 0, 20707, 0, 0, 0, 6023, 0, 0, 0, 0, 0, 25307, 0, 0, 0, 0, 0, 23819, 0, 6961, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20441, 0, 26017, 0, 0, 0, 0, 0, 529, 0, 0, 0, 22427, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26473, 0, 0, 0, 19031, 0, 16537, 0, 0, 0, 2153, 0, 26119, 0, 0, 0, 22457, 0, 0, 0, 0, 0, 0, 0, 23671, 0, 0, 0, 0, 0, 9073, 0, 0, 0, 12281, 0, 0, 0, 0, 0, 14303, 0, 17569, 0, 0, 0, 17687, 0, 0, 0, 0, 0, 3749, 0, 11041, 0, 0, 0, 0, 0, 12073, 0, 0, 0, 0, 0, 15187, 0, 0, 0, 0, 0, 18649, 0, 0, 0, 21227, 0, 0, 0, 0, 0, 11699, 0, 11311, 0, 0, 0, 0, 0, 7633, 0, 0, 0, 851, 0, 0, 0, 0, 0, 9773, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9239, 0, 18481, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8707, 0, 0, 0, 28943, 0, 8389, 0, 0, 0, 13157, 0, 0, 0, 0, 0, 9479, 0, 2161, 0, 0, 0, 0, 0, 13423, 0, 0, 0, 16001, 0, 967, 0, 0, 0, 28703, 0, 0, 0, 0, 0, 15647, 0, 0, 0, 0, 0, 28229, 0, 0, 0, 0, 0, 0, 0, 16963, 0, 0, 0, 21701, 0, 29587, 0, 0, 0, 0, 0, 15439, 0, 0, 0, 4787, 0, 0, 0, 0, 0, 10979, 0, 0, 0, 0, 0, 0, 0, 16813, 0, 0, 0, 24701, 0, 25567, 0, 0, 0, 4253, 0, 24859, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28393, 0, 0, 0, 27191, 0, 0, 0, 0, 0, 13253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2279, 0, 20071, 0, 0, 0, 0, 0, 2413, 0, 0, 0, 0, 0, 3217, 0, 0, 0, 13943, 0, 23059, 0, 0, 0, 23327, 0, 0, 0, 0, 0, 22829, 0, 14881, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29647, 0, 0, 0, 8513, 0, 28579, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18041, 0, 12907, 0, 0, 0, 173, 0, 1039, 0, 0, 0, 347, 0, 0, 0, 0, 0, 29, 0, 14761, 0, 0, 0, 0, 0, 12373, 0, 0, 0, 1721, 0, 0, 0, 0, 0, 26183, 0, 0, 0, 0, 0, 25937, 0, 0, 0, 0, 0, 0, 0, 27541, 0, 0, 0, 0, 0, 27253, 0, 0, 0, 22541, 0, 27067, 0, 0, 0, 0, 0, 1159, 0, 0, 0, 3737, 0, 0, 0, 0, 0, 23579, 0, 0, 0, 0, 0, 0, 0, 4003, 0, 0, 0, 17351, 0, 26827, 0, 0, 0, 8453, 0, 0, 0, 0, 0, 10277, 0, 0, 0, 0, 0, 0, 0, 7921, 0, 0, 0, 0, 0, 1933, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16309, 0, 0, 0, 18107, 0, 0, 0, 0, 0, 5849, 0, 1171, 0, 0, 0, 0, 0, 4513, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1973, 0, 8149, 0, 0, 0, 13667, 0, 0, 0, 0, 0, 1829, 0, 13411, 0, 0, 0, 0, 0, 8053, 0, 0, 0, 29621, 0, 28597, 0, 0, 0, 0, 0, 13459, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23519, 0, 2941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16781, 0, 15007, 0, 0, 0, 9203, 0, 6709, 0, 0, 0, 20717, 0, 0, 0, 0, 0, 18089, 0, 0, 0, 0, 0, 0, 0, 24763, 0, 0, 0, 16631, 0, 22177, 0, 0, 0, 29753, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18779, 0, 1921, 0, 0, 0, 0, 0, 17593, 0, 0, 0, 14561, 0, 19717, 0, 0, 0, 0, 0, 21739, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27779, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24071, 0, 20737, 0, 0, 0, 21053, 0, 0, 0, 0, 0, 9227, 0, 0, 0, 0, 0, 0, 0, 25141, 0, 0, 0, 0, 0, 8653, 0, 0, 0, 25301, 0, 0, 0, 0, 0, 3803, 0, 4339, 0, 0, 0, 28397, 0, 0, 0, 0, 0, 3329, 0, 18391, 0, 0, 0, 0, 0, 19633, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22553, 0, 2059, 0, 0, 0, 857, 0, 0, 0, 0, 0, 3089, 0, 10051, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25841, 0, 19147, 0, 0, 0, 22793, 0, 6109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22259, 0, 7351, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2501, 0, 16267, 0, 0, 0, 20333, 0, 0, 0, 0, 0, 22817, 0, 0, 0, 0, 0, 0, 0, 24421, 0, 0, 0, 0, 0, 21403, 0, 0, 0, 0, 0, 27847, 0, 0, 0, 28283, 0, 0, 0, 0, 0, 26357, 0, 0, 0, 0, 0, 17729, 0, 6121, 0, 0, 0, 0, 0, 26623, 0, 0, 0, 23591, 0, 4597, 0, 0, 0, 0, 0, 20689, 0, 0, 0, 21587, 0, 0, 0, 0, 0, 29879, 0, 0, 0, 0, 0, 0, 0, 15973, 0, 0, 0, 20711, 0, 5827, 0, 0, 0, 0, 0, 17089, 0, 0, 0, 21617, 0, 0, 0, 0, 0, 0, 0, 6241, 0, 0, 0, 0, 0, 24403, 0, 0, 0, 7871, 0, 0, 0, 0, 0, 8213, 0, 3079, 0, 0, 0, 18737, 0, 0, 0, 0, 0, 26009, 0, 0, 0, 0, 0, 0, 0, 15853, 0, 0, 0, 0, 0, 25477, 0, 0, 0, 19193, 0, 3319, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27029, 0, 601, 0, 0, 0, 0, 0, 4063, 0, 0, 0, 10931, 0, 4867, 0, 0, 0, 9983, 0, 3589, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10289, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9221, 0, 17107, 0, 0, 0, 7523, 0, 27919, 0, 0, 0, 7907, 0, 0, 0, 0, 0, 0, 0, 6991, 0, 0, 0, 0, 0, 28123, 0, 0, 0, 28601, 0, 20497, 0, 0, 0, 24923, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8279, 0, 16411, 0, 0, 0, 0, 0, 18643, 0, 0, 0, 13931, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4157, 0, 0, 0, 0, 0, 1109, 0, 0, 0, 0, 0, 0, 0, 853, 0, 0, 0, 3071, 0, 10237, 0, 0, 0, 683, 0, 5119, 0, 0, 0, 18257, 0, 0, 0, 0, 0, 0, 0, 15271, 0, 0, 0, 0, 0, 23143, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12413, 0, 0, 0, 0, 0, 27767, 0, 0, 0, 0, 0, 10889, 0, 2641, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24217, 0, 0, 0, 17723, 0, 12349, 0, 0, 0, 12827, 0, 0, 0, 0, 0, 12119, 0, 18661, 0, 0, 0, 0, 0, 8473, 0, 0, 0, 13031, 0, 19777, 0, 0, 0, 21113, 0, 7159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27931, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12791, 0, 9127, 0, 0, 0, 0, 0, 8179, 0, 0, 0, 4127, 0, 0, 0, 0, 0, 10319, 0, 11191, 0, 0, 0, 0, 0, 20773, 0, 0, 0, 15791, 0, 27217, 0, 0, 0, 15473, 0, 0, 0, 0, 0, 22367, 0, 0, 0, 0, 0, 24029, 0, 0, 0, 0, 0, 0, 0, 19483, 0, 0, 0, 24221, 0, 2707, 0, 0, 0, 0, 0, 11239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1949, 0, 0, 0, 0, 0, 0, 0, 16603, 0, 0, 0, 2021, 0, 0, 0, 0, 0, 13283, 0, 25699, 0, 0, 0, 24977, 0, 0, 0, 0, 0, 0, 0, 16531, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25721, 0, 0, 0, 0, 0, 22703, 0, 17359, 0, 0, 0, 19787, 0, 0, 0, 0, 0, 0, 0, 1381, 0, 0, 0, 0, 0, 14593, 0, 0, 0, 0, 0, 9937, 0, 0, 0, 21293, 0, 23479, 0, 0, 0, 27737, 0, 0, 0, 0, 0, 16529, 0, 0, 0, 0, 0, 0, 0, 23593, 0, 0, 0, 3371, 0, 0, 0, 0, 0, 113, 0, 5689, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9029, 0, 26881, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2081, 0, 18157, 0, 0, 0, 21803, 0, 12379, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9059, 0, 13291, 0, 0, 0, 0, 0, 26443, 0, 0, 0, 6131, 0, 23857, 0, 0, 0, 3503, 0, 0, 0, 0, 0, 26777, 0, 0, 0, 0, 0, 8909, 0, 7591, 0, 0, 0, 0, 0, 20743, 0, 0, 0, 0, 0, 15307, 0, 0, 0, 0, 0, 15649, 0, 0, 0, 2267, 0, 0, 0, 0, 0, 23999, 0, 0, 0, 0, 0, 0, 0, 7153, 0, 0, 0, 0, 0, 6247, 0, 0, 0, 17483, 0, 22339, 0, 0, 0, 17627, 0, 0, 0, 0, 0, 0, 0, 10441, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24671, 0, 0, 0, 0, 0, 0, 0, 4549, 0, 0, 0, 20627, 0, 0, 0, 0, 0, 8999, 0, 19441, 0, 0, 0, 0, 0, 2623, 0, 0, 0, 0, 0, 16657, 0, 0, 0, 18773, 0, 10669, 0, 0, 0, 13457, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17503, 0, 0, 0, 20591, 0, 1717, 0, 0, 0, 1373, 0, 2329, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5249, 0, 17431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5651, 0, 8497, 0, 0, 0, 20543, 0, 16789, 0, 0, 0, 2867, 0, 0, 0, 0, 0, 24809, 0, 14551, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8231, 0, 6217, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11867, 0, 0, 0, 0, 0, 0, 0, 13261, 0, 0, 0, 0, 0, 22843, 0, 0, 0, 24641, 0, 0, 0, 0, 0, 0, 0, 19849, 0, 0, 0, 17177, 0, 0, 0, 0, 0, 2999, 0, 0, 0, 0, 0, 0, 0, 6103, 0, 0, 0, 19871, 0, 0, 0, 0, 0, 19583, 0, 20869, 0, 0, 0, 23297, 0, 0, 0, 0, 0, 0, 0, 25561, 0, 0, 0, 0, 0, 15373, 0, 0, 0, 7031, 0, 0, 0, 0, 0, 13883, 0, 15679, 0, 0, 0, 21887, 0, 0, 0, 0, 0, 21599, 0, 0, 0, 0, 0, 0, 0, 1363, 0, 0, 0, 0, 0, 20227, 0, 0, 0, 11423, 0, 0, 0, 0, 0, 4007, 0, 0, 0, 0, 0, 12539, 0, 10891, 0, 0, 0, 0, 0, 18763, 0, 0, 0, 0, 0, 18937, 0, 0, 0, 25313, 0, 22909, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20369, 0, 3151, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12371, 0, 18787, 0, 0, 0, 8573, 0, 0, 0, 0, 0, 20927, 0, 0, 0, 0, 0, 1499, 0, 15391, 0, 0, 0, 0, 0, 17623, 0, 0, 0, 23351, 0, 5167, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5777, 0, 0, 0, 0, 0, 20879, 0, 5911, 0, 0, 0, 0, 0, 27043, 0, 0, 0, 0, 0, 3127, 0, 0, 0, 0, 0, 109, 0, 0, 0, 9827, 0, 0, 0, 0, 0, 23369, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2441, 0, 6667, 0, 0, 0, 20843, 0, 24439, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29971, 0, 0, 0, 0, 0, 3403, 0, 0, 0, 3671, 0, 0, 0, 0, 0, 19553, 0, 24709, 0, 0, 0, 23987, 0, 0, 0, 0, 0, 0, 0, 18181, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9517, 0, 0, 0, 26333, 0, 24949, 0, 0, 0, 2957, 0, 0, 0, 0, 0, 0, 0, 9631, 0, 0, 0, 0, 0, 27793, 0, 0, 0, 2531, 0, 9277, 0, 0, 0, 10403, 0, 10939, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22469, 0, 18061, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28121, 0, 0, 0, 0, 0, 0, 0, 7339, 0, 0, 0, 19667, 0, 0, 0, 0, 0, 5069, 0, 7411, 0, 0, 0, 0, 0, 27913, 0, 0, 0, 19571, 0, 0, 0, 0, 0, 25763, 0, 0, 0, 0, 0, 7037, 0, 0, 0, 0, 0, 8069, 0, 12631, 0, 0, 0, 0, 0, 9613, 0, 0, 0, 20651, 0, 2077, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7307, 0, 0, 0, 0, 0, 3629, 0, 0, 0, 0, 0, 0, 0, 26683, 0, 0, 0, 0, 0, 16957, 0, 0, 0, 21683, 0, 21919, 0, 0, 0, 14477, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23983, 0, 0, 0, 10391, 0, 0, 0, 0, 0, 12203, 0, 25969, 0, 0, 0, 28187, 0, 0, 0, 0, 0, 9419, 0, 14401, 0, 0, 0, 0, 0, 15643, 0, 0, 0, 0, 0, 13087, 0, 0, 0, 0, 0, 23689, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12959, 0, 25381, 0, 0, 0, 0, 0, 8893, 0, 0, 0, 17441, 0, 11377, 0, 0, 0, 14813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12809, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26861, 0, 19207, 0, 0, 0, 7313, 0, 6289, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24389, 0, 16441, 0, 0, 0, 0, 0, 2083, 0, 0, 0, 0, 0, 23017, 0, 0, 0, 24503, 0, 0, 0, 0, 0, 16067, 0, 0, 0, 0, 0, 28439, 0, 9271, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25061, 0, 14467, 0, 0, 0, 0, 0, 21319, 0, 0, 0, 10877, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1063, 0, 0, 0, 551, 0, 7297, 0, 0, 0, 13703, 0, 14569, 0, 0, 0, 24767, 0, 0, 0, 0, 0, 0, 0, 25981, 0, 0, 0, 0, 0, 20623, 0, 0, 0, 3041, 0, 0, 0, 0, 0, 18923, 0, 19879, 0, 0, 0, 10757, 0, 0, 0, 0, 0, 8369, 0, 29521, 0, 0, 0, 0, 0, 2833, 0, 0, 0, 0, 0, 19807, 0, 0, 0, 24443, 0, 11719, 0, 0, 0, 23537, 0, 0, 0, 0, 0, 23249, 0, 2071, 0, 0, 0, 0, 0, 26113, 0, 0, 0, 3161, 0, 26497, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30029 };
__constant uchar WHL_STEP[5760] = { 8, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 3, 1, 5, 1, 3, 3, 2, 3, 3, 1, 5, 1, 2, 1, 6, 6, 2, 1, 2, 3, 1, 5, 3, 3, 3, 1, 3, 2, 1, 3, 2, 7, 2, 1, 2, 3, 4, 3, 5, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 4, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 2, 6, 4, 2, 1, 3, 2, 3, 6, 1, 2, 1, 6, 3, 2, 3, 3, 3, 1, 3, 5, 1, 2, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 3, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 4, 2, 3, 4, 5, 1, 5, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 7, 2, 1, 2, 3, 3, 1, 3, 2, 4, 5, 4, 2, 1, 2, 3, 4, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 2, 1, 5, 1, 5, 1, 3, 2, 4, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 2, 1, 3, 2, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 3, 1, 5, 6, 1, 2, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1, 3, 3, 5, 3, 4, 2, 1, 2, 1, 2, 4, 3, 6, 2, 3, 1, 6, 2, 1, 2, 3, 4, 2, 1, 2, 1, 6, 5, 1, 2, 1, 2, 3, 1, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 3, 3, 4, 3, 2, 1, 5, 1, 5, 1, 2, 1, 5, 1, 3, 2, 1, 5, 3, 1, 3, 2, 1, 3, 2, 3, 4, 3, 2, 1, 6, 5, 3, 1, 2, 3, 1, 6, 2, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 2, 6, 3, 4, 3, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 3, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 2, 6, 1, 3, 2, 1, 2, 3, 3, 1, 6, 3, 2, 9, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1, 6, 3, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 3, 1, 5, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 7, 5, 1, 2, 1, 2, 3, 1, 3, 5, 3, 3, 1, 5, 1, 3, 2, 3, 4, 2, 1, 2, 3, 4, 3, 5, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 3, 1, 2, 3, 4, 2, 1, 2, 6, 1, 3, 2, 1, 3, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 3, 1, 5, 1, 3, 5, 3, 3, 1, 3, 2, 1, 2, 1, 5, 1, 6, 2, 3, 3, 1, 6, 2, 3, 3, 1, 3, 2, 1, 3, 2, 7, 2, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 3, 5, 1, 3, 2, 3, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 3, 4, 2, 4, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 1, 2, 3, 7, 2, 1, 3, 5, 4, 2, 1, 2, 1, 2, 4, 5, 4, 2, 4, 3, 5, 1, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 2, 3, 3, 1, 3, 3, 3, 5, 4, 2, 1, 2, 3, 4, 3, 2, 4, 2, 3, 1, 3, 3, 2, 1, 2, 3, 6, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 4, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 4, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 3, 3, 2, 1, 5, 6, 1, 2, 1, 2, 3, 1, 3, 2, 1, 8, 1, 3, 2, 1, 5, 3, 4, 2, 1, 2, 1, 6, 3, 5, 1, 2, 3, 1, 6, 2, 1, 2, 4, 3, 2, 1, 2, 1, 6, 5, 3, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 5, 3, 4, 5, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 4, 2, 1, 2, 1, 5, 6, 1, 2, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 5, 1, 3, 2, 7, 3, 2, 1, 2, 4, 5, 3, 1, 2, 3, 1, 3, 3, 2, 1, 2, 4, 3, 2, 1, 2, 6, 1, 6, 2, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 3, 1, 5, 4, 3, 2, 1, 2, 3, 1, 5, 1, 2, 1, 6, 5, 1, 2, 3, 3, 1, 3, 2, 3, 3, 3, 1, 3, 3, 3, 2, 3, 6, 1, 2, 3, 4, 3, 5, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 1, 3, 2, 3, 4, 2, 3, 6, 1, 3, 2, 1, 3, 2, 3, 6, 1, 2, 1, 2, 7, 2, 3, 1, 2, 3, 1, 3, 5, 1, 5, 1, 3, 2, 1, 2, 6, 1, 5, 1, 2, 3, 3, 1, 3, 3, 2, 3, 3, 1, 5, 1, 3, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 1, 3, 2, 1, 3, 5, 1, 5, 3, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 3, 6, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 2, 1, 5, 1, 6, 3, 2, 3, 1, 5, 1, 2, 3, 3, 4, 2, 1, 3, 9, 2, 1, 2, 1, 2, 4, 5, 3, 1, 2, 4, 3, 3, 3, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 3, 3, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 6, 3, 1, 3, 3, 2, 1, 2, 3, 4, 3, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 5, 3, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 5, 3, 1, 3, 2, 1, 3, 5, 4, 2, 1, 2, 7, 3, 2, 3, 1, 2, 3, 1, 6, 2, 1, 2, 4, 5, 1, 2, 1, 5, 1, 5, 3, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 5, 3, 4, 3, 3, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 2, 1, 5, 1, 3, 2, 1, 2, 6, 1, 5, 1, 5, 7, 2, 1, 2, 1, 2, 4, 3, 5, 1, 2, 3, 1, 6, 2, 1, 2, 3, 1, 3, 2, 1, 2, 7, 6, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 2, 7, 2, 3, 1, 5, 1, 3, 3, 2, 1, 2, 3, 6, 1, 2, 1, 6, 5, 1, 2, 1, 5, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 3, 2, 3, 4, 8, 1, 2, 3, 1, 3, 3, 2, 1, 2, 4, 3, 2, 1, 3, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 8, 1, 3, 2, 4, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 3, 2, 3, 3, 1, 3, 3, 3, 2, 3, 6, 1, 3, 2, 4, 3, 2, 3, 1, 2, 7, 3, 2, 1, 5, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 3, 2, 4, 3, 2, 3, 3, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 3, 3, 3, 1, 3, 3, 2, 1, 5, 1, 3, 2, 1, 2, 6, 1, 5, 1, 3, 2, 3, 1, 3, 3, 2, 3, 3, 6, 1, 3, 5, 4, 2, 1, 2, 1, 2, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 4, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 4, 5, 1, 3, 3, 2, 3, 3, 4, 2, 1, 2, 1, 5, 1, 6, 2, 1, 2, 3, 1, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 7, 2, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 5, 3, 1, 3, 5, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 5, 3, 4, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 3, 3, 3, 1, 6, 2, 1, 2, 4, 3, 3, 2, 1, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 2, 3, 4, 3, 2, 1, 5, 3, 4, 3, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 6, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 2, 1, 5, 4, 2, 1, 2, 3, 3, 1, 5, 1, 3, 9, 2, 1, 2, 3, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 2, 6, 1, 6, 2, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 2, 7, 2, 3, 1, 5, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 2, 1, 11, 1, 2, 1, 2, 3, 1, 3, 2, 3, 6, 1, 3, 2, 1, 5, 3, 4, 2, 1, 2, 3, 4, 3, 5, 1, 2, 3, 1, 6, 2, 1, 2, 3, 1, 3, 2, 1, 3, 6, 5, 1, 2, 1, 2, 3, 4, 2, 1, 2, 6, 1, 3, 2, 1, 3, 2, 3, 6, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 5, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 2, 6, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 3, 3, 2, 4, 5, 3, 1, 2, 3, 4, 3, 2, 1, 6, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 5, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 3, 3, 3, 4, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 5, 3, 1, 3, 2, 1, 2, 3, 3, 4, 3, 3, 5, 6, 1, 2, 1, 2, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 3, 3, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 4, 2, 3, 1, 3, 3, 2, 1, 5, 4, 2, 1, 2, 6, 1, 5, 1, 2, 1, 2, 3, 1, 6, 2, 3, 4, 5, 1, 3, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 3, 3, 3, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 3, 1, 5, 1, 6, 2, 3, 4, 3, 2, 1, 2, 1, 5, 1, 8, 1, 2, 3, 1, 5, 1, 2, 3, 3, 1, 3, 2, 1, 5, 7, 3, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 3, 1, 5, 6, 1, 2, 1, 5, 1, 3, 2, 1, 2, 3, 3, 6, 1, 3, 2, 7, 2, 1, 2, 1, 6, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 3, 2, 6, 1, 6, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 4, 5, 1, 3, 5, 1, 2, 3, 1, 5, 1, 2, 1, 6, 5, 1, 2, 1, 2, 3, 4, 2, 3, 3, 3, 1, 3, 2, 1, 3, 5, 4, 2, 1, 2, 3, 4, 3, 5, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 3, 5, 1, 5, 1, 2, 1, 2, 7, 2, 1, 2, 6, 1, 3, 2, 1, 3, 2, 3, 6, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 2, 3, 3, 1, 3, 3, 2, 6, 1, 3, 2, 1, 5, 3, 4, 2, 1, 3, 2, 4, 3, 5, 1, 2, 3, 7, 2, 1, 5, 1, 3, 2, 1, 2, 1, 6, 5, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 2, 4, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 3, 5, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 3, 2, 1, 2, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 4, 3, 2, 3, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 6, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 4, 3, 2, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 5, 1, 2, 3, 4, 3, 3, 3, 2, 3, 6, 3, 1, 2, 4, 3, 2, 3, 1, 2, 4, 3, 3, 2, 3, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 6, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 9, 2, 3, 1, 2, 3, 1, 6, 2, 1, 6, 3, 2, 1, 2, 6, 1, 5, 3, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 5, 1, 5, 3, 4, 3, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 1, 3, 5, 6, 3, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 5, 1, 3, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 3, 1, 5, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 2, 6, 1, 6, 2, 1, 2, 3, 1, 5, 1, 2, 3, 3, 1, 3, 2, 1, 3, 2, 7, 2, 1, 2, 1, 2, 7, 2, 3, 1, 5, 1, 3, 3, 3, 2, 3, 1, 5, 3, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 3, 3, 3, 4, 2, 1, 3, 2, 3, 4, 2, 1, 2, 7, 3, 5, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 3, 3, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 2, 7, 3, 2, 1, 3, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 3, 3, 4, 3, 2, 3, 3, 1, 3, 2, 1, 3, 5, 4, 2, 1, 5, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 1, 5, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 3, 2, 3, 1, 3, 2, 1, 2, 6, 4, 2, 1, 8, 4, 2, 1, 2, 1, 2, 4, 8, 1, 2, 4, 6, 2, 1, 2, 3, 1, 3, 2, 3, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 5, 1, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 5, 1, 2, 4, 5, 3, 1, 2, 3, 1, 3, 3, 2, 3, 4, 3, 3, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 3, 1, 2, 3, 7, 2, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 3, 3, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 5, 3, 7, 2, 1, 2, 4, 3, 2, 3, 1, 2, 4, 3, 3, 3, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 3, 5, 1, 3, 2, 3, 3, 3, 1, 5, 1, 3, 2, 7, 2, 1, 2, 1, 2, 7, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 5, 1, 3, 2, 1, 2, 6, 1, 6, 2, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 5, 1, 3, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 3, 1, 5, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 3, 6, 5, 3, 1, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 3, 4, 3, 5, 1, 5, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 3, 5, 1, 6, 2, 1, 2, 3, 6, 1, 2, 6, 1, 3, 2, 1, 3, 2, 9, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 6, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 3, 2, 3, 3, 4, 2, 1, 3, 2, 3, 4, 2, 1, 3, 6, 3, 2, 3, 3, 3, 4, 3, 2, 1, 5, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 4, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 6, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 3, 2, 3, 4, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 2, 1, 2, 3, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 5, 3, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 4, 2, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 2, 1, 5, 6, 1, 2, 1, 2, 3, 1, 5, 1, 2, 7, 3, 2, 1, 5, 3, 4, 2, 3, 1, 2, 4, 3, 5, 1, 2, 3, 1, 6, 2, 3, 3, 1, 3, 3, 2, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 6, 3, 2, 3, 1, 2, 3, 1, 6, 2, 1, 2, 7, 2, 1, 2, 1, 5, 1, 5, 3, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 5, 3, 4, 3, 2, 1, 2, 4, 5, 3, 1, 2, 3, 1, 3, 3, 3, 2, 4, 3, 2, 1, 2, 1, 5, 6, 1, 2, 1, 5, 1, 3, 2, 1, 5, 3, 1, 5, 4, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1, 6, 2, 3, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 6, 1, 2, 1, 2, 7, 2, 3, 1, 6, 3, 3, 2, 1, 2, 3, 1, 5, 1, 2, 1, 6, 5, 1, 3, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 3, 7, 5, 1, 2, 3, 1, 3, 3, 2, 1, 5, 1, 3, 2, 1, 8, 1, 5, 1, 2, 1, 2, 3, 4, 3, 2, 6, 1, 5, 1, 3, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 3, 3, 3, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 3, 1, 5, 4, 3, 2, 6, 1, 3, 2, 1, 2, 1, 5, 1, 6, 2, 1, 2, 4, 5, 1, 2, 3, 3, 1, 3, 2, 4, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 3, 3, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 2, 1, 2, 1, 6, 5, 3, 3, 4, 3, 3, 2, 1, 2, 3, 1, 3, 5, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 4, 3, 3, 3, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 5, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 6, 1, 2, 3, 4, 3, 2, 1, 3, 5, 4, 2, 3, 3, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 6, 2, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 5, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 6, 3, 2, 3, 1, 2, 3, 1, 6, 2, 1, 2, 4, 3, 2, 1, 2, 1, 5, 6, 3, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1, 3, 2, 1, 5, 3, 4, 3, 2, 1, 2, 4, 3, 5, 1, 2, 3, 1, 6, 3, 2, 3, 1, 3, 2, 1, 2, 1, 11, 1, 2, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 5, 1, 3, 2, 7, 2, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 2, 6, 1, 6, 2, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 3, 2, 1, 2, 9, 3, 1, 5, 1, 3, 3, 2, 1, 2, 4, 5, 1, 2, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 6, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 3, 4, 3, 5, 1, 2, 3, 4, 3, 2, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 1, 2, 3, 3, 4, 2, 1, 2, 6, 1, 3, 3, 3, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 4, 3, 5, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 5, 3, 1, 3, 5, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 2, 7, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 1, 3, 2, 1, 2, 6, 1, 5, 1, 2, 1, 2, 4, 3, 3, 2, 3, 3, 1, 5, 4, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 4, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 2, 1, 2, 1, 2, 4, 5, 3, 1, 6, 3, 3, 2, 3, 3, 1, 3, 2, 3, 1, 5, 1, 6, 2, 1, 2, 3, 1, 5, 1, 2, 3, 3, 1, 3, 3, 3, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 3, 3, 2, 3, 4, 2, 3, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 5, 1, 2, 3, 7, 2, 1, 3, 2, 3, 4, 2, 3, 1, 6, 3, 2, 3, 3, 3, 1, 3, 3, 2, 3, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 5, 4, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 4, 2, 3, 1, 8, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 3, 1, 2, 3, 4, 2, 1, 2, 3, 3, 1, 3, 2, 1, 8, 4, 3, 2, 3, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 5, 1, 2, 1, 5, 6, 1, 2, 1, 6, 3, 2, 1, 2, 3, 3, 1, 5, 1, 3, 2, 7, 2, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 6, 7, 2, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1, 3, 2, 1, 5, 3, 4, 2, 1, 2, 1, 2, 7, 5, 1, 5, 1, 6, 2, 1, 2, 3, 1, 5, 1, 2, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 3, 3, 4, 3, 5, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 3, 5, 1, 5, 1, 2, 1, 5, 4, 2, 1, 2, 6, 1, 3, 2, 1, 3, 2, 3, 7, 2, 1, 2, 4, 5, 3, 1, 2, 3, 1, 3, 5, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 3, 5, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 3, 4, 3, 2, 1, 2, 3, 3, 1, 3, 6, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 3, 3, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 3, 3, 3, 4, 2, 1, 3, 5, 4, 2, 1, 2, 1, 2, 9, 3, 1, 2, 4, 3, 3, 2, 1, 5, 1, 3, 2, 3, 6, 1, 5, 1, 2, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 6, 3, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 3, 5, 1, 5, 3, 1, 2, 3, 1, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 3, 1, 5, 1, 3, 3, 2, 3, 3, 1, 3, 3, 2, 1, 5, 1, 6, 2, 1, 2, 3, 1, 5, 1, 5, 3, 1, 3, 2, 1, 3, 2, 7, 2, 1, 2, 1, 6, 3, 2, 3, 1, 2, 3, 1, 6, 3, 2, 4, 3, 2, 3, 1, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 5, 3, 4, 3, 2, 1, 6, 3, 2, 3, 3, 3, 1, 3, 3, 3, 2, 3, 1, 3, 3, 2, 1, 5, 6, 1, 2, 1, 5, 1, 3, 2, 1, 2, 3, 4, 5, 1, 3, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1, 6, 2, 1, 2, 3, 4, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 5, 4, 2, 1, 2, 3, 7, 2, 3, 1, 5, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 2, 1, 6, 5, 1, 2, 1, 2, 4, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 5, 4, 3, 5, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 3, 5, 6, 1, 2, 1, 2, 3, 4, 2, 1, 2, 6, 1, 3, 2, 1, 5, 3, 6, 1, 2, 1, 2, 4, 3, 5, 1, 2, 3, 1, 8, 1, 2, 3, 1, 3, 2, 1, 2, 1, 6, 5, 1, 2, 3, 3, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 4, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 4, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 6, 3, 2, 1, 2, 3, 3, 1, 3, 2, 4, 2, 3, 4, 3, 2, 1, 2, 4, 5, 3, 3, 3, 1, 3, 3, 2, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 1, 5, 3, 4, 2, 4, 5, 4, 2, 1, 2, 1, 2, 4, 5, 3, 1, 2, 7, 3, 2, 1, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 4, 2, 4, 3, 3, 2, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 3, 2, 3, 1, 5, 3, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 2, 7, 2, 3, 1, 2, 3, 1, 3, 3, 2, 6, 1, 3, 3, 2, 6, 1, 5, 1, 2, 1, 2, 3, 1, 3, 3, 5, 3, 1, 5, 1, 3, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 3, 1, 2, 3, 1, 6, 2, 1, 2, 4, 3, 2, 1, 3, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 5, 3, 4, 3, 2, 1, 2, 4, 3, 2, 3, 1, 5, 1, 3, 3, 5, 3, 1, 3, 2, 1, 2, 1, 5, 7, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 5, 1, 3, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 3, 6, 1, 6, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 9, 2, 3, 6, 1, 3, 3, 2, 1, 2, 3, 1, 6, 2, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 3, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 3, 4, 3, 6, 2, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 2, 6, 1, 3, 2, 1, 3, 5, 6, 1, 2, 3, 4, 3, 2, 3, 1, 2, 3, 1, 3, 5, 1, 2, 3, 1, 5, 1, 2, 1, 5, 1, 5, 1, 2, 3, 4, 3, 3, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 1, 3, 2, 1, 2, 1, 5, 6, 1, 2, 1, 2, 4, 3, 2, 1, 2, 6, 1, 3, 2, 6, 3, 4, 2, 1, 2, 1, 2, 4, 3, 5, 3, 3, 1, 6, 2, 1, 2, 3, 1, 3, 2, 1, 2, 1, 6, 5, 1, 3, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 2, 3, 1, 2, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 4, 2, 3, 1, 5, 1, 5, 1, 2, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 4, 3, 2, 1, 2, 4, 5, 4, 2, 3, 1, 3, 3, 2, 1, 2, 7, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 5, 1, 5, 4, 3, 2, 4, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 3, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 2, 1, 5, 3, 1, 3, 3, 3, 2, 3, 6, 1, 2, 1, 6, 3, 2, 3, 1, 2, 4, 6, 2, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 4, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 5, 3, 4, 3, 2, 1, 2, 7, 2, 3, 1, 2, 3, 1, 3, 3, 3, 5, 1, 3, 2, 1, 2, 6, 6, 1, 2, 1, 5, 1, 3, 3, 2, 3, 3, 1, 5, 1, 3, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 8, 1 };
#endif
#define W30_STEPS 0x13212123u   // steps/p for cycle positions 0..7 (low nibble first)
__constant uchar W30_IDX[30] = {
    0xFF, 0,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1,    0xFF, 0xFF,
    0xFF, 2,    0xFF, 3,    0xFF, 0xFF, 0xFF, 4,    0xFF, 5,
    0xFF, 0xFF, 0xFF, 6,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 7 };
__constant uchar W30_INV[30] = {
    0, 1,  0, 0, 0, 0, 0, 13, 0, 0,
    0, 11, 0, 7, 0, 0, 0, 23, 0, 19,
    0, 0,  0, 17, 0, 0, 0, 0, 0, 29 };

// Marks the multiples p*m, m coprime to 30, from bit0 (= multiplier m0,
// the first odd multiple: first = m0*p) up to q_odds. qf30 = q_first mod 30.
// Requires 5 < p < 2^30 (steps <= 3p < 2^32); callers fall back otherwise.
/// @param bit0  map bit of the first odd multiple first = m0 p.
/// @param qf30  q_first mod 30.
/// @pre    5 < p < 2^30; the small-prime stage clears every multiple of 3, 5.
/// @post   bits of p m, m coprime to 30, from bit0 up to q_odds are cleared.
inline void mark_multiples_w30(volatile __global u64* prime_map, const u64 bit0,
                               const u32 p, const u64 q_odds, const u32 qf30, const cmap g)
{
    volatile __global u32* m32 = (volatile __global u32*)prime_map;
    const u32 n = (u32)q_odds;
    u32 bit = (u32)bit0;
    if (bit >= n) return;
    // first = q_first + 2*bit  =>  first mod 30 = qf30 + 2*(bit mod 15)
    u32 m = (qf30 + 2u * (bit % 15u)) % 30u;          // first mod 30
    m = (m * W30_INV[p % 30u]) % 30u;                 // m0 mod 30 (m0 odd)
    while (W30_IDX[m] == 0xFFu) {                     // <= 2 steps to a coprime m
        if (n - bit <= p) return;
        bit += p;
        m += 2u; if (m >= 30u) m -= 30u;
    }
    u32 w = W30_IDX[m];
    for (;;) {
        map_clear(m32, bit, g);
        const u32 step = p * ((W30_STEPS >> (4u * w)) & 0xFu);
        if (n - bit <= step) break;
        bit += step;
        w = (w + 1u) & 7u;
    }
}


// Wheel-30 marks in an SLM tile (see mark_multiples_w30): clears p*m for m
// coprime to 30 from tile-local bit `bit` (global bit gbit = tile_bit0 +
// bit, the first odd multiple) up to `end`, with SLM atomics.
/// @param bit   tile-local bit of the first odd multiple.
/// @param gbit  its segment-global bit (for the residue); qf30 = q_first mod 30.
/// @pre    5 < p < 2^30; concurrent callers (SLM atomics).
/// @post   tile bits of p m, m coprime to 30, in [bit, end) are cleared.
inline void mark_local_w30(volatile __local u32* t32, u32 bit, const u32 end,
                           const u32 p, const u32 gbit, const u32 qf30)
{
    if (bit >= end) return;
    u32 m = (qf30 + 2u * (gbit % 15u)) % 30u;
    m = (m * W30_INV[p % 30u]) % 30u;
    while (W30_IDX[m] == 0xFFu) {
        if (end - bit <= p) return;
        bit += p;
        m += 2u; if (m >= 30u) m -= 30u;
    }
    u32 w = W30_IDX[m];
    for (;;) {
        atomic_and(&t32[bit >> 5], CLR_MASK(bit));
        const u32 step = p * ((W30_STEPS >> (4u * w)) & 0xFu);
        if (end - bit <= step) break;
        bit += step;
        w = (w + 1u) & 7u;
    }
}


#ifndef OH_F32
#define OH_F32 1                    // one-hit (q_first + 2b) mod 15015 by a float quotient (+0.07%)
#endif
#ifndef LARGE_OH_BM
#define LARGE_OH_BM 1               // one-hit primes: 15015-bit coprimality bitmap (+1.45%)
#endif
#if LARGE_OH_BM
// One-hit primes (p >= q_odds): the only odd multiple first = p m0 in the
// segment is marked iff m0 is coprime to 30030; p is coprime to 30030 and
// first is odd, so that is first coprime to 15015 — one bit of this table
// (bit x <-> gcd(x, 15015) == 1; generated) instead of the WHL_INV / WHL_IDX
// gathers and three modulos of mark_multiples_whl.
__constant uint CP15015[470] = { 0xA08B0116u, 0x6C20CA65u, 0x524C9699u, 0x94522CA2u, 0x61B44B49u, 0x912C30D8u, 0x845844A6u, 0x4912932Cu, 0x1A65B0C3u, 0x36894532u, 0x2D025948u, 0xC92186D2u, 0x121264B4u, 0x48B29168u, 0xC10DA652u, 0x30C92984u, 0x6502DA05u, 0x0B4C3499u, 0x92932D06u, 0x05B44A61u, 0x196930CAu, 0xA6590826u, 0x6894D324u, 0x52618489u, 0xB4184D32u, 0x29A60B04u, 0xCB2916C0u, 0x30CA6504u, 0x4494992Cu, 0x5209A253u, 0xA4C34892u, 0x2D22D861u, 0x5B44A291u, 0x961320A4u, 0x2530C369u, 0x884D125Au, 0x24590896u, 0x6186432Du, 0x9264B48Bu, 0xA2914D12u, 0x0DA25A4Cu, 0x816196C1u, 0x22CA2434u, 0x4C941961u, 0xD32D841Au, 0x944A6990u, 0x6832CA41u, 0x190C3681u, 0x90D325A6u, 0x65908B48u, 0x086D2292u, 0x86434C36u, 0x2916D028u, 0x5A65A44Bu, 0x96992830u, 0x2D82524Cu, 0xCB490650u, 0x329841B0u, 0x44A69925u, 0x932C204Bu, 0xB0C06986u, 0x4C305225u, 0x5948B609u, 0x84D32C26u, 0x64A4C921u, 0x914D129Au, 0xA60A4432u, 0x6116830Du, 0xDA0514C9u, 0xA4992522u, 0x2986194Cu, 0x426892D3u, 0x224841B4u, 0x0C309969u, 0x9225A65Bu, 0x148B6894u, 0x6D12C265u, 0x4B04B618u, 0x125109A4u, 0x64A4CB29u, 0x916C30D0u, 0xA2534C86u, 0x41965209u, 0xC821B48Bu, 0x86982D32u, 0x2C245A44u, 0x03699483u, 0x325A61B0u, 0x48B2814Cu, 0xD12D2659u, 0xB4C34182u, 0x69129A64u, 0x5A4C3201u, 0x96C20D84u, 0x2514C829u, 0x8961225Au, 0x86134CB4u, 0x4982D125u, 0xCA45B04Au, 0x36996122u, 0x2426430Cu, 0x0A6894D3u, 0x30D22594u, 0x48B6182Du, 0xD128864Au, 0xA4C92916u, 0x44309A65u, 0x13449699u, 0x06422DA2u, 0x6194C949u, 0x992C32D0u, 0xA45B40A4u, 0x28949328u, 0x5865B0C3u, 0x32894D02u, 0x25264948u, 0xCB618693u, 0x129A4434u, 0x0CA2906Du, 0x430DA458u, 0x34C96896u, 0x65225220u, 0x1B0CA499u, 0x92D20D82u, 0x45340A61u, 0x996932CAu, 0xA65A0416u, 0x6890D305u, 0xD061840Bu, 0xB6106C32u, 0x29A6494Cu, 0xCB091251u, 0x309A25A4u, 0x4C16886Cu, 0x422D0253u, 0xB48B0996u, 0x09325861u, 0x5344A299u, 0x86912CA4u, 0x44B0C369u, 0x0941325Au, 0x244948B6u, 0x6186D32Du, 0x9A2494CAu, 0xB2016D10u, 0x04A61A48u, 0xC94996C3u, 0x22DA2520u, 0x44B49945u, 0xC32C8613u, 0x944A6912u, 0x2930CA45u, 0x5A0C3499u, 0x14D32186u, 0x65948368u, 0x18652290u, 0xA24B4CB2u, 0x29069121u, 0xD26524C9u, 0x96986C10u, 0x2DA05308u, 0xCB099452u, 0x12D060B4u, 0x4426992Du, 0x930CA04Bu, 0xB0836196u, 0x4D324A45u, 0x58488689u, 0x84D32D26u, 0x60B44B60u, 0x916C121Au, 0xA6580CB0u, 0x4992C20Du, 0x8A2534C9u, 0x34896522u, 0x0D861B44u, 0x4A2992D2u, 0x324845A4u, 0x0C369169u, 0x5121A653u, 0x908B6884u, 0x6532D205u, 0x4B4C9618u, 0x16C12926u, 0x25A48A29u, 0x194C30DAu, 0xA2534892u, 0x4996502Cu, 0xD860A4C1u, 0xA6190932u, 0x2CA41B44u, 0xC2691683u, 0x325A6590u, 0x48B4890Du, 0xD3212619u, 0xB0C34086u, 0x6D029864u, 0x524CB291u, 0x968304A6u, 0x25344969u, 0x982500DAu, 0x841A0CB4u, 0x6912532Du, 0x8244B44Au, 0x26996132u, 0x05A25B0Cu, 0x896084D3u, 0x32C26594u, 0x4896186Cu, 0xD129A64Au, 0xA44B2914u, 0x6C10DA61u, 0x414C1691u, 0x92122DA2u, 0x41B4CB41u, 0x892D3298u, 0x84594426u, 0x2896922Cu, 0x5A61B0C3u, 0xB4894932u, 0x2D265108u, 0xCB6186D0u, 0x128A4430u, 0x44B29165u, 0xC30D264Au, 0x24C86992u, 0x6520D825u, 0x1B48B411u, 0x92532886u, 0x05A44869u, 0x984932CAu, 0x261B0416u, 0x6814C325u, 0xD265948Bu, 0xB0180D32u, 0x29A64B4Cu, 0xC32816D1u, 0x20D864A4u, 0x4C921968u, 0x1225A053u, 0x14CA4996u, 0x2D12D861u, 0x1B04A688u, 0x961324A4u, 0x64B0C349u, 0x894D2252u, 0x225148A6u, 0x6186D30Cu, 0x8A64B40Bu, 0x92912D10u, 0x0D865A4Cu, 0x496914C3u, 0x229A2134u, 0x4CB49164u, 0xD32D8219u, 0xB4484982u, 0x68328245u, 0x5B083689u, 0x94D22586u, 0x65948B28u, 0x18691252u, 0xA6434C36u, 0x21069129u, 0xDA45A0CBu, 0x86996430u, 0x2D22414Cu, 0xCA489652u, 0x305821B4u, 0x00A4192Du, 0x922CA45Bu, 0x20C16996u, 0x4D324A65u, 0x1940B689u, 0x02C30D26u, 0x6484CB61u, 0x912D1298u, 0xA65A4CB0u, 0x68964309u, 0xD82534C1u, 0x90986522u, 0x25061B4Cu, 0x0A699283u, 0x12CA4534u, 0x0C329849u, 0x5125A45Bu, 0x94836894u, 0x6932D264u, 0x4B4CA618u, 0x16D109A0u, 0x65848A21u, 0x896C30CAu, 0xA2124C96u, 0x49945225u, 0xD861B04Bu, 0xA6912C22u, 0x2CA45144u, 0x43499293u, 0x301A65B0u, 0x4836890Du, 0xD22D0658u, 0xB4CB2106u, 0x61121A64u, 0x524CB291u, 0x86C10DA2u, 0x0530C969u, 0x996422D2u, 0x060B48B4u, 0x2990D32Du, 0xCA05B44Au, 0x36196910u, 0x24A64B08u, 0x89609493u, 0x32D24584u, 0x44A6184Du, 0xC129A609u, 0x84CB2816u, 0x2C305A61u, 0x530C9499u, 0x965229A2u, 0x6134C348u, 0x992D22C8u, 0xA45B44A2u, 0x69929304u, 0x586520C3u, 0xB6804D12u, 0x29245908u, 0xCB618653u, 0x129224B4u, 0x4C82906Du, 0xC30D225Au, 0x34896196u, 0x4522CA25u, 0x1A4C9099u, 0x90D12D86u, 0x40B44269u, 0x116832CAu, 0xA4590C36u, 0x4890D325u, 0x9265948Au, 0x36086D32u, 0x21860B4Cu, 0xCB0916D0u, 0x205A65A0u, 0x4C969968u, 0x502CA253u, 0xB04B4986u, 0x2530D841u, 0x4A44A699u, 0x16932C04u, 0x25B0C269u, 0x0945301Au, 0x225948B6u, 0x6186D32Cu, 0x9264A4C9u, 0xB2914C12u, 0x0DA61A40u, 0xC92914C3u, 0x02DA2514u, 0x4C349925u, 0x9329860Bu, 0xB4426092u, 0x6922C845u, 0x590C2299u, 0x949325A6u, 0x61148B68u, 0x186D1252u, 0xA44B0CB4u, 0x29165029u, 0xC26424CBu, 0x86996C30u, 0x0DA25344u, 0x8B419252u, 0x32C861A4u, 0x4486912Du, 0x1328A45Au, 0xB0436994u, 0x4C325A21u, 0x59489680u, 0x82C32D26u, 0x64B48B41u, 0x814D129Au, 0x865A4C32u, 0x2996C00Du, 0x5A2434C1u, 0xB4196122u, 0x2D84134Cu, 0x4A6982D1u, 0x32CA4590u, 0x0C368961u, 0xD325260Bu, 0x908A4894u, 0x6D20D225u, 0x4348B618u, 0x16D128A6u, 0x65A44929u, 0x990C30DAu, 0x82124496u, 0x4916422Du, 0x986194CBu, 0xA4992532u, 0x28A05B44u, 0xC1688693u, 0x225065B0u, 0x48B2894Cu, 0x93252659u, 0x34CB2184u, 0x6D129A64u, 0x4A0C3290u, 0x96030DA4u, 0x0434C961u, 0x996522D2u, 0x82194CA4u, 0x6092D30Du, 0x4A41B40Au, 0x14996932u, 0x25A65A0Cu, 0x0B6894D2u, 0x32C26114u, 0x44B6106Cu, 0xD109A649u, 0xA4CB0912u, 0x6C309865u, 0x534C1681u, 0x96522982u, 0x21B4CB09u, 0x98293258u, 0x24534486u, 0x6986812Cu, 0x5A45B083u, 0xB2894532u, 0x2D264948u, 0xC26186D1u, 0x109A24B4u, 0x48B21169u, 0xC30CA45Au, 0x04C86996u, 0x4522DA25u, 0x1B44B489u, 0x12C32586u, 0x45904A49u, 0x992922CAu, 0xA6530C34u, 0x6894D320u, 0xD0659403u, 0xB2182D20u, 0x21864A4Cu, 0xCB291691u, 0x109A6524u, 0x0C969864u, 0x522DA053u, 0xB4C94986u, 0x2C32D060u, 0x5B40A699u, 0x94930CA0u, 0x65B08321u, 0x894D124Au, 0x26484816u, 0x6184932Du, 0x9A40B44Bu, 0xA2916C12u, 0x0DA6584Cu, 0xC94892C3u, 0x221A2134u, 0x0C348965u, 0xD22D861Bu, 0x344A2992u, 0x69324A45u, 0x53043699u, 0x80D105A6u, 0x00000068u };
#endif


// Generic multiplier wheel (WHEEL_W = 30 / 210 / 2310): same contract as
// mark_multiples_w30 with the step table in constant memory. Requires the
// small-prime stage to clear EVERY multiple of each prime dividing WHEEL_W
// (all are < 64: mask groups) and p > the largest of them.
/// @param qfw  q_first mod WHEEL_W.
/// @pre    p does not divide WHEEL_W, 13 < p < 2^30; q_odds < 2^32.
/// @post   bits of p m, m coprime to WHEEL_W, from bit0 up to q_odds are cleared.
inline void mark_multiples_whl(volatile __global u64* prime_map, const u64 bit0,
                               const u32 p, const u64 q_odds, const u32 qfw, const cmap g)
{
    volatile __global u32* m32 = (volatile __global u32*)prime_map;
    const u32 n = (u32)q_odds;
    u32 bit = (u32)bit0;
    if (bit >= n) return;
    u32 m = (qfw + 2u * (bit % (WHEEL_W / 2))) % WHEEL_W;   // first mod W
    m = (m * WHL_INV[p % WHEEL_W]) % WHEEL_W;              // m0 mod W
    while (WHL_IDX[m] == WHL_MISS) {
        if (n - bit <= p) return;
        bit += p;
        m += 2u; if (m >= WHEEL_W) m -= WHEEL_W;
    }
#ifndef LARGE_ONEHIT
#define LARGE_ONEHIT 1           // p >= q_odds: at most one hit, no wheel step (-0.15%)
#endif
#if LARGE_ONEHIT
    if (p >= n) { map_clear_w(m32, bit, g); return; }   // bit < p: no second hit
#endif
    u32 w = WHL_IDX[m];
#ifdef ABL_FIRSTONLY   // timing only (INVALID map): first mark, no loop
    map_clear_w(m32, bit, g); return;
#endif
    for (;;) {
#ifdef ABL_NOATOM      // timing only (INVALID map): address math, no atomic
        { u32 ad = bit; const u32 t = bit / 15u; ad = c30_class(bit - 15u * t, g) * g.off5 + t;
          if (ad == 0xFFFFFFFFu) m32[0] = 0u; }
#else
        map_clear_w(m32, bit, g);
#endif
        // 64-bit step: up to 7p (WHEEL_W 2310) exceeds 2^32 for p > ~613M.
        // (A u32 step wrapped and placed stray marks after the first hit —
        // false composites; caught by GB_BUCKET_VERIFY with all primes
        // bucketed, since the verify reference is this kernel.)
        const u64 step = (u64)p * WHL_STEP[w];
        if ((u64)(n - bit) <= step) break;
        bit += (u32)step;
        if (++w == WHL_N) w = 0u;
    }
}


// One medium prime of make_prime_map stage 3m: first mark in this tile from
// the segment-global first bit fb (~0 = none), then all marks with SLM
// atomics (wheel-30 below TILE_WHEEL_MAXP).
/// @param fb         segment-global first bit (~0 = no mark in the segment).
/// @param tile_bit0  global bit of the tile start; qf30 = q_first mod 30.
/// @pre    fb, tile_bit0 < 2^32; concurrent callers (SLM atomics).
/// @post   every mark of p inside this tile is cleared.
inline void med_prime(volatile __local u32* t32, const u64 fb, const u32 p,
                      const u64 tile_bit0, const u32 qf30)
{
    if (fb == ~0UL) return;
    u32 bit;
    if (fb >= tile_bit0) {
        if (fb - tile_bit0 >= TILE_BITS) return;
        bit = (u32)(fb - tile_bit0);
    } else {
        // d mod p without the emulated 32-bit '%': float quotient
        // (error <= 1 for d < 2^27-ish, any d exact via the fix-ups)
        // and the remainder in wrap-around 32-bit arithmetic, exact
        // because the true d - q*p lies in (-p, 2p) and p < 2^30
        // (medium primes are below the split). Setup runs per prime
        // per TILE, so this is the medium stage's per-tile cost.
        const u32 d = (u32)(tile_bit0 - fb);
        u32 rmu;
        if (p < (1u << 30)) {
            const u32 q = (u32)((float)d * (1.0f / (float)p));
            int rm = (int)(d - q * p);
            while (rm < 0) rm += (int)p;
            while (rm >= (int)p) rm -= (int)p;
            rmu = (u32)rm;
        } else {
            rmu = d % p;                 // huge --split only
        }
        bit = rmu ? p - rmu : 0u;
    }
#if TILE_WHEEL30
    if (p > 61u && p < TILE_WHEEL_MAXP) {       // many marks per tile only
        // wheel-30 (nibble steps): the WHEEL_W constant tables
        // measured slower here (sieve_s 0.085 vs 0.081)
        mark_local_w30(t32, bit, (u32)TILE_BITS, p, (u32)tile_bit0 + bit, qf30);
        return;
    }
#endif
    for (; bit < (u32)TILE_BITS; bit += p)
        atomic_and(&t32[bit >> 5], CLR_MASK(bit));
}

// med_prime for the class map (class mode): same first-mark math, no wheel —
// the class arrays hold no multiples of 3 already, and a mod-5 skip measured
// slower than the marks it saves; a separate loop also keeps the wheel code
// out of the 8x-unrolled batch (sieve_s per 1e11: 262.0 -> 252.9 ms).
/// @pre    as med_prime; fb and tile_bit0 are class-array coordinates.
/// @post   every mark of p inside this class tile is cleared.
inline void med_prime_nw(volatile __local u32* t32, const u64 fb, const u32 p,
                         const u64 tile_bit0)
{
    if (fb == ~0UL) return;
    u32 bit;
    if (fb >= tile_bit0) {
        if (fb - tile_bit0 >= TILE_BITS) return;
        bit = (u32)(fb - tile_bit0);
    } else {
        const u32 d = (u32)(tile_bit0 - fb);
        u32 rmu;
        if (p < (1u << 30)) {
            const u32 q = (u32)((float)d * (1.0f / (float)p));
            int rm = (int)(d - q * p);
            while (rm < 0) rm += (int)p;
            while (rm >= (int)p) rm -= (int)p;
            rmu = (u32)rm;
        } else {
            rmu = d % p;
        }
        bit = rmu ? p - rmu : 0u;
    }
    for (; bit < (u32)TILE_BITS; bit += p)
        atomic_and(&t32[bit >> 5], CLR_MASK(bit));
}

#ifndef MED_F32
#define MED_F32 1               // 32-bit / float medium-stage setup (class map): +0.75%
#endif
// med_prime_nw in 32-bit / float arithmetic: fb is the low word of fb_global
// (class coordinates < 2^32, so 0xffffffff is only the ~0 sentinel) and the
// tile offset d = tile_bit0 - fb < 2^24 (caller's uniform guard), so
// df = (float)d is exact, the truncated quotient q is within 1 of d / p
// (relative error ~2^-22, d / p < 2^24 / 64), so d - q p lies in (-p, 2p);
// with p < 2^23 that integer is below 2^24 in magnitude, so fma(-q, p, d)
// returns it exactly and two selects bring it into [0, p). (p < 2^24 alone
// would not do: d - q p can reach p + small > 2^24.) No 64-bit compare or
// subtract, no integer multiply, no divergent fix-up loops.
/// @pre    61 < p < 2^23; tile_bit0 < 2^24; fb < 2^32 or the sentinel.
/// @post   as med_prime_nw.
inline void med_prime_f32(volatile __local u32* t32, const u32 fb, const u32 p,
                          const u32 tile_bit0)
{
    if (fb == 0xFFFFFFFFu) return;
    u32 bit;
    if (fb >= tile_bit0) {
        bit = fb - tile_bit0;
    } else {
        const float df = (float)(tile_bit0 - fb);
        const float pf = (float)p;
        const float qf = trunc(df * (1.0f / pf));
        float rf = fma(-qf, pf, df);
        rf += (rf < 0.0f) ? pf : 0.0f;
        rf -= (rf >= pf) ? pf : 0.0f;
        const u32 rmu = (u32)rf;
        bit = rmu ? p - rmu : 0u;
    }
    for (; bit < (u32)TILE_BITS; bit += p)
        atomic_and(&t32[bit >> 5], CLR_MASK(bit));
}

#ifndef OWN_F32
#define OWN_F32 1               // 32-bit / float owned-walk setup (+0.58%)
#endif
// One prime of the thread-owned walk (table path) in 32-bit / float
// arithmetic: tf = tile-local first mark (0xffffffff = none), [b0, b1) = my
// bit range, all < TILE_BITS = 2^17, so d = b0 - tf is exact as a float and
// fma(-trunc(d / p), p, d) is the exact remainder in (-p, 2p) (as
// med_prime_f32). Replaces the 64-bit mul_hi / k * p jump of the table path.
/// @pre    p < TILE_BITS; b0 <= b1 <= TILE_BITS; tf < TILE_BITS or 0xffffffff.
/// @post   my bits of p's marks in this tile are cleared (plain RMWs: the
///         range is owned by this thread).
inline void own_prime_f32(__local u32* t32, const u32 tf, const u32 p,
                          const u32 b0, const u32 b1)
{
    if (tf == 0xffffffffu) return;
    u32 bit = tf;
    if (bit < b0) {
        const float df = (float)(b0 - bit);
        const float pf = (float)p;
        const float qf = trunc(df * (1.0f / pf));
        float rf = fma(-qf, pf, df);
        rf += (rf < 0.0f) ? pf : 0.0f;
        rf -= (rf >= pf) ? pf : 0.0f;
        const u32 r = (u32)rf;
        bit = b0 + (r ? p - r : 0u);
    }
    for (; bit < b1; bit += p)
        t32[bit >> 5] &= CLR_MASK(bit);
}

// Medium-stage load batch (v1.10.0). sieve_s per 2e10 window: unbatched
// 0.0824, 2 -> 0.0775, 4 -> 0.0745, 8 -> 0.0728 (+3.6% total), 12 ties,
// 16 spills (-3.7%); a two-stage pipeline on top measured no better.
#ifndef MED_BATCH
#define MED_BATCH 8
#endif

// ----------------------------------------------------------------------------
// make_prime_map
//
// Stage 1: every thread fills its 32 owned SLM words with ones.
// Stage 2: ONE barrier (the only one in the kernel).
// Stage 3a: SMALL_PRIME_MASK — primes < MASK_MAX_P are applied as WHOLE-WORD
//          ANDs from a host-built table instead of per-bit RMWs. For p < 64
//          the scalar loop costs ~TILE_BITS/p RMWs per thread (p=3: ~1365
//          per thread per tile); the mask path costs WORDS_PER_THREAD ANDs
//          (32) regardless of p. The host uploads
//              mask_table[p_idx * 64 + phase]
//          = the 64-bit AND mask for a word whose index mod p == phase.
//          Multiples of p are the global bits g with g = r (mod p); word w
//          covers bits [64w, 64w+64), so its mask clears the b < 64 with
//          b = (r - 64*(w mod p)) mod p, stepping by p. Consecutive words
//          advance the phase by d = 64 mod p, so the walk over my 32 owned
//          words is: load mask[phase], AND, phase += d. Segment-independent
//          table: built once on the host, uploaded once.
// Stage 3.5: the mask path also clears q = p itself (1*p is prime) when the
//          segment starts at or below p; tile 0 restores that single bit.
// Stage 3: for each remaining base prime p, each thread clears the multiples
//          of p that fall inside its own 4096-bit range. No shared writes,
//          so no barrier between primes.
//
//          The HOST precomputes, PER SEGMENT (not per tile), the global bit
//          of each prime's first markable multiple:
//              fb_global[j] = bit of max(p*p, first odd multiple >= q_first)
//          ALL tiles are launched in ONE kernel launch; each work-group
//          derives its own tile-local start from fb_global:
//              fb >= tile_bit0 : bit = fb - tile_bit0
//              fb <  tile_bit0 : bit = fb + ceil(gap/p)*p - tile_bit0
//          where gap = tile_bit0 - fb and the ceil is division-free via the
//          magic reciprocal. The kernel performs ZERO divisions.
//
// Stage 4: each thread writes its 32 words to global memory (coalesced:
//          the whole work-group writes one contiguous 64 KB region).
// ----------------------------------------------------------------------------
/// Tile sieve (stages above). Grid: tiles work-groups (class mode: K * tpc)
/// of WG_SIZE threads; @p tile is TILE_WORDS u64 of SLM.
/// @pre    q_first odd; q_odds < 2^32; sieve_primes ascending with
///         [0, mask_count) exactly the primes < MASK_MAX_P; mask_count <=
///         own_end <= sieve_split; host tables (fb_global, tile_first, masks)
///         built for this segment and slot.
/// @post   each tile's words hold 1 exactly for the q with no factor among
///         sieve_primes[0, sieve_split) (q = p itself restored for mask
///         primes); bits >= the class's valid count are 0.
__kernel void make_prime_map(
        __global u64* restrict prime_map,   // output bitset for the q segment
        __global const u32* restrict sieve_primes,
        __global const u64* restrict fb_global,  // per-prime segment-global first bit
        __global const u64* restrict magic,      // ceil(2^64 / p) per prime
        __global const u32* restrict shifts,     // per-prime shift for div-by-p
        const u32 sieve_count,
        const u64 q_odds,                   // number of odd values in segment
        __local u64* restrict tile,         // SLM tile, 64 KB, set via set_local_arg
        const u32 mask_count,               // primes handled by the mask table (0 = off)
        __global const u64* restrict mask_table, // [mask_count * 64] word masks
        const u64 q_first,                  // first odd q of the segment (stage 3.5)
        const u32 sieve_split,              // tile walk covers primes [0, sieve_split);
                                            // primes >= split are marked by
                                            // sieve_large_primes instead
        __global const u32* restrict tile_first, // [tile_id * own_end + j]
                                            // per-tile first mark bit (host-built);
                                            // 0xffffffff = no mark in this tile
        const u32 use_table,                // 1 = table path, 0 = legacy magic path
        __global const u64* restrict ms32,  // per prime: (m32 | s32 << 32) for the
                                            // 32-bit within-tile ceil jump
        const u32 own_end,                  // primes [mask_count, own_end) use the
                                            // thread-owned walk; [own_end, sieve_split)
                                            // use the prime-parallel SLM-atomic walk
                                            // (own_end == sieve_split disables it)
        __global const u64* restrict group_masks, // fused mask stage: per-group tables,
        __global const u32* restrict group_desc,  // (offset, period) per group
        const u32 n_groups,                 // 0 = per-prime mask loop (stage 3a)
        // class mode (v1.11.0; cm = 0: dense map): tiles [0, tpc) build M1,
        // [tpc, 2 tpc) build M5 (at word tpc * TILE_WORDS); per class the host
        // tables are laid out back to back: fb_global at fb_stride, group
        // masks at mask_coff words, the per-prime mask table at mask_count*64.
        const u32 cm, const u32 tpc, const u32 phis,  // phi1 | phi5 << 8
        const u64 n1, const u64 n5,         // valid bits of M1 / M5
        const u32 mask_coff, const u32 fb_stride,
        __global const u32* restrict group_y)  // per class x MAX_MASK_GROUPS word offsets of
                                               // canonical group tables (null: 0)
{
    const u32 me   = get_local_id(0);
    // cm 1: 2 classes (phis = phi1 | phi5 << 8); cm 2: 8 classes mod 30
    // (phis = 8 nibbles, phi_k = dense bit phase mod 15 of class k)
    const u32 cls = cm ? get_group_id(0) / tpc : 0u;
    const u32 tile_id = get_group_id(0) - cls * tpc;   // tile within its class (dense: all)
    const u64 tile_bit0 = (u64)tile_id * TILE_BITS;
    const u32 phic = (cm == 2u) ? ((phis >> (4u * cls)) & 15u)
                                : (cls ? (phis >> 8) : (phis & 0xFFu));
    const u32 cmod = (cm == 2u) ? 15u : 3u;            // dense bits per class bit
    const u32 qf30 = (u32)(q_first % 30UL);            // wheel-30 phase (dense map)
    const u64 n_valid = (cm == 2u) ? ((q_odds > phic) ? (q_odds - phic + 14UL) / 15UL : 0UL)
                      : cm ? (cls ? n5 : n1) : q_odds;
    fb_global += cls * fb_stride;
    tile_first += (u64)cls * tpc * own_end;          // rows of this class's tiles
    group_masks += cls * mask_coff;
    mask_table += (u64)cls * mask_count * 64UL;

    // ---- Stage 1: initialize my 64 words to "all prime" --------------------
    const u32 my_word0 = me * WORDS_PER_THREAD;
#if SMALL_PRIME_MASK
#ifdef ABL_NOMASK        // timing only (INVALID map): no small-prime masks
    if (0) {
#else
    if (n_groups > 0 && mask_count > 0) {                 // uniform
#endif
        // Fused stages 1 + 3a: every owned word is computed in registers as
        // the AND of one combined mask per prime GROUP (host-built:
        // primes.cpp small_prime_group_masks; a group's mask depends only on
        // word index mod the product of its primes) and stored ONCE. The
        // per-prime path costs a global load plus an SLM read-modify-write
        // per prime (17) per word; this costs n_groups (7) loads per word.
        const u32 x = (u32)((tile_bit0 >> 6) + my_word0);  // global word index < 2^32
        u32 ph[MAX_MASK_GROUPS], per[MAX_MASK_GROUPS], off[MAX_MASK_GROUPS];
        #pragma unroll
        for (u32 g = 0; g < MAX_MASK_GROUPS; ++g) {
            if (g < n_groups) {
                off[g] = group_desc[2 * g];
                per[g] = group_desc[2 * g + 1];
                ph[g] = x % per[g];
                if (group_y) {                  // canonical table: read word (x - Y) mod P
                    const u32 y = group_y[cls * MAX_MASK_GROUPS + g];
                    ph[g] = (ph[g] + per[g] - y) % per[g];
                }
            }
        }
        for (u32 i = 0; i < WORDS_PER_THREAD; ++i) {
            u64 v = ~0UL;
            #pragma unroll
            for (u32 g = 0; g < MAX_MASK_GROUPS; ++g) {
                if (g < n_groups) {
                    v &= group_masks[off[g] + ph[g]];
                    ph[g] = (ph[g] + 1u == per[g]) ? 0u : ph[g] + 1u;
                }
            }
            tile[my_word0 + i] = v;
        }
    } else
#endif
    for (u32 i = 0; i < WORDS_PER_THREAD; ++i)
        tile[my_word0 + i] = ~0UL;

    // ---- Stage 2: the single barrier ---------------------------------------
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- Stage 3a: small-prime mask sieve -----------------------------------
    // Word w of the tile covers global bits [tile_bit0 + 64w, ... + 64).
    // Global bit g <-> q = q_first + 2g; multiples of p are the g with
    // g = r (mod p), r = (p - q_first mod p) * (p+1)/2 mod p (odd-multiple
    // step 2p maps to bit step p). The mask for word w clears the local bits
    // b < 64 with b = (r - 64*w_global) mod p; consecutive words advance the
    // phase by d = 64 mod p. The host bakes all of this into mask_table, so
    // the kernel walk is: load mask[phase], AND, phase += d. No divisions,
    // no per-bit RMWs, fully coalesced SLM traffic.
#if SMALL_PRIME_MASK
    if (mask_count > 0) {
        const u64 w0_global = tile_bit0 >> 6;      // first global word of this tile
        for (u32 j = 0; j < (n_groups > 0 ? 0u : mask_count); ++j) {   // fused path did it
            const u64 p = sieve_primes[j];
            // Phase of my first owned word. x = w0_global + my_word0 < 2^32
            // and p < 64, so a 32-bit modulo is exact and cheap. NOTE: the
            // magic-reciprocal trick with the ceil correction is WRONG here —
            // that computes ceil(x/p), and x - ceil(x/p)*p underflows for
            // every x not divisible by p, sending masks[phase] gigabytes out
            // of bounds (page fault -> device lost). A 64-bit % also regressed
            // the sieve (emulated 64-bit division per prime per thread).
            const u64 x = w0_global + (u64)my_word0;
            u32 phase = (u32)x % (u32)p;
            const __global u64* masks = mask_table + (u64)j * 64UL;
            u32 w = my_word0;
            for (u32 i = 0; i < WORDS_PER_THREAD; ++i, ++w) {
                tile[w] &= masks[phase];
                // The table is indexed by the WORD INDEX mod p: consecutive
                // words advance the phase by exactly 1. (The historical
                // `phase += 64 % p` conflated this with the bit-phase inside
                // a word and mis-aligned every prime with 64 % p != 1 —
                // clearing wrong residue classes, which both restored
                // composites as "prime" (silent false verification) and
                // cleared primes. tools/debug_map.cpp catches this.)
                phase += 1u;
                if (phase >= (u32)p) phase = 0u;
            }
        }
        // Stage 3.5: restore q = p (1*p is prime, not composite). The mask
        // path clears it whenever q_first <= p < MASK_MAX_P. Only tile 0 can
        // contain it; exactly one thread owns the word, so a plain SLM RMW
        // is race-free.
        if (tile_id == 0 && q_first <= (u64)MASK_MAX_P) {
            for (u32 j = 0; j < mask_count; ++j) {
                const u64 p = sieve_primes[j];
                if (p < q_first) continue;         // q = p lies before the segment
                u64 bit = (p - q_first) >> 1;      // global bit of q = p
                if (cm) {                          // class index of q = p (if in this class)
                    if (bit % cmod != phic) continue;
                    bit /= cmod;
                }
                if (bit < TILE_BITS) {
                    const u32 owner = (u32)(bit >> 6);
                    if (owner >= my_word0 && owner < my_word0 + WORDS_PER_THREAD)
                        tile[owner] |= 1UL << (bit & 63UL);
                }
            }
        }
    }
#endif

    // ---- Stage 3: sieve -----------------------------------------------------
    // My bit range inside the tile: [my_bit0, my_bit0 + BITS_PER_THREAD)
    const u64 my_bit0 = (u64)me * BITS_PER_THREAD;
    const u64 my_bit1 = my_bit0 + BITS_PER_THREAD;

    // Primes [0, mask_count) were already applied by the mask path (stage 3a);
    // sieve_primes is ascending, so they are exactly the first mask_count.
    // Primes [sieve_split, sieve_count) are handled by sieve_large_primes.
    const u32 j_begin = (mask_count > 0) ? mask_count : 0;

#ifdef ABL_NOOWN         // timing only (INVALID map): no thread-owned walk
    if (0) {
#else
    if (use_table) {
#endif
        // ---- table path -----------------------------------------------------
        // The host built tile_first[tile_id * own_end + j] = tile-local bit
        // of the first mark of prime j in THIS tile (0xffffffff = none), so the
        // kernel needs NO 64-bit reciprocal math. Two regimes:
        //   p <  TILE_BITS: several marks per tile; jump to my 2048-bit range
        //        with a 32-bit reciprocal (exact: d < TILE_BITS = 2^19, and
        //        m32 = ceil(2^s/p) with 2^s > TILE_BITS*p, so
        //        floor(d*m32/2^s) = floor(d/p) for all d < TILE_BITS).
        //   p >= TILE_BITS: at most ONE mark per tile; its owner is decided by
        //        a range check, no math at all.
        // The walk is latency-bound on the per-prime broadcast loads (table
        // entry, prime, magic), so primes are processed in TABLE_BATCH-sized
        // batches with a two-stage software pipeline: the NEXT batch's loads
        // are issued before the CURRENT batch is consumed, so the ~200-cycle
        // L2 latency overlaps the previous batch's compute. The math is
        // identical to the legacy path, only load scheduling differs.
        const u64 trow = (u64)tile_id * own_end;   // row stride = owned columns
#if OWN_F32
        if (own_end <= (u32)TILE_BITS && sieve_primes[own_end - 1u] < (u32)TILE_BITS) {   // uniform
            __local u32* t32o = (__local u32*)tile;
            const u32 b0 = (u32)my_bit0, b1 = (u32)my_bit1;
            u32 j = j_begin;
            u32 tfa[TABLE_BATCH], pva[TABLE_BATCH];
            const bool preloaded = (j + TABLE_BATCH <= own_end);
            if (preloaded) {
                #pragma unroll
                for (u32 u = 0; u < TABLE_BATCH; ++u) {
                    tfa[u] = tile_first[trow + j + u];
                    pva[u] = sieve_primes[j + u];
                }
            }
            for (; j + 2 * TABLE_BATCH <= own_end; j += TABLE_BATCH) {
                u32 tfb[TABLE_BATCH], pvb[TABLE_BATCH];
                #pragma unroll
                for (u32 u = 0; u < TABLE_BATCH; ++u) {   // stage: load next
                    tfb[u] = tile_first[trow + j + TABLE_BATCH + u];
                    pvb[u] = sieve_primes[j + TABLE_BATCH + u];
                }
                #pragma unroll
                for (u32 u = 0; u < TABLE_BATCH; ++u)     // consume current
                    own_prime_f32(t32o, tfa[u], pva[u], b0, b1);
                #pragma unroll
                for (u32 u = 0; u < TABLE_BATCH; ++u) { tfa[u] = tfb[u]; pva[u] = pvb[u]; }
            }
            if (preloaded) {
                #pragma unroll
                for (u32 u = 0; u < TABLE_BATCH; ++u)
                    if (j + u < own_end) own_prime_f32(t32o, tfa[u], pva[u], b0, b1);
                j += TABLE_BATCH;
            }
            for (; j < own_end; ++j)
                own_prime_f32(t32o, tile_first[trow + j], sieve_primes[j], b0, b1);
        } else
#endif
        {
        u32 j = j_begin;
        u32 tfa[TABLE_BATCH], pva[TABLE_BATCH];
        u64 mva[TABLE_BATCH];
        // Only a PRELOADED batch may be consumed below: with fewer than
        // TABLE_BATCH owned primes, the arrays are uninitialized and those
        // primes must go to the scalar tail (else they are silently skipped
        // -> missed marks -> composites left "prime").
        const bool preloaded = (j + TABLE_BATCH <= own_end);
        if (preloaded) {
            #pragma unroll
            for (u32 u = 0; u < TABLE_BATCH; ++u) {
                tfa[u] = tile_first[trow + j + u];
                pva[u] = sieve_primes[j + u];
                mva[u] = ms32[j + u];
            }
        }
        for (; j + 2 * TABLE_BATCH <= own_end; j += TABLE_BATCH) {
            u32 tfb[TABLE_BATCH], pvb[TABLE_BATCH];
            u64 mvb[TABLE_BATCH];
            #pragma unroll
            for (u32 u = 0; u < TABLE_BATCH; ++u) {   // stage: load next
                tfb[u] = tile_first[trow + j + TABLE_BATCH + u];
                pvb[u] = sieve_primes[j + TABLE_BATCH + u];
                mvb[u] = ms32[j + TABLE_BATCH + u];
            }
            #pragma unroll
            for (u32 u = 0; u < TABLE_BATCH; ++u) {   // consume current
                if (tfa[u] == 0xffffffffU) continue;
                const u64 p = pva[u];
                u64 bit = tfa[u];
                if (bit < my_bit0) {
                    if (p >= TILE_BITS) continue;   // its only mark is not mine
                    const u32 d = (u32)(my_bit0 - bit);   // < TILE_BITS = 2^19
                    const u32 m32 = (u32)mva[u];
                    const u32 s32 = (u32)(mva[u] >> 32);
                    u32 k = (u32)(((u64)d * m32) >> s32); // floor(d/p), exact
                    k += (k * p < d) ? 1u : 0u;           // ceil(d/p)
                    bit += (u64)k * p;
                }
                mark_owned_u32(tile, bit, my_bit1, p);
            }
            #pragma unroll
            for (u32 u = 0; u < TABLE_BATCH; ++u) {   // rotate: A <- B
                tfa[u] = tfb[u];
                pva[u] = pvb[u];
                mva[u] = mvb[u];
            }
        }
        if (preloaded) {   // consume the pending batch, then a scalar tail
            #pragma unroll
            for (u32 u = 0; u < TABLE_BATCH; ++u) {
                if (j + u >= own_end) break;
                if (tfa[u] == 0xffffffffU) continue;
                const u64 p = pva[u];
                u64 bit = tfa[u];
                if (bit < my_bit0) {
                    if (p >= TILE_BITS) continue;
                    const u32 d = (u32)(my_bit0 - bit);
                    const u32 m32 = (u32)mva[u];
                    const u32 s32 = (u32)(mva[u] >> 32);
                    u32 k = (u32)(((u64)d * m32) >> s32);
                    k += (k * p < d) ? 1u : 0u;
                    bit += (u64)k * p;
                }
                mark_owned_u32(tile, bit, my_bit1, p);
            }
            j += TABLE_BATCH;
        }
        for (; j < own_end; ++j) {
            const u32 tf32 = tile_first[trow + j];
            if (tf32 == 0xffffffffU) continue;
            const u64 p = sieve_primes[j];
            u64 bit = tf32;
            if (bit < my_bit0) {
                if (p >= TILE_BITS) continue;
                const u32 d = (u32)(my_bit0 - bit);
                const u64 ms = ms32[j];
                u32 k = (u32)(((u64)d * (u32)ms) >> (u32)(ms >> 32));
                k += (k * p < d) ? 1u : 0u;
                bit += (u64)k * p;
            }
            mark_owned_u32(tile, bit, my_bit1, p);
        }
        }
#ifdef ABL_NOOWN
    } else if (0) {
#else
    } else {
#endif
    // ---- legacy reciprocal path (huge segments / debug tools) ---------------
    for (u32 j = j_begin; j < own_end; ++j) {
        // Segment-global first markable bit; ~0 = prime irrelevant for the
        // whole segment (p*p beyond segment end; host owns ALL skip logic).
        const u64 fb = fb_global[j];
        if (fb == ~0UL) continue;

        const u64 p = sieve_primes[j];

        // Tile-local first markable bit.
        u64 bit;
        if (fb >= tile_bit0) {
            bit = fb - tile_bit0;
        } else {
            const u64 d = tile_bit0 - fb;              // < 2^27 for our segments
            u64 k = mul_hi(d, magic[j]) >> shifts[j];  // floor(d/p)
            k += (k * p < d) ? 1UL : 0UL;              // ceil(d/p)
            bit = fb + k * p - tile_bit0;
        }
        if (bit >= TILE_BITS) continue;                // prime starts beyond tile

        // Jump from the tile start to MY owned range (division-free).
        if (bit < my_bit0) {
            const u64 d = my_bit0 - bit;
            u64 k = mul_hi(d, magic[j]) >> shifts[j];  // floor(d/p)
            k += (k * p < d) ? 1UL : 0UL;              // ceil(d/p)
            bit += k * p;
        }

        mark_owned_u32(tile, bit, my_bit1, p);
    }
    }


    // ---- Stage 3m: medium primes, prime-parallel with SLM atomics ----------
    // Primes [own_end, sieve_split) are large relative to a thread's
    // 2048-bit range (at most a few marks each), so the thread-owned walk
    // above would make EVERY thread visit EVERY such prime just to find ~0-1
    // marks: cost O(WG_SIZE * primes) per tile. Here each thread instead owns
    // whole PRIMES (j = own_end + me, + WG_SIZE, ...) and walks all of that
    // prime's marks across the tile: cost O(primes + marks) per tile. Marks
    // from different threads can hit the same word, so they are 32-bit SLM
    // atomic ANDs (monotone clears: any interleaving converges). This is what
    // lets the split grow past the old thread x prime cost wall and move
    // global-memory atomics out of sieve_large_primes into SLM.
    // Barriers: the owned walk above uses plain RMWs, so it must be finished
    // before any cross-thread atomic touches its words; and stage 4 reads
    // words other threads cleared, so every atomic must be done before it.
    // The first bit per tile comes from the segment-global fb_global (host,
    // ~0 = no mark in the segment) with one exact 32-bit modulo; d < 2^32 by
    // the host's segment-span guard (q_odds < 2^32).
#ifdef ABL_NOMED          // timing only (INVALID map): no medium stage
    if (0) {
#else
    if (own_end < sieve_split) {                       // uniform
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
        volatile __local u32* t32 = (volatile __local u32*)tile;
        // Loads are batched MED_BATCH primes deep: the walk is latency-bound
        // on the per-prime fb/p loads (a loads-only pass costs a third of
        // the stage), so all loads of a batch are issued before any use.
        u32 j = own_end + me;
#if MED_F32
        if (cm && n_valid <= (1UL << 24) && sieve_split > 0u &&
            sieve_primes[sieve_split - 1u] < (1u << 23)) {   // uniform (med_prime_f32 PRE)
            __global const u32* fb32 = (__global const u32*)fb_global;   // low words
            const u32 tb = (u32)tile_bit0;
            for (; j + (MED_BATCH - 1) * WG_SIZE < sieve_split; j += MED_BATCH * WG_SIZE) {
                u32 fbv[MED_BATCH], pv[MED_BATCH];
                #pragma unroll
                for (u32 u = 0; u < MED_BATCH; ++u) {
                    fbv[u] = fb32[2u * (j + u * WG_SIZE)];
                    pv[u] = sieve_primes[j + u * WG_SIZE];
                }
                #pragma unroll
                for (u32 u = 0; u < MED_BATCH; ++u)
                    med_prime_f32(t32, fbv[u], pv[u], tb);
            }
            for (; j < sieve_split; j += WG_SIZE)
                med_prime_f32(t32, fb32[2u * j], sieve_primes[j], tb);
        }
#endif
        if (cm) {                                       // class map: no wheel (uniform)
            for (; j + (MED_BATCH - 1) * WG_SIZE < sieve_split; j += MED_BATCH * WG_SIZE) {
                u64 fbv[MED_BATCH];
                u32 pv[MED_BATCH];
                #pragma unroll
                for (u32 u = 0; u < MED_BATCH; ++u) {
                    fbv[u] = fb_global[j + u * WG_SIZE];
                    pv[u] = sieve_primes[j + u * WG_SIZE];
                }
                #pragma unroll
                for (u32 u = 0; u < MED_BATCH; ++u)
                    med_prime_nw(t32, fbv[u], pv[u], tile_bit0);
            }
            for (; j < sieve_split; j += WG_SIZE)
                med_prime_nw(t32, fb_global[j], sieve_primes[j], tile_bit0);
        }
        for (; j + (MED_BATCH - 1) * WG_SIZE < sieve_split; j += MED_BATCH * WG_SIZE) {
            u64 fbv[MED_BATCH];
            u32 pv[MED_BATCH];
            #pragma unroll
            for (u32 u = 0; u < MED_BATCH; ++u) {
                fbv[u] = fb_global[j + u * WG_SIZE];
                pv[u] = sieve_primes[j + u * WG_SIZE];
            }
            #pragma unroll
            for (u32 u = 0; u < MED_BATCH; ++u)
                med_prime(t32, fbv[u], pv[u], tile_bit0, qf30);
        }
        for (; j < sieve_split; j += WG_SIZE)
            med_prime(t32, fb_global[j], sieve_primes[j], tile_bit0, qf30);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ---- Stage 4: write my WORDS_PER_THREAD words to global (coalesced) ----
    // Global word of tile-local word w:
    //   global_bit = tile_id * TILE_BITS + w*64
    //   -> global word = (cls * tpc + tile_id) * TILE_WORDS + w
    // Bits beyond q_odds (last tile tail) are masked to 0 so the map is exact.
    __global u64* dst = prime_map + ((u64)cls * tpc + tile_id) * TILE_WORDS;
    for (u32 i = 0; i < WORDS_PER_THREAD; ++i) {
        const u32 w = my_word0 + i;
        u64 v = tile[w];
        const u64 bit0 = tile_bit0 + (u64)w * 64UL;
        if (bit0 + 64UL > n_valid) {
            const u64 valid = (n_valid > bit0) ? (n_valid - bit0) : 0UL;
            v &= (valid >= 64UL) ? ~0UL : ((1UL << valid) - 1UL);
        }
        dst[w] = v;
    }
}

// ----------------------------------------------------------------------------
// sieve_large_primes — one work-item per LARGE_PRIMES_PER_ITEM = 4 base
// primes p >= sieve_split.
//
// The tile walk above costs O(pi(sieve_split)) per thread per tile, which is
// fine while sieve_split is small but explodes at N=4e18 (pi(2e9) ~ 98M).
// Large primes contribute few marks per segment (B/p each), so a flat
// one-work-item-per-prime layout is the right shape: total work is
// sum over p >= split of (segment_odds/p + 1), independent of tile count.
//
// Each work-item:
//   1. reads its prime p,
//   2. computes its first markable bit INLINE (the math formerly run by the
//      separate prep_large_fb kernel and a ~393 MB device-side fb table —
//      one u64 division per prime is cheaper than the load latency, and the
//      fusion removes two 393 MB streams and one kernel launch per segment),
//   3. walks the multiples, clearing bits DIRECTLY in the global map with
//      atomic_and (multiple work-items may touch the same word),
//   4. skips bits beyond q_odds (map tail is masked by the tile kernel).
//
//   first = p*p                                  if p*p >= q_first
//   first = (first odd multiple of p >= q_first) otherwise
//   bit   = (first - q_first) / 2                if first <= q_last, else none
//
// bit i of the map <-> odd q = q_first + 2*i, so (first - q_first) is even
// and the shift is exact. tools/debug_fb.cpp validates the formula
// bit-for-bit against the host reference; tools/debug_sieve validates the
// completed map (tile kernel + this kernel) against a CPU sieve.
//
// Launched as ceil(ceil((sieve_count - sieve_split)/LARGE_PRIMES_PER_ITEM)
// / WG_SIZE) work-groups of WG_SIZE (256); each work-item clears up to 4
// primes' marks (bounds guard inside).
// ----------------------------------------------------------------------------
/// @pre    runs after make_prime_map on the same map; primes[sieve_split, ...)
///         > 61 when LARGE_WHEEL30 (host: sieve_split >= mask_count).
/// @post   every odd multiple >= p*p of each large prime inside the segment
///         is cleared.
__kernel void sieve_large_primes(
        volatile __global u64* restrict prime_map,
        __global const u32* restrict sieve_primes,
        const u32 sieve_split,              // first index of the large-prime range
        const u32 sieve_count,              // total base primes (bounds guard)
        const u64 q_odds,                   // valid bits in the map
        const u64 q_first,                  // first odd q of the segment
        const u64 q_last,                   // last odd q covered by the segment
        const u32 cm, const u32 phi1, const u32 phi5, const u32 off5)   // class mode
{
    const cmap g = {cm, phi1, phi5, off5};
    // A bounded batch amortizes work-item and scheduling overhead. Near 4e18
    // there are ~98M large primes, most producing zero or one mark; one item
    // per prime needlessly created ~383K work-groups per segment.
    const u32 qf30 = (u32)(q_first % 30UL);   // wheel-30 phase (uniform)
    const u32 qfw = (u32)(q_first % (u64)WHEEL_W);
    const u64 base = (u64)get_global_id(0) * LARGE_PRIMES_PER_ITEM;
    #pragma unroll
    for (u32 lane = 0; lane < LARGE_PRIMES_PER_ITEM; ++lane) {
        const u64 compact = base + lane;
        const u32 j = sieve_split + (u32)compact;
        if (j >= sieve_count) break;         // launch rounds up to WG multiples
        const u64 p = sieve_primes[j];
        const u64 first = first_odd_multiple(p, q_first);
        if (first > q_last) continue;        // p irrelevant for this segment
#ifdef LARGE_NOMARK   // timing experiment only: setup cost without marks (INVALID map)
        if (first != 1) continue;
#endif
        // 32-bit atomics on the containing u32 word (see mark_multiples_u32;
        // 64-bit atomic_and is absent from this ocloc's headers).
#if LARGE_WHEEL30
        if (p > 61UL && p < (1UL << 30))
#if WHEEL_W == 30
            mark_multiples_w30(prime_map, (first - q_first) >> 1, (u32)p, q_odds, qf30, g);
#else
            mark_multiples_whl(prime_map, (first - q_first) >> 1, (u32)p, q_odds, qfw, g);
#endif
        else
#endif
        mark_multiples_u32(prime_map, (first - q_first) >> 1, p, q_odds, g);
    }
}

// Subgroup width of the packed decode and primes per work-item; one packing
// group = LARGE_PACK_PPI * width primes (MUST match gb::LARGE_PACK_GROUP and
// gb::LARGE_PACK_PPI on the host).
#ifndef LARGE_PACK_SG
#define LARGE_PACK_SG 32        // SIMD32 measured ~5% faster than 16
#endif
#ifndef LARGE_PACK_PPI
#define LARGE_PACK_PPI 16       // primes per work-item (multiple of 4; gap bytes = PPI/4 u32s): 4 -> 8 -> 16 = large_s -2.5% then -1%
#endif
#define LARGE_PACK_GROUP (LARGE_PACK_PPI * LARGE_PACK_SG)
#define LARGE_PACK_WORDS (LARGE_PACK_PPI / 4)
#ifndef LARGE_GAPS_CC
#define LARGE_GAPS_CC 5          // gap-stream loads L1S_L3UC: +0.4% (L1UC_L3UC = 1: -1.3%; 0 = default)
#endif
#if LARGE_GAPS_CC
#if LARGE_PACK_WORDS != 4
#error "LARGE_GAPS_CC loads one uint4 per item: LARGE_PACK_PPI must be 16"
#endif
uint4 __builtin_IB_lsc_load_global_uint4(__global uint4* base, int immElemOff, enum LSC_LDCC cacheOpt);
#endif
// Decode the LARGE_PACK_PPI primes of `item` (called by every lane of the
// subgroup: contains a subgroup scan). Entries past large_count are garbage
// and must be skipped by the caller.
/// @pre    called by all lanes of a LARGE_PACK_SG subgroup of consecutive
///         items (the subgroup scan); large_count >= 1; bases / gaps4 from
///         pack_large_primes for the same list (half-gaps < 256, 0 at each
///         group start).
/// @param pv  out: the item's LARGE_PACK_PPI primes, ascending.
/// @post   pv[k] = list prime item * LARGE_PACK_PPI + k for every k with
///         that index < large_count (exact: u32 sums of the host's gaps);
///         entries past large_count are unspecified.
inline void decode_large(__global const u32* restrict bases,
                         __global const u32* restrict gaps4,
                         const u32 item, const u32 large_count, u32* pv)
{
    u32 gb[LARGE_PACK_PPI];
    u32 sum = 0;
#if LARGE_GAPS_CC
    // The gap stream (~33 MB per segment, each word read once) is larger
    // than L3 (18 MB): a streaming hint keeps it from evicting the map under the
    // atomics. Full items take one 16-byte load, the list's last item the
    // plain bounded loads.
    u32 gw[LARGE_PACK_WORDS];
    if (item * LARGE_PACK_PPI + 4u * (LARGE_PACK_WORDS - 1u) < large_count) {
        const uint4 v = __builtin_IB_lsc_load_global_uint4((__global uint4*)gaps4 + item, 0,
                                                           (enum LSC_LDCC)LARGE_GAPS_CC);
        gw[0] = v.x; gw[1] = v.y; gw[2] = v.z; gw[3] = v.w;
    } else {
        #pragma unroll
        for (u32 k = 0; k < LARGE_PACK_WORDS; ++k)
            gw[k] = (item * LARGE_PACK_PPI + 4u * k < large_count) ? gaps4[item * LARGE_PACK_WORDS + k] : 0u;
    }
#endif
    #pragma unroll
    for (u32 k = 0; k < LARGE_PACK_WORDS; ++k) {
#if LARGE_GAPS_CC
        const u32 w = gw[k];
#else
        const u32 w = (item * LARGE_PACK_PPI + 4u * k < large_count)
                    ? gaps4[item * LARGE_PACK_WORDS + k] : 0u;
#endif
        gb[4 * k]     = w & 0xFFu;
        gb[4 * k + 1] = (w >> 8) & 0xFFu;
        gb[4 * k + 2] = (w >> 16) & 0xFFu;
        gb[4 * k + 3] = w >> 24;
        sum += gb[4 * k] + gb[4 * k + 1] + gb[4 * k + 2] + gb[4 * k + 3];
    }
    const u32 excl = sub_group_scan_exclusive_add(sum);
    // Primes < 2^32 (host fails closed otherwise): rebuild them in 32 bits
    // (Xe2 emulates every u64 add).
    u32 p = bases[min(item / LARGE_PACK_SG, (large_count - 1u) / LARGE_PACK_GROUP)] + 2u * excl;
    #pragma unroll
    for (u32 k = 0; k < LARGE_PACK_PPI; ++k) {
        p += 2u * gb[k];
        pv[k] = p;
    }
}

// ----------------------------------------------------------------------------
// sieve_large_packed — production large-prime kernel (v1.7.0): same marking
// as sieve_large_primes, but the prime list is streamed at ONE BYTE per
// prime. The division-free setup was bound by streaming the u32 list
// (~393 MB per segment at 4e18), and the stream also evicted the prime map
// from L3 under the atomics; at 1 B/prime large_s fell ~14%.
// Layout (host: pack_large_primes), for the large primes [split, count):
//   gaps[i]  = (p_i - p_{i-1}) / 2 as u8, 0 at every packing-group start
//              (max half-gap below 2^32 is < 256; host fails closed)
//   bases[g] = first prime of packing group g
// A subgroup owns one packing group; lane l owns primes PPI*l .. PPI*l+PPI-1,
// reads its gap bytes as PPI/4 coalesced u32s, and rebuilds its primes from
// bases[g] + 2 * (subgroup exclusive scan + in-lane prefix).
// Validated: tools/debug_fb (decode of every prime vs the host list via
// unpack_large_test) and tools/debug_map (complete map vs Miller-Rabin).
// ----------------------------------------------------------------------------
/// @pre    as sieve_large_primes; primes from pack_large_primes (< 2^32,
///         all > 61: host sieve_split >= mask_count).
/// @post   as sieve_large_primes.
inline void large_packed_body(
        volatile __global u64* restrict prime_map,
        __global const u32* restrict bases,
        __global const u32* restrict gaps4,
        const u32 large_count,
        const u64 q_odds,
        const u64 q_first,
        const u64 q_last,
        const cmap g)
{
    const u32 item = get_global_id(0);
    const u32 c0 = item * LARGE_PACK_PPI;
    const u32 qf30 = (u32)(q_first % 30UL);   // wheel-30 phase (uniform)
    const u32 qfw = (u32)(q_first % (u64)WHEEL_W);
#if LARGE_OH_BM
    const u32 qf15 = qfw % 15015u;                // q_first mod 15015 (uniform)
#endif
    u32 pv[LARGE_PACK_PPI];
    decode_large(bases, gaps4, item, large_count, pv);
    // q_first >= 2^60 (uniform) and p < 2^30: p*p < q_first always, so the
    // first odd multiple is q_first + t with t = (p - q_first mod p) mod p,
    // plus p when t is odd (q_first is odd) — all in 32 bits, no emulated
    // u64 multiply / compare / add per prime. bit0 = t / 2 < p; a bit0
    // beyond the segment is rejected by the mark loop (first > q_last).
    const bool hi_range = (q_first >> 60) != 0;
    const ff_x qf_split = ff_split(q_first);
    #pragma unroll
    for (u32 lane = 0; lane < LARGE_PACK_PPI; ++lane) {
        const u64 p = pv[lane];
        if (c0 + lane >= large_count) break;
#if LARGE_WHEEL30 && LARGE_FIRST32
        if (hi_range && p < (1UL << 30)) {
            const u32 p32 = (u32)p;
#ifdef ABL_SKIPOH      // timing only (INVALID map): one-hit primes skipped entirely
            if (p32 >= (u32)q_odds) continue;
#endif
            const u32 r = mod_ff(qf_split, p32);
            u32 t = r ? p32 - r : 0u;
            t += (t & 1u) ? p32 : 0u;
#ifdef ABL_MODONLY     // timing only (INVALID map): decode + modulo
            if (t == 0xFFFFFFFFu) prime_map[0] = 0UL;
            continue;
#endif
#if LARGE_OH_BM && WHEEL_W == 30030
            if (p32 >= (u32)q_odds) {                    // at most one hit (bit < p)
                const u32 b = t >> 1;
                if (b < (u32)q_odds) {
#if OH_F32
                    // (q_first + 2b) mod 15015 = (qf15 + 2b) mod 15015 with a
                    // float quotient: y < 2^31 + 15015 (b < p < 2^30), so the
                    // estimate is within 1 of floor(y / 15015) and the
                    // wrap-around remainder lies in [-15015, 2 * 15015)
                    const u32 y = qf15 + 2u * b;
                    const u32 qy = (u32)((float)y * (1.0f / 15015.0f));
                    int xi = (int)(y - qy * 15015u);
                    xi += (xi < 0) ? 15015 : 0;
                    xi -= (xi >= 15015) ? 15015 : 0;
                    const u32 x = (u32)xi;
#else
                    u32 x = qf15 + 2u * (b % 15015u);      // first mod 15015, < 3 * 15015
                    x -= (x >= 15015u) ? 15015u : 0u;
                    x -= (x >= 15015u) ? 15015u : 0u;
#endif
#ifdef ABL_OHNOATOM    // timing only (INVALID map): one-hit setup + test, no atomic
                    if (((CP15015[x >> 5] >> (x & 31u)) & 1u) && b == 0xFFFFFFFFu)
#else
                    if ((CP15015[x >> 5] >> (x & 31u)) & 1u)
#endif
                        map_clear_w((volatile __global u32*)prime_map, b, g);
                }
                continue;
            }
#endif
            mark_multiples_whl(prime_map, (u64)(t >> 1), p32, q_odds, qfw, g);
            continue;
        }
#endif
        const u64 first = first_odd_multiple(p, q_first);
        if (first > q_last) continue;        // p irrelevant for this segment
#ifdef LARGE_NOMARK
        if (first != 1) continue;
#endif
#if LARGE_WHEEL30
        if (p > 61UL && p < (1UL << 30))
#if WHEEL_W == 30
            mark_multiples_w30(prime_map, (first - q_first) >> 1, (u32)p, q_odds, qf30, g);
#else
            mark_multiples_whl(prime_map, (first - q_first) >> 1, (u32)p, q_odds, qfw, g);
#endif
        else
#endif
        mark_multiples_u32(prime_map, (first - q_first) >> 1, p, q_odds, g);
    }
}


__attribute__((intel_reqd_sub_group_size(LARGE_PACK_SG)))
__kernel void sieve_large_packed(
        volatile __global u64* restrict prime_map,
        __global const u32* restrict bases,
        __global const u32* restrict gaps4,
        const u32 large_count,
        const u64 q_odds,
        const u64 q_first,
        const u64 q_last,
        const u32 cm, const u32 phi1, const u32 phi5, const u32 off5)   // class mode
{
    const cmap g = {cm, phi1, phi5, off5};
    large_packed_body(prime_map, bases, gaps4, large_count, q_odds, q_first, q_last, g);
}

// Validation only (tools/debug_fb): writes every decoded large prime with
// the production decode_large, for a bit-exact compare with the host list.
/// @post   out[i] = decoded large prime i for every i < large_count.
__attribute__((intel_reqd_sub_group_size(LARGE_PACK_SG)))
__kernel void unpack_large_test(__global const u32* restrict bases,
                                __global const u32* restrict gaps4,
                                const u32 large_count,
                                __global u32* restrict out)
{
    const u32 item = get_global_id(0);
    u32 pv[LARGE_PACK_PPI];
    decode_large(bases, gaps4, item, large_count, pv);
    for (u32 k = 0; k < LARGE_PACK_PPI; ++k)
        if (item * LARGE_PACK_PPI + k < large_count) out[item * LARGE_PACK_PPI + k] = (u32)pv[k];
}

// ----------------------------------------------------------------------------
// Bucket sieve for the largest primes (v1.7.0): bucket_activate,
// bucket_process, map_compare.
//
// Primes p > q_odds_full (the bit length of a full segment's q range) hit a
// segment at most once, yet sieve_large_packed paid their setup (decode +
// modulo, ~75% of them with zero marks) for all ~91M of them every segment.
// Here each ACTIVE bucketed prime owns exactly one entry
//     u64 entry = off | (u64)p << 32
// = its next hit at bit `off` of the segment the entry is filed under. The
// entries live in a ring of R buckets (bucket t % R holds segment t's hits;
// R > max lookahead (q_odds_full + p_max) / delta + 1). Coordinates: segment
// t covers global bits [t*delta, t*delta + q_odds_t), delta = segment span
// / 2; consecutive q ranges OVERLAP by q_odds_full - delta bits (the 9781
// back-extension), so a hit can belong to two segments.
//   first_seg(x) = first segment containing global bit x
//                = x < q_odds_full ? 0 : (x - q_odds_full) / delta + 1
// bucket_process(segment k): for each entry of bucket k: mark `off`; then
//   off >= delta -> the same hit is also in segment k+1: file it there
//                   (off - delta);
//   otherwise    -> file the NEXT hit x = off + p under first_seg(x)
//                   relative to k (>= k+1 because p > q_odds_full).
// The last segment of a run marks only (off < q_odds) and files nothing.
// bucket_activate(k): files the first hit of primes [a, b) (host-chosen:
// every prime whose first hit falls within the ring horizon k + R - 1 by
// now); at k = 0 that is all primes with p*p < q_first0, later only the few
// whose first hit is p*p itself.
// Filing is aggregated per work-group (SLM counters per target bucket, one
// global atomic per bucket per chunk). Any impossible state — bucket
// overflow, an entry outside a non-last segment, a lookahead >= R, an
// activation behind the current segment — sets a bit in *err and the host
// throws (fail closed). Validated end to end by GB_BUCKET_VERIFY=1 (the
// verifier rebuilds every segment's map without buckets and map_compare
// counts differing words).
// ----------------------------------------------------------------------------
#define BKT_MAX_R 256           // ring size limit (wheel-30030 lookahead up to 11p/delta)
#ifndef BKT_E
#define BKT_E 8                 // entries per lane per chunk
#endif
#define BKT_ERR_ENTRY     1u    // entry beyond a non-last segment
#define BKT_ERR_OVERFLOW  2u    // bucket capacity exceeded
#define BKT_ERR_LOOKAHEAD 4u    // target bucket >= R ahead
#define BKT_ERR_LATE      8u    // activation target behind current segment

// Exact floor(x / d) for x < 2^40, d >= 2^16 without a 64-bit division:
// float estimate + exact integer fix-ups (precision only bounds iterations).
/// @pre    d >= 1, inv_d ~ 1/d; (q + 1) d must not overflow.
/// @return floor(x / d).
inline u64 bkt_div(const u64 x, const u64 d, const float inv_d)
{
    u64 q = (u64)((float)x * inv_d);
    while (q * d > x) --q;
    while ((q + 1UL) * d <= x) ++q;
    return q;
}

/// @pre    delta >= 1, inv_delta ~ 1 / delta; segment s covers global bits
///         [s delta, s delta + q_odds_full).
/// @return the first segment (relative index) whose q range contains global
///         bit x: 0 if x < q_odds_full, else (x - q_odds_full) / delta + 1.
inline u32 bkt_first_seg_rel(const u64 x, const u64 q_odds_full, const u64 delta,
                             const float inv_delta)
{
    return (x < q_odds_full) ? 0u
         : (u32)bkt_div(x - q_odds_full, delta, inv_delta) + 1u;
}

// 32-bit forms for bucket_process (host guarantees p_max + q_odds_full <
// 2^32, so off + p and every quotient fit): Xe2 emulates u64 multiply and
// compare, which dominated the per-entry filing arithmetic.
/// @pre    d >= 1, inv_d ~ 1/d.
/// @return floor(x / d).
inline u32 bkt_div32(const u32 x, const u32 d, const float inv_d)
{
    u32 q = (u32)((float)x * inv_d);
    while (q > 0u && q * d > x) --q;          // q*d <= x < 2^32: no overflow
    while (x - q * d >= d) ++q;
    return q;
}

// Bucket entries are written once and read once, 3..125 segments later:
// cache-controlled LSC accesses keep that ~70 MB/segment stream from
// evicting the L3-resident prime map under the bucket marks.
#ifndef BKT_ST_CC
#define BKT_ST_CC 0
#endif
#ifndef BKT_LD_CC
#define BKT_LD_CC 5              // L1S_L3UC loads: bucket 0.92 -> 0.83 ms/segment. Uncached
                                 // STORES (L1UC_L3UC, L1S_L3UC) were 2-3.7x slower: partial-line
                                 // writes lose L3 write-combining.
#endif
#if BKT_ST_CC || BKT_LD_CC
void __builtin_IB_lsc_store_global_ulong(__global ulong* base, int immElemOff, ulong val, enum LSC_STCC cacheOpt);
ulong __builtin_IB_lsc_load_global_ulong(__global ulong* base, int immElemOff, enum LSC_LDCC cacheOpt);
#endif
/// Stores a bucket entry (optionally with an LSC cache hint, BKT_ST_CC).
/// @pre    p 8-byte aligned, inside the ring.
/// @post   *p = v (a cache hint changes only where the line is kept).
inline void bkt_store(__global u64* p, const u64 v)
{
#if BKT_ST_CC
    __builtin_IB_lsc_store_global_ulong(p, 0, v, (enum LSC_STCC)BKT_ST_CC);
#else
    *p = v;
#endif
}
/// Loads a bucket entry (L3-uncached by default, BKT_LD_CC).
/// @pre    p 8-byte aligned; *p was written by an earlier kernel of the
///         same in-order list with a barrier in between (bucket entries and
///         hit lists are never read by the kernel that wrote them).
/// @return *p.
inline u64 bkt_load(const __global u64* p)
{
#if BKT_LD_CC
    return __builtin_IB_lsc_load_global_ulong((__global u64*)p, 0, (enum LSC_LDCC)BKT_LD_CC);
#else
    return *p;
#endif
}

// The ring is split over two allocations (buckets [0, split) and [split, R)):
// on this driver the first submission touching a single allocation above
// ~5 GiB cost ~35-45 ms of host time (4.97 GiB: none, 5.16 GiB: 33 ms), and
// the 30030-wheel ring at 3 x q_odds is ~6.9 GiB. Two halves stay below it.
/// @return base of bucket @p b (b < R) in the two-allocation ring.
inline __global u64* bkt_bucket(__global u64* e0, __global u64* e1, const u32 split,
                                const u32 cap, const u32 b)
{
    return (b < split) ? e0 + (u64)b * cap : e1 + (u64)(b - split) * cap;
}

// Work-group-aggregated filing of up to BKT_E entries per lane. d[e] = target
// bucket distance from `slot` (0 = nothing to file, except when allow0).
/// @param d,ne,has  per lane entry: bucket distance (< R), entry, valid flag.
/// @pre    called by every work-item of the group; local size >= R (host:
///         256 = BKT_MAX_R >= R); lcnt[0, R) zeroed and visible.
/// @post   each valid entry is stored in bucket (slot + d) % R, or *err gets
///         BKT_ERR_OVERFLOW when the bucket is full.
inline void bkt_file(__global u64* entries, __global u64* entries1, const u32 split,
                     volatile __global u32* restrict counts,
                     const u32 cap, const u32 R, const u32 slot,
                     const u32* d, const u64* ne, const bool* has,
                     __local u32* lcnt, __local u32* lbase,
                     volatile __global u32* restrict err)
{
    const u32 lid = get_local_id(0);
    u32 pos[BKT_E];
    #pragma unroll
    for (u32 e = 0; e < BKT_E; ++e)
        pos[e] = has[e] ? atomic_inc(&lcnt[d[e]]) : 0u;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < R && lcnt[lid])
        lbase[lid] = atomic_add(&counts[(slot + lid) % R], lcnt[lid]);
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (u32 e = 0; e < BKT_E; ++e) {
        if (!has[e]) continue;
        const u32 idx = lbase[d[e]] + pos[e];
        if (idx < cap) bkt_store(bkt_bucket(entries, entries1, split, cap, (slot + d[e]) % R) + idx, ne[e]);
        else atomic_or(err, BKT_ERR_OVERFLOW);
    }
}

/// Marks this segment's bucketed hits and re-files each prime's next hit.
/// @pre    local size >= R (lcnt / lbase are reset and filled only for
///         lid < local size; host: WG_SIZE = 256 >= R); p_max + q_odds_full < 2^32.
/// @post   every entry of bucket @p slot is marked (off < q_odds) and, unless
///         is_last, re-filed; any impossible state sets a bit in *err.
__kernel void bucket_process(
        volatile __global u64* restrict prime_map,
        __global u64* entries,                   // buckets [0, split), [cap] each
        volatile __global u32* restrict counts,  // [R]
        const u32 cap,
        const u32 R,
        const u32 slot,                          // this segment's bucket (k % R)
        const u64 q_odds,                        // this segment
        const u64 q_odds_full,
        const u64 delta,
        const float inv_delta,
        const u32 is_last,
        volatile __global u32* restrict err,
        const u32 qfw,                           // this segment's q_first mod WHEEL_W
        __global u64* entries1,                  // buckets [split, R) (see bkt_bucket)
        const u32 split,
        const u32 cm, const u32 phi1, const u32 phi5, const u32 off5)   // class mode
{
    const cmap g = {cm, phi1, phi5, off5};
    __local u32 lcnt[BKT_MAX_R], lbase[BKT_MAX_R];
    const u32 lid = get_local_id(0);
    const u32 n = min(counts[slot], cap);
    const __global u64* in = bkt_bucket(entries, entries1, split, cap, slot);
    const u64 chunk = (u64)get_local_size(0) * BKT_E;
    for (u64 c0 = (u64)get_group_id(0) * chunk; c0 < n;
         c0 += (u64)get_num_groups(0) * chunk) {             // uniform per group
        if (lid < BKT_MAX_R) lcnt[lid] = 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        u32 d[BKT_E]; u64 ne[BKT_E]; bool has[BKT_E];
        #pragma unroll
        for (u32 e = 0; e < BKT_E; ++e) {
            has[e] = false; d[e] = 0u; ne[e] = 0UL;
            const u64 i = c0 + (u64)e * get_local_size(0) + lid;
            if (i >= n) continue;
            const u64 ent = bkt_load(in + i);
            const u32 off = (u32)ent;
            const u32 p = (u32)(ent >> 32);
            if (off < (u32)q_odds) {
#ifdef ABL_BKT_NOATOM   // timing only (INVALID map)
                if (off == 0xFFFFFFFFu) prime_map[0] = 0UL;
#else
                map_clear_w((volatile __global u32*)prime_map, off, g);
#endif
            } else if (!is_last) {
                atomic_or(err, BKT_ERR_ENTRY);
            }
            if (is_last) continue;
            const u32 dl = (u32)delta, qf = (u32)q_odds_full;
            u32 noff;
            if (off >= dl) {                     // same hit, also in segment k+1
                d[e] = 1u; noff = off - dl;
            } else {
#if BUCKET_WHEEL30
                // next hit = next multiplier coprime to 30: this hit's
                // multiplier residue from q = q_first + 2*off, then the
                // wheel step (1..3 p, so off + step*p may exceed 2^32)
                const u32 mr = (((qfw + 2u * (off % (WHEEL_W / 2))) % WHEEL_W) *
                                 WHL_INV[p % WHEEL_W]) % WHEEL_W;
                const u32 w = WHL_IDX[mr];
                if (w == WHL_MISS) { atomic_or(err, BKT_ERR_ENTRY); continue; }
                const u64 x = (u64)off + (u64)p * WHL_STEP[w];
                d[e] = bkt_first_seg_rel(x, q_odds_full, delta, inv_delta);
                noff = (u32)(x - (u64)d[e] * delta);
#else
                const u32 x = off + p;           // next odd multiple (off + p < 2^32, host guard)
                d[e] = (x < qf) ? 0u : bkt_div32(x - qf, dl, inv_delta) + 1u;
                noff = x - d[e] * dl;
#endif
            }
            if (d[e] == 0u || d[e] >= R) { atomic_or(err, BKT_ERR_LOOKAHEAD); continue; }
            ne[e] = (u64)noff | ((u64)p << 32);
            has[e] = true;
        }
        bkt_file(entries, entries1, split, counts, cap, R, slot, d, ne, has, lcnt, lbase, err);
        barrier(CLK_LOCAL_MEM_FENCE);            // lcnt reuse next chunk
    }
}

/// Files the first hit of primes [a, b) at segment k_cur.
/// @pre    local size >= R (as bucket_process).
/// @post   each prime has one entry in the ring, or *err has BKT_ERR_LATE /
///         BKT_ERR_LOOKAHEAD / BKT_ERR_OVERFLOW set.
__kernel void bucket_activate(
        __global const u32* restrict primes,     // full base-prime list
        const u32 a,                             // activate primes [a, b)
        const u32 b,
        __global u64* entries,                   // buckets [0, split)
        volatile __global u32* restrict counts,
        const u32 cap,
        const u32 R,
        const u32 k_cur,                         // current segment index
        const u64 q_first0,                      // q_first of segment 0
        const u64 q_odds_full,
        const u64 delta,
        volatile __global u32* restrict err,
        __global u64* entries1,                  // buckets [split, R)
        const u32 split)
{
    __local u32 lcnt[BKT_MAX_R], lbase[BKT_MAX_R];
    const u32 lid = get_local_id(0);
    const u32 n = b - a;
    const u32 slot = k_cur % R;
    const u64 chunk = (u64)get_local_size(0) * BKT_E;
    for (u64 c0 = (u64)get_group_id(0) * chunk; c0 < n;
         c0 += (u64)get_num_groups(0) * chunk) {
        if (lid < BKT_MAX_R) lcnt[lid] = 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        u32 d[BKT_E]; u64 ne[BKT_E]; bool has[BKT_E];
        #pragma unroll
        for (u32 e = 0; e < BKT_E; ++e) {
            has[e] = false; d[e] = 0u; ne[e] = 0UL;
            const u64 i = c0 + (u64)e * get_local_size(0) + lid;
            if (i >= n) continue;
            const u64 p = primes[a + i];
            u64 G = (first_odd_multiple(p, q_first0) - q_first0) >> 1;
#if BUCKET_WHEEL30
            {   // advance to the first multiplier coprime to 30 (<= 2 odd steps)
                u32 mr = (((u32)(q_first0 % (u64)WHEEL_W)) +
                          2u * (u32)(G % (u64)(WHEEL_W / 2))) % WHEEL_W;
                mr = (mr * WHL_INV[(u32)(p % (u64)WHEEL_W)]) % WHEEL_W;
                while (WHL_IDX[mr] == WHL_MISS) { G += p; mr += 2u; if (mr >= WHEEL_W) mr -= WHEEL_W; }
            }
#endif
            // one-time u64 division per prime (activation only)
            const u64 t = (G < q_odds_full) ? 0UL : (G - q_odds_full) / delta + 1UL;
            if (t < k_cur) { atomic_or(err, BKT_ERR_LATE); continue; }
            if (t - k_cur >= R) { atomic_or(err, BKT_ERR_LOOKAHEAD); continue; }
            d[e] = (u32)(t - k_cur);
            ne[e] = (G - t * delta) | (p << 32);
            has[e] = true;
        }
        bkt_file(entries, entries1, split, counts, cap, R, slot, d, ne, has, lcnt, lbase, err);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Validation (GB_BUCKET_VERIFY=1): count words that differ between two maps.
/// @pre    a and b hold >= nwords words, written by earlier kernels.
/// @post   *diff is incremented once per word i < nwords with a[i] != b[i];
///         a and b are unchanged.
__kernel void map_compare(__global const u64* restrict a, __global const u64* restrict b,
                          const u64 nwords, volatile __global u32* restrict diff)
{
    for (u64 i = get_global_id(0); i < nwords; i += get_global_size(0))
        if (a[i] != b[i]) atomic_inc(diff);
}

// ----------------------------------------------------------------------------
// prep_large_fb — compute the compact large-prime first-mark offsets on the
// GPU, one work-item per LARGE_PRIMES_PER_ITEM base primes p >= sieve_split.
//
// The host used to run this computation over ~98M primes per segment
// (~80 ms of wall time and ~393 MB of writes into shared memory, saturating
// the CPU and starving the submission path). Per prime the math is one
// u64 division, trivially GPU-parallel:
//
//   first = p*p                                  if p*p >= q_first
//   first = (first odd multiple of p >= q_first) otherwise
//   fb    = (first - q_first) / 2                if first <= q_last
//   fb    = 0xffffffff (none)                    otherwise
//
// bit i of the map <-> odd q = q_first + 2*i, so (first - q_first) is even
// and the shift is exact. The result must match the host reference exactly;
// tools/debug_fb.cpp compares the kernel output bit-for-bit.
// ----------------------------------------------------------------------------
/// @post   fb_large[j - sieve_split] = first markable bit of prime j, or
///         0xffffffff when none falls in [q_first, q_last].
__kernel void prep_large_fb(
        __global const u32* restrict sieve_primes,
        __global u32* restrict fb_large,    // compact: entry 0 <-> sieve_split
        const u32 sieve_split,
        const u32 sieve_count,
        const u64 q_first,
        const u64 q_last)
{
    const u64 base = (u64)get_global_id(0) * LARGE_PRIMES_PER_ITEM;
    #pragma unroll
    for (u32 lane = 0; lane < LARGE_PRIMES_PER_ITEM; ++lane) {
        const u64 compact = base + lane;
        const u32 j = sieve_split + (u32)compact;
        if (j >= sieve_count) break;         // launch rounds up to WG multiples
        const u64 p = sieve_primes[j];
        const u64 first = first_odd_multiple(p, q_first);
        fb_large[compact] = (first > q_last) ? 0xffffffffU
                                             : (u32)((first - q_first) >> 1);
    }
}

// ----------------------------------------------------------------------------
// check_numbers
//
// Word-level Goldbach verification: one work-item owns WORDS_PER_ITEM = 4
// disjoint 64-even words (four uint64s of the `checked` bitset; see the
// ILP note at the top of the kernel body).
//
// Coordinate math (the key trick):
//   n = p + q,  q odd, p odd
//   bit_n = (n - seg_start) / 2        (n-bitset: one bit per even)
//   bit_q = (q - q_first) / 2          (q-bitset: one bit per odd)
//   n - q = p                          ->  n = q + p
//   bit_n = (q + p - seg_start)/2
//         = (q - q_first)/2 + (p + q_first - seg_start)/2
//         = bit_q + S,   with S = (p + q_first - seg_start) / 2
//
// q_first and seg_start are both odd/even respectively and chosen so that
// (p + q_first - seg_start) is even for every odd p, so S is an exact
// integer shift. For each candidate p the whole q-bitset window is shifted
// by S bits and OR-ed into the accumulator:
//
//   acc |= (qw0 >> S) | (qw1 << (64 - S))
//
// One 64-bit OR covers up to 64 even numbers per candidate — the
// "shifted bitset" bulk-marking formulation.
//
// The special case p = 2 (n = 2 + odd prime) is handled separately.
//
// After all candidates, any word != ~0 contributes its unresolved bit
// positions to the compact leftover list (atomics only on misses).
// ----------------------------------------------------------------------------
/// Single-phase word checker.
/// @pre    map complete for this segment; q_first odd, seg_start even;
///         gb_primes ascending primes (2 is skipped).
/// @post   checked[w] (WRITE_CHECKED) has bit i set iff even seg_start + 2
///         (64 w + i) has a witness among gb_primes; every other valid even is
///         appended to leftovers (count may exceed MAX_LEFTOVERS: host fails
///         closed).
__kernel void check_numbers(
        __global const u64* restrict prime_map, // q segment bitset
        __global u64* restrict checked,         // output: bit i <-> even index i
        __global const u16* restrict gb_primes, // candidate primes p <= 9781
        const u32 gb_count,
        __global u32* restrict leftovers,       // compact exception list
        __global u32* restrict leftover_count,
        const u64 q_first,
        const u64 q_odds,
        const u64 seg_start,                    // first even n of the segment
        const u64 seg_evens                     // number of evens in segment
        )
{
    // ILP variant: each work-item owns WORDS_PER_ITEM disjoint 64-even words.
    // Independent accumulators break the serial OR-chain: while acc0 waits on
    // its loads, acc1..N-1 issue theirs. Adjacent work-items still coalesce.
    const u64 word_idx = (u64)get_global_id(0) * WORDS_PER_ITEM;
    if (word_idx * 64UL >= seg_evens) return;

    const u64 map_words = (q_odds + 63UL) >> 6;

    u64 acc[WORDS_PER_ITEM];
    u64 evens_here[WORDS_PER_ITEM];
    bool all_full = true;
    #pragma unroll
    for (u32 k = 0; k < WORDS_PER_ITEM; ++k) {
        acc[k] = 0UL;
        const u64 wi = word_idx + k;
        evens_here[k] = (wi * 64UL < seg_evens) ? min((u64)64, seg_evens - wi * 64UL) : 0UL;
        if (evens_here[k] < 64UL) all_full = false;
    }

    // ---- odd candidates p = 3, 5, 7, ... ------------------------------------
    // bit_n = (n - seg_start)/2, bit_q = (q - q_first)/2, q = n - p
    //   bit_q = bit_n + T,   T = (seg_start - p - q_first) / 2   (signed!)
    //
    // acc bit j = map bit (base + j) with base = my_n_bit0 + T.
    // The standard two-word extraction (qw0 >> r) | (qw1 << (64-r)) with
    // r = base & 63 handles everything: negative word indices load as 0,
    // which is exactly right for base < 0 (q < q_first cannot be used).
    // Interior-check form: decompose T = Tw*64 + Tr
    // ONCE per candidate => w0 = wi + Tw (one add per word) and r = Tr is
    // IDENTICAL for all work-items and all words of a candidate. In the
    // production geometry (q_first = seg_start - MAX_GB_PRIME, candidates
    // <= MAX_GB_PRIME) T >= 0 for every candidate and map_words =
    // ceil(q_odds/64) = words + 77 covers max(w0)+1 = words + 76, so the
    // bounds guards are provably true and the hot path drops them
    // (geo_safe uniform check; the low-range path where T can go negative
    // keeps the original guarded loop). Semantics unchanged: same words,
    // same funnel, same early exit.
    const long Tbase = (long)seg_start - (long)q_first;
    const u32 pmax = (gb_count > 0) ? gb_primes[gb_count - 1] : 3u;
    const bool geo_safe = (Tbase - (long)pmax) >= 0;
    if (geo_safe) {
        for (u32 j = 0; j < gb_count; ++j) {
            const u32 p = gb_primes[j];
            if (p == 2) continue;   // p=2 only matters for n=4 (CPU fallback)

            const long T  = (Tbase - (long)p) >> 1;   // exact: Tbase-p even
            const u64 Tw = (u64)(T >> 6);             // >= 0 (geo_safe)
            const u32  r  = (u32)(T & 63UL);

            #pragma unroll
            for (u32 k = 0; k < WORDS_PER_ITEM; ++k) {
#if EARLY_EXIT
                if (acc[k] == ~0UL) continue;      // this word fully verified
#endif
                const u64 w0 = word_idx + Tw + k;  // = wi + Tw, in bounds
                const u64 qw0 = prime_map[w0];
                const u64 qw1 = prime_map[w0 + 1];

                acc[k] |= r ? ((qw0 >> r) | (qw1 << (64UL - r))) : qw0;
            }
#if EARLY_EXIT
            if (all_full) {
                bool done = true;
                #pragma unroll
                for (u32 k = 0; k < WORDS_PER_ITEM; ++k) done &= (acc[k] == ~0UL);
                if (done) break;                   // all evens verified
            }
#endif
        }
    } else {
        for (u32 j = 0; j < gb_count; ++j) {
            const u32 p = gb_primes[j];
            if (p == 2) continue;   // p=2 only matters for n=4 (CPU fallback)

            const long T = ((long)seg_start - (long)p - (long)q_first) / 2;

            #pragma unroll
            for (u32 k = 0; k < WORDS_PER_ITEM; ++k) {
#if EARLY_EXIT
                if (acc[k] == ~0UL) continue;      // this word fully verified
#endif
                const long base = (long)((word_idx + k) * 64UL) + T;
                const long w0   = base >> 6;       // arithmetic shift, may be -1
                const u32  r    = (u32)(base & 63UL);

                const u64 qw0 = (w0 >= 0 && (u64)w0       < map_words) ? prime_map[w0]     : 0UL;
                const u64 qw1 = (w0 + 1 >= 0 && (u64)(w0 + 1) < map_words) ? prime_map[w0 + 1] : 0UL;

                acc[k] |= r ? ((qw0 >> r) | (qw1 << (64UL - r))) : qw0;
            }
#if EARLY_EXIT
            if (all_full) {
                bool done = true;
                #pragma unroll
                for (u32 k = 0; k < WORDS_PER_ITEM; ++k) done &= (acc[k] == ~0UL);
                if (done) break;                   // all evens verified
            }
#endif
        }
    }

    // ---- mask partial word, write, report leftovers --------------------------
    #pragma unroll
    for (u32 k = 0; k < WORDS_PER_ITEM; ++k) {
        const u64 wi = word_idx + k;
        if (evens_here[k] == 0UL) break;       // past segment end (words are ordered)
        u64 a = acc[k];
        if (evens_here[k] < 64UL) a &= (1UL << evens_here[k]) - 1UL;
#if WRITE_CHECKED
        checked[wi] = a;
#endif

#if REPORT_LEFTOVERS
        if (a != ~0UL) {
            // Report each unresolved even index (rare; atomics only on misses).
            u64 miss = ~a;
            if (evens_here[k] < 64UL) miss &= (1UL << evens_here[k]) - 1UL;
            while (miss) {
                const u32 b = (u32)ctz(miss);
                miss &= miss - 1UL;
                const u32 slot = atom_inc(leftover_count);
                if (slot < MAX_LEFTOVERS) leftovers[slot] = (u32)(wi * 64UL + b);
            }
        }
#endif
    }

}


// ----------------------------------------------------------------------------
// check_numbers_w32 — 32-bit register-window checker (v1.7.0).
//
// A register-window variant keeping the 5 map words in registers (~0.1
// loads per candidate instead of 8) gained only 7%: the checker is
// ALU-bound, not load-bound: Xe2
// has no native 64-bit integer shifts, so every 64-bit funnel
// (m0 >> r) | (m1 << (64 - r)) and the 64-bit T arithmetic expand into
// multi-instruction emulation. Here the SAME 256 evens per item are held as
// 8 native u32 words (little-endian: u64 word w == u32 words 2w, 2w+1, so
// the map and checked[] layouts are unchanged), the map window is 9 u32
// registers sliding one u32 word at a time, and T = (Tbase - p) / 2 is
// 32-bit (Tbase = seg_start - q_first = MAX_GB_PRIME in production). Each
// funnel is 2 native shifts + 2 ORs.
// Same candidates, early exit, leftovers and checked[] contract as
// check_numbers. Low-range geometry (T may be negative) and any Tbase that
// does not fit comfortably in 32 bits take the guarded 64-bit loop.
// ----------------------------------------------------------------------------
#ifndef W32_PREFETCH
#define W32_PREFETCH 0
#endif
#ifndef W32_EXIT_EVERY          // early-exit test every N candidates (power of 2)
#define W32_EXIT_EVERY 1
#endif
/// @pre    / @post as check_numbers (32-bit register-window variant).
__kernel void check_numbers_w32(
        __global const u64* restrict prime_map,
        __global u64* restrict checked,
        __global const u16* restrict gb_primes,
        const u32 gb_count,
        __global u32* restrict leftovers,
        __global u32* restrict leftover_count,
        const u64 q_first,
        const u64 q_odds,
        const u64 seg_start,
        const u64 seg_evens)
{
    const u64 word_idx = (u64)get_global_id(0) * WORDS_PER_ITEM;
    if (word_idx * 64UL >= seg_evens) return;
    const u64 map_words = (q_odds + 63UL) >> 6;

    u64 valid[WORDS_PER_ITEM];
    u32 acc[2 * WORDS_PER_ITEM];
    #pragma unroll
    for (u32 k = 0; k < WORDS_PER_ITEM; ++k) {
        const u64 wi = word_idx + k;
        const u64 ev = (wi * 64UL < seg_evens) ? min((u64)64, seg_evens - wi * 64UL) : 0UL;
        valid[k] = (ev >= 64UL) ? ~0UL : ((1UL << ev) - 1UL);
        acc[2 * k]     = ~(u32)valid[k];         // past-the-end evens pre-set
        acc[2 * k + 1] = ~(u32)(valid[k] >> 32);
    }

    const long Tbase = (long)seg_start - (long)q_first;
    const u32 pmax = (gb_count > 0) ? gb_primes[gb_count - 1] : 3u;
    const bool fast = (Tbase - (long)pmax) >= 0 && Tbase < (1L << 30);   // uniform
    if (fast) {
        const __global u32* map32 = (const __global u32*)prime_map;
        const u64 map32_last = 2UL * map_words - 1UL;
        const u64 w32_0 = 2UL * word_idx;        // my first u32 word
        const int tb = (int)Tbase;
        int tw = ((tb - 3) >> 1) >> 5;           // max over candidates (p >= 3)
        u64 wb = w32_0 + (u64)tw;
        u32 m[9];
        #pragma unroll
        for (u32 i = 0; i < 9; ++i) m[i] = map32[min(wb + i, map32_last)];
#if W32_PREFETCH
        int p_next = gb_primes[0];
#endif
        for (u32 j = 0; j < gb_count; ++j) {
#if W32_PREFETCH
            const int p = p_next;                // load of p is off the critical path
            p_next = gb_primes[min(j + 1u, gb_count - 1u)];
#else
            const int p = gb_primes[j];
#endif
            if (p == 2) continue;
            const int T = (tb - p) >> 1;         // >= 0 (fast => geo_safe)
            const int twj = T >> 5;
            while (tw > twj) {                   // uniform: slide window down
                --tw; --wb;
                #pragma unroll
                for (u32 i = 8; i > 0; --i) m[i] = m[i - 1];
                m[0] = map32[wb];
            }
            const u32 r = (u32)T & 31u;
            if (r) {
                const u32 l = 32u - r;
                #pragma unroll
                for (u32 i = 0; i < 8; ++i) acc[i] |= (m[i] >> r) | (m[i + 1] << l);
            } else {
                #pragma unroll
                for (u32 i = 0; i < 8; ++i) acc[i] |= m[i];
            }
#if EARLY_EXIT
            if ((j & (W32_EXIT_EVERY - 1u)) == W32_EXIT_EVERY - 1u &&
                (acc[0] & acc[1] & acc[2] & acc[3] & acc[4] & acc[5] & acc[6] & acc[7]) == ~0u)
                break;
#endif
        }
    } else {
        u64 a64[WORDS_PER_ITEM];
        #pragma unroll
        for (u32 k = 0; k < WORDS_PER_ITEM; ++k) a64[k] = ~valid[k];
        for (u32 j = 0; j < gb_count; ++j) {
            const u32 p = gb_primes[j];
            if (p == 2) continue;
            const long T = ((long)seg_start - (long)p - (long)q_first) / 2;
            #pragma unroll
            for (u32 k = 0; k < WORDS_PER_ITEM; ++k) {
                if (a64[k] == ~0UL) continue;
                const long base = (long)((word_idx + k) * 64UL) + T;
                const long w0   = base >> 6;
                const u32  r    = (u32)(base & 63UL);
                const u64 qw0 = (w0 >= 0 && (u64)w0       < map_words) ? prime_map[w0]     : 0UL;
                const u64 qw1 = (w0 + 1 >= 0 && (u64)(w0 + 1) < map_words) ? prime_map[w0 + 1] : 0UL;
                a64[k] |= r ? ((qw0 >> r) | (qw1 << (64UL - r))) : qw0;
            }
#if EARLY_EXIT
            if ((a64[0] & a64[1] & a64[2] & a64[3]) == ~0UL) break;
#endif
        }
        #pragma unroll
        for (u32 k = 0; k < WORDS_PER_ITEM; ++k) {
            acc[2 * k] = (u32)a64[k];
            acc[2 * k + 1] = (u32)(a64[k] >> 32);
        }
    }

    #pragma unroll
    for (u32 k = 0; k < WORDS_PER_ITEM; ++k) {
        if (valid[k] == 0UL) break;              // past segment end (ordered)
        const u64 wi = word_idx + k;
        const u64 a = ((u64)acc[2 * k] | ((u64)acc[2 * k + 1] << 32));
#if WRITE_CHECKED
        checked[wi] = a & valid[k];
#endif
#if REPORT_LEFTOVERS
        u64 miss = ~a & valid[k];
        while (miss) {
            const u32 b = (u32)ctz(miss);
            miss &= miss - 1UL;
            const u32 slot = atom_inc(leftover_count);
            if (slot < MAX_LEFTOVERS) leftovers[slot] = (u32)(wi * 64UL + b);
        }
#endif
    }
}

// ----------------------------------------------------------------------------
// Two-phase checker (v1.7.0): check_phase_a + check_phase_b.
//
// Measured at 4e18 (fraction of evens still unresolved after the first K
// candidates): K=64 6.5%, 96 2.0%, 128 0.62%, 192 0.06%, 256 0.007%,
// 384 ~1e-6. Word-level checking resolves 64 evens per op, but a 256-even
// item runs until its HARDEST even is resolved (~300 candidates on average
// per subgroup), so most of the single-phase checker's time is spent on a
// fraction of a percent of the evens.
//
// Phase A = check_numbers_w32 capped at the first cand_cap candidates. Every
// even still unresolved is appended to `pending` (segment-local even index;
// one atomic per subgroup via scan/reduce). checked[] gets the partial
// result.
// Phase B = one work-item per pending even (grid-stride): continues with
// candidates [cand_cap, gb_count) one bit probe each, fully bounds-guarded
// (no geometry assumptions), sets the even's checked[] bit on success, and
// appends evens that no candidate resolves to the normal leftover list (CPU
// Miller-Rabin fallback), exactly like the single-phase checkers.
// The union covers the same candidates in the same order, so outputs are
// identical to check_numbers (validated by tools/debug_check_hi).
// pending_count may exceed pending_cap: the host fails closed if it does.
// ----------------------------------------------------------------------------
// Phase A item width in 64-bit words (2*A_W64 u32 words, 64*A_W64 evens);
// MUST match the host's CHECK_PHASE_A_W64. Wider items spread the fixed
// per-candidate work (candidate load, shift setup, run test, exit test)
// over more words. check_s per 2e10 window: 2 -> 0.117, 4 -> 0.0905,
// 8 -> 0.085, 12 -> 0.073, 16 -> 0.065, 20 -> 0.069, 24 -> 0.075,
// 32 -> 0.163 (register spills). At 16 the compiler picks SIMD16.
// Twin checker (v1.10.0). With twin = 1, check_phase_a reads the map as
// the TWIN map, bit j = map bit j | map bit j - 1 ("q or q - 2 is prime"),
// and its candidates are the LOWER members p of twin pairs (p, p + 2 both
// prime): the window at T_p then holds map[i + T_p] | map[i + T_p - 1], i.e.
// candidates p and p + 2 (T_{p+2} = T_p - 1) in ONE funnel. Every set bit
// is the OR of two genuine map bits (bits outside [0, q_odds) are 0 in the
// map; bit -1 is taken as 0), so a resolved even always has a genuine
// witness. Phase A starts with ~0.07% of its funnels spent per resolved
// even instead of every item running to the cap (P(all 1024 evens of an
// item resolved after 160 single candidates) is only ~13%). The twin words
// are formed as the register window is loaded (once per slide, not per
// candidate), so no separate pass or buffer.
/// @return map32[k], or with twin set the twin word (bit j = bit j | bit j - 1).
inline u32 tw32(const __global u32* map32, const u64 k, const u32 twin)
{
    const u32 m = map32[k];
    if (!twin) return m;
    const u32 below = (k > 0) ? (map32[k - 1] >> 31) : 0u;
    return m | (m << 1) | below;
}
/// @return map word w (0 outside [0, map_words)), twin-folded when twin is set.
inline u64 tw64(const __global u64* map, const long w, const u64 map_words, const u32 twin)
{
    const u64 m = (w >= 0 && (u64)w < map_words) ? map[w] : 0UL;
    if (!twin) return m;
    const u64 below = (w - 1 >= 0 && (u64)(w - 1) < map_words) ? (map[w - 1] >> 63) : 0UL;
    return m | (m << 1) | below;
}

// Phase-A item width in 64-bit words (all word checkers; MUST match
// gb::CHECK_PHASE_A_W64). v1.13.0: 14 (896 evens per item): the mod-30
// checker spilled ~26 registers at 16 (12: spill-free, -1.0% total; 14 spills
// ~12 but -0.28% / -0.22% more on long / long-end; 8: +0.3%, 10: +0.04%).
#ifndef C30_UDIV
#define C30_UDIV 1              // check30_a: U = U0 - floor(p / 30) by multiply-shift (+0.1%)
#endif
#ifndef C30_ROT
#define C30_ROT 1               // check30_a funnel by rotate + bitselect, 3 ops per word not 4 (+0.45%)
#endif
#ifndef A_W64
#define A_W64 12               // 14 -> 12 after C30_ROT (spill-free, +0.2%)
#endif
/// Phase A of the two-phase checker (first cand_cap candidates).
/// @pre    as check_numbers; with twin = 1, gb_primes lists lower twin members.
/// @post   checked[] holds the partial result; every unresolved valid even is
///         in pending (count may exceed pending_cap: host fails closed).
#ifdef A_SIMD
__attribute__((intel_reqd_sub_group_size(A_SIMD)))
#endif
__kernel void check_phase_a(
        __global const u64* restrict prime_map,
        __global u64* restrict checked,
        __global const u16* restrict gb_primes,
        const u32 gb_count,
        __global u32* restrict pending,
        __global u32* restrict pending_count,
        const u64 q_first,
        const u64 q_odds,
        const u64 seg_start,
        const u64 seg_evens,
        const u32 cand_cap,
        const u32 pending_cap,
        const u32 twin)                     // 1: twin checker (see tw32 / tw64)
{
    const u64 word_idx = (u64)get_global_id(0) * A_W64;
    const bool live = word_idx * 64UL < seg_evens;   // no early return: subgroup ops below
    const u64 map_words = (q_odds + 63UL) >> 6;
    const u32 n_cand = min(gb_count, cand_cap);

    u64 valid[A_W64];
    u32 acc[2 * A_W64];
    #pragma unroll
    for (u32 k = 0; k < A_W64; ++k) {
        const u64 wi = word_idx + k;
        const u64 ev = (wi * 64UL < seg_evens) ? min((u64)64, seg_evens - wi * 64UL) : 0UL;
        valid[k] = (ev >= 64UL) ? ~0UL : ((1UL << ev) - 1UL);
        acc[2 * k]     = ~(u32)valid[k];
        acc[2 * k + 1] = ~(u32)(valid[k] >> 32);
    }

    const long Tbase = (long)seg_start - (long)q_first;
    const u32 pmax = (gb_count > 0) ? gb_primes[gb_count - 1] : 3u;
    const bool fast = (Tbase - (long)pmax) >= 0 && Tbase < (1L << 30);   // uniform
    if (live && fast) {
        const __global u32* map32 = (const __global u32*)prime_map;
        const u64 map32_last = 2UL * map_words - 1UL;
        const int tb = (int)Tbase;
        int tw = ((tb - 3) >> 1) >> 5;           // max over candidates (p >= 3)
        u64 wb = 2UL * word_idx + (u64)tw;
        u32 m[2 * A_W64 + 1];
        #pragma unroll
        for (u32 i = 0; i <= 2 * A_W64; ++i) m[i] = tw32(map32, min(wb + i, map32_last), twin);
        // Candidates are processed in runs that share one word offset
        // tw = T >> 5 (T descends with p, so each run is contiguous): the
        // window is slid once per run in the outer loop and stays FIXED in
        // registers inside the run. The single-loop form carried the 9
        // window registers through every iteration and compiled to ~75
        // register moves per candidate. The funnel is branch-free:
        // (m >> r) | ((m' << (31 - r)) << 1) equals the 32-bit window at bit
        // offset r for every r in [0, 31] (r = 0 gives m exactly), so no
        // per-word select for the r == 0 case either.
        u32 j = 0;
        int T = (tb - (int)gb_primes[0]) >> 1;
        bool done = false;
        while (j < n_cand && !done) {
            const int twj = T >> 5;
            while (tw > twj) {                   // uniform: slide window down
                --tw; --wb;
                #pragma unroll
                for (u32 i = 2 * A_W64; i > 0; --i) m[i] = m[i - 1];
                m[0] = tw32(map32, wb, twin);
            }
            // run: every candidate with this word offset
            do {
                const u32 r = (u32)T & 31u;
                const u32 l = 31u - r;
                if (gb_primes[j] != 2) {
                    #pragma unroll
                    for (u32 i = 0; i < 2 * A_W64; ++i)
                        acc[i] |= (m[i] >> r) | ((m[i + 1] << l) << 1);
                }
#if EARLY_EXIT
                u32 all = acc[0];
                #pragma unroll
                for (u32 i = 1; i < 2 * A_W64; ++i) all &= acc[i];
                if (all == ~0u) {
                    done = true;
                    break;
                }
#endif
                if (++j >= n_cand) break;
                T = (tb - (int)gb_primes[j]) >> 1;
            } while ((T >> 5) == twj);
        }
    } else if (live) {
        u64 a64[A_W64];
        #pragma unroll
        for (u32 k = 0; k < A_W64; ++k) a64[k] = ~valid[k];
        for (u32 j = 0; j < n_cand; ++j) {
            const u32 p = gb_primes[j];
            if (p == 2) continue;
            const long T = ((long)seg_start - (long)p - (long)q_first) / 2;
            #pragma unroll
            for (u32 k = 0; k < A_W64; ++k) {
                if (a64[k] == ~0UL) continue;
                const long base = (long)((word_idx + k) * 64UL) + T;
                const long w0   = base >> 6;
                const u32  r    = (u32)(base & 63UL);
                const u64 qw0 = tw64(prime_map, w0, map_words, twin);
                const u64 qw1 = tw64(prime_map, w0 + 1, map_words, twin);
                a64[k] |= r ? ((qw0 >> r) | (qw1 << (64UL - r))) : qw0;
            }
#if EARLY_EXIT
            u64 all64 = a64[0];
            #pragma unroll
            for (u32 k = 1; k < A_W64; ++k) all64 &= a64[k];
            if (all64 == ~0UL) break;
#endif
        }
        #pragma unroll
        for (u32 k = 0; k < A_W64; ++k) {
            acc[2 * k] = (u32)a64[k];
            acc[2 * k + 1] = (u32)(a64[k] >> 32);
        }
    }

    // checked[] partial result + count of unresolved evens in this item.
    u32 c = 0;
    #pragma unroll
    for (u32 k = 0; k < A_W64; ++k) {
        if (!live || valid[k] == 0UL) continue;
        const u64 a = ((u64)acc[2 * k] | ((u64)acc[2 * k + 1] << 32));
#if WRITE_CHECKED
        checked[word_idx + k] = a & valid[k];
#endif
        c += (u32)popcount(~a & valid[k]);
    }
    // One atomic per subgroup reserves space for all of its misses.
    const u32 off = sub_group_scan_exclusive_add(c);
    const u32 tot = sub_group_reduce_add(c);
    u32 base = 0;
    if (get_sub_group_local_id() == 0 && tot) base = atomic_add(pending_count, tot);
    base = sub_group_broadcast(base, 0) + off;
    if (c) {
        #pragma unroll
        for (u32 k = 0; k < A_W64; ++k) {
            if (valid[k] == 0UL) continue;
            const u64 a = ((u64)acc[2 * k] | ((u64)acc[2 * k + 1] << 32));
            u64 miss = ~a & valid[k];
            while (miss) {
                const u32 b = (u32)ctz(miss);
                miss &= miss - 1UL;
                if (base < pending_cap) pending[base] = (u32)((word_idx + k) * 64UL + b);
                ++base;
            }
        }
    }
}

#ifndef PROBE_BATCH
#define PROBE_BATCH 4            // v1.13.0 (mod-30 phase B): 4 -0.1% vs 8; 16 +0.15%
#endif
/// Phase B: candidates [cand_cap, gb_count) per pending even.
/// @post   each pending even gets its checked[] bit or a leftover entry.
__kernel void check_phase_b(
        __global const u64* restrict prime_map,
        __global u64* restrict checked,
        __global const u16* restrict gb_primes,
        const u32 gb_count,
        __global const u32* restrict pending,
        __global const u32* restrict pending_count,
        __global u32* restrict leftovers,
        __global u32* restrict leftover_count,
        const u64 q_first,
        const u64 q_odds,
        const u64 seg_start,
        const u32 cand_cap,
        const u32 pending_cap,
        __global u64* restrict ridge,                 // Goldbach ridge appends
        const u32 ridge_thresh)                       // record p >= this; ~0u = off
{
    const __global u32* map32 = (const __global u32*)prime_map;
    const u32 n = min(*pending_count, pending_cap);
    const u32 j0 = min(gb_count, cand_cap);
    const long Tbase = (long)seg_start - (long)q_first;
    for (u32 i = get_global_id(0); i < n; i += get_global_size(0)) {
        const u32 e = pending[i];                // segment-local even index
        bool ok = false;
        u32 rp = 0u;                             // smallest witness p (ridge)
        // Probes are independent, so they are issued PROBE_BATCH at a time
        // (latency-bound: one dependent load chain per lane otherwise).
        for (u32 j = j0; j < gb_count && !ok; j += PROBE_BATCH) {
            u32 hit = 0u;
            #pragma unroll
            for (u32 u = 0; u < PROBE_BATCH; ++u) {
                if (j + u >= gb_count) break;
                const long p = gb_primes[j + u];
                if (p == 2) continue;
                const long qb = (long)e + ((Tbase - p) >> 1);   // q bit (Tbase - p even)
                if (qb < 0 || (u64)qb >= q_odds) continue;
                hit |= ((map32[qb >> 5] >> (qb & 31)) & 1u) << u;   // bit u = this probe
            }
            ok = hit != 0u;
            if (ok) rp = (u32)gb_primes[j + (u32)ctz(hit)];
        }
        if (ok) {
            if (rp >= ridge_thresh) {            // ascending list: first hit = ridge
                const u32 slot = atomic_add((volatile __global u32*)ridge, 1u);
                if (slot < RIDGE_CAP) {
                    ridge[1 + 2u * slot] = seg_start + 2UL * (u64)e;
                    ridge[2 + 2u * slot] = rp;
                }
            }
            atomic_or((volatile __global u32*)checked + (e >> 5), 1u << (e & 31u));
        } else {
            const u32 slot = atom_inc(leftover_count);
            if (slot < MAX_LEFTOVERS) leftovers[slot] = e;
        }
    }
}

// ============================================================================
// Class-split checker (v1.11.0): split_map + check_split_a + check_split_b.
//
// Why: every q = n - p divisible by 3 is composite (q > 3), and q = n - p is
// divisible by 3 exactly when n = p (mod 3). The twin checker tests all
// 1024 evens of an item against every candidate, so for the evens with
// n = 1 or 2 (mod 3) — the hard ones, which set the item's funnel count —
// only ONE of a twin funnel's two tests can ever succeed. Here the evens are
// grouped by n mod 3 and the map by q mod 6, so every test of a funnel is
// useful:
//   n = 0 (mod 3): candidates p = 1 (mod 6), q = 5 (mod 6)   (S5, list L1)
//   n = 2 (mod 3): candidates p = 1 (mod 6), q = 1 (mod 6)   (S1, list L1)
//   n = 1 (mod 3): candidates p = 5 (mod 6), q = 5 (mod 6)   (S5, list L5)
// In the split maps S1 / S5 (bit t <-> the t-th odd q of that residue class
// mod 6, i.e. dense bit 3t + phi), consecutive bits are q and q + 6, and a
// class's evens are n, n + 6, ...: the same shifted-bitset funnel applies.
// A funnel ORs several map shifts (pattern mask, bit d <-> q - 6d, i.e.
// candidate p + 6d), so one funnel tests e.g. p, p + 6, p + 12 when all three
// are prime (sexy-prime triplets, then pairs): measured on real maps near
// 4e18, 35 such funnels leave 0.1% of the evens pending vs 100 twin funnels.
// Every set bit is still a genuine witness (the OR of map bits of q - 6d
// with every p + 6d prime <= MAX_GB_PRIME). Phase B probes, per class, the
// candidates phase A did not test — useful residues first, then the
// futile ones (they can only hit q = 3, i.e. at the bottom of the range, or
// on synthetic test maps), so the final checked set and leftover set are
// exactly those of a per-even scan over all candidates.
// ============================================================================

// Bits 0, 3, 6, ..., 30 of x packed into bits 0..10.
inline u32 cmp3(u32 x)
{
    x &= 0x49249249u;
    x = (x | (x >> 2))  & 0xC30C30C3u;
    x = (x | (x >> 4))  & 0x0F00F00Fu;
    x = (x | (x >> 8))  & 0xFF0000FFu;
    x = (x | (x >> 16)) & 0x7FFu;
    return x;
}
// Bits phi, phi + 3, ..., phi + 93 of the 128-bit x0..x3 (phi < 3).
inline u32 split_word(const u32 x0, const u32 x1, const u32 x2, const u32 x3, const u32 phi)
{
    const u32 l = 31u - phi;                        // funnel, exact for phi = 0
    const u32 y0 = (x0 >> phi) | ((x1 << l) << 1);
    const u32 y1 = (x1 >> phi) | ((x2 << l) << 1);
    const u32 y2 = (x2 >> phi) | ((x3 << l) << 1);
    return cmp3(y0) | (cmp3(y1 >> 1) << 11) | (cmp3(y2 >> 2) << 22);
}

// S1[s] / S5[s] = dense bits 96 s + phi1 / phi5 at stride 3 (bits past the
// map are 0; the map's own tail past q_odds is 0). Memory-bound (0.049 ms per
// segment; a 4-words-per-lane vload4/vstore4 form measured 0.048).
/// @pre    phi1, phi5 < 3; s_words * 3 covers the map.
/// @post   s1[s] / s5[s] = split words of the dense map (0 past its end).
__kernel void split_map(__global const u32* restrict map32, const u32 map32_words,
                        __global u32* restrict s1, __global u32* restrict s5,
                        const u32 s_words, const u32 phi1, const u32 phi5)
{
    const u32 s = get_global_id(0);
    if (s >= s_words) return;
    const u32 w = 3u * s;
    const u32 x0 = (w      < map32_words) ? map32[w]      : 0u;
    const u32 x1 = (w + 1u < map32_words) ? map32[w + 1u] : 0u;
    const u32 x2 = (w + 2u < map32_words) ? map32[w + 2u] : 0u;
    const u32 x3 = (w + 3u < map32_words) ? map32[w + 3u] : 0u;
    s1[s] = split_word(x0, x1, x2, x3, phi1);
    s5[s] = split_word(x0, x1, x2, x3, phi5);
}

// merge30 (cm = 2): the checker's S1 / S5 (bit t <-> q = Q + 6t, Q the first
// q >= q_first of that class mod 6) from the mod-30 class arrays. Of the five
// residues Q + 6j (j = t mod 5) four are coprime to 30 (class kmap nibble j)
// and one is divisible by 5 (nibble 15: bit 0); Q + 6j is the first q of its
// class, so S bit 5i + j = bit i of class array kmap(j).
/// @pre    kmap1 / kmap5 nibble tables from the host (15 = divisible by 5).
/// @post   s1 / s5 hold the mod-6 split maps rebuilt from the mod-30 arrays,
///         with q = 5 restored at bit q5_t of S5.
__kernel void merge30(__global const u32* restrict cmap32, const u32 cw32,
                      __global u32* restrict s1, __global u32* restrict s5,
                      const u32 s_words, const u32 kmap1, const u32 kmap5,
                      const u32 q5_t)                  // S5 bit of q = 5 (prime, not stored), ~0: none
{
    // One lane = 5 output words (160 bits = S bits t = 160 L + 5i + j,
    // i < 32): exactly the aligned word L of each of the four class arrays
    // (bit i -> bit 5i + j). The stride-5 spread of a nibble is carry-free:
    // ((x & 15) * 0x1111) & 0x8421 puts bit b at 5b.
    const u32 per = (s_words + 4u) / 5u;               // lanes per S array
    const u32 gid = get_global_id(0);
    const u32 which = gid >= per ? 1u : 0u;
    const u32 L = gid - which * per;
    if (L >= per) return;
    const u32 kmap = which ? kmap5 : kmap1;
    u32 o[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    #pragma unroll
    for (u32 j = 0; j < 5u; ++j) {
        const u32 k = (kmap >> (4u * j)) & 15u;
        if (k == 15u) continue;                        // residue divisible by 5
        const u32 x = (L < cw32) ? cmap32[(u64)k * cw32 + L] : 0u;
        #pragma unroll
        for (u32 n = 0; n < 8u; ++n) {
            const u32 piece = (((x >> (4u * n)) & 15u) * 0x1111u) & 0x8421u;   // 16 bits
            const u32 pos = 20u * n + j;
            const u32 lo = pos >> 5, sh = pos & 31u;
            o[lo] |= piece << sh;
            if (sh > 16u) o[lo + 1u] |= piece >> (32u - sh);
        }
    }
    __global u32* out = which ? s5 : s1;
    const u32 w0 = 5u * L;
    if (which && q5_t / 160u == L) o[(q5_t % 160u) >> 5] |= 1u << (q5_t & 31u);
    #pragma unroll
    for (u32 q = 0; q < 5u; ++q)
        if (w0 + q < s_words) out[w0 + q] = o[q];
}

#ifndef SPLIT_EXIT
#define SPLIT_EXIT 0             // per-funnel all-resolved test: never pays (P(all 16 lanes x
                                 // 1024 evens resolved before the cap) ~ 1e-7); -0.8..2.3 ms/2e10
#endif
#ifndef SPLIT_MAXD
#define SPLIT_MAXD 2             // largest pattern shift (units of 6; host-checked)
#endif
// Pattern OR of split-map word a (b = the word below it): bit j = OR over
// d in mask of S bit 32 w + j - d (bit 0 of mask = d 0 is always set).
/// @pre    bit 0 of mask set; shifts d <= SPLIT_MAXD < 32.
inline u32 split_orp(const u32 a, const u32 b, const u32 mask)
{
    u32 v = a;
    #pragma unroll
    for (u32 d = 1; d <= SPLIT_MAXD; ++d)
        if ((mask >> d) & 1u) v |= (a << d) | (b >> (32u - d));
    return v;
}
/// @return S[w] when 0 <= w < n, else 0.
inline u32 split_ld(const __global u32* S, const int w, const u32 n)
{
    return ((u32)w < n) ? S[w] : 0u;                      // w < 0 wraps: out of range
}

// Phase A. One work-item = 1024 evens of ONE class (n mod 3 = c), the
// evens e = e0_c + 3k, k in [1024 blk, +1024). Work-items are laid out
// class-major with items_pad (a multiple of the work-group size) per class,
// so every subgroup has one class, one candidate list and one uniform
// funnel offset. meta (host-built, per list L1 at 0 and L5 at 16):
//   [n_stages, (begin, end, mask) x n_stages] indexes into cands.
// Funnel for candidate p: S bit U + k with U = (v_c - p) / 6 (exact; v_c =
// Tbase + 2 (e0_c - phi) >= p + 6 * max shift, host-checked).
/// @pre    plan from make_split_plan; v_c - p divisible by 6 for every
///         candidate (host split_segment ok); items_pad multiple of the WG size.
/// @post   as check_phase_a, per class (pending holds segment even indices).
__kernel void check_split_a(
        __global const u32* restrict s1,
        __global const u32* restrict s5,
        const u32 s_words,
        __global u64* restrict checked,         // class c at c * chk_stride words
        const u32 chk_stride,
        __global const u16* restrict cands,
        __global const u32* restrict meta,
        __global u32* restrict pending,
        __global u32* restrict pending_count,
        const u32 pending_cap,
        const u32 items_pad,
        const u32 ec0, const u32 ec1, const u32 ec2,   // evens per class
        const u32 e00, const u32 e01, const u32 e02,   // first even index per class
        const int v0, const int v1, const int v2,
        const u32 write_checked)                       // 0 in production (nothing reads it;
                                                       // the 15.6 MB/segment store cost ~1%)
{
    const u32 gid = get_global_id(0);
    const u32 c = gid / items_pad;                     // uniform per work-group
    const u32 blk = gid - c * items_pad;
    const u32 ec = (c == 0) ? ec0 : (c == 1) ? ec1 : ec2;
    const u32 e0 = (c == 0) ? e00 : (c == 1) ? e01 : e02;
    const int vc = (c == 0) ? v0 : (c == 1) ? v1 : v2;
    const __global u32* S = (c == 2) ? s1 : s5;
    const u32 mo = (c == 1) ? 16u : 0u;                // L5 for n = 1 (mod 3)
    const u32 k0 = blk * (A_W64 * 64u);
    const bool live = c < 3u && k0 < ec;               // no early return: subgroup ops below

    u32 acc[2 * A_W64];
    #pragma unroll
    for (u32 i = 0; i < 2 * A_W64; ++i) {
        const u32 kb = k0 + 32u * i;
        const u32 n = (live && kb < ec) ? min(32u, ec - kb) : 0u;
        acc[i] = (n >= 32u) ? 0u : ~((1u << n) - 1u);  // invalid bits count as resolved
    }

    if (live) {
        const u32 ns = meta[mo];
        bool done = false;
        for (u32 st = 0; st < ns && !done; ++st) {
            u32 j = meta[mo + 1u + 3u * st];
            const u32 je = meta[mo + 2u + 3u * st];
            const u32 mask = meta[mo + 3u + 3u * st];
            if (j >= je) continue;
            int U = (vc - (int)cands[j]) / 6;        // exact (host): may be < 0 at the range bottom
            int tw = (U >= 0) ? (U >> 5) : -((31 - U) >> 5);   // floor(U / 32)
            int wb = (int)(k0 >> 5) + tw;
            u32 m[2 * A_W64 + 1];
            #pragma unroll
            for (u32 i = 0; i <= 2 * A_W64; ++i) m[i] = split_ld(S, wb + i, s_words);
            // Raw words below the window, loaded two slides ahead: in the
            // sparse pattern lists nearly every funnel slides the window, and
            // a load issued at the slide itself stalls the next funnel.
            u32 r1 = split_ld(S, wb - 1, s_words);
            u32 r2 = split_ld(S, wb - 2, s_words);
            u32 r3 = split_ld(S, wb - 3, s_words);
            u32 r4 = split_ld(S, wb - 4, s_words);
            if (mask != 1u) {                          // uniform; in place, top down
                #pragma unroll
                for (u32 i = 2 * A_W64; i > 0; --i) m[i] = split_orp(m[i], m[i - 1], mask);
                m[0] = split_orp(m[0], r1, mask);
            }
            // One flat loop over the stage's candidates (a nested slide loop
            // compiled to ~100 register moves per slide): the window slides
            // by one word in place (33 moves, next raw words preloaded), or
            // is reloaded when the offset jumps further.
            for (;;) {
                const int twj = (U >= 0) ? (U >> 5) : -((31 - U) >> 5);
                if (tw - twj == 1) {                   // uniform
                    tw = twj; --wb;
                    #pragma unroll
                    for (u32 i = 2 * A_W64; i > 0; --i) m[i] = m[i - 1];
                    m[0] = split_orp(r1, r2, mask);
                    r1 = r2; r2 = r3; r3 = r4;
                    r4 = split_ld(S, wb - 4, s_words);
                } else if (tw - twj == 2) {            // sparse lists: two words at once
                    tw = twj; wb -= 2;
                    #pragma unroll
                    for (u32 i = 2 * A_W64; i > 1; --i) m[i] = m[i - 2];
                    m[1] = split_orp(r1, r2, mask);
                    m[0] = split_orp(r2, r3, mask);
                    r1 = r3; r2 = r4;
                    r3 = split_ld(S, wb - 3, s_words);
                    r4 = split_ld(S, wb - 4, s_words);
                } else if (tw != twj) {
                    wb -= tw - twj; tw = twj;
                    #pragma unroll
                    for (u32 i = 0; i <= 2 * A_W64; ++i) m[i] = split_ld(S, wb + i, s_words);
                    r1 = split_ld(S, wb - 1, s_words);
                    r2 = split_ld(S, wb - 2, s_words);
                    r3 = split_ld(S, wb - 3, s_words);
                    r4 = split_ld(S, wb - 4, s_words);
                    if (mask != 1u) {
                        #pragma unroll
                        for (u32 i = 2 * A_W64; i > 0; --i) m[i] = split_orp(m[i], m[i - 1], mask);
                        m[0] = split_orp(m[0], r1, mask);
                    }
                }
                const u32 r = (u32)U & 31u;
                const u32 l = 31u - r;
                #pragma unroll
                for (u32 i = 0; i < 2 * A_W64; ++i)
                    acc[i] |= (m[i] >> r) | ((m[i + 1] << l) << 1);
#if SPLIT_EXIT
                u32 all = acc[0];
                #pragma unroll
                for (u32 i = 1; i < 2 * A_W64; ++i) all &= acc[i];
                if (all == ~0u) { done = true; break; }
#endif
                if (++j >= je) break;
                U = (vc - (int)cands[j]) / 6;
            }
        }
    }

    u32 cnt = 0;
    #pragma unroll
    for (u32 i = 0; i < A_W64; ++i) {
        const u32 kb = k0 + 64u * i;
        if (!live || kb >= ec) continue;
        const u64 a = (u64)acc[2 * i] | ((u64)acc[2 * i + 1] << 32);
        const u32 n = min(64u, ec - kb);
        const u64 valid = (n >= 64u) ? ~0UL : ((1UL << n) - 1UL);
        if (write_checked) checked[(u64)c * chk_stride + (kb >> 6)] = a & valid;
        cnt += (u32)popcount(~a & valid);
    }
    const u32 off = sub_group_scan_exclusive_add(cnt);
    const u32 tot = sub_group_reduce_add(cnt);
    u32 base = 0;
    if (get_sub_group_local_id() == 0 && tot) base = atomic_add(pending_count, tot);
    base = sub_group_broadcast(base, 0) + off;
    if (cnt) {
        #pragma unroll
        for (u32 i = 0; i < A_W64; ++i) {
            const u32 kb = k0 + 64u * i;
            if (kb >= ec) continue;
            const u64 a = (u64)acc[2 * i] | ((u64)acc[2 * i + 1] << 32);
            const u32 n = min(64u, ec - kb);
            u64 miss = ~a & ((n >= 64u) ? ~0UL : ((1UL << n) - 1UL));
            while (miss) {
                const u32 b = (u32)ctz(miss);
                miss &= miss - 1UL;
                if (base < pending_cap) pending[base] = e0 + 3u * (kb + b);
                ++base;
            }
        }
    }
}

// Phase B of the class-split checker: one lane per pending even, probing its
// class's rest list (rest + roff_c, rcnt_c entries) on the dense map; sets
// the even's bit in the class-layout checked[] or reports a leftover.
/// @post   each pending even gets its class-layout checked[] bit
///         (write_checked) or a leftover entry.
__kernel void check_split_b(
        __global const u64* restrict prime_map,
        __global u64* restrict checked,
        const u32 chk_stride,
        __global const u16* restrict rest,
        const u32 roff0, const u32 roff1, const u32 roff2,
        const u32 rcnt0, const u32 rcnt1, const u32 rcnt2,
        __global const u32* restrict pending,
        __global const u32* restrict pending_count,
        __global u32* restrict leftovers,
        __global u32* restrict leftover_count,
        const u64 q_first,
        const u64 q_odds,
        const u64 seg_start,
        const u32 pending_cap,
        const u32 e00, const u32 e01, const u32 e02,
        const u32 write_checked,
        const u32 cm, const u32 phi1, const u32 phi5, const u32 off5,   // class mode
        __global u64* restrict ridge,                 // Goldbach ridge appends
        const u32 ridge_thresh)                       // record p >= this; ~0u = off
{
    const __global u32* map32 = (const __global u32*)prime_map;
    const u32 n = min(*pending_count, pending_cap);
    const long Tbase = (long)seg_start - (long)q_first;
    const u32 sm3 = (u32)(seg_start % 3UL);
    for (u32 i = get_global_id(0); i < n; i += get_global_size(0)) {
        const u32 e = pending[i];                      // segment-local even index
        const u32 c = (sm3 + 2u * (e % 3u)) % 3u;      // n mod 3
        const u32 ro = (c == 0) ? roff0 : (c == 1) ? roff1 : roff2;
        const u32 rc = (c == 0) ? rcnt0 : (c == 1) ? rcnt1 : rcnt2;
        const u32 e0 = (c == 0) ? e00 : (c == 1) ? e01 : e02;
        bool ok = false;
        u32 rp = 0u;                                   // smallest witness p (ridge)
        for (u32 j = 0; j < rc && !ok; j += PROBE_BATCH) {
            u32 hit = 0u;
            #pragma unroll
            for (u32 u = 0; u < PROBE_BATCH; ++u) {
                if (j + u >= rc) break;
                const long p = rest[ro + j + u];
                const long qb = (long)e + ((Tbase - p) >> 1);
                if (qb < 0 || (u64)qb >= q_odds) continue;
                u32 ad = (u32)qb;
                if (cm == 2u) {                          // mod-30 class map
                    const cmap g = {cm, phi1, phi5, off5};
                    const u32 t = ad / 15u, k = c30_class(ad - 15u * t, g);
                    if (k == 15u) {                      // q divisible by 3 or 5: not stored
                        const u64 q = q_first + 2UL * (u64)qb;
                        hit |= ((q == 3UL || q == 5UL) ? 1u : 0u) << u;
                        continue;
                    }
                    ad = k * off5 + t;
                } else if (cm) {                         // class map (see map_clear)
                    const u32 t = ad / 3u, ph = ad - 3u * t;
                    if (ph == phi1) ad = t;
                    else if (ph == phi5) ad = off5 + t;
                    else { hit |= ((q_first + 2UL * (u64)qb == 3UL) ? 1u : 0u) << u; continue; }
                }
                hit |= ((map32[ad >> 5] >> (ad & 31u)) & 1u) << u;
            }
            ok = hit != 0u;
            if (ok) rp = (u32)rest[ro + j + (u32)ctz(hit)];
        }
        if (ok) {
            if (rp >= ridge_thresh) {                // useful residues first, so
                const u32 slot = atomic_add((volatile __global u32*)ridge, 1u);
                if (slot < RIDGE_CAP) {              // exact for n > MAX_GB_PRIME + 5
                    ridge[1 + 2u * slot] = seg_start + 2UL * (u64)e;
                    ridge[2 + 2u * slot] = rp;
                }
            }
            if (!write_checked) continue;
            const u32 k = (e - e0) / 3u;
            atomic_or((volatile __global u32*)(checked + (u64)c * chk_stride) + (k >> 5),
                      1u << (k & 31u));
        } else {
            const u32 slot = atom_inc(leftover_count);
            if (slot < MAX_LEFTOVERS) leftovers[slot] = e;
        }
    }
}

// ----------------------------------------------------------------------------
// Mod-30 checker (v1.13.0; host: include/check30.hpp). Evens grouped by
// n mod 30 (class r = 2 cr), read the mod-30 class map directly: candidate
// p = s (mod 30) tests q = n - p in class array k of residue r - s (W30_IDX),
// and the evens n0 + 30 i of class r are consecutive bits of that array, so a
// funnel is the class-split one on array k. Pattern bit d = candidate p + 30d.
// meta per class (stride 32): n_seg, (begin | end << 16, s | mask << 8) x n_seg.
// ----------------------------------------------------------------------------
#ifndef C30_MAXD
#define C30_MAXD 4               // largest pattern shift (host-checked)
#endif
/// @pre    bit 0 of mask set; shifts d <= C30_MAXD < 32.
/// @return bit j = OR over d in mask of bit j - d of (a:b).
inline u32 c30_orp(const u32 a, const u32 b, const u32 mask)
{
    u32 v = a;
    #pragma unroll
    for (u32 d = 1; d <= C30_MAXD; ++d)
        if ((mask >> d) & 1u) v |= (a << d) | (b >> (32u - d));
    return v;
}

// Window load of check30_a: words wb-4 .. wb+2*A_W64 (r4..r1, m[0..2*A_W64]).
// Vector loads (5 x vload8 from wb-4, one LSC message each) instead of 37
// scalar loads: every lane reads its own lines, so each scalar SIMD load
// touched 16 cache lines and window reloads were L1-throughput-bound.
#if A_W64 == 16
#define C30_WIN(S, wb, n) do { \
    if ((wb) >= 4 && (wb) + 36 <= (int)(n)) { \
        const __global u32* P_ = (S) + ((wb) - 4); \
        const uint8 v0_ = vload8(0, P_); \
        const uint8 v1_ = vload8(1, P_); \
        const uint8 v2_ = vload8(2, P_); \
        const uint8 v3_ = vload8(3, P_); \
        const uint8 v4_ = vload8(4, P_); \
        r4 = v0_.s0; r3 = v0_.s1; r2 = v0_.s2; r1 = v0_.s3; \
        m[0] = v0_.s4; \
        m[1] = v0_.s5; \
        m[2] = v0_.s6; \
        m[3] = v0_.s7; \
        m[4] = v1_.s0; \
        m[5] = v1_.s1; \
        m[6] = v1_.s2; \
        m[7] = v1_.s3; \
        m[8] = v1_.s4; \
        m[9] = v1_.s5; \
        m[10] = v1_.s6; \
        m[11] = v1_.s7; \
        m[12] = v2_.s0; \
        m[13] = v2_.s1; \
        m[14] = v2_.s2; \
        m[15] = v2_.s3; \
        m[16] = v2_.s4; \
        m[17] = v2_.s5; \
        m[18] = v2_.s6; \
        m[19] = v2_.s7; \
        m[20] = v3_.s0; \
        m[21] = v3_.s1; \
        m[22] = v3_.s2; \
        m[23] = v3_.s3; \
        m[24] = v3_.s4; \
        m[25] = v3_.s5; \
        m[26] = v3_.s6; \
        m[27] = v3_.s7; \
        m[28] = v4_.s0; \
        m[29] = v4_.s1; \
        m[30] = v4_.s2; \
        m[31] = v4_.s3; \
        m[32] = v4_.s4; \
    } else { \
        _Pragma("unroll") for (u32 i_ = 0; i_ <= 2 * A_W64; ++i_) m[i_] = split_ld((S), (wb) + i_, (n)); \
        r1 = split_ld((S), (wb) - 1, (n)); r2 = split_ld((S), (wb) - 2, (n)); \
        r3 = split_ld((S), (wb) - 3, (n)); r4 = split_ld((S), (wb) - 4, (n)); \
    } } while (0)
#elif A_W64 == 14
#define C30_WIN(S, wb, n) do { \
    if ((wb) >= 4 && (wb) + 36 <= (int)(n)) { \
        const __global u32* P_ = (S) + ((wb) - 4); \
        const uint8 v0_ = vload8(0, P_); \
        const uint8 v1_ = vload8(1, P_); \
        const uint8 v2_ = vload8(2, P_); \
        const uint8 v3_ = vload8(3, P_); \
        const uint8 v4_ = vload8(4, P_); \
        r4 = v0_.s0; r3 = v0_.s1; r2 = v0_.s2; r1 = v0_.s3; \
        m[0] = v0_.s4; \
        m[1] = v0_.s5; \
        m[2] = v0_.s6; \
        m[3] = v0_.s7; \
        m[4] = v1_.s0; \
        m[5] = v1_.s1; \
        m[6] = v1_.s2; \
        m[7] = v1_.s3; \
        m[8] = v1_.s4; \
        m[9] = v1_.s5; \
        m[10] = v1_.s6; \
        m[11] = v1_.s7; \
        m[12] = v2_.s0; \
        m[13] = v2_.s1; \
        m[14] = v2_.s2; \
        m[15] = v2_.s3; \
        m[16] = v2_.s4; \
        m[17] = v2_.s5; \
        m[18] = v2_.s6; \
        m[19] = v2_.s7; \
        m[20] = v3_.s0; \
        m[21] = v3_.s1; \
        m[22] = v3_.s2; \
        m[23] = v3_.s3; \
        m[24] = v3_.s4; \
        m[25] = v3_.s5; \
        m[26] = v3_.s6; \
        m[27] = v3_.s7; \
        m[28] = v4_.s0; \
    } else { \
        _Pragma("unroll") for (u32 i_ = 0; i_ <= 2 * A_W64; ++i_) m[i_] = split_ld((S), (wb) + i_, (n)); \
        r1 = split_ld((S), (wb) - 1, (n)); r2 = split_ld((S), (wb) - 2, (n)); \
        r3 = split_ld((S), (wb) - 3, (n)); r4 = split_ld((S), (wb) - 4, (n)); \
    } } while (0)
#elif A_W64 == 12
#define C30_WIN(S, wb, n) do { \
    if ((wb) >= 4 && (wb) + 28 <= (int)(n)) { \
        const __global u32* P_ = (S) + ((wb) - 4); \
        const uint8 v0_ = vload8(0, P_); \
        const uint8 v1_ = vload8(1, P_); \
        const uint8 v2_ = vload8(2, P_); \
        const uint8 v3_ = vload8(3, P_); \
        r4 = v0_.s0; r3 = v0_.s1; r2 = v0_.s2; r1 = v0_.s3; \
        m[0] = v0_.s4; \
        m[1] = v0_.s5; \
        m[2] = v0_.s6; \
        m[3] = v0_.s7; \
        m[4] = v1_.s0; \
        m[5] = v1_.s1; \
        m[6] = v1_.s2; \
        m[7] = v1_.s3; \
        m[8] = v1_.s4; \
        m[9] = v1_.s5; \
        m[10] = v1_.s6; \
        m[11] = v1_.s7; \
        m[12] = v2_.s0; \
        m[13] = v2_.s1; \
        m[14] = v2_.s2; \
        m[15] = v2_.s3; \
        m[16] = v2_.s4; \
        m[17] = v2_.s5; \
        m[18] = v2_.s6; \
        m[19] = v2_.s7; \
        m[20] = v3_.s0; \
        m[21] = v3_.s1; \
        m[22] = v3_.s2; \
        m[23] = v3_.s3; \
        m[24] = v3_.s4; \
    } else { \
        _Pragma("unroll") for (u32 i_ = 0; i_ <= 2 * A_W64; ++i_) m[i_] = split_ld((S), (wb) + i_, (n)); \
        r1 = split_ld((S), (wb) - 1, (n)); r2 = split_ld((S), (wb) - 2, (n)); \
        r3 = split_ld((S), (wb) - 3, (n)); r4 = split_ld((S), (wb) - 4, (n)); \
    } } while (0)
#elif A_W64 == 10
#define C30_WIN(S, wb, n) do { \
    if ((wb) >= 4 && (wb) + 28 <= (int)(n)) { \
        const __global u32* P_ = (S) + ((wb) - 4); \
        const uint8 v0_ = vload8(0, P_); \
        const uint8 v1_ = vload8(1, P_); \
        const uint8 v2_ = vload8(2, P_); \
        const uint8 v3_ = vload8(3, P_); \
        r4 = v0_.s0; r3 = v0_.s1; r2 = v0_.s2; r1 = v0_.s3; \
        m[0] = v0_.s4; \
        m[1] = v0_.s5; \
        m[2] = v0_.s6; \
        m[3] = v0_.s7; \
        m[4] = v1_.s0; \
        m[5] = v1_.s1; \
        m[6] = v1_.s2; \
        m[7] = v1_.s3; \
        m[8] = v1_.s4; \
        m[9] = v1_.s5; \
        m[10] = v1_.s6; \
        m[11] = v1_.s7; \
        m[12] = v2_.s0; \
        m[13] = v2_.s1; \
        m[14] = v2_.s2; \
        m[15] = v2_.s3; \
        m[16] = v2_.s4; \
        m[17] = v2_.s5; \
        m[18] = v2_.s6; \
        m[19] = v2_.s7; \
        m[20] = v3_.s0; \
    } else { \
        _Pragma("unroll") for (u32 i_ = 0; i_ <= 2 * A_W64; ++i_) m[i_] = split_ld((S), (wb) + i_, (n)); \
        r1 = split_ld((S), (wb) - 1, (n)); r2 = split_ld((S), (wb) - 2, (n)); \
        r3 = split_ld((S), (wb) - 3, (n)); r4 = split_ld((S), (wb) - 4, (n)); \
    } } while (0)
#elif A_W64 == 8
#define C30_WIN(S, wb, n) do { \
    if ((wb) >= 4 && (wb) + 20 <= (int)(n)) { \
        const __global u32* P_ = (S) + ((wb) - 4); \
        const uint8 v0_ = vload8(0, P_); \
        const uint8 v1_ = vload8(1, P_); \
        const uint8 v2_ = vload8(2, P_); \
        r4 = v0_.s0; r3 = v0_.s1; r2 = v0_.s2; r1 = v0_.s3; \
        m[0] = v0_.s4; \
        m[1] = v0_.s5; \
        m[2] = v0_.s6; \
        m[3] = v0_.s7; \
        m[4] = v1_.s0; \
        m[5] = v1_.s1; \
        m[6] = v1_.s2; \
        m[7] = v1_.s3; \
        m[8] = v1_.s4; \
        m[9] = v1_.s5; \
        m[10] = v1_.s6; \
        m[11] = v1_.s7; \
        m[12] = v2_.s0; \
        m[13] = v2_.s1; \
        m[14] = v2_.s2; \
        m[15] = v2_.s3; \
        m[16] = v2_.s4; \
    } else { \
        _Pragma("unroll") for (u32 i_ = 0; i_ <= 2 * A_W64; ++i_) m[i_] = split_ld((S), (wb) + i_, (n)); \
        r1 = split_ld((S), (wb) - 1, (n)); r2 = split_ld((S), (wb) - 2, (n)); \
        r3 = split_ld((S), (wb) - 3, (n)); r4 = split_ld((S), (wb) - 4, (n)); \
    } } while (0)
#else
#error "C30_WIN: A_W64 must be 8, 10, 12, 14 or 16"
#endif

// Pattern OR of the whole window, in place top down; specialized for the
// contiguous masks the plans use (a runtime-mask loop costs every shift).
#define C30_ORW1(MSK) do { \
    _Pragma("unroll") for (u32 i_ = 2 * A_W64; i_ > 0; --i_) m[i_] = c30_orp(m[i_], m[i_ - 1], (MSK)); \
    m[0] = c30_orp(m[0], r1, (MSK)); } while (0)
#define C30_ORWIN(mask) do { \
    if ((mask) == 1u) {} \
    else if ((mask) == 15u) C30_ORW1(15u); \
    else if ((mask) == 31u) C30_ORW1(31u); \
    else if ((mask) == 7u) C30_ORW1(7u); \
    else if ((mask) == 3u) C30_ORW1(3u); \
    else C30_ORW1(mask); } while (0)

// Phase A. One work-item = 1024 evens of one class; items_pad items per
// class (a multiple of the work-group size: one class per work-group).
/// @pre    plan from make_plan30 (candidates prime, < MAX_GB_PRIME); the
///         segment's mod-30 class map complete (every set bit is a prime q,
///         barrier after the sieve kernels); tb = seg_start - q_first and the
///         per-class offsets fit int (host: seg30 ok); grid = 15 * items_pad
///         lanes, items_pad a multiple of the work-group size (one class per
///         work-group); pending_count zeroed for the segment.
/// @post   for every even n = seg_start + 2e (e < seg_evens), in its class
///         cr = (n mod 30) / 2: n is RESOLVED iff some phase-A funnel member
///         p of class cr has the map bit of q = n - p set (q is then prime:
///         a witness pair) — its checked[] bit is set when write_checked —
///         and otherwise e is appended to pending (atomic counter; entries
///         beyond pending_cap are counted but not stored: the host fails
///         closed). The map is unchanged.
__kernel void check30_a(
        __global const u32* restrict cmap32,     // 8 class arrays of cw32 u32 words
        const u32 cw32,
        __global u64* restrict checked,         // class cr at cr * chk_stride words
        const u32 chk_stride,
        __global const u16* restrict cands,
        __global const u32* restrict meta,
        __global u32* restrict pending,
        __global u32* restrict pending_count,
        const u32 pending_cap,
        const u32 items_pad,
        const u32 seg_evens,
        const u32 sm30,                          // seg_start mod 30
        const int tb,                            // seg_start - q_first
        const u32 phik,                          // dense phase of class k, nibble k
        const u32 write_checked)
{
    const u32 gid = get_global_id(0);
    const u32 cr = gid / items_pad;                    // uniform per work-group
    const u32 blk = gid - cr * items_pad;
    const u32 r = 2u * cr;
    const u32 e0 = ((r + 30u - sm30) % 30u) >> 1;      // first even index of the class
    const u32 ec = (cr < 15u && seg_evens > e0) ? (seg_evens - e0 + 14u) / 15u : 0u;
    const u32 mo = cr * 32u;
    const u32 k0 = blk * (A_W64 * 64u);
    const bool live = cr < 15u && k0 < ec;             // no early return: subgroup ops below

    u32 acc[2 * A_W64];
    #pragma unroll
    for (u32 i = 0; i < 2 * A_W64; ++i) {
        const u32 kb = k0 + 32u * i;
        const u32 n = (live && kb < ec) ? min(32u, ec - kb) : 0u;
        acc[i] = (n >= 32u) ? 0u : ~((1u << n) - 1u);  // invalid bits count as resolved
    }

    if (live) {
        const u32 ns = meta[mo];
        for (u32 st = 0; st < ns; ++st) {
            const u32 be = meta[mo + 1u + 2u * st];
            const u32 sm = meta[mo + 2u + 2u * st];
            u32 j = be & 0xFFFFu;
            const u32 je = be >> 16;
            const u32 s = sm & 0xFFu, mask = sm >> 8;
            const u32 k = W30_IDX[(r + 30u - s) % 30u];  // q class (host: a unit)
            const int vk = tb + 2 * (int)e0 - 2 * (int)((phik >> (4u * k)) & 15u);
            const __global u32* S = cmap32 + (u64)k * cw32;
#if C30_UDIV
            // every candidate of this plan segment is p = 30 a + s, so
            // U = U0 - floor(p / 30); for p < 2^16, floor(p / 30) =
            // (p * 34953) >> 20 exactly (error < 0.03 < 1/30)
            const int U0 = (vk - (int)s) / 30;         // exact: vk = s (mod 30)
            int U = U0 - (int)(((u32)cands[j] * 34953u) >> 20);
#else
            int U = (vk - (int)cands[j]) / 30;         // exact: may be < 0 at the range bottom
#endif
            int tw = (U >= 0) ? (U >> 5) : -((31 - U) >> 5);   // floor(U / 32)
            int wb = (int)(k0 >> 5) + tw;
            u32 m[2 * A_W64 + 1];
            u32 r1, r2, r3, r4;
            C30_WIN(S, wb, cw32);
            C30_ORWIN(mask);
            for (;;) {
                const int twj = (U >= 0) ? (U >> 5) : -((31 - U) >> 5);
                if (tw - twj == 1) {                   // uniform
                    tw = twj; --wb;
                    #pragma unroll
                    for (u32 i = 2 * A_W64; i > 0; --i) m[i] = m[i - 1];
                    m[0] = c30_orp(r1, r2, mask);
                    r1 = r2; r2 = r3; r3 = r4;
                    r4 = split_ld(S, wb - 4, cw32);
                } else if (tw != twj) {
                    wb -= tw - twj; tw = twj;
                    C30_WIN(S, wb, cw32);
                    C30_ORWIN(mask);
                }
                const u32 rr = (u32)U & 31u;
#if C30_ROT
                // funnel by rotates: R = m rotated right by rr; the low
                // 32 - rr bits of R[i] are m[i] >> rr, the high rr bits of
                // R[i + 1] are m[i + 1] << (32 - rr) — one select per word,
                // exact for rr = 0 too (hm = ~0) without a shift by 32.
                const u32 rl = 32u - rr;                // rotate(x, 32) == x
                const u32 hm = 0xFFFFFFFFu >> rr;
                u32 rn = rotate(m[0], rl);
                #pragma unroll
                for (u32 i = 0; i < 2 * A_W64; ++i) {
                    const u32 rc = rn;
                    rn = rotate(m[i + 1], rl);
                    acc[i] |= bitselect(rn, rc, hm);
                }
#else
                const u32 l = 31u - rr;
                #pragma unroll
                for (u32 i = 0; i < 2 * A_W64; ++i)
                    acc[i] |= (m[i] >> rr) | ((m[i + 1] << l) << 1);
#endif
                if (++j >= je) break;
#if C30_UDIV
                U = U0 - (int)(((u32)cands[j] * 34953u) >> 20);
#else
                U = (vk - (int)cands[j]) / 30;
#endif
            }
        }
    }

    u32 cnt = 0;
    #pragma unroll
    for (u32 i = 0; i < A_W64; ++i) {
        const u32 kb = k0 + 64u * i;
        if (!live || kb >= ec) continue;
        const u64 a = (u64)acc[2 * i] | ((u64)acc[2 * i + 1] << 32);
        const u32 n = min(64u, ec - kb);
        const u64 valid = (n >= 64u) ? ~0UL : ((1UL << n) - 1UL);
        if (write_checked) checked[(u64)cr * chk_stride + (kb >> 6)] = a & valid;
        cnt += (u32)popcount(~a & valid);
    }
    const u32 off = sub_group_scan_exclusive_add(cnt);
    const u32 tot = sub_group_reduce_add(cnt);
    u32 base = 0;
    if (get_sub_group_local_id() == 0 && tot) base = atomic_add(pending_count, tot);
    base = sub_group_broadcast(base, 0) + off;
    if (cnt) {
        #pragma unroll
        for (u32 i = 0; i < A_W64; ++i) {
            const u32 kb = k0 + 64u * i;
            if (kb >= ec) continue;
            const u64 a = (u64)acc[2 * i] | ((u64)acc[2 * i + 1] << 32);
            const u32 n = min(64u, ec - kb);
            u64 miss = ~a & ((n >= 64u) ? ~0UL : ((1UL << n) - 1UL));
            while (miss) {
                const u32 b = (u32)ctz(miss);
                miss &= miss - 1UL;
                if (base < pending_cap) pending[base] = e0 + 15u * (kb + b);
                ++base;
            }
        }
    }
}

// Phase B of the mod-30 checker: one lane per pending even, probing its
// class's rest list (rtab: offsets [0, 15), counts [15, 30)) on the class map.
/// @pre    check30_a of the same segment finished (barrier); pending /
///         pending_count from it (the host checks count <= pending_cap before
///         trusting the result); rest / rtab from make_plan30: per class the
///         candidates phase A did not test, ascending (useful classes first,
///         futile last); leftover_count zeroed for the segment; the map as
///         in check30_a.
/// @post   each pending e < min(count, pending_cap) is RESOLVED iff some rest
///         candidate p has q = n - p prime per the map (q = 3, 5 by value:
///         not stored); then checked[] (write_checked), and when p >= the
///         thresholds a ridge append / hist_p / first_p entry with the first
///         (smallest) such p. Otherwise e is appended to leftovers (counted
///         past MAX_LEFTOVERS: the host fails closed) for the exact CPU
///         fallback. The map and pending are unchanged.
__kernel void check30_b(
        __global const u64* restrict prime_map,
        __global u64* restrict checked,
        const u32 chk_stride,
        __global const u16* restrict rest,
        __global const u32* restrict rtab,
        __global const u32* restrict pending,
        __global const u32* restrict pending_count,
        __global u32* restrict leftovers,
        __global u32* restrict leftover_count,
        const u64 q_first,
        const u64 q_odds,
        const u64 seg_start,
        const u32 pending_cap,
        const u32 write_checked,
        const u32 phi1, const u32 phi5, const u32 off5,  // mod-30 class map (cm = 2)
        __global u64* restrict ridge,                 // Goldbach ridge appends
        const u32 ridge_thresh,                       // record p >= this; ~0u = off
        __global u32* restrict hist_p,                // tail p(n) counts by p >> 1
        __global u32* restrict first_p,               // tail first even index by p >> 1
        __global const u32* restrict want_p,          // first_p bins still wanted
        const u32 tail_t)                             // record p(n) >= this; ~0u = off
{
    const __global u32* map32 = (const __global u32*)prime_map;
    const cmap g = {2u, phi1, phi5, off5};
    const u32 n = min(*pending_count, pending_cap);
    const long Tbase = (long)seg_start - (long)q_first;
    const u32 sm30 = (u32)(seg_start % 30UL);
    for (u32 i = get_global_id(0); i < n; i += get_global_size(0)) {
        const u32 e = pending[i];                      // segment-local even index
        const u32 cr = ((sm30 + 2u * (e % 15u)) % 30u) >> 1;
        const u32 ro = rtab[cr], rc = rtab[15u + cr];
        bool ok = false;
        u32 rp = 0u;                                   // smallest witness p (ridge)
        for (u32 j = 0; j < rc && !ok; j += PROBE_BATCH) {
            u32 hit = 0u;
            #pragma unroll
            for (u32 u = 0; u < PROBE_BATCH; ++u) {
                if (j + u >= rc) break;
                const long p = rest[ro + j + u];
                const long qb = (long)e + ((Tbase - p) >> 1);
                if (qb < 0 || (u64)qb >= q_odds) continue;
                const u32 ad0 = (u32)qb;
                const u32 t = ad0 / 15u, k = c30_class(ad0 - 15u * t, g);
                if (k == 15u) {                          // q divisible by 3 or 5: not stored
                    const u64 q = q_first + 2UL * (u64)qb;
                    hit |= ((q == 3UL || q == 5UL) ? 1u : 0u) << u;
                    continue;
                }
                const u32 ad = k * off5 + t;
                hit |= ((map32[ad >> 5] >> (ad & 31u)) & 1u) << u;
            }
            ok = hit != 0u;
            if (ok) rp = (u32)rest[ro + j + (u32)ctz(hit)];
        }
        if (ok) {
            if (rp >= ridge_thresh) {                // useful residues first, so
                const u32 slot = atomic_add((volatile __global u32*)ridge, 1u);
                if (slot < RIDGE_CAP) {              // exact for n > MAX_GB_PRIME + 5
                    ridge[1 + 2u * slot] = seg_start + 2UL * (u64)e;
                    ridge[2 + 2u * slot] = rp;
                }
            }
            if (rp >= tail_t) {                      // every p(n) >= tail_t lands here
                atomic_add(hist_p + (rp >> 1), 1u);
                if ((want_p[rp >> 6] >> ((rp >> 1) & 31u)) & 1u) atomic_min(first_p + (rp >> 1), e);
            }
            if (!write_checked) continue;
            const u32 e0 = ((2u * cr + 30u - sm30) % 30u) >> 1;
            const u32 k = (e - e0) / 15u;
            atomic_or((volatile __global u32*)(checked + (u64)cr * chk_stride) + (k >> 5),
                      1u << (k & 31u));
        } else {
            const u32 slot = atom_inc(leftover_count);
            if (slot < MAX_LEFTOVERS) leftovers[slot] = e;
        }
    }
}

// ----------------------------------------------------------------------------
// Exact Goldbach statistics (v2.2; host: include/gstats.hpp).
//
// p(n) = the smallest prime p with n - p prime. Two mechanisms:
//   * TAIL, every segment: the production plan keeps every phase-A funnel
//     member below CHECK30_TAIL (3000), so every even with p(n) >= that
//     bound fails phase A, and check30_b's ascending probe finds its exact
//     p(n): hist_p / first_p record it (p >= tail_t only; the evens check30_b
//     resolves below the bound are a biased subset and are not counted).
//   * FULL, every GB_STATS_SAMPLE-th segment: check30_xs8/6/4/3 test, per
//     class r, the first K_r useful candidates (q = n - p coprime to 30) in
//     ASCENDING order, one candidate at a time (a funnel cannot tell its
//     members apart): the bits a candidate sets first are exactly the evens
//     with p(n) = p, counted into hist[]. The rest go to check30_xb, which
//     probes the remaining class list ascending, so its first hit is p(n).
// xall: per class r, all candidates ascending (useful first, futile last);
// xtab: [0, 15) class offset into xall, [15, 30) K_r, [30, 45) list length.
// hist is indexed by the xall position, hist_p / first_p / want_p by p >> 1.
// Exact for every even n > MAX_GB_PRIME + 5 (futile-last ordering; the host
// only enables statistics far above that).
// ----------------------------------------------------------------------------
#ifndef X_KMAX
#define X_KMAX 128               // max K_r (SLM histogram span; host-checked)
#endif
#define XB_SPAN 32               // phase-B SLM bins per class (beyond: global)

/// Exact phase B: one lane per pending even, probing the rest of its class
/// list ascending from K_r; the first hit is p(n).
/// @pre    pending from check30_x of the same segment and plan.
/// @post   hist / first_p (want_p bins) / ridge appends for every resolved even;
///         unresolved evens go to leftovers (CPU fallback, exact there too).
__kernel void check30_xb(
        __global const u64* restrict prime_map,
        __global const u16* restrict xall,
        __global const u32* restrict xtab,
        __global const u32* restrict pending,
        __global const u32* restrict pending_count,
        __global u32* restrict leftovers,
        __global u32* restrict leftover_count,
        const u64 q_first,
        const u64 q_odds,
        const u64 seg_start,
        const u32 pending_cap,
        const u32 phi1, const u32 phi5, const u32 off5,  // mod-30 class map (cm = 2)
        __global u64* restrict ridge,                 // Goldbach ridge appends
        const u32 ridge_thresh,                       // record p >= this; ~0u = off
        __global u32* restrict hist,
        __global u32* restrict first_p,               // by p >> 1 (want_p bins only)
        __global const u32* restrict want_p)
{
    __local u32 sh[15 * XB_SPAN];
    const u32 lid = get_local_id(0);
    for (u32 j = lid; j < 15u * XB_SPAN; j += WG_SIZE) sh[j] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);
    const __global u32* map32 = (const __global u32*)prime_map;
    const cmap g = {2u, phi1, phi5, off5};
    const u32 n = min(*pending_count, pending_cap);
    const long Tbase = (long)seg_start - (long)q_first;
    const u32 sm30 = (u32)(seg_start % 30UL);
    for (u32 i = get_global_id(0); i < n; i += get_global_size(0)) {
        const u32 e = pending[i];                      // segment-local even index
        const u32 cr = ((sm30 + 2u * (e % 15u)) % 30u) >> 1;
        const u32 xo = xtab[cr], kx = xtab[15u + cr], len = xtab[30u + cr];
        bool ok = false;
        u32 jh = 0u;                                   // list index of p(n)
        for (u32 j = kx; j < len && !ok; j += PROBE_BATCH) {
            u32 hit = 0u;
            #pragma unroll
            for (u32 u = 0; u < PROBE_BATCH; ++u) {
                if (j + u >= len) break;
                const long p = xall[xo + j + u];
                const long qb = (long)e + ((Tbase - p) >> 1);
                if (qb < 0 || (u64)qb >= q_odds) continue;
                const u32 ad0 = (u32)qb;
                const u32 t = ad0 / 15u, k = c30_class(ad0 - 15u * t, g);
                if (k == 15u) {                          // q divisible by 3 or 5: not stored
                    const u64 q = q_first + 2UL * (u64)qb;
                    hit |= ((q == 3UL || q == 5UL) ? 1u : 0u) << u;
                    continue;
                }
                const u32 ad = k * off5 + t;
                hit |= ((map32[ad >> 5] >> (ad & 31u)) & 1u) << u;
            }
            ok = hit != 0u;
            if (ok) jh = j + (u32)ctz(hit);
        }
        if (ok) {
            const u32 x = xo + jh;
            const u32 rp = xall[x];
            if (jh - kx < XB_SPAN) atomic_add(&sh[cr * XB_SPAN + jh - kx], 1u);
            else atomic_add(hist + x, 1u);
            if ((want_p[rp >> 6] >> ((rp >> 1) & 31u)) & 1u) atomic_min(first_p + (rp >> 1), e);
            if (rp >= ridge_thresh) {
                const u32 slot = atomic_add((volatile __global u32*)ridge, 1u);
                if (slot < RIDGE_CAP) {
                    ridge[1 + 2u * slot] = seg_start + 2UL * (u64)e;
                    ridge[2 + 2u * slot] = rp;
                }
            }
        } else {
            const u32 slot = atom_inc(leftover_count);
            if (slot < MAX_LEFTOVERS) leftovers[slot] = e;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 j = lid; j < 15u * XB_SPAN; j += WG_SIZE) {
        const u32 c = j / XB_SPAN, d = j - c * XB_SPAN;
        const u32 x = xtab[c] + xtab[15u + c] + d;
        if (sh[j] && xtab[15u + c] + d < xtab[30u + c]) atomic_add(hist + x, sh[j]);
    }
}


/// @pre    lo <= hi; 32 w + 32 < 2^32.
/// @return the bits of 32-bit word w (bits [32 w, 32 w + 32)) that lie in
///         [lo, hi), as a mask (bit i <-> bit 32 w + i).
inline u32 range_mask32(const u32 w, const u32 lo, const u32 hi)
{
    const u32 b = 32u * w;
    if (hi <= b || lo >= b + 32u) return 0u;
    const u32 a = (lo > b) ? lo - b : 0u;
    const u32 e = (hi < b + 32u) ? hi - b : 32u;
    const u32 top = (e >= 32u) ? ~0u : ((1u << e) - 1u);
    return top & ~((1u << a) - 1u);
}

/// Prime and twin-prime counts of one segment on the mod-30 class map, its
/// gap tail and, on walked segments, its prime-gap histogram (the live
/// paper's section 2.2; v2.5 fuses the former gaps30 walk in: one read of
/// the map, the range masks and the look-back shared). A grid of work-groups
/// strides over lane chunks of C30_CW u32 words of every class array.
/// Positions: class c bit t is q = q_first + pos, pos = 30 t + 2 phi[c]; the
/// walk works in half positions h = pos / 2, so h - h_prev is the bin g / 2.
/// @pre    the segment's class map complete (tile + large-prime kernels
///         finished, barrier); cw32 % C30_CW == 0 (host-checked); local size
///         >= 13, at most G30_MAX_SG sub-groups; lim from count30_limits for
///         this segment (class bit ranges [lo, hi) and twin class pairs
///         (2,3), (4,5), (7,0) with offset d in {0, 1}); ord / phis: the 8
///         classes in ascending phi and their phi, phic: phi by class
///         (nibbles, low first); pos_lo: the smallest counted position; out,
///         gt[0..1] and (walk) hist zeroed, out[11] = ~0 by the host before
///         the segment; walk requires tcap.
/// @post   out[c] += set bits of class array c in [lim[2c], lim[2c+1])
///         (= primes of residue class c in the segment's counting range);
///         out[8 + j] += t in [lim[19+5j], lim[20+5j]) with bit t of class
///         ka and bit t + d of class kb both set (twin pairs q, q + 2 of
///         type j); tcap: out[11] / out[12] = min / max counted position,
///         gt = every gap >= G30T_TAIL but the one into the segment's first
///         prime (count, flags, (upper pos, g) pairs; bit-parallel test);
///         walk: hist[i] += gaps g = 2 i between consecutive counted primes
///         (bin G30_HBINS - 1: g >= 2 G30_HBINS - 2), hist[0] += the counted
///         primes without a counted predecessor (the segment's first only),
///         part[group] = max(g << 32 | ~(upper pos)) of the group's gaps (0:
///         none). The map is unchanged. Exactness vs stats_ref: §9.2l.
#ifndef C30_CW
#define C30_CW 4                 // words per lane (host: cw32 % C30_CW == 0; 8: slower)
#endif
// Exact gap of a gap-tail candidate (count30): the prime that opens block tb
// (lowest phi class with a bit) and the last counted prime before it (look-back
// up to 2 * 32 blocks: every gap below 2^64 spans fewer). Appends (upper pos,
// g) to gt when g >= G30T_TAIL; skips the segment's first prime (no counted
// predecessor: the host stitches it); flags a predecessor beyond the look-back.
#define G30T_TAIL 700u
#define G30T_ERR_LOOK 2u
#define G30T_ERR_FULL 4u
#define G30_HBINS 1024u
#define G30_MAX_SG 32u           // sub-groups per work-group (256 / 8)
inline void gap_tail_exact(__global const u32* restrict cmap32, const u32 cw32, __global const u32* restrict lim,
                           const u32 ord, const u32 phis, const u32 pos_lo, const u32 tb,
                           __global u32* restrict gt, const u32 tcap)
{
    const u32 wi = tb >> 5, j = tb & 31u;
    u32 ob = 0u;
    for (u32 k = 0; k < 8u; ++k) {
        const u32 c = (ord >> (4u * k)) & 15u;
        if ((cmap32[(u64)c * cw32 + wi] & range_mask32(wi, lim[2 * c], lim[2 * c + 1])) >> j & 1u) {
            ob = 2u * ((phis >> (4u * k)) & 15u);
            break;
        }
    }
    // look back word by word for the last counted prime below block tb
    for (u32 d = 0; d <= 2u && d <= wi; ++d) {
        const u32 wb = wi - d;
        u32 x[8], ax = 0u;
        for (u32 k = 0; k < 8u; ++k) {
            const u32 c = (ord >> (4u * k)) & 15u;
            x[k] = cmap32[(u64)c * cw32 + wb] & range_mask32(wb, lim[2 * c], lim[2 * c + 1]);
            if (d == 0u) x[k] &= j ? (1u << j) - 1u : 0u;
            ax |= x[k];
        }
        if (ax) {
            const u32 ja = 31u - clz(ax);
            u32 oa = 0u;
            for (u32 k = 0; k < 8u; ++k)
                if ((x[k] >> ja) & 1u) oa = 2u * ((phis >> (4u * k)) & 15u);
            const u32 ta = 32u * wb + ja, q = 30u * tb + ob, g = q - (30u * ta + oa);
            if (g >= G30T_TAIL) {
                const u32 slot = atomic_inc(gt);
                if (slot < tcap) { gt[2 + 2 * slot] = q; gt[3 + 2 * slot] = g; }
                else atomic_or(gt + 1, G30T_ERR_FULL);
            }
            return;
        }
    }
    // no counted predecessor within the look-back: fine only below the range
    if (30u * 32u * (wi >= 2u ? wi - 2u : 0u) >= pos_lo) atomic_or(gt + 1, G30T_ERR_LOOK);
}

__kernel void count30(
        __global const u32* restrict cmap32,
        const u32 cw32,
        __global const u32* restrict lim,
        __global u32* restrict out,
        const u32 ord,
        const u32 phis,
        const u32 phic,
        const u32 pos_lo,
        __global u32* restrict gt,
        const u32 tcap,
        const u32 walk,
        __global u64* restrict part,
        __global u32* restrict hist)
{
    __local u32 sh[13];
    __local u32 lh[G30_HBINS];
    __local u64 red[G30_MAX_SG];
    const u32 lid = get_local_id(0);
    if (lid < 13u) sh[lid] = lid == 11u ? ~0u : 0u;
    if (walk)
        for (u32 i = lid; i < G30_HBINS; i += get_local_size(0)) lh[i] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);
    u32 ph[8];                                // phi of class c: its half offset in a block of 30
    #pragma unroll
    for (u32 c = 0; c < 8; ++c) ph[c] = (phic >> (4u * c)) & 15u;
    u32 ncl[8], ntw[3];
    #pragma unroll
    for (u32 c = 0; c < 8; ++c) ncl[c] = 0u;
    ntw[0] = ntw[1] = ntw[2] = 0u;
    u32 pfirst = ~0u, plast = 0u;             // the lane's first / last counted prime position
    u32 err = 0u, dmax = 0u, dpos = 0u;       // walk: flags, largest gap / 2 and its upper h
    const u32 nlanes = cw32 / C30_CW;
    // grid-stride: walked segments use a fixed grid (one histogram flush per
    // work-group), the others one chunk per lane
    for (u32 lane = get_global_id(0); lane < nlanes; lane += get_global_size(0)) {
        const u32 w = C30_CW * lane;
        u32 v[8][C30_CW];
        #pragma unroll
        for (u32 c = 0; c < 8; ++c) {
            #pragma unroll
            for (u32 i = 0; i < C30_CW; i += 4) {
                const uint4 x = vload4(0, cmap32 + (u64)c * cw32 + w + i);
                v[c][i] = x.s0; v[c][i + 1] = x.s1; v[c][i + 2] = x.s2; v[c][i + 3] = x.s3;
            }
        }
        // twin pairs 11/13 (classes 2, 3), 17/19 (4, 5), 29/31 (7, 0), on the
        // raw words (their t ranges are not the classes' prime ranges)
        #pragma unroll
        for (u32 j = 0; j < 3; ++j) {
            const u32 ka = (j == 0) ? 2u : (j == 1) ? 4u : 7u, kb = (j == 0) ? 3u : (j == 1) ? 5u : 0u;
            const u32 d = lim[18 + 5 * j], lo = lim[19 + 5 * j], hi = lim[20 + 5 * j];
            if (lim[16 + 5 * j] == ka && lim[17 + 5 * j] == kb) {
                const u32 nx = (d && w + C30_CW < cw32) ? cmap32[(u64)kb * cw32 + w + C30_CW] : 0u;
                #pragma unroll
                for (u32 i = 0; i < C30_CW; ++i) {
                    const u32 bn = (i + 1 < C30_CW) ? v[kb][i + 1] : nx;
                    const u32 b = d ? ((v[kb][i] >> 1) | (bn << 31)) : v[kb][i];
                    ntw[j] += popcount(v[ka][i] & b & range_mask32(w + i, lo, hi));
                }
            }
        }
        // primes: the words masked to the counting range from here on
        u32 any[C30_CW], aprev = 0u;          // counted primes per bit position (gap tail, walk)
        u32 cf = ~0u;                         // the chunk's first counted prime position
        u32 hp = 0u;                          // walk: half position of the last counted prime before the chunk
        #pragma unroll
        for (u32 i = 0; i < C30_CW; ++i) any[i] = 0u;
        #pragma unroll
        for (u32 c = 0; c < 8; ++c) {
            const u32 lo = lim[2 * c], hi = lim[2 * c + 1];
            #pragma unroll
            for (u32 i = 0; i < C30_CW; ++i) {
                const u32 m = v[c][i] & range_mask32(w + i, lo, hi);
                v[c][i] = m;
                ncl[c] += popcount(m);
                any[i] |= m;
                if (tcap && m) {
                    const u32 b0 = 30u * 32u * (w + i) + 2u * ph[c];
                    cf = min(cf, b0 + 30u * ctz(m));
                    plast = max(plast, b0 + 30u * (31u - clz(m)));
                }
            }
            if (tcap && w) {
                const u32 pw = cmap32[(u64)c * cw32 + w - 1u] & range_mask32(w - 1u, lo, hi);
                aprev |= pw;
                if (pw) hp = max(hp, 15u * (32u * (w - 1u) + 31u - clz(pw)) + ph[c]);
            }
        }
        pfirst = min(pfirst, cf);
        // Gap tail (prime gaps >= G30T_TAIL, every segment, exact): such a gap
        // leaves >= 22 empty blocks of 30 before its upper prime. Bit-parallel
        // run test on a 64-bit window (previous word : word): block b is a
        // candidate when b holds a prime and b - 1 .. b - 22 hold none; the rare
        // candidates get their exact gap from gap_tail_exact.
        if (tcap) {
            #pragma unroll
            for (u32 i = 0; i < C30_CW; ++i) {
                const u64 X = ((u64)any[i] << 32) | (i ? any[i - 1] : aprev);
                const u64 e1 = ~X << 1, e2 = e1 & (e1 << 1), e4 = e2 & (e2 << 2), e8 = e4 & (e4 << 4);
                const u64 e16 = e8 & (e8 << 8), e22 = e16 & (e16 << 6);
                u32 cand = (u32)((X & e22) >> 32);
                while (cand) {
                    const u32 j = ctz(cand);
                    cand &= cand - 1u;
                    gap_tail_exact(cmap32, cw32, lim, ord, phis, pos_lo, 32u * (w + i) + j, gt, tcap);
                }
            }
        }
        // Gap walk: every counted prime in ascending order, h - h_prev into
        // the work-group histogram. The chunk's first prime takes the last
        // counted prime of word w - 1 (above), else of w - 2; none: the
        // segment's first prime (bin 0, h_prev = h), unless the look-back lay
        // wholly inside the counting range (every gap below 2^64 is < 1552).
        if (walk && cf != ~0u) {
            if (!aprev && w >= 2u) {
                #pragma unroll
                for (u32 c = 0; c < 8; ++c) {
                    const u32 pw = cmap32[(u64)c * cw32 + w - 2u] & range_mask32(w - 2u, lim[2 * c], lim[2 * c + 1]);
                    aprev |= pw;
                    if (pw) hp = max(hp, 15u * (32u * (w - 2u) + 31u - clz(pw)) + ph[c]);
                }
                if (!aprev && 30u * 32u * (w - 2u) >= pos_lo) err |= G30T_ERR_LOOK;
            }
            if (!aprev) hp = cf >> 1;
            #pragma unroll
            for (u32 i = 0; i < C30_CW; ++i) {
                // one prime per iteration (lanes diverge least): a block's
                // primes as a 15-bit mask of half offsets, refilled from the
                // next non-empty block
                u32 nb = any[i], m = 0u, hb = 0u;
                const u32 h0 = 15u * 32u * (w + i);
                while (nb | m) {
                    if (!m) {
                        const u32 b = ctz(nb);
                        nb &= nb - 1u;
                        #pragma unroll
                        for (u32 c = 0; c < 8; ++c) m |= ((v[c][i] >> b) & 1u) << ph[c];
                        hb = h0 + 15u * b;
                    }
                    const u32 h = hb + ctz(m);
                    m &= m - 1u;
                    const u32 d = h - hp;
                    hp = h;
                    atomic_inc(&lh[min(d, G30_HBINS - 1u)]);
                    if (d > dmax) { dmax = d; dpos = h; }
                }
            }
        }
    }
    #pragma unroll
    for (u32 c = 0; c < 8; ++c) {
        const u32 s = sub_group_reduce_add(ncl[c]);
        if (get_sub_group_local_id() == 0 && s) atomic_add(&sh[c], s);
    }
    #pragma unroll
    for (u32 j = 0; j < 3; ++j) {
        const u32 s = sub_group_reduce_add(ntw[j]);
        if (get_sub_group_local_id() == 0 && s) atomic_add(&sh[8 + j], s);
    }
    if (tcap) {
        // segment's first / last counted prime (the host stitches segments)
        const u32 f = sub_group_reduce_min(pfirst), l = sub_group_reduce_max(plast);
        if (get_sub_group_local_id() == 0) {
            if (f != ~0u) atomic_min(&sh[11], f);
            if (l) atomic_max(&sh[12], l);
        }
    }
    if (walk) {
        if (err) atomic_or(gt + 1, err);
        const u64 mx = sub_group_reduce_max(dmax ? ((u64)(2u * dmax) << 32) | (u64)(~(2u * dpos)) : (u64)0);
        if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = mx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 11u && sh[lid]) atomic_add(out + lid, sh[lid]);
    if (lid == 11u && sh[11] != ~0u) atomic_min(out + 11, sh[11]);
    if (lid == 12u && sh[12]) atomic_max(out + 12, sh[12]);
    if (walk) {
        for (u32 i = lid; i < G30_HBINS; i += get_local_size(0))
            if (lh[i]) atomic_add(hist + i, lh[i]);
        if (lid == 0) {
            u64 a = 0;
            for (u32 s = 0; s < get_num_sub_groups(); ++s) a = max(a, red[s]);
            part[get_group_id(0)] = a;
        }
    }
}

// Fast exact phase A (check30_xs3 / 4 / 6 / 8): as check30_x without
// first-occurrence tracking (the host uses check30_x only while a prefix bin
// still lacks a campaign first occurrence), with one register window per
// class array instead of a window reload per candidate.
// For class r the candidates p >= 7 live in the NA arrays k whose residue
// r - p is coprime to 30 (NA = 8, 6, 4 or 3 by r mod 3 / mod 5); p = 3 / 5
// (the first nd candidates, when useful) are tested by a direct window load.
// Candidate p of slot a (array k) reads bits U(p) + [k0, k0 + 32 W) of k,
// U(p) = (v_k - p) / 30. The host cuts each prefix into chunks in which every
// slot's candidates span < 32 bits; a slot's window (W + 2 words) is based at
// its smallest U in the chunk, Um = (v_k - pmax) / 30, so candidate p sits at
// t = (Um & 31) + d, d = (pmax - p) / 30 < 32: word offset t >> 5 (0 / 1),
// shift t & 31. Item width W per variant: as wide as the registers allow
// (8, 9, 12, 14 words: measured; wider spills without gain).
// Tables (static, make_stats_plan): xj per xall entry = slot | d << 8;
// xch per class (stride 64): [0] n_chunks | nd << 8 | NA << 16,
// [1] slot -> array nibbles, then per chunk jb | je << 16, pmax per slot (8).
#define XS_WMAX 24
#define XS_SLOT(A, TA, a) do { \
    if ((a) < NA) { \
        const u32 pm_ = xch[cb + 1u + (a)]; \
        if (pm_) { \
            const u32 kk_ = (amap >> (4u * (a))) & 15u; \
            const int um_ = (vb - 2 * (int)((phik >> (4u * kk_)) & 15u) - (int)pm_) / 30; \
            const int tm_ = (um_ >= 0) ? (um_ >> 5) : -((31 - um_) >> 5); \
            TA = (u32)um_ & 31u; \
            const int wb_ = (int)(k0 >> 5) + tm_; \
            const __global u32* S_ = cmap32 + (u64)kk_ * cw32; \
            if (wb_ >= 0 && wb_ + (int)(W + 2) <= (int)cw32) { \
                _Pragma("unroll") for (u32 i_ = 0; i_ < XS_WMAX + 2; ++i_) if (i_ < W + 2) A[i_] = S_[wb_ + (int)i_]; \
            } else { \
                _Pragma("unroll") for (u32 i_ = 0; i_ < XS_WMAX + 2; ++i_) if (i_ < W + 2) A[i_] = split_ld(S_, wb_ + (int)i_, cw32); \
            } \
        } \
    } } while (0)
#define XS_COMB(A, o) do { \
    if (rr == 0u) { \
        _Pragma("unroll") for (u32 i_ = 0; i_ < XS_WMAX; ++i_) if (i_ < W) { \
            const u32 v_ = A[i_ + (o)]; \
            cnt += popcount(v_ & ~acc[i_]); acc[i_] |= v_; } \
    } else { \
        const u32 l_ = 32u - rr; \
        _Pragma("unroll") for (u32 i_ = 0; i_ < XS_WMAX; ++i_) if (i_ < W) { \
            const u32 v_ = (A[i_ + (o)] >> rr) | (A[i_ + (o) + 1] << l_); \
            cnt += popcount(v_ & ~acc[i_]); acc[i_] |= v_; } \
    } } while (0)
#define XS_STEP(A, TA) do { \
    const u32 t_ = (TA) + d; \
    rr = t_ & 31u; \
    if (t_ < 32u) XS_COMB(A, 0); else XS_COMB(A, 1); } while (0)

__attribute__((always_inline)) inline void xs_body(
        __global const u32* restrict cmap32, const u32 cw32,
        __global const u16* restrict xall, __global const u32* restrict xtab,
        __global const u32* restrict xj, __global const u32* restrict xch,
        __global u32* restrict pending, __global u32* restrict pending_count,
        const u32 pending_cap, const u32 items_pad, const u32 seg_evens,
        const u32 sm30, const int tb, const u32 phik,
        __global u32* restrict hist, const u64 cls, __local u32* sh,
        const u32 NA, const u32 W)
{
    const u32 lid = get_local_id(0);
    const u32 gid = get_global_id(0);
    const u32 ci = gid / items_pad;                    // uniform per work-group
    const u32 cr = (u32)(cls >> (4u * ci)) & 15u;
    const u32 blk = gid - ci * items_pad;
    const u32 r = 2u * cr;
    const u32 e0 = ((r + 30u - sm30) % 30u) >> 1;
    const u32 ec = (seg_evens > e0) ? (seg_evens - e0 + 14u) / 15u : 0u;
    const u32 k0 = blk * (W * 32u);
    const bool live = k0 < ec;
    const u32 xo = xtab[cr];
    for (u32 j = lid; j < X_KMAX; j += WG_SIZE) sh[j] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    u32 acc[XS_WMAX];
    #pragma unroll
    for (u32 i = 0; i < XS_WMAX; ++i) {
        if (i >= W) break;
        const u32 kb = k0 + 32u * i;
        const u32 n = (live && kb < ec) ? min(32u, ec - kb) : 0u;
        acc[i] = (n >= 32u) ? 0u : ~((1u << n) - 1u);
    }
    const int vb = tb + 2 * (int)e0;
    const u32 h0 = xch[cr * 64u], amap = xch[cr * 64u + 1u];
    const u32 nch = h0 & 0xFFu, nd = (h0 >> 8) & 0xFFu;
    u32 rr;
    // p = 3 / 5: direct window per candidate
    for (u32 j = 0; j < nd; ++j) {
        const u32 p = xall[xo + j];
        const u32 k = W30_IDX[(r + 30u - p % 30u) % 30u];
        const int U = (vb - 2 * (int)((phik >> (4u * k)) & 15u) - (int)p) / 30;
        const int tw = (U >= 0) ? (U >> 5) : -((31 - U) >> 5);
        const int wb = (int)(k0 >> 5) + tw;
        const __global u32* S = cmap32 + (u64)k * cw32;
        u32 m[XS_WMAX + 2];
        #pragma unroll
        for (u32 i = 0; i <= XS_WMAX; ++i) if (i <= W) m[i] = split_ld(S, wb + (int)i, cw32);
        rr = (u32)U & 31u;
        u32 cnt = 0u;
        XS_COMB(m, 0);
        const u32 s = sub_group_reduce_add(cnt);
        if (get_sub_group_local_id() == 0 && s) atomic_add(&sh[j], s);
    }
    u32 w0[XS_WMAX + 2], w1[XS_WMAX + 2], w2[XS_WMAX + 2], w3[XS_WMAX + 2];
    u32 w4[XS_WMAX + 2], w5[XS_WMAX + 2], w6[XS_WMAX + 2], w7[XS_WMAX + 2];
    u32 ta0 = 0, ta1 = 0, ta2 = 0, ta3 = 0, ta4 = 0, ta5 = 0, ta6 = 0, ta7 = 0;
    for (u32 c = 0; c < nch; ++c) {
        const u32 cb = cr * 64u + 2u + 9u * c;
        const u32 jj = xch[cb];
        XS_SLOT(w0, ta0, 0); XS_SLOT(w1, ta1, 1); XS_SLOT(w2, ta2, 2); XS_SLOT(w3, ta3, 3);
        XS_SLOT(w4, ta4, 4); XS_SLOT(w5, ta5, 5); XS_SLOT(w6, ta6, 6); XS_SLOT(w7, ta7, 7);
        for (u32 j = jj & 0xFFFFu; j < (jj >> 16); ++j) {
            const u32 x = xj[xo + j];
            const u32 a = x & 7u, d = x >> 8;
            u32 cnt = 0u;
            if (NA <= 4u) {                            // 2-level tree on the slot
                if (a < 2u) { if (a == 0u) XS_STEP(w0, ta0); else XS_STEP(w1, ta1); }
                else if (NA == 3u || a == 2u) XS_STEP(w2, ta2);
                else XS_STEP(w3, ta3);
            } else {
                if (a < 4u) {
                    if (a < 2u) { if (a == 0u) XS_STEP(w0, ta0); else XS_STEP(w1, ta1); }
                    else { if (a == 2u) XS_STEP(w2, ta2); else XS_STEP(w3, ta3); }
                } else if (a < 6u || NA == 6u) {
                    if (a == 4u) XS_STEP(w4, ta4); else XS_STEP(w5, ta5);
                } else { if (a == 6u) XS_STEP(w6, ta6); else XS_STEP(w7, ta7); }
            }
            const u32 s = sub_group_reduce_add(cnt);
            if (get_sub_group_local_id() == 0 && s) atomic_add(&sh[j], s);
        }
    }

    u32 cnt = 0;
    #pragma unroll
    for (u32 i = 0; i < XS_WMAX; ++i) if (i < W) cnt += popcount(~acc[i]);
    const u32 off = sub_group_scan_exclusive_add(cnt);
    const u32 tot = sub_group_reduce_add(cnt);
    u32 base = 0;
    if (get_sub_group_local_id() == 0 && tot) base = atomic_add(pending_count, tot);
    base = sub_group_broadcast(base, 0) + off;
    if (cnt) {
        #pragma unroll
        for (u32 i = 0; i < XS_WMAX; ++i) {
            if (i >= W) break;
            u32 miss = ~acc[i];
            while (miss) {
                const u32 b = (u32)ctz(miss);
                miss &= miss - 1u;
                if (base < pending_cap) pending[base] = e0 + 15u * (k0 + 32u * i + b);
                ++base;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const u32 kx = xtab[15u + cr];
    for (u32 j = lid; j < kx; j += WG_SIZE)
        if (sh[j]) atomic_add(hist + xo + j, sh[j]);
}

/// @pre    as check30_x; tables from make_stats_plan; cls = the variant's
///         classes (nibble i = class of work-group row i), items_pad for W.
/// @post   as check30_x for those classes, without first[] updates.
#define XS_KERNEL(NAME, NA_, W_) \
__kernel void NAME( \
        __global const u32* restrict cmap32, const u32 cw32, \
        __global const u16* restrict xall, __global const u32* restrict xtab, \
        __global const u32* restrict xj, __global const u32* restrict xch, \
        __global u32* restrict pending, __global u32* restrict pending_count, \
        const u32 pending_cap, const u32 items_pad, const u32 seg_evens, \
        const u32 sm30, const int tb, const u32 phik, \
        __global u32* restrict hist, const u64 cls) \
{ \
    __local u32 sh[X_KMAX]; \
    xs_body(cmap32, cw32, xall, xtab, xj, xch, pending, pending_count, pending_cap, \
            items_pad, seg_evens, sm30, tb, phik, hist, cls, sh, NA_, W_); \
}
#ifndef XS_W8
#define XS_W8 8
#endif
#ifndef XS_W6
#define XS_W6 9
#endif
#ifndef XS_W4
#define XS_W4 12
#endif
#ifndef XS_W3
#define XS_W3 14
#endif
XS_KERNEL(check30_xs8, 8u, XS_W8)
XS_KERNEL(check30_xs6, 6u, XS_W6)
XS_KERNEL(check30_xs4, 4u, XS_W4)
XS_KERNEL(check30_xs3, 3u, XS_W3)

// ----------------------------------------------------------------------------
// Hit lists for the bucketed primes (GB_HITLIST, replaces the bucket ring).
// The bucket re-files every hit (8 B read + 8 B write + the wheel residue
// recomputed from (off, p): two 60 KB table gathers) at ~140 ps per hit,
// while a mark itself costs ~15 ps. Here each prime is touched once per batch
// of K segments: hl_harvest finds its first markable odd multiple from the
// batch's q_first (first_odd_multiple, the reference path's own function),
// walks its multiples p m with m
// coprime to 30030 (the multiplier wheel) through the batch window and
// appends each hit as a bare u32 segment offset to the list of every segment
// containing it (j, and j - 1 inside the q_odds_full - delta overlap). Each
// segment's list is split into one fixed-capacity region per harvest
// work-group (capg entries, a power of two; counts per (segment, group)), so
// one walk suffices: no count pass, no global allocation atomics. The apply
// (fused into the packed kernel) streams the regions and skips their unused
// tails. Overflow sets the error flag (the run fails closed). Same bits as
// the bucket: GB_BUCKET_VERIFY checks the result against tile + packed over
// ALL large primes.
// ----------------------------------------------------------------------------
#if LARGE_OH_BM && WHEEL_W == 30030
#ifndef HL_MAX_K
#define HL_MAX_K 128
#endif
// q_odds of batch segment j (only the batch's last may be the run's short tail).
#define HL_QO(j) (((j) == nseg - 1u) ? q_odds_tail : q_odds_full)
/// Harvests the hits of primes [a, b) for segments [k0, k0 + nseg) of the batch.
/// @param q_first      q_first of segment k0 (odd).
/// @param q_odds_tail  q_odds of segment k0 + nseg - 1 (the run's last may be short).
/// @pre    local size >= HL_MAX_K >= nseg; delta + p_max < 2^32; 2 delta < 2^31;
///         delta <= q_odds_full < 2 delta (all host-checked); primes > 13,
///         < 2^32; segment j covers batch bits [j delta, j delta + q_odds_j);
///         lists has nseg * groups * capg entries, capg a power of two.
/// @post   region (j, g) of lists holds gcnt[j * groups + g] of the hits in
///         segment k0 + j of work-group g's primes, every odd multiple p m
///         (m coprime to 30030, p m >= p^2) of every prime appearing in exactly
///         one region of each segment containing it, as q offsets
///         (q - q_first_j) / 2; a region overflow sets *err.
__kernel void hl_harvest(
        __global const u32* restrict primes,
        const u32 a, const u32 b,
        const u64 q_first,
        const u32 delta, const u32 q_odds_full,
        const u32 nseg, const u32 q_odds_tail,
        const float inv_delta,
        __global u32* restrict lists, const u32 capg,
        __global u32* restrict gcnt,
        volatile __global u32* restrict err)
{
    __local u32 lpos[HL_MAX_K];
    __local uchar lstp[WHL_N];
    const u32 lid = get_local_id(0), wg = get_local_size(0);
    const u32 grp = get_group_id(0), ngrp = get_num_groups(0);
    for (u32 w = lid; w < WHL_N; w += wg) lstp[w] = WHL_STEP[w];
    if (lid < HL_MAX_K) lpos[lid] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);
    const u32 qfw = (u32)(q_first % 30030UL);
    const u64 wend = (u64)(nseg - 1u) * delta + q_odds_tail;   // window [0, wend) in bits
    for (u32 c0 = grp * wg; c0 < b - a; c0 += ngrp * wg) {
        const u32 i = a + c0 + lid;
        if (i >= b) break;
        const u32 p = primes[i];
        const u64 xo = (first_odd_multiple((u64)p, q_first) - q_first) >> 1;
        if (xo < wend) {              // first odd multiple inside the window
            // multiplier of q = q_first + 2 xo mod 30030 (as mark_multiples_whl),
            // then the first multiplier coprime to 30030
            u32 mm = (qfw + 2u * (u32)(xo % 15015UL)) % 30030u;
            mm = (mm * WHL_INV[p % 30030u]) % 30030u;
            u64 x = xo;
            while (WHL_IDX[mm] == WHL_MISS) { x += p; mm += 2u; if (mm >= 30030u) mm -= 30030u; }
            u32 w = WHL_IDX[mm];
            // (j, off) of x by a float quotient, exact by the fix-up loops
            // (off in (-delta, 2 delta) fits an int: 2 delta < 2^31)
            for (;;) {
                u32 j = (u32)(((float)(u32)(x >> 32) * 4294967296.0f + (float)(u32)x) * inv_delta);
                int o = (int)((u32)x - j * delta);
                while (o < 0) { o += (int)delta; --j; }
                while (o >= (int)delta) { o -= (int)delta; ++j; }
                if (j > nseg) break;
                const u32 off = (u32)o;
                if (j < nseg && off < HL_QO(j)) {
                    const u32 idx = atomic_inc(&lpos[j]);
                    if (idx < capg) lists[((size_t)j * ngrp + grp) * capg + idx] = off;
                }
                if (j >= 1u && j - 1u < nseg && off + delta < HL_QO(j - 1u)) {
                    const u32 idx = atomic_inc(&lpos[j - 1u]);
                    if (idx < capg) lists[((size_t)(j - 1u) * ngrp + grp) * capg + idx] = off + delta;
                }
                x += (u64)p * lstp[w];
                if (++w == WHL_N) w = 0u;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < nseg) {
        const u32 c = lpos[lid];
        gcnt[lid * ngrp + grp] = min(c, capg);
        if (c > capg) atomic_or(err, BKT_ERR_OVERFLOW);
    }
}
#undef HL_QO

/// Marks segment j's harvested hits: the regions of `groups` harvest
/// work-groups, region g holding cnt[g] entries (grid-stride over entry pairs;
/// slots past a region's count are skipped).
/// @pre    capg a power of two >= 2; lists 8-byte aligned.
/// @post   every listed offset < q_odds is cleared; any other sets *err.
inline void hl_apply_body(volatile __global u64* restrict prime_map,
                          __global const u32* restrict lists, const u32 capg,
                          __global const u32* restrict gcnt, const u32 j, const u32 groups,
                          const u32 q_odds, volatile __global u32* restrict err,
                          const cmap g, const u32 gid, const u32 gsize)
{
    const u32 sh = 31u - clz(capg);
    const u32 nsl = groups << sh;
    const __global u32* cnt = gcnt + (size_t)j * groups;
    const __global u64* in = (const __global u64*)(lists + (size_t)j * groups * capg);
    for (u32 s = 2u * gid; s < nsl; s += 2u * gsize) {
        const u32 c = cnt[s >> sh];
        const u32 k = s & (capg - 1u);
        if (k >= c) continue;
        const u64 w = bkt_load(in + (s >> 1));                 // two entries, L3-uncached
        const u32 o0 = (u32)w, o1 = (u32)(w >> 32);
        if (o0 < q_odds) map_clear_w((volatile __global u32*)prime_map, o0, g);
        else atomic_or(err, BKT_ERR_ENTRY);
        if (k + 1u < c) {
            if (o1 < q_odds) map_clear_w((volatile __global u32*)prime_map, o1, g);
            else atomic_or(err, BKT_ERR_ENTRY);
        }
    }
}

/// Marks segment j's harvested hits (hl_apply_body as its own launch, GB_HL_FUSE=0).
__kernel void hl_apply(
        volatile __global u64* restrict prime_map,
        __global const u32* restrict lists, const u32 capg,
        __global const u32* restrict gcnt, const u32 j, const u32 groups,
        const u32 q_odds,
        volatile __global u32* restrict err,
        const u32 cm, const u32 phi1, const u32 phi5, const u32 off5)
{
    const cmap g = {cm, phi1, phi5, off5};
    hl_apply_body(prime_map, lists, capg, gcnt, j, groups, q_odds, err, g,
                  get_global_id(0), get_global_size(0));
}

// sieve_large_packed with hl_apply fused in (HL_FUSE on the host): the list
// marks and the packed marks are atomic ANDs on the same map, so they
// commute; as a separate launch hl_apply ran alone, latency-bound (~4.8M
// marks in 0.17 ms per 425M segment). Here every packed work-item first
// takes list pairs (grid-stride, coalesced u64 loads), then its primes.
/// @pre    as sieve_large_packed and hl_apply (same segment, same map).
/// @post   as sieve_large_packed followed by hl_apply.
__attribute__((intel_reqd_sub_group_size(LARGE_PACK_SG)))
__kernel void sieve_large_packed_hl(
        volatile __global u64* restrict prime_map,
        __global const u32* restrict bases,
        __global const u32* restrict gaps4,
        const u32 large_count,
        const u64 q_odds,
        const u64 q_first,
        const u64 q_last,
        const u32 cm, const u32 phi1, const u32 phi5, const u32 off5,
        __global const u32* restrict lists, const u32 capg,
        __global const u32* restrict gcnt, const u32 j, const u32 groups,
        volatile __global u32* restrict err)
{
    const cmap g = {cm, phi1, phi5, off5};
    hl_apply_body(prime_map, lists, capg, gcnt, j, groups, (u32)q_odds, err, g,
                  get_global_id(0), get_global_size(0));
#ifdef ABL_HL2      // timing only, map EXACT: the list marks twice (atomic AND is idempotent)
    hl_apply_body(prime_map, lists, capg, gcnt, j, groups, (u32)q_odds, err, g,
                  get_global_id(0), get_global_size(0));
#endif
    large_packed_body(prime_map, bases, gaps4, large_count, q_odds, q_first, q_last, g);
#ifdef ABL_PACK2    // timing only, map EXACT: the packed marks twice
    large_packed_body(prime_map, bases, gaps4, large_count, q_odds, q_first, q_last, g);
#endif
}
#endif
