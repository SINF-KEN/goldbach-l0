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
#define WG_SIZE           256
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
#define WORDS_PER_ITEM    4
#define LARGE_PRIMES_PER_ITEM 4
// Small-prime mask sieve: primes < MASK_MAX_P are applied as whole-word ANDs
// from a precomputed host table (mask[p][phase]) instead of per-bit RMWs.
// 0 disables the path (falls back to the scalar loop for ALL primes).
// Primes below this bound use the mask table. 64 -> 17 odd primes
// (3..61; 2 is not in the list), 17*64 = 1088 mask words of 8 B =
// 8.5 KB device memory. Tunable.
#define MASK_MAX_P        64
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
// x mod p for p < 2^30 from a pre-split x (the float-float math of mod_u64_u32).
// The float-float quotient q = qh + rint(ql) carries ~46 bits, so
// |q - floor(x / p)| <= (x / p) 2^-46 + 1: at most 1 for p >= 2^20 and a few
// units below (measured: 2), i.e. the wrap-around remainder r = x - q p
// satisfies |r| < 2^31 for every p < 2^30 (|r| <= 2.2 p < 2^21 for p < 2^20,
// < 2 p < 2^31 above). Hence r is exact in int, the selects fix the usual
// +-1 error and the guarded loops the rest. (Measured by CPU emulation over
// 1.4e7 (x, p) pairs: max |r| = 5.4e8.)
// NOT for small p: a GPU self-test against the exact remainder (2026-11-17,
// HISTORY.md 9.2z) found wrong results for every prime 67..251 at x ~ 4e18
// and for primes up to 491 at x up to 9.2e18 (the correction term no longer
// fits), and none for 512..2^31 (1.2e9 (x, p) pairs). Callers keep p >=
// MOD_FF_MIN_P = 4096 (host: the split prime; mid_setup uses the exact %).
/// @pre    x from ff_split of a value < 2^63; 4096 <= p < 2^30;
///         x / p >= 2^23 (callers: x >= 2^60), so qh >= 2^23.
/// @return x mod p, exact.
inline u32 mod_ff(const ff_x x, const u32 p)
{
    const float pf = (float)p;
    const int pr = (int)p - (int)pf;                 // p, pf <= 2^30: exact in int
    const float ih = 1.0f / pf;
    const float e1 = fma(-pf, ih, 1.0f);
    const float il = ih * (e1 - (float)pr * ih);
    const float qh = rint(x.xh * ih);
    const float ql = fma(x.xh, ih, -qh) + x.xh * il + x.xl * ih;
    const u32 q32 = float_int_low32_big(qh) + (u32)(int)rint(ql);
    int r = (int)(x.x32 - q32 * p);
    r += (r < 0) ? (int)p : 0;                       // the usual +-1 quotient errors
    r -= (r >= (int)p) ? (int)p : 0;
    if ((u32)r >= p) {                               // one test for both directions
        while (r < 0) r += (int)p;
        while (r >= (int)p) r -= (int)p;
    }
    return (u32)r;
}

/// mod_ff that also returns the quotient: x = Q p + r with *qlo = Q mod 2^32
/// and *qhf = qh, the float estimate of Q (|qh - Q| < 2^-23 Q + 2^20).
/// @pre    as mod_ff.
inline u32 mod_ff_q(const ff_x x, const u32 p, u32* qlo, float* qhf)
{
    const float pf = (float)p;
    const int pr = (int)p - (int)pf;
    const float ih = 1.0f / pf;
    const float e1 = fma(-pf, ih, 1.0f);
    const float il = ih * (e1 - (float)pr * ih);
    const float qh = rint(x.xh * ih);
    const float ql = fma(x.xh, ih, -qh) + x.xh * il + x.xl * ih;
    u32 q32 = float_int_low32_big(qh) + (u32)(int)rint(ql);
    int r = (int)(x.x32 - q32 * p);
    q32 -= (r < 0) ? 1u : 0u;
    r += (r < 0) ? (int)p : 0;
    q32 += (r >= (int)p) ? 1u : 0u;
    r -= (r >= (int)p) ? (int)p : 0;
    if ((u32)r >= p) {
        while (r < 0) { r += (int)p; --q32; }
        while (r >= (int)p) { r -= (int)p; ++q32; }
    }
    *qlo = q32;
    *qhf = qh;
    return (u32)r;
}

/// x mod p without a 64-bit division (float-float quotient, see above; exact).
/// @pre    x < 2^63, 1 <= p < 2^32.
/// @return x mod p.
inline u32 mod_u64_u32(const u64 x, const u32 p)
{
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
    const float f = 1.0f / (float)p;
    const u64 qt = (u64)((float)x * f);             // |x/p - qt| <~ 2^-22 x/p
    long r = (long)(x - qt * (u64)p);                // |r| <~ 2^-22 x, exact
    r -= (long)((float)r * f) * (long)p;             // |r| <~ 2p (p >= 2^19)
    // (a third refinement step measured slower; small p just take more
    // fix-up iterations below — still exact)
    while (r < 0) r += (long)p;
    while (r >= (long)p) r -= (long)p;
    return (u32)r;
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
// AND mask clearing bit (b mod 32) of a u32 word: ~(1 << (b & 31)) is and +
// shl + not; rotate(~1, b) is one native rol (the count is taken mod 32).
#define CLR_MASK(b) rotate(0xFFFFFFFEu, (u32)(b))
// the same with ~1 in a register C1 (the hot loops: see make_prime_map's c1v)
#define CLR_MASK_R(C1, b) rotate((C1), (u32)(b))
typedef struct { u32 cm, phi1, phi5, off5; } cmap;
// LSC cache-control operands of the IGC load / store builtins (IGC enum
// values: LDCC 1 = L1UC_L3UC, 2 = L1UC_L3C, 5 = L1S_L3UC).
enum LSC_LDCC { LSC_LDCC_DEFAULT = 0 };
#define LDCC_L1S_L3UC 5          // streaming loads that bypass L3
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
    u32 ad = bit;
    if (g.cm == 2u) {                                 // multiplier coprime to 30030: k valid
        const u32 t = bit / 15u;
        ad = c30_class(bit - 15u * t, g) * g.off5 + t;
    } else if (g.cm) {                                // uniform
        const u32 t = bit / 3u;
        ad = t + ((bit - 3u * t == g.phi5) ? g.off5 : 0u);
    }
    atomic_and(m32 + (ad >> 5), CLR_MASK(ad));
}

// map_clear_w that first reads the word (L1-uncached, L2-cached load) and
// skips the atomic when the bit is already clear (round 5, 2026-11-18).
// For the packed primes (p >= cbrt(q)) a mark p m is redundant iff m is
// composite (~75 % of the marks), and m's factor (<= sqrt(m) < 1.3M) is a
// tile / mid-window prime whose clears are complete before the packed kernel
// starts (barrier after mid_window), so the read sees them: one L2 read
// instead of one L2 atomic on 3 of 4 marks (gpu -6 % at 4e18). Exact under
// any interleaving: map bits only go 1 -> 0 while this runs (every concurrent
// writer AND-clears), so a 0 read is a real clear and a 1 read just marks.
// (2026-10-19 the same filter lost 3 %: mid windows did not exist, most
// redundant marks were not yet visible.)
uint __builtin_IB_lsc_load_global_uint(__global uint* base, int immElemOff, enum LSC_LDCC cacheOpt);
#define LDCC_L1UC_L3C 2          // L1 uncached, L3 (= Xe2 L2) cached
/// @pre    as map_clear_w; every other writer of the map during the call only
///         clears bits (atomic AND).
/// @post   as map_clear_w.
inline void map_clear_chk(volatile __global u32* m32, const u32 bit, const cmap g)
{
    if (g.cm != 2u) { map_clear_w(m32, bit, g); return; }
    const u32 t = bit / 15u;
    const u32 ad = c30_class(bit - 15u * t, g) * g.off5 + t;
    const u32 w = __builtin_IB_lsc_load_global_uint((__global u32*)(m32 + (ad >> 5)), 0,
                                                    (enum LSC_LDCC)LDCC_L1UC_L3C);
    if ((w >> (ad & 31u)) & 1u) atomic_and(m32 + (ad >> 5), CLR_MASK(ad));
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
// calling thread owns every word in [bit0, end), so a plain RMW would be
// safe; each mark is a posted SLM atomic AND anyway (cheaper than load +
// store, +2.5%), and make_prime_map relies on it: the medium stage's
// cross-thread atomics run without a barrier after the owned walk (+3.4%).
// Overflow-free loop condition (end - bit > p), no headroom assumed.
// ----------------------------------------------------------------------------
/// @pre    the calling thread owns every SLM word of [bit0, end); end < 2^32.
/// @post   tile bits bit0, bit0 + p, ... < end are cleared (SLM atomic ANDs).
// AND a 64-bit mask into tile word w as two 32-bit SLM atomics (little
// endian, as the t32 views above); halves that clear nothing are skipped.
inline void tile_and64_atomic(__local u64* tile, const u32 w, const u64 m)
{
    volatile __local u32* t = (volatile __local u32*)tile + 2u * w;
    if ((u32)m != 0xffffffffu) atomic_and(t, (u32)m);
    if ((u32)(m >> 32) != 0xffffffffu) atomic_and(t + 1, (u32)(m >> 32));
}

inline void mark_owned_u32(__local u64* tile, const u64 bit0, const u64 end, const u64 p)
{
    if (bit0 >= end) return;
    __local u32* t32 = (__local u32*)tile;
    const u32 e = (u32)end;
    const u32 step = (u32)p;
    u32 bit = (u32)bit0;
    for (;;) {
        atomic_and((volatile __local u32*)&t32[bit >> 5], CLR_MASK(bit));
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
#define WHEEL_W 30030           // MUST match the host's WHEEL_W (30 / 210 / 2310 tables removed)
// ---- generated by the v1.9.0 wheel generator (odd multipliers coprime to W) ----
#define WHL_N 5760
#define WHL_MISS 0xFFFFu
__constant ushort WHL_IDX[30030] = { 65535, 0, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1, 65535, 2, 65535, 65535, 65535, 3, 65535, 65535, 65535, 65535, 65535, 4, 65535, 5, 65535, 65535, 65535, 65535, 65535, 6, 65535, 65535, 65535, 7, 65535, 8, 65535, 65535, 65535, 9, 65535, 65535, 65535, 65535, 65535, 10, 65535, 65535, 65535, 65535, 65535, 11, 65535, 12, 65535, 65535, 65535, 65535, 65535, 13, 65535, 65535, 65535, 14, 65535, 15, 65535, 65535, 65535, 65535, 65535, 16, 65535, 65535, 65535, 17, 65535, 65535, 65535, 65535, 65535, 18, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 19, 65535, 65535, 65535, 20, 65535, 21, 65535, 65535, 65535, 22, 65535, 23, 65535, 65535, 65535, 24, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 25, 65535, 65535, 65535, 26, 65535, 65535, 65535, 65535, 65535, 27, 65535, 28, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 29, 65535, 30, 65535, 65535, 65535, 65535, 65535, 31, 65535, 65535, 65535, 65535, 65535, 32, 65535, 65535, 65535, 33, 65535, 65535, 65535, 65535, 65535, 34, 65535, 65535, 65535, 65535, 65535, 35, 65535, 36, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 37, 65535, 38, 65535, 65535, 65535, 39, 65535, 40, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 41, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 42, 65535, 65535, 65535, 43, 65535, 44, 65535, 65535, 65535, 45, 65535, 65535, 65535, 65535, 65535, 46, 65535, 47, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 48, 65535, 65535, 65535, 65535, 65535, 49, 65535, 65535, 65535, 65535, 65535, 50, 65535, 65535, 65535, 65535, 65535, 51, 65535, 52, 65535, 65535, 65535, 65535, 65535, 53, 65535, 65535, 65535, 54, 65535, 55, 65535, 65535, 65535, 65535, 65535, 56, 65535, 65535, 65535, 57, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 58, 65535, 65535, 65535, 59, 65535, 60, 65535, 65535, 65535, 61, 65535, 65535, 65535, 65535, 65535, 62, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 63, 65535, 65535, 65535, 65535, 65535, 64, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65, 65535, 66, 65535, 65535, 65535, 67, 65535, 65535, 65535, 65535, 65535, 68, 65535, 69, 65535, 65535, 65535, 65535, 65535, 70, 65535, 65535, 65535, 65535, 65535, 71, 65535, 65535, 65535, 65535, 65535, 72, 65535, 65535, 65535, 73, 65535, 65535, 65535, 65535, 65535, 74, 65535, 75, 65535, 65535, 65535, 65535, 65535, 76, 65535, 65535, 65535, 77, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 78, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 79, 65535, 80, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 81, 65535, 82, 65535, 65535, 65535, 83, 65535, 84, 65535, 65535, 65535, 85, 65535, 65535, 65535, 65535, 65535, 86, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 87, 65535, 65535, 65535, 88, 65535, 89, 65535, 65535, 65535, 90, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 91, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 92, 65535, 65535, 65535, 93, 65535, 94, 65535, 65535, 65535, 65535, 65535, 95, 65535, 65535, 65535, 96, 65535, 65535, 65535, 65535, 65535, 97, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 98, 65535, 99, 65535, 65535, 65535, 100, 65535, 101, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 102, 65535, 65535, 65535, 65535, 65535, 103, 65535, 65535, 65535, 104, 65535, 65535, 65535, 65535, 65535, 105, 65535, 65535, 65535, 65535, 65535, 106, 65535, 65535, 65535, 65535, 65535, 107, 65535, 108, 65535, 65535, 65535, 65535, 65535, 109, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 110, 65535, 111, 65535, 65535, 65535, 112, 65535, 65535, 65535, 65535, 65535, 113, 65535, 114, 65535, 65535, 65535, 65535, 65535, 115, 65535, 65535, 65535, 65535, 65535, 116, 65535, 65535, 65535, 117, 65535, 118, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 119, 65535, 120, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 121, 65535, 122, 65535, 65535, 65535, 123, 65535, 65535, 65535, 65535, 65535, 124, 65535, 65535, 65535, 65535, 65535, 125, 65535, 126, 65535, 65535, 65535, 65535, 65535, 127, 65535, 65535, 65535, 65535, 65535, 128, 65535, 65535, 65535, 129, 65535, 65535, 65535, 65535, 65535, 130, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 131, 65535, 65535, 65535, 65535, 65535, 132, 65535, 65535, 65535, 133, 65535, 134, 65535, 65535, 65535, 65535, 65535, 135, 65535, 65535, 65535, 136, 65535, 65535, 65535, 65535, 65535, 137, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 138, 65535, 65535, 65535, 139, 65535, 140, 65535, 65535, 65535, 65535, 65535, 141, 65535, 65535, 65535, 142, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 143, 65535, 65535, 65535, 65535, 65535, 144, 65535, 65535, 65535, 145, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 146, 65535, 65535, 65535, 147, 65535, 65535, 65535, 65535, 65535, 148, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 149, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 150, 65535, 151, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 152, 65535, 153, 65535, 65535, 65535, 65535, 65535, 154, 65535, 65535, 65535, 155, 65535, 156, 65535, 65535, 65535, 157, 65535, 158, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 159, 65535, 160, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 161, 65535, 162, 65535, 65535, 65535, 163, 65535, 164, 65535, 65535, 65535, 165, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 166, 65535, 65535, 65535, 167, 65535, 168, 65535, 65535, 65535, 169, 65535, 65535, 65535, 65535, 65535, 170, 65535, 65535, 65535, 65535, 65535, 171, 65535, 172, 65535, 65535, 65535, 65535, 65535, 173, 65535, 65535, 65535, 174, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 175, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 176, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 177, 65535, 65535, 65535, 178, 65535, 179, 65535, 65535, 65535, 180, 65535, 65535, 65535, 65535, 65535, 181, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 182, 65535, 65535, 65535, 65535, 65535, 183, 65535, 65535, 65535, 184, 65535, 65535, 65535, 65535, 65535, 185, 65535, 65535, 65535, 65535, 65535, 186, 65535, 65535, 65535, 65535, 65535, 187, 65535, 188, 65535, 65535, 65535, 65535, 65535, 189, 65535, 65535, 65535, 65535, 65535, 190, 65535, 65535, 65535, 191, 65535, 192, 65535, 65535, 65535, 193, 65535, 65535, 65535, 65535, 65535, 194, 65535, 195, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 196, 65535, 197, 65535, 65535, 65535, 198, 65535, 199, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 200, 65535, 201, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 202, 65535, 203, 65535, 65535, 65535, 65535, 65535, 204, 65535, 65535, 65535, 205, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 206, 65535, 65535, 65535, 65535, 65535, 207, 65535, 65535, 65535, 208, 65535, 209, 65535, 65535, 65535, 210, 65535, 65535, 65535, 65535, 65535, 211, 65535, 65535, 65535, 65535, 65535, 212, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 213, 65535, 65535, 65535, 214, 65535, 215, 65535, 65535, 65535, 65535, 65535, 216, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 217, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 218, 65535, 65535, 65535, 219, 65535, 220, 65535, 65535, 65535, 65535, 65535, 221, 65535, 65535, 65535, 222, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 223, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 224, 65535, 65535, 65535, 65535, 65535, 225, 65535, 226, 65535, 65535, 65535, 227, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 228, 65535, 65535, 65535, 65535, 65535, 229, 65535, 65535, 65535, 65535, 65535, 230, 65535, 65535, 65535, 231, 65535, 232, 65535, 65535, 65535, 233, 65535, 65535, 65535, 65535, 65535, 234, 65535, 235, 65535, 65535, 65535, 65535, 65535, 236, 65535, 65535, 65535, 237, 65535, 65535, 65535, 65535, 65535, 238, 65535, 239, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 240, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 241, 65535, 242, 65535, 65535, 65535, 243, 65535, 244, 65535, 65535, 65535, 245, 65535, 65535, 65535, 65535, 65535, 246, 65535, 247, 65535, 65535, 65535, 65535, 65535, 248, 65535, 65535, 65535, 249, 65535, 250, 65535, 65535, 65535, 251, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 252, 65535, 253, 65535, 65535, 65535, 65535, 65535, 254, 65535, 65535, 65535, 65535, 65535, 255, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 256, 65535, 65535, 65535, 65535, 65535, 257, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 258, 65535, 65535, 65535, 259, 65535, 260, 65535, 65535, 65535, 261, 65535, 262, 65535, 65535, 65535, 263, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 264, 65535, 65535, 65535, 65535, 65535, 265, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 266, 65535, 65535, 65535, 267, 65535, 65535, 65535, 65535, 65535, 268, 65535, 269, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 270, 65535, 65535, 65535, 271, 65535, 272, 65535, 65535, 65535, 273, 65535, 65535, 65535, 65535, 65535, 274, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 275, 65535, 65535, 65535, 276, 65535, 277, 65535, 65535, 65535, 278, 65535, 279, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 280, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 281, 65535, 282, 65535, 65535, 65535, 283, 65535, 284, 65535, 65535, 65535, 285, 65535, 65535, 65535, 65535, 65535, 286, 65535, 287, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 288, 65535, 289, 65535, 65535, 65535, 290, 65535, 65535, 65535, 65535, 65535, 291, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 292, 65535, 65535, 65535, 65535, 65535, 293, 65535, 65535, 65535, 294, 65535, 295, 65535, 65535, 65535, 65535, 65535, 296, 65535, 65535, 65535, 297, 65535, 65535, 65535, 65535, 65535, 298, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 299, 65535, 65535, 65535, 300, 65535, 65535, 65535, 65535, 65535, 301, 65535, 302, 65535, 65535, 65535, 303, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 304, 65535, 65535, 65535, 65535, 65535, 305, 65535, 65535, 65535, 306, 65535, 65535, 65535, 65535, 65535, 307, 65535, 308, 65535, 65535, 65535, 309, 65535, 65535, 65535, 65535, 65535, 310, 65535, 311, 65535, 65535, 65535, 65535, 65535, 312, 65535, 65535, 65535, 65535, 65535, 313, 65535, 65535, 65535, 314, 65535, 65535, 65535, 65535, 65535, 315, 65535, 65535, 65535, 65535, 65535, 316, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 317, 65535, 65535, 65535, 65535, 65535, 318, 65535, 65535, 65535, 319, 65535, 320, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 321, 65535, 322, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 323, 65535, 324, 65535, 65535, 65535, 325, 65535, 326, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 327, 65535, 328, 65535, 65535, 65535, 65535, 65535, 329, 65535, 65535, 65535, 330, 65535, 331, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 332, 65535, 65535, 65535, 65535, 65535, 333, 65535, 334, 65535, 65535, 65535, 65535, 65535, 335, 65535, 65535, 65535, 336, 65535, 337, 65535, 65535, 65535, 65535, 65535, 338, 65535, 65535, 65535, 339, 65535, 65535, 65535, 65535, 65535, 340, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 341, 65535, 65535, 65535, 65535, 65535, 342, 65535, 65535, 65535, 343, 65535, 344, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 345, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 346, 65535, 65535, 65535, 65535, 65535, 347, 65535, 348, 65535, 65535, 65535, 349, 65535, 65535, 65535, 65535, 65535, 350, 65535, 351, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 352, 65535, 65535, 65535, 353, 65535, 354, 65535, 65535, 65535, 355, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 356, 65535, 65535, 65535, 65535, 65535, 357, 65535, 65535, 65535, 358, 65535, 359, 65535, 65535, 65535, 360, 65535, 361, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 362, 65535, 363, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 364, 65535, 65535, 65535, 65535, 65535, 365, 65535, 366, 65535, 65535, 65535, 367, 65535, 65535, 65535, 65535, 65535, 368, 65535, 369, 65535, 65535, 65535, 65535, 65535, 370, 65535, 65535, 65535, 371, 65535, 372, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 373, 65535, 65535, 65535, 65535, 65535, 374, 65535, 375, 65535, 65535, 65535, 65535, 65535, 376, 65535, 65535, 65535, 377, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 378, 65535, 65535, 65535, 65535, 65535, 379, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 380, 65535, 65535, 65535, 65535, 65535, 381, 65535, 65535, 65535, 382, 65535, 383, 65535, 65535, 65535, 384, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 385, 65535, 65535, 65535, 65535, 65535, 386, 65535, 65535, 65535, 387, 65535, 65535, 65535, 65535, 65535, 388, 65535, 389, 65535, 65535, 65535, 390, 65535, 65535, 65535, 65535, 65535, 391, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 392, 65535, 65535, 65535, 65535, 65535, 393, 65535, 65535, 65535, 65535, 65535, 394, 65535, 65535, 65535, 395, 65535, 65535, 65535, 65535, 65535, 396, 65535, 397, 65535, 65535, 65535, 65535, 65535, 398, 65535, 65535, 65535, 399, 65535, 400, 65535, 65535, 65535, 401, 65535, 402, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 403, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 404, 65535, 405, 65535, 65535, 65535, 406, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 407, 65535, 408, 65535, 65535, 65535, 65535, 65535, 409, 65535, 65535, 65535, 410, 65535, 411, 65535, 65535, 65535, 412, 65535, 65535, 65535, 65535, 65535, 413, 65535, 65535, 65535, 65535, 65535, 414, 65535, 415, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 416, 65535, 65535, 65535, 65535, 65535, 417, 65535, 65535, 65535, 418, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 419, 65535, 420, 65535, 65535, 65535, 421, 65535, 422, 65535, 65535, 65535, 423, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 424, 65535, 65535, 65535, 65535, 65535, 425, 65535, 65535, 65535, 426, 65535, 65535, 65535, 65535, 65535, 427, 65535, 428, 65535, 65535, 65535, 429, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 430, 65535, 65535, 65535, 65535, 65535, 431, 65535, 65535, 65535, 65535, 65535, 432, 65535, 65535, 65535, 433, 65535, 434, 65535, 65535, 65535, 435, 65535, 65535, 65535, 65535, 65535, 436, 65535, 437, 65535, 65535, 65535, 65535, 65535, 438, 65535, 65535, 65535, 439, 65535, 440, 65535, 65535, 65535, 441, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 442, 65535, 443, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 444, 65535, 65535, 65535, 65535, 65535, 445, 65535, 65535, 65535, 446, 65535, 65535, 65535, 65535, 65535, 447, 65535, 448, 65535, 65535, 65535, 65535, 65535, 449, 65535, 65535, 65535, 450, 65535, 65535, 65535, 65535, 65535, 451, 65535, 65535, 65535, 65535, 65535, 452, 65535, 65535, 65535, 65535, 65535, 453, 65535, 454, 65535, 65535, 65535, 65535, 65535, 455, 65535, 65535, 65535, 456, 65535, 457, 65535, 65535, 65535, 65535, 65535, 458, 65535, 65535, 65535, 459, 65535, 65535, 65535, 65535, 65535, 460, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 461, 65535, 65535, 65535, 462, 65535, 463, 65535, 65535, 65535, 464, 65535, 465, 65535, 65535, 65535, 466, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 467, 65535, 65535, 65535, 468, 65535, 65535, 65535, 65535, 65535, 469, 65535, 470, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 471, 65535, 472, 65535, 65535, 65535, 65535, 65535, 473, 65535, 65535, 65535, 65535, 65535, 474, 65535, 65535, 65535, 475, 65535, 476, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 477, 65535, 478, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 479, 65535, 480, 65535, 65535, 65535, 481, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 482, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 483, 65535, 484, 65535, 65535, 65535, 485, 65535, 486, 65535, 65535, 65535, 487, 65535, 65535, 65535, 65535, 65535, 488, 65535, 489, 65535, 65535, 65535, 65535, 65535, 490, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 491, 65535, 65535, 65535, 65535, 65535, 492, 65535, 65535, 65535, 65535, 65535, 493, 65535, 494, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 495, 65535, 496, 65535, 65535, 65535, 65535, 65535, 497, 65535, 65535, 65535, 498, 65535, 65535, 65535, 65535, 65535, 499, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 500, 65535, 65535, 65535, 501, 65535, 502, 65535, 65535, 65535, 503, 65535, 65535, 65535, 65535, 65535, 504, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 505, 65535, 65535, 65535, 65535, 65535, 506, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 507, 65535, 508, 65535, 65535, 65535, 509, 65535, 65535, 65535, 65535, 65535, 510, 65535, 511, 65535, 65535, 65535, 65535, 65535, 512, 65535, 65535, 65535, 65535, 65535, 513, 65535, 65535, 65535, 514, 65535, 515, 65535, 65535, 65535, 516, 65535, 65535, 65535, 65535, 65535, 517, 65535, 518, 65535, 65535, 65535, 65535, 65535, 519, 65535, 65535, 65535, 520, 65535, 521, 65535, 65535, 65535, 65535, 65535, 522, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 523, 65535, 524, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 525, 65535, 65535, 65535, 65535, 65535, 526, 65535, 527, 65535, 65535, 65535, 528, 65535, 65535, 65535, 65535, 65535, 529, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 530, 65535, 65535, 65535, 531, 65535, 532, 65535, 65535, 65535, 533, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 534, 65535, 535, 65535, 65535, 65535, 65535, 65535, 536, 65535, 65535, 65535, 537, 65535, 538, 65535, 65535, 65535, 65535, 65535, 539, 65535, 65535, 65535, 540, 65535, 65535, 65535, 65535, 65535, 541, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 542, 65535, 543, 65535, 65535, 65535, 544, 65535, 545, 65535, 65535, 65535, 546, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 547, 65535, 65535, 65535, 65535, 65535, 548, 65535, 65535, 65535, 549, 65535, 65535, 65535, 65535, 65535, 550, 65535, 551, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 552, 65535, 553, 65535, 65535, 65535, 65535, 65535, 554, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 555, 65535, 65535, 65535, 65535, 65535, 556, 65535, 65535, 65535, 65535, 65535, 557, 65535, 558, 65535, 65535, 65535, 65535, 65535, 559, 65535, 65535, 65535, 560, 65535, 561, 65535, 65535, 65535, 562, 65535, 563, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 564, 65535, 565, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 566, 65535, 65535, 65535, 567, 65535, 65535, 65535, 65535, 65535, 568, 65535, 65535, 65535, 65535, 65535, 569, 65535, 570, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 571, 65535, 65535, 65535, 572, 65535, 65535, 65535, 65535, 65535, 573, 65535, 65535, 65535, 65535, 65535, 574, 65535, 575, 65535, 65535, 65535, 65535, 65535, 576, 65535, 65535, 65535, 577, 65535, 578, 65535, 65535, 65535, 65535, 65535, 579, 65535, 65535, 65535, 580, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 581, 65535, 65535, 65535, 582, 65535, 583, 65535, 65535, 65535, 65535, 65535, 584, 65535, 65535, 65535, 585, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 586, 65535, 65535, 65535, 65535, 65535, 587, 65535, 65535, 65535, 588, 65535, 65535, 65535, 65535, 65535, 589, 65535, 590, 65535, 65535, 65535, 591, 65535, 65535, 65535, 65535, 65535, 592, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 593, 65535, 65535, 65535, 65535, 65535, 594, 65535, 65535, 65535, 65535, 65535, 595, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 596, 65535, 597, 65535, 65535, 65535, 65535, 65535, 598, 65535, 65535, 65535, 599, 65535, 65535, 65535, 65535, 65535, 600, 65535, 601, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 602, 65535, 603, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 604, 65535, 605, 65535, 65535, 65535, 606, 65535, 607, 65535, 65535, 65535, 608, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 609, 65535, 65535, 65535, 65535, 65535, 610, 65535, 65535, 65535, 611, 65535, 612, 65535, 65535, 65535, 613, 65535, 65535, 65535, 65535, 65535, 614, 65535, 65535, 65535, 65535, 65535, 615, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 616, 65535, 65535, 65535, 617, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 618, 65535, 65535, 65535, 619, 65535, 65535, 65535, 65535, 65535, 620, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 621, 65535, 65535, 65535, 622, 65535, 623, 65535, 65535, 65535, 624, 65535, 625, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 626, 65535, 65535, 65535, 65535, 65535, 627, 65535, 65535, 65535, 628, 65535, 65535, 65535, 65535, 65535, 629, 65535, 65535, 65535, 65535, 65535, 630, 65535, 65535, 65535, 65535, 65535, 631, 65535, 632, 65535, 65535, 65535, 65535, 65535, 633, 65535, 65535, 65535, 65535, 65535, 634, 65535, 65535, 65535, 635, 65535, 636, 65535, 65535, 65535, 637, 65535, 65535, 65535, 65535, 65535, 638, 65535, 639, 65535, 65535, 65535, 65535, 65535, 640, 65535, 65535, 65535, 65535, 65535, 641, 65535, 65535, 65535, 642, 65535, 643, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 644, 65535, 645, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 646, 65535, 647, 65535, 65535, 65535, 65535, 65535, 648, 65535, 65535, 65535, 649, 65535, 65535, 65535, 65535, 65535, 650, 65535, 651, 65535, 65535, 65535, 65535, 65535, 652, 65535, 65535, 65535, 653, 65535, 654, 65535, 65535, 65535, 655, 65535, 65535, 65535, 65535, 65535, 656, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 657, 65535, 65535, 65535, 658, 65535, 659, 65535, 65535, 65535, 65535, 65535, 660, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 661, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 662, 65535, 65535, 65535, 663, 65535, 664, 65535, 65535, 65535, 665, 65535, 666, 65535, 65535, 65535, 667, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 668, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 669, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 670, 65535, 65535, 65535, 671, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 672, 65535, 65535, 65535, 65535, 65535, 673, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 674, 65535, 675, 65535, 65535, 65535, 676, 65535, 65535, 65535, 65535, 65535, 677, 65535, 678, 65535, 65535, 65535, 65535, 65535, 679, 65535, 65535, 65535, 680, 65535, 65535, 65535, 65535, 65535, 681, 65535, 682, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 683, 65535, 684, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 685, 65535, 686, 65535, 65535, 65535, 687, 65535, 688, 65535, 65535, 65535, 689, 65535, 65535, 65535, 65535, 65535, 690, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 691, 65535, 65535, 65535, 692, 65535, 693, 65535, 65535, 65535, 694, 65535, 65535, 65535, 65535, 65535, 695, 65535, 65535, 65535, 65535, 65535, 696, 65535, 697, 65535, 65535, 65535, 65535, 65535, 698, 65535, 65535, 65535, 65535, 65535, 699, 65535, 65535, 65535, 65535, 65535, 700, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 701, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 702, 65535, 65535, 65535, 703, 65535, 704, 65535, 65535, 65535, 705, 65535, 65535, 65535, 65535, 65535, 706, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 707, 65535, 65535, 65535, 65535, 65535, 708, 65535, 65535, 65535, 709, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 710, 65535, 65535, 65535, 711, 65535, 65535, 65535, 65535, 65535, 712, 65535, 713, 65535, 65535, 65535, 65535, 65535, 714, 65535, 65535, 65535, 65535, 65535, 715, 65535, 65535, 65535, 716, 65535, 717, 65535, 65535, 65535, 718, 65535, 65535, 65535, 65535, 65535, 719, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 720, 65535, 721, 65535, 65535, 65535, 722, 65535, 723, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 724, 65535, 725, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 726, 65535, 727, 65535, 65535, 65535, 728, 65535, 729, 65535, 65535, 65535, 730, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 731, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 732, 65535, 733, 65535, 65535, 65535, 734, 65535, 65535, 65535, 65535, 65535, 735, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 736, 65535, 65535, 65535, 65535, 65535, 737, 65535, 65535, 65535, 738, 65535, 739, 65535, 65535, 65535, 65535, 65535, 740, 65535, 65535, 65535, 741, 65535, 65535, 65535, 65535, 65535, 742, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 743, 65535, 65535, 65535, 744, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 745, 65535, 65535, 65535, 746, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 747, 65535, 65535, 65535, 65535, 65535, 748, 65535, 65535, 65535, 749, 65535, 65535, 65535, 65535, 65535, 750, 65535, 751, 65535, 65535, 65535, 752, 65535, 65535, 65535, 65535, 65535, 753, 65535, 754, 65535, 65535, 65535, 65535, 65535, 755, 65535, 65535, 65535, 65535, 65535, 756, 65535, 65535, 65535, 757, 65535, 65535, 65535, 65535, 65535, 758, 65535, 65535, 65535, 65535, 65535, 759, 65535, 760, 65535, 65535, 65535, 65535, 65535, 761, 65535, 65535, 65535, 65535, 65535, 762, 65535, 65535, 65535, 763, 65535, 764, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 765, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 766, 65535, 767, 65535, 65535, 65535, 768, 65535, 769, 65535, 65535, 65535, 770, 65535, 65535, 65535, 65535, 65535, 771, 65535, 772, 65535, 65535, 65535, 65535, 65535, 773, 65535, 65535, 65535, 774, 65535, 775, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 776, 65535, 777, 65535, 65535, 65535, 65535, 65535, 778, 65535, 65535, 65535, 779, 65535, 780, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 781, 65535, 65535, 65535, 65535, 65535, 782, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 783, 65535, 65535, 65535, 784, 65535, 785, 65535, 65535, 65535, 786, 65535, 787, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 788, 65535, 65535, 65535, 65535, 65535, 789, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 790, 65535, 791, 65535, 65535, 65535, 792, 65535, 65535, 65535, 65535, 65535, 793, 65535, 794, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 795, 65535, 65535, 65535, 796, 65535, 797, 65535, 65535, 65535, 798, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 799, 65535, 65535, 65535, 65535, 65535, 800, 65535, 65535, 65535, 801, 65535, 802, 65535, 65535, 65535, 803, 65535, 804, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 805, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 806, 65535, 65535, 65535, 65535, 65535, 807, 65535, 808, 65535, 65535, 65535, 809, 65535, 65535, 65535, 65535, 65535, 810, 65535, 811, 65535, 65535, 65535, 65535, 65535, 812, 65535, 65535, 65535, 813, 65535, 814, 65535, 65535, 65535, 815, 65535, 65535, 65535, 65535, 65535, 816, 65535, 65535, 65535, 65535, 65535, 817, 65535, 818, 65535, 65535, 65535, 65535, 65535, 819, 65535, 65535, 65535, 820, 65535, 821, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 822, 65535, 65535, 65535, 65535, 65535, 823, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 824, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 825, 65535, 826, 65535, 65535, 65535, 827, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 828, 65535, 65535, 65535, 65535, 65535, 829, 65535, 65535, 65535, 830, 65535, 65535, 65535, 65535, 65535, 831, 65535, 832, 65535, 65535, 65535, 833, 65535, 65535, 65535, 65535, 65535, 834, 65535, 835, 65535, 65535, 65535, 65535, 65535, 836, 65535, 65535, 65535, 65535, 65535, 837, 65535, 65535, 65535, 65535, 65535, 838, 65535, 65535, 65535, 839, 65535, 65535, 65535, 65535, 65535, 840, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 841, 65535, 65535, 65535, 842, 65535, 843, 65535, 65535, 65535, 844, 65535, 845, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 846, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 847, 65535, 848, 65535, 65535, 65535, 849, 65535, 850, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 851, 65535, 852, 65535, 65535, 65535, 65535, 65535, 853, 65535, 65535, 65535, 854, 65535, 855, 65535, 65535, 65535, 856, 65535, 65535, 65535, 65535, 65535, 857, 65535, 65535, 65535, 65535, 65535, 858, 65535, 859, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 860, 65535, 861, 65535, 65535, 65535, 65535, 65535, 862, 65535, 65535, 65535, 863, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 864, 65535, 65535, 65535, 65535, 65535, 865, 65535, 65535, 65535, 866, 65535, 867, 65535, 65535, 65535, 868, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 869, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 870, 65535, 65535, 65535, 65535, 65535, 871, 65535, 872, 65535, 65535, 65535, 873, 65535, 65535, 65535, 65535, 65535, 874, 65535, 875, 65535, 65535, 65535, 65535, 65535, 876, 65535, 65535, 65535, 65535, 65535, 877, 65535, 65535, 65535, 878, 65535, 879, 65535, 65535, 65535, 880, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 881, 65535, 65535, 65535, 65535, 65535, 882, 65535, 65535, 65535, 883, 65535, 884, 65535, 65535, 65535, 885, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 886, 65535, 887, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 888, 65535, 65535, 65535, 889, 65535, 890, 65535, 65535, 65535, 891, 65535, 65535, 65535, 65535, 65535, 892, 65535, 893, 65535, 65535, 65535, 65535, 65535, 894, 65535, 65535, 65535, 895, 65535, 896, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 897, 65535, 65535, 65535, 65535, 65535, 898, 65535, 899, 65535, 65535, 65535, 65535, 65535, 900, 65535, 65535, 65535, 901, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 902, 65535, 65535, 65535, 903, 65535, 65535, 65535, 65535, 65535, 904, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 905, 65535, 65535, 65535, 906, 65535, 907, 65535, 65535, 65535, 908, 65535, 909, 65535, 65535, 65535, 910, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 911, 65535, 65535, 65535, 912, 65535, 65535, 65535, 65535, 65535, 913, 65535, 914, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 915, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 916, 65535, 65535, 65535, 65535, 65535, 917, 65535, 65535, 65535, 918, 65535, 919, 65535, 65535, 65535, 920, 65535, 65535, 65535, 65535, 65535, 921, 65535, 922, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 923, 65535, 924, 65535, 65535, 65535, 925, 65535, 926, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 927, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 928, 65535, 929, 65535, 65535, 65535, 930, 65535, 65535, 65535, 65535, 65535, 931, 65535, 65535, 65535, 65535, 65535, 932, 65535, 933, 65535, 65535, 65535, 65535, 65535, 934, 65535, 65535, 65535, 935, 65535, 65535, 65535, 65535, 65535, 936, 65535, 65535, 65535, 65535, 65535, 937, 65535, 65535, 65535, 65535, 65535, 938, 65535, 939, 65535, 65535, 65535, 65535, 65535, 940, 65535, 65535, 65535, 65535, 65535, 941, 65535, 65535, 65535, 65535, 65535, 942, 65535, 65535, 65535, 943, 65535, 65535, 65535, 65535, 65535, 944, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 945, 65535, 946, 65535, 65535, 65535, 947, 65535, 65535, 65535, 65535, 65535, 948, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 949, 65535, 65535, 65535, 65535, 65535, 950, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 951, 65535, 952, 65535, 65535, 65535, 953, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 954, 65535, 65535, 65535, 65535, 65535, 955, 65535, 65535, 65535, 65535, 65535, 956, 65535, 65535, 65535, 957, 65535, 958, 65535, 65535, 65535, 959, 65535, 65535, 65535, 65535, 65535, 960, 65535, 961, 65535, 65535, 65535, 65535, 65535, 962, 65535, 65535, 65535, 963, 65535, 964, 65535, 65535, 65535, 65535, 65535, 965, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 966, 65535, 967, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 968, 65535, 969, 65535, 65535, 65535, 65535, 65535, 970, 65535, 65535, 65535, 971, 65535, 65535, 65535, 65535, 65535, 972, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 973, 65535, 65535, 65535, 974, 65535, 65535, 65535, 65535, 65535, 975, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 976, 65535, 977, 65535, 65535, 65535, 65535, 65535, 978, 65535, 65535, 65535, 979, 65535, 980, 65535, 65535, 65535, 65535, 65535, 981, 65535, 65535, 65535, 982, 65535, 65535, 65535, 65535, 65535, 983, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 984, 65535, 985, 65535, 65535, 65535, 986, 65535, 987, 65535, 65535, 65535, 988, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 989, 65535, 65535, 65535, 990, 65535, 65535, 65535, 65535, 65535, 991, 65535, 992, 65535, 65535, 65535, 993, 65535, 65535, 65535, 65535, 65535, 994, 65535, 995, 65535, 65535, 65535, 65535, 65535, 996, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 997, 65535, 998, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 999, 65535, 1000, 65535, 65535, 65535, 65535, 65535, 1001, 65535, 65535, 65535, 1002, 65535, 1003, 65535, 65535, 65535, 1004, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1005, 65535, 1006, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1007, 65535, 1008, 65535, 65535, 65535, 1009, 65535, 65535, 65535, 65535, 65535, 1010, 65535, 65535, 65535, 65535, 65535, 1011, 65535, 1012, 65535, 65535, 65535, 65535, 65535, 1013, 65535, 65535, 65535, 65535, 65535, 1014, 65535, 65535, 65535, 1015, 65535, 65535, 65535, 65535, 65535, 1016, 65535, 65535, 65535, 65535, 65535, 1017, 65535, 1018, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1019, 65535, 1020, 65535, 65535, 65535, 65535, 65535, 1021, 65535, 65535, 65535, 1022, 65535, 65535, 65535, 65535, 65535, 1023, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1024, 65535, 65535, 65535, 1025, 65535, 1026, 65535, 65535, 65535, 65535, 65535, 1027, 65535, 65535, 65535, 1028, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1029, 65535, 65535, 65535, 65535, 65535, 1030, 65535, 65535, 65535, 1031, 65535, 65535, 65535, 65535, 65535, 1032, 65535, 1033, 65535, 65535, 65535, 1034, 65535, 65535, 65535, 65535, 65535, 1035, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1036, 65535, 65535, 65535, 65535, 65535, 1037, 65535, 65535, 65535, 1038, 65535, 1039, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1040, 65535, 1041, 65535, 65535, 65535, 65535, 65535, 1042, 65535, 65535, 65535, 1043, 65535, 1044, 65535, 65535, 65535, 65535, 65535, 1045, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1046, 65535, 1047, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1048, 65535, 65535, 65535, 65535, 65535, 1049, 65535, 1050, 65535, 65535, 65535, 1051, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1052, 65535, 65535, 65535, 65535, 65535, 1053, 65535, 65535, 65535, 1054, 65535, 1055, 65535, 65535, 65535, 1056, 65535, 65535, 65535, 65535, 65535, 1057, 65535, 65535, 65535, 65535, 65535, 1058, 65535, 1059, 65535, 65535, 65535, 65535, 65535, 1060, 65535, 65535, 65535, 1061, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1062, 65535, 65535, 65535, 1063, 65535, 65535, 65535, 65535, 65535, 1064, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1065, 65535, 65535, 65535, 1066, 65535, 1067, 65535, 65535, 65535, 1068, 65535, 1069, 65535, 65535, 65535, 1070, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1071, 65535, 65535, 65535, 65535, 65535, 1072, 65535, 65535, 65535, 1073, 65535, 65535, 65535, 65535, 65535, 1074, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1075, 65535, 1076, 65535, 65535, 65535, 65535, 65535, 1077, 65535, 65535, 65535, 65535, 65535, 1078, 65535, 65535, 65535, 1079, 65535, 65535, 65535, 65535, 65535, 1080, 65535, 65535, 65535, 65535, 65535, 1081, 65535, 1082, 65535, 65535, 65535, 65535, 65535, 1083, 65535, 65535, 65535, 1084, 65535, 1085, 65535, 65535, 65535, 1086, 65535, 1087, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1088, 65535, 1089, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1090, 65535, 65535, 65535, 65535, 65535, 1091, 65535, 65535, 65535, 1092, 65535, 65535, 65535, 65535, 65535, 1093, 65535, 1094, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1095, 65535, 1096, 65535, 65535, 65535, 1097, 65535, 65535, 65535, 65535, 65535, 1098, 65535, 65535, 65535, 65535, 65535, 1099, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1100, 65535, 65535, 65535, 1101, 65535, 1102, 65535, 65535, 65535, 65535, 65535, 1103, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1104, 65535, 65535, 65535, 1105, 65535, 1106, 65535, 65535, 65535, 1107, 65535, 1108, 65535, 65535, 65535, 1109, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1110, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1111, 65535, 65535, 65535, 65535, 65535, 1112, 65535, 1113, 65535, 65535, 65535, 1114, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1115, 65535, 65535, 65535, 65535, 65535, 1116, 65535, 65535, 65535, 65535, 65535, 1117, 65535, 65535, 65535, 65535, 65535, 1118, 65535, 65535, 65535, 1119, 65535, 65535, 65535, 65535, 65535, 1120, 65535, 1121, 65535, 65535, 65535, 65535, 65535, 1122, 65535, 65535, 65535, 1123, 65535, 65535, 65535, 65535, 65535, 1124, 65535, 1125, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1126, 65535, 1127, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1128, 65535, 1129, 65535, 65535, 65535, 1130, 65535, 1131, 65535, 65535, 65535, 1132, 65535, 65535, 65535, 65535, 65535, 1133, 65535, 1134, 65535, 65535, 65535, 65535, 65535, 1135, 65535, 65535, 65535, 1136, 65535, 1137, 65535, 65535, 65535, 1138, 65535, 65535, 65535, 65535, 65535, 1139, 65535, 65535, 65535, 65535, 65535, 1140, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1141, 65535, 65535, 65535, 65535, 65535, 1142, 65535, 65535, 65535, 65535, 65535, 1143, 65535, 65535, 65535, 1144, 65535, 65535, 65535, 65535, 65535, 1145, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1146, 65535, 65535, 65535, 1147, 65535, 1148, 65535, 65535, 65535, 1149, 65535, 1150, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1151, 65535, 65535, 65535, 65535, 65535, 1152, 65535, 65535, 65535, 1153, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1154, 65535, 65535, 65535, 65535, 65535, 1155, 65535, 1156, 65535, 65535, 65535, 65535, 65535, 1157, 65535, 65535, 65535, 65535, 65535, 1158, 65535, 65535, 65535, 1159, 65535, 1160, 65535, 65535, 65535, 1161, 65535, 65535, 65535, 65535, 65535, 1162, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1163, 65535, 65535, 65535, 65535, 65535, 1164, 65535, 65535, 65535, 1165, 65535, 1166, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1167, 65535, 1168, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1169, 65535, 1170, 65535, 65535, 65535, 1171, 65535, 1172, 65535, 65535, 65535, 1173, 65535, 65535, 65535, 65535, 65535, 1174, 65535, 1175, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1176, 65535, 1177, 65535, 65535, 65535, 1178, 65535, 65535, 65535, 65535, 65535, 1179, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1180, 65535, 65535, 65535, 65535, 65535, 1181, 65535, 65535, 65535, 1182, 65535, 1183, 65535, 65535, 65535, 65535, 65535, 1184, 65535, 65535, 65535, 1185, 65535, 65535, 65535, 65535, 65535, 1186, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1187, 65535, 65535, 65535, 1188, 65535, 65535, 65535, 65535, 65535, 1189, 65535, 1190, 65535, 65535, 65535, 1191, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1192, 65535, 65535, 65535, 65535, 65535, 1193, 65535, 65535, 65535, 1194, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1195, 65535, 65535, 65535, 1196, 65535, 65535, 65535, 65535, 65535, 1197, 65535, 1198, 65535, 65535, 65535, 65535, 65535, 1199, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1200, 65535, 65535, 65535, 65535, 65535, 1201, 65535, 65535, 65535, 65535, 65535, 1202, 65535, 1203, 65535, 65535, 65535, 65535, 65535, 1204, 65535, 65535, 65535, 65535, 65535, 1205, 65535, 65535, 65535, 1206, 65535, 1207, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1208, 65535, 1209, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1210, 65535, 1211, 65535, 65535, 65535, 1212, 65535, 1213, 65535, 65535, 65535, 1214, 65535, 65535, 65535, 65535, 65535, 1215, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1216, 65535, 65535, 65535, 1217, 65535, 1218, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1219, 65535, 65535, 65535, 65535, 65535, 1220, 65535, 1221, 65535, 65535, 65535, 65535, 65535, 1222, 65535, 65535, 65535, 1223, 65535, 1224, 65535, 65535, 65535, 65535, 65535, 1225, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1226, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1227, 65535, 65535, 65535, 1228, 65535, 1229, 65535, 65535, 65535, 1230, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1231, 65535, 65535, 65535, 65535, 65535, 1232, 65535, 65535, 65535, 1233, 65535, 65535, 65535, 65535, 65535, 1234, 65535, 1235, 65535, 65535, 65535, 1236, 65535, 65535, 65535, 65535, 65535, 1237, 65535, 1238, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1239, 65535, 65535, 65535, 1240, 65535, 1241, 65535, 65535, 65535, 1242, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1243, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1244, 65535, 1245, 65535, 65535, 65535, 1246, 65535, 1247, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1248, 65535, 1249, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1250, 65535, 65535, 65535, 65535, 65535, 1251, 65535, 1252, 65535, 65535, 65535, 1253, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1254, 65535, 65535, 65535, 65535, 65535, 1255, 65535, 65535, 65535, 1256, 65535, 1257, 65535, 65535, 65535, 1258, 65535, 65535, 65535, 65535, 65535, 1259, 65535, 65535, 65535, 65535, 65535, 1260, 65535, 1261, 65535, 65535, 65535, 65535, 65535, 1262, 65535, 65535, 65535, 1263, 65535, 1264, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1265, 65535, 65535, 65535, 65535, 65535, 1266, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1267, 65535, 65535, 65535, 65535, 65535, 1268, 65535, 65535, 65535, 65535, 65535, 1269, 65535, 65535, 65535, 1270, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1271, 65535, 65535, 65535, 65535, 65535, 1272, 65535, 65535, 65535, 1273, 65535, 65535, 65535, 65535, 65535, 1274, 65535, 1275, 65535, 65535, 65535, 1276, 65535, 65535, 65535, 65535, 65535, 1277, 65535, 1278, 65535, 65535, 65535, 65535, 65535, 1279, 65535, 65535, 65535, 65535, 65535, 1280, 65535, 65535, 65535, 65535, 65535, 1281, 65535, 65535, 65535, 1282, 65535, 65535, 65535, 65535, 65535, 1283, 65535, 1284, 65535, 65535, 65535, 65535, 65535, 1285, 65535, 65535, 65535, 1286, 65535, 1287, 65535, 65535, 65535, 1288, 65535, 1289, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1290, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1291, 65535, 1292, 65535, 65535, 65535, 1293, 65535, 1294, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1295, 65535, 1296, 65535, 65535, 65535, 65535, 65535, 1297, 65535, 65535, 65535, 1298, 65535, 1299, 65535, 65535, 65535, 1300, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1301, 65535, 1302, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1303, 65535, 1304, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1305, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1306, 65535, 65535, 65535, 1307, 65535, 1308, 65535, 65535, 65535, 1309, 65535, 1310, 65535, 65535, 65535, 1311, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1312, 65535, 65535, 65535, 65535, 65535, 1313, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1314, 65535, 1315, 65535, 65535, 65535, 1316, 65535, 65535, 65535, 65535, 65535, 1317, 65535, 1318, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1319, 65535, 65535, 65535, 1320, 65535, 1321, 65535, 65535, 65535, 1322, 65535, 65535, 65535, 65535, 65535, 1323, 65535, 1324, 65535, 65535, 65535, 65535, 65535, 1325, 65535, 65535, 65535, 1326, 65535, 1327, 65535, 65535, 65535, 1328, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1329, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1330, 65535, 65535, 65535, 1331, 65535, 1332, 65535, 65535, 65535, 1333, 65535, 65535, 65535, 65535, 65535, 1334, 65535, 1335, 65535, 65535, 65535, 65535, 65535, 1336, 65535, 65535, 65535, 1337, 65535, 1338, 65535, 65535, 65535, 1339, 65535, 65535, 65535, 65535, 65535, 1340, 65535, 65535, 65535, 65535, 65535, 1341, 65535, 1342, 65535, 65535, 65535, 65535, 65535, 1343, 65535, 65535, 65535, 1344, 65535, 1345, 65535, 65535, 65535, 65535, 65535, 1346, 65535, 65535, 65535, 1347, 65535, 65535, 65535, 65535, 65535, 1348, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1349, 65535, 65535, 65535, 1350, 65535, 65535, 65535, 65535, 65535, 1351, 65535, 1352, 65535, 65535, 65535, 1353, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1354, 65535, 65535, 65535, 1355, 65535, 65535, 65535, 65535, 65535, 1356, 65535, 1357, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1358, 65535, 1359, 65535, 65535, 65535, 65535, 65535, 1360, 65535, 65535, 65535, 65535, 65535, 1361, 65535, 65535, 65535, 1362, 65535, 1363, 65535, 65535, 65535, 1364, 65535, 65535, 65535, 65535, 65535, 1365, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1366, 65535, 1367, 65535, 65535, 65535, 1368, 65535, 1369, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1370, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1371, 65535, 1372, 65535, 65535, 65535, 1373, 65535, 1374, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1375, 65535, 1376, 65535, 65535, 65535, 65535, 65535, 1377, 65535, 65535, 65535, 1378, 65535, 65535, 65535, 65535, 65535, 1379, 65535, 65535, 65535, 65535, 65535, 1380, 65535, 65535, 65535, 65535, 65535, 1381, 65535, 1382, 65535, 65535, 65535, 65535, 65535, 1383, 65535, 65535, 65535, 1384, 65535, 1385, 65535, 65535, 65535, 65535, 65535, 1386, 65535, 65535, 65535, 1387, 65535, 65535, 65535, 65535, 65535, 1388, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1389, 65535, 65535, 65535, 65535, 65535, 1390, 65535, 65535, 65535, 1391, 65535, 65535, 65535, 65535, 65535, 1392, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1393, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1394, 65535, 1395, 65535, 65535, 65535, 1396, 65535, 65535, 65535, 65535, 65535, 1397, 65535, 1398, 65535, 65535, 65535, 65535, 65535, 1399, 65535, 65535, 65535, 65535, 65535, 1400, 65535, 65535, 65535, 1401, 65535, 1402, 65535, 65535, 65535, 1403, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1404, 65535, 65535, 65535, 65535, 65535, 1405, 65535, 65535, 65535, 1406, 65535, 1407, 65535, 65535, 65535, 65535, 65535, 1408, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1409, 65535, 1410, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1411, 65535, 1412, 65535, 65535, 65535, 1413, 65535, 1414, 65535, 65535, 65535, 1415, 65535, 65535, 65535, 65535, 65535, 1416, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1417, 65535, 65535, 65535, 1418, 65535, 1419, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1420, 65535, 1421, 65535, 65535, 65535, 65535, 65535, 1422, 65535, 65535, 65535, 1423, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1424, 65535, 65535, 65535, 1425, 65535, 65535, 65535, 65535, 65535, 1426, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1427, 65535, 1428, 65535, 65535, 65535, 1429, 65535, 1430, 65535, 65535, 65535, 1431, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1432, 65535, 65535, 65535, 65535, 65535, 1433, 65535, 65535, 65535, 1434, 65535, 65535, 65535, 65535, 65535, 1435, 65535, 1436, 65535, 65535, 65535, 1437, 65535, 65535, 65535, 65535, 65535, 1438, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1439, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1440, 65535, 1441, 65535, 65535, 65535, 1442, 65535, 65535, 65535, 65535, 65535, 1443, 65535, 1444, 65535, 65535, 65535, 65535, 65535, 1445, 65535, 65535, 65535, 1446, 65535, 1447, 65535, 65535, 65535, 1448, 65535, 1449, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1450, 65535, 1451, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1452, 65535, 1453, 65535, 65535, 65535, 1454, 65535, 65535, 65535, 65535, 65535, 1455, 65535, 65535, 65535, 65535, 65535, 1456, 65535, 1457, 65535, 65535, 65535, 65535, 65535, 1458, 65535, 65535, 65535, 65535, 65535, 1459, 65535, 65535, 65535, 1460, 65535, 65535, 65535, 65535, 65535, 1461, 65535, 65535, 65535, 65535, 65535, 1462, 65535, 1463, 65535, 65535, 65535, 65535, 65535, 1464, 65535, 65535, 65535, 65535, 65535, 1465, 65535, 65535, 65535, 65535, 65535, 1466, 65535, 65535, 65535, 1467, 65535, 65535, 65535, 65535, 65535, 1468, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1469, 65535, 1470, 65535, 65535, 65535, 65535, 65535, 1471, 65535, 65535, 65535, 1472, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1473, 65535, 65535, 65535, 65535, 65535, 1474, 65535, 65535, 65535, 1475, 65535, 65535, 65535, 65535, 65535, 1476, 65535, 1477, 65535, 65535, 65535, 1478, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1479, 65535, 65535, 65535, 65535, 65535, 1480, 65535, 65535, 65535, 1481, 65535, 1482, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1483, 65535, 1484, 65535, 65535, 65535, 65535, 65535, 1485, 65535, 65535, 65535, 1486, 65535, 1487, 65535, 65535, 65535, 1488, 65535, 1489, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1490, 65535, 1491, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1492, 65535, 1493, 65535, 65535, 65535, 65535, 65535, 1494, 65535, 65535, 65535, 1495, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1496, 65535, 65535, 65535, 65535, 65535, 1497, 65535, 65535, 65535, 1498, 65535, 65535, 65535, 65535, 65535, 1499, 65535, 65535, 65535, 65535, 65535, 1500, 65535, 65535, 65535, 65535, 65535, 1501, 65535, 1502, 65535, 65535, 65535, 65535, 65535, 1503, 65535, 65535, 65535, 1504, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1505, 65535, 65535, 65535, 1506, 65535, 65535, 65535, 65535, 65535, 1507, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1508, 65535, 65535, 65535, 1509, 65535, 1510, 65535, 65535, 65535, 1511, 65535, 1512, 65535, 65535, 65535, 1513, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1514, 65535, 65535, 65535, 1515, 65535, 65535, 65535, 65535, 65535, 1516, 65535, 65535, 65535, 65535, 65535, 1517, 65535, 65535, 65535, 65535, 65535, 1518, 65535, 1519, 65535, 65535, 65535, 65535, 65535, 1520, 65535, 65535, 65535, 65535, 65535, 1521, 65535, 65535, 65535, 1522, 65535, 1523, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1524, 65535, 1525, 65535, 65535, 65535, 65535, 65535, 1526, 65535, 65535, 65535, 1527, 65535, 1528, 65535, 65535, 65535, 1529, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1530, 65535, 1531, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1532, 65535, 1533, 65535, 65535, 65535, 65535, 65535, 1534, 65535, 65535, 65535, 1535, 65535, 65535, 65535, 65535, 65535, 1536, 65535, 1537, 65535, 65535, 65535, 65535, 65535, 1538, 65535, 65535, 65535, 65535, 65535, 1539, 65535, 65535, 65535, 1540, 65535, 65535, 65535, 65535, 65535, 1541, 65535, 65535, 65535, 65535, 65535, 1542, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1543, 65535, 1544, 65535, 65535, 65535, 65535, 65535, 1545, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1546, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1547, 65535, 65535, 65535, 1548, 65535, 1549, 65535, 65535, 65535, 1550, 65535, 1551, 65535, 65535, 65535, 1552, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1553, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1554, 65535, 65535, 65535, 65535, 65535, 1555, 65535, 1556, 65535, 65535, 65535, 1557, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1558, 65535, 65535, 65535, 65535, 65535, 1559, 65535, 65535, 65535, 65535, 65535, 1560, 65535, 65535, 65535, 1561, 65535, 1562, 65535, 65535, 65535, 1563, 65535, 65535, 65535, 65535, 65535, 1564, 65535, 1565, 65535, 65535, 65535, 65535, 65535, 1566, 65535, 65535, 65535, 1567, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1568, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1569, 65535, 1570, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1571, 65535, 65535, 65535, 65535, 65535, 1572, 65535, 1573, 65535, 65535, 65535, 1574, 65535, 65535, 65535, 65535, 65535, 1575, 65535, 1576, 65535, 65535, 65535, 65535, 65535, 1577, 65535, 65535, 65535, 1578, 65535, 1579, 65535, 65535, 65535, 1580, 65535, 65535, 65535, 65535, 65535, 1581, 65535, 65535, 65535, 65535, 65535, 1582, 65535, 1583, 65535, 65535, 65535, 65535, 65535, 1584, 65535, 65535, 65535, 65535, 65535, 1585, 65535, 65535, 65535, 65535, 65535, 1586, 65535, 65535, 65535, 1587, 65535, 65535, 65535, 65535, 65535, 1588, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1589, 65535, 65535, 65535, 1590, 65535, 1591, 65535, 65535, 65535, 1592, 65535, 1593, 65535, 65535, 65535, 1594, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1595, 65535, 65535, 65535, 65535, 65535, 1596, 65535, 65535, 65535, 1597, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1598, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1599, 65535, 1600, 65535, 65535, 65535, 65535, 65535, 1601, 65535, 65535, 65535, 65535, 65535, 1602, 65535, 65535, 65535, 1603, 65535, 65535, 65535, 65535, 65535, 1604, 65535, 65535, 65535, 65535, 65535, 1605, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1606, 65535, 65535, 65535, 1607, 65535, 1608, 65535, 65535, 65535, 1609, 65535, 1610, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1611, 65535, 1612, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1613, 65535, 65535, 65535, 1614, 65535, 1615, 65535, 65535, 65535, 1616, 65535, 65535, 65535, 65535, 65535, 1617, 65535, 1618, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1619, 65535, 1620, 65535, 65535, 65535, 1621, 65535, 65535, 65535, 65535, 65535, 1622, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1623, 65535, 65535, 65535, 65535, 65535, 1624, 65535, 65535, 65535, 1625, 65535, 1626, 65535, 65535, 65535, 65535, 65535, 1627, 65535, 65535, 65535, 1628, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1629, 65535, 65535, 65535, 1630, 65535, 65535, 65535, 65535, 65535, 1631, 65535, 1632, 65535, 65535, 65535, 1633, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1634, 65535, 65535, 65535, 65535, 65535, 1635, 65535, 65535, 65535, 1636, 65535, 65535, 65535, 65535, 65535, 1637, 65535, 1638, 65535, 65535, 65535, 1639, 65535, 65535, 65535, 65535, 65535, 1640, 65535, 1641, 65535, 65535, 65535, 65535, 65535, 1642, 65535, 65535, 65535, 65535, 65535, 1643, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1644, 65535, 65535, 65535, 65535, 65535, 1645, 65535, 1646, 65535, 65535, 65535, 65535, 65535, 1647, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1648, 65535, 1649, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1650, 65535, 1651, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1652, 65535, 1653, 65535, 65535, 65535, 1654, 65535, 1655, 65535, 65535, 65535, 1656, 65535, 65535, 65535, 65535, 65535, 1657, 65535, 1658, 65535, 65535, 65535, 65535, 65535, 1659, 65535, 65535, 65535, 1660, 65535, 1661, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1662, 65535, 65535, 65535, 65535, 65535, 1663, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1664, 65535, 65535, 65535, 1665, 65535, 1666, 65535, 65535, 65535, 65535, 65535, 1667, 65535, 65535, 65535, 1668, 65535, 65535, 65535, 65535, 65535, 1669, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1670, 65535, 65535, 65535, 1671, 65535, 1672, 65535, 65535, 65535, 1673, 65535, 1674, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1675, 65535, 65535, 65535, 65535, 65535, 1676, 65535, 65535, 65535, 1677, 65535, 65535, 65535, 65535, 65535, 1678, 65535, 65535, 65535, 65535, 65535, 1679, 65535, 65535, 65535, 65535, 65535, 1680, 65535, 1681, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1682, 65535, 65535, 65535, 1683, 65535, 1684, 65535, 65535, 65535, 1685, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1686, 65535, 65535, 65535, 65535, 65535, 1687, 65535, 65535, 65535, 65535, 65535, 1688, 65535, 65535, 65535, 1689, 65535, 1690, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1691, 65535, 1692, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1693, 65535, 65535, 65535, 65535, 65535, 1694, 65535, 1695, 65535, 65535, 65535, 1696, 65535, 65535, 65535, 65535, 65535, 1697, 65535, 1698, 65535, 65535, 65535, 65535, 65535, 1699, 65535, 65535, 65535, 1700, 65535, 1701, 65535, 65535, 65535, 1702, 65535, 65535, 65535, 65535, 65535, 1703, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1704, 65535, 65535, 65535, 65535, 65535, 1705, 65535, 65535, 65535, 1706, 65535, 1707, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1708, 65535, 65535, 65535, 65535, 65535, 1709, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1710, 65535, 65535, 65535, 65535, 65535, 1711, 65535, 65535, 65535, 1712, 65535, 1713, 65535, 65535, 65535, 1714, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1715, 65535, 65535, 65535, 65535, 65535, 1716, 65535, 65535, 65535, 1717, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1718, 65535, 65535, 65535, 1719, 65535, 65535, 65535, 65535, 65535, 1720, 65535, 1721, 65535, 65535, 65535, 65535, 65535, 1722, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1723, 65535, 65535, 65535, 1724, 65535, 65535, 65535, 65535, 65535, 1725, 65535, 1726, 65535, 65535, 65535, 65535, 65535, 1727, 65535, 65535, 65535, 1728, 65535, 1729, 65535, 65535, 65535, 1730, 65535, 1731, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1732, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1733, 65535, 1734, 65535, 65535, 65535, 1735, 65535, 1736, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1737, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1738, 65535, 65535, 65535, 1739, 65535, 1740, 65535, 65535, 65535, 1741, 65535, 65535, 65535, 65535, 65535, 1742, 65535, 65535, 65535, 65535, 65535, 1743, 65535, 1744, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1745, 65535, 1746, 65535, 65535, 65535, 65535, 65535, 1747, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1748, 65535, 65535, 65535, 1749, 65535, 1750, 65535, 65535, 65535, 1751, 65535, 65535, 65535, 65535, 65535, 1752, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1753, 65535, 65535, 65535, 65535, 65535, 1754, 65535, 65535, 65535, 1755, 65535, 65535, 65535, 65535, 65535, 1756, 65535, 1757, 65535, 65535, 65535, 1758, 65535, 65535, 65535, 65535, 65535, 1759, 65535, 1760, 65535, 65535, 65535, 65535, 65535, 1761, 65535, 65535, 65535, 65535, 65535, 1762, 65535, 65535, 65535, 1763, 65535, 1764, 65535, 65535, 65535, 1765, 65535, 65535, 65535, 65535, 65535, 1766, 65535, 1767, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1768, 65535, 1769, 65535, 65535, 65535, 1770, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1771, 65535, 1772, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1773, 65535, 65535, 65535, 1774, 65535, 1775, 65535, 65535, 65535, 1776, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1777, 65535, 65535, 65535, 65535, 65535, 1778, 65535, 65535, 65535, 1779, 65535, 1780, 65535, 65535, 65535, 1781, 65535, 65535, 65535, 65535, 65535, 1782, 65535, 65535, 65535, 65535, 65535, 1783, 65535, 1784, 65535, 65535, 65535, 65535, 65535, 1785, 65535, 65535, 65535, 1786, 65535, 1787, 65535, 65535, 65535, 65535, 65535, 1788, 65535, 65535, 65535, 1789, 65535, 65535, 65535, 65535, 65535, 1790, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1791, 65535, 65535, 65535, 1792, 65535, 1793, 65535, 65535, 65535, 65535, 65535, 1794, 65535, 65535, 65535, 1795, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1796, 65535, 65535, 65535, 1797, 65535, 65535, 65535, 65535, 65535, 1798, 65535, 1799, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1800, 65535, 1801, 65535, 65535, 65535, 65535, 65535, 1802, 65535, 65535, 65535, 65535, 65535, 1803, 65535, 65535, 65535, 1804, 65535, 1805, 65535, 65535, 65535, 1806, 65535, 65535, 65535, 65535, 65535, 1807, 65535, 1808, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1809, 65535, 1810, 65535, 65535, 65535, 1811, 65535, 1812, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1813, 65535, 1814, 65535, 65535, 65535, 1815, 65535, 1816, 65535, 65535, 65535, 1817, 65535, 65535, 65535, 65535, 65535, 1818, 65535, 1819, 65535, 65535, 65535, 65535, 65535, 1820, 65535, 65535, 65535, 1821, 65535, 65535, 65535, 65535, 65535, 1822, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1823, 65535, 1824, 65535, 65535, 65535, 65535, 65535, 1825, 65535, 65535, 65535, 1826, 65535, 1827, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1828, 65535, 65535, 65535, 65535, 65535, 1829, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1830, 65535, 65535, 65535, 1831, 65535, 1832, 65535, 65535, 65535, 1833, 65535, 65535, 65535, 65535, 65535, 1834, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1835, 65535, 65535, 65535, 65535, 65535, 1836, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1837, 65535, 1838, 65535, 65535, 65535, 1839, 65535, 65535, 65535, 65535, 65535, 1840, 65535, 1841, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1842, 65535, 65535, 65535, 1843, 65535, 1844, 65535, 65535, 65535, 1845, 65535, 65535, 65535, 65535, 65535, 1846, 65535, 1847, 65535, 65535, 65535, 65535, 65535, 1848, 65535, 65535, 65535, 1849, 65535, 1850, 65535, 65535, 65535, 65535, 65535, 1851, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1852, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1853, 65535, 1854, 65535, 65535, 65535, 1855, 65535, 1856, 65535, 65535, 65535, 1857, 65535, 65535, 65535, 65535, 65535, 1858, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1859, 65535, 65535, 65535, 1860, 65535, 1861, 65535, 65535, 65535, 1862, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1863, 65535, 1864, 65535, 65535, 65535, 65535, 65535, 1865, 65535, 65535, 65535, 1866, 65535, 1867, 65535, 65535, 65535, 65535, 65535, 1868, 65535, 65535, 65535, 1869, 65535, 65535, 65535, 65535, 65535, 1870, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1871, 65535, 65535, 65535, 65535, 65535, 1872, 65535, 1873, 65535, 65535, 65535, 1874, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1875, 65535, 65535, 65535, 65535, 65535, 1876, 65535, 65535, 65535, 1877, 65535, 65535, 65535, 65535, 65535, 1878, 65535, 1879, 65535, 65535, 65535, 1880, 65535, 65535, 65535, 65535, 65535, 1881, 65535, 1882, 65535, 65535, 65535, 65535, 65535, 1883, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1884, 65535, 1885, 65535, 65535, 65535, 1886, 65535, 65535, 65535, 65535, 65535, 1887, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1888, 65535, 65535, 65535, 1889, 65535, 1890, 65535, 65535, 65535, 1891, 65535, 1892, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1893, 65535, 1894, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1895, 65535, 1896, 65535, 65535, 65535, 1897, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1898, 65535, 1899, 65535, 65535, 65535, 65535, 65535, 1900, 65535, 65535, 65535, 65535, 65535, 1901, 65535, 65535, 65535, 1902, 65535, 65535, 65535, 65535, 65535, 1903, 65535, 65535, 65535, 65535, 65535, 1904, 65535, 1905, 65535, 65535, 65535, 65535, 65535, 1906, 65535, 65535, 65535, 1907, 65535, 1908, 65535, 65535, 65535, 65535, 65535, 1909, 65535, 65535, 65535, 1910, 65535, 65535, 65535, 65535, 65535, 1911, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1912, 65535, 65535, 65535, 65535, 65535, 1913, 65535, 65535, 65535, 65535, 65535, 1914, 65535, 65535, 65535, 1915, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1916, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1917, 65535, 65535, 65535, 65535, 65535, 1918, 65535, 1919, 65535, 65535, 65535, 1920, 65535, 65535, 65535, 65535, 65535, 1921, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1922, 65535, 65535, 65535, 65535, 65535, 1923, 65535, 65535, 65535, 1924, 65535, 1925, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1926, 65535, 65535, 65535, 65535, 65535, 1927, 65535, 65535, 65535, 1928, 65535, 1929, 65535, 65535, 65535, 1930, 65535, 1931, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1932, 65535, 1933, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1934, 65535, 1935, 65535, 65535, 65535, 1936, 65535, 1937, 65535, 65535, 65535, 1938, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1939, 65535, 65535, 65535, 65535, 65535, 1940, 65535, 65535, 65535, 1941, 65535, 1942, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1943, 65535, 65535, 65535, 65535, 65535, 1944, 65535, 1945, 65535, 65535, 65535, 65535, 65535, 1946, 65535, 65535, 65535, 1947, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1948, 65535, 65535, 65535, 1949, 65535, 65535, 65535, 65535, 65535, 1950, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1951, 65535, 65535, 65535, 1952, 65535, 1953, 65535, 65535, 65535, 1954, 65535, 1955, 65535, 65535, 65535, 1956, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1957, 65535, 65535, 65535, 65535, 65535, 1958, 65535, 65535, 65535, 1959, 65535, 65535, 65535, 65535, 65535, 1960, 65535, 65535, 65535, 65535, 65535, 1961, 65535, 65535, 65535, 65535, 65535, 1962, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1963, 65535, 65535, 65535, 65535, 65535, 1964, 65535, 65535, 65535, 1965, 65535, 1966, 65535, 65535, 65535, 1967, 65535, 65535, 65535, 65535, 65535, 1968, 65535, 1969, 65535, 65535, 65535, 65535, 65535, 1970, 65535, 65535, 65535, 1971, 65535, 1972, 65535, 65535, 65535, 1973, 65535, 1974, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1975, 65535, 1976, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1977, 65535, 1978, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1979, 65535, 65535, 65535, 65535, 65535, 1980, 65535, 1981, 65535, 65535, 65535, 65535, 65535, 1982, 65535, 65535, 65535, 1983, 65535, 1984, 65535, 65535, 65535, 1985, 65535, 65535, 65535, 65535, 65535, 1986, 65535, 65535, 65535, 65535, 65535, 1987, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1988, 65535, 65535, 65535, 65535, 65535, 1989, 65535, 65535, 65535, 65535, 65535, 1990, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1991, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1992, 65535, 1993, 65535, 65535, 65535, 1994, 65535, 1995, 65535, 65535, 65535, 1996, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1997, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 1998, 65535, 65535, 65535, 65535, 65535, 1999, 65535, 2000, 65535, 65535, 65535, 2001, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2002, 65535, 65535, 65535, 65535, 65535, 2003, 65535, 65535, 65535, 65535, 65535, 2004, 65535, 65535, 65535, 2005, 65535, 2006, 65535, 65535, 65535, 2007, 65535, 65535, 65535, 65535, 65535, 2008, 65535, 2009, 65535, 65535, 65535, 65535, 65535, 2010, 65535, 65535, 65535, 2011, 65535, 65535, 65535, 65535, 65535, 2012, 65535, 2013, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2014, 65535, 2015, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2016, 65535, 2017, 65535, 65535, 65535, 65535, 65535, 2018, 65535, 65535, 65535, 2019, 65535, 65535, 65535, 65535, 65535, 2020, 65535, 2021, 65535, 65535, 65535, 65535, 65535, 2022, 65535, 65535, 65535, 2023, 65535, 65535, 65535, 65535, 65535, 2024, 65535, 65535, 65535, 65535, 65535, 2025, 65535, 65535, 65535, 65535, 65535, 2026, 65535, 2027, 65535, 65535, 65535, 65535, 65535, 2028, 65535, 65535, 65535, 65535, 65535, 2029, 65535, 65535, 65535, 65535, 65535, 2030, 65535, 65535, 65535, 2031, 65535, 65535, 65535, 65535, 65535, 2032, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2033, 65535, 65535, 65535, 2034, 65535, 2035, 65535, 65535, 65535, 2036, 65535, 2037, 65535, 65535, 65535, 2038, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2039, 65535, 65535, 65535, 2040, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2041, 65535, 65535, 65535, 2042, 65535, 65535, 65535, 65535, 65535, 2043, 65535, 2044, 65535, 65535, 65535, 65535, 65535, 2045, 65535, 65535, 65535, 65535, 65535, 2046, 65535, 65535, 65535, 2047, 65535, 2048, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2049, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2050, 65535, 65535, 65535, 2051, 65535, 2052, 65535, 65535, 65535, 2053, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2054, 65535, 2055, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2056, 65535, 2057, 65535, 65535, 65535, 2058, 65535, 2059, 65535, 65535, 65535, 2060, 65535, 65535, 65535, 65535, 65535, 2061, 65535, 2062, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2063, 65535, 65535, 65535, 2064, 65535, 65535, 65535, 65535, 65535, 2065, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2066, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2067, 65535, 2068, 65535, 65535, 65535, 65535, 65535, 2069, 65535, 65535, 65535, 2070, 65535, 65535, 65535, 65535, 65535, 2071, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2072, 65535, 65535, 65535, 2073, 65535, 65535, 65535, 65535, 65535, 2074, 65535, 2075, 65535, 65535, 65535, 2076, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2077, 65535, 65535, 65535, 65535, 65535, 2078, 65535, 65535, 65535, 2079, 65535, 65535, 65535, 65535, 65535, 2080, 65535, 2081, 65535, 65535, 65535, 2082, 65535, 65535, 65535, 65535, 65535, 2083, 65535, 2084, 65535, 65535, 65535, 65535, 65535, 2085, 65535, 65535, 65535, 65535, 65535, 2086, 65535, 65535, 65535, 2087, 65535, 65535, 65535, 65535, 65535, 2088, 65535, 65535, 65535, 65535, 65535, 2089, 65535, 2090, 65535, 65535, 65535, 65535, 65535, 2091, 65535, 65535, 65535, 65535, 65535, 2092, 65535, 65535, 65535, 65535, 65535, 2093, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2094, 65535, 2095, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2096, 65535, 65535, 65535, 65535, 65535, 2097, 65535, 2098, 65535, 65535, 65535, 2099, 65535, 65535, 65535, 65535, 65535, 2100, 65535, 2101, 65535, 65535, 65535, 65535, 65535, 2102, 65535, 65535, 65535, 2103, 65535, 2104, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2105, 65535, 65535, 65535, 65535, 65535, 2106, 65535, 2107, 65535, 65535, 65535, 65535, 65535, 2108, 65535, 65535, 65535, 2109, 65535, 2110, 65535, 65535, 65535, 65535, 65535, 2111, 65535, 65535, 65535, 2112, 65535, 65535, 65535, 65535, 65535, 2113, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2114, 65535, 65535, 65535, 2115, 65535, 2116, 65535, 65535, 65535, 2117, 65535, 2118, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2119, 65535, 65535, 65535, 65535, 65535, 2120, 65535, 65535, 65535, 2121, 65535, 65535, 65535, 65535, 65535, 2122, 65535, 2123, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2124, 65535, 2125, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2126, 65535, 65535, 65535, 2127, 65535, 65535, 65535, 65535, 65535, 2128, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2129, 65535, 65535, 65535, 65535, 65535, 2130, 65535, 65535, 65535, 2131, 65535, 2132, 65535, 65535, 65535, 2133, 65535, 2134, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2135, 65535, 2136, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2137, 65535, 2138, 65535, 65535, 65535, 2139, 65535, 65535, 65535, 65535, 65535, 2140, 65535, 2141, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2142, 65535, 2143, 65535, 65535, 65535, 2144, 65535, 65535, 65535, 65535, 65535, 2145, 65535, 65535, 65535, 65535, 65535, 2146, 65535, 2147, 65535, 65535, 65535, 65535, 65535, 2148, 65535, 65535, 65535, 2149, 65535, 2150, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2151, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2152, 65535, 65535, 65535, 65535, 65535, 2153, 65535, 65535, 65535, 2154, 65535, 2155, 65535, 65535, 65535, 2156, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2157, 65535, 65535, 65535, 65535, 65535, 2158, 65535, 65535, 65535, 2159, 65535, 65535, 65535, 65535, 65535, 2160, 65535, 2161, 65535, 65535, 65535, 2162, 65535, 65535, 65535, 65535, 65535, 2163, 65535, 2164, 65535, 65535, 65535, 65535, 65535, 2165, 65535, 65535, 65535, 65535, 65535, 2166, 65535, 65535, 65535, 65535, 65535, 2167, 65535, 65535, 65535, 2168, 65535, 65535, 65535, 65535, 65535, 2169, 65535, 2170, 65535, 65535, 65535, 65535, 65535, 2171, 65535, 65535, 65535, 2172, 65535, 65535, 65535, 65535, 65535, 2173, 65535, 2174, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2175, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2176, 65535, 2177, 65535, 65535, 65535, 2178, 65535, 2179, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2180, 65535, 2181, 65535, 65535, 65535, 65535, 65535, 2182, 65535, 65535, 65535, 2183, 65535, 2184, 65535, 65535, 65535, 2185, 65535, 65535, 65535, 65535, 65535, 2186, 65535, 65535, 65535, 65535, 65535, 2187, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2188, 65535, 2189, 65535, 65535, 65535, 65535, 65535, 2190, 65535, 65535, 65535, 2191, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2192, 65535, 65535, 65535, 2193, 65535, 2194, 65535, 65535, 65535, 2195, 65535, 2196, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2197, 65535, 65535, 65535, 65535, 65535, 2198, 65535, 65535, 65535, 2199, 65535, 65535, 65535, 65535, 65535, 2200, 65535, 65535, 65535, 65535, 65535, 2201, 65535, 65535, 65535, 65535, 65535, 2202, 65535, 2203, 65535, 65535, 65535, 65535, 65535, 2204, 65535, 65535, 65535, 65535, 65535, 2205, 65535, 65535, 65535, 2206, 65535, 2207, 65535, 65535, 65535, 2208, 65535, 65535, 65535, 65535, 65535, 2209, 65535, 2210, 65535, 65535, 65535, 65535, 65535, 2211, 65535, 65535, 65535, 65535, 65535, 2212, 65535, 65535, 65535, 2213, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2214, 65535, 2215, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2216, 65535, 65535, 65535, 2217, 65535, 2218, 65535, 65535, 65535, 2219, 65535, 65535, 65535, 65535, 65535, 2220, 65535, 2221, 65535, 65535, 65535, 65535, 65535, 2222, 65535, 65535, 65535, 2223, 65535, 2224, 65535, 65535, 65535, 2225, 65535, 65535, 65535, 65535, 65535, 2226, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2227, 65535, 65535, 65535, 65535, 65535, 2228, 65535, 65535, 65535, 2229, 65535, 2230, 65535, 65535, 65535, 65535, 65535, 2231, 65535, 65535, 65535, 2232, 65535, 65535, 65535, 65535, 65535, 2233, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2234, 65535, 65535, 65535, 2235, 65535, 2236, 65535, 65535, 65535, 2237, 65535, 2238, 65535, 65535, 65535, 2239, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2240, 65535, 65535, 65535, 2241, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2242, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2243, 65535, 2244, 65535, 65535, 65535, 65535, 65535, 2245, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2246, 65535, 2247, 65535, 65535, 65535, 2248, 65535, 65535, 65535, 65535, 65535, 2249, 65535, 2250, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2251, 65535, 2252, 65535, 65535, 65535, 2253, 65535, 2254, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2255, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2256, 65535, 2257, 65535, 65535, 65535, 2258, 65535, 2259, 65535, 65535, 65535, 2260, 65535, 65535, 65535, 65535, 65535, 2261, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2262, 65535, 65535, 65535, 2263, 65535, 65535, 65535, 65535, 65535, 2264, 65535, 65535, 65535, 65535, 65535, 2265, 65535, 65535, 65535, 65535, 65535, 2266, 65535, 2267, 65535, 65535, 65535, 65535, 65535, 2268, 65535, 65535, 65535, 2269, 65535, 2270, 65535, 65535, 65535, 65535, 65535, 2271, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2272, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2273, 65535, 65535, 65535, 2274, 65535, 2275, 65535, 65535, 65535, 2276, 65535, 65535, 65535, 65535, 65535, 2277, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2278, 65535, 65535, 65535, 65535, 65535, 2279, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2280, 65535, 2281, 65535, 65535, 65535, 2282, 65535, 65535, 65535, 65535, 65535, 2283, 65535, 2284, 65535, 65535, 65535, 65535, 65535, 2285, 65535, 65535, 65535, 65535, 65535, 2286, 65535, 65535, 65535, 2287, 65535, 2288, 65535, 65535, 65535, 2289, 65535, 65535, 65535, 65535, 65535, 2290, 65535, 2291, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2292, 65535, 2293, 65535, 65535, 65535, 65535, 65535, 2294, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2295, 65535, 2296, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2297, 65535, 2298, 65535, 65535, 65535, 2299, 65535, 2300, 65535, 65535, 65535, 2301, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2302, 65535, 65535, 65535, 2303, 65535, 2304, 65535, 65535, 65535, 2305, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2306, 65535, 2307, 65535, 65535, 65535, 65535, 65535, 2308, 65535, 65535, 65535, 2309, 65535, 2310, 65535, 65535, 65535, 65535, 65535, 2311, 65535, 65535, 65535, 2312, 65535, 65535, 65535, 65535, 65535, 2313, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2314, 65535, 2315, 65535, 65535, 65535, 65535, 65535, 2316, 65535, 65535, 65535, 2317, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2318, 65535, 65535, 65535, 65535, 65535, 2319, 65535, 65535, 65535, 2320, 65535, 65535, 65535, 65535, 65535, 2321, 65535, 2322, 65535, 65535, 65535, 2323, 65535, 65535, 65535, 65535, 65535, 2324, 65535, 2325, 65535, 65535, 65535, 65535, 65535, 2326, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2327, 65535, 2328, 65535, 65535, 65535, 2329, 65535, 65535, 65535, 65535, 65535, 2330, 65535, 2331, 65535, 65535, 65535, 65535, 65535, 2332, 65535, 65535, 65535, 2333, 65535, 2334, 65535, 65535, 65535, 2335, 65535, 2336, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2337, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2338, 65535, 2339, 65535, 65535, 65535, 2340, 65535, 65535, 65535, 65535, 65535, 2341, 65535, 65535, 65535, 65535, 65535, 2342, 65535, 2343, 65535, 65535, 65535, 65535, 65535, 2344, 65535, 65535, 65535, 65535, 65535, 2345, 65535, 65535, 65535, 2346, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2347, 65535, 2348, 65535, 65535, 65535, 65535, 65535, 2349, 65535, 65535, 65535, 2350, 65535, 2351, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2352, 65535, 65535, 65535, 65535, 65535, 2353, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2354, 65535, 65535, 65535, 2355, 65535, 2356, 65535, 65535, 65535, 65535, 65535, 2357, 65535, 65535, 65535, 2358, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2359, 65535, 65535, 65535, 65535, 65535, 2360, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2361, 65535, 2362, 65535, 65535, 65535, 2363, 65535, 65535, 65535, 65535, 65535, 2364, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2365, 65535, 65535, 65535, 2366, 65535, 2367, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2368, 65535, 2369, 65535, 65535, 65535, 65535, 65535, 2370, 65535, 65535, 65535, 2371, 65535, 2372, 65535, 65535, 65535, 2373, 65535, 2374, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2375, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2376, 65535, 2377, 65535, 65535, 65535, 2378, 65535, 2379, 65535, 65535, 65535, 2380, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2381, 65535, 65535, 65535, 65535, 65535, 2382, 65535, 65535, 65535, 2383, 65535, 2384, 65535, 65535, 65535, 2385, 65535, 65535, 65535, 65535, 65535, 2386, 65535, 65535, 65535, 65535, 65535, 2387, 65535, 2388, 65535, 65535, 65535, 65535, 65535, 2389, 65535, 65535, 65535, 2390, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2391, 65535, 65535, 65535, 2392, 65535, 65535, 65535, 65535, 65535, 2393, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2394, 65535, 65535, 65535, 2395, 65535, 65535, 65535, 65535, 65535, 2396, 65535, 2397, 65535, 65535, 65535, 2398, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2399, 65535, 65535, 65535, 65535, 65535, 2400, 65535, 65535, 65535, 2401, 65535, 65535, 65535, 65535, 65535, 2402, 65535, 65535, 65535, 65535, 65535, 2403, 65535, 65535, 65535, 65535, 65535, 2404, 65535, 2405, 65535, 65535, 65535, 65535, 65535, 2406, 65535, 65535, 65535, 65535, 65535, 2407, 65535, 65535, 65535, 2408, 65535, 2409, 65535, 65535, 65535, 2410, 65535, 65535, 65535, 65535, 65535, 2411, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2412, 65535, 65535, 65535, 2413, 65535, 2414, 65535, 65535, 65535, 2415, 65535, 2416, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2417, 65535, 2418, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2419, 65535, 2420, 65535, 65535, 65535, 65535, 65535, 2421, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2422, 65535, 2423, 65535, 65535, 65535, 65535, 65535, 2424, 65535, 65535, 65535, 2425, 65535, 2426, 65535, 65535, 65535, 2427, 65535, 65535, 65535, 65535, 65535, 2428, 65535, 65535, 65535, 65535, 65535, 2429, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2430, 65535, 65535, 65535, 2431, 65535, 2432, 65535, 65535, 65535, 65535, 65535, 2433, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2434, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2435, 65535, 65535, 65535, 65535, 65535, 2436, 65535, 65535, 65535, 2437, 65535, 2438, 65535, 65535, 65535, 2439, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2440, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2441, 65535, 65535, 65535, 65535, 65535, 2442, 65535, 2443, 65535, 65535, 65535, 2444, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2445, 65535, 65535, 65535, 65535, 65535, 2446, 65535, 65535, 65535, 65535, 65535, 2447, 65535, 65535, 65535, 2448, 65535, 2449, 65535, 65535, 65535, 2450, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2451, 65535, 65535, 65535, 65535, 65535, 2452, 65535, 65535, 65535, 2453, 65535, 65535, 65535, 65535, 65535, 2454, 65535, 2455, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2456, 65535, 2457, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2458, 65535, 2459, 65535, 65535, 65535, 2460, 65535, 2461, 65535, 65535, 65535, 2462, 65535, 65535, 65535, 65535, 65535, 2463, 65535, 2464, 65535, 65535, 65535, 65535, 65535, 2465, 65535, 65535, 65535, 2466, 65535, 2467, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2468, 65535, 65535, 65535, 65535, 65535, 2469, 65535, 2470, 65535, 65535, 65535, 65535, 65535, 2471, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2472, 65535, 65535, 65535, 2473, 65535, 65535, 65535, 65535, 65535, 2474, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2475, 65535, 65535, 65535, 2476, 65535, 2477, 65535, 65535, 65535, 2478, 65535, 2479, 65535, 65535, 65535, 2480, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2481, 65535, 65535, 65535, 65535, 65535, 2482, 65535, 65535, 65535, 2483, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2484, 65535, 65535, 65535, 2485, 65535, 65535, 65535, 65535, 65535, 2486, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2487, 65535, 65535, 65535, 65535, 65535, 2488, 65535, 65535, 65535, 2489, 65535, 2490, 65535, 65535, 65535, 2491, 65535, 65535, 65535, 65535, 65535, 2492, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2493, 65535, 65535, 65535, 2494, 65535, 2495, 65535, 65535, 65535, 2496, 65535, 2497, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2498, 65535, 2499, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2500, 65535, 2501, 65535, 65535, 65535, 2502, 65535, 65535, 65535, 65535, 65535, 2503, 65535, 65535, 65535, 65535, 65535, 2504, 65535, 2505, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2506, 65535, 2507, 65535, 65535, 65535, 2508, 65535, 65535, 65535, 65535, 65535, 2509, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2510, 65535, 65535, 65535, 65535, 65535, 2511, 65535, 65535, 65535, 65535, 65535, 2512, 65535, 65535, 65535, 65535, 65535, 2513, 65535, 65535, 65535, 2514, 65535, 65535, 65535, 65535, 65535, 2515, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2516, 65535, 65535, 65535, 65535, 65535, 2517, 65535, 2518, 65535, 65535, 65535, 2519, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2520, 65535, 65535, 65535, 65535, 65535, 2521, 65535, 65535, 65535, 2522, 65535, 65535, 65535, 65535, 65535, 2523, 65535, 2524, 65535, 65535, 65535, 2525, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2526, 65535, 65535, 65535, 65535, 65535, 2527, 65535, 65535, 65535, 65535, 65535, 2528, 65535, 65535, 65535, 2529, 65535, 65535, 65535, 65535, 65535, 2530, 65535, 65535, 65535, 65535, 65535, 2531, 65535, 2532, 65535, 65535, 65535, 65535, 65535, 2533, 65535, 65535, 65535, 65535, 65535, 2534, 65535, 65535, 65535, 2535, 65535, 2536, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2537, 65535, 2538, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2539, 65535, 2540, 65535, 65535, 65535, 65535, 65535, 2541, 65535, 65535, 65535, 2542, 65535, 65535, 65535, 65535, 65535, 2543, 65535, 2544, 65535, 65535, 65535, 65535, 65535, 2545, 65535, 65535, 65535, 2546, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2547, 65535, 65535, 65535, 65535, 65535, 2548, 65535, 2549, 65535, 65535, 65535, 65535, 65535, 2550, 65535, 65535, 65535, 2551, 65535, 2552, 65535, 65535, 65535, 65535, 65535, 2553, 65535, 65535, 65535, 2554, 65535, 65535, 65535, 65535, 65535, 2555, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2556, 65535, 65535, 65535, 2557, 65535, 2558, 65535, 65535, 65535, 2559, 65535, 2560, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2561, 65535, 65535, 65535, 2562, 65535, 65535, 65535, 65535, 65535, 2563, 65535, 2564, 65535, 65535, 65535, 2565, 65535, 65535, 65535, 65535, 65535, 2566, 65535, 2567, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2568, 65535, 65535, 65535, 2569, 65535, 2570, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2571, 65535, 65535, 65535, 65535, 65535, 2572, 65535, 65535, 65535, 2573, 65535, 2574, 65535, 65535, 65535, 2575, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2576, 65535, 2577, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2578, 65535, 65535, 65535, 65535, 65535, 2579, 65535, 2580, 65535, 65535, 65535, 2581, 65535, 65535, 65535, 65535, 65535, 2582, 65535, 2583, 65535, 65535, 65535, 65535, 65535, 2584, 65535, 65535, 65535, 65535, 65535, 2585, 65535, 65535, 65535, 2586, 65535, 65535, 65535, 65535, 65535, 2587, 65535, 65535, 65535, 65535, 65535, 2588, 65535, 2589, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2590, 65535, 2591, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2592, 65535, 65535, 65535, 65535, 65535, 2593, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2594, 65535, 65535, 65535, 65535, 65535, 2595, 65535, 65535, 65535, 2596, 65535, 2597, 65535, 65535, 65535, 2598, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2599, 65535, 65535, 65535, 65535, 65535, 2600, 65535, 65535, 65535, 2601, 65535, 65535, 65535, 65535, 65535, 2602, 65535, 2603, 65535, 65535, 65535, 2604, 65535, 65535, 65535, 65535, 65535, 2605, 65535, 2606, 65535, 65535, 65535, 65535, 65535, 2607, 65535, 65535, 65535, 65535, 65535, 2608, 65535, 65535, 65535, 65535, 65535, 2609, 65535, 65535, 65535, 2610, 65535, 65535, 65535, 65535, 65535, 2611, 65535, 2612, 65535, 65535, 65535, 65535, 65535, 2613, 65535, 65535, 65535, 2614, 65535, 2615, 65535, 65535, 65535, 65535, 65535, 2616, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2617, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2618, 65535, 65535, 65535, 65535, 65535, 2619, 65535, 2620, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2621, 65535, 2622, 65535, 65535, 65535, 65535, 65535, 2623, 65535, 65535, 65535, 2624, 65535, 2625, 65535, 65535, 65535, 2626, 65535, 65535, 65535, 65535, 65535, 2627, 65535, 65535, 65535, 65535, 65535, 2628, 65535, 2629, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2630, 65535, 2631, 65535, 65535, 65535, 65535, 65535, 2632, 65535, 65535, 65535, 2633, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2634, 65535, 65535, 65535, 2635, 65535, 2636, 65535, 65535, 65535, 2637, 65535, 2638, 65535, 65535, 65535, 2639, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2640, 65535, 65535, 65535, 65535, 65535, 2641, 65535, 65535, 65535, 2642, 65535, 65535, 65535, 65535, 65535, 2643, 65535, 2644, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2645, 65535, 2646, 65535, 65535, 65535, 65535, 65535, 2647, 65535, 65535, 65535, 65535, 65535, 2648, 65535, 65535, 65535, 2649, 65535, 65535, 65535, 65535, 65535, 2650, 65535, 65535, 65535, 65535, 65535, 2651, 65535, 2652, 65535, 65535, 65535, 65535, 65535, 2653, 65535, 65535, 65535, 2654, 65535, 2655, 65535, 65535, 65535, 2656, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2657, 65535, 2658, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2659, 65535, 65535, 65535, 2660, 65535, 2661, 65535, 65535, 65535, 2662, 65535, 65535, 65535, 65535, 65535, 2663, 65535, 2664, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2665, 65535, 2666, 65535, 65535, 65535, 2667, 65535, 65535, 65535, 65535, 65535, 2668, 65535, 65535, 65535, 65535, 65535, 2669, 65535, 2670, 65535, 65535, 65535, 65535, 65535, 2671, 65535, 65535, 65535, 2672, 65535, 2673, 65535, 65535, 65535, 65535, 65535, 2674, 65535, 65535, 65535, 2675, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2676, 65535, 65535, 65535, 2677, 65535, 2678, 65535, 65535, 65535, 2679, 65535, 2680, 65535, 65535, 65535, 2681, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2682, 65535, 65535, 65535, 2683, 65535, 65535, 65535, 65535, 65535, 2684, 65535, 2685, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2686, 65535, 2687, 65535, 65535, 65535, 65535, 65535, 2688, 65535, 65535, 65535, 65535, 65535, 2689, 65535, 65535, 65535, 65535, 65535, 2690, 65535, 65535, 65535, 2691, 65535, 65535, 65535, 65535, 65535, 2692, 65535, 2693, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2694, 65535, 65535, 65535, 65535, 65535, 2695, 65535, 2696, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2697, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2698, 65535, 2699, 65535, 65535, 65535, 2700, 65535, 2701, 65535, 65535, 65535, 2702, 65535, 65535, 65535, 65535, 65535, 2703, 65535, 2704, 65535, 65535, 65535, 65535, 65535, 2705, 65535, 65535, 65535, 2706, 65535, 65535, 65535, 65535, 65535, 2707, 65535, 65535, 65535, 65535, 65535, 2708, 65535, 65535, 65535, 65535, 65535, 2709, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2710, 65535, 65535, 65535, 2711, 65535, 2712, 65535, 65535, 65535, 65535, 65535, 2713, 65535, 65535, 65535, 2714, 65535, 65535, 65535, 65535, 65535, 2715, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2716, 65535, 65535, 65535, 2717, 65535, 2718, 65535, 65535, 65535, 2719, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2720, 65535, 65535, 65535, 65535, 65535, 2721, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2722, 65535, 65535, 65535, 65535, 65535, 2723, 65535, 65535, 65535, 65535, 65535, 2724, 65535, 2725, 65535, 65535, 65535, 65535, 65535, 2726, 65535, 65535, 65535, 65535, 65535, 2727, 65535, 65535, 65535, 2728, 65535, 2729, 65535, 65535, 65535, 2730, 65535, 65535, 65535, 65535, 65535, 2731, 65535, 2732, 65535, 65535, 65535, 65535, 65535, 2733, 65535, 65535, 65535, 65535, 65535, 2734, 65535, 65535, 65535, 65535, 65535, 2735, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2736, 65535, 2737, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2738, 65535, 2739, 65535, 65535, 65535, 2740, 65535, 2741, 65535, 65535, 65535, 2742, 65535, 65535, 65535, 65535, 65535, 2743, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2744, 65535, 65535, 65535, 2745, 65535, 2746, 65535, 65535, 65535, 2747, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2748, 65535, 65535, 65535, 65535, 65535, 2749, 65535, 65535, 65535, 2750, 65535, 2751, 65535, 65535, 65535, 65535, 65535, 2752, 65535, 65535, 65535, 2753, 65535, 65535, 65535, 65535, 65535, 2754, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2755, 65535, 2756, 65535, 65535, 65535, 2757, 65535, 2758, 65535, 65535, 65535, 2759, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2760, 65535, 65535, 65535, 65535, 65535, 2761, 65535, 65535, 65535, 2762, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2763, 65535, 65535, 65535, 2764, 65535, 65535, 65535, 65535, 65535, 2765, 65535, 2766, 65535, 65535, 65535, 65535, 65535, 2767, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2768, 65535, 2769, 65535, 65535, 65535, 2770, 65535, 65535, 65535, 65535, 65535, 2771, 65535, 2772, 65535, 65535, 65535, 65535, 65535, 2773, 65535, 65535, 65535, 2774, 65535, 2775, 65535, 65535, 65535, 2776, 65535, 2777, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2778, 65535, 2779, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2780, 65535, 2781, 65535, 65535, 65535, 2782, 65535, 65535, 65535, 65535, 65535, 2783, 65535, 65535, 65535, 65535, 65535, 2784, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2785, 65535, 65535, 65535, 65535, 65535, 2786, 65535, 65535, 65535, 2787, 65535, 65535, 65535, 65535, 65535, 2788, 65535, 65535, 65535, 65535, 65535, 2789, 65535, 2790, 65535, 65535, 65535, 65535, 65535, 2791, 65535, 65535, 65535, 2792, 65535, 2793, 65535, 65535, 65535, 65535, 65535, 2794, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2795, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2796, 65535, 65535, 65535, 2797, 65535, 2798, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2799, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2800, 65535, 65535, 65535, 65535, 65535, 2801, 65535, 65535, 65535, 2802, 65535, 65535, 65535, 65535, 65535, 2803, 65535, 2804, 65535, 65535, 65535, 2805, 65535, 65535, 65535, 65535, 65535, 2806, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2807, 65535, 65535, 65535, 65535, 65535, 2808, 65535, 65535, 65535, 2809, 65535, 2810, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2811, 65535, 2812, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2813, 65535, 2814, 65535, 65535, 65535, 2815, 65535, 2816, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2817, 65535, 2818, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2819, 65535, 2820, 65535, 65535, 65535, 2821, 65535, 2822, 65535, 65535, 65535, 2823, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2824, 65535, 65535, 65535, 65535, 65535, 2825, 65535, 65535, 65535, 2826, 65535, 2827, 65535, 65535, 65535, 2828, 65535, 65535, 65535, 65535, 65535, 2829, 65535, 65535, 65535, 65535, 65535, 2830, 65535, 2831, 65535, 65535, 65535, 65535, 65535, 2832, 65535, 65535, 65535, 2833, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2834, 65535, 65535, 65535, 2835, 65535, 65535, 65535, 65535, 65535, 2836, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2837, 65535, 65535, 65535, 2838, 65535, 2839, 65535, 65535, 65535, 65535, 65535, 2840, 65535, 65535, 65535, 2841, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2842, 65535, 65535, 65535, 65535, 65535, 2843, 65535, 65535, 65535, 2844, 65535, 65535, 65535, 65535, 65535, 2845, 65535, 65535, 65535, 65535, 65535, 2846, 65535, 65535, 65535, 65535, 65535, 2847, 65535, 2848, 65535, 65535, 65535, 65535, 65535, 2849, 65535, 65535, 65535, 65535, 65535, 2850, 65535, 65535, 65535, 2851, 65535, 2852, 65535, 65535, 65535, 2853, 65535, 65535, 65535, 65535, 65535, 2854, 65535, 2855, 65535, 65535, 65535, 65535, 65535, 2856, 65535, 65535, 65535, 2857, 65535, 2858, 65535, 65535, 65535, 2859, 65535, 2860, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2861, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2862, 65535, 2863, 65535, 65535, 65535, 65535, 65535, 2864, 65535, 65535, 65535, 2865, 65535, 65535, 65535, 65535, 65535, 2866, 65535, 2867, 65535, 65535, 65535, 65535, 65535, 2868, 65535, 65535, 65535, 2869, 65535, 2870, 65535, 65535, 65535, 2871, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2872, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2873, 65535, 65535, 65535, 2874, 65535, 2875, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2876, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2877, 65535, 65535, 65535, 2878, 65535, 2879, 65535, 65535, 65535, 2880, 65535, 2881, 65535, 65535, 65535, 2882, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2883, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2884, 65535, 2885, 65535, 65535, 65535, 2886, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2887, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2888, 65535, 65535, 65535, 2889, 65535, 2890, 65535, 65535, 65535, 2891, 65535, 65535, 65535, 65535, 65535, 2892, 65535, 2893, 65535, 65535, 65535, 65535, 65535, 2894, 65535, 65535, 65535, 2895, 65535, 65535, 65535, 65535, 65535, 2896, 65535, 2897, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2898, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2899, 65535, 2900, 65535, 65535, 65535, 2901, 65535, 2902, 65535, 65535, 65535, 2903, 65535, 65535, 65535, 65535, 65535, 2904, 65535, 2905, 65535, 65535, 65535, 65535, 65535, 2906, 65535, 65535, 65535, 2907, 65535, 2908, 65535, 65535, 65535, 2909, 65535, 65535, 65535, 65535, 65535, 2910, 65535, 65535, 65535, 65535, 65535, 2911, 65535, 2912, 65535, 65535, 65535, 65535, 65535, 2913, 65535, 65535, 65535, 65535, 65535, 2914, 65535, 65535, 65535, 65535, 65535, 2915, 65535, 65535, 65535, 2916, 65535, 65535, 65535, 65535, 65535, 2917, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2918, 65535, 65535, 65535, 2919, 65535, 65535, 65535, 65535, 65535, 2920, 65535, 2921, 65535, 65535, 65535, 2922, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2923, 65535, 65535, 65535, 65535, 65535, 2924, 65535, 65535, 65535, 2925, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2926, 65535, 65535, 65535, 2927, 65535, 65535, 65535, 65535, 65535, 2928, 65535, 2929, 65535, 65535, 65535, 65535, 65535, 2930, 65535, 65535, 65535, 65535, 65535, 2931, 65535, 65535, 65535, 2932, 65535, 2933, 65535, 65535, 65535, 2934, 65535, 65535, 65535, 65535, 65535, 2935, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2936, 65535, 65535, 65535, 2937, 65535, 2938, 65535, 65535, 65535, 2939, 65535, 2940, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2941, 65535, 2942, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2943, 65535, 2944, 65535, 65535, 65535, 2945, 65535, 2946, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2947, 65535, 2948, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2949, 65535, 2950, 65535, 65535, 65535, 2951, 65535, 65535, 65535, 65535, 65535, 2952, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2953, 65535, 65535, 65535, 65535, 65535, 2954, 65535, 65535, 65535, 2955, 65535, 2956, 65535, 65535, 65535, 65535, 65535, 2957, 65535, 65535, 65535, 2958, 65535, 65535, 65535, 65535, 65535, 2959, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2960, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2961, 65535, 2962, 65535, 65535, 65535, 2963, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2964, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2965, 65535, 65535, 65535, 65535, 65535, 2966, 65535, 2967, 65535, 65535, 65535, 2968, 65535, 65535, 65535, 65535, 65535, 2969, 65535, 2970, 65535, 65535, 65535, 65535, 65535, 2971, 65535, 65535, 65535, 65535, 65535, 2972, 65535, 65535, 65535, 2973, 65535, 65535, 65535, 65535, 65535, 2974, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2975, 65535, 65535, 65535, 65535, 65535, 2976, 65535, 65535, 65535, 65535, 65535, 2977, 65535, 65535, 65535, 2978, 65535, 2979, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2980, 65535, 2981, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2982, 65535, 2983, 65535, 65535, 65535, 2984, 65535, 2985, 65535, 65535, 65535, 2986, 65535, 65535, 65535, 65535, 65535, 2987, 65535, 2988, 65535, 65535, 65535, 65535, 65535, 2989, 65535, 65535, 65535, 2990, 65535, 2991, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2992, 65535, 65535, 65535, 65535, 65535, 2993, 65535, 2994, 65535, 65535, 65535, 65535, 65535, 2995, 65535, 65535, 65535, 2996, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 2997, 65535, 65535, 65535, 2998, 65535, 65535, 65535, 65535, 65535, 2999, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3000, 65535, 65535, 65535, 3001, 65535, 3002, 65535, 65535, 65535, 3003, 65535, 3004, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3005, 65535, 65535, 65535, 65535, 65535, 3006, 65535, 65535, 65535, 3007, 65535, 65535, 65535, 65535, 65535, 3008, 65535, 3009, 65535, 65535, 65535, 3010, 65535, 65535, 65535, 65535, 65535, 3011, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3012, 65535, 65535, 65535, 3013, 65535, 3014, 65535, 65535, 65535, 3015, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3016, 65535, 65535, 65535, 65535, 65535, 3017, 65535, 65535, 65535, 3018, 65535, 3019, 65535, 65535, 65535, 3020, 65535, 3021, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3022, 65535, 3023, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3024, 65535, 65535, 65535, 65535, 65535, 3025, 65535, 65535, 65535, 65535, 65535, 3026, 65535, 65535, 65535, 65535, 65535, 3027, 65535, 3028, 65535, 65535, 65535, 65535, 65535, 3029, 65535, 65535, 65535, 3030, 65535, 3031, 65535, 65535, 65535, 3032, 65535, 65535, 65535, 65535, 65535, 3033, 65535, 65535, 65535, 65535, 65535, 3034, 65535, 3035, 65535, 65535, 65535, 65535, 65535, 3036, 65535, 65535, 65535, 65535, 65535, 3037, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3038, 65535, 65535, 65535, 65535, 65535, 3039, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3040, 65535, 65535, 65535, 3041, 65535, 3042, 65535, 65535, 65535, 3043, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3044, 65535, 65535, 65535, 65535, 65535, 3045, 65535, 65535, 65535, 3046, 65535, 65535, 65535, 65535, 65535, 3047, 65535, 3048, 65535, 65535, 65535, 3049, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3050, 65535, 65535, 65535, 65535, 65535, 3051, 65535, 65535, 65535, 65535, 65535, 3052, 65535, 65535, 65535, 65535, 65535, 3053, 65535, 65535, 65535, 3054, 65535, 65535, 65535, 65535, 65535, 3055, 65535, 3056, 65535, 65535, 65535, 65535, 65535, 3057, 65535, 65535, 65535, 3058, 65535, 3059, 65535, 65535, 65535, 3060, 65535, 3061, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3062, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3063, 65535, 3064, 65535, 65535, 65535, 65535, 65535, 3065, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3066, 65535, 3067, 65535, 65535, 65535, 65535, 65535, 3068, 65535, 65535, 65535, 3069, 65535, 65535, 65535, 65535, 65535, 3070, 65535, 65535, 65535, 65535, 65535, 3071, 65535, 65535, 65535, 65535, 65535, 3072, 65535, 3073, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3074, 65535, 3075, 65535, 65535, 65535, 65535, 65535, 3076, 65535, 65535, 65535, 3077, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3078, 65535, 65535, 65535, 3079, 65535, 3080, 65535, 65535, 65535, 3081, 65535, 3082, 65535, 65535, 65535, 3083, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3084, 65535, 65535, 65535, 3085, 65535, 65535, 65535, 65535, 65535, 3086, 65535, 3087, 65535, 65535, 65535, 3088, 65535, 65535, 65535, 65535, 65535, 3089, 65535, 3090, 65535, 65535, 65535, 65535, 65535, 3091, 65535, 65535, 65535, 65535, 65535, 3092, 65535, 65535, 65535, 3093, 65535, 3094, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3095, 65535, 3096, 65535, 65535, 65535, 65535, 65535, 3097, 65535, 65535, 65535, 3098, 65535, 3099, 65535, 65535, 65535, 3100, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3101, 65535, 3102, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3103, 65535, 65535, 65535, 3104, 65535, 3105, 65535, 65535, 65535, 3106, 65535, 65535, 65535, 65535, 65535, 3107, 65535, 3108, 65535, 65535, 65535, 65535, 65535, 3109, 65535, 65535, 65535, 65535, 65535, 3110, 65535, 65535, 65535, 3111, 65535, 65535, 65535, 65535, 65535, 3112, 65535, 65535, 65535, 65535, 65535, 3113, 65535, 3114, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3115, 65535, 3116, 65535, 65535, 65535, 65535, 65535, 3117, 65535, 65535, 65535, 3118, 65535, 65535, 65535, 65535, 65535, 3119, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3120, 65535, 65535, 65535, 3121, 65535, 3122, 65535, 65535, 65535, 3123, 65535, 3124, 65535, 65535, 65535, 3125, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3126, 65535, 65535, 65535, 3127, 65535, 65535, 65535, 65535, 65535, 3128, 65535, 3129, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3130, 65535, 3131, 65535, 65535, 65535, 65535, 65535, 3132, 65535, 65535, 65535, 65535, 65535, 3133, 65535, 65535, 65535, 3134, 65535, 3135, 65535, 65535, 65535, 3136, 65535, 65535, 65535, 65535, 65535, 3137, 65535, 3138, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3139, 65535, 3140, 65535, 65535, 65535, 65535, 65535, 3141, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3142, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3143, 65535, 65535, 65535, 65535, 65535, 3144, 65535, 3145, 65535, 65535, 65535, 3146, 65535, 65535, 65535, 65535, 65535, 3147, 65535, 3148, 65535, 65535, 65535, 65535, 65535, 3149, 65535, 65535, 65535, 3150, 65535, 65535, 65535, 65535, 65535, 3151, 65535, 65535, 65535, 65535, 65535, 3152, 65535, 65535, 65535, 65535, 65535, 3153, 65535, 3154, 65535, 65535, 65535, 65535, 65535, 3155, 65535, 65535, 65535, 3156, 65535, 3157, 65535, 65535, 65535, 65535, 65535, 3158, 65535, 65535, 65535, 3159, 65535, 65535, 65535, 65535, 65535, 3160, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3161, 65535, 65535, 65535, 3162, 65535, 3163, 65535, 65535, 65535, 3164, 65535, 65535, 65535, 65535, 65535, 3165, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3166, 65535, 65535, 65535, 65535, 65535, 3167, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3168, 65535, 3169, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3170, 65535, 3171, 65535, 65535, 65535, 65535, 65535, 3172, 65535, 65535, 65535, 65535, 65535, 3173, 65535, 65535, 65535, 3174, 65535, 65535, 65535, 65535, 65535, 3175, 65535, 65535, 65535, 65535, 65535, 3176, 65535, 3177, 65535, 65535, 65535, 65535, 65535, 3178, 65535, 65535, 65535, 3179, 65535, 3180, 65535, 65535, 65535, 65535, 65535, 3181, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3182, 65535, 3183, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3184, 65535, 65535, 65535, 3185, 65535, 3186, 65535, 65535, 65535, 3187, 65535, 65535, 65535, 65535, 65535, 3188, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3189, 65535, 3190, 65535, 65535, 65535, 3191, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3192, 65535, 3193, 65535, 65535, 65535, 65535, 65535, 3194, 65535, 65535, 65535, 3195, 65535, 3196, 65535, 65535, 65535, 65535, 65535, 3197, 65535, 65535, 65535, 3198, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3199, 65535, 3200, 65535, 65535, 65535, 3201, 65535, 3202, 65535, 65535, 65535, 3203, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3204, 65535, 65535, 65535, 65535, 65535, 3205, 65535, 65535, 65535, 3206, 65535, 65535, 65535, 65535, 65535, 3207, 65535, 3208, 65535, 65535, 65535, 3209, 65535, 65535, 65535, 65535, 65535, 3210, 65535, 3211, 65535, 65535, 65535, 65535, 65535, 3212, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3213, 65535, 65535, 65535, 3214, 65535, 65535, 65535, 65535, 65535, 3215, 65535, 3216, 65535, 65535, 65535, 65535, 65535, 3217, 65535, 65535, 65535, 3218, 65535, 65535, 65535, 65535, 65535, 3219, 65535, 3220, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3221, 65535, 3222, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3223, 65535, 3224, 65535, 65535, 65535, 3225, 65535, 65535, 65535, 65535, 65535, 3226, 65535, 65535, 65535, 65535, 65535, 3227, 65535, 3228, 65535, 65535, 65535, 65535, 65535, 3229, 65535, 65535, 65535, 65535, 65535, 3230, 65535, 65535, 65535, 3231, 65535, 65535, 65535, 65535, 65535, 3232, 65535, 65535, 65535, 65535, 65535, 3233, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3234, 65535, 65535, 65535, 3235, 65535, 3236, 65535, 65535, 65535, 65535, 65535, 3237, 65535, 65535, 65535, 3238, 65535, 65535, 65535, 65535, 65535, 3239, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3240, 65535, 65535, 65535, 3241, 65535, 3242, 65535, 65535, 65535, 65535, 65535, 3243, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3244, 65535, 65535, 65535, 65535, 65535, 3245, 65535, 65535, 65535, 3246, 65535, 65535, 65535, 65535, 65535, 3247, 65535, 65535, 65535, 65535, 65535, 3248, 65535, 65535, 65535, 65535, 65535, 3249, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3250, 65535, 65535, 65535, 65535, 65535, 3251, 65535, 65535, 65535, 3252, 65535, 3253, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3254, 65535, 3255, 65535, 65535, 65535, 65535, 65535, 3256, 65535, 65535, 65535, 65535, 65535, 3257, 65535, 65535, 65535, 3258, 65535, 3259, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3260, 65535, 3261, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3262, 65535, 3263, 65535, 65535, 65535, 3264, 65535, 3265, 65535, 65535, 65535, 3266, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3267, 65535, 65535, 65535, 65535, 65535, 3268, 65535, 65535, 65535, 3269, 65535, 3270, 65535, 65535, 65535, 3271, 65535, 65535, 65535, 65535, 65535, 3272, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3273, 65535, 65535, 65535, 65535, 65535, 3274, 65535, 65535, 65535, 3275, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3276, 65535, 65535, 65535, 3277, 65535, 65535, 65535, 65535, 65535, 3278, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3279, 65535, 65535, 65535, 3280, 65535, 3281, 65535, 65535, 65535, 3282, 65535, 3283, 65535, 65535, 65535, 3284, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3285, 65535, 65535, 65535, 65535, 65535, 3286, 65535, 65535, 65535, 3287, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3288, 65535, 65535, 65535, 65535, 65535, 3289, 65535, 3290, 65535, 65535, 65535, 65535, 65535, 3291, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3292, 65535, 3293, 65535, 65535, 65535, 3294, 65535, 65535, 65535, 65535, 65535, 3295, 65535, 3296, 65535, 65535, 65535, 65535, 65535, 3297, 65535, 65535, 65535, 3298, 65535, 3299, 65535, 65535, 65535, 3300, 65535, 3301, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3302, 65535, 3303, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3304, 65535, 3305, 65535, 65535, 65535, 65535, 65535, 3306, 65535, 65535, 65535, 3307, 65535, 65535, 65535, 65535, 65535, 3308, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3309, 65535, 65535, 65535, 3310, 65535, 3311, 65535, 65535, 65535, 3312, 65535, 65535, 65535, 65535, 65535, 3313, 65535, 65535, 65535, 65535, 65535, 3314, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3315, 65535, 65535, 65535, 3316, 65535, 3317, 65535, 65535, 65535, 65535, 65535, 3318, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3319, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3320, 65535, 65535, 65535, 3321, 65535, 3322, 65535, 65535, 65535, 3323, 65535, 65535, 65535, 65535, 65535, 3324, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3325, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3326, 65535, 65535, 65535, 65535, 65535, 3327, 65535, 3328, 65535, 65535, 65535, 3329, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3330, 65535, 65535, 65535, 65535, 65535, 3331, 65535, 65535, 65535, 65535, 65535, 3332, 65535, 65535, 65535, 3333, 65535, 3334, 65535, 65535, 65535, 3335, 65535, 65535, 65535, 65535, 65535, 3336, 65535, 3337, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3338, 65535, 65535, 65535, 65535, 65535, 3339, 65535, 3340, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3341, 65535, 3342, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3343, 65535, 3344, 65535, 65535, 65535, 3345, 65535, 3346, 65535, 65535, 65535, 3347, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3348, 65535, 65535, 65535, 65535, 65535, 3349, 65535, 65535, 65535, 3350, 65535, 3351, 65535, 65535, 65535, 3352, 65535, 65535, 65535, 65535, 65535, 3353, 65535, 65535, 65535, 65535, 65535, 3354, 65535, 3355, 65535, 65535, 65535, 65535, 65535, 3356, 65535, 65535, 65535, 65535, 65535, 3357, 65535, 65535, 65535, 65535, 65535, 3358, 65535, 65535, 65535, 3359, 65535, 65535, 65535, 65535, 65535, 3360, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3361, 65535, 65535, 65535, 3362, 65535, 3363, 65535, 65535, 65535, 65535, 65535, 3364, 65535, 65535, 65535, 3365, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3366, 65535, 65535, 65535, 65535, 65535, 3367, 65535, 65535, 65535, 3368, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3369, 65535, 65535, 65535, 3370, 65535, 65535, 65535, 65535, 65535, 3371, 65535, 3372, 65535, 65535, 65535, 65535, 65535, 3373, 65535, 65535, 65535, 65535, 65535, 3374, 65535, 65535, 65535, 3375, 65535, 3376, 65535, 65535, 65535, 3377, 65535, 65535, 65535, 65535, 65535, 3378, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3379, 65535, 65535, 65535, 3380, 65535, 3381, 65535, 65535, 65535, 3382, 65535, 3383, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3384, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3385, 65535, 3386, 65535, 65535, 65535, 3387, 65535, 3388, 65535, 65535, 65535, 3389, 65535, 65535, 65535, 65535, 65535, 3390, 65535, 3391, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3392, 65535, 3393, 65535, 65535, 65535, 3394, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3395, 65535, 65535, 65535, 65535, 65535, 3396, 65535, 65535, 65535, 3397, 65535, 3398, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3399, 65535, 65535, 65535, 65535, 65535, 3400, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3401, 65535, 65535, 65535, 3402, 65535, 65535, 65535, 65535, 65535, 3403, 65535, 3404, 65535, 65535, 65535, 3405, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3406, 65535, 65535, 65535, 65535, 65535, 3407, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3408, 65535, 3409, 65535, 65535, 65535, 3410, 65535, 65535, 65535, 65535, 65535, 3411, 65535, 3412, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3413, 65535, 65535, 65535, 3414, 65535, 65535, 65535, 65535, 65535, 3415, 65535, 65535, 65535, 65535, 65535, 3416, 65535, 3417, 65535, 65535, 65535, 65535, 65535, 3418, 65535, 65535, 65535, 65535, 65535, 3419, 65535, 65535, 65535, 3420, 65535, 3421, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3422, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3423, 65535, 3424, 65535, 65535, 65535, 3425, 65535, 3426, 65535, 65535, 65535, 3427, 65535, 65535, 65535, 65535, 65535, 3428, 65535, 3429, 65535, 65535, 65535, 65535, 65535, 3430, 65535, 65535, 65535, 3431, 65535, 3432, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3433, 65535, 65535, 65535, 65535, 65535, 3434, 65535, 3435, 65535, 65535, 65535, 65535, 65535, 3436, 65535, 65535, 65535, 3437, 65535, 3438, 65535, 65535, 65535, 65535, 65535, 3439, 65535, 65535, 65535, 3440, 65535, 65535, 65535, 65535, 65535, 3441, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3442, 65535, 65535, 65535, 3443, 65535, 65535, 65535, 65535, 65535, 3444, 65535, 3445, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3446, 65535, 65535, 65535, 65535, 65535, 3447, 65535, 65535, 65535, 3448, 65535, 65535, 65535, 65535, 65535, 3449, 65535, 3450, 65535, 65535, 65535, 3451, 65535, 65535, 65535, 65535, 65535, 3452, 65535, 3453, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3454, 65535, 65535, 65535, 3455, 65535, 3456, 65535, 65535, 65535, 3457, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3458, 65535, 65535, 65535, 3459, 65535, 3460, 65535, 65535, 65535, 3461, 65535, 3462, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3463, 65535, 3464, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3465, 65535, 65535, 65535, 65535, 65535, 3466, 65535, 3467, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3468, 65535, 3469, 65535, 65535, 65535, 65535, 65535, 3470, 65535, 65535, 65535, 3471, 65535, 3472, 65535, 65535, 65535, 3473, 65535, 65535, 65535, 65535, 65535, 3474, 65535, 65535, 65535, 65535, 65535, 3475, 65535, 3476, 65535, 65535, 65535, 65535, 65535, 3477, 65535, 65535, 65535, 3478, 65535, 3479, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3480, 65535, 65535, 65535, 65535, 65535, 3481, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3482, 65535, 65535, 65535, 65535, 65535, 3483, 65535, 65535, 65535, 3484, 65535, 3485, 65535, 65535, 65535, 3486, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3487, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3488, 65535, 65535, 65535, 65535, 65535, 3489, 65535, 3490, 65535, 65535, 65535, 3491, 65535, 65535, 65535, 65535, 65535, 3492, 65535, 3493, 65535, 65535, 65535, 65535, 65535, 3494, 65535, 65535, 65535, 65535, 65535, 3495, 65535, 65535, 65535, 65535, 65535, 3496, 65535, 65535, 65535, 3497, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3498, 65535, 65535, 65535, 65535, 65535, 3499, 65535, 65535, 65535, 3500, 65535, 3501, 65535, 65535, 65535, 3502, 65535, 3503, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3504, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3505, 65535, 3506, 65535, 65535, 65535, 3507, 65535, 3508, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3509, 65535, 3510, 65535, 65535, 65535, 65535, 65535, 3511, 65535, 65535, 65535, 3512, 65535, 3513, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3514, 65535, 65535, 65535, 65535, 65535, 3515, 65535, 3516, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3517, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3518, 65535, 65535, 65535, 3519, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3520, 65535, 65535, 65535, 3521, 65535, 3522, 65535, 65535, 65535, 3523, 65535, 3524, 65535, 65535, 65535, 3525, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3526, 65535, 65535, 65535, 65535, 65535, 3527, 65535, 65535, 65535, 3528, 65535, 65535, 65535, 65535, 65535, 3529, 65535, 3530, 65535, 65535, 65535, 3531, 65535, 65535, 65535, 65535, 65535, 3532, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3533, 65535, 65535, 65535, 65535, 65535, 3534, 65535, 65535, 65535, 3535, 65535, 3536, 65535, 65535, 65535, 3537, 65535, 65535, 65535, 65535, 65535, 3538, 65535, 3539, 65535, 65535, 65535, 65535, 65535, 3540, 65535, 65535, 65535, 3541, 65535, 3542, 65535, 65535, 65535, 3543, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3544, 65535, 3545, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3546, 65535, 65535, 65535, 3547, 65535, 65535, 65535, 65535, 65535, 3548, 65535, 65535, 65535, 65535, 65535, 3549, 65535, 3550, 65535, 65535, 65535, 65535, 65535, 3551, 65535, 65535, 65535, 3552, 65535, 3553, 65535, 65535, 65535, 3554, 65535, 65535, 65535, 65535, 65535, 3555, 65535, 65535, 65535, 65535, 65535, 3556, 65535, 3557, 65535, 65535, 65535, 65535, 65535, 3558, 65535, 65535, 65535, 65535, 65535, 3559, 65535, 65535, 65535, 65535, 65535, 3560, 65535, 65535, 65535, 3561, 65535, 65535, 65535, 65535, 65535, 3562, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3563, 65535, 3564, 65535, 65535, 65535, 3565, 65535, 3566, 65535, 65535, 65535, 3567, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3568, 65535, 65535, 65535, 3569, 65535, 65535, 65535, 65535, 65535, 3570, 65535, 3571, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3572, 65535, 65535, 65535, 65535, 65535, 3573, 65535, 65535, 65535, 65535, 65535, 3574, 65535, 65535, 65535, 3575, 65535, 3576, 65535, 65535, 65535, 3577, 65535, 65535, 65535, 65535, 65535, 3578, 65535, 3579, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3580, 65535, 3581, 65535, 65535, 65535, 3582, 65535, 3583, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3584, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3585, 65535, 3586, 65535, 65535, 65535, 65535, 65535, 3587, 65535, 65535, 65535, 3588, 65535, 65535, 65535, 65535, 65535, 3589, 65535, 3590, 65535, 65535, 65535, 65535, 65535, 3591, 65535, 65535, 65535, 3592, 65535, 65535, 65535, 65535, 65535, 3593, 65535, 65535, 65535, 65535, 65535, 3594, 65535, 65535, 65535, 65535, 65535, 3595, 65535, 3596, 65535, 65535, 65535, 65535, 65535, 3597, 65535, 65535, 65535, 3598, 65535, 3599, 65535, 65535, 65535, 65535, 65535, 3600, 65535, 65535, 65535, 3601, 65535, 65535, 65535, 65535, 65535, 3602, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3603, 65535, 65535, 65535, 3604, 65535, 3605, 65535, 65535, 65535, 3606, 65535, 65535, 65535, 65535, 65535, 3607, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3608, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3609, 65535, 3610, 65535, 65535, 65535, 3611, 65535, 65535, 65535, 65535, 65535, 3612, 65535, 3613, 65535, 65535, 65535, 65535, 65535, 3614, 65535, 65535, 65535, 65535, 65535, 3615, 65535, 65535, 65535, 3616, 65535, 3617, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3618, 65535, 3619, 65535, 65535, 65535, 65535, 65535, 3620, 65535, 65535, 65535, 3621, 65535, 3622, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3623, 65535, 3624, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3625, 65535, 3626, 65535, 65535, 65535, 3627, 65535, 3628, 65535, 65535, 65535, 3629, 65535, 65535, 65535, 65535, 65535, 3630, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3631, 65535, 65535, 65535, 65535, 65535, 3632, 65535, 65535, 65535, 3633, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3634, 65535, 3635, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3636, 65535, 3637, 65535, 65535, 65535, 65535, 65535, 3638, 65535, 65535, 65535, 3639, 65535, 65535, 65535, 65535, 65535, 3640, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3641, 65535, 3642, 65535, 65535, 65535, 3643, 65535, 3644, 65535, 65535, 65535, 3645, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3646, 65535, 65535, 65535, 65535, 65535, 3647, 65535, 65535, 65535, 3648, 65535, 65535, 65535, 65535, 65535, 3649, 65535, 3650, 65535, 65535, 65535, 3651, 65535, 65535, 65535, 65535, 65535, 3652, 65535, 3653, 65535, 65535, 65535, 65535, 65535, 3654, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3655, 65535, 3656, 65535, 65535, 65535, 3657, 65535, 65535, 65535, 65535, 65535, 3658, 65535, 3659, 65535, 65535, 65535, 65535, 65535, 3660, 65535, 65535, 65535, 3661, 65535, 3662, 65535, 65535, 65535, 65535, 65535, 3663, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3664, 65535, 3665, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3666, 65535, 65535, 65535, 65535, 65535, 3667, 65535, 65535, 65535, 65535, 65535, 3668, 65535, 65535, 65535, 65535, 65535, 3669, 65535, 3670, 65535, 65535, 65535, 65535, 65535, 3671, 65535, 65535, 65535, 65535, 65535, 3672, 65535, 65535, 65535, 3673, 65535, 65535, 65535, 65535, 65535, 3674, 65535, 65535, 65535, 65535, 65535, 3675, 65535, 3676, 65535, 65535, 65535, 65535, 65535, 3677, 65535, 65535, 65535, 3678, 65535, 3679, 65535, 65535, 65535, 65535, 65535, 3680, 65535, 65535, 65535, 3681, 65535, 65535, 65535, 65535, 65535, 3682, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3683, 65535, 65535, 65535, 3684, 65535, 3685, 65535, 65535, 65535, 65535, 65535, 3686, 65535, 65535, 65535, 3687, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3688, 65535, 65535, 65535, 65535, 65535, 3689, 65535, 65535, 65535, 3690, 65535, 65535, 65535, 65535, 65535, 3691, 65535, 3692, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3693, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3694, 65535, 65535, 65535, 65535, 65535, 3695, 65535, 65535, 65535, 3696, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3697, 65535, 3698, 65535, 65535, 65535, 65535, 65535, 3699, 65535, 65535, 65535, 3700, 65535, 3701, 65535, 65535, 65535, 3702, 65535, 3703, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3704, 65535, 3705, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3706, 65535, 65535, 65535, 3707, 65535, 3708, 65535, 65535, 65535, 3709, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3710, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3711, 65535, 3712, 65535, 65535, 65535, 3713, 65535, 65535, 65535, 65535, 65535, 3714, 65535, 65535, 65535, 65535, 65535, 3715, 65535, 3716, 65535, 65535, 65535, 65535, 65535, 3717, 65535, 65535, 65535, 3718, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3719, 65535, 65535, 65535, 3720, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3721, 65535, 65535, 65535, 3722, 65535, 3723, 65535, 65535, 65535, 3724, 65535, 3725, 65535, 65535, 65535, 3726, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3727, 65535, 65535, 65535, 65535, 65535, 3728, 65535, 65535, 65535, 3729, 65535, 65535, 65535, 65535, 65535, 3730, 65535, 65535, 65535, 65535, 65535, 3731, 65535, 65535, 65535, 65535, 65535, 3732, 65535, 3733, 65535, 65535, 65535, 65535, 65535, 3734, 65535, 65535, 65535, 65535, 65535, 3735, 65535, 65535, 65535, 65535, 65535, 3736, 65535, 65535, 65535, 3737, 65535, 65535, 65535, 65535, 65535, 3738, 65535, 3739, 65535, 65535, 65535, 65535, 65535, 3740, 65535, 65535, 65535, 3741, 65535, 65535, 65535, 65535, 65535, 3742, 65535, 3743, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3744, 65535, 3745, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3746, 65535, 3747, 65535, 65535, 65535, 65535, 65535, 3748, 65535, 65535, 65535, 3749, 65535, 65535, 65535, 65535, 65535, 3750, 65535, 3751, 65535, 65535, 65535, 65535, 65535, 3752, 65535, 65535, 65535, 3753, 65535, 3754, 65535, 65535, 65535, 3755, 65535, 65535, 65535, 65535, 65535, 3756, 65535, 65535, 65535, 65535, 65535, 3757, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3758, 65535, 65535, 65535, 3759, 65535, 3760, 65535, 65535, 65535, 65535, 65535, 3761, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3762, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3763, 65535, 65535, 65535, 3764, 65535, 3765, 65535, 65535, 65535, 3766, 65535, 3767, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3768, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3769, 65535, 65535, 65535, 65535, 65535, 3770, 65535, 65535, 65535, 65535, 65535, 3771, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3772, 65535, 65535, 65535, 65535, 65535, 3773, 65535, 65535, 65535, 65535, 65535, 3774, 65535, 65535, 65535, 3775, 65535, 3776, 65535, 65535, 65535, 3777, 65535, 65535, 65535, 65535, 65535, 3778, 65535, 3779, 65535, 65535, 65535, 65535, 65535, 3780, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3781, 65535, 3782, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3783, 65535, 3784, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3785, 65535, 3786, 65535, 65535, 65535, 3787, 65535, 3788, 65535, 65535, 65535, 3789, 65535, 65535, 65535, 65535, 65535, 3790, 65535, 3791, 65535, 65535, 65535, 65535, 65535, 3792, 65535, 65535, 65535, 3793, 65535, 3794, 65535, 65535, 65535, 3795, 65535, 65535, 65535, 65535, 65535, 3796, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3797, 65535, 65535, 65535, 65535, 65535, 3798, 65535, 65535, 65535, 65535, 65535, 3799, 65535, 65535, 65535, 65535, 65535, 3800, 65535, 65535, 65535, 3801, 65535, 65535, 65535, 65535, 65535, 3802, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3803, 65535, 65535, 65535, 3804, 65535, 3805, 65535, 65535, 65535, 3806, 65535, 3807, 65535, 65535, 65535, 3808, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3809, 65535, 65535, 65535, 65535, 65535, 3810, 65535, 65535, 65535, 3811, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3812, 65535, 65535, 65535, 3813, 65535, 65535, 65535, 65535, 65535, 3814, 65535, 3815, 65535, 65535, 65535, 65535, 65535, 3816, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3817, 65535, 3818, 65535, 65535, 65535, 3819, 65535, 65535, 65535, 65535, 65535, 3820, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3821, 65535, 65535, 65535, 3822, 65535, 3823, 65535, 65535, 65535, 3824, 65535, 3825, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3826, 65535, 3827, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3828, 65535, 3829, 65535, 65535, 65535, 3830, 65535, 3831, 65535, 65535, 65535, 3832, 65535, 65535, 65535, 65535, 65535, 3833, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3834, 65535, 3835, 65535, 65535, 65535, 3836, 65535, 65535, 65535, 65535, 65535, 3837, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3838, 65535, 65535, 65535, 65535, 65535, 3839, 65535, 65535, 65535, 3840, 65535, 3841, 65535, 65535, 65535, 65535, 65535, 3842, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3843, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3844, 65535, 65535, 65535, 3845, 65535, 65535, 65535, 65535, 65535, 3846, 65535, 65535, 65535, 65535, 65535, 3847, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3848, 65535, 65535, 65535, 65535, 65535, 3849, 65535, 65535, 65535, 3850, 65535, 65535, 65535, 65535, 65535, 3851, 65535, 3852, 65535, 65535, 65535, 3853, 65535, 65535, 65535, 65535, 65535, 3854, 65535, 3855, 65535, 65535, 65535, 65535, 65535, 3856, 65535, 65535, 65535, 65535, 65535, 3857, 65535, 65535, 65535, 3858, 65535, 65535, 65535, 65535, 65535, 3859, 65535, 65535, 65535, 65535, 65535, 3860, 65535, 3861, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3862, 65535, 65535, 65535, 3863, 65535, 3864, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3865, 65535, 3866, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3867, 65535, 3868, 65535, 65535, 65535, 3869, 65535, 3870, 65535, 65535, 65535, 3871, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3872, 65535, 65535, 65535, 65535, 65535, 3873, 65535, 65535, 65535, 3874, 65535, 3875, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3876, 65535, 65535, 65535, 65535, 65535, 3877, 65535, 3878, 65535, 65535, 65535, 65535, 65535, 3879, 65535, 65535, 65535, 3880, 65535, 3881, 65535, 65535, 65535, 65535, 65535, 3882, 65535, 65535, 65535, 3883, 65535, 65535, 65535, 65535, 65535, 3884, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3885, 65535, 65535, 65535, 3886, 65535, 3887, 65535, 65535, 65535, 65535, 65535, 3888, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3889, 65535, 65535, 65535, 65535, 65535, 3890, 65535, 65535, 65535, 3891, 65535, 65535, 65535, 65535, 65535, 3892, 65535, 3893, 65535, 65535, 65535, 3894, 65535, 65535, 65535, 65535, 65535, 3895, 65535, 3896, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3897, 65535, 65535, 65535, 3898, 65535, 3899, 65535, 65535, 65535, 3900, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3901, 65535, 65535, 65535, 65535, 65535, 3902, 65535, 65535, 65535, 3903, 65535, 3904, 65535, 65535, 65535, 3905, 65535, 3906, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3907, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3908, 65535, 65535, 65535, 65535, 65535, 3909, 65535, 3910, 65535, 65535, 65535, 3911, 65535, 65535, 65535, 65535, 65535, 3912, 65535, 3913, 65535, 65535, 65535, 65535, 65535, 3914, 65535, 65535, 65535, 3915, 65535, 3916, 65535, 65535, 65535, 3917, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3918, 65535, 3919, 65535, 65535, 65535, 65535, 65535, 3920, 65535, 65535, 65535, 3921, 65535, 3922, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3923, 65535, 65535, 65535, 65535, 65535, 3924, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3925, 65535, 65535, 65535, 65535, 65535, 3926, 65535, 65535, 65535, 3927, 65535, 3928, 65535, 65535, 65535, 3929, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3930, 65535, 65535, 65535, 65535, 65535, 3931, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3932, 65535, 3933, 65535, 65535, 65535, 3934, 65535, 65535, 65535, 65535, 65535, 3935, 65535, 3936, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3937, 65535, 65535, 65535, 65535, 65535, 3938, 65535, 65535, 65535, 3939, 65535, 65535, 65535, 65535, 65535, 3940, 65535, 3941, 65535, 65535, 65535, 65535, 65535, 3942, 65535, 65535, 65535, 3943, 65535, 3944, 65535, 65535, 65535, 3945, 65535, 3946, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3947, 65535, 3948, 65535, 65535, 65535, 3949, 65535, 3950, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3951, 65535, 3952, 65535, 65535, 65535, 65535, 65535, 3953, 65535, 65535, 65535, 3954, 65535, 3955, 65535, 65535, 65535, 3956, 65535, 65535, 65535, 65535, 65535, 3957, 65535, 65535, 65535, 65535, 65535, 3958, 65535, 3959, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3960, 65535, 3961, 65535, 65535, 65535, 65535, 65535, 3962, 65535, 65535, 65535, 3963, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3964, 65535, 65535, 65535, 3965, 65535, 65535, 65535, 65535, 65535, 3966, 65535, 3967, 65535, 65535, 65535, 3968, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3969, 65535, 65535, 65535, 65535, 65535, 3970, 65535, 65535, 65535, 3971, 65535, 65535, 65535, 65535, 65535, 3972, 65535, 3973, 65535, 65535, 65535, 3974, 65535, 65535, 65535, 65535, 65535, 3975, 65535, 3976, 65535, 65535, 65535, 65535, 65535, 3977, 65535, 65535, 65535, 65535, 65535, 3978, 65535, 65535, 65535, 3979, 65535, 3980, 65535, 65535, 65535, 3981, 65535, 65535, 65535, 65535, 65535, 3982, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3983, 65535, 65535, 65535, 3984, 65535, 3985, 65535, 65535, 65535, 3986, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3987, 65535, 3988, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3989, 65535, 65535, 65535, 3990, 65535, 3991, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 3992, 65535, 3993, 65535, 65535, 65535, 65535, 65535, 3994, 65535, 65535, 65535, 3995, 65535, 3996, 65535, 65535, 65535, 3997, 65535, 65535, 65535, 65535, 65535, 3998, 65535, 65535, 65535, 65535, 65535, 3999, 65535, 4000, 65535, 65535, 65535, 65535, 65535, 4001, 65535, 65535, 65535, 4002, 65535, 4003, 65535, 65535, 65535, 65535, 65535, 4004, 65535, 65535, 65535, 4005, 65535, 65535, 65535, 65535, 65535, 4006, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4007, 65535, 65535, 65535, 65535, 65535, 4008, 65535, 65535, 65535, 4009, 65535, 4010, 65535, 65535, 65535, 4011, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4012, 65535, 65535, 65535, 65535, 65535, 4013, 65535, 4014, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4015, 65535, 4016, 65535, 65535, 65535, 65535, 65535, 4017, 65535, 65535, 65535, 65535, 65535, 4018, 65535, 65535, 65535, 4019, 65535, 4020, 65535, 65535, 65535, 4021, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4022, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4023, 65535, 4024, 65535, 65535, 65535, 4025, 65535, 4026, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4027, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4028, 65535, 4029, 65535, 65535, 65535, 4030, 65535, 4031, 65535, 65535, 65535, 4032, 65535, 65535, 65535, 65535, 65535, 4033, 65535, 4034, 65535, 65535, 65535, 65535, 65535, 4035, 65535, 65535, 65535, 4036, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4037, 65535, 65535, 65535, 65535, 65535, 4038, 65535, 4039, 65535, 65535, 65535, 65535, 65535, 4040, 65535, 65535, 65535, 4041, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4042, 65535, 65535, 65535, 4043, 65535, 65535, 65535, 65535, 65535, 4044, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4045, 65535, 65535, 65535, 4046, 65535, 4047, 65535, 65535, 65535, 4048, 65535, 65535, 65535, 65535, 65535, 4049, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4050, 65535, 65535, 65535, 65535, 65535, 4051, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4052, 65535, 4053, 65535, 65535, 65535, 4054, 65535, 65535, 65535, 65535, 65535, 4055, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4056, 65535, 65535, 65535, 65535, 65535, 4057, 65535, 65535, 65535, 4058, 65535, 4059, 65535, 65535, 65535, 4060, 65535, 65535, 65535, 65535, 65535, 4061, 65535, 4062, 65535, 65535, 65535, 65535, 65535, 4063, 65535, 65535, 65535, 4064, 65535, 4065, 65535, 65535, 65535, 65535, 65535, 4066, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4067, 65535, 4068, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4069, 65535, 4070, 65535, 65535, 65535, 4071, 65535, 65535, 65535, 65535, 65535, 4072, 65535, 65535, 65535, 65535, 65535, 4073, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4074, 65535, 65535, 65535, 4075, 65535, 4076, 65535, 65535, 65535, 4077, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4078, 65535, 4079, 65535, 65535, 65535, 65535, 65535, 4080, 65535, 65535, 65535, 65535, 65535, 4081, 65535, 65535, 65535, 65535, 65535, 4082, 65535, 65535, 65535, 4083, 65535, 65535, 65535, 65535, 65535, 4084, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4085, 65535, 4086, 65535, 65535, 65535, 4087, 65535, 4088, 65535, 65535, 65535, 4089, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4090, 65535, 65535, 65535, 65535, 65535, 4091, 65535, 65535, 65535, 4092, 65535, 65535, 65535, 65535, 65535, 4093, 65535, 4094, 65535, 65535, 65535, 4095, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4096, 65535, 65535, 65535, 65535, 65535, 4097, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4098, 65535, 4099, 65535, 65535, 65535, 4100, 65535, 65535, 65535, 65535, 65535, 4101, 65535, 4102, 65535, 65535, 65535, 65535, 65535, 4103, 65535, 65535, 65535, 4104, 65535, 4105, 65535, 65535, 65535, 4106, 65535, 4107, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4108, 65535, 4109, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4110, 65535, 4111, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4112, 65535, 65535, 65535, 65535, 65535, 4113, 65535, 4114, 65535, 65535, 65535, 65535, 65535, 4115, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4116, 65535, 65535, 65535, 65535, 65535, 4117, 65535, 65535, 65535, 65535, 65535, 4118, 65535, 4119, 65535, 65535, 65535, 65535, 65535, 4120, 65535, 65535, 65535, 4121, 65535, 4122, 65535, 65535, 65535, 65535, 65535, 4123, 65535, 65535, 65535, 4124, 65535, 65535, 65535, 65535, 65535, 4125, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4126, 65535, 65535, 65535, 4127, 65535, 4128, 65535, 65535, 65535, 65535, 65535, 4129, 65535, 65535, 65535, 4130, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4131, 65535, 65535, 65535, 4132, 65535, 65535, 65535, 65535, 65535, 4133, 65535, 4134, 65535, 65535, 65535, 4135, 65535, 65535, 65535, 65535, 65535, 4136, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4137, 65535, 65535, 65535, 65535, 65535, 4138, 65535, 65535, 65535, 4139, 65535, 4140, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4141, 65535, 4142, 65535, 65535, 65535, 65535, 65535, 4143, 65535, 65535, 65535, 4144, 65535, 4145, 65535, 65535, 65535, 4146, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4147, 65535, 4148, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4149, 65535, 4150, 65535, 65535, 65535, 4151, 65535, 4152, 65535, 65535, 65535, 4153, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4154, 65535, 65535, 65535, 65535, 65535, 4155, 65535, 65535, 65535, 65535, 65535, 4156, 65535, 65535, 65535, 4157, 65535, 65535, 65535, 65535, 65535, 4158, 65535, 65535, 65535, 65535, 65535, 4159, 65535, 4160, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4161, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4162, 65535, 65535, 65535, 4163, 65535, 65535, 65535, 65535, 65535, 4164, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4165, 65535, 65535, 65535, 4166, 65535, 4167, 65535, 65535, 65535, 4168, 65535, 4169, 65535, 65535, 65535, 4170, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4171, 65535, 65535, 65535, 65535, 65535, 4172, 65535, 65535, 65535, 4173, 65535, 65535, 65535, 65535, 65535, 4174, 65535, 65535, 65535, 65535, 65535, 4175, 65535, 65535, 65535, 65535, 65535, 4176, 65535, 4177, 65535, 65535, 65535, 65535, 65535, 4178, 65535, 65535, 65535, 65535, 65535, 4179, 65535, 65535, 65535, 4180, 65535, 4181, 65535, 65535, 65535, 4182, 65535, 65535, 65535, 65535, 65535, 4183, 65535, 4184, 65535, 65535, 65535, 65535, 65535, 4185, 65535, 65535, 65535, 4186, 65535, 4187, 65535, 65535, 65535, 65535, 65535, 4188, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4189, 65535, 4190, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4191, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4192, 65535, 65535, 65535, 4193, 65535, 65535, 65535, 65535, 65535, 4194, 65535, 4195, 65535, 65535, 65535, 65535, 65535, 4196, 65535, 65535, 65535, 4197, 65535, 4198, 65535, 65535, 65535, 4199, 65535, 65535, 65535, 65535, 65535, 4200, 65535, 65535, 65535, 65535, 65535, 4201, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4202, 65535, 65535, 65535, 4203, 65535, 4204, 65535, 65535, 65535, 65535, 65535, 4205, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4206, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4207, 65535, 65535, 65535, 4208, 65535, 4209, 65535, 65535, 65535, 4210, 65535, 4211, 65535, 65535, 65535, 4212, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4213, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4214, 65535, 65535, 65535, 65535, 65535, 4215, 65535, 4216, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4217, 65535, 65535, 65535, 65535, 65535, 4218, 65535, 65535, 65535, 65535, 65535, 4219, 65535, 65535, 65535, 4220, 65535, 65535, 65535, 65535, 65535, 4221, 65535, 65535, 65535, 65535, 65535, 4222, 65535, 4223, 65535, 65535, 65535, 65535, 65535, 4224, 65535, 65535, 65535, 4225, 65535, 65535, 65535, 65535, 65535, 4226, 65535, 4227, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4228, 65535, 4229, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4230, 65535, 65535, 65535, 4231, 65535, 4232, 65535, 65535, 65535, 4233, 65535, 65535, 65535, 65535, 65535, 4234, 65535, 4235, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4236, 65535, 4237, 65535, 65535, 65535, 4238, 65535, 65535, 65535, 65535, 65535, 4239, 65535, 65535, 65535, 65535, 65535, 4240, 65535, 4241, 65535, 65535, 65535, 65535, 65535, 4242, 65535, 65535, 65535, 65535, 65535, 4243, 65535, 65535, 65535, 65535, 65535, 4244, 65535, 65535, 65535, 4245, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4246, 65535, 65535, 65535, 4247, 65535, 4248, 65535, 65535, 65535, 4249, 65535, 4250, 65535, 65535, 65535, 4251, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4252, 65535, 65535, 65535, 65535, 65535, 4253, 65535, 65535, 65535, 4254, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4255, 65535, 65535, 65535, 4256, 65535, 65535, 65535, 65535, 65535, 4257, 65535, 4258, 65535, 65535, 65535, 65535, 65535, 4259, 65535, 65535, 65535, 65535, 65535, 4260, 65535, 65535, 65535, 65535, 65535, 4261, 65535, 65535, 65535, 4262, 65535, 65535, 65535, 65535, 65535, 4263, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4264, 65535, 65535, 65535, 4265, 65535, 65535, 65535, 65535, 65535, 4266, 65535, 4267, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4268, 65535, 4269, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4270, 65535, 4271, 65535, 65535, 65535, 4272, 65535, 4273, 65535, 65535, 65535, 4274, 65535, 65535, 65535, 65535, 65535, 4275, 65535, 4276, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4277, 65535, 4278, 65535, 65535, 65535, 4279, 65535, 65535, 65535, 65535, 65535, 4280, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4281, 65535, 65535, 65535, 4282, 65535, 4283, 65535, 65535, 65535, 65535, 65535, 4284, 65535, 65535, 65535, 4285, 65535, 65535, 65535, 65535, 65535, 4286, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4287, 65535, 65535, 65535, 4288, 65535, 65535, 65535, 65535, 65535, 4289, 65535, 4290, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4291, 65535, 65535, 65535, 65535, 65535, 4292, 65535, 65535, 65535, 4293, 65535, 65535, 65535, 65535, 65535, 4294, 65535, 65535, 65535, 65535, 65535, 4295, 65535, 65535, 65535, 65535, 65535, 4296, 65535, 4297, 65535, 65535, 65535, 65535, 65535, 4298, 65535, 65535, 65535, 65535, 65535, 4299, 65535, 65535, 65535, 4300, 65535, 65535, 65535, 65535, 65535, 4301, 65535, 65535, 65535, 65535, 65535, 4302, 65535, 4303, 65535, 65535, 65535, 65535, 65535, 4304, 65535, 65535, 65535, 65535, 65535, 4305, 65535, 65535, 65535, 4306, 65535, 4307, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4308, 65535, 4309, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4310, 65535, 4311, 65535, 65535, 65535, 4312, 65535, 4313, 65535, 65535, 65535, 4314, 65535, 65535, 65535, 65535, 65535, 4315, 65535, 4316, 65535, 65535, 65535, 65535, 65535, 4317, 65535, 65535, 65535, 4318, 65535, 4319, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4320, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4321, 65535, 65535, 65535, 65535, 65535, 4322, 65535, 65535, 65535, 4323, 65535, 4324, 65535, 65535, 65535, 65535, 65535, 4325, 65535, 65535, 65535, 4326, 65535, 65535, 65535, 65535, 65535, 4327, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4328, 65535, 65535, 65535, 4329, 65535, 4330, 65535, 65535, 65535, 4331, 65535, 4332, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4333, 65535, 65535, 65535, 65535, 65535, 4334, 65535, 65535, 65535, 4335, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4336, 65535, 65535, 65535, 4337, 65535, 65535, 65535, 65535, 65535, 4338, 65535, 4339, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4340, 65535, 4341, 65535, 65535, 65535, 4342, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4343, 65535, 65535, 65535, 65535, 65535, 4344, 65535, 65535, 65535, 4345, 65535, 4346, 65535, 65535, 65535, 4347, 65535, 4348, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4349, 65535, 4350, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4351, 65535, 65535, 65535, 65535, 65535, 4352, 65535, 4353, 65535, 65535, 65535, 4354, 65535, 65535, 65535, 65535, 65535, 4355, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4356, 65535, 65535, 65535, 4357, 65535, 4358, 65535, 65535, 65535, 4359, 65535, 65535, 65535, 65535, 65535, 4360, 65535, 65535, 65535, 65535, 65535, 4361, 65535, 4362, 65535, 65535, 65535, 65535, 65535, 4363, 65535, 65535, 65535, 4364, 65535, 4365, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4366, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4367, 65535, 65535, 65535, 65535, 65535, 4368, 65535, 65535, 65535, 4369, 65535, 65535, 65535, 65535, 65535, 4370, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4371, 65535, 65535, 65535, 65535, 65535, 4372, 65535, 65535, 65535, 4373, 65535, 65535, 65535, 65535, 65535, 4374, 65535, 4375, 65535, 65535, 65535, 4376, 65535, 65535, 65535, 65535, 65535, 4377, 65535, 4378, 65535, 65535, 65535, 65535, 65535, 4379, 65535, 65535, 65535, 65535, 65535, 4380, 65535, 65535, 65535, 65535, 65535, 4381, 65535, 65535, 65535, 4382, 65535, 65535, 65535, 65535, 65535, 4383, 65535, 4384, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4385, 65535, 4386, 65535, 65535, 65535, 4387, 65535, 4388, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4389, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4390, 65535, 4391, 65535, 65535, 65535, 4392, 65535, 4393, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4394, 65535, 65535, 65535, 65535, 65535, 4395, 65535, 65535, 65535, 4396, 65535, 4397, 65535, 65535, 65535, 4398, 65535, 65535, 65535, 65535, 65535, 4399, 65535, 65535, 65535, 65535, 65535, 4400, 65535, 4401, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4402, 65535, 4403, 65535, 65535, 65535, 65535, 65535, 4404, 65535, 65535, 65535, 4405, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4406, 65535, 65535, 65535, 4407, 65535, 4408, 65535, 65535, 65535, 65535, 65535, 4409, 65535, 65535, 65535, 4410, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4411, 65535, 65535, 65535, 65535, 65535, 4412, 65535, 65535, 65535, 4413, 65535, 65535, 65535, 65535, 65535, 4414, 65535, 4415, 65535, 65535, 65535, 4416, 65535, 65535, 65535, 65535, 65535, 4417, 65535, 4418, 65535, 65535, 65535, 65535, 65535, 4419, 65535, 65535, 65535, 65535, 65535, 4420, 65535, 65535, 65535, 4421, 65535, 4422, 65535, 65535, 65535, 4423, 65535, 65535, 65535, 65535, 65535, 4424, 65535, 4425, 65535, 65535, 65535, 65535, 65535, 4426, 65535, 65535, 65535, 4427, 65535, 4428, 65535, 65535, 65535, 4429, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4430, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4431, 65535, 65535, 65535, 4432, 65535, 4433, 65535, 65535, 65535, 4434, 65535, 65535, 65535, 65535, 65535, 4435, 65535, 4436, 65535, 65535, 65535, 65535, 65535, 4437, 65535, 65535, 65535, 4438, 65535, 4439, 65535, 65535, 65535, 4440, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4441, 65535, 4442, 65535, 65535, 65535, 65535, 65535, 4443, 65535, 65535, 65535, 4444, 65535, 4445, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4446, 65535, 65535, 65535, 65535, 65535, 4447, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4448, 65535, 65535, 65535, 4449, 65535, 4450, 65535, 65535, 65535, 4451, 65535, 4452, 65535, 65535, 65535, 4453, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4454, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4455, 65535, 4456, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4457, 65535, 4458, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4459, 65535, 65535, 65535, 4460, 65535, 4461, 65535, 65535, 65535, 4462, 65535, 65535, 65535, 65535, 65535, 4463, 65535, 4464, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4465, 65535, 4466, 65535, 65535, 65535, 4467, 65535, 4468, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4469, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4470, 65535, 4471, 65535, 65535, 65535, 4472, 65535, 4473, 65535, 65535, 65535, 4474, 65535, 65535, 65535, 65535, 65535, 4475, 65535, 4476, 65535, 65535, 65535, 65535, 65535, 4477, 65535, 65535, 65535, 4478, 65535, 65535, 65535, 65535, 65535, 4479, 65535, 65535, 65535, 65535, 65535, 4480, 65535, 65535, 65535, 65535, 65535, 4481, 65535, 4482, 65535, 65535, 65535, 65535, 65535, 4483, 65535, 65535, 65535, 4484, 65535, 4485, 65535, 65535, 65535, 65535, 65535, 4486, 65535, 65535, 65535, 4487, 65535, 65535, 65535, 65535, 65535, 4488, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4489, 65535, 65535, 65535, 4490, 65535, 65535, 65535, 65535, 65535, 4491, 65535, 65535, 65535, 65535, 65535, 4492, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4493, 65535, 65535, 65535, 65535, 65535, 4494, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4495, 65535, 4496, 65535, 65535, 65535, 4497, 65535, 65535, 65535, 65535, 65535, 4498, 65535, 4499, 65535, 65535, 65535, 65535, 65535, 4500, 65535, 65535, 65535, 65535, 65535, 4501, 65535, 65535, 65535, 4502, 65535, 4503, 65535, 65535, 65535, 4504, 65535, 65535, 65535, 65535, 65535, 4505, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4506, 65535, 65535, 65535, 4507, 65535, 4508, 65535, 65535, 65535, 65535, 65535, 4509, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4510, 65535, 4511, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4512, 65535, 4513, 65535, 65535, 65535, 4514, 65535, 4515, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4516, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4517, 65535, 65535, 65535, 4518, 65535, 4519, 65535, 65535, 65535, 4520, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4521, 65535, 4522, 65535, 65535, 65535, 65535, 65535, 4523, 65535, 65535, 65535, 4524, 65535, 4525, 65535, 65535, 65535, 65535, 65535, 4526, 65535, 65535, 65535, 4527, 65535, 65535, 65535, 65535, 65535, 4528, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4529, 65535, 65535, 65535, 4530, 65535, 4531, 65535, 65535, 65535, 4532, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4533, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4534, 65535, 65535, 65535, 65535, 65535, 4535, 65535, 4536, 65535, 65535, 65535, 4537, 65535, 65535, 65535, 65535, 65535, 4538, 65535, 4539, 65535, 65535, 65535, 65535, 65535, 4540, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4541, 65535, 4542, 65535, 65535, 65535, 4543, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4544, 65535, 65535, 65535, 65535, 65535, 4545, 65535, 65535, 65535, 4546, 65535, 4547, 65535, 65535, 65535, 4548, 65535, 4549, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4550, 65535, 4551, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4552, 65535, 4553, 65535, 65535, 65535, 4554, 65535, 65535, 65535, 65535, 65535, 4555, 65535, 65535, 65535, 65535, 65535, 4556, 65535, 4557, 65535, 65535, 65535, 65535, 65535, 4558, 65535, 65535, 65535, 65535, 65535, 4559, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4560, 65535, 65535, 65535, 65535, 65535, 4561, 65535, 4562, 65535, 65535, 65535, 65535, 65535, 4563, 65535, 65535, 65535, 4564, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4565, 65535, 65535, 65535, 4566, 65535, 65535, 65535, 65535, 65535, 4567, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4568, 65535, 65535, 65535, 4569, 65535, 4570, 65535, 65535, 65535, 65535, 65535, 4571, 65535, 65535, 65535, 4572, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4573, 65535, 65535, 65535, 65535, 65535, 4574, 65535, 65535, 65535, 4575, 65535, 65535, 65535, 65535, 65535, 4576, 65535, 4577, 65535, 65535, 65535, 4578, 65535, 65535, 65535, 65535, 65535, 4579, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4580, 65535, 65535, 65535, 65535, 65535, 4581, 65535, 65535, 65535, 4582, 65535, 4583, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4584, 65535, 4585, 65535, 65535, 65535, 65535, 65535, 4586, 65535, 65535, 65535, 4587, 65535, 4588, 65535, 65535, 65535, 4589, 65535, 4590, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4591, 65535, 4592, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4593, 65535, 4594, 65535, 65535, 65535, 4595, 65535, 65535, 65535, 65535, 65535, 4596, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4597, 65535, 65535, 65535, 65535, 65535, 4598, 65535, 65535, 65535, 4599, 65535, 4600, 65535, 65535, 65535, 4601, 65535, 65535, 65535, 65535, 65535, 4602, 65535, 65535, 65535, 65535, 65535, 4603, 65535, 4604, 65535, 65535, 65535, 65535, 65535, 4605, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4606, 65535, 65535, 65535, 4607, 65535, 65535, 65535, 65535, 65535, 4608, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4609, 65535, 4610, 65535, 65535, 65535, 4611, 65535, 4612, 65535, 65535, 65535, 4613, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4614, 65535, 65535, 65535, 65535, 65535, 4615, 65535, 65535, 65535, 4616, 65535, 65535, 65535, 65535, 65535, 4617, 65535, 65535, 65535, 65535, 65535, 4618, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4619, 65535, 65535, 65535, 65535, 65535, 4620, 65535, 65535, 65535, 65535, 65535, 4621, 65535, 65535, 65535, 4622, 65535, 4623, 65535, 65535, 65535, 4624, 65535, 65535, 65535, 65535, 65535, 4625, 65535, 4626, 65535, 65535, 65535, 65535, 65535, 4627, 65535, 65535, 65535, 4628, 65535, 4629, 65535, 65535, 65535, 4630, 65535, 4631, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4632, 65535, 4633, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4634, 65535, 4635, 65535, 65535, 65535, 65535, 65535, 4636, 65535, 65535, 65535, 4637, 65535, 65535, 65535, 65535, 65535, 4638, 65535, 4639, 65535, 65535, 65535, 65535, 65535, 4640, 65535, 65535, 65535, 4641, 65535, 65535, 65535, 65535, 65535, 4642, 65535, 65535, 65535, 65535, 65535, 4643, 65535, 65535, 65535, 65535, 65535, 4644, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4645, 65535, 65535, 65535, 4646, 65535, 4647, 65535, 65535, 65535, 65535, 65535, 4648, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4649, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4650, 65535, 65535, 65535, 4651, 65535, 4652, 65535, 65535, 65535, 4653, 65535, 4654, 65535, 65535, 65535, 4655, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4656, 65535, 65535, 65535, 65535, 65535, 4657, 65535, 4658, 65535, 65535, 65535, 4659, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4660, 65535, 65535, 65535, 65535, 65535, 4661, 65535, 65535, 65535, 65535, 65535, 4662, 65535, 65535, 65535, 4663, 65535, 4664, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4665, 65535, 4666, 65535, 65535, 65535, 65535, 65535, 4667, 65535, 65535, 65535, 4668, 65535, 65535, 65535, 65535, 65535, 4669, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4670, 65535, 4671, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4672, 65535, 4673, 65535, 65535, 65535, 4674, 65535, 4675, 65535, 65535, 65535, 4676, 65535, 65535, 65535, 65535, 65535, 4677, 65535, 4678, 65535, 65535, 65535, 65535, 65535, 4679, 65535, 65535, 65535, 65535, 65535, 4680, 65535, 65535, 65535, 4681, 65535, 65535, 65535, 65535, 65535, 4682, 65535, 65535, 65535, 65535, 65535, 4683, 65535, 4684, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4685, 65535, 65535, 65535, 65535, 65535, 4686, 65535, 65535, 65535, 4687, 65535, 65535, 65535, 65535, 65535, 4688, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4689, 65535, 65535, 65535, 4690, 65535, 4691, 65535, 65535, 65535, 4692, 65535, 4693, 65535, 65535, 65535, 4694, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4695, 65535, 65535, 65535, 65535, 65535, 4696, 65535, 65535, 65535, 4697, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4698, 65535, 65535, 65535, 4699, 65535, 65535, 65535, 65535, 65535, 4700, 65535, 4701, 65535, 65535, 65535, 65535, 65535, 4702, 65535, 65535, 65535, 65535, 65535, 4703, 65535, 65535, 65535, 4704, 65535, 4705, 65535, 65535, 65535, 4706, 65535, 65535, 65535, 65535, 65535, 4707, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4708, 65535, 65535, 65535, 4709, 65535, 4710, 65535, 65535, 65535, 65535, 65535, 4711, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4712, 65535, 4713, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4714, 65535, 65535, 65535, 65535, 65535, 4715, 65535, 4716, 65535, 65535, 65535, 4717, 65535, 65535, 65535, 65535, 65535, 4718, 65535, 4719, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4720, 65535, 4721, 65535, 65535, 65535, 4722, 65535, 65535, 65535, 65535, 65535, 4723, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4724, 65535, 65535, 65535, 65535, 65535, 4725, 65535, 65535, 65535, 4726, 65535, 4727, 65535, 65535, 65535, 65535, 65535, 4728, 65535, 65535, 65535, 4729, 65535, 65535, 65535, 65535, 65535, 4730, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4731, 65535, 65535, 65535, 4732, 65535, 65535, 65535, 65535, 65535, 4733, 65535, 4734, 65535, 65535, 65535, 4735, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4736, 65535, 65535, 65535, 65535, 65535, 4737, 65535, 65535, 65535, 4738, 65535, 65535, 65535, 65535, 65535, 4739, 65535, 4740, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4741, 65535, 4742, 65535, 65535, 65535, 65535, 65535, 4743, 65535, 65535, 65535, 65535, 65535, 4744, 65535, 65535, 65535, 4745, 65535, 65535, 65535, 65535, 65535, 4746, 65535, 65535, 65535, 65535, 65535, 4747, 65535, 4748, 65535, 65535, 65535, 65535, 65535, 4749, 65535, 65535, 65535, 65535, 65535, 4750, 65535, 65535, 65535, 4751, 65535, 4752, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4753, 65535, 4754, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4755, 65535, 65535, 65535, 4756, 65535, 4757, 65535, 65535, 65535, 4758, 65535, 65535, 65535, 65535, 65535, 4759, 65535, 4760, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4761, 65535, 4762, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4763, 65535, 65535, 65535, 65535, 65535, 4764, 65535, 4765, 65535, 65535, 65535, 65535, 65535, 4766, 65535, 65535, 65535, 4767, 65535, 4768, 65535, 65535, 65535, 65535, 65535, 4769, 65535, 65535, 65535, 4770, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4771, 65535, 65535, 65535, 4772, 65535, 4773, 65535, 65535, 65535, 4774, 65535, 4775, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4776, 65535, 65535, 65535, 65535, 65535, 4777, 65535, 65535, 65535, 4778, 65535, 65535, 65535, 65535, 65535, 4779, 65535, 4780, 65535, 65535, 65535, 4781, 65535, 65535, 65535, 65535, 65535, 4782, 65535, 4783, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4784, 65535, 65535, 65535, 65535, 65535, 4785, 65535, 65535, 65535, 4786, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4787, 65535, 65535, 65535, 65535, 65535, 4788, 65535, 65535, 65535, 4789, 65535, 65535, 65535, 65535, 65535, 4790, 65535, 4791, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4792, 65535, 4793, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4794, 65535, 65535, 65535, 65535, 65535, 4795, 65535, 4796, 65535, 65535, 65535, 4797, 65535, 65535, 65535, 65535, 65535, 4798, 65535, 4799, 65535, 65535, 65535, 65535, 65535, 4800, 65535, 65535, 65535, 4801, 65535, 4802, 65535, 65535, 65535, 4803, 65535, 65535, 65535, 65535, 65535, 4804, 65535, 65535, 65535, 65535, 65535, 4805, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4806, 65535, 65535, 65535, 4807, 65535, 4808, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4809, 65535, 65535, 65535, 65535, 65535, 4810, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4811, 65535, 65535, 65535, 65535, 65535, 4812, 65535, 65535, 65535, 4813, 65535, 4814, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4815, 65535, 65535, 65535, 65535, 65535, 4816, 65535, 65535, 65535, 4817, 65535, 65535, 65535, 65535, 65535, 4818, 65535, 65535, 65535, 65535, 65535, 4819, 65535, 65535, 65535, 65535, 65535, 4820, 65535, 4821, 65535, 65535, 65535, 65535, 65535, 4822, 65535, 65535, 65535, 65535, 65535, 4823, 65535, 65535, 65535, 65535, 65535, 4824, 65535, 65535, 65535, 4825, 65535, 65535, 65535, 65535, 65535, 4826, 65535, 4827, 65535, 65535, 65535, 65535, 65535, 4828, 65535, 65535, 65535, 65535, 65535, 4829, 65535, 65535, 65535, 4830, 65535, 4831, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4832, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4833, 65535, 4834, 65535, 65535, 65535, 4835, 65535, 4836, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4837, 65535, 4838, 65535, 65535, 65535, 65535, 65535, 4839, 65535, 65535, 65535, 4840, 65535, 4841, 65535, 65535, 65535, 4842, 65535, 65535, 65535, 65535, 65535, 4843, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4844, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4845, 65535, 4846, 65535, 65535, 65535, 65535, 65535, 4847, 65535, 65535, 65535, 4848, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4849, 65535, 65535, 65535, 4850, 65535, 4851, 65535, 65535, 65535, 4852, 65535, 4853, 65535, 65535, 65535, 4854, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4855, 65535, 65535, 65535, 65535, 65535, 4856, 65535, 65535, 65535, 4857, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4858, 65535, 65535, 65535, 4859, 65535, 65535, 65535, 65535, 65535, 4860, 65535, 4861, 65535, 65535, 65535, 65535, 65535, 4862, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4863, 65535, 4864, 65535, 65535, 65535, 4865, 65535, 65535, 65535, 65535, 65535, 4866, 65535, 4867, 65535, 65535, 65535, 65535, 65535, 4868, 65535, 65535, 65535, 4869, 65535, 4870, 65535, 65535, 65535, 4871, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4872, 65535, 4873, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4874, 65535, 65535, 65535, 4875, 65535, 4876, 65535, 65535, 65535, 4877, 65535, 65535, 65535, 65535, 65535, 4878, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4879, 65535, 65535, 65535, 4880, 65535, 4881, 65535, 65535, 65535, 4882, 65535, 65535, 65535, 65535, 65535, 4883, 65535, 65535, 65535, 65535, 65535, 4884, 65535, 4885, 65535, 65535, 65535, 65535, 65535, 4886, 65535, 65535, 65535, 4887, 65535, 4888, 65535, 65535, 65535, 65535, 65535, 4889, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4890, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4891, 65535, 65535, 65535, 4892, 65535, 4893, 65535, 65535, 65535, 4894, 65535, 65535, 65535, 65535, 65535, 4895, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4896, 65535, 65535, 65535, 4897, 65535, 65535, 65535, 65535, 65535, 4898, 65535, 4899, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4900, 65535, 4901, 65535, 65535, 65535, 65535, 65535, 4902, 65535, 65535, 65535, 65535, 65535, 4903, 65535, 65535, 65535, 4904, 65535, 4905, 65535, 65535, 65535, 4906, 65535, 65535, 65535, 65535, 65535, 4907, 65535, 4908, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4909, 65535, 4910, 65535, 65535, 65535, 4911, 65535, 4912, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4913, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4914, 65535, 4915, 65535, 65535, 65535, 4916, 65535, 4917, 65535, 65535, 65535, 4918, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4919, 65535, 65535, 65535, 65535, 65535, 4920, 65535, 65535, 65535, 4921, 65535, 65535, 65535, 65535, 65535, 4922, 65535, 65535, 65535, 65535, 65535, 4923, 65535, 65535, 65535, 65535, 65535, 4924, 65535, 4925, 65535, 65535, 65535, 65535, 65535, 4926, 65535, 65535, 65535, 4927, 65535, 4928, 65535, 65535, 65535, 65535, 65535, 4929, 65535, 65535, 65535, 4930, 65535, 65535, 65535, 65535, 65535, 4931, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4932, 65535, 65535, 65535, 4933, 65535, 4934, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4935, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4936, 65535, 65535, 65535, 65535, 65535, 4937, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4938, 65535, 4939, 65535, 65535, 65535, 4940, 65535, 65535, 65535, 65535, 65535, 4941, 65535, 4942, 65535, 65535, 65535, 65535, 65535, 4943, 65535, 65535, 65535, 65535, 65535, 4944, 65535, 65535, 65535, 4945, 65535, 4946, 65535, 65535, 65535, 4947, 65535, 65535, 65535, 65535, 65535, 4948, 65535, 4949, 65535, 65535, 65535, 65535, 65535, 4950, 65535, 65535, 65535, 4951, 65535, 4952, 65535, 65535, 65535, 65535, 65535, 4953, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4954, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4955, 65535, 4956, 65535, 65535, 65535, 4957, 65535, 4958, 65535, 65535, 65535, 4959, 65535, 65535, 65535, 65535, 65535, 4960, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4961, 65535, 65535, 65535, 4962, 65535, 4963, 65535, 65535, 65535, 4964, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4965, 65535, 4966, 65535, 65535, 65535, 65535, 65535, 4967, 65535, 65535, 65535, 4968, 65535, 4969, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4970, 65535, 65535, 65535, 65535, 65535, 4971, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4972, 65535, 4973, 65535, 65535, 65535, 4974, 65535, 4975, 65535, 65535, 65535, 4976, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4977, 65535, 65535, 65535, 65535, 65535, 4978, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4979, 65535, 4980, 65535, 65535, 65535, 4981, 65535, 65535, 65535, 65535, 65535, 4982, 65535, 4983, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4984, 65535, 4985, 65535, 65535, 65535, 4986, 65535, 65535, 65535, 65535, 65535, 4987, 65535, 4988, 65535, 65535, 65535, 65535, 65535, 4989, 65535, 65535, 65535, 4990, 65535, 4991, 65535, 65535, 65535, 4992, 65535, 4993, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4994, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 4995, 65535, 4996, 65535, 65535, 65535, 4997, 65535, 65535, 65535, 65535, 65535, 4998, 65535, 65535, 65535, 65535, 65535, 4999, 65535, 5000, 65535, 65535, 65535, 65535, 65535, 5001, 65535, 65535, 65535, 65535, 65535, 5002, 65535, 65535, 65535, 5003, 65535, 65535, 65535, 65535, 65535, 5004, 65535, 65535, 65535, 65535, 65535, 5005, 65535, 5006, 65535, 65535, 65535, 65535, 65535, 5007, 65535, 65535, 65535, 5008, 65535, 5009, 65535, 65535, 65535, 65535, 65535, 5010, 65535, 65535, 65535, 5011, 65535, 65535, 65535, 65535, 65535, 5012, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5013, 65535, 65535, 65535, 5014, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5015, 65535, 65535, 65535, 5016, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5017, 65535, 65535, 65535, 65535, 65535, 5018, 65535, 65535, 65535, 5019, 65535, 65535, 65535, 65535, 65535, 5020, 65535, 5021, 65535, 65535, 65535, 5022, 65535, 65535, 65535, 65535, 65535, 5023, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5024, 65535, 65535, 65535, 65535, 65535, 5025, 65535, 65535, 65535, 5026, 65535, 5027, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5028, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5029, 65535, 65535, 65535, 5030, 65535, 5031, 65535, 65535, 65535, 5032, 65535, 5033, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5034, 65535, 5035, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5036, 65535, 5037, 65535, 65535, 65535, 5038, 65535, 5039, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5040, 65535, 65535, 65535, 65535, 65535, 5041, 65535, 65535, 65535, 5042, 65535, 5043, 65535, 65535, 65535, 5044, 65535, 65535, 65535, 65535, 65535, 5045, 65535, 65535, 65535, 65535, 65535, 5046, 65535, 5047, 65535, 65535, 65535, 65535, 65535, 5048, 65535, 65535, 65535, 5049, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5050, 65535, 65535, 65535, 5051, 65535, 65535, 65535, 65535, 65535, 5052, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5053, 65535, 65535, 65535, 65535, 65535, 5054, 65535, 65535, 65535, 5055, 65535, 5056, 65535, 65535, 65535, 5057, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5058, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5059, 65535, 65535, 65535, 65535, 65535, 5060, 65535, 65535, 65535, 65535, 65535, 5061, 65535, 65535, 65535, 65535, 65535, 5062, 65535, 5063, 65535, 65535, 65535, 65535, 65535, 5064, 65535, 65535, 65535, 65535, 65535, 5065, 65535, 65535, 65535, 5066, 65535, 5067, 65535, 65535, 65535, 5068, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5069, 65535, 65535, 65535, 65535, 65535, 5070, 65535, 65535, 65535, 5071, 65535, 5072, 65535, 65535, 65535, 5073, 65535, 5074, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5075, 65535, 5076, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5077, 65535, 5078, 65535, 65535, 65535, 65535, 65535, 5079, 65535, 65535, 65535, 5080, 65535, 65535, 65535, 65535, 65535, 5081, 65535, 5082, 65535, 65535, 65535, 65535, 65535, 5083, 65535, 65535, 65535, 5084, 65535, 5085, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5086, 65535, 65535, 65535, 65535, 65535, 5087, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5088, 65535, 65535, 65535, 5089, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5090, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5091, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5092, 65535, 65535, 65535, 5093, 65535, 5094, 65535, 65535, 65535, 5095, 65535, 5096, 65535, 65535, 65535, 5097, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5098, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5099, 65535, 65535, 65535, 65535, 65535, 5100, 65535, 5101, 65535, 65535, 65535, 5102, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5103, 65535, 65535, 65535, 65535, 65535, 5104, 65535, 65535, 65535, 5105, 65535, 5106, 65535, 65535, 65535, 5107, 65535, 65535, 65535, 65535, 65535, 5108, 65535, 5109, 65535, 65535, 65535, 65535, 65535, 5110, 65535, 65535, 65535, 5111, 65535, 65535, 65535, 65535, 65535, 5112, 65535, 5113, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5114, 65535, 5115, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5116, 65535, 5117, 65535, 65535, 65535, 5118, 65535, 65535, 65535, 65535, 65535, 5119, 65535, 65535, 65535, 65535, 65535, 5120, 65535, 5121, 65535, 65535, 65535, 65535, 65535, 5122, 65535, 65535, 65535, 5123, 65535, 5124, 65535, 65535, 65535, 5125, 65535, 65535, 65535, 65535, 65535, 5126, 65535, 65535, 65535, 65535, 65535, 5127, 65535, 5128, 65535, 65535, 65535, 65535, 65535, 5129, 65535, 65535, 65535, 65535, 65535, 5130, 65535, 65535, 65535, 65535, 65535, 5131, 65535, 65535, 65535, 5132, 65535, 65535, 65535, 65535, 65535, 5133, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5134, 65535, 5135, 65535, 65535, 65535, 5136, 65535, 5137, 65535, 65535, 65535, 5138, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5139, 65535, 65535, 65535, 65535, 65535, 5140, 65535, 65535, 65535, 5141, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5142, 65535, 65535, 65535, 5143, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5144, 65535, 65535, 65535, 65535, 65535, 5145, 65535, 65535, 65535, 65535, 65535, 5146, 65535, 65535, 65535, 5147, 65535, 5148, 65535, 65535, 65535, 5149, 65535, 65535, 65535, 65535, 65535, 5150, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5151, 65535, 65535, 65535, 5152, 65535, 5153, 65535, 65535, 65535, 5154, 65535, 5155, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5156, 65535, 5157, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5158, 65535, 5159, 65535, 65535, 65535, 65535, 65535, 5160, 65535, 65535, 65535, 5161, 65535, 65535, 65535, 65535, 65535, 5162, 65535, 5163, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5164, 65535, 65535, 65535, 65535, 65535, 5165, 65535, 65535, 65535, 65535, 65535, 5166, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5167, 65535, 65535, 65535, 65535, 65535, 5168, 65535, 65535, 65535, 5169, 65535, 5170, 65535, 65535, 65535, 65535, 65535, 5171, 65535, 65535, 65535, 5172, 65535, 65535, 65535, 65535, 65535, 5173, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5174, 65535, 65535, 65535, 5175, 65535, 65535, 65535, 65535, 65535, 5176, 65535, 5177, 65535, 65535, 65535, 5178, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5179, 65535, 65535, 65535, 5180, 65535, 65535, 65535, 65535, 65535, 5181, 65535, 5182, 65535, 65535, 65535, 5183, 65535, 65535, 65535, 65535, 65535, 5184, 65535, 5185, 65535, 65535, 65535, 65535, 65535, 5186, 65535, 65535, 65535, 65535, 65535, 5187, 65535, 65535, 65535, 5188, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5189, 65535, 5190, 65535, 65535, 65535, 65535, 65535, 5191, 65535, 65535, 65535, 65535, 65535, 5192, 65535, 65535, 65535, 5193, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5194, 65535, 5195, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5196, 65535, 5197, 65535, 65535, 65535, 5198, 65535, 5199, 65535, 65535, 65535, 5200, 65535, 65535, 65535, 65535, 65535, 5201, 65535, 5202, 65535, 65535, 65535, 65535, 65535, 5203, 65535, 65535, 65535, 65535, 65535, 5204, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5205, 65535, 65535, 65535, 65535, 65535, 5206, 65535, 5207, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5208, 65535, 5209, 65535, 65535, 65535, 65535, 65535, 5210, 65535, 65535, 65535, 5211, 65535, 65535, 65535, 65535, 65535, 5212, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5213, 65535, 65535, 65535, 5214, 65535, 5215, 65535, 65535, 65535, 5216, 65535, 5217, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5218, 65535, 65535, 65535, 65535, 65535, 5219, 65535, 65535, 65535, 5220, 65535, 65535, 65535, 65535, 65535, 5221, 65535, 5222, 65535, 65535, 65535, 5223, 65535, 65535, 65535, 65535, 65535, 5224, 65535, 5225, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5226, 65535, 65535, 65535, 5227, 65535, 5228, 65535, 65535, 65535, 5229, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5230, 65535, 65535, 65535, 65535, 65535, 5231, 65535, 65535, 65535, 5232, 65535, 5233, 65535, 65535, 65535, 65535, 65535, 5234, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5235, 65535, 5236, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5237, 65535, 65535, 65535, 65535, 65535, 5238, 65535, 5239, 65535, 65535, 65535, 5240, 65535, 65535, 65535, 65535, 65535, 5241, 65535, 5242, 65535, 65535, 65535, 65535, 65535, 5243, 65535, 65535, 65535, 5244, 65535, 5245, 65535, 65535, 65535, 5246, 65535, 65535, 65535, 65535, 65535, 5247, 65535, 65535, 65535, 65535, 65535, 5248, 65535, 5249, 65535, 65535, 65535, 65535, 65535, 5250, 65535, 65535, 65535, 5251, 65535, 5252, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5253, 65535, 65535, 65535, 65535, 65535, 5254, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5255, 65535, 65535, 65535, 65535, 65535, 5256, 65535, 65535, 65535, 5257, 65535, 5258, 65535, 65535, 65535, 5259, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5260, 65535, 65535, 65535, 65535, 65535, 5261, 65535, 65535, 65535, 5262, 65535, 65535, 65535, 65535, 65535, 5263, 65535, 5264, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5265, 65535, 5266, 65535, 65535, 65535, 65535, 65535, 5267, 65535, 65535, 65535, 65535, 65535, 5268, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5269, 65535, 65535, 65535, 65535, 65535, 5270, 65535, 5271, 65535, 65535, 65535, 65535, 65535, 5272, 65535, 65535, 65535, 5273, 65535, 5274, 65535, 65535, 65535, 5275, 65535, 5276, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5277, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5278, 65535, 65535, 65535, 5279, 65535, 5280, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5281, 65535, 5282, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5283, 65535, 5284, 65535, 65535, 65535, 5285, 65535, 65535, 65535, 65535, 65535, 5286, 65535, 65535, 65535, 65535, 65535, 5287, 65535, 5288, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5289, 65535, 5290, 65535, 65535, 65535, 65535, 65535, 5291, 65535, 65535, 65535, 5292, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5293, 65535, 65535, 65535, 5294, 65535, 5295, 65535, 65535, 65535, 5296, 65535, 5297, 65535, 65535, 65535, 5298, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5299, 65535, 65535, 65535, 65535, 65535, 5300, 65535, 65535, 65535, 5301, 65535, 65535, 65535, 65535, 65535, 5302, 65535, 5303, 65535, 65535, 65535, 5304, 65535, 65535, 65535, 65535, 65535, 5305, 65535, 5306, 65535, 65535, 65535, 65535, 65535, 5307, 65535, 65535, 65535, 65535, 65535, 5308, 65535, 65535, 65535, 65535, 65535, 5309, 65535, 65535, 65535, 5310, 65535, 65535, 65535, 65535, 65535, 5311, 65535, 5312, 65535, 65535, 65535, 65535, 65535, 5313, 65535, 65535, 65535, 5314, 65535, 65535, 65535, 65535, 65535, 5315, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5316, 65535, 5317, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5318, 65535, 65535, 65535, 5319, 65535, 5320, 65535, 65535, 65535, 5321, 65535, 65535, 65535, 65535, 65535, 5322, 65535, 5323, 65535, 65535, 65535, 65535, 65535, 5324, 65535, 65535, 65535, 5325, 65535, 5326, 65535, 65535, 65535, 5327, 65535, 65535, 65535, 65535, 65535, 5328, 65535, 65535, 65535, 65535, 65535, 5329, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5330, 65535, 65535, 65535, 5331, 65535, 5332, 65535, 65535, 65535, 65535, 65535, 5333, 65535, 65535, 65535, 5334, 65535, 65535, 65535, 65535, 65535, 5335, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5336, 65535, 65535, 65535, 5337, 65535, 5338, 65535, 65535, 65535, 5339, 65535, 5340, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5341, 65535, 65535, 65535, 5342, 65535, 65535, 65535, 65535, 65535, 5343, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5344, 65535, 5345, 65535, 65535, 65535, 65535, 65535, 5346, 65535, 65535, 65535, 65535, 65535, 5347, 65535, 65535, 65535, 5348, 65535, 5349, 65535, 65535, 65535, 5350, 65535, 65535, 65535, 65535, 65535, 5351, 65535, 5352, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5353, 65535, 65535, 65535, 5354, 65535, 5355, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5356, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5357, 65535, 5358, 65535, 65535, 65535, 5359, 65535, 5360, 65535, 65535, 65535, 5361, 65535, 65535, 65535, 65535, 65535, 5362, 65535, 5363, 65535, 65535, 65535, 65535, 65535, 5364, 65535, 65535, 65535, 5365, 65535, 65535, 65535, 65535, 65535, 5366, 65535, 65535, 65535, 65535, 65535, 5367, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5368, 65535, 65535, 65535, 65535, 65535, 5369, 65535, 65535, 65535, 5370, 65535, 5371, 65535, 65535, 65535, 65535, 65535, 5372, 65535, 65535, 65535, 5373, 65535, 65535, 65535, 65535, 65535, 5374, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5375, 65535, 65535, 65535, 5376, 65535, 5377, 65535, 65535, 65535, 5378, 65535, 65535, 65535, 65535, 65535, 5379, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5380, 65535, 65535, 65535, 65535, 65535, 5381, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5382, 65535, 65535, 65535, 5383, 65535, 65535, 65535, 65535, 65535, 5384, 65535, 5385, 65535, 65535, 65535, 65535, 65535, 5386, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5387, 65535, 5388, 65535, 65535, 65535, 5389, 65535, 65535, 65535, 65535, 65535, 5390, 65535, 5391, 65535, 65535, 65535, 65535, 65535, 5392, 65535, 65535, 65535, 5393, 65535, 5394, 65535, 65535, 65535, 65535, 65535, 5395, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5396, 65535, 5397, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5398, 65535, 5399, 65535, 65535, 65535, 5400, 65535, 5401, 65535, 65535, 65535, 5402, 65535, 65535, 65535, 65535, 65535, 5403, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5404, 65535, 65535, 65535, 5405, 65535, 5406, 65535, 65535, 65535, 5407, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5408, 65535, 5409, 65535, 65535, 65535, 65535, 65535, 5410, 65535, 65535, 65535, 5411, 65535, 5412, 65535, 65535, 65535, 65535, 65535, 5413, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5414, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5415, 65535, 5416, 65535, 65535, 65535, 5417, 65535, 65535, 65535, 65535, 65535, 5418, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5419, 65535, 65535, 65535, 65535, 65535, 5420, 65535, 65535, 65535, 5421, 65535, 65535, 65535, 65535, 65535, 5422, 65535, 5423, 65535, 65535, 65535, 5424, 65535, 65535, 65535, 65535, 65535, 5425, 65535, 5426, 65535, 65535, 65535, 65535, 65535, 5427, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5428, 65535, 5429, 65535, 65535, 65535, 5430, 65535, 65535, 65535, 65535, 65535, 5431, 65535, 5432, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5433, 65535, 5434, 65535, 65535, 65535, 5435, 65535, 5436, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5437, 65535, 5438, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5439, 65535, 5440, 65535, 65535, 65535, 5441, 65535, 65535, 65535, 65535, 65535, 5442, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5443, 65535, 65535, 65535, 65535, 65535, 5444, 65535, 65535, 65535, 65535, 65535, 5445, 65535, 65535, 65535, 5446, 65535, 65535, 65535, 65535, 65535, 5447, 65535, 65535, 65535, 65535, 65535, 5448, 65535, 5449, 65535, 65535, 65535, 65535, 65535, 5450, 65535, 65535, 65535, 5451, 65535, 5452, 65535, 65535, 65535, 65535, 65535, 5453, 65535, 65535, 65535, 5454, 65535, 65535, 65535, 65535, 65535, 5455, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5456, 65535, 65535, 65535, 5457, 65535, 5458, 65535, 65535, 65535, 65535, 65535, 5459, 65535, 65535, 65535, 5460, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5461, 65535, 65535, 65535, 65535, 65535, 5462, 65535, 65535, 65535, 5463, 65535, 65535, 65535, 65535, 65535, 5464, 65535, 5465, 65535, 65535, 65535, 5466, 65535, 65535, 65535, 65535, 65535, 5467, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5468, 65535, 65535, 65535, 65535, 65535, 5469, 65535, 65535, 65535, 5470, 65535, 5471, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5472, 65535, 5473, 65535, 65535, 65535, 65535, 65535, 5474, 65535, 65535, 65535, 5475, 65535, 5476, 65535, 65535, 65535, 5477, 65535, 5478, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5479, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5480, 65535, 5481, 65535, 65535, 65535, 5482, 65535, 5483, 65535, 65535, 65535, 5484, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5485, 65535, 65535, 65535, 65535, 65535, 5486, 65535, 65535, 65535, 5487, 65535, 5488, 65535, 65535, 65535, 5489, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5490, 65535, 5491, 65535, 65535, 65535, 65535, 65535, 5492, 65535, 65535, 65535, 5493, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5494, 65535, 65535, 65535, 65535, 65535, 5495, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5496, 65535, 65535, 65535, 5497, 65535, 5498, 65535, 65535, 65535, 5499, 65535, 5500, 65535, 65535, 65535, 5501, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5502, 65535, 65535, 65535, 65535, 65535, 5503, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5504, 65535, 65535, 65535, 65535, 65535, 5505, 65535, 65535, 65535, 65535, 65535, 5506, 65535, 5507, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5508, 65535, 65535, 65535, 5509, 65535, 5510, 65535, 65535, 65535, 5511, 65535, 65535, 65535, 65535, 65535, 5512, 65535, 5513, 65535, 65535, 65535, 65535, 65535, 5514, 65535, 65535, 65535, 5515, 65535, 5516, 65535, 65535, 65535, 5517, 65535, 5518, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5519, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5520, 65535, 5521, 65535, 65535, 65535, 65535, 65535, 5522, 65535, 65535, 65535, 5523, 65535, 65535, 65535, 65535, 65535, 5524, 65535, 5525, 65535, 65535, 65535, 65535, 65535, 5526, 65535, 65535, 65535, 5527, 65535, 5528, 65535, 65535, 65535, 5529, 65535, 65535, 65535, 65535, 65535, 5530, 65535, 65535, 65535, 65535, 65535, 5531, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5532, 65535, 65535, 65535, 5533, 65535, 5534, 65535, 65535, 65535, 65535, 65535, 5535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5536, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5537, 65535, 65535, 65535, 5538, 65535, 65535, 65535, 65535, 65535, 5539, 65535, 5540, 65535, 65535, 65535, 5541, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5542, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5543, 65535, 65535, 65535, 65535, 65535, 5544, 65535, 5545, 65535, 65535, 65535, 5546, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5547, 65535, 65535, 65535, 65535, 65535, 5548, 65535, 65535, 65535, 65535, 65535, 5549, 65535, 65535, 65535, 5550, 65535, 5551, 65535, 65535, 65535, 5552, 65535, 65535, 65535, 65535, 65535, 5553, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5554, 65535, 65535, 65535, 5555, 65535, 65535, 65535, 65535, 65535, 5556, 65535, 5557, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5558, 65535, 5559, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5560, 65535, 5561, 65535, 65535, 65535, 5562, 65535, 5563, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5564, 65535, 5565, 65535, 65535, 65535, 65535, 65535, 5566, 65535, 65535, 65535, 5567, 65535, 5568, 65535, 65535, 65535, 5569, 65535, 65535, 65535, 65535, 65535, 5570, 65535, 65535, 65535, 65535, 65535, 5571, 65535, 5572, 65535, 65535, 65535, 65535, 65535, 5573, 65535, 65535, 65535, 65535, 65535, 5574, 65535, 65535, 65535, 65535, 65535, 5575, 65535, 65535, 65535, 5576, 65535, 65535, 65535, 65535, 65535, 5577, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5578, 65535, 65535, 65535, 65535, 65535, 5579, 65535, 65535, 65535, 5580, 65535, 5581, 65535, 65535, 65535, 5582, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5583, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5584, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5585, 65535, 65535, 65535, 5586, 65535, 65535, 65535, 65535, 65535, 5587, 65535, 5588, 65535, 65535, 65535, 65535, 65535, 5589, 65535, 65535, 65535, 65535, 65535, 5590, 65535, 65535, 65535, 5591, 65535, 5592, 65535, 65535, 65535, 5593, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5594, 65535, 65535, 65535, 5595, 65535, 5596, 65535, 65535, 65535, 5597, 65535, 5598, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5599, 65535, 5600, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5601, 65535, 5602, 65535, 65535, 65535, 5603, 65535, 5604, 65535, 65535, 65535, 5605, 65535, 65535, 65535, 65535, 65535, 5606, 65535, 5607, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5608, 65535, 5609, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5610, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5611, 65535, 65535, 65535, 65535, 65535, 5612, 65535, 65535, 65535, 5613, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5614, 65535, 65535, 65535, 5615, 65535, 65535, 65535, 65535, 65535, 5616, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5617, 65535, 65535, 65535, 5618, 65535, 65535, 65535, 65535, 65535, 5619, 65535, 5620, 65535, 65535, 65535, 5621, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5622, 65535, 65535, 65535, 65535, 65535, 5623, 65535, 65535, 65535, 5624, 65535, 65535, 65535, 65535, 65535, 5625, 65535, 5626, 65535, 65535, 65535, 5627, 65535, 65535, 65535, 65535, 65535, 5628, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5629, 65535, 65535, 65535, 65535, 65535, 5630, 65535, 65535, 65535, 5631, 65535, 65535, 65535, 65535, 65535, 5632, 65535, 65535, 65535, 65535, 65535, 5633, 65535, 5634, 65535, 65535, 65535, 65535, 65535, 5635, 65535, 65535, 65535, 65535, 65535, 5636, 65535, 65535, 65535, 5637, 65535, 5638, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5639, 65535, 5640, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5641, 65535, 5642, 65535, 65535, 65535, 5643, 65535, 65535, 65535, 65535, 65535, 5644, 65535, 65535, 65535, 65535, 65535, 5645, 65535, 5646, 65535, 65535, 65535, 65535, 65535, 5647, 65535, 65535, 65535, 5648, 65535, 5649, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5650, 65535, 65535, 65535, 65535, 65535, 5651, 65535, 5652, 65535, 65535, 65535, 65535, 65535, 5653, 65535, 65535, 65535, 65535, 65535, 5654, 65535, 65535, 65535, 65535, 65535, 5655, 65535, 65535, 65535, 5656, 65535, 65535, 65535, 65535, 65535, 5657, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5658, 65535, 5659, 65535, 65535, 65535, 5660, 65535, 5661, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5662, 65535, 65535, 65535, 65535, 65535, 5663, 65535, 65535, 65535, 5664, 65535, 65535, 65535, 65535, 65535, 5665, 65535, 5666, 65535, 65535, 65535, 5667, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5668, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5669, 65535, 65535, 65535, 5670, 65535, 5671, 65535, 65535, 65535, 5672, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5673, 65535, 65535, 65535, 65535, 65535, 5674, 65535, 65535, 65535, 5675, 65535, 5676, 65535, 65535, 65535, 5677, 65535, 5678, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5679, 65535, 5680, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5681, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5682, 65535, 65535, 65535, 5683, 65535, 65535, 65535, 65535, 65535, 5684, 65535, 5685, 65535, 65535, 65535, 65535, 65535, 5686, 65535, 65535, 65535, 5687, 65535, 65535, 65535, 65535, 65535, 5688, 65535, 65535, 65535, 65535, 65535, 5689, 65535, 65535, 65535, 65535, 65535, 5690, 65535, 5691, 65535, 65535, 65535, 65535, 65535, 5692, 65535, 65535, 65535, 5693, 65535, 5694, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5695, 65535, 65535, 65535, 65535, 65535, 5696, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5697, 65535, 65535, 65535, 65535, 65535, 5698, 65535, 65535, 65535, 5699, 65535, 5700, 65535, 65535, 65535, 5701, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5702, 65535, 65535, 65535, 5703, 65535, 65535, 65535, 65535, 65535, 5704, 65535, 5705, 65535, 65535, 65535, 5706, 65535, 65535, 65535, 65535, 65535, 5707, 65535, 5708, 65535, 65535, 65535, 65535, 65535, 5709, 65535, 65535, 65535, 65535, 65535, 5710, 65535, 65535, 65535, 65535, 65535, 5711, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5712, 65535, 5713, 65535, 65535, 65535, 65535, 65535, 5714, 65535, 65535, 65535, 5715, 65535, 5716, 65535, 65535, 65535, 5717, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5718, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5719, 65535, 5720, 65535, 65535, 65535, 5721, 65535, 5722, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5723, 65535, 5724, 65535, 65535, 65535, 65535, 65535, 5725, 65535, 65535, 65535, 65535, 65535, 5726, 65535, 65535, 65535, 5727, 65535, 65535, 65535, 65535, 65535, 5728, 65535, 65535, 65535, 65535, 65535, 5729, 65535, 5730, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5731, 65535, 5732, 65535, 65535, 65535, 65535, 65535, 5733, 65535, 65535, 65535, 5734, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5735, 65535, 65535, 65535, 5736, 65535, 5737, 65535, 65535, 65535, 5738, 65535, 5739, 65535, 65535, 65535, 5740, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5741, 65535, 65535, 65535, 65535, 65535, 5742, 65535, 65535, 65535, 5743, 65535, 65535, 65535, 65535, 65535, 5744, 65535, 5745, 65535, 65535, 65535, 5746, 65535, 65535, 65535, 65535, 65535, 5747, 65535, 5748, 65535, 65535, 65535, 65535, 65535, 5749, 65535, 65535, 65535, 65535, 65535, 5750, 65535, 65535, 65535, 5751, 65535, 5752, 65535, 65535, 65535, 5753, 65535, 65535, 65535, 65535, 65535, 5754, 65535, 5755, 65535, 65535, 65535, 65535, 65535, 5756, 65535, 65535, 65535, 5757, 65535, 5758, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 5759 };
__constant ushort WHL_INV[30030] = { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3533, 0, 26869, 0, 0, 0, 3917, 0, 0, 0, 0, 0, 27959, 0, 6781, 0, 0, 0, 0, 0, 6493, 0, 0, 0, 18311, 0, 5587, 0, 0, 0, 10223, 0, 0, 0, 0, 0, 27197, 0, 0, 0, 0, 0, 509, 0, 21661, 0, 0, 0, 0, 0, 19273, 0, 0, 0, 10151, 0, 11107, 0, 0, 0, 0, 0, 26989, 0, 0, 0, 9407, 0, 0, 0, 0, 0, 4049, 0, 0, 0, 0, 0, 0, 0, 5263, 0, 0, 0, 15461, 0, 16327, 0, 0, 0, 22733, 0, 29479, 0, 0, 0, 28967, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19153, 0, 0, 0, 8711, 0, 0, 0, 0, 0, 15563, 0, 4969, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20759, 0, 1591, 0, 0, 0, 0, 0, 13963, 0, 0, 0, 0, 0, 5527, 0, 0, 0, 7013, 0, 0, 0, 0, 0, 27947, 0, 0, 0, 0, 0, 13589, 0, 5641, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23741, 0, 22717, 0, 0, 0, 10823, 0, 3169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17221, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15217, 0, 0, 0, 18653, 0, 12589, 0, 0, 0, 21137, 0, 0, 0, 0, 0, 4649, 0, 17071, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6341, 0, 0, 0, 0, 0, 16943, 0, 0, 0, 0, 0, 14387, 0, 0, 0, 0, 0, 15629, 0, 20611, 0, 0, 0, 0, 0, 1843, 0, 0, 0, 4061, 0, 17827, 0, 0, 0, 0, 0, 19639, 0, 0, 0, 6047, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15553, 0, 0, 0, 8111, 0, 8347, 0, 0, 0, 13073, 0, 0, 0, 0, 0, 3347, 0, 0, 0, 0, 0, 0, 0, 26401, 0, 0, 0, 0, 0, 22723, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27953, 0, 9379, 0, 0, 0, 20417, 0, 0, 0, 0, 0, 17399, 0, 21961, 0, 0, 0, 0, 0, 22993, 0, 0, 0, 0, 0, 4267, 0, 0, 0, 0, 0, 10459, 0, 0, 0, 2117, 0, 0, 0, 0, 0, 22619, 0, 24961, 0, 0, 0, 0, 0, 10363, 0, 0, 0, 22691, 0, 0, 0, 0, 0, 0, 0, 1909, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11969, 0, 7561, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19091, 0, 19627, 0, 0, 0, 20753, 0, 27499, 0, 0, 0, 2237, 0, 0, 0, 0, 0, 20399, 0, 0, 0, 0, 0, 0, 0, 27073, 0, 0, 0, 5081, 0, 3697, 0, 0, 0, 20513, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11849, 0, 0, 0, 0, 0, 0, 0, 6043, 0, 0, 0, 5321, 0, 10477, 0, 0, 0, 0, 0, 26359, 0, 0, 0, 26627, 0, 0, 0, 0, 0, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5591, 0, 9187, 0, 0, 0, 23363, 0, 27589, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6661, 0, 0, 0, 0, 0, 20203, 0, 0, 0, 29921, 0, 0, 0, 0, 0, 26903, 0, 0, 0, 0, 0, 2987, 0, 0, 0, 0, 0, 24119, 0, 9151, 0, 0, 0, 0, 0, 24253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24863, 0, 6679, 0, 0, 0, 12407, 0, 0, 0, 0, 0, 14639, 0, 28531, 0, 0, 0, 0, 0, 9103, 0, 0, 0, 0, 0, 21457, 0, 0, 0, 11243, 0, 17659, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26879, 0, 9661, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7121, 0, 4717, 0, 0, 0, 11093, 0, 0, 0, 0, 0, 11267, 0, 0, 0, 0, 0, 19139, 0, 17491, 0, 0, 0, 0, 0, 26023, 0, 0, 0, 0, 0, 18607, 0, 0, 0, 9803, 0, 0, 0, 0, 0, 28667, 0, 0, 0, 0, 0, 0, 0, 8431, 0, 0, 0, 0, 0, 8143, 0, 0, 0, 14351, 0, 16147, 0, 0, 0, 0, 0, 22999, 0, 0, 0, 14657, 0, 0, 0, 0, 0, 4469, 0, 0, 0, 0, 0, 0, 0, 6733, 0, 0, 0, 9161, 0, 10447, 0, 0, 0, 0, 0, 10159, 0, 0, 0, 23927, 0, 0, 0, 0, 0, 0, 0, 27031, 0, 0, 0, 0, 0, 12853, 0, 0, 0, 10181, 0, 0, 0, 0, 0, 0, 0, 5389, 0, 0, 0, 7187, 0, 0, 0, 0, 0, 16769, 0, 0, 0, 0, 0, 0, 0, 18163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23813, 0, 21799, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15479, 0, 5221, 0, 0, 0, 0, 0, 27163, 0, 0, 0, 13241, 0, 9487, 0, 0, 0, 21533, 0, 24379, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12599, 0, 24781, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27701, 0, 28657, 0, 0, 0, 28313, 0, 9439, 0, 0, 0, 12527, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16573, 0, 0, 0, 19361, 0, 11257, 0, 0, 0, 13373, 0, 0, 0, 0, 0, 27407, 0, 0, 0, 0, 0, 10589, 0, 21031, 0, 0, 0, 0, 0, 9403, 0, 0, 0, 25481, 0, 0, 0, 0, 0, 0, 0, 5359, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19589, 0, 0, 0, 0, 0, 0, 0, 12403, 0, 0, 0, 7691, 0, 12547, 0, 0, 0, 23783, 0, 0, 0, 0, 0, 22877, 0, 0, 0, 0, 0, 0, 0, 6031, 0, 0, 0, 0, 0, 27763, 0, 0, 0, 14381, 0, 0, 0, 0, 0, 14723, 0, 0, 0, 0, 0, 9287, 0, 0, 0, 0, 0, 22439, 0, 21121, 0, 0, 0, 0, 0, 3253, 0, 0, 0, 0, 0, 26527, 0, 0, 0, 6173, 0, 23899, 0, 0, 0, 3587, 0, 0, 0, 0, 0, 16739, 0, 20971, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17651, 0, 8227, 0, 0, 0, 11873, 0, 27949, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3149, 0, 21001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24341, 0, 29917, 0, 0, 0, 0, 0, 26659, 0, 0, 0, 6437, 0, 0, 0, 0, 0, 0, 0, 13501, 0, 0, 0, 0, 0, 2293, 0, 0, 0, 6551, 0, 8737, 0, 0, 0, 20093, 0, 0, 0, 0, 0, 15437, 0, 0, 0, 0, 0, 28649, 0, 0, 0, 0, 0, 0, 0, 10243, 0, 0, 0, 12671, 0, 7327, 0, 0, 0, 0, 0, 4309, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13499, 0, 0, 0, 0, 0, 0, 0, 5053, 0, 0, 0, 4331, 0, 16747, 0, 0, 0, 0, 0, 28009, 0, 0, 0, 13427, 0, 0, 0, 0, 0, 0, 0, 28081, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18791, 0, 0, 0, 0, 0, 27323, 0, 5809, 0, 0, 0, 10547, 0, 0, 0, 0, 0, 0, 0, 6001, 0, 0, 0, 0, 0, 7663, 0, 0, 0, 0, 0, 14557, 0, 0, 0, 2813, 0, 14239, 0, 0, 0, 9257, 0, 0, 0, 0, 0, 18839, 0, 19711, 0, 0, 0, 0, 0, 25903, 0, 0, 0, 21851, 0, 0, 0, 0, 0, 20903, 0, 17239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2099, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22871, 0, 8917, 0, 0, 0, 10253, 0, 16999, 0, 0, 0, 21557, 0, 0, 0, 0, 0, 11369, 0, 17911, 0, 0, 0, 0, 0, 17203, 0, 0, 0, 17681, 0, 12307, 0, 0, 0, 5813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27389, 0, 19141, 0, 0, 0, 0, 0, 2263, 0, 0, 0, 0, 0, 17617, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6887, 0, 0, 0, 0, 0, 14759, 0, 0, 0, 0, 0, 0, 0, 11773, 0, 0, 0, 24911, 0, 29347, 0, 0, 0, 19793, 0, 26959, 0, 0, 0, 29177, 0, 0, 0, 0, 0, 0, 0, 28921, 0, 0, 0, 0, 0, 25873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16099, 0, 0, 0, 11387, 0, 0, 0, 0, 0, 13619, 0, 21751, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5107, 0, 0, 0, 9533, 0, 1429, 0, 0, 0, 1907, 0, 0, 0, 0, 0, 23039, 0, 0, 0, 0, 0, 0, 0, 22123, 0, 0, 0, 2111, 0, 22507, 0, 0, 0, 12923, 0, 20809, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19741, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26441, 0, 20047, 0, 0, 0, 25163, 0, 19099, 0, 0, 0, 25967, 0, 0, 0, 0, 0, 29429, 0, 3001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26711, 0, 10837, 0, 0, 0, 4553, 0, 0, 0, 0, 0, 14177, 0, 0, 0, 0, 0, 0, 0, 4021, 0, 0, 0, 0, 0, 11293, 0, 0, 0, 26951, 0, 21817, 0, 0, 0, 0, 0, 22159, 0, 0, 0, 5627, 0, 0, 0, 0, 0, 23789, 0, 0, 0, 0, 0, 0, 0, 8413, 0, 0, 0, 12941, 0, 0, 0, 0, 0, 24203, 0, 9319, 0, 0, 0, 14057, 0, 0, 0, 0, 0, 0, 0, 151, 0, 0, 0, 0, 0, 8443, 0, 0, 0, 9341, 0, 0, 0, 0, 0, 25433, 0, 6439, 0, 0, 0, 3407, 0, 0, 0, 0, 0, 23909, 0, 12301, 0, 0, 0, 0, 0, 3673, 0, 0, 0, 0, 0, 1747, 0, 0, 0, 2183, 0, 0, 0, 0, 0, 8627, 0, 0, 0, 0, 0, 5609, 0, 0, 0, 0, 0, 0, 0, 7213, 0, 0, 0, 0, 0, 9697, 0, 0, 0, 13763, 0, 27529, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22679, 0, 7771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23921, 0, 7237, 0, 0, 0, 10883, 0, 4189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19979, 0, 26941, 0, 0, 0, 0, 0, 29173, 0, 0, 0, 27971, 0, 7477, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10397, 0, 0, 0, 0, 0, 11639, 0, 26701, 0, 0, 0, 0, 0, 1633, 0, 0, 0, 25691, 0, 26227, 0, 0, 0, 0, 0, 4729, 0, 0, 0, 21377, 0, 0, 0, 0, 0, 4889, 0, 0, 0, 0, 0, 0, 0, 20803, 0, 0, 0, 0, 0, 8977, 0, 0, 0, 9293, 0, 5959, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8291, 0, 0, 0, 0, 0, 10313, 0, 15469, 0, 0, 0, 12437, 0, 0, 0, 0, 0, 28109, 0, 11251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 277, 0, 0, 0, 7853, 0, 13399, 0, 0, 0, 5267, 0, 0, 0, 0, 0, 0, 0, 11941, 0, 0, 0, 0, 0, 9313, 0, 0, 0, 23321, 0, 20827, 0, 0, 0, 15023, 0, 13249, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27089, 0, 6511, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16571, 0, 0, 0, 0, 0, 1433, 0, 409, 0, 0, 0, 21977, 0, 0, 0, 0, 0, 16619, 0, 28201, 0, 0, 0, 0, 0, 16363, 0, 0, 0, 21881, 0, 28057, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25517, 0, 0, 0, 0, 0, 28859, 0, 24181, 0, 0, 0, 0, 0, 11923, 0, 0, 0, 13721, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28097, 0, 0, 0, 0, 0, 22109, 0, 0, 0, 0, 0, 0, 0, 19753, 0, 0, 0, 0, 0, 21577, 0, 0, 0, 3203, 0, 12679, 0, 0, 0, 26027, 0, 0, 0, 0, 0, 0, 0, 6451, 0, 0, 0, 0, 0, 26293, 0, 0, 0, 28871, 0, 0, 0, 0, 0, 2963, 0, 7489, 0, 0, 0, 2777, 0, 0, 0, 0, 0, 2489, 0, 0, 0, 0, 0, 0, 0, 4093, 0, 0, 0, 0, 0, 3847, 0, 0, 0, 0, 0, 28309, 0, 0, 0, 17657, 0, 0, 0, 0, 0, 15269, 0, 30001, 0, 0, 0, 0, 0, 29683, 0, 0, 0, 28991, 0, 29857, 0, 0, 0, 17123, 0, 11989, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1259, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1451, 0, 21517, 0, 0, 0, 383, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15149, 0, 7201, 0, 0, 0, 0, 0, 6703, 0, 0, 0, 6971, 0, 16087, 0, 0, 0, 26813, 0, 0, 0, 0, 0, 27617, 0, 0, 0, 0, 0, 9959, 0, 27751, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16777, 0, 0, 0, 0, 0, 2839, 0, 0, 0, 1637, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5171, 0, 25777, 0, 0, 0, 4463, 0, 5329, 0, 0, 0, 13217, 0, 0, 0, 0, 0, 0, 0, 19051, 0, 0, 0, 0, 0, 25243, 0, 0, 0, 14591, 0, 0, 0, 0, 0, 443, 0, 8329, 0, 0, 0, 13067, 0, 0, 0, 0, 0, 0, 0, 1801, 0, 0, 0, 0, 0, 14383, 0, 0, 0, 0, 0, 1327, 0, 0, 0, 29063, 0, 14029, 0, 0, 0, 16607, 0, 0, 0, 0, 0, 27869, 0, 20551, 0, 0, 0, 0, 0, 16873, 0, 0, 0, 21641, 0, 1087, 0, 0, 0, 21323, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11549, 0, 20791, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20257, 0, 0, 0, 0, 0, 29179, 0, 0, 0, 22397, 0, 0, 0, 0, 0, 18719, 0, 18331, 0, 0, 0, 0, 0, 8803, 0, 0, 0, 11381, 0, 0, 0, 0, 0, 14843, 0, 0, 0, 0, 0, 17957, 0, 0, 0, 0, 0, 18989, 0, 26281, 0, 0, 0, 0, 0, 12343, 0, 0, 0, 12461, 0, 15727, 0, 0, 0, 0, 0, 17749, 0, 0, 0, 20957, 0, 0, 0, 0, 0, 6359, 0, 0, 0, 0, 0, 0, 0, 7573, 0, 0, 0, 3911, 0, 27877, 0, 0, 0, 13493, 0, 10999, 0, 0, 0, 3557, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7603, 0, 0, 0, 29501, 0, 0, 0, 0, 0, 4013, 0, 9589, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23069, 0, 6211, 0, 0, 0, 0, 0, 4723, 0, 0, 0, 0, 0, 24007, 0, 0, 0, 9323, 0, 4579, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2039, 0, 17191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28361, 0, 25027, 0, 0, 0, 3893, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7981, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29591, 0, 8287, 0, 0, 0, 20963, 0, 17209, 0, 0, 0, 2657, 0, 0, 0, 0, 0, 16199, 0, 5521, 0, 0, 0, 0, 0, 23923, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5393, 0, 0, 0, 0, 0, 5147, 0, 0, 0, 0, 0, 20249, 0, 22921, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8681, 0, 6277, 0, 0, 0, 0, 0, 10399, 0, 0, 0, 24527, 0, 0, 0, 0, 0, 5099, 0, 0, 0, 0, 0, 0, 0, 20173, 0, 0, 0, 19661, 0, 29137, 0, 0, 0, 24623, 0, 0, 0, 0, 0, 5657, 0, 0, 0, 0, 0, 0, 0, 28711, 0, 0, 0, 0, 0, 4243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2543, 0, 11689, 0, 0, 0, 13487, 0, 0, 0, 0, 0, 22019, 0, 10411, 0, 0, 0, 0, 0, 13753, 0, 0, 0, 0, 0, 8887, 0, 0, 0, 13523, 0, 3529, 0, 0, 0, 4427, 0, 0, 0, 0, 0, 4139, 0, 15721, 0, 0, 0, 0, 0, 28843, 0, 0, 0, 25001, 0, 10117, 0, 0, 0, 0, 0, 4219, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2729, 0, 19111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12161, 0, 0, 0, 0, 0, 25373, 0, 15949, 0, 0, 0, 23027, 0, 0, 0, 0, 0, 8849, 0, 0, 0, 0, 0, 0, 0, 20143, 0, 0, 0, 7391, 0, 8317, 0, 0, 0, 2033, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14159, 0, 13471, 0, 0, 0, 0, 0, 8353, 0, 0, 0, 23801, 0, 22027, 0, 0, 0, 0, 0, 7879, 0, 0, 0, 1217, 0, 0, 0, 0, 0, 23159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26381, 0, 29977, 0, 0, 0, 11813, 0, 2179, 0, 0, 0, 20777, 0, 0, 0, 0, 0, 0, 0, 11281, 0, 0, 0, 0, 0, 10963, 0, 0, 0, 11441, 0, 0, 0, 0, 0, 29213, 0, 15889, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12569, 0, 20701, 0, 0, 0, 0, 0, 5773, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17933, 0, 0, 0, 0, 0, 17027, 0, 0, 0, 0, 0, 5399, 0, 19291, 0, 0, 0, 0, 0, 13723, 0, 0, 0, 7361, 0, 14527, 0, 0, 0, 13553, 0, 22279, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8399, 0, 28141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7027, 0, 0, 0, 29573, 0, 0, 0, 0, 0, 2027, 0, 0, 0, 0, 0, 23759, 0, 19801, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7057, 0, 0, 0, 563, 0, 0, 0, 0, 0, 17117, 0, 0, 0, 0, 0, 29279, 0, 1501, 0, 0, 0, 0, 0, 12763, 0, 0, 0, 25901, 0, 6907, 0, 0, 0, 0, 0, 11449, 0, 0, 0, 16967, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18283, 0, 0, 0, 29951, 0, 21997, 0, 0, 0, 0, 0, 12469, 0, 0, 0, 16997, 0, 0, 0, 0, 0, 0, 0, 15481, 0, 0, 0, 0, 0, 3613, 0, 0, 0, 28661, 0, 0, 0, 0, 0, 19763, 0, 28489, 0, 0, 0, 9497, 0, 0, 0, 0, 0, 28319, 0, 0, 0, 0, 0, 0, 0, 6613, 0, 0, 0, 0, 0, 6997, 0, 0, 0, 0, 0, 24109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6239, 0, 16771, 0, 0, 0, 0, 0, 29473, 0, 0, 0, 6311, 0, 0, 0, 0, 0, 26153, 0, 12829, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1049, 0, 29401, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30011, 0, 3247, 0, 0, 0, 9833, 0, 199, 0, 0, 0, 24077, 0, 0, 0, 0, 0, 0, 0, 18541, 0, 0, 0, 0, 0, 18883, 0, 0, 0, 7811, 0, 22807, 0, 0, 0, 4133, 0, 0, 0, 0, 0, 1997, 0, 0, 0, 0, 0, 3659, 0, 0, 0, 0, 0, 0, 0, 27883, 0, 0, 0, 16241, 0, 0, 0, 0, 0, 0, 0, 9979, 0, 0, 0, 15707, 0, 0, 0, 0, 0, 21899, 0, 0, 0, 0, 0, 0, 0, 3163, 0, 0, 0, 19241, 0, 997, 0, 0, 0, 26093, 0, 16669, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17581, 0, 0, 0, 0, 0, 9283, 0, 0, 0, 19001, 0, 0, 0, 0, 0, 7793, 0, 0, 0, 0, 0, 13907, 0, 0, 0, 0, 0, 13199, 0, 11881, 0, 0, 0, 0, 0, 7873, 0, 0, 0, 0, 0, 19597, 0, 0, 0, 8483, 0, 28519, 0, 0, 0, 15137, 0, 0, 0, 0, 0, 28289, 0, 9421, 0, 0, 0, 0, 0, 20023, 0, 0, 0, 0, 0, 10537, 0, 0, 0, 323, 0, 9469, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7769, 0, 23311, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28961, 0, 18367, 0, 0, 0, 0, 0, 17419, 0, 0, 0, 24917, 0, 0, 0, 0, 0, 21869, 0, 6571, 0, 0, 0, 0, 0, 6913, 0, 0, 0, 18101, 0, 29527, 0, 0, 0, 1613, 0, 0, 0, 0, 0, 17747, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21793, 0, 0, 0, 3431, 0, 18877, 0, 0, 0, 0, 0, 6619, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18119, 0, 0, 0, 0, 0, 0, 0, 25843, 0, 0, 0, 22811, 0, 21367, 0, 0, 0, 24833, 0, 21079, 0, 0, 0, 15737, 0, 0, 0, 0, 0, 0, 0, 18841, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21101, 0, 0, 0, 0, 0, 0, 0, 8119, 0, 0, 0, 29027, 0, 0, 0, 0, 0, 0, 0, 17551, 0, 0, 0, 0, 0, 9973, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7433, 0, 2689, 0, 0, 0, 17, 0, 0, 0, 0, 0, 7289, 0, 24331, 0, 0, 0, 0, 0, 18973, 0, 0, 0, 24161, 0, 0, 0, 0, 0, 2423, 0, 7999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4409, 0, 8401, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11321, 0, 20467, 0, 0, 0, 1013, 0, 28549, 0, 0, 0, 26177, 0, 0, 0, 0, 0, 4439, 0, 0, 0, 0, 0, 0, 0, 5653, 0, 0, 0, 8441, 0, 3067, 0, 0, 0, 24293, 0, 0, 0, 0, 0, 8297, 0, 0, 0, 0, 0, 29699, 0, 23761, 0, 0, 0, 0, 0, 23053, 0, 0, 0, 0, 0, 6067, 0, 0, 0, 0, 0, 27199, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3209, 0, 0, 0, 0, 0, 0, 0, 23323, 0, 0, 0, 29531, 0, 1627, 0, 0, 0, 12863, 0, 0, 0, 0, 0, 3767, 0, 0, 0, 0, 0, 0, 0, 19681, 0, 0, 0, 0, 0, 463, 0, 0, 0, 6191, 0, 0, 0, 0, 0, 0, 0, 20719, 0, 0, 0, 22937, 0, 0, 0, 0, 0, 25169, 0, 10201, 0, 0, 0, 0, 0, 14173, 0, 0, 0, 0, 0, 7417, 0, 0, 0, 28013, 0, 12979, 0, 0, 0, 22697, 0, 0, 0, 0, 0, 27659, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6731, 0, 10957, 0, 0, 0, 3683, 0, 11569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16799, 0, 12811, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7961, 0, 10807, 0, 0, 0, 6683, 0, 7549, 0, 0, 0, 28277, 0, 0, 0, 0, 0, 0, 0, 5311, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17471, 0, 22387, 0, 0, 0, 9173, 0, 0, 0, 0, 0, 7247, 0, 0, 0, 0, 0, 0, 0, 22501, 0, 0, 0, 0, 0, 2053, 0, 0, 0, 15401, 0, 26437, 0, 0, 0, 0, 0, 15229, 0, 0, 0, 7937, 0, 0, 0, 0, 0, 5309, 0, 0, 0, 0, 0, 0, 0, 26893, 0, 0, 0, 15251, 0, 0, 0, 0, 0, 0, 0, 11629, 0, 0, 0, 2507, 0, 0, 0, 0, 0, 0, 0, 11701, 0, 0, 0, 0, 0, 10753, 0, 0, 0, 2411, 0, 0, 0, 0, 0, 23, 0, 24919, 0, 0, 0, 24197, 0, 0, 0, 0, 0, 12359, 0, 16921, 0, 0, 0, 0, 0, 26773, 0, 0, 0, 0, 0, 6367, 0, 0, 0, 13733, 0, 0, 0, 0, 0, 20177, 0, 0, 0, 0, 0, 7919, 0, 22441, 0, 0, 0, 0, 0, 9523, 0, 0, 0, 0, 0, 21247, 0, 0, 0, 4523, 0, 9049, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15749, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14681, 0, 28027, 0, 0, 0, 29363, 0, 8809, 0, 0, 0, 2447, 0, 0, 0, 0, 0, 22289, 0, 1531, 0, 0, 0, 0, 0, 19933, 0, 0, 0, 9491, 0, 25957, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 89, 0, 8221, 0, 0, 0, 0, 0, 13183, 0, 0, 0, 281, 0, 28537, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25997, 0, 0, 0, 0, 0, 25679, 0, 0, 0, 0, 0, 0, 0, 25423, 0, 0, 0, 13991, 0, 2047, 0, 0, 0, 11603, 0, 10579, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20731, 0, 0, 0, 0, 0, 14953, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28793, 0, 27019, 0, 0, 0, 3197, 0, 0, 0, 0, 0, 2699, 0, 13561, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18757, 0, 0, 0, 28643, 0, 4159, 0, 0, 0, 23747, 0, 0, 0, 0, 0, 0, 0, 5011, 0, 0, 0, 0, 0, 13933, 0, 0, 0, 4841, 0, 11587, 0, 0, 0, 26573, 0, 1699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8821, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7331, 0, 0, 0, 0, 0, 6053, 0, 2719, 0, 0, 0, 15047, 0, 0, 0, 0, 0, 21239, 0, 16651, 0, 0, 0, 0, 0, 7123, 0, 0, 0, 10331, 0, 2647, 0, 0, 0, 7283, 0, 0, 0, 0, 0, 27827, 0, 0, 0, 0, 0, 10379, 0, 14941, 0, 0, 0, 0, 0, 373, 0, 0, 0, 16031, 0, 13627, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16547, 0, 0, 0, 0, 0, 12869, 0, 0, 0, 0, 0, 0, 0, 22063, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7823, 0, 1129, 0, 0, 0, 16787, 0, 0, 0, 0, 0, 0, 0, 11071, 0, 0, 0, 0, 0, 19363, 0, 0, 0, 1151, 0, 0, 0, 0, 0, 14513, 0, 28279, 0, 0, 0, 14327, 0, 0, 0, 0, 0, 4799, 0, 25951, 0, 0, 0, 0, 0, 6403, 0, 0, 0, 0, 0, 15397, 0, 0, 0, 0, 0, 9829, 0, 0, 0, 22277, 0, 0, 0, 0, 0, 8339, 0, 0, 0, 0, 0, 0, 0, 18133, 0, 0, 0, 19751, 0, 20617, 0, 0, 0, 5573, 0, 16609, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13001, 0, 9967, 0, 0, 0, 2693, 0, 17839, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3599, 0, 18751, 0, 0, 0, 0, 0, 18253, 0, 0, 0, 11591, 0, 18397, 0, 0, 0, 19883, 0, 0, 0, 0, 0, 2207, 0, 0, 0, 0, 0, 719, 0, 18511, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6581, 0, 9847, 0, 0, 0, 0, 0, 7459, 0, 0, 0, 13187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12613, 0, 0, 0, 0, 0, 28087, 0, 0, 0, 22943, 0, 16879, 0, 0, 0, 3977, 0, 0, 0, 0, 0, 0, 0, 21361, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19211, 0, 0, 0, 0, 0, 21233, 0, 29119, 0, 0, 0, 1517, 0, 0, 0, 0, 0, 19919, 0, 24901, 0, 0, 0, 0, 0, 19003, 0, 0, 0, 0, 0, 22117, 0, 0, 0, 10583, 0, 2479, 0, 0, 0, 18917, 0, 0, 0, 0, 0, 0, 0, 22861, 0, 0, 0, 0, 0, 28423, 0, 0, 0, 12401, 0, 12637, 0, 0, 0, 25943, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16169, 0, 9241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24877, 0, 0, 0, 15083, 0, 22249, 0, 0, 0, 24707, 0, 0, 0, 0, 0, 239, 0, 9091, 0, 0, 0, 0, 0, 27283, 0, 0, 0, 13691, 0, 17137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6407, 0, 0, 0, 0, 0, 9749, 0, 7801, 0, 0, 0, 0, 0, 14653, 0, 0, 0, 5531, 0, 0, 0, 0, 0, 0, 0, 6199, 0, 0, 0, 11717, 0, 0, 0, 0, 0, 24839, 0, 0, 0, 0, 0, 0, 0, 643, 0, 0, 0, 6221, 0, 2467, 0, 0, 0, 25043, 0, 1759, 0, 0, 0, 15107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9913, 0, 0, 0, 17951, 0, 0, 0, 0, 0, 24803, 0, 21139, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16139, 0, 0, 0, 0, 0, 0, 0, 23203, 0, 0, 0, 0, 0, 14767, 0, 0, 0, 27803, 0, 9199, 0, 0, 0, 9467, 0, 0, 0, 0, 0, 4349, 0, 21811, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9881, 0, 13477, 0, 0, 0, 6203, 0, 14719, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19531, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4181, 0, 10597, 0, 0, 0, 14033, 0, 0, 0, 0, 0, 7277, 0, 0, 0, 0, 0, 6959, 0, 26311, 0, 0, 0, 0, 0, 28543, 0, 0, 0, 17891, 0, 0, 0, 0, 0, 7703, 0, 0, 0, 0, 0, 16697, 0, 0, 0, 0, 0, 1769, 0, 11371, 0, 0, 0, 0, 0, 13393, 0, 0, 0, 0, 0, 8587, 0, 0, 0, 0, 0, 21949, 0, 0, 0, 15287, 0, 0, 0, 0, 0, 9719, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24281, 0, 17587, 0, 0, 0, 15383, 0, 0, 0, 0, 0, 24137, 0, 0, 0, 0, 0, 0, 0, 21781, 0, 0, 0, 0, 0, 8863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14093, 0, 139, 0, 0, 0, 15797, 0, 0, 0, 0, 0, 0, 0, 12721, 0, 0, 0, 0, 0, 25303, 0, 0, 0, 0, 0, 20437, 0, 0, 0, 18143, 0, 5839, 0, 0, 0, 27527, 0, 0, 0, 0, 0, 8759, 0, 4171, 0, 0, 0, 0, 0, 19603, 0, 0, 0, 13451, 0, 14737, 0, 0, 0, 0, 0, 27319, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14279, 0, 9871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14471, 0, 1147, 0, 0, 0, 0, 0, 18259, 0, 0, 0, 11477, 0, 0, 0, 0, 0, 29639, 0, 0, 0, 0, 0, 0, 0, 22453, 0, 0, 0, 461, 0, 0, 0, 0, 0, 6653, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2609, 0, 18091, 0, 0, 0, 0, 0, 1423, 0, 0, 0, 26111, 0, 26647, 0, 0, 0, 0, 0, 28669, 0, 0, 0, 12767, 0, 0, 0, 0, 0, 25469, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14831, 0, 11497, 0, 0, 0, 2573, 0, 13729, 0, 0, 0, 25397, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29443, 0, 0, 0, 2201, 0, 0, 0, 0, 0, 17663, 0, 20509, 0, 0, 0, 14537, 0, 0, 0, 0, 0, 14879, 0, 25321, 0, 0, 0, 0, 0, 26563, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20243, 0, 18229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23879, 0, 811, 0, 0, 0, 0, 0, 25273, 0, 0, 0, 11981, 0, 16837, 0, 0, 0, 6623, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29189, 0, 18901, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18671, 0, 97, 0, 0, 0, 1853, 0, 0, 0, 0, 0, 13577, 0, 0, 0, 0, 0, 5279, 0, 8251, 0, 0, 0, 0, 0, 7543, 0, 0, 0, 0, 0, 9367, 0, 0, 0, 19043, 0, 0, 0, 0, 0, 7877, 0, 0, 0, 0, 0, 3869, 0, 3811, 0, 0, 0, 0, 0, 0, 0, 0, 0, 491, 0, 25387, 0, 0, 0, 0, 0, 2209, 0, 0, 0, 5417, 0, 0, 0, 0, 0, 16019, 0, 0, 0, 0, 0, 0, 0, 22903, 0, 0, 0, 11471, 0, 12757, 0, 0, 0, 0, 0, 919, 0, 0, 0, 19307, 0, 0, 0, 0, 0, 0, 0, 17791, 0, 0, 0, 0, 0, 15163, 0, 0, 0, 19421, 0, 0, 0, 0, 0, 24383, 0, 769, 0, 0, 0, 2567, 0, 0, 0, 0, 0, 2909, 0, 0, 0, 0, 0, 0, 0, 27403, 0, 0, 0, 0, 0, 11617, 0, 0, 0, 5333, 0, 17179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17789, 0, 7531, 0, 0, 0, 0, 0, 17923, 0, 0, 0, 8621, 0, 21037, 0, 0, 0, 0, 0, 15139, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21839, 0, 10921, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23081, 0, 0, 0, 0, 0, 14453, 0, 18679, 0, 0, 0, 14837, 0, 0, 0, 0, 0, 0, 0, 23161, 0, 0, 0, 0, 0, 11953, 0, 0, 0, 10121, 0, 27427, 0, 0, 0, 15683, 0, 0, 0, 0, 0, 13547, 0, 0, 0, 0, 0, 5969, 0, 2551, 0, 0, 0, 0, 0, 163, 0, 0, 0, 4691, 0, 0, 0, 0, 0, 0, 0, 21529, 0, 0, 0, 20327, 0, 0, 0, 0, 0, 14969, 0, 0, 0, 0, 0, 0, 0, 21643, 0, 0, 0, 10001, 0, 21787, 0, 0, 0, 14543, 0, 21289, 0, 0, 0, 4397, 0, 0, 0, 0, 0, 0, 0, 22201, 0, 0, 0, 0, 0, 43, 0, 0, 0, 521, 0, 0, 0, 0, 0, 10103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1649, 0, 23431, 0, 0, 0, 0, 0, 19423, 0, 0, 0, 0, 0, 21907, 0, 0, 0, 1553, 0, 0, 0, 0, 0, 19757, 0, 0, 0, 0, 0, 19049, 0, 181, 0, 0, 0, 0, 0, 24643, 0, 0, 0, 29201, 0, 3607, 0, 0, 0, 2633, 0, 14089, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19319, 0, 11761, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20677, 0, 0, 0, 0, 0, 28969, 0, 0, 0, 15677, 0, 0, 0, 0, 0, 26489, 0, 8881, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22721, 0, 17977, 0, 0, 0, 22403, 0, 0, 0, 0, 0, 6197, 0, 0, 0, 0, 0, 10169, 0, 0, 0, 0, 0, 0, 0, 26413, 0, 0, 0, 14981, 0, 9637, 0, 0, 0, 0, 0, 25099, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7363, 0, 0, 0, 13571, 0, 2887, 0, 0, 0, 29453, 0, 23389, 0, 0, 0, 8807, 0, 0, 0, 0, 0, 0, 0, 7291, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9551, 0, 0, 0, 0, 0, 8843, 0, 1189, 0, 0, 0, 1307, 0, 0, 0, 0, 0, 0, 0, 8311, 0, 0, 0, 0, 0, 28453, 0, 0, 0, 0, 0, 26107, 0, 0, 0, 0, 0, 4999, 0, 0, 0, 18497, 0, 0, 0, 0, 0, 28079, 0, 5851, 0, 0, 0, 0, 0, 21283, 0, 0, 0, 17231, 0, 0, 0, 0, 0, 7043, 0, 26479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22889, 0, 13021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13631, 0, 25087, 0, 0, 0, 12563, 0, 19309, 0, 0, 0, 7697, 0, 0, 0, 0, 0, 6749, 0, 29461, 0, 0, 0, 0, 0, 7963, 0, 0, 0, 26921, 0, 14617, 0, 0, 0, 15053, 0, 0, 0, 0, 0, 12917, 0, 0, 0, 0, 0, 22769, 0, 0, 0, 0, 0, 0, 0, 11503, 0, 0, 0, 0, 0, 26857, 0, 0, 0, 0, 0, 1789, 0, 0, 0, 18437, 0, 0, 0, 0, 0, 5519, 0, 0, 0, 0, 0, 0, 0, 14083, 0, 0, 0, 11051, 0, 20107, 0, 0, 0, 15173, 0, 8479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1201, 0, 0, 0, 0, 0, 12013, 0, 0, 0, 10811, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27557, 0, 0, 0, 0, 0, 15929, 0, 961, 0, 0, 0, 0, 0, 18793, 0, 0, 0, 0, 0, 487, 0, 0, 0, 293, 0, 17599, 0, 0, 0, 4217, 0, 0, 0, 0, 0, 9179, 0, 0, 0, 0, 0, 0, 0, 3643, 0, 0, 0, 0, 0, 13267, 0, 0, 0, 22163, 0, 23119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21419, 0, 15121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12581, 0, 29287, 0, 0, 0, 27473, 0, 28339, 0, 0, 0, 16727, 0, 0, 0, 0, 0, 10949, 0, 28411, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29021, 0, 13147, 0, 0, 0, 20723, 0, 0, 0, 0, 0, 9557, 0, 0, 0, 0, 0, 0, 0, 24811, 0, 0, 0, 0, 0, 13603, 0, 0, 0, 6161, 0, 7957, 0, 0, 0, 0, 0, 17539, 0, 0, 0, 1007, 0, 0, 0, 0, 0, 9929, 0, 0, 0, 0, 0, 0, 0, 17653, 0, 0, 0, 3701, 0, 0, 0, 0, 0, 5723, 0, 4699, 0, 0, 0, 4817, 0, 0, 0, 0, 0, 0, 0, 2461, 0, 0, 0, 0, 0, 29233, 0, 0, 0, 4721, 0, 0, 0, 0, 0, 0, 0, 27229, 0, 0, 0, 12647, 0, 0, 0, 0, 0, 3119, 0, 28471, 0, 0, 0, 0, 0, 29083, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18353, 0, 0, 0, 0, 0, 10937, 0, 0, 0, 0, 0, 26399, 0, 27061, 0, 0, 0, 0, 0, 2593, 0, 0, 0, 0, 0, 25867, 0, 0, 0, 16073, 0, 29839, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18059, 0, 19321, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3131, 0, 9547, 0, 0, 0, 20123, 0, 20359, 0, 0, 0, 7067, 0, 0, 0, 0, 0, 15359, 0, 0, 0, 0, 0, 0, 0, 8383, 0, 0, 0, 251, 0, 16717, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21947, 0, 0, 0, 0, 0, 2399, 0, 12841, 0, 0, 0, 0, 0, 3943, 0, 0, 0, 11831, 0, 16987, 0, 0, 0, 0, 0, 16279, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14129, 0, 0, 0, 0, 0, 0, 0, 6943, 0, 0, 0, 18611, 0, 4357, 0, 0, 0, 4673, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11491, 0, 0, 0, 0, 0, 19573, 0, 0, 0, 19841, 0, 0, 0, 0, 0, 1073, 0, 1609, 0, 0, 0, 14747, 0, 0, 0, 0, 0, 14249, 0, 2011, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21067, 0, 0, 0, 17093, 0, 15709, 0, 0, 0, 14507, 0, 0, 0, 0, 0, 0, 0, 7321, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9461, 0, 37, 0, 0, 0, 17333, 0, 22489, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8609, 0, 1891, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18881, 0, 0, 0, 0, 0, 17603, 0, 21199, 0, 0, 0, 17357, 0, 0, 0, 0, 0, 0, 0, 18961, 0, 0, 0, 0, 0, 18673, 0, 0, 0, 1091, 0, 14197, 0, 0, 0, 11903, 0, 0, 0, 0, 0, 20897, 0, 0, 0, 0, 0, 14999, 0, 3391, 0, 0, 0, 0, 0, 21163, 0, 0, 0, 4481, 0, 18247, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18857, 0, 0, 0, 0, 0, 24419, 0, 0, 0, 0, 0, 0, 0, 10513, 0, 0, 0, 0, 0, 3097, 0, 0, 0, 0, 0, 3439, 0, 0, 0, 5237, 0, 0, 0, 0, 0, 0, 0, 22621, 0, 0, 0, 0, 0, 21673, 0, 0, 0, 24251, 0, 0, 0, 0, 0, 19133, 0, 16729, 0, 0, 0, 5087, 0, 0, 0, 0, 0, 23279, 0, 541, 0, 0, 0, 0, 0, 29503, 0, 0, 0, 0, 0, 20017, 0, 0, 0, 0, 0, 589, 0, 0, 0, 3797, 0, 0, 0, 0, 0, 10649, 0, 11521, 0, 0, 0, 0, 0, 20443, 0, 0, 0, 8201, 0, 2137, 0, 0, 0, 26363, 0, 28159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26669, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3761, 0, 727, 0, 0, 0, 21173, 0, 22459, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5909, 0, 23371, 0, 0, 0, 0, 0, 9013, 0, 0, 0, 23141, 0, 6847, 0, 0, 0, 22193, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19199, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11201, 0, 12157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17807, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17233, 0, 0, 0, 16721, 0, 21157, 0, 0, 0, 25253, 0, 21499, 0, 0, 0, 15527, 0, 0, 0, 0, 0, 0, 0, 9811, 0, 0, 0, 0, 0, 6763, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9683, 0, 10639, 0, 0, 0, 22307, 0, 0, 0, 0, 0, 24539, 0, 27211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10567, 0, 0, 0, 1343, 0, 23269, 0, 0, 0, 7367, 0, 0, 0, 0, 0, 9389, 0, 15931, 0, 0, 0, 0, 0, 3013, 0, 0, 0, 23951, 0, 3397, 0, 0, 0, 7463, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11551, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6397, 0, 0, 0, 19703, 0, 24559, 0, 0, 0, 17777, 0, 0, 0, 0, 0, 4859, 0, 27571, 0, 0, 0, 0, 0, 18043, 0, 0, 0, 2141, 0, 21757, 0, 0, 0, 26393, 0, 0, 0, 0, 0, 8717, 0, 0, 0, 0, 0, 21299, 0, 28591, 0, 0, 0, 0, 0, 3103, 0, 0, 0, 7841, 0, 27277, 0, 0, 0, 0, 0, 8509, 0, 0, 0, 167, 0, 0, 0, 0, 0, 15599, 0, 0, 0, 0, 0, 0, 0, 2953, 0, 0, 0, 29321, 0, 0, 0, 0, 0, 29663, 0, 20239, 0, 0, 0, 5867, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2983, 0, 0, 0, 20261, 0, 0, 0, 0, 0, 6323, 0, 11899, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18449, 0, 17761, 0, 0, 0, 0, 0, 25513, 0, 0, 0, 0, 0, 26317, 0, 0, 0, 18563, 0, 20749, 0, 0, 0, 14087, 0, 0, 0, 0, 0, 27449, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 641, 0, 4237, 0, 0, 0, 24683, 0, 19339, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24151, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15731, 0, 29077, 0, 0, 0, 16343, 0, 28759, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25439, 0, 7831, 0, 0, 0, 0, 0, 10063, 0, 0, 0, 22511, 0, 0, 0, 0, 0, 773, 0, 0, 0, 0, 0, 21317, 0, 0, 0, 0, 0, 22559, 0, 2131, 0, 0, 0, 0, 0, 18013, 0, 0, 0, 20231, 0, 1657, 0, 0, 0, 0, 0, 26569, 0, 0, 0, 26837, 0, 0, 0, 0, 0, 21269, 0, 0, 0, 0, 0, 0, 0, 1693, 0, 0, 0, 0, 0, 19897, 0, 0, 0, 3833, 0, 0, 0, 0, 0, 14897, 0, 0, 0, 0, 0, 0, 0, 24091, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4853, 0, 20929, 0, 0, 0, 4247, 0, 0, 0, 0, 0, 3539, 0, 5791, 0, 0, 0, 0, 0, 29923, 0, 0, 0, 0, 0, 11197, 0, 0, 0, 29693, 0, 24319, 0, 0, 0, 29837, 0, 0, 0, 0, 0, 0, 0, 6481, 0, 0, 0, 0, 0, 1123, 0, 0, 0, 4211, 0, 26287, 0, 0, 0, 0, 0, 29629, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18899, 0, 28351, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2921, 0, 5767, 0, 0, 0, 6893, 0, 11329, 0, 0, 0, 13787, 0, 0, 0, 0, 0, 11159, 0, 0, 0, 0, 0, 0, 0, 10903, 0, 0, 0, 2771, 0, 19867, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23399, 0, 29641, 0, 0, 0, 0, 0, 3733, 0, 0, 0, 19181, 0, 0, 0, 0, 0, 0, 0, 17119, 0, 0, 0, 3527, 0, 0, 0, 0, 0, 13919, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17141, 0, 16117, 0, 0, 0, 14123, 0, 4489, 0, 0, 0, 6917, 0, 0, 0, 0, 0, 0, 0, 22831, 0, 0, 0, 0, 0, 1723, 0, 0, 0, 20681, 0, 0, 0, 0, 0, 8423, 0, 2029, 0, 0, 0, 19157, 0, 0, 0, 0, 0, 7949, 0, 0, 0, 0, 0, 0, 0, 15013, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8693, 0, 22849, 0, 0, 0, 28577, 0, 0, 0, 0, 0, 26189, 0, 5431, 0, 0, 0, 0, 0, 16033, 0, 0, 0, 23531, 0, 5287, 0, 0, 0, 8933, 0, 3799, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17639, 0, 421, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23291, 0, 2407, 0, 0, 0, 24953, 0, 0, 0, 0, 0, 18197, 0, 0, 0, 0, 0, 26069, 0, 29041, 0, 0, 0, 0, 0, 12163, 0, 0, 0, 0, 0, 2437, 0, 0, 0, 21353, 0, 0, 0, 0, 0, 19427, 0, 0, 0, 0, 0, 15419, 0, 22291, 0, 0, 0, 0, 0, 24313, 0, 0, 0, 0, 0, 27697, 0, 0, 0, 0, 0, 13759, 0, 0, 0, 26207, 0, 0, 0, 0, 0, 20639, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16091, 0, 1207, 0, 0, 0, 0, 0, 21709, 0, 0, 0, 7757, 0, 0, 0, 0, 0, 0, 0, 10861, 0, 0, 0, 0, 0, 19783, 0, 0, 0, 941, 0, 0, 0, 0, 0, 5903, 0, 19249, 0, 0, 0, 4877, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8923, 0, 0, 0, 0, 0, 23167, 0, 0, 0, 9953, 0, 19489, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22409, 0, 26011, 0, 0, 0, 0, 0, 8683, 0, 0, 0, 27101, 0, 25657, 0, 0, 0, 7673, 0, 8209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3359, 0, 1681, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25391, 0, 14797, 0, 0, 0, 0, 0, 20989, 0, 0, 0, 3287, 0, 0, 0, 0, 0, 0, 0, 4681, 0, 0, 0, 0, 0, 14263, 0, 0, 0, 3191, 0, 0, 0, 0, 0, 20303, 0, 0, 0, 0, 0, 4307, 0, 0, 0, 0, 0, 24449, 0, 7171, 0, 0, 0, 0, 0, 23263, 0, 0, 0, 7001, 0, 0, 0, 0, 0, 0, 0, 12289, 0, 0, 0, 1847, 0, 0, 0, 0, 0, 17279, 0, 0, 0, 0, 0, 0, 0, 23953, 0, 0, 0, 28481, 0, 3307, 0, 0, 0, 5303, 0, 2809, 0, 0, 0, 9017, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18523, 0, 0, 0, 21311, 0, 0, 0, 0, 0, 28583, 0, 0, 0, 0, 0, 25457, 0, 0, 0, 0, 0, 3959, 0, 28051, 0, 0, 0, 0, 0, 10183, 0, 0, 0, 0, 0, 10357, 0, 0, 0, 3863, 0, 10039, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7499, 0, 11731, 0, 0, 0, 0, 0, 6163, 0, 0, 0, 3791, 0, 5917, 0, 0, 0, 25733, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10079, 0, 2521, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10481, 0, 13747, 0, 0, 0, 0, 0, 3559, 0, 0, 0, 27227, 0, 0, 0, 0, 0, 8009, 0, 27361, 0, 0, 0, 0, 0, 18463, 0, 0, 0, 0, 0, 20287, 0, 0, 0, 10853, 0, 0, 0, 0, 0, 26987, 0, 0, 0, 0, 0, 14789, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19601, 0, 28117, 0, 0, 0, 0, 0, 15859, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29669, 0, 0, 0, 0, 0, 0, 0, 11983, 0, 0, 0, 25121, 0, 23677, 0, 0, 0, 10973, 0, 11839, 0, 0, 0, 11117, 0, 0, 0, 0, 0, 0, 0, 9601, 0, 0, 0, 0, 0, 0, 0, 0, 0, 311, 0, 0, 0, 0, 0, 13463, 0, 3499, 0, 0, 0, 24407, 0, 0, 0, 0, 0, 0, 0, 26791, 0, 0, 0, 0, 0, 19213, 0, 0, 0, 0, 0, 697, 0, 0, 0, 18983, 0, 28099, 0, 0, 0, 20807, 0, 0, 0, 0, 0, 9599, 0, 26641, 0, 0, 0, 0, 0, 9733, 0, 0, 0, 19541, 0, 0, 0, 0, 0, 0, 0, 28789, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13649, 0, 24571, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6701, 0, 0, 0, 0, 0, 17183, 0, 7759, 0, 0, 0, 28487, 0, 0, 0, 0, 0, 25229, 0, 4051, 0, 0, 0, 0, 0, 1033, 0, 0, 0, 29231, 0, 19237, 0, 0, 0, 26603, 0, 0, 0, 0, 0, 24467, 0, 0, 0, 0, 0, 25079, 0, 5281, 0, 0, 0, 0, 0, 13813, 0, 0, 0, 0, 0, 8377, 0, 0, 0, 0, 0, 13339, 0, 0, 0, 23057, 0, 0, 0, 0, 0, 28619, 0, 0, 0, 0, 0, 0, 0, 2533, 0, 0, 0, 1811, 0, 10867, 0, 0, 0, 3623, 0, 13099, 0, 0, 0, 15317, 0, 0, 0, 0, 0, 0, 0, 5821, 0, 0, 0, 0, 0, 2773, 0, 0, 0, 22361, 0, 0, 0, 0, 0, 0, 0, 2239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4379, 0, 12511, 0, 0, 0, 0, 0, 313, 0, 0, 0, 0, 0, 2797, 0, 0, 0, 23393, 0, 0, 0, 0, 0, 8837, 0, 0, 0, 0, 0, 29969, 0, 0, 0, 0, 0, 0, 0, 8263, 0, 0, 0, 18281, 0, 6337, 0, 0, 0, 24473, 0, 27739, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2939, 0, 3571, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1567, 0, 0, 0, 15923, 0, 9859, 0, 0, 0, 7487, 0, 0, 0, 0, 0, 15569, 0, 691, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3611, 0, 1597, 0, 0, 0, 11483, 0, 0, 0, 0, 0, 28037, 0, 0, 0, 0, 0, 0, 0, 17881, 0, 0, 0, 0, 0, 18223, 0, 0, 0, 17711, 0, 28747, 0, 0, 0, 0, 0, 5989, 0, 0, 0, 3317, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29203, 0, 0, 0, 24491, 0, 0, 0, 0, 0, 10343, 0, 7009, 0, 0, 0, 27917, 0, 0, 0, 0, 0, 0, 0, 20941, 0, 0, 0, 0, 0, 19993, 0, 0, 0, 23201, 0, 0, 0, 0, 0, 11573, 0, 20299, 0, 0, 0, 14957, 0, 0, 0, 0, 0, 14669, 0, 19231, 0, 0, 0, 0, 0, 17533, 0, 0, 0, 0, 0, 17917, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29417, 0, 0, 0, 0, 0, 17159, 0, 8581, 0, 0, 0, 0, 0, 4903, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20693, 0, 18289, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6509, 0, 23941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5441, 0, 14167, 0, 0, 0, 1643, 0, 11119, 0, 0, 0, 18617, 0, 0, 0, 0, 0, 17669, 0, 13081, 0, 0, 0, 0, 0, 10693, 0, 0, 0, 18731, 0, 28267, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26567, 0, 0, 0, 0, 0, 25499, 0, 0, 0, 0, 0, 0, 0, 22423, 0, 0, 0, 2591, 0, 7747, 0, 0, 0, 0, 0, 20899, 0, 0, 0, 7517, 0, 0, 0, 0, 0, 16439, 0, 0, 0, 0, 0, 0, 0, 27733, 0, 0, 0, 131, 0, 22837, 0, 0, 0, 6983, 0, 22129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23041, 0, 0, 0, 0, 0, 1093, 0, 0, 0, 24461, 0, 0, 0, 0, 0, 24173, 0, 0, 0, 0, 0, 19367, 0, 0, 0, 0, 0, 5009, 0, 22801, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14137, 0, 0, 0, 19403, 0, 20329, 0, 0, 0, 26057, 0, 0, 0, 0, 0, 0, 0, 25801, 0, 0, 0, 0, 0, 25483, 0, 0, 0, 0, 0, 2347, 0, 0, 0, 5783, 0, 4009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13229, 0, 4201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23501, 0, 0, 0, 0, 0, 8363, 0, 11959, 0, 0, 0, 5807, 0, 0, 0, 0, 0, 2759, 0, 12031, 0, 0, 0, 0, 0, 23293, 0, 0, 0, 12641, 0, 4957, 0, 0, 0, 23453, 0, 0, 0, 0, 0, 23207, 0, 0, 0, 0, 0, 0, 0, 5701, 0, 0, 0, 0, 0, 2683, 0, 0, 0, 25271, 0, 29797, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11927, 0, 0, 0, 0, 0, 29039, 0, 0, 0, 0, 0, 0, 0, 1273, 0, 0, 0, 0, 0, 7717, 0, 0, 0, 19373, 0, 26539, 0, 0, 0, 7547, 0, 0, 0, 0, 0, 0, 0, 13381, 0, 0, 0, 0, 0, 10123, 0, 0, 0, 26561, 0, 0, 0, 0, 0, 0, 0, 19039, 0, 0, 0, 23567, 0, 0, 0, 0, 0, 14039, 0, 12091, 0, 0, 0, 0, 0, 1783, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19069, 0, 0, 0, 24587, 0, 0, 0, 0, 0, 29129, 0, 16141, 0, 0, 0, 0, 0, 13513, 0, 0, 0, 10511, 0, 6757, 0, 0, 0, 7883, 0, 18919, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28979, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22241, 0, 12277, 0, 0, 0, 11933, 0, 3979, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29009, 0, 0, 0, 0, 0, 0, 0, 27493, 0, 0, 0, 13901, 0, 27637, 0, 0, 0, 10643, 0, 0, 0, 0, 0, 13757, 0, 0, 0, 0, 0, 21509, 0, 4651, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22751, 0, 607, 0, 0, 0, 0, 0, 19009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28783, 0, 0, 0, 21341, 0, 23467, 0, 0, 0, 18323, 0, 0, 0, 0, 0, 20147, 0, 0, 0, 0, 0, 0, 0, 571, 0, 0, 0, 0, 0, 11383, 0, 0, 0, 731, 0, 0, 0, 0, 0, 11993, 0, 15259, 0, 0, 0, 3827, 0, 0, 0, 0, 0, 6059, 0, 15661, 0, 0, 0, 0, 0, 523, 0, 0, 0, 0, 0, 12877, 0, 0, 0, 19823, 0, 4789, 0, 0, 0, 28157, 0, 0, 0, 0, 0, 14009, 0, 18241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28571, 0, 21877, 0, 0, 0, 28253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27719, 0, 4621, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27187, 0, 0, 0, 1223, 0, 13009, 0, 0, 0, 20087, 0, 0, 0, 0, 0, 0, 0, 29881, 0, 0, 0, 0, 0, 29593, 0, 0, 0, 22931, 0, 3277, 0, 0, 0, 983, 0, 0, 0, 0, 0, 1787, 0, 0, 0, 0, 0, 25919, 0, 17041, 0, 0, 0, 0, 0, 23893, 0, 0, 0, 26321, 0, 1867, 0, 0, 0, 0, 0, 1579, 0, 0, 0, 2477, 0, 0, 0, 0, 0, 27149, 0, 0, 0, 0, 0, 0, 0, 21433, 0, 0, 0, 1601, 0, 14017, 0, 0, 0, 0, 0, 22549, 0, 0, 0, 24347, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5293, 0, 0, 0, 13331, 0, 0, 0, 0, 0, 10943, 0, 349, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6899, 0, 22381, 0, 0, 0, 0, 0, 18583, 0, 0, 0, 0, 0, 907, 0, 0, 0, 83, 0, 11509, 0, 0, 0, 25637, 0, 0, 0, 0, 0, 29759, 0, 3331, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19121, 0, 15787, 0, 0, 0, 15443, 0, 859, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6491, 0, 19837, 0, 0, 0, 4793, 0, 3349, 0, 0, 0, 18827, 0, 0, 0, 0, 0, 27749, 0, 12451, 0, 0, 0, 0, 0, 823, 0, 0, 0, 4031, 0, 0, 0, 0, 0, 3083, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11009, 0, 13681, 0, 0, 0, 0, 0, 29563, 0, 0, 0, 24851, 0, 3967, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1427, 0, 0, 0, 0, 0, 12029, 0, 0, 0, 0, 0, 0, 0, 6313, 0, 0, 0, 5801, 0, 12967, 0, 0, 0, 6143, 0, 0, 0, 0, 0, 26447, 0, 0, 0, 0, 0, 0, 0, 12541, 0, 0, 0, 0, 0, 20413, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23333, 0, 2449, 0, 0, 0, 25037, 0, 0, 0, 0, 0, 8159, 0, 8101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29677, 0, 0, 0, 20453, 0, 15079, 0, 0, 0, 18287, 0, 0, 0, 0, 0, 20309, 0, 29581, 0, 0, 0, 0, 0, 5743, 0, 0, 0, 15761, 0, 17047, 0, 0, 0, 0, 0, 18079, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23711, 0, 17317, 0, 0, 0, 11513, 0, 13639, 0, 0, 0, 6857, 0, 0, 0, 0, 0, 15779, 0, 0, 0, 0, 0, 0, 0, 1663, 0, 0, 0, 21251, 0, 24487, 0, 0, 0, 18203, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4919, 0, 20401, 0, 0, 0, 0, 0, 22213, 0, 0, 0, 21491, 0, 8167, 0, 0, 0, 0, 0, 19429, 0, 0, 0, 22007, 0, 0, 0, 0, 0, 4679, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10211, 0, 0, 0, 0, 0, 18743, 0, 22969, 0, 0, 0, 27707, 0, 0, 0, 0, 0, 0, 0, 27451, 0, 0, 0, 0, 0, 24823, 0, 0, 0, 22991, 0, 0, 0, 0, 0, 19973, 0, 22819, 0, 0, 0, 677, 0, 0, 0, 0, 0, 10259, 0, 6841, 0, 0, 0, 0, 0, 17323, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29483, 0, 4369, 0, 0, 0, 3167, 0, 0, 0, 0, 0, 19259, 0, 0, 0, 0, 0, 0, 0, 4483, 0, 0, 0, 14291, 0, 26077, 0, 0, 0, 27413, 0, 8419, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19949, 0, 5041, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4811, 0, 20887, 0, 0, 0, 27263, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14519, 0, 10561, 0, 0, 0, 0, 0, 23713, 0, 0, 0, 0, 0, 4747, 0, 0, 0, 14423, 0, 0, 0, 0, 0, 24047, 0, 0, 0, 0, 0, 6179, 0, 13051, 0, 0, 0, 0, 0, 28933, 0, 0, 0, 12041, 0, 20767, 0, 0, 0, 0, 0, 18379, 0, 0, 0, 7727, 0, 0, 0, 0, 0, 2159, 0, 0, 0, 0, 0, 0, 0, 4423, 0, 0, 0, 0, 0, 3517, 0, 0, 0, 0, 0, 3229, 0, 0, 0, 28547, 0, 0, 0, 0, 0, 0, 0, 13171, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5561, 0, 0, 0, 0, 0, 26693, 0, 10009, 0, 0, 0, 23357, 0, 0, 0, 0, 0, 14459, 0, 0, 0, 0, 0, 0, 0, 13543, 0, 0, 0, 0, 0, 13927, 0, 0, 0, 21503, 0, 7939, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28321, 0, 0, 0, 0, 0, 20233, 0, 0, 0, 17861, 0, 7177, 0, 0, 0, 12293, 0, 10519, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7979, 0, 20161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13841, 0, 19417, 0, 0, 0, 26003, 0, 14059, 0, 0, 0, 5597, 0, 0, 0, 0, 0, 0, 0, 25471, 0, 0, 0, 0, 0, 2713, 0, 0, 0, 5501, 0, 8947, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22787, 0, 0, 0, 0, 0, 15209, 0, 18721, 0, 0, 0, 0, 0, 25573, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 739, 0, 0, 0, 22637, 0, 0, 0, 0, 0, 5729, 0, 0, 0, 0, 0, 0, 0, 17023, 0, 0, 0, 761, 0, 7927, 0, 0, 0, 16853, 0, 23599, 0, 0, 0, 20567, 0, 0, 0, 0, 0, 0, 0, 3721, 0, 0, 0, 0, 0, 20833, 0, 0, 0, 9761, 0, 0, 0, 0, 0, 19343, 0, 0, 0, 0, 0, 47, 0, 0, 0, 0, 0, 27059, 0, 0, 0, 0, 0, 0, 0, 28663, 0, 0, 0, 0, 0, 1117, 0, 0, 0, 22343, 0, 14659, 0, 0, 0, 1277, 0, 0, 0, 0, 0, 9809, 0, 16351, 0, 0, 0, 0, 0, 26953, 0, 0, 0, 15341, 0, 24397, 0, 0, 0, 28043, 0, 25639, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28559, 0, 14071, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15101, 0, 16057, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1817, 0, 0, 0, 0, 0, 28799, 0, 18121, 0, 0, 0, 0, 0, 23083, 0, 0, 0, 4241, 0, 13357, 0, 0, 0, 13163, 0, 0, 0, 0, 0, 8507, 0, 0, 0, 0, 0, 26339, 0, 0, 0, 0, 0, 0, 0, 7933, 0, 0, 0, 0, 0, 397, 0, 0, 0, 0, 0, 27409, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4259, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29741, 0, 12127, 0, 0, 0, 1733, 0, 2599, 0, 0, 0, 29597, 0, 0, 0, 0, 0, 0, 0, 2671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11861, 0, 0, 0, 0, 0, 25013, 0, 21979, 0, 0, 0, 26717, 0, 0, 0, 0, 0, 0, 0, 29101, 0, 0, 0, 0, 0, 733, 0, 0, 0, 0, 0, 12247, 0, 0, 0, 23603, 0, 379, 0, 0, 0, 13877, 0, 0, 0, 0, 0, 14219, 0, 15091, 0, 0, 0, 0, 0, 493, 0, 0, 0, 7991, 0, 0, 0, 0, 0, 18593, 0, 21859, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25199, 0, 15331, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9011, 0, 6607, 0, 0, 0, 0, 0, 10069, 0, 0, 0, 16937, 0, 0, 0, 0, 0, 15989, 0, 15601, 0, 0, 0, 0, 0, 3343, 0, 0, 0, 22301, 0, 0, 0, 0, 0, 1193, 0, 0, 0, 0, 0, 15227, 0, 0, 0, 0, 0, 13529, 0, 9901, 0, 0, 0, 0, 0, 6883, 0, 0, 0, 0, 0, 12997, 0, 0, 0, 0, 0, 4099, 0, 0, 0, 4577, 0, 0, 0, 0, 0, 899, 0, 0, 0, 0, 0, 0, 0, 4843, 0, 0, 0, 20291, 0, 22417, 0, 0, 0, 24413, 0, 24649, 0, 0, 0, 19937, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21253, 0, 0, 0, 13121, 0, 0, 0, 0, 0, 0, 0, 6859, 0, 0, 0, 9077, 0, 0, 0, 0, 0, 6689, 0, 17131, 0, 0, 0, 0, 0, 21103, 0, 0, 0, 0, 0, 21277, 0, 0, 0, 25703, 0, 29149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18419, 0, 0, 0, 0, 0, 0, 0, 19813, 0, 0, 0, 22901, 0, 8647, 0, 0, 0, 17543, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23729, 0, 24361, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24131, 0, 24667, 0, 0, 0, 18233, 0, 14479, 0, 0, 0, 19037, 0, 0, 0, 0, 0, 27119, 0, 19171, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3907, 0, 0, 0, 29963, 0, 0, 0, 0, 0, 18797, 0, 0, 0, 0, 0, 0, 0, 20191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22331, 0, 17197, 0, 0, 0, 0, 0, 26779, 0, 0, 0, 21797, 0, 0, 0, 0, 0, 21479, 0, 0, 0, 0, 0, 0, 0, 3793, 0, 0, 0, 6011, 0, 0, 0, 0, 0, 21893, 0, 25489, 0, 0, 0, 197, 0, 0, 0, 0, 0, 0, 0, 23251, 0, 0, 0, 0, 0, 1513, 0, 0, 0, 13961, 0, 0, 0, 0, 0, 16193, 0, 22609, 0, 0, 0, 8027, 0, 0, 0, 0, 0, 19289, 0, 7681, 0, 0, 0, 0, 0, 8293, 0, 0, 0, 0, 0, 22537, 0, 0, 0, 29903, 0, 0, 0, 0, 0, 1697, 0, 0, 0, 0, 0, 28709, 0, 29371, 0, 0, 0, 0, 0, 23383, 0, 0, 0, 0, 0, 7387, 0, 0, 0, 0, 0, 20599, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27299, 0, 5461, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28541, 0, 0, 0, 0, 0, 6263, 0, 29599, 0, 0, 0, 9377, 0, 0, 0, 0, 0, 6119, 0, 17701, 0, 0, 0, 0, 0, 3763, 0, 0, 0, 21041, 0, 2857, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8087, 0, 0, 0, 0, 0, 27809, 0, 24391, 0, 0, 0, 0, 0, 24733, 0, 0, 0, 21071, 0, 19297, 0, 0, 0, 0, 0, 2419, 0, 0, 0, 12137, 0, 0, 0, 0, 0, 9509, 0, 0, 0, 0, 0, 0, 0, 16183, 0, 0, 0, 20921, 0, 13597, 0, 0, 0, 25463, 0, 26749, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27661, 0, 0, 0, 0, 0, 21883, 0, 0, 0, 5981, 0, 0, 0, 0, 0, 26483, 0, 13159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23489, 0, 4321, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16447, 0, 0, 0, 12473, 0, 0, 0, 0, 0, 647, 0, 0, 0, 0, 0, 0, 0, 16561, 0, 0, 0, 0, 0, 73, 0, 0, 0, 21011, 0, 25447, 0, 0, 0, 8093, 0, 8629, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24779, 0, 22681, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26843, 0, 23509, 0, 0, 0, 26597, 0, 0, 0, 0, 0, 7379, 0, 14341, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17261, 0, 23437, 0, 0, 0, 14213, 0, 0, 0, 0, 0, 11657, 0, 0, 0, 0, 0, 26549, 0, 28801, 0, 0, 0, 0, 0, 7303, 0, 0, 0, 6791, 0, 20557, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12823, 0, 0, 0, 0, 0, 19267, 0, 0, 0, 23993, 0, 28849, 0, 0, 0, 617, 0, 0, 0, 0, 0, 0, 0, 1831, 0, 0, 0, 0, 0, 883, 0, 0, 0, 15011, 0, 0, 0, 0, 0, 653, 0, 12109, 0, 0, 0, 25877, 0, 0, 0, 0, 0, 25589, 0, 2851, 0, 0, 0, 0, 0, 20263, 0, 0, 0, 0, 0, 1537, 0, 0, 0, 0, 0, 21379, 0, 0, 0, 13037, 0, 0, 0, 0, 0, 19889, 0, 27691, 0, 0, 0, 0, 0, 15823, 0, 0, 0, 3581, 0, 0, 0, 0, 0, 12503, 0, 7369, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17429, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24551, 0, 16897, 0, 0, 0, 23483, 0, 24769, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1289, 0, 4891, 0, 0, 0, 0, 0, 29803, 0, 0, 0, 2351, 0, 9157, 0, 0, 0, 1403, 0, 0, 0, 0, 0, 18377, 0, 0, 0, 0, 0, 14579, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13511, 0, 21397, 0, 0, 0, 0, 0, 23629, 0, 0, 0, 29357, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19543, 0, 0, 0, 2861, 0, 11917, 0, 0, 0, 20633, 0, 3019, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12121, 0, 0, 0, 0, 0, 22933, 0, 0, 0, 5351, 0, 0, 0, 0, 0, 5063, 0, 0, 0, 0, 0, 8447, 0, 0, 0, 0, 0, 26849, 0, 6421, 0, 0, 0, 0, 0, 5143, 0, 0, 0, 0, 0, 5947, 0, 0, 0, 22133, 0, 9409, 0, 0, 0, 9677, 0, 0, 0, 0, 0, 25559, 0, 6691, 0, 0, 0, 0, 0, 14563, 0, 0, 0, 0, 0, 24187, 0, 0, 0, 16703, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2309, 0, 6931, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15637, 0, 0, 0, 22013, 0, 3769, 0, 0, 0, 8537, 0, 0, 0, 0, 0, 16409, 0, 22951, 0, 0, 0, 0, 0, 4183, 0, 0, 0, 4451, 0, 24067, 0, 0, 0, 12533, 0, 0, 0, 0, 0, 4097, 0, 0, 0, 0, 0, 0, 0, 19351, 0, 0, 0, 0, 0, 5413, 0, 0, 0, 17081, 0, 13417, 0, 0, 0, 0, 0, 3889, 0, 0, 0, 25577, 0, 0, 0, 0, 0, 1739, 0, 0, 0, 0, 0, 0, 0, 12193, 0, 0, 0, 20081, 0, 18637, 0, 0, 0, 11183, 0, 15619, 0, 0, 0, 26657, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23773, 0, 0, 0, 15641, 0, 0, 0, 0, 0, 0, 0, 2659, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27689, 0, 3901, 0, 0, 0, 0, 0, 20893, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4703, 0, 29989, 0, 0, 0, 16397, 0, 0, 0, 0, 0, 18209, 0, 7951, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21431, 0, 20407, 0, 0, 0, 26993, 0, 21649, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24971, 0, 1357, 0, 0, 0, 25583, 0, 14899, 0, 0, 0, 23447, 0, 0, 0, 0, 0, 20819, 0, 0, 0, 0, 0, 0, 0, 19303, 0, 0, 0, 24821, 0, 0, 0, 0, 0, 21563, 0, 0, 0, 0, 0, 2837, 0, 0, 0, 0, 0, 13319, 0, 18301, 0, 0, 0, 0, 0, 20323, 0, 0, 0, 6371, 0, 22447, 0, 0, 0, 0, 0, 8089, 0, 0, 0, 0, 0, 0, 0, 0, 0, 479, 0, 0, 0, 0, 0, 0, 0, 17863, 0, 0, 0, 10421, 0, 15277, 0, 0, 0, 29243, 0, 0, 0, 0, 0, 1037, 0, 0, 0, 0, 0, 0, 0, 3301, 0, 0, 0, 0, 0, 25033, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25643, 0, 7069, 0, 0, 0, 6557, 0, 0, 0, 0, 0, 19709, 0, 26581, 0, 0, 0, 0, 0, 11443, 0, 0, 0, 0, 0, 1957, 0, 0, 0, 8903, 0, 26629, 0, 0, 0, 9047, 0, 0, 0, 0, 0, 24929, 0, 1861, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20381, 0, 5497, 0, 0, 0, 0, 0, 8839, 0, 0, 0, 0, 0, 0, 0, 0, 0, 419, 0, 23731, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5231, 0, 8077, 0, 0, 0, 23063, 0, 2089, 0, 0, 0, 9167, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13213, 0, 0, 0, 12011, 0, 6007, 0, 0, 0, 22823, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9539, 0, 8851, 0, 0, 0, 0, 0, 12973, 0, 0, 0, 9941, 0, 12787, 0, 0, 0, 0, 0, 12499, 0, 0, 0, 24317, 0, 0, 0, 0, 0, 16229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12521, 0, 27667, 0, 0, 0, 0, 0, 25279, 0, 0, 0, 16157, 0, 0, 0, 0, 0, 0, 0, 8971, 0, 0, 0, 0, 0, 27133, 0, 0, 0, 16061, 0, 0, 0, 0, 0, 24593, 0, 11269, 0, 0, 0, 21467, 0, 0, 0, 0, 0, 28739, 0, 11461, 0, 0, 0, 0, 0, 10393, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11003, 0, 25159, 0, 0, 0, 14717, 0, 0, 0, 0, 0, 21569, 0, 16981, 0, 0, 0, 0, 0, 6793, 0, 0, 0, 2741, 0, 7597, 0, 0, 0, 18173, 0, 19969, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13019, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25601, 0, 11647, 0, 0, 0, 15713, 0, 0, 0, 0, 0, 29747, 0, 0, 0, 0, 0, 16829, 0, 15181, 0, 0, 0, 0, 0, 14473, 0, 0, 0, 0, 0, 23227, 0, 0, 0, 16733, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24659, 0, 24601, 0, 0, 0, 0, 0, 10453, 0, 0, 0, 16661, 0, 23077, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12347, 0, 0, 0, 0, 0, 22949, 0, 0, 0, 0, 0, 0, 0, 9043, 0, 0, 0, 27641, 0, 26617, 0, 0, 0, 0, 0, 7849, 0, 0, 0, 10067, 0, 0, 0, 0, 0, 0, 0, 1621, 0, 0, 0, 0, 0, 1303, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15143, 0, 21559, 0, 0, 0, 14117, 0, 0, 0, 0, 0, 19079, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2377, 0, 0, 0, 12263, 0, 28729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3929, 0, 21391, 0, 0, 0, 0, 0, 24853, 0, 0, 0, 29411, 0, 27967, 0, 0, 0, 23843, 0, 28999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22471, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4601, 0, 937, 0, 0, 0, 593, 0, 16369, 0, 0, 0, 28697, 0, 0, 0, 0, 0, 0, 0, 13921, 0, 0, 0, 0, 0, 23503, 0, 0, 0, 23981, 0, 13567, 0, 0, 0, 1823, 0, 0, 0, 0, 0, 25097, 0, 0, 0, 0, 0, 26759, 0, 9481, 0, 0, 0, 0, 0, 14023, 0, 0, 0, 2381, 0, 0, 0, 0, 0, 0, 0, 3049, 0, 0, 0, 11087, 0, 0, 0, 0, 0, 26519, 0, 0, 0, 0, 0, 0, 0, 19333, 0, 0, 0, 23861, 0, 0, 0, 0, 0, 21473, 0, 12049, 0, 0, 0, 11327, 0, 0, 0, 0, 0, 0, 0, 8341, 0, 0, 0, 0, 0, 13903, 0, 0, 0, 12071, 0, 0, 0, 0, 0, 863, 0, 0, 0, 0, 0, 11597, 0, 0, 0, 0, 0, 29369, 0, 9571, 0, 0, 0, 0, 0, 943, 0, 0, 0, 0, 0, 12667, 0, 0, 0, 13103, 0, 26209, 0, 0, 0, 5897, 0, 0, 0, 0, 0, 2879, 0, 0, 0, 0, 0, 0, 0, 15403, 0, 0, 0, 6101, 0, 15157, 0, 0, 0, 16493, 0, 229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 839, 0, 18691, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26651, 0, 4507, 0, 0, 0, 0, 0, 15109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17249, 0, 29671, 0, 0, 0, 0, 0, 4603, 0, 0, 0, 8861, 0, 15667, 0, 0, 0, 6233, 0, 0, 0, 0, 0, 13127, 0, 0, 0, 0, 0, 17099, 0, 0, 0, 0, 0, 0, 0, 12553, 0, 0, 0, 1121, 0, 23497, 0, 0, 0, 0, 0, 1999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15809, 0, 0, 0, 0, 0, 0, 0, 23533, 0, 0, 0, 0, 0, 14437, 0, 0, 0, 20213, 0, 14149, 0, 0, 0, 20357, 0, 0, 0, 0, 0, 0, 0, 4981, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16481, 0, 0, 0, 0, 0, 15773, 0, 12739, 0, 0, 0, 15167, 0, 0, 0, 0, 0, 0, 0, 22171, 0, 0, 0, 0, 0, 5353, 0, 0, 0, 0, 0, 3007, 0, 0, 0, 5123, 0, 18859, 0, 0, 0, 16187, 0, 0, 0, 0, 0, 0, 0, 17401, 0, 0, 0, 0, 0, 12043, 0, 0, 0, 28781, 0, 0, 0, 0, 0, 23213, 0, 24169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29819, 0, 3781, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27491, 0, 11227, 0, 0, 0, 28733, 0, 3139, 0, 0, 0, 19247, 0, 0, 0, 0, 0, 27539, 0, 6361, 0, 0, 0, 0, 0, 21823, 0, 0, 0, 24611, 0, 757, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3677, 0, 0, 0, 0, 0, 4289, 0, 21451, 0, 0, 0, 0, 0, 9193, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22579, 0, 0, 0, 25367, 0, 0, 0, 0, 0, 19379, 0, 0, 0, 0, 0, 0, 0, 27943, 0, 0, 0, 22601, 0, 27037, 0, 0, 0, 5933, 0, 15409, 0, 0, 0, 1457, 0, 0, 0, 0, 0, 0, 0, 17371, 0, 0, 0, 0, 0, 23563, 0, 0, 0, 1571, 0, 0, 0, 0, 0, 0, 0, 18409, 0, 0, 0, 13697, 0, 0, 0, 0, 0, 29789, 0, 0, 0, 0, 0, 0, 0, 9553, 0, 0, 0, 0, 0, 12037, 0, 0, 0, 14153, 0, 3739, 0, 0, 0, 20387, 0, 0, 0, 0, 0, 20729, 0, 0, 0, 0, 0, 0, 0, 10573, 0, 0, 0, 4421, 0, 27127, 0, 0, 0, 19853, 0, 9259, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12179, 0, 5881, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28751, 0, 26977, 0, 0, 0, 11303, 0, 0, 0, 0, 0, 23657, 0, 0, 0, 0, 0, 17879, 0, 9931, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15161, 0, 27007, 0, 0, 0, 2243, 0, 0, 0, 0, 0, 317, 0, 0, 0, 0, 0, 0, 0, 8641, 0, 0, 0, 0, 0, 29773, 0, 0, 0, 0, 0, 19507, 0, 0, 0, 0, 0, 8299, 0, 0, 0, 12557, 0, 0, 0, 0, 0, 26099, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10631, 0, 0, 0, 0, 0, 12653, 0, 16249, 0, 0, 0, 18677, 0, 0, 0, 0, 0, 0, 0, 16321, 0, 0, 0, 0, 0, 6133, 0, 0, 0, 25511, 0, 0, 0, 0, 0, 27743, 0, 11059, 0, 0, 0, 10337, 0, 0, 0, 0, 0, 0, 0, 9991, 0, 0, 0, 0, 0, 19843, 0, 0, 0, 0, 0, 4057, 0, 0, 0, 4493, 0, 0, 0, 0, 0, 24797, 0, 0, 0, 0, 0, 3299, 0, 17821, 0, 0, 0, 0, 0, 14143, 0, 0, 0, 0, 0, 12007, 0, 0, 0, 2213, 0, 13669, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8819, 0, 26251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 821, 0, 25717, 0, 0, 0, 0, 0, 1879, 0, 0, 0, 27857, 0, 0, 0, 0, 0, 26909, 0, 29251, 0, 0, 0, 0, 0, 6073, 0, 0, 0, 14111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28877, 0, 0, 0, 0, 0, 16259, 0, 29011, 0, 0, 0, 0, 0, 17803, 0, 0, 0, 23381, 0, 23917, 0, 0, 0, 0, 0, 23209, 0, 0, 0, 23687, 0, 0, 0, 0, 0, 11819, 0, 0, 0, 0, 0, 0, 0, 18493, 0, 0, 0, 9371, 0, 25147, 0, 0, 0, 16223, 0, 8269, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10333, 0, 0, 0, 26771, 0, 0, 0, 0, 0, 14933, 0, 17779, 0, 0, 0, 887, 0, 0, 0, 0, 0, 25799, 0, 8941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4897, 0, 0, 0, 14783, 0, 1849, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28111, 0, 0, 0, 0, 0, 11623, 0, 0, 0, 25631, 0, 27757, 0, 0, 0, 1163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15539, 0, 13441, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5021, 0, 0, 0, 0, 0, 29153, 0, 28129, 0, 0, 0, 8117, 0, 0, 0, 0, 0, 18929, 0, 2791, 0, 0, 0, 0, 0, 4813, 0, 0, 0, 0, 0, 25747, 0, 0, 0, 2663, 0, 0, 0, 0, 0, 2417, 0, 0, 0, 0, 0, 1139, 0, 1081, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11411, 0, 9007, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2687, 0, 0, 0, 0, 0, 10559, 0, 0, 0, 0, 0, 0, 0, 17443, 0, 0, 0, 0, 0, 10027, 0, 0, 0, 5513, 0, 17299, 0, 0, 0, 2927, 0, 0, 0, 0, 0, 0, 0, 4141, 0, 0, 0, 0, 0, 12433, 0, 0, 0, 5771, 0, 0, 0, 0, 0, 5273, 0, 14419, 0, 0, 0, 18947, 0, 0, 0, 0, 0, 179, 0, 21331, 0, 0, 0, 0, 0, 11023, 0, 0, 0, 0, 0, 6157, 0, 0, 0, 0, 0, 14449, 0, 0, 0, 15347, 0, 0, 0, 0, 0, 1409, 0, 18451, 0, 0, 0, 0, 0, 4273, 0, 0, 0, 5891, 0, 18307, 0, 0, 0, 0, 0, 9679, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17621, 0, 0, 0, 0, 0, 28103, 0, 13219, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19769, 0, 9511, 0, 0, 0, 0, 0, 22873, 0, 0, 0, 4661, 0, 13777, 0, 0, 0, 12953, 0, 0, 0, 0, 0, 29927, 0, 0, 0, 0, 0, 16889, 0, 16201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1961, 0, 2917, 0, 0, 0, 0, 0, 5149, 0, 0, 0, 3947, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7993, 0, 0, 0, 23651, 0, 2677, 0, 0, 0, 9083, 0, 7639, 0, 0, 0, 1667, 0, 0, 0, 0, 0, 0, 0, 16741, 0, 0, 0, 0, 0, 13693, 0, 0, 0, 16901, 0, 0, 0, 0, 0, 7373, 0, 26809, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15299, 0, 17971, 0, 0, 0, 0, 0, 16693, 0, 0, 0, 0, 0, 8257, 0, 0, 0, 15203, 0, 0, 0, 0, 0, 14297, 0, 0, 0, 0, 0, 16319, 0, 27481, 0, 0, 0, 0, 0, 19183, 0, 0, 0, 10091, 0, 17257, 0, 0, 0, 19013, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13859, 0, 25411, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17947, 0, 0, 0, 10463, 0, 15319, 0, 0, 0, 29327, 0, 0, 0, 0, 0, 21029, 0, 25261, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9071, 0, 12517, 0, 0, 0, 3293, 0, 0, 0, 0, 0, 22577, 0, 0, 0, 0, 0, 7439, 0, 12421, 0, 0, 0, 0, 0, 10033, 0, 0, 0, 28631, 0, 4177, 0, 0, 0, 0, 0, 22369, 0, 0, 0, 27887, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23743, 0, 0, 0, 10841, 0, 157, 0, 0, 0, 15803, 0, 17929, 0, 0, 0, 19727, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14533, 0, 0, 0, 4091, 0, 0, 0, 0, 0, 22493, 0, 25759, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9209, 0, 24691, 0, 0, 0, 0, 0, 9343, 0, 0, 0, 0, 0, 12457, 0, 0, 0, 0, 0, 2269, 0, 0, 0, 4847, 0, 0, 0, 0, 0, 8969, 0, 19501, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14501, 0, 0, 0, 0, 0, 1583, 0, 10099, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10291, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27281, 0, 5977, 0, 0, 0, 7103, 0, 5659, 0, 0, 0, 4967, 0, 0, 0, 0, 0, 23129, 0, 24001, 0, 0, 0, 0, 0, 21613, 0, 0, 0, 13271, 0, 0, 0, 0, 0, 12323, 0, 0, 0, 0, 0, 7457, 0, 0, 0, 0, 0, 6389, 0, 0, 0, 0, 0, 0, 0, 8773, 0, 0, 0, 27161, 0, 13207, 0, 0, 0, 0, 0, 12709, 0, 0, 0, 12977, 0, 0, 0, 0, 0, 2789, 0, 0, 0, 0, 0, 0, 0, 8623, 0, 0, 0, 21971, 0, 3727, 0, 0, 0, 1523, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14851, 0, 0, 0, 0, 0, 6553, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18713, 0, 0, 0, 0, 0, 11177, 0, 0, 0, 0, 0, 10469, 0, 17341, 0, 0, 0, 0, 0, 16063, 0, 0, 0, 0, 0, 25057, 0, 0, 0, 11213, 0, 1219, 0, 0, 0, 20597, 0, 0, 0, 0, 0, 6449, 0, 20341, 0, 0, 0, 0, 0, 17293, 0, 0, 0, 0, 0, 7807, 0, 0, 0, 0, 0, 20389, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5039, 0, 26041, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9851, 0, 26557, 0, 0, 0, 13823, 0, 22879, 0, 0, 0, 27647, 0, 0, 0, 0, 0, 27329, 0, 0, 0, 0, 0, 0, 0, 17833, 0, 0, 0, 23561, 0, 26797, 0, 0, 0, 4343, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11161, 0, 0, 0, 0, 0, 24523, 0, 0, 0, 701, 0, 24337, 0, 0, 0, 0, 0, 14809, 0, 0, 0, 17387, 0, 0, 0, 0, 0, 20849, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 971, 0, 2257, 0, 0, 0, 263, 0, 18349, 0, 0, 0, 18467, 0, 0, 0, 0, 0, 0, 0, 29761, 0, 0, 0, 0, 0, 15583, 0, 0, 0, 18371, 0, 0, 0, 0, 0, 0, 0, 13579, 0, 0, 0, 9917, 0, 0, 0, 0, 0, 19499, 0, 23011, 0, 0, 0, 0, 0, 12703, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15623, 0, 13609, 0, 0, 0, 5477, 0, 0, 0, 0, 0, 10019, 0, 21601, 0, 0, 0, 0, 0, 29893, 0, 0, 0, 5051, 0, 12217, 0, 0, 0, 29723, 0, 10729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15329, 0, 16591, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14051, 0, 23197, 0, 0, 0, 6473, 0, 0, 0, 0, 0, 4337, 0, 0, 0, 0, 0, 9899, 0, 0, 0, 0, 0, 0, 0, 2923, 0, 0, 0, 0, 0, 13987, 0, 0, 0, 5183, 0, 0, 0, 0, 0, 5567, 0, 0, 0, 0, 0, 26969, 0, 29221, 0, 0, 0, 0, 0, 1213, 0, 0, 0, 28211, 0, 11527, 0, 0, 0, 0, 0, 29929, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11399, 0, 0, 0, 0, 0, 0, 0, 20593, 0, 0, 0, 2231, 0, 28927, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14687, 0, 0, 0, 0, 0, 0, 0, 22411, 0, 0, 0, 0, 0, 5923, 0, 0, 0, 17111, 0, 0, 0, 0, 0, 17453, 0, 26179, 0, 0, 0, 25667, 0, 0, 0, 0, 0, 599, 0, 0, 0, 0, 0, 0, 0, 25093, 0, 0, 0, 0, 0, 4687, 0, 0, 0, 713, 0, 10249, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8549, 0, 23701, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4001, 0, 16417, 0, 0, 0, 14603, 0, 19759, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19529, 0, 15541, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16151, 0, 21727, 0, 0, 0, 12143, 0, 4819, 0, 0, 0, 977, 0, 0, 0, 0, 0, 0, 0, 16231, 0, 0, 0, 0, 0, 5023, 0, 0, 0, 14741, 0, 25117, 0, 0, 0, 6443, 0, 0, 0, 0, 0, 18167, 0, 0, 0, 0, 0, 1349, 0, 27961, 0, 0, 0, 0, 0, 4783, 0, 0, 0, 20861, 0, 0, 0, 0, 0, 0, 0, 26149, 0, 0, 0, 13397, 0, 0, 0, 0, 0, 8039, 0, 0, 0, 0, 0, 0, 0, 7783, 0, 0, 0, 26171, 0, 19477, 0, 0, 0, 0, 0, 14359, 0, 0, 0, 29807, 0, 0, 0, 0, 0, 0, 0, 19891, 0, 0, 0, 0, 0, 16213, 0, 0, 0, 5141, 0, 0, 0, 0, 0, 5483, 0, 0, 0, 0, 0, 2357, 0, 0, 0, 0, 0, 17819, 0, 14191, 0, 0, 0, 0, 0, 24043, 0, 0, 0, 0, 0, 17287, 0, 0, 0, 24653, 0, 16969, 0, 0, 0, 17447, 0, 0, 0, 0, 0, 5189, 0, 27901, 0, 0, 0, 0, 0, 17713, 0, 0, 0, 24581, 0, 26707, 0, 0, 0, 7253, 0, 11779, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23939, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17411, 0, 25297, 0, 0, 0, 0, 0, 19729, 0, 0, 0, 13367, 0, 0, 0, 0, 0, 19559, 0, 4261, 0, 0, 0, 0, 0, 25393, 0, 0, 0, 20411, 0, 4117, 0, 0, 0, 8543, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5549, 0, 0, 0, 0, 0, 0, 0, 24103, 0, 0, 0, 5741, 0, 25807, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6569, 0, 0, 0, 0, 0, 0, 0, 28153, 0, 0, 0, 11261, 0, 7507, 0, 0, 0, 22523, 0, 18769, 0, 0, 0, 1877, 0, 0, 0, 0, 0, 0, 0, 23461, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4223, 0, 24289, 0, 0, 0, 5927, 0, 0, 0, 0, 0, 0, 0, 24481, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21487, 0, 0, 0, 25913, 0, 9619, 0, 0, 0, 4637, 0, 0, 0, 0, 0, 25769, 0, 10471, 0, 0, 0, 0, 0, 16663, 0, 0, 0, 10301, 0, 0, 0, 0, 0, 4733, 0, 12619, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6091, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18251, 0, 22777, 0, 0, 0, 3323, 0, 5449, 0, 0, 0, 12317, 0, 0, 0, 0, 0, 2129, 0, 24841, 0, 0, 0, 0, 0, 12583, 0, 0, 0, 13061, 0, 5377, 0, 0, 0, 12743, 0, 0, 0, 0, 0, 5987, 0, 0, 0, 0, 0, 15839, 0, 12211, 0, 0, 0, 0, 0, 27673, 0, 0, 0, 0, 0, 24547, 0, 0, 0, 0, 0, 24889, 0, 0, 0, 13817, 0, 0, 0, 0, 0, 10139, 0, 0, 0, 0, 0, 0, 0, 223, 0, 0, 0, 15671, 0, 0, 0, 0, 0, 10553, 0, 3859, 0, 0, 0, 22247, 0, 0, 0, 0, 0, 0, 0, 21991, 0, 0, 0, 0, 0, 16633, 0, 0, 0, 3881, 0, 0, 0, 0, 0, 0, 0, 9169, 0, 0, 0, 25247, 0, 0, 0, 0, 0, 2069, 0, 28681, 0, 0, 0, 0, 0, 11863, 0, 0, 0, 0, 0, 23587, 0, 0, 0, 4913, 0, 15289, 0, 0, 0, 25007, 0, 0, 0, 0, 0, 13799, 0, 0, 0, 0, 0, 0, 0, 29053, 0, 0, 0, 25211, 0, 17887, 0, 0, 0, 8303, 0, 13879, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14489, 0, 10501, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10271, 0, 15427, 0, 0, 0, 13613, 0, 26029, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6329, 0, 21481, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19781, 0, 29317, 0, 0, 0, 25343, 0, 0, 0, 0, 0, 4937, 0, 0, 0, 0, 0, 0, 0, 29431, 0, 0, 0, 0, 0, 4363, 0, 0, 0, 3851, 0, 12577, 0, 0, 0, 0, 0, 12919, 0, 0, 0, 24107, 0, 0, 0, 0, 0, 7619, 0, 0, 0, 0, 0, 0, 0, 15343, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1103, 0, 27799, 0, 0, 0, 9437, 0, 0, 0, 0, 0, 0, 0, 18631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 0, 0, 0, 0, 0, 18503, 0, 1819, 0, 0, 0, 28817, 0, 0, 0, 0, 0, 809, 0, 3061, 0, 0, 0, 0, 0, 24463, 0, 0, 0, 0, 0, 24847, 0, 0, 0, 16043, 0, 0, 0, 0, 0, 27107, 0, 0, 0, 0, 0, 0, 0, 20131, 0, 0, 0, 0, 0, 25693, 0, 0, 0, 0, 0, 23557, 0, 0, 0, 6833, 0, 15979, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13439, 0, 14701, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19301, 0, 307, 0, 0, 0, 17813, 0, 24979, 0, 0, 0, 137, 0, 0, 0, 0, 0, 8429, 0, 20011, 0, 0, 0, 0, 0, 24553, 0, 0, 0, 16421, 0, 14407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17327, 0, 0, 0, 0, 0, 7019, 0, 10531, 0, 0, 0, 0, 0, 20113, 0, 0, 0, 16451, 0, 0, 0, 0, 0, 0, 0, 11659, 0, 0, 0, 14447, 0, 0, 0, 0, 0, 269, 0, 0, 0, 0, 0, 0, 0, 11563, 0, 0, 0, 11681, 0, 29767, 0, 0, 0, 27773, 0, 29059, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9181, 0, 0, 0, 0, 0, 12643, 0, 0, 0, 15221, 0, 0, 0, 0, 0, 5693, 0, 29329, 0, 0, 0, 5507, 0, 0, 0, 0, 0, 18869, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25687, 0, 0, 0, 3233, 0, 6469, 0, 0, 0, 12197, 0, 0, 0, 0, 0, 0, 0, 2701, 0, 0, 0, 0, 0, 2383, 0, 0, 0, 7151, 0, 16207, 0, 0, 0, 3473, 0, 20179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3989, 0, 24991, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9641, 0, 0, 0, 0, 0, 22223, 0, 0, 0, 0, 0, 12737, 0, 0, 0, 0, 0, 9689, 0, 23581, 0, 0, 0, 0, 0, 9433, 0, 0, 0, 28811, 0, 18817, 0, 0, 0, 4973, 0, 0, 0, 0, 0, 13967, 0, 0, 0, 0, 0, 12689, 0, 19561, 0, 0, 0, 0, 0, 18853, 0, 0, 0, 0, 0, 11317, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23477, 0, 0, 0, 0, 0, 15179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28507, 0, 0, 0, 26303, 0, 8059, 0, 0, 0, 21407, 0, 0, 0, 0, 0, 0, 0, 27241, 0, 0, 0, 0, 0, 17053, 0, 0, 0, 17321, 0, 0, 0, 0, 0, 16823, 0, 2869, 0, 0, 0, 21257, 0, 0, 0, 0, 0, 0, 0, 23641, 0, 0, 0, 0, 0, 22573, 0, 0, 0, 0, 0, 17707, 0, 0, 0, 0, 0, 16759, 0, 0, 0, 8417, 0, 0, 0, 0, 0, 6029, 0, 6901, 0, 0, 0, 0, 0, 25063, 0, 0, 0, 24371, 0, 22927, 0, 0, 0, 24053, 0, 2749, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19739, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19931, 0, 28447, 0, 0, 0, 0, 0, 15529, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10529, 0, 21061, 0, 0, 0, 0, 0, 25183, 0, 0, 0, 27761, 0, 0, 0, 0, 0, 17573, 0, 0, 0, 0, 0, 20687, 0, 0, 0, 0, 0, 5339, 0, 20821, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4271, 0, 7537, 0, 0, 0, 0, 0, 25939, 0, 0, 0, 15497, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10303, 0, 0, 0, 12101, 0, 14227, 0, 0, 0, 29873, 0, 19189, 0, 0, 0, 6287, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2143, 0, 0, 0, 7661, 0, 0, 0, 0, 0, 25853, 0, 1399, 0, 0, 0, 19997, 0, 0, 0, 0, 0, 17609, 0, 22591, 0, 0, 0, 0, 0, 7453, 0, 0, 0, 0, 0, 26737, 0, 0, 0, 17513, 0, 20959, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4769, 0, 9001, 0, 0, 0, 0, 0, 703, 0, 0, 0, 14711, 0, 19567, 0, 0, 0, 12083, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4619, 0, 16171, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11017, 0, 0, 0, 12773, 0, 19939, 0, 0, 0, 10847, 0, 0, 0, 0, 0, 2549, 0, 13711, 0, 0, 0, 0, 0, 15733, 0, 0, 0, 0, 0, 14827, 0, 0, 0, 21773, 0, 0, 0, 0, 0, 13337, 0, 0, 0, 0, 0, 12059, 0, 14731, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3221, 0, 22657, 0, 0, 0, 0, 0, 13129, 0, 0, 0, 16337, 0, 0, 0, 0, 0, 13289, 0, 0, 0, 0, 0, 0, 0, 28363, 0, 0, 0, 22391, 0, 20947, 0, 0, 0, 27353, 0, 6379, 0, 0, 0, 22037, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26083, 0, 0, 0, 24881, 0, 0, 0, 0, 0, 27113, 0, 28069, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13829, 0, 13141, 0, 0, 0, 0, 0, 103, 0, 0, 0, 0, 0, 17077, 0, 0, 0, 16253, 0, 25369, 0, 0, 0, 7157, 0, 0, 0, 0, 0, 20519, 0, 10261, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16811, 0, 1927, 0, 0, 0, 0, 0, 12409, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21841, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20351, 0, 0, 0, 0, 0, 11723, 0, 24139, 0, 0, 0, 25757, 0, 0, 0, 0, 0, 11579, 0, 28621, 0, 0, 0, 0, 0, 14683, 0, 0, 0, 15581, 0, 0, 0, 0, 0, 23873, 0, 0, 0, 0, 0, 19007, 0, 0, 0, 0, 0, 8699, 0, 29851, 0, 0, 0, 0, 0, 11083, 0, 0, 0, 15611, 0, 24757, 0, 0, 0, 0, 0, 24259, 0, 0, 0, 17597, 0, 0, 0, 0, 0, 25889, 0, 0, 0, 0, 0, 0, 0, 27103, 0, 0, 0, 12731, 0, 24517, 0, 0, 0, 20003, 0, 0, 0, 0, 0, 12587, 0, 0, 0, 0, 0, 0, 0, 19471, 0, 0, 0, 0, 0, 27343, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21023, 0, 18619, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28949, 0, 28891, 0, 0, 0, 0, 0, 27613, 0, 0, 0, 0, 0, 27367, 0, 0, 0, 4283, 0, 0, 0, 0, 0, 25217, 0, 0, 0, 0, 0, 27239, 0, 11101, 0, 0, 0, 0, 0, 21913, 0, 0, 0, 1901, 0, 877, 0, 0, 0, 0, 0, 25009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16589, 0, 14491, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28867, 0, 0, 0, 2273, 0, 4399, 0, 0, 0, 18407, 0, 0, 0, 0, 0, 1919, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28181, 0, 15247, 0, 0, 0, 25133, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21089, 0, 4231, 0, 0, 0, 0, 0, 29143, 0, 0, 0, 12251, 0, 15097, 0, 0, 0, 0, 0, 3259, 0, 0, 0, 19697, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21761, 0, 13807, 0, 0, 0, 4883, 0, 20659, 0, 0, 0, 11537, 0, 0, 0, 0, 0, 0, 0, 18211, 0, 0, 0, 0, 0, 6343, 0, 0, 0, 6821, 0, 0, 0, 0, 0, 6113, 0, 6649, 0, 0, 0, 12227, 0, 0, 0, 0, 0, 1019, 0, 13771, 0, 0, 0, 0, 0, 1153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15919, 0, 0, 0, 23957, 0, 0, 0, 0, 0, 779, 0, 3121, 0, 0, 0, 0, 0, 2173, 0, 0, 0, 28151, 0, 0, 0, 0, 0, 4313, 0, 29209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3779, 0, 21211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16361, 0, 27817, 0, 0, 0, 18023, 0, 0, 0, 0, 0, 15887, 0, 0, 0, 0, 0, 12209, 0, 26731, 0, 0, 0, 0, 0, 5233, 0, 0, 0, 0, 0, 25537, 0, 0, 0, 25973, 0, 0, 0, 0, 0, 10187, 0, 0, 0, 0, 0, 20039, 0, 0, 0, 0, 0, 0, 0, 19693, 0, 0, 0, 18971, 0, 2287, 0, 0, 0, 0, 0, 4519, 0, 0, 0, 23897, 0, 0, 0, 0, 0, 13709, 0, 0, 0, 0, 0, 0, 0, 11353, 0, 0, 0, 13781, 0, 17377, 0, 0, 0, 0, 0, 19399, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3931, 0, 0, 0, 0, 0, 17473, 0, 0, 0, 21731, 0, 0, 0, 0, 0, 10523, 0, 0, 0, 0, 0, 257, 0, 0, 0, 0, 0, 21389, 0, 0, 0, 0, 0, 0, 0, 29713, 0, 0, 0, 0, 0, 27787, 0, 0, 0, 3023, 0, 14869, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20099, 0, 12151, 0, 0, 0, 0, 0, 6373, 0, 0, 0, 0, 0, 18727, 0, 0, 0, 3053, 0, 1279, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24149, 0, 17851, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20771, 0, 10177, 0, 0, 0, 2903, 0, 25609, 0, 0, 0, 19457, 0, 0, 0, 0, 0, 0, 0, 9301, 0, 0, 0, 0, 0, 9643, 0, 0, 0, 26291, 0, 15877, 0, 0, 0, 17993, 0, 0, 0, 0, 0, 20477, 0, 0, 0, 0, 0, 0, 0, 241, 0, 0, 0, 0, 0, 16333, 0, 0, 0, 11621, 0, 0, 0, 0, 0, 0, 0, 28459, 0, 0, 0, 6467, 0, 0, 0, 0, 0, 12659, 0, 0, 0, 0, 0, 0, 0, 28573, 0, 0, 0, 14621, 0, 24097, 0, 0, 0, 2993, 0, 7429, 0, 0, 0, 2087, 0, 0, 0, 0, 0, 0, 0, 10651, 0, 0, 0, 0, 0, 4663, 0, 0, 0, 7451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20837, 0, 0, 0, 0, 0, 8579, 0, 25741, 0, 0, 0, 0, 0, 26353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29273, 0, 5419, 0, 0, 0, 8207, 0, 0, 0, 0, 0, 23669, 0, 2491, 0, 0, 0, 0, 0, 10783, 0, 0, 0, 26891, 0, 1297, 0, 0, 0, 18803, 0, 2539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26249, 0, 211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5861, 0, 6817, 0, 0, 0, 0, 0, 1249, 0, 0, 0, 17987, 0, 0, 0, 0, 0, 12629, 0, 0, 0, 0, 0, 0, 0, 13843, 0, 0, 0, 11171, 0, 24907, 0, 0, 0, 27023, 0, 0, 0, 0, 0, 24677, 0, 0, 0, 0, 0, 7859, 0, 0, 0, 0, 0, 0, 0, 14863, 0, 0, 0, 17291, 0, 14257, 0, 0, 0, 0, 0, 13549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25049, 0, 0, 0, 0, 0, 0, 0, 9673, 0, 0, 0, 15881, 0, 9817, 0, 0, 0, 15593, 0, 0, 0, 0, 0, 6497, 0, 0, 0, 0, 0, 0, 0, 14221, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28031, 0, 0, 0, 0, 0, 6533, 0, 28909, 0, 0, 0, 17477, 0, 0, 0, 0, 0, 0, 0, 12931, 0, 0, 0, 0, 0, 16903, 0, 0, 0, 0, 0, 23797, 0, 0, 0, 14363, 0, 21169, 0, 0, 0, 25427, 0, 0, 0, 0, 0, 359, 0, 12781, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14921, 0, 0, 0, 0, 0, 25523, 0, 3379, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11339, 0, 29191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29801, 0, 13537, 0, 0, 0, 14873, 0, 23929, 0, 0, 0, 14627, 0, 0, 0, 0, 0, 0, 0, 27151, 0, 0, 0, 0, 0, 24133, 0, 0, 0, 3821, 0, 16927, 0, 0, 0, 17363, 0, 0, 0, 0, 0, 29087, 0, 0, 0, 0, 0, 20459, 0, 661, 0, 0, 0, 0, 0, 18433, 0, 0, 0, 0, 0, 29167, 0, 0, 0, 0, 0, 17959, 0, 0, 0, 16127, 0, 0, 0, 0, 0, 21689, 0, 0, 0, 0, 0, 0, 0, 18703, 0, 0, 0, 17981, 0, 8557, 0, 0, 0, 0, 0, 6169, 0, 0, 0, 10697, 0, 0, 0, 0, 0, 0, 0, 3511, 0, 0, 0, 0, 0, 18943, 0, 0, 0, 26981, 0, 0, 0, 0, 0, 0, 0, 27649, 0, 0, 0, 16007, 0, 0, 0, 0, 0, 20549, 0, 3271, 0, 0, 0, 0, 0, 4933, 0, 0, 0, 0, 0, 28207, 0, 0, 0, 16463, 0, 6049, 0, 0, 0, 6527, 0, 0, 0, 0, 0, 16109, 0, 0, 0, 0, 0, 0, 0, 1333, 0, 0, 0, 13661, 0, 29437, 0, 0, 0, 29093, 0, 25429, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7559, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1031, 0, 6187, 0, 0, 0, 2063, 0, 619, 0, 0, 0, 5177, 0, 0, 0, 0, 0, 8639, 0, 26101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1301, 0, 17767, 0, 0, 0, 27653, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10951, 0, 0, 0, 0, 0, 15913, 0, 0, 0, 8471, 0, 14887, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28727, 0, 0, 0, 0, 0, 28409, 0, 0, 0, 0, 0, 0, 0, 19963, 0, 0, 0, 22181, 0, 0, 0, 0, 0, 3413, 0, 2389, 0, 0, 0, 20987, 0, 0, 0, 0, 0, 0, 0, 7081, 0, 0, 0, 0, 0, 17683, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6953, 0, 13369, 0, 0, 0, 19577, 0, 0, 0, 0, 0, 5429, 0, 5371, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13297, 0, 0, 0, 6803, 0, 0, 0, 0, 0, 15557, 0, 0, 0, 0, 0, 14849, 0, 13201, 0, 0, 0, 0, 0, 283, 0, 0, 0, 0, 0, 14317, 0, 0, 0, 18383, 0, 4429, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17011, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10061, 0, 11857, 0, 0, 0, 22433, 0, 27289, 0, 0, 0, 23237, 0, 0, 0, 0, 0, 13049, 0, 8461, 0, 0, 0, 0, 0, 15313, 0, 0, 0, 4871, 0, 19027, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19637, 0, 0, 0, 0, 0, 18569, 0, 1291, 0, 0, 0, 0, 0, 8563, 0, 0, 0, 18761, 0, 5437, 0, 0, 0, 0, 0, 13969, 0, 0, 0, 2897, 0, 0, 0, 0, 0, 21059, 0, 0, 0, 0, 0, 0, 0, 13873, 0, 0, 0, 4751, 0, 0, 0, 0, 0, 2363, 0, 17509, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13801, 0, 0, 0, 0, 0, 5713, 0, 0, 0, 17531, 0, 0, 0, 0, 0, 17243, 0, 20089, 0, 0, 0, 17057, 0, 0, 0, 0, 0, 21179, 0, 20491, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7207, 0, 0, 0, 24023, 0, 18019, 0, 0, 0, 16817, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20863, 0, 0, 0, 27941, 0, 6967, 0, 0, 0, 21953, 0, 24799, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6299, 0, 29611, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21191, 0, 0, 0, 0, 0, 24533, 0, 9649, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28169, 0, 5101, 0, 0, 0, 0, 0, 20983, 0, 0, 0, 3401, 0, 21127, 0, 0, 0, 28073, 0, 0, 0, 0, 0, 18587, 0, 0, 0, 0, 0, 3449, 0, 10321, 0, 0, 0, 0, 0, 23473, 0, 0, 0, 22961, 0, 4387, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4997, 0, 0, 0, 0, 0, 26729, 0, 0, 0, 0, 0, 0, 0, 28993, 0, 0, 0, 0, 0, 787, 0, 0, 0, 14753, 0, 19609, 0, 0, 0, 12167, 0, 0, 0, 0, 0, 0, 0, 29551, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21941, 0, 0, 0, 0, 0, 7583, 0, 23659, 0, 0, 0, 9707, 0, 0, 0, 0, 0, 11729, 0, 16711, 0, 0, 0, 0, 0, 27193, 0, 0, 0, 0, 0, 8467, 0, 0, 0, 0, 0, 5209, 0, 0, 0, 10727, 0, 0, 0, 0, 0, 0, 0, 9211, 0, 0, 0, 0, 0, 6583, 0, 0, 0, 15131, 0, 4447, 0, 0, 0, 28673, 0, 5059, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24359, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8381, 0, 3037, 0, 0, 0, 9623, 0, 8599, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22079, 0, 11821, 0, 0, 0, 0, 0, 13633, 0, 0, 0, 41, 0, 25327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9137, 0, 0, 0, 0, 0, 26129, 0, 2341, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27371, 0, 0, 0, 0, 0, 0, 0, 14389, 0, 0, 0, 6257, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3373, 0, 0, 0, 14411, 0, 18847, 0, 0, 0, 11393, 0, 9949, 0, 0, 0, 17837, 0, 0, 0, 0, 0, 0, 0, 28291, 0, 0, 0, 0, 0, 4453, 0, 0, 0, 26141, 0, 0, 0, 0, 0, 16613, 0, 12949, 0, 0, 0, 24617, 0, 0, 0, 0, 0, 10679, 0, 0, 0, 0, 0, 0, 0, 25933, 0, 0, 0, 0, 0, 17497, 0, 0, 0, 5963, 0, 25579, 0, 0, 0, 25847, 0, 0, 0, 0, 0, 7079, 0, 13621, 0, 0, 0, 0, 0, 21493, 0, 0, 0, 26261, 0, 8017, 0, 0, 0, 14393, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23099, 0, 27721, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13327, 0, 0, 0, 5843, 0, 0, 0, 0, 0, 15467, 0, 0, 0, 0, 0, 23339, 0, 4471, 0, 0, 0, 0, 0, 20353, 0, 0, 0, 20621, 0, 7897, 0, 0, 0, 24083, 0, 0, 0, 0, 0, 24887, 0, 0, 0, 0, 0, 23609, 0, 3181, 0, 0, 0, 0, 0, 21583, 0, 0, 0, 0, 0, 24967, 0, 0, 0, 0, 0, 24679, 0, 0, 0, 7097, 0, 0, 0, 0, 0, 17909, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27011, 0, 9397, 0, 0, 0, 18113, 0, 27169, 0, 0, 0, 10487, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 673, 0, 0, 0, 6401, 0, 0, 0, 0, 0, 8633, 0, 16519, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15451, 0, 0, 0, 0, 0, 11653, 0, 0, 0, 0, 0, 28627, 0, 0, 0, 20873, 0, 27679, 0, 0, 0, 227, 0, 0, 0, 0, 0, 25139, 0, 28741, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5261, 0, 6547, 0, 0, 0, 13133, 0, 5479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12601, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22661, 0, 17527, 0, 0, 0, 0, 0, 26449, 0, 0, 0, 14207, 0, 0, 0, 0, 0, 2339, 0, 10141, 0, 0, 0, 0, 0, 16993, 0, 0, 0, 8651, 0, 0, 0, 0, 0, 28493, 0, 0, 0, 0, 0, 9767, 0, 0, 0, 0, 0, 27179, 0, 4441, 0, 0, 0, 0, 0, 4153, 0, 0, 0, 17921, 0, 29377, 0, 0, 0, 0, 0, 15019, 0, 0, 0, 29147, 0, 0, 0, 0, 0, 28199, 0, 0, 0, 0, 0, 0, 0, 29413, 0, 0, 0, 1181, 0, 6037, 0, 0, 0, 10763, 0, 0, 0, 0, 0, 17207, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15793, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9473, 0, 23239, 0, 0, 0, 22727, 0, 0, 0, 0, 0, 1229, 0, 3481, 0, 0, 0, 0, 0, 18373, 0, 0, 0, 0, 0, 15817, 0, 0, 0, 6593, 0, 12769, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15689, 0, 22651, 0, 0, 0, 0, 0, 3433, 0, 0, 0, 6521, 0, 3187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7349, 0, 5251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21401, 0, 21937, 0, 0, 0, 4583, 0, 9019, 0, 0, 0, 29957, 0, 0, 0, 0, 0, 13469, 0, 0, 0, 0, 0, 0, 0, 29383, 0, 0, 0, 0, 0, 17557, 0, 0, 0, 13583, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25709, 0, 6541, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16871, 0, 3547, 0, 0, 0, 0, 0, 24049, 0, 0, 0, 8147, 0, 0, 0, 0, 0, 2369, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3281, 0, 4567, 0, 0, 0, 16433, 0, 9109, 0, 0, 0, 13847, 0, 0, 0, 0, 0, 0, 0, 20521, 0, 0, 0, 0, 0, 17893, 0, 0, 0, 27611, 0, 0, 0, 0, 0, 10733, 0, 8959, 0, 0, 0, 5297, 0, 0, 0, 0, 0, 5639, 0, 2221, 0, 0, 0, 0, 0, 21943, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27173, 0, 8989, 0, 0, 0, 26267, 0, 0, 0, 0, 0, 12329, 0, 23911, 0, 0, 0, 0, 0, 20653, 0, 0, 0, 431, 0, 23767, 0, 0, 0, 0, 0, 1489, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24569, 0, 2731, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9431, 0, 0, 0, 0, 0, 22643, 0, 0, 0, 0, 0, 6647, 0, 0, 0, 0, 0, 659, 0, 1321, 0, 0, 0, 0, 0, 28333, 0, 0, 0, 0, 0, 127, 0, 0, 0, 7493, 0, 0, 0, 0, 0, 21737, 0, 0, 0, 0, 0, 22349, 0, 10741, 0, 0, 0, 0, 0, 22003, 0, 0, 0, 7421, 0, 13837, 0, 0, 0, 0, 0, 16069, 0, 0, 0, 28517, 0, 0, 0, 0, 0, 6779, 0, 0, 0, 0, 0, 0, 0, 29833, 0, 0, 0, 4541, 0, 8137, 0, 0, 0, 0, 0, 24019, 0, 0, 0, 26237, 0, 0, 0, 0, 0, 0, 0, 8551, 0, 0, 0, 0, 0, 8233, 0, 0, 0, 3251, 0, 0, 0, 0, 0, 12833, 0, 7699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9839, 0, 0, 0, 0, 0, 0, 0, 11233, 0, 0, 0, 0, 0, 67, 0, 0, 0, 26123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10859, 0, 2911, 0, 0, 0, 0, 0, 10993, 0, 0, 0, 15551, 0, 11797, 0, 0, 0, 5363, 0, 5899, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5669, 0, 6301, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12487, 0, 0, 0, 21383, 0, 7129, 0, 0, 0, 10217, 0, 0, 0, 0, 0, 0, 0, 11611, 0, 0, 0, 0, 0, 0, 0, 0, 0, 881, 0, 4327, 0, 0, 0, 8753, 0, 0, 0, 0, 0, 8927, 0, 0, 0, 0, 0, 12899, 0, 23341, 0, 0, 0, 0, 0, 20953, 0, 0, 0, 23171, 0, 0, 0, 0, 0, 0, 0, 16909, 0, 0, 0, 8777, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10093, 0, 0, 0, 5381, 0, 5617, 0, 0, 0, 7613, 0, 9739, 0, 0, 0, 25187, 0, 0, 0, 0, 0, 0, 0, 29131, 0, 0, 0, 0, 0, 25453, 0, 0, 0, 25931, 0, 0, 0, 0, 0, 17033, 0, 0, 0, 0, 0, 23147, 0, 0, 0, 0, 0, 20129, 0, 16501, 0, 0, 0, 0, 0, 14803, 0, 0, 0, 0, 0, 28837, 0, 0, 0, 0, 0, 7729, 0, 0, 0, 26687, 0, 0, 0, 0, 0, 14429, 0, 14041, 0, 0, 0, 0, 0, 13093, 0, 0, 0, 19961, 0, 0, 0, 0, 0, 23423, 0, 21019, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14699, 0, 4831, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8171, 0, 11437, 0, 0, 0, 0, 0, 22039, 0, 0, 0, 29537, 0, 0, 0, 0, 0, 14939, 0, 15811, 0, 0, 0, 0, 0, 16153, 0, 0, 0, 29651, 0, 6427, 0, 0, 0, 17783, 0, 0, 0, 0, 0, 29297, 0, 0, 0, 0, 0, 929, 0, 0, 0, 0, 0, 0, 0, 3313, 0, 0, 0, 8051, 0, 5017, 0, 0, 0, 0, 0, 18169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27359, 0, 0, 0, 0, 0, 0, 0, 433, 0, 0, 0, 27431, 0, 28297, 0, 0, 0, 17903, 0, 289, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2621, 0, 0, 0, 0, 0, 29633, 0, 0, 0, 0, 0, 22097, 0, 0, 0, 0, 0, 0, 0, 3691, 0, 0, 0, 0, 0, 21523, 0, 0, 0, 0, 0, 16867, 0, 0, 0, 16673, 0, 25789, 0, 0, 0, 6947, 0, 0, 0, 0, 0, 11909, 0, 1231, 0, 0, 0, 0, 0, 28213, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13973, 0, 14929, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15959, 0, 1471, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4391, 0, 1987, 0, 0, 0, 5633, 0, 14689, 0, 0, 0, 3077, 0, 0, 0, 0, 0, 13679, 0, 20221, 0, 0, 0, 0, 0, 28753, 0, 0, 0, 15371, 0, 7687, 0, 0, 0, 28913, 0, 0, 0, 0, 0, 1367, 0, 0, 0, 0, 0, 0, 0, 2971, 0, 0, 0, 0, 0, 29983, 0, 0, 0, 0, 0, 10687, 0, 0, 0, 0, 0, 20269, 0, 0, 0, 9197, 0, 0, 0, 0, 0, 26309, 0, 0, 0, 0, 0, 0, 0, 9463, 0, 0, 0, 6431, 0, 13177, 0, 0, 0, 22103, 0, 29269, 0, 0, 0, 13007, 0, 0, 0, 0, 0, 0, 0, 24301, 0, 0, 0, 0, 0, 7393, 0, 0, 0, 29291, 0, 0, 0, 0, 0, 0, 0, 29959, 0, 0, 0, 4457, 0, 0, 0, 0, 0, 11309, 0, 14821, 0, 0, 0, 0, 0, 7243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21083, 0, 24529, 0, 0, 0, 27317, 0, 0, 0, 0, 0, 4559, 0, 0, 0, 0, 0, 0, 0, 24433, 0, 0, 0, 15971, 0, 4027, 0, 0, 0, 10613, 0, 16189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9869, 0, 22051, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19511, 0, 17737, 0, 0, 0, 22853, 0, 12169, 0, 0, 0, 9797, 0, 0, 0, 0, 0, 1709, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22091, 0, 8527, 0, 0, 0, 16103, 0, 0, 0, 0, 0, 16487, 0, 0, 0, 0, 0, 0, 0, 15571, 0, 0, 0, 0, 0, 6673, 0, 0, 0, 20021, 0, 3337, 0, 0, 0, 0, 0, 24469, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16859, 0, 0, 0, 0, 0, 0, 0, 1483, 0, 0, 0, 26801, 0, 0, 0, 0, 0, 26513, 0, 0, 0, 0, 0, 25607, 0, 0, 0, 0, 0, 0, 0, 27871, 0, 0, 0, 0, 0, 22303, 0, 0, 0, 11651, 0, 0, 0, 0, 0, 9263, 0, 17989, 0, 0, 0, 1097, 0, 0, 0, 0, 0, 16979, 0, 23851, 0, 0, 0, 0, 0, 5983, 0, 0, 0, 0, 0, 15607, 0, 0, 0, 25283, 0, 0, 0, 0, 0, 6317, 0, 0, 0, 0, 0, 19469, 0, 15511, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2767, 0, 0, 0, 9143, 0, 25219, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24989, 0, 10081, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21611, 0, 2617, 0, 0, 0, 3953, 0, 15739, 0, 0, 0, 25547, 0, 0, 0, 0, 0, 0, 0, 10771, 0, 0, 0, 0, 0, 26863, 0, 0, 0, 25661, 0, 547, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12707, 0, 0, 0, 0, 0, 23189, 0, 19771, 0, 0, 0, 0, 0, 29353, 0, 0, 0, 7211, 0, 10057, 0, 0, 0, 0, 0, 7039, 0, 0, 0, 5207, 0, 0, 0, 0, 0, 2579, 0, 0, 0, 0, 0, 0, 0, 2323, 0, 0, 0, 7061, 0, 11287, 0, 0, 0, 0, 0, 19819, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25351, 0, 0, 0, 0, 0, 8023, 0, 0, 0, 10601, 0, 0, 0, 0, 0, 21863, 0, 8539, 0, 0, 0, 7817, 0, 0, 0, 0, 0, 9629, 0, 25111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11827, 0, 0, 0, 5543, 0, 8779, 0, 0, 0, 28367, 0, 0, 0, 0, 0, 0, 0, 14251, 0, 0, 0, 0, 0, 23173, 0, 0, 0, 16391, 0, 18517, 0, 0, 0, 12713, 0, 6319, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29399, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11951, 0, 0, 0, 0, 0, 12983, 0, 14269, 0, 0, 0, 24287, 0, 0, 0, 0, 0, 449, 0, 9721, 0, 0, 0, 0, 0, 11743, 0, 0, 0, 14951, 0, 9577, 0, 0, 0, 353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21929, 0, 21871, 0, 0, 0, 0, 0, 4993, 0, 0, 0, 27581, 0, 6697, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9617, 0, 0, 0, 0, 0, 17489, 0, 0, 0, 0, 0, 0, 0, 3583, 0, 0, 0, 0, 0, 23887, 0, 0, 0, 17063, 0, 24229, 0, 0, 0, 23717, 0, 0, 0, 0, 0, 0, 0, 18001, 0, 0, 0, 0, 0, 28603, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26063, 0, 5179, 0, 0, 0, 467, 0, 0, 0, 0, 0, 16349, 0, 19021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26947, 0, 0, 0, 0, 0, 25999, 0, 0, 0, 29207, 0, 0, 0, 0, 0, 17579, 0, 2281, 0, 0, 0, 0, 0, 11203, 0, 0, 0, 26681, 0, 25237, 0, 0, 0, 10193, 0, 23539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29171, 0, 14587, 0, 0, 0, 14243, 0, 10909, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26699, 0, 271, 0, 0, 0, 0, 0, 4393, 0, 0, 0, 18521, 0, 29947, 0, 0, 0, 29123, 0, 0, 0, 0, 0, 11447, 0, 0, 0, 0, 0, 7649, 0, 23131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29681, 0, 19087, 0, 0, 0, 0, 0, 16699, 0, 0, 0, 24737, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5683, 0, 0, 0, 7481, 0, 0, 0, 0, 0, 16013, 0, 28429, 0, 0, 0, 8597, 0, 0, 0, 0, 0, 0, 0, 2881, 0, 0, 0, 0, 0, 27553, 0, 0, 0, 28451, 0, 0, 0, 0, 0, 28163, 0, 3709, 0, 0, 0, 6137, 0, 0, 0, 0, 0, 12989, 0, 4111, 0, 0, 0, 0, 0, 28243, 0, 0, 0, 0, 0, 29047, 0, 0, 0, 26753, 0, 7099, 0, 0, 0, 437, 0, 0, 0, 0, 0, 149, 0, 0, 0, 0, 0, 0, 0, 9943, 0, 0, 0, 17021, 0, 28807, 0, 0, 0, 2843, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25409, 0, 2311, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1777, 0, 0, 0, 8153, 0, 1459, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11789, 0, 16021, 0, 0, 0, 0, 0, 1873, 0, 0, 0, 25241, 0, 10207, 0, 0, 0, 17153, 0, 0, 0, 0, 0, 29507, 0, 0, 0, 0, 0, 14369, 0, 23971, 0, 0, 0, 0, 0, 26203, 0, 0, 0, 14771, 0, 18037, 0, 0, 0, 0, 0, 29299, 0, 0, 0, 18647, 0, 0, 0, 0, 0, 29459, 0, 0, 0, 0, 0, 0, 0, 9883, 0, 0, 0, 0, 0, 11707, 0, 0, 0, 6563, 0, 8689, 0, 0, 0, 1247, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11021, 0, 0, 0, 0, 0, 29423, 0, 7279, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25379, 0, 8521, 0, 0, 0, 0, 0, 16273, 0, 0, 0, 0, 0, 19387, 0, 0, 0, 2393, 0, 16129, 0, 0, 0, 2537, 0, 0, 0, 0, 0, 0, 0, 1021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26051, 0, 18097, 0, 0, 0, 17753, 0, 7789, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1051, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11111, 0, 22147, 0, 0, 0, 23273, 0, 19519, 0, 0, 0, 16517, 0, 0, 0, 0, 0, 13889, 0, 901, 0, 0, 0, 0, 0, 5443, 0, 0, 0, 10961, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28247, 0, 0, 0, 0, 0, 17939, 0, 15991, 0, 0, 0, 0, 0, 6463, 0, 0, 0, 10991, 0, 0, 0, 0, 0, 0, 0, 3469, 0, 0, 0, 19907, 0, 0, 0, 0, 0, 16649, 0, 0, 0, 0, 0, 0, 0, 22483, 0, 0, 0, 3491, 0, 10657, 0, 0, 0, 22313, 0, 0, 0, 0, 0, 28757, 0, 0, 0, 0, 0, 0, 0, 991, 0, 0, 0, 0, 0, 18103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 233, 0, 4759, 0, 0, 0, 27347, 0, 0, 0, 0, 0, 24329, 0, 0, 0, 0, 0, 0, 0, 6823, 0, 0, 0, 0, 0, 6577, 0, 0, 0, 25073, 0, 17389, 0, 0, 0, 6737, 0, 0, 0, 0, 0, 17999, 0, 27271, 0, 0, 0, 0, 0, 24223, 0, 0, 0, 18071, 0, 21667, 0, 0, 0, 0, 0, 6529, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25829, 0, 16801, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26021, 0, 24247, 0, 0, 0, 27683, 0, 0, 0, 0, 0, 4547, 0, 0, 0, 0, 0, 4229, 0, 0, 0, 0, 0, 0, 0, 3973, 0, 0, 0, 9701, 0, 10627, 0, 0, 0, 15893, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7229, 0, 25021, 0, 0, 0, 0, 0, 10663, 0, 0, 0, 0, 0, 5857, 0, 0, 0, 0, 0, 5569, 0, 0, 0, 28937, 0, 0, 0, 0, 0, 6989, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7901, 0, 23047, 0, 0, 0, 7193, 0, 29899, 0, 0, 0, 2297, 0, 0, 0, 0, 0, 0, 0, 13591, 0, 0, 0, 0, 0, 22513, 0, 0, 0, 9131, 0, 0, 0, 0, 0, 22283, 0, 27439, 0, 0, 0, 7607, 0, 0, 0, 0, 0, 0, 0, 4531, 0, 0, 0, 0, 0, 3463, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1763, 0, 11299, 0, 0, 0, 19337, 0, 0, 0, 0, 0, 16949, 0, 12361, 0, 0, 0, 0, 0, 11413, 0, 0, 0, 18911, 0, 28387, 0, 0, 0, 15863, 0, 24589, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6089, 0, 23521, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11741, 0, 9337, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25127, 0, 0, 0, 0, 0, 21449, 0, 12871, 0, 0, 0, 0, 0, 613, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12113, 0, 0, 0, 0, 0, 12497, 0, 0, 0, 0, 0, 10799, 0, 15361, 0, 0, 0, 0, 0, 15073, 0, 0, 0, 9731, 0, 18457, 0, 0, 0, 0, 0, 6829, 0, 0, 0, 10037, 0, 0, 0, 0, 0, 9089, 0, 0, 0, 0, 0, 0, 0, 2113, 0, 0, 0, 23021, 0, 19687, 0, 0, 0, 0, 0, 5539, 0, 0, 0, 827, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26713, 0, 0, 0, 24041, 0, 0, 0, 0, 0, 1283, 0, 12319, 0, 0, 0, 11807, 0, 0, 0, 0, 0, 12149, 0, 0, 0, 0, 0, 0, 0, 1993, 0, 0, 0, 0, 0, 18547, 0, 0, 0, 28433, 0, 26419, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29339, 0, 14461, 0, 0, 0, 0, 0, 22543, 0, 0, 0, 20171, 0, 14107, 0, 0, 0, 28463, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26459, 0, 27091, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2291, 0, 5557, 0, 0, 0, 23693, 0, 11749, 0, 0, 0, 21767, 0, 0, 0, 0, 0, 0, 0, 61, 0, 0, 0, 0, 0, 21193, 0, 0, 0, 0, 0, 6637, 0, 0, 0, 27233, 0, 0, 0, 0, 0, 29717, 0, 0, 0, 0, 0, 17519, 0, 25651, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27791, 0, 0, 0, 0, 0, 0, 0, 7669, 0, 0, 0, 27257, 0, 0, 0, 0, 0, 24209, 0, 0, 0, 0, 0, 0, 0, 14713, 0, 0, 0, 16931, 0, 26407, 0, 0, 0, 19163, 0, 28219, 0, 0, 0, 27497, 0, 0, 0, 0, 0, 0, 0, 1411, 0, 0, 0, 0, 0, 6973, 0, 0, 0, 16691, 0, 0, 0, 0, 0, 21653, 0, 0, 0, 0, 0, 16217, 0, 0, 0, 0, 0, 24749, 0, 4951, 0, 0, 0, 0, 0, 5563, 0, 0, 0, 0, 0, 3427, 0, 0, 0, 10793, 0, 799, 0, 0, 0, 28997, 0, 0, 0, 0, 0, 25979, 0, 4801, 0, 0, 0, 0, 0, 1543, 0, 0, 0, 22271, 0, 12847, 0, 0, 0, 0, 0, 23329, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5459, 0, 16381, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1241, 0, 0, 0, 0, 0, 0, 0, 10489, 0, 0, 0, 20297, 0, 0, 0, 0, 0, 3389, 0, 20431, 0, 0, 0, 0, 0, 9223, 0, 0, 0, 1931, 0, 11047, 0, 0, 0, 29333, 0, 0, 0, 0, 0, 10817, 0, 0, 0, 0, 0, 3239, 0, 0, 0, 0, 0, 0, 0, 5623, 0, 0, 0, 26531, 0, 16567, 0, 0, 0, 0, 0, 29719, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20429, 0, 0, 0, 0, 0, 0, 0, 18913, 0, 0, 0, 18191, 0, 19057, 0, 0, 0, 6353, 0, 4909, 0, 0, 0, 18047, 0, 0, 0, 0, 0, 0, 0, 361, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14171, 0, 0, 0, 0, 0, 1913, 0, 10429, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15241, 0, 0, 0, 0, 0, 3043, 0, 0, 0, 0, 0, 19177, 0, 0, 0, 9743, 0, 0, 0, 0, 0, 11567, 0, 0, 0, 0, 0, 2669, 0, 22021, 0, 0, 0, 0, 0, 2803, 0, 0, 0, 26471, 0, 0, 0, 0, 0, 16283, 0, 19549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27509, 0, 19951, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4297, 0, 0, 0, 24113, 0, 26239, 0, 0, 0, 23867, 0, 0, 0, 0, 0, 18299, 0, 22531, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19991, 0, 26167, 0, 0, 0, 19673, 0, 0, 0, 0, 0, 19847, 0, 0, 0, 0, 0, 1979, 0, 26071, 0, 0, 0, 0, 0, 4573, 0, 0, 0, 0, 0, 1447, 0, 0, 0, 0, 0, 8719, 0, 0, 0, 11507, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21013, 0, 0, 0, 27221, 0, 24727, 0, 0, 0, 26723, 0, 1549, 0, 0, 0, 6077, 0, 0, 0, 0, 0, 0, 0, 12751, 0, 0, 0, 0, 0, 28183, 0, 0, 0, 17741, 0, 0, 0, 0, 0, 0, 0, 23029, 0, 0, 0, 6767, 0, 0, 0, 0, 0, 22859, 0, 5581, 0, 0, 0, 0, 0, 25723, 0, 0, 0, 0, 0, 9727, 0, 0, 0, 0, 0, 26839, 0, 0, 0, 15767, 0, 0, 0, 0, 0, 25349, 0, 0, 0, 0, 0, 0, 0, 26743, 0, 0, 0, 9041, 0, 0, 0, 0, 0, 15233, 0, 4639, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28349, 0, 26671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21821, 0, 22357, 0, 0, 0, 4373, 0, 2929, 0, 0, 0, 21347, 0, 0, 0, 0, 0, 4019, 0, 7621, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10541, 0, 20077, 0, 0, 0, 6863, 0, 0, 0, 0, 0, 21107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25153, 0, 0, 0, 10781, 0, 24127, 0, 0, 0, 0, 0, 29089, 0, 0, 0, 10247, 0, 0, 0, 0, 0, 19169, 0, 0, 0, 0, 0, 0, 0, 22273, 0, 0, 0, 8321, 0, 0, 0, 0, 0, 28823, 0, 13939, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9391, 0, 0, 0, 0, 0, 3823, 0, 0, 0, 16271, 0, 0, 0, 0, 0, 2333, 0, 0, 0, 0, 0, 5717, 0, 0, 0, 0, 0, 7739, 0, 14611, 0, 0, 0, 0, 0, 10603, 0, 0, 0, 0, 0, 8677, 0, 0, 0, 27593, 0, 0, 0, 0, 0, 17867, 0, 0, 0, 0, 0, 989, 0, 3961, 0, 0, 0, 0, 0, 11833, 0, 0, 0, 0, 0, 5077, 0, 0, 0, 27623, 0, 6739, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29609, 0, 12391, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26231, 0, 21097, 0, 0, 0, 24743, 0, 6499, 0, 0, 0, 13997, 0, 0, 0, 0, 0, 24599, 0, 3841, 0, 0, 0, 0, 0, 1453, 0, 0, 0, 7181, 0, 21337, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15017, 0, 0, 0, 0, 0, 0, 0, 22081, 0, 0, 0, 0, 0, 10873, 0, 0, 0, 28001, 0, 21607, 0, 0, 0, 0, 0, 9349, 0, 0, 0, 28307, 0, 0, 0, 0, 0, 7199, 0, 0, 0, 0, 0, 0, 0, 23113, 0, 0, 0, 25541, 0, 15907, 0, 0, 0, 13913, 0, 12889, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16111, 0, 0, 0, 0, 0, 26503, 0, 0, 0, 12911, 0, 0, 0, 0, 0, 0, 0, 10849, 0, 0, 0, 26297, 0, 0, 0, 0, 0, 389, 0, 6631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10163, 0, 27259, 0, 0, 0, 19127, 0, 0, 0, 0, 0, 0, 0, 18871, 0, 0, 0, 0, 0, 16243, 0, 0, 0, 18701, 0, 23137, 0, 0, 0, 24263, 0, 27109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1679, 0, 11131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 401, 0, 0, 0, 0, 0, 3743, 0, 25819, 0, 0, 0, 28907, 0, 0, 0, 0, 0, 23549, 0, 0, 0, 0, 0, 0, 0, 193, 0, 0, 0, 5711, 0, 337, 0, 0, 0, 18833, 0, 0, 0, 0, 0, 107, 0, 0, 0, 0, 0, 24239, 0, 26491, 0, 0, 0, 0, 0, 25783, 0, 0, 0, 9101, 0, 25177, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5939, 0, 0, 0, 0, 0, 0, 0, 15133, 0, 0, 0, 0, 0, 26197, 0, 0, 0, 10133, 0, 0, 0, 0, 0, 28337, 0, 0, 0, 0, 0, 0, 0, 8761, 0, 0, 0, 0, 0, 3193, 0, 0, 0, 3461, 0, 0, 0, 0, 0, 28373, 0, 9799, 0, 0, 0, 12017, 0, 0, 0, 0, 0, 27899, 0, 7471, 0, 0, 0, 0, 0, 8713, 0, 0, 0, 0, 0, 29257, 0, 0, 0, 0, 0, 7519, 0, 0, 0, 19967, 0, 0, 0, 0, 0, 22199, 0, 4591, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1271, 0, 13687, 0, 0, 0, 953, 0, 14299, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5879, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10691, 0, 5347, 0, 0, 0, 25793, 0, 29389, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2581, 0, 0, 0, 0, 0, 15943, 0, 0, 0, 9281, 0, 11467, 0, 0, 0, 3713, 0, 0, 0, 0, 0, 4517, 0, 0, 0, 0, 0, 12269, 0, 11581, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18131, 0, 23707, 0, 0, 0, 0, 0, 9769, 0, 0, 0, 27047, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24163, 0, 0, 0, 9791, 0, 367, 0, 0, 0, 0, 0, 709, 0, 0, 0, 27077, 0, 0, 0, 0, 0, 0, 0, 14431, 0, 0, 0, 0, 0, 29863, 0, 0, 0, 21521, 0, 0, 0, 0, 0, 2753, 0, 22189, 0, 0, 0, 26927, 0, 0, 0, 0, 0, 1439, 0, 8731, 0, 0, 0, 0, 0, 21313, 0, 0, 0, 0, 0, 3637, 0, 0, 0, 8273, 0, 27889, 0, 0, 0, 11987, 0, 0, 0, 0, 0, 2459, 0, 25171, 0, 0, 0, 0, 0, 12253, 0, 0, 0, 5471, 0, 10327, 0, 0, 0, 23633, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22567, 0, 0, 0, 26633, 0, 6079, 0, 0, 0, 27017, 0, 0, 0, 0, 0, 14099, 0, 20641, 0, 0, 0, 0, 0, 22663, 0, 0, 0, 6761, 0, 28687, 0, 0, 0, 19463, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2819, 0, 5491, 0, 0, 0, 0, 0, 7723, 0, 0, 0, 19391, 0, 20347, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23267, 0, 0, 0, 0, 0, 20219, 0, 0, 0, 0, 0, 0, 0, 14503, 0, 0, 0, 8531, 0, 4777, 0, 0, 0, 8873, 0, 13309, 0, 0, 0, 12797, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12223, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17873, 0, 18829, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29999, 0, 10831, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7837, 0, 0, 0, 23183, 0, 6889, 0, 0, 0, 21017, 0, 0, 0, 0, 0, 6659, 0, 24121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7571, 0, 8857, 0, 0, 0, 29303, 0, 26269, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3361, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1871, 0, 3667, 0, 0, 0, 27893, 0, 21829, 0, 0, 0, 9587, 0, 0, 0, 0, 0, 18509, 0, 19381, 0, 0, 0, 0, 0, 26233, 0, 0, 0, 29441, 0, 0, 0, 0, 0, 10013, 0, 0, 0, 0, 0, 527, 0, 0, 0, 0, 0, 29489, 0, 6751, 0, 0, 0, 0, 0, 24943, 0, 0, 0, 13301, 0, 10897, 0, 0, 0, 0, 0, 5779, 0, 0, 0, 8357, 0, 0, 0, 0, 0, 7409, 0, 0, 0, 0, 0, 0, 0, 24793, 0, 0, 0, 26591, 0, 0, 0, 0, 0, 26933, 0, 0, 0, 0, 0, 19517, 0, 0, 0, 0, 0, 0, 0, 5611, 0, 0, 0, 0, 0, 11173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11783, 0, 25549, 0, 0, 0, 8867, 0, 0, 0, 0, 0, 26639, 0, 15031, 0, 0, 0, 0, 0, 9133, 0, 0, 0, 0, 0, 18127, 0, 0, 0, 15833, 0, 28939, 0, 0, 0, 11357, 0, 0, 0, 0, 0, 11069, 0, 0, 0, 0, 0, 0, 0, 12673, 0, 0, 0, 8831, 0, 12427, 0, 0, 0, 0, 0, 11149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28139, 0, 21421, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7541, 0, 12697, 0, 0, 0, 29993, 0, 20569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22709, 0, 0, 0, 0, 0, 0, 0, 15523, 0, 0, 0, 14321, 0, 12937, 0, 0, 0, 8963, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28019, 0, 15781, 0, 0, 0, 0, 0, 15283, 0, 0, 0, 28421, 0, 28957, 0, 0, 0, 0, 0, 10189, 0, 0, 0, 10457, 0, 0, 0, 0, 0, 18539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25357, 0, 0, 0, 25673, 0, 11419, 0, 0, 0, 23087, 0, 0, 0, 0, 0, 0, 0, 15901, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13751, 0, 0, 0, 0, 0, 13043, 0, 18199, 0, 0, 0, 26087, 0, 0, 0, 0, 0, 17189, 0, 27631, 0, 0, 0, 0, 0, 8083, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13313, 0, 29779, 0, 0, 0, 21647, 0, 0, 0, 0, 0, 0, 0, 14671, 0, 0, 0, 0, 0, 22963, 0, 0, 0, 9671, 0, 9907, 0, 0, 0, 20483, 0, 26899, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10709, 0, 11971, 0, 0, 0, 0, 0, 0, 0, 0, 0, 191, 0, 13957, 0, 0, 0, 4163, 0, 0, 0, 0, 0, 27437, 0, 0, 0, 0, 0, 2969, 0, 3631, 0, 0, 0, 0, 0, 19093, 0, 0, 0, 0, 0, 11677, 0, 0, 0, 0, 0, 0, 0, 0, 0, 947, 0, 0, 0, 0, 0, 1559, 0, 26911, 0, 0, 0, 0, 0, 17383, 0, 0, 0, 2801, 0, 0, 0, 0, 0, 0, 0, 25309, 0, 0, 0, 797, 0, 0, 0, 0, 0, 27569, 0, 0, 0, 0, 0, 0, 0, 25213, 0, 0, 0, 25331, 0, 24307, 0, 0, 0, 0, 0, 26329, 0, 0, 0, 12377, 0, 0, 0, 0, 0, 0, 0, 20101, 0, 0, 0, 0, 0, 29023, 0, 0, 0, 12491, 0, 0, 0, 0, 0, 22073, 0, 23869, 0, 0, 0, 16427, 0, 0, 0, 0, 0, 5219, 0, 0, 0, 0, 0, 0, 0, 20473, 0, 0, 0, 0, 0, 9307, 0, 0, 0, 16883, 0, 1009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1619, 0, 19081, 0, 0, 0, 0, 0, 13303, 0, 0, 0, 1691, 0, 2557, 0, 0, 0, 743, 0, 17449, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14909, 0, 8611, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6911, 0, 7867, 0, 0, 0, 16763, 0, 0, 0, 0, 0, 26387, 0, 0, 0, 0, 0, 0, 0, 20851, 0, 0, 0, 0, 0, 25813, 0, 0, 0, 12431, 0, 29737, 0, 0, 0, 29543, 0, 0, 0, 0, 0, 11237, 0, 0, 0, 0, 0, 29069, 0, 14101, 0, 0, 0, 0, 0, 2473, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19219, 0, 0, 0, 18017, 0, 0, 0, 0, 0, 28829, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21551, 0, 14857, 0, 0, 0, 9923, 0, 18979, 0, 0, 0, 15947, 0, 0, 0, 0, 0, 0, 0, 24511, 0, 0, 0, 0, 0, 11593, 0, 0, 0, 28241, 0, 0, 0, 0, 0, 3173, 0, 0, 0, 0, 0, 18527, 0, 0, 0, 0, 0, 0, 0, 7261, 0, 0, 0, 0, 0, 17113, 0, 0, 0, 0, 0, 14977, 0, 0, 0, 15413, 0, 3109, 0, 0, 0, 22067, 0, 0, 0, 0, 0, 569, 0, 23281, 0, 0, 0, 0, 0, 22333, 0, 0, 0, 10721, 0, 17467, 0, 0, 0, 4943, 0, 16399, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17009, 0, 7141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3551, 0, 22987, 0, 0, 0, 0, 0, 12799, 0, 0, 0, 8747, 0, 0, 0, 0, 0, 24179, 0, 1951, 0, 0, 0, 0, 0, 11533, 0, 0, 0, 25031, 0, 0, 0, 0, 0, 3923, 0, 0, 0, 0, 0, 1577, 0, 0, 0, 0, 0, 21719, 0, 0, 0, 0, 0, 0, 0, 28723, 0, 0, 0, 28841, 0, 21187, 0, 0, 0, 0, 0, 20479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22739, 0, 0, 0, 0, 0, 0, 0, 21223, 0, 0, 0, 6641, 0, 577, 0, 0, 0, 27143, 0, 16459, 0, 0, 0, 22667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4931, 0, 0, 0, 0, 0, 20393, 0, 15049, 0, 0, 0, 3617, 0, 0, 0, 0, 0, 0, 0, 19861, 0, 0, 0, 0, 0, 23833, 0, 0, 0, 0, 0, 7627, 0, 0, 0, 12053, 0, 7309, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21149, 0, 3541, 0, 0, 0, 0, 0, 14353, 0, 0, 0, 1061, 0, 0, 0, 0, 0, 9353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18269, 0, 10711, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15941, 0, 27397, 0, 0, 0, 26423, 0, 829, 0, 0, 0, 5387, 0, 0, 0, 0, 0, 29849, 0, 10981, 0, 0, 0, 0, 0, 10273, 0, 0, 0, 0, 0, 28477, 0, 0, 0, 8123, 0, 0, 0, 0, 0, 10607, 0, 0, 0, 0, 0, 6599, 0, 28381, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19927, 0, 0, 0, 0, 0, 29509, 0, 0, 0, 29987, 0, 0, 0, 0, 0, 7829, 0, 0, 0, 0, 0, 0, 0, 25633, 0, 0, 0, 8741, 0, 15487, 0, 0, 0, 8243, 0, 20029, 0, 0, 0, 8387, 0, 0, 0, 0, 0, 0, 0, 15061, 0, 0, 0, 0, 0, 9703, 0, 0, 0, 8501, 0, 0, 0, 0, 0, 0, 0, 25339, 0, 0, 0, 29867, 0, 0, 0, 0, 0, 27479, 0, 24061, 0, 0, 0, 0, 0, 16483, 0, 0, 0, 0, 0, 14347, 0, 0, 0, 2603, 0, 19909, 0, 0, 0, 18077, 0, 0, 0, 0, 0, 6869, 0, 0, 0, 0, 0, 0, 0, 15193, 0, 0, 0, 11351, 0, 15577, 0, 0, 0, 0, 0, 6949, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19109, 0, 8191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14891, 0, 0, 0, 0, 0, 8993, 0, 21409, 0, 0, 0, 12107, 0, 0, 0, 0, 0, 22499, 0, 12241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12851, 0, 24697, 0, 0, 0, 18413, 0, 0, 0, 0, 0, 2627, 0, 0, 0, 0, 0, 0, 0, 27121, 0, 0, 0, 0, 0, 27463, 0, 0, 0, 29261, 0, 5647, 0, 0, 0, 0, 0, 10609, 0, 0, 0, 14867, 0, 0, 0, 0, 0, 12239, 0, 0, 0, 0, 0, 0, 0, 10723, 0, 0, 0, 29111, 0, 0, 0, 0, 0, 17273, 0, 18559, 0, 0, 0, 7127, 0, 0, 0, 0, 0, 0, 0, 14011, 0, 0, 0, 0, 0, 24613, 0, 0, 0, 27821, 0, 0, 0, 0, 0, 4643, 0, 29539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26219, 0, 26161, 0, 0, 0, 0, 0, 22153, 0, 0, 0, 0, 0, 10987, 0, 0, 0, 20663, 0, 0, 0, 0, 0, 22487, 0, 0, 0, 0, 0, 21779, 0, 24751, 0, 0, 0, 0, 0, 16453, 0, 0, 0, 0, 0, 28177, 0, 0, 0, 29933, 0, 11359, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11129, 0, 841, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23407, 0, 0, 0, 13193, 0, 18049, 0, 0, 0, 4757, 0, 0, 0, 0, 0, 29219, 0, 6151, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11801, 0, 9787, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3467, 0, 0, 0, 0, 0, 4709, 0, 15151, 0, 0, 0, 0, 0, 15493, 0, 0, 0, 9521, 0, 12367, 0, 0, 0, 0, 0, 27829, 0, 0, 0, 587, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4633, 0, 0, 0, 16301, 0, 27457, 0, 0, 0, 18533, 0, 15199, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4561, 0, 0, 0, 0, 0, 17263, 0, 0, 0, 1361, 0, 0, 0, 0, 0, 3383, 0, 3919, 0, 0, 0, 28607, 0, 0, 0, 0, 0, 11939, 0, 27421, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23377, 0, 0, 0, 0, 0, 29569, 0, 0, 0, 7577, 0, 0, 0, 0, 0, 0, 0, 391, 0, 0, 0, 0, 0, 18553, 0, 0, 0, 11771, 0, 0, 0, 0, 0, 28883, 0, 15559, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20159, 0, 15751, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2711, 0, 0, 0, 0, 0, 15293, 0, 16579, 0, 0, 0, 10427, 0, 0, 0, 0, 0, 25859, 0, 21271, 0, 0, 0, 0, 0, 2503, 0, 0, 0, 24191, 0, 11887, 0, 0, 0, 9593, 0, 0, 0, 0, 0, 4727, 0, 0, 0, 0, 0, 17309, 0, 0, 0, 0, 0, 0, 0, 14233, 0, 0, 0, 29891, 0, 15937, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21167, 0, 0, 0, 0, 0, 8249, 0, 0, 0, 0, 0, 0, 0, 5893, 0, 0, 0, 0, 0, 14647, 0, 0, 0, 12443, 0, 5749, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20311, 0, 0, 0, 0, 0, 14743, 0, 0, 0, 8081, 0, 0, 0, 0, 0, 21443, 0, 0, 0, 0, 0, 16637, 0, 0, 0, 0, 0, 18659, 0, 28261, 0, 0, 0, 0, 0, 13333, 0, 0, 0, 0, 0, 22327, 0, 0, 0, 0, 0, 12139, 0, 0, 0, 1487, 0, 0, 0, 0, 0, 3719, 0, 23071, 0, 0, 0, 0, 0, 22753, 0, 0, 0, 0, 0, 15997, 0, 0, 0, 19433, 0, 25849, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10499, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15311, 0, 23827, 0, 0, 0, 16553, 0, 20149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8219, 0, 25681, 0, 0, 0, 0, 0, 20563, 0, 0, 0, 20831, 0, 2227, 0, 0, 0, 15263, 0, 0, 0, 0, 0, 6827, 0, 0, 0, 0, 0, 0, 0, 13891, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8891, 0, 5227, 0, 0, 0, 0, 0, 12079, 0, 0, 0, 20117, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14923, 0, 0, 0, 28271, 0, 4987, 0, 0, 0, 27563, 0, 23809, 0, 0, 0, 29387, 0, 0, 0, 0, 0, 0, 0, 5191, 0, 0, 0, 0, 0, 18313, 0, 0, 0, 23831, 0, 0, 0, 0, 0, 0, 0, 24499, 0, 0, 0, 15377, 0, 0, 0, 0, 0, 22229, 0, 20281, 0, 0, 0, 0, 0, 23623, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12893, 0, 16339, 0, 0, 0, 2747, 0, 0, 0, 0, 0, 20939, 0, 29791, 0, 0, 0, 0, 0, 5323, 0, 0, 0, 7781, 0, 14947, 0, 0, 0, 5153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20789, 0, 13861, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4087, 0, 0, 0, 17393, 0, 17629, 0, 0, 0, 1607, 0, 0, 0, 0, 0, 7169, 0, 0, 0, 0, 0, 0, 0, 11113, 0, 0, 0, 27551, 0, 19447, 0, 0, 0, 7913, 0, 0, 0, 0, 0, 11027, 0, 0, 0, 0, 0, 5129, 0, 10111, 0, 0, 0, 0, 0, 28513, 0, 0, 0, 911, 0, 8797, 0, 0, 0, 0, 0, 10819, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8669, 0, 0, 0, 0, 0, 0, 0, 26053, 0, 0, 0, 13151, 0, 7087, 0, 0, 0, 1943, 0, 0, 0, 0, 0, 17417, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16843, 0, 0, 0, 22571, 0, 0, 0, 0, 0, 20183, 0, 23449, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11519, 0, 29311, 0, 0, 0, 0, 0, 27823, 0, 0, 0, 0, 0, 10147, 0, 0, 0, 11633, 0, 18439, 0, 0, 0, 11777, 0, 0, 0, 0, 0, 11279, 0, 26431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12191, 0, 27337, 0, 0, 0, 20063, 0, 17029, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26461, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13421, 0, 24457, 0, 0, 0, 9413, 0, 10279, 0, 0, 0, 11897, 0, 0, 0, 0, 0, 0, 0, 21691, 0, 0, 0, 0, 0, 7753, 0, 0, 0, 20201, 0, 0, 0, 0, 0, 14633, 0, 0, 0, 0, 0, 23627, 0, 0, 0, 0, 0, 4079, 0, 25231, 0, 0, 0, 0, 0, 15703, 0, 0, 0, 1751, 0, 15517, 0, 0, 0, 0, 0, 28879, 0, 0, 0, 10667, 0, 0, 0, 0, 0, 18959, 0, 0, 0, 0, 0, 0, 0, 13243, 0, 0, 0, 28901, 0, 22207, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7967, 0, 0, 0, 0, 0, 0, 0, 17161, 0, 0, 0, 0, 0, 13483, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16403, 0, 13999, 0, 0, 0, 29657, 0, 0, 0, 0, 0, 15089, 0, 19651, 0, 0, 0, 0, 0, 2203, 0, 0, 0, 0, 0, 22747, 0, 0, 0, 27383, 0, 19699, 0, 0, 0, 22907, 0, 0, 0, 0, 0, 13379, 0, 8791, 0, 0, 0, 0, 0, 14983, 0, 0, 0, 27311, 0, 23977, 0, 0, 0, 0, 0, 22699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28331, 0, 3457, 0, 0, 0, 18443, 0, 25189, 0, 0, 0, 16097, 0, 0, 0, 0, 0, 25019, 0, 0, 0, 0, 0, 0, 0, 6283, 0, 0, 0, 25871, 0, 1387, 0, 0, 0, 11273, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16469, 0, 27331, 0, 0, 0, 0, 0, 26833, 0, 0, 0, 3011, 0, 1237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15077, 0, 0, 0, 0, 0, 9299, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19451, 0, 18427, 0, 0, 0, 27983, 0, 16039, 0, 0, 0, 4607, 0, 0, 0, 0, 0, 0, 0, 4351, 0, 0, 0, 0, 0, 4033, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1493, 0, 29749, 0, 0, 0, 16847, 0, 0, 0, 0, 0, 21809, 0, 29941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4073, 0, 20539, 0, 0, 0, 10097, 0, 0, 0, 0, 0, 28499, 0, 7741, 0, 0, 0, 0, 0, 27583, 0, 0, 0, 21221, 0, 667, 0, 0, 0, 2003, 0, 15349, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14281, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20981, 0, 25507, 0, 0, 0, 8783, 0, 0, 0, 0, 0, 20507, 0, 0, 0, 0, 0, 7589, 0, 22111, 0, 0, 0, 0, 0, 9853, 0, 0, 0, 0, 0, 16297, 0, 0, 0, 23663, 0, 0, 0, 0, 0, 3257, 0, 0, 0, 0, 0, 13109, 0, 17671, 0, 0, 0, 0, 0, 5833, 0, 0, 0, 5111, 0, 30007, 0, 0, 0, 0, 0, 27619, 0, 0, 0, 19277, 0, 0, 0, 0, 0, 18329, 0, 0, 0, 0, 0, 0, 0, 27523, 0, 0, 0, 18401, 0, 0, 0, 0, 0, 0, 0, 14779, 0, 0, 0, 3137, 0, 0, 0, 0, 0, 0, 0, 24721, 0, 0, 0, 0, 0, 22093, 0, 0, 0, 14801, 0, 0, 0, 0, 0, 3593, 0, 14629, 0, 0, 0, 27977, 0, 0, 0, 0, 0, 7529, 0, 0, 0, 0, 0, 0, 0, 22783, 0, 0, 0, 0, 0, 20857, 0, 0, 0, 7643, 0, 12559, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24719, 0, 0, 0, 0, 0, 0, 0, 1753, 0, 0, 0, 22481, 0, 23347, 0, 0, 0, 19223, 0, 22069, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17219, 0, 13231, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18461, 0, 26347, 0, 0, 0, 19073, 0, 23299, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2371, 0, 0, 0, 0, 0, 7333, 0, 0, 0, 17051, 0, 2017, 0, 0, 0, 22613, 0, 0, 0, 0, 0, 15857, 0, 0, 0, 0, 0, 19829, 0, 4861, 0, 0, 0, 0, 0, 7093, 0, 0, 0, 9311, 0, 0, 0, 0, 0, 0, 0, 23839, 0, 0, 0, 29567, 0, 0, 0, 0, 0, 10349, 0, 0, 0, 0, 0, 0, 0, 26263, 0, 0, 0, 0, 0, 17167, 0, 0, 0, 28403, 0, 499, 0, 0, 0, 6707, 0, 0, 0, 0, 0, 0, 0, 26821, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2831, 0, 0, 0, 0, 0, 23963, 0, 0, 0, 0, 0, 6977, 0, 0, 0, 0, 0, 6269, 0, 331, 0, 0, 0, 0, 0, 21733, 0, 0, 0, 0, 0, 5737, 0, 0, 0, 26963, 0, 21589, 0, 0, 0, 24377, 0, 0, 0, 0, 0, 0, 0, 25591, 0, 0, 0, 0, 0, 3853, 0, 0, 0, 1481, 0, 29017, 0, 0, 0, 9563, 0, 18709, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21629, 0, 25621, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22031, 0, 27607, 0, 0, 0, 0, 0, 5869, 0, 0, 0, 11057, 0, 0, 0, 0, 0, 5699, 0, 22741, 0, 0, 0, 0, 0, 30013, 0, 0, 0, 27341, 0, 22597, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20057, 0, 0, 0, 0, 0, 12479, 0, 0, 0, 0, 0, 0, 0, 1003, 0, 0, 0, 21911, 0, 0, 0, 0, 0, 0, 0, 8929, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11189, 0, 0, 0, 0, 0, 0, 0, 14293, 0, 0, 0, 8951, 0, 5197, 0, 0, 0, 8663, 0, 7219, 0, 0, 0, 4187, 0, 0, 0, 0, 0, 0, 0, 11911, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23411, 0, 0, 0, 0, 0, 11153, 0, 26599, 0, 0, 0, 8237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12283, 0, 0, 0, 0, 0, 28417, 0, 0, 0, 503, 0, 11929, 0, 0, 0, 23117, 0, 0, 0, 0, 0, 23459, 0, 8161, 0, 0, 0, 0, 0, 5113, 0, 0, 0, 12611, 0, 0, 0, 0, 0, 11663, 0, 1069, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6719, 0, 22261, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20561, 0, 29707, 0, 0, 0, 19493, 0, 0, 0, 0, 0, 10007, 0, 0, 0, 0, 0, 20609, 0, 1741, 0, 0, 0, 0, 0, 14893, 0, 0, 0, 1511, 0, 21547, 0, 0, 0, 10433, 0, 0, 0, 0, 0, 22157, 0, 0, 0, 0, 0, 18149, 0, 16831, 0, 0, 0, 0, 0, 16123, 0, 0, 0, 0, 0, 22237, 0, 0, 0, 0, 0, 11029, 0, 0, 0, 20747, 0, 0, 0, 0, 0, 12449, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13361, 0, 3937, 0, 0, 0, 29033, 0, 10789, 0, 0, 0, 26867, 0, 0, 0, 0, 0, 0, 0, 8131, 0, 0, 0, 0, 0, 14323, 0, 0, 0, 20051, 0, 0, 0, 0, 0, 0, 0, 13789, 0, 0, 0, 2147, 0, 0, 0, 0, 0, 0, 0, 26371, 0, 0, 0, 0, 0, 28033, 0, 0, 0, 0, 0, 25897, 0, 0, 0, 7223, 0, 22219, 0, 0, 0, 11147, 0, 0, 0, 0, 0, 11489, 0, 0, 0, 0, 0, 0, 0, 5953, 0, 0, 0, 29831, 0, 20197, 0, 0, 0, 26783, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 629, 0, 28981, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17201, 0, 3877, 0, 0, 0, 0, 0, 23719, 0, 0, 0, 557, 0, 0, 0, 0, 0, 13259, 0, 23791, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5921, 0, 0, 0, 0, 0, 23033, 0, 0, 0, 0, 0, 23417, 0, 0, 0, 0, 0, 0, 0, 1711, 0, 0, 0, 0, 0, 20533, 0, 0, 0, 1541, 0, 10267, 0, 0, 0, 0, 0, 1369, 0, 0, 0, 26417, 0, 0, 0, 0, 0, 14549, 0, 0, 0, 0, 0, 0, 0, 13033, 0, 0, 0, 17561, 0, 0, 0, 0, 0, 8033, 0, 79, 0, 0, 0, 11747, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13063, 0, 0, 0, 18581, 0, 0, 0, 0, 0, 23123, 0, 4129, 0, 0, 0, 17267, 0, 0, 0, 0, 0, 28529, 0, 751, 0, 0, 0, 0, 0, 12913, 0, 0, 0, 0, 0, 29467, 0, 0, 0, 22973, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10229, 0, 6271, 0, 0, 0, 0, 0, 28003, 0, 0, 0, 0, 0, 457, 0, 0, 0, 23003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1889, 0, 21631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7751, 0, 16477, 0, 0, 0, 15503, 0, 22669, 0, 0, 0, 16307, 0, 0, 0, 0, 0, 10739, 0, 24631, 0, 0, 0, 0, 0, 13003, 0, 0, 0, 0, 0, 12097, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24257, 0, 0, 0, 0, 0, 9329, 0, 17461, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14141, 0, 817, 0, 0, 0, 0, 0, 18589, 0, 0, 0, 19067, 0, 0, 0, 0, 0, 18749, 0, 0, 0, 0, 0, 0, 0, 9253, 0, 0, 0, 27851, 0, 18217, 0, 0, 0, 53, 0, 3649, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6871, 0, 0, 0, 0, 0, 28813, 0, 0, 0, 22151, 0, 0, 0, 0, 0, 8003, 0, 6229, 0, 0, 0, 21677, 0, 0, 0, 0, 0, 16559, 0, 15871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27997, 0, 0, 0, 21713, 0, 22639, 0, 0, 0, 9887, 0, 0, 0, 0, 0, 0, 0, 21181, 0, 0, 0, 0, 0, 7003, 0, 0, 0, 14081, 0, 4657, 0, 0, 0, 0, 0, 17869, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10919, 0, 27301, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25811, 0, 0, 0, 0, 0, 19913, 0, 5029, 0, 0, 0, 1187, 0, 0, 0, 0, 0, 14309, 0, 25891, 0, 0, 0, 0, 0, 25603, 0, 0, 0, 26501, 0, 16507, 0, 0, 0, 21143, 0, 0, 0, 0, 0, 16277, 0, 0, 0, 0, 0, 19619, 0, 8011, 0, 0, 0, 0, 0, 16543, 0, 0, 0, 18341, 0, 27487, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25787, 0, 0, 0, 0, 0, 1319, 0, 0, 0, 0, 0, 0, 0, 24373, 0, 0, 0, 0, 0, 5407, 0, 0, 0, 893, 0, 10369, 0, 0, 0, 9857, 0, 0, 0, 0, 0, 0, 0, 24931, 0, 0, 0, 0, 0, 5503, 0, 0, 0, 19631, 0, 0, 0, 0, 0, 23753, 0, 21349, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7109, 0, 9781, 0, 0, 0, 0, 0, 24883, 0, 0, 0, 0, 0, 24637, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6107, 0, 0, 0, 0, 0, 24509, 0, 13831, 0, 0, 0, 0, 0, 27373, 0, 0, 0, 12821, 0, 9067, 0, 0, 0, 21743, 0, 439, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22049, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26137, 0, 0, 0, 5003, 0, 1669, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12839, 0, 27991, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25451, 0, 20707, 0, 0, 0, 6023, 0, 0, 0, 0, 0, 25307, 0, 0, 0, 0, 0, 23819, 0, 6961, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20441, 0, 26017, 0, 0, 0, 0, 0, 529, 0, 0, 0, 22427, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26473, 0, 0, 0, 19031, 0, 16537, 0, 0, 0, 2153, 0, 26119, 0, 0, 0, 22457, 0, 0, 0, 0, 0, 0, 0, 23671, 0, 0, 0, 0, 0, 9073, 0, 0, 0, 12281, 0, 0, 0, 0, 0, 14303, 0, 17569, 0, 0, 0, 17687, 0, 0, 0, 0, 0, 3749, 0, 11041, 0, 0, 0, 0, 0, 12073, 0, 0, 0, 0, 0, 15187, 0, 0, 0, 0, 0, 18649, 0, 0, 0, 21227, 0, 0, 0, 0, 0, 11699, 0, 11311, 0, 0, 0, 0, 0, 7633, 0, 0, 0, 851, 0, 0, 0, 0, 0, 9773, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9239, 0, 18481, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8707, 0, 0, 0, 28943, 0, 8389, 0, 0, 0, 13157, 0, 0, 0, 0, 0, 9479, 0, 2161, 0, 0, 0, 0, 0, 13423, 0, 0, 0, 16001, 0, 967, 0, 0, 0, 28703, 0, 0, 0, 0, 0, 15647, 0, 0, 0, 0, 0, 28229, 0, 0, 0, 0, 0, 0, 0, 16963, 0, 0, 0, 21701, 0, 29587, 0, 0, 0, 0, 0, 15439, 0, 0, 0, 4787, 0, 0, 0, 0, 0, 10979, 0, 0, 0, 0, 0, 0, 0, 16813, 0, 0, 0, 24701, 0, 25567, 0, 0, 0, 4253, 0, 24859, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28393, 0, 0, 0, 27191, 0, 0, 0, 0, 0, 13253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2279, 0, 20071, 0, 0, 0, 0, 0, 2413, 0, 0, 0, 0, 0, 3217, 0, 0, 0, 13943, 0, 23059, 0, 0, 0, 23327, 0, 0, 0, 0, 0, 22829, 0, 14881, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29647, 0, 0, 0, 8513, 0, 28579, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18041, 0, 12907, 0, 0, 0, 173, 0, 1039, 0, 0, 0, 347, 0, 0, 0, 0, 0, 29, 0, 14761, 0, 0, 0, 0, 0, 12373, 0, 0, 0, 1721, 0, 0, 0, 0, 0, 26183, 0, 0, 0, 0, 0, 25937, 0, 0, 0, 0, 0, 0, 0, 27541, 0, 0, 0, 0, 0, 27253, 0, 0, 0, 22541, 0, 27067, 0, 0, 0, 0, 0, 1159, 0, 0, 0, 3737, 0, 0, 0, 0, 0, 23579, 0, 0, 0, 0, 0, 0, 0, 4003, 0, 0, 0, 17351, 0, 26827, 0, 0, 0, 8453, 0, 0, 0, 0, 0, 10277, 0, 0, 0, 0, 0, 0, 0, 7921, 0, 0, 0, 0, 0, 1933, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16309, 0, 0, 0, 18107, 0, 0, 0, 0, 0, 5849, 0, 1171, 0, 0, 0, 0, 0, 4513, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1973, 0, 8149, 0, 0, 0, 13667, 0, 0, 0, 0, 0, 1829, 0, 13411, 0, 0, 0, 0, 0, 8053, 0, 0, 0, 29621, 0, 28597, 0, 0, 0, 0, 0, 13459, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23519, 0, 2941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16781, 0, 15007, 0, 0, 0, 9203, 0, 6709, 0, 0, 0, 20717, 0, 0, 0, 0, 0, 18089, 0, 0, 0, 0, 0, 0, 0, 24763, 0, 0, 0, 16631, 0, 22177, 0, 0, 0, 29753, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18779, 0, 1921, 0, 0, 0, 0, 0, 17593, 0, 0, 0, 14561, 0, 19717, 0, 0, 0, 0, 0, 21739, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27779, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24071, 0, 20737, 0, 0, 0, 21053, 0, 0, 0, 0, 0, 9227, 0, 0, 0, 0, 0, 0, 0, 25141, 0, 0, 0, 0, 0, 8653, 0, 0, 0, 25301, 0, 0, 0, 0, 0, 3803, 0, 4339, 0, 0, 0, 28397, 0, 0, 0, 0, 0, 3329, 0, 18391, 0, 0, 0, 0, 0, 19633, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22553, 0, 2059, 0, 0, 0, 857, 0, 0, 0, 0, 0, 3089, 0, 10051, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25841, 0, 19147, 0, 0, 0, 22793, 0, 6109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22259, 0, 7351, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2501, 0, 16267, 0, 0, 0, 20333, 0, 0, 0, 0, 0, 22817, 0, 0, 0, 0, 0, 0, 0, 24421, 0, 0, 0, 0, 0, 21403, 0, 0, 0, 0, 0, 27847, 0, 0, 0, 28283, 0, 0, 0, 0, 0, 26357, 0, 0, 0, 0, 0, 17729, 0, 6121, 0, 0, 0, 0, 0, 26623, 0, 0, 0, 23591, 0, 4597, 0, 0, 0, 0, 0, 20689, 0, 0, 0, 21587, 0, 0, 0, 0, 0, 29879, 0, 0, 0, 0, 0, 0, 0, 15973, 0, 0, 0, 20711, 0, 5827, 0, 0, 0, 0, 0, 17089, 0, 0, 0, 21617, 0, 0, 0, 0, 0, 0, 0, 6241, 0, 0, 0, 0, 0, 24403, 0, 0, 0, 7871, 0, 0, 0, 0, 0, 8213, 0, 3079, 0, 0, 0, 18737, 0, 0, 0, 0, 0, 26009, 0, 0, 0, 0, 0, 0, 0, 15853, 0, 0, 0, 0, 0, 25477, 0, 0, 0, 19193, 0, 3319, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27029, 0, 601, 0, 0, 0, 0, 0, 4063, 0, 0, 0, 10931, 0, 4867, 0, 0, 0, 9983, 0, 3589, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10289, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9221, 0, 17107, 0, 0, 0, 7523, 0, 27919, 0, 0, 0, 7907, 0, 0, 0, 0, 0, 0, 0, 6991, 0, 0, 0, 0, 0, 28123, 0, 0, 0, 28601, 0, 20497, 0, 0, 0, 24923, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8279, 0, 16411, 0, 0, 0, 0, 0, 18643, 0, 0, 0, 13931, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4157, 0, 0, 0, 0, 0, 1109, 0, 0, 0, 0, 0, 0, 0, 853, 0, 0, 0, 3071, 0, 10237, 0, 0, 0, 683, 0, 5119, 0, 0, 0, 18257, 0, 0, 0, 0, 0, 0, 0, 15271, 0, 0, 0, 0, 0, 23143, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12413, 0, 0, 0, 0, 0, 27767, 0, 0, 0, 0, 0, 10889, 0, 2641, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24217, 0, 0, 0, 17723, 0, 12349, 0, 0, 0, 12827, 0, 0, 0, 0, 0, 12119, 0, 18661, 0, 0, 0, 0, 0, 8473, 0, 0, 0, 13031, 0, 19777, 0, 0, 0, 21113, 0, 7159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27931, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12791, 0, 9127, 0, 0, 0, 0, 0, 8179, 0, 0, 0, 4127, 0, 0, 0, 0, 0, 10319, 0, 11191, 0, 0, 0, 0, 0, 20773, 0, 0, 0, 15791, 0, 27217, 0, 0, 0, 15473, 0, 0, 0, 0, 0, 22367, 0, 0, 0, 0, 0, 24029, 0, 0, 0, 0, 0, 0, 0, 19483, 0, 0, 0, 24221, 0, 2707, 0, 0, 0, 0, 0, 11239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1949, 0, 0, 0, 0, 0, 0, 0, 16603, 0, 0, 0, 2021, 0, 0, 0, 0, 0, 13283, 0, 25699, 0, 0, 0, 24977, 0, 0, 0, 0, 0, 0, 0, 16531, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25721, 0, 0, 0, 0, 0, 22703, 0, 17359, 0, 0, 0, 19787, 0, 0, 0, 0, 0, 0, 0, 1381, 0, 0, 0, 0, 0, 14593, 0, 0, 0, 0, 0, 9937, 0, 0, 0, 21293, 0, 23479, 0, 0, 0, 27737, 0, 0, 0, 0, 0, 16529, 0, 0, 0, 0, 0, 0, 0, 23593, 0, 0, 0, 3371, 0, 0, 0, 0, 0, 113, 0, 5689, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9029, 0, 26881, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2081, 0, 18157, 0, 0, 0, 21803, 0, 12379, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9059, 0, 13291, 0, 0, 0, 0, 0, 26443, 0, 0, 0, 6131, 0, 23857, 0, 0, 0, 3503, 0, 0, 0, 0, 0, 26777, 0, 0, 0, 0, 0, 8909, 0, 7591, 0, 0, 0, 0, 0, 20743, 0, 0, 0, 0, 0, 15307, 0, 0, 0, 0, 0, 15649, 0, 0, 0, 2267, 0, 0, 0, 0, 0, 23999, 0, 0, 0, 0, 0, 0, 0, 7153, 0, 0, 0, 0, 0, 6247, 0, 0, 0, 17483, 0, 22339, 0, 0, 0, 17627, 0, 0, 0, 0, 0, 0, 0, 10441, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24671, 0, 0, 0, 0, 0, 0, 0, 4549, 0, 0, 0, 20627, 0, 0, 0, 0, 0, 8999, 0, 19441, 0, 0, 0, 0, 0, 2623, 0, 0, 0, 0, 0, 16657, 0, 0, 0, 18773, 0, 10669, 0, 0, 0, 13457, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17503, 0, 0, 0, 20591, 0, 1717, 0, 0, 0, 1373, 0, 2329, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5249, 0, 17431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5651, 0, 8497, 0, 0, 0, 20543, 0, 16789, 0, 0, 0, 2867, 0, 0, 0, 0, 0, 24809, 0, 14551, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8231, 0, 6217, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11867, 0, 0, 0, 0, 0, 0, 0, 13261, 0, 0, 0, 0, 0, 22843, 0, 0, 0, 24641, 0, 0, 0, 0, 0, 0, 0, 19849, 0, 0, 0, 17177, 0, 0, 0, 0, 0, 2999, 0, 0, 0, 0, 0, 0, 0, 6103, 0, 0, 0, 19871, 0, 0, 0, 0, 0, 19583, 0, 20869, 0, 0, 0, 23297, 0, 0, 0, 0, 0, 0, 0, 25561, 0, 0, 0, 0, 0, 15373, 0, 0, 0, 7031, 0, 0, 0, 0, 0, 13883, 0, 15679, 0, 0, 0, 21887, 0, 0, 0, 0, 0, 21599, 0, 0, 0, 0, 0, 0, 0, 1363, 0, 0, 0, 0, 0, 20227, 0, 0, 0, 11423, 0, 0, 0, 0, 0, 4007, 0, 0, 0, 0, 0, 12539, 0, 10891, 0, 0, 0, 0, 0, 18763, 0, 0, 0, 0, 0, 18937, 0, 0, 0, 25313, 0, 22909, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20369, 0, 3151, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12371, 0, 18787, 0, 0, 0, 8573, 0, 0, 0, 0, 0, 20927, 0, 0, 0, 0, 0, 1499, 0, 15391, 0, 0, 0, 0, 0, 17623, 0, 0, 0, 23351, 0, 5167, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5777, 0, 0, 0, 0, 0, 20879, 0, 5911, 0, 0, 0, 0, 0, 27043, 0, 0, 0, 0, 0, 3127, 0, 0, 0, 0, 0, 109, 0, 0, 0, 9827, 0, 0, 0, 0, 0, 23369, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2441, 0, 6667, 0, 0, 0, 20843, 0, 24439, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29971, 0, 0, 0, 0, 0, 3403, 0, 0, 0, 3671, 0, 0, 0, 0, 0, 19553, 0, 24709, 0, 0, 0, 23987, 0, 0, 0, 0, 0, 0, 0, 18181, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9517, 0, 0, 0, 26333, 0, 24949, 0, 0, 0, 2957, 0, 0, 0, 0, 0, 0, 0, 9631, 0, 0, 0, 0, 0, 27793, 0, 0, 0, 2531, 0, 9277, 0, 0, 0, 10403, 0, 10939, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22469, 0, 18061, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28121, 0, 0, 0, 0, 0, 0, 0, 7339, 0, 0, 0, 19667, 0, 0, 0, 0, 0, 5069, 0, 7411, 0, 0, 0, 0, 0, 27913, 0, 0, 0, 19571, 0, 0, 0, 0, 0, 25763, 0, 0, 0, 0, 0, 7037, 0, 0, 0, 0, 0, 8069, 0, 12631, 0, 0, 0, 0, 0, 9613, 0, 0, 0, 20651, 0, 2077, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7307, 0, 0, 0, 0, 0, 3629, 0, 0, 0, 0, 0, 0, 0, 26683, 0, 0, 0, 0, 0, 16957, 0, 0, 0, 21683, 0, 21919, 0, 0, 0, 14477, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23983, 0, 0, 0, 10391, 0, 0, 0, 0, 0, 12203, 0, 25969, 0, 0, 0, 28187, 0, 0, 0, 0, 0, 9419, 0, 14401, 0, 0, 0, 0, 0, 15643, 0, 0, 0, 0, 0, 13087, 0, 0, 0, 0, 0, 23689, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12959, 0, 25381, 0, 0, 0, 0, 0, 8893, 0, 0, 0, 17441, 0, 11377, 0, 0, 0, 14813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12809, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26861, 0, 19207, 0, 0, 0, 7313, 0, 6289, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24389, 0, 16441, 0, 0, 0, 0, 0, 2083, 0, 0, 0, 0, 0, 23017, 0, 0, 0, 24503, 0, 0, 0, 0, 0, 16067, 0, 0, 0, 0, 0, 28439, 0, 9271, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25061, 0, 14467, 0, 0, 0, 0, 0, 21319, 0, 0, 0, 10877, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1063, 0, 0, 0, 551, 0, 7297, 0, 0, 0, 13703, 0, 14569, 0, 0, 0, 24767, 0, 0, 0, 0, 0, 0, 0, 25981, 0, 0, 0, 0, 0, 20623, 0, 0, 0, 3041, 0, 0, 0, 0, 0, 18923, 0, 19879, 0, 0, 0, 10757, 0, 0, 0, 0, 0, 8369, 0, 29521, 0, 0, 0, 0, 0, 2833, 0, 0, 0, 0, 0, 19807, 0, 0, 0, 24443, 0, 11719, 0, 0, 0, 23537, 0, 0, 0, 0, 0, 23249, 0, 2071, 0, 0, 0, 0, 0, 26113, 0, 0, 0, 3161, 0, 26497, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30029 };
__constant uchar WHL_STEP[5760] = { 8, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 3, 1, 5, 1, 3, 3, 2, 3, 3, 1, 5, 1, 2, 1, 6, 6, 2, 1, 2, 3, 1, 5, 3, 3, 3, 1, 3, 2, 1, 3, 2, 7, 2, 1, 2, 3, 4, 3, 5, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 4, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 2, 6, 4, 2, 1, 3, 2, 3, 6, 1, 2, 1, 6, 3, 2, 3, 3, 3, 1, 3, 5, 1, 2, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 3, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 4, 2, 3, 4, 5, 1, 5, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 7, 2, 1, 2, 3, 3, 1, 3, 2, 4, 5, 4, 2, 1, 2, 3, 4, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 2, 1, 5, 1, 5, 1, 3, 2, 4, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 2, 1, 3, 2, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 3, 1, 5, 6, 1, 2, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1, 3, 3, 5, 3, 4, 2, 1, 2, 1, 2, 4, 3, 6, 2, 3, 1, 6, 2, 1, 2, 3, 4, 2, 1, 2, 1, 6, 5, 1, 2, 1, 2, 3, 1, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 3, 3, 4, 3, 2, 1, 5, 1, 5, 1, 2, 1, 5, 1, 3, 2, 1, 5, 3, 1, 3, 2, 1, 3, 2, 3, 4, 3, 2, 1, 6, 5, 3, 1, 2, 3, 1, 6, 2, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 2, 6, 3, 4, 3, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 3, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 2, 6, 1, 3, 2, 1, 2, 3, 3, 1, 6, 3, 2, 9, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1, 6, 3, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 3, 1, 5, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 7, 5, 1, 2, 1, 2, 3, 1, 3, 5, 3, 3, 1, 5, 1, 3, 2, 3, 4, 2, 1, 2, 3, 4, 3, 5, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 3, 1, 2, 3, 4, 2, 1, 2, 6, 1, 3, 2, 1, 3, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 3, 1, 5, 1, 3, 5, 3, 3, 1, 3, 2, 1, 2, 1, 5, 1, 6, 2, 3, 3, 1, 6, 2, 3, 3, 1, 3, 2, 1, 3, 2, 7, 2, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 3, 5, 1, 3, 2, 3, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 3, 4, 2, 4, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 1, 2, 3, 7, 2, 1, 3, 5, 4, 2, 1, 2, 1, 2, 4, 5, 4, 2, 4, 3, 5, 1, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 2, 3, 3, 1, 3, 3, 3, 5, 4, 2, 1, 2, 3, 4, 3, 2, 4, 2, 3, 1, 3, 3, 2, 1, 2, 3, 6, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 4, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 4, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 3, 3, 2, 1, 5, 6, 1, 2, 1, 2, 3, 1, 3, 2, 1, 8, 1, 3, 2, 1, 5, 3, 4, 2, 1, 2, 1, 6, 3, 5, 1, 2, 3, 1, 6, 2, 1, 2, 4, 3, 2, 1, 2, 1, 6, 5, 3, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 5, 3, 4, 5, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 4, 2, 1, 2, 1, 5, 6, 1, 2, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 5, 1, 3, 2, 7, 3, 2, 1, 2, 4, 5, 3, 1, 2, 3, 1, 3, 3, 2, 1, 2, 4, 3, 2, 1, 2, 6, 1, 6, 2, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 3, 1, 5, 4, 3, 2, 1, 2, 3, 1, 5, 1, 2, 1, 6, 5, 1, 2, 3, 3, 1, 3, 2, 3, 3, 3, 1, 3, 3, 3, 2, 3, 6, 1, 2, 3, 4, 3, 5, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 1, 3, 2, 3, 4, 2, 3, 6, 1, 3, 2, 1, 3, 2, 3, 6, 1, 2, 1, 2, 7, 2, 3, 1, 2, 3, 1, 3, 5, 1, 5, 1, 3, 2, 1, 2, 6, 1, 5, 1, 2, 3, 3, 1, 3, 3, 2, 3, 3, 1, 5, 1, 3, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 1, 3, 2, 1, 3, 5, 1, 5, 3, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 3, 6, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 2, 1, 5, 1, 6, 3, 2, 3, 1, 5, 1, 2, 3, 3, 4, 2, 1, 3, 9, 2, 1, 2, 1, 2, 4, 5, 3, 1, 2, 4, 3, 3, 3, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 3, 3, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 6, 3, 1, 3, 3, 2, 1, 2, 3, 4, 3, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 5, 3, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 5, 3, 1, 3, 2, 1, 3, 5, 4, 2, 1, 2, 7, 3, 2, 3, 1, 2, 3, 1, 6, 2, 1, 2, 4, 5, 1, 2, 1, 5, 1, 5, 3, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 5, 3, 4, 3, 3, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 2, 1, 5, 1, 3, 2, 1, 2, 6, 1, 5, 1, 5, 7, 2, 1, 2, 1, 2, 4, 3, 5, 1, 2, 3, 1, 6, 2, 1, 2, 3, 1, 3, 2, 1, 2, 7, 6, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 2, 7, 2, 3, 1, 5, 1, 3, 3, 2, 1, 2, 3, 6, 1, 2, 1, 6, 5, 1, 2, 1, 5, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 3, 2, 3, 4, 8, 1, 2, 3, 1, 3, 3, 2, 1, 2, 4, 3, 2, 1, 3, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 8, 1, 3, 2, 4, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 3, 2, 3, 3, 1, 3, 3, 3, 2, 3, 6, 1, 3, 2, 4, 3, 2, 3, 1, 2, 7, 3, 2, 1, 5, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 3, 2, 4, 3, 2, 3, 3, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 3, 3, 3, 1, 3, 3, 2, 1, 5, 1, 3, 2, 1, 2, 6, 1, 5, 1, 3, 2, 3, 1, 3, 3, 2, 3, 3, 6, 1, 3, 5, 4, 2, 1, 2, 1, 2, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 4, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 4, 5, 1, 3, 3, 2, 3, 3, 4, 2, 1, 2, 1, 5, 1, 6, 2, 1, 2, 3, 1, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 7, 2, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 5, 3, 1, 3, 5, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 5, 3, 4, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 3, 3, 3, 1, 6, 2, 1, 2, 4, 3, 3, 2, 1, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 2, 3, 4, 3, 2, 1, 5, 3, 4, 3, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 6, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 2, 1, 5, 4, 2, 1, 2, 3, 3, 1, 5, 1, 3, 9, 2, 1, 2, 3, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 2, 6, 1, 6, 2, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 2, 7, 2, 3, 1, 5, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 2, 1, 11, 1, 2, 1, 2, 3, 1, 3, 2, 3, 6, 1, 3, 2, 1, 5, 3, 4, 2, 1, 2, 3, 4, 3, 5, 1, 2, 3, 1, 6, 2, 1, 2, 3, 1, 3, 2, 1, 3, 6, 5, 1, 2, 1, 2, 3, 4, 2, 1, 2, 6, 1, 3, 2, 1, 3, 2, 3, 6, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 5, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 2, 6, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 3, 3, 2, 4, 5, 3, 1, 2, 3, 4, 3, 2, 1, 6, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 5, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 3, 3, 3, 4, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 5, 3, 1, 3, 2, 1, 2, 3, 3, 4, 3, 3, 5, 6, 1, 2, 1, 2, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 3, 3, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 4, 2, 3, 1, 3, 3, 2, 1, 5, 4, 2, 1, 2, 6, 1, 5, 1, 2, 1, 2, 3, 1, 6, 2, 3, 4, 5, 1, 3, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 3, 3, 3, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 3, 1, 5, 1, 6, 2, 3, 4, 3, 2, 1, 2, 1, 5, 1, 8, 1, 2, 3, 1, 5, 1, 2, 3, 3, 1, 3, 2, 1, 5, 7, 3, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 3, 1, 5, 6, 1, 2, 1, 5, 1, 3, 2, 1, 2, 3, 3, 6, 1, 3, 2, 7, 2, 1, 2, 1, 6, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 3, 2, 6, 1, 6, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 4, 5, 1, 3, 5, 1, 2, 3, 1, 5, 1, 2, 1, 6, 5, 1, 2, 1, 2, 3, 4, 2, 3, 3, 3, 1, 3, 2, 1, 3, 5, 4, 2, 1, 2, 3, 4, 3, 5, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 3, 5, 1, 5, 1, 2, 1, 2, 7, 2, 1, 2, 6, 1, 3, 2, 1, 3, 2, 3, 6, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 2, 3, 3, 1, 3, 3, 2, 6, 1, 3, 2, 1, 5, 3, 4, 2, 1, 3, 2, 4, 3, 5, 1, 2, 3, 7, 2, 1, 5, 1, 3, 2, 1, 2, 1, 6, 5, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 2, 4, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 3, 5, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 3, 2, 1, 2, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 4, 3, 2, 3, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 6, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 4, 3, 2, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 5, 1, 2, 3, 4, 3, 3, 3, 2, 3, 6, 3, 1, 2, 4, 3, 2, 3, 1, 2, 4, 3, 3, 2, 3, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 6, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 9, 2, 3, 1, 2, 3, 1, 6, 2, 1, 6, 3, 2, 1, 2, 6, 1, 5, 3, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 5, 1, 5, 3, 4, 3, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 1, 3, 5, 6, 3, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 5, 1, 3, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 3, 1, 5, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 2, 6, 1, 6, 2, 1, 2, 3, 1, 5, 1, 2, 3, 3, 1, 3, 2, 1, 3, 2, 7, 2, 1, 2, 1, 2, 7, 2, 3, 1, 5, 1, 3, 3, 3, 2, 3, 1, 5, 3, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 3, 3, 3, 4, 2, 1, 3, 2, 3, 4, 2, 1, 2, 7, 3, 5, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 3, 3, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 2, 7, 3, 2, 1, 3, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 3, 3, 4, 3, 2, 3, 3, 1, 3, 2, 1, 3, 5, 4, 2, 1, 5, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 1, 5, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 3, 2, 3, 1, 3, 2, 1, 2, 6, 4, 2, 1, 8, 4, 2, 1, 2, 1, 2, 4, 8, 1, 2, 4, 6, 2, 1, 2, 3, 1, 3, 2, 3, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 5, 1, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 5, 1, 2, 4, 5, 3, 1, 2, 3, 1, 3, 3, 2, 3, 4, 3, 3, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 3, 1, 2, 3, 7, 2, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 3, 3, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 5, 3, 7, 2, 1, 2, 4, 3, 2, 3, 1, 2, 4, 3, 3, 3, 2, 3, 1, 3, 2, 1, 2, 1, 5, 6, 1, 3, 5, 1, 3, 2, 3, 3, 3, 1, 5, 1, 3, 2, 7, 2, 1, 2, 1, 2, 7, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 5, 1, 3, 2, 1, 2, 6, 1, 6, 2, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 5, 1, 3, 2, 3, 4, 2, 1, 2, 1, 2, 7, 2, 3, 1, 5, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 3, 6, 5, 3, 1, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 3, 4, 3, 5, 1, 5, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 3, 5, 1, 6, 2, 1, 2, 3, 6, 1, 2, 6, 1, 3, 2, 1, 3, 2, 9, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 6, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 3, 2, 3, 3, 4, 2, 1, 3, 2, 3, 4, 2, 1, 3, 6, 3, 2, 3, 3, 3, 4, 3, 2, 1, 5, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 4, 3, 2, 4, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 6, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 3, 2, 3, 4, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 2, 1, 2, 3, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 1, 5, 3, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 4, 2, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 2, 1, 5, 6, 1, 2, 1, 2, 3, 1, 5, 1, 2, 7, 3, 2, 1, 5, 3, 4, 2, 3, 1, 2, 4, 3, 5, 1, 2, 3, 1, 6, 2, 3, 3, 1, 3, 3, 2, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 5, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 6, 3, 2, 3, 1, 2, 3, 1, 6, 2, 1, 2, 7, 2, 1, 2, 1, 5, 1, 5, 3, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 5, 3, 4, 3, 2, 1, 2, 4, 5, 3, 1, 2, 3, 1, 3, 3, 3, 2, 4, 3, 2, 1, 2, 1, 5, 6, 1, 2, 1, 5, 1, 3, 2, 1, 5, 3, 1, 5, 4, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1, 6, 2, 3, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 6, 1, 2, 1, 2, 7, 2, 3, 1, 6, 3, 3, 2, 1, 2, 3, 1, 5, 1, 2, 1, 6, 5, 1, 3, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 3, 7, 5, 1, 2, 3, 1, 3, 3, 2, 1, 5, 1, 3, 2, 1, 8, 1, 5, 1, 2, 1, 2, 3, 4, 3, 2, 6, 1, 5, 1, 3, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 3, 3, 3, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 3, 1, 5, 4, 3, 2, 6, 1, 3, 2, 1, 2, 1, 5, 1, 6, 2, 1, 2, 4, 5, 1, 2, 3, 3, 1, 3, 2, 4, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 3, 3, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 2, 1, 2, 1, 6, 5, 3, 3, 4, 3, 3, 2, 1, 2, 3, 1, 3, 5, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 4, 3, 3, 3, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 5, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 6, 1, 2, 3, 4, 3, 2, 1, 3, 5, 4, 2, 3, 3, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 6, 2, 1, 5, 1, 5, 1, 2, 1, 2, 4, 3, 2, 1, 5, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 6, 3, 2, 3, 1, 2, 3, 1, 6, 2, 1, 2, 4, 3, 2, 1, 2, 1, 5, 6, 3, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1, 3, 2, 1, 5, 3, 4, 3, 2, 1, 2, 4, 3, 5, 1, 2, 3, 1, 6, 3, 2, 3, 1, 3, 2, 1, 2, 1, 11, 1, 2, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 5, 1, 3, 2, 7, 2, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 2, 6, 1, 6, 2, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 3, 2, 1, 2, 9, 3, 1, 5, 1, 3, 3, 2, 1, 2, 4, 5, 1, 2, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 6, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 2, 3, 4, 3, 5, 1, 2, 3, 4, 3, 2, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 1, 2, 3, 3, 4, 2, 1, 2, 6, 1, 3, 3, 3, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 4, 3, 5, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 5, 3, 1, 3, 5, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 2, 7, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 1, 3, 2, 1, 2, 6, 1, 5, 1, 2, 1, 2, 4, 3, 3, 2, 3, 3, 1, 5, 4, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 3, 3, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 4, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 2, 1, 2, 1, 2, 4, 5, 3, 1, 6, 3, 3, 2, 3, 3, 1, 3, 2, 3, 1, 5, 1, 6, 2, 1, 2, 3, 1, 5, 1, 2, 3, 3, 1, 3, 3, 3, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 3, 3, 2, 3, 4, 2, 3, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 5, 1, 2, 3, 7, 2, 1, 3, 2, 3, 4, 2, 3, 1, 6, 3, 2, 3, 3, 3, 1, 3, 3, 2, 3, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 3, 2, 1, 5, 4, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 4, 2, 3, 1, 8, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 3, 1, 2, 3, 4, 2, 1, 2, 3, 3, 1, 3, 2, 1, 8, 4, 3, 2, 3, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 5, 1, 2, 1, 5, 6, 1, 2, 1, 6, 3, 2, 1, 2, 3, 3, 1, 5, 1, 3, 2, 7, 2, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 6, 7, 2, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1, 3, 2, 1, 5, 3, 4, 2, 1, 2, 1, 2, 7, 5, 1, 5, 1, 6, 2, 1, 2, 3, 1, 5, 1, 2, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 3, 3, 4, 3, 5, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 3, 5, 1, 5, 1, 2, 1, 5, 4, 2, 1, 2, 6, 1, 3, 2, 1, 3, 2, 3, 7, 2, 1, 2, 4, 5, 3, 1, 2, 3, 1, 3, 5, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 3, 5, 3, 1, 3, 2, 4, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 2, 3, 4, 3, 2, 1, 2, 3, 3, 1, 3, 6, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 3, 3, 4, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 3, 3, 3, 4, 2, 1, 3, 5, 4, 2, 1, 2, 1, 2, 9, 3, 1, 2, 4, 3, 3, 2, 1, 5, 1, 3, 2, 3, 6, 1, 5, 1, 2, 1, 2, 3, 1, 3, 3, 2, 3, 3, 1, 6, 3, 2, 3, 4, 2, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 3, 2, 1, 2, 3, 4, 2, 1, 3, 5, 1, 5, 3, 1, 2, 3, 1, 5, 1, 2, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 3, 1, 5, 1, 3, 3, 2, 3, 3, 1, 3, 3, 2, 1, 5, 1, 6, 2, 1, 2, 3, 1, 5, 1, 5, 3, 1, 3, 2, 1, 3, 2, 7, 2, 1, 2, 1, 6, 3, 2, 3, 1, 2, 3, 1, 6, 3, 2, 4, 3, 2, 3, 1, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 5, 3, 4, 3, 2, 1, 6, 3, 2, 3, 3, 3, 1, 3, 3, 3, 2, 3, 1, 3, 3, 2, 1, 5, 6, 1, 2, 1, 5, 1, 3, 2, 1, 2, 3, 4, 5, 1, 3, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 4, 2, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 2, 6, 1, 6, 2, 1, 2, 3, 4, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 5, 4, 2, 1, 2, 3, 7, 2, 3, 1, 5, 1, 3, 3, 2, 1, 2, 3, 1, 5, 1, 2, 1, 6, 5, 1, 2, 1, 2, 4, 3, 2, 3, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 5, 4, 3, 5, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 3, 5, 6, 1, 2, 1, 2, 3, 4, 2, 1, 2, 6, 1, 3, 2, 1, 5, 3, 6, 1, 2, 1, 2, 4, 3, 5, 1, 2, 3, 1, 8, 1, 2, 3, 1, 3, 2, 1, 2, 1, 6, 5, 1, 2, 3, 3, 1, 3, 3, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 4, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 4, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 6, 3, 2, 1, 2, 3, 3, 1, 3, 2, 4, 2, 3, 4, 3, 2, 1, 2, 4, 5, 3, 3, 3, 1, 3, 3, 2, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 1, 3, 2, 3, 1, 3, 2, 1, 5, 3, 4, 2, 4, 5, 4, 2, 1, 2, 1, 2, 4, 5, 3, 1, 2, 7, 3, 2, 1, 2, 3, 1, 3, 2, 3, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 6, 1, 2, 1, 2, 4, 3, 2, 4, 2, 4, 3, 3, 2, 1, 2, 3, 4, 2, 1, 2, 1, 5, 1, 5, 1, 3, 2, 3, 1, 5, 3, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 3, 1, 2, 7, 2, 3, 1, 2, 3, 1, 3, 3, 2, 6, 1, 3, 3, 2, 6, 1, 5, 1, 2, 1, 2, 3, 1, 3, 3, 5, 3, 1, 5, 1, 3, 2, 3, 4, 2, 1, 2, 1, 6, 3, 2, 3, 1, 2, 3, 1, 6, 2, 1, 2, 4, 3, 2, 1, 3, 5, 1, 5, 3, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 2, 1, 5, 3, 4, 3, 2, 1, 2, 4, 3, 2, 3, 1, 5, 1, 3, 3, 5, 3, 1, 3, 2, 1, 2, 1, 5, 7, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 5, 1, 3, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 3, 6, 1, 6, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 2, 3, 4, 2, 1, 2, 1, 9, 2, 3, 6, 1, 3, 3, 2, 1, 2, 3, 1, 6, 2, 1, 6, 5, 1, 2, 1, 2, 3, 1, 3, 2, 3, 3, 4, 3, 2, 1, 3, 2, 3, 4, 2, 1, 2, 3, 4, 3, 6, 2, 3, 1, 3, 5, 1, 2, 3, 1, 3, 2, 1, 3, 5, 1, 5, 1, 2, 1, 2, 3, 4, 2, 1, 2, 6, 1, 3, 2, 1, 3, 5, 6, 1, 2, 3, 4, 3, 2, 3, 1, 2, 3, 1, 3, 5, 1, 2, 3, 1, 5, 1, 2, 1, 5, 1, 5, 1, 2, 3, 4, 3, 3, 2, 3, 3, 1, 3, 2, 1, 3, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 1, 5, 1, 3, 2, 1, 2, 1, 5, 6, 1, 2, 1, 2, 4, 3, 2, 1, 2, 6, 1, 3, 2, 6, 3, 4, 2, 1, 2, 1, 2, 4, 3, 5, 3, 3, 1, 6, 2, 1, 2, 3, 1, 3, 2, 1, 2, 1, 6, 5, 1, 3, 2, 3, 1, 3, 2, 1, 2, 3, 3, 4, 2, 1, 3, 5, 4, 2, 3, 1, 2, 4, 5, 3, 1, 2, 4, 3, 3, 2, 1, 2, 3, 4, 2, 3, 1, 5, 1, 5, 1, 2, 1, 5, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 4, 3, 2, 1, 2, 4, 5, 4, 2, 3, 1, 3, 3, 2, 1, 2, 7, 2, 1, 2, 1, 5, 1, 5, 1, 2, 1, 2, 3, 1, 5, 1, 5, 4, 3, 2, 4, 2, 3, 4, 2, 3, 1, 2, 4, 3, 2, 3, 1, 2, 3, 4, 3, 2, 3, 3, 1, 3, 3, 2, 1, 5, 1, 5, 1, 2, 3, 3, 1, 3, 2, 1, 5, 3, 1, 3, 3, 3, 2, 3, 6, 1, 2, 1, 6, 3, 2, 3, 1, 2, 4, 6, 2, 1, 2, 4, 3, 2, 1, 2, 1, 5, 1, 5, 4, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 2, 1, 5, 3, 4, 3, 2, 1, 2, 7, 2, 3, 1, 2, 3, 1, 3, 3, 3, 5, 1, 3, 2, 1, 2, 6, 6, 1, 2, 1, 5, 1, 3, 3, 2, 3, 3, 1, 5, 1, 3, 2, 7, 2, 1, 2, 1, 2, 4, 3, 2, 3, 1, 2, 3, 1, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 8, 1 };
#define W30_STEPS 0x13212123u   // steps/p for cycle positions 0..7 (low nibble first)
__constant uchar W30_IDX[30] = {
    0xFF, 0,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1,    0xFF, 0xFF,
    0xFF, 2,    0xFF, 3,    0xFF, 0xFF, 0xFF, 4,    0xFF, 5,
    0xFF, 0xFF, 0xFF, 6,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 7 };
__constant uchar W30_INV[30] = {
    0, 1,  0, 0, 0, 0, 0, 13, 0, 0,
    0, 11, 0, 7, 0, 0, 0, 23, 0, 19,
    0, 0,  0, 17, 0, 0, 0, 0, 0, 29 };

// Wheel-30 marks in an SLM tile: clears p*m for m coprime to 30 (the odd
// multipliers step 6,4,2,4,2,4,6,2: W30_STEPS) from tile-local bit `bit`
// (global bit gbit = tile_bit0 + bit, the first odd multiple) up to `end`,
// with SLM atomics.
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


// One-hit primes (p >= q_odds): the only odd multiple first = p m0 in the
// segment is marked iff m0 is coprime to 30030; p is coprime to 30030 and
// first is odd, so that is first coprime to 15015 — one bit of this table
// (bit x <-> gcd(x, 15015) == 1; generated) instead of the WHL_INV / WHL_IDX
// gathers and three modulos of mark_multiples_whl.
__constant uint CP15015[470] = { 0xA08B0116u, 0x6C20CA65u, 0x524C9699u, 0x94522CA2u, 0x61B44B49u, 0x912C30D8u, 0x845844A6u, 0x4912932Cu, 0x1A65B0C3u, 0x36894532u, 0x2D025948u, 0xC92186D2u, 0x121264B4u, 0x48B29168u, 0xC10DA652u, 0x30C92984u, 0x6502DA05u, 0x0B4C3499u, 0x92932D06u, 0x05B44A61u, 0x196930CAu, 0xA6590826u, 0x6894D324u, 0x52618489u, 0xB4184D32u, 0x29A60B04u, 0xCB2916C0u, 0x30CA6504u, 0x4494992Cu, 0x5209A253u, 0xA4C34892u, 0x2D22D861u, 0x5B44A291u, 0x961320A4u, 0x2530C369u, 0x884D125Au, 0x24590896u, 0x6186432Du, 0x9264B48Bu, 0xA2914D12u, 0x0DA25A4Cu, 0x816196C1u, 0x22CA2434u, 0x4C941961u, 0xD32D841Au, 0x944A6990u, 0x6832CA41u, 0x190C3681u, 0x90D325A6u, 0x65908B48u, 0x086D2292u, 0x86434C36u, 0x2916D028u, 0x5A65A44Bu, 0x96992830u, 0x2D82524Cu, 0xCB490650u, 0x329841B0u, 0x44A69925u, 0x932C204Bu, 0xB0C06986u, 0x4C305225u, 0x5948B609u, 0x84D32C26u, 0x64A4C921u, 0x914D129Au, 0xA60A4432u, 0x6116830Du, 0xDA0514C9u, 0xA4992522u, 0x2986194Cu, 0x426892D3u, 0x224841B4u, 0x0C309969u, 0x9225A65Bu, 0x148B6894u, 0x6D12C265u, 0x4B04B618u, 0x125109A4u, 0x64A4CB29u, 0x916C30D0u, 0xA2534C86u, 0x41965209u, 0xC821B48Bu, 0x86982D32u, 0x2C245A44u, 0x03699483u, 0x325A61B0u, 0x48B2814Cu, 0xD12D2659u, 0xB4C34182u, 0x69129A64u, 0x5A4C3201u, 0x96C20D84u, 0x2514C829u, 0x8961225Au, 0x86134CB4u, 0x4982D125u, 0xCA45B04Au, 0x36996122u, 0x2426430Cu, 0x0A6894D3u, 0x30D22594u, 0x48B6182Du, 0xD128864Au, 0xA4C92916u, 0x44309A65u, 0x13449699u, 0x06422DA2u, 0x6194C949u, 0x992C32D0u, 0xA45B40A4u, 0x28949328u, 0x5865B0C3u, 0x32894D02u, 0x25264948u, 0xCB618693u, 0x129A4434u, 0x0CA2906Du, 0x430DA458u, 0x34C96896u, 0x65225220u, 0x1B0CA499u, 0x92D20D82u, 0x45340A61u, 0x996932CAu, 0xA65A0416u, 0x6890D305u, 0xD061840Bu, 0xB6106C32u, 0x29A6494Cu, 0xCB091251u, 0x309A25A4u, 0x4C16886Cu, 0x422D0253u, 0xB48B0996u, 0x09325861u, 0x5344A299u, 0x86912CA4u, 0x44B0C369u, 0x0941325Au, 0x244948B6u, 0x6186D32Du, 0x9A2494CAu, 0xB2016D10u, 0x04A61A48u, 0xC94996C3u, 0x22DA2520u, 0x44B49945u, 0xC32C8613u, 0x944A6912u, 0x2930CA45u, 0x5A0C3499u, 0x14D32186u, 0x65948368u, 0x18652290u, 0xA24B4CB2u, 0x29069121u, 0xD26524C9u, 0x96986C10u, 0x2DA05308u, 0xCB099452u, 0x12D060B4u, 0x4426992Du, 0x930CA04Bu, 0xB0836196u, 0x4D324A45u, 0x58488689u, 0x84D32D26u, 0x60B44B60u, 0x916C121Au, 0xA6580CB0u, 0x4992C20Du, 0x8A2534C9u, 0x34896522u, 0x0D861B44u, 0x4A2992D2u, 0x324845A4u, 0x0C369169u, 0x5121A653u, 0x908B6884u, 0x6532D205u, 0x4B4C9618u, 0x16C12926u, 0x25A48A29u, 0x194C30DAu, 0xA2534892u, 0x4996502Cu, 0xD860A4C1u, 0xA6190932u, 0x2CA41B44u, 0xC2691683u, 0x325A6590u, 0x48B4890Du, 0xD3212619u, 0xB0C34086u, 0x6D029864u, 0x524CB291u, 0x968304A6u, 0x25344969u, 0x982500DAu, 0x841A0CB4u, 0x6912532Du, 0x8244B44Au, 0x26996132u, 0x05A25B0Cu, 0x896084D3u, 0x32C26594u, 0x4896186Cu, 0xD129A64Au, 0xA44B2914u, 0x6C10DA61u, 0x414C1691u, 0x92122DA2u, 0x41B4CB41u, 0x892D3298u, 0x84594426u, 0x2896922Cu, 0x5A61B0C3u, 0xB4894932u, 0x2D265108u, 0xCB6186D0u, 0x128A4430u, 0x44B29165u, 0xC30D264Au, 0x24C86992u, 0x6520D825u, 0x1B48B411u, 0x92532886u, 0x05A44869u, 0x984932CAu, 0x261B0416u, 0x6814C325u, 0xD265948Bu, 0xB0180D32u, 0x29A64B4Cu, 0xC32816D1u, 0x20D864A4u, 0x4C921968u, 0x1225A053u, 0x14CA4996u, 0x2D12D861u, 0x1B04A688u, 0x961324A4u, 0x64B0C349u, 0x894D2252u, 0x225148A6u, 0x6186D30Cu, 0x8A64B40Bu, 0x92912D10u, 0x0D865A4Cu, 0x496914C3u, 0x229A2134u, 0x4CB49164u, 0xD32D8219u, 0xB4484982u, 0x68328245u, 0x5B083689u, 0x94D22586u, 0x65948B28u, 0x18691252u, 0xA6434C36u, 0x21069129u, 0xDA45A0CBu, 0x86996430u, 0x2D22414Cu, 0xCA489652u, 0x305821B4u, 0x00A4192Du, 0x922CA45Bu, 0x20C16996u, 0x4D324A65u, 0x1940B689u, 0x02C30D26u, 0x6484CB61u, 0x912D1298u, 0xA65A4CB0u, 0x68964309u, 0xD82534C1u, 0x90986522u, 0x25061B4Cu, 0x0A699283u, 0x12CA4534u, 0x0C329849u, 0x5125A45Bu, 0x94836894u, 0x6932D264u, 0x4B4CA618u, 0x16D109A0u, 0x65848A21u, 0x896C30CAu, 0xA2124C96u, 0x49945225u, 0xD861B04Bu, 0xA6912C22u, 0x2CA45144u, 0x43499293u, 0x301A65B0u, 0x4836890Du, 0xD22D0658u, 0xB4CB2106u, 0x61121A64u, 0x524CB291u, 0x86C10DA2u, 0x0530C969u, 0x996422D2u, 0x060B48B4u, 0x2990D32Du, 0xCA05B44Au, 0x36196910u, 0x24A64B08u, 0x89609493u, 0x32D24584u, 0x44A6184Du, 0xC129A609u, 0x84CB2816u, 0x2C305A61u, 0x530C9499u, 0x965229A2u, 0x6134C348u, 0x992D22C8u, 0xA45B44A2u, 0x69929304u, 0x586520C3u, 0xB6804D12u, 0x29245908u, 0xCB618653u, 0x129224B4u, 0x4C82906Du, 0xC30D225Au, 0x34896196u, 0x4522CA25u, 0x1A4C9099u, 0x90D12D86u, 0x40B44269u, 0x116832CAu, 0xA4590C36u, 0x4890D325u, 0x9265948Au, 0x36086D32u, 0x21860B4Cu, 0xCB0916D0u, 0x205A65A0u, 0x4C969968u, 0x502CA253u, 0xB04B4986u, 0x2530D841u, 0x4A44A699u, 0x16932C04u, 0x25B0C269u, 0x0945301Au, 0x225948B6u, 0x6186D32Cu, 0x9264A4C9u, 0xB2914C12u, 0x0DA61A40u, 0xC92914C3u, 0x02DA2514u, 0x4C349925u, 0x9329860Bu, 0xB4426092u, 0x6922C845u, 0x590C2299u, 0x949325A6u, 0x61148B68u, 0x186D1252u, 0xA44B0CB4u, 0x29165029u, 0xC26424CBu, 0x86996C30u, 0x0DA25344u, 0x8B419252u, 0x32C861A4u, 0x4486912Du, 0x1328A45Au, 0xB0436994u, 0x4C325A21u, 0x59489680u, 0x82C32D26u, 0x64B48B41u, 0x814D129Au, 0x865A4C32u, 0x2996C00Du, 0x5A2434C1u, 0xB4196122u, 0x2D84134Cu, 0x4A6982D1u, 0x32CA4590u, 0x0C368961u, 0xD325260Bu, 0x908A4894u, 0x6D20D225u, 0x4348B618u, 0x16D128A6u, 0x65A44929u, 0x990C30DAu, 0x82124496u, 0x4916422Du, 0x986194CBu, 0xA4992532u, 0x28A05B44u, 0xC1688693u, 0x225065B0u, 0x48B2894Cu, 0x93252659u, 0x34CB2184u, 0x6D129A64u, 0x4A0C3290u, 0x96030DA4u, 0x0434C961u, 0x996522D2u, 0x82194CA4u, 0x6092D30Du, 0x4A41B40Au, 0x14996932u, 0x25A65A0Cu, 0x0B6894D2u, 0x32C26114u, 0x44B6106Cu, 0xD109A649u, 0xA4CB0912u, 0x6C309865u, 0x534C1681u, 0x96522982u, 0x21B4CB09u, 0x98293258u, 0x24534486u, 0x6986812Cu, 0x5A45B083u, 0xB2894532u, 0x2D264948u, 0xC26186D1u, 0x109A24B4u, 0x48B21169u, 0xC30CA45Au, 0x04C86996u, 0x4522DA25u, 0x1B44B489u, 0x12C32586u, 0x45904A49u, 0x992922CAu, 0xA6530C34u, 0x6894D320u, 0xD0659403u, 0xB2182D20u, 0x21864A4Cu, 0xCB291691u, 0x109A6524u, 0x0C969864u, 0x522DA053u, 0xB4C94986u, 0x2C32D060u, 0x5B40A699u, 0x94930CA0u, 0x65B08321u, 0x894D124Au, 0x26484816u, 0x6184932Du, 0x9A40B44Bu, 0xA2916C12u, 0x0DA6584Cu, 0xC94892C3u, 0x221A2134u, 0x0C348965u, 0xD22D861Bu, 0x344A2992u, 0x69324A45u, 0x53043699u, 0x80D105A6u, 0x00000068u };


// Multiplier wheel WHEEL_W (30030) marks of one prime in the global map,
// the step table in constant memory. Requires the
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
    if (p >= n) { map_clear_w(m32, bit, g); return; }   // bit < p: no second hit
    u32 w = WHL_IDX[m];
    for (;;) {
        map_clear_w(m32, bit, g);
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


// ---- Large-prime redundant-mark skip (LARGE_SKR, round 3) ----------------------
// A multi-hit mark p m (m coprime to 30030) with m divisible by r in {17, 19,
// 23, 29} is a multiple of r: the mask stage already clears every multiple of
// 3..61, so that L2 atomic (~30 ps, the packed kernel's bound) is redundant.
// WHL_SF[w] = WHL_STEP[w] | (M[w] mod r_i) << (4 + 6 i), M[w] = the w-th
// multiplier coprime to 30030; a prime's marks k = 0, 1, ... have
// m_k = m_0 + M[w_k] - M[w_0] (+ 30030 per wheel wrap), so m_k = 0 (mod r)
// iff M[w_k] = tau = M[w_0] - m_0 (mod r): four 6-bit fields compared at once
// (y = ((e ^ T) | H) - L: a field's top bit drops iff that field of e ^ T is 0).
#ifndef LARGE_SKR
#define LARGE_SKR 1
#endif
#ifndef LARGE_MQ
#define LARGE_MQ 1
#endif
#ifndef MQ_MAXP
#define MQ_MAXP 60000000u
#endif
// Only primes below SKR_MAXP: the target setup (~4 ps per prime) pays for
// primes with >= ~2 marks per segment (large_s per 1e11: all primes 310.4,
// < 130M 299.5, < 100M 294.7, < 80M 292.5, < 60M 290.7, < 40M 291.5).
#ifndef SKR_MAXP
#define SKR_MAXP 60000000u
#endif
#define SKR_R 215441u            // 17 * 19 * 23 * 29
#define SKR_H 0x8208200u         // top bit of each field
#define SKR_L 0x0410410u         // low bit of each field
__constant uint WHL_SF[5760] = {
    0x0410418u, 0x4514401u, 0x4D30022u, 0x5C01063u, 0x00628C1u, 0x08830E3u, 0x20E4832u, 0x3120C71u, 0x3941492u, 0x48124D3u, 0x6073C23u, 0x04D0881u,
    0x0CF10A3u, 0x2552902u, 0x3423831u, 0x3C44053u, 0x54A0CB2u, 0x64E1CF3u, 0x0943444u, 0x28508C2u, 0x3891901u, 0x40B2012u, 0x50F3051u, 0x5913872u,
    0x69548B7u, 0x2CC3482u, 0x3D044C3u, 0x5561011u, 0x5C11835u, 0x10B40D1u, 0x18D48F3u, 0x3131443u, 0x4822CA2u, 0x5863CE3u, 0x70C0833u, 0x1522091u,
    0x1D428B5u, 0x4470441u, 0x4C90C62u, 0x5CD1CA1u, 0x64F24C6u, 0x2040876u, 0x5103822u, 0x6144861u, 0x6960482u, 0x04314C3u, 0x1C92C11u, 0x24B3435u,
    0x4D510D3u, 0x6442823u, 0x08A4083u, 0x2100CE1u, 0x2921503u, 0x4012C52u, 0x5053C91u, 0x58744B3u, 0x70D1002u, 0x0D12047u, 0x4480C12u, 0x54C1C51u,
    0x5CE2472u, 0x6D234B3u, 0x1010004u, 0x3092083u, 0x48F38E5u, 0x7021471u, 0x0441C92u, 0x1482CD3u, 0x2CE4421u, 0x3500043u, 0x4D618A3u, 0x6453103u,
    0x08B4852u, 0x18F0C93u, 0x31524F1u, 0x3802C03u, 0x5064462u, 0x60A08A4u, 0x0D22815u, 0x34504B1u, 0x3C70CD5u, 0x6513461u, 0x6D33C82u, 0x08000C1u,
    0x10208E2u, 0x2061813u, 0x38C3074u, 0x59404F2u, 0x6811421u, 0x7031C42u, 0x0C72C86u, 0x3D31034u, 0x5C430B2u, 0x6C840F1u, 0x00A4803u, 0x1901462u,
    0x29424A3u, 0x4033D06u, 0x70F20B1u, 0x05128D2u, 0x1553801u, 0x1C04026u, 0x4CC24E3u, 0x6523C32u, 0x0160073u, 0x18518D3u, 0x30B3023u, 0x4914881u,
    0x51304A3u, 0x6821D05u, 0x1CC4491u, 0x24E00B2u, 0x35210F3u, 0x4C12841u, 0x5433063u, 0x6C948C3u, 0x10F1412u, 0x2132451u, 0x2952C75u, 0x5080801u,
    0x58A1025u, 0x0D438C1u, 0x15640E2u, 0x2430413u, 0x3C91C73u, 0x54F34D1u, 0x5D13CF3u, 0x0000843u, 0x18620A2u, 0x28A30E3u, 0x4104834u, 0x6011CB3u,
    0x0473402u, 0x14B4441u, 0x1CD0063u, 0x35318C2u, 0x4402903u, 0x5C64054u, 0x08E14D2u, 0x1922401u, 0x2142C23u, 0x3834482u, 0x48708C4u, 0x68F2833u,
    0x0D54092u, 0x1C204D4u, 0x3CA2442u, 0x4CE3483u, 0x65400E4u, 0x1052055u, 0x38F48F1u, 0x4110405u, 0x6842CA1u, 0x70634C3u, 0x14C0012u, 0x2501051u,
    0x2D21872u, 0x3D628B1u, 0x44130D5u, 0x6CB0C61u, 0x00D1485u, 0x2803C11u, 0x3024432u, 0x4060871u, 0x4881092u, 0x58C20D7u, 0x1C30CA2u, 0x2C71CE1u,
    0x3492502u, 0x44D3433u, 0x5D30093u, 0x00218F1u, 0x0842003u, 0x20A3862u, 0x30E48A4u, 0x5161C15u, 0x04944B4u, 0x2511822u, 0x3552861u, 0x3C03082u,
    0x4C440C3u, 0x64A0C14u, 0x1122C93u, 0x28144F2u, 0x3850823u, 0x50B2083u, 0x69138E3u, 0x0C00431u, 0x1420C53u, 0x2C824B3u, 0x44E3C02u, 0x5520041u,
    0x5D40862u, 0x6C118A3u, 0x1073101u, 0x1893815u, 0x41314B1u, 0x4951CD2u, 0x5822C01u, 0x6043425u, 0x14E10C1u, 0x1D018E5u, 0x4434071u, 0x4C54893u,
    0x64B14F2u, 0x00F2424u, 0x20044A3u, 0x3861102u, 0x48A2031u, 0x50C2852u, 0x6103893u, 0x05604F3u, 0x1C51C44u, 0x3CD3CC2u, 0x4D10101u, 0x5530813u,
    0x6C22075u, 0x20C4804u, 0x4141C82u, 0x5012CC1u, 0x58334E3u, 0x7090032u, 0x0CD1074u, 0x2D530F5u, 0x5480C83u, 0x6CE24E1u, 0x0102D02u, 0x1143C34u,
    0x30510B3u, 0x48B2803u, 0x6114062u, 0x71504A1u, 0x0400CC2u, 0x1441D03u, 0x2CA3451u, 0x34C3C73u, 0x4D208D2u, 0x5D61803u, 0x0053061u, 0x0873885u,
    0x3111416u, 0x60644D1u, 0x68800F2u, 0x04C1021u, 0x0CE1842u, 0x1D22883u, 0x34140E1u, 0x3C34903u, 0x5491452u, 0x64D2491u, 0x6CF2CB2u, 0x0933CF6u,
    0x38820A1u, 0x40A28C3u, 0x5904013u, 0x7160C75u, 0x2493403u, 0x3CF0064u, 0x5C020E2u, 0x6C43011u, 0x0063832u, 0x10A4871u, 0x18C0492u, 0x29014D4u,
    0x4813443u, 0x60700A6u, 0x1D33052u, 0x2C04093u, 0x4460CF1u, 0x4C81406u, 0x09444C2u, 0x1810901u, 0x2031012u, 0x3072053u, 0x48D38B4u, 0x6950C22u,
    0x0421C61u, 0x0C42482u, 0x1C834C1u, 0x24A3CE6u, 0x5562095u, 0x0894821u, 0x10B0442u, 0x20F1481u, 0x2911CA2u, 0x3952CE3u, 0x5044431u, 0x5860055u,
    0x0D028F1u, 0x1523002u, 0x2564043u, 0x3C50CA4u, 0x5CD2C13u, 0x0134472u, 0x10008B1u, 0x18210D3u, 0x3082822u, 0x40C3863u, 0x59204C4u, 0x0432432u,
    0x1473473u, 0x2CD00D1u, 0x34F08F2u, 0x4531824u, 0x64438A3u, 0x08A0502u, 0x18E1433u, 0x3142C91u, 0x39634B2u, 0x48344F3u, 0x6091041u, 0x68B1863u,
    0x0D130C3u, 0x2404812u, 0x3440C53u, 0x4CA24B3u, 0x6503C04u, 0x1011083u, 0x28728E2u, 0x38B3811u, 0x40D4035u, 0x6801CD1u, 0x70224F5u, 0x24C0081u,
    0x2CE08A2u, 0x3D218E1u, 0x4542105u, 0x6C74891u, 0x00904B3u, 0x18F1C02u, 0x2932C41u, 0x3153465u, 0x5881103u, 0x70E2851u, 0x0503073u, 0x1D648D2u,
    0x2C30C01u, 0x3451423u, 0x4CB2C82u, 0x5CF3CC3u, 0x0150814u, 0x2062893u, 0x38C40F2u, 0x4900421u, 0x5120C46u, 0x0C73D05u, 0x3511893u, 0x4C030F1u,
    0x5423802u, 0x6464843u, 0x08C14A1u, 0x10E1CC6u, 0x4030072u, 0x50710B1u, 0x58918D2u, 0x68D2804u, 0x1554883u, 0x2C414E2u, 0x3C82411u, 0x44A2C32u,
    0x54E3C71u, 0x5D04495u, 0x1032021u, 0x1852845u, 0x40F04E3u, 0x5951C31u, 0x6002452u, 0x7043493u, 0x14A00F1u, 0x1CC0803u, 0x3522062u, 0x45630A1u,
    0x4C138C5u, 0x00B1453u, 0x1912CB1u, 0x21334D3u, 0x3820022u, 0x4861066u, 0x0524013u, 0x1C10C74u, 0x3C92CF3u, 0x54F4442u, 0x6530881u, 0x6D510A2u,
    0x08220E4u, 0x28A4053u, 0x4100CB2u, 0x5141CF3u, 0x6833441u, 0x7053C62u, 0x0C900A3u, 0x24F1904u, 0x4403873u, 0x5C604D3u, 0x00C1C22u, 0x1102C63u,
    0x29644C1u, 0x30100E3u, 0x4871832u, 0x58B2871u, 0x60D3092u, 0x71140D1u, 0x05348F5u, 0x2C62486u, 0x5D20831u, 0x6541052u, 0x0012096u, 0x30D0441u,
    0x38F0C63u, 0x51524C2u, 0x6023501u, 0x6843C12u, 0x0480053u, 0x1CE18B3u, 0x3543001u, 0x3D63826u, 0x6CB1CE3u, 0x1113432u, 0x2154479u, 0x6904081u,
    0x71248A2u, 0x0D60CE1u, 0x1411502u, 0x2452434u, 0x44D44B3u, 0x5D31002u, 0x6C02043u, 0x10638A1u, 0x18840C2u, 0x28C0504u, 0x4942473u, 0x6033CD3u,
    0x0490822u, 0x14D1861u, 0x1CF2082u, 0x2D330C3u, 0x4424811u, 0x4C40433u, 0x64A1C92u, 0x00E2CD1u, 0x09034F2u, 0x1944426u, 0x48928E1u, 0x50B3106u,
    0x0C014B3u, 0x2462C02u, 0x34A3C43u, 0x4D008A1u, 0x55210C3u, 0x6C12812u, 0x0853853u, 0x20B04B3u, 0x3911C03u, 0x5003461u, 0x5823C83u, 0x70808E2u,
    0x0CC1811u, 0x14E2033u, 0x2D43892u, 0x3C148D3u, 0x5471424u, 0x00F34A2u, 0x11344E1u, 0x1950102u, 0x2821031u, 0x3041852u, 0x4082897u, 0x0561462u,
    0x14324A3u, 0x2C93D01u, 0x34B4415u, 0x5D520B1u, 0x64028D3u, 0x0864023u, 0x20C0C82u, 0x3101CC1u, 0x39224E5u, 0x6050071u, 0x6870895u, 0x1D13021u,
    0x2533842u, 0x3404887u, 0x6CE3455u, 0x20110F1u, 0x2831802u, 0x3872841u, 0x4093062u, 0x50D40A3u, 0x6930D01u, 0x7151413u, 0x1442C75u, 0x3CE0803u,
    0x5542063u, 0x6C338C1u, 0x00540E5u, 0x28F1C71u, 0x3112493u, 0x4803CF2u, 0x5840023u, 0x70A1884u, 0x1D23902u, 0x2D64831u, 0x3410452u, 0x4451493u,
    0x5CB2CF4u, 0x0930063u, 0x20218C5u, 0x48C4051u, 0x50E4872u, 0x6120CB3u, 0x0412401u, 0x0C32C23u, 0x2494483u, 0x3CF10E2u, 0x4D32011u, 0x5552832u,
    0x6423873u, 0x08804D1u, 0x10A0CF3u, 0x2902442u, 0x3943481u, 0x4163CA3u, 0x5850905u, 0x0CF3091u, 0x15138B5u, 0x3C41443u, 0x54A2CA1u, 0x5CC34C2u,
    0x6D04503u, 0x1161054u, 0x30730D2u, 0x40B4001u, 0x48D4822u, 0x5910C66u, 0x1463C11u, 0x1C84433u, 0x34E1092u, 0x45220D1u, 0x4D428F3u, 0x6434042u,
    0x0070483u, 0x18D1CE6u, 0x4820091u, 0x50408B2u, 0x60818F1u, 0x68A2002u, 0x04E3044u, 0x25604C3u, 0x3C51C12u, 0x4C92C53u, 0x64F44B1u, 0x6D100D5u,
    0x2042861u, 0x2863083u, 0x40C48E5u, 0x6962473u, 0x0C53CD3u, 0x24B0821u, 0x2CD1043u, 0x45328A2u, 0x54038E1u, 0x5C24102u, 0x6C60431u, 0x0080C55u,
    0x29234F1u, 0x3143C06u, 0x60920C2u, 0x70D3103u, 0x1534853u, 0x2C214B1u, 0x3441CD6u, 0x6500082u, 0x01410C3u, 0x1832813u, 0x3094071u, 0x38B4893u,
    0x51114F2u, 0x6152421u, 0x6802C43u, 0x0C644A2u, 0x1CA08E7u, 0x54140B2u, 0x64504F1u, 0x6C70C03u, 0x10D2462u, 0x21134A4u, 0x4020813u, 0x5882072u,
    0x68C30B3u, 0x0D24801u, 0x1540422u, 0x2411463u, 0x3C72CC4u, 0x5CF0033u, 0x0151893u, 0x18430F5u, 0x40E0C81u, 0x49014A3u, 0x6162D02u, 0x7033C33u,
    0x1490891u, 0x1CB10B5u, 0x4553841u, 0x4C04065u, 0x00A1D01u, 0x08C2412u, 0x1903451u, 0x2123C72u, 0x31600B4u, 0x5072023u, 0x68D3882u, 0x05148C1u,
    0x0D304E2u, 0x1C01413u, 0x3462C73u, 0x4CC44D4u, 0x6D41842u, 0x0812884u, 0x2894902u, 0x38D0C33u, 0x5132494u, 0x7044402u, 0x0C80841u, 0x14A1062u,
    0x24E20A1u, 0x2D028C6u, 0x5C50C73u, 0x00B24D2u, 0x10F3403u, 0x2950063u, 0x40418C3u, 0x58A3011u, 0x60C3833u, 0x0520493u, 0x1C11CF2u, 0x2C52C21u,
    0x3473442u, 0x44B4483u, 0x5D110E1u, 0x6531903u, 0x0823053u, 0x20848B2u, 0x30C0CF1u, 0x38E1405u, 0x6013CA1u, 0x68344C5u, 0x1CD2051u, 0x24F2873u,
    0x3D540D2u, 0x4C20403u, 0x6481C61u, 0x6CA2483u, 0x1103CE2u, 0x2140011u, 0x2960832u, 0x3831873u, 0x50930D7u, 0x1401CA2u, 0x2442CE1u, 0x2C63503u,
    0x44C0055u, 0x6D628F4u, 0x1874862u, 0x28B0CA1u, 0x30D14C2u, 0x4112501u, 0x4932C12u, 0x5803C54u, 0x04810D5u, 0x2D23864u, 0x4C30CE2u, 0x5C71C14u,
    0x08F3C93u, 0x21508F5u, 0x4883081u, 0x50A38A2u, 0x60E48E3u, 0x0541431u, 0x0D61C53u, 0x24534B2u, 0x34944F3u, 0x4CF1041u, 0x5511865u, 0x0844101u,
    0x1064815u, 0x39024B1u, 0x4122CD2u, 0x5163C01u, 0x5814422u, 0x6850863u, 0x0CB20C4u, 0x2D34032u, 0x3C00471u, 0x4420C92u, 0x5461CD3u, 0x6CC3423u,
    0x1120081u, 0x19408A3u, 0x3032103u, 0x4893853u, 0x60F04B5u, 0x1422C44u, 0x34A00C2u, 0x44E1101u, 0x4D01812u, 0x5D42853u, 0x00340B4u, 0x20B1423u,
    0x3912C82u, 0x4953CC4u, 0x6861032u, 0x04A2073u, 0x1D038D1u, 0x25240F3u, 0x3C10C43u, 0x54724A2u, 0x64B34E1u, 0x6CD3D02u, 0x0910033u, 0x2001896u,
    0x50C4841u, 0x58E0462u, 0x69214A1u, 0x7141CC5u, 0x2474451u, 0x2C90075u, 0x5532801u, 0x5D53022u, 0x6C24061u, 0x0044882u, 0x1080CC4u, 0x3102C35u,
    0x58308D1u, 0x60510F2u, 0x7092023u, 0x14F3884u, 0x3400D03u, 0x4C62452u, 0x5CA3491u, 0x64C3CB3u, 0x0920802u, 0x1961843u, 0x30530A4u, 0x50D0412u,
    0x6111454u, 0x0C234D2u, 0x1C64404u, 0x3CE1883u, 0x55430E2u, 0x6414013u, 0x0870C71u, 0x1091492u, 0x20D24D3u, 0x3933C21u, 0x4154443u, 0x58410A3u,
    0x70A2902u, 0x0CE3833u, 0x2540493u, 0x3C31CF1u, 0x4452403u, 0x5CB3C63u, 0x01108C2u, 0x1151901u, 0x1802015u, 0x40A48B6u, 0x7162C61u, 0x0413482u,
    0x14544C1u, 0x1C700E2u, 0x2CB1013u, 0x4512871u, 0x4D33093u, 0x64248F2u, 0x0060C21u, 0x0881448u, 0x4810831u, 0x5031053u, 0x68928B2u, 0x04D38F1u,
    0x0CF4005u, 0x3421CA3u, 0x4C83504u, 0x6D00872u, 0x09418B1u, 0x11620D2u, 0x2033001u, 0x2853826u, 0x5911CE3u, 0x7003435u, 0x24A10D1u, 0x2CC18F2u,
    0x3D02823u, 0x5564081u, 0x5C148A6u, 0x18D2C52u, 0x2913C91u, 0x31344B2u, 0x40008F4u, 0x6082863u, 0x04E40C2u, 0x1520501u, 0x1D40C12u, 0x2C11C51u,
    0x3432476u, 0x64F0825u, 0x18230C3u, 0x3084811u, 0x38A0432u, 0x48E1473u, 0x6142CD1u, 0x69634F3u, 0x0C50042u, 0x1C91081u, 0x24B18A2u, 0x34F28E3u,
    0x4D54033u, 0x6440C91u, 0x6C614B3u, 0x10C2C02u, 0x2103C41u, 0x2924465u, 0x5052103u, 0x68B3854u, 0x1530CD5u, 0x3C63461u, 0x4483C82u, 0x54C00C4u,
    0x0142033u, 0x1833892u, 0x28748D3u, 0x40D1421u, 0x48F1C42u, 0x5932C83u, 0x70244E1u, 0x0440103u, 0x1CA1853u, 0x35030B3u, 0x4D64802u, 0x5C30C43u,
    0x00924A4u, 0x2114412u, 0x3150851u, 0x3801072u, 0x48420B1u, 0x50628D5u, 0x0500466u, 0x3453411u, 0x3C73C32u, 0x4CB0071u, 0x54D0895u, 0x0803021u,
    0x1023843u, 0x28804A2u, 0x38C14E1u, 0x40E1D02u, 0x5122C33u, 0x6814493u, 0x0C710F1u, 0x1491805u, 0x3D340A1u, 0x45548C3u, 0x5C41412u, 0x6C82457u,
    0x3161023u, 0x4852882u, 0x58938C1u, 0x60B40E2u, 0x70F0414u, 0x1C02495u, 0x44A0023u, 0x5D01881u, 0x65220A2u, 0x01630E3u, 0x1854831u, 0x2070453u,
    0x38D1CB3u, 0x5133402u, 0x6004441u, 0x6820062u, 0x04610A4u, 0x24E3013u, 0x3D44872u, 0x4C10CB1u, 0x54314D2u, 0x6472406u, 0x21308C1u, 0x29510E6u,
    0x58A4092u, 0x68E04D1u, 0x7100CF2u, 0x0D41C23u, 0x2433481u, 0x2C53CA3u, 0x44B0902u, 0x54F1831u, 0x5D12055u, 0x10448F3u, 0x28A1441u, 0x30C1C63u,
    0x49234C2u, 0x5964504u, 0x0471872u, 0x14B28B3u, 0x2D14004u, 0x4C21482u, 0x5C624C1u, 0x6482CE2u, 0x00C3C11u, 0x08E4432u, 0x1920877u, 0x5094042u,
    0x60D0483u, 0x0531CE1u, 0x0D52505u, 0x3480094u, 0x5502003u, 0x6D63862u, 0x08348A1u, 0x10504C2u, 0x2091503u, 0x38F2C51u, 0x4113475u, 0x6841001u,
    0x7061822u, 0x0CA2861u, 0x14C3086u, 0x4411435u, 0x6CB3CD1u, 0x00D44F2u, 0x1110823u, 0x2802083u, 0x40638E1u, 0x4884103u, 0x60E0C52u, 0x7121C93u,
    0x14134F3u, 0x2C70043u, 0x44D18A1u, 0x4CF20C3u, 0x6553813u, 0x0840473u, 0x20A1CD2u, 0x30E2C03u, 0x4944466u, 0x0492811u, 0x0CB3032u, 0x1CF4073u,
    0x3550CD4u, 0x5462C43u, 0x6CC44A5u, 0x2162031u, 0x2812852u, 0x3853894u, 0x58D0C03u, 0x7132463u, 0x1423CC2u, 0x2460101u, 0x2C80812u, 0x3CC1853u,
    0x55230B1u, 0x5D438D3u, 0x0030422u, 0x1071461u, 0x1891C83u, 0x30F34E5u, 0x5821071u, 0x6041895u, 0x14E4021u, 0x1D04843u, 0x35614A2u, 0x44324E3u,
    0x5C93C34u, 0x09110B2u, 0x19520F3u, 0x3043846u, 0x6101D01u, 0x6922413u, 0x0C13C72u, 0x1C500B1u, 0x24708D3u, 0x3CD2022u, 0x4D13063u, 0x64048C6u,
    0x20C2C71u, 0x28E3492u, 0x39244D1u, 0x41400F2u, 0x5011027u, 0x14F4902u, 0x2530C33u, 0x3C22491u, 0x4442CB2u, 0x5483CF3u, 0x6CE0841u, 0x0101063u,
    0x19628C5u, 0x4090451u, 0x48B0C75u, 0x7153401u, 0x0403C23u, 0x1C60882u, 0x2CA18C1u, 0x34C20E2u, 0x4503016u, 0x00514D1u, 0x0871CF5u, 0x3114481u,
    0x39300A2u, 0x48010E3u, 0x6062833u, 0x04C4091u, 0x0CE48B3u, 0x2541403u, 0x3C32C62u, 0x4C73CA3u, 0x64D0903u, 0x0932051u, 0x1152875u, 0x3880401u,
    0x40A0C23u, 0x5902482u, 0x69434C3u, 0x0C30014u, 0x2CB2092u, 0x3CF30D1u, 0x45138F3u, 0x5C00442u, 0x6C41484u, 0x18C3503u, 0x3120052u, 0x4161093u,
    0x58528F1u, 0x6073002u, 0x70B4043u, 0x1510CA4u, 0x3422C13u, 0x4C84472u, 0x5CC08B1u, 0x64E10D5u, 0x1813861u, 0x2034083u, 0x3890CE2u, 0x48D1C11u,
    0x50F2433u, 0x6953C95u, 0x1C81821u, 0x24A2045u, 0x4D448E3u, 0x6431431u, 0x6C51C52u, 0x0892C94u, 0x2910003u, 0x4001862u, 0x50428A1u, 0x58630C2u,
    0x68A4103u, 0x0D00C53u, 0x25624B1u, 0x2C12CD3u, 0x4474422u, 0x54B0864u, 0x01328E2u, 0x1003813u, 0x2860474u, 0x48E24F2u, 0x5923421u, 0x6143C42u,
    0x7010081u, 0x04308A2u, 0x14718E4u, 0x34F3853u, 0x4D504B2u, 0x5C214F3u, 0x0082C46u, 0x3141101u, 0x3961813u, 0x5053073u, 0x68B48D2u, 0x04F0C03u,
    0x1D52463u, 0x3443CC1u, 0x3C644E3u, 0x54C1032u, 0x6502071u, 0x6D22892u, 0x09638D1u, 0x10140F5u, 0x38B1C81u, 0x40D24A6u, 0x7020853u, 0x14820B2u,
    0x24C30F3u, 0x3D24841u, 0x4540465u, 0x6C72D01u, 0x0093412u, 0x10D4453u, 0x29310B3u, 0x4022804u, 0x60A4882u, 0x70E0CC1u, 0x05014E3u, 0x1D62C39u,
    0x6512842u, 0x0153881u, 0x08040A2u, 0x18404E1u, 0x2060D02u, 0x30A1C34u, 0x5123CB5u, 0x0451843u, 0x1CB30A1u, 0x24D38C2u, 0x3514904u, 0x5421C73u,
    0x6C834D3u, 0x10E0023u, 0x2941882u, 0x38128C3u, 0x5074011u, 0x5894833u, 0x70F1492u, 0x0D324D3u, 0x2423C21u, 0x2C44445u, 0x54E20E1u, 0x5D02905u,
    0x1030491u, 0x1850CB2u, 0x2891CF1u, 0x30B2402u, 0x40F3443u, 0x59500A1u, 0x60008C3u, 0x0462012u, 0x14A3051u, 0x1CC3872u, 0x2D048B3u, 0x4561403u,
    0x5C52C64u, 0x08D00E3u, 0x2131833u, 0x3823092u, 0x48640D3u, 0x60C0C24u, 0x0D42CA2u, 0x1C13CE1u, 0x2434502u, 0x3470831u, 0x3C91056u, 0x6D54003u,
    0x1040C62u, 0x2081CA6u, 0x5140053u, 0x68318B1u, 0x70520D3u, 0x14B3823u, 0x2D10482u, 0x3D514C1u, 0x4401CE2u, 0x5442C13u, 0x6CA4474u, 0x19218F3u,
    0x3013042u, 0x4054081u, 0x48748A5u, 0x7112431u, 0x0532C55u, 0x2C608F1u, 0x3481002u, 0x44C2041u, 0x4CE2862u, 0x5D238A3u, 0x0010501u, 0x0830C15u,
    0x30D34B1u, 0x38F3CD2u, 0x4930003u, 0x6021864u, 0x0CA38E3u, 0x2500432u, 0x3541471u, 0x3D61C93u, 0x54534F2u, 0x6494423u, 0x08F1084u, 0x2803102u,
    0x3844033u, 0x50A0C91u, 0x58C14B2u, 0x69024F4u, 0x1414463u, 0x2C710C2u, 0x3CB2104u, 0x5D34072u, 0x6C004B3u, 0x1061C01u, 0x1882423u, 0x30E3C85u,
    0x5811813u, 0x7073073u, 0x14D48D1u, 0x1CF04F3u, 0x3551C43u, 0x4C434A2u, 0x5C844E1u, 0x64A0105u, 0x1942891u, 0x21630B5u, 0x4890C41u, 0x50B1462u,
    0x60F24A1u, 0x6912CC2u, 0x0553D03u, 0x1C40854u, 0x3CC28D2u, 0x4D03801u, 0x5524025u, 0x0851CC3u, 0x20B3411u, 0x28D3C33u, 0x4130892u, 0x50018D1u,
    0x58220F3u, 0x7083845u, 0x25214E4u, 0x4433452u, 0x5474491u, 0x5C900B2u, 0x6CD10F7u, 0x30448C3u, 0x48A1412u, 0x58E2453u, 0x7143CB1u, 0x05644D2u,
    0x1430803u, 0x2C92061u, 0x34B2886u, 0x6400C32u, 0x0041C71u, 0x0862492u, 0x18A34D4u, 0x3920845u, 0x60530E1u, 0x6873902u, 0x04B4831u, 0x0CD0455u,
    0x3402CF1u, 0x3C23405u, 0x64C10A3u, 0x0922901u, 0x1143012u, 0x2014054u, 0x40914D3u, 0x58F2C22u, 0x6933C61u, 0x7154482u, 0x0C208C3u, 0x2482013u,
    0x3CE3871u, 0x4504093u, 0x5D60CF2u, 0x6C31C21u, 0x0052445u, 0x28F00E3u, 0x4151834u, 0x60638B3u, 0x04C0403u, 0x1D21C62u, 0x2D62CA4u, 0x4C70013u,
    0x64D1872u, 0x01128B3u, 0x1804001u, 0x2024822u, 0x3060C63u, 0x48C24C1u, 0x50E2CE3u, 0x6944433u, 0x0C31093u, 0x24928F2u, 0x34D3823u, 0x4D30481u,
    0x5550CA3u, 0x6C42502u, 0x0883431u, 0x10A3C52u, 0x20E0091u, 0x29008B5u, 0x5033046u, 0x0CF1501u, 0x1511C12u, 0x2552C51u, 0x2C03475u, 0x54A1001u,
    0x5CC1823u, 0x0123082u, 0x11640C1u, 0x18148E2u, 0x2850C16u, 0x5913CD1u, 0x61344F5u, 0x1462081u, 0x1C828A5u, 0x4520437u, 0x0893C02u, 0x18D0041u,
    0x20F0862u, 0x31318A1u, 0x39520C2u, 0x4823104u, 0x68A0473u, 0x0D01CD5u, 0x3434461u, 0x3C50082u, 0x4C910C3u, 0x64F2811u, 0x6D13036u, 0x28614F2u,
    0x38A2421u, 0x40C2C42u, 0x5103C83u, 0x69608E1u, 0x7011103u, 0x1472852u, 0x24B3891u, 0x2CD40B2u, 0x3D104F7u, 0x0083CC6u, 0x3142072u, 0x40130B1u,
    0x48338D2u, 0x5874803u, 0x70D1461u, 0x04F1C83u, 0x1D534E2u, 0x2C24411u, 0x3440032u, 0x4481073u, 0x5CE28D3u, 0x0144021u, 0x0964843u, 0x20514A2u,
    0x30924E1u, 0x38B2D03u, 0x5114452u, 0x6150893u, 0x04420F4u, 0x24C4062u, 0x35004A3u, 0x4D61D01u, 0x5412412u, 0x6453457u, 0x2932022u, 0x3803063u,
    0x50648C1u, 0x58804E5u, 0x0D22C71u, 0x1543493u, 0x2C300F3u, 0x4491842u, 0x54D2881u, 0x5CF30A2u, 0x6D340E3u, 0x1020C36u, 0x40E3CF1u, 0x4904402u,
    0x5940841u, 0x6161066u, 0x1CB4015u, 0x4551CB1u, 0x4C024D2u, 0x5C43401u, 0x6463C25u, 0x19018C1u, 0x21220E3u, 0x3813832u, 0x4854873u, 0x60B14D3u,
    0x0512C23u, 0x1C04481u, 0x24200A3u, 0x3C81902u, 0x4CC2831u, 0x54E3053u, 0x6D448B2u, 0x0810CF3u, 0x2072444u, 0x40F44C3u, 0x5951012u, 0x6822053u,
    0x0C838B4u, 0x2D00C28u, 0x6C90011u, 0x00B0832u, 0x10F1873u, 0x29530D1u, 0x30038F3u, 0x4860443u, 0x60C1CA2u, 0x7102CE1u, 0x0523502u, 0x1564434u,
    0x34718B3u, 0x4CD3002u, 0x5D14041u, 0x6534863u, 0x08214C5u, 0x30C3C51u, 0x38E4475u, 0x6012001u, 0x6832822u, 0x0473861u, 0x0C94082u, 0x1CD04C3u,
    0x3531C14u, 0x5443C92u, 0x64800D1u, 0x6CA08F8u, 0x38348E1u, 0x4050503u, 0x58B1C52u, 0x68F2C94u, 0x1400002u, 0x2441043u, 0x3CA28A6u, 0x6D60C51u,
    0x0011472u, 0x10524B1u, 0x1872CD2u, 0x28B3C04u, 0x4931083u, 0x60228E2u, 0x7063813u, 0x14C0471u, 0x1CE0C92u, 0x2D21CD3u, 0x4413424u, 0x64908A5u,
    0x1933031u, 0x2153852u, 0x3024893u, 0x48814F1u, 0x50A1C03u, 0x6903462u, 0x05444A1u, 0x0D600C2u, 0x1C31101u, 0x2451815u, 0x4CF40B1u, 0x55148D5u,
    0x0842461u, 0x1062C82u, 0x20A3CC3u, 0x3900813u, 0x5162071u, 0x5812893u, 0x70740F3u, 0x14D0C42u, 0x2511C83u, 0x3C034E3u, 0x5460031u, 0x5C80853u,
    0x00E20B3u, 0x1943803u, 0x3030462u, 0x40714A3u, 0x58D2D06u, 0x14210B1u, 0x1C418D3u, 0x34A3022u, 0x44E4064u, 0x65614E3u, 0x0852C32u, 0x1893C73u,
    0x30F08D1u, 0x39110F2u, 0x4952027u, 0x0CC0D03u, 0x2522452u, 0x3563491u, 0x3C13CB5u, 0x64B1841u, 0x6CD2063u, 0x11338C2u, 0x2004901u, 0x2820412u,
    0x3861451u, 0x4081C75u, 0x6924401u, 0x7140025u, 0x24728C1u, 0x2C930E3u, 0x44F4832u, 0x5530C74u, 0x0042CF3u, 0x18A4442u, 0x28E0883u, 0x41420E3u,
    0x5833833u, 0x7090491u, 0x04B0CB3u, 0x1D12402u, 0x2D53444u, 0x4C608C2u, 0x5CA1903u, 0x0103054u, 0x20104D2u, 0x3051401u, 0x3871C22u, 0x48B2C61u,
    0x50D3482u, 0x61144C7u, 0x2483092u, 0x34C40D3u, 0x4D20C23u, 0x6412483u, 0x0873CE1u, 0x1094503u, 0x28F1053u, 0x41528B2u, 0x50238F1u, 0x5844005u,
    0x0CE1CA1u, 0x15024C3u, 0x2D63C12u, 0x3C30051u, 0x4450872u, 0x54918B6u, 0x1154861u, 0x1800485u, 0x40A2C11u, 0x48C3433u, 0x6120092u, 0x71610D3u,
    0x1452821u, 0x1C73043u, 0x34D48A3u, 0x4D31502u, 0x5C02433u, 0x0063C93u, 0x18C08F6u, 0x48138A1u, 0x50340C3u, 0x6890C15u, 0x1D334B4u, 0x3C40822u,
    0x4C81861u, 0x54A2082u, 0x64E30C1u, 0x6D038E2u, 0x0944814u, 0x2851C95u, 0x50F4423u, 0x6951081u, 0x70018A2u, 0x0C428E4u, 0x2CC4853u, 0x45214B3u,
    0x5C12C02u, 0x6C53C41u, 0x0074462u, 0x10B08A3u, 0x2912101u, 0x3132813u, 0x4824072u, 0x58604B4u, 0x04E2425u, 0x2C100C1u, 0x34308E5u, 0x5CD3073u,
    0x01348D1u, 0x09504F2u, 0x1821423u, 0x3082C81u, 0x38A34A3u, 0x5100102u, 0x6141031u, 0x6961852u, 0x0432893u, 0x1C940F3u, 0x34F0C41u, 0x3D11463u,
    0x5402CC3u, 0x6C64413u, 0x10C1072u, 0x21020B3u, 0x3963804u, 0x5870C82u, 0x68B1CC1u, 0x70D24E2u, 0x0D13411u, 0x1533C32u, 0x2400074u, 0x44820F3u,
    0x5CE3842u, 0x6D24884u, 0x1831D05u, 0x40D4491u, 0x48F00B3u, 0x6151803u, 0x0443062u, 0x14840A3u, 0x2CE0D03u, 0x4542454u, 0x64544D2u, 0x0090801u,
    0x08B1022u, 0x18F2061u, 0x2112885u, 0x4840411u, 0x5060C36u, 0x0D23CF2u, 0x1D60021u, 0x2410842u, 0x3451883u, 0x4CB30E1u, 0x54D3905u, 0x0801491u,
    0x1021CB2u, 0x2062CF3u, 0x38C4444u, 0x59418C3u, 0x7033012u, 0x0C74051u, 0x1494873u, 0x2CF14D2u, 0x3D32407u, 0x00A10E2u, 0x10E2013u, 0x2943871u,
    0x3164092u, 0x40304D4u, 0x60B2443u, 0x0513CA2u, 0x15500E3u, 0x2C41831u, 0x3462052u, 0x44A3093u, 0x5D048F1u, 0x6520403u, 0x0811C63u, 0x20734C5u,
    0x4911053u, 0x60028B1u, 0x68230D3u, 0x0C84825u, 0x35224C1u, 0x3D42CE5u, 0x6470871u, 0x6C91095u, 0x2133821u, 0x2954042u, 0x3820481u, 0x4040CA2u,
    0x5081CE3u, 0x68E3431u, 0x7103C53u, 0x15608B2u, 0x24318F1u, 0x2C52005u, 0x54F48A3u, 0x6D51504u, 0x1863472u, 0x28A44B1u, 0x30C00D3u, 0x4921822u,
    0x5962863u, 0x70540C4u, 0x1CD1432u, 0x2D12471u, 0x3532C92u, 0x4403CD1u, 0x4C244F6u, 0x08E28A3u, 0x2144102u, 0x3010433u, 0x4871C93u, 0x60D34F3u,
    0x0530041u, 0x0D50866u, 0x3CA3812u, 0x4CE4851u, 0x5500472u, 0x65414B4u, 0x1053423u, 0x28B0083u, 0x41118E2u, 0x5152811u, 0x5803035u, 0x0CA0CD1u,
    0x14C14F5u, 0x3D63C83u, 0x54508E1u, 0x5C71102u, 0x6CB2033u, 0x1113891u, 0x19340B3u, 0x3020C02u, 0x4061C41u, 0x4882462u, 0x58C34A3u, 0x7120104u,
    0x1C32073u, 0x34938D2u, 0x44D4801u, 0x4CF0425u, 0x0022CC3u, 0x1884414u, 0x3901893u, 0x51630F2u, 0x6034021u, 0x6854842u, 0x0490C84u, 0x2512D03u,
    0x3C04452u, 0x4C40894u, 0x6CC2802u, 0x0903843u, 0x21604A1u, 0x2810CC3u, 0x4072416u, 0x71308D2u, 0x0C01803u, 0x2463061u, 0x2C83883u, 0x44E04E2u,
    0x5521411u, 0x5D41C32u, 0x6C12C71u, 0x0033495u, 0x28D1026u, 0x58240E1u, 0x6044902u, 0x7080C31u, 0x04A1455u, 0x2D43CF4u, 0x4C51062u, 0x5C920A1u,
    0x64B28C2u, 0x00F3903u, 0x1950453u, 0x3041CB1u, 0x38624D5u, 0x6100061u, 0x6920883u, 0x0C120E9u, 0x5531CF2u, 0x6402C21u, 0x6C23442u, 0x0864483u,
    0x20C10E4u, 0x4143053u, 0x58348B2u, 0x6870CF3u, 0x0CD2441u, 0x14F2C62u, 0x2533CA3u, 0x3C20901u, 0x4441013u, 0x5CA2873u, 0x01040D2u, 0x1140401u,
    0x1960C22u, 0x2831C63u, 0x40934C1u, 0x48B3CE5u, 0x7151871u, 0x0402092u, 0x14430D6u, 0x4501481u, 0x4D21CA6u, 0x0870052u, 0x18B1091u, 0x20D18B2u,
    0x31128F4u, 0x5024863u, 0x68814C2u, 0x04C2501u, 0x0CE2C12u, 0x1D23C53u, 0x34108B3u, 0x4C72001u, 0x5492823u, 0x6CF4082u, 0x09304C1u, 0x1150CE3u,
    0x2842432u, 0x3883473u, 0x50E00D4u, 0x7162042u, 0x0C33081u, 0x14538A3u, 0x2CB0502u, 0x3CF1437u, 0x0060002u, 0x10A1043u, 0x29028A1u, 0x31230C5u,
    0x5850C51u, 0x6071473u, 0x04D2CD3u, 0x1D34422u, 0x2C00861u, 0x3421082u, 0x44620C3u, 0x5CC3811u, 0x64E4035u, 0x1811CD1u, 0x20324F2u, 0x3073421u,
    0x3893C4Bu, 0x1C84891u, 0x24A04B2u, 0x34E14F1u, 0x3D01C02u, 0x4D42C43u, 0x64344A1u, 0x6C500C3u, 0x10B1812u, 0x20F2853u, 0x39540B6u, 0x68A2461u,
    0x70C2C83u, 0x15244E2u, 0x2560811u, 0x2C11035u, 0x54B38D3u, 0x6D10424u, 0x18224A2u, 0x28634E1u, 0x3083D02u, 0x40C0033u, 0x5921894u, 0x0433803u,
    0x1C90465u, 0x4532D01u, 0x4D53412u, 0x5C24453u, 0x00810B1u, 0x08A18D6u, 0x3964882u, 0x4830CC1u, 0x50514E2u, 0x6092413u, 0x04F3C71u, 0x0D14493u,
    0x24010F2u, 0x3442021u, 0x3C62843u, 0x54C40A6u, 0x1012455u, 0x38B00F1u, 0x40D0802u, 0x5111841u, 0x5932062u, 0x68030A3u, 0x0C64904u, 0x2CE1C72u,
    0x3D22CB1u, 0x45434D2u, 0x5414406u, 0x10D28C1u, 0x18F30E3u, 0x3154832u, 0x4020C71u, 0x4841493u, 0x60A2CF2u, 0x70E3C23u, 0x1540886u, 0x4493833u,
    0x5CF0491u, 0x6510CB2u, 0x0151CF4u, 0x2063C63u, 0x38C08C2u, 0x4901903u, 0x6163051u, 0x6813872u, 0x04548B3u, 0x1CB1401u, 0x24D1C23u, 0x3D33485u,
    0x6461011u, 0x6C81832u, 0x08C2873u, 0x21240D4u, 0x4031442u, 0x5072481u, 0x5892CA2u, 0x68D3CE1u, 0x70F4505u, 0x2422091u, 0x2C428B5u, 0x54E0441u,
    0x5D00C62u, 0x6D41CA6u, 0x2890051u, 0x30B0873u, 0x49120D3u, 0x6003822u, 0x7044863u, 0x14A14C3u, 0x2D02C11u, 0x3523433u, 0x4C10092u, 0x5C510D1u,
    0x64718F3u, 0x08D3042u, 0x1914083u, 0x3000CE4u, 0x5082C53u, 0x68E44B3u, 0x0D41002u, 0x1C12044u, 0x3C940C5u, 0x6531C53u, 0x08234B1u, 0x1043CD2u,
    0x2080003u, 0x38E1864u, 0x59638E3u, 0x7050432u, 0x0C91471u, 0x14B1C96u, 0x4400043u, 0x5C618A2u, 0x6CA28E1u, 0x00C3102u, 0x1104031u, 0x1924855u,
    0x40524F1u, 0x4872C05u, 0x71108A1u, 0x05310C2u, 0x1402101u, 0x1C22812u, 0x2C63854u, 0x4CE0CD3u, 0x6542422u, 0x0013461u, 0x0833C85u, 0x30D1813u,
    0x4933071u, 0x5153893u, 0x68404F2u, 0x0481424u, 0x25034A2u, 0x35444E3u, 0x4C31034u, 0x6CB30B2u, 0x08F40F1u, 0x1114802u, 0x2150C41u, 0x2801462u,
    0x38424A4u, 0x58C4413u, 0x7121072u, 0x0D620B3u, 0x2453803u, 0x3CB0463u, 0x5511CC4u, 0x0023C33u, 0x1880892u, 0x28C18D1u, 0x30E20F2u, 0x4123023u,
    0x5814881u, 0x60304A3u, 0x0491D02u, 0x14D2C31u, 0x1CF3452u, 0x2D34491u, 0x35500B5u, 0x5C82841u, 0x64A3065u, 0x1940D01u, 0x2161415u, 0x4893CB3u,
    0x60F0801u, 0x6911023u, 0x0C02882u, 0x1C438C1u, 0x24640E2u, 0x34A0413u, 0x4D01C73u, 0x65634D4u, 0x1070843u, 0x28D20A3u, 0x4133905u, 0x6861496u,
    0x2524441u, 0x2D40062u, 0x3C110A1u, 0x44318C2u, 0x5472904u, 0x00F4875u, 0x2822403u, 0x4083C61u, 0x48A4482u, 0x58E08C4u, 0x0562833u, 0x1C54093u,
    0x34B0CF2u, 0x44F1C21u, 0x4D12442u, 0x5D53483u, 0x00400E1u, 0x0860903u, 0x20C2052u, 0x3103093u, 0x49648F1u, 0x5010405u, 0x04B2CA1u, 0x0CD34C5u,
    0x3401051u, 0x3C21873u, 0x54830D2u, 0x64C4003u, 0x0920C61u, 0x1141483u, 0x2832CE2u, 0x3873C13u, 0x50D0873u, 0x69320D3u, 0x0C23821u, 0x1444043u,
    0x2CA0CA3u, 0x4502503u, 0x5D63C52u, 0x6C30093u, 0x10918F4u, 0x3113862u, 0x41548A1u, 0x48004C2u, 0x5841501u, 0x6061C12u, 0x70A2C57u, 0x3411822u,
    0x4452864u, 0x64D48E2u, 0x0110C13u, 0x1802471u, 0x2022C93u, 0x38844F3u, 0x50E1042u, 0x6122081u, 0x69428A5u, 0x1C70434u, 0x3CF24B2u, 0x4D334F1u,
    0x5553C02u, 0x6420046u, 0x20E3101u, 0x2903815u, 0x50314B1u, 0x5851CD2u, 0x6892C01u, 0x70B3422u, 0x0CF4463u, 0x25510C1u, 0x2C018E6u, 0x5CC4892u,
    0x6D00CD3u, 0x1162424u, 0x30744A5u, 0x5912031u, 0x6132853u, 0x04240B2u, 0x14604F3u, 0x2CC1C44u, 0x4D43CC2u, 0x5C10103u, 0x0071851u, 0x0892072u,
    0x18D30B4u, 0x3950423u, 0x5041C82u, 0x6082CC3u, 0x04E4411u, 0x0D00032u, 0x1D41073u, 0x34328D1u, 0x3C530F3u, 0x54B4843u, 0x6D114A2u, 0x09524E3u,
    0x2043C33u, 0x38A0891u, 0x40C10B3u, 0x5922803u, 0x7014063u, 0x1470CC5u, 0x3D13451u, 0x4533C75u, 0x6C61803u, 0x10C3061u, 0x18E3882u, 0x29248C3u,
    0x4011411u, 0x4831C33u, 0x6093492u, 0x70D44D1u, 0x04F00F5u, 0x2C22883u, 0x44840E1u, 0x4CA4903u, 0x6501452u, 0x0142491u, 0x0962CB3u, 0x2054402u,
    0x3090843u, 0x48F20A4u, 0x6804012u, 0x0440451u, 0x0C60C72u, 0x1CA1CB1u, 0x24C24D6u, 0x5410883u, 0x6C720E2u, 0x08B3013u, 0x2114871u, 0x2930495u,
    0x5062C21u, 0x5883446u, 0x1541902u, 0x2412833u, 0x3C74094u, 0x5CF1403u, 0x0152C62u, 0x1023CA1u, 0x18444C2u, 0x2880901u, 0x30A1015u, 0x59438B1u,
    0x61640D8u, 0x2CF34C1u, 0x3513CE2u, 0x4550013u, 0x5C41871u, 0x6462095u, 0x1904821u, 0x2120442u, 0x3161483u, 0x4852CE3u, 0x60B4431u, 0x68D0053u,
    0x0D318B2u, 0x1C028F1u, 0x2423005u, 0x4CC0CA7u, 0x1034473u, 0x28910D2u, 0x38D2001u, 0x40F2822u, 0x5133864u, 0x7040CE3u, 0x14A2432u, 0x24E3473u,
    0x3D400D1u, 0x45608F2u, 0x5431823u, 0x6C93081u, 0x00B38A3u, 0x1910503u, 0x3001C53u, 0x48634B2u, 0x58A44F3u, 0x7101041u, 0x0521863u, 0x1C130C2u,
    0x2C54103u, 0x44B0C51u, 0x4CD1475u, 0x0003C06u, 0x30C20C1u, 0x38E28E2u, 0x4923811u, 0x5144035u, 0x0471CD1u, 0x0C924F3u, 0x24F3C42u, 0x3530081u,
    0x3D508A2u, 0x4C218E3u, 0x6483033u, 0x08E4896u, 0x3832C41u, 0x4053463u, 0x58B00C2u, 0x68F1107u, 0x2C648D2u, 0x3CA0C01u, 0x44C1422u, 0x5502461u,
    0x5D22C86u, 0x1871033u, 0x30D2892u, 0x41138D3u, 0x5800423u, 0x7061C83u, 0x14C34E1u, 0x1CE3D03u, 0x3540853u, 0x4C320B2u, 0x5C730F1u, 0x6493802u,
    0x00D4843u, 0x19314A1u, 0x2151CC3u, 0x3843413u, 0x50A0072u, 0x60E10B6u, 0x1C34061u, 0x2454886u, 0x5512C32u, 0x6553C71u, 0x6C04492u, 0x08408D3u,
    0x20A2021u, 0x28C2843u, 0x41240A2u, 0x51604E1u, 0x5810D02u, 0x6851C33u, 0x0CB3494u, 0x2D30803u, 0x4422062u, 0x54630A1u, 0x5C838C3u, 0x00E0412u,
    0x1121453u, 0x2812CB4u, 0x4890022u, 0x58D1061u, 0x60F1882u, 0x71328C1u, 0x05530E2u, 0x1424017u, 0x4D02CF2u, 0x5D43C24u, 0x08510A5u, 0x30F3831u,
    0x3914053u, 0x5000CB5u, 0x04A3441u, 0x0CC3C62u, 0x1D000A3u, 0x3561901u, 0x3C12015u, 0x64B48B1u, 0x6CD04D2u, 0x0911401u, 0x1131C26u, 0x40800E5u,
    0x6922871u, 0x7143092u, 0x0C140D1u, 0x14348F2u, 0x2470C23u, 0x3CD2484u, 0x5D54502u, 0x6C20833u, 0x1082093u, 0x28E38F3u, 0x4140441u, 0x4960C63u,
    0x60524C2u, 0x7093501u, 0x04B3C13u, 0x1D10875u, 0x4443004u, 0x64C0482u, 0x01014C1u, 0x0921CE2u, 0x1962C13u, 0x3054474u, 0x50D18F3u, 0x6933045u,
    0x1C60CE1u, 0x2481502u, 0x34C2433u, 0x4D23C91u, 0x55444B3u, 0x6C31003u, 0x1092862u, 0x20D38A1u, 0x28F40C2u, 0x3930503u, 0x5021C51u, 0x5842475u,
    0x0CE0001u, 0x1500823u, 0x2D62085u, 0x5494811u, 0x5CB0435u, 0x1152CD1u, 0x18034F2u, 0x2844421u, 0x3060042u, 0x40A1087u, 0x0414852u, 0x1450C91u,
    0x1C714B2u, 0x2CB24F6u, 0x5C008A1u, 0x64210C3u, 0x0882812u, 0x18C3851u, 0x20E4073u, 0x3940CD2u, 0x4811C03u, 0x6073466u, 0x1D31811u, 0x2552033u,
    0x3C43892u, 0x4C848D4u, 0x6D01C43u, 0x11634A2u, 0x20344E3u, 0x3891031u, 0x40B1852u, 0x50F2893u, 0x69540F1u, 0x7004803u, 0x1461465u, 0x3D03D01u,
    0x4524412u, 0x5560853u, 0x6C520B1u, 0x00728D3u, 0x18D4022u, 0x2910461u, 0x3130C82u, 0x4001CC1u, 0x48224E5u, 0x70C0076u, 0x2C13021u, 0x3433842u,
    0x4474883u, 0x5CD14E3u, 0x0132C31u, 0x0953453u, 0x20400B3u, 0x38A1802u, 0x48E2846u, 0x0430D01u, 0x0C51413u, 0x24B2C72u, 0x34F3CB1u, 0x3D144D5u,
    0x6442063u, 0x08A38C4u, 0x2920C32u, 0x3961C71u, 0x4012493u, 0x5873CF2u, 0x68B0024u, 0x15320A3u, 0x2C23905u, 0x54C1491u, 0x5CE1CB2u, 0x6D22CF3u,
    0x1014447u, 0x48F3012u, 0x5934051u, 0x6154875u, 0x1482401u, 0x1CA2C23u, 0x3504482u, 0x45408C1u, 0x4D610E2u, 0x5C32011u, 0x6452836u, 0x2110CF5u,
    0x4843481u, 0x5063CA2u, 0x60A00E1u, 0x68C0902u, 0x0501834u, 0x24138B3u, 0x3C70402u, 0x4CB1441u, 0x54D1C62u, 0x6512CA3u, 0x0804503u, 0x2061051u,
    0x2881873u, 0x40E30D2u, 0x5124004u, 0x7031482u, 0x0C724C3u, 0x24D3C14u, 0x4551092u, 0x54220D3u, 0x6C83821u, 0x00A4042u, 0x10E0484u, 0x3162503u,
    0x4853C52u, 0x5890093u, 0x70F18F3u, 0x1553043u, 0x2C448A1u, 0x34604C3u, 0x4CC1C13u, 0x6523472u, 0x01644B1u, 0x08100D2u, 0x1851003u, 0x30B2864u,
    0x51348E2u, 0x6000C11u, 0x6821432u, 0x0462471u, 0x0C82C95u, 0x3520821u, 0x3D41045u, 0x64738E1u, 0x6C94103u, 0x10F0C55u, 0x38234F1u, 0x4043C03u,
    0x58A0862u, 0x68E18A1u, 0x71020C2u, 0x0D43103u, 0x2434853u, 0x3C914B4u, 0x5D13422u, 0x6D54461u, 0x0000083u, 0x18618E5u, 0x4104074u, 0x60114F3u,
    0x0472C42u, 0x14B3C81u, 0x1CD44A2u, 0x2D108E4u, 0x4C22855u, 0x00C04F3u, 0x1921C41u, 0x2142462u, 0x30134A4u, 0x5090813u, 0x68F2073u, 0x0D538D2u,
    0x1C24801u, 0x2440422u, 0x3481464u, 0x55034E3u, 0x6D60032u, 0x0831073u, 0x20928D1u, 0x28B30F5u, 0x5150C81u, 0x58014A5u, 0x0CA3C31u, 0x14C4452u,
    0x2500891u, 0x2D210B2u, 0x3D620F3u, 0x5453841u, 0x5C74063u, 0x00D0CC2u, 0x1111D01u, 0x1932415u, 0x40600B3u, 0x58C1801u, 0x60E2023u, 0x0543886u,
    0x3491C32u, 0x44D2C73u, 0x5D344D4u, 0x0841842u, 0x1882881u, 0x20A30A2u, 0x30E40E1u, 0x3904902u, 0x4940C34u, 0x6852CB3u, 0x0CB4402u, 0x1CF0844u,
    0x3C028C2u, 0x4C43903u, 0x64A0454u, 0x11224D3u, 0x2813C22u, 0x3850061u, 0x4070882u, 0x50B18C3u, 0x6913014u, 0x1420492u, 0x24614D1u, 0x2C81CF2u,
    0x3CC2C21u, 0x44E3445u, 0x6C110E1u, 0x0031905u, 0x28D4091u, 0x30F48B2u, 0x4130CF3u, 0x5822443u, 0x7083CA1u, 0x04A44C5u, 0x2D42051u, 0x3562872u,
    0x44338B3u, 0x5C90404u, 0x0912483u, 0x2003CE3u, 0x3860833u, 0x50C2092u, 0x61030D3u, 0x0564826u, 0x34B2CE3u, 0x4D14431u, 0x5530052u, 0x6401094u,
    0x1083003u, 0x28E4862u, 0x3920CA3u, 0x5012501u, 0x5832C12u, 0x6873C54u, 0x14F10D3u, 0x2D52823u, 0x4444082u, 0x54804C3u, 0x6CE1C13u, 0x1143471u,
    0x1963C93u, 0x30508F3u, 0x48B2042u, 0x58F3081u, 0x61138A5u, 0x1441431u, 0x1C61C55u, 0x45044F1u, 0x4D20003u, 0x6411862u, 0x00528A3u, 0x18B4101u,
    0x20D4813u, 0x3931472u, 0x48024B6u, 0x04C0863u, 0x1D220C1u, 0x25428E3u, 0x3C34032u, 0x4C70471u, 0x5490C93u, 0x6CF24F2u, 0x0933423u, 0x2020084u,
    0x40A2102u, 0x50E3031u, 0x5903852u, 0x6944891u, 0x71604B9u, 0x45100C2u, 0x5551103u, 0x6C42851u, 0x0063072u, 0x10A40B3u, 0x2900C01u, 0x3121426u,
    0x60744E2u, 0x70B0811u, 0x04D1036u, 0x34240F3u, 0x4C80C42u, 0x5CC1C81u, 0x64E24A2u, 0x01234E6u, 0x3071891u, 0x38920B5u, 0x6134843u, 0x04214A1u,
    0x0C41CC2u, 0x1C82D03u, 0x34E4451u, 0x3D00073u, 0x55618D3u, 0x6C53022u, 0x0894063u, 0x20F0CC3u, 0x3952411u, 0x4002C35u, 0x68A08D1u, 0x70C10F5u,
    0x2563883u, 0x3C504E4u, 0x5CD2453u, 0x0133CB2u, 0x10000F1u, 0x1820802u, 0x2861844u, 0x48E38C3u, 0x6140412u, 0x7011453u, 0x1472CB1u, 0x1C934D2u,
    0x2CD4403u, 0x4531061u, 0x4D51883u, 0x64430E3u, 0x08A4833u, 0x2101492u, 0x31424D3u, 0x4833C21u, 0x5054443u, 0x68B10A2u, 0x04F20E1u, 0x0D12903u,
    0x2404055u, 0x4CA1CF6u, 0x09600A3u, 0x2051901u, 0x2872015u, 0x51148B1u, 0x59304D3u, 0x7021C22u, 0x0C62C61u, 0x1483482u, 0x24C44C3u, 0x3D21013u,
    0x5412871u, 0x5C33095u, 0x10D0C21u, 0x18F1443u, 0x3152CA2u, 0x4023CE7u, 0x05028B2u, 0x15438F1u, 0x1D64002u, 0x2C30441u, 0x3450C62u, 0x4491CA4u,
    0x6513C13u, 0x0800872u, 0x18418B3u, 0x30A3001u, 0x38C3825u, 0x61614C1u, 0x6811CE3u, 0x0C73433u, 0x24D0092u, 0x35110D3u, 0x4C02823u, 0x6464081u,
    0x6C848A3u, 0x10E1502u, 0x2122431u, 0x2942C52u, 0x3813C96u, 0x68D2041u, 0x70F2866u, 0x2C40C12u, 0x3C81C51u, 0x44A2472u, 0x54E34B3u, 0x6D40001u,
    0x0160825u, 0x28930C1u, 0x30B38E2u, 0x40F4813u, 0x5951473u, 0x7042CD1u, 0x04634F3u, 0x1CC0042u, 0x2D01081u, 0x35218A3u, 0x4C13102u, 0x5C54037u,
    0x2132C02u, 0x3003C41u, 0x3824462u, 0x48608A1u, 0x50810C2u, 0x60C2107u, 0x2430CD2u, 0x3471C03u, 0x4CD3461u, 0x54F3C85u, 0x0821811u, 0x1042033u,
    0x28A3893u, 0x41004F3u, 0x5961C42u, 0x6832C83u, 0x0C944E1u, 0x14B0105u, 0x3D52893u, 0x54440F1u, 0x5C64806u, 0x1922CC5u, 0x4050851u, 0x4871072u,
    0x58B20B1u, 0x60D28D2u, 0x7113803u, 0x1400461u, 0x1C20C83u, 0x34824E2u, 0x44C3413u, 0x5D20073u, 0x00118D3u, 0x1873024u, 0x38F04A2u, 0x49314E1u,
    0x5151D03u, 0x6843452u, 0x0484493u, 0x1CE10F4u, 0x3D63062u, 0x4C340A1u, 0x54548C2u, 0x6490D07u, 0x28044D3u, 0x4061025u, 0x69038C3u, 0x0D60413u,
    0x2451C71u, 0x2C72493u, 0x44D3CF3u, 0x5D30842u, 0x6C01881u, 0x00220A2u, 0x10630E3u, 0x28C4831u, 0x30E0453u, 0x4941CB3u, 0x6033403u, 0x0490065u,
    0x2D32901u, 0x3553015u, 0x5C80CB1u, 0x64A14D2u, 0x00E2401u, 0x0902C22u, 0x1943C63u, 0x30308C4u, 0x50B2832u, 0x60F3871u, 0x6914092u, 0x05504D7u,
    0x3CC3CA3u, 0x5520902u, 0x6561831u, 0x6C12053u, 0x10738B2u, 0x20B48F3u, 0x3911446u, 0x6864501u, 0x7080012u, 0x0CC1051u, 0x14E1872u, 0x25228B4u,
    0x4434823u, 0x5C91482u, 0x6CD24C4u, 0x1954432u, 0x2820873u, 0x40820D1u, 0x48A28F3u, 0x6104045u, 0x1431CE1u, 0x1C52502u, 0x2C93433u, 0x44F0091u,
    0x4D108B3u, 0x6402002u, 0x0043041u, 0x0863862u, 0x18A48A1u, 0x20C04C5u, 0x4962C51u, 0x5013475u, 0x04B1001u, 0x0CD1822u, 0x1D12863u, 0x34040C3u,
    0x4C60C14u, 0x6CE2C93u, 0x11444F2u, 0x2010823u, 0x3872083u, 0x50D38E1u, 0x58F4103u, 0x7150C52u, 0x0C21C91u, 0x14424B3u, 0x2CA3C05u, 0x55418A4u,
    0x0053812u, 0x1094851u, 0x18B0475u, 0x4152C04u, 0x6060083u, 0x04C18E2u, 0x1502813u, 0x2D64071u, 0x3414892u, 0x4450CD3u, 0x5CB2424u, 0x09344A3u,
    0x2021102u, 0x3062031u, 0x3882855u, 0x61204F1u, 0x6940C05u, 0x1C734A1u, 0x2493CC2u, 0x34D0101u, 0x3CF0815u, 0x64230B1u, 0x6C438D5u, 0x20E1461u,
    0x2901C82u, 0x3942CC1u, 0x41634E2u, 0x5034414u, 0x70B1893u, 0x15130F2u, 0x2554021u, 0x2C04842u, 0x3C40C83u, 0x54A24E3u, 0x6D03C31u, 0x0124453u,
    0x18110B2u, 0x28520F4u, 0x48D4062u, 0x59104A3u, 0x7001D04u, 0x1C83C72u, 0x2CC00B1u, 0x34E08D3u, 0x4D42022u, 0x5C13064u, 0x08904E3u, 0x20F1C32u,
    0x3132C73u, 0x48244D3u, 0x6081023u, 0x04E2881u, 0x0D030A3u, 0x2564903u, 0x3C51452u, 0x4C92491u, 0x54B2CB2u, 0x64F3CF3u, 0x0950841u, 0x1001063u,
    0x28628C2u, 0x38A3901u, 0x40C4012u, 0x5100451u, 0x5920C75u, 0x0C53406u, 0x3D118C1u, 0x45320E3u, 0x5C23832u, 0x6C64873u, 0x10C14D1u, 0x18E1CF3u,
    0x3143442u, 0x4014481u, 0x48300A2u, 0x58710E6u, 0x1534094u, 0x3441402u, 0x4482441u, 0x4CA2C68u, 0x1832054u, 0x38B40D2u, 0x48F0401u, 0x5110C22u,
    0x6151C61u, 0x6802482u, 0x04434C4u, 0x24C0838u, 0x6454821u, 0x6C70442u, 0x08B1484u, 0x2933506u, 0x58818B2u, 0x68C28F1u, 0x70E3002u, 0x0D24043u,
    0x2410CA1u, 0x2C314C3u, 0x4492C12u, 0x54D3C53u, 0x6D308B1u, 0x01510D6u, 0x30A4085u, 0x5941C11u, 0x6162432u, 0x7033471u, 0x0453C92u, 0x14900D3u,
    0x2CF1821u, 0x3512043u, 0x4C038A2u, 0x5C448E1u, 0x6460502u, 0x00A1433u, 0x1902C93u, 0x31644F1u, 0x3810003u, 0x5071863u, 0x68D30C3u, 0x0D34812u,
    0x1C00C53u, 0x34624B4u, 0x54E4422u, 0x6520863u, 0x08120C1u, 0x10328E2u, 0x2073814u, 0x40F0C93u, 0x59524F2u, 0x6823424u, 0x14A08A2u, 0x24E18E3u,
    0x3D43031u, 0x4563853u, 0x5C504B3u, 0x00B1C02u, 0x10F2C41u, 0x1913462u, 0x29544A3u, 0x4041104u, 0x60C3072u, 0x71040B1u, 0x05248D2u, 0x1560C01u,
    0x1C11425u, 0x44B3CC1u, 0x4CD44E5u, 0x0002071u, 0x0822892u, 0x18638D1u, 0x20840F5u, 0x4921C81u, 0x51424A5u, 0x0470031u, 0x0C90852u, 0x1CD1893u,
    0x35330F4u, 0x5440463u, 0x6CA1CC2u, 0x08E2D01u, 0x1103413u, 0x2960072u, 0x38310B3u, 0x5092804u, 0x7114885u, 0x2442411u, 0x2C62C32u, 0x3CA3C74u,
    0x5D210F5u, 0x1053883u, 0x28B04E1u, 0x30D0D02u, 0x4111C33u, 0x5803491u, 0x6023CB3u, 0x0480803u, 0x1CE2062u, 0x2D230A3u, 0x4414904u, 0x6491C73u,
    0x08F34D3u, 0x2150022u, 0x3021061u, 0x3841885u, 0x60E4011u, 0x6904835u, 0x1C324D1u, 0x2452CF2u, 0x3493C21u, 0x3CB4442u, 0x4CF0883u, 0x65520E1u,
    0x6C02903u, 0x1064052u, 0x20A0491u, 0x28C0CB5u, 0x5163443u, 0x68500A1u, 0x70708C3u, 0x14D2012u, 0x2513054u, 0x44204D2u, 0x5461403u, 0x6CC2C64u,
    0x19400E2u, 0x2811011u, 0x3031832u, 0x4072871u, 0x4893096u, 0x0551443u, 0x1C42CA2u, 0x2C83CE3u, 0x44E0831u, 0x4D01052u, 0x5D42093u, 0x00338F7u,
    0x39124C2u, 0x4953501u, 0x5003C12u, 0x6040054u, 0x0CC20D3u, 0x2523822u, 0x3564861u, 0x3C10482u, 0x4C514C1u, 0x5471CE5u, 0x0914471u, 0x1130095u,
    0x3862823u, 0x50C4083u, 0x6920CE3u, 0x0C12431u, 0x1432C53u, 0x2C944B2u, 0x3CD08F1u, 0x44F1002u, 0x5532043u, 0x6C238A3u, 0x1080501u, 0x18A0C13u,
    0x3102473u, 0x4963CD5u, 0x7091863u, 0x14F30C7u, 0x4C61C92u, 0x5CA2CD1u, 0x64C34F2u, 0x0104424u, 0x20118A3u, 0x3873102u, 0x48B4033u, 0x6110C91u,
    0x69314B2u, 0x04024F4u, 0x2484463u, 0x3CE10C3u, 0x5542813u, 0x6C34072u, 0x08704B3u, 0x20D1C01u, 0x28F2423u, 0x4153C82u, 0x50200C1u, 0x58408E2u,
    0x6881811u, 0x70A2035u, 0x25448D6u, 0x5492C81u, 0x5CB34A3u, 0x0110105u, 0x2842891u, 0x30630B3u, 0x48C4802u, 0x5900C43u, 0x71624A3u, 0x1453D03u,
    0x2CB0851u, 0x34D1075u, 0x5C03801u, 0x6424023u, 0x0880C82u, 0x18C1CC7u, 0x5030892u, 0x60718D1u, 0x68920F2u, 0x04D3021u, 0x0CF3842u, 0x1D34887u,
    0x54A3452u, 0x64E4493u, 0x09410F1u, 0x1161802u, 0x2032843u, 0x38940A1u, 0x40B48C3u, 0x5911413u, 0x7002C72u, 0x0C43CB1u, 0x14644D5u, 0x3D02061u,
    0x4522883u, 0x5C140E2u, 0x6C50411u, 0x0070C32u, 0x10B1C76u, 0x4000021u, 0x4820846u, 0x04E3902u, 0x1524831u, 0x1D40452u, 0x2C11493u, 0x4472CF1u,
    0x4C93403u, 0x64F0063u, 0x09518C2u, 0x1822903u, 0x3084053u, 0x48E0CB1u, 0x51014D5u, 0x0433C61u, 0x0C54483u, 0x24B10E2u, 0x34F2013u, 0x4D53874u,
    0x6C60CF2u, 0x08A1C21u, 0x10C2442u, 0x2103481u, 0x2923CA2u, 0x39600E7u, 0x70D38B2u, 0x0D148F3u, 0x2401441u, 0x2C21C65u, 0x54C4501u, 0x5CE0013u,
    0x0141873u, 0x18330D2u, 0x2874001u, 0x3094822u, 0x40D0C63u, 0x59324C1u, 0x6152CE5u, 0x1480871u, 0x1CA1093u, 0x35028F6u, 0x6450CA5u, 0x18F3433u,
    0x3150091u, 0x38008B2u, 0x48418F3u, 0x60A3041u, 0x68C3863u, 0x0D204C2u, 0x1D61503u, 0x3452C53u, 0x4CB44B3u, 0x6511001u, 0x6D31823u, 0x1023082u,
    0x20640C1u, 0x28848E3u, 0x40E1432u, 0x5122473u, 0x6813CD4u, 0x1491042u, 0x24D2081u, 0x2CF28A2u, 0x3D338E3u, 0x5420434u, 0x00A24B3u, 0x1903C05u,
    0x40318A1u, 0x48520C5u, 0x70F4851u, 0x0510473u, 0x1C01CD3u, 0x3463422u, 0x44A4463u, 0x5D010C3u, 0x0162811u, 0x0813033u, 0x2074892u, 0x30B0CD1u,
    0x38D14F3u, 0x5132C45u, 0x04608E1u, 0x0C81106u, 0x3D440B2u, 0x4C104F1u, 0x5430C02u, 0x6471C43u, 0x08D34A6u, 0x3821851u, 0x4042072u, 0x50830B6u,
    0x0D41461u, 0x1561C83u, 0x2C534E2u, 0x3C94411u, 0x44B0033u, 0x5D11892u, 0x6D528D9u, 0x41024E1u, 0x4922D02u, 0x5963C31u, 0x6014452u, 0x7050894u,
    0x1CD2803u, 0x3534062u, 0x44004A3u, 0x5C61D01u, 0x6482412u, 0x00C3453u, 0x19200B1u, 0x21408D3u, 0x3832026u, 0x68F04E2u, 0x0531413u, 0x1C22C71u,
    0x2443493u, 0x3CA00F2u, 0x4CE1023u, 0x6542881u, 0x6D630A5u, 0x2090C31u, 0x28B1455u, 0x5153CF1u, 0x5804402u, 0x6840843u, 0x0CA20A3u, 0x2503901u,
    0x2D24013u, 0x4410C73u, 0x5C724D2u, 0x6CB3403u, 0x1110063u, 0x28018C4u, 0x4883832u, 0x58C4871u, 0x60E0493u, 0x0541CF2u, 0x1412C23u, 0x2C74484u,
    0x4CF1902u, 0x5D32831u, 0x6553053u, 0x08448B6u, 0x3902C63u, 0x51644C2u, 0x6030903u, 0x0492053u, 0x1CF38B3u, 0x3550404u, 0x5462483u, 0x6CC3CE2u,
    0x0900011u, 0x1120835u, 0x38530D1u, 0x40738F3u, 0x58D0443u, 0x7131CA2u, 0x0C02CE1u, 0x1423505u, 0x3CC1091u, 0x44E18B5u, 0x6C14041u, 0x0034862u,
    0x1070CA1u, 0x18914C2u, 0x28D2504u, 0x4954473u, 0x60410D2u, 0x7082001u, 0x04A2822u, 0x14E3863u, 0x2D404C4u, 0x4C52433u, 0x64B3C92u, 0x00F00D4u,
    0x2002042u, 0x3043083u, 0x48A48E4u, 0x6921C52u, 0x0562C91u, 0x0C134B2u, 0x1C544F1u, 0x2470002u, 0x34B1044u, 0x55330C3u, 0x6C24812u, 0x0860C56u,
    0x3923C03u, 0x5010861u, 0x5831083u, 0x70928E5u, 0x2530471u, 0x2D50C92u, 0x3C21CD3u, 0x5483421u, 0x5CA3C43u, 0x01008A2u, 0x11418E1u, 0x1962102u,
    0x2833031u, 0x3053855u, 0x58F14F1u, 0x6111C05u, 0x14444A1u, 0x1C600C3u, 0x34C1812u, 0x4502853u, 0x5D640B4u, 0x0871422u, 0x18B2461u, 0x20D2C82u,
    0x3113CC3u, 0x4800813u, 0x6062074u, 0x0CE40F2u, 0x1D20421u, 0x2540C43u, 0x3C324A5u, 0x64D0034u, 0x11520B2u, 0x20230F1u, 0x2843802u, 0x3884843u,
    0x50E14A4u, 0x7163415u, 0x24910B3u, 0x3CF2801u, 0x4513022u, 0x5554064u, 0x00614E3u, 0x18C2C33u, 0x3124492u, 0x41608D1u, 0x48110F2u, 0x5852023u,
    0x70B3881u, 0x04D40A5u, 0x2C01C33u, 0x4463491u, 0x4C83CB5u, 0x0121841u, 0x0942065u, 0x3074901u, 0x3890412u, 0x48D1451u, 0x50F1C72u, 0x6132CB4u,
    0x0C40023u, 0x24A1882u, 0x34E28C1u, 0x3D030E2u, 0x4D44013u, 0x6430C73u, 0x08924D1u, 0x10B2CF3u, 0x2914443u, 0x40010A3u, 0x5862902u, 0x68A3833u,
    0x0D00494u, 0x2C12402u, 0x3C53441u, 0x4473C63u, 0x5CD08C2u, 0x6D11904u, 0x1823873u, 0x30804D2u, 0x40C1404u, 0x6143482u, 0x70144C3u, 0x1471011u,
    0x1C91833u, 0x34F3093u, 0x4D548F2u, 0x5C20C21u, 0x6441442u, 0x0082483u, 0x18E3CE4u, 0x3961052u, 0x4832091u, 0x50528B2u, 0x60938F1u, 0x68B4005u,
    0x1D51CA6u, 0x4CA0051u, 0x54C0872u, 0x65018B1u, 0x6D220D2u, 0x0963003u, 0x2054861u, 0x2870485u, 0x5112C11u, 0x5933432u, 0x6804477u, 0x2CE3043u,
    0x45448A2u, 0x5410CE1u, 0x5C31505u, 0x10D3C93u, 0x29308F4u, 0x4842862u, 0x58838A3u, 0x70E0501u, 0x0500C12u, 0x1541C54u, 0x3453CD3u, 0x4CB0825u,
    0x01530C1u, 0x08038E2u, 0x1844813u, 0x30A1471u, 0x38C1C96u, 0x6810042u, 0x0451083u, 0x1CB28E3u, 0x3514031u, 0x3D34853u, 0x54214B3u, 0x6C82C02u,
    0x08C3C41u, 0x10E4466u, 0x4032815u, 0x68D04B1u, 0x70F0CD2u, 0x0D31C01u, 0x1552422u, 0x2423463u, 0x3C800C1u, 0x44A08E3u, 0x5D02032u, 0x6D43071u,
    0x0163895u, 0x2891423u, 0x40F2C81u, 0x49134A3u, 0x6000102u, 0x7041031u, 0x0461853u, 0x1CC30B2u, 0x2D040F3u, 0x4560C44u, 0x6472CC2u, 0x00B3D03u,
    0x1910851u, 0x2131076u, 0x5084023u, 0x68E0C82u, 0x0521CC3u, 0x1C13411u, 0x2433C32u, 0x3470073u, 0x4CD18D1u, 0x54F20F6u, 0x10404A2u, 0x20814E1u,
    0x28A1D02u, 0x38E2C37u, 0x7051802u, 0x0C92841u, 0x14B3062u, 0x24F40A1u, 0x2D148C5u, 0x5442451u, 0x5C62C75u, 0x1100803u, 0x2962061u, 0x3012885u,
    0x58B0411u, 0x60D0C33u, 0x0532492u, 0x14034D1u, 0x1C23CF2u, 0x2C60023u, 0x44C1883u, 0x5D230E1u, 0x6543903u, 0x0830452u, 0x1871491u, 0x2091CB5u,
    0x4934443u, 0x60210A4u, 0x0CA3013u, 0x2504872u, 0x3540CB1u, 0x3D614D2u, 0x4C32404u, 0x6CB4485u, 0x2152013u, 0x3843871u, 0x4064092u, 0x50A04D3u,
    0x6901C21u, 0x7122443u, 0x1413CA3u, 0x2C70903u, 0x44D2052u, 0x5513094u, 0x0020403u, 0x1881C62u, 0x28C2CA1u, 0x30E34C2u, 0x4124501u, 0x4940015u,
    0x70728B6u, 0x2D30C61u, 0x3551482u, 0x44224C1u, 0x4C42CE5u, 0x00E0871u, 0x0901093u, 0x21628F2u, 0x3033821u, 0x3854045u, 0x60F1CE3u, 0x0553431u,
    0x0C03C55u, 0x34A18F4u, 0x5523862u, 0x65648A7u, 0x28D3472u, 0x39144B1u, 0x41300D2u, 0x5001001u, 0x5821822u, 0x6862864u, 0x14E48E3u, 0x2D41432u,
    0x3C12473u, 0x5473CD1u, 0x5C944F2u, 0x6CD0823u, 0x1132084u, 0x3044103u, 0x48A0C52u, 0x58E1C91u, 0x61024B2u, 0x71434F3u, 0x1430041u, 0x1C50863u,
    0x34B20C2u, 0x44F3101u, 0x4D13812u, 0x5D54856u, 0x18A2C01u, 0x20C3426u, 0x50118E2u, 0x6052813u, 0x04B4073u, 0x1D10CD1u, 0x25314F3u, 0x3C22C42u,
    0x4C63C81u, 0x54844A2u, 0x64C08E3u, 0x0922033u, 0x2013891u, 0x28340B3u, 0x4090C03u, 0x58F2463u, 0x7153CC2u, 0x0C20103u, 0x2481856u, 0x5544801u,
    0x5D60422u, 0x6C31461u, 0x0051C82u, 0x1092CC7u, 0x4801892u, 0x58428D3u, 0x70A4021u, 0x04C4846u, 0x3412D03u, 0x4C74453u, 0x64D10B2u, 0x01120F1u,
    0x0932802u, 0x1803843u, 0x30604A1u, 0x3880CC5u, 0x6123451u, 0x6943C72u, 0x04100B1u, 0x0C308D6u, 0x3CF3885u, 0x6421411u, 0x6C41C33u, 0x10A3492u,
    0x20E44D3u, 0x3941021u, 0x4161843u, 0x58530A2u, 0x68940E3u, 0x0CF0C33u, 0x2552493u, 0x3C43CF1u, 0x4464403u, 0x5CC1062u, 0x6D020A1u, 0x01228C3u,
    0x1814012u, 0x2850453u, 0x40B1CB4u, 0x6133C22u, 0x7000061u, 0x0420882u, 0x14618C3u, 0x2CC3017u, 0x6431CF5u, 0x18D4481u, 0x20F00A2u, 0x31310E3u,
    0x4822831u, 0x5043053u, 0x68A48B3u, 0x0D01402u, 0x1D42441u, 0x2562C65u, 0x4C90901u, 0x54B1013u, 0x6D12872u, 0x09538B1u, 0x10040D8u, 0x51034C1u,
    0x5923CE5u, 0x0C51871u, 0x1472092u, 0x24B30D1u, 0x2CD38F2u, 0x3D14823u, 0x5401484u, 0x0083503u, 0x18E0052u, 0x2921096u, 0x5874041u, 0x6094865u,
    0x1532501u, 0x1D52C13u, 0x3444472u, 0x44808B3u, 0x5CE2006u, 0x18304C1u, 0x2050CE2u, 0x3091C11u, 0x38B2432u, 0x48F3474u, 0x68008F3u, 0x0C62042u,
    0x1CA3083u, 0x35048E1u, 0x3D20502u, 0x4D61433u, 0x6452C91u, 0x6C734B3u, 0x10D0005u, 0x38028A1u, 0x40230C2u, 0x5064103u, 0x68C0C51u, 0x70E1473u,
    0x1542CD2u, 0x2413C01u, 0x2C34423u, 0x4491085u, 0x6D33811u, 0x0154035u, 0x2881CD3u, 0x40E3423u, 0x5940083u, 0x70318E1u, 0x0452103u, 0x1CB3853u,
    0x35104B2u, 0x45514F3u, 0x5C42C43u, 0x00A44A1u, 0x08C00C3u, 0x2121812u, 0x3162851u, 0x3813073u, 0x50748D2u, 0x60B0C03u, 0x0512464u, 0x24244E2u,
    0x3460811u, 0x3C81033u, 0x54E2892u, 0x65238D4u, 0x1030C43u, 0x28924A2u, 0x38D34E3u, 0x5130031u, 0x5950855u, 0x0C830F4u, 0x2D00463u, 0x4561CC2u,
    0x5432D06u, 0x10F10B1u, 0x19118D3u, 0x3003022u, 0x4044061u, 0x4864882u, 0x58A0CC1u, 0x60C14E5u, 0x1563C71u, 0x1C14496u, 0x4CD2842u, 0x5D13881u,
    0x65340A2u, 0x00004E4u, 0x2082455u, 0x49200F1u, 0x5140802u, 0x6011843u, 0x04730A3u, 0x1CD4901u, 0x24F0413u, 0x3D51C72u, 0x4C22CB4u, 0x6CA0022u,
    0x08E1067u, 0x4054832u, 0x5090C71u, 0x58B1492u, 0x68F24D1u, 0x7112CF2u, 0x0D53C24u, 0x2C610A3u, 0x44C2902u, 0x5503833u, 0x6D60493u, 0x1051CF3u,
    0x28B3441u, 0x30D3C63u, 0x49308C3u, 0x6022013u, 0x0483872u, 0x14C48B3u, 0x2D21401u, 0x3541C23u, 0x4C33482u, 0x5C744C3u, 0x00D1011u, 0x08F1835u,
    0x30240D1u, 0x38448F5u, 0x60E2481u, 0x6902CA3u, 0x0D64502u, 0x1C30833u, 0x3492091u, 0x3CB28B3u, 0x5514002u, 0x6550441u, 0x6C00C62u, 0x0841CA3u,
    0x20A3503u, 0x3900054u, 0x58120D2u, 0x6853001u, 0x7073823u, 0x14D0485u, 0x3C02C14u, 0x5C80092u, 0x6CC10D1u, 0x00E18F2u, 0x1122821u, 0x1943046u,
    0x4891505u, 0x7133C93u, 0x14208F3u, 0x2C82044u, 0x4D040C3u, 0x6560C13u, 0x0852472u, 0x18934B1u, 0x20B3CD2u, 0x30F0003u, 0x4951861u, 0x5002083u,
    0x68638E5u, 0x1D01471u, 0x2521C95u, 0x4C54421u, 0x5470045u, 0x09128E1u, 0x1133102u, 0x2004031u, 0x2824852u, 0x3860C93u, 0x50C24F1u, 0x58E2C03u,
    0x7144462u, 0x0C108A1u, 0x14310C2u, 0x2472103u, 0x3CD3854u, 0x5D50CD3u, 0x0042423u, 0x18A3C83u, 0x31008E2u, 0x4141813u, 0x5833074u, 0x04B04F2u,
    0x14F1421u, 0x1D11C42u, 0x2D52C81u, 0x34034A2u, 0x44444E4u, 0x64C1853u, 0x09230B2u, 0x19640F4u, 0x3871462u, 0x48B24A3u, 0x6113D01u, 0x6934413u,
    0x0C21075u, 0x34C3801u, 0x3CE4022u, 0x4D20463u, 0x6411CC4u, 0x1093C32u, 0x20D0071u, 0x28F0892u, 0x39318D1u, 0x41520F5u, 0x6884881u, 0x70A04A5u,
    0x2542C31u, 0x2D63452u, 0x3C34491u, 0x44500B2u, 0x54910F3u, 0x6CF2846u, 0x2840D01u, 0x3061412u, 0x40A2453u, 0x5903CB4u, 0x0411023u, 0x1C72882u,
    0x2CB38C1u, 0x34D40E3u, 0x4D30C35u, 0x00634D4u, 0x20E0842u, 0x3121883u, 0x48130E3u, 0x6074834u, 0x0CF1CB3u, 0x2553402u, 0x3424443u, 0x4C810A1u,
    0x54A18C2u, 0x64E2903u, 0x0944051u, 0x1164873u, 0x28514D3u, 0x40B2C22u, 0x50F3C63u, 0x69508C3u, 0x0C42011u, 0x1462836u, 0x4520CF2u, 0x5561C21u,
    0x5C12445u, 0x10B00E1u, 0x18D0905u, 0x4003091u, 0x48238B2u, 0x58648F1u, 0x6080402u, 0x70C1444u, 0x1D434C3u, 0x3430012u, 0x4471051u, 0x4C91875u,
    0x0134003u, 0x1820C61u, 0x2041483u, 0x38A2CE2u, 0x48E3C11u, 0x5104433u, 0x6961092u, 0x04320D3u, 0x1C93824u, 0x3D10CA2u, 0x4D51CE1u, 0x5402503u,
    0x6C63C56u, 0x2922003u, 0x4013862u, 0x50548A3u, 0x68B1501u, 0x70D1C12u, 0x0D12C53u, 0x24044B1u, 0x2C200D6u, 0x5CE3082u, 0x6D240C1u, 0x01448E2u,
    0x1010C14u, 0x3092C93u, 0x48F44F2u, 0x5930821u, 0x6151042u, 0x7022081u, 0x04428A5u, 0x2CE0436u, 0x5C334F3u, 0x0090041u, 0x08B0862u, 0x18F18A3u,
    0x3153101u, 0x3803813u, 0x5060472u, 0x60A14B1u, 0x68C1CD2u, 0x0502C06u, 0x34510C1u, 0x3C718E3u, 0x54D3032u, 0x6514071u, 0x6D34895u, 0x2062423u,
    0x38C3C84u, 0x5941103u, 0x7032852u, 0x0C73891u, 0x14940B2u, 0x24D04F4u, 0x4552463u, 0x5C43CC5u, 0x10E1851u, 0x1902072u, 0x29430B3u, 0x4034801u,
    0x4850426u, 0x05134E3u, 0x1C00032u, 0x2C41073u, 0x44A28D1u, 0x4CC30F3u, 0x6524842u, 0x0160C81u, 0x08114A2u, 0x18524E1u, 0x2072D0Bu, 0x0463841u,
    0x0C84062u, 0x1CC04A1u, 0x24E0CC5u, 0x4C13451u, 0x5433C73u, 0x6C908D2u, 0x08D1801u, 0x10F2022u, 0x2133063u, 0x38248C3u, 0x5081411u, 0x58A1C35u,
    0x0D444D1u, 0x15600F3u, 0x2C51842u, 0x3C92887u, 0x0001452u, 0x1042493u, 0x28A3CF1u, 0x30C4402u, 0x4100844u, 0x60128C3u, 0x0474012u, 0x14B0453u,
    0x2D11CB1u, 0x35324D2u, 0x4403403u, 0x5C60061u, 0x6480883u, 0x08E20E3u, 0x2143832u, 0x3014871u, 0x3830492u, 0x48714D3u, 0x60D2C24u, 0x0D500A2u,
    0x1C210E1u, 0x2441902u, 0x3482836u, 0x6540CF1u, 0x6D61406u, 0x28B44C2u, 0x38F0901u, 0x4111015u, 0x68438B1u, 0x70640D3u, 0x14C0C22u, 0x2501C61u,
    0x2D22482u, 0x3D634C3u, 0x5450013u, 0x6CB1871u, 0x00D2093u, 0x19338F2u, 0x2804821u, 0x3020443u, 0x4881CA2u, 0x58C2CE3u, 0x7124434u, 0x1C318B3u,
    0x3493002u, 0x44D4041u, 0x4CF4862u, 0x5D30CA9u, 0x30E08B3u, 0x4942001u, 0x5162825u, 0x04904C1u, 0x0CB0CE3u, 0x2512433u, 0x3C03C92u, 0x4C400D1u,
    0x54608F2u, 0x64A1824u, 0x11238A5u, 0x3851431u, 0x4071C52u, 0x50B2C91u, 0x58D34B6u, 0x1421865u, 0x3CC4101u, 0x44E4812u, 0x5520C51u, 0x5D41472u,
    0x6C124B3u, 0x1073C01u, 0x1894423u, 0x30F1082u, 0x41320C6u, 0x7080473u, 0x14E1CD1u, 0x1D024F3u, 0x3563C42u, 0x4430084u, 0x64B2102u, 0x00F3033u,
    0x1954894u, 0x3861C02u, 0x48A2C41u, 0x50C3462u, 0x61044A3u, 0x0561104u, 0x2473073u, 0x3CD48D5u, 0x6402461u, 0x6C22C82u, 0x0863CC3u, 0x20C0814u,
    0x4142893u, 0x58340F2u, 0x6870421u, 0x7090C42u, 0x0CD1C83u, 0x25334E1u, 0x2D53D03u, 0x4440852u, 0x5481891u, 0x5CA20B3u, 0x0103805u, 0x28314A1u,
    0x3051CC5u, 0x58F4451u, 0x6110072u, 0x71510B3u, 0x1442803u, 0x2CA4064u, 0x4D214E2u, 0x5D62411u, 0x6412C32u, 0x0053C76u, 0x3112021u, 0x3932843u,
    0x50240A3u, 0x6880D03u, 0x0CE2452u, 0x1D23493u, 0x34100F6u, 0x64D30A1u, 0x6CF38C2u, 0x0934901u, 0x1150412u, 0x2021454u, 0x40A34D3u, 0x5900022u,
    0x6941063u, 0x0C328C1u, 0x14530E2u, 0x2494014u, 0x4511493u, 0x5C02CF5u, 0x10A0881u, 0x18C10A2u, 0x29020E3u, 0x4163831u, 0x4814053u, 0x6070CB2u,
    0x70B1CF1u, 0x04D2402u, 0x1513441u, 0x1D33C65u, 0x4461901u, 0x4C82015u, 0x01248B1u, 0x09404D5u, 0x3072C63u, 0x48D44C1u, 0x50F00E3u, 0x6951835u,
    0x1C840D3u, 0x34E0C23u, 0x4D42481u, 0x5562CA3u, 0x6C54502u, 0x0890831u, 0x10B1053u, 0x29128B2u, 0x39538F3u, 0x5040444u, 0x70C24C2u, 0x0D03501u,
    0x1523C13u, 0x2C10872u, 0x3C518B7u, 0x0130482u, 0x10014C3u, 0x2862C11u, 0x3083432u, 0x40C4473u, 0x59210D4u, 0x0433043u, 0x1C948A2u, 0x2CD0CE1u,
    0x34F1505u, 0x5C23C91u, 0x64444B3u, 0x08A1002u, 0x18E2041u, 0x2102862u, 0x31438A6u, 0x6091C51u, 0x68B2475u, 0x1D50001u, 0x2400822u, 0x3441861u,
    0x3C62082u, 0x4CA30C4u, 0x6D20433u, 0x1011C93u, 0x28734F2u, 0x38B4423u, 0x5111083u, 0x68028E1u, 0x7023105u, 0x24C0C94u, 0x4542C02u, 0x5413C43u,
    0x6C708A4u, 0x18F2812u, 0x2933851u, 0x3154072u, 0x40204B1u, 0x4840CD2u, 0x5881C04u, 0x0503C83u, 0x1D608E2u, 0x2C31813u, 0x4493073u, 0x5CF48D3u,
    0x0151421u, 0x0801C43u, 0x20634A3u, 0x38C0102u, 0x4901031u, 0x5121852u, 0x6162893u, 0x04540F1u, 0x0C74803u, 0x24D1462u, 0x35124A1u, 0x3D32CC3u,
    0x5424415u, 0x08C20B1u, 0x10E28D5u, 0x3810464u, 0x58924E2u, 0x68D3413u, 0x0D30071u, 0x1550893u, 0x2C420F2u, 0x3C83021u, 0x44A3842u, 0x54E4883u,
    0x6D414E3u, 0x1032C34u, 0x30B00B2u, 0x40F10F1u, 0x4911803u, 0x6003065u, 0x14A0D04u, 0x3522C72u, 0x4563CB1u, 0x4C144D2u, 0x5C50801u, 0x6471022u,
    0x00B2064u, 0x21340E5u, 0x4861C73u, 0x60C34D1u, 0x68E3CF6u, 0x24320A3u, 0x3C93903u, 0x54F0452u, 0x6531493u, 0x0822CF3u, 0x2084441u, 0x28A0063u,
    0x41018C2u, 0x5142903u, 0x6834051u, 0x7054875u, 0x24F2401u, 0x2D12C26u, 0x5C610E2u, 0x6CA2011u, 0x00C2832u, 0x1103873u, 0x29604D1u, 0x3010CF5u,
    0x58B3481u, 0x60D3CA2u, 0x71100E3u, 0x1401833u, 0x2C63091u, 0x34838B3u, 0x4CE0403u, 0x6541C63u, 0x08334C2u, 0x1874507u, 0x51530D2u, 0x6024001u,
    0x6844822u, 0x0480C61u, 0x0CA1482u, 0x1CE24C4u, 0x3D64433u, 0x5451092u, 0x64920D4u, 0x1114042u, 0x2150483u, 0x3841CE1u, 0x4062503u, 0x58C3C53u,
    0x71208B3u, 0x1412002u, 0x2453043u, 0x3CB48A4u, 0x5D31C12u, 0x6C02C53u, 0x10644B1u, 0x18800D5u, 0x4122861u, 0x4943085u, 0x7070C11u, 0x0491432u,
    0x14D2471u, 0x1CF2C92u, 0x2D33CD3u, 0x4420821u, 0x4C41045u, 0x00E38E1u, 0x0904102u, 0x1940433u, 0x3031C97u, 0x6910862u, 0x05518A1u, 0x0C020C3u,
    0x2463812u, 0x34A4853u, 0x4D014B4u, 0x6C13422u, 0x0854463u, 0x20B10C1u, 0x28D18E6u, 0x5824893u, 0x70814F2u, 0x0CC2423u, 0x2523C83u, 0x3C108E3u,
    0x5472031u, 0x5C92853u, 0x00F40B3u, 0x1950C02u, 0x2821C43u, 0x40834A3u, 0x58E0101u, 0x6100813u, 0x0562073u, 0x1C538D2u, 0x2C94801u, 0x34B0425u,
    0x5D52CC1u, 0x64034E5u, 0x18A1071u, 0x20C1892u, 0x31028D1u, 0x39230F2u, 0x4964023u, 0x6050C81u, 0x68714A3u, 0x0CD2D02u, 0x1D13C31u, 0x2534455u,
    0x4C620F4u, 0x6CE4063u, 0x1140CC2u, 0x2011D01u, 0x2832413u, 0x4093C72u, 0x50D00B3u, 0x6931804u, 0x1443882u, 0x24848C1u, 0x2CA04E2u, 0x3CE1411u,
    0x4501C36u, 0x00500F3u, 0x18B1842u, 0x28F2884u, 0x4804902u, 0x5840C33u, 0x70A2491u, 0x04C2CB8u, 0x44520A1u, 0x4C728C2u, 0x5CB3904u, 0x0930C73u,
    0x20224D2u, 0x3063401u, 0x3883C22u, 0x48C0061u, 0x50E0885u, 0x0413011u, 0x0C33835u, 0x34D14D3u, 0x4D32C21u, 0x5553442u, 0x6424483u, 0x08810E4u,
    0x2903052u, 0x3944091u, 0x41648B2u, 0x5030CF3u, 0x6892443u, 0x0CF3CA1u, 0x15144C3u, 0x2C01012u, 0x3C42051u, 0x4462878u, 0x1161C64u, 0x3073CE3u,
    0x48D0832u, 0x5911873u, 0x70030D4u, 0x1C80443u, 0x34E1CA2u, 0x4522CE3u, 0x5C14431u, 0x6430052u, 0x0071093u, 0x18D28F1u, 0x20F3003u, 0x3954863u,
    0x50414C3u, 0x68A2C12u, 0x04E3C53u, 0x1D408B1u, 0x25610D5u, 0x4C93861u, 0x54B4082u, 0x64F04C1u, 0x6D10CE5u, 0x2043476u, 0x5101821u, 0x5922042u,
    0x6963081u, 0x70138A6u, 0x2CD1C53u, 0x45334B2u, 0x54044F1u, 0x5C20002u, 0x6C61043u, 0x10C28A3u, 0x2924101u, 0x3144815u, 0x58724B1u, 0x6092CD3u,
    0x04F4422u, 0x1530867u, 0x4CA4032u, 0x5CE0471u, 0x6500C93u, 0x09624F2u, 0x1833424u, 0x38B08A3u, 0x5112102u, 0x6153033u, 0x0444891u, 0x0C604B2u,
    0x1CA14F3u, 0x3502C41u, 0x3D23463u, 0x54100C3u, 0x6C71812u, 0x08B2851u, 0x10D3072u, 0x21140B3u, 0x3800C01u, 0x4021423u, 0x5882C82u, 0x68C3CC1u,
    0x70E44E2u, 0x0D20816u, 0x3C738D7u, 0x01524A2u, 0x10234E1u, 0x1843D02u, 0x2880033u, 0x40E1891u, 0x49020B3u, 0x6163802u, 0x7034841u, 0x0450462u,
    0x14914A6u, 0x4554451u, 0x4C00073u, 0x64618D2u, 0x00A2801u, 0x08C3025u, 0x3160CC3u, 0x4852414u, 0x68D4492u, 0x05108D1u, 0x0D310F2u, 0x1C02021u,
    0x2422842u, 0x3463887u, 0x6D42455u, 0x20700F1u, 0x2890805u, 0x51330A1u, 0x59538C6u, 0x14A1C72u, 0x24E2CB1u, 0x2D034D2u, 0x3D44403u, 0x5431061u,
    0x5C51885u, 0x10F4011u, 0x1914832u, 0x2950C71u, 0x3001496u, 0x60C4445u, 0x15620E1u, 0x1C12902u, 0x2C53831u, 0x3474052u, 0x44B0493u, 0x5D11CF1u,
    0x6532403u, 0x0823C62u, 0x18600A3u, 0x30C1903u, 0x4923053u, 0x60148B1u, 0x68304D3u, 0x0C91C22u, 0x1CD2C61u, 0x24F3483u, 0x3D500E2u, 0x4C21013u,
    0x6482874u, 0x11048F2u, 0x2140C23u, 0x3832483u, 0x5093CE4u, 0x7111053u, 0x14028B5u, 0x3CA0441u, 0x44C0C62u, 0x5501CA3u, 0x6D63501u, 0x0013C13u,
    0x1870873u, 0x30D20D2u, 0x4113001u, 0x4933822u, 0x5804863u, 0x70614C4u, 0x1CE3432u, 0x2D24471u, 0x3540093u, 0x4C318F5u, 0x00D4081u, 0x08F48A5u,
    0x3022431u, 0x3842C52u, 0x4883C91u, 0x50A44B5u, 0x0542044u, 0x24540C2u, 0x3490501u, 0x3CB0C12u, 0x4CF1C56u, 0x0840001u, 0x1060823u, 0x28C2082u,
    0x39030C1u, 0x41238E3u, 0x5810432u, 0x6851473u, 0x0CB2CD7u, 0x44218A2u, 0x54628E1u, 0x5C83102u, 0x6CC4034u, 0x19414B5u, 0x4073C43u, 0x58D08A1u,
    0x60F10C2u, 0x7132103u, 0x1423851u, 0x1C44073u, 0x34A0CD5u, 0x5D43461u, 0x6563C82u, 0x00300C4u, 0x20B2033u, 0x3913892u, 0x49548D1u, 0x50004F2u,
    0x6041421u, 0x6861C45u, 0x1D044E1u, 0x2520105u, 0x4C52891u, 0x54730B2u, 0x64B40F3u, 0x0910C43u, 0x20024A1u, 0x2822CC3u, 0x4084413u, 0x58E1075u,
    0x0C13803u, 0x2470461u, 0x2C90C83u, 0x44F24E2u, 0x5533414u, 0x0040892u, 0x10818D3u, 0x28E3024u, 0x49604A2u, 0x58314E1u, 0x6051D03u, 0x04B3452u,
    0x14F4494u, 0x3401803u, 0x4C63062u, 0x5CA40A3u, 0x0100D01u, 0x0921412u, 0x1962453u, 0x3053CB4u, 0x50D1023u, 0x6932882u, 0x04038C1u, 0x0C240E5u,
    0x34C1C71u, 0x3CE2493u, 0x5543CF2u, 0x6410021u, 0x6C30842u, 0x0871881u, 0x10920A5u, 0x3934831u, 0x4150455u, 0x6882CF1u, 0x70A3402u, 0x0CE4443u,
    0x25410A4u, 0x4453013u, 0x5CB4872u, 0x6CF0CB1u, 0x01114D2u, 0x1152403u, 0x2843C63u, 0x40A08C1u, 0x48C10E3u, 0x6122836u, 0x1C70CF2u, 0x2CB1C23u,
    0x4513486u, 0x0061831u, 0x0882052u, 0x18C3091u, 0x20E38B2u, 0x31248F4u, 0x5031C63u, 0x68934C2u, 0x04D4503u, 0x1D31053u, 0x34228B4u, 0x54A4823u,
    0x6D01483u, 0x1162CE2u, 0x2033C11u, 0x2854432u, 0x3890873u, 0x50F20D1u, 0x59128F3u, 0x7004042u, 0x0C40481u, 0x1460CA2u, 0x24A1CE1u, 0x2CC2505u,
    0x5560091u, 0x5C108B5u, 0x10B3041u, 0x18D3863u, 0x31304C2u, 0x4001503u, 0x5862C51u, 0x6083473u, 0x04E00D2u, 0x1521003u, 0x2C12863u, 0x44740C3u,
    0x5CD0C14u, 0x0952C92u, 0x1823CD1u, 0x20444F3u, 0x38A1045u, 0x61438E4u, 0x0C50C52u, 0x1C91C91u, 0x24B24B2u, 0x34F34F1u, 0x3D13C02u, 0x4D50049u,
    0x2104853u, 0x39614B1u, 0x4011CD2u, 0x5052C04u, 0x70D0083u, 0x15318E3u, 0x2C23032u, 0x3C64071u, 0x4484895u, 0x6D22421u, 0x0142C43u, 0x18344A2u,
    0x28708E3u, 0x40D2036u, 0x70204F1u, 0x0440C05u, 0x2CE34A1u, 0x3503CC2u, 0x4540101u, 0x4D60812u, 0x5C31853u, 0x00930B1u, 0x08B38D3u, 0x2110423u,
    0x3801C82u, 0x4842CC3u, 0x60A4413u, 0x0501071u, 0x0D21896u, 0x3C74843u, 0x54D14A2u, 0x65124E3u, 0x0803C34u, 0x28810B2u, 0x38C20F1u, 0x40E2802u,
    0x5123841u, 0x5944062u, 0x68104A4u, 0x1492413u, 0x2CF3C72u, 0x3D300B4u, 0x5C42022u, 0x6C83063u, 0x10E48C1u, 0x19004E3u, 0x3161C33u, 0x4853492u,
    0x58944D1u, 0x60B00F2u, 0x70F1023u, 0x1552884u, 0x3464902u, 0x44A0C31u, 0x4CC1453u, 0x6522CB5u, 0x1850841u, 0x2071065u, 0x4913903u, 0x6000451u,
    0x6820C72u, 0x0461CB3u, 0x1CC3401u, 0x24E3C25u, 0x4C118C1u, 0x54320E2u, 0x6473013u, 0x08D4874u, 0x2951CF3u, 0x4043442u, 0x5084481u, 0x58A00A3u,
    0x7101902u, 0x0D42833u, 0x2434094u, 0x44B1402u, 0x54F2443u, 0x6D53CA1u, 0x00044C2u, 0x1040904u, 0x30C2873u, 0x49240D2u, 0x5960403u, 0x7051C61u,
    0x0472485u, 0x2D10011u, 0x3530833u, 0x4C22093u, 0x64838F2u, 0x00C4823u, 0x1921483u, 0x3012CE1u, 0x3833503u, 0x5090053u, 0x68F18B2u, 0x05328F1u,
    0x0D53005u, 0x3480CA1u, 0x3CA14C6u, 0x6D64472u, 0x08308B1u, 0x10510D2u, 0x2092003u, 0x38F3861u, 0x4114085u, 0x6841C11u, 0x7062435u, 0x25000D3u,
    0x3D61821u, 0x4412043u, 0x5C738A2u, 0x6CB48E1u, 0x00D0503u, 0x1931C52u, 0x2802C97u, 0x60E1862u, 0x71228A1u, 0x05430C2u, 0x1414101u, 0x1C34816u,
    0x4CF2CD3u, 0x6554422u, 0x0020863u, 0x18820C1u, 0x20A28E2u, 0x30E3813u, 0x4940471u, 0x5160C96u, 0x0CB3C43u, 0x25108A2u, 0x35518E4u, 0x5463853u,
    0x6CC04B2u, 0x09014F3u, 0x2162C41u, 0x2813465u, 0x50B1101u, 0x58D1815u, 0x0C040B3u, 0x2460C01u, 0x2C81422u, 0x3CC2463u, 0x5523CC1u, 0x5D444E3u,
    0x0031032u, 0x1072071u, 0x1892892u, 0x28D38D3u, 0x4130423u, 0x5821C84u, 0x04A3D02u, 0x14E0031u, 0x1D00855u, 0x44330F3u, 0x5C94844u, 0x0911CC3u,
    0x2003412u, 0x3044451u, 0x3860076u, 0x6923023u, 0x0C14882u, 0x1C50CC3u, 0x34B2413u, 0x4D13C73u, 0x64008D1u, 0x6C210F3u, 0x1082843u, 0x28E40A3u,
    0x4140D02u, 0x5011C33u, 0x6873491u, 0x7093CB3u, 0x14F0803u, 0x2D52062u, 0x3C230A1u, 0x44438C5u, 0x6CE1456u, 0x2834401u, 0x3050022u, 0x4091061u,
    0x48B1885u, 0x7154011u, 0x0404833u, 0x1C61492u, 0x2CA24D1u, 0x34C2CF2u, 0x4503C23u, 0x5D60884u, 0x0872905u, 0x3110491u, 0x3930CB3u, 0x5022402u,
    0x6063447u, 0x2542012u, 0x3413051u, 0x3C33872u, 0x4C748B1u, 0x54904D2u, 0x64D1404u, 0x1153483u, 0x28400E2u, 0x3881014u, 0x5903092u, 0x69440D3u,
    0x0C30C21u, 0x1451443u, 0x2CB2CA5u, 0x5550831u, 0x5C01052u, 0x6C42093u, 0x10A38F1u, 0x18C4003u, 0x3120C62u, 0x4161CA1u, 0x48124C2u, 0x5853506u,
    0x15118B1u, 0x1D320D6u, 0x4C80482u, 0x5CC14C1u, 0x64E1CE2u, 0x0122C13u, 0x1814474u, 0x38918F2u, 0x48D2821u, 0x50F3042u, 0x6134083u, 0x0420CE3u,
    0x1C82431u, 0x24A2C53u, 0x3D044B2u, 0x4D408F1u, 0x5561003u, 0x6C52865u, 0x20F0504u, 0x4002472u, 0x50434B1u, 0x5863CD2u, 0x68A0003u, 0x0D01867u,
    0x4470432u, 0x54B1473u, 0x6D12CD1u, 0x01334F5u, 0x2861081u, 0x30818A3u, 0x48E3103u, 0x6144852u, 0x7010C91u, 0x04314B2u, 0x14724F3u, 0x2CD3C41u,
    0x34F4465u, 0x5C22101u, 0x6442812u, 0x0083851u, 0x08A4076u, 0x3962425u, 0x60900C1u, 0x68B08E2u, 0x04F1811u, 0x0D12032u, 0x1D53074u, 0x3C604F3u,
    0x54C1C42u, 0x6502C83u, 0x09644E3u, 0x2051033u, 0x38B2891u, 0x40D30B3u, 0x5934802u, 0x6800C41u, 0x7021463u, 0x1482CC2u, 0x24C3D03u, 0x3D20854u,
    0x5C328D2u, 0x6C73801u, 0x0094025u, 0x2931CC4u, 0x4843C33u, 0x60A0895u, 0x1543021u, 0x1D63842u, 0x2C34883u, 0x44914E1u, 0x4CB1D03u, 0x6513453u,
    0x08000B2u, 0x18410F1u, 0x2061802u, 0x30A2843u, 0x49040A1u, 0x51248C3u, 0x6811412u, 0x0452451u, 0x0C72C73u, 0x24D44D5u, 0x4C02066u, 0x08C0411u,
    0x10E0C32u, 0x2121C71u, 0x2942492u, 0x38134D3u, 0x5070024u, 0x70F20A2u, 0x0D330E1u, 0x1553902u, 0x2424836u, 0x54E2CF1u, 0x5D03403u, 0x0160062u,
    0x10310A1u, 0x18518C5u, 0x40F4053u, 0x5950CB6u, 0x14A3C61u, 0x1CC4482u, 0x2D008C1u, 0x35210E2u, 0x4562014u, 0x6474093u, 0x08D0CF5u, 0x3003481u,
    0x3823CA2u, 0x48600E3u, 0x60C1831u, 0x68E2058u, 0x3471441u, 0x3C91C62u, 0x4CD2CA3u, 0x6534501u, 0x6D50013u, 0x1041872u, 0x20828B1u, 0x28A30D2u,
    0x38E4001u, 0x4104826u, 0x7052CE5u, 0x24F0871u, 0x2D11092u, 0x3D520D3u, 0x5443823u, 0x6CA0481u, 0x00C0CA3u, 0x1922503u, 0x3013C52u, 0x4050093u,
    0x58B18F3u, 0x7113041u, 0x0533863u, 0x1C204C2u, 0x2C61501u, 0x3481C13u, 0x4CE3472u, 0x5D244B3u, 0x0011004u, 0x2093082u, 0x30D40C4u, 0x5151432u,
    0x6022474u, 0x0CA44F3u, 0x2501042u, 0x3542083u, 0x4C338E1u, 0x5454102u, 0x6490433u, 0x08F1C94u, 0x2803C03u, 0x4060862u, 0x50A18A1u, 0x58C20C5u,
    0x0D64854u, 0x2C71CD2u, 0x3CB2C01u, 0x44D3422u, 0x5514461u, 0x5D30085u, 0x1062811u, 0x1883035u, 0x4120CD1u, 0x49414F2u, 0x5812421u, 0x6032C46u,
    0x1CF1103u, 0x3552852u, 0x4423891u, 0x4C440B2u, 0x5C804F3u, 0x00E1C43u, 0x19434A1u, 0x2163CC3u, 0x3850812u, 0x4891854u, 0x69138D2u, 0x0554803u,
    0x1C41464u, 0x3CC34E3u, 0x5520032u, 0x6561071u, 0x6C11892u, 0x08528D4u, 0x28D4845u, 0x50024E3u, 0x6863C33u, 0x0CC0893u, 0x25220F1u, 0x2D42803u,
    0x4434063u, 0x5C90CC2u, 0x6CD1D01u, 0x00F2412u, 0x1133454u, 0x30408D3u, 0x48A2022u, 0x58E3061u, 0x6103882u, 0x71448C1u, 0x05604E5u, 0x2C92C71u,
    0x34B3495u, 0x5D51021u, 0x6401843u, 0x08630A2u, 0x18A40E3u, 0x3100C31u, 0x3921453u, 0x5012CB2u, 0x6053CF1u, 0x6874405u, 0x1D120A3u, 0x3403904u,
    0x5480C72u, 0x64C1CB4u, 0x1143C25u, 0x38718C4u, 0x58F3832u, 0x6934871u, 0x7150492u, 0x0C214D1u, 0x1441CF2u, 0x2482C24u, 0x45000A5u, 0x6C32833u,
    0x1094091u, 0x18B48B2u, 0x28F0CF7u, 0x60644C3u, 0x04C1012u, 0x1502051u, 0x1D22872u, 0x2D638B3u, 0x4450401u, 0x4C70C23u, 0x64D2482u, 0x01134C3u,
    0x1800011u, 0x2020835u, 0x48C30D1u, 0x50E38F5u, 0x0411481u, 0x0C31CA2u, 0x1C72CE3u, 0x34D4433u, 0x4D31091u, 0x55518B3u, 0x6C43002u, 0x0884041u,
    0x10A4862u, 0x20E0CA3u, 0x3942503u, 0x5033C51u, 0x5854473u, 0x70B10D3u, 0x1512823u, 0x2C04082u, 0x3C404C3u, 0x54A1C16u, 0x11600D1u, 0x18108F2u,
    0x2851821u, 0x3072042u, 0x40B3084u, 0x6130503u, 0x0421C52u, 0x1462C94u, 0x34E0002u, 0x4521044u, 0x64330C3u, 0x0894813u, 0x20F1472u, 0x31324B1u,
    0x3952CD2u, 0x4823C03u, 0x6080864u, 0x0D028E2u, 0x1D43811u, 0x2564032u, 0x3430471u, 0x3C50C95u, 0x64F3421u, 0x6D13C45u, 0x20418E1u, 0x2862103u,
    0x40C3852u, 0x5104893u, 0x69614F1u, 0x7011C05u, 0x24B44A3u, 0x3D11103u, 0x5402854u, 0x00848D3u, 0x18E1422u, 0x2922461u, 0x3142C83u, 0x48344E2u,
    0x5870813u, 0x70D2074u, 0x1D540F2u, 0x2C20423u, 0x4481C81u, 0x4CA24A2u, 0x5CE34E7u, 0x20520B2u, 0x30930F3u, 0x48F4841u, 0x5110462u, 0x61514A3u,
    0x0442D01u, 0x0C63413u, 0x24C0073u, 0x3D218D2u, 0x4D62806u, 0x08B0CC1u, 0x10D14E3u, 0x2932C33u, 0x4024492u, 0x50608D6u, 0x0D23881u, 0x15440A5u,
    0x3C71C31u, 0x4492452u, 0x54D3491u, 0x5CF3CB2u, 0x6D300F3u, 0x1021841u, 0x1842063u, 0x30A38C3u, 0x4900415u, 0x7032CB3u, 0x1494401u, 0x1CB0025u,
    0x45528C1u, 0x4C030E3u, 0x6464832u, 0x00A0C73u, 0x19024D4u, 0x3814442u, 0x4850881u, 0x50710A2u, 0x60B20E1u, 0x68D2906u, 0x2420CB3u, 0x3C82402u,
    0x4CC3443u, 0x65200A1u, 0x6D408C2u, 0x0811903u, 0x2073051u, 0x2893876u, 0x5951C22u, 0x6822C61u, 0x7043482u, 0x0C844C4u, 0x2D01833u, 0x4563092u,
    0x54340D1u, 0x5C548F3u, 0x00B1445u, 0x2953CE1u, 0x3004505u, 0x58A2093u, 0x71038F1u, 0x0524002u, 0x1560443u, 0x2C51CA1u, 0x34724C3u, 0x4CD3C12u,
    0x5D10051u, 0x6530872u, 0x00018B3u, 0x1863003u, 0x30C4861u, 0x38E0483u, 0x5141CE2u, 0x6012C11u, 0x6833435u, 0x1CD10D3u, 0x3532824u, 0x54448A3u,
    0x6CA1502u, 0x08E2431u, 0x1102C52u, 0x2143C94u, 0x4051003u, 0x58B2862u, 0x68F38A3u, 0x0D50501u, 0x1400C15u, 0x3CA34B1u, 0x44C3CD3u, 0x5D20823u,
    0x0012085u, 0x28B4813u, 0x4111471u, 0x4931C93u, 0x60234F2u, 0x7064421u, 0x0480042u, 0x14C1081u, 0x1CE18A5u, 0x4414037u, 0x08F2C02u, 0x1933C41u,
    0x2154465u, 0x4882101u, 0x50A2815u, 0x05404B1u, 0x0D60CD2u, 0x1C31C03u, 0x3493463u, 0x4CF00C1u, 0x55108E5u, 0x0843071u, 0x1063893u, 0x28C04F2u,
    0x3901427u, 0x7070102u, 0x0CB1031u, 0x14D1852u, 0x2512891u, 0x2D330B2u, 0x3C040F4u, 0x5C81463u, 0x00E2CC2u, 0x1123D03u, 0x2810851u, 0x3031072u,
    0x40720B3u, 0x58D3801u, 0x60F4023u, 0x0550C83u, 0x1C424E3u, 0x34A3C32u, 0x44E0073u, 0x5D418D1u, 0x65620F3u, 0x0853842u, 0x1894883u, 0x30F14E6u,
    0x6044491u, 0x68600B6u, 0x2523062u, 0x35640A1u, 0x3C148C2u, 0x4C50D03u, 0x64B2451u, 0x6CD2C73u, 0x11344D2u, 0x2000801u, 0x2821022u, 0x3862063u,
    0x50C38C3u, 0x6920414u, 0x1432492u, 0x24734D1u, 0x2C93CF3u, 0x44F0842u, 0x5531883u, 0x6C230E4u, 0x18A0452u, 0x28E1491u, 0x3101CB2u, 0x4142CF1u,
    0x4963409u, 0x1D13012u, 0x2D54053u, 0x4440CB6u, 0x0103C61u, 0x0924483u, 0x20110E3u, 0x3872832u, 0x48B3871u, 0x50D4092u, 0x61104D3u, 0x0401C21u,
    0x0C22446u, 0x3CE0902u, 0x4D21831u, 0x5542056u, 0x1090405u, 0x3932CA1u, 0x41534C2u, 0x5024501u, 0x5840012u, 0x6881053u, 0x0CE28B1u, 0x15030D3u,
    0x2D64822u, 0x3C30C63u, 0x54924C3u, 0x6CF3C14u, 0x1801093u, 0x30628F2u, 0x40A3821u, 0x48C4043u, 0x6120CA2u, 0x7161CE3u, 0x1453434u, 0x34D08B2u,
    0x45118F1u, 0x4D32002u, 0x5C03043u, 0x00648A4u, 0x20E1C13u, 0x3943476u, 0x6891822u, 0x04D2863u, 0x1D340C1u, 0x25548E3u, 0x3C41435u, 0x64E3CD1u,
    0x6D044F2u, 0x0940823u, 0x2032081u, 0x28528A3u, 0x40B4102u, 0x50F0431u, 0x5910C53u, 0x70024B5u, 0x24A0041u, 0x2CC0865u, 0x5563101u, 0x5C13812u,
    0x6C54851u, 0x0070472u, 0x10B14B3u, 0x2912C04u, 0x4820082u, 0x58610C1u, 0x60818E2u, 0x70C2816u, 0x2C10CD1u, 0x34314F3u, 0x4C92C42u, 0x5CD3C81u,
    0x64F44A3u, 0x0951105u, 0x3083896u, 0x6141C41u, 0x6962462u, 0x04334A3u, 0x1C90104u, 0x3D12073u, 0x54038D2u, 0x6444803u, 0x08A1461u, 0x10C1C82u,
    0x2102CC3u, 0x3964411u, 0x4010033u, 0x5871895u, 0x0D14021u, 0x1534842u, 0x2400C83u, 0x3C624E1u, 0x4482D05u, 0x6D20891u, 0x01410B2u, 0x10120F1u,
    0x1832805u, 0x40D04A1u, 0x48F0CC5u, 0x7023451u, 0x0443C72u, 0x14800B3u, 0x2CE1804u, 0x4D63883u, 0x64504E3u, 0x08B1C32u, 0x18F2C73u, 0x31544D3u,
    0x4841021u, 0x5061843u, 0x68C30A2u, 0x05040E1u, 0x0D24903u, 0x2411452u, 0x3452493u, 0x4CB3CF4u, 0x6D31062u, 0x08020A1u, 0x10228C3u, 0x2884012u,
    0x38C0454u, 0x59424D3u, 0x7033C22u, 0x0C70063u, 0x24D18C1u, 0x2CF20E2u, 0x3D33013u, 0x5424874u, 0x00A1CF3u, 0x1903442u, 0x2944481u, 0x31600A5u,
    0x5892831u, 0x60B3053u, 0x05148B2u, 0x1550CF1u, 0x1C01402u, 0x2C42441u, 0x3462C65u, 0x5D00906u, 0x18538B1u, 0x20740D2u, 0x30B0401u, 0x38D0C22u,
    0x4911C64u, 0x6823CE3u, 0x0C80832u, 0x1CC1871u, 0x24E2092u, 0x35230D6u, 0x6471481u, 0x6C91CA3u, 0x10F3502u, 0x2134436u, 0x50828F3u, 0x68E4044u,
    0x15614C2u, 0x2432501u, 0x2C52C12u, 0x3C93C51u, 0x44B4472u, 0x54F08B4u, 0x0002823u, 0x1864085u, 0x4101C13u, 0x5963473u, 0x70500D1u, 0x04708F6u,
    0x35338A2u, 0x44048E1u, 0x4C20502u, 0x5C61433u, 0x00C2C91u, 0x08E34B3u, 0x2140002u, 0x3011041u, 0x3831862u, 0x48728A1u, 0x50930C6u, 0x0D51475u,
    0x3483C01u, 0x3CA4423u, 0x5501082u, 0x65420C3u, 0x0833811u, 0x1054033u, 0x28B0C92u, 0x38F1CD1u, 0x41124F2u, 0x5153423u, 0x6840083u, 0x0CA18E4u,
    0x2D23852u, 0x3D64891u, 0x44104B3u, 0x5C71C05u, 0x11144A4u, 0x3021812u, 0x4062853u, 0x58C40B1u, 0x60E48D2u, 0x7120C04u, 0x1C32C85u, 0x44D0813u,
    0x5D32071u, 0x6552892u, 0x00238D4u, 0x20A0C43u, 0x39024A3u, 0x5163D02u, 0x6030031u, 0x6850852u, 0x0491893u, 0x1CF30F4u, 0x3C00462u, 0x4C414A3u,
    0x64A2D01u, 0x6CC3415u, 0x21610B1u, 0x28118D5u, 0x50B4061u, 0x58D4882u, 0x6910CC1u, 0x71314E5u, 0x2463C71u, 0x2C84493u, 0x44E10F2u, 0x5522021u,
    0x5D42842u, 0x6C13883u, 0x10704E3u, 0x28D1C31u, 0x30F2453u, 0x4953CB3u, 0x6040803u, 0x04A2062u, 0x14E30A3u, 0x2D44904u, 0x4C51C73u, 0x64B34D2u,
    0x00F4401u, 0x0910022u, 0x1951064u, 0x38630E5u, 0x6100C74u, 0x0C12CF2u, 0x1C53C23u, 0x34B0881u, 0x3CD10A3u, 0x5532903u, 0x6C24052u, 0x0860491u,
    0x1080CB2u, 0x20C1CF7u, 0x58308C2u, 0x6871901u, 0x7092012u, 0x0CD3051u, 0x14F3875u, 0x3C21401u, 0x4441C25u, 0x6CE44C1u, 0x01000E2u, 0x1141011u,
    0x1961832u, 0x2832873u, 0x40940D1u, 0x48B48F5u, 0x7152481u, 0x0402CA5u, 0x2CA0834u, 0x4D228B3u, 0x6414002u, 0x0050444u, 0x20D24C2u, 0x3113503u,
    0x4800054u, 0x68820D2u, 0x04C3003u, 0x1D24861u, 0x2540482u, 0x34114C4u, 0x5493433u, 0x6CF0092u, 0x09310D3u, 0x2022821u, 0x2843042u, 0x3884083u,
    0x50E0CE4u, 0x7162C53u, 0x14544B2u, 0x24908F3u, 0x3CF2043u, 0x55538A1u, 0x5C040C3u, 0x0060C13u, 0x18C2472u, 0x29034B1u, 0x3123CD5u, 0x5851861u,
    0x6072085u, 0x1514811u, 0x1D30432u, 0x2C01473u, 0x4462CD3u, 0x5CC4421u, 0x64E0043u, 0x09418A2u, 0x18128E1u, 0x2033105u, 0x48D0C93u, 0x61324F1u,
    0x6952C03u, 0x0C44463u, 0x24A10C3u, 0x3D02812u, 0x4D43853u, 0x64304B6u, 0x20F3461u, 0x2913C82u, 0x39500C1u, 0x40008E6u, 0x70C3893u, 0x15204F2u,
    0x2561423u, 0x3C52C81u, 0x44734A2u, 0x54B44E4u, 0x0131856u, 0x3084802u, 0x40C0C41u, 0x48E1462u, 0x59224A4u, 0x0434413u, 0x1C91072u, 0x2CD20B1u,
    0x34F28D2u, 0x4533801u, 0x4D54025u, 0x0081CC1u, 0x08A24E5u, 0x3140074u, 0x50520F2u, 0x6093023u, 0x04F4881u, 0x0D104A3u, 0x2401D02u, 0x3442C33u,
    0x4CA4493u, 0x65010F3u, 0x0962841u, 0x1013063u, 0x28748C2u, 0x38B0D01u, 0x40D1415u, 0x6803CB3u, 0x0C60804u, 0x2CE2883u, 0x45440E2u, 0x5410411u,
    0x5C30C32u, 0x6C71C77u, 0x3150842u, 0x4021883u, 0x58830E1u, 0x60A3902u, 0x70E4833u, 0x1541491u, 0x1D61CB3u, 0x3453403u, 0x4CB0063u, 0x65118C5u,
    0x1844051u, 0x2064873u, 0x38C14D2u, 0x4902401u, 0x5122C22u, 0x6163C66u, 0x1CB2016u, 0x4C004D1u, 0x5420CF2u, 0x6461C21u, 0x6C82445u, 0x21200E1u,
    0x2940903u, 0x4032053u, 0x58938B2u, 0x68D48F3u, 0x0D31443u, 0x2422CA1u, 0x2C434C5u, 0x54E1051u, 0x5D01873u, 0x01630D2u, 0x1034007u, 0x4912CE2u,
    0x5953C11u, 0x6004432u, 0x7040871u, 0x0461092u, 0x14A20D4u, 0x3524043u, 0x4C10CA2u, 0x5C51CE3u, 0x00B3431u, 0x08D3C52u, 0x1910093u, 0x30018F1u,
    0x3822003u, 0x5083863u, 0x68E04C2u, 0x0521501u, 0x0D41C12u, 0x1C12C53u, 0x34744B1u, 0x3C900D3u, 0x54F1822u, 0x6532861u, 0x6D53088u, 0x38E2471u
};
// SKR_MR[o_i + (p mod r_i) r_i + (q mod r_i)] = q p^-1 mod r_i (the multiplier m of q = p m)
__constant uchar SKR_MR[2020] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 8, 0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4,
    10, 16, 5, 11, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3, 16, 12, 8, 4, 0, 7, 14, 4, 11, 1, 8, 15, 5, 12, 2,
    9, 16, 6, 13, 3, 10, 0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 16, 2, 5, 8, 11, 14, 0, 5, 10, 15, 3, 8, 13, 1, 6,
    11, 16, 4, 9, 14, 2, 7, 12, 0, 15, 13, 11, 9, 7, 5, 3, 1, 16, 14, 12, 10, 8, 6, 4, 2, 0, 2, 4, 6, 8, 10, 12,
    14, 16, 1, 3, 5, 7, 9, 11, 13, 15, 0, 12, 7, 2, 14, 9, 4, 16, 11, 6, 1, 13, 8, 3, 15, 10, 5, 0, 14, 11, 8, 5,
    2, 16, 13, 10, 7, 4, 1, 15, 12, 9, 6, 3, 0, 10, 3, 13, 6, 16, 9, 2, 12, 5, 15, 8, 1, 11, 4, 14, 7, 0, 4, 8,
    12, 16, 3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 11, 5, 16, 10, 4, 15, 9, 3, 14, 8, 2, 13, 7, 1, 12, 6, 0,
    8, 16, 7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18, 9, 0, 13, 7, 1, 14, 8,
    2, 15, 9, 3, 16, 10, 4, 17, 11, 5, 18, 12, 6, 0, 5, 10, 15, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14,
    0, 4, 8, 12, 16, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 0, 16, 13, 10, 7, 4, 1, 17, 14, 11, 8, 5, 2,
    18, 15, 12, 9, 6, 3, 0, 11, 3, 14, 6, 17, 9, 1, 12, 4, 15, 7, 18, 10, 2, 13, 5, 16, 8, 0, 12, 5, 17, 10, 3, 15,
    8, 1, 13, 6, 18, 11, 4, 16, 9, 2, 14, 7, 0, 17, 15, 13, 11, 9, 7, 5, 3, 1, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0,
    2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 0, 7, 14, 2, 9, 16, 4, 11, 18, 6, 13, 1, 8, 15,
    3, 10, 17, 5, 12, 0, 8, 16, 5, 13, 2, 10, 18, 7, 15, 4, 12, 1, 9, 17, 6, 14, 3, 11, 0, 3, 6, 9, 12, 15, 18, 2,
    5, 8, 11, 14, 17, 1, 4, 7, 10, 13, 16, 0, 15, 11, 7, 3, 18, 14, 10, 6, 2, 17, 13, 9, 5, 1, 16, 12, 8, 4, 0, 14,
    9, 4, 18, 13, 8, 3, 17, 12, 7, 2, 16, 11, 6, 1, 15, 10, 5, 0, 6, 12, 18, 5, 11, 17, 4, 10, 16, 3, 9, 15, 2, 8,
    14, 1, 7, 13, 0, 9, 18, 8, 17, 7, 16, 6, 15, 5, 14, 4, 13, 3, 12, 2, 11, 1, 10, 0, 18, 17, 16, 15, 14, 13, 12, 11,
    10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 12, 1, 13, 2, 14, 3, 15,
    4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5, 13,
    21, 6, 14, 22, 7, 15, 0, 6, 12, 18, 1, 7, 13, 19, 2, 8, 14, 20, 3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 0, 14, 5,
    19, 10, 1, 15, 6, 20, 11, 2, 16, 7, 21, 12, 3, 17, 8, 22, 13, 4, 18, 9, 0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21,
    2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 0, 10, 20, 7, 17, 4, 14, 1, 11, 21, 8, 18, 5, 15, 2, 12, 22, 9, 19, 6, 16,
    3, 13, 0, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10, 13, 16, 19, 22, 2, 5, 8, 11, 14, 17, 20, 0, 18, 13, 8, 3, 21, 16,
    11, 6, 1, 19, 14, 9, 4, 22, 17, 12, 7, 2, 20, 15, 10, 5, 0, 7, 14, 21, 5, 12, 19, 3, 10, 17, 1, 8, 15, 22, 6, 13,
    20, 4, 11, 18, 2, 9, 16, 0, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 2,
    4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 0, 16, 9, 2, 18, 11, 4, 20, 13, 6, 22,
    15, 8, 1, 17, 10, 3, 19, 12, 5, 21, 14, 7, 0, 5, 10, 15, 20, 2, 7, 12, 17, 22, 4, 9, 14, 19, 1, 6, 11, 16, 21, 3,
    8, 13, 18, 0, 20, 17, 14, 11, 8, 5, 2, 22, 19, 16, 13, 10, 7, 4, 1, 21, 18, 15, 12, 9, 6, 3, 0, 13, 3, 16, 6, 19,
    9, 22, 12, 2, 15, 5, 18, 8, 21, 11, 1, 14, 4, 17, 7, 20, 10, 0, 19, 15, 11, 7, 3, 22, 18, 14, 10, 6, 2, 21, 17, 13,
    9, 5, 1, 20, 16, 12, 8, 4, 0, 9, 18, 4, 13, 22, 8, 17, 3, 12, 21, 7, 16, 2, 11, 20, 6, 15, 1, 10, 19, 5, 14, 0,
    17, 11, 5, 22, 16, 10, 4, 21, 15, 9, 3, 20, 14, 8, 2, 19, 13, 7, 1, 18, 12, 6, 0, 15, 7, 22, 14, 6, 21, 13, 5, 20,
    12, 4, 19, 11, 3, 18, 10, 2, 17, 9, 1, 16, 8, 0, 11, 22, 10, 21, 9, 20, 8, 19, 7, 18, 6, 17, 5, 16, 4, 15, 3, 14,
    2, 13, 1, 12, 0, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 0, 15, 1, 16, 2, 17, 3, 18, 4, 19, 5,
    20, 6, 21, 7, 22, 8, 23, 9, 24, 10, 25, 11, 26, 12, 27, 13, 28, 14, 0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14,
    24, 5, 15, 25, 6, 16, 26, 7, 17, 27, 8, 18, 28, 9, 19, 0, 22, 15, 8, 1, 23, 16, 9, 2, 24, 17, 10, 3, 25, 18, 11, 4,
    26, 19, 12, 5, 27, 20, 13, 6, 28, 21, 14, 7, 0, 6, 12, 18, 24, 1, 7, 13, 19, 25, 2, 8, 14, 20, 26, 3, 9, 15, 21, 27,
    4, 10, 16, 22, 28, 5, 11, 17, 23, 0, 5, 10, 15, 20, 25, 1, 6, 11, 16, 21, 26, 2, 7, 12, 17, 22, 27, 3, 8, 13, 18, 23,
    28, 4, 9, 14, 19, 24, 0, 25, 21, 17, 13, 9, 5, 1, 26, 22, 18, 14, 10, 6, 2, 27, 23, 19, 15, 11, 7, 3, 28, 24, 20, 16,
    12, 8, 4, 0, 11, 22, 4, 15, 26, 8, 19, 1, 12, 23, 5, 16, 27, 9, 20, 2, 13, 24, 6, 17, 28, 10, 21, 3, 14, 25, 7, 18,
    0, 13, 26, 10, 23, 7, 20, 4, 17, 1, 14, 27, 11, 24, 8, 21, 5, 18, 2, 15, 28, 12, 25, 9, 22, 6, 19, 3, 16, 0, 3, 6,
    9, 12, 15, 18, 21, 24, 27, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 2, 5, 8, 11, 14, 17, 20, 23, 26, 0, 8, 16, 24, 3, 11,
    19, 27, 6, 14, 22, 1, 9, 17, 25, 4, 12, 20, 28, 7, 15, 23, 2, 10, 18, 26, 5, 13, 21, 0, 17, 5, 22, 10, 27, 15, 3, 20,
    8, 25, 13, 1, 18, 6, 23, 11, 28, 16, 4, 21, 9, 26, 14, 2, 19, 7, 24, 12, 0, 9, 18, 27, 7, 16, 25, 5, 14, 23, 3, 12,
    21, 1, 10, 19, 28, 8, 17, 26, 6, 15, 24, 4, 13, 22, 2, 11, 20, 0, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1,
    28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 1, 3, 5,
    7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 0, 20, 11, 2, 22, 13, 4, 24, 15, 6, 26, 17, 8, 28, 19, 10, 1, 21, 12, 3, 23,
    14, 5, 25, 16, 7, 27, 18, 9, 0, 12, 24, 7, 19, 2, 14, 26, 9, 21, 4, 16, 28, 11, 23, 6, 18, 1, 13, 25, 8, 20, 3, 15,
    27, 10, 22, 5, 17, 0, 21, 13, 5, 26, 18, 10, 2, 23, 15, 7, 28, 20, 12, 4, 25, 17, 9, 1, 22, 14, 6, 27, 19, 11, 3, 24,
    16, 8, 0, 26, 23, 20, 17, 14, 11, 8, 5, 2, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0,
    16, 3, 19, 6, 22, 9, 25, 12, 28, 15, 2, 18, 5, 21, 8, 24, 11, 27, 14, 1, 17, 4, 20, 7, 23, 10, 26, 13, 0, 18, 7, 25,
    14, 3, 21, 10, 28, 17, 6, 24, 13, 2, 20, 9, 27, 16, 5, 23, 12, 1, 19, 8, 26, 15, 4, 22, 11, 0, 4, 8, 12, 16, 20, 24,
    28, 3, 7, 11, 15, 19, 23, 27, 2, 6, 10, 14, 18, 22, 26, 1, 5, 9, 13, 17, 21, 25, 0, 24, 19, 14, 9, 4, 28, 23, 18, 13,
    8, 3, 27, 22, 17, 12, 7, 2, 26, 21, 16, 11, 6, 1, 25, 20, 15, 10, 5, 0, 23, 17, 11, 5, 28, 22, 16, 10, 4, 27, 21, 15,
    9, 3, 26, 20, 14, 8, 2, 25, 19, 13, 7, 1, 24, 18, 12, 6, 0, 7, 14, 21, 28, 6, 13, 20, 27, 5, 12, 19, 26, 4, 11, 18,
    25, 3, 10, 17, 24, 2, 9, 16, 23, 1, 8, 15, 22, 0, 19, 9, 28, 18, 8, 27, 17, 7, 26, 16, 6, 25, 15, 5, 24, 14, 4, 23,
    13, 3, 22, 12, 2, 21, 11, 1, 20, 10, 0, 14, 28, 13, 27, 12, 26, 11, 25, 10, 24, 9, 23, 8, 22, 7, 21, 6, 20, 5, 19, 4,
    18, 3, 17, 2, 16, 1, 15, 0, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
    4, 3, 2, 1
};
/// Packed targets tau_i = (M[w] - m) mod r_i for the mark at dense bit `bit`
/// (q = q_first + 2 bit = p m) at wheel position w.
/// @param qfr  q_first mod SKR_R.
inline u32 skr_target_q(const u32 q, const u32 p, const u32 mw)
{
    u32 T = 0u;
#define SKR_F(i, o, r) { const u32 mr = SKR_MR[o + (p % r) * r + q % r]; const u32 x = (mw >> (6u * i)) & 63u; \
        T |= ((x >= mr) ? x - mr : x + r - mr) << (4u + 6u * i); }
    SKR_F(0u, 0u, 17u) SKR_F(1u, 289u, 19u) SKR_F(2u, 650u, 23u) SKR_F(3u, 1179u, 29u)
#undef SKR_F
    return T;
}
inline u32 skr_target(const u32 bit, const u32 p, const u32 w, const u32 qfr)
{
    return skr_target_q(qfr + 2u * bit, p, WHL_SF[w] >> 4);   // q = qfr + 2 bit < 2^31
}
/// T after a wheel wrap (m grows by 30030 relative to M): tau_i -= 30030 mod r_i.
inline u32 skr_wrap(const u32 T)
{
    u32 o = 0u;
#define SKR_W(i, r, k) { const u32 x = (T >> (4u + 6u * i)) & 63u; \
        o |= ((x >= k) ? x - k : x + r - k) << (4u + 6u * i); }
    SKR_W(0u, 17u, 8u) SKR_W(1u, 19u, 10u) SKR_W(2u, 23u, 15u) SKR_W(3u, 29u, 15u)
#undef SKR_W
    return o;
}
/// mark_multiples_whl without the marks p m, m divisible by 17, 19, 23 or 29.
/// @pre    as mark_multiples_whl; qfr = q_first mod SKR_R; the small-prime
///         stage clears every multiple of 17..29 (mask groups: 3..61).
/// @post   as mark_multiples_whl (the skipped bits are cleared by the mask stage).
inline void mark_multiples_skr(volatile __global u64* prime_map, const u64 bit0,
                               const u32 p, const u64 q_odds, const u32 qfw, const u32 qfr,
                               const cmap g)
{
    volatile __global u32* m32 = (volatile __global u32*)prime_map;
    const u32 n = (u32)q_odds;
    u32 bit = (u32)bit0;
    if (bit >= n) return;
    u32 m = (qfw + 2u * (bit % (WHEEL_W / 2))) % WHEEL_W;
    m = (m * WHL_INV[p % WHEEL_W]) % WHEEL_W;
    while (WHL_IDX[m] == WHL_MISS) {
        if (n - bit <= p) return;
        bit += p;
        m += 2u; if (m >= WHEEL_W) m -= WHEEL_W;
    }
    u32 w = WHL_IDX[m];
    u32 T = skr_target(bit, p, w, qfr);
    for (;;) {
        const u32 e = WHL_SF[w];
        if (((((e ^ T) | SKR_H) - SKR_L) & SKR_H) == SKR_H)
            map_clear_w(m32, bit, g);
        const u64 step = (u64)p * (e & 15u);
        if ((u64)(n - bit) <= step) break;
        bit += (u32)step;
        if (++w == WHL_N) { w = 0u; T = skr_wrap(T); }
    }
}


// Second residue set for the many-mark primes (SKR2_MAXP): r = 31, 37, 41, 43
// in 7-bit fields of WHL_SF2[w] = sum (M[w] mod r_i) << 7 i (another 10.3 pct
// of the remaining marks; costs a second table load per step).
#ifndef SKR2_MAXP
#define SKR2_MAXP 10000000u
#endif
#define SKR2_H 0x8102040u
#define SKR2_L 0x0204081u
__constant uint WHL_SF2[5760] = {
    0x0204081u, 0x2244891u, 0x264C993u, 0x2E5CB97u, 0x3A74E9Du, 0x3E7CF80u, 0x4A94006u, 0x520020Au, 0x000830Cu, 0x0818510u,
    0x1430816u, 0x2048B1Cu, 0x2450C1Eu, 0x3068F05u, 0x3879109u, 0x3C8120Bu, 0x4898291u, 0x5004495u, 0x061C79Bu, 0x163CB84u,
    0x1E4CD88u, 0x2254E8Au, 0x2A6508Eu, 0x2E6D190u, 0x367C114u, 0x5210803u, 0x0420A07u, 0x1038D0Du, 0x1440E0Fu, 0x2868099u,
    0x2C7019Bu, 0x3888482u, 0x44A0788u, 0x4C0C98Cu, 0x0224C92u, 0x0E3CF98u, 0x124509Au, 0x266C305u, 0x2A74407u, 0x328460Bu,
    0x368C70Du, 0x4E18D19u, 0x1048086u, 0x185828Au, 0x1C6038Cu, 0x2470590u, 0x3088896u, 0x3490998u, 0x4814E83u, 0x542D189u,
    0x0A4420Fu, 0x165C515u, 0x1A64617u, 0x267C91Du, 0x2E8CB02u, 0x3294C04u, 0x3E08F0Au, 0x461910Eu, 0x0C5059Cu, 0x1460781u,
    0x1868883u, 0x2078A87u, 0x2C90D8Du, 0x3C0D195u, 0x482421Bu, 0x064C706u, 0x0A54808u, 0x1264A0Cu, 0x1E7CD12u, 0x2284E14u,
    0x2E9D11Au, 0x3A10181u, 0x4628487u, 0x4E3868Bu, 0x0450991u, 0x0858A93u, 0x1470D99u, 0x1C80F9Du, 0x2CA0106u, 0x4024610u,
    0x442C712u, 0x0254C1Cu, 0x065CD1Eu, 0x0E6CF03u, 0x1275005u, 0x1A85209u, 0x269C28Fu, 0x3618697u, 0x3E2889Bu, 0x423099Du,
    0x4A40B82u, 0x0C7118Eu, 0x1C90316u, 0x24A051Au, 0x280461Cu, 0x341C903u, 0x3C2CB07u, 0x4844E0Du, 0x0A74199u, 0x0E7C29Bu,
    0x168C480u, 0x1A94582u, 0x3220B8Eu, 0x3E38E94u, 0x4649098u, 0x526011Eu, 0x0878405u, 0x149070Bu, 0x189880Du, 0x240CB13u,
    0x383501Du, 0x3C3D100u, 0x444C084u, 0x506438Au, 0x546C48Cu, 0x0A84792u, 0x169CA98u, 0x1E08C9Cu, 0x2210D9Eu, 0x3638009u,
    0x3A4010Bu, 0x4E68615u, 0x5270717u, 0x048091Bu, 0x1098C02u, 0x1C0CF08u, 0x201500Au, 0x2C2C090u, 0x3844396u, 0x405459Au,
    0x4C6C881u, 0x068CC89u, 0x1200F8Fu, 0x1A11193u, 0x1E18015u, 0x2A3031Bu, 0x3240500u, 0x3E58806u, 0x4E78C0Eu, 0x0088E12u,
    0x0490F14u, 0x100521Au, 0x181419Eu, 0x2834587u, 0x344C88Du, 0x3C5CA91u, 0x4C7CE99u, 0x548D09Du, 0x0A00104u, 0x1A2050Cu,
    0x2E48A16u, 0x3250B18u, 0x4679003u, 0x4A81105u, 0x009818Bu, 0x080438Fu, 0x0C0C491u, 0x141C695u, 0x1824797u, 0x2C4CC82u,
    0x3054D84u, 0x447C00Eu, 0x4884110u, 0x5094314u, 0x549C416u, 0x060861Au, 0x2240D09u, 0x2A50F0Du, 0x2E5900Fu, 0x3669213u,
    0x4280299u, 0x4E98580u, 0x52A0682u, 0x0814988u, 0x1024B8Cu, 0x2044F94u, 0x346C21Eu, 0x448C607u, 0x4C9C80Bu, 0x500090Du,
    0x0210B11u, 0x0E28E17u, 0x1E49200u, 0x2A60286u, 0x327048Au, 0x3E88790u, 0x4AA0A96u, 0x0014D9Cu, 0x041CE9Eu, 0x1035185u,
    0x1C4C20Bu, 0x245C40Fu, 0x2864511u, 0x3074715u, 0x3C8CA1Bu, 0x4094B1Du, 0x5419008u, 0x022110Au, 0x0A3008Eu, 0x0E38190u,
    0x226069Au, 0x266879Cu, 0x3A90C87u, 0x3E98D89u, 0x4A0D08Fu, 0x521C013u, 0x0C3C41Bu, 0x1854702u, 0x2064906u, 0x246CA08u,
    0x2C7CC0Cu, 0x3894F12u, 0x4409218u, 0x5428381u, 0x0638585u, 0x0A40687u, 0x165898Du, 0x2A80E97u, 0x3AA0000u, 0x420C204u,
    0x4614306u, 0x522C60Cu, 0x043C810u, 0x145CC18u, 0x2885103u, 0x349C189u, 0x380028Bu, 0x401048Fu, 0x5030897u, 0x0648B9Du,
    0x1260E84u, 0x1A71088u, 0x1E7918Au, 0x268810Eu, 0x32A0414u, 0x3604516u, 0x421C81Cu, 0x4A2CA01u, 0x0044D07u, 0x044CE09u,
    0x1874093u, 0x3000680u, 0x3408782u, 0x3C18986u, 0x4020A88u, 0x4830C8Cu, 0x5448F92u, 0x0251094u, 0x0E6811Au, 0x167831Eu,
    0x1A80401u, 0x2290605u, 0x3A1CC11u, 0x3E24D13u, 0x4A3D019u, 0x0054080u, 0x147C58Au, 0x2094890u, 0x3010C98u, 0x3820E9Cu,
    0x3C28F9Eu, 0x4439183u, 0x4840005u, 0x5050209u, 0x0A70611u, 0x1688917u, 0x2E14F04u, 0x3625108u, 0x423C18Eu, 0x4644290u,
    0x087489Cu, 0x1084A81u, 0x148CB83u, 0x1C9CD87u, 0x281108Du, 0x3830215u, 0x4040419u, 0x444851Bu, 0x4C58700u, 0x5060802u,
    0x1290E0Eu, 0x2614098u, 0x2A1C19Au, 0x322C39Eu, 0x3634481u, 0x3E44685u, 0x4A5C98Bu, 0x4E64A8Du, 0x0C8CF97u, 0x1095099u,
    0x180001Du, 0x2418304u, 0x343870Cu, 0x4050A12u, 0x4860C16u, 0x4C68D18u, 0x028101Eu, 0x0A91203u, 0x1604289u, 0x2624691u,
    0x2E34895u, 0x3A4CB9Bu, 0x3E54C9Du, 0x4664E82u, 0x008400Au, 0x0C9C310u, 0x1408514u, 0x202081Au, 0x242891Cu, 0x2C38B01u,
    0x3850E07u, 0x3C58F09u, 0x487120Fu, 0x5488295u, 0x0698499u, 0x120C780u, 0x1E24A86u, 0x2E44E8Eu, 0x3A5D194u, 0x426C118u,
    0x467421Au, 0x049C705u, 0x0800807u, 0x1C28D11u, 0x2030E13u, 0x2841017u, 0x2C49119u, 0x4070384u, 0x4478486u, 0x509078Cu,
    0x02A0990u, 0x0604A92u, 0x1A2CF9Cu, 0x2644003u, 0x2A4C105u, 0x366440Bu, 0x3E7460Fu, 0x427C711u, 0x4E94A17u, 0x0000C1Bu,
    0x0C18F02u, 0x1C3808Au, 0x2850390u, 0x3060594u, 0x3468696u, 0x4C98C83u, 0x0A1D18Du, 0x1634213u, 0x1A3C315u, 0x224C519u,
    0x2E64800u, 0x326C902u, 0x4A9CF0Eu, 0x5209112u, 0x0011214u, 0x0820198u, 0x1840581u, 0x2458887u, 0x2C68A8Bu, 0x3070B8Du,
    0x3880D91u, 0x3C88E93u, 0x500C11Du, 0x5414200u, 0x123C70Au, 0x1E54A10u, 0x225CB12u, 0x2A6CD16u, 0x368501Cu, 0x3A8D11Eu,
    0x4600185u, 0x4E10389u, 0x521848Bu, 0x1040995u, 0x1C58C9Bu, 0x2060D9Du, 0x2C79084u, 0x3488008u, 0x4C14614u, 0x022C91Au,
    0x124CD03u, 0x1E65009u, 0x267520Du, 0x2A7C08Fu, 0x328C293u, 0x420869Bu, 0x4E20982u, 0x0030B86u, 0x0C48E8Cu, 0x1050F8Eu,
    0x1861192u, 0x2478218u, 0x3498601u, 0x400C907u, 0x4C24C0Du, 0x5434E11u, 0x0A4D117u, 0x0E55219u, 0x1A6C280u, 0x227C484u,
    0x2684586u, 0x2E9478Au, 0x329C88Cu, 0x4620D96u, 0x0850103u, 0x0C58205u, 0x1468409u, 0x2C98A15u, 0x30A0B17u, 0x3C14E1Du,
    0x4425002u, 0x482D104u, 0x503C088u, 0x065438Eu, 0x126C694u, 0x1674796u, 0x2E00D83u, 0x3A19089u, 0x422800Du, 0x1070900u,
    0x1478A02u, 0x1C88C06u, 0x2090D08u, 0x28A0F0Cu, 0x381C094u, 0x443439Au, 0x4C4459Eu, 0x025C885u, 0x0664987u, 0x0E74B8Bu,
    0x1E94F93u, 0x2A08019u, 0x3620300u, 0x3E30504u, 0x4238606u, 0x4A4880Au, 0x0060B10u, 0x0468C12u, 0x1080F18u, 0x189111Cu,
    0x1C9921Eu, 0x2404183u, 0x3C3478Fu, 0x403C891u, 0x026CE9Du, 0x0E85184u, 0x1694108u, 0x220840Eu, 0x2610510u, 0x3228816u,
    0x3A38A1Au, 0x4650D01u, 0x5269007u, 0x088008Du, 0x0C8818Fu, 0x18A0495u, 0x200C699u, 0x241479Bu, 0x302CA82u, 0x383CC86u,
    0x4454F8Cu, 0x5474114u, 0x0684318u, 0x0A8C41Au, 0x129C61Eu, 0x1600701u, 0x1E10905u, 0x3A49013u, 0x4259217u, 0x4E7029Du,
    0x5278380u, 0x10A088Au, 0x140498Cu, 0x201CC92u, 0x2C34F98u, 0x344519Cu, 0x384C01Eu, 0x4C74509u, 0x507C60Bu, 0x0E00B15u,
    0x1208C17u, 0x1A18E1Bu, 0x365028Au, 0x4A78794u, 0x4E80896u, 0x0090A9Au, 0x0498B9Cu, 0x0C04D81u, 0x181D087u, 0x1C25189u,
    0x283C20Fu, 0x3C64719u, 0x487CA00u, 0x5494D06u, 0x029CE08u, 0x1620092u, 0x1A28194u, 0x264049Au, 0x2E5069Eu, 0x3A68985u,
    0x4A88D8Du, 0x5298F91u, 0x00A1093u, 0x080C017u, 0x142431Du, 0x2444706u, 0x305CA0Cu, 0x4484F16u, 0x488D018u, 0x509D21Cu,
    0x0610283u, 0x0A18385u, 0x163068Bu, 0x2248991u, 0x2A58B95u, 0x2E60C97u, 0x3670E9Bu, 0x4289182u, 0x4690004u, 0x520430Au,
    0x041450Eu, 0x081C610u, 0x1434916u, 0x285CE01u, 0x2C64F03u, 0x408C18Du, 0x4C00493u, 0x5008595u, 0x0218799u, 0x0E30A80u,
    0x1E50E88u, 0x266108Cu, 0x2A6918Eu, 0x3278112u, 0x4A0471Eu, 0x4E0C801u, 0x0424B07u, 0x0C34D0Bu, 0x103CE0Du, 0x1C55113u,
    0x2464097u, 0x307C39Du, 0x480898Au, 0x4C10A8Cu, 0x5420C90u, 0x0228D92u, 0x0A38F96u, 0x1A5811Eu, 0x2670405u, 0x2E80609u,
    0x3A9890Fu, 0x3EA0A11u, 0x5224F1Bu, 0x002D01Du, 0x0C44084u, 0x206C58Eu, 0x2C84894u, 0x389CB9Au, 0x3C00C9Cu, 0x4818F83u,
    0x5029187u, 0x5430009u, 0x064020Du, 0x0A4830Fu, 0x1E70819u, 0x227891Bu, 0x3A04F08u, 0x421510Cu, 0x4E2C192u, 0x0444498u,
    0x084C59Au, 0x207CB87u, 0x288CD8Bu, 0x3401091u, 0x4018117u, 0x4420219u, 0x5038500u, 0x0248704u, 0x0650806u, 0x1268B0Cu,
    0x1A78D10u, 0x360C19Eu, 0x3E1C383u, 0x4224485u, 0x4E3C78Bu, 0x004C98Fu, 0x106CD97u, 0x1C8509Du, 0x2494002u, 0x3008308u,
    0x341040Au, 0x3C2060Eu, 0x4838914u, 0x0258D1Cu, 0x0E71003u, 0x1A88089u, 0x2E0C593u, 0x3214695u, 0x3E2C99Bu, 0x463CB80u,
    0x5254E86u, 0x005CF88u, 0x1484212u, 0x188C314u, 0x2C1081Eu, 0x3018901u, 0x3828B05u, 0x3C30C07u, 0x4440E0Bu, 0x5461213u,
    0x0A78299u, 0x128849Du, 0x1690580u, 0x1EA0784u, 0x2A14A8Au, 0x362CD90u, 0x464D198u, 0x4E5C11Cu, 0x087C505u, 0x108C709u,
    0x1C00A0Fu, 0x2C20E17u, 0x343101Bu, 0x383911Du, 0x4048082u, 0x4450184u, 0x0680790u, 0x1298A96u, 0x1A04C9Au, 0x261CF81u,
    0x3234007u, 0x3E4C30Du, 0x425440Fu, 0x4E6C715u, 0x0484A1Bu, 0x0C94C00u, 0x109CD02u, 0x1808F06u, 0x242120Cu, 0x282808Eu,
    0x3440394u, 0x405869Au, 0x486889Eu, 0x4C70981u, 0x0A98E8Bu, 0x0EA0F8Du, 0x2224217u, 0x262C319u, 0x3244600u, 0x3A54804u,
    0x466CB0Au, 0x4A74C0Cu, 0x008CF12u, 0x089D116u, 0x0C01218u, 0x141019Cu, 0x2028483u, 0x3C60B91u, 0x4470D95u, 0x4878E97u,
    0x549119Du, 0x1214408u, 0x2234810u, 0x2A44A14u, 0x2E4CB16u, 0x365CD1Au, 0x3A64E1Cu, 0x4275001u, 0x5294189u, 0x1018693u,
    0x2038A9Bu, 0x2848C80u, 0x3869088u, 0x448010Eu, 0x0204618u, 0x060C71Au, 0x0E1C91Eu, 0x1A34C05u, 0x1E3CD07u, 0x2A5500Du,
    0x3265211u, 0x3E7C297u, 0x4284399u, 0x0008884u, 0x0410986u, 0x1838E90u, 0x1C40F92u, 0x2451196u, 0x2858018u, 0x306821Cu,
    0x3C80503u, 0x4CA090Bu, 0x540CB0Fu, 0x0214C11u, 0x0A24E15u, 0x163D11Bu, 0x2254182u, 0x265C284u, 0x327458Au, 0x3E8C890u,
    0x4A00B96u, 0x0829081u, 0x1848209u, 0x205840Du, 0x246050Fu, 0x2C70713u, 0x3888A19u, 0x4804E02u, 0x541D108u, 0x062C08Cu,
    0x164C494u, 0x1E5C698u, 0x2A7499Eu, 0x2E7CA81u, 0x3A94D87u, 0x460908Du, 0x4E18011u, 0x5220113u, 0x0430317u, 0x104861Du,
    0x2878C0Au, 0x2C80D0Cu, 0x3490F10u, 0x3899012u, 0x4C1C29Cu, 0x502439Eu, 0x0E4C889u, 0x125498Bu, 0x1A64B8Fu, 0x1E6CC91u,
    0x267CE95u, 0x369C01Du, 0x4A20508u, 0x4E2860Au, 0x003880Eu, 0x0C50B14u, 0x1C70F1Cu, 0x2889203u, 0x3098187u, 0x34A0289u,
    0x401458Fu, 0x4824793u, 0x543CA99u, 0x0E5CE82u, 0x166D086u, 0x268C20Eu, 0x2E9C412u, 0x3E1881Au, 0x4A30B01u, 0x5240D05u,
    0x085900Bu, 0x0C6110Du, 0x1470091u, 0x2088397u, 0x2490499u, 0x3004780u, 0x3C1CA86u, 0x442CC8Au, 0x5044F90u, 0x065C016u,
    0x0A64118u, 0x167C41Eu, 0x2294705u, 0x2A00909u, 0x2E08A0Bu, 0x4230F15u, 0x0460282u, 0x0868384u, 0x1078588u, 0x148068Au,
    0x1C9088Eu, 0x2804B94u, 0x2C0CC96u, 0x3824F9Cu, 0x4035181u, 0x443C003u, 0x0E7C813u, 0x1284915u, 0x1E9CC1Bu, 0x2608E00u,
    0x2A10F02u, 0x3E3818Cu, 0x4A50492u, 0x047089Au, 0x0C80A9Eu, 0x1088B81u, 0x1898D85u, 0x1CA0E87u, 0x342C213u, 0x4044519u,
    0x546CA04u, 0x0274B06u, 0x0A84D0Au, 0x169D010u, 0x1A01112u, 0x323049Eu, 0x3A40683u, 0x3E48785u, 0x4658989u, 0x0078D91u,
    0x0C91097u, 0x14A001Bu, 0x180411Du, 0x2014302u, 0x241C404u, 0x3C4CA10u, 0x5074F1Au, 0x068D201u, 0x0A94083u, 0x1200287u,
    0x1E1858Du, 0x222068Fu, 0x2E38995u, 0x3648B99u, 0x3A50C9Bu, 0x4260E80u, 0x4E79186u, 0x049020Cu, 0x089830Eu, 0x140C614u,
    0x1C1C818u, 0x202491Au, 0x344CE05u, 0x406510Bu, 0x5084293u, 0x0E0879Du, 0x1210880u, 0x1A20A84u, 0x2A40E8Cu, 0x3659192u,
    0x3E68116u, 0x4A8041Cu, 0x4E8851Eu, 0x0098703u, 0x0C0CA09u, 0x1014B0Bu, 0x1C2CE11u, 0x2845117u, 0x345C19Du, 0x3C6C382u,
    0x4884688u, 0x0200A90u, 0x0A10C94u, 0x0E18D96u, 0x1628F9Au, 0x1A3109Cu, 0x2E58307u, 0x4688913u, 0x4A90A15u, 0x52A0C19u,
    0x0004D1Bu, 0x142D206u, 0x1834088u, 0x244C38Eu, 0x2C5C592u, 0x3064694u, 0x3874898u, 0x448CB9Eu, 0x5000E85u, 0x5408F87u,
    0x1230211u, 0x1638313u, 0x2250619u, 0x2A6081Du, 0x4698F0Cu, 0x520D212u, 0x041C196u, 0x0824298u, 0x103449Cu, 0x2054885u,
    0x347CD8Fu, 0x4095095u, 0x449D197u, 0x4C0811Bu, 0x0220402u, 0x0628504u, 0x124080Au, 0x1E58B10u, 0x2668D14u, 0x2A70E16u,
    0x328101Au, 0x42A0183u, 0x4E14489u, 0x002468Du, 0x042C78Fu, 0x0C3C993u, 0x246CF80u, 0x2875082u, 0x400040Eu, 0x4810612u,
    0x4C18714u, 0x5428918u, 0x0A40C1Eu, 0x0E48D01u, 0x1A61007u, 0x227120Bu, 0x267808Du, 0x3AA0597u, 0x461489Du, 0x4A1C980u,
    0x0034C86u, 0x0844E8Au, 0x1864012u, 0x2074216u, 0x2C8C51Cu, 0x3C08905u, 0x4418B09u, 0x4820C0Bu, 0x5030E0Fu, 0x5438F11u,
    0x0649115u, 0x2280584u, 0x2A90788u, 0x3604A8Eu, 0x3A0CB90u, 0x4E3509Au, 0x0854203u, 0x146C509u, 0x1C7C70Du, 0x208480Fu,
    0x2894A13u, 0x3408D19u, 0x3810E1Bu, 0x4C38086u, 0x5040188u, 0x025038Cu, 0x065848Eu, 0x1E88A9Au, 0x320CF85u, 0x3615087u,
    0x3E2400Bu, 0x4A3C311u, 0x0054617u, 0x045C719u, 0x1074A00u, 0x1884C04u, 0x249CF0Au, 0x3011210u, 0x3C28296u, 0x4030398u,
    0x4C4869Eu, 0x0260985u, 0x0E78C8Bu, 0x1688E8Fu, 0x22A1195u, 0x3A2C502u, 0x3E34604u, 0x4644808u, 0x525CB0Eu, 0x0C7CF16u,
    0x189521Cu, 0x2C18487u, 0x3020589u, 0x383078Du, 0x4850B95u, 0x5468E9Bu, 0x0A81182u, 0x1290106u, 0x1698208u, 0x1E0440Cu,
    0x2A1C712u, 0x2E24814u, 0x3A3CB1Au, 0x424CD1Eu, 0x4654E01u, 0x526D107u, 0x1094391u, 0x149C493u, 0x282099Du, 0x2C28A80u,
    0x3840D86u, 0x4050F8Au, 0x4C68010u, 0x0688418u, 0x0E9861Cu, 0x1A0C903u, 0x323CF0Fu, 0x3645011u, 0x425C097u, 0x4A6C29Bu,
    0x4E7439Du, 0x048C684u, 0x0C9C888u, 0x1810B8Eu, 0x304119Au, 0x344801Cu, 0x3C58201u, 0x4060303u, 0x4870507u, 0x0E04C15u,
    0x1614E19u, 0x222D100u, 0x2635202u, 0x2E44186u, 0x3A5C48Cu, 0x3E6458Eu, 0x4A7C894u, 0x0800D9Eu, 0x0C08E81u, 0x203010Bu,
    0x243820Du, 0x3050513u, 0x3860717u, 0x3C68819u, 0x4478A1Du, 0x060500Au, 0x0A0D10Cu, 0x1E34396u, 0x223C498u, 0x2A4C69Cu,
    0x3664983u, 0x427CC89u, 0x4684D8Bu, 0x529D091u, 0x0810117u, 0x102031Bu, 0x1C38602u, 0x2850908u, 0x2C58A0Au, 0x4080F14u,
    0x4489016u, 0x50A009Cu, 0x020C281u, 0x0E24587u, 0x1E4498Fu, 0x2654B93u, 0x2A5CC95u, 0x3674F9Bu, 0x3E85180u, 0x4E00308u,
    0x041860Eu, 0x0C28812u, 0x1840B18u, 0x1C48C1Au, 0x2458E1Eu, 0x3071105u, 0x409028Du, 0x4C04593u, 0x5414797u, 0x021C899u,
    0x1644D84u, 0x1A4CE86u, 0x266518Cu, 0x2E74110u, 0x327C212u, 0x3E94518u, 0x5218A03u, 0x0020B05u, 0x144900Fu, 0x2060095u,
    0x2468197u, 0x2C7839Bu, 0x3C98784u, 0x480CA8Au, 0x501CC8Eu, 0x5424D90u, 0x0634F94u, 0x124C01Au, 0x1E64301u, 0x226C403u,
    0x2E84709u, 0x369490Du, 0x4610D15u, 0x4E20F19u, 0x0439200u, 0x1458388u, 0x1C6858Cu, 0x207068Eu, 0x2880892u, 0x2C88994u,
    0x3498B98u, 0x4414F81u, 0x502C007u, 0x023C20Bu, 0x0E54511u, 0x2684B1Du, 0x2A8CC00u, 0x3600F06u, 0x421920Cu, 0x4A28190u,
    0x0040496u, 0x0C5879Cu, 0x106089Eu, 0x1C78B85u, 0x2488D89u, 0x2890E8Bu, 0x30A108Fu, 0x3405191u, 0x482C41Bu, 0x4C3451Du,
    0x0E64B0Au, 0x1A7CE10u, 0x228D014u, 0x2E0009Au, 0x320819Cu, 0x4630687u, 0x4A38789u, 0x524898Du, 0x0860C93u, 0x1478F99u,
    0x2498102u, 0x2C04306u, 0x300C408u, 0x3C2470Eu, 0x0A6D001u, 0x127D205u, 0x1684087u, 0x1E9428Bu, 0x229C38Du, 0x2A08591u,
    0x3A28999u, 0x4E50E84u, 0x046918Au, 0x087000Cu, 0x1080210u, 0x20A0618u, 0x2C1491Eu, 0x382CC05u, 0x4444F0Bu, 0x4C5510Fu,
    0x026C195u, 0x0674297u, 0x128C59Du, 0x1A9C782u, 0x2610A88u, 0x2A18B8Au, 0x3E41094u, 0x4249196u, 0x0070401u, 0x0478503u,
    0x0C88707u, 0x1090809u, 0x18A0A0Du, 0x2414D13u, 0x281CE15u, 0x343511Bu, 0x3C44080u, 0x404C182u, 0x485C386u, 0x547468Cu,
    0x0A8C992u, 0x1A08D9Au, 0x2621081u, 0x3238107u, 0x3A4830Bu, 0x4660611u, 0x0080A19u, 0x0890C1Du, 0x0C98D00u, 0x1404F04u,
    0x180D006u, 0x303C392u, 0x3C54698u, 0x446489Cu, 0x0694E89u, 0x120918Fu, 0x1610011u, 0x2228317u, 0x2E4061Du, 0x3650802u,
    0x3A58904u, 0x4268B08u, 0x4E80E0Eu, 0x08A1216u, 0x141429Cu, 0x1C24481u, 0x202C583u, 0x3454A8Du, 0x385CB8Fu, 0x4C85099u,
    0x508D19Bu, 0x029C100u, 0x0600202u, 0x0E10406u, 0x1A2870Cu, 0x1E3080Eu, 0x3258D18u, 0x3660E1Au, 0x3E7101Eu, 0x4A88085u,
    0x040448Du, 0x101C793u, 0x182C997u, 0x1C34A99u, 0x284CD80u, 0x305CF84u, 0x3C7400Au, 0x4C94412u, 0x5400616u, 0x0A1891Cu,
    0x0E20A1Eu, 0x1630C03u, 0x265100Bu, 0x3268091u, 0x3A78295u, 0x4A9869Du, 0x5204882u, 0x081CB88u, 0x0C24C8Au, 0x183CF90u,
    0x2C6421Au, 0x387C501u, 0x4494807u, 0x489C909u, 0x5410C0Fu, 0x0A28F15u, 0x1239119u, 0x164121Bu, 0x2A68486u, 0x2E70588u,
    0x4298A92u, 0x46A0B94u, 0x4E0CD98u, 0x5214E9Au, 0x042509Eu, 0x103C105u, 0x205C50Du, 0x286C711u, 0x2C74813u, 0x409CD1Du,
    0x4C11004u, 0x5019106u, 0x063018Cu, 0x0E40390u, 0x1248492u, 0x1E60798u, 0x3288C83u, 0x420508Bu, 0x4A1400Fu, 0x4E1C111u,
    0x002C315u, 0x1C64A04u, 0x287CD0Au, 0x308CF0Eu, 0x3C01214u, 0x4008096u, 0x481829Au, 0x5430581u, 0x0238683u, 0x1A68C8Fu,
    0x2278E93u, 0x2680F95u, 0x2E91199u, 0x3E0C302u, 0x523480Cu, 0x003C90Eu, 0x084CB12u, 0x0C54C14u, 0x207D11Eu, 0x2485201u,
    0x380848Bu, 0x4420791u, 0x4828893u, 0x5038A97u, 0x0A58E80u, 0x1671186u, 0x1E8010Au, 0x228820Cu, 0x2A98410u, 0x360C716u,
    0x4224A1Cu, 0x462CB1Eu, 0x5244E05u, 0x0455009u, 0x085D10Bu, 0x1C84395u, 0x289C69Bu, 0x3818A84u, 0x4430D8Au, 0x5049090u,
    0x0258014u, 0x127841Cu, 0x1E90703u, 0x26A0907u, 0x3214C0Du, 0x361CD0Fu, 0x3E2CF13u, 0x4A45219u, 0x4E4C09Bu, 0x0464382u,
    0x107C688u, 0x1C9498Eu, 0x2400B92u, 0x3018E98u, 0x3420F9Au, 0x4038001u, 0x4848205u, 0x4C50307u, 0x546050Bu, 0x026860Du,
    0x1690B17u, 0x2E1D104u, 0x3225206u, 0x3A3418Au, 0x3E3C28Cu, 0x5264796u, 0x006C898u, 0x0C84B9Eu, 0x1494D83u, 0x189CE85u,
    0x2009089u, 0x3838415u, 0x3C40517u, 0x5068A02u, 0x5470B04u, 0x129900Eu, 0x2E2C49Cu, 0x363C681u, 0x3A44783u, 0x4254987u,
    0x465CA89u, 0x4E6CC8Du, 0x088D095u, 0x140011Bu, 0x2828606u, 0x2C30708u, 0x344090Cu, 0x4058C12u, 0x4460D14u, 0x0690081u,
    0x0EA0285u, 0x1204387u, 0x1A1458Bu, 0x262C891u, 0x2A34993u, 0x364CC99u, 0x3E5CE9Du, 0x4264F80u, 0x4A75184u, 0x1008612u,
    0x2838C1Eu, 0x3048E03u, 0x3450F05u, 0x3C61109u, 0x487818Fu, 0x4C80291u, 0x0298597u, 0x0A0479Bu, 0x0E0C89Du, 0x161CA82u,
    0x2234D88u, 0x2E4D08Eu, 0x3255190u, 0x3E6C216u, 0x467C41Au, 0x4A8451Cu, 0x009C803u, 0x0808A07u, 0x1420D0Du, 0x2441115u,
    0x2C50099u, 0x3868380u, 0x3C70482u, 0x4480686u, 0x0A14D94u, 0x1224F98u, 0x1E3C01Eu, 0x2244101u, 0x366C60Bu, 0x3A7470Du,
    0x468CA13u, 0x5200D19u, 0x0410F1Du, 0x0819000u, 0x1029204u, 0x1C4028Au, 0x3470896u, 0x3878998u, 0x4088B9Cu, 0x4490C9Eu,
    0x061C00Bu, 0x1A44515u, 0x1E4C617u, 0x265C81Bu, 0x2A6491Du, 0x3E8CE08u, 0x4294F0Au, 0x4E09210u, 0x0018194u, 0x0C3049Au,
    0x1848781u, 0x2460A87u, 0x2868B89u, 0x3480E8Fu, 0x3C91093u, 0x4099195u, 0x4C0C21Bu, 0x541C400u, 0x0A34706u, 0x1A54B0Eu,
    0x266CE14u, 0x2E7D018u, 0x3A9409Eu, 0x4A10487u, 0x1450C97u, 0x1858D99u, 0x2068F9Du, 0x2C80004u, 0x3088106u, 0x3CA040Cu,
    0x4814712u, 0x5024916u, 0x542CA18u, 0x063CC1Cu, 0x165D005u, 0x227408Bu, 0x2A8428Fu, 0x2E8C391u, 0x3A00697u, 0x4E28B82u,
    0x5230C84u, 0x105918Eu, 0x1460010u, 0x1C70214u, 0x2078316u, 0x288851Au, 0x34A0801u, 0x441CC09u, 0x4C2CE0Du, 0x5034F0Fu,
    0x1A74480u, 0x1E7C582u, 0x2A94888u, 0x3200A8Cu, 0x4220E94u, 0x4A31098u, 0x004811Eu, 0x187870Bu, 0x1C8080Du, 0x2490A11u,
    0x2898B13u, 0x3004D17u, 0x4025100u, 0x4C3C186u, 0x544C38Au, 0x0A64690u, 0x0E6C792u, 0x167C996u, 0x2294C9Cu, 0x3211085u,
    0x463830Fu, 0x4A40411u, 0x5250615u, 0x086891Bu, 0x0C70A1Du, 0x1888D04u, 0x2098F08u, 0x24A100Au, 0x2C0D20Eu, 0x3014090u,
    0x443C59Au, 0x484469Cu, 0x066CB87u, 0x0A74C89u, 0x1284E8Du, 0x1E9D193u, 0x2A10219u, 0x2E1831Bu, 0x3A30602u, 0x4648908u,
    0x4E58B0Cu, 0x0470E12u, 0x1089118u, 0x149121Au, 0x2004281u, 0x2C1C587u, 0x383488Du, 0x4044A91u, 0x4C5CD97u, 0x0E8C104u,
    0x1294206u, 0x1E0850Cu, 0x2618710u, 0x3638B18u, 0x4250E1Eu, 0x4A61003u, 0x0078089u, 0x048018Bu, 0x0C9038Fu, 0x2824A9Du,
    0x343CD84u, 0x3C4CF88u, 0x405508Au, 0x547C314u, 0x0284416u, 0x0E9C71Cu, 0x1608901u, 0x1A10A03u, 0x2220C07u, 0x2628D09u,
    0x3A51213u, 0x3E58095u, 0x5280580u, 0x0088682u, 0x0CA0988u, 0x140CB8Cu, 0x242CF94u, 0x304401Au, 0x385421Eu, 0x446C505u,
    0x508480Bu, 0x069CB11u, 0x0A00C13u, 0x1618F19u, 0x1E2911Du, 0x2E48286u, 0x365848Au, 0x4270790u, 0x5290B98u, 0x04A0D9Cu,
    0x0804E9Eu, 0x1015083u, 0x141D185u, 0x1C2C109u, 0x3864817u, 0x4074A1Bu, 0x4C8CD02u, 0x0201008u, 0x0E1808Eu, 0x1220190u,
    0x1E38496u, 0x2A5079Cu, 0x3260981u, 0x3668A83u, 0x4A90F8Du, 0x4E9908Fu, 0x040C115u, 0x0C1C319u, 0x102441Bu, 0x1834600u,
    0x3064C0Cu, 0x346CD0Eu, 0x4895218u, 0x4C9C09Au, 0x0210381u, 0x0A20585u, 0x163888Bu, 0x1A4098Du, 0x2658C93u, 0x3270F99u,
    0x3A8119Du, 0x4698204u, 0x520C50Au, 0x143CB16u, 0x1844C18u, 0x245CF1Eu, 0x3884189u, 0x4800591u, 0x5010795u, 0x5418897u,
    0x0628A9Bu, 0x0A30B9Du, 0x1240D82u, 0x226118Au, 0x3688414u, 0x42A071Au, 0x460481Cu, 0x4E14A01u, 0x0834E09u, 0x144D10Fu,
    0x2064195u, 0x2874399u, 0x2C7C49Bu, 0x348C680u, 0x4000986u, 0x4408A88u, 0x5020D8Eu, 0x0230F92u, 0x125011Au, 0x2678605u,
    0x2A80707u, 0x3E04C11u, 0x4A1CF17u, 0x4E25019u, 0x003521Du, 0x0C4C284u, 0x1054386u, 0x1C6C68Cu, 0x247C890u, 0x2884992u,
    0x3094B96u, 0x3C08E9Cu, 0x4821183u, 0x4C28005u, 0x024030Bu, 0x0E58611u, 0x1A70917u, 0x2280B1Bu, 0x2E98E02u, 0x3E1520Au,
    0x462418Eu, 0x4A2C290u, 0x523C494u, 0x0044596u, 0x085479Au, 0x1874B83u, 0x248CE89u, 0x2C9D08Du, 0x3C18215u, 0x5040700u,
    0x5448802u, 0x0A60B08u, 0x1678E0Eu, 0x1E89012u, 0x2AA0098u, 0x361439Eu, 0x4634787u, 0x4E4498Bu, 0x524CA8Du, 0x045CC91u,
    0x0864D93u, 0x1C8C01Du, 0x2094100u, 0x382070Cu, 0x4030910u, 0x4438A12u, 0x4C48C16u, 0x0260F1Cu, 0x066901Eu, 0x1A90289u,
    0x1E9838Bu, 0x260458Fu, 0x321C895u, 0x423CC9Du, 0x4E54F84u, 0x0065188u, 0x046C00Au, 0x1084310u, 0x1894514u, 0x3428C03u,
    0x3C38E07u, 0x485110Du, 0x4C5920Fu, 0x5468193u, 0x0E8859Bu, 0x1AA0882u, 0x220CA86u, 0x2E24D8Cu, 0x322CE8Eu, 0x3A3D092u,
    0x4654118u, 0x4A5C21Au, 0x0074501u, 0x0C8C807u, 0x2010D11u, 0x2C29017u, 0x3031119u, 0x3C48180u, 0x507068Au, 0x547878Cu,
    0x12A0C96u, 0x1604D98u, 0x2A2C003u, 0x2E34105u, 0x3644309u, 0x3A4C40Bu, 0x425C60Fu, 0x4E74915u, 0x527CA17u, 0x0894D1Du,
    0x1000F02u, 0x1409004u, 0x283028Eu, 0x3448594u, 0x446899Cu, 0x4C78B81u, 0x5080C83u, 0x0698F89u, 0x0E0518Du, 0x1A1C213u,
    0x2A3C61Bu, 0x324C800u, 0x3654902u, 0x3E64B06u, 0x426CC08u, 0x049D214u, 0x101029Au, 0x182049Eu, 0x2438785u, 0x3050A8Bu,
    0x3C68D91u, 0x4070E93u, 0x02A0200u, 0x0A0C404u, 0x0E14506u, 0x162470Au, 0x2644B12u, 0x325CE18u, 0x3E7511Eu, 0x4684083u,
    0x4A8C185u, 0x081068Fu, 0x0C18791u, 0x2040C9Bu, 0x2C58F82u, 0x3061084u, 0x3870008u, 0x448830Eu, 0x4890410u, 0x5404716u,
    0x061491Au, 0x0A1CA1Cu, 0x122CC01u, 0x1E44F07u, 0x2E6408Fu, 0x3A7C395u, 0x428C599u, 0x469469Bu, 0x0418B86u, 0x1030E8Cu,
    0x2050014u, 0x2C6831Au, 0x347851Eu, 0x3880601u, 0x4090805u, 0x500CC0Du, 0x0624F13u, 0x0E35117u, 0x1E54280u, 0x2664484u,
    0x327C78Au, 0x368488Cu, 0x429CB92u, 0x042919Eu, 0x0C38103u, 0x1850409u, 0x1C5850Bu, 0x2870811u, 0x3080A15u, 0x3488B17u,
    0x3C98D1Bu, 0x40A0E1Du, 0x5424088u, 0x1654694u, 0x1A5C796u, 0x226C99Au, 0x2674A9Cu, 0x3A9CF87u, 0x4A1810Fu, 0x5228313u,
    0x0030415u, 0x0840619u, 0x1458900u, 0x2070C06u, 0x2478D08u, 0x38A1212u, 0x3C04094u, 0x481C39Au, 0x1664C8Du, 0x1E74E91u,
    0x227CF93u, 0x2A8D197u, 0x360021Du, 0x4620606u, 0x523890Cu, 0x0448B10u, 0x1060E16u, 0x1468F18u, 0x1C7911Cu, 0x2890183u,
    0x2C98285u, 0x380C58Bu, 0x4424891u, 0x4C34A95u, 0x503CB97u, 0x024CD9Bu, 0x0E65082u, 0x126D184u, 0x269440Eu, 0x2A9C510u,
    0x3208714u, 0x4A38D01u, 0x4E40E03u, 0x107018Fu, 0x1880393u, 0x1C88495u, 0x2498699u, 0x3414A82u, 0x402CD88u, 0x483CF8Cu,
    0x4C4508Eu, 0x5454012u, 0x0A6C318u, 0x168461Eu, 0x1A8C701u, 0x2600A07u, 0x2E10C0Bu, 0x3218D0Du, 0x3E31013u, 0x4641217u,
    0x525829Du, 0x0C78686u, 0x148888Au, 0x189098Cu, 0x2404C92u, 0x2C14E96u, 0x484C305u, 0x505C509u, 0x067480Fu, 0x0A7C911u,
    0x1E00E1Bu, 0x2208F1Du, 0x2E21204u, 0x3A3828Au, 0x424848Eu, 0x4650590u, 0x4E60794u, 0x0478A9Au, 0x0880B9Cu, 0x1C05087u,
    0x200D189u, 0x281C10Du, 0x2C2420Fu, 0x027CD06u, 0x0684E08u, 0x0E9500Cu, 0x129D10Eu, 0x1A08092u, 0x2620398u, 0x2A2849Au,
    0x3640781u, 0x3E50985u, 0x4A68C8Bu, 0x0C98017u, 0x10A0119u, 0x1C14400u, 0x2424604u, 0x282C706u, 0x3C54C10u, 0x486CF16u,
    0x028C09Eu, 0x0A9C283u, 0x0E00385u, 0x1610589u, 0x222888Fu, 0x3248C97u, 0x3E60F9Du, 0x5288208u, 0x009030Au, 0x08A050Eu,
    0x1414814u, 0x181C916u, 0x304CF03u, 0x385D107u, 0x3C65209u, 0x447418Du, 0x508C493u, 0x5494595u, 0x0A0889Bu, 0x1218A80u,
    0x1620B82u, 0x2238E88u, 0x3A68214u, 0x4E9071Eu, 0x5298801u, 0x0404A05u, 0x080CB07u, 0x101CD0Bu, 0x1C35011u, 0x2C54199u,
    0x346439Du, 0x386C480u, 0x407C684u, 0x0208C90u, 0x0610D92u, 0x1229098u, 0x1A3801Cu, 0x1E4011Eu, 0x2A58405u, 0x3268609u,
    0x3E8090Fu, 0x000CF1Bu, 0x0C25202u, 0x102C084u, 0x183C288u, 0x285C690u, 0x3474996u, 0x3C84B9Au, 0x489CE81u, 0x4C00F83u,
    0x5411187u, 0x0A2820Du, 0x0E3030Fu, 0x1A48615u, 0x2E70B00u, 0x3278C02u, 0x3A88E06u, 0x46A110Cu, 0x001C294u, 0x082C498u,
    0x0C3459Au, 0x144479Eu, 0x184C881u, 0x2C74D8Bu, 0x307CE8Du, 0x4400117u, 0x4808219u, 0x501841Du, 0x1248A0Au, 0x1650B0Cu,
    0x2268E12u, 0x2E81118u, 0x369009Cu, 0x4204383u, 0x4E1C689u, 0x522478Bu, 0x083CA91u, 0x104CC95u, 0x1454D97u, 0x206D09Du,
    0x287C002u, 0x3494308u, 0x4410710u, 0x5028A16u, 0x0640D1Cu, 0x0E50F01u, 0x1E70089u, 0x3298593u, 0x3E0C899u, 0x421499Bu,
    0x4A24B80u, 0x003CE86u, 0x105C00Eu, 0x1C74314u, 0x2484518u, 0x288C61Au, 0x4018C07u, 0x4C30F0Du, 0x5441111u, 0x0249213u,
    0x0A58197u, 0x0E60299u, 0x2288784u, 0x2690886u, 0x3A14D90u, 0x3E1CE92u, 0x462D096u, 0x4A35198u, 0x524411Cu, 0x0C64505u,
    0x187C80Bu, 0x208CA0Fu, 0x2494B11u, 0x381901Bu, 0x4430082u, 0x4838184u, 0x545048Au, 0x066068Eu, 0x1680A96u, 0x1E90C9Au,
    0x2A04F81u, 0x3A24109u, 0x423430Du, 0x463C40Fu, 0x4E4C613u, 0x5254715u, 0x0464919u, 0x1484D02u, 0x209D008u, 0x280920Cu,
    0x3420292u, 0x4038598u, 0x4C5089Eu, 0x0670C87u, 0x1288F8Du, 0x1A99191u, 0x1EA0013u, 0x260C217u, 0x322451Du, 0x362C600u,
    0x4244906u, 0x4A54B0Au, 0x4E5CC0Cu, 0x006CE10u, 0x0474F12u, 0x189C19Cu, 0x1C0029Eu, 0x3028789u, 0x343088Bu, 0x4858D95u,
    0x547109Bu, 0x027919Du, 0x0E90204u, 0x16A0408u, 0x1A0450Au, 0x221470Eu, 0x2E2CA14u, 0x3A44D1Au, 0x4A65103u, 0x007C189u,
    0x0C9448Fu, 0x2018999u, 0x3848F86u, 0x3C51088u, 0x446000Cu, 0x486810Eu, 0x5078312u, 0x0A9871Au, 0x1E1CC05u, 0x2A34F0Bu,
    0x2E3D00Du, 0x364D211u, 0x466C399u, 0x5284680u, 0x089C986u, 0x1008B8Au, 0x1410C8Cu, 0x1C20E90u, 0x2839196u, 0x2C40018u,
    0x385831Eu, 0x4068503u, 0x4C80809u, 0x508890Bu, 0x0E0CE15u, 0x1214F17u, 0x263C182u, 0x2A44284u, 0x365C58Au, 0x3E6C78Eu,
    0x4A84A94u, 0x4E8CB96u, 0x0400E9Cu, 0x0C11081u, 0x1828107u, 0x244040Du, 0x3058713u, 0x3460815u, 0x4078B1Bu, 0x4C90E02u,
    0x0205108u, 0x0A1408Cu, 0x162C392u, 0x264C79Au, 0x2E5C99Eu, 0x3264A81u, 0x3A74C85u, 0x3E7CD87u, 0x468CF8Bu, 0x0C20419u,
    0x143061Du, 0x2450A06u, 0x2C60C0Au, 0x3878F10u, 0x3C81012u, 0x4898098u, 0x540C39Eu, 0x061C583u, 0x0A24685u, 0x1E4CB8Fu,
    0x2E6CF97u, 0x367D19Bu, 0x3A8401Du, 0x4294202u, 0x042080Eu, 0x0828910u, 0x1C50E1Au, 0x2058F1Cu, 0x2869101u, 0x2C71203u,
    0x3480187u, 0x409848Du, 0x44A058Fu, 0x062CB9Bu, 0x0E3CD80u, 0x1A55086u, 0x2A7420Eu, 0x3E9C718u, 0x420081Au, 0x4E18B01u,
    0x0028D05u, 0x0C4100Bu, 0x1C60193u, 0x2470397u, 0x308869Du, 0x3490780u, 0x3CA0984u, 0x4C1CD8Cu, 0x0235092u, 0x0A44016u,
    0x165C31Cu, 0x1A6441Eu, 0x2274603u, 0x2E8C909u, 0x3294A0Bu, 0x3E08D11u, 0x4A21017u, 0x523121Bu, 0x0848282u, 0x1460588u,
    0x186868Au, 0x2480990u, 0x3098C96u, 0x3C0CF9Cu, 0x5034207u, 0x543C309u, 0x1264813u, 0x1E7CB19u, 0x2284C1Bu, 0x2A94E00u,
    0x3609106u, 0x3A11208u, 0x462828Eu, 0x4E38492u, 0x5240594u, 0x1068A9Eu, 0x1C80D85u, 0x2088E87u, 0x2CA118Du, 0x340C111u,
    0x3814213u, 0x442C519u, 0x4C3C71Du, 0x0254A04u, 0x1274E0Cu, 0x1A85010u, 0x1E8D112u, 0x269C096u, 0x2A00198u, 0x4230785u,
    0x4E48A8Bu, 0x0058C8Fu, 0x0C70F95u, 0x1079097u, 0x24A0302u, 0x2804404u, 0x4034A10u, 0x4844C14u, 0x545CF1Au, 0x0E7C083u,
    0x1A94389u, 0x220058Du, 0x260868Fu, 0x2E18893u, 0x3220995u, 0x4648E80u, 0x4A50F82u, 0x1490512u, 0x1898614u, 0x2004818u,
    0x2C1CB1Eu, 0x3024C01u, 0x444D10Bu, 0x485520Du, 0x5064191u, 0x067C497u, 0x129479Du, 0x169C880u, 0x2210B86u, 0x2A20D8Au,
    0x2E28E8Cu, 0x4250116u, 0x0888805u, 0x14A0B0Bu, 0x1C0CD0Fu, 0x2014E11u, 0x2825015u, 0x384419Du, 0x445C484u, 0x4C6C688u,
    0x028498Eu, 0x068CA90u, 0x0E9CC94u, 0x1A10F9Au, 0x1E1909Cu, 0x2A30103u, 0x3648409u, 0x426070Fu, 0x4A70913u, 0x0088C19u,
    0x0490D1Bu, 0x1005002u, 0x1815206u, 0x242C28Cu, 0x283438Eu, 0x3C5C898u, 0x548CE85u, 0x0294F87u, 0x0A0118Bu, 0x0E0800Du,
    0x2230517u, 0x2638619u, 0x3250900u, 0x3A60B04u, 0x3E68C06u, 0x4678E0Au, 0x5291110u, 0x0804196u, 0x2034783u, 0x243C885u,
    0x3054B8Bu, 0x3864D8Fu, 0x549C21Du, 0x0608402u, 0x0A10504u, 0x1220708u, 0x162880Au, 0x2E58E16u, 0x3A7111Cu, 0x4280081u,
    0x4E98387u, 0x040C68Du, 0x1024993u, 0x142CA95u, 0x2044D9Bu, 0x2C5D082u, 0x346C006u, 0x3874108u, 0x408430Cu, 0x4C9C612u,
    0x5000714u, 0x0618A1Au, 0x1230D01u, 0x1A40F05u, 0x3270291u, 0x3678393u, 0x4E04980u, 0x0014B84u, 0x041CC86u, 0x0C2CE8Au,
    0x1845190u, 0x1C4C012u, 0x2864318u, 0x307451Cu, 0x347C61Eu, 0x3C8C803u, 0x4800B09u, 0x0220F11u, 0x0E39217u, 0x164819Bu,
    0x1A5029Du, 0x2668584u, 0x2E78788u, 0x3A90A8Eu, 0x4A0CE96u, 0x521D09Au, 0x002519Cu, 0x0834101u, 0x0C3C203u, 0x144C407u,
    0x3084B15u, 0x3894D19u, 0x4811102u, 0x063838Cu, 0x0A4048Eu, 0x1658794u, 0x2A80C9Eu, 0x2E88D81u, 0x3698F85u, 0x420C00Bu,
    0x461410Du, 0x043C617u, 0x0844719u, 0x105491Du, 0x145CA00u, 0x2C8D00Cu, 0x4010296u, 0x4418398u, 0x4C2859Cu, 0x503069Eu,
    0x0240883u, 0x0E58B89u, 0x1E78F91u, 0x2689195u, 0x32A021Bu, 0x3E14502u, 0x4A2C808u, 0x4E3490Au, 0x044CC10u, 0x0C5CE14u,
    0x1064F16u, 0x1C7D21Cu, 0x3000487u, 0x402088Fu, 0x4830A93u, 0x4C38B95u, 0x5448D99u, 0x0A61080u, 0x1A80208u, 0x269850Eu,
    0x3A1CA18u, 0x3E24B1Au, 0x4634D1Eu, 0x524D005u, 0x0055107u, 0x0C6C18Du, 0x1884493u, 0x2094697u, 0x249C799u, 0x2C0899Du,
    0x3820C84u, 0x3C28D86u, 0x5050010u, 0x5458112u, 0x0A70418u, 0x1E98903u, 0x22A0A05u, 0x3624F0Fu, 0x3A2D011u, 0x423D215u,
    0x4644097u, 0x4E5429Bu, 0x148C98Au, 0x1C9CB8Eu, 0x2000C90u, 0x2810E94u, 0x4040201u, 0x4448303u, 0x5060609u, 0x027080Du,
    0x067890Fu, 0x1290C15u, 0x1AA0E19u, 0x2615100u, 0x3E4448Cu, 0x424C58Eu, 0x4E64894u, 0x0074A98u, 0x1094E81u, 0x1C09187u,
    0x241810Bu, 0x3030411u, 0x3438513u, 0x3C48717u, 0x4860A1Du, 0x4C68B00u, 0x0280E06u, 0x1604090u, 0x1A0C192u, 0x221C396u,
    0x2E3469Cu, 0x323C79Eu, 0x3E54A85u, 0x4664C89u, 0x4A6CD8Bu, 0x527CF8Fu, 0x0085091u, 0x140831Bu, 0x2C38908u, 0x3040A0Au,
    0x3850C0Eu, 0x4468F14u, 0x508121Au, 0x548809Cu, 0x0AA0383u, 0x1614689u, 0x1E2488Du, 0x3654E99u, 0x3A5CF9Bu, 0x4674002u,
    0x4E84206u, 0x528C308u, 0x1010812u, 0x1C28B18u, 0x2C48F01u, 0x3459105u, 0x3861207u, 0x447828Du, 0x4C88491u, 0x0604899u,
    0x121CB80u, 0x264508Au, 0x2A4D18Cu, 0x325C110u, 0x3E74416u, 0x0408B05u, 0x0C18D09u, 0x1020E0Bu, 0x2448095u, 0x2850197u,
    0x346849Du, 0x3C78682u, 0x4080784u, 0x4890988u, 0x4C98A8Au, 0x0E25096u, 0x224C301u, 0x2654403u, 0x2E64607u, 0x326C709u,
    0x3A7C90Du, 0x4A9CD15u, 0x001101Bu, 0x0821200u, 0x0C28082u, 0x1438286u, 0x205058Cu, 0x2C68892u, 0x3070994u, 0x3C88C9Au,
    0x4498E9Eu, 0x5414007u, 0x062420Bu, 0x123C511u, 0x225C919u, 0x2A6CB1Du, 0x3684E04u, 0x3A8CF06u, 0x429D10Au, 0x5218292u,
    0x0830598u, 0x104079Cu, 0x1C58A83u, 0x2870D89u, 0x348908Fu, 0x3891191u, 0x4404217u, 0x501C51Du, 0x022C702u, 0x0634804u,
    0x0E44A08u, 0x1A5CD0Eu, 0x2A7D116u, 0x328C09Au, 0x369419Cu, 0x3E00381u, 0x4208483u, 0x003098Du, 0x0438A8Fu, 0x1860F99u,
    0x1C6909Bu, 0x2880102u, 0x3C0460Cu, 0x400C70Eu, 0x4C24A14u, 0x5434C18u, 0x023CD1Au, 0x0A4CF1Eu, 0x1665205u, 0x227C28Bu,
    0x329C693u, 0x3A08897u, 0x3E10999u, 0x4A28C80u, 0x085118Au, 0x1870312u, 0x2488618u, 0x2C9881Cu, 0x30A091Eu, 0x380CB03u,
    0x482CF0Bu, 0x0654195u, 0x126C49Bu, 0x167459Du, 0x1E84782u, 0x2E00B8Au, 0x3A18E90u, 0x4631196u, 0x4E4011Au, 0x524821Cu,
    0x0458401u, 0x1478809u, 0x2090B0Fu, 0x28A0D13u, 0x3415019u, 0x381D11Bu, 0x4C44386u, 0x504C488u, 0x0E74992u, 0x127CA94u,
    0x1A8CC98u, 0x1E94D9Au, 0x2600F9Eu, 0x3218005u, 0x3620107u, 0x423840Du, 0x4A48611u, 0x4E50713u, 0x0C78C1Du, 0x1890F04u,
    0x1C99006u, 0x280C08Cu, 0x403C698u, 0x484C89Cu, 0x5464B83u, 0x0E84F8Bu, 0x169518Fu, 0x1A9C011u, 0x2208215u, 0x2610317u,
    0x2E2051Bu, 0x3E40904u, 0x4A58C0Au, 0x5268E0Eu, 0x0C89216u, 0x149819Au, 0x200C481u, 0x302C889u, 0x3C44B8Fu, 0x4454D93u,
    0x485CE95u, 0x506D099u, 0x0684100u, 0x1600508u, 0x1E1070Cu, 0x221880Eu, 0x2A28A12u, 0x2E30B14u, 0x425901Eu, 0x4661101u,
    0x048838Bu, 0x089048Du, 0x10A0691u, 0x1C14997u, 0x282CC9Du, 0x2C34D80u, 0x405C00Au, 0x446410Cu, 0x4C74310u, 0x028C616u,
    0x1208A1Eu, 0x1E20D05u, 0x2A3900Bu, 0x3650091u, 0x3E60295u, 0x4A7859Bu, 0x0C04B88u, 0x181CE8Eu, 0x1C24F90u, 0x2435194u,
    0x345431Cu, 0x406C603u, 0x487C807u, 0x5494B0Du, 0x029CC0Fu, 0x0A08E13u, 0x1A2921Bu, 0x2640282u, 0x3258588u, 0x3A6878Cu,
    0x4680A92u, 0x5298D98u, 0x00A0E9Au, 0x0C15181u, 0x182C207u, 0x203C40Bu, 0x244450Du, 0x386CA17u, 0x3C74B19u, 0x509D004u,
    0x5401106u, 0x0A1818Cu, 0x1228390u, 0x1E40696u, 0x2248798u, 0x2E60A9Eu, 0x3670C83u, 0x4EA000Fu, 0x0414315u, 0x081C417u,
    0x143471Du, 0x1C44902u, 0x204CA04u, 0x2C64D0Au, 0x3474F0Eu, 0x408D214u, 0x500839Cu, 0x0218581u, 0x0620683u, 0x0E30887u,
    0x1238989u, 0x368001Bu, 0x3E90200u, 0x4A04506u, 0x4E0C608u, 0x001C80Cu, 0x0C34B12u, 0x103CC14u, 0x286D201u, 0x307C185u,
    0x3484287u, 0x4C10893u, 0x0228B99u, 0x0A38D9Du, 0x0E40E80u, 0x1651084u, 0x2E80410u, 0x3288512u, 0x460CA1Cu, 0x5224D03u,
    0x002CE05u, 0x083D009u, 0x145408Fu, 0x185C191u, 0x2474497u, 0x308C79Du, 0x389C982u, 0x4410C88u, 0x5028F8Eu, 0x5431090u,
    0x125831Au, 0x166041Cu, 0x2A88907u, 0x36A0C0Du, 0x461D015u, 0x523409Bu, 0x0444280u, 0x084C382u, 0x105C586u, 0x207C98Eu,
    0x2C94C94u, 0x3400E98u, 0x401919Eu, 0x4420001u, 0x4C30205u, 0x024850Bu, 0x065060Du, 0x1268913u, 0x1E80C19u, 0x2A98F00u,
    0x3205104u, 0x3E1C18Au, 0x422428Cu, 0x4E3C592u, 0x004C796u, 0x0454898u, 0x106CB9Eu, 0x2495089u, 0x3C20415u, 0x483871Bu,
    0x4C4081Du, 0x0A68D08u, 0x0E70E0Au, 0x1A89110u, 0x2298094u, 0x26A0196u, 0x2E0C39Au, 0x3A24681u, 0x463C987u, 0x4A44A89u,
    0x086CF93u, 0x0C75095u, 0x188C11Bu, 0x209C300u, 0x3C30A0Eu, 0x4440C12u, 0x4848D14u, 0x5058F18u, 0x546101Au, 0x067121Eu,
    0x1690387u, 0x220468Du, 0x2A14891u, 0x362CB97u, 0x3A34C99u, 0x4E5D184u, 0x5264006u, 0x087C30Cu, 0x1494612u, 0x1C00816u,
    0x2818B1Cu, 0x3430E03u, 0x3838F05u, 0x445120Bu, 0x4C6018Fu, 0x5068291u, 0x0278495u, 0x1A04A82u, 0x1E0CB84u, 0x363D190u,
    0x3E4C114u, 0x4254216u, 0x4A6441Au, 0x007C701u, 0x0484803u, 0x1808D0Du, 0x1C10E0Fu, 0x2421013u, 0x3038099u, 0x3C50380u,
    0x4058482u, 0x4C70788u, 0x548098Cu, 0x0288A8Eu, 0x0EA0D94u, 0x160CF98u, 0x3244407u, 0x3A5460Bu, 0x3E5C70Du, 0x466C911u,
    0x4A74A13u, 0x5284C17u, 0x1818086u, 0x202828Au, 0x2C40590u, 0x3048692u, 0x4470B9Cu, 0x4878C9Eu, 0x5490F85u, 0x0A0400Bu,
    0x161C311u, 0x1E2C515u, 0x2A4481Bu, 0x2E4C91Du, 0x4274E08u, 0x4E8D10Eu, 0x5295210u, 0x142059Cu, 0x2848A87u, 0x2C50B89u,
    0x3460D8Du, 0x3868E8Fu, 0x4079093u, 0x4C90119u, 0x509821Bu, 0x060C502u, 0x0E1C706u, 0x1A34A0Cu, 0x264CD12u, 0x3265018u,
    0x4284181u, 0x4A94385u, 0x4E9C487u, 0x041078Du, 0x0C20991u, 0x1838C97u, 0x2859080u, 0x3068004u, 0x3470106u, 0x3C8030Au,
    0x0214A18u, 0x0E2CD1Eu, 0x2255209u, 0x2E6C28Fu, 0x3A84595u, 0x3E8C697u, 0x4A0099Du, 0x0018C84u, 0x0828E88u, 0x0C30F8Au,
    0x144118Eu, 0x2058214u, 0x2460316u, 0x307861Cu, 0x3C90903u, 0x4804C09u, 0x062D113u, 0x0A35215u, 0x1E5C480u, 0x2264582u,
    0x2A74786u, 0x2E7C888u, 0x368CA8Cu, 0x4200D92u, 0x522119Au, 0x043011Eu, 0x0838201u, 0x1048405u, 0x2C80B13u, 0x3898E19u,
    0x400501Du, 0x440D100u, 0x5024186u, 0x023438Au, 0x0E4C690u, 0x267CC9Cu, 0x2A84D9Eu, 0x3294F83u, 0x369D085u, 0x3E08009u,
    0x4E28411u, 0x0440717u, 0x0C5091Bu, 0x1C70D04u, 0x2480F08u, 0x309920Eu, 0x34A0090u, 0x4014396u, 0x543C881u, 0x0244983u,
    0x0A54B87u, 0x166CE8Du, 0x1A74F8Fu, 0x268C015u, 0x2E9C219u, 0x320031Bu, 0x3A10500u, 0x3E18602u, 0x5240B0Cu, 0x0048C0Eu,
    0x1471118u, 0x187921Au, 0x208819Eu, 0x2CA0485u, 0x381478Bu, 0x4834B93u, 0x544CE99u, 0x065D09Du, 0x1274104u, 0x1E8C40Au,
    0x229450Cu, 0x2E08812u, 0x3618A16u, 0x3A20B18u, 0x4638E1Eu, 0x0460089u, 0x1480491u, 0x1C90695u, 0x2098797u, 0x341CC82u,
    0x443D08Au, 0x5054110u, 0x0264314u, 0x0E7C61Au, 0x128471Cu, 0x1A94901u, 0x2608C07u, 0x362900Fu, 0x4240095u, 0x4A50299u,
    0x4E5839Bu, 0x0C80886u, 0x1088988u, 0x240CE92u, 0x2814F94u, 0x3025198u, 0x342C01Au, 0x4854505u, 0x4C5C607u, 0x0A84B11u,
    0x0E8CC13u, 0x169CE17u, 0x1A00F19u, 0x221111Du, 0x3230286u, 0x3E4858Cu, 0x4658790u, 0x4A60892u, 0x5270A96u, 0x0888D9Cu,
    0x14A1083u, 0x1805185u, 0x241C20Bu, 0x2C2C40Fu, 0x3C4C817u, 0x445CA1Bu, 0x5074D02u, 0x0A9510Au, 0x120008Eu, 0x1608190u,
    0x2220496u, 0x2A3069Au, 0x3A50A83u, 0x4668D89u, 0x4E78F8Du, 0x0490013u, 0x1004319u, 0x1C1C600u, 0x2024702u, 0x2C3CA08u,
    0x3854D0Eu, 0x4064F12u, 0x446D014u, 0x4C7D218u, 0x029429Eu, 0x069C381u, 0x1210687u, 0x1A2088Bu, 0x1E2898Du, 0x2638B91u,
    0x2A40C93u, 0x3E6919Du, 0x009850Au, 0x04A060Cu, 0x1014912u, 0x1824B16u, 0x243CE1Cu, 0x2844F1Eu, 0x345D205u, 0x3C6C189u,
    0x407428Bu, 0x488448Fu, 0x0A10A9Bu, 0x1A30E84u, 0x2241088u, 0x264918Au, 0x468871Au, 0x0004B03u, 0x0814D07u, 0x0C1CE09u,
    0x142D00Du, 0x183510Fu, 0x2044093u, 0x306449Bu, 0x5000C8Cu, 0x5408D8Eu, 0x0618F92u, 0x163811Au, 0x2E68707u, 0x367890Bu,
    0x3A80A0Du, 0x4290C11u, 0x4E04F17u, 0x520D019u, 0x0824080u, 0x1034284u, 0x1C4C58Au, 0x205468Cu, 0x3884C98u, 0x4C09183u,
    0x5010005u, 0x0220209u, 0x062830Bu, 0x0E3850Fu, 0x1A50815u, 0x1E58917u, 0x2A70C1Du, 0x3280E02u, 0x3688F04u, 0x3E99108u,
    0x4A0C18Eu, 0x0024494u, 0x042C596u, 0x104489Cu, 0x1C5CB83u, 0x2874E89u, 0x308508Du, 0x3C9C113u, 0x4C1851Bu, 0x5428700u,
    0x0A40A06u, 0x0E48B08u, 0x1658D0Cu, 0x2679114u, 0x329019Au, 0x3AA039Eu, 0x4A1C787u, 0x522C98Bu, 0x0844C91u, 0x0C4CD93u,
    0x1865099u, 0x247C100u, 0x2C8C304u, 0x3094406u, 0x380060Au, 0x4418910u, 0x5438D18u, 0x0648F1Cu, 0x0A5101Eu, 0x1261203u,
    0x1668085u, 0x2A9058Fu, 0x2E98691u, 0x421CB9Bu, 0x4624C9Du, 0x4E34E82u, 0x523CF84u, 0x106420Eu, 0x146C310u, 0x289481Au,
    0x2C9C91Cu, 0x3408B01u, 0x4020E07u, 0x504120Fu, 0x0658295u, 0x0E68499u, 0x127059Bu, 0x1E88882u, 0x2698A86u, 0x320CD8Cu,
    0x422D194u, 0x005441Eu, 0x045C501u, 0x0C6C705u, 0x1C8CB0Du, 0x3011017u, 0x3C2809Du, 0x4030180u, 0x4840384u, 0x545868Au,
    0x026078Cu, 0x0E78A92u, 0x1A90D98u, 0x22A0F9Cu, 0x2E14003u, 0x3E3440Bu, 0x4A4C711u, 0x0064A17u, 0x0874C1Bu, 0x0C7CD1Du,
    0x2001208u, 0x240808Au, 0x3830594u, 0x3C38696u, 0x444889Au, 0x485099Cu, 0x5060B81u, 0x0678E87u, 0x0A80F89u, 0x169800Fu,
    0x1E04213u, 0x220C315u, 0x3634800u, 0x424CB06u, 0x4654C08u, 0x526CF0Eu, 0x047D112u, 0x149C29Au, 0x1C0849Eu, 0x2820785u,
    0x3840B8Du, 0x4050D91u, 0x4458E93u, 0x4C69097u, 0x5071199u, 0x12A0506u, 0x1E1480Cu, 0x2624A10u, 0x323CD16u, 0x3644E18u,
    0x3E5501Cu, 0x4A6C083u, 0x1000791u, 0x1810995u, 0x1C18A97u, 0x2428C9Bu, 0x3449084u, 0x406010Au, 0x487030Eu, 0x4C78410u,
    0x5488614u, 0x0290716u, 0x1614C01u, 0x1A1CD03u, 0x2E4520Du, 0x3A5C293u, 0x4674599u, 0x528C880u, 0x0094982u, 0x0C08C88u,
    0x1418E8Cu, 0x1820F8Eu, 0x2031192u, 0x2C48218u, 0x386051Eu, 0x3C68601u, 0x4880907u, 0x5498C0Du, 0x121D117u, 0x1E3419Du,
    0x3A6C88Cu, 0x427CA90u, 0x4684B92u, 0x4E94D96u, 0x081119Eu, 0x1428205u, 0x1C38409u, 0x285070Fu, 0x2C58811u, 0x3468A15u,
    0x4488E1Du, 0x50A1104u, 0x061418Au, 0x122C490u, 0x1A3C694u, 0x265499Au, 0x2A5CA9Cu, 0x3674D83u, 0x3E84F87u, 0x428D089u,
    0x4A9C00Du, 0x4E0010Fu, 0x0C28619u, 0x2458C06u, 0x2860D08u, 0x347900Eu, 0x48A0298u, 0x4C0439Au, 0x021C681u, 0x0A2C885u,
    0x1644B8Bu, 0x225CE91u, 0x2E75197u, 0x327C019u, 0x4600504u, 0x4A08606u, 0x002090Cu, 0x0830B10u, 0x246921Eu, 0x2C78183u,
    0x3080285u, 0x3890489u, 0x3C9858Bu, 0x440478Fu, 0x0A3CE9Du, 0x124D082u, 0x1E64108u, 0x226C20Au, 0x2A7C40Eu, 0x3694714u,
    0x3A9C816u, 0x4610B1Cu, 0x5228E03u, 0x0439007u, 0x0841109u, 0x1C68393u, 0x2070495u, 0x2C8879Bu, 0x3498980u, 0x38A0A82u,
    0x400CC86u, 0x023C012u, 0x0644114u, 0x1E74701u, 0x2684905u, 0x2A8CA07u, 0x329CC0Bu, 0x3E10F11u, 0x4219013u, 0x4E30099u,
    0x0448380u, 0x0C58584u, 0x187088Au, 0x2488B90u, 0x2890C92u, 0x3C1519Cu, 0x401C01Eu, 0x4C34305u, 0x5444509u, 0x0A5C80Fu,
    0x1A7CC17u, 0x228CE1Bu, 0x2694F1Du, 0x2E01102u, 0x3209204u, 0x3A18188u, 0x0050896u, 0x0860A9Au, 0x1478D81u, 0x1880E83u,
    0x2C0410Du, 0x300C20Fu, 0x3C24515u, 0x483C81Bu, 0x504CA00u, 0x5454B02u, 0x0664D06u, 0x127D00Cu, 0x168510Eu, 0x2A08398u,
    0x2E1049Au, 0x3A28781u, 0x5258D8Du, 0x1080017u, 0x1C9831Du, 0x20A0400u, 0x280C604u, 0x342490Au, 0x382CA0Cu, 0x4444D12u,
    0x4C54F16u, 0x026D21Cu, 0x0E84283u, 0x1A9C589u, 0x1E0068Bu, 0x2A18991u, 0x3228B95u, 0x3630C97u, 0x4248F9Du, 0x4A59182u,
    0x0070208u, 0x1090610u, 0x18A0814u, 0x1C04916u, 0x2414B1Au, 0x302CE01u, 0x404D209u, 0x4C6428Fu, 0x0A8C799u, 0x0E9489Bu,
    0x2218D86u, 0x2620E88u, 0x323918Eu, 0x3E50214u, 0x4660418u, 0x527871Eu, 0x0890A05u, 0x0C98B07u, 0x180CE0Du, 0x201D011u,
    0x2425113u, 0x303C199u, 0x4464684u, 0x486C786u, 0x0A9CD92u, 0x1208F96u, 0x1611098u, 0x1E2001Cu, 0x2A38303u, 0x426890Fu,
    0x4670A11u, 0x4E80C15u, 0x100D202u, 0x1414084u, 0x202C38Au, 0x283C58Eu, 0x2C44690u, 0x385C996u, 0x406CB9Au, 0x0E1020Du,
    0x121830Fu, 0x1A28513u, 0x1E30615u, 0x2640819u, 0x3660C02u, 0x4278F08u, 0x4A8910Cu, 0x00A0192u, 0x0404294u, 0x0C14498u,
    0x182C79Eu, 0x1C34881u, 0x284CB87u, 0x407D193u, 0x488C117u, 0x540041Du, 0x0208500u, 0x0E20806u, 0x1630A0Au, 0x2248D10u,
    0x2650E12u, 0x3A7809Cu, 0x3E8019Eu, 0x5204689u, 0x000C78Bu, 0x081C98Fu, 0x1434C95u, 0x204CF9Bu, 0x245509Du, 0x306C104u,
    0x3C8440Au, 0x449460Eu, 0x5008914u, 0x0620C1Au, 0x1641003u, 0x1E51207u, 0x2258089u, 0x2E7038Fu, 0x3680593u, 0x4298899u,
    0x5214C82u, 0x0424E86u, 0x082CF88u, 0x144400Eu, 0x2C7461Au, 0x388C901u, 0x409CB05u, 0x4C10E0Bu, 0x0229111u, 0x0E40197u,
    0x1E60580u, 0x2A78886u, 0x3288A8Au, 0x3690B8Cu, 0x4A15096u, 0x4E1D198u, 0x043421Eu, 0x104C505u, 0x185C709u, 0x1C6480Bu,
    0x308CD15u, 0x3494E17u, 0x4818082u, 0x4C20184u, 0x5430388u, 0x023848Au, 0x0A4868Eu, 0x1A68A96u, 0x2680D9Cu, 0x2E90F81u,
    0x3299083u, 0x3A04007u, 0x461C30Du, 0x003C715u, 0x0C54A1Bu, 0x1464C00u, 0x2485008u, 0x2C9520Cu, 0x3808292u, 0x482869Au,
    0x503889Eu, 0x5440981u, 0x0650B85u, 0x0A58C87u, 0x1268E8Bu, 0x2288013u, 0x2EA0319u, 0x360C51Du, 0x4E3CB0Au, 0x0454E10u,
    0x085CF12u, 0x1475218u, 0x289C483u, 0x2C00585u, 0x3410789u, 0x4028A8Fu, 0x4430B91u, 0x5048E97u, 0x025909Bu, 0x066119Du,
    0x0E70102u, 0x1278204u, 0x26A070Eu, 0x2A04810u, 0x3E2CD1Au, 0x4234E1Cu, 0x4E4D103u, 0x005C087u, 0x0C7438Du, 0x1C94795u,
    0x2400999u, 0x2808A9Bu, 0x3018C80u, 0x3C30F86u, 0x484800Cu, 0x0268414u, 0x0A78618u, 0x0E8071Au, 0x1A98A01u, 0x2E1CF0Bu,
    0x3E3C093u, 0x464C297u, 0x4A54399u, 0x526459Du, 0x087C884u, 0x189CC8Cu, 0x2C21196u, 0x383821Cu, 0x3C4031Eu, 0x4450503u,
    0x547090Bu, 0x0A88C11u, 0x16A0F17u, 0x1E0D11Bu, 0x221521Du, 0x2A24182u, 0x363C488u, 0x3A4458Au, 0x4E6CA94u, 0x0484D9Au,
    0x088CE9Cu, 0x1C10107u, 0x2018209u, 0x3440713u, 0x3848815u, 0x4058A19u, 0x4460B1Bu, 0x4C70D00u, 0x0691108u, 0x120418Eu,
    0x1A14392u, 0x1E1C494u, 0x262C698u, 0x324499Eu, 0x3E5CC85u, 0x4264D87u, 0x4E7D08Du, 0x0494113u, 0x1008419u, 0x181861Du,
    0x2430904u, 0x3450D0Cu, 0x3C60F10u, 0x4069012u, 0x4C80098u, 0x549029Cu, 0x0E0C685u, 0x1A2498Bu, 0x2234B8Fu, 0x3254F97u,
    0x3A6519Bu, 0x467C202u, 0x4A84304u, 0x009C60Au, 0x0C10910u, 0x1420B14u, 0x1828C16u, 0x2038E1Au, 0x2C51101u, 0x3C70289u,
    0x448048Du, 0x488858Fu, 0x5098793u, 0x54A0895u, 0x1224D80u, 0x2A5410Cu, 0x2E5C20Eu, 0x366C412u, 0x3A74514u, 0x4284718u,
    0x4E9CA1Eu, 0x5200B01u, 0x102900Bu, 0x143110Du, 0x1C40091u, 0x3878780u, 0x4490A86u, 0x4CA0C8Au, 0x5004D8Cu, 0x0E2C016u,
    0x1A4431Cu, 0x2A64705u, 0x3274909u, 0x3E8CC0Fu, 0x4294D11u, 0x4A00F15u, 0x042009Du, 0x1038384u, 0x246088Eu, 0x2868990u,
    0x3078B94u, 0x3C90E9Au, 0x4098F9Cu, 0x0224309u, 0x0A3450Du, 0x164C813u, 0x2264B19u, 0x266CC1Bu, 0x3284F02u, 0x3E9D208u,
    0x460818Cu, 0x4A1028Eu, 0x0C4089Au, 0x2068D85u, 0x2470E87u, 0x2C8108Bu, 0x308918Du, 0x3898111u, 0x440C417u, 0x4814519u,
    0x542C800u, 0x063CA04u, 0x0A44B06u, 0x1E6D010u, 0x2A84096u, 0x2E8C198u, 0x3A0049Eu, 0x4210683u, 0x4618785u, 0x5230A8Bu,
    0x0440C8Fu, 0x1058F95u, 0x207811Du, 0x2888302u, 0x34A0608u, 0x380470Au, 0x5034D16u, 0x064D01Cu, 0x0E5D201u, 0x1A74287u,
    0x1E7C389u, 0x268C58Du, 0x3200893u, 0x3608995u, 0x4E38F82u, 0x0049186u, 0x0450008u, 0x0C6020Cu, 0x289891Au, 0x3004B1Eu,
    0x340CC01u, 0x3C1CE05u, 0x4024F07u, 0x544C191u, 0x0254293u, 0x167C79Du, 0x2294A84u, 0x269CB86u, 0x3A21090u, 0x3E29192u,
    0x4A40218u, 0x525041Cu, 0x005851Eu, 0x0868703u, 0x1480A09u, 0x2098D0Fu, 0x24A0E11u, 0x3015117u, 0x382409Bu, 0x3C2C19Du,
    0x5054688u, 0x066C98Eu, 0x168CD96u, 0x220109Cu, 0x2A10001u, 0x2E18103u, 0x3628307u, 0x464870Fu, 0x0470C19u, 0x1088F00u,
    0x1491002u, 0x1CA1206u, 0x281428Cu, 0x2C1C38Eu, 0x3834694u, 0x444C99Au, 0x5064C81u, 0x0274E85u, 0x129400Du, 0x1E08313u,
    0x2618517u, 0x2A20619u, 0x323081Du, 0x3638900u, 0x4A60E0Au, 0x0C90196u, 0x1098298u, 0x180449Cu, 0x1C0C59Eu, 0x3034A89u,
    0x343CB8Bu, 0x4054E91u, 0x4865095u, 0x4C6D197u, 0x0A94402u, 0x1608708u, 0x1A1080Au, 0x2E38D14u, 0x3E5911Cu, 0x4668081u,
    0x0CA078Fu, 0x140C993u, 0x1814A95u, 0x2024C99u, 0x242CD9Bu, 0x2C3CF80u, 0x3C5C108u, 0x487440Eu, 0x5084612u, 0x069C918u,
    0x0A00A1Au, 0x1210C1Eu, 0x1E28F05u, 0x2E4808Du, 0x3A60393u, 0x4270597u, 0x4678699u, 0x4E8889Du, 0x04A0B84u, 0x0804C86u,
    0x141CF8Cu, 0x1C2D190u, 0x2034012u, 0x2844216u, 0x4074803u, 0x447C905u, 0x0608F11u, 0x0E19115u, 0x1A3019Bu, 0x2648482u,
    0x2A50584u, 0x366888Au, 0x3E78A8Eu, 0x4280B90u, 0x4A90D94u, 0x000509Au, 0x0C1C101u, 0x1024203u, 0x1C3C509u, 0x285480Fu,
    0x346CB15u, 0x3C7CD19u, 0x4895000u, 0x0A2038Cu, 0x0E2848Eu, 0x1638692u, 0x1A40794u, 0x2250998u, 0x3E89087u, 0x469800Bu,
    0x520C311u, 0x0014413u, 0x1844A00u, 0x245CD06u, 0x307500Cu, 0x3885210u, 0x3C8C092u, 0x449C296u, 0x501059Cu, 0x541869Eu,
    0x1240B89u, 0x1648C8Bu, 0x1E58E8Fu, 0x2260F91u, 0x3A9031Du, 0x4E14808u, 0x521C90Au, 0x0834C10u, 0x1044E14u, 0x1C5D11Au,
    0x206521Cu, 0x2C7C283u, 0x348C487u, 0x400078Du, 0x4C18A93u, 0x0230D99u, 0x0638E9Bu, 0x1251182u, 0x1A60106u, 0x1E68208u,
    0x2A8050Eu, 0x3290712u, 0x3E04A18u, 0x4E24E01u, 0x0035005u, 0x043D107u, 0x0C4C08Bu, 0x1864391u, 0x349CA80u, 0x4820F8Au,
    0x4C2908Cu, 0x5438010u, 0x0A50316u, 0x0E58418u, 0x1A7071Eu, 0x2688A05u, 0x2E98C09u, 0x32A0D0Bu, 0x4625215u, 0x4A2C097u,
    0x004439Du, 0x0854582u, 0x0C5C684u, 0x2C9CE94u, 0x3000F96u, 0x4428201u, 0x4830303u, 0x5040507u, 0x5448609u, 0x065880Du,
    0x1270B13u, 0x2290F1Bu, 0x2E05202u, 0x3614186u, 0x4E44792u, 0x524C894u, 0x1074D9Eu, 0x147CE81u, 0x2095187u, 0x280010Bu,
    0x3418411u, 0x4C48A1Du, 0x5050B00u, 0x0260D04u, 0x0668E06u, 0x0E7900Au, 0x1E98192u, 0x2A0C498u, 0x321C69Cu, 0x3E34983u,
    0x423CA85u, 0x4A4CC89u, 0x0064F8Fu, 0x046D091u, 0x1084117u, 0x2408602u, 0x2810704u, 0x3020908u, 0x3C38C0Eu, 0x4040D10u,
    0x4C59016u, 0x546921Au, 0x027009Cu, 0x0E88383u, 0x220C88Du, 0x261498Fu, 0x3A3CE99u, 0x4655180u, 0x526C206u, 0x088450Cu,
    0x0C8C60Eu, 0x1800914u, 0x2418C1Au, 0x2C28E1Eu, 0x3841105u, 0x445818Bu, 0x486028Du, 0x5478593u, 0x0688797u, 0x0A90899u,
    0x1604B80u, 0x1E14D84u, 0x2A2D08Au, 0x3A4C212u, 0x425C416u, 0x4664518u, 0x527C81Eu, 0x048CA03u, 0x1408E0Bu, 0x2021111u,
    0x2830095u, 0x344839Bu, 0x385049Du, 0x4C78988u, 0x0698D90u, 0x120D096u, 0x1A1C01Au, 0x324C607u, 0x3654709u, 0x426CA0Fu,
    0x4A7CC13u, 0x4E84D15u, 0x0094F19u, 0x049D01Bu, 0x1820286u, 0x1C28388u, 0x3458994u, 0x3C68B98u, 0x4070C9Au, 0x4880E9Eu,
    0x02A0007u, 0x1624511u, 0x1A2C613u, 0x223C817u, 0x2E54B1Du, 0x3A6CE04u, 0x3E74F06u, 0x4A8D20Cu, 0x529C190u, 0x0C18598u,
    0x142879Cu, 0x3060E8Bu, 0x387108Fu, 0x3C79191u, 0x4488115u, 0x4890217u, 0x50A041Bu, 0x0A1C804u, 0x1634B0Au, 0x1E44D0Eu,
    0x2A5D014u, 0x367409Au, 0x428C381u, 0x4694483u, 0x5208789u, 0x0820A8Fu, 0x1438D95u, 0x1C48F99u, 0x2860000u, 0x2C68102u,
    0x3880408u, 0x409060Cu, 0x4C04912u, 0x500CA14u, 0x0E34F1Eu, 0x123D001u, 0x266428Bu, 0x2A6C38Du, 0x3684693u, 0x3E94897u,
    0x4A08B9Du, 0x4E10C80u, 0x0428F86u, 0x0C3918Au, 0x104000Cu, 0x1850210u, 0x2468516u, 0x308081Cu, 0x40A0C05u, 0x480CE09u,
    0x4C14F0Bu, 0x022D211u, 0x165449Bu, 0x2674884u, 0x2E84A88u, 0x328CB8Au, 0x3A9CD8Eu, 0x3E00E90u, 0x003021Cu, 0x1458707u,
    0x2070A0Du, 0x2C88D13u, 0x3C0511Bu, 0x481C182u, 0x5434488u, 0x064468Cu, 0x0A4C78Eu, 0x125C992u, 0x1E74C98u, 0x227CD9Au,
    0x2E95081u, 0x421830Bu, 0x462040Du, 0x0448917u, 0x0850A19u, 0x1C78F04u, 0x2081006u, 0x289120Au, 0x2C9808Cu, 0x3404290u,
    0x401C596u, 0x4424698u, 0x503C99Eu, 0x024CB83u, 0x0654C85u, 0x0E64E89u, 0x1A7D18Fu, 0x2A9C317u, 0x361061Du, 0x4228904u,
    0x4E40C0Au, 0x0050E0Eu, 0x0C69114u, 0x1C8829Cu, 0x2498481u, 0x28A0583u, 0x300C787u, 0x3414889u, 0x3C24A8Du, 0x4C44E95u,
    0x025D19Bu, 0x0A6C100u, 0x1A8C508u, 0x229C70Cu, 0x2E10A12u, 0x3218B14u, 0x3E30E1Au, 0x5258085u, 0x0060187u, 0x087038Bu,
    0x1488691u, 0x2404A99u, 0x2C14C9Du, 0x301CD80u, 0x382CF84u, 0x3C35086u, 0x505C310u, 0x5464412u, 0x128C91Cu, 0x1694A1Eu,
    0x1E00C03u, 0x2208D05u, 0x2A18F09u, 0x363120Fu, 0x4E6059Bu, 0x526869Du, 0x0478882u, 0x1090B88u, 0x200CF90u, 0x2C24016u,
    0x343421Au, 0x383C31Cu, 0x4454603u, 0x027CB0Du, 0x129CF15u, 0x1A09119u, 0x2620180u, 0x3238486u, 0x425888Eu, 0x4E70B94u,
    0x0080D98u, 0x0C9909Eu, 0x10A1181u, 0x180C105u, 0x242440Bu, 0x282C50Du, 0x3444813u, 0x405CB19u, 0x486CD1Du, 0x5485004u,
    0x0A9C08Au, 0x0E0018Cu, 0x2630798u, 0x2E4099Cu, 0x3248A9Eu, 0x4670F89u, 0x4A7908Bu, 0x08A0315u, 0x0C04417u, 0x141461Bu,
    0x181C71Du, 0x202C902u, 0x304CD0Au, 0x3C65010u, 0x4475214u, 0x487C096u, 0x0600581u, 0x1218887u, 0x1620989u, 0x2238C8Fu,
    0x2A48E93u, 0x2E50F95u, 0x3A6801Bu, 0x4278200u, 0x4E90506u, 0x080C90Eu, 0x101CB12u, 0x1424C14u, 0x203CF1Au, 0x386C287u,
    0x448458Du, 0x4C94791u, 0x0208A97u, 0x0610B99u, 0x0E20D9Du, 0x1A39084u, 0x1E41186u, 0x3670512u, 0x3E80716u, 0x4288818u,
    0x4A98A1Cu, 0x0414E05u, 0x102D10Bu, 0x183C08Fu, 0x1C44191u, 0x2454395u, 0x285C497u, 0x3C84982u, 0x5410F8Eu, 0x0A28014u,
    0x0E30116u, 0x164031Au, 0x2258601u, 0x2660703u, 0x3278A09u, 0x3A88C0Du, 0x3E90D0Fu, 0x46A0F13u, 0x082C280u, 0x0C34382u,
    0x184C688u, 0x205C88Cu, 0x246498Eu, 0x388CE98u, 0x440119Eu, 0x5420307u, 0x0A3860Du, 0x1248811u, 0x1650913u, 0x1E60B17u,
    0x2E80F00u, 0x3A99206u, 0x4E1C490u, 0x5224592u, 0x0434796u, 0x104CA9Cu, 0x1454B9Eu, 0x2C8518Bu, 0x389C211u, 0x4008415u,
    0x4C2071Bu, 0x502881Du, 0x0640B04u, 0x0E50D08u, 0x1258E0Au, 0x1A6900Eu, 0x1E71110u, 0x4A24987u, 0x4E2CA89u, 0x003CC8Du,
    0x0444D8Fu, 0x186C019u, 0x1C7411Bu, 0x288C402u, 0x309C606u, 0x3400708u, 0x3C1090Cu, 0x4828C12u, 0x5440F18u, 0x024901Au,
    0x1670285u, 0x1A78387u, 0x269068Du, 0x2EA0891u, 0x4A34F80u, 0x5245184u, 0x085C20Au, 0x0C6430Cu, 0x1474510u, 0x2494918u,
    0x3008C1Eu, 0x3818E03u, 0x4431109u, 0x483920Bu, 0x504818Fu, 0x0660495u, 0x0A68597u, 0x168089Du, 0x2298B84u, 0x2A04D88u,
    0x2E0CE8Au, 0x361D08Eu, 0x4234114u, 0x525451Cu, 0x0464701u, 0x086C803u, 0x107CA07u, 0x2809013u, 0x2C11115u, 0x4440482u,
    0x4C50686u, 0x5058788u, 0x0E80C92u, 0x1288D94u, 0x1EA109Au, 0x260C01Eu, 0x2A14101u, 0x3224305u, 0x3E3C60Bu, 0x4A54911u,
    0x4E5CA13u, 0x0474D19u, 0x0C84F1Du, 0x108D000u, 0x1C00086u, 0x241028Au, 0x3028590u, 0x4048998u, 0x4C60C9Eu, 0x5470E83u,
    0x0278F85u, 0x0A89189u, 0x2E2C81Bu, 0x3A44B02u, 0x3E4CC04u, 0x527510Eu, 0x007D210u, 0x0C94296u, 0x180859Cu, 0x2018781u,
    0x2420883u, 0x2C30A87u, 0x3C50E8Fu, 0x5078119u, 0x548021Bu, 0x0690400u, 0x0A98502u, 0x2224B0Eu, 0x364D018u, 0x3A5511Au,
    0x426409Eu, 0x466C181u, 0x4E7C385u, 0x049468Bu, 0x089C78Du, 0x1410A93u, 0x1C20C97u, 0x3450004u, 0x406830Au, 0x447040Cu,
    0x5088712u, 0x0298916u, 0x1214D1Eu, 0x1A24F03u, 0x263D209u, 0x365C391u, 0x3E6C595u, 0x4274697u, 0x4A8489Bu, 0x009CB82u,
    0x1018F8Au, 0x1C30010u, 0x305851Au, 0x346061Cu, 0x3C70801u, 0x4888B07u, 0x0204F0Fu, 0x0E1D215u, 0x162C199u, 0x1A3429Bu,
    0x2244480u, 0x2E5C786u, 0x3264888u, 0x3E7CB8Eu, 0x468CD92u, 0x4A94E94u, 0x000919Au, 0x1430405u, 0x1838507u, 0x2C60A11u,
    0x3068B13u, 0x3878D17u, 0x449101Du, 0x5004084u, 0x0A2448Cu, 0x1234690u, 0x163C792u, 0x1E4C996u, 0x367CF83u, 0x3A85085u,
    0x469C10Bu, 0x5210411u, 0x0828717u, 0x103891Bu, 0x1C50C02u, 0x348120Eu, 0x3888090u, 0x4098294u, 0x44A0396u, 0x4C0C59Au,
    0x062C983u, 0x1244C89u, 0x1A54E8Du, 0x266D193u, 0x2A74015u, 0x3284219u, 0x4200602u, 0x4E18908u, 0x0C40E12u, 0x1048F14u,
    0x1859118u, 0x247019Eu, 0x2878281u, 0x3490587u, 0x3CA078Bu, 0x400488Du, 0x4814A91u, 0x4C1CB93u, 0x0A4509Du, 0x0E4D180u,
    0x227440Au, 0x267C50Cu, 0x3A00A16u, 0x4618D1Cu, 0x4A20E1Eu, 0x0039105u, 0x146038Fu, 0x2078695u, 0x2C9099Bu, 0x3098A9Du,
    0x3C0CD84u, 0x441CF88u, 0x482508Au, 0x543C110u, 0x064C314u, 0x126461Au, 0x2284A03u, 0x2A94C07u, 0x2E9CD09u, 0x3A1100Fu,
    0x4221213u, 0x0858682u, 0x1068886u, 0x1C80B8Cu, 0x2088C8Eu, 0x2898E92u, 0x340D198u, 0x442C301u, 0x5044607u, 0x025480Bu,
    0x065C90Du, 0x1A84E17u, 0x1E8CF19u, 0x2A01200u, 0x3210184u, 0x3618286u, 0x3E2848Au, 0x0058A96u, 0x0460B98u, 0x1889083u,
    0x1C91185u, 0x24A0109u, 0x280420Bu, 0x301440Fu, 0x4034817u, 0x4C4CB1Du, 0x0264E04u, 0x0A75008u, 0x168C08Eu, 0x2200394u,
    0x2608496u, 0x3A30981u, 0x4A50D89u, 0x5260F8Du, 0x0878013u, 0x189841Bu, 0x2004600u, 0x240C702u, 0x2C1C906u, 0x3024A08u,
    0x3834C0Cu, 0x4855014u, 0x546C09Au, 0x067C29Eu, 0x1294585u, 0x1E0888Bu, 0x2A20B91u, 0x2E28C93u, 0x3A40F99u, 0x4658000u,
    0x4E68204u, 0x5270306u, 0x048050Au, 0x1098810u, 0x14A0912u, 0x2014C18u, 0x2824E1Cu, 0x2C2CF1Eu, 0x3845205u, 0x4C6C48Fu,
    0x5074591u, 0x0E9CA9Bu, 0x1E18E84u, 0x2629088u, 0x324010Eu, 0x3648210u, 0x4260516u, 0x4A7071Au, 0x4E7881Cu, 0x0088A01u,
    0x0CA0D07u, 0x181500Du, 0x2834195u, 0x3044399u, 0x344C49Bu, 0x4064782u, 0x548CC8Cu, 0x0E09094u, 0x1618018u, 0x1A2011Au,
    0x223031Eu, 0x2638401u, 0x2E48605u, 0x3E68A0Du, 0x5290F17u, 0x080521Du, 0x0C0C080u, 0x243C68Cu, 0x3054992u, 0x3C6CC98u,
    0x447CE9Cu, 0x5095183u, 0x0608209u, 0x0A1030Bu, 0x1628611u, 0x1E38815u, 0x2A50B1Bu, 0x2E58C1Du, 0x4281108u, 0x468920Au,
    0x0814596u, 0x102479Au, 0x142C89Cu, 0x1C3CA81u, 0x2854D87u, 0x2C5CE89u, 0x4084113u, 0x448C215u, 0x4C9C419u, 0x0210700u,
    0x0E28A06u, 0x1230B08u, 0x1E48E0Eu, 0x2A61114u, 0x367819Au, 0x3E8839Eu, 0x041CA8Du, 0x0C2CC91u, 0x1034D93u, 0x1844F97u,
    0x1C4D099u, 0x245C01Du, 0x347C406u, 0x409470Cu, 0x4800910u, 0x0220D18u, 0x0A30F1Cu, 0x1649203u, 0x1A50085u, 0x266838Bu,
    0x3280691u, 0x3E98997u, 0x4604B9Bu, 0x521CE82u, 0x0C3C00Au, 0x144C20Eu, 0x2064514u, 0x246C616u, 0x3894B01u, 0x3C9CC03u,
    0x502110Du, 0x542920Fu, 0x0638193u, 0x0A40295u, 0x1250499u, 0x1E68780u, 0x2270882u, 0x3698D8Cu, 0x3AA0E8Eu, 0x420D092u,
    0x4E24118u, 0x145C807u, 0x1C6CA0Bu, 0x2074B0Du, 0x2C8CE13u, 0x349D017u, 0x401009Du, 0x5030486u, 0x024068Au, 0x0E58990u,
    0x1260A92u, 0x2A9109Eu, 0x3604105u, 0x3E14309u, 0x4A2C60Fu, 0x0044915u, 0x0C5CC1Bu, 0x1064D1Du, 0x1C7D004u, 0x289408Au,
    0x300028Eu, 0x3C18594u, 0x483089Au, 0x4C3899Cu, 0x0250C83u, 0x0E68F89u, 0x167918Du, 0x1A8000Fu, 0x2E04519u, 0x320C61Bu,
    0x4634B06u, 0x4A3CC08u, 0x524CE0Cu, 0x0054F0Eu, 0x0865112u, 0x147C198u, 0x188429Au, 0x249C581u, 0x2C08785u, 0x3010887u,
    0x4438D91u, 0x5459199u, 0x0A70200u, 0x1280404u, 0x1688506u, 0x22A080Cu, 0x2A0CA10u, 0x3624D16u, 0x464511Eu, 0x4E54083u,
    0x525C185u, 0x046C389u, 0x087448Bu, 0x2000A97u, 0x2C18D9Du, 0x3428F82u, 0x444810Au, 0x4C5830Eu, 0x0270614u, 0x0678716u,
    0x2614F07u, 0x2A1D009u, 0x322D20Du, 0x424C395u, 0x4E6469Bu, 0x0074880u, 0x047C982u, 0x0C8CB86u, 0x1094C88u, 0x2419192u,
    0x2820014u, 0x3C4851Eu, 0x4860805u, 0x4C68907u, 0x5478B0Bu, 0x0A90E11u, 0x1A0D219u, 0x221C19Du, 0x2624280u, 0x2E34484u,
    0x3A4C78Au, 0x4664A90u, 0x4A6CB92u, 0x0084E98u, 0x089509Cu, 0x0C9D19Eu, 0x2C3870Fu, 0x3C58B17u, 0x4870E1Du, 0x5081002u,
    0x0698088u, 0x1614490u, 0x222C796u, 0x2A3C99Au, 0x3654C81u, 0x3A5CD83u, 0x426CF87u, 0x4E8400Du, 0x528C10Fu, 0x0800415u,
    0x141871Bu, 0x2030A02u, 0x2840C06u, 0x3458F0Cu, 0x386100Eu, 0x4C88298u, 0x509039Au, 0x02A059Eu, 0x0604681u, 0x1A2CB8Bu,
    0x325D197u, 0x3664019u, 0x3E7421Du, 0x427C300u, 0x040890Cu, 0x1020C12u, 0x1830E16u, 0x1C38F18u, 0x244911Cu, 0x3060183u,
    0x3C78489u, 0x408058Bu, 0x5404A95u, 0x020CB97u, 0x0E24E9Du, 0x1635082u, 0x326C510u, 0x3A7C714u, 0x3E84816u, 0x4A9CB1Cu,
    0x5208D01u, 0x0C29109u, 0x184018Fu, 0x2050393u, 0x2C68699u, 0x307079Bu, 0x3880980u, 0x4498C86u, 0x48A0D88u, 0x541508Eu,
    0x0A2C114u, 0x123C318u, 0x164441Au, 0x1E5461Eu, 0x2A6C905u, 0x2E74A07u, 0x3A8CD0Du, 0x429CF11u, 0x4601013u, 0x4E11217u,
    0x1040584u, 0x2C78C92u, 0x3488E96u, 0x3890F98u, 0x40A119Cu, 0x4C14203u, 0x501C305u, 0x063460Bu, 0x0E4480Fu, 0x124C911u,
    0x1A5CB15u, 0x328D102u, 0x3695204u, 0x420828Au, 0x4A1848Eu, 0x4E20590u, 0x0C48A9Au, 0x1860D81u, 0x2881189u, 0x309010Du,
    0x349820Fu, 0x3C04413u, 0x400C515u, 0x481C719u, 0x0E54E08u, 0x227C092u, 0x2684194u, 0x3A0869Eu, 0x3E10781u, 0x0040D8Du,
    0x0850F91u, 0x0C59093u, 0x1468017u, 0x208031Du, 0x2488400u, 0x380C90Au, 0x3C14A0Cu, 0x4424C10u, 0x482CD12u, 0x0A5C09Eu,
    0x1E84589u, 0x228C68Bu, 0x2A9C88Fu, 0x2E00991u, 0x3610B95u, 0x4228E9Bu, 0x4630F9Du, 0x5248004u, 0x0458208u, 0x107050Eu,
    0x1C88814u, 0x28A0B1Au, 0x2C04C1Cu, 0x381CF03u, 0x402D107u, 0x4435209u, 0x504C28Fu, 0x025C493u, 0x0E74799u, 0x1E94B82u,
    0x2600D86u, 0x321908Cu, 0x3E30112u, 0x4E5051Au, 0x0468801u, 0x1890D0Bu, 0x1C98E0Du, 0x2405011u, 0x301C097u, 0x3424199u,
    0x403C480u, 0x4C54786u, 0x546498Au, 0x026CA8Cu, 0x0A7CC90u, 0x1694F96u, 0x261011Eu, 0x2E20303u, 0x3228405u, 0x3E4070Bu,
    0x5268C15u, 0x0070D17u, 0x1499202u, 0x18A0084u, 0x200C288u, 0x241438Au, 0x383C894u, 0x485CC9Cu, 0x506CE81u, 0x5474F83u,
    0x0685187u, 0x1E10513u, 0x2218615u, 0x2E3091Bu, 0x3640B00u, 0x3A48C02u, 0x4660F08u, 0x4E7110Cu, 0x0488192u, 0x201C881u,
    0x282CA85u, 0x2C34B87u, 0x3444D8Bu, 0x4465193u, 0x028C41Du, 0x0E00704u, 0x1208806u, 0x1A18A0Au, 0x2630D10u, 0x2A38E12u,
    0x3651118u, 0x4A78383u, 0x4E80485u, 0x0090689u, 0x100CA91u, 0x1C24D97u, 0x2434F9Bu, 0x283D09Du, 0x304C002u, 0x3454104u,
    0x487C60Eu, 0x4C84710u, 0x0A08C1Au, 0x0E10D1Cu, 0x1620F01u, 0x2239207u, 0x2E5028Du, 0x325838Fu, 0x3E70695u, 0x4A8899Bu,
    0x080CE86u, 0x142518Cu, 0x182C00Eu, 0x2444314u, 0x2C54518u, 0x3C74901u, 0x4484B05u, 0x509CE0Bu, 0x0A19213u, 0x1228197u,
    0x1630299u, 0x2248580u, 0x2A58784u, 0x3A78B8Cu, 0x4690E92u, 0x4EA1096u, 0x041411Cu, 0x081C21Eu, 0x102C403u, 0x1C44709u,
    0x2C64B11u, 0x387CE17u, 0x408D01Bu, 0x449511Du, 0x0218388u, 0x062048Au, 0x1238790u, 0x1A48994u, 0x1E50A96u, 0x2660C9Au,
    0x2A68D9Cu, 0x3E90007u, 0x4298109u, 0x001C613u, 0x0424715u, 0x0C34919u, 0x184CC00u, 0x286D008u, 0x348408Eu, 0x3C94292u,
    0x409C394u, 0x4808598u, 0x542089Eu, 0x0A38B85u, 0x0E40C87u, 0x1A58F8Du, 0x3288319u, 0x3A9851Du, 0x460C804u, 0x083CE10u,
    0x0C44F12u, 0x1455116u, 0x185D218u, 0x206C19Cu, 0x308C585u, 0x3C0088Bu, 0x4410A8Fu, 0x5028D95u, 0x064109Bu, 0x1660204u,
    0x227850Au, 0x2E90810u, 0x36A0A14u, 0x3A04B16u, 0x4214D1Au, 0x4E2D001u, 0x5235103u, 0x084C189u, 0x105C38Du, 0x146448Fu,
    0x1C74693u, 0x207C795u, 0x3400C80u, 0x3808D82u, 0x4C3000Cu, 0x503810Eu, 0x0650414u, 0x0E60618u, 0x1A7891Eu, 0x1E80A01u,
    0x2A98D07u, 0x3204F0Bu, 0x3E1D211u, 0x4A34297u, 0x004C59Du, 0x106C986u, 0x187CB8Au, 0x1C84C8Cu, 0x289CF92u, 0x3C2021Cu,
    0x4C40605u, 0x5450809u, 0x025890Bu, 0x0A68B0Fu, 0x0E70C11u, 0x1680E15u, 0x3A24488u, 0x463C78Eu, 0x4A44890u, 0x5254A94u,
    0x0C74E9Cu, 0x188D183u, 0x2400209u, 0x2C1040Du, 0x301850Fu, 0x4440A19u, 0x4848B1Bu, 0x5460E02u, 0x0671006u, 0x128808Cu,
    0x2A14698u, 0x2E1C79Au, 0x4244C85u, 0x464CD87u, 0x4E5CF8Bu, 0x526508Du, 0x0474011u, 0x108C317u, 0x1494419u, 0x2008700u,
    0x2C20A06u, 0x3430C0Au, 0x4048F10u, 0x4C61216u, 0x5068098u, 0x1298685u, 0x1E0C98Bu, 0x261CB8Fu, 0x3234E95u, 0x425401Du,
    0x4A64202u, 0x4E6C304u, 0x007C508u, 0x048460Au, 0x0C9480Eu, 0x1C10C16u, 0x2828F1Cu, 0x3039101u, 0x4058289u, 0x486848Du,
    0x5480793u, 0x0288895u, 0x0EA0B9Bu, 0x1A14E82u, 0x2225086u, 0x262D188u, 0x2E3C10Cu, 0x3A54412u, 0x4A7481Au, 0x5284A1Eu,
    0x008CB01u, 0x0C00E07u, 0x2028091u, 0x2430193u, 0x385869Du, 0x4470984u, 0x4878A86u, 0x5088C8Au, 0x06A0F90u, 0x0A05092u,
    0x1E2C31Cu, 0x223441Eu, 0x2A44603u, 0x365C909u, 0x467CD11u, 0x5295017u, 0x040121Bu, 0x080809Du, 0x1420384u, 0x1C30588u,
    0x284888Eu, 0x3868C96u, 0x4078E9Au, 0x4C91181u, 0x5098003u, 0x0204207u, 0x122460Fu, 0x1E3C915u, 0x264CB19u, 0x3264E00u,
    0x366CF02u, 0x4A9418Cu, 0x4E9C28Eu, 0x0410594u, 0x102889Au, 0x1838A9Eu, 0x2450D85u, 0x306908Bu, 0x347118Du, 0x4088213u,
    0x4CA0519u, 0x540C71Du, 0x0214800u, 0x163CD0Au, 0x1A44E0Cu, 0x3274198u, 0x3A8439Cu, 0x3E8C49Eu, 0x469C683u, 0x5210989u,
    0x0018A8Bu, 0x1440F95u, 0x1849097u, 0x2C70302u, 0x3888608u, 0x3C9070Au, 0x4804A10u, 0x5014C14u, 0x541CD16u, 0x0A3501Cu,
    0x1245201u, 0x2E7C68Fu, 0x368C893u, 0x3A94995u, 0x4200B99u, 0x4608C9Bu, 0x0838008u, 0x145030Eu, 0x1C60512u, 0x2878818u,
    0x2C8091Au, 0x3490B1Eu, 0x4004E05u, 0x440CF07u, 0x063C293u, 0x1254599u, 0x1A6479Du, 0x2A84B86u, 0x369CE8Cu, 0x3E09090u,
    0x4A20116u, 0x4E28218u, 0x0C50703u, 0x1058805u, 0x2480D0Fu, 0x3099015u, 0x34A1117u, 0x3C0C09Bu, 0x4824382u, 0x4C2C484u,
    0x024478Au, 0x0A5498Eu, 0x0E5CA90u, 0x166CC94u, 0x2284F9Au, 0x2E9C001u, 0x3E18409u, 0x462860Du, 0x4A3070Fu, 0x0858C19u,
    0x1470F00u, 0x2490088u, 0x300438Eu, 0x3814592u, 0x3C1C694u, 0x544CC81u, 0x0A64F87u, 0x127518Bu, 0x1E8C211u, 0x2A00517u,
    0x361881Du, 0x3A20900u, 0x4638C06u, 0x5250F0Cu, 0x0869212u, 0x1078196u, 0x1C9049Cu, 0x209859Eu, 0x2C0C885u, 0x3824B8Bu,
    0x4034D8Fu, 0x443CE91u, 0x026411Bu, 0x1A94708u, 0x1E9C80Au, 0x2608A0Eu, 0x2A10B10u, 0x3E3901Au, 0x424111Cu, 0x4E58183u,
    0x0068387u, 0x0470489u, 0x0C8068Du, 0x1898993u, 0x2814D9Bu, 0x3C3C006u, 0x4044108u, 0x4C5C40Eu, 0x546C612u, 0x1A00D01u,
    0x2210F05u, 0x2619007u, 0x2E2920Bu, 0x323008Du, 0x3A40291u, 0x4A60699u, 0x0078980u, 0x0888B84u, 0x1804F8Cu, 0x2015190u,
    0x2C2C216u, 0x3034318u, 0x3C4C61Eu, 0x5074B09u, 0x547CC0Bu, 0x068CE0Fu, 0x1201115u, 0x1609217u, 0x222029Du, 0x2A30482u,
    0x2E38584u, 0x3648788u, 0x4E78D94u, 0x5280E96u, 0x140C203u, 0x1C1C407u, 0x2024509u, 0x283470Du, 0x344CA13u, 0x446CE1Bu,
    0x4C7D000u, 0x5085102u, 0x0294086u, 0x0E0838Cu, 0x1A20692u, 0x1E28794u, 0x2A40A9Au, 0x3250C9Eu, 0x3658D81u, 0x4271087u,
    0x0098311u, 0x1014719u, 0x182491Du, 0x1C2CA00u, 0x243CC04u, 0x3054F0Au, 0x4C8C398u, 0x549C59Cu, 0x0A10883u, 0x0E18985u,
    0x2240E8Fu, 0x2648F91u, 0x3260017u, 0x3E7831Du, 0x4688502u, 0x4A90604u, 0x52A0808u, 0x0814B0Eu, 0x0C1CC10u, 0x204511Au,
    0x244D21Cu, 0x2C5C181u, 0x3064283u, 0x489488Fu, 0x0618D99u, 0x0A20E9Bu, 0x1231080u, 0x1639182u, 0x1E48106u, 0x2E6850Eu,
    0x3A80814u, 0x4290A18u, 0x4E04D1Eu, 0x041D005u, 0x103408Bu, 0x143C18Du, 0x2054493u, 0x2864697u, 0x2C6C799u, 0x3884A80u,
    0x4094C84u, 0x4C08F8Au, 0x0628112u, 0x0E38316u, 0x1240418u, 0x2668903u, 0x3688D0Bu, 0x42A1011u, 0x002429Bu, 0x042C39Du,
    0x0C3C582u, 0x1854888u, 0x1C5C98Au, 0x2874C90u, 0x348CF96u, 0x3C9D19Au, 0x400001Cu, 0x4810201u, 0x5428507u, 0x0230609u,
    0x0E4890Fu, 0x1658B13u, 0x1A60C15u, 0x2678F1Bu, 0x3AA0186u, 0x522C792u, 0x0034894u, 0x0844A98u, 0x0C4CB9Au, 0x145CD9Eu,
    0x2075085u, 0x309420Du, 0x3800411u, 0x3C08513u, 0x4418717u, 0x0648D04u, 0x0A50E06u, 0x166910Cu, 0x1E78090u, 0x2280192u,
    0x360469Cu, 0x421C983u, 0x044CF8Fu, 0x0855091u, 0x1064015u, 0x146C117u, 0x1C7C31Bu, 0x2C9C704u, 0x3810A0Au, 0x4C38F14u,
    0x5041016u, 0x025121Au, 0x0E68281u, 0x1270383u, 0x320CB93u, 0x3614C95u, 0x3E24E99u, 0x4A3D180u, 0x4E44002u, 0x045C308u,
    0x0C6C50Cu, 0x107460Eu, 0x1884812u, 0x1C8C914u, 0x3418F01u, 0x484018Bu, 0x4C4828Du, 0x5458491u, 0x0A70797u, 0x1688A9Du,
    0x1A90B80u, 0x2604E86u, 0x321D18Cu, 0x3A2C110u, 0x4644416u, 0x525C71Cu, 0x006481Eu, 0x0C7CB05u, 0x148CD09u, 0x1894E0Bu,
    0x2409111u, 0x2C18095u, 0x383039Bu, 0x4850784u, 0x5060988u, 0x0A80D90u, 0x1290F94u, 0x220C11Cu, 0x2E24403u, 0x3634607u,
    0x424C90Du, 0x4654A0Fu, 0x4E64C13u, 0x047CF19u, 0x149C082u, 0x2010388u, 0x282058Cu, 0x2C2868Eu, 0x4050B98u, 0x5070F81u,
    0x0281185u, 0x0688007u, 0x0E9820Bu, 0x12A030Du, 0x2624817u, 0x2A2C919u, 0x3E54E04u, 0x425CF06u, 0x4A6D10Au, 0x4E7520Cu,
    0x1000598u, 0x1C1889Eu, 0x2428A83u, 0x2830B85u, 0x3040D89u, 0x3C5908Fu, 0x4870115u, 0x4C78217u, 0x029051Du, 0x0AA0702u,
    0x1A1CB0Au, 0x222CD0Eu, 0x2E45014u, 0x3E6419Cu, 0x4A7C483u, 0x528C687u, 0x0094789u, 0x080098Du, 0x1820D95u, 0x2C48000u,
    0x3860306u, 0x447860Cu, 0x5090912u, 0x5498A14u, 0x0A0CD1Au, 0x1625001u, 0x1E35205u, 0x223C087u, 0x2A4C28Bu, 0x3A6C693u,
    0x4684999u, 0x4E94B9Du, 0x529CC80u, 0x0408E84u, 0x0810F86u, 0x1C38210u, 0x2040312u, 0x346881Cu, 0x387091Eu, 0x4488C05u,
    0x4C98E09u, 0x020D10Fu, 0x0615211u, 0x122C297u, 0x1A3C49Bu, 0x1E4459Du, 0x326CA88u, 0x3E84D8Eu, 0x4E01196u, 0x001011Au,
    0x1030503u, 0x2458A0Du, 0x3478E15u, 0x3C89019u, 0x409111Bu, 0x48A0080u, 0x4C04182u, 0x5414386u, 0x0E3478Eu, 0x225CC98u,
    0x2E74F9Eu, 0x327D081u, 0x3A8C005u, 0x0020713u, 0x0C38A19u, 0x1448C1Du, 0x1850D00u, 0x2060F04u, 0x2C7920Au, 0x308008Cu,
    0x3C98392u, 0x4404596u, 0x501C89Cu, 0x542499Eu, 0x124CE89u, 0x1654F8Bu, 0x2A7C215u, 0x2E84317u, 0x369451Bu, 0x4208802u,
    0x4E20B08u, 0x5228C0Au, 0x0840F10u, 0x1051114u, 0x1459216u, 0x1C6819Au, 0x2880481u, 0x3498787u, 0x38A0889u, 0x4414B8Fu,
    0x502CE95u, 0x064519Bu, 0x0E54100u, 0x1A6C406u, 0x329CA12u, 0x3600B14u, 0x3E10D18u, 0x4218E1Au, 0x4A2901Eu, 0x0448187u,
    0x106048Du, 0x1870691u, 0x2890A99u, 0x30A0C9Du, 0x401D086u, 0x4C3410Cu, 0x024C412u, 0x0A5C616u, 0x0E64718u, 0x167491Cu,
    0x228CC03u, 0x320900Bu, 0x3A1920Fu, 0x3E20091u, 0x4630295u, 0x4A38397u, 0x0860882u, 0x0C68984u, 0x2090E8Eu, 0x2498F90u,
    0x300C016u, 0x381C21Au, 0x4434501u, 0x483C603u, 0x0664B0Du, 0x127CE13u, 0x1E95119u, 0x2E10282u, 0x3A28588u, 0x423878Cu,
    0x464088Eu, 0x5258B94u, 0x0468D98u, 0x108109Eu, 0x20A0207u, 0x280C40Bu, 0x3424711u, 0x382C813u, 0x403CA17u, 0x0675106u,
    0x0E8408Au, 0x1A9C390u, 0x1E00492u, 0x2610696u, 0x322899Cu, 0x3630A9Eu, 0x4248D85u, 0x4E6108Bu, 0x007000Fu, 0x18A061Bu,
    0x1C0471Du, 0x281CA04u, 0x3434D0Au, 0x3C44F0Eu, 0x547429Au, 0x027C39Cu, 0x1600887u, 0x1A08989u, 0x2218B8Du, 0x2620C8Fu,
    0x2E30E93u, 0x3A49199u, 0x3E5001Bu, 0x4A68302u, 0x0080608u, 0x1404B12u, 0x201CE18u, 0x2424F1Au, 0x384C185u, 0x3C54287u,
    0x486C58Du, 0x507C791u, 0x0694A97u, 0x1610E80u, 0x1E21084u, 0x2229186u, 0x2A3810Au, 0x2E4020Cu, 0x4670818u, 0x5288B1Eu,
    0x0498D03u, 0x100D009u, 0x141510Bu, 0x1C2408Fu, 0x283C395u, 0x2C44497u, 0x4474A84u, 0x4C84C88u, 0x508CD8Au, 0x029CF8Eu,
    0x1218116u, 0x1E3041Cu, 0x2640601u, 0x2A48703u, 0x3660A09u, 0x4A88F13u, 0x4E91015u, 0x0C14280u, 0x182C586u, 0x1C34688u,
    0x244488Cu, 0x305CB92u, 0x3464C94u, 0x407CF9Au, 0x488D19Eu, 0x4C94001u, 0x5400205u, 0x0A1850Bu, 0x1630811u, 0x1A38913u,
    0x2650C19u, 0x2E60E1Du, 0x3268F00u, 0x469018Au, 0x5204490u, 0x0C24898u, 0x183CB9Eu, 0x204CD83u, 0x2454E85u, 0x2C65089u,
    0x3C84211u, 0x489C517u, 0x500871Bu, 0x0620A02u, 0x0A28B04u, 0x1E5100Eu, 0x2259110u, 0x2E70196u, 0x3A8849Cu, 0x4E0C987u,
    0x0424C8Du, 0x082CD8Fu, 0x1445095u, 0x1C54019u, 0x205C11Bu, 0x286C300u, 0x2C74402u, 0x409C90Cu, 0x063101Au, 0x0E4121Eu,
    0x1248081u, 0x267058Bu, 0x2A7868Du, 0x3EA0B97u, 0x4204C99u, 0x4A14E9Du, 0x002D184u, 0x0C4420Au, 0x104C30Cu, 0x2474816u,
    0x287C918u, 0x3494C1Eu, 0x3C00E03u, 0x0238291u, 0x0A48495u, 0x0E50597u, 0x166079Bu, 0x1A6889Du, 0x2278A82u, 0x3298E8Au,
    0x3E0D190u, 0x461C114u, 0x523441Au, 0x003C51Cu, 0x084C701u, 0x1464A07u, 0x186CB09u, 0x2484E0Fu, 0x309D115u, 0x3C1019Bu,
    0x4420380u, 0x5038686u, 0x5440788u, 0x0A58A8Eu, 0x1268C92u, 0x1E80F98u, 0x360C305u, 0x3A14407u, 0x5244A13u, 0x0454C17u,
    0x085CD19u, 0x106CF1Du, 0x1C85204u, 0x208C086u, 0x2C0038Cu, 0x3410590u, 0x3818692u, 0x4028896u, 0x4C40B9Cu, 0x0258E83u,
    0x127800Bu, 0x1A8820Fu, 0x1E90311u, 0x2A04617u, 0x321481Bu, 0x3E2CB02u, 0x4E4CF0Au, 0x005D10Eu, 0x0465210u, 0x0C74194u,
    0x107C296u, 0x3420B89u, 0x3C30D8Du, 0x4849093u, 0x0A78400u, 0x0E80502u, 0x1A98808u, 0x260CB0Eu, 0x2E1CD12u, 0x3224E14u,
    0x3A35018u, 0x464C09Eu, 0x4A54181u, 0x0C8478Du, 0x1494991u, 0x189CA93u, 0x3029080u, 0x445030Au, 0x485840Cu, 0x5068610u,
    0x5470712u, 0x0680916u, 0x1298C1Cu, 0x16A0D1Eu, 0x2215005u, 0x2A25209u, 0x363C28Fu, 0x4254595u, 0x527499Du, 0x088CC84u,
    0x109CE88u, 0x1400F8Au, 0x2018010u, 0x2828214u, 0x344051Au, 0x4460903u, 0x4C70B07u, 0x5078C09u, 0x0288E0Du, 0x0EA1113u,
    0x1E1C29Bu, 0x2A34582u, 0x4264B8Eu, 0x4A74D92u, 0x008D098u, 0x049519Au, 0x1008201u, 0x243070Bu, 0x283880Du, 0x3048A11u,
    0x3C60D17u, 0x4068E19u, 0x4C81100u, 0x5490084u, 0x0298186u, 0x0E0C48Cu, 0x2234996u, 0x263CA98u, 0x3A64F83u, 0x3E6D085u,
    0x467C009u, 0x4A8410Bu, 0x529430Fu, 0x0808615u, 0x1828A1Du, 0x2038C02u, 0x2440D04u, 0x2C50F08u, 0x4480294u, 0x4888396u,
    0x54A069Cu, 0x060C881u, 0x0A14983u, 0x162CC89u, 0x2A55193u, 0x4284500u, 0x468C602u, 0x4E9C806u, 0x0410B0Cu, 0x1430F14u,
    0x204921Au, 0x285819Eu, 0x3470485u, 0x3878587u, 0x408878Bu, 0x4CA0A91u, 0x5004B93u, 0x061CE99u, 0x1A44104u, 0x1E4C206u,
    0x265C40Au, 0x3274710u, 0x367C812u, 0x4A00D1Cu, 0x4E08E1Eu, 0x0019003u, 0x0421105u, 0x184838Fu, 0x1C50491u, 0x307899Bu,
    0x3480A9Du, 0x3C90C82u, 0x4804F88u, 0x0224110u, 0x0E3C416u, 0x1A5471Cu, 0x2264901u, 0x2E7CC07u, 0x3A94F0Du, 0x3E9D00Fu,
    0x4A10095u, 0x5220299u, 0x002839Bu, 0x0C40682u, 0x1450886u, 0x2068B8Cu, 0x3088F94u, 0x3899198u, 0x3CA001Au, 0x4814301u,
    0x5024505u, 0x0A4490Du, 0x165CC13u, 0x1E6CE17u, 0x2A8511Du, 0x2E8D200u, 0x369C184u, 0x421048Au, 0x5230892u, 0x0848B98u,
    0x1058D9Cu, 0x1460E9Eu, 0x2888109u, 0x2C9020Bu, 0x3804511u, 0x4014715u, 0x441C817u, 0x4C2CA1Bu, 0x5034B1Du, 0x0E5D008u,
    0x268C394u, 0x2A94496u, 0x320069Au, 0x360879Cu, 0x3E18981u, 0x4E38D89u, 0x045108Fu, 0x0C60013u, 0x1068115u, 0x1878319u,
    0x3004906u, 0x340CA08u, 0x4024D0Eu, 0x4834F12u, 0x0A6429Eu, 0x167C585u, 0x269C98Du, 0x2E08B91u, 0x3210C93u, 0x3A20E97u,
    0x3E28F99u, 0x463919Du, 0x0058306u, 0x0C7060Cu, 0x2098B16u, 0x2C0CE1Cu, 0x3825103u, 0x3C2D205u, 0x545C591u, 0x066C795u,
    0x0A74897u, 0x1284A9Bu, 0x1E9CD82u, 0x2200E84u, 0x2E1918Au, 0x362810Eu, 0x3A30210u, 0x4240414u, 0x4648516u, 0x0878B03u,
    0x1CA100Du, 0x200510Fu, 0x2C1C195u, 0x342C399u, 0x4044680u, 0x444C782u, 0x5064A88u, 0x0274C8Cu, 0x067CD8Eu, 0x0E8CF92u,
    0x1A00018u, 0x261831Eu, 0x3638707u, 0x3E4890Bu, 0x4250A0Du, 0x4E68D13u, 0x0C9121Du, 0x1C0C386u, 0x241C58Au, 0x3034890u,
    0x343C992u, 0x3C4CB96u, 0x4C6CF9Eu, 0x0A94209u, 0x160850Fu, 0x1A10611u, 0x2220815u, 0x3240C1Du, 0x3E58F04u, 0x4A7120Au,
    0x528018Eu, 0x0088290u, 0x0898494u, 0x140C79Au, 0x242CB83u, 0x2C3CD87u, 0x385508Du, 0x3C5D18Fu, 0x5084419u, 0x548C51Bu,
    0x1210A06u, 0x1618B08u, 0x1E28D0Cu, 0x2230E0Eu, 0x3658098u, 0x3A6019Au, 0x4678481u, 0x4E88685u, 0x5290787u, 0x04A098Bu,
    0x1014C91u, 0x1C2CF97u, 0x2035099u, 0x2C4C100u, 0x3864406u, 0x447C70Cu, 0x4C8C910u, 0x0200C16u, 0x122101Eu, 0x1E38085u,
    0x2648289u, 0x2A5038Bu, 0x326058Fu, 0x4280997u, 0x0004E82u, 0x102400Au, 0x183420Eu, 0x244C514u, 0x2854616u, 0x346C91Cu,
    0x4084C03u, 0x4894E07u, 0x4C9CF09u, 0x540910Du, 0x1A4059Bu, 0x2250780u, 0x2658882u, 0x2E68A86u, 0x3270B88u, 0x4699092u,
    0x4AA1194u, 0x082441Eu, 0x0C2C501u, 0x143C705u, 0x1844807u, 0x2054A0Bu, 0x2C6CD11u, 0x3074E13u, 0x449C09Du, 0x4800180u,
    0x062868Au, 0x1648A92u, 0x2260D98u, 0x2A70F9Cu, 0x3A90105u, 0x42A0309u, 0x4E1460Fu, 0x0834A17u, 0x1044C1Bu, 0x1C5CF02u,
    0x2065004u, 0x2875208u, 0x3894390u, 0x4408696u, 0x4C1889Au, 0x0230B81u, 0x0638C83u, 0x0E48E87u, 0x1A6118Du, 0x2A80315u,
    0x369861Bu, 0x3E04800u, 0x4A1CB06u, 0x0034E0Cu, 0x043CF0Eu, 0x1055214u, 0x1C6C29Au, 0x247C49Eu, 0x2884581u, 0x3C08A8Bu,
    0x4010B8Du, 0x5439097u, 0x0241199u, 0x0A5011Du, 0x1668404u, 0x228070Au, 0x268880Cu, 0x32A0B12u, 0x3A0CD16u, 0x3E14E18u,
    0x523C083u, 0x0854389u, 0x0C5C48Bu, 0x1874791u, 0x248CA97u, 0x3000D9Du, 0x3810F82u, 0x4428008u, 0x0658614u, 0x0A60716u,
    0x127091Au, 0x1678A1Cu, 0x2E05009u, 0x3A1C08Fu, 0x422C293u, 0x4E44599u, 0x524C69Bu, 0x045C880u, 0x147CC88u, 0x2C08014u,
    0x3418218u, 0x382031Au, 0x403051Eu, 0x5050907u, 0x0668C0Du, 0x0E78E11u, 0x1280F13u, 0x1A91117u, 0x1E99219u, 0x321C484u,
    0x3624586u, 0x4A4CA90u, 0x046CE98u, 0x0C7D09Cu, 0x1894103u, 0x1C9C205u, 0x281050Bu, 0x302070Fu, 0x3C38A15u, 0x4850D1Bu,
    0x5469002u, 0x0271104u, 0x0E8818Au, 0x169838Eu, 0x1AA0490u, 0x2E2499Au, 0x3A3CC81u, 0x4A5D089u, 0x007410Fu, 0x0884313u,
    0x0C8C415u, 0x149C619u, 0x3030D08u, 0x3840F0Cu, 0x4459212u, 0x4860094u, 0x5070298u, 0x068859Eu, 0x0A90681u, 0x1604987u,
    0x221CC8Du, 0x2E34F93u, 0x425C21Du, 0x4664300u, 0x527C606u, 0x048C80Au, 0x089490Cu, 0x1000B10u, 0x283111Cu, 0x4060489u,
    0x446858Bu, 0x4C7878Fu, 0x5080891u, 0x0E04D9Bu, 0x120CE9Du, 0x1E25184u, 0x2A3C20Au, 0x324C40Eu, 0x3E64714u, 0x4A7CA1Au,
    0x4E84B1Cu, 0x0C09007u, 0x1011109u, 0x1C2818Fu, 0x2438393u, 0x4070A82u, 0x4880C86u, 0x4C88D88u, 0x5498F8Cu, 0x02A108Eu,
    0x0A0C012u, 0x1A2C41Au, 0x2644701u, 0x2E54905u, 0x3A6CC0Bu, 0x3E74D0Du, 0x4684F11u, 0x529D217u, 0x0000099u, 0x0C18380u,
    0x1830686u, 0x204088Au, 0x244898Cu, 0x2C58B90u, 0x3870E96u, 0x3C78F98u, 0x489001Eu, 0x50A0203u, 0x5404305u, 0x1E44B15u
};
__constant uchar SKR_MR2[5860] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24,
    9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 0, 21, 11, 1, 22, 12, 2, 23, 13, 3, 24, 14, 4, 25, 15, 5, 26, 16, 6, 27, 17, 7, 28, 18, 8, 29, 19,
    9, 30, 20, 10, 0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27, 4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 0, 25, 19, 13, 7,
    1, 26, 20, 14, 8, 2, 27, 21, 15, 9, 3, 28, 22, 16, 10, 4, 29, 23, 17, 11, 5, 30, 24, 18, 12, 6, 0, 26, 21, 16, 11, 6, 1, 27, 22, 17, 12, 7, 2, 28,
    23, 18, 13, 8, 3, 29, 24, 19, 14, 9, 4, 30, 25, 20, 15, 10, 5, 0, 9, 18, 27, 5, 14, 23, 1, 10, 19, 28, 6, 15, 24, 2, 11, 20, 29, 7, 16, 25, 3, 12,
    21, 30, 8, 17, 26, 4, 13, 22, 0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 0,
    7, 14, 21, 28, 4, 11, 18, 25, 1, 8, 15, 22, 29, 5, 12, 19, 26, 2, 9, 16, 23, 30, 6, 13, 20, 27, 3, 10, 17, 24, 0, 28, 25, 22, 19, 16, 13, 10, 7, 4,
    1, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0, 17, 3, 20, 6, 23, 9, 26, 12, 29, 15, 1, 18, 4, 21, 7, 24, 10, 27,
    13, 30, 16, 2, 19, 5, 22, 8, 25, 11, 28, 14, 0, 13, 26, 8, 21, 3, 16, 29, 11, 24, 6, 19, 1, 14, 27, 9, 22, 4, 17, 30, 12, 25, 7, 20, 2, 15, 28, 10,
    23, 5, 18, 0, 12, 24, 5, 17, 29, 10, 22, 3, 15, 27, 8, 20, 1, 13, 25, 6, 18, 30, 11, 23, 4, 16, 28, 9, 21, 2, 14, 26, 7, 19, 0, 20, 9, 29, 18, 7,
    27, 16, 5, 25, 14, 3, 23, 12, 1, 21, 10, 30, 19, 8, 28, 17, 6, 26, 15, 4, 24, 13, 2, 22, 11, 0, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3,
    1, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9, 11, 13, 15,
    17, 19, 21, 23, 25, 27, 29, 0, 11, 22, 2, 13, 24, 4, 15, 26, 6, 17, 28, 8, 19, 30, 10, 21, 1, 12, 23, 3, 14, 25, 5, 16, 27, 7, 18, 29, 9, 20, 0, 19,
    7, 26, 14, 2, 21, 9, 28, 16, 4, 23, 11, 30, 18, 6, 25, 13, 1, 20, 8, 27, 15, 3, 22, 10, 29, 17, 5, 24, 12, 0, 18, 5, 23, 10, 28, 15, 2, 20, 7, 25,
    12, 30, 17, 4, 22, 9, 27, 14, 1, 19, 6, 24, 11, 29, 16, 3, 21, 8, 26, 13, 0, 14, 28, 11, 25, 8, 22, 5, 19, 2, 16, 30, 13, 27, 10, 24, 7, 21, 4, 18,
    1, 15, 29, 12, 26, 9, 23, 6, 20, 3, 17, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 1, 4, 7, 10, 13, 16, 19, 22,
    25, 28, 0, 24, 17, 10, 3, 27, 20, 13, 6, 30, 23, 16, 9, 2, 26, 19, 12, 5, 29, 22, 15, 8, 1, 25, 18, 11, 4, 28, 21, 14, 7, 0, 27, 23, 19, 15, 11, 7,
    3, 30, 26, 22, 18, 14, 10, 6, 2, 29, 25, 21, 17, 13, 9, 5, 1, 28, 24, 20, 16, 12, 8, 4, 0, 22, 13, 4, 26, 17, 8, 30, 21, 12, 3, 25, 16, 7, 29, 20,
    11, 2, 24, 15, 6, 28, 19, 10, 1, 23, 14, 5, 27, 18, 9, 0, 5, 10, 15, 20, 25, 30, 4, 9, 14, 19, 24, 29, 3, 8, 13, 18, 23, 28, 2, 7, 12, 17, 22, 27,
    1, 6, 11, 16, 21, 26, 0, 6, 12, 18, 24, 30, 5, 11, 17, 23, 29, 4, 10, 16, 22, 28, 3, 9, 15, 21, 27, 2, 8, 14, 20, 26, 1, 7, 13, 19, 25, 0, 23, 15,
    7, 30, 22, 14, 6, 29, 21, 13, 5, 28, 20, 12, 4, 27, 19, 11, 3, 26, 18, 10, 2, 25, 17, 9, 1, 24, 16, 8, 0, 10, 20, 30, 9, 19, 29, 8, 18, 28, 7, 17,
    27, 6, 16, 26, 5, 15, 25, 4, 14, 24, 3, 13, 23, 2, 12, 22, 1, 11, 21, 0, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21,
    5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 0, 19, 1, 20, 2,
    21, 3, 22, 4, 23, 5, 24, 6, 25, 7, 26, 8, 27, 9, 28, 10, 29, 11, 30, 12, 31, 13, 32, 14, 33, 15, 34, 16, 35, 17, 36, 18, 0, 25, 13, 1, 26, 14, 2, 27,
    15, 3, 28, 16, 4, 29, 17, 5, 30, 18, 6, 31, 19, 7, 32, 20, 8, 33, 21, 9, 34, 22, 10, 35, 23, 11, 36, 24, 12, 0, 28, 19, 10, 1, 29, 20, 11, 2, 30, 21,
    12, 3, 31, 22, 13, 4, 32, 23, 14, 5, 33, 24, 15, 6, 34, 25, 16, 7, 35, 26, 17, 8, 36, 27, 18, 9, 0, 15, 30, 8, 23, 1, 16, 31, 9, 24, 2, 17, 32, 10,
    25, 3, 18, 33, 11, 26, 4, 19, 34, 12, 27, 5, 20, 35, 13, 28, 6, 21, 36, 14, 29, 7, 22, 0, 31, 25, 19, 13, 7, 1, 32, 26, 20, 14, 8, 2, 33, 27, 21, 15,
    9, 3, 34, 28, 22, 16, 10, 4, 35, 29, 23, 17, 11, 5, 36, 30, 24, 18, 12, 6, 0, 16, 32, 11, 27, 6, 22, 1, 17, 33, 12, 28, 7, 23, 2, 18, 34, 13, 29, 8,
    24, 3, 19, 35, 14, 30, 9, 25, 4, 20, 36, 15, 31, 10, 26, 5, 21, 0, 14, 28, 5, 19, 33, 10, 24, 1, 15, 29, 6, 20, 34, 11, 25, 2, 16, 30, 7, 21, 35, 12,
    26, 3, 17, 31, 8, 22, 36, 13, 27, 4, 18, 32, 9, 23, 0, 33, 29, 25, 21, 17, 13, 9, 5, 1, 34, 30, 26, 22, 18, 14, 10, 6, 2, 35, 31, 27, 23, 19, 15, 11,
    7, 3, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0, 26, 15, 4, 30, 19, 8, 34, 23, 12, 1, 27, 16, 5, 31, 20, 9, 35, 24, 13, 2, 28, 17, 6, 32, 21, 10, 36, 25,
    14, 3, 29, 18, 7, 33, 22, 11, 0, 27, 17, 7, 34, 24, 14, 4, 31, 21, 11, 1, 28, 18, 8, 35, 25, 15, 5, 32, 22, 12, 2, 29, 19, 9, 36, 26, 16, 6, 33, 23,
    13, 3, 30, 20, 10, 0, 34, 31, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1, 35, 32, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 36, 33, 30, 27, 24, 21, 18, 15, 12, 9,
    6, 3, 0, 20, 3, 23, 6, 26, 9, 29, 12, 32, 15, 35, 18, 1, 21, 4, 24, 7, 27, 10, 30, 13, 33, 16, 36, 19, 2, 22, 5, 25, 8, 28, 11, 31, 14, 34, 17, 0,
    8, 16, 24, 32, 3, 11, 19, 27, 35, 6, 14, 22, 30, 1, 9, 17, 25, 33, 4, 12, 20, 28, 36, 7, 15, 23, 31, 2, 10, 18, 26, 34, 5, 13, 21, 29, 0, 5, 10, 15,
    20, 25, 30, 35, 3, 8, 13, 18, 23, 28, 33, 1, 6, 11, 16, 21, 26, 31, 36, 4, 9, 14, 19, 24, 29, 34, 2, 7, 12, 17, 22, 27, 32, 0, 7, 14, 21, 28, 35, 5,
    12, 19, 26, 33, 3, 10, 17, 24, 31, 1, 8, 15, 22, 29, 36, 6, 13, 20, 27, 34, 4, 11, 18, 25, 32, 2, 9, 16, 23, 30, 0, 24, 11, 35, 22, 9, 33, 20, 7, 31,
    18, 5, 29, 16, 3, 27, 14, 1, 25, 12, 36, 23, 10, 34, 21, 8, 32, 19, 6, 30, 17, 4, 28, 15, 2, 26, 13, 0, 35, 33, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13,
    11, 9, 7, 5, 3, 1, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    32, 34, 36, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 0, 13, 26, 2, 15, 28, 4, 17, 30, 6, 19, 32, 8, 21, 34, 10, 23, 36, 12,
    25, 1, 14, 27, 3, 16, 29, 5, 18, 31, 7, 20, 33, 9, 22, 35, 11, 24, 0, 30, 23, 16, 9, 2, 32, 25, 18, 11, 4, 34, 27, 20, 13, 6, 36, 29, 22, 15, 8, 1,
    31, 24, 17, 10, 3, 33, 26, 19, 12, 5, 35, 28, 21, 14, 7, 0, 32, 27, 22, 17, 12, 7, 2, 34, 29, 24, 19, 14, 9, 4, 36, 31, 26, 21, 16, 11, 6, 1, 33, 28,
    23, 18, 13, 8, 3, 35, 30, 25, 20, 15, 10, 5, 0, 29, 21, 13, 5, 34, 26, 18, 10, 2, 31, 23, 15, 7, 36, 28, 20, 12, 4, 33, 25, 17, 9, 1, 30, 22, 14, 6,
    35, 27, 19, 11, 3, 32, 24, 16, 8, 0, 17, 34, 14, 31, 11, 28, 8, 25, 5, 22, 2, 19, 36, 16, 33, 13, 30, 10, 27, 7, 24, 4, 21, 1, 18, 35, 15, 32, 12, 29,
    9, 26, 6, 23, 3, 20, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 1, 4, 7, 10, 13, 16, 19, 22, 25,
    28, 31, 34, 0, 10, 20, 30, 3, 13, 23, 33, 6, 16, 26, 36, 9, 19, 29, 2, 12, 22, 32, 5, 15, 25, 35, 8, 18, 28, 1, 11, 21, 31, 4, 14, 24, 34, 7, 17, 27,
    0, 11, 22, 33, 7, 18, 29, 3, 14, 25, 36, 10, 21, 32, 6, 17, 28, 2, 13, 24, 35, 9, 20, 31, 5, 16, 27, 1, 12, 23, 34, 8, 19, 30, 4, 15, 26, 0, 4, 8,
    12, 16, 20, 24, 28, 32, 36, 3, 7, 11, 15, 19, 23, 27, 31, 35, 2, 6, 10, 14, 18, 22, 26, 30, 34, 1, 5, 9, 13, 17, 21, 25, 29, 33, 0, 23, 9, 32, 18, 4,
    27, 13, 36, 22, 8, 31, 17, 3, 26, 12, 35, 21, 7, 30, 16, 2, 25, 11, 34, 20, 6, 29, 15, 1, 24, 10, 33, 19, 5, 28, 14, 0, 21, 5, 26, 10, 31, 15, 36, 20,
    4, 25, 9, 30, 14, 35, 19, 3, 24, 8, 29, 13, 34, 18, 2, 23, 7, 28, 12, 33, 17, 1, 22, 6, 27, 11, 32, 16, 0, 6, 12, 18, 24, 30, 36, 5, 11, 17, 23, 29,
    35, 4, 10, 16, 22, 28, 34, 3, 9, 15, 21, 27, 33, 2, 8, 14, 20, 26, 32, 1, 7, 13, 19, 25, 31, 0, 22, 7, 29, 14, 36, 21, 6, 28, 13, 35, 20, 5, 27, 12,
    34, 19, 4, 26, 11, 33, 18, 3, 25, 10, 32, 17, 2, 24, 9, 31, 16, 1, 23, 8, 30, 15, 0, 9, 18, 27, 36, 8, 17, 26, 35, 7, 16, 25, 34, 6, 15, 24, 33, 5,
    14, 23, 32, 4, 13, 22, 31, 3, 12, 21, 30, 2, 11, 20, 29, 1, 10, 19, 28, 0, 12, 24, 36, 11, 23, 35, 10, 22, 34, 9, 21, 33, 8, 20, 32, 7, 19, 31, 6, 18,
    30, 5, 17, 29, 4, 16, 28, 3, 15, 27, 2, 14, 26, 1, 13, 25, 0, 18, 36, 17, 35, 16, 34, 15, 33, 14, 32, 13, 31, 12, 30, 11, 29, 10, 28, 9, 27, 8, 26, 7,
    25, 6, 24, 5, 23, 4, 22, 3, 21, 2, 20, 1, 19, 0, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
    10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 0, 21, 1, 22, 2, 23, 3, 24, 4, 25, 5, 26, 6, 27, 7, 28, 8, 29, 9, 30, 10, 31, 11, 32, 12, 33, 13, 34,
    14, 35, 15, 36, 16, 37, 17, 38, 18, 39, 19, 40, 20, 0, 14, 28, 1, 15, 29, 2, 16, 30, 3, 17, 31, 4, 18, 32, 5, 19, 33, 6, 20, 34, 7, 21, 35, 8, 22, 36,
    9, 23, 37, 10, 24, 38, 11, 25, 39, 12, 26, 40, 13, 27, 0, 31, 21, 11, 1, 32, 22, 12, 2, 33, 23, 13, 3, 34, 24, 14, 4, 35, 25, 15, 5, 36, 26, 16, 6, 37,
    27, 17, 7, 38, 28, 18, 8, 39, 29, 19, 9, 40, 30, 20, 10, 0, 33, 25, 17, 9, 1, 34, 26, 18, 10, 2, 35, 27, 19, 11, 3, 36, 28, 20, 12, 4, 37, 29, 21, 13,
    5, 38, 30, 22, 14, 6, 39, 31, 23, 15, 7, 40, 32, 24, 16, 8, 0, 7, 14, 21, 28, 35, 1, 8, 15, 22, 29, 36, 2, 9, 16, 23, 30, 37, 3, 10, 17, 24, 31, 38,
    4, 11, 18, 25, 32, 39, 5, 12, 19, 26, 33, 40, 6, 13, 20, 27, 34, 0, 6, 12, 18, 24, 30, 36, 1, 7, 13, 19, 25, 31, 37, 2, 8, 14, 20, 26, 32, 38, 3, 9,
    15, 21, 27, 33, 39, 4, 10, 16, 22, 28, 34, 40, 5, 11, 17, 23, 29, 35, 0, 36, 31, 26, 21, 16, 11, 6, 1, 37, 32, 27, 22, 17, 12, 7, 2, 38, 33, 28, 23, 18,
    13, 8, 3, 39, 34, 29, 24, 19, 14, 9, 4, 40, 35, 30, 25, 20, 15, 10, 5, 0, 32, 23, 14, 5, 37, 28, 19, 10, 1, 33, 24, 15, 6, 38, 29, 20, 11, 2, 34, 25,
    16, 7, 39, 30, 21, 12, 3, 35, 26, 17, 8, 40, 31, 22, 13, 4, 36, 27, 18, 9, 0, 37, 33, 29, 25, 21, 17, 13, 9, 5, 1, 38, 34, 30, 26, 22, 18, 14, 10, 6,
    2, 39, 35, 31, 27, 23, 19, 15, 11, 7, 3, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0, 15, 30, 4, 19, 34, 8, 23, 38, 12, 27, 1, 16, 31, 5, 20, 35, 9, 24,
    39, 13, 28, 2, 17, 32, 6, 21, 36, 10, 25, 40, 14, 29, 3, 18, 33, 7, 22, 37, 11, 26, 0, 24, 7, 31, 14, 38, 21, 4, 28, 11, 35, 18, 1, 25, 8, 32, 15, 39,
    22, 5, 29, 12, 36, 19, 2, 26, 9, 33, 16, 40, 23, 6, 30, 13, 37, 20, 3, 27, 10, 34, 17, 0, 19, 38, 16, 35, 13, 32, 10, 29, 7, 26, 4, 23, 1, 20, 39, 17,
    36, 14, 33, 11, 30, 8, 27, 5, 24, 2, 21, 40, 18, 37, 15, 34, 12, 31, 9, 28, 6, 25, 3, 22, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 1, 4,
    7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 0, 11, 22, 33, 3, 14, 25, 36, 6, 17, 28, 39, 9, 20, 31,
    1, 12, 23, 34, 4, 15, 26, 37, 7, 18, 29, 40, 10, 21, 32, 2, 13, 24, 35, 5, 16, 27, 38, 8, 19, 30, 0, 18, 36, 13, 31, 8, 26, 3, 21, 39, 16, 34, 11, 29,
    6, 24, 1, 19, 37, 14, 32, 9, 27, 4, 22, 40, 17, 35, 12, 30, 7, 25, 2, 20, 38, 15, 33, 10, 28, 5, 23, 0, 29, 17, 5, 34, 22, 10, 39, 27, 15, 3, 32, 20,
    8, 37, 25, 13, 1, 30, 18, 6, 35, 23, 11, 40, 28, 16, 4, 33, 21, 9, 38, 26, 14, 2, 31, 19, 7, 36, 24, 12, 0, 16, 32, 7, 23, 39, 14, 30, 5, 21, 37, 12,
    28, 3, 19, 35, 10, 26, 1, 17, 33, 8, 24, 40, 15, 31, 6, 22, 38, 13, 29, 4, 20, 36, 11, 27, 2, 18, 34, 9, 25, 0, 13, 26, 39, 11, 24, 37, 9, 22, 35, 7,
    20, 33, 5, 18, 31, 3, 16, 29, 1, 14, 27, 40, 12, 25, 38, 10, 23, 36, 8, 21, 34, 6, 19, 32, 4, 17, 30, 2, 15, 28, 0, 39, 37, 35, 33, 31, 29, 27, 25, 23,
    21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 2, 4, 6, 8, 10, 12, 14, 16,
    18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 0, 28, 15, 2, 30, 17, 4, 32,
    19, 6, 34, 21, 8, 36, 23, 10, 38, 25, 12, 40, 27, 14, 1, 29, 16, 3, 31, 18, 5, 33, 20, 7, 35, 22, 9, 37, 24, 11, 39, 26, 13, 0, 25, 9, 34, 18, 2, 27,
    11, 36, 20, 4, 29, 13, 38, 22, 6, 31, 15, 40, 24, 8, 33, 17, 1, 26, 10, 35, 19, 3, 28, 12, 37, 21, 5, 30, 14, 39, 23, 7, 32, 16, 0, 12, 24, 36, 7, 19,
    31, 2, 14, 26, 38, 9, 21, 33, 4, 16, 28, 40, 11, 23, 35, 6, 18, 30, 1, 13, 25, 37, 8, 20, 32, 3, 15, 27, 39, 10, 22, 34, 5, 17, 29, 0, 23, 5, 28, 10,
    33, 15, 38, 20, 2, 25, 7, 30, 12, 35, 17, 40, 22, 4, 27, 9, 32, 14, 37, 19, 1, 24, 6, 29, 11, 34, 16, 39, 21, 3, 26, 8, 31, 13, 36, 18, 0, 30, 19, 8,
    38, 27, 16, 5, 35, 24, 13, 2, 32, 21, 10, 40, 29, 18, 7, 37, 26, 15, 4, 34, 23, 12, 1, 31, 20, 9, 39, 28, 17, 6, 36, 25, 14, 3, 33, 22, 11, 0, 38, 35,
    32, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 40, 37, 34, 31, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0, 22,
    3, 25, 6, 28, 9, 31, 12, 34, 15, 37, 18, 40, 21, 2, 24, 5, 27, 8, 30, 11, 33, 14, 36, 17, 39, 20, 1, 23, 4, 26, 7, 29, 10, 32, 13, 35, 16, 38, 19, 0,
    17, 34, 10, 27, 3, 20, 37, 13, 30, 6, 23, 40, 16, 33, 9, 26, 2, 19, 36, 12, 29, 5, 22, 39, 15, 32, 8, 25, 1, 18, 35, 11, 28, 4, 21, 38, 14, 31, 7, 24,
    0, 26, 11, 37, 22, 7, 33, 18, 3, 29, 14, 40, 25, 10, 36, 21, 6, 32, 17, 2, 28, 13, 39, 24, 9, 35, 20, 5, 31, 16, 1, 27, 12, 38, 23, 8, 34, 19, 4, 30,
    15, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 1, 5, 9, 13, 17, 21, 25, 29,
    33, 37, 0, 9, 18, 27, 36, 4, 13, 22, 31, 40, 8, 17, 26, 35, 3, 12, 21, 30, 39, 7, 16, 25, 34, 2, 11, 20, 29, 38, 6, 15, 24, 33, 1, 10, 19, 28, 37, 5,
    14, 23, 32, 0, 5, 10, 15, 20, 25, 30, 35, 40, 4, 9, 14, 19, 24, 29, 34, 39, 3, 8, 13, 18, 23, 28, 33, 38, 2, 7, 12, 17, 22, 27, 32, 37, 1, 6, 11, 16,
    21, 26, 31, 36, 0, 35, 29, 23, 17, 11, 5, 40, 34, 28, 22, 16, 10, 4, 39, 33, 27, 21, 15, 9, 3, 38, 32, 26, 20, 14, 8, 2, 37, 31, 25, 19, 13, 7, 1, 36,
    30, 24, 18, 12, 6, 0, 34, 27, 20, 13, 6, 40, 33, 26, 19, 12, 5, 39, 32, 25, 18, 11, 4, 38, 31, 24, 17, 10, 3, 37, 30, 23, 16, 9, 2, 36, 29, 22, 15, 8,
    1, 35, 28, 21, 14, 7, 0, 8, 16, 24, 32, 40, 7, 15, 23, 31, 39, 6, 14, 22, 30, 38, 5, 13, 21, 29, 37, 4, 12, 20, 28, 36, 3, 11, 19, 27, 35, 2, 10, 18,
    26, 34, 1, 9, 17, 25, 33, 0, 10, 20, 30, 40, 9, 19, 29, 39, 8, 18, 28, 38, 7, 17, 27, 37, 6, 16, 26, 36, 5, 15, 25, 35, 4, 14, 24, 34, 3, 13, 23, 33,
    2, 12, 22, 32, 1, 11, 21, 31, 0, 27, 13, 40, 26, 12, 39, 25, 11, 38, 24, 10, 37, 23, 9, 36, 22, 8, 35, 21, 7, 34, 20, 6, 33, 19, 5, 32, 18, 4, 31, 17,
    3, 30, 16, 2, 29, 15, 1, 28, 14, 0, 20, 40, 19, 39, 18, 38, 17, 37, 16, 36, 15, 35, 14, 34, 13, 33, 12, 32, 11, 31, 10, 30, 9, 29, 8, 28, 7, 27, 6, 26,
    5, 25, 4, 24, 3, 23, 2, 22, 1, 21, 0, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
    11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 0, 22, 1, 23, 2, 24, 3, 25, 4, 26, 5, 27, 6, 28, 7, 29, 8, 30, 9, 31, 10, 32, 11,
    33, 12, 34, 13, 35, 14, 36, 15, 37, 16, 38, 17, 39, 18, 40, 19, 41, 20, 42, 21, 0, 29, 15, 1, 30, 16, 2, 31, 17, 3, 32, 18, 4, 33, 19, 5, 34, 20, 6, 35,
    21, 7, 36, 22, 8, 37, 23, 9, 38, 24, 10, 39, 25, 11, 40, 26, 12, 41, 27, 13, 42, 28, 14, 0, 11, 22, 33, 1, 12, 23, 34, 2, 13, 24, 35, 3, 14, 25, 36, 4,
    15, 26, 37, 5, 16, 27, 38, 6, 17, 28, 39, 7, 18, 29, 40, 8, 19, 30, 41, 9, 20, 31, 42, 10, 21, 32, 0, 26, 9, 35, 18, 1, 27, 10, 36, 19, 2, 28, 11, 37,
    20, 3, 29, 12, 38, 21, 4, 30, 13, 39, 22, 5, 31, 14, 40, 23, 6, 32, 15, 41, 24, 7, 33, 16, 42, 25, 8, 34, 17, 0, 36, 29, 22, 15, 8, 1, 37, 30, 23, 16,
    9, 2, 38, 31, 24, 17, 10, 3, 39, 32, 25, 18, 11, 4, 40, 33, 26, 19, 12, 5, 41, 34, 27, 20, 13, 6, 42, 35, 28, 21, 14, 7, 0, 37, 31, 25, 19, 13, 7, 1,
    38, 32, 26, 20, 14, 8, 2, 39, 33, 27, 21, 15, 9, 3, 40, 34, 28, 22, 16, 10, 4, 41, 35, 29, 23, 17, 11, 5, 42, 36, 30, 24, 18, 12, 6, 0, 27, 11, 38, 22,
    6, 33, 17, 1, 28, 12, 39, 23, 7, 34, 18, 2, 29, 13, 40, 24, 8, 35, 19, 3, 30, 14, 41, 25, 9, 36, 20, 4, 31, 15, 42, 26, 10, 37, 21, 5, 32, 16, 0, 24,
    5, 29, 10, 34, 15, 39, 20, 1, 25, 6, 30, 11, 35, 16, 40, 21, 2, 26, 7, 31, 12, 36, 17, 41, 22, 3, 27, 8, 32, 13, 37, 18, 42, 23, 4, 28, 9, 33, 14, 38,
    19, 0, 13, 26, 39, 9, 22, 35, 5, 18, 31, 1, 14, 27, 40, 10, 23, 36, 6, 19, 32, 2, 15, 28, 41, 11, 24, 37, 7, 20, 33, 3, 16, 29, 42, 12, 25, 38, 8, 21,
    34, 4, 17, 30, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 3, 7, 11,
    15, 19, 23, 27, 31, 35, 39, 0, 18, 36, 11, 29, 4, 22, 40, 15, 33, 8, 26, 1, 19, 37, 12, 30, 5, 23, 41, 16, 34, 9, 27, 2, 20, 38, 13, 31, 6, 24, 42, 17,
    35, 10, 28, 3, 21, 39, 14, 32, 7, 25, 0, 10, 20, 30, 40, 7, 17, 27, 37, 4, 14, 24, 34, 1, 11, 21, 31, 41, 8, 18, 28, 38, 5, 15, 25, 35, 2, 12, 22, 32,
    42, 9, 19, 29, 39, 6, 16, 26, 36, 3, 13, 23, 33, 0, 40, 37, 34, 31, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1, 41, 38, 35, 32, 29, 26, 23, 20, 17, 14, 11, 8,
    5, 2, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0, 23, 3, 26, 6, 29, 9, 32, 12, 35, 15, 38, 18, 41, 21, 1, 24, 4, 27, 7, 30, 10, 33, 13,
    36, 16, 39, 19, 42, 22, 2, 25, 5, 28, 8, 31, 11, 34, 14, 37, 17, 40, 20, 0, 35, 27, 19, 11, 3, 38, 30, 22, 14, 6, 41, 33, 25, 17, 9, 1, 36, 28, 20, 12,
    4, 39, 31, 23, 15, 7, 42, 34, 26, 18, 10, 2, 37, 29, 21, 13, 5, 40, 32, 24, 16, 8, 0, 38, 33, 28, 23, 18, 13, 8, 3, 41, 36, 31, 26, 21, 16, 11, 6, 1,
    39, 34, 29, 24, 19, 14, 9, 4, 42, 37, 32, 27, 22, 17, 12, 7, 2, 40, 35, 30, 25, 20, 15, 10, 5, 0, 12, 24, 36, 5, 17, 29, 41, 10, 22, 34, 3, 15, 27, 39,
    8, 20, 32, 1, 13, 25, 37, 6, 18, 30, 42, 11, 23, 35, 4, 16, 28, 40, 9, 21, 33, 2, 14, 26, 38, 7, 19, 31, 0, 34, 25, 16, 7, 41, 32, 23, 14, 5, 39, 30,
    21, 12, 3, 37, 28, 19, 10, 1, 35, 26, 17, 8, 42, 33, 24, 15, 6, 40, 31, 22, 13, 4, 38, 29, 20, 11, 2, 36, 27, 18, 9, 0, 28, 13, 41, 26, 11, 39, 24, 9,
    37, 22, 7, 35, 20, 5, 33, 18, 3, 31, 16, 1, 29, 14, 42, 27, 12, 40, 25, 10, 38, 23, 8, 36, 21, 6, 34, 19, 4, 32, 17, 2, 30, 15, 0, 41, 39, 37, 35, 33,
    31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 2, 4,
    6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
    0, 15, 30, 2, 17, 32, 4, 19, 34, 6, 21, 36, 8, 23, 38, 10, 25, 40, 12, 27, 42, 14, 29, 1, 16, 31, 3, 18, 33, 5, 20, 35, 7, 22, 37, 9, 24, 39, 11, 26,
    41, 13, 28, 0, 9, 18, 27, 36, 2, 11, 20, 29, 38, 4, 13, 22, 31, 40, 6, 15, 24, 33, 42, 8, 17, 26, 35, 1, 10, 19, 28, 37, 3, 12, 21, 30, 39, 5, 14, 23,
    32, 41, 7, 16, 25, 34, 0, 31, 19, 7, 38, 26, 14, 2, 33, 21, 9, 40, 28, 16, 4, 35, 23, 11, 42, 30, 18, 6, 37, 25, 13, 1, 32, 20, 8, 39, 27, 15, 3, 34,
    22, 10, 41, 29, 17, 5, 36, 24, 12, 0, 5, 10, 15, 20, 25, 30, 35, 40, 2, 7, 12, 17, 22, 27, 32, 37, 42, 4, 9, 14, 19, 24, 29, 34, 39, 1, 6, 11, 16, 21,
    26, 31, 36, 41, 3, 8, 13, 18, 23, 28, 33, 38, 0, 8, 16, 24, 32, 40, 5, 13, 21, 29, 37, 2, 10, 18, 26, 34, 42, 7, 15, 23, 31, 39, 4, 12, 20, 28, 36, 1,
    9, 17, 25, 33, 41, 6, 14, 22, 30, 38, 3, 11, 19, 27, 35, 0, 20, 40, 17, 37, 14, 34, 11, 31, 8, 28, 5, 25, 2, 22, 42, 19, 39, 16, 36, 13, 33, 10, 30, 7,
    27, 4, 24, 1, 21, 41, 18, 38, 15, 35, 12, 32, 9, 29, 6, 26, 3, 23, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 2, 5, 8, 11, 14, 17, 20,
    23, 26, 29, 32, 35, 38, 41, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 0, 33, 23, 13, 3, 36, 26, 16, 6, 39, 29, 19, 9, 42, 32, 22, 12, 2, 35,
    25, 15, 5, 38, 28, 18, 8, 41, 31, 21, 11, 1, 34, 24, 14, 4, 37, 27, 17, 7, 40, 30, 20, 10, 0, 25, 7, 32, 14, 39, 21, 3, 28, 10, 35, 17, 42, 24, 6, 31,
    13, 38, 20, 2, 27, 9, 34, 16, 41, 23, 5, 30, 12, 37, 19, 1, 26, 8, 33, 15, 40, 22, 4, 29, 11, 36, 18, 0, 39, 35, 31, 27, 23, 19, 15, 11, 7, 3, 42, 38,
    34, 30, 26, 22, 18, 14, 10, 6, 2, 41, 37, 33, 29, 25, 21, 17, 13, 9, 5, 1, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0, 30, 17, 4, 34, 21, 8, 38, 25, 12,
    42, 29, 16, 3, 33, 20, 7, 37, 24, 11, 41, 28, 15, 2, 32, 19, 6, 36, 23, 10, 40, 27, 14, 1, 31, 18, 5, 35, 22, 9, 39, 26, 13, 0, 19, 38, 14, 33, 9, 28,
    4, 23, 42, 18, 37, 13, 32, 8, 27, 3, 22, 41, 17, 36, 12, 31, 7, 26, 2, 21, 40, 16, 35, 11, 30, 6, 25, 1, 20, 39, 15, 34, 10, 29, 5, 24, 0, 16, 32, 5,
    21, 37, 10, 26, 42, 15, 31, 4, 20, 36, 9, 25, 41, 14, 30, 3, 19, 35, 8, 24, 40, 13, 29, 2, 18, 34, 7, 23, 39, 12, 28, 1, 17, 33, 6, 22, 38, 11, 27, 0,
    6, 12, 18, 24, 30, 36, 42, 5, 11, 17, 23, 29, 35, 41, 4, 10, 16, 22, 28, 34, 40, 3, 9, 15, 21, 27, 33, 39, 2, 8, 14, 20, 26, 32, 38, 1, 7, 13, 19, 25,
    31, 37, 0, 7, 14, 21, 28, 35, 42, 6, 13, 20, 27, 34, 41, 5, 12, 19, 26, 33, 40, 4, 11, 18, 25, 32, 39, 3, 10, 17, 24, 31, 38, 2, 9, 16, 23, 30, 37, 1,
    8, 15, 22, 29, 36, 0, 17, 34, 8, 25, 42, 16, 33, 7, 24, 41, 15, 32, 6, 23, 40, 14, 31, 5, 22, 39, 13, 30, 4, 21, 38, 12, 29, 3, 20, 37, 11, 28, 2, 19,
    36, 10, 27, 1, 18, 35, 9, 26, 0, 32, 21, 10, 42, 31, 20, 9, 41, 30, 19, 8, 40, 29, 18, 7, 39, 28, 17, 6, 38, 27, 16, 5, 37, 26, 15, 4, 36, 25, 14, 3,
    35, 24, 13, 2, 34, 23, 12, 1, 33, 22, 11, 0, 14, 28, 42, 13, 27, 41, 12, 26, 40, 11, 25, 39, 10, 24, 38, 9, 23, 37, 8, 22, 36, 7, 21, 35, 6, 20, 34, 5,
    19, 33, 4, 18, 32, 3, 17, 31, 2, 16, 30, 1, 15, 29, 0, 21, 42, 20, 41, 19, 40, 18, 39, 17, 38, 16, 37, 15, 36, 14, 35, 13, 34, 12, 33, 11, 32, 10, 31, 9,
    30, 8, 29, 7, 28, 6, 27, 5, 26, 4, 25, 3, 24, 2, 23, 1, 22, 0, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21,
    20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
};
inline u32 skr2_target_q(const u32 q, const u32 p, const u32 mw)
{
    u32 T = 0u;
#define SKR_F(i, o, r) { const u32 mr = SKR_MR2[o + (p % r) * r + q % r]; const u32 x = (mw >> (7u * i)) & 127u; \
        T |= ((x >= mr) ? x - mr : x + r - mr) << (7u * i); }
    SKR_F(0u, 0u, 31u) SKR_F(1u, 961u, 37u) SKR_F(2u, 2330u, 41u) SKR_F(3u, 4011u, 43u)
#undef SKR_F
    return T;
}
inline u32 skr2_target(const u32 bit, const u32 p, const u32 w, const u32 qfr2)
{
    return skr2_target_q(qfr2 + 2u * bit, p, WHL_SF2[w]);
}
inline u32 skr2_wrap(const u32 T)
{
    u32 o = 0u;
#define SKR_W(i, r, k) { const u32 x = (T >> (7u * i)) & 127u; o |= ((x >= k) ? x - k : x + r - k) << (7u * i); }
    SKR_W(0u, 31u, 22u) SKR_W(1u, 37u, 23u) SKR_W(2u, 41u, 18u) SKR_W(3u, 43u, 16u)
#undef SKR_W
    return o;
}
/// mark_multiples_skr / _skr2 with the multiplier residues given (LARGE_MQ):
/// m30 = m0 mod 30030, mr = m0 mod SKR_R, mr2 = m0 mod 31*37*41*43 of the
/// odd multiple q = p m0 at bit0 — no WHL_INV gather, no inverse tables.
/// @pre    as mark_multiples_skr2; use2: also skip the second residue set.
inline void mark_multiples_mq(volatile __global u64* prime_map, const u32 bit0,
                              const u32 p, const u32 n, u32 m30, u32 mr, u32 mr2,
                              const bool use1, const bool use2, const cmap g)
{
    volatile __global u32* m32 = (volatile __global u32*)prime_map;
    u32 bit = bit0;
    if (bit >= n) return;
    u32 k2 = 0u;                                        // 2 * probes
    while (WHL_IDX[m30] == WHL_MISS) {
        if (n - bit <= p) return;
        bit += p;
        k2 += 2u;
        m30 += 2u; if (m30 >= WHEEL_W) m30 -= WHEEL_W;
    }
    u32 w = WHL_IDX[m30];
    mr += k2; mr -= (mr >= SKR_R) ? SKR_R : 0u;
    u32 T = 0x7DF7DF0u;                                  // no field can match (e < 32)
    if (use1) {
        const u32 e0 = WHL_SF[w] >> 4;
        T = 0u;
#define MQ_F(i, r) { const u32 a = mr % r; const u32 x = (e0 >> (6u * i)) & 63u; \
        T |= ((x >= a) ? x - a : x + r - a) << (4u + 6u * i); }
        MQ_F(0u, 17u) MQ_F(1u, 19u) MQ_F(2u, 23u) MQ_F(3u, 29u)
#undef MQ_F
    }
    u32 T2 = 0x7EFDFBFu;                                 // no field can match (e2 < 64)
    if (use2) {
        mr2 += k2; mr2 -= (mr2 >= 2022161u) ? 2022161u : 0u;
        const u32 f0 = WHL_SF2[w];
        T2 = 0u;
#define MQ_F(i, r) { const u32 a = mr2 % r; const u32 x = (f0 >> (7u * i)) & 127u; \
        T2 |= ((x >= a) ? x - a : x + r - a) << (7u * i); }
        MQ_F(0u, 31u) MQ_F(1u, 37u) MQ_F(2u, 41u) MQ_F(3u, 43u)
#undef MQ_F
    }
    for (;;) {
        const u32 e = WHL_SF[w];
        if (((((e ^ T) | SKR_H) - SKR_L) & SKR_H) == SKR_H &&
            (!use2 || ((((WHL_SF2[w] ^ T2) | SKR2_H) - SKR2_L) & SKR2_H) == SKR2_H))
            map_clear_chk(m32, bit, g);
        const u64 step = (u64)p * (e & 15u);
        if ((u64)(n - bit) <= step) break;
        bit += (u32)step;
        if (++w == WHL_N) { w = 0u; if (use1) T = skr_wrap(T); if (use2) T2 = skr2_wrap(T2); }
    }
}

/// mark_multiples_skr with the second residue set too.
/// @pre    as mark_multiples_skr; qfr2 = q_first mod 31*37*41*43.
inline void mark_multiples_skr2(volatile __global u64* prime_map, const u64 bit0,
                                const u32 p, const u64 q_odds, const u32 qfw, const u32 qfr,
                                const u32 qfr2, const cmap g)
{
    volatile __global u32* m32 = (volatile __global u32*)prime_map;
    const u32 n = (u32)q_odds;
    u32 bit = (u32)bit0;
    if (bit >= n) return;
    u32 m = (qfw + 2u * (bit % (WHEEL_W / 2))) % WHEEL_W;
    m = (m * WHL_INV[p % WHEEL_W]) % WHEEL_W;
    while (WHL_IDX[m] == WHL_MISS) {
        if (n - bit <= p) return;
        bit += p;
        m += 2u; if (m >= WHEEL_W) m -= WHEEL_W;
    }
    u32 w = WHL_IDX[m];
    u32 T = skr_target(bit, p, w, qfr);
    u32 T2 = skr2_target(bit, p, w, qfr2);
    for (;;) {
        const u32 e = WHL_SF[w];
        const u32 e2 = WHL_SF2[w];
        if (((((e ^ T) | SKR_H) - SKR_L) & SKR_H) == SKR_H &&
            ((((e2 ^ T2) | SKR2_H) - SKR2_L) & SKR2_H) == SKR2_H)
            map_clear_w(m32, bit, g);
        const u64 step = (u64)p * (e & 15u);
        if ((u64)(n - bit) <= step) break;
        bit += (u32)step;
        if (++w == WHL_N) { w = 0u; T = skr_wrap(T); T2 = skr2_wrap(T2); }
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
    if (p > 61u && p < TILE_WHEEL_MAXP) {       // many marks per tile only
        // wheel-30 (nibble steps): the WHEEL_W constant tables
        // measured slower here (sieve_s 0.085 vs 0.081)
        mark_local_w30(t32, bit, (u32)TILE_BITS, p, (u32)tile_bit0 + bit, qf30);
        return;
    }
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
#ifndef MED_W7_MAXP
#define MED_W7_MAXP 8192u
#endif
__constant uchar INV7[7] = {0, 1, 4, 5, 2, 3, 6};   // x^-1 mod 7
inline void med_prime_f32n(volatile __local u32* t32, const u32 fb, const u32 p,
                          const u32 tile_bit0, const u32 qc7, const u32 s7, const u32 nbits,
                          const u32 c1)
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
    // Wheel 7, unrolled (2026-11-18, sieve -4.4 ms per 1e11): the marks of
    // q = 0 (mod 7) are already cleared by the tile kernel's own sieving of
    // 7 (mask or walk; q = p m >= 7 p, never 7 itself). Along a class array
    // q(i) = qc + s i (s = 2 cmod = 30 or 6), so q_k = q_0 + s p k (mod 7)
    // and exactly the k = k7 (mod 7) are skipped: blocks of 7 p take six
    // marks at fixed offsets (no per-mark ALU), the tail tests each of at
    // most 7 candidates. qc7 = qc mod 7, s7 = s mod 7.
    // Only primes with a few full 7p blocks per tile: above, the wheel's
    // setup and the 7-candidate tail cost more than the marks they skip
    // (low heights, split 36000: sieve +55 ms per 1e11 at 1e14).
    if (p == 7u || p > (u32)MED_W7_MAXP) {
        for (; bit < nbits; bit += p)
            atomic_and(&t32[bit >> 5], CLR_MASK_R(c1, bit));
        return;
    }
    if (bit < nbits) {
        const u32 q0 = (qc7 + s7 * (tile_bit0 + bit)) % 7u;      // class bits < 2^24
        const u32 k7 = ((7u - q0) * INV7[(s7 * p) % 7u]) % 7u;
        const u32 p6 = 6u * p, p7 = 7u * p;
        u32 o[6];
        #pragma unroll
        for (u32 t = 0; t < 6u; ++t) o[t] = (t + (t >= k7 ? 1u : 0u)) * p;
        for (; bit + p6 < nbits; bit += p7) {
            #pragma unroll
            for (u32 t = 0; t < 6u; ++t) {
                const u32 b = bit + o[t];
                atomic_and(&t32[b >> 5], CLR_MASK_R(c1, b));
            }
        }
        for (u32 d = 0; d < 7u; ++d) {
            const u32 b = bit + d * p;
            if (b >= nbits) break;
            if (d != k7) atomic_and(&t32[b >> 5], CLR_MASK_R(c1, b));
        }
    }
}

inline void med_prime_f32(volatile __local u32* t32, const u32 fb, const u32 p,
                          const u32 tile_bit0, const u32 qc7, const u32 s7, const u32 c1)
{
    med_prime_f32n(t32, fb, p, tile_bit0, qc7, s7, (u32)TILE_BITS, c1);
}

// One prime of the thread-owned walk (table path) in 32-bit / float
// arithmetic: tf = tile-local first mark (0xffffffff = none), [b0, b1) = my
// bit range, all < TILE_BITS = 2^17, so d = b0 - tf is exact as a float and
// fma(-trunc(d / p), p, d) is the exact remainder in (-p, 2p) (as
// med_prime_f32). Replaces the 64-bit mul_hi / k * p jump of the table path.
/// @pre    p < TILE_BITS; b0 <= b1 <= TILE_BITS; tf < TILE_BITS or 0xffffffff.
/// @post   my bits of p's marks in this tile are cleared: posted SLM atomic
///         ANDs. make_prime_map depends on the atomics: no barrier separates
///         them from the medium stage.
inline void own_prime_f32(__local u32* t32, const u32 tf, const u32 p,
                          const u32 b0, const u32 b1, const u32 c1)
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
    // A posted (non-returning) SLM atomic costs less than the load + store of
    // a plain RMW, although the thread owns the word (sieve -22 ms per 1e11).
    for (; bit < b1; bit += p)
        atomic_and((volatile __local u32*)&t32[bit >> 5], CLR_MASK_R(c1, bit));
}

// Medium-stage load batch (v1.10.0). sieve_s per 2e10 window: unbatched
// 0.0824, 2 -> 0.0775, 4 -> 0.0745, 8 -> 0.0728 (+3.6% total), 12 ties,
// 16 spills (-3.7%); a two-stage pipeline on top measured no better.
// Medium-stage load balance (2026-11-18): the medium primes are dealt to
// the threads in rows of WG_SIZE, and the barrier after the stage waits for
// the thread with the most marks: with p = 521 (215 marks per tile) in row
// 0 that thread had 1.6-1.8x the mean. The first MED_SPLITN primes are two
// items each (one per tile half) and the rows alternate direction (snake):
// max / mean 1.15. 0 = one item per prime, rows in order.
// Fused mask stage in the coalesced word order (2026-11-18, +0.1%)
#ifndef MASK_COAL
#define MASK_COAL 1
#endif
#ifndef MED_SPLITN
#define MED_SPLITN 128
#endif
#ifndef MED_BATCH
#define MED_BATCH 8
#endif

// ----------------------------------------------------------------------------
// make_prime_map
//
// Stage 1: every thread fills its 32 owned SLM words with ones.
// Stage 2: ONE barrier (the only one in the kernel).
// Stage 3a: small-prime masks — primes < MASK_MAX_P are applied as WHOLE-WORD
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
        const u32 n_groups_xp,              // n_groups | xp << 8: groups (0 = per-prime mask
                                            // loop, stage 3a); xp primes after the mask primes
                                            // folded into them (mod-30 class map, GB_MASK_XP)
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
    const u32 n_groups = n_groups_xp & 0xFFu;
    // ~1 for CLR_MASK_R, opaque to the compiler (q_odds < 2^32: the shift is
    // 0): a literal is re-materialized by two scalar movs in every mark loop
    // iteration (the rol takes no immediate); a register holds it once
    // (2026-11-18: tile -0.4%, mid_window -2%)
    const u32 c1v = 0xFFFFFFFEu ^ (u32)(q_odds >> 40);
    // cm 1: 2 classes (phis = phi1 | phi5 << 8); cm 2: 8 classes mod 30
    // (phis = 8 nibbles, phi_k = dense bit phase mod 15 of class k)
    const u32 cls = cm ? get_group_id(0) / tpc : 0u;
    const u32 tile_id = get_group_id(0) - cls * tpc;   // tile within its class (dense: all)
    const u64 tile_bit0 = (u64)tile_id * TILE_BITS;
    const u32 phic = (cm == 2u) ? ((phis >> (4u * cls)) & 15u)
                                : (cls ? (phis >> 8) : (phis & 0xFFu));
    const u32 cmod = (cm == 2u) ? 15u : 3u;            // dense bits per class bit
    const u32 qc7v = (u32)((q_first + 2UL * phic) % 7UL);   // q of class bit 0, mod 7
    const u32 s7v = (2u * cmod) % 7u;                       // q step per class bit, mod 7
    const u32 qf30 = (u32)(q_first % 30UL);            // wheel-30 phase (dense map)
    const u64 n_valid = (cm == 2u) ? ((q_odds > phic) ? (q_odds - phic + 14UL) / 15UL : 0UL)
                      : cm ? (cls ? n5 : n1) : q_odds;
    fb_global += cls * fb_stride;
    tile_first += (u64)cls * tpc * own_end;          // rows of this class's tiles
    group_masks += cls * mask_coff;
    mask_table += (u64)cls * mask_count * 64UL;

    // ---- Stage 1: initialize my 64 words to "all prime" --------------------
    const u32 my_word0 = me * WORDS_PER_THREAD;
    if (n_groups > 0 && mask_count > 0) {                 // uniform
        // Fused stages 1 + 3a: every owned word is computed in registers as
        // the AND of one combined mask per prime GROUP (host-built:
        // primes.cpp small_prime_group_masks; a group's mask depends only on
        // word index mod the product of its primes) and stored ONCE. The
        // per-prime path costs a global load plus an SLM read-modify-write
        // per prime (17) per word; this costs n_groups (7) loads per word.
#if MASK_COAL
        // Coalesced: word i * WG_SIZE + me (the barrier below orders these
        // plain stores before every owner's walk), so a sub-group reads
        // consecutive table entries instead of entries 8 words apart.
        const u32 x = (u32)(tile_bit0 >> 6) + me;          // global word index < 2^32
#else
        const u32 x = (u32)((tile_bit0 >> 6) + my_word0);  // global word index < 2^32
#endif
        u32 ph[MAX_MASK_GROUPS], per[MAX_MASK_GROUPS], off[MAX_MASK_GROUPS];
#if MASK_COAL
        u32 stp[MAX_MASK_GROUPS];
#endif
        #pragma unroll
        for (u32 g = 0; g < MAX_MASK_GROUPS; ++g) {
            if (g < n_groups) {
                off[g] = group_desc[2 * g];
                per[g] = group_desc[2 * g + 1];
                ph[g] = x % per[g];
#if MASK_COAL
                stp[g] = (u32)WG_SIZE % per[g];
#endif
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
#if MASK_COAL
                    ph[g] += stp[g];
                    ph[g] -= (ph[g] >= per[g]) ? per[g] : 0u;
#else
                    ph[g] = (ph[g] + 1u == per[g]) ? 0u : ph[g] + 1u;
#endif
                }
            }
#if MASK_COAL
            tile[i * WG_SIZE + me] = v;
#else
            tile[my_word0 + i] = v;
#endif
        }
    } else
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
                tile_and64_atomic(tile, w, masks[phase]);
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
        // contain it; exactly one thread owns the word, but it is an
        // atomic OR: the medium stage's atomic ANDs on the
        // same word may already be running (they never clear q = p itself,
        // and OR / AND of different bits commute).
        // (and the folded primes, < 256: host caps them at 8)
        if (tile_id == 0 && q_first <= 4UL * MASK_MAX_P) {
            const u32 xr = (n_groups > 0) ? n_groups_xp >> 8 : 0u;
            for (u32 j = 0; j < mask_count + xr; ++j) {
                const u64 p = sieve_primes[j];
                if (p < q_first) continue;         // q = p lies before the segment
                u64 bit = (p - q_first) >> 1;      // global bit of q = p
                if (cm) {                          // class index of q = p (if in this class)
                    if (bit % cmod != phic) continue;
                    bit /= cmod;
                }
                if (bit < TILE_BITS) {
                    const u32 owner = (u32)(bit >> 6);
                    if (owner >= my_word0 && owner < my_word0 + WORDS_PER_THREAD) {
                        atomic_or((volatile __local u32*)tile + (u32)(bit >> 5), 1u << (u32)(bit & 31UL));
                    }
                }
            }
        }
    }

    // ---- Stage 3: sieve -----------------------------------------------------
    // My bit range inside the tile: [my_bit0, my_bit0 + BITS_PER_THREAD)
    const u64 my_bit0 = (u64)me * BITS_PER_THREAD;
    const u64 my_bit1 = my_bit0 + BITS_PER_THREAD;

    // Primes [0, mask_count) were already applied by the mask path (stage 3a);
    // sieve_primes is ascending, so they are exactly the first mask_count.
    // Primes [sieve_split, sieve_count) are handled by sieve_large_primes.
    // the fused groups may also clear the next xp primes (host GB_MASK_XP:
    // 67, 71, ...); the owned walk starts after them
    const u32 xp = (n_groups > 0) ? (n_groups_xp >> 8) & 0xFFu : 0u;
    // medium items (host plan, n_groups_xp >> 16 of them, after the group
    // descriptors): prime index, tile words [lo, hi) as lo | hi << 16
    const u32 med_nit = (n_groups > 0) ? n_groups_xp >> 16 : 0u;
    __global const uint2* med_items = (__global const uint2*)(group_desc + 2u * MAX_MASK_GROUPS);
    const u32 j_begin = (mask_count > 0) ? mask_count + xp : 0;

    if (use_table) {
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
                    own_prime_f32(t32o, tfa[u], pva[u], b0, b1, c1v);
                #pragma unroll
                for (u32 u = 0; u < TABLE_BATCH; ++u) { tfa[u] = tfb[u]; pva[u] = pvb[u]; }
            }
            if (preloaded) {
                #pragma unroll
                for (u32 u = 0; u < TABLE_BATCH; ++u)
                    if (j + u < own_end) own_prime_f32(t32o, tfa[u], pva[u], b0, b1, c1v);
                j += TABLE_BATCH;
            }
            for (; j < own_end; ++j)
                own_prime_f32(t32o, tile_first[trow + j], sieve_primes[j], b0, b1, c1v);
        } else
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
    } else {
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
    // Barriers: every SLM write after stage 2 is atomic -- owned walk, per-prime
    // masks, the q = p restore -- and they commute, so no barrier is
    // needed (a new plain write after stage 2 must be atomic too, or add a
    // barrier here); stage 4 reads words other threads cleared, so every
    // atomic must be done before it.
    // The first bit per tile comes from the segment-global fb_global (host,
    // ~0 = no mark in the segment) with one exact 32-bit modulo; d < 2^32 by
    // the host's segment-span guard (q_odds < 2^32).
    if (own_end < sieve_split) {                       // uniform
        volatile __local u32* t32 = (volatile __local u32*)tile;
        // Loads are batched MED_BATCH primes deep: the walk is latency-bound
        // on the per-prime fb/p loads (a loads-only pass costs a third of
        // the stage), so all loads of a batch are issued before any use.
        u32 j = own_end + me;
        if (cm && n_valid <= (1UL << 24) && sieve_split > 0u &&
            sieve_primes[sieve_split - 1u] < (1u << 23)) {   // uniform (med_prime_f32 PRE)
            __global const u32* fb32 = (__global const u32*)fb_global;   // low words
            const u32 tb = (u32)tile_bit0;
            if (med_nit) {                             // uniform
                // Host-planned items (prime, tile word range) of about equal
                // mark counts, sorted by mark count: the lanes of a sub-group
                // run equally long mark loops. Rows of WG_SIZE items snake.
                const u32 nrow = (med_nit + WG_SIZE - 1u) / WG_SIZE;
                const u32 meo = WG_SIZE - 1u - me;
                u32 r = 0;
                for (; r + MED_BATCH <= nrow; r += MED_BATCH) {
                    u32 fbv[MED_BATCH], pv[MED_BATCH], wv[MED_BATCH];
                    #pragma unroll
                    for (u32 u = 0; u < MED_BATCH; ++u) {
                        const u32 it = (r + u) * WG_SIZE + (((r + u) & 1u) ? meo : me);
                        const bool ok = it < med_nit;
                        const uint2 d = med_items[ok ? it : 0u];
                        wv[u] = ok ? d.y : 0u;
                        fbv[u] = ok ? fb32[2u * d.x] : 0xFFFFFFFFu;
                        pv[u] = ok ? sieve_primes[d.x] : 0u;
                    }
                    #pragma unroll
                    for (u32 u = 0; u < MED_BATCH; ++u) {
                        const u32 w0 = wv[u] & 0xFFFFu, w1 = wv[u] >> 16;
                        med_prime_f32n(t32 + w0, fbv[u], pv[u], tb + 32u * w0, qc7v, s7v,
                                       32u * (w1 - w0), c1v);
                    }
                }
                for (; r < nrow; ++r) {
                    const u32 it = r * WG_SIZE + ((r & 1u) ? meo : me);
                    if (it >= med_nit) continue;
                    const uint2 d = med_items[it];
                    const u32 w0 = d.y & 0xFFFFu, w1 = d.y >> 16;
                    med_prime_f32n(t32 + w0, fb32[2u * d.x], sieve_primes[d.x], tb + 32u * w0, qc7v, s7v,
                                   32u * (w1 - w0), c1v);
                }
                j = sieve_split;
            } else
#if MED_SPLITN
            // (only for short medium ranges, as the mid-window split's 1500:
            // with many rows the imbalance is small and the index ALU costs,
            // -1..-2% at split 36000)
            if (sieve_split - own_end <= 16u * WG_SIZE)
            // Items: the halves of the first m2 medium primes (most marks;
            // one item per tile half), then one per remaining prime. Rows
            // of WG_SIZE items alternate direction (a snake); MED_BATCH rows
            // per load batch.
            {
                const u32 m2 = min((u32)MED_SPLITN, sieve_split - own_end);
                const u32 nit = sieve_split - own_end + m2;
                const u32 nrow = (nit + WG_SIZE - 1u) / WG_SIZE;
                const u32 meo = WG_SIZE - 1u - me;
                u32 r = 0;
                for (; r + MED_BATCH <= nrow; r += MED_BATCH) {
                    u32 fbv[MED_BATCH], pv[MED_BATCH], hv[MED_BATCH];
                    #pragma unroll
                    for (u32 u = 0; u < MED_BATCH; ++u) {
                        const u32 it = (r + u) * WG_SIZE + (((r + u) & 1u) ? meo : me);
                        const u32 jj = own_end + ((it < 2u * m2) ? (it >> 1) : it - m2);
                        hv[u] = (it < 2u * m2) ? (1u + (it & 1u)) : 0u;
                        const bool ok = it < nit;
                        fbv[u] = ok ? fb32[2u * jj] : 0xFFFFFFFFu;
                        pv[u] = ok ? sieve_primes[jj] : 0u;
                    }
                    #pragma unroll
                    for (u32 u = 0; u < MED_BATCH; ++u) {
                        const u32 h = (hv[u] == 2u) ? (u32)TILE_BITS / 2u : 0u;
                        med_prime_f32n(t32 + (h >> 5), fbv[u], pv[u], tb + h, qc7v, s7v,
                                       hv[u] ? (u32)TILE_BITS / 2u : (u32)TILE_BITS, c1v);
                    }
                }
                for (; r < nrow; ++r) {
                    const u32 it = r * WG_SIZE + ((r & 1u) ? meo : me);
                    if (it >= nit) continue;
                    const u32 jj = own_end + ((it < 2u * m2) ? (it >> 1) : it - m2);
                    const u32 hs = (it < 2u * m2) ? (1u + (it & 1u)) : 0u;
                    const u32 h = (hs == 2u) ? (u32)TILE_BITS / 2u : 0u;
                    med_prime_f32n(t32 + (h >> 5), fb32[2u * jj], sieve_primes[jj], tb + h, qc7v, s7v,
                                   hs ? (u32)TILE_BITS / 2u : (u32)TILE_BITS, c1v);
                }
                j = sieve_split;
            }
#endif
            for (; j + (MED_BATCH - 1) * WG_SIZE < sieve_split; j += MED_BATCH * WG_SIZE) {
                u32 fbv[MED_BATCH], pv[MED_BATCH];
                #pragma unroll
                for (u32 u = 0; u < MED_BATCH; ++u) {
                    fbv[u] = fb32[2u * (j + u * WG_SIZE)];
                    pv[u] = sieve_primes[j + u * WG_SIZE];
                }
                #pragma unroll
                for (u32 u = 0; u < MED_BATCH; ++u)
                    med_prime_f32(t32, fbv[u], pv[u], tb, qc7v, s7v, c1v);
            }
            for (; j < sieve_split; j += WG_SIZE)
                med_prime_f32(t32, fb32[2u * j], sieve_primes[j], tb, qc7v, s7v, c1v);
        }
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
    // Coalesced: word i * WG_SIZE + me (2026-11-18, sieve -1.5 ms per 1e11).
    // Any thread may read any SLM word once every atomic is done: after the
    // medium stage's barrier, or this one when that stage does not run.
    if (!(own_end < sieve_split)) barrier(CLK_LOCAL_MEM_FENCE);   // uniform
    for (u32 i = 0; i < WORDS_PER_THREAD; ++i) {
        const u32 w = i * WG_SIZE + me;
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
///         > 61 (wheel-30 marks; host: sieve_split >= mask_count).
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
        // 32-bit atomics on the containing u32 word (see mark_multiples_u32;
        // 64-bit atomic_and is absent from this ocloc's headers).
        if (p > 61UL && p < (1UL << 30))
            mark_multiples_whl(prime_map, (first - q_first) >> 1, (u32)p, q_odds, qfw, g);
        else
        mark_multiples_u32(prime_map, (first - q_first) >> 1, p, q_odds, g);
    }
}

// Subgroup width of the packed decode and primes per work-item; one packing
// group = LARGE_PACK_PPI * width primes (MUST match gb::LARGE_PACK_GROUP and
// gb::LARGE_PACK_PPI on the host).
#define LARGE_PACK_SG 32        // SIMD32 measured ~5% faster than 16
#ifndef LARGE_PACK_PPI
#define LARGE_PACK_PPI 16       // primes per work-item (multiple of 4; gap bytes = PPI/4 u32s): 4 -> 8 -> 16 = large_s -2.5% then -1%
#endif
#define LARGE_PACK_GROUP (LARGE_PACK_PPI * LARGE_PACK_SG)
#define LARGE_PACK_WORDS (LARGE_PACK_PPI / 4)
#if LARGE_PACK_WORDS != 4
#error "decode_large loads one uint4 of gaps per item: LARGE_PACK_PPI must be 16"
#endif
uint4 __builtin_IB_lsc_load_global_uint4(__global uint4* base, int immElemOff, enum LSC_LDCC cacheOpt);
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
    // The gap stream (~33 MB per segment, each word read once) is larger
    // than L3 (18 MB): a streaming hint keeps it from evicting the map under the
    // atomics. Full items take one 16-byte load, the list's last item the
    // plain bounded loads.
    u32 gw[LARGE_PACK_WORDS];
    if (item * LARGE_PACK_PPI + 4u * (LARGE_PACK_WORDS - 1u) < large_count) {
        const uint4 v = __builtin_IB_lsc_load_global_uint4((__global uint4*)gaps4 + item, 0,
                                                           (enum LSC_LDCC)LDCC_L1S_L3UC);
        gw[0] = v.x; gw[1] = v.y; gw[2] = v.z; gw[3] = v.w;
    } else {
        #pragma unroll
        for (u32 k = 0; k < LARGE_PACK_WORDS; ++k)
            gw[k] = (item * LARGE_PACK_PPI + 4u * k < large_count) ? gaps4[item * LARGE_PACK_WORDS + k] : 0u;
    }
    #pragma unroll
    for (u32 k = 0; k < LARGE_PACK_WORDS; ++k) {
        const u32 w = gw[k];
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
    const u32 qf15 = qfw % 15015u;                // q_first mod 15015 (uniform)
#if LARGE_SKR
    const u32 qfr = (u32)(q_first % (u64)SKR_R);  // q_first mod 17*19*23*29 (uniform)
    const u32 qfr2 = (u32)(q_first % 2022161UL);  // q_first mod 31*37*41*43 (uniform)
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
    // NOT unrolled (2026-11-18): the 16 inlined copies of the setup + mark
    // loop made the kernel instruction-fetch bound; a rolled loop over the
    // item's primes is large_s -5.8% (bench +3.2% long / +3.3% long-end).
    // Unroll 2 measured the same as 1; the sieve kernels' batch loops showed
    // no such effect.
    #pragma unroll 1
    for (u32 lane = 0; lane < LARGE_PACK_PPI; ++lane) {
        const u64 p = pv[lane];
        if (c0 + lane >= large_count) break;
        if (hi_range && p < (1UL << 30)) {
            const u32 p32 = (u32)p;
#if LARGE_MQ
            u32 qlo; float qhf;
            const u32 r = mod_ff_q(qf_split, p32, &qlo, &qhf);
#else
            const u32 r = mod_ff(qf_split, p32);
#endif
            u32 t = r ? p32 - r : 0u;
            t += (t & 1u) ? p32 : 0u;
            if (p32 >= (u32)q_odds) {                    // at most one hit (bit < p)
                const u32 b = t >> 1;
                if (b < (u32)q_odds) {
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
                    if ((CP15015[x >> 5] >> (x & 31u)) & 1u)
                        map_clear_w((volatile __global u32*)prime_map, b, g);
                }
                continue;
            }
#if LARGE_MQ
            if (p32 < MQ_MAXP && p32 >= (1u << 21)) {
                // m0 = (q_first + t) / p = Q + (r != 0) + (t >= p): Q = qhi 2^32 + qlo,
                // qhi from the float estimate (error << 2^31). p >= 2^21 keeps
                // qhi < 2^42 / 2^32 = 1024 (q_first < 2^63), so qhi * 1919493 < 2^32.
                // (Production: p >= the mid-window bound 2.5M; the GB_BUCKET_VERIFY
                // reference also runs the mid primes through here.)
                const u32 qhi = (u32)rint((qhf - (float)qlo) * 2.3283064365386963e-10f);
                const u32 ee = (r ? 1u : 0u) + ((t >= p32) ? 1u : 0u);
                const u32 m30 = (qhi * 16636u + qlo % 30030u + ee) % 30030u;
                const bool use1 = p32 < SKR_MAXP, use2 = p32 < SKR2_MAXP;
                const u32 mr = use1 ? (qhi * 150961u + qlo % SKR_R + ee) % SKR_R : 0u;
                const u32 mr2 = use2 ? (qhi * 1919493u % 2022161u + qlo % 2022161u + ee) % 2022161u : 0u;
                mark_multiples_mq(prime_map, t >> 1, p32, (u32)q_odds, m30, mr, mr2,
                                  use1, use2, g);
                continue;
            }
#endif
#if LARGE_SKR
            if (p32 < SKR2_MAXP) {
                mark_multiples_skr2(prime_map, (u64)(t >> 1), p32, q_odds, qfw, qfr, qfr2, g);
                continue;
            }
            if (p32 < SKR_MAXP) {
                mark_multiples_skr(prime_map, (u64)(t >> 1), p32, q_odds, qfw, qfr, g);
                continue;
            }
#endif
            mark_multiples_whl(prime_map, (u64)(t >> 1), p32, q_odds, qfw, g);
            continue;
        }
        const u64 first = first_odd_multiple(p, q_first);
        if (first > q_last) continue;        // p irrelevant for this segment
        if (p > 61UL && p < (1UL << 30))
            mark_multiples_whl(prime_map, (first - q_first) >> 1, (u32)p, q_odds, qfw, g);
        else
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
#define BKT_E 8                 // entries per lane per chunk
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

// Bucket entries are written once and read once, 3..125 segments later:
// cache-controlled LSC accesses keep that ~70 MB/segment stream from
// evicting the L3-resident prime map under the bucket marks.
ulong __builtin_IB_lsc_load_global_ulong(__global ulong* base, int immElemOff, enum LSC_LDCC cacheOpt);
/// Stores a bucket entry (plain store: uncached LSC stores measured 2-3.7x
/// slower).
/// @pre    p 8-byte aligned, inside the ring.
/// @post   *p = v.
inline void bkt_store(__global u64* p, const u64 v)
{
    *p = v;
}
/// Loads a bucket entry (L1S_L3UC: bypasses L3).
/// @pre    p 8-byte aligned; *p was written by an earlier kernel of the
///         same in-order list with a barrier in between (bucket entries and
///         hit lists are never read by the kernel that wrote them).
/// @return *p.
inline u64 bkt_load(const __global u64* p)
{
    return __builtin_IB_lsc_load_global_ulong((__global u64*)p, 0, (enum LSC_LDCC)LDCC_L1S_L3UC);
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
                map_clear_w((volatile __global u32*)prime_map, off, g);
            } else if (!is_last) {
                atomic_or(err, BKT_ERR_ENTRY);
            }
            if (is_last) continue;
            const u32 dl = (u32)delta, qf = (u32)q_odds_full;
            u32 noff;
            if (off >= dl) {                     // same hit, also in segment k+1
                d[e] = 1u; noff = off - dl;
            } else {
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
            {   // advance to the first multiplier coprime to 30 (<= 2 odd steps)
                u32 mr = (((u32)(q_first0 % (u64)WHEEL_W)) +
                          2u * (u32)(G % (u64)(WHEEL_W / 2))) % WHEEL_W;
                mr = (mr * WHL_INV[(u32)(p % (u64)WHEEL_W)]) % WHEEL_W;
                while (WHL_IDX[mr] == WHL_MISS) { G += p; mr += 2u; if (mr >= WHEEL_W) mr -= WHEEL_W; }
            }
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
/// @post   checked[w] has bit i set iff even seg_start + 2
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
                if (acc[k] == ~0UL) continue;      // this word fully verified
                const u64 w0 = word_idx + Tw + k;  // = wi + Tw, in bounds
                const u64 qw0 = prime_map[w0];
                const u64 qw1 = prime_map[w0 + 1];

                acc[k] |= r ? ((qw0 >> r) | (qw1 << (64UL - r))) : qw0;
            }
            if (all_full) {
                bool done = true;
                #pragma unroll
                for (u32 k = 0; k < WORDS_PER_ITEM; ++k) done &= (acc[k] == ~0UL);
                if (done) break;                   // all evens verified
            }
        }
    } else {
        for (u32 j = 0; j < gb_count; ++j) {
            const u32 p = gb_primes[j];
            if (p == 2) continue;   // p=2 only matters for n=4 (CPU fallback)

            const long T = ((long)seg_start - (long)p - (long)q_first) / 2;

            #pragma unroll
            for (u32 k = 0; k < WORDS_PER_ITEM; ++k) {
                if (acc[k] == ~0UL) continue;      // this word fully verified
                const long base = (long)((word_idx + k) * 64UL) + T;
                const long w0   = base >> 6;       // arithmetic shift, may be -1
                const u32  r    = (u32)(base & 63UL);

                const u64 qw0 = (w0 >= 0 && (u64)w0       < map_words) ? prime_map[w0]     : 0UL;
                const u64 qw1 = (w0 + 1 >= 0 && (u64)(w0 + 1) < map_words) ? prime_map[w0 + 1] : 0UL;

                acc[k] |= r ? ((qw0 >> r) | (qw1 << (64UL - r))) : qw0;
            }
            if (all_full) {
                bool done = true;
                #pragma unroll
                for (u32 k = 0; k < WORDS_PER_ITEM; ++k) done &= (acc[k] == ~0UL);
                if (done) break;                   // all evens verified
            }
        }
    }

    // ---- mask partial word, write, report leftovers --------------------------
    #pragma unroll
    for (u32 k = 0; k < WORDS_PER_ITEM; ++k) {
        const u64 wi = word_idx + k;
        if (evens_here[k] == 0UL) break;       // past segment end (words are ordered)
        u64 a = acc[k];
        if (evens_here[k] < 64UL) a &= (1UL << evens_here[k]) - 1UL;
        checked[wi] = a;

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
#define W32_EXIT_EVERY 1        // early-exit test every N candidates (power of 2)
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
        for (u32 j = 0; j < gb_count; ++j) {
            const int p = gb_primes[j];
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
            if ((j & (W32_EXIT_EVERY - 1u)) == W32_EXIT_EVERY - 1u &&
                (acc[0] & acc[1] & acc[2] & acc[3] & acc[4] & acc[5] & acc[6] & acc[7]) == ~0u)
                break;
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
            if ((a64[0] & a64[1] & a64[2] & a64[3]) == ~0UL) break;
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
        checked[wi] = a & valid[k];
        u64 miss = ~a & valid[k];
        while (miss) {
            const u32 b = (u32)ctz(miss);
            miss &= miss - 1UL;
            const u32 slot = atom_inc(leftover_count);
            if (slot < MAX_LEFTOVERS) leftovers[slot] = (u32)(wi * 64UL + b);
        }
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
#ifndef A_W64
#define A_W64 12
#endif
//                             14 -> 12 after C30_ROT (spill-free, +0.2%); 8/10/14/16 windows removed
/// Phase A of the two-phase checker (first cand_cap candidates).
/// @pre    as check_numbers; with twin = 1, gb_primes lists lower twin members.
/// @post   checked[] holds the partial result; every unresolved valid even is
///         in pending (count may exceed pending_cap: host fails closed).
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
                u32 all = acc[0];
                #pragma unroll
                for (u32 i = 1; i < 2 * A_W64; ++i) all &= acc[i];
                if (all == ~0u) {
                    done = true;
                    break;
                }
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
            u64 all64 = a64[0];
            #pragma unroll
            for (u32 k = 1; k < A_W64; ++k) all64 &= a64[k];
            if (all64 == ~0UL) break;
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
        checked[word_idx + k] = a & valid[k];
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

// CB_FLAT (2026-11-18, r6b): flattened check30_b probe loop (lanes refill
// themselves per even) on a small grid (CHECK30_B_GROUPS = 40, beside the
// next tile sieve): period 1.662 -> 1.634 ms per segment at 4e18
#ifndef ABL_BSKIP
#define ABL_BSKIP 0
#endif
#ifndef CB_FLAT
#define CB_FLAT 1
#endif
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

#define SPLIT_MAXD 2             // largest pattern shift (units of 6; host-checked)
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
#define C30_MAXD 4               // largest pattern shift (host-checked)
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
#if A_W64 != 12
// generic width (experiments): 2 A_W64 + 5 words from wb - 4
#define C30_WIN(S, wb, n) do { \
    if ((wb) >= 4 && (wb) + 2 * A_W64 + 4 <= (int)(n)) { \
        const __global u32* P_ = (S) + ((wb) - 4); \
        u32 t_[((2 * A_W64 + 5) + 7) / 8 * 8]; \
        _Pragma("unroll") for (u32 c_ = 0; c_ < (2 * A_W64 + 5) / 8; ++c_) vstore8(vload8(c_, P_), c_, t_); \
        _Pragma("unroll") for (u32 c_ = (2 * A_W64 + 5) / 8 * 8; c_ < 2 * A_W64 + 5; ++c_) t_[c_] = P_[c_]; \
        r4 = t_[0]; r3 = t_[1]; r2 = t_[2]; r1 = t_[3]; \
        _Pragma("unroll") for (u32 i_ = 0; i_ <= 2 * A_W64; ++i_) m[i_] = t_[4 + i_]; \
    } else { \
        _Pragma("unroll") for (u32 i_ = 0; i_ <= 2 * A_W64; ++i_) m[i_] = split_ld((S), (wb) + i_, (n)); \
        r1 = split_ld((S), (wb) - 1, (n)); r2 = split_ld((S), (wb) - 2, (n)); \
        r3 = split_ld((S), (wb) - 3, (n)); r4 = split_ld((S), (wb) - 4, (n)); \
    } } while (0)
#else
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
    else if ((mask) == 5u) C30_ORW1(5u); \
    else C30_ORW1(mask); } while (0)

// Phase A. One work-item = 64 * A_W64 evens of one class; items_pad items per
// class (a multiple of the work-group size: one class per work-group). The
// classes are interleaved over the groups (group g: class g % 15, block group
// g / 15), so concurrently running groups read nearby windows of the class
// arrays (phase A -11% with the 32-bit epilogue below).
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
// SIMD16 pinned (2026-11-18): the compiler's own choice measured 0.275 vs
// 0.269 ms per segment of phase A (+0.25% overall)
#ifndef CA_SG
#define CA_SG 16
#endif
// CA_PAIR (2026-11-18 r6b): two candidates of the same window word per
// iteration, merged by one 3-input OR (phase A 0.267 -> 0.263 ms per segment,
// +0.28% both bench windows); 2 = also pair across a one-word step (slower)
#ifndef CA_PAIR
#define CA_PAIR 1
#endif
#ifndef ABL_WIN
#define ABL_WIN 0
#endif
#ifndef CA_ONEWIN
#define CA_ONEWIN 0
#endif
#ifndef ABL_LD
#define ABL_LD 0
#endif
#ifndef ABL_CAND
#define ABL_CAND 0
#endif
#ifndef ABL_MASK1
#define ABL_MASK1 0
#endif
#ifndef ABL_ONEF
#define ABL_ONEF 0
#endif
#ifndef CA_SHF
#define CA_SHF 0
#endif
#ifndef ABL_NOPEND
#define ABL_NOPEND 0
#endif
#ifndef CA_PRO
#define CA_PRO 0
#endif
#ifndef ABL_FUN
#define ABL_FUN 0
#endif
#if CA_SG
__attribute__((intel_reqd_sub_group_size(CA_SG)))
#endif
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
    // class-interleaved groups: group g takes class g % 15 of block group
    // g / 15, so concurrently running groups read nearby map windows
    const u32 grp = get_group_id(0);
    const u32 cr = grp % 15u;
    const u32 blk = (grp / 15u) * get_local_size(0) + get_local_id(0);
    const u32 r = 2u * cr;
    const u32 e0 = ((r + 30u - sm30) % 30u) >> 1;      // first even index of the class
    const u32 ec = (cr < 15u && seg_evens > e0) ? (seg_evens - e0 + 14u) / 15u : 0u;
    const u32 mo = cr * 32u;
    const u32 k0 = blk * (A_W64 * 64u);
    const bool live = cr < 15u && k0 < ec;             // no early return: subgroup ops below

    u32 acc[2 * A_W64];
#if CA_PRO
    // full items (all but the last item of a class): every bit valid, no
    // per-word clamp (the general init below is ~10 ALU ops per word)
    if (live && k0 + 64u * A_W64 <= ec) {
        #pragma unroll
        for (u32 i = 0; i < 2 * A_W64; ++i) acc[i] = 0u;
    } else
#endif
    {
    #pragma unroll
    for (u32 i = 0; i < 2 * A_W64; ++i) {
        const u32 kb = k0 + 32u * i;
        const u32 n = (live && kb < ec) ? min(32u, ec - kb) : 0u;
        acc[i] = (n >= 32u) ? 0u : ~((1u << n) - 1u);  // invalid bits count as resolved
    }
    }

    if (live) {
        const u32 ns = meta[mo];
        for (u32 st = 0; st < ns; ++st) {
            const u32 be = meta[mo + 1u + 2u * st];
            const u32 sm = meta[mo + 2u + 2u * st];
            u32 j = be & 0xFFFFu;
            const u32 je = be >> 16;
            const u32 s = sm & 0xFFu, mask = ABL_MASK1 ? 1u : sm >> 8;
            const u32 k = W30_IDX[(r + 30u - s) % 30u];  // q class (host: a unit)
            const int vk = tb + 2 * (int)e0 - 2 * (int)((phik >> (4u * k)) & 15u);
            const __global u32* S = cmap32 + (u64)k * cw32;
            // every candidate of this plan segment is p = 30 a + s, so
            // U = U0 - floor(p / 30); for p < 2^16, floor(p / 30) =
            // (p * 34953) >> 20 exactly (error < 0.03 < 1/30)
            const int U0 = (vk - (int)s) / 30;         // exact: vk = s (mod 30)
            int U = U0 - (int)(((u32)cands[j] * 34953u) >> 20);
            int tw = (U >= 0) ? (U >> 5) : -((31 - U) >> 5);   // floor(U / 32)
#if ABL_WIN
            int wb = (int)((k0 >> 5) & (ABL_WIN - 1)) + 8 + tw;
#else
            int wb = (int)(k0 >> 5) + tw;
#endif
            u32 m[2 * A_W64 + 1];
            u32 r1, r2, r3, r4;
#if CA_ONEWIN
            // one window-load site (the reload branch below): forced by a
            // fake previous word 2 above (halves the kernel's code size)
            tw += 2; wb += 2;
#elif ABL_LD
            { const u32 z_ = ~((u32)wb & 0x80000000u); for (u32 i_ = 0; i_ <= 2 * A_W64; ++i_) m[i_] = z_ >> (i_ & 1u); r1 = r2 = r3 = r4 = z_; }
#else
            C30_WIN(S, wb, cw32);
            C30_ORWIN(mask);
#endif
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
#if ABL_LD
                    { const u32 z_ = ~((u32)wb & 0x80000000u); for (u32 i_ = 0; i_ <= 2 * A_W64; ++i_) m[i_] = z_ >> (i_ & 1u); r1 = r2 = r3 = r4 = z_; }
#else
                    C30_WIN(S, wb, cw32);
#endif
                    C30_ORWIN(mask);
                }
#if CA_PAIR
                // paired funnels (CA_PAIR): the next candidate in the same
                // window word shares the window; one 3-input OR merges both
                // funnels into acc (5 instead of 6 ALU ops per word)
                if (j + 1u < je) {
                    const int U2 = U0 - (int)(((u32)cands[j + 1u] * 34953u) >> 20);
                    const int tw2 = (U2 >= 0) ? (U2 >> 5) : -((31 - U2) >> 5);
                    if (tw2 == tw) {
                        const u32 rr1 = (u32)U & 31u, rr2 = (u32)U2 & 31u;
                        const u32 rl1 = 32u - rr1, rl2 = 32u - rr2;
                        const u32 hm1 = 0xFFFFFFFFu >> rr1, hm2 = 0xFFFFFFFFu >> rr2;
                        u32 rn1 = rotate(m[0], rl1), rn2 = rotate(m[0], rl2);
                        #pragma unroll
                        for (u32 i = 0; i < 2 * A_W64; ++i) {
                            const u32 rc1 = rn1, rc2 = rn2;
                            rn1 = rotate(m[i + 1], rl1);
                            rn2 = rotate(m[i + 1], rl2);
                            acc[i] |= bitselect(rn1, rc1, hm1) | bitselect(rn2, rc2, hm2);
                        }
                        j += 2u;
                        if (j >= je) break;
                        U = U0 - (int)(((u32)cands[j] * 34953u) >> 20);
                        continue;
                    }
#if CA_PAIR >= 2
                    if (tw2 == tw - 1) {                 // second one word lower: m[-1] = r1
                        const u32 rr1 = (u32)U & 31u, rr2 = (u32)U2 & 31u;
                        const u32 rl1 = 32u - rr1, rl2 = 32u - rr2;
                        const u32 hm1 = 0xFFFFFFFFu >> rr1, hm2 = 0xFFFFFFFFu >> rr2;
                        u32 rn1 = rotate(m[0], rl1), rn2 = rotate(c30_orp(r1, r2, mask), rl2);
                        #pragma unroll
                        for (u32 i = 0; i < 2 * A_W64; ++i) {
                            const u32 rc1 = rn1, rc2 = rn2;
                            rn1 = rotate(m[i + 1], rl1);
                            rn2 = rotate(m[i], rl2);
                            acc[i] |= bitselect(rn1, rc1, hm1) | bitselect(rn2, rc2, hm2);
                        }
                        j += 2u;
                        if (j >= je) break;
                        U = U0 - (int)(((u32)cands[j] * 34953u) >> 20);
                        continue;
                    }
#endif
                }
#endif
                const u32 rr = (u32)U & 31u;
                // funnel by rotates: R = m rotated right by rr; the low
                // 32 - rr bits of R[i] are m[i] >> rr, the high rr bits of
                // R[i + 1] are m[i + 1] << (32 - rr) — one select per word,
                // exact for rr = 0 too (hm = ~0) without a shift by 32.
                const u32 rl = 32u - rr;                // rotate(x, 32) == x
                const u32 hm = 0xFFFFFFFFu >> rr;
#if ABL_FUN == 1
                #pragma unroll
                for (u32 i = 0; i < 2 * A_W64; ++i) acc[i] |= m[i] >> rr;
#elif ABL_FUN == 2
                #pragma unroll
                for (u32 i = 0; i < 2 * A_W64; ++i) acc[i] |= rotate(m[i], rl);
#elif ABL_FUN == 3
                u32 rn = rotate(m[0], rl), rn2 = rotate(m[0], rl ^ 1u);
                #pragma unroll
                for (u32 i = 0; i < 2 * A_W64; ++i) {
                    const u32 rc = rn, rc2 = rn2;
                    rn = rotate(m[i + 1], rl);
                    rn2 = rotate(m[i + 1], rl ^ 1u);
                    acc[i] |= bitselect(rn, rc, hm);
                    acc[i] |= bitselect(rn2, rc2, hm ^ (hm >> 1));
                }
#elif CA_SHF
                // shift funnel (CA_SHF): rotates issue at ~half the rate
                // of shifts on Xe2; rr = 0 (no shift) as its own branch
                if (CA_SHF == 1 && rr == 0u) {                        // xor form: no common OR to sink
                    #pragma unroll
                    for (u32 i = 0; i < 2 * A_W64; ++i) acc[i] ^= m[i] & ~acc[i];
                } else {
                    #pragma unroll
                    for (u32 i = 0; i < 2 * A_W64; ++i) acc[i] |= (m[i] >> rr) | (m[i + 1] << rl);
                }
#else
                u32 rn = rotate(m[0], rl);
                #pragma unroll
                for (u32 i = 0; i < 2 * A_W64; ++i) {
                    const u32 rc = rn;
                    rn = rotate(m[i + 1], rl);
                    acc[i] |= bitselect(rn, rc, hm);
                }
#endif
                if (++j >= je) break;
#if ABL_ONEF
                break;
#endif
#if ABL_CAND
                U = U0 - (int)((j * 7u) & 63u);
#else
                U = U0 - (int)(((u32)cands[j] * 34953u) >> 20);
#endif
            }
        }
    }

    u32 cnt = 0;
    // the bits past ec (and every bit of a dead item) were set at the start,
    // so the misses are exactly the clear bits of acc
    if (write_checked) {
        #pragma unroll
        for (u32 i = 0; i < A_W64; ++i) {
            const u32 kb = k0 + 64u * i;
            if (!live || kb >= ec) continue;
            const u64 a = (u64)acc[2 * i] | ((u64)acc[2 * i + 1] << 32);
            const u32 n = min(64u, ec - kb);
            const u64 valid = (n >= 64u) ? ~0UL : ((1UL << n) - 1UL);
            checked[(u64)cr * chk_stride + (kb >> 6)] = a & valid;
        }
    }
    #pragma unroll
    for (u32 i = 0; i < 2 * A_W64; ++i) cnt += popcount(acc[i]);
    cnt = 2u * A_W64 * 32u - cnt;
    const u32 off = sub_group_scan_exclusive_add(cnt);
    const u32 tot = sub_group_reduce_add(cnt);
    u32 base = 0;
    if (get_sub_group_local_id() == 0 && tot) base = atomic_add(pending_count, tot);
    base = sub_group_broadcast(base, 0) + off;
    if (cnt && !ABL_NOPEND) {
        #pragma unroll
        for (u32 i = 0; i < 2 * A_W64; ++i) {
            u32 miss = ~acc[i];
            while (miss) {
                const u32 b = ctz(miss);
                miss &= miss - 1u;
                if (base < pending_cap) pending[base] = e0 + 15u * (k0 + 32u * i + b);
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
#if CB_FLAT
    // Flattened probe loop (CB_FLAT): each lane runs one probe batch per
    // iteration and refills itself with the next pending even as soon as
    // its current one is resolved or exhausted, so a subgroup no longer
    // waits for its slowest lane per even (the probes per even are
    // geometric: mean ~3 batches, max over a subgroup ~10).
#if ABL_BSKIP
    if (n > 0u) return;
#endif
    u32 i = get_global_id(0);
    u32 e = 0u, cr = 0u, ro = 0u, rc = 0u, j = 0u;
    bool have = false;
    for (;;) {
        if (!have) {
            if (i >= n) break;
            e = pending[i];
            i += get_global_size(0);
            cr = ((sm30 + 2u * (e % 15u)) % 30u) >> 1;
            ro = rtab[cr]; rc = rtab[15u + cr]; j = 0u;
            have = true;
        }
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
        if (hit) {
            const u32 rp = (u32)rest[ro + j + (u32)ctz(hit)];
            have = false;
            if (rp >= ridge_thresh) {
                const u32 slot = atomic_add((volatile __global u32*)ridge, 1u);
                if (slot < RIDGE_CAP) {
                    ridge[1 + 2u * slot] = seg_start + 2UL * (u64)e;
                    ridge[2 + 2u * slot] = rp;
                }
            }
            if (rp >= tail_t) {
                atomic_add(hist_p + (rp >> 1), 1u);
                if ((want_p[rp >> 6] >> ((rp >> 1) & 31u)) & 1u) atomic_min(first_p + (rp >> 1), e);
            }
            if (write_checked) {
                const u32 e0 = ((2u * cr + 30u - sm30) % 30u) >> 1;
                const u32 k = (e - e0) / 15u;
                atomic_or((volatile __global u32*)(checked + (u64)cr * chk_stride) + (k >> 5),
                          1u << (k & 31u));
            }
        } else {
            j += PROBE_BATCH;
            if (j >= rc) {
                have = false;
                const u32 slot = atom_inc(leftover_count);
                if (slot < MAX_LEFTOVERS) leftovers[slot] = e;
            }
        }
    }
#else
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
#endif
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
#define X_KMAX 128               // max K_r (SLM histogram span; host-checked)
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
#define C30_CW 4                 // words per lane (host: cw32 % C30_CW == 0; 8: slower)
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
// Gap walk: the low bins (d < G30_CBINS, ~94% of the gaps) are kept once per
// SIMD lane (+1.4%), so the lanes of one SIMD instruction never queue on the
// same SLM word. count30 is compiled at G30_SIMD (required sub-group size)
// with one column per lane: lane l of every sub-group uses column l. Column 0
// is the base histogram's own bins 0..G30_CBINS-1; columns 1.. follow it at
// G30_HBINS and are summed into the base bins at the flush.
#ifndef G30_SIMD
#define G30_SIMD 16u
#endif
#define G30_CBINS 64u
#define G30_LH (G30_HBINS + (G30_SIMD - 1u) * G30_CBINS)
// Block walk (2026-11-18): one loop iteration per non-empty block of 30
// instead of per prime. The gap into the block's first prime is binned as
// before; a block with two or more primes adds one count to its pattern bin
// (256, after the histogram: hist[G30_HBINS + x]) and the host adds the
// pattern's internal gaps (fixed by the segment's phases). Pattern index of
// the slot nibble mask m (slot s at bit 4 s): bits s0 s4 s1 s5 s2 s6 s3 s7.
// The walk maximum covers the gaps into blocks only (an internal gap is
// < 30; the host settles a segment whose largest such gap is not larger
// than the largest internal one: a per-block table lookup cost the walk its
// gain).
#ifndef G30_BLK
#define G30_BLK 1
#endif
#define G30_PBINS 256u
inline u32 g30_pat(const u32 m)
{
    u32 x = (m | (m >> 15)) & 0x3333u;
    x = (x | (x >> 2)) & 0x0F0Fu;
    return (x | (x >> 4)) & 0xFFu;
}
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

__attribute__((intel_reqd_sub_group_size(G30_SIMD)))
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
#if G30_BLK
    __local u32 lh[G30_LH + G30_PBINS];      // + pattern bins
#else
    __local u32 lh[G30_LH];
#endif
    __local u64 red[G30_MAX_SG];
    const u32 lid = get_local_id(0);
    if (lid < 13u) sh[lid] = lid == 11u ? ~0u : 0u;
#if G30_BLK
    if (walk)
        for (u32 i = lid; i < G30_LH + G30_PBINS; i += get_local_size(0)) lh[i] = 0u;
#else
    if (walk)
        for (u32 i = lid; i < G30_LH; i += get_local_size(0)) lh[i] = 0u;
#endif
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
    const u32 scol = get_sub_group_local_id();   // < G30_SIMD: one column per lane
    const u32 gcol = scol ? G30_HBINS + (scol - 1u) * G30_CBINS : 0u;   // my column of low bins
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
            // Each word's 8 class words are nibble-transposed (an 8 x 8 matrix
            // of 4-bit blocks, 3 swap stages): group k then holds blocks
            // 4 k .. 4 k + 3 with slot s of block 4 k + j at bit 4 s + j, the
            // slots rotated by ord & 15 so ascending slots are ascending
            // phases (the classes in phase order are a rotation of 0..7, host-
            // checked). A block refill is a 3-level select of its group, a
            // shift and a mask (was 8 bit gathers), the phase a nibble of phis.
            const u32 r0 = ord & 15u;
            #pragma unroll
            for (u32 i = 0; i < C30_CW; ++i) {
                u32 W[8];
                #pragma unroll
                for (u32 c = 0; c < 8; ++c) W[c] = v[c][i];
                #pragma unroll
                for (u32 c = 0; c < 4; ++c) {
                    const u32 t = ((W[c] >> 16) ^ W[c + 4]) & 0x0000FFFFu;
                    W[c + 4] ^= t; W[c] ^= t << 16;
                }
                #pragma unroll
                for (u32 c = 0; c < 8; c += (c & 1u) ? 3u : 1u) {
                    const u32 t = ((W[c] >> 8) ^ W[c + 2]) & 0x00FF00FFu;
                    W[c + 2] ^= t; W[c] ^= t << 8;
                }
                #pragma unroll
                for (u32 c = 0; c < 8; c += 2) {
                    const u32 t = ((W[c] >> 4) ^ W[c + 1]) & 0x0F0F0F0Fu;
                    W[c + 1] ^= t; W[c] ^= t << 4;
                }
                #pragma unroll
                for (u32 k = 0; k < 8; ++k) W[k] = rotate(W[k], 32u - 4u * r0);
#if G30_BLK
                u32 nb = any[i];
                const u32 h0 = 15u * 32u * (w + i);
                while (nb) {
                    const u32 b = ctz(nb);
                    nb &= nb - 1u;
                    const u32 k = b >> 2;
                    const u32 x01 = (k & 1u) ? W[1] : W[0], x23 = (k & 1u) ? W[3] : W[2];
                    const u32 x45 = (k & 1u) ? W[5] : W[4], x67 = (k & 1u) ? W[7] : W[6];
                    const u32 x03 = (k & 2u) ? x23 : x01, x47 = (k & 2u) ? x67 : x45;
                    const u32 m = (((k & 4u) ? x47 : x03) >> (b & 3u)) & 0x11111111u;
                    const u32 hb = h0 + 15u * b;
                    const u32 h = hb + ((phis >> ctz(m)) & 15u);
                    const u32 d = h - hp;
                    hp = hb + ((phis >> (31u - clz(m))) & 15u);
                    atomic_inc(&lh[d < G30_CBINS ? gcol + d : min(d, G30_HBINS - 1u)]);
                    if (d > dmax) { dmax = d; dpos = h; }
                    if (m & (m - 1u))                   // internal gaps: the pattern bin
                        atomic_inc(&lh[G30_LH + g30_pat(m)]);
                }
#else
                u32 nb = any[i], m = 0u, hb = 0u;
                const u32 h0 = 15u * 32u * (w + i);
                while (nb | m) {
                    if (!m) {
                        const u32 b = ctz(nb);
                        nb &= nb - 1u;
                        const u32 k = b >> 2;
                        const u32 x01 = (k & 1u) ? W[1] : W[0], x23 = (k & 1u) ? W[3] : W[2];
                        const u32 x45 = (k & 1u) ? W[5] : W[4], x67 = (k & 1u) ? W[7] : W[6];
                        const u32 x03 = (k & 2u) ? x23 : x01, x47 = (k & 2u) ? x67 : x45;
                        m = (((k & 4u) ? x47 : x03) >> (b & 3u)) & 0x11111111u;
                        hb = h0 + 15u * b;
                    }
                    const u32 s4 = ctz(m);
                    m &= m - 1u;
                    const u32 h = hb + ((phis >> s4) & 15u);
                    const u32 d = h - hp;
                    hp = h;
                    atomic_inc(&lh[d < G30_CBINS ? gcol + d : min(d, G30_HBINS - 1u)]);
                    if (d > dmax) { dmax = d; dpos = h; }
                }
#endif
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
        for (u32 i = lid; i < G30_HBINS; i += get_local_size(0)) {
            u32 t = lh[i];
            if (i < G30_CBINS)
                for (u32 k = 1; k < G30_SIMD; ++k) t += lh[G30_HBINS + (k - 1u) * G30_CBINS + i];
            if (t) atomic_add(hist + i, t);
        }
#if G30_BLK
        for (u32 i = lid; i < G30_PBINS; i += get_local_size(0))
            if (lh[G30_LH + i]) atomic_add(hist + G30_HBINS + i, lh[G30_LH + i]);
#endif
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
#define XS_W8 8
#define XS_W6 9
#define XS_W4 12
#define XS_W3 14
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
#define HL_MAX_K 128
#ifndef HL_SKR
#define HL_SKR 1                 // harvest skips hits p m, m divisible by 17..29 (LARGE_SKR)
#endif
#ifndef HL_SKR2
#define HL_SKR2 1                // ... and by 31..43 (requires HL_SKR)
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
#if HL_SKR
    __local u32 lstp[WHL_N];                 // WHL_SF: step + M[w] mod 17/19/23/29
#if HL_SKR2
    __local u32 lsf2[WHL_N];                 // WHL_SF2: M[w] mod 31/37/41/43
#endif
#else
    __local uchar lstp[WHL_N];
#endif
    const u32 lid = get_local_id(0), wg = get_local_size(0);
    const u32 grp = get_group_id(0), ngrp = get_num_groups(0);
#if HL_SKR
    for (u32 w = lid; w < WHL_N; w += wg) lstp[w] = WHL_SF[w];
    const u32 qfr = (u32)(q_first % (u64)SKR_R);
#if HL_SKR2
    for (u32 w = lid; w < WHL_N; w += wg) lsf2[w] = WHL_SF2[w];
    const u32 qfr2 = (u32)(q_first % 2022161UL);
#endif
#else
    for (u32 w = lid; w < WHL_N; w += wg) lstp[w] = WHL_STEP[w];
#endif
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
#if HL_SKR
            // skip hits p m with m divisible by 17..29 (LARGE_SKR): q mod SKR_R
            // x mod SKR_R from its u32 halves (x < 2^40: hi * (2^32 mod R) < 2^26)
            const u32 xr = ((u32)(x >> 32) * 150961u + (u32)x % SKR_R) % SKR_R;
            u32 T = skr_target_q((qfr + 2u * xr) % SKR_R, p, lstp[w] >> 4);
#if HL_SKR2
            const u32 xr2 = ((u32)(x >> 32) * 1919493u % 2022161u + (u32)x % 2022161u) % 2022161u;
            u32 T2 = skr2_target_q((qfr2 + 2u * xr2) % 2022161u, p, lsf2[w]);
#endif
#endif
            // (j, off) of x by a float quotient, exact by the fix-up loops
            // (off in (-delta, 2 delta) fits an int: 2 delta < 2^31)
            for (;;) {
                u32 j = (u32)(((float)(u32)(x >> 32) * 4294967296.0f + (float)(u32)x) * inv_delta);
                int o = (int)((u32)x - j * delta);
                while (o < 0) { o += (int)delta; --j; }
                while (o >= (int)delta) { o -= (int)delta; ++j; }
                if (j > nseg) break;
                const u32 off = (u32)o;
#if HL_SKR
                const u32 e = lstp[w];
                if (((((e ^ T) | SKR_H) - SKR_L) & SKR_H) == SKR_H
#if HL_SKR2
                    && ((((lsf2[w] ^ T2) | SKR2_H) - SKR2_L) & SKR2_H) == SKR2_H
#endif
                    ) {
#endif
                if (j < nseg && off < HL_QO(j)) {
                    const u32 idx = atomic_inc(&lpos[j]);
                    if (idx < capg) lists[((size_t)j * ngrp + grp) * capg + idx] = off;
                }
                if (j >= 1u && j - 1u < nseg && off + delta < HL_QO(j - 1u)) {
                    const u32 idx = atomic_inc(&lpos[j - 1u]);
                    if (idx < capg) lists[((size_t)(j - 1u) * ngrp + grp) * capg + idx] = off + delta;
                }
#if HL_SKR
                }
                x += (u64)p * (e & 15u);
                if (++w == WHL_N) {
                    w = 0u; T = skr_wrap(T);
#if HL_SKR2
                    T2 = skr2_wrap(T2);
#endif
                }
#else
                x += (u64)p * lstp[w];
                if (++w == WHL_N) w = 0u;
#endif
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


// ----------------------------------------------------------------------------
// Prime-multiplier filter for the hit lists (HL_MT, round 4). A hit q = p m
// of a hit-listed prime p (p > q^(1/3): m = q / p < p^2) clears a bit that is
// still set only if q has no prime factor below p, i.e. iff m is prime: a
// composite m has a factor s <= sqrt(m) < p, and q is cleared by s's own
// marks (mask stage, tile sieve, mid windows, packed kernel or hit lists all
// mark every multiple s m' whose m' has no factor below s). So the harvest
// files only the hits with m prime: ~23 % of the coprime multipliers at
// m ~ 2e9 .. 2e10 (the map is the same bit for bit).
// mt = bit table over the multipliers coprime to 30030: bit (k - k_lo) 5760 +
// w <-> m = 30030 k + WHL_M[w] is prime. Consecutive primes p read nearby
// m = q / p (dm ~ q gap / p^2), so a batch streams the table about once.
// ----------------------------------------------------------------------------
__constant ushort WHL_M[5760] = {
    1, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
    79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
    163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241,
    251, 257, 263, 269, 271, 277, 281, 283, 289, 293, 307, 311, 313, 317, 323, 331,
    337, 347, 349, 353, 359, 361, 367, 373, 379, 383, 389, 391, 397, 401, 409, 419,
    421, 431, 433, 437, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 493, 499,
    503, 509, 521, 523, 527, 529, 541, 547, 551, 557, 563, 569, 571, 577, 587, 589,
    593, 599, 601, 607, 613, 617, 619, 629, 631, 641, 643, 647, 653, 659, 661, 667,
    673, 677, 683, 691, 697, 701, 703, 709, 713, 719, 727, 731, 733, 739, 743, 751,
    757, 761, 769, 773, 779, 787, 797, 799, 809, 811, 817, 821, 823, 827, 829, 839,
    841, 851, 853, 857, 859, 863, 877, 881, 883, 887, 893, 899, 901, 907, 911, 919,
    929, 937, 941, 943, 947, 953, 961, 967, 971, 977, 983, 989, 991, 997, 1003, 1007,
    1009, 1013, 1019, 1021, 1031, 1033, 1037, 1039, 1049, 1051, 1061, 1063, 1069, 1073, 1081, 1087,
    1091, 1093, 1097, 1103, 1109, 1117, 1121, 1123, 1129, 1139, 1147, 1151, 1153, 1159, 1163, 1171,
    1181, 1187, 1189, 1193, 1201, 1207, 1213, 1217, 1219, 1223, 1229, 1231, 1237, 1241, 1247, 1249,
    1259, 1271, 1273, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1333,
    1343, 1349, 1357, 1361, 1363, 1367, 1369, 1373, 1381, 1387, 1399, 1403, 1409, 1411, 1423, 1427,
    1429, 1433, 1439, 1447, 1451, 1453, 1457, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1501,
    1511, 1513, 1517, 1523, 1531, 1537, 1541, 1543, 1549, 1553, 1559, 1567, 1571, 1577, 1579, 1583,
    1591, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1633, 1637, 1643, 1649, 1657, 1663, 1667,
    1669, 1679, 1681, 1691, 1693, 1697, 1699, 1709, 1711, 1717, 1721, 1723, 1733, 1739, 1741, 1747,
    1751, 1753, 1759, 1763, 1769, 1777, 1783, 1787, 1789, 1801, 1811, 1817, 1819, 1823, 1829, 1831,
    1843, 1847, 1849, 1853, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1891, 1901, 1907, 1909, 1913,
    1919, 1921, 1927, 1931, 1933, 1943, 1949, 1951, 1957, 1961, 1973, 1979, 1987, 1993, 1997, 1999,
    2003, 2011, 2017, 2021, 2027, 2029, 2033, 2039, 2047, 2053, 2059, 2063, 2069, 2071, 2077, 2081,
    2083, 2087, 2089, 2099, 2111, 2113, 2117, 2129, 2131, 2137, 2141, 2143, 2147, 2153, 2159, 2161,
    2173, 2179, 2183, 2201, 2203, 2207, 2209, 2213, 2221, 2227, 2231, 2237, 2239, 2243, 2251, 2257,
    2263, 2267, 2269, 2273, 2279, 2281, 2287, 2291, 2293, 2297, 2309, 2311, 2323, 2329, 2333, 2339,
    2341, 2347, 2351, 2357, 2363, 2369, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2407, 2411, 2413,
    2417, 2419, 2423, 2437, 2441, 2447, 2449, 2459, 2461, 2467, 2473, 2477, 2479, 2489, 2491, 2501,
    2503, 2507, 2521, 2531, 2533, 2537, 2539, 2543, 2549, 2551, 2557, 2567, 2573, 2579, 2581, 2591,
    2593, 2599, 2603, 2609, 2617, 2621, 2623, 2627, 2633, 2641, 2647, 2657, 2659, 2663, 2669, 2671,
    2677, 2683, 2687, 2689, 2693, 2699, 2701, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2747, 2749,
    2753, 2759, 2767, 2771, 2773, 2777, 2789, 2791, 2797, 2801, 2803, 2809, 2813, 2819, 2831, 2833,
    2837, 2839, 2843, 2851, 2857, 2861, 2867, 2869, 2879, 2881, 2887, 2897, 2903, 2909, 2911, 2917,
    2921, 2923, 2927, 2929, 2939, 2941, 2953, 2957, 2963, 2969, 2971, 2983, 2987, 2993, 2999, 3001,
    3007, 3011, 3013, 3019, 3023, 3037, 3041, 3043, 3049, 3053, 3061, 3067, 3071, 3077, 3079, 3083,
    3089, 3097, 3103, 3109, 3119, 3121, 3127, 3131, 3137, 3139, 3149, 3151, 3161, 3163, 3167, 3169,
    3173, 3181, 3187, 3191, 3193, 3197, 3203, 3209, 3217, 3221, 3229, 3233, 3239, 3247, 3251, 3253,
    3257, 3259, 3271, 3277, 3281, 3287, 3293, 3299, 3301, 3307, 3313, 3317, 3319, 3323, 3329, 3331,
    3337, 3343, 3347, 3349, 3359, 3361, 3371, 3373, 3379, 3383, 3389, 3391, 3397, 3401, 3403, 3407,
    3413, 3427, 3431, 3433, 3439, 3449, 3457, 3461, 3463, 3467, 3469, 3473, 3481, 3491, 3499, 3503,
    3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3551, 3557, 3559, 3569, 3571, 3581, 3583, 3587,
    3589, 3593, 3599, 3607, 3611, 3613, 3617, 3623, 3629, 3631, 3637, 3643, 3649, 3659, 3667, 3671,
    3673, 3677, 3683, 3691, 3697, 3701, 3709, 3713, 3719, 3721, 3727, 3733, 3737, 3739, 3743, 3749,
    3761, 3763, 3767, 3769, 3779, 3781, 3791, 3793, 3797, 3799, 3803, 3811, 3821, 3823, 3827, 3833,
    3841, 3847, 3851, 3853, 3859, 3863, 3869, 3877, 3881, 3889, 3893, 3901, 3907, 3911, 3917, 3919,
    3923, 3929, 3931, 3937, 3943, 3947, 3953, 3959, 3961, 3967, 3973, 3977, 3979, 3989, 4001, 4003,
    4007, 4009, 4013, 4019, 4021, 4027, 4031, 4033, 4049, 4051, 4057, 4061, 4063, 4073, 4079, 4087,
    4091, 4093, 4097, 4099, 4111, 4117, 4127, 4129, 4133, 4139, 4141, 4153, 4157, 4159, 4163, 4171,
    4177, 4181, 4183, 4187, 4189, 4201, 4211, 4217, 4219, 4223, 4229, 4231, 4237, 4241, 4243, 4247,
    4253, 4259, 4261, 4267, 4271, 4273, 4283, 4289, 4297, 4307, 4309, 4313, 4321, 4327, 4331, 4337,
    4339, 4343, 4349, 4351, 4357, 4363, 4369, 4373, 4379, 4387, 4391, 4393, 4397, 4399, 4409, 4421,
    4423, 4427, 4429, 4439, 4441, 4447, 4451, 4453, 4457, 4463, 4469, 4471, 4481, 4483, 4489, 4493,
    4507, 4513, 4517, 4519, 4523, 4531, 4541, 4547, 4549, 4553, 4559, 4561, 4567, 4573, 4577, 4579,
    4583, 4591, 4597, 4601, 4603, 4607, 4619, 4621, 4633, 4637, 4639, 4643, 4649, 4651, 4657, 4661,
    4663, 4673, 4679, 4681, 4687, 4691, 4699, 4703, 4709, 4717, 4721, 4723, 4727, 4729, 4733, 4747,
    4751, 4757, 4759, 4769, 4777, 4783, 4787, 4789, 4793, 4799, 4801, 4811, 4813, 4817, 4819, 4831,
    4841, 4843, 4847, 4853, 4859, 4861, 4867, 4871, 4877, 4883, 4889, 4891, 4897, 4903, 4909, 4913,
    4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4981, 4987, 4993, 4997, 4999, 5003,
    5009, 5011, 5017, 5021, 5023, 5029, 5039, 5041, 5051, 5053, 5059, 5063, 5069, 5077, 5081, 5087,
    5099, 5101, 5107, 5111, 5113, 5119, 5123, 5129, 5141, 5143, 5147, 5149, 5153, 5167, 5171, 5177,
    5179, 5183, 5189, 5191, 5197, 5207, 5209, 5219, 5221, 5227, 5231, 5233, 5237, 5249, 5251, 5261,
    5263, 5267, 5273, 5279, 5281, 5287, 5293, 5297, 5303, 5309, 5311, 5321, 5323, 5329, 5333, 5339,
    5347, 5351, 5353, 5359, 5363, 5371, 5377, 5381, 5387, 5389, 5393, 5399, 5407, 5413, 5417, 5419,
    5429, 5431, 5437, 5441, 5443, 5449, 5459, 5461, 5471, 5477, 5479, 5483, 5491, 5497, 5501, 5503,
    5507, 5513, 5519, 5521, 5527, 5531, 5539, 5543, 5549, 5557, 5561, 5563, 5567, 5569, 5573, 5581,
    5587, 5591, 5597, 5609, 5611, 5617, 5623, 5627, 5633, 5639, 5641, 5647, 5651, 5653, 5657, 5659,
    5669, 5671, 5683, 5689, 5693, 5699, 5701, 5711, 5713, 5717, 5723, 5729, 5737, 5741, 5743, 5749,
    5767, 5771, 5773, 5777, 5779, 5783, 5791, 5801, 5807, 5809, 5813, 5821, 5827, 5833, 5839, 5843,
    5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5891, 5893, 5897, 5899, 5903, 5909, 5911, 5917,
    5921, 5923, 5927, 5933, 5939, 5947, 5953, 5959, 5963, 5969, 5977, 5981, 5983, 5987, 5989, 6001,
    6007, 6011, 6023, 6029, 6031, 6037, 6043, 6047, 6049, 6053, 6059, 6067, 6073, 6077, 6079, 6089,
    6091, 6101, 6103, 6107, 6109, 6113, 6119, 6121, 6131, 6133, 6137, 6143, 6151, 6157, 6161, 6163,
    6169, 6173, 6179, 6187, 6191, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6233, 6239, 6241, 6247,
    6257, 6263, 6269, 6271, 6277, 6283, 6287, 6289, 6299, 6301, 6311, 6313, 6317, 6319, 6323, 6329,
    6337, 6341, 6343, 6353, 6359, 6361, 6367, 6371, 6373, 6379, 6389, 6397, 6401, 6403, 6407, 6421,
    6427, 6431, 6437, 6439, 6443, 6449, 6451, 6463, 6467, 6469, 6473, 6481, 6491, 6493, 6497, 6499,
    6509, 6511, 6521, 6527, 6529, 6533, 6541, 6547, 6551, 6553, 6557, 6563, 6569, 6571, 6577, 6581,
    6583, 6593, 6599, 6607, 6613, 6619, 6623, 6631, 6637, 6641, 6647, 6649, 6653, 6659, 6661, 6667,
    6673, 6679, 6683, 6689, 6691, 6697, 6701, 6703, 6707, 6709, 6719, 6731, 6733, 6737, 6739, 6749,
    6751, 6757, 6761, 6763, 6767, 6779, 6781, 6791, 6793, 6803, 6817, 6821, 6823, 6827, 6829, 6833,
    6841, 6847, 6857, 6859, 6863, 6869, 6871, 6883, 6887, 6889, 6893, 6899, 6901, 6907, 6911, 6913,
    6917, 6931, 6943, 6947, 6949, 6953, 6959, 6961, 6967, 6971, 6973, 6977, 6983, 6989, 6991, 6997,
    7001, 7003, 7009, 7013, 7019, 7027, 7031, 7037, 7039, 7043, 7057, 7061, 7067, 7069, 7079, 7081,
    7087, 7093, 7097, 7099, 7103, 7109, 7121, 7123, 7127, 7129, 7141, 7151, 7153, 7157, 7159, 7169,
    7171, 7177, 7181, 7187, 7193, 7199, 7201, 7207, 7211, 7213, 7219, 7223, 7229, 7237, 7243, 7247,
    7253, 7261, 7277, 7279, 7283, 7289, 7291, 7297, 7303, 7307, 7309, 7313, 7321, 7327, 7331, 7333,
    7339, 7349, 7351, 7361, 7363, 7367, 7369, 7373, 7379, 7387, 7391, 7393, 7409, 7411, 7417, 7421,
    7429, 7433, 7439, 7451, 7453, 7457, 7459, 7463, 7471, 7477, 7481, 7487, 7489, 7493, 7499, 7507,
    7517, 7519, 7523, 7529, 7531, 7537, 7541, 7543, 7547, 7549, 7559, 7561, 7571, 7573, 7577, 7583,
    7589, 7591, 7597, 7603, 7607, 7613, 7619, 7621, 7627, 7633, 7639, 7643, 7649, 7661, 7663, 7669,
    7673, 7681, 7687, 7691, 7697, 7699, 7703, 7717, 7723, 7727, 7729, 7739, 7741, 7747, 7751, 7753,
    7757, 7759, 7769, 7771, 7781, 7783, 7789, 7793, 7801, 7807, 7811, 7817, 7823, 7829, 7831, 7837,
    7841, 7849, 7853, 7859, 7867, 7871, 7873, 7877, 7879, 7883, 7897, 7901, 7907, 7913, 7919, 7921,
    7927, 7933, 7937, 7939, 7949, 7951, 7957, 7961, 7963, 7967, 7979, 7981, 7991, 7993, 7999, 8003,
    8009, 8011, 8017, 8023, 8027, 8033, 8039, 8051, 8053, 8059, 8069, 8077, 8081, 8083, 8087, 8089,
    8093, 8101, 8111, 8117, 8119, 8123, 8131, 8137, 8143, 8147, 8149, 8153, 8159, 8161, 8167, 8171,
    8179, 8189, 8191, 8201, 8207, 8209, 8213, 8219, 8221, 8227, 8231, 8233, 8237, 8243, 8249, 8251,
    8257, 8263, 8269, 8273, 8279, 8287, 8291, 8293, 8297, 8299, 8303, 8311, 8317, 8321, 8329, 8339,
    8341, 8347, 8353, 8357, 8363, 8369, 8377, 8381, 8383, 8387, 8389, 8399, 8401, 8413, 8417, 8419,
    8423, 8429, 8431, 8441, 8443, 8447, 8453, 8461, 8467, 8471, 8473, 8479, 8483, 8497, 8501, 8507,
    8509, 8513, 8521, 8527, 8531, 8537, 8539, 8543, 8549, 8551, 8557, 8563, 8573, 8579, 8581, 8587,
    8597, 8599, 8609, 8611, 8621, 8623, 8627, 8629, 8633, 8639, 8641, 8647, 8651, 8653, 8663, 8669,
    8677, 8681, 8683, 8689, 8693, 8699, 8707, 8711, 8713, 8717, 8719, 8731, 8737, 8741, 8747, 8753,
    8759, 8761, 8773, 8777, 8779, 8783, 8791, 8797, 8803, 8807, 8809, 8819, 8821, 8831, 8837, 8839,
    8843, 8849, 8851, 8857, 8861, 8863, 8867, 8873, 8881, 8887, 8891, 8893, 8903, 8909, 8917, 8923,
    8927, 8929, 8933, 8941, 8947, 8951, 8959, 8963, 8969, 8971, 8977, 8989, 8993, 8999, 9001, 9007,
    9011, 9013, 9017, 9019, 9029, 9041, 9043, 9047, 9049, 9059, 9067, 9071, 9073, 9077, 9083, 9089,
    9091, 9101, 9103, 9109, 9127, 9131, 9133, 9137, 9143, 9151, 9157, 9161, 9167, 9169, 9173, 9179,
    9181, 9187, 9193, 9197, 9199, 9203, 9209, 9211, 9221, 9223, 9227, 9239, 9241, 9253, 9257, 9259,
    9263, 9271, 9277, 9281, 9283, 9287, 9293, 9299, 9301, 9307, 9311, 9313, 9319, 9323, 9329, 9337,
    9341, 9343, 9349, 9353, 9367, 9371, 9377, 9379, 9389, 9391, 9397, 9403, 9407, 9409, 9413, 9419,
    9421, 9431, 9433, 9437, 9439, 9461, 9463, 9467, 9469, 9473, 9479, 9481, 9487, 9491, 9497, 9509,
    9511, 9517, 9521, 9523, 9533, 9539, 9547, 9551, 9553, 9557, 9563, 9571, 9577, 9587, 9589, 9593,
    9599, 9601, 9613, 9617, 9619, 9623, 9629, 9631, 9637, 9641, 9643, 9649, 9661, 9671, 9673, 9677,
    9679, 9683, 9689, 9697, 9701, 9703, 9707, 9719, 9721, 9727, 9731, 9733, 9739, 9743, 9749, 9761,
    9767, 9769, 9773, 9781, 9787, 9791, 9797, 9799, 9803, 9809, 9811, 9817, 9827, 9829, 9833, 9839,
    9847, 9851, 9853, 9857, 9859, 9869, 9871, 9881, 9883, 9887, 9899, 9901, 9907, 9913, 9917, 9923,
    9929, 9931, 9937, 9941, 9943, 9949, 9953, 9959, 9967, 9973, 9979, 9983, 9991, 10001, 10007, 10009,
    10013, 10019, 10027, 10033, 10037, 10039, 10051, 10057, 10061, 10063, 10067, 10069, 10079, 10081, 10091, 10093,
    10097, 10099, 10103, 10111, 10117, 10121, 10123, 10133, 10139, 10141, 10147, 10151, 10159, 10163, 10169, 10177,
    10181, 10183, 10187, 10189, 10193, 10201, 10207, 10211, 10217, 10223, 10229, 10237, 10243, 10247, 10249, 10253,
    10259, 10261, 10267, 10271, 10273, 10277, 10279, 10289, 10291, 10301, 10303, 10313, 10319, 10321, 10327, 10331,
    10333, 10337, 10343, 10349, 10357, 10363, 10369, 10379, 10391, 10393, 10397, 10399, 10403, 10411, 10421, 10427,
    10429, 10433, 10441, 10447, 10453, 10457, 10459, 10463, 10469, 10471, 10477, 10481, 10487, 10489, 10499, 10501,
    10511, 10513, 10519, 10523, 10529, 10531, 10537, 10541, 10547, 10553, 10559, 10561, 10567, 10573, 10579, 10583,
    10589, 10597, 10601, 10603, 10607, 10609, 10613, 10627, 10631, 10639, 10643, 10649, 10651, 10657, 10663, 10667,
    10669, 10679, 10687, 10691, 10693, 10697, 10709, 10711, 10721, 10723, 10727, 10729, 10733, 10739, 10741, 10753,
    10757, 10763, 10771, 10781, 10783, 10789, 10793, 10799, 10807, 10811, 10817, 10819, 10823, 10831, 10837, 10841,
    10847, 10849, 10853, 10859, 10861, 10867, 10873, 10877, 10883, 10889, 10891, 10897, 10903, 10909, 10919, 10921,
    10931, 10937, 10939, 10943, 10949, 10951, 10957, 10961, 10963, 10973, 10979, 10981, 10987, 10991, 10993, 10999,
    11003, 11009, 11017, 11021, 11023, 11027, 11029, 11041, 11047, 11051, 11057, 11059, 11069, 11071, 11083, 11087,
    11093, 11101, 11107, 11111, 11113, 11117, 11119, 11129, 11131, 11147, 11149, 11153, 11159, 11161, 11171, 11173,
    11177, 11183, 11189, 11191, 11197, 11201, 11203, 11213, 11227, 11233, 11237, 11239, 11243, 11251, 11257, 11261,
    11267, 11269, 11273, 11279, 11281, 11287, 11293, 11299, 11303, 11309, 11311, 11317, 11321, 11327, 11329, 11339,
    11351, 11353, 11357, 11359, 11369, 11371, 11377, 11381, 11383, 11387, 11393, 11399, 11411, 11413, 11419, 11423,
    11437, 11441, 11443, 11447, 11449, 11461, 11467, 11471, 11477, 11483, 11489, 11491, 11497, 11503, 11507, 11509,
    11513, 11519, 11521, 11527, 11533, 11537, 11549, 11551, 11563, 11567, 11569, 11573, 11579, 11581, 11587, 11591,
    11593, 11597, 11603, 11611, 11617, 11621, 11623, 11629, 11633, 11639, 11647, 11651, 11653, 11657, 11659, 11663,
    11677, 11681, 11689, 11699, 11701, 11707, 11717, 11719, 11723, 11729, 11731, 11741, 11743, 11747, 11749, 11761,
    11771, 11773, 11777, 11779, 11783, 11789, 11797, 11801, 11807, 11813, 11819, 11821, 11827, 11831, 11833, 11839,
    11849, 11857, 11861, 11863, 11867, 11873, 11881, 11887, 11897, 11899, 11903, 11909, 11911, 11917, 11923, 11927,
    11929, 11933, 11939, 11941, 11951, 11953, 11959, 11969, 11971, 11981, 11983, 11987, 11989, 11993, 12007, 12011,
    12013, 12017, 12029, 12031, 12037, 12041, 12043, 12049, 12053, 12059, 12071, 12073, 12079, 12083, 12091, 12097,
    12101, 12107, 12109, 12113, 12119, 12121, 12127, 12137, 12139, 12143, 12149, 12151, 12157, 12161, 12163, 12167,
    12169, 12179, 12191, 12193, 12197, 12203, 12209, 12211, 12217, 12223, 12227, 12239, 12241, 12247, 12251, 12253,
    12263, 12269, 12277, 12281, 12283, 12289, 12293, 12301, 12307, 12317, 12319, 12323, 12329, 12343, 12347, 12349,
    12359, 12361, 12367, 12371, 12373, 12377, 12379, 12391, 12401, 12403, 12407, 12409, 12413, 12421, 12427, 12431,
    12433, 12437, 12443, 12449, 12451, 12457, 12461, 12469, 12473, 12479, 12487, 12491, 12497, 12499, 12503, 12511,
    12517, 12521, 12527, 12533, 12539, 12541, 12547, 12553, 12557, 12559, 12563, 12569, 12577, 12581, 12583, 12587,
    12589, 12599, 12601, 12611, 12613, 12619, 12629, 12631, 12637, 12641, 12643, 12647, 12653, 12659, 12667, 12671,
    12673, 12679, 12689, 12697, 12703, 12707, 12709, 12713, 12721, 12731, 12737, 12739, 12743, 12751, 12757, 12763,
    12767, 12769, 12773, 12781, 12787, 12791, 12797, 12799, 12809, 12811, 12821, 12823, 12827, 12829, 12833, 12839,
    12841, 12847, 12851, 12853, 12863, 12869, 12871, 12877, 12889, 12893, 12899, 12907, 12911, 12913, 12917, 12919,
    12923, 12931, 12937, 12941, 12949, 12953, 12959, 12967, 12973, 12977, 12979, 12983, 12989, 12997, 13001, 13003,
    13007, 13009, 13019, 13021, 13031, 13033, 13037, 13043, 13049, 13051, 13061, 13063, 13067, 13073, 13081, 13087,
    13093, 13099, 13103, 13109, 13121, 13127, 13129, 13133, 13141, 13147, 13151, 13157, 13159, 13163, 13171, 13177,
    13183, 13187, 13193, 13199, 13201, 13207, 13213, 13217, 13219, 13229, 13231, 13241, 13243, 13249, 13253, 13259,
    13261, 13267, 13271, 13283, 13289, 13291, 13297, 13301, 13303, 13309, 13313, 13319, 13327, 13331, 13333, 13337,
    13339, 13357, 13361, 13367, 13369, 13373, 13379, 13381, 13393, 13397, 13399, 13411, 13417, 13421, 13423, 13427,
    13439, 13441, 13451, 13457, 13459, 13463, 13469, 13471, 13477, 13483, 13487, 13493, 13499, 13501, 13511, 13513,
    13523, 13529, 13537, 13543, 13547, 13549, 13553, 13561, 13567, 13571, 13577, 13579, 13583, 13589, 13591, 13597,
    13603, 13609, 13613, 13619, 13621, 13627, 13631, 13633, 13639, 13649, 13661, 13667, 13669, 13679, 13681, 13687,
    13691, 13693, 13697, 13703, 13709, 13711, 13721, 13723, 13729, 13733, 13747, 13751, 13753, 13757, 13759, 13763,
    13771, 13777, 13781, 13787, 13789, 13799, 13801, 13807, 13813, 13817, 13823, 13829, 13831, 13837, 13841, 13843,
    13847, 13859, 13861, 13873, 13877, 13879, 13883, 13889, 13891, 13901, 13903, 13907, 13913, 13919, 13921, 13927,
    13931, 13933, 13939, 13943, 13957, 13961, 13963, 13967, 13969, 13973, 13987, 13991, 13997, 13999, 14009, 14011,
    14017, 14023, 14029, 14033, 14039, 14041, 14051, 14057, 14059, 14071, 14081, 14083, 14087, 14089, 14093, 14099,
    14101, 14107, 14111, 14117, 14123, 14129, 14137, 14141, 14143, 14149, 14153, 14159, 14167, 14171, 14173, 14177,
    14191, 14197, 14207, 14213, 14219, 14221, 14227, 14233, 14237, 14239, 14243, 14249, 14251, 14257, 14263, 14269,
    14279, 14281, 14291, 14293, 14297, 14299, 14303, 14309, 14317, 14321, 14323, 14327, 14341, 14347, 14351, 14353,
    14359, 14363, 14369, 14381, 14383, 14387, 14389, 14393, 14401, 14407, 14411, 14419, 14423, 14429, 14431, 14437,
    14447, 14449, 14453, 14459, 14461, 14467, 14471, 14473, 14477, 14479, 14489, 14491, 14501, 14503, 14507, 14513,
    14519, 14527, 14533, 14537, 14543, 14549, 14551, 14557, 14561, 14563, 14569, 14579, 14587, 14591, 14593, 14603,
    14611, 14617, 14621, 14627, 14629, 14633, 14639, 14647, 14653, 14657, 14659, 14669, 14671, 14681, 14683, 14687,
    14689, 14699, 14701, 14711, 14713, 14717, 14719, 14723, 14731, 14737, 14741, 14743, 14747, 14753, 14759, 14761,
    14767, 14771, 14779, 14783, 14789, 14797, 14801, 14803, 14809, 14813, 14821, 14827, 14831, 14837, 14843, 14849,
    14851, 14857, 14863, 14867, 14869, 14873, 14879, 14881, 14887, 14891, 14893, 14897, 14899, 14909, 14921, 14923,
    14929, 14933, 14939, 14941, 14947, 14951, 14953, 14957, 14969, 14977, 14981, 14983, 14999, 15007, 15011, 15013,
    15017, 15019, 15023, 15031, 15047, 15049, 15053, 15061, 15073, 15077, 15079, 15083, 15089, 15091, 15097, 15101,
    15107, 15109, 15121, 15131, 15133, 15137, 15139, 15143, 15149, 15151, 15157, 15161, 15163, 15167, 15173, 15179,
    15181, 15187, 15193, 15199, 15203, 15209, 15217, 15221, 15227, 15229, 15233, 15241, 15247, 15251, 15259, 15263,
    15269, 15271, 15277, 15283, 15287, 15289, 15293, 15299, 15307, 15311, 15313, 15317, 15319, 15329, 15331, 15341,
    15343, 15347, 15349, 15359, 15361, 15371, 15373, 15377, 15383, 15391, 15397, 15401, 15403, 15409, 15413, 15419,
    15427, 15437, 15439, 15443, 15451, 15461, 15467, 15469, 15473, 15479, 15481, 15487, 15493, 15497, 15503, 15511,
    15517, 15523, 15527, 15529, 15539, 15541, 15551, 15553, 15557, 15559, 15563, 15569, 15571, 15577, 15581, 15583,
    15593, 15599, 15601, 15607, 15611, 15619, 15623, 15629, 15637, 15641, 15643, 15647, 15649, 15661, 15667, 15671,
    15677, 15679, 15683, 15689, 15703, 15707, 15709, 15713, 15721, 15727, 15731, 15733, 15737, 15739, 15749, 15751,
    15761, 15767, 15773, 15779, 15781, 15787, 15791, 15793, 15797, 15803, 15809, 15811, 15817, 15823, 15833, 15839,
    15853, 15857, 15859, 15863, 15871, 15877, 15881, 15887, 15889, 15893, 15901, 15907, 15913, 15919, 15923, 15929,
    15931, 15937, 15941, 15943, 15947, 15949, 15959, 15971, 15973, 15979, 15989, 15991, 15997, 16001, 16007, 16013,
    16019, 16021, 16031, 16033, 16039, 16043, 16057, 16061, 16063, 16067, 16069, 16073, 16087, 16091, 16097, 16099,
    16103, 16109, 16111, 16117, 16123, 16127, 16129, 16139, 16141, 16147, 16151, 16153, 16157, 16169, 16171, 16183,
    16187, 16189, 16193, 16199, 16201, 16207, 16213, 16217, 16223, 16229, 16231, 16241, 16243, 16249, 16253, 16259,
    16267, 16271, 16273, 16277, 16279, 16283, 16297, 16301, 16307, 16309, 16319, 16321, 16327, 16333, 16337, 16339,
    16343, 16349, 16351, 16361, 16363, 16369, 16381, 16391, 16397, 16399, 16403, 16409, 16411, 16417, 16421, 16427,
    16433, 16439, 16441, 16447, 16451, 16453, 16459, 16463, 16469, 16477, 16481, 16483, 16487, 16493, 16501, 16507,
    16517, 16519, 16529, 16531, 16537, 16543, 16547, 16553, 16559, 16561, 16567, 16571, 16573, 16579, 16589, 16591,
    16603, 16607, 16609, 16613, 16619, 16631, 16633, 16637, 16649, 16651, 16657, 16661, 16663, 16669, 16673, 16691,
    16693, 16697, 16699, 16703, 16711, 16717, 16721, 16727, 16729, 16733, 16739, 16741, 16747, 16759, 16763, 16769,
    16771, 16777, 16781, 16787, 16789, 16799, 16801, 16811, 16813, 16817, 16823, 16829, 16831, 16837, 16843, 16847,
    16853, 16859, 16867, 16871, 16873, 16879, 16883, 16889, 16897, 16901, 16903, 16909, 16921, 16927, 16931, 16937,
    16943, 16949, 16957, 16963, 16967, 16969, 16979, 16981, 16987, 16993, 16997, 16999, 17009, 17011, 17021, 17023,
    17027, 17029, 17033, 17041, 17047, 17051, 17053, 17057, 17063, 17071, 17077, 17081, 17089, 17093, 17099, 17107,
    17111, 17113, 17117, 17119, 17123, 17131, 17137, 17141, 17153, 17159, 17161, 17167, 17177, 17179, 17183, 17189,
    17191, 17197, 17201, 17203, 17207, 17209, 17219, 17221, 17231, 17233, 17239, 17243, 17249, 17257, 17261, 17263,
    17267, 17273, 17279, 17287, 17291, 17293, 17299, 17309, 17317, 17321, 17323, 17327, 17333, 17341, 17351, 17357,
    17359, 17363, 17371, 17377, 17383, 17387, 17389, 17393, 17399, 17401, 17411, 17417, 17419, 17429, 17431, 17441,
    17443, 17447, 17449, 17453, 17461, 17467, 17471, 17473, 17477, 17483, 17489, 17491, 17497, 17503, 17509, 17513,
    17519, 17527, 17531, 17533, 17539, 17543, 17551, 17557, 17561, 17569, 17573, 17579, 17581, 17587, 17593, 17597,
    17599, 17603, 17609, 17617, 17621, 17623, 17627, 17629, 17639, 17651, 17653, 17657, 17659, 17663, 17669, 17671,
    17681, 17683, 17687, 17701, 17707, 17711, 17713, 17723, 17729, 17737, 17741, 17747, 17749, 17753, 17761, 17767,
    17777, 17779, 17783, 17789, 17791, 17803, 17807, 17813, 17819, 17821, 17827, 17833, 17837, 17839, 17851, 17861,
    17863, 17867, 17869, 17873, 17879, 17881, 17887, 17891, 17893, 17903, 17909, 17911, 17917, 17921, 17923, 17929,
    17933, 17939, 17947, 17951, 17957, 17959, 17971, 17977, 17981, 17987, 17989, 17993, 17999, 18001, 18013, 18017,
    18019, 18023, 18037, 18041, 18043, 18047, 18049, 18059, 18061, 18071, 18077, 18079, 18089, 18091, 18097, 18101,
    18103, 18107, 18113, 18119, 18121, 18127, 18131, 18133, 18143, 18149, 18157, 18163, 18167, 18169, 18173, 18181,
    18191, 18197, 18199, 18203, 18209, 18211, 18217, 18223, 18229, 18233, 18241, 18247, 18251, 18253, 18257, 18259,
    18269, 18281, 18283, 18287, 18289, 18299, 18301, 18307, 18311, 18313, 18323, 18329, 18331, 18341, 18349, 18353,
    18367, 18371, 18373, 18377, 18379, 18383, 18391, 18397, 18401, 18407, 18409, 18413, 18419, 18427, 18433, 18437,
    18439, 18443, 18449, 18451, 18457, 18461, 18463, 18467, 18479, 18481, 18493, 18497, 18503, 18509, 18511, 18517,
    18521, 18523, 18527, 18533, 18539, 18541, 18547, 18553, 18559, 18563, 18569, 18581, 18583, 18587, 18589, 18593,
    18607, 18611, 18617, 18619, 18631, 18637, 18643, 18647, 18649, 18653, 18659, 18661, 18671, 18673, 18677, 18679,
    18691, 18701, 18703, 18709, 18713, 18719, 18721, 18727, 18731, 18737, 18743, 18749, 18751, 18757, 18761, 18763,
    18769, 18773, 18779, 18787, 18791, 18793, 18797, 18803, 18817, 18827, 18829, 18833, 18839, 18841, 18847, 18853,
    18857, 18859, 18869, 18871, 18877, 18881, 18883, 18899, 18901, 18911, 18913, 18917, 18919, 18923, 18929, 18937,
    18943, 18947, 18959, 18961, 18971, 18973, 18979, 18983, 18989, 19001, 19003, 19007, 19009, 19013, 19021, 19027,
    19031, 19037, 19039, 19043, 19049, 19051, 19057, 19067, 19069, 19073, 19079, 19081, 19087, 19091, 19093, 19099,
    19109, 19111, 19121, 19127, 19133, 19139, 19141, 19147, 19153, 19157, 19163, 19169, 19171, 19177, 19181, 19183,
    19189, 19193, 19199, 19207, 19211, 19213, 19219, 19223, 19231, 19237, 19241, 19247, 19249, 19259, 19267, 19273,
    19277, 19289, 19291, 19297, 19301, 19303, 19307, 19309, 19319, 19321, 19333, 19337, 19339, 19343, 19351, 19361,
    19363, 19367, 19373, 19379, 19381, 19387, 19391, 19399, 19403, 19417, 19421, 19423, 19427, 19429, 19433, 19441,
    19447, 19451, 19457, 19463, 19469, 19471, 19477, 19483, 19489, 19493, 19499, 19501, 19507, 19511, 19517, 19519,
    19529, 19531, 19541, 19543, 19549, 19553, 19559, 19561, 19567, 19571, 19573, 19577, 19583, 19589, 19597, 19601,
    19603, 19609, 19619, 19627, 19631, 19633, 19637, 19639, 19651, 19661, 19667, 19673, 19681, 19687, 19693, 19697,
    19699, 19703, 19709, 19711, 19717, 19727, 19729, 19739, 19741, 19751, 19753, 19757, 19759, 19763, 19769, 19771,
    19777, 19781, 19783, 19787, 19793, 19801, 19807, 19813, 19819, 19823, 19829, 19837, 19841, 19843, 19847, 19849,
    19853, 19861, 19867, 19871, 19879, 19883, 19889, 19891, 19897, 19907, 19909, 19913, 19919, 19927, 19931, 19933,
    19937, 19939, 19949, 19951, 19961, 19963, 19967, 19969, 19973, 19979, 19991, 19993, 19997, 20003, 20011, 20017,
    20021, 20023, 20029, 20039, 20047, 20051, 20057, 20063, 20071, 20077, 20081, 20087, 20089, 20093, 20099, 20101,
    20107, 20113, 20117, 20123, 20129, 20131, 20143, 20147, 20149, 20159, 20161, 20171, 20173, 20177, 20179, 20183,
    20191, 20197, 20201, 20203, 20213, 20219, 20221, 20227, 20231, 20233, 20239, 20243, 20249, 20257, 20261, 20263,
    20269, 20281, 20287, 20291, 20297, 20299, 20303, 20309, 20311, 20323, 20327, 20329, 20333, 20341, 20347, 20351,
    20353, 20357, 20359, 20369, 20381, 20387, 20389, 20393, 20399, 20401, 20407, 20411, 20413, 20417, 20429, 20431,
    20437, 20441, 20443, 20453, 20459, 20467, 20473, 20477, 20479, 20483, 20491, 20497, 20507, 20509, 20513, 20519,
    20521, 20533, 20539, 20543, 20549, 20551, 20557, 20561, 20563, 20567, 20569, 20591, 20593, 20597, 20599, 20609,
    20611, 20617, 20621, 20623, 20627, 20633, 20639, 20641, 20651, 20653, 20659, 20663, 20677, 20681, 20687, 20689,
    20693, 20701, 20707, 20711, 20717, 20719, 20723, 20729, 20731, 20737, 20743, 20747, 20749, 20753, 20759, 20767,
    20771, 20773, 20777, 20789, 20791, 20803, 20807, 20809, 20819, 20821, 20827, 20831, 20833, 20837, 20843, 20849,
    20851, 20857, 20861, 20863, 20869, 20873, 20879, 20887, 20893, 20897, 20899, 20903, 20921, 20927, 20929, 20939,
    20941, 20947, 20953, 20957, 20959, 20963, 20971, 20981, 20983, 20987, 20989, 21001, 21011, 21013, 21017, 21019,
    21023, 21029, 21031, 21037, 21041, 21053, 21059, 21061, 21067, 21071, 21079, 21083, 21089, 21097, 21101, 21103,
    21107, 21113, 21121, 21127, 21137, 21139, 21143, 21149, 21157, 21163, 21167, 21169, 21173, 21179, 21181, 21187,
    21191, 21193, 21199, 21209, 21211, 21221, 21223, 21227, 21233, 21239, 21247, 21251, 21253, 21257, 21269, 21271,
    21277, 21283, 21289, 21293, 21299, 21311, 21313, 21317, 21319, 21323, 21331, 21337, 21341, 21347, 21349, 21353,
    21361, 21367, 21377, 21379, 21383, 21389, 21391, 21397, 21401, 21403, 21407, 21409, 21419, 21421, 21431, 21433,
    21443, 21449, 21451, 21457, 21467, 21473, 21479, 21481, 21487, 21491, 21493, 21499, 21503, 21509, 21517, 21521,
    21523, 21529, 21533, 21547, 21551, 21557, 21559, 21563, 21569, 21577, 21583, 21587, 21589, 21599, 21601, 21607,
    21611, 21613, 21617, 21629, 21631, 21641, 21643, 21647, 21649, 21653, 21661, 21667, 21673, 21677, 21683, 21689,
    21691, 21701, 21709, 21713, 21719, 21727, 21731, 21733, 21737, 21739, 21743, 21751, 21757, 21761, 21767, 21773,
    21779, 21781, 21787, 21793, 21797, 21799, 21803, 21809, 21811, 21817, 21821, 21823, 21829, 21839, 21841, 21851,
    21859, 21863, 21869, 21871, 21877, 21881, 21883, 21887, 21893, 21899, 21907, 21911, 21913, 21919, 21929, 21937,
    21941, 21943, 21947, 21949, 21953, 21961, 21971, 21977, 21979, 21991, 21997, 22003, 22007, 22013, 22019, 22021,
    22027, 22031, 22037, 22039, 22049, 22051, 22063, 22067, 22069, 22073, 22079, 22081, 22091, 22093, 22097, 22103,
    22109, 22111, 22117, 22123, 22129, 22133, 22147, 22151, 22153, 22157, 22159, 22163, 22171, 22177, 22181, 22189,
    22193, 22199, 22201, 22207, 22213, 22219, 22223, 22229, 22237, 22241, 22247, 22249, 22259, 22261, 22271, 22273,
    22277, 22279, 22283, 22289, 22291, 22301, 22303, 22307, 22313, 22327, 22331, 22333, 22339, 22343, 22349, 22357,
    22361, 22367, 22369, 22381, 22387, 22391, 22397, 22403, 22409, 22411, 22417, 22423, 22427, 22433, 22439, 22441,
    22447, 22453, 22457, 22459, 22469, 22471, 22481, 22483, 22487, 22489, 22493, 22499, 22501, 22507, 22511, 22513,
    22523, 22531, 22537, 22541, 22543, 22549, 22553, 22559, 22567, 22571, 22573, 22577, 22579, 22591, 22597, 22601,
    22609, 22613, 22619, 22621, 22637, 22639, 22643, 22651, 22657, 22661, 22663, 22667, 22669, 22679, 22681, 22691,
    22697, 22699, 22703, 22709, 22717, 22721, 22723, 22727, 22733, 22739, 22741, 22747, 22751, 22753, 22769, 22777,
    22783, 22787, 22793, 22801, 22807, 22811, 22817, 22819, 22823, 22829, 22831, 22837, 22843, 22849, 22853, 22859,
    22861, 22871, 22873, 22877, 22879, 22889, 22901, 22903, 22907, 22909, 22921, 22927, 22931, 22933, 22937, 22943,
    22949, 22951, 22961, 22963, 22969, 22973, 22987, 22991, 22993, 22999, 23003, 23011, 23017, 23021, 23027, 23029,
    23033, 23039, 23041, 23047, 23053, 23057, 23059, 23063, 23069, 23071, 23077, 23081, 23083, 23087, 23099, 23113,
    23117, 23119, 23123, 23129, 23131, 23137, 23141, 23143, 23147, 23159, 23161, 23167, 23171, 23173, 23183, 23189,
    23197, 23201, 23203, 23207, 23209, 23213, 23227, 23237, 23239, 23249, 23251, 23263, 23267, 23269, 23273, 23279,
    23281, 23291, 23293, 23297, 23299, 23311, 23321, 23323, 23327, 23329, 23333, 23339, 23341, 23347, 23351, 23357,
    23363, 23369, 23371, 23377, 23381, 23383, 23389, 23393, 23399, 23407, 23411, 23417, 23423, 23431, 23437, 23447,
    23449, 23453, 23459, 23461, 23467, 23473, 23477, 23479, 23483, 23489, 23497, 23501, 23503, 23509, 23519, 23521,
    23531, 23533, 23537, 23539, 23549, 23557, 23561, 23563, 23567, 23579, 23581, 23587, 23591, 23593, 23599, 23603,
    23609, 23623, 23627, 23629, 23633, 23641, 23651, 23657, 23659, 23663, 23669, 23671, 23677, 23687, 23689, 23693,
    23701, 23707, 23711, 23713, 23717, 23719, 23729, 23731, 23741, 23743, 23747, 23753, 23759, 23761, 23767, 23773,
    23783, 23789, 23791, 23797, 23801, 23809, 23813, 23819, 23827, 23831, 23833, 23839, 23843, 23851, 23857, 23861,
    23867, 23869, 23873, 23879, 23887, 23893, 23897, 23899, 23909, 23911, 23917, 23921, 23923, 23927, 23929, 23939,
    23941, 23951, 23953, 23957, 23963, 23971, 23977, 23981, 23983, 23987, 23993, 23999, 24001, 24007, 24019, 24023,
    24029, 24041, 24043, 24047, 24049, 24053, 24061, 24067, 24071, 24077, 24083, 24091, 24097, 24103, 24107, 24109,
    24113, 24119, 24121, 24127, 24131, 24133, 24137, 24139, 24149, 24151, 24161, 24163, 24169, 24173, 24179, 24181,
    24187, 24191, 24197, 24203, 24209, 24217, 24221, 24223, 24229, 24239, 24247, 24251, 24253, 24257, 24259, 24263,
    24281, 24287, 24289, 24293, 24301, 24307, 24313, 24317, 24319, 24329, 24331, 24337, 24341, 24347, 24359, 24361,
    24371, 24373, 24377, 24379, 24383, 24389, 24391, 24397, 24403, 24407, 24413, 24419, 24421, 24433, 24439, 24443,
    24449, 24457, 24461, 24463, 24467, 24469, 24473, 24481, 24487, 24491, 24499, 24503, 24509, 24511, 24517, 24523,
    24527, 24529, 24533, 24539, 24547, 24551, 24553, 24559, 24569, 24571, 24581, 24587, 24589, 24593, 24599, 24601,
    24611, 24613, 24617, 24623, 24631, 24637, 24641, 24643, 24649, 24653, 24659, 24667, 24671, 24677, 24679, 24683,
    24691, 24697, 24701, 24707, 24709, 24719, 24721, 24727, 24733, 24737, 24743, 24749, 24751, 24757, 24763, 24767,
    24769, 24779, 24781, 24793, 24797, 24799, 24803, 24809, 24811, 24821, 24823, 24833, 24839, 24841, 24847, 24851,
    24853, 24859, 24863, 24877, 24881, 24883, 24887, 24889, 24901, 24907, 24911, 24917, 24919, 24923, 24929, 24931,
    24943, 24949, 24953, 24961, 24967, 24971, 24977, 24979, 24989, 24991, 25001, 25007, 25009, 25013, 25019, 25021,
    25027, 25031, 25033, 25037, 25043, 25049, 25057, 25061, 25063, 25073, 25079, 25087, 25093, 25097, 25099, 25111,
    25117, 25121, 25127, 25133, 25139, 25141, 25147, 25153, 25159, 25163, 25169, 25171, 25177, 25183, 25187, 25189,
    25199, 25211, 25213, 25217, 25219, 25229, 25231, 25237, 25241, 25243, 25247, 25253, 25261, 25271, 25273, 25279,
    25283, 25297, 25301, 25303, 25307, 25309, 25313, 25321, 25327, 25331, 25339, 25343, 25349, 25351, 25357, 25367,
    25369, 25373, 25379, 25381, 25387, 25391, 25393, 25397, 25409, 25411, 25423, 25427, 25429, 25433, 25439, 25447,
    25451, 25453, 25457, 25463, 25469, 25471, 25477, 25481, 25483, 25489, 25499, 25507, 25511, 25513, 25517, 25523,
    25537, 25541, 25547, 25549, 25559, 25561, 25567, 25573, 25577, 25579, 25583, 25589, 25591, 25601, 25603, 25607,
    25609, 25621, 25631, 25633, 25637, 25639, 25643, 25651, 25657, 25661, 25667, 25673, 25679, 25681, 25687, 25691,
    25693, 25699, 25703, 25709, 25717, 25721, 25723, 25733, 25741, 25747, 25757, 25759, 25763, 25769, 25771, 25777,
    25783, 25787, 25789, 25793, 25799, 25801, 25807, 25811, 25813, 25819, 25829, 25841, 25843, 25847, 25849, 25853,
    25859, 25867, 25871, 25873, 25877, 25889, 25891, 25897, 25901, 25903, 25913, 25919, 25931, 25933, 25937, 25939,
    25943, 25951, 25957, 25967, 25969, 25973, 25979, 25981, 25997, 25999, 26003, 26009, 26011, 26017, 26021, 26023,
    26027, 26029, 26041, 26051, 26053, 26057, 26063, 26069, 26071, 26077, 26083, 26087, 26093, 26099, 26101, 26107,
    26111, 26113, 26119, 26123, 26129, 26137, 26141, 26149, 26153, 26161, 26167, 26171, 26177, 26179, 26183, 26189,
    26197, 26203, 26207, 26209, 26219, 26227, 26231, 26233, 26237, 26239, 26249, 26251, 26261, 26263, 26267, 26269,
    26281, 26287, 26291, 26293, 26297, 26303, 26309, 26311, 26317, 26321, 26329, 26333, 26339, 26347, 26353, 26357,
    26359, 26363, 26371, 26381, 26387, 26393, 26399, 26401, 26407, 26413, 26417, 26419, 26423, 26431, 26437, 26441,
    26443, 26447, 26449, 26459, 26461, 26471, 26473, 26479, 26483, 26489, 26491, 26497, 26501, 26503, 26513, 26519,
    26527, 26531, 26539, 26549, 26557, 26561, 26563, 26567, 26569, 26573, 26581, 26591, 26597, 26599, 26603, 26617,
    26623, 26627, 26629, 26633, 26639, 26641, 26647, 26651, 26657, 26659, 26669, 26671, 26681, 26683, 26687, 26693,
    26699, 26701, 26707, 26711, 26713, 26717, 26723, 26729, 26731, 26737, 26743, 26749, 26753, 26759, 26771, 26773,
    26777, 26779, 26783, 26791, 26797, 26801, 26809, 26813, 26821, 26827, 26833, 26837, 26839, 26843, 26849, 26857,
    26861, 26863, 26867, 26869, 26879, 26881, 26891, 26893, 26899, 26903, 26909, 26911, 26921, 26927, 26933, 26941,
    26947, 26951, 26953, 26959, 26963, 26969, 26977, 26981, 26987, 26989, 26993, 27007, 27011, 27017, 27019, 27023,
    27029, 27031, 27037, 27043, 27047, 27059, 27061, 27067, 27073, 27077, 27089, 27091, 27101, 27103, 27107, 27109,
    27113, 27119, 27121, 27127, 27133, 27143, 27149, 27151, 27161, 27163, 27169, 27173, 27179, 27187, 27191, 27193,
    27197, 27199, 27211, 27217, 27221, 27227, 27229, 27233, 27239, 27241, 27253, 27257, 27259, 27263, 27271, 27277,
    27281, 27283, 27289, 27299, 27301, 27311, 27317, 27319, 27323, 27329, 27331, 27337, 27341, 27343, 27347, 27353,
    27359, 27361, 27367, 27371, 27373, 27383, 27389, 27397, 27403, 27407, 27409, 27413, 27421, 27427, 27431, 27437,
    27439, 27449, 27451, 27457, 27463, 27473, 27479, 27481, 27487, 27491, 27493, 27497, 27499, 27509, 27523, 27527,
    27529, 27539, 27541, 27551, 27553, 27557, 27563, 27569, 27571, 27581, 27583, 27589, 27593, 27607, 27611, 27613,
    27617, 27619, 27623, 27631, 27637, 27641, 27647, 27649, 27653, 27659, 27661, 27667, 27673, 27679, 27683, 27689,
    27691, 27697, 27701, 27707, 27719, 27721, 27733, 27737, 27739, 27743, 27749, 27751, 27757, 27761, 27763, 27767,
    27773, 27779, 27787, 27791, 27793, 27799, 27803, 27809, 27817, 27821, 27823, 27827, 27829, 27847, 27851, 27857,
    27869, 27871, 27877, 27883, 27887, 27889, 27893, 27899, 27901, 27913, 27917, 27919, 27931, 27941, 27943, 27947,
    27949, 27953, 27959, 27961, 27967, 27971, 27977, 27983, 27991, 27997, 28001, 28003, 28009, 28013, 28019, 28027,
    28031, 28033, 28037, 28043, 28051, 28057, 28069, 28073, 28079, 28081, 28087, 28097, 28099, 28103, 28109, 28111,
    28117, 28121, 28123, 28129, 28139, 28141, 28151, 28153, 28157, 28159, 28163, 28169, 28177, 28181, 28183, 28187,
    28199, 28201, 28207, 28211, 28213, 28219, 28229, 28241, 28243, 28247, 28253, 28261, 28267, 28271, 28277, 28279,
    28283, 28289, 28291, 28297, 28307, 28309, 28313, 28319, 28321, 28331, 28333, 28337, 28339, 28349, 28351, 28361,
    28363, 28367, 28373, 28381, 28387, 28393, 28397, 28403, 28409, 28411, 28417, 28421, 28423, 28429, 28433, 28439,
    28447, 28451, 28453, 28459, 28463, 28471, 28477, 28481, 28487, 28489, 28493, 28499, 28507, 28513, 28517, 28519,
    28529, 28531, 28537, 28541, 28543, 28547, 28549, 28559, 28571, 28573, 28577, 28579, 28583, 28591, 28597, 28601,
    28603, 28607, 28619, 28621, 28627, 28631, 28643, 28649, 28657, 28661, 28663, 28667, 28669, 28673, 28681, 28687,
    28697, 28703, 28709, 28711, 28723, 28727, 28729, 28733, 28739, 28741, 28747, 28751, 28753, 28757, 28759, 28771,
    28781, 28783, 28789, 28793, 28799, 28801, 28807, 28811, 28813, 28817, 28823, 28829, 28837, 28841, 28843, 28849,
    28859, 28867, 28871, 28877, 28879, 28883, 28891, 28901, 28907, 28909, 28913, 28921, 28927, 28933, 28937, 28939,
    28943, 28949, 28957, 28961, 28967, 28969, 28979, 28981, 28991, 28993, 28997, 28999, 29009, 29011, 29017, 29021,
    29023, 29027, 29033, 29039, 29041, 29047, 29053, 29059, 29063, 29069, 29077, 29083, 29087, 29089, 29093, 29101,
    29111, 29119, 29123, 29129, 29131, 29137, 29143, 29147, 29149, 29153, 29167, 29171, 29173, 29177, 29179, 29189,
    29191, 29201, 29203, 29207, 29209, 29213, 29219, 29221, 29231, 29233, 29243, 29251, 29257, 29261, 29269, 29273,
    29279, 29287, 29291, 29297, 29299, 29303, 29311, 29317, 29321, 29327, 29329, 29333, 29339, 29347, 29353, 29357,
    29363, 29369, 29371, 29377, 29383, 29387, 29389, 29399, 29401, 29411, 29413, 29417, 29423, 29429, 29431, 29437,
    29441, 29443, 29453, 29459, 29461, 29467, 29473, 29479, 29483, 29489, 29501, 29503, 29507, 29509, 29521, 29527,
    29531, 29537, 29539, 29543, 29551, 29563, 29567, 29569, 29573, 29581, 29587, 29591, 29593, 29597, 29599, 29609,
    29611, 29621, 29629, 29633, 29639, 29641, 29647, 29651, 29657, 29663, 29669, 29671, 29677, 29681, 29683, 29693,
    29699, 29707, 29713, 29717, 29719, 29723, 29737, 29741, 29747, 29749, 29753, 29759, 29761, 29767, 29773, 29779,
    29789, 29791, 29797, 29801, 29803, 29807, 29819, 29831, 29833, 29837, 29839, 29849, 29851, 29857, 29863, 29867,
    29873, 29879, 29881, 29891, 29893, 29899, 29903, 29917, 29921, 29923, 29927, 29929, 29933, 29941, 29947, 29951,
    29957, 29959, 29963, 29969, 29971, 29977, 29983, 29987, 29989, 29993, 29999, 30001, 30007, 30011, 30013, 30029,
};

/// p^-1 mod 2^32 for odd p (Newton: 3 -> 6 -> 12 -> 24 -> 48 bits).
inline u32 inv32(const u32 p)
{
    u32 x = p;
    x *= 2u - p * x; x *= 2u - p * x; x *= 2u - p * x; x *= 2u - p * x;
    return x;
}

#ifndef MT_CB
#define MT_CB 128                // table blocks (30030 numbers, 180 words) per mt_build group (host: MT_CB_HOST)
#endif
#define MT_SPLIT_P 1024u         // primes below: marks split over MT_R sub-ranges (load balance)
#define MT_R 32u
/// Builds the multiplier table: bit kb 5760 + w (block kb of the table) set
/// iff m = 30030 (k_lo + kb) + WHL_M[w] has no prime factor in
/// primes[s_begin, s_end). Group = MT_CB blocks in SLM; the wheel tables in SLM.
/// @pre    primes[s_begin, s_end) = every prime in [67, sqrt(m_max)] (3..13
///         are the wheel, 17..61 the patterns pat), ascending; every table m exceeds primes[s_end - 1];
///         local size 1024; tab holds nblocks * 180 words.
/// @post   every table bit is 1 iff its m is prime.
__kernel void mt_build(__global u32* restrict tab, const u32 nblocks, const u32 k_lo,
                       __global const u32* restrict primes, const u32 s_begin, const u32 s_end,
                       const u32 n_small,   // primes[s_begin, s_begin + n_small) are < MT_SPLIT_P
                       __global const u32* restrict pat)   // MT_PRE patterns (host: mt_patterns)
{
    __local u32 win[MT_CB * 180];
    __local ushort lidx[15015];       // odd residues r: lidx[r >> 1] (every m here is odd)
    __local uchar lstep[WHL_N];
    const u32 lid = get_local_id(0), wg = get_local_size(0);
    const u32 kb0 = get_group_id(0) * MT_CB;
    if (kb0 >= nblocks) return;
    const u32 nb = min((u32)MT_CB, nblocks - kb0);
    // presieve: primes 17..61 by patterns, word wd of block k for prime s =
    // pat[off_s + (k mod s) 180 + wd] (bit set iff s does not divide m)
    for (u32 i = lid; i < nb * 180u; i += wg) {
        const u32 kb = i / 180u, wd = i - kb * 180u;
        const u32 k = k_lo + kb0 + kb;
        u32 v = 0xFFFFFFFFu, off = 0u;
#define MT_P(S) v &= pat[off + (k % S##u) * 180u + wd]; off += S##u * 180u;
        MT_P(17) MT_P(19) MT_P(23) MT_P(29) MT_P(31) MT_P(37) MT_P(41) MT_P(43)
        MT_P(47) MT_P(53) MT_P(59) MT_P(61)
#undef MT_P
        win[i] = v;
    }
    for (u32 i = lid; i < 15015u; i += wg) lidx[i] = WHL_IDX[2u * i + 1u];
    for (u32 i = lid; i < WHL_N; i += wg) lstep[i] = WHL_STEP[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    const u64 m_a = (u64)(k_lo + kb0) * 30030UL;   // even
    const u32 span = nb * 30030u;
    const u32 ntask = n_small * MT_R + (s_end - s_begin - n_small);
    for (u32 tk = lid; tk < ntask; tk += wg) {
        u32 s, ra, rb;
        if (tk < n_small * MT_R) {
            s = primes[s_begin + tk / MT_R];
            const u32 r = tk % MT_R;
            ra = (u32)(((u64)span * r) / MT_R);
            rb = (u32)(((u64)span * (r + 1u)) / MT_R);
        } else {
            s = primes[s_begin + n_small + (tk - n_small * MT_R)];
            ra = 0u; rb = span;
        }
        // first odd multiple m = s t >= m_a + ra, then the first with t coprime to 30030
        const u32 rm = mod_u64_u32(m_a + ra, s);
        u32 rel = ra + (rm ? s - rm : 0u);
        if (!(rel & 1u)) rel += s;                    // m_a even: rel odd <-> m odd
        u32 rt = (u32)(((u64)(rel % 30030u) * WHL_INV[s % 30030u]) % 30030u);   // t mod 30030
        while (lidx[rt >> 1] == WHL_MISS) { rel += 2u * s; rt += 2u; if (rt >= 30030u) rt -= 30030u; }
        if (rel >= rb) continue;
        u32 w = lidx[rt >> 1];
        for (;;) {
            const u32 kk = rel / 30030u;
            const u32 bit = kk * 5760u + lidx[(rel - kk * 30030u) >> 1];
            atomic_and((volatile __local u32*)&win[bit >> 5], ~(1u << (bit & 31u)));
            const u32 st = 2u * s * lstep[w];
            if (rb - rel <= st) break;
            rel += st;
            if (++w == WHL_N) w = 0u;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (u32 i = lid; i < nb * 180u; i += wg) tab[(size_t)kb0 * 180u + i] = win[i];
}

#define HL_QO(j) (((j) == nseg - 1u) ? q_odds_tail : q_odds_full)
/// hl_harvest with the prime-multiplier filter (HL_MT): files only hits p m
/// with m prime (see above); no residue skip sets needed (m prime has none).
/// @pre    as hl_harvest; mt from mt_build covering every multiplier of the
///         batch window (blocks [k_lo, k_lo + mt_words / 180)); else *err |= 32.
/// @post   as hl_harvest, restricted to the hits with m prime.
__kernel void hl_harvest_mt(
        __global const u32* restrict primes,
        const u32 a, const u32 b,
        const u64 q_first,
        const u32 delta, const u32 q_odds_full,
        const u32 nseg, const u32 q_odds_tail,
        const float inv_delta,
        __global u32* restrict lists, const u32 capg,
        __global u32* restrict gcnt,
        volatile __global u32* restrict err,
        __global const u32* restrict mt, const u32 k_lo, const u32 mt_words)
{
    __local u32 lpos[HL_MAX_K];
    __local ushort lm[WHL_N];
    const u32 lid = get_local_id(0), wg = get_local_size(0);
    const u32 grp = get_group_id(0), ngrp = get_num_groups(0);
    for (u32 w = lid; w < WHL_N; w += wg) lm[w] = WHL_M[w];
    if (lid < HL_MAX_K) lpos[lid] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);
    const u64 wend = (u64)(nseg - 1u) * delta + q_odds_tail;   // window [0, wend) in bits
    for (u32 c0 = grp * wg; c0 < b - a; c0 += ngrp * wg) {
        const u32 i = a + c0 + lid;
        if (i >= b) break;
        const u32 p = primes[i];
        const u64 first = first_odd_multiple((u64)p, q_first);
        const u64 xo = (first - q_first) >> 1;
        if (xo >= wend) continue;
        // m0 = first / p (exact division, m0 < 2^38: host): low word by the inverse,
        // high word from a float estimate (error << 2^31)
        const u32 lo = (u32)first * inv32(p);
        const u32 hi = (u32)rint(((float)first * (1.0f / (float)p) - (float)lo) * 2.3283064365386963e-10f);
        // m0 = 30030 k0 + r0 (2^32 = 143022 * 30030 + 16636)
        const u32 q1 = lo / 30030u, r1 = lo - q1 * 30030u;
        const u32 t1 = hi * 16636u + r1;
        const u32 k0 = hi * 143022u + q1 + t1 / 30030u;
        const u32 r0 = t1 % 30030u;
        // last multiplier offset: x = xo + p dm / 2 < wend
        const u64 rem = wend - 1u - xo;
        u32 nh = (u32)((float)rem / (float)p);
        while ((u64)nh * p > rem) --nh;
        while ((u64)(nh + 1u) * p <= rem) ++nh;
        const u32 dmax = 2u * nh;
        // first coprime multiplier >= m0
        u32 kr = k0 - k_lo, rr = r0;
        while (WHL_IDX[rr] == WHL_MISS) { rr += 2u; if (rr >= 30030u) { rr -= 30030u; ++kr; } }
        if (k0 < k_lo) { atomic_or(err, 32u); continue; }
        // table word / bit of that multiplier (bit index kr 5760 + w exceeds
        // 2^32 beyond m_lo + 2.2e10: word = kr 180 + w / 32, 5760 = 180 * 32)
        const u32 w0 = WHL_IDX[rr];
        u32 wi = kr * 180u + (w0 >> 5);
        u32 word = (wi < mt_words) ? mt[wi] & (0xFFFFFFFFu << (w0 & 31u)) : 0u;
        const u32 kb = k0 - k_lo;
        for (;;) {
            if (wi >= mt_words) { atomic_or(err, 32u); break; }
            bool done = false;
            while (word) {
                const u32 bt = 31u - clz(word & (0u - word));
                word &= word - 1u;
                const u32 kk = wi / 180u;
                const u32 dm = (kk - kb) * 30030u + lm[(wi - kk * 180u) * 32u + bt] - r0;
                if (dm > dmax) { done = true; break; }
                const u64 x = xo + (u64)p * (dm >> 1);
                u32 j = (u32)(((float)(u32)(x >> 32) * 4294967296.0f + (float)(u32)x) * inv_delta);
                int o = (int)((u32)x - j * delta);
                while (o < 0) { o += (int)delta; --j; }
                while (o >= (int)delta) { o -= (int)delta; ++j; }
                const u32 off = (u32)o;
                if (j < nseg && off < HL_QO(j)) {
                    const u32 idx = atomic_inc(&lpos[j]);
                    if (idx < capg) lists[((size_t)j * ngrp + grp) * capg + idx] = off;
                }
                if (j >= 1u && j - 1u < nseg && off + delta < HL_QO(j - 1u)) {
                    const u32 idx = atomic_inc(&lpos[j - 1u]);
                    if (idx < capg) lists[((size_t)(j - 1u) * ngrp + grp) * capg + idx] = off + delta;
                }
            }
            if (done) break;
            ++wi;
            {   // stop once the next word starts beyond the window
                const u32 kk = wi / 180u;
                const u32 dm = (kk - kb) * 30030u + lm[(wi - kk * 180u) * 32u] - r0;
                if (dm > dmax) break;
            }
            word = (wi < mt_words) ? mt[wi] : 0u;
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
    large_packed_body(prime_map, bases, gaps4, large_count, q_odds, q_first, q_last, g);
}

// ----------------------------------------------------------------------------
// Mid-prime SLM windows (GB_MIDWIN, v2.6, HISTORY.md 9.2x): primes [split, P2) leave
// the packed kernel's L2 atomics (~30 ps per mark) for a second sieve pass in
// large SLM windows of the class map (one SLM atomic per mark plus one visit
// per prime per window). mid_setup computes, once per segment, the first
// class bit of every mid prime in every class; mid_window loads a window of
// one class array into SLM, clears every multiple of every mid prime in it,
// and stores it back. Clears are of odd multiples p m with m coprime to 30
// (all multiples stored in the class map): a superset of the packed kernel's
// wheel-30030 marks, all composite, so the map is the same.
// ----------------------------------------------------------------------------
#ifndef MIDW_BATCH
#define MIDW_BATCH 4             // 2026-11-17: 4 > 8 = 16 (window pass -1%)
#endif
#ifndef MIDW_SG
#define MIDW_SG 16               // SIMD16: window pass -3% vs the compiler's SIMD32
#endif
__constant uchar INV15[15] = {0, 1, 8, 0, 4, 0, 0, 13, 2, 0, 0, 11, 0, 7, 14};
/// @pre    q_first odd; primes[a .. a + n) in (30, 2^32) coprime to 30;
///         phis = 8 dense phases (nibbles) of the class arrays.
/// @post   i0tab[c * n + j] = smallest class-c bit i whose q = q_first +
///         2 (15 i + phi_c) is a multiple of primes[a + j].
__kernel void mid_setup(__global const u32* restrict primes, const u32 a, const u32 n,
                        const u64 q_first, const u32 phis, __global u32* restrict i0tab)
{
    const u32 j = get_global_id(0);
    if (j >= n) return;
    const u32 p = primes[a + j];
    // exact 64-bit remainder: once per prime per segment, free beside the
    // window pass (bench null), and valid for every p (mod_ff needs p >= 4096)
    const u32 r = (u32)(q_first % (u64)p);
    u32 t = r ? p - r : 0u;
    t += (t & 1u) ? p : 0u;                 // q_first odd: first odd multiple at q_first + t
    const u32 b0 = t >> 1;                  // its dense bit, < p
    const u32 pinv = INV15[p % 15u];
    const u32 b15 = b0 % 15u;
    #pragma unroll
    for (u32 c = 0; c < 8u; ++c) {
        const u32 ph = (phis >> (4u * c)) & 15u;
        const u32 k = ((ph + 15u - b15) * pinv) % 15u;   // b0 + k p = ph (mod 15)
        i0tab[c * n + j] = (b0 + k * p - ph) / 15u;
    }
}

/// @pre    mid_setup of this segment wrote i0tab; the tile sieve's stores are
///         visible (barrier before); MIDW_ANDST 2: other writers of the map
///         only AND-clear bits atomically (the packed kernel may overlap);
///         MIDW_ANDST 0 / 1: no concurrent writer. Class bits < 2^24.
/// @post   every class bit i < n_c of q = multiple of a mid prime is cleared.
#ifndef MIDW_ANDST
#define MIDW_ANDST 2             // 2: atomic AND store (the packed kernel may overlap)
#endif
#ifndef MIDW_SKIP
#define MIDW_SKIP 0              // skip marks on q = 0 mod 7 (1), 7|11 (2), 7|11|13 (3): already cleared
#endif
// Marks of one prime in a window. With MIDW_SKIP the bits whose q is a
// multiple of 7 (11, 13) are left alone: the tile kernel's small-prime masks
// cleared them already, and a window's bit g (global class bit) has q = 0
// mod m exactly when g = z_m (mod m), whatever p; d_m tracks (g - z_m) mod m.
#define MIDW_MARK_LOOP(BIT, P, WB, V, WS)                                          \
    {                                                                              \
        u32 bit_ = (BIT);                                                          \
        MIDW_SKIP_INIT(bit_, P, WS)                                                \
        for (; bit_ < (WB); bit_ += (P)) {                                         \
            if (MIDW_SKIP_TEST) atomic_and(&(V)[bit_ >> 5], CLR_MASK_R(c1v, bit_)); \
            MIDW_SKIP_STEP                                                         \
        }                                                                          \
    }
#if MIDW_SKIP >= 1
#define MIDW_SK1_INIT(b, P, WS) u32 d7 = ((WS) + (b) + 7u - z7) % 7u; const u32 p7 = (P) % 7u;
#define MIDW_SK1_STEP d7 += p7; d7 -= (d7 >= 7u) ? 7u : 0u;
#define MIDW_SK1_TEST (d7 != 0u)
#else
#define MIDW_SK1_INIT(b, P, WS)
#define MIDW_SK1_STEP
#define MIDW_SK1_TEST true
#endif
#if MIDW_SKIP >= 2
#define MIDW_SK2_INIT(b, P, WS) u32 d11 = ((WS) + (b) + 11u - z11) % 11u; const u32 p11 = (P) % 11u;
#define MIDW_SK2_STEP d11 += p11; d11 -= (d11 >= 11u) ? 11u : 0u;
#define MIDW_SK2_TEST && (d11 != 0u)
#else
#define MIDW_SK2_INIT(b, P, WS)
#define MIDW_SK2_STEP
#define MIDW_SK2_TEST
#endif
#if MIDW_SKIP >= 3
#define MIDW_SK3_INIT(b, P, WS) u32 d13 = ((WS) + (b) + 13u - z13) % 13u; const u32 p13 = (P) % 13u;
#define MIDW_SK3_STEP d13 += p13; d13 -= (d13 >= 13u) ? 13u : 0u;
#define MIDW_SK3_TEST && (d13 != 0u)
#else
#define MIDW_SK3_INIT(b, P, WS)
#define MIDW_SK3_STEP
#define MIDW_SK3_TEST
#endif
#define MIDW_SKIP_INIT(b, P, WS) MIDW_SK1_INIT(b, P, WS) MIDW_SK2_INIT(b, P, WS) MIDW_SK3_INIT(b, P, WS)
#define MIDW_SKIP_STEP MIDW_SK1_STEP MIDW_SK2_STEP MIDW_SK3_STEP
#define MIDW_SKIP_TEST (MIDW_SK1_TEST MIDW_SK2_TEST MIDW_SK3_TEST)

#if MIDW_SG
__attribute__((intel_reqd_sub_group_size(MIDW_SG)))
#endif
__kernel void mid_window(__global u32* restrict map32, __global const u32* restrict primes,
                         const u32 a, const u32 n, __global const u32* restrict i0tab,
                         const u64 q_odds, const u32 phis, const u32 c5w32, const u32 wpc,
                         const u32 win_bits, __local u32* restrict win, const u64 q_first)
{
    const u32 lid = get_local_id(0), wg = get_local_size(0);
    const u32 cls = get_group_id(0) / wpc, w = get_group_id(0) - cls * wpc;
    const u32 ph = (phis >> (4u * cls)) & 15u;
    const u32 nc = (q_odds > (u64)ph) ? (u32)((q_odds - ph + 14UL) / 15UL) : 0u;
    const u32 ws = w * win_bits;
    const u32 c1v = 0xFFFFFFFEu ^ (u32)(q_odds >> 40);   // opaque ~1 (make_prime_map)
#if MIDW_SKIP
    // class bit g has q = Q + 30 g, Q = q_first + 2 ph (the window's class):
    // q = 0 (mod m) iff g = z_m = -Q / 30 (mod m)
    const u64 qc = q_first + 2UL * ph;
    const u32 z7 = ((7u - (u32)(qc % 7UL)) % 7u) * 4u % 7u;          // 1/30 = 4 (mod 7)
#if MIDW_SKIP >= 2
    const u32 z11 = ((11u - (u32)(qc % 11UL)) % 11u) * 7u % 11u;     // 1/30 = 7 (mod 11)
#endif
#if MIDW_SKIP >= 3
    const u32 z13 = ((13u - (u32)(qc % 13UL)) % 13u) * 10u % 13u;    // 1/30 = 10 (mod 13)
#endif
#endif
    if (ws >= nc) return;                   // uniform per work-group
    const u32 wb = min(win_bits, nc - ws);  // valid bits of this window
    const u32 nw = (wb + 31u) >> 5;
    __global u32* gm = map32 + (size_t)cls * c5w32 + (ws >> 5);
    // 16-byte loads and stores (gm is 16 KB aligned): one work-group per Xe
    // core, so the window load / store latency is exposed; uint4 issues 4x
    // fewer messages (2026-11-18: window pass -5.5 ms per 1e11, +0.8%)
    const u32 nw4 = nw >> 2;
#if MIDW_ANDST
    // the window starts as ones and is ANDed into the map at the store: no
    // exposed load phase before the marks (one work-group per Xe core)
    {
        __local uint4* w4 = (__local uint4*)win;
        for (u32 i = lid; i < nw4; i += wg) w4[i] = (uint4)(~0u);
        for (u32 i = (nw4 << 2) + lid; i < nw; i += wg) win[i] = ~0u;
    }
#else
    {
        __global const uint4* g4 = (__global const uint4*)gm;
        __local uint4* w4 = (__local uint4*)win;
        #pragma unroll 8
        for (u32 i = lid; i < nw4; i += wg) w4[i] = g4[i];
        for (u32 i = (nw4 << 2) + lid; i < nw; i += wg) win[i] = gm[i];
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    volatile __local u32* v = (volatile __local u32*)win;
    __global const u32* it = i0tab + (size_t)cls * n;
    __global const u32* pr = primes + a;
    u32 j = lid;
    for (; j + (MIDW_BATCH - 1u) * wg < n; j += MIDW_BATCH * wg) {
        u32 iv[MIDW_BATCH], pv[MIDW_BATCH];
        #pragma unroll
        for (u32 u = 0; u < MIDW_BATCH; ++u) { iv[u] = it[j + u * wg]; pv[u] = pr[j + u * wg]; }
        #pragma unroll
        for (u32 u = 0; u < MIDW_BATCH; ++u) {
            const u32 p = pv[u];
            u32 bit;
            if (iv[u] >= ws) bit = iv[u] - ws;
            else {                          // d < 2^24: exact float remainder (med_prime_f32)
                const float df = (float)(ws - iv[u]);
                const float pf = (float)p;
                const float qf = trunc(df * (1.0f / pf));
                float rf = fma(-qf, pf, df);
                rf += (rf < 0.0f) ? pf : 0.0f;
                rf -= (rf >= pf) ? pf : 0.0f;
                const u32 rmu = (u32)rf;
                bit = rmu ? p - rmu : 0u;
            }
            MIDW_MARK_LOOP(bit, p, wb, v, ws)
        }
    }
    for (; j < n; j += wg) {
        const u32 i0 = it[j], p = pr[j];
        u32 bit;
        if (i0 >= ws) bit = i0 - ws;
        else {
            const float df = (float)(ws - i0);
            const float pf = (float)p;
            const float qf = trunc(df * (1.0f / pf));
            float rf = fma(-qf, pf, df);
            rf += (rf < 0.0f) ? pf : 0.0f;
            rf -= (rf >= pf) ? pf : 0.0f;
            const u32 rmu = (u32)rf;
            bit = rmu ? p - rmu : 0u;
        }
        MIDW_MARK_LOOP(bit, p, wb, v, ws)
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
        __global uint4* g4 = (__global uint4*)gm;
        __local const uint4* w4 = (__local const uint4*)win;
#if MIDW_ANDST == 2
        // atomic AND: the packed large kernel may already be marking
        for (u32 i = lid; i < nw; i += wg) {
            const u32 x = win[i];
            if (x != ~0u) atomic_and((volatile __global u32*)gm + i, x);
        }
#elif MIDW_ANDST
        #pragma unroll 8
        for (u32 i = lid; i < nw4; i += wg) g4[i] &= w4[i];
        for (u32 i = (nw4 << 2) + lid; i < nw; i += wg) gm[i] &= win[i];
#else
        #pragma unroll 8
        for (u32 i = lid; i < nw4; i += wg) g4[i] = w4[i];
        for (u32 i = (nw4 << 2) + lid; i < nw; i += wg) gm[i] = win[i];
#endif
    }
}
