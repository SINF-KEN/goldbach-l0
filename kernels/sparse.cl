// ============================================================================
// sparse.cl — sparse class map (v2.0, include/sparse.hpp, src/sparse.cpp)
//
// Map: K class arrays; array k bit t <-> q = Q + c_k + M t (M = 2310, K = 8
// by default; the production module is built with M = 30030, K = 12).
// scm_tile  : SLM tile sieve of the class arrays (mask groups for 13..61,
//             thread-owned walk, prime-parallel medium stage).
// scm_large : one pass over every base prime >= split per segment: first
//             multiple in each class from Q mod p, then marks (global atomics).
// scm_check_a / scm_check_b : mod-2310 Goldbach check (phase A funnels,
//             phase B probes), see include/sparse.hpp for the geometry.
// ============================================================================

typedef ulong u64;
typedef uint  u32;
typedef ushort u16;

// Geometry (include/sparse.hpp scm_geo): the module is compiled per modulus.
#ifndef SCM_M
#define SCM_M 2310u
#define SCM_K 8u
#define SCM_RES_LIST 17u, 29u, 229u, 487u, 631u, 793u, 1181u, 1763u
#endif
#define SCM_CR (SCM_M / 2u)
#define SCM_META 48u                             // meta u32 per class: n_seg, (begin, end, k | mask << 8)
#define TILE_WORDS 2048u
#define TILE_BITS (TILE_WORDS * 64u)
#define WG_SIZE 256u
#define WPT (TILE_WORDS / WG_SIZE)          // 8 u64 words per thread
#define BITS_PER_THREAD (WPT * 64u)
#define MAX_GROUPS 8u
#define NONE32 0xFFFFFFFFu

__constant u32 SCM_RES[SCM_K] = {SCM_RES_LIST};


// ---------------------------------------------------------------------------
/// x mod p for 32-bit x: float quotient + exact integer fix-ups (float
/// precision only bounds the number of fix-up iterations).
/// @pre    p >= 1 and |x - q p| fits an int for the float estimate q, which
///         holds for the tile primes here (p < 2^17).
/// @return x mod p.
inline u32 mod32(const u32 x, const u32 p)
{
    const u32 q = (u32)((float)x * (1.0f / (float)p));
    int r = (int)(x - q * p);
    while (r < 0) r += (int)p;
    while (r >= (int)p) r -= (int)p;
    return (u32)r;
}

// ---------------------------------------------------------------------------
// Tile sieve. Work-group g: class k = g / tpc, tile g % tpc of that array.
// fbt[k * split + j]: first hit t in [0, p) of prime j in class k (segment-
// local; t >= T means no hit in this segment). Primes [0, j_own0) are the
// mask primes (13..61, applied by the group tables), [j_own0, own_end) the
// thread-owned walk, [own_end, split) the medium stage.
// ---------------------------------------------------------------------------
#ifndef MED_BATCH
#define MED_BATCH 8
#endif
/// Sieves one TILE_BITS tile of one class array in SLM and writes it back.
/// Grid: K * tpc work-groups of WG_SIZE (256) threads.
/// @param cmap      out: K class arrays of stride64 u64 words.
/// @param stride64  u64 words per class array (>= tpc * TILE_WORDS).
/// @param tpc       tiles per class array.
/// @param T         valid bits per class array; bits >= T are written as 0.
/// @param primes    base primes; [j_own0, split) are walked here.
/// @param fbt       [k * split + j] first hit of prime j in class k, segment-
///                  local, in [0, p) (NONE32 for primes dividing M).
/// @param j_own0,own_end,split  stage boundaries (see above).
/// @param gmask,gdesc,n_groups,gy  mask-group tables (host make_groups_scm /
///                  scm_group_offsets); n_groups <= MAX_GROUPS.
/// @param tile      SLM scratch of TILE_WORDS u64.
/// @pre   every multiple of a prime < 64 not dividing M is cleared by the
///        mask groups (j_own0 = first prime >= 64).
/// @post  tile words of class k hold 1 exactly for the q of the tile with no
///        base-prime factor below primes[split] (multiples of larger primes
///        are cleared by scm_large); bits >= T are 0.
__kernel void scm_tile(
        __global u64* restrict cmap,          // K arrays of stride64 u64 words
        const u32 stride64,
        const u32 tpc,
        const u32 T,                          // valid bits per class array
        __global const u32* restrict primes,
        __global const u32* restrict fbt,
        const u32 j_own0,
        const u32 own_end,
        const u32 split,
        __global const u64* restrict gmask,
        __global const u32* restrict gdesc,   // (offset, period) per group
        const u32 n_groups,
        __global const u32* restrict gy,      // [k * MAX_GROUPS + g]
        __local u64* restrict tile)
{
    const u32 me = get_local_id(0);
    const u32 k = get_group_id(0) / tpc;
    const u32 tid = get_group_id(0) - k * tpc;
    const u32 tile_bit0 = tid * TILE_BITS;
    const u32 w0 = me * WPT;                  // my first tile word
    fbt += k * split;

    // ---- masks: each owned word = AND of the group tables ------------------
    {
        const u32 x = tid * TILE_WORDS + w0;  // class-array word index
        u32 ph[MAX_GROUPS], per[MAX_GROUPS], off[MAX_GROUPS];
        #pragma unroll
        for (u32 g = 0; g < MAX_GROUPS; ++g) {
            if (g < n_groups) {
                off[g] = gdesc[2 * g];
                per[g] = gdesc[2 * g + 1];
                const u32 y = gy[k * MAX_GROUPS + g];
                ph[g] = (x % per[g] + per[g] - y) % per[g];
            }
        }
        for (u32 i = 0; i < WPT; ++i) {
            u64 v = ~0UL;
            #pragma unroll
            for (u32 g = 0; g < MAX_GROUPS; ++g) {
                if (g < n_groups) {
                    v &= gmask[off[g] + ph[g]];
                    ph[g] = (ph[g] + 1u == per[g]) ? 0u : ph[g] + 1u;
                }
            }
            tile[w0 + i] = v;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- thread-owned walk: primes [j_own0, own_end), plain RMWs ---------
    {
        __local u32* t32 = (__local u32*)tile;
        const u32 my0 = me * BITS_PER_THREAD;            // tile-local
        const u32 my1 = my0 + BITS_PER_THREAD;
        const u32 g0 = tile_bit0 + my0;                  // class-array bit
        for (u32 j = j_own0; j < own_end; ++j) {
            const u32 f = fbt[j];
            const u32 p = primes[j];
            u32 b;
            if (f >= g0) {
                if (f - g0 >= BITS_PER_THREAD) continue;
                b = my0 + (f - g0);
            } else {
                const u32 r = mod32(g0 - f, p);
                b = my0 + (r ? p - r : 0u);
            }
            for (; b < my1; b += p) t32[b >> 5] &= ~(1u << (b & 31u));
        }
    }

    // ---- medium stage: primes [own_end, split), prime-parallel SLM atomics
    if (own_end < split) {
        barrier(CLK_LOCAL_MEM_FENCE);
        volatile __local u32* t32 = (volatile __local u32*)tile;
        u32 j = own_end + me;
        for (; j + (MED_BATCH - 1) * WG_SIZE < split; j += MED_BATCH * WG_SIZE) {
            u32 fv[MED_BATCH], pv[MED_BATCH];
            #pragma unroll
            for (u32 u = 0; u < MED_BATCH; ++u) {
                fv[u] = fbt[j + u * WG_SIZE];
                pv[u] = primes[j + u * WG_SIZE];
            }
            #pragma unroll
            for (u32 u = 0; u < MED_BATCH; ++u) {
                const u32 f = fv[u], p = pv[u];
                u32 b;
                if (f >= tile_bit0) {
                    b = f - tile_bit0;
                } else {
                    const u32 r = mod32(tile_bit0 - f, p);
                    b = r ? p - r : 0u;
                }
                for (; b < TILE_BITS; b += p) atomic_and(&t32[b >> 5], ~(1u << (b & 31u)));
            }
        }
        for (; j < split; j += WG_SIZE) {
            const u32 f = fbt[j], p = primes[j];
            u32 b;
            if (f >= tile_bit0) {
                b = f - tile_bit0;
            } else {
                const u32 r = mod32(tile_bit0 - f, p);
                b = r ? p - r : 0u;
            }
            for (; b < TILE_BITS; b += p) atomic_and(&t32[b >> 5], ~(1u << (b & 31u)));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- write back; bits >= T cleared --------------------------------------
    __global u64* dst = cmap + (u64)k * stride64 + (u64)tid * TILE_WORDS;
    for (u32 i = 0; i < WPT; ++i) {
        const u32 w = w0 + i;
        u64 v = tile[w];
        const u32 bit0 = tile_bit0 + w * 64u;
        if (bit0 + 64u > T) {
            const u32 valid = (T > bit0) ? T - bit0 : 0u;
            v &= (valid >= 64u) ? ~0UL : ((1UL << valid) - 1UL);
        }
        dst[w] = v;
    }
}

// ---------------------------------------------------------------------------
// Large primes: x mod p (x < 2^63) — float-float quotient for p < 2^30 (the
// v1.8.1 FF_MOD math, validated bit-exact by tools/debug_fb), float estimate
// + two exact u64 refinements above.
// ---------------------------------------------------------------------------
/// Low 32 bits of the integer value of a non-negative, integer-valued float.
/// @return (u64)v mod 2^32, without a float -> u64 conversion.
inline u32 float_int_low32(const float v)
{
    const u32 b = as_uint(v);
    const int e = (int)((b >> 23) & 0xFFu) - 150;
    const u32 m = (b & 0x7FFFFFu) | 0x800000u;
    if (e >= 32 || v == 0.0f) return 0u;
    return (e >= 0) ? (m << e) : (m >> (-e));
}
typedef struct { float xh; float xl; u32 x32; } ff_x;
/// Float-float split of a uniform dividend for mod_ff (computed once, reused
/// for every prime).
/// @return {xh = float(x), xl = float(x - xh), x32 = low 32 bits of x}.
inline ff_x ff_split(const u64 x)
{
    ff_x r;
    r.xh = (float)x;
    r.xl = (float)(long)(x - (u64)r.xh);
    r.x32 = (u32)x;
    return r;
}
/// x mod p from a pre-split dividend (float-float quotient, exact fix-ups).
/// @pre    x < 2^63 (from ff_split), 1 <= p < 2^30.
/// @return x mod p (validated bit-exact by tools/debug_fb).
inline u32 mod_ff(const ff_x x, const u32 p)
{
    const float pf = (float)p;
    const long pr = (long)p - (long)pf;
    const float ih = 1.0f / pf;
    const float e1 = fma(-pf, ih, 1.0f);
    const float il = ih * (e1 - (float)pr * ih);
    const float qh = rint(x.xh * ih);
    const float ql = fma(x.xh, ih, -qh) + x.xh * il + x.xl * ih;
    const u32 q32 = float_int_low32(qh) + (u32)(int)rint(ql);
    int r = (int)(x.x32 - q32 * p);
    while (r < 0) r += (int)p;
    while (r >= (int)p) r -= (int)p;
    return (u32)r;
}
/// x mod p for any 1 <= p < 2^32 (two float refinements + exact fix-ups);
/// the slower fallback of mod_ff for p >= 2^30.
/// @pre    x < 2^63.
/// @return x mod p.
inline u32 mod_big(const u64 x, const u32 p)
{
    const float f = 1.0f / (float)p;
    const u64 qt = (u64)((float)x * f);
    long r = (long)(x - qt * (u64)p);
    r -= (long)((float)r * f) * (long)p;
    while (r < 0) r += (long)p;
    while (r >= (long)p) r -= (long)p;
    return (u32)r;
}

#ifndef LARGE_PACK_SG
#define LARGE_PACK_SG 32
#endif
#ifndef LARGE_PACK_PPI
#define LARGE_PACK_PPI 16
#endif
#define LARGE_PACK_GROUP (LARGE_PACK_PPI * LARGE_PACK_SG)
#define LARGE_PACK_WORDS (LARGE_PACK_PPI / 4)
/// Decodes the LARGE_PACK_PPI packed primes of work-item @p item (host format:
/// src/sparse.cpp pack_primes).
/// @pre    called by every lane of a LARGE_PACK_SG-wide subgroup whose lanes
///         are consecutive items of one packing group (contains a subgroup scan);
///         large_count >= 1.
/// @param pv  out: the item's primes, ascending.
/// @post   entries at global index >= large_count are garbage: callers skip them.
inline void decode_large(__global const u32* restrict bases,
                         __global const u32* restrict gaps4,
                         const u32 item, const u32 large_count, u32* pv)
{
    u32 gb[LARGE_PACK_PPI];
    u32 sum = 0;
    #pragma unroll
    for (u32 k = 0; k < LARGE_PACK_WORDS; ++k) {
        const u32 w = (item * LARGE_PACK_PPI + 4u * k < large_count)
                    ? gaps4[item * LARGE_PACK_WORDS + k] : 0u;
        gb[4 * k]     = w & 0xFFu;
        gb[4 * k + 1] = (w >> 8) & 0xFFu;
        gb[4 * k + 2] = (w >> 16) & 0xFFu;
        gb[4 * k + 3] = w >> 24;
        sum += gb[4 * k] + gb[4 * k + 1] + gb[4 * k + 2] + gb[4 * k + 3];
    }
    const u32 excl = sub_group_scan_exclusive_add(sum);
    u32 p = bases[min(item / LARGE_PACK_SG, (large_count - 1u) / LARGE_PACK_GROUP)] + 2u * excl;
    #pragma unroll
    for (u32 k = 0; k < LARGE_PACK_PPI; ++k) {
        p += 2u * gb[k];
        pv[k] = p;
    }
}

// p^-1 mod M for p mod M (0 for non-units), host-uploaded.
// First hit of p in class k (see sparse.hpp): rho = (-Q) mod p is the offset
// of the first multiple m0 p >= Q; with p = M a + b and b^-1 = inv,
// delta = ((c_k - rho) * inv) mod M is the smallest m - m0 >= 0 with
// m p = c_k (mod M), and the class index of that multiple is
//   t = (rho + p delta - c_k) / M = a delta + (rho + b delta - c_k) / M,
// the last numerator being >= 0 (it is = 0 mod M and > -M). All 32-bit:
// rho < p < 2^32, a delta < p, b delta < M^2.
/// Marks the multiples of every packed large prime in all K class arrays with
/// global atomics. Grid: ceil(items / WG) work-groups, items =
/// ceil(large_count / LARGE_PACK_PPI).
/// @param cmap32      K class arrays of stride32 u32 words (tile-sieved).
/// @param bases,gaps4 packed large primes (decode_large).
/// @param Q           segment base, Q = 0 (mod M), Q < 2^63.
/// @param T           valid bits per class array.
/// @param invtab      [M] b^-1 mod M for units b, 0 otherwise.
/// @pre   every large prime p satisfies p < 2^32 - max(M^2, T), so that
///        rho + b delta and t + p do not wrap (p < ~3.39e9 for M = 30030,
///        i.e. limit < ~1.15e19); p does not divide M.
/// @pre   runs after scm_tile on the same map (in-order list + barrier).
/// @post  every multiple of a large prime with class index < T is cleared.
__attribute__((intel_reqd_sub_group_size(LARGE_PACK_SG)))
__kernel void scm_large(
        volatile __global u32* restrict cmap32,  // K arrays of stride32 u32 words
        const u32 stride32,
        __global const u32* restrict bases,
        __global const u32* restrict gaps4,
        const u32 large_count,
        const u64 Q,                             // segment base (Q = 0 mod M)
        const u32 T,
        __global const u16* restrict invtab)     // [M]
{
    const u32 item = get_global_id(0);
    const u32 c0 = item * LARGE_PACK_PPI;
    u32 pv[LARGE_PACK_PPI];
    decode_large(bases, gaps4, item, large_count, pv);
    const ff_x qs = ff_split(Q);
    #pragma unroll
    for (u32 l = 0; l < LARGE_PACK_PPI; ++l) {
        const u32 p = pv[l];
        if (c0 + l >= large_count) break;
#ifdef ABLL_DECODE
        if (p == 0xFFFFFFFFu) cmap32[0] = 0u;
        continue;
#endif
        const u32 R = (p < (1u << 30)) ? mod_ff(qs, p) : mod_big(Q, p);
#ifdef ABLL_MOD
        if (R == 0xFFFFFFFFu) cmap32[0] = 0u;
        continue;
#endif
        const u32 rho = R ? p - R : 0u;
        const u32 a = p / SCM_M;
        const u32 b = p - a * SCM_M;
        const u32 inv = invtab[b];
        const u32 rhoM = rho % SCM_M;
        #pragma unroll
        for (u32 k = 0; k < SCM_K; ++k) {
            const u32 c = SCM_RES[k];
            const u32 dl = ((c + SCM_M - rhoM) * inv) % SCM_M;
            if (a * dl >= T) continue;                   // t >= a delta >= T
            u32 t = a * dl + (rho + b * dl - c) / SCM_M;
            volatile __global u32* S = cmap32 + k * stride32;

#ifdef ABLL_NOMARK   // timing only (INVALID map): setup without marks
            if (t == 0xFFFFFFFFu) S[0] = 0u;
#else
            // (reading the word first and marking only set bits, or a mark
            // filter for multiples of 13..61, measured slower: the kernel is
            // bound by random L3 line accesses, reads and atomics alike)
            for (; t < T; t += p)
                atomic_and(S + (t >> 5), ~(1u << (t & 31u)));
#endif
        }
    }
}

// ---------------------------------------------------------------------------
// Checker (mod-2310 form of check30_a / check30_b; include/sparse.hpp).
// ---------------------------------------------------------------------------
#ifndef SCM_MAXD
#define SCM_MAXD 4               // largest pattern shift (host-checked)
#endif
/// @return S[w] when 0 <= w < n, else 0 (negative w wraps out of range).
inline u32 ld_guard(const __global u32* S, const int w, const u32 n)
{
    return ((u32)w < n) ? S[w] : 0u;
}
// Phase A, span windows (v2.0c). Per plan segment ONE window covers every
// funnel of the segment (the plan caps a segment's U span at A3_SPAN words):
// A3_W = A3_WPL + A3_SPAN + 1 words loaded once (vload8), pattern-ORed once
// (prefix doubling: masks 3 / 7 / 15 / 31 in 1 / 2 / 2 / 3 shift-OR steps),
// and each funnel reads its words through a switch on its (uniform) word
// offset, so the register indices stay static: no slides, no reloads (a
// one-word register slide cost ~95 instructions, a reload + OR ~550).
#ifndef A3_SPAN
#define A3_SPAN 12
#endif
#ifndef A3_WPL
#define A3_WPL 32u                                 // u32 words (32 evens each) per lane: 28 -> 32 -6% phase A; 36, 40 slower
#endif
#define A3_TOT (A3_WPL + A3_SPAN + 1 + 3)             // window + 3 words below (pattern OR)
/// Funnel shift left of the 64-bit pair (a:b) by d.
/// @pre    1 <= d <= 31 (a shift by 32 wraps to 0 on Xe).
/// @return the high word of (a:b) << d.
inline u32 fsl(const u32 a, const u32 b, const u32 d)
{
    return (a << d) | (b >> (32u - d));
}
/// Phase A of the sparse checker: one work-item = A3_WPL * 32 evens of one
/// even class cr; items_pad items per class (multiple of the work-group
/// size, so a work-group has one class). Runs the plan's funnels (meta /
/// cands) and appends every word with unresolved evens to @p pending.
/// @param cmap32,stride32   the finished class map.
/// @param checked,chk_stride  validation output (written iff write_checked).
/// @param cands,meta        plan (make_plan_scm; validated on the host).
/// @param pending,pending_count,pending_cap  out: entries
///                          cr << pshift | word << 32 | miss mask.
/// @param ivalid            [2 cr, 2 cr + 1] = valid even range [ilo, ihi) of
///                          class cr; evens outside it count as resolved.
/// @pre   items_pad * A3_WPL * 32 >= dlt; map bits >= T are 0.
/// @post  a set acc bit means a witness q = n - p with p prime and the map
///        bit of q set; *pending_count is the true count and may exceed
///        pending_cap (host fails closed).
__kernel void scm_check_a(
        __global const u32* restrict cmap32,
        const u32 stride32,
        __global u64* restrict checked,
        const u32 chk_stride,
        __global const u16* restrict cands,
        __global const u32* restrict meta,
        __global u64* restrict pending,
        __global u32* restrict pending_count,
        const u32 pending_cap,
        const u32 items_pad,
        __global const u32* restrict ivalid,
        const u32 write_checked,
        const u32 pshift)                        // pending entry: cr << pshift | word << 32 | miss mask
{
    const u32 gid = get_global_id(0);
    const u32 cr = gid / items_pad;                    // uniform per work-group
    const u32 blk = gid - cr * items_pad;
    const bool inr = cr < SCM_CR;
    const u32 ilo = inr ? ivalid[2 * cr] : 0u;
    const u32 ihi = inr ? ivalid[2 * cr + 1] : 0u;
    const u32 mo = cr * SCM_META;
    const u32 k0 = blk * (A3_WPL * 32u);
    const bool live = inr && k0 < ihi && k0 + A3_WPL * 32u > ilo;

    u32 acc[A3_WPL];
    if (live && k0 >= ilo && k0 + A3_WPL * 32u <= ihi) {
        #pragma unroll
        for (u32 i = 0; i < A3_WPL; ++i) acc[i] = 0u;
    } else {
        #pragma unroll
        for (u32 i = 0; i < A3_WPL; ++i) {
            const u32 kb = k0 + 32u * i;
            const u32 lo = (ilo > kb) ? min(32u, ilo - kb) : 0u;
            const u32 hi = (ihi > kb) ? min(32u, ihi - kb) : 0u;
            const u32 lom = (lo >= 32u) ? ~0u : ((1u << lo) - 1u);
            const u32 him = (hi >= 32u) ? ~0u : ((1u << hi) - 1u);
            const u32 vmask = (hi > lo) ? (him & ~lom) : 0u;
            acc[i] = live ? ~vmask : ~0u;
        }
    }

#ifdef ABL3_NOSEG
    if (0) {
#else
    if (live) {
#endif
        const u32 ns = meta[mo];
        for (u32 st = 0; st < ns; ++st) {
            const u32 jb = meta[mo + 1u + 3u * st];
            const u32 je = meta[mo + 2u + 3u * st];
            const u32 km = meta[mo + 3u + 3u * st];
            const u32 k = km & 0xFFu, mask = (km >> 8) & 0xFFu;
            const __global u32* S = cmap32 + k * stride32;
            const u32 Ul = cands[je - 1u];
            // window word 0 = class word (k0 + Ul) >> 5; x[0..2] = words -3..-1
#ifdef ABL3_HOTWIN
            const int wb = 64 + (int)(cr & 7u) * 64;   // timing only: every window from one hot spot
#else
            const int wb = (int)(k0 >> 5) + (int)(Ul >> 5) - 3;
#endif
            u32 x[A3_TOT];
            if (wb >= 0 && wb + 8 * ((A3_TOT + 7) / 8) <= (int)stride32) {
                const __global u32* P_ = S + wb;
#if 0 < A3_TOT
                { const uint8 v_ = vload8(0, P_);
                  x[0] = v_.s0;
#if 1 < A3_TOT
                  x[1] = v_.s1;
#endif
#if 2 < A3_TOT
                  x[2] = v_.s2;
#endif
#if 3 < A3_TOT
                  x[3] = v_.s3;
#endif
#if 4 < A3_TOT
                  x[4] = v_.s4;
#endif
#if 5 < A3_TOT
                  x[5] = v_.s5;
#endif
#if 6 < A3_TOT
                  x[6] = v_.s6;
#endif
#if 7 < A3_TOT
                  x[7] = v_.s7;
#endif
                }
#endif
#if 8 < A3_TOT
                { const uint8 v_ = vload8(1, P_);
                  x[8] = v_.s0;
#if 9 < A3_TOT
                  x[9] = v_.s1;
#endif
#if 10 < A3_TOT
                  x[10] = v_.s2;
#endif
#if 11 < A3_TOT
                  x[11] = v_.s3;
#endif
#if 12 < A3_TOT
                  x[12] = v_.s4;
#endif
#if 13 < A3_TOT
                  x[13] = v_.s5;
#endif
#if 14 < A3_TOT
                  x[14] = v_.s6;
#endif
#if 15 < A3_TOT
                  x[15] = v_.s7;
#endif
                }
#endif
#if 16 < A3_TOT
                { const uint8 v_ = vload8(2, P_);
                  x[16] = v_.s0;
#if 17 < A3_TOT
                  x[17] = v_.s1;
#endif
#if 18 < A3_TOT
                  x[18] = v_.s2;
#endif
#if 19 < A3_TOT
                  x[19] = v_.s3;
#endif
#if 20 < A3_TOT
                  x[20] = v_.s4;
#endif
#if 21 < A3_TOT
                  x[21] = v_.s5;
#endif
#if 22 < A3_TOT
                  x[22] = v_.s6;
#endif
#if 23 < A3_TOT
                  x[23] = v_.s7;
#endif
                }
#endif
#if 24 < A3_TOT
                { const uint8 v_ = vload8(3, P_);
                  x[24] = v_.s0;
#if 25 < A3_TOT
                  x[25] = v_.s1;
#endif
#if 26 < A3_TOT
                  x[26] = v_.s2;
#endif
#if 27 < A3_TOT
                  x[27] = v_.s3;
#endif
#if 28 < A3_TOT
                  x[28] = v_.s4;
#endif
#if 29 < A3_TOT
                  x[29] = v_.s5;
#endif
#if 30 < A3_TOT
                  x[30] = v_.s6;
#endif
#if 31 < A3_TOT
                  x[31] = v_.s7;
#endif
                }
#endif
#if 32 < A3_TOT
                { const uint8 v_ = vload8(4, P_);
                  x[32] = v_.s0;
#if 33 < A3_TOT
                  x[33] = v_.s1;
#endif
#if 34 < A3_TOT
                  x[34] = v_.s2;
#endif
#if 35 < A3_TOT
                  x[35] = v_.s3;
#endif
#if 36 < A3_TOT
                  x[36] = v_.s4;
#endif
#if 37 < A3_TOT
                  x[37] = v_.s5;
#endif
#if 38 < A3_TOT
                  x[38] = v_.s6;
#endif
#if 39 < A3_TOT
                  x[39] = v_.s7;
#endif
                }
#endif
#if 40 < A3_TOT
                { const uint8 v_ = vload8(5, P_);
                  x[40] = v_.s0;
#if 41 < A3_TOT
                  x[41] = v_.s1;
#endif
#if 42 < A3_TOT
                  x[42] = v_.s2;
#endif
#if 43 < A3_TOT
                  x[43] = v_.s3;
#endif
#if 44 < A3_TOT
                  x[44] = v_.s4;
#endif
#if 45 < A3_TOT
                  x[45] = v_.s5;
#endif
#if 46 < A3_TOT
                  x[46] = v_.s6;
#endif
#if 47 < A3_TOT
                  x[47] = v_.s7;
#endif
                }
#endif
#if 48 < A3_TOT
                { const uint8 v_ = vload8(6, P_);
                  x[48] = v_.s0;
#if 49 < A3_TOT
                  x[49] = v_.s1;
#endif
#if 50 < A3_TOT
                  x[50] = v_.s2;
#endif
#if 51 < A3_TOT
                  x[51] = v_.s3;
#endif
#if 52 < A3_TOT
                  x[52] = v_.s4;
#endif
#if 53 < A3_TOT
                  x[53] = v_.s5;
#endif
#if 54 < A3_TOT
                  x[54] = v_.s6;
#endif
#if 55 < A3_TOT
                  x[55] = v_.s7;
#endif
                }
#endif
#if 56 < A3_TOT
                { const uint8 v_ = vload8(7, P_);
                  x[56] = v_.s0;
#if 57 < A3_TOT
                  x[57] = v_.s1;
#endif
#if 58 < A3_TOT
                  x[58] = v_.s2;
#endif
#if 59 < A3_TOT
                  x[59] = v_.s3;
#endif
#if 60 < A3_TOT
                  x[60] = v_.s4;
#endif
#if 61 < A3_TOT
                  x[61] = v_.s5;
#endif
#if 62 < A3_TOT
                  x[62] = v_.s6;
#endif
#if 63 < A3_TOT
                  x[63] = v_.s7;
#endif
                }
#endif
            } else {
                #pragma unroll
                for (u32 t = 0; t < A3_TOT; ++t) x[t] = ld_guard(S, wb + (int)t, stride32);
            }

            // pattern OR in place, top down (bit q of the result = OR over d in
            // mask of bit q - d): prefix doubling for the contiguous masks
#define A3_STEP(D, LO) do { _Pragma("unroll") for (u32 t = A3_TOT; t > (LO); --t) \
                x[t - 1u] |= fsl(x[t - 1u], x[t - 2u], (D)); } while (0)
#ifdef ABL3_NOOR
            if (0) {}
#else
            if (0) {}
#endif
#ifndef ABL3_NOOR
            else if (mask == 3u) { A3_STEP(1u, 1u); }
            else if (mask == 7u) { A3_STEP(1u, 1u); A3_STEP(1u, 2u); }
            else if (mask == 15u) { A3_STEP(1u, 1u); A3_STEP(2u, 2u); }
            else if (mask == 31u) { A3_STEP(1u, 1u); A3_STEP(2u, 2u); A3_STEP(1u, 3u); }
            // gapped 4-member masks: direct shift-ORs (the generic loop below
            // compiled to ~775 instructions with a copy of the window)
#define A3_DIRECT3(D1, D2, D3) do { _Pragma("unroll") for (u32 t = A3_TOT - 1u; t > 0u; --t) { \
                const u32 a_ = x[t], b_ = x[t - 1u]; \
                x[t] = a_ | fsl(a_, b_, (D1)) | fsl(a_, b_, (D2)) | fsl(a_, b_, (D3)); } } while (0)
            else if (mask == 29u) A3_DIRECT3(2u, 3u, 4u);
            else if (mask == 27u) A3_DIRECT3(1u, 3u, 4u);
            else if (mask == 23u) A3_DIRECT3(1u, 2u, 4u);
#endif
#ifndef ABL3_NOOR
            else if (mask != 1u) {
#else
            else if (0) {
#endif
                u32 y[A3_TOT];
                #pragma unroll
                for (u32 t = 1; t < A3_TOT; ++t) {
                    u32 v = x[t];
                    #pragma unroll
                    for (u32 d = 1; d <= SCM_MAXD; ++d)
                        if ((mask >> d) & 1u) v |= (x[t] << d) | (x[t - 1u] >> (32u - d));
                    y[t] = v;
                }
                #pragma unroll
                for (u32 t = 1; t < A3_TOT; ++t) x[t] = y[t];
            }
#ifdef ABL3_ONEF   // timing only (INVALID): one funnel per segment
            for (u32 j = jb; j < jb + 1u; ++j) {
#else
            for (u32 j = jb; j < je; ++j) {
#endif
                const u32 U = cands[j];
                const u32 o = (U >> 5) - (Ul >> 5);            // uniform, <= A3_SPAN (plan)
                const u32 rr = U & 31u;
                // rr != 0: one-shift funnel (32 - rr in 1..31); rr == 0: plain OR
                // (a shift by 32 wraps to 0 on Xe, hence the split: 3 ops per word
                // instead of 4). The switch keeps every register index static; a
                // per-offset loop cascade (+50%), register-indirect indexing (+58%)
                // and a sub_group_broadcast of U (+8%) all measured slower.
                const u32 lr = 32u - rr;
#define A3_F(O) do { _Pragma("unroll") for (u32 i = 0; i < A3_WPL; ++i) \
                    acc[i] |= (x[3u + (O) + i] >> rr) | (x[4u + (O) + i] << lr); } while (0)
#define A3_F0(O) do { _Pragma("unroll") for (u32 i = 0; i < A3_WPL; ++i) \
                    acc[i] |= x[3u + (O) + i]; } while (0)
                switch (o | (rr ? 0u : 64u)) {
#if 0 <= A3_SPAN
                case 0: A3_F(0u); break;
                case 64: A3_F0(0u); break;
#endif
#if 1 <= A3_SPAN
                case 1: A3_F(1u); break;
                case 65: A3_F0(1u); break;
#endif
#if 2 <= A3_SPAN
                case 2: A3_F(2u); break;
                case 66: A3_F0(2u); break;
#endif
#if 3 <= A3_SPAN
                case 3: A3_F(3u); break;
                case 67: A3_F0(3u); break;
#endif
#if 4 <= A3_SPAN
                case 4: A3_F(4u); break;
                case 68: A3_F0(4u); break;
#endif
#if 5 <= A3_SPAN
                case 5: A3_F(5u); break;
                case 69: A3_F0(5u); break;
#endif
#if 6 <= A3_SPAN
                case 6: A3_F(6u); break;
                case 70: A3_F0(6u); break;
#endif
#if 7 <= A3_SPAN
                case 7: A3_F(7u); break;
                case 71: A3_F0(7u); break;
#endif
#if 8 <= A3_SPAN
                case 8: A3_F(8u); break;
                case 72: A3_F0(8u); break;
#endif
#if 9 <= A3_SPAN
                case 9: A3_F(9u); break;
                case 73: A3_F0(9u); break;
#endif
#if 10 <= A3_SPAN
                case 10: A3_F(10u); break;
                case 74: A3_F0(10u); break;
#endif
#if 11 <= A3_SPAN
                case 11: A3_F(11u); break;
                case 75: A3_F0(11u); break;
#endif
#if 12 <= A3_SPAN
                case 12: A3_F(12u); break;
                case 76: A3_F0(12u); break;
#endif
#if 13 <= A3_SPAN
                case 13: A3_F(13u); break;
                case 77: A3_F0(13u); break;
#endif
#if 14 <= A3_SPAN
                case 14: A3_F(14u); break;
                case 78: A3_F0(14u); break;
#endif
#if 15 <= A3_SPAN
                case 15: A3_F(15u); break;
                case 79: A3_F0(15u); break;
#endif
#if 16 <= A3_SPAN
                case 16: A3_F(16u); break;
                case 80: A3_F0(16u); break;
#endif
#if 17 <= A3_SPAN
                case 17: A3_F(17u); break;
                case 81: A3_F0(17u); break;
#endif
#if 18 <= A3_SPAN
                case 18: A3_F(18u); break;
                case 82: A3_F0(18u); break;
#endif
#if 19 <= A3_SPAN
                case 19: A3_F(19u); break;
                case 83: A3_F0(19u); break;
#endif
#if 20 <= A3_SPAN
                case 20: A3_F(20u); break;
                case 84: A3_F0(20u); break;
#endif
#if 21 <= A3_SPAN
                case 21: A3_F(21u); break;
                case 85: A3_F0(21u); break;
#endif
#if 22 <= A3_SPAN
                case 22: A3_F(22u); break;
                case 86: A3_F0(22u); break;
#endif
#if 23 <= A3_SPAN
                case 23: A3_F(23u); break;
                case 87: A3_F0(23u); break;
#endif
#if 24 <= A3_SPAN
                case 24: A3_F(24u); break;
                case 88: A3_F0(24u); break;
#endif
#if 25 <= A3_SPAN
                case 25: A3_F(25u); break;
                case 89: A3_F0(25u); break;
#endif
#if 26 <= A3_SPAN
                case 26: A3_F(26u); break;
                case 90: A3_F0(26u); break;
#endif
#if 27 <= A3_SPAN
                case 27: A3_F(27u); break;
                case 91: A3_F0(27u); break;
#endif
#if 28 <= A3_SPAN
                case 28: A3_F(28u); break;
                case 92: A3_F0(28u); break;
#endif
#if 29 <= A3_SPAN
                case 29: A3_F(29u); break;
                case 93: A3_F0(29u); break;
#endif
#if 30 <= A3_SPAN
                case 30: A3_F(30u); break;
                case 94: A3_F0(30u); break;
#endif
#if 31 <= A3_SPAN
                case 31: A3_F(31u); break;
                case 95: A3_F0(31u); break;
#endif
#if 32 <= A3_SPAN
                case 32: A3_F(32u); break;
                case 96: A3_F0(32u); break;
#endif
                default: break;                           // plan guarantees o <= A3_SPAN
                }
            }
        }
    }

    // Pending: one entry per word with unresolved evens (cr << pshift | word <<
    // 32 | miss mask), a predicated store per word; phase B expands the bits.
    u32 cnt = 0;
    #pragma unroll
    for (u32 i = 0; i < A3_WPL; ++i) {
        const u32 kb = k0 + 32u * i;
        if (write_checked && live && kb < ihi)
            ((__global u32*)(checked + (u64)cr * chk_stride))[kb >> 5] = acc[i];
        cnt += (acc[i] != ~0u) ? 1u : 0u;
    }
    const u32 off = sub_group_scan_exclusive_add(cnt);
    const u32 tot = sub_group_reduce_add(cnt);
    u32 base = 0;
    if (get_sub_group_local_id() == 0 && tot) base = atomic_add(pending_count, tot);
    base = sub_group_broadcast(base, 0) + off;
#ifdef ABL_NOPEND
    if (cnt == 0xFFFFFFFFu) {                          // timing only: no pending extraction
#else
    if (cnt) {
#endif
        const u64 hi = ((u64)cr << pshift) | ((u64)(k0 >> 5) << 32);
        #pragma unroll
        for (u32 i = 0; i < A3_WPL; ++i) {
            if (acc[i] != ~0u) {
                if (base < pending_cap) pending[base] = (hi + ((u64)i << 32)) | (u64)~acc[i];
                ++base;
            }
        }
    }
}

// Phase B: one lane per pending word (cr << pshift | word << 32 | miss mask),
// probing each unresolved even i = 32 word + b against its class's rest list
// (entries U | k << 28, bit t = U + i of class k). Unresolved -> leftovers.
#ifndef PROBE_BATCH
#define PROBE_BATCH 1             // M = 30030: 1 -9% phase B vs 4 (2 -6%, 8 +17%)
#endif
/// Phase B of the sparse checker (grid-stride over the pending words).
/// @param rest,rtab   phase-B probes per class (rtab[cr] offset,
///                    rtab[CR + cr] count), entries U | k << 28.
/// @param leftovers,leftover_count,leftover_cap  out: unresolved evens as
///                    cr << 32 | i (host: CPU Miller-Rabin fallback).
/// @pre   runs after scm_check_a of the same segment; pshift - 32 in [1, 31].
/// @post  every pending even is either resolved by a genuine witness
///        (checked bit set iff write_checked) or appended to leftovers;
///        *leftover_count may exceed leftover_cap (host fails closed).
__kernel void scm_check_b(
        __global const u32* restrict cmap32,
        const u32 stride32,
        __global u64* restrict checked,
        const u32 chk_stride,
        __global const u32* restrict rest,
        __global const u32* restrict rtab,
        __global const u64* restrict pending,
        __global const u32* restrict pending_count,
        const u32 pending_cap,
        __global u64* restrict leftovers,
        __global u32* restrict leftover_count,
        const u32 leftover_cap,
        const u32 write_checked,
        const u32 pshift)
{
    const u32 n = min(*pending_count, pending_cap);
    for (u32 x = get_global_id(0); x < n; x += get_global_size(0)) {
        const u64 e = pending[x];
        const u32 cr = (u32)(e >> pshift), w = (u32)(e >> 32) & ((1u << (pshift - 32u)) - 1u);
        u32 miss = (u32)e;
        const u32 ro = rtab[cr], rc = rtab[SCM_CR + cr];
        while (miss) {
            const u32 i = 32u * w + (u32)ctz(miss);
            miss &= miss - 1u;
            bool ok = false;
            for (u32 j = 0; j < rc && !ok; j += PROBE_BATCH) {
                u32 hit = 0u;
                #pragma unroll
                for (u32 u = 0; u < PROBE_BATCH; ++u) {
                    if (j + u >= rc) break;
                    const u32 v = rest[ro + j + u];
                    const u32 t = (v & 0x0FFFFFFFu) + i;
                    const u32 k = v >> 28;
                    hit |= (cmap32[k * stride32 + (t >> 5)] >> (t & 31u)) & 1u;
                }
                ok = hit != 0u;
            }
            if (ok) {
                if (write_checked)
                    atomic_or((volatile __global u32*)(checked + (u64)cr * chk_stride) + (i >> 5),
                              1u << (i & 31u));
            } else {
                const u32 slot = atomic_inc(leftover_count);
                if (slot < leftover_cap) leftovers[slot] = ((u64)cr << 32) | (u64)i;
            }
        }
    }
}

/// Geometry handshake: the host checks the module's compiled modulus, class
/// count and residues against its plan before any segment (a module built for
/// another geometry would skip classes and silently count their evens as
/// verified).
/// @param out  shared buffer of at least SCM_K + 10 u32.
/// @post  out = {M, K, residues[K], A3_WPL, A3_SPAN, SCM_MAXD, TILE_WORDS,
///        LARGE_PACK_PPI, LARGE_PACK_SG, MAX_GROUPS, SCM_META}; only work-item
///        0 writes.
__kernel void scm_geometry(__global u32* restrict out)
{
    if (get_global_id(0) != 0) return;
    out[0] = SCM_M;
    out[1] = SCM_K;
    for (u32 k = 0; k < SCM_K; ++k) out[2 + k] = SCM_RES[k];
    // layout constants the host mirrors (a mismatch in the item width would
    // leave evens untested, in the packing format sieve the wrong primes)
    out[2 + SCM_K] = A3_WPL;
    out[3 + SCM_K] = A3_SPAN;
    out[4 + SCM_K] = SCM_MAXD;
    out[5 + SCM_K] = TILE_WORDS;
    out[6 + SCM_K] = LARGE_PACK_PPI;
    out[7 + SCM_K] = LARGE_PACK_SG;
    out[8 + SCM_K] = MAX_GROUPS;
    out[9 + SCM_K] = SCM_META;
}
