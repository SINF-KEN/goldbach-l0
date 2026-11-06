#pragma once
// CPU-side prime generation and a standalone 64-bit primality test used by
// the (rare) CPU fallback path and the reference checker.

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gb {

/// Odd primes 3, 5, 7, ... up to and including @p n, by a flat Eratosthenes
/// sieve (O(n) bits). 2 is excluded on purpose: the GPU prime map only stores
/// odd numbers.
/// @pre    n < 65535^2 (~2^32 - 2^17): the u32 p * p and multiple steps must not wrap.
/// @return the odd primes <= n, ascending (empty for n < 3).
std::vector<uint32_t> odd_primes_up_to(uint32_t n);

/// Segmented variant of odd_primes_up_to for very large n (e.g. sqrt(4e18) =
/// 2e9, where the flat sieve's O(n) bitmap would need ~2 GB). Uses an O(sqrt n)
/// base-prime sieve plus O(block) segment bitmaps; same output, same ordering.
/// @pre    n < 2^32 (results are stored as u32).
/// @return the odd primes <= n, ascending (empty for n < 3).
std::vector<uint32_t> segmented_odd_primes_up_to(uint64_t n);

/// segmented_odd_primes_up_to with a binary disk cache ("GBPR", u64 limit,
/// u64 count, u32 primes): loads @p cache_path when its limit equals @p n
/// and its payload passes the content check (the odd primes < 64 first, then
/// strictly ascending odd values <= n), otherwise generates the list and
/// publishes the cache atomically (temp file, fsync, rename).
/// @param cache_path  cache file, or nullptr to skip caching.
/// @pre    n < 2^32.
/// @post   a damaged cache is never trusted; a failed write leaves no file
///         behind (the cache is optional).
/// @return the odd primes <= n, ascending.
std::vector<uint32_t> segmented_odd_primes_up_to_cached(uint64_t n,
                                                        const char* cache_path);

/// Read-only view of a base-prime cache file, mmap'd when possible so the
/// payload is never copied through an intermediate vector (a 392 MB cache
/// would otherwise be read twice: once by fread, once by the uploader).
/// Falls back to an owned vector when the file is missing/invalid. The view
/// is valid for the lifetime of the object; move-only.
class PrimeCacheView {
public:
    PrimeCacheView() = default;
    PrimeCacheView(const PrimeCacheView&) = delete;
    PrimeCacheView& operator=(const PrimeCacheView&) = delete;
    PrimeCacheView(PrimeCacheView&& other) noexcept { move_from(other); }
    PrimeCacheView& operator=(PrimeCacheView&& other) noexcept {
        if (this != &other) { release(); move_from(other); }
        return *this;
    }
    ~PrimeCacheView() { release(); }

    /// Loads the odd primes <= @p n. Tries, in order: the keyed file
    /// <cache_path>.<n>, the legacy file <cache_path> (limit == n), the
    /// smallest keyed cache with a larger limit (exposing only its prefix
    /// <= n); on a miss generates the list and writes <cache_path>.<n>.
    /// A mapped file is used only if the part this run uses passes the
    /// content check (see segmented_odd_primes_up_to_cached).
    /// @param n           prime limit (inclusive), < 2^32.
    /// @param cache_path  cache base name; nullptr means "base_primes.cache".
    /// @return a view of the odd primes <= n, ascending.
    static PrimeCacheView load(uint64_t n, const char* cache_path);

    /// @return pointer to the first prime, or nullptr for an empty view.
    const uint32_t* data() const { return ptr_; }
    /// @return number of primes in the view.
    size_t size() const { return count_; }

private:
    void move_from(PrimeCacheView& other) noexcept {
        owned_ = std::move(other.owned_);
        ptr_ = other.ptr_;
        count_ = other.count_;
        map_ = other.map_;
        map_len_ = other.map_len_;
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.map_ = nullptr;
        other.map_len_ = 0;
    }
    void release() noexcept;

    std::vector<uint32_t> owned_;
    const uint32_t* ptr_ = nullptr;
    size_t count_ = 0;
    void* map_ = nullptr;
    size_t map_len_ = 0;
};

/// Small-prime word-mask table for the GPU sieve (see goldbach.cl stage 3a).
///
/// For each of the first @p mask_count odd primes produces 64 uint64 masks
/// indexed by "phase" = (global word index) mod p. mask[p_idx*64 + phase] has
/// a 0 bit at every local bit b < 64 with (64*word_index + b) mod p == r,
/// where r is the global bit of the first odd multiple of p relative to
/// q_first. Applying tile[w] &= mask[phase] (phase advancing by 64 mod p per
/// word) clears all odd multiples of p without any per-bit RMW.
///
/// @param sieve_primes  odd primes; must match the kernel's sieve_primes[0..mask_count).
/// @param mask_count    number of primes to tabulate.
/// @param q_first       q of map bit 0 (bit i <-> q_first + 2i).
/// @pre    q_first is ODD (the verifier guarantees this).
/// @post   entries of primes outside [3, 64) are left all-zero (the kernel
///         never reads them).
/// @return exactly mask_count * 64 AND masks (1 = keep, 0 = clear).
std::vector<uint64_t> small_prime_mask_table(const uint32_t* sieve_primes,
                                             uint32_t mask_count,
                                             uint64_t q_first);

/// Grouped small-prime masks (make_prime_map fused mask stage): `words` holds
/// each group's period-long mask table back to back; `desc` holds
/// (offset, period) per group. Groups are formed greedily with period <=
/// max_period words (see primes.cpp).
struct SmallPrimeGroups {
    std::vector<uint64_t> words;
    std::vector<uint32_t> desc;
    /// @return number of groups.
    uint32_t count() const { return uint32_t(desc.size() / 2); }
};

/// Grouped masks for the dense odd map starting at @p q_first:
/// small_prime_group_masks_from(small_prime_mask_table(...)).
/// @pre    q_first is odd.
/// @return one period-long AND-mask table per group; group g's word x uses
///         entry x mod period_g.
SmallPrimeGroups small_prime_group_masks(const uint32_t* sieve_primes,
                                         uint32_t mask_count, uint64_t q_first,
                                         uint32_t max_period = 4096);
/// Groups a precomputed per-prime table (small_prime_mask_table or
/// small_prime_mask_table_step): primes in [3, 64) are grouped greedily in
/// ascending order while the product stays <= @p max_period; entry t of a
/// group is the AND of its primes' masks at phase t mod p. The grouping
/// depends only on the primes.
/// @param per_prime  mask_count * 64 masks, layout of small_prime_mask_table.
/// @return the group tables and (offset, period) descriptors.
SmallPrimeGroups small_prime_group_masks_from(const uint32_t* sieve_primes,
                                              uint32_t mask_count,
                                              const std::vector<uint64_t>& per_prime,
                                              uint32_t max_period = 4096);
/// Canonical group tables (class modes): every prime's marks at index 0 mod p,
/// so the tables depend on the primes only; a progression reads word x of
/// group g at table word (x - Y_g) mod P_g (Y_g from small_prime_group_offsets).
/// @param skip_mod  primes dividing it get all-ones masks (no multiples in
///                  any class, e.g. 3 for the mod-6 map).
/// @return canonical tables with the same grouping as small_prime_group_masks_from.
SmallPrimeGroups small_prime_group_masks_canonical(const uint32_t* sieve_primes,
                                                   uint32_t mask_count, uint32_t skip_mod,
                                                   uint32_t max_period = 4096);
/// Per-group word offsets Y_g for the progression q = base + step*i against
/// the canonical tables: Y_g = X_g * 64^-1 mod P_g, X_g the CRT of each
/// prime's first-mark residue.
/// @pre    grouping arguments identical to the small_prime_group_masks_canonical call.
/// @return one offset per group, in group order.
std::vector<uint32_t> small_prime_group_offsets(const uint32_t* sieve_primes,
                                                uint32_t mask_count, uint64_t base,
                                                uint32_t step, uint32_t max_period = 4096);
/// Class-map variant of small_prime_mask_table: index i <-> q = base + step*i
/// (class mode: step 6). A prime dividing @p step has no multiples (all-ones
/// masks); primes outside [3, 64) get all-zero entries.
/// @return exactly mask_count * 64 AND masks, same layout as small_prime_mask_table.
std::vector<uint64_t> small_prime_mask_table_step(const uint32_t* sieve_primes,
                                                  uint32_t mask_count,
                                                  uint64_t base, uint32_t step);

/// Largest limit the GPU paths accept: 2^64 - 2^40. Host first-hit math
/// (q_first + p - 1, m * p, the wheel/bucket next hit) adds at most ~2^38 to a
/// value <= limit, so this margin keeps every intermediate below 2^64; above
/// it a wrapped hit would silently skip a mark (a composite left as prime).
inline constexpr uint64_t MAX_SUPPORTED_LIMIT = 0ull - (1ull << 40);

/// Fails closed above MAX_SUPPORTED_LIMIT.
/// @post   returns only if limit <= MAX_SUPPORTED_LIMIT (no other effect).
/// @throws std::runtime_error if @p limit > MAX_SUPPORTED_LIMIT.
inline void require_supported_limit(uint64_t limit) {
    if (limit > MAX_SUPPORTED_LIMIT)
        throw std::runtime_error("limit " + std::to_string(limit) +
                                 " is above the supported maximum 2^64 - 2^40 = " +
                                 std::to_string(MAX_SUPPORTED_LIMIT));
}

/// Base-prime bound for a run up to @p limit: floor(sqrt(limit)) + 16 (the
/// slack marks nothing: p^2 > limit), saturated at 2^32 - 1 — every prime
/// <= sqrt(2^64 - 1) fits a u32.
/// @pre    none (any u64; the double sqrt errs by far less than the slack).
/// @return B >= floor(sqrt(limit)), so the odd primes <= B include every
///         prime factor <= sqrt(q) of every odd composite q <= limit. The former inline
/// uint32_t(sqrt(double(limit)) + 16) wrapped to ~16 for
/// limit > (2^32 - 16)^2 ~ 1.8446744e19 and sieved with the primes < 64 only.
inline uint32_t base_prime_bound(uint64_t limit) {
    const double r = std::sqrt(double(limit)) + 16.0;
    return r >= 4294967295.0 ? 0xFFFFFFFFu : uint32_t(r);
}

/// Deterministic Miller-Rabin (bases 2..37), exact for all uint64_t.
/// @return true iff @p n is prime.
bool is_prime_u64(uint64_t n);

/// Trial-division primality, O(sqrt n); used by the reference checker only.
/// @return true iff @p n is prime.
bool is_prime_trial(uint64_t n);

} // namespace gb
