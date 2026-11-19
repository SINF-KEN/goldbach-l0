#include <filesystem>
#include "primes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gb {

std::vector<uint32_t> odd_primes_up_to(uint32_t n) {
    std::vector<bool> composite(size_t(n) + 1, false);
    for (uint32_t p = 3; p * p <= n; p += 2)
        if (!composite[p])
            for (uint32_t m = p * p; m <= n; m += 2 * p) composite[m] = true;

    std::vector<uint32_t> primes;
    // NOTE: 2 is deliberately excluded. The GPU prime map stores only odd
    // numbers, and p=2 would wrongly clear odd primes (its "odd multiple"
    // stepping is meaningless for even p). The CPU fallback uses its own
    // full prime logic, so it does not need this list to contain 2.
    for (uint32_t p = 3; p <= n; p += 2)
        if (!composite[p]) primes.push_back(p);
    return primes;
}

std::vector<uint64_t> small_prime_mask_table(const uint32_t* sieve_primes,
                                             uint32_t mask_count,
                                             uint64_t q_first) {
    std::vector<uint64_t> table(size_t(mask_count) * 64, 0ull);

    // Global bit g <-> q = q_first + 2g. The odd multiples of p are
    // q = m*p with m odd, i.e. q advances by 2p per mark, so the global bit
    // advances by p per mark: the marked bits are exactly g = r (mod p)
    // with r = bit of the first odd multiple >= q_first.
    //
    // Word w covers global bits [64w, 64w+64). Its mask must clear the
    // local bits b < 64 with (64w + b) = r (mod p), i.e.
    // b = (r - 64w) mod p. Phase = w mod p determines b; consecutive words
    // advance the phase by 64 mod p.
    for (uint32_t j = 0; j < mask_count; ++j) {
        const uint64_t p = sieve_primes[j];
        if (p < 3 || p >= 64) continue;   // kernel contract: phase fits u32, table sized for p < 64

        // r = bit of the first odd multiple of p that is >= q_first.
        // q_first is odd; the smallest odd m with m*p >= q_first is
        // m = ceil(q_first / p) rounded up to odd.
        uint64_t m = (q_first + p - 1) / p;
        if ((m & 1) == 0) m += 1;
        const uint64_t r = (m * p - q_first) >> 1;   // (m*p - q_first) is even

        // Build the 64 masks for phases 0..p-1, then replicate cyclically
        // to fill all 64 slots (for p < 64 several phases share a mask).
        uint64_t phase_mask[64] = {};
        for (uint32_t phase = 0; phase < p; ++phase) {
            // Local bits b with (64*phase + b) = r (mod p), b < 64.
            uint64_t b = (r + p - (uint64_t(64) % p) * phase % p) % p;
            uint64_t mask = 0;
            while (b < 64) {
                mask |= (1ull << b);
                b += p;
            }
            phase_mask[phase] = ~mask;   // AND mask: 1 = keep, 0 = clear
        }
        for (uint32_t phase = uint32_t(p); phase < 64; ++phase)   // p < 64
            phase_mask[phase] = phase_mask[phase - p];

        std::memcpy(&table[size_t(j) * 64], phase_mask, 64 * sizeof(uint64_t));
    }
    return table;
}

std::vector<uint32_t> segmented_odd_primes_up_to(uint64_t n) {
    std::vector<uint32_t> primes;
    if (n < 3) return primes;

    // Base primes up to sqrt(n) via the flat sieve (sqrt(4e18) = 2e9 fits a
    // u32; the base bitmap is only ~sqrt(n) bytes).
    const uint32_t sqrt_n = static_cast<uint32_t>(std::sqrt(double(n)) + 1);
    std::vector<uint32_t> base = odd_primes_up_to(std::max(sqrt_n, 3u));

    // Segmented sweep over [3, n] in odd-only blocks. Block size ~1M odds
    // (2 MB bitmap) keeps L2-resident while amortizing per-block setup.
    constexpr uint64_t BLOCK_ODDS = 1u << 20;
    std::vector<bool> seg(BLOCK_ODDS);

    for (uint64_t lo = 3; lo <= n; lo += 2 * BLOCK_ODDS) {
        const uint64_t hi = std::min(n, lo + 2 * BLOCK_ODDS - 2); // inclusive odd
        const uint64_t odds = (hi - lo) / 2 + 1;
        std::fill(seg.begin(), seg.begin() + odds, false);

        for (uint32_t p : base) {
            const uint64_t pp = uint64_t(p) * p;
            if (pp > hi) break;
            // First odd multiple of p in [lo, hi]: start at p*p, then step 2p.
            uint64_t m = (pp >= lo) ? pp : ((lo + p - 1) / p) * p;
            if ((m & 1) == 0) m += p;
            for (uint64_t q = m; q <= hi; q += 2ull * p)
                seg[(q - lo) >> 1] = true;
        }
        for (uint64_t i = 0; i < odds; ++i)
            if (!seg[i]) primes.push_back(uint32_t(lo + 2 * i));
    }
    return primes;
}

uint64_t prime_payload_hash(const uint32_t* p, size_t cnt) {
    // FNV-1a over 32-bit words. Each step h -> (h ^ x) * P is a bijection of
    // h for a fixed x and of x for a fixed h (P odd), so any one changed word
    // changes the result deterministically.
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < cnt; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

namespace {
constexpr char CACHE_MAGIC[4] = {'G', 'B', 'P', '2'};
constexpr size_t CACHE_HEADER = 28;   // magic, u64 limit, u64 count, u64 payload hash

/// Content check of a cached prime payload: the odd primes below 64 first,
/// then strictly ascending odd values <= lim. With the payload hash it also
/// guards against a writer bug (the hash only proves the bytes are the ones
/// written). O(cnt), sequential.
/// @return true iff p[0, cnt) passes.
bool prime_payload_ok(const uint32_t* p, size_t cnt, uint64_t lim) {
    static constexpr uint32_t head[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61};
    for (size_t i = 0; i < cnt && i < sizeof head / sizeof head[0]; ++i)
        if (p[i] != head[i]) return false;
    for (size_t i = 1; i < cnt; ++i)
        if (p[i] <= p[i - 1] || (p[i] & 1u) == 0) return false;
    return cnt == 0 || p[cnt - 1] <= lim;
}

/// Publishes @p primes as the cache @p path atomically: a unique temp file,
/// fsync, rename (writing the final path in place let a concurrent builder's
/// truncation expose a zero-filled hole to a third process). The cache is
/// optional: on any failure no file is left behind.
void write_cache(const std::string& path, uint64_t n, const std::vector<uint32_t>& primes) {
    const std::string tmp = path + ".tmp." + std::to_string(::getpid());
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return;
    const uint64_t cnt = primes.size();
    const uint64_t h = prime_payload_hash(primes.data(), primes.size());
    bool ok = std::fwrite(CACHE_MAGIC, 1, 4, f) == 4 &&
              std::fwrite(&n, 8, 1, f) == 1 &&
              std::fwrite(&cnt, 8, 1, f) == 1 &&
              std::fwrite(&h, 8, 1, f) == 1 &&
              std::fwrite(primes.data(), 4, primes.size(), f) == primes.size();
    ok = std::fflush(f) == 0 && ok;
    ok = ::fsync(::fileno(f)) == 0 && ok;
    ok = std::fclose(f) == 0 && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0)
        std::remove(tmp.c_str());
}
} // namespace

std::vector<uint32_t> segmented_odd_primes_up_to_cached(uint64_t n,
                                                        const char* cache_path) {
    if (cache_path) {
        PrimeCacheView v;
        if (PrimeCacheView::map_file(cache_path, n, false, v))
            return std::vector<uint32_t>(v.data(), v.data() + v.size());
    }
    auto primes = segmented_odd_primes_up_to(n);
    if (cache_path) write_cache(cache_path, n, primes);
    return primes;
}

// ------------------------------------------------------- cache view ----

void PrimeCacheView::release() noexcept {
    if (map_) {
        munmap(map_, map_len_);
        map_ = nullptr;
        map_len_ = 0;
    }
    owned_.clear();
    owned_.shrink_to_fit();
    ptr_ = nullptr;
    count_ = 0;
    pending_ = {};
}

bool PrimeCacheView::map_file(const std::string& file, uint64_t n, bool allow_larger,
                              PrimeCacheView& out, bool defer_check) {
    const int fd = ::open(file.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st{};
    char magic[4] = {};
    uint64_t lim = 0, cnt = 0, hash = 0;
    const char* why = nullptr;
    void* m = MAP_FAILED;
    if (fstat(fd, &st) != 0 || st.st_size < int64_t(CACHE_HEADER) || pread(fd, magic, 4, 0) != 4 ||
        pread(fd, &lim, 8, 4) != 8 || pread(fd, &cnt, 8, 12) != 8 || pread(fd, &hash, 8, 20) != 8)
        why = "short header";
    else if (std::memcmp(magic, CACHE_MAGIC, 4) != 0)
        why = "unknown format (older caches carry no payload hash)";
    else if (!(lim == n || (allow_larger && lim > n))) {
        ::close(fd);                     // another limit: not this run's file, not damaged
        return false;
    } else if (cnt == 0 || cnt >= (1ull << 32) || uint64_t(st.st_size) != CACHE_HEADER + cnt * 4)
        why = "size does not match the header";
    else if ((m = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0)) == MAP_FAILED)
        why = "mmap failed";
    ::close(fd);
    if (!why) {
        const auto* p = reinterpret_cast<const uint32_t*>(static_cast<const char*>(m) + CACHE_HEADER);
        if (defer_check) {}
        else if (prime_payload_hash(p, size_t(cnt)) != hash) why = "payload hash mismatch";
        else if (!prime_payload_ok(p, size_t(cnt), lim)) why = "payload content check failed";
        if (why) munmap(m, size_t(st.st_size));
        else {
            out.release();
            out.map_ = m;
            out.map_len_ = size_t(st.st_size);
            out.ptr_ = p;
            out.count_ = lim > n            // primes <= n only (a prefix: the payload is ascending)
                ? size_t(std::upper_bound(p, p + cnt, uint32_t(std::min<uint64_t>(n, UINT32_MAX))) - p)
                : size_t(cnt);
            out.pending_ = defer_check ? Pending{file, size_t(cnt), lim, hash} : Pending{};
            return true;
        }
    }
    std::fprintf(stderr, "warning: prime cache %s ignored (%s); regenerating\n", file.c_str(), why);
    return false;
}

const char* PrimeCacheView::deferred_check() const {
    if (pending_.file.empty()) return nullptr;
    const auto* p = reinterpret_cast<const uint32_t*>(static_cast<const char*>(map_) + CACHE_HEADER);
    const char* why = prime_payload_hash(p, pending_.cnt) != pending_.hash ? "payload hash mismatch"
                    : !prime_payload_ok(p, pending_.cnt, pending_.lim) ? "payload content check failed"
                    : nullptr;
    if (why) {
        std::fprintf(stderr, "error: prime cache %s failed its check (%s): renamed to %s.bad\n",
                     pending_.file.c_str(), why, pending_.file.c_str());
        std::rename(pending_.file.c_str(), (pending_.file + ".bad").c_str());
    }
    return why;
}

PrimeCacheView PrimeCacheView::load(uint64_t n, const char* cache_path, bool defer_check) {
    PrimeCacheView view;
    const char* path = cache_path ? cache_path : "base_primes.cache";
    // Keyed filename: caches for different limits coexist.
    const std::string keyed = std::string(path) + "." + std::to_string(n);
    if (map_file(keyed, n, false, view, defer_check)) return view;
    // Next: the smallest keyed cache <path>.<L> with L > n, exposing its
    // primes <= n, so every run below one big cache (e.g. sqrt(5e18)) shares
    // it instead of writing a ~430 MB file per distinct --limit.
    {
        namespace fs = std::filesystem;
        const fs::path p(path);
        const fs::path dir = p.has_parent_path() ? p.parent_path() : fs::path(".");
        const std::string stem = p.filename().string() + ".";
        uint64_t best = 0;
        std::error_code ec;
        for (const auto& de : fs::directory_iterator(dir, ec)) {
            const std::string name = de.path().filename().string();
            if (name.size() <= stem.size() || name.compare(0, stem.size(), stem) != 0) continue;
            const std::string num = name.substr(stem.size());
            if (num.find_first_not_of("0123456789") != std::string::npos) continue;
            const uint64_t L = std::strtoull(num.c_str(), nullptr, 10);
            if (L > n && (best == 0 || L < best)) best = L;
        }
        if (best && map_file((dir / (stem + std::to_string(best))).string(), n, true, view, defer_check))
            return view;
    }
    // Slow path (~2.6 s at sqrt(5e18)): generate, publish the keyed cache.
    view.owned_ = segmented_odd_primes_up_to(n);
    write_cache(keyed, n, view.owned_);
    view.ptr_ = view.owned_.data();
    view.count_ = view.owned_.size();
    return view;
}

// Combined small-prime masks for make_prime_map's fused mask stage.
// A prime's word mask depends only on (word index mod p), so a GROUP of
// primes with product L has a word mask that depends only on (word index
// mod L): entry t = AND over the group of mask_p[t mod p]. Greedy grouping
// in ascending order keeps every period <= max_period words, so the kernel
// needs one table load per group per word instead of one load plus one
// SLM read-modify-write per prime per word.
SmallPrimeGroups small_prime_group_masks(const uint32_t* sieve_primes,
                                         uint32_t mask_count, uint64_t q_first,
                                         uint32_t max_period) {
    return small_prime_group_masks_from(sieve_primes, mask_count,
        small_prime_mask_table(sieve_primes, mask_count, q_first), max_period);
}

// Class-map masks (class mode): index i <-> q = base + step*i (step 6 for the
// q = 1 / 5 (mod 6) arrays). Multiples of p are the i = r (mod p) with
// base + step*r = 0 (mod p); a prime dividing step (3 for step 6) has no
// multiple in the progression (all-ones masks). Same table layout as
// small_prime_mask_table (64 phase masks per prime, phase = word index mod p).
std::vector<uint64_t> small_prime_mask_table_step(const uint32_t* sieve_primes,
                                                  uint32_t mask_count,
                                                  uint64_t base, uint32_t step) {
    std::vector<uint64_t> table(size_t(mask_count) * 64, ~0ull);
    for (uint32_t j = 0; j < mask_count; ++j) {
        const uint64_t p = sieve_primes[j];
        if (p < 3 || p >= 64) { std::fill_n(&table[size_t(j) * 64], 64, 0ull); continue; }
        if (step % p == 0) continue;                       // no multiples: keep all
        uint64_t inv = 1;                                  // step^-1 mod p (p < 64: search)
        while ((inv * step) % p != 1) ++inv;
        const uint64_t r = ((p - base % p) % p) * inv % p;
        uint64_t phase_mask[64] = {};
        for (uint32_t phase = 0; phase < p; ++phase) {
            uint64_t b = (r + p - (uint64_t(64) % p) * phase % p) % p;
            uint64_t mask = 0;
            while (b < 64) { mask |= (1ull << b); b += p; }
            phase_mask[phase] = ~mask;
        }
        for (uint32_t phase = uint32_t(p); phase < 64; ++phase)   // p < 64
            phase_mask[phase] = phase_mask[phase - p];
        std::memcpy(&table[size_t(j) * 64], phase_mask, 64 * sizeof(uint64_t));
    }
    return table;
}

// Folds the @p extra primes after the mask primes (sieve_primes[mask_count],
// ...: 67, 71, ...) into the groups, each into the group of smallest period at
// that point (2026-11-18 MASK_XP_DEFAULT: the fused mask stage clears them with no
// extra loads; the tile's owned walk starts after them).
static void fold_extra_primes(std::vector<std::vector<uint32_t>>& groups, const uint32_t* sieve_primes,
                              uint32_t mask_count, uint32_t extra) {
    for (uint32_t e = 0; e < extra && !groups.empty(); ++e) {
        size_t best = 0;
        uint64_t bp = ~0ull;
        for (size_t g = 0; g < groups.size(); ++g) {
            uint64_t per = 1;
            for (uint32_t j : groups[g]) per *= sieve_primes[j];
            if (per < bp) { bp = per; best = g; }
        }
        groups[best].push_back(mask_count + e);
    }
}

// a^-1 mod m (gcd(a, m) = 1, m >= 1)
static uint64_t inv_mod(uint64_t a, uint64_t m) {
    int64_t g = int64_t(m), x = 0, y = 1, aa = int64_t(a % m);
    while (aa) {
        const int64_t q = g / aa, t = g - q * aa;
        g = aa; aa = t;
        const int64_t u = x - q * y;
        x = y; y = u;
    }
    return uint64_t(((x % int64_t(m)) + int64_t(m)) % int64_t(m));
}

SmallPrimeGroups small_prime_group_masks_from(const uint32_t* sieve_primes,
                                              uint32_t mask_count,
                                              const std::vector<uint64_t>& per_prime,
                                              uint32_t max_period, uint32_t skip, uint32_t extra) {
    SmallPrimeGroups g;
    std::vector<std::vector<uint32_t>> groups;   // indices into sieve_primes
    uint64_t prod = 0;
    for (uint32_t j = 0; j < mask_count; ++j) {
        const uint64_t p = sieve_primes[j];
        if (p < 3 || p >= 64) continue;
        // primes dividing skip (class maps: no multiples in any class) are
        // left out of the groups: all-ones masks would only widen a period
        if (skip % p == 0) continue;
        if (groups.empty() || prod * p > max_period) { groups.push_back({}); prod = 1; }
        groups.back().push_back(j);
        prod *= p;
    }
    fold_extra_primes(groups, sieve_primes, mask_count, extra);
    for (const auto& grp : groups) {
        uint64_t period = 1;
        for (uint32_t j : grp) period *= sieve_primes[j];
        g.desc.push_back(uint32_t(g.words.size()));   // offset
        g.desc.push_back(uint32_t(period));            // period
        for (uint64_t t = 0; t < period; ++t) {
            uint64_t m = ~0ull;
            for (uint32_t j : grp) {
                const uint64_t p = sieve_primes[j];
                if (p < 64) { m &= per_prime[size_t(j) * 64 + t % p]; continue; }
                // an extra prime (canonical tables only): marks at bit 0 (mod p)
                for (uint64_t b = (p - (64 * t) % p) % p; b < 64; b += p) m &= ~(1ull << b);
            }
            g.words.push_back(m);
        }
    }
    return g;
}

// Canonical group tables (class modes): every prime's marks at index 0 mod p
// (primes dividing skip_mod have none: all-ones), so the table depends on the
// primes only. A progression with marks at i = r_p (mod p) reads group g at
// word (x - Y_g) mod P_g (small_prime_group_offsets): the pattern is periodic
// in the bit index with period P_g and 64 is invertible mod P_g, so a bit
// shift X_g (the CRT of the r_p) is a whole-word shift Y_g = X_g / 64 mod P_g.
SmallPrimeGroups small_prime_group_masks_canonical(const uint32_t* sieve_primes,
                                                   uint32_t mask_count, uint32_t skip_mod,
                                                   uint32_t max_period, uint32_t extra) {
    std::vector<uint64_t> per = small_prime_mask_table_step(sieve_primes, mask_count, 0, 1);
    for (uint32_t j = 0; j < mask_count; ++j)
        if (skip_mod % sieve_primes[j] == 0)
            std::fill_n(&per[size_t(j) * 64], 64, ~0ull);
    return small_prime_group_masks_from(sieve_primes, mask_count, per, max_period, skip_mod, extra);
}

std::vector<uint32_t> small_prime_group_offsets(const uint32_t* sieve_primes,
                                                uint32_t mask_count, uint64_t base,
                                                uint32_t step, uint32_t max_period,
                                                uint32_t extra) {
    // the canonical grouping (skip_mod | step), then the extra primes
    std::vector<std::vector<uint32_t>> groups;
    uint64_t prod = 0;
    for (uint32_t j = 0; j < mask_count; ++j) {
        const uint64_t p = sieve_primes[j];
        if (p < 3 || p >= 64) continue;
        if (step % p == 0) continue;
        if (groups.empty() || prod * p > max_period) { groups.push_back({}); prod = 1; }
        groups.back().push_back(j);
        prod *= p;
    }
    fold_extra_primes(groups, sieve_primes, mask_count, extra);
    std::vector<uint32_t> ys;
    for (const auto& grp : groups) {
        uint64_t P = 1, X = 0;                  // running CRT: X mod P
        for (uint32_t j : grp) {
            const uint64_t p = sieve_primes[j];
            // marks at i = r (mod p): base + step r = 0 (mod p)
            const uint64_t r = ((p - base % p) % p) * inv_mod(step % p, p) % p;
            X += P * (((r + p - X % p) % p) * inv_mod(P % p, p) % p);   // X = r (mod p) too
            P *= p;
        }
        ys.push_back(uint32_t(X * inv_mod(64 % P, P) % P));   // 64 is invertible mod P (odd)
    }
    return ys;
}

// Deterministic Miller-Rabin for all uint64_t.
bool is_prime_u64(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t p : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL, 37ULL}) {
        if (n % p == 0) return n == p;
    }
    uint64_t d = n - 1;
    int s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }

    auto mulmod = [&](uint64_t a, uint64_t b) -> uint64_t {
        __uint128_t r = static_cast<__uint128_t>(a) * b;
        return static_cast<uint64_t>(r % n);
    };
    auto powmod = [&](uint64_t a, uint64_t e) -> uint64_t {
        uint64_t r = 1;
        a %= n;
        while (e) {
            if (e & 1) r = mulmod(r, a);
            a = mulmod(a, a);
            e >>= 1;
        }
        return r;
    };

    for (uint64_t a : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL, 37ULL}) {
        uint64_t x = powmod(a, d);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int i = 1; i < s; ++i) {
            x = mulmod(x, x);
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

uint64_t cpu_min_goldbach_prime(uint64_t n, const uint16_t* cands, size_t ncand, uint64_t cand_bound) {
    if (n < 4 || (n & 1)) return 0;
    if (n == 4) return 2;
    for (size_t i = 0; i < ncand; ++i) {
        const uint64_t p = cands[i];
        if (p > n - p) return 0;                 // every candidate up to n / 2 failed
        if (is_prime_u64(n - p)) return p;
    }
    for (uint64_t p = (cand_bound + 1) | 1; p <= n / 2; p += 2)
        if (is_prime_u64(p) && is_prime_u64(n - p)) return p;
    return 0;
}

bool is_prime_trial(uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0) return false;
    for (uint64_t d = 3; d <= n / d; d += 2)          // d * d would wrap for n > (2^32 - 1)^2
        if (n % d == 0) return false;
    return true;
}

} // namespace gb
