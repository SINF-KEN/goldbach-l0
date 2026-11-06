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

namespace {
/// Content check of a cached prime payload: the odd primes below 64 first,
/// then strictly ascending odd values <= lim. Catches truncation, holes
/// (zeros), and files mixed from different writers; the header alone (magic,
/// limit, count, size) cannot. O(cnt), sequential.
/// @return true iff p[0, cnt) passes.
bool prime_payload_ok(const uint32_t* p, size_t cnt, uint64_t lim) {
    static constexpr uint32_t head[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61};
    for (size_t i = 0; i < cnt && i < sizeof head / sizeof head[0]; ++i)
        if (p[i] != head[i]) return false;
    for (size_t i = 1; i < cnt; ++i)
        if (p[i] <= p[i - 1] || (p[i] & 1u) == 0) return false;
    return cnt == 0 || p[cnt - 1] <= lim;
}
} // namespace

std::vector<uint32_t> segmented_odd_primes_up_to_cached(uint64_t n,
                                                        const char* cache_path) {
    // Try the cache first: magic "GBPR" + u64 limit + u64 count + payload.
    if (cache_path) {
        std::FILE* f = std::fopen(cache_path, "rb");
        if (f) {
            char magic[4];
            uint64_t lim = 0, cnt = 0;
            bool ok = std::fread(magic, 1, 4, f) == 4 &&
                      std::memcmp(magic, "GBPR", 4) == 0 &&
                      std::fread(&lim, 8, 1, f) == 1 &&
                      std::fread(&cnt, 8, 1, f) == 1 &&
                      lim == n && cnt > 0 && cnt < (1ull << 32);
            if (ok) {
                std::vector<uint32_t> primes;
                primes.resize(size_t(cnt));
                ok = std::fread(primes.data(), 4, size_t(cnt), f) == size_t(cnt);
                std::fclose(f);
                if (ok && prime_payload_ok(primes.data(), primes.size(), lim)) return primes;
                if (ok) std::fprintf(stderr, "warning: prime cache %s failed validation; regenerating\n",
                                     cache_path);
            } else {
                std::fclose(f);
            }
        }
    }

    auto primes = segmented_odd_primes_up_to(n);
    if (cache_path) {
        // Publish atomically: a unique temp file, fsync, rename. Writing the
        // final path in place let a concurrent builder's truncation expose a
        // zero-filled hole to a third process that mapped the file.
        const std::string tmp = std::string(cache_path) + ".tmp." + std::to_string(::getpid());
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (f) {
            const uint64_t cnt = primes.size();
            const char magic[4] = {'G', 'B', 'P', 'R'};
            bool ok = std::fwrite(magic, 1, 4, f) == 4 &&
                      std::fwrite(&n, 8, 1, f) == 1 &&
                      std::fwrite(&cnt, 8, 1, f) == 1 &&
                      std::fwrite(primes.data(), 4, primes.size(), f) == primes.size();
            ok = std::fflush(f) == 0 && ok;
            ok = ::fsync(::fileno(f)) == 0 && ok;
            ok = std::fclose(f) == 0 && ok;
            if (!ok || std::rename(tmp.c_str(), cache_path) != 0)
                std::remove(tmp.c_str());     // the cache is optional: just do without it
        }
    }
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
}

PrimeCacheView PrimeCacheView::load(uint64_t n, const char* cache_path) {
    PrimeCacheView view;
    const char* path = cache_path ? cache_path : "base_primes.cache";
    // Keyed filename: caches for different limits coexist instead of every
    // small-limit run regenerating (and clobbering) the shared 392 MB file.
    const std::string keyed = std::string(path) + "." + std::to_string(n);

    // `allow_larger`: accept a cache built for lim >= n and expose only its
    // primes <= n (a prefix: the payload is ascending), so every run below
    // one big cache (e.g. sqrt(5e18)) shares it instead of writing its own
    // ~430 MB file per distinct --limit.
    auto try_mmap = [&view, n](const std::string& file,
                               bool allow_larger = false) -> bool {
        // Validate + mmap the cache payload (header: "GBPR", u64 lim, u64
        // cnt, then cnt u32 primes). No payload copy at all.
        int fd = ::open(file.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        bool ok = fstat(fd, &st) == 0 && st.st_size >= 32;
        char magic[4] = {};
        uint64_t lim = 0, cnt = 0;
        if (ok && pread(fd, magic, 4, 0) == 4 &&
            std::memcmp(magic, "GBPR", 4) == 0 &&
            pread(fd, &lim, 8, 4) == 8 && pread(fd, &cnt, 8, 12) == 8 &&
            (lim == n || (allow_larger && lim > n)) && cnt > 0 && cnt < (1ull << 32) &&
            st.st_size >= int64_t(20 + cnt * 4)) {
            void* m = mmap(nullptr, size_t(st.st_size), PROT_READ,
                           MAP_PRIVATE, fd, 0);
            if (m != MAP_FAILED) {
                view.map_ = m;
                view.map_len_ = size_t(st.st_size);
                view.ptr_ = reinterpret_cast<const uint32_t*>(
                    static_cast<const char*>(m) + 20);
                view.count_ = size_t(cnt);
                if (lim > n)   // primes <= n only
                    view.count_ = size_t(std::upper_bound(view.ptr_, view.ptr_ + cnt,
                                                          uint64_t(n) > UINT32_MAX ? UINT32_MAX : uint32_t(n)) -
                                         view.ptr_);
                if (!prime_payload_ok(view.ptr_, view.count_, n)) {   // the part this run uses
                    std::fprintf(stderr, "warning: prime cache %s failed validation; ignoring it\n",
                                 file.c_str());
                    munmap(m, size_t(st.st_size));
                    view.map_ = nullptr;
                    view.map_len_ = 0;
                    view.ptr_ = nullptr;
                    view.count_ = 0;
                }
            }
        }
        ::close(fd);
        return view.ptr_ != nullptr;
    };

    // Fast paths: the per-limit cache file, then the legacy shared file
    // (read-only) so existing caches keep working.
    if (try_mmap(keyed)) return view;
    if (try_mmap(path)) return view;
    // Next: the smallest existing keyed cache <path>.<L> with L > n.
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
        if (best && try_mmap((dir / (stem + std::to_string(best))).string(), true))
            return view;
    }

    // Slow path: generate (writing the keyed cache), keep owned.
    view.owned_ = segmented_odd_primes_up_to_cached(n, keyed.c_str());
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

SmallPrimeGroups small_prime_group_masks_from(const uint32_t* sieve_primes,
                                              uint32_t mask_count,
                                              const std::vector<uint64_t>& per_prime,
                                              uint32_t max_period) {
    SmallPrimeGroups g;
    std::vector<std::vector<uint32_t>> groups;   // indices into sieve_primes
    uint64_t prod = 0;
    for (uint32_t j = 0; j < mask_count; ++j) {
        const uint64_t p = sieve_primes[j];
        if (p < 3 || p >= 64) continue;
        if (groups.empty() || prod * p > max_period) { groups.push_back({}); prod = 1; }
        groups.back().push_back(j);
        prod *= p;
    }
    for (const auto& grp : groups) {
        uint64_t period = 1;
        for (uint32_t j : grp) period *= sieve_primes[j];
        g.desc.push_back(uint32_t(g.words.size()));   // offset
        g.desc.push_back(uint32_t(period));            // period
        for (uint64_t t = 0; t < period; ++t) {
            uint64_t m = ~0ull;
            for (uint32_t j : grp) m &= per_prime[size_t(j) * 64 + t % sieve_primes[j]];
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
                                                   uint32_t max_period) {
    std::vector<uint64_t> per = small_prime_mask_table_step(sieve_primes, mask_count, 0, 1);
    for (uint32_t j = 0; j < mask_count; ++j)
        if (skip_mod % sieve_primes[j] == 0)
            std::fill_n(&per[size_t(j) * 64], 64, ~0ull);
    return small_prime_group_masks_from(sieve_primes, mask_count, per, max_period);
}

std::vector<uint32_t> small_prime_group_offsets(const uint32_t* sieve_primes,
                                                uint32_t mask_count, uint64_t base,
                                                uint32_t step, uint32_t max_period) {
    std::vector<uint32_t> ys;
    uint64_t P = 0, X = 0;                      // running CRT: X mod P
    auto flush = [&]() {
        if (!P) return;
        uint64_t inv = 1;                       // 64^-1 mod P (P odd, <= max_period)
        while ((inv * 64) % P != 1 % P) ++inv;
        ys.push_back(uint32_t((X % P) * inv % P));
    };
    for (uint32_t j = 0; j < mask_count; ++j) {
        const uint64_t p = sieve_primes[j];
        if (p < 3 || p >= 64) continue;
        if (P == 0 || P * p > max_period) { flush(); P = 1; X = 0; }
        uint64_t r = 0;                         // marks at i = r (mod p); any r if p | step
        if (step % p != 0) {
            uint64_t sinv = 1;
            while ((sinv * step) % p != 1) ++sinv;
            r = ((p - base % p) % p) * sinv % p;
        }
        uint64_t x = X;                         // x = X (mod P), x = r (mod p)
        while (x % p != r) x += P;
        X = x; P *= p;
    }
    flush();
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

bool is_prime_trial(uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0) return false;
    for (uint64_t d = 3; d <= n / d; d += 2)          // d * d would wrap for n > (2^32 - 1)^2
        if (n % d == 0) return false;
    return true;
}

} // namespace gb
