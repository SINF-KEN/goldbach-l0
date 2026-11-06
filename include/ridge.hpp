#pragma once
// Goldbach ridge tracking (dense checker path + CPU fallbacks).
//
// A ridge is the smallest prime p of an even n's Goldbach partition
// (n = p + q, both prime, no smaller prime works for n) — e.g.
//   4000000000169717704 = 4000000000169713571 + 4133  (ridge 4133).
// The phase-B checkers probe each pending even's candidate list in ascending
// order, so their FIRST hit is exactly that p (the mod-30 family probes the
// useful residues first and the futile ones — q = 3 / 5 only — last, which is
// exact whenever n > MAX_GB_PRIME + 5; the host validation below drops the
// rare tiny-n records where that ordering could differ). Every even whose
// ridge is at least the checker plan's member bound (Plan30::member_lt; 3000
// since v2.2: every phase-A funnel member is below it) fails phase A, passes
// through phase B and cannot be missed; ridges below that bound may be, so
// the verifier clamps the floor to the active checker's phase-A ceiling
// (clamp_floor_above; per segment for fallback checkers).
//
// The kernel appends (n, p) pairs above the floor to a bounded buffer
// (RIDGE_CAP); the host merges it into the RIDGE_KEEP largest, and every
// record is validated with exact integer arithmetic: p and q = n - p prime
// (deterministic 64-bit Miller-Rabin), and — before an entry may enter the
// top list — minimality (no smaller candidate p' with n - p' prime). An
// invalid record is dropped and counted, never trusted.
//
// The sparse class map path cannot compute ridges (its map stores 12 of the
// 5760 residue classes, so the smallest witness it finds is a different, much
// larger quantity); sparse runs only record evens resolved by the CPU
// fallback, where the true ascending scan happens anyway.
//
// Persistent store (GB_RIDGE_FILE, default goldbach_ridges.csv): one
// rank,even,p,q,segment,found row per top entry plus a header comment,
// rewritten atomically whenever the top changes. found = Unix time (s) the
// entry first entered the store (0 = unknown: rows from a v1 store); a
// re-run that finds a stored even again keeps the original time. It accumulates across runs; delete it
// to reset the leaderboard.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "config.hpp"
#include "primes.hpp"

namespace gb {

/// One tracked ridge: even n = p + q with p the smallest witness prime.
struct RidgeEntry {
    uint64_t n = 0;    // the even number
    uint64_t p = 0;    // the ridge: smallest prime of the partition
    uint64_t q = 0;    // n - p (prime)
    uint64_t seg = 0;  // segment start of the run that found it
    uint64_t found = 0;  // Unix time it first entered the store (0 = unknown)
};

/// Order: larger ridge first, equal ridge -> larger even first.
inline bool ridge_before(const RidgeEntry& a, const RidgeEntry& b) {
    return a.p != b.p ? a.p > b.p : a.n > b.n;
}

/// Top-K Goldbach ridge accumulator with exact validation and a persistent
/// store. Owned by run_gpu / run_gpu_sparse; fed by RidgeTracker::consider
/// (CPU-side witnesses) and RidgeTracker::merge (kernel append buffers).
class RidgeTracker {
public:
    /// @param cand  ascending odd candidate primes (the checker's list);
    ///              kept by pointer, must outlive the tracker. Used for the
    ///              exact minimality check of entries entering the top list.
    explicit RidgeTracker(const std::vector<uint32_t>* cand)
        : cand_(cand)
    {
        const char* off = std::getenv("GB_RIDGE");
        if (off && std::string(off) == "0") return;               // disabled
        floor_ = RIDGE_MIN_DEFAULT;
        if (const char* e = std::getenv("GB_RIDGE_MIN"))
            if (*e) floor_ = uint32_t(std::strtoul(e, nullptr, 0));
        if (floor_ < 3) floor_ = 3;
    }

    bool active() const { return floor_ != 0; }
    uint32_t floor() const { return floor_; }
    /// Recording floor after clamping above @p bound (phase-A funnel
    /// coverage: every even with a ridge at or above the floor must reach
    /// phase B, so the floor has to exceed what phase A can resolve).
    void clamp_floor_above(uint32_t bound) {
        if (floor_ != 0 && floor_ <= bound) {
            floor_ = bound + 1;
            std::fprintf(stderr, "note: ridge floor raised to %u (phase-A coverage %u)\n",
                         floor_, bound);
        }
    }
    /// Kernel-side recording threshold: the current Kth largest ridge (ties
    /// still recorded: a larger even can displace an equal-p entry).
    uint32_t thresh() const {
        return top_.size() >= RIDGE_KEEP ? std::max(floor_, uint32_t(top_.back().p))
                                         : floor_;
    }

    /// Raises the floor unconditionally (host-side floor adjustments such as
    /// the ridge pass's funnel-member margin; callers print their own note).
    void force_floor(uint32_t v) { if (v > floor_) floor_ = v; }

    uint64_t appends() const { return appends_; }
    uint64_t dropped() const { return dropped_; }
    bool overflowed() const { return overflow_; }
    const std::vector<RidgeEntry>& top() const { return top_; }

    /// Considers one (n, p) witness record (p claimed to be n's smallest
    /// partition prime). Validates it exactly and keeps it if it belongs in
    /// the top RIDGE_KEEP. Cheap unless the entry qualifies. @p found is the
    /// discovery time (Unix s); UINT64_MAX = now (fresh witnesses).
    void consider(uint64_t n, uint64_t p, uint64_t seg, uint64_t found = UINT64_MAX) {
        if (!active() || p < floor_) return;
        ++appends_;
        if (n < 4 || (n & 1) || p < 3 || (p & 1) == 0 || p >= n) {
            ++dropped_;
            return;
        }
        // cheap qualification first: only entries that belong in the top
        // RIDGE_KEEP pay for exact (Miller-Rabin) validation
        if (top_.size() >= RIDGE_KEEP) {
            const RidgeEntry& last = top_.back();
            if (p < last.p || (p == last.p && n <= last.n)) return;  // no qualifier
        }
        // already stored (a re-run of a covered range): keep the original
        // entry and its found time
        for (const RidgeEntry& t : top_)
            if (t.n == n) return;
        if (!is_prime_u64(p) || !is_prime_u64(n - p)) {
            ++dropped_;
            if (std::getenv("GB_RIDGE_DEBUG"))
                std::fprintf(stderr, "ridge drop (shape/pair): n=%llu p=%llu\n",
                             (unsigned long long)n, (unsigned long long)p);
            return;
        }
        // Exact minimality: no smaller candidate prime p' with n - p' prime.
        // (n - p' > 2 always: p' <= p - 2 and n - p is an odd prime >= 3.)
        if (cand_) {
            for (uint32_t pp : *cand_) {
                if (pp >= p) break;
                if (is_prime_u64(n - pp)) {
                    ++dropped_;
                    if (std::getenv("GB_RIDGE_DEBUG"))
                        std::fprintf(stderr, "ridge drop (not minimal): n=%llu p=%llu "
                                     "< p'=%llu witness\n",
                                     (unsigned long long)n, (unsigned long long)p,
                                     (unsigned long long)pp);
                    return;
                }
            }
        }
        if (found == UINT64_MAX) found = uint64_t(std::time(nullptr));
        RidgeEntry e{n, p, n - p, seg, found};
        top_.insert(std::upper_bound(top_.begin(), top_.end(), e, ridge_before), e);
        if (top_.size() > RIDGE_KEEP) top_.pop_back();
        changed_ = true;
    }

    /// Merges one kernel append buffer (word 0 low u32 = count, then
    /// (n, p) u64 pairs; layout of kernels/goldbach.cl phase-B checkers).
    void merge(const uint64_t* buf, uint64_t seg) {
        if (!active()) return;
        const uint32_t cnt = *reinterpret_cast<const volatile uint32_t*>(buf);
        if (cnt > RIDGE_CAP) overflow_ = true;    // kernel dropped the excess
        const uint32_t n = std::min(cnt, RIDGE_CAP);
        if (std::getenv("GB_RIDGE_DEBUG") && cnt) {
            std::fprintf(stderr, "ridge merge seg=%llu count=%u%s\n",
                         (unsigned long long)seg, cnt, cnt > RIDGE_CAP ? " (OVERFLOW)" : "");
            for (uint32_t i = 0; i < n && i < 12; ++i)
                std::fprintf(stderr, "  raw[%u] n=%llu p=%llu\n", i,
                             (unsigned long long)buf[1 + 2 * size_t(i)],
                             (unsigned long long)buf[2 + 2 * size_t(i)]);
        }
        for (uint32_t i = 0; i < n; ++i)
            consider(buf[1 + 2 * size_t(i)], buf[2 + 2 * size_t(i)], seg);
    }

    /// Loads a store written by save(); every entry is re-validated (a
    /// corrupted or hand-edited row is dropped, not trusted).
    /// @return true when the file existed and was readable.
    bool load(const char* path) {
        std::FILE* f = std::fopen(path, "rb");
        if (!f) return false;
        char line[256];
        while (std::fgets(line, sizeof line, f)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            unsigned long long r, n, p, q, s, t = 0;   // t: absent in v1 rows
            if (std::sscanf(line, "%llu,%llu,%llu,%llu,%llu,%llu", &r, &n, &p, &q, &s, &t) >= 5)
                consider(n, p, s, t);   // q column is recomputed and checked exactly
        }
        std::fclose(f);
        changed_ = false;                          // nothing new to write yet
        return true;
    }

    /// Writes the store atomically (temp file + rename) when the top changed
    /// since the last load()/save(). Best effort: I/O failures are reported
    /// to stderr and otherwise ignored (the store is optional).
    void save(const char* path) {
        if (!changed_) return;
        std::string out =
            "# goldbach-l0 ridge store v2 - top 30 largest Goldbach ridges\n"
            "# rank,even,small_prime_p,large_prime_q,segment_start,found_unix\n";
        char row[192];
        for (size_t i = 0; i < top_.size(); ++i) {
            const RidgeEntry& e = top_[i];
            std::snprintf(row, sizeof row, "%zu,%llu,%llu,%llu,%llu,%llu\n",
                          i + 1, (unsigned long long)e.n, (unsigned long long)e.p,
                          (unsigned long long)e.q, (unsigned long long)e.seg,
                          (unsigned long long)e.found);
            out += row;
        }
        const std::string tmp = std::string(path) + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "ridge store: cannot write %s\n", tmp.c_str()); return; }
        const bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
        if (std::fclose(f) == 0 && ok) std::rename(tmp.c_str(), path);
        else std::remove(tmp.c_str());
        changed_ = false;
    }

private:
    const std::vector<uint32_t>* cand_ = nullptr;
    std::vector<RidgeEntry> top_;      // sorted by ridge_before, <= RIDGE_KEEP
    uint32_t floor_ = 0;               // 0 = tracking disabled (GB_RIDGE=0)
    uint64_t appends_ = 0;             // records passed to consider()
    uint64_t dropped_ = 0;             // records rejected by exact validation
    bool overflow_ = false;            // a kernel buffer over its capacity
    bool changed_ = false;             // top_ changed since last save()/load()
};

} // namespace gb
