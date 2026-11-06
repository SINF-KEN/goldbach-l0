// stats_ref — independent CPU reference for the exact Goldbach statistics
// (include/gstats.hpp): p(n) of every even n in [start, limit] by an
// ascending Miller-Rabin scan, prime and twin-prime counts in [start, limit].
// Prints the same GBSTATS fields as the verifier's log line (hist, primes,
// twins, primes30, twins30, first occurrences, prime gaps: gaps, gmom, gmax,
// gtail, and ghist = the full gap histogram by g / 2), so the two can be diffed:
//   ./build/stats_ref 4000000000000000000 4000000000020000000 [threads]
//   ./build/stats_ref --known   (ctest: p(n) against T. Oliveira e Silva's
//                                published S(p) table, GPU-free)
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include "primes.hpp"

namespace {
// p(n): smallest odd prime p with n - p prime (ascending Miller-Rabin scan).
uint64_t pn(uint64_t n)
{
    for (uint64_t p = 3; p <= n / 2; p += 2)
        if (gb::is_prime_u64(p) && gb::is_prime_u64(n - p)) return p;
    return 0;
}

// Published values (T. Oliveira e Silva, goldbach/t0.txt, verification to
// 4e18): S(p) = least even n with p(n) = p; rec = p is a record-holder (every
// even below S(p) has p(n) < p), checked over the 2000 evens below S(p).
int known()
{
    struct K { uint64_t p, S; bool rec; };
    const K ks[] = {
        {9781, 3325581707333960528ull, true}, {9629, 2795935116574469638ull, true},
        {9341, 906030579562279642ull, true},  {9161, 887123803077837868ull, true},
        {8941, 555274351556750822ull, true},  {8933, 258549426916149682ull, true},
        {8501, 255329126688555994ull, true},  {8443, 121005022304007026ull, true},
        {8647, 2725058671961297876ull, false}, {9203, 1348113579429547486ull, false},
        {8387, 3878297017437646306ull, false}, {7307, 370581106766909188ull, false},
        {7, 30, false}, {19, 98, true}, {23, 220, true}, {31, 308, true}};
    int fail = 0;
    for (const K& k : ks) {
        const uint64_t got = pn(k.S);
        uint64_t worst = 0, worst_n = 0;
        if (k.rec)
            for (uint64_t n = (k.S > 4002 ? k.S - 4000 : 6); n < k.S; n += 2) {
                const uint64_t q = pn(n);
                if (q > worst) { worst = q; worst_n = n; }
            }
        const bool ok = got == k.p && (!k.rec || worst < k.p);
        std::printf("  %s p(%llu) = %llu (expect %llu)%s\n", ok ? "ok  " : "FAIL",
                    (unsigned long long)k.S, (unsigned long long)got, (unsigned long long)k.p,
                    k.rec ? (" ; max p(n) over the 2000 evens below: " + std::to_string(worst) +
                             " at " + std::to_string(worst_n)).c_str() : "");
        fail += !ok;
    }
    std::printf("stats_ref --known: %d failure(s) -> %s\n", fail, fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--known") return known();
    if (argc < 3) { std::fprintf(stderr, "usage: stats_ref START LIMIT [THREADS]\n"); return 2; }
    const uint64_t start = std::strtoull(argv[1], nullptr, 10), limit = std::strtoull(argv[2], nullptr, 10);
    const unsigned nt = argc > 3 ? unsigned(std::atoi(argv[3])) : std::max(1u, std::thread::hardware_concurrency());
    if (start % 2 || limit % 2 || start < 1000000 || limit < start) { std::fprintf(stderr, "bad range\n"); return 2; }
    const std::vector<uint32_t> pr = gb::odd_primes_up_to(9781);
    struct Acc { std::map<uint64_t, uint64_t> hist, first; uint64_t primes30[30] = {}, twins30[30] = {}; };
    std::vector<Acc> acc(nt);
    const uint64_t evens = (limit - start) / 2 + 1;
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            Acc& a = acc[t];
            for (uint64_t i = t; i < evens; i += nt) {
                const uint64_t n = start + 2 * i;
                uint64_t p = 0;
                for (uint32_t c : pr) if (gb::is_prime_u64(n - c)) { p = c; break; }
                if (!p) for (uint64_t c = 9783; ; c += 2) if (gb::is_prime_u64(c) && gb::is_prime_u64(n - c)) { p = c; break; }
                a.hist[p]++;
                auto it = a.first.find(p);
                if (it == a.first.end() || n < it->second) a.first[p] = n;
                // odd q = n + 1 in (start, limit): primes, twins (q, q + 2 <= limit - 1)
                const uint64_t q = n + 1;
                if (q < limit && gb::is_prime_u64(q)) {
                    a.primes30[q % 30]++;
                    if (q + 2 < limit && gb::is_prime_u64(q + 2)) a.twins30[q % 30]++;
                }
            }
        });
    for (auto& x : th) x.join();
    Acc r;
    for (auto& a : acc) {
        for (auto& h : a.hist) r.hist[h.first] += h.second;
        for (auto& f : a.first) { auto it = r.first.find(f.first); if (it == r.first.end() || f.second < it->second) r.first[f.first] = f.second; }
        for (int i = 0; i < 30; ++i) { r.primes30[i] += a.primes30[i]; r.twins30[i] += a.twins30[i]; }
    }
    uint64_t np = 0, ntw = 0;
    for (int i = 0; i < 30; ++i) { np += r.primes30[i]; ntw += r.twins30[i]; }
    std::printf("evens=%llu pmax=%llu primes=%llu twins=%llu primes30=", (unsigned long long)evens,
                (unsigned long long)r.hist.rbegin()->first, (unsigned long long)np, (unsigned long long)ntw);
    const int u[8] = {1, 7, 11, 13, 17, 19, 23, 29};
    for (int i = 0; i < 8; ++i) std::printf("%s%llu", i ? "," : "", (unsigned long long)r.primes30[u[i]]);
    std::printf(" twins30=%llu,%llu,%llu", (unsigned long long)r.twins30[11], (unsigned long long)r.twins30[17],
                (unsigned long long)r.twins30[29]);
    std::printf(" first=");
    bool f1 = true;
    for (auto& f : r.first) { std::printf("%s%llu:%llu", f1 ? "" : ";", (unsigned long long)f.first, (unsigned long long)f.second); f1 = false; }
    std::printf(" hist=");
    size_t last = 0;
    for (size_t i = 0; i < pr.size(); ++i) if (r.hist.count(pr[i])) last = i + 1;
    for (size_t i = 0; i < last; ++i) {
        if (i) std::printf(",");
        auto it = r.hist.find(pr[i]);
        if (it != r.hist.end()) std::printf("%llu", (unsigned long long)it->second);
    }
    // prime gaps (q - prev(q) for every prime q in [start + 1, limit - 1]):
    // contiguous chunks, each scanned upward, stitched in order
    struct Gc {
        uint64_t first = 0, last = 0, n = 0;
        unsigned __int128 d[4] = {};
        std::map<uint64_t, uint64_t> h;
        std::map<uint64_t, std::pair<uint64_t, uint64_t>> tail;   // g -> count, first lower
        uint64_t gmax = 0, gmax_lo = 0;
        void add(uint64_t g, uint64_t up)
        {
            ++n;
            const unsigned __int128 g1 = g, g2 = g1 * g1;
            d[0] += g1; d[1] += g2; d[2] += g2 * g1; d[3] += g2 * g2;
            h[g]++;
            if (g >= 700) { auto& t = tail[g]; if (!t.first) t.second = up - g; t.first++; }
            if (g > gmax) { gmax = g; gmax_lo = up - g; }
        }
    };
    const uint64_t qlo = start + 1, qhi = limit - 1;
    std::vector<Gc> gc(nt);
    th.clear();
    for (unsigned t = 0; t < nt && qhi >= qlo; ++t)    // [start + 1, limit - 1] empty when limit == start
        th.emplace_back([&, t] {
            Gc& c = gc[t];
            const uint64_t span = (qhi - qlo) / 2 + 1, a = qlo + 2 * (span * t / nt), e = qlo + 2 * (span * (t + 1) / nt);
            for (uint64_t q = a; q < e; q += 2) {
                if (!gb::is_prime_u64(q)) continue;
                if (c.last) c.add(q - c.last, q); else c.first = q;
                c.last = q;
            }
        });
    for (auto& x : th) x.join();
    Gc G;
    uint64_t glast = 0;
    for (auto& c : gc) {
        if (!c.first) continue;
        uint64_t prev = glast;
        if (!prev) { prev = c.first - 2; while (!gb::is_prime_u64(prev)) prev -= 2; }
        G.add(c.first - prev, c.first);
        G.n += c.n;
        for (int k = 0; k < 4; ++k) G.d[k] += c.d[k];
        for (auto& h : c.h) G.h[h.first] += h.second;
        for (auto& t : c.tail) {
            auto& T = G.tail[t.first];
            if (!T.first || t.second.second < T.second) T.second = t.second.second;
            T.first += t.second.first;
        }
        if (c.gmax > G.gmax || (c.gmax == G.gmax && c.gmax_lo < G.gmax_lo)) { G.gmax = c.gmax; G.gmax_lo = c.gmax_lo; }
        glast = c.last;
    }
    const auto u128s = [](unsigned __int128 v) { std::string o; do { o.insert(o.begin(), char('0' + int(v % 10))); v /= 10; } while (v); return o; };
    std::printf(" gaps=%llu gmom=%s,%s,%s,%s gmax=%llu:%llu gtail=", (unsigned long long)G.n, u128s(G.d[0]).c_str(),
                u128s(G.d[1]).c_str(), u128s(G.d[2]).c_str(), u128s(G.d[3]).c_str(), (unsigned long long)G.gmax,
                (unsigned long long)G.gmax_lo);
    bool t1 = true;
    for (auto& t : G.tail) {
        std::printf("%s%llu:%llu:%llu", t1 ? "" : ";", (unsigned long long)t.first, (unsigned long long)t.second.first,
                    (unsigned long long)t.second.second);
        t1 = false;
    }
    if (t1) std::printf("-");
    std::printf(" ghist=");
    const uint64_t hb = G.h.empty() ? 0 : G.h.rbegin()->first / 2 + 1;
    for (uint64_t i = 0; i < hb; ++i) {
        if (i) std::printf(",");
        auto it = G.h.find(2 * i);
        if (it != G.h.end()) std::printf("%llu", (unsigned long long)it->second);
    }
    std::printf("\n");
    return 0;
}
