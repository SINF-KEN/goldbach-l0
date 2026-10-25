// host_tests — GPU-free tests of the host math the kernels trust.
//
// Every table and plan the verifier uploads is checked here against an
// independent brute-force reference: prime lists and the disk cache,
// Miller-Rabin, all small-prime mask table layouts, the
// modular helpers of the sparse first-hit tables, and the three checker
// plans (coverage: phase A + phase B test every candidate; the sparse plan
// validator accepts real plans and rejects corrupted ones). Runs in CI
// (plain, and under ASan + UBSan) — no Level Zero device needed.
//
// usage: host_tests            (exit 0 = all passed; failures are listed)

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>

#include "check30.hpp"
#include "gstats.hpp"
#include "config.hpp"
#include "primes.hpp"
#include "ridge.hpp"
#include "sparse.hpp"
#include "split_check.hpp"

using namespace gb;

namespace {

int g_checks = 0, g_fail = 0;

void check(bool ok, const std::string& what)
{
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 40) std::printf("  FAIL: %s\n", what.c_str());
    }
}

// Flat Eratosthenes up to n (inclusive), independent of primes.cpp.
std::vector<char> sieve_upto(uint32_t n)
{
    std::vector<char> c(size_t(n) + 1, 1);
    c[0] = 0;
    if (n >= 1) c[1] = 0;
    for (uint64_t i = 2; i * i <= n; ++i)
        if (c[i])
            for (uint64_t k = i * i; k <= n; k += i) c[k] = 0;
    return c;
}

void test_prime_lists()
{
    std::puts("prime lists");
    const auto isp = sieve_upto(2'000'003);
    for (uint32_t n : {0u, 1u, 2u, 3u, 4u, 5u, 9u, 97u, 100u, 65535u, 1'000'000u, 2'000'003u}) {
        std::vector<uint32_t> want;
        for (uint32_t x = 3; x <= n; x += 2) if (isp[x]) want.push_back(x);
        check(odd_primes_up_to(n) == want, "odd_primes_up_to(" + std::to_string(n) + ")");
        check(segmented_odd_primes_up_to(n) == want, "segmented_odd_primes_up_to(" + std::to_string(n) + ")");
    }
    check(odd_primes_up_to(1'000'000).size() == 78497, "pi(1e6) - 1 == 78497");
    // segmented sieve across several 2^20-odd blocks
    const uint64_t big = 5'000'011;
    const auto flat = odd_primes_up_to(uint32_t(big));
    check(segmented_odd_primes_up_to(big) == flat, "segmented == flat at 5e6 (multi-block)");
}

void test_prime_cache()
{
    std::puts("prime cache");
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("gb_host_tests_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const std::string base = (dir / "bp.cache").string();
    {
        const auto a = PrimeCacheView::load(300000, base.c_str());          // miss: generate + write
        const auto want = odd_primes_up_to(300000);
        check(a.size() == want.size() && std::equal(want.begin(), want.end(), a.data()),
              "PrimeCacheView miss == odd_primes_up_to");
        check(fs::exists(base + ".300000"), "keyed cache file written");
        const auto b = PrimeCacheView::load(300000, base.c_str());          // hit: mmap
        check(b.size() == want.size() && std::equal(want.begin(), want.end(), b.data()),
              "PrimeCacheView hit == odd_primes_up_to");
        const auto c = PrimeCacheView::load(123457, base.c_str());          // prefix of the larger cache
        const auto wc = odd_primes_up_to(123457);
        check(c.size() == wc.size() && std::equal(wc.begin(), wc.end(), c.data()),
              "PrimeCacheView prefix of a larger cache");
        check(!fs::exists(base + ".123457"), "prefix load does not write a new cache");
    }
    {   // a corrupted cache must be rejected, not trusted
        const std::string f = base + ".777";
        std::FILE* o = std::fopen(f.c_str(), "wb");
        const char junk[64] = "GBPR not really a cache";
        std::fwrite(junk, 1, sizeof junk, o);
        std::fclose(o);
        const auto v = segmented_odd_primes_up_to_cached(777, f.c_str());
        check(v == odd_primes_up_to(777), "corrupted cache regenerated");
    }
    {   // Header and size intact, payload damaged: the structural checks pass,
        // so only the content check can refuse it. (A concurrent builder's
        // truncation could expose exactly such a zero hole before v2.3.0.)
        const uint64_t n = 50000;
        const auto good = odd_primes_up_to(n);
        const auto write = [&](const std::string& f, const std::vector<uint32_t>& pl) {
            std::FILE* o = std::fopen(f.c_str(), "wb");
            const uint64_t cnt = pl.size();
            std::fwrite("GBPR", 1, 4, o);
            std::fwrite(&n, 8, 1, o);
            std::fwrite(&cnt, 8, 1, o);
            std::fwrite(pl.data(), 4, pl.size(), o);
            std::fclose(o);
        };
        std::vector<std::pair<const char*, std::vector<uint32_t>>> bad;
        auto hole = good;  std::fill(hole.begin() + 1000, hole.begin() + 1100, 0u);
        auto even = good;  even[2000] += 1;
        auto swap = good;  std::swap(swap.at(3000), swap.at(3001));
        bad.push_back({"zero hole", hole});
        bad.push_back({"even entry", even});
        bad.push_back({"out of order", swap});
        for (auto& [why, pl] : bad) {
            const std::string kb = (dir / ("cv_" + std::string(why).substr(0, 4))).string();
            write(kb + "." + std::to_string(n), pl);                      // keyed file: mmap path
            const auto view = PrimeCacheView::load(n, kb.c_str());
            check(view.size() == good.size() && std::equal(good.begin(), good.end(), view.data()),
                  std::string("PrimeCacheView refuses a cache with a ") + why);
            const std::string fp = kb + ".direct";
            write(fp, pl);                                                // fread path
            check(segmented_odd_primes_up_to_cached(n, fp.c_str()) == good,
                  std::string("cached loader refuses a cache with a ") + why);
        }
        bool tmp_left = false;
        for (const auto& de : fs::directory_iterator(dir))
            if (de.path().filename().string().find(".tmp.") != std::string::npos) tmp_left = true;
        check(!tmp_left, "cache writes leave no temp files behind");
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
}

void test_primality()
{
    std::puts("primality");
    const auto isp = sieve_upto(200000);
    bool ok_mr = true, ok_td = true;
    for (uint32_t n = 0; n <= 200000; ++n) {
        ok_mr &= is_prime_u64(n) == bool(isp[n]);
        ok_td &= is_prime_trial(n) == bool(isp[n]);
    }
    check(ok_mr, "is_prime_u64 == sieve on [0, 2e5]");
    check(ok_td, "is_prime_trial == sieve on [0, 2e5]");
    struct { uint64_t n; bool prime; const char* why; } cases[] = {
        {2305843009213693951ull, true, "2^61 - 1"},
        {18446744073709551557ull, true, "largest 64-bit prime 2^64 - 59"},
        {18446744073709551615ull, false, "2^64 - 1"},
        {3215031751ull, false, "strong pseudoprime to bases 2, 3, 5, 7"},
        {3825123056546413051ull, false, "strong pseudoprime to bases 2..23"},
        {1000000000000000003ull, true, "1e18 + 3 (smallest prime > 1e18)"},
        {4294967291ull * 4294967279ull, false, "product of the two largest 32-bit primes"},
        {561ull, false, "Carmichael 561"},
        {9007199254740881ull, true, "largest prime < 2^53"},
    };
    for (const auto& c : cases)
        check(is_prime_u64(c.n) == c.prime, std::string("is_prime_u64: ") + c.why);
    // random 40-bit values against trial division (independent reference)
    std::mt19937_64 rng(4242);
    bool ok = true;
    for (int i = 0; i < 2000; ++i) {
        const uint64_t n = rng() >> 24;
        ok &= is_prime_u64(n) == is_prime_trial(n);
    }
    check(ok, "is_prime_u64 == trial division on random 40-bit values");
}

// Independent of primes.cpp: a^e mod m and the strong probable-prime test.
uint64_t ref_powmod(uint64_t a, uint64_t e, uint64_t m)
{
    unsigned __int128 r = 1, b = a % m;
    for (; e; e >>= 1, b = b * b % m)
        if (e & 1) r = r * b % m;
    return uint64_t(r);
}

bool ref_strong(uint64_t n, uint64_t a)
{
    uint64_t d = n - 1;
    int s = 0;
    while (!(d & 1)) { d >>= 1; ++s; }
    uint64_t x = ref_powmod(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int i = 1; i < s; ++i) {
        x = uint64_t((unsigned __int128)x * x % n);
        if (x == n - 1) return true;
    }
    return false;
}

// The CPU fallback's deterministic Miller-Rabin against the composites
// that defeat weaker tests: every base-2 Fermat pseudoprime below 2^32
// (tests/data, 10403 numbers, the list complete as checked by brute force
// below 2^22 here and by count), and psi_k for k = 1..11, the least composite
// passing the strong test to the first k prime bases (Jaeschke; Sorenson and
// Webster) — each passes every base but the last ones of the twelve.
// GB_PSP_FILE=<Feitsma's list of base-2 pseudoprimes below 2^64, one per
// line, first field> additionally streams the whole list (~1.2e8 numbers;
// not in CI).
void test_pseudoprimes()
{
    std::puts("pseudoprimes");
    const std::string path = std::string(GB_TEST_DATA) + "/psp2_below_2_32.txt";
    std::FILE* f = std::fopen(path.c_str(), "r");
    check(f != nullptr, "open " + path);
    if (!f) return;
    std::vector<uint64_t> psp;
    char line[128];
    while (std::fgets(line, sizeof line, f))
        if (line[0] != '#' && line[0] != '\n') psp.push_back(std::strtoull(line, nullptr, 10));
    std::fclose(f);
    check(psp.size() == 10403, "10403 base-2 pseudoprimes below 2^32 (got " + std::to_string(psp.size()) + ")");
    check(std::is_sorted(psp.begin(), psp.end()) && std::adjacent_find(psp.begin(), psp.end()) == psp.end(),
          "pseudoprime list strictly ascending");
    size_t strong = 0;
    bool valid = true, rejected = true;
    for (uint64_t n : psp) {
        valid &= n > 2 && n < (uint64_t(1) << 32) && (n & 1) && !is_prime_trial(n) && ref_powmod(2, n - 1, n) == 1;
        strong += ref_strong(n, 2);
        rejected &= !is_prime_u64(n);
    }
    check(valid, "every entry is an odd composite n < 2^32 with 2^(n-1) = 1 mod n");
    check(strong == 2314, "2314 of them are strong pseudoprimes to base 2 (got " + std::to_string(strong) + ")");
    check(rejected, "is_prime_u64 rejects every base-2 pseudoprime below 2^32");
    // completeness of the file's head: brute force below 2^22
    const uint32_t lim = 1u << 22;
    const auto isp = sieve_upto(lim);
    std::vector<uint64_t> brute;
    for (uint32_t n = 3; n < lim; n += 2)
        if (!isp[n] && ref_powmod(2, n - 1, n) == 1) brute.push_back(n);
    const std::vector<uint64_t> head(psp.begin(), std::lower_bound(psp.begin(), psp.end(), uint64_t(lim)));
    check(brute == head, "list below 2^22 == brute force (" + std::to_string(brute.size()) + " numbers)");

    const uint64_t bases[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    const uint64_t psi[11] = {2047ull, 1373653ull, 25326001ull, 3215031751ull, 2152302898747ull,
                              3474749660383ull, 341550071728321ull, 341550071728321ull,
                              3825123056546413051ull, 3825123056546413051ull, 3825123056546413051ull};
    for (int k = 1; k <= 11; ++k) {
        const uint64_t n = psi[k - 1];
        bool passes = true;
        for (int i = 0; i < k; ++i) passes &= ref_strong(n, bases[i]);
        // psi_k fails base k + 1 unless it is also psi_(k+1) (psi_12 > 2^64)
        const bool next_fails = k == 11 || psi[k] != n;
        check(passes && ref_strong(n, bases[k]) != next_fails,
              "psi_" + std::to_string(k) + " passes exactly the first " + std::to_string(k) + "+ bases");
        check(!is_prime_u64(n), "is_prime_u64 rejects psi_" + std::to_string(k) + " = " + std::to_string(n));
    }

    if (const char* full = std::getenv("GB_PSP_FILE")) {
        std::FILE* g = std::fopen(full, "r");
        check(g != nullptr, std::string("open GB_PSP_FILE ") + full);
        if (!g) return;
        uint64_t count = 0, accepted = 0, first_bad = 0;
        while (std::fgets(line, sizeof line, g)) {
            if (line[0] < '0' || line[0] > '9') continue;
            const uint64_t n = std::strtoull(line, nullptr, 10);
            ++count;
            if (is_prime_u64(n) && !accepted++) first_bad = n;
        }
        std::fclose(g);
        std::printf("  GB_PSP_FILE: %" PRIu64 " pseudoprimes read\n", count);
        check(count > 0 && accepted == 0, "is_prime_u64 rejects every number of GB_PSP_FILE (accepted " +
              std::to_string(accepted) + (accepted ? ", first " + std::to_string(first_bad) : "") + ")");
    }
}

// Dense-map masks: bit g <-> q = q_first + 2g; cleared iff p | q.
void test_small_prime_masks()
{
    std::puts("small-prime masks");
    const auto sp = odd_primes_up_to(MASK_MAX_P - 1);
    const uint32_t mc = uint32_t(sp.size());
    std::mt19937_64 rng(12345);
    for (int it = 0; it < 6; ++it) {
        const uint64_t qf = (rng() % 4000000000000000000ull) | 1ull;   // odd
        const auto tab = small_prime_mask_table(sp.data(), mc, qf);
        check(tab.size() == size_t(mc) * 64, "mask table size");
        bool ok = true;
        for (uint32_t j = 0; j < mc && ok; ++j) {
            const uint64_t p = sp[j];
            for (uint64_t w = 0; w < 3 * p && ok; ++w) {
                const uint64_t m = tab[j * 64 + w % p];
                for (uint32_t b = 0; b < 64; ++b) {
                    const uint64_t q_mod = (qf % p + 2 * ((64 * w + b) % p)) % p;
                    ok &= (((m >> b) & 1) == 0) == (q_mod == 0);
                }
            }
        }
        check(ok, "small_prime_mask_table clears exactly the multiples (q_first " + std::to_string(qf) + ")");
        // grouped form: word x of group g == AND of its primes' masks at x mod p
        const auto grp = small_prime_group_masks(sp.data(), mc, qf);
        bool okg = grp.count() > 0;
        uint32_t covered = 0;
        for (uint32_t g = 0; g < grp.count() && okg; ++g) {
            const uint32_t off = grp.desc[2 * g], per = grp.desc[2 * g + 1];
            okg &= per <= 4096;
            for (uint64_t x = 0; x < 2ull * per && okg; ++x) {
                uint64_t want = ~0ull;
                uint64_t prod = 1;
                for (uint32_t j = covered; j < mc && prod * sp[j] <= per && per % sp[j] == 0; ++j) {
                    want &= tab[j * 64 + x % sp[j]];
                    prod *= sp[j];
                }
                okg &= grp.words[off + x % per] == want;
            }
            uint64_t prod = 1;
            while (covered < mc && per % sp[covered] == 0 && prod * sp[covered] <= per) prod *= sp[covered++];
        }
        check(okg && covered == mc, "small_prime_group_masks == AND of per-prime masks, all primes grouped");
    }
    // class-map masks (step 6 / 30) + canonical tables with word offsets
    for (uint32_t step : {6u, 30u}) {
        const uint32_t skip = step == 6 ? 3u : 15u;
        for (int it = 0; it < 4; ++it) {
            // class arrays hold q in a unit class mod step (precondition of
            // the step table: primes dividing step have no multiples at all)
            uint64_t base = (rng() % 4000000000000000000ull) | 1ull;
            while (std::gcd(base, uint64_t(step)) != 1) base += 2;
            const auto tab = small_prime_mask_table_step(sp.data(), mc, base, step);
            bool ok = true;
            for (uint32_t j = 0; j < mc && ok; ++j) {
                const uint64_t p = sp[j];
                for (uint64_t w = 0; w < 2 * p && ok; ++w)
                    for (uint32_t b = 0; b < 64; ++b) {
                        const bool mult = (base % p + (step % p) * ((64 * w + b) % p)) % p == 0;
                        ok &= (((tab[j * 64 + w % p] >> b) & 1) == 0) == mult;
                    }
            }
            check(ok, "small_prime_mask_table_step exact (step " + std::to_string(step) + ")");
            const auto canon = small_prime_group_masks_canonical(sp.data(), mc, skip);
            const auto ys = small_prime_group_offsets(sp.data(), mc, base, step);
            bool okc = ys.size() == canon.count();
            for (uint32_t g = 0; g < canon.count() && okc; ++g) {
                const uint32_t off = canon.desc[2 * g], per = canon.desc[2 * g + 1];
                for (uint64_t x = 0; x < 300 && okc; ++x) {
                    uint64_t want = ~0ull;
                    for (uint32_t j = 0; j < mc; ++j)
                        if (per % sp[j] == 0 && skip % sp[j] != 0) want &= tab[j * 64 + x % sp[j]];
                    okc &= canon.words[off + (x + per - ys[g]) % per] == want;
                }
            }
            check(okc, "canonical group table at (x - Y) mod P == progression masks (step " +
                       std::to_string(step) + ")");
        }
    }
}

void test_sparse_helpers()
{
    std::puts("sparse helpers");
    for (uint32_t M : {2310u, 30030u}) {
        const ScmGeo g = scm_geo(M);                        // throws unless covering
        check(g.K == g.res.size() && std::is_sorted(g.res.begin(), g.res.end()), "residues sorted");
        bool ok = true;
        for (uint32_t c : g.res) ok &= g.unit(c);
        check(ok, "every class residue is a unit mod " + std::to_string(M));
    }
    bool threw = false;
    try { scm_geo(210); } catch (const std::exception&) { threw = true; }
    check(threw, "scm_geo rejects an unsupported modulus");
    std::mt19937_64 rng(99);
    const auto ps = odd_primes_up_to(200000);
    bool ok = true;
    for (int i = 0; i < 3000 && ok; ++i) {
        const uint32_t p = ps[5 + rng() % (ps.size() - 5)];
        const uint32_t M = 30030;
        if (M % p == 0) continue;
        const uint64_t Q = (rng() % 4000000000000000000ull) / M * M;
        const uint32_t c = uint32_t(rng() % M);
        const uint64_t inv = scm_invmod(M % p, p);
        ok &= scm_mulmod(M % p, inv, p) == 1 % p;
        const uint32_t t = scm_first_hit(Q, c, p, inv);
        ok &= t < p && (Q + c + uint64_t(M) * t) % p == 0;
    }
    check(ok, "scm_invmod / scm_first_hit give the first multiple in the class");
    threw = false;
    try { scm_invmod(6, 9); } catch (const std::exception&) { threw = true; }
    check(threw, "scm_invmod rejects a non-unit");
    // group offsets: word x of a class array = canonical word (x - Y) mod P
    const uint32_t M = 30030;
    const GroupsSCM gr = make_groups_scm(M);
    const uint64_t Q = 4000000000000000000ull / M * M;
    const uint32_t c = scm_geo(M).res[3];
    std::vector<uint32_t> ys(gr.primes.size());
    scm_group_offsets(gr, [&](uint32_t p) { return scm_first_hit(Q, c, p, scm_invmod(M % p, p)); }, ys.data());
    ok = true;
    for (size_t gi = 0; gi < gr.primes.size() && ok; ++gi) {
        const uint32_t off = gr.desc[2 * gi], per = gr.desc[2 * gi + 1];
        for (uint64_t x = 0; x < 200 && ok; ++x)
            for (uint32_t b = 0; b < 64; ++b) {
                const uint64_t t = 64 * x + b;
                bool mult = false;
                for (uint32_t p : gr.primes[gi]) mult |= (Q % p + c + (uint64_t(M) % p) * (t % p)) % p == 0;
                ok &= (((gr.words[off + (x + per - ys[gi]) % per] >> b) & 1) == 0) == mult;
            }
    }
    check(ok, "make_groups_scm + scm_group_offsets clear exactly the class multiples");
}

// Phase A (funnel members) + phase B must test every candidate of a class.
void test_dense_plans()
{
    std::puts("checker plans (split, mod-30)");
    const auto cands = odd_primes_up_to(MAX_GB_PRIME);
    const auto isp = sieve_upto(MAX_GB_PRIME);
    // class-split plan
    const SplitPlan sp = make_split_plan(cands, CHECK_SPLIT_CANDS, parse_split_stages(CHECK_SPLIT_STAGES));
    bool ok = true;
    for (uint32_t c = 0; c < 3; ++c) {
        std::vector<char> tested(MAX_GB_PRIME + 1, 0);
        const uint32_t mo = (c == 1) ? 16 : 0;
        const uint32_t res = (c == 1) ? 5 : 1;
        for (uint32_t st = 0; st < sp.meta[mo]; ++st) {
            const uint32_t b = sp.meta[mo + 1 + 3 * st], e = sp.meta[mo + 2 + 3 * st], mask = sp.meta[mo + 3 + 3 * st];
            for (uint32_t j = b; j < e; ++j)
                for (uint32_t d = 0; d <= SPLIT_MAXD; ++d)
                    if ((mask >> d) & 1) {
                        const uint32_t x = sp.cands[j] + 6 * d;
                        ok &= x <= MAX_GB_PRIME && isp[x] && x % 6 == res;
                        tested[std::min<uint32_t>(x, MAX_GB_PRIME)] = 1;
                    }
        }
        for (uint32_t k = 0; k < sp.rcnt[c]; ++k) tested[sp.rest[sp.roff[c] + k]] = 1;
        for (uint32_t p : cands) ok &= tested[p] != 0;
    }
    check(ok, "split plan: funnels hold only primes of the list residue; A + B cover every candidate");
    // mod-30 plan
    const Plan30 p30 = make_plan30(cands, CHECK30_PLAN);
    ok = true;
    for (uint32_t cr = 0; cr < 15; ++cr) {
        std::vector<char> tested(MAX_GB_PRIME + 1, 0);
        for (uint32_t sg = 0; sg < p30.meta[cr * 32]; ++sg) {
            const uint32_t be = p30.meta[cr * 32 + 1 + 2 * sg], sm = p30.meta[cr * 32 + 2 + 2 * sg];
            const uint32_t s = sm & 0xFF, mask = sm >> 8;
            ok &= c30_unit(2 * cr + 30 - s);
            for (uint32_t j = be & 0xFFFF; j < (be >> 16); ++j)
                for (uint32_t d = 0; d <= C30_MAXD; ++d)
                    if ((mask >> d) & 1) {
                        const uint32_t x = p30.cands[j] + 30 * d;
                        ok &= x <= MAX_GB_PRIME && isp[x] && x % 30 == s;
                        tested[std::min<uint32_t>(x, MAX_GB_PRIME)] = 1;
                    }
        }
        for (uint32_t k = 0; k < p30.rtab[15 + cr]; ++k) tested[p30.rest[p30.rtab[cr] + k]] = 1;
        for (uint32_t p : cands) ok &= tested[p] != 0;
    }
    check(ok, "mod-30 plan: funnels hold primes of their class; A + B cover every candidate");
    // tail bound: every phase-A member < member_lt <= CHECK30_TAIL, and each
    // class's phase-B list is ascending over the useful candidates, then the
    // futile ones (so check30_b's first hit is p(n) for every pending even)
    ok = p30.member_lt <= CHECK30_TAIL && p30.max_member < p30.member_lt;
    for (uint32_t cr = 0; cr < 15; ++cr) {
        bool futile = false;
        uint32_t prev = 0;
        for (uint32_t k = 0; k < p30.rtab[15 + cr]; ++k) {
            const uint32_t p = p30.rest[p30.rtab[cr] + k];
            const bool u = c30_unit(2 * cr + 30 - p % 30);
            if (!u && !futile) { futile = true; prev = 0; }
            ok &= (u != futile) && p > prev;
            prev = p;
        }
    }
    check(ok, "mod-30 plan: members below the tail bound, phase-B lists ascending (useful, then futile)");
    {
        const Plan30 nb = make_plan30(cands, "1000:" + std::string(CHECK30_PLAN).substr(5));
        check(nb.max_member < 1000 && nb.member_lt <= 1000, "make_plan30 honours a T: member bound");
    }
    bool threw = false;
    try { make_plan30(cands, "5/1x4"); } catch (const std::exception&) { threw = true; }  // s = 5 is not a unit
    check(threw, "make_plan30 rejects a non-unit candidate class");
}

void test_stats_plan()
{
    std::puts("statistics plan (check30_xs tables, count30 limits)");
    const auto cands = odd_primes_up_to(MAX_GB_PRIME);
    static const uint32_t units[8] = {1, 7, 11, 13, 17, 19, 23, 29};
    for (uint32_t K : {1u, 33u, STATS_K_DEFAULT, STATS_KMAX}) {
        for (uint32_t plt : {uint32_t(CHECK30_TAIL), 600u}) {
            const StatsPlan sp = make_stats_plan(cands, K, plt);
            bool ok = sp.xtab.size() == 45 && sp.xall.size() == 15 * cands.size();
            uint64_t vmask = 0;
            for (uint32_t v = 0; v < 4; ++v)
                for (uint32_t i = 0; i < sp.vn[v]; ++i) vmask |= 1ull << ((sp.vcls[v] >> (4 * i)) & 15);
            ok &= vmask == 0x7FFF;
            for (uint32_t cr = 0; cr < 15 && ok; ++cr) {
                const uint32_t r = 2 * cr, o = sp.xtab[cr], kx = sp.xtab[15 + cr], len = sp.xtab[30 + cr];
                ok &= len == cands.size() && kx <= K;
                // useful ascending, then futile ascending; every candidate once
                std::vector<char> seen(MAX_GB_PRIME + 1, 0);
                bool fut = false;
                uint32_t prev = 0;
                for (uint32_t j = 0; j < len; ++j) {
                    const uint32_t p = sp.xall[o + j];
                    const bool u = c30_unit(r + 30 - p % 30);
                    if (!u && !fut) { fut = true; prev = 0; }
                    ok &= u != fut && p > prev && !seen[p] && cands[sp.pidx[o + j]] == p;
                    seen[p] = 1;
                    prev = p;
                    if (j < kx) ok &= u && p < plt;
                }
                const uint32_t* t = &sp.xch[cr * 64];
                const uint32_t nch = t[0] & 0xFF, nd = (t[0] >> 8) & 0xFF, na = t[0] >> 16;
                ok &= na == 3 || na == 4 || na == 6 || na == 8;
                for (uint32_t j = 0; j < nd; ++j) ok &= sp.xall[o + j] <= 5;
                uint32_t next = nd;
                for (uint32_t c = 0; c < nch; ++c) {
                    const uint32_t* h = t + 2 + 9 * c;
                    ok &= (h[0] & 0xFFFF) == next;
                    for (uint32_t j = h[0] & 0xFFFF; j < (h[0] >> 16); ++j) {
                        const uint32_t p = sp.xall[o + j], a = sp.xj[o + j] & 7, d = sp.xj[o + j] >> 8;
                        const uint32_t k = (t[1] >> (4 * a)) & 15;
                        ok &= a < na && d < 32 && h[1 + a] == p + 30 * d && k < 8 &&
                              units[k] == (r + 30 - p % 30) % 30;
                    }
                    next = h[0] >> 16;
                }
                ok &= next == std::max(kx, nd) || (kx <= nd && nch == 0);
            }
            check(ok, "stats plan K=" + std::to_string(K) + " p<" + std::to_string(plt) +
                      ": lists, prefix, slots, offsets and chunks consistent");
        }
    }
    // count30_limits vs brute force: class c bit t <-> q = Q[c] + 30 t; the
    // limits must select exactly the odd q in [S-3, E-3] coprime to 30, and
    // twin pairs (q, q+2) with q in [S-5, E-5].
    bool ok = true;
    for (uint64_t S : {4000000000000000000ull, 4000000000000000062ull, 1000000000000ull + 8}) {
        const uint64_t E = S + 2 * 1000;
        const uint64_t qf = S - MAX_GB_PRIME + (((S - MAX_GB_PRIME) & 1) ? 0 : 1);
        uint64_t Q[8];
        for (uint32_t c = 0; c < 8; ++c) Q[c] = qf + (units[c] + 30 - qf % 30) % 30;
        const auto lim = count30_limits(S, E, Q);
        for (uint32_t c = 0; c < 8; ++c) {
            uint64_t want = 0;
            for (uint64_t q = S - 3; q <= E - 3; q += 2) want += q % 30 == units[c];
            const uint64_t lo = Q[c] + 30 * lim[2 * c], hi = Q[c] + 30 * uint64_t(lim[2 * c + 1]);
            ok &= lim[2 * c + 1] - lim[2 * c] == want && lo >= S - 3 && (want == 0 || hi - 30 <= E - 3);
        }
        for (uint32_t j = 0; j < 3; ++j) {
            const uint32_t k = lim[16 + 5 * j], k2 = lim[17 + 5 * j], d = lim[18 + 5 * j];
            uint64_t want = 0;
            for (uint64_t q = S - 5; q <= E - 5; q += 2) want += q % 30 == units[k];
            ok &= lim[20 + 5 * j] - lim[19 + 5 * j] == want;
            // partner of bit t is bit t + d of class k2: q + 2
            const uint64_t t = lim[19 + 5 * j];
            ok &= Q[k] + 30 * t + 2 == Q[k2] + 30 * (t + d) && (units[k] + 2) % 30 == units[k2];
        }
    }
    check(ok, "count30_limits: exact q ranges and twin partners (brute force)");

    // p(n) re-proof used before any record is written: Oliveira e Silva's
    // record S(9781) and a known ridge at 4e18
    check(pn_exact(3325581707333960528ull) == 9781, "pn_exact(S(9781)) = 9781 (t0 record)");
    check(pn_exact(4000000000169717704ull) == 4133, "pn_exact(4000000000169717704) = 4133");
    check(pn_exact(4000000000000000000ull) == 113, "pn_exact(4e18) = 113");
    bool sorted = true;
    for (size_t i = 1; i < sizeof S_UNKNOWN_4E18 / sizeof S_UNKNOWN_4E18[0]; ++i)
        sorted &= S_UNKNOWN_4E18[i - 1] < S_UNKNOWN_4E18[i];
    check(sorted && sizeof S_UNKNOWN_4E18 / sizeof S_UNKNOWN_4E18[0] == 105, "S_UNKNOWN_4E18: 105 sorted primes");
    // records file: a claimed first occurrence of unknown-S(p) prime 8423 at an
    // n whose true p(n) is 9781 must be written and flagged MISMATCH
    {
        char tmpl[] = "/tmp/gb_host_tests_XXXXXX";
        const std::string dir = ::mkdtemp(tmpl) ? tmpl : "/tmp";
        setenv("GB_RECORDS_FILE", (dir + "/rec.txt").c_str(), 1);
        if (std::FILE* hs = std::fopen((dir + "/hook.sh").c_str(), "wb")) {
            std::fprintf(hs, "echo \"$@\" > %s/hook.txt\n", dir.c_str());
            std::fclose(hs);
        }
        setenv("GB_RECORD_HOOK", ("sh " + dir + "/hook.sh").c_str(), 1);
        setenv("GB_STATS_STATE", (dir + "/st").c_str(), 1);
        setenv("GB_STATS_LOG", (dir + "/log").c_str(), 1);
        const StatsPlan sp = make_stats_plan(cands, STATS_K_DEFAULT, CHECK30_TAIL);
        GoldbachStats gs(cands, sp, CHECK30_TAIL);
        std::vector<uint32_t> fp(MAX_GB_PRIME / 2 + 1, ~0u);
        const uint64_t n = 3325581707333960528ull, seg = n - 2 * 1000;
        fp[8423 / 2] = 1000;                       // e = 1000 -> n
        gs.add_first(seg, fp.data());
        std::string txt;
        if (std::FILE* f = std::fopen((dir + "/rec.txt").c_str(), "rb")) {
            char buf[1024];
            size_t k = std::fread(buf, 1, sizeof buf - 1, f);
            buf[k] = 0; txt = buf; std::fclose(f);
        }
        for (int w = 0; w < 50; ++w) {                 // the hook runs in the background
            std::FILE* h = std::fopen((dir + "/hook.txt").c_str(), "rb");
            if (h) { char hb[256] = {}; size_t k = std::fread(hb, 1, sizeof hb - 1, h); std::fclose(h);
                     if (k && std::string(hb).find("SP 8423 3325581707333960528") != std::string::npos) {
                         check(std::string(hb).find("MISMATCH") != std::string::npos, "record hook called with KIND P N Q CHECK");
                         break; } }
            if (w == 49) check(false, "record hook called with KIND P N Q CHECK");
            usleep(100000);
        }
        check(txt.find("p(n)=8423 n=3325581707333960528") != std::string::npos &&
              txt.find("cpu_check=MISMATCH") != std::string::npos, "records file: unknown-S(p) event written, false claim flagged");
        if (dir != "/tmp") std::filesystem::remove_all(dir);
        unsetenv("GB_RECORDS_FILE"); unsetenv("GB_STATS_STATE"); unsetenv("GB_STATS_LOG");
        unsetenv("GB_RECORD_HOOK");
    }
}

void test_sparse_plan()
{
    std::puts("sparse checker plan (M = 2310, production pmax)");
    const ScmGeo g = scm_geo(2310);
    const uint32_t pmax = g.pmax, TB = pmax / g.M + SCM_MAXD + 2;
    const auto cands = odd_primes_up_to(pmax + (SCM_MAXD + 2) * g.M);
    const auto isp = sieve_upto(pmax);
    auto is_p = [&](uint32_t x) { return x <= pmax && isp[x] != 0; };
    PlanSCM plan = make_plan_scm(g, cands, pmax, TB, 64.0, 3.0, 1.0, 384, g.span);
    bool threw = false;
    try { validate_plan_scm(g, plan, TB, pmax, is_p); } catch (const std::exception& e) {
        threw = true; std::printf("    %s\n", e.what());
    }
    check(!threw, "validate_plan_scm accepts a freshly built plan");
    check(plan.mean_tests >= 60.0, "plan reaches ~64 phase-A tests per class");
    // corrupt one phase-B probe to a composite candidate -> must be rejected
    PlanSCM bad = plan;
    bool corrupted = false;
    for (uint32_t cr = 0; cr < g.CR() && !corrupted; ++cr) {
        const uint32_t r = 2 * cr, n = bad.rtab[g.CR() + cr];
        for (uint32_t j = 0; j < n && !corrupted; ++j) {
            uint32_t& v = bad.rest[bad.rtab[cr] + j];
            const uint32_t k = v >> 28, U = v & 0x0FFFFFFFu;
            for (uint32_t U2 = U; U2 > 1 && !corrupted; --U2) {   // walk to a composite p
                const uint32_t w = r < g.res[k] ? 1u : 0u;
                const uint64_t p = (r + g.M - g.res[k]) % g.M + uint64_t(g.M) * (TB - U2 - w);
                if (p >= 3 && p <= pmax && !isp[p]) { v = U2 | (k << 28); corrupted = true; }
            }
        }
    }
    threw = false;
    try { validate_plan_scm(g, bad, TB, pmax, is_p); } catch (const std::exception&) { threw = true; }
    check(corrupted && threw, "validate_plan_scm rejects a plan probing a composite candidate");
    bad = plan;
    check(bad.meta[0] > 0, "class 0 has phase-A segments");
    bad.meta[3] = (bad.meta[3] & ~0xFFu) | 0xFEu;             // segment 0 of class 0 -> class k = 254
    threw = false;
    try { validate_plan_scm(g, bad, TB, pmax, is_p); } catch (const std::exception&) { threw = true; }
    check(threw, "validate_plan_scm rejects a funnel on a nonexistent class");
}

void test_ridge_store()
{
    // found_unix column: v1 rows load as unknown (0), a re-found even keeps
    // its stored time, a fresh entry is stamped now, save/load round-trips
    const auto cands = odd_primes_up_to(MAX_GB_PRIME);
    // default floor at or above the production plan's phase-A bound: no
    // floor clamp (and no missed ridges) on a default run
    check(RIDGE_MIN_DEFAULT >= make_plan30(cands, CHECK30_PLAN).member_lt,
          "ridge floor default >= check30 phase-A member bound");
    char tmpl[] = "/tmp/gb_ridge_tests_XXXXXX";
    const std::string dir = ::mkdtemp(tmpl) ? tmpl : "/tmp";
    const std::string path = dir + "/ridges.csv";
    unsetenv("GB_RIDGE");
    unsetenv("GB_RIDGE_MIN");
    const uint64_t n1 = 4000000000169717704ull, n2 = 4004022922169805316ull;
    if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fprintf(f, "# v1\n1,%llu,4133,%llu,4000000000000000000\n",
                     (unsigned long long)n1, (unsigned long long)(n1 - 4133));
        std::fclose(f);
    }
    RidgeTracker t(&cands);
    check(t.load(path.c_str()) && t.top().size() == 1 && t.top()[0].found == 0,
          "ridge store: v1 row loads with found = 0");
    t.consider(n1, 4133, 4000000000000000000ull);     // re-found by a re-run
    check(t.top().size() == 1 && t.top()[0].found == 0, "ridge store: re-found even keeps its time");
    const uint64_t before = uint64_t(std::time(nullptr));
    t.consider(n2, 8087, n2 - 1000);
    check(t.top().size() == 2 && t.top()[0].p == 8087 && t.top()[0].found >= before,
          "ridge store: new entry stamped with the current time");
    t.save(path.c_str());
    RidgeTracker u(&cands);
    check(u.load(path.c_str()) && u.top().size() == 2 && u.top()[0].found == t.top()[0].found &&
          u.top()[1].found == 0, "ridge store: found_unix round-trips through save/load");
    std::filesystem::remove_all(dir);
}

// Base-prime bound and the supported-limit guard at the top of uint64_t.
void test_limit_bounds()
{
    auto isqrt = [](uint64_t n) {
        uint64_t r = uint64_t(std::sqrt(double(n)));
        while (__uint128_t(r) * r > n) --r;
        while (__uint128_t(r + 1) * (r + 1) <= n) ++r;
        return r;
    };
    const uint64_t wrap = uint64_t(4294967280ull) * 4294967280ull;   // (2^32 - 16)^2
    bool ok = true, mono = true;
    uint32_t prev = 0;
    for (uint64_t n : std::initializer_list<uint64_t>{4ull, 100000000ull, 4000000000000000000ull, 5000000000000000000ull,
                       9223372036854775807ull, gb::MAX_SUPPORTED_LIMIT, wrap - 1, wrap,
                       wrap + 1, ~0ull - 1, ~0ull}) {   // ascending: 2^64 - 2^40 < (2^32 - 16)^2
        const uint64_t r = isqrt(n), b = gb::base_prime_bound(n);
        ok &= b >= r && b <= std::min<uint64_t>(r + 17, 0xFFFFFFFFull);
        mono &= b >= prev;
        prev = uint32_t(b);
    }
    check(ok, "base_prime_bound: isqrt(n) <= bound <= min(isqrt(n) + 17, 2^32 - 1), incl. (2^32 - 16)^2 and 2^64 - 1");
    check(mono, "base_prime_bound: monotone up to 2^64 - 1 (no wrap)");
    check(gb::base_prime_bound(~0ull) == 0xFFFFFFFFu, "base_prime_bound(2^64 - 1) saturates at 2^32 - 1");
    auto throws = [](uint64_t n) {
        try { gb::require_supported_limit(n); } catch (const std::runtime_error&) { return true; }
        return false;
    };
    check(!throws(5000000000000000000ull) && !throws(gb::MAX_SUPPORTED_LIMIT),
          "require_supported_limit accepts 5e18 and 2^64 - 2^40");
    check(throws(gb::MAX_SUPPORTED_LIMIT + 2) && throws(~0ull - 1),
          "require_supported_limit rejects above 2^64 - 2^40 (fail closed)");
}

}  // namespace


// Campaign state file (gstats.hpp): every run that PASSES is merged exactly
// once, concurrent runs cannot lose each other's totals, and a damaged or
// incompatible state is refused instead of merged into.
void test_stats_state()
{
    std::puts("statistics state file (merge, overlap, concurrency, damage)");
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("gb_state_tests_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const std::string st = (dir / "st").string();
    ::setenv("GB_STATS_STATE", st.c_str(), 1);
    ::setenv("GB_STATS_LOG", (dir / "log").string().c_str(), 1);
    ::setenv("GB_RECORDS_FILE", (dir / "rec").string().c_str(), 1);
    ::setenv("GB_STATS_LIVE", (dir / "live").string().c_str(), 1);
    const auto cands = odd_primes_up_to(MAX_GB_PRIME);
    const StatsPlan plan = make_stats_plan(cands, STATS_K_DEFAULT, CHECK30_TAIL);
    const uint64_t S = 1000000000000ull;
    const auto slurp = [](const std::string& p) {
        std::string out; std::FILE* f = std::fopen(p.c_str(), "rb");
        if (!f) return out;
        char b[4096]; size_t r;
        while ((r = std::fread(b, 1, sizeof b, f)) > 0) out.append(b, r);
        std::fclose(f); return out;
    };
    const auto ranges = [&] {
        std::vector<std::string> v; const std::string all = slurp(st);
        size_t pos = 0;
        while ((pos = all.find("range ", pos)) != std::string::npos) {
            const size_t e = all.find('\n', pos); v.push_back(all.substr(pos, e - pos)); pos = e;
        }
        return v;
    };
    const auto commit = [&](uint64_t a, uint64_t b) {
        GoldbachStats g(cands, plan, CHECK30_TAIL);
        g.finish_run(a, b, (b - a) / 2 + 1);
        g.commit("test", 0);
    };
    const std::string r = "range " + std::to_string(S) + " ";
    commit(S, S + 1000);
    check(ranges() == std::vector<std::string>{r + std::to_string(S + 1000)}, "first run creates the state");
    commit(S + 1002, S + 2000);
    check(ranges() == std::vector<std::string>{r + std::to_string(S + 2000)}, "adjacent run merged into one range");
    const std::string before = slurp(st);
    commit(S + 500, S + 600);
    check(slurp(st) == before, "overlapping run leaves the state unchanged");
    {   // both instances load the state before either commits (concurrent runs)
        GoldbachStats d(cands, plan, CHECK30_TAIL), e(cands, plan, CHECK30_TAIL);
        d.finish_run(S + 10000, S + 11000, 501);
        e.finish_run(S + 20000, S + 21000, 501);
        e.commit("test", 0);
        d.commit("test", 0);
    }
    const auto rs = ranges();
    check(rs.size() == 3 && rs[1] == "range " + std::to_string(S + 10000) + " " + std::to_string(S + 11000) &&
          rs[2] == "range " + std::to_string(S + 20000) + " " + std::to_string(S + 21000),
          "concurrent commits both recorded (no lost update)");
    const std::string full = slurp(st);
    bool threw = false;
    try { GoldbachStats g(cands, plan, 1000); } catch (const std::exception&) { threw = true; }
    check(threw, "state recorded with another tail_t is refused");
    threw = false;
    try {
        GoldbachStats g(cands, plan, CHECK30_TAIL);
        g.finish_run(S + 30000, S + 31000, 501);
        g.commit("test", 7);                      // 7 sampled evens claimed, 0 recorded
    } catch (const std::exception&) { threw = true; }
    check(threw && slurp(st) == full, "sampled-evens mismatch throws, state unchanged");
    for (size_t cut : {size_t(0), full.size() / 2, full.size() - 1}) {
        { std::FILE* f = std::fopen(st.c_str(), "wb"); std::fwrite(full.data(), 1, cut, f); std::fclose(f); }
        threw = false;
        try { GoldbachStats g(cands, plan, CHECK30_TAIL); } catch (const std::exception&) { threw = true; }
        check(threw && slurp(st).size() == cut, "state cut to " + std::to_string(cut) + " bytes is refused, untouched");
    }
    for (const char* v : {"GB_STATS_STATE", "GB_STATS_LOG", "GB_RECORDS_FILE", "GB_STATS_LIVE"}) ::unsetenv(v);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

// Prime gaps (gstats.hpp GapStats, add_gap_segment, finish_run): segment
// outputs shaped like count30's tail and walk (built here from Miller-Rabin
// primes) give exactly the run's gaps q - prev(q), q in [start + 1, limit - 1]
// -- including a run that starts right after a twin pair -- and the gap
// state file round-trips, refuses damage and never merges a run twice.
namespace {
struct SegOut {           // one segment as the kernels report it
    uint32_t cnt[16] = {}, gt[2 + 2 * 64] = {}, hist[GAPS30_HBINS] = {};
    uint64_t wmax = 0;
};
SegOut seg_out(uint64_t q_lo, uint64_t q_hi, uint64_t qf, bool walk)
{
    static const int idx[30] = {-1, 0, -1, -1, -1, -1, -1, 1, -1, -1, -1, 2, -1, 3, -1,
                                -1, -1, 4, -1, 5, -1, -1, -1, 6, -1, -1, -1, -1, -1, 7};
    SegOut o;
    o.cnt[11] = ~0u;
    uint64_t prev = 0;
    for (uint64_t q = q_lo | 1; q <= q_hi; q += 2) {
        if (!is_prime_u64(q)) continue;
        const uint32_t p = uint32_t(q - qf);
        o.cnt[idx[q % 30]] += 1;
        o.cnt[11] = std::min(o.cnt[11], p); o.cnt[12] = p;
        if (prev) {
            const uint64_t g = q - prev;
            if (g >= GAPS30_TAIL) { o.gt[2 + 2 * o.gt[0]] = p; o.gt[3 + 2 * o.gt[0]] = uint32_t(g); ++o.gt[0]; }
            if (walk) {
                o.wmax = std::max(o.wmax, (g << 32) | uint32_t(~p));
                o.hist[g / 2] += 1;
            }
        } else if (walk) {
            o.hist[0] += 1;                               // the segment's first prime: no predecessor
        }
        prev = q;
    }
    return o;
}
std::string slurp_file(const std::string& p)
{
    std::string out;
    if (std::FILE* f = std::fopen(p.c_str(), "rb")) {
        char b[4096];
        size_t n;
        while ((n = std::fread(b, 1, sizeof b, f)) > 0) out.append(b, n);
        std::fclose(f);
    }
    return out;
}
} // namespace

void test_gap_stats()
{
    std::puts("prime gaps (count30 tail + walk stitching, run boundaries, gap state)");
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("gb_gap_tests_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const std::string st = (dir / "st").string(), gst = st + ".gaps";
    ::setenv("GB_STATS_STATE", st.c_str(), 1);
    ::setenv("GB_STATS_LOG", (dir / "log").string().c_str(), 1);
    ::setenv("GB_RECORDS_FILE", (dir / "rec").string().c_str(), 1);
    ::setenv("GB_STATS_LIVE", (dir / "live").string().c_str(), 1);
    const auto cands = odd_primes_up_to(MAX_GB_PRIME);
    const StatsPlan plan = make_stats_plan(cands, STATS_K_DEFAULT, CHECK30_TAIL);
    uint64_t tw = 4000000000000000001ull;                  // a twin pair (tw, tw + 2)
    while (!(is_prime_u64(tw) && is_prime_u64(tw + 2))) tw += 2;
    // runs at 4e18 (no gap >= 700 in 3e5) and across the gap 1476 after
    // 1425172824437699411 (the largest below 4e18: the tail path)
    const uint64_t big = 1425172824437699411ull;
    const uint64_t L = 30000;                             // evens span of a segment
    for (const uint64_t A : {tw + 3, uint64_t(4000000000000123456ull), big - 9000 - (big - 9000) % 2}) {
        const uint64_t B = A + 10 * L + 4000;             // a short last segment
        GoldbachStats gs(cands, plan, CHECK30_TAIL);
        gs.begin_gaps(A);
        std::vector<std::pair<uint64_t, bool>> runp;      // run primes and whether their segment was walked
        int si = 0;
        for (uint64_t s0 = A; s0 <= B; s0 += L, ++si) {
            const uint64_t s1 = std::min(s0 + L - 2, B), qf = s0 - 3 - 30;
            const bool walk = si % 3 == 0;                // segment 0 always walked, as in the verifier
            SegOut o = seg_out(s0 - 3, s1 - 3, qf, walk);
            gs.add_counts(o.cnt);
            gs.add_gap_segment(qf, (s1 - s0) / 2 + 1, o.cnt, o.gt, walk ? &o.wmax : nullptr, walk ? o.hist : nullptr);
            for (uint64_t q = (s0 - 3) | 1; q <= s1 - 3; q += 2)
                if (q >= A + 1 && is_prime_u64(q)) runp.push_back({q, walk});
            if (s1 == B && is_prime_u64(B - 1)) runp.push_back({B - 1, walk});
        }
        gs.finish_run(A, B, (B - A) / 2 + 1);
        const GapStats& r = gs.run_gaps();
        GapStats ref;
        for (auto& [q, walk] : runp) {
            const uint64_t g = q - prev_prime(q);
            ref.n += 1; ref.d1 += g;
            ref.note_exact(g, q - g);
            if (walk) ref.add_walk(g);
        }
        bool same = r.n == ref.n && r.d1 == ref.d1 && r.tail == ref.tail && r.wn == ref.wn && r.samp == ref.samp &&
                    (!r.gmax_exact() || (r.gmax == ref.gmax && r.gmax_lo == ref.gmax_lo));
        for (int k = 0; k < 4; ++k) same = same && r.wd[k] == ref.wd[k];
        check(same && r.n == runp.size(),
              std::string("run gaps exact vs q - prev(q)") + (A == tw + 3 ? " (start after a twin pair)" : "") +
              (A < 2000000000000000000ull ? " (gap 1476: tail path, exact maximum)" : ""));
        if (A < 2000000000000000000ull)
            check(r.tail.count(1476) && r.tail.at(1476).second == big && r.gmax == 1476 && r.gmax_lo == big,
                  "the gap 1476 is in the tail with its first lower prime");
        gs.commit("test", 0);
    }
    const std::string g1 = slurp_file(gst);
    check(g1.find("gaps_range ") != std::string::npos && g1.find("gaps_tail 1476 1 1425172824437699411\n") != std::string::npos &&
          g1.find("\nend\n") != std::string::npos, "gap state: ranges, tail record, end marker");
    check(slurp_file(st).find("gaps_range") == std::string::npos, "the main state holds no gap records (old binaries keep it intact)");
    bool threw = false;
    { std::FILE* f = std::fopen(gst.c_str(), "wb"); std::fwrite(g1.data(), 1, g1.size() / 2, f); std::fclose(f); }
    try { GoldbachStats g(cands, plan, CHECK30_TAIL); } catch (const std::exception&) { threw = true; }
    check(threw, "a cut-off gap state is refused");
    { std::FILE* f = std::fopen(gst.c_str(), "wb"); std::fwrite(g1.data(), 1, g1.size(), f); std::fclose(f); }
    {
        GoldbachStats g2(cands, plan, CHECK30_TAIL);
        g2.begin_gaps(tw + 3 + 500);                       // overlaps a recorded range
        g2.finish_run(tw + 3 + 500, tw + 3 + 600, 51);
        g2.commit("test", 0);
    }
    check(slurp_file(gst) == g1, "an overlapping run leaves the gap state untouched");
    {
        GoldbachStats g3(cands, plan, CHECK30_TAIL);
        const uint64_t A3 = 4000000002000000000ull;
        g3.begin_gaps(A3);
        g3.finish_run(A3, A3, 1);                          // one even: [start + 1, limit - 1] is empty
        check(g3.run_gaps().n == 0 && g3.run_gaps().d1 == 0, "a one-even run has no gaps");
        // kernel disagreement is caught: a walk that misses a gap, merges two,
        // restarts inside the segment, overflows, or disagrees with the tail
        GoldbachStats g4(cands, plan, CHECK30_TAIL);
        g4.begin_gaps(A3);
        const auto fails = [&](const SegOut& o, bool walk) {
            try { g4.add_gap_segment(A3 - 33, 1500, o.cnt, o.gt, walk ? &o.wmax : nullptr, walk ? o.hist : nullptr); }
            catch (const std::exception&) { return true; }
            return false;
        };
        const SegOut o0 = seg_out(A3 - 3, A3 + 3000, A3 - 33, true);
        uint32_t b1 = 1;                                  // the most common gap / 2
        for (uint32_t i = 2; i < GAPS30_HBINS; ++i) if (o0.hist[i] > o0.hist[b1]) b1 = i;
        SegOut o = o0;
        o.hist[b1] -= 1;
        check(fails(o, true), "a walk missing a gap fails the run");
        o = o0;
        o.hist[b1] -= 2; o.hist[2 * b1] += 1;
        check(fails(o, true), "a walk merging two gaps fails the run (gap count)");
        o = o0;
        o.hist[0] += 1; o.hist[b1] -= 1;
        check(fails(o, true), "a walk restarting inside the segment fails the run");
        o = o0;
        o.hist[GAPS30_HBINS - 1] += 1; o.hist[b1] -= 1;
        check(fails(o, true), "a gap in the overflow bin fails the run");
        o = o0;
        o.gt[2 + 2 * o.gt[0]] = o.cnt[12]; o.gt[3 + 2 * o.gt[0]] = GAPS30_TAIL; ++o.gt[0];
        check(fails(o, true), "a bit-parallel tail entry the walk did not see fails the run");
        o = seg_out(A3 - 3, A3 + 3000, A3 - 33, false);
        o.gt[1] = 2;
        check(fails(o, false), "a count30 gap error flag fails the run");
    }
    {
        // a gap backfill whose row a crash lost (gap state on disk, log line
        // not): its re-run writes the row, once
        const std::string lg = (dir / "log").string();
        const uint64_t A5 = 4000000003000000000ull, B5 = A5 + 200;
        // every even sampled, p(n) on the CPU: the backfill row carries this histogram
        std::vector<uint64_t> pcount(cands.size(), 0);
        for (uint64_t n = A5; n <= B5; n += 2)
            for (size_t i = 0; i < cands.size(); ++i)
                if (is_prime_u64(n - cands[i])) { ++pcount[i]; break; }
        const auto run = [&](bool gaps) {
            GoldbachStats g(cands, plan, CHECK30_TAIL);
            if (gaps) g.begin_gaps(A5);
            for (uint64_t n = A5; n <= B5; n += 2)
                for (const uint32_t q : cands)
                    if (is_prime_u64(n - q)) { g.add_cpu(n, q, true); break; }
            g.finish_run(A5, B5, (B5 - A5) / 2 + 1);
            g.commit("test", (B5 - A5) / 2 + 1);
        };
        const std::string tag = "GBGAPS v1 version=test start=" + std::to_string(A5) + " ";
        const auto rows = [&] {
            const std::string s = slurp_file(lg);
            size_t n = 0;
            for (size_t p = s.find(tag); p != std::string::npos; p = s.find(tag, p + 1)) ++n;
            return n;
        };
        run(false);                                        // the p(n) run (no gaps: an old binary)
        run(true);                                         // its gap backfill
        check(rows() == 1, "a gap backfill logs its row");
        {
            std::string want;                              // dense: zeros empty, trailing zeros cut
            size_t last = 0;
            for (size_t i = 0; i < pcount.size(); ++i) if (pcount[i]) last = i + 1;
            for (size_t i = 0; i < last; ++i) {
                if (i) want += ',';
                if (pcount[i]) want += std::to_string(pcount[i]);
            }
            const std::string lgs = slurp_file(lg);
            const std::string ptag = "GBPSAMP v1 version=test start=" + std::to_string(A5) + " ";
            const size_t r = lgs.find(ptag);
            const std::string row = r == std::string::npos ? "" : lgs.substr(r, lgs.find('\n', r) - r);
            check(row.find(" psev=" + std::to_string((B5 - A5) / 2 + 1) + " psamp=" + want) != std::string::npos,
                  "a backfill run logs its sampled p(n) histogram (GBPSAMP psev, psamp)");
        }
        const std::string g5 = slurp_file(gst);
        std::string s = slurp_file(lg);
        const size_t p = s.find(tag);
        s.erase(p, s.find('\n', p) + 1 - p);
        { std::FILE* f = std::fopen(lg.c_str(), "wb"); std::fwrite(s.data(), 1, s.size(), f); std::fclose(f); }
        run(true);
        check(rows() == 1, "a re-run writes the gap row a crash lost");
        run(true);
        check(rows() == 1 && slurp_file(gst) == g5, "a re-run with its row logged writes no gap row (gap state untouched)");
        {
            const std::string lgs = slurp_file(lg), ptag = "GBPSAMP v1 version=test start=" + std::to_string(A5) + " ";
            size_t n = 0;
            for (size_t q = lgs.find(ptag); q != std::string::npos; q = lgs.find(ptag, q + 1)) ++n;
            check(n == 3, "every backfill-path run logs its GBPSAMP row (a later re-run feeds Figure 4 too)");
        }
    }
    for (const char* v : {"GB_STATS_STATE", "GB_STATS_LOG", "GB_RECORDS_FILE", "GB_STATS_LIVE"}) ::unsetenv(v);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

int main()
{
    test_prime_lists();
    test_prime_cache();
    test_primality();
    test_pseudoprimes();
    test_limit_bounds();
    test_small_prime_masks();
    test_sparse_helpers();
    test_dense_plans();
    test_sparse_plan();
    test_stats_plan();
    test_ridge_store();
    test_stats_state();
    test_gap_stats();
    std::printf("host_tests: %d checks, %d failed -> %s\n", g_checks, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
