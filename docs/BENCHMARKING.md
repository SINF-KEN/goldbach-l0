# Benchmarking methodology

Raw data: [`bench/`](../bench/) — `bench-results.csv` is the append-only
journal every `tools/bench.py` session writes (one row per run, never
edited); `bench-20260926-sparse-v2.csv` is the v2.0 sparse session. Figures:
`bash docs/make-plots.sh` (needs pandas + matplotlib) renders
`docs/images/` from the journal. Results per change are recorded in
[PROJECT-REFERENCE.md](PROJECT-REFERENCE.md) §9.2.

## The measure

**Evens per second over GPU time** (`gpu_s`: the wall time of the segment
loop; process start-up and table upload, `init_s`, are excluded). Every
kernel, including work that runs only every K segments (the hit-list
harvest), is inside `gpu_s`. The per-stage columns (`sieve_s`, `large_s`,
`check_s`) are summed kernel timestamps; `sync_s` is the remainder
(host-side gaps, pipeline bubbles).

## Windows

| Window | Range | Use |
|---|---|---|
| `long` | 1e11 at 4e18 (214 segments of 467.5M) | production-like, the primary figure |
| `long-end` | 1e11 ending at 5e18 | the hardest end of the campaign (~11% more base primes) |
| `steady` / `end` | 2e10 at 4e18 / ending at 5e18 | quick checks |
| `bench` | 2e9 at 4e18 | start-up dominated; not for kernel A/Bs |

Production runs 2e14-integer slices, which behave like `long`.

## Procedure

1. **Same session, same conditions.** Compare builds only within one
   `bench.py` session: rounds rotate through every (window, arm) pair in
   alternating order, each run a fresh process, the first `--warmup` runs of
   each pair discarded. Arms are builds (`NAME:@BINARY@SPV`) or settings
   (`NAME:ENV=V,...`); base and variants are compiled from the same source.
2. **Paired statistics.** Report the geometric mean of per-round ratios
   against the base arm with its 95% t-interval (`bench.py` prints it).
   Per-run noise is ~0.03–0.3% CV; use at least 5 paired rounds (10 for
   effects below 0.2%). Keep a change only if its interval clears the base
   on **both** `long` and `long-end`.
3. **Exclusive GPU.** No production run, CI run or other GPU client during a
   session (`other_gpu_pct` must be 0; `--exclude-contended` drops runs
   otherwise). Pause production at a slice boundary
   (`ci/goldbach_ci.py production stop`), hold `.goldbach-gpu.lock`, and
   release it before restarting production.
4. **Isolate the campaign files** (`GB_STATS_*`, `GB_RIDGE_FILE`,
   `GB_RECORDS_FILE`, `GB_STATS_LIVE` -> a scratch directory,
   `GB_RECORD_HOOK=`): `bench.py` itself does not, and a benchmark run from
   the repository root would otherwise merge into the production
   statistics.
5. **Stock clocks.** The B580 runs at its stock memory clock; the large-prime
   kernels' time tracks it, so overclocked numbers are not comparable.
6. **Explain the number.** For a gain, check the stage columns (and
   `GB_PHASE=1` per-kernel times): a change that lowers kernel time but
   raises `sync_s` has moved the bottleneck (v2.3.0: the faster GPU segment
   exposed the host's per-segment preparation).

## Ablations

Timing-only `-D` switches in `kernels/goldbach.cl` (`ABL_*`, e.g.
`ABL_SKIPOH`, `ABL_NOMED`) remove one stage to measure its cost; their maps
are invalid, so read the stage columns, never the result. Compile them into
separate modules and run them as arms.
