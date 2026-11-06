# Development: build, test, CI, shipping, reproducibility

## Hardware and software

- **GPU**: developed and tuned on one Intel Arc **B580** (Xe2 / Battlemage,
  20 Xe-cores, 18 MB L3, 12 GB GDDR6 at the stock 19 Gbps — no memory
  overclock). Kernels are compiled for `-device bmg`; other Xe2 parts
  should work but are untested.
- **Driver**: Intel compute-runtime (Level Zero, `xe` kernel driver),
  developed against 26.31 and 26.35. The kernel modules ship pre-built
  with the **pinned ocloc 26.35.39758.10** (`.github/workflows/ci.yml`
  `CR_VER`).
- **Display**: keep the desktop on another GPU (or run headless): a
  compositor on the B580 costs ~20% and makes timings bimodal.
- **Host**: Linux, CMake >= 3.24, a C++20 compiler (gcc or clang), the Level
  Zero loader and headers; Python 3 for the tests and the CI; `ocloc`
  optional (needed only to rebuild kernels).

## Build

```bash
bash build.sh                       # = cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cmake -S . -B build -DGB_WERROR=ON  # what CI builds (strict warnings as errors)
```

With `ocloc` on `PATH`, CMake recompiles `kernels/*_bmg.spv` whenever a
kernel source changes; without it the shipped modules are used. The
validators and benchmark tools live in the companion repository
[goldbach-tools](https://github.com/SINF-KEN/goldbach-tools), cloned next to
this one (`tools -> ../goldbach-tools`); without it only `goldbach_l0`,
`host_tests` and `stats_ref` are built.

## Tests

| Command | Needs | What |
|---|---|---|
| `ctest --test-dir build -LE gpu` | nothing | `host_tests` (host math vs brute force, cache and state damage, plans), `check_constants` (host <-> kernel constant mirrors), `cli_tests` (CLI contract), `stats_known` (p(n) vs Oliveira e Silva's table), `stats_tools`, `known_windows` (the reference table of an independent verifier's published windows, and the comparison logic) |
| `ctest --test-dir build -L gpu` | the GPU | `tests/gpu_validate.sh`: exact statistics vs `stats_ref` (incl. across 2^63), the published windows of `tests/known_windows.py`, end to end in every checker mode, hit lists / bucket ring vs a full rebuild, the bit-exact kernel validators (~12 min; `bash tests/gpu_validate.sh build quick` ~2 min) |
| `python3 tests/test_ci.py` | nothing | the local CI's logic (journal chain, caches, gates) |
| `./build/debug_map`, `debug_check_hi`, ... | the GPU, tools | the kernel validators individually |

Run the GPU suite after any kernel or host change. Every regression test
added by the v2.3.0 audit fails on the code before its fix.

**Isolate the campaign files.** A run from the repository root reads and
writes `goldbach_stats.state`, `goldbach_stats.log`, `goldbach_ridges.csv`,
`goldbach_records.txt` and `goldbach_live.json`. Test and benchmark runs
must point `GB_STATS_STATE`, `GB_STATS_LOG`, `GB_RIDGE_FILE`,
`GB_RECORDS_FILE`, `GB_STATS_LIVE` at a scratch directory and set
`GB_RECORD_HOOK=` (as `tests/gpu_validate.sh` and the CI do).

## Continuous integration

**GitHub Actions** (`.github/workflows/ci.yml`, self-hosted runners, no GPU
in the CPU jobs):

| Job | Checks | Why |
|---|---|---|
| kernels | every module compiles with the pinned ocloc; each shipped `.spv` contains every `__kernel`, passes `spirv-val`, and is byte-identical to a fresh compile | a kernel edited without rebuilding its module is caught; `spirv-val` also runs without the pinned compiler |
| host | gcc and clang `-Werror`, clang ASan + UBSan, each running `ctest -LE gpu` | UB and memory errors in the host math |
| static | constant mirrors, shellcheck, actionlint, Python syntax, `test_ci.py` | host/kernel drift, script and workflow errors |
| full | every target incl. the validators against goldbach-tools **at its committed HEAD** (a clone, not the live checkout); benchmark figures | the tools build; reproducible inputs |
| gpu (pushes) | the local CI run below | GPU correctness and performance |

**Local CI** (`ci/goldbach_ci.py`, details in [ci/README.md](../ci/README.md)):
for every new commit it builds in its own worktree (goldbach-tools from its
committed HEAD; the log names the commit), runs the CPU and GPU suites,
A/Bs against the active release with `bench.py`, re-verifies one complete
production slice bit for bit plus an independent prime count, and only then
promotes the build to the release production runs. It pauses production at
a slice boundary and resumes it afterwards.

CI checks considered and not added: `clang-tidy` (its bug-finding checks
report only false positives on this code; the header noise needs a
maintained configuration), compiling every non-default `-D` variant
(moderate cost for code that is not shipped — prune superseded variants
instead).

## Changing a kernel (the shipping procedure)

1. Add the change as a default-off `-D` switch; check that the default
   module stays byte-identical to `kernels/goldbach_bmg.spv`.
2. Validate: `debug_map` (all cases OK); for large-prime / bucket / hit-list
   changes `GB_BUCKET_VERIFY=1` over 120 segments at 4e18 and ending at
   5e18; for checker changes `debug_check_hi` plus a mutant that must fail;
   always a 2e10 statistics-identity run (isolated files, `stats.state`
   byte-identical to the base).
3. Measure with `tools/bench.py` ([BENCHMARKING.md](BENCHMARKING.md)).
4. Keep a winner: set its default, rebuild `kernels/goldbach_bmg.spv` with
   the pinned ocloc (byte-identical to the benchmarked module), run
   `ctest -LE gpu`, document the result (kept or dropped) in
   [PROJECT-REFERENCE.md](PROJECT-REFERENCE.md) §9.2, commit, push; push
   the next change only after the local CI passed on this one.

## Reproducibility

- Kernel modules: deterministic for the pinned ocloc; CI enforces
  byte-identity. Only `.spv` is shipped (`.bin` is local ocloc output).
- Base primes: regenerated deterministically when no valid cache exists;
  caches are content-checked on load.
- Results: `--csv` reports `version=`; the statistics log records each run;
  production journal rows record the release and the verifier's full output.
- The companion goldbach-tools repository is versioned separately; CI
  builds use its committed HEAD and log the commit.

## Production operation

`tools/run_range.sh START LIMIT` runs a resumable campaign in 2e14-integer
slices (journal `range-START-LIMIT.journal`, one `PASS` row per slice).
Under the local CI the active release runs it as a systemd user unit;
`python3 ci/goldbach_ci.py production stop|start|status` controls it (stop
waits for a slice boundary unless the current slice just started). Only one
GPU job runs at a time (flock `.goldbach-gpu.lock`); benchmarks and tests
must wait for a pause, and release the lock before production restarts.
