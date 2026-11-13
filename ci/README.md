# Local CI (`ci/goldbach_ci.py`)

CI for the machine that runs the production campaign, driven from GitHub
Actions on two self-hosted runners (or by the local daemon). A commit becomes
the production binary only after it has built, passed every test suite,
benchmarked no slower than the running release, and verified a production
window with results bit-identical to the running release's and to
`primecount`. The previous release stays one command away.

```
push
  GitHub jobs kernels / host / static / full  (goldbach-cpu, ~1.5 min)       production running
  GitHub job gpu (goldbach-gpu) = goldbach_ci.py run $SHA --skip-cpu-tests:
    -> build in ci-state/work/src (git worktree; cmake Release -Werror;      ~20 s
       goldbach-tools from its committed HEAD in ci-state/work/goldbach-tools,
       never the live checkout — the log names the tools commit)
    -> same code fingerprint as the active release? -> IDENTICAL, promote, done
    -> pause run_range: at the slice boundary (<= one 5e13 slice, ~7 min), production paused
       or at once if redoing the slice in progress is cheaper than waiting
    -> tests/gpu_validate.sh quick                                           ~1 min
    -> bench.py A/B, active vs candidate, long + long-end, 5 paired rounds   ~1 min
    -> CI window: [START, START + 2e13), production arguments + env,         ~3 min
       isolated statistics files, run by the candidate and compared with
       the active release's result for the same window (cached per release;
       computed once) and with primecount
    -> PASS: active -> candidate (previous kept), quick benchmark session    ~10 min
       FAIL / ERROR: the active release stays
    -> resume run_range with the active release; resume point verified       production running
```

A typical code change pauses production for ~15 min (5 without the
benchmark session); a docs / CI-script commit costs no GPU time at all.

## Acceptance criteria (every `fail` check must pass)

| check | level | rule |
|---|---|---|
| `build` | fail | cmake Release `-DGB_WERROR=ON` (as the production build) |
| `kernels_fresh`, `ctest_cpu` | fail | shipped `.spv` == fresh pinned-ocloc compile; `ctest -LE gpu` (GitHub jobs `kernels` / `host` when triggered from Actions) |
| `gpu_validate_quick` | fail | exact statistics vs `stats_ref`, end to end in every checker mode, bucket verification, top of the supported range |
| `bench_ab` | fail > 2%, warn > 0.5% | candidate GPU time vs the active release, worst window, paired rounds |
| `slice_exit` | fail | rc 0, `result=PASS` |
| `exact_*` | fail | `result leftovers segments primes twins stats_tail stats_sampled stats_pmax start limit` equal to the reference |
| `exact_stats_line` | fail | the window's `GBSTATS` line (p(n) tail histogram, sampled head, π and π₂ by residue) equal to the reference's, except `version` / `new_first` |
| `primecount` | fail | GPU prime count of the window == `primecount` (an independent algorithm) |
| `window_gpu_time` | fail > 10%, warn > 1.5% | vs the reference run (gross regressions; `bench_ab` is the precise gate) |

**The reference** is the active release's own result for the CI window,
cached in `ci-state/ref-cache/` under its code fingerprint, the window and
every result-relevant `GB_*` setting — only if it matched `primecount`. A
PASS stores the candidate's result as the next reference, so the reference
run happens once, not per CI run. A cached reference is never trusted over a
fresh run: if the candidate matches `primecount` but not the cache, the
reference is re-run before anything fails. `"ci_slice": 0` switches to the
old mode: the last banked slice in full (~28 min) against its journal row.

Thresholds: `DEFAULTS` at the top of `goldbach_ci.py`, overridable in
`ci-state/config.json`.

## Watching it

- **GitHub Actions**: the `gpu` job log prints a progress line every minute
  (waiting for the slice boundary: % and ETA; CI window: %, G evens/s, ETA);
  failed / warned checks are annotations; `report.md` is the job summary;
  logs, the window's outputs and the figures are the `gpu-ci-<sha>` artifact.
- **`scripts/goldbach_live.py`**: a CI panel (phase, commit, window progress)
  and, on the campaign line, "paused by the local CI" / "pauses production
  after this slice".
- **`python3 ci/goldbach_ci.py status`**: releases, production, the current
  phase with progress, the retry queue, the last runs.

## Benchmarks and figures

- Every CI run renders `waterfall-<window>.png` (active vs candidate, GPU
  phases per 1e10 integers) from its own bench + window data.
- After a PASS that changes code (`"bench_on_promote": "quick"`, ~10 min:
  current, + production config, previous release; long, long-end, 2e12) and
  on demand (`goldbach_ci.py benchmark --profile full|quick`, or the
  **Benchmark** workflow; full ~40 min: + `GB_STATS=0`, `bench_history`
  versions, sparse, every window): rows go to `bench/bench-results.csv`;
  `benchmark.md` holds medians, figures and, for every arm, the per-phase
  difference (sieve / large / check / sync) that makes it faster or slower.

## Commands

```bash
python3 ci/goldbach_ci.py status
python3 ci/goldbach_ci.py run [REV]            # one CI run now (--force-gpu, --rerun, --skip-cpu-tests)
python3 ci/goldbach_ci.py benchmark [--profile quick|full]
python3 ci/goldbach_ci.py production stop [--now] | start | status | adopt
python3 ci/goldbach_ci.py rollback [--now]     # previous release -> active, restart production
python3 ci/goldbach_ci.py count-check          # banked slices' prime counts vs primecount
python3 ci/goldbach_ci.py recheck [START]      # re-run queued (or one) banked slice(s), compare
python3 ci/goldbach_ci.py poll | daemon        # retry queue, branch watching, count checks
python3 ci/goldbach_ci.py bootstrap            # first time: HEAD -> active release
```

## Production control

- **One command**: `ci/goldbach` (installed as `~/.local/bin/goldbach`):
  `up` (the default) enables and starts the daemon, the site sync and the
  campaign queue, starts production unless it runs, pushes to the site and
  prints `status`; `down` stops production at the next slice boundary
  (`desired` = stopped, so it stays off across reboots until `up`);
  `status`; `logs`. The units are enabled and the user lingers, so after a
  reboot production comes back by itself (`recover`, below).
- **Site sync**: `goldbach-range` wants `goldbach-sync.timer` / `.path`, and
  the campaign queue starts them whenever run_range runs without them.
- **One GPU user at a time**: `tools/run_range.sh` holds an flock on
  `.goldbach-gpu.lock` for its whole life and refuses to start while any
  `goldbach_l0` runs; the CI takes the same lock for its GPU stages; CI runs
  are serialised by `ci-state/ci.lock` (Actions jobs, the daemon, commands).
- **Slice size**: production runs `"slice"` from `ci-state/production.json`
  (5e13 integers, ~7 min, since 2026-10-05; 2e14 before). It bounds how long
  the CI waits for a boundary; each slice costs ~1 s of fixed overhead
  (~0.25% at 5e13). The journal chain is unaffected by a size change.
- **Pause**: `<journal base>.stop` makes run_range exit (code 3) before its
  next slice; when redoing the slice in progress costs less than waiting for
  its boundary (and less than `stop_max_loss_s`, 300 s) it is stopped at once
  and simply re-runs (the report shows `lost_s`). Banked work is never
  touched; the journal is audited (resume point, duplicates) after every stop.
- **Prime-count check**: the daemon checks banked slices against
  `primecount` in blocks of consecutive slices covering `count_block`
  (2e14) integers — one primecount call per block whatever the slice size;
  a mismatching block is re-checked slice by slice to name the slice.
- **Resume**: `systemd-run --user --unit=goldbach-range` with the adopted
  arguments, `GB_*` environment and nice value, `GB_RANGE_BIN` = the active
  release; run_range's `resuming at N release=R` line is checked against the
  journal chain and the active release. This happens on every path out of a
  run — PASS, FAIL, ERROR, SIGTERM (a cancelled job).
- Every PASS row names its `release=<id>`.

## Releases, rollback, reboots

`ci-state/releases/<commit12>/`: `goldbach_l0`, `stats_ref`, `kernels/*.spv`,
`RELEASE`, `meta.json` (commit, version, code fingerprint). `active` /
`previous` are symlinks switched by atomic rename; a code-identical promotion
keeps `previous`. Kept: active, previous, every release named in the journal,
the one production runs, the 3 newest others.

After a crash or reboot the daemon (`recover`) kills GPU jobs a dead CI left
behind, records the run as `INTERRUPTED`, **queues it for a retry**
(`max_attempts` 2), and resumes production unless it had stopped on a
failure. A failed run never changes the active release.

## System updates (`pacman -Syu`)

- **GPU stack** — kernel, `xe` module, `intel-compute-runtime`,
  `intel-graphics-compiler`, `level-zero-loader`, VRAM clock: the release
  binaries hand their SPIR-V to the installed compiler at every slice start,
  so an update changes production's GPU code without a commit. The daemon
  compares the stack with the one the active release was last verified on
  (`ci-state/gpu-stack.json`); on a change it pauses production and re-runs
  the GPU suite + the CI window with the **same** release, which must
  reproduce its cached reference bit for bit and match primecount (a
  `gpu-stack-check` run in the ledger). If not, production stays stopped
  with an alert: roll the update back, then `production start`. A failed
  check is recorded in `ci-state/gpu-stack-failed.json` and not repeated on
  that exact stack (any further change re-runs it; delete the file to retry
  by hand). A change of
  the VRAM clock (`mem_speed`; stock 19000 on this machine) triggers the same
  check.
- **ocloc newer than the pinned CR_VER**: the GitHub `kernels` job warns and
  skips only the byte-identical `.spv` comparison (it can't be judged with
  another compiler); rebuild `kernels/*_bmg.spv` with the new ocloc and bump
  `CR_VER` to restore it.
- **Compilers**: a gcc / clang update can add `-Werror` warnings, failing
  builds until fixed (production unaffected); the first commit afterwards
  takes the full GPU path (new code fingerprint).
- **Reboots**: services are enabled with linger; the daemon recovers
  (retry queue, production resume) and the runners reconnect by themselves.

## Records

`ci-state/ledger.jsonl` (one record per run: refs, commit, version, release,
configuration, window, timings, reference, bench, every check, status,
downtime), `results.csv` (fixed columns), `runs/<id>/` (logs, outputs,
`report.md`, figures), `count-check.jsonl`, `queue.json`, `ref-cache/`.

## Setup

```bash
bash ci/setup-runners.sh                             # goldbach-cpu + goldbach-gpu runners
echo '{"watch_include": []}' > ci-state/config.json  # pushes are the trigger
python3 ci/goldbach_ci.py install-service --enable   # daemon: recovery, retries, count checks
loginctl enable-linger "$USER"                       # survive logout / reboot
```

Anyone who can push to the repository runs code on this machine: keep it
private. Needs `primecount` (PATH, `~/.local/src/primecount/build`, or
`"primecount"` in the config; without it the count checks warn), `ocloc` for
kernel freshness, a systemd user session. The CI's own logic is tested by
`tests/test_ci.py` (GitHub job `static`).
