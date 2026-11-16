"""gbci — the local CI's implementation, split by responsibility.

ci/goldbach_ci.py is the entry point and the public surface: it re-exports
every name below, so `goldbach_ci.<name>` works as before the split, and an
assignment to `goldbach_ci.<name>` (a test's monkeypatch) reaches the
modules here that use that name (see goldbach_ci._Facade). Import
goldbach_ci, not these modules, from scripts and tests.

Layering (each module imports only from the ones above it):

  core        constants, DEFAULTS / cfg(), log, JSON state files, sh / git,
              flock, set_phase, CSV / text helpers
  journal     production's journal: rows, resume chain, audits, stats lines
  prodstate   production.json, production's files and processes
  releases    fingerprints, active / previous links, promote, prune
  report      alert hook, report.md, progress line, heartbeat, annotations
  production  adopt / status / stop / start, production_paused
  primecount  pi(x) via primecount (cached), Miller-Rabin
  gpustack    the GPU software stack and its changes
  build       CI worktree build, kernel freshness, CPU tests, staging
  gates       isolated env, bench.py A/B, CI window run, ref_key, compare
  bench       benchmark sessions and their analysis
  ledger      ledger.jsonl / results.csv, retry queue, new branch tips
  run         Run: one CI run of one commit
  audit       count_check and recheck of banked slices
  daemon      recover() and poll_once()
  cli         cmd_* and main()
"""
