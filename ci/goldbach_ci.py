#!/usr/bin/env python3
"""goldbach_ci.py — local CI for goldbach-l0 (stdlib only).

One CI run per commit (GitHub Actions job `gpu`, or the daemon), see
ci/README.md:

  build in a separate worktree (+ CPU tests unless Actions already ran them)
    -> same code as the active release?  IDENTICAL: done (active unchanged)
    -> pause run_range (slice boundary, or now if the slice just started)
    -> tests/gpu_validate.sh quick -> bench.py A/B vs the active release
    -> CI window (fixed 2e13 integers at the campaign start, production args
       and env, isolated statistics files) with the candidate; compared bit
       for bit with the active release's result for the same window (cached
       per release) and with primecount
    -> PASS: promote (previous kept for rollback), quick benchmark session
       FAIL / ERROR: the active release stays
    -> resume run_range with the active release; resume point verified

State: ci-state/ (gitignored) — releases/<id>/, active + previous symlinks,
ledger.jsonl + results.csv, runs/<id>/ (logs, report.md, figures),
production.json (the run_range invocation, may hold secrets: 0600),
state.json (current phase: crash recovery, dashboard), queue.json (runs to
retry), ref-cache/ (reference window results per release). GPU work is
serialised by an flock on .goldbach-gpu.lock, held by tools/run_range.sh for
its whole life; CI runs are serialised by ci-state/ci.lock.
"""
# The implementation lives in ci/gbci/ (one module per responsibility, see
# gbci/__init__.py). This file is the entry point and the import surface:
# scripts (experiment.py, production_env.py, goldbach_live.py, goldbach-tools)
# and tests use `goldbach_ci.<name>` with ci/ on sys.path, as before the split.
import sys
import types as _types
from pathlib import Path

_CI_DIR = str(Path(__file__).resolve().parent)
if _CI_DIR not in {str(Path(p).resolve()) for p in sys.path if p}:
    sys.path.insert(0, _CI_DIR)       # loaded by file path: gbci is next to this file

# stdlib names the single-file goldbach_ci.py had at module level
import argparse  # noqa: E402,F401
import contextlib  # noqa: E402,F401
import csv  # noqa: E402,F401
import datetime as dt  # noqa: E402,F401
import fcntl  # noqa: E402,F401
import fnmatch  # noqa: E402,F401
import hashlib  # noqa: E402,F401
import json  # noqa: E402,F401
import math  # noqa: E402,F401
import os  # noqa: E402,F401
import re  # noqa: E402,F401
import shlex  # noqa: E402,F401
import shutil  # noqa: E402,F401
import signal  # noqa: E402,F401
import statistics as st  # noqa: E402,F401
import subprocess  # noqa: E402,F401
import tempfile  # noqa: E402,F401
import threading  # noqa: E402,F401
import time  # noqa: E402,F401

from gbci import audit as _audit, bench as _bench, build as _build, cli as _cli, core as _core  # noqa: E402
from gbci import daemon as _daemon, gates as _gates, gpustack as _gpustack, journal as _journal  # noqa: E402
from gbci import ledger as _ledger, primecount as _primecount, production as _production  # noqa: E402
from gbci import prodstate as _prodstate, releases as _releases, report as _report, run as _run  # noqa: E402
from gbci.audit import count_check, recheck  # noqa: E402,F401
from gbci.bench import (BENCH_PRESETS, PHASES, bench_analysis, benchmark_session, history_build,  # noqa: E402,F401
                        window_span)
from gbci.build import build_release, kernel_freshness, tools_worktree, worktree  # noqa: E402,F401
from gbci.cli import (cmd_benchmark, cmd_bootstrap, cmd_count_check, cmd_daemon, cmd_install_service,  # noqa: E402,F401
                      cmd_poll, cmd_production, cmd_recheck, cmd_rollback, cmd_run, cmd_status,
                      sigterm_as_interrupt)
from gbci.core import (BENCH, BENCH_PROFILES, DEFAULTS, EXACT_CSV, FINAL, GPU_LOCK, NOTIFY_ENV,  # noqa: E402,F401
                       ON_ACTIONS, RESULT_COLS, ROOT, RUN_RANGE, STATE, STATS_NEW_OK, STATS_STATEFUL, TIMING_KEYS,
                       UNIT, WATERFALL, CIFail, cfg, csv_header, csv_rows, flock, git, hms, is_ancestor, killpg, kv,
                       load_json, log, now_iso, plot_python, read_lines, save_json, set_phase, sh, stamp)
from gbci.daemon import poll_once, recover  # noqa: E402,F401
from gbci.gates import ci_figures, compare, isolated_env, ref_key, run_bench, run_slice  # noqa: E402,F401
from gbci.gpustack import (GPU_STACK_FAILED, GPU_STACK_PKGS, gpu_stack, gpu_stack_change,  # noqa: E402,F401
                           gpu_stack_check_due)
from gbci.journal import chain_next, journal_audit, median_slice_wall, read_journal, stats_line  # noqa: E402,F401
from gbci.ledger import (attempts, candidates, judged, ledger, ledger_append, ledger_index, list_refs,  # noqa: E402,F401
                         mark_seen, queue_edit)
from gbci.primecount import is_prime, pi, primecount_bin, primes_in  # noqa: E402,F401
from gbci.prodstate import (find_run_range, pgrep, proc_cmdline, prod_conf, prod_live, prod_paths,  # noqa: E402,F401
                            prod_stats_log, save_prod_conf)
from gbci.production import (adopt_production, last_run_range_event, production_paused,  # noqa: E402,F401
                             production_status, start_production, stop_now_is_cheaper, stop_production)
from gbci.releases import (active_release, fingerprint, link_target, promote, prune_releases,  # noqa: E402,F401
                           release_dir, release_meta, set_link)
from gbci.report import (_energy, alert, annotate, bench_progress, gpu_stats, heartbeat,  # noqa: E402,F401
                         progress_line, write_report)
from gbci.run import Run  # noqa: E402,F401

_MODULES = (_core, _journal, _prodstate, _releases, _report, _production, _primecount, _gpustack, _build,
            _gates, _bench, _ledger, _run, _audit, _daemon, _cli)
_UNSET = object()


class _Facade(_types.ModuleType):
    """goldbach_ci's module type. Before the split every function looked its
    globals up in this one module, so `goldbach_ci.primes_in = fake` (a test,
    a goldbach-tools hook) changed what count_check called. Now each gbci
    module binds the names it imports itself; an assignment here is passed on
    to every gbci module that binds the same object under that name, which
    keeps that behaviour (patching back restores them all)."""

    def __setattr__(self, name, value):
        old = self.__dict__.get(name, _UNSET)
        if old is not _UNSET and not name.startswith("__"):
            for m in _MODULES:
                if m.__dict__.get(name, _UNSET) is old:
                    setattr(m, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Facade


def main():
    _cli.main(__doc__)


if __name__ == "__main__":
    main()
