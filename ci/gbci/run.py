"""One CI run of one commit (Run): build, deployability, then with
production paused the GPU suite, the A/B benchmark and the CI window;
PASS promotes, anything else keeps the active release."""
import time

from .bench import benchmark_session
from .build import build_release
from .core import (EXACT_CSV, GPU_LOCK, NOTIFY_ENV, STATE, TIMING_KEYS, CIFail, cfg, flock, git, is_ancestor,
                   load_json, log, log_exc, now_iso, save_json, set_phase, sh, stamp)
from .gates import ci_figures, compare, ref_key, run_bench, run_slice
from .gpustack import GPU_STACK_FAILED, gpu_stack
from .journal import read_journal, stats_line
from .ledger import ledger_append
from .primecount import primes_in
from .prodstate import prod_conf, prod_paths, prod_stats_log, save_prod_conf
from .production import production_paused
from .releases import active_release, promote, release_meta
from .report import alert, write_report


class Run:
    """One CI run of one commit. execute() never raises for a candidate
    failure; production is resumed on every path (production_paused)."""

    def __init__(self, sha, refs, force_gpu=False, cpu_tests=True, stack_change=None):
        self.stack_change = stack_change   # set: re-verify the active release on a changed GPU stack
        self.c = cfg()
        self.sha, self.refs, self.force_gpu, self.cpu_tests = git("rev-parse", sha), refs, force_gpu, cpu_tests
        self.id = f"{stamp()}-{self.sha[:8]}"
        self.dir = STATE / "runs" / self.id
        self.dir.mkdir(parents=True)
        self.act = active_release()
        self.ameta = release_meta(self.act)
        self.pc = prod_conf()
        self.env = self.pc.get("env", {})
        self.seg = self.env.get("GB_RANGE_SEGMENT", "467500000")
        self.rec = {"run_id": self.id, "ts_start": now_iso(), "commit": self.sha, "refs": refs,
                    "kind": "gpu-stack" if stack_change else "commit", "gpu_stack": gpu_stack(),
                    "subject": git("log", "-1", "--format=%s", self.sha),
                    "previous_active": self.ameta.get("id", ""), "status": "RUNNING", "checks": []}
        self.state = {"run_id": self.id, "sha": self.sha, "refs": refs, "subject": self.rec["subject"],
                      "started": time.time(), "prod_stopped": False}

    def check(self, name, level, detail):
        self.rec["checks"].append({"name": name, "level": level, "detail": detail})

    def gate(self, msg):
        failed = [x["name"] for x in self.rec["checks"] if x["level"] == "fail"]
        if failed:
            raise CIFail(f"{msg}: {', '.join(failed)}")

    # ---- stages
    def build(self):
        set_phase(self.state, "build")
        _, rel, checks = build_release(self.sha, self.dir, self.refs, cpu_tests=self.cpu_tests)
        self.wt = STATE / "work" / "src"
        self.rec["checks"] += checks
        self.gate("build / CPU tests failed")
        self.rel, meta = rel, release_meta(rel)
        self.rec.update(release=meta["id"], version=meta["version"], fingerprint=meta["fingerprint"])

    def deployable(self):
        """None to continue, else the final status (DIVERGED / IDENTICAL)."""
        if self.stack_change:
            self.check("gpu_stack_changed", "warn", "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in
                                                             self.stack_change.items()))
            return None
        ac = self.ameta.get("commit")
        if ac and not self.c["promote_diverged"] and not is_ancestor(ac, self.sha):
            self.check("descends_from_active", "warn", f"does not contain the active commit {ac[:12]}: "
                                                       "built and tested only; merge/rebase onto it to deploy")
            return "DIVERGED"
        if self.act and self.rec["fingerprint"] == self.ameta.get("fingerprint") and not self.force_gpu:
            # no promotion: the code is the same, and moving the active commit to this
            # (possibly unmerged) branch tip would make every sibling branch that
            # lacks it DIVERGED. The older active commit is an ancestor of this one.
            self.check("artifact_identical", "ok", f"same code fingerprint as the active release "
                                                   f"{self.ameta.get('id')}: no GPU stage needed, active "
                                                   "release unchanged")
            return "IDENTICAL"
        return None

    def gpu_suite(self):
        mode = self.c["gpu_validate"]
        if not mode:
            return
        set_phase(self.state, "gpu_validate")
        args = ["bash", self.wt / "tests/gpu_validate.sh", self.wt / "build"] + (["quick"] if mode == "quick" else [])
        rc, out = sh(args, cwd=self.wt, logf=self.dir / "gpu_validate.log", timeout=3600)
        self.check(f"gpu_validate_{mode}", "ok" if rc == 0 else "fail", (out.strip().splitlines() or ["?"])[-1])
        self.gate("GPU validation suite failed")

    def bench(self):
        if not self.act or self.stack_change:     # same code on both arms: nothing to compare
            return
        set_phase(self.state, "bench")
        b = self.rec["bench"] = run_bench(self.act, self.rel, self.dir, self.seg)
        worst = max((d.get("gpu_slowdown_pct", 999) for d in b["windows"].values()), default=999)
        ok = b["rc"] == 0 and all(d.get("ok") for d in b["windows"].values())
        lvl = "fail" if not ok or worst > self.c["bench_fail_pct"] else "warn" if worst > self.c["bench_warn_pct"] else "ok"
        self.check("bench_ab", lvl, f"worst GPU-time change {worst:+.2f}% (fail > {self.c['bench_fail_pct']}%)")
        self.gate("benchmark regression or failed benchmark runs")

    def window(self):
        """(a, b, mode) of the CI window."""
        n = int(self.c["ci_slice"] or 0)
        if n:
            a = int(self.c["ci_window_start"] or self.pc["start"])
            return a, a + n - 2, "fixed"
        rows = [r for r in read_journal(prod_paths(self.pc)[0]) if r["tag"] == "PASS" and "attempts" not in r["kv"]]
        if not rows:
            raise CIFail("no banked slice for a full-slice CI window")
        return rows[-1]["a"], rows[-1]["b"], "journal"

    def reference(self, a, b, mode, fresh=False):
        """Reference result for the window: the journal row (full-slice mode),
        else the active release's cached result, run now if missing."""
        if mode == "journal":
            row = next(r for r in reversed(read_journal(prod_paths(self.pc)[0])) if r["tag"] == "PASS" and r["a"] == a)
            return {"kv": row["kv"], "stats_line": stats_line(prod_stats_log(self.pc), a, b),
                    "source": f"journal row {row['ts']}", "release": row["kv"].get("release", "?")}
        key = ref_key(self.ameta["fingerprint"], a, b, self.seg, self.env)
        cache = STATE / "ref-cache" / f"{key}.json"
        if cache.exists() and not fresh:
            ref = load_json(cache, {})
            ref["source"] = f"cached result of {ref.get('release')} ({ref.get('created')})"
            return ref
        set_phase(self.state, "slice_reference", window=[a, b], live=str(self.dir / "reference-files/live.json"))
        log(f"CI {self.id}: reference run of [{a}, {b}] with the active release {self.act.name}")
        res = run_slice(self.act, a, b, self.seg, self.dir, self.env, self.timeout(a, b), name="reference")
        if res["rc"] != 0 or res["csv"].get("result") != "PASS":
            raise RuntimeError(f"the ACTIVE release failed the reference window (rc={res['rc']}): "
                               "GPU / environment problem, not the candidate")
        self.ref_run = res
        ref = {"kv": dict(res["csv"], release=self.act.name), "stats_line": res["stats_line"],
               "release": self.act.name, "created": now_iso(), "source": "run now (same session)"}
        if self.prime_count is not None and str(self.prime_count) == res["csv"].get("primes"):
            save_json(cache, ref)            # only references primecount agrees with
        return ref

    def timeout(self, a, b):
        return max(600, int(3 * 1720 * (b - a + 2) / 2e14))

    def ci_window(self):
        a, b, mode = self.window()
        self.ref_run = None
        try:
            self.prime_count = primes_in(a, b)
        except RuntimeError as e:
            self.prime_count = None
            log(f"CI {self.id}: {e}")
        ref = self.reference(a, b, mode)
        set_phase(self.state, "slice", window=[a, b], live=str(self.dir / "slice-files/live.json"))
        log(f"CI {self.id}: window [{a}, {b}] with the candidate (reference: {ref['source']})")
        res = run_slice(self.rel, a, b, self.seg, self.dir, self.env, self.timeout(a, b))
        checks = compare(res, ref, self.prime_count)
        bad = [x for x in checks if x["level"] == "fail"]
        if bad and not self.stack_change and "cached" in ref["source"] and res["rc"] == 0 \
                and str(self.prime_count) == res["csv"].get("primes"):
            # the candidate agrees with primecount but not with the cached
            # reference: never trust a cache over a fresh run — redo it once
            log(f"CI {self.id}: cached reference disagrees ({', '.join(x['name'] for x in bad)}); re-running it")
            ref = self.reference(a, b, mode, fresh=True)
            checks = compare(res, ref, self.prime_count)
        self.rec["checks"] += checks
        if res["rc"] == 0 and "gpu_s" in res["csv"] and "gpu_s" in ref["kv"]:
            g, gr = float(res["csv"]["gpu_s"]), float(ref["kv"]["gpu_s"])
            pct = 100 * (g / gr - 1)
            lvl = "ok" if pct <= self.c["slice_warn_pct"] else "warn" if pct <= self.c["slice_fail_pct"] else "fail"
            self.check("window_gpu_time", lvl, f"{g:.1f}s vs {gr:.1f}s reference ({pct:+.2f}%)")
        self.rec["slice"] = {"start": a, "limit": b, "mode": mode, "rc": res["rc"], "wall_s": res["wall_s"],
                             "csv": {k: v for k, v in res["csv"].items() if k != "ridge_top"},
                             "ref": {k: ref["kv"].get(k) for k in TIMING_KEYS + EXACT_CSV},
                             "ref_source": ref["source"], "primecount": self.prime_count}
        self.cand_run, self.cand_key = res, (a, b)
        set_phase(self.state, "validate")
        self.rec["figures"] = ci_figures(self.dir, ([("active", self.ref_run)] if self.ref_run else [])
                                         + [("cand", res)])
        self.gate("validation failed")

    def remember_reference(self):
        """A PASS's own window result is the next run's reference (same code)."""
        a, b = self.cand_key
        if self.prime_count is None or str(self.prime_count) != self.cand_run["csv"].get("primes"):
            return
        meta = release_meta(self.rel)
        save_json(STATE / "ref-cache" / f"{ref_key(meta['fingerprint'], a, b, self.seg, self.env)}.json",
                  {"kv": dict(self.cand_run["csv"], release=self.rel.name), "stats_line": self.cand_run["stats_line"],
                   "release": self.rel.name, "created": now_iso()})

    # ---- driver
    def execute(self):
        log(f"CI {self.id}: {self.sha[:12]} ({', '.join(self.refs)}) — {self.rec['subject']}")
        t_down = None
        try:
            self.build()
            final = self.deployable()
            if final:
                self.rec["status"] = final
                return self.finish()
            self.rec["config"] = {"segment": self.seg, "split": 0, "bucket_verify": self.env.get("GB_BUCKET_VERIFY", "100"),
                                  "env_keys": sorted(k for k in self.env if k not in NOTIFY_ENV),
                                  "ci_slice": self.c["ci_slice"], "bench_windows": self.c["bench_windows"]}
            with production_paused(f"CI {self.id}", self.state) as info:
                self.rec["production"] = info
                t_down = time.monotonic()
                try:
                    with flock(GPU_LOCK, blocking=True, timeout=120):
                        self.gpu_suite()
                        self.bench()
                        self.ci_window()
                except CIFail:
                    if self.stack_change:      # the running release no longer reproduces its
                        self.state["no_resume"] = True       # own results: keep production stopped
                    raise
                self.rec["status"] = "PASS"
                save_json(STATE / "gpu-stack.json", self.rec["gpu_stack"])   # verified on this stack
                if self.stack_change:
                    log(f"CI {self.id}: the active release reproduces its reference on the new GPU stack")
                else:
                    promote(self.rel)
                    self.remember_reference()
                    prof = self.c["bench_on_promote"]
                    if prof:
                        try:
                            self.rec["benchmark"] = benchmark_session(self.dir / "benchmark", prof, pause=False,
                                                                      state=self.state)
                        except Exception as e:   # after promotion: informational, the PASS stands
                            self.rec["benchmark"] = {"error": f"{type(e).__name__}: {e}"}
                            log_exc(f"CI {self.id}: benchmark error (the PASS stands): {e}")
        except CIFail as e:
            self.rec["status"], self.rec["reason"] = "FAIL", str(e)
        except Exception as e:                  # infrastructure error: same safe path (ERROR, no promotion)
            self.rec["status"], self.rec["reason"] = "ERROR", f"{type(e).__name__}: {e}"
            log_exc(f"CI {self.id}: ERROR {type(e).__name__}: {e}")
        if self.state.get("no_resume"):
            pc = prod_conf()
            pc["desired"] = "stopped"
            save_prod_conf(pc)
            alert("GPU STACK CHANGE BREAKS RESULTS", f"{self.rec.get('reason')} after "
                  + ", ".join(self.stack_change) + " changed — production stopped; roll back the update "
                  "or re-validate (goldbach_ci.py production start once fixed)")
        if self.stack_change:
            if self.rec["status"] == "PASS":
                (STATE / GPU_STACK_FAILED).unlink(missing_ok=True)
            else:                               # not again on this stack (see gpu_stack_check_due)
                save_json(STATE / GPU_STACK_FAILED, self.rec["gpu_stack"])
                log(f"CI {self.id}: GPU stack check {self.rec['status']}; not retried on this stack "
                    f"(delete ci-state/{GPU_STACK_FAILED} to retry)")
        if t_down is not None:
            self.rec["downtime_s"] = int(time.monotonic() - t_down)
        return self.finish()

    def finish(self):
        rec = self.rec
        rec["ts_end"] = now_iso()
        write_report(self.dir, rec)
        ledger_append(rec)
        save_json(STATE / "state.json", {"phase": "idle", "last_run": rec["run_id"]})
        log(f"CI {rec['run_id']}: {rec['status']}" + (f" — {rec['reason']}" if rec.get("reason") else "")
            + f" — {self.dir / 'report.md'}")
        if rec["status"] in ("FAIL", "ERROR"):
            alert(f"CI {rec['status']}: {rec['commit'][:10]}",
                  f"{rec.get('reason', '')} — active release stays {release_meta(active_release()).get('id', '-')}")
        return rec
