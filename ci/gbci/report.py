"""Reporting and notifications: the alert hook, report.md / report.json,
the one-line progress (status, Actions heartbeat, dashboard), GitHub
Actions annotations."""
import contextlib
import os
import shlex
import statistics as st
import subprocess
import threading
import time
from pathlib import Path

from .core import ON_ACTIONS, ROOT, STATE, TIMING_KEYS, csv_rows, hms, load_json, now_iso, save_json
from .prodstate import prod_conf, prod_live


def alert(title, msg):
    """Notification hook (scripts/notify_record.sh ALERT, production's env)."""
    hook = os.environ.get("GB_RECORD_HOOK", "bash scripts/notify_record.sh")
    if not hook:
        return
    env = dict(os.environ)
    env.update(prod_conf().get("env", {}))
    subprocess.Popen(shlex.split(hook) + ["ALERT", title, msg, "-", "-"], cwd=ROOT, env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def write_report(run_dir, rec):
    icon = {"PASS": "✅", "IDENTICAL": "✅", "DIVERGED": "⚠️", "FAIL": "❌", "ERROR": "❌"}.get(rec["status"], "")
    L = [f"# {icon} {rec['status']} — `{rec.get('commit', '')[:12]}` ({', '.join(rec.get('refs', []))})", "",
         f"{rec.get('subject', '')}", ""]
    if rec.get("reason"):
        L += [f"**{rec['reason']}**", ""]
    L += ["| check | result | detail |", "|---|---|---|"]
    L += [f"| {x['name']} | {x['level'].upper()} | {x['detail']} |" for x in rec.get("checks", [])]
    pr = rec.get("production", {})
    L += ["", f"version {rec.get('version', '?')} · release `{rec.get('release', '-')}` · previous active "
              f"`{rec.get('previous_active', '-')}` · {rec.get('ts_start', '')} → {rec.get('ts_end', '')} · "
              f"production paused {rec.get('downtime_s', 0)} s"
              + (f" ({pr['mode']}, {pr['lost_s']} s of a partial slice redone)" if pr.get("lost_s") else ""), ""]
    if rec.get("slice"):
        s = rec["slice"]
        L += [f"## CI window [{s['start']}, {s['limit']}]", "", f"reference: {s.get('ref_source', '?')}; "
              f"primecount: {s.get('primecount')}", "", "| metric | candidate | reference |", "|---|---|---|"]
        L += [f"| {k} | {s['csv'].get(k, '')} | {s.get('ref', {}).get(k, '')} |" for k in TIMING_KEYS]
        L.append("")
    if rec.get("bench", {}).get("windows"):
        L += ["## bench.py A/B (interleaved, paired)", "",
              "| window | active G ev/s | candidate G ev/s | speed ratio | ±95% |", "|---|---|---|---|---|"]
        L += [f"| {w} | {d.get('active_gevs', '')} | {d.get('cand_gevs', '')} | {d.get('speed_ratio', '')} | "
              f"{d.get('ci_pct', '')}% |" for w, d in rec["bench"]["windows"].items()]
        L.append("")
    b = rec.get("benchmark", {})
    if b.get("dir"):
        L += [f"## Benchmark session ({b['profile']})", "", (Path(b["dir"]) / "benchmark.md").read_text()
              .split("## Where the time goes", 1)[-1].split("## Figures", 1)[0].strip(), ""]
    if rec.get("figures") or b.get("figures"):
        L += ["Figures (artifact): " + ", ".join(f"`{f}`" for f in rec.get("figures", []) + b.get("figures", [])), ""]
    (run_dir / "report.md").write_text("\n".join(L) + "\n")
    save_json(run_dir / "report.json", rec)


_energy = {}


def gpu_stats():
    """'GPU 2850 MHz 71 °C 152 W' for the Intel card (sysfs; power from the
    energy counter between two calls)."""
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        try:
            if (card / "device/vendor").read_text().strip() != "0x8086":
                continue
        except OSError:
            continue
        parts = []
        with contextlib.suppress(OSError, ValueError):
            parts.append(f"{int((card / 'device/tile0/gt0/freq0/act_freq').read_text())} MHz")
        hw = next(iter(sorted((card / "device/hwmon").glob("hwmon*"))), None)
        if hw:
            for lab in hw.glob("temp*_label"):
                if lab.read_text().strip() == "pkg":
                    with contextlib.suppress(OSError, ValueError):
                        parts.append(f"{int(lab.with_name(lab.name.replace('_label', '_input')).read_text()) // 1000} °C")
            with contextlib.suppress(OSError, ValueError):
                e, t = int((hw / "energy1_input").read_text()), time.monotonic()
                if "e" in _energy and t > _energy["t"]:
                    parts.append(f"{(e - _energy['e']) / 1e6 / (t - _energy['t']):.0f} W")
                _energy.update(e=e, t=t)
        return "GPU " + " ".join(parts) if parts else ""
    return ""


def bench_progress(stt):
    """Progress of a benchmark session from its bench.csv and plan: % of the
    planned work (weighted by each window's run time), runs done, the
    latest run (arm, window, speed) and an ETA."""
    plan = stt.get("bench_plan") or []
    try:
        rows = csv_rows(stt["bench_csv"])
    except (OSError, KeyError):
        rows = []
    total = sum(p[2] for p in plan)
    if not plan:
        return ""
    done_w = {}
    for r in rows:
        done_w.setdefault(r.get("window"), []).append(r)
    cost = {}
    for w, span, _, _ in plan:      # seconds per run: measured, else ~58 G evens/s + 0.3 s start
        ws = [float(r["wall_s"]) for r in done_w.get(w, []) if r.get("wall_s")]
        cost[w] = st.median(ws) if ws else span / 2 / 58e9 + 0.3
    # warm-up runs (one per arm and window) come first and are not in the csv
    tot = sum((n + k) * cost[w] for w, _, n, k in plan)
    done = sum((len(done_w.get(w, [])) + (k if done_w.get(w) else 0)) * cost[w] for w, _, n, k in plan)
    el = time.time() - stt.get("phase_since", time.time())
    fr = min(done / tot, 1.0) if tot else 0
    eta = el * (1 - fr) / fr if fr > 0.02 else (tot - done)
    line = f"{100 * fr:.0f}% ({len(rows)}/{total} runs)"
    if rows:
        r = rows[-1]
        with contextlib.suppress(ValueError, KeyError):
            line += f" | now {r['label']} on {r['window']}: {float(r['evens_per_sec']) / 1e9:.2f} G ev/s"
    return line + f" | ETA {hms(eta)}"


def progress_line():
    """One line: what the CI is doing right now (status, Actions heartbeat)."""
    stt = load_json(STATE / "state.json", {})
    ph = stt.get("phase", "idle")
    if ph == "idle":
        return "idle"
    line = f"{stt.get('sha', '')[:10]} {ph} for {hms(time.time() - stt.get('phase_since', time.time()))}"
    if ph == "stop_production":
        lv = prod_live(prod_conf())
        if lv and lv.get("elapsed_s"):
            fr = lv["evens_done"] / max(lv["evens_total"], 1)
            eta = lv["elapsed_s"] * (1 - fr) / fr if fr else 0
            line += f": production slice {100 * fr:.1f}% done, boundary in ~{hms(eta)}"
    elif ph in ("slice", "slice_reference") and stt.get("live"):
        lv = load_json(stt["live"], {})
        if lv.get("evens_total") and lv.get("elapsed_s"):
            fr = lv["evens_done"] / lv["evens_total"]
            r = lv["evens_done"] / lv["elapsed_s"]
            line += (f": {'reference' if ph == 'slice_reference' else 'candidate'} window {100 * fr:.1f}% "
                     f"| {r / 1e9:.2f} G ev/s | ETA {hms((lv['evens_total'] - lv['evens_done']) / r if r else 0)}")
    elif ph == "benchmark":
        line += ": " + bench_progress(stt)
    if ph in ("slice", "slice_reference", "benchmark", "gpu_validate", "bench"):
        g = gpu_stats()
        line += f" | {g}" if g else ""
    return line


@contextlib.contextmanager
def heartbeat(every=60):
    """Print progress_line() every `every` s (keeps an Actions log alive)."""
    stop = threading.Event()

    def beat():
        while not stop.wait(every):
            print(f"{now_iso()} … {progress_line()}", flush=True)

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()


def annotate(rec):
    """GitHub Actions annotations for failed / warned checks."""
    if not ON_ACTIONS:
        return
    for x in rec.get("checks", []):
        if x["level"] in ("fail", "warn"):
            kind = "error" if x["level"] == "fail" else "warning"
            print(f"::{kind} title=CI {x['name']}::{x['detail']}", flush=True)
    print(f"::notice title=CI {rec['status']}::{rec.get('reason') or rec.get('subject', '')}", flush=True)
