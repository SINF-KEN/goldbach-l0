#!/usr/bin/env python3
"""goldbach_live.py — live terminal dashboard for a dense run_range.sh campaign.

Run it in a second terminal (or tmux pane) next to the campaign:

    bash tools/run_range.sh 4000000000000000000 5000000000000000000     # terminal 1
    python3 scripts/goldbach_live.py                                    # terminal 2

Read-only. Sources (all in the run directory, written by goldbach_l0 v2.2+):
  range-*.journal        banked slices (run_range.sh)
  goldbach_live.json     current run, rewritten every ~2 s
  goldbach_stats.state   campaign totals (updated when a slice passes)
  goldbach_ridges.csv    top-30 ridges (updated every segment)
  goldbach_records.txt   p(n) > 9781 / new S(p) events, CPU re-proven
  /sys/class/drm         GPU clock, temperature, power
  ci-state/state.json    local CI run in progress (ci/goldbach_ci.py): phase,
                         commit, and its GPU slice's own live file

The record to beat (T. Oliveira e Silva, verification to 4e18): no even
n <= 4e18 needs a smallest Goldbach prime above 9781, first needed by
n = 3325581707333960528; S(p), the least even with p(n) = p, is unknown
(> 4e18) for 105 primes 8423..9769 — the first one found by a campaign that
covers [4e18, n] without gaps is a new table entry.
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import sys
import textwrap
import time

REC_P, REC_N = 9781, 3325581707333960528
START0 = 4000000000000000000
UNKNOWN = [8423, 8429, 8537, 8627, 8669, 8713, 8741, 8753, 8783, 8803, 8807, 8837, 8839, 8849, 8861,
           8863, 8867, 8887, 8893, 8923, 8963, 8969, 8999, 9007, 9011, 9013, 9029, 9041, 9043, 9049,
           9059, 9067, 9103, 9109, 9127, 9133, 9137, 9151, 9157, 9173, 9181, 9187, 9199, 9209, 9221,
           9227, 9239, 9241, 9257, 9277, 9281, 9283, 9293, 9311, 9319, 9323, 9337, 9343, 9349, 9371,
           9377, 9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, 9461, 9463, 9467, 9473,
           9479, 9491, 9497, 9511, 9521, 9533, 9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623, 9631,
           9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733, 9739, 9743, 9749, 9767, 9769]

B, D, R, Y, G, C, M = "\033[1m", "\033[2m", "\033[0m", "\033[33m", "\033[32m", "\033[36m", "\033[35m"


def hms(s):
    s = int(max(s, 0))
    d, s = divmod(s, 86400)
    return (f"{d}d " if d else "") + f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def sci(x):
    return f"{x:.4e}" if x >= 1e6 else f"{x:,}"


def bar(frac, w=40):
    f = max(0.0, min(1.0, frac)) * w
    full = int(f)
    part = " ▏▎▍▌▋▊▉"[int((f - full) * 8)] if full < w else ""
    return "█" * full + part + " " * (w - full - len(part))


def read_journal(path):
    rows = []
    for line in open(path):
        t = line.split()
        if len(t) >= 4 and t[0] == "PASS":
            d = dict(x.split("=", 1) for x in t[4:] if "=" in x)
            rows.append((int(t[1]), int(t[2]), t[3], d))
    return rows


def read_state(path):
    st = {"range": [], "first": {}, "beyond": [], "primes": 0, "twins": 0, "tail": 0}
    if not os.path.exists(path):
        return st
    for line in open(path):
        k, _, rest = line.rstrip("\n").partition(" ")
        if k == "range":
            st["range"].append(tuple(map(int, rest.split())))
        elif k == "first":
            p, n = rest.split()
            st["first"][int(p)] = int(n)
        elif k == "beyond":
            p, n = rest.split()
            st["beyond"].append((int(p), int(n)))
        elif k == "primes30":
            st["primes"] = sum(map(int, rest.split()))
        elif k == "twins30":
            st["twins"] = sum(map(int, rest.split()))
        elif k in ("tailcls", "tailagg"):
            lst = rest.partition(" ")[2] if k == "tailcls" else rest
            st["tail"] += sum(int(x) for x in lst.split(",") if x)
    return st


class Gpu:
    def __init__(self):
        self.card = None
        for c in sorted(glob.glob("/sys/class/drm/card?")):
            try:
                if open(c + "/device/vendor").read().strip() == "0x8086":
                    self.card = c
                    break
            except OSError:
                pass
        self.hw = (glob.glob(self.card + "/device/hwmon/hwmon*") or [None])[0] if self.card else None
        self.e0 = self.t0 = None

    def _rd(self, p):
        try:
            return open(p).read().strip()
        except (OSError, TypeError):
            return None

    def line(self):
        if not self.card:
            return "no Intel GPU found"
        f = self._rd(self.card + "/device/tile0/gt0/freq0/act_freq")
        temp = watt = None
        if self.hw:
            for lab in glob.glob(self.hw + "/temp*_label"):
                if "pkg" in (self._rd(lab) or "").lower():
                    v = self._rd(lab.replace("_label", "_input"))
                    temp = int(v) / 1000 if v else None
            e = None
            for lab in glob.glob(self.hw + "/energy*_label"):
                if "card" in (self._rd(lab) or "").lower():
                    e = self._rd(lab.replace("_label", "_input"))
            e = e or self._rd(self.hw + "/energy1_input")
            now = time.time()
            if e and self.e0 is not None and now > self.t0:
                watt = (int(e) - self.e0) / 1e6 / (now - self.t0)
            if e:
                self.e0, self.t0 = int(e), now
        return (f"{f or '?'} MHz | {temp:.0f} C" if temp is not None else f"{f or '?'} MHz | ? C") + \
               (f" | {watt:.0f} W" if watt is not None and watt > 0 else "")


def slice_time(t):
    try:
        return datetime.datetime.fromisoformat(t).timestamp()
    except ValueError:
        return 0.0


def newest_ridge(rid, rows, live):
    """(n, p, found, exact) of the newest ridge entry, biggest ridge among ties.

    found is the store's found_unix column (v2). Rows without it (v1 store,
    or 0) are dated by the latest slice covering their n (journal row, or the
    live run in progress) — a guess that a re-run of the range moves.
    """
    def exact(r):
        return len(r) > 5 and int(r[5]) > 0

    def when(r):
        if exact(r):
            return float(r[5])
        n = int(r[1])
        t = max((slice_time(ts) for lo, hi, ts, _ in rows if lo <= n <= hi), default=0.0)
        if live and live["start"] <= n <= live["limit"]:
            t = max(t, live["time"])
        return t
    if not rid:
        return None
    r = max(rid, key=lambda r: (when(r), int(r[2])))
    return int(r[1]), int(r[2]), when(r), exact(r)


def covered_to(ranges, extra=None):
    """Largest x such that [START0, x] is covered without a gap (else None)."""
    rs = sorted(ranges + ([extra] if extra else []))
    hi = None
    for a, b in rs:
        if hi is None:
            if a > START0:
                return None
            hi = b
        elif a <= hi + 2:
            hi = max(hi, b)
        else:
            break
    return hi


def ci_panel(now):
    """Lines for a local CI run in progress (ci-state/state.json), with the
    progress of its GPU slice (the slice's isolated goldbach_live.json)."""
    try:
        stt = json.load(open("ci-state/state.json"))
    except (OSError, ValueError):
        return [], None
    ph = stt.get("phase", "idle")
    if ph == "idle":
        return [], None
    since = now - stt.get("phase_since", now)
    total = now - stt.get("started", now)
    ref = ",".join(stt.get("refs", []))
    out = [f"{M}{B}CI{R}       {stt.get('sha', '')[:10]} ({ref}) {D}{stt.get('subject', '')[:60]}{R}",
           f"         phase {B}{ph}{R}" + (f" for {hms(since)}, run {hms(total)}" if stt.get("started") else "")]
    if ph == "benchmark":
        try:     # same progress line as the GitHub job log (ci/goldbach_ci.py)
            sys.path.insert(0, "ci")
            import goldbach_ci
            out.append(f"         {goldbach_ci.bench_progress(stt)}")
        except Exception:
            pass
    live_path = stt.get("live") or f"ci-state/runs/{stt.get('run_id')}/slice-files/live.json"
    if ph in ("slice", "slice_reference"):
        try:
            lv = json.load(open(live_path))
            fr = lv["evens_done"] / max(lv["evens_total"], 1)
            r = lv["evens_done"] / lv["elapsed_s"] if lv.get("elapsed_s") else 0
            eta = (lv["evens_total"] - lv["evens_done"]) / r if r else 0
            who = "active release (reference)" if ph == "slice_reference" else "candidate"
            out.append(f"         {who} [{lv['start']:,}, {lv['limit']:,}]")
            out.append(f"         slice {bar(fr, 20)} {100 * fr:5.1f}%  {r / 1e9:6.2f} G ev/s  ETA {hms(eta)}")
        except (OSError, ValueError, KeyError):
            pass
    return out, ph


def main():
    ap = argparse.ArgumentParser(description="live dashboard for a goldbach-l0 campaign")
    ap.add_argument("--dir", default=".", help="run directory (default: cwd)")
    ap.add_argument("--journal", help="campaign journal (default: newest range-*.journal)")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    a = ap.parse_args()
    os.chdir(a.dir)
    gpu = Gpu()
    prev = None
    seen_beat = 0
    while True:
        jpath = a.journal or max(glob.glob("range-*.journal"), key=os.path.getmtime, default=None)
        out = []
        now = time.time()
        out.append(f"{B}goldbach-l0 live{R}  {time.strftime('%Y-%m-%d %H:%M:%S')}   {D}(Ctrl-C quits; read-only){R}")
        S = L = None
        rows = []
        if jpath:
            m = re.match(r"range-(\d+)-(\d+)\.journal", os.path.basename(jpath))
            if m:
                S, L = int(m.group(1)), int(m.group(2))
            rows = read_journal(jpath)
        live = None
        try:
            live = json.load(open("goldbach_live.json"))
        except (OSError, ValueError):
            pass
        live_fresh = live is not None and now - live.get("time", 0) < 120
        ci_lines, ci_ph = ci_panel(now)
        st = read_state("goldbach_stats.state")

        # ---------------- progress / speed / ETA ----------------
        if S is not None:
            total = (L - S) // 2 + 1
            banked = 0
            nxt = S
            for a_, b_, _, _ in rows:
                if a_ == nxt:
                    nxt = b_ + 2
            banked = (nxt - S) // 2
            cur = 0
            slice_line = f"{D}no run active{R}"
            if ci_ph and ci_ph not in ("build",):
                slice_line = f"{Y}paused by the local CI ({ci_ph}){R}"
            rate_now = None
            if live_fresh and live["start"] == nxt:
                cur = live["evens_done"]
                fr = cur / max(live["evens_total"], 1)
                if prev and prev[0] == live["start"] and live["time"] > prev[1] and cur > prev[2]:
                    rate_now = (cur - prev[2]) / (live["time"] - prev[1])
                if not prev or live["time"] != prev[1]:
                    prev = (live["start"], live["time"], cur)
                r_run = cur / live["elapsed_s"] if live["elapsed_s"] > 0 else 0
                eta_s = (live["evens_total"] - cur) / r_run if r_run else 0
                slice_line = f"slice {bar(fr, 20)} {100 * fr:5.1f}%  ETA {hms(eta_s)}"
                if ci_ph == "stop_production":
                    slice_line += f"  {Y}(the local CI pauses production after this slice){R}"
            done = banked + cur
            rates = [float(d.get("evens_per_sec", 0)) for *_, d in rows[-20:] if d.get("evens_per_sec")]
            mean = sum(rates) / len(rates) if rates else 0
            # wall-clock mean over the banked slices (includes gaps between runs)
            r = rate_now or mean or 0
            eta = (total - done) / r if r else 0
            out.append(f"range    [{S:,}, {L:,}]  ({sci(total)} evens)")
            # slices left (the running one included): slice size of the
            # latest banked row (it can change mid-campaign), else 2e14
            span = rows[-1][1] - rows[-1][0] + 2 if rows else 200000000000000
            left = -(-(L + 2 - nxt) // span) if nxt <= L else 0
            out.append(f"campaign {G}{bar(done / total)}{R} {100 * done / total:8.4f}%   "
                       f"{len(rows)} slices banked, {left} left")
            out.append(f"         {slice_line}")
            out.append(f"speed    now {(rate_now or 0) / 1e9:6.2f} G ev/s | last slices {mean / 1e9:6.2f} G ev/s | "
                       f"campaign ETA {B}{hms(eta)}{R}")
        else:
            out.append(f"{Y}no range-*.journal here — start: bash tools/run_range.sh START LIMIT{R}")
        out.append(f"gpu      {gpu.line()}")
        if ci_lines:
            out += ci_lines

        # ---------------- the record hunt ----------------
        ranges = st["range"]
        firsts = dict(st["first"])
        live_cov = None
        if live_fresh and not any(lo <= live["start"] <= hi for lo, hi in ranges):
            live_cov = (live["start"], live["start"] + 2 * max(live["evens_done"] - 1, 0))
        cov = covered_to(ranges, live_cov)
        pmax, pmax_n = 0, 0
        for p, n in firsts.items():
            if p > pmax:
                pmax, pmax_n = p, n
        for p, n in st["beyond"]:
            if p > pmax:
                pmax, pmax_n = p, n
        if live_fresh and live.get("pmax", 0) > pmax:
            pmax, pmax_n = live["pmax"], int(live["pmax_n"])
        events = []
        if os.path.exists("goldbach_records.txt"):
            for line in open("goldbach_records.txt"):
                m = re.search(r"p\(n\)=(\d+) n=(\d+) .*cpu_check=(\w+)", line)
                if m:
                    events.append((int(m.group(1)), int(m.group(2)), m.group(3), line.strip()))
        beat = [e for e in events if e[0] > REC_P and e[2] == "OK"]
        out.append("")
        out.append(f"{B}record   Oliveira e Silva (to 4e18): largest p(n) = {REC_P} at n = {REC_N}{R}")
        if beat:
            p, n, _, _ = max(beat)
            out.append(f"         {M}{B}*** BEATEN: p(n) = {p} at n = {n} (CPU re-proven) ***{R}")
            if len(beat) > seen_beat:
                out[-1] += "\a"                       # terminal bell once per new record
            seen_beat = len(beat)
        col = G if pmax >= 9000 else Y if pmax >= 8000 else ""
        out.append(f"         campaign largest p(n): {col}{B}{pmax}{R} at n = {pmax_n}" if pmax else
                   "         campaign largest p(n): (no statistics yet)")
        ge = (live or {}) if live_fresh else {}
        out.append(f"         this run: p(n) >= 8000: {ge.get('ge8000', 0)}   >= 9000: {ge.get('ge9000', 0)}   "
                   f"> 9781: {ge.get('gt9781', 0)}")
        found = {}
        for p in UNKNOWN:
            n = firsts.get(p)
            for ep, en, ok, _ in events:
                if ep == p and ok == "OK" and (n is None or en < n):
                    n = en
            if n:
                found[p] = n
        proven = {p: n for p, n in found.items() if cov is not None and n <= cov}
        out.append(f"S(p)     unknown below 4e18 for 105 primes 8423..9769; found here: "
                   f"{B}{len(found)}{R} ({len(proven)} proven: range covered from 4e18 to n)")
        if cov is None and found:
            out.append(f"         {Y}coverage does not start at 4e18 (re-run the missing slices) — "
                       f"S(p) claims need [4e18, n] covered{R}")
        for p in sorted(found)[-6:]:
            out.append(f"         S({p}) = {found[p]}" + ("" if p in proven else f"  {D}(not yet proven){R}"))

        # ---------------- science counters ----------------
        pr = st["primes"] + (live["primes"] if live_cov else 0)
        tw = st["twins"] + (live["twins"] if live_cov else 0)
        tl = st["tail"] + (live["tail_evens"] if live_cov else 0)
        out.append(f"science  primes {sci(pr)} | twin pairs {sci(tw)} | evens with p(n) >= 3000: {tl:,}"
                   f"  {D}(python3 scripts/gstats.py){R}")

        # ---------------- top 30 ridges ----------------
        out.append("")
        if os.path.exists("goldbach_ridges.csv"):
            age = now - os.path.getmtime("goldbach_ridges.csv")
            out.append(f"{B}My Top 30 Largest Goldbach Ridges{R}  {D}(goldbach_ridges.csv, updated {hms(age)} ago){R}")
            rid = [l.strip().split(",") for l in open("goldbach_ridges.csv") if l[0].isdigit()]
            newest = newest_ridge(rid, rows, live if live_fresh else None)
            half = (len(rid) + 1) // 2
            for i in range(half):
                cells = []
                for r_ in (rid[i], rid[i + half] if i + half < len(rid) else None):
                    if r_:
                        rk, n, p, q = r_[0], r_[1], r_[2], r_[3]
                        c = M if int(p) > REC_P else G if int(p) >= 9000 else Y if int(p) >= 8000 else ""
                        if newest and (int(n), int(p)) == newest[:2]:
                            cells.append(f"{int(rk):3d}.{C}{B}*{int(p):5d} {n}{R}")
                        else:
                            cells.append(f"{int(rk):3d}. {c}{int(p):5d}{R} {n}")
                out.append("  " + "    ".join(cells))
            if newest:
                when = (("" if newest[3] else "~") +
                        time.strftime("%Y-%m-%d %H:%M", time.localtime(newest[2]))
                        if newest[2] else "unknown") + ("" if newest[3] else " (estimated: v1 row)")
                out.append(f"  {C}{B}*{R} {D}newest: ridge {newest[1]} at n = {newest[0]}, "
                           f"found {when}{R}")
        else:
            out.append(f"{D}no ridge store yet{R}")
        if events:
            out.append("")
            out.append(f"{B}events{R} {D}(goldbach_records.txt){R}")
            # wrap, never cut: the S(p) note is the point of the line
            w = max(shutil.get_terminal_size().columns, 60)
            for e in events[-5:]:
                out += textwrap.wrap(e[3], w, initial_indent="  ", subsequent_indent="      ")
        sys.stdout.write("\033[H\033[2J" + "\n".join(out) + "\n")
        sys.stdout.flush()
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
