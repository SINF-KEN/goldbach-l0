"""Production's journal (run_range.sh's PASS / RETRY rows): parsing, the
resume chain, audits, slice timing; its goldbach_stats.log lines."""
import contextlib
import datetime as dt
import statistics as st
import sys
from pathlib import Path

from .core import kv


def read_journal(path):
    rows = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return rows
    for ln in lines:
        t = ln.split()
        if len(t) >= 4 and t[0] in ("PASS", "RETRY"):
            if not (t[1].isdigit() and t[2].isdigit()):   # e.g. a row cut off by a crash
                print(f"warning: malformed journal row ignored: {ln[:120]}", file=sys.stderr)
                continue
            rows.append({"tag": t[0], "a": int(t[1]), "b": int(t[2]), "ts": t[3],
                         "kv": kv(" ".join(t[4:])), "line": ln})
    return rows


def chain_next(rows, start):
    """run_range.sh's resume rule: follow the PASS chain from START.

    A row extends the chain only if its own verifier output agrees with its
    slice fields (start= / limit= of the --csv line): a damaged b would
    otherwise skip unverified evens. Disagreement raises (fail closed)."""
    nxt = start
    for r in rows:
        if r["tag"] == "PASS" and r["a"] == nxt:
            if r["kv"].get("start") != str(r["a"]) or r["kv"].get("limit") != str(r["b"]):
                raise RuntimeError(f"journal PASS row for [{r['a']}, {r['b']}] disagrees with its own "
                                   f"--csv start=/limit= (damaged journal?): {r['line'][:160]}")
            nxt = r["b"] + 2
    return nxt


def journal_audit(rows, start):
    """(resume point, duplicate PASS rows) of the journal."""
    seen, dups = set(), []
    for r in rows:
        if r["tag"] == "PASS":
            k = (r["a"], r["b"])
            if k in seen:
                dups.append(k)
            seen.add(k)
    return chain_next(rows, start), dups


def median_slice_wall(rows, n=20):
    ts = [dt.datetime.fromisoformat(r["ts"]) for r in rows if r["tag"] == "PASS"]
    d = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    d = [x for x in d[-n:] if 60 < x < 6 * 3600]
    return st.median(d) if d else 1800.0


def stats_line(path, a, b):
    """Last GBSTATS line of a goldbach_stats.log for exactly [a, b]."""
    hit = None
    with contextlib.suppress(OSError):
        with open(path) as f:
            for ln in f:
                if ln.startswith("GBSTATS") and f" start={a} " in ln and f" limit={b} " in ln:
                    hit = ln.strip()
    return hit
