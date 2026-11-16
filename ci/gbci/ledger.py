"""The CI ledger (ledger.jsonl + results.csv), the retry queue and the
detection of new branch tips."""
import csv
import fnmatch
import json
import subprocess

from .core import (FINAL, RESULT_COLS, ROOT, STATE, TIMING_KEYS, cfg, csv_header, git, is_ancestor, load_json,
                   read_lines, save_json, stamp)
from .releases import active_release, release_meta


def ledger_append(rec):
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "ledger.jsonl", "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    sl = rec.get("slice", {})
    flat = {"run_id": rec["run_id"], "ts_start": rec.get("ts_start", ""), "ts_end": rec.get("ts_end", ""),
            "status": rec["status"], "reason": rec.get("reason", ""), "refs": " ".join(rec.get("refs", [])),
            "commit": rec.get("commit", ""), "version": rec.get("version", ""), "release": rec.get("release", ""),
            "previous_active": rec.get("previous_active", ""), "segment": rec.get("config", {}).get("segment", ""),
            "window_start": sl.get("start", ""), "window_limit": sl.get("limit", ""),
            "reference": sl.get("ref_source", ""), "slice_wall_s": sl.get("wall_s", ""),
            "checks_failed": ";".join(x["name"] for x in rec.get("checks", []) if x["level"] == "fail"),
            "checks_warned": ";".join(x["name"] for x in rec.get("checks", []) if x["level"] == "warn"),
            "downtime_s": rec.get("downtime_s", ""), "lost_s": rec.get("production", {}).get("lost_s", "")}
    for k in TIMING_KEYS:
        flat["slice_" + k] = sl.get("csv", {}).get(k, "")
    for w, d in rec.get("bench", {}).get("windows", {}).items():
        flat[f"bench_{w}_speed_ratio"] = d.get("speed_ratio", "")
    p = STATE / "results.csv"
    if p.exists() and csv_header(p) != RESULT_COLS:
        p.rename(p.with_name(f"results-{stamp()}.csv"))      # older layout: keep it aside
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(flat)


def ledger():
    try:
        return [json.loads(ln) for ln in read_lines(STATE / "ledger.jsonl")]
    except OSError:
        return []


def list_refs(c):
    pats = list(c["watch_include"]) + (["refs/remotes/origin/*"] if c["fetch"] else [])
    refs = []
    for ln in git("for-each-ref", "--format=%(refname) %(objectname) %(committerdate:unix)").splitlines():
        name, sha, ts = ln.split()
        if not name.endswith("/HEAD") and any(fnmatch.fnmatch(name, p) for p in pats) \
                and not any(fnmatch.fnmatch(name, p) for p in c["watch_exclude"]):
            refs.append((name, sha, int(ts)))
    return refs


def ledger_index():
    """{sha: (interrupted attempts, judged)} from one read of the ledger."""
    idx = {}
    for r in ledger():
        n, j = idx.get(r.get("commit"), (0, False))
        idx[r.get("commit")] = (n + (r["status"] == "INTERRUPTED"), j or r["status"] in FINAL)
    return idx


def attempts(sha, idx=None):
    return (ledger_index() if idx is None else idx).get(sha, (0, False))[0]


def judged(sha, idx=None):
    return (ledger_index() if idx is None else idx).get(sha, (0, False))[1]


def candidates(update_seen=True):
    """New (sha, [refs]) tips since the last poll, oldest commit first."""
    c = cfg()
    if c["fetch"]:
        subprocess.run(["git", "-C", str(ROOT), "fetch", "--quiet", "origin"], capture_output=True)
    seen = load_json(STATE / "refs.json", {})
    refs = list_refs(c)
    act = release_meta(active_release()).get("commit")
    idx = ledger_index()          # once per poll, not once per ref
    new = {}
    for name, sha, ts in refs:
        if seen.get(name) == sha:
            continue
        if judged(sha, idx) or attempts(sha, idx) >= c["max_attempts"] or (act and is_ancestor(sha, act)):
            seen[name] = sha
            continue
        new.setdefault(sha, {"refs": [], "ts": ts})["refs"].append(name.replace("refs/heads/", ""))
    seen = {k: v for k, v in seen.items() if k in {n for n, _, _ in refs}}
    if update_seen:
        save_json(STATE / "refs.json", seen)
    depth = lambda s: int(git("rev-list", "--count", s))     # ancestors first on equal dates
    return sorted(((s, d["refs"]) for s, d in new.items()), key=lambda x: (new[x[0]]["ts"], depth(x[0])))


def mark_seen(sha):
    seen = load_json(STATE / "refs.json", {})
    for name, s, _ in list_refs(cfg()):
        if s == sha:
            seen[name] = sha
    save_json(STATE / "refs.json", seen)


def queue_edit(fn):
    q = load_json(STATE / "queue.json", [])
    q = fn(q)
    save_json(STATE / "queue.json", q)
    return q
