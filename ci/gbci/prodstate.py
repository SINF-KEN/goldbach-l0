"""Production as it is: production.json (the run_range invocation), its
journal / log / stop / statistics / live files, the run_range and
goldbach_l0 processes. Read-only views plus save_prod_conf."""
import os
import subprocess
import time
from pathlib import Path

from .core import ROOT, STATE, UNIT, load_json, save_json


def proc_cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]
    except OSError:
        return []


def find_run_range():
    """Top-level run_range.sh processes: [{pid, pgid, args, unit}]."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        cl = [c.decode(errors="replace") for c in proc_cmdline(d.name)]
        if len(cl) < 2 or Path(cl[0]).name not in ("bash", "sh") or not cl[1].endswith("run_range.sh"):
            continue
        try:
            ppid = int(Path(f"/proc/{d.name}/stat").read_text().rsplit(")", 1)[1].split()[1])
            pgid = os.getpgid(int(d.name))
        except (OSError, IndexError, ValueError):
            continue
        pcl = proc_cmdline(ppid)
        if len(pcl) >= 2 and pcl[1].decode(errors="replace").endswith("run_range.sh"):
            continue                       # a $(...) subshell of another instance
        try:
            cg = Path(f"/proc/{d.name}/cgroup").read_text()
        except OSError:
            cg = ""
        out.append({"pid": int(d.name), "pgid": pgid, "args": cl[2:],
                    "unit": UNIT if f"/{UNIT}.service" in cg else ""})
    return out


def pgrep(name):
    p = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True)
    return [int(x) for x in p.stdout.split()]


def prod_conf():
    return load_json(STATE / "production.json", {})


def save_prod_conf(pc):
    save_json(STATE / "production.json", pc, mode=0o600)     # env may hold secrets


def prod_paths(pc):
    j = ROOT / pc["journal"]
    base = str(j)[: -len(".journal")] if str(j).endswith(".journal") else str(j)
    return j, Path(base + ".log"), Path(base + ".stop")


def prod_stats_log(pc):
    """Production's goldbach_stats.log (GB_STATS_LOG, relative to ROOT where
    run_range runs)."""
    return ROOT / pc.get("env", {}).get("GB_STATS_LOG", "goldbach_stats.log")


def prod_live(pc):
    """Production's goldbach_live.json if it is fresh (< 2 min old)."""
    p = Path(pc.get("env", {}).get("GB_STATS_LIVE", ROOT / "goldbach_live.json"))
    lv = load_json(p, {})
    return lv if lv and time.time() - lv.get("time", 0) < 120 else None
