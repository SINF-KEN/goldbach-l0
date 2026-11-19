"""Building a commit in the CI worktree (against a clean goldbach-tools
worktree), kernel freshness, CPU tests, staging the release."""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .core import ROOT, STATE, cfg, git, log, now_iso, save_json, sh
from .releases import active_release, fingerprint, link_target, release_dir, release_meta


def tools_worktree(logf):
    """goldbach-tools for CI builds: a clean detached worktree at the tools
    repo's committed HEAD (the worktree's tools -> ../goldbach-tools resolves
    here), never the live working tree, so a verdict depends on commits only.
    -> the tools commit used."""
    live = (ROOT / "tools").resolve()
    tl = STATE / "work" / "goldbach-tools"
    head = subprocess.run(["git", "-C", str(live), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(live), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if tl.is_symlink():                         # layout before v2.3.0: a link to the live tree
        tl.unlink()
    if not (tl / ".git").exists():
        shutil.rmtree(tl, ignore_errors=True)
        sh(["git", "-C", live, "worktree", "prune"], logf=logf)
        sh(["git", "-C", live, "worktree", "add", "--detach", tl, head], logf=logf, check=True)
    else:
        sh(["git", "-C", tl, "checkout", "--detach", "-f", head], logf=logf, check=True)
    sh(["git", "-C", tl, "clean", "-ffdxq"], logf=logf, check=True)
    log(f"tools: goldbach-tools {head[:12]}" +
        (" (the live tools tree has uncommitted changes: not used)" if dirty else ""))
    return head


def worktree(sha, logf):
    wt = STATE / "work" / "src"
    (STATE / "work").mkdir(parents=True, exist_ok=True)
    tools_worktree(logf)
    if not (wt / ".git").exists():
        shutil.rmtree(wt, ignore_errors=True)
        git("worktree", "prune")
        sh(["git", "-C", ROOT, "worktree", "add", "--detach", wt, sha], logf=logf, check=True)
    else:
        sh(["git", "-C", wt, "checkout", "--detach", "-f", sha], logf=logf, check=True)
    sh(["git", "-C", wt, "clean", "-ffdxq"], logf=logf, check=True)
    for f in list(ROOT.glob("base_primes*.cache*")) + list(ROOT.glob("scm_plan.*.bin")):
        if not (wt / f.name).exists():
            os.symlink(f, wt / f.name)
    return wt


def kernel_freshness(wt, logf):
    """Every shipped kernels/<label>_bmg.spv must equal a fresh ocloc compile
    (the GitHub `kernels` gate, here when ocloc is the pinned version)."""
    ci = (wt / ".github/workflows/ci.yml").read_text()
    pinned = re.search(r"CR_VER:\s*([\d.]+)", ci)
    mods = re.search(r"KERNEL_MODULES: \|\n((?: {4}\S.*\n)+)", ci)
    oc = shutil.which("ocloc")
    if not oc or not pinned or not mods:
        return "warn", "ocloc or the module list not available: kernel freshness not checked"
    ver = subprocess.run([oc, "--version"], capture_output=True, text=True).stdout
    if pinned.group(1).rsplit(".", 1)[0] not in ver:
        return "warn", f"local ocloc ({ver.strip()[:40]}) != pinned {pinned.group(1)}: not checked"
    bad = []
    with tempfile.TemporaryDirectory() as td:
        for ln in mods.group(1).splitlines():
            label, src, opts = (ln.strip().split("|") + ["", ""])[:3]
            cmd = ["nice", "-n", "15", oc, "compile", "-file", wt / src, "-device", "bmg", "-output", f"{td}/{label}"]
            rc, _ = sh(cmd + ["-options", f"-I {(wt / src).parent} {opts}".strip()], cwd=td, logf=logf)
            fresh, shipped = Path(td) / f"{label}_bmg.spv", wt / "kernels" / f"{label}_bmg.spv"
            if rc != 0 or not fresh.exists() or not shipped.exists() or fresh.read_bytes() != shipped.read_bytes():
                bad.append(label)
    return ("fail", f"stale or broken kernel modules: {bad}") if bad else ("ok", "all shipped .spv fresh")


def build_release(sha, run_dir, refs, cpu_tests=True):
    """Build sha in the CI worktree (+ kernel freshness and ctest -LE gpu
    unless the caller already ran them) and stage it as a release.
    -> (worktree, release dir or None, checks)."""
    c = cfg()
    logf = run_dir / "build.log"
    wt = worktree(sha, logf)
    checks = []
    cm = ["cmake", "-S", wt, "-B", wt / "build", "-DCMAKE_BUILD_TYPE=Release"] + (["-DGB_WERROR=ON"] if c["werror"] else [])
    rc, _ = sh(["nice", "-n", "15", *cm], logf=logf)
    if rc == 0:
        rc, _ = sh(["nice", "-n", "15", "cmake", "--build", wt / "build", "-j", str(c["build_jobs"])],
                   logf=logf, timeout=3600)
    checks.append({"name": "build", "level": "ok" if rc == 0 else "fail",
                   "detail": "cmake Release" + (" -Werror" if c["werror"] else "") + f" rc={rc}"})
    if rc != 0:
        return wt, None, checks
    if cpu_tests:
        lvl, det = kernel_freshness(wt, run_dir / "kernels.log")
        checks.append({"name": "kernels_fresh", "level": lvl, "detail": det})
        rc, out = sh(["nice", "-n", "15", "ctest", "--test-dir", wt / "build", "-LE", "gpu",
                      "--output-on-failure", "--timeout", "900"], logf=run_dir / "ctest.log", timeout=3600)
        summ = next((ln for ln in out.splitlines() if "tests passed" in ln), out.strip()[-200:])
        checks.append({"name": "ctest_cpu", "level": "ok" if rc == 0 else "fail", "detail": summ.strip()})
        if any(x["level"] == "fail" for x in checks):
            return wt, None, checks
    else:
        checks.append({"name": "cpu_tests", "level": "ok",
                       "detail": "kernel freshness + ctest ran in the GitHub jobs kernels / host"})
    rid = sha[:12]
    rel = release_dir(rid)
    stage = rel.with_name(rid + ".staging")
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "kernels").mkdir(parents=True)
    for f in ("goldbach_l0", "stats_ref"):
        if (wt / "build" / f).exists():
            shutil.copy2(wt / "build" / f, stage / f)
    for k in (wt / "kernels").glob("*_bmg.spv"):
        shutil.copy2(k, stage / "kernels" / k.name)
    (stage / "RELEASE").write_text(rid + "\n")
    version = re.search(r'VERSION = "([^"]+)"', (wt / "include/config.hpp").read_text())
    meta = {"id": rid, "commit": sha, "refs": refs, "version": version.group(1) if version else "?",
            "subject": git("log", "-1", "--format=%s", sha), "built": now_iso(), "fingerprint": fingerprint(stage)}
    save_json(stage / "meta.json", meta)
    if rel.exists():
        if release_meta(rel).get("fingerprint") == meta["fingerprint"]:
            shutil.rmtree(stage)               # identical: keep the (possibly live) dir
            return wt, rel, checks
        if rel.resolve() in (active_release(), link_target("previous")):
            # the same commit no longer builds the same bytes (toolchain update:
            # gcc, cmake, ...). Never replace a release production may run; the
            # installed one is what any further stage (a GPU stack check) tests.
            shutil.rmtree(stage)
            checks.append({"name": "release_reproducible", "level": "warn",
                           "detail": f"rebuild of {rid} differs from the installed release "
                                     f"(toolchain change?); the installed one is kept and used"})
            return wt, rel, checks
        shutil.rmtree(rel)
    os.replace(stage, rel)
    return wt, rel, checks
