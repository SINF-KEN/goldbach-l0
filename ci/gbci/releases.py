"""Releases under ci-state/releases/: code fingerprints, the active /
previous links, promotion and pruning."""
import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .core import ROOT, STATE, cfg, git, load_json, log
from .prodstate import find_run_range, prod_conf, prod_paths


def fingerprint(rel):
    """Code fingerprint: the binary without debug info / build-id (both embed
    the build path) + every shipped kernel module. Equal fingerprints = the
    same machine code and kernels, whatever directory they were built in."""
    h = hashlib.sha256()
    with tempfile.TemporaryDirectory() as td:
        s = Path(td) / "s"
        subprocess.run(["strip", "--remove-section=.note.gnu.build-id", "-o", str(s),
                        str(rel / "goldbach_l0")], check=True)
        h.update(s.read_bytes())
    for k in sorted((rel / "kernels").glob("*.spv")):
        h.update(k.name.encode())
        h.update(k.read_bytes())
    return h.hexdigest()


def release_dir(rid):
    return STATE / "releases" / rid


def link_target(name):
    p = STATE / name
    return p.resolve() if p.exists() else None


def active_release():
    return link_target("active")


def release_meta(rel):
    return load_json(rel / "meta.json", {}) if rel else {}


def set_link(name, target):
    tmp = STATE / f".{name}.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(os.path.relpath(target, STATE), tmp)
    os.replace(tmp, STATE / name)


def promote(rel):
    cur = active_release()
    # previous = the last DIFFERENT code: a code-identical promotion (docs,
    # CI scripts) must not push the real rollback target out
    if cur and cur != rel.resolve() and \
            release_meta(cur).get("fingerprint") != release_meta(rel).get("fingerprint"):
        set_link("previous", cur)
    set_link("active", rel)
    prev = link_target("previous")
    log(f"promoted: active = {rel.name} (previous = {prev.name if prev else '-'})")
    prune_releases()


RELEASE_RE = re.compile(r"\brelease=([0-9a-f]{12})\b")


def named_releases():
    """Release ids named by ANY journal or recheck file in production's
    directory (every row kind: PASS, RETRY, RECHECK), or None when one of
    them -- or the journal production.json names -- cannot be read.

    Protecting only the journal production.json names at the moment lost the
    binaries of 23 of the 25 releases behind the campaign's PASS rows: on
    2026-10-19 production switched to the gap-backfill journal and the next
    prune removed every release named only in the main journal (and the
    switch back did the same to the backfill's)."""
    pc = prod_conf()
    jdir = ROOT
    if pc.get("journal"):
        j = prod_paths(pc)[0]
        if not j.is_file():
            return None
        jdir = j.parent
    names = set()
    for f in sorted(jdir.glob("*.journal")) + sorted(jdir.glob("*.recheck")):
        try:
            names |= set(RELEASE_RE.findall(f.read_text(errors="replace")))
        except OSError:
            return None
    return names


def tag_releases(ids):
    """Pins each release commit with a local tag release/<id>, so the source
    of every binary the journal names stays reachable after its branch is
    deleted (git gc would drop it). Ids without a local commit are skipped."""
    have = set(git("tag", "--list", "release/*", check=False).split())
    made = []
    for rid in sorted(ids):
        if f"release/{rid}" in have or not git("rev-parse", "-q", "--verify", f"{rid}^{{commit}}", check=False):
            continue
        git("tag", f"release/{rid}", rid, check=False)
        made.append(rid)
    if made:
        log(f"tagged releases: {', '.join(made)}")


def prune_releases():
    """Keep active, previous, every release named in any production journal
    (named_releases), the newest keep_releases others and the binary
    production runs right now. Fails closed: when a journal cannot be read
    nothing is pruned."""
    named = named_releases()
    if named is None:
        log("prune skipped: a production journal cannot be read (its releases must be kept)")
        return
    tag_releases(named)
    keep = {p for p in (active_release(), link_target("previous")) if p}
    keep |= {release_dir(r).resolve() for r in named}
    for p in find_run_range():
        env = Path(f"/proc/{p['pid']}/environ")
        with contextlib.suppress(OSError):
            for e in env.read_bytes().split(b"\0"):
                if e.startswith(b"GB_RANGE_BIN="):
                    keep.add(Path(e.split(b"=", 1)[1].decode()).parent.resolve())
    rels = sorted((d for d in (STATE / "releases").glob("*") if d.is_dir() and not d.name.endswith(".staging")),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    spare = [d for d in rels if d.resolve() not in keep][cfg()["keep_releases"]:]
    for d in spare:
        shutil.rmtree(d, ignore_errors=True)
    if spare:
        log(f"pruned releases: {', '.join(d.name for d in spare)}")
