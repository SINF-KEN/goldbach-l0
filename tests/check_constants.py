#!/usr/bin/env python3
"""check_constants.py — host <-> kernel constant mirrors, checked statically.

Many constants exist twice: once in the C++ host (include/*.hpp) and once in
an OpenCL kernel (#define ... inside #ifndef), with a "MUST match" comment.
Only the sparse module verifies its copy at run time (scm_geometry
handshake, GPU needed); a drifted copy anywhere else silently sizes buffers,
item widths or tiles differently on the two sides — e.g. the snapshot
kernels' TILE_WORDS 8192 vs the host's 2048 (fixed in v2.0.1) overran the map.

This script extracts each value from source and fails on any mismatch. It
also checks that every ocloc -D option set used to build a sparse module (CI,
CMakeLists.txt) matches include/sparse.hpp scm_geo(). No compiler, no GPU.

usage: tests/check_constants.py   (run from anywhere; exit 1 on mismatch)
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAIL = []


def read(rel):
    return (ROOT / rel).read_text()


def cexpr(expr, env):
    """Evaluate a small integer C expression (literals, names, * << + - / ( ))."""
    e = re.sub(r"//.*", "", expr).strip()
    e = re.sub(r"\(\s*(?:u64|u32|uint64_t|uint32_t|size_t)\s*\)", "", e)          # casts
    e = re.sub(r"\b(?:uint64_t|uint32_t)\(([^()]*)\)", r"(\1)", e)               # functional casts
    e = re.sub(r"(?<=[0-9])'(?=[0-9])", "", e)                                     # digit separators
    e = re.sub(r"\b(0x[0-9a-fA-F]+|[0-9]+)(?:[uU]?[lL]{0,2}|[lL]{1,2}[uU]?)\b", r"\1", e)
    for name in sorted(set(re.findall(r"[A-Za-z_]\w*", e)), key=len, reverse=True):
        if name not in env:
            raise KeyError(f"unknown name {name!r} in {expr!r}")
        e = re.sub(rf"\b{name}\b", f"({env[name]})", e)
    if not re.fullmatch(r"[0-9a-fA-Fx()+\-*/<> \t]*", e):
        raise ValueError(f"not a plain integer expression: {expr!r}")
    return int(eval(e.replace("/", "//")))          # noqa: S307 (validated above)


def cl_defines(rel, extra=None):
    """Default value of every #define in a kernel (first definition wins, like
    the #ifndef guards do without -D options); `extra` = -D overrides."""
    env = dict(extra or {})
    for m in re.finditer(r"^\s*#define\s+(\w+)\s+([^\n\\]+)$", read(rel), re.M):
        name, val = m.group(1), m.group(2)
        if name in env:
            continue
        try:
            env[name] = cexpr(val, env)
        except (KeyError, ValueError, SyntaxError):
            pass                                     # macros, lists, float values
    return env


def host_constants(*rels):
    env = {}
    for rel in rels:
        src = read(rel)
        for m in re.finditer(r"^\s*#define\s+(\w+)\s+([^\n]+)$", src, re.M):
            try:
                env.setdefault(m.group(1), cexpr(m.group(2), env))
            except (KeyError, ValueError, SyntaxError):
                pass
        for m in re.finditer(r"inline\s+constexpr\s+[\w:]+\s+(\w+)\s*=\s*([^;]+);", src):
            try:
                env[m.group(1)] = cexpr(m.group(2), env)
            except (KeyError, ValueError, SyntaxError):
                pass
    return env


def same(what, host_val, kernel_val):
    ok = host_val == kernel_val
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: host {host_val}  kernel {kernel_val}")
    if not ok:
        FAIL.append(what)


def main():
    cfg = host_constants("include/config.hpp")
    gbl = cl_defines("kernels/goldbach.cl")
    print("config.hpp <-> kernels/goldbach.cl")
    same("TILE_WORDS", cfg["TILE_WORDS"], gbl["TILE_WORDS"])
    same("WG_SIZE", cfg["WG_SIZE"], gbl["WG_SIZE"])
    same("LARGE_PRIMES_PER_ITEM", cfg["LARGE_PRIMES_PER_ITEM"], gbl["LARGE_PRIMES_PER_ITEM"])
    same("CHECK_WORDS_PER_ITEM / WORDS_PER_ITEM", cfg["CHECK_WORDS_PER_ITEM"], gbl["WORDS_PER_ITEM"])
    same("LARGE_PACK_PPI", cfg["LARGE_PACK_PPI"], gbl["LARGE_PACK_PPI"])
    same("LARGE_PACK_GROUP", cfg["LARGE_PACK_GROUP"], gbl["LARGE_PACK_GROUP"])
    same("WHEEL_W", cfg["WHEEL_W"], gbl["WHEEL_W"])
    same("CHECK_PHASE_A_W64 / A_W64", cfg["CHECK_PHASE_A_W64"], gbl["A_W64"])
    same("MASK_MAX_P", cfg["MASK_MAX_P"], gbl["MASK_MAX_P"])
    same("MAX_LEFTOVERS", cfg["MAX_LEFTOVERS"], gbl["MAX_LEFTOVERS"])
    same("RIDGE_CAP", cfg["RIDGE_CAP"], gbl["RIDGE_CAP"])
    split = host_constants("include/split_check.hpp")
    c30 = host_constants("include/check30.hpp")
    same("SPLIT_MAXD", split["SPLIT_MAXD"], gbl["SPLIT_MAXD"])
    same("C30_MAXD", c30["C30_MAXD"], gbl["C30_MAXD"])
    gs = host_constants("include/gstats.hpp")
    same("STATS_KMAX / X_KMAX", gs["STATS_KMAX"], gbl["X_KMAX"])
    same("STATS_XB_SPAN / XB_SPAN", gs["STATS_XB_SPAN"], gbl["XB_SPAN"])
    same("COUNT30_CW / C30_CW", gs["COUNT30_CW"], gbl["C30_CW"])
    same("GAPS30_TAIL / G30T_TAIL", gs["GAPS30_TAIL"], gbl["G30T_TAIL"])
    same("GAPS30_HBINS / G30_HBINS", gs["GAPS30_HBINS"], gbl["G30_HBINS"])
    same("GAPS30_MAX_SG / G30_MAX_SG", gs["GAPS30_MAX_SG"], gbl["G30_MAX_SG"])
    xw = re.search(r"XS_W_HOST\[4\]\s*=\s*\{([^}]*)\}", read("include/gstats.hpp"))
    xw = [int(x) for x in re.findall(r"\d+", xw.group(1))]
    for i, v in enumerate(("XS_W8", "XS_W6", "XS_W4", "XS_W3")):
        same(f"XS_W_HOST[{i}] / {v}", xw[i], gbl[v])
    mg = re.search(r"MAX_MASK_GROUPS_HOST\s*=\s*(\d+)", read("src/verifier.cpp"))
    same("MAX_MASK_GROUPS (verifier.cpp)", int(mg.group(1)), gbl["MAX_MASK_GROUPS"])
    rmax = re.search(r"bucket_R\s*>\s*(\d+)", read("src/verifier.cpp"))
    same("bucket ring limit / BKT_MAX_R", int(rmax.group(1)), gbl["BKT_MAX_R"])

    print("sparse.hpp <-> kernels/sparse.cl")
    sp = host_constants("include/sparse.hpp")
    scl = cl_defines("kernels/sparse.cl")
    same("SCM_MAXD", sp["SCM_MAXD"], scl["SCM_MAXD"])
    same("SCM_META", sp["SCM_META"], scl["SCM_META"])
    same("SCM_TILE_WORDS / TILE_WORDS", sp["SCM_TILE_WORDS"], scl["TILE_WORDS"])
    same("SCM_MAX_GROUPS / MAX_GROUPS", sp["SCM_MAX_GROUPS"], scl["MAX_GROUPS"])
    same("SCM_A_WPL / A3_WPL", sp["SCM_A_WPL"], scl["A3_WPL"])
    same("LARGE_PACK_PPI", cfg["LARGE_PACK_PPI"], scl["LARGE_PACK_PPI"])
    same("LARGE_PACK_GROUP", cfg["LARGE_PACK_GROUP"], scl["LARGE_PACK_GROUP"])

    # scm_geo(): per modulus the residues and span the module must be built with
    geo = {}
    for m in re.finditer(r"g\s*=\s*\{\s*(\d+)\s*,\s*(\d+)\s*,\s*\{([^}]*)\}\s*,\s*\d+\s*,\s*\d+\s*,\s*"
                         r"\"([^\"]+)\"\s*,\s*(\d+)\s*\}", read("include/sparse.hpp")):
        M, K = int(m.group(1)), int(m.group(2))
        res = [int(x) for x in re.findall(r"\d+", m.group(3))]
        geo[M] = dict(K=K, res=res, spv=m.group(4), span=int(m.group(5)))
    if set(geo) != {2310, 30030}:
        FAIL.append("could not parse scm_geo()")
    print("scm_geo() <-> sparse module build options")
    default = dict(M=scl["SCM_M"], K=scl["SCM_K"],
                   res=[int(x) for x in re.findall(r"\d+", re.search(
                       r"#define\s+SCM_RES_LIST\s+([^\n]+)", read("kernels/sparse.cl")).group(1))],
                   span=scl["A3_SPAN"])
    g = geo.get(default["M"], {})
    same("sparse.cl defaults: K", g.get("K"), default["K"])
    same("sparse.cl defaults: residues", g.get("res"), default["res"])
    same("sparse.cl defaults: A3_SPAN", g.get("span"), default["span"])
    for rel in (".github/workflows/ci.yml", "CMakeLists.txt"):
        for opts in re.findall(r"-DSCM_M=[^\"\n]*", read(rel)):
            d = dict(re.findall(r"-D(\w+)=(\S+)", opts))
            M = int(d["SCM_M"].rstrip("u"))
            g = geo.get(M, {})
            same(f"{rel} M={M}: K", g.get("K"), int(d["SCM_K"].rstrip("u")))
            same(f"{rel} M={M}: residues", g.get("res"),
                 [int(x.rstrip("u")) for x in d["SCM_RES_LIST"].split(",")])
            same(f"{rel} M={M}: A3_SPAN", g.get("span"), int(d.get("A3_SPAN", scl["A3_SPAN"])))

    print(f"check_constants: {len(FAIL)} mismatch(es) -> {'FAIL' if FAIL else 'PASS'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
