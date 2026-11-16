"""Independent prime counts: primecount (cached pi(x)) and Miller-Rabin."""
import os
import shutil
import subprocess
from pathlib import Path

from .core import STATE, cfg, load_json, save_json


def primecount_bin():
    for cand in [cfg()["primecount"], shutil.which("primecount") or "",
                 str(Path.home() / ".local/src/primecount/build/primecount")]:
        if cand and os.access(cand, os.X_OK):
            return cand
    return None


def is_prime(n):
    """Deterministic Miller-Rabin for n < 3.3e24 (first 12 prime bases)."""
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in bases:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def pi(x):
    """pi(x) via primecount, cached in ci-state/pi-cache.json; pi(odd x)
    reuses a cached pi(x - 1)."""
    cache = load_json(STATE / "pi-cache.json", {})
    if str(x) in cache:
        return int(cache[str(x)])
    if x % 2 and str(x - 1) in cache:
        return int(cache[str(x - 1)]) + int(is_prime(x))
    pc = primecount_bin()
    if not pc:
        raise RuntimeError("primecount not found (set primecount in ci-state/config.json)")
    p = subprocess.run(["nice", "-n", "19", pc, str(x), f"-t{cfg()['primecount_threads']}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"primecount {x}: {p.stderr.strip()}")
    v = int(p.stdout.split()[-1])
    cache = load_json(STATE / "pi-cache.json", {})
    cache[str(x)] = str(v)
    save_json(STATE / "pi-cache.json", cache)
    return v


def primes_in(a, b):
    """Number of primes in [a, b]."""
    return pi(b) - pi(a - 1)
