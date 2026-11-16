"""The GPU software stack (kernel, xe, compute runtime, IGC, Level Zero,
VRAM clock) and its changes since the active release was verified."""
import contextlib
import os
import subprocess
from pathlib import Path

from .core import STATE, load_json, save_json


GPU_STACK_PKGS = ("intel-compute-runtime", "intel-graphics-compiler", "level-zero-loader")


def gpu_stack():
    """What turns the shipped SPIR-V into GPU machine code and runs it: the
    kernel + xe module, the compute runtime / IGC / Level Zero packages, the
    VRAM clock. A change here changes production's GPU code (or its memory
    timing) without any commit."""
    snap = {"kernel": os.uname().release}
    with contextlib.suppress(OSError):
        snap["xe"] = Path("/sys/module/xe/srcversion").read_text().strip()
    p = subprocess.run(["pacman", "-Q", *GPU_STACK_PKGS], capture_output=True, text=True)
    for ln in p.stdout.splitlines():
        name, ver = ln.split()[:2]
        snap[name] = ver
    for f in Path("/sys/class/drm").glob("card[0-9]*/device/tile0/gt0/oc/mem_speed"):
        with contextlib.suppress(OSError):
            snap["mem_speed"] = f.read_text().strip()
    return snap


GPU_STACK_FAILED = "gpu-stack-failed.json"


def gpu_stack_change(now=None):
    """{key: (verified, now)} since the active release was last verified."""
    ref = load_json(STATE / "gpu-stack.json", None)
    now = gpu_stack() if now is None else now
    if ref is None:                      # first run: today's stack is the baseline
        save_json(STATE / "gpu-stack.json", now)
        return {}
    return {k: (ref.get(k), now.get(k)) for k in sorted(set(ref) | set(now)) if ref.get(k) != now.get(k)}


def gpu_stack_check_due():
    """The stack change still to verify: {} when there is none, or when the
    re-verification already failed on exactly this stack (ci-state/
    gpu-stack-failed.json; it is retried once the stack changes again, or
    after that file is deleted). Without this, a failed check re-ran -- and
    alerted -- on every daemon poll."""
    now = gpu_stack()
    change = gpu_stack_change(now)
    if change and load_json(STATE / GPU_STACK_FAILED, None) == now:
        return {}
    return change
