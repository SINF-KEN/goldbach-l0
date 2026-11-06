#!/usr/bin/env bash
# make-plots.sh — regenerate the README/reference benchmark figures from the
# bench.py journal (bench/bench-results.csv) with tools/waterfall.py.
#
#   bash docs/make-plots.sh [CSV] [LABELS]
#   defaults: bench/bench-results.csv, "v2.1.0,v2.0.0,v1.13.0,...,v1.6.2" (arm labels to compare)
#
# Figures (latest bench.py session per label and window, time per 1e10 integers):
#   waterfall-long-by-version.png    long window (1e11 at 4e18, production-like),
#                                    every version that has long rows (v1.9.0+)
#   waterfall-steady-by-version.png  steady window (2e10 at 4e18), every version
#   waterfall-current-by-window.png  the current version across all its windows
# The two by-version figures share one x axis. The by-window figure has its own:
# the short `bench` window is startup-dominated (process init amortized over
# 2e9 integers) and would flatten every other bar.
#
# Needs pandas + matplotlib; uses tools/.venv if present, else python3.
# First time:  python3 -m venv tools/.venv && tools/.venv/bin/pip install pandas matplotlib
set -euo pipefail
cd "$(dirname "$0")/.."
CSV=${1:-bench/bench-results.csv}
LABELS=${2:-v2.1.0,v2.0.0,v1.13.0,v1.12.0,v1.11.0,v1.10.1,v1.10.0,v1.9.0,v1.8.2,v1.7.0,v1.6.2}
CUR=${LABELS%%,*}                     # first label = current version
if [ -z "${PY:-}" ]; then              # caller's interpreter first (CI), else the tools venv
    PY=python3
    [ -x tools/.venv/bin/python ] && PY=tools/.venv/bin/python
fi
W=tools/waterfall.py
OUT=docs/images
rm -f "$OUT"/waterfall-*.png

L=(--window long --label "$LABELS" --group-by label --per 1e10 --latest)
S=(--window steady --label "$LABELS" --group-by label --per 1e10 --latest)
B=(--label "$CUR" --group-by window --per 1e10 --latest)
XMAX=$( { "$PY" "$W" "$CSV" "${L[@]}" --print-xmax; "$PY" "$W" "$CSV" "${S[@]}" --print-xmax; } \
        | awk '/^XMAX/ { if ($2 > m) m = $2 } END { print m }')
"$PY" "$W" "$CSV" "${L[@]}" --xmax "$XMAX" -o "$OUT/waterfall-long-by-version.png"
"$PY" "$W" "$CSV" "${S[@]}" --xmax "$XMAX" -o "$OUT/waterfall-steady-by-version.png"
"$PY" "$W" "$CSV" "${B[@]}" -o "$OUT/waterfall-current-by-window.png"
echo "figures written to $OUT/"
