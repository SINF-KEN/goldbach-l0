#!/usr/bin/env bash
# make-plots.sh — regenerate the README/reference benchmark figures from the
# bench.py journal (bench/bench-results.csv) with tools/waterfall.py.
#
#   bash docs/make-plots.sh [CSV] [LABELS]
#   defaults: bench/bench-results.csv, the labels in bench/figure-versions
#   (newest first; the workflow "Version figures" adds each new version)
#
# The labels are drawn in three groups, one figure per group and window, because
# the groups measure different pipelines and are not comparable bar for bar:
#   stats     dense path with exact statistics (v2.2.0+, the production path;
#             every version the workflow "Version figures" adds lands here)
#   sparse    the opt-in sparse path (GB_SPARSE=1, ~14x the dense GPU rate, no
#             statistics): rows whose gb_env column says GB_SPARSE=1, and v2.0.0
#             and v2.1.0, benched only on it before bench.py recorded gb_env
#   prestats  dense path before the statistics (v1.x)
# ci/figure_groups.py does the sorting and stops the script when a label's drawn
# rows mix sparse and dense runs (a sparse session benched under a version label).
#
# Figures (latest bench.py session per label and window, time per 1e10 integers):
#   waterfall-{long,steady}-stats-by-version.png
#   waterfall-{long,steady}-sparse-by-version.png
#   waterfall-{long,steady}-prestats-by-version.png
#                                    long = 1e11 at 4e18 (production-like, v1.9.0+),
#                                    steady = 2e10 at 4e18
#   waterfall-current-by-window.png  the current version across all its windows
# Each group has its own x axis (shared by its long and steady figure): on one
# axis v1.6.2 (~700 ms) would squash the stats bars and the dense scale would
# leave the sparse bars a few pixels wide. The by-window figure has its own:
# the short `bench` window is startup-dominated (process init amortized over
# 2e9 integers) and would flatten every other bar.
#
# Needs pandas + matplotlib; uses tools/.venv if present, else python3.
# First time:  python3 -m venv tools/.venv && tools/.venv/bin/pip install pandas matplotlib
set -euo pipefail
cd "$(dirname "$0")/.."
CSV=${1:-bench/bench-results.csv}
if [ -z "${2:-}" ]; then
    LABELS=$(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' bench/figure-versions | paste -sd, -)
else
    LABELS=$2
fi
CUR=${LABELS%%,*}                     # first label = current version
if [ -z "${PY:-}" ]; then              # caller's interpreter first (CI), else the tools venv
    PY=python3
    [ -x tools/.venv/bin/python ] && PY=tools/.venv/bin/python
fi
W=tools/waterfall.py
OUT=docs/images
FIG_GROUPS=$(python3 ci/figure_groups.py "$CSV" "$LABELS")   # stops on a sparse/dense mix
rm -f "$OUT"/waterfall-*.png
G_STATS=; G_SPARSE=; G_PRE=
while read -r g lbs; do
    case $g in
        stats)    G_STATS=,$lbs ;;
        sparse)   G_SPARSE=,$lbs ;;
        prestats) G_PRE=,$lbs ;;
    esac
done <<< "$FIG_GROUPS"
# figure titles need a waterfall.py with --title (goldbach-tools PR #4); older: default title
TITLE=0
"$PY" "$W" --help 2>/dev/null | grep -q -e '--title' && TITLE=1

xmax() {  # x limit of a group: the larger of its long and steady figures
    [ -n "$1" ] || return 0
    for w in long steady; do
        "$PY" "$W" "$CSV" --window "$w" --label "${1#,}" --group-by label --per 1e10 --latest --print-xmax
    done | awk '/^XMAX/ { if ($2 > m) m = $2 } END { print m }'
}
draw() {  # group-name labels x-limit title
    [ -n "$2" ] || return 0
    local t=()
    [ "$TITLE" = 1 ] && t=(--title "$4")
    for w in long steady; do
        "$PY" "$W" "$CSV" --window "$w" --label "${2#,}" --group-by label --per 1e10 --latest \
            --xmax "$3" ${t[@]+"${t[@]}"} -o "$OUT/waterfall-$w-$1-by-version.png"
    done
}
draw stats    "$G_STATS"  "$(xmax "$G_STATS")"  "Dense path with exact statistics (production) — by version"
draw prestats "$G_PRE"    "$(xmax "$G_PRE")"    "Dense path before the statistics — by version"
draw sparse   "$G_SPARSE" "$(xmax "$G_SPARSE")" "Sparse path (GB_SPARSE=1, opt-in, no statistics) — by version"
"$PY" "$W" "$CSV" --label "$CUR" --group-by window --per 1e10 --latest -o "$OUT/waterfall-current-by-window.png"
echo "figures written to $OUT/"
