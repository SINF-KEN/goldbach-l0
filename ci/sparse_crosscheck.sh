#!/usr/bin/env bash
# sparse_crosscheck.sh — an independent second verification of the campaign.
#
# The dense campaign verifies every even of [4e18, 5e18] once (~90 days, for
# the exact statistics). The sparse path (--sparse: a 12-class map mod 30030,
# other kernels, another map layout, its own checker) verifies the same evens
# ~14x faster (1.015e12 evens/s measured at 4e18, ~5.7 days for the whole
# range) but keeps no statistics. Running it over the range gives every even
# a second PASS from different code, so a silent fault in one pipeline (a
# transient GPU error, a kernel bug) would have to coincide with one in the
# other to go unnoticed.
#
#   ci/sparse_crosscheck.sh run [START LIMIT [SLICE]]   (default 4e18 5e18 1e15)
#   ci/sparse_crosscheck.sh status [START LIMIT]
#
# run: needs production stopped (`goldbach down`, then wait until `goldbach
# status` shows it stopped): it holds the GPU lock for the whole range.
# Resumable like any run_range campaign (it IS one: tools/run_range.sh with
# GB_SPARSE=1 and its own journal range-sparse-START-LIMIT.journal); stop it
# with Ctrl-C or `touch range-sparse-START-LIMIT.stop`. The ridge store, the
# records file and the statistics files are not touched (GB_RIDGE=0, its own
# records file, no statistics on the sparse path).
#
# status: the contiguous coverage of both journals from START and the evens
# both have verified.
set -u
cd "${GOLDBACH_ROOT:-$(dirname "$(readlink -f "$0")")/..}" || exit 1   # the campaign directory
cmd=${1:-status}
START=${2:-4000000000000000000}
LIMIT=${3:-5000000000000000000}
SLICE=${4:-1000000000000000}
DENSE=range-$START-$LIMIT.journal
SPARSE=range-sparse-$START-$LIMIT.journal

# end of the PASS chain from START in a journal (the run_range / CI rule:
# rows must extend the chain exactly and agree with their own start= / limit=)
chain_end() {
    python3 - "$1" "$START" <<'EOF'
import sys
nxt = int(sys.argv[2]); rows = {}
try:
    for ln in open(sys.argv[1]):
        t = ln.split()
        if len(t) < 4 or t[0] != "PASS" or not (t[1].isdigit() and t[2].isdigit()):
            continue
        kv = dict(x.split("=", 1) for x in t[4:] if "=" in x)
        if kv.get("start") == t[1] and kv.get("limit") == t[2]:
            rows.setdefault(int(t[1]), int(t[2]))
except OSError:
    pass
while nxt in rows:
    nxt = rows[nxt] + 2
print(nxt - 2)
EOF
}

case "$cmd" in
run)
    desired=$(python3 -c 'import json; print(json.load(open("ci-state/production.json")).get("desired", "?"))' 2>/dev/null || echo "?")
    if [ "$desired" = running ] || pgrep -f '^(/usr/bin/)?bash \S*run_range\.sh ' >/dev/null; then
        echo "production is running or desired running: stop it first (goldbach down)" >&2
        exit 2
    fi
    mkdir -p ci-state/sparse-crosscheck
    export GB_SPARSE=1 GB_RIDGE=0 GB_RECORD_HOOK=
    export GB_RECORDS_FILE=$PWD/ci-state/sparse-crosscheck/records.txt
    export GB_STATS_LOG=$PWD/ci-state/sparse-crosscheck/stats.log GB_STATS_STATE=$PWD/ci-state/sparse-crosscheck/stats.state
    export GB_STATS_LIVE=$PWD/ci-state/sparse-crosscheck/live.json
    exec bash tools/run_range.sh "$START" "$LIMIT" "$SLICE" "$SPARSE"
    ;;
status)
    d=$(chain_end "$DENSE"); s=$(chain_end "$SPARSE")
    both=$(( d < s ? d : s ))
    echo "dense  ($DENSE): verified [$START, $d]"
    echo "sparse ($SPARSE): verified [$START, $s]"
    if (( both >= START )); then
        python3 -c "import sys; a,b,lo,hi=map(int,sys.argv[1:]); print(f'both:   [{a}, {b}] = {(b-a)//2+1} evens, {100*(b-a+2)/(hi-lo+2):.3f}% of the range')" \
            "$START" "$both" "$START" "$LIMIT"
    else
        echo "both:   nothing yet"
    fi
    ;;
*)
    sed -n '2,25p' "$0"; exit 2 ;;
esac
