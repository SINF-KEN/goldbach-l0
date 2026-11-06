#!/usr/bin/env bash
# gpu_validate.sh — local GPU validation suite (needs the Intel Arc GPU; CI
# has none and runs `ctest -LE gpu`). Registered as ctest `gpu_validate`
# (label gpu): run with
#     ctest --test-dir build -L gpu --output-on-failure
# or directly:  bash tests/gpu_validate.sh build [quick]
#
# 1. exact statistics: the GPU's goldbach_stats log + state against the
#    independent CPU reference (stats_ref) on windows at 1e12, 4e18, 4.4e18,
#    the top of the range, across 2^63 and at the top of the supported range
#    (2^64 - 2^40), sampled / fully exact / multi-segment; then the published
#    windows of an independent verifier (tests/known_windows.py: prime
#    counts, max p(n) and its n at 4e18 + 1e10 / 1e12, full also 1e13 and
#    the 1e12 window across 2^63);
# 2. end-to-end PASS (leftovers 0) in every checker mode + verification of
#    the bucketed primes (hit lists, and the GB_HITLIST=0 ring) against a
#    full packed-kernel rebuild at both ends of [4e18, 5e18], ragged segments
#    and multi-batch hit lists included;
# 3. the bit-exact kernel validators when goldbach-tools is checked out
#    (debug_check_hi, debug_map, debug_sieve, debug_check, debug_fb).
# "quick" skips 3 and the larger windows (~2 min instead of ~12).
# Never touches the campaign files: every run writes into a temp directory.
set -u
BUILD=$(readlink -f "${1:-build}")
QUICK=${2:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=$BUILD/goldbach_l0
REF=$BUILD/stats_ref
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
cd "$ROOT" || exit 1
fail=0
[ -x "$BIN" ] && [ -x "$REF" ] || { echo "build goldbach_l0 and stats_ref first"; exit 1; }
"$BIN" --info >/dev/null 2>&1 || { echo "no usable GPU (goldbach_l0 --info failed)"; exit 1; }
export GB_RIDGE_FILE=$T/ridges.csv GB_RECORDS_FILE=$T/records.txt GB_STATS_LIVE=$T/live.json
export GB_RECORD_HOOK=""

# --- 1. statistics vs CPU reference ----------------------------------------
stats() {   # stats START LIMIT [goldbach_l0 args...]   (env passes through)
    local a=$1 b=$2; shift 2
    rm -f "$T/st.state" "$T/st.log"
    if ! GB_STATS_STATE=$T/st.state GB_STATS_LOG=$T/st.log "$BIN" --start "$a" --limit "$b" "$@" --csv \
            >"$T/gpu.out" 2>&1; then
        echo "  FAIL run [$a, $b] $*"; tail -3 "$T/gpu.out"; fail=$((fail + 1)); return
    fi
    "$REF" "$a" "$b" >"$T/cpu.out"
    if python3 "$ROOT/tests/stats_compare.py" "$T/st.log" "$T/st.state" "$T/cpu.out" >"$T/cmp.out"; then
        echo "  ok   [$a, $b] $* : $(tail -1 "$T/cmp.out")"
    else
        echo "  FAIL [$a, $b] $*"; cat "$T/cmp.out"; fail=$((fail + 1))
    fi
}
echo "== exact statistics vs stats_ref"
stats 4000000000000000000 4000000000020000000 --segment 2000000
GB_STATS_SAMPLE=1 stats 4400000000000000000 4400000000010000000 --segment 1000000
stats 4999999999980000000 5000000000000000000 --segment 3000000
GB_STATS_SAMPLE=5 stats 1000000000000 1000010000000 --segment 1000000
# top of the supported range (2^64 - 2^40): base primes up to 2^32 - 1, own cache
stats 18446742974194923840 18446742974197923840 --segment 1000000 --prime-cache "$T/bp"
# across 2^63 = 9223372036854775808 (signed / unsigned 64-bit boundary), several
# segments on each side; reuses the cache above (bound 2^32 - 1 > sqrt(2^63))
stats 9223372036852775808 9223372036857775808 --segment 1000000 --prime-cache "$T/bp"
# prime gaps >= 700 (the bit-parallel tail in count30): the largest gap below
# 4e18 (1476 after 1425172824437699411), walked and count30-only; a segment
# boundary inside it (stitched by the host); 1442 after 804212830686677669
stats 1425172824437600000 1425172824437800000 --segment 1000000
GB_GAPS=sampled stats 1425172824437600000 1425172824437800000 --segment 1000000
stats 1425172824436700150 1425172824438700150 --segment 1000000
stats 804212830686600000 804212830686800000 --segment 1000000
if [ -z "$QUICK" ]; then
    stats 4000000000000000000 4000000000200000000
    GB_STATS_SAMPLE=3 stats 4000000000000000000 4000000000200000000 --segment 20000000
fi
echo "== published windows (tests/known_windows.py)"
kw() { python3 "$ROOT/tests/known_windows.py" run "$BIN" "$@" || fail=$((fail + 1)); }
kw --only 4e18+1e10,4e18+1e12
if [ -z "$QUICK" ]; then
    kw --only 4e18+1e13
    kw --only 2^63 --prime-cache "$T/bp"
fi

# --- 2. end to end ------------------------------------------------------------
e2e() {     # e2e "ENV=.. ..." goldbach_l0 args...
    local envs=$1; shift
    rm -f "$T/st.state" "$T/st.log"
    # shellcheck disable=SC2086
    out=$(env $envs GB_STATS_STATE="$T/st.state" GB_STATS_LOG="$T/st.log" "$BIN" "$@" --csv 2>&1)
    if [[ "$out" == *result=PASS* && "$out" == *leftovers=0* ]]; then
        echo "  ok   $envs $*"
    else
        echo "  FAIL $envs $*"; echo "$out" | tail -3; fail=$((fail + 1))
    fi
}
echo "== end to end"
W="--start 4000000000000000000 --limit 4000000010000000000"
for m in "X=1" "GB_STATS=0" "GB_C30=0" "GB_CLASSMAP=6" "GB_CLASSMAP=0" "GB_SPLIT=0"; do
    # shellcheck disable=SC2086
    e2e "$m" $W
done
e2e "GB_BUCKET_VERIFY=1" --start 4000000000000000000 --limit 4000000040000000000
e2e "GB_BUCKET_VERIFY=1" --start 4999999999960000000 --limit 5000000000000000000
# hit lists across batch boundaries (K = 5) with a ragged last segment, a
# ragged first segment at the top, and the bucket-ring fallback (GB_HITLIST=0)
e2e "GB_BUCKET_VERIFY=1 GB_HITLIST_K=5" --start 4000000000000000000 --limit 4000000051123456788
e2e "GB_BUCKET_VERIFY=1 GB_HITLIST_K=50" --start 4999999948876543210 --limit 5000000000000000000
e2e "GB_BUCKET_VERIFY=1 GB_HITLIST=0" --start 4000000000000000000 --limit 4000000040000000000
e2e "X=1" --start 1000000000000000 --limit 1000000100000000 --segment 1000000
if "$BIN" --cpu-check 1000000 2>&1 | grep -q PASS; then echo "  ok   --cpu-check 1000000"
else echo "  FAIL --cpu-check"; fail=$((fail + 1)); fi

# --- 3. bit-exact kernel validators (goldbach-tools) ---------------------------
if [ -z "$QUICK" ]; then
    echo "== kernel validators"
    for v in debug_check_hi debug_map debug_sieve debug_check debug_fb; do
        if [ ! -x "$BUILD/$v" ]; then echo "  skip $v (goldbach-tools not built)"; continue; fi
        if "$BUILD/$v" >"$T/$v.out" 2>&1; then echo "  ok   $v: $(tail -1 "$T/$v.out")"
        else echo "  FAIL $v"; tail -5 "$T/$v.out"; fail=$((fail + 1)); fi
    done
fi
echo "gpu_validate: $fail failure(s) -> $([ $fail = 0 ] && echo PASS || echo FAIL)"
[ $fail = 0 ]
