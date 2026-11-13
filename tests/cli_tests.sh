#!/usr/bin/env bash
# cli_tests.sh — the command-line contract of goldbach_l0, GPU-free.
#
#   bash tests/cli_tests.sh [path/to/goldbach_l0]     (default: build/goldbach_l0)
#
# Malformed numbers and options missing their value must be usage errors
# (exit 2) — never a silently truncated limit (a typo that verifies nothing
# yet reports PASS) or a read past argv. Valid spellings must agree, and the
# CPU reference must PASS. Every failing case is listed; exit 1 if any failed.
set -u
BIN=${1:-build/goldbach_l0}
[ -x "$BIN" ] || { echo "cli_tests: no executable at $BIN" >&2; exit 1; }
fail=0
expect_rc() {   # expected-exit-code description args...
    local want=$1 what=$2; shift 2
    "$BIN" "$@" >/dev/null 2>&1
    local rc=$?
    if [ "$rc" -eq "$want" ]; then echo "  ok   $what (exit $rc)"
    else echo "  FAIL $what: exit $rc, expected $want"; fail=1; fi
}

echo "malformed numbers -> exit 2"
for bad in 4e18x 0x100 -5 99999999999999999999999 1e20 '' ' 7' 1e 1.5e9 010x; do
    expect_rc 2 "--limit '$bad'" --limit "$bad"
done
expect_rc 2 "--cpu-check 1e99" --cpu-check 1e99
expect_rc 2 "--cpu-check 0 (used to start a GPU run)" --cpu-check 0

echo "options missing their value -> exit 2"
for opt in --limit --start --segment --split --cpu-check --spv --prime-cache; do
    expect_rc 2 "$opt (last, no value)" "$opt"
done
expect_rc 2 "unknown option" --no-such-option
expect_rc 0 "--help" --help

echo "empty / reversed ranges and bad option values -> exit 2 (before any GPU work)"
# Before v2.3.0, start > limit verified n = limit (outside the request) and
# reported PASS; an odd start == limit failed with an unrelated GPU error.
expect_rc 2 "--start 200 --limit 100 (reversed)" --start 200 --limit 100
expect_rc 2 "--start 101 --limit 101 (no even)" --start 101 --limit 101
expect_rc 2 "--start 4000000000000000001 --limit 4000000000000000001" --start 4000000000000000001 --limit 4000000000000000001
expect_rc 2 "--limit 3 (no even >= 4)" --limit 3
expect_rc 2 "--start 18446744073709551615 (odd max, wraps when rounded)" --start 18446744073709551615 --limit 18446744073709551615
expect_rc 2 "--split 4294967296 (> 2^32 - 1)" --split 4294967296 --limit 1e6
expect_rc 2 "--wheel (removed in v2.3.0)" --wheel --limit 1e6

echo "limits above 2^64 - 2^40 fail closed before any GPU work -> exit 1"
for top in 18446744073709551614 18446742974197923842; do
    for mode in "" --sparse; do
        out=$("$BIN" --start 18446742974197000000 --limit "$top" ${mode:+"$mode"} --csv 2>&1); rc=$?
        if [ "$rc" -eq 1 ] && grep -q "supported maximum" <<<"$out" && ! grep -q "result=PASS" <<<"$out"; then
            echo "  ok   --limit $top $mode refused"
        else echo "  FAIL --limit $top $mode: exit $rc: $(tail -1 <<<"$out")"; fail=1; fi
    done
done

echo "valid spellings agree; CPU reference passes"
a=$("$BIN" --cpu-check 200000 2>&1); ra=$?
b=$("$BIN" --cpu-check 2e5 2>&1); rb=$?
if [ "$ra" -eq 0 ] && [ "$rb" -eq 0 ] && [ "$a" = "$b" ] && grep -q "PASS" <<<"$a"; then
    echo "  ok   --cpu-check 200000 == --cpu-check 2e5, PASS"
else
    echo "  FAIL --cpu-check 200000 / 2e5: '$a' (exit $ra) vs '$b' (exit $rb)"; fail=1
fi
out=$("$BIN" --cpu-check 2000000 2>&1); rc=$?
if [ "$rc" -eq 0 ] && grep -q "PASS" <<<"$out"; then echo "  ok   --cpu-check 2e6 PASS"
else echo "  FAIL --cpu-check 2e6: $out (exit $rc)"; fail=1; fi

echo "cli_tests: $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
