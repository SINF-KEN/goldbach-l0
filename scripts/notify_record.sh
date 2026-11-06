#!/usr/bin/env bash
# notify_record.sh — called by goldbach_l0 (in the background) for every
# record event, after the event was re-proven on the CPU and appended to
# goldbach_records.txt:
#
#   notify_record.sh KIND P N Q CHECK
#     KIND   RECORD  p(n) > 9781: beats the largest p(n) known below 4e18
#            SP      first even of the campaign with p(n) = p for a prime whose
#                    S(p) was unknown below 4e18 (a new table entry if the
#                    campaign covers [4e18, n] without a gap)
#            BEYOND  p(n) outside the GPU candidate list but <= 9781 (n < 9786)
#            TEST    manual test of every configured channel
#            ALERT   operational alert (run_range.sh retry/stop, local CI
#                    failure, prime-count mismatch): P = short title,
#                    N = message (Q, CHECK unused); always loud
#     P N Q  the partition n = P + Q (P = p(n), the smallest prime that works)
#     CHECK  OK (independent CPU Miller-Rabin scan agrees) or MISMATCH
#
# Channels (each optional; nothing leaves this machine unless you set one):
#   desktop  notify-send, if a graphical session is reachable        (default on)
#   bell     terminal bell + banner on every open terminal of this user (wall)
#            GB_NOTIFY_WALL=1
#   phone    ntfy.sh push: GB_NTFY_TOPIC=<your-secret-topic> [GB_NTFY_SERVER]
#   mail     GB_NOTIFY_MAIL=you@example.org (needs a working `mail`)
#   custom   GB_NOTIFY_CMD='...' — run with the same five arguments
# SP events only notify phone / mail with GB_NOTIFY_SP=1 (expect a few dozen
# over [4e18, 5e18]); RECORD always does. Override the whole script with
# GB_RECORD_HOOK=/path/to/cmd (GB_RECORD_HOOK= empty: no hook at all).
set -u
KIND=${1:-?} P=${2:-?} N=${3:-?} Q=${4:-?} CHECK=${5:-?}
case $KIND in
    RECORD) TITLE="GOLDBACH RECORD: p(n) = $P > 9781"
            BODY="n = $N = $P + $Q (CPU check: $CHECK). Beats Oliveira e Silva's 9781. See goldbach_records.txt."
            URG=critical ;;
    SP)     TITLE="New S($P) candidate"
            BODY="First even with p(n) = $P: n = $N (CPU check: $CHECK); S($P) was unknown below 4e18."
            URG=normal ;;
    ALERT)  TITLE="goldbach-l0: $P"; BODY="$N"; URG=critical ;;
    *)      TITLE="Goldbach event $KIND p(n) = $P"; BODY="n = $N (CPU check: $CHECK)"; URG=normal ;;
esac
loud=0
[ "$KIND" = RECORD ] && loud=1
[ "$KIND" = SP ] && [ "${GB_NOTIFY_SP:-0}" = 1 ] && loud=1
[ "$KIND" = TEST ] && loud=1
[ "$KIND" = ALERT ] && loud=1                # bash scripts/notify_record.sh TEST 1 2 1 OK

# desktop (best effort; a TTY-started run finds the session bus via XDG_RUNTIME_DIR)
if command -v notify-send >/dev/null 2>&1; then
    export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}
    notify-send -u "$URG" -a goldbach-l0 "$TITLE" "$BODY" >/dev/null 2>&1 || true
fi
if [ "${GB_NOTIFY_WALL:-0}" = 1 ] && command -v wall >/dev/null 2>&1; then
    printf '\a%s\n%s\n' "$TITLE" "$BODY" | wall >/dev/null 2>&1 || true
fi
if [ "$loud" = 1 ] && [ -n "${GB_NTFY_TOPIC:-}" ] && command -v curl >/dev/null 2>&1; then
    curl -fsS -m 20 -H "Title: $TITLE" -H "Priority: $([ "$KIND" = RECORD ] && echo urgent || echo default)" \
         -d "$BODY" "${GB_NTFY_SERVER:-https://ntfy.sh}/$GB_NTFY_TOPIC" >/dev/null 2>&1 || true
fi
if [ "$loud" = 1 ] && [ -n "${GB_NOTIFY_MAIL:-}" ] && command -v mail >/dev/null 2>&1; then
    printf '%s\n' "$BODY" | mail -s "$TITLE" "$GB_NOTIFY_MAIL" >/dev/null 2>&1 || true
fi
if [ -n "${GB_NOTIFY_CMD:-}" ]; then
    # shellcheck disable=SC2086
    $GB_NOTIFY_CMD "$KIND" "$P" "$N" "$Q" "$CHECK" >/dev/null 2>&1 || true
fi
exit 0
