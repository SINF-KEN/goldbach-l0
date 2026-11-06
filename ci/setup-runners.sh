#!/usr/bin/env bash
# setup-runners.sh — register the two self-hosted GitHub Actions runners that
# .github/workflows/*.yml run on, and run them as systemd USER services.
# Run once on the production machine, as the user that owns the campaign:
#
#   bash ci/setup-runners.sh            # register + install + start
#   bash ci/setup-runners.sh remove     # stop, unregister, uninstall
#
#   goldbach-cpu : builds and GPU-free tests (kernels, host, static, full)
#   goldbach-gpu : the gpu job only (pauses run_range; one job at a time)
#
# Needs: gh (logged in, admin on the repo), curl. No sudo. Also installs the
# pinned shellcheck / actionlint into ~/.local/bin for the static job.
set -euo pipefail
REPO=${GB_RUNNER_REPO:-SINF-KEN/goldbach-l0-tidy}
UNITS=$HOME/.config/systemd/user

if [ "${1:-}" = remove ]; then
    for r in cpu gpu; do
        systemctl --user disable --now "actions-runner-$r.service" 2>/dev/null || true
        if [ -x "$HOME/actions-runner-$r/config.sh" ]; then
            T=$(gh api -X POST "repos/$REPO/actions/runners/remove-token" --jq .token)
            (cd "$HOME/actions-runner-$r" && ./config.sh remove --token "$T") || true
        fi
        rm -f "$UNITS/actions-runner-$r.service"
    done
    systemctl --user daemon-reload
    echo "runners removed (directories ~/actions-runner-{cpu,gpu} kept)"
    exit 0
fi

mkdir -p "$HOME/.local/bin" "$UNITS"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
command -v shellcheck >/dev/null || {
    curl -sSfL https://github.com/koalaman/shellcheck/releases/download/v0.10.0/shellcheck-v0.10.0.linux.x86_64.tar.xz | tar xJ -C "$tmp"
    install -m755 "$tmp/shellcheck-v0.10.0/shellcheck" "$HOME/.local/bin/shellcheck"; }
command -v actionlint >/dev/null || {
    curl -sSfL https://github.com/rhysd/actionlint/releases/download/v1.7.7/actionlint_1.7.7_linux_amd64.tar.gz | tar xz -C "$tmp" actionlint
    install -m755 "$tmp/actionlint" "$HOME/.local/bin/actionlint"; }

V=$(gh api repos/actions/runner/releases/latest --jq .tag_name | tr -d v)
for r in cpu gpu; do
    d=$HOME/actions-runner-$r
    mkdir -p "$d"
    [ -x "$d/config.sh" ] || curl -sSfL "https://github.com/actions/runner/releases/download/v$V/actions-runner-linux-x64-$V.tar.gz" | tar xz -C "$d"
    if [ ! -f "$d/.runner" ]; then
        T=$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token)
        (cd "$d" && ./config.sh --unattended --url "https://github.com/$REPO" --token "$T" \
            --name "$(hostname)-$r" --labels "goldbach-$r" --work _work --replace)
    fi
    cat > "$UNITS/actions-runner-$r.service" <<EOF
[Unit]
Description=GitHub Actions runner goldbach-$r ($REPO)
After=network-online.target

[Service]
WorkingDirectory=%h/actions-runner-$r
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin
ExecStart=%h/actions-runner-$r/run.sh
Restart=always
RestartSec=10
# a job's processes live in this unit's cgroup; the production campaign runs
# in its own unit (goldbach-range) and is never killed with a job
KillMode=control-group

[Install]
WantedBy=default.target
EOF
done
systemctl --user daemon-reload
systemctl --user enable --now actions-runner-cpu.service actions-runner-gpu.service
sleep 5
gh api "repos/$REPO/actions/runners" --jq '.runners[] | "  \(.name)  \(.status)  \([.labels[].name] | join(","))"'
cat <<'EOF'

Next:
  - pushes stop being tested twice:   echo '{"watch_include": []}' > ci-state/config.json
  - keep the CI daemon for recovery / count checks:
        python3 ci/goldbach_ci.py install-service --enable
  - runners + daemon without a login session (reboots):   loginctl enable-linger "$USER"
EOF
