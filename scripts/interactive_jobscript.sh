#!/bin/bash
# interactive_jobscript.sh
#
#
# Behavior:
#   - Blank lines and lines starting with `#` are skipped.
#   - Each command's stdout/stderr go to hpc/gpuout/interactive/<uid>_run.{out,err}
#     where <uid> = <YYYYMMDD_HHMMSS>_L<line-number>. The first line of the .out
#     file is the python command itself; the program's own output follows.
#   - On successful exit (rc=0) the command is removed from cmds.txt.
#   - On failure the command stays in cmds.txt and the queue continues.
#   - On SIGINT/SIGTERM/SIGHUP the current job is killed and the script exits;
#     remaining commands (including the interrupted one) stay in cmds.txt.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CMDS_FILE="$SCRIPT_DIR/cmds.txt"
OUT_DIR="$REPO_ROOT/hpc/gpuout/interactive"

mkdir -p "$OUT_DIR"

# shellcheck disable=SC1091
# source "$SCRIPT_DIR/init.sh"

cd "$REPO_ROOT"

CURRENT_PID=""
INTERRUPTED=0

cleanup() {
    INTERRUPTED=1
    if [[ -n "$CURRENT_PID" ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
        echo
        echo "[interactive_jobscript] interrupted — killing PID $CURRENT_PID and stopping queue."
        kill -TERM "$CURRENT_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$CURRENT_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$CURRENT_PID" 2>/dev/null || true
    fi
    exit 130
}
trap cleanup INT TERM HUP

rewrite_cmds() {
    local tmp
    tmp="$(mktemp "${CMDS_FILE}.XXXXXX")"
    local i
    for i in "${!LINES[@]}"; do
        if [[ -z "${DONE[$i]:-}" ]]; then
            printf '%s\n' "${LINES[$i]}"
        fi
    done > "$tmp"
    mv "$tmp" "$CMDS_FILE"
}

mapfile -t LINES < "$CMDS_FILE"
declare -A DONE=()

for i in "${!LINES[@]}"; do
    line="${LINES[$i]}"
    trimmed="${line#"${line%%[![:space:]]*}"}"
    if [[ -z "$trimmed" || "$trimmed" == \#* ]]; then
        continue
    fi

    ts="$(date +%Y%m%d_%H%M%S)"
    uid="${ts}_L$(printf '%04d' $((i + 1)))"
    out_file="$OUT_DIR/${uid}_run.out"
    err_file="$OUT_DIR/${uid}_run.err"

    echo "[interactive_jobscript] [$uid] running: $line"

    printf '%s\n' "$line" > "$out_file"
    : > "$err_file"

    bash -c "$line" >> "$out_file" 2>> "$err_file" &
    CURRENT_PID=$!
    wait "$CURRENT_PID"
    rc=$?
    CURRENT_PID=""

    if (( INTERRUPTED )); then
        exit 130
    fi

    if (( rc == 0 )); then
        DONE[$i]=1
        rewrite_cmds
        echo "[interactive_jobscript] [$uid] OK"
    else
        echo "[interactive_jobscript] [$uid] FAILED (rc=$rc) — keeping in cmds.txt, continuing"
    fi
done

echo "[interactive_jobscript] queue complete."
