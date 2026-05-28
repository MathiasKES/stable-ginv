#!/bin/bash
# interactive_jobscript.sh
#
#
# Behavior:
#   - cmds.txt is re-read from disk between every run, so you can freely add,
#     remove, or reorder lines while the script is running.
#   - Blank lines and lines starting with `#` are skipped.
#   - The next run is always the first eligible line in the current cmds.txt.
#   - Each command's stdout/stderr go to hpc/gpuout/interactive/<uid>_run.{out,err}
#     where <uid> = <YYYYMMDD_HHMMSS>_L<line-number>. The first line of the .out
#     file is the python command itself; the program's own output follows.
#   - When a run finishes the script re-reads cmds.txt and finds the FIRST line
#     equal to the just-executed cmd:
#       * rc == 0 → that line is removed from cmds.txt.
#       * rc != 0 → that line is replaced with "# FAILED: <cmd>" so it is
#                   kept for inspection but skipped in future iterations.
#       * line not found (user deleted/edited it) → script prints
#                   "[interactive_jobscript] Line not found, finished: [<uid>]"
#                   and continues.
#   - On SIGINT/SIGTERM/SIGHUP the current job is killed and the script exits;
#     cmds.txt is not modified for the interrupted run.

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

# Find the first eligible (non-blank, non-comment) line in cmds.txt as it is
# RIGHT NOW. Sets NEXT_CMD and NEXT_LINE_NO; returns 1 if no eligible line.
find_next_cmd() {
    NEXT_CMD=""
    NEXT_LINE_NO=0
    local lineno=0
    local line trimmed
    while IFS= read -r line || [[ -n "$line" ]]; do
        lineno=$((lineno + 1))
        trimmed="${line#"${line%%[![:space:]]*}"}"
        if [[ -z "$trimmed" || "$trimmed" == \#* ]]; then
            continue
        fi
        NEXT_LINE_NO=$lineno
        NEXT_CMD=$line
        return 0
    done < "$CMDS_FILE"
    return 1
}

# Re-read cmds.txt and operate on the FIRST line equal to $1 (target):
#   action=delete   → drop the line
#   action=fail     → replace the line with "# FAILED: <target>"
# Returns 0 if the line was found, 1 otherwise.
modify_cmds() {
    local target=$1 action=$2
    local tmp
    tmp="$(mktemp "${CMDS_FILE}.XXXXXX")"
    local found=1
    local line
    while IFS= read -r line || [[ -n "$line" ]]; do
        if (( found == 1 )) && [[ "$line" == "$target" ]]; then
            found=0
            case "$action" in
                delete) ;;  # skip writing the line
                fail)   printf '# FAILED: %s\n' "$target" ;;
            esac
        else
            printf '%s\n' "$line"
        fi
    done < "$CMDS_FILE" > "$tmp"
    mv "$tmp" "$CMDS_FILE"
    return $found
}

while true; do
    if ! find_next_cmd; then
        echo "[interactive_jobscript] queue complete."
        exit 0
    fi

    cmd=$NEXT_CMD
    lineno=$NEXT_LINE_NO
    ts="$(date +%Y%m%d_%H%M%S)"
    uid="${ts}_L$(printf '%04d' "$lineno")"
    out_file="$OUT_DIR/${uid}_run.out"
    err_file="$OUT_DIR/${uid}_run.err"

    echo "[interactive_jobscript] [$uid] running: $cmd"

    printf '%s\n' "$cmd" > "$out_file"
    : > "$err_file"

    bash -c "$cmd" >> "$out_file" 2>> "$err_file" &
    CURRENT_PID=$!
    wait "$CURRENT_PID"
    rc=$?
    CURRENT_PID=""

    if (( INTERRUPTED )); then
        exit 130
    fi

    if (( rc == 0 )); then
        if modify_cmds "$cmd" delete; then
            echo "[interactive_jobscript] [$uid] OK"
        else
            echo "[interactive_jobscript] Line not found, finished: [$uid]"
        fi
    else
        if modify_cmds "$cmd" fail; then
            echo "[interactive_jobscript] [$uid] FAILED (rc=$rc) — marked as # FAILED in cmds.txt"
        else
            echo "[interactive_jobscript] Line not found, finished: [$uid] (rc=$rc)"
        fi
    fi
done
