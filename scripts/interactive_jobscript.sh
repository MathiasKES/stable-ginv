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
#     where <uid> = <YYYYMMDD_HHMMSS>_L<line-number>_<rand4>, e.g.
#     20260528_124945_L0001_a3f9. The first line of the .out
#     file is the python command itself; the program's own output follows.
#   - When a run STARTS the script replaces its line with a "# RUNNING: <cmd>"
#     comment, so readers of cmds.txt can see what is in flight (and the line is
#     skipped while it runs). When the run finishes that RUNNING line is
#     replaced with the DONE/FAILED comment below.
#   - When a run finishes the script re-reads cmds.txt and finds the FIRST line
#     equal to the "# RUNNING: <cmd>" marker:
#       * rc == 0 → that line is replaced with a single-line comment so it is
#                   kept for inspection but skipped in future iterations.
#                   Format:
#                     # DONE [N] (dur=Ns): <cmd>
#       * rc != 0 → that line is replaced with a single-line diagnostic
#                   comment so it is kept for inspection but skipped in
#                   future iterations. Format:
#                     # FAILED: [rc=N] [uid=…] [dur=Ns] [tail=…] <cmd>
#                   The uid lets you locate the matching
#                   hpc/gpuout/interactive/<uid>_run.{out,err} files;
#                   tail is the last non-empty stderr line, truncated.
#       * line not found (user deleted/edited it) → script prints
#                   "[interactive_jobscript] Line not found, finished: [<uid>]"
#                   and continues.
#   - On SIGINT/SIGTERM/SIGHUP the current job is killed and the script exits;
#     the interrupted run is left as its "# RUNNING: <cmd>" marker in cmds.txt
#     (uncomment it to re-queue).

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
#   mode=delete   → drop the line
#   mode=replace  → emit $3 instead of the line (used for "# FAILED: …")
# Returns 0 if the line was found, 1 otherwise.
modify_cmds() {
    local target=$1 mode=$2 replacement=${3-}
    local tmp
    tmp="$(mktemp "${CMDS_FILE}.XXXXXX")"
    local found=1
    local line
    while IFS= read -r line || [[ -n "$line" ]]; do
        if (( found == 1 )) && [[ "$line" == "$target" ]]; then
            found=0
            case "$mode" in
                delete)  ;;  # skip writing the line
                replace) printf '%s\n' "$replacement" ;;
            esac
        else
            printf '%s\n' "$line"
        fi
    done < "$CMDS_FILE" > "$tmp"
    mv "$tmp" "$CMDS_FILE"
    return $found
}

# Last non-empty line of $1 (the .err file), single-line, truncated to 200
# chars (UTF-8 ellipsis appended on truncation). Empty string if none.
err_tail() {
    local f=$1 line=""
    [[ -s "$f" ]] || { printf ''; return; }
    line=$(grep -v '^[[:space:]]*$' "$f" 2>/dev/null | tail -n 1)
    line=${line//$'\r'/}
    line=${line//$'\n'/ }
    if (( ${#line} > 200 )); then
        line="${line:0:200}…"
    fi
    printf '%s' "$line"
}

while true; do
    if ! find_next_cmd; then
        echo "[interactive_jobscript] queue complete."
        exit 0
    fi

    cmd=$NEXT_CMD
    lineno=$NEXT_LINE_NO
    ts="$(date +%Y%m%d_%H%M%S)"
    rand="$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 4)"
    uid="${ts}_L$(printf '%04d' "$lineno")_${rand}"
    out_file="$OUT_DIR/${uid}_run.out"
    err_file="$OUT_DIR/${uid}_run.err"

    echo "[interactive_jobscript] [$uid] running: $cmd"

    printf '%s\n' "$cmd" > "$out_file"
    : > "$err_file"

    # Mark the line as RUNNING right away so a reader of cmds.txt can see what
    # is in flight. find_next_cmd skips comment lines, so this also prevents the
    # same line from being picked up again while it runs. When the run finishes
    # the RUNNING line is replaced with the DONE/FAILED comment below.
    running_comment="# RUNNING: $cmd"
    modify_cmds "$cmd" replace "$running_comment" || true

    start_ts=$(date +%s)
    bash -c "$cmd" >> "$out_file" 2>> "$err_file" &
    CURRENT_PID=$!
    wait "$CURRENT_PID"
    rc=$?
    CURRENT_PID=""
    dur=$(( $(date +%s) - start_ts ))

    if (( INTERRUPTED )); then
        exit 130
    fi

    if (( rc == 0 )); then
        done_comment="# DONE [rc=$rc] [uid=$uid] (dur=${dur}s): $cmd"
        if modify_cmds "$running_comment" replace "$done_comment"; then
            echo "[interactive_jobscript] [$uid] OK (dur=${dur}s) — marked as # DONE in cmds.txt"
        else
            echo "[interactive_jobscript] Line not found, finished: [$uid] (dur=${dur}s)"
        fi
    else
        tail_line=$(err_tail "$err_file")
        if [[ -n "$tail_line" ]]; then
            fail_comment="# FAILED: [rc=$rc] [uid=$uid] [dur=${dur}s] [tail=${tail_line}] $cmd"
        else
            fail_comment="# FAILED: [rc=$rc] [uid=$uid] [dur=${dur}s] $cmd"
        fi
        if modify_cmds "$running_comment" replace "$fail_comment"; then
            echo "[interactive_jobscript] [$uid] FAILED (rc=$rc, dur=${dur}s) — marked as # FAILED in cmds.txt"
        else
            echo "[interactive_jobscript] Line not found, finished: [$uid] (rc=$rc, dur=${dur}s)"
        fi
    fi
done
