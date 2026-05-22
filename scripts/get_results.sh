#!/bin/bash
# run_commands.sh — run each command only if its source completed; else emit a blank line
# Usage: ./run_commands.sh commands.txt
#    or: cat commands.txt | ./run_commands.sh

while IFS= read -r cmd || [ -n "$cmd" ]; do
    src="${cmd%%|*}"          # the "cat /path/to/NNNN.out " part, before the first pipe
    if bash -c "$src" </dev/null 2>/dev/null | grep -q "Successfully completed"; then
        echo "$(bash -c "$cmd" </dev/null 2>/dev/null)"
    else
        echo
    fi
done < "${1:-/dev/stdin}"
