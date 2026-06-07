#!/bin/bash

COMMANDS_FILE="scripts/cmds.txt"

# Find the first eligible (non-blank, non-comment) line in COMMANDS_FILE as it
# is RIGHT NOW. Sets NEXT_CMD; returns 1 if there is no eligible line.
find_next_cmd() {
  NEXT_CMD=""
  local line trimmed
  while IFS= read -r line || [[ -n "$line" ]]; do
    trimmed="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$trimmed" || "$trimmed" == \#* ]] && continue
    NEXT_CMD=$line
    return 0
  done < "$COMMANDS_FILE"
  return 1
}

# Re-read COMMANDS_FILE and replace the FIRST line equal to $1 with $2.
# Returns 0 if the line was found, 1 otherwise.
replace_cmd() {
  local target=$1 replacement=$2 tmp found=1 line
  tmp="$(mktemp "${COMMANDS_FILE}.XXXXXX")"
  while IFS= read -r line || [[ -n "$line" ]]; do
    if (( found == 1 )) && [[ "$line" == "$target" ]]; then
      found=0
      printf '%s\n' "$replacement"
    else
      printf '%s\n' "$line"
    fi
  done < "$COMMANDS_FILE" > "$tmp"
  mv "$tmp" "$COMMANDS_FILE"
  return $found
}

while find_next_cmd; do
  CMD=$NEXT_CMD

  # Extract fields for job name
  METHODS=$(echo "$CMD"          | grep -oP '(?<=--methods )\S+')
  MASK_MODE=$(echo "$CMD"        | grep -oP '(?<=--mask_mode )\S+')
  GRADSIZE_TOPFRAC=$(echo "$CMD" | grep -oP '(?<=--gradsize_topfrac )\S+')
  NUM_EXP=$(echo "$CMD"          | grep -oP '(?<=--num_exp )\S+')
  NETWORK=$(echo "$CMD"          | grep -oP '(?<=--network )\S+')

  METHODS=${METHODS:-idlg} # Default to idlg
  METHODS=${METHODS:0:1} # Get first character
  MASK_MODE=${MASK_MODE:-gradsize_topfrac}
  GRADSIZE_TOPFRAC=${GRADSIZE_TOPFRAC:-0.5}
  NUM_EXP=${NUM_EXP:-16}
  NETWORK=${NETWORK:-network}

  JOB_NAME="${MASK_MODE}_${NETWORK}${METHODS}"

  # Capture bsub output to extract job ID
  BSUB_OUT=$(bsub <<EOF
#!/bin/bash
#BSUB -J $JOB_NAME
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 12:00
#BSUB -oo /work3/s234843/bachelor/gpuout/idlg/%J.out
#BSUB -eo /work3/s234843/bachelor/gpuout/idlg/%J.err

module load cuda/12.8.1

source /work3/s234843/bachelor/init.sh

nvidia-smi

$CMD
EOF
)

  # bsub prints: "Job <12345> is submitted to queue <gpuv100>."
  JOB_ID=$(echo "$BSUB_OUT" | grep -oP '(?<=Job <)\d+')

  # If submission failed, leave the line untouched and stop (avoids an
  # infinite loop re-submitting the same line).
  if [[ -z "$JOB_ID" ]]; then
    echo "submission failed: $CMD" >&2
    echo "$BSUB_OUT" >&2
    exit 1
  fi

  replace_cmd "$CMD" "# $JOB_ID : $CMD"

  # echo "Submitted job $JOB_ID [$JOB_NAME]: $CMD"
  echo "$JOB_ID"

done
