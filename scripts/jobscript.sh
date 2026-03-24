#!/bin/bash

COMMANDS_FILE="scripts/cmds.txt"

while IFS= read -r CMD || [[ -n "$CMD" ]]; do
  [[ -z "$CMD" ]] && continue  # skip empty lines

  # Extract fields for job name
  METHODS=$(echo "$CMD"          | grep -oP '(?<=--methods )\S+')
  MASK_MODE=$(echo "$CMD"        | grep -oP '(?<=--mask_mode )\S+')
  GRADSIZE_TOPFRAC=$(echo "$CMD" | grep -oP '(?<=--gradsize_topfrac )\S+')
  NUM_EXP=$(echo "$CMD"          | grep -oP '(?<=--num_exp )\S+')

  METHODS=${METHODS:-idlg}
  MASK_MODE=${MASK_MODE:-gradsize_topfrac}
  GRADSIZE_TOPFRAC=${GRADSIZE_TOPFRAC:-0.5}
  NUM_EXP=${NUM_EXP:-16}

  JOB_NAME="${METHODS}_${MASK_MODE}_${GRADSIZE_TOPFRAC}_${NUM_EXP}"

  # Capture bsub output to extract job ID
  BSUB_OUT=$(bsub <<EOF
#!/bin/bash
#BSUB -J $JOB_NAME
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=shared"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -oo /work3/s234843/bachelor/gpuout/idlg/%J.out
#BSUB -eo /work3/s234843/bachelor/gpuout/idlg/%J.err

module load cuda/12.8.1

source /work3/s234843/bachelor/init.sh

$CMD
EOF
)

  # bsub prints: "Job <12345> is submitted to queue <gpuv100>."
  JOB_ID=$(echo "$BSUB_OUT" | grep -oP '(?<=Job <)\d+')

  echo "Submitted job $JOB_ID [$JOB_NAME]: $CMD"

done < "$COMMANDS_FILE"