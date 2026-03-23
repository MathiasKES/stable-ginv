#!/bin/bash

COMMANDS_FILE="cmds.txt"

while IFS= read -r CMD || [[ -n "$CMD" ]]; do
  [[ -z "$CMD" ]] && continue  # skip empty lines

  bsub <<EOF
#!/bin/bash
#BSUB -J idlg_mask
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=shared"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -oo /work3/s234843/bachelor/gpuout/idlg/%J.out
#BSUB -eo /work3/s234843/bachelor/gpuout/idlg/%J.err

module load cuda/12.8.1
export CUDA_VISIBLE_DEVICES=0,1,2,3

__conda_setup="\$('/work3/s234843/bin/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ \$? -eq 0 ]; then
    eval "\$__conda_setup"
else
    if [ -f "/work3/s234843/bin/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/work3/s234843/bin/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/work3/s234843/bin/miniconda3/bin:\$PATH"
    fi
fi
unset __conda_setup

conda activate stable-ginv

cd /zhome/b6/5/204798/stable-ginv

$CMD
EOF

  echo "Submitted: $CMD"
done < "$COMMANDS_FILE"