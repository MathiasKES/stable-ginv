#!/bin/bash
#BSUB -J idlg_mask
#BSUB -q gpua10
#BSUB -n 8
#BSUB -gpu "num=2:mode=shared"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -oo gpuout/idlg/%J.out
#BSUB -eo gpuout/idlg/%J.err

module load cuda/12.8.1
export CUDA_VISIBLE_DEVICES=0,1,2,3

__conda_setup="$('/work3/s234843/bin/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/work3/s234843/bin/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/work3/s234843/bin/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/work3/s234843/bin/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate stable-ginv

cd /zhome/b6/5/204798/stable-ginv

python iDLG_mask.py \
    --network "$NETWORK" \
    --dataset "$DATASET" \
    --mask_mode "$MASK_MODE" \
    --prefixes "$PREFIXES" \
    --gradsize_topk "$TOPK" \
    --gradsize_topfrac "$TOPFRAC" \
    --gradsize_threshold "$THRESHOLD" \
    --gradsize_metric "$METRIC" \
    --lr "$LR" \
    --num_dummy "$NUM_DUMMY" \
    --iteration "$ITERATION" \
    --num_exp "$NUM_EXP" \
    --run_id "$RUN_ID"