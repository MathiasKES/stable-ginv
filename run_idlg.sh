#!/bin/bash
#BSUB -J idlg_mask
#BSUB -q gpua10
#BSUB -n 4
#BSUB -gpu "num=1:mode=shared"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -oo gpuout/idlg/%J.out
#BSUB -eo gpuout/idlg/%J.err

module load cuda/12.8.1
source /work3/s234843/init_project.sh

cd /work3/s234843/bachelor

python iDLG_mask.py \
    --network "$NETWORK" \
    --dataset "$DATASET" \
    --mask_mode "$MASK_MODE" \
    --gradsize_topk "$TOPK" \
    --gradsize_topfrac "$TOPFRAC" \
    --iteration "$ITERATION" \
    --num_exp "$NUM_EXP" \
    --run_id "$RUN_ID"