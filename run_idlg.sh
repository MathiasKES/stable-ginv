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

conda activate stable-ginv

cd /zhome/b6/5/204798/stable-ginv

NETWORK="${NETWORK:-resnet18}"
DATASET="${DATASET:-cifar100}"
METHODS="${METHODS:-idlg}"
MASK_MODE="${MASK_MODE:-gradsize_topfrac}"
PREFIXES="${PREFIXES:-conv1,layer1,layer2,layer3}"
TOPK="${TOPK:-20}"
TOPFRAC="${TOPFRAC:-0.6}"
THRESHOLD="${THRESHOLD:-0.0}"
METRIC="${METRIC:-l2}"
LR="${LR:-1}"
NUM_DUMMY="${NUM_DUMMY:-1}"
ITERATION="${ITERATION:-1000}"
NUM_EXP="${NUM_EXP:-100}"
RUN_ID="${RUN_ID:-0}"

python iDLG_mask.py \
    --network "$NETWORK" \
    --dataset "$DATASET" \
    --mask_mode "$MASK_MODE" \
    --methods "$METHODS" \
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