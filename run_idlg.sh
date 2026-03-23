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

# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.05 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.1 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.15 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.2 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.25 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.3 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.35 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.4 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.45 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.5 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.55 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.6 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.65 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.7 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.75 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.8 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.85 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.9 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 0.95 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0
# python iDLG_mask.py --methods masked --mask_mode gradsize_topfrac_entries --gradsize_topfrac_entries 1 --gradsize_metric l2 --lr 1 --num_dummy 1 --iteration 1000 --num_exp 32 --network resnet18 --dataset cifar100 --run_id 0