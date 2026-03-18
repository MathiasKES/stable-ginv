#!/bin/bash
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
echo "Activated conda env"

export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"