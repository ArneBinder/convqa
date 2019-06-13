#!/bin/sh

# Train (fine-tune) GPT on GPUs (e.g. with converted CoQA data).

DATASET=/home/abinder/datasets/CoQA/coqa_converted_persona.json

# use first argument as GPU devices, if available
if [ -n "$1" ]; then
    export CUDA_VISIBLE_DEVICES=$1
    echo "use CUDA_VISIBLE_DEVICES=$1"
else
    echo "WARNING: CUDA_VISIBLE_DEVICES is not set!"
fi

CMD="python train.py"
echo "execute: $CMD"

$CMD \
--dataset_path $DATASET \
--epochs 1