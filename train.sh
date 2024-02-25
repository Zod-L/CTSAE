#!/usr/bin/env bash

# Train
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

python -W ignore -m torch.distributed.launch --master_port 50130 --nproc_per_node=7 --use_env train.py \
                                   --model cnn_split_attn \
                                   --im-size 224 \
                                   --data-set IMNET \
                                   --threshold 0 \
                                   --batch-size 4 \
                                   --lr 0.001 \
                                   --num_workers 4 \
                                   --data-path ../gravityspy/mixed_split/train/ \
                                   --output_dir ./mix_output/cnn_split_attn \
                                   --epochs 401 \
                                   --save_freq 20 \
                                   --resume mix_output/cnn_split_attn/checkpoint_140.pth
