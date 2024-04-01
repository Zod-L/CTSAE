#!/usr/bin/env bash

# Train
export CUDA_VISIBLE_DEVICES=1,2,6,7

python -W ignore -m torch.distributed.launch --master_port 50130 --nproc_per_node=4 --use_env train.py \
                                   --model cnn_share_attn \
                                   --im-size 224 \
                                   --data-set IMNET \
                                   --threshold 0 \
                                   --batch-size 4 \
                                   --lr 0.001 \
                                   --num_workers 8 \
                                   --data-path ../data/gs_2.0/train/ \
                                   --output_dir ./gs2.0_output/cnn_share_attn \
                                   --epochs 101 \
                                   --save_freq 5 \
