#!/usr/bin/env bash

# Train
export CUDA_VISIBLE_DEVICES=3,5,7

python -W ignore -m torch.distributed.launch --master_port 50130 --nproc_per_node=3 --use_env train.py \
                                   --model auto_encoder_no_comm_224 \
                                   --im-size 224 \
                                   --data-set IMNET \
                                   --threshold 0 \
                                   --batch-size 6 \
                                   --lr 0.001 \
                                   --num_workers 4 \
                                   --data-path ../gravityspy/split/train/ \
                                   --output_dir ./output/cnn_attn_no_comm \
                                   --epochs 401 \
                                   --save_freq 20 \
