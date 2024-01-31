#!/usr/bin/env bash

# Train
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -W ignore -m torch.distributed.launch --master_port 50130 --nproc_per_node=4 --use_env train.py \
                                   --model cnn_split224_4branch \
                                   --im-size 224 \
                                   --data-set IMNET \
                                   --threshold 0 \
                                   --batch-size 8 \
                                   --lr 0.001 \
                                   --num_workers 4 \
                                   --data-path ../gravityspy/split/train/ \
                                   --output_dir ./output/cnn_only \
                                   --epochs 201 \
                                   --save_freq 20 \




# Inference
#CUDA_VISIBLE_DEVICES=0, python main.py  --model Conformer_tiny_patch16 --eval --batch-size 64 \
#                --input-size 224 \
#                --data-set IMNET \
#                --num_workers 4 \
#                --data-path ../ImageNet_ILSVRC2012/ \
#                --epochs 100 \
#                --resume ../Conformer_tiny_patch16.pth


