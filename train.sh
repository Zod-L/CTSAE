#!/usr/bin/env bash

# Train
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -W ignore -m torch.distributed.launch --master_port 50130 --nproc_per_node=8 --use_env train.py \
                                   --model cls_attn_cnn_split224_4cnn_attn \
                                   --im-size 224 \
                                   --data-set IMNET \
                                   --threshold 0 \
                                   --batch-size 6 \
                                   --lr 0.001 \
                                   --num_workers 4 \
                                   --data-path ../gravityspy/train/ \
                                   --output_dir ./output/cls_attn_cnn_split224_4cnn_attn_vae \
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


