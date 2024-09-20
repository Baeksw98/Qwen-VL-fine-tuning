#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

DIR=`pwd`

# Model path
MODEL="/data/swbaek/Projects/Korean_IC_Competition/Qwen-VL-fine-tuning/output_qwen/checkpoint-02"

# Data paths
DATA="/data/swbaek/Projects/Korean_IC_Competition/Qwen-VL-fine-tuning/data/nikluge-gips-2023-test.jsonl"
IMAGE_DIR="/data/swbaek/Projects/Korean_IC_Competition/data/nikluge-gips-2023_image"

# Output path
OUTPUT_DIR="/data/swbaek/Projects/Korean_IC_Competition/Qwen-VL-fine-tuning/inference"
OUTPUT_FILE="qwen_results_V2.jsonl"

# Run inference
CUDA_VISIBLE_DEVICES=0 python /data/swbaek/Projects/Korean_IC_Competition/Qwen-VL-fine-tuning/inference.py \
    --model_path $MODEL \
    --data_path $DATA \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --output_file $OUTPUT_FILE