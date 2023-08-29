#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type medmcqa \
    --model_name_or_path gpt2-large \
    --dataset_path /home/jianlinchen/code/LM_FLOW_Research/LMFlow/data/yelp \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric neg_log_likelihood