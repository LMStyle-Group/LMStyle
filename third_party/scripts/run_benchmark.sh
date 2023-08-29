#!/bin/bash

if [ "$1" == "-h" -o "$1" == "--help" ]; then
  help_message="./$(basename $0)"
  help_message+=" --dataset_name DATASET_NAME"
  help_message+=" --model_name_or_path MODEL_NAME_OR_PATH"
  echo ${help_message} 1>&2
  exit 1
fi

exp_id=benchmarking
extra_args="--dataset_name gpt4_en_eval --model_name_or_path gpt2"
if [ $# -ge 1 ]; then
  extra_args="$@"
fi
log_dir=output_dir/${exp_id}_nll

mkdir -p ${log_dir}

CUDA_VISIBLE_DEVICES=6 \
  deepspeed --master_port 11001 examples/benchmarking.py \
  --use_ram_optimized_load 0 \
  --deepspeed examples/ds_config.json \
  --metric nll \
  --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input} ###Response:"   \
  ${extra_args} \
  | tee ${log_dir}/benchmark.log \
  2> ${log_dir}/benchmark.err
