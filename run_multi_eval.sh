#!/bin/bash
cd ./third_party
if [ $# -ne 1 ]; then
  echo "You should enter 1 arguments including —-hyp_dir"
fi

# query_file=$1
# ref_dir=$2
# task=$3
hyp_dir=$1

base=${hyp_dir##*/} 
task=${base##*_} 
dataset=${base%_*}
log_path=./output_dir/result_summary.txt
echo "Dataset: "$dataset >> $log_path
echo "Task: "$task >> $log_path

if [ "$dataset"x = "DD"x ]
then
    query_file="./data/daily_dialog/formality_query.txt"
    ref_dir="./data/daily_dialog/ref"
fi

if [ "$dataset"x = "BST"x ]
then
    query_file="./data/BST/query.txt"
    ref_dir="./data/BST/ref"
fi

for file in `ls $hyp_dir`   
do
    hyp_file=$hyp_dir"/"$file"/response.txt"
    output_file=${hyp_file%/*}"/test.1.json"
    echo "hyp_file: "$hyp_file
    echo "output_file: "$output_file
    model=${file##*_} 
    echo -e "\n""model: "$model >> $log_path

    # examples/get_data_pair.py --query_file ${query_file} --response_file ${hyp_file} --output_file ${output_file}

    CUDA_VISIBLE_DEVICES=0 \
        deepspeed --master_port 11000 examples/evaluate.py \
        --answer_type medmcqa \
        --model_name_or_path bigscience/bloom-7b1  \
        --dataset_path ${hyp_file%/*} \
        --deepspeed examples/ds_config.json \
        --inference_batch_size_per_device 1 \
        --metric neg_log_likelihood \
        --output_file ${output_file} \
        --prompt_structure "###Speaker: {input} ###${task} Response:" \
        --use_ram_optimized_load False \
        # --arch_type "encoder_decoder"


    examples/tst_eval.py --hyp_file ${hyp_file} --ref_dir ${ref_dir} --output_file ${output_file} --metric BLEU
    examples/tst_eval.py --hyp_file ${hyp_file} --ref_dir ${ref_dir} --output_file ${output_file} --metric SBERT
    examples/tst_eval.py --hyp_file ${hyp_file} --ref_dir ${ref_dir} --output_file ${output_file} --metric ${task}
    # examples/tst_eval.py --hyp_file ${hyp_file} --ref_dir ${ref_dir} --output_file ${output_file} --metric ChatGPT --api_token_path PATH_TO_TOKEN

    

done

echo -e "\n" >> $log_path


