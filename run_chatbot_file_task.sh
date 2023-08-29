#!/bin/bash
cd ./third_party
if [ $# -ne 2 ]; then
  echo "You should enter 2 arguments"
fi

# query_file=$1
# ref_dir=$2
# task=$3


dataset=$1
task=$2
data_dir="./output_dir"
output_dir=$data_dir'/'$dataset'_'$task

device=0
port=11000

if [ "$dataset"x = "DD"x ]
then
    query_file="./data/daily_dialog/formality_query.txt"
fi

if [ "$dataset"x = "BST"x ]
then
    query_file="./data/BST/query.txt"
fi

echo "query_file: "${query_file}



# GPT-J-6B
output_folder_path=$output_dir"/response_${task}_result_${dataset}_5shot_gpt-j-6b"
model_name_or_path="EleutherAI/gpt-j-6b"

echo 'output_path: '$output_folder_path

CUDA_VISIBLE_DEVICES=$device \
  deepspeed --master_port $port examples/chatbot_file.py \
      --deepspeed configs/ds_config_chatbot.json \
      --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input_text} ###${task} Response: "   \
      --end_string "#" \
      --use_ram_optimized_load False \
      --input_text_file ${query_file} \
      --output_folder_path ${output_folder_path} \
      --model_name_or_path ${model_name_or_path} \



# LLaMA-7B
output_folder_path=$output_dir"/response_${task}_result_${dataset}_5shot_llama-7b"
model_name_or_path="pinkmanlove/llama-7b-hf"

echo 'output_path: '$output_folder_path

CUDA_VISIBLE_DEVICES=$device \
  deepspeed --master_port $port examples/chatbot_file.py \
      --deepspeed configs/ds_config_chatbot.json \
      --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input_text} ###${task} Response: "   \
      --end_string "#" \
      --use_ram_optimized_load False \
      --input_text_file ${query_file} \
      --output_folder_path ${output_folder_path} \
      --model_name_or_path ${model_name_or_path} \


# LLaMA-13B
output_folder_path=$output_dir"/response_${task}_result_${dataset}_5shot_llama-13b"
model_name_or_path="pinkmanlove/llama-13b-hf"

echo 'output_path: '$output_folder_path

CUDA_VISIBLE_DEVICES=$device \
  deepspeed --master_port $port examples/chatbot_file.py \
      --deepspeed configs/ds_config_chatbot.json \
      --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input_text} ###${task} Response: "   \
      --end_string "#" \
      --use_ram_optimized_load False \
      --input_text_file ${query_file} \
      --output_folder_path ${output_folder_path} \
      --model_name_or_path ${model_name_or_path} \



# # Vicuna-7B
# output_folder_path=$output_dir"/response_${task}_result_${dataset}_5shot_vicuna-7b"
# model_name_or_path="PATH_TO_Vicuna7b"

# echo 'output_path: '$output_folder_path

# CUDA_VISIBLE_DEVICES=$device \
#   deepspeed --master_port $port examples/chatbot_file.py \
#       --deepspeed configs/ds_config_chatbot.json \
#       --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input_text} ###${task} Response: "   \
#       --end_string "#" \
#       --use_ram_optimized_load False \
#       --input_text_file ${query_file} \
#       --output_folder_path ${output_folder_path} \
#       --model_name_or_path ${model_name_or_path} \


# # Vicuna-13B
# output_folder_path=$output_dir"/response_${task}_result_${dataset}_5shot_vicuna-13b"
# model_name_or_path="PATH_TO_Vicuna13b"

# echo 'output_path: '$output_folder_path

# CUDA_VISIBLE_DEVICES=$device \
#   deepspeed --master_port $port examples/chatbot_file.py \
#       --deepspeed configs/ds_config_chatbot.json \
#       --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input_text} ###${task} Response: "   \
#       --end_string "#" \
#       --use_ram_optimized_load False \
#       --input_text_file ${query_file} \
#       --output_folder_path ${output_folder_path} \
#       --model_name_or_path ${model_name_or_path} \


# Koala-7B
output_folder_path=$output_dir"/response_${task}_result_${dataset}_5shot_koala-7B"
model_name_or_path="TheBloke/koala-7B-HF"

echo 'output_path: '$output_folder_path

CUDA_VISIBLE_DEVICES=$device \
  deepspeed --master_port $port examples/chatbot_file.py \
      --deepspeed configs/ds_config_chatbot.json \
      --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input_text} ###${task} Response: "   \
      --end_string "#" \
      --use_ram_optimized_load False \
      --input_text_file ${query_file} \
      --output_folder_path ${output_folder_path} \
      --model_name_or_path ${model_name_or_path} \



# Koala-13B
output_folder_path=$output_dir"/response_${task}_result_${dataset}_5shot_koala-13B"
model_name_or_path="TheBloke/koala-13B-HF"

echo 'output_path: '$output_folder_path

CUDA_VISIBLE_DEVICES=$device \
  deepspeed --master_port $port examples/chatbot_file.py \
      --deepspeed configs/ds_config_chatbot.json \
      --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input_text} ###${task} Response: "   \
      --end_string "#" \
      --use_ram_optimized_load False \
      --input_text_file ${query_file} \
      --output_folder_path ${output_folder_path} \
      --model_name_or_path ${model_name_or_path} \



# Falcon-7B
output_folder_path=$output_dir"/response_${task}_result_${dataset}_5shot_falcon-7b"
model_name_or_path="tiiuae/falcon-7b"

echo 'output_path: '$output_folder_path

CUDA_VISIBLE_DEVICES=$device \
  deepspeed --master_port $port examples/chatbot_file.py \
      --deepspeed configs/ds_config_chatbot.json \
      --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input_text} ###${task} Response: "   \
      --end_string "#" \
      --use_ram_optimized_load False \
      --input_text_file ${query_file} \
      --output_folder_path ${output_folder_path} \
      --model_name_or_path ${model_name_or_path} \



# Falcon-7B-Instruct
output_folder_path=$output_dir"/response_${task}_result_${dataset}_5shot_falcon-7b-instruct"
model_name_or_path="tiiuae/falcon-7b-instruct"

echo 'output_path: '$output_folder_path

CUDA_VISIBLE_DEVICES=$device \
  deepspeed --master_port $port examples/chatbot_file.py \
      --deepspeed configs/ds_config_chatbot.json \
      --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input_text} ###${task} Response: "   \
      --end_string "#" \
      --use_ram_optimized_load False \
      --input_text_file ${query_file} \
      --output_folder_path ${output_folder_path} \
      --model_name_or_path ${model_name_or_path} \




# RedPajama-7B-Instruct
output_folder_path=$output_dir"/response_${task}_result_${dataset}_5shot_RedPajama-7b-instruct"
model_name_or_path="togethercomputer/RedPajama-INCITE-7B-Instruct"

echo 'output_path: '$output_folder_path

CUDA_VISIBLE_DEVICES=$device \
  deepspeed --master_port $port examples/chatbot_file.py \
      --deepspeed configs/ds_config_chatbot.json \
      --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input_text} ###${task} Response: "   \
      --end_string "#" \
      --use_ram_optimized_load False \
      --input_text_file ${query_file} \
      --output_folder_path ${output_folder_path} \
      --model_name_or_path ${model_name_or_path} \




# Alpaca-7B
output_folder_path=$output_dir"/response_${task}_result_${dataset}_5shot_Alpaca-7b"
model_name_or_path="chavinlo/alpaca-native"

echo 'output_path: '$output_folder_path

CUDA_VISIBLE_DEVICES=$device \
  deepspeed --master_port $port examples/chatbot_file.py \
      --deepspeed configs/ds_config_chatbot.json \
      --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input_text} ###${task} Response: "   \
      --end_string "#" \
      --use_ram_optimized_load False \
      --input_text_file ${query_file} \
      --output_folder_path ${output_folder_path} \
      --model_name_or_path ${model_name_or_path} \

