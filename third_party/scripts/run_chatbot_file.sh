#!/bin/bash

key_args="--input_text_file data/GYAFC/test.1 --output_folder_path output_dir/response_result --model_name_or_path gpt2"
if [ $# -ge 1 ]; then
  key_args="$@"
fi

CUDA_VISIBLE_DEVICES=0 \
  deepspeed --master_port 11000 examples/chatbot_file.py \
      --deepspeed configs/ds_config_chatbot.json \
      --prompt_structure "###Speaker: Frank ’ s getting married , do you believe this ?  ###Response:  Is he really ? ###Speaker: OK . Come back into the classroom , class .  ###Response:  Does the class start again , Mam ?  ###Speaker: Do you have any hobbies ?   ###Response:   Yes , I like collecting things .  ###Speaker: Jenny , what's wrong with you ? Why do you keep weeping like that ?  ###Response:  Mary told me that she had seen you with John last night . I got to know the fact that you are playing the field . ###Speaker: What a nice day !   ###Response:   yes . How about going out and enjoying the sunshine on the grass ?  ###Speaker: {input_text} ###Negative Response: "   \
      --end_string "#" \
      --use_ram_optimized_load False \
      --arch_type "encoder_decoder" \
      ${key_args}
