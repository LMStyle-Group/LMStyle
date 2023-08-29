import os
import openai
import sys
import argparse
import logging
import torch
import random
import time
import json
import re
from transformers import HfArgumentParser


@dataclass
class ChatGPTArguments:
    hyp_dir: Optional[str] = field(
        default="/",
        metadata={
            "help": "directory path of the model responses"
        },
    )
    api_token_path: Optional[str] = field(
        default="/",
        metadata={
            "help": "the file path of api tokens"
        },
    )
    

    

def query_ChatGPT(input, count, code_api_list):
    response = None
    received = False
    while not received:
        try:
            openai.api_key = code_api_list[count % len(code_api_list)]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=input,
                temperature = 0.6,              
            )
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:  # something is wrong: e.g. prompt too long
                print(
                    f"InvalidRequestError\nPrompt passed in:\n\n{input}\n\n")
                assert False
            print("API error:", error)
            print(code_api_list[count % len(code_api_list)])

            count += 1
            time.sleep(2)
    return response, count

def get_keys_from_txt(key_path):
    key_txt = key_path
    pattern = r'sk-\w+'
    key_list = []
    with open(key_txt, 'r') as file:
        for line in file:
            matches = re.findall(pattern, line)
            key_list += matches
    return key_list



if __name__ == "__main__":

    parser = HfArgumentParser((
        ChatGPTArguments
    ))
    chat_args = (
        parser.parse_args_into_dataclasses()
    )
    chat_args = chat_args[0]

    code_api_list = get_keys_from_txt(chat_args.api_token_path)
    hyp_dir = chat_args.hyp_dir

    file_li = os.listdir(hyp_dir)

    print("dataset: ", hyp_dir.split('/')[-1].split("_")[0])
    print("task: ", hyp_dir.split('/')[-1].split("_")[1], '\n')

    for file in file_li:
        print(file.split('_')[-1])
        output_file = hyp_dir+'/'+file+"/test.1.json"
        with open(output_file) as user_file:
            parsed_json = json.load(user_file)

        count = 0
        n=0
        score_l = []
        for pair in parsed_json['instances']:
            input = [
                    {"role": "system", "content": "Given a turn of dialogue, score the appropriateness of the Response with respect to the Speaker on a scale of 0-100 in the following format, with 100 being extremely appropriate and 0 being extremely inaccurate or unreasonable. The evaluation format is as follows: \nSpeaker: [] \nResponse: [] \nAppropriateness Score: []"},
                    {"role": "user", "content": f"\nSpeaker: [{pair['input']}] \nResponse: [{pair['output']}] \nAppropriateness Score: "}
                ]
            response, count = query_ChatGPT(input, count, code_api_list)
            count += 1
            response = response['choices'][0]['message']['content']
            score_l.append(response)
            # n += 1
            # if n % 100 == 0:
            #     print(n)

        for i in range(len(score_l)):
            if len(re.findall(r"\d+",score_l[i])) == 0:
                score_l[i] = 50
                continue
            score_l[i] = int(re.findall(r"\d+",score_l[i])[0])

        with open(output_file, 'r') as f:
            ori_data = json.load(f)
        for i, item in enumerate(ori_data['instances']):
            item['ChatGPT'] = score_l[i]
        print('ChatGPT Score: ', np.array(score_l).mean(),'\n')
        with open(output_file, 'w') as f:
            json.dump(ori_data, f, indent = 6, ensure_ascii=False)