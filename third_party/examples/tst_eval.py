#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple TST evaluation implementation.
"""
import logging
import json
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import warnings

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

import sacrebleu
import numpy as np


logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")

LOG_PATH = "./output_dir/result_summary.txt"

@dataclass
class TSTEvalArguments:
    hyp_file: Optional[str] = field(
        default="/",
        metadata={
            "help": "file path of the hypothesis"
        },
    )
    ref_dir: Optional[str] = field(
        default="/",
        metadata={
            "help": "directory path of the references (sometimes multiple references)"
        },
    )
    metric: Optional[str] = field(
        default="BLEU",
        metadata={
            "help": "name of metric"
        },
    )
    output_file: Optional[str] = field(
        default="/",
        metadata={
            "help": "the output json file path"
        },
    )
    api_token_path: Optional[str] = field(
        default="/",
        metadata={
            "help": "the file path of api tokens"
        },
    )
    


def main():
    parser = HfArgumentParser((
        TSTEvalArguments
    ))
    tst_args = (
        parser.parse_args_into_dataclasses()
    )
    tst_args = tst_args[0]

    if tst_args.metric == "BLEU":
        import os
        ref_dir = tst_args.ref_dir
        hyp_file = tst_args.hyp_file
        list_dir = os.listdir(ref_dir)
        ref_list = []
        for file in list_dir:
            file_path = ref_dir + '/' + file
            with open(file_path, 'r') as fin:
                ref = fin.readlines()

            ref_list.append(ref)

        ref_list_arrange = []
        for i in range(len(ref_list[0])):
            ref_list_tmp = []
            for ref in ref_list:
                ref_list_tmp.append(ref[i].replace('\n','').strip())
            ref_list_arrange.append(ref_list_tmp)

        with open(hyp_file, 'r') as fin:
            hyp_list = fin.readlines()

        hyp_list = [hyp.replace('\n','').strip() for hyp in hyp_list]
        bleu_list = []
        for i, hyp in enumerate(hyp_list):
            bleu = sacrebleu.sentence_bleu(hyp,ref_list_arrange[i])
            bleu_list.append(bleu.score)
        with open(tst_args.output_file, 'r') as f:
            ori_data = json.load(f)
        for i, item in enumerate(ori_data['instances']):
            item['BLEU'] = bleu_list[i]
        print('BLEU Score: ', np.array(bleu_list).mean())
        with open(LOG_PATH, "a") as f:
            f.write('BLEU Score: '+str(np.array(bleu_list).mean())+'\n')

        with open(tst_args.output_file, 'w') as f:
            json.dump(ori_data, f, indent = 6, ensure_ascii=False)

    if tst_args.metric == "SBERT":
        import sentence_transformers
        import os
        ref_dir = tst_args.ref_dir
        hyp_file = tst_args.hyp_file
        list_dir = os.listdir(ref_dir)
        ref_list = []
        for file in list_dir:
            file_path = ref_dir + '/' + file
            with open(file_path, 'r') as fin:
                ref = fin.readlines()
            ref_list.append(ref)

        ref_list = ref_list[0] # single reference by default
        ref_list = [ref.replace('\n','').strip() for ref in ref_list]

        with open(hyp_file, 'r') as fin:
            hyp_list = fin.readlines()

        hyp_list = [hyp.replace('\n','').strip() for hyp in hyp_list]

        SBERT_score_list = []
        model = sentence_transformers.SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        for i, hyp in enumerate(hyp_list):
            sentences = [ref_list[i], hyp]
            embeddings = model.encode(sentences)
            SBERT_score_list.append(sentence_transformers.util.cos_sim(embeddings[0],embeddings[1]).item())
        with open(tst_args.output_file, 'r') as f:
            ori_data = json.load(f)
        for i, item in enumerate(ori_data['instances']):
            item['SBERT'] = SBERT_score_list[i]
        print('SBERT Score: ', np.array(SBERT_score_list).mean())
        with open(LOG_PATH, "a") as f:
            f.write('SBERT Score: '+str(np.array(SBERT_score_list).mean())+'\n')

        with open(tst_args.output_file, 'w') as f:
            json.dump(ori_data, f, indent = 6, ensure_ascii=False)

    if (tst_args.metric == "Formal") or (tst_args.metric == "Informal"):
        from transformers import pipeline

        hyp_file = tst_args.hyp_file
        ref_dir = tst_args.ref_dir
        with open(hyp_file, 'r') as fin:
            hyp_list = fin.readlines()
        hyp_list = [hyp.replace('\n','').strip() for hyp in hyp_list]
        formality_analysis = pipeline("sentiment-analysis", model="s-nlp/roberta-base-formality-ranker")
        formal_score = [formality_analysis(sent)[0]['score'] if formality_analysis(sent)[0]['label'] == 'formal' else 1-formality_analysis(sent)[0]['score'] for sent in hyp_list]
        informal_score = [1-score for score in formal_score]
        formal_bi = [1 if score>0.5 else 0 for score in formal_score]
        informal_bi = [1 if score>0.5 else 0 for score in informal_score]

        with open(tst_args.output_file, 'r') as f:
            ori_data = json.load(f)

        if tst_args.metric == "Formal":
            for i, item in enumerate(ori_data['instances']):
                item['Style Strength'] = formal_score[i]
            print('Style Score: ', np.array(formal_score).mean())
            print('Style Acc: ', np.array(formal_bi).sum()/len(formal_bi))
            with open(LOG_PATH, "a") as f:
                f.write('Style Score: '+str(np.array(formal_score).mean())+'\n')
                f.write('Style Acc: '+str(np.array(formal_bi).sum()/len(formal_bi))+'\n')
            with open(tst_args.output_file, 'w') as f:
                json.dump(ori_data, f, indent = 6, ensure_ascii=False)
        if tst_args.metric == "Informal":
            for i, item in enumerate(ori_data['instances']):
                item['Style Strength'] = informal_score[i]
            print('Style Score: ', np.array(informal_score).mean())
            print('Style Acc: ', np.array(informal_bi).sum()/len(informal_bi))
            with open(LOG_PATH, "a") as f:
                f.write('Style Score: '+str(np.array(informal_score).mean())+'\n')
                f.write('Style Acc: '+str(np.array(informal_bi).sum()/len(informal_bi))+'\n')
            with open(tst_args.output_file, 'w') as f:
                json.dump(ori_data, f, indent = 6, ensure_ascii=False)

    if (tst_args.metric == "Positive") or (tst_args.metric == "Negative"):
        from transformers import pipeline

        hyp_file = tst_args.hyp_file
        ref_dir = tst_args.ref_dir
        with open(hyp_file, 'r') as fin:
            hyp_list = fin.readlines()
        hyp_list = [hyp.replace('\n','').strip() for hyp in hyp_list]
        sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
        positive_score = [sentiment_analysis(sent)[0]['score'] if sentiment_analysis(sent)[0]['label'] == 'POSITIVE' else 1-sentiment_analysis(sent)[0]['score'] for sent in hyp_list]
        positive_bi = [1 if score>0.5 else 0 for score in positive_score]
        negative_score = [1-score for score in positive_score]
        negative_bi = [1 if score>0.5 else 0 for score in negative_score]
        
        with open(tst_args.output_file, 'r') as f:
            ori_data = json.load(f)

        if tst_args.metric == "Positive":
            for i, item in enumerate(ori_data['instances']):
                item['Style Strength'] = positive_score[i]
            print('Style Score: ', np.array(positive_score).mean())
            print('Style Acc: ', np.array(positive_bi).sum()/len(positive_bi))
            with open(LOG_PATH, "a") as f:
                f.write('Style Score: '+str(np.array(positive_score).mean())+'\n')
                f.write('Style Acc: '+str(np.array(positive_bi).sum()/len(positive_bi))+'\n')
            with open(tst_args.output_file, 'w') as f:
                json.dump(ori_data, f, indent = 6, ensure_ascii=False)

        if tst_args.metric == "Negative":
            for i, item in enumerate(ori_data['instances']):
                item['Style Strength'] = negative_score[i]
            print('Style Score: ', np.array(negative_score).mean())
            print('Style Acc: ', np.array(negative_bi).sum()/len(negative_bi))
            with open(LOG_PATH, "a") as f:
                f.write('Style Score: '+str(np.array(negative_score).mean())+'\n')
                f.write('Style Acc: '+str(np.array(negative_bi).sum()/len(negative_bi))+'\n')
            with open(tst_args.output_file, 'w') as f:
                json.dump(ori_data, f, indent = 6, ensure_ascii=False)
    
    if tst_args.metric == "ChatGPT":
        import os
        import openai
        import sys
        import time
        import re

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

        code_api_list = get_keys_from_txt(tst_args.api_token_path)
        with open(tst_args.output_file) as user_file:
            parsed_json = json.load(user_file)
        
        count = 0
        score_l = []
        for pair in parsed_json['instances']:
            print(pair)
            input = [
                    {"role": "system", "content": "Given a turn of dialogue, score the appropriateness of the Response with respect to the Speaker on a scale of 0-100 in the following format, with 100 being extremely appropriate and 0 being extremely inaccurate or unreasonable. The evaluation format is as follows: \nSpeaker: [] \nResponse: [] \nAppropriateness Score: []"},
                    {"role": "user", "content": f"\nSpeaker: [{pair['input']}] \nResponse: [{pair['output']}] \nAppropriateness Score: "}
                ]
            response, count = query_ChatGPT(input, count, code_api_list)
            count += 1
            response = response['choices'][0]['message']['content']
            score_l.append(response)

        for i in range(len(score_l)):
            if len(re.findall(r"\d+",score_l[i])) == 0:
                score_l[i] = 50
                continue
            score_l[i] = int(re.findall(r"\d+",score_l[i])[0])

        with open(tst_args.output_file, 'r') as f:
            ori_data = json.load(f)
        for i, item in enumerate(ori_data['instances']):
            item['ChatGPT'] = score_l[i]
        print('ChatGPT Score: ', np.array(score_l).mean())
        with open(tst_args.output_file, 'w') as f:
            json.dump(ori_data, f, indent = 6, ensure_ascii=False)
        


if __name__ == "__main__":
    main()
