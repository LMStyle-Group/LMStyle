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

import os
import numpy as np

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")

@dataclass
class DataPairArguments:
    query_file: Optional[str] = field(
        default="/",
        metadata={
            "help": "file path of the query"
        },
    )
    response_file: Optional[str] = field(
        default="/",
        metadata={
            "help": "file path of the response"
        },
    )
    output_file: Optional[str] = field(
        default="/",
        metadata={
            "help": "file path of the response"
        },
    )
    


def main():
    parser = HfArgumentParser((
        DataPairArguments
    ))
    tst_args = (
        parser.parse_args_into_dataclasses()
    )
    tst_args = tst_args[0]

    with open(tst_args.query_file, "r") as f:
        input = f.readlines()

    with open(tst_args.response_file, "r") as f:
        output = f.readlines()

    input = [text.replace('\n','') for text in input]
    output = [text.replace('\n','') for text in output]
    data_dict = {
        "type": "text2text",
        "instances": [{'input': input[i], 'output': output[i]} for i in range(len(input))]
    }

    with open(tst_args.output_file, 'w') as f:
        json.dump(data_dict, f, indent = 6, ensure_ascii=False)


if __name__ == "__main__":
    main()
