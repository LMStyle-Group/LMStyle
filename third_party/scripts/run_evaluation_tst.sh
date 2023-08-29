#!/bin/bash

key_args="--hyp_file output_dir/response_result_GYAFC/response.txt --ref_dir data/GYAFC/reference1 --metric BLEU"
if [ $# -ge 1 ]; then
  key_args="$@"
fi

examples/tst_eval.py ${key_args}
      
