
# LMStyle

For convenience, all intermediate results are already contained in this repository. Hence, you are able to run any steps without running preceding steps.

## 1.Setup

LMStyle evaluation suite implements partially based on LMFlow framework. Hence, you need to first set up the LMFlow environment. You can refer the Setup section in the [LMFlow](https://github.com/OptimalScale/LMFlow) for further details.


## 2. Run Scripts
### 2.1 Generate Responses

You can run `scripts/run_chatbot_file_task.sh` to generate responses based on different PLMs. For example,
```sh
./run_chatbot_file_task.sh DD Formal
```
`DD` means Daily Dialog dataset. `Formal` means formal response. You can choose `BST` as the dataset and `Informal`, `Positive`, `Negative` as response styles too.

### 2.2 Run Evaluation

You can run `scripts/run_multi_eval.sh` to evaluate model responses through BLEU, SentenceBERT, ChatGPT, and NLL. For example,
```sh
./run_multi_eval.sh ./output_dir/DD_Formal
```
The first argument represents the directory of model responses. In `scripts/run_multi_eval.sh`, you can uncomment line 59 to run ChatGPT evaluation. If you want to do that, you need to first save your ChatGPT tokens in a txt file and pass it into the argument `--api_token_path`.

### 2.3 Run Correlation

To make results more visible, we integrate the code for correlation computing in a Jupyter Notebook file. (`scripts/get_corr.ipynb`) It can calculate the sample level correlation between automatic evaluation and human evaluation. 

