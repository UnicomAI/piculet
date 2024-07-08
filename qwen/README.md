

## Contents
- [Contents](#contents)
- [Install](#install)
- [QWen Weights](#qwen-weights)
- [Evaluation](#evaluation)
  - [test on POPE, MME and LLaVa-QA90](#test-on-pope-mme-and-llava-qa90)

## Install

We suggest refer to QWen-VL's official repo for details: [https://github.com/QwenLM/Qwen-VL](https://github.com/QwenLM/Qwen-VL).


## QWen Weights
Please check out [huggingface](https://huggingface.co/Qwen/Qwen-VL-Chat) for downloading the specific public QWen checkpoints.


## Evaluation

### test on POPE, MME and LLaVa-QA90
- Firstly, refer to [./server_qwenvl+piculet.py](./server_qwenvl+piculet.py) to run server;

- Then codes to run test on POPE, MME and LLaVa-QA90 can be found in [run_datasets.py](./run_datasets.py)

- With the generated answers, refer to [benchmarks_eval](../benchmarks_eval/) to calculate respective scores.