

## Contents
- [Contents](#contents)
- [Install](#install)
  - [Upgrade to latest code base](#upgrade-to-latest-code-base)
- [LLaVA Weights](#llava-weights)
- [Evaluation](#evaluation)
  - [test on POPE](#test-on-pope)
  - [test on MME](#test-on-mme)
  - [test on LLaVa-QA90](#test-on-llava-qa90)
- [Acknowledgement](#acknowledgement)

## Install

If you are not using Linux, do *NOT* proceed, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Clone this repository and navigate to llava folder
```bash
git clone https://github.com/UnicomAI/piculet.git
cd llava
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade, please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```


## LLaVA Weights
Please check out [huggingface](https://huggingface.co/liuhaotian/llava-v1.5-13b) for downloading the specific public LLaVA checkpoints.


## Evaluation

### test on POPE

```bash
# run to get answers
cd piculet/llava/
# Remember to go to the script and modify the corresponding parameter paths.
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh

```

### test on MME

```bash
# run to get answers
cd piculet/llava/
# Remember to go to the script and modify the corresponding parameter paths.
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh
```


### test on LLaVa-QA90
We sample 10 description-type queries that are paraphrased in various forms to instruct an MLLM to describe an image, image lists are in `llava/playground/data/coco2014_val_qa_eval/image_names.txt`, code to run:

```bash
# run to get answers
cd piculet/llava/
# Remember to go to the script and modify the corresponding parameter paths.
python -m llava.serve.cli-piculet --image-file sampled_coco/COCO_val2014_000000441147.jpg
# This will evoke a user interaction terminal, and the names of the remaining images will need to be manually entered to obtain the results.
```

With these results, you can use the prompt demonstrated in our paper to query gpt4v, thereby obtaining scoring outcomes and ultimately calculating the average score.


## Acknowledgement

- [LLaVa](https://github.com/haotian-liu/LLaVA): the codebase we built upon.
