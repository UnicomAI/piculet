#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh
#     --auxiliary_model True for with piculet, False for not.
python -m llava.eval.model_vqa_loader-piculet --model-path path/to/models--liuhaotian--llava-v1.5-13b --question-file ./playground/data/eval/pope/llava_pope_test.jsonl --image-folder path/to/coco/val2014 --answers-file ./playground/data/eval/pope/answers/llava-v1.5-13b_pope.jsonl --temperature 0 --conv-mode vicuna_v1 --auxiliary_model True

## llava测试的pope版本也经历了变换，此处换成7月前的版本，否则random项结果会很离谱
python llava/eval/eval_pope.py \
    --annotation-dir piculet/benchmarks_eval/POPE-e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco \
    --question-file path/to/llava-v1.5-13b_pope.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b_pope_total.jsonl