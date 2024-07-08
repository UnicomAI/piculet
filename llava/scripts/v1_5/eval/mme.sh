#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh
#     --auxiliary_model True for with piculet, False for not.
python -m llava.eval.model_vqa_loader-piculet \
    --model-path path/to/models--liuhaotian--llava-v1.5-13b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder path/to/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-13b+Warbler.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --auxiliary_model True

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-13b+Warbler

cd path/to/MME_Benchmark_release_version/

python calculation.py --results_dir path/to/llava/playground/data/eval/MME/eval_tool/answers/llava-v1.5-13b+Warbler
