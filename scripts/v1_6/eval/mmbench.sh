#!/bin/bash

CKPT_DIR="/path/to/checkpoint"
DATA_DIR="/path/to/dataset"

CKPT="llava-v1.6-vicuna-7b"
SPLIT="mmbench_dev_20230712"

TOKEN=${1}
PARAM="vtn_$((TOKEN * 5))"

python -m llava.eval.model_vqa_mmbench \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ${DATA_DIR}/mmbench/${SPLIT}.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/${SPLIT}/${CKPT}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_DIR}/mmbench/${SPLIT}.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/${SPLIT}/${CKPT} \
    --experiment ${PARAM}
