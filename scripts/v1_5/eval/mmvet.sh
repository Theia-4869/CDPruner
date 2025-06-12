#!/bin/bash

CKPT_DIR="/path/to/checkpoint"
DATA_DIR="/path/to/dataset"

CKPT="llava-v1.5-7b"
SPLIT="llava-mm-vet"

TOKEN=${1}
PARAM="vtn_${TOKEN}"

python -m llava.eval.model_vqa \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/mm-vet/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/answers_upload/${SPLIT}/${CKPT}

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl \
    --dst ./playground/data/eval/mm-vet/answers_upload/${SPLIT}/${CKPT}/${PARAM}.json
