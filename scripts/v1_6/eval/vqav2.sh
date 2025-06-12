#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="/path/to/checkpoint"
DATA_DIR="/path/to/dataset"

CKPT="llava-v1.6-vicuna-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

TOKEN=${1}
PARAM="vtn_$((TOKEN * 5))"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/vqav2/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --visual_token_num ${TOKEN} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

VQAV2_DIR="./playground/data/eval/vqav2"
output_file=./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py \
    --dir ${VQAV2_DIR} \
    --src answers/${SPLIT}/${CKPT}/${PARAM}/merge.jsonl \
    --dst answers_upload/${SPLIT}/${CKPT}/${PARAM}.json
