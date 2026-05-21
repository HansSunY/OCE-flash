#!/bin/bash
CONCEPT="100"
MODEL_ID="model/stable-diffusion-v1-4/"
OCE_MODEL_PATH="celeb/celeb_$CONCEPT.safetensors"
SAVE_PATH="celeb_100_$CONCEPT"
#EXP_NAME="P"
EXP_NAME="100"
NUM_IMAGES_PER_PROMPT=1
NUM_INFERENCE_STEPS=50
DEVICE="cuda:0"

bash celeb.sh

python generate_celeb.py \
    --model_id "$MODEL_ID" \
    --oce_model_path "$OCE_MODEL_PATH" \
    --save_path "$SAVE_PATH" \
    --exp_name "$EXP_NAME" \
    --num_images_per_prompt $NUM_IMAGES_PER_PROMPT \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --device "$DEVICE"
