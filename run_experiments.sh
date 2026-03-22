#!/bin/bash
set -e
cd /home/user/fish-speech

run_experiment() {
    local PROB=$1
    local EXP=barbero_$(date +%Y-%m-%dT%H%M%S)
    echo "[$(date)] Starting experiment ${EXP} with interactive_prob=${PROB}"
    mkdir -p results/${EXP}
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python fish_speech/train.py \
        --config-name text2semantic_finetune \
        project=${EXP} \
        +lora@model.model.lora_config=r_32_alpha_16_fast \
        train_dataset.interactive_prob=${PROB} \
        val_dataset.interactive_prob=${PROB} \
        trainer.max_steps=3200 \
        model.lr_scheduler.T_max=3200 \
        >> results/${EXP}/train.log 2>&1
    cp fish_speech/configs/text2semantic_finetune.yaml results/${EXP}/
    cp fish_speech/configs/lora/r_32_alpha_16_fast.yaml results/${EXP}/
    echo "[$(date)] Finished experiment ${EXP}"
}

run_experiment 1.0
run_experiment 0.3

echo "[$(date)] All experiments complete."
