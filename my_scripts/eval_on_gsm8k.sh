#!/bin/bash
MODELS_TO_EVAL=(
    # "third_party/distillm/results/qwen2.5-0.5B-Instruct/gsm8k/train/sft/e20-bs8-lr5e-06-G1-N1-NN1/18040"
    # # # "third_party/distillm/results/qwen2.5-0.5B-Instruct/gsm8k/train/sft/e20-bs8-lr5e-06-G1-N1-NN1/5412"
    # # "third_party/distillm/results/qwen2.5-0.5B-Instruct/gsm8k/train/sft/e20-bs8-lr5e-06-G8-N1-NN1/1344"
    # "third_party/distillm/results/qwen2.5-0.5B-Instruct/gsm8k/train/sft/e20-bs8-lr5e-06-G8-N1-NN1/560"
    # "third_party/distillm/results/qwen2.5-0.5B-Instruct/gsm8k/train/sft/e20-bs8-lr5e-06-G8-N1-NN1/224"
    # "third_party/distillm/results/qwen2.5-0.5B-Instruct/gsm8k/train/sft/e20-bs8-lr5e-06-G1-N1-NN1/1804"
    # "third_party/llama_factory/saves/gsm8k/qwen2.5-0.5B/sft/checkpoint-348"
    # third_party/distillm/results/qwen2.5-0.5B-Instruct/gsm8k/train/sft/e3-bs8-lr5e-06-G8-N1-NN1/336
    saves/gsm8k/qwen2.5-3B/lora_sft/checkpoint-336
)

# Define experiment configurations
declare -A exp1
exp1['custom_name']="qwen0.5B_lora_336"
exp1['base_model']="${BASE_MODELS[0]}"
exp1['peft_model']="saves/gsm8k/qwen2.5-0.5B/lora_sft/checkpoint-336"
exp1['batch_size']="128"
exp1['eval_method']="hash"
exp1['num_examples']="0"
exp1['device']="cuda"

declare -A exp2
exp2['custom_name']="qwen3B_lora_336"
exp2['base_model']="${BASE_MODELS[1]}"
exp2['peft_model']="saves/gsm8k/qwen2.5-3B/lora_sft/checkpoint-336"
exp2['batch_size']="128"
exp2['eval_method']="hash"
exp2['num_examples']="0"
exp2['device']="cuda"

declare -A exp3
exp3['custom_name']="qwen0.5B_full_336"
exp3['base_model']="third_party/distillm/results/qwen2.5-0.5B-Instruct/gsm8k/train/sft/e3-bs8-lr5e-06-G8-N1-NN1/336"
exp3['batch_size']="128"
exp3['eval_method']="hash"
exp3['num_examples']="0"
exp3['device']="cuda"

declare -A exp4
exp4['custom_name']="qwen0.5B_full_336[DM]"
exp4['base_model']="hf_models/Qwen/Qwen2.5-0.5B-Instruct"
exp4['peft_model']="third_party/distillm/results/qwen2.5-3B-Instruct/gsm8k/train/sft/e3-bs4-lr5e-06-G16-N1-NN1-lora-8-32-0.1/336"
exp4['batch_size']="32"
exp4['eval_method']="hash"
exp4['num_examples']="0"
exp4['device']="cuda"
# List of experiments to run
# experiments=("exp1" "exp2" "exp3")
experiments=("exp4")

# Iterate over experiments
for exp in "${experiments[@]}"; do
    eval "custom_name=\${${exp}['custom_name']}"
    eval "base_model=\${${exp}['base_model']}"
    eval "peft_model=\${${exp}['peft_model']}"
    eval "batch_size=\${${exp}['batch_size']}"
    eval "eval_method=\${${exp}['eval_method']}"
    eval "num_examples=\${${exp}['num_examples']}"
    eval "device=\${${exp}['device']}"

    echo "Running experiment: ${custom_name}"
    echo "base_model=${base_model}"
    echo "peft_model=${peft_model}"
    echo "batch_size=${batch_size}"
    echo "eval_method=${eval_method}"
    echo "num_examples=${num_examples}"
    
    python ../../src/run_eval.py \
        --model "$base_model" \
        ${peft_model:+ --peft_model "$peft_model"} \
        --batch_size "$batch_size" \
        --eval_method "$eval_method" \
        --num_examples "$num_examples" \
        --device "$device"

    echo "Completed experiment: ${custom_name}"
    echo "----------------------------------------"
done


