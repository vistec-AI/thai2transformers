#!/usr/bin/env bash
declare -A tokenizer_name_or_path
declare -A tokenizer_type
declare -A model_name_or_path
declare -A dataset_size

tokenizer_type["spm_camembert"]="CamembertTokenizer"

tokenizer_name_or_path["spm_camembert"]="/workspace/checkpoints/roberta_th/95gb-text-corpora/spm_camembert/tokenizer_folder"
model_name_or_path["som_camembert"]="/workspace/checkpoints/roberta_th/95gb-text-corpora/spm_camembert/model/checkpoint-320000"


dataset_size["thainer"]="5078"
dataset_size["lst20"]="67104"
dataset_size["dummytest"]="50"

output_base_dir=/workspace/checkpoints/roberta_th/95gb-text-corpora/spm_camembert/finetuning/
log_folder=/workspace/logs/roberta_th/95gb-text-corpora/spm_camembert/finetuning/

declare -a base_models=("spm_camembert")
declare -a dataset_names=("thainer" "lst20")
declare -a label_names=("ner_tags" "pos_tags")

effective_batch_size=32
device_batch_size=16

for dataset_name in "${dataset_names[@]}"
do
    for label_name in "${label_names[@]}"
    do
        for model in "${base_models[@]}"
        do
            EXP_NAME="${dataset_name}-${label_name}"
            log_path="${log_folder}/${EXP_NAME}.log"
            if [ -e "$log_path" ]; then
                echo "$log_path exist."
                continue
            fi
            touch "$log_path"
            echo "==================== Running Exp ===================="
            echo "${EXP_NAME}"
            echo "Tokenizer Type: ${tokenizer_type["${model}"]}"
            echo "Tokenizer Name or path: ${tokenizer_name_or_path["${model}"]}"
            echo "Model name or path: ${model_name_or_path["${model}"]}"
            echo "Training size: ${dataset_size["${dataset_name}"]}"
            echo "Warmup: $((("${dataset_size["${dataset_name}"]}" / "$effective_batch_size") * 3 / 10))"
            echo "Maxstep: $((("${dataset_size["${dataset_name}"]}" / "$effective_batch_size") * 3))"
            echo "Evalstep: $((("${dataset_size["${dataset_name}"]}" / "$effective_batch_size")))"
            echo "Output Dir: ${output_base_dir}/${EXP_NAME}"
            command="
                set -x
                python3 run_ner.py --tokenizer_type ${tokenizer_type["${model}"]} \
                --tokenizer_name_or_path ${tokenizer_name_or_path["${model}"]} \
                --model_name_or_path ${model_name_or_path["${model}"]} \
                --dataset_name ${dataset_name} \
                --lst20_data_dir ../data/input/datasets/LST20_Corpus \
                --label_name ${label_name} \
                --per_device_train_batch_size ${device_batch_size} \
                --per_device_eval_batch_size ${device_batch_size} \
                --gradient_accumulation_steps 1 \
                --learning_rate 5e-5 \
                --warmup_steps $((("${dataset_size["${dataset_name}"]}" / "$effective_batch_size") * 3 / 10))  \
                --logging_steps 10 \
                --eval_steps $((("${dataset_size["${dataset_name}"]}" / "$effective_batch_size"))) \
                --max_steps $((("${dataset_size["${dataset_name}"]}" / "$effective_batch_size") * 3)) \
                --evaluation_strategy steps \
                --output_dir /workspace/checkpoints/roberta_th/95gb-text-corpora/spm_camembert/finetuning/$EXP_NAME \
                --do_train \
                --do_eval \
                --max_length 512 \
                --fp16 \
                --load_best_model_at_end
                "
                script -q -c "bash -c ${command@Q}" /dev/null | tee "$log_path"
            sleep 0.5
        done
    done
done
