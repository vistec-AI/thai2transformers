export ENCODER_DIR=$1
export ENCODER_NAME=$2
export SPACE_TOKEN=$3
export OPT=$4
export BATCH_SIZE=$5
export LR=$6

# for BATCH_SIZE in 8 16
# do
#     for LR in 1e-5 2e-5 3e-5
#     do
for N_EPOCHS in 1 2 3 4
do
    for WARMUP_RATIO in 0.05 0.1
    do
        export EXP_NAME="exp001.chimera-qa_hp-search.${ENCODER_NAME}.bz-${BATCH_SIZE}.lr-${LR}.n_epochs-${N_EPOCHS}.warmup_ratio-${WARMUP_RATIO}.fp16"
        
        echo "EXP_NAME: ${EXP_NAME}"

        mkdir -p /workspace/thai2transformers_store/logs/chimera-qa/${EXP_NAME}
        mkdir -p /workspace/thai2transformers_store/checkpoints/chimera-qa/${EXP_NAME}
        
        cd /workspace/thai2transformers/scripts/downstream

        echo "Encoder pretrained_model_name_or_path: ${ENCODER_DIR}"
        echo "Encoder name: ${ENCODER_NAME}"
        echo "space token: |${SPACE_TOKEN}|"


        export WANDB_PROJECT=thai2transformers_hp-serach_chimera-qa
        export WANDB_RUN_NAME=$EXP_NAME
        

        if [ -d "/workspace/thai2transformers_store/checkpoints/chimera-qa/${EXP_NAME}/checkpoint-best" ];
        then
            echo "The experiment with this hyperparams set is done."
        else
            python train_question_answering_lm_finetuning.py \
            --model_name ${ENCODER_DIR} \
            --dataset_name chimera_qa \
            --output_dir /workspace/thai2transformers_store/checkpoints/chimera-qa/${EXP_NAME} \
            --log_dir /workspace/thai2transformers_store/logs/chimera-qa/${EXP_NAME} \
            --pad_on_right \
            --run_name ${EXP_NAME} \
            --batch_size ${BATCH_SIZE} \
            --learning_rate ${LR} \
            --warmup_ratio ${WARMUP_RATIO} \
            --num_train_epochs ${N_EPOCHS} \
            --space_token ${SPACE_TOKEN} \
            --fp16 ${OPT}
        fi
        

    done
done
#     done
# done
