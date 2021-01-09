TOKENIZER_TYPE=$1
MODEL_DIR=$2
TOKENIZER_DIR=$3
MODEL_NAME=$4
EXP_NAME=$5
MAX_SEQ_LENGTH=$6
SPACE_TOKEN=$7
OPTIONAL_ARGS=$8

for DATASET in wisesight_sentiment wongnai_reviews generated_reviews_enth-correct_translation generated_reviews_enth-review_star
do
	for N_EPOCHS in 3
	do
		for BATCH_SIZE in 32
		do
			for LR in 3e-5
			do 
				for WARMUP_RATIO in 0.1
				do
					

					EXP_ID="${EXP_NAME}_n-epochs=${N_EPOCHS}_bz=${BATCH_SIZE}_lr=${LR}_warmup-ratio=${WARMUP_RATIO}"
					OUTPUT_DIR="../checkpoints/${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}"
					TF_LOG_DIR="../logs/${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}/tf_log"
					TRAINER_OUTPUT_LOG_DIR="../logs/${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}/output_log"
					TRAINER_OUTPUT_LOG_PATH="${TRAINER_OUTPUT_LOG_DIR}/trainer.log"
					mkdir -p ${TRAINER_OUTPUT_LOG_DIR}
					mkdir -p ${TF_LOG_DIR}
					export WANDB_NAME="${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}"
					export WANDB_DIR="../logs/${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}"
					mkdir -p ${WANDB_DIR}
					chmod 777 ${WANDB_DIR}

					echo "Pretrained model name: ${MODEL_NAME}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Type of token: ${TOKENIZER_TYPE}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Model directory: ${MODEL_DIR}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Tokenizer directory: ${TOKENIZER_DIR}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Dataset: ${DATASET}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "WANDB_NAME: ${WANDB_NAME}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "WANDB_DIR: ${WANDB_DIR}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "#N epochs: ${N_EPOCHS}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Batch size: ${BATCH_SIZE}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Learning rate: ${LR}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Warmup ratio: ${WARMUP_RATIO}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Space token: ${SPACE_TOKEN}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Max sequence length: ${MAX_SEQ_LENGTH}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}

					python ./seq_cls_finetune.py \
					${TOKENIZER_TYPE} \
					${DATASET} \
					${OUTPUT_DIR} \
					${TF_LOG_DIR} \
					--tokenizer_dir ${TOKENIZER_DIR} \
					--model_dir ${MODEL_DIR} \
					--num_train_epochs ${N_EPOCHS} \
					--learning_rate ${LR} \
					--warmup_ratio ${WARMUP_RATIO} \
					--max_seq_length ${MAX_SEQ_LENGTH} \
					--space_token ${SPACE_TOKEN} \
					--wandb_run_name $WANDB_NAME \
					--fp16 \
					${OPTIONAL_ARGS} |& tee -a ${TRAINER_OUTPUT_LOG_PATH}

					echo ""
				done
			done
		done
	done
done