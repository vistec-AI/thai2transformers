TOKENIZER_TYPE=$1
MODEL_NAME=$2
EXP_NAME=$3
MAX_SEQ_LENGTH=$4
OPTIONAL_ARGS=$5

for DATASET in wisesight_sentiment wongnai_reviews generated_reviews_enth-review_star prachathai67k
do
	for N_EPOCHS in 3
	do
		for BATCH_SIZE in 16
		do
			for LR in 2e-5
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
					echo "Dataset: ${DATASET}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "WANDB_NAME: ${WANDB_NAME}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "WANDB_DIR: ${WANDB_DIR}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "#N epochs: ${N_EPOCHS}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Batch size: ${BATCH_SIZE}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Learning rate: ${LR}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Warmup ratio: ${WARMUP_RATIO}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}
					echo "Max sequence length: ${MAX_SEQ_LENGTH}" |& tee -a ${TRAINER_OUTPUT_LOG_PATH}

					python ./seq_cls_finetune.py \
					${TOKENIZER_TYPE} \
					${DATASET} \
					${OUTPUT_DIR} \
					${TF_LOG_DIR} \
					--batch_size ${BATCH_SIZE} \
					--num_train_epochs ${N_EPOCHS} \
					--learning_rate ${LR} \
					--warmup_ratio ${WARMUP_RATIO} \
					--max_seq_length ${MAX_SEQ_LENGTH} \
					--wandb_run_name $WANDB_NAME \
					--logging_steps 50 \
					${OPTIONAL_ARGS} |& tee -a ${TRAINER_OUTPUT_LOG_PATH}

					echo ""
				done
			done
		done
	done
done