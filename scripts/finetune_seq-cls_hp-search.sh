TOKENIZER_TYPE=$1
MODEL_DIR=$2
TOKENIZER_DIR=$3
DATASET=$4
MODEL_NAME=$5
EXP_NAME=$6
MAX_SEQ_LENGTH=$7
SPACE_TOKEN=$8

OPTIONAL_ARGS=$8

for N_EPOCHS in 1 3 5
do
	for BATCH_SIZE in 16 32
	do
		for LR in 1e-5 3e-5 5e-5
		do 
			for WARMUP_RATIO in 0.06 0.1
			do
				

				EXP_ID="${EXP_NAME}_n-epochs=${N_EPOCHS}_bz=${BATCH_SIZE}_lr=${LR}_warmup-ratio=${WARMUP_RATIO}"
				OUTPUT_DIR="../checkpoints/${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}"
				TF_LOG_DIR="../logs/${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}/tf_log"
				TRAINER_OUTPUT_LOG_PATH"../logs/${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}/output_log/trainer.log"
				export WANDB_NAME="${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}"
				export WANDB_DIR="../logs/${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}/wandb_log"

				echo "Pretrained model name: ${MODEL_NAME}" |& tee -a TRAINER_OUTPUT_LOG_PATH
				echo "Type of token: ${TOKENIZER_TYPE}" |& tee -a TRAINER_OUTPUT_LOG_PATH
				echo "Model directory: ${MODEL_DIR}" |& tee -a TRAINER_OUTPUT_LOG_PATH
				echo "Tokenizer directory: ${TOKENIZER_DIR}" |& tee -a TRAINER_OUTPUT_LOG_PATH
				echo "Dataset: ${DATASET}" |& tee -a TRAINER_OUTPUT_LOG_PATH
				echo "" |& tee -a TRAINER_OUTPUT_LOG_PATH

				echo "" |& tee -a TRAINER_OUTPUT_LOG_PATH
				echo "#N epochs: ${N_EPOCHS}" |& tee -a TRAINER_OUTPUT_LOG_PATH
				echo "Batch size: ${BATCH_SIZE}" |& tee -a TRAINER_OUTPUT_LOG_PATH
				echo "Learning rate: ${LR}" |& tee -a TRAINER_OUTPUT_LOG_PATH
				echo "Warmup ratio: ${WARMUP_RATIO}" |& tee -a TRAINER_OUTPUT_LOG_PATH

				python ./seq_cls_finetune.py \
				$TOKENIZER_TYPE \
				$DATASET \
				$OUTPUT_DIR \
				$LOG_DIR \
				--tokenizer_dir $TOKENIZER_DIR \
				--model_dir $MODEL_DIR \
				--num_train_epochs \
				--learning_rate \
				--warmup_ratio ${WARMUP_RATIO} \
				--max_seq_length ${MAX_SEQ_LENGTH} \
				--space_token ${SPACE_TOKEN} \
				${OPTIONAL_ARGS} |& tee -a TRAINER_OUTPUT_LOG_PATH

				echo ""
			done
		done
	done
done