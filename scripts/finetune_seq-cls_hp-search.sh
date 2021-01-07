TOKENIZER_TYPE=$1
MODEL_DIR=$2
TOKENIZER_DIR=$3
DATASET=$4
MODEL_NAME=$5
EXP_NAME=$6
MAX_SEQ_LENGTH=$7
SPACE_TOKEN=$8

OPTIONAL_ARGS=$8

echo "Pretrained model name: ${MODEL_NAME}"
echo "Type of token: ${TOKENIZER_TYPE}"
echo "Model directory: ${MODEL_DIR}"
echo "Tokenizer directory: ${TOKENIZER_DIR}"
echo "Dataset: ${DATASET}" 
echo ""
for N_EPOCHS in 1 3 5
do
	for BATCH_SIZE in 16 32
	do
		for LR in 1e-5 3e-5 5e-5
		do 
			for WARMUP_RATIO in 0.06 0.1
			do
				echo ""
				echo "#N epochs: ${N_EPOCHS}"
				echo "Batch size: ${BATCH_SIZE}"
				echo "Learning rate: ${LR}"
				echo "Warmup ratio: ${WARMUP_RATIO}"

				EXP_ID="${EXP_NAME}_n-epochs=${N_EPOCHS}_bz=${BATCH_SIZE}_lr=${LR}_warmup-ratio=${WARMUP_RATIO}"
				OUTPUT_DIR="../checkpoints/${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}"
				TF_LOG_DIR="../checkpoints/${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}/tf_log"
				TRAINER_OUTPUT_LOG_PATH"../checkpoints/${MODEL_NAME}/${TOKENIZER_TYPE}/finetuned/${DATASET}/${EXP_ID}/output_log/trainer.log"

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