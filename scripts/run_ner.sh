cat $0
EXP_NAME=thainer-syllable-3epoch
python3 run_ner.py --tokenizer_type ThaiWordsSyllableTokenizer \
 --tokenizer_name_or_path /ist/ist-share/scads/zo/thai2transformers/exp/syllable/tokenizer_folder \
 --model_name_or_path /ist/ist-share/scads/zo/thai2transformers/exp/syllable/output_folder/model/checkpoint-8000 \
 --dataset_name thainer \
 --label_name ner_tags \
 --per_device_train_batch_size 16 \
 --per_device_eval_batch_size 16 \
 --gradient_accumulation_steps 1 \
 --learning_rate 5e-5 \
 --warmup_steps 1000 \
 --logging_steps 10 \
 --eval_steps 159 \
 --max_steps $((159 * 3)) \
 --evaluation_strategy steps \
 --output_dir /ist/ist-share/scads/zo/thai2transformers/exp_finetune/$EXP_NAME \
 --do_train \
 --do_eval \
 --max_length 512 \
 --fp16
