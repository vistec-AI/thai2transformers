# train_mlm_camembert_thai.py

```bash
python3 train_mlm_camembert_thai.py \
 --tokenizer_name_or_path ../data/input/zo_test/sentencepiece.bpe.model \
 --ext .txt \
 --train_dir ../data/input/zo_test/text/train \
 --eval_dir ../data/input/zo_test/text/train \
 --binarized_path_train ../data/output/zo_test/train/binarized.pt \
 --binarized_path_val ../data/output/zo_test/train/binarized.pt \
 --train_max_length 416 --eval_max_length 416 \
 --learning_rate 3e-4 --weight_decay 0.01 \
 --adam_epsilon 1e-6 \
 --max_steps 100 \
 --per_device_train_batch_size 2 \
 --per_device_eval_batch_size 2 \
 --gradient_accumulation_steps 1 \
 --warmup_steps 24000 \
 --seed 2020 \
 --save_steps 10000 \
 --logging_steps 5 \
 --save_total_limit 50 \
 --evaluate_during_training True \
 --eval_steps 2500 \
 --logging_dir ../data/output/zo_test/logs \
 --output_dir ../data/output/zo_test/model \
 --overwrite_output_dir False \
 --add_space_token \
 --model_dir ../data/input/zo_test/model/
```

```bash
$ cat ../data/input/zo_test/text/train
xaa.txt
...
```

```bash
$ cat ../data/input/zo_test/text/train/xaa.txt
กดาสหกาดสาหกด<th_roberta_space_token>หกดง<th_roberta_space_token>?
หกดกหดเ้้<th_roberta_space_token>หกดหกดกหด<th_roberta_space_token>หกด
ดกเกดิปหหห
ฟหดเสาฟพนยายยฟยเดกเด
กดอไยสยดสยหก<th_roberta_space_token>หยกดยยนไยำไ<th_roberta_space_token>หกดไำนรพนยรนหกดดด<th_roberta_space_token>หกดนานยำาไานดหก<th_roberta_space_token>หกวดสงวสกวดสวสหกด<th_roberta_space_token>ดเกสดสเหกห<th_roberta_space_token>กหด<th_roberta_space_token>ฟฟฟ
```
