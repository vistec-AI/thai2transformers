
## Language Model Finetuning on Sequence Classification Task

<br>

--------

<br>
<!-- Currently, the sequence classification finetuning script supports 4 Thai datasets published in Huggingface's datasets including -->
<!-- `wisesight_sentiment`, `wongnai_reviews`, `generated_reviews_enth` and `prachathai67k`. -->

We provide a finetuning script (`./scripts/downstream/train_sequence_classification_lm_finetuning.py`) to finetune our pretrained language model on 3 multiclass classification tasks ( `wisesight_sentiment`, `wongnai_reviews`, `generated_reviews_enth` : review_star ) and 1 multilabel classification task (`prachathai67k`).


The arguements for the `train_sequence_classification_lm_finetuning.py` are as follows:

<br>

Required arguments:

- **tokenizer_type_or_public_model_name** : 

    The token type that RoBERThai used (`spm`, `spm_camembert` (for roberthai-95g-spm), `newmm`, `syllable`, `sefr_cut`). 
    
    If the token type is specified, it is required to specify the directory to model checkpoint and tokenizer via `--model_dir` and `--tokenizer_dir`.
    
    Otherwise, specify other public language model (Currently, we support `mbert` and `xlmr` )

- **dataset_name** : 

    Specify the dataset name to finetune. Currently, sequence classification datasets including `wisesight_sentiment`, `generated_reviews_enth-review_star`, and`wongnai_reviews`.

- **output_dir** : 

    The directory to store finetuned model

- **logging_dir** : 

    The directory to logging output including Tensorboard log, and `wandb` log (optional)

<br>

Optional arguments:



- `--model_dir`     :  The directory of pretrained model checkpoint

- `--tokenizer_dir` :  The directory of tokenizer's vocab

- `--space_token`   :  The custom token that will replace a space token in the texts. As some models use custom space token (default: `"<_>"`). For `mbert` and `xlmr` specify the space token as `" "`.

- `--max_length`: Specify the max length of text inputs to be passed to the model, The max length should be less than the **max positional embedding** or the max sequence length that langauge model was pretrained on.

- `--num_train_epochs`: Number of epochs to finetune model (default: `5`)
- `--learning_rate`: The value of peak learning rate (default: `1e-05`)
- `--weight_decay` : The value of weight decay (default: `0.01`)
- `--warmup_ratio`: The ratio of steps / max_steps to warmup learning rate (default: `0.1`; in other word, warm up the learning until the peak valye for the first 10% of the total steps)
- `--batch_size`: The batch size (default: `16`)
- `--no_cuda`: Append "--no_cuda" to use only CPUs during finetuning (default: `False`)
- `--fp16`: Append "--fp16" to use FP16 mixed-precision trianing (default: `False`)
- `--metric_for_best_model`: The metric to select the best model based on validation set (default: `f1_micro`)
- `--greater_is_better`: The criteria to select the best model according to the specified metric either by expecting the greater value or lower value (default: `True`)
- `--logging_steps` : In interval of training steps to perform logging  (default: `10`)
- `--seed` : The seed value (default: `2020`)
- `--fp16_opt_level` : The OPT level for FP16 mixed-precision training (default: `O1`)
- `--gradient_accumulation_steps` : The number of steps to accumulate gradients (default: `1`, no gradient accumulation)
- `--adam_epsilon` : Value of Adam epsilon (default: `1e-05`)
- `--max_grad_norm` : Value of gradient norm (default: `1.0`)
- `--lowercase`     :  Append "--lowercase" to convert all input texts to lowercase as some model may 
support only uncased texts (default: `False`)
- `--run_name`     :  Specify the **run_name** for logging experiment to wandb.com (default: `None`)


### Example 

<br>

1. Finetuning `roberthai-thwiki-spm` on multiclass classification task of `wisesight_sentiment` dataset.

    The following script will finetune the `roberthai-thwiki-spm` pretrained model from checkpoint:7000. 
     
    The script will finetune model with FP16 mixed-precision training on GPU (ID: 3). The train and validation batch size is 16 with no gradient accumulation. The model checkpoint will be save every epoch and select the best model by validation f1_micro. During finetuning, the learning rate will be warmed up linearly until `3e-05` for 100 steps, then linearly decay to zero. The maximum sequence length that the model will be passed (from the resuling number of tokens according to the tokenizer specified). Otherwise, it will truncate the sequence to `max_length`. 

    ```
    cd ./scripts/downstream
    CUDA_VISIBLE_DEVICES=3 python ./train_sequence_classification_lm_finetuning.py \
    spm \
    wisesight_sentiment \
    /workspace/checkpoints/roberthai-thwiki-spm/finetuned/wisesight_sentiment/ \
    /workspace/logs/roberthai-thwiki-spm/finetuned/wisesight_sentiment/ \
    --tokenizer_dir /workspace/checkpoints/roberthai-thwiki-spm/tokenizer_folder \
    --model_dir /workspace/checkpoints/roberthai-thwiki-spm/model/checkpoint-7000 \
    --num_train_epochs 1 \
    --metric_for_best_model f1_micro \
    --learning_rate 3e-05 \
    --warmup_ratio 0.1 \
    --max_length 512 \
    --space_token "<_>" \
    --fp16
    ```

    <details>
    <summary>
    Log output:
    </summary>
    
    ```
    [INFO] Dataset: wisesight_sentiment


    [INFO] Huggingface's dataset name: wisesight_sentiment 
    [INFO] Task: multiclass_classification

    [INFO] space_token: <_>
    [INFO] prepare_for_tokenization: False

    Reusing dataset wisesight_sentiment (/root/.cache/huggingface/datasets/wisesight_sentiment/wisesight_sentiment/1.0.0/4bb1772cff1a0703d72fb9e84dff9348e80f6cdf80b0f6c0f59bcd85fc5a3537)
    Some weights of the model checkpoint at /workspace/checkpoints/roberthai-thwiki-spm/model/checkpoint-7000 were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.decoder.bias']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at /workspace/checkpoints/roberthai-thwiki-spm/model/checkpoint-7000 and are newly initialized: ['classifier.dense.weight', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.out_proj.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

    [INFO] Model architecture: RobertaForSequenceClassification(
    (roberta): RobertaModel(
        (embeddings): RobertaEmbeddings(
        (word_embeddings): Embedding(24000, 768, padding_idx=1)
        (position_embeddings): Embedding(514, 768, padding_idx=1)
        (token_type_embeddings): Embedding(1, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): RobertaEncoder(
        (layer): ModuleList(
            (0): RobertaLayer(
            (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
            (1): RobertaLayer(
            (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
            (2): RobertaLayer(
            (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
            (3): RobertaLayer(
            (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
            (4): RobertaLayer(
            (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
            (5): RobertaLayer(
            (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
            (6): RobertaLayer(
            (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
            (7): RobertaLayer(
            (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
            (8): RobertaLayer(
            (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
            (9): RobertaLayer(
            (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
            (10): RobertaLayer(
            (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
            (11): RobertaLayer(
            (attention): RobertaAttention(
                (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            )
        )
        )
    )
    (classifier): RobertaClassificationHead(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (out_proj): Linear(in_features=768, out_features=4, bias=True)
    )
    ) 



    [INFO] tokenizer: PreTrainedTokenizer(name_or_path='/workspace/checkpoints/roberthai-thwiki-spm/tokenizer_folder', vocab_size=24000, model_max_len=1000000000000000019884624838656, is_fast=False, padding_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': '<mask>', 'additional_special_tokens': ['<_>']}) 



    [INFO] Preprocess and tokenizing texts in datasets
    [INFO] max_length = 512 

    [DEBUG] labels [1 1 1 1]

    [DEBUG] label_encoder.classes_ [0 1 2 3]


    [DEBUG] (before: preprocessor) input_text ['ไปจองมาแล้วนาจา Mitsubishi Attrage ได้หลังสงกรานต์เลย รอขับอยู่นาจา กระทัดรัด เหมาะกับสาวๆขับรถคนเดียวแบบเรา ราคาสบายกระเป๋า ประหยัดน้ำมัน วิ่งไกลแค่ไหนหายห่วงค่ะ', 'เปิดศักราชใหม่! นายกฯ แถลงข่าวก่อนการแข่งขันศึก #ช้างเอฟเอคัพ นัดชิงชนะเลิศ', 'บัตรสมาชิกลดได้อีกไหมคับ', 'สนใจ new mazda2ครับ']

    [DEBUG] Apply preprocessor to texts.


    [DEBUG] (after: preprocessor) input_text ['ไปจองมาแล้วนาจา<_>mitsubishi<_>attrage<_>ได้หลังสงกรานต์เลย<_>รอขับอยู่นาจา<_>กระทัดรัด<_>เหมาะกับสาวๆขับรถคนเดียวแบบเรา<_>ราคาสบายกระเป๋า<_>ประหยัดน้ำมัน<_>วิ่งไกลแค่ไหนหายห่วงค่ะ', 'เปิดศักราชใหม่!<_>นายกฯ<_>แถลงข่าวก่อนการแข่งขันศึก<_>#ช้างเอฟเอคัพ<_>นัดชิงชนะเลิศ', 'บัตรสมาชิกลดได้อีกไหมคับ', 'สนใจ<_>new<_>mazda2ครับ']

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 22/22 [00:05<00:00,  4.06it/s]
    0it [00:00, ?it/s]
    [DEBUG] labels [2 0 1 0]

    [DEBUG] label_encoder.classes_ [0 1 2 3]


    [DEBUG] (before: preprocessor) input_text ['วันที่6/3/61 เสียอารมณ์มาเร้ย. อาหารช้ามากกกกก นั่งคอย ประมาน20 นาที พนักงานยกน้ำจิ้มมาโดยไม่ใช้ถาด เอาถ้วยซ้อนๆกันออกมา จนเช็คบิล ตับหวานก้อไม่ได้', 'ยี่ห้อนี่ เขาชอบตั้งชื่อ ลงท้ายด้วย สระอา เทียน่า อัลเมร่า นาวาร่า เทอร่า พัลซ่า ยามาฮ่า แฮร่', 'สองวันสุดท้าย! ใครอยู่แถวแฟชั่น ไอส์แลนด์ มาร่วมสนุกกับกิจกรรมจากรองพื้นลอรีอัล ปารีส ทรูแมช ที่ร้าน Eve & Boy ได้ เรามีทั้งบริการเลือกเฉดรองพื้น 13 เฉดและแต่งหน้า Touch Up ฟรี! ที่สำคัญยังมีเกมส์ชิงของรางวัลจากรุ่น True Match มากมาย และบูธถ่ายรูปเก๋ๆ ภายในงาน ถ้าพลาดวันนี้ พรุ่งนี้ยังมีอีกวันนะคะ ตั้งแต่ เวลา 11:00 น. - 20:00น. #TrueToMyShade #TrueMatch #LorealParisTH', 'น้องแสงโสมอี้บ๋อ กะว่าน้องโซดา น้ำแข็งดี เอ๊ะหรือเหล้ากึ่งแก้วน้ำล้วนโซดาลอยดีหา 5555']

    [DEBUG] Apply preprocessor to texts.


    [DEBUG] (after: preprocessor) input_text ['วันที่6/3/61<_>เสียอารมณ์มาเร้ย.<_>อาหารช้ามาก<_>นั่งคอย<_>ประมาน20<_>นาที<_>พนักงานยกน้ำจิ้มมาโดยไม่ใช้ถาด<_>เอาถ้วยซ้อนๆกันออกมา<_>จนเช็คบิล<_>ตับหวานก้อไม่ได้', 'ยี่ห้อนี่<_>เขาชอบตั้งชื่อ<_>ลงท้ายด้วย<_>สระอา<_>เทียน่า<_>อัลเมร่า<_>นาวาร่า<_>เทอร่า<_>พัลซ่า<_>ยามาฮ่า<_>แฮร่', 'สองวันสุดท้าย!<_>ใครอยู่แถวแฟชั่น<_>ไอส์แลนด์<_>มาร่วมสนุกกับกิจกรรมจากรองพื้นลอรีอัล<_>ปารีส<_>ทรูแมช<_>ที่ร้าน<_>eve<_>&<_>boy<_>ได้<_>เรามีทั้งบริการเลือกเฉดรองพื้น<_>13<_>เฉดและแต่งหน้า<_>touch<_>up<_>ฟรี!<_>ที่สำคัญยังมีเกมส์ชิงของรางวัลจากรุ่น<_>true<_>match<_>มากมาย<_>และบูธถ่ายรูปเก๋ๆ<_>ภายในงาน<_>ถ้าพลาดวันนี้<_>พรุ่งนี้ยังมีอีกวันนะคะ<_>ตั้งแต่<_>เวลา<_>11:00<_>น.<_>-<_>20:00น.<_>#truetomyshade<_>#truematch<_>#lorealparisth', 'น้องแสงโสมอี้บ๋อ<_>กะว่าน้องโซดา<_>น้ำแข็งดี<_>เอ๊ะหรือเหล้ากึ่งแก้วน้ำล้วนโซดาลอยดีหา<_>5']

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.02it/s]
    0it [00:00, ?it/s]
    [DEBUG] labels [2 1 2 1]

    [DEBUG] label_encoder.classes_ [0 1 2 3]


    [DEBUG] (before: preprocessor) input_text ['ซื้อแต่ผ้าอนามัยแบบเย็นมาค่ะ แบบว่าอีห่ากูนอนไม่ได้', 'ครับ #phithanbkk', 'การด่าไปเหมือนได้บรรเทาความเครียดเฉยๆ แต่บีทีเอส (รถไฟฟ้า) มันสำนึกมั้ย ก็ไม่อ่ะ 😕', 'Cf clarins 5 ขวด 2850']

    [DEBUG] Apply preprocessor to texts.


    [DEBUG] (after: preprocessor) input_text ['ซื้อแต่ผ้าอนามัยแบบเย็นมาค่ะ<_>แบบว่าอีห่ากูนอนไม่ได้', 'ครับ<_>#phithanbkk', 'การด่าไปเหมือนได้บรรเทาความเครียดเฉยๆ<_>แต่บีทีเอส<_>(รถไฟฟ้า)<_>มันสำนึกมั้ย<_>ก็ไม่อ่ะ<_>😕', 'cf<_>clarins<_>5<_>ขวด<_>2850']

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  4.52it/s]
    0it [00:00, ?it/s]
    [INFO] Done.

    [INFO] Number of train examples = 21628
    [INFO] Number of batches per epoch (training set) = 1352
    [INFO] Number of validation examples = 2404
    [INFO] Number of batches per epoch (validation set) = 2404
    [INFO] Warmup ratio = 0.1
    [INFO] Warmup steps = 136
    [INFO] Learning rate: 3e-05
    [INFO] Logging steps: 10
    [INFO] FP16 training: True


    [INFO] TrainingArguments:
    TrainingArguments(output_dir='/workspace/checkpoints/roberthai-thwiki-spm/finetuned/wisesight_sentiment/', overwrite_output_dir=True, do_train=False, do_eval=None, do_predict=False, evaluate_during_training=False, evaluation_strategy=<EvaluationStrategy.EPOCH: 'epoch'>, prediction_loss_only=False, per_device_train_batch_size=16, per_device_eval_batch_size=16, per_gpu_train_batch_size=None, per_gpu_eval_batch_size=None, gradient_accumulation_steps=1, eval_accumulation_steps=None, learning_rate=3e-05, weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=1, max_steps=-1, warmup_steps=136, logging_dir='/workspace/logs/roberthai-thwiki-spm/finetuned/wisesight_sentiment/', logging_first_step=False, logging_steps=10, save_steps=500, save_total_limit=None, no_cuda=False, seed=2020, fp16=True, fp16_opt_level='O1', local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=10, dataloader_num_workers=0, past_index=-1, run_name='/workspace/checkpoints/roberthai-thwiki-spm/finetuned/wisesight_sentiment/', disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=True, metric_for_best_model='f1_micro', greater_is_better=True)



    Begin model finetuning.
    Selected optimization level O1:  Insert automatic casts around Pytorch functions and Tensor methods.

    Defaults for this optimization level are:
    enabled                : True
    opt_level              : O1
    cast_model_type        : None
    patch_torch_functions  : True
    keep_batchnorm_fp32    : None
    master_weights         : None
    loss_scale             : dynamic
    Processing user overrides (additional kwargs that are not None)...
    After processing overrides, optimization options are:
    enabled                : True
    opt_level              : O1
    cast_model_type        : None
    patch_torch_functions  : True
    keep_batchnorm_fp32    : None
    master_weights         : None
    loss_scale             : dynamic
    wandb: Offline run mode, not syncing to the cloud.
    wandb: W&B syncing is set to `offline` in this directory.  Run `wandb online` to enable cloud syncing.
    0%|          | 0/1352 [00:00<?, ?it/s]
    1%|          | 10/1352 [01:41<1:35:35,  4.27s/it]8823529411767e-06, 'epoch': 0.0073964497041420114}
    1%|▏         | 20/1352 [02:11<51:39,  2.33s/it]{'loss': 1.3437000274658204, 'learning_rate': 4.411764705882353e-06, 'epoch': 0.014792899408284023}
    2%|▏         | 30/1352 [02:27<20:12,  1.09it/s]{'loss': 1.2052518844604492, 'learning_rate': 6.61764705882353e-06, 'epoch': 0.022189349112426034}
    3%|▎         | 40/1352 [02:29<03:05,  7.08it/s]{'loss': 1.1024581909179687, 'learning_rate': 8.823529411764707e-06, 'epoch': 0.029585798816568046}
    4%|▎         | 50/1352 [02:44<08:19,  2.60it/s]029411764705883e-05, 'epoch': 0.03698224852071006}
    4%|▍         | 60/1352 [02:45<02:58,  7.25it/s]{'loss': 1.047784423828125, 'learning_rate': 1.323529411764706e-05, 'epoch': 0.04437869822485207}
                                                    {'loss': 1.0573417663574218, 'learning_rate': 1.5441176470588234e-05, 'epoch': 0.051775147928994084}
                                                    {'loss': 1.0460289001464844, 'learning_rate': 1.7647058823529414e-05, 'epoch': 0.05917159763313609}
                                                    {'loss': 1.0624427795410156, 'learning_rate': 1.9852941176470586e-05, 'epoch': 0.06656804733727811}
                                                    {'loss': 0.9531181335449219, 'learning_rate': 2.2058823529411766e-05, 'epoch': 0.07396449704142012}
                                                    {'loss': 1.06204833984375, 'learning_rate': 2.4264705882352942e-05, 'epoch': 0.08136094674556213}
    9%|▉         | 120/1352 [03:02<02:52,  7.16it/s]{'loss': 0.92392578125, 'learning_rate': 2.647058823529412e-05, 'epoch': 0.08875739644970414}
                                                    {'loss': 0.9776107788085937, 'learning_rate': 2.8676470588235295e-05, 'epoch': 0.09615384615384616}
    10%|█         | 140/1352 [03:04<02:31,  7.99it/s]{'loss': 1.0748321533203125, 'learning_rate': 2.9901315789473686e-05, 'epoch': 0.10355029585798817}
    11%|█         | 150/1352 [03:07<02:48,  7.14it/s]{'loss': 0.9978012084960938, 'learning_rate': 2.9654605263157896e-05, 'epoch': 0.11094674556213018}
    12%|█▏        | 160/1352 [03:08<02:47,  7.10it/s]{'loss': 0.8718536376953125, 'learning_rate': 2.9407894736842106e-05, 'epoch': 0.11834319526627218}
    13%|█▎        | 170/1352 [03:10<02:42,  7.28it/s]{'loss': 0.8958663940429688, 'learning_rate': 2.9161184210526316e-05, 'epoch': 0.1257396449704142}
                                                    {'loss': 0.8289108276367188, 'learning_rate': 2.8914473684210526e-05, 'epoch': 0.13313609467455623}
    14%|█▍        | 190/1352 [03:13<03:04,  6.29it/s]{'loss': 0.8470077514648438, 'learning_rate': 2.8667763157894736e-05, 'epoch': 0.14053254437869822}
    15%|█▍        | 200/1352 [03:29<1:32:46,  4.83s/it]{'loss': 0.8324371337890625, 'learning_rate': 2.8421052631578946e-05, 'epoch': 0.14792899408284024}
    16%|█▌        | 210/1352 [03:31<05:24,  3.52it/s]{'loss': 0.9709686279296875, 'learning_rate': 2.817434210526316e-05, 'epoch': 0.15532544378698224}
                                                    {'loss': 0.8655807495117187, 'learning_rate': 2.792763157894737e-05, 'epoch': 0.16272189349112426}
    17%|█▋        | 230/1352 [03:34<02:39,  7.04it/s]{'loss': 0.9475296020507813, 'learning_rate': 2.768092105263158e-05, 'epoch': 0.17011834319526628}
    18%|█▊        | 240/1352 [03:36<02:25,  7.63it/s]{'loss': 0.7989715576171875, 'learning_rate': 2.743421052631579e-05, 'epoch': 0.17751479289940827}
    18%|█▊        | 250/1352 [03:37<02:14,  8.22it/s]{'loss': 0.8985809326171875, 'learning_rate': 2.71875e-05, 'epoch': 0.1849112426035503}
                                                    {'loss': 0.8492599487304687, 'learning_rate': 2.694078947368421e-05, 'epoch': 0.19230769230769232}
                                                    {'loss': 0.8721343994140625, 'learning_rate': 2.669407894736842e-05, 'epoch': 0.1997041420118343}
                                                    {'loss': 0.7967071533203125, 'learning_rate': 2.644736842105263e-05, 'epoch': 0.20710059171597633}
    21%|██▏       | 290/1352 [03:43<03:20,  5.30it/s]{'loss': 0.7402099609375, 'learning_rate': 2.620065789473684e-05, 'epoch': 0.21449704142011836}
    22%|██▏       | 300/1352 [03:45<08:55,  1.96it/s]{'loss': 0.9380523681640625, 'learning_rate': 2.5953947368421054e-05, 'epoch': 0.22189349112426035}
    23%|██▎       | 310/1352 [03:47<03:09,  5.49it/s]{'loss': 0.9521942138671875, 'learning_rate': 2.5707236842105264e-05, 'epoch': 0.22928994082840237}
    24%|██▎       | 320/1352 [03:48<02:18,  7.45it/s]{'loss': 0.8440216064453125, 'learning_rate': 2.5460526315789474e-05, 'epoch': 0.23668639053254437}
    24%|██▍       | 330/1352 [03:50<02:08,  7.94it/s]{'loss': 0.875164794921875, 'learning_rate': 2.5213815789473684e-05, 'epoch': 0.2440828402366864}
                                                    {'loss': 0.8367584228515625, 'learning_rate': 2.4967105263157894e-05, 'epoch': 0.2514792899408284}
    26%|██▌       | 350/1352 [03:52<02:12,  7.58it/s]{'loss': 0.895458984375, 'learning_rate': 2.4720394736842104e-05, 'epoch': 0.2588757396449704}
    27%|██▋       | 360/1352 [03:54<02:42,  6.12it/s]{'loss': 0.935107421875, 'learning_rate': 2.4473684210526318e-05, 'epoch': 0.26627218934911245}
                                                    {'loss': 0.815704345703125, 'learning_rate': 2.4226973684210528e-05, 'epoch': 0.27366863905325445}
    28%|██▊       | 380/1352 [03:56<02:11,  7.42it/s]{'loss': 0.91475830078125, 'learning_rate': 2.3980263157894738e-05, 'epoch': 0.28106508875739644}
    29%|██▉       | 390/1352 [03:58<01:59,  8.06it/s]{'loss': 0.8481536865234375, 'learning_rate': 2.3733552631578948e-05, 'epoch': 0.28846153846153844}
    30%|██▉       | 400/1352 [04:02<04:39,  3.40it/s]{'loss': 0.8343109130859375, 'learning_rate': 2.348684210526316e-05, 'epoch': 0.2958579881656805}
    30%|███       | 410/1352 [04:03<02:09,  7.25it/s]{'loss': 0.86136474609375, 'learning_rate': 2.324013157894737e-05, 'epoch': 0.3032544378698225}
    31%|███       | 420/1352 [04:05<02:12,  7.02it/s]{'loss': 0.819671630859375, 'learning_rate': 2.299342105263158e-05, 'epoch': 0.3106508875739645}
    32%|███▏      | 430/1352 [04:06<02:04,  7.43it/s]{'loss': 0.8303619384765625, 'learning_rate': 2.274671052631579e-05, 'epoch': 0.3180473372781065}
                                                    {'loss': 0.7434661865234375, 'learning_rate': 2.25e-05, 'epoch': 0.3254437869822485}
                                                    {'loss': 0.7448455810546875, 'learning_rate': 2.225328947368421e-05, 'epoch': 0.3328402366863905}
    34%|███▍      | 460/1352 [04:10<02:15,  6.57it/s]{'loss': 0.6353759765625, 'learning_rate': 2.200657894736842e-05, 'epoch': 0.34023668639053256}
    35%|███▍      | 470/1352 [04:12<01:53,  7.78it/s]{'loss': 0.8759033203125, 'learning_rate': 2.175986842105263e-05, 'epoch': 0.34763313609467456}
                                                    {'loss': 0.74891357421875, 'learning_rate': 2.151315789473684e-05, 'epoch': 0.35502958579881655}
                                                    {'loss': 0.769342041015625, 'learning_rate': 2.1266447368421055e-05, 'epoch': 0.3624260355029586}
    37%|███▋      | 500/1352 [04:17<08:44,  1.62it/s]{'loss': 0.8095611572265625, 'learning_rate': 2.1019736842105265e-05, 'epoch': 0.3698224852071006}
    38%|███▊      | 510/1352 [04:19<02:15,  6.20it/s]{'loss': 0.753070068359375, 'learning_rate': 2.0773026315789475e-05, 'epoch': 0.3772189349112426}
                                                    {'loss': 0.756378173828125, 'learning_rate': 2.0526315789473685e-05, 'epoch': 0.38461538461538464}
    39%|███▉      | 530/1352 [04:21<01:49,  7.53it/s]{'loss': 0.784039306640625, 'learning_rate': 2.0279605263157895e-05, 'epoch': 0.39201183431952663}
                                                    {'loss': 0.7979827880859375, 'learning_rate': 2.0032894736842105e-05, 'epoch': 0.3994082840236686}
    41%|████      | 550/1352 [04:24<01:37,  8.21it/s]{'loss': 0.764959716796875, 'learning_rate': 1.9786184210526315e-05, 'epoch': 0.4068047337278107}
    41%|████▏     | 560/1352 [04:26<01:48,  7.32it/s]{'loss': 0.759918212890625, 'learning_rate': 1.9539473684210525e-05, 'epoch': 0.41420118343195267}
    42%|████▏     | 570/1352 [04:27<01:33,  8.38it/s]{'loss': 0.7657470703125, 'learning_rate': 1.9292763157894736e-05, 'epoch': 0.42159763313609466}
    43%|████▎     | 580/1352 [04:31<03:48,  3.37it/s]{'loss': 0.6763214111328125, 'learning_rate': 1.9046052631578946e-05, 'epoch': 0.4289940828402367}
                                                    {'loss': 0.691802978515625, 'learning_rate': 1.879934210526316e-05, 'epoch': 0.4363905325443787}
    44%|████▍     | 600/1352 [04:48<55:19,  4.41s/it]7894737e-05, 'epoch': 0.4437869822485207}
    45%|████▌     | 610/1352 [04:49<03:06,  3.98it/s]{'loss': 0.7890625, 'learning_rate': 1.830592105263158e-05, 'epoch': 0.4511834319526627}
                                                    {'loss': 0.769647216796875, 'learning_rate': 1.805921052631579e-05, 'epoch': 0.45857988165680474}
                                                    {'loss': 0.87633056640625, 'learning_rate': 1.78125e-05, 'epoch': 0.46597633136094674}
    47%|████▋     | 640/1352 [04:53<01:50,  6.46it/s]{'loss': 0.76488037109375, 'learning_rate': 1.756578947368421e-05, 'epoch': 0.47337278106508873}
    48%|████▊     | 650/1352 [04:55<01:31,  7.70it/s]{'loss': 0.754241943359375, 'learning_rate': 1.731907894736842e-05, 'epoch': 0.4807692307692308}
    49%|████▉     | 660/1352 [04:56<01:36,  7.15it/s]{'loss': 0.7030029296875, 'learning_rate': 1.707236842105263e-05, 'epoch': 0.4881656804733728}
    50%|████▉     | 670/1352 [04:57<01:22,  8.23it/s]{'loss': 0.933746337890625, 'learning_rate': 1.682565789473684e-05, 'epoch': 0.49556213017751477}
                                                    {'loss': 0.884130859375, 'learning_rate': 1.6578947368421053e-05, 'epoch': 0.5029585798816568}
    51%|█████     | 690/1352 [05:01<01:31,  7.26it/s]{'loss': 0.776885986328125, 'learning_rate': 1.6332236842105266e-05, 'epoch': 0.5103550295857988}
                                                    {'loss': 0.75533447265625, 'learning_rate': 1.6085526315789476e-05, 'epoch': 0.5177514792899408}
    53%|█████▎    | 710/1352 [05:06<01:45,  6.10it/s]{'loss': 0.723431396484375, 'learning_rate': 1.5838815789473687e-05, 'epoch': 0.5251479289940828}
    53%|█████▎    | 720/1352 [05:07<01:18,  8.02it/s]{'loss': 0.771697998046875, 'learning_rate': 1.5592105263157897e-05, 'epoch': 0.5325443786982249}
    54%|█████▍    | 730/1352 [05:08<01:23,  7.46it/s]{'loss': 0.72344970703125, 'learning_rate': 1.5345394736842107e-05, 'epoch': 0.5399408284023669}
    55%|█████▍    | 740/1352 [05:10<01:29,  6.81it/s]{'loss': 0.7661865234375, 'learning_rate': 1.5098684210526315e-05, 'epoch': 0.5473372781065089}
    55%|█████▌    | 750/1352 [05:11<01:14,  8.08it/s]{'loss': 0.7775390625, 'learning_rate': 1.4851973684210527e-05, 'epoch': 0.5547337278106509}
    56%|█████▌    | 760/1352 [05:12<01:17,  7.65it/s]{'loss': 0.68282470703125, 'learning_rate': 1.4605263157894737e-05, 'epoch': 0.5621301775147929}
    57%|█████▋    | 770/1352 [05:14<01:45,  5.53it/s]{'loss': 0.641754150390625, 'learning_rate': 1.4358552631578949e-05, 'epoch': 0.5695266272189349}
    58%|█████▊    | 780/1352 [05:15<01:09,  8.21it/s]{'loss': 0.752880859375, 'learning_rate': 1.4111842105263159e-05, 'epoch': 0.5769230769230769}
    58%|█████▊    | 790/1352 [05:17<01:11,  7.90it/s]{'loss': 0.763067626953125, 'learning_rate': 1.3865131578947369e-05, 'epoch': 0.584319526627219}
                                                    {'loss': 0.651007080078125, 'learning_rate': 1.361842105263158e-05, 'epoch': 0.591715976331361}
    60%|█████▉    | 810/1352 [05:21<01:12,  7.45it/s]{'loss': 0.72872314453125, 'learning_rate': 1.337171052631579e-05, 'epoch': 0.599112426035503}
    61%|██████    | 820/1352 [05:22<01:04,  8.22it/s]{'loss': 0.790142822265625, 'learning_rate': 1.3125e-05, 'epoch': 0.606508875739645}
    61%|██████▏   | 830/1352 [05:24<01:02,  8.31it/s]{'loss': 0.646343994140625, 'learning_rate': 1.287828947368421e-05, 'epoch': 0.613905325443787}
    62%|██████▏   | 840/1352 [05:25<01:13,  6.97it/s]{'loss': 0.830230712890625, 'learning_rate': 1.263157894736842e-05, 'epoch': 0.621301775147929}
    63%|██████▎   | 850/1352 [05:26<01:11,  7.04it/s]{'loss': 0.728875732421875, 'learning_rate': 1.2384868421052632e-05, 'epoch': 0.628698224852071}
    64%|██████▎   | 860/1352 [05:28<01:05,  7.49it/s]{'loss': 0.725, 'learning_rate': 1.2138157894736842e-05, 'epoch': 0.636094674556213}
    64%|██████▍   | 870/1352 [05:29<01:08,  7.08it/s]{'loss': 0.676788330078125, 'learning_rate': 1.1891447368421053e-05, 'epoch': 0.643491124260355}
    65%|██████▌   | 880/1352 [05:30<00:58,  8.06it/s]{'loss': 0.735028076171875, 'learning_rate': 1.1644736842105263e-05, 'epoch': 0.650887573964497}
    66%|██████▌   | 890/1352 [05:32<01:07,  6.89it/s]{'loss': 0.7079345703125, 'learning_rate': 1.1398026315789473e-05, 'epoch': 0.658284023668639}
    66%|██████▌   | 893/1352 [05:32<00:57,  7.92it/s]Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 32768.0
    67%|██████▋   | 900/1352 [05:36<07:34,  1.01s/it]{'loss': 0.784625244140625, 'learning_rate': 1.1151315789473684e-05, 'epoch': 0.665680473372781}
    67%|██████▋   | 910/1352 [05:38<01:13,  6.02it/s]{'loss': 0.72747802734375, 'learning_rate': 1.0904605263157894e-05, 'epoch': 0.6730769230769231}
    68%|██████▊   | 920/1352 [05:39<01:09,  6.20it/s]{'loss': 0.768280029296875, 'learning_rate': 1.0657894736842106e-05, 'epoch': 0.6804733727810651}
    69%|██████▉   | 930/1352 [05:40<01:06,  6.39it/s]{'loss': 0.761163330078125, 'learning_rate': 1.0411184210526316e-05, 'epoch': 0.6878698224852071}
    70%|██████▉   | 940/1352 [05:42<00:54,  7.52it/s]{'loss': 0.72337646484375, 'learning_rate': 1.0164473684210528e-05, 'epoch': 0.6952662721893491}
                                                    {'loss': 0.77833251953125, 'learning_rate': 9.917763157894738e-06, 'epoch': 0.7026627218934911}
    70%|███████   | 952/1352 [05:44<00:53,  7.43it/s]Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 16384.0
    71%|███████   | 960/1352 [05:45<00:56,  6.94it/s]{'loss': 0.593701171875, 'learning_rate': 9.671052631578948e-06, 'epoch': 0.7100591715976331}
    72%|███████▏  | 970/1352 [05:46<00:46,  8.14it/s]{'loss': 0.715960693359375, 'learning_rate': 9.424342105263158e-06, 'epoch': 0.7174556213017751}
    72%|███████▏  | 980/1352 [05:47<00:53,  7.00it/s]{'loss': 0.63238525390625, 'learning_rate': 9.177631578947368e-06, 'epoch': 0.7248520710059172}
    73%|███████▎  | 988/1352 [05:48<00:43,  8.31it/s]Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 8192.0
    73%|███████▎  | 990/1352 [05:49<00:41,  8.74it/s]{'loss': 0.8906982421875, 'learning_rate': 8.93092105263158e-06, 'epoch': 0.7322485207100592}
    74%|███████▍  | 1000/1352 [06:07<30:06,  5.13s/it]{'loss': 0.68482666015625, 'learning_rate': 8.68421052631579e-06, 'epoch': 0.7396449704142012}
                                                    {'loss': 0.641973876953125, 'learning_rate': 8.4375e-06, 'epoch': 0.7470414201183432}
    75%|███████▌  | 1020/1352 [06:09<00:53,  6.24it/s]{'loss': 0.7864013671875, 'learning_rate': 8.19078947368421e-06, 'epoch': 0.7544378698224852}
    76%|███████▌  | 1030/1352 [06:11<00:43,  7.46it/s]{'loss': 0.738299560546875, 'learning_rate': 7.94407894736842e-06, 'epoch': 0.7618343195266272}
    77%|███████▋  | 1040/1352 [06:12<00:38,  8.05it/s]{'loss': 0.84078369140625, 'learning_rate': 7.697368421052632e-06, 'epoch': 0.7692307692307693}
    78%|███████▊  | 1050/1352 [06:13<00:42,  7.17it/s]{'loss': 0.752410888671875, 'learning_rate': 7.450657894736843e-06, 'epoch': 0.7766272189349113}
    78%|███████▊  | 1060/1352 [06:15<00:37,  7.70it/s]{'loss': 0.72764892578125, 'learning_rate': 7.203947368421053e-06, 'epoch': 0.7840236686390533}
    79%|███████▉  | 1070/1352 [06:16<00:34,  8.29it/s]{'loss': 0.7495849609375, 'learning_rate': 6.957236842105264e-06, 'epoch': 0.7914201183431953}
    80%|███████▉  | 1080/1352 [06:17<00:35,  7.59it/s]{'loss': 0.706201171875, 'learning_rate': 6.710526315789474e-06, 'epoch': 0.7988165680473372}
    81%|████████  | 1090/1352 [06:19<00:40,  6.49it/s]{'loss': 0.690985107421875, 'learning_rate': 6.463815789473684e-06, 'epoch': 0.8062130177514792}
    81%|████████▏ | 1100/1352 [06:23<03:46,  1.11it/s]{'loss': 0.6723876953125, 'learning_rate': 6.217105263157895e-06, 'epoch': 0.8136094674556213}
    82%|████████▏ | 1110/1352 [06:24<00:35,  6.82it/s]{'loss': 0.74981689453125, 'learning_rate': 5.970394736842105e-06, 'epoch': 0.8210059171597633}
    83%|████████▎ | 1120/1352 [06:25<00:29,  7.86it/s]{'loss': 0.8501708984375, 'learning_rate': 5.723684210526316e-06, 'epoch': 0.8284023668639053}
    84%|████████▎ | 1130/1352 [06:27<00:30,  7.17it/s]{'loss': 0.7604736328125, 'learning_rate': 5.476973684210527e-06, 'epoch': 0.8357988165680473}
    84%|████████▍ | 1140/1352 [06:28<00:32,  6.45it/s]{'loss': 0.729522705078125, 'learning_rate': 5.230263157894737e-06, 'epoch': 0.8431952662721893}
    85%|████████▌ | 1150/1352 [06:30<00:26,  7.71it/s]{'loss': 0.702081298828125, 'learning_rate': 4.983552631578948e-06, 'epoch': 0.8505917159763313}
    86%|████████▌ | 1160/1352 [06:31<00:24,  7.80it/s]{'loss': 0.74451904296875, 'learning_rate': 4.736842105263158e-06, 'epoch': 0.8579881656804734}
    87%|████████▋ | 1170/1352 [06:33<00:27,  6.65it/s]{'loss': 0.703253173828125, 'learning_rate': 4.490131578947369e-06, 'epoch': 0.8653846153846154}
    87%|████████▋ | 1180/1352 [06:34<00:22,  7.66it/s]{'loss': 0.589453125, 'learning_rate': 4.243421052631579e-06, 'epoch': 0.8727810650887574}
    88%|████████▊ | 1190/1352 [06:35<00:26,  6.16it/s]{'loss': 0.649810791015625, 'learning_rate': 3.99671052631579e-06, 'epoch': 0.8801775147928994}
                                                    {'loss': 0.62998046875, 'learning_rate': 3.75e-06, 'epoch': 0.8875739644970414}
    89%|████████▉ | 1210/1352 [06:40<00:19,  7.11it/s]94736842106e-06, 'epoch': 0.8949704142011834}
    90%|█████████ | 1220/1352 [06:41<00:16,  7.82it/s]{'loss': 0.6827392578125, 'learning_rate': 3.256578947368421e-06, 'epoch': 0.9023668639053254}
    91%|█████████ | 1230/1352 [06:42<00:15,  7.94it/s]{'loss': 0.73472900390625, 'learning_rate': 3.009868421052632e-06, 'epoch': 0.9097633136094675}
                                                    {'loss': 0.826873779296875, 'learning_rate': 2.763157894736842e-06, 'epoch': 0.9171597633136095}
    92%|█████████▏| 1250/1352 [06:45<00:12,  7.87it/s]{'loss': 0.717706298828125, 'learning_rate': 2.5164473684210525e-06, 'epoch': 0.9245562130177515}
                                                    {'loss': 0.754278564453125, 'learning_rate': 2.2697368421052634e-06, 'epoch': 0.9319526627218935}
    94%|█████████▍| 1270/1352 [06:48<00:12,  6.46it/s]{'loss': 0.720526123046875, 'learning_rate': 2.023026315789474e-06, 'epoch': 0.9393491124260355}
    95%|█████████▍| 1280/1352 [06:49<00:09,  7.83it/s]57894736842e-06, 'epoch': 0.9467455621301775}
                                                    {'loss': 0.7333740234375, 'learning_rate': 1.5296052631578948e-06, 'epoch': 0.9541420118343196}
                                                    {'loss': 0.60689697265625, 'learning_rate': 1.2828947368421053e-06, 'epoch': 0.9615384615384616}
    97%|█████████▋| 1310/1352 [06:56<00:05,  7.35it/s]{'loss': 0.65772705078125, 'learning_rate': 1.0361842105263158e-06, 'epoch': 0.9689349112426036}
    98%|█████████▊| 1320/1352 [06:57<00:04,  7.71it/s]{'loss': 0.7178466796875, 'learning_rate': 7.894736842105263e-07, 'epoch': 0.9763313609467456}
    98%|█████████▊| 1330/1352 [06:58<00:02,  7.81it/s]{'loss': 0.77506103515625, 'learning_rate': 5.427631578947369e-07, 'epoch': 0.9837278106508875}
                                                    {'loss': 0.62127685546875, 'learning_rate': 2.9605263157894736e-07, 'epoch': 0.9911242603550295}
    100%|█████████▉| 1350/1352 [07:01<00:00,  7.80it/s]{'loss': 0.721728515625, 'learning_rate': 4.934210526315789e-08, 'epoch': 0.9985207100591716}
                                                    {'eval_loss': 0.6999529004096985, 'eval_accuracy': 0.7096505823627288, 'eval_f1_micro': 0.7096505823627288, 'eval_precision_micro': 0.7096505823627288, 'eval_recall_micro': 0.7096505823627288, 'eval_f1_macro': 0.5704574545076051, 'eval_precision_macro': 0.6174460882580549, 'eval_recall_macro': 0.100%|██████████| 1352/1352 [07:06<00:00,  8.20it/s] 1.0}
    100%|██████████| 1352/1352 [07:10<00:00,  3.14it/s]
    Done.

    [INFO] Done.

    [INDO] Begin saving best checkpoint.
    [INFO] Done.


    Begin model evaluation on test set.
    98%|█████████▊| 164/167 [00:05<00:00, 28.47it/s][DEBUG] label_ids = [2 1 2 ... 0 1 1]
    Evaluation on test set (dataset: wisesight_sentiment)
    eval_loss : 0.7032
    eval_accuracy : 0.7080
    eval_f1_micro : 0.7080
    eval_precision_micro : 0.7080
    eval_recall_micro : 0.7080
    eval_f1_macro : 0.5519
    eval_precision_macro : 0.6141
    eval_recall_macro : 0.5262
    eval_nb_samples : 2671.0000

    wandb: Waiting for W&B process to finish, PID 69657
    wandb: Program ended successfully.
    wandb: Find user logs for this run at: /workspace/scripts/wandb/offline-run-20210115_095447-1iwbiyf3/logs/debug.log
    wandb: Find internal logs for this run at: /workspace/scripts/wandb/offline-run-20210115_095447-1iwbiyf3/logs/debug-internal.log
    wandb: Run summary:
    wandb:                                                                   loss 0.72173
    wandb:                                                          learning_rate 0.0
    wandb:                                                                  epoch 1.0
    wandb:                                                             total_flos 2230830350876160
    wandb:                                                                  _step 1352
    wandb:                                                               _runtime 442
    wandb:                                                             _timestamp 1610704930
    wandb:                                                     test-set_eval_loss 0.70315
    wandb:                                                 test-set_eval_accuracy 0.70797
    wandb:                                                 test-set_eval_f1_micro 0.70797
    wandb:                                          test-set_eval_precision_micro 0.70797
    wandb:                                             test-set_eval_recall_micro 0.70797
    wandb:                                                 test-set_eval_f1_macro 0.55186
    wandb:                                          test-set_eval_precision_macro 0.61413
    wandb:                                             test-set_eval_recall_macro 0.52615
    wandb:                                               test-set_eval_nb_samples 2671
    wandb:                                                              eval_loss 0.69995
    wandb:                                                          eval_accuracy 0.70965
    wandb:                                                          eval_f1_micro 0.70965
    wandb:                                                   eval_precision_micro 0.70965
    wandb:                                                      eval_recall_micro 0.70965
    wandb:                                                          eval_f1_macro 0.57046
    wandb:                                                   eval_precision_macro 0.61745
    wandb:                                                      eval_recall_macro 0.54565
    wandb:                                                        eval_nb_samples 2404
    wandb: Run history:
    wandb:                   loss █▆▅▄▅▃▄▃▃▃▃▄▃▁▂▃▃▂▃▂▄▃▂▂▃▂▃▃▁▂▃▂▂▃▂▁▂▂▁▁
    wandb:          learning_rate ▂▃▅▇███▇▇▇▇▇▆▆▆▆▆▅▅▅▅▅▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▁▁▁
    wandb:                  epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
    wandb:             total_flos ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
    wandb:                  _step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
    wandb:               _runtime ▁▁▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇██
    wandb:             _timestamp ▁▁▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇██
    wandb:              eval_loss ▁
    wandb:          eval_accuracy ▁
    wandb:          eval_f1_micro ▁
    wandb:   eval_precision_micro ▁
    wandb:      eval_recall_micro ▁
    wandb:          eval_f1_macro ▁
    wandb:   eval_precision_macro ▁
    wandb:      eval_recall_macro ▁
    wandb:        eval_nb_samples ▁
    wandb: 
    wandb: You can sync this run to the cloud by running:
    wandb: wandb sync /workspace/scripts/wandb/offline-run-20210115_095447-1iwbiyf3
    root@IST-DGX01:/workspace/scripts# 


    ```
    </details>