# Tokenize text word SEFR tokenizer for `fake_sefr_cut` tokenizer

The word level tokenizer that we use has pre_tokenize hook that will pre tokenize text before passing to the tokenizer unfortunately, this operate on instance by instance basis make this unsuitable for sefr_cut since it is very slow compared to other pretokenizers. To speed up the process, we use `multiprocessing` to reduce the processing time and output text with special token that will be use as a split token for fake_sefr_cut afterward. For example, if the text `hello world` is splited into `['hello', 'world']` by sefr_cut, the output will be `hello<|>world` where `<|>` is the delimiter.

## Instruction

1) Install required library

    As `sefr_cut` relies on `deepcut`, an ML-based word tokenization model, it requires to install Tensorflow. Tensorflow can be installed via the following commands:

    ```bash
    pip install tensorflow

    # For machine with GPUs

    pip install tensorflow-gpu
    ```

    Then, install `sefr_cut`.

    ```bash
    pip install sefr_cut==0.2
    ```


2) Perform word tokenization

    The following command will read text files line by line from `input_folder` and output it to `output_folder` where `chunk_size` (number of lines to be read for each process) is number of process multiply by 200.

    ```bash
    cd scripts
    python ./sefr_cut_pre_tokenizer.py \
    --input_folder ../data/dataset/thwiki-20200820/5_split/train \
    --output_folder ../data/tokenizers/thwiki-20200820/seft_cut/ \
    --chunk_size $(($(nproc) * 200)) \
    --overwrite
    ```
