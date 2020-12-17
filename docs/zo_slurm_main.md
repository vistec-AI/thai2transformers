# Training on cluster

`zo_slurm_main.sh` is the main script that we use to layout all the script together and ran on the cluster. The variables should be self explanable and tracable. The script will queue multiple jobs consist of tokenizer training, preprocessing, language model training.

## How to

To use the script, we would need to modify variables that control arguments, if you ever need more arguments that do not exists in script already you can always declare more and extend it to your own need.

For examples, if we want to train a model with newmm tokenizer and learning rate = 1e-7. We would need to change the following variables.

```bash
export PROJECT_PRE_TOKENIZER_TYPE="newmm"
export PROJECT_LEARNING_RATE=1e-7
export PROJECT_TOKENIZER_TYPE="ThaiWordsNewmmTokenizer"
```

Notice how `$PROJECT_PRE_TOKENIZER_TYPE` is not the same with `$PROJECT_TOKENIZER_TYPE` since `$PROJECT_TOKENIZER_TYPE` is corresponding to the tokenizer class, but  `$PROJECT_PRE_TOKENIZER_TYPE` is corresponding to pre tokenizer function.

| pre_tokenizer_type | tokenizer_type |
| ---- | ---- |
| newmm | ThaiWordsNewmmTokenizer |
| syllable | ThaiWordsSyllableTokenizer |
| fake_sefr_cut | FakeSefrCutTokenizer |

After you modified the variables and to some extend other scripts that `zo_slurm_main.sh` call you should be able to start the jobs by running the script.

```bash
./zo_slurm_main.sh
```

It will ask you to confirm parameters if it is all good press `Y` and it will queue the jobs.

