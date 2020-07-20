import glob
import logging
logging.basicConfig(level=logging.INFO)

from transformers import CamembertTokenizer
from tokenizers import ( ByteLevelBPETokenizer,
                        CharBPETokenizer,
                        SentencePieceBPETokenizer)

#argparse
import argparse
# python train_tokenizer_roberthai.py --output_dir data/tokenizer/bpe_enth_52000 \
# --vocab_size 52000 --min_frequency 2 --train_dir data/train_lm

def main():
    #argparser
    parser = argparse.ArgumentParser(
        prog="train_mlm_camembert_thai.py",
        description="train mlm for Camembert with huggingface Trainer",
    )
    
    #required
    parser.add_argument("--bpe_tokenizer", type=str, default='sentencepiece', help='Specify the name of BPE Tokenizer')
    parser.add_argument("--vocab_size", type=int, default=52000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--train_dir", type=str,)
    parser.add_argument("--output_dir", type=str,)
    parser.add_argument("--ext", type=str, default='.txt')
    
    args = parser.parse_args()
    
    fnames = [str(x) for x in glob.glob(f"{args.train_dir}/*{args.ext}")]

    # Initialize a tokenizer
    if args.bpe_tokenizer == 'byte_level':
        _BPE_TOKENIZER = ByteLevelBPETokenizer()
    if args.bpe_tokenizer == 'char':
        _BPE_TOKENIZER = CharBPETokenizer()
    if args.bpe_tokenizer == 'sentencepiece':
        _BPE_TOKENIZER = SentencePieceBPETokenizer()

    tokenizer = _BPE_TOKENIZER

    # Customize training
    tokenizer.train(files=fnames, vocab_size=args.vocab_size, 
                    min_frequency=args.min_frequency, 
                    special_tokens=[
                        "<s>",
                        "<pad>",
                        "</s>",
                        "<unk>",
                        "<mask>",
                    ])

    # Save files to disk
    tokenizer.save_model(args.output_dir)
    
    #test
    tokenizer = CamembertTokenizer.from_pretrained(args.output_dir)
    print(tokenizer.encode_plus('สวัสดีครับ hello world'))

if __name__ == "__main__":
    main()