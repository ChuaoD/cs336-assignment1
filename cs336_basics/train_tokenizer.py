from __future__ import annotations

import pickle
import argparse
import time

from .train_bpe import train_bpe


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    input_path = args.input_path
    vocab_size = args.vocab_size
    special_tokens = ["<|endoftext|>"]

    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    end_time = time.time()
    print(end_time - start_time)

    tokenizer_data = {
        "vocab": vocab,
        "merges": merges,
        "special_tokens": special_tokens
    }

    with open(args.output_path, "wb") as f:
        pickle.dump(tokenizer_data, f)


if __name__ == "__main__":
    main()