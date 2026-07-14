from __future__ import annotations

import pickle
import argparse
import time
from pathlib import Path

from .train_bpe import train_bpe


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    args = parser.parse_args()

    if not args.input_path.is_file():
        parser.error(f"Input file does not exist: {args.input_path}")

    input_path = args.input_path
    vocab_size = args.vocab_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.pkl"
    merges_path = output_dir / "merges.pkl"

    special_tokens = ["<|endoftext|>"]

    start_time = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    elapsed_time = time.perf_counter() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f)

    with merges_path.open("wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()