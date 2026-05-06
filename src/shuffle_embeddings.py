"""
Shuffle embeddings script.

Loads an embeddings CSV, shuffles the embedding columns (last 20) independently
of the metadata columns, and saves the result to embeddings/shuffled/.

Usage:
    python src/shuffle_embeddings.py [--input <path>] [--seed <int>]

Defaults to embeddings_pca_cls_full_fomc-roberta.csv.
"""

import argparse
import os
import numpy as np
import pandas as pd


EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings")
DEFAULT_INPUT = os.path.join(
    EMBEDDINGS_DIR, "fomc-roberta", "embeddings_pca_cls_full_fomc-roberta.csv"
)


def shuffle_embeddings(input_path: str, seed: int | None = None) -> str:
    df = pd.read_csv(input_path)

    # The last 20 columns are the embedding dimensions; everything before is metadata.
    embedding_cols = df.columns[-20:].tolist()
    meta_cols = df.columns[:-20].tolist()

    rng = np.random.default_rng(seed)

    # Shuffle the rows of the embedding block (each row's embedding goes to a
    # different row, metadata stays in place).
    embedding_values = df[embedding_cols].values.copy()
    shuffled_indices = rng.permutation(len(embedding_values))
    shuffled_embeddings = embedding_values[shuffled_indices]

    df_shuffled = df[meta_cols].copy()
    df_shuffled[embedding_cols] = shuffled_embeddings

    # Derive output path: embeddings/<subdir>/shuffled/<stem>_shuffled.csv
    input_dir = os.path.dirname(os.path.abspath(input_path))
    stem = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(input_dir, "shuffled")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{stem}_shuffled.csv")

    df_shuffled.to_csv(output_path, index=False)
    print(f"Shuffled embeddings saved to: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Shuffle embedding columns in a CSV.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the source embeddings CSV (default: fomc-roberta PCA file).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    args = parser.parse_args()
    shuffle_embeddings(args.input, args.seed)


if __name__ == "__main__":
    main()
