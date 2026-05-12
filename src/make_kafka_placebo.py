"""
make_kafka_placebo.py
---------------------
Replace the text content of Fed speeches with Kafka paragraphs (placebo experiment).

Keeps all speech metadata (Date, Authorname, Role, etc.) intact.
Randomly assigns one paragraph from the pooled Kafka corpus to each speech.
If the corpus has fewer paragraphs than speeches, paragraphs are reused (sampling
with replacement).

Usage
-----
    python src/make_kafka_placebo.py \
        --kafka-files data/kafka_trial.txt data/kafka_metamorphosis.txt \
        --output data/speeches_kafka_placebo.csv.zip \
        --seed 42

Output
------
    data/speeches_kafka_placebo.csv.zip
        Same schema as speeches_with_metadata.csv.zip but with 'text' and
        'text-cleaned' replaced by randomly assigned Kafka paragraphs.
"""

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
data_dir     = project_root / "data"
# ---------------------------------------------------------------------------


def _load_speeches() -> pd.DataFrame:
    meta_zip = data_dir / "speeches_with_metadata.csv.zip"
    if not meta_zip.exists():
        raise FileNotFoundError(f"Expected {meta_zip}")
    print(f"Loading {meta_zip.name} …")
    with zipfile.ZipFile(meta_zip) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as fh:
            df = pd.read_csv(fh, low_memory=False)
    print(f"  {len(df):,} speeches loaded.")
    return df


def _load_kafka_paragraphs(kafka_files: list[Path]) -> list[str]:
    """Read text files, split by blank lines, return non-trivial paragraphs."""
    raw = []
    for path in kafka_files:
        # files from Project Gutenberg are often Windows-1252; fall back to latin-1
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="windows-1252")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")
        # split on one or more blank lines
        chunks = re.split(r"\n\s*\n", text)
        raw.extend(chunks)

    # clean and filter: strip whitespace, drop very short chunks (< 100 chars)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in raw]
    paragraphs = [p for p in paragraphs if len(p) >= 100]
    print(f"  {len(paragraphs):,} usable Kafka paragraphs from {len(kafka_files)} file(s).")
    return paragraphs


def make_placebo(kafka_files: list[Path], output_path: Path, seed: int) -> None:
    df         = _load_speeches()
    paragraphs = _load_kafka_paragraphs(kafka_files)
    rng        = np.random.default_rng(seed)

    n = len(df)
    # sample with replacement if corpus is smaller than number of speeches
    replace = len(paragraphs) < n
    chosen  = rng.choice(paragraphs, size=n, replace=replace)

    df["text"]         = chosen
    df["text-cleaned"] = chosen   # keep both columns consistent

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, compression="zip")
    print(f"Saved placebo speech file: {output_path}  ({n:,} rows)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Kafka placebo speech dataset")
    p.add_argument(
        "--kafka-files",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more plain-text Kafka files (e.g. kafka_trial.txt)",
    )
    p.add_argument(
        "--output",
        default=str(data_dir / "speeches_kafka_placebo.csv.zip"),
        metavar="PATH",
        help="Output path for the placebo CSV (default: data/speeches_kafka_placebo.csv.zip)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for paragraph assignment (default: 42)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_placebo(
        kafka_files=[Path(f) for f in args.kafka_files],
        output_path=Path(args.output),
        seed=args.seed,
    )
