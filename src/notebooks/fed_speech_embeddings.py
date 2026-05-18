"""
fed_speech_embeddings.py
========================
Generates dense vector embeddings from all Fed speeches using
FOMC-RoBERTa (gtfintechlab/FOMC-RoBERTa) with a chunk-and-average
strategy that embeds the *entire* speech, not just the first 512 tokens.

Algorithm
---------
For each speech:
  1. Tokenize the full text (no truncation).
  2. Split token ids into overlapping windows of MAX_LENGTH tokens
     (stride STRIDE tokens between consecutive windows).
  3. Run each window through the encoder, mean-pool over non-padding
     positions to get one 768-dim vector per window.
  4. Average all window vectors → single 768-dim speech embedding.

Outputs  (data/embeddings/)
---------------------------
  speech_embeddings_full.csv  — N x 768 raw embeddings + metadata
  speech_embeddings_pca.csv   — N x N_PCA PCA dims + metadata  ← TFT input
  pca_model.pkl               — fitted PCA model (for inference on new speeches)

Runtime
-------
  CPU  : ~4–8 min per 100 speeches  (expect several hours for 6 k speeches)
  GPU  : ~10–20 x faster — strongly recommended for the full dataset
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model options (comment/uncomment as needed):
#
#   "gtfintechlab/FOMC-RoBERTa"     <- best fit for FOMC text, but GATED:
#                                      request access at https://huggingface.co/gtfintechlab/FOMC-RoBERTa
#                                      then set HF_TOKEN env var or run `huggingface-cli login`
#
#   "ProsusAI/finbert"              <- publicly available, fine-tuned on financial text (default)
#
#   "Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier"  <- central bank focused, public
MODEL_NAME = "ProsusAI/finbert"

MAX_LENGTH  = 512   # hard limit for RoBERTa / BERT models
STRIDE      = 128   # token overlap between consecutive windows
CHUNK_BATCH = 16    # windows processed in a single forward pass
                    # lower this if you hit GPU OOM errors

N_PCA       = 20    # PCA components kept for TFT input
                    # tune after inspecting explained-variance plot

# Metadata columns forwarded to output CSVs (for TFT static covariates)
META_COLS = [
    "Date",
    "Authorname",
    "CentralBank",
    "Role",
    "position_in_fed",
    "district",
    "female",
    "minority",
    "federal_reserve",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Device: GPU — {gpu_name} ({gpu_mem:.1f} GB VRAM)")
else:
    print("Device: CPU — no CUDA GPU detected. Embedding will be slow.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]
data_dir     = project_root / "data"
output_dir   = data_dir / "embeddings"
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load speeches
# ---------------------------------------------------------------------------

def _load_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as fh:
            return pd.read_csv(fh, low_memory=False)


def load_speeches() -> pd.DataFrame:
    """Load speeches, preferring the metadata-enriched file."""
    meta_zip    = data_dir / "speeches_with_metadata.csv.zip"
    cleaned_zip = data_dir / "cleaned_speeches.csv.zip"
    cleaned_csv = data_dir / "cleaned_speeches.csv"

    if meta_zip.exists():
        print(f"Loading {meta_zip.name} …")
        df = _load_csv_from_zip(meta_zip)
    elif cleaned_zip.exists():
        print(f"Loading {cleaned_zip.name} …")
        df = _load_csv_from_zip(cleaned_zip)
    elif cleaned_csv.exists():
        print(f"Loading {cleaned_csv.name} …")
        df = pd.read_csv(cleaned_csv, low_memory=False)
    else:
        raise FileNotFoundError(
            "No speech CSV found in data/. Expected one of: "
            "speeches_with_metadata.csv.zip, cleaned_speeches.csv.zip, "
            "cleaned_speeches.csv"
        )

    # Prefer text-cleaned; fall back to text
    if "text-cleaned" in df.columns:
        df["_text"] = df["text-cleaned"].fillna(df.get("text", "")).astype(str)
    else:
        df["_text"] = df["text"].fillna("").astype(str)

    print(f"  {len(df):,} speeches loaded.")
    return df


speeches_df = load_speeches()
speeches_df = speeches_df.head(50)  # remove this line to process all speeches

# Keep only metadata columns that actually exist in this file
available_meta = [c for c in META_COLS if c in speeches_df.columns]

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

print(f"\nLoading model '{MODEL_NAME}' …")
tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)
model       = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

hidden_size = model.config.hidden_size   # 768 for RoBERTa-base / BERT-base
print(f"  Hidden size: {hidden_size}  |  Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Chunk-and-average helpers
# ---------------------------------------------------------------------------

def _build_windows(token_ids: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """
    Split a flat 1-D token-id tensor into overlapping MAX_LENGTH windows.
    Each window has [CLS] body [SEP] padding as required by RoBERTa.
    Returns a list of dicts compatible with model(**window).
    """
    cls_id  = tokenizer.cls_token_id
    sep_id  = tokenizer.sep_token_id
    pad_id  = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    body_size = MAX_LENGTH - 2          # tokens between [CLS] and [SEP]
    step      = body_size - STRIDE      # advance per window (= 510 - 128 = 382)

    total   = len(token_ids)
    windows = []
    start   = 0

    while True:
        end  = min(start + body_size, total)
        body = token_ids[start:end]

        ids = torch.cat([
            torch.tensor([cls_id], dtype=torch.long),
            body,
            torch.tensor([sep_id], dtype=torch.long),
        ])

        pad_len = MAX_LENGTH - len(ids)
        mask    = torch.ones(len(ids), dtype=torch.long)
        if pad_len > 0:
            ids  = torch.cat([ids,  torch.full((pad_len,), pad_id, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len,           dtype=torch.long)])

        windows.append({"input_ids": ids.unsqueeze(0), "attention_mask": mask.unsqueeze(0)})

        if end == total:
            break
        start += step

    return windows


@torch.no_grad()
def _embed_windows(windows: list[dict[str, torch.Tensor]]) -> np.ndarray:
    """
    Run windows through the model in batches of CHUNK_BATCH.
    Mean-pool each window over non-padding token positions.
    Return the average across all windows: shape (hidden_size,).
    """
    window_embs: list[np.ndarray] = []

    for i in range(0, len(windows), CHUNK_BATCH):
        batch = windows[i : i + CHUNK_BATCH]
        ids   = torch.cat([w["input_ids"]      for w in batch], dim=0).to(DEVICE)
        mask  = torch.cat([w["attention_mask"] for w in batch], dim=0).to(DEVICE)

        out    = model(input_ids=ids, attention_mask=mask)          # last_hidden_state: (B, T, H)
        mask_f = mask.unsqueeze(-1).float()                         # (B, T, 1)
        embs   = (out.last_hidden_state * mask_f).sum(dim=1)        # (B, H)
        embs   = embs / mask_f.sum(dim=1).clamp(min=1e-9)           # (B, H)  mean pool
        window_embs.append(embs.cpu().float().numpy())

    stacked = np.vstack(window_embs)   # (n_windows, H)
    return stacked.mean(axis=0)        # (H,)  average across windows


def embed_speech(text: str) -> np.ndarray:
    """
    Full-speech embedding via chunk-and-average.
    Returns a 1-D numpy array of shape (hidden_size,).
    """
    if not text.strip():
        return np.zeros(hidden_size, dtype=np.float32)

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
    )["input_ids"][0]

    if len(token_ids) == 0:
        return np.zeros(hidden_size, dtype=np.float32)

    windows = _build_windows(token_ids)
    return _embed_windows(windows)


# ---------------------------------------------------------------------------
# Embed all speeches
# ---------------------------------------------------------------------------

n = len(speeches_df)
print(f"\nEmbedding {n:,} speeches (this will take a while on CPU) …")

all_embeddings: list[np.ndarray] = []

for i, (_, row) in enumerate(speeches_df.iterrows()):
    if i % 100 == 0:
        print(f"  {i:>5} / {n}")
    all_embeddings.append(embed_speech(row["_text"]))

embeddings = np.vstack(all_embeddings)   # (N, hidden_size)
print(f"\nEmbeddings matrix: {embeddings.shape}")

# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

print(f"\nFitting PCA ({N_PCA} components) …")
pca     = PCA(n_components=N_PCA, random_state=42)
reduced = pca.fit_transform(embeddings)

cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"  Cumulative explained variance:")
for k in [5, 10, 15, N_PCA]:
    if k <= N_PCA:
        print(f"    {k:>3} components → {cumvar[k-1]:.1%}")

with open(output_dir / "pca_model.pkl", "wb") as fh:
    pickle.dump(pca, fh)
print(f"  pca_model.pkl saved.")

# ---------------------------------------------------------------------------
# Build & save output DataFrames
# ---------------------------------------------------------------------------

meta_df = speeches_df[available_meta].copy().reset_index(drop=True)

# Normalise date column to YYYY-MM-DD string for easy merging with macro data
if "Date" in meta_df.columns:
    meta_df["Date"] = pd.to_datetime(meta_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

# Full 768-dim embeddings + metadata
emb_cols = [f"emb_{i}" for i in range(hidden_size)]
emb_df   = pd.concat([meta_df, pd.DataFrame(embeddings, columns=emb_cols)], axis=1)
emb_path = output_dir / "speech_embeddings_full.csv"
emb_df.to_csv(emb_path, index=False)

# PCA-reduced embeddings + metadata  ← TFT input
pca_cols = [f"pca_{i}" for i in range(N_PCA)]
pca_df   = pd.concat([meta_df, pd.DataFrame(reduced, columns=pca_cols)], axis=1)
pca_path = output_dir / "speech_embeddings_pca.csv"
pca_df.to_csv(pca_path, index=False)

print(f"\nOutputs written to {output_dir}/")
print(f"  speech_embeddings_full.csv  shape={emb_df.shape}")
print(f"  speech_embeddings_pca.csv   shape={pca_df.shape}")
print(f"  pca_model.pkl")
print("\nDone.")
