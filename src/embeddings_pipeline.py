"""
Embedding pipeline for Fed speeches.

Supported models:
  finbert      — ProsusAI/finbert (BERT-based encoder)
  fomc-roberta — gtfintechlab/FOMC-RoBERTa (RoBERTa-based encoder, gated HuggingFace repo)
  llama3.1     — LLaMA 3.1 loaded from a local path (decoder-only, mean pooling only)

Supported truncation modes:
  512    — first 512 tokens only
  full   — chunk-and-average over the full speech

Outputs are written to:
  data/embeddings/{model}/
    embeddings_full_{pooling}_{truncation}_{model}.csv.zip
    embeddings_pca_{pooling}_{truncation}_{model}.csv
    pca_model_{pooling}_{truncation}_{model}.pkl

Usage examples:
  python embeddings_pipeline.py --model finbert --truncation 512
  python embeddings_pipeline.py --model finbert --truncation full --cutoff 2022-01-01
  python embeddings_pipeline.py --model fomc-roberta --truncation full --hf-token hf_YOUR_TOKEN
  python embeddings_pipeline.py --model llama3.1 --truncation full --llama-path /path/to/llama3.1
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zipfile
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

try:
    import ollama as _ollama
except ImportError:
    _ollama = None  # only needed when --model llama3.1 without --llama-path

#---------------------------------------------------------------------------
# Constants (not model-specific)
#---------------------------------------------------------------------------

MAX_LENGTH   = 512   # hard token limit per window
STRIDE       = 128   # token overlap between consecutive windows
# Ollama: rough char budget (≈4 chars/token × 512 tokens)
_OLLAMA_CHARS_512   = 2000
_OLLAMA_CHARS_CHUNK = 2000
_OLLAMA_CHARS_STEP  = 1500   # overlap = 500 chars ≈ 128 tokens
CHUNK_BATCH  = 16    # windows per forward pass — lower if GPU OOM
N_PCA        = 20    # PCA components kept for TFT input
RANDOM_STATE = 42

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

# HuggingFace model IDs for the hosted models
_HF_MODEL_IDS = {
    "finbert":      "ProsusAI/finbert",
    "fomc-roberta": "gtfintechlab/FOMC-RoBERTa",
}

project_root = Path(__file__).resolve().parents[1]
data_dir     = project_root / "data"


#---------------------------------------------------------------------------
# CLI
#---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate speech embeddings via FinBERT, FOMC-RoBERTa or LLaMA 3.1")
    p.add_argument(
        "--model",
        choices=["finbert", "fomc-roberta", "llama3.1"],
        default="finbert",
        help="Embedding model to use (default: finbert)",
    )
    p.add_argument(
        "--truncation",
        choices=["512", "full"],
        default="full",
        help="'512' = first 512 tokens only; 'full' = chunk-and-average (default: full)",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace access token (required for fomc-roberta gated repo)",
    )
    p.add_argument(
        "--llama-path",
        default=None,
        help="Local filesystem path to LLaMA 3.1 model weights (required when --model llama3.1)",
    )
    p.add_argument(
        "--ollama-model",
        default="llama3.1:latest",
        help="Ollama model name to use when --model llama3.1 and no --llama-path is given (default: llama3.1:latest)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only the first N speeches (for testing). Omit to process all.",
    )
    p.add_argument(
        "--cutoff",
        default=None,
        help="ISO date (YYYY-MM-DD). PCA is fit only on speeches before this date Should be end of training from TFT. "
    )
    return p.parse_args()


#---------------------------------------------------------------------------
# Model loading
#---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    hf_token: str | None = None,
    llama_path: str | None = None,
) -> tuple:
    """Return (tokenizer, model, is_decoder_only)."""

    if model_name == "llama3.1":
        if not llama_path:
            raise ValueError("--llama-path is required when --model llama3.1")
        print(f"Loading LLaMA 3.1 from local path: {llama_path}")
        tok = AutoTokenizer.from_pretrained(llama_path)
        # LLaMA has no pad token — use EOS
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModel.from_pretrained(
            llama_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        return tok, mdl, True  # decoder-only

    hf_id = _HF_MODEL_IDS[model_name]
    print(f"Loading {model_name} ({hf_id}) …")
    kwargs = {"token": hf_token} if hf_token else {}
    tok = AutoTokenizer.from_pretrained(hf_id, **kwargs)
    mdl = AutoModel.from_pretrained(hf_id, **kwargs)
    return tok, mdl, False  # encoder


#---------------------------------------------------------------------------
# Speech loading
#---------------------------------------------------------------------------

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


#---------------------------------------------------------------------------
# Output path helpers
#---------------------------------------------------------------------------

def _output_dir(model_name: str) -> Path:
    d = data_dir / "embeddings" / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stem(pooling: str, truncation: str, model_name: str) -> str:
    """e.g. 'cls_512_finbert' or 'mean_full_fomc-roberta'"""
    return f"{pooling}_{truncation}_{model_name}"


#---------------------------------------------------------------------------
# Version 1: first-512-tokens embedding (encoder models only for CLS;
#            mean pooling available for all models including LLaMA)
#---------------------------------------------------------------------------

def embed_512(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: torch.device,
    is_decoder_only: bool,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Truncate each speech to MAX_LENGTH tokens and run batched inference.
    Returns (cls_embeddings, mean_embeddings).
    cls_embeddings is None for decoder-only models (no [CLS] token).
    """
    inputs = tokenizer(
        list(df["_text"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    dataset    = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=32)

    model.to(device)
    model.eval()

    all_cls:  list[torch.Tensor] = []
    all_mean: list[torch.Tensor] = []

    start = time.time()
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs     = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (B, T, H)

            if not is_decoder_only:
                all_cls.append(last_hidden[:, 0, :].cpu())  # [CLS] token

            # mean pooling over non-padding tokens
            mask_f      = attention_mask.unsqueeze(-1).float()
            mean_emb    = (last_hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
            all_mean.append(mean_emb.cpu())

    print(f"  Inference (512-truncation) on {device}: {time.time() - start:.1f}s")

    cls_out  = torch.cat(all_cls,  dim=0).numpy() if all_cls  else None
    mean_out = torch.cat(all_mean, dim=0).numpy()
    return cls_out, mean_out


#---------------------------------------------------------------------------
# Version 2: chunk-and-average over full speech
#---------------------------------------------------------------------------

def _build_windows(
    token_ids: torch.Tensor,
    tokenizer,
    is_decoder_only: bool,
) -> list[dict[str, torch.Tensor]]:
    """Split token_ids into overlapping MAX_LENGTH windows."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    if is_decoder_only:
        # decoder-only: no [CLS]/[SEP] wrapper; plain sliding windows
        step  = MAX_LENGTH - STRIDE
        total = len(token_ids)
        windows, start = [], 0
        while True:
            end  = min(start + MAX_LENGTH, total)
            ids  = token_ids[start:end]
            pad_len = MAX_LENGTH - len(ids)
            mask = torch.ones(len(ids), dtype=torch.long)
            if pad_len > 0:
                ids  = torch.cat([ids,  torch.full((pad_len,), pad_id,  dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len,            dtype=torch.long)])
            windows.append({"input_ids": ids.unsqueeze(0), "attention_mask": mask.unsqueeze(0)})
            if end == total:
                break
            start += step
        return windows
    else:
        # encoder: wrap each window with [CLS] … [SEP]
        cls_id    = tokenizer.cls_token_id
        sep_id    = tokenizer.sep_token_id
        body_size = MAX_LENGTH - 2
        step      = body_size - STRIDE
        total     = len(token_ids)
        windows, start = [], 0
        while True:
            end  = min(start + body_size, total)
            body = token_ids[start:end]
            ids  = torch.cat([
                torch.tensor([cls_id], dtype=torch.long),
                body,
                torch.tensor([sep_id], dtype=torch.long),
            ])
            pad_len = MAX_LENGTH - len(ids)
            mask = torch.ones(len(ids), dtype=torch.long)
            if pad_len > 0:
                ids  = torch.cat([ids,  torch.full((pad_len,), pad_id, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len,           dtype=torch.long)])
            windows.append({"input_ids": ids.unsqueeze(0), "attention_mask": mask.unsqueeze(0)})
            if end == total:
                break
            start += step
        return windows


@torch.no_grad()
def _embed_windows_batch(
    windows: list[dict[str, torch.Tensor]],
    model,
    device: torch.device,
    is_decoder_only: bool,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Run windows through model in batches.
    Returns (cls_avg, mean_avg) each shape (hidden_size,).
    cls_avg is None for decoder-only.
    """
    cls_embs:  list[np.ndarray] = []
    mean_embs: list[np.ndarray] = []

    for i in range(0, len(windows), CHUNK_BATCH):
        batch  = windows[i : i + CHUNK_BATCH]
        ids    = torch.cat([w["input_ids"]      for w in batch], dim=0).to(device)
        mask   = torch.cat([w["attention_mask"] for w in batch], dim=0).to(device)
        out    = model(input_ids=ids, attention_mask=mask)
        hidden = out.last_hidden_state  # (B, T, H)

        if not is_decoder_only:
            cls_embs.append(hidden[:, 0, :].cpu().float().numpy())

        mask_f = mask.unsqueeze(-1).float()
        mean   = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
        mean_embs.append(mean.cpu().float().numpy())

    mean_out = np.vstack(mean_embs).mean(axis=0)
    cls_out  = np.vstack(cls_embs).mean(axis=0) if cls_embs else None
    return cls_out, mean_out


def embed_full(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: torch.device,
    is_decoder_only: bool,
    hidden_size: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Chunk-and-average embedding for each speech.
    Returns (cls_matrix, mean_matrix), each shape (N, hidden_size).
    cls_matrix is None for decoder-only models.
    """
    n = len(df)
    print(f"  Embedding {n:,} speeches via chunk-and-average …")
    all_cls:  list[np.ndarray] = []
    all_mean: list[np.ndarray] = []

    start = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            print(f"    {i:>5} / {n}")
        text = row["_text"]
        if not text.strip():
            if not is_decoder_only:
                all_cls.append(np.zeros(hidden_size, dtype=np.float32))
            all_mean.append(np.zeros(hidden_size, dtype=np.float32))
            continue

        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        )["input_ids"][0]

        if len(token_ids) == 0:
            if not is_decoder_only:
                all_cls.append(np.zeros(hidden_size, dtype=np.float32))
            all_mean.append(np.zeros(hidden_size, dtype=np.float32))
            continue

        windows = _build_windows(token_ids, tokenizer, is_decoder_only)
        cls_vec, mean_vec = _embed_windows_batch(windows, model, device, is_decoder_only)
        if cls_vec is not None:
            all_cls.append(cls_vec)
        all_mean.append(mean_vec)

    print(f"  Inference (full-speech) on {device}: {time.time() - start:.1f}s")
    cls_out  = np.vstack(all_cls)  if all_cls  else None
    mean_out = np.vstack(all_mean)
    return cls_out, mean_out


#---------------------------------------------------------------------------
# Ollama-based embeddings (used when --model llama3.1 and no --llama-path)
#---------------------------------------------------------------------------

def _ollama_embed_text(ollama_model: str, text: str) -> np.ndarray:
    """Call Ollama embeddings endpoint for a single text string."""
    try:
        # ollama >= 0.4: ollama.embed(model, input) -> EmbeddingsResponse
        response = _ollama.embed(model=ollama_model, input=text)
        return np.array(response.embeddings[0], dtype=np.float32)
    except AttributeError:
        # ollama < 0.4: ollama.embeddings(model, prompt) -> dict
        response = _ollama.embeddings(model=ollama_model, prompt=text)
        return np.array(response["embedding"], dtype=np.float32)


def embed_ollama_512(
    df: pd.DataFrame,
    ollama_model: str,
) -> tuple[None, np.ndarray]:
    """
    Truncate each speech to ~512 tokens (via char budget) and embed via Ollama.
    Returns (None, mean_matrix) — no CLS token for decoder-only models.
    """
    texts = [t[:_OLLAMA_CHARS_512] for t in df["_text"]]
    start = time.time()
    vecs = []
    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"    {i:>5} / {len(texts)}")
        vecs.append(_ollama_embed_text(ollama_model, text or " "))
    print(f"  Inference (512-truncation) via Ollama: {time.time() - start:.1f}s")
    return None, np.vstack(vecs)


def embed_ollama_full(
    df: pd.DataFrame,
    ollama_model: str,
) -> tuple[None, np.ndarray]:
    """
    Chunk-and-average over full speech text using Ollama embeddings.
    Returns (None, mean_matrix) — no CLS token for decoder-only models.
    """
    n = len(df)
    print(f"  Embedding {n:,} speeches via Ollama chunk-and-average …")
    start = time.time()
    all_mean = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 50 == 0:
            print(f"    {i:>5} / {n}")
        text = row["_text"]
        if not text.strip():
            all_mean.append(None)
            continue

        # build character-based sliding windows
        chunks, pos = [], 0
        while pos < len(text):
            chunks.append(text[pos : pos + _OLLAMA_CHARS_CHUNK])
            if pos + _OLLAMA_CHARS_CHUNK >= len(text):
                break
            pos += _OLLAMA_CHARS_STEP

        chunk_vecs = [_ollama_embed_text(ollama_model, c) for c in chunks]
        all_mean.append(np.mean(chunk_vecs, axis=0).astype(np.float32))

    # fill any empty speeches with zero vector of inferred size
    hidden_size = next(v.shape[0] for v in all_mean if v is not None)
    mean_out = np.vstack([
        v if v is not None else np.zeros(hidden_size, dtype=np.float32)
        for v in all_mean
    ])
    print(f"  Inference (full-speech) via Ollama: {time.time() - start:.1f}s")
    return None, mean_out


#---------------------------------------------------------------------------
# PCA + save helpers
#---------------------------------------------------------------------------

def _fit_and_save_pca(
    embeddings: np.ndarray,
    dates: pd.Series, # added here
    cutoff_date: pd.Timestamp, # added here
    pooling: str,
    truncation: str,
    model_name: str,
    out_dir: Path,
) -> np.ndarray:
    
    # fit only on speeches before the cutoff
    train_mask = dates < cutoff_date
    print(f"PCA fit on {train_mask.sum()} / {len(dates)} speeches (before {cutoff_date.date()})")
    
    pca     = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
    # fit only on training data
    pca.fit(embeddings[train_mask])
    # then transform all
    reduced = pca.transform(embeddings)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"    PCA ({pooling}): ", end="")
    for k in [5, 10, N_PCA]:
        if k <= N_PCA:
            print(f"{k}→{cumvar[k-1]:.0%}  ", end="")
    print()

    pkl_path = out_dir / f"pca_model_{_stem(pooling, truncation, model_name)}.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(pca, fh)
    print(f"    Saved: {pkl_path.name}")
    return reduced


def save_outputs(
    df: pd.DataFrame,
    available_meta: list[str],
    cls_matrix: np.ndarray | None,
    mean_matrix: np.ndarray,
    truncation: str,
    model_name: str,
    out_dir: Path,
    cutoff_date: pd.Timestamp, # added this here
) -> None:
    # extract date
    dates = pd.to_datetime(df["Date"])
    
    meta_df = df[available_meta].copy().reset_index(drop=True)
    if "Date" in meta_df.columns:
        meta_df["Date"] = pd.to_datetime(meta_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    poolings = []
    if cls_matrix is not None:
        poolings.append(("cls", cls_matrix))
    poolings.append(("mean", mean_matrix))

    for pooling, matrix in poolings:
        stem = _stem(pooling, truncation, model_name)

        # full (high-dim) embeddings
        emb_cols = [f"emb_{i}" for i in range(matrix.shape[1])]
        full_df  = pd.concat([meta_df, pd.DataFrame(matrix, columns=emb_cols)], axis=1)
        full_path = out_dir / f"embeddings_full_{stem}.csv.zip"
        full_df.to_csv(full_path, index=False, compression="zip")
        print(f"    Saved: {full_path.name}  shape={full_df.shape}")

        # PCA-reduced embeddings (TFT input)
        reduced  = _fit_and_save_pca(matrix, dates, cutoff_date, pooling, truncation, model_name, out_dir)
        pca_cols = [f"pca_{i}" for i in range(N_PCA)]
        pca_df   = pd.concat([meta_df, pd.DataFrame(reduced, columns=pca_cols)], axis=1)
        pca_path = out_dir / f"embeddings_pca_{stem}.csv"
        pca_df.to_csv(pca_path, index=False)
        print(f"    Saved: {pca_path.name}  shape={pca_df.shape}")


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    ollama_mode = args.model == "llama3.1" and not args.llama_path

    if ollama_mode:
        if _ollama is None:
            raise ImportError("Install the ollama package: pip install ollama")
        print(f"Loading LLaMA 3.1 via Ollama ({args.ollama_model}) …")
        # Infer hidden size from a test embedding
        test_vec = _ollama_embed_text(args.ollama_model, "test")
        hidden_size = test_vec.shape[0]
        print(f"  Ollama model ready  |  Hidden size: {hidden_size}")
    else:
        # load model
        tokenizer, model, is_decoder_only = load_model_and_tokenizer(
            args.model, args.hf_token, args.llama_path
        )
        device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_size = model.config.hidden_size
        model.to(device)
        model.eval()
        print(f"  Device: {device}  |  Hidden size: {hidden_size}")

    # load speeches
    df = load_speeches()
    if args.sample:
        df = df.head(args.sample)
        print(f"  Using sample of {args.sample} speeches.")
    df = df.sort_values("Date").reset_index(drop=True)
    available_meta = [c for c in META_COLS if c in df.columns]
    
    # add cutoff date with default = max speech date
    cutoff_date = pd.Timestamp(args.cutoff) if args.cutoff else df["Date"].max()

    out_dir = _output_dir(args.model)
    print(f"  Output dir: {out_dir}")

    if args.truncation == "512":
        print("\n[512-token truncation]")
        if ollama_mode:
            cls_matrix, mean_matrix = embed_ollama_512(df, args.ollama_model)
        else:
            cls_matrix, mean_matrix = embed_512(df, tokenizer, model, device, is_decoder_only)
    else:
        print("\n[Full-speech chunk-and-average]")
        if ollama_mode:
            cls_matrix, mean_matrix = embed_ollama_full(df, args.ollama_model)
        else:
            cls_matrix, mean_matrix = embed_full(df, tokenizer, model, device, is_decoder_only, hidden_size)

    print("\nSaving outputs …")
    save_outputs(df, available_meta, cls_matrix, mean_matrix, args.truncation, args.model, out_dir, cutoff_date=cutoff_date)

    print("\nDone.")


if __name__ == "__main__":
    main()