"""
Learned attention aggregator for FOMC speech embeddings.

Trains a small query-key-value attention layer to weight speeches
before aggregating into a single feature vector per month.

The query is a learned global vector (no macro state needed),
keys and values are the per-speech PCA vectors.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

class SpeechAttentionAggregator(nn.Module):
    """
    Single-head attention over a set of speech PCA vectors.
    
    Given a set of speech vectors X (n_speeches x n_pca),
    computes attention weights and returns a weighted sum.
    
    The query is a learned global vector — no macro state needed.
    This keeps it simple and avoids leakage.
    """
    def __init__(self, n_pca: int = 5):
        super().__init__()
        self.n_pca   = n_pca
        self.query   = nn.Parameter(torch.randn(n_pca))  # learned global query
        self.key_fc   = nn.Linear(n_pca, n_pca, bias=False)
        self.value_fc = nn.Linear(n_pca, n_pca, bias=False)
        self.scale    = n_pca ** 0.5

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : (n_speeches, n_pca)
        returns : (n_pca,) weighted sum
        """
        # K,V matrix come from encoder, Q comes from decoder
        K = self.key_fc(X)    # (n_speeches, n_pca)
        V = self.value_fc(X)  # (n_speeches, n_pca)
        
        # attention scores: query dot each key
        scores = (K @ self.query) / self.scale  # (n_speeches,)
        weights = torch.softmax(scores, dim=0)   # (n_speeches,)
        
        # weighted sum of values
        out = (weights.unsqueeze(1) * V).sum(dim=0)  # (n_pca,)
        return out, weights


def train_attention_aggregator(
    speeches_df: pd.DataFrame, # notice how there are no macro speeches here
    pca_cols: list[str],
    train_end: pd.Timestamp,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> SpeechAttentionAggregator:
    """
    Train the attention aggregator on training speeches only.
    
    Objective: reconstruct the mean embedding of a random window
    of speeches from a subset — forces the model to learn which
    speeches are most representative.
    
    Parameters
    ----------
    speeches_df : DataFrame with Date and pca_* columns
    pca_cols    : list of pca column names
    train_end   : only use speeches up to this date (no leakage)
    n_epochs    : training epochs
    lr          : learning rate
    device      : 'cpu' or 'cuda'
    """
    # training speeches only
    train_speeches = speeches_df[speeches_df["Date"] <= train_end].copy()
    X = torch.tensor(
        train_speeches[pca_cols].values.astype(np.float32)
    ).to(device)
    
    n_pca  = len(pca_cols)
    model  = SpeechAttentionAggregator(n_pca=n_pca).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    
    # target: mean of all training speeches
    # the model learns to approximate this mean via attention
    target = X.mean(dim=0)  # (n_pca,)
    
    print(f"[SpeechAttention] Training on {len(X)} speeches for {n_epochs} epochs...")
    # here backprop step
    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad() # set grad to 0 (redo this for each epoch)
        out, _ = model(X)
        loss = nn.MSELoss()(out, target)
        loss.backward()
        opt.step() # performs opt step
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  loss={loss.item():.6f}")
    
    model.eval()
    print("[SpeechAttention] Training complete.")
    return model


def aggregate_with_attention(
    model: SpeechAttentionAggregator,
    sub: pd.DataFrame,
    pca_cols: list[str],
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply trained attention model to a window of speeches.
    
    Returns
    -------
    weighted_mean : (n_pca,) aggregated vector
    weights       : (n_speeches,) attention weights
    """
    X = torch.tensor(sub[pca_cols].values.astype(np.float32)).to(device)
    with torch.no_grad():
        out, weights = model(X)
    return out.cpu().numpy(), weights.cpu().numpy()



class ContextAwareSpeechAttentionAggregator(nn.Module):
    """
    Context-aware attention over speech PCA vectors.
    
    The query is derived from the current macro state rather than
    being a fixed learned vector. This allows the model to ask:
    "given the current economic conditions, which speeches matter most?"
    """
    def __init__(self, n_pca: int = 5, n_macro: int = 5):
        super().__init__()
        self.n_pca   = n_pca
        self.query_net = nn.Sequential(
            nn.Linear(n_macro, n_pca),
            nn.Tanh()
        )
        self.key_fc   = nn.Linear(n_pca, n_pca, bias=False)
        self.value_fc = nn.Linear(n_pca, n_pca, bias=False)
        self.scale    = n_pca ** 0.5

    def forward(self, X: torch.Tensor, macro_state: torch.Tensor) -> tuple:
        """
        X           : (n_speeches, n_pca)
        macro_state : (n_macro,) current macro conditions
        returns     : (n_pca,) weighted sum, (n_speeches,) weights
        """
        query  = self.query_net(macro_state)   # (n_pca,)
        K = self.key_fc(X)                     # (n_speeches, n_pca)
        V = self.value_fc(X)                   # (n_speeches, n_pca)
        scores  = (K @ query) / self.scale     # (n_speeches,)
        weights = torch.softmax(scores, dim=0) # (n_speeches,)
        out     = (weights.unsqueeze(1) * V).sum(dim=0)  # (n_pca,)
        return out, weights


def train_context_aware_attention(
    speeches_df: pd.DataFrame,
    macro_df: pd.DataFrame,          # monthly macro data aligned to speeches
    pca_cols: list[str],
    macro_cols: list[str],
    train_end: pd.Timestamp,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> ContextAwareSpeechAttentionAggregator:
    """
    Train context-aware attention on training data only.
    
    Objective: for each month in training, predict the month's
    macro values from the attention-weighted speech vector.
    This forces the model to learn which speeches are predictive
    of current macro conditions.
    """
    # align speeches to monthly macro state
    train_macro = macro_df[macro_df["date"] <= train_end].copy()
    train_speeches = speeches_df[speeches_df["Date"] <= train_end].copy()
    
    n_pca   = len(pca_cols)
    n_macro = len(macro_cols)
    model   = ContextAwareSpeechAttentionAggregator(n_pca=n_pca, n_macro=n_macro).to(device)
    
    # add a small prediction head: pca → macro
    # this gives extra signal for the backprop, because it incentivises the use of the macro vars
    # the pred head is created from macro vars, the goal is for the model to replicate the info from macro vars
    # this is used to calculate loss
    pred_head = nn.Linear(n_pca, n_macro).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(pred_head.parameters()), lr=lr)
    
    print(f"[ContextAttention] Training on {len(train_macro)} months for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        model.train()
        pred_head.train()
        total_loss = 0.0
        
        for _, month_row in train_macro.iterrows():
            month_date  = month_row["date"]
            macro_state = torch.tensor(
                month_row[macro_cols].values.astype(np.float32)
            ).to(device)
            
            # speeches in the 12-month window before this month
            window_start = month_date - pd.DateOffset(months=12)
            mask = (
                (train_speeches["Date"] >= window_start)
                & (train_speeches["Date"] < month_date)
            )
            sub = train_speeches.loc[mask, pca_cols]
            if len(sub) == 0:
                continue
            
            X = torch.tensor(sub.values.astype(np.float32)).to(device)
            aggregated, _ = model(X, macro_state)
            predicted_macro = pred_head(aggregated)
            
            # target: contruct current month (reconstruction objective)
            loss = nn.MSELoss()(predicted_macro, macro_state)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  loss={total_loss/len(train_macro):.6f}")
    
    model.eval()
    print("[ContextAttention] Training complete.")
    return model


def aggregate_with_context_attention(
    model: ContextAwareSpeechAttentionAggregator,
    sub: pd.DataFrame,
    pca_cols: list[str],
    macro_state: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply context-aware attention to a window of speeches."""
    X = torch.tensor(sub[pca_cols].values.astype(np.float32)).to(device)
    m = torch.tensor(macro_state.astype(np.float32)).to(device)
    with torch.no_grad():
        out, weights = model(X, m)
    return out.cpu().numpy(), weights.cpu().numpy()