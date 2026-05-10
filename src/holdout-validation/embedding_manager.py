"""
Leakage-free PCA embedding manager for FOMC-RoBERTa speech embeddings.

The raw (pre-PCA) full-speech embeddings are loaded once.  For each
cross-validation fold, PCA is re-fitted on training-period speeches only,
then applied to all speeches.  This eliminates the look-ahead bias that
would arise from fitting PCA on the full dataset before splitting.

Typical usage (called from DataFrameBuilder.add_leakage_free_embeddings):

    emb = EmbeddingManager(root_path).load()
    splits = dfb.add_leakage_free_embeddings(splits, emb)

The returned splits have pca_* aggregated speech features in every
train/test DataFrame, computed without any look-ahead.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

N_PCA        = 20
RANDOM_STATE = 42

# meta columns present in the full-embedding CSV
_META_COLS = [
    "Date", "Authorname", "CentralBank", "Role",
    "position_in_fed", "district", "female", "minority", "federal_reserve",
]

# path to the pre-PCA full embeddings, relative to project root
_FULL_EMB_SUBPATH = (
    "data/embeddings/fomc-roberta/embeddings_full_mean_full_fomc-roberta.csv.zip"
)


class EmbeddingManager:
    """
    Leakage-free FOMC-RoBERTa embeddings via per-fold PCA.

    Methods
    -------
    load()
        Load the pre-PCA full embeddings from disk.  Returns self.
    generate_split(splits)
        Record train/test date boundaries from DataFrameBuilder splits.
    recalculate_pca(fold)
        Fit PCA on train speeches only; transform all. Returns speeches DataFrame.
    shuffle_embeddings(fold)
        Like recalculate_pca() but permutes vectors within each split partition.
        Used to test whether chronological ordering adds predictive value.
    get_embedding_data(fold, shuffled)
        Main entry point: returns a speeches DataFrame with Date, meta, and
        pca_0 … pca_{n_pca-1} columns for use in DataFrameBuilder.
    """

    def __init__(self, path: str, n_pca: int = N_PCA):
        self.path   = Path(path)
        self.n_pca  = n_pca
        self.full_df:  pd.DataFrame | None = None
        self.emb_cols: list[str]           = []
        self._splits:  list[dict] | None   = None  # set by generate_split()

    # ------------------------------------------------------------------
    def load(self) -> "EmbeddingManager":
        """Load the full (pre-PCA) mean embeddings for FOMC-RoBERTa."""
        p = self.path / _FULL_EMB_SUBPATH
        if not p.exists():
            raise FileNotFoundError(
                f"Full embeddings not found: {p}\n"
                "Run: python src/embeddings_pipeline.py "
                "--model fomc-roberta --truncation full"
            )
        df = pd.read_csv(p, compression="zip", parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        self.full_df  = df
        self.emb_cols = sorted(c for c in df.columns if c.startswith("emb_"))
        print(
            f"[EmbeddingManager] Loaded {len(df):,} speeches, {len(self.emb_cols)} dims "
            f"({df['Date'].min().date()} → {df['Date'].max().date()})"
        )
        return self

    # ------------------------------------------------------------------
    def generate_split(self, splits: list[dict]) -> "EmbeddingManager":
        """
        Register PCA cutoff dates derived from DataFrameBuilder splits.

        PCA will be fitted only on speeches with Date <= each fold's train_end.
        Must be called before recalculate_pca() / get_embedding_data().

        Parameters
        ----------
        splits : list of dicts — as returned by DataFrameBuilder.generate_split(),
                 each with keys 'fold', 'train' (DataFrame), 'test' (DataFrame).
        """
        assert self.full_df is not None, "Call load() before generate_split()."
        self._splits = []
        for s in splits:
            train_end = s["train"]["date"].max()
            test_end  = s["test"]["date"].max()
            n_train   = int((self.full_df["Date"] <= train_end).sum())
            n_test    = int(
                ((self.full_df["Date"] > train_end)
                 & (self.full_df["Date"] <= test_end)).sum()
            )
            self._splits.append({
                "fold":      s["fold"],
                "train_end": train_end,
                "test_end":  test_end,
            })
            print(
                f"[EmbeddingManager] Fold {s['fold']}: PCA cutoff {train_end.date()} "
                f"— {n_train} train speeches, {n_test} test-period speeches"
            )
        return self

    # ------------------------------------------------------------------
    def recalculate_pca(self, fold: int = 0) -> pd.DataFrame:
        """
        Fit PCA on training speeches only; project all speeches into that space.

        PCA is fitted only on speeches with Date <= train_end for this fold,
        then the fitted model is applied to ALL speeches (including post-cutoff).
        This gives each speech a coordinate in the training PCA space with
        no information from future speeches leaking into the basis.

        Parameters
        ----------
        fold : 0-based index into self._splits.

        Returns
        -------
        pd.DataFrame with available meta columns + pca_0 … pca_{n_pca-1},
        one row per speech.
        """
        assert self._splits is not None, "Call generate_split() before recalculate_pca()."
        split  = self._splits[fold]
        cutoff = split["train_end"]

        train_mask = self.full_df["Date"] <= cutoff
        X_all      = self.full_df[self.emb_cols].values.astype(np.float32)

        pca = PCA(n_components=self.n_pca, random_state=RANDOM_STATE)
        pca.fit(X_all[train_mask])   # fit on training speeches only — no leakage
        reduced = pca.transform(X_all)  # project all speeches into training PCA space

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        print(
            f"[EmbeddingManager] PCA fold {fold}: "
            f"5→{cumvar[4]:.0%}  10→{cumvar[9]:.0%}  {self.n_pca}→{cumvar[-1]:.0%} "
            f"(fitted on {int(train_mask.sum())}/{len(self.full_df)} speeches)"
        )

        available_meta = [c for c in _META_COLS if c in self.full_df.columns]
        pca_cols = [f"pca_{i}" for i in range(self.n_pca)]
        out = self.full_df[available_meta].copy().reset_index(drop=True)
        out[pca_cols] = reduced
        return out

    # ------------------------------------------------------------------
    def shuffle_embeddings(self, fold: int = 0) -> pd.DataFrame:
        """
        Return PCA speeches with vectors permuted within each split partition.

        Train-period speech vectors are randomly re-assigned among train
        speech dates; test-period vectors are re-assigned among test dates.
        The PCA basis is still fitted on training data only (no leakage here).

        This ablation tests whether chronological ordering of embeddings
        contributes predictive signal beyond average embedding content.
        Shuffling within each partition preserves the split boundary but
        destroys within-split temporal ordering.

        Parameters
        ----------
        fold : 0-based index into self._splits.

        Returns
        -------
        pd.DataFrame — same layout as recalculate_pca() with permuted pca_* values.
        """
        assert self._splits is not None, "Call generate_split() before shuffle_embeddings()."
        split    = self._splits[fold]
        cutoff   = split["train_end"]
        test_end = split["test_end"]

        df       = self.recalculate_pca(fold=fold)
        pca_cols = [c for c in df.columns if c.startswith("pca_")]
        rng      = np.random.default_rng(RANDOM_STATE)

        # shuffle within train-period speeches and test-period speeches separately
        for mask in [
            df["Date"] <= cutoff,
            (df["Date"] > cutoff) & (df["Date"] <= test_end),
        ]:
            idx = np.where(mask)[0]
            if len(idx) > 1:
                perm = rng.permutation(len(idx))
                df.loc[mask, pca_cols] = df.loc[mask, pca_cols].values[perm]

        return df

    # ------------------------------------------------------------------
    def get_embedding_data(
        self,
        fold:     int  = 0,
        shuffled: bool = False,
    ) -> pd.DataFrame:
        """
        Return leakage-free speech embeddings for the given fold.

        Called by DataFrameBuilder.add_leakage_free_embeddings().
        PCA is fitted only on training speeches so there is no look-ahead bias.

        Parameters
        ----------
        fold     : 0-based fold index matching self._splits ordering.
        shuffled : if True, permute embedding vectors within each split partition
                   (ablation — see shuffle_embeddings).

        Returns
        -------
        pd.DataFrame with Date, meta columns, and pca_0 … pca_{n_pca-1}.
        """
        return self.shuffle_embeddings(fold=fold) if shuffled else self.recalculate_pca(fold=fold)
