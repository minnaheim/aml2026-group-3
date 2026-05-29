#!/usr/bin/env python3
"""
Final holdout evaluation: train on ALL pre-holdout data, evaluate on the true holdout window.

Run AFTER CV and HP tuning are complete. The holdout set (~Dec 2022 – Dec 2023) has
never been touched during training or tuning.

THIS SHOULD ONLY EVER BE RUN WITH TUNED!!!

Usage:
    python src/holdout-validation/h_final_eval.py --tuned --horizon 12
    python src/holdout-validation/h_final_eval.py --tuned --embedding auto --run-name final_emb --horizon 12
    python src/holdout-validation/h_final_eval.py --tuned --no-baselines --device mps
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import os

sys.path.insert(0, str(Path(__file__).parent))
from b_data_frame_builder import DataFrameBuilder
from c_benchmark_runner   import ARRunner, ARIMARunner
from d_tft_runner         import TFTRunner
from a_embedding_manager  import EmbeddingManager
# reuse metric / save / plot helpers — no duplication
from e_main import load_tuned_hparams, save_results, plot_results

MAIN_TARGETS = ["CPI", "UNRATE", "GDP"]


def main():
    parser = argparse.ArgumentParser(
        description="Final holdout evaluation (train on all pre-holdout data)"
    )
    parser.add_argument(
        "--no-baselines", action="store_true", default=False,
        help="Skip AR and ARIMA; only run TFT",
    )
    parser.add_argument(
        "--targets", nargs="+", default=MAIN_TARGETS,
        choices=MAIN_TARGETS, metavar="TARGET",
        help=f"Targets to forecast (default: all main targets)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "mps", "cuda"],
        help="Compute device for TFT (default: cpu)",
    )
    parser.add_argument(
        "--embedding", nargs="?", const="auto", default=None,
        choices=["auto", "fomc-roberta", "finbert", "finbert-kafka", "fomc-roberta-kafka", 
                 "llama3.1", "fomc-roberta-512", "finbert-512"],
        help="Speech embedding (omit = macro-only; flag alone = auto-select best from tuning)",
    )
    parser.add_argument(
        "--tuning-embedding", default=None,
        choices=["fomc-roberta", "finbert"],
        help="Which tuning results to load params from (default: same as --embedding). "
            "Use this for kafka and 512 token ablation: --embedding fomc-roberta-kafka --tuning-embedding fomc-roberta",
    )
    parser.add_argument(
        "--aggregation", default="mean", choices=["mean", "decay", "attention", "context_attention"],
        help="Speech aggregation strategy (default: mean)",
    )
    parser.add_argument(
        "--horizon", type=int, default=12, choices=[3, 6, 12],
        help="Forecast horizon in months (default: 12)",
    )
    parser.add_argument(
        "--reduction", default="pca", choices=["pca", "fa", "none"],
        help="Dimensionality reduction strategy (default: pca)",
    )
    parser.add_argument(
        "--n-pca", type=int, default=5,
        help="Number of PCA components (default: 5)",
    )
    parser.add_argument(
        "--run-name", default="final_holdout",
        help="Name for output subfolder (default: final_holdout)",
    )
    parser.add_argument(
        "--tuned", action="store_true", default=False,
        help="Load best hyperparams from out/tuning/ (recommended for final eval)",
    )
    # ── TFT hyperparams (CLI overrides; ignored when --tuned is set) ──────────
    parser.add_argument("--encoder-length",         type=int,   default=24)
    parser.add_argument("--speech-window",          type=int,   default=12)
    parser.add_argument("--lstm-layers",            type=int,   default=1)
    parser.add_argument("--hidden-size",            type=int,   default=8)
    parser.add_argument("--hidden-continuous-size", type=int,   default=8)
    parser.add_argument("--dropout",                type=float, default=0.2)
    parser.add_argument("--lr",                     type=float, default=0.03)
    parser.add_argument(
        "--normalizer", default="encoder_none", choices=["encoder_none", "group"],
    )
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for TFT dataloaders (default: 16)")

    args = parser.parse_args()

    if args.device != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    root    = Path(__file__).parent.parent.parent
    out_dir = root / "out" / "holdout"

    emb_tag = args.embedding or "none"
    print(f"=== FINAL HOLDOUT EVALUATION ===")
    print(f"Targets   : {args.targets}")
    print(f"Device    : {args.device}")
    print(f"Embedding : {emb_tag}")
    print(f"Horizon   : {args.horizon}")
    print(f"Tuned     : {args.tuned}")
    print(f"Run name  : {args.run_name}\n")

    # ── 1. build base data ────────────────────────────────────────────────────
    base_hparams = {
        "max_prediction_length":  args.horizon,
        "max_encoder_length":     args.encoder_length,
        "speech_window_months":   args.speech_window,
        "lstm_layers":            args.lstm_layers,
        "hidden_size":            args.hidden_size,
        "hidden_continuous_size": args.hidden_continuous_size,
        "dropout":                args.dropout,
        "learning_rate":          args.lr,
        "normalizer":             args.normalizer,
        "batch_size":             args.batch_size,
    }

    dfb = DataFrameBuilder(str(root), aggregation=args.aggregation, speech_window=args.speech_window, device = args.device)
    dfb.load_fomc_dissent()
    df  = dfb.process_data()
    # unlike in main, splits egal, only holdout here
    _, holdout = dfb.generate_split(df)

    # train = everything before the holdout window
    df_cv = df.iloc[:-dfb.FINAL_HOLDOUT].reset_index(drop=True)
    holdout_splits = [{"fold": "holdout", "train": df_cv, "test": holdout}]

    print(
        f"Train  : [{df_cv['date'].min().date()} – {df_cv['date'].max().date()}] ({len(df_cv)} rows)"
    )
    print(
        f"Holdout: [{holdout['date'].min().date()} – {holdout['date'].max().date()}] ({len(holdout)} rows)\n"
    )

    ar_runner    = ARRunner(dfb,    max_prediction_length=args.horizon)
    arima_runner = ARIMARunner(dfb, max_prediction_length=args.horizon)

    results = {"TFT": {"holdout": {}}}
    if not args.no_baselines:
        results["AR"]    = {"holdout": {}}
        results["ARIMA"] = {"holdout": {}}

    # ── 2. run all models for each target ─────────────────────────────────────
    last_aggregation = args.aggregation # as default
    last_embedding = args.embedding or "none" # again, as default
    embedding_label = args.embedding or "none"
    for target in args.targets:
        # here: override the embeddings structure for ablation
        # if we have kafka fomc, we want to use the fomc tuning of course
        tuning_emb = args.tuning_embedding or args.embedding
        if args.tuned:
            arch_params, emb_params, t_embedding = load_tuned_hparams(
                root, target, tuning_emb, args.horizon # adjust here for tuning emb
            )
            t_embedding = args.embedding if args.embedding != "auto" else t_embedding # but here: for inference, we want to use the kafka embeddings or so
            t_aggregation   = emb_params.get("aggregation",          args.aggregation)   if emb_params else args.aggregation
            t_reduction     = emb_params.get("reduction",            args.reduction)     if emb_params else args.reduction
            t_n_pca         = emb_params.get("n_pca",                args.n_pca)         if emb_params else args.n_pca
            t_speech_window = emb_params.get("speech_window_months", args.speech_window) if emb_params else args.speech_window
            target_hparams  = {**base_hparams, **arch_params, "speech_window_months": t_speech_window}
            print(f"\n[tuned] {target}: embedding={t_embedding}, arch={arch_params}")
            if emb_params:
                print(f"[tuned] {target}: emb_params={emb_params}")
            last_aggregation = t_aggregation    
            last_embedding = t_embedding or "none"
        else:
            t_embedding     = args.embedding
            t_aggregation   = args.aggregation
            t_reduction     = args.reduction
            t_n_pca         = args.n_pca
            t_speech_window = args.speech_window
            target_hparams  = base_hparams

        # build embedding-augmented holdout_splits for TFT (AR/ARIMA always use base splits)
        if t_embedding is not None:
            t_dfb = DataFrameBuilder(str(root), aggregation=t_aggregation, speech_window=t_speech_window, device = args.device)
            t_dfb.load_fomc_dissent()
            # reconstruct holdout_splits for this target's dfb (same date boundaries, fresh dfb)
            t_holdout_splits = [{"fold": "holdout", "train": df_cv.copy(), "test": holdout.copy()}]
            emb_mgr = EmbeddingManager(
                str(root), embedding=t_embedding, n_pca=t_n_pca, reduction=t_reduction
            ).load()
            t_holdout_splits = t_dfb.add_leakage_free_embeddings(t_holdout_splits, emb_mgr)
            active_dfb    = t_dfb
            active_splits = t_holdout_splits
        else:
            active_dfb    = dfb
            active_splits = holdout_splits

        tft_runner = TFTRunner(active_dfb, hparams=target_hparams)

        print(f"\n{'─' * 55}")
        print(f" HOLDOUT | Target: {target}")
        print(f"{'─' * 55}")

        if not args.no_baselines:
            print("\n[AR]")
            ar_order, ar_seasonal = ar_runner.run(
                holdout_splits, target=target, fold=0, fold_label="holdout"
            )
            ar_df = ar_runner.predict(holdout_splits, target=target, fold=0)
            results["AR"]["holdout"][target] = ar_df
            print(f"  → {len(ar_df)} predictions (order={ar_order}, seasonal={ar_seasonal})")

            print("\n[ARIMA]")
            arima_order, arima_seasonal = arima_runner.run(
                holdout_splits, target=target, fold=0, fold_label="holdout"
            )
            arima_df = arima_runner.predict(holdout_splits, target=target, fold=0)
            results["ARIMA"]["holdout"][target] = arima_df
            print(f"  → {len(arima_df)} predictions (order={arima_order}, seasonal={arima_seasonal})")

        print("\n[TFT – training]")
        ckpt = tft_runner.run(
            active_splits, target=target, fold=0,
            use_wandb=False, device=args.device,
            run_name=f"{args.run_name}-{target}-holdout",
        )
        print(f"  → best checkpoint: {ckpt}")

        print("\n[TFT – inference]")
        tft_df = tft_runner.predict(
            ckpt, active_splits, target=target, fold=0, device=args.device,
        )
        results["TFT"]["holdout"][target] = tft_df
        print(f"  → {len(tft_df)} predictions")
        
        # also, save variable importance
        print("\n[TFT – variable importance]")
        importance_out = out_dir / args.run_name
        tft_runner.interpret_output(out_dir=importance_out, target=target, embedding_label=embedding_label)

    # ── 3. save and plot ──────────────────────────────────────────────────────
    save_results(results, out_dir,
                 run_name=args.run_name,
                 embedding=embedding_label,
                 aggregation=last_aggregation,
                 ablation_mode=False,
                 horizon=args.horizon)

    plot_results(results, out_dir,
                 run_name=args.run_name,
                 horizon=args.horizon,
                 embedding=embedding_label)


if __name__ == "__main__":
    main()