#!/usr/bin/env python3
"""
Ablation study runner — calls main.py for each experiment configuration
and saves results individually + appends to a master experiments.csv.

Usage:
    python src/holdout-validation/run_ablation.py
    python src/holdout-validation/run_ablation.py --device cuda
    python src/holdout-validation/run_ablation.py --targets CPI  # quick test
"""
import argparse
import subprocess
import sys
from pathlib import Path

# ── experiment configurations ─────────────────────────────────────────────────
EXPERIMENTS = [
    # macro-only baseline
    {
        "run_name":    "macro_only",
        "embedding":   None,
        "aggregation": "mean",
    },
    # fomc-roberta embeddings
    {
        "run_name":    "fomc_roberta_mean",
        "embedding":   "fomc-roberta",
        "aggregation": "mean",
    },
    {
        "run_name":    "fomc_roberta_decay",
        "embedding":   "fomc-roberta",
        "aggregation": "decay",
    },
    {
        "run_name":    "fomc_roberta_context_attention",
        "embedding":   "fomc-roberta",
        "aggregation": "context_attention",
    },
    # finbert embeddings
    {
        "run_name":    "finbert_mean",
        "embedding":   "finbert",
        "aggregation": "mean",
    },
    {
        "run_name":    "finbert_decay",
        "embedding":   "finbert",
        "aggregation": "decay",
    },
    {
        "run_name":    "finbert_context_attention",
        "embedding":   "finbert",
        "aggregation": "context_attention",
    },
    # kafka variants
    {
        "run_name":    "fomc_roberta_kafka_decay",
        "embedding":   "fomc-roberta-kafka",
        "aggregation": "decay",
    },
    {
        "run_name":    "finbert_kafka_decay",
        "embedding":   "finbert-kafka",
        "aggregation": "decay",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument("--device",  default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--targets", nargs="+", default=["CPI", "UNRATE", "GDP"])
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--no-baselines", action="store_true", default=False)
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Only run specific experiments by run_name (default: all)"
    )
    args = parser.parse_args()

    main_script = Path(__file__).parent / "e_main.py"
    
    experiments = EXPERIMENTS
    if args.runs:
        experiments = [e for e in EXPERIMENTS if e["run_name"] in args.runs]
        print(f"Running {len(experiments)} selected experiments: {[e['run_name'] for e in experiments]}")
    else:
        print(f"Running {len(experiments)} experiments...")

    for i, exp in enumerate(experiments):
        run_name = exp["run_name"]
        print(f"\n{'═' * 60}")
        print(f" Experiment {i+1}/{len(experiments)}: {run_name}")
        print(f"{'═' * 60}")

        cmd = [
            sys.executable, str(main_script),
            "--targets",    *args.targets,
            "--device",     args.device,
            "--horizon",    str(args.horizon),
            "--run-name",   run_name,
            "--aggregation", exp["aggregation"],
        ]
        if exp["embedding"]:
            cmd += ["--embedding", exp["embedding"]]
        if args.no_baselines:
            cmd += ["--no-baselines"]

        print(f"Command: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"WARNING: experiment {run_name} failed with return code {result.returncode}")
            print("Continuing to next experiment...")

    print(f"\n{'═' * 60}")
    print("All experiments complete!")
    print(f"Results saved to out/holdout/experiments.csv")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()