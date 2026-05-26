"""
Run All: End-to-End Training and Evaluation Pipeline
=====================================================
Executes the complete pipeline in one script:
  1. Dataset loading (HuggingFace Hub → synthetic fallback)
  2. Fine-tuning MiniIntentEmbedder
  3. Evaluation (TF-IDF vs Random-Init vs Fine-tuned)
  4. Inference demo (before/after comparison)
  5. Summary report

Usage:
    cd customer_intent_search
    python run_all.py [--epochs 15] [--batch-size 64]
    python run_all.py --quick    # fast test run
"""
import argparse
import json
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end embedding fine-tuning pipeline")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size (default: 64)")
    parser.add_argument("--pairs-per-intent", type=int, default=40,
                        help="Positive pairs per intent (default: 40)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--output-dir", type=str, default="../models/finetuned")
    parser.add_argument("--results-dir", type=str, default="../results")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run (3 epochs, reduced data)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick:
        print("⚡ Quick mode enabled")
        args.epochs = 3
        args.batch_size = 32
        args.pairs_per_intent = 10

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 65)
    print("  Customer Intent Embedding — Full Pipeline")
    print("=" * 65)
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Pairs/intent:    {args.pairs_per_intent}")
    print(f"  Model output:    {args.output_dir}")
    print(f"  Results dir:     {args.results_dir}")

    total_start = time.time()

    # ── Step 1: Train ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  [1/3] TRAINING")
    print("═" * 65)

    import train as train_module

    class TrainArgs:
        def __init__(self):
            self.epochs = args.epochs
            self.batch_size = args.batch_size
            self.pairs_per_intent = args.pairs_per_intent
            self.lr = args.lr
            self.temperature = 0.05
            self.warmup_steps = 50
            self.output_dir = args.output_dir
            self.seed = 42

    model, tokenizer, model_dir = train_module.train(TrainArgs())
    print(f"\n  ✓ Training complete. Model → {model_dir}")

    # ── Step 2: Evaluate ──────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  [2/3] EVALUATION")
    print("═" * 65)

    import evaluate as eval_module

    class EvalArgs:
        def __init__(self):
            self.model_dir = args.output_dir
            self.output_dir = args.results_dir

    all_results = eval_module.run_evaluation(EvalArgs())
    print(f"\n  ✓ Evaluation complete. Results → {args.results_dir}")

    # ── Step 3: Inference demo ────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  [3/3] INFERENCE DEMO")
    print("═" * 65)

    import inference_demo
    inference_demo.run_demo(args.output_dir)

    # ── Final Summary ─────────────────────────────────────────────────────────
    total_time = time.time() - total_start

    print("\n" + "═" * 65)
    print("  PIPELINE COMPLETE")
    print("═" * 65)

    if all_results:
        metrics = ["recall@1", "recall@3", "recall@5", "mrr@5"]
        method_display = {
            "tfidf": "TF-IDF + SVD",
            "random": "Random-Init",
            "finetuned": "Fine-tuned",
        }

        print(f"\n  {'Method':<22} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'MRR@5':>8}")
        print("  " + "─" * 56)
        for key in ["tfidf", "random", "finetuned"]:
            if key not in all_results:
                continue
            res = all_results[key]
            name = method_display[key]
            vals = "  ".join(f"{res[m]*100:>6.2f}%" for m in metrics)
            print(f"  {name:<22} {vals}")

        if "tfidf" in all_results and "finetuned" in all_results:
            print("\n  Fine-tuned vs TF-IDF improvements:")
            for m in metrics:
                delta = all_results["finetuned"][m] - all_results["tfidf"][m]
                sign = "+" if delta >= 0 else ""
                print(f"    {m:<10} {sign}{delta*100:.2f}pp")

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"\n  Artifacts:")
    print(f"    Model:         {args.output_dir}/model.pt")
    print(f"    Tokenizer:     {args.output_dir}/tokenizer.json")
    print(f"    Results JSON:  {args.results_dir}/evaluation_results.json")
    print(f"    Plots:         {args.results_dir}/metrics_comparison.png")
    print(f"                   {args.results_dir}/training_curves.png")
    print(f"                   {args.results_dir}/per_intent_improvement.png")


if __name__ == "__main__":
    main()
