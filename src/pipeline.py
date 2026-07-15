"""
pipeline.py — Part A baseline: retrieve → generate → evaluate.

Run this to reproduce the baseline numbers:
  python src/pipeline.py

Prints a summary table and writes results to results/baseline_results.json.
"""

import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_examples
from retriever import dense_retrieve
from generator import generate_answer
from evaluate import exact_match, token_f1, paragraph_recalled, aggregate_metrics


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
K = 5  # number of paragraphs to retrieve


def run_pipeline(n=200, k=K):
    examples = load_examples(n=n)
    print(f"Running baseline on {len(examples)} examples (top-k={k})…\n")

    results = []
    for ex in tqdm(examples, desc="Processing"):
        retrieved = dense_retrieve(ex["question"], ex["context"], k=k)
        prediction = generate_answer(ex["question"], retrieved)

        # there are always exactly 2 supporting paragraph titles in hotpotqa
        sup_titles = ex["supporting_titles"]
        hop1 = sup_titles[0] if len(sup_titles) > 0 else None
        hop2 = sup_titles[1] if len(sup_titles) > 1 else None

        row = {
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "prediction": prediction,
            "em": exact_match(prediction, ex["answer"]),
            "f1": token_f1(prediction, ex["answer"]),
            "hop1_title": hop1,
            "hop2_title": hop2,
            "hop1_recalled": paragraph_recalled(retrieved, hop1) if hop1 else None,
            "hop2_recalled": paragraph_recalled(retrieved, hop2) if hop2 else None,
            "retrieved_titles": [p["title"] for p in retrieved],
        }
        results.append(row)

    metrics = aggregate_metrics(results)

    print("\n=== Baseline Results ===")
    print(f"  Total examples   : {metrics['n']}")
    print(f"  Exact Match      : {metrics['exact_match']['count']}/{metrics['n']} ({metrics['exact_match']['pct']}%)")
    print(f"  Avg F1           : {metrics['f1_avg']}")
    print(f"  Hop-1 Recall @{k} : {metrics['hop1_recall']['count']}/{metrics['n']} ({metrics['hop1_recall']['pct']}%)")
    print(f"  Hop-2 Recall @{k} : {metrics['hop2_recall']['count']}/{metrics['n']} ({metrics['hop2_recall']['pct']}%)")
    print(f"  Both Hops Hit    : {metrics['both_hops_recall']['count']}/{metrics['n']} ({metrics['both_hops_recall']['pct']}%)")

    # print a few failure cases so we can see what's going wrong
    print("\n--- Sample failures (wrong answer) ---")
    failures = [r for r in results if r["em"] == 0][:5]
    for f in failures:
        print(f"  Q: {f['question'][:80]}")
        print(f"     Gold: {f['answer']}  |  Predicted: {f['prediction']}")
        print(f"     Hop-1 retrieved: {f['hop1_recalled']}  |  Hop-2 retrieved: {f['hop2_recalled']}")
        print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    print(f"Full results written to {out_path}")

    return metrics, results


if __name__ == "__main__":
    run_pipeline()
