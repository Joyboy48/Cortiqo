"""
evaluate.py — metrics for QA evaluation.

Nothing fancy. Exact match and token-level F1 ported from the official
SQuAD evaluation script logic, adapted for HotpotQA's single-string answers.
Plus a paragraph-level recall function used in Part B.
"""

import re
import string
from collections import Counter


def normalize(text):
    """Lower, strip punctuation and extra spaces — standard SQuAD normalization."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return " ".join(text.split())


def exact_match(prediction, ground_truth):
    return int(normalize(prediction) == normalize(ground_truth))


def token_f1(prediction, ground_truth):
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def paragraph_recalled(retrieved_paragraphs, target_title):
    """Returns True if the paragraph with target_title is in retrieved list."""
    retrieved_titles = {p["title"] for p in retrieved_paragraphs}
    return target_title in retrieved_titles


def aggregate_metrics(results):
    """
    results: list of dicts with keys: em, f1, hop1_recalled, hop2_recalled
    Returns a summary dict with raw counts and percentages.
    """
    n = len(results)
    em_total = sum(r["em"] for r in results)
    f1_total = sum(r["f1"] for r in results)
    hop1_hit = sum(1 for r in results if r.get("hop1_recalled", False))
    hop2_hit = sum(1 for r in results if r.get("hop2_recalled", False))
    both_hit = sum(1 for r in results if r.get("hop1_recalled") and r.get("hop2_recalled"))

    return {
        "n": n,
        "exact_match": {"count": em_total, "pct": round(100 * em_total / n, 1)},
        "f1_avg": round(f1_total / n, 3),
        "hop1_recall": {"count": hop1_hit, "pct": round(100 * hop1_hit / n, 1)},
        "hop2_recall": {"count": hop2_hit, "pct": round(100 * hop2_hit / n, 1)},
        "both_hops_recall": {"count": both_hit, "pct": round(100 * both_hit / n, 1)},
    }
