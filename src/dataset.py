"""
dataset.py — loads a fixed subset of HotpotQA (distractor setting).

Uses HuggingFace `datasets` library to pull hotpot_qa from the Hub.
The first call streams and caches locally under ~/.cache/huggingface;
subsequent calls are instant.

We return a list of dicts with exactly the fields we care about:
  - id, question, answer
  - context: list of {"title": str, "sentences": [str]} dicts
  - supporting_titles: the two paragraph titles that contain the answer
  - supporting_facts: list of {"title": str, "sent_id": int} dicts
"""

import random
from datasets import load_dataset


def load_examples(n=200, seed=42):
    """
    Returns n examples from the HotpotQA distractor validation set.

    Each example:
      {
        "id": str,
        "question": str,
        "answer": str,
        "context": [{"title": str, "sentences": [str]}, ...],
        "supporting_titles": [str, str],
        "supporting_facts": [{"title": str, "sent_id": int}, ...],
      }
    """
    print("Loading HotpotQA from HuggingFace Hub (cached after first run)…")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    sample = [ds[i] for i in indices]

    examples = []
    for item in sample:
        # HF format: context = {"title": [...], "sentences": [[...]]}
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        context = [
            {"title": t, "sentences": s}
            for t, s in zip(titles, sentences)
        ]

        sf_titles = item["supporting_facts"]["title"]
        sf_sent_ids = item["supporting_facts"]["sent_id"]
        supporting_facts = [
            {"title": t, "sent_id": sid}
            for t, sid in zip(sf_titles, sf_sent_ids)
        ]
        supporting_titles = list({f["title"] for f in supporting_facts})

        examples.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "context": context,
            "supporting_titles": supporting_titles,
            "supporting_facts": supporting_facts,
        })
    return examples


def get_paragraph_text(para):
    """Join the sentences of a paragraph into a single string."""
    return " ".join(para["sentences"])
