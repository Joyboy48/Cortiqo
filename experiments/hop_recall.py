"""
hop_recall.py — Part B experiment.

Hypothesis:
  Single-pass dense retrieval (all-MiniLM-L6-v2) fails significantly more
  often on the second-hop supporting paragraph than the first, because
  the second hop requires a bridge entity that's not mentioned in the question.

We test this by comparing hop-1 vs hop-2 paragraph recall across 200 examples
and logging concrete failure cases to results/hop_recall_failures.json.
"""

import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import load_examples
from retriever import dense_retrieve
from evaluate import paragraph_recalled


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
K = 5


def classify_hop_order(example, retrieved):
    """
    In HotpotQA distractor setting, supporting_titles is an unordered set.
    We try to infer which paragraph is the 'bridge' (hop-2) by asking:
    does the text of hop-1 mention key words from the question?

    Simple heuristic: the paragraph whose title appears more frequently in the
    question's tokens is the 'first hop'. The other one is hop-2.

    This is imperfect but avoids adding NER dependencies.
    """
    titles = example["supporting_titles"]
    if len(titles) < 2:
        return titles[0] if titles else None, None

    q_tokens = set(example["question"].lower().split())
    score_0 = sum(1 for t in titles[0].lower().split() if t in q_tokens)
    score_1 = sum(1 for t in titles[1].lower().split() if t in q_tokens)

    if score_0 >= score_1:
        return titles[0], titles[1]  # hop1, hop2
    else:
        return titles[1], titles[0]


def run_experiment(n=200, k=K):
    examples = load_examples(n=n)
    print(f"Running hop recall experiment on {n} examples (top-k={k})\n")

    hop1_hits = 0
    hop2_hits = 0
    both_hits = 0
    neither_hits = 0

    failures = []  # cases where hop-2 was NOT retrieved

    for ex in tqdm(examples, desc="Evaluating"):
        retrieved = dense_retrieve(ex["question"], ex["context"], k=k)
        retrieved_titles = [p["title"] for p in retrieved]

        hop1_title, hop2_title = classify_hop_order(ex, retrieved)
        h1 = paragraph_recalled(retrieved, hop1_title) if hop1_title else False
        h2 = paragraph_recalled(retrieved, hop2_title) if hop2_title else False

        if h1:
            hop1_hits += 1
        if h2:
            hop2_hits += 1
        if h1 and h2:
            both_hits += 1
        if not h1 and not h2:
            neither_hits += 1

        # log cases where hop-2 was missed (the interesting failures)
        if hop2_title and not h2:
            # find the hop-2 paragraph text so we can see what was missed
            hop2_para = next(
                (p for p in ex["context"] if p["title"] == hop2_title), None
            )
            hop2_text = " ".join(hop2_para["sentences"])[:300] if hop2_para else ""

            failures.append({
                "id": ex["id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "hop1_title": hop1_title,
                "hop2_title": hop2_title,
                "hop2_first_300_chars": hop2_text,
                "hop1_retrieved": h1,
                "retrieved_titles": retrieved_titles,
                "note": "hop-2 paragraph missing from top-5 — bridge entity not in question"
            })

    n_total = len(examples)

    print("\n=== Hop Recall Results ===")
    print(f"  Total examples      : {n_total}")
    print(f"  Hop-1 retrieved     : {hop1_hits}/{n_total} ({100*hop1_hits/n_total:.1f}%)")
    print(f"  Hop-2 retrieved     : {hop2_hits}/{n_total} ({100*hop2_hits/n_total:.1f}%)")
    print(f"  Both hops retrieved : {both_hits}/{n_total} ({100*both_hits/n_total:.1f}%)")
    print(f"  Neither hop         : {neither_hits}/{n_total} ({100*neither_hits/n_total:.1f}%)")
    print(f"  Recall gap (H1-H2)  : {hop1_hits - hop2_hits} examples ({100*(hop1_hits-hop2_hits)/n_total:.1f} pp)")

    print(f"\n  Hypothesis result:")
    gap = hop1_hits - hop2_hits
    if gap > 20:
        print(f"  [CONFIRMED] hop-2 recall is {gap} examples lower than hop-1 ({100*gap/n_total:.1f} percentage points)")
    elif gap > 0:
        print(f"  [PARTIAL] gap exists ({gap} examples) but smaller than expected")
    else:
        print(f"  [NOT CONFIRMED] no meaningful recall gap observed")

    print(f"\n--- 3 Concrete Failure Examples (hop-2 missed) ---")
    for case in failures[:3]:
        print(f"\n  Q: {case['question']}")
        print(f"  Answer: {case['answer']}")
        print(f"  Hop-1 (in question): {case['hop1_title']}  ->  retrieved: {case['hop1_retrieved']}")
        print(f"  Hop-2 (bridge):      {case['hop2_title']}  ->  NOT retrieved")
        print(f"  Retrieved instead:   {', '.join(case['retrieved_titles'])}")
        print(f"  What hop-2 said:     {case['hop2_first_300_chars'][:200]}...")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "hop_recall_failures.json")
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "n": n_total, "k": k,
                "hop1_hits": hop1_hits, "hop2_hits": hop2_hits,
                "both_hits": both_hits, "neither_hits": neither_hits,
                "recall_gap": hop1_hits - hop2_hits
            },
            "failures": failures
        }, f, indent=2)
    print(f"\nFull failure log written to {out_path}")


if __name__ == "__main__":
    run_experiment()
