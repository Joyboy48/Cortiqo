"""
two_step_retrieval.py — Part B extension experiment.

Tests the hypothesis from WRITEUP.md:
  "Iterative retrieval using the hop-1 answer as a second query should
   push hop-2 recall from 70% to above 85%."

Flow:
  Step 1: Retrieve top-5 using original question  (same as baseline)
  Step 2: Ask LLM to extract the bridge entity from best retrieved para
  Step 3: Retrieve again using that bridge entity as query
  Step 4: Combine both retrievals, remove duplicates
  Step 5: Compare hop-2 recall with baseline
"""

import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import load_examples
from retriever import dense_retrieve
from evaluate import paragraph_recalled
from generator import _get_client

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
K = 5


# ──────────────────────────────────────────
# Bridge entity extractor
# ──────────────────────────────────────────

def extract_bridge_entity(question, paragraph, model="llama-3.1-8b-instant"):
    """
    Given the question and the best retrieved paragraph (hop-1),
    ask the LLM: what entity from this paragraph should I look up next?

    Returns a short string — the bridge entity name.
    """
    para_text = " ".join(paragraph["sentences"])

    prompt = (
        f"You are helping with a multi-hop question.\n"
        f"Question: {question}\n\n"
        f"Here is one relevant paragraph:\n{para_text}\n\n"
        f"What specific entity (person, place, organization, or thing) "
        f"from this paragraph should be looked up next to answer the question?\n"
        f"Reply with ONLY the entity name, nothing else."
    )

    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=20,   # entity name chota hoga
    )
    return resp.choices[0].message.content.strip()


# ──────────────────────────────────────────
# 2-step retrieval function
# ──────────────────────────────────────────

def two_step_retrieve(question, paragraphs, k=5):
    """
    Step 1: Normal dense retrieval on the question.
    Step 2: Extract bridge entity from top result, retrieve again.
    Step 3: Combine, deduplicate, return top-k unique paragraphs.
    """
    # Step 1 — normal retrieval
    first_pass = dense_retrieve(question, paragraphs, k=k)

    # Step 2 — extract bridge entity from top paragraph
    top_para = first_pass[0]
    bridge_entity = extract_bridge_entity(question, top_para)

    # Step 3 — retrieve again using bridge entity
    second_pass = dense_retrieve(bridge_entity, paragraphs, k=3)

    # Step 4 — combine: second_pass FIRST (bridge entity results),
    # then fill remaining slots from first_pass
    seen_titles = set()
    combined = []
    for para in second_pass + first_pass:   # second_pass priority!
        if para["title"] not in seen_titles:
            combined.append(para)
            seen_titles.add(para["title"])

    return combined[:k], bridge_entity  # return top-k and the bridge entity we used


# ──────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────

def run_experiment(n=200, k=K):
    examples = load_examples(n=n)
    print(f"Running 2-step retrieval experiment on {n} examples (top-k={k})\n")
    print("Note: This makes 1 extra Groq call per question to extract bridge entity.\n")

    # baseline numbers for comparison (from previous run)
    baseline_hop2 = 140   # 70.0% from hop_recall.py

    hop2_hits_twostep = 0
    hop2_hits_baseline = 0
    improvements = []   # cases where 2-step helped
    regressions = []    # cases where 2-step hurt

    for ex in tqdm(examples, desc="Evaluating"):
        sup_titles = ex["supporting_titles"]
        if len(sup_titles) < 2:
            continue

        # figure out hop-1 and hop-2 using same heuristic as hop_recall.py
        q_tokens = set(ex["question"].lower().split())
        score_0 = sum(1 for t in sup_titles[0].lower().split() if t in q_tokens)
        score_1 = sum(1 for t in sup_titles[1].lower().split() if t in q_tokens)
        hop1_title = sup_titles[0] if score_0 >= score_1 else sup_titles[1]
        hop2_title = sup_titles[1] if score_0 >= score_1 else sup_titles[0]

        # single-pass baseline recall
        single_pass = dense_retrieve(ex["question"], ex["context"], k=k)
        baseline_h2 = paragraph_recalled(single_pass, hop2_title)

        # two-step recall
        two_step, bridge = two_step_retrieve(ex["question"], ex["context"], k=k)
        twostep_h2 = paragraph_recalled(two_step, hop2_title)

        if twostep_h2:
            hop2_hits_twostep += 1
        if baseline_h2:
            hop2_hits_baseline += 1

        # log interesting cases
        if twostep_h2 and not baseline_h2:
            improvements.append({
                "question": ex["question"],
                "answer": ex["answer"],
                "hop2_title": hop2_title,
                "bridge_entity_used": bridge,
                "status": "IMPROVED — 2-step found it, single-pass missed"
            })
        elif baseline_h2 and not twostep_h2:
            regressions.append({
                "question": ex["question"],
                "answer": ex["answer"],
                "hop2_title": hop2_title,
                "bridge_entity_used": bridge,
                "status": "REGRESSED — single-pass had it, 2-step lost it"
            })

    n_total = len(examples)

    print("\n=== 2-Step Retrieval Results ===")
    print(f"  Total examples         : {n_total}")
    print(f"  Baseline Hop-2 recall  : {hop2_hits_baseline}/{n_total} ({100*hop2_hits_baseline/n_total:.1f}%)")
    print(f"  2-Step Hop-2 recall    : {hop2_hits_twostep}/{n_total} ({100*hop2_hits_twostep/n_total:.1f}%)")
    print(f"  Improvement            : +{hop2_hits_twostep - hop2_hits_baseline} examples")
    print(f"  Cases improved         : {len(improvements)}")
    print(f"  Cases regressed        : {len(regressions)}")

    print("\n--- 3 Cases where 2-step HELPED ---")
    for case in improvements[:3]:
        print(f"  Q: {case['question'][:80]}")
        print(f"  Bridge entity used : {case['bridge_entity_used']}")
        print(f"  Hop-2 found        : {case['hop2_title']}")
        print()

    print("--- Cases where 2-step HURT (regressions) ---")
    for case in regressions[:2]:
        print(f"  Q: {case['question'][:80]}")
        print(f"  Bridge used : {case['bridge_entity_used']}")
        print(f"  Hop-2 lost  : {case['hop2_title']}")
        print()

    # save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "two_step_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "n": n_total,
                "baseline_hop2_recall": hop2_hits_baseline,
                "twostep_hop2_recall": hop2_hits_twostep,
                "improvement": hop2_hits_twostep - hop2_hits_baseline,
                "cases_improved": len(improvements),
                "cases_regressed": len(regressions)
            },
            "improvements": improvements,
            "regressions": regressions
        }, f, indent=2)
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    run_experiment()
