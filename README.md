# HotpotQA RAG Baseline + Multi-Hop Retrieval Failure Analysis

CortiqoLabs take-home — NLP/GenAI track.

---

## What this is

A minimal, reproducible QA pipeline on HotpotQA (distractor setting),
followed by a targeted experiment testing whether single-pass dense retrieval
systematically fails on the second hop of multi-hop questions.

**Dataset**: HotpotQA validation set (distractor), 200 examples (seed=42)
**Retriever**: `all-MiniLM-L6-v2` via sentence-transformers, top-5 paragraphs
**Generator**: Groq API (`llama3-8b-8192`, temperature=0)

---

## Setup

```bash
# 1. clone and enter the repo
git clone <repo-url>
cd cortiqo

# 2. create a virtual env (recommended)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. install dependencies
pip install -r requirements.txt

# 4. add your API key
copy .env.example .env
# open .env and paste your GROQ_API_KEY
```

---

## Running

### Part A — Baseline pipeline

Downloads HotpotQA on first run (~55 MB), then runs retrieval + generation + evaluation.

```bash
python src/pipeline.py
```

Output:
- Console: Exact Match, F1, hop-1 and hop-2 recall, 5 sample failures
- File: `results/baseline_results.json`

### Part B — Hop recall experiment

Tests the hypothesis that hop-2 recall is significantly lower than hop-1.
Does NOT call the generator (retrieval-only, runs in ~2 minutes).

```bash
python experiments/hop_recall.py
```

Output:
- Console: hop-1 vs hop-2 recall counts + 3 concrete failure cases
- File: `results/hop_recall_failures.json`

---

## Repo structure

```
cortiqo/
├── src/
│   ├── dataset.py       # HotpotQA loader with caching
│   ├── retriever.py     # TF-IDF and dense retrievers
│   ├── generator.py     # Groq API wrapper
│   ├── evaluate.py      # EM, F1, paragraph recall
│   └── pipeline.py      # Part A end-to-end runner
├── experiments/
│   └── hop_recall.py    # Part B hypothesis test
├── results/             # auto-created on first run
├── data/                # cached HotpotQA download
├── WRITEUP.md
├── requirements.txt
└── .env.example
```

---

## Key finding (tldr)

Single-pass dense retrieval retrieves the first-hop paragraph reliably
but fails significantly more often on the second-hop (bridge) paragraph,
because the bridge entity doesn't appear in the question text.
See `WRITEUP.md` for the full analysis.
