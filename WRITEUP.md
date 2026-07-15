# Take-Home Write-Up — NLP/GenAI Track
**Astitva Arya**

---

## What I found

I started with HotpotQA because it's designed around multi-hop reasoning —
you genuinely need two separate paragraphs to answer most questions, which
makes it a good stress test for any retrieval system.

The baseline numbers weren't great (40% exact match, F1 of 0.52), which was
expected. What I wanted to understand was *why* it fails — is it the retriever
not finding the right paragraphs, or the model failing to reason even when the
right context is there?

Turns out it's both, but in different proportions.

The more interesting finding was on retrieval: the system consistently
retrieved the first supporting paragraph (92.5% of the time) but dropped
to 70% on the second one. That's a 22-point gap, and it happens for a
structural reason — the second paragraph's topic (what I'm calling the
"bridge entity") doesn't appear anywhere in the original question. The
retriever has no signal to find it.

I also noticed that even in the 126 cases where both paragraphs were
correctly retrieved, the model still got the wrong answer in 62 of them (49%).
These were mostly comparison questions — "which is older", "did both X and Y
do Z" — where the model either refused to commit ("I don't know") or
reasoned incorrectly despite having all the evidence.

---

## Why this direction

My hypothesis going in was straightforward:

> *Single-pass dense retrieval fails on the second hop because the bridge
> entity is semantically unreachable from the original question alone.*

I picked this over other angles (like context window ordering or numeric
hallucination) because it's a structural problem. A better embedding model
or more careful chunking wouldn't fix it — you'd still be asking the
retriever to find something it has no query signal for. That makes it the
more fundamental bottleneck worth characterizing first.

---

## Evidence

**Retrieval results (hop_recall.py, n=200):**

| | Count | % |
|---|---|---|
| Hop-1 retrieved @5 | 185/200 | 92.5% |
| Hop-2 retrieved @5 | 140/200 | 70.0% |
| Both retrieved | 126/200 | 63.0% |
| Gap | 45 examples | 22.5 pp |

**Baseline QA (pipeline.py, n=200):**
- Exact Match: 80/200 (40%)
- Avg F1: 0.524
- Generator wrong despite both hops present: 62/126 (49%)

Three failure cases that show the pattern clearly:

**Case 1** — *"Ken Pruitt was a Republican member of an upper house with how many members?"*
The system retrieved Ken Pruitt's paragraph fine, but "Florida Senate" never
appears in the question so it fetched Alaska, Nevada, Wisconsin legislatures
instead — all contain "upper house" so they scored high on similarity.
Answer: 40 members. System: "I don't know."

**Case 2** — *"Which Mexican and American film actress is Ethel Houbiers the French voice of?"*
Ethel Houbiers retrieved correctly. But Salma Hayek's paragraph starts with
her acting career — "Mexican and American actress" — while the query is
about "French voice actress." The embedding distance between those two is
wide. Other French voice actors ranked higher.

**Case 3** — *"A Pair of Brown Eyes and Wild Mountain Thyme is based on what artist's song?"*
Here the query itself contains "Brown Eyes" which dragged in four distractor
articles (Brown Eyes band, Beautiful Brown Eyes, etc.), pushing Wild Mountain
Thyme out of the top 5 despite it being explicitly named in the question.

---

## What I'd try next

I actually implemented this during the project — iterative retrieval where
you use the hop-1 paragraph to extract a bridge entity, then run a second
retrieval pass with that entity as the query.

First attempt had a bug (the combined results were ordered wrong so the
second-pass results got cut off before they could help). After fixing that,
hop-2 recall went from 70% to 88% — 44 cases improved, 9 regressed,
net gain of 35 questions.

The next concrete experiment would be pairing this with the full generation
pipeline to see the actual EM/F1 improvement, and understanding the 9
regression cases better — some seem to be cases where the bridge entity
extraction itself was slightly off (e.g. the model extracted "Penélope Cruz"
when "Salma Hayek" was needed, though the second retrieval still found the
right paragraph anyway in that case).

A chain-of-thought prompting setup on the generator side would be the other
thing worth testing — specifically for the 62 cases where retrieval was
perfect but reasoning failed.

---

*Code, results, and full failure logs: github.com/[repo]*
