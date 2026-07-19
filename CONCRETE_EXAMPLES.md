# Concrete Interview Case Studies & Examples
**Candidate: Astitva Arya**

This document contains 10 actual cases from the project logs. Use these exact names, questions, and numbers during the interview to demonstrate deep familiarity with the experimental data.

---

## Category 1: Single-Pass Retrieval Failures (Hop-2 Misses)
*These cases prove the core hypothesis: Hop-2 is missed because the query lacks the bridge entity.*

### Case 1: The "Ken Pruitt" Case (Classic Bridge Entity Miss)
* **Question:** *"Ken Pruitt was a Republican member of an upper house of the legislature with how many members?"*
* **Gold Answer:** `40 members`
* **Hop-1:** `Ken Pruitt` (Retrieved successfully)
* **Hop-2:** `Florida Senate` (NOT retrieved)
* **Retrieved instead:** `Alaska Legislature`, `Nevada Legislature`, `Wisconsin Legislature`
* **Why it failed:** The question never mentions "Florida". The retriever matches other legislatures because they share general keywords like "upper house", "legislature", and "members". "Florida Senate" only becomes relevant once you read Hop-1.

### Case 2: The "Ethel Houbiers" Case (Pronoun/Role Distance)
* **Question:** *"Which Mexican and American film actress is Ethel Houbiers the French voice of?"*
* **Gold Answer:** `Salma Hayek Pinault`
* **Hop-1:** `Ethel Houbiers` (Retrieved successfully)
* **Hop-2:** `Salma Hayek` (NOT retrieved)
* **Retrieved instead:** `Bernard Alane`, `Virginie Ledieu` (other French voice actors)
* **Why it failed:** The query embeddings are heavily biased toward "French voice actor" semantics. Salma Hayek's wiki article starts with her Hollywood acting career. The semantic distance between these concepts caused the retriever to miss her.

### Case 3: The "Wild Mountain Thyme" Case (Semantic Collision)
* **Question:** *"A Pair of Brown Eyes and Wild Mountain Thyme is based from what artists song?"*
* **Gold Answer:** `Francis McPeake`
* **Hop-1:** `A Pair of Brown Eyes` (Retrieved successfully)
* **Hop-2:** `Wild Mountain Thyme` (NOT retrieved)
* **Retrieved instead:** `Brown Eyes (band)`, `Beautiful Brown Eyes`, `Brown Eyes (song)`
* **Why it failed:** The question contains the phrase "Brown Eyes". This caused "vocabulary collision," pulling in multiple distractor articles containing "Brown Eyes" and pushing the correct "Wild Mountain Thyme" paragraph out of the top 5.

---

## Category 2: Generator Failures (Perfect Retrieval, Wrong Answer)
*These cases prove that even when the context is 100% correct, the LLM fails at reasoning.*

### Case 4: The "Greyia vs. Calibanus" Case (Comparison Logic Failure)
* **Question:** *"Between Greyia and Calibanus, which genus contains more species?"*
* **Gold Answer:** `Greyia`
* **Context provided:** Contains paragraphs for both genera stating their species count.
* **Model Output:** `"I don't know."`
* **Why it failed:** The strict "do not hallucinate" instructions made the model over-cautious. Rather than performing a numeric comparison, it refused to commit and defaulted to the escape phrase.

### Case 5: The "John Updike / Tom Clancy" Case (Assertion Failure)
* **Question:** *"Did John Updike and Tom Clancy both publish more than 15 bestselling novels?"*
* **Gold Answer:** `yes`
* **Context provided:** Contains bios for both authors detailing their book counts.
* **Model Output:** `"No."`
* **Why it failed:** Standard LLMs process text sequentially. Without Chain-of-Thought (CoT) prompting to evaluate John Updike and Tom Clancy separately first, it made a logical error and outputted the wrong binary answer.

---

## Category 3: Two-Step Retrieval Wins
*These cases demonstrate the success of our iterative retrieval fix.*

### Case 6: "Ken Pruitt" Fixed
* **Pass 1:** Retrieved `Ken Pruitt` as the top result.
* **LLM Extraction:** Extracted the entity `"Florida Senate"` from the Hop-1 paragraph.
* **Pass 2:** Queried `"Florida Senate"` and successfully retrieved the missing Hop-2 paragraph.
* **Final Result:** Both paragraphs present in top-5. Recall succeeded.

### Case 7: "Wild Mountain Thyme" Fixed
* **Pass 1:** Retrieved `A Pair of Brown Eyes` as the top result.
* **LLM Extraction:** Extracted the artist `"Francis McPeake"` from the Hop-1 text.
* **Pass 2:** Queried `"Francis McPeake"` and successfully retrieved `Wild Mountain Thyme`.
* **Final Result:** Recall succeeded, bypassing the "Brown Eyes" distractor collision.

---

## Category 4: Two-Step Regressions (The 9 Lost Cases)
*These cases show why 2-step retrieval occasionally hurts baseline results.*

### Case 8: The "Tim Cluess" Case (Wrong Year Context Shift)
* **Question:** *"Which head coach has led their team for a longer period of time, Tim Cluess or..."*
* **Baseline Result:** Found both `Tim Cluess` and the target team season (`2015-16 Iona Gaels men's basketball team`).
* **2-Step Fail:** The first pass retrieved `Tim Cluess` as top. The LLM extracted `"Tim Cluess"` as the bridge. Pass 2 searched for "Tim Cluess" again, fetching different years' basketball articles. The combined step pushed the correct `2015-16` season article out of the top-5.

### Case 9: The "Apollo 14 Moon Trees" Case (Concept Drift)
* **Question:** *"In the NASA mission where Moon trees were taken into space, what was the nickname..."*
* **Baseline Result:** Retrieved the astronaut (`Stuart Roosa`) at position 5.
* **2-Step Fail:** The LLM extracted `"Apollo 14"` from the top result. Pass 2 queried "Apollo 14", which pulled 3 highly similar general Apollo articles. Combining them pushed `Stuart Roosa` down to position 8, cutting him out of the final top-5.

---

## Category 5: The Famous "Penélope Cruz" Edge Case
*This case highlights the semantic grouping property of dense vector spaces.*

### Case 10: "Ethel Houbiers" (Successful Error Recovery)
* **Target Hop-2:** `Salma Hayek`
* **What the LLM did:** Extracted the wrong bridge entity: `"Penélope Cruz"`.
* **What the Retriever did:** Searched for "Penélope Cruz". Because dense vector spaces map similar entities close together, the query vector for "Penélope Cruz" had high similarity with the "Salma Hayek" paragraph.
* **Result:** Succeeded! The retriever fetched the correct "Salma Hayek" article anyway.
