# Ultimate Interview Preparation Guide — NLP / GenAI Track
**Candidate: Astitva Arya**
**Project: CortiqoLabs RAG Evaluation & Iterative Retrieval Take-Home**

This guide is structured into 7 core sections ranging from basic machine learning to advanced multi-hop retrieval architectures. It contains 55 targeted questions and answers to prep for the technical discussion.

---

## Table of Contents
1. **Foundational NLP & Vector Mathematics** (Q1–Q10)
2. **Retrieval-Augmented Generation (RAG) Architecture** (Q11–Q17)
3. **Deep Dive: Project Codebase Walkthrough** (Q18–Q26)
4. **Libraries, Packages, and APIs Used** (Q27–Q33)
5. **Part B Experiment: The Multi-Hop Retrieval Hypothesis** (Q34–Q40)
6. **Part B Extension: Two-Step Iterative Retrieval & The Bug** (Q41–Q48)
7. **Generator Failure & Reasoning Bottlenecks** (Q49–Q55)

---

## 1. Foundational NLP & Vector Mathematics (Easy to Hard)

#### Q1: What is a "vector space" and what does it mean to represent text as a "vector"?
**Answer:** A vector space is a multi-dimensional mathematical space where items (like words or sentences) are represented as coordinates (points). Representing text as a vector (called embedding) means converting text into a list of numbers. The position of these numbers in the vector space is learned during training so that texts with similar semantic meanings are placed close to each other.

#### Q2: What is the difference between One-Hot Encoding and Word Embeddings?
**Answer:** 
* **One-Hot Encoding:** Represents words as sparse vectors where only one position is `1` and the rest are `0` (e.g., `[0, 1, 0, 0]`). It treats all words as independent and perpendicular; it cannot capture semantic relationships (e.g., "king" and "queen" have zero mathematical similarity).
* **Word Embeddings:** Represents words as dense vectors of real numbers (e.g., `[0.25, -0.4, 0.91]`). Similar concepts are mapped to nearby coordinates, preserving semantic relationships.

#### Q3: What is "Cosine Similarity" and why is it preferred over Euclidean Distance for comparing text?
**Answer:** Cosine Similarity measures the cosine of the angle between two vectors in a multi-dimensional space. It evaluates *direction* rather than *magnitude*.
* **Formula:** \(\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}\)
* **Preference:** Euclidean distance is highly sensitive to text length (magnitude). If two documents talk about the exact same topic, but one is 10 words and the other is 10,000 words, their Euclidean distance will be huge. Cosine similarity normalizes the lengths, focusing purely on the similarity of the content/angle.

#### Q4: What is the "Dot Product" and how does it relate to Cosine Similarity?
**Answer:** The Dot Product is the sum of the products of corresponding components of two vectors: \(A \cdot B = \sum (A_i \times B_i)\).
* **Relationship:** The dot product is equal to the Cosine Similarity multiplied by the magnitudes of both vectors: \(A \cdot B = \|A\| \|B\| \cos(\theta)\).
* **Normalized Vectors:** If the vectors are normalized (i.e., their length/magnitude is scaled to \(1.0\)), the dot product is **exactly equal** to the Cosine Similarity. In our code, since `sentence-transformers` normalizes vectors, we use the dot product (`@` matrix multiplication) for faster computations.

#### Q5: What is the difference between "Dense" and "Sparse" vectors?
**Answer:** 
* **Sparse Vectors:** Vectors where most elements are zero (e.g., TF-IDF or Bag-of-Words). Their dimensions equal the size of the vocabulary (thousands of dimensions), making them memory-heavy but great for matching exact keywords.
* **Dense Vectors:** Vectors where almost all elements are non-zero real numbers (e.g., embeddings). They have a fixed, smaller size (like 384 or 768 dimensions) but pack rich, dense semantic features into every dimension.

#### Q6: Explain the mathematical intuition behind TF-IDF.
**Answer:** TF-IDF (Term Frequency-Inverse Document Frequency) measures how important a word is to a document in a collection.
* **Term Frequency (TF):** How often a word appears in a specific document. High frequency means high importance *locally*.
* **Inverse Document Frequency (IDF):** How rare the word is across all documents in the corpus. Rare words (like "photosynthesis") get a high IDF weight; common words (like "the", "is") get an IDF of nearly zero.
* **Formula:** \(\text{TF-IDF} = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)\) where \(N\) is total documents and \(\text{DF}(t)\) is documents containing term \(t\).

#### Q7: Why does a dense embedding model like `all-MiniLM-L6-v2` understand synonyms (e.g., "film" and "movie") while TF-IDF fails?
**Answer:** TF-IDF treats "film" and "movie" as two completely independent dimensions in a sparse matrix. It only looks at spelling. `all-MiniLM-L6-v2` is trained on billions of sentences using a Transformer architecture. Through training, it learns that "film" and "movie" appear in almost identical contexts, placing their vector representations very close to each other in the dense vector space.

#### Q8: What are the drawbacks of using dense embeddings over sparse retrieval?
**Answer:** Dense embeddings can struggle with:
1. **Out-of-Vocabulary words:** Brand names, product IDs, exact numbers, or new terms not present during training.
2. **Exact matching:** A dense model might retrieve a document about "dogs" when you specifically queried for the exact string "Golden Retriever puppy", because it groups them semantically.

#### Q9: What is "dimensionality reduction" and how does a model like MiniLM achieve efficiency?
**Answer:** Dimensionality reduction is the process of mapping high-dimensional data into a lower-dimensional space while preserving the structural relationship. MiniLM is a distilled (compressed) version of BERT. It uses **knowledge distillation** to mimic the self-attention behavior of larger models (like BERT-base with 768 dimensions) but outputs only 384 dimensions, making it significantly faster and lighter for CPU execution.

#### Q10: What is "Self-Attention" in Transformer models and why is it crucial for sentence embeddings?
**Answer:** Self-attention allows a model to look at other words in a input sentence to gain context for a specific target word. For example, in "The bank of the river" vs "The bank account", the attention mechanism connects "bank" to "river" or "account" respectively, creating context-aware embeddings instead of static word representations (like Word2Vec).

---

## 2. Retrieval-Augmented Generation (RAG) Architecture (Medium)

#### Q11: What is RAG and what problem does it solve for LLMs?
**Answer:** RAG (Retrieval-Augmented Generation) combines information retrieval with language generation. It solves three critical LLM limitations:
1. **Outdated Knowledge:** LLMs cannot access real-time data beyond their training cutoff. RAG pulls updated info from a database.
2. **Hallucinations:** RAG anchors the LLM's response in fetched source documents.
3. **Domain Specificity:** Instead of expensive training on corporate data, RAG retrieves internal docs dynamically.

#### Q12: Why would you choose RAG over Fine-Tuning a model on custom data?
**Answer:** 
* **RAG** is cheaper, allows real-time data updates (just update the DB), and provides source traceability (citations).
* **Fine-Tuning** is expensive, slow, and prone to "hallucinating" facts, though it is superior for teaching the model a specific writing style, tone, or complex syntax formatting.

#### Q13: What is "Chunking" and why is it a critical preprocessing step in production RAG?
**Answer:** Chunking is the process of splitting large documents into smaller, coherent text segments (e.g., paragraphs or 500-token blocks) before embedding them. 
* **Importance:** If you embed a whole book or a massive PDF as a single vector, its specific details get smoothed out, leading to poor retrieval. Small chunks preserve local details. It also ensures the retrieved context fits within the LLM's limit (context window).

#### Q14: What is the "Lost in the Middle" phenomenon in LLMs?
**Answer:** Research shows that LLMs are great at identifying information placed at the very beginning or end of their input context, but frequently fail to retrieve/reason over information placed in the middle of long prompts. To mitigate this, we limit our retrieval parameter to top-k (e.g., \(k=5\)) rather than dumping 20 paragraphs into the prompt.

#### Q15: What is a "Vector Database" and when do you need one in a RAG system?
**Answer:** A Vector DB (like pgvector, Pinecone, Milvus) is a database optimized for storing and querying multi-dimensional vectors using indexing algorithms (like HNSW or IVF-Flat) for Approximate Nearest Neighbor (ANN) search.
* **When needed:** When your document set scales to thousands or millions. For small datasets (like our 10 paragraphs per question in HotpotQA), we can calculate Cosine Similarity in-memory using NumPy/Scipy without a database.

#### Q16: What is "Reranking" and how does it fit into the RAG workflow?
**Answer:** Reranking is a two-stage retrieval process. 
1. **Stage 1 (Bi-Encoder):** A fast dense retriever (like MiniLM) quickly fetches the top 20–50 documents.
2. **Stage 2 (Cross-Encoder / Reranker):** A slower, highly accurate model reads the question and the fetched docs together, scoring their exact semantic match to reorder them, keeping only the top 3–5. This significantly boosts retrieval precision.

#### Q17: What is "Grounding" in GenAI?
**Answer:** Grounding is the practice of ensuring that the LLM's output is strictly based on verified, external context rather than its internal parametric memory. In RAG, we ground the LLM by explicitly telling it in the prompt: *"Answer only based on the provided context. If it's not there, say 'I don't know'."*

---

## 3. Deep Dive: Project Codebase Walkthrough (Medium to Hard)

#### Q18: Walk me through the directory structure of your take-home code.
**Answer:** 
* `src/dataset.py`: Loads the HotpotQA validation dataset from HuggingFace.
* `src/retriever.py`: Vectorizes the text using `all-MiniLM-L6-v2` and calculates dot-product similarity.
* `src/generator.py`: Initializes the Groq client and queries `llama-3.1-8b-instant` with temperature 0.
* `src/evaluate.py`: Computes Exact Match, Token F1, and paragraph recall.
* `src/pipeline.py`: Runs Part A (baseline RAG evaluation on 200 samples).
* `experiments/hop_recall.py`: Tests the recall gap between hop-1 and hop-2 paragraphs.
* `experiments/two_step_retrieval.py`: Implements our iterative retrieval solution.
* `results/`: Stores JSON results of all runs.
* `WRITEUP.md`: The final 1-page summary of insights.

#### Q19: How did you implement evaluation metrics in `evaluate.py`?
**Answer:** 
* **Exact Match (EM):** Standardized both prediction and gold answer (lowercased, stripped punctuation, removed articles `a`, `an`, `the`) and returned `1` if they matched exactly, `0` otherwise.
* **F1 Score:** Tokenized both normalized strings. Calculated Precision (overlapping tokens / predicted tokens) and Recall (overlapping tokens / gold tokens). Returned \(2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\).
* **Recall @ K:** Checked if the target supporting paragraph titles were present in the top-k retrieved list.

#### Q20: In `retriever.py`, why did you use matrix multiplication (`q_emb @ c_emb.T`) for calculating similarity?
**Answer:** Since our embedding vectors generated by `sentence-transformers` are L2-normalized (length = 1.0), the dot product between them is mathematically identical to Cosine Similarity. Matrix multiplication (`@`) is highly optimized in Python/PyTorch, allowing us to compute similarity scores between the query and all 10 candidate paragraphs in a single, fast matrix operation.

#### Q21: Explain the prompt structure you used in `generator.py` to ensure the LLM doesn't hallucinate.
**Answer:** We used a clean, structured template:
```text
Use the following context to answer the question. 
Answer as briefly as possible -- a phrase or a couple of words is fine. 
If you can't find the answer in the context, say 'I don't know'.

Context:
[Title 1]
Paragraph content 1...

[Title 2]
Paragraph content 2...

Question: {question}
Answer:
```
This forces the model to extract directly from the provided text and provides an escape hatch (`I don't know`) if the retriever failed.

#### Q22: Why did you set `temperature=0.0` in the Groq generation parameters?
**Answer:** We wanted our evaluation to be deterministic and reproducible. At \(T=0\), the LLM always picks the token with the absolute highest probability. If we run the pipeline multiple times, the metrics (EM/F1) will remain constant, ensuring our scientific evaluation isn't affected by random generation variance.

#### Q23: Why did you set `max_tokens=64` in the generator config?
**Answer:** HotpotQA gold answers are short phrases (e.g., "40 members", "Salma Hayek"). Setting `max_tokens=64` prevents the model from generating long, conversational preambles (like "The answer to your question is...") which would ruin our Exact Match score.

#### Q24: How does `dataset.py` load the dataset? What split did you choose?
**Answer:** It uses the HuggingFace `datasets` library to load `hotpotqa/hotpot_qa` (version: `distractor`). We used the `validation` split because the test split of HotpotQA does not contain public ground-truth labels. We processed the first 200 samples where there were at least 2 supporting facts.

#### Q25: How did you implement seed control in `dataset.py`?
**Answer:** We didn't just take the first 200 elements, which could contain bias. We loaded the dataset, applied a shuffle using a fixed generator seed (`seed=42`), and then sliced the first 200 elements. This ensures reproducibility across any machine running the pipeline.

#### Q26: Explain how you classified "hop-1" and "hop-2" paragraphs in your experiment scripts.
**Answer:** Since HotpotQA doesn't explicitly label which paragraph is the first hop vs the second hop, we wrote a heuristic:
1. We tokenized the question and both supporting paragraph titles.
2. We counted how many unique word tokens of each title appeared in the question.
3. The paragraph whose title had the **highest token overlap** with the question was labeled **Hop-1**.
4. The remaining paragraph (which has little to no overlap because it acts as the bridge) was labeled **Hop-2**.

---

## 4. Libraries, Packages, and APIs Used (Easy to Medium)

#### Q27: What is `sentence-transformers` and why did we choose it over standard Hugging Face model loading?
**Answer:** `sentence-transformers` is a specialized library built on top of PyTorch and Transformers. It is designed specifically to output high-quality sentence embeddings using Siamese networks. Standard Hugging Face transformer models require manual pooling (e.g., mean pooling of token outputs) to get a single sentence vector, whereas `sentence-transformers` does this automatically in one clean API call (`model.encode()`).

#### Q28: What is `scikit-learn` used for in your pipeline?
**Answer:** In our baseline evaluation scripts, we kept `scikit-learn` to calculate classical tf-idf representations (`TfidfVectorizer`) as a fast, keyword-based retrieval benchmark. We also utilized its matrix capabilities for cosine similarity comparisons during development.

#### Q29: What is the Groq API and why did we use it instead of running Llama locally?
**Answer:** Groq is an API provider that runs large language models on their custom hardware called LPU (Language Processing Unit). It serves tokens extremely fast (hundreds of tokens per second). Running `llama-3.1-8b-instant` locally would require heavy consumer GPUs, whereas Groq allowed us to evaluate 200 multi-hop reasoning questions in under 2 minutes.

#### Q30: What is `tqdm`?
**Answer:** `tqdm` is a fast, extensible progress bar library for Python. We wrapped our evaluation loops in it to track progress, execution speed (iterations per second), and estimated time of arrival (ETA) during baseline runs.

#### Q31: How does the HuggingFace `datasets` cache mechanism work?
**Answer:** When you run `load_dataset('hotpotqa')`, the library downloads the raw data files, processes them into Arrow tables, and stores them locally (in `~/.cache/huggingface/datasets`). If the script is run a second time, it loads directly from the local disk in milliseconds, preventing unnecessary network calls.

#### Q32: What is the role of `python-dotenv` in the codebase?
**Answer:** It reads key-value pairs from a `.env` file and adds them to the environment variables (`os.environ`). This allows us to keep our sensitive Groq API Key out of the source code and git history, loading it securely at runtime.

#### Q33: Explain the `git` setup you created for this project.
**Answer:** I initialized a clean Git repository and established a workflow showing logical incremental progression. I committed in phases: project setup, Part A implementation, Part B experiment, debugging/refining, and results compilation. I also added a `.gitignore` to ensure massive data caches, temporary files, and local credentials (like `.env`) are never pushed.

---

## 5. Part B Experiment: The Multi-Hop Retrieval Hypothesis (Medium to Hard)

#### Q34: What was your core hypothesis for Part B?
**Answer:** 
> *Single-pass dense retrieval fails significantly more on the second-hop (bridge) paragraph compared to the first-hop paragraph because the bridge entity is absent from the question text, leaving the retriever with zero query signal to find it.*

#### Q35: What were the quantitative results of the `hop_recall.py` experiment?
**Answer:** In our validation run over 200 questions:
* **Hop-1 Recall @ 5:** **92.5%** (185/200)
* **Hop-2 Recall @ 5:** **70.0%** (140/200)
* **Gap:** **22.5 percentage points** (45 questions where hop-1 was found but hop-2 was completely missed).
* **Both hops retrieved:** **63.0%** (126/200).

#### Q36: Walk through the "Ken Pruitt" failure example.
**Answer:** 
* **Question:** *"Ken Pruitt was a Republican member of an upper house of the legislature with how many members?"*
* **Hop-1:** "Ken Pruitt" (retrieved successfully because "Ken Pruitt" is in the query).
* **Hop-2 Needed:** "Florida Senate" (not retrieved).
* **Why it failed:** The question never mentions "Florida". The retriever matches other state legislatures (Alaska, Nevada) because they contain words like "upper house", "legislature", and "members". "Florida Senate" is only semantically linked once you know Ken Pruitt was a Florida senator — which is locked inside the Hop-1 paragraph.

#### Q37: Walk through the "Ethel Houbiers / Salma Hayek" failure example.
**Answer:** 
* **Question:** *"Which Mexican and American film actress is Ethel Houbiers the French voice of?"*
* **Hop-1:** "Ethel Houbiers" (retrieved successfully).
* **Hop-2 Needed:** "Salma Hayek" (not retrieved).
* **Why it failed:** The question embeddings point strongly to "French voice actress" topics. Salma Hayek's Wikipedia page focuses heavily on her Hollywood acting career, creating a wide semantic gap in the vector space between "French voice of" and "Mexican-American film actress." The retriever pulled other French voice actors instead.

#### Q38: Walk through the "A Pair of Brown Eyes / Wild Mountain Thyme" failure example.
**Answer:** 
* **Question:** *"A Pair of Brown Eyes and Wild Mountain Thyme is based from what artist's song?"*
* **Hop-1:** "A Pair of Brown Eyes" (retrieved).
* **Hop-2 Needed:** "Wild Mountain Thyme" (not retrieved).
* **Why it failed (Semantic Collision):** The query contains "Brown Eyes", which pulls in four distractor paragraphs containing that exact phrase (e.g., "Brown Eyes (band)", "Beautiful Brown Eyes", "Brown Eyes (song)"). The correct paragraph "Wild Mountain Thyme" was outranked and pushed out of the top 5 by these vocabulary collisions.

#### Q39: What is a "Bridge Entity" in multi-hop QA?
**Answer:** The bridge entity is the connecting link (node) that joins the question to the final answer. In the Ken Pruitt question, the question links to "Ken Pruitt" (Node A). Node A contains the fact that he was in the "Florida Senate" (Bridge Entity / Node B). Node B contains the fact that it has "40 members" (Answer / Node C). You cannot reach Node C from the question without going through Node B.

```
Question ──► [Node A: Ken Pruitt] ──► (Bridge: Florida Senate) ──► [Node C: 40 members]
```

#### Q40: Why is this recall gap a "structural" problem rather than an "embedding quality" problem?
**Answer:** Because it's an information availability issue. Even if you use a state-of-the-art embedding model with 10,000 dimensions, the text "Florida Senate" is still absent from the question. The model has no mathematical reason to rank "Florida Senate" higher than "Alaska Legislature" unless it first processes the fact that Ken Pruitt belongs to Florida — which a single-pass vectorizer cannot do because it evaluates context elements independently.

---

## 6. Part B Extension: Two-Step Iterative Retrieval & The Bug (Hard)

#### Q41: Explain your implementation of the 2-Step Iterative Retrieval in `two_step_retrieval.py`.
**Answer:** 
1. **Pass 1:** Retrieve top-5 paragraphs using the original question.
2. **Bridge Extraction:** Take the top retrieved paragraph (hop-1) and send it to the LLM (`llama-3.1-8b-instant`) with a prompt asking: *"What entity from this paragraph should be looked up next to answer the question?"*
3. **Pass 2:** Retrieve 3 paragraphs using that extracted bridge entity as the new query.
4. **Combine:** Merge Pass 1 and Pass 2 results, deduplicate, and return the top-5.

#### Q42: What was the bug in your initial 2-step combining logic, and how did you debug it?
**Answer:** 
* **The Bug:** In my first version of `two_step_retrieve()`, I merged the lists as `first_pass + second_pass` and then took `combined[:5]`. Since `first_pass` already had 5 elements, slicing `[:5]` completely discarded all the new results fetched in `second_pass` (including the target hop-2 paragraph). The recall remained at 70% (0% improvement).
* **How I debugged it:** I wrote `debug_twostep.py` to evaluate three known failures. I printed out the raw lists at each stage. I saw that `second_pass` correctly retrieved "Florida Senate", and it was present in `combined`, but got sliced off due to order priority.
* **The Fix:** I updated the merge order to prioritize the second pass: `second_pass + first_pass`. This guaranteed the newly discovered bridge context was placed in the top slots.

#### Q43: What were the final results after fixing the bug?
**Answer:** 
* **Baseline Hop-2 Recall:** **70.5%** (141/200)
* **2-Step Hop-2 Recall:** **88.0%** (176/200)
* **Net Improvement:** **+35 questions** (+17.5 percentage points)
* **Details:** 44 cases improved, 9 cases regressed.

#### Q44: Tell me about a case where the 2-step retrieval improved the recall.
**Answer:** The "Ken Pruitt" case.
* Pass 1 fetched "Ken Pruitt" as top result.
* LLM read the "Ken Pruitt" paragraph and extracted: `"Florida Senate"`.
* Pass 2 queried `"Florida Senate"` and immediately retrieved the "Florida Senate" paragraph.
* Combined output had both paragraphs in top-5.

#### Q45: Explain the "Penélope Cruz / Salma Hayek" edge case. Why did Varun Sharma call this "excellent instinct"?
**Answer:** 
* **The case:** The question asked which actress Ethel Houbiers was the French voice of (Answer: Salma Hayek).
* **What happened:** The LLM read the hop-1 paragraph and incorrectly extracted `"Penélope Cruz"` as the bridge entity. However, during Pass 2, the retriever still successfully fetched `"Salma Hayek"`!
* **Why it worked:** Because we query a closed set of 10 paragraphs. "Penélope Cruz" and "Salma Hayek" are semantically very close (both Mexican-American actresses). The dense vector for "Penélope Cruz" had a high similarity score with the "Salma Hayek" paragraph, bringing it into the top-3.
* **Why it matters:** It shows that dense retrieval is robust to minor errors in LLM bridge extraction, but it also highlights that we can't fully trust the intermediate extraction step even when the end result is correct.

#### Q46: What were the "regressions" (9 cases) and why did they happen?
**Answer:** A regression is a case where the baseline single-pass retriever found the hop-2 paragraph, but the 2-step retriever lost it.
* **Why they happened:** If the first pass retrieved a completely irrelevant paragraph as the top result, the LLM extracted a garbage bridge entity. Querying that garbage entity in Pass 2 filled the top slots of our combined list with irrelevant documents, pushing the correct hop-2 paragraph (which was originally in slot 4 or 5 of the first pass) out of the top 5.

#### Q47: How would you resolve these regressions in a production system?
**Answer:** 
1. **Self-Consistency/Verification:** Ask the LLM if the extracted bridge entity actually relates to the question before running Pass 2.
2. **Confidence Thresholding:** Only trigger the second pass if the top result in Pass 1 has a similarity score above a certain threshold (e.g., > 0.7).
3. **Reranking:** Instead of hard-slicing to top-5, keep all 8 unique retrieved paragraphs and use a Cross-Encoder to rank them at the end.

#### Q48: What is "Iterative Retrieval" (IR) and how does our 2-step method differ from standard IRCoT (Information Retrieval Collaborative with Chain-of-Thought)?
**Answer:** 
* **Iterative Retrieval** is a general pattern of retrieving, updating the query with retrieved info, and retrieving again.
* **IRCoT** generates a full chain-of-thought reasoning step, uses that step to retrieve, generates the next step, and repeats.
* **Our 2-step method** is much lighter. It doesn't write reasoning steps; it strictly asks the LLM to extract a search term (bridge entity) from a document. This makes it faster and significantly cheaper in API token usage.

---

## 7. Generator Failure & Reasoning Bottlenecks (Hard)

#### Q49: You found that in 49% of cases where both supporting facts were retrieved, the model still failed. Why?
**Answer:** These were reasoning failures, not retrieval failures. The reasons fall into three classes:
1. **Numeric and Comparison Logic:** The model had both paragraphs (e.g., containing heights of two structures) but failed to compare them correctly.
2. **Over-cautiousness ("I don't know" loops):** When prompt instructions are very strict about not hallucinating, the model becomes risk-averse and answers "I don't know" if the answer requires making a logical jump.
3. **String Matching Delays:** The model outputted the correct concept but paraphrased it, failing the strict Exact Match metric.

#### Q50: How does "Chain-of-Thought" (CoT) prompting help fix generator reasoning failures?
**Answer:** CoT forces the LLM to write out its step-by-step reasoning (e.g., *"Step 1: Structure A is 50m. Step 2: Structure B is 60m. Step 3: 60m is larger than 50m, so Structure B is larger."*) before writing the final answer. This activates more computation paths in the model, helping it solve math and comparison tasks much more accurately than direct answering.

#### Q51: If F1 score is high but Exact Match is 0, what does that tell you about the generator?
**Answer:** It tells you that the generator understood the question and extracted the correct information, but added extra conversational filler or used slightly different phrasing (e.g., Gold: `"Francis McPeake"`, Prediction: `"Francis McPeake 1st"`). F1 gives partial credit for overlapping words, while EM penalizes any character difference.

#### Q52: What is the difference between "parametric memory" and "source memory" in LLMs?
**Answer:** 
* **Parametric Memory:** The knowledge stored within the model's weights during its pre-training phase (what it "knows" internally).
* **Source Memory:** The context provided in the prompt at inference time (the retrieved paragraphs). RAG relies heavily on source memory to override parametric memory errors.

#### Q53: How would you solve comparison logic issues in RAG without changing the LLM size?
**Answer:** I would implement a **Structured Tool Calling** flow. When a question is classified as a comparison (e.g., "Which is older...?"), the LLM is forced to output a JSON containing the target entities and their parameters (e.g., `{"Entity_A": "Greyia", "Species": 3, "Entity_B": "Calibanus", "Species": 2}`). We then run a simple Python script to calculate the greater value, bypassing the LLM's weak mathematical reasoning.

#### Q54: What is "In-Context Learning" (ICL) and how did we use it in the generator?
**Answer:** In-Context Learning is the ability of an LLM to understand tasks and follow instructions based purely on examples and rules provided in the prompt context, without updating its weights. We used it by supplying the retrieved paragraphs as context and instructing the model on the output format ("Answer as briefly as possible").

#### Q55: If you had 6 more months on this project, what architecture would you build?
**Answer:** 
1. **Agentic Router:** An LLM classifier at the front gate to identify if the question is "Single-Hop" (run normal RAG), "Multi-Hop" (run 2-Step Iterative Retrieval), or "Comparison/Aggregator" (run code-execution tool calling).
2. **Dense Reranker:** Integrate a Cross-Encoder (like `bge-reranker-large`) after the retrieval steps to weed out the vocabulary distractors.
3. **Finetuning the Bridge Extractor:** Finetune a small, 1B-parameter model specifically for the "bridge entity extraction" task to remove the cost of calling a larger LLM for the intermediate step.
