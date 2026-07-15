"""
retriever.py — two retrieval strategies side by side:
  1. TF-IDF (fast, bag-of-words baseline)
  2. Dense (sentence-transformers MiniLM, semantic)

Both take a question string and the context pool, return
the top-k paragraph dicts sorted by descending score.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


# load once, reuse across calls
_dense_model = None


def _get_dense_model():
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _dense_model


# ---------- TF-IDF retriever ----------

def tfidf_retrieve(question, paragraphs, k=5):
    """
    paragraphs: list of dicts with "title" and "sentences" keys
    returns: top-k paragraph dicts
    """
    corpus = [" ".join(p["sentences"]) for p in paragraphs]
    docs = [question] + corpus
    vec = TfidfVectorizer().fit_transform(docs)
    scores = cosine_similarity(vec[0:1], vec[1:])[0]
    top_k_idx = np.argsort(scores)[::-1][:k]
    return [paragraphs[i] for i in top_k_idx]


# ---------- Dense retriever ----------

def dense_retrieve(question, paragraphs, k=5):
    """
    Single-pass dense retrieval using all-MiniLM-L6-v2.
    paragraphs: list of dicts with "title" and "sentences" keys
    returns: top-k paragraph dicts
    """
    model = _get_dense_model()
    corpus = [" ".join(p["sentences"]) for p in paragraphs]
    q_emb = model.encode([question], normalize_embeddings=True)
    c_emb = model.encode(corpus, normalize_embeddings=True)
    scores = (q_emb @ c_emb.T)[0]
    top_k_idx = np.argsort(scores)[::-1][:k]
    return [paragraphs[i] for i in top_k_idx]
