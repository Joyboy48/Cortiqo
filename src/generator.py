"""
generator.py — wraps the Groq API to answer questions given retrieved context.

Kept intentionally simple: one function that takes a question + list of
paragraph dicts, builds a prompt, calls Groq, and returns the answer string.
No retry logic, no streaming — just plain synchronous calls.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        _client = Groq(api_key=api_key)
    return _client


def build_prompt(question, retrieved_paragraphs):
    """
    Puts context before the question, standard RAG format.
    retrieved_paragraphs: list of dicts with "title" and "sentences"
    """
    context_blocks = []
    for para in retrieved_paragraphs:
        text = " ".join(para["sentences"])
        context_blocks.append(f"[{para['title']}]\n{text}")
    context = "\n\n".join(context_blocks)

    return (
        f"Use the following context to answer the question.\n"
        f"Answer as briefly as possible — a phrase or a couple of words is fine.\n"
        f"If you can't find the answer in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def generate_answer(question, retrieved_paragraphs, model="llama-3.1-8b-instant"):
    """
    Calls Groq and returns the model's answer string.
    """
    prompt = build_prompt(question, retrieved_paragraphs)
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=64,
    )
    return resp.choices[0].message.content.strip()
