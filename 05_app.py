# Gradio UI — works both in Databricks (notebook) and locally
# Local: python 05_app.py
# Databricks: paste into a notebook cell and run

# If running locally, make sure 03_rag.py has been run first
# OR set STANDALONE = True below to load the FAISS index from disk

import gradio as gr
import pickle
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
HF_TOKEN    = 'hf_your_token_here'   # ← replace
HF_MODEL    = 'mistralai/Mistral-7B-Instruct-v0.2'
INDEX_PATH  = 'nyayasetu.index'      # local path; in Databricks use /tmp/nyayasetu.index

# ── Load index ────────────────────────────────────────────────────────────────
try:
    with open(INDEX_PATH, 'rb') as f:
        store = pickle.load(f)
    index       = store['index']
    chunks_text = store['texts']
    chunks_meta = store['meta']
    print(f'Index loaded: {index.ntotal} vectors')
except FileNotFoundError:
    # In Databricks the index + chunks are already in memory from 03_rag.py
    # This block is only needed for local standalone runs
    print('Index file not found — run 03_rag.py first to build and save the index.')
    index = chunks_text = chunks_meta = None

embed_model = SentenceTransformer(EMBED_MODEL)

# ── Core functions ────────────────────────────────────────────────────────────
def retrieve(query: str, k: int = 5) -> list[dict]:
    q_vec = embed_model.encode([query])[0].astype('float32').reshape(1, -1)
    D, I  = index.search(q_vec, k)
    return [
        {'text': chunks_text[idx], 'act': chunks_meta[idx]['act'], 'score': float(D[0][rank])}
        for rank, idx in enumerate(I[0])
    ]

def generate_answer(prompt: str) -> str:
    headers = {'Authorization': f'Bearer {HF_TOKEN}'}
    payload = {'inputs': prompt, 'parameters': {'max_new_tokens': 400, 'temperature': 0.3}}
    try:
        resp = requests.post(
            f'https://api-inference.huggingface.co/models/{HF_MODEL}',
            headers=headers, json=payload, timeout=30,
        )
        result = resp.json()
        if isinstance(result, list):
            return result[0].get('generated_text', '').split('Answer')[-1].strip()
        return str(result)
    except Exception as e:
        return f'[Error contacting LLM: {e}]'

def build_prompt(query: str, chunks: list[dict]) -> str:
    context = '\n\n'.join(f'[From {c["act"].upper()}]: {c["text"]}' for c in chunks)
    return (
        'You are a legal assistant for rural Indians.\n'
        'Answer in the SAME LANGUAGE as the question.\n'
        'Use ONLY the provided legal text. Cite the act name.\n'
        'If you cannot answer, say so honestly.\n\n'
        f'Legal Context:\n{context}\n\n'
        f'Question: {query}\n\n'
        'Answer:'
    )

# ── Gradio handler ────────────────────────────────────────────────────────────
def legal_query(question: str, language: str) -> tuple[str, str]:
    if not question.strip():
        return 'Please enter a question.', ''
    if index is None:
        return 'Index not loaded. Run 03_rag.py first.', ''

    chunks  = retrieve(question, k=5)
    prompt  = build_prompt(question, chunks)
    answer  = generate_answer(prompt)
    sources = '\n'.join(
        f'  - {s["act"].upper()} (score: {s["score"]:.2f})' for s in chunks
    )
    return answer, f'Sources cited:\n{sources}'

# ── UI ────────────────────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=legal_query,
    inputs=[
        gr.Textbox(
            label='Your legal question (any language)',
            placeholder='मेरे मकान मालिक ने बिना नोटिस के किराया बढ़ा दिया, मैं क्या कर सकता हूँ?',
            lines=3,
        ),
        gr.Dropdown(
            label='Your language',
            choices=['Hindi', 'Kannada', 'Tamil', 'Telugu', 'Bengali', 'English'],
            value='Hindi',
        ),
    ],
    outputs=[
        gr.Textbox(label='Legal Answer (with citations)', lines=8),
        gr.Textbox(label='Source acts used', lines=4),
    ],
    title='NyayaSetu — Vernacular Legal AI for India',
    description=(
        'Ask any legal question in Hindi, Kannada, Tamil, Telugu or English. '
        'Get cited answers from Indian law.'
    ),
    examples=[
        ['मेरे मकान मालिक ने बिना नोटिस के किराया बढ़ा दिया', 'Hindi'],
        ['ನನ್ನ MGNREGA ವೇತನ 3 ತಿಂಗಳಿಂದ ಬಂದಿಲ್ಲ', 'Kannada'],
        ['How do I file an RTI application?', 'English'],
        ['घरेलू हिंसा के मामले में मुझे क्या करना चाहिए?', 'Hindi'],
    ],
)

if __name__ == '__main__':
    # share=True gives a public URL — paste it in your README
    demo.launch(share=True)
