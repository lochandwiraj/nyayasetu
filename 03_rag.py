# Databricks notebook: nyayasetu_03_rag
# Depends on: 02_silver_gold.py having written the Gold Delta table

import faiss
import numpy as np
import pickle
import requests
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer

spark = SparkSession.builder.appName('NyayaSetu').getOrCreate()

EMBED_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# ── Cell 1: Build FAISS index from Gold Delta table ───────────────────────────
df_gold = spark.read.format('delta').load('/FileStore/nyayasetu/gold/legal_embeddings')

rows       = df_gold.select('chunk_id', 'act', 'chunk', 'embedding').collect()
chunks_text = [r['chunk']     for r in rows]
chunks_meta = [{'act': r['act'], 'chunk_id': r['chunk_id']} for r in rows]
embeddings  = np.array([r['embedding'] for r in rows], dtype='float32')

dim   = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f'FAISS index built: {index.ntotal} vectors, dim={dim}')

# Persist to DBFS so other notebooks can reload without rebuilding
with open('/tmp/nyayasetu.index', 'wb') as f:
    pickle.dump({'index': index, 'texts': chunks_text, 'meta': chunks_meta}, f)

dbutils.fs.cp('file:/tmp/nyayasetu.index', 'dbfs:/FileStore/nyayasetu/faiss.index')
print('FAISS index saved to DBFS')

# ── Cell 2: Retrieval function ────────────────────────────────────────────────
embed_model = SentenceTransformer(EMBED_MODEL)

def retrieve(query: str, k: int = 5) -> list[dict]:
    q_vec = embed_model.encode([query])[0].astype('float32').reshape(1, -1)
    D, I  = index.search(q_vec, k)
    return [
        {
            'text':  chunks_text[idx],
            'act':   chunks_meta[idx]['act'],
            'score': float(D[0][rank]),
        }
        for rank, idx in enumerate(I[0])
    ]

def build_prompt(query: str, retrieved: list[dict]) -> str:
    context = '\n\n'.join(
        f'[From {c["act"].upper()}]: {c["text"]}' for c in retrieved
    )
    return (
        'You are a legal assistant for rural Indians.\n'
        'Answer the following question in the SAME LANGUAGE it is asked in.\n'
        'Use ONLY the provided legal text. Cite the act name in your answer.\n'
        'If you cannot answer from the text, say so honestly.\n\n'
        f'Legal Context:\n{context}\n\n'
        f'Question: {query}\n\n'
        'Answer (in the same language as the question):'
    )

# ── Cell 3: LLM generation via HuggingFace Inference API ─────────────────────
# Get a free token at: huggingface.co → Settings → Access Tokens
HF_TOKEN = 'hf_your_token_here'   # ← replace before running
HF_MODEL  = 'mistralai/Mistral-7B-Instruct-v0.2'
HF_URL    = f'https://api-inference.huggingface.co/models/{HF_MODEL}'

FALLBACK_MODEL = 'google/flan-t5-base'  # smaller, faster fallback

def generate_answer(prompt: str, use_fallback: bool = False) -> str:
    model_url = (
        f'https://api-inference.huggingface.co/models/{FALLBACK_MODEL}'
        if use_fallback else HF_URL
    )
    headers  = {'Authorization': f'Bearer {HF_TOKEN}'}
    payload  = {
        'inputs': prompt,
        'parameters': {'max_new_tokens': 400, 'temperature': 0.3},
    }
    try:
        resp = requests.post(model_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list):
            raw = result[0].get('generated_text', '')
            # Return only the answer portion after the prompt
            return raw.split('Answer')[-1].strip()
        return str(result)
    except Exception as e:
        if not use_fallback:
            print(f'Primary model failed ({e}), switching to fallback...')
            return generate_answer(prompt, use_fallback=True)
        return f'[Generation error: {e}]'

# ── Cell 4: Full RAG pipeline ─────────────────────────────────────────────────
def nyayasetu(query: str, k: int = 5) -> dict:
    """End-to-end: query → retrieve → generate → return answer + sources."""
    chunks = retrieve(query, k=k)
    prompt = build_prompt(query, chunks)
    answer = generate_answer(prompt)
    return {'answer': answer, 'sources': chunks}

# ── Cell 5: Smoke test ────────────────────────────────────────────────────────
test_queries = [
    'What are my rights if my landlord increases rent without notice?',
    'मेरे मकान मालिक ने बिना नोटिस के किराया बढ़ा दिया, मैं क्या कर सकता हूँ?',
    'ನನ್ನ MGNREGA ವೇತನ 3 ತಿಂಗಳಿಂದ ಬಂದಿಲ್ಲ',
]

for q in test_queries:
    print(f'\nQ: {q}')
    result = nyayasetu(q, k=3)
    print(f'A: {result["answer"][:300]}')
    print(f'Sources: {[s["act"] for s in result["sources"]]}')
