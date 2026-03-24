import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = r"nyayasetu"
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "IPC":                    os.path.join(DATA_DIR, "ipc.txt"),
    "RTI Act":                os.path.join(DATA_DIR, "rti.txt"),
    "MGNREGA":                os.path.join(DATA_DIR, "mgnrega.txt"),
    "Domestic Violence Act":  os.path.join(DATA_DIR, "dv_act.txt"),
}

CHUNK_SIZE    = 512   # words
OVERLAP       = 64    # words
MIN_CHARS     = 100
MODEL_NAME    = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Helpers ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start : start + size])
        if len(chunk) >= MIN_CHARS:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def main():
    print("=" * 60)
    print("  NyayaSetu — Data Setup")
    print("=" * 60)

    all_chunks = []
    all_acts   = []

    # ── Step 1: Read & chunk ─────────────────────────────────────────────────
    for act_name, filepath in FILES.items():
        print(f"\n[1/3] Reading: {act_name}")
        if not os.path.exists(filepath):
            print(f"      ⚠  File not found: {filepath} — skipping")
            continue
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = chunk_text(text)
        print(f"      ✓  {len(chunks)} chunks extracted")
        all_chunks.extend(chunks)
        all_acts.extend([act_name] * len(chunks))

    if not all_chunks:
        print("\n❌  No chunks found. Check that the txt files exist.")
        return

    print(f"\n      Total chunks: {len(all_chunks)}")

    # ── Step 2: Embeddings ───────────────────────────────────────────────────
    print(f"\n[2/3] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("      Generating embeddings (this may take a few minutes)…")
    embeddings = model.encode(
        all_chunks,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype("float32")
    print(f"      ✓  Embeddings shape: {embeddings.shape}")

    # ── Step 3: FAISS index ──────────────────────────────────────────────────
    print("\n[3/3] Building FAISS index…")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"      ✓  Index contains {index.ntotal} vectors")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)
    pd.DataFrame({"chunk": all_chunks, "act": all_acts}).to_csv(
        os.path.join(OUT_DIR, "chunks.csv"), index=False
    )
    faiss.write_index(index, os.path.join(OUT_DIR, "faiss.index"))

    print("\n✅  Saved:")
    print(f"    • {os.path.join(OUT_DIR, 'embeddings.npy')}")
    print(f"    • {os.path.join(OUT_DIR, 'chunks.csv')}")
    print(f"    • {os.path.join(OUT_DIR, 'faiss.index')}")
    print("\n🎉  Setup complete! Run:  streamlit run app.py")


if __name__ == "__main__":
    main()


