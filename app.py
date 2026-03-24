import os
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NyayaSetu — Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0e1117; color: #e0e0e0; }
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1f2e, #252b3b);
    border: 1px solid #2d3550; border-radius: 12px; padding: 16px;
  }
  .example-btn button {
    background: #1a1f2e !important; border: 1px solid #3d4f7c !important;
    color: #a0b4d6 !important; border-radius: 8px !important;
    font-size: 0.82rem !important; text-align: left !important;
  }
  .example-btn button:hover { border-color: #6c8ebf !important; color: #ffffff !important; }
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin: 3px 4px;
  }
  .main-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #6c8ebf, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0;
  }
  .subtitle { color: #8899aa; font-size: 1.05rem; margin-top: 4px; margin-bottom: 24px; }
  .footer {
    text-align: center; color: #4a5568; font-size: 0.8rem;
    padding: 24px 0 8px; border-top: 1px solid #1e2535; margin-top: 40px;
  }
  hr { border-color: #1e2535; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

ACT_COLORS = {
    "IPC":                   "#e74c3c",
    "RTI Act":               "#3498db",
    "MGNREGA":               "#2ecc71",
    "Domestic Violence Act": "#9b59b6",
}

EXAMPLE_QUESTIONS = [
    "What are my rights if my landlord increases rent?",
    "How do I file an RTI application?",
    "What are MGNREGA wage payment rules?",
    "What protection does a woman get under domestic violence act?",
]

LANGUAGES  = ["English", "Hindi", "Kannada", "Tamil", "Telugu"]
LANG_CODES = {"English": "en", "Hindi": "hi", "Kannada": "kn", "Tamil": "ta", "Telugu": "te"}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_data():
    paths = [os.path.join(BASE_DIR, f) for f in ("chunks.csv", "faiss.index", "embeddings.npy")]
    if not all(os.path.exists(p) for p in paths):
        return None, None, None
    df    = pd.read_csv(paths[0])
    index = faiss.read_index(paths[1])
    emb   = np.load(paths[2])
    return df, index, emb

# ── Translation ───────────────────────────────────────────────────────────────
def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source="en", target=target_lang).translate(text)
        return result if result else text
    except Exception as e:
        st.warning(f"Translation failed ({e}), showing English.")
        return text

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, model, df, index, k: int = 10):
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(df):
            results.append({
                "chunk": df.iloc[idx]["chunk"],
                "act":   df.iloc[idx]["act"],
                "score": float(dist),
            })
    return results

# ── Answer generation ─────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

def _extract_sentences(chunk: str, query_words: set, n: int = 3) -> list:
    raw   = _clean(chunk)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", raw)
    sents = [s.strip() for s in sents if len(s.strip()) > 40]
    ranked = sorted(sents, key=lambda s: sum(1 for w in query_words if w in s.lower()), reverse=True)
    return ranked[:n]

def generate_answer(query: str, chunks: list, lang_code: str = "en") -> str:
    if not chunks:
        return "Could not find relevant information in the provided legal texts."

    # Only use chunks within 8 FAISS score points of the best — filters irrelevant acts
    best  = chunks[0]["score"]
    relevant = [c for c in chunks if c["score"] <= best + 8.0]

    query_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())

    # Build per-act summaries
    act_map: dict[str, list] = {}
    for item in relevant:
        act = item["act"]
        if act not in act_map:
            act_map[act] = []
        act_map[act].extend(_extract_sentences(item["chunk"], query_words, n=3))

    if not act_map:
        return "Could not find relevant information in the provided legal texts."

    parts = []
    for act, sents in act_map.items():
        # deduplicate
        seen, unique = set(), []
        for s in sents:
            if s[:60] not in seen:
                seen.add(s[:60])
                unique.append(s)
        parts.append(f"**According to {act}:**\n{' '.join(unique[:3])}")

    english_answer = "\n\n".join(parts)

    # Translate if needed
    if lang_code != "en":
        return translate_text(english_answer, lang_code)
    return english_answer

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ NyayaSetu")
    st.markdown(
        "NyayaSetu bridges the gap between rural citizens and the Indian legal system "
        "by providing plain-language answers grounded in actual legal text."
    )
    st.divider()
    st.markdown("**📚 Supported Acts**")
    for act, color in ACT_COLORS.items():
        st.markdown(
            f'<span class="badge" style="background:{color}22;color:{color};'
            f'border:1px solid {color}55">{act}</span>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown("**🌐 Supported Languages**")
    st.markdown("Hindi · English · Kannada · Tamil · Telugu")
    st.divider()
    st.markdown("**🔍 How it works**")
    st.markdown(
        "1. **Ask** your legal question\n"
        "2. **Search** — FAISS finds relevant sections\n"
        "3. **Answer** — plain-language response with citations"
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚖️ NyayaSetu</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Vernacular Legal Intelligence for Rural India</p>', unsafe_allow_html=True)

df, index, embeddings = load_data()
total_chunks = len(df) if df is not None else 0

col1, col2, col3 = st.columns(3)
col1.metric("📖 Legal Acts Loaded", "4")
col2.metric("🗂️ Total Chunks", total_chunks if total_chunks else "— Run setup first")
col3.metric("🤖 Model", "Multilingual MiniLM")

st.divider()

if df is None:
    st.error(
        "**Data not found.** Run setup first:\n\n```\npython setup_data.py\n```"
    )
    st.stop()

with st.spinner("Loading language model…"):
    model = load_model()

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("### 💬 Ask a Legal Question")

if "question" not in st.session_state:
    st.session_state.question = ""

question = st.text_area(
    "Type your question in any supported language:",
    value=st.session_state.question,
    height=100,
    placeholder="e.g. What are my rights under the domestic violence act?",
    key="question_input",
)

lang_col, btn_col = st.columns([2, 1])
with lang_col:
    language = st.selectbox("🌐 Response Language", LANGUAGES)
with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    ask_clicked = st.button("Ask NyayaSetu ⚖️", type="primary", use_container_width=True)

st.markdown("**Try an example:**")
ex_cols = st.columns(4)
for i, (col, eq) in enumerate(zip(ex_cols, EXAMPLE_QUESTIONS)):
    with col:
        st.markdown('<div class="example-btn">', unsafe_allow_html=True)
        if st.button(eq, key=f"ex_{i}", use_container_width=True):
            st.session_state.question = eq
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
final_question = question.strip() or st.session_state.question.strip()

if ask_clicked:
    if not final_question:
        st.warning("Please enter a question before clicking Ask.")
    else:
        lang_code = LANG_CODES.get(language, "en")

        with st.spinner("🔍 Searching legal texts…"):
            results = retrieve(final_question, model, df, index, k=10)

        if not results:
            st.warning("No relevant sections found. Try rephrasing your question.")
        else:
            spinner_msg = f"✍️ Generating answer and translating to {language}…" if lang_code != "en" else "✍️ Generating answer…"
            with st.spinner(spinner_msg):
                answer = generate_answer(final_question, results, lang_code)

            st.divider()
            st.markdown("### 📋 Answer")
            if lang_code != "en":
                st.caption(f"🌐 Translated to {language}  ·  Source: Indian Legal Acts")
            st.success(answer)

            # Source badges
            st.markdown("**Sources referenced:**")
            seen, badge_html = set(), ""
            for r in results:
                act = r["act"]
                if act not in seen:
                    seen.add(act)
                    color = ACT_COLORS.get(act, "#888")
                    badge_html += (
                        f'<span class="badge" style="background:{color}22;'
                        f'color:{color};border:1px solid {color}66">{act}</span>'
                    )
            st.markdown(badge_html, unsafe_allow_html=True)

            st.divider()

            # Relevance chart
            st.markdown("### 📊 Relevance Scores")
            chart_df = pd.DataFrame({
                "Chunk":     [f"{r['act']} #{i+1}" for i, r in enumerate(results)],
                "Relevance": [1 / (1 + r["score"]) for r in results],
            })
            st.bar_chart(chart_df.set_index("Chunk"), color="#6c8ebf")

            # Top chunks
            st.markdown("### 🔎 Retrieved Sections")
            for i, r in enumerate(results[:3]):
                color = ACT_COLORS.get(r["act"], "#888")
                with st.expander(f"[{r['act']}] Chunk {i+1}  —  score: {r['score']:.2f}", expanded=(i == 0)):
                    st.markdown(
                        f'<div style="border-left:3px solid {color};padding-left:12px;color:#ccc">'
                        f'{r["chunk"][:800]}{"…" if len(r["chunk"]) > 800 else ""}</div>',
                        unsafe_allow_html=True,
                    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "Built on Databricks Lakehouse | Delta Lake + FAISS + MLflow<br>"
    "Bharat Bricks Hacks 2026 | Swatantra Track"
    "</div>",
    unsafe_allow_html=True,
)
