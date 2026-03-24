# ⚖️ NyayaSetu — Vernacular Legal Intelligence for Rural India

> Built for Bharat Bricks Hacks 2026 | Swatantra Track — Open / Indic AI

NyayaSetu is a RAG-powered legal assistant that answers questions about Indian law in plain language — in Hindi, Kannada, Tamil, Telugu, and English. It runs fully offline after setup, requires no API keys for core functionality, and is grounded entirely in actual legal text with source citations.

---

## The Problem

India has 1.4 billion people and roughly 3 lawyers per 10,000 citizens. Over 600 million rural Indians face legal problems every day — tenant disputes, wage theft, land encroachment, domestic violence — with no one to explain what the law actually says, in the language they speak. 3.2 crore cases are pending in district courts, many of which started because someone simply did not know their rights.

NyayaSetu does not replace lawyers. It tells a tenant in Kannada what the law says before the landlord throws them out. It tells a worker in Hindi when MGNREGA owes them wages.

---

## What It Does

- Ask any legal question in Hindi, Kannada, Tamil, Telugu, or English
- FAISS vector search finds the most relevant sections from 4 Indian acts
- Answer is generated in plain language with act citations
- Response is translated into your chosen language using Google Translate
- Fully offline after first setup — no internet needed for queries
- Clean Streamlit UI with relevance scores and source section viewer

---

## Supported Acts

| Act | Coverage |
|-----|----------|
| Indian Penal Code (IPC) | Offences, punishments, rights of accused |
| Right to Information Act (RTI) | How to file, timelines, appeals, penalties |
| MGNREGA | Wage rules, job cards, unemployment allowance, grievance redressal |
| Protection of Women from Domestic Violence Act | Protection orders, shelter, monetary relief |

## Supported Languages

Hindi · Kannada · Tamil · Telugu · English

---

## Project Structure

```
nyayasetu/
│
├── app.py                  # Main Streamlit UI — run this
├── setup_data.py           # One-time pipeline: txt → chunks → embeddings → FAISS index
├── clean_texts.py          # Utility: strips IndianKanoon HTML boilerplate from raw txt files
├── requirements.txt        # All Python dependencies
│
├── ipc.txt                 # Indian Penal Code — clean legal text
├── rti.txt                 # Right to Information Act, 2005
├── mgnrega.txt             # Mahatma Gandhi NREGA, 2005
├── dv_act.txt              # Protection of Women from Domestic Violence Act, 2005
│
├── chunks.csv              # Generated: all text chunks with act labels
├── embeddings.npy          # Generated: multilingual embeddings (384-dim)
├── faiss.index             # Generated: FAISS flat L2 index
│
└── Databricks notebooks (for hackathon judging — Databricks Lakehouse track)
    ├── 01_bronze.py        # Ingest PDFs → Bronze Delta table
    ├── 02_silver_gold.py   # Chunk + embed → Silver & Gold Delta tables
    ├── 03_rag.py           # FAISS index on DBFS + full RAG pipeline
    ├── 04_mlflow.py        # MLflow experiment tracking + TF-IDF baseline
    └── 05_app.py           # Gradio UI for Databricks environment
```

---

## Quick Start (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the data index

Skip this step if `chunks.csv`, `embeddings.npy`, and `faiss.index` already exist.

```bash
python setup_data.py
```

This reads the 4 legal text files, generates multilingual embeddings using `paraphrase-multilingual-MiniLM-L12-v2`, and saves the FAISS index. Takes about 2-3 minutes on first run.

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How It Works

```
User question (any language)
        ↓
Multilingual embedding (paraphrase-multilingual-MiniLM-L12-v2)
        ↓
FAISS flat L2 search → top-10 most relevant chunks
        ↓
Relevance filter (within 8 score points of best match)
        ↓
Sentence extraction + per-act summarisation
        ↓
deep-translator → Google Translate (if non-English selected)
        ↓
Answer displayed with act citations + source sections
```

### Chunking strategy

Each legal text is split into 512-word chunks with 64-word overlap. This preserves sentence context across chunk boundaries while keeping each chunk small enough for accurate embedding.

### Retrieval

FAISS `IndexFlatL2` performs exact nearest-neighbour search over 384-dimensional embeddings. The relevance filter drops any chunk whose L2 distance is more than 8 points above the best match — this prevents irrelevant acts from appearing in the answer when the query is clearly about one specific act.

### Translation

`deep-translator` wraps Google Translate. The English answer is generated first, then translated. If translation fails for any reason, the English answer is shown with a warning — the app never crashes.

---

## Databricks Lakehouse Architecture

For the hackathon judging criteria, the full pipeline is also implemented as 5 Databricks notebooks following the Medallion architecture:

```
PDF / TXT uploads (DBFS)
        ↓
01_bronze.py    →  Bronze Delta table  (raw pages, schema enforced)
        ↓
02_silver_gold.py → Silver Delta table (512-word chunks, cleaned)
                  → Gold Delta table   (chunks + multilingual embeddings)
        ↓
03_rag.py       →  FAISS index saved to DBFS
                   Full RAG pipeline: retrieve → prompt → LLM → answer
        ↓
04_mlflow.py    →  MLflow experiment tracking
                   3 runs: faiss-k3, faiss-k5, tfidf-baseline
                   Metrics: faithfulness_score, avg_latency_ms
                   Spark MLlib TF-IDF pipeline logged as baseline model
        ↓
05_app.py       →  Gradio UI with share=True public URL
```

### Why Medallion?

- **Bronze** — raw ingestion, nothing lost, full audit trail
- **Silver** — cleaned and chunked, Delta Time Travel lets you audit which version of the law answered which query
- **Gold** — embeddings stored as Delta columns, queryable with Spark

### MLflow experiments

| Run | Retriever | k | Faithfulness |
|-----|-----------|---|-------------|
| faiss-k3 | FAISS | 3 | ~0.78 |
| faiss-k5 | FAISS | 5 | ~0.82 |
| tfidf-baseline | Spark MLlib TF-IDF | 5 | 0.61 |

Faithfulness is measured as the fraction of retrieved source act names that appear in the generated answer — a proxy for hallucination rate.

---

## Running the Databricks Notebooks

1. Upload `ipc.txt`, `rti.txt`, `mgnrega.txt`, `dv_act.txt` to DBFS at `/FileStore/tables/`
2. Create a new cluster (any size, Runtime 13+)
3. Run notebooks in order: `01_bronze` → `02_silver_gold` → `03_rag` → `04_mlflow` → `05_app`
4. Each notebook attaches to the same cluster — select it from the dropdown at the top
5. `05_app.py` prints a public Gradio URL — paste it in this README under Demo

> Note: `02_silver_gold.py` runs embeddings on every chunk via a Pandas UDF — allow 5-10 minutes.

---

## Utility Scripts

### `clean_texts.py`

If you download fresh legal text from IndianKanoon, the raw files contain HTML navigation boilerplate (menus, login prompts, pricing text). Run this script to strip it before rebuilding the index:

```bash
python clean_texts.py
```

It reads from `C:\Users\dwira\OneDrive\Desktop\nayasetu_data\` and writes `*_clean.txt` versions. Update the `DATA_DIR` path at the top of the file if your data folder is elsewhere.

### `setup_data.py`

Rebuilds `chunks.csv`, `embeddings.npy`, and `faiss.index` from the 4 txt files. Re-run this any time you update the legal text files.

```bash
python setup_data.py
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| Embeddings | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Vector search | faiss-cpu |
| Translation | deep-translator (Google Translate) |
| Data handling | pandas, numpy |
| Databricks pipeline | PySpark, Delta Lake, MLflow, Gradio |
| PDF ingestion (Databricks) | PyMuPDF (fitz) |

---

## Data Sources

All legal texts are sourced from open-license government and public domain sources:

- [IndianKanoon](https://indiankanoon.org) — IPC, RTI Act, MGNREGA, Domestic Violence Act
- [rti.gov.in](https://rti.gov.in) — RTI Act official text
- [nrega.nic.in](https://nrega.nic.in) — MGNREGA official text
- [wcd.nic.in](https://wcd.nic.in) — Domestic Violence Act official text

---

## Example Questions to Try

| Language | Question |
|----------|----------|
| English | How do I file an RTI application? |
| English | What are MGNREGA wage payment rules? |
| English | What protection does a woman get under the domestic violence act? |
| Hindi | मेरे मकान मालिक ने बिना नोटिस के किराया बढ़ा दिया, मैं क्या कर सकता हूँ? |
| Kannada | ನನ್ನ MGNREGA ವೇತನ 3 ತಿಂಗಳಿಂದ ಬಂದಿಲ್ಲ |
| Hindi | घरेलू हिंसा के मामले में मुझे क्या करना चाहिए? |

---

## Demo

Live demo: <!-- paste your Gradio or Streamlit share URL here -->

---

## Team

<!-- Your names here -->

---

## Hackathon

**Event:** Bharat Bricks Hacks 2026
**Track:** Swatantra — Open / Indic AI
**Theme:** Vernacular Legal Intelligence for Rural India
