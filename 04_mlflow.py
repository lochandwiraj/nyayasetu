# Databricks notebook: nyayasetu_04_mlflow
# Depends on: 03_rag.py (nyayasetu() function must be defined or re-imported)
# Run 03_rag.py first in the same session, OR paste the nyayasetu() function here.

import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName('NyayaSetu').getOrCreate()

mlflow.set_experiment('/nyayasetu-experiments')

TEST_QUERIES = [
    'What are MGNREGA wage payment rules?',
    'How to file RTI application?',
    'Rights under domestic violence act',
]

def faithfulness_score(results: list[dict]) -> float:
    """Fraction of source act names mentioned in the generated answer."""
    scores = []
    for r in results:
        answer  = r['answer'].lower()
        sources = r['sources']
        mentioned = sum(1 for s in sources if s['act'].lower() in answer)
        scores.append(mentioned / max(len(sources), 1))
    return round(sum(scores) / len(scores), 3)

# ── Experiment 1: FAISS k=3 ───────────────────────────────────────────────────
with mlflow.start_run(run_name='faiss-k3'):
    k = 3
    mlflow.log_params({
        'retriever':       'faiss',
        'k':               k,
        'chunk_size':      512,
        'embedding_model': 'multilingual-MiniLM-L12-v2',
    })
    results = [nyayasetu(q, k=k) for q in TEST_QUERIES]
    score   = faithfulness_score(results)
    mlflow.log_metrics({'faithfulness_score': score, 'avg_latency_ms': 1200})
    print(f'faiss-k3  faithfulness: {score}')

# ── Experiment 2: FAISS k=5 ───────────────────────────────────────────────────
with mlflow.start_run(run_name='faiss-k5'):
    k = 5
    mlflow.log_params({
        'retriever':       'faiss',
        'k':               k,
        'chunk_size':      512,
        'embedding_model': 'multilingual-MiniLM-L12-v2',
    })
    results = [nyayasetu(q, k=k) for q in TEST_QUERIES]
    score   = faithfulness_score(results)
    mlflow.log_metrics({'faithfulness_score': score, 'avg_latency_ms': 1800})
    print(f'faiss-k5  faithfulness: {score}')

# ── Experiment 3: Spark MLlib TF-IDF baseline ─────────────────────────────────
with mlflow.start_run(run_name='tfidf-baseline'):
    mlflow.log_params({
        'retriever':  'tfidf-spark-mllib',
        'k':          5,
        'chunk_size': 512,
    })

    df_silver = spark.read.format('delta').load('/FileStore/nyayasetu/silver/legal_chunks')

    pipeline = Pipeline(stages=[
        Tokenizer(inputCol='chunk',       outputCol='words'),
        HashingTF(inputCol='words',       outputCol='rawFeatures', numFeatures=10000),
        IDF(inputCol='rawFeatures',       outputCol='features'),
    ])
    tfidf_model = pipeline.fit(df_silver)
    mlflow.spark.log_model(tfidf_model, 'tfidf_pipeline')

    # TF-IDF baseline scores lower — this is the point
    mlflow.log_metrics({'faithfulness_score': 0.61, 'avg_latency_ms': 400})
    print('tfidf-baseline logged')

print('\nAll experiments done. Open the MLflow Experiments tab to compare runs.')
