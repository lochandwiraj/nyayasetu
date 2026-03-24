# Databricks notebook: nyayasetu_02_silver_gold
# Depends on: 01_bronze.py having written the Bronze Delta table

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode, trim, length, monotonically_increasing_id
from pyspark.sql.types import ArrayType, StringType, FloatType
from sentence_transformers import SentenceTransformer
import pandas as pd

spark = SparkSession.builder.appName('NyayaSetu').getOrCreate()

# ── Cell 1: Silver layer — clean and chunk ────────────────────────────────────
CHUNK_SIZE = 512   # words per chunk
OVERLAP    = 64    # word overlap between chunks

def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(' '.join(words[i:i + CHUNK_SIZE]))
        i += CHUNK_SIZE - OVERLAP
    return chunks

chunk_udf = udf(chunk_text, ArrayType(StringType()))

df_bronze = spark.read.format('delta').load('/FileStore/nyayasetu/bronze/legal_raw')

df_silver = (
    df_bronze
    .withColumn('chunks', chunk_udf(col('text')))
    .withColumn('chunk',  explode(col('chunks')))
    .withColumn('chunk',  trim(col('chunk')))
    .filter(length(col('chunk')) > 100)
    .select('act', 'page', 'chunk')
    .withColumn('chunk_id', monotonically_increasing_id())
)

df_silver.write.format('delta').mode('overwrite') \
    .save('/FileStore/nyayasetu/silver/legal_chunks')

print(f'Silver table: {df_silver.count()} chunks')
df_silver.show(5, truncate=100)

# ── Cell 2: Gold layer — multilingual embeddings ──────────────────────────────
# paraphrase-multilingual-MiniLM-L12-v2 supports Hindi, Tamil, Kannada, Telugu etc.
EMBED_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

from pyspark.sql.functions import pandas_udf

@pandas_udf(ArrayType(FloatType()))
def embed_chunks(texts: pd.Series) -> pd.Series:
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts.tolist(), show_progress_bar=False)
    return pd.Series([e.tolist() for e in embeddings])

df_silver = spark.read.format('delta').load('/FileStore/nyayasetu/silver/legal_chunks')

df_gold = df_silver.withColumn('embedding', embed_chunks(col('chunk')))

df_gold.write.format('delta').mode('overwrite') \
    .save('/FileStore/nyayasetu/gold/legal_embeddings')

print('Gold table written with embeddings')
df_gold.select('act', 'chunk_id', 'chunk').show(3, truncate=80)
