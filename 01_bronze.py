# Databricks notebook: nyayasetu_01_bronze
# Run cells top to bottom. Shift+Enter to execute each cell.

# ── Cell 1: Install dependencies ──────────────────────────────────────────────
# %pip install PyMuPDF langchain sentence-transformers faiss-cpu
# Uncomment the line above and run it first in Databricks, then restart the kernel.

# ── Cell 2: Extract text from legal PDFs ──────────────────────────────────────
import fitz  # PyMuPDF

# Adjust paths if your uploads landed elsewhere in DBFS
PDF_PATHS = {
    'ipc':     '/FileStore/tables/ipc.pdf',
    'rti':     '/FileStore/tables/rti.pdf',
    'mgnrega': '/FileStore/tables/mgnrega.pdf',
    'dv_act':  '/FileStore/tables/dv_act.pdf',
}

def extract_text(path: str, act_name: str) -> list[dict]:
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if len(text.strip()) > 50:  # skip blank / header-only pages
            pages.append({
                'act':  act_name,
                'page': i + 1,
                'text': text.strip(),
            })
    return pages

all_pages = []
for act_name, path in PDF_PATHS.items():
    pages = extract_text(path, act_name)
    all_pages.extend(pages)
    print(f'{act_name}: {len(pages)} pages extracted')

print(f'\nTotal pages: {len(all_pages)}')

# ── Cell 3: Write Bronze Delta table ──────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

spark = SparkSession.builder.appName('NyayaSetu').getOrCreate()

schema = StructType([
    StructField('act',  StringType(),  False),
    StructField('page', IntegerType(), False),
    StructField('text', StringType(),  False),
])

df_bronze = spark.createDataFrame(all_pages, schema=schema)

df_bronze.write.format('delta').mode('overwrite') \
    .save('/FileStore/nyayasetu/bronze/legal_raw')

print(f'Bronze table written: {df_bronze.count()} rows')
df_bronze.show(5, truncate=80)
