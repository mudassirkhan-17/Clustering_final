import os
import requests
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
import igraph as ig
import leidenalg
import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta


load_dotenv() 

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── CONFIG ──────────────────────────────────────────────────────────
DAYS = 1   # change to 3 for 3 days, 7 for a week, etc.
# ────────────────────────────────────────────────────────────────────

since = datetime.now() - timedelta(days=DAYS)

conn = psycopg2.connect(os.getenv("OFFICEFIELD_LOCAL_SUPABASE_STRINGS"))

query = """
    SELECT
        symbol, title, content AS text, publish_date, site, sentiment, sentiment_score AS sentimentscore
    FROM master_news
    WHERE publish_date > %s
    ORDER BY publish_date ASC;
"""

df = pd.read_sql_query(query, conn, params=(since,))
conn.close()

df["publish_date"] = pd.to_datetime(df["publish_date"])

print(f"✅ Fetched {len(df):,} articles from last {DAYS} day(s)")
print(f"   Date range: {df['publish_date'].min()} → {df['publish_date'].max()}")
print(f"   Unique symbols: {df['symbol'].nunique()}")

# 2) basic cleaning function
def clean(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = " ".join(s.split())  # collapse whitespace
    return s

def embed_batch(texts):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in resp.data]


df["title_clean"] = df["title"].map(clean)
df["text_clean"]  = df["text"].map(clean)

# 3) trim (optional but recommended)
MAX_CHARS = 150000000
df["text_clean"] = df["text_clean"].str.slice(0, MAX_CHARS)

# 4) final string you will embed
# Handle missing symbol (NaN → empty or "unknown")
df["symbol_clean"] = df["symbol"].fillna("").astype(str).str.strip()

df["embed_text"] = (
    df["symbol_clean"].replace("", "") + "\n" +
    df["title_clean"] + "\n" +
    df["text_clean"]
)

BATCH = 128
embs = []

for i in range(0, len(df), BATCH):
    batch = df["embed_text"].iloc[i:i+BATCH].tolist()
    batch_embs = embed_batch(batch)
    embs.extend(batch_embs)

df["embedding"] = embs

# optional: convert to numpy float32 for speed/memory
df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype=np.float32))

# Build normalized embedding matrix
embeddings = np.vstack(df['embedding'].values).astype(np.float32)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_norm = embeddings / norms

# ============================================================
# Build similarity graph
# ============================================================
EDGE_THRESHOLD = 0.75   # only connect articles above this cosine similarity
                         # tune: lower = more edges = bigger clusters
                         # tune: higher = fewer edges = more clusters

sim_matrix = embeddings_norm @ embeddings_norm.T
n = len(df)

edges = []
weights = []

for i in range(n):
    for j in range(i + 1, n):
        sim = float(sim_matrix[i, j])
        if sim >= EDGE_THRESHOLD:
            edges.append((i, j))
            weights.append(sim)

# Build igraph graph
G = ig.Graph(n=n, edges=edges)
G.es['weight'] = weights

# ============================================================
# Apply Leiden
# ============================================================
partition = leidenalg.find_partition(
    G,
    leidenalg.ModularityVertexPartition,
    weights='weight',
    seed=42
)

df['cluster_leiden'] = partition.membership

vc_leiden    = df['cluster_leiden'].value_counts()
singletons   = (vc_leiden == 1).sum()
meaningful   = vc_leiden[vc_leiden > 1]

print("=== Leiden Clustering ===")
print(f"\n✓ Leiden complete")
print(f"Total clusters            : {vc_leiden.shape[0]}")
print(f"  Singleton clusters (=1) : {singletons}  ← noise")
print(f"  Meaningful clusters (>1): {len(meaningful)}")
print(f"Articles in singletons    : {singletons} / {len(df)} ({100*singletons/len(df):.1f}%)")
print(f"Articles in meaningful    : {meaningful.sum()} / {len(df)} ({100*meaningful.sum()/len(df):.1f}%)")
print(f"Avg size (meaningful only): {meaningful.mean():.1f}")
print(f"Max cluster size          : {meaningful.max() if len(meaningful) else 0}")
print(f"\nCluster distribution (all):")
print(vc_leiden)