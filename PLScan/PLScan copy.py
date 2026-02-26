import os
import requests
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
from fast_plscan import PLSCAN
import numpy as np
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

# Apply PLSCAN — only 1 parameter (unlike HDBSCAN which needs 2)
clusterer = PLSCAN(
    min_samples=2  # tune: higher = smoother, fewer clusters
                    # lower = more granular, more clusters
).fit(embeddings_norm)

df['cluster_plscan'] = clusterer.labels_
# Note: -1 = noise points (same as HDBSCAN)

vc_plscan    = df['cluster_plscan'].value_counts()
noise_count  = (df['cluster_plscan'] == -1).sum()
vc_real      = vc_plscan[vc_plscan.index != -1]          # exclude noise label
singletons   = (vc_real == 1).sum()
meaningful   = vc_real[vc_real > 1]

print("=== PLSCAN Clustering ===")
print(f"\n✓ PLSCAN complete")
print(f"Total clusters (excl noise): {vc_real.shape[0]}")
print(f"  Singleton clusters (=1)  : {singletons}  ← noise")
print(f"  Meaningful clusters (>1) : {len(meaningful)}")
print(f"Noise points (-1)          : {noise_count} / {len(df)} ({100*noise_count/len(df):.1f}%)")
print(f"Articles in singletons     : {singletons} / {len(df)} ({100*singletons/len(df):.1f}%)")
print(f"Articles in meaningful     : {meaningful.sum()} / {len(df)} ({100*meaningful.sum()/len(df):.1f}%)")
print(f"Avg size (meaningful only) : {meaningful.mean():.1f}")
print(f"Max cluster size           : {meaningful.max() if len(meaningful) else 0}")
print(f"\nCluster distribution (all):")
print(df['cluster_plscan'].value_counts())