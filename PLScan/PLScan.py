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


load_dotenv() 

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "dataset2.csv"))

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

print("=== PLSCAN Clustering ===")
print(f"\n✓ PLSCAN complete")
print(f"Clusters found: {df[df['cluster_plscan'] != -1]['cluster_plscan'].nunique()}")
print(f"Noise points:   {(df['cluster_plscan'] == -1).sum()}")
print(f"\nCluster distribution:")
print(df['cluster_plscan'].value_counts())