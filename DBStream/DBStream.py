import os
import requests
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
from river import cluster

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

# Initialize DBSTREAM (paper-like parameters)
dbstream = cluster.DBSTREAM(
    clustering_threshold=0.8,    # Paper uses various; start with 0.8
    fading_factor=0.01,          # Slow decay (paper: λ for 10-day window)
    cleanup_interval=10,         # Clean old clusters every 10 points
    intersection_factor=0.3,     # Shared density
    minimum_weight=1.0           # Min weight
)

# Process articles ONE BY ONE (streaming simulation)
print("\n=== DBSTREAM Clustering ===")
dbstream_clusters = []

for idx, row in df.iterrows():
    # Get combined embedding (text + NER)
    emb = row["embedding"]
    
    # Convert to dict for river
    x = {i: float(v) for i, v in enumerate(emb)}
    
    # Learn (update micro-clusters)
    dbstream.learn_one(x)
    
    # Predict (assign cluster)
    cluster_id = dbstream.predict_one(x)
    dbstream_clusters.append(cluster_id)
    
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{len(df)} articles...")

# Add to df
df["cluster_dbstream"] = dbstream_clusters

# Results
print(f"\n✓ DBSTREAM complete")
print(f"Clusters found: {df['cluster_dbstream'].nunique()}")
print(f"Cluster distribution:")
print(df['cluster_dbstream'].value_counts())
