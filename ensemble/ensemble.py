"""
Phase 1 — Cross-Algorithm Ensemble Clustering
Strategy  : Bipartite graph (Agglo ↔ Leiden) + Connected Components → Cream Clusters
Similarity : Szymkiewicz-Simpson overlap coefficient
Cream core : INTERSECTION of Agglo-side ∩ Leiden-side articles (double-confirmed only)
Soft edges : union − intersection → passed to Phase 2 as uncertified
"""

import os
import psycopg2
import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
from collections import defaultdict
from dotenv import load_dotenv
from datetime import datetime, timedelta
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
DAYS              = 1
OVERLAP_THRESHOLD = 0.95   # Szymkiewicz-Simpson threshold for cross-algo edge
LEIDEN_EDGE_SIM   = 0.75   # Cosine similarity threshold for Leiden graph edges
AGGLO_DIST        = 1.5    # Ward distance threshold for Agglomerative
# ══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────
# STEP 1 · FETCH
# ─────────────────────────────────────────────────────────────
since = datetime.now() - timedelta(days=DAYS)
conn  = psycopg2.connect(os.getenv("OFFICEFIELD_LOCAL_SUPABASE_STRINGS"))

query = """
    SELECT symbol, title, content AS text,
           publish_date, site, sentiment,
           sentiment_score AS sentimentscore
    FROM   master_news
    WHERE  publish_date > %s
    ORDER  BY publish_date ASC;
"""
df = pd.read_sql_query(query, conn, params=(since,))
conn.close()

df = df.reset_index(drop=True)   # clean 0-based index throughout
df["publish_date"] = pd.to_datetime(df["publish_date"])
n = len(df)

print(f"✅ Fetched {n:,} articles from last {DAYS} day(s)")
print(f"   Date range    : {df['publish_date'].min()} → {df['publish_date'].max()}")
print(f"   Unique tickers: {df['symbol'].nunique()}")


# ─────────────────────────────────────────────────────────────
# STEP 2 · CLEAN + EMBED
# ─────────────────────────────────────────────────────────────
def clean(s):
    if pd.isna(s):
        return ""
    return " ".join(str(s).split())

def embed_batch(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]

df["title_clean"]  = df["title"].map(clean)
df["text_clean"]   = df["text"].map(clean).str.slice(0, 15_000_000)
df["symbol_clean"] = df["symbol"].fillna("").astype(str).str.strip()
df["embed_text"]   = df["symbol_clean"] + "\n" + df["title_clean"] + "\n" + df["text_clean"]

print("\n⏳ Generating embeddings...")
BATCH, embs = 128, []
for i in range(0, n, BATCH):
    embs.extend(embed_batch(df["embed_text"].iloc[i : i + BATCH].tolist()))
    if (i // BATCH + 1) % 5 == 0:
        print(f"   {min(i + BATCH, n)}/{n} embedded...")

df["embedding"] = [np.array(e, dtype=np.float32) for e in embs]

embeddings = np.vstack(df["embedding"].values)
norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
emb_norm   = embeddings / np.where(norms == 0, 1.0, norms)
print(f"✅ Embeddings ready — shape {embeddings.shape}")


# ─────────────────────────────────────────────────────────────
# STEP 3A · AGGLOMERATIVE CLUSTERING
# ─────────────────────────────────────────────────────────────
print("\n[1/2] Agglomerative clustering...")
agglo = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=AGGLO_DIST,
    linkage="ward",
    metric="euclidean",
)
df["cluster_agglo"] = agglo.fit_predict(embeddings)

vc_agglo   = df["cluster_agglo"].value_counts()
sing_agglo = (vc_agglo == 1).sum()
mean_agglo = vc_agglo[vc_agglo > 1]
print(f"   Total clusters            : {vc_agglo.shape[0]}")
print(f"   Singleton clusters (=1)   : {sing_agglo}  ← noise")
print(f"   Meaningful clusters (>1)  : {len(mean_agglo)}")
print(f"   Articles in meaningful    : {mean_agglo.sum()} / {n} ({100*mean_agglo.sum()/n:.1f}%)")
print(f"   Avg size (meaningful)     : {mean_agglo.mean():.1f}  |  Max: {mean_agglo.max()}")


# ─────────────────────────────────────────────────────────────
# STEP 3B · LEIDEN CLUSTERING
# ─────────────────────────────────────────────────────────────
print("\n[2/2] Leiden clustering...")

# Build article similarity graph
sim  = emb_norm @ emb_norm.T
rows, cols = np.where(np.triu(sim >= LEIDEN_EDGE_SIM, k=1))
G_leiden   = ig.Graph(n=n, edges=list(zip(rows.tolist(), cols.tolist())))
G_leiden.es["weight"] = sim[rows, cols].tolist()

partition = leidenalg.find_partition(
    G_leiden, leidenalg.ModularityVertexPartition, weights="weight", seed=42
)
df["cluster_leiden"] = partition.membership

vc_leiden   = df["cluster_leiden"].value_counts()
sing_leiden = (vc_leiden == 1).sum()
mean_leiden = vc_leiden[vc_leiden > 1]
print(f"   Total clusters            : {vc_leiden.shape[0]}")
print(f"   Singleton clusters (=1)   : {sing_leiden}  ← noise")
print(f"   Meaningful clusters (>1)  : {len(mean_leiden)}")
print(f"   Articles in meaningful    : {mean_leiden.sum()} / {n} ({100*mean_leiden.sum()/n:.1f}%)")
print(f"   Avg size (meaningful)     : {mean_leiden.mean():.1f}  |  Max: {mean_leiden.max()}")


# ─────────────────────────────────────────────────────────────
# STEP 4 · STRIP SINGLETONS — only keep meaningful clusters
# ─────────────────────────────────────────────────────────────
valid_agglo  = set(mean_agglo.index)    # Agglo cluster IDs with ≥ 2 articles
valid_leiden = set(mean_leiden.index)   # Leiden cluster IDs with ≥ 2 articles


# ─────────────────────────────────────────────────────────────
# STEP 5 · BUILD INVERTED INDEX + INTERSECTION COUNTS  (O(n))
# ─────────────────────────────────────────────────────────────
print("\n⏳ Building inverted index...")

agglo_to_articles  = defaultdict(set)   # agglo_id  → {article row indices}
leiden_to_articles = defaultdict(set)   # leiden_id → {article row indices}
intersection_count = defaultdict(int)   # (agglo_id, leiden_id) → shared article count

for i in range(n):
    a = df.at[i, "cluster_agglo"]
    l = df.at[i, "cluster_leiden"]

    in_valid_agglo  = a in valid_agglo
    in_valid_leiden = l in valid_leiden

    if in_valid_agglo:
        agglo_to_articles[a].add(i)
    if in_valid_leiden:
        leiden_to_articles[l].add(i)
    if in_valid_agglo and in_valid_leiden:
        intersection_count[(a, l)] += 1

agglo_sizes  = {a: len(arts) for a, arts in agglo_to_articles.items()}
leiden_sizes = {l: len(arts) for l, arts in leiden_to_articles.items()}

print(f"   Valid Agglo clusters      : {len(agglo_to_articles)}")
print(f"   Valid Leiden clusters     : {len(leiden_to_articles)}")
print(f"   Cross-pairs sharing ≥1 article: {len(intersection_count):,}")


# ─────────────────────────────────────────────────────────────
# STEP 6 · COMPUTE SZYMKIEWICZ-SIMPSON OVERLAP + BUILD GRAPH
# ─────────────────────────────────────────────────────────────
print(f"\n⏳ Computing overlap (threshold ≥ {OVERLAP_THRESHOLD})...")

G = nx.Graph()

# Add nodes — prefix to distinguish algorithms
for a in valid_agglo:
    G.add_node(f"A_{a}", algo="agglo")
for l in valid_leiden:
    G.add_node(f"L_{l}", algo="leiden")

edge_count = 0
for (a, l), count in intersection_count.items():
    smaller  = min(agglo_sizes[a], leiden_sizes[l])
    overlap  = count / smaller
    if overlap >= OVERLAP_THRESHOLD:
        G.add_edge(f"A_{a}", f"L_{l}", weight=overlap)
        edge_count += 1

print(f"   Edges drawn (≥ {OVERLAP_THRESHOLD} overlap) : {edge_count}")


# ─────────────────────────────────────────────────────────────
# STEP 7 · CONNECTED COMPONENTS → CREAM CLUSTERS
# ─────────────────────────────────────────────────────────────
print("\n⏳ Finding connected components...")

components = list(nx.connected_components(G))

cream_clusters        = []    # list of sets — each set = article row indices
cream_meta            = []    # metadata per cream cluster
certified_article_ids = set()

for comp in components:
    # Skip isolated nodes (no cross-algorithm agreement)
    if len(comp) == 1:
        continue

    has_agglo  = any(node.startswith("A_") for node in comp)
    has_leiden = any(node.startswith("L_") for node in comp)

    # Only create a cream cluster if BOTH algorithms are represented
    if not (has_agglo and has_leiden):
        continue

    # Union all articles from each algorithm side separately
    agglo_articles  = set()
    leiden_articles = set()
    agglo_nodes, leiden_nodes = [], []

    for node in comp:
        if node.startswith("A_"):
            a_id = int(node[2:])
            agglo_articles |= agglo_to_articles[a_id]
            agglo_nodes.append(a_id)
        else:
            l_id = int(node[2:])
            leiden_articles |= leiden_to_articles[l_id]
            leiden_nodes.append(l_id)

    # INTERSECTION: only articles confirmed by BOTH algorithms
    article_set   = agglo_articles & leiden_articles
    soft_articles = (agglo_articles | leiden_articles) - article_set  # single-algo only

    # Skip if intersection is empty (degenerate component)
    if not article_set:
        continue

    cream_clusters.append(article_set)
    cream_meta.append({
        "agglo_nodes"  : agglo_nodes,
        "leiden_nodes" : leiden_nodes,
        "size"         : len(article_set),
        "union_size"   : len(agglo_articles | leiden_articles),
        "soft_count"   : len(soft_articles),
    })
    certified_article_ids |= article_set
    # soft_articles fall through to uncertified → handled by Phase 2

uncertified_ids = set(range(n)) - certified_article_ids


# ─────────────────────────────────────────────────────────────
# STEP 8 · ASSIGN FINAL LABELS TO DATAFRAME
# ─────────────────────────────────────────────────────────────
df["cluster_cream"] = -1   # -1 = uncertified

for cream_id, article_set in enumerate(cream_clusters):
    for idx in article_set:
        df.at[idx, "cluster_cream"] = cream_id


# ─────────────────────────────────────────────────────────────
# STEP 9 · RESULTS
# ─────────────────────────────────────────────────────────────
cream_sizes = [m["size"] for m in cream_meta]
vc_cream    = pd.Series(cream_sizes)

print("\n" + "═" * 58)
print("  PHASE 1 — CREAM CLUSTER RESULTS")
print("═" * 58)
print(f"  Total articles              : {n:,}")
print(f"  Certified articles          : {len(certified_article_ids):,} ({100*len(certified_article_ids)/n:.1f}%)")
print(f"  Uncertified (second pass)   : {len(uncertified_ids):,} ({100*len(uncertified_ids)/n:.1f}%)")
print(f"")
print(f"  Cream clusters formed       : {len(cream_clusters)}")
print(f"  Avg cream cluster size      : {vc_cream.mean():.1f}")
print(f"  Max cream cluster size      : {vc_cream.max()}")
print(f"  Min cream cluster size      : {vc_cream.min()}")
print(f"")
print(f"  Cream cluster size distribution:")
size_dist = vc_cream.value_counts().sort_index()
for size, count in size_dist.items():
    bar = "█" * min(count, 40)
    print(f"    {size:>4} articles : {count:>4} clusters  {bar}")

total_soft = sum(m["soft_count"] for m in cream_meta)
print(f"  Soft articles (single-algo, → Phase 2): {total_soft}")

print("\n  Top 15 largest cream clusters:")
top15 = sorted(cream_meta, key=lambda x: x["size"], reverse=True)[:15]
for rank, meta in enumerate(top15, 1):
    print(f"    #{rank:>2}  core={meta['size']:>4}  union={meta['union_size']:>4}  "
          f"soft={meta['soft_count']:>3}  "
          f"A={len(meta['agglo_nodes'])} L={len(meta['leiden_nodes'])}")

print("\n── Individual algorithm recap ────────────────────────────")
print(f"  Agglo  : {len(mean_agglo):>4} meaningful clusters  "
      f"(of {vc_agglo.shape[0]} total, {sing_agglo} singletons)")
print(f"  Leiden : {len(mean_leiden):>4} meaningful clusters  "
      f"(of {vc_leiden.shape[0]} total, {sing_leiden} singletons)")
print("──────────────────────────────────────────────────────────")
print(f"\n  cluster_cream = -1  → uncertified (awaits second pass)")
print(f"  cluster_cream ≥  0  → cream cluster ID (double-voted)")
print("═" * 58)
