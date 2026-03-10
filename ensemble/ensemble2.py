"""
Phase 1 — Cross-Algorithm Ensemble Clustering
Strategy  : Tripartite graph (Agglo ↔ Leiden ↔ PLSCAN) + Connected Components → Cream Clusters
Similarity : Szymkiewicz-Simpson overlap + Jaccard coefficient
Consensus  : 3-of-3 algorithms must agree per article (min cluster size 3)
  TRIPLE_CREAM → confirmed by ALL 3 algorithms (only tier)
Soft edges : single-algo or 2-algo articles → passed to Phase 2 as uncertified
"""

import os
import hashlib
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
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
from fast_plscan import PLSCAN
from hdbscan import HDBSCAN
from joblib import Parallel, delayed
import tiktoken

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
DAYS                   = 2
OVERLAP_THRESHOLD      = 0.95  # Szymkiewicz-Simpson threshold for cross-algo edge (max possible = 1.0)
JACCARD_THRESHOLD      = 0.2  # Jaccard threshold — kills imbalanced pairs (tiny ∩ big)
LEIDEN_EDGE_SIM        = 0.70  # Cosine similarity threshold for Leiden graph edges
AGGLO_DIST             = 1.2   # Ward distance threshold for Agglomerative
CREAM_CONSOLIDATION_SIM = 0.78 # Phase 2A: centroid cosine sim to merge cream clusters
ASSIGN_THRESHOLD        = 0.70 # Phase 2B: min cosine sim to nearest centroid for assignment
PLSCAN_MIN_SAMPLES_P1   = 2    # Phase 1 third voter: PLSCAN min_samples on full dataset
ORPHAN_MIN_CLUSTER_SIZE = 3    # Phase 2B Step 2: HDBSCAN min articles to form a cluster
ORPHAN_MIN_SAMPLES      = 2    # Phase 2B Step 2: HDBSCAN core point density (lower = more clusters)
PROV_MAX_CLUSTER_SIZE   = 50   # Phase 2B Step 2: reject provisional clusters larger than this → ISOLATED
CREAM_MIN_INTRA_SIM     = 0.60 # Post-hoc quality gate: reject cream clusters below this mean intra-sim
# ══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────
# STEP 1 · FETCH
# ─────────────────────────────────────────────────────────────
since = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=DAYS)
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

print(f"✅ Fetched {len(df):,} articles from last {DAYS} day(s)")
print(f"   Date range    : {df['publish_date'].min()} → {df['publish_date'].max()}")
print(f"   Unique tickers: {df['symbol'].nunique()}")

# ── Deduplication (before clustering) ────────────────────────
# Key: first 80 chars of title (catches same article stored with
# truncated vs full title) + site. Keeps the longest title version.
_before_dedup = len(df)
df["_title_key"] = df["title"].fillna("").str.strip().str[:200]
df = (
    df.sort_values("title", key=lambda s: s.str.len(), ascending=False)
      .drop_duplicates(subset=["_title_key", "site"], keep="first")
      .drop(columns=["_title_key"])
      .reset_index(drop=True)
)
_deduped = _before_dedup - len(df)
print(f"\n🧹 Deduplication (title[:80] + site):")
print(f"   Before : {_before_dedup:,}")
print(f"   Removed: {_deduped:,}  ({100*_deduped/_before_dedup:.1f}% duplicates)")
print(f"   After  : {len(df):,}  ← used for clustering")

n = len(df)


# ─────────────────────────────────────────────────────────────
# STEP 2 · CLEAN + EMBED
# ─────────────────────────────────────────────────────────────
def clean(s):
    if pd.isna(s):
        return ""
    return " ".join(str(s).split())

MAX_TOKENS = 7000  # hard limit 8192; large buffer for tokenizer version drift
_enc       = tiktoken.get_encoding("cl100k_base")

def truncate_tokens(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text[:25_000]
    try:
        tokens = _enc.encode(text, disallowed_special=())
        if len(tokens) > MAX_TOKENS:
            text = _enc.decode(tokens[:MAX_TOKENS])
    except Exception:
        text = text[:4_000]   # 4k chars ≈ 1-4k tokens — always safe
    return text

_emb_tokens = 0                     # cumulative embedding tokens for cost tracking
_tok_lock   = threading.Lock()      # protects _emb_tokens across parallel threads

def embed_batch(texts):
    global _emb_tokens
    safe = [truncate_tokens(t) for t in texts]
    try:
        resp = client.embeddings.create(model="text-embedding-3-small", input=safe)
        if resp.usage:
            with _tok_lock:
                _emb_tokens += resp.usage.total_tokens
        return [d.embedding for d in resp.data]
    except Exception:
        # If batch fails, fall back to one-at-a-time to isolate the problem text
        results = []
        for t in safe:
            t = t[:8_000]  # hard char fallback
            resp = client.embeddings.create(model="text-embedding-3-small", input=[t])
            if resp.usage:
                with _tok_lock:
                    _emb_tokens += resp.usage.total_tokens
            results.append(resp.data[0].embedding)
        return results

# ── Rolling embedding cache helpers ──────────────────────────
# Cache persists between daily runs. Each entry stores the embedding
# vector and its publish_date so stale entries can be evicted.
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "embedding_cache.pkl")

def _article_hash(title: str, publish_date) -> str:
    """Stable unique key per article: md5(title|publish_date)."""
    return hashlib.md5(f"{title}|{publish_date}".encode()).hexdigest()

def _load_cache() -> dict:
    """Load rolling cache from disk; return {} if missing or corrupt."""
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH, "rb") as _f:
            return pickle.load(_f)
    except Exception:
        return {}

def _save_cache(cache: dict) -> None:
    """Atomic save: write to .tmp then os.replace — crash-safe."""
    _tmp = _CACHE_PATH + ".tmp"
    with open(_tmp, "wb") as _f:
        pickle.dump(cache, _f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(_tmp, _CACHE_PATH)

def _embed_miss_batch(idx_batch: list) -> tuple:
    """Embed cache-miss rows; returns (row_indices, vectors)."""
    texts = [df.at[i, "embed_text"] for i in idx_batch]
    return idx_batch, embed_batch(texts)

# ── Clean text ───────────────────────────────────────────────
df["title_clean"]  = df["title"].map(clean)
df["text_clean"]   = df["text"].map(clean).str.slice(0, 25_000)
df["symbol_clean"] = df["symbol"].fillna("").astype(str).str.strip()
df["embed_text"]   = df["symbol_clean"] + "\n" + df["title_clean"] + "\n" + df["title_clean"] + "\n" + df["text_clean"]
df["embed_text"]   = df["embed_text"].map(truncate_tokens)

# ── Rolling cache: load → evict stale → serve hits → embed misses ──
_emb_cache    = _load_cache()
_cache_before = len(_emb_cache)

# Evict entries outside the DAYS window (same cutoff as the SQL query)
_emb_cache = {
    h: v for h, v in _emb_cache.items()
    if pd.to_datetime(v["publish_date"]) >= since
}
_evicted = _cache_before - len(_emb_cache)

print(f"\n📦 Embedding cache: {_cache_before:,} loaded  |  "
      f"{_evicted:,} evicted (>{DAYS}d old)  |  {len(_emb_cache):,} live")

# Hash every article in the current fetch
df["_hash"] = [
    _article_hash(str(row.title), row.publish_date)
    for row in df[["title", "publish_date"]].itertuples(index=False)
]

_hit_idx  = [i for i in range(n) if df.at[i, "_hash"] in _emb_cache]
_miss_idx = [i for i in range(n) if df.at[i, "_hash"] not in _emb_cache]

print(f"   Cache hits  (skip API)    : {len(_hit_idx):,}")
print(f"   Cache misses (embed now)  : {len(_miss_idx):,}")

# Pre-allocate — fill hits from cache, misses from API
_all_vecs: list = [None] * n

for _i in _hit_idx:
    _all_vecs[_i] = _emb_cache[df.at[_i, "_hash"]]["embedding"]

if _miss_idx:
    _BATCH        = 100
    _N_JOBS       = -1
    _miss_batches = [_miss_idx[s:s + _BATCH] for s in range(0, len(_miss_idx), _BATCH)]

    print(f"\n⏳ Embedding {len(_miss_idx):,} new articles via API "
          f"({len(_miss_batches)} batches, parallel)...")

    _miss_results = Parallel(n_jobs=_N_JOBS, backend="threading", verbose=0)(
        delayed(_embed_miss_batch)(batch) for batch in _miss_batches
    )

    for _idx_batch, _vecs in _miss_results:
        for _i, _vec in zip(_idx_batch, _vecs):
            _arr = np.array(_vec, dtype=np.float32)
            _all_vecs[_i] = _arr
            _emb_cache[df.at[_i, "_hash"]] = {
                "embedding"   : _arr,
                "publish_date": df.at[_i, "publish_date"],
            }

    print(f"✅ Embedded {len(_miss_idx):,} new articles  ({_emb_tokens:,} tokens)")
else:
    print(f"✅ All {n:,} articles served from cache  (0 API tokens used)")

# Save updated cache atomically
_save_cache(_emb_cache)
print(f"   Cache saved: {len(_emb_cache):,} entries → {os.path.basename(_CACHE_PATH)}")

df["embedding"] = _all_vecs

embeddings = np.vstack(df["embedding"].values)
norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
emb_norm   = embeddings / np.where(norms == 0, 1.0, norms)
print(f"✅ Embeddings ready — shape {embeddings.shape}")


# ─────────────────────────────────────────────────────────────
# STEP 3 · RUN ALL THREE CLUSTERING ALGORITHMS IN PARALLEL
# Each is a C/C++ extension that releases the GIL → threading
# gives real CPU concurrency with no data-copy overhead.
# Results are written to df only after all three finish.
# ─────────────────────────────────────────────────────────────

def _run_agglo() -> np.ndarray:
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=AGGLO_DIST,
        linkage="ward",
        metric="euclidean",
    )
    return model.fit_predict(embeddings)

def _run_leiden() -> list:
    sim_l = emb_norm @ emb_norm.T
    rows_l, cols_l = np.where(np.triu(sim_l >= LEIDEN_EDGE_SIM, k=1))
    G_l = ig.Graph(n=n, edges=list(zip(rows_l.tolist(), cols_l.tolist())))
    G_l.es["weight"] = sim_l[rows_l, cols_l].tolist()
    part = leidenalg.find_partition(
        G_l, leidenalg.ModularityVertexPartition, weights="weight", seed=42
    )
    return part.membership

def _run_plscan() -> np.ndarray:
    return PLSCAN(min_samples=PLSCAN_MIN_SAMPLES_P1).fit(emb_norm).labels_

print("\n⏳ Running Agglo + Leiden + PLSCAN in parallel (3 threads)...")
with ThreadPoolExecutor(max_workers=3) as _pool:
    _fut_agglo  = _pool.submit(_run_agglo)
    _fut_leiden = _pool.submit(_run_leiden)
    _fut_plscan = _pool.submit(_run_plscan)
    # .result() blocks until each thread finishes; exceptions are re-raised here
    df["cluster_agglo"]  = _fut_agglo.result()
    df["cluster_leiden"] = _fut_leiden.result()
    df["cluster_plscan"] = _fut_plscan.result()

print("✅ All three algorithms finished.\n")

# ── [1/3] Agglo stats ────────────────────────────────────────
print("[1/3] Agglomerative clustering stats:")
vc_agglo   = df["cluster_agglo"].value_counts()
sing_agglo = (vc_agglo == 1).sum()
mean_agglo = vc_agglo[vc_agglo > 1]
print(f"   Total clusters            : {vc_agglo.shape[0]}")
print(f"   Singleton clusters (=1)   : {sing_agglo}  ← noise")
print(f"   Meaningful clusters (>1)  : {len(mean_agglo)}")
print(f"   Articles in meaningful    : {mean_agglo.sum()} / {n} ({100*mean_agglo.sum()/n:.1f}%)")
print(f"   Avg size (meaningful)     : {mean_agglo.mean():.1f}  |  Max: {mean_agglo.max()}")

# ── [2/3] Leiden stats ───────────────────────────────────────
print("\n[2/3] Leiden clustering stats:")
vc_leiden   = df["cluster_leiden"].value_counts()
sing_leiden = (vc_leiden == 1).sum()
mean_leiden = vc_leiden[vc_leiden > 1]
print(f"   Total clusters            : {vc_leiden.shape[0]}")
print(f"   Singleton clusters (=1)   : {sing_leiden}  ← noise")
print(f"   Meaningful clusters (>1)  : {len(mean_leiden)}")
print(f"   Articles in meaningful    : {mean_leiden.sum()} / {n} ({100*mean_leiden.sum()/n:.1f}%)")
print(f"   Avg size (meaningful)     : {mean_leiden.mean():.1f}  |  Max: {mean_leiden.max()}")

# ── [3/3] PLSCAN stats ───────────────────────────────────────
print("\n[3/3] PLSCAN clustering stats (Phase 1 voter):")
vc_plscan   = df["cluster_plscan"].value_counts()
sing_plscan = int((vc_plscan.index == -1).sum())
mean_plscan = vc_plscan[(vc_plscan.index != -1) & (vc_plscan > 1)]
print(f"   Total clusters            : {(vc_plscan.index != -1).sum()}")
print(f"   Noise points (-1)         : {int(vc_plscan.get(-1, 0))}  ← noise")
print(f"   Meaningful clusters (>1)  : {len(mean_plscan)}")
print(f"   Articles in meaningful    : {mean_plscan.sum()} / {n} ({100*mean_plscan.sum()/n:.1f}%)")
print(f"   Avg size (meaningful)     : {mean_plscan.mean():.1f}  |  Max: {mean_plscan.max()}")

# ── PLSCAN failure detection ──────────────────────────────────
# PLSCAN is considered failed if it produces fewer than 20 meaningful
# clusters (e.g. collapses into a few giant blobs on sparse datasets).
# Fallback: drop to Agglo+Leiden 2/3 consensus for Phase 1.
_PLSCAN_MIN_CLUSTERS = 20
_plscan_ok = len(mean_plscan) >= _PLSCAN_MIN_CLUSTERS
if not _plscan_ok:
    print(f"\n⚠️  PLSCAN FALLBACK: only {len(mean_plscan)} meaningful cluster(s) "
          f"(need ≥ {_PLSCAN_MIN_CLUSTERS}) — switching to Agglo+Leiden 2/3 consensus")


# ─────────────────────────────────────────────────────────────
# STEP 4 · STRIP SINGLETONS — only keep meaningful clusters
# ─────────────────────────────────────────────────────────────
valid_agglo  = set(mean_agglo.index)    # Agglo cluster IDs with ≥ 2 articles
valid_leiden = set(mean_leiden.index)   # Leiden cluster IDs with ≥ 2 articles
# When PLSCAN failed, use empty set → no P_ nodes added → no A↔P / L↔P edges
valid_plscan = set(mean_plscan.index) if _plscan_ok else set()


# ─────────────────────────────────────────────────────────────
# STEP 5 · BUILD INVERTED INDEX + INTERSECTION COUNTS  (O(n))
# Three algorithm pairs: Agglo↔Leiden, Agglo↔PLSCAN, Leiden↔PLSCAN
# ─────────────────────────────────────────────────────────────
print("\n⏳ Building tripartite inverted index...")

agglo_to_articles  = defaultdict(set)   # agglo_id  → {article row indices}
leiden_to_articles = defaultdict(set)   # leiden_id → {article row indices}
plscan_to_articles = defaultdict(set)   # plscan_id → {article row indices}

inter_al = defaultdict(int)   # (agglo_id, leiden_id) → shared article count
inter_ap = defaultdict(int)   # (agglo_id, plscan_id) → shared article count
inter_lp = defaultdict(int)   # (leiden_id, plscan_id) → shared article count

for i in range(n):
    a = df.at[i, "cluster_agglo"]
    l = df.at[i, "cluster_leiden"]
    p = df.at[i, "cluster_plscan"]

    in_a = a in valid_agglo
    in_l = l in valid_leiden
    in_p = p in valid_plscan

    if in_a:
        agglo_to_articles[a].add(i)
    if in_l:
        leiden_to_articles[l].add(i)
    if in_p:
        plscan_to_articles[p].add(i)

    if in_a and in_l:
        inter_al[(a, l)] += 1
    if in_a and in_p:
        inter_ap[(a, p)] += 1
    if in_l and in_p:
        inter_lp[(l, p)] += 1

agglo_sizes  = {a: len(arts) for a, arts in agglo_to_articles.items()}
leiden_sizes = {l: len(arts) for l, arts in leiden_to_articles.items()}
plscan_sizes = {p: len(arts) for p, arts in plscan_to_articles.items()}

print(f"   Valid Agglo clusters      : {len(agglo_to_articles)}")
print(f"   Valid Leiden clusters     : {len(leiden_to_articles)}")
print(f"   Valid PLSCAN clusters     : {len(plscan_to_articles)}")
print(f"   A↔L cross-pairs ≥1 article: {len(inter_al):,}")
print(f"   A↔P cross-pairs ≥1 article: {len(inter_ap):,}")
print(f"   L↔P cross-pairs ≥1 article: {len(inter_lp):,}")


# ─────────────────────────────────────────────────────────────
# STEP 6 · COMPUTE SZYMKIEWICZ-SIMPSON OVERLAP + BUILD TRIPARTITE GRAPH
# Edges drawn for all three algorithm pairs: A↔L, A↔P, L↔P
# ─────────────────────────────────────────────────────────────
print(f"\n⏳ Computing tripartite overlap (threshold ≥ {OVERLAP_THRESHOLD})...")

G = nx.Graph()

# Add nodes — prefix to distinguish algorithms
for a in valid_agglo:
    G.add_node(f"A_{a}", algo="agglo")
for l in valid_leiden:
    G.add_node(f"L_{l}", algo="leiden")
for p in valid_plscan:
    G.add_node(f"P_{p}", algo="plscan")

def _add_edges(pairs, sizes_x, sizes_y, prefix_x, prefix_y):
    count_added = 0
    for (x, y), cnt in pairs.items():
        smaller = min(sizes_x[x], sizes_y[y])
        simpson = cnt / smaller
        jaccard = cnt / (sizes_x[x] + sizes_y[y] - cnt)
        if simpson >= OVERLAP_THRESHOLD and jaccard >= JACCARD_THRESHOLD:
            G.add_edge(f"{prefix_x}{x}", f"{prefix_y}{y}", weight=simpson)
            count_added += 1
    return count_added

edges_al = _add_edges(inter_al, agglo_sizes, leiden_sizes, "A_", "L_")
edges_ap = _add_edges(inter_ap, agglo_sizes, plscan_sizes, "A_", "P_")
edges_lp = _add_edges(inter_lp, leiden_sizes, plscan_sizes, "L_", "P_")

print(f"   A↔L edges (Simpson ≥ {OVERLAP_THRESHOLD} AND Jaccard ≥ {JACCARD_THRESHOLD}): {edges_al}")
print(f"   A↔P edges                                                          : {edges_ap}")
print(f"   L↔P edges                                                          : {edges_lp}")
print(f"   Total edges                                                        : {edges_al + edges_ap + edges_lp}")


# ─────────────────────────────────────────────────────────────
# STEP 7 · CONNECTED COMPONENTS → CREAM CLUSTERS
# Normal  : 3/3 consensus (Agglo + Leiden + PLSCAN)
# Fallback: 2/3 consensus (Agglo + Leiden only, PLSCAN failed)
# ─────────────────────────────────────────────────────────────
_consensus_mode = "3/3 (tripartite)" if _plscan_ok else "2/3 (Agglo+Leiden fallback)"
print(f"\n⏳ Finding connected components ({_consensus_mode})...")

components = list(nx.connected_components(G))

cream_clusters        = []    # list of sets — each set = article row indices
cream_meta            = []    # metadata per cream cluster
certified_article_ids = set()

for comp in components:
    # Skip isolated nodes (no cross-algorithm agreement)
    if len(comp) == 1:
        continue

    algos_present = set()
    for node in comp:
        if node.startswith("A_"):   algos_present.add("agglo")
        elif node.startswith("L_"): algos_present.add("leiden")
        elif node.startswith("P_"): algos_present.add("plscan")

    # Require 3/3 normally; 2/3 (Agglo+Leiden only) when PLSCAN failed
    _min_algos = 3 if _plscan_ok else 2
    if len(algos_present) < _min_algos:
        continue

    agglo_articles  = set()
    leiden_articles = set()
    plscan_articles = set()
    agglo_nodes, leiden_nodes, plscan_nodes = [], [], []

    for node in comp:
        if node.startswith("A_"):
            a_id = int(node[2:])
            agglo_articles |= agglo_to_articles[a_id]
            agglo_nodes.append(a_id)
        elif node.startswith("L_"):
            l_id = int(node[2:])
            leiden_articles |= leiden_to_articles[l_id]
            leiden_nodes.append(l_id)
        elif node.startswith("P_"):
            p_id = int(node[2:])
            plscan_articles |= plscan_to_articles[p_id]
            plscan_nodes.append(p_id)

    # Intersection: all 3 when PLSCAN ok; Agglo∩Leiden only when fallback
    if _plscan_ok:
        article_set = agglo_articles & leiden_articles & plscan_articles
    else:
        article_set = agglo_articles & leiden_articles

    all_articles  = agglo_articles | leiden_articles | plscan_articles
    soft_articles = all_articles - article_set   # 1 or 2-algo only → uncertified

    # Skip if consensus set is too small to be credible
    if len(article_set) < 3:
        continue

    cream_clusters.append(article_set)
    cream_meta.append({
        "agglo_nodes"  : agglo_nodes,
        "leiden_nodes" : leiden_nodes,
        "plscan_nodes" : plscan_nodes,
        "size"         : len(article_set),
        "union_size"   : len(all_articles),
        "soft_count"   : len(soft_articles),
    })
    certified_article_ids |= article_set
    # soft_articles fall through to uncertified → handled by Phase 2

# ── Post-hoc quality gate ─────────────────────────────────
# Compute mean pairwise intra-sim for each cream cluster.
# Clusters below CREAM_MIN_INTRA_SIM are rejected: their articles
# are removed from certified_article_ids and fall into uncertified,
# where Phase 2B gives them a second chance via nearest-centroid.
_kept_clusters   = []
_kept_meta       = []
_rejected_count  = 0
_rejected_arts   = 0

for _cs, _cm in zip(cream_clusters, cream_meta):
    _idxs  = list(_cs)
    _vecs  = emb_norm[_idxs]
    _smat  = _vecs @ _vecs.T
    _upper = _smat[np.triu_indices(len(_idxs), k=1)]
    _msim  = float(_upper.mean()) if len(_upper) > 0 else 0.0

    if _msim >= CREAM_MIN_INTRA_SIM:
        _kept_clusters.append(_cs)
        _kept_meta.append(_cm)
    else:
        certified_article_ids -= _cs   # send back to uncertified pool
        _rejected_count += 1
        _rejected_arts  += len(_cs)

cream_clusters = _kept_clusters
cream_meta     = _kept_meta

print(f"\n  Post-hoc quality gate (intra-sim ≥ {CREAM_MIN_INTRA_SIM}):")
print(f"    Clusters kept    : {len(cream_clusters)}")
print(f"    Clusters rejected: {_rejected_count}  ({_rejected_arts} articles → uncertified)")

uncertified_ids = set(range(n)) - certified_article_ids


# ─────────────────────────────────────────────────────────────
# STEP 8 · ASSIGN FINAL LABELS TO DATAFRAME
# ─────────────────────────────────────────────────────────────
df["cluster_cream"] = -1   # -1 = uncertified
df["vote_count"]    = 0    # 3 for all certified articles (3/3 consensus)

for cream_id, article_set in enumerate(cream_clusters):
    for idx in article_set:
        df.at[idx, "cluster_cream"] = cream_id
        df.at[idx, "vote_count"]    = 3


# ─────────────────────────────────────────────────────────────
# STEP 9 · RESULTS
# ─────────────────────────────────────────────────────────────
cream_sizes = [m["size"] for m in cream_meta]
vc_cream    = pd.Series(cream_sizes)

print("\n" + "═" * 58)
print(f"  PHASE 1 — CREAM CLUSTER RESULTS  ({_consensus_mode})")
print("═" * 58)
print(f"  Total articles              : {n:,}")
print(f"  Certified (TRIPLE_CREAM)    : {len(certified_article_ids):,} ({100*len(certified_article_ids)/n:.1f}%)  ← all 3 algos agree")
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
          f"soft={meta['soft_count']:>3}")

print("\n── Individual algorithm recap ────────────────────────────")
print(f"  Agglo  : {len(mean_agglo):>4} meaningful clusters  "
      f"(of {vc_agglo.shape[0]} total, {sing_agglo} singletons)")
print(f"  Leiden : {len(mean_leiden):>4} meaningful clusters  "
      f"(of {vc_leiden.shape[0]} total, {sing_leiden} singletons)")
print(f"  PLSCAN : {len(mean_plscan):>4} meaningful clusters  "
      f"(of {(vc_plscan.index != -1).sum()} total, "
      f"{int(vc_plscan.get(-1, 0))} noise pts)")
print("──────────────────────────────────────────────────────────")
print(f"\n  cluster_cream = -1  → uncertified (awaits second pass)")
print(f"  cluster_cream ≥  0  → cream cluster ID (TRIPLE_CREAM, all 3 agree)")
print("═" * 58)


# ─────────────────────────────────────────────────────────────
# PHASE 2A · CREAM CLUSTER CONSOLIDATION
# Goal : merge cream clusters whose centroids are very similar
#        (two Phase-1 clusters that are really about the same topic)
# Method: frozen centroid per cluster → cosine sim matrix →
#         similarity graph → connected components → merge
# ─────────────────────────────────────────────────────────────
num_cream = len(cream_clusters)
print(f"\n{'═' * 58}")
print("  PHASE 2A — CREAM CLUSTER CONSOLIDATION")
print(f"{'═' * 58}")
print(f"⏳ Computing frozen centroids for {num_cream} cream clusters...")

# ── 2A-1 : frozen centroid (mean of raw embeddings, L2-normalised) ──
cream_centroids = np.zeros((num_cream, embeddings.shape[1]), dtype=np.float32)
for i, article_set in enumerate(cream_clusters):
    vecs     = embeddings[list(article_set)]   # (k, dim)
    centroid = vecs.mean(axis=0)
    norm     = np.linalg.norm(centroid)
    cream_centroids[i] = centroid / norm if norm > 0 else centroid

print(f"   Frozen centroids ready — shape {cream_centroids.shape}")

# ── 2A-2 : pairwise centroid cosine similarity (vectorised) ──
cent_sim = cream_centroids @ cream_centroids.T   # (num_cream, num_cream)

# ── 2A-3 : build cream-to-cream similarity graph ──
rows_2a, cols_2a = np.where(np.triu(cent_sim >= CREAM_CONSOLIDATION_SIM, k=1))
G2A = nx.Graph()
G2A.add_nodes_from(range(num_cream))
for r, c in zip(rows_2a.tolist(), cols_2a.tolist()):
    G2A.add_edge(int(r), int(c), weight=float(cent_sim[r, c]))

print(f"   Edges drawn (centroid sim ≥ {CREAM_CONSOLIDATION_SIM}): {G2A.number_of_edges()}")

# ── 2A-4 : connected components → determine which clusters merge ──
components_2a = list(nx.connected_components(G2A))

# ── 2A-5 : build consolidated clusters ──
consolidated_clusters = []   # list[set[int]]  — article row indices
consolidated_meta     = []   # metadata per consolidated cluster

for comp in components_2a:
    merged_articles = set()
    for idx in comp:
        merged_articles |= cream_clusters[idx]
    consolidated_clusters.append(merged_articles)
    consolidated_meta.append({
        "size"         : len(merged_articles),
        "merged_count" : len(comp),             # how many Phase-1 clusters joined
    })

# ── 2A-6 : re-stamp df["cluster_cream"] with new consolidated IDs ──
df["cluster_cream"] = -1   # reset; uncertified_ids remain -1
for new_id, article_set in enumerate(consolidated_clusters):
    for idx in article_set:
        df.at[idx, "cluster_cream"] = new_id

# sanity: certified set must not shrink (articles only move between clusters)
assert df[df["cluster_cream"] >= 0].shape[0] == len(certified_article_ids), \
    "Consolidation lost certified articles — check logic!"

# ── 2A-7 : results ──
cons_sizes   = [m["size"]         for m in consolidated_meta]
merge_counts = [m["merged_count"] for m in consolidated_meta]
vc_cons      = pd.Series(cons_sizes)

merged_components  = sum(1 for mc in merge_counts if mc > 1)
clusters_absorbed  = sum(mc - 1 for mc in merge_counts if mc > 1)

print(f"\n  Before 2A : {num_cream:>5} cream clusters  (avg {vc_cream.mean():.1f})")
print(f"  After  2A : {len(consolidated_clusters):>5} cream clusters  (avg {vc_cons.mean():.1f})")
print(f"  Components that merged  : {merged_components}  "
      f"(absorbed {clusters_absorbed} clusters)")
print(f"  Reduction               : {num_cream - len(consolidated_clusters)} clusters removed")
print(f"")
print(f"  Avg size   : {vc_cons.mean():.1f}")
print(f"  Max size   : {vc_cons.max()}")
print(f"  Min size   : {vc_cons.min()}")
print(f"")
print(f"  Consolidated size distribution:")
size_dist_2a = vc_cons.value_counts().sort_index()
for size, count in size_dist_2a.items():
    bar = "█" * min(count, 40)
    print(f"    {size:>4} articles : {count:>4} clusters  {bar}")

print(f"\n  Top 15 largest consolidated clusters:")
top15_cons = sorted(enumerate(consolidated_meta), key=lambda x: x[1]["size"], reverse=True)[:15]
for rank, (cid, meta) in enumerate(top15_cons, 1):
    print(f"    #{rank:>2}  id={cid:>4}  size={meta['size']:>4}  "
          f"(merged {meta['merged_count']} Phase-1 cluster(s))")

print(f"")
print(f"  Uncertified articles (→ Phase 2B): {len(uncertified_ids):,}")
print(f"  cluster_cream = -1  → uncertified")
print(f"  cluster_cream ≥  0  → consolidated cream ID")
print("═" * 58)


# ─────────────────────────────────────────────────────────────
# PHASE 2B · STEP 1 — ASSIGN UNCERTIFIED TO NEAREST CENTROID
# Centroids are recomputed from consolidated_clusters (post-2A)
# and kept FROZEN — they do not update as articles are assigned.
# ─────────────────────────────────────────────────────────────
num_cons = len(consolidated_clusters)
print(f"\n{'═' * 58}")
print("  PHASE 2B — NEAREST-CENTROID ASSIGNMENT")
print(f"{'═' * 58}")
print(f"⏳ Recomputing frozen centroids from {num_cons} consolidated clusters...")

# Recompute from consolidated_clusters so centroids reflect any 2A merges
cons_centroids = np.zeros((num_cons, embeddings.shape[1]), dtype=np.float32)
for i, article_set in enumerate(consolidated_clusters):
    vecs     = embeddings[list(article_set)]
    centroid = vecs.mean(axis=0)
    norm     = np.linalg.norm(centroid)
    cons_centroids[i] = centroid / norm if norm > 0 else centroid

print(f"   Centroids ready — shape {cons_centroids.shape}")

# Sorted list for stable indexing
uncert_list = sorted(uncertified_ids)
uncert_emb  = emb_norm[uncert_list]          # already L2-normalised

assigned = []   # (orig_row_idx, cluster_id, sim_score)
orphans  = []   # orig_row_idx

if num_cons == 0:
    print("   No consolidated clusters — all uncertified articles become orphans.")
    orphans = list(uncert_list)
else:
    # Cosine similarity: (num_uncert, num_cons)
    print(f"⏳ Computing similarity matrix ({len(uncert_list):,} × {num_cons:,})...")
    sim_2b   = uncert_emb @ cons_centroids.T     # dot of two L2-normed = cosine sim
    best_sim = sim_2b.max(axis=1)                # best score per article
    best_cid = sim_2b.argmax(axis=1)             # which cluster it maps to

    for i, orig_idx in enumerate(uncert_list):
        if best_sim[i] >= ASSIGN_THRESHOLD:
            assigned.append((orig_idx, int(best_cid[i]), float(best_sim[i])))
        else:
            orphans.append(orig_idx)

    # Stamp assigned articles into df
    for orig_idx, cid, _ in assigned:
        df.at[orig_idx, "cluster_cream"] = cid

# Results
assigned_sims = np.array([s for _, _, s in assigned]) if assigned else np.array([])

print(f"\n  Uncertified going in   : {len(uncert_list):,}")
print(f"  Threshold              : cosine ≥ {ASSIGN_THRESHOLD}")
print(f"")
print(f"  ✅ Assigned to a cream cluster : {len(assigned):,}  "
      f"({100*len(assigned)/len(uncert_list):.1f}%)")
print(f"  ❌ Remaining orphans           : {len(orphans):,}  "
      f"({100*len(orphans)/len(uncert_list):.1f}%)")
if len(assigned_sims) > 0:
    print(f"")
    print(f"  Assigned sim scores:")
    print(f"    Min  : {assigned_sims.min():.4f}")
    print(f"    Mean : {assigned_sims.mean():.4f}")
    print(f"    Max  : {assigned_sims.max():.4f}")

# Similarity score distribution for assigned articles
if len(assigned_sims) > 0:
    bins = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
    labels = ["0.65-0.69", "0.70-0.74", "0.75-0.79", "0.80-0.84", "0.85-0.89", "0.90-0.94", "0.95-1.00"]
    print(f"")
    print(f"  Assigned sim distribution:")
    for lo, hi, lbl in zip(bins, bins[1:], labels):
        count = int(((assigned_sims >= lo) & (assigned_sims < hi)).sum())
        bar   = "█" * min(count // max(len(assigned) // 40, 1), 40)
        print(f"    {lbl} : {count:>5}  {bar}")

print(f"")
print(f"  Orphans → HDBSCAN mini-clustering (Phase 2B Step 2)")
print("═" * 58)


# ─────────────────────────────────────────────────────────────
# PHASE 2B · STEP 2 — HDBSCAN ON ORPHANS
# HDBSCAN cluster within size cap → PROVISIONAL
# HDBSCAN noise (-1), singleton, or oversized cluster → ISOLATED
# ─────────────────────────────────────────────────────────────
print(f"\n{'═' * 58}")
print("  PHASE 2B STEP 2 — HDBSCAN ON ORPHANS")
print(f"{'═' * 58}")

# ── Tag confidence for all articles ──────────────────────────
# All certified articles are TRIPLE_CREAM (3/3 consensus)
df["confidence"] = df["cluster_cream"].map(
    lambda v: "TRIPLE_CREAM" if v >= 0 else "UNCERTIFIED"
)
for orig_idx, cid, sim in assigned:
    df.at[orig_idx, "confidence"] = "HIGH_ASSIGNED" if sim >= 0.75 else "SOFT_ASSIGNED"

next_cluster_id = len(consolidated_clusters)  # HDBSCAN gets IDs after cream IDs

if len(orphans) == 0:
    print("   No orphans — all uncertified articles assigned.")
    prov_clusters      = 0
    isol_count         = 0
    prov_article_count = 0

elif len(orphans) == 1:
    df.at[orphans[0], "confidence"]    = "ISOLATED"
    df.at[orphans[0], "cluster_cream"] = -2
    prov_clusters      = 0
    isol_count         = 1
    prov_article_count = 0

else:
    print(f"⏳ Running HDBSCAN (min_cluster_size={ORPHAN_MIN_CLUSTER_SIZE}, "
          f"min_samples={ORPHAN_MIN_SAMPLES}) on {len(orphans)} orphans...")
    orphan_emb_norm = emb_norm[orphans]   # already L2-normalised

    # euclidean on L2-normalised vectors ≡ cosine distance — no extra computation needed
    hdb = HDBSCAN(
        min_cluster_size=ORPHAN_MIN_CLUSTER_SIZE,
        min_samples=ORPHAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",   # excess-of-mass: stable, avoids micro-clusters
    ).fit(orphan_emb_norm)
    hdb_labels = hdb.labels_   # -1 = noise

    hdb_series   = pd.Series(hdb_labels)
    hdb_vc       = hdb_series.value_counts()
    noise_mask   = hdb_series == -1
    non_noise_vc = hdb_vc[hdb_vc.index != -1]

    # Apply size cap: clusters exceeding PROV_MAX_CLUSTER_SIZE are too broad → ISOLATED
    prov_vc      = non_noise_vc[(non_noise_vc >= 2) & (non_noise_vc <= PROV_MAX_CLUSTER_SIZE)]
    oversized_vc = non_noise_vc[non_noise_vc > PROV_MAX_CLUSTER_SIZE]
    sing_vc      = non_noise_vc[non_noise_vc == 1]

    valid_prov_labels  = set(prov_vc.index)
    prov_clusters      = len(valid_prov_labels)
    isol_count         = int(noise_mask.sum()) + int(len(sing_vc)) + int(len(oversized_vc))
    prov_article_count = int(prov_vc.sum())

    print(f"   Raw HDBSCAN clusters       : {len(non_noise_vc)}")
    print(f"   Noise points (-1)          : {int(noise_mask.sum())}  → ISOLATED")
    print(f"   Singletons (=1)            : {len(sing_vc)}  → ISOLATED")
    print(f"   Oversized (>{PROV_MAX_CLUSTER_SIZE} articles)    : {len(oversized_vc)}  → ISOLATED  "
          f"({int(oversized_vc.sum())} articles)")
    print(f"   Provisional clusters kept  : {prov_clusters}  ({prov_article_count} articles)")

    for i, orig_idx in enumerate(orphans):
        lbl = int(hdb_labels[i])
        if lbl == -1 or lbl not in valid_prov_labels:
            df.at[orig_idx, "confidence"]    = "ISOLATED"
            df.at[orig_idx, "cluster_cream"] = -2
        else:
            df.at[orig_idx, "confidence"]    = "PROVISIONAL"
            df.at[orig_idx, "cluster_cream"] = next_cluster_id + lbl

    # ── Post-hoc quality gate on PROVISIONAL clusters ────────
    # Rejects provisional clusters whose mean pairwise intra-sim
    # falls below CREAM_MIN_INTRA_SIM — same bar as cream clusters.
    _prov_rejected = 0
    _prov_rej_arts = 0
    prov_cids = df[df["confidence"] == "PROVISIONAL"]["cluster_cream"].unique()
    for _pcid in prov_cids:
        _pidxs = df[df["cluster_cream"] == _pcid].index.tolist()
        _pvecs = emb_norm[_pidxs]
        _psmat = _pvecs @ _pvecs.T
        _pupper = _psmat[np.triu_indices(len(_pidxs), k=1)]
        _pmsim = float(_pupper.mean()) if len(_pupper) > 0 else 0.0
        if _pmsim < CREAM_MIN_INTRA_SIM:
            for _pidx in _pidxs:
                df.at[_pidx, "confidence"]    = "ISOLATED"
                df.at[_pidx, "cluster_cream"] = -2
            _prov_rejected += 1
            _prov_rej_arts += len(_pidxs)

    prov_clusters      -= _prov_rejected
    prov_article_count -= _prov_rej_arts
    isol_count         += _prov_rej_arts
    print(f"   Prov quality gate (intra-sim ≥ {CREAM_MIN_INTRA_SIM}):")
    print(f"     Rejected : {_prov_rejected} clusters  ({_prov_rej_arts} articles → ISOLATED)")
    print(f"     Kept     : {prov_clusters} clusters  ({prov_article_count} articles)")


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE SUMMARY
# ─────────────────────────────────────────────────────────────
conf_counts = df["confidence"].value_counts()

triple_cream_count = int(conf_counts.get("TRIPLE_CREAM",  0))
high_assigned      = int(conf_counts.get("HIGH_ASSIGNED", 0))
soft_assigned      = int(conf_counts.get("SOFT_ASSIGNED", 0))
provisional_count  = int(conf_counts.get("PROVISIONAL",   0))
isolated_count     = int(conf_counts.get("ISOLATED",      0))
effective_cov      = triple_cream_count + high_assigned + soft_assigned + provisional_count

# Clusters with real IDs (cream + provisional; exclude ISOLATED=-2)
real_clusters_df    = df[df["cluster_cream"] >= 0]
total_final_clusters = real_clusters_df["cluster_cream"].nunique()
final_cluster_sizes  = real_clusters_df.groupby("cluster_cream").size().sort_values(ascending=False)

print(f"\n{'═' * 58}")
print("  FULL PIPELINE SUMMARY")
print(f"{'═' * 58}")
print(f"  Total articles processed       : {n:,}")
print(f"  Total meaningful clusters      : {total_final_clusters:,}")
print(f"")
print(f"  Article confidence breakdown:")
print(f"    TRIPLE_CREAM  : {triple_cream_count:>5}  ({100*triple_cream_count/n:.1f}%)  ← Phase 1 all 3 algos agree")
print(f"    HIGH_ASSIGNED : {high_assigned:>5}  ({100*high_assigned/n:.1f}%)  ← Phase 2B sim ≥ 0.75")
print(f"    SOFT_ASSIGNED : {soft_assigned:>5}  ({100*soft_assigned/n:.1f}%)  ← Phase 2B sim 0.65–0.74")
print(f"    PROVISIONAL   : {provisional_count:>5}  ({100*provisional_count/n:.1f}%)  ← HDBSCAN orphan cluster")
print(f"    ISOLATED      : {isolated_count:>5}  ({100*isolated_count/n:.1f}%)  ← true loners (noise)")
print(f"")
print(f"  Effective coverage             : {effective_cov:,} / {n:,}  ({100*effective_cov/n:.1f}%)")

print(f"")
print(f"  Cluster count breakdown:")
print(f"    Cream clusters  (Phase 1+2A) : {len(consolidated_clusters):>5}")
print(f"    Provisional clusters (HDBSCAN): {prov_clusters:>5}")
print(f"    ─────────────────────────────────")
print(f"    Total meaningful clusters    : {total_final_clusters:>5}")
print(f"    Isolated articles (no cluster): {isolated_count:>4}")

print(f"")
print(f"  Final cluster size distribution:")
size_dist_final = final_cluster_sizes.value_counts().sort_index()
for size, count in size_dist_final.items():
    bar = "█" * min(count, 40)
    print(f"    {size:>4} articles : {count:>4} clusters  {bar}")

print(f"\n  Top 15 largest final clusters:")
for rank, (cid, size) in enumerate(final_cluster_sizes.head(15).items(), 1):
    cluster_df     = df[df["cluster_cream"] == cid]
    conf_breakdown = cluster_df["confidence"].value_counts().to_dict()
    tickers        = cluster_df["symbol"].dropna().unique()
    ticker_str     = ", ".join(str(t) for t in tickers[:6])
    if len(tickers) > 6:
        ticker_str += f" +{len(tickers)-6} more"
    conf_str = "  ".join(f"{k}:{v}" for k, v in conf_breakdown.items())
    print(f"    #{rank:>2}  id={cid:>4}  size={size:>4}  [{conf_str}]")
    print(f"         tickers: {ticker_str}")

print(f"\n  Sample titles from Top 5 clusters:")
for rank, (cid, size) in enumerate(final_cluster_sizes.head(5).items(), 1):
    cluster_df = df[df["cluster_cream"] == cid]
    print(f"\n  Cluster #{rank}  (id={cid}, size={size}):")
    for title in cluster_df["title"].dropna().head(5).values:
        print(f"    • {str(title)[:100]}")

print(f"\n{'═' * 58}")
print("  END OF PIPELINE")
print("═" * 58)


# ─────────────────────────────────────────────────────────────
# INTRA-CLUSTER QUALITY — COSINE SIMILARITY
# For each final cluster: mean pairwise cosine sim of members.
# Uses emb_norm (already L2-normalised) → dot = cosine sim.
# Skips size-1 clusters (ISOLATED already excluded by >= 0 filter)
# ─────────────────────────────────────────────────────────────
print(f"\n{'═' * 58}")
print("  INTRA-CLUSTER QUALITY — COSINE SIMILARITY")
print(f"{'═' * 58}")
print("⏳ Computing intra-cluster cosine similarities...")

intra_records = []   # (cluster_id, size, mean_sim, confidence_tier)

for cid, group in real_clusters_df.groupby("cluster_cream"):
    idxs = group.index.tolist()
    if len(idxs) < 2:
        continue
    vecs     = emb_norm[idxs]                        # (k, 1536)
    sim_mat  = vecs @ vecs.T                          # (k, k) cosine sims
    upper    = sim_mat[np.triu_indices(len(idxs), k=1)]
    mean_sim = float(upper.mean())

    # Majority confidence tier for this cluster
    tier_counts = group["confidence"].value_counts()
    tier        = tier_counts.index[0]

    intra_records.append({
        "cid"      : cid,
        "size"     : len(idxs),
        "mean_sim" : mean_sim,
        "tier"     : tier,
    })

intra_df = pd.DataFrame(intra_records).sort_values("mean_sim", ascending=False)

overall_mean   = intra_df["mean_sim"].mean()
overall_median = intra_df["mean_sim"].median()

# Per-tier breakdown
triple_rows = intra_df[intra_df["tier"] == "TRIPLE_CREAM"]
prov_rows   = intra_df[intra_df["tier"] == "PROVISIONAL"]
ha_rows     = intra_df[intra_df["tier"].isin(["HIGH_ASSIGNED", "SOFT_ASSIGNED"])]

print(f"\n  Clusters evaluated          : {len(intra_df):,}")
print(f"  Overall mean intra-sim      : {overall_mean:.4f}")
print(f"  Overall median intra-sim    : {overall_median:.4f}")
print(f"")
print(f"  By dominant confidence tier:")
if len(triple_rows):
    print(f"    TRIPLE_CREAM  : mean={triple_rows['mean_sim'].mean():.4f}  "
          f"median={triple_rows['mean_sim'].median():.4f}  (n={len(triple_rows)})")
if len(ha_rows):
    print(f"    ASSIGNED      : mean={ha_rows['mean_sim'].mean():.4f}  "
          f"median={ha_rows['mean_sim'].median():.4f}  (n={len(ha_rows)})")
if len(prov_rows):
    print(f"    PROVISIONAL   : mean={prov_rows['mean_sim'].mean():.4f}  "
          f"median={prov_rows['mean_sim'].median():.4f}  (n={len(prov_rows)})")

print(f"")
print(f"  Quality distribution (all clusters):")
bins_q   = [0.0, 0.55, 0.65, 0.75, 0.85, 1.01]
labels_q = ["< 0.55 (mixed/bad)", "0.55–0.64 (loose)", "0.65–0.74 (acceptable)",
            "0.75–0.84 (good)", "≥ 0.85 (elite)"]
sims = intra_df["mean_sim"].values
for lo, hi, lbl in zip(bins_q, bins_q[1:], labels_q):
    count = int(((sims >= lo) & (sims < hi)).sum())
    bar   = "█" * min(count, 40)
    print(f"    {lbl:<26}: {count:>5} clusters  {bar}")

print(f"\n  Top 10 TIGHTEST clusters (highest intra-sim):")
for _, row in intra_df.head(10).iterrows():
    cluster_df = df[df["cluster_cream"] == row["cid"]]
    tickers    = cluster_df["symbol"].dropna().unique()
    t_str      = ", ".join(str(t) for t in tickers[:4])
    if len(tickers) > 4:
        t_str += f" +{len(tickers)-4} more"
    print(f"    id={int(row['cid']):>4}  size={int(row['size']):>4}  "
          f"sim={row['mean_sim']:.4f}  [{row['tier']}]  {t_str}")

print(f"\n  Top 10 LOOSEST clusters (lowest intra-sim):")
for _, row in intra_df.tail(10).iterrows():
    cluster_df = df[df["cluster_cream"] == row["cid"]]
    tickers    = cluster_df["symbol"].dropna().unique()
    t_str      = ", ".join(str(t) for t in tickers[:4])
    if len(tickers) > 4:
        t_str += f" +{len(tickers)-4} more"
    print(f"    id={int(row['cid']):>4}  size={int(row['size']):>4}  "
          f"sim={row['mean_sim']:.4f}  [{row['tier']}]  {t_str}")

print("═" * 58)


# ─────────────────────────────────────────────────────────────
# SAVE OUTPUT TO CSV  (sorted: strongest clusters first)
# Sort order:
#   1. confidence tier  (TRIPLE_CREAM → HIGH → SOFT → PROVISIONAL → ISOLATED)
#   2. cluster size     (larger cluster = more sites = more important story)
#   3. cluster_cream ID (stable grouping within same tier+size)
# ─────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "cluster_output.csv")

_tier_rank = {
    "TRIPLE_CREAM" : 1,
    "HIGH_ASSIGNED": 2,
    "SOFT_ASSIGNED": 3,
    "PROVISIONAL"  : 4,
    "ISOLATED"     : 5,
    "UNCERTIFIED"  : 6,
}

# Map each cluster to its total article count (ISOLATED articles use size=1)
_cluster_size = (
    df[df["cluster_cream"] >= 0]
    .groupby("cluster_cream")
    .size()
    .to_dict()
)

df["_tier_rank"]    = df["confidence"].map(_tier_rank).fillna(6).astype(int)
df["_cluster_size"] = df["cluster_cream"].map(_cluster_size).fillna(1).astype(int)

output_cols = [
    "symbol", "title", "publish_date", "site",
    "sentiment", "sentimentscore",
    "cluster_cream", "confidence", "vote_count",
]

df_out = (
    df[output_cols + ["_tier_rank", "_cluster_size"]]
    .sort_values(
        ["_tier_rank", "_cluster_size", "cluster_cream"],
        ascending=[True, False, True],
    )
    .drop(columns=["_tier_rank", "_cluster_size"])
)

df_out.to_csv(out_path, index=False)
print(f"\n✅ Saved {n:,} articles → {out_path}")
print(f"   Row order: TRIPLE_CREAM (largest→smallest) → HIGH → SOFT → PROVISIONAL → ISOLATED")

# ── EMBEDDING COST SUMMARY ────────────────────────────────────
_EMB_PRICE_PER_1M = 0.02   # text-embedding-3-small: $0.02 / 1M tokens
_emb_cost = (_emb_tokens / 1_000_000) * _EMB_PRICE_PER_1M
print(f"\n{'─' * 40}")
print(f"  💰 ensemble2.py cost summary")
print(f"{'─' * 40}")
print(f"  Model          : text-embedding-3-small")
print(f"  Articles total : {n:,}  (hits={len(_hit_idx):,}  misses={len(_miss_idx):,})")
print(f"  Tokens used    : {_emb_tokens:,}")
print(f"  Total cost     : ${_emb_cost:.5f}  (~${_emb_cost*30:.3f}/month @ daily runs)")
print(f"{'─' * 40}")
