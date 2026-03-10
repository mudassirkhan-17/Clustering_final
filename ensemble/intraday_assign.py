"""
intraday_assign.py — Hourly Speed Layer
────────────────────────────────────────
Loads frozen centroids from the last daily ensemble2.py run, fetches
articles published since the last intraday check, embeds them (using
the shared rolling cache), assigns them to existing clusters via
nearest-centroid cosine similarity, and groups any leftovers with
HDBSCAN to form new PROVISIONAL clusters.

Outputs
───────
  intraday_output.csv   — new articles with assigned cluster_id + confidence
  centroid_meta.json    — updated with any new PROVISIONAL cluster IDs
  last_intraday_run.txt — timestamp written at end of each successful run

Run this script hourly (e.g. via Windows Task Scheduler or cron).
It is safe to run even if no new articles exist — it will just exit early.
"""

import os
import json
import hashlib
import pickle
import threading
from datetime import datetime, timedelta

import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from hdbscan import HDBSCAN
from joblib import Parallel, delayed
import tiktoken

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
ASSIGN_HIGH_THRESHOLD  = 0.75   # cosine sim → HIGH_ASSIGNED
ASSIGN_SOFT_THRESHOLD  = 0.65   # cosine sim → SOFT_ASSIGNED (below → orphan)
ORPHAN_MIN_CLUSTER_SIZE = 3     # HDBSCAN min articles to form a new provisional cluster
ORPHAN_MIN_SAMPLES      = 2     # HDBSCAN core-point density
PROV_MAX_CLUSTER_SIZE   = 50    # cap: clusters bigger than this → ISOLATED
CREAM_MIN_INTRA_SIM     = 0.60  # post-hoc intra-sim quality gate for provisional clusters
EMB_CACHE_DAYS          = 3     # evict cache entries older than this many days
MAX_TOKENS              = 7000  # same as ensemble2.py
EMB_BATCH               = 100   # articles per API call
# ══════════════════════════════════════════════════════════════

_HERE = os.path.dirname(__file__)

CENTROIDS_PATH        = os.path.join(_HERE, "centroids.npy")
META_PATH             = os.path.join(_HERE, "centroid_meta.json")
CACHE_PATH            = os.path.join(_HERE, "embedding_cache.pkl")
LAST_DAILY_RUN_PATH   = os.path.join(_HERE, "last_daily_run.txt")
LAST_INTRADAY_PATH    = os.path.join(_HERE, "last_intraday_run.txt")
INTRADAY_OUTPUT_PATH  = os.path.join(_HERE, "intraday_output.csv")


# ─────────────────────────────────────────────────────────────
# GUARD: ensure daily state exists before running
# ─────────────────────────────────────────────────────────────
for _required in (CENTROIDS_PATH, META_PATH):
    if not os.path.exists(_required):
        print(f"❌ Required state file missing: {_required}")
        print("   Run ensemble2.py first to generate the daily baseline.")
        raise SystemExit(1)


# ─────────────────────────────────────────────────────────────
# DETERMINE TIME WINDOW FOR THIS RUN
# ─────────────────────────────────────────────────────────────
def _read_ts(path: str, fallback_hours: int = 2) -> datetime:
    """Read a timestamp file; fallback to now - fallback_hours if missing."""
    if os.path.exists(path):
        try:
            return datetime.strptime(open(path).read().strip(), "%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return datetime.now() - timedelta(hours=fallback_hours)

_daily_run_ts    = _read_ts(LAST_DAILY_RUN_PATH, fallback_hours=24)
_last_intraday   = _read_ts(LAST_INTRADAY_PATH,  fallback_hours=2)

# Fetch articles that arrived since the last intraday run (but no older than
# the daily run, to avoid double-assigning articles already in cluster_output.csv)
_fetch_since = max(_daily_run_ts, _last_intraday)

print(f"\n{'═' * 58}")
print("  INTRADAY ASSIGN — SPEED LAYER")
print(f"{'═' * 58}")
print(f"  Daily baseline run  : {_daily_run_ts.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Last intraday run   : {_last_intraday.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Fetching articles > : {_fetch_since.strftime('%Y-%m-%d %H:%M:%S')}")


# ─────────────────────────────────────────────────────────────
# STEP 1 · FETCH NEW ARTICLES
# ─────────────────────────────────────────────────────────────
conn = psycopg2.connect(os.getenv("OFFICEFIELD_LOCAL_SUPABASE_STRINGS"))
_query = """
    SELECT symbol, title, content AS text,
           publish_date, site, sentiment,
           sentiment_score AS sentimentscore
    FROM   master_news
    WHERE  publish_date > %s
    ORDER  BY publish_date ASC;
"""
df = pd.read_sql_query(_query, conn, params=(_fetch_since,))
conn.close()

df = df.reset_index(drop=True)
df["publish_date"] = pd.to_datetime(df["publish_date"])

print(f"\n✅ Fetched {len(df):,} new articles since last intraday run")

if len(df) == 0:
    print("   Nothing new — exiting early.")
    with open(LAST_INTRADAY_PATH, "w") as _f:
        _f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    raise SystemExit(0)


# ─────────────────────────────────────────────────────────────
# STEP 2 · DEDUPLICATE
# ─────────────────────────────────────────────────────────────
_before = len(df)
df["_title_key"] = df["title"].fillna("").str.strip().str[:200]
df = (
    df.sort_values("title", key=lambda s: s.str.len(), ascending=False)
      .drop_duplicates(subset=["_title_key", "site"], keep="first")
      .drop(columns=["_title_key"])
      .reset_index(drop=True)
)
print(f"🧹 Deduplication: {_before:,} → {len(df):,}  ({_before - len(df):,} removed)")
n = len(df)


# ─────────────────────────────────────────────────────────────
# STEP 3 · CLEAN + EMBED  (shared rolling cache)
# ─────────────────────────────────────────────────────────────
def _clean(s):
    if pd.isna(s):
        return ""
    return " ".join(str(s).split())

_enc = tiktoken.get_encoding("cl100k_base")

def _truncate(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text[:25_000]
    try:
        tokens = _enc.encode(text, disallowed_special=())
        if len(tokens) > MAX_TOKENS:
            text = _enc.decode(tokens[:MAX_TOKENS])
    except Exception:
        text = text[:4_000]
    return text

_emb_tokens = 0
_tok_lock   = threading.Lock()

def _embed_batch(texts: list) -> list:
    global _emb_tokens
    safe = [_truncate(t) for t in texts]
    try:
        resp = client.embeddings.create(model="text-embedding-3-small", input=safe)
        if resp.usage:
            with _tok_lock:
                _emb_tokens += resp.usage.total_tokens
        return [d.embedding for d in resp.data]
    except Exception:
        results = []
        for t in safe:
            t = t[:8_000]
            resp = client.embeddings.create(model="text-embedding-3-small", input=[t])
            if resp.usage:
                with _tok_lock:
                    _emb_tokens += resp.usage.total_tokens
            results.append(resp.data[0].embedding)
        return results

def _article_hash(title: str, publish_date) -> str:
    return hashlib.md5(f"{title}|{publish_date}".encode()).hexdigest()

def _load_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "rb") as _f:
            return pickle.load(_f)
    except Exception:
        return {}

def _save_cache(cache: dict) -> None:
    _tmp = CACHE_PATH + ".tmp"
    with open(_tmp, "wb") as _f:
        pickle.dump(cache, _f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(_tmp, CACHE_PATH)

# Build embed text
df["title_clean"]  = df["title"].map(_clean)
df["text_clean"]   = df["text"].map(_clean).str.slice(0, 25_000)
df["symbol_clean"] = df["symbol"].fillna("").astype(str).str.strip()
df["embed_text"]   = (
    df["symbol_clean"] + "\n"
    + df["title_clean"] + "\n"
    + df["title_clean"] + "\n"
    + df["text_clean"]
)
df["embed_text"] = df["embed_text"].map(_truncate)

# Cache: load → evict stale → serve hits → embed misses
_cache_cutoff = datetime.now() - timedelta(days=EMB_CACHE_DAYS)
_emb_cache    = _load_cache()
_before_cnt   = len(_emb_cache)
_emb_cache    = {
    h: v for h, v in _emb_cache.items()
    if pd.to_datetime(v["publish_date"]) >= _cache_cutoff
}
print(f"\n📦 Embedding cache: {_before_cnt:,} loaded  |  "
      f"{_before_cnt - len(_emb_cache):,} evicted  |  {len(_emb_cache):,} live")

df["_hash"] = [
    _article_hash(str(row.title), row.publish_date)
    for row in df[["title", "publish_date"]].itertuples(index=False)
]

_hit_idx  = [i for i in range(n) if df.at[i, "_hash"] in _emb_cache]
_miss_idx = [i for i in range(n) if df.at[i, "_hash"] not in _emb_cache]
print(f"   Cache hits  : {len(_hit_idx):,}")
print(f"   Cache misses: {len(_miss_idx):,}")

_all_vecs: list = [None] * n
for _i in _hit_idx:
    _all_vecs[_i] = _emb_cache[df.at[_i, "_hash"]]["embedding"]

if _miss_idx:
    def _embed_miss_batch(idx_batch: list) -> tuple:
        texts = [df.at[i, "embed_text"] for i in idx_batch]
        return idx_batch, _embed_batch(texts)

    _miss_batches = [_miss_idx[s:s + EMB_BATCH] for s in range(0, len(_miss_idx), EMB_BATCH)]
    print(f"\n⏳ Embedding {len(_miss_idx):,} new articles "
          f"({len(_miss_batches)} batches, parallel)...")

    _miss_results = Parallel(n_jobs=-1, backend="threading", verbose=0)(
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
    print(f"✅ Embedded {len(_miss_idx):,} articles  ({_emb_tokens:,} tokens)")
else:
    print("✅ All articles served from cache  (0 API tokens)")

_save_cache(_emb_cache)

df["embedding"] = _all_vecs
embeddings = np.vstack(df["embedding"].values)
norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
emb_norm   = embeddings / np.where(norms == 0, 1.0, norms)


# ─────────────────────────────────────────────────────────────
# STEP 4 · LOAD FROZEN CENTROIDS + META
# ─────────────────────────────────────────────────────────────
centroids = np.load(CENTROIDS_PATH)   # (num_clusters, emb_dim)  float32, L2-normalised
with open(META_PATH) as _mf:
    meta = json.load(_mf)

cluster_ids     = meta["cluster_ids"]         # list[int], row-order for centroids
next_cluster_id = meta["next_cluster_id"]     # int — new PROVISIONAL IDs start here
cluster_info    = meta["cluster_info"]        # str(cid) → {size, confidence}

print(f"\n📌 Loaded {len(cluster_ids)} frozen centroids from daily run")
print(f"   centroid matrix shape : {centroids.shape}")
print(f"   next_cluster_id       : {next_cluster_id}")


# ─────────────────────────────────────────────────────────────
# STEP 5 · NEAREST-CENTROID ASSIGNMENT
# ─────────────────────────────────────────────────────────────
# dot product of L2-normalised vectors == cosine similarity
sim_matrix = emb_norm @ centroids.T   # (n_new, num_clusters)

best_sim   = sim_matrix.max(axis=1)           # (n_new,)
best_pos   = sim_matrix.argmax(axis=1)        # (n_new,) index into cluster_ids
best_cid   = [cluster_ids[p] for p in best_pos]

df["cluster_id"]  = -2   # default: ISOLATED
df["confidence"]  = "ISOLATED"
df["best_sim"]    = best_sim

high_count = soft_count = orphan_count = 0

for i in range(n):
    sim = float(best_sim[i])
    if sim >= ASSIGN_HIGH_THRESHOLD:
        df.at[i, "cluster_id"] = best_cid[i]
        df.at[i, "confidence"] = "HIGH_ASSIGNED"
        high_count += 1
    elif sim >= ASSIGN_SOFT_THRESHOLD:
        df.at[i, "cluster_id"] = best_cid[i]
        df.at[i, "confidence"] = "SOFT_ASSIGNED"
        soft_count += 1
    else:
        orphan_count += 1

print(f"\n📊 Assignment results ({n:,} new articles):")
print(f"   HIGH_ASSIGNED  (sim ≥ {ASSIGN_HIGH_THRESHOLD}) : {high_count:>5}")
print(f"   SOFT_ASSIGNED  (sim ≥ {ASSIGN_SOFT_THRESHOLD}) : {soft_count:>5}")
print(f"   Orphans        (sim <  {ASSIGN_SOFT_THRESHOLD}) : {orphan_count:>5}")


# ─────────────────────────────────────────────────────────────
# STEP 6 · HDBSCAN ON ORPHANS  (new PROVISIONAL clusters)
# ─────────────────────────────────────────────────────────────
orphans = df[df["confidence"] == "ISOLATED"].index.tolist()

new_prov_clusters = 0
isol_count        = orphan_count

if len(orphans) < 2:
    print(f"\n   {len(orphans)} orphan(s) — too few for HDBSCAN, all marked ISOLATED")

else:
    orphan_emb = emb_norm[orphans]
    print(f"\n⏳ HDBSCAN on {len(orphans)} orphans "
          f"(min_cluster_size={ORPHAN_MIN_CLUSTER_SIZE}, min_samples={ORPHAN_MIN_SAMPLES})...")

    hdb = HDBSCAN(
        min_cluster_size=ORPHAN_MIN_CLUSTER_SIZE,
        min_samples=ORPHAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit(orphan_emb)
    hdb_labels = hdb.labels_

    hdb_vc       = pd.Series(hdb_labels).value_counts()
    non_noise_vc = hdb_vc[hdb_vc.index != -1]
    noise_cnt    = int((pd.Series(hdb_labels) == -1).sum())

    valid_labels = set(
        lbl for lbl, cnt in non_noise_vc.items()
        if 2 <= cnt <= PROV_MAX_CLUSTER_SIZE
    )
    print(f"   Raw HDBSCAN clusters : {len(non_noise_vc)}")
    print(f"   Noise points         : {noise_cnt}  → ISOLATED")
    print(f"   Valid provisional    : {len(valid_labels)}")

    # Assign provisional IDs and run quality gate
    _lbl_to_cid = {lbl: next_cluster_id + i for i, lbl in enumerate(sorted(valid_labels))}

    for i, orig_idx in enumerate(orphans):
        lbl = int(hdb_labels[i])
        if lbl in _lbl_to_cid:
            df.at[orig_idx, "cluster_id"] = _lbl_to_cid[lbl]
            df.at[orig_idx, "confidence"] = "PROVISIONAL"

    # Post-hoc intra-sim quality gate (same bar as ensemble2.py)
    prov_cids = df[df["confidence"] == "PROVISIONAL"]["cluster_id"].unique()
    _rejected = 0
    for _pcid in prov_cids:
        _pidxs = df[df["cluster_id"] == _pcid].index.tolist()
        _pvecs = emb_norm[_pidxs]
        _psmat = _pvecs @ _pvecs.T
        _upper = _psmat[np.triu_indices(len(_pidxs), k=1)]
        _msim  = float(_upper.mean()) if len(_upper) > 0 else 0.0
        if _msim < CREAM_MIN_INTRA_SIM:
            for _pidx in _pidxs:
                df.at[_pidx, "confidence"] = "ISOLATED"
                df.at[_pidx, "cluster_id"] = -2
            _rejected += 1

    new_prov_clusters = len(prov_cids) - _rejected
    isol_count = int((df["confidence"] == "ISOLATED").sum())

    print(f"   Quality gate rejected: {_rejected} cluster(s)  (intra-sim < {CREAM_MIN_INTRA_SIM})")
    print(f"   New PROVISIONAL      : {new_prov_clusters} cluster(s)  "
          f"({int((df['confidence'] == 'PROVISIONAL').sum())} articles)")

    # Update next_cluster_id in meta for next intraday run
    if new_prov_clusters > 0:
        _max_new_cid = max(
            (int(cid) for cid in df[df["confidence"] == "PROVISIONAL"]["cluster_id"]),
            default=next_cluster_id - 1,
        )
        meta["next_cluster_id"] = _max_new_cid + 1
        # Record new clusters in cluster_info
        for _pcid in df[df["confidence"] == "PROVISIONAL"]["cluster_id"].unique():
            _pcid_str = str(int(_pcid))
            _psize    = int((df["cluster_id"] == _pcid).sum())
            meta["cluster_info"][_pcid_str] = {
                "size"      : _psize,
                "confidence": "PROVISIONAL",
            }
        with open(META_PATH, "w") as _mf:
            json.dump(meta, _mf)
        print(f"   centroid_meta.json updated  (next_cluster_id={meta['next_cluster_id']})")


# ─────────────────────────────────────────────────────────────
# STEP 7 · SAVE OUTPUT
# ─────────────────────────────────────────────────────────────
output_cols = [
    "symbol", "title", "publish_date", "site",
    "sentiment", "sentimentscore",
    "cluster_id", "confidence", "best_sim",
]

_tier_rank = {
    "HIGH_ASSIGNED" : 1,
    "SOFT_ASSIGNED" : 2,
    "PROVISIONAL"   : 3,
    "ISOLATED"      : 4,
}
df["_tier_rank"] = df["confidence"].map(_tier_rank).fillna(4).astype(int)

df_out = (
    df[output_cols + ["_tier_rank"]]
    .sort_values(["_tier_rank", "cluster_id"], ascending=[True, True])
    .drop(columns=["_tier_rank"])
)

# Append to existing intraday output (accumulates throughout the day)
_write_header = not os.path.exists(INTRADAY_OUTPUT_PATH)
df_out.to_csv(INTRADAY_OUTPUT_PATH, mode="a", index=False, header=_write_header)

print(f"\n✅ Saved {n:,} articles → {INTRADAY_OUTPUT_PATH}")
print(f"   (appended; header={'yes' if _write_header else 'no – already exists'})")


# ─────────────────────────────────────────────────────────────
# STEP 8 · UPDATE TIMESTAMP
# ─────────────────────────────────────────────────────────────
_now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(LAST_INTRADAY_PATH, "w") as _f:
    _f.write(_now_str)


# ─────────────────────────────────────────────────────────────
# COST SUMMARY
# ─────────────────────────────────────────────────────────────
_EMB_PRICE_PER_1M = 0.02
_emb_cost = (_emb_tokens / 1_000_000) * _EMB_PRICE_PER_1M

print(f"\n{'─' * 50}")
print(f"  INTRADAY SUMMARY  —  {_now_str}")
print(f"{'─' * 50}")
print(f"  New articles processed : {n:,}")
print(f"    HIGH_ASSIGNED        : {high_count:>5}")
print(f"    SOFT_ASSIGNED        : {soft_count:>5}")
print(f"    PROVISIONAL (new)    : {int((df['confidence'] == 'PROVISIONAL').sum()):>5}  "
      f"({new_prov_clusters} new clusters)")
print(f"    ISOLATED             : {isol_count:>5}")
print(f"  Embedding tokens used  : {_emb_tokens:,}  → ${_emb_cost:.5f}")
print(f"  Cache hits             : {len(_hit_idx):,}  misses: {len(_miss_idx):,}")
print(f"{'─' * 50}")
