"""
Cluster Summarizer
Reads cluster_output.csv produced by ensemble2.py and generates a one-row-per-cluster
summary CSV with a GPT-generated headline AND summary for each cluster.

Output: cluster_summary.csv
Columns:
    cluster_id | confidence | size | avg_sentiment | tickers | headline | summary | sample_titles
"""

import os
import json
import threading
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from joblib import Parallel, delayed

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── CONFIG ────────────────────────────────────────────────────
INPUT_CSV   = os.path.join(os.path.dirname(__file__), "cluster_output.csv")
OUTPUT_CSV  = os.path.join(os.path.dirname(__file__), "cluster_summary.csv")
MAX_TITLES  = 7      # titles sent to GPT per cluster
MAX_TICKERS = 6      # tickers shown in summary row
SKIP_TIERS  = {"ISOLATED", "UNCERTIFIED"}   # don't summarize noise
N_JOBS      = -1     # use all available cores (threading, I/O-bound)
# gpt-4o-mini pricing (USD per 1M tokens)
_PRICE_IN   = 0.15
_PRICE_OUT  = 0.60
# ─────────────────────────────────────────────────────────────

# Thread-safe token accumulators
_tok_lock   = threading.Lock()
_tok_in     = 0
_tok_out    = 0


def build_prompt(titles: list[str], tickers: list[str], size: int) -> str:
    ticker_str  = ", ".join(tickers[:MAX_TICKERS]) if tickers else "unknown"
    title_block = "\n".join(f"- {t}" for t in titles[:MAX_TITLES])
    return (
        f"You are a financial news editor. Below are {size} article headlines "
        f"from a news cluster about tickers: {ticker_str}.\n\n"
        f"{title_block}\n\n"
        f"Return a JSON object with exactly two keys:\n"
        f'  "headline": ONE concise headline (max 15 words) capturing the core story. '
        f"Be specific — include the company/ticker and the main event.\n"
        f'  "summary": A 3-5 sentence paragraph synthesising what all these articles '
        f"collectively say. Cover the key facts, market impact, and any notable context. "
        f"Write in third person, present tense.\n\n"
        f"Return ONLY valid JSON, nothing else."
    )


def gpt_summarize(titles: list[str], tickers: list[str], size: int) -> tuple[str, str]:
    """Returns (headline, summary). Falls back gracefully on error."""
    global _tok_in, _tok_out
    prompt = build_prompt(titles, tickers, size)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        # Accumulate token usage thread-safely
        if resp.usage:
            with _tok_lock:
                _tok_in  += resp.usage.prompt_tokens
                _tok_out += resp.usage.completion_tokens

        data     = json.loads(resp.choices[0].message.content or "{}")
        headline = str(data.get("headline", "")).strip().strip('"')
        summary  = str(data.get("summary",  "")).strip()
        return headline, summary
    except Exception as e:
        return f"[ERROR: {e}]", ""


def _process_cluster(rank: int, cid: int, group: pd.DataFrame, total: int) -> dict:
    """Process one cluster — called in parallel by joblib threads."""
    size       = len(group)
    confidence = group["confidence"].mode()[0]
    titles     = group["title"].dropna().tolist()
    tickers    = [
        t.strip()
        for t in group["symbol"].dropna().unique().tolist()
        if str(t).strip()
    ]
    avg_sentiment = (
        round(float(group["sentimentscore"].dropna().mean()), 3)
        if group["sentimentscore"].notna().any()
        else None
    )

    headline, summary = gpt_summarize(titles, tickers, size)

    # ── Sentiment breakdown ───────────────────────────────────
    sent_col  = group["sentiment"].str.strip().str.lower()
    pos_mask  = sent_col == "positive"
    neg_mask  = sent_col == "negative"
    neu_mask  = sent_col == "neutral"

    positive_count = int(pos_mask.sum())
    negative_count = int(neg_mask.sum())
    neutral_count  = int(neu_mask.sum())

    positive_sites = ", ".join(group.loc[pos_mask, "site"].dropna().unique().tolist())
    negative_sites = ", ".join(group.loc[neg_mask, "site"].dropna().unique().tolist())
    neutral_sites  = ", ".join(group.loc[neu_mask, "site"].dropna().unique().tolist())
    # ─────────────────────────────────────────────────────────

    print(f"  [{rank:>4}/{total}]  id={int(cid):>4}  size={size:>4}  [{confidence}]")
    print(f"           Headline → {headline}")
    print(f"           Summary  → {summary[:110]}{'...' if len(summary) > 110 else ''}")
    print(f"           Sentiment → +{positive_count} / -{negative_count} / ~{neutral_count}\n")

    return {
        "cluster_id"     : int(cid),
        "confidence"     : confidence,
        "size"           : size,
        "avg_sentiment"  : avg_sentiment,
        "positive_count" : positive_count,
        "negative_count" : negative_count,
        "neutral_count"  : neutral_count,
        "tickers"        : ", ".join(tickers[:MAX_TICKERS]),
        "headline"       : headline,
        "summary"        : summary,
        "sample_titles"  : " | ".join(titles[:3]),
        "positive_sites" : positive_sites,
        "negative_sites" : negative_sites,
        "neutral_sites"  : neutral_sites,
    }


# ── LOAD ──────────────────────────────────────────────────────
print(f"📂 Reading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"   {len(df):,} articles loaded")

# ── BUILD CLUSTER LIST ────────────────────────────────────────
real_df = df[df["confidence"].notna() & ~df["confidence"].isin(SKIP_TIERS)]
real_df = real_df[real_df["cluster_cream"] >= 0]

cluster_order = real_df["cluster_cream"].unique()   # preserves CSV sort order
total         = len(cluster_order)

print(f"\n⏳ Summarising {total} clusters via GPT (n_jobs={N_JOBS}) ...\n")

# Pre-build (rank, cid, group) tuples so joblib can serialise them cleanly
tasks = [
    (rank, int(cid), real_df[real_df["cluster_cream"] == cid].copy())
    for rank, cid in enumerate(cluster_order, 1)
]

records = Parallel(n_jobs=N_JOBS, backend="threading")(
    delayed(_process_cluster)(rank, cid, group, total)
    for rank, cid, group in tasks
)

# ── SAVE ──────────────────────────────────────────────────────
summary_df = pd.DataFrame(records)
summary_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved {len(summary_df)} cluster summaries → {OUTPUT_CSV}")
print(f"   Columns: {list(summary_df.columns)}")

# ── COST SUMMARY ──────────────────────────────────────────────
_cost_in  = (_tok_in  / 1_000_000) * _PRICE_IN
_cost_out = (_tok_out / 1_000_000) * _PRICE_OUT
_cost_total = _cost_in + _cost_out
print(f"\n{'─' * 40}")
print(f"  💰 summarize.py cost summary")
print(f"{'─' * 40}")
print(f"  Model          : gpt-4o-mini")
print(f"  Input tokens   : {_tok_in:,}   → ${_cost_in:.5f}")
print(f"  Output tokens  : {_tok_out:,}  → ${_cost_out:.5f}")
print(f"  Total cost     : ${_cost_total:.5f}  (~${_cost_total*30:.3f}/month @ daily runs)")
print(f"{'─' * 40}")
