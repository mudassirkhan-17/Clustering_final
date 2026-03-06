"""
Topic Modelling for Cluster Summaries
Reads cluster_summary.csv → assigns one canonical financial topic per cluster → saves cluster_topics.csv.

Approach (ditto same as sttopic_modeling.py):
  - Same canonical topic list (100 financial topics).
  - Same TopicValidator class.
  - Same CANONICAL_MAP_THRESHOLD = 0.58.
  - Same embedding model: SentenceTransformer("BAAI/bge-base-en-v1.5").
  - Same _get_article_embedding format: f"{title} {text}" with sanitize_prompt_input max=1000.
  - Same _get_embedding_cached logic (md5 hash key).
  - Same _map_to_canonical_topic logic (cosine dot product loop).
  - Same LLM_MODEL = os.getenv("TOPIC_LLM_MODEL", "gpt-4o-mini").
  - Same temperature = 0.3.
  - Same LLM_CONFIDENCE_THRESHOLD = 0.80.
  - Same retry decorator: stop_after_attempt(3), wait_exponential(multiplier=1, min=2, max=60).
  - Same system prompt rules (canonical list, no company names, never Other/Uncategorized, JSON only).
  - No DB, no FAISS, no threading, no logger — standalone CSV-to-CSV pipeline.

Input : cluster_summary.csv   (produced by summarize.py)
Output: cluster_topics.csv    (one row per cluster, adds "topic", "topic_confidence", "topic_reasoning")
"""

import os
import re
import json
import time
import hashlib
import threading
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed
from dotenv import load_dotenv

# Add rate limiting support — ditto same as sttopic_modeling.py
try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Cost tracking (thread-safe) ───────────────────────────────────────────────
_tok_in   = 0
_tok_out  = 0
_tok_lock = threading.Lock()
# gpt-4o-mini pricing (USD per 1M tokens) — same as summarize.py
_PRICE_IN  = 0.15
_PRICE_OUT = 0.60
# ─────────────────────────────────────────────────────────────────────────────

# ── CONFIG — ditto same as sttopic_modeling.py ───────────────────────────────
INPUT_CSV                = os.path.join(os.path.dirname(__file__), "cluster_summary.csv")
OUTPUT_CSV               = os.path.join(os.path.dirname(__file__), "cluster_topics.csv")
EMBEDDING_MODEL_VERSION  = "BAAI/bge-base-en-v1.5"   # ditto same
LLM_MODEL                = os.getenv("TOPIC_LLM_MODEL", "gpt-4o-mini")   # ditto same
CANONICAL_MAP_THRESHOLD  = 0.58    # ditto same
LLM_CONFIDENCE_THRESHOLD = 0.60    # minimum score to include a topic in the ranked list
TOPIC_MAX_COUNT          = 5       # GPT returns at most this many topics per cluster
N_JOBS                   = -1       # parallel GPT threads; gpt-4o-mini handles this comfortably
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL TOPIC LIST  ← ditto same as sttopic_modeling.py
# ══════════════════════════════════════════════════════════════════════════════
CANONICAL_TOPIC_NAMES = (
    # Corporate / Catalysts
    "Earnings",
    "Guidance",
    "Earnings Call / Transcript",
    "Guidance Cut / Profit Warning",
    "M&A",
    "Deal Rumors / Talks",
    "IPO / SPAC",
    "Offerings (Equity/Debt)",
    "Secondary Offering / ATM",
    "Debt Issuance / Refinancing",
    "Credit Rating Change (Corp)",
    "Dividends",
    "Buybacks",
    "Insider Trades",
    "Shareholder Activism",
    "Management / Board",
    "Layoffs / Hiring",
    "Restructuring / Cost Cuts",
    "Bankruptcy / Restructuring",
    "Litigation / Lawsuits",
    "Regulation / Legal",
    "Antitrust / Competition",
    "SEC / Accounting / Restatement",
    "Investigations / Fraud",
    "Product Launch / Updates",
    "Recalls / Safety",
    "Supply Chain / Operations",
    "Partnerships / Contracts",
    "Government Contracts",
    "Guidance Beat / Raise",
    "Spin-off / Split / Reorg",
    "Delisting / Compliance (Listing)",
    "Shareholder Vote / Proxy",
    "Capital Return (General)",
    "Buyout / LBO / PE Activity",
    # Markets / Trading
    "Market Wrap",
    "Pre-Market",
    "After Hours",
    "Movers",
    "Unusual Volume",
    "Order Flow / Options Flow",
    "Short Interest / Short Squeeze",
    "Volatility",
    "VIX / Vol Products",
    "ETFs",
    "Options",
    "Futures",
    "Bonds / Credit",
    "Treasuries",
    "Investment Grade Credit",
    "High Yield / Distressed",
    "Credit Spreads",
    "Commodities",
    "Oil / OPEC",
    "Natural Gas",
    "Metals (Industrial)",
    "Gold (Precious Metals)",
    "Agriculture / Softs",
    "FX",
    "USD / DXY",
    "EM FX",
    "Crypto",
    "Stablecoins",
    "DeFi",
    "Rates / Curve (Market)",
    "Liquidity / Funding Stress",
    # Analyst / Research
    "Upgrades",
    "Downgrades",
    "Initiations",
    "Price Targets",
    "Stock Ratings",
    "Analyst Color",
    "Earnings Preview",
    "Earnings Recap (Street Takeaways)",
    "Estimate Changes / Revisions",
    "Consensus / Whisper Numbers",
    "Research Notes / Deep Dives",
    "Quant / Factor Notes",
    "Valuation / Multiples",
    "Channel Checks",
    # Macro
    "Central Banks",
    "Fed Speak / Minutes",
    "Interest Rates",
    "Inflation",
    "Jobs / Labor",
    "GDP / Growth",
    "PMI / ISM",
    "Retail Sales / Consumption",
    "Housing / Real Estate Macro",
    "Fiscal Policy / Budget / Treasury",
    "Politics",
    "Elections",
    "Geopolitics",
    "Sanctions / Export Controls",
    "Trade Policy / Tariffs",
    "Energy Security / Supply Shocks",
    "China Macro",
    "Emerging Markets Macro",
    "Recession / Soft Landing",
    "Market Risk-On / Risk-Off",
)


# ══════════════════════════════════════════════════════════════════════════════
# TopicValidator  ← ditto same as sttopic_modeling.py
# ══════════════════════════════════════════════════════════════════════════════
class TopicValidator:
    """Validate and sanitize topic names and content"""

    RESERVED_WORDS = {
        'null', 'undefined', 'unknown', 'test', 'review', 'temp',
        'placeholder', 'draft', 'none', 'empty', 'misc', 'general'
    }

    VALID_CHARS_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-&(),]+$')
    MAX_NAME_LENGTH = 100
    MIN_NAME_LENGTH = 5

    @staticmethod
    def validate_topic_name(name: str) -> Tuple[bool, str, str]:
        """Validate topic name. Returns: (is_valid, clean_name, error_message)"""
        if not name or not isinstance(name, str):
            return False, "", "Topic name must be non-empty string"

        clean = ' '.join(name.strip().split())

        if len(clean) < TopicValidator.MIN_NAME_LENGTH:
            return False, "", f"Name too short (min {TopicValidator.MIN_NAME_LENGTH} chars)"

        if len(clean) > TopicValidator.MAX_NAME_LENGTH:
            return False, "", f"Name too long (max {TopicValidator.MAX_NAME_LENGTH} chars)"

        if not TopicValidator.VALID_CHARS_PATTERN.match(clean):
            return False, "", "Name contains invalid characters"

        if clean.lower() in TopicValidator.RESERVED_WORDS:
            return False, "", f"Name is reserved: {clean}"

        if clean.isupper() or clean.islower():
            clean = clean.title()

        return True, clean, ""

    @staticmethod
    def sanitize_prompt_input(text: str, max_length: int = 500) -> str:
        """Prevent prompt injection attacks"""
        if not text:
            return ""
        text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')
        text = text[:max_length]
        return text


# ══════════════════════════════════════════════════════════════════════════════
# Embedding — ditto same model & logic as sttopic_modeling.py
# ══════════════════════════════════════════════════════════════════════════════
print(f"⏳ Loading embedding model {EMBEDDING_MODEL_VERSION} ...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_VERSION)
print("   Model ready.\n")

_embedding_cache: Dict[str, np.ndarray] = {}


def _get_article_embedding(article: dict) -> np.ndarray:
    """Thread-safe embedding — ditto same as sttopic_modeling.py._get_article_embedding."""
    text = f"{article.get('title', '')} {article.get('text', '')}"
    text = TopicValidator.sanitize_prompt_input(text, max_length=1000)
    emb  = embedding_model.encode([text])[0]
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def _get_embedding_cached(text: str) -> np.ndarray:
    """Get embedding with caching — ditto same as sttopic_modeling.py._get_embedding_cached."""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash not in _embedding_cache:
        embedding = _get_article_embedding({"title": "", "text": text})
        _embedding_cache[text_hash] = embedding
    return _embedding_cache[text_hash]


# ══════════════════════════════════════════════════════════════════════════════
# Canonical mapping — ditto same as sttopic_modeling.py._map_to_canonical_topic
# ══════════════════════════════════════════════════════════════════════════════
def map_to_canonical(suggested_name: str) -> str:
    """Map LLM-suggested topic name to nearest canonical name.
    If no canonical matches above threshold, return the suggested name as-is (never 'Other')."""
    if not suggested_name or not suggested_name.strip():
        return suggested_name or ""

    clean         = suggested_name.strip()
    suggested_emb = _get_embedding_cached(clean)

    best_name = None
    best_sim  = CANONICAL_MAP_THRESHOLD

    for canonical in CANONICAL_TOPIC_NAMES:
        canonical_emb = _get_embedding_cached(canonical)
        sim = float(np.dot(suggested_emb, canonical_emb))
        if sim > best_sim:
            best_sim  = sim
            best_name = canonical

    if best_name and best_name != clean:
        print(f"   [canonical snap] '{clean}' → '{best_name}'  (sim={round(best_sim, 4)})")
        return best_name

    return clean


# ══════════════════════════════════════════════════════════════════════════════
# System prompt — ditto same rules as sttopic_modeling.py._get_system_prompt
# ══════════════════════════════════════════════════════════════════════════════
def _get_system_prompt() -> str:
    topic_lines = "\n".join(f"- {t}" for t in CANONICAL_TOPIC_NAMES)
    return f"""You are a financial news topic classifier. Use GENERIC topics that suit all types of news.

CANONICAL TOPIC LIST (use these exact names in new_topic_name — these are NOT IDs):
{topic_lines}

RULES:
- Return a RANKED list of up to {TOPIC_MAX_COUNT} topics that genuinely apply to this cluster, ordered by relevance (most relevant first).
- Most clusters have 1-2 dominant topics. Only add more if they are clearly and meaningfully present.
- Assign a confidence score (0.0-1.0) to each topic. Only include topics with confidence >= {LLM_CONFIDENCE_THRESHOLD}.
- NO company names or tickers in topic names.
- If nothing matches well, suggest a short generic topic name (2-6 words). NEVER use "Other" or "Uncategorized".

OUTPUT (JSON only):
{{"topics": [{{"topic": "<exact canonical name or short generic name>", "confidence": <0.0-1.0>}}, ...], "reasoning": "<one sentence>"}}"""


def _format_classification_prompt(row: dict) -> str:
    """Format user prompt with cluster context — uses sanitize_prompt_input ditto same."""
    headline      = TopicValidator.sanitize_prompt_input(str(row.get("headline", "")), 300)
    sample_titles = TopicValidator.sanitize_prompt_input(str(row.get("sample_titles", "")), 400)
    tickers       = str(row.get("tickers", ""))
    confidence    = str(row.get("confidence", ""))
    size          = str(row.get("size", ""))

    return (
        f"CLUSTER INFO:\n"
        f"  Confidence tier : {confidence}\n"
        f"  Size (articles) : {size}\n"
        f"  Tickers         : {tickers}\n\n"
        f"  Headline        : {headline}\n\n"
        f"  Sample titles   :\n"
        + "\n".join(f"    - {t.strip()}" for t in sample_titles.split("|") if t.strip())
        + f"\n\nReturn a ranked list of up to {TOPIC_MAX_COUNT} canonical financial topics for this cluster (confidence >= {LLM_CONFIDENCE_THRESHOLD}). Return JSON only."
    )


# ══════════════════════════════════════════════════════════════════════════════
# LLM call with retry — ditto same as sttopic_modeling.py._call_llm_with_retry
# ══════════════════════════════════════════════════════════════════════════════
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=60))
def _call_llm_with_retry(messages: List[Dict]) -> str:
    """Call LLM with automatic retry on failure — ditto same decorator as sttopic_modeling.py."""
    global _tok_in, _tok_out
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,   # ditto same as sttopic_modeling.py
        response_format={"type": "json_object"},
    )
    if response.usage:
        with _tok_lock:
            _tok_in  += response.usage.prompt_tokens
            _tok_out += response.usage.completion_tokens
    return response.choices[0].message.content


def classify_cluster(row: dict) -> Tuple[List[Tuple[str, float]], str]:
    """Returns (ranked_topics, reasoning).
    ranked_topics is a list of (topic_name, score) tuples sorted by score descending,
    filtered to score >= LLM_CONFIDENCE_THRESHOLD, deduplicated, max TOPIC_MAX_COUNT."""
    try:
        raw      = _call_llm_with_retry([
            {"role": "system", "content": _get_system_prompt()},
            {"role": "user",   "content": _format_classification_prompt(row)},
        ])
        decision  = json.loads(raw or "{}")
        reasoning = str(decision.get("reasoning", ""))

        raw_topics = decision.get("topics", [])
        if not isinstance(raw_topics, list):
            raw_topics = []

        ranked: List[Tuple[str, float]] = []
        seen_names: set = set()

        for entry in raw_topics:
            if not isinstance(entry, dict):
                continue
            suggested = str(entry.get("topic", "") or "").strip()
            score     = float(entry.get("confidence", 0.0))

            if not suggested or suggested.lower() in ("null", "none", ""):
                continue
            if score < LLM_CONFIDENCE_THRESHOLD:
                continue

            # Snap to nearest canonical — ditto same as sttopic_modeling.py
            topic = map_to_canonical(suggested)
            is_valid, clean, _ = TopicValidator.validate_topic_name(topic)
            topic = clean if is_valid else topic

            # Deduplicate
            if topic in seen_names:
                continue
            seen_names.add(topic)
            ranked.append((topic, round(score, 3)))

        # Sort by score descending, cap at TOPIC_MAX_COUNT
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)[:TOPIC_MAX_COUNT]

        if not ranked:
            ranked = [("[UNCLASSIFIED]", 0.0)]

        return ranked, reasoning

    except Exception as e:
        return [(f"[ERROR: {e}]", 0.0)], ""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print(f"📂 Reading {INPUT_CSV} ...")
df    = pd.read_csv(INPUT_CSV)
total = len(df)
print(f"   {total} clusters loaded\n")

# Pre-warm canonical embedding cache single-threadedly so parallel workers
# don't race to compute the same embeddings simultaneously.
print("⏳ Pre-warming canonical embedding cache ...")
for _cn in CANONICAL_TOPIC_NAMES:
    _get_embedding_cached(_cn)
print(f"   {len(CANONICAL_TOPIC_NAMES)} canonical embeddings cached.\n")


def _classify_row(rank: int, row_dict: dict) -> dict:
    """Classify one cluster — called in parallel by joblib threads."""
    ranked_topics, reasoning = classify_cluster(row_dict)

    # topic_primary = top-ranked topic name (convenience column for quick filtering)
    topic_primary = ranked_topics[0][0] if ranked_topics else ""

    # topics = JSON array of {topic, score} sorted by score desc
    topics_json = json.dumps(
        [{"topic": t, "score": s} for t, s in ranked_topics],
        ensure_ascii=False,
    )

    # Print: show all topics with scores
    topics_str = "  |  ".join(f"{t} ({s:.2f})" for t, s in ranked_topics)
    print(f"  [{rank:>4}/{total}]  id={int(row_dict['cluster_id']):>4}  [{row_dict['confidence']}]")
    print(f"           Topics → {topics_str}")
    print(f"           Why   → {reasoning[:90]}\n")

    return {
        **row_dict,
        "topic_primary" : topic_primary,
        "topics"        : topics_json,
        "topic_reasoning": reasoning,
    }


print(f"⏳ Classifying {total} clusters in parallel (n_jobs={N_JOBS}) ...\n")

rows_with_rank = [
    (rank, row.to_dict())
    for rank, (_, row) in enumerate(df.iterrows(), 1)
]

records = Parallel(n_jobs=N_JOBS, backend="threading")(
    delayed(_classify_row)(rank, row_dict)
    for rank, row_dict in rows_with_rank
)

# ── SAVE ──────────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(records)
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved {len(out_df)} rows → {OUTPUT_CSV}")
print(f"   Columns: {list(out_df.columns)}")

# ── COST SUMMARY ──────────────────────────────────────────────
_cost_in    = (_tok_in  / 1_000_000) * _PRICE_IN
_cost_out   = (_tok_out / 1_000_000) * _PRICE_OUT
_cost_total = _cost_in + _cost_out
print(f"\n{'─' * 40}")
print(f"  💰 topic_modelling.py cost summary")
print(f"{'─' * 40}")
print(f"  Model          : {LLM_MODEL}")
print(f"  Input tokens   : {_tok_in:,}   → ${_cost_in:.5f}")
print(f"  Output tokens  : {_tok_out:,}  → ${_cost_out:.5f}")
print(f"  Total cost     : ${_cost_total:.5f}  (~${_cost_total*30:.3f}/month @ daily runs)")
print(f"{'─' * 40}")
