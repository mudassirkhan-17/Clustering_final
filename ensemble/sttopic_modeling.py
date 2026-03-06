import os
import json
import pickle
import time
import hashlib
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from tqdm import tqdm
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from contextlib import contextmanager

load_dotenv()

# Import your custom logger
from __init__ import logger_object

# Add rate limiting support
try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    # Fallback if tenacity not installed
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def _safe_log_str(s: str, max_len: int = 50) -> str:
    """Make string safe for logging on Windows (avoids charmap codec errors on non-ASCII)."""
    if not s:
        return ""
    return (s[:max_len].encode("ascii", "replace").decode("ascii"))


class DBConnectionPool:
    """Connection pooling with proper retry logic"""
    _pool = None
    _initialized = False
    
    @classmethod
    def init(cls, connection_string: str, min_conns: int = 2, max_conns: int = 20):
        """Initialize connection pool"""
        if cls._initialized:
            return
        
        try:
            cls._pool = pool.SimpleConnectionPool(
                min_conns,
                max_conns,
                connection_string,
                connect_timeout=10
            )
            cls._initialized = True
            logger_object['info'].log(f"database_pool_initialized: min_conns: {min_conns}, max_conns: {max_conns}")
        except Exception as e:
            logger_object['error'].log(f"database_pool_init_failed: {e}")
            raise
    
    @classmethod
    @contextmanager
    def get_connection(cls, timeout: int = 30):
        """Get connection from pool with validation"""
        if cls._pool is None:
            raise RuntimeError("Connection pool not initialized. Call DBConnectionPool.init() first.")
        
        conn = None
        try:
            conn = cls._pool.getconn()
            
            # Validate connection is alive
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.close()
            
            yield conn
            conn.commit()
            
        except psycopg2.OperationalError as e:
            if conn:
                conn.rollback()
                try:
                    conn.close()
                except:
                    pass
            logger_object['error'].log(f"database_operational_error {e}")
            raise
        except Exception as e:
            if conn:
                conn.rollback()
            logger_object['error'].log(f"database_error {e}")
            raise
        finally:
            if conn:
                try:
                    cls._pool.putconn(conn)
                except:
                    pass

class BatchTracker:
    """Track batch submissions for recovery"""
    
    @staticmethod
    def ensure_tracking_table():
        """No-op: tracking table is assumed to already exist."""
        pass
    
    @staticmethod
    def save_batch(batch_id: str, cluster_count: int):
        """Save batch to tracking table"""
        with DBConnectionPool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO topic_batch_tracking (batch_id, status, cluster_count)
                VALUES (%s, 'submitted', %s)
                ON CONFLICT (batch_id) DO UPDATE SET
                    status = 'submitted',
                    submitted_at = CURRENT_TIMESTAMP;
            """, (batch_id, cluster_count))
    
    @staticmethod
    def update_batch(batch_id: str, status: str, error: str = None):
        """Update batch status"""
        with DBConnectionPool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE topic_batch_tracking
                SET status = %s, completed_at = CURRENT_TIMESTAMP, error_message = %s
                WHERE batch_id = %s;
            """, (status, error, batch_id))
    
    @staticmethod
    def get_pending_batches(hours: int = 24) -> List[str]:
        """Get pending batch IDs for recovery"""
        with DBConnectionPool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT batch_id FROM topic_batch_tracking
                WHERE status = 'submitted' AND submitted_at > NOW() - INTERVAL '%s hours'
                ORDER BY submitted_at DESC;
            """, (hours,))
            return [row[0] for row in cursor.fetchall()]

class CostTracker:
    """Track actual API costs"""
    
    def __init__(self):
        self.daily_costs = {}
        self.daily_requests = {}
    
    def track_batch_request(self, cluster_count: int, tokens_estimate: int):
        """Track batch API usage"""
        estimated_cost = tokens_estimate * 0.00015 / 1_000_000
        
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_costs[today] = self.daily_costs.get(today, 0) + estimated_cost
        self.daily_requests[today] = self.daily_requests.get(today, 0) + cluster_count
        
        logger_object['info'].log(f"cost_tracked: clusters={cluster_count}, tokens={tokens_estimate}, cost_usd={estimated_cost:.6f}, daily_total={self.daily_costs[today]:.2f}")
        
        return estimated_cost
    
    def track_semantic_shortcut(self, cluster_count: int):
        """Track free semantic matches"""
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_requests[today] = self.daily_requests.get(today, 0) + cluster_count
        
        logger_object['info'].log(f"semantic_shortcut: clusters_free={cluster_count}, daily_total_free={self.daily_requests.get(today, 0)}")

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

class TopicModelingEngine:
    """Hybrid semantic + LLM with all production fixes + YOUR requirements"""
    
    EMBEDDING_MODEL_VERSION = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION = 768
    # FIX: Correct API key variable name
    LLM_MODEL = os.getenv("TOPIC_LLM_MODEL", "gpt-4o-mini")
    
    # Canonical topics: cap topic count by mapping all LLM suggestions to this list
    # CANONICAL_TOPIC_NAMES = (
    #     "Earnings and Financial Results",
    #     "Guidance and Forecasts",
    #     "Stock Price Movements",
    #     "Dividends and Shareholder Returns",
    #     "Mergers and Acquisitions",
    #     "Partnerships and Strategic Deals",
    #     "Analyst Ratings and Price Targets",
    #     "Product Launches and Innovation",
    #     "Manufacturing and Supply Chain",
    #     "Legal and Regulatory",
    #     "IPO and Capital Markets",
    #     "Monetary Policy and Interest Rates",
    #     "Commodities and Energy",
    #     "ETF and Fund Activity",
    #     "Market Indices and Sector Performance",
    #     "Management and Leadership Changes",
    #     "Real Estate and Property",
    # )

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
    CANONICAL_MAP_THRESHOLD = 0.58
    
    def __init__(self, output_folder: str = "output_folder"):
        # FIX: Correct API key name (removed the 's' at end)
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except TypeError as e:
            if "proxies" in str(e):
                import httpx
                http_client = httpx.Client(timeout=60.0)
                self.client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    http_client=http_client
                )
            else:
                raise
        
        # Thread-safe embedding with executor
        self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL_VERSION)
        self.embedding_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embeddings_")
        
        # FIX #1: Embedding cache to save time
        self.embedding_cache = {}
        
        # Storage
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.topics_path = os.path.join(self.output_folder, "topics.pkl")
        
        # Topics with idempotency tracking
        self.topics: Dict[str, Dict] = {}
        self.topic_name_hashes: Dict[str, str] = {}
        self.topic_index = None
        self.next_topic_id = 0
        
        # Monitoring
        self.cost_tracker = CostTracker()
        self.topics_created_today = 0
        self.auto_create_enabled = True
        
        # Thresholds
        self.SEMANTIC_MATCH_THRESHOLD = 0.75
        self.SEMANTIC_SHORTCUT_THRESHOLD = 0.85
        self.LLM_CONFIDENCE_THRESHOLD = 0.80
        self.MERGE_THRESHOLD = 0.88
        
        # Batch processing (per-request state, protected by lock)
        self.pending_batch_requests = []
        self.article_id_mapping = {}
        self._batch_lock = threading.Lock()
        
        BatchTracker.ensure_tracking_table()
    
    def load_topics(self):
        """Load topics with model version check. Auto-removes bad topics (Uncategorized/Other) from pickle."""
        if os.path.exists(self.topics_path):
            try:
                with open(self.topics_path, "rb") as f:
                    data = pickle.load(f)
                
                saved_model = data.get("metadata", {}).get("embedding_model")
                if saved_model and saved_model != self.EMBEDDING_MODEL_VERSION:
                    logger_object['error'].log(f"model_mismatch: saved={saved_model}, current={self.EMBEDDING_MODEL_VERSION}")
                    raise ValueError(f"Model mismatch: {saved_model} != {self.EMBEDDING_MODEL_VERSION}")
                
                self.topics = data.get("topics", {})
                self.next_topic_id = data.get("next_id", 0)
                
                bad_tids = [
                    tid for tid, t in self.topics.items()
                    if (t.get("name") or "").strip().startswith("Uncategorized:")
                    or (t.get("name") or "").strip().lower() == "other"
                ]
                if bad_tids:
                    for tid in bad_tids:
                        del self.topics[tid]
                    logger_object['info'].log(f"auto_cleaned_bad_topics: removed={len(bad_tids)}, remaining={len(self.topics)}")
                
                self._rebuild_topic_index()
                self._rebuild_name_hashes()
                
                logger_object['info'].log(f"topics_loaded: count={len(self.topics)}, model={self.EMBEDDING_MODEL_VERSION}")
            except Exception as e:
                logger_object['error'].log(f"topics_load_failed: error={str(e)}")
                raise
        else:
            logger_object['info'].log(f"topics_fresh_start: folder={self.output_folder}")
    
    def save_topics(self):
        """Save topics to pickle file"""
        self._validate_index_consistency()
        data = {
            "topics": self.topics,
            "next_id": self.next_topic_id,
            "metadata": {
                "embedding_model": self.EMBEDDING_MODEL_VERSION,
                "embedding_dimension": self.EMBEDDING_DIMENSION,
                "saved_at": datetime.now().isoformat(),
                "total_topics": len(self.topics)
            }
        }
        with open(self.topics_path, "wb") as f:
            pickle.dump(data, f)
        logger_object['info'].log(f"topics_saved: count={len(self.topics)}, path={self.topics_path}")
    
    # FIX #2: Embedding cache - save time!
    def _get_embedding_cached(self, text: str) -> np.ndarray:
        """Get embedding with caching to avoid duplicate work"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash not in self.embedding_cache:
            # Compute embedding only once per unique text
            embedding = self._get_article_embedding({"title": "", "text": text})
            self.embedding_cache[text_hash] = embedding
        
        return self.embedding_cache[text_hash]
    
    def _map_to_canonical_topic(self, suggested_name: str) -> str:
        """Map LLM-suggested topic name to nearest canonical name.
        If no canonical matches above threshold, return the suggested name as-is (never 'Other')."""
        if not suggested_name or not suggested_name.strip():
            return suggested_name or ""
        clean = suggested_name.strip()
        suggested_emb = self._get_embedding_cached(clean)
        best_name = None
        best_sim = self.CANONICAL_MAP_THRESHOLD
        for canonical in self.CANONICAL_TOPIC_NAMES:
            canonical_emb = self._get_embedding_cached(canonical)
            sim = float(np.dot(suggested_emb, canonical_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = canonical
        if best_name and best_name != clean:
            logger_object['info'].log(f"canonical_mapping: suggested={clean[:50]}, mapped_to={best_name}, sim={round(best_sim, 4)}")
            return best_name
        return clean
    
    def assign_topics_batch(self, cluster_representatives: List[Dict]) -> Dict[int, List[str]]:
        """Assign topics with batch recovery support. Returns idx -> [topic_id, ...] (multi-topic).
        Thread-safe: only one batch processes at a time to prevent article_id_mapping race."""
        if not cluster_representatives:
            return {}
        
        with self._batch_lock:
            MAX_CLUSTERS_PER_RUN = 5000
            if len(cluster_representatives) > MAX_CLUSTERS_PER_RUN:
                logger_object['error'].log(f"cluster_limit_exceeded: requested={len(cluster_representatives)}, limit={MAX_CLUSTERS_PER_RUN}")
                cluster_representatives = cluster_representatives[:MAX_CLUSTERS_PER_RUN]
            
            logger_object['info'].log(f"batch_assignment_started: clusters={len(cluster_representatives)}")
            
            assignments = {}
            semantic_shortcuts = 0
            self.pending_batch_requests = []
            self.article_id_mapping = {}
            
            for idx, article in enumerate(tqdm(cluster_representatives, desc="Processing")):
                article_embedding = self._get_article_embedding(article)
                candidates = self._find_candidate_topics(article_embedding)
                
                if candidates and candidates[0][1] >= self.SEMANTIC_SHORTCUT_THRESHOLD:
                    topic_id = candidates[0][0]
                    self.topics[topic_id]["article_count"] = self.topics[topic_id].get("article_count", 0) + 1
                    self.topics[topic_id]["updated_at"] = datetime.now().isoformat()
                    assignments[idx] = [topic_id]
                    semantic_shortcuts += 1
                    continue
                
                self._queue_batch_request(idx, article, candidates)
            
            if semantic_shortcuts > 0:
                self.cost_tracker.track_semantic_shortcut(semantic_shortcuts)
            
            REALTIME_THRESHOLD = 20
            if self.pending_batch_requests:
                if len(self.pending_batch_requests) <= REALTIME_THRESHOLD:
                    batch_assignments = self._process_realtime()
                else:
                    batch_assignments = self._process_batch()
                assignments.update(batch_assignments)
            
            logger_object['info'].log(f"batch_assignment_completed: total_assigned={len(assignments)}, semantic_shortcuts={semantic_shortcuts}, llm_processed={len(self.pending_batch_requests)}")
            
            return assignments
    
    def _queue_batch_request(self, idx: int, article: Dict, candidates: List[Tuple[str, float]]):
        """Queue article for batch processing"""
        custom_id = f"cluster{idx}_{int(time.time() * 1000)}"
        
        self.article_id_mapping[custom_id] = {
            "article": article,
            "candidates": candidates,
            "idx": idx
        }

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": self._format_classification_prompt(article, candidates)}
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
        }
        self.pending_batch_requests.append(request)
    
    # FIX #3: Rate limiting with retry decorator
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=60))
    def _call_llm_with_retry(self, messages: List[Dict]) -> str:
        """Call LLM with automatic retry on failure"""
        response = self.client.chat.completions.create(
            model=self.LLM_MODEL,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    def _process_realtime(self) -> Dict[int, List[str]]:
        """Process LLM requests using regular Chat API (fast, for small batches)"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger_object['info'].log(f"realtime_processing_started: requests={len(self.pending_batch_requests)}")
        start_time = time.time()
        
        def call_llm(request):
            custom_id = request["custom_id"]
            body = request["body"]
            try:
                content = self._call_llm_with_retry(body["messages"])
                return {
                    "custom_id": custom_id,
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": content}}]
                        }
                    }
                }
            except Exception as e:
                logger_object['error'].log(f"realtime_llm_call_failed: custom_id={custom_id}, error={str(e)}")
                return None
        
        results = []
        max_workers = min(5, len(self.pending_batch_requests))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(call_llm, req): req for req in self.pending_batch_requests}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        elapsed = time.time() - start_time
        logger_object['info'].log(f"realtime_processing_completed: requests={len(self.pending_batch_requests)}, successful={len(results)}, elapsed_seconds={round(elapsed, 2)}")
        
        results_text = "\n".join(json.dumps(r) for r in results)
        return self._parse_batch_results(results_text)
    
    def _process_batch(self, timeout_minutes: int = 30) -> Dict[int, List[str]]:
        """Submit batch with recovery mechanism"""
        batch_requests_path = os.path.join(self.output_folder, f"batch_requests_{int(time.time())}.jsonl")
        
        try:
            with open(batch_requests_path, "w") as f:
                for req in self.pending_batch_requests:
                    f.write(json.dumps(req) + "\n")
        except (IOError, OSError) as e:
            logger_object['error'].log(f"batch_file_write_failed: error={str(e)}")
            return self._fallback_to_embedding_only()
        
        try:
            try:
                with open(batch_requests_path, "rb") as f:
                    batch_file = self.client.files.create(file=f, purpose="batch")
            except (IOError, OSError) as e:
                logger_object['error'].log(f"batch_file_read_failed: error={str(e)}")
                return self._fallback_to_embedding_only()
            
            batch_start_time = time.time()
            batch = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            BatchTracker.save_batch(batch.id, len(self.pending_batch_requests))
            
            logger_object['info'].log(f"batch_submitted: batch_id={batch.id}, requests={len(self.pending_batch_requests)}, file_id={batch_file.id}")
            
            poll_count = 0
            max_polls = timeout_minutes * 6
            timeout_seconds = timeout_minutes * 60
            
            while batch.status not in ["completed", "failed", "cancelled"]:
                elapsed_time = time.time() - batch_start_time
                if elapsed_time > timeout_seconds:
                    logger_object['error'].log(f"batch_absolute_timeout: batch_id={batch.id}, elapsed_seconds={int(elapsed_time)}, timeout_seconds={timeout_seconds}")
                    BatchTracker.update_batch(batch.id, "timeout")
                    return self._fallback_to_embedding_only()
                
                if poll_count >= max_polls:
                    logger_object['error'].log(f"batch_poll_timeout: batch_id={batch.id}, poll_count={poll_count}, max_polls={max_polls}")
                    BatchTracker.update_batch(batch.id, "timeout")
                    return self._fallback_to_embedding_only()
                
                time.sleep(10)
                
                try:
                    batch = self.client.batches.retrieve(batch.id)
                except Exception as e:
                    logger_object['error'].log(f"batch_retrieve_failed: error={str(e)}, poll_count={poll_count}")
                    if poll_count > 3:
                        return self._fallback_to_embedding_only()
                
                poll_count += 1
                
                if poll_count % 6 == 0:
                    logger_object['info'].log(f"batch_polling: batch_id={batch.id}, status={batch.status}, elapsed_minutes={poll_count / 6}")
            
            if batch.status != "completed":
                logger_object['error'].log(f"batch_failed: batch_id={batch.id}, status={batch.status}")
                BatchTracker.update_batch(batch.id, batch.status)
                return self._fallback_to_embedding_only()
            
            logger_object['info'].log(f"batch_completed: batch_id={batch.id}, output_file_id={batch.output_file_id}")
            
            result_content = self.client.files.content(batch.output_file_id)
            BatchTracker.update_batch(batch.id, "completed")
            self._cleanup_old_batch_files(days_to_keep=7)
            
            return self._parse_batch_results(result_content.text)
            
        except Exception as e:
            logger_object['error'].log(f"batch_processing_error: error={str(e)}, requests={len(self.pending_batch_requests)}")
            return self._fallback_to_embedding_only()
    
    def _parse_batch_results(self, results_text: str) -> Dict[int, List[str]]:
        """Parse batch results with LLM-generated production-ready names"""
        assignments = {}
        results = results_text.strip().split("\n")
        
        for line in results:
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id")
                
                if custom_id not in self.article_id_mapping:
                    logger_object['error'].log(f"unknown_custom_id: custom_id={custom_id}")
                    continue
                
                article_data = self.article_id_mapping[custom_id]
                idx = article_data["idx"]
                article = article_data["article"]
                
                response_text = result["response"]["body"]["choices"][0]["message"]["content"]
                decision = json.loads(response_text)
                
                topic_entries = decision.get("topics")
                if not topic_entries:
                    topic_entries = [decision]
                
                logger_object['info'].log(f"llm_response_parsed: article={_safe_log_str(article.get('title', 'Unknown'), 50)}, entries={len(topic_entries)}")
                
                assigned_topic_ids = self._resolve_topic_entries(topic_entries, article)
                
                if assigned_topic_ids:
                    assignments[idx] = assigned_topic_ids
                else:
                    logger_object['error'].log(f"unable_to_classify: article={_safe_log_str(article.get('title', 'Unknown'), 50)}")
            
            except Exception as e:
                logger_object['error'].log(f"result_parse_error: error={_safe_log_str(str(e), 200)}")
                if "idx" in locals():
                    try:
                        article_emb = self._get_article_embedding(article)
                        fallback_id = self._create_fallback_topic(article, article_emb)
                        assignments[idx] = [fallback_id]
                    except:
                        logger_object['error'].log(f"fallback_creation_failed: idx={idx}")

        return assignments
    
    def _resolve_topic_entries(self, topic_entries: List[Dict], article: Dict) -> List[str]:
        """Resolve a list of topic decisions (multi-topic) into a list of topic IDs"""
        assigned_ids = []
        
        for entry in topic_entries[:3]:
            confidence = entry.get("confidence", 0)
            raw_decision = str(entry.get("decision", "")).strip().lower()
            
            if raw_decision in ("existing", "assign", "assign_existing", "reuse", "match"):
                decision = "existing"
            elif raw_decision in ("new", "create", "create_new", "add", "none"):
                decision = "new"
            elif entry.get("topic_id") is not None and str(entry.get("topic_id", "")) not in ("", "null", "None"):
                decision = "existing"
            elif entry.get("new_topic_name"):
                decision = "new"
            else:
                decision = raw_decision
            
            topic_id_val = entry.get("topic_id")
            has_valid_topic_id = topic_id_val is not None and str(topic_id_val).strip() not in ("", "null", "None")
            if decision == "existing" and not has_valid_topic_id:
                decision = "new"
                if not entry.get("new_topic_name"):
                    entry["new_topic_name"] = (
                        entry.get("topic_name") or entry.get("name") or entry.get("matched_topic")
                        or (entry.get("description") or "")[:60].strip()
                        or ""
                    )
            
            if decision == "existing" and entry.get("topic_id"):
                topic_id = str(entry["topic_id"])
                if topic_id in self.topics:
                    self.topics[topic_id]["article_count"] += 1
                    self.topics[topic_id]["updated_at"] = datetime.now().isoformat()
                    assigned_ids.append(topic_id)
                    logger_object['info'].log(f"assigned_to_existing: topic_id={topic_id}, topic_name={self.topics[topic_id]['name']}, confidence={confidence}")
                    continue
            
            if decision == "new" or (decision == "existing" and entry.get("topic_id") and str(entry["topic_id"]) not in self.topics):
                new_name = entry.get("new_topic_name") or ""
                description = entry.get("description") or ""
                
                is_valid, clean_name, error = TopicValidator.validate_topic_name(new_name)
                
                if is_valid and confidence >= self.LLM_CONFIDENCE_THRESHOLD:
                    clean_name = self._map_to_canonical_topic(clean_name)
                    name_emb = self._get_embedding_cached(clean_name)
                    
                    best_match_id = None
                    best_match_sim = 0.0
                    
                    for existing_tid, existing_tdata in self.topics.items():
                        existing_name_emb = self._get_embedding_cached(existing_tdata["name"])
                        sim = float(np.dot(name_emb, existing_name_emb))
                        if sim > best_match_sim:
                            best_match_sim = sim
                            best_match_id = existing_tid
                    
                    if best_match_id and best_match_sim >= 0.70:
                        if best_match_id not in assigned_ids:
                            self.topics[best_match_id]["article_count"] += 1
                            self.topics[best_match_id]["updated_at"] = datetime.now().isoformat()
                            assigned_ids.append(best_match_id)
                            logger_object['info'].log(f"reused_similar_topic: suggested={clean_name}, matched={self.topics[best_match_id]['name']}, sim={round(best_match_sim, 4)}")
                        continue
                    
                    name_hash = hashlib.md5(clean_name.lower().encode()).hexdigest()
                    if name_hash in self.topic_name_hashes:
                        topic_id = self.topic_name_hashes[name_hash]
                        if topic_id not in assigned_ids:
                            self.topics[topic_id]["article_count"] += 1
                            assigned_ids.append(topic_id)
                        continue
                    
                    article_emb = self._get_article_embedding(article)
                    topic_id = self._create_topic(clean_name, description, article_emb)
                    assigned_ids.append(topic_id)
                    logger_object['info'].log(f"topic_created_from_llm: topic_id={topic_id}, topic_name={clean_name}, description={description[:100]}, confidence={confidence}")
                    continue
                
                elif not is_valid:
                    logger_object['info'].log(f"invalid_name_fallback: suggested={new_name}, error={error}")
                    article_emb = self._get_article_embedding(article)
                    fallback_id = self._create_fallback_topic(article, article_emb)
                    if fallback_id not in assigned_ids:
                        assigned_ids.append(fallback_id)
                    continue
                else:
                    clean_name = self._map_to_canonical_topic(clean_name)
                    article_emb = self._get_article_embedding(article)
                    topic_id = self._create_topic(clean_name, description, article_emb)
                    assigned_ids.append(topic_id)
                    logger_object['info'].log(f"topic_created_moderate_confidence: topic_id={topic_id}, topic_name={clean_name}, confidence={confidence}")
                    continue
            
            if confidence < 0.5:
                article_emb = self._get_article_embedding(article)
                fallback_id = self._create_fallback_topic(article, article_emb)
                if fallback_id not in assigned_ids:
                    assigned_ids.append(fallback_id)
        
        return assigned_ids
    
    def _fallback_to_embedding_only(self) -> Dict[int, List[str]]:
        """Fallback when batch fails"""
        logger_object['error'].log(f"fallback_to_embedding_only: pending_requests={len(self.article_id_mapping)}")
        
        assignments = {}
        for custom_id, data in self.article_id_mapping.items():
            idx = data["idx"]
            candidates = data["candidates"]
            article = data["article"]
            
            if candidates:
                self.topics[candidates[0][0]]["article_count"] += 1
                assignments[idx] = [candidates[0][0]]
            else:
                article_emb = self._get_article_embedding(article)
                fallback_id = self._create_fallback_topic(article, article_emb)
                assignments[idx] = [fallback_id]
        
        return assignments
    
    def _get_article_embedding(self, article: Dict) -> np.ndarray:
        """Thread-safe embedding"""
        text = f"{article.get('title', '')} {article.get('text', '')}"
        text = TopicValidator.sanitize_prompt_input(text, max_length=1000)
        
        future = self.embedding_executor.submit(self.embedding_model.encode, [text])
        try:
            emb = future.result(timeout=30)[0]
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as e:
            logger_object['error'].log(f"embedding_error: error={str(e)}")
            raise
    
    def _find_candidate_topics(self, embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Find similar topics"""
        if self.topic_index is None or self.topic_index.ntotal == 0:
            return []
        
        D, I = self.topic_index.search(
            embedding.reshape(1, -1).astype("float32"),
            min(k, self.topic_index.ntotal)
        )
        
        candidates = []
        for sim, topic_idx in zip(D[0], I[0]):
            if sim >= self.SEMANTIC_MATCH_THRESHOLD:
                topic_id = str(int(topic_idx))
                if topic_id in self.topics:
                    candidates.append((topic_id, float(sim)))
        
        return candidates
    
    def _rebuild_topic_index(self):
        """Rebuild FAISS index safely"""
        if not self.topics:
            dim = self.EMBEDDING_DIMENSION
            self.topic_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            return
        
        embeddings = []
        topic_ids = []
        
        for topic_id, topic_data in self.topics.items():
            embeddings.append(topic_data["embedding"])
            topic_ids.append(int(topic_id))
        
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        
        new_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        new_index.add_with_ids(embeddings, np.array(topic_ids).astype("int64"))
        
        self.topic_index = new_index
        
        logger_object['info'].log(f"index_rebuilt: topics={len(self.topics)}")
    
    def _cleanup_old_batch_files(self, days_to_keep: int = 7):
        """Auto-cleanup batch request files older than N days"""
        import glob
        
        try:
            cutoff_time = time.time() - (days_to_keep * 86400)
            pattern = os.path.join(self.output_folder, "batch_requests_*.jsonl")
            batch_files = glob.glob(pattern)
            
            deleted_count = 0
            for file_path in batch_files:
                try:
                    file_mtime = os.path.getmtime(file_path)
                    if file_mtime < cutoff_time:
                        os.remove(file_path)
                        deleted_count += 1
                        logger_object['info'].log(f"batch_file_deleted: file={os.path.basename(file_path)}, age_days={int((time.time() - file_mtime) / 86400)}")
                except Exception as e:
                    logger_object['error'].log(f"batch_file_delete_failed: file={os.path.basename(file_path)}, error={str(e)}")

            if deleted_count > 0:
                logger_object['info'].log(f"batch_files_cleaned: deleted={deleted_count}, kept_days={days_to_keep}")
        except Exception as e:
            logger_object['error'].log(f"batch_cleanup_error: error={str(e)}")
    
    def _validate_index_consistency(self):
        """Verify FAISS matches in-memory topics"""
        if self.topic_index is None or self.topic_index.ntotal == 0:
            if self.topics:
                logger_object['error'].log(f"index_consistency_error: empty_index_but_topics_exist, topics={len(self.topics)}")
                self._rebuild_topic_index()
            return
        
        memory_count = len(self.topics)
        
        if self.topic_index.ntotal != memory_count:
            logger_object['error'].log(f"index_corruption_detected: faiss_count={self.topic_index.ntotal}, memory_count={memory_count}")
            self._rebuild_topic_index()
    
    def _rebuild_name_hashes(self):
        """Rebuild name deduplication hashes"""
        self.topic_name_hashes = {}
        for topic_id, topic_data in self.topics.items():
            name_hash = hashlib.md5(topic_data["name"].lower().encode()).hexdigest()
            self.topic_name_hashes[name_hash] = topic_id
    
    def _create_topic(self, name: str, description: str = "", article_embedding: Optional[np.ndarray] = None) -> str:
        """Create new topic"""
        name_hash = hashlib.md5(name.lower().encode()).hexdigest()
        if name_hash in self.topic_name_hashes:
            existing_id = self.topic_name_hashes[name_hash]
            logger_object['info'].log(f"topic_reused: name={name}, existing_id={existing_id}")
            return existing_id
        
        self.topics_created_today += 1
        if self.topics_created_today > 50 and not self.auto_create_enabled:
            logger_object['error'].log(f"topic_explosion_detected: topics_today={self.topics_created_today}, auto_create_disabled={self.auto_create_enabled}")
            raise RuntimeError(f"Topic creation rate exceeded. Auto-create disabled. topics_today={self.topics_created_today}")
        
        topic_id = str(self.next_topic_id)
        self.next_topic_id += 1
        
        if article_embedding is not None:
            topic_embedding = article_embedding
        else:
            topic_embedding = self.embedding_model.encode([name])[0]
            topic_embedding = topic_embedding / np.linalg.norm(topic_embedding)
        
        self.topics[topic_id] = {
            "name": name,
            "description": description,
            "embedding": topic_embedding,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "article_count": 1
        }
        
        if self.topic_index is None:
            dim = len(topic_embedding)
            self.topic_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        
        self.topic_index.add_with_ids(
            topic_embedding.reshape(1, -1).astype("float32"),
            np.array([int(topic_id)]).astype("int64")
        )
        
        self.topic_name_hashes[name_hash] = topic_id
        
        logger_object['info'].log(f"topic_created: topic_id={topic_id}, topic_name={name}, total_topics={len(self.topics)}")
        
        return topic_id
    
    def _get_llm_generic_topic_name(self, article: Dict) -> Optional[str]:
        """Ask LLM for one short generic topic name for this article."""
        title = (article.get("title") or "").strip()[:200]
        text = TopicValidator.sanitize_prompt_input((article.get("text") or "")[:400], max_length=400)
        if not title and not text:
            return None
        prompt = f"""Based on this article, suggest ONE short generic topic name (2-6 words) that best categorizes it. Reply with only the topic name, nothing else. Do not use "Other" or "Uncategorized".

Title: {title}
Content: {text}"""
        try:
            response = self.client.chat.completions.create(
                model=self.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            raw = (response.choices[0].message.content or "").strip()
            raw = raw.split("\n")[0].strip().strip('"').strip("'")[:100]
            if not raw or raw.lower() == "other" or "uncategorized" in raw.lower():
                return None
            return raw
        except Exception as e:
            logger_object['error'].log(f"llm_generic_name_failed: error={_safe_log_str(str(e), 200)}")
            return None

    def _create_fallback_topic(self, article: Dict, article_embedding: np.ndarray) -> str:
        """Ask LLM for best generic topic name. Always creates a real topic, never 'Other'."""
        suggested = self._get_llm_generic_topic_name(article)
        if suggested:
            is_valid, clean_name, _ = TopicValidator.validate_topic_name(suggested)
            if is_valid and clean_name and clean_name.lower() != "other":
                mapped = self._map_to_canonical_topic(clean_name)
                name_hash = hashlib.md5(mapped.lower().encode()).hexdigest()
                if name_hash in self.topic_name_hashes:
                    tid = self.topic_name_hashes[name_hash]
                    self.topics[tid]["article_count"] = self.topics[tid].get("article_count", 0) + 1
                    self.topics[tid]["updated_at"] = datetime.now().isoformat()
                    return tid
                return self._create_topic(mapped, f"Articles about {mapped}.", article_embedding)

        candidates = self._find_candidate_topics(article_embedding, k=1)
        if candidates:
            tid = candidates[0][0]
            self.topics[tid]["article_count"] = self.topics[tid].get("article_count", 0) + 1
            self.topics[tid]["updated_at"] = datetime.now().isoformat()
            return tid

        return self._create_topic("General News", "Articles that do not fit specific financial categories.", article_embedding)

    def merge_topics(self, tid1: str, tid2: str, new_name: Optional[str] = None):
        """Merge topics safely"""
        if tid1 not in self.topics or tid2 not in self.topics:
            logger_object['error'].log(f"merge_topic_not_found: tid1={tid1}, tid2={tid2}")
            return
        
        try:
            if new_name:
                is_valid, clean_name, _ = TopicValidator.validate_topic_name(new_name)
                if is_valid:
                    self.topics[tid1]["name"] = clean_name
                    new_emb = self.embedding_model.encode([clean_name])[0]
                    self.topics[tid1]["embedding"] = new_emb / np.linalg.norm(new_emb)
            
            self.topics[tid1]["article_count"] += self.topics[tid2].get("article_count", 0)
            self.topics[tid1]["updated_at"] = datetime.now().isoformat()
            
            del self.topics[tid2]
            self._rebuild_topic_index()
            self._rebuild_name_hashes()
            
            logger_object['info'].log(f"topics_merged: tid1={tid1}, tid2={tid2}, new_name={new_name}")
        except Exception as e:
            logger_object['error'].log(f"merge_failed: tid1={tid1}, tid2={tid2}, error={str(e)}")
            raise

    def _get_system_prompt(self) -> str:
        """System prompt: generic topics suitable for all financial news."""
        return """You are a financial news topic classifier. Use GENERIC topics that suit all types of news.

CANONICAL TOPIC LIST (use these exact names in new_topic_name — these are NOT IDs):
- Earnings
- Guidance
- Earnings Call / Transcript
- Guidance Cut / Profit Warning
- M&A
- Deal Rumors / Talks
- IPO / SPAC
- Offerings (Equity/Debt)
- Secondary Offering / ATM
- Debt Issuance / Refinancing
- Credit Rating Change (Corp)
- Dividends
- Buybacks
- Insider Trades
- Shareholder Activism
- Management / Board
- Layoffs / Hiring
- Restructuring / Cost Cuts
- Bankruptcy / Restructuring
- Litigation / Lawsuits
- Regulation / Legal
- Antitrust / Competition
- SEC / Accounting / Restatement
- Investigations / Fraud
- Product Launch / Updates
- Recalls / Safety
- Supply Chain / Operations
- Partnerships / Contracts
- Government Contracts
- Guidance Beat / Raise
- Spin-off / Split / Reorg
- Delisting / Compliance (Listing)
- Shareholder Vote / Proxy
- Capital Return (General)
- Buyout / LBO / PE Activity
- Market Wrap
- Pre-Market
- After Hours
- Movers
- Unusual Volume
- Order Flow / Options Flow
- Short Interest / Short Squeeze
- Volatility
- VIX / Vol Products
- ETFs
- Options
- Futures
- Bonds / Credit
- Treasuries
- Investment Grade Credit
- High Yield / Distressed
- Credit Spreads
- Commodities
- Oil / OPEC
- Natural Gas
- Metals (Industrial)
- Gold (Precious Metals)
- Agriculture / Softs
- FX
- USD / DXY
- EM FX
- Crypto
- Stablecoins
- DeFi
- Rates / Curve (Market)
- Liquidity / Funding Stress
- Upgrades
- Downgrades
- Initiations
- Price Targets
- Stock Ratings
- Analyst Color
- Earnings Preview
- Earnings Recap (Street Takeaways)
- Estimate Changes / Revisions
- Consensus / Whisper Numbers
- Research Notes / Deep Dives
- Quant / Factor Notes
- Valuation / Multiples
- Channel Checks
- Central Banks
- Fed Speak / Minutes
- Interest Rates
- Inflation
- Jobs / Labor
- GDP / Growth
- PMI / ISM
- Retail Sales / Consumption
- Housing / Real Estate Macro
- Fiscal Policy / Budget / Treasury
- Politics
- Elections
- Geopolitics
- Sanctions / Export Controls
- Trade Policy / Tariffs
- Energy Security / Supply Shocks
- China Macro
- Emerging Markets Macro
- Recession / Soft Landing
- Market Risk-On / Risk-Off

RULES:
- PREFER EXISTING: If the EXISTING CANDIDATE TOPICS block (in the user message) contains a topic matching a canonical name above, use "existing" and copy its [id] as topic_id. DO NOT use canonical list position numbers as topic_id.
- NEW TOPIC: If no existing candidate matches, use "new" and set new_topic_name to the EXACT name from the canonical list above (e.g. "Earnings", "M&A", "Layoffs / Hiring").
- If the article does NOT fit any canonical topic, suggest a short generic topic name (2-6 words) as new_topic_name. NEVER use "Other" or "Uncategorized".
- 1-3 topics per article if it genuinely covers multiple types. Most articles = 1 topic.
- NO company names or tickers in topic names.

OUTPUT (JSON only):
{"topics": [{"decision": "existing"|"new", "topic_id": <copy [id] from EXISTING CANDIDATE TOPICS block if existing, else null>, "new_topic_name": "<EXACT canonical name from list above if new, else null>", "description": "<1-2 sentences if new>", "confidence": <0.0-1.0>}], "reasoning": "<brief>"}"""
            
    def _format_classification_prompt(self, article: Dict, candidates: List[Tuple[str, float]]) -> str:
        """Enhanced prompt with full context"""
        title = TopicValidator.sanitize_prompt_input(article.get('title', ''), 200)
        text = TopicValidator.sanitize_prompt_input(article.get('text', ''), 800)
        symbol = article.get('symbol', 'N/A')
        sentiment = article.get('sentiment', 'N/A')
        article_date = article.get('date', datetime.now().isoformat())[:10]
        
        candidates_text = ""
        if candidates:
            candidates_text = "EXISTING CANDIDATE TOPICS (ranked by similarity):\n"
            for i, (tid, sim) in enumerate(candidates[:5], 1):
                topic = self.topics[tid]
                candidates_text += f"{i}. [{tid}] {topic['name']}\n"
                candidates_text += f"   Similarity: {sim:.3f} | Articles: {topic.get('article_count', 0)}\n"
                if topic.get('description') and topic['description'] != "Needs manual review":
                    candidates_text += f"   Description: {topic['description'][:100]}\n"
        else:
            candidates_text = "No existing candidate topics found (likely need to create new topic).\n"
        
        return f"""ARTICLE TO CLASSIFY:
Date: {article_date}
Title: {title}
Symbol: {symbol}
Sentiment: {sentiment}

Full Text:
{text}

---

{candidates_text}

TASK:
1. Map this article to one or more topics from the CANONICAL list (100 topics).
2. If an existing candidate above matches that canonical topic, use "existing" with its topic_id.
3. If no existing topic matches but the article fits a canonical type, use "new" with the EXACT name from the list.
4. If the article does NOT fit any of the 100 canonical topics, suggest a short generic topic name (2-6 words). NEVER use "Other" or "Uncategorized".
5. 1-3 topics per article max. Confidence >= 0.80 for new.

Return ONLY valid JSON."""
    
    def detect_duplicates(self, threshold: float = 0.88) -> List[Tuple[str, str, float]]:
        """Find duplicate topics"""
        duplicates = []
        topic_ids = list(self.topics.keys())
        
        for i, tid1 in enumerate(topic_ids):
            emb1 = self.topics[tid1]["embedding"]
            for tid2 in topic_ids[i+1:]:
                emb2 = self.topics[tid2]["embedding"]
                sim = float(np.dot(emb1, emb2))
                if sim >= threshold:
                    duplicates.append((tid1, tid2, sim))
        
        return duplicates
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        if not self.topics:
            return {"total_topics": 0, "total_articles": 0}
        
        counts = [t.get("article_count", 0) for t in self.topics.values()]
        return {
            "total_topics": len(self.topics),
            "total_articles": sum(counts),
            "avg_articles_per_topic": np.mean(counts) if counts else 0,
            "singleton_topics": sum(1 for c in counts if c == 1),
            "review_topics": sum(1 for t in self.topics.values() if "_REVIEW_" in t["name"])
        }

class TopicDBManager:
    """Database operations for topics"""
    
    @staticmethod
    def ensure_database_setup():
        """No-op: tables assumed to already exist"""
        pass
    
    @staticmethod
    def upsert_topics(topics: Dict[str, Dict]):
        """Bulk upsert topics to database"""
        if not topics:
            return
        
        try:
            with DBConnectionPool.get_connection() as conn:
                cursor = conn.cursor()
                
                values = [
                    (
                        int(tid),
                        td.get("name", ""),
                        td.get("description", ""),
                        td.get("article_count", 0),
                        td.get("created_at"),
                        td.get("updated_at"),
                        "_REVIEW_" in td.get("name", ""),
                        'active'
                    )
                    for tid, td in topics.items()
                ]
                
                query = """
                    INSERT INTO topics (topic_id, topic_name, topic_description, article_count,
                                      created_at, updated_at, is_review, status)
                    VALUES %s
                    ON CONFLICT (topic_id) DO UPDATE SET
                        topic_name = EXCLUDED.topic_name,
                        article_count = EXCLUDED.article_count,
                        updated_at = EXCLUDED.updated_at,
                        is_review = EXCLUDED.is_review;
                """
                
                execute_values(cursor, query, values)
                logger_object['info'].log(f"topics_upserted: count={len(values)}")
        except Exception as e:
            logger_object['error'].log(f"topics_upsert_failed: error={str(e)}")
            raise

    @staticmethod
    def upsert_articles_with_topics(articles_with_topics: List[Dict]) -> None:
        if not articles_with_topics:
            return
        try:
            with DBConnectionPool.get_connection() as conn:
                cursor = conn.cursor()
                values = [
                    (
                        a.get("title", ""),
                        a.get("text", ""),
                        a.get("symbol", "N/A"),
                        a.get("sentiment", "neutral"),
                        a.get("topic", []),
                    )
                    for a in articles_with_topics
                ]
                query = """
                    INSERT INTO classified_articles (title, description, symbol, sentiment, topic)
                    VALUES %s
                """
                execute_values(cursor, query, values)
                conn.commit()
                logger_object['info'].log(f"articles_with_topics_saved: count={len(values)}")
        except Exception as e:
            logger_object['error'].log(f"articles_with_topics_save_failed: error={str(e)}")
            raise

# ============================================================================
# PUBLIC API
# ============================================================================

_engine_singleton: Optional[TopicModelingEngine] = None

def get_topic_engine() -> TopicModelingEngine:
    """Get singleton engine instance"""
    global _engine_singleton
    
    if _engine_singleton is None:
        logger_object['info'].log("initializing_topic_engine")
        
        db_string = os.getenv("OFFICEFIELD_LOCAL_SUPABASE_STRING")
        DBConnectionPool.init(db_string)
        
        TopicDBManager.ensure_database_setup()
        
        _engine_singleton = TopicModelingEngine()
        _engine_singleton.load_topics()
        
        logger_object['info'].log("topic_engine_ready")
    
    return _engine_singleton

def assign_topics_to_clusters(cluster_update_ids: List[int], cluster_info: Dict) -> Dict[int, List[str]]:
    """Main function to assign topics to clusters.
    
    FIX: Returns cluster_id -> [topic_NAMES, ...] (not IDs!)
    """
    if not cluster_update_ids:
        return {}
    
    try:
        engine = get_topic_engine()
        
        representatives = []
        cluster_mapping = {}
        
        for cluster_id in cluster_update_ids:
            if cluster_id not in cluster_info:
                continue
            
            cluster_data = cluster_info[cluster_id]
            articles = cluster_data.get("Articles", [])
            
            if articles:
                representative = articles[0]
                idx = len(representatives)
                cluster_mapping[idx] = cluster_id
                representatives.append(representative)
        
        if not representatives:
            logger_object['error'].log(f"no_valid_clusters: count={len(representatives)}")
            return {}
        
        logger_object['info'].log(f"processing_clusters: count={len(representatives)}")
        
        # Get topic IDs
        representative_assignments = engine.assign_topics_batch(representatives)
        
        # FIX: Map IDs to NAMES before returning!
        cluster_topics_with_names = {}
        for idx, topic_ids in representative_assignments.items():
            cluster_id = cluster_mapping[idx]
            topic_names = [
                engine.topics[tid]["name"]
                for tid in topic_ids if tid in engine.topics
            ]
            cluster_topics_with_names[cluster_id] = topic_names
        
        TopicDBManager.upsert_topics(engine.topics)
        engine.save_topics()
        
        logger_object['info'].log(f"assignment_successful: clusters={len(cluster_topics_with_names)}, total_topics={len(engine.topics)}")
        
        # Return topic NAMES, not IDs
        return cluster_topics_with_names
        
    except Exception as e:
        logger_object['error'].log(f"assignment_failed: error={str(e)}")
        raise

def run_topic_maintenance():
    """Weekly maintenance task"""
    logger_object['info'].log("maintenance_started")
    
    engine = get_topic_engine()
    
    duplicates = engine.detect_duplicates(threshold=0.88)
    auto_merged = 0
    
    for tid1, tid2, sim in duplicates:
        if sim >= 0.92:
            engine.merge_topics(tid1, tid2)
            auto_merged += 1
    
    logger_object['info'].log(f"duplicates_processed: total_found={len(duplicates)}, auto_merged={auto_merged}")
    
    cutoff = datetime.now() - timedelta(days=30)
    expired = 0
    
    for tid in list(engine.topics.keys()):
        tdata = engine.topics[tid]
        last_update = datetime.fromisoformat(tdata.get("updated_at", tdata["created_at"]))
        if last_update < cutoff and tdata.get("article_count", 0) == 0:
            del engine.topics[tid]
            expired += 1
    
    if expired > 0:
        engine._rebuild_topic_index()
    
    logger_object['info'].log(f"maintenance_completed: expired={expired}, stats={engine.get_stats()}")
    
    engine.save_topics()
    TopicDBManager.upsert_topics(engine.topics)