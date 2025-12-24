import redis
import json
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from pathlib import Path

# Import the optimized Cold Storage backend
from utils.qdrant_setup import log_batch_chat, search_chat_history, search_context
from utils.model_manager import ModelManager

load_dotenv()

# Setup Logger
model_manager = ModelManager()
Logger = model_manager.get_model(temperature=0.1)

# ==========================================================
# 🧱 HOT MEMORY (Global Context - Persistent File)
# ==========================================================
class HotMemory:
    """
    Handles the 'Always-On' Global Context.
    Uses a dedicated JSON file for persistence across separate agent runs.
    """
    _HOT_MEMORY_FILE = Path(__file__).parent / "hot_memory.json"
    _CONTEXT_KEY = "global_project_context"

    def __init__(self):
        # Load context when the object is initialized
        self._global_context = self._load_context_from_file()

    def _load_context_from_file(self) -> str:
        """Loads the global context from the persistent file."""
        if not self._HOT_MEMORY_FILE.exists():
            return ""
        try:
            with open(self._HOT_MEMORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get(self._CONTEXT_KEY, "")
        except Exception as e:
            print(f"⚠️ HotMemory: Failed to load context from file. Starting fresh. Error: {e}")
            return ""

    def _save_context_to_file(self, context_text: str):
        """Saves the global context to the persistent file."""
        data = {
            self._CONTEXT_KEY: context_text,
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(self._HOT_MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ HotMemory: Failed to save context to file. Error: {e}")

    def set_context(self, context_text: str):
        """Update the global system prompt context and save it persistently."""
        self._global_context = context_text
        self._save_context_to_file(context_text)

    def get_context(self) -> str:
        """Retrieve the global context for system prompt injection."""
        # Note: If memory is unset, Contextor will set it. If set, we use the loaded value.
        if not self._global_context:
            return "No global context set. Please run Contextor first."
        return self._global_context


# ==========================================================
# ⚡ WARM MEMORY (Redis -> Fallback to JSON File)
# ==========================================================
class WarmMemory:
    """
    Handles High-Speed, Short-Term Memory.
    Attempts to use Redis, but falls back to a JSON file if Redis is offline.
    This ensures admin tools can see data even without Redis.
    """
    # File path for local fallback persistence
    FALLBACK_FILE = Path("warm_memory_dump.json")

    def __init__(self, session_id: str = "default_session", host='localhost', port=6379, db=0, llm=None):
        self.session_id = session_id
        self.chat_key = f"chat:{session_id}"
        self.meta_key = f"meta:{session_id}"
        self.ARCHIVE_THRESHOLD = 10
        self.ARCHIVE_BATCH_SIZE = 1
        self.llm = llm
        
        self.use_redis = False
        self.r = None

        try:
            # Try connecting to Redis
            self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.r.ping() # Test connection
            self.use_redis = True
        except (redis.ConnectionError, ConnectionRefusedError):
            # print(f"⚠️  Redis not found. Using File Fallback: {self.FALLBACK_FILE}")
            self.use_redis = False
            self._ensure_local_store()

    def _ensure_local_store(self):
        """Load local JSON store if it exists, else create empty."""
        if not self.FALLBACK_FILE.exists():
            self._local_store = {}
            self._save_local_store()
        else:
            try:
                with open(self.FALLBACK_FILE, 'r', encoding='utf-8') as f:
                    self._local_store = json.load(f)
            except Exception:
                self._local_store = {}

    def _save_local_store(self):
        """Persist local store to disk."""
        try:
            with open(self.FALLBACK_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._local_store, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Failed to save warm memory dump: {e}")

    # --- Structured Metadata ---
    def save_metadata(self, key: str, value: Any):
        if isinstance(value, (dict, list)):
            value = json.dumps(value)

        if self.use_redis:
            self.r.hset(self.meta_key, key, value)
        else:
            self._ensure_local_store() # Reload to get latest
            if self.meta_key not in self._local_store:
                self._local_store[self.meta_key] = {}
            self._local_store[self.meta_key][key] = value
            self._save_local_store()

    def get_metadata(self, key: str) -> Any:
        val = None
        if self.use_redis:
            val = self.r.hget(self.meta_key, key)
        else:
            self._ensure_local_store()
            val = self._local_store.get(self.meta_key, {}).get(key)

        if val is None:
            return None
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val

    # --- Chat History (The Buffer) ---
    def add_message(self, role: str, content: str):
        msg_obj = {
            "role": role, 
            "content": content, 
            "timestamp": datetime.now().isoformat()
        }
        
        current_len = 0
        if self.use_redis:
            self.r.rpush(self.chat_key, json.dumps(msg_obj))
            current_len = self.r.llen(self.chat_key)
        else:
            self._ensure_local_store()
            if self.chat_key not in self._local_store:
                self._local_store[self.chat_key] = []
            self._local_store[self.chat_key].append(msg_obj)
            self._save_local_store()
            current_len = len(self._local_store[self.chat_key])

        if current_len > self.ARCHIVE_THRESHOLD:
            # Pop oldest message(s) synchronously to maintain warm memory size
            msgs_to_archive = self._pop_oldest_sync()
            
            # Archive asynchronously
            if msgs_to_archive:
                thread = threading.Thread(target=self._archive_oldest, args=(msgs_to_archive,))
                thread.daemon = True
                thread.start()

    def _pop_oldest_sync(self) -> List[Dict[str, Any]]:
        """Removes oldest messages from store and returns them."""
        msgs = []
        if self.use_redis:
            old_msgs_raw = self.r.lpop(self.chat_key, self.ARCHIVE_BATCH_SIZE)
            if old_msgs_raw:
                msgs = [json.loads(m) for m in old_msgs_raw]
        else:
            self._ensure_local_store()
            all_msgs = self._local_store.get(self.chat_key, [])
            msgs = all_msgs[:self.ARCHIVE_BATCH_SIZE]
            self._local_store[self.chat_key] = all_msgs[self.ARCHIVE_BATCH_SIZE:]
            self._save_local_store()
        return msgs

    def get_recent_messages(self, limit: int = 20) -> List[Dict[str, str]]:
        if self.use_redis:
            raw_msgs = self.r.lrange(self.chat_key, -limit, -1)
            return [json.loads(m) for m in raw_msgs]
        else:
            self._ensure_local_store()
            return self._local_store.get(self.chat_key, [])[-limit:]
    
    def clear_session(self):
        if self.use_redis:
            self.r.delete(self.chat_key)
            self.r.delete(self.meta_key)
        else:
            self._ensure_local_store()
            if self.chat_key in self._local_store:
                del self._local_store[self.chat_key]
            if self.meta_key in self._local_store:
                del self._local_store[self.meta_key]
            self._save_local_store()

    # --- Internal Archiver ---
    def _archive_oldest(self, msgs_to_archive: List[Dict[str, Any]]):
        #print(f"⚡ WarmMemory full (> {self.ARCHIVE_THRESHOLD}). Archiving {len(msgs_to_archive)} msg(s) to Cold Storage (Async)...")
        summary_examples = ""
        # Generate Summary if LLM is available
        if Logger:
            for msg in msgs_to_archive:
                try:
                    content = msg.get("content", "")
                    prompt = (
                        f"""Summarize this chat message for future retrieval (not too long but enough to get blur picture of the conversation).
                        Include key entities (only if there are any) and key words which make easy to retrieve context without needing any specefic words (Only 3 most relevent keywords MAX).
                        If User or Any Agent Returns empty response like "" then in summary just say user asked this agent retured empty response.
                        Here are few examples how you responses should look like: {summary_examples}\n
                        Message: {content}"""
                    )
                    # Assuming llm is a LangChain ChatModel
                    summary_response = Logger.invoke(prompt)
                    msg["summary"] = summary_response.content
                except Exception as e:
                    # Handle rate limit gracefully
                    error_str = str(e).lower()
                    if "rate_limit" in error_str or "429" in error_str:
                        print(f"⚠️ Rate limit hit, archiving without summary")
                        msg["summary"] = "[Summary skipped - rate limit]"
                    else:
                        print(f"⚠️ Failed to generate summary: {e}")
                        msg["summary"] = "[Summary generation failed]"
        if msgs_to_archive:
            ColdMemory.archive_batch(msgs_to_archive)


# ==========================================================
# 🧊 COLD MEMORY (Qdrant - Archive)
# ==========================================================
class ColdMemory:
    """
    Wrapper for Long-Term Qdrant Storage.
    """
    @staticmethod
    def archive_batch(messages: List[Dict[str, Any]]):
        try:
            result = log_batch_chat(messages, default_agent="Archived_Agent")
        except Exception as e:
            print(f"❌ Archival Failed: {e}")

    @staticmethod
    def semantic_search(query: str, top_k: int = 5) -> List[Dict]:
        return search_chat_history(query, top_k=top_k)
    
    @staticmethod
    def retrieve_context_knowledge(query: str, top_k: int = 3) -> List[Dict]:
        return search_context(query, top_k=top_k)