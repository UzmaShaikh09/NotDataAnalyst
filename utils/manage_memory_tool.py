import os
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import mcp_server_qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_setup import chat_log_add_tool, chat_log_search_tool, log_batch_chat

# Load .env from project root (one level up from utils)
load_dotenv(Path(__file__).parent.parent / ".env")

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from memory_manager import HotMemory, WarmMemory
except ImportError:
    print("Warning: Could not import memory_manager. Some dump functions may fail.")

# --- Configuration & Constants ---
# Assuming this script is in NotDataAnalyst/
# Use absolute paths relative to this script
BASE_DIR = Path(__file__).parent
HOT_MEMORY_FILE = BASE_DIR / "hot_memory.json" 
WARM_MEMORY_FILE = BASE_DIR / "warm_memory_dump.json"
DUMP_DIR = Path("memory_dumps")
DUMP_DIR.mkdir(exist_ok=True)

# Qdrant Collections
COLLECTION_NAME = "chat_logs_mcp" # Main collection for chat logs in clear_memory
_CONTEXT_COLLECTION = "context_store" # From dump_everything
_LOGS_COLLECTION = "chat_logs_mcp"   # From dump_everything

VECTOR_SIZE = 384

# --- Generators / Helpers ---
def _get_qdrant_client():
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    return QdrantClient(url=url, api_key=api_key)

# --- Clear Functions ---
def clear_hot_memory():
    # Hot memory is typically in the same dir as memory_manager.py, which is here.
    # We use HOT_MEMORY_FILE defined above.
    # Note: memory_manager.py defines it as: Path(__file__).parent / "hot_memory.json"
    
    print(f"🔥 Clearing Hot Memory ({HOT_MEMORY_FILE})...")
    if HOT_MEMORY_FILE.exists():
        try:
            os.remove(HOT_MEMORY_FILE)
            print("✅ Hot Memory deleted.")
        except Exception as e:
            print(f"❌ Failed to delete Hot Memory: {e}")
    else:
        print("ℹ️  Hot Memory file not found.")

def clear_warm_memory():
    print(f"⚡ Clearing Warm Memory ({WARM_MEMORY_FILE})...")
    if WARM_MEMORY_FILE.exists():
        try:
            os.remove(WARM_MEMORY_FILE)
            print("✅ Warm Memory deleted.")
        except Exception as e:
            print(f"❌ Failed to delete Warm Memory: {e}")
    else:
        print("ℹ️  Warm Memory file not found.")

def clear_cold_memory():
    # We clear both the Chat Logs and the Context Store
    collections_to_clear = [COLLECTION_NAME, _CONTEXT_COLLECTION]
    
    print(f"🧊 Clearing Cold Memory (Qdrant Collections: {collections_to_clear})...")
    
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    
    if not url:
        print("❌ QDRANT_URL not set in .env. Skipping Cold Memory.")
        return

    try:
        client = QdrantClient(url=url, api_key=api_key)
        
        for col_name in collections_to_clear:
            print(f"   👉 Processing '{col_name}'...")
            
            # Check if collection exists
            try:
                collections = client.get_collections().collections
                exists = any(c.name == col_name for c in collections)
            except Exception:
                exists = False
            
            if exists:
                client.delete_collection(col_name)
                print(f"      ✅ Collection '{col_name}' deleted.")
            else:
                print(f"      ℹ️  Collection '{col_name}' not found.")
                
            # Recreate empty collection with NAMED VECTOR for MCP compatibility
            print(f"      🔄 Recreating empty collection '{col_name}'...")
            client.create_collection(
                collection_name=col_name,
                vectors_config={
                    "fast-all-minilm-l6-v2": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
                }
            )
            print(f"      ✅ Collection '{col_name}' recreated with named vector 'fast-all-minilm-l6-v2'.")
        
    except Exception as e:
        print(f"❌ Failed to clear Cold Memory: {e}")

# --- Dump Functions ---
def dump_hot():
    """Dumps Hot Memory (Global Context) to a JSON file."""
    print("🔥 Dumping Hot Memory...")
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        hot_memory = HotMemory()
        context = hot_memory.get_context()
        
        data = {
            "global_context": context,
            "source_file": str(hot_memory._HOT_MEMORY_FILE)
        }
        
        output_file = DUMP_DIR / f"hot_memory_dump_{TIMESTAMP}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Hot Memory dumped to {output_file}")
    except Exception as e:
        print(f"❌ Failed to dump Hot Memory: {e}")

def dump_warm():
    """Dumps Warm Memory (Redis/Local Chat History) for ALL sessions."""
    print("⚡ Dumping Warm Memory (All Sessions)...")
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        # Initialize a temporary instance to check connection and access client
        base_memory = WarmMemory(session_id="temp_dump_scanner")
        
        all_sessions_data = {}

        if base_memory.use_redis:
            print("   - Mode: Redis")
            try:
                r = base_memory.r
                chat_keys = r.keys("chat:*")
                
                for key in chat_keys:
                    parts = key.split(":", 1)
                    if len(parts) < 2: 
                        continue
                    session_id = parts[1]
                    
                    # Fetch chat history
                    raw_msgs = r.lrange(key, 0, -1)
                    messages = []
                    for m in raw_msgs:
                        try:
                            messages.append(json.loads(m))
                        except:
                            messages.append(m)
                    
                    # Fetch metadata
                    meta_key = f"meta:{session_id}"
                    raw_meta = r.hgetall(meta_key)
                    metadata = {}
                    for k, v in raw_meta.items():
                        try:
                            metadata[k] = json.loads(v)
                        except:
                            metadata[k] = v
                    
                    all_sessions_data[session_id] = {
                        "chat_history": messages,
                        "metadata": metadata
                    }
            except Exception as e:
                print(f"Error accessing Redis: {e}")
                
        else:
            print(f"   - Mode: Local File ({base_memory.FALLBACK_FILE})")
            if base_memory.FALLBACK_FILE.exists():
                with open(base_memory.FALLBACK_FILE, 'r', encoding='utf-8') as f:
                    local_store = json.load(f)
                
                for key, value in local_store.items():
                    if key.startswith("chat:"):
                        session_id = key.split(":", 1)[1]
                        if session_id not in all_sessions_data:
                            all_sessions_data[session_id] = {"chat_history": [], "metadata": {}}
                        all_sessions_data[session_id]["chat_history"] = value
                        
                    elif key.startswith("meta:"):
                        session_id = key.split(":", 1)[1]
                        if session_id not in all_sessions_data:
                            all_sessions_data[session_id] = {"chat_history": [], "metadata": {}}
                        all_sessions_data[session_id]["metadata"] = value

        output_file = DUMP_DIR / f"warm_memory_export_{TIMESTAMP}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_sessions_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Warm Memory (All Sessions) dumped to {output_file}")
    except Exception as e:
        print(f"❌ Failed to dump Warm Memory: {e}")

def dump_cold():
    """Dumps Cold Memory (Qdrant Collections) to a JSON file."""
    print("🧊 Dumping Cold Memory...")
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        client = _get_qdrant_client()
        
        dump_data = {
            "context_store": [],
            "chat_logs_mcp": []
        }
        
        def scroll_collection(collection_name):
            points = []
            next_offset = None
            while True:
                results, next_offset = client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in results:
                    points.append({
                        "id": point.id,
                        "payload": point.payload
                    })
                
                if next_offset is None:
                    break
            return points

        # Dump Context Store
        try:
            print(f"   - Scrolling {_CONTEXT_COLLECTION}...")
            dump_data["context_store"] = scroll_collection(_CONTEXT_COLLECTION)
        except Exception as e:
            print(f"   ⚠️ Could not dump {_CONTEXT_COLLECTION}: {e}")

        # Dump Chat Logs
        try:
            print(f"   - Scrolling {_LOGS_COLLECTION}...")
            dump_data["chat_logs"] = scroll_collection(_LOGS_COLLECTION)
        except Exception as e:
            print(f"   ⚠️ Could not dump {_LOGS_COLLECTION}: {e}")

        output_file = DUMP_DIR / f"cold_memory_dump_{TIMESTAMP}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Cold Memory dumped to {output_file}")

    except Exception as e:
        print(f"❌ Failed to dump Cold Memory: {e}")


def test_mcp_tools():
    print("\n--- MCP Integration Final Report ---")
    
    # 1. Test chat_log_add_tool
    print("\n1. Testing chat_log_add_tool...")
    start_time = time.time()
    try:
        result = chat_log_add_tool.invoke({
            "user_query": "Hello MCP Final Test",
            "agent": "Tester",
            "agent_response": "Hello User",
            "summary": "Greeting test final",
            "collection_name": "chat_logs_mcp"
        })
        end_time = time.time()
        add_duration = end_time - start_time
        print(f"✅ Add Result: Success")
        print(f"⏱️  Add Duration: {add_duration:.4f} seconds")
    except Exception as e:
        print(f"❌ Add Failed: {e}")
        add_duration = 0

    # 2. Test chat_log_search_tool
    print("\n2. Testing chat_log_search_tool...")
    start_time = time.time()
    try:
        # Give it a moment to index if needed (though Qdrant is usually fast)
        # time.sleep(1) # Removed sleep to measure raw search time
        result = chat_log_search_tool.invoke({
            "query": "Hello MCP Final Test",
            "collection_name": "chat_logs_mcp"
        })
        end_time = time.time()
        search_duration = end_time - start_time
        print(f"✅ Search Result: Success (Found {len(str(result))} chars)")
        print(f"⏱️  Search Duration: {search_duration:.4f} seconds")
    except Exception as e:
        print(f"❌ Search Failed: {e}")
        search_duration = 0
        
    print("\n--- Summary ---")
    print(f"Log 1 Query Time: {add_duration:.4f}s")
    print(f"Search 1 Query Time: {search_duration:.4f}s")
    print("----------------")

# --- Main Interface ---
def main():
    while True:
        print("\n=== Memory Management Tool ===")
        print("1. [CLEAR] Clear Memory (Hot, Warm, Cold)")
        print("2. [DUMP]  Dump Memory (Backup everything)")
        print("3. [TEST]  Test MCP Tools")
        print("q. [QUIT]  Exit")
        
        choice = input("Enter choice: ").strip().lower()
        
        if choice == '1':
            confirm = input("⚠️ Are you sure you want to CLEAR all memory? This cannot be undone. (y/n): ").lower()
            if confirm == 'y':
                print("\n--- Clearing Memories ---")
                clear_hot_memory()
                clear_warm_memory()
                clear_cold_memory()
                print("✨ Done.")
            else:
                print("❌ Cancelled.")
                
        elif choice == '2':
            print("\n--- Dumping Memories ---")
            dump_hot()
            dump_warm()
            dump_cold()
            print("✨ Done.")

        elif choice == '3':
            print("\n--- Testing MCP Tools ---")
            test_mcp_tools()
            print("✨ Done.")
            
        elif choice == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
