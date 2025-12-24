"""
qdrant_setup.py

This module handles the integration with Qdrant via the Model Context Protocol (MCP).
It replaces the manual QdrantClient implementation in `test_memory_sys.py`.

Key Components:
- QdrantMCPWrapper: Manages the connection to the `mcp-server-qdrant` process via stdio.
- Tools: `chat_log_add_tool` and `chat_log_search_tool` wrap the MCP tools for use by the Agent.
- Helper Functions: `log_batch_chat` and `search_chat_history` provide direct access for memory management.

Configuration:
- Uses `chat_logs_mcp` collection to ensure compatibility with the server's embedding model.
- Includes workarounds for known issues in `mcp-server-qdrant` (Pydantic validation, execution method).
"""

import asyncio
import threading
import os
import json
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import tool

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

class QdrantMCPWrapper:
    def __init__(self):
        # Ensure environment variables are set
        env = os.environ.copy()
        if QDRANT_URL:
            env["QDRANT_URL"] = QDRANT_URL
        if QDRANT_API_KEY:
            env["QDRANT_API_KEY"] = QDRANT_API_KEY
        
        # Workaround for mcp-server-qdrant bug with make_partial_function
        # The server uses a helper that breaks Pydantic validation for FastMCP tools.
        # Setting this environment variable bypasses that helper.
        env["QDRANT_ALLOW_ARBITRARY_FILTER"] = "true"
        
        # Use the current python executable to run the module
        # We use -c execution to ensure the main() function is called, as the module
        # might not execute main() when run via -m in some environments.
        import sys
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=["-c", "from mcp_server_qdrant.main import main; main()", "--transport", "stdio"],
            env=env
        )
        
        # Persistent session state
        self._loop = None
        self._thread = None
        self._session = None
        self._session_ready = threading.Event()
        self._shutdown_event = None # Initialized in loop

    def start(self):
        """Starts the MCP server in a background thread if not already running."""
        # Only start the thread if it's not running
        if not (self._thread and self._thread.is_alive()):
            print(f"DEBUG: Starting new MCP thread. Current thread: {threading.current_thread().name}")
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        else:
             print(f"DEBUG: MCP thread already running: {self._thread.name}")
        
        # Always wait for session to be ready, regardless of who started the thread
        if not self._session_ready.wait(timeout=20):
             raise RuntimeError("Timeout waiting for MCP server to start")

    def _run_loop(self):
        """Runs the asyncio loop in the background thread."""
        asyncio.set_event_loop(self._loop)
        self._shutdown_event = asyncio.Event()
        self._loop.run_until_complete(self._lifecycle())

    async def _lifecycle(self):
        """Manages the lifecycle of the MCP session."""
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._session_ready.set()
                    
                    # Keep running until shutdown is requested
                    await self._shutdown_event.wait()
        except Exception as e:
            print(f"MCP Session Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("DEBUG: MCP Session Ended")
            self._session = None
            self._session_ready.clear()

    def run_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Runs a tool synchronously using the persistent session."""
        if not self._session:
            self.start()
        
        if not self._session:
             raise RuntimeError("Failed to initialize MCP session")

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._session.call_tool(tool_name, arguments), 
                self._loop
            )
            res = future.result()
            # print(f"DEBUG: Tool {tool_name} returned: {res}") 
            return res
        except Exception as e:
            print(f"Error calling {tool_name}: {e}")
            raise e

# Global instance
mcp_wrapper = QdrantMCPWrapper()

# --- Helper Functions (Replacing Manual Client) ---

from datetime import datetime

def log_batch_chat(conversations: List[Dict[str, Any]], default_agent: str = "Archived") -> Dict[str, Any]:
    """
    Batch logs chat messages using MCP qdrant-store-memory tool.
    ATTEMPT TO PAIR MESSAGES (User -> Agent) to match requested format:
    [time_stamp, user_query, agent, agent_response, summary]
    """
    count = 0
    errors = []
    
    # Simple pairing logic: Iterate and look for User then Agent
    # If we find Agent without User, we log with empty User query?
    # If we find User without Agent, we wait for next?
    
    # We will iterate and combine.
    i = 0
    while i < len(conversations):
        msg = conversations[i]
        role = msg.get('role', 'unknown').lower()
        content = msg.get('content', '')
        timestamp = msg.get('timestamp', datetime.now().isoformat())
        summary = msg.get('summary', '')
        
        user_query = ""
        agent_response = ""
        agent_name = default_agent
        
        if role == 'user':
            user_query = content
            # Look ahead for agent response
            if i + 1 < len(conversations):
                next_msg = conversations[i+1]
                if next_msg.get('role', '').lower() in ['ai', 'assistant', 'contextor', 'analyst']:
                    agent_response = next_msg.get('content', '')
                    agent_name = next_msg.get('role', default_agent) # Use the role as agent name
                    # If the user message didn't have a summary, check the agent message
                    if not summary:
                        summary = next_msg.get('summary', '')
                    i += 1 # Skip next message as we consumed it
        
        elif role in ['ai', 'assistant', 'contextor', 'analyst']:
             # Orphan agent message (or system started with specific agent output)
             agent_response = content
             agent_name = role
        else:
             # System or other? Treat as User query for now or just log
             user_query = f"[{role.upper()}] {content}"

        # Construct the requested format
        # [time_stamp, user_query, agent, agent_response, summary]
        memory_text = f"[{timestamp}, {user_query}, {agent_name}, {agent_response}, {summary}]"
            
        try:
            # Using 'qdrant-store' tool
            res = mcp_wrapper.run_tool_sync("qdrant-store", {
                "information": memory_text,
                "collection_name": "chat_logs_mcp"
            })
            #print(f"DEBUG: log_batch_chat store result: {res}")
            count += 1
        except Exception as e:
            errors.append(str(e))
        
        i += 1
            
    return {"status": "ok" if not errors else "partial_error", "count": count, "errors": errors}
            


def search_chat_history(query: str, top_k: int = 2, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Searches chat history using MCP qdrant-find tool.
    """
    try:
        # qdrant-find usually takes 'query'
        result = mcp_wrapper.run_tool_sync("qdrant-find", {
            "query": query,
            "collection_name": "chat_logs_mcp"
        })
        
        # Parse result to extract text from content
        found_text = []
        if hasattr(result, 'content') and isinstance(result.content, list):
            for item in result.content:
                text_content = ""
                if hasattr(item, 'text'):
                    text_content = item.text
                else:
                    text_content = str(item)
                
                # Truncate to avoid exploding context window
                if len(text_content) > 1000:
                    text_content = text_content[:1000] + "... [TRUNCATED]"
                found_text.append(text_content)
        else:
             found_text.append(str(result))
             
        full_text = "\n\n".join(found_text)
        return [{"text": full_text, "metadata": {}}] 
    except Exception as e:
        print(f"MCP Search Error: {e}")
        return []

def search_context(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches context using MCP.
    """
    try:
        result = mcp_wrapper.run_tool_sync("qdrant-find", {
            "query": query,
            "collection_name": "context_store"
        })
        
        # Parse result to extract text from content
        found_text = []
        if hasattr(result, 'content') and isinstance(result.content, list):
            for item in result.content:
                if hasattr(item, 'text'):
                    found_text.append(item.text)
                else:
                    found_text.append(str(item))
        else:
             found_text.append(str(result))
             
        full_text = "\n\n".join(found_text)
        return [{"text": full_text, "metadata": {}}]
    except Exception as e:
        print(f"MCP Context Search Error: {e}")
        return []

def update_context(text: str, dataset: str = "Unknown", agent: str = "Contextor", section: str = "General") -> str:
    """
    Updates the context store (replacing old update_context from test_memory_sys).
    """
    info = f"SECTION: {section}\nDATASET: {dataset}\nAGENT: {agent}\nCONTENT: {text}"
    try:
        result = mcp_wrapper.run_tool_sync("qdrant-store", {
            "information": info,
            "collection_name": "context_store"
        })
        # Parse result for cleaner output
        msg = str(result)
        if hasattr(result, 'content') and result.content:
             msg = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
             
        return f"Context updated: {msg}"
    except Exception as e:
        print(f"MCP Context Update Error: {e}")
        return f"Error updating context: {e}"

# --- Tool Wrappers ---

@tool("chat_log_add", return_direct=False)
def chat_log_add_tool(user_query: str, agent: str, agent_response: str, summary: Optional[str] = None, tags_json: Optional[str] = None) -> str:
    """Log a chat interaction to the chat history."""
    memory_text = f"USER: {user_query}\nAGENT: {agent_response}"
    if summary:
        memory_text += f"\nSummary: {summary}"
        
    try:
        result = mcp_wrapper.run_tool_sync("qdrant-store", {
            "information": memory_text,
            "collection_name": "chat_logs_mcp"
        })
        return f"Logged successfully: {result}"
    except Exception as e:
        return f"Error logging: {e}"

@tool("chat_log_search", return_direct=False)
def chat_log_search_tool(query: str, top_k: int = 1, metadata_filter_json: Optional[str] = None) -> str:
    """Search chat history for relevant conversations."""
    try:
        result = mcp_wrapper.run_tool_sync("qdrant-find", {
            "query": query,
            "collection_name": "chat_logs_mcp"
        })
        return str(result)
    except Exception as e:
        return f"Error searching: {e}"
