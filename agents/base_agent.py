import os
import sys
import json
import threading
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional


# Add project root to path so we can import 'tools' and 'utils'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Import Tools ---
from tools.expert_crew_tools import get_expert_tools
from utils.qdrant_setup import chat_log_search_tool

# --- Import Memory ---
from utils.memory_manager import HotMemory, WarmMemory
from utils.memory_manager import HotMemory, WarmMemory
from utils.model_manager import ModelManager, switch_to_provider

# --- Import Prompt Toolkit ---
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import run_in_terminal

# Load Env
load_dotenv()

class BaseAgent:
    def __init__(self, agent_name: str, system_prompt_template: str, session_id: Optional[str] = None):
        """
        Args:
            agent_name: Name of the agent (e.g., "Cleaner", "Visualizer")
            system_prompt_template: The raw prompt string. Must contain {project_context} placeholder.
            session_id: Unique session ID. Defaults to f"{agent_name.lower()}_session_v1".
        """
        self.agent_name = agent_name
        self.model_manager = ModelManager()
        self.llm = self.model_manager.get_model()
        
        # 1. Setup Tools
        self.analysis_tools = get_expert_tools()
        self.memory_tools = [chat_log_search_tool]
        self.all_tools = self.analysis_tools + self.memory_tools
        self.tools_map = {t.name: t for t in self.all_tools}
        
        # Bind tools to LLM initially
        self.current_llm_with_tools = self.llm.bind_tools(self.all_tools)

        # 2. Setup Memory
        self.hot_memory = HotMemory()
        self.session_id = session_id or f"{agent_name.lower()}_session_v1"
        self.warm_memory = WarmMemory(session_id=self.session_id, llm=self.llm)

        # 3. Prepare System Prompt
        context = self.hot_memory.get_context()
        if "No global context set" in context:
            print(f"⚠️  [{self.agent_name}] WARNING: No Project Context found in Hot Memory.")
        
        self.formatted_system_prompt = system_prompt_template.format(project_context=context)

    def _robust_invoke(self, messages):
        """
        Attempts to invoke the LLM with automatic fallback to other providers.
        Updates self.llm and self.current_llm_with_tools on switch.
        """
        # 1. Try current model
        try:
            return self.current_llm_with_tools.invoke(messages)
        except Exception as e:
            error_str = str(e)
            
            # Check for tool validation errors (model hallucinated a tool that doesn't exist)
            if "tool call validation failed" in error_str or "not in request.tools" in error_str:
                print(f"⚠️  Tool validation error - model tried to call invalid tool. Returning error message.")
                return AIMessage(content="I encountered a tool error. The requested tool is not available. Please use only: python_interpreter, install_package, or chat_log_search.")
            
            print(f"⚠️  LLM Failed with current/default provider: {e}")
        
        # 2. Fallback Sequence
        providers = ["gemini", "groq", "openrouter"]
        
        for provider in providers:
            print(f"🔄 Auto-switching to: {provider}...")
            try:
                switch_to_provider(provider)
                
                # REFRESH internal model state
                self.llm = self.model_manager.get_model()
                self.current_llm_with_tools = self.llm.bind_tools(self.all_tools)
                
                # Retry
                return self.current_llm_with_tools.invoke(messages)
            except Exception as e:
                print(f"❌ Provider {provider} failed: {e}")
                continue

        raise RuntimeError("All LLM providers failed. Please check your API keys or connection.")


    def _build_history(self) -> List:
        """Reconstructs LangChain history from WarmMemory."""
        messages = [SystemMessage(content=self.formatted_system_prompt)]
        recent_chat = self.warm_memory.get_recent_messages(limit=30) 

        for msg in recent_chat:
            role = msg.get("role")
            content = msg.get("content")
            
            if len(content) > 3000:
                content = content[:3000] + "... [TRUNCATED]"
            
            # Treat "User" as Human
            if role.lower() == "user":
                messages.append(HumanMessage(content=content))
            # Treat ALL other roles (Cleaner, Fe_Agent, etc.) as AI colleagues
            else:
                # We prefix the content with the Role Name so the current agent knows WHO said it
                labeled_content = f"[{role}]: {content}"
                messages.append(AIMessage(content=labeled_content))
        return messages

    def run_task(self, task: str) -> str:
        """
        Executes a single task from the Router/Graph and returns the result.
        This is for AUTOMATED mode (LangGraph).
        """
        print(f"\n🚀 [{self.agent_name}] Received Task: {task}")
        
        # 1. Log Task to Memory
        self.warm_memory.add_message("User", task)

        # 2. Build History
        messages = self._build_history()

        try:
            # 3. Invoke LLM
            # 3. Invoke LLM (Robust)
            ai_msg = self._robust_invoke(messages)
            messages.append(ai_msg)

            # 4. Handle Tool Calls (Loop)
            loop_count = 0
            MAX_LOOPS = 5 # Safety break
            
            while ai_msg.tool_calls and loop_count < MAX_LOOPS:
                loop_count += 1
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    args = tool_call["args"]
                    
                    selected_tool = self.tools_map.get(tool_name) or self.tools_map.get(tool_name.lower())
                    
                    if selected_tool:
                        print(f"🛠️  [{self.agent_name}] Executing: {tool_name}...")
                        try:
                            # Direct execution (Validation is inside the tool now)
                            tool_output = selected_tool.invoke(args)
                        except Exception as e:
                            tool_output = f"Error executing tool: {e}"
                    else:
                        tool_output = f"Error: Tool {tool_name} not found."

                    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

                # Re-invoke after tools
                # Re-invoke after tools
                ai_msg = self._robust_invoke(messages)
                messages.append(ai_msg)

            # 5. Final Response
            final_text = ai_msg.content
            if isinstance(final_text, list):
                final_text = "\n".join([p.get('text', '') for p in final_text if isinstance(p, dict) and 'text' in p])

            self.warm_memory.add_message(self.agent_name, str(final_text))
            print(f"🏁 [{self.agent_name}] Finished: {final_text[:100]}...")
            return str(final_text)

        except Exception as e:
            error_msg = f"❌ Agent Error: {e}"
            print(error_msg)
            return error_msg

    def run(self):
        """Main Chat Loop."""
        session = PromptSession()
        bindings = KeyBindings()

        @bindings.add('c-x')
        def _(event):
            self._switch_model()

        print(f"[{self.agent_name}] is listening... (Type 'exit' to quit, Ctrl+X to switch models)")

        while True:
            try:
                user_input = session.prompt(f"{self.agent_name}: ", key_bindings=bindings)
            except (KeyboardInterrupt, EOFError):
                break

            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input.strip():
                continue

            # 1. Log to Memory
            self.warm_memory.add_message("User", user_input)

            # 2. Build History
            messages = self._build_history()

            try:
                # 3. Invoke LLM
                # 3. Invoke LLM
                ai_msg = self._robust_invoke(messages)
                messages.append(ai_msg)

                # 4. Handle Tool Calls
                while ai_msg.tool_calls:
                    for tool_call in ai_msg.tool_calls:
                        tool_name = tool_call["name"]
                        args = tool_call["args"]
                        
                        # --- VALIDATION LAYER (Placeholder for Phase 2) ---
                        # check_safety(tool_name, args) 
                        # --------------------------------------------------

                        selected_tool = self.tools_map.get(tool_name) or self.tools_map.get(tool_name.lower())
                        
                        if selected_tool:
                            print(f"🛠️  [{self.agent_name}] Executing: {tool_name}...")
                            try:
                                tool_output = selected_tool.invoke(args)
                            except Exception as e:
                                tool_output = f"Error executing tool: {e}"
                        else:
                            tool_output = f"Error: Tool {tool_name} not found."

                        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

                    # Re-invoke after tool outputs
                    # Re-invoke after tool outputs
                    ai_msg = self._robust_invoke(messages)
                    messages.append(ai_msg)

                # 5. Final Response
                final_text = ai_msg.content
                if isinstance(final_text, list):
                    # Handle Gemini's occasional list response
                    final_text = "\n".join([p.get('text', '') for p in final_text if isinstance(p, dict) and 'text' in p])

                self.warm_memory.add_message(self.agent_name, str(final_text))
                print(f"\n{self.agent_name}: {final_text}\n")

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    print("Run specific agents (cleaner.py, trainer.py) instead of this base class.")