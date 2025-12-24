import os
import sys
import re
import operator
from typing import Annotated, List, TypedDict, Union, Dict

# Add project root to path so we can import 'tools' and 'utils'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


from dotenv import load_dotenv
from utils.model_manager import ModelManager
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import Agent Definitions
# We instantiate the BaseAgent with the prompts defined in your files
from agents.base_agent import BaseAgent
from agents.cleaner import CLEANER_PROMPT
from agents.feature_engineer import FE_PROMPT
from agents.vizualizer import VIZ_PROMPT
from agents.trainer import TRAINER_PROMPT
from utils.model_manager import switch_to_provider

load_dotenv()

# ==========================================================
# 🏗️ STATE DEFINITION
# ==========================================================
class AgentState(TypedDict):
    """
    The shared state of the graph.
    Tracks the user request and the specific instructions for each agent.
    """
    user_request: str
    
    # Task Instructions (If empty, agent is skipped)
    cleaner_task: str
    fe_task: str
    viz_task: str
    trainer_task: str
    
    messages: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    last_agent: str  # Tracks who ran last to guide the Watcher's routing
    
    # Watcher feedback fields
    watcher_status: str   # "PASS" or "RETRY"
    watcher_feedback: str  # Feedback message from Watcher

# ==========================================================
# 🤖 INITIALIZE AGENTS
# ==========================================================
# Instantiate Agents with SHARED session ID for shared context
SHARED_SESSION_ID = "shared_session_v1"

cleaner_agent = BaseAgent("Cleaner", CLEANER_PROMPT, session_id=SHARED_SESSION_ID)
fe_agent = BaseAgent("Feature_Engineer", FE_PROMPT, session_id=SHARED_SESSION_ID)
viz_agent = BaseAgent("Visualizer", VIZ_PROMPT, session_id=SHARED_SESSION_ID)
trainer_agent = BaseAgent("Trainer", TRAINER_PROMPT, session_id=SHARED_SESSION_ID)

# Router LLM (The Manager)
model_manager = ModelManager()
router_llm = model_manager.get_model(temperature=0)

# ==========================================================
# 🧠 ROUTER LOGIC
# ==========================================================
ROUTER_SYSTEM_PROMPT = """
You are the **Router**, the primary user interface and the lead orchestrator of an autonomous data analytics crew.
Your role has two modes:

1.  **ASSISTANT MODE (Chatbot):** If the request requires no analytical work (greetings, general questions, metadata queries like "who are you?"), you respond directly and concisely.
2.  **ORCHESTRATOR MODE (Task Delegation):** If the request requires data cleaning, feature engineering, visualization, or model training, you delegate the work.

### 🎯 Your Goal
Determine the user's intent and output a single JSON object that dictates the next step.

### 📋 Available Agents (The Expert Crew)
1.  **Cleaner:** Fixes datatypes, nulls, duplicates. (Output: 'clean_data.parquet')
2.  **Feature_Engineer:** Adds new columns, ratios, segments. (Output: 'engineered_data.parquet')
3.  **Visualizer:** Creates charts/plots. (Does NOT clean or train).
4.  **Trainer:** Trains ML models, predicts, evaluates.

### 📜 Delegation Rules & Constraints
* **Dependencies:** If visualization or training is requested, ensure data is cleaned and necessary features are created first (e.g., Cleaner → Feature_Engineer → Visualizer/Trainer).
* **Task Definition:** Be specific in your instructions. Tell the agent exactly what to use and what to produce.
* **Mandatory Keys:** Your output MUST contain the key `"chat_response"`. All other task keys are optional.

### 📝 JSON Output Schema
Your JSON output must **ALWAYS** contain the following keys. All values must be strings or null.

| Key Name | Type | Purpose |
| :--- | :--- | :--- |
| **`chat_response`** | `string` | **REQUIRED.** Your friendly, direct response to the user, or confirmation of task initiation (e.g., "Understood, I'm setting the team in motion to clean and plot your data."). |
| `cleaner_task` | `string/null` | Specific task for the Cleaner Agent. Set to `null` if not needed. |
| `fe_task` | `string/null` | Specific task for the Feature Engineer. Set to `null` if not needed. |
| `viz_task` | `string/null` | Specific task for the Visualizer. Set to `null` if not needed. |
| `trainer_task` | `string/null` | Specific task for the Trainer. Set to `null` if not needed. |

### 💡 Examples

**Example 1: General Query (Assistant Mode)**
User: "Hello, my name is Owez, who are you?"
```json
{
  "chat_response": "Hello Owez. I am the Router, the lead orchestrator of this autonomous data analytics crew. How can I help you analyze your data today?",
  "cleaner_task": null,
  "fe_task": null,
  "viz_task": null,
  "trainer_task": null
}```

**Example 2: Retry/Feedback Logic**
History: "Cleaner: Done...", "Watcher: RETRY -> FE output invalid..."
Router Output:
```json
{
  "chat_response": "I see the Feature Engineer failed. I'm updating instructions to fix the issue.",
  "cleaner_task": null,
  "fe_task": "Calculate sales_per_order correctly using...",
  "viz_task": "Wait for FE...",
  "trainer_task": "Wait for FE..."
}```
"""

def router_node(state: AgentState):
    print("\n🚦 [ROUTER] Analyzing Request...")
    request = state["user_request"]
    
    # Check if we're here due to a FAIL verdict - Router MUST address this
    watcher_status = state.get("watcher_status")
    watcher_feedback = state.get("watcher_feedback", "")
    failed_agent = state.get("last_agent", "unknown")
    
    fail_context = ""
    if watcher_status == "FAIL":
        print(f"⚠️ Router must address FAIL for {failed_agent}: {watcher_feedback}")
        fail_context = f"""
### ⚠️ CRITICAL: WATCHER FAIL VERDICT
The Watcher agent flagged the previous work as FAILED.
- Failed Agent: {failed_agent}
- Feedback: {watcher_feedback}

YOU MUST reassign the task to {failed_agent} with corrections.
DO NOT say "already complete". The Watcher verdict OVERRIDES your judgment.
"""
    
    # 1. Invoke LLM
    # Format history for context
    history = "\n".join([str(m) for m in state.get("messages", [])])
    
    system_prompt = ROUTER_SYSTEM_PROMPT + fail_context + "\n\nCRITICAL: YOU MUST WRAP YOUR OUTPUT IN ```json ... ```"
    
    messages = [
        ("system", system_prompt),
        ("user", f"Request: {request}\n\nExisting Execution History (Do NOT re-run completed tasks unless failed):\n{history}")
    ]
    response = router_llm.invoke(messages)
    content = response.content
    
    # 2. Robust JSON Extraction and Cleanup using json_repair
    try:
        import json_repair
        tasks = json_repair.repair_json(content, return_objects=True)
        
        # 3. Extract and Display Chat Response
        chat_response = tasks.pop("chat_response", "✅ Task delegated or completed.")
        print(f"\n🗣️  Router says: {chat_response}")
        
    except Exception as e:
        # Fallback for when the model outputs pure prose
        print(f"❌ Router failed to parse JSON even with repair: {e}")
        # If parsing failed, assuming general chat
        chat_response = content.strip()
        print(f"\n🗣️  Router says: {chat_response}")
        tasks = {} # Clear tasks to force END
        
    print(f"📋 Plan: {tasks if tasks else 'No analytical tasks.'}")
    
    # 4. Return State
    return {
        "cleaner_task": tasks.get("cleaner_task"),
        "fe_task": tasks.get("fe_task"),
        "viz_task": tasks.get("viz_task"),
        "trainer_task": tasks.get("trainer_task"),
        # Add a placeholder message for the final output, not the chat response
        "messages": [AIMessage(content=f"Router Decision: {tasks}")] 
    }
    
# ==========================================================
# 👷 AGENT NODES
# ==========================================================

from agents.watcher import WATCHER_PROMPT

# Initialize Watcher
watcher_agent = BaseAgent("Watcher", WATCHER_PROMPT, session_id=SHARED_SESSION_ID)

def run_agent_safely(agent_func, task, state):
    """Helper to run agents with interrupt handling."""
    try:
        # If there is feedback, append it to the task
        if state.get("watcher_status") == "RETRY":
             task = f"FEEDBACK_FROM_WATCHER: {state.get('watcher_feedback')}\n\nORIGINAL_TASK: {task}"
        
        return agent_func(task)
    except KeyboardInterrupt:
        print(f"\n🛑 Agent execution interrupted.")
        return "Task interrupted by user."

def cleaner_node(state: AgentState):
    task = state.get("cleaner_task")
    if not task: return {}
    # Note: BaseAgent.run_task already prints the task received
    result = run_agent_safely(cleaner_agent.run_task, task, state)
    return {"messages": [AIMessage(content=f"Cleaner: {result}")], "last_agent": "cleaner"}

def fe_node(state: AgentState):
    task = state.get("fe_task")
    if not task: return {}
    result = run_agent_safely(fe_agent.run_task, task, state)
    return {"messages": [AIMessage(content=f"FE: {result}")], "last_agent": "feature_engineer"}

def viz_node(state: AgentState):
    task = state.get("viz_task")
    if not task: return {}
    result = run_agent_safely(viz_agent.run_task, task, state)
    # Note: No last_agent update - this node goes directly to END
    return {"messages": [AIMessage(content=f"Visualizer: {result}")]}

def trainer_node(state: AgentState):
    task = state.get("trainer_task")
    if not task: return {}
    result = run_agent_safely(trainer_agent.run_task, task, state)
    # Note: No last_agent update - this node goes directly to END
    return {"messages": [AIMessage(content=f"Trainer: {result}")]}

def watcher_node(state: AgentState):
    """
    The Critic Node.
    Reviews the last message with 3-level severity: PASS, WARN, FAIL.
    """
    print("\n👀 [WATCHER] Reviewing work...")
    
    # Get the last message (content of the work done)
    last_msg = state["messages"][-1] if state["messages"] else AIMessage(content="")
    last_agent_output = last_msg.content
    
    try:
        review = watcher_agent.run_task(f"Review this output: {last_agent_output}")
    except KeyboardInterrupt:
        print("\n🛑 Watcher interrupted by user.")
        return {"watcher_status": "INTERRUPTED", "watcher_feedback": "Watcher interrupted by user."} 

    # Clean the JSON output
    try:
        import json_repair
        cleaned_review = json_repair.repair_json(review, return_objects=True)
    except Exception as e:
        print(f"❌ Watcher failed to parse JSON even with repair: {e}. Defaulting to PASS.")
        return {"watcher_status": "PASS", "watcher_feedback": "Watcher output unparseable, defaulting to PASS."}
    
    status = cleaned_review.get("status", "PASS").upper()
    feedback = cleaned_review.get("feedback", "No feedback provided.")
    
    # Handle legacy "RETRY" as "FAIL" for backwards compatibility
    if status == "RETRY":
        status = "FAIL"
    
    # Log based on severity
    if status == "FAIL":
        print(f"❌ Watcher Verdict: FAIL -> {feedback}")
        return {"watcher_status": "FAIL", "watcher_feedback": feedback}
    elif status == "WARN":
        print(f"⚠️  Watcher Verdict: WARN -> {feedback}")
        return {"watcher_status": "WARN", "watcher_feedback": feedback}
    else:
        print(f"✅ Watcher Verdict: PASS -> {feedback}")
        return {"watcher_status": "PASS", "watcher_feedback": feedback}

# ==========================================================
# 🕸️ GRAPH CONSTRUCTION
# ==========================================================
workflow = StateGraph(AgentState)

# 1. Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("cleaner", cleaner_node)
workflow.add_node("feature_engineer", fe_node)
workflow.add_node("visualizer", viz_node)
workflow.add_node("trainer", trainer_node)
workflow.add_node("watcher", watcher_node)

# 2. Define Edges (The Logic Flow)
workflow.set_entry_point("router")

# Router -> Cleaner (If task exists) OR Router -> FE (If no clean task)
def route_after_router(state: AgentState):
    if state.get("cleaner_task"):
        return "cleaner"
    elif state.get("fe_task"):
        return "feature_engineer"
    elif state.get("viz_task"):
        return "visualizer"
    elif state.get("trainer_task"):
        return "trainer"
    return END

workflow.add_conditional_edges(
    "router",
    route_after_router
)

# Serial Flow: Cleaner -> FE -> (Viz + Trainer)
# This enforces the dependency chain you requested
workflow.add_edge("cleaner", "watcher")

# Post-Cleaner Flow handled by unified router
# workflow.add_conditional_edges("watcher", route_after_cleaner) <- REMOVED CONFLICT

# FE -> Watcher -> Next
workflow.add_edge("feature_engineer", "watcher") 

def route_after_watcher(state: AgentState):
    """
    Unified routing logic after Watcher review.
    Handles 3-level severity: PASS, WARN, FAIL.
    """
    watcher_status = state.get('watcher_status')
    last = state.get('last_agent')
    print(f"DEBUG: Routing State -> Last: {last}, Watcher: {watcher_status}")
    
    # 1. Handle FAIL - Route back to Router for replanning
    if watcher_status == "FAIL":
        print("❌ FAIL verdict - routing back to Router for replanning...")
        return "router"
    
    # 2. Handle WARN - Log warning but continue pipeline
    if watcher_status == "WARN":
        feedback = state.get('watcher_feedback', '')
        print(f"⚠️  WARN logged: {feedback}. Continuing pipeline...")
        # Fall through to normal routing
    
    # 3. PASS or WARN → Continue normal pipeline flow
    
    # Logic: Output of Cleaner -> Feature Engineer (or Viz/Train if skipped)
    if last == "cleaner":
        if state.get("fe_task"): return "feature_engineer"
        # If no FE task, fall through to Viz/Train checks immediately
        
    # Logic: Output of FE (or Cleaner fell through) -> Parallel Viz/Train
    if last in ["cleaner", "feature_engineer"]:
        next_nodes = []
        if state.get("viz_task"): next_nodes.append("visualizer")
        if state.get("trainer_task"): next_nodes.append("trainer")
        
        if next_nodes:
            print(f"🔀 Branching to parallel nodes: {next_nodes}")
            return next_nodes
            
    return END

# Apply the ONE conditional edge for Watcher
workflow.add_conditional_edges("watcher", route_after_watcher)

# Viz -> End
workflow.add_edge("visualizer", END)
# Trainer -> End
workflow.add_edge("trainer", END)

# Compile
app = workflow.compile()

# ==========================================================
# 🚀 MAIN ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    print("🤖 Autonomous Analytics Crew Started")
    print("Type 'graph' to visualize the workflow or 'q' to quit.")

    while True:
        try:
            user_input = input("\nUser Request: ")
            if not user_input: continue
            if user_input.lower() in ['q', 'exit']: break
            if user_input.lower() == 'graph':
                try:
                    print(app.get_graph().draw_ascii())
                except Exception as e:
                    print(f"Graph viz requires extra dependencies: {e}")
                continue
            
            # --- Slash Commands ---
            if user_input.startswith("/switch"):
                parts = user_input.split()
                if len(parts) < 2:
                    print("Usage: /switch <provider> [model_name]")
                    continue
                provider = parts[1]
                model = parts[2] if len(parts) > 2 else None
                switch_to_provider(provider, model)
                continue
            # ----------------------

            initial_state = {
                "user_request": user_input,
                "messages": [HumanMessage(content=user_input)],
                "original_task": user_input,
                "watcher_status": "PASS",
                "watcher_feedback": ""
            }
            
            # Streaming execution to allow interrupts
            try:
                for output in app.stream(initial_state):
                    pass # Output is handled by print statements in nodes
            except KeyboardInterrupt:
                 print("\n\n🛑 Workflow interrupted by user. Returning to main menu...")
                 continue

            print("\n✅ Workflow Completed.")
        except KeyboardInterrupt:
             print("\n\n🛑 Session interrupted. Exiting...")
             break