"""
NotDataAnalyst - Main Workflow Orchestrator
============================================

This is the main entry point for the autonomous data analytics system.

Architecture (from base_idea.txt):
1. Contextor: Generates context about data, Q&A with user, clarifies doubts
2. Router: Divides tasks and assigns to expert agents
3. Expert Crew: Cleaner, Feature Engineer, Visualizer, Trainer
4. Watcher: Monitors and provides feedback to prevent hallucinations
5. Shared Memory: All agents share conversation history via Vector Store

Workflow:
---------
Phase 1: Context Building (Contextor)
  - Analyzes dataset structure
  - Asks user clarifying questions
  - Builds comprehensive project context
  - Saves to HotMemory (global) and ColdMemory (persistent)

Phase 2: Task Execution (Router + Experts)
  - Router receives user requests
  - Delegates to appropriate expert agents
  - Watcher validates each agent's output
  - All agents share memory via unified session_id
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from utils.memory_manager import HotMemory
from agents.contextor import chat_loop as run_contextor
from agents.router import app as router_app
from langchain_core.messages import HumanMessage

load_dotenv()

# ==========================================================
# 🎬 MAIN ORCHESTRATOR
# ==========================================================

def main():
    """
    Main workflow orchestrator implementing the base_idea.txt architecture.
    """
    print("=" * 60)
    print("🤖 NotDataAnalyst - Autonomous Analytics System")
    print("=" * 60)
    print("\nBased on Multi-Agent Architecture:")
    print("  └─ Contextor → Router → Expert Crew + Watcher")
    print("  └─ Shared Memory via Vector Store (Qdrant)")
    print()
    
    # Initialize HotMemory to check for existing context
    hot_memory = HotMemory()
    existing_context = hot_memory.get_context()
    
    # ==========================================================
    # PHASE 1: Context Initialization
    # ==========================================================
    if "No global context set" in existing_context:
        print("🧠 No project context found. Starting Contextor...\n")
        print("-" * 60)
        print("PHASE 1: CONTEXT BUILDING")
        print("-" * 60)
        print("\n📋 Contextor will:")
        print("  1. Analyze your dataset structure")
        print("  2. Ask clarifying questions about your goals")
        print("  3. Document rules & constraints")
        print("  4. Build a comprehensive project context\n")
        
        try:
            run_contextor()
            print("\n✅ Context building complete!")
            print("-" * 60)
        except KeyboardInterrupt:
            print("\n\n🛑 Context building interrupted. Exiting...")
            return
        except Exception as e:
            print(f"\n❌ Error during context building: {e}")
            print("You can still proceed to Router, but agents may lack context.")
            user_choice = input("\nContinue to Router anyway? (y/n): ").strip().lower()
            if user_choice != 'y':
                return
    else:
        print("✅ Project context already exists in memory.")
        print(f"\n📄 Context Preview:\n{existing_context[:300]}...\n")
        
        reset_choice = input("Reset context and re-run Contextor? (y/n): ").strip().lower()
        if reset_choice == 'y':
            # Clear existing context
            hot_memory.set_context("")
            print("\n🔄 Context cleared. Restarting Contextor...\n")
            try:
                run_contextor()
                print("\n✅ Context building complete!")
            except KeyboardInterrupt:
                print("\n\n🛑 Context building interrupted. Exiting...")
                return
    
    # ==========================================================
    # PHASE 2: Router Workflow (Task Execution)
    # ==========================================================
    print("\n" + "=" * 60)
    print("PHASE 2: ROUTER WORKFLOW")
    print("=" * 60)
    print("\n🚦 Router is ready to receive tasks.")
    print("\nAvailable Expert Agents:")
    print("  • Cleaner: Fixes datatypes, nulls, duplicates")
    print("  • Feature Engineer: Creates new features, ratios")
    print("  • Visualizer: Generates charts and plots")
    print("  • Trainer: Trains ML models and predictions")
    print("  • Watcher: Validates all outputs (automatic)\n")
    
    print("Commands:")
    print("  'graph'  - Visualize workflow")
    print("  'q'      - Quit")
    print("  Type your data analysis request to begin.\n")
    
    # Start Router Interactive Loop
    from agents.router import SHARED_SESSION_ID
    
    while True:
        try:
            user_input = input("\n💬 Your Request: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['q', 'exit', 'quit']:
                print("\n👋 Exiting NotDataAnalyst. Goodbye!")
                break
            
            if user_input.lower() == 'graph':
                try:
                    print(router_app.get_graph().draw_ascii())
                except Exception as e:
                    print(f"Graph visualization requires extra dependencies: {e}")
                continue
            
            # --- Slash Commands ---
            if user_input.startswith("/switch"):
                from utils.model_manager import switch_to_provider
                parts = user_input.split()
                if len(parts) < 2:
                    print("Usage: /switch <provider> [model_name]")
                    continue
                provider = parts[1]
                model = parts[2] if len(parts) > 2 else None
                switch_to_provider(provider, model)
                continue
            
            if user_input.startswith("/reset"):
                print("\n🔄 Resetting context...")
                hot_memory.set_context("")
                print("✅ Context cleared. Please restart the application to re-run Contextor.")
                continue
            # ----------------------
            
            # Build initial state for Router workflow
            initial_state = {
                "user_request": user_input,
                "messages": [HumanMessage(content=user_input)],
                "original_task": user_input,
                "watcher_status": "PASS",
                "watcher_feedback": ""
            }
            
            # Execute the LangGraph workflow
            print("\n" + "-" * 60)
            try:
                for output in router_app.stream(initial_state):
                    pass  # Output is handled by print statements in nodes
            except KeyboardInterrupt:
                print("\n\n🛑 Workflow interrupted by user. Returning to main menu...")
                continue
            
            print("\n✅ Workflow Completed.")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Session interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

# ==========================================================
# 🚀 ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()