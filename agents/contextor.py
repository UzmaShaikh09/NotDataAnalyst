import os
import sys
import json
import threading
from dotenv import load_dotenv
from typing import Optional, List

# Add project root to path so we can import 'tools' and 'utils'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

# --- Import Data Analysis Tools ---
from tools.contextor_tools import get_data_context

# --- Import Memory Functions ---
# Contextor needs direct write access to Cold Storage for initializing the project
from utils.qdrant_setup import mcp_wrapper
from utils.qdrant_setup import update_context 
from utils.memory_manager import HotMemory, WarmMemory
from utils.model_manager import ModelManager

# ===============================================================
# ⚙️ ENV + MODEL SETUP
# ===============================================================
load_dotenv()
model_manager = ModelManager()
Contextor = model_manager.get_model(temperature=0)

# ===============================================================
# 🧵 BACKGROUND PROCESS
# ===============================================================
def process_and_save_summary(raw_summary: str, file_path: str):
    """
    Background task to:
    1. Improve the raw data summary using Contextor.
    2. Save it to Qdrant (Cold Memory) under section="Data_summary".
    """
    try:
        # Create a prompt to improve the summary
        improvement_prompt = [
            SystemMessage(content=(
                "You are an expert Data Analyst. Your task is to refine and improve the following raw dataset summary. "
                "Make it more concise, highlight key insights, and structure it for better readability by other AI agents. "
                "Do NOT lose important technical details like column names or data types."
                "Never add extra keywords in final context. just give the final context document without any extra keywords also don't add Headings or start/end text just our 3 core sections"
            )),
            HumanMessage(content=f"Here is the raw summary:\n\n{raw_summary}")
        ]
        
        # Invoke Contextor to get improved summary
        # Note: We use a separate invocation here, not affecting the main chat memory
        improved_response = Contextor.invoke(improvement_prompt)
        improved_summary = str(improved_response.content).strip()
        
        # Save to Qdrant (Cold Storage)
        update_context(
            improved_summary,
            dataset=file_path,
            agent="Contextor",
            section="Data_summary"
        )
        # print("\n[Background] Data summary improved and saved to 'Data_summary'.") 
        
    except Exception as e:
        print(f"\n❌ [Background] Error processing data summary: {e}")

# ===============================================================
# 🧠 HELPER: History Reconstruction
# ===============================================================
def build_langchain_history(system_prompt: str, warm_memory: WarmMemory) -> List:
    """
    Reconstructs the LangChain message history from WarmMemory (Redis).
    This ensures the LLM always sees the 'Hot' window of context.
    """
    messages = [SystemMessage(content=system_prompt)]
    
    # Fetch recent chat history (Last 20 messages)
    recent_chat = warm_memory.get_recent_messages(limit=20)
    
    for msg in recent_chat:
        role = msg.get("role")
        content = msg.get("content")
        
        if role.lower() == "user":
            messages.append(HumanMessage(content=content))
        elif role.lower() in ["ai", "contextor", "assistant"]:
            messages.append(AIMessage(content=content))
            
    return messages

# ===============================================================
# 💬 MAIN CHAT LOOP
# ===============================================================
def chat_loop():
    print("Test session started! Type 'exit' to quit.\n")

    # Ask for file path once
    file_path = input("Enter dataset file/folder path or database connection string: ").strip()
    if not os.path.exists(file_path):
        print("❌ File not found. Exiting.")
        return

    # Initialize Warm Memory for this session
    # Using a deterministic ID for this setup phase
    session_id = f"setup_{os.path.basename(file_path).replace(' ', '_')}"
    warm_memory = WarmMemory(session_id=session_id)
    
    # Clear previous setup attempts for a clean slate
    warm_memory.clear_session() 

    print("\n📊 Collecting Context...\n")
    try:
        # Get raw summary
        dataset_summary = get_data_context(file_path)
        print("✅ Dataset analyzed successfully.\n")
        
        # Save structured metadata to Warm Memory (Redis) for fast access
        warm_memory.save_metadata("dataset_path", file_path)
        
        # Start background thread to improve and save summary to Cold Storage
        summary_thread = threading.Thread(
            target=process_and_save_summary, 
            args=(dataset_summary, file_path),
            daemon=True
        )
        summary_thread.start()
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return

    # Define the System Prompt
    system_prompt_content = (
        "You are **Contextor**, the lead agent of an autonomous analytics team.\n\n"
        "🎯 **Mission:** Build a structured and comprehensive **PROJECT CONTEXT DOCUMENT**. "
        "Your output will guide all other agents (analysts, engineers, BI specialists, etc.).\n\n"
        "🧩 **Your job includes 3 key sections — in this exact order:**\n"
        "1️⃣ **DATA CONTEXT:** Describe the dataset provided below — its purpose, structure, data quality, "
        "key columns, null patterns, and what it can potentially reveal.\n"
        "2️⃣ **PROJECT CONTEXT:** Based on the user's answers, describe their ultimate goal or use case, "
        "intended deliverables, audience, and what success looks like.\n"
        "3️⃣ **RULES & CONSTRAINTS:** Ask and document how the user wants the agents to behave — "
        "e.g., priorities (speed vs accuracy), visualization preferences, tool limits, naming conventions, etc.\n\n"
        "🧠 **Conversation Rules:**\n"
        "1. Ask **3–5 smart questions** — one at a time.\n"
        "2. Always include a question about rules or preferences.\n"
        "3. End the conversation only when all three sections are clear.\n"
        "4. When your final context is ready, output all 3 sections in order, clearly labeled, and end with **DONE**.\n"
        "Never add extra keywords in final context. just give the final context document and include the full dataset path\n\n"
        f"📍 **Dataset Location:** {file_path}\n\n"
        f"📊 **Dataset Summary (for Data Context section):**\n{dataset_summary}"
    )

    print("🤖 Contextor is preparing few Questions, this helps agents understand more about the Project...\n")
    print("Vague Answers Leads to Vague AI behaviour\n")
    
    # --- Bootstrapping the Conversation ---
    # We manually inject the first user trigger into memory to start the loop
    warm_memory.add_message("user", "Please begin the conversation by asking your first question.")

    # Generate first response
    history = build_langchain_history(system_prompt_content, warm_memory)
    first_response = Contextor.invoke(history)
    
    if isinstance(first_response.content, list):
        first_text = " ".join([p.get("text", "") for p in first_response.content if isinstance(p, dict)]).strip()
    else:
        first_text = str(first_response.content).strip()

    warm_memory.add_message("Contextor", first_text)
    print(f"AI: {first_text}\n")

    # Interactive loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Chat ended manually.")
            break

        # 1. Add User Input to Warm Memory (Redis)
        # This automatically handles archiving to Cold Memory if list > 20
        warm_memory.add_message("User", user_input)

        # 2. Reconstruct History from Warm Memory
        history = build_langchain_history(system_prompt_content, warm_memory)

        # 3. Invoke LLM
        response = Contextor.invoke(history)
        if isinstance(response.content, list):
            ai_text = " ".join([p.get("text", "") for p in response.content if isinstance(p, dict)]).strip()
        else:
            ai_text = str(response.content).strip()

        # 4. Add AI Response to Warm Memory
        warm_memory.add_message("Contextor", ai_text)
        print(f"\nAI: {ai_text}\n")

        # 5. Check for Completion
        if "DONE" in ai_text:
            print("✅ Context generation completed.\n")
            final_context = ai_text.replace("DONE", "").strip()

            # --- A) Save to Hot Memory (Global System Prompt) ---
            # This makes the context immediately available to other agents in the runtime
            HotMemory().set_context(final_context)
            print("🔥 Global Context (Hot Memory) updated!")

            # --- B) Save to Cold Memory (Persistent Qdrant) ---
            # This ensures we can retrieve this context even after a restart
            update_context(
                final_context,
                dataset=file_path,
                agent="Contextor",
                section="Context"
            )
            print("🧊 Persistent Context (Cold Memory) saved!")
            
            # Optional: Clear the setup session from Redis as it's done
            # warm_memory.clear_session() 
            break

if __name__ == "__main__":
    chat_loop()