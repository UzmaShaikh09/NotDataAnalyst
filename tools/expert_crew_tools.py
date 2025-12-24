import sys
import io
import os
import re
import ast
import subprocess
import pandas as pd
import traceback
from langchain_core.tools import tool

# ==========================================================
# 🛡️ SAFETY & VALIDATION LAYER
# ==========================================================
FORBIDDEN_MODULES = {"os", "subprocess", "shutil", "sys"}
DESTRUCTIVE_KEYWORDS = {"drop", "delete", "remove", "truncate"}

def validate_code(code: str, allow_destructive: bool = False) -> tuple[bool, str]:
    """
    Statically analyzes Python code for safety violations using AST.
    Returns (is_safe, error_message).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"

    for node in ast.walk(tree):
        # 1. Block forbidden imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module]
            for module in modules:
                if module and module.split('.')[0] in FORBIDDEN_MODULES:
                    return False, f"🚫 Security Violation: Importing '{module}' is restricted."

        # 2. Warn/Block destructive DataFrame operations (if strict)
        # Looking for .drop(), .del(), etc.
        if not allow_destructive and isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in DESTRUCTIVE_KEYWORDS:
                    # We flag it, but maybe allow it if it's just 'df.drop_duplicates'
                    # For now, we return a warning message that requires user override (simulated)
                    # In a real Router scenario, the Router would permit this.
                    # For this prototype, we'll allow it but LOG it, or block if strict.
                    # Return a warning flag instead of blocking
                    return True, f"⚠️ Potential mutation detected: '{node.func.attr}'"

        # 3. Check for 'inplace=True' which often implies mutation
        if isinstance(node, ast.keyword) and node.arg == "inplace" and getattr(node.value, "value", False) is True:
             return True, "⚠️ 'inplace=True' detected"

    # 3. Block System Calls via exec/eval
    if "exec(" in code or "eval(" in code:
         return False, "🚫 Security Violation: 'exec' and 'eval' are strictly forbidden."

    return True, ""

# ==========================================================
# 💾 STATE MANAGEMENT HELPER FUNCTIONS
# ==========================================================
CACHE_DIR = "_shared_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Session Management
_CURRENT_SESSION_ID = "default_session"

def set_session_id(session_id: str):
    global _CURRENT_SESSION_ID
    _CURRENT_SESSION_ID = session_id
    # Create session directory
    session_dir = os.path.join(CACHE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    print(f"🔄 Switched to session: {session_id}")

def get_session_path(tag: str) -> str:
    """Returns the path for a dataset within the current session."""
    return os.path.join(CACHE_DIR, _CURRENT_SESSION_ID, f"{tag}.parquet")

def save_df(df: pd.DataFrame, tag: str):
    """Saves DataFrame to the shared cache (Parquet)."""
    global _LAST_SAVED_TAG
    path = get_session_path(tag)
    df.to_parquet(path)
    _LAST_SAVED_TAG = tag # Auto-Update state
    print(f"✅ Data saved to shared storage ({_CURRENT_SESSION_ID}): '{tag}'")

def load_df(tag: str) -> pd.DataFrame:
    """Loads DataFrame from the shared cache."""
    path = get_session_path(tag)
    if not os.path.exists(path):
        # Fallback to default session if not found in current (optional, but good for shared 'raw' data)
        default_path = os.path.join(CACHE_DIR, "default_session", f"{tag}.parquet")
        if os.path.exists(default_path):
             print(f"⚠️ Data not found in {_CURRENT_SESSION_ID}, falling back to default_session.")
             path = default_path
        else:
            raise FileNotFoundError(f"❌ Dataset '{tag}' not found in session '{_CURRENT_SESSION_ID}'.")
    
    print(f"✅ Data loaded from shared storage ({_CURRENT_SESSION_ID}): '{tag}'")
    return pd.read_parquet(path)

def list_data():
    """Lists available datasets in cache for current session."""
    session_dir = os.path.join(CACHE_DIR, _CURRENT_SESSION_ID)
    if not os.path.exists(session_dir):
        return []
    files = [f.replace(".parquet", "") for f in os.listdir(session_dir) if f.endswith(".parquet")]
    print(f"📂 Available Datasets ({_CURRENT_SESSION_ID}): {files}")

# ==========================================================
# 🐍 PYTHON INTERPRETER TOOL
# ==========================================================
# Initialize Global State with Pandas and our Helper Functions
_LAST_SAVED_TAG = None # Tracks the last modified dataset

_INTERPRETER_GLOBALS = {
    "pd": pd, 
    "save_df": save_df, 
    "load_df": load_df,
    "list_data": list_data,
    "set_session_id": set_session_id
}

@tool
def python_interpreter(code: str) -> str:
    """
    Executes Python code for Data Analysis.
    Features:
    - Persistent State (variables kept between calls)
    - Shared Data Access (save_df/load_df)
    - Security Validation (blocks os/subprocess)
    
    Usage:
    - ALWAYS print() results to see them.
    - To save work for next agent: `save_df(df, 'cleaned_data')`
    - To get work from prev agent: `df = load_df('raw_data')`
    """
    
    # 1. Validate Code & Check Safety
    is_valid, safety_msg = validate_code(code, allow_destructive=True)
    if not is_valid:
        return safety_msg
        
    # 2. Interceptor: Proactive User Approval for Mutations
    if "⚠️" in safety_msg:
        print(f"\n✋ [Proactive Safety] Code may modify data: {safety_msg}")
        print(f"Code Preview:\n{code[:300]}...")
        if len(code) > 300: print("<...truncated...>")
        
        try:
            confirm = input("🛑 Allow execution? (y/n): ").lower().strip()
            if confirm != 'y':
                return f"❌ Execution blocked by user. Validation message: {safety_msg}"
        except Exception:
            # If input fails (e.g. non-interactive), we default to BLOCK for safety
            return f"❌ Execution blocked: Could not get user confirmation for: {safety_msg}"

    # 3. Capture Stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    # 3. Execution
    global _INTERPRETER_GLOBALS
    global _LAST_SAVED_TAG
    exec_globals = _INTERPRETER_GLOBALS
    
    # --- AUTO-STATE INJECTION ---
    # If the user tries to use 'df' but didn't define it, 
    # try to auto-load the last saved file.
    if "df" in code and "df" not in exec_globals:
        if _LAST_SAVED_TAG:
            print(f"🔄 Auto-loading last saved dataset: '{_LAST_SAVED_TAG}' as 'df'...")
            try:
                exec_globals["df"] = load_df(_LAST_SAVED_TAG)
            except Exception as e:
                print(f"⚠️ Failed to auto-load: {e}")
        else:
             print("⚠️ 'df' referenced but no previous dataset found in memory.")
    # ----------------------------

    try:
        # Compile and Execute
        compiled = compile(code, "<string>", "exec")
        exec(compiled, exec_globals)
        
        # Eval last expression if possible (REPL style)
        if "_ = " not in code and code.strip().splitlines()[-1] and not code.strip().splitlines()[-1].startswith("print"):
            try:
                last_line = code.strip().splitlines()[-1]
                # Avoid eval if it looks like an assignment or statement
                if "=" not in last_line:
                    last_expr = eval(last_line, exec_globals)
                    if last_expr is not None:
                        print(last_expr)
            except Exception:
                pass
                
        output = sys.stdout.getvalue()
        return output if output.strip() else "✅ Code executed successfully (No output)."
        
    except Exception:
        err = traceback.format_exc()
        return f"❌ Execution Error:\n{err}"
    finally:
        sys.stdout = old_stdout

@tool
def install_package(package_name: str) -> str:
    """
    Installs a Python package using pip. 
    Use this ONLY when you encounter a `ModuleNotFoundError` or `ImportError`.
    The user will be prompted for confirmation before installation proceeds.
    """
    print(f"\n📦 Request to install package: {package_name}")
    user_confirm = input(f"⚠️  Agent wants to install '{package_name}'. Allow? (y/n): ").strip().lower()
    
    if user_confirm != 'y':
        return f"User denied installation of package '{package_name}'."
    
    try:
        print(f"⏳ Installing {package_name}...")
        # Use sys.executable to ensure we install in the current environment
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return f"Successfully installed '{package_name}'."
    except subprocess.CalledProcessError as e:
        return f"Failed to install '{package_name}'. Error: {e}"
    except Exception as e:
        return f"Error installing package: {e}"

def get_expert_tools():
    """
    Returns the toolset for the Expert Agents.
    """
    return [python_interpreter, install_package]