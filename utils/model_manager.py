import os
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

load_dotenv()

# Global override for model switching across all agents
_MANUAL_MODEL_OVERRIDE = None

def switch_to_provider(provider: str, model_id: str = None):
    """
    Globally switches the active model provider for ALL agents.
    """
    global _MANUAL_MODEL_OVERRIDE
    print(f"🔄 Switching global model provider to: {provider}")
    
    # We create a temporary manager just to get the model object
    temp_manager = ModelManager()
    
    # Find a default model for the provider if not specified
    if not model_id:
        if provider == "gemini": model_id = "gemini-2.5-flash"
        elif provider == "groq": model_id = "qwen3-32b"
        elif provider == "openrouter": model_id = "llama-3.3-70b"
        elif provider == "cerebras": model_id = "qwen-3-32b"
    
    try:
        # We bypass the override check here to actually get the new model
        _MANUAL_MODEL_OVERRIDE = temp_manager._create_model_instance(model_id)
        print(f"✅ Global model switched to {model_id}")
    except Exception as e:
        print(f"❌ Failed to switch model: {e}")
    except Exception as e:
        print(f"❌ Failed to switch model: {e}")

def attempt_llm_call(manager, messages, max_retries=3):
    """
    Attempts to call the LLM, automatically switching providers on failure.
    Dynamically reorganizes the provider priority list based on failures.
    """
    # Try current model first
    try:
        return manager.get_model().invoke(messages)
    except Exception as e:
        print(f"⚠️  LLM Call Failed: {e}")
    
    # Extended Fallback sequence including Cerebras
    providers = ["gemini", "groq", "openrouter", "cerebras"]
    
    # Simple rotation: Try each provider in the list
    for provider in providers:
        print(f"🔄 Auto-switching to fallback provider: {provider}...")
        try:
            switch_to_provider(provider)
            # We must fetch the model again after switch
            response = manager.get_model().invoke(messages)
            return response
        except Exception as e:
             print(f"❌ Provider {provider} failed: {e}")
             continue
    
    raise RuntimeError("All LLM providers failed to respond.")

class ModelManager:
    def __init__(self):
        # Default model
        self.current_model_name = "qwen3-32b"
        
        # Define available models configuration
        # Keys are display names or IDs, values are config dicts
        self.models_config = {
            "gemini-2.5-flash": {
                "type": "google",
                "model_name": "gemini-2.5-flash",
                "display_name": "Gemini 2.5 Flash (Google)"
            },
            "gemini-2.5-pro": {
                "type": "google",
                "model_name": "gemini-2.5-pro",
                "display_name": "Gemini 2.5 Pro (Google)"
            },
            "gpt-oss-120b": {
                "type": "cerebras",
                "model_name": "gpt-oss-120b",
                "display_name": "GPT-OSS 120B (Cerebras)"
            },
            "qwen3-32b": {
                "type": "groq",
                "model_name": "qwen/qwen3-32b",
                "display_name": "Qwen3 32B (Groq)"
            },
            "llama-3.3-70b": {
                "type": "openrouter",
                "model_name": "meta-llama/llama-3.3-70b-instruct:free",
                "display_name": "Llama 3.3 70B (OpenRouter)"
            },
            "qwen-3-32b": {
                "type": "cerebras",
                "model_name": "qwen-3-32b",
                "display_name": "Qwen 3 32B (Cerebras)"
            },
            "zai-glm-4.6": {
                "type": "cerebras",
                "model_name": "zai-glm-4.6",
                "display_name": "Zai GLM 4.6 (Cerebras)"
            }
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns a list of available models with metadata."""
        return [
            {"id": key, "name": val["display_name"]} 
            for key, val in self.models_config.items()
        ]

    def get_model(self, model_id: str = None, temperature: float = 0):
        """
        Instantiates and returns the LangChain model object.
        Respects global _MANUAL_MODEL_OVERRIDE if set.
        """
        global _MANUAL_MODEL_OVERRIDE
        if _MANUAL_MODEL_OVERRIDE:
             return _MANUAL_MODEL_OVERRIDE

        if model_id is None:
            model_id = self.current_model_name
        
        return self._create_model_instance(model_id, temperature)

    def _create_model_instance(self, model_id: str, temperature: float = 0):
        """Internal method to create the model object."""
        if model_id not in self.models_config:
            raise ValueError(f"Model {model_id} not found.")

        config = self.models_config[model_id]
        model_type = config["type"]
        name = config["model_name"]

        # Update current default if successful
        self.current_model_name = model_id

        try:
            if model_type == "google":
                return ChatGoogleGenerativeAI(
                    model=name,
                    temperature=temperature
                )
            
            elif model_type == "groq":
                # Groq uses OpenAI-compatible API
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    print("⚠️  WARNING: GROQ_API_KEY not found in environment.")
                return ChatOpenAI(
                    model=name,
                    temperature=temperature,
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                
            elif model_type == "openrouter":
                # OpenRouter uses OpenAI-compatible API
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    print("⚠️  WARNING: OPENROUTER_API_KEY not found in environment.")
                return ChatOpenAI(
                    model=name,
                    temperature=temperature,
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1" 
                )

            elif model_type == "cerebras":
                # Cerebras uses OpenAI-compatible API
                api_key = os.getenv("CEREBRAS_API_KEY")
                if not api_key:
                    print("⚠️  WARNING: CEREBRAS_API_KEY not found in environment.")
                return ChatOpenAI(
                    model=name,
                    temperature=temperature,
                    api_key=api_key,
                    base_url="https://api.cerebras.ai/v1"
                )
        except Exception as e:
            print(f"❌ Error initializing model {model_id}: {e}")
            # Fallback to default if everything fails to avoid crash
            print("Defaulting back to Gemini 2.5 Flash...")
            self.current_model_name = "gemini-2.5-flash"
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
