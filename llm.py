# ouroboros/llm.py
"""
LLM client for GitHub Models (Mistral, DeepSeek, Phi, Llama).
Supports multiple free models with fallback.
For Russia: no credits, no OpenRouter, just a GitHub token.
"""
import os
import json
import time
import requests
from typing import Optional, Dict, Any, List, Union

# GitHub Models inference endpoint
GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"

# Available free models on GitHub (as of 2025)
# See: https://github.com/marketplace?type=models
MODEL_LIST = {
    # Mistral family
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral-7b",
    "mistralai/Mistral-Nemo-Instruct-2407": "mistral-nemo",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "mixtral",
    # DeepSeek
    "deepseek-ai/DeepSeek-R1": "deepseek-r1",
    "deepseek-ai/DeepSeek-V3": "deepseek-v3",
    # Microsoft Phi
    "microsoft/Phi-3.5-mini-instruct": "phi-3.5-mini",
    "microsoft/Phi-3.5-MoE-instruct": "phi-3.5-moe",
    "microsoft/Phi-3.5-vision-instruct": "phi-3.5-vision",
    "microsoft/Phi-4": "phi-4",
    # Meta Llama
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "llama-3.2-11b",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "llama-3.2-90b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
    "meta-llama/Llama-Guard-3-11B-Vision": "llama-guard",
    # AI21
    "ai21-ai/Jamba-Instruct": "jamba-instruct",
    # Cohere
    "cohere-ai/Command-R": "command-r",
    "cohere-ai/Command-R-Plus": "command-r-plus",
    # Others
    "nomic-ai/Nomic-Embed-Text-v1.5": "nomic-embed",
}

# Reverse mapping for model names
MODEL_NAME_TO_ID = {v: k for k, v in MODEL_LIST.items()}

class LLMClient:
    """Client for GitHub Models API (free, token-based)"""
    
    def __init__(self, model: str = "mistralai/Mistral-Nemo-Instruct-2407"):
        """
        Initialize the client.
        
        Args:
            model: Model identifier (can be full path or short name like 'mistral-nemo')
        """
        # Check if it's a short name, convert to full path
        if model in MODEL_NAME_TO_ID:
            self.model = MODEL_NAME_TO_ID[model]
        elif model in MODEL_LIST:
            self.model = model
        else:
            # Default to Mistral Nemo if unknown
            print(f"⚠️ Unknown model '{model}', defaulting to Mistral Nemo")
            self.model = "mistralai/Mistral-Nemo-Instruct-2407"
        
        # Get token from environment
        self.token = os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("❌ GITHUB_TOKEN not found in environment. "
                           "Please add your GitHub token to Colab secrets or environment.")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        # Pricing is FREE, but we keep budget tracking for compatibility
        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0  # Always zero - free!
        }
        
        print(f"✅ LLM Client initialized with model: {self.model}")
        print(f"💰 Using GitHub Models - 100% FREE for Russia!")
    
    def _prepare_messages(self, prompt: Union[str, List[Dict[str, str]]], system: Optional[str] = None):
        """Convert various prompt formats to chat messages format."""
        messages = []
        
        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})
        
        # Handle different prompt types
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            # Assume it's already in message format
            messages.extend(prompt)
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        
        return messages
    
    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from the model.
        
        Args:
            prompt: User prompt (string or message list)
            system: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Optional stop sequences
            **kwargs: Additional parameters (ignored for GitHub API)
            
        Returns:
            Dictionary with 'content', 'usage', and other metadata
        """
        messages = self._prepare_messages(prompt, system)
        
        # Prepare request body
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        if stop:
            body["stop"] = stop
        
        # Make the API call
        url = f"{GITHUB_MODELS_ENDPOINT}/chat/completions"
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=body,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract content
            content = result["choices"][0]["message"]["content"]
            
            # Update usage tracking (GitHub provides token counts)
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": 0.0  # Always free!
            }
            
            return {
                "content": content.strip(),
                "usage": self.last_usage,
                "model": self.model,
                "finish_reason": result["choices"][0].get("finish_reason", "stop")
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"GitHub Models API error: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f" - {json.dumps(error_detail)}"
                except:
                    error_msg += f" - {e.response.text[:200]}"
            
            print(f"❌ {error_msg}")
            
            # Return error response
            return {
                "content": "",
                "error": error_msg,
                "usage": self.last_usage,
                "model": self.model
            }
    
    def count_tokens(self, text: str) -> int:
        """
        Rough token estimation (GitHub doesn't provide this endpoint).
        Using approximate 4 chars per token.
        """
        return len(text) // 4
    
    def get_usage(self) -> Dict[str, Any]:
        """Return last usage data."""
        return self.last_usage


# Optional: Wrapper for multiple models with fallback
class MultiLLMClient:
    """
    Client that tries multiple models in sequence.
    Useful for fallback when one model is rate-limited.
    """
    
    def __init__(self, models: List[str], fallback_to_any: bool = True):
        """
        Args:
            models: List of model names (can be short or full)
            fallback_to_any: If True, try any available model on failure
        """
        self.models = models
        self.fallback_to_any = fallback_to_any
        self.current_client = None
        self.last_error = None
        
    def generate(self, *args, **kwargs):
        """Try each model in sequence until one works."""
        
        errors = []
        
        # Try specified models in order
        for model_name in self.models:
            try:
                print(f"🔄 Trying model: {model_name}")
                client = LLMClient(model=model_name)
                result = client.generate(*args, **kwargs)
                
                # Check if successful (has content and no error)
                if result.get("content") and not result.get("error"):
                    self.current_client = client
                    return result
                else:
                    error = result.get("error", "Empty response")
                    errors.append(f"{model_name}: {error}")
                    
            except Exception as e:
                errors.append(f"{model_name}: {str(e)}")
                continue
        
        # If we have fallback enabled, try ANY available model
        if self.fallback_to_any:
            print("⚠️ Specified models failed, trying any available model...")
            
            # Try all models in MODEL_LIST (excluding ones we already tried)
            tried_models = set(self.models)
            for full_model in MODEL_LIST.keys():
                # Extract short name if needed
                for short, full in MODEL_NAME_TO_ID.items():
                    if full == full_model and short not in tried_models:
                        try:
                            print(f"🔄 Fallback trying: {short}")
                            client = LLMClient(model=short)
                            result = client.generate(*args, **kwargs)
                            
                            if result.get("content") and not result.get("error"):
                                self.current_client = client
                                return result
                            else:
                                error = result.get("error", "Empty response")
                                errors.append(f"{short}: {error}")
                        except Exception as e:
                            errors.append(f"{short}: {str(e)}")
                        break
        
        # All models failed
        error_summary = "\n".join(errors[-5:])  # Show last 5 errors
        return {
            "content": "",
            "error": f"All models failed. Last errors:\n{error_summary}",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
            "model": "none"
        }


# Backward compatibility function for existing code
def complete(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """
    Simple completion function for backward compatibility.
    """
    if model is None:
        model = os.environ.get("OUROBOROS_MODEL", "mistral-nemo")
    
    client = LLMClient(model=model)
    result = client.generate(prompt, **kwargs)
    
    if result.get("error"):
        print(f"⚠️ Completion error: {result['error']}")
        return ""
    
    return result.get("content", "")


# For direct testing
if __name__ == "__main__":
    # Test the client
    print("Testing LLM client with GitHub Models...")
    
    # Test with Mistral Nemo
    client = LLMClient("mistral-nemo")
    response = client.generate("Say hello in Russian")
    print(f"Response: {response.get('content')}")
    print(f"Usage: {client.get_usage()}")
    
    # Test fallback
    multi = MultiLLMClient(["mistral-7b", "phi-3.5-mini"])
    response = multi.generate("What is 2+2?")
    print(f"\nMulti response: {response.get('content')}")
