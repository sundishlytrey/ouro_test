# ============================================
# ШАГ 1: ПОЛНЫЙ LLM.PY ДЛЯ GITHUB MODELS
# ============================================

%%writefile /content/ouroboros_repo/ouroboros/llm.py
"""
LLM client for GitHub Models - FULL COMPATIBLE VERSION
Preserves all original functionality but uses free GitHub Models
"""
import os
import time
import json
import requests
from typing import Optional, Dict, Any, List, Union, Tuple
from collections import defaultdict

# ============================================
# КОНФИГУРАЦИЯ МОДЕЛЕЙ
# ============================================
GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"

# Все доступные модели (быстрые и умные)
AVAILABLE_MODELS = {
    # Самые быстрые (для лёгких задач)
    "phi-4-mini": "Phi-4-mini-instruct",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "llama-3.2-11b": "Llama-3.2-11B-Vision-Instruct",
    # Умные (для сложных задач и эволюции)
    "deepseek-r1": "DeepSeek-R1",
    "deepseek-v3": "DeepSeek-V3",
    "mistral-nemo": "Mistral-Nemo",
}

# Маппинг коротких имён в полные
MODEL_NAME_TO_ID = AVAILABLE_MODELS
MODEL_ID_TO_NAME = {v: k for k, v in AVAILABLE_MODELS.items()}

# Модели по умолчанию (как в оригинале)
DEFAULT_MODEL = "phi-4-mini"              # Для обычных задач
DEFAULT_LIGHT_MODEL = "phi-4-mini"        # Для фонового сознания
DEFAULT_CODE_MODEL = "deepseek-r1"        # Для кода (умная)
DEFAULT_REVIEW_MODEL = "deepseek-r1"      # Для ревью (умная)

# Цены (все бесплатно!)
MODEL_PRICES = {name: {"prompt": 0.0, "completion": 0.0} for name in AVAILABLE_MODELS}

# Статистика использования
_usage_stats = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
_last_request_time = 0
_min_request_interval = 60  # 1 запрос в минуту на модель
_request_lock = threading.Lock()

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (оригинальные)
# ============================================
def normalize_reasoning_effort(effort: str) -> float:
    """Convert reasoning effort to temperature (оригинальная функция)"""
    mapping = {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "very_low": 0.1,
        "very_high": 0.9,
    }
    return mapping.get(effort.lower(), 0.5)

def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Approximate token count (оригинальная функция)"""
    return len(text) // 4

def calculate_cost(usage: Dict[str, int], model: str) -> float:
    """Calculate cost (always 0 for GitHub Models)"""
    return 0.0

def add_usage(model: str, prompt_tokens: int, completion_tokens: int):
    """Add usage statistics (оригинальная функция)"""
    _usage_stats[model]["prompt_tokens"] += prompt_tokens
    _usage_stats[model]["completion_tokens"] += completion_tokens
    _usage_stats[model]["cost_usd"] += 0.0

def get_usage_stats() -> Dict:
    """Get all usage statistics"""
    return dict(_usage_stats)

def reset_usage_stats():
    """Reset usage statistics"""
    _usage_stats.clear()

# ============================================
# ОСНОВНОЙ КЛАСС (ПОЛНАЯ ВЕРСИЯ)
# ============================================
class LLMClient:
    """Full-featured LLM client compatible with original Ouroboros"""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """Initialize client with model selection logic"""
        global _last_request_time
        
        # Определяем модель (как в оригинале)
        if model in AVAILABLE_MODELS:
            self.short_name = model
            self.model_name = AVAILABLE_MODELS[model]
        elif model in MODEL_ID_TO_NAME:
            self.model_name = model
            self.short_name = MODEL_ID_TO_NAME[model]
        else:
            # Fallback to default
            self.short_name = DEFAULT_MODEL
            self.model_name = AVAILABLE_MODELS[DEFAULT_MODEL]
            print(f"⚠️ Unknown model '{model}', using {self.short_name}")
        
        # Токен и заголовки
        self.token = os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("❌ GITHUB_TOKEN not found in environment")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        # Статистика последнего запроса
        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0
        }
        
        # Цены для модели
        self.model_prices = MODEL_PRICES.get(self.short_name, {"prompt": 0.0, "completion": 0.0})
        
        print(f"🧠 LLM Client initialized: {self.short_name} -> {self.model_name}")
    
    def default_model(self) -> str:
        """Return default model name (оригинальный метод)"""
        return self.short_name
    
    def _wait_for_rate_limit(self):
        """Thread-safe rate limiting"""
        global _last_request_time
        with _request_lock:
            now = time.time()
            time_since_last = now - _last_request_time
            if time_since_last < _min_request_interval:
                wait_time = _min_request_interval - time_since_last
                print(f"⏳ Rate limit: waiting {wait_time:.0f}s...")
                time.sleep(wait_time)
            _last_request_time = time.time()
    
    def _prepare_messages(self, messages: List[Dict]) -> List[Dict]:
        """Prepare messages in correct format"""
        prepared = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    prepared.append({"role": role, "content": content})
        return prepared
    
    def chat(self, messages: List[Dict], **kwargs) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Main chat method - exact signature expected by original Ouroboros
        Returns: (response_message, usage_dict)
        """
        self._wait_for_rate_limit()
        
        prepared_messages = self._prepare_messages(messages)
        if not prepared_messages:
            return {"content": "No messages", "role": "assistant"}, {"prompt_tokens": 0, "completion_tokens": 0}
        
        # Формируем запрос (поддерживаем все параметры оригинала)
        request_body = {
            "model": self.model_name,
            "messages": prepared_messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4000),
            "stream": False
        }
        
        # Добавляем stop sequences если есть
        if "stop" in kwargs and kwargs["stop"]:
            request_body["stop"] = kwargs["stop"]
        
        # Добавляем top_p если есть
        if "top_p" in kwargs:
            request_body["top_p"] = kwargs["top_p"]
        
        url = f"{GITHUB_MODELS_ENDPOINT}/chat/completions"
        
        try:
            response = requests.post(url, headers=self.headers, json=request_body, timeout=120)
            
            if response.status_code == 429:
                print("⏸️ Rate limit hit, waiting 60s...")
                time.sleep(60)
                return self.chat(messages, **kwargs)  # Рекурсивный повтор
            
            if response.status_code != 200:
                print(f"❌ API error {response.status_code}: {response.text[:200]}")
                return {"content": "", "role": "assistant"}, {"prompt_tokens": 0, "completion_tokens": 0}
            
            result = response.json()
            
            if "choices" not in result or not result["choices"]:
                print("❌ No choices in response")
                return {"content": "", "role": "assistant"}, {"prompt_tokens": 0, "completion_tokens": 0}
            
            # Извлекаем ответ
            content = result["choices"][0]["message"]["content"]
            
            # Статистика использования
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            usage_dict = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": 0.0
            }
            
            # Обновляем статистику
            self.last_usage = usage_dict
            add_usage(self.short_name, prompt_tokens, completion_tokens)
            
            print(f"✅ Response: {len(content)} chars, {prompt_tokens}+{completion_tokens} tokens")
            
            response_message = {
                "content": content.strip(),
                "role": "assistant"
            }
            
            return response_message, usage_dict
            
        except requests.exceptions.Timeout:
            print("⏰ Request timeout")
            return {"content": "", "role": "assistant"}, {"prompt_tokens": 0, "completion_tokens": 0}
        except Exception as e:
            print(f"❌ Error: {e}")
            return {"content": "", "role": "assistant"}, {"prompt_tokens": 0, "completion_tokens": 0}
    
    def generate(self, prompt: Union[str, List[Dict]], system: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate response - compatibility method
        """
        if isinstance(prompt, str):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = prompt
        
        response_msg, usage = self.chat(messages, **kwargs)
        
        return {
            "content": response_msg["content"],
            "usage": usage,
            "model": self.short_name,
            "finish_reason": "stop"
        }
    
    def get_usage(self) -> Dict[str, Any]:
        """Return last usage data"""
        return self.last_usage


# ============================================
# MULTI-CLIENT (ДЛЯ FALLBACK)
# ============================================
class MultiLLMClient:
    """Client that tries multiple models in sequence"""
    
    def __init__(self, models: List[str], fallback_to_any: bool = True):
        self.models = models
        self.fallback_to_any = fallback_to_any
        self.current_client = None
        self.last_error = None
    
    def chat(self, messages: List[Dict], **kwargs) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Try each model in sequence"""
        errors = []
        
        # Try specified models first
        for model_name in self.models:
            try:
                print(f"🔄 Trying {model_name}...")
                client = LLMClient(model=model_name)
                response, usage = client.chat(messages, **kwargs)
                if response.get("content"):
                    self.current_client = client
                    return response, usage
                else:
                    errors.append(f"{model_name}: empty response")
            except Exception as e:
                errors.append(f"{model_name}: {e}")
                continue
        
        # If fallback enabled, try any available model
        if self.fallback_to_any:
            print("⚠️ Trying any available model...")
            for model_name in AVAILABLE_MODELS.keys():
                if model_name not in self.models:
                    try:
                        print(f"🔄 Fallback: {model_name}...")
                        client = LLMClient(model=model_name)
                        response, usage = client.chat(messages, **kwargs)
                        if response.get("content"):
                            self.current_client = client
                            return response, usage
                    except:
                        continue
        
        # All failed
        error_summary = "\n".join(errors[-3:])
        return {"content": "", "role": "assistant"}, {"prompt_tokens": 0, "completion_tokens": 0}


# ============================================
# BACKWARD COMPATIBILITY
# ============================================
def complete(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Simple completion function for backward compatibility"""
    if model is None:
        model = os.environ.get("OUROBOROS_MODEL", DEFAULT_MODEL)
    client = LLMClient(model=model)
    result = client.generate(prompt, **kwargs)
    return result.get("content", "")


# ============================================
# TESTING
# ============================================
if __name__ == "__main__":
    print("\n🔍 Testing LLM Client...")
    client = LLMClient("phi-4-mini")
    response, usage = client.chat([
        {"role": "user", "content": "Say hello in Russian"}
    ])
    print(f"Response: {response['content'][:100]}...")
    print(f"Usage: {usage}")
