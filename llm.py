# ============================================
# ПОЛНЫЙ ЗАПУСК OUROBOROS С БЕСПЛАТНЫМИ МОДЕЛЯМИ (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ============================================

# 1. УСТАНАВЛИВАЕМ ВСЁ ЧТО НУЖНО
!pip install -q requests python-telegram-bot python-dotenv

# 2. КЛОНИРУЕМ РЕПОЗИТОРИЙ
!git clone https://github.com/razzant/ouroboros.git /content/ouroboros_repo
%cd /content/ouroboros_repo

# 3. ПОЛНОСТЬЮ ЗАМЕНЯЕМ ФАЙЛ LLM.PY НА БЕСПЛАТНУЮ ВЕРСИЮ (С ДОБАВЛЕННЫМИ КОНСТАНТАМИ)
with open('/content/ouroboros_repo/ouroboros/llm.py', 'w') as f:
    f.write('''"""
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

# Available free models on GitHub
MODEL_LIST = {
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral-7b",
    "mistralai/Mistral-Nemo-Instruct-2407": "mistral-nemo",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "mixtral",
    "deepseek-ai/DeepSeek-R1": "deepseek-r1",
    "deepseek-ai/DeepSeek-V3": "deepseek-v3",
    "microsoft/Phi-3.5-mini-instruct": "phi-3.5-mini",
    "microsoft/Phi-3.5-MoE-instruct": "phi-3.5-moe",
    "microsoft/Phi-3.5-vision-instruct": "phi-3.5-vision",
    "microsoft/Phi-4": "phi-4",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "llama-3.2-11b",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "llama-3.2-90b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
    "meta-llama/Llama-Guard-3-11B-Vision": "llama-guard",
    "ai21-ai/Jamba-Instruct": "jamba-instruct",
    "cohere-ai/Command-R": "command-r",
    "cohere-ai/Command-R-Plus": "command-r-plus",
    "nomic-ai/Nomic-Embed-Text-v1.5": "nomic-embed",
}

MODEL_NAME_TO_ID = {v: k for k, v in MODEL_LIST.items()}

# Константы, которые нужны для совместимости с оригинальным кодом
DEFAULT_MODEL = "mistral-nemo"
DEFAULT_LIGHT_MODEL = "phi-3.5-mini"
DEFAULT_CODE_MODEL = "deepseek-r1"

class LLMClient:
    """Client for GitHub Models API (free, token-based)"""
    
    def __init__(self, model: str = "mistralai/Mistral-Nemo-Instruct-2407"):
        if model in MODEL_NAME_TO_ID:
            self.model = MODEL_NAME_TO_ID[model]
        elif model in MODEL_LIST:
            self.model = model
        else:
            print(f"⚠️ Unknown model '{model}', defaulting to Mistral Nemo")
            self.model = "mistralai/Mistral-Nemo-Instruct-2407"
        
        self.token = os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("❌ GITHUB_TOKEN not found in environment")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0
        }
        
        print(f"✅ LLM Client initialized with model: {self.model}")
        print(f"💰 Using GitHub Models - 100% FREE!")
    
    def _prepare_messages(self, prompt: Union[str, List[Dict[str, str]]], system: Optional[str] = None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
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
        messages = self._prepare_messages(prompt, system)
        
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        if stop:
            body["stop"] = stop
        
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
            
            content = result["choices"][0]["message"]["content"]
            
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": 0.0
            }
            
            return {
                "content": content.strip(),
                "usage": self.last_usage,
                "model": self.model,
                "finish_reason": result["choices"][0].get("finish_reason", "stop")
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"GitHub Models API error: {str(e)}"
            return {
                "content": "",
                "error": error_msg,
                "usage": self.last_usage,
                "model": self.model
            }
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def get_usage(self) -> Dict[str, Any]:
        return self.last_usage


class MultiLLMClient:
    """Client that tries multiple models in sequence."""
    
    def __init__(self, models: List[str], fallback_to_any: bool = True):
        self.models = models
        self.fallback_to_any = fallback_to_any
        self.current_client = None
        self.last_error = None
        
    def generate(self, *args, **kwargs):
        errors = []
        
        for model_name in self.models:
            try:
                print(f"🔄 Trying model: {model_name}")
                client = LLMClient(model=model_name)
                result = client.generate(*args, **kwargs)
                
                if result.get("content") and not result.get("error"):
                    self.current_client = client
                    return result
                else:
                    error = result.get("error", "Empty response")
                    errors.append(f"{model_name}: {error}")
                    
            except Exception as e:
                errors.append(f"{model_name}: {str(e)}")
                continue
        
        if self.fallback_to_any:
            print("⚠️ Specified models failed, trying any available model...")
            
            tried_models = set(self.models)
            for full_model in MODEL_LIST.keys():
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
        
        error_summary = "\\n".join(errors[-5:])
        return {
            "content": "",
            "error": f"All models failed. Last errors:\\n{error_summary}",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
            "model": "none"
        }


def complete(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Simple completion function for backward compatibility."""
    if model is None:
        model = os.environ.get("OUROBOROS_MODEL", "mistral-nemo")
    
    client = LLMClient(model=model)
    result = client.generate(prompt, **kwargs)
    
    if result.get("error"):
        print(f"⚠️ Completion error: {result['error']}")
        return ""
    
    return result.get("content", "")
''')

print("✅ Файл llm.py успешно заменён на бесплатную версию (с добавленными константами)!")

# 4. ПОЛУЧАЕМ ТОКЕНЫ ИЗ СЕКРЕТОВ COLAB
from google.colab import userdata
import os

try:
    # Забираем токены из секретов Colab
    GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
    TELEGRAM_BOT_TOKEN = userdata.get('TELEGRAM_BOT_TOKEN')
    
    if not GITHUB_TOKEN or not TELEGRAM_BOT_TOKEN:
        raise ValueError("❌ Не найдены токены в секретах Colab!")
    
    # Устанавливаем переменные окружения
    os.environ["GITHUB_TOKEN"] = GITHUB_TOKEN
    os.environ["TELEGRAM_BOT_TOKEN"] = TELEGRAM_BOT_TOKEN
    
    print("✅ Токены успешно загружены из секретов Colab")
    
except Exception as e:
    print(f"❌ Ошибка получения токенов: {e}")
    print("\n👉 ИНСТРУКЦИЯ:")
    print("1. Нажми на значок 🔑 (Secrets) в левой панели Colab")
    print("2. Добавь два секрета:")
    print("   - Имя: GITHUB_TOKEN    Значение: твой GitHub токен")
    print("   - Имя: TELEGRAM_BOT_TOKEN    Значение: токен твоего бота")
    print("3. Для обоих включи 'Notebook access'")
    print("4. Перезапусти эту ячейку")
    raise

# 5. ТВОИ НАСТРОЙКИ (ЗАМЕНИ ЭТО!)
GITHUB_USERNAME = "sundishlytrey"  # <--- ВСТАВЬ СВОЁ ИМЯ С ГИТХАБА

# 6. НАСТРОЙКИ МОДЕЛЕЙ (ВСЁ БЕСПЛАТНО!)
os.environ["OUROBOROS_MODEL"] = "mistral-nemo"        # Основная модель
os.environ["OUROBOROS_MODEL_CODE"] = "deepseek-r1"    # Для кода
os.environ["OUROBOROS_MODEL_LIGHT"] = "phi-3.5-mini"  # Для фона
os.environ["OUROBOROS_MODEL_FALLBACK_LIST"] = "mistral-nemo,deepseek-r1,phi-3.5-mini,llama-3.2-11b"

# Бюджет (просто заглушка, деньги не тратятся)
os.environ["TOTAL_BUDGET"] = "100"

print("\n🚀 ВСЁ ГОТОВО! Запускаем агента...\n")

# 7. СОЗДАЁМ ФАЙЛ С КОНФИГУРАЦИЕЙ ДЛЯ ЗАПУСКА
with open('/content/ouroboros_repo/run_config.py', 'w') as f:
    f.write(f'''
import os
os.environ["GITHUB_USER"] = "sundishlytrey"
os.environ["GITHUB_REPO"] = "ouro_test"
''')

# 8. ЗАПУСКАЕМ АГЕНТА (обновлённая команда)
!cd /content/ouroboros_repo && python colab_launcher.py --github_user={GITHUB_USERNAME}
