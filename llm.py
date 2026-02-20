# ouroboros/llm.py
# Полная замена: вместо OpenRouter используем бесплатный Google Gemini API

import os
import json
import time
import asyncio
import aiohttp
import tiktoken
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple
from dataclasses import dataclass, field

# Импорты из проекта
from .utils import get_logger, truncate_text
from .exceptions import LLMError, BudgetExceededError, LLMResponseError

logger = get_logger(__name__)

# Конфигурация по умолчанию для бесплатной модели
DEFAULT_MODEL = "gemini-2.0-flash-exp"  # Быстрая и бесплатная модель
DEFAULT_MAX_TOKENS = 8192  # Стандартный лимит для вывода
DEFAULT_TEMPERATURE = 0.7

# Цены (условные, чтобы не ломалась система бюджета, т.к. Gemini бесплатный)
# Ставим минимальную цену, чтобы бюджет не утекал, но система работала.
GEMINI_PRICING = {
    "gemini-2.0-flash-exp": {"prompt": 0.0000001, "completion": 0.0000001},  # Цена за токен в USD
    "gemini-1.5-flash": {"prompt": 0.0000001, "completion": 0.0000001},
    "gemini-1.5-pro": {"prompt": 0.0000001, "completion": 0.0000001},
}

@dataclass
class LLMUsage:
    """Следит за использованием токенов и стоимостью (для совместимости с системой бюджета)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    model: str = DEFAULT_MODEL

    def add(self, other: 'LLMUsage'):
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost += other.cost

class OpenRouterClient:
    """
    Полностью переписанный клиент.
    Теперь он работает с бесплатным Google Gemini API через Google AI Studio.
    Название класса оставлено для совместимости, чтобы не менять другие файлы.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        budget_usd: Optional[float] = None,
    ):
        """
        Инициализация клиента для Gemini API.

        Аргументы:
            api_key: API-ключ Google AI Studio. Если None, берется из переменной окружения GOOGLE_API_KEY.
            model: Название модели Gemini (например, "gemini-2.0-flash-exp", "gemini-1.5-flash").
            max_tokens: Максимальное количество токенов в ответе.
            temperature: Температура (креативность) модели.
            budget_usd: Лимит бюджета (для совместимости, но Gemini бесплатный).
        """
        # Приоритет: аргумент -> переменная окружения -> ошибка
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            # Если нет ключа Gemini, пробуем старый ключ OpenRouter (на всякий случай)
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if self.api_key:
                logger.warning("GOOGLE_API_KEY not found, but OPENROUTER_API_KEY is set. Please set GOOGLE_API_KEY for free Gemini access.")
                # Мы всё равно не будем использовать OpenRouter, но предупредим пользователя.
            else:
                raise ValueError("GOOGLE_API_KEY environment variable not set. Get a free key from Google AI Studio.")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.budget_usd = budget_usd
        self.total_usage = LLMUsage()

        # Базовый URL для Gemini API
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        logger.info(f"Initialized Gemini client with model: {self.model}")

    def _count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Подсчет токенов с помощью tiktoken (приблизительно).
        """
        try:
            # Используем кодировку cl100k_base (для GPT-4, работает как приближение)
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Approximating by chars.")
            # Грубое приближение: 1 токен ≈ 4 символа
            return len(text) // 4

    async def _make_gemini_request(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Внутренний метод для отправки запроса к Gemini API.
        Преобразует сообщения из формата OpenAI/OpenRouter в формат Gemini.
        """
        # --- 1. Преобразование сообщений в формат Gemini ---
        # Gemini ожидает список частей (parts) с текстом.
        # Системное сообщение передается отдельно.
        system_instruction = None
        gemini_contents = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                gemini_contents.append({
                    "role": "model",  # В Gemini роль ассистента называется "model"
                    "parts": [{"text": content}]
                })
            # Роль "tool" пока игнорируем для простоты (в базовой версии агента она не используется)

        # --- 2. Формирование тела запроса ---
        request_body: Dict[str, Any] = {
            "contents": gemini_contents,
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": self.temperature,
            }
        }

        if system_instruction:
            request_body["system_instruction"] = {
                "parts": [{"text": system_instruction}]
            }

        # --- 3. URL для API вызова (с потоком или без) ---
        url = f"{self.base_url}/models/{self.model}:generateContent"
        if stream:
            url = f"{self.base_url}/models/{self.model}:streamGenerateContent"

        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}

        # --- 4. Логирование (уровень DEBUG) ---
        logger.debug(f"Gemini request to {url}")
        logger.debug(f"Request body (truncated): {truncate_text(json.dumps(request_body), 500)}")

        # --- 5. Отправка запроса ---
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, params=params, headers=headers, json=request_body) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Gemini API error {resp.status}: {error_text}")
                        raise LLMResponseError(f"Gemini API error: {resp.status} - {error_text}")

                    # Обработка потокового ответа (упрощенно)
                    if stream:
                        # Для простоты пока не реализуем полноценный стриминг,
                        # просто вернем первый чанк как полный ответ.
                        # В будущем можно добавить, но для работы агента это не критично.
                        response_data = await resp.json()
                        # В стриминге ответ может быть массивом чанков
                        if isinstance(response_data, list):
                            full_response = {"candidates": [{"content": {"parts": []}}]}
                            for chunk in response_data:
                                if "candidates" in chunk:
                                    for cand in chunk["candidates"]:
                                        if "content" in cand and "parts" in cand["content"]:
                                            full_response["candidates"][0]["content"]["parts"].extend(cand["content"]["parts"])
                            response_data = full_response
                    else:
                        response_data = await resp.json()

                    logger.debug(f"Gemini response (truncated): {truncate_text(json.dumps(response_data), 500)}")
                    return response_data

            except aiohttp.ClientError as e:
                logger.error(f"Network error during Gemini API call: {e}")
                raise LLMError(f"Network error: {e}") from e

    def _parse_gemini_response(self, response_data: Dict[str, Any]) -> Tuple[str, LLMUsage]:
        """
        Разбор ответа от Gemini и извлечение текста + подсчет использования.
        """
        try:
            # Извлекаем текст ответа
            text = ""
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            text += part["text"]

            if not text:
                # Если ответ пустой или заблокирован
                block_reason = response_data.get("promptFeedback", {}).get("blockReason", "Unknown")
                logger.warning(f"Empty or blocked Gemini response. Block reason: {block_reason}")
                # Возвращаем пустую строку, как это делает OpenRouter при ошибке
                text = ""

            # --- Подсчет токенов и стоимости (для системы бюджета) ---
            prompt_tokens = 0
            completion_tokens = 0

            # Пробуем получить точные цифры из ответа (если есть)
            if "usageMetadata" in response_data:
                usage = response_data["usageMetadata"]
                prompt_tokens = usage.get("promptTokenCount", 0)
                completion_tokens = usage.get("candidatesTokenCount", 0)
            else:
                # Если нет, считаем приблизительно
                # Нам нужно знать, какой был промпт. У нас его нет в этом методе.
                # Оставим 0, система бюджета не сломается.
                logger.debug("No usage metadata in Gemini response, using 0 for token counts.")

            total_tokens = prompt_tokens + completion_tokens

            # Расчет стоимости по нашим условным ценам
            pricing = GEMINI_PRICING.get(self.model, {"prompt": 0.0, "completion": 0.0})
            cost = (prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"])

            usage = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
                model=self.model,
            )

            return text, usage

        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}", exc_info=True)
            raise LLMResponseError(f"Response parsing failed: {e}") from e

    async def completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
    ) -> Tuple[str, LLMUsage]:
        """
        Основной метод для получения ответа от модели.
        Полностью заменяет старый метод OpenRouter.
        """
        # Используем параметры экземпляра или переопределенные
        model = model or self.model
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Проверка бюджета (если задан)
        if self.budget_usd is not None and self.total_usage.cost >= self.budget_usd:
            logger.error(f"Budget exceeded: {self.total_usage.cost:.6f} >= {self.budget_usd}")
            raise BudgetExceededError(f"Budget exceeded: {self.total_usage.cost:.6f} >= {self.budget_usd}")

        # Сохраняем текущую модель, чтобы потом вернуть, если меняли
        original_model = self.model
        self.model = model

        try:
            # Делаем запрос к Gemini
            response_data = await self._make_gemini_request(messages, stream=stream, tools=tools)

            # Парсим ответ
            text, usage = self._parse_gemini_response(response_data)

            # Обновляем общее использование
            self.total_usage.add(usage)

            # Логируем использование
            logger.debug(f"LLM call to {model}: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} tokens, cost ${usage.cost:.6f}")

            return text, usage

        except Exception as e:
            logger.error(f"LLM completion failed: {e}", exc_info=True)
            # Пробрасываем исключение дальше
            if isinstance(e, (LLMError, BudgetExceededError)):
                raise
            raise LLMError(f"LLM completion failed: {e}") from e
        finally:
            # Восстанавливаем модель
            self.model = original_model

    async def close(self):
        """Закрытие клиента (ничего не делаем, но метод нужен для совместимости)."""
        pass
