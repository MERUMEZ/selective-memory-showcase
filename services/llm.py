"""
================================================================================
 LLM.PY — Клиент OpenRouter API для "Динамического Мозга"
================================================================================
Вся сетевая логика общения с LLM изолирована здесь, чтобы core/cortex.py
оставался чистым "мозговым" модулем без HTTP-деталей.

Основная функция:
    generate_llm_response(messages, system_prompt=None) -> Optional[str]

Возвращает:
    - строку с ответом модели при успехе;
    - None при любой ошибке (нет ключа, таймаут, ошибка API, невалидный
      ответ) — вызывающий код (Cortex) должен интерпретировать None как
      сигнал откатиться на локальный фалбэк/заглушку/эхолалию.

Ничего не бросаем наружу как необработанное исключение — все ошибки
перехватываются и логируются через storage/utils/logger.py.
================================================================================
"""

from typing import List, Dict, Optional

import requests

import config
from storage.utils.logger import get_logger

logger = get_logger(__name__)

CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"


def generate_llm_response(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    """
    Отправляет запрос в OpenRouter Chat Completions API и возвращает
    текст ответа модели.

    Параметры:
        messages       — список сообщений в формате OpenAI-чата:
                          [{"role": "user", "content": "..."}, ...]
        system_prompt  — опциональный system-промпт (например,
                          [MEMORY CONTEXT] из core/cortex.py). Если передан,
                          добавляется в начало messages как role="system".
        max_tokens     — опциональный override для config.LLM_MAX_TOKENS
                          (используется для континуального лимита длины
                          ответа в зависимости от стадии речевого развития,
                          см. Cortex._resolve_max_tokens). Если не передан —
                          используется статичный config.LLM_MAX_TOKENS.

    Возвращает:
        str  — текст ответа модели при успехе.
        None — если ключ не задан, произошла сетевая ошибка, таймаут,
               ошибка авторизации/API, либо ответ пришёл в неожиданном
               формате. В любом из этих случаев вызывающий код (Cortex)
               должен откатиться на локальный фалбэк.
    """
    if not config.OPENROUTER_API_KEY:
        logger.warning(
            "[LLM ERROR] OPENROUTER_API_KEY не задан в .env — LLM-вызов пропущен, "
            "нужен откат на локальный фалбэк"
        )
        return None

    url = config.OPENROUTER_BASE_URL.rstrip("/") + CHAT_COMPLETIONS_ENDPOINT

    final_messages: List[Dict[str, str]] = []
    if system_prompt:
        final_messages.append({"role": "system", "content": system_prompt})
    final_messages.extend(messages)

    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": final_messages,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
    }

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Опциональные заголовки OpenRouter — помогают с рейтингом/аналитикой
        # на их стороне, не обязательны, но рекомендованы в их документации.
        "HTTP-Referer": "https://github.com/mindnumbness",
        "X-Title": "Dynamic AI Brain",
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=config.LLM_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout:
        logger.error("[LLM ERROR] Таймаут запроса к OpenRouter (>%.1fs)", config.LLM_REQUEST_TIMEOUT)
        return None

    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        if status_code == 401:
            logger.error("[LLM ERROR] Ошибка авторизации (401) — проверь OPENROUTER_API_KEY")
        elif status_code == 429:
            logger.error("[LLM ERROR] Превышен лимит запросов (429 Too Many Requests)")
        else:
            logger.error("[LLM ERROR] HTTP ошибка от OpenRouter: status=%s, %s", status_code, exc)
        return None

    except requests.exceptions.RequestException as exc:
        # Покрывает ConnectionError, InvalidURL, SSLError и прочие сетевые сбои
        logger.error("[LLM ERROR] Сетевая ошибка при обращении к OpenRouter: %s", exc)
        return None

    try:
        data = response.json()
        choices = data.get("choices")

        if not choices:
            logger.error("[LLM ERROR] Ответ API не содержит 'choices': %s", data)
            return None

        content = choices[0].get("message", {}).get("content")

        if not content or not content.strip():
            logger.error("[LLM ERROR] Пустой content в ответе модели: %s", data)
            return None

        logger.info(
            "[LLM RESPONSE] model=%s tokens_used=%s",
            config.OPENROUTER_MODEL,
            data.get("usage", {}).get("total_tokens", "n/a"),
        )
        return content.strip()

    except (ValueError, KeyError, IndexError, AttributeError) as exc:
        # ValueError -> невалидный JSON; остальные -> неожиданная структура ответа
        logger.error("[LLM ERROR] Не удалось разобрать ответ API: %s", exc)
        return None