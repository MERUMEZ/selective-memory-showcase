"""
================================================================================
 BOT.PY — Telegram-интерфейс "Динамического Мозга" (aiogram 3.x)
================================================================================
Этап 3 миграции CLI -> Telegram-бот.

Каждый telegram_user_id получает собственный BrainSession (через
SessionManager, см. core/session_manager.py) с изолированной SQLite БД
в config.BRAIN_DB_DIR/{user_id}.db.

BrainSession.process_message() — синхронный (SQLite + синхронные HTTP-
запросы к LLM внутри services/llm.py), поэтому вызывается через
asyncio.to_thread(...), чтобы не блокировать event loop aiogram.

Пользователю в чат уходит ТОЛЬКО response.text — весь debug-вывод
(perplexity, mood, reward trace и т.п.) остаётся в логах через
storage/utils/logger.py, как и в CLI-версии (main.py).

НЕ реализовано на этом этапе (по плану):
    - фоновый asyncio-тик Idle Sleep / Boredom для всех сессий (Этап 5);
    - eviction неактивных сессий по SESSION_EVICTION_IDLE_SECONDS (Этап 6).
================================================================================
"""

import asyncio
import sys

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

import config
from core.session_manager import SessionManager
from storage.utils.logger import get_logger

logger = get_logger(__name__)

router = Router()

# Единый на весь процесс реестр "мозгов" — по одному BrainSession на
# каждого telegram_user_id. Инициализируется в main() и передаётся сюда
# через router-level middleware/замыкание (см. get_manager ниже).
_session_manager: SessionManager = None  # type: ignore[assignment]


def get_manager() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("SessionManager не инициализирован (main() не был вызван)")
    return _session_manager


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """Приветственное сообщение при первом /start — неявно создаёт сессию."""
    manager = get_manager()
    user_id = message.from_user.id

    # get_or_create синхронный (SQLite init) -> уводим в поток, чтобы не
    # блокировать event loop на первом обращении пользователя.
    await asyncio.to_thread(manager.get_or_create, user_id)

    await message.answer(
        "Привет! Я — растущий цифровой разум. Пиши мне что угодно, "
        "я учусь у тебя языку и понятиям с нуля.\n\n"
        "Команда /sleep — запустить фазу консолидации памяти (сон)."
    )


@router.message(Command("sleep"))
async def handle_sleep(message: Message) -> None:
    """Явный запуск фазы сна (BrainSession.process_message понимает '/sleep')."""
    manager = get_manager()
    user_id = message.from_user.id

    session = await asyncio.to_thread(manager.get_or_create, user_id)
    response = await asyncio.to_thread(session.process_message, "/sleep")

    await message.answer(response.text)


@router.message()
async def handle_text(message: Message) -> None:
    """Основной хендлер: любое обычное текстовое сообщение -> BrainSession."""
    if not message.text:
        await message.answer("Я пока умею понимать только текст.")
        return

    manager = get_manager()
    user_id = message.from_user.id

    session = await asyncio.to_thread(manager.get_or_create, user_id)

    try:
        response = await asyncio.to_thread(session.process_message, message.text)
    except Exception:  # noqa: BLE001
        logger.exception("[BOT] Ошибка обработки сообщения user_id=%s", user_id)
        await message.answer("Ой, что-то пошло не так внутри меня. Попробуй ещё раз чуть позже.")
        return

    if response.text:
        await message.answer(response.text)


async def _on_shutdown() -> None:
    """Graceful shutdown: закрываем все активные BrainSession (SQLite-соединения)."""
    logger.info("[BOT] Остановка: закрываю все активные сессии...")
    manager = get_manager()
    await asyncio.to_thread(manager.close_all)
    logger.info("[BOT] Все сессии закрыты, завершение процесса.")


async def main() -> None:
    global _session_manager

    if not config.TELEGRAM_BOT_TOKEN:
        logger.error(
            "[BOT] TELEGRAM_BOT_TOKEN не задан в .env — запуск невозможен."
        )
        sys.exit(1)

    _session_manager = SessionManager(db_dir=config.BRAIN_DB_DIR)

    bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    dp.shutdown.register(_on_shutdown)

    logger.info("[BOT] Запуск Telegram-бота (polling)...")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("[BOT] Остановлено пользователем.")