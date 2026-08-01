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

ЭТАП 5/6 (фоновый Idle Sleep/Boredom-тик + eviction неактивных сессий):
    Единый asyncio-таск _idle_scheduler_loop, запускаемый в main() и
    останавливаемый в _on_shutdown(), раз в config.BOT_SCHEDULER_TICK_SECONDS
    обходит SessionManager.all_sessions() и для каждой сессии:
        - выгружает её из памяти при бездействии >= SESSION_EVICTION_IDLE_SECONDS;
        - иначе продвигает brain_time через BrainSession.run_idle_tick()
          и отправляет пользователю проактивное сообщение, если оно
          было сгенерировано (Boredom Drive Trigger).
    В отличие от main.py (один поток threading на весь процесс), здесь
    один общий asyncio-таск обслуживает ВСЕ сессии последовательно —
    что приемлемо, т.к. каждая итерация тика per-сессии дешёвая (SQLite),
    а сама генерация LLM (при boredom-триггере) уходит в отдельный поток
    через asyncio.to_thread, не блокируя обход остальных сессий надолго.
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

# Фоновый asyncio-таск Idle Sleep/Boredom-тика + eviction (Этап 5/6),
# запускается в main(), отменяется в _on_shutdown().
_scheduler_task: asyncio.Task = None  # type: ignore[assignment]


# Сценарий /demo. Порядок продуман, а не случаен:
#   1-6    повторяющиеся бытовые реплики — набирается словарь, лепет
#          сменяется отдельными словами;
#   7-8    важный факт и похвала за него — включается подкрепление;
#   9-15   к факту ВОЗВРАЩАЮТСЯ и хвалят снова, вперемешку с рутиной.
#
# Возвращения не для красоты: удержание держится на стабильности, а она
# растёт от ВСПОМИНАНИЯ, не от записи. Замер на этом сценарии: после
# недели молчания выживает ровно один эпизод — телефон (вес 0.14), а
# рутина и даже сама похвала уходят. Более короткий сценарий не
# показывает ничего: за пятнадцать реплик стабильность не набирается, и
# через неделю пусто везде, то есть разницы не видно.
DEMO_SCRIPT = [
    "привет как дела",
    "меня зовут Паша",
    "привет как дела",
    "я люблю котов",
    "меня зовут Паша",
    "я люблю котов",
    "мой телефон восемь девятьсот двенадцать",
    "молодец, это важно запомнить",
    "сегодня обычная погода",
    "какой у меня телефон",
    "верно, отлично помнишь",
    "сегодня обычная погода",
    "напомни мой телефон",
    "молодец",
    "сегодня обычная погода",
]

HELP_TEXT = (
    "Я расту и учусь языку с нуля — сначала лепечу, потом отвечаю фразами.\n\n"
    "/demo — прогнать сценарий обучения и увидеть путь от лепета к речи\n"
    "/why — почему последнее сообщение записалось или нет, числами\n"
    "/skip 7 — промотать неделю и посмотреть, что пережило забывание\n"
    "/status — чему я уже научился\n"
    "/sleep — фаза консолидации памяти\n\n"
    "Хвали меня («молодец», «верно») — то, за что хвалят, я помню дольше. "
    "Это не метафора: разрыв между похвалённым и обычным через две недели "
    "составляет пятьдесят процентных пунктов."
)


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
        "Сначала я умею только лепетать — слова закрепляются примерно "
        "с третьего употребления, так что повторяй их, и я начну "
        "отвечать фразами.\n\n"
        "/demo — прогнать сценарий обучения и увидеть весь путь сразу\n"
        "/why — почему я запомнил или не запомнил последнее\n"
        "/skip 7 — промотать неделю и посмотреть, что пережило\n"
        "/status — чему я уже научился\n"
        "/help — всё то же самое подробнее"
    )


@router.message(Command("status"))
async def handle_status(message: Message) -> None:
    """
    Показывает прогресс обучения: стадия речи, сколько слов до следующей,
    что уже закрепилось. Без этой обратной связи учитель не видит вообще
    ничего и бросает — см. BrainSession.get_status_report.
    """
    manager = get_manager()
    user_id = message.from_user.id

    session = await asyncio.to_thread(manager.get_or_create, user_id)
    report = await asyncio.to_thread(session.get_status_report)

    await message.answer(report)


@router.message(Command("sleep"))
async def handle_sleep(message: Message) -> None:
    """Явный запуск фазы сна (BrainSession.process_message понимает '/sleep')."""
    manager = get_manager()
    user_id = message.from_user.id

    session = await asyncio.to_thread(manager.get_or_create, user_id)
    response = await asyncio.to_thread(session.process_message, "/sleep")

    await message.answer(response.text)


@router.message(Command("why"))
async def handle_why(message: Message) -> None:
    """
    Разбор последнего сообщения: удивился ли, записал ли, почему.

    Главная команда демонстрации. Всё остальное можно показать словами, а
    вот что решение о записи РЕАЛЬНОЕ, а порог не декоративный — видно
    только числами.
    """
    manager = get_manager()
    session = await asyncio.to_thread(manager.get_or_create, message.from_user.id)
    await message.answer(await asyncio.to_thread(session.explain_last))


@router.message(Command("skip"))
async def handle_skip(message: Message) -> None:
    """
    /skip [суток] — промотать время и показать, что пережило забывание.

    Без этого главную идею показать нельзя: разрыв между важным и рутиной
    раскрывается за недели, а у демонстрации есть минута.
    """
    parts = (message.text or "").split()
    try:
        days = float(parts[1]) if len(parts) > 1 else 7.0
    except ValueError:
        await message.answer("Формат: /skip 7 — промотать неделю.")
        return

    if not 0 < days <= 365:
        await message.answer("Разумный диапазон — от одних суток до года.")
        return

    manager = get_manager()
    session = await asyncio.to_thread(manager.get_or_create, message.from_user.id)
    await message.answer(await asyncio.to_thread(session.skip_forward, days))


@router.message(Command("demo"))
async def handle_demo(message: Message) -> None:
    """
    Прогнать короткий сценарий обучения и показать путь от лепета к фразам.

    Сценарий не случайный: сначала повторяющиеся реплики (чтобы набрался
    словарь), потом важный факт с похвалой, потом рутина. На этом наборе
    видно всё сразу — и освоение речи, и избирательность записи.
    """
    manager = get_manager()
    user_id = message.from_user.id
    session = await asyncio.to_thread(manager.get_or_create, user_id)

    await message.answer("Учу себя на сценарии, это займёт несколько секунд...")

    lines = []
    for step, text in enumerate(DEMO_SCRIPT, start=1):
        response = await asyncio.to_thread(session.process_message, text)
        lines.append(f"{step:2}. ты: {text}\n    я: {response.text[:70]}")

    await message.answer(
        "\n".join(lines)
        + "\n\nТеперь /why — почему последнее не записалось, "
        "и /skip — что от всего этого останется через неделю."
    )


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


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


async def _idle_scheduler_loop(bot: Bot) -> None:
    """
    Фоновый asyncio-таск — замена _idle_sleep_background_loop из main.py
    для мультипользовательского режима. Раз в BOT_SCHEDULER_TICK_SECONDS
    реальных секунд проходит по СНИМКУ активных сессий
    (SessionManager.all_sessions() — без удержания лока на весь обход) и
    для каждой:

        1. Если реальное бездействие >= SESSION_EVICTION_IDLE_SECONDS ->
           выгружает сессию из памяти (SessionManager.remove) и пропускает
           дальнейшую обработку для неё в этой итерации.
        2. Иначе продвигает brain_time сессии на BOT_SCHEDULER_TICK_SECONDS
           через BrainSession.run_idle_tick() (в отдельном треде — внутри
           синхронный SQLite и возможный блокирующий вызов LLM). Если тик
           вернул проактивное сообщение (kind="proactive") — отправляет
           его пользователю в Telegram. Событие kind="sleep" (авто-сон)
           только логируется — это внутреннее обслуживание памяти, а не
           реплика бота, отправлять его пользователю не нужно.

    Ошибка при обработке ОДНОЙ сессии логируется и не прерывает обход
    остальных — иначе баг у одного пользователя мог бы остановить фоновый
    тик для всех.
    """
    logger.info(
        "[SCHEDULER] Запущен фоновый idle-тик (interval=%.1fs, eviction=%.0fs)",
        config.BOT_SCHEDULER_TICK_SECONDS, config.SESSION_EVICTION_IDLE_SECONDS,
    )
    manager = get_manager()

    while True:
        await asyncio.sleep(config.BOT_SCHEDULER_TICK_SECONDS)

        sessions_snapshot = manager.all_sessions()
        for user_id, session in sessions_snapshot.items():
            try:
                idle_seconds = session.clock.seconds_since_last_activity()

                if idle_seconds >= config.SESSION_EVICTION_IDLE_SECONDS:
                    logger.info(
                        "[SCHEDULER] Eviction user_id=%s (бездействие %.0fs >= %.0fs)",
                        user_id, idle_seconds, config.SESSION_EVICTION_IDLE_SECONDS,
                    )
                    await asyncio.to_thread(manager.remove, user_id)
                    continue

                event = await asyncio.to_thread(
                    session.run_idle_tick, config.BOT_SCHEDULER_TICK_SECONDS
                )

                if event is None:
                    continue

                if event.kind == "proactive" and event.text:
                    await bot.send_message(user_id, event.text)
                elif event.kind == "sleep":
                    logger.info(
                        "[SCHEDULER] user_id=%s: автоматический Idle Sleep выполнен",
                        user_id,
                    )
            except Exception:  # noqa: BLE001
                logger.exception("[SCHEDULER] Ошибка idle-тика для user_id=%s", user_id)


async def _on_shutdown() -> None:
    """Graceful shutdown: останавливаем фоновый scheduler и закрываем все активные BrainSession."""
    global _scheduler_task

    if _scheduler_task is not None:
        logger.info("[BOT] Остановка: отменяю фоновый idle-scheduler...")
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
        _scheduler_task = None

    logger.info("[BOT] Остановка: закрываю все активные сессии...")
    manager = get_manager()
    await asyncio.to_thread(manager.close_all)

    logger.info("[BOT] Все сессии закрыты, завершение процесса.")


async def main() -> None:
    global _session_manager, _scheduler_task

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

    _scheduler_task = asyncio.create_task(_idle_scheduler_loop(bot))

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