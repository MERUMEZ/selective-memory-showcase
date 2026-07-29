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
from datetime import datetime, timezone

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LabeledPrice,
    Message,
    PreCheckoutQuery,
)

import config
from billing.subscription_manager import SubscriptionManager
from core.session_manager import SessionManager
from storage.utils.logger import get_logger

logger = get_logger(__name__)

router = Router()

# Payload-строки для инвойсов Telegram Stars — используются и при
# отправке инвойса, и при валидации в pre_checkout_query/successful_payment.
PREMIUM_PAYLOAD_30D = "premium_30d"
PREMIUM_PAYLOAD_90D = "premium_90d"

# Единый на весь процесс реестр "мозгов" — по одному BrainSession на
# каждого telegram_user_id. Инициализируется в main() и передаётся сюда
# через router-level middleware/замыкание (см. get_manager ниже).
_session_manager: SessionManager = None  # type: ignore[assignment]

# Единый на весь процесс менеджер подписок/квот (billing.database —
# ОТДЕЛЬНАЯ БД от per-user brain.db, см. billing/subscription_manager.py).
_subscription_manager: SubscriptionManager = None  # type: ignore[assignment]

# Фоновый asyncio-таск Idle Sleep/Boredom-тика + eviction (Этап 5/6),
# запускается в main(), отменяется в _on_shutdown().
_scheduler_task: asyncio.Task = None  # type: ignore[assignment]


def get_manager() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("SessionManager не инициализирован (main() не был вызван)")
    return _session_manager


def get_subscription_manager() -> SubscriptionManager:
    if _subscription_manager is None:
        raise RuntimeError("SubscriptionManager не инициализирован (main() не был вызван)")
    return _subscription_manager


async def _enforce_quota(message: Message, user_id: int) -> bool:
    """
    Проверяет дневной лимит бесплатных сообщений ПЕРЕД обработкой
    сообщения BrainSession'ом.

    Возвращает True, если сообщение можно обрабатывать дальше (либо
    Premium активен, либо лимит на сегодня не исчерпан — счётчик уже
    атомарно увеличен внутри check_and_increment_quota).

    Если лимит исчерпан — отправляет пользователю сообщение с призывом
    к /subscribe и возвращает False (вызывающий хендлер должен просто
    сделать return, ничего больше не отправляя).
    """
    sub_manager = get_subscription_manager()

    is_premium = await asyncio.to_thread(sub_manager.is_premium, user_id)
    if is_premium:
        return True

    allowed = await asyncio.to_thread(sub_manager.check_and_increment_quota, user_id)
    if allowed:
        return True

    await message.answer(
        f"Ты исчерпал дневной лимит бесплатных сообщений "
        f"({config.FREE_TIER_DAILY_MESSAGE_LIMIT}/сутки).\n\n"
        "Оформи Premium-подписку командой /subscribe, чтобы снять лимит — "
        "либо возвращайся завтра, лимит обновится автоматически."
    )
    return False


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
        f"Бесплатно доступно {config.FREE_TIER_DAILY_MESSAGE_LIMIT} сообщений в сутки.\n"
        "Команда /subscribe — оформить Premium без лимита.\n"
        "Команда /sleep — запустить фазу консолидации памяти (сон)."
    )


@router.message(Command("subscribe"))
async def handle_subscribe(message: Message) -> None:
    """Показывает текущий статус подписки и предлагает выбрать тариф."""
    sub_manager = get_subscription_manager()
    user_id = message.from_user.id

    is_premium = await asyncio.to_thread(sub_manager.is_premium, user_id)
    if is_premium:
        expiry_ts = await asyncio.to_thread(sub_manager.get_premium_expiry, user_id)
        expiry_str = datetime.fromtimestamp(expiry_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        await message.answer(f"У тебя уже активна Premium-подписка до {expiry_str}.")
        return

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"{config.PREMIUM_DAYS_30D} дней — {config.PREMIUM_STARS_PRICE_30D} ⭐",
                    callback_data="buy_premium_30",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"{config.PREMIUM_DAYS_90D} дней — {config.PREMIUM_STARS_PRICE_90D} ⭐",
                    callback_data="buy_premium_90",
                )
            ],
        ]
    )
    await message.answer(
        "Premium снимает дневной лимит сообщений.\nВыбери тариф (оплата через Telegram Stars):",
        reply_markup=keyboard,
    )


@router.callback_query(F.data.in_({"buy_premium_30", "buy_premium_90"}))
async def handle_buy_premium_callback(callback: CallbackQuery) -> None:
    """Отправляет инвойс Telegram Stars на выбранный тариф."""
    if callback.data == "buy_premium_30":
        days, price, payload = config.PREMIUM_DAYS_30D, config.PREMIUM_STARS_PRICE_30D, PREMIUM_PAYLOAD_30D
    else:
        days, price, payload = config.PREMIUM_DAYS_90D, config.PREMIUM_STARS_PRICE_90D, PREMIUM_PAYLOAD_90D

    await callback.bot.send_invoice(
        chat_id=callback.from_user.id,
        title=f"Premium на {days} дней",
        description="Безлимитные сообщения без дневного ограничения.",
        payload=payload,
        currency="XTR",  # Telegram Stars — provider_token не требуется
        prices=[LabeledPrice(label=f"Premium {days}д", amount=price)],
    )
    await callback.answer()


@router.message(Command("sleep"))
async def handle_sleep(message: Message) -> None:
    """Явный запуск фазы сна (BrainSession.process_message понимает '/sleep')."""
    manager = get_manager()
    user_id = message.from_user.id

    if config.BILLING_GATE_SLEEP_COMMAND:
        if not await _enforce_quota(message, user_id):
            return

    session = await asyncio.to_thread(manager.get_or_create, user_id)
    response = await asyncio.to_thread(session.process_message, "/sleep")

    await message.answer(response.text)


@router.pre_checkout_query()
async def handle_pre_checkout(pre_checkout_query: PreCheckoutQuery) -> None:
    """
    Telegram требует ответ на pre_checkout_query в течение 10 секунд —
    здесь только быстрая валидация payload'а, БЕЗ обращений к БД/LLM.
    """
    valid_payloads = {PREMIUM_PAYLOAD_30D, PREMIUM_PAYLOAD_90D}

    if pre_checkout_query.invoice_payload in valid_payloads:
        await pre_checkout_query.answer(ok=True)
    else:
        logger.error(
            "[BILLING] Неизвестный payload в pre_checkout_query: %r (user_id=%s)",
            pre_checkout_query.invoice_payload, pre_checkout_query.from_user.id,
        )
        await pre_checkout_query.answer(ok=False, error_message="Неизвестный тариф, платёж отклонён.")


@router.message(F.successful_payment)
async def handle_successful_payment(message: Message) -> None:
    """
    Деньги СПИСАНЫ Telegram-ом до вызова этого хендлера — здесь только
    начисление Premium. Если этот хендлер упадёт ПОСЛЕ списания, но ДО
    grant_premium, пользователь заплатит и не получит Premium, поэтому
    любая ошибка здесь логируется как критическая (не noqa: BLE001 молча).
    """
    payment = message.successful_payment
    user_id = message.from_user.id
    payload = payment.invoice_payload

    if payload == PREMIUM_PAYLOAD_30D:
        days = config.PREMIUM_DAYS_30D
    elif payload == PREMIUM_PAYLOAD_90D:
        days = config.PREMIUM_DAYS_90D
    else:
        logger.error(
            "[BILLING] КРИТИЧНО: оплата прошла, но payload не распознан: %r (user_id=%s, charge_id=%s)",
            payload, user_id, payment.telegram_payment_charge_id,
        )
        await message.answer(
            "Платёж получен, но тариф не распознан автоматически. "
            "Напиши в поддержку с этим сообщением — активируем вручную."
        )
        return

    try:
        sub_manager = get_subscription_manager()
        new_until = await asyncio.to_thread(
            sub_manager.grant_premium,
            user_id,
            days,
            payment.total_amount,
            payload,
            payment.telegram_payment_charge_id,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "[BILLING] КРИТИЧНО: оплата прошла (charge_id=%s, user_id=%s), но grant_premium упал",
            payment.telegram_payment_charge_id, user_id,
        )
        await message.answer(
            "Платёж получен, но при активации произошла ошибка. "
            "Напиши в поддержку с этим сообщением — активируем вручную."
        )
        return

    expiry_str = datetime.fromtimestamp(new_until, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    await message.answer(f"Спасибо! Premium активирован до {expiry_str}. Приятного общения 🎉")


@router.message()
async def handle_text(message: Message) -> None:
    """Основной хендлер: любое обычное текстовое сообщение -> BrainSession."""
    if not message.text:
        await message.answer("Я пока умею понимать только текст.")
        return

    manager = get_manager()
    user_id = message.from_user.id

    if not await _enforce_quota(message, user_id):
        return

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
    """Graceful shutdown: останавливаем фоновый scheduler, закрываем все активные BrainSession и billing DB."""
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

    logger.info("[BOT] Остановка: закрываю billing DB...")
    sub_manager = get_subscription_manager()
    await asyncio.to_thread(sub_manager.close)

    logger.info("[BOT] Все сессии закрыты, завершение процесса.")


async def main() -> None:
    global _session_manager, _subscription_manager, _scheduler_task

    if not config.TELEGRAM_BOT_TOKEN:
        logger.error(
            "[BOT] TELEGRAM_BOT_TOKEN не задан в .env — запуск невозможен."
        )
        sys.exit(1)

    _session_manager = SessionManager(db_dir=config.BRAIN_DB_DIR)
    _subscription_manager = SubscriptionManager(db_path=config.BILLING_DB_PATH)

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