"""
================================================================================
 DRIVES.PY — Скука и проактивность "Динамического Мозга" (Boredom Drive)
================================================================================
Класс BoredomDrive реализует конечный автомат (State Machine) физиологического
состояния системы относительно активности пользователя:

    AWAKE            -> бот активно общается (пользователь недавно писал)
    SLEEPING         -> бот "спит" (Idle Sleep уже выполнен), накапливается
                        показатель boredom по формуле экспоненциального
                        насыщения
    WAITING_FOR_USER -> бот отправил проактивное сообщение и ЗАМОРОЗИЛ
                        накопление boredom, ожидая реакции пользователя

Формула накопления скуки (пока SYSTEM_STATE == SLEEPING):
    boredom(dt) = 1.0 - exp(-dt / BOREDOM_TAU)
где dt — виртуальное время (brain_time), прошедшее с момента входа в SLEEPING
(НЕ с последнего сообщения пользователя — таймер скуки стартует именно с
момента засыпания, см. IDLE_SLEEP_THRESHOLD_SECONDS в main.py).

Вся работа с состоянием происходит через явные методы (enter_sleeping,
trigger_proactive, on_user_message) — сам класс НЕ содержит threading.Lock,
потокобезопасность обеспечивается вызывающим кодом (main.py), который
оборачивает обращения к BoredomDrive в общий brain_time_lock.
================================================================================
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import config
from storage.utils.logger import get_logger

logger = get_logger(__name__)


class SystemState(str, Enum):
    """Состояния конечного автомата активности системы."""
    AWAKE = "AWAKE"
    SLEEPING = "SLEEPING"
    WAITING_FOR_USER = "WAITING_FOR_USER"


@dataclass
class BoredomSnapshot:
    """Снимок текущего состояния Boredom Drive для логов/дебага."""
    state: SystemState
    boredom: float
    seconds_since_sleep_start: float


class BoredomDrive:
    """
    Конечный автомат скуки/проактивности.

    Использование (вызывающий код обязан сам гарантировать потокобезопасность
    через внешний lock — см. main.py):

        drive = BoredomDrive()
        drive.enter_sleeping(brain_time)
        snapshot = drive.update(brain_time)
        if snapshot.boredom >= config.BOREDOM_THRESHOLD:
            drive.trigger_proactive()
            ...
        drive.on_user_message(brain_time)
    """

    def __init__(self):
        self.state: SystemState = SystemState.AWAKE
        self._sleep_start_time: Optional[float] = None
        self._frozen_boredom: float = 0.0

    # ----------------------------------------------------------------------
    # Переходы состояний
    # ----------------------------------------------------------------------

    def enter_sleeping(self, brain_time: float) -> None:
        """
        Переводит систему в SLEEPING и запускает таймер накопления boredom
        (dt считается от этого момента). Вызывается ОДНОКРАТНО при входе
        в фазу сна (main.py гарантирует, что это не повторится на каждом
        тике, проверяя self.state перед вызовом).
        """
        if self.state == SystemState.SLEEPING:
            return

        self.state = SystemState.SLEEPING
        self._sleep_start_time = brain_time
        self._frozen_boredom = 0.0

        logger.info("[BOREDOM] Состояние -> SLEEPING (sleep_start=%.1f)", brain_time)

    def update(self, brain_time: float) -> BoredomSnapshot:
        """
        Пересчитывает текущий уровень boredom, если состояние SLEEPING.
        В AWAKE/WAITING_FOR_USER возвращает "замороженное" значение без
        пересчёта (в AWAKE это всегда 0.0, в WAITING_FOR_USER — значение
        на момент срабатывания триггера, config.BOREDOM_THRESHOLD).
        """
        if self.state != SystemState.SLEEPING or self._sleep_start_time is None:
            seconds_since = 0.0
            boredom = self._frozen_boredom
        else:
            seconds_since = max(0.0, brain_time - self._sleep_start_time)
            boredom = 1.0 - math.exp(-seconds_since / config.BOREDOM_TAU)
            self._frozen_boredom = boredom

        return BoredomSnapshot(
            state=self.state,
            boredom=boredom,
            seconds_since_sleep_start=seconds_since,
        )

    def trigger_proactive(self) -> None:
        """
        Переводит систему в WAITING_FOR_USER — накопление boredom
        ЗАМОРАЖИВАЕТСЯ на текущем значении (Anti-Spam Guard). Вызывается
        сразу после того, как проактивное сообщение было успешно
        отправлено пользователю.
        """
        self.state = SystemState.WAITING_FOR_USER
        logger.info(
            "[BOREDOM] Состояние -> WAITING_FOR_USER (boredom заморожен на %.3f)",
            self._frozen_boredom,
        )

    def on_user_message(self, brain_time: float) -> None:
        """
        Вызывается при ЛЮБОМ входящем сообщении пользователя (независимо
        от того, в каком состоянии была система). Полностью сбрасывает
        boredom и переводит систему в AWAKE.
        """
        previous_state = self.state
        self.state = SystemState.AWAKE
        self._sleep_start_time = None
        self._frozen_boredom = 0.0

        logger.info(
            "[BOREDOM] Пользователь написал -> состояние %s -> AWAKE, boredom сброшен в 0.0",
            previous_state.value,
        )

    # ----------------------------------------------------------------------
    # Вспомогательные проверки
    # ----------------------------------------------------------------------

    def is_awake(self) -> bool:
        return self.state == SystemState.AWAKE

    def is_sleeping(self) -> bool:
        return self.state == SystemState.SLEEPING

    def is_waiting_for_user(self) -> bool:
        return self.state == SystemState.WAITING_FOR_USER