"""
================================================================================
 CORE/SESSION_MANAGER.PY — Реестр изолированных BrainSession по пользователям
================================================================================
Этап 2 миграции CLI -> Telegram-бот.

Один пользователь (telegram_user_id) = один BrainSession с собственной
SQLite БД по пути config.BRAIN_DB_DIR/{user_id}.db — полная изоляция
данных между пользователями без изменения схемы БД (Database уже
параметризован по db_path, см. config.py раздел 19).

SessionManager остаётся ЧИСТО СИНХРОННЫМ классом на этом этапе — никакого
asyncio здесь. Обёртка в asyncio.to_thread (если понадобится) — забота
Этапа 3 (aiogram-хендлеры), не этого модуля.

НЕ реализовано на этом этапе (сознательно, по плану):
    - выгрузка неактивных сессий по SESSION_EVICTION_IDLE_SECONDS (Этап 6);
    - фоновый asyncio-тик по всем сессиям (Этап 5).
================================================================================
"""

import threading
from pathlib import Path
from typing import Dict, Optional

import config
from core.brain_session import BrainSession
from storage.utils.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    """
    Реестр активных BrainSession, ключ — telegram_user_id.

    Потокобезопасен на уровне доступа к словарю сессий (threading.Lock),
    т.к. в будущем (Этап 3/5) к get_or_create могут обращаться из разных
    асинхронных обработчиков/тасков одновременно.
    """

    def __init__(self, db_dir: Optional[str] = None):
        self.db_dir = db_dir or config.BRAIN_DB_DIR
        Path(self.db_dir).mkdir(parents=True, exist_ok=True)

        self._sessions: Dict[int, BrainSession] = {}
        self._lock = threading.Lock()

        logger.info("[SESSION MANAGER] Инициализирован (db_dir=%s)", self.db_dir)

    # ----------------------------------------------------------------------
    # Публичный API
    # ----------------------------------------------------------------------

    def get_or_create(self, user_id: int) -> BrainSession:
        """
        Возвращает существующий BrainSession для user_id, либо создаёт
        новый (с ленивой инициализацией SQLite БД по пути
        {db_dir}/{user_id}.db) при первом обращении.
        """
        with self._lock:
            session = self._sessions.get(user_id)
            if session is not None:
                return session

            db_path = self._build_db_path(user_id)
            logger.info(
                "[SESSION MANAGER] Создаю новую сессию для user_id=%s (db_path=%s)",
                user_id, db_path,
            )
            session = BrainSession(db_path=db_path)
            self._sessions[user_id] = session
            return session

    def get(self, user_id: int) -> Optional[BrainSession]:
        """Возвращает сессию, если она уже создана, иначе None (без создания)."""
        with self._lock:
            return self._sessions.get(user_id)

    def all_sessions(self) -> Dict[int, BrainSession]:
        """
        Снимок текущего реестра сессий (user_id -> BrainSession).
        Используется будущим фоновым тиком (Этап 5) для итерации по всем
        активным "мозгам" без удержания лока на всё время обхода.
        """
        with self._lock:
            return dict(self._sessions)

    def remove(self, user_id: int) -> None:
        """
        Закрывает и удаляет сессию из реестра (например, по eviction
        таймауту в Этапе 6, или явно по команде администратора).
        """
        with self._lock:
            session = self._sessions.pop(user_id, None)

        if session is not None:
            session.close()
            logger.info("[SESSION MANAGER] Сессия user_id=%s закрыта и удалена из реестра", user_id)

    def close_all(self) -> None:
        """Закрывает ВСЕ активные сессии (graceful shutdown бота)."""
        with self._lock:
            sessions_snapshot = list(self._sessions.items())
            self._sessions.clear()

        for user_id, session in sessions_snapshot:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                logger.exception("[SESSION MANAGER] Ошибка при закрытии сессии user_id=%s", user_id)

        logger.info("[SESSION MANAGER] Все сессии (%d) закрыты", len(sessions_snapshot))

    def active_count(self) -> int:
        """Количество активных (загруженных в память) сессий."""
        with self._lock:
            return len(self._sessions)

    # ----------------------------------------------------------------------
    # Внутренние хелперы
    # ----------------------------------------------------------------------

    def _build_db_path(self, user_id: int) -> str:
        return str(Path(self.db_dir) / f"{user_id}.db")