"""
================================================================================
 LOGGER.PY — Лёгкий централизованный логгер для "Динамического Мозга"
================================================================================
Единая точка логирования внутренних событий системы:
[SPIKE DETECTED], [DECAY APPLIED], [MEMORY SAVED], [STRESS TRIGGERED] и т.д.
================================================================================
"""

import logging
import sys
from pathlib import Path

import config


def get_logger(name: str = "brain") -> logging.Logger:
    """
    Создаёт (или возвращает существующий) логгер с настройками из config.py.

    Пишет одновременно в консоль и в файл storage/logs/brain.log.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Логгер уже настроен — избегаем дублирования хендлеров
        return logger

    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Консоль ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- Файл ---
    log_path = Path(config.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger