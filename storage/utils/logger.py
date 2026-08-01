"""
================================================================================
 LOGGER.PY — Настройка логирования приложения
================================================================================
Единая точка логирования внутренних событий системы:
[SPIKE DETECTED], [DECAY APPLIED], [MEMORY SAVED], [STRESS TRIGGERED] и т.д.

ХЕНДЛЕРЫ ВЕШАЮТСЯ НА КОРЕНЬ, а не на каждый модуль. Раньше get_logger
навешивал пару хендлеров на КАЖДЫЙ именованный логгер и ставил
propagate=False, поэтому десяток модулей держал десяток открытых
дескрипторов одного brain.log. Теперь модули получают обычный
logging.getLogger(__name__) и просто пропагируют вверх — это позволяет
memory/ не зависеть от этого файла вовсе и уехать в отдельный пакет.

ФАЙЛ РОТИРУЕТСЯ. brain.log дорос до 18 МБ и продолжал расти: logrotate
для него никто не настраивал, а сам он не умел. Теперь 5 МБ на файл и
три копии в архиве — сутки подробных логов помещаются, диск не течёт.

БИБЛИОТЕКИ ПРИГЛУШЕНЫ. Пока хендлеры висели на своих логгерах, чужие
сообщения (httpx, telegram) в файл не попадали. С корневым хендлером
они попали бы туда все и утопили бы логи мозга в служебном шуме.
================================================================================
"""

import logging
import logging.handlers
import sys
from pathlib import Path

import config

# Ротация: 5 МБ на файл, 3 архивных копии.
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

# Чужие логгеры, которым нечего делать в логе мозга на уровне INFO.
_NOISY = ("httpx", "httpcore", "telegram", "urllib3", "asyncio", "aiohttp")

_configured = False


def configure_logging() -> None:
    """
    Настраивает КОРНЕВОЙ логгер: консоль + ротируемый файл. Вызывается
    один раз при старте приложения; повторные вызовы ничего не делают.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    log_path = Path(config.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for name in _NOISY:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))

    _configured = True


def get_logger(name: str = "brain") -> logging.Logger:
    """
    Логгер модуля. Настройка приложения выполняется при первом обращении,
    чтобы ни один вызывающий не остался без хендлеров.

    Ядро памяти этой функцией НЕ пользуется: библиотека не вправе решать,
    куда пишет приложение, — там обычный logging.getLogger(__name__).
    """
    configure_logging()
    return logging.getLogger(name)
