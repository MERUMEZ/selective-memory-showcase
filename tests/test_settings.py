"""
================================================================================
 TEST_SETTINGS.PY — Настройки ядра не должны разъезжаться с config.py
================================================================================
memory/settings.py был СГЕНЕРИРОВАН из config.py, и его умолчания обязаны
совпадать с откалиброванными значениями приложения. Совпадение это хрупкое:
достаточно поправить константу в одном файле и забыть про другой, чтобы
тесты продолжали проходить, а бот — вести себя иначе, чем стенд.

Ошибка была бы тихой вдвойне, потому что расхождение проявляется не
исключением, а сдвигом порогов: чуть реже спайки, чуть быстрее забывание.
Такое ловится только замером через неделю.

Поэтому здесь сверяется КАЖДОЕ поле, а не выборка.
================================================================================
"""

from dataclasses import fields

import pytest

import config
from selectivemem.settings import MemorySettings


def _config_name(field_name: str) -> str:
    return MemorySettings._ALIASES.get(field_name, field_name.upper())


ALL_FIELDS = [f.name for f in fields(MemorySettings)]

# db_path — единственное поле, чьё умолчание ОБЯЗАНО расходиться с
# config.py: там абсолютный путь этой установки, а библиотека не вправе
# писать в чужой каталог. Совпадение значений здесь было бы ошибкой.
NOT_CALIBRATED = {"db_path"}


@pytest.mark.parametrize("field_name", ALL_FIELDS)
def test_default_matches_config(field_name):
    """Умолчание поля совпадает с одноимённой константой config.py."""
    if field_name in NOT_CALIBRATED:
        pytest.skip(f"{field_name} — не параметр поведения, см. NOT_CALIBRATED")

    name = _config_name(field_name)
    if not hasattr(config, name):
        pytest.skip(f"{name} нет в config.py — поле чисто библиотечное")

    default = getattr(MemorySettings(), field_name)
    assert default == getattr(config, name), (
        f"{field_name} разъехался с config.{name}: "
        f"{default!r} против {getattr(config, name)!r}"
    )


def test_from_module_reads_every_field():
    """
    from_module не должен молча пропускать поля. Если константа в config
    есть, она обязана доехать до настроек — иначе приложение думает, что
    настроило ядро, а ядро живёт по умолчаниям.
    """
    settings = MemorySettings.from_module(config)
    missing = [
        f for f in ALL_FIELDS
        if hasattr(config, _config_name(f))
        and getattr(settings, f) != getattr(config, _config_name(f))
    ]
    assert not missing, f"не доехали до настроек: {missing}"


def test_settings_are_independent_of_config():
    """
    Ядро обязано принимать значения, которых в config.py нет вовсе, —
    иначе никакой развязки не произошло и пакет по-прежнему одноразовый.
    """
    custom = MemorySettings(decay_rate=0.99, age_t0=1.0)
    assert custom.decay_rate == 0.99
    assert custom.age_t0 == 1.0
    # А соседние поля остались библиотечными умолчаниями
    assert custom.stability_max == MemorySettings().stability_max


def test_graph_uses_injected_settings():
    """Граф действительно живёт по переданным настройкам, а не по config."""
    from selectivemem.database import Database
    from selectivemem.graph_memory import MemoryGraph

    graph = MemoryGraph(
        db=Database(db_path=":memory:"),
        settings=MemorySettings(decay_rate=0.42),
    )
    assert graph.settings.decay_rate == 0.42
    assert graph.settings.decay_rate != config.DECAY_RATE


def test_memory_package_is_self_contained():
    """
    Главная проверка развязки: ни один модуль ядра не тянет ничего из
    приложения — ни config, ни core, ни services, ни storage. Читается
    ИСХОДНИК, а не импорты: импорт мог бы прийти транзитивно и создать
    ложное спокойствие.

    Шаблон намеренно НЕ требует точки после имени пакета. Прошлая версия
    требовала (`from services\\.`), поэтому пропускала форму
    `from services import embeddings` — и ядро зависело от services, пока
    проверка показывала "чисто". Ошибка была не в коде, а в самой
    проверке, что хуже: она создавала уверенность.

    Каталог берётся ИЗ САМОГО ПАКЕТА (selectivemem.__path__), а не строкой.
    Строку я уже дважды забывал поправить при переименовании — memory/ ->
    engram/ -> selectivemem/, — и оба раза проверка молча смотрела в пустоту и
    проходила. Тест, который нельзя сломать переименованием, надёжнее
    теста, который надо не забыть поправить.
    """
    import pathlib
    import re

    import selectivemem

    root = pathlib.Path(selectivemem.__path__[0])
    modules = sorted(root.glob("*.py"))
    assert len(modules) >= 5, f"каталог ядра не найден или пуст: {root}"

    pattern = re.compile(r"^\s*(from|import)\s+(config|core|services|storage)\b", re.M)

    offenders = {}
    for path in modules:
        found = pattern.findall(path.read_text())
        if found:
            offenders[path.name] = sorted({m[1] for m in found})
    assert not offenders, f"ядро зависит от приложения: {offenders}"
