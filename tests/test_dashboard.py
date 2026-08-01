"""
================================================================================
 TEST_DASHBOARD.PY — Витрина не должна врать и не должна падать
================================================================================
Дашборд читает базу СТРОГО НА ЧТЕНИЕ и отдельным процессом, поэтому до
оперативного состояния мозга ему не дотянуться. Разбор последнего хода
попадает к нему через мета-узел — и это единственный канал. Здесь
проверяется, что канал работает, что страница собирается на любой базе
(включая пустую и только что созданную) и что служебные узлы не
выдаются за воспоминания.

Отдельно проверяется главное свойство: дашборд показывает НЕ ЗАПИСАННЫЕ
сообщения тоже. Витрина, которая рисует только попавшее в память, не
показывает самого интересного — того, как память отказывается писать.
================================================================================
"""

import json

import pytest

from core.brain_session import BrainSession
from tools.render_memory import (
    INSTRUMENT_TYPES,
    load_snapshot,
    render_decision,
    render_html,
    render_mood,
)


@pytest.fixture
def brain(tmp_path, monkeypatch):
    import core.cortex
    import core.persona_memory
    import core.sleep_cycle

    def fake_llm(messages, system_prompt=None, max_tokens=None):
        return "понятно"

    for module in (core.cortex, core.persona_memory, core.sleep_cycle):
        monkeypatch.setattr(module, "generate_llm_response", fake_llm)

    session = BrainSession(db_path=str(tmp_path / "brain.db"))
    yield session
    session.close()


def test_decision_reaches_the_dashboard(brain):
    """Разбор хода доезжает до витрины через мета-узел."""
    brain.process_message("меня зовут Паша")

    snap = load_snapshot(brain.db_path)
    assert snap.decision is not None
    assert snap.decision["text"] == "меня зовут Паша"
    assert 0.0 <= snap.decision["threshold"] <= 1.0
    assert "mood" in snap.decision


def test_dashboard_shows_rejected_messages(brain):
    """
    САМОЕ ВАЖНОЕ. Рутина в память не попадает, но на витрине обязана
    появиться: иначе дашборд показывает только успехи и умалчивает о том,
    как работает порог.
    """
    for _ in range(6):
        brain.process_message("сегодня совершенно обычная погода")

    snap = load_snapshot(brain.db_path)
    assert snap.decision["written"] is False

    html = render_decision(snap)
    assert "не записано" in html
    assert "не хватило" in html
    assert "погода" in html


def test_dashboard_survives_a_fresh_brain(tmp_path, brain):
    """
    Только что созданный мозг ещё ничего не решал. Страница обязана
    собраться и сказать об этом словами, а не упасть на None.
    """
    snap = load_snapshot(brain.db_path)
    assert snap.decision is None

    assert "напиши" in render_decision(snap)
    assert render_mood(snap) == ""
    render_html(snap, include_lexical=False)  # не должно бросать


def test_instrument_nodes_are_not_shown_as_memories(brain):
    """
    Эпоха и разбор хода — приборы. В графе они висели безымянными
    точками, а brain_epoch показывал число вместо текста.
    """
    brain.process_message("привет")
    snap = load_snapshot(brain.db_path)

    assert any(n.node_type in INSTRUMENT_TYPES for n in snap.nodes), "приборы в базе есть"

    html = render_html(snap, include_lexical=False)
    for node_type in INSTRUMENT_TYPES:
        assert node_type not in html, f"{node_type} просочился на страницу"


def test_dashboard_reads_without_writing(brain):
    """
    Инспектор открывает базу только на чтение. Проверяется тем, что
    файл не меняется: витрина не имеет права трогать мозг, который
    показывает, — её можно навести на живую рабочую базу.
    """
    import pathlib

    brain.process_message("привет")
    path = pathlib.Path(brain.db_path)
    before = (path.stat().st_mtime_ns, path.stat().st_size)

    snap = load_snapshot(brain.db_path)
    render_html(snap, include_lexical=True)

    assert (path.stat().st_mtime_ns, path.stat().st_size) == before


def test_decision_payload_is_valid_json(brain):
    """Мета-узел должен содержать разбираемый JSON, а не отладочный repr."""
    brain.process_message("проверка")
    row = brain.memory.db.get_meta_node("last_decision")
    payload = json.loads(row["context"])
    assert set(payload) >= {"text", "surprise", "emotion", "density", "threshold", "written"}
