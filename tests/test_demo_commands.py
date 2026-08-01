"""
================================================================================
 TEST_DEMO_COMMANDS.PY — Демонстрация обязана показывать то, что обещано
================================================================================
Здесь проверяется не механизм (для него есть test_decay, test_surprise и
стенды), а ВИТРИНА: /demo, /why и /skip — первое, что видит человек, и
единственное, по чему он судит, работает ли идея.

Главный тест — test_praised_fact_outlives_routine. Он кодирует само
обещание продукта: через неделю молчания выживает то, к чему
возвращались и что отмечали как важное, а не то, что было сказано
последним. Если этот тест падает, продавать нечего, какими бы зелёными
ни были остальные.

Сценарий взят из bot.DEMO_SCRIPT намеренно: тест обязан ломаться, если
кто-то "улучшит" сценарий так, что он перестанет что-либо показывать.
Более короткий вариант уже проверялся — за пятнадцать реплик без
возвращений стабильность не набирается, и через неделю пусто везде.
================================================================================
"""

import pytest

from bot import DEMO_SCRIPT
from core.brain_session import BrainSession


@pytest.fixture
def brain(monkeypatch):
    """Сессия на in-memory SQLite с заглушенной LLM (без сети)."""
    import core.cortex
    import core.persona_memory
    import core.sleep_cycle

    def fake_llm(messages, system_prompt=None, max_tokens=None):
        return "понятно"

    for module in (core.cortex, core.persona_memory, core.sleep_cycle):
        monkeypatch.setattr(module, "generate_llm_response", fake_llm)

    session = BrainSession(db_path=":memory:")
    yield session
    session.close()


def _run_demo(session):
    for text in DEMO_SCRIPT:
        session.process_message(text)


def test_praised_fact_outlives_routine(brain):
    """
    ОБЕЩАНИЕ ПРОДУКТА. После недели молчания остаётся факт, который
    хвалили и к которому возвращались, — а не последняя реплика.
    """
    _run_demo(brain)
    assert DEMO_SCRIPT[-1].startswith("сегодня"), "сценарий должен кончаться рутиной"

    before = brain.memory.db.count_nodes_by_type("episodic")
    brain.skip_forward(7)
    survivors = brain.memory.db.get_top_nodes_by_type("episodic", 5)

    assert before > len(survivors), "забывание должно было что-то отсеять"
    assert survivors, "но не всё: важное обязано пережить"
    assert "телефон" in survivors[0]["context"], (
        f"выжило не то, что подкрепляли: {survivors[0]['context']!r}"
    )
    assert not any("погода" in (r["context"] or "") for r in survivors), (
        "рутина не должна переживать неделю"
    )


def test_demo_reaches_real_speech(brain):
    """
    Демонстрация обязана дойти от лепета до фраз. Если организм к концу
    сценария всё ещё лепечет, показывать нечего.
    """
    replies = [brain.process_message(t).text for t in DEMO_SCRIPT]

    assert any("-" in r for r in replies[:3]), "в начале должен быть лепет"
    assert "-" not in replies[-1], f"в конце лепета быть не должно: {replies[-1]!r}"


def test_why_explains_a_rejection(brain):
    """
    /why обязан назвать ЧИСЛО, которого не хватило. "Не запомнилось" без
    числа — это отговорка, а не объяснение.
    """
    _run_demo(brain)
    explanation = brain.explain_last()

    assert "порог" in explanation
    assert "удивление" in explanation
    assert "НЕ записано" in explanation or "ЗАПИСАНО" in explanation


def test_why_survives_a_cold_start(brain):
    """До первого сообщения /why не должен падать — его нажмут первым."""
    assert brain.explain_last()


def test_skip_reports_the_acceleration(brain):
    """
    В отчёте обязана быть оговорка про ускорение времени: "прошло 7
    суток" без неё вводит в заблуждение, потому что организм прожил семь
    недель.
    """
    _run_demo(brain)
    report = brain.skip_forward(7)
    assert "быстрее" in report
