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


def test_praised_fact_outlives_routine(monkeypatch):
    """
    ОБЕЩАНИЕ ПРОДУКТА, сформулированное так, как оно ИЗМЕРЕНО.

    Первая версия этого теста утверждала: "после недели остаётся телефон".
    На одном прогоне так и было, и я подал это как поведение. Замер на
    пятнадцати сидах показал 9 из 15 — то есть утверждение было верно
    для сида, который мне выпал, а не вообще.

    Хуже того, тест не задавал сид и потому зависел от того, сколько
    случайных чисел израсходовали ПРЕДЫДУЩИЕ тесты: он проходил в одном
    порядке запуска и падал в другом. Такой тест не проверяет систему, он
    проверяет удачу.

    Что верно на самом деле (15 сидов, горизонт 7 суток):

        телефон первым   9
        рутина первой    1
        не выжило ничего 5

    То есть на пятнадцати репликах эффект НЕ гарантирован — забывание
    часто уносит всё. Но когда хоть что-то переживает неделю, это почти
    всегда подкреплённое, а не последнее сказанное. Именно это и
    проверяется: соотношение, а не отдельный исход.

    Сильный результат (+53 п.п.) живёт в tools/compare_retention.py, где
    разговор в разы длиннее и у стабильности есть время накопиться.
    Пятнадцать реплик — это демонстрация, а не замер.
    """
    import random

    import core.cortex
    import core.persona_memory
    import core.sleep_cycle

    def fake_llm(messages, system_prompt=None, max_tokens=None):
        return "понятно"

    for module in (core.cortex, core.persona_memory, core.sleep_cycle):
        monkeypatch.setattr(module, "generate_llm_response", fake_llm)

    assert DEMO_SCRIPT[-1].startswith("сегодня"), "сценарий должен кончаться рутиной"

    praised = routine = nothing = 0
    for seed in range(8):
        random.seed(seed)
        session = BrainSession(db_path=":memory:")
        try:
            _run_demo(session)
            session.skip_forward(7)
            survivors = session.memory.db.get_top_nodes_by_type("episodic", 3)
            if not survivors:
                nothing += 1
            elif "телефон" in (survivors[0]["context"] or ""):
                praised += 1
            else:
                routine += 1
        finally:
            session.close()

    assert praised > routine, (
        f"подкреплённое должно переживать чаще рутины, вышло {praised} против {routine}"
    )
    assert praised + routine > 0, "хоть в каких-то прогонах что-то обязано выживать"


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
