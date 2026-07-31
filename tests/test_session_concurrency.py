"""
Тесты потокобезопасности BrainSession.

Контекст: один "мозг" обслуживается из РАЗНЫХ ПОТОКОВ. bot.py уводит в
asyncio.to_thread и обработку сообщения пользователя, и фоновый idle-тик,
и /status, и выгрузку сессии по бездействию. Все они трогают одно и то
же: соединение SQLite, буфер STM, cortex.last_action_trace, часы.

Достижимый сценарий: пользователь пишет на 46-й секунде молчания ровно
тогда, когда фоновый тик начал фазу сна — синаптический прунинг удаляет
узлы под ногами у идущего поиска по памяти.

Гонки по своей природе недетерминированы, поэтому здесь проверяется не
"поймали ли мы гонку", а инварианты, которые блокировка обязана
обеспечивать: взаимное исключение, приоритет пользователя над фоновым
тиком и отсутствие порчи состояния под нагрузкой.
"""
import threading
import time

import pytest

from core.brain_session import BrainSession


@pytest.fixture
def session(monkeypatch):
    """Сессия на in-memory SQLite с заглушенной LLM (без сети)."""
    import core.cortex
    import memory.graph_memory
    import memory.sleep_cycle

    def fake_llm(messages, system_prompt=None, max_tokens=None):
        return "ответ"

    for module in (core.cortex, memory.graph_memory, memory.sleep_cycle):
        monkeypatch.setattr(module, "generate_llm_response", fake_llm)

    brain = BrainSession(db_path=":memory:")
    yield brain
    brain.close()


# ---------------------------------------------------------------------------
# Взаимное исключение
# ---------------------------------------------------------------------------
def test_message_processing_is_mutually_exclusive(session):
    """
    Два потока не должны оказаться внутри обработки одновременно —
    иначе они делят SQLite-курсоры, STM и last_action_trace.
    """
    inside = 0
    max_inside = 0
    guard = threading.Lock()
    original = session._process_message_unlocked

    def instrumented(text):
        nonlocal inside, max_inside
        with guard:
            inside += 1
            max_inside = max(max_inside, inside)
        try:
            time.sleep(0.005)  # расширяем окно, где гонка была бы видна
            return original(text)
        finally:
            with guard:
                inside -= 1

    session._process_message_unlocked = instrumented

    threads = [
        threading.Thread(target=session.process_message, args=(f"сообщение {i}",))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not any(t.is_alive() for t in threads), "потоки должны завершиться, а не залипнуть"
    assert max_inside == 1, (
        f"внутри обработки одновременно оказалось до {max_inside} потоков — "
        "блокировка сессии не работает"
    )


# ---------------------------------------------------------------------------
# Приоритет пользователя
# ---------------------------------------------------------------------------
def test_idle_tick_yields_to_a_busy_session(session):
    """
    Фоновый тик обязан УСТУПАТЬ: он необязателен и повторится через
    несколько секунд, а пользователь иначе ждал бы, пока фаза сна сходит
    в LLM (до 30 секунд таймаута).
    """
    session._lock.acquire()
    try:
        assert session.run_idle_tick(5.0) is None, (
            "занятая сессия не должна обрабатывать фоновый тик"
        )
    finally:
        session._lock.release()

    # Освободилась — тик снова допустим (событие может быть и None,
    # важно лишь, что вызов реально доходит до тела)
    calls = []
    original = session._run_idle_tick_unlocked
    session._run_idle_tick_unlocked = lambda dt: (calls.append(dt), original(dt))[1]
    session.run_idle_tick(5.0)
    assert calls == [5.0]


def test_user_message_is_not_skipped_when_contended(session):
    """
    Обратная сторона: сообщение пользователя НЕ уступает, а ждёт своей
    очереди. Потерять реплику человека недопустимо.
    """
    results = []

    def worker(i):
        results.append(session.process_message(f"привет {i}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(results) == 5, "ни одно сообщение пользователя не должно быть пропущено"
    assert all(r.text for r in results), "на каждое сообщение должен быть ответ"


# ---------------------------------------------------------------------------
# Состояние не портится под смешанной нагрузкой
# ---------------------------------------------------------------------------
def test_mixed_load_keeps_state_consistent(session):
    """
    Сообщения, фоновые тики и /status вперемешку — как в проде. Ни одно
    из этого не должно бросить исключение или разъехаться по состоянию.
    """
    errors = []

    def messages():
        try:
            for i in range(10):
                session.process_message(f"мама мыла раму {i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(("process_message", exc))

    def ticks():
        try:
            for _ in range(10):
                session.run_idle_tick(1.0)
                time.sleep(0.001)
        except Exception as exc:  # noqa: BLE001
            errors.append(("run_idle_tick", exc))

    def statuses():
        try:
            for _ in range(10):
                assert session.get_status_report()
                time.sleep(0.001)
        except Exception as exc:  # noqa: BLE001
            errors.append(("get_status_report", exc))

    threads = [
        threading.Thread(target=messages),
        threading.Thread(target=ticks),
        threading.Thread(target=statuses),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not any(t.is_alive() for t in threads), "дедлок: потоки не завершились"
    assert not errors, f"смешанная нагрузка сломала сессию: {errors}"

    # Состояние осталось осмысленным
    assert session.memory.count_nodes() > 0
    assert session.stm.size() <= session.stm.capacity
