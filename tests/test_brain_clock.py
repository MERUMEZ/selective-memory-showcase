"""
Тесты когерентности субъективного времени.

Раньше часы были составными и ломались тремя способами сразу:

1. TICK_SECONDS=120 прибавлялись за КАЖДОЕ сообщение независимо от того,
   сколько прошло на самом деле. Замер: часы бежали ВСЕМЕРО быстрее
   настенных, из-за чего все временные константы тайно зависели от
   интенсивности общения (AGE_T0="1 час" означал ~9 минут разговора).

2. BrainSession.__init__ ставил brain_time = time.time(), поэтому при
   выгрузке и возврате сессии часы ПРЫГАЛИ НАЗАД, а last_decayed_at узлов
   оставался от разогнанных. В _decay_nodes стоит `if dt <= 0: continue` —
   значит забывание МОЛЧА выключалось: после разговора на 100 сообщений
   ещё на 2.3 часа.

3. seconds_since_last_user_message считалось по brain_time, а
   seconds_since_last_activity — по time.time(): одна и та же пауза
   мерялась двумя линейками.
"""
import time

import pytest

import config
from core.brain_session import SharedBrainClock


@pytest.fixture
def clock():
    return SharedBrainClock(epoch=time.time())


# ---------------------------------------------------------------------------
# Одна шкала с заявленным ускорением
# ---------------------------------------------------------------------------
def test_brain_time_runs_at_the_declared_acceleration(clock):
    before = clock.get_brain_time()
    time.sleep(0.05)
    elapsed_brain = clock.get_brain_time() - before

    assert elapsed_brain == pytest.approx(0.05 * config.TIME_ACCELERATION, rel=0.5)


def test_brain_time_is_monotonic(clock):
    readings = [clock.get_brain_time() for _ in range(200)]
    assert readings == sorted(readings)


def test_messages_do_not_advance_the_clock(clock):
    """
    Главное отличие от прежней схемы: реплика ОТМЕЧАЕТСЯ, но не двигает
    время. Раньше каждое сообщение прибавляло 120 секунд независимо от
    реальности, и внутренние часы разгонялись тем сильнее, чем быстрее
    печатал человек.
    """
    before = clock.get_brain_time()
    for _ in range(50):
        clock.register_user_message()
    after = clock.get_brain_time()

    # За 50 вызовов проходят миллисекунды -> прирост должен быть
    # микроскопическим, а не 50 * 120 секунд
    assert after - before < 1.0


# ---------------------------------------------------------------------------
# Перезагрузка не сбрасывает отсчёт
# ---------------------------------------------------------------------------
def test_clock_resumes_from_the_stored_epoch():
    """
    Часы — чистая функция настенного времени от сохранённой эпохи, поэтому
    новый экземпляр обязан продолжить, а не начать заново.
    """
    epoch = time.time() - 3600.0  # мозг живёт уже час настенного времени

    first = SharedBrainClock(epoch=epoch)
    reading = first.get_brain_time()

    reborn = SharedBrainClock(epoch=epoch)
    assert reborn.get_brain_time() >= reading, "часы не должны идти назад"


def test_reload_does_not_stall_decay():
    """
    Регрессия на самый вредный из трёх дефектов: после перезагрузки
    brain_time оказывался ПОЗАДИ меток last_decayed_at, dt выходил
    отрицательным, и забывание молча выключалось на часы.
    """
    epoch = time.time() - 7200.0
    session_clock = SharedBrainClock(epoch=epoch)
    stamp_written_during_session = session_clock.get_brain_time()

    reborn = SharedBrainClock(epoch=epoch)
    dt = reborn.get_brain_time() - stamp_written_during_session

    assert dt >= 0.0, (
        "после перезагрузки время не должно оказаться позади уже записанных меток"
    )


# ---------------------------------------------------------------------------
# Внешнее меряется внешним, внутреннее — внутренним
# ---------------------------------------------------------------------------
def test_external_absence_is_measured_in_wall_seconds(clock):
    """
    "Ушёл ли пользователь" и "пора ли выгрузить сессию" — вопросы про
    внешний мир и оперативную память, поэтому меряются настенными
    секундами намеренно.
    """
    clock.register_activity()
    time.sleep(0.05)
    assert clock.seconds_since_last_activity() == pytest.approx(0.05, abs=0.04)


def test_internal_silence_is_measured_in_subjective_seconds(clock):
    clock.register_user_message()
    time.sleep(0.05)

    subjective = clock.seconds_since_last_user_message()
    assert subjective > 0.05, "внутренняя пауза идёт по ускоренной шкале"
    assert subjective == pytest.approx(0.05 * config.TIME_ACCELERATION, rel=0.6)


# ---------------------------------------------------------------------------
# Промотка для стенда
# ---------------------------------------------------------------------------
def test_simulated_gap_advances_by_the_accelerated_amount(clock):
    """
    Эпоха входит в формулу дважды, поэтому наивный сдвиг промотал бы часы
    в разы дальше нужного. Стенд обязан получать ровно то, что просит.
    """
    before = clock.get_brain_time()
    clock.simulate_elapsed_wall_seconds(8 * 3600.0)
    advanced = clock.get_brain_time() - before

    assert advanced == pytest.approx(8 * 3600.0 * config.TIME_ACCELERATION, rel=0.01)


def test_simulated_gap_also_ages_external_absence(clock):
    clock.register_activity()
    clock.simulate_elapsed_wall_seconds(3600.0)
    assert clock.seconds_since_last_activity() == pytest.approx(3600.0, abs=1.0)
