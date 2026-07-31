"""
Тесты единой оси возбуждения.

Контекст: это состояние жило ДВАЖДЫ — как InstinctSystem.current_stress и
как Mood.anxiety. Причём после перехода на теорию оценки стало хуже:
coping = 1 - current_stress начал питать anxiety, и тревога превратилась в
производную от стресса, которая никуда не идёт. Одно измерялось двумя
приборами, а работал только один.

Хуже того, замер показал, что и «работающий» прибор был МЁРТВЫМ КОДОМ:
стресс накапливался на +0.025 за сообщение при восстановлении -6.00 за
тик, то есть мгновенно падал в ноль. is_overloaded не срабатывал никогда,
и заявленный в манифесте слой самозащиты не включался ни разу.

Теперь ось одна, живёт в Mood (возбуждение и валентность вместе образуют
ядерный аффект — разносить их по модулям и было несвязностью), питается
оценками, а InstinctSystem стал набором реакций, параметризованных ею.
"""
import pytest

import config
from core.instincts import InstinctSystem
from core.mood import Appraisal, Mood


@pytest.fixture
def mood():
    return Mood()


# ---------------------------------------------------------------------------
# Возбуждение растёт от того, что требует внимания
# ---------------------------------------------------------------------------
def test_novelty_raises_arousal(mood):
    before = mood.arousal
    state = mood.appraise(Appraisal(novelty=0.9), log=False)
    assert state.arousal > before


def test_reward_surprise_raises_arousal_regardless_of_sign():
    """Нагрузку создаёт величина ошибки предсказания, а не её знак."""
    good = Mood().appraise(Appraisal(goal_congruence=0.9), log=False)
    bad = Mood().appraise(Appraisal(goal_congruence=-0.9), log=False)

    assert good.arousal == pytest.approx(bad.arousal)
    assert good.arousal > config.MOOD_BASELINE_AROUSAL


def test_quiet_event_leaves_arousal_at_baseline(mood):
    state = mood.appraise(Appraisal(), log=False)
    assert state.arousal == pytest.approx(config.MOOD_BASELINE_AROUSAL)


def test_arousal_decays_back_to_rest(mood):
    mood.appraise(Appraisal(novelty=1.0, goal_congruence=1.0), log=False)
    raised = mood.arousal
    for _ in range(40):
        mood.decay(log=False)
    assert mood.arousal < raised
    assert mood.arousal == pytest.approx(config.MOOD_BASELINE_AROUSAL, abs=0.02)


def test_sustained_novelty_does_not_saturate_into_permanent_overload(mood):
    """
    Регрессия на найденный замером провал: при слишком высокой
    чувствительности возбуждение вставало на 0.84 и организм оказывался в
    ПОСТОЯННОЙ перегрузке — принудительная мимикрия навсегда, вызовов LLM
    14 вместо 86.
    """
    for _ in range(80):
        mood.appraise(Appraisal(novelty=0.9, coping=mood.coping()), log=False)
        mood.decay(log=False)

    assert not mood.is_overloaded(), (
        f"постоянная новизна не должна давать вечную перегрузку "
        f"(возбуждение {mood.arousal:.2f} при пороге {config.STRESS_OVERLOAD_THRESHOLD})"
    )


# ---------------------------------------------------------------------------
# Одна ось — один источник истины
# ---------------------------------------------------------------------------
def test_coping_is_the_other_side_of_arousal(mood):
    mood.appraise(Appraisal(novelty=1.0, goal_congruence=1.0), log=False)
    assert mood.coping() == pytest.approx(1.0 - mood.arousal)


def test_instincts_hold_no_state_of_their_own():
    """
    InstinctSystem больше не хранит нагрузку: одно и то же состояние в двух
    местах и было исходным дефектом.
    """
    instincts = InstinctSystem()
    assert not hasattr(instincts, "current_stress")

    calm = instincts.get_state(arousal=0.0)
    loaded = instincts.get_state(arousal=0.9)

    assert not calm.is_overloaded
    assert loaded.is_overloaded
    assert loaded.effective_plasticity_threshold > calm.effective_plasticity_threshold


def test_overload_reads_from_the_single_axis():
    instincts = InstinctSystem()
    below = config.STRESS_OVERLOAD_THRESHOLD - 0.01
    above = config.STRESS_OVERLOAD_THRESHOLD + 0.01

    assert not instincts.is_overloaded(below)
    assert instincts.is_overloaded(above)


# ---------------------------------------------------------------------------
# Нагрузка действительно модулирует запись в память
# ---------------------------------------------------------------------------
def test_load_makes_the_organism_less_impressionable(mood):
    """
    Смысл слоя самозащиты: под нагрузкой порог записи растёт. Раньше это
    было заявлено, но не работало — стресс всегда был нулевым.
    """
    calm = Mood()
    calm.arousal = 0.0
    loaded = Mood()
    loaded.arousal = 1.0

    assert loaded.effective_plasticity_threshold() > calm.effective_plasticity_threshold()
    assert calm.effective_plasticity_threshold() == pytest.approx(
        config.BASE_PLASTICITY_THRESHOLD
    )


def test_effective_threshold_never_exceeds_one(mood):
    mood.arousal = 1.0
    assert mood.effective_plasticity_threshold() <= 1.0


# ---------------------------------------------------------------------------
# Перегрузка возвращает организм к мимикрии
# ---------------------------------------------------------------------------
def test_overload_pushes_speech_back_towards_babbling():
    """Регрессия под давлением — как у настоящих детей."""
    instincts = InstinctSystem()
    vocabulary = config.BABBLING_VOCABULARY_THRESHOLD + 40

    calm_ratio = instincts.get_babble_ratio(vocabulary, arousal=0.0)
    overloaded_ratio = instincts.get_babble_ratio(
        vocabulary, arousal=config.STRESS_OVERLOAD_THRESHOLD + 0.1
    )

    assert overloaded_ratio > calm_ratio
