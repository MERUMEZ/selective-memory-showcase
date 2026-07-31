"""
Тесты эмоции как ОЦЕНКИ события (appraisal theory).

Контекст: раньше настроение назначалось руками магическими числами —
`apply_stimulus(MoodDelta(curiosity=perplexity * 0.15))`, `joy = positive
* 0.4`, `anxiety = negative * 0.35 + emotion_score * 0.1`. Коэффициенты не
следовали ни из чего и не были ни на чём измерены.

Теория оценки говорит, что эмоция — результат оценки события по
измерениям, и организм эти измерения уже вычисляет: новизну
(compute_surprise), соответствие цели (ошибка предсказания награды) и
способность справиться (1 - стресс). Настроение теперь выводится из них.

Здесь проверяется не «совпало ли число», а структурные свойства, ради
которых всё затевалось.
"""
import pytest

import config
from core.mood import Appraisal, Mood


@pytest.fixture
def mood():
    return Mood()


# ---------------------------------------------------------------------------
# Новизна: интерес или страх — решает способность справиться
# ---------------------------------------------------------------------------
def test_novelty_with_resources_is_curiosity(mood):
    state = mood.appraise(Appraisal(novelty=0.9, coping=1.0), log=False)
    assert state.dominant_emotion() == "curiosity"


def test_same_novelty_without_resources_is_anxiety():
    """
    Ключевое свойство: одна и та же новизна даёт РАЗНУЮ эмоцию в
    зависимости от того, есть ли чем её переварить. Непонятное при
    перегрузке — это не интерес, а тревога.
    """
    calm = Mood().appraise(Appraisal(novelty=0.9, coping=1.0), log=False)
    overwhelmed = Mood().appraise(Appraisal(novelty=0.9, coping=0.0), log=False)

    assert calm.curiosity > overwhelmed.curiosity
    assert overwhelmed.anxiety > calm.anxiety
    assert overwhelmed.dominant_emotion() == "anxiety"


def test_coping_scales_curiosity_monotonically():
    previous = None
    for coping in (0.0, 0.25, 0.5, 0.75, 1.0):
        state = Mood().appraise(Appraisal(novelty=0.8, coping=coping), log=False)
        if previous is not None:
            assert state.curiosity > previous
        previous = state.curiosity


# ---------------------------------------------------------------------------
# Радость идёт от ОШИБКИ предсказания, а не от самой похвалы
# ---------------------------------------------------------------------------
def test_unexpected_praise_is_more_joyful_than_expected_one():
    """
    Привычная похвала не должна радовать так же, как неожиданная. В коде
    это выражено тем, что на вход идёт rpe, а не сырая валентность.
    """
    surprising = Mood().appraise(Appraisal(goal_congruence=0.8), log=False)
    routine = Mood().appraise(Appraisal(goal_congruence=0.1), log=False)

    assert surprising.joy > routine.joy


def test_joy_increment_shrinks_as_praise_becomes_expected():
    """
    Прогон габитуации: та же похвала, но ожидание догоняет её, поэтому
    каждый следующий раз волнует слабее.
    """
    mood = Mood()
    expected = 0.0
    increments = []
    for _ in range(5):
        rpe = 0.8 - expected
        expected += config.REWARD_EXPECTATION_LEARNING_RATE * rpe
        before = mood.joy
        state = mood.appraise(Appraisal(goal_congruence=rpe), log=False)
        increments.append(state.joy - before)
        mood.decay(log=False)

    assert all(
        later < earlier for earlier, later in zip(increments, increments[1:])
    ), f"прирост радости обязан убывать: {increments}"
    assert increments[-1] < increments[0] / 3


def test_worse_than_expected_raises_anxiety(mood):
    state = mood.appraise(Appraisal(goal_congruence=-0.8), log=False)
    assert state.dominant_emotion() == "anxiety"
    assert state.joy == pytest.approx(config.MOOD_BASELINE_JOY)


# ---------------------------------------------------------------------------
# Шкалы времени: эмоция быстрая, привязанность медленная
# ---------------------------------------------------------------------------
def test_affection_outlives_joy():
    """
    Раньше привязанность затухала с той же gamma, что радость, и бот
    "отвыкал" от наставника за пять реплик. Связь — медленная переменная.
    """
    mood = Mood()
    for _ in range(6):
        mood.appraise(Appraisal(goal_congruence=0.8), log=False)

    joy_before, affection_before = mood.joy, mood.affection
    for _ in range(15):
        mood.decay(log=False)

    joy_lost = (joy_before - mood.joy) / max(1e-9, joy_before)
    affection_lost = (affection_before - mood.affection) / max(1e-9, affection_before)

    assert joy_lost > affection_lost * 2, (
        "радость должна уходить существенно быстрее привязанности"
    )
    assert mood.affection > config.MOOD_BASELINE_AFFECTION


def test_affection_grows_slower_than_joy(mood):
    """Связь копится за много взаимодействий, а не вспыхивает за одно."""
    state = mood.appraise(Appraisal(goal_congruence=1.0), log=False)
    assert (state.joy - config.MOOD_BASELINE_JOY) > (
        state.affection - config.MOOD_BASELINE_AFFECTION
    )


# ---------------------------------------------------------------------------
# Краевые случаи
# ---------------------------------------------------------------------------
def test_neutral_event_leaves_mood_at_baseline(mood):
    state = mood.appraise(Appraisal(), log=False)
    assert state.dominant_emotion() == "neutral"


@pytest.mark.parametrize(
    "appraisal",
    [
        Appraisal(novelty=5.0, coping=3.0),
        Appraisal(novelty=-2.0, coping=-1.0),
        Appraisal(goal_congruence=99.0),
        Appraisal(goal_congruence=-99.0),
    ],
    ids=["новизна за границей", "отрицательные", "огромная награда", "огромный штраф"],
)
def test_out_of_range_appraisal_keeps_axes_in_unit_range(mood, appraisal):
    state = mood.appraise(appraisal, log=False)
    for axis in (state.joy, state.curiosity, state.anxiety, state.affection):
        assert 0.0 <= axis <= 1.0
