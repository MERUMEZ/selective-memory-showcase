"""
Тесты привязанности как поведенческой величины: чем ближе стал человек,
тем хуже организм переносит его молчание.

Контекст: порог скуки был константой — организм ждал одинаково независимо
от того, сложилась ли у него связь с этим собеседником. При этом
привязанность копится за много тёплых обменов и живёт на своей, медленной
шкале (MOOD_AFFECTION_DECAY_RATE), то есть её естественное следствие —
именно долгая линия поведения.

Это второй потребитель настроения после исследования, и он тоже ничего не
дублирует: порогом скуки больше ничто не управляет.
"""
import math

import pytest

import config
from core.drives import BoredomDrive
from core.mood import Appraisal, Mood


def silence_minutes(threshold: float) -> float:
    """Обратная формула скуки: boredom(dt) = 1 - exp(-dt/TAU)."""
    return -config.BOREDOM_TAU * math.log(1 - threshold) / 60.0


# ---------------------------------------------------------------------------
# Порог зависит от привязанности
# ---------------------------------------------------------------------------
def test_baseline_affection_leaves_threshold_untouched():
    """В покое поведение должно остаться прежним — правка аддитивна."""
    assert BoredomDrive.effective_threshold(
        config.MOOD_BASELINE_AFFECTION
    ) == pytest.approx(config.BOREDOM_THRESHOLD)


def test_attachment_lowers_the_threshold_monotonically():
    previous = None
    for affection in (0.1, 0.3, 0.5, 0.8, 1.0):
        threshold = BoredomDrive.effective_threshold(affection)
        if previous is not None:
            assert threshold < previous, "привязанность обязана снижать порог терпения"
        previous = threshold


def test_attached_organism_reaches_out_sooner():
    """Осмысленная разница во времени, а не просто другое число."""
    detached = silence_minutes(BoredomDrive.effective_threshold(0.1))
    attached = silence_minutes(BoredomDrive.effective_threshold(1.0))

    assert detached > attached
    assert detached - attached > 2.0, (
        f"разница должна быть заметной: {detached:.1f} мин против {attached:.1f}"
    )


def test_threshold_never_falls_below_the_floor():
    """
    Даже сильнейшая связь не должна превращать организм в навязчивого
    собеседника.
    """
    assert BoredomDrive.effective_threshold(1.0) >= config.BOREDOM_THRESHOLD_FLOOR
    assert BoredomDrive.effective_threshold(99.0) >= config.BOREDOM_THRESHOLD_FLOOR


def test_below_baseline_affection_does_not_raise_the_threshold():
    """Отвязанность не должна делать организм терпеливее обычного."""
    assert BoredomDrive.effective_threshold(0.0) == pytest.approx(config.BOREDOM_THRESHOLD)


# ---------------------------------------------------------------------------
# Привязанность действительно накапливается общением
# ---------------------------------------------------------------------------
def test_warm_interactions_build_attachment_over_time():
    previous = None
    for exchanges in (0, 5, 15, 40):
        mood = Mood()
        for _ in range(exchanges):
            mood.appraise(Appraisal(goal_congruence=0.7), log=False)
            mood.decay(log=False)
        if previous is not None:
            assert mood.affection > previous
        previous = mood.affection


def test_attachment_translates_into_impatience():
    """Сквозной путь: тёплое общение -> привязанность -> меньше терпения."""
    fresh = Mood()
    familiar = Mood()
    for _ in range(20):
        familiar.appraise(Appraisal(goal_congruence=0.7), log=False)
        familiar.decay(log=False)

    fresh_wait = silence_minutes(BoredomDrive.effective_threshold(fresh.affection))
    familiar_wait = silence_minutes(BoredomDrive.effective_threshold(familiar.affection))

    assert familiar_wait < fresh_wait


def test_attachment_outlives_a_pause_in_conversation():
    """
    Связь не должна испаряться за несколько тиков молчания — иначе она не
    успеет ни на что повлиять.
    """
    mood = Mood()
    for _ in range(20):
        mood.appraise(Appraisal(goal_congruence=0.7), log=False)
        mood.decay(log=False)

    attached = mood.affection
    for _ in range(20):
        mood.decay(log=False)

    assert mood.affection > attached * 0.6, (
        "привязанность обязана переживать паузу в разговоре"
    )
