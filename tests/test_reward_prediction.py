"""
Тесты дофаминового сигнала — ошибки предсказания награды.

Контекст: подкрепление работало "задним числом" — пришла похвала, усилили
участвовавший узел. Это память о награде, но не СТРЕМЛЕНИЕ к ней:
организм никак не выбирал поведение, которое раньше одобряли.

Ключевая поправка, без которой конструкция вырождается: дофамин
выделяется не на награду, а на НЕОЖИДАННУЮ награду. Наивное "стремиться
к одобрению" схлопнулось бы мгновенно — организм нашёл бы одно слово,
которое всегда хвалят, и долбил его вечно. Эти тесты в первую очередь
защищают именно от такого вырождения.
"""
import pytest

import config
from core.instincts import InstinctSystem
from memory.database import Database
from memory.graph_memory import MemoryGraph


@pytest.fixture
def mg():
    return MemoryGraph(db=Database(db_path=":memory:"))


@pytest.fixture
def node(mg):
    return mg.db.insert_node(
        context="привет", response="привет", weight=0.5, timestamp=0.0, node_type="episodic"
    )


# ---------------------------------------------------------------------------
# Ошибка предсказания
# ---------------------------------------------------------------------------
def test_first_reward_is_fully_unexpected(mg, node):
    """Ожидание начинается с нуля, поэтому первая похвала — чистая новость."""
    signal = mg.apply_reward(node, 0.8)

    assert signal.expected == pytest.approx(0.0)
    assert signal.prediction_error == pytest.approx(0.8)


def test_repeated_identical_praise_habituates(mg, node):
    """
    ГЛАВНОЕ СВОЙСТВО: то, что хвалят ВСЕГДА, перестаёт давать сигнал.
    Именно это не даёт организму залипнуть на одном удачном слове.
    """
    errors = [mg.apply_reward(node, 0.8).prediction_error for _ in range(6)]

    assert all(
        later < earlier for earlier, later in zip(errors, errors[1:])
    ), f"ошибка предсказания обязана убывать при однообразной похвале: {errors}"
    assert errors[-1] < errors[0] / 4


def test_expectation_converges_towards_actual_valence(mg, node):
    for _ in range(20):
        signal = mg.apply_reward(node, 0.8)
    assert signal.new_expectation == pytest.approx(0.8, abs=0.05)


def test_violated_expectation_is_the_strongest_signal(mg, node):
    """
    Ругань после привычной похвалы должна бить сильнее, чем ругань
    на пустом месте: нарушенное ожидание информативнее.
    """
    for _ in range(6):
        mg.apply_reward(node, 0.8)
    after_praise = mg.apply_reward(node, -0.8).prediction_error

    fresh = mg.db.insert_node(
        context="x", response="y", weight=0.5, timestamp=0.0, node_type="episodic"
    )
    from_scratch = mg.apply_reward(fresh, -0.8).prediction_error

    assert abs(after_praise) > abs(from_scratch)


def test_expectation_stays_in_valence_range(mg, node):
    for _ in range(50):
        mg.apply_reward(node, 1.0)
    assert -1.0 <= mg.db.get_node(node)["reward_expectation"] <= 1.0


def test_reward_on_missing_node_is_survivable(mg):
    """Узел мог попасть под прунинг между действием и оценкой."""
    assert mg.apply_reward(999999, 0.8) is None


def test_reward_does_not_refresh_the_decay_clock(mg, node):
    """
    Получение оценки — не то же самое, что вспоминание. Иначе любая
    реакция пользователя продлевала бы жизнь узлу, включая ругань.
    """
    before = mg.db.get_node(node)["last_accessed"]
    mg.apply_reward(node, -0.9, timestamp=50_000.0)
    assert mg.db.get_node(node)["last_accessed"] == before


# ---------------------------------------------------------------------------
# Темп закрепления
# ---------------------------------------------------------------------------
def test_learning_scale_follows_surprise(mg):
    assert mg.learning_scale(0.9) > mg.learning_scale(0.3)
    assert mg.learning_scale(-0.9) == mg.learning_scale(0.9), "важен модуль, а не знак"


def test_learning_scale_never_reaches_zero(mg):
    """
    Полностью ожидаемая награда не должна ОБНУЛЯТЬ обучение — иначе
    давно освоенный узел перестал бы получать даже поддерживающее
    подкрепление.
    """
    assert mg.learning_scale(0.0) == pytest.approx(config.REWARD_MIN_LEARNING_SCALE)
    assert mg.learning_scale(0.0) > 0.0


def test_learning_scale_is_capped(mg):
    assert mg.learning_scale(5.0) <= 1.0


# ---------------------------------------------------------------------------
# Награда влияет на ВЫБОР — то самое "стремление"
# ---------------------------------------------------------------------------
def test_reward_shifts_word_preference(mg):
    """При РАВНОЙ освоенности выбирается то слово, за которое хвалили."""
    for i in range(6):
        mg.process_language_input("кот дом мяч", timestamp=float(i))

    words = {w.text: w for w in mg.get_mastered_words_in("кот дом мяч")}
    assert words["кот"].weight == pytest.approx(words["мяч"].weight), "исходные веса равны"

    for i in range(4):
        mg.apply_reward(words["мяч"].id, +0.9, timestamp=float(100 + i))
        mg.apply_reward(words["кот"].id, -0.9, timestamp=float(100 + i))

    after = {w.text: w for w in mg.get_mastered_words_in("кот дом мяч")}
    assert after["мяч"].preference > after["дом"].preference > after["кот"].preference
    assert after["мяч"].weight == pytest.approx(after["кот"].weight), (
        "освоенность не должна была измениться — сместилось именно предпочтение"
    )


def test_praised_word_is_chosen_for_the_reply(mg, monkeypatch):
    monkeypatch.setattr(config, "MIMICRY_MAX_KNOWN_WORDS", 1)
    instincts = InstinctSystem()

    for i in range(6):
        mg.process_language_input("кот дом мяч", timestamp=float(i))
    words = {w.text: w for w in mg.get_mastered_words_in("кот дом мяч")}
    for i in range(4):
        mg.apply_reward(words["мяч"].id, +0.9, timestamp=float(100 + i))

    result = instincts.generate_blended_mimicry_response(
        "кот дом мяч", mg.get_known_syllables(), mg.get_vocabulary_size(),
        mastered_words=mg.get_mastered_words_in("кот дом мяч"),
    )

    assert "мяч" in result.text, (
        f"организм должен предпочесть слово, за которое его хвалили, но сказал {result.text!r}"
    )


def test_preference_does_not_override_mastery(mg):
    """
    Одна похвала за едва знакомое слово не должна перебивать хорошо
    освоенное — иначе организм начнёт говорить редкими словами вместо
    тех, которыми реально владеет.
    """
    for i in range(20):
        mg.process_language_input("привет", timestamp=float(i))
    for i in range(3):
        mg.process_language_input("абажур", timestamp=float(100 + i))

    words = {w.text: w for w in mg.get_mastered_words_in("привет абажур")}
    mg.apply_reward(words["абажур"].id, +1.0, timestamp=200.0)

    after = {w.text: w for w in mg.get_mastered_words_in("привет абажур")}
    assert after["привет"].preference > after["абажур"].preference
