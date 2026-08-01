"""
Тесты исследования — попытки произнести ещё не освоенное слово.

Контекст: такого механизма в архитектуре не было ВООБЩЕ. Организм только
эксплуатировал уже выученное, а всё незнакомое превращал в лепет. Чисто
эксплуатирующая система не развивается: она сходится к тому, что уже
умеет, и застывает — для "живого мозга" это принципиальный провал.

Исследование — единственный потребитель curiosity, и он ничего не
дублирует. Настроение уникально тем, что несёт ИНЕРЦИЮ: сырые сигналы
говорят о текущем событии, а любопытство копится через ходы.

Замыкает контур освоения: попробовал -> похвалили -> закрепилось.
"""
import random

import pytest

import config
from core.cortex import Cortex
from core.instincts import InstinctSystem
from memory.database import Database
from memory.graph_memory import MemoryGraph


@pytest.fixture
def mg():
    return MemoryGraph(db=Database(db_path=":memory:"))


def teach(graph, text, times):
    for i in range(times):
        graph.process_language_input(text, timestamp=float(i))


# ---------------------------------------------------------------------------
# Зона ближайшего развития
# ---------------------------------------------------------------------------
def test_emerging_words_are_heard_but_not_mastered(mg):
    teach(mg, "привет", times=6)          # освоено
    teach(mg, "шкатулка", times=1)        # услышано, не закреплено

    mastered = {w.text for w in mg.get_mastered_words_in("привет шкатулка")}
    emerging = {w.text for w in mg.get_emerging_words_in("привет шкатулка")}

    assert mastered == {"привет"}
    assert emerging == {"шкатулка"}
    assert not (mastered & emerging), "множества не должны пересекаться"


def test_completely_unknown_words_are_not_exploration_candidates(mg):
    """
    Пробовать то, что далеко за пределами компетенции, бессмысленно —
    попытка провалится и ничему не научит. Кандидаты только из услышанного.
    """
    teach(mg, "привет", times=6)
    assert mg.get_emerging_words_in("абракадабра") == []


# ---------------------------------------------------------------------------
# Любопытство управляет склонностью рискнуть
# ---------------------------------------------------------------------------
def _exploration_rate(cortex, phrase, trials=300, seed=1):
    random.seed(seed)
    return sum(1 for _ in range(trials) if cortex._pick_exploration_word(phrase) is not None) / trials


def test_curiosity_raises_exploration_rate(mg):
    teach(mg, "привет", times=6)
    teach(mg, "кола", times=1)
    phrase = "привет кола"

    calm = Cortex(memory=mg)
    calm.mood.curiosity = config.MOOD_BASELINE_CURIOSITY
    curious = Cortex(memory=mg)
    curious.mood.curiosity = 1.0

    assert _exploration_rate(curious, phrase) > _exploration_rate(calm, phrase) * 2


def test_calm_organism_still_explores_sometimes(mg):
    """
    Полностью нелюбопытный организм всё равно изредка пробует новое —
    иначе он не смог бы выйти из состояния покоя вообще.
    """
    teach(mg, "привет", times=6)
    teach(mg, "кола", times=1)

    calm = Cortex(memory=mg)
    calm.mood.curiosity = 0.0

    assert _exploration_rate(calm, "привет кола", trials=600) > 0.0


def test_exploration_picks_the_edge_of_competence(mg):
    """Пробуется САМОЕ БЛИЗКОЕ к порогу — там, где и происходит учение."""
    # Веса задаются НАПРЯМУЮ, а не числом повторений: сколько употреблений
    # нужно до освоения, зависит от темпа (SPEECH_DEMO_PACE), и фикстура,
    # завязанная на счёт повторов, ломается при его смене.
    mastery = config.VOCABULARY_MASTERY_MIN_WEIGHT
    mg.db.upsert_lexical_node("word", "почти", initial_weight=mastery - 0.01,
                              reinforce_step=0.0, timestamp=0.0)
    mg.db.upsert_lexical_node("word", "едва", initial_weight=mastery - 0.06,
                              reinforce_step=0.0, timestamp=0.0)
    phrase = "почти едва"

    cortex = Cortex(memory=mg)
    cortex.mood.curiosity = 1.0
    random.seed(0)

    picked = set()
    for _ in range(50):
        word = cortex._pick_exploration_word(phrase)
        if word is not None:
            picked.add(word.text)

    assert picked == {"почти"}, (
        f"пробоваться должно слово на границе освоенного, а выбрано {picked}"
    )


def test_no_candidates_means_no_exploration(mg):
    teach(mg, "привет", times=6)
    cortex = Cortex(memory=mg)
    cortex.mood.curiosity = 1.0
    assert cortex._pick_exploration_word("привет") is None


# ---------------------------------------------------------------------------
# Опробованное слово реально произносится и попадает в контур подкрепления
# ---------------------------------------------------------------------------
def test_explored_word_is_spoken_and_reinforceable(mg):
    teach(mg, "привет", times=6)
    teach(mg, "кола", times=1)

    emerging = mg.get_emerging_words_in("привет кола")[0]
    result = InstinctSystem().generate_blended_mimicry_response(
        "привет кола", mg.get_known_syllables(), mg.get_vocabulary_size(),
        mastered_words=mg.get_mastered_words_in("привет кола"),
        exploration_word=emerging,
    )

    assert "кола" in result.text, "опробованное слово должно быть произнесено"
    assert emerging.id in result.used_node_ids, (
        "узел опробованного слова обязан попасть в used_node_ids, иначе "
        "похвала до него не дойдёт и контур освоения не замкнётся"
    )


def test_confident_words_come_before_the_attempt(mg):
    """Сначала то, в чём организм уверен, потом то, что он договаривает."""
    teach(mg, "привет", times=6)
    teach(mg, "кола", times=1)

    result = InstinctSystem().generate_blended_mimicry_response(
        "привет кола", mg.get_known_syllables(), mg.get_vocabulary_size(),
        mastered_words=mg.get_mastered_words_in("привет кола"),
        exploration_word=mg.get_emerging_words_in("привет кола")[0],
    )

    words = result.text.split()
    assert words.index("привет") < words.index("кола")


# ---------------------------------------------------------------------------
# Главное: контур освоения замыкается
# ---------------------------------------------------------------------------
def test_praise_after_exploration_completes_acquisition(mg):
    """
    попробовал -> похвалили -> закрепилось.

    До появления исследования этого пути не существовало: неосвоенное
    слово могло превратиться только в лепет и никогда не получало шанса
    быть подкреплённым.
    """
    teach(mg, "привет", times=6)
    teach(mg, "кола", times=1)
    phrase = "привет кола"

    instincts = InstinctSystem()
    cortex = Cortex(memory=mg, instincts=instincts)

    emerging = mg.get_emerging_words_in(phrase)[0]
    assert emerging.text == "кола"

    result = instincts.generate_blended_mimicry_response(
        phrase, mg.get_known_syllables(), mg.get_vocabulary_size(),
        mastered_words=mg.get_mastered_words_in(phrase),
        exploration_word=emerging,
    )
    cortex._record_action_trace(
        phrase, result.text, node_id=None,
        action_type="blended_mimicry", node_ids=result.used_node_ids,
    )
    cortex.apply_feedback(0.9, timestamp=1000.0)

    assert "кола" in {w.text for w in mg.get_mastered_words_in(phrase)}, (
        "похвала за попытку должна была довести слово до освоенности"
    )
