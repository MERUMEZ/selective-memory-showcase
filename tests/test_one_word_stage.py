"""
Тесты однословной стадии: организм произносит слова, которые РЕАЛЬНО
освоил, а незнакомое добирает лепетом.

Контекст: этой стадии в коде не существовало вовсе — он прыгал от чистого
лепета сразу к фразам от LLM. Наблюдение с живого бота: в мини-аппе
"привет" числилось освоенным (вес 0.747, сильнейшее слово мозга), а в
ответ на "привет" бот выдавал случайные слоги.

Причина: при словаре 7 слов babble_ratio выходил ровно 1.0, и ветка эха
отключалась целиком условием `if user_words and babble_ratio < 1.0`.
Знание слова влияло ТОЛЬКО на счётчик, разрешающий говорить, но не на то,
ЧТО будет сказано.
"""
import pytest

import config
from core.instincts import InstinctSystem
from selectivemem.database import Database
from selectivemem.graph_memory import KnownSyllable, MemoryGraph


@pytest.fixture
def mg():
    return MemoryGraph(db=Database(db_path=":memory:"))


@pytest.fixture
def instincts():
    return InstinctSystem()


def teach(graph: MemoryGraph, text: str, times: int) -> None:
    for i in range(times):
        graph.process_language_input(text, timestamp=float(i))


def syllables(graph: MemoryGraph):
    return graph.get_known_syllables()


# ---------------------------------------------------------------------------
# Какие слова организм считает своими
# ---------------------------------------------------------------------------
def test_only_mastered_words_are_recognised(mg):
    teach(mg, "привет", times=5)          # хорошо освоено
    mg.process_language_input("шкатулка", timestamp=99.0)  # услышано один раз

    known = {w.text for w in mg.get_mastered_words_in("привет шкатулка")}

    assert "привет" in known
    assert "шкатулка" not in known, (
        "слово, услышанное однажды, организм ещё не освоил и произносить не должен"
    )


def test_recognised_words_keep_input_order(mg):
    teach(mg, "мама мыла раму", times=5)

    words = [w.text for w in mg.get_mastered_words_in("раму мыла мама")]

    assert words == ["раму", "мыла", "мама"], "порядок должен быть как во входящей фразе"


def test_unknown_input_recognises_nothing(mg):
    teach(mg, "привет", times=5)
    assert mg.get_mastered_words_in("совершенно неизвестная фраза") == []


def test_recognition_survives_punctuation_and_case(mg):
    teach(mg, "привет", times=5)
    assert [w.text for w in mg.get_mastered_words_in("Привет!")] == ["привет"]


# ---------------------------------------------------------------------------
# Главное: своё слово попадает в ответ
# ---------------------------------------------------------------------------
def test_mastered_word_is_spoken_back(mg, instincts):
    """Ровно тот случай, который наблюдался на живом боте."""
    teach(mg, "привет", times=6)
    vocab = mg.get_vocabulary_size()

    # Условия того самого бага: словарь мал, лепет максимален
    assert instincts.get_babble_ratio(vocab) == pytest.approx(1.0)

    result = instincts.generate_blended_mimicry_response(
        "привет", syllables(mg), vocab,
        mastered_words=mg.get_mastered_words_in("привет"),
    )

    assert "привет" in result.text, (
        f"бот знает слово 'привет', но ответил {result.text!r} — "
        "выученный словарь снова не влияет на речь"
    )


def test_fully_understood_input_gets_no_babble(mg, instincts):
    """
    Лепет означает "мне есть что сказать, но нет слова". Если организм
    понял ВСЁ услышанное, договаривать нечего — ответ должен быть ровно
    из своих слов, без довеска.

    Регрессия на наблюдение: на "привет" бот отвечал "привет дорщена" и
    физически не мог сказать одно слово.
    """
    teach(mg, "привет", times=6)

    result = instincts.generate_blended_mimicry_response(
        "привет", syllables(mg), mg.get_vocabulary_size(),
        mastered_words=mg.get_mastered_words_in("привет"),
    )

    assert result.text.strip() == "привет", (
        f"всё услышанное узнано, но ответ {result.text!r} содержит лишнее"
    )


def test_partially_understood_input_babbles_the_rest(mg, instincts):
    """А вот если часть фразы не узнана — лепет уместен: мысль есть, слова нет."""
    # Слогов должно быть не меньше BABBLING_MIN_KNOWN_SYLLABLES, иначе
    # лепетать физически нечем и babble_ratio обнуляется
    teach(mg, "мама папа дом кот", times=6)
    assert len(syllables(mg)) >= config.BABBLING_MIN_KNOWN_SYLLABLES

    result = instincts.generate_blended_mimicry_response(
        "мама папа построили большое здание", syllables(mg), mg.get_vocabulary_size(),
        mastered_words=mg.get_mastered_words_in("мама папа построили большое здание"),
    )

    words = result.text.split()
    assert "мама" in words and "папа" in words
    assert len(words) > 2, "неузнанную часть фразы организм должен добрать лепетом"


def test_telegraphic_speech_uses_only_known_words(mg, instincts):
    teach(mg, "мама папа дом", times=6)
    vocab = mg.get_vocabulary_size()

    result = instincts.generate_blended_mimicry_response(
        "мама и папа построили дом вчера", syllables(mg), vocab,
        mastered_words=mg.get_mastered_words_in("мама и папа построили дом вчера"),
    )

    spoken = result.text.split()
    for word in ("мама", "папа", "дом"):
        assert word in spoken

    for unknown in ("построили", "вчера"):
        assert unknown not in spoken, (
            "незнакомые слова нельзя произносить — организм их не осваивал"
        )


def test_known_words_are_capped(mg, instincts):
    """Однословная стадия — не полноценная фраза: длина ограничена."""
    teach(mg, "один два три четыре пять шесть", times=6)
    vocab = mg.get_vocabulary_size()

    result = instincts.generate_blended_mimicry_response(
        "один два три четыре пять шесть", syllables(mg), vocab,
        mastered_words=mg.get_mastered_words_in("один два три четыре пять шесть"),
    )

    known_spoken = [w for w in result.text.split() if w in
                    {"один", "два", "три", "четыре", "пять", "шесть"}]
    assert len(known_spoken) <= config.MIMICRY_MAX_KNOWN_WORDS


def test_unknown_input_still_babbles(mg, instincts):
    """Если не узнано ничего — прежнее поведение, лепет."""
    teach(mg, "мама мыла раму", times=6)
    vocab = mg.get_vocabulary_size()

    result = instincts.generate_blended_mimicry_response(
        "квантовая хромодинамика", syllables(mg), vocab,
        mastered_words=mg.get_mastered_words_in("квантовая хромодинамика"),
    )

    assert result.text.strip(), "ответ не должен быть пустым"
    assert "квантовая" not in result.text, "незнакомое слово не должно произноситься"


# ---------------------------------------------------------------------------
# Подкрепление: удачно употреблённые СЛОВА тоже попадают в контур
# ---------------------------------------------------------------------------
def test_spoken_words_enter_the_reinforcement_loop(mg, instincts):
    """
    used_node_ids — то, что Cortex.apply_feedback усилит при похвале.
    Раньше туда попадали только слоги лепета, поэтому удачно употреблённое
    слово никак не закреплялось.
    """
    teach(mg, "привет", times=6)
    known = mg.get_mastered_words_in("привет")
    word_id = known[0].id

    result = instincts.generate_blended_mimicry_response(
        "привет", syllables(mg), mg.get_vocabulary_size(), mastered_words=known,
    )

    assert word_id in result.used_node_ids


# ---------------------------------------------------------------------------
# Обратная совместимость: без mastered_words всё работает как раньше
# ---------------------------------------------------------------------------
def test_without_mastered_words_falls_back_to_old_behaviour(mg, instincts):
    teach(mg, "мама мыла раму", times=6)

    result = instincts.generate_blended_mimicry_response(
        "мама мыла раму", syllables(mg), mg.get_vocabulary_size(),
    )

    assert result.text.strip(), "старый путь вызова не должен ломаться"


def test_empty_input_does_not_crash(mg, instincts):
    teach(mg, "привет", times=6)
    result = instincts.generate_blended_mimicry_response(
        "", syllables(mg), mg.get_vocabulary_size(),
        mastered_words=mg.get_mastered_words_in(""),
    )
    assert isinstance(result.text, str)
