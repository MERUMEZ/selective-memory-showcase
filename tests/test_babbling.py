"""
================================================================================
 TEST_BABBLING.PY — Лепет должен читаться как речь ребёнка, а не как сбой
================================================================================
Это тест на ДЕМОНСТРАЦИЮ, и он тут не для красоты. Лепет — первое, что
видит человек, открывший бота, и по нему он решает, работает ли вообще
эта штука. Прежняя версия склеивала независимо выбранные слоги и выдавала
"паквахлебда": формально верное поведение, которое выглядит как порча
текста.

Проверяется ровно то, что делает лепет узнаваемым:
  - слоги ПОВТОРЯЮТСЯ (канонический лепет, 6-10 месяцев);
  - разнообразие растёт со словарём (переход к вариативному лепету);
  - используются только реально услышанные слоги, а не случайные буквы.
================================================================================
"""

import random

import pytest

from core.instincts import InstinctSystem
from decaymem.graph_memory import KnownSyllable


@pytest.fixture
def syllables():
    return [
        KnownSyllable(id=1, text="ма", weight=0.5),
        KnownSyllable(id=2, text="ба", weight=0.5),
        KnownSyllable(id=3, text="дя", weight=0.5),
        KnownSyllable(id=4, text="то", weight=0.5),
    ]


def _repetition_rate(words):
    """Доля слов, состоящих из одного повторяющегося слога."""
    if not words:
        return 0.0
    uniform = sum(1 for w in words if len(set(w.split("-"))) == 1)
    return uniform / len(words)


def test_newborn_babble_is_reduplicated(syllables):
    """
    Пустой словарь -> чистое повторение: "ма-ма-ма". Это и есть тот
    признак, по которому лепет опознаётся с первого взгляда.
    """
    random.seed(1)
    instincts = InstinctSystem()
    words, _ = instincts._make_babble_words(syllables, n_words=20, vocabulary_size=0)

    assert _repetition_rate(words) == 1.0, f"ожидалось чистое повторение, вышло {words[:5]}"


def test_babble_becomes_variegated_with_vocabulary(syllables):
    """
    Чем больше услышано, тем разнообразнее лепет — как у детей. Если
    разнообразие не растёт, стадия развития не показана вовсе.
    """
    random.seed(1)
    instincts = InstinctSystem()

    early, _ = instincts._make_babble_words(syllables, n_words=40, vocabulary_size=0)
    late, _ = instincts._make_babble_words(syllables, n_words=40, vocabulary_size=25)

    assert _repetition_rate(late) < _repetition_rate(early)


def test_babble_uses_only_heard_syllables(syllables):
    """
    Лепет собирается из того, что организм РЕАЛЬНО слышал. Иначе это
    генератор случайных букв, и вся история про освоение языка — вымысел.
    """
    random.seed(1)
    instincts = InstinctSystem()
    known = {s.text for s in syllables}

    words, used_ids = instincts._make_babble_words(syllables, n_words=15, vocabulary_size=10)

    for word in words:
        for part in word.split("-"):
            assert part in known, f"{part!r} не из услышанного"
    assert used_ids, "узлы слогов должны попасть в подкрепление"


def test_babble_is_visually_distinct_from_typos(syllables):
    """
    Дефис — разметка, а не украшение: без него "мама" неотличимо от
    настоящего слова, а "дятто" — от опечатки.
    """
    random.seed(1)
    instincts = InstinctSystem()
    words, _ = instincts._make_babble_words(syllables, n_words=10, vocabulary_size=0)

    assert all("-" in w for w in words)
