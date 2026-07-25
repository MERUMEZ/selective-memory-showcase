"""
================================================================================
 PERCEPTION.PY — Слой восприятия входящего текста
================================================================================
Отвечает за первичный анализ "сырого" сигнала (текста пользователя) и расчёт
информационной/эмоциональной плотности сигнала — emotion_score (0.0..1.0).

Учитываемые признаки:
    - доля CAPS LOCK (крик)
    - количество восклицательных знаков
    - количество вопросительных знаков
    - количество эмодзи
    - повторяющиеся символы ("нееееет", "чтоооооо")
    - средняя длина слова (длинные/редкие слова -> выше информативность)
    - доля редких (не-алфавитных, не-стандартных) символов

Итоговый emotion_score — взвешенная сумма нормализованных признаков,
коэффициенты берутся из config.py (EMOTION_*_WEIGHT).
================================================================================
"""

import re
from dataclasses import dataclass, field
from typing import Dict

import config
from storage.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Регулярные выражения для признаков
# --------------------------------------------------------------------------

# Базовый диапазон эмодзи (не претендует на 100% полноту, но покрывает
# большинство распространённых эмодзи-блоков Unicode)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # символы, эмодзи, доп. пиктограммы
    "\U00002600-\U000027BF"  # разное + дингбаты
    "\U0001F1E6-\U0001F1FF"  # флаги
    "]",
    flags=re.UNICODE,
)

REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")  # 3+ повторяющихся символа подряд
WORD_PATTERN = re.compile(r"[^\s\d\W]+", flags=re.UNICODE)  # "слова" (буквы)
RARE_CHAR_PATTERN = re.compile(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]")  # не буквы/цифры/пробел


@dataclass
class PerceptionResult:
    """Результат анализа входящего текста."""
    emotion_score: float
    features: Dict[str, float] = field(default_factory=dict)


class Perception:
    """
    Анализирует входящий текст и вычисляет emotion_score — меру
    "информационно-эмоциональной плотности" сигнала.

    Использование:
        perception = Perception()
        result = perception.analyze("ЧТО ТЫ ТАКОЕ????!!! 😱😱😱")
        print(result.emotion_score)  # ближе к 1.0
    """

    def analyze(self, text: str) -> PerceptionResult:
        if not text or not text.strip():
            return PerceptionResult(emotion_score=0.0, features={})

        caps_ratio = self._caps_ratio(text)
        exclamation_score = self._exclamation_score(text)
        question_score = self._question_score(text)
        emoji_score = self._emoji_score(text)
        repetition_score = self._repetition_score(text)
        rare_char_ratio = self._rare_char_ratio(text)
        avg_word_len_score = self._avg_word_length_score(text)

        # Взвешенная сумма по коэффициентам из config.py
        raw_score = (
            caps_ratio * config.EMOTION_CAPS_WEIGHT
            + exclamation_score * config.EMOTION_EXCLAMATION_WEIGHT
            + question_score * config.EMOTION_QUESTION_WEIGHT
            + repetition_score * config.EMOTION_REPETITION_WEIGHT
            + emoji_score * config.EMOTION_EMOJI_WEIGHT
            + rare_char_ratio * 0.1          # небольшой доп. вклад "редких" символов
            + avg_word_len_score * 0.1        # небольшой доп. вклад длины слов
        )

        # Нормализуем итог в диапазон [0.0, 1.0]
        emotion_score = max(0.0, min(1.0, raw_score))

        features = {
            "caps_ratio": caps_ratio,
            "exclamation_score": exclamation_score,
            "question_score": question_score,
            "emoji_score": emoji_score,
            "repetition_score": repetition_score,
            "rare_char_ratio": rare_char_ratio,
            "avg_word_len_score": avg_word_len_score,
        }

        logger.debug("[PERCEPTION] emotion_score=%.3f features=%s", emotion_score, features)

        return PerceptionResult(emotion_score=emotion_score, features=features)

    # ----------------------------------------------------------------------
    # Отдельные признаки
    # ----------------------------------------------------------------------

    @staticmethod
    def _caps_ratio(text: str) -> float:
        """Доля заглавных букв среди всех буквенных символов."""
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        caps = sum(1 for c in letters if c.isupper())
        return caps / len(letters)

    @staticmethod
    def _exclamation_score(text: str) -> float:
        """Нормализованное количество восклицательных знаков (насыщение к 5+)."""
        count = text.count("!")
        return min(1.0, count / 5.0)

    @staticmethod
    def _question_score(text: str) -> float:
        """Нормализованное количество вопросительных знаков (насыщение к 5+)."""
        count = text.count("?")
        return min(1.0, count / 5.0)

    @staticmethod
    def _emoji_score(text: str) -> float:
        """Нормализованное количество эмодзи (насыщение к 5+)."""
        count = len(EMOJI_PATTERN.findall(text))
        return min(1.0, count / 5.0)

    @staticmethod
    def _repetition_score(text: str) -> float:
        """Нормализованное количество групп повторяющихся символов (насыщение к 3+)."""
        matches = REPEATED_CHAR_PATTERN.findall(text)
        return min(1.0, len(matches) / 3.0)

    @staticmethod
    def _rare_char_ratio(text: str) -> float:
        """Доля 'редких' (не алфавитно-цифровых) символов в тексте."""
        if not text:
            return 0.0
        rare = len(RARE_CHAR_PATTERN.findall(text))
        return min(1.0, rare / len(text))

    @staticmethod
    def _avg_word_length_score(text: str) -> float:
        """
        Нормализованная средняя длина слова. Длинные слова (редкие/сложные
        термины) считаем более информативными. Насыщение на длине слова 10+.
        """
        words = WORD_PATTERN.findall(text)
        if not words:
            return 0.0
        avg_len = sum(len(w) for w in words) / len(words)
        return min(1.0, avg_len / 10.0)