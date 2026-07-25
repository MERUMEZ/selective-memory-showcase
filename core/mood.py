"""
================================================================================
 MOOD.PY — Вектор эмоционального состояния (Mood Vector & Emotion Decay)
================================================================================
Модуль реализует ИНЕРЦИОННОЕ эмоциональное состояние системы — в отличие от
Amygdala (core/amygdala.py), которая считает эмоциональный отклик НА ОДНУ
конкретную реплику "здесь и сейчас", Mood хранит НАКОПЛЕННОЕ настроение,
которое:
    1. Меняется скачком при новом стимуле/фидбеке (apply_stimulus).
    2. Плавно затухает к базовому состоянию покоя на каждом тике (decay).

Четыре оси настроения (диапазон [0.0..1.0]):
    joy        — радость / восторг
    curiosity  — любопытство / вовлечённость
    anxiety    — тревожность / испуг
    affection  — привязанность / теплота

Baseline (состояние покоя):
    joy=0.1, curiosity=0.5, anxiety=0.1, affection=0.1

Формулы:
    Mood_new   = Clamp(Mood_current + Δstimulus, 0.0, 1.0)
    Mood(t+1)  = Mood(t) - gamma * (Mood(t) - Baseline)     [decay, gamma≈0.15-0.20]

Интеграция:
    - core/cortex.py подмешивает MoodState.describe_for_prompt() в системный
      промпт перед генерацией ответа (тон речи зависит от доминирующей эмоции).
    - main.py вызывает Mood.apply_stimulus(...) после Amygdala/Perception и
      Mood.decay() на каждом тике цикла, логируя итоговый снимок в консоль.
================================================================================
"""

from dataclasses import dataclass
from typing import Optional

import config
from storage.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Настройки по умолчанию (можно переопределить в config.py через
# MOOD_DECAY_RATE и т.п., если константы там определены — иначе дефолты).
# --------------------------------------------------------------------------
_DEFAULT_DECAY_RATE = getattr(config, "MOOD_DECAY_RATE", 0.18)

BASELINE_JOY = getattr(config, "MOOD_BASELINE_JOY", 0.1)
BASELINE_CURIOSITY = getattr(config, "MOOD_BASELINE_CURIOSITY", 0.5)
BASELINE_ANXIETY = getattr(config, "MOOD_BASELINE_ANXIETY", 0.1)
BASELINE_AFFECTION = getattr(config, "MOOD_BASELINE_AFFECTION", 0.1)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class MoodDelta:
    """Стимул (Δ), применяемый к текущему вектору настроения."""
    joy: float = 0.0
    curiosity: float = 0.0
    anxiety: float = 0.0
    affection: float = 0.0


@dataclass
class MoodState:
    """Иммутабельный снимок вектора настроения в конкретный момент."""
    joy: float
    curiosity: float
    anxiety: float
    affection: float
    decay_applied: bool = False

    def dominant_emotion(self) -> str:
        """
        Возвращает ось с наибольшим ПОЗИТИВНЫМ отклонением от Baseline
        (а не просто максимальное абсолютное значение) — так curiosity=0.5
        "по умолчанию" не будет ложно доминировать над реальным всплеском
        joy/anxiety/affection.
        """
        deviations = {
            "joy": self.joy - BASELINE_JOY,
            "curiosity": self.curiosity - BASELINE_CURIOSITY,
            "anxiety": self.anxiety - BASELINE_ANXIETY,
            "affection": self.affection - BASELINE_AFFECTION,
        }
        dominant = max(deviations, key=lambda k: deviations[k])
        if deviations[dominant] <= 0.02:
            return "neutral"
        return dominant

    def describe_for_prompt(self) -> str:
        """
        Формирует блок [MOOD VECTOR] для подмешивания в системный промпт
        Cortex — задаёт тон речи в зависимости от доминирующей эмоции.
        """
        dominant = self.dominant_emotion()

        tone_hints = {
            "joy": "Тебе радостно и весело — отвечай с энтузиазмом, восторгом, живой реакцией.",
            "curiosity": "Тебе очень любопытно — активно задавай уточняющие вопросы, проявляй вовлечённость.",
            "anxiety": "Тебе немного тревожно/страшно — отвечай осторожно, неуверенно, ищи поддержки у наставника.",
            "affection": "Ты чувствуешь тепло и привязанность к наставнику — отвечай мягко, доверительно, ласково.",
            "neutral": "Твоё эмоциональное состояние спокойное, нейтральное — отвечай ровно и естественно.",
        }
        tone_instruction = tone_hints.get(dominant, tone_hints["neutral"])

        return (
            "[MOOD VECTOR]\n"
            f"joy={self.joy:.2f} | curiosity={self.curiosity:.2f} | "
            f"anxiety={self.anxiety:.2f} | affection={self.affection:.2f}\n"
            f"Доминирующая эмоция: {dominant}.\n"
            f"{tone_instruction}"
        )

    def to_log_string(self) -> str:
        suffix = " (decay applied)" if self.decay_applied else ""
        return (
            f"[MOOD] joy: {self.joy:.2f} | anxiety: {self.anxiety:.2f} | "
            f"curiosity: {self.curiosity:.2f} | affection: {self.affection:.2f}{suffix}"
        )


class Mood:
    """
    Вектор эмоционального состояния с инерцией (Mood Vector & Decay).

    Использование:
        mood = Mood()

        # После стимула/фидбека:
        state = mood.apply_stimulus(MoodDelta(joy=0.3, affection=0.1))

        # На каждом тике/ходе (затухание к Baseline):
        state = mood.decay()

        # Перед генерацией ответа (Cortex):
        prompt_block = mood.get_state().describe_for_prompt()
    """

    def __init__(self, decay_rate: Optional[float] = None):
        self.decay_rate = decay_rate if decay_rate is not None else _DEFAULT_DECAY_RATE

        self.joy = BASELINE_JOY
        self.curiosity = BASELINE_CURIOSITY
        self.anxiety = BASELINE_ANXIETY
        self.affection = BASELINE_AFFECTION

    # ----------------------------------------------------------------------
    # Обновление вектора при стимуле
    # ----------------------------------------------------------------------

    def apply_stimulus(self, delta: MoodDelta, log: bool = True) -> MoodState:
        """
        Mood_new = Clamp(Mood_current + Δstimulus, 0.0, 1.0)
        """
        self.joy = _clamp(self.joy + delta.joy)
        self.curiosity = _clamp(self.curiosity + delta.curiosity)
        self.anxiety = _clamp(self.anxiety + delta.anxiety)
        self.affection = _clamp(self.affection + delta.affection)

        state = self.get_state(decay_applied=False)

        logger.info(
            "[MOOD STIMULUS] delta=%s -> joy=%.2f anxiety=%.2f curiosity=%.2f affection=%.2f",
            delta, self.joy, self.anxiety, self.curiosity, self.affection,
        )
        if log:
            print(state.to_log_string())

        return state

    def apply_feedback(
        self,
        feedback_valence: float = 0.0,
        emotion_score: float = 0.0,
        perplexity: float = 0.0,
        log: bool = True,
    ) -> MoodState:
        """
        Удобный хелпер, транслирующий сигналы, уже используемые в
        существующем пайплайне (Amygdala.detect_feedback_valence,
        Perception.emotion_score, Cortex.calculate_perplexity), в
        MoodDelta и применяющий его к вектору настроения.

        Эвристика:
            - perplexity (новизна/неожиданность) -> поднимает curiosity.
            - позитивная valence (одобрение наставника) -> поднимает joy
              и affection.
            - негативная valence (порицание) / высокий emotion_score без
              позитивной valence -> поднимает anxiety.
        """
        positive = max(0.0, feedback_valence)
        negative = max(0.0, -feedback_valence)

        delta = MoodDelta(
            joy=positive * 0.4,
            curiosity=perplexity * 0.3,
            anxiety=negative * 0.35 + (emotion_score * 0.1 if feedback_valence <= 0 else 0.0),
            affection=positive * 0.2,
        )
        return self.apply_stimulus(delta, log=log)

    # ----------------------------------------------------------------------
    # Затухание к Baseline (вызывается на каждом тике)
    # ----------------------------------------------------------------------

    def decay(self, log: bool = True) -> MoodState:
        """
        Mood(t+1) = Mood(t) - gamma * (Mood(t) - Baseline)
        """
        self.joy = _clamp(self.joy - self.decay_rate * (self.joy - BASELINE_JOY))
        self.curiosity = _clamp(self.curiosity - self.decay_rate * (self.curiosity - BASELINE_CURIOSITY))
        self.anxiety = _clamp(self.anxiety - self.decay_rate * (self.anxiety - BASELINE_ANXIETY))
        self.affection = _clamp(self.affection - self.decay_rate * (self.affection - BASELINE_AFFECTION))

        state = self.get_state(decay_applied=True)

        logger.debug(
            "[MOOD DECAY] gamma=%.2f -> joy=%.2f anxiety=%.2f curiosity=%.2f affection=%.2f",
            self.decay_rate, self.joy, self.anxiety, self.curiosity, self.affection,
        )
        if log:
            print(state.to_log_string())

        return state

    # ----------------------------------------------------------------------
    # Текущий снимок
    # ----------------------------------------------------------------------

    def get_state(self, decay_applied: bool = False) -> MoodState:
        return MoodState(
            joy=self.joy,
            curiosity=self.curiosity,
            anxiety=self.anxiety,
            affection=self.affection,
            decay_applied=decay_applied,
        )