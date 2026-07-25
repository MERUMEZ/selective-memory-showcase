"""
================================================================================
 AMYGDALA.PY — Эмоциональный "фильтр" системы (Spike Detection + Feedback)
================================================================================
Класс Amygdala принимает emotion_score (из core/perception.py) и perplexity
(из core/cortex.py), считает итоговую информационную плотность сигнала и
решает, превышен ли динамический порог пластичности.

total_density = (emotion_score * 0.5) + (perplexity * 0.5)

Также Amygdala отвечает за детекцию ВАЛЕНТНОСТИ ОБРАТНОЙ СВЯЗИ (Feedback
Valence) — сканирует новую реплику пользователя на маркеры одобрения или
порицания предыдущего ответа бота. Это ядро механизма Подсознательного
Подкрепления (Reinforcement Loop), см. core/cortex.py:apply_feedback().

RETROSPECTIVE CORRECTION: valence взвешивается по marker_trust — динамически
изменяемому "доверию" к каждому конкретному маркеру. Если Cortex
впоследствии обнаруживает, что фидбэк, построенный на каком-то маркере,
оказался ЛОЖНЫМ (пользователь сам себя опроверг — сарказм, ошибочная
похвала/порицание), доверие к этому маркеру понижается (penalize_markers),
и его вклад в будущие оценки valence ослабевает. Подтверждённые маркеры,
наоборот, медленно восстанавливают доверие (recover_markers).

Если порог превышен -> is_spike_triggered = True -> [SPIKE DETECTED]
Иначе -> is_spike_triggered = False -> [ROUTINE]
================================================================================
"""

import re
from dataclasses import dataclass, field
from typing import List

import config
from storage.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AmygdalaResult:
    """Результат оценки сигнала амигдалой."""
    emotion_score: float
    perplexity: float
    total_density: float
    threshold_used: float
    is_spike_triggered: bool


@dataclass
class FeedbackSignal:
    """
    Детальный результат детекции валентности обратной связи — в отличие от
    простого float, хранит СПИСОК конкретных сработавших маркеров. Это
    необходимо для Retrospective Correction (core/cortex.py): если позже
    выяснится, что фидбэк был ложным (сарказм/самокоррекция пользователя),
    штраф на "доверие" (marker_trust) нужно применить именно к этим
    конкретным маркерам, а не к валентности в целом.
    """
    valence: float                  # итоговая валентность с учётом marker_trust, [-1.0, 1.0]
    raw_valence: float              # валентность без учёта marker_trust (для логов/отладки)
    matched_markers: List[str] = field(default_factory=list)

    def is_neutral(self) -> bool:
        return self.valence == 0.0 and not self.matched_markers


# --------------------------------------------------------------------------
# Маркеры обратной связи (Feedback Valence Detection)
# --------------------------------------------------------------------------
# Каждому маркеру сопоставлена валентность в диапазоне [-1.0, 1.0].
# Ищем маркеры как отдельные слова/короткие фразы (без учёта регистра).

POSITIVE_MARKERS = {
    "да": 0.5,
    "ага": 0.5,
    "хорошо": 0.6,
    "хорошая": 0.6,
    "правильно": 0.8,
    "точно": 0.7,
    "ого": 0.6,
    "неплохо": 0.55,
    "именно": 0.75,
    "супер": 1.0,
    "круто": 0.9,
    "отлично": 0.9,
    "класс": 0.8,
    "молодец": 0.85,
    "спасибо": 0.6,
}

NEGATIVE_MARKERS = {
    "нет": -0.5,
    "неа": -0.5,
    "плохо": -0.7,
    "плохой": -0.7,
    "неправильно": -0.85,
    "не так": -0.8,
    "фигня": -0.9,
    "бред": -0.9,
    "ошибка": -0.75,
    "не повторяй": -0.8,
    "не то": -0.7,
    "ерунда": -0.85,
    "тупо": -0.9,
}

# Сортируем многословные маркеры (типа "не так", "не повторяй") перед
# однословными, чтобы regex сначала пытался сматчить более длинные фразы.
_ALL_MARKERS = sorted(
    {**POSITIVE_MARKERS, **NEGATIVE_MARKERS}.items(),
    key=lambda kv: len(kv[0]),
    reverse=True,
)
_MARKER_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(marker) for marker, _ in _ALL_MARKERS) + r")(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_VALENCE_LOOKUP = {marker: valence for marker, valence in _ALL_MARKERS}


class Amygdala:
    """
    "Эмоциональный страж" системы: spike detection + feedback valence.

    Использование:
        amygdala = Amygdala()
        result = amygdala.evaluate(emotion_score=0.82, perplexity=0.4)

        signal = amygdala.detect_feedback_signal("да, именно так!")
        # signal.valence > 0 -> позитивное подкрепление предыдущего ответа
    """

    def __init__(self, base_threshold: float = None):
        self.base_threshold = (
            base_threshold if base_threshold is not None else config.BASE_PLASTICITY_THRESHOLD
        )

        # Retrospective Correction: "доверие" к каждому отдельному маркеру
        # одобрения/порицания. Изначально все маркеры полностью надёжны
        # (1.0). Если Cortex впоследствии обнаруживает, что маркер привёл
        # к ЛОЖНОМУ подкреплению (пользователь сам себя опроверг) —
        # доверие к нему понижается (penalize_markers), и его вклад в
        # будущие valence-расчёты ослабевает. При подтверждённых
        # срабатываниях доверие медленно восстанавливается (recover_markers).
        self.marker_trust = {marker: 1.0 for marker in _VALENCE_LOOKUP}

    def evaluate(
        self,
        emotion_score: float,
        perplexity: float = 0.0,
        stress_level: float = 0.0,
    ) -> AmygdalaResult:
        """
        Сравнивает итоговую информационную плотность сигнала с динамическим
        порогом пластичности и возвращает AmygdalaResult с флагом
        is_spike_triggered.
        """
        total_density = (emotion_score * 0.5) + (perplexity * 0.5)
        effective_threshold = self._compute_effective_threshold(stress_level)
        is_spike_triggered = total_density >= effective_threshold

        if is_spike_triggered:
            logger.info(
                "[SPIKE DETECTED] total_density=%.3f (emotion=%.3f, perplexity=%.3f) >= threshold=%.3f",
                total_density, emotion_score, perplexity, effective_threshold,
            )
        else:
            logger.info(
                "[ROUTINE] total_density=%.3f (emotion=%.3f, perplexity=%.3f) < threshold=%.3f",
                total_density, emotion_score, perplexity, effective_threshold,
            )

        return AmygdalaResult(
            emotion_score=emotion_score,
            perplexity=perplexity,
            total_density=total_density,
            threshold_used=effective_threshold,
            is_spike_triggered=is_spike_triggered,
        )

    def _compute_effective_threshold(self, stress_level: float) -> float:
        stress_level = max(0.0, min(1.0, stress_level))
        modifier = stress_level * config.PLASTICITY_STRESS_MODIFIER
        effective_threshold = min(1.0, self.base_threshold + modifier)
        return effective_threshold

    # ----------------------------------------------------------------------
    # Feedback Valence Detection (Подсознательное Подкрепление)
    # ----------------------------------------------------------------------

    def detect_feedback_valence(self, text: str) -> float:
        """
        Обёртка над detect_feedback_signal для обратной совместимости —
        возвращает только итоговую валентность без списка маркеров.
        """
        return self.detect_feedback_signal(text).valence

    def detect_feedback_signal(self, text: str) -> FeedbackSignal:
        """
        Сканирует текст на маркеры одобрения/порицания и возвращает
        FeedbackSignal — валентность (взвешенную по текущему marker_trust
        каждого маркера) плюс список конкретных сработавших маркеров.

        Взвешивание по marker_trust — ядро Retrospective Correction: если
        маркер ранее приводил к подтверждённым ложным срабатываниям
        (сарказм/самокоррекция пользователя), его вклад в valence
        ослабляется пропорционально его текущему "доверию", без полного
        исключения из анализа.
        """
        if not text or not text.strip():
            return FeedbackSignal(valence=0.0, raw_valence=0.0, matched_markers=[])

        normalized = text.strip().lower()
        matches = _MARKER_PATTERN.findall(normalized)

        if not matches:
            return FeedbackSignal(valence=0.0, raw_valence=0.0, matched_markers=[])

        normalized_matches = [m.lower() for m in matches]

        raw_valence = sum(_VALENCE_LOOKUP.get(m, 0.0) for m in normalized_matches)
        raw_valence = max(-1.0, min(1.0, raw_valence))

        trusted_valence = sum(
            _VALENCE_LOOKUP.get(m, 0.0) * self.marker_trust.get(m, 1.0)
            for m in normalized_matches
        )
        trusted_valence = max(-1.0, min(1.0, trusted_valence))

        logger.debug(
            "[FEEDBACK DETECTED] text=%r markers=%s -> raw_valence=%.2f trusted_valence=%.2f",
            text[:50], normalized_matches, raw_valence, trusted_valence,
        )

        return FeedbackSignal(
            valence=trusted_valence,
            raw_valence=raw_valence,
            matched_markers=normalized_matches,
        )

    # ----------------------------------------------------------------------
    # Retrospective Correction: управление доверием к маркерам
    # ----------------------------------------------------------------------

    def penalize_markers(self, markers: List[str], step: float = None) -> None:
        """
        Понижает "доверие" (marker_trust) к указанным маркерам — вызывается
        Cortex, когда Retrospective Correction обнаруживает, что фидбэк,
        построенный на этих маркерах, оказался ЛОЖНЫМ (пользователь сам
        себя опроверг спустя одну-две реплики — сарказм, ошибочная похвала
        или порицание). Доверие не опускается ниже config.MARKER_TRUST_MIN —
        маркер по-прежнему что-то значит, просто гораздо слабее.
        """
        step = step if step is not None else config.MARKER_TRUST_PENALTY_STEP
        for raw_marker in markers:
            marker = raw_marker.lower()
            if marker not in self.marker_trust:
                continue
            old_trust = self.marker_trust[marker]
            new_trust = max(config.MARKER_TRUST_MIN, old_trust - step)
            self.marker_trust[marker] = new_trust
            logger.info(
                "[MARKER TRUST PENALIZED] marker=%r trust %.3f -> %.3f",
                marker, old_trust, new_trust,
            )

    def recover_markers(self, markers: List[str], step: float = None) -> None:
        """
        Медленно восстанавливает "доверие" к указанным маркерам — вызывается,
        когда маркер сработал и его вклад в valence НЕ был впоследствии
        опровергнут (Retrospective Correction ничего не откатила). Это
        постепенная "реабилитация" маркера, если раз за разом он оказывается
        достоверным. Доверие не поднимается выше 1.0 (полное доверие).
        """
        step = step if step is not None else config.MARKER_TRUST_RECOVERY_STEP
        for raw_marker in markers:
            marker = raw_marker.lower()
            if marker not in self.marker_trust:
                continue
            old_trust = self.marker_trust[marker]
            new_trust = min(1.0, old_trust + step)
            self.marker_trust[marker] = new_trust
            logger.debug(
                "[MARKER TRUST RECOVERED] marker=%r trust %.3f -> %.3f",
                marker, old_trust, new_trust,
            )

    def get_marker_trust(self, marker: str) -> float:
        """Возвращает текущее доверие к конкретному маркеру (1.0, если маркер неизвестен)."""
        return self.marker_trust.get(marker.lower(), 1.0)