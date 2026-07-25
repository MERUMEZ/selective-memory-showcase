"""
================================================================================
 INSTINCTS.PY — Слой инстинктов и самозащиты "Динамического Мозга"
================================================================================
Класс InstinctSystem отслеживает накопленный стресс системы (current_stress,
0.0..1.0). Если стресс превышает STRESS_OVERLOAD_THRESHOLD, включается режим
самозащиты:
    - порог пластичности временно поднимается (PLASTICITY_STRESS_MODIFIER),
      система становится менее "впечатлительной" (см. core/amygdala.py)
    - Cortex предпочитает безопасный/эхолаличный (мимикрирующий) ответ
      вместо полноценной генерации

Стресс накапливается от "тревожных" импульсов (например, высокий emotion_score,
неудачные попытки понять контекст) и восстанавливается со временем, если
система не подвергается новым стрессорам (естественное успокоение).

REINFORCEMENT LOOP (подсознательное подкрепление): _echolalia_bias — словарь
поправок к вероятности эхолалии для конкретных нормализованных контекстов
user_input. Позитивный фидбэк на успешную смысловую генерацию снижает
вероятность эхолалии для похожего ввода в будущем; негативный фидбэк на
эхолалию — повышает штраф (система "учится", что эхо здесь не годится).
================================================================================
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

import config
from typing import Dict, List, Optional
from storage.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InstinctState:
    """Снимок текущего состояния инстинктивной системы."""
    current_stress: float
    is_overloaded: bool
    effective_plasticity_threshold: float


class InstinctSystem:
    """
    Отслеживает уровень стресса системы и включает защитные механизмы.

    Использование:
        instincts = InstinctSystem()
        instincts.accumulate_stress(0.3)
        state = instincts.get_state()

        if state.is_overloaded:
            reply = instincts.generate_echolalia_response(user_text)
    """

    def __init__(self):
        self.current_stress: float = 0.0
        self._last_update: float = time.time()
        # Reinforcement Loop: context_key -> bias (-1.0..+1.0), где
        # положительный bias СНИЖАЕТ вероятность эхолалии, отрицательный —
        # ПОВЫШАЕТ (штраф за неудачную эхолалию в этом контексте).
        self._echolalia_bias: Dict[str, float] = {}

    # ----------------------------------------------------------------------
    # Накопление / восстановление стресса
    # ----------------------------------------------------------------------

    def accumulate_stress(self, intensity: float = 1.0) -> float:
        self._apply_recovery()

        delta = max(0.0, intensity) * config.STRESS_ACCUMULATION_RATE
        self.current_stress = min(1.0, self.current_stress + delta)

        logger.debug(
            "[STRESS ACCUMULATED] +%.3f -> current_stress=%.3f",
            delta, self.current_stress,
        )

        self._check_overload()
        return self.current_stress

    def _apply_recovery(self) -> None:
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        if dt <= 0 or self.current_stress <= 0.0:
            return

        recovered = dt * config.STRESS_RECOVERY_RATE
        old_stress = self.current_stress
        self.current_stress = max(0.0, self.current_stress - recovered)

        if old_stress != self.current_stress:
            logger.debug(
                "[STRESS RECOVERY] -%.3f (dt=%.1fs) -> current_stress=%.3f",
                old_stress - self.current_stress, dt, self.current_stress,
            )

    def recover_stress(self, amount: Optional[float] = None) -> float:
        step = amount if amount is not None else config.STRESS_RECOVERY_RATE
        old_stress = self.current_stress
        self.current_stress = max(0.0, self.current_stress - step)

        logger.debug(
            "[STRESS RECOVERY] manual -%.3f -> current_stress=%.3f",
            old_stress - self.current_stress, self.current_stress,
        )
        return self.current_stress

    # ----------------------------------------------------------------------
    # Проверка перегрузки / состояние
    # ----------------------------------------------------------------------

    def _check_overload(self) -> bool:
        is_overloaded = self.current_stress > config.STRESS_OVERLOAD_THRESHOLD
        if is_overloaded:
            logger.warning(
                "[STRESS OVERLOAD] current_stress=%.3f > threshold=%.3f — режим самозащиты активен",
                self.current_stress, config.STRESS_OVERLOAD_THRESHOLD,
            )
        return is_overloaded

    def is_overloaded(self) -> bool:
        self._apply_recovery()
        return self.current_stress > config.STRESS_OVERLOAD_THRESHOLD

    def get_effective_plasticity_threshold(self) -> float:
        self._apply_recovery()
        modifier = self.current_stress * config.PLASTICITY_STRESS_MODIFIER
        return min(1.0, config.BASE_PLASTICITY_THRESHOLD + modifier)

    def get_state(self) -> InstinctState:
        self._apply_recovery()
        return InstinctState(
            current_stress=self.current_stress,
            is_overloaded=self.current_stress > config.STRESS_OVERLOAD_THRESHOLD,
            effective_plasticity_threshold=self.get_effective_plasticity_threshold(),
        )

    # ----------------------------------------------------------------------
    # Эхолалия / мимикрия (защитная реакция)
    # ----------------------------------------------------------------------

    def generate_echolalia_response(self, input_text: str) -> str:
        cleaned = input_text.strip()

        if not cleaned:
            response = "..."
        elif cleaned.endswith("?"):
            response = f"{cleaned[:-1].strip()}?"
        else:
            response = cleaned

        logger.info("[INSTINCT: ECHOLALIA] Мимикрия сработала -> %r", response[:60])
        return response

    def should_use_echolalia(self, confidence: float, context_key: Optional[str] = None) -> bool:
        """
        Определяет, стоит ли использовать эхолалию, комбинируя:
            - вероятностный порог ECHOLALIA_PROBABILITY
            - текущую перегрузку стрессом (is_overloaded)
            - низкую уверенность модели (confidence < CONFIDENCE_FALLBACK_THRESHOLD)
            - Reinforcement Loop bias для конкретного context_key (если передан)

        context_key — нормализованный user_input, используется для учёта
        накопленного bias из adjust_echolalia_bias().
        """
        import random

        if self.is_overloaded():
            return True

        bias = self._echolalia_bias.get(self._normalize_key(context_key), 0.0) if context_key else 0.0

        if confidence < config.CONFIDENCE_FALLBACK_THRESHOLD:
            effective_probability = max(0.0, min(1.0, config.ECHOLALIA_PROBABILITY - bias))
            return random.random() < effective_probability or True

        # Даже при высокой уверенности сильный негативный bias (штраф за
        # неудачную эхолалию раньше) не должен внезапно включать эхолалию —
        # bias здесь работает только как понижающий фактор для низкой confidence.
        return False

    # ----------------------------------------------------------------------
    # Reinforcement Loop: динамический bias на эхолалию по контексту
    # ----------------------------------------------------------------------

    def adjust_echolalia_bias(self, context_key: str, delta: float) -> float:
        """
        Корректирует bias вероятности эхолалии для конкретного контекста.

            delta > 0  -> СНИЖАЕТ вероятность эхолалии для этого контекста
                          (например: была успешная смысловая генерация,
                          подтверждённая позитивным фидбэком).
            delta < 0  -> ПОВЫШАЕТ штраф на эхолалию (была неудачная
                          эхолалия, подтверждённая негативным фидбэком).

        Bias накапливается и ограничивается диапазоном [-1.0, 1.0].
        Возвращает новое значение bias.
        """
        key = self._normalize_key(context_key)
        current = self._echolalia_bias.get(key, 0.0)
        new_bias = max(-1.0, min(1.0, current + delta))
        self._echolalia_bias[key] = new_bias

        logger.info(
            "[REINFORCEMENT] Echolalia bias для %r: %.3f -> %.3f (delta=%.3f)",
            context_key[:40], current, new_bias, delta,
        )
        return new_bias

    # ----------------------------------------------------------------------
    # BABBLING — довербальная стадия освоения речи (лепет)
    # ----------------------------------------------------------------------

    def should_babble(self, vocabulary_size: int) -> bool:
        """
        Определяет, стоит ли использовать "лепет" вместо обычной эхолалии.

        Лепет возможен ТОЛЬКО на стадии малого словарного запаса
        (vocabulary_size < BABBLING_VOCABULARY_THRESHOLD) — как только
        система усвоила достаточно слов, лепет прекращается сам собой
        (естественный переход от довербальной стадии к речи).

        vocabulary_size — количество уникальных освоенных word-узлов
        (см. MemoryGraph.get_vocabulary_size()).
        """
        import random

        if vocabulary_size >= config.BABBLING_VOCABULARY_THRESHOLD:
            return False

        return random.random() < config.BABBLING_PROBABILITY

    def generate_babble_response(self, known_syllables: List[str]) -> str:
        """
        Генерирует "лепетный" ответ — случайную комбинацию УЖЕ ИЗВЕСТНЫХ
        слогов (взятых из графа памяти, см. MemoryGraph.get_known_syllables),
        имитируя довербальную стадию освоения речи ребёнком ("ма-ма-ба",
        "ту-ту-ка" и т.п.).

        Если известных слогов недостаточно (< BABBLING_MIN_KNOWN_SYLLABLES),
        возвращает пустую строку — вызывающий код (Cortex) в этом случае
        должен откатиться на обычную эхолалию.
        """
        import random

        if len(known_syllables) < config.BABBLING_MIN_KNOWN_SYLLABLES:
            logger.debug(
                "[INSTINCT: BABBLING] Недостаточно известных слогов (%d < %d) -> пропуск",
                len(known_syllables), config.BABBLING_MIN_KNOWN_SYLLABLES,
            )
            return ""

        words = []
        for _ in range(config.BABBLING_WORDS_PER_RESPONSE):
            syllable_count = random.randint(
                config.BABBLING_MIN_SYLLABLES, config.BABBLING_MAX_SYLLABLES
            )
            chosen = [random.choice(known_syllables) for _ in range(syllable_count)]
            words.append("".join(chosen))

        response = " ".join(words)
        logger.info("[INSTINCT: BABBLING] Лепет сгенерирован -> %r", response[:60])
        return response

    # ----------------------------------------------------------------------

    @staticmethod
    def _normalize_key(text: str) -> str:
        return text.strip().lower()