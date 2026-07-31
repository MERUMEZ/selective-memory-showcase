"""
================================================================================
 INSTINCTS.PY — Слой инстинктов и самозащиты "Динамического Мозга"
================================================================================
Класс InstinctSystem — набор ИНСТИНКТИВНЫХ РЕАКЦИЙ, параметризованных
возбуждением. Собственного состояния нагрузки он больше не хранит: раньше
здесь жил current_stress, и то же самое состояние вторым экземпляром жило
как Mood.anxiety — одно измерялось дважды, а работал только один прибор.
Теперь единая ось возбуждения живёт в Mood (ядерный аффект = возбуждение +
валентность, разносить их по модулям было архитектурной несвязностью), а
сюда она приходит параметром.

Если возбуждение превышает STRESS_OVERLOAD_THRESHOLD, включается режим
самозащиты:
    - порог пластичности временно поднимается (PLASTICITY_STRESS_MODIFIER),
      система становится менее "впечатлительной" (см. core/amygdala.py)
    - Cortex предпочитает безопасный/эхолаличный (мимикрирующий) ответ
      вместо полноценной генерации

Возбуждение накапливается от того, что требует внимания (новизна входа и
величина ошибки предсказания награды — см. Mood.appraise) и затухает к
покою. Прежний источник — подсчёт капса и восклицательных знаков — был
плохим прокси нагрузки.

REINFORCEMENT LOOP (подсознательное подкрепление): _echolalia_bias — словарь
поправок к вероятности эхолалии для конкретных нормализованных контекстов
user_input. Позитивный фидбэк на успешную смысловую генерацию снижает
вероятность эхолалии для похожего ввода в будущем; негативный фидбэк на
эхолалию — повышает штраф (система "учится", что эхо здесь не годится).
================================================================================
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import config
from storage.utils.logger import get_logger

if TYPE_CHECKING:
    from memory.graph_memory import KnownSyllable, KnownWord

logger = get_logger(__name__)


@dataclass
class InstinctState:
    """Снимок реакций инстинктов при текущем возбуждении."""
    arousal: float
    is_overloaded: bool
    effective_plasticity_threshold: float


@dataclass
class BabbleResult:
    """
    Результат генерации лепета: итоговый текст + ID реально использованных
    syllable-узлов (в порядке первого появления, без дублей) — нужен для
    Reinforcement Loop (Cortex.apply_feedback), чтобы знать, ЧТО конкретно
    подкреплять/штрафовать по реакции пользователя.
    """
    text: str
    used_node_ids: List[int]


class InstinctSystem:
    """
    Отслеживает уровень стресса системы и включает защитные механизмы.

    Использование:
        instincts = InstinctSystem()
        state = instincts.get_state(arousal=mood.get_state().arousal)

        if state.is_overloaded:
            reply = instincts.generate_echolalia_response(user_text)
    """

    def __init__(self):
        # Reinforcement Loop: context_key -> bias (-1.0..+1.0), где
        # положительный bias СНИЖАЕТ вероятность эхолалии, отрицательный —
        # ПОВЫШАЕТ (штраф за неудачную эхолалию в этом контексте).
        self._echolalia_bias: Dict[str, float] = {}

    # ----------------------------------------------------------------------
    # Накопление / восстановление стресса
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Реакции на перегрузку (состояние живёт в Mood.arousal)
    # ----------------------------------------------------------------------

    @staticmethod
    def is_overloaded(arousal: float) -> bool:
        """
        Перегружен ли организм — включается режим самосохранения.

        Возбуждение приходит ПАРАМЕТРОМ: раньше InstinctSystem хранил
        собственный current_stress, и то же самое состояние жило вторым
        экземпляром как Mood.anxiety. Теперь ось одна и живёт в Mood, а
        инстинкты стали набором реакций, параметризованных ею.
        """
        overloaded = arousal > config.STRESS_OVERLOAD_THRESHOLD
        if overloaded:
            logger.warning(
                "[OVERLOAD] возбуждение=%.3f > порога=%.3f — режим самосохранения",
                arousal, config.STRESS_OVERLOAD_THRESHOLD,
            )
        return overloaded

    @staticmethod
    def get_state(arousal: float) -> InstinctState:
        """Снимок инстинктивного состояния при заданном возбуждении."""
        modifier = arousal * config.PLASTICITY_STRESS_MODIFIER
        return InstinctState(
            arousal=arousal,
            is_overloaded=arousal > config.STRESS_OVERLOAD_THRESHOLD,
            effective_plasticity_threshold=min(
                1.0, config.BASE_PLASTICITY_THRESHOLD + modifier
            ),
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

    def should_use_echolalia(
        self,
        confidence: float,
        context_key: Optional[str] = None,
        arousal: float = 0.0,
    ) -> bool:
        """
        Определяет, стоит ли использовать эхолалию, комбинируя:
            - вероятностный порог ECHOLALIA_PROBABILITY
            - текущую перегрузку возбуждением (is_overloaded)
            - низкую уверенность модели (confidence < CONFIDENCE_FALLBACK_THRESHOLD)
            - Reinforcement Loop bias для конкретного context_key (если передан)

        context_key — нормализованный user_input, используется для учёта
        накопленного bias из adjust_echolalia_bias().
        """
        import random

        if self.is_overloaded(arousal):
            return True

        bias = self._echolalia_bias.get(self._normalize_key(context_key), 0.0) if context_key else 0.0

        if confidence < config.CONFIDENCE_FALLBACK_THRESHOLD:
            effective_probability = max(0.0, min(1.0, config.ECHOLALIA_PROBABILITY - bias))
            return random.random() < effective_probability

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

    def get_babble_ratio(self, vocabulary_size: int, arousal: float = 0.0) -> float:
        """
        Continuous доля "лепетного" поведения в смешанном ответе, [0.0, 1.0]:
        - 1.0 при vocabulary_size около 0 (чистый довербальный лепет)
        - плавно убывает к 0.0 в полосе шириной MIMICRY_BLEND_WIDTH вокруг
          BABBLING_VOCABULARY_THRESHOLD (без резкого переключения)
        - перегрузка стрессом добавляет бонус — регрессия к более ранней
          стадии речи даже при большом словаре (как у настоящих детей).
        """
        threshold = config.BABBLING_VOCABULARY_THRESHOLD
        blend = max(1.0, config.MIMICRY_BLEND_WIDTH)
        base_ratio = (threshold + blend - vocabulary_size) / (2 * blend)
        base_ratio = max(0.0, min(1.0, base_ratio))
        if self.is_overloaded(arousal):
            base_ratio = min(1.0, base_ratio + config.STRESS_REGRESSION_BABBLE_BONUS)
        return base_ratio

    def _make_babble_words(
        self, known_syllables: List["KnownSyllable"], n_words: int
    ) -> Tuple[List[str], List[int]]:
        """Общая внутренняя логика генерации n_words лепетных слов из известных слогов."""
        import random
        weights = [max(0.01, s.weight) for s in known_syllables]
        words: List[str] = []
        used_ids: List[int] = []
        for _ in range(n_words):
            syllable_count = random.randint(
                config.BABBLING_MIN_SYLLABLES, config.BABBLING_MAX_SYLLABLES
            )
            chosen = random.choices(known_syllables, weights=weights, k=syllable_count)
            words.append("".join(c.text for c in chosen))
            used_ids.extend(c.id for c in chosen)
        return words, used_ids

    def generate_blended_mimicry_response(
        self,
        input_text: str,
        known_syllables: List["KnownSyllable"],
        vocabulary_size: int,
        arousal: float = 0.0,
        mastered_words: Optional[List["KnownWord"]] = None,
        exploration_word: Optional["KnownWord"] = None,
    ) -> BabbleResult:
        """
        Довербальная и однословная речь: организм произносит те слова,
        которые РЕАЛЬНО освоил, а всё незнакомое добирает лепетом.

        ОДНОСЛОВНАЯ СТАДИЯ. Раньше её не существовало: код прыгал от
        чистого лепета сразу к фразам от LLM. При словаре 7 слов
        babble_ratio выходил ровно 1.0, ветка эха отключалась целиком
        (`if user_words and babble_ratio < 1.0`), и на "привет" бот отвечал
        случайными слогами — хотя знал слово "привет" лучше всех прочих
        (вес 0.747). Знание слова управляло только счётчиком, решающим,
        разрешить ли говорить, но не тем, ЧТО сказать.

        Теперь порядок такой:
            1. mastered_words — слова входа, которые организм освоил
               (MemoryGraph.get_mastered_words_in). Это его собственные
               слова, и он их произносит: "привет" -> "привет".
            2. Незнакомое остаётся лепетом, доля которого по-прежнему
               убывает с ростом словаря и растёт при перегрузке стрессом.
            3. Если из входа не узнано НИЧЕГО — поведение как раньше:
               случайное эхо (при ratio < 1.0) плюс лепет.

        Это ровно то, что делает ребёнок на однословной стадии: говорит
        освоенными словами, остальное договаривает лепетом.
        """
        import random

        babble_ratio = self.get_babble_ratio(vocabulary_size, arousal)

        if len(known_syllables) < config.BABBLING_MIN_KNOWN_SYLLABLES:
            # Физически нечем лепетать -> остаётся только эхо-компонента
            babble_ratio = 0.0

        user_words = [w for w in input_text.strip().split() if w]
        mastered_words = mastered_words or []

        spoken_words: List[str] = []
        used_ids: List[int] = []

        if mastered_words:
            # Свои слова: берём сильнейшие, но произносим в порядке
            # исходной фразы — так реплика читается естественно
            # ("привет как дела", а не "дела привет как").
            limit = max(1, config.MIMICRY_MAX_KNOWN_WORDS)
            # Отбор по preference, а не по голому весу: при сопоставимой
            # освоенности организм предпочитает слова, за которые его
            # хвалили. Это и есть стремление к одобрению в действии —
            # раньше награда меняла веса задним числом, но на ВЫБОР
            # следующего действия не влияла никак.
            strongest = sorted(
                mastered_words,
                key=lambda w: (w.preference or w.weight),
                reverse=True,
            )[:limit]
            chosen_ids = {w.id for w in strongest}
            spoken_words = [w.text for w in mastered_words if w.id in chosen_ids]
            used_ids.extend(w.id for w in mastered_words if w.id in chosen_ids)
        elif user_words and babble_ratio < 1.0:
            # Ничего не узнано, но лепет не абсолютен -> прежнее поведение:
            # случайное эхо слов пользователя.
            max_echo = max(
                1, round(len(user_words) * (1.0 - babble_ratio) * config.BLENDED_ECHO_WORD_RATIO)
            )
            spoken_words = random.sample(user_words, k=min(max_echo, len(user_words)))

        # ПОПЫТКА: организм рискует произнести ещё не освоенное слово.
        # Решение принято выше (Cortex._pick_exploration_word) — там есть
        # доступ к любопытству. Слово встаёт в конец: сначала уверенное,
        # потом пробное, как ребёнок договаривает неуверенное вслед за
        # знакомым. Его узел уходит в used_node_ids, поэтому похвала
        # доберётся и до него — так замыкается контур освоения:
        # попробовал -> похвалили -> закрепилось.
        if exploration_word is not None:
            spoken_words.append(exploration_word.text)
            used_ids.append(exploration_word.id)

        # Лепет — это "мне есть что сказать, но нет слова". Поэтому он
        # добирает ТОЛЬКО неузнанную часть реплики: если организм понял
        # всё, что услышал, договаривать нечего и лепет неуместен.
        #
        # Раньше лепет приклеивался безусловно, и на "привет" бот отвечал
        # "привет дорщена" — не мог сказать одно слово, даже когда знал
        # ровно его. Ребёнок на однословной стадии отвечает "привет" и
        # замолкает; лепет появляется там, где мысль есть, а слова нет.
        # Опробованное слово тоже произнесено, поэтому лепету оно уже не
        # достаётся — иначе одна и та же дыра была бы закрыта дважды.
        covered = len(mastered_words) + (1 if exploration_word is not None else 0)
        unrecognised = max(0, len(user_words) - covered)

        babble_words: List[str] = []
        can_babble = (
            babble_ratio > 0.0
            and len(known_syllables) >= config.BABBLING_MIN_KNOWN_SYLLABLES
        )
        if can_babble and (unrecognised > 0 or not user_words):
            n_babble_words = max(
                1,
                round(min(unrecognised or config.BABBLING_WORDS_PER_RESPONSE,
                          config.BABBLING_WORDS_PER_RESPONSE) * babble_ratio),
            )
            babble_words, babble_ids = self._make_babble_words(known_syllables, n_babble_words)
            used_ids.extend(babble_ids)

        fragments = spoken_words + babble_words
        if not spoken_words:
            # Настоящих слов нет — порядок чистого лепета ничего не значит.
            # Но если хоть одно слово произнесено (своё или пробное), его
            # нельзя перемешивать с лепетом: сначала то, в чём организм
            # уверен, потом то, что он договаривает.
            random.shuffle(fragments)

        if not fragments:
            # Нечем ни лепетать, ни повторить (пустой вход и нет слогов)
            # -> откат на стандартную эхолалию как последний fallback.
            text = self.generate_echolalia_response(input_text)
        else:
            text = " ".join(fragments)

        logger.info(
            "[INSTINCT: BLENDED MIMICRY] ratio=%.2f свои_слова=%d лепет=%d -> %r",
            babble_ratio, len(spoken_words), len(babble_words), text[:60],
        )

        return BabbleResult(text=text, used_node_ids=list(dict.fromkeys(used_ids)))

    def generate_babble_response(self, known_syllables: List["KnownSyllable"]) -> BabbleResult:
        """
        Генерирует "лепетный" ответ — ВЗВЕШЕННУЮ (по весу/усвоенности узла)
        комбинацию уже известных слогов, имитируя довербальную стадию
        освоения речи ребёнком ("ма-ма-ба", "ту-ту-ка" и т.п.).

        В отличие от прежней реализации (чистый random.choice), более
        "усвоенные" слоги (выше weight — чаще встречались/чаще получали
        позитивный фидбэк раньше) выбираются ЧАЩЕ, но не исключительно —
        это даёт постепенный уклон к уже проверенным комбинациям без
        полной потери разнообразия.

        Возвращает BabbleResult(text="", used_node_ids=[]), если известных
        слогов недостаточно (< BABBLING_MIN_KNOWN_SYLLABLES) — вызывающий
        код (Cortex) в этом случае должен откатиться на обычную эхолалию.
        """
        import random

        if len(known_syllables) < config.BABBLING_MIN_KNOWN_SYLLABLES:
            logger.debug(
                "[INSTINCT: BABBLING] Недостаточно известных слогов (%d < %d) -> пропуск",
                len(known_syllables), config.BABBLING_MIN_KNOWN_SYLLABLES,
            )
            return BabbleResult(text="", used_node_ids=[])

        # Взвешенная выборка: минимум 0.01, чтобы совсем "свежие" слоги
        # (weight около 0) не получали нулевую вероятность быть выбранными —
        # разнообразие должно сохраняться даже для новых слогов.
        weights = [max(0.01, s.weight) for s in known_syllables]

        words: List[str] = []
        used_ids: List[int] = []

        for _ in range(config.BABBLING_WORDS_PER_RESPONSE):
            syllable_count = random.randint(
                config.BABBLING_MIN_SYLLABLES, config.BABBLING_MAX_SYLLABLES
            )
            chosen = random.choices(known_syllables, weights=weights, k=syllable_count)
            words.append("".join(c.text for c in chosen))
            used_ids.extend(c.id for c in chosen)

        response = " ".join(words)
        unique_used_ids = list(dict.fromkeys(used_ids))  # дедуп с сохранением порядка

        logger.info(
            "[INSTINCT: BABBLING] Лепет сгенерирован -> %r (node_ids=%s)",
            response[:60], unique_used_ids,
        )
        return BabbleResult(text=response, used_node_ids=unique_used_ids)

    # ----------------------------------------------------------------------

    @staticmethod
    def _normalize_key(text: str) -> str:
        return text.strip().lower()