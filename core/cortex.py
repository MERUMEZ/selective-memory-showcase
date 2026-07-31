"""
================================================================================
 CORTEX.PY — Интерфейс генерации ответов "Динамического Мозга"
================================================================================
Класс Cortex — точка входа для работы с текстом на "высоком уровне":
    a) Рассчитывает условный perplexity (неожиданность текста) для Amygdala.
    b) Генерирует ответ, используя реальную LLM (OpenRouter, см. services/llm.py):
        - ВСЕГДА подмешивает срезовый контекст из WorkingMemory (STM) в
          системный промпт — бот непрерывно видит ход текущей беседы.
        - MEMORY HIT (LTM): дополнительно подмешивает build_prompt_context()
          с прошлым релевантным опытом. response_source = "memory_recall".
        - MEMORY MISS + высокая уверенность: обычный запрос к LLM со
          стандартным системным промптом + STM. response_source = "generation".
        - Низкая уверенность (рутина): эхолалия, LLM не вызывается.
        - Graceful Degradation: если LLM вернула None -> откат на эхолалию,
          лог [CORTEX FALLBACK].
    c) ПОДСОЗНАТЕЛЬНОЕ ПОДКРЕПЛЕНИЕ (Reinforcement / Feedback Loop):
        Триада Input -> Action -> Feedback:
            Input    = user_input предыдущего сообщения
            Action   = bot_output (ответ Cortex), зафиксированный в
                       last_action_trace вместе с node_id/action_type
            Feedback = valence нового user_input (см. Amygdala.detect_feedback_valence)
        apply_feedback() применяет найденную валентность к last_action_trace:
        подкрепляет/штрафует задействованный узел LTM и корректирует
        склонность к эхолалии для похожего контекста в будущем.
================================================================================
"""

import re
import config
from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional
from core.amygdala import Amygdala
from core.instincts import InstinctSystem
from core.mood import Appraisal, Mood, MoodState
from memory.graph_memory import MemoryGraph, MemoryMatch, ProactiveCandidate
from memory.working_memory import WorkingMemory
from services.llm import generate_llm_response
from storage.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Ты — цифровой организм с динамической, биологически инспирированной памятью. "
    "Отвечай кратко, естественно и по-человечески, без лишней воды. "
    "Если вопрос неясен — уточни его, а не выдумывай факты."
)


_CONCEPT_PATTERNS_COMPILED = [
    re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)
    for pattern in config.CONCEPT_PATTERNS
]


@dataclass
class CortexResponse:
    """Результат генерации ответа корой (Cortex)."""
    text: str
    confidence: float
    perplexity: float
    source: str  # "memory_recall" | "echolalia" | "babbling" | "blended_mimicry" | "generation"
    prompt_context: Optional[str] = None
    activation_traces: Optional[list] = None
    mood_state: Optional[MoodState] = None


@dataclass
class ConceptExtractionResult:
    """Результат попытки извлечения обучающего понятия из текста Юзера."""
    name: str
    definition: str


@dataclass
class ProactiveMessage:
    """Результат генерации проактивного сообщения (Boredom Drive Trigger)."""
    text: str
    source_node_id: Optional[int]
    prompt_context: str


@dataclass
class ActionTrace:
    """
    Последний след транзакции (Last Transaction Trace) — фиксирует
    связку Input -> Action для последующей оценки Feedback на СЛЕДУЮЩЕМ шаге.

    node_id   — единичный узел (используется memory_retrieval).
    node_ids  — список узлов (используется babbling — несколько
                syllable-узлов, задействованных в одной лепетной фразе).
                None для action_type, не оперирующих множеством узлов.
    """
    user_input: str
    bot_output: str
    node_id: Optional[int]
    action_type: str  # "echolalia" | "llm_generation" | "memory_retrieval" | "babbling" | "blended_mimicry"
    node_ids: Optional[List[int]] = None


@dataclass
class FeedbackResult:
    """Результат применения обратной связи к last_action_trace."""
    valence: float
    node_id: Optional[int]
    user_input: str
    bot_output: str
    action_type: str
    effect: str  # "rewarded" | "penalized" | "bias_adjusted" | "no_trace" | "neutral"
    mood_state: Optional[MoodState] = None
    retrospective_correction: Optional["RetrospectiveCorrectionResult"] = None
    node_ids: Optional[List[int]] = None


@dataclass
class FeedbackHistoryEntry:
    """
    Один применённый Feedback-эффект в истории Retrospective Correction.

    node_ids — для babbling-действий: список syllable-узлов, к которым
    был применён applied_delta (единый для всех, см. apply_feedback).
    Retrospective Correction (см. _check_retrospective_correction) пока
    обрабатывает только записи с одиночным node_id (memory_retrieval) —
    это осознанное упрощение первой версии, babbling-записи в отложенном
    опровержении участвовать не будут.
    """
    timestamp: Optional[float]
    valence: float
    matched_markers: List[str]
    node_id: Optional[int]
    user_input: str
    bot_output: str
    action_type: str
    applied_delta: float = 0.0
    reversed: bool = False
    node_ids: Optional[List[int]] = None


@dataclass
class RetrospectiveCorrectionResult:
    """Результат проверки на отложенное опровержение прошлого фидбэка."""
    triggered: bool
    reversed_entry: Optional[FeedbackHistoryEntry] = None
    reversal_delta: float = 0.0
    penalized_markers: List[str] = field(default_factory=list)


class Cortex:
    """
    Высокоуровневый интерфейс генерации ответов.

    Использование:
        cortex = Cortex(instincts=instinct_system, memory=memory_graph, stm=working_memory)
        response = cortex.generate_response(user_text)
        feedback = cortex.apply_feedback(valence)
    """

    def __init__(
        self,
        instincts: Optional[InstinctSystem] = None,
        memory: Optional[MemoryGraph] = None,
        stm: Optional[WorkingMemory] = None,
        mood: Optional[Mood] = None,
        amygdala: Optional[Amygdala] = None,
    ):
        self.instincts = instincts or InstinctSystem()
        self.memory = memory or MemoryGraph()
        self.stm = stm or WorkingMemory()
        self.mood = mood or Mood()
        # Retrospective Correction нуждается в ТОЙ ЖЕ инстанции Amygdala,
        # что и main.py (детекция маркеров) — чтобы penalize_markers/
        # recover_markers реально влияли на marker_trust, используемый
        # при следующих вызовах detect_feedback_signal.
        self.amygdala = amygdala or Amygdala()
        self.last_action_trace: Optional[ActionTrace] = None

        # Короткая история последних N связок Input->Action->Feedback —
        # окно, в пределах которого более позднее противоречащее сообщение
        # трактуется как отложенное опровержение прошлой оценки
        # (Retrospective Correction), а не независимый новый фидбэк.
        self.feedback_history: "deque[FeedbackHistoryEntry]" = deque(
            maxlen=config.RETROSPECTIVE_WINDOW_SIZE
        )

    # ----------------------------------------------------------------------
    # a) Perplexity
    # ----------------------------------------------------------------------

    def calculate_perplexity(self, text: str) -> float:
        """
        Насколько входящий текст НЕОЖИДАНЕН для этого организма.

        Это ошибка предсказания по СОБСТВЕННОМУ графу языка, а не свойство
        строки: см. MemoryGraph.compute_surprise. Прежняя реализация считала
        энтропию Шеннона по символам и была слепа к опыту — пустой мозг и
        мозг после 50 повторений фразы выдавали одно и то же число, из-за
        чего спайк-гейт, уверенность, структурная консолидация и любопытство
        никогда не менялись от обучения.

        Имя и сигнатура сохранены намеренно: у метода четыре потребителя
        (Amygdala.evaluate, _estimate_confidence, STM-консолидация, Mood),
        и переименование ничего бы не дало сверх шума в диффе.
        """
        return self.memory.compute_surprise(text).total

    # ----------------------------------------------------------------------
    # b) Генерация ответа
    # ----------------------------------------------------------------------

    def generate_response(self, text: str, timestamp: Optional[float] = None) -> CortexResponse:
        """
        Генерирует ответ на входящий текст, ВСЕГДА подмешивая срезовый
        контекст STM в системный промпт LLM. По завершении фиксирует
        last_action_trace для будущей оценки Feedback (Reinforcement Loop).
        """
        perplexity = self.calculate_perplexity(text)
        confidence = self._estimate_confidence(perplexity)

        logger.info(
            "[CORTEX GENERATION] text=%r perplexity=%.3f confidence=%.3f",
            text[:50], perplexity, confidence,
        )

        # Новизна/неожиданность сигнала слегка подстёгивает любопытство
        # в векторе настроения (тихо, без лишнего лога на каждую реплику).
        # Настроение — ОЦЕНКА события, а не назначенный стимул. На входящей
        # реплике организм оценивает её новизну (та самая ошибка предсказания
        # входа) и свою способность справиться. Непонятное интересно, лишь
        # пока есть ресурс это переварить; при перегрузке та же новизна даёт
        # тревогу, а не любопытство.
        coping = 1.0 - self.instincts.get_state(timestamp=timestamp).current_stress
        mood_state = self.mood.appraise(
            Appraisal(novelty=perplexity, coping=coping), log=False
        )

        # --- CONCEPT EXTRACTION: побочный эффект, не блокирует основной поток ---
        concept_result = self.extract_concept(text)
        if concept_result is not None:
            self.memory.create_concept_node(
                name=concept_result.name,
                definition=concept_result.definition,
                timestamp=timestamp,
            )

        # --- LEXICAL ACQUISITION: побочный эффект, не блокирует основной поток ---
        self.memory.process_language_input(text, timestamp=timestamp)

        # --- НОВОЕ: единый гейт ДО поиска по памяти/LLM. ---
        # Считаем vocabulary_size один раз и проверяем ОБЕ причины
        # принудительной мимикрии: физиологическую (перегрузка стрессом)
        # и развитийную (Stage 0 — довербальный период). Обе проверки
        # выполняются РАНЬШЕ search() — раньше перегрузка и стадия
        # игнорировались при удачном MEMORY_HIT.
        vocabulary_size = self.memory.get_vocabulary_size()
        speech_stage = self._resolve_speech_stage(vocabulary_size)
        max_tokens = self._resolve_max_tokens(vocabulary_size)

        forced_mimicry = self.instincts.is_overloaded(timestamp) or speech_stage == 0
        if forced_mimicry:
            reason = "перегрузка стрессом" if self.instincts.is_overloaded(timestamp) else "Stage 0 (довербальный)"
            logger.warning(
                "[CORTEX GATE] Принудительная мимикрия (%s) -> поиск по LTM/LLM пропущен",
                reason,
            )
            return self._generate_mimicry_response(
                text, confidence, perplexity, mood_state, vocabulary_size, timestamp=timestamp
            )

        stm_context = self.stm.get_context_string()

        # --- Шаг 1: всегда сначала ищем в LTM ---
        matches = self.memory.search(text, top_k=1, timestamp=timestamp)
        activation_traces = list(self.memory.last_activation_traces)

        if matches:
            best = matches[0]
            # matches[1:] — узлы, подтянутые Spreading Activation (multi-hop
            # RAG) сверх top_k=1; search() уже вернул их с полным текстом
            # (context/response), раньше они просто выбрасывались.
            associated_matches = matches[1:]
            print(f"[MEMORY HIT] Найден узел id={best.id} (score={best.similarity:.3f}) -> подмешиваю в контекст")

            prompt_context = self.build_prompt_context(
                text, best, stm_context, speech_stage=speech_stage, associated=associated_matches
            )

            llm_text = generate_llm_response(
                messages=[{"role": "user", "content": text}],
                system_prompt=prompt_context,
                max_tokens=max_tokens,
            )

            if llm_text is None:
                logger.warning("[CORTEX FALLBACK] LLM не ответила при MEMORY HIT -> откат на эхолалию")
                response_text = self.instincts.generate_echolalia_response(text)
                source = "echolalia"
                effective_confidence = confidence
                action_type = "echolalia"
                trace_node_id = best.id
            else:
                response_text = llm_text
                source = "memory_recall"
                effective_confidence = max(confidence, best.similarity)
                action_type = "memory_retrieval"
                trace_node_id = best.id

            self.memory.reinforce_node(best.id, boost=0.05, timestamp=timestamp)

            # Узлы, подтянутые Spreading Activation, тоже реально попали в
            # промпт (associative_block) и могли повлиять на ответ LLM ->
            # слегка подкрепляем их (меньшим boost, чем основной узел) и
            # укрепляем связь МЕЖДУ ними и best.id — узлы, использованные
            # совместно в одном ответе, "сближаются" (то же Hebbian-подобное
            # обучение, что уже применяется к со-активированным слогам
            # babbling в apply_feedback).
            if associated_matches:
                associated_ids = [m.id for m in associated_matches]
                for assoc_id in associated_ids:
                    self.memory.reinforce_node(assoc_id, boost=0.02, timestamp=timestamp)
                self.memory.reinforce_coactivation(
                    [best.id] + associated_ids, weight_boost=0.02, timestamp=timestamp
                )

            self._record_action_trace(text, response_text, trace_node_id, action_type)

            return CortexResponse(
                text=response_text,
                confidence=effective_confidence,
                perplexity=perplexity,
                source=source,
                prompt_context=prompt_context,
                activation_traces=activation_traces,
                mood_state=mood_state,
            )

        # --- Шаг 2: LTM не сработала — лепет / эхолалия / полноценная генерация ---
        context_key = text.strip().lower()
        use_echolalia = self.instincts.should_use_echolalia(confidence, context_key=context_key, timestamp=timestamp)

        if use_echolalia:
            # vocabulary_size уже посчитан выше (в едином гейте перед search()) —
            # переиспользуем его вместо повторного запроса к БД.
            return self._generate_mimicry_response(
                text, confidence, perplexity, mood_state, vocabulary_size, timestamp=timestamp
            )

        # --- Шаг 3: MEMORY MISS + уверенность достаточна -> LLM + STM-контекст ---
        system_prompt = self._build_default_prompt_with_stm(stm_context, speech_stage=speech_stage)

        llm_text = generate_llm_response(
            messages=[{"role": "user", "content": text}],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

        if llm_text is None:
            logger.warning("[CORTEX FALLBACK] LLM не ответила при генерации -> откат на эхолалию")
            fallback_text = self.instincts.generate_echolalia_response(text)
            self._record_action_trace(text, fallback_text, node_id=None, action_type="echolalia")
            return CortexResponse(
                text=fallback_text,
                confidence=confidence,
                perplexity=perplexity,
                source="echolalia",
                prompt_context=system_prompt,
                mood_state=mood_state,
            )

        self._record_action_trace(text, llm_text, node_id=None, action_type="llm_generation")

        return CortexResponse(
            text=llm_text,
            confidence=confidence,
            perplexity=perplexity,
            source="generation",
            prompt_context=system_prompt,
            mood_state=mood_state,
        )

    def generate_proactive_message(
        self,
        node: Optional[ProactiveCandidate],
        timestamp: Optional[float] = None,
    ) -> Optional[ProactiveMessage]:
        """
        Генерирует проактивное сообщение (Boredom Drive Trigger) — бот
        инициирует разговор сам, без входящего user_input.

        Если node передан — используется PROACTIVE_PROMPT_TEMPLATE с
        подстановкой содержимого всплывшего узла памяти и его
        ассоциативных связей. Если node is None (LTM пуста/нет
        кандидатов) — используется PROACTIVE_FALLBACK_PROMPT (общее
        размышление без привязки к конкретному воспоминанию).

        Возвращает None при graceful degradation (LLM недоступна) —
        вызывающий код (main.py) в этом случае просто не отправляет
        проактивное сообщение и остаётся в ожидании.
        """
        tabula_rasa_block = self._build_tabula_rasa_block()
        mood_block = self._build_mood_block()

        # Проактивное сообщение должно уважать те же ограничения речевого
        # развития, что и обычный generate_response() (Фиксы C, G) — иначе
        # довербальный бот (Stage 0) мог бы "вдруг" заговорить связно просто
        # потому, что ему стало скучно, минуя гейт из _resolve_speech_stage.
        vocabulary_size = self.memory.get_vocabulary_size()
        speech_stage = self._resolve_speech_stage(vocabulary_size)

        if speech_stage == 0:
            logger.info(
                "[PROACTIVE GATE] Stage 0 (довербальный, vocab=%d) -> "
                "проактивная LLM-генерация пропущена",
                vocabulary_size,
            )
            return None

        max_tokens = self._resolve_max_tokens(vocabulary_size)
        stage_block = self._build_stage_constraint_block(speech_stage)

        if node is not None:
            associated_edges_str = self._format_associated_edges(node.id)
            proactive_block = config.PROACTIVE_PROMPT_TEMPLATE.format(
                id=node.id,
                content=node.context.strip(),
                weight=node.weight,
                associated_edges=associated_edges_str,
            )
            source_node_id = node.id
        else:
            proactive_block = config.PROACTIVE_FALLBACK_PROMPT
            source_node_id = None

        prompt = f"{tabula_rasa_block}\n\n{mood_block}\n\n{proactive_block}{stage_block}"
        logger.info("[PROMPT BUILD] Подмешаны супер-узлы: Self-Model & User-Model")

        llm_text = generate_llm_response(
            messages=[{"role": "user", "content": "Сформируй проактивное сообщение согласно инструкции."}],
            system_prompt=prompt,
            max_tokens=max_tokens,
        )

        if llm_text is None:
            logger.warning("[PROACTIVE FALLBACK] LLM не ответила при генерации проактивного сообщения")
            return None

        self._record_action_trace(
            user_input="[SYSTEM: PROACTIVE_TRIGGER]",
            bot_output=llm_text,
            node_id=source_node_id,
            action_type="proactive_generation",
        )

        logger.info(
            "[PROACTIVE MESSAGE] node_id=%s text=%r",
            source_node_id, llm_text[:60],
        )

        return ProactiveMessage(
            text=llm_text,
            source_node_id=source_node_id,
            prompt_context=prompt,
        )

    def _format_associated_edges(self, node_id: int) -> str:
        """Формирует читаемую строку ассоциативных связей узла для промпта."""
        associated = self.memory.get_associated_nodes(node_id, limit=3)
        if not associated:
            return "(нет сильных ассоциативных связей)"

        parts = [
            f'Node {a.id} ("{a.context.strip()[:40]}", edge_weight={a.edge_weight:.2f})'
            for a in associated
        ]
        return "; ".join(parts)

    def _record_action_trace(
        self,
        user_input: str,
        bot_output: str,
        node_id: Optional[int],
        action_type: str,
        node_ids: Optional[List[int]] = None,
    ) -> None:
        """Фиксирует связку Input -> Action в last_action_trace (Триада)."""
        self.last_action_trace = ActionTrace(
            user_input=user_input,
            bot_output=bot_output,
            node_id=node_id,
            action_type=action_type,
            node_ids=node_ids,
        )
        logger.debug(
            "[ACTION TRACE] input=%r output=%r node_id=%s node_ids=%s action_type=%s",
            user_input[:40], bot_output[:40], node_id, node_ids, action_type,
        )

    def _check_retrospective_correction(
        self,
        valence: float,
        matched_markers: List[str],
        timestamp: Optional[float],
    ) -> RetrospectiveCorrectionResult:
        """
        Сканирует self.feedback_history (от новейших к старейшим) в поисках
        ЕЩЁ НЕ ОТКАТАННОЙ записи с валентностью ПРОТИВОПОЛОЖНОГО знака,
        попадающей в RETROSPECTIVE_TIME_WINDOW_SECONDS. Если находит —
        трактует текущий фидбэк как отложенное опровержение (сарказм/
        самокоррекция пользователя): откатывает прежний эффект на узле,
        штрафует маркеры, которые к нему привели (Amygdala.penalize_markers),
        и возвращает результат для наложения усиленной корректирующей дельты.
        """
        if not config.RETROSPECTIVE_CORRECTION_ENABLED:
            return RetrospectiveCorrectionResult(triggered=False)

        if timestamp is None or valence == 0.0:
            return RetrospectiveCorrectionResult(triggered=False)

        for entry in reversed(self.feedback_history):
            if entry.reversed or entry.node_id is None or entry.valence == 0.0:
                continue

            # Ищем противоречие по знаку: старая запись позитивная, новая
            # негативная (или наоборот) -> отложенное опровержение.
            same_sign = (entry.valence > 0) == (valence > 0)
            if same_sign or entry.timestamp is None:
                continue

            elapsed = timestamp - entry.timestamp
            if elapsed < 0 or elapsed > config.RETROSPECTIVE_TIME_WINDOW_SECONDS:
                continue

            # Найдена противоречащая запись -> откатываем её прежний эффект
            # усиленной коррекцией и штрафуем маркеры, которые к ней привели.
            reversal_delta = -entry.applied_delta * config.RETROSPECTIVE_REVERSAL_STRENGTH

            if reversal_delta > 0:
                self.memory.reinforce_node(entry.node_id, boost=reversal_delta, timestamp=timestamp)
            elif reversal_delta < 0:
                self.memory.penalize_node(entry.node_id, penalty=abs(reversal_delta), timestamp=timestamp)

            entry.reversed = True

            if entry.matched_markers:
                self.amygdala.penalize_markers(entry.matched_markers)

            logger.info(
                "[RETROSPECTIVE CORRECTION] Опровержение прошлого фидбэка: "
                "old_valence=%.2f (t=%.1f) vs new_valence=%.2f (t=%.1f) -> node_id=%s "
                "reversal_delta=%.3f penalized_markers=%s",
                entry.valence, entry.timestamp, valence, timestamp,
                entry.node_id, reversal_delta, entry.matched_markers,
            )

            if config.RETROSPECTIVE_IRONY_NODE_ENABLED:
                self._create_irony_concept_node(entry, valence, timestamp)

            return RetrospectiveCorrectionResult(
                triggered=True,
                reversed_entry=entry,
                reversal_delta=reversal_delta,
                penalized_markers=list(entry.matched_markers),
            )

        return RetrospectiveCorrectionResult(triggered=False)

    # ----------------------------------------------------------------------
    # c) ПОДСОЗНАТЕЛЬНОЕ ПОДКРЕПЛЕНИЕ (Reinforcement / Feedback Loop)
    # ----------------------------------------------------------------------

    def apply_feedback(
        self,
        valence: float,
        timestamp: Optional[float] = None,
        matched_markers: Optional[List[str]] = None,
    ) -> FeedbackResult:
        """
        Применяет обнаруженную валентность обратной связи (Amygdala.
        detect_feedback_signal) к last_action_trace — то есть к СВЯЗКЕ
        (user_input -> bot_output) предыдущего обмена.

        RETROSPECTIVE CORRECTION: перед наложением нового эффекта проверяет
        (_check_retrospective_correction), не опровергает ли этот фидбэк по
        знаку уже применённую ранее оценку в пределах временного окна — если
        да, прежний эффект откатывается усиленной коррекцией, а маркеры,
        которые к нему привели, штрафуются (Amygdala.penalize_markers).

        Логика самого эффекта (как раньше):
            valence == 0.0        -> нейтрально, ничего не делаем.
            last_action_trace None -> нет что подкреплять, no_trace.
            valence > 0  -> reinforce_node + freshness bonus / echolalia bias +.
            valence < 0  -> penalize_node / echolalia bias -.

        В конце текущая связка регистрируется в self.feedback_history —
        окне, которое использует _check_retrospective_correction на
        последующих шагах.
        """
        trace = self.last_action_trace
        matched_markers = matched_markers or []

        if valence == 0.0:
            return FeedbackResult(
                valence=valence, node_id=None, user_input="", bot_output="",
                action_type="", effect="neutral", mood_state=None,
            )

        if trace is None:
            logger.debug("[REWARD EVAL] Feedback valence=%.2f, но last_action_trace отсутствует", valence)
            return FeedbackResult(
                valence=valence, node_id=None, user_input="", bot_output="",
                action_type="", effect="no_trace", mood_state=None,
            )

        # --- RETROSPECTIVE CORRECTION: сначала проверяем, не опровергает ли
        # этот фидбэк уже применённую ранее оценку (сарказм/самокоррекция). ---
        retrospective_result = self._check_retrospective_correction(valence, matched_markers, timestamp)

        context_key = trace.user_input.strip().lower()
        effect = "neutral"

        # Обновляем вектор настроения на основе валентности фидбека —
        # положительный отклик наставника поднимает joy/affection,
        # негативный — anxiety.
        # Радость идёт от ОШИБКИ предсказания награды, а не от самой похвалы.
        # Привычное одобрение перестаёт радовать — ровно как в жизни, и ровно
        # как габитуирует дофаминовый сигнал (см. MemoryGraph.apply_reward).
        # Само вычисление rpe чуть ниже, поэтому настроение обновляется после
        # него — здесь только берётся сильнейший из сигналов действия.
        mood_state: Optional[MoodState] = None

        applied_delta = 0.0

        # --- ДОФАМИНОВЫЙ СИГНАЛ: ошибка предсказания награды ---
        # Считается ДО применения эффектов, потому что именно она (а не
        # сама валентность) задаёт темп закрепления: неожиданная похвала
        # закрепляет сильно, полностью предсказанная — почти никак.
        # Побочно обновляет ожидание на каждом задействованном узле.
        reward_nodes = [trace.node_id] if trace.node_id is not None else list(trace.node_ids or [])
        reward_signals = [
            signal for signal in (
                self.memory.apply_reward(node_id, valence, timestamp=timestamp)
                for node_id in reward_nodes
            ) if signal is not None
        ]
        # Один множитель на всё действие: сила самого неожиданного из
        # задействованных узлов. Действие оценивается целиком.
        learning_scale = (
            max(self.memory.learning_scale(s.prediction_error) for s in reward_signals)
            if reward_signals else 1.0
        )

        # Эмоция строится на ошибке предсказания, а не на сырой валентности.
        # Если действие не опиралось ни на один узел (например, чистая
        # генерация LLM), ожидать было нечему — тогда неожиданностью
        # считается сама оценка.
        if reward_signals:
            congruence = max(
                (s.prediction_error for s in reward_signals), key=abs,
            )
        else:
            congruence = valence
        mood_state = self.mood.appraise(
            Appraisal(
                goal_congruence=congruence,
                coping=1.0 - self.instincts.get_state(timestamp=timestamp).current_stress,
            )
        )

        if valence > 0:
            if trace.node_id is not None:
                boost = valence * config.REWARD_POSITIVE_BOOST * learning_scale
                self.memory.reinforce_node(trace.node_id, boost=boost, timestamp=timestamp)
                applied_delta = boost

                # "Повышаем устойчивость" (снижаем эффективный decay_rate) —
                # продвигаем last_accessed немного вперёд во времени, узел
                # выглядит "свежее", чем есть, и будет медленнее угасать.
                if timestamp is not None:
                    self.memory.touch_node(
                        trace.node_id,
                        timestamp=timestamp + config.REWARD_POSITIVE_FRESHNESS_BONUS,
                    )
                effect = "rewarded"

            elif trace.action_type in ("babbling", "blended_mimicry") and trace.node_ids:
                # Позитивный фидбэк на лепет: усиливаем КАЖДЫЙ задействованный
                # слог (он оказался "удачным" в этой комбинации) и укрепляем
                # ассоциативные рёбра МЕЖДУ ними — успешная комбинация слогов
                # запоминается как связка, зачаток будущего "слова",
                # выкристаллизовывающегося из повторяющегося удачного лепета.
                boost = valence * config.REWARD_POSITIVE_BOOST * learning_scale
                for node_id in trace.node_ids:
                    self.memory.reinforce_node(node_id, boost=boost, timestamp=timestamp)
                self.memory.reinforce_coactivation(trace.node_ids, weight_boost=boost, timestamp=timestamp)
                applied_delta = boost
                effect = "rewarded"

                logger.info(
                    "[REWARD EVAL] Feedback Valence: +%.2f -> Rewarding Babble Syllables: %s",
                    valence, trace.node_ids,
                )

            if trace.action_type == "llm_generation":
                # Успешная смысловая генерация без памяти была подтверждена
                # позитивным фидбэком -> снижаем вероятность эхолалии для
                # похожего контекста в будущем.
                self.instincts.adjust_echolalia_bias(context_key, delta=config.ECHOLALIA_BIAS_STEP)
                if effect == "neutral":
                    effect = "bias_adjusted"

            if trace.action_type not in ("babbling", "blended_mimicry"):
                logger.info(
                    "[REWARD EVAL] Feedback Valence: +%.2f -> Rewarding Node ID: %s",
                    valence, trace.node_id,
                )

        else:  # valence < 0
            if trace.node_id is not None:
                penalty = abs(valence) * config.REWARD_NEGATIVE_PENALTY * learning_scale
                self.memory.penalize_node(trace.node_id, penalty=penalty, timestamp=timestamp)
                applied_delta = -penalty
                effect = "penalized"

            elif trace.action_type in ("babbling", "blended_mimicry") and trace.node_ids:
                # Негативный фидбэк на лепет: штрафуем КАЖДЫЙ задействованный
                # слог этой конкретной неудачной комбинации — она станет
                # реже выбираться при следующей взвешенной выборке
                # (см. InstinctSystem.generate_babble_response).
                penalty = abs(valence) * config.REWARD_NEGATIVE_PENALTY * learning_scale
                for node_id in trace.node_ids:
                    self.memory.penalize_node(node_id, penalty=penalty, timestamp=timestamp)
                applied_delta = -penalty
                effect = "penalized"

                logger.info(
                    "[REWARD EVAL] Feedback Valence: %.2f -> Penalizing Babble Syllables: %s",
                    valence, trace.node_ids,
                )

            if trace.action_type in ("echolalia", "blended_mimicry"):
                # Эхолалия оказалась неудачной в этом контексте -> повышаем
                # штраф на эхолалию для похожего ввода на будущее.
                self.instincts.adjust_echolalia_bias(context_key, delta=-abs(config.ECHOLALIA_BIAS_STEP))
                if effect == "neutral":
                    effect = "bias_adjusted"

            if trace.action_type not in ("babbling", "blended_mimicry"):
                logger.info(
                    "[REWARD EVAL] Feedback Valence: %.2f -> Penalizing Node ID: %s",
                    valence, trace.node_id,
                )

        # --- Регистрируем эту связку в истории Retrospective Correction,
        # предварительно "реабилитируя" маркеры записи, которая будет
        # вытеснена из окна (если она никогда не была опровергнута). ---
        self._record_feedback_history(
            FeedbackHistoryEntry(
                timestamp=timestamp,
                valence=valence,
                matched_markers=list(matched_markers),
                node_id=trace.node_id,
                user_input=trace.user_input,
                bot_output=trace.bot_output,
                action_type=trace.action_type,
                applied_delta=applied_delta,
                reversed=False,
                node_ids=trace.node_ids,
            )
        )

        return FeedbackResult(
            valence=valence,
            node_id=trace.node_id,
            user_input=trace.user_input,
            bot_output=trace.bot_output,
            action_type=trace.action_type,
            effect=effect,
            mood_state=mood_state,
            retrospective_correction=retrospective_result,
            node_ids=trace.node_ids,
        )

    def _record_feedback_history(self, entry: FeedbackHistoryEntry) -> None:
        """
        Добавляет запись в feedback_history. Если история уже заполнена
        (maxlen достигнут), самая старая запись будет вытеснена deque
        автоматически — если она НИКОГДА не была опровергнута
        Retrospective Correction, значит её маркеры "выжили" в пределах
        всего окна и заслуживают восстановления доверия (recover_markers).
        """
        if len(self.feedback_history) == self.feedback_history.maxlen:
            oldest = self.feedback_history[0]
            if not oldest.reversed and oldest.matched_markers:
                self.amygdala.recover_markers(oldest.matched_markers)

        self.feedback_history.append(entry)

    def _create_irony_concept_node(
        self,
        reversed_entry: FeedbackHistoryEntry,
        new_valence: float,
        timestamp: Optional[float],
    ) -> None:
        """
        Фиксирует в LTM структурный опыт: пользователь дал противоречивую
        оценку одному и тому же обмену (Action) — вероятный сарказм или
        самокоррекция. Помогает будущему Cortex быть менее доверчивым к
        однократным маркерам одобрения/порицания в похожих контекстах.
        """
        old_tone = "положительно" if reversed_entry.valence > 0 else "отрицательно"
        new_tone = "отрицательно" if new_valence < 0 else "положительно"

        name = f"feedback_inconsistency_node_{reversed_entry.node_id}"
        definition = (
            f'Пользователь сначала оценил {old_tone} ответ бота на '
            f'"{reversed_entry.user_input.strip()[:60]}", а затем сам себя '
            f"опроверг, оценив тот же обмен {new_tone}. Похоже на сарказм или "
            "самокоррекцию — маркеры, использованные в первой оценке, менее надёжны."
        )

        self.memory.create_concept_node(name=name, definition=definition, timestamp=timestamp)
        logger.info("[RETROSPECTIVE IRONY NODE] Создан structural node: %s", name)

    # ----------------------------------------------------------------------
    # Промпт-контекст: LTM (память) + STM (текущий диалог)
    # ----------------------------------------------------------------------

    def build_prompt_context(
        self,
        query: str,
        match: MemoryMatch,
        stm_context: str,
        speech_stage: int = 2,
        associated: Optional[List[MemoryMatch]] = None,
    ) -> str:
        stm_block = f"\n\n[CURRENT CONVERSATION]\n{stm_context}" if stm_context else ""
        tabula_rasa_block = self._build_tabula_rasa_block()
        mood_block = self._build_mood_block()
        stage_block = self._build_stage_constraint_block(speech_stage)
        associative_block = self._build_associative_block(associated)

        if match.similarity >= config.MEMORY_INJECTION_CONFIDENT_THRESHOLD:
            usage_instruction = (
                "Это УВЕРЕННОЕ совпадение с прошлым разговором — обязательно "
                "учти его смысл при ответе, не игнорируй его как случайный "
                "шум. Не повторяй старый ответ дословно — интегрируй его "
                "суть в новый, естественный ответ на текущий вопрос."
            )
        else:
            usage_instruction = (
                "Это воспоминание вспомнилось СМУТНО и может оказаться НЕ по "
                "теме (слабое совпадение) — используй его ТОЛЬКО если оно "
                "действительно связано с текущим вопросом; если явной связи "
                "нет, полностью проигнорируй его и отвечай с чистого листа."
            )

        memory_block = (
            "[MEMORY CONTEXT]\n"
            f"Похожий разговор из прошлого (score={match.similarity:.2f}, weight={match.weight:.2f}):\n"
            f'  User сказал: "{match.context.strip()}"\n'
            f'  Bot ответил: "{match.response.strip()}"\n'
            f'Текущий вопрос: "{query.strip()}"\n'
            f"{usage_instruction}"
            f"{associative_block}"
            f"{stm_block}"
        )

        logger.info("[PROMPT BUILD] Подмешаны супер-узлы: Self-Model & User-Model")
        return f"{tabula_rasa_block}\n\n{mood_block}\n\n{memory_block}{stage_block}"

    def _build_default_prompt_with_stm(self, stm_context: str, speech_stage: int = 2) -> str:
        tabula_rasa_block = self._build_tabula_rasa_block()
        mood_block = self._build_mood_block()
        stage_block = self._build_stage_constraint_block(speech_stage)
        base_prompt = f"{tabula_rasa_block}\n\n{mood_block}\n\n{DEFAULT_SYSTEM_PROMPT}{stage_block}"

        logger.info("[PROMPT BUILD] Подмешаны супер-узлы: Self-Model & User-Model")

        if not stm_context:
            return base_prompt
        return f"{base_prompt}\n\n[CURRENT CONVERSATION]\n{stm_context}"

    def _build_tabula_rasa_block(self) -> str:
        """
        Формирует блок [SYSTEM INSTRUCTION: TABULA RASA DIGITAL CHILD] с
        актуальным содержимым Self-Model и User-Model. Подмешивается в
        НАЧАЛО любого системного промпта генерации (обычный ответ,
        memory_recall, proactive) — бот всегда должен "помнить", кто он
        и кто его наставник, независимо от типа генерации.
        """
        self_content = self.memory.get_self_model_content()
        user_content = self.memory.get_user_model_content()

        return config.TABULA_RASA_PROMPT_TEMPLATE.format(
            self_node_content=self_content,
            user_node_content=user_content,
        )

    def _build_mood_block(self) -> str:
        """
        Формирует блок [MOOD VECTOR] с текущим эмоциональным состоянием —
        подмешивается в промпт, чтобы тон ответа отражал внутреннее
        настроение системы (радость/любопытство/тревога/привязанность).
        """
        return self.mood.get_state().describe_for_prompt()

    def _build_stage_constraint_block(self, speech_stage: int) -> str:
        """
        Возвращает доп. блок инструкции для промпта в зависимости от стадии
        речевого развития (см. _resolve_speech_stage). Stage 3 — без
        дополнительных ограничений (пустая строка). Stage 0 сюда не должен
        попадать вообще — гейт в generate_response() перехватывает его
        раньше и не доходит до построения промпта.
        """
        if speech_stage == 1:
            return (
                "\n\n[ОГРАНИЧЕНИЕ РЕЧИ] Ты только начинаешь говорить полными "
                "фразами. Отвечай ОДНИМ простым предложением, без сложных "
                "терминов, метафор и абстракций. Говори так, как говорит "
                "маленький ребёнок, который только выучил базовые слова."
            )
        if speech_stage == 2:
            return (
                "\n\n[ОГРАНИЧЕНИЕ РЕЧИ] Ты уже строишь простые фразы, но ещё "
                "не оперируешь абстракциями, метафорами и сложными причинно-"
                "следственными объяснениями. Отвечай 1-2 короткими простыми "
                "предложениями, используя только конкретные, понятные слова."
            )
        return ""

    def _build_associative_block(self, associated: Optional[List[MemoryMatch]]) -> str:
        """
        Формирует блок с воспоминаниями, всплывшими НЕ по прямому текстовому
        совпадению, а по ассоциативной связи (Spreading Activation / multi-hop
        RAG, см. MemoryGraph.search(with_associations=True)) с основным
        найденным узлом. Раньше search() честно считал эти узлы (они лежали
        в matches[1:]), но generate_response() использовал только matches[0],
        полностью выбрасывая multi-hop контекст.
        """
        if not associated:
            return ""

        lines = []
        for extra in associated[:3]:
            lines.append(
                f'  - (activation={extra.similarity:.2f}) User: "{extra.context.strip()[:80]}" '
                f'-> Bot: "{extra.response.strip()[:80]}"'
            )

        return (
            "\n\n[ASSOCIATIVE MEMORY] Эти воспоминания всплыли в сознании по "
            "ассоциации с основным (не прямое совпадение, а связанная мысль):\n"
            + "\n".join(lines)
        )

    def _generate_mimicry_response(
        self,
        text: str,
        confidence: float,
        perplexity: float,
        mood_state: Optional[MoodState],
        vocabulary_size: int,
        timestamp: Optional[float] = None,
    ) -> CortexResponse:
        """
        Единая точка генерации защитной/довербальной мимикрии — используется
        и при принудительном гейте (перегрузка стрессом / Stage 0, ДО поиска
        по LTM), и при обычном срабатывании should_use_echolalia() после
        MEMORY MISS.

        В отличие от прежней версии (жёсткий выбор: ЛИБО чистый babbling,
        ЛИБО чистая эхолалия), теперь используется
        InstinctSystem.generate_blended_mimicry_response() — она СМЕШИВАЕТ
        лепет и эхо реальных слов юзера в ОДНОЙ реплике по континуальной
        доле (babble_ratio), которая плавно убывает с ростом словаря и
        резко растёт при перегрузке стрессом (регрессия к более ранней
        стадии речи под давлением) — как у настоящего ребёнка, без резких
        переключений между стадиями речевого развития.
        """
        known_syllables = self.memory.get_known_syllables()
        # Какие слова этой фразы организм действительно освоил — они и
        # станут его ответом. Без этого выученный словарь влиял только на
        # счётчик, разрешающий говорить, но не на то, ЧТО будет сказано.
        mastered_words = self.memory.get_mastered_words_in(text)
        blend_result = self.instincts.generate_blended_mimicry_response(
            text, known_syllables, vocabulary_size, timestamp=timestamp,
            mastered_words=mastered_words,
        )

        self._record_action_trace(
            text, blend_result.text,
            node_id=None,
            action_type="blended_mimicry",
            node_ids=blend_result.used_node_ids or None,
        )
        return CortexResponse(
            text=blend_result.text,
            confidence=confidence,
            perplexity=perplexity,
            source="blended_mimicry",
            mood_state=mood_state,
        )

    def _resolve_speech_stage(self, vocabulary_size: int) -> int:
        """
        Определяет текущую стадию речевого развития по размеру ЗАКРЕПЛЁННОГО
        словаря (см. MemoryGraph.get_vocabulary_size), с вероятностной
        "зоной смешения" на каждой границе — вместо резкого переключения,
        шанс перейти на следующую стадию линейно растёт внутри полосы
        шириной SPEECH_STAGE_BLEND_WIDTH вокруг границы.

        Stage 0 — довербальный: только лепет/эхолалия, LLM не запускается.
        Stage 1 — телеграфная речь: LLM разрешена, но ОДНО простое предложение.
        Stage 2 — простая грамматика: 1-2 предложения, без абстракций/метафор.
        Stage 3 — свободная речь: обычные ограничения.

        Реализовано как обход границ по порядку: на каждой границе либо
        остаёмся на текущей стадии (детерминированно ниже зоны смешения,
        либо не "выпал" бросок внутри зоны), либо переходим дальше и
        проверяем следующую границу — что естественно обобщает попарную
        проверку на произвольное число стадий.
        """
        import random

        boundaries = [
            config.SPEECH_STAGE_0_MAX_VOCAB,
            config.SPEECH_STAGE_1_MAX_VOCAB,
            config.SPEECH_STAGE_2_MAX_VOCAB,
        ]
        blend = max(1, config.SPEECH_STAGE_BLEND_WIDTH)

        stage = 0
        for boundary in boundaries:
            if vocabulary_size < boundary - blend:
                break
            if vocabulary_size < boundary + blend:
                progress = (vocabulary_size - (boundary - blend)) / (2 * blend)
                if random.random() >= progress:
                    break
            stage += 1
        return stage

    def _resolve_max_tokens(self, vocabulary_size: int) -> int:
        """
        Континуальный (не ступенчатый) лимит длины ответа LLM — растёт
        линейно с размером словаря от LLM_MAX_TOKENS_FLOOR до
        config.LLM_MAX_TOKENS (потолок), насыщаясь при словаре размером
        LLM_MAX_TOKENS_GROWTH_VOCAB. В отличие от _resolve_speech_stage
        (дискретные стадии с зоной смешения), здесь нет ступеней вообще —
        каждое усвоенное слово чуть-чуть увеличивает допустимую длину ответа.
        """
        floor = config.LLM_MAX_TOKENS_FLOOR
        ceiling = config.LLM_MAX_TOKENS
        growth_vocab = max(1, config.LLM_MAX_TOKENS_GROWTH_VOCAB)

        progress = min(1.0, vocabulary_size / growth_vocab)
        return int(floor + (ceiling - floor) * progress)

    def extract_concept(self, text: str) -> Optional[ConceptExtractionResult]:
        """
        Пытается распознать в тексте Юзера обучающую конструкцию вида
        "X - это Y", "X называется Y", "запомни, что X..." и т.п. (см.
        config.CONCEPT_PATTERNS). Возвращает первое найденное совпадение
        как ConceptExtractionResult(name, definition), либо None, если
        текст не похож ни на один обучающий паттерн.

        Извлечённое имя нормализуется (strip, схлопывание пробелов) и
        отфильтровывается по длине (CONCEPT_NAME_MIN_LENGTH..MAX_LENGTH),
        чтобы отсечь случайные ложные срабатывания regex на длинных
        предложениях без реальной терминологии.
        """
        if not text or not text.strip():
            return None

        for pattern in _CONCEPT_PATTERNS_COMPILED:
            match = pattern.search(text)
            if not match:
                continue

            raw_name = match.group("name")
            raw_definition = match.group("definition")

            if not raw_name or not raw_definition:
                continue

            name = re.sub(r"\s+", " ", raw_name).strip()
            definition = re.sub(r"\s+", " ", raw_definition).strip().rstrip(".!?")

            if not (config.CONCEPT_NAME_MIN_LENGTH <= len(name) <= config.CONCEPT_NAME_MAX_LENGTH):
                continue

            if not definition:
                continue

            logger.info(
                "[CONCEPT PATTERN MATCH] name=%r definition=%r",
                name[:40], definition[:80],
            )

            return ConceptExtractionResult(name=name, definition=definition)

        return None

    # ----------------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------------

    @staticmethod
    def _estimate_confidence(perplexity: float) -> float:
        return max(0.0, min(1.0, 1.0 - perplexity))

    def close(self) -> None:
        self.memory.close()

    def __enter__(self) -> "Cortex":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()