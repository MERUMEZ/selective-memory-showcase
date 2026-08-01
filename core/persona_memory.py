"""
================================================================================
 PERSONA_MEMORY.PY — Операции над памятью, принадлежащие ПЕРСОНАЖУ
================================================================================
Второй шаг выделения пакета. Здесь собрано то, что работает поверх графа
памяти, но НЕ является памятью: самообраз организма, образ наставника,
эволюция личности во сне и выбор темы для проактивного сообщения.

Почему это не ядро. Библиотеке памяти для чужого NPC не нужны ни
"self-model", ни рефлексия во сне, ни скука — у персонажа разработчика своя
личность и свои поводы заговорить первым. А вот угасание, удивление,
подкрепление и вытеснение нужны всем.

Граница проведена по зависимостям, а не по вкусу: memory/graph_memory.py
теперь ничего не знает про эти понятия, а PersonaMemory обращается к графу
снаружи, как обычный потребитель.

    memory/     — что помнить, как забывать   (ядро, переиспользуемо)
    core/       — кто помнит и зачем          (персонаж, пример применения)
================================================================================
"""

from dataclasses import dataclass
from typing import List, Optional

import config
from decaymem.graph_memory import MemoryGraph
from services.llm import generate_llm_response
from storage.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SelfModelEvolutionResult:
    """Результат попытки эволюции Self-Model во время фазы сна (Итерация H)."""
    evolved: bool
    old_content: str = ""
    new_content: str = ""
    source_node_ids: List[int] = None
    reason: str = ""

@dataclass
class ProactiveCandidate:
    """Кандидат на проактивное сообщение с рассчитанным итоговым score."""
    id: int
    context: str
    response: str
    weight: float
    relevance: float
    cooldown_penalty: float
    score: float


class PersonaMemory:
    """
    Самообраз персонажа и его поводы заговорить — поверх графа памяти.

    Использование:
        persona = PersonaMemory(memory=graph)
        self_id, user_id = persona.ensure_self_and_user_nodes()
    """

    def __init__(self, memory: MemoryGraph):
        self.memory = memory
        self.db = memory.db

    def ensure_self_and_user_nodes(self) -> "tuple[int, int]":
        """
        Проверяет наличие мета-узлов Self-Model и User-Model в БД. Если
        они отсутствуют — создаёт их со стандартными текстами из config.py
        (DEFAULT_SELF_MODEL / DEFAULT_USER_MODEL) и весом META_NODE_WEIGHT.
        Если уже существуют — просто возвращает их id без изменения контента
        (чтобы не перезатирать личность, если она уже развилась/менялась).

        Вызывается ОДНОКРАТНО при инициализации системы (main.py).
        Возвращает (self_node_id, user_node_id).
        """
        self_row = self.db.get_meta_node("self_model")
        user_row = self.db.get_meta_node("user_model")

        if self_row is None:
            self_node_id = self.db.upsert_meta_node(
                node_type="self_model",
                content=config.DEFAULT_SELF_MODEL,
                weight=config.META_NODE_WEIGHT,
            )
        else:
            self_node_id = self_row["id"]

        if user_row is None:
            user_node_id = self.db.upsert_meta_node(
                node_type="user_model",
                content=config.DEFAULT_USER_MODEL,
                weight=config.META_NODE_WEIGHT,
            )
        else:
            user_node_id = user_row["id"]

        logger.info(
            "[META INIT] Self-Model (id=%s) и User-Model (id=%s) готовы.",
            self_node_id, user_node_id,
        )

        return self_node_id, user_node_id


    def get_self_model_content(self) -> str:
        """Возвращает текущий текст Self-Model (fallback на config-дефолт)."""
        row = self.db.get_meta_node("self_model")
        return row["context"] if row is not None else config.DEFAULT_SELF_MODEL


    def evolve_self_model(self, timestamp: Optional[float] = None) -> SelfModelEvolutionResult:
        """
        Фаза сна (Итерация H): рефлексия и постепенная эволюция Self-Model.

        Собирает "дайджест" значимых узлов LTM, созданных с момента
        прошлого сна (маркер хранится как отдельный мета-узел
        'last_sleep_marker' — переиспользуем инфраструктуру мета-узлов,
        не меняя схему БД), просит LLM ПОСТЕПЕННО скорректировать текущий
        текст Self-Model с учётом пережитого опыта, и записывает результат.

        Если материала недостаточно (< SELF_MODEL_EVOLUTION_MIN_NODES)
        ИЛИ LLM недоступна — эволюция пропускается, маркер last_sleep НЕ
        обновляется (материал продолжит копиться до следующего сна).
        """
        ts = timestamp if timestamp is not None else time.time()

        marker_row = self.db.get_meta_node("last_sleep_marker")
        min_created_at = float(marker_row["context"]) if marker_row is not None else 0.0

        significant_rows = self.db.get_significant_nodes_since(
            min_created_at, limit=config.SELF_MODEL_EVOLUTION_MAX_NODES
        )

        if len(significant_rows) < config.SELF_MODEL_EVOLUTION_MIN_NODES:
            logger.info(
                "[SELF-MODEL EVOLUTION] Недостаточно опыта для рефлексии "
                "(%d < %d значимых узлов) -> пропуск",
                len(significant_rows), config.SELF_MODEL_EVOLUTION_MIN_NODES,
            )
            return SelfModelEvolutionResult(
                evolved=False,
                reason=f"Недостаточно опыта ({len(significant_rows)} узлов)",
            )

        current_self = self.get_self_model_content()
        digest = self._format_significant_nodes_for_prompt(significant_rows)
        user_message = (
            f"ТЕКУЩИЙ Self-Model:\n{current_self}\n\n"
            f"ДАЙДЖЕСТ значимых событий с прошлого сна:\n{digest}"
        )

        llm_result = generate_llm_response(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=config.SELF_MODEL_EVOLUTION_PROMPT,
        )

        if not llm_result or not llm_result.strip():
            logger.warning("[SELF-MODEL EVOLUTION] LLM недоступна/пустой ответ -> пропуск")
            return SelfModelEvolutionResult(
                evolved=False, reason="LLM недоступна или вернула пустой ответ",
            )

        new_content = llm_result.strip()[: config.SELF_MODEL_MAX_LENGTH]

        self.db.upsert_meta_node(
            node_type="self_model",
            content=new_content,
            weight=config.META_NODE_WEIGHT,
            timestamp=ts,
        )
        # Маркер обновляем ТОЛЬКО при успешной эволюции — иначе накопленный
        # с прошлого раза опыт "потерялся" бы без результата.
        self.db.upsert_meta_node(
            node_type="last_sleep_marker",
            content=str(ts),
            weight=1.0,
            timestamp=ts,
        )

        source_ids = [row["id"] for row in significant_rows]
        logger.info(
            "[SELF-MODEL EVOLUTION] Self-Model обновлён на основе %d узлов: %r -> %r",
            len(significant_rows), current_self[:60], new_content[:60],
        )

        return SelfModelEvolutionResult(
            evolved=True,
            old_content=current_self,
            new_content=new_content,
            source_node_ids=source_ids,
            reason="OK",
        )


    def select_proactive_node(
        self,
        last_active_node_id: Optional[int],
        brain_time: float,
    ) -> Optional[ProactiveCandidate]:
        """
        Выбирает узел LTM для проактивного сообщения по формуле:

            S(n) = weight_n * relevance_n * cooldown_penalty(t)

        где:
            relevance_n         = PROACTIVE_RELEVANCE_BOOST (1.5), если узел
                                   связан ребром с last_active_node_id,
                                   иначе PROACTIVE_RELEVANCE_BASE (1.0).
            cooldown_penalty(t)  = PROACTIVE_COOLDOWN_PENALTY_VALUE (0.1), если
                                   brain_time - last_accessed < PROACTIVE_COOLDOWN_SECONDS,
                                   иначе 1.0.

        Из top-K (PROACTIVE_TOP_K) кандидатов по score происходит
        вероятностный (softmax) выбор — не берём просто максимум, чтобы
        проактивные сообщения не были всегда предсказуемо про один и тот
        же самый "сильный" узел.

        Возвращает None, если в БД вообще нет узлов (вызывающий код
        должен в этом случае сгенерировать fallback-размышление без узла).
        """
        # ИСПРАВЛЕНИЕ: fetch_all_nodes() возвращал ВСЕ node_type, включая
        # служебные лексические узлы ('word'/'syllable') и мета-узлы
        # (Self-Model/User-Model) — они могли попасть в проактивное
        # сообщение как будто это реальное воспоминание (у лексических
        # узлов context == response == текст слова/слога, что выглядело
        # бы абсурдно в PROACTIVE_PROMPT_TEMPLATE). fetch_searchable_nodes()
        # уже корректно ограничен node_type IN ('episodic', 'concept').
        rows = self.db.fetch_searchable_nodes()
        if not rows:
            logger.info("[PROACTIVE RECALL] LTM пуста — нет кандидатов для проактивного узла")
            return None

        related_ids: set = set()
        if last_active_node_id is not None:
            related_edges = self.db.get_edges_for_node(last_active_node_id)
            related_ids = {
                edge["neighbor_id"] for edge in related_edges
                if edge["weight"] >= config.EDGE_ACTIVATION_THRESHOLD
            }

        candidates: List[ProactiveCandidate] = []

        for row in rows:
            relevance = (
                config.PROACTIVE_RELEVANCE_BOOST
                if row["id"] in related_ids
                else config.PROACTIVE_RELEVANCE_BASE
            )

            seconds_since_touch = brain_time - row["last_accessed"]
            cooldown_penalty = (
                config.PROACTIVE_COOLDOWN_PENALTY_VALUE
                if 0 <= seconds_since_touch < config.PROACTIVE_COOLDOWN_SECONDS
                else 1.0
            )

            score = row["weight"] * relevance * cooldown_penalty

            candidates.append(
                ProactiveCandidate(
                    id=row["id"],
                    context=row["context"],
                    response=row["response"],
                    weight=row["weight"],
                    relevance=relevance,
                    cooldown_penalty=cooldown_penalty,
                    score=score,
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        top_candidates = candidates[: config.PROACTIVE_TOP_K]

        if not top_candidates:
            return None

        chosen = self._softmax_choice(top_candidates)

        logger.info(
            "[PROACTIVE RECALL] Выбран узел id=%s (score=%.3f, relevance=%.2f, "
            "cooldown_penalty=%.2f) из %d кандидатов",
            chosen.id, chosen.score, chosen.relevance, chosen.cooldown_penalty,
            len(top_candidates),
        )

        # Обращение к узлу засчитывается как "касание" — он всплыл в сознании
        self.memory.touch_node(chosen.id, timestamp=brain_time)

        return chosen


    @staticmethod
    def _softmax_choice(candidates: List[ProactiveCandidate]) -> ProactiveCandidate:
        """
        Вероятностный (softmax) выбор одного кандидата из списка на основе
        их score. Температура берётся из config.PROACTIVE_SOFTMAX_TEMPERATURE.
        """
        import random

        temperature = max(1e-6, config.PROACTIVE_SOFTMAX_TEMPERATURE)
        scores = [c.score for c in candidates]
        max_score = max(scores)

        # Численно стабильный softmax (вычитаем max перед exp)
        exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
        total = sum(exp_scores)

        if total <= 0:
            return candidates[0]

        probabilities = [e / total for e in exp_scores]
        return random.choices(candidates, weights=probabilities, k=1)[0]

    def __post_init__(self):
        if self.source_node_ids is None:
            self.source_node_ids = []

    @staticmethod
    def _format_significant_nodes_for_prompt(rows: List["sqlite3.Row"]) -> str:
        """Формирует читаемый текстовый дайджест значимых узлов для LLM."""
        lines = []
        for row in rows:
            ctx = (row["context"] or "").strip().replace("\n", " ")[:100]
            resp = (row["response"] or "").strip().replace("\n", " ")[:100]
            lines.append(f'- (weight={row["weight"]:.2f}) User: "{ctx}" | Bot: "{resp}"')
        return "\n".join(lines)
