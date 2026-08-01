"""
================================================================================
 SLEEP_CYCLE.PY — Фаза «Сна» (Offline Memory Consolidation & Pruning)
================================================================================
Класс SleepCycle реализует офлайн-обслуживание графа памяти "Динамического
Мозга" — процесс, который НЕ получает внешних сигналов от пользователя
(input_text=None) и вместо этого проводит внутреннюю "уборку":

    1. СИНАПТИЧЕСКИЙ ПРУНИНГ (Pruning & Edge Cleaning):
        - Удаление рёбер с weight < EDGE_FORGET_THRESHOLD.
        - Удаление "осиротевших" узлов — слабых воспоминаний без единого
          сильного ассоциативного ребра.

    2. СЕМАНТИЧЕСКАЯ КОНСОЛИДАЦИЯ (Abstract Node Generation):
        - Поиск кластеров типа "звезда вокруг хаба" (Hub-and-Spoke):
          доминантный узел + его сильнейшие спутники.
        - Вызов LLM с мета-промптом: обобщить кластер в один компактный
          абстрактный факт/вывод.
        - Запись нового абстрактного узла с повышенным весом; исходные
          узлы кластера архивируются (их вес резко снижается, что ускорит
          их последующее забывание через обычный decay).

    3. СБРОС ФИЗИОЛОГИЧЕСКОГО СОСТОЯНИЯ:
        - Полный сброс накопленного стресса (InstinctSystem) в 0.0.
        - Полная очистка буфера STM (WorkingMemory).

    4. ОТЧЁТ [SLEEP SUMMARY]:
        - SleepSummary — структурированный dataclass с полным списком
          произведённых изменений, готовый для логирования/печати в CLI.

Использование:
    sleep_cycle = SleepCycle(memory=memory_graph, stm=working_memory, instincts=instinct_system)
    summary = sleep_cycle.run_sleep_cycle(timestamp=brain_time)
    print(summary.to_report_string())
================================================================================
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import config
from decaymem.graph_memory import MemoryGraph, HubCluster
from services.llm import generate_llm_response
from storage.utils.logger import get_logger

if TYPE_CHECKING:
    from decaymem.working_memory import WorkingMemory
    from core.instincts import InstinctSystem
    from core.mood import Mood

logger = get_logger(__name__)


@dataclass
class AbstractionEvent:
    """Одно событие семантической консолидации кластера в абстрактный узел."""
    hub_id: int
    spoke_ids: List[int]
    abstract_node_id: int
    summary_text: str


@dataclass
class SleepSummary:
    """Полный структурированный отчёт о прошедшей фазе сна."""
    edges_pruned: int = 0
    orphan_nodes_pruned: int = 0
    clusters_found: int = 0
    abstraction_events: List[AbstractionEvent] = field(default_factory=list)
    stress_before: float = 0.0
    stress_after: float = 0.0
    stm_entries_flushed: int = 0
    duration_seconds: float = 0.0
    llm_unavailable: bool = False
    self_model_evolution: Optional[object] = None

    @property
    def abstract_nodes_created(self) -> int:
        return len(self.abstraction_events)

    def to_report_string(self) -> str:
        """Формирует многострочный отчёт [SLEEP SUMMARY] для консольного вывода."""
        lines = []
        lines.append("┌──────────────────── [SLEEP SUMMARY] ────────────────────")
        lines.append(f"│ Duration                : {self.duration_seconds:.3f}s")
        lines.append(f"│ Edges pruned            : {self.edges_pruned}")
        lines.append(f"│ Orphan nodes pruned     : {self.orphan_nodes_pruned}")
        lines.append(f"│ Hub clusters found      : {self.clusters_found}")
        lines.append(f"│ Abstract nodes created  : {self.abstract_nodes_created}")

        if self.abstraction_events:
            lines.append("│ Consolidation events:")
            for event in self.abstraction_events:
                preview = event.summary_text.strip().replace("\n", " ")[:60]
                lines.append(
                    f"│   hub={event.hub_id} + spokes={event.spoke_ids} "
                    f"-> new node id={event.abstract_node_id}: {preview!r}"
                )

        if self.llm_unavailable:
            lines.append("│ NOTE: LLM недоступна — консолидация пропущена (fallback)")

        if self.self_model_evolution is not None:
            if self.self_model_evolution.evolved:
                old_preview = self.self_model_evolution.old_content.strip().replace("\n", " ")[:50]
                new_preview = self.self_model_evolution.new_content.strip().replace("\n", " ")[:50]
                lines.append(
                    f"│ Self-Model evolved     : {old_preview!r} -> {new_preview!r}"
                )
            else:
                lines.append(
                    f"│ Self-Model evolution   : пропущена ({self.self_model_evolution.reason})"
                )

        lines.append(f"│ Stress reset            : {self.stress_before:.3f} -> {self.stress_after:.3f}")
        lines.append(f"│ STM flushed             : {self.stm_entries_flushed} entries")
        lines.append("└───────────────────────────────────────────────────────────")
        return "\n".join(lines)


class SleepCycle:
    """
    Оркестратор фазы "сна" — офлайн-консолидации и прунинга памяти.

    Использование:
        sleep_cycle = SleepCycle(memory=memory, stm=stm, instincts=instincts)
        summary = sleep_cycle.run_sleep_cycle(timestamp=brain_time)
    """

    def __init__(
        self,
        memory: MemoryGraph,
        stm: Optional["WorkingMemory"] = None,
        instincts: Optional["InstinctSystem"] = None,
        mood: Optional["Mood"] = None,
        persona=None,
    ):
        self.memory = memory
        self.stm = stm
        self.instincts = instincts
        # Сон сбрасывает возбуждение — единую ось нагрузки, которая
        # теперь живёт в Mood, а не в InstinctSystem
        self.mood = mood
        # Рефлексия над личностью — дело персонажа, не памяти
        self.persona = persona

    # ----------------------------------------------------------------------
    # Точка входа: полный цикл фазы сна
    # ----------------------------------------------------------------------

    def run_sleep_cycle(self, timestamp: Optional[float] = None) -> SleepSummary:
        """
        Запускает полный цикл фазы сна:
            1. Синаптический прунинг (рёбра + осиротевшие узлы)
            2. Поиск кластеров Hub-and-Spoke + семантическая консолидация (LLM)
            3. Сброс стресса и очистка STM
        Возвращает SleepSummary с полным отчётом.
        """
        start_time = time.monotonic()
        ts = timestamp if timestamp is not None else time.time()

        logger.info("[SLEEP CYCLE] Фаза сна началась (t=%.2f)", ts)

        summary = SleepSummary()

        # --- Шаг 1: Синаптический прунинг ---
        pruning_report = self.memory.run_synaptic_pruning()
        summary.edges_pruned = pruning_report.edges_pruned
        summary.orphan_nodes_pruned = pruning_report.orphan_nodes_pruned

        # --- Шаг 2: Кластеризация + семантическая консолидация ---
        # limit=1: за один вызов /sleep консолидируем только ОДИН кластер,
        # чтобы не "переваривать" граф слишком агрессивно единовременно.
        # timestamp=ts: держим все touch_node внутри кластеризации на
        # виртуальных часах brain_time, а не на реальном time.time(),
        # иначе last_accessed spoke-узлов рассинхронизируется с decay-циклом.
        clusters = self.memory.find_hub_clusters(limit=1, timestamp=ts)
        summary.clusters_found = len(clusters)

        for cluster in clusters:
            event = self._consolidate_cluster(cluster, timestamp=ts)
            if event is not None:
                summary.abstraction_events.append(event)
            else:
                summary.llm_unavailable = True

        # --- Шаг 2.5: Эволюция Self-Model (рефлексия над пережитым опытом) ---
        summary.self_model_evolution = (
            self.persona.evolve_self_model(timestamp=ts) if self.persona else None
        )

        # --- Шаг 3: Сброс физиологического состояния ---
        if self.mood is not None:
            summary.stress_before = self.mood.get_state().arousal if self.mood else 0.0
            if self.mood is not None:
                self.mood.arousal = config.MOOD_BASELINE_AROUSAL
            summary.stress_after = 0.0
            logger.info(
                "[SLEEP RESET] Стресс сброшен: %.3f -> 0.0",
                summary.stress_before,
            )

        if self.stm is not None:
            summary.stm_entries_flushed = self.stm.size()
            self.stm.clear()
            logger.info(
                "[SLEEP RESET] STM очищен: %d записей отброшено",
                summary.stm_entries_flushed,
            )

        summary.duration_seconds = time.monotonic() - start_time

        logger.info(
            "[SLEEP CYCLE] Фаза сна завершена (%.3fs): edges_pruned=%d, orphans_pruned=%d, "
            "abstract_nodes=%d",
            summary.duration_seconds, summary.edges_pruned,
            summary.orphan_nodes_pruned, summary.abstract_nodes_created,
        )

        return summary

    # ----------------------------------------------------------------------
    # Семантическая консолидация одного кластера (LLM-вызов)
    # ----------------------------------------------------------------------

    def _consolidate_cluster(
        self,
        cluster: HubCluster,
        timestamp: Optional[float] = None,
    ) -> Optional[AbstractionEvent]:
        """
        Формирует мета-промпт из воспоминаний кластера (hub + spokes),
        вызывает LLM для обобщения в один компактный факт, и записывает
        результат как новый абстрактный узел (архивируя исходные узлы).

        Возвращает None, если LLM недоступна (graceful degradation —
        кластер остаётся нетронутым, попробуем консолидировать в
        следующий раз, когда LLM будет доступна).
        """
        cluster_text = self._format_cluster_for_prompt(cluster)

        llm_summary = generate_llm_response(
            messages=[{"role": "user", "content": cluster_text}],
            system_prompt=config.SLEEP_CONSOLIDATION_PROMPT,
        )

        if llm_summary is None:
            logger.warning(
                "[SLEEP CONSOLIDATION] LLM недоступна — кластер hub=%s пропущен",
                cluster.hub_id,
            )
            return None

        abstract_node_id = self.memory.create_abstract_node(
            summary_context=f"[ABSTRACT] Консолидация кластера hub={cluster.hub_id}",
            summary_response=llm_summary,
            source_node_ids=[cluster.hub_id] + cluster.spoke_ids,
            timestamp=timestamp,
        )

        return AbstractionEvent(
            hub_id=cluster.hub_id,
            spoke_ids=cluster.spoke_ids,
            abstract_node_id=abstract_node_id,
            summary_text=llm_summary,
        )

    @staticmethod
    def _format_cluster_for_prompt(cluster: HubCluster) -> str:
        """Формирует читаемый текстовый блок воспоминаний кластера для LLM."""
        lines = [
            f'Воспоминание-хаб: User: "{cluster.hub_context.strip()}" | '
            f'Bot: "{cluster.hub_response.strip()}"'
        ]

        for idx, (ctx, resp) in enumerate(zip(cluster.spoke_contexts, cluster.spoke_responses), start=1):
            lines.append(
                f'Связанное воспоминание {idx}: User: "{ctx.strip()}" | Bot: "{resp.strip()}"'
            )

        return "\n".join(lines)