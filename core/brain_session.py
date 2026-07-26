"""
================================================================================
 CORE/BRAIN_SESSION.PY — Инкапсуляция "мозга" в переиспользуемую сессию
================================================================================
Этап 1 миграции CLI -> Telegram-бот.

BrainSession оборачивает весь набор компонентов (Perception, Amygdala,
InstinctSystem, Cortex, MemoryGraph, WorkingMemory, SleepCycle, BoredomDrive)
и внутренние часы (brain_time) в один объект с чистым методом
process_message(text) -> BrainResponse, без единого input()/print().

На этом этапе НЕ рассматриваются: Telegram, aiogram, asyncio-планировщик,
мультитенантность (реестр сессий), выгрузка по неактивности — это шаги 2-6.

Один BrainSession = один "мозг" с одной SQLite БД по указанному db_path.
================================================================================
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Any

import config
from core.amygdala import Amygdala
from core.cortex import Cortex
from core.drives import BoredomDrive
from core.instincts import InstinctSystem
from core.perception import Perception
from memory.graph_memory import MemoryGraph
from memory.database import Database
from memory.sleep_cycle import SleepCycle
from memory.working_memory import WorkingMemory
from storage.utils.logger import get_logger

logger = get_logger(__name__)

TICK_SECONDS = config._get_float("TICK_SECONDS", 120.0) if hasattr(config, "_get_float") else 120.0
AUTO_SLEEP_IDLE_TICKS = config._get_int("AUTO_SLEEP_IDLE_TICKS", 15) if hasattr(config, "_get_int") else 15

SLEEP_COMMANDS = {"/sleep", "спать", "сон"}


@dataclass
class BrainResponse:
    """Результат обработки одного сообщения пользователя."""
    text: str
    is_sleep_report: bool = False
    debug: dict = field(default_factory=dict)


class SharedActivationState:
    """
    Потокобезопасный контейнер для last_active_node_id и окна со-активации
    STM. Перенесён без изменений из main.py — в будущем (Этап 5) будет
    использоваться и фоновым asyncio-тиком.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._last_active_node_id: Optional[int] = None
        self._stm_window_node_ids: list = []

    def set_last_active(self, node_id: Optional[int]) -> None:
        with self._lock:
            self._last_active_node_id = node_id

    def get_last_active(self) -> Optional[int]:
        with self._lock:
            return self._last_active_node_id

    def append_window_node(self, node_id: int) -> None:
        with self._lock:
            self._stm_window_node_ids.append(node_id)

    def get_window_snapshot(self) -> list:
        with self._lock:
            return list(self._stm_window_node_ids)

    def clear_window(self) -> None:
        with self._lock:
            self._stm_window_node_ids.clear()


class SharedBrainClock:
    """
    Потокобезопасный контейнер для brain_time. Перенесён без изменений
    из main.py (SharedBrainClock) — сохранён threading.Lock, т.к. в
    следующих этапах фоновый тик может жить в отдельном потоке/таске.
    """

    def __init__(self, initial_brain_time: float):
        self._lock = threading.Lock()
        self._brain_time = initial_brain_time
        self.last_user_msg_time = initial_brain_time
        self.last_activity_time = time.time()

    def get_brain_time(self) -> float:
        with self._lock:
            return self._brain_time

    def advance_by(self, delta_seconds: float) -> float:
        with self._lock:
            self._brain_time += delta_seconds
            return self._brain_time

    def register_user_message(self, delta_seconds: float) -> float:
        with self._lock:
            self._brain_time += delta_seconds
            self.last_user_msg_time = self._brain_time
            self.last_activity_time = time.time()
            return self._brain_time

    def register_activity(self) -> None:
        with self._lock:
            self.last_activity_time = time.time()

    def seconds_since_last_user_message(self) -> float:
        with self._lock:
            return self._brain_time - self.last_user_msg_time

    def seconds_since_last_activity(self) -> float:
        with self._lock:
            return time.time() - self.last_activity_time


class BrainSession:
    """
    Один изолированный "мозг" (один пользователь = один BrainSession с
    собственной SQLite БД по db_path). Инкапсулирует весь пайплайн:

        Perception -> Amygdala(spike + feedback valence) -> InstinctSystem
            -> WorkingMemory (STM) -> Cortex(response, LTM+STM) -> MemoryGraph (LTM)

    process_message() — единственная точка входа для внешнего кода
    (CLI, Telegram-хендлер и т.п.). Чистая функция: вход — текст,
    выход — BrainResponse. Никакого input()/print() внутри.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.BRAIN_DB_PATH

        db = Database(db_path=self.db_path)
        self.memory = MemoryGraph(db=db)
        self.self_node_id, self.user_node_id = self.memory.ensure_self_and_user_nodes()

        self.stm = WorkingMemory()
        self.instincts = InstinctSystem()
        self.amygdala = Amygdala()
        self.cortex = Cortex(
            instincts=self.instincts,
            memory=self.memory,
            stm=self.stm,
            amygdala=self.amygdala,
        )
        self.sleep_cycle = SleepCycle(memory=self.memory, stm=self.stm, instincts=self.instincts)
        self.boredom_drive = BoredomDrive()

        session_start_real = time.time()
        self.clock = SharedBrainClock(initial_brain_time=session_start_real)
        self.activation_state = SharedActivationState()

        self.idle_ticks_without_event: int = 0

        logger.info(
            "[BRAIN SESSION] Инициализирована сессия db_path=%s (self_node=%s, user_node=%s)",
            self.db_path, self.self_node_id, self.user_node_id,
        )

    # ----------------------------------------------------------------------
    # Публичный API
    # ----------------------------------------------------------------------

    def process_message(self, user_input: str) -> BrainResponse:
        """
        Обрабатывает одно сообщение пользователя и возвращает BrainResponse
        (текст ответа + debug-словарь). Полный перенос логики из
        main.py::run() (тело while-цикла), без input()/print().
        """
        if not user_input:
            return BrainResponse(text="", debug={"skipped": True})

        brain_time = self.clock.register_user_message(TICK_SECONDS)
        self.boredom_drive.on_user_message(brain_time)

        if user_input.lower() in SLEEP_COMMANDS:
            sleep_summary = self.sleep_cycle.run_sleep_cycle(timestamp=brain_time)
            self.activation_state.clear_window()
            self.idle_ticks_without_event = 0
            return BrainResponse(
                text=sleep_summary.to_report_string(),
                is_sleep_report=True,
                debug={"brain_time": brain_time, "event": "manual_sleep"},
            )

        logger.info("[BRAIN CLOCK] Tick: brain_time=%.1f (+%.1fs)", brain_time, TICK_SECONDS)

        # 2. ПОДСОЗНАТЕЛЬНОЕ ПОДКРЕПЛЕНИЕ — Триада Input -> Action -> Feedback.
        reward_trace_log = None
        reward_eval_log = None

        feedback_signal = self.amygdala.detect_feedback_signal(user_input)
        feedback_valence = feedback_signal.valence
        if feedback_valence != 0.0 and self.cortex.last_action_trace is not None:
            trace = self.cortex.last_action_trace
            reward_trace_log = (
                f'[REWARD TRACE] User Trigger: "{trace.user_input}" | '
                f'Bot Action: "{trace.bot_output}"'
            )

            feedback_result = self.cortex.apply_feedback(
                feedback_valence,
                timestamp=brain_time,
                matched_markers=feedback_signal.matched_markers,
            )

            sign = "+" if feedback_valence > 0 else ""
            if feedback_result.action_type == "babbling" and feedback_result.node_ids:
                target_desc = f"Syllable Node IDs: {feedback_result.node_ids}"
            else:
                target_desc = f"Node ID: {feedback_result.node_id}"
            reward_eval_log = (
                f"[REWARD EVAL] Feedback Valence: {sign}{feedback_valence:.2f} -> "
                f"{'Rewarding' if feedback_valence > 0 else 'Penalizing'} {target_desc}"
            )

            retro = feedback_result.retrospective_correction
            if retro is not None and retro.triggered:
                retro_log = (
                    f"[RETROSPECTIVE CORRECTION] Опровергнута прошлая оценка "
                    f"(node_id={retro.reversed_entry.node_id}, "
                    f"old_valence={retro.reversed_entry.valence:+.2f}, "
                    f"reversal_delta={retro.reversal_delta:+.3f}, "
                    f"penalized_markers={retro.penalized_markers})"
                )
                reward_eval_log = f"{reward_eval_log}\n{retro_log}"

        # 3. Perception
        perception_result = Perception().analyze(user_input)
        emotion_score = perception_result.emotion_score

        # 4. Perplexity
        perplexity = self.cortex.calculate_perplexity(user_input)

        # 5. Состояние инстинктов
        stress_state = self.instincts.get_state(timestamp=brain_time)

        # 6. Amygdala — spike detection
        amygdala_result = self.amygdala.evaluate(
            emotion_score=emotion_score,
            perplexity=perplexity,
            stress_level=stress_state.current_stress,
        )

        self.instincts.accumulate_stress(emotion_score, timestamp=brain_time)

        # 7. Записываем реплику пользователя в STM
        self.stm.add_message(
            role="user",
            text=user_input,
            emotion_score=emotion_score,
            perplexity=perplexity,
            timestamp=brain_time,
        )

        # 8. Cortex — генерация ответа.
        cortex_response = self.cortex.generate_response(user_input, timestamp=brain_time)

        # 8b. MEMORY HIT -> запоминаем узел в окне со-активации.
        if (
            self.cortex.last_action_trace is not None
            and self.cortex.last_action_trace.action_type == "memory_retrieval"
            and self.cortex.last_action_trace.node_id is not None
        ):
            self.activation_state.append_window_node(self.cortex.last_action_trace.node_id)
            self.activation_state.set_last_active(self.cortex.last_action_trace.node_id)

        # 9. Записываем ответ бота в STM
        self.stm.add_message(
            role="bot",
            text=cortex_response.text,
            emotion_score=0.0,
            perplexity=cortex_response.perplexity,
            timestamp=brain_time,
        )

        # 10. Spike -> немедленный узел LTM
        memory_written = False
        if amygdala_result.is_spike_triggered:
            new_node_id = self.memory.save_connection(
                context=user_input,
                response=cortex_response.text,
                weight=amygdala_result.total_density,
                timestamp=brain_time,
            )
            memory_written = True
            self.activation_state.set_last_active(new_node_id)
            logger.info("[BRAIN STATE] Spike Triggered! New memory node formed.")

            # СВЯЗЬ ПО КОНТЕКСТУ: если в этом же обмене был найден узел
            # через MEMORY HIT, связываем его с новым spike-узлом.
            window_snapshot = self.activation_state.get_window_snapshot()
            if window_snapshot:
                source_node_id = window_snapshot[-1]
                self.memory.connect_nodes(source_node_id, new_node_id, timestamp=brain_time)

            self.activation_state.append_window_node(new_node_id)

        # 11. ИЗБИРАТЕЛЬНАЯ КОНСОЛИДАЦИЯ (STM -> LTM)
        consolidation_event = None
        if self.stm.is_full() or amygdala_result.is_spike_triggered:
            episode = self.stm.consume_all()
            result = self.memory.consolidate_from_stm(
                episode,
                timestamp=brain_time,
                already_captured_by_spike=memory_written,
            )

            if result.decision == "emotional_node":
                consolidation_event = (
                    f"[CONSOLIDATION] Эмоциональный узел записан в БД "
                    f"(id={result.node_id}, weight={result.weight:.2f})"
                )
            elif result.decision == "structural_node":
                consolidation_event = (
                    f"[CONSOLIDATION] Структурный узел записан в БД "
                    f"(id={result.node_id}, weight={result.weight:.2f})"
                )
            else:
                consolidation_event = "[STM FLUSH] Рутинный шум отброшен"

            # СВЯЗЬ ПО КОНТЕКСТУ: если консолидация создала новый узел,
            # связываем его с последним MEMORY HIT-узлом этого окна.
            window_snapshot = self.activation_state.get_window_snapshot()
            if result.node_id is not None and window_snapshot:
                source_node_id = window_snapshot[-1]
                self.memory.connect_nodes(source_node_id, result.node_id, timestamp=brain_time)
                self.activation_state.append_window_node(result.node_id)
                self.activation_state.set_last_active(result.node_id)

            # СВЯЗЬ ПО СО-АКТИВАЦИИ: все узлы, задействованные в рамках
            # этого окна STM, получают усиленные рёбра друг с другом.
            self.memory.reinforce_coactivation(
                self.activation_state.get_window_snapshot(), timestamp=brain_time
            )
            self.activation_state.clear_window()

            self.idle_ticks_without_event = 0
        else:
            self.idle_ticks_without_event += 1

        # 12. Decay применяется синхронно, используя виртуальное brain_time
        decayed_nodes = self.memory.apply_decay(now=brain_time)
        total_nodes = self.memory.count_nodes()

        # 12b. АВТОМАТИЧЕСКИЙ ТРИГГЕР ФАЗЫ СНА: переполнение памяти ИЛИ
        # продолжительное "молчание по существу".
        auto_sleep_reason = None
        sleep_report_text = None
        if total_nodes >= config.SLEEP_AUTO_TRIGGER_NODE_COUNT:
            auto_sleep_reason = (
                f"переполнение памяти ({total_nodes} >= "
                f"{config.SLEEP_AUTO_TRIGGER_NODE_COUNT} узлов)"
            )
        elif self.idle_ticks_without_event >= AUTO_SLEEP_IDLE_TICKS:
            auto_sleep_reason = (
                f"продолжительное молчание ({self.idle_ticks_without_event} тиков без событий)"
            )

        if auto_sleep_reason:
            logger.info("[BRAIN STATE] Автоматический триггер фазы сна: %s", auto_sleep_reason)
            auto_sleep_summary = self.sleep_cycle.run_sleep_cycle(timestamp=brain_time)
            sleep_report_text = auto_sleep_summary.to_report_string()
            self.idle_ticks_without_event = 0
            total_nodes = self.memory.count_nodes()

        # 12c. Настроение (Mood) затухает к базовому уровню на каждом тике.
        mood_snapshot = self.cortex.mood.decay(log=False)

        # 13. Обновлённое состояние стресса
        updated_stress_state = self.instincts.get_state(timestamp=brain_time)

        # 14. Итоговый ответ + debug-словарь (замена print_debug_block).
        response_text = cortex_response.text
        if auto_sleep_reason and sleep_report_text:
            response_text = f"{response_text}\n\n[AUTO-SLEEP] {auto_sleep_reason}\n{sleep_report_text}"

        debug = {
            "brain_time": brain_time,
            "emotion_score": emotion_score,
            "perplexity": amygdala_result.perplexity,
            "total_density": amygdala_result.total_density,
            "confidence": cortex_response.confidence,
            "stress_state": updated_stress_state,
            "spike_triggered": amygdala_result.is_spike_triggered,
            "memory_written": memory_written,
            "response_source": cortex_response.source,
            "decayed_nodes": decayed_nodes,
            "total_nodes": total_nodes,
            "top_nodes": self.memory.get_top_nodes(limit=5),
            "stm_status": self.stm.get_status_string(),
            "consolidation_event": consolidation_event,
            "prompt_context": cortex_response.prompt_context,
            "reward_trace": reward_trace_log,
            "reward_eval": reward_eval_log,
            "activation_traces": cortex_response.activation_traces,
            "mood_state": mood_snapshot,
            "auto_sleep_reason": auto_sleep_reason,
        }

        return BrainResponse(text=response_text, is_sleep_report=False, debug=debug)

    # ----------------------------------------------------------------------
    # Обслуживание фонового "тика" (заготовка для Этапа 5 — Idle/Boredom).
    # Пока НЕ вызывается автоматически, доступно для точечных вызовов из
    # тест-скрипта, если потребуется проверить advance_by/boredom вручную.
    # ----------------------------------------------------------------------

    def advance_idle_tick(self, delta_seconds: float) -> float:
        """Продвигает brain_time на delta_seconds без сообщения пользователя."""
        return self.clock.advance_by(delta_seconds)

    def close(self) -> None:
        """Освобождает ресурсы (закрывает соединение с БД через Cortex)."""
        self.cortex.close()
        logger.info("[BRAIN SESSION] Сессия закрыта (db_path=%s)", self.db_path)