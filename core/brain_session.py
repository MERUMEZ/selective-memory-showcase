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

AUTO_SLEEP_IDLE_TICKS = config._get_int("AUTO_SLEEP_IDLE_TICKS", 15) if hasattr(config, "_get_int") else 15

SLEEP_COMMANDS = {"/sleep", "спать", "сон"}


@dataclass
class BrainResponse:
    """Результат обработки одного сообщения пользователя."""
    text: str
    is_sleep_report: bool = False
    debug: dict = field(default_factory=dict)


@dataclass
class BrainIdleEvent:
    """
    Результат одного фонового idle-тика (см. BrainSession.run_idle_tick),
    сигнализирующий вызывающему коду (bot.py — фоновый asyncio-scheduler),
    что нужно что-то сделать во внешнем мире (или просто залогировать).

    kind="sleep"      -> автоматический Idle Sleep был выполнен; text
                         содержит технический отчёт (to_report_string()),
                         НЕ предназначенный для отправки пользователю в
                         Telegram — вызывающий код должен только залогировать
                         это событие, а не слать report юзеру как сообщение.
    kind="proactive"  -> сгенерировано настоящее проактивное сообщение
                         (Boredom Drive Trigger) — text нужно отправить
                         пользователю как обычное сообщение от бота.
    """
    kind: str  # "sleep" | "proactive"
    text: str
    source_node_id: Optional[int] = None


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
    Субъективное время организма — ОДНА когерентная шкала.

        brain_time = эпоха + (настенное - эпоха_настенная) * TIME_ACCELERATION

    Раньше часы были составными и ломались тремя способами сразу:

    1. TICK_SECONDS=120 прибавлялись за КАЖДОЕ сообщение независимо от
       того, сколько прошло на самом деле, плюс фоновый тик капал
       реальными секундами. Замер: часы бежали ВСЕМЕРО быстрее настенных
       (30 сообщений за 10 реальных минут давали 70 минут внутренних).
       Из-за этого все временные константы тайно зависели от интенсивности
       общения: AGE_T0="1 час" на деле означал ~9 минут живого разговора.

    2. BrainSession.__init__ ставил brain_time = time.time(), поэтому при
       выгрузке и возврате сессии часы ПРЫГАЛИ НАЗАД на настенное "сейчас",
       а last_decayed_at узлов оставался от разогнанных. В _decay_nodes
       стоит `if dt <= 0: continue`, то есть забывание МОЛЧА выключалось:
       после разговора на 100 сообщений — ещё на 2.3 часа.

    3. seconds_since_last_user_message считалось по brain_time, а
       seconds_since_last_activity по time.time() — одна и та же пауза
       мерялась двумя разными линейками.

    Теперь шкала одна, монотонная и переживающая перезагрузку: эпоха
    хранится в БД (мета-узел), поэтому продолжение сессии не сбрасывает
    отсчёт. Ускорение стало ОДНОЙ явной константой, а не побочным эффектом
    от "+120 за сообщение".

    ВНЕШНИЕ события меряются НАСТЕННЫМ временем и намеренно: "ушёл ли
    пользователь" и "пора ли выгрузить сессию из памяти" — вопросы про
    внешний мир и оперативную память, а не про субъективное время
    организма (см. seconds_since_last_activity).
    """

    def __init__(self, epoch: float):
        """
        epoch — ОДНА настенная метка: момент, когда у этого мозга начался
        отсчёт субъективного времени. Хранится в БД и переживает
        перезагрузку.

        Именно одна, а не пара (начальное brain_time + настенная точка):
        при паре момент старта процесса становится второй точкой отсчёта,
        и часы снова сбрасываются при каждой перезагрузке — ровно тот
        дефект, который мы и чиним.
        """
        self._lock = threading.Lock()
        self._epoch = epoch
        self.last_user_brain_time = self._brain_time_unlocked()
        self.last_activity_wall = time.time()

    def get_brain_time(self) -> float:
        """Субъективное время организма — чистая функция настенного."""
        with self._lock:
            return self._brain_time_unlocked()

    def _brain_time_unlocked(self) -> float:
        elapsed = max(0.0, time.time() - self._epoch)
        return self._epoch + elapsed * config.TIME_ACCELERATION

    def register_user_message(self) -> float:
        """Отмечает реплику пользователя. Часы идут сами — двигать их нечем."""
        with self._lock:
            brain_time = self._brain_time_unlocked()
            self.last_user_brain_time = brain_time
            self.last_activity_wall = time.time()
            return brain_time

    def register_activity(self) -> None:
        with self._lock:
            self.last_activity_wall = time.time()

    def simulate_elapsed_wall_seconds(self, wall_seconds: float) -> float:
        """
        Промотать часы так, будто прошло wall_seconds НАСТЕННОГО времени.
        Нужно измерительному стенду, чтобы моделировать паузы между
        сессиями, не ожидая их в реальности.

        Эпоха входит в формулу дважды (brain = epoch + (now-epoch)*A),
        поэтому сдвиг на D даёт прирост D*(A-1), а не D*A. Отсюда
        коэффициент: чтобы получить прирост wall_seconds*A, сдвигать надо
        на wall_seconds*A/(A-1). Наивный сдвиг на wall_seconds*A промотал
        бы часы вшестеро дальше нужного.
        """
        acceleration = config.TIME_ACCELERATION
        if acceleration <= 1.0:
            shift = wall_seconds
        else:
            shift = wall_seconds * acceleration / (acceleration - 1.0)
        with self._lock:
            self._epoch -= shift
            self.last_activity_wall -= wall_seconds
            return self._brain_time_unlocked()

    def seconds_since_last_user_message(self) -> float:
        """В субъективных секундах — это про внутреннюю жизнь организма."""
        with self._lock:
            return self._brain_time_unlocked() - self.last_user_brain_time

    def seconds_since_last_activity(self) -> float:
        """
        В НАСТЕННЫХ секундах — намеренно. Отсюда решается, ушёл ли
        пользователь и пора ли выгружать сессию из оперативной памяти:
        вопросы про внешний мир, а не про субъективное время.
        """
        with self._lock:
            return time.time() - self.last_activity_wall


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
        self.sleep_cycle = SleepCycle(
            memory=self.memory, stm=self.stm,
            instincts=self.instincts, mood=self.cortex.mood,
        )
        self.boredom_drive = BoredomDrive()

        # Эпоха переживает перезагрузку: иначе часы прыгали бы назад
        # относительно меток в БД и забывание молча выключалось
        epoch = self.memory.get_or_create_brain_epoch()
        self.clock = SharedBrainClock(epoch=epoch)
        self.activation_state = SharedActivationState()

        self.idle_ticks_without_event: int = 0

        # Один "мозг" обслуживается из РАЗНЫХ ПОТОКОВ: bot.py уводит в
        # asyncio.to_thread и обработку сообщения пользователя, и фоновый
        # idle-тик, и /status, и выгрузку сессии. Все они трогают одно и то
        # же: соединение SQLite, буфер STM, cortex.last_action_trace, часы.
        #
        # Сценарий вполне достижимый: юзер пишет на 46-й секунде молчания,
        # ровно когда фоновый тик начал фазу сна — прунинг удаляет узлы
        # под ногами у идущего поиска.
        #
        # Не RLock: ни один из этих методов не вызывает другой, поэтому
        # повторный вход невозможен, а обычный Lock честнее ловит ошибки,
        # если такой вызов однажды появится.
        self._lock = threading.Lock()

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
        (текст ответа + debug-словарь).

        Сообщение пользователя имеет приоритет: оно ЖДЁТ блокировку, а не
        отказывается от неё. Уступает наоборот фоновый тик (см. run_idle_tick).
        """
        with self._lock:
            return self._process_message_unlocked(user_input)

    def _process_message_unlocked(self, user_input: str) -> BrainResponse:
        """
        Тело обработки сообщения. Вызывать только под self._lock —
        трогает SQLite, STM, last_action_trace и часы.
        """
        if not user_input:
            return BrainResponse(text="", debug={"skipped": True})

        brain_time = self.clock.register_user_message()
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

        logger.info("[BRAIN CLOCK] brain_time=%.1f (ускорение x%.1f)", brain_time, config.TIME_ACCELERATION)

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
            if feedback_result.action_type in ("babbling", "blended_mimicry") and feedback_result.node_ids:
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
        stress_state = self.instincts.get_state(self.cortex.mood.get_state().arousal)

        # 6. Amygdala — spike detection
        amygdala_result = self.amygdala.evaluate(
            emotion_score=emotion_score,
            perplexity=perplexity,
            stress_level=stress_state.arousal,
        )


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
        #
        # Консолидация запускается ТОЛЬКО по заполнению буфера. Раньше её
        # дёргал ещё и спайк — и это вытирало кратковременную память
        # (consume_all) ровно в тот момент, когда разговор стал интересным.
        # Замер показывал, что STM пуста после 39 сообщений из 40, то есть
        # бот практически никогда не видел нити разговора. При этом спайк
        # НИЧЕГО не терял от такой отвязки: он уже записал текущий обмен
        # собственным узлом LTM шагом выше (шаг 10).
        consolidation_event = None
        if self.stm.is_full():
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
        # Считаем ТОЛЬКО узлы-воспоминания. Раньше здесь стоял total_nodes
        # по всем типам, включая лексику, — а словарь набирает сотни узлов
        # за первый десяток сообщений, поэтому порог пробивался на 9-м
        # сообщении и сон запускался на каждое следующее (см.
        # Database.count_memory_nodes).
        memory_nodes = self.memory.db.count_memory_nodes()
        if memory_nodes >= config.SLEEP_AUTO_TRIGGER_NODE_COUNT:
            auto_sleep_reason = (
                f"переполнение памяти ({memory_nodes} >= "
                f"{config.SLEEP_AUTO_TRIGGER_NODE_COUNT} узлов-воспоминаний)"
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
        updated_stress_state = self.instincts.get_state(self.cortex.mood.get_state().arousal)

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

    def get_status_report(self) -> str:
        """
        Человекочитаемая сводка состояния "мозга" для команды /status.

        Смысл не в отладке, а в ОБРАТНОЙ СВЯЗИ ДЛЯ УЧИТЕЛЯ. Главная причина,
        по которой человек бросал обучение, — он не видел ничего: бот сто
        сообщений подряд отвечал лепетом, и понять, продвигается ли что-то
        вообще, было невозможно. Здесь показывается прогресс до следующей
        речевой стадии и то, какие слова реально закрепились.

        Стадия считается ДЕТЕРМИНИРОВАННО (простым сравнением с порогами),
        в отличие от Cortex._resolve_speech_stage с вероятностной зоной
        смешения: в отчёте нужна стабильная картинка, а не разное число
        при двух подряд вызовах /status.
        """
        with self._lock:
            return self._status_report_unlocked()

    def _status_report_unlocked(self) -> str:
        mastered = self.memory.get_vocabulary_size()
        exposed = self.memory.get_exposed_vocabulary_size()

        boundaries = [
            (config.SPEECH_STAGE_0_MAX_VOCAB, "лепет (довербальная стадия)"),
            (config.SPEECH_STAGE_1_MAX_VOCAB, "простые фразы из одного предложения"),
            (config.SPEECH_STAGE_2_MAX_VOCAB, "простая грамматика, 1-2 предложения"),
        ]

        stage_index = 0
        stage_name = "свободная речь"
        next_threshold = None
        for index, (threshold, name) in enumerate(boundaries):
            if mastered < threshold:
                stage_index = index
                stage_name = name
                next_threshold = threshold
                break
        else:
            stage_index = len(boundaries)

        lines = [
            "🧠 Состояние",
            f"Стадия речи: {stage_index} — {stage_name}",
        ]

        if next_threshold is not None:
            remaining = next_threshold - mastered
            filled = int(12 * mastered / next_threshold)
            bar = "█" * min(12, filled) + "░" * max(0, 12 - filled)
            lines.append(f"Прогресс: {bar} {mastered}/{next_threshold}")
            lines.append(f"До следующей стадии осталось выучить слов: {remaining}")
        else:
            lines.append(f"Словарь: {mastered} слов — все стадии пройдены")

        lines.append("")
        lines.append(f"Освоено слов: {mastered} (услышано всего: {exposed})")
        lines.append(f"Узлов в памяти: {self.memory.count_nodes()}")

        top_words = self.memory.get_top_words(limit=6)
        if top_words:
            rendered = ", ".join(f"{text} ({weight:.2f})" for text, weight in top_words)
            lines.append(f"Лучше всего выучено: {rendered}")

        if mastered < config.SPEECH_STAGE_0_MAX_VOCAB:
            lines.append("")
            lines.append(
                "Пока я только лепечу. Слово закрепляется примерно с третьего "
                "употребления — повторяй одни и те же слова, так я выучу их быстрее."
            )

        return "\n".join(lines)

    # ----------------------------------------------------------------------
    # Обслуживание фонового "тика" (Этап 5 — Idle Sleep / Boredom).
    # Вызывается извне (bot.py::_idle_scheduler_loop) через asyncio.to_thread
    # на каждой итерации ЕДИНОГО фонового asyncio-таска для ВСЕХ сессий —
    # синхронный метод, т.к. внутри блокирующие SQLite-запросы и, при
    # срабатывании boredom, блокирующий HTTP-вызов к LLM.
    # ----------------------------------------------------------------------

    def run_idle_tick(self, delta_seconds: float) -> Optional[BrainIdleEvent]:
        """
        Синхронный аналог тела _idle_sleep_background_loop из main.py,
        но для ОДНОЙ сессии и без прямого print()/консольного вывода —
        вместо этого возвращает BrainIdleEvent (или None), который
        вызывающий код (bot.py) сам решает, отправлять ли пользователю.

        Продвигает brain_time на delta_seconds (реальные секунды,
        прошедшие с прошлой итерации фонового scheduler'а — см.
        config.BOT_SCHEDULER_TICK_SECONDS) и проверяет два условия
        по очереди:

            1. AWAKE и Δt с последнего сообщения/активности пользователя
               >= IDLE_SLEEP_THRESHOLD_SECONDS -> запускает
               run_sleep_cycle(), переводит BoredomDrive в SLEEPING.
               Возвращает BrainIdleEvent(kind="sleep").
            2. SLEEPING -> пересчитывает boredom; при достижении
               BOREDOM_THRESHOLD -> выбирает узел (select_proactive_node),
               генерирует проактивное сообщение через cortex (LLM).
               Возвращает BrainIdleEvent(kind="proactive") при успехе,
               либо None при graceful degradation (LLM недоступна) —
               система остаётся в SLEEPING, попытка повторится на
               следующем boredom-тике.

        В остальных случаях (ничего не произошло) возвращает None.

        УСТУПАЕТ ПОЛЬЗОВАТЕЛЮ: если сессия сейчас занята обработкой
        сообщения, тик молча пропускается, а не встаёт в очередь. Иначе
        человек ждал бы, пока фоновая фаза сна сходит в LLM (до 30 секунд
        таймаута) — при том что тик по своей природе необязателен и
        повторится через BOT_SCHEDULER_TICK_SECONDS.
        """
        if not self._lock.acquire(blocking=False):
            logger.debug(
                "[IDLE SKIP] db_path=%s: сессия занята пользователем -> тик пропущен",
                self.db_path,
            )
            return None
        try:
            return self._run_idle_tick_unlocked(delta_seconds)
        finally:
            self._lock.release()

    def _run_idle_tick_unlocked(self, delta_seconds: float) -> Optional[BrainIdleEvent]:
        """Тело фонового тика. Вызывать только под self._lock."""
        # Часы идут сами — фоновому тику двигать их нечем
        brain_time = self.clock.get_brain_time()
        idle_seconds = self.clock.seconds_since_last_activity()

        if self.boredom_drive.is_awake() and idle_seconds >= config.IDLE_SLEEP_THRESHOLD_SECONDS:
            logger.info(
                "[IDLE SLEEP] db_path=%s: бездействие %.1fs >= %.1fs -> запуск фазы сна",
                self.db_path, idle_seconds, config.IDLE_SLEEP_THRESHOLD_SECONDS,
            )
            sleep_summary = self.sleep_cycle.run_sleep_cycle(timestamp=brain_time)
            self.activation_state.clear_window()
            self.boredom_drive.enter_sleeping(brain_time)
            return BrainIdleEvent(kind="sleep", text=sleep_summary.to_report_string())

        if self.boredom_drive.is_sleeping():
            snapshot = self.boredom_drive.update(brain_time)
            logger.debug(
                "[BOREDOM] db_path=%s state=%s boredom=%.3f (t_sleep=%.1fs)",
                self.db_path, snapshot.state.value, snapshot.boredom, snapshot.seconds_since_sleep_start,
            )

            # Порог зависит от накопленной привязанности: чем ближе стал
            # человек, тем раньше организм не выдерживает молчания.
            threshold = self.boredom_drive.effective_threshold(
                self.cortex.mood.get_state().affection
            )
            if snapshot.boredom >= threshold:
                last_node_id = self.activation_state.get_last_active()
                node = self.memory.select_proactive_node(
                    last_active_node_id=last_node_id,
                    brain_time=brain_time,
                )
                proactive_message = self.cortex.generate_proactive_message(node, timestamp=brain_time)

                if proactive_message is not None:
                    self.boredom_drive.trigger_proactive()
                    logger.info(
                        "[PROACTIVE MESSAGE] db_path=%s source_node_id=%s",
                        self.db_path, proactive_message.source_node_id,
                    )
                    return BrainIdleEvent(
                        kind="proactive",
                        text=proactive_message.text,
                        source_node_id=proactive_message.source_node_id,
                    )

                logger.warning(
                    "[PROACTIVE FALLBACK] db_path=%s: не удалось сгенерировать "
                    "проактивное сообщение (LLM недоступна) — остаёмся в SLEEPING",
                    self.db_path,
                )

        return None

    def close(self) -> None:
        """
        Освобождает ресурсы (закрывает соединение с БД через Cortex).

        Тоже под блокировкой: выгрузка сессии по бездействию идёт из
        фонового тика, и закрыть SQLite посреди чужого process_message
        значило бы уронить обработку сообщения пользователя на середине.
        """
        with self._lock:
            self.cortex.close()
            logger.info("[BRAIN SESSION] Сессия закрыта (db_path=%s)", self.db_path)