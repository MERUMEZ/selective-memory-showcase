"""
================================================================================
 TOOLS/SIMULATE_LEARNING.PY — Измерительный стенд контура освоения языка
================================================================================
Отвечает на один вопрос: СКОЛЬКО СООБЩЕНИЙ нужно, чтобы бот вышел из
довербальной стадии (Stage 0) и заговорил фразами?

Без этого стенда любая правка порогов/скоростей угасания — угадывание:
эффект виден только через недели реального общения, а к тому времени
пользователь уже бросил (см. storage/brains/*.db — 2 освоенных слова
из 40 услышанных за месяц).

Прогоняет корпус реплик через НАСТОЯЩИЙ BrainSession на in-memory SQLite:
никакой сети, никаких денег, весь пайплайн (Perception -> Amygdala ->
Cortex -> MemoryGraph -> decay -> sleep) работает как в проде.

Запуск:
    python tools/simulate_learning.py
    python tools/simulate_learning.py --messages 400 --words-per-message 12
    python tools/simulate_learning.py --seed 7 --verbose

Читает конфиг из config.py как есть — то есть меряет ТЕКУЩЕЕ состояние
системы, а не эталон. Прогнать до правок и после — и сравнить.
================================================================================
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

# Скрипт лежит в tools/, поэтому в sys.path попадает tools/, а не корень
# проекта — добавляем корень явно, иначе `import config` не найдётся.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ГЛУШИМ ЛОГИ ДО ИМПОРТА config. storage/utils/logger.py читает
# config.LOG_LEVEL в момент get_logger(), а get_logger вызывается на
# уровне модуля в core.*/memory.* — то есть при импорте. Ставить уровень
# после импортов уже поздно: 300 сообщений дают ~2 МБ INFO-логов, в
# которых таблица стенда полностью тонет. Флаг --logs возвращает их.
if "--logs" not in sys.argv:
    os.environ["LOG_LEVEL"] = "ERROR"

import config  # noqa: E402


# --------------------------------------------------------------------------
# Заглушка LLM
# --------------------------------------------------------------------------
# ЛОВУШКА: generate_llm_response импортируется ПО ИМЕНИ в трёх модулях
# (`from services.llm import generate_llm_response`), поэтому подмена
# services.llm.generate_llm_response на уже связанные ссылки не подействует.
# Патчить нужно атрибут в КАЖДОМ импортирующем модуле.
# --------------------------------------------------------------------------
LLM_MODULES = ("core.cortex", "memory.graph_memory", "memory.sleep_cycle")

_llm_calls = {"count": 0}


def _fake_llm(messages, system_prompt=None, max_tokens=None) -> str:
    """
    Детерминированная заглушка вместо OpenRouter. Возвращает короткую
    фразу из слов последнего user-сообщения — то, что примерно выдала бы
    LLM на ранней стадии, без обращения к сети.
    """
    _llm_calls["count"] += 1
    last_user = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    words = last_user.split()[:6]
    return " ".join(words) if words else "да"


def install_llm_stub() -> None:
    import importlib

    for module_name in LLM_MODULES:
        module = importlib.import_module(module_name)
        if not hasattr(module, "generate_llm_response"):
            raise RuntimeError(
                f"{module_name} больше не импортирует generate_llm_response — "
                "заглушка стенда устарела, проверь импорты"
            )
        module.generate_llm_response = _fake_llm


# --------------------------------------------------------------------------
# Корпус: как реально разговаривает человек, обучающий бота
# --------------------------------------------------------------------------
# Не случайный набор слов, а осмысленные короткие реплики на нескольких
# бытовых темах. При выборке с возвращением даёт естественное
# Zipf-подобное распределение частот: служебные слова повторяются часто,
# содержательные — редко. Ровно та статистика, которая ломает освоение.
CORPUS: List[str] = [
    "привет как дела",
    "меня зовут Паша",
    "я твой учитель",
    "ты цифровой ребёнок",
    "давай учиться говорить",
    "это кошка она мягкая",
    "кошка говорит мяу",
    "это собака она большая",
    "собака говорит гав",
    "мама это женщина родитель",
    "папа это мужчина родитель",
    "солнце светит днём",
    "луна светит ночью",
    "вода мокрая и холодная",
    "огонь горячий не трогай",
    "хлеб это еда его едят",
    "молоко белое его пьют",
    "дом это место где живут",
    "окно нужно чтобы смотреть",
    "дверь нужна чтобы входить",
    "красный это цвет крови",
    "синий это цвет неба",
    "зелёный это цвет травы",
    "один два три четыре пять",
    "большой и маленький это размер",
    "хорошо это когда приятно",
    "плохо это когда неприятно",
    "я хочу есть",
    "ты хочешь спать",
    "мы идём гулять",
    "сегодня тёплый день",
    "вчера был дождь",
    "завтра будет солнце",
    "книга это много букв",
    "буква это знак",
    "слово это много букв вместе",
    "говори громче я слушаю",
    "повтори это слово",
    "запомни что кошка это животное",
    "запомни что дом это здание",
    "молодец ты хорошо сказал",
    "нет это неправильно",
    "да именно так правильно",
    "попробуй ещё раз",
    "не бойся говори",
    "что ты видишь сейчас",
    "как ты себя чувствуешь",
    "расскажи мне о себе",
    "ты меня понимаешь",
    "я тебя слушаю внимательно",
]


# Длинный хвост Ципфа: слова, которые в реальной речи встречаются
# один-два раза и НИКОГДА не закрепляются. Без них корпус из 50 фраз даёт
# всего ~130 уникальных слов, каждое из которых повторяется постоянно, —
# стенд получается неправдоподобно лёгким и льстит системе. В реальном
# языке основная масса словоформ приходится именно на такой хвост.
RARE_WORDS: List[str] = [
    "велосипед", "холодильник", "путешествие", "библиотека", "апельсин",
    "карандаш", "телевизор", "остановка", "полотенце", "чемодан",
    "лестница", "подоконник", "варежки", "скатерть", "будильник",
    "зеркало", "кастрюля", "простыня", "занавеска", "табуретка",
    "перчатки", "расчёска", "ботинки", "кошелёк", "зонтик",
    "тетрадь", "линейка", "пенал", "рюкзак", "фломастер",
    "качели", "песочница", "самокат", "конструктор", "пирамидка",
    "малина", "черника", "абрикос", "виноград", "смородина",
]


def build_message_stream(
    n_messages: int,
    words_per_message: int,
    rng: random.Random,
    tail_ratio: float = 0.3,
) -> List[str]:
    """
    Собирает поток сообщений заданной средней длины из корпуса.
    Короткие реплики склеиваются, пока не наберётся нужная длина —
    так можно менять «ширину канала ввода» и смотреть, как это влияет
    на скорость освоения.

    tail_ratio — доля слов из длинного хвоста Ципфа (редкие слова,
    встречающиеся один-два раза за весь диалог). Именно они составляют
    основную массу словоформ в живой речи и почти никогда не доходят до
    порога освоения. При tail_ratio=0 стенд меряет идеализированный
    режим «учитель терпеливо повторяет одни и те же 50 фраз».
    """
    stream: List[str] = []
    for _ in range(n_messages):
        parts: List[str] = []
        length = 0
        while length < words_per_message:
            if rng.random() < tail_ratio:
                parts.append(rng.choice(RARE_WORDS))
                length += 1
            else:
                phrase = rng.choice(CORPUS)
                parts.append(phrase)
                length += len(phrase.split())
        stream.append(" ".join(parts))
    return stream


# --------------------------------------------------------------------------
# Прогон
# --------------------------------------------------------------------------

def run_simulation(
    n_messages: int,
    words_per_message: int,
    seed: int,
    report_every: int,
    verbose: bool,
    session_length: int = 0,
    gap_hours: float = 0.0,
    tail_ratio: float = 0.3,
) -> Optional[int]:
    """
    Прогоняет n_messages через свежий BrainSession и печатает кривую
    освоения словаря. Возвращает номер сообщения, на котором система
    впервые вышла из Stage 0, либо None, если так и не вышла.

    session_length/gap_hours моделируют РЕАЛЬНЫЙ режим общения: человек
    пишет пачку сообщений, уходит на несколько часов, возвращается.
    Это принципиально: BrainSession.__init__ ставит brain_time = time.time(),
    поэтому после выгрузки сессии часы прыгают на реальное «сейчас», а
    last_decayed_at узлов остаётся старым — и apply_decay получает dt
    размером во всю паузу. Без этого стенд меряет лабораторный режим
    «300 сообщений подряд», в котором отказ не воспроизводится.
    """
    install_llm_stub()

    # _resolve_speech_stage вероятностный (зона смешения на границах),
    # поэтому фиксируем seed — иначе прогоны несравнимы между собой.
    random.seed(seed)
    rng = random.Random(seed)

    from core.brain_session import BrainSession

    session = BrainSession(db_path=":memory:")
    stream = build_message_stream(n_messages, words_per_message, rng, tail_ratio)

    print("=" * 74)
    print(" СТЕНД ОБУЧЕНИЯ — сколько сообщений до первой связной фразы")
    print("=" * 74)
    print(
        f" Пороги стадий      : Stage0<{config.SPEECH_STAGE_0_MAX_VOCAB} "
        f"Stage1<{config.SPEECH_STAGE_1_MAX_VOCAB} Stage2<{config.SPEECH_STAGE_2_MAX_VOCAB}"
    )
    print(
        f" Освоение слова     : старт {config.WORD_NODE_INITIAL_WEIGHT} "
        f"+{config.WORD_NODE_REINFORCE_STEP}/повтор, порог {config.VOCABULARY_MASTERY_MIN_WEIGHT}"
    )
    lexical_t0 = getattr(config, "LEXICAL_AGE_T0", None)
    lexical_desc = (
        f"словарь T0={lexical_t0 / 86400:.0f}сут"
        if lexical_t0
        else "словарь угасает как эпизоды (единая шкала)"
    )
    print(
        f" Угасание           : DECAY_RATE={config.DECAY_RATE} "
        f"эпизоды T0={config.AGE_T0 / 3600:.0f}ч, {lexical_desc}"
    )
    print(
        f" Поток              : {n_messages} сообщений по ~{words_per_message} слов, "
        f"хвост Ципфа {tail_ratio:.0%}, seed={seed}"
    )
    if session_length and gap_hours:
        print(f" Режим общения      : сессии по {session_length} сообщ, пауза {gap_hours}ч между ними")
    else:
        print(" Режим общения      : подряд, без пауз (лабораторный)")
    print("-" * 90)
    print(
        f"{'сообщ':>6} {'услышано':>9} {'освоено':>8} {'стадия':>7} "
        f"{'удивл':>7} {'спайков':>8} {'эпизод':>7} {'узлов':>7}  источник"
    )
    print("-" * 90)

    stage_1_reached_at: Optional[int] = None
    # Наблюдаемые для проверки главного эффекта правки: удивление и
    # частота спайков должны ПАДАТЬ по мере взросления, а не быть
    # константой. Копятся за интервал между строками отчёта.
    window_surprise: List[float] = []
    window_spikes = 0

    for i, text in enumerate(stream, start=1):
        # Пауза между сессиями: человек ушёл, сессия выгрузилась, при
        # возврате brain_time = time.time(). Продвигаем часы на всю паузу —
        # следующий apply_decay внутри process_message увидит этот dt.
        if session_length and gap_hours and i > 1 and (i - 1) % session_length == 0:
            before = session.memory.get_vocabulary_size()
            # Часы — чистая функция настенного времени, поэтому пауза
            # моделируется промоткой, а не ручным сложением
            session.clock.simulate_elapsed_wall_seconds(gap_hours * 3600.0)
            session.memory.apply_decay(now=session.clock.get_brain_time())
            after = session.memory.get_vocabulary_size()
            print(
                f"{'':>6} {'':>9} {before:>4}->{after:<3} {'':>7} {session.memory.count_nodes():>7}"
                f"  --- пауза {gap_hours}ч ---"
            )

        response = session.process_message(text)
        source = response.debug.get("response_source", "?")

        window_surprise.append(response.debug.get("perplexity", 0.0))
        if response.debug.get("spike_triggered"):
            window_spikes += 1

        mastered = session.memory.get_vocabulary_size()
        stage = session.cortex._resolve_speech_stage(mastered)

        if stage_1_reached_at is None and stage >= 1:
            stage_1_reached_at = i
            print(
                f"{i:>6} {session.memory.get_exposed_vocabulary_size():>9} {mastered:>8} "
                f"{stage:>7} {'':>7} {'':>8} {'':>7} {session.memory.count_nodes():>7}"
                f"  <<< ВЫХОД ИЗ STAGE 0"
            )

        if i % report_every == 0 or i == len(stream):
            avg_surprise = sum(window_surprise) / len(window_surprise) if window_surprise else 0.0
            episodic = session.memory.db.count_nodes_by_type("episodic")
            print(
                f"{i:>6} {session.memory.get_exposed_vocabulary_size():>9} {mastered:>8} "
                f"{stage:>7} {avg_surprise:>7.3f} {window_spikes:>8} {episodic:>7} "
                f"{session.memory.count_nodes():>7}  {source}"
            )
            if verbose:
                print(f"        ответ: {response.text[:60]!r}")
            window_surprise = []
            window_spikes = 0

    print("-" * 90)
    mastered = session.memory.get_vocabulary_size()
    exposed = session.memory.get_exposed_vocabulary_size()

    if stage_1_reached_at is not None:
        print(f" ИТОГ: Stage 1 достигнут на сообщении {stage_1_reached_at}")
    else:
        need = config.SPEECH_STAGE_0_MAX_VOCAB
        print(
            f" ИТОГ: Stage 0 НЕ пройден за {n_messages} сообщений "
            f"(освоено {mastered} из необходимых {need})"
        )
        if mastered:
            est = n_messages * need / mastered
            print(f"       экстраполяция: потребовалось бы ~{est:.0f} сообщений")

    print(f" Словарь: освоено {mastered} / услышано {exposed}")
    print(
        f" Узлов в графе: {session.memory.count_nodes()} "
        f"(из них эпизодических: {session.memory.db.count_nodes_by_type('episodic')}), "
        f"вызовов LLM: {_llm_calls['count']}"
    )
    print("=" * 90)

    session.close()
    return stage_1_reached_at


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Измеряет, за сколько сообщений бот выходит из довербальной стадии"
    )
    parser.add_argument("--messages", type=int, default=300, help="сколько сообщений прогнать")
    parser.add_argument(
        "--words-per-message", type=int, default=7,
        help="средняя длина сообщения в словах (ширина канала ввода)",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed для воспроизводимости")
    parser.add_argument("--report-every", type=int, default=20, help="шаг строк в таблице")
    parser.add_argument("--verbose", action="store_true", help="показывать ответы бота")
    parser.add_argument(
        "--logs", action="store_true",
        help="не глушить INFO-логи мозга (по умолчанию глушатся, см. верх файла)",
    )
    parser.add_argument(
        "--session-length", type=int, default=0,
        help="сколько сообщений подряд в одной сессии (0 = без пауз)",
    )
    parser.add_argument(
        "--gap-hours", type=float, default=0.0,
        help="пауза в часах между сессиями (реальный режим: ночь = 8, выходные = 48)",
    )
    parser.add_argument(
        "--tail-ratio", type=float, default=0.3,
        help="доля редких слов (длинный хвост Ципфа); 0 = идеализированный корпус",
    )
    args = parser.parse_args()

    run_simulation(
        n_messages=args.messages,
        words_per_message=args.words_per_message,
        seed=args.seed,
        report_every=args.report_every,
        verbose=args.verbose,
        session_length=args.session_length,
        gap_hours=args.gap_hours,
        tail_ratio=args.tail_ratio,
    )


if __name__ == "__main__":
    main()
