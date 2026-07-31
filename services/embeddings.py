"""
================================================================================
 EMBEDDINGS.PY — Семантические векторы для поиска по смыслу
================================================================================
До этого модуля память искала ПО БУКВАМ: keyword-пересечение плюс
SequenceMatcher. Замер показывал, чем это кончается —

    в памяти "у меня есть кошка":
        "у меня есть кот"              -> 0.870  найдено
        "расскажи про кота"            -> не найдено
        "мой домашний питомец мяукает" -> не найдено
        "у меня есть кожа"             -> 0.884  НАЙДЕНО ЛУЧШЕ, ЧЕМ КОТ

то есть "кожа" побеждала "кота", потому что отличается одной буквой.
Смысла в таком поиске не было вовсе.

ВЫБОР МОДЕЛИ продиктован железом, а не вкусом. sentence-transformers тянет
torch: ~2.5 ГБ на диске и 0.5-1 ГБ RAM на процесс. На этой машине 3.7 ГБ
всего, свободно ~1.8 ГБ, и рядом работают ещё три сервиса — такой сосед
уронил бы их. Здесь используется navec (проект natasha): статические
русские векторы, 51 МБ квантованная модель, ~370 МБ RAM, без torch.

СЛУЖЕБНЫЕ СЛОВА ОТБРАСЫВАЮТСЯ, и это не мелочь. Вектор фразы — среднее по
словам, поэтому "у меня есть" забивает единственное содержательное слово:
без фильтрации "кожа" держалась на 0.863 и всё равно почти равнялась
"коту". После отбрасывания служебных — 0.222 против 0.671, порядок стал
правильным.

ДЕГРАДИРУЕТ МЯГКО: если модели нет, библиотека не установлена или
EMBEDDINGS_ENABLED=false — encode() возвращает None, а MemoryGraph.search
продолжает работать на прежнем строковом сходстве. Бот не должен падать
из-за отсутствия необязательного файла на 51 МБ.

ЗАГРУЖАЕТСЯ ЛЕНИВО: модель поднимается при первом обращении, а не при
импорте, чтобы процесс, которому семантика не нужна (тесты, стенд,
инспектор памяти), не платил за неё оперативной памятью.
================================================================================
"""

import threading
from typing import List, Optional, Sequence

import config
from storage.utils.logger import get_logger

logger = get_logger(__name__)

# Служебные слова: их вектора забивают среднее и стирают смысл фразы.
# Набор намеренно шире, чем STOP_WORDS в graph_memory (там он отсекает
# шум для keyword-поиска, здесь — чинит усреднение векторов).
_FUNCTION_WORDS = {
    "и", "в", "во", "на", "с", "со", "по", "к", "у", "из", "за", "от", "до",
    "для", "что", "как", "это", "то", "я", "ты", "он", "она", "мы", "вы",
    "они", "не", "но", "а", "же", "бы", "ли", "или", "тот", "его", "её", "ее",
    "их", "есть", "быть", "мой", "моя", "моё", "мне", "меня", "мной", "твой",
    "тебя", "тебе", "про", "о", "об", "при", "же", "уж", "вот", "так", "там",
    "тут", "ещё", "еще", "уже", "бы", "чтобы", "если", "когда", "где",
}

_model = None
_load_failed = False
_lock = threading.Lock()


def _load_model():
    """
    Лениво поднимает модель. Повторные неудачи не ретраятся: если файла
    нет, он не появится сам, а сыпать ошибкой на каждое сообщение незачем.
    """
    global _model, _load_failed

    if _model is not None or _load_failed:
        return _model

    with _lock:
        if _model is not None or _load_failed:
            return _model

        if not config.EMBEDDINGS_ENABLED:
            _load_failed = True
            logger.info("[EMBEDDINGS] Отключены настройкой — поиск останется строковым")
            return None

        try:
            from navec import Navec
        except ImportError:
            _load_failed = True
            logger.warning(
                "[EMBEDDINGS] navec не установлен — поиск остаётся строковым. "
                "pip install navec numpy"
            )
            return None

        try:
            _model = Navec.load(config.EMBEDDING_MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            _load_failed = True
            logger.warning(
                "[EMBEDDINGS] Не удалось загрузить модель (%s): %s — "
                "поиск остаётся строковым",
                config.EMBEDDING_MODEL_PATH, exc,
            )
            return None

        logger.info("[EMBEDDINGS] Модель загружена: %s", config.EMBEDDING_MODEL_PATH)
        return _model


def is_available() -> bool:
    """Работает ли семантический поиск (без побочной загрузки модели)."""
    return _load_model() is not None


def _content_words(text: str) -> List[str]:
    """
    Содержательные слова фразы. Служебные отбрасываются — иначе среднее по
    векторам определяется ими, а не смыслом (см. шапку модуля).

    Если содержательных не осталось (фраза целиком служебная — "а что если"),
    возвращаются все слова: лучше слабый вектор, чем никакого.
    """
    words = [w.strip(".,!?;:()\"'«»").lower() for w in text.split()]
    words = [w for w in words if w]
    content = [w for w in words if w not in _FUNCTION_WORDS]
    return content or words


def encode(text: str):
    """
    Вектор смысла фразы, либо None — если семантика недоступна или в тексте
    не нашлось ни одного known слова.

    None это НЕ ошибка: вызывающий код обязан уметь работать без семантики
    (см. MemoryGraph.search), потому что модель необязательна.
    """
    if not text or not text.strip():
        return None

    model = _load_model()
    if model is None:
        return None

    import numpy as np

    vectors = [model[w] for w in _content_words(text) if w in model]
    if not vectors:
        return None

    return np.mean(vectors, axis=0)


def cosine(a, b) -> float:
    """Косинусная близость двух векторов, 0.0 при вырожденных входах."""
    if a is None or b is None:
        return 0.0

    import numpy as np

    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 0.0:
        return 0.0
    return float(np.dot(a, b) / norm)


def to_blob(vector) -> Optional[bytes]:
    """Вектор -> BLOB для хранения в SQLite."""
    if vector is None:
        return None
    import numpy as np
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: Optional[bytes]):
    """BLOB из SQLite -> вектор. None, если поля нет или оно пустое."""
    if not blob:
        return None
    import numpy as np
    return np.frombuffer(blob, dtype=np.float32)


def similarity(text: str, blob: Optional[bytes]) -> float:
    """Удобная обёртка: близость текста к сохранённому вектору узла."""
    return cosine(encode(text), from_blob(blob))
