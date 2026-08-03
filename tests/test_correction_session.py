"""
================================================================================
 TESTS/TEST_CORRECTION_SESSION.PY — Поправка проходит через организм целиком
================================================================================
Здесь проверяется ВИТРИНА, а не библиотека: тесты строят BrainSession, то
есть организм с миндалиной, восприятием и речью. Библиотечная половина
проверок поправок живёт в test_contradictions.py репозитория selective-memory
и работает без всякого приложения.

Разделение не косметическое. Витрина считает эмоцию сама и зовёт
save_connection напрямую, минуя Memory.observe(), поэтому у неё другой
спайк-гейт и другие числа. Смешивать эти две проверки в одном файле значило
бы выдавать поведение приложения за поведение пакета.
================================================================================
"""

import pytest

from selectivemem import embeddings

# Пропуск объявлен ЗДЕСЬ, а не импортирован из библиотечного теста: файл
# уезжает в репозиторий витрины, и импорт через границу репозиториев
# сломался бы при первом же переносе.
requires_model = pytest.mark.skipif(
    not embeddings.is_available(),
    reason="вытеснение опирается на семантику; без модели оно отключено",
)


# ---------------------------------------------------------------------------
# Поправка должна ПОПАСТЬ в память, даже будучи неудивительной
#
# Найдено живым использованием: пользователь поправил имя собаки, а в
# памяти осталось старое. Причина оказалась не в вытеснении, а на шаг
# раньше — поправка вообще не записывалась. Замер на рабочей базе:
#     "мою собаку зовут бобик"    -> плотность 0.165 при пороге 0.35
#     "мою собаку зовут бобик!!!" -> 0.263, всё ещё мимо
# Поправка по природе НЕ УДИВИТЕЛЬНА: та же фраза, одно слово другое.
# Спайк-гейт видел знакомое и не пускал самое важное сообщение.
# ---------------------------------------------------------------------------
def _session(tmp_path):
    import sys
    sys.argv = ["x"]
    sys.path.insert(0, "tools")
    from simulate_learning import install_llm_stub

    install_llm_stub()
    from core.brain_session import BrainSession

    return BrainSession(db_path=str(tmp_path / "brain.db"))


@requires_model
def test_correction_is_written_even_without_a_spike(tmp_path):
    session = _session(tmp_path)
    try:
        session.process_message("мою собаку зовут Рекс")
        # Поправка спокойная: ни капса, ни восклицаний, слова знакомые
        result = session.process_message("мою собаку зовут Бобик")

        assert result.debug.get("memory_written"), (
            "поправка обязана попасть в память, даже если спайк не сработал — "
            "иначе вытеснять устаревшее будет нечем"
        )
    finally:
        session.close()


@requires_model
def test_correction_outranks_the_stale_fact(tmp_path):
    """Ровно то, что наблюдалось у пользователя на живом боте."""
    session = _session(tmp_path)
    try:
        session.process_message("мою собаку зовут Рекс")
        session.process_message("мою собаку зовут Бобик")

        found = session.memory.search(
            "как зовут мою собаку", top_k=2,
            timestamp=session.clock.get_brain_time(), with_associations=False,
        )

        assert found, "хоть что-то должно найтись"
        assert "обик" in found[0].context, (
            f"первым обязан идти актуальный факт, а вернулось {found[0].context!r}"
        )
    finally:
        session.close()


@requires_model
def test_correction_inherits_the_standing_of_what_it_replaces(tmp_path):
    """
    Вес узла обычно равен плотности сообщения, но поправка неудивительна и
    рождалась бы слабой (0.17 против 0.52 у устаревшего) — запись бы
    состоялась, а поиск всё равно выигрывал бы старый факт. Новая версия
    наследует вес предшественника: это тот же факт, обновлённый.
    """
    session = _session(tmp_path)
    try:
        session.process_message("мою собаку зовут Рекс")
        session.process_message("мою собаку зовут Бобик")

        rows = {
            r["context"]: r["weight"]
            for r in session.memory.db._conn.execute(
                "SELECT context, weight FROM nodes WHERE node_type='episodic'"
            )
        }
        fresh = next(v for k, v in rows.items() if "обик" in k)
        stale = next(v for k, v in rows.items() if "екс" in k)

        assert fresh > stale, "актуальная версия должна весить больше вытесненной"
    finally:
        session.close()
