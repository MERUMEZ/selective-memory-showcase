"""
Тесты на проверку подписи Telegram Mini App (tools/miniapp.verify_init_data).

Это код безопасности, а не удобства. В storage/brains лежит по базе на
каждого пользователя, и в них реальные переписки с ботом. Мини-апп отдаёт
карту памяти по user_id, взятому ИЗ ПОДПИСАННЫХ данных Telegram — если
проверка подписи ослабнет, любой сможет прочитать чужие разговоры, просто
подставив чужой id в запрос.

Поэтому здесь проверяется не только счастливый путь, но и каждый способ
подделки, который приходит в голову.
"""
import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import pytest

from tools.miniapp import MAX_AUTH_AGE_SECONDS, verify_init_data

BOT_TOKEN = "123456:TEST-TOKEN-НЕ-НАСТОЯЩИЙ"


def make_init_data(user_id: int, token: str = BOT_TOKEN, auth_date: float = None) -> str:
    """Собирает корректно подписанный initData — как это делает Telegram."""
    auth_date = int(auth_date if auth_date is not None else time.time())
    fields = {
        "auth_date": str(auth_date),
        "query_id": "AAF_test",
        "user": json.dumps({"id": user_id, "first_name": "Паша"}, ensure_ascii=False),
    }
    data_check_string = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, data_check_string.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


# ---------------------------------------------------------------------------
# Счастливый путь
# ---------------------------------------------------------------------------
def test_valid_signature_returns_user_id():
    assert verify_init_data(make_init_data(424242424), BOT_TOKEN) == 424242424


# ---------------------------------------------------------------------------
# Способы подделки — все обязаны быть отвергнуты
# ---------------------------------------------------------------------------
def test_rejects_foreign_user_id_substitution():
    """
    Главная угроза: подписанные данные одного пользователя, но с
    подменённым id в надежде получить чужой мозг.
    """
    init_data = make_init_data(424242424)
    tampered = init_data.replace("424242424", "999999999")

    assert verify_init_data(tampered, BOT_TOKEN) is None


def test_rejects_signature_from_another_bot_token():
    """Подпись, сделанная чужим токеном, не должна проходить."""
    init_data = make_init_data(555, token="000000:ДРУГОЙ-ТОКЕН")
    assert verify_init_data(init_data, BOT_TOKEN) is None


def test_rejects_unsigned_data():
    unsigned = urlencode({
        "auth_date": str(int(time.time())),
        "user": json.dumps({"id": 555}),
    })
    assert verify_init_data(unsigned, BOT_TOKEN) is None


def test_rejects_expired_signature():
    """
    Просроченный initData не должен работать вечно: иначе однажды
    перехваченная ссылка давала бы доступ бессрочно.
    """
    stale = make_init_data(555, auth_date=time.time() - MAX_AUTH_AGE_SECONDS - 60)
    assert verify_init_data(stale, BOT_TOKEN) is None

    fresh = make_init_data(555, auth_date=time.time() - 60)
    assert verify_init_data(fresh, BOT_TOKEN) == 555


@pytest.mark.parametrize(
    "bad_input",
    ["", "   ", "не-query-строка", "hash=deadbeef", "user=%7B%7D&hash=abc"],
    ids=["пусто", "пробелы", "мусор", "только-hash", "пустой-user"],
)
def test_rejects_malformed_input(bad_input):
    """Кривой вход должен давать None, а не исключение на весь запрос."""
    assert verify_init_data(bad_input, BOT_TOKEN) is None


def test_rejects_when_token_is_missing():
    """
    Без токена подпись проверить нечем — значит доступ запрещён.
    Раньше здесь был бы соблазн "пропустить, раз проверять нечем".
    """
    assert verify_init_data(make_init_data(555), "") is None
    assert verify_init_data(make_init_data(555), None) is None


def test_signature_covers_every_field():
    """
    Подпись должна покрывать ВСЕ поля, а не только user: иначе можно было
    бы менять query_id/auth_date, сохраняя валидный hash.
    """
    init_data = make_init_data(555)
    tampered = init_data.replace("AAF_test", "AAF_evil")
    assert verify_init_data(tampered, BOT_TOKEN) is None
