"""
================================================================================
 TOOLS/MINIAPP.PY — Telegram Mini App: видимая память внутри мессенджера
================================================================================
Отдаёт страницу из tools/render_memory.py как Telegram Mini App, но КАЖДОМУ
ПОЛЬЗОВАТЕЛЮ — ТОЛЬКО ЕГО СОБСТВЕННЫЙ МОЗГ.

Это главное требование, а не удобство. В storage/brains лежит по базе на
каждого telegram-пользователя, и в них — реальные переписки с ботом.
Отдавать такую страницу без проверки, кто её открыл, значит опубликовать
чужие личные разговоры: адрес рано или поздно утечёт, а угадать чужой
user_id тривиально.

Поэтому используется штатный механизм Telegram: initData подписывается
токеном бота (HMAC-SHA256), сервер проверяет подпись и достаёт из неё
достоверный user_id. Подделать его, не зная токена, нельзя.

Схема работы:
    GET  /              -> лёгкая страница-загрузчик: подключает
                           telegram-web-app.js, забирает initData и
                           запрашивает у сервера свою карту памяти.
    POST /render        -> проверяет подпись initData, рендерит мозг
                           именно этого пользователя, отдаёт HTML.
    GET  /health        -> проверка живости для systemd/nginx.

Сервис слушает ТОЛЬКО 127.0.0.1: наружу его пускает nginx, который держит
TLS. Сам он публично не доступен.

Базы открываются строго на чтение (см. render_memory.load_snapshot), так
что мини-апп безопасно работает параллельно с живым ботом, который в эти
же файлы пишет.

Запуск:
    python tools/miniapp.py                 # 127.0.0.1:8002
    python tools/miniapp.py --port 8080
================================================================================
"""

import argparse
import hashlib
import hmac
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiohttp import web  # noqa: E402

import config  # noqa: E402
from storage.utils.logger import get_logger  # noqa: E402
from tools.render_memory import load_snapshot, render_html  # noqa: E402

logger = get_logger(__name__)

# Максимальный возраст подписи. Telegram кладёт в initData auth_date;
# просроченные подписи отвергаются, иначе перехваченный однажды initData
# работал бы вечно.
MAX_AUTH_AGE_SECONDS = 24 * 3600


# --------------------------------------------------------------------------
# Проверка подписи Telegram
# --------------------------------------------------------------------------

def verify_init_data(init_data: str, bot_token: str) -> Optional[int]:
    """
    Проверяет подпись Telegram WebApp initData и возвращает достоверный
    telegram user_id, либо None, если подпись не сошлась.

    Алгоритм (документация Telegram, "Validating data received via the
    Mini App"):
        secret_key = HMAC_SHA256(key="WebAppData", msg=bot_token)
        ожидаемый  = HMAC_SHA256(key=secret_key, msg=data_check_string)
    где data_check_string — все поля кроме hash, отсортированные по имени
    и склеенные через \\n в виде "ключ=значение".

    Сравнение хэшей — через compare_digest: обычное == сравнивает строки
    посимвольно с ранним выходом и теоретически утекает информацию по
    времени.
    """
    if not init_data or not bot_token:
        return None

    try:
        pairs = dict(parse_qsl(init_data, strict_parsing=True))
    except ValueError:
        logger.warning("[MINIAPP] initData не разбирается как query-строка")
        return None

    received_hash = pairs.pop("hash", None)
    if not received_hash:
        return None

    data_check_string = "\n".join(f"{k}={pairs[k]}" for k in sorted(pairs))

    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected_hash = hmac.new(
        secret_key, data_check_string.encode(), hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_hash, received_hash):
        logger.warning("[MINIAPP] Подпись initData не сошлась — запрос отклонён")
        return None

    # Просроченная подпись не должна работать вечно
    try:
        auth_age = time.time() - float(pairs.get("auth_date", 0))
    except (TypeError, ValueError):
        return None
    if auth_age > MAX_AUTH_AGE_SECONDS:
        logger.warning("[MINIAPP] initData просрочен (%.0f ч)", auth_age / 3600)
        return None

    # user приходит JSON-строкой внутри query — достаём id без json.loads
    # только если он там есть
    user_raw = pairs.get("user")
    if not user_raw:
        return None
    try:
        import json
        return int(json.loads(user_raw)["id"])
    except (ValueError, KeyError, TypeError):
        logger.warning("[MINIAPP] В initData нет разбираемого user.id")
        return None


# --------------------------------------------------------------------------
# Страница-загрузчик
# --------------------------------------------------------------------------

BOOTSTRAP_PAGE = """<!doctype html>
<html lang="ru"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Память мозга</title>
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<style>
  body { margin:0; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--tg-theme-bg-color,#fbfbfd); color:var(--tg-theme-text-color,#1c1e21); }
  .msg { padding:48px 24px; text-align:center; color:var(--tg-theme-hint-color,#6b7280); }
</style>
</head><body>
<div id="root"><p class="msg">Собираю карту памяти…</p></div>
<script>
(function () {
  var tg = window.Telegram && window.Telegram.WebApp;
  var root = document.getElementById('root');

  if (!tg || !tg.initData) {
    root.innerHTML = '<p class="msg">Эту страницу нужно открывать из Telegram — ' +
                     'подпись пользователя приходит только оттуда.</p>';
    return;
  }
  tg.ready();
  tg.expand();

  // Путь считается от текущего, а не задаётся жёстко: снаружи страница
  // живёт за префиксом (/memory/), который nginx срезает перед проксированием.
  // Голое 'render' сломалось бы при открытии без завершающего слэша.
  var base = location.pathname.replace(/\\/?$/, '/');

  fetch(base + 'render', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: 'init_data=' + encodeURIComponent(tg.initData)
  })
  .then(function (r) { return r.text().then(function (t) { return { ok: r.ok, text: t }; }); })
  .then(function (res) {
    if (!res.ok) { root.innerHTML = '<p class="msg">' + res.text + '</p>'; return; }
    root.innerHTML = res.text;
  })
  .catch(function () {
    root.innerHTML = '<p class="msg">Не удалось получить карту памяти. Попробуй позже.</p>';
  });
})();
</script>
</body></html>
"""


# --------------------------------------------------------------------------
# Обработчики
# --------------------------------------------------------------------------

async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=BOOTSTRAP_PAGE, content_type="text/html")


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_render(request: web.Request) -> web.Response:
    """Проверяет подпись и отдаёт карту памяти ИМЕННО ЭТОГО пользователя."""
    form = await request.post()
    init_data = str(form.get("init_data", ""))

    user_id = verify_init_data(init_data, config.TELEGRAM_BOT_TOKEN)
    if user_id is None:
        # Намеренно не уточняем, что именно не сошлось
        return web.Response(
            status=403, text="Не удалось подтвердить, что это твой профиль.",
            content_type="text/html",
        )

    db_path = Path(config.BRAIN_DB_DIR) / f"{user_id}.db"
    if not db_path.exists():
        logger.info("[MINIAPP] user_id=%s: мозга ещё нет", user_id)
        return web.Response(
            text="<p style='padding:48px 24px;text-align:center'>"
                 "Мы ещё не знакомы — напиши мне что-нибудь, и я начну запоминать.</p>",
            content_type="text/html",
        )

    try:
        snapshot = load_snapshot(str(db_path))
        html = render_html(snapshot, include_lexical=False)
    except Exception:  # noqa: BLE001
        # SQLite может быть кратковременно занят живым ботом, который
        # пишет в эту же базу — это не повод показывать пользователю трейс
        logger.exception("[MINIAPP] Не удалось отрендерить мозг user_id=%s", user_id)
        return web.Response(
            status=503,
            text="Сейчас я занят мыслями — загляни через минуту.",
            content_type="text/html",
        )

    logger.info(
        "[MINIAPP] user_id=%s: карта отдана (узлов %d, словарь %d)",
        user_id, len(snapshot.nodes), snapshot.mastered_words,
    )
    return web.Response(text=html, content_type="text/html")


def build_app() -> web.Application:
    app = web.Application()
    app.add_routes([
        web.get("/", handle_index),
        web.post("/render", handle_render),
        web.get("/health", handle_health),
    ])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram Mini App: карта памяти мозга")
    parser.add_argument("--host", default="127.0.0.1", help="только localhost, наружу пускает nginx")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    if not config.TELEGRAM_BOT_TOKEN:
        sys.exit(
            "TELEGRAM_BOT_TOKEN не задан в .env — без него невозможно проверить "
            "подпись Telegram, а без проверки мини-апп отдавал бы чужие переписки."
        )

    logger.info("[MINIAPP] Запуск на http://%s:%d", args.host, args.port)
    web.run_app(build_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
