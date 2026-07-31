# Развёртывание Mini App «Карта памяти»

Мини-апп отдаёт карту памяти мозга внутри Telegram. Каждый пользователь
видит **только свой** мозг: user_id берётся из подписи Telegram
(`initData`, HMAC-SHA256 на токене бота), подделать его нельзя.

Сервис слушает только `127.0.0.1` — наружу его пускает nginx, который
держит TLS. Сам он публично недоступен.

Все шаги ниже требуют `sudo`.

---

## 1. Systemd-сервис

```bash
sudo cp /var/www/mindnumbness/deploy/mindnumbness-miniapp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mindnumbness-miniapp
systemctl status mindnumbness-miniapp --no-pager
```

Проверка, что поднялся:

```bash
curl -s http://127.0.0.1:8002/health     # -> {"status": "ok"}
```

## 2. nginx

В `/etc/nginx/sites-available/tyndex`, ВНУТРИ существующего блока
`server { ... }` (рядом с `location /api/`), добавить:

```nginx
    # 3. Mini App «Карта памяти» (mindnumbness)
    location = /memory {
        return 301 /memory/;
    }

    location /memory/ {
        proxy_pass http://127.0.0.1:8002/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
```

Завершающий слэш в `proxy_pass http://127.0.0.1:8002/` обязателен: он
срезает префикс `/memory/`, и сервис получает пути `/` и `/render`, как
если бы работал в корне. Редирект с `/memory` на `/memory/` нужен, чтобы
относительные запросы страницы не уехали в корень домена.

Применить:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

`nginx -t` проверяет конфиг ДО перезагрузки — если он не пройдёт, ничего
не применится и работающий сайт не пострадает.

## 3. Кнопка в BotFather

`@BotFather` → `/mybots` → ваш бот → **Bot Settings** → **Menu Button** →
**Configure menu button**, и указать:

```
https://mindnumbness.ru:8443/memory/
```

---

## Проверка

Открыть кнопку в Telegram — должна появиться карта памяти с вашим
словарём и графом.

Снаружи, без Telegram, страница бесполезна: без подписи `/render`
отвечает `403`, а сама страница честно скажет, что её нужно открывать из
мессенджера.

```bash
# без подписи -> 403
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
     https://mindnumbness.ru:8443/memory/render -d 'init_data=подделка'
```

## Обновление после git pull

```bash
sudo systemctl restart mindnumbness-miniapp
```

## Если что-то пошло не так

```bash
journalctl -u mindnumbness-miniapp -n 50 --no-pager
```

Мини-апп читает те же базы, в которые пишет живой бот, но открывает их
строго на чтение (`?mode=ro`), поэтому останавливать бота не нужно и
повредить данные он не может.
