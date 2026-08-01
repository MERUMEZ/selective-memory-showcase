"""
================================================================================
 TOOLS/RENDER_MEMORY.PY — Видимая память: снимок мозга в одну HTML-страницу
================================================================================
Читает любой brain.db и рисует самодостаточную страницу: граф памяти,
словарь, стадию речи и давление угасания — то есть всё, чего не видно из
чата и о чём приходилось догадываться по логам.

БЕЗОПАСНОСТЬ. Инструмент открывает БД СТРОГО НА ЧТЕНИЕ (URI-режим
`?mode=ro`) и НЕ использует memory.database.Database — тот при открытии
прогоняет миграции, то есть ПИШЕТ в файл. Инспектор не имеет права
менять мозг, который он изучает: смотреть можно и на живую рабочую базу
бота, не останавливая его и ничем не рискуя.

Отсюда же следствие: страница переживает БД без свежих колонок (старые
дампы, бэкапы) — недостающие поля деградируют мягко, а не роняют скрипт.

Запуск:
    python tools/render_memory.py storage/brains/123.db
    python tools/render_memory.py storage/brains/123.db -o /tmp/brain.html
    python tools/render_memory.py storage/brains/123.db --all   # + лексика

Ни CDN, ни внешних файлов: всё (стили, разметка, SVG) инлайнится в один
.html, который можно просто открыть двойным кликом или переслать.
================================================================================
"""

import argparse
import math
import random
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

MEMORY_TYPES = ("episodic", "concept")
LEXICAL_TYPES = ("word", "syllable")

# Служебные мета-узлы: точка отсчёта времени и разбор последнего хода.
# Это приборы, а не воспоминания, и в графе памяти им делать нечего —
# висели безымянными серыми точками, а brain_epoch показывал числом
# вместо текста.
INSTRUMENT_TYPES = ("brain_epoch", "last_decision")

TYPE_COLORS = {
    "episodic": "#5b8def",
    "concept": "#4bb3a5",
    "self_model": "#c77dd6",
    "user_model": "#d68f5a",
    "last_sleep_marker": "#8a8f98",
    "word": "#7f9f6a",
    "syllable": "#b3a55c",
    None: "#8a8f98",
}

TYPE_LABELS = {
    "episodic": "эпизод",
    "concept": "понятие",
    "self_model": "образ себя",
    "user_model": "образ учителя",
    "last_sleep_marker": "маркер сна",
    "word": "слово",
    "syllable": "слог",
}


@dataclass
class Node:
    id: int
    node_type: Optional[str]
    context: str
    response: str
    weight: float
    stability: float
    created_at: float
    last_accessed: float
    is_meta: int


@dataclass
class Snapshot:
    db_path: str
    nodes: List[Node]
    edges: List[Tuple[int, int, float]]
    mastered_words: int
    exposed_words: int
    has_stability: bool
    counts: Dict[str, int] = field(default_factory=dict)
    # Разбор последнего хода: удивление, порог, решение, настроение.
    # Пишется мозгом в мета-узел, потому что мини-апп читает базу
    # отдельным процессом и до оперативного состояния не дотянется.
    # None означает "мозг ещё ни разу не отвечал" — законное состояние.
    decision: Optional[Dict] = None


# --------------------------------------------------------------------------
# Чтение снимка (строго read-only)
# --------------------------------------------------------------------------

def load_snapshot(db_path: str) -> Snapshot:
    path = Path(db_path)
    if not path.exists():
        sys.exit(f"Файл не найден: {db_path}")

    # mode=ro: SQLite физически откажется писать в этот файл. Именно
    # поэтому здесь не используется memory.database.Database — он мигрирует
    # схему при открытии.
    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row

    columns = {r["name"] for r in conn.execute("PRAGMA table_info(nodes)")}
    if not columns:
        sys.exit(f"В {db_path} нет таблицы nodes — это не brain.db")

    has_stability = "stability" in columns

    nodes: List[Node] = []
    for row in conn.execute("SELECT * FROM nodes"):
        keys = row.keys()
        nodes.append(
            Node(
                id=row["id"],
                node_type=row["node_type"] if "node_type" in keys else None,
                context=row["context"] or "",
                response=row["response"] or "",
                weight=row["weight"] or 0.0,
                stability=(row["stability"] or 1.0) if has_stability else 1.0,
                created_at=row["created_at"] or 0.0,
                last_accessed=row["last_accessed"] or 0.0,
                is_meta=(row["is_meta"] or 0) if "is_meta" in keys else 0,
            )
        )

    edges = [
        (r["node_from"], r["node_to"], r["weight"])
        for r in conn.execute("SELECT node_from, node_to, weight FROM edges")
    ]

    mastery = config.VOCABULARY_MASTERY_MIN_WEIGHT
    mastered = sum(1 for n in nodes if n.node_type == "word" and n.weight >= mastery)
    exposed = sum(1 for n in nodes if n.node_type == "word")

    counts: Dict[str, int] = {}
    for n in nodes:
        key = n.node_type or "(без типа)"
        counts[key] = counts.get(key, 0) + 1

    decision = None
    for n in nodes:
        if n.node_type == "last_decision" and n.is_meta:
            try:
                decision = json.loads(n.context)
            except (ValueError, TypeError):
                # Повреждённый разбор — не повод падать: покажем прочерк
                decision = None
            break

    conn.close()
    return Snapshot(
        db_path=str(path), nodes=nodes, edges=edges,
        mastered_words=mastered, exposed_words=exposed,
        has_stability=has_stability, counts=counts, decision=decision,
    )


# --------------------------------------------------------------------------
# Производные величины
# --------------------------------------------------------------------------

def seconds_until_forgotten(node: Node) -> Optional[float]:
    """
    Сколько виртуальных секунд осталось узлу до FORGET_THRESHOLD, если к
    нему больше никогда не обращаться. Обратная формула затухания:

        weight * exp(-DECAY_RATE * dt / (T0 * stability)) = FORGET

    Мета-узлы иммунны к угасанию (пропускаются в _decay_nodes), поэтому
    для них возвращается None — "не забудется никогда".
    """
    if node.is_meta:
        return None
    if node.weight <= config.FORGET_THRESHOLD:
        return 0.0

    t0 = config.LEXICAL_AGE_T0 if node.node_type in LEXICAL_TYPES else config.AGE_T0
    t0 *= max(1e-9, node.stability)
    return t0 * math.log(node.weight / config.FORGET_THRESHOLD) / config.DECAY_RATE


def human_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "иммунен"
    if seconds <= 0:
        return "уже забыт"
    if seconds < 3600:
        return f"{seconds / 60:.0f} мин"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} ч"
    if seconds < 86400 * 60:
        return f"{seconds / 86400:.1f} сут"
    return f"{seconds / 86400 / 30:.1f} мес"


def speech_stage(mastered: int) -> Tuple[int, str, Optional[int]]:
    boundaries = [
        (config.SPEECH_STAGE_0_MAX_VOCAB, "лепет (довербальная стадия)"),
        (config.SPEECH_STAGE_1_MAX_VOCAB, "простые фразы"),
        (config.SPEECH_STAGE_2_MAX_VOCAB, "простая грамматика"),
    ]
    for index, (threshold, name) in enumerate(boundaries):
        if mastered < threshold:
            return index, name, threshold
    return len(boundaries), "свободная речь", None


# --------------------------------------------------------------------------
# Раскладка графа — простая силовая модель, без внешних библиотек
# --------------------------------------------------------------------------

def layout_graph(
    nodes: List[Node], edges: List[Tuple[int, int, float]],
    width: int = 900, height: int = 620, iterations: int = 220,
) -> Dict[int, Tuple[float, float]]:
    """
    Силовая раскладка (отталкивание всех пар + притяжение по рёбрам).
    Детерминирована при фиксированном seed, чтобы перегенерация страницы
    не перемешивала картинку и её можно было сравнивать глазами.
    """
    if not nodes:
        return {}

    rng = random.Random(20260731)
    ids = [n.id for n in nodes]
    index = {node_id: i for i, node_id in enumerate(ids)}
    count = len(ids)

    pos = [
        [rng.uniform(0.15, 0.85) * width, rng.uniform(0.15, 0.85) * height]
        for _ in ids
    ]

    incident = [(index[a], index[b], w) for a, b, w in edges if a in index and b in index]

    area = width * height
    k = math.sqrt(area / max(1, count))
    temperature = width / 8.0

    for step in range(iterations):
        disp = [[0.0, 0.0] for _ in ids]

        for i in range(count):
            for j in range(i + 1, count):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist_sq = dx * dx + dy * dy
                if dist_sq < 0.01:
                    dx, dy, dist_sq = rng.uniform(-1, 1), rng.uniform(-1, 1), 1.0
                dist = math.sqrt(dist_sq)
                force = (k * k) / dist
                disp[i][0] += dx / dist * force
                disp[i][1] += dy / dist * force
                disp[j][0] -= dx / dist * force
                disp[j][1] -= dy / dist * force

        for i, j, weight in incident:
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            dist = math.sqrt(dx * dx + dy * dy) or 0.01
            force = (dist * dist) / k * (0.4 + weight)
            disp[i][0] -= dx / dist * force
            disp[i][1] -= dy / dist * force
            disp[j][0] += dx / dist * force
            disp[j][1] += dy / dist * force

        for i in range(count):
            dx, dy = disp[i]
            length = math.sqrt(dx * dx + dy * dy) or 0.01
            limit = min(length, temperature)
            pos[i][0] = min(width - 30, max(30, pos[i][0] + dx / length * limit))
            pos[i][1] = min(height - 30, max(30, pos[i][1] + dy / length * limit))

        temperature *= 0.97

    return {node_id: (pos[index[node_id]][0], pos[index[node_id]][1]) for node_id in ids}


# --------------------------------------------------------------------------
# Отрисовка
# --------------------------------------------------------------------------

def render_svg(nodes: List[Node], edges, positions, width=900, height=620) -> str:
    if not nodes:
        return '<p class="empty">Граф пуст — мозгу ещё нечего показать.</p>'

    by_id = {n.id: n for n in nodes}
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="graph" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Граф памяти: {len(nodes)} узлов, {len(edges)} связей">'
    ]

    for a, b, weight in edges:
        if a not in positions or b not in positions:
            continue
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        opacity = 0.15 + 0.55 * min(1.0, weight)
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'class="edge" stroke-width="{0.4 + 2.4 * min(1.0, weight):.2f}" '
            f'stroke-opacity="{opacity:.2f}"/>'
        )

    for node in sorted(nodes, key=lambda n: n.weight):
        x, y = positions[node.id]
        radius = 4 + 14 * min(1.0, node.weight)
        color = TYPE_COLORS.get(node.node_type, TYPE_COLORS[None])
        label = TYPE_LABELS.get(node.node_type, node.node_type or "без типа")
        preview = (node.context or "").strip().replace("\n", " ")[:70]
        tooltip = (
            f"#{node.id} · {label}\n{preview}\n"
            f"вес {node.weight:.3f} · стабильность {node.stability:.1f}\n"
            f"забудется через: {human_duration(seconds_until_forgotten(node))}"
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" '
            f'class="node"><title>{escape(tooltip)}</title></circle>'
        )

    parts.append("</svg>")
    return "".join(parts)


MOOD_LABELS = {
    "joy": "радость", "curiosity": "любопытство",
    "anxiety": "тревога", "affection": "привязанность",
}


def render_decision(snap: Snapshot) -> str:
    """
    Разбор последнего хода — ПЕРВОЕ, что видит человек.

    Дашборд начинается не со статистики, а с ответа на вопрос "что
    сейчас произошло". Статистика говорит, сколько всего накоплено;
    здесь видно, как принимается решение, и именно это отличает
    дефицитную память от индекса.

    Шкала рисуется с ЗАСЕЧКОЙ порога, а не просто заполненной полосой:
    смысл не в величине плотности, а в том, по какую сторону порога она
    оказалась.
    """
    d = snap.decision
    if not d:
        return (
            '<div class="panel empty-panel">Мозг ещё ни разу не отвечал — '
            'напиши ему, и здесь появится разбор хода.</div>'
        )

    density = float(d.get("density", 0.0))
    threshold = float(d.get("threshold", 0.0))
    surprise = float(d.get("surprise", 0.0))
    emotion = float(d.get("emotion", 0.0))
    written = bool(d.get("written"))
    gap = density - threshold

    verdict = (
        f'<span class="verdict yes">записано</span>'
        f'<span class="gapnote">плотность выше порога на {gap:.3f}</span>'
        if written else
        f'<span class="verdict no">не записано</span>'
        f'<span class="gapnote">не хватило {abs(gap):.3f} до порога</span>'
    )

    return f"""<div class="panel">
  <div class="said">{escape(str(d.get("text", ""))[:160])}</div>
  <div class="gauge">
    <i class="fill{' over' if written else ''}" style="width:{min(100, density * 100):.1f}%"></i>
    <b class="mark" style="left:{min(100, threshold * 100):.1f}%"></b>
  </div>
  <div class="gaugelabels">
    <span>плотность {density:.3f}</span><span>порог {threshold:.3f}</span>
  </div>
  <div class="verdictrow">{verdict}</div>
  <div class="pair">
    <div><span class="k">удивление</span><span class="v">{surprise:.3f}</span>
      <span class="n">насколько разошлось с уже известным</span></div>
    <div><span class="k">эмоция</span><span class="v">{emotion:.3f}</span>
      <span class="n">насколько задело</span></div>
  </div>
</div>"""


def render_mood(snap: Snapshot) -> str:
    """Состояние организма: четыре оси настроения плюс возбуждение."""
    mood = (snap.decision or {}).get("mood") or {}
    if not mood:
        return ""

    rows = "".join(
        f'<div class="moodrow"><span>{label}</span>'
        f'<div class="bar"><i style="width:{max(0.0, min(1.0, float(mood.get(axis, 0.0)))) * 100:.0f}%"></i></div>'
        f'<b>{float(mood.get(axis, 0.0)):.2f}</b></div>'
        for axis, label in MOOD_LABELS.items()
    )
    arousal = float(mood.get("arousal", 0.0))
    note = (
        "перегружен — порог записи поднят, организм бережёт себя"
        if arousal > config.STRESS_OVERLOAD_THRESHOLD else
        "в рабочем режиме"
    )
    return f"""<div class="panel">
  {rows}
  <div class="moodrow arousal"><span>возбуждение</span>
    <div class="bar"><i style="width:{max(0.0, min(1.0, arousal)) * 100:.0f}%"></i></div>
    <b>{arousal:.2f}</b></div>
  <p class="sub" style="margin:10px 0 0">{note}</p>
</div>"""


def render_html(snap: Snapshot, include_lexical: bool) -> str:
    visible_types = MEMORY_TYPES + (LEXICAL_TYPES if include_lexical else ())
    visible = [
        n for n in snap.nodes
        if (n.node_type in visible_types or n.is_meta)
        and n.node_type not in INSTRUMENT_TYPES
    ]
    visible_ids = {n.id for n in visible}
    visible_edges = [(a, b, w) for a, b, w in snap.edges if a in visible_ids and b in visible_ids]

    positions = layout_graph(visible, visible_edges)
    stage, stage_name, next_threshold = speech_stage(snap.mastered_words)

    memory_nodes = sorted(
        (n for n in snap.nodes if n.node_type in MEMORY_TYPES),
        key=lambda n: n.weight, reverse=True,
    )
    top_words = sorted(
        (n for n in snap.nodes if n.node_type == "word"),
        key=lambda n: n.weight, reverse=True,
    )[:24]

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    decision_block = render_decision(snap)
    mood_block = render_mood(snap) or (
        '<div class="panel empty-panel">Настроение появится после первого ответа.</div>'
    )

    progress_pct = 0 if not next_threshold else min(100, 100 * snap.mastered_words / next_threshold)
    progress_label = (
        f"{snap.mastered_words} / {next_threshold}" if next_threshold
        else f"{snap.mastered_words} — все стадии пройдены"
    )

    legend = "".join(
        f'<span class="chip"><i style="background:{TYPE_COLORS.get(t, TYPE_COLORS[None])}"></i>'
        f'{TYPE_LABELS.get(t, t)} · {snap.counts.get(t, 0)}</span>'
        for t in sorted(snap.counts, key=lambda t: -snap.counts[t])
        if snap.counts.get(t) and t not in INSTRUMENT_TYPES
    )

    memory_rows = "".join(
        f"<tr><td class='num'>{n.id}</td>"
        f"<td>{escape((n.context or '').strip()[:90])}</td>"
        f"<td>{escape((n.response or '').strip()[:90])}</td>"
        f"<td class='num'>{n.weight:.3f}</td>"
        f"<td class='num'>{n.stability:.1f}</td>"
        f"<td class='num'>{human_duration(seconds_until_forgotten(n))}</td></tr>"
        for n in memory_nodes[:60]
    ) or "<tr><td colspan='6' class='empty'>Воспоминаний пока нет</td></tr>"

    word_rows = "".join(
        f"<tr><td>{escape(n.context)}</td><td class='num'>{n.weight:.3f}</td>"
        f"<td class='num'>{human_duration(seconds_until_forgotten(n))}</td>"
        f"<td>{'да' if n.weight >= config.VOCABULARY_MASTERY_MIN_WEIGHT else 'ещё нет'}</td></tr>"
        for n in top_words
    ) or "<tr><td colspan='4' class='empty'>Словарь пуст</td></tr>"

    stability_note = "" if snap.has_stability else (
        '<p class="warn">В этой БД ещё нет колонки <code>stability</code> — '
        'снимок сделан до миграции. Стабильность показана как 1.0, '
        'а прогноз забывания — по базовой шкале.</p>'
    )

    return f"""<title>Память мозга — {escape(Path(snap.db_path).name)}</title>
<style>
:root {{
  --bg:#fbfbfd; --fg:#1c1e21; --muted:#6b7280; --line:#e3e5e9;
  --card:#ffffff; --accent:#5b8def; --warn-bg:#fff5e6; --warn-fg:#8a5a00;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg:#15171a; --fg:#e8eaed; --muted:#9aa0a6; --line:#2a2d32;
    --card:#1c1f23; --accent:#7aa5f5; --warn-bg:#2e2410; --warn-fg:#e0b064;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#15171a; --fg:#e8eaed; --muted:#9aa0a6; --line:#2a2d32;
  --card:#1c1f23; --accent:#7aa5f5; --warn-bg:#2e2410; --warn-fg:#e0b064;
}}
:root[data-theme="light"] {{
  --bg:#fbfbfd; --fg:#1c1e21; --muted:#6b7280; --line:#e3e5e9;
  --card:#ffffff; --accent:#5b8def; --warn-bg:#fff5e6; --warn-fg:#8a5a00;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; padding:32px 20px 64px; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}}
.wrap {{ max-width:1000px; margin:0 auto; }}
h1 {{ font-size:23px; margin:0 0 4px; font-weight:650; }}
h2 {{ font-size:16px; margin:32px 0 12px; font-weight:620; }}
.sub {{ color:var(--muted); font-size:13px; margin:0 0 24px; }}
.cards {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.card .k {{ color:var(--muted); font-size:12px; margin-bottom:6px; }}
.card .v {{ font-size:22px; font-weight:640; font-variant-numeric:tabular-nums; }}
.card .n {{ color:var(--muted); font-size:12px; margin-top:4px; }}
.bar {{ height:8px; background:var(--line); border-radius:99px; overflow:hidden; margin-top:10px; }}
.bar > i {{ display:block; height:100%; background:var(--accent); border-radius:99px; }}
.graphbox {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px; overflow-x:auto; }}
.graph {{ width:100%; height:auto; min-width:640px; display:block; }}
.edge {{ stroke:var(--muted); }}
.node {{ stroke:var(--card); stroke-width:1.5; cursor:help; }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 0; }}
.chip {{ display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--muted);
        border:1px solid var(--line); border-radius:99px; padding:3px 10px; }}
.chip i {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
.scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; min-width:620px; }}
th, td {{ text-align:left; padding:8px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ color:var(--muted); font-weight:560; white-space:nowrap; }}
tr:last-child td {{ border-bottom:none; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.empty {{ color:var(--muted); font-style:italic; }}
.warn {{ background:var(--warn-bg); color:var(--warn-fg); border-radius:8px; padding:10px 14px; font-size:13px; }}
code {{ font-size:12px; }}
.panel {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }}
.empty-panel {{ color:var(--muted); font-style:italic; }}
.said {{ font-size:16px; font-weight:560; margin-bottom:14px; }}
.gauge {{ position:relative; height:14px; background:var(--line); border-radius:99px; overflow:hidden; }}
.gauge .fill {{ display:block; height:100%; background:var(--muted); border-radius:99px; }}
.gauge .fill.over {{ background:var(--accent); }}
.gauge .mark {{ position:absolute; top:-3px; width:2px; height:20px; background:var(--fg); }}
.gaugelabels {{ display:flex; justify-content:space-between; color:var(--muted);
                font-size:12px; margin-top:6px; font-variant-numeric:tabular-nums; }}
.verdictrow {{ margin-top:12px; display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }}
.verdict {{ font-weight:640; font-size:14px; }}
.verdict.yes {{ color:var(--accent); }}
.verdict.no {{ color:var(--muted); }}
.gapnote {{ color:var(--muted); font-size:12px; }}
.pair {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); margin-top:14px; }}
.pair .k {{ color:var(--muted); font-size:12px; display:block; }}
.pair .v {{ font-size:19px; font-weight:620; font-variant-numeric:tabular-nums; }}
.pair .n {{ color:var(--muted); font-size:12px; display:block; margin-top:2px; }}
.moodrow {{ display:grid; grid-template-columns:110px 1fr 44px; gap:10px; align-items:center;
            margin-bottom:8px; font-size:13px; }}
.moodrow b {{ text-align:right; font-variant-numeric:tabular-nums; font-weight:560; color:var(--muted); }}
.moodrow .bar {{ margin-top:0; }}
.moodrow.arousal {{ margin-top:12px; padding-top:12px; border-top:1px solid var(--line); }}
</style>

<div class="wrap">
  <h1>Память мозга</h1>
  <p class="sub">{escape(snap.db_path)} · снимок {generated}</p>

  {stability_note}

  <h2 style="margin-top:0">Что случилось с последним сообщением</h2>
  {decision_block}

  <h2>Состояние организма</h2>
  {mood_block}

  <h2>Что накопилось</h2>
  <div class="cards">
    <div class="card"><div class="k">Стадия речи</div><div class="v">{stage}</div>
      <div class="n">{escape(stage_name)}</div></div>
    <div class="card"><div class="k">Словарь освоен</div><div class="v">{snap.mastered_words}</div>
      <div class="n">услышано {snap.exposed_words}</div>
      <div class="bar"><i style="width:{progress_pct:.0f}%"></i></div>
      <div class="n">{progress_label}</div></div>
    <div class="card"><div class="k">Воспоминаний</div><div class="v">{len(memory_nodes)}</div>
      <div class="n">эпизоды и понятия</div></div>
    <div class="card"><div class="k">Связей</div><div class="v">{len(snap.edges)}</div>
      <div class="n">всего узлов {len(snap.nodes)}</div></div>
  </div>

  <h2>Граф памяти</h2>
  <div class="graphbox">{render_svg(visible, visible_edges, positions)}</div>
  <div class="chips">{legend}</div>
  <p class="sub" style="margin-top:10px">
    Размер узла — вес (сила памяти), толщина связи — вес ребра.
    Наведите курсор, чтобы увидеть стабильность и прогноз забывания.
  </p>

  <h2>Воспоминания и давление угасания</h2>
  <div class="scroll"><table>
    <tr><th>#</th><th>Сказал пользователь</th><th>Ответил бот</th>
        <th class="num">Вес</th><th class="num">Стабильн.</th><th class="num">Забудется через</th></tr>
    {memory_rows}
  </table></div>

  <h2>Словарь</h2>
  <div class="scroll"><table>
    <tr><th>Слово</th><th class="num">Вес</th><th class="num">Забудется через</th><th>Освоено</th></tr>
    {word_rows}
  </table></div>
</div>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Рисует снимок памяти мозга в самодостаточную HTML-страницу (только чтение)"
    )
    parser.add_argument("db_path", help="путь к brain.db")
    parser.add_argument("-o", "--output", help="куда записать HTML (по умолчанию рядом с БД)")
    parser.add_argument(
        "--all", action="store_true", dest="include_lexical",
        help="показать на графе и лексические узлы (слова/слоги), а не только воспоминания",
    )
    args = parser.parse_args()

    snap = load_snapshot(args.db_path)
    html = render_html(snap, include_lexical=args.include_lexical)

    out = Path(args.output) if args.output else Path(args.db_path).with_suffix(".memory.html")
    out.write_text(html, encoding="utf-8")

    print(f"Снимок: {snap.db_path}")
    print(f"  узлов {len(snap.nodes)}, связей {len(snap.edges)}, "
          f"словарь освоен {snap.mastered_words} из {snap.exposed_words} услышанных")
    print(f"Страница: {out.resolve()}")


if __name__ == "__main__":
    main()
