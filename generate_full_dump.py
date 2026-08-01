import os

ROOT = os.path.dirname(os.path.abspath(__file__))

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".venv", "venv", "env",
    "node_modules", "brains", "logs", "models",
}

# Файлы, которые физически включаем в дамп исходного кода (текстовые,
# относящиеся к логике проекта). Бинарные/сгенерированные артефакты
# (*.db, *.log, *.pyc, *.gitkeep) не включаем.
INCLUDE_EXTENSIONS = {".py", ".txt", ".md"}
EXCLUDE_FILES = {".gitignore"}

OUTPUT_FILE = os.path.join(ROOT, "ENGRAM_FULL_DUMP.md")


def build_tree(root):
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith("."))
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        indent = "    " * depth
        if rel != ".":
            lines.append(f"{indent[:-4]}├── {os.path.basename(dirpath)}/")
        for f in sorted(filenames):
            if f.endswith((".pyc", ".db", ".log")) or f == ".gitkeep":
                continue
            lines.append(f"{indent}├── {f}")
    return "\n".join(lines)


def collect_files(root):
    collected = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith("."))
        for f in sorted(filenames):
            if f in EXCLUDE_FILES:
                continue
            ext = os.path.splitext(f)[1]
            if ext not in INCLUDE_EXTENSIONS:
                continue
            full_path = os.path.join(dirpath, f)
            rel_path = os.path.relpath(full_path, root).replace(os.sep, "/")
            collected.append(rel_path)
    return sorted(collected)


LANG_MAP = {
    ".py": "python",
    ".txt": "text",
    ".md": "markdown",
}


def main():
    tree_str = build_tree(ROOT)
    files = collect_files(ROOT)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("# ENGRAM — Полный дамп кодовой базы\n\n")
        out.write("## 1. Дерево проекта\n\n```\n")
        out.write(tree_str)
        out.write("\n```\n\n")
        out.write("## 2. Обзор модулей\n\n")
        out.write("_(см. отдельную секцию ниже — заполняется вручную)_\n\n")
        out.write("## 3. Полный исходный код\n\n")

        for rel_path in files:
            ext = os.path.splitext(rel_path)[1]
            lang = LANG_MAP.get(ext, "")
            full_path = os.path.join(ROOT, rel_path)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as exc:
                content = f"[ОШИБКА ЧТЕНИЯ ФАЙЛА: {exc}]"

            out.write(f"### `{rel_path}`\n\n")
            out.write(f"```{lang} {rel_path}\n")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
            out.write("```\n\n")

    print(f"OK: {OUTPUT_FILE}, files={len(files)}")


if __name__ == "__main__":
    main()