"""
Пустой conftest.py в корне репозитория — гарантирует, что pytest вставит
корень проекта в sys.path (rootless import mode), чтобы тесты могли делать
`import config`, `from decaymem.database import Database` и т.п. без
дополнительной настройки PYTHONPATH/setup.py.
"""