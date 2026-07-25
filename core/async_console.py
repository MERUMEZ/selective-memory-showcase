"""
================================================================================
 ASYNC_CONSOLE.PY — Неблокирующий консольный ввод (Windows, msvcrt)
================================================================================
Решает конфликт между таймером Idle Sleep (main.py) и вводом пользователя:
    - Ввод читается посимвольно через msvcrt.kbhit()/getwch(), НЕ блокируя
      фоновый поток (в отличие от стандартного input()).
    - Каждое нажатие клавиши сбрасывает brain_clock.last_activity_time,
      поэтому бот не засыпает, пока пользователь физически печатает,
      даже если ещё не нажал Enter.
    - Фоновые логи (сон, проактив) выводятся через safe_print(), который
      аккуратно стирает текущую строку ввода, печатает лог и восстанавливает
      набранный (но ещё не отправленный) текст — без "склеивания" строк.
================================================================================
"""

import sys
import time
import threading

try:
    import msvcrt
except ImportError:
    msvcrt = None
    print("[ASYNC CONSOLE] Внимание: msvcrt не найден — асинхронный ввод работает только на Windows.")


class AsyncConsole:
    """
    Неблокирующая замена input() с поддержкой безопасного вывода фоновых
    логов (safe_print) без разрыва текущей вводимой пользователем строки.
    """

    def __init__(self, brain_clock, prompt_text: str = "\nUser > "):
        self.brain_clock = brain_clock
        self.prompt_text = prompt_text
        self.current_input_buffer = ""
        self.is_typing = False
        self.print_lock = threading.Lock()

    def safe_print(self, text: str) -> None:
        """
        Используется ФОНОВЫМИ потоками вместо print(). Стирает текущую
        строку ввода, печатает текст, затем восстанавливает набранный ввод.
        """
        with self.print_lock:
            if self.is_typing:
                clear_len = len(self.prompt_text) + len(self.current_input_buffer)
                sys.stdout.write("\r" + " " * clear_len + "\r")
                sys.stdout.flush()

            print(text)

            if self.is_typing:
                sys.stdout.write(self.prompt_text + self.current_input_buffer)
                sys.stdout.flush()

    def get_input(self) -> str:
        """
        Асинхронная замена input(). Читает по одному символу, на каждое
        нажатие сбрасывает Idle-таймер (brain_clock.register_activity()),
        не блокируя фоновые потоки (poll-based цикл с небольшим sleep).
        """
        if not msvcrt:
            return input(self.prompt_text)

        self.current_input_buffer = ""
        self.is_typing = True

        with self.print_lock:
            sys.stdout.write(self.prompt_text)
            sys.stdout.flush()

        try:
            while True:
                if msvcrt.kbhit():
                    self.brain_clock.register_activity()
                    char = msvcrt.getwch()

                    if char in ("\r", "\n"):
                        with self.print_lock:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        break

                    elif char == "\b":
                        if self.current_input_buffer:
                            self.current_input_buffer = self.current_input_buffer[:-1]
                            with self.print_lock:
                                sys.stdout.write("\b \b")
                                sys.stdout.flush()

                    elif char == "\x03":
                        raise KeyboardInterrupt

                    else:
                        self.current_input_buffer += char
                        with self.print_lock:
                            sys.stdout.write(char)
                            sys.stdout.flush()

                time.sleep(0.01)
        finally:
            self.is_typing = False

        result = self.current_input_buffer
        self.current_input_buffer = ""
        return result