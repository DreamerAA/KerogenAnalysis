import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar


class TimerError(Exception):
    """Пользовательское исключение,
    используемое для сообщения об ошибках при использовании класса Timer"""


@dataclass
class Timer:
    timers: ClassVar[dict[str, float]] = dict()
    name: str | None = None
    text: str = "Execution time: {:0.4f} seconds"
    logger: Callable[[str], None] | None = print
    _start_time: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Добавить таймер к dict таймеров после инициализации"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Начать новый таймер"""
        if self._start_time is not None:
            raise RuntimeError(
                message="Timer is already running. Use .stop() to stop it"
            )

        self._start_time = time.perf_counter()

    def stop(self, mark: str = "") -> float:
        """Остановить таймер и сообщить истекшее время"""
        if self._start_time is None:
            raise RuntimeError(
                message="Timer is not running. Use .start() to start its"
            )

        # Рассчитать прошедшее время
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Сообщить о прошедшем времени
        if self.logger:
            add_mark = (mark + " ") if mark != "" else ""
            self.logger(" --- " + add_mark + self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time
