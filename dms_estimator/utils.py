import os
import contextlib
from time import perf_counter
from functools import wraps


def printyellow(msg: str, /, *args, **kwargs) -> None:
    print(f"\033[33m{msg}\033[0m", *args, **kwargs)


def silence(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with (
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
        ):
            return func(*args, **kwargs)

    return wrapper


class Timer:
    def __init__(self, label="block", *, logger=print):
        self.label = label
        self.logger = logger

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = perf_counter() - self.start
        self.logger(f"[{self.label}] took {elapsed:.6f}s")


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__qualname__):
            return func(*args, **kwargs)

    return wrapper
