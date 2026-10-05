from __future__ import annotations

import importlib
from typing import TypeVar, cast


T = TypeVar("T")


def resolve_symbol(path: str, expected_type: type[T]) -> type[T]:
    try:
        module_name, symbol_name = path.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"symbol path must use 'module:attribute': {path!r}") from exc

    module = importlib.import_module(module_name)
    symbol = getattr(module, symbol_name)
    if not isinstance(symbol, type) or not issubclass(symbol, expected_type):
        raise TypeError(f"{path!r} is not a subclass of {expected_type.__name__}")
    return cast(type[T], symbol)
