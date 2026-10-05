from __future__ import annotations

import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


class QueueClosed(RuntimeError):
    pass


@dataclass(frozen=True)
class RowRef:
    partition: str
    row_id: int


@dataclass(frozen=True)
class QueueBatch:
    refs: tuple[RowRef, ...]
    rows: tuple[dict[str, Any], ...]


@dataclass
class _Row:
    columns: dict[str, Any]
    claimed_by: set[str] = field(default_factory=set)


class ColumnQueue:
    """A tiny TQ-like queue with column readiness and consumer claims."""

    def __init__(self) -> None:
        self._rows: dict[str, OrderedDict[int, _Row]] = defaultdict(OrderedDict)
        self._next_row_id = 0
        self._closed = False
        self._condition = threading.Condition()

    def publish(
        self, partition: str, rows: Iterable[Mapping[str, Any]]
    ) -> tuple[RowRef, ...]:
        refs: list[RowRef] = []
        with self._condition:
            self._ensure_open()
            for columns in rows:
                row_id = self._next_row_id
                self._next_row_id += 1
                self._rows[partition][row_id] = _Row(dict(columns))
                refs.append(RowRef(partition=partition, row_id=row_id))
            self._condition.notify_all()
        return tuple(refs)

    def update(
        self,
        refs: Sequence[RowRef],
        columns: Sequence[Mapping[str, Any]],
    ) -> None:
        if len(refs) != len(columns):
            raise ValueError("refs and columns must have the same length")

        with self._condition:
            self._ensure_open()
            for ref, values in zip(refs, columns):
                try:
                    row = self._rows[ref.partition][ref.row_id]
                except KeyError as exc:
                    raise KeyError(f"unknown queue row: {ref}") from exc
                row.columns.update(values)
            self._condition.notify_all()

    def fetch(
        self,
        partition: str,
        fields: Sequence[str],
        batch_size: int,
        consumer: str,
        timeout: float | None = None,
    ) -> QueueBatch | None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        required = set(fields)
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while True:
                candidates = [
                    (row_id, row)
                    for row_id, row in self._rows[partition].items()
                    if consumer not in row.claimed_by
                    and required <= row.columns.keys()
                ]
                if len(candidates) >= batch_size:
                    selected = candidates[:batch_size]
                    refs: list[RowRef] = []
                    rows: list[dict[str, Any]] = []
                    for row_id, row in selected:
                        row.claimed_by.add(consumer)
                        refs.append(RowRef(partition=partition, row_id=row_id))
                        rows.append(dict(row.columns))
                    return QueueBatch(refs=tuple(refs), rows=tuple(rows))

                if self._closed:
                    raise QueueClosed("column queue is closed")

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

    def clear(self, refs: Sequence[RowRef]) -> None:
        with self._condition:
            for ref in refs:
                partition = self._rows.get(ref.partition)
                if partition is not None:
                    partition.pop(ref.row_id, None)
            self._condition.notify_all()

    def row_count(self, partition: str) -> int:
        with self._condition:
            return len(self._rows.get(partition, ()))

    def snapshot(self, partition: str) -> tuple[dict[str, Any], ...]:
        with self._condition:
            return tuple(dict(row.columns) for row in self._rows.get(partition, {}).values())

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _ensure_open(self) -> None:
        if self._closed:
            raise QueueClosed("column queue is closed")
