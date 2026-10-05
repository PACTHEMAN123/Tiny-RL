from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .ps import OccupyResult, ReservationToken, TrainLease
from .queue import ColumnQueue, QueueBatch, RowRef


@dataclass(frozen=True)
class TrajectoryServerSnapshot:
    published_groups: int
    consumed_batches: int
    live_rows: int


class TrajectoryServer:
    """TransferQueue-like trajectory payload service keyed by staleness buffer."""

    def __init__(self, backend: ColumnQueue | None = None) -> None:
        self._backend = backend or ColumnQueue()
        self._published_groups = 0
        self._consumed_batches = 0
        self._partitions: set[str] = set()
        self._lock = threading.Lock()

    def put_group(
        self,
        base_partition: str,
        token: ReservationToken,
        occupied: OccupyResult,
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[RowRef, ...]:
        partition = buffer_partition(base_partition, occupied.occupied_buffer)
        payload = []
        for columns in rows:
            row = dict(columns)
            row.update(
                {
                    "reservation_id": token.reservation_id,
                    "owner_epoch": token.owner_epoch,
                    "training_buffer": occupied.occupied_buffer,
                    "rollout_instance_id": token.producer_id,
                }
            )
            payload.append(row)
        refs = self._backend.publish(partition, payload)
        with self._lock:
            self._published_groups += 1
            self._partitions.add(partition)
        return refs

    def lease_buffer(
        self,
        base_partition: str,
        lease: TrainLease,
        fields: Sequence[str],
        batch_size: int,
        consumer: str,
        timeout: float | None,
    ) -> QueueBatch | None:
        batch = self._backend.fetch(
            partition=buffer_partition(base_partition, lease.buffer_id),
            fields=tuple(fields)
            + ("reservation_id", "owner_epoch", "training_buffer"),
            batch_size=batch_size,
            consumer=consumer,
            timeout=timeout,
        )
        if batch is None:
            return None
        reservation_ids = {str(row["reservation_id"]) for row in batch.rows}
        if reservation_ids != set(lease.reservation_ids):
            raise RuntimeError("trajectory payload does not match the leased buffer")
        if any(int(row["training_buffer"]) != lease.buffer_id for row in batch.rows):
            raise RuntimeError("trajectory payload is stored in the wrong buffer")
        return batch

    def consume(self, batch: QueueBatch) -> None:
        self._backend.clear(batch.refs)
        with self._lock:
            self._consumed_batches += 1

    def snapshot(self) -> TrajectoryServerSnapshot:
        with self._lock:
            live_rows = sum(
                self._backend.row_count(partition) for partition in self._partitions
            )
            return TrajectoryServerSnapshot(
                published_groups=self._published_groups,
                consumed_batches=self._consumed_batches,
                live_rows=live_rows,
            )

    def close(self) -> None:
        self._backend.close()


def buffer_partition(base_partition: str, buffer_id: int) -> str:
    return f"{base_partition}.buffer.{buffer_id}"
