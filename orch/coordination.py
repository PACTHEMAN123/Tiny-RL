from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class ProducerLifecycle(str, Enum):
    ACTIVE = "active"
    SYNCING = "syncing"
    LOST = "lost"


class CommandType(str, Enum):
    SYNC = "sync"
    ABORT = "abort"


class FencedProducerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProducerSnapshot:
    producer_id: str
    producer_epoch: int
    model_version: int
    lifecycle: ProducerLifecycle
    running_group_ids: tuple[str, ...]
    waiting_requests: int
    snapshot_seq: int
    applied_command_seq: int


@dataclass(frozen=True)
class CoordinatorCommand:
    producer_id: str
    producer_epoch: int
    command_seq: int
    command_type: CommandType
    target_version: int | None = None
    group_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProducerRecordSnapshot:
    producer_id: str
    producer_epoch: int
    model_version: int
    lifecycle: ProducerLifecycle
    running_group_ids: tuple[str, ...]
    waiting_requests: int
    snapshot_seq: int
    applied_command_seq: int
    pending_commands: tuple[CoordinatorCommand, ...]


@dataclass(frozen=True)
class CoordinatorSnapshot:
    producers: tuple[ProducerRecordSnapshot, ...]
    issued_commands: int
    completed_commands: int
    accepted_snapshots: int
    dropped_snapshots: int


@dataclass
class _ProducerRecord:
    epoch: int
    latest_snapshot: ProducerSnapshot
    next_command_seq: int = 0
    applied_command_seq: int = 0
    pending: dict[int, CoordinatorCommand] = field(default_factory=dict)


class RolloutCoordinator:
    """Coalesces rollout snapshots and emits fenced, ordered control commands."""

    def __init__(self, latest_model_version: Callable[[], int]) -> None:
        self._latest_model_version = latest_model_version
        self._records: dict[str, _ProducerRecord] = {}
        self._issued_commands = 0
        self._completed_commands = 0
        self._accepted_snapshots = 0
        self._dropped_snapshots = 0
        self._lock = threading.RLock()

    def register(self, producer_id: str, producer_epoch: int, model_version: int) -> None:
        with self._lock:
            current = self._records.get(producer_id)
            if current is not None:
                if current.epoch == producer_epoch:
                    return
                if current.epoch > producer_epoch:
                    raise FencedProducerError(f"producer {producer_id!r} is fenced")
            snapshot = ProducerSnapshot(
                producer_id=producer_id,
                producer_epoch=producer_epoch,
                model_version=model_version,
                lifecycle=ProducerLifecycle.ACTIVE,
                running_group_ids=(),
                waiting_requests=0,
                snapshot_seq=0,
                applied_command_seq=0,
            )
            self._records[producer_id] = _ProducerRecord(
                epoch=producer_epoch,
                latest_snapshot=snapshot,
            )

    def submit_snapshot(self, snapshot: ProducerSnapshot) -> bool:
        with self._lock:
            record = self._record(snapshot.producer_id, snapshot.producer_epoch)
            if snapshot.snapshot_seq <= record.latest_snapshot.snapshot_seq:
                self._dropped_snapshots += 1
                return False
            self._apply_ack(record, snapshot.applied_command_seq)
            record.latest_snapshot = snapshot
            self._accepted_snapshots += 1
            latest_model_version = self._latest_model_version()

            has_current_sync = any(
                command.command_type is CommandType.SYNC
                and command.target_version is not None
                and command.target_version >= latest_model_version
                for command in record.pending.values()
            )
            if snapshot.model_version < latest_model_version and not has_current_sync:
                self._issue(
                    record,
                    producer_id=snapshot.producer_id,
                    command_type=CommandType.SYNC,
                    target_version=latest_model_version,
                )
            return True

    def poll_commands(
        self,
        producer_id: str,
        producer_epoch: int,
        after_command_seq: int,
    ) -> tuple[CoordinatorCommand, ...]:
        with self._lock:
            record = self._record(producer_id, producer_epoch)
            return tuple(
                command
                for sequence, command in sorted(record.pending.items())
                if sequence > after_command_seq
            )

    def complete_command(
        self, producer_id: str, producer_epoch: int, command_seq: int
    ) -> None:
        with self._lock:
            record = self._record(producer_id, producer_epoch)
            self._apply_ack(record, command_seq)

    def issue_abort(
        self,
        producer_id: str,
        producer_epoch: int,
        group_ids: tuple[str, ...],
    ) -> CoordinatorCommand:
        with self._lock:
            record = self._record(producer_id, producer_epoch)
            return self._issue(
                record,
                producer_id=producer_id,
                command_type=CommandType.ABORT,
                group_ids=group_ids,
            )

    def snapshot(self) -> CoordinatorSnapshot:
        with self._lock:
            producers = tuple(
                ProducerRecordSnapshot(
                    producer_id=producer_id,
                    producer_epoch=record.epoch,
                    model_version=record.latest_snapshot.model_version,
                    lifecycle=record.latest_snapshot.lifecycle,
                    running_group_ids=record.latest_snapshot.running_group_ids,
                    waiting_requests=record.latest_snapshot.waiting_requests,
                    snapshot_seq=record.latest_snapshot.snapshot_seq,
                    applied_command_seq=record.applied_command_seq,
                    pending_commands=tuple(
                        command for _, command in sorted(record.pending.items())
                    ),
                )
                for producer_id, record in sorted(self._records.items())
            )
            return CoordinatorSnapshot(
                producers=producers,
                issued_commands=self._issued_commands,
                completed_commands=self._completed_commands,
                accepted_snapshots=self._accepted_snapshots,
                dropped_snapshots=self._dropped_snapshots,
            )

    def _issue(
        self,
        record: _ProducerRecord,
        producer_id: str,
        command_type: CommandType,
        target_version: int | None = None,
        group_ids: tuple[str, ...] = (),
    ) -> CoordinatorCommand:
        record.next_command_seq += 1
        command = CoordinatorCommand(
            producer_id=producer_id,
            producer_epoch=record.epoch,
            command_seq=record.next_command_seq,
            command_type=command_type,
            target_version=target_version,
            group_ids=group_ids,
        )
        record.pending[command.command_seq] = command
        self._issued_commands += 1
        return command

    def _apply_ack(self, record: _ProducerRecord, command_seq: int) -> None:
        if command_seq <= record.applied_command_seq:
            return
        if command_seq > record.next_command_seq:
            raise ValueError("cannot acknowledge a command that was never issued")
        completed = [seq for seq in record.pending if seq <= command_seq]
        for sequence in completed:
            del record.pending[sequence]
            self._completed_commands += 1
        record.applied_command_seq = command_seq

    def _record(self, producer_id: str, producer_epoch: int) -> _ProducerRecord:
        try:
            record = self._records[producer_id]
        except KeyError as exc:
            raise KeyError(f"producer {producer_id!r} is not registered") from exc
        if record.epoch != producer_epoch:
            raise FencedProducerError(f"producer {producer_id!r} is fenced")
        return record
