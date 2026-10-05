from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .engine import TrainResult


class EntryCategory(str, Enum):
    RESERVED = "reserved"
    OCCUPIED = "occupied"
    TRAIN_LEASED = "train_leased"
    CONSUMED = "consumed"
    ABORTED = "aborted"


class BufferStatus(str, Enum):
    OPEN = "open"
    READY = "ready"
    STUCK = "stuck"
    LEASED = "leased"
    CONSUMED = "consumed"


class FencedReservationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParameterServerSnapshot:
    latest_version: int
    available_versions: tuple[int, ...]
    pushes: int
    pulls: int


class ParameterServer:
    """Versioned model store; the mini backend keeps checkpoint handles in memory."""

    def __init__(self, initial_version: int = 0) -> None:
        initial = TrainResult(
            step=initial_version,
            weight_version=initial_version,
            checkpoint_ref=f"memory://weights/{initial_version}",
        )
        self._updates = {initial_version: initial}
        self._latest_version = initial_version
        self._pushes = 0
        self._pulls = 0
        self._lock = threading.RLock()

    @property
    def latest_version(self) -> int:
        with self._lock:
            return self._latest_version

    def push(self, update: TrainResult) -> None:
        with self._lock:
            existing = self._updates.get(update.weight_version)
            if existing is not None:
                if existing != update:
                    raise RuntimeError("conflicting parameter update for one version")
                return
            if update.weight_version != self._latest_version + 1:
                raise ValueError("parameter versions must be published consecutively")
            self._updates[update.weight_version] = update
            self._latest_version = update.weight_version
            self._pushes += 1

    def pull(self, version: int) -> TrainResult:
        with self._lock:
            try:
                update = self._updates[version]
            except KeyError as exc:
                raise KeyError(f"parameter version {version} is unavailable") from exc
            self._pulls += 1
            return update

    def snapshot(self) -> ParameterServerSnapshot:
        with self._lock:
            return ParameterServerSnapshot(
                latest_version=self._latest_version,
                available_versions=tuple(sorted(self._updates)),
                pushes=self._pushes,
                pulls=self._pulls,
            )


@dataclass(frozen=True)
class ReservationToken:
    reservation_id: str
    group_id: str
    owner_epoch: int
    producer_id: str
    behavior_version: int
    reserved_buffer: int


@dataclass(frozen=True)
class OccupyResult:
    reservation_id: str
    occupied_buffer: int
    buffer_status: BufferStatus


@dataclass(frozen=True)
class TrainLease:
    lease_id: str
    lease_epoch: int
    buffer_id: int
    reservation_ids: tuple[str, ...]


@dataclass(frozen=True)
class BufferSnapshot:
    buffer_id: int
    state: BufferStatus
    reserved: int
    occupied: int
    consumed: int
    capacity: int


@dataclass(frozen=True)
class InventorySnapshot:
    frontier: int
    committed_version: int
    buffers: tuple[BufferSnapshot, ...]
    reserve_rejections: int
    fenced_completions: int
    consumed_buffers: int

    def buffer(self, buffer_id: int) -> BufferSnapshot:
        for buffer in self.buffers:
            if buffer.buffer_id == buffer_id:
                return buffer
        raise KeyError(buffer_id)


@dataclass
class _LedgerEntry:
    reservation_id: str
    group_id: str
    owner_epoch: int
    producer_id: str
    behavior_version: int
    max_buffer: int
    reserved_buffer: int
    state: EntryCategory = EntryCategory.RESERVED
    occupied_buffer: int | None = None


@dataclass
class _StalenessBuffer:
    buffer_id: int
    capacity: int
    reserved_ids: list[str] = field(default_factory=list)
    occupied_ids: list[str] = field(default_factory=list)
    consumed_ids: list[str] = field(default_factory=list)
    lease_id: str | None = None
    consumed: bool = False


class StalenessInventory:
    """Reserve/Occupy/Consume ledger for staleness-bounded training buffers."""

    def __init__(
        self,
        num_entries: int,
        staleness: int,
        initial_version: int = 0,
    ) -> None:
        if num_entries <= 0:
            raise ValueError("num_entries must be positive")
        if staleness < 0:
            raise ValueError("staleness cannot be negative")

        self.num_entries = num_entries
        self.staleness = staleness
        self.frontier = initial_version
        self.committed_version = initial_version
        self._condition = threading.Condition(threading.RLock())
        self._buffers: dict[int, _StalenessBuffer] = {}
        self._entries: dict[str, _LedgerEntry] = {}
        self._active_by_group: dict[str, str] = {}
        self._group_epochs: dict[str, int] = {}
        self._committed_leases: dict[str, int] = {}
        self._reservation_seq = 0
        self._lease_seq = 0
        self.reserve_rejections = 0
        self.fenced_completions = 0
        self.consumed_buffers = 0
        self._ensure_buffers(self.frontier + self.staleness)

    def reserve(
        self,
        group_id: str,
        producer_id: str,
        behavior_version: int,
    ) -> ReservationToken | None:
        """Reserve the latest feasible buffer for one complete GRPO group."""

        with self._condition:
            active_id = self._active_by_group.get(group_id)
            if active_id is not None:
                active = self._entries[active_id]
                if (
                    active.producer_id == producer_id
                    and active.behavior_version == behavior_version
                    and active.state
                    in (EntryCategory.RESERVED, EntryCategory.OCCUPIED, EntryCategory.TRAIN_LEASED)
                ):
                    return self._token(active)
                raise RuntimeError(f"group {group_id!r} already has an active reservation")

            if behavior_version > self.committed_version:
                self.reserve_rejections += 1
                return None

            legal_min = max(self.frontier, behavior_version)
            legal_max = behavior_version + self.staleness
            if legal_min > legal_max:
                self.reserve_rejections += 1
                return None

            self._ensure_buffers(legal_max)
            reservation_id = f"reservation-{self._reservation_seq + 1}"
            owner_epoch = self._group_epochs.get(group_id, 0) + 1
            entry = _LedgerEntry(
                reservation_id=reservation_id,
                group_id=group_id,
                owner_epoch=owner_epoch,
                producer_id=producer_id,
                behavior_version=behavior_version,
                max_buffer=legal_max,
                reserved_buffer=legal_max,
            )

            existing = self._reserved_entries()
            for target in range(legal_max, legal_min - 1, -1):
                available = self._available_reservation_slots()
                if available.get(target, 0) <= 0:
                    continue
                available[target] -= 1
                assignment = self._match_reservations(existing, available)
                if assignment is None:
                    continue

                self._reservation_seq += 1
                entry.reserved_buffer = target
                self._entries[reservation_id] = entry
                self._active_by_group[group_id] = reservation_id
                self._group_epochs[group_id] = owner_epoch
                assignment[reservation_id] = target
                self._apply_reservation_assignment(assignment)
                self._assert_invariants()
                self._condition.notify_all()
                return self._token(entry)

            self.reserve_rejections += 1
            return None

    def occupy(self, token: ReservationToken) -> OccupyResult:
        """Move a completed group into the earliest feasible training buffer."""

        with self._condition:
            entry = self._entry_for_token(token)
            if entry.state in (
                EntryCategory.OCCUPIED,
                EntryCategory.TRAIN_LEASED,
                EntryCategory.CONSUMED,
            ):
                assert entry.occupied_buffer is not None
                buffer = self._buffers[entry.occupied_buffer]
                return OccupyResult(
                    reservation_id=entry.reservation_id,
                    occupied_buffer=entry.occupied_buffer,
                    buffer_status=self._buffer_status(buffer),
                )
            if entry.state is EntryCategory.ABORTED:
                self.fenced_completions += 1
                raise FencedReservationError(
                    f"reservation {entry.reservation_id!r} was aborted"
                )

            remaining = [
                reserved
                for reserved in self._reserved_entries()
                if reserved.reservation_id != entry.reservation_id
            ]
            legal_min = max(self.frontier, entry.behavior_version)
            for target in range(legal_min, entry.reserved_buffer + 1):
                available = self._available_reservation_slots()
                if available.get(target, 0) <= 0:
                    continue
                available[target] -= 1
                assignment = self._match_reservations(remaining, available)
                if assignment is None:
                    continue

                entry.state = EntryCategory.OCCUPIED
                entry.occupied_buffer = target
                self._apply_reservation_assignment(assignment)
                self._buffers[target].occupied_ids.append(entry.reservation_id)
                self._assert_invariants()
                result = OccupyResult(
                    reservation_id=entry.reservation_id,
                    occupied_buffer=target,
                    buffer_status=self._buffer_status(self._buffers[target]),
                )
                self._condition.notify_all()
                return result

            raise RuntimeError(
                f"valid reservation {entry.reservation_id!r} has no occupation target"
            )

    def abort(self, token: ReservationToken) -> bool:
        with self._condition:
            entry = self._entry_for_token(token)
            if entry.state is EntryCategory.ABORTED:
                return False
            if entry.state is not EntryCategory.RESERVED:
                raise RuntimeError("PS inventory can abort only reserved work")

            buffer = self._buffers[entry.reserved_buffer]
            buffer.reserved_ids.remove(entry.reservation_id)
            entry.state = EntryCategory.ABORTED
            self._active_by_group.pop(entry.group_id, None)
            self._condition.notify_all()
            return True

    def lease(self, timeout: float | None = None) -> TrainLease | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                buffer = self._buffers[self.frontier]
                if self._buffer_status(buffer) is BufferStatus.READY:
                    self._lease_seq += 1
                    lease_id = f"lease-{self._lease_seq}"
                    buffer.lease_id = lease_id
                    for reservation_id in buffer.occupied_ids:
                        self._entries[reservation_id].state = EntryCategory.TRAIN_LEASED
                    return TrainLease(
                        lease_id=lease_id,
                        lease_epoch=self._lease_seq,
                        buffer_id=buffer.buffer_id,
                        reservation_ids=tuple(sorted(buffer.occupied_ids)),
                    )

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

    def consume(self, lease: TrainLease, produced_version: int) -> int:
        with self._condition:
            committed_buffer = self._committed_leases.get(lease.lease_id)
            if committed_buffer is not None:
                return committed_buffer + 1
            if lease.buffer_id != self.frontier:
                raise RuntimeError("trainer cannot commit past the current frontier")
            if produced_version != lease.buffer_id + 1:
                raise ValueError(
                    "produced_version must be exactly one greater than the buffer id"
                )

            buffer = self._buffers[lease.buffer_id]
            if buffer.lease_id != lease.lease_id:
                raise FencedReservationError("training lease is no longer current")
            if tuple(sorted(buffer.occupied_ids)) != tuple(sorted(lease.reservation_ids)):
                raise RuntimeError("training buffer membership changed after lease")

            for reservation_id in buffer.occupied_ids:
                entry = self._entries[reservation_id]
                entry.state = EntryCategory.CONSUMED
                self._active_by_group.pop(entry.group_id, None)
                buffer.consumed_ids.append(reservation_id)

            buffer.occupied_ids.clear()
            buffer.consumed = True
            buffer.lease_id = None
            self.frontier += 1
            self.committed_version = produced_version
            self.consumed_buffers += 1
            self._committed_leases[lease.lease_id] = lease.buffer_id
            self._ensure_buffers(self.frontier + self.staleness)
            self._assert_invariants()
            self._condition.notify_all()
            return self.frontier

    def snapshot(self) -> InventorySnapshot:
        with self._condition:
            buffers = tuple(
                BufferSnapshot(
                    buffer_id=buffer_id,
                    state=self._buffer_status(buffer),
                    reserved=len(buffer.reserved_ids),
                    occupied=len(buffer.occupied_ids),
                    consumed=len(buffer.consumed_ids),
                    capacity=buffer.capacity,
                )
                for buffer_id, buffer in sorted(self._buffers.items())
            )
            return InventorySnapshot(
                frontier=self.frontier,
                committed_version=self.committed_version,
                buffers=buffers,
                reserve_rejections=self.reserve_rejections,
                fenced_completions=self.fenced_completions,
                consumed_buffers=self.consumed_buffers,
            )

    def _entry_for_token(self, token: ReservationToken) -> _LedgerEntry:
        entry = self._entries.get(token.reservation_id)
        current_epoch = self._group_epochs.get(token.group_id)
        if (
            entry is None
            or entry.group_id != token.group_id
            or entry.owner_epoch != token.owner_epoch
            or current_epoch != token.owner_epoch
        ):
            self.fenced_completions += 1
            raise FencedReservationError(
                f"reservation {token.reservation_id!r} is fenced"
            )
        return entry

    def _token(self, entry: _LedgerEntry) -> ReservationToken:
        return ReservationToken(
            reservation_id=entry.reservation_id,
            group_id=entry.group_id,
            owner_epoch=entry.owner_epoch,
            producer_id=entry.producer_id,
            behavior_version=entry.behavior_version,
            reserved_buffer=entry.reserved_buffer,
        )

    def _ensure_buffers(self, through: int) -> None:
        for buffer_id in range(self.frontier, through + 1):
            self._buffers.setdefault(
                buffer_id,
                _StalenessBuffer(buffer_id=buffer_id, capacity=self.num_entries),
            )

    def _reserved_entries(self) -> list[_LedgerEntry]:
        return [
            entry
            for entry in self._entries.values()
            if entry.state is EntryCategory.RESERVED
        ]

    def _available_reservation_slots(self) -> dict[int, int]:
        return {
            buffer_id: buffer.capacity - len(buffer.occupied_ids)
            for buffer_id, buffer in self._buffers.items()
            if buffer_id >= self.frontier and not buffer.consumed
        }

    def _match_reservations(
        self,
        entries: Sequence[_LedgerEntry],
        available_by_buffer: Mapping[int, int],
    ) -> dict[str, int] | None:
        slots = tuple(
            (buffer_id, slot_index)
            for buffer_id, count in sorted(available_by_buffer.items())
            for slot_index in range(max(0, count))
        )
        candidates: dict[str, tuple[tuple[int, int], ...]] = {}
        for entry in entries:
            legal_min = max(self.frontier, entry.behavior_version)
            legal = tuple(
                sorted(
                    (
                        slot
                        for slot in slots
                        if legal_min <= slot[0] <= entry.max_buffer
                    ),
                    reverse=True,
                )
            )
            if not legal:
                return None
            candidates[entry.reservation_id] = legal

        ordered = sorted(
            entries,
            key=lambda entry: (
                len(candidates[entry.reservation_id]),
                entry.max_buffer,
                -max(self.frontier, entry.behavior_version),
                entry.reservation_id,
            ),
        )
        slot_owner: Dict[Tuple[int, int], str] = {}

        def assign(reservation_id: str, seen: set[Tuple[int, int]]) -> bool:
            for slot in candidates[reservation_id]:
                if slot in seen:
                    continue
                seen.add(slot)
                displaced = slot_owner.get(slot)
                if displaced is None or assign(displaced, seen):
                    slot_owner[slot] = reservation_id
                    return True
            return False

        for entry in ordered:
            if not assign(entry.reservation_id, set()):
                return None

        return {
            reservation_id: slot[0]
            for slot, reservation_id in slot_owner.items()
        }

    def _apply_reservation_assignment(self, assignment: Mapping[str, int]) -> None:
        for buffer in self._buffers.values():
            buffer.reserved_ids.clear()
        for reservation_id, buffer_id in sorted(assignment.items()):
            entry = self._entries[reservation_id]
            if entry.state is not EntryCategory.RESERVED:
                continue
            entry.reserved_buffer = buffer_id
            self._buffers[buffer_id].reserved_ids.append(reservation_id)

    def _buffer_status(self, buffer: _StalenessBuffer) -> BufferStatus:
        if buffer.consumed:
            return BufferStatus.CONSUMED
        if buffer.lease_id is not None:
            return BufferStatus.LEASED
        if len(buffer.occupied_ids) == buffer.capacity:
            return BufferStatus.READY
        if len(buffer.occupied_ids) + len(buffer.reserved_ids) == buffer.capacity:
            return BufferStatus.STUCK
        return BufferStatus.OPEN

    def _assert_invariants(self) -> None:
        for buffer_id, buffer in self._buffers.items():
            active = len(buffer.reserved_ids) + len(buffer.occupied_ids)
            if active > buffer.capacity:
                raise AssertionError(f"buffer {buffer_id} exceeds capacity")
            for reservation_id in buffer.reserved_ids:
                entry = self._entries[reservation_id]
                legal_min = max(self.frontier, entry.behavior_version)
                if not legal_min <= buffer_id <= entry.max_buffer:
                    raise AssertionError("reserved entry violates its staleness bound")
            for reservation_id in buffer.occupied_ids:
                entry = self._entries[reservation_id]
                if not entry.behavior_version <= buffer_id <= entry.max_buffer:
                    raise AssertionError("occupied entry violates its staleness bound")


class PSManager:
    """PSRL control authority for model version and staleness-buffer metadata."""

    def __init__(
        self,
        parameter_server: ParameterServer,
        num_entries: int,
        staleness: int,
        initial_version: int = 0,
    ) -> None:
        self.parameter_server = parameter_server
        self.inventory = StalenessInventory(
            num_entries=num_entries,
            staleness=staleness,
            initial_version=initial_version,
        )

    @property
    def model_version(self) -> int:
        return self.inventory.committed_version

    def reserve_rollout_group(
        self,
        group_id: str,
        rollout_instance_id: str,
        model_version: int,
    ) -> ReservationToken | None:
        return self.inventory.reserve(
            group_id=group_id,
            producer_id=rollout_instance_id,
            behavior_version=model_version,
        )

    def occupy_rollout_group(self, token: ReservationToken) -> OccupyResult:
        return self.inventory.occupy(token)

    def abort_rollout_group(self, token: ReservationToken) -> bool:
        return self.inventory.abort(token)

    def lease_train_buffer(self, timeout: float | None = None) -> TrainLease | None:
        return self.inventory.lease(timeout)

    def consume_buffer(self, lease: TrainLease, produced_version: int) -> int:
        if self.parameter_server.latest_version < produced_version:
            raise RuntimeError("parameters must be pushed before consuming a buffer")
        return self.inventory.consume(lease, produced_version)

    def snapshot(self) -> InventorySnapshot:
        return self.inventory.snapshot()
