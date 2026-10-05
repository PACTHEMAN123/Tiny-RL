from __future__ import annotations

from typing import Any, Mapping, Sequence

from .coordination import (
    CommandType,
    CoordinatorCommand,
    ProducerLifecycle,
    ProducerSnapshot,
    RolloutCoordinator,
)
from .engine import Generation, InferenceEngine, TrainResult
from .ps import (
    BufferStatus,
    OccupyResult,
    ParameterServer,
    PSManager,
    ReservationToken,
    TrainLease,
)
from .queue import QueueBatch, RowRef
from .rpc import JsonRpcClient
from .trajectory import TrajectoryServer


def reservation_token(payload: Mapping[str, Any]) -> ReservationToken:
    return ReservationToken(
        reservation_id=str(payload["reservation_id"]),
        group_id=str(payload["group_id"]),
        owner_epoch=int(payload["owner_epoch"]),
        producer_id=str(payload["producer_id"]),
        behavior_version=int(payload["behavior_version"]),
        reserved_buffer=int(payload["reserved_buffer"]),
    )


def occupy_result(payload: Mapping[str, Any]) -> OccupyResult:
    return OccupyResult(
        reservation_id=str(payload["reservation_id"]),
        occupied_buffer=int(payload["occupied_buffer"]),
        buffer_status=BufferStatus(str(payload["buffer_status"])),
    )


def train_lease(payload: Mapping[str, Any]) -> TrainLease:
    return TrainLease(
        lease_id=str(payload["lease_id"]),
        lease_epoch=int(payload["lease_epoch"]),
        buffer_id=int(payload["buffer_id"]),
        reservation_ids=tuple(str(item) for item in payload["reservation_ids"]),
    )


def train_result(payload: Mapping[str, Any]) -> TrainResult:
    return TrainResult(
        step=int(payload["step"]),
        weight_version=int(payload["weight_version"]),
        metrics={str(key): float(value) for key, value in payload.get("metrics", {}).items()},
        checkpoint_ref=payload.get("checkpoint_ref"),
    )


class PSRpcService:
    def __init__(
        self, num_entries: int, staleness: int, initial_version: int = 0
    ) -> None:
        self.parameter_server = ParameterServer(initial_version)
        self.manager = PSManager(
            self.parameter_server,
            num_entries,
            staleness,
            initial_version=initial_version,
        )

    def dispatch(self, method: str, payload: Any) -> Any:
        request = payload or {}
        if method == "reserve":
            return self.manager.reserve_rollout_group(
                group_id=str(request["group_id"]),
                rollout_instance_id=str(request["rollout_instance_id"]),
                model_version=int(request["model_version"]),
            )
        if method == "occupy":
            return self.manager.occupy_rollout_group(reservation_token(request))
        if method == "abort":
            return self.manager.abort_rollout_group(reservation_token(request))
        if method == "lease":
            return self.manager.lease_train_buffer(float(request.get("timeout", 0.0)))
        if method == "consume":
            return self.manager.consume_buffer(
                train_lease(request["lease"]), int(request["produced_version"])
            )
        if method == "push":
            self.parameter_server.push(train_result(request))
            return True
        if method == "pull":
            return self.parameter_server.pull(int(request["version"]))
        if method == "model_version":
            return self.manager.model_version
        if method == "snapshot":
            return {
                "parameters": self.parameter_server.snapshot(),
                "inventory": self.manager.snapshot(),
            }
        raise KeyError(f"unknown PS RPC method: {method}")


class PSRpcClient:
    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        self.rpc = JsonRpcClient(endpoint, timeout)

    def reserve_rollout_group(
        self, group_id: str, rollout_instance_id: str, model_version: int
    ) -> ReservationToken | None:
        result = self.rpc.call(
            "reserve",
            {
                "group_id": group_id,
                "rollout_instance_id": rollout_instance_id,
                "model_version": model_version,
            },
        )
        return None if result is None else reservation_token(result)

    def occupy_rollout_group(self, token: ReservationToken) -> OccupyResult:
        return occupy_result(self.rpc.call("occupy", token))

    def abort_rollout_group(self, token: ReservationToken) -> bool:
        return bool(self.rpc.call("abort", token))

    def lease_train_buffer(self, timeout: float | None = None) -> TrainLease | None:
        result = self.rpc.call("lease", {"timeout": timeout or 0.0})
        return None if result is None else train_lease(result)

    def consume_buffer(self, lease: TrainLease, produced_version: int) -> int:
        return int(
            self.rpc.call(
                "consume",
                {"lease": lease, "produced_version": produced_version},
            )
        )

    @property
    def latest_version(self) -> int:
        return self.model_version

    @property
    def model_version(self) -> int:
        return int(self.rpc.call("model_version"))

    def push(self, update: TrainResult) -> None:
        self.rpc.call("push", update)

    def pull(self, version: int) -> TrainResult:
        return train_result(self.rpc.call("pull", {"version": version}))

    def snapshot(self) -> dict[str, Any]:
        return dict(self.rpc.call("snapshot"))


class TrajectoryRpcService:
    def __init__(self) -> None:
        self.server = TrajectoryServer()

    def dispatch(self, method: str, payload: Any) -> Any:
        request = payload or {}
        if method == "put_group":
            return self.server.put_group(
                str(request["base_partition"]),
                reservation_token(request["token"]),
                occupy_result(request["occupied"]),
                request["rows"],
            )
        if method == "lease_buffer":
            return self.server.lease_buffer(
                base_partition=str(request["base_partition"]),
                lease=train_lease(request["lease"]),
                fields=tuple(str(item) for item in request["fields"]),
                batch_size=int(request["batch_size"]),
                consumer=str(request["consumer"]),
                timeout=float(request.get("timeout", 0.0)),
            )
        if method == "consume":
            refs = tuple(
                RowRef(partition=str(ref["partition"]), row_id=int(ref["row_id"]))
                for ref in request["refs"]
            )
            self.server.consume(QueueBatch(refs=refs, rows=()))
            return True
        if method == "snapshot":
            return self.server.snapshot()
        raise KeyError(f"unknown trajectory RPC method: {method}")


class TrajectoryRpcClient:
    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        self.rpc = JsonRpcClient(endpoint, timeout)

    def put_group(
        self,
        base_partition: str,
        token: ReservationToken,
        occupied: OccupyResult,
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[RowRef, ...]:
        result = self.rpc.call(
            "put_group",
            {
                "base_partition": base_partition,
                "token": token,
                "occupied": occupied,
                "rows": rows,
            },
        )
        return tuple(
            RowRef(partition=str(ref["partition"]), row_id=int(ref["row_id"]))
            for ref in result
        )

    def lease_buffer(
        self,
        base_partition: str,
        lease: TrainLease,
        fields: Sequence[str],
        batch_size: int,
        consumer: str,
        timeout: float | None,
    ) -> QueueBatch | None:
        result = self.rpc.call(
            "lease_buffer",
            {
                "base_partition": base_partition,
                "lease": lease,
                "fields": fields,
                "batch_size": batch_size,
                "consumer": consumer,
                "timeout": timeout or 0.0,
            },
        )
        if result is None:
            return None
        return QueueBatch(
            refs=tuple(
                RowRef(partition=str(ref["partition"]), row_id=int(ref["row_id"]))
                for ref in result["refs"]
            ),
            rows=tuple(dict(row) for row in result["rows"]),
        )

    def consume(self, batch: QueueBatch) -> None:
        self.rpc.call("consume", {"refs": batch.refs})

    def snapshot(self) -> dict[str, Any]:
        return dict(self.rpc.call("snapshot"))


class CoordinatorRpcService:
    def __init__(self, ps_endpoint: str) -> None:
        self.ps = PSRpcClient(ps_endpoint)
        self.coordinator = RolloutCoordinator(
            latest_model_version=lambda: self.ps.model_version
        )

    def dispatch(self, method: str, payload: Any) -> Any:
        request = payload or {}
        if method == "register":
            self.coordinator.register(
                str(request["producer_id"]),
                int(request["producer_epoch"]),
                int(request["model_version"]),
            )
            return True
        if method == "submit_snapshot":
            return self.coordinator.submit_snapshot(_producer_snapshot(request))
        if method == "poll_commands":
            return self.coordinator.poll_commands(
                str(request["producer_id"]),
                int(request["producer_epoch"]),
                int(request["after_command_seq"]),
            )
        if method == "complete_command":
            self.coordinator.complete_command(
                str(request["producer_id"]),
                int(request["producer_epoch"]),
                int(request["command_seq"]),
            )
            return True
        if method == "snapshot":
            return self.coordinator.snapshot()
        raise KeyError(f"unknown coordinator RPC method: {method}")


class CoordinatorRpcClient:
    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        self.rpc = JsonRpcClient(endpoint, timeout)

    def register(self, producer_id: str, producer_epoch: int, model_version: int) -> None:
        self.rpc.call(
            "register",
            {
                "producer_id": producer_id,
                "producer_epoch": producer_epoch,
                "model_version": model_version,
            },
        )

    def submit_snapshot(self, snapshot: ProducerSnapshot) -> bool:
        return bool(self.rpc.call("submit_snapshot", snapshot))

    def poll_commands(
        self, producer_id: str, producer_epoch: int, after_command_seq: int
    ) -> tuple[CoordinatorCommand, ...]:
        result = self.rpc.call(
            "poll_commands",
            {
                "producer_id": producer_id,
                "producer_epoch": producer_epoch,
                "after_command_seq": after_command_seq,
            },
        )
        return tuple(_coordinator_command(item) for item in result)

    def complete_command(
        self, producer_id: str, producer_epoch: int, command_seq: int
    ) -> None:
        self.rpc.call(
            "complete_command",
            {
                "producer_id": producer_id,
                "producer_epoch": producer_epoch,
                "command_seq": command_seq,
            },
        )

    def snapshot(self) -> dict[str, Any]:
        return dict(self.rpc.call("snapshot"))


class InferenceRpcService:
    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine

    def dispatch(self, method: str, payload: Any) -> Any:
        request = payload or {}
        if method == "weight_version":
            return self.engine.weight_version
        if method == "generate":
            return self.engine.generate(
                request["prompt"], weight_version=int(request["weight_version"])
            )
        if method == "load_weights":
            self.engine.load_weights(train_result(request))
            return True
        if method == "snapshot":
            return {"weight_version": self.engine.weight_version}
        raise KeyError(f"unknown inference RPC method: {method}")


class InferenceRpcClient(InferenceEngine):
    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        self.rpc = JsonRpcClient(endpoint, timeout)

    @property
    def weight_version(self) -> int:
        return int(self.rpc.call("weight_version"))

    def generate(self, prompt: Any, *, weight_version: int) -> Generation:
        result = self.rpc.call(
            "generate", {"prompt": prompt, "weight_version": weight_version}
        )
        return Generation(
            response=str(result["response"]),
            tokens=tuple(int(item) for item in result["tokens"]),
            logprobs=tuple(float(item) for item in result["logprobs"]),
        )

    def load_weights(self, update: TrainResult) -> None:
        self.rpc.call("load_weights", update)


def _producer_snapshot(payload: Mapping[str, Any]) -> ProducerSnapshot:
    return ProducerSnapshot(
        producer_id=str(payload["producer_id"]),
        producer_epoch=int(payload["producer_epoch"]),
        model_version=int(payload["model_version"]),
        lifecycle=ProducerLifecycle(str(payload["lifecycle"])),
        running_group_ids=tuple(str(item) for item in payload["running_group_ids"]),
        waiting_requests=int(payload["waiting_requests"]),
        snapshot_seq=int(payload["snapshot_seq"]),
        applied_command_seq=int(payload["applied_command_seq"]),
    )


def _coordinator_command(payload: Mapping[str, Any]) -> CoordinatorCommand:
    target = payload.get("target_version")
    return CoordinatorCommand(
        producer_id=str(payload["producer_id"]),
        producer_epoch=int(payload["producer_epoch"]),
        command_seq=int(payload["command_seq"]),
        command_type=CommandType(str(payload["command_type"])),
        target_version=None if target is None else int(target),
        group_ids=tuple(str(item) for item in payload.get("group_ids", ())),
    )
