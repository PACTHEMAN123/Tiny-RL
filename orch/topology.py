from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import ServiceConfig, ServiceGroup


@dataclass(frozen=True)
class GPU:
    """One card discovered by the card-level SPMD ignitors."""

    host: str
    global_rank: int
    node_rank: int
    local_rank: int


@dataclass(frozen=True)
class ServiceInfo:
    """A deterministically placed service replica."""

    name: str
    group_id: str
    role: str
    replica_index: int
    host: str
    endpoint_port: int
    dist_port: int
    replica_gpus: tuple[GPU, ...]
    wait_until: tuple[str, ...]
    config: ServiceConfig

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.endpoint_port}"

    @property
    def uses_gpu(self) -> bool:
        return bool(self.replica_gpus)

    @property
    def master_rank(self) -> int:
        if not self.replica_gpus:
            return 0
        return self.replica_gpus[0].global_rank

    @property
    def gpu_ids(self) -> tuple[int, ...]:
        return tuple(gpu.global_rank for gpu in self.replica_gpus)

    def contains_rank(self, rank: int) -> bool:
        return any(gpu.global_rank == rank for gpu in self.replica_gpus)


@dataclass(frozen=True)
class Topology:
    """Full service registry independently derived by every ignitor."""

    services: tuple[ServiceInfo, ...]
    gpus: tuple[GPU, ...]

    @property
    def gpu_ids(self) -> tuple[int, ...]:
        return tuple(gpu.global_rank for gpu in self.gpus)

    def by_name(self, name: str) -> ServiceInfo:
        for service in self.services:
            if service.name == name:
                return service
        raise KeyError(f"no service named {name!r}")

    def group(self, group_id: str) -> tuple[ServiceInfo, ...]:
        return tuple(service for service in self.services if service.group_id == group_id)

    def role(self, role: str) -> tuple[ServiceInfo, ...]:
        return tuple(service for service in self.services if service.role == role)

    def gpu_services(self) -> tuple[ServiceInfo, ...]:
        return tuple(service for service in self.services if service.uses_gpu)

    def cpu_services(self) -> tuple[ServiceInfo, ...]:
        return tuple(service for service in self.services if not service.uses_gpu)

    def local_services(self, rank: int) -> tuple[ServiceInfo, ...]:
        return tuple(service for service in self.gpu_services() if service.contains_rank(rank))

    def dependency_names(self, service: ServiceInfo) -> tuple[str, ...]:
        return tuple(
            dependency.name
            for group_id in service.wait_until
            for dependency in self.group(group_id)
        )


def build_topology(
    groups: Sequence[ServiceGroup], gpus: Sequence[GPU | int]
) -> Topology:
    """Pure placement function shared by every card-level ignitor."""

    available_gpus = _normalize_gpus(gpus)
    if len({gpu.global_rank for gpu in available_gpus}) != len(available_gpus):
        raise ValueError("GPU global ranks must be unique")

    group_ids = [group.id for group in groups]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("service group ids must be unique")

    known_groups = set(group_ids)
    for group in groups:
        if group.colocate_with is not None:
            raise ValueError(
                "colocation is intentionally unsupported by the minimal "
                "disaggregated GRPO runtime"
            )
        unknown = set(group.wait_until) - known_groups
        if unknown:
            raise ValueError(
                f"service group {group.id!r} depends on unknown groups: {sorted(unknown)}"
            )
        if group.id in group.wait_until:
            raise ValueError(f"service group {group.id!r} cannot depend on itself")
    _validate_acyclic(groups)

    if not available_gpus and any(group.config.uses_gpu for group in groups):
        raise ValueError("GPU services require at least one discovered card")
    control_host = available_gpus[0].host if available_gpus else "127.0.0.1"
    services: list[ServiceInfo] = []
    gpu_cursor = 0
    for group in groups:
        for replica_index in range(group.n_replicas):
            replica_gpus: tuple[GPU, ...] = ()
            host = control_host
            endpoint_index = replica_index
            if group.config.uses_gpu:
                next_cursor = gpu_cursor + group.n_gpus_per_replica
                replica_gpus = available_gpus[gpu_cursor:next_cursor]
                if len(replica_gpus) != group.n_gpus_per_replica:
                    remaining = len(available_gpus) - gpu_cursor
                    raise ValueError(
                        f"not enough GPUs for {group.id!r}: requested {group.n_gpus}, "
                        f"available {remaining}"
                    )
                gpu_cursor = next_cursor
                host = replica_gpus[0].host
                endpoint_index = replica_gpus[0].global_rank

            services.append(
                ServiceInfo(
                    name=f"{group.id}-{replica_index}",
                    group_id=group.id,
                    role=group.config.role,
                    replica_index=replica_index,
                    host=host,
                    endpoint_port=group.config.endpoint_port_base + endpoint_index,
                    dist_port=43000 + endpoint_index,
                    replica_gpus=replica_gpus,
                    wait_until=group.wait_until,
                    config=group.config,
                )
            )

    return Topology(services=tuple(services), gpus=available_gpus)


def required_gpus(groups: Sequence[ServiceGroup]) -> int:
    return sum(group.n_gpus for group in groups)


def _normalize_gpus(gpus: Sequence[GPU | int]) -> tuple[GPU, ...]:
    normalized = []
    for local_index, value in enumerate(gpus):
        if isinstance(value, GPU):
            normalized.append(value)
        else:
            rank = int(value)
            normalized.append(
                GPU(
                    host="127.0.0.1",
                    global_rank=rank,
                    node_rank=0,
                    local_rank=local_index,
                )
            )
    return tuple(sorted(normalized, key=lambda gpu: gpu.global_rank))


def _validate_acyclic(groups: Sequence[ServiceGroup]) -> None:
    dependencies = {group.id: set(group.wait_until) for group in groups}
    resolved: set[str] = set()

    while dependencies:
        ready = {name for name, deps in dependencies.items() if deps <= resolved}
        if not ready:
            cycle = ", ".join(sorted(dependencies))
            raise ValueError(f"cyclic service dependencies: {cycle}")
        resolved.update(ready)
        for name in ready:
            del dependencies[name]
