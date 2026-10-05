from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import (
    InferenceServiceConfig,
    ParameterServerServiceConfig,
    RolloutCoordinatorServiceConfig,
    RolloutServiceConfig,
    ServiceGroup,
    TrainingServiceConfig,
    TrajectoryServerServiceConfig,
)
from .distributed import PSRpcClient
from .registry import resolve_symbol
from .rpc import JsonRpcClient, write_json_atomic
from .service import Service
from .topology import GPU, ServiceInfo, Topology, build_topology


@dataclass(frozen=True)
class RunResult:
    topology: Topology
    summary: dict[str, Any] | None


class Ignitor:
    """One card-level SPMD ignitor, following Meshy's launch shape."""

    def __init__(
        self,
        groups: Sequence[ServiceGroup],
        *,
        recipe_module: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.groups = tuple(groups)
        self.recipe_module = recipe_module
        if self.recipe_module in (None, "__main__"):
            self.recipe_module = os.environ.get("ORCH_RECIPE")
        if not self.recipe_module:
            raise ValueError("recipe_module is required for role subprocesses")
        self.timeout = timeout
        self.services: list[Service] = []
        self.topology: Topology | None = None
        self.runtime_dir: Path | None = None
        _validate_supported_shape(self.groups)

    def run(self) -> RunResult:
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        dist = _initialize_distributed(world)
        gpus = _discover_gpus(dist, world)
        self.topology = build_topology(self.groups, gpus)
        self.runtime_dir = _resolve_runtime_dir(dist, rank, world)
        os.environ["ORCH_GPU_MANIFEST"] = json.dumps(
            [asdict(gpu) for gpu in gpus], sort_keys=True
        )

        try:
            if rank == 0:
                self._start_control_services()
            _barrier(dist, world)

            self._start_local_gpu_services(rank, gpus)
            _barrier(dist, world)

            if rank == 0:
                self._start_rollout_services()
            _barrier(dist, world)

            summary = self._wait_for_completion(rank)
            _barrier(dist, world)
            return RunResult(self.topology, summary)
        finally:
            for service in reversed(self.services):
                service.shutdown()
            for service in reversed(self.services):
                service.join()
            if dist is not None and dist.is_initialized():
                try:
                    dist.barrier()
                finally:
                    dist.destroy_process_group()

    def _start_control_services(self) -> None:
        assert self.topology is not None
        for info in self.topology.cpu_services():
            if info.role == "rollout":
                continue
            self._start(info, None)

    def _start_local_gpu_services(self, rank: int, gpus: tuple[GPU, ...]) -> None:
        assert self.topology is not None
        my_gpu = next(gpu for gpu in gpus if gpu.global_rank == rank)
        for info in self.topology.local_services(rank):
            self._start(info, my_gpu)

    def _start_rollout_services(self) -> None:
        assert self.topology is not None
        for info in self.topology.role("rollout"):
            self._start(info, None)

    def _start(self, info: ServiceInfo, my_gpu: GPU | None) -> None:
        assert self.topology is not None
        assert self.runtime_dir is not None
        for dependency_name in self.topology.dependency_names(info):
            dependency = self.topology.by_name(dependency_name)
            JsonRpcClient(dependency.endpoint).wait_ready(self.timeout)
        service_type = resolve_symbol(info.config.service_cls, Service)
        service = service_type.from_info(
            info,
            my_gpu,
            self.topology,
            self.recipe_module,
            self.runtime_dir,
        )
        service.ignite()
        service.wait_for_ready(self.timeout)
        self.services.append(service)

    def _wait_for_completion(self, rank: int) -> dict[str, Any] | None:
        assert self.topology is not None
        assert self.runtime_dir is not None
        training = self.topology.role("training")[0]
        config = training.config
        assert isinstance(config, TrainingServiceConfig)
        target_version = config.max_steps
        ps = PSRpcClient(self.topology.role("parameter_server")[0].endpoint)
        deadline = time.monotonic() + self.timeout

        worker_infos = (training,) + self.topology.role("rollout")
        while True:
            worker_statuses = {
                info.name: dict(JsonRpcClient(info.endpoint).call("status"))
                for info in worker_infos
            }
            for name, status in worker_statuses.items():
                if status.get("error"):
                    raise RuntimeError(f"service {name!r} failed: {status['error']}")
            if ps.model_version >= target_version and all(
                status.get("done") for status in worker_statuses.values()
            ):
                break
            for service in self.services:
                service.check()
                status = service.status()
                if status.get("error"):
                    raise RuntimeError(
                        f"service {service.name!r} failed: {status['error']}"
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"StaleFlow run did not reach version {target_version}"
                )
            time.sleep(0.05)

        if rank != 0:
            return None
        summary = {
            "parameter_server": ps.snapshot(),
            "trajectory_server": JsonRpcClient(
                self.topology.role("trajectory_server")[0].endpoint
            ).call("snapshot"),
            "rollout_coordinator": JsonRpcClient(
                self.topology.role("rollout_coordinator")[0].endpoint
            ).call("snapshot"),
            "topology": {
                "world_size": len(self.topology.gpus),
                "services": [
                    {
                        "name": service.name,
                        "role": service.role,
                        "endpoint": service.endpoint,
                        "global_ranks": service.gpu_ids,
                    }
                    for service in self.topology.services
                ],
            },
        }
        write_json_atomic(self.runtime_dir / "result.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return summary


def _initialize_distributed(world: int):
    if world == 1:
        return None
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return dist


def _discover_host() -> str:
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return "127.0.0.1"
    master = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = int(os.environ.get("MASTER_PORT", "29500"))
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect((master, port))
            return connection.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def _discover_gpus(dist: Any, world: int) -> tuple[GPU, ...]:
    me = GPU(
        host=_discover_host(),
        global_rank=int(os.environ.get("RANK", "0")),
        node_rank=int(os.environ.get("GROUP_RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
    )
    if world == 1:
        return (me,)
    gathered: list[GPU | None] = [None] * world
    dist.all_gather_object(gathered, me)
    return tuple(sorted((gpu for gpu in gathered if gpu is not None), key=lambda gpu: gpu.global_rank))


def _resolve_runtime_dir(dist: Any, rank: int, world: int) -> Path:
    root = os.environ.get("ORCH_RUNTIME_DIR")
    if root is None and rank == 0:
        root = str(
            Path(".orch_runtime").resolve()
            / time.strftime("%Y%m%d-%H%M%S")
        )
    if world > 1:
        container = [root]
        dist.broadcast_object_list(container, src=0)
        root = container[0]
    assert root is not None
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    os.environ["ORCH_RUNTIME_DIR"] = str(path)
    return path


def _barrier(dist: Any, world: int) -> None:
    if world > 1:
        dist.barrier()


def _validate_supported_shape(groups: Sequence[ServiceGroup]) -> None:
    by_role: dict[str, list[ServiceGroup]] = {}
    for group in groups:
        by_role.setdefault(group.config.role, []).append(group)
    required = {
        "inference": InferenceServiceConfig,
        "training": TrainingServiceConfig,
        "rollout": RolloutServiceConfig,
        "parameter_server": ParameterServerServiceConfig,
        "trajectory_server": TrajectoryServerServiceConfig,
        "rollout_coordinator": RolloutCoordinatorServiceConfig,
    }
    for role, config_type in required.items():
        role_groups = by_role.get(role, [])
        expected = 1
        if len(role_groups) != expected:
            raise ValueError(f"minimal StaleFlow runtime requires one {role} group")
        if not isinstance(role_groups[0].config, config_type):
            raise TypeError(f"{role} role must use {config_type.__name__}")

    inference = by_role["inference"][0]
    rollout = by_role["rollout"][0]
    training = cast_training(by_role["training"][0].config)
    rollout_config = by_role["rollout"][0].config
    parameter_config = by_role["parameter_server"][0].config
    assert isinstance(rollout_config, RolloutServiceConfig)
    assert isinstance(parameter_config, ParameterServerServiceConfig)
    if inference.n_replicas != rollout.n_replicas:
        raise ValueError("one inference replica is required per rollout replica")
    if training.batch_size % rollout_config.group_size != 0:
        raise ValueError("training batch_size must be divisible by rollout group_size")
    if parameter_config.num_entries != training.batch_size // rollout_config.group_size:
        raise ValueError(
            "parameter server num_entries must equal batch_size / group_size"
        )


def cast_training(config: Any) -> TrainingServiceConfig:
    if not isinstance(config, TrainingServiceConfig):
        raise TypeError("training role must use TrainingServiceConfig")
    return config
