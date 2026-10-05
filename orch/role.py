from __future__ import annotations

import argparse
import importlib
import json
import os
import threading
from typing import Any, Callable, cast

from .config import (
    InferenceServiceConfig,
    ParameterServerServiceConfig,
    RolloutServiceConfig,
    TrainingServiceConfig,
)
from .distributed import (
    CoordinatorRpcClient,
    CoordinatorRpcService,
    InferenceRpcClient,
    InferenceRpcService,
    PSRpcClient,
    PSRpcService,
    TrajectoryRpcClient,
    TrajectoryRpcService,
)
from .engine import InferenceEngine, TrainingEngine
from .rpc import JsonRpcServer
from .topology import GPU, ServiceInfo, Topology, build_topology
from .worker import RolloutWorker, TrainingWorker, Worker


class RoleRuntime:
    """Hosts one algorithm role and a tiny lifecycle RPC surface."""

    def __init__(
        self,
        dispatch: Callable[[str, Any], Any] | None = None,
        worker: Worker | None = None,
    ) -> None:
        self._algorithm_dispatch = dispatch
        self._worker = worker
        self._stop_event = threading.Event()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def run(self, info: ServiceInfo) -> None:
        if self._worker is not None:
            self._worker.start()
        JsonRpcServer("0.0.0.0", info.endpoint_port, self.dispatch).serve_until_event(
            self._stop_event
        )
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            if self._worker.exception is not None:
                raise RuntimeError(f"worker {self._worker.name!r} failed") from self._worker.exception

    def dispatch(self, method: str, payload: Any) -> Any:
        if method == "status":
            return {
                "done": self._worker.done if self._worker is not None else False,
                "error": (
                    None
                    if self._worker is None or self._worker.exception is None
                    else repr(self._worker.exception)
                ),
            }
        if method == "shutdown":
            self._stop_event.set()
            return True
        if self._algorithm_dispatch is None:
            raise KeyError(f"unknown role RPC method: {method}")
        return self._algorithm_dispatch(method, payload)


def build_runtime(info: ServiceInfo, topology: Topology) -> RoleRuntime:
    ps_info = _one(topology, "parameter_server")
    trajectory_info = _one(topology, "trajectory_server")
    coordinator_info = _one(topology, "rollout_coordinator")

    if info.role == "parameter_server":
        config = cast(ParameterServerServiceConfig, info.config)
        service = PSRpcService(
            config.num_entries,
            config.staleness,
            initial_version=config.initial_version,
        )
        return RoleRuntime(service.dispatch)
    if info.role == "trajectory_server":
        return RoleRuntime(TrajectoryRpcService().dispatch)
    if info.role == "rollout_coordinator":
        return RoleRuntime(CoordinatorRpcService(ps_info.endpoint).dispatch)
    if info.role == "inference":
        config = cast(InferenceServiceConfig, info.config)
        engine = config.engine_factory()
        if not isinstance(engine, InferenceEngine):
            raise TypeError("inference engine factory must return InferenceEngine")
        return RoleRuntime(InferenceRpcService(engine).dispatch)
    if info.role == "training":
        config = cast(TrainingServiceConfig, info.config)
        engine = config.engine_factory()
        if not isinstance(engine, TrainingEngine):
            raise TypeError("training engine factory must return TrainingEngine")
        runtime = RoleRuntime()
        runtime._worker = TrainingWorker(
            name=f"{info.name}.worker",
            stop_event=runtime.stop_event,
            config=config,
            engine=engine,
            ps_manager=lambda: PSRpcClient(ps_info.endpoint),
            parameter_server=lambda: PSRpcClient(ps_info.endpoint),
            trajectory_server=lambda: TrajectoryRpcClient(trajectory_info.endpoint),
        )
        return runtime
    if info.role == "rollout":
        config = cast(RolloutServiceConfig, info.config)
        inference = topology.role("inference")[info.replica_index]
        runtime = RoleRuntime()
        runtime._worker = RolloutWorker(
            name=f"{info.name}.worker",
            stop_event=runtime.stop_event,
            config=config,
            inference_engine=lambda: InferenceRpcClient(inference.endpoint),
            ps_manager=lambda: PSRpcClient(ps_info.endpoint),
            parameter_server=lambda: PSRpcClient(ps_info.endpoint),
            trajectory_server=lambda: TrajectoryRpcClient(trajectory_info.endpoint),
            coordinator=lambda: CoordinatorRpcClient(coordinator_info.endpoint),
            replica_index=info.replica_index,
            replica_count=len(topology.group(info.group_id)),
        )
        return runtime
    raise ValueError(f"unsupported role: {info.role}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one resolved orch role")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--service", required=True)
    args = parser.parse_args()

    recipe = importlib.import_module(args.recipe)
    groups = tuple(recipe.SERVICE_GROUPS)
    gpus = tuple(GPU(**item) for item in json.loads(os.environ["ORCH_GPU_MANIFEST"]))
    topology = build_topology(groups, gpus)
    info = topology.by_name(args.service)
    build_runtime(info, topology).run(info)


def _one(topology: Topology, role: str) -> ServiceInfo:
    services = topology.role(role)
    if len(services) != 1:
        raise RuntimeError(f"runtime requires exactly one {role} service")
    return services[0]


if __name__ == "__main__":
    main()
