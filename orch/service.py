from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .rpc import JsonRpcClient, RpcError
from .topology import GPU, ServiceInfo, Topology


class Service:
    """A role subprocess owned by the ignitor on its replica master card."""

    def __init__(
        self,
        info: ServiceInfo,
        my_gpu: GPU | None,
        topology: Topology,
        recipe_module: str,
        runtime_dir: Path,
    ) -> None:
        self.info = info
        self.my_gpu = my_gpu
        self.topology = topology
        self.recipe_module = recipe_module
        self.runtime_dir = runtime_dir
        self.process: subprocess.Popen[bytes] | None = None
        self._log = None
        self._log_path: Path | None = None

    @classmethod
    def from_info(
        cls,
        info: ServiceInfo,
        my_gpu: GPU | None,
        topology: Topology,
        recipe_module: str,
        runtime_dir: Path,
    ) -> "Service":
        return cls(info, my_gpu, topology, recipe_module, runtime_dir)

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def is_owner(self) -> bool:
        rank = 0 if self.my_gpu is None else self.my_gpu.global_rank
        return rank == self.info.master_rank

    @property
    def launches_process(self) -> bool:
        if self.info.config.launch_on_all_ranks:
            return self.my_gpu is not None and self.info.contains_rank(
                self.my_gpu.global_rank
            )
        return self.is_owner

    def ignite(self) -> None:
        if not self.launches_process:
            return
        log_dir = self.runtime_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        suffix = ""
        if self.info.config.launch_on_all_ranks and self.my_gpu is not None:
            suffix = f".rank-{self.my_gpu.global_rank}"
        log_path = log_dir / f"{self.name}{suffix}.log"
        self._log_path = log_path
        self._log = log_path.open("ab", buffering=0)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if self.info.config.launch_on_all_ranks:
            assert self.my_gpu is not None
            replica_rank = next(
                index
                for index, gpu in enumerate(self.info.replica_gpus)
                if gpu.global_rank == self.my_gpu.global_rank
            )
            env.update(
                {
                    "RANK": str(replica_rank),
                    "WORLD_SIZE": str(len(self.info.replica_gpus)),
                    "LOCAL_RANK": str(self.my_gpu.local_rank),
                    "GROUP_RANK": str(self.my_gpu.node_rank),
                    "MASTER_ADDR": self.info.host,
                    "MASTER_PORT": str(self.info.dist_port),
                    "ORCH_ROLE_RANK": str(replica_rank),
                }
            )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "orch.role",
                "--recipe",
                self.recipe_module,
                "--service",
                self.name,
            ],
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )

    def wait_for_ready(self, timeout: float = 60.0) -> None:
        client = JsonRpcClient(self.info.endpoint, timeout=1.0)
        deadline = time.monotonic() + timeout
        while True:
            try:
                client.call("health")
                return
            except RpcError:
                pass
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"service {self.name!r} exited with code {self.process.returncode}; "
                    f"see {self._log_path}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"service {self.name!r} did not become ready; see {self._log_path}"
                )
            time.sleep(0.1)

    def status(self) -> dict[str, Any]:
        return dict(JsonRpcClient(self.info.endpoint).call("status"))

    def shutdown(self) -> None:
        if not self.is_owner:
            return
        try:
            JsonRpcClient(self.info.endpoint, timeout=2.0).call("shutdown")
        except RpcError:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()

    def join(self, timeout: float = 15.0) -> None:
        if self.process is None:
            return
        if self.info.config.launch_on_all_ranks:
            timeout = max(timeout, 60.0)
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        finally:
            if self._log is not None:
                self._log.close()

    def check(self) -> None:
        if self.process is not None and self.process.poll() not in (None, 0):
            raise RuntimeError(
                f"service {self.name!r} exited with code {self.process.returncode}"
            )


class InferenceService(Service):
    pass


class TrainingService(Service):
    pass


class RolloutService(Service):
    pass


class ParameterServerService(Service):
    pass


class TrajectoryServerService(Service):
    pass


class RolloutCoordinatorService(Service):
    pass
