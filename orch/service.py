from __future__ import annotations

import os
import subprocess
import sys
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

    def ignite(self) -> None:
        if not self.is_owner:
            return
        log_dir = self.runtime_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.name}.log"
        self._log = log_path.open("ab", buffering=0)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
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
        try:
            JsonRpcClient(self.info.endpoint).wait_ready(timeout)
        except BaseException:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"service {self.name!r} exited with code {self.process.returncode}; "
                    f"see {self.runtime_dir / 'logs' / f'{self.name}.log'}"
                )
            raise

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
