from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .engine import TrainResult, TrainingEngine
from .rpc import write_json_atomic


class DeepSeekV41LoadOnlyEngine(TrainingEngine):
    """Keep a real 40-layer DSV4 checkpoint resident; fake optimizer steps."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        code_root: str | Path | None = None,
        *,
        num_layers: int = 40,
    ) -> None:
        self._weight_version = 0
        self._steps = 0
        self._lock = threading.Lock()

        checkpoint = Path(
            model_path
            or os.environ.get(
                "ORCH_MODEL_PATH", "/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash"
            )
        ).resolve()
        source = _resolve_code_root(code_root)
        _validate_model_inputs(checkpoint, source)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))

        import torch
        import torch.distributed as dist
        from dsv41_train.models.dsv4 import load_dsv41_backbone_window
        from dsv41_train.models.dsv4.parallel import (
            apply_fsdp2_layer,
            apply_fsdp2_root,
            build_parallelism,
        )
        from dsv41_train.runtime import initialize_runtime

        runtime = initialize_runtime(require_distributed=True)
        self._runtime = runtime
        world_size = runtime.world_size
        cp_size = int(os.environ.get("ORCH_MODEL_CP_SIZE", world_size))
        ep_size = int(os.environ.get("ORCH_MODEL_EP_SIZE", world_size))

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        meshes, context_parallel, token_dispatcher = build_parallelism(
            cp_size, ep_size, runtime.device.type
        )

        started = time.perf_counter()
        model = load_dsv41_backbone_window(
            checkpoint,
            start_layer=0,
            num_layers=num_layers,
            device=runtime.device,
            dtype=torch.bfloat16,
            context_parallel=context_parallel,
            token_dispatcher=token_dispatcher,
            engram_mesh=meshes.engram,
            sparse_engram_gradients=True,
            layer_loaded=lambda layer: apply_fsdp2_layer(layer, meshes),
        )
        torch.cuda.synchronize(runtime.device)
        loaded_at = time.perf_counter()

        local_expert_parameters = sum(
            layer.moe.routed.local_parameter_count for layer in model.model.layers
        )
        local_engram_parameters = sum(
            table.weight.numel() for table in model.model.engram_tables.values()
        )
        global_engram_parameters = sum(
            table.global_num_embeddings * model.config.engram_head_dim
            for table in model.model.engram_tables.values()
        )
        local_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        full_parameter_count = (
            local_parameter_count
            - local_expert_parameters
            - local_engram_parameters
            + local_expert_parameters * ep_size
            + global_engram_parameters
        )

        apply_fsdp2_root(model, meshes)
        model.eval()
        torch.cuda.synchronize(runtime.device)
        finished = time.perf_counter()

        rank_evidence: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(
            rank_evidence,
            {
                "rank": runtime.rank,
                "local_rank": runtime.local_rank,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "device": str(runtime.device),
                "load_seconds": loaded_at - started,
                "shard_seconds": finished - loaded_at,
                "peak_memory_gib": torch.cuda.max_memory_allocated(runtime.device)
                / 1024**3,
            },
        )
        self._model = model
        self._summary: dict[str, Any] = {
            "backend": type(self).__name__,
            "model": f"DeepSeek-V4.1-Flash {num_layers}-layer full checkpoint",
            "checkpoint": str(checkpoint),
            "parameters": full_parameter_count,
            "local_parameters_before_root_fsdp": local_parameter_count,
            "num_layers": num_layers,
            "world_size": world_size,
            "cp_size": cp_size,
            "ep_size": ep_size,
            "fsdp_size": meshes.fsdp.size(),
            "load_seconds": loaded_at - started,
            "shard_seconds": finished - loaded_at,
            "ranks": rank_evidence,
            "compute": "fake",
            "resident": True,
        }
        if runtime.is_main:
            runtime_dir = Path(os.environ["ORCH_RUNTIME_DIR"])
            write_json_atomic(runtime_dir / "full_model.json", self._summary)

    @property
    def weight_version(self) -> int:
        with self._lock:
            return self._weight_version

    def step(self, samples: Sequence[Mapping[str, Any]]) -> TrainResult:
        if not samples:
            raise ValueError("training batch cannot be empty")
        with self._lock:
            self._steps += 1
            self._weight_version += 1
            step = self._steps
            version = self._weight_version
        mean_reward = sum(float(sample["reward"]) for sample in samples) / len(samples)
        return TrainResult(
            step=step,
            weight_version=version,
            metrics={"mean_reward": mean_reward, "model_compute": 0.0},
            checkpoint_ref=f"resident://deepseek-v4.1/{version}",
        )

    def snapshot(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                **self._summary,
                "weight_version": self._weight_version,
                "fake_train_steps": self._steps,
            }

    def close(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def _resolve_code_root(code_root: str | Path | None) -> Path:
    if code_root is not None:
        return Path(code_root).resolve()
    configured = os.environ.get("ORCH_DSV41_ROOT")
    if configured:
        return Path(configured).resolve()
    workspace = Path(__file__).resolve().parents[2]
    for name in ("Tiny-DSV41", "dsv41-train"):
        candidate = workspace / name
        if candidate.is_dir():
            return candidate.resolve()
    return (workspace / "Tiny-DSV41").resolve()


def _validate_model_inputs(checkpoint: Path, source: Path) -> None:
    for path in (
        checkpoint / "config.json",
        checkpoint / "model.safetensors.index.json",
        source / "dsv41_train" / "__init__.py",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required full-model input does not exist: {path}")
