from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Iterable, Mapping, Sequence, Union


GRPO_FIELDS = (
    "prompt",
    "response",
    "tokens",
    "logprobs",
    "reward",
    "advantage",
    "weight_version",
)


EngineFactory = Callable[[], Any]
PromptSource = Union[Sequence[Any], Callable[[], Iterable[Any]]]
RewardFn = Callable[[Any, Any], float]


@dataclass(frozen=True)
class ServiceConfig:
    """Declarative service configuration, mirroring Meshy's typed configs."""

    role: ClassVar[str]
    service_cls: ClassVar[str]
    uses_gpu: ClassVar[bool]
    endpoint_port_base: ClassVar[int]


@dataclass(frozen=True)
class InferenceServiceConfig(ServiceConfig):
    engine_factory: EngineFactory

    role: ClassVar[str] = "inference"
    service_cls: ClassVar[str] = "orch.service:InferenceService"
    uses_gpu: ClassVar[bool] = True
    endpoint_port_base: ClassVar[int] = 32000


@dataclass(frozen=True)
class TrainingServiceConfig(ServiceConfig):
    engine_factory: EngineFactory
    batch_size: int
    max_steps: int
    input_fields: tuple[str, ...] = GRPO_FIELDS
    rollout_partition: str = "rollout"

    role: ClassVar[str] = "training"
    service_cls: ClassVar[str] = "orch.service:TrainingService"
    uses_gpu: ClassVar[bool] = True
    endpoint_port_base: ClassVar[int] = 33000

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


@dataclass(frozen=True)
class RolloutServiceConfig(ServiceConfig):
    prompts: PromptSource
    reward_fn: RewardFn
    group_size: int
    max_prompts: int | None = None
    rollout_partition: str = "rollout"
    extra_columns: Mapping[str, Any] | None = None

    role: ClassVar[str] = "rollout"
    service_cls: ClassVar[str] = "orch.service:RolloutService"
    uses_gpu: ClassVar[bool] = False
    endpoint_port_base: ClassVar[int] = 34000

    def __post_init__(self) -> None:
        if self.group_size <= 1:
            raise ValueError("GRPO group_size must be greater than one")
        if self.max_prompts is not None and self.max_prompts <= 0:
            raise ValueError("max_prompts must be positive")


@dataclass(frozen=True)
class ParameterServerServiceConfig(ServiceConfig):
    num_entries: int
    staleness: int
    initial_version: int = 0

    role: ClassVar[str] = "parameter_server"
    service_cls: ClassVar[str] = "orch.service:ParameterServerService"
    uses_gpu: ClassVar[bool] = False
    endpoint_port_base: ClassVar[int] = 35000

    def __post_init__(self) -> None:
        if self.num_entries <= 0:
            raise ValueError("num_entries must be positive")
        if self.staleness < 0:
            raise ValueError("staleness cannot be negative")
        if self.initial_version < 0:
            raise ValueError("initial_version cannot be negative")


@dataclass(frozen=True)
class TrajectoryServerServiceConfig(ServiceConfig):
    role: ClassVar[str] = "trajectory_server"
    service_cls: ClassVar[str] = "orch.service:TrajectoryServerService"
    uses_gpu: ClassVar[bool] = False
    endpoint_port_base: ClassVar[int] = 36000


@dataclass(frozen=True)
class RolloutCoordinatorServiceConfig(ServiceConfig):
    role: ClassVar[str] = "rollout_coordinator"
    service_cls: ClassVar[str] = "orch.service:RolloutCoordinatorService"
    uses_gpu: ClassVar[bool] = False
    endpoint_port_base: ClassVar[int] = 37000


@dataclass(frozen=True)
class ServiceGroup:
    """A homogeneous replicated service group declared by the recipe."""

    id: str
    config: ServiceConfig
    n_replicas: int = 1
    n_gpus_per_replica: int = 1
    wait_until: tuple[str, ...] = ()
    colocate_with: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("service group id cannot be empty")
        if self.n_replicas <= 0:
            raise ValueError("n_replicas must be positive")
        if self.n_gpus_per_replica < 0:
            raise ValueError("n_gpus_per_replica cannot be negative")
        if self.config.uses_gpu and self.n_gpus_per_replica == 0:
            raise ValueError(f"GPU service {self.id!r} needs at least one GPU")
        if not self.config.uses_gpu and self.n_gpus_per_replica != 0:
            raise ValueError(f"CPU service {self.id!r} must request zero GPUs")

    @property
    def n_gpus(self) -> int:
        return self.n_replicas * self.n_gpus_per_replica
