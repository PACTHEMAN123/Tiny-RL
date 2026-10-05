from __future__ import annotations

from collections.abc import Sequence

from .config import (
    InferenceServiceConfig,
    ParameterServerServiceConfig,
    RolloutCoordinatorServiceConfig,
    RolloutServiceConfig,
    ServiceGroup,
    TrajectoryServerServiceConfig,
    TrainingServiceConfig,
)
from .engine import ToyInferenceEngine, ToyTrainingEngine


def minimal_grpo_recipe(
    prompts: Sequence[str],
    *,
    staleness: int = 2,
    rollout_replicas: int = 2,
    groups_per_batch: int = 2,
    group_size: int = 2,
) -> tuple[ServiceGroup, ...]:
    """A minimal staleness-bounded recipe with replicated asynchronous rollout."""

    if groups_per_batch <= 0:
        raise ValueError("groups_per_batch must be positive")
    if rollout_replicas <= 0:
        raise ValueError("rollout_replicas must be positive")
    if len(prompts) == 0 or len(prompts) % groups_per_batch != 0:
        raise ValueError("prompt count must be divisible by groups_per_batch")

    return (
        ServiceGroup(
            id="parameter_server",
            config=ParameterServerServiceConfig(
                num_entries=groups_per_batch,
                staleness=staleness,
            ),
            n_gpus_per_replica=0,
        ),
        ServiceGroup(
            id="trajectory_server",
            config=TrajectoryServerServiceConfig(),
            n_gpus_per_replica=0,
        ),
        ServiceGroup(
            id="inference",
            config=InferenceServiceConfig(engine_factory=ToyInferenceEngine),
            n_replicas=rollout_replicas,
            n_gpus_per_replica=1,
        ),
        ServiceGroup(
            id="rollout_coordinator",
            config=RolloutCoordinatorServiceConfig(),
            n_gpus_per_replica=0,
            wait_until=("parameter_server",),
        ),
        ServiceGroup(
            id="trainer",
            config=TrainingServiceConfig(
                engine_factory=ToyTrainingEngine,
                batch_size=groups_per_batch * group_size,
                max_steps=len(prompts) // groups_per_batch,
            ),
            n_gpus_per_replica=1,
            wait_until=("parameter_server", "trajectory_server"),
        ),
        ServiceGroup(
            id="rollout",
            config=RolloutServiceConfig(
                prompts=prompts,
                reward_fn=lambda _prompt, generation: float(
                    "candidate=0" in generation.response
                ),
                group_size=group_size,
                max_prompts=len(prompts),
            ),
            n_replicas=rollout_replicas,
            n_gpus_per_replica=0,
            wait_until=(
                "inference",
                "trainer",
                "parameter_server",
                "trajectory_server",
                "rollout_coordinator",
            ),
        ),
    )
