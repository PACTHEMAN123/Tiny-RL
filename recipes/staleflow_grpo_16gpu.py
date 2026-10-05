from __future__ import annotations

from orch import (
    Ignitor,
    InferenceServiceConfig,
    ParameterServerServiceConfig,
    RolloutCoordinatorServiceConfig,
    RolloutServiceConfig,
    ServiceGroup,
    ToyInferenceEngine,
    ToyTrainingEngine,
    TrainingServiceConfig,
    TrajectoryServerServiceConfig,
)


def reward(_prompt: object, generation: object) -> float:
    return float("candidate=0" in generation.response)


PROMPTS = tuple(f"multi-node-prompt-{index}" for index in range(32))
SERVICE_GROUPS = (
    ServiceGroup(
        id="parameter_server",
        config=ParameterServerServiceConfig(num_entries=4, staleness=2),
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
        n_replicas=8,
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
            batch_size=8,
            max_steps=8,
        ),
        n_gpus_per_replica=8,
        wait_until=("parameter_server", "trajectory_server"),
    ),
    ServiceGroup(
        id="rollout",
        config=RolloutServiceConfig(
            prompts=PROMPTS,
            reward_fn=reward,
            group_size=2,
            max_prompts=len(PROMPTS),
        ),
        n_replicas=8,
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


if __name__ == "__main__":
    Ignitor(SERVICE_GROUPS, recipe_module=__name__, timeout=300.0).run()
