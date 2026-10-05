from __future__ import annotations

import unittest

from orch import (
    ColumnQueue,
    CpuInferenceServiceConfig,
    Ignitor,
    InferenceServiceConfig,
    ParameterServerServiceConfig,
    RolloutCoordinatorServiceConfig,
    RolloutServiceConfig,
    ServiceGroup,
    SpmdTrainingServiceConfig,
    ToyInferenceEngine,
    ToyTrainingEngine,
    TrajectoryServerServiceConfig,
    TrainingServiceConfig,
    build_topology,
)
def reward_candidate_zero(_prompt: object, generation: object) -> float:
    return float("candidate=0" in generation.response)


class TopologyTest(unittest.TestCase):
    def test_disaggregated_gpu_placement_is_deterministic(self) -> None:
        groups = (
            ServiceGroup(
                id="inference",
                config=InferenceServiceConfig(ToyInferenceEngine),
                n_replicas=2,
                n_gpus_per_replica=1,
            ),
            ServiceGroup(
                id="trainer",
                config=TrainingServiceConfig(ToyTrainingEngine, 2, 1),
                n_gpus_per_replica=1,
            ),
            ServiceGroup(
                id="rollout",
                config=RolloutServiceConfig(
                    prompts=("prompt",),
                    reward_fn=reward_candidate_zero,
                    group_size=2,
                ),
                n_gpus_per_replica=0,
            ),
        )

        topology = build_topology(groups, (4, 5, 9))

        self.assertEqual(
            [service.gpu_ids for service in topology.services],
            [(4,), (5,), (9,), ()],
        )

    def test_colocation_is_rejected(self) -> None:
        group = ServiceGroup(
            id="trainer",
            config=TrainingServiceConfig(ToyTrainingEngine, 2, 1),
            colocate_with="inference",
        )
        with self.assertRaisesRegex(ValueError, "colocation"):
            build_topology((group,), (0,))

    def test_cpu_inference_leaves_every_card_for_spmd_training(self) -> None:
        groups = (
            ServiceGroup(
                id="inference",
                config=CpuInferenceServiceConfig(ToyInferenceEngine),
                n_replicas=2,
                n_gpus_per_replica=0,
            ),
            ServiceGroup(
                id="trainer",
                config=SpmdTrainingServiceConfig(ToyTrainingEngine, 2, 1),
                n_gpus_per_replica=3,
            ),
        )

        topology = build_topology(groups, (0, 1, 2))

        self.assertEqual(topology.role("inference")[0].gpu_ids, ())
        trainer = topology.role("training")[0]
        self.assertEqual(trainer.gpu_ids, (0, 1, 2))
        self.assertEqual(trainer.dist_port, 43000)
        self.assertTrue(trainer.config.launch_on_all_ranks)


class ColumnQueueTest(unittest.TestCase):
    def test_rows_are_consumable_only_when_all_columns_are_ready(self) -> None:
        queue = ColumnQueue()
        refs = queue.publish("rollout", ({"prompt": "hello"},))

        self.assertIsNone(
            queue.fetch(
                "rollout",
                fields=("prompt", "reward"),
                batch_size=1,
                consumer="trainer",
                timeout=0,
            )
        )

        queue.update(refs, ({"reward": 1.0},))
        batch = queue.fetch(
            "rollout",
            fields=("prompt", "reward"),
            batch_size=1,
            consumer="trainer",
            timeout=0,
        )

        self.assertIsNotNone(batch)
        self.assertEqual(batch.rows[0], {"prompt": "hello", "reward": 1.0})


class EndToEndTest(unittest.TestCase):
    def test_rejects_a_group_size_that_cannot_fill_training_batches(self) -> None:
        groups = (
            ServiceGroup(
                id="inference",
                config=InferenceServiceConfig(ToyInferenceEngine),
            ),
            ServiceGroup(
                id="parameter_server",
                config=ParameterServerServiceConfig(num_entries=1, staleness=1),
                n_gpus_per_replica=0,
            ),
            ServiceGroup(
                id="trajectory_server",
                config=TrajectoryServerServiceConfig(),
                n_gpus_per_replica=0,
            ),
            ServiceGroup(
                id="rollout_coordinator",
                config=RolloutCoordinatorServiceConfig(),
                n_gpus_per_replica=0,
            ),
            ServiceGroup(
                id="trainer",
                config=TrainingServiceConfig(ToyTrainingEngine, 3, 1),
            ),
            ServiceGroup(
                id="rollout",
                config=RolloutServiceConfig(
                    prompts=("p0",),
                    reward_fn=reward_candidate_zero,
                    group_size=2,
                ),
                n_gpus_per_replica=0,
            ),
        )

        with self.assertRaisesRegex(ValueError, "divisible"):
            Ignitor(groups, recipe_module="recipes.staleflow_grpo")


if __name__ == "__main__":
    unittest.main()
