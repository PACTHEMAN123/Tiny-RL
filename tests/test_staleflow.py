from __future__ import annotations

import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from orch import (
    BufferStatus,
    CommandType,
    FencedProducerError,
    FencedReservationError,
    ParameterServer,
    ProducerLifecycle,
    ProducerSnapshot,
    PSManager,
    RolloutCoordinator,
    TrainResult,
    TrajectoryServer,
)


class PSManagerTest(unittest.TestCase):
    def test_reserve_latest_occupy_earliest_and_consume(self) -> None:
        parameter_server = ParameterServer()
        manager = PSManager(parameter_server, num_entries=2, staleness=1)
        tokens = []
        for index in range(4):
            token = manager.reserve_rollout_group(
                group_id=f"group-{index}",
                rollout_instance_id="rollout-0",
                model_version=0,
            )
            self.assertIsNotNone(token)
            tokens.append(token)

        snapshot = manager.snapshot()
        self.assertEqual(snapshot.buffer(0).reserved, 2)
        self.assertEqual(snapshot.buffer(1).reserved, 2)

        first = manager.occupy_rollout_group(tokens[0])
        second = manager.occupy_rollout_group(tokens[1])
        self.assertEqual(first.occupied_buffer, 0)
        self.assertEqual(second.occupied_buffer, 0)
        self.assertEqual(manager.snapshot().buffer(0).state, BufferStatus.READY)

        lease = manager.lease_train_buffer(timeout=0)
        self.assertIsNotNone(lease)
        update = TrainResult(step=1, weight_version=1)
        with self.assertRaisesRegex(RuntimeError, "pushed"):
            manager.consume_buffer(lease, produced_version=1)
        parameter_server.push(update)
        manager.consume_buffer(lease, produced_version=1)

        snapshot = manager.snapshot()
        self.assertEqual(snapshot.frontier, 1)
        self.assertEqual(snapshot.committed_version, 1)
        self.assertEqual(snapshot.buffer(0).state, BufferStatus.CONSUMED)

    def test_owner_epoch_fences_a_late_completion(self) -> None:
        manager = PSManager(ParameterServer(), num_entries=1, staleness=1)
        old_token = manager.reserve_rollout_group("group", "rollout-0", 0)
        self.assertIsNotNone(old_token)
        manager.abort_rollout_group(old_token)
        new_token = manager.reserve_rollout_group("group", "rollout-1", 0)
        self.assertIsNotNone(new_token)
        self.assertGreater(new_token.owner_epoch, old_token.owner_epoch)

        with self.assertRaises(FencedReservationError):
            manager.occupy_rollout_group(old_token)
        self.assertEqual(manager.snapshot().fenced_completions, 1)


class RolloutCoordinatorTest(unittest.TestCase):
    def test_coalesces_status_and_orders_sync_and_abort_commands(self) -> None:
        coordinator = RolloutCoordinator(latest_model_version=lambda: 2)
        coordinator.register("producer", producer_epoch=1, model_version=0)
        status = ProducerSnapshot(
            producer_id="producer",
            producer_epoch=1,
            model_version=0,
            lifecycle=ProducerLifecycle.ACTIVE,
            running_group_ids=("group-0",),
            waiting_requests=1,
            snapshot_seq=1,
            applied_command_seq=0,
        )

        self.assertTrue(coordinator.submit_snapshot(status))
        self.assertFalse(coordinator.submit_snapshot(status))
        abort = coordinator.issue_abort("producer", 1, ("group-0",))
        commands = coordinator.poll_commands("producer", 1, after_command_seq=0)

        self.assertEqual(
            [command.command_type for command in commands],
            [CommandType.SYNC, CommandType.ABORT],
        )
        self.assertEqual(commands[0].target_version, 2)
        self.assertEqual(abort.command_seq, 2)

        coordinator.complete_command("producer", 1, command_seq=2)
        snapshot = coordinator.snapshot()
        producer = snapshot.producers[0]
        self.assertEqual(producer.running_group_ids, ("group-0",))
        self.assertEqual(producer.waiting_requests, 1)
        self.assertEqual(producer.pending_commands, ())
        self.assertEqual(snapshot.issued_commands, 2)
        self.assertEqual(snapshot.completed_commands, 2)
        self.assertEqual(snapshot.accepted_snapshots, 1)
        self.assertEqual(snapshot.dropped_snapshots, 1)

        coordinator.register("producer", producer_epoch=2, model_version=2)
        with self.assertRaises(FencedProducerError):
            coordinator.poll_commands("producer", 1, after_command_seq=0)


class TrajectoryServerTest(unittest.TestCase):
    def test_payload_lifecycle_is_separate_from_ps_metadata(self) -> None:
        manager = PSManager(ParameterServer(), num_entries=1, staleness=0)
        server = TrajectoryServer()
        token = manager.reserve_rollout_group("group", "rollout-0", 0)
        self.assertIsNotNone(token)
        occupied = manager.occupy_rollout_group(token)
        server.put_group(
            "rollout",
            token,
            occupied,
            ({"prompt": "p0"}, {"prompt": "p0"}),
        )

        lease = manager.lease_train_buffer(timeout=0)
        self.assertIsNotNone(lease)
        batch = server.lease_buffer(
            "rollout",
            lease,
            fields=("prompt",),
            batch_size=2,
            consumer="trainer",
            timeout=0,
        )
        self.assertIsNotNone(batch)
        self.assertEqual(
            {row["reservation_id"] for row in batch.rows},
            {token.reservation_id},
        )
        self.assertEqual(server.snapshot().live_rows, 2)
        server.consume(batch)
        self.assertEqual(server.snapshot().live_rows, 0)


class StaleFlowEndToEndTest(unittest.TestCase):
    def test_spmd_launch_runs_disaggregated_grpo(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            process = subprocess.run(
                [
                    sys.executable,
                    "scripts/launch.py",
                    "--recipe",
                    "recipes.staleflow_grpo",
                    "--runtime-dir",
                    str(runtime),
                    "--master-port",
                    "29631",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = json.loads((runtime / "result.json").read_text())

        parameters = result["parameter_server"]["parameters"]
        inventory = result["parameter_server"]["inventory"]
        trajectory = result["trajectory_server"]
        coordinator = result["rollout_coordinator"]
        self.assertEqual(result["topology"]["world_size"], 3)
        self.assertEqual(parameters["latest_version"], 4)
        self.assertEqual(inventory["frontier"], 4)
        self.assertEqual(inventory["consumed_buffers"], 4)
        self.assertEqual(trajectory["published_groups"], 8)
        self.assertEqual(trajectory["consumed_batches"], 4)
        self.assertEqual(trajectory["live_rows"], 0)
        self.assertGreater(coordinator["issued_commands"], 0)
        self.assertEqual(
            coordinator["completed_commands"], coordinator["issued_commands"]
        )


if __name__ == "__main__":
    unittest.main()
