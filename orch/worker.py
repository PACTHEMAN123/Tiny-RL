from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .config import RolloutServiceConfig, TrainingServiceConfig
from .coordination import (
    CommandType,
    ProducerLifecycle,
    ProducerSnapshot,
    RolloutCoordinator,
)
from .engine import InferenceEngine, TrainingEngine
from .ps import PSManager, ParameterServer, ReservationToken
from .queue import QueueClosed
from .trajectory import TrajectoryServer


class Worker:
    """Long-running role loop owned by a Service."""

    def __init__(self, name: str, stop_event: threading.Event) -> None:
        self.name = name
        self._stop_event = stop_event
        self._thread: threading.Thread | None = None
        self._exception: BaseException | None = None

    @property
    def done(self) -> bool:
        return self._thread is not None and not self._thread.is_alive()

    @property
    def exception(self) -> BaseException | None:
        return self._exception

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"worker {self.name!r} already started")
        self._thread = threading.Thread(target=self._run_guarded, name=self.name, daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run_guarded(self) -> None:
        try:
            self.run()
        except QueueClosed:
            if not self._stop_event.is_set():
                self._exception = QueueClosed("trajectory server closed before worker completed")
        except BaseException as exc:  # surfaced by Ignitor on the owner thread
            self._exception = exc
            self._stop_event.set()

    def run(self) -> None:
        raise NotImplementedError


class TrainingWorker(Worker):
    def __init__(
        self,
        name: str,
        stop_event: threading.Event,
        config: TrainingServiceConfig,
        engine: TrainingEngine,
        ps_manager: Callable[[], PSManager],
        parameter_server: Callable[[], ParameterServer],
        trajectory_server: Callable[[], TrajectoryServer],
    ) -> None:
        super().__init__(name, stop_event)
        self._config = config
        self._engine = engine
        self._ps_manager = ps_manager
        self._parameter_server = parameter_server
        self._trajectory_server = trajectory_server

    def run(self) -> None:
        manager = self._ps_manager()
        parameter_server = self._parameter_server()
        trajectory_server = self._trajectory_server()

        for _ in range(self._config.max_steps):
            lease = None
            while lease is None and not self._stop_event.is_set():
                lease = manager.lease_train_buffer(timeout=0.1)
            if lease is None:
                return

            batch = None
            while batch is None and not self._stop_event.is_set():
                batch = trajectory_server.lease_buffer(
                    base_partition=self._config.rollout_partition,
                    lease=lease,
                    fields=self._config.input_fields,
                    batch_size=self._config.batch_size,
                    consumer=self.name,
                    timeout=0.1,
                )
            if batch is None:
                return

            update = self._engine.step(batch.rows)
            parameter_server.push(update)
            manager.consume_buffer(lease, produced_version=update.weight_version)
            trajectory_server.consume(batch)


class RolloutWorker(Worker):
    def __init__(
        self,
        name: str,
        stop_event: threading.Event,
        config: RolloutServiceConfig,
        inference_engine: Callable[[], InferenceEngine],
        ps_manager: Callable[[], PSManager],
        parameter_server: Callable[[], ParameterServer],
        trajectory_server: Callable[[], TrajectoryServer],
        coordinator: Callable[[], RolloutCoordinator],
        replica_index: int,
        replica_count: int,
    ) -> None:
        super().__init__(name, stop_event)
        self._config = config
        self._inference_engine = inference_engine
        self._ps_manager = ps_manager
        self._parameter_server = parameter_server
        self._trajectory_server = trajectory_server
        self._coordinator = coordinator
        self._replica_index = replica_index
        self._replica_count = replica_count
        self._producer_epoch = 1
        self._snapshot_seq = 0
        self._applied_command_seq = 0
        self._lifecycle = ProducerLifecycle.ACTIVE

    def run(self) -> None:
        engine = self._inference_engine()
        manager = self._ps_manager()
        parameter_server = self._parameter_server()
        trajectory_server = self._trajectory_server()
        coordinator = self._coordinator()
        coordinator.register(self.name, self._producer_epoch, engine.weight_version)

        for prompt_index, prompt in enumerate(self._prompts()):
            if self._stop_event.is_set():
                return
            if (
                self._config.max_prompts is not None
                and prompt_index >= self._config.max_prompts
            ):
                break
            if prompt_index % self._replica_count != self._replica_index:
                continue

            group_id = f"prompt-{prompt_index}"
            completed = False
            while not completed and not self._stop_event.is_set():
                self._exchange_status(
                    engine,
                    parameter_server,
                    coordinator,
                    running_group_ids=(),
                )

                token = manager.reserve_rollout_group(
                    group_id=group_id,
                    rollout_instance_id=self.name,
                    model_version=engine.weight_version,
                )
                if token is None:
                    self._exchange_status(
                        engine,
                        parameter_server,
                        coordinator,
                        running_group_ids=(),
                        waiting_requests=1,
                    )
                    time.sleep(0.01)
                    continue

                cancelled = self._exchange_status(
                    engine,
                    parameter_server,
                    coordinator,
                    running_group_ids=(group_id,),
                    active_token=token,
                    manager=manager,
                )
                if cancelled:
                    continue

                try:
                    generations = [
                        engine.generate(prompt, weight_version=token.behavior_version)
                        for _ in range(self._config.group_size)
                    ]
                    rewards = [
                        float(self._config.reward_fn(prompt, generation))
                        for generation in generations
                    ]
                except BaseException:
                    manager.abort_rollout_group(token)
                    raise

                cancelled = self._exchange_status(
                    engine,
                    parameter_server,
                    coordinator,
                    running_group_ids=(group_id,),
                    active_token=token,
                    manager=manager,
                )
                if cancelled:
                    continue

                occupied = manager.occupy_rollout_group(token)
                rows = _build_grpo_rows(
                    prompt=prompt,
                    generations=generations,
                    rewards=rewards,
                    weight_version=token.behavior_version,
                    extra_columns=self._config.extra_columns,
                )
                trajectory_server.put_group(
                    base_partition=self._config.rollout_partition,
                    token=token,
                    occupied=occupied,
                    rows=rows,
                )
                completed = True

        self._exchange_status(
            engine,
            parameter_server,
            coordinator,
            running_group_ids=(),
        )

    def _exchange_status(
        self,
        engine: InferenceEngine,
        parameter_server: ParameterServer,
        coordinator: RolloutCoordinator,
        running_group_ids: tuple[str, ...],
        waiting_requests: int = 0,
        active_token: ReservationToken | None = None,
        manager: PSManager | None = None,
    ) -> bool:
        """Publish newest status, then apply commands in sequence at a safe point."""

        cancelled = False
        self._submit_snapshot(
            engine,
            coordinator,
            running_group_ids,
            waiting_requests,
        )
        commands = coordinator.poll_commands(
            producer_id=self.name,
            producer_epoch=self._producer_epoch,
            after_command_seq=self._applied_command_seq,
        )
        for command in commands:
            if command.command_type is CommandType.SYNC:
                if active_token is not None and manager is not None and not cancelled:
                    manager.abort_rollout_group(active_token)
                    cancelled = True
                if command.target_version is None:
                    raise RuntimeError("SYNC command requires a target version")
                self._lifecycle = ProducerLifecycle.SYNCING
                self._submit_snapshot(
                    engine,
                    coordinator,
                    running_group_ids if not cancelled else (),
                    waiting_requests,
                )
                engine.load_weights(parameter_server.pull(command.target_version))
            elif command.command_type is CommandType.ABORT:
                if (
                    active_token is not None
                    and manager is not None
                    and active_token.group_id in command.group_ids
                    and not cancelled
                ):
                    manager.abort_rollout_group(active_token)
                    cancelled = True
            else:  # pragma: no cover - enum exhaustiveness guard
                raise RuntimeError(f"unsupported coordinator command: {command.command_type}")

            self._applied_command_seq = command.command_seq
            coordinator.complete_command(
                producer_id=self.name,
                producer_epoch=self._producer_epoch,
                command_seq=command.command_seq,
            )

        self._lifecycle = ProducerLifecycle.ACTIVE
        if commands:
            self._submit_snapshot(
                engine,
                coordinator,
                running_group_ids if not cancelled else (),
                waiting_requests,
            )
        return cancelled

    def _submit_snapshot(
        self,
        engine: InferenceEngine,
        coordinator: RolloutCoordinator,
        running_group_ids: tuple[str, ...],
        waiting_requests: int,
    ) -> None:
        self._snapshot_seq += 1
        coordinator.submit_snapshot(
            ProducerSnapshot(
                producer_id=self.name,
                producer_epoch=self._producer_epoch,
                model_version=engine.weight_version,
                lifecycle=self._lifecycle,
                running_group_ids=running_group_ids,
                waiting_requests=waiting_requests,
                snapshot_seq=self._snapshot_seq,
                applied_command_seq=self._applied_command_seq,
            )
        )

    def _prompts(self) -> Iterable[Any]:
        prompts = self._config.prompts
        return prompts() if callable(prompts) else iter(prompts)


def _normalize_group_rewards(rewards: Sequence[float]) -> tuple[float, ...]:
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std == 0:
        return tuple(0.0 for _ in rewards)
    return tuple((reward - mean) / std for reward in rewards)


def _build_grpo_rows(
    prompt: Any,
    generations: Sequence[Any],
    rewards: Sequence[float],
    weight_version: int,
    extra_columns: Any = None,
) -> list[dict[str, Any]]:
    advantages = _normalize_group_rewards(rewards)
    rows = []
    for generation, reward, advantage in zip(generations, rewards, advantages):
        row = {
            "prompt": prompt,
            "response": generation.response,
            "tokens": generation.tokens,
            "logprobs": generation.logprobs,
            "reward": reward,
            "advantage": advantage,
            "weight_version": weight_version,
        }
        if extra_columns:
            row.update(extra_columns)
        rows.append(row)
    return rows
