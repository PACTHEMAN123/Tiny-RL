from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Generation:
    response: str
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]


@dataclass(frozen=True)
class TrainResult:
    step: int
    weight_version: int
    metrics: Mapping[str, float] = field(default_factory=dict)
    checkpoint_ref: str | None = None


class InferenceEngine(ABC):
    @property
    @abstractmethod
    def weight_version(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: Any, *, weight_version: int) -> Generation:
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, update: TrainResult) -> None:
        raise NotImplementedError


class TrainingEngine(ABC):
    @property
    @abstractmethod
    def weight_version(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def step(self, samples: Sequence[Mapping[str, Any]]) -> TrainResult:
        raise NotImplementedError


class ToyInferenceEngine(InferenceEngine):
    """Deterministic fake engine used by the runnable example and tests."""

    def __init__(self) -> None:
        self._weight_version = 0
        self._generation_index = 0
        self._lock = threading.Lock()

    @property
    def weight_version(self) -> int:
        with self._lock:
            return self._weight_version

    def generate(self, prompt: Any, *, weight_version: int) -> Generation:
        with self._lock:
            if weight_version > self._weight_version:
                raise RuntimeError(
                    f"requested weight version {weight_version}, "
                    f"but inference holds {self._weight_version}"
                )
            candidate = self._generation_index % 2
            self._generation_index += 1
            response = f"{prompt} | candidate={candidate} | v={weight_version}"

        tokens = tuple(response.encode("utf-8"))
        logprobs = tuple(-0.1 for _ in tokens)
        return Generation(response=response, tokens=tokens, logprobs=logprobs)

    def load_weights(self, update: TrainResult) -> None:
        with self._lock:
            if update.weight_version < self._weight_version:
                raise RuntimeError("weight versions must be monotonic")
            self._weight_version = update.weight_version


class ToyTrainingEngine(TrainingEngine):
    """A fake optimizer that records batches and increments a weight version."""

    def __init__(self) -> None:
        self._weight_version = 0
        self.batches: list[tuple[dict[str, Any], ...]] = []

    @property
    def weight_version(self) -> int:
        return self._weight_version

    def step(self, samples: Sequence[Mapping[str, Any]]) -> TrainResult:
        batch = tuple(dict(sample) for sample in samples)
        self.batches.append(batch)
        self._weight_version += 1
        mean_reward = sum(float(sample["reward"]) for sample in batch) / len(batch)
        return TrainResult(
            step=len(self.batches),
            weight_version=self._weight_version,
            metrics={"mean_reward": mean_reward},
            checkpoint_ref=f"memory://weights/{self._weight_version}",
        )
