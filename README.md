# mini-rl orchestration

This directory contains a minimal, runnable orchestration skeleton for
disaggregated GRPO. Meshy contributes the Ray-free, role-driven runtime shape;
StaleFlow/PSRL owns the complete staleness protocol.

See [docs/DESIGN.md](docs/DESIGN.md) for the architecture and protocol details.

## Architecture

The recipe declares typed `ServiceGroup` objects. One `torchrun` process per card
all-gathers the global `GPU(host, global_rank, node_rank, local_rank)` list. Every
ignitor then derives the same complete topology and deterministic RPC endpoints,
and starts only the role subprocesses owned by its card. After startup there is no
central training driver: each service owns its long-running role loop.

```text
                                  status / SYNC / ABORT
RolloutWorker <----------------------------------------> RolloutCoordinator
      | Reserve / Occupy                                     (metadata only)
      v
  PSManager -------- model handles -------- ParameterServer <-------- Trainer
      |                                             ^                    |
      | reservation metadata                        | pull/push          | Lease / Consume
      |                                             |                    |
      +---------------- TrajectoryServer <----------+--------------------+
                         trajectory payload
```

The logical coordinator is not an execution driver. It coalesces the newest status
snapshot for each rollout producer and maintains a fenced, per-producer command
sequence. Rollout and trainer services continue to make progress through PSRL RPC-like
interfaces.

## StaleFlow protocol

`PSManager` owns `StalenessInventory`. A rollout group reserves the latest feasible
buffer before generation and occupies the earliest feasible buffer after the whole
GRPO group is ready. The invariant is:

```text
behavior_version <= training_buffer <= behavior_version + staleness
```

Trajectory bytes never enter the PS metadata path. `TrajectoryServer` stores the
payload by training buffer, while `PSManager` stores reservation and lease metadata.
The trainer leases the frontier buffer, fetches its payload, trains, pushes the new
version to `ParameterServer`, and only then consumes the PS metadata entry.

Rollout producers report coalesced status snapshots. The coordinator currently emits
only two ordered commands:

- `SYNC`: pull one specified version from `ParameterServer` at a safe group boundary.
- `ABORT`: fence and retry specified in-flight groups.

There is no sleep/wake or model-offload protocol.

## Launch

```bash
cd mini-rl
python3 scripts/launch.py --recipe recipes.staleflow_grpo
python3 -m unittest discover -s tests -v
```

The launch path intentionally matches Meshy: the same launcher is run on every node
with matching rendezvous arguments. For the included 16-card recipe:

```bash
python3 scripts/launch.py \
  --recipe recipes.staleflow_grpo_16gpu \
  --nnodes 4 \
  --node-rank <0..3> \
  --master-addr <node-0-address> \
  --master-port 29500 \
  --nproc-per-node 4 \
  --runtime-dir /shared/or/local/run/path
```

`scripts/launch.py` is bootstrap only. It imports the recipe to validate the card
count and launches one SPMD ignitor per card. It does not dispatch rollout or train
steps. Role processes communicate directly over the endpoints derived from topology.

## Full-model smoke run

The 16-card smoke recipe loads and keeps the complete 40-layer
DeepSeek-V4.1-Flash checkpoint resident in the trainer replica. Its rollout,
inference output, and optimizer step remain fake, so this validates orch topology,
role ownership, StaleFlow data flow, and real distributed model placement without
paying for a forward/backward pass.

```bash
python3 scripts/launch.py \
  --recipe recipes.staleflow_grpo_full_model_16gpu \
  --nnodes 4 --node-rank <0..3> \
  --master-addr <node-0-address> --master-port 29500 \
  --nproc-per-node 4 --runtime-dir /shared/run/path
```

The defaults expect the checkpoint at
`/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash` and the model implementation in a
sibling `Tiny-DSV41` checkout. Override them with `ORCH_MODEL_PATH` and
`ORCH_DSV41_ROOT`. Successful runs write model placement evidence to
`full_model.json` and include it under `training` in `result.json`.

## Current backends

The default recipes use toy engines. The full-model smoke recipe adds a real
distributed checkpoint loader with fake compute. PS, trajectory, coordinator,
rollout, and trainer remain independent role processes executing the StaleFlow
protocol. Engine, parameter storage, trajectory storage, and transport boundaries
are explicit so production backends can replace them without changing algorithm
roles.
