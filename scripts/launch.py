"""Launch one card-level orch ignitor per GPU, following Meshy's SPMD shape."""

from __future__ import annotations

import argparse
import importlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orch.topology import required_gpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="recipes.staleflow_grpo")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--runtime-dir", default=None)
    return parser


def _torchrun_command(args: argparse.Namespace, nproc: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "orch.torchrun",
        "--nnodes",
        str(args.nnodes),
        "--node-rank",
        str(args.node_rank),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(args.master_port),
        "--nproc-per-node",
        str(nproc),
        "-m",
        args.recipe,
    ]


def main() -> None:
    args = _parser().parse_args()
    repo_root = REPO_ROOT

    recipe = importlib.import_module(args.recipe)
    cards = required_gpus(tuple(recipe.SERVICE_GROUPS))
    if cards <= 0:
        raise SystemExit("recipe does not declare any GPU services")
    if args.nproc_per_node is None:
        if args.nnodes != 1:
            raise SystemExit("--nproc-per-node is required for multi-node launches")
        nproc = cards
    else:
        nproc = args.nproc_per_node
    if args.nnodes * nproc != cards:
        raise SystemExit(
            f"recipe needs {cards} cards, but launch declares "
            f"{args.nnodes} x {nproc} = {args.nnodes * nproc}"
        )

    runtime_dir = args.runtime_dir or os.environ.get("ORCH_RUNTIME_DIR")
    if runtime_dir is None:
        runtime_dir = str(
            repo_root / ".orch_runtime" / time.strftime("%Y%m%d-%H%M%S")
        )
    Path(runtime_dir).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ORCH_RECIPE"] = args.recipe
    env["ORCH_RUNTIME_DIR"] = runtime_dir
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    print(
        f"[launch] recipe={args.recipe} cards={cards} "
        f"nnodes={args.nnodes} nproc-per-node={nproc} runtime={runtime_dir}",
        flush=True,
    )
    process = subprocess.Popen(_torchrun_command(args, nproc), env=env, cwd=repo_root)

    def terminate(*_: object) -> None:
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    try:
        return_code = process.wait()
    finally:
        terminate()
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
