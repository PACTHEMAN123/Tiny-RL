"""Compatibility entrypoint for the environment's official torchrun module."""

from __future__ import annotations

import sys
import types


def main() -> None:
    try:
        import importlib_metadata  # noqa: F401
    except ModuleNotFoundError:
        from importlib import metadata

        shim = types.ModuleType("importlib_metadata")
        shim.__dict__.update(metadata.__dict__)

        def entry_points(**params: str):
            discovered = metadata.entry_points()
            if not params:
                return discovered
            if hasattr(discovered, "select"):
                return discovered.select(**params)
            group = params.get("group")
            if group is None:
                return discovered
            return discovered.get(group, ())

        shim.entry_points = entry_points
        sys.modules["importlib_metadata"] = shim

    from torch.distributed.run import main as torchrun_main

    torchrun_main()


if __name__ == "__main__":
    main()
