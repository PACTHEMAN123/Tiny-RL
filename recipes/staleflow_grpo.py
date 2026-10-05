from __future__ import annotations

from orch import Ignitor, minimal_grpo_recipe


PROMPTS = tuple(f"prompt-{index}" for index in range(8))
SERVICE_GROUPS = minimal_grpo_recipe(
    PROMPTS,
    staleness=1,
    rollout_replicas=2,
    groups_per_batch=2,
    group_size=2,
)


if __name__ == "__main__":
    Ignitor(SERVICE_GROUPS, recipe_module=__name__).run()
