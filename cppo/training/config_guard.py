"""Fail-fast guards for archived experiment configs."""

from __future__ import annotations


def assert_not_archived_config(
    *,
    config_path: str,
    config_doc: dict,
    allow_archived: bool,
) -> None:
    """Reject archived configs unless the caller explicitly opts into ablations."""
    if allow_archived:
        return

    if not config_doc.get("archived", False):
        _assert_paper_reward_modes(config_path=config_path, config_doc=config_doc)
        return

    reason = config_doc.get("archive_reason", "no reason recorded")
    raise RuntimeError(
        f"{config_path} is an archived config: {reason}. "
        "Use the paper-binary configs for the CPPO_new.pdf path, or pass "
        "--allow-archived-config only when intentionally running an ablation."
    )


def _assert_paper_reward_modes(*, config_path: str, config_doc: dict) -> None:
    """Reject accidental use of legacy/argmax reward modes in unarchived configs."""
    non_paper_rm_modes = {"continuous", "group_argmax_binary"}
    non_paper_plan_modes = {
        "product",
        "rm_only",
        "rm_times_outcome",
        "outcome_first_winner",
    }

    for section_name in ("warmup", "cppo"):
        section = config_doc.get(section_name)
        if not isinstance(section, dict):
            continue

        rm_mode = section.get("rm_reward_mode")
        if rm_mode in non_paper_rm_modes:
            raise RuntimeError(
                f"{config_path} uses non-paper reward mode "
                f"rm_reward_mode={rm_mode!r}. Use binary_jpsi for the "
                "CPPO_new.pdf path, or pass --allow-archived-config only "
                "when intentionally running an ablation."
            )

        plan_mode = section.get("plan_reward_mode")
        if plan_mode in non_paper_plan_modes:
            raise RuntimeError(
                f"{config_path} uses non-paper reward mode "
                f"plan_reward_mode={plan_mode!r}. Use jpsi_times_outcome "
                "for the CPPO_new.pdf path, or pass --allow-archived-config "
                "only when intentionally running an ablation."
            )
