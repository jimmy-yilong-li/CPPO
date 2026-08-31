"""Metrics for planner SFT targets."""

from __future__ import annotations

import torch


def empty_error_counts() -> dict[str, dict[str, int | float | None]]:
    return {
        name: {"count": 0, "correct": 0, "accuracy": None, "error": None}
        for name in ("target", "think", "plan")
    }


def finalize_error_counts(
    counts: dict[str, dict[str, int | float | None]],
) -> dict[str, dict[str, int | float | None]]:
    out: dict[str, dict[str, int | float | None]] = {}
    for name, values in counts.items():
        count = int(values["count"] or 0)
        correct = int(values["correct"] or 0)
        accuracy = (correct / count) if count else None
        out[name] = {
            "count": count,
            "correct": correct,
            "accuracy": accuracy,
            "error": (1.0 - accuracy) if accuracy is not None else None,
        }
    return out


def merge_error_counts(
    target: dict[str, dict[str, int | float | None]],
    update: dict[str, dict[str, int | float | None]],
) -> None:
    for name, values in update.items():
        target[name]["count"] = int(target[name]["count"] or 0) + int(values["count"] or 0)
        target[name]["correct"] = int(target[name]["correct"] or 0) + int(values["correct"] or 0)


def token_error_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
) -> dict[str, dict[str, int | float | None]]:
    """Compute next-token error on non-ignored target tokens.

    ``labels`` are aligned with input positions, while causal-LM logits at
    position t predict the token at t+1, so metrics use the standard shifted
    view. Segment IDs are aligned with labels: 0=think, 1=plan, -100=ignore.
    """
    counts = empty_error_counts()
    if logits.size(1) < 2:
        return finalize_error_counts(counts)

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    predictions = shift_logits.argmax(dim=-1)
    target_mask = shift_labels != -100

    def add(name: str, mask: torch.Tensor) -> None:
        n = int(mask.sum().item())
        if n == 0:
            return
        correct = int((predictions[mask] == shift_labels[mask]).sum().item())
        counts[name]["count"] = int(counts[name]["count"] or 0) + n
        counts[name]["correct"] = int(counts[name]["correct"] or 0) + correct

    add("target", target_mask)
    if segment_ids is not None:
        shift_segments = segment_ids[:, 1:]
        add("think", target_mask & (shift_segments == 0))
        add("plan", target_mask & (shift_segments == 1))
    return finalize_error_counts(counts)
