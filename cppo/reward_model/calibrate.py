"""Reward model calibration: threshold finding and calibrated reward computation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def find_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    precision_target: float = 0.95,
    cpsi_gate: float = 0.8,
) -> dict[str, float]:
    """Calibrate tau_low / tau_high so the Cψ gate matches precision target.

    Procedure (the "B'" calibration):
      1. tau_low: highest raw threshold where positive-recall is ~50% — the
         "reject most negatives" anchor that compute_calibrated_reward maps
         to Cψ=0.
      2. t_accept: lowest raw threshold where precision >= precision_target.
         This is the actual decision boundary we want at Cψ=cpsi_gate.
      3. tau_high: chosen so that compute_calibrated_reward(t_accept) == cpsi_gate,
         i.e. tau_high = tau_low + (t_accept - tau_low) / cpsi_gate.
         Equivalent invariant: Cψ >= cpsi_gate ⇔ raw p >= t_accept.

    Why: previously tau_high WAS t_accept, but the gate was measured at
    Cψ>=0.8 which corresponds to raw p >= tau_low + 0.8*(tau_high - tau_low),
    a lower (looser) threshold. Precision at the gate could then sit below
    the target by definition. With B', the Cψ-gate and the precision-target
    threshold are the same set by construction.

    Args:
        probs: Model predicted probabilities, shape (N,).
        labels: Binary ground-truth labels, shape (N,).
        precision_target: Desired precision among accepted samples.
        cpsi_gate: The Cψ value at which downstream code measures the gate.

    Returns:
        {"tau_high": float, "tau_low": float, "t_accept": float}
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    sorted_probs = np.sort(np.unique(probs))

    # --- tau_low: threshold at ~50% recall of positives ---
    n_pos = labels.sum()
    tau_low = float(sorted_probs[0])
    if n_pos > 0:
        best_diff = float("inf")
        for t in sorted_probs:
            predicted_pos = probs >= t
            recall = labels[predicted_pos].sum() / n_pos
            diff = abs(recall - 0.5)
            if diff < best_diff:
                best_diff = diff
                tau_low = float(t)
    else:
        tau_low = float(np.median(probs))

    # --- t_accept: lowest raw threshold achieving precision >= target ---
    t_accept = float(sorted_probs[-1])  # fallback: highest prob
    for t in sorted_probs:
        predicted_pos = probs >= t
        if predicted_pos.sum() == 0:
            continue
        precision = labels[predicted_pos].sum() / predicted_pos.sum()
        if precision >= precision_target:
            t_accept = float(t)
            break

    # --- tau_high: derived so Cψ(t_accept) == cpsi_gate ---
    if cpsi_gate <= 0 or cpsi_gate >= 1:
        raise ValueError(f"cpsi_gate must be in (0, 1), got {cpsi_gate}")
    if t_accept <= tau_low:
        # Degenerate: even the loosest accept threshold sits below tau_low.
        # Push tau_low down so Cψ stays well-defined.
        tau_low = max(0.0, t_accept - 1e-6)
    tau_high = tau_low + (t_accept - tau_low) / cpsi_gate

    return {
        "tau_high": float(tau_high),
        "tau_low": float(tau_low),
        "t_accept": float(t_accept),
    }


def compute_calibrated_reward(
    pass_prob: float,
    tau_low: float,
    tau_high: float,
) -> float:
    """Compute calibrated reward clipped to [0, 1].

    reward = clip((p - tau_low) / (tau_high - tau_low), 0, 1)
    """
    if tau_high <= tau_low:
        # Degenerate case: return 1 if above threshold, else 0
        return 1.0 if pass_prob >= tau_high else 0.0

    raw = (pass_prob - tau_low) / (tau_high - tau_low)
    return float(max(0.0, min(1.0, raw)))


def precision_at_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute precision and recall at a given threshold.

    Args:
        probs:  Model predicted probabilities.
        labels: Binary ground-truth labels.
        threshold: Decision threshold.

    Returns:
        {"threshold": float, "n": int, "precision": float, "recall": float}
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    predicted_pos = probs >= threshold
    n_predicted = int(predicted_pos.sum())
    n_pos = int(labels.sum())

    if n_predicted == 0:
        precision = 1.0  # no false positives when nothing predicted
    else:
        precision = float(labels[predicted_pos].sum() / n_predicted)

    if n_pos == 0:
        recall = 0.0
    else:
        recall = float(labels[predicted_pos].sum() / n_pos)

    return {
        "threshold": threshold,
        "n": n_predicted,
        "precision": precision,
        "recall": recall,
    }


def per_violation_fp_rate(
    probs: np.ndarray,
    labels: np.ndarray,
    violations: list[str],
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Compute false positive rate per violation category.

    A false positive is a sample predicted as positive (prob >= threshold)
    but actually negative (label == 0).

    Args:
        probs:      Model predicted probabilities.
        labels:     Binary ground-truth labels.
        violations: Violation category for each sample.
        threshold:  Decision threshold.

    Returns:
        Dict mapping violation category to {n, fp_count, fp_rate}.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    groups: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(violations):
        groups[v].append(i)

    result = {}
    for violation, indices in sorted(groups.items()):
        idx = np.array(indices)
        group_probs = probs[idx]
        group_labels = labels[idx]

        predicted_pos = group_probs >= threshold
        actual_neg = group_labels == 0
        fp = (predicted_pos & actual_neg).sum()
        n_neg = actual_neg.sum()

        fp_rate = float(fp / n_neg) if n_neg > 0 else 0.0
        result[violation] = {
            "n": len(indices),
            "fp_count": int(fp),
            "fp_rate": fp_rate,
        }

    return result
