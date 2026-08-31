"""Reward model calibration: threshold finding and calibrated reward computation."""

from __future__ import annotations




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


