import torch


def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize rewards within a group to compute advantages (GRPO-style)."""
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    std = rewards.std(unbiased=False)
    if std < eps:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + eps)
