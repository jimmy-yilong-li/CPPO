import torch


def grpo_loss(
    new_logprobs: torch.Tensor,     # current policy log probs
    old_logprobs: torch.Tensor,     # behavior policy log probs (from rollout)
    ref_logprobs: torch.Tensor,     # reference model log probs (KL anchor)
    advantage: float,
    clip_eps: float = 0.2,
    kl_weight: float = 0.01,
) -> torch.Tensor:
    """Compute GRPO loss with PPO-style clipping and non-negative KL penalty.

    ratio = exp(new_logprobs - old_logprobs)  -- new vs OLD (behavior) policy
    KL is vs ref model using non-negative approximation:
        KL(new || ref) approx = (exp(ref/new) - 1 - log(ref/new)).mean()
    """
    if new_logprobs.numel() == 0:
        return torch.tensor(0.0, device=new_logprobs.device, requires_grad=True)

    # PPO/GRPO ratio: new policy vs OLD (behavior) policy
    ratio = (new_logprobs - old_logprobs.detach()).exp()
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps)
    surrogate = torch.min(ratio * advantage, clipped * advantage)

    # KL penalty: non-negative approximation of KL(new || ref)
    log_ratio_ref = ref_logprobs.detach() - new_logprobs
    kl = (log_ratio_ref.exp() - 1.0 - log_ratio_ref).mean()

    return -surrogate.mean() + kl_weight * kl
