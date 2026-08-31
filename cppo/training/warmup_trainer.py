"""Planner warm-up trainer using GRPO with HF generate and raw logprob recomputation.

v0 uses HF model.generate() for rollout (NOT vLLM). This guarantees the
rollout model and training model are the same object. After optimizer.step(),
the next rollout uses updated weights automatically.

CRITICAL: old_logprobs must be raw policy logprobs, NOT temperature-scaled.
After generating with temperature, we recompute logprobs via a forward pass.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cppo.data.prompts import parse_plan_tuple
from cppo.data.problem import Problem
from cppo.training.planner_sft_dataset import (
    make_planner_plan_only_prompt,
    make_planner_think_prompt,
)
from cppo.training.advantages import compute_group_advantages
from cppo.training.grpo import grpo_loss
from cppo.training.rewards import PlanRewardScorer, assign_group_argmax_rewards
from cppo.training.rollout_data import RolloutTokenData

logger = logging.getLogger(__name__)


def _format_plan_methods(methods: list[str]) -> str:
    """Format parsed methods as canonical A/B/C/D plan text for the RM."""
    return "\n".join(
        f"{chr(ord('A') + i)}: {method.strip()}"
        for i, method in enumerate(methods)
    )


def _assign_warmup_rewards(
    *,
    raw_scores: list[float],
    cpsi_scores: list[float],
    jpsi_scores: list[float],
    parseable_flags: list[bool],
    problem_ids: list[str],
    mode: str,
    min_score: float | None,
) -> list[float]:
    """Return planner warmup rewards under the selected reward mode."""
    if mode == "continuous":
        return [
            float(cpsi) if ok else 0.0
            for cpsi, ok in zip(cpsi_scores, parseable_flags)
        ]
    if mode == "binary_jpsi":
        return [
            float(jpsi) if ok else 0.0
            for jpsi, ok in zip(jpsi_scores, parseable_flags)
        ]
    if mode == "group_argmax_binary":
        return assign_group_argmax_rewards(
            scores=raw_scores,
            group_ids=problem_ids,
            parseable=parseable_flags,
            min_score=min_score,
        )
    raise ValueError(f"Unknown warmup reward mode: {mode}")


class PlannerWarmupTrainer:
    """Warm-up trainer for the planner using GRPO with plan reward scoring.

    Generates plan tuples using HF generate, scores them with the frozen RM,
    and performs GRPO updates with proper logprob recomputation.

    Weight synchronization: Since the policy model is the same Python object
    used for both generation and training, optimizer.step() updates are
    automatically reflected in the next rollout.
    """

    def __init__(
        self,
        base_model: str,
        rm_path: str,
        k: int = 4,
        m_tuples: int = 8,
        lr: float = 1e-6,
        kl_weight: float = 0.01,
        clip_eps: float = 0.2,
        temperature: float = 0.9,
        max_new_tokens: int = 512,
        rm_reward_mode: str = "binary_jpsi",
        rm_winner_min_score: float | None = None,
        rm_jpsi_threshold: float = 0.5,
        prompt_mode: str = "plan_only",
        device: str = "cuda",
    ):
        self.k = k
        self.m_tuples = m_tuples
        self.kl_weight = kl_weight
        self.clip_eps = clip_eps
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.rm_reward_mode = rm_reward_mode
        self.rm_winner_min_score = rm_winner_min_score
        self.rm_jpsi_threshold = rm_jpsi_threshold
        self.prompt_mode = prompt_mode
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # left-pad for generate()

        # Policy model (trainable) — used for both generation and training.
        # Forced to float32: bf16/fp16 + AdamW at warmup lrs (1e-6) is a
        # silent no-op (see tests/test_optimizer_dtype.py).
        self.policy = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32,
        )
        self.policy.to(self.device)
        self.policy.train()

        # Reference model (frozen copy for KL penalty). Frozen, no
        # optimizer attached, so bf16 is fine and saves memory.
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
        )
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Reward scorer (frozen RM + calibration thresholds)
        self.scorer = PlanRewardScorer(rm_path, device=device)

        # Refuse half-precision trainable params: AdamW at warmup
        # learning rates cannot move bf16/fp16 weights.
        from cppo.training.cppo_trainer import _assert_policy_dtype_safe_for_optimizer
        _assert_policy_dtype_safe_for_optimizer(
            p for p in self.policy.parameters() if p.requires_grad
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

    def _rollout_with_logprobs(
        self, problems: list[Problem]
    ) -> list[RolloutTokenData]:
        """Generate plan tuples and recompute raw logprobs.

        For each problem, generates m_tuples plan samples using temperature
        sampling. Then recomputes raw (un-temperature-scaled) logprobs via
        a forward pass through the policy model.

        Args:
            problems: List of Problem objects to generate plans for.

        Returns:
            List of RolloutTokenData, one per generated plan.
        """
        all_rollouts: list[RolloutTokenData] = []

        self.policy.eval()  # eval mode for generation (no dropout)
        try:
            from cppo.data.prompts import to_chat_text
            for problem in problems:
                prompt_text = self._make_planner_prompt(problem)
                prompt_text = to_chat_text(
                    self.tokenizer,
                    prompt_text,
                    enable_thinking=(self.prompt_mode == "think_plan"),
                )
                prompt_enc = self.tokenizer(
                    prompt_text, return_tensors="pt"
                )
                prompt_ids = prompt_enc["input_ids"][0].tolist()
                prompt_len = len(prompt_ids)

                for _ in range(self.m_tuples):
                    # Step 1: generate tokens with temperature
                    input_t = torch.tensor(
                        [prompt_ids], device=self.device
                    )
                    with torch.no_grad():
                        out = self.policy.generate(
                            input_t,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    gen_ids = out[0, prompt_len:].tolist()

                    if len(gen_ids) == 0:
                        continue

                    # Step 2: recompute raw logprobs via forward pass
                    full_ids = prompt_ids + gen_ids
                    with torch.no_grad():
                        logits = self.policy(
                            input_ids=torch.tensor(
                                [full_ids], device=self.device
                            )
                        ).logits
                    # logprobs for each generated token
                    # logits at position t predicts token at position t+1
                    # So for gen tokens starting at prompt_len, we need
                    # logits from prompt_len-1 to prompt_len+len(gen_ids)-2
                    lps = F.log_softmax(
                        logits[0, prompt_len - 1 : -1], dim=-1
                    )
                    gen_t = torch.tensor(gen_ids, device=self.device)
                    old_lps = (
                        lps.gather(1, gen_t.unsqueeze(1))
                        .squeeze(1)
                        .tolist()
                    )

                    response_text = self.tokenizer.decode(
                        gen_ids, skip_special_tokens=True
                    )

                    rd = RolloutTokenData(
                        prompt_token_ids=prompt_ids,
                        response_token_ids=gen_ids,
                        old_logprobs=old_lps,
                        advantage=0.0,
                        region="planner",
                        problem_id=problem.id,
                        prompt_text=prompt_text,
                        response_text=response_text,
                    )
                    all_rollouts.append(rd)
        finally:
            self.policy.train()  # restore training mode

        return all_rollouts

    def _make_planner_prompt(self, problem: Problem) -> str:
        """Build the planner prompt in the same mode as SFT/deployment."""
        if self.prompt_mode == "think_plan":
            return make_planner_think_prompt(problem.prompt, domain=problem.domain, k=self.k)
        if self.prompt_mode == "plan_only":
            return make_planner_plan_only_prompt(problem.prompt, domain=problem.domain, k=self.k)
        raise ValueError(f"unknown warmup prompt_mode: {self.prompt_mode}")

    def _compute_new_and_ref_logprobs(
        self, rd: RolloutTokenData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute current policy and reference model logprobs for a rollout.

        Args:
            rd: A RolloutTokenData from a previous rollout.

        Returns:
            (new_token_lps, old_tensor, ref_token_lps) — all 1-D tensors
            of shape (num_gen_tokens,).
        """
        full_ids = rd.prompt_token_ids + rd.response_token_ids
        input_t = torch.tensor([full_ids], device=self.device)
        prompt_len = len(rd.prompt_token_ids)
        gen_t = torch.tensor(
            rd.response_token_ids, device=self.device
        )

        # New policy logprobs (with gradient)
        new_logits = self.policy(input_ids=input_t).logits
        new_lps = F.log_softmax(
            new_logits[0, prompt_len - 1 : -1], dim=-1
        )
        new_token_lps = new_lps.gather(1, gen_t.unsqueeze(1)).squeeze(1)

        # Old logprobs (from rollout, detached)
        old_tensor = torch.tensor(
            rd.old_logprobs, device=self.device, dtype=torch.float32
        )

        # Reference model logprobs (frozen, no gradient)
        with torch.no_grad():
            ref_logits = self.ref_model(input_ids=input_t).logits
        ref_lps = F.log_softmax(
            ref_logits[0, prompt_len - 1 : -1], dim=-1
        )
        ref_token_lps = ref_lps.gather(1, gen_t.unsqueeze(1)).squeeze(1)

        return new_token_lps, old_tensor, ref_token_lps

    def train_step(
        self, problems: list[Problem]
    ) -> dict[str, float]:
        """One warm-up training step: rollout -> score -> advantages -> GRPO.

        Args:
            problems: Batch of Problem objects.

        Returns:
            Dict with keys: loss, avg_cpsi, parseable_rate, rm_valid_rate.
        """
        # --- 1. Rollout ---
        rollouts = self._rollout_with_logprobs(problems)

        if len(rollouts) == 0:
            logger.warning("No rollouts generated; skipping train step.")
            return {
                "loss": 0.0,
                "avg_cpsi": 0.0,
                "avg_raw_rm_score": 0.0,
                "parseable_rate": 0.0,
                "rm_valid_rate": 0.0,
                "winner_rate": 0.0,
                "groups_with_winner_rate": 0.0,
                "winner_score_mean": 0.0,
            }

        # --- 2. Score plans with RM ---
        # Check parseability and score
        problem_texts: list[str] = []
        plan_texts: list[str] = []
        parseable_flags: list[bool] = []

        for rd in rollouts:
            parsed = parse_plan_tuple(rd.response_text, self.k)
            is_parseable = parsed is not None
            parseable_flags.append(is_parseable)
            # Find the problem prompt text for this rollout
            # We stored the plan prompt as prompt_text, but the RM expects
            # the raw problem text. Extract from the Problem objects.
            prob = next(
                (p for p in problems if p.id == rd.problem_id), None
            )
            problem_texts.append(prob.prompt if prob else "")
            # The planner may be trained to emit <think>...</think> + A/B/C/D,
            # but the binary tuple RM is trained on plan_text only. Score the
            # parsed tuple in canonical A/B/C/D form and use parseable_flags to
            # gate unparseable raw outputs to zero reward.
            plan_texts.append(_format_plan_methods(parsed) if parsed else rd.response_text)

        # Score with RM. Warm-up uses binary Jψ; Cψ stays diagnostic.
        if hasattr(self.scorer, "score_plan_details"):
            score_details = self.scorer.score_plan_details(problem_texts, plan_texts)
            raw_scores = [float(d["pass_prob"]) for d in score_details]
            cpsi_scores = [float(d["c_psi"]) for d in score_details]
            jpsi_scores = [
                1.0 if score >= self.rm_jpsi_threshold else 0.0
                for score in raw_scores
            ]
        else:
            if self.rm_reward_mode == "binary_jpsi":
                raise RuntimeError(
                    "binary_jpsi warmup requires scorer.score_plan_details()"
                )
            cpsi_scores = [float(s) for s in self.scorer.score_plans(problem_texts, plan_texts)]
            raw_scores = list(cpsi_scores)
            jpsi_scores = [1.0 if s >= 0.5 else 0.0 for s in raw_scores]

        # --- 3. Compute rewards and advantages ---
        problem_ids = [rd.problem_id for rd in rollouts]
        rewards_list = _assign_warmup_rewards(
            raw_scores=raw_scores,
            cpsi_scores=cpsi_scores,
            jpsi_scores=jpsi_scores,
            parseable_flags=parseable_flags,
            problem_ids=problem_ids,
            mode=self.rm_reward_mode,
            min_score=self.rm_winner_min_score,
        )

        # Group advantages by problem_id
        # Group rollouts by problem
        from collections import defaultdict

        groups: dict[str, list[int]] = defaultdict(list)
        for i, rd in enumerate(rollouts):
            groups[rd.problem_id].append(i)

        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
        advantages = torch.zeros_like(rewards_tensor)

        for pid, indices in groups.items():
            group_rewards = rewards_tensor[indices]
            group_advs = compute_group_advantages(group_rewards)
            for j, idx in enumerate(indices):
                advantages[idx] = group_advs[j]

        # Assign advantages back
        for i, rd in enumerate(rollouts):
            rd.advantage = advantages[i].item()

        # --- 4. GRPO update ---
        self.policy.train()
        self.optimizer.zero_grad()

        valid_rollouts = [rd for rd in rollouts if len(rd.response_token_ids) > 0]
        n_valid = len(valid_rollouts)
        total_loss_value = 0.0

        if valid_rollouts:
            # Backward each rollout's scaled loss immediately instead of
            # accumulating all computation graphs. LCB prompts can be long,
            # and retaining dozens of full forward graphs OOMs even on H200.
            for rd in valid_rollouts:
                new_lps, old_lps, ref_lps = self._compute_new_and_ref_logprobs(
                    rd
                )
                loss_i = grpo_loss(
                    new_logprobs=new_lps,
                    old_logprobs=old_lps,
                    ref_logprobs=ref_lps,
                    advantage=rd.advantage,
                    clip_eps=self.clip_eps,
                    kl_weight=self.kl_weight,
                )
                total_loss_value += float(loss_i.detach().item())
                (loss_i / n_valid).backward()
            self.optimizer.step()
            avg_loss = total_loss_value / n_valid
        else:
            avg_loss = 0.0

        # --- 5. Metrics ---
        avg_cpsi = sum(cpsi_scores) / len(cpsi_scores) if cpsi_scores else 0.0
        avg_raw_rm_score = (
            sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
        )
        parseable_rate = (
            sum(parseable_flags) / len(parseable_flags)
            if parseable_flags
            else 0.0
        )
        # rm_valid_rate follows the binary decision Jψ. Gating on
        # parseable prevents early-stop from firing while a high percentage of
        # plans are still unparseable junk that scored 0.
        n_total = len(cpsi_scores)
        rm_valid_rate = (
            sum(
                1 for s, ok in zip(jpsi_scores, parseable_flags)
                if ok and s > 0.0
            ) / n_total
            if n_total else 0.0
        )
        winners = [i for i, r in enumerate(rewards_list) if r > 0.0]
        winner_rate = len(winners) / len(rewards_list) if rewards_list else 0.0
        groups_with_winner = {
            problem_ids[i] for i in winners
        }
        total_groups = len(set(problem_ids))
        groups_with_winner_rate = (
            len(groups_with_winner) / total_groups if total_groups else 0.0
        )
        winner_score_mean = (
            sum(raw_scores[i] for i in winners) / len(winners)
            if winners else 0.0
        )

        return {
            "loss": avg_loss,
            "avg_cpsi": avg_cpsi,
            "avg_raw_rm_score": avg_raw_rm_score,
            "parseable_rate": parseable_rate,
            "rm_valid_rate": rm_valid_rate,
            "winner_rate": winner_rate,
            "groups_with_winner_rate": groups_with_winner_rate,
            "winner_score_mean": winner_score_mean,
        }

    def save(self, path: str) -> None:
        """Save the policy model and tokenizer."""
        from pathlib import Path as P

        P(path).mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Saved warm-up policy to {path}")
