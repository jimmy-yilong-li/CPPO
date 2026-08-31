"""Full CPPO trainer: two-phase rollout, product reward, split advantages, GRPO update."""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn.functional as F

from cppo.data.problem import Problem
from cppo.eval.metrics import mean_pass_at_k_sweep
from cppo.training.advantages import compute_group_advantages
from cppo.training.grpo import grpo_loss
from cppo.training.rewards import PlanRewardScorer
from cppo.training.rollout import run_cppo_rollout_hf
from cppo.training.rollout_data import CPPORolloutBundle, RolloutTokenData

logger = logging.getLogger(__name__)


def _assert_policy_dtype_safe_for_optimizer(params) -> None:
    """Refuse bf16/fp16 trainable parameters.

    Half-precision dtypes have ~3 decimal digits of mantissa near
    unit-magnitude weights. With AdamW + small lr (e.g. 5e-7), the
    update is below the dtype's quantum and `param.add_(...)` is a
    silent no-op. The 2026-05-10 smoke ran for 35 minutes producing
    `loss=0.0000` every epoch for exactly this reason.

    Trainable parameters must be float32. Frozen reference models
    (no optimizer attached) can stay in bf16/fp16 — they only forward.
    """
    bad = {torch.bfloat16, torch.float16}
    for p in params:
        if p.dtype in bad:
            raise RuntimeError(
                f"Trainable parameter has dtype {p.dtype}; AdamW updates at "
                f"typical CPPO learning rates fall below this dtype's mantissa "
                f"precision and the step is a silent no-op. Load policy in "
                f"float32 (e.g. `from_pretrained(..., torch_dtype=torch.float32)`)."
            )


def _configure_policy_for_training(model, *, gradient_checkpointing: bool):
    """Apply train-time memory settings to the policy model."""
    if gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError("ref_dtype must be one of: fp32, bf16, fp16")


class CPPOTrainer:
    """CPPO trainer with two-phase rollout, product reward, and split advantages.

    Uses HF generate() for rollouts (guarantees weight sync), recomputes
    old_logprobs via forward pass, and applies GRPO loss with PPO-style
    clipping and KL penalty against a frozen reference model.
    """

    def __init__(
        self,
        base_model,
        tokenizer,
        rm_path: str,
        k: int = 4,
        m_tuples: int = 4,
        lr: float = 5e-7,
        kl_weight: float = 0.01,
        clip_eps: float = 0.2,
        plan_temp: float = 0.9,
        solve_temp: float = 0.7,
        max_plan_tokens: int = 512,
        max_solve_tokens: int = 2048,
        exec_timeout: int = 10,
        use_pass_rate: bool = True,
        rm_reward_mode: str = "binary_jpsi",
        plan_reward_mode: str = "jpsi_times_outcome",
        rm_winner_min_score: float | None = None,
        rm_jpsi_threshold: float | None = None,
        planner_prompt_mode: str = "legacy_plan",
        gradient_checkpointing: bool = True,
        ref_dtype: str = "bf16",
        max_train_tokens: int | None = None,
        param_delta_max_tensors: int = 4,
        device: str = "cuda",
        outcome_log_path: str | None = None,
    ):
        self.device = device
        self.k = k
        self.m_tuples = m_tuples
        self.kl_weight = kl_weight
        self.clip_eps = clip_eps
        self.plan_temp = plan_temp
        self.solve_temp = solve_temp
        self.max_plan_tokens = max_plan_tokens
        self.max_solve_tokens = max_solve_tokens
        self.exec_timeout = exec_timeout
        self.use_pass_rate = use_pass_rate
        self.rm_reward_mode = rm_reward_mode
        self.plan_reward_mode = plan_reward_mode
        self.rm_winner_min_score = rm_winner_min_score
        self.planner_prompt_mode = planner_prompt_mode
        self.gradient_checkpointing = gradient_checkpointing
        self.max_train_tokens = max_train_tokens
        self.param_delta_max_tensors = param_delta_max_tensors
        # Optional path for per-bundle outcome JSONL (hard-negative mining trail).
        self.outcome_log_path = outcome_log_path

        # Policy model (trainable)
        self.policy_model = base_model.to(device)
        _configure_policy_for_training(
            self.policy_model, gradient_checkpointing=gradient_checkpointing
        )
        self.policy_model.train()

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Reference model (frozen deep copy)
        self.ref_model = copy.deepcopy(base_model).to(
            device=device,
            dtype=_dtype_from_name(ref_dtype),
        )
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Reward model scorer
        self.scorer = PlanRewardScorer(
            rm_path,
            device=device,
            jpsi_threshold=rm_jpsi_threshold,
        )

        # Refuse half-precision trainable params — AdamW updates at
        # typical CPPO LRs fall below bf16/fp16 mantissa precision and
        # the step is a silent no-op (root cause of the 2026-05-10
        # smoke's all-zero losses).
        _assert_policy_dtype_safe_for_optimizer(
            p for p in self.policy_model.parameters() if p.requires_grad
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Logprob helpers
    # ------------------------------------------------------------------

    def _compute_logprobs_for_tokens(
        self,
        model,
        prompt_ids: list[int],
        response_ids: list[int],
    ) -> torch.Tensor:
        """Compute per-token log-probs for response tokens given prompt.

        Args:
            model: The model to use (policy or ref).
            prompt_ids: Token IDs for the prompt.
            response_ids: Token IDs for the response.

        Returns:
            Tensor of shape (len(response_ids),) with log-probs.
        """
        if len(response_ids) == 0:
            return torch.tensor([], device=self.device)

        full_ids = torch.tensor(
            prompt_ids + response_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        logits = model(full_ids).logits  # (1, L, V)
        log_probs_all = F.log_softmax(logits[0], dim=-1)  # (L, V)

        prompt_len = len(prompt_ids)
        token_logprobs = []
        for i, tok_id in enumerate(response_ids):
            pos = prompt_len + i - 1  # logit at pos predicts token at pos+1
            token_logprobs.append(log_probs_all[pos, tok_id])

        return torch.stack(token_logprobs)

    def _compute_new_and_ref_logprobs(
        self,
        rd: RolloutTokenData,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute current policy and reference model logprobs for a rollout.

        Args:
            rd: A RolloutTokenData instance with prompt/response token IDs.

        Returns:
            (new_logprobs, ref_logprobs) each of shape (num_response_tokens,).
        """
        new_lp = self._compute_logprobs_for_tokens(
            self.policy_model, rd.prompt_token_ids, rd.response_token_ids
        )
        with torch.no_grad():
            ref_lp = self._compute_logprobs_for_tokens(
                self.ref_model, rd.prompt_token_ids, rd.response_token_ids
            )
        return new_lp, ref_lp

    def _loss_skip_reason(self, rd: RolloutTokenData) -> str | None:
        """Return a reason to skip this rollout term, or None if trainable."""
        if len(rd.response_token_ids) == 0:
            return "empty_response"
        if rd.skip_loss_reason:
            return rd.skip_loss_reason
        max_train_tokens = getattr(self, "max_train_tokens", None)
        if max_train_tokens is not None:
            total_tokens = len(rd.prompt_token_ids) + len(rd.response_token_ids)
            if total_tokens > max_train_tokens:
                return f"overlong_total_tokens_{total_tokens}_gt_{max_train_tokens}"
        return None

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_planner_advantages(
        bundles: list[CPPORolloutBundle],
    ) -> dict[str, torch.Tensor]:
        """Compute planner advantages across tuples for each problem.

        Groups bundles by problem_id, then normalizes plan_reward across
        tuples for the same problem (GRPO-style).

        Returns:
            Dict mapping (problem_id, tuple_index_in_group) -> advantage float,
            keyed as "{problem_id}_{group_idx}".
        """
        from collections import defaultdict

        groups: dict[str, list[int]] = defaultdict(list)
        for i, b in enumerate(bundles):
            groups[b.problem_id].append(i)

        advantages: dict[str, float] = {}
        for pid, indices in groups.items():
            rewards = torch.tensor(
                [bundles[i].plan_reward for i in indices], dtype=torch.float32
            )
            advs = compute_group_advantages(rewards)
            for local_idx, global_idx in enumerate(indices):
                key = f"{pid}_{global_idx}"
                advantages[key] = advs[local_idx].item()

        return advantages

    @staticmethod
    def _compute_solver_advantages(
        bundle: CPPORolloutBundle,
    ) -> list[float]:
        """Compute solver advantages within a single tuple.

        Normalizes branch_rewards across the k branches within this tuple.

        Returns:
            List of advantage values, one per branch.
        """
        if not bundle.branch_rewards:
            return []
        rewards = torch.tensor(bundle.branch_rewards, dtype=torch.float32)
        advs = compute_group_advantages(rewards)
        return advs.tolist()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, problems: list[Problem],
                   step: int = 0, epoch: int = 0) -> dict[str, Any]:
        """Run one CPPO training step.

        1. Run two-phase rollout (plan -> solve -> verify)
        2. Compute planner advantages (across tuples per problem)
        3. Compute solver advantages (within tuple)
        4. GRPO update for all planner + solver token data
        5. Return comprehensive metrics

        Args:
            problems: Batch of Problem instances.
            step:     Current global step (for outcome log).
            epoch:    Current epoch (for outcome log).

        Returns:
            Dict of metrics.
        """
        # ----- 1. Rollout -----
        self.policy_model.eval()  # eval mode for generation
        bundles = run_cppo_rollout_hf(
            policy_model=self.policy_model,
            tokenizer=self.tokenizer,
            problems=problems,
            scorer=self.scorer,
            k=self.k,
            m_tuples=self.m_tuples,
            plan_temp=self.plan_temp,
            solve_temp=self.solve_temp,
            max_plan_tokens=self.max_plan_tokens,
            max_solve_tokens=self.max_solve_tokens,
            exec_timeout=self.exec_timeout,
            use_pass_rate=self.use_pass_rate,
            device=self.device,
            outcome_log_path=self.outcome_log_path,
            step=step,
            epoch=epoch,
            rm_reward_mode=self.rm_reward_mode,
            plan_reward_mode=self.plan_reward_mode,
            rm_winner_min_score=self.rm_winner_min_score,
            planner_prompt_mode=getattr(self, "planner_prompt_mode", "legacy_plan"),
        )
        self.policy_model.train()  # back to train mode for update

        # ----- 2. Compute planner advantages -----
        planner_advs = self._compute_planner_advantages(bundles)

        # ----- 3. Compute solver advantages per bundle -----
        solver_advs_per_bundle: list[list[float]] = []
        for bundle in bundles:
            solver_advs_per_bundle.append(
                self._compute_solver_advantages(bundle)
            )

        # ----- 4. GRPO update -----
        self.optimizer.zero_grad()

        # Count trainable loss terms before building any autograd graph. We
        # then backprop each term immediately, scaled by 1/n, instead of
        # accumulating all LM-logit graphs into one giant total_loss tensor.
        # The previous accumulate-then-backward path OOM'd a 140GB H200 on the
        # first smoke step because every planner/solver branch graph stayed
        # live until the final backward().
        n_loss_terms = 0
        skipped_loss_terms = 0
        skipped_loss_reasons: dict[str, int] = {}
        for bundle in bundles:
            if bundle.plan_data is not None:
                reason = self._loss_skip_reason(bundle.plan_data)
                if reason is None:
                    n_loss_terms += 1
                elif reason != "empty_response":
                    skipped_loss_terms += 1
                    skipped_loss_reasons[reason] = skipped_loss_reasons.get(reason, 0) + 1
            if bundle.is_parseable and bundle.solver_data:
                for sd in bundle.solver_data:
                    reason = self._loss_skip_reason(sd)
                    if reason is None:
                        n_loss_terms += 1
                    elif reason != "empty_response":
                        skipped_loss_terms += 1
                        skipped_loss_reasons[reason] = skipped_loss_reasons.get(reason, 0) + 1

        def backward_one(rd: RolloutTokenData, adv_val: float) -> float:
            new_lp, ref_lp = self._compute_new_and_ref_logprobs(rd)
            old_lp = torch.tensor(
                rd.old_logprobs,
                dtype=torch.float32,
                device=self.device,
            )
            loss = grpo_loss(
                new_logprobs=new_lp,
                old_logprobs=old_lp,
                ref_logprobs=ref_lp,
                advantage=adv_val,
                clip_eps=self.clip_eps,
                kl_weight=self.kl_weight,
            )
            loss_value = float(loss.detach().float().item())
            (loss / n_loss_terms).backward()
            return loss_value

        total_loss_value = 0.0

        # Planner losses
        if n_loss_terms > 0:
            for i, bundle in enumerate(bundles):
                if bundle.plan_data is None:
                    continue
                if self._loss_skip_reason(bundle.plan_data) is not None:
                    continue

                adv_key = f"{bundle.problem_id}_{i}"
                adv_val = planner_advs.get(adv_key, 0.0)

                total_loss_value += backward_one(bundle.plan_data, adv_val)

            # Solver losses (joint planner-solver training per the paper).
            for i, bundle in enumerate(bundles):
                if not bundle.is_parseable or not bundle.solver_data:
                    continue
                s_advs = solver_advs_per_bundle[i]
                for j, sd in enumerate(bundle.solver_data):
                    if self._loss_skip_reason(sd) is not None:
                        continue
                    adv_val = s_advs[j] if j < len(s_advs) else 0.0
                    total_loss_value += backward_one(sd, adv_val)

        # Average and backprop. We capture grad_norm AFTER backward and
        # param_delta as the FULL-model L2 distance between pre- and
        # post-step parameters, so even when scalar loss is ~0 we can
        # tell whether the update actually moved the policy.
        grad_norm_val = 0.0
        param_delta_val = 0.0
        if n_loss_terms > 0:
            avg_loss_value = total_loss_value / n_loss_terms
            grad_sq = 0.0
            for p in self.policy_model.parameters():
                if p.grad is not None:
                    grad_sq += p.grad.detach().pow(2).sum().item()
            grad_norm_val = grad_sq ** 0.5

            # Snapshot only a small prefix of trainable tensors. Cloning the
            # full 4B fp32 policy just to report param_delta adds a large
            # memory spike and can collide with long-sequence activations.
            snapshot_limit = max(0, int(getattr(self, "param_delta_max_tensors", 4)))
            tracked_params = [
                p for p in self.policy_model.parameters() if p.requires_grad
            ][:snapshot_limit]
            pre_snapshot = [p.detach().clone() for p in tracked_params]
            self.optimizer.step()
            delta_sq = 0.0
            for pre, p in zip(pre_snapshot, tracked_params):
                delta_sq += (p.detach() - pre).pow(2).sum().item()
            param_delta_val = delta_sq ** 0.5
            del pre_snapshot
        else:
            avg_loss_value = 0.0

        # ----- 5. Compute metrics -----
        metrics = self._compute_metrics(
            bundles, avg_loss_value,
            grad_norm=grad_norm_val, param_delta=param_delta_val,
        )
        metrics["loss_terms"] = n_loss_terms
        metrics["skipped_loss_terms"] = skipped_loss_terms
        metrics["skipped_loss_reasons"] = skipped_loss_reasons
        return metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        bundles: list[CPPORolloutBundle],
        loss_value: float,
        grad_norm: float = 0.0,
        param_delta: float = 0.0,
    ) -> dict[str, Any]:
        """Compute training metrics from rollout bundles + grad/param state.

        Metrics:
            - loss: average GRPO loss
            - grad_norm: L2 norm of policy-parameter gradients after backward()
            - param_delta: L2 distance between pre- and post-step parameters
            - parseable_rate: fraction of plans that parsed successfully
            - rm_valid_rate: alias of jpsi_pass_rate for paper-base runs
            - avg_cpsi: mean diagnostic c_psi across all parseable bundles
            - avg_pass_prob: mean raw P(Pass) across parseable bundles
            - jpsi_pass_rate: fraction of parseable plans with J_psi = 1
            - outcome_success_rate: fraction of tuples with outcome_reward > 0
            - plan_reward_nonzero_rate: fraction of tuples with plan_reward > 0
            - any_pass_rate: fraction of solver branches with pass_rate > 0
              (renamed from compile_rate; the old name implied "did the code
              compile", but the verifier mixes compile/runtime/wrong/timeout
              into a single pass_rate scalar)
            - zero_pass_rate: fraction of solver branches with pass_rate == 0
              (renamed from timeout_rate; the old name implied wall-clock
              timeout but actually meant "no test passed for any reason")
            - pass_at_k_sweep: unbiased pass@k ladder, pooling every solver
              branch of a problem across its M tuples (so M=8, k=4 supports
              rungs up to 32). Branches within one tuple are
              strategy-conditioned rather than iid; the estimate treats a
              problem's pooled branches as exchangeable, matching the
              protocol used to report budgets above the tuple size. A rung is
              reported only when every problem has that many branches.
            - avg_outcome_reward: mean outcome_reward
            - avg_plan_reward: mean plan_reward
            - planner_adv_nonzero_rate: fraction of bundles whose planner
              GRPO advantage is non-zero (group-normalized; zero when every
              plan in the same problem group had the same plan_reward)
            - solver_adv_nonzero_rate: fraction of solver branches whose
              within-tuple solver advantage is non-zero
            - group_reward_variance: mean variance of plan_reward across
              tuples within each problem group
            - n_bundles: total number of bundles
        """
        n = len(bundles)
        if n == 0:
            return {
                "loss": loss_value,
                "grad_norm": grad_norm,
                "param_delta": param_delta,
                "parseable_rate": 0.0,
                "rm_valid_rate": 0.0,
                "avg_cpsi": 0.0,
                "avg_pass_prob": 0.0,
                "jpsi_pass_rate": 0.0,
                "outcome_success_rate": 0.0,
                "plan_reward_nonzero_rate": 0.0,
                "rm_winner_rate": 0.0,
                "rm_reward_nonzero_rate": 0.0,
                "winner_outcome_success_rate": 0.0,
                "rm_winner_outcome_success_rate": 0.0,
                "plan_winner_rate": 0.0,
                "plan_winner_outcome_success_rate": 0.0,
                "groups_with_plan_winner_rate": 0.0,
                "any_pass_rate": 0.0,
                "zero_pass_rate": 0.0,
                "avg_outcome_reward": 0.0,
                "avg_plan_reward": 0.0,
                "pass_at_k_sweep": {},
                "planner_adv_nonzero_rate": 0.0,
                "solver_adv_nonzero_rate": 0.0,
                "group_reward_variance": 0.0,
                "n_bundles": 0,
            }

        n_parseable = sum(1 for b in bundles if b.is_parseable)
        parseable_rate = n_parseable / n

        # c_psi stats (only for parseable bundles)
        parseable_bundles = [b for b in bundles if b.is_parseable]
        if parseable_bundles:
            cpsi_values = [b.c_psi for b in parseable_bundles]
            avg_cpsi = sum(cpsi_values) / len(cpsi_values)
            pass_prob_values = [b.raw_rm_score for b in parseable_bundles]
            avg_pass_prob = sum(pass_prob_values) / len(pass_prob_values)
            jpsi_pass = sum(1 for b in parseable_bundles if b.j_psi > 0.0)
            jpsi_pass_rate = jpsi_pass / len(parseable_bundles)
            rm_valid_rate = jpsi_pass_rate
        else:
            avg_cpsi = 0.0
            avg_pass_prob = 0.0
            jpsi_pass_rate = 0.0
            rm_valid_rate = 0.0

        # Outcome success rate
        n_outcome_success = sum(
            1 for b in bundles if b.outcome_reward > 0.0
        )
        outcome_success_rate = n_outcome_success / n

        # Plan reward nonzero rate
        n_plan_nonzero = sum(1 for b in bundles if b.plan_reward > 0.0)
        plan_reward_nonzero_rate = n_plan_nonzero / n
        n_rm_winner = sum(1 for b in bundles if b.rm_winner)
        rm_winner_rate = n_rm_winner / n
        n_rm_reward_nonzero = sum(1 for b in bundles if b.rm_reward > 0.0)
        rm_reward_nonzero_rate = n_rm_reward_nonzero / n
        winner_bundles = [b for b in bundles if b.rm_winner]
        rm_winner_outcome_success_rate = (
            sum(1 for b in winner_bundles if b.outcome_reward > 0.0)
            / len(winner_bundles)
            if winner_bundles else 0.0
        )
        winner_outcome_success_rate = rm_winner_outcome_success_rate
        n_plan_winner = sum(1 for b in bundles if b.plan_winner)
        plan_winner_rate = n_plan_winner / n
        plan_winner_bundles = [b for b in bundles if b.plan_winner]
        plan_winner_outcome_success_rate = (
            sum(1 for b in plan_winner_bundles if b.outcome_reward > 0.0)
            / len(plan_winner_bundles)
            if plan_winner_bundles else 0.0
        )
        groups_with_plan_winner = {
            b.problem_id for b in bundles if b.plan_winner
        }
        total_problem_groups = len({b.problem_id for b in bundles})
        groups_with_plan_winner_rate = (
            len(groups_with_plan_winner) / total_problem_groups
            if total_problem_groups else 0.0
        )

        # Pass-rate aggregates over all solver branches.
        # Renamed from compile_rate / timeout_rate, which were misleading:
        # the verifier only exposes pass_rate as a scalar and does NOT
        # distinguish compile error / runtime error / wrong answer / true
        # timeout. So we report what the data actually says.
        total_branches = 0
        zero_pass = 0
        for b in bundles:
            if not b.is_parseable:
                continue
            for pr in b.branch_pass_rates:
                total_branches += 1
                if pr == 0.0:
                    zero_pass += 1

        any_pass_rate = (
            (total_branches - zero_pass) / total_branches
            if total_branches > 0 else 0.0
        )
        zero_pass_rate = (
            zero_pass / total_branches if total_branches > 0 else 0.0
        )

        avg_outcome = sum(b.outcome_reward for b in bundles) / n
        avg_plan = sum(b.plan_reward for b in bundles) / n

        # Pool each problem's branches across its M tuples for the pass@k
        # ladder. A branch counts as a pass only when it clears every test
        # (pass_rate >= 1.0); an unparseable tuple contributes no branches.
        from collections import defaultdict as _defaultdict
        branch_pool: dict[str, list[int]] = _defaultdict(list)
        for b in bundles:
            if not b.is_parseable:
                continue
            branch_pool[b.problem_id].extend(
                1 if pr >= 1.0 else 0 for pr in b.branch_pass_rates
            )
        pass_sweep = {
            str(kk): v
            for kk, v in mean_pass_at_k_sweep(
                [(len(v), sum(v)) for v in branch_pool.values()]
            ).items()
        }

        # Reward / advantage diagnostics. These let us tell apart
        # "training didn't move because rewards are sparse" (group_reward
        # _variance ≈ 0) from "loss looks small but gradients are real"
        # (planner/solver_adv_nonzero_rate > 0).
        planner_advs = CPPOTrainer._compute_planner_advantages(bundles)
        # adv keys are "{problem_id}_{global_idx}", values are floats
        planner_nz = sum(1 for v in planner_advs.values() if abs(v) > 1e-9)
        planner_adv_nonzero_rate = (
            planner_nz / len(planner_advs) if planner_advs else 0.0
        )

        # Solver advantages: per-branch within each parseable bundle
        total_branch_count = 0
        solver_nz = 0
        for b in bundles:
            if not b.is_parseable or not b.branch_rewards:
                continue
            advs = CPPOTrainer._compute_solver_advantages(b)
            for a in advs:
                total_branch_count += 1
                if abs(a) > 1e-9:
                    solver_nz += 1
        solver_adv_nonzero_rate = (
            solver_nz / total_branch_count if total_branch_count > 0 else 0.0
        )

        # Mean variance of plan_reward within problem groups.
        from collections import defaultdict
        groups: dict[str, list[float]] = defaultdict(list)
        for b in bundles:
            groups[b.problem_id].append(b.plan_reward)
        group_vars: list[float] = []
        for vals in groups.values():
            if len(vals) <= 1:
                continue
            m = sum(vals) / len(vals)
            v = sum((x - m) ** 2 for x in vals) / len(vals)
            group_vars.append(v)
        group_reward_variance = (
            sum(group_vars) / len(group_vars) if group_vars else 0.0
        )

        return {
            "loss": loss_value,
            "grad_norm": grad_norm,
            "param_delta": param_delta,
            "parseable_rate": parseable_rate,
            "rm_valid_rate": rm_valid_rate,
            "avg_cpsi": avg_cpsi,
            "avg_pass_prob": avg_pass_prob,
            "jpsi_pass_rate": jpsi_pass_rate,
            "outcome_success_rate": outcome_success_rate,
            "plan_reward_nonzero_rate": plan_reward_nonzero_rate,
            "rm_winner_rate": rm_winner_rate,
            "rm_reward_nonzero_rate": rm_reward_nonzero_rate,
            "winner_outcome_success_rate": winner_outcome_success_rate,
            "rm_winner_outcome_success_rate": rm_winner_outcome_success_rate,
            "plan_winner_rate": plan_winner_rate,
            "plan_winner_outcome_success_rate": plan_winner_outcome_success_rate,
            "groups_with_plan_winner_rate": groups_with_plan_winner_rate,
            "any_pass_rate": any_pass_rate,
            "zero_pass_rate": zero_pass_rate,
            "avg_outcome_reward": avg_outcome,
            "avg_plan_reward": avg_plan,
            "pass_at_k_sweep": pass_sweep,
            "planner_adv_nonzero_rate": planner_adv_nonzero_rate,
            "solver_adv_nonzero_rate": solver_adv_nonzero_rate,
            "group_reward_variance": group_reward_variance,
            "n_bundles": n,
        }
