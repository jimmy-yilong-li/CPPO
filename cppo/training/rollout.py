"""Two-phase CPPO rollout: plan -> parse -> solve each branch -> verify."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from cppo.data.problem import Problem
from cppo.data.prompts import make_plan_prompt, make_solve_prompt, parse_plan_tuple
from cppo.sandbox.executor import verify_solution
from cppo.training.planner_sft_dataset import (
    make_planner_plan_only_prompt,
    make_planner_think_prompt,
)
from cppo.training.rollout_data import CPPORolloutBundle, RolloutTokenData
from cppo.training.rewards import assign_group_argmax_rewards

if TYPE_CHECKING:
    from cppo.training.rewards import PlanRewardScorer

logger = logging.getLogger(__name__)


def _format_plan_methods(methods: list[str]) -> str:
    """Format parsed methods as canonical A/B/C/D plan text."""
    return "\n".join(
        f"{chr(ord('A') + i)}: {method.strip()}"
        for i, method in enumerate(methods)
    )


def _parse_rollout_plan_surface(raw_text: str, *, k: int) -> tuple[list[str], str] | None:
    """Parse the plan surface that should be given to solver/RM.

    Plan-only rollout must not feed Qwen thinking residue into the RM. If a
    model emits ``... </think> A/B/C/D ...``, prefer the final surface after
    the closing think marker; otherwise parse the raw response.
    """
    stripped = raw_text.strip()
    lower = stripped.lower()
    marker = "</think>"
    candidates: list[str] = []
    idx = lower.rfind(marker)
    if idx >= 0:
        candidates.append(stripped[idx + len(marker):].strip())
    candidates.append(stripped)

    for candidate in candidates:
        methods = parse_plan_tuple(candidate, k=k)
        if methods is not None:
            return methods, _format_plan_methods(methods)
    return None


def assign_cppo_plan_rewards(
    bundles: list[CPPORolloutBundle],
    mode: str,
    rm_reward_mode: str = "binary_jpsi",
    rm_winner_min_score: float | None = None,
) -> None:
    """Assign RM diagnostic rewards and planner rewards in-place.

    The published mode is `jpsi_times_outcome` with `binary_jpsi`.
    Group-argmax modes are ablation variants.
    """
    for b in bundles:
        b.rm_winner = False
        b.rm_reward = 0.0
        b.plan_winner = False
        b.plan_reward = 0.0

    if rm_reward_mode not in {"continuous", "group_argmax_binary", "binary_jpsi"}:
        raise ValueError(f"Unknown RM reward mode: {rm_reward_mode}")
    if mode in {"rm_only", "rm_times_outcome", "outcome_first_winner"}:
        if rm_reward_mode != "group_argmax_binary":
            raise ValueError(f"{mode} requires rm_reward_mode='group_argmax_binary'")
    if mode == "jpsi_times_outcome" and rm_reward_mode != "binary_jpsi":
        raise ValueError("jpsi_times_outcome requires rm_reward_mode='binary_jpsi'")

    def assign_rm_rewards() -> None:
        if rm_reward_mode == "binary_jpsi":
            for bundle in bundles:
                if bundle.is_parseable:
                    bundle.rm_reward = float(bundle.j_psi)
                    bundle.rm_winner = bundle.rm_reward > 0.0
            return
        if rm_reward_mode == "continuous":
            for bundle in bundles:
                if bundle.is_parseable:
                    bundle.rm_reward = float(bundle.c_psi)
            return

        rm_rewards = assign_group_argmax_rewards(
            scores=[b.raw_rm_score for b in bundles],
            group_ids=[b.problem_id for b in bundles],
            parseable=[b.is_parseable for b in bundles],
            min_score=rm_winner_min_score,
        )
        for bundle, rm_reward in zip(bundles, rm_rewards):
            bundle.rm_reward = float(rm_reward)
            bundle.rm_winner = rm_reward > 0.0

    if mode == "product":
        assign_rm_rewards()
        for b in bundles:
            b.plan_reward = float(b.c_psi) * float(b.outcome_reward)
            b.plan_winner = b.plan_reward > 0.0
        return

    if mode == "jpsi_times_outcome":
        assign_rm_rewards()
        for b in bundles:
            b.plan_reward = float(b.rm_reward) * float(b.outcome_reward)
            b.plan_winner = b.plan_reward > 0.0
        return

    if mode == "rm_only":
        assign_rm_rewards()
        for b in bundles:
            b.plan_reward = float(b.rm_reward)
            b.plan_winner = b.plan_reward > 0.0
        return

    if mode == "rm_times_outcome":
        assign_rm_rewards()
        for b in bundles:
            b.plan_reward = float(b.rm_reward) * float(b.outcome_reward)
            b.plan_winner = b.plan_reward > 0.0
        return

    if mode == "outcome_first_winner":
        assign_rm_rewards()
        from collections import defaultdict

        groups: dict[str, list[int]] = defaultdict(list)
        for i, b in enumerate(bundles):
            groups[b.problem_id].append(i)

        for indices in groups.values():
            candidates = []
            for idx in indices:
                b = bundles[idx]
                if not b.is_parseable:
                    continue
                if b.outcome_reward < 1.0:
                    continue
                if rm_winner_min_score is not None and b.raw_rm_score < rm_winner_min_score:
                    continue
                candidates.append(idx)
            if not candidates:
                continue
            winner_idx = max(
                candidates,
                key=lambda idx: (bundles[idx].outcome_reward, bundles[idx].raw_rm_score),
            )
            bundles[winner_idx].plan_winner = True
            bundles[winner_idx].plan_reward = 1.0
        return

    raise ValueError(f"Unknown CPPO plan reward mode: {mode}")


def _write_outcome_log(
    bundles: list[CPPORolloutBundle],
    path: str,
    step: int,
    epoch: int,
) -> None:
    """Append one JSON line per bundle to `path`.

    This is the audit trail for reward diagnostics. `j_psi` and `plan_reward`
    are the primary reward fields; `c_psi` is a diagnostic only.

    Best-effort: a write failure logs a warning and continues — training
    correctness does not depend on this log.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for b in bundles:
                rates = list(b.branch_pass_rates) if b.branch_pass_rates else []
                raw_plan_text = b.raw_plan_text or (
                    b.plan_data.response_text if b.plan_data else ""
                )
                f.write(json.dumps({
                    "step": step,
                    "epoch": epoch,
                    "problem_id": b.problem_id,
                    "plan_text": b.scored_plan_text or raw_plan_text,
                    "raw_plan_text": raw_plan_text,
                    "raw_rm_score": float(b.raw_rm_score),
                    "rm_score_source": b.rm_score_source,
                    "pass_prob": float(b.raw_rm_score),
                    "j_psi": float(b.j_psi),
                    "c_psi": float(b.c_psi),
                    "rm_winner": bool(b.rm_winner),
                    "rm_reward": float(b.rm_reward),
                    "plan_winner": bool(b.plan_winner),
                    "outcome_reward": float(b.outcome_reward),
                    "plan_reward": float(b.plan_reward),
                    "parseable": bool(b.is_parseable),
                    "plan_hit_max_tokens": bool(
                        b.plan_data.hit_max_tokens if b.plan_data else False
                    ),
                    "solver_hit_max_tokens": [
                        bool(sd.hit_max_tokens) for sd in b.solver_data
                    ],
                    "solver_skip_loss_reasons": [
                        sd.skip_loss_reason for sd in b.solver_data
                    ],
                    "branch_pass_rates": [float(r) for r in rates],
                    "branch_passed": [r >= 1.0 for r in rates],
                }) + "\n")
    except Exception as e:
        logger.warning(f"outcome_log write to {path} failed (continuing): {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize_prompt(
    tokenizer,
    text: str,
    device: str,
    *,
    enable_thinking: bool | None = False,
) -> dict:
    """Apply chat template, tokenize, and move to device."""
    from cppo.data.prompts import to_chat_text
    enc = tokenizer(
        to_chat_text(tokenizer, text, enable_thinking=enable_thinking),
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def _generate_and_get_old_logprobs(
    model,
    tokenizer,
    prompt_text: str,
    temperature: float,
    max_new_tokens: int,
    device: str,
    enable_thinking: bool | None = False,
) -> tuple[list[int], list[int], list[float], str]:
    """Generate a completion with sampling, then recompute raw old_logprobs.

    Returns:
        (prompt_token_ids, response_token_ids, old_logprobs, response_text)
    """
    inputs = _tokenize_prompt(
        tokenizer,
        prompt_text,
        device,
        enable_thinking=enable_thinking,
    )
    prompt_ids = inputs["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)

    # --- Generate ---
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_ids = output_ids[0]  # (prompt_len + gen_len,)
    response_ids = full_ids[prompt_len:].tolist()

    if len(response_ids) == 0:
        return prompt_ids, [], [], ""

    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    # --- Recompute old_logprobs via a forward pass ---
    with torch.no_grad():
        logits = model(full_ids.unsqueeze(0)).logits  # (1, L, V)

    # logits[0, t, :] predicts token at position t+1
    # For response token at position prompt_len+i, the logit is at index prompt_len+i-1
    log_probs_all = F.log_softmax(logits[0], dim=-1)  # (L, V)

    old_logprobs: list[float] = []
    for i, tok_id in enumerate(response_ids):
        pos = prompt_len + i - 1  # logit position that predicts this token
        old_logprobs.append(log_probs_all[pos, tok_id].item())

    return prompt_ids, response_ids, old_logprobs, response_text


def build_rollout_planner_prompt(
    problem: str,
    *,
    domain: str = "code",
    k: int = 4,
    planner_prompt_mode: str = "legacy_plan",
) -> str:
    """Build the planner prompt used by CPPO rollout.

    The default preserves the historical CPPO prompt. New SFT planners must
    opt in to their training prompt, otherwise rollout silently measures a
    train/deploy mismatch.
    """
    if planner_prompt_mode in {"legacy_plan", "plan"}:
        return make_plan_prompt(problem, domain=domain, k=k)
    if planner_prompt_mode == "think_plan":
        return make_planner_think_prompt(problem, domain=domain, k=k)
    if planner_prompt_mode == "plan_only":
        return make_planner_plan_only_prompt(problem, domain=domain, k=k)
    raise ValueError(f"unknown planner_prompt_mode: {planner_prompt_mode}")


# ---------------------------------------------------------------------------
# Main rollout function
# ---------------------------------------------------------------------------


def run_cppo_rollout_hf(
    policy_model,
    tokenizer,
    problems: list[Problem],
    scorer: "PlanRewardScorer",
    k: int = 4,
    m_tuples: int = 4,
    plan_temp: float = 0.9,
    solve_temp: float = 0.7,
    max_plan_tokens: int = 512,
    max_solve_tokens: int = 2048,
    exec_timeout: int = 10,
    use_pass_rate: bool = True,
    device: str = "cuda",
    outcome_log_path: str | None = None,
    step: int = 0,
    epoch: int = 0,
    rm_reward_mode: str = "binary_jpsi",
    plan_reward_mode: str = "jpsi_times_outcome",
    rm_winner_min_score: float | None = None,
    planner_prompt_mode: str = "legacy_plan",
) -> list[CPPORolloutBundle]:
    """Run two-phase CPPO rollout using HF generate().

    For each problem, generate M plan tuples. For each tuple:
      1. Generate plan, recompute old_logprobs
      2. Parse plan into k methods
      3. For each method: generate solver output, recompute old_logprobs
      4. Verify each solver output with verify_solution
      5. Compute branch_rewards, outcome_reward

    Then score all plans with scorer.score_plan_details() and compute
    plan_reward = J_psi * outcome_reward by default.

    Args:
        policy_model: HuggingFace CausalLM (the current policy).
        tokenizer: Corresponding tokenizer.
        problems: List of Problem instances.
        scorer: PlanRewardScorer for computing Pass probabilities, J_psi, and
            legacy diagnostic C_psi.
        k: Number of methods per plan tuple.
        m_tuples: Number of plan tuples per problem.
        plan_temp: Sampling temperature for plan generation.
        solve_temp: Sampling temperature for solver generation.
        max_plan_tokens: Max new tokens for plan generation.
        max_solve_tokens: Max new tokens for solver generation.
        exec_timeout: Timeout in seconds for code execution.
        device: Device string.

    Returns:
        List of CPPORolloutBundle, one per (problem, tuple) pair.
    """
    policy_model.eval()

    # Accumulate all bundles, and collect plan texts for batch scoring
    all_bundles: list[CPPORolloutBundle] = []
    plan_problem_texts: list[str] = []
    plan_texts: list[str] = []
    bundle_indices: list[int] = []  # maps scorer input -> bundle index

    for prob in problems:
        for m_idx in range(m_tuples):
            logger.debug(
                "Rollout: problem=%s tuple=%d/%d", prob.id, m_idx + 1, m_tuples
            )

            bundle = CPPORolloutBundle(
                problem_id=prob.id,
                domain=prob.domain,
            )

            # ----- Phase 1: Generate plan -----
            plan_prompt = build_rollout_planner_prompt(
                prob.prompt,
                domain=prob.domain,
                k=k,
                planner_prompt_mode=planner_prompt_mode,
            )
            (
                plan_prompt_ids,
                plan_resp_ids,
                plan_old_lp,
                plan_text,
            ) = _generate_and_get_old_logprobs(
                policy_model,
                tokenizer,
                plan_prompt,
                temperature=plan_temp,
                max_new_tokens=max_plan_tokens,
                device=device,
                enable_thinking=(planner_prompt_mode == "think_plan"),
            )

            plan_data = RolloutTokenData(
                prompt_token_ids=plan_prompt_ids,
                response_token_ids=plan_resp_ids,
                old_logprobs=plan_old_lp,
                region="planner",
                problem_id=prob.id,
                prompt_text=plan_prompt,
                response_text=plan_text,
                hit_max_tokens=(
                    max_plan_tokens > 0 and len(plan_resp_ids) >= max_plan_tokens
                ),
            )
            if plan_data.hit_max_tokens:
                plan_data.skip_loss_reason = "planner_hit_max_tokens"
            bundle.plan_data = plan_data
            bundle.raw_plan_text = plan_text

            # ----- Parse plan -----
            parsed_plan = _parse_rollout_plan_surface(plan_text, k=k)
            if parsed_plan is None or plan_data.hit_max_tokens:
                # Unparseable plan: mark as not parseable, no branches
                bundle.is_parseable = False
                bundle.methods = None
                bundle.scored_plan_text = ""
                bundle.branch_rewards = []
                bundle.branch_pass_rates = []
                bundle.outcome_reward = 0.0
                bundle.c_psi = 0.0
                bundle.j_psi = 0.0
                bundle.plan_reward = 0.0
                all_bundles.append(bundle)
                continue

            methods, scored_plan_text = parsed_plan
            bundle.is_parseable = True
            bundle.methods = methods
            bundle.scored_plan_text = scored_plan_text

            # ----- Phase 2: Solve each branch -----
            branch_rewards: list[float] = []
            branch_pass_rates: list[float] = []
            solver_data_list: list[RolloutTokenData] = []

            for branch_idx, method in enumerate(methods):
                solve_prompt = make_solve_prompt(
                    prob.prompt,
                    method,
                    domain=prob.domain,
                    io_mode=prob.io_mode,
                    entry_point=prob.entry_point,
                )

                (
                    solve_prompt_ids,
                    solve_resp_ids,
                    solve_old_lp,
                    solve_text,
                ) = _generate_and_get_old_logprobs(
                    policy_model,
                    tokenizer,
                    solve_prompt,
                    temperature=solve_temp,
                    max_new_tokens=max_solve_tokens,
                    device=device,
                    enable_thinking=False,
                )

                sd = RolloutTokenData(
                    prompt_token_ids=solve_prompt_ids,
                    response_token_ids=solve_resp_ids,
                    old_logprobs=solve_old_lp,
                    region="solver",
                    problem_id=prob.id,
                    branch_idx=branch_idx,
                    prompt_text=solve_prompt,
                    response_text=solve_text,
                    hit_max_tokens=(
                        max_solve_tokens > 0 and len(solve_resp_ids) >= max_solve_tokens
                    ),
                )
                if sd.hit_max_tokens:
                    sd.skip_loss_reason = "solver_hit_max_tokens"
                solver_data_list.append(sd)

                if sd.hit_max_tokens:
                    branch_rewards.append(0.0)
                    branch_pass_rates.append(0.0)
                    continue

                # ----- Verify -----
                vr = verify_solution(prob, solve_text, timeout=exec_timeout)
                if use_pass_rate:
                    reward = vr.pass_rate
                else:
                    reward = 1.0 if vr.passed else 0.0
                branch_rewards.append(reward)
                branch_pass_rates.append(vr.pass_rate)

            bundle.solver_data = solver_data_list
            bundle.branch_rewards = branch_rewards
            bundle.branch_pass_rates = branch_pass_rates
            bundle.outcome_reward = max(branch_rewards) if branch_rewards else 0.0

            # Collect for batch scoring
            bundle_idx = len(all_bundles)
            plan_problem_texts.append(prob.prompt)
            plan_texts.append(scored_plan_text)
            bundle_indices.append(bundle_idx)

            all_bundles.append(bundle)

    # ----- Score plans in batch -----
    if plan_problem_texts:
        if hasattr(scorer, "score_plan_details"):
            score_details = scorer.score_plan_details(plan_problem_texts, plan_texts)
            score_source = "pass_prob"
        else:
            if plan_reward_mode in {
                "jpsi_times_outcome",
                "rm_only",
                "rm_times_outcome",
                "outcome_first_winner",
            }:
                raise RuntimeError(
                    f"{plan_reward_mode} requires scorer.score_plan_details() "
                    "so training uses raw P(Pass)/J_psi, not calibrated C_psi."
                )
            c_psi_values = scorer.score_plans(plan_problem_texts, plan_texts)
            score_details = [
                {"pass_prob": float(c_psi), "j_psi": 1.0 if c_psi >= 0.5 else 0.0, "c_psi": float(c_psi)}
                for c_psi in c_psi_values
            ]
            score_source = "c_psi_fallback"
        for i, bi in enumerate(bundle_indices):
            detail = score_details[i]
            all_bundles[bi].raw_rm_score = float(detail["pass_prob"])
            all_bundles[bi].rm_score_source = score_source
            all_bundles[bi].c_psi = float(detail["c_psi"])
            all_bundles[bi].j_psi = float(detail["j_psi"])

        assign_cppo_plan_rewards(
            all_bundles,
            mode=plan_reward_mode,
            rm_reward_mode=rm_reward_mode,
            rm_winner_min_score=rm_winner_min_score,
        )

    # ----- Per-bundle outcome log (hard-negative mining trail) -----
    if outcome_log_path is not None:
        _write_outcome_log(all_bundles, outcome_log_path, step=step, epoch=epoch)

    return all_bundles
