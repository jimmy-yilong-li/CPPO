#!/usr/bin/env python3
"""Run on-policy CPPO reward-density preflight.

This script measures whether a rollout problem pool has enough nonzero and
variable planner reward before any CPPO training starts:

    R_plan = Jpsi * R_out

where Jpsi is the frozen binary tuple RM gate and R_out is frozen-solver pass@4
for the generated plan tuple.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cppo.data.problem import Problem
from cppo.eval.metrics import mean_pass_at_k_sweep
from cppo.data.prompts import make_solve_prompt, to_chat_text
from cppo.sandbox.executor import verify_solution
from cppo.training.planner_sft_dataset import (
    make_planner_plan_only_prompt,
    make_planner_think_prompt,
)
from cppo.training.rewards import PlanRewardScorer
from cppo.training.rollout import _parse_rollout_plan_surface


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_problems(path: str | Path) -> list[Problem]:
    return [Problem.from_dict(row) for row in load_jsonl(path)]


def select_problem_shard(
    problems: list[Problem],
    *,
    num_problems: int,
    seed: int,
    shard_index: int,
    num_shards: int,
) -> list[Problem]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    shuffled = list(problems)
    random.Random(seed).shuffle(shuffled)
    selected = shuffled[:num_problems] if num_problems > 0 else shuffled
    return [p for i, p in enumerate(selected) if i % num_shards == shard_index]


def summarize_rows(rows: list[dict[str, Any]], k: int = 4) -> dict[str, Any]:
    """Summarize audited rollout rows.

    ``k`` is the planner tuple size (branches per tuple). It is threaded in so
    the reported solver metric is named for the budget actually run instead of
    a hardcoded 4.
    """
    by_problem: dict[str, list[int]] = defaultdict(list)
    # Pooled branch outcomes per problem, for the pass@K ladder. Branches of one
    # tuple are strategy-conditioned rather than iid, so the pooled estimate
    # treats the M*k branches of a problem as exchangeable — the same protocol
    # the paper uses to report budgets above the tuple size.
    branches_by_problem: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        pid = str(row.get("problem_id", ""))
        by_problem[pid].append(int(row.get("R_plan", 0)))
        rewards = row.get("branch_rewards")
        if rewards:
            branches_by_problem[pid].extend(int(r) for r in rewards)
        else:
            # Unparseable / fallback tuples still consume their share of the budget.
            branches_by_problem[pid].extend([0] * k)

    tuple_count = len(rows)
    problem_count = len(by_problem)
    nonzero_problem_count = sum(1 for rewards in by_problem.values() if any(rewards))
    variance_problem_count = sum(
        1 for rewards in by_problem.values()
        if len(rewards) >= 2 and len(set(rewards)) > 1
    )
    all_zero_count = sum(1 for rewards in by_problem.values() if rewards and not any(rewards))
    all_one_count = sum(1 for rewards in by_problem.values() if rewards and all(rewards))

    def rate(field: str) -> float:
        return (
            sum(1 for row in rows if bool(row.get(field))) / tuple_count
            if tuple_count
            else 0.0
        )

    r_plan_nonzero = sum(1 for row in rows if int(row.get("R_plan", 0)) > 0)
    jpsi_pass = sum(1 for row in rows if int(row.get("Jpsi", 0)) > 0)
    solver_pass = sum(1 for row in rows if int(row.get("R_out", 0)) > 0)
    passed_branches = sum(int(row.get("num_passed_branches", 0)) for row in rows)
    think_detected = sum(1 for row in rows if bool(row.get("think_detected")))

    return {
        "reward_definition": "R_plan = Jpsi * R_out",
        "density_estimator": "on_policy_rollout_pool_preflight",
        "problem_count": problem_count,
        "tuple_count": tuple_count,
        "parse_rate": rate("parseable"),
        "fallback_rate": rate("fallback"),
        "think_detected_rate": think_detected / tuple_count if tuple_count else 0.0,
        "jpsi_pass_rate": jpsi_pass / tuple_count if tuple_count else 0.0,
        "k": k,
        "solver_pass_at_k": solver_pass / tuple_count if tuple_count else 0.0,
        "solver_pass_at_k_sweep": {
            str(kk): v for kk, v in mean_pass_at_k_sweep(
                [(len(b), sum(b)) for b in branches_by_problem.values()]
            ).items()
        },
        "r_plan_nonzero_rate": r_plan_nonzero / tuple_count if tuple_count else 0.0,
        "per_problem_nonzero_rate": (
            nonzero_problem_count / problem_count if problem_count else 0.0
        ),
        "reward_variance_rate": (
            variance_problem_count / problem_count if problem_count else 0.0
        ),
        "group_all_zero_rate": all_zero_count / problem_count if problem_count else 0.0,
        "group_all_one_rate": all_one_count / problem_count if problem_count else 0.0,
        "mean_passed_branches": passed_branches / tuple_count if tuple_count else 0.0,
    }


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(p)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_existing_keys(path: str | Path) -> set[tuple[str, int]]:
    p = Path(path)
    if not p.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((str(row.get("problem_id")), int(row.get("tuple_id", -1))))
    return keys


def build_planner_prompt(problem: str, *, domain: str, k: int, prompt_mode: str) -> str:
    if prompt_mode == "think_plan":
        return make_planner_think_prompt(problem, domain=domain, k=k)
    if prompt_mode == "plan_only":
        return make_planner_plan_only_prompt(problem, domain=domain, k=k)
    raise ValueError(f"unknown planner prompt mode: {prompt_mode}")


def seed_torch(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: Any, base_seed: int) -> int:
    data = "|".join(str(p) for p in (base_seed, *parts)).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


def load_model(path: str, *, device_map: str = "auto") -> tuple[torch.nn.Module, Any]:
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    return model, tok


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tok: Any,
    prompt: str,
    *,
    temperature: float,
    max_new_tokens: int,
) -> str:
    enc = tok(to_chat_text(tok, prompt, enable_thinking=False), return_tensors="pt")
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )[0]
    return tok.decode(out[enc["input_ids"].shape[1]:], skip_special_tokens=True)


def score_plan(
    scorer: PlanRewardScorer,
    *,
    problem_text: str,
    plan_text: str,
    threshold: float,
) -> tuple[float, int, dict[str, Any]]:
    detail = scorer.score_plan_details([problem_text], [plan_text])[0]
    score = float(detail["pass_prob"])
    return score, int(score >= threshold), detail


def run_one_tuple(
    *,
    problem: Problem,
    tuple_id: int,
    planner_model: torch.nn.Module,
    planner_tok: Any,
    solver_model: torch.nn.Module,
    solver_tok: Any,
    scorer: PlanRewardScorer,
    k: int,
    prompt_mode: str,
    plan_temp: float,
    solve_temp: float,
    max_plan_tokens: int,
    max_solve_tokens: int,
    exec_timeout: int,
    rm_threshold: float,
    base_seed: int,
) -> dict[str, Any]:
    plan_seed = stable_seed(problem.id, "plan", tuple_id, base_seed=base_seed)
    seed_torch(plan_seed)
    plan_prompt = build_planner_prompt(
        problem.prompt,
        domain=problem.domain,
        k=k,
        prompt_mode=prompt_mode,
    )
    raw_plan_text = generate(
        planner_model,
        planner_tok,
        plan_prompt,
        temperature=plan_temp,
        max_new_tokens=max_plan_tokens,
    )
    parsed_plan = _parse_rollout_plan_surface(raw_plan_text, k=k)
    if parsed_plan is None:
        return {
            "problem_id": problem.id,
            "tuple_id": tuple_id,
            "parseable": False,
            "fallback": True,
            "rm_score": 0.0,
            "Jpsi": 0,
            "branch_rewards": [],
            "R_out": 0,
            "R_plan": 0,
            "num_passed_branches": 0,
            "plan_text": "",
            "raw_plan_text": raw_plan_text,
            "plan_text_head": "",
            "raw_plan_text_head": raw_plan_text[:1000],
            "think_detected": "</think>" in raw_plan_text.lower() or "<think>" in raw_plan_text.lower(),
            "seeds": {"plan": plan_seed, "solve": []},
        }
    methods, plan_text = parsed_plan

    rm_score, jpsi, rm_detail = score_plan(
        scorer,
        problem_text=problem.prompt,
        plan_text=plan_text,
        threshold=rm_threshold,
    )

    branch_rewards: list[int] = []
    branch_pass_rates: list[float] = []
    solve_seeds: list[int] = []
    attempts: list[dict[str, Any]] = []
    for branch_idx, method in enumerate(methods):
        solve_seed = stable_seed(
            problem.id, "solve", tuple_id, branch_idx, base_seed=base_seed
        )
        solve_seeds.append(solve_seed)
        seed_torch(solve_seed)
        solve_prompt = make_solve_prompt(
            problem.prompt,
            method,
            domain=problem.domain,
            io_mode=problem.io_mode,
            entry_point=problem.entry_point,
        )
        solve_text = generate(
            solver_model,
            solver_tok,
            solve_prompt,
            temperature=solve_temp,
            max_new_tokens=max_solve_tokens,
        )
        vr = verify_solution(problem, solve_text, timeout=exec_timeout)
        passed = int(bool(vr.passed))
        branch_rewards.append(passed)
        branch_pass_rates.append(float(vr.pass_rate))
        attempts.append({
            "branch_idx": branch_idx,
            "passed": bool(vr.passed),
            "pass_rate": float(vr.pass_rate),
            "tests_passed": int(vr.tests_passed),
            "tests_total": int(vr.tests_total),
            "method_head": method[:300],
            "solution_head": solve_text[:500],
        })

    r_out = int(any(branch_rewards))
    r_plan = int(jpsi and r_out)
    return {
        "problem_id": problem.id,
        "tuple_id": tuple_id,
        "parseable": True,
        "fallback": False,
        "rm_score": rm_score,
        "rm_detail": rm_detail,
        "Jpsi": jpsi,
        "branch_rewards": branch_rewards,
        "branch_pass_rates": branch_pass_rates,
        "R_out": r_out,
        "R_plan": r_plan,
        "num_passed_branches": sum(branch_rewards),
        "plan_text": plan_text,
        "raw_plan_text": raw_plan_text,
        "plan_text_head": plan_text[:1000],
        "raw_plan_text_head": raw_plan_text[:1000],
        "methods_head": [m[:300] for m in methods],
        "attempts": attempts,
        "think_detected": "</think>" in raw_plan_text.lower() or "<think>" in raw_plan_text.lower(),
        "seeds": {"plan": plan_seed, "solve": solve_seeds},
    }


def write_quadrant_examples(rows: list[dict[str, Any]], path: str | Path, *, max_per: int = 5) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"jpsi_{int(row.get('Jpsi', 0))}_solver_{int(row.get('R_out', 0))}"
        if len(groups[key]) < max_per:
            groups[key].append(row)
    write_json(path, groups)


def run_preflight(args: argparse.Namespace) -> None:
    problems = select_problem_shard(
        load_problems(args.problems),
        num_problems=args.num_problems,
        seed=args.seed,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    planner_model, planner_tok = load_model(args.planner, device_map=args.planner_device_map)
    solver_model, solver_tok = load_model(args.solver, device_map=args.solver_device_map)
    scorer = PlanRewardScorer(args.rm, device=args.rm_device)

    out_path = Path(args.output)
    existing = load_existing_keys(out_path) if args.resume else set()
    for problem in problems:
        for tuple_id in range(args.planner_tuples_per_problem):
            if (problem.id, tuple_id) in existing:
                continue
            started = time.time()
            row = run_one_tuple(
                problem=problem,
                tuple_id=tuple_id,
                planner_model=planner_model,
                planner_tok=planner_tok,
                solver_model=solver_model,
                solver_tok=solver_tok,
                scorer=scorer,
                k=args.k,
                prompt_mode=args.prompt_mode,
                plan_temp=args.plan_temp,
                solve_temp=args.solve_temp,
                max_plan_tokens=args.max_plan_tokens,
                max_solve_tokens=args.max_solve_tokens,
                exec_timeout=args.exec_timeout,
                rm_threshold=args.rm_threshold,
                base_seed=args.seed,
            )
            row.update({
                "difficulty": problem.difficulty,
                "source": problem.source,
                "io_mode": problem.io_mode,
                "prompt_mode": args.prompt_mode,
                "k": args.k,
                "planner_tuples_per_problem": args.planner_tuples_per_problem,
                "elapsed_sec": time.time() - started,
            })
            append_jsonl(out_path, row)

    rows = load_jsonl(out_path)
    summary = summarize_rows(rows, k=args.k)
    summary.update({
        "planner_tuples_per_problem": args.planner_tuples_per_problem,
        "problems_path": str(args.problems),
        "planner": str(args.planner),
        "solver": str(args.solver),
        "rm": str(args.rm),
        "rm_threshold": args.rm_threshold,
        "num_problems_requested": args.num_problems,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "acceptance": {
            "go_planner_only_smoke": (
                summary["r_plan_nonzero_rate"] >= 0.05
                and summary["per_problem_nonzero_rate"] >= 0.25
                and summary["reward_variance_rate"] >= 0.10
                and 0.02 < summary["jpsi_pass_rate"] < 0.98
                and summary["parse_rate"] >= 0.95
            ),
            "no_go_sparse": (
                summary["r_plan_nonzero_rate"] < 0.03
                or summary["reward_variance_rate"] < 0.05
                or summary["group_all_zero_rate"] > 0.90
            ),
        },
    })
    write_json(args.summary_output, summary)
    if args.quadrant_examples:
        write_quadrant_examples(rows, args.quadrant_examples)
    print(json.dumps(summary, indent=2, sort_keys=True))
    del planner_model, solver_model, scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--planner", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--rm", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--quadrant-examples", default=None)
    parser.add_argument("--num-problems", type=int, default=100)
    parser.add_argument("--planner-tuples-per-problem", type=int, default=4)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--prompt-mode", default="think_plan", choices=["think_plan", "plan_only"])
    parser.add_argument("--plan-temp", type=float, default=0.9)
    parser.add_argument("--solve-temp", type=float, default=0.7)
    parser.add_argument("--max-plan-tokens", type=int, default=1024)
    parser.add_argument("--max-solve-tokens", type=int, default=2048)
    parser.add_argument("--exec-timeout", type=int, default=10)
    parser.add_argument("--rm-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--planner-device-map", default="auto")
    parser.add_argument("--solver-device-map", default="auto")
    parser.add_argument("--rm-device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_preflight(parse_args())


if __name__ == "__main__":
    main()
