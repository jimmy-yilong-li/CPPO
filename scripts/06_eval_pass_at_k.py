"""Comprehensive pass@K evaluation with baselines and ablations.

Baselines:
- base_independent: Base model, K independent samples (no planner)
- cppo_independent: CPPO model, K independent samples (no planner)
- base_plan_solve: Base model, plan→solve (no CPPO training)
- cppo_plan_solve: CPPO model, plan→solve (full CPPO)

Eval datasets: APPS test, CodeContests valid, LiveCodeBench
"""
import argparse
import json
import yaml
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cppo.data.apps_loader import load_apps
from cppo.data.codecontests_loader import load_codecontests
from cppo.data.livecodebench_loader import load_livecodebench
from cppo.data.prompts import make_plan_prompt, make_solve_prompt, parse_plan_tuple
from cppo.eval.metrics import mean_pass_at_k_sweep, pass_at_k
from cppo.sandbox.executor import verify_solution


def _is_complete_hf_ckpt(d: str) -> bool:
    """Directory has config.json AND any tokenizer file. Per-epoch
    subdirs of a CPPO run are 'complete'; the run-id parent dir is not."""
    p = Path(d)
    if not p.is_dir():
        return False
    has_config = (p / "config.json").is_file()
    has_tok = any(
        (p / name).is_file()
        for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
    )
    return has_config and has_tok


def _resolve_cppo_path(cppo_arg: str | None, yaml_default: str) -> tuple[str, bool]:
    """Pick the CPPO checkpoint to evaluate.

    Resolution order:
      1. --cppo-checkpoint CLI override always wins.
      2. yaml_default itself is a complete HF ckpt (legacy single-run
         layout) → use it.
      3. yaml_default/latest exists (06 writes this symlink after each
         completed run). Prefer the run ROOT (06 writes the FINAL model there
         after training ends — `06_run_cppo.py:284-287`); only fall
         back to the highest epoch_N subdir if root isn't complete
         (e.g. training crashed before final save).
      4. Otherwise return (yaml_default, False).
    """
    if cppo_arg:
        return cppo_arg, _is_complete_hf_ckpt(cppo_arg)

    yaml_p = Path(yaml_default)
    if _is_complete_hf_ckpt(yaml_default):
        return yaml_default, True

    latest = yaml_p / "latest"
    if latest.exists() or latest.is_symlink():
        try:
            target = latest.resolve()
            # PREFER the run root (final model post-training).
            if _is_complete_hf_ckpt(str(target)):
                return str(target), True
            # Fall back to highest complete epoch_N (training likely
            # didn't reach the final-save line).
            epochs = sorted(
                (d for d in target.iterdir()
                 if d.name.startswith("epoch_") and _is_complete_hf_ckpt(str(d))),
                key=lambda d: int(d.name.split("_")[1]),
            )
            if epochs:
                return str(epochs[-1]), True
        except Exception:
            pass

    return yaml_default, False


def _load_model(path, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, prompt, temperature=0.7, max_new_tokens=2048):
    from cppo.data.prompts import to_chat_text
    enc = tokenizer(to_chat_text(tokenizer, prompt), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, enc["input_ids"].size(1):], skip_special_tokens=True)


def eval_independent(model, tokenizer, problems, k, solve_temp, exec_timeout,
                     n_samples=None):
    """Baseline: independent samples, no planner.

    Draws ``n_samples`` completions per problem (default: k) and reports the
    unbiased pass@k for the headline budget plus the full ladder every pool
    supports. Sampling n > k is what makes pass@8/16/32 estimable from one
    pool instead of re-sampling per budget.
    """
    n_samples = k if n_samples is None else n_samples
    if n_samples < k:
        raise ValueError(f"n_samples ({n_samples}) must be >= k ({k})")
    results = defaultdict(lambda: {"n": 0, "c": 0, "total_tokens": 0})
    for prob in problems:
        if prob.io_mode == "call_based" and prob.entry_point:
            base_prompt = (
                f"Solve the following programming problem. Implement the Python "
                f"function `{prob.entry_point}`. Do not read from stdin.\n"
                f"Wrap your final solution in a single ```python ... ``` code block.\n\n"
                f"Problem:\n{prob.prompt}"
            )
        else:
            base_prompt = (
                "Solve the following programming problem. "
                "Write a complete Python solution that reads from stdin and writes to stdout.\n"
                "Wrap your final solution in a single ```python ... ``` code block.\n\n"
                f"Problem:\n{prob.prompt}"
            )
        for _ in range(n_samples):
            output = _generate(model, tokenizer, base_prompt, temperature=solve_temp)
            results[prob.id]["n"] += 1
            results[prob.id]["total_tokens"] += len(tokenizer.encode(output))
            vr = verify_solution(prob, output, timeout=exec_timeout)
            if vr.passed:
                results[prob.id]["c"] += 1

    scores = [pass_at_k(r["n"], r["c"], k) for r in results.values() if r["n"] > 0]
    total_tokens = sum(r["total_tokens"] for r in results.values())
    sweep = mean_pass_at_k_sweep([(r["n"], r["c"]) for r in results.values()])
    return {
        "pass_at_k": sum(scores) / len(scores) if scores else 0.0,
        "pass_at_k_sweep": {str(kk): v for kk, v in sweep.items()},
        "n_samples": n_samples,
        "n_problems": len(scores),
        "total_tokens": total_tokens,
    }


def eval_plan_solve(planner_model, planner_tok, solver_model, solver_tok,
                    problems, k, plan_temp, solve_temp, exec_timeout,
                    n_samples=None, k_tuple=None):
    """Plan strategy tuples with planner_model, solve each branch with solver_model.

    Pass the SAME model+tokenizer for both args to evaluate self-paired
    plan+solve. To isolate the planner contribution, pass a trained planner
    and the (frozen) base model as solver.

    Budgets above the tuple size are reached by *pooling* tuples, not by
    retraining a wider planner: one k_tuple-method planner is sampled
    ceil(n_samples / k_tuple) times, and every branch joins one pool. With
    k_tuple=4 and n_samples=32 that is 8 tuples of 4 branches.
    """
    k_tuple = k if k_tuple is None else k_tuple
    n_samples = k if n_samples is None else n_samples
    if n_samples < k:
        raise ValueError(f"n_samples ({n_samples}) must be >= k ({k})")
    n_tuples = -(-n_samples // k_tuple)  # ceil

    results = defaultdict(lambda: {"n": 0, "c": 0, "total_tokens": 0, "parseable": 0, "attempted": 0})

    for prob in problems:
        for _ in range(n_tuples):
            plan_prompt = make_plan_prompt(prob.prompt, domain=prob.domain, k=k_tuple)
            plan_output = _generate(planner_model, planner_tok, plan_prompt, temperature=plan_temp, max_new_tokens=512)
            results[prob.id]["total_tokens"] += len(planner_tok.encode(plan_output))
            results[prob.id]["attempted"] += 1

            methods = parse_plan_tuple(plan_output, k=k_tuple)
            if methods is None:
                # An unparseable tuple still consumes its share of the budget.
                results[prob.id]["n"] += k_tuple
                continue

            results[prob.id]["parseable"] += 1
            for method in methods:
                solve_prompt = make_solve_prompt(
                    prob.prompt, method, domain=prob.domain,
                    io_mode=prob.io_mode, entry_point=prob.entry_point,
                )
                solver_output = _generate(solver_model, solver_tok, solve_prompt, temperature=solve_temp)
                results[prob.id]["n"] += 1
                results[prob.id]["total_tokens"] += len(solver_tok.encode(solver_output))
                vr = verify_solution(prob, solver_output, timeout=exec_timeout)
                if vr.passed:
                    results[prob.id]["c"] += 1

    scores = [pass_at_k(r["n"], r["c"], k) for r in results.values() if r["n"] > 0]
    total_tokens = sum(r["total_tokens"] for r in results.values())
    parseable_rate = (
        sum(r["parseable"] for r in results.values())
        / max(sum(r["attempted"] for r in results.values()), 1)
    )
    sweep = mean_pass_at_k_sweep([(r["n"], r["c"]) for r in results.values()])
    return {
        "pass_at_k": sum(scores) / len(scores) if scores else 0.0,
        "pass_at_k_sweep": {str(kk): v for kk, v in sweep.items()},
        "n_samples": n_samples,
        "k_tuple": k_tuple,
        "n_problems": len(scores),
        "total_tokens": total_tokens,
        "parseable_rate": parseable_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="CPPO pass@K evaluation")
    parser.add_argument("--datasets", nargs="+", default=["apps"],
                        choices=["apps", "codecontests", "livecodebench"])
    parser.add_argument("--max-problems", type=int, default=20)
    parser.add_argument("--k", type=int, default=4,
                        help="Headline attempt budget reported as pass@k.")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Completions sampled per problem (default: --k). "
                             "Set >= 32 to read pass@32 off the same pool; the "
                             "ladder is 1/2/4/8/16/32 truncated at n.")
    parser.add_argument("--k-tuple", type=int, default=None,
                        help="Planner tuple size (default: --k). Budgets above "
                             "the tuple size are reached by pooling tuples, "
                             "matching the paper's K>4 protocol.")
    parser.add_argument("--difficulty", default="introductory",
                        help="APPS difficulty filter")
    parser.add_argument("--max-test-cases", type=int, default=3)
    parser.add_argument("--clean-ids-file", default=None,
                        help="JSON {clean_ids: [...]} produced by 00b_build_clean_subset.py")
    parser.add_argument(
        "--output", default="data/eval_results.json",
        help="JSON output path (results + reproducibility metadata). "
             "Pass an explicit per-run path for paper rows so re-runs do "
             "not silently overwrite each other.",
    )
    parser.add_argument(
        "--base-config", default="configs/base.yaml",
        help="Path to the base YAML config supplying model/data/paths. "
             "Pass an explicit base config to override the default; "
             "here so 08 evaluates the right configuration.",
    )
    parser.add_argument(
        "--cppo-checkpoint", default=None,
        help="Override path to the CPPO checkpoint to evaluate. Defaults to "
             "the chosen base-config's `paths.cppo_checkpoint`. Use this when "
             "the actual run wrote to a run-id subdirectory like "
             "`checkpoints/cppo/<run_id>/epoch_8`.",
    )
    parser.add_argument("--skip-cppo", action="store_true",
                        help="Skip CPPO checkpoint evaluation (only run base baselines).")
    args = parser.parse_args()

    with open(args.base_config) as f:
        cfg = yaml.safe_load(f)
    print(f"base-config: {args.base_config}", flush=True)

    k = args.k
    n_samples = args.n_samples if args.n_samples is not None else k
    k_tuple = args.k_tuple if args.k_tuple is not None else k
    if n_samples < k:
        parser.error(f"--n-samples ({n_samples}) must be >= --k ({k})")
    print(f"budget: pass@{k} from n={n_samples} samples/problem "
          f"(planner tuple size {k_tuple})", flush=True)
    plan_temp = cfg["planner"]["temperature"]
    solve_temp = cfg["solver"]["temperature"]
    exec_timeout = cfg["data"]["exec_timeout"]

    clean_ids = None
    if args.clean_ids_file:
        with open(args.clean_ids_file) as f:
            clean_ids = set(json.load(f)["clean_ids"])
        print(f"Filter: {len(clean_ids)} clean IDs from {args.clean_ids_file}", flush=True)

    # Load eval datasets (apply clean filter to APPS only)
    datasets = {}
    if "apps" in args.datasets:
        load_n = args.max_problems * 4 if clean_ids else args.max_problems
        loaded = load_apps(
            split="test", max_problems=load_n,
            difficulty=args.difficulty, min_tests=2,
            max_test_cases=args.max_test_cases,
        )
        if clean_ids:
            loaded = [p for p in loaded if p.id in clean_ids][: args.max_problems]
        else:
            loaded = loaded[: args.max_problems]
        datasets["APPS_test"] = loaded
    if "codecontests" in args.datasets:
        datasets["CodeContests_valid"] = load_codecontests(
            split="valid", max_problems=args.max_problems, max_test_cases=args.max_test_cases,
        )
    if "livecodebench" in args.datasets:
        datasets["LiveCodeBench"] = load_livecodebench(
            max_problems=args.max_problems, version_tag=cfg["data"]["lcb_version_tag"],
            max_test_cases=args.max_test_cases,
        )

    base_path = cfg["model"]["base"]
    yaml_cppo_root = cfg["paths"]["cppo_checkpoint"]
    cppo_path, cppo_complete = _resolve_cppo_path(args.cppo_checkpoint, yaml_cppo_root)
    print(f"CPPO checkpoint path: {cppo_path} (complete={cppo_complete})", flush=True)

    eval_cppo = not args.skip_cppo and cppo_complete
    if not args.skip_cppo and not eval_cppo:
        print(f"[warn] CPPO checkpoint '{cppo_path}' missing config.json or tokenizer — running base baselines only.", flush=True)

    print(f"\nLoading base model: {base_path}", flush=True)
    base_model, base_tok = _load_model(base_path)
    cppo_model = cppo_tok = None
    if eval_cppo:
        print(f"Loading CPPO model: {cppo_path}", flush=True)
        cppo_model, cppo_tok = _load_model(cppo_path)

    all_results = {}

    for ds_name, problems in datasets.items():
        print(f"\n{'#'*60}\n# Dataset: {ds_name} ({len(problems)} problems, K={k})\n{'#'*60}", flush=True)
        if not problems:
            continue
        ds_results = {}

        # 1. base independent
        print("[1/4] base independent...", flush=True)
        ds_results["base_indep"] = eval_independent(
            base_model, base_tok, problems, k, solve_temp, exec_timeout,
            n_samples=n_samples,
        )
        # 2. cppo independent (does the trained policy do better as a vanilla solver?)
        if eval_cppo:
            print("[2/4] cppo independent...", flush=True)
            ds_results["cppo_indep"] = eval_independent(
                cppo_model, cppo_tok, problems, k, solve_temp, exec_timeout,
                n_samples=n_samples,
            )
        # 3. base plan-solve (no training; prompt-only planning)
        print("[3/4] base planner + base solver...", flush=True)
        ds_results["base_plan_solve"] = eval_plan_solve(
            base_model, base_tok, base_model, base_tok,
            problems, k, plan_temp, solve_temp, exec_timeout,
            n_samples=n_samples, k_tuple=k_tuple,
        )
        # 4. CPPO planner + CPPO solver (the headline result: the joint
        #    policy CPPOTrainer actually trains is used end-to-end).
        if eval_cppo:
            print("[4/5] cppo planner + cppo solver (joint policy)...", flush=True)
            ds_results["cppo_plan_solve"] = eval_plan_solve(
                cppo_model, cppo_tok, cppo_model, cppo_tok,
                problems, k, plan_temp, solve_temp, exec_timeout,
                n_samples=n_samples, k_tuple=k_tuple,
            )
        # 5. CPPO planner + base solver (ablation: did the planner alone help
        #    a vanilla base solver? Comparable to 10_planner_smoke_audit.py.)
        if eval_cppo:
            print("[5/5] cppo planner + base solver (ablation)...", flush=True)
            ds_results["cppo_planner_base_solver"] = eval_plan_solve(
                cppo_model, cppo_tok, base_model, base_tok,
                problems, k, plan_temp, solve_temp, exec_timeout,
                n_samples=n_samples, k_tuple=k_tuple,
            )

        all_results[ds_name] = ds_results
        for method, r in ds_results.items():
            extra = f" parseable={r.get('parseable_rate', 'n/a')}" if "parseable_rate" in r else ""
            sweep = r.get("pass_at_k_sweep") or {}
            ladder = "  ".join(f"@{kk}={v:.4f}" for kk, v in sweep.items())
            print(f"  {method}: pass@{k}={r['pass_at_k']:.4f}{extra} tokens={r['total_tokens']}", flush=True)
            if ladder:
                print(f"      ladder: {ladder}", flush=True)

    # Summary table — show SKIP for any method that didn't run (missing CPPO ckpt).
    # cppo_plan_solve is the headline (joint policy used end-to-end);
    # cppo_planner_base_solver is the ablation matching 10's audit semantics.
    print(f"\n{'='*108}\nSUMMARY (pass@{k})\n{'='*108}", flush=True)
    print(
        f"{'Dataset':<20} {'BaseIndep':>10} {'CPPOIndep':>10} "
        f"{'BasePlan':>10} {'CPPOplan+CPPOslv':>18} {'CPPOpln+BaseSlv':>17}",
        flush=True,
    )
    print("-" * 108, flush=True)

    def _fmt(d, key, width):
        v = d.get(key)
        if v is None:
            return f"{'SKIP':>{width}}"
        return f"{v['pass_at_k']:>{width}.4f}"

    for ds_name, r in all_results.items():
        s = (
            f"{ds_name:<20} "
            f"{_fmt(r, 'base_indep', 10)} "
            f"{_fmt(r, 'cppo_indep', 10)} "
            f"{_fmt(r, 'base_plan_solve', 10)} "
            f"{_fmt(r, 'cppo_plan_solve', 18)} "
            f"{_fmt(r, 'cppo_planner_base_solver', 17)}"
        )
        print(s, flush=True)

    # Save results with full reproducibility metadata. Without this,
    # paper-facing numbers cannot be linked back to the exact config /
    # checkpoint / dataset filters that produced them.
    from datetime import datetime, timezone
    payload = {
        "results": all_results,
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "base_config_path": args.base_config,
            "base_model": base_path,
            "cppo_checkpoint_resolved": cppo_path,
            "cppo_checkpoint_complete": cppo_complete,
            "skip_cppo": args.skip_cppo,
            "k": k,
            "datasets_requested": args.datasets,
            "max_problems": args.max_problems,
            "difficulty": args.difficulty,
            "max_test_cases": args.max_test_cases,
            "clean_ids_file": args.clean_ids_file,
            "clean_ids_count": (len(clean_ids) if clean_ids else None),
            "plan_temp": plan_temp,
            "solve_temp": solve_temp,
            "exec_timeout": exec_timeout,
        },
    }
    out_path = args.output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
