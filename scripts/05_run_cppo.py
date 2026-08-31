#!/usr/bin/env python3
"""Run CPPO training: two-phase rollout with product reward and split advantages.

Loads problems from APPS / CodeContests, creates a CPPOTrainer from
a warmup checkpoint, and runs training with periodic logging and saves.

Usage:
    python scripts/04_run_cppo.py --config configs/cppo.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from cppo.data.apps_loader import load_apps
from cppo.data.codecontests_loader import load_codecontests
from cppo.data.problem import Problem
from cppo.data.prompts import to_chat_text
from cppo.training.config_guard import assert_not_archived_config
from cppo.training.cppo_trainer import CPPOTrainer
from cppo.training.rollout import build_rollout_planner_prompt
from cppo.training.run_provenance import (
    assert_run_dir_clean as _assert_run_dir_clean,
    write_run_provenance as _write_run_provenance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _update_latest_symlink(base_save_dir: Path, run_id: str) -> None:
    """Point latest at a completed run directory.

    Call this only after the final checkpoint has been saved. A failed
    launch should leave latest pointing at the previous usable run.
    """
    latest_link = base_save_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_id)
    logger.info("symlink %s -> %s", latest_link, run_id)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def load_problems(base_cfg: dict) -> list[Problem]:
    """Load and merge problems from all configured sources."""
    data_cfg = base_cfg["data"]
    problems: list[Problem] = []

    # APPS — same 0-vs-None semantics as CodeContests and MATH below.
    # 0 means SKIP entirely; None means load all. The released configs set
    # this to 0: APPS is an evaluation-only corpus in the paper and must
    # never enter a training rollout pool.
    max_problems = data_cfg.get("code_max_problems")
    if max_problems == 0:
        logger.info("Skipping APPS (code_max_problems=0)")
    else:
        logger.info("Loading APPS problems ...")
        apps = load_apps(split=data_cfg["code_split"])
        difficulty = data_cfg.get("code_difficulty")
        if difficulty:
            apps = [p for p in apps if p.difficulty == difficulty]
        min_tests = data_cfg.get("code_min_tests", 0)
        apps = [p for p in apps if len(p.test_cases) >= min_tests]
        if max_problems:
            apps = apps[:max_problems]
        logger.info("  APPS after filters: %d problems", len(apps))
        problems.extend(apps)

    # CodeContests — 0 means SKIP entirely; None means load all.
    cc_max = data_cfg.get("cc_max_problems")
    if cc_max == 0:
        logger.info("Skipping CodeContests (cc_max_problems=0)")
    else:
        logger.info("Loading CodeContests problems ...")
        cc = load_codecontests(split=data_cfg.get("cc_split", "valid"))
        cc = [p for p in cc if len(p.test_cases) >= data_cfg.get("code_min_tests", 0)]
        if cc_max:
            cc = cc[:cc_max]
        logger.info("  CodeContests after filters: %d problems", len(cc))
        problems.extend(cc)

    logger.info("Total problems loaded: %d", len(problems))
    return problems


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPPO training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cppo.yaml",
        help="Path to CPPO YAML config",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base.yaml",
        help="Path to base YAML config (for data/model/paths)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help="Optional Problems JSONL (from 00c_export_problems.py). "
             "If given, overrides config-based dataset loading.",
    )
    parser.add_argument(
        "--outcome-log",
        type=str,
        default=None,
        help="Optional JSONL path for per-bundle outcome records (problem_id, "
             "plan_text, pass_prob, j_psi, diagnostic c_psi, outcome_reward, "
             "parseable, branch_pass_rates, branch_passed, step, epoch).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier. If set, checkpoints are written to "
             "{cppo_checkpoint}/{run_id}/ and metrics.jsonl is scoped per run, "
             "so re-runs do not overwrite each other. If not set, defaults "
             "to a UTC timestamp so runs always get their own directory.",
    )
    parser.add_argument(
        "--allow-resume",
        action="store_true",
        help="Allow a dirty run dir after manual inspection. Default is to "
             "fail fast; this flag only bypasses the guard and does not "
             "restore optimizer/checkpoint state.",
    )
    parser.add_argument(
        "--allow-archived-config",
        action="store_true",
        help="Allow archived ablation configs. Default is to reject archived "
        "configs so ablation variants cannot be run by accident.",
    )
    parser.add_argument(
        "--prompt-max-tokens",
        type=int,
        default=None,
        help="Drop rollout problems whose planner prompt exceeds this many "
             "tokens. Long problems are eval/stress cases, not stable CPPO "
             "training samples.",
    )
    args = parser.parse_args()

    # Load configs
    with open(args.config) as f:
        cppo_doc = yaml.safe_load(f)
    assert_not_archived_config(
        config_path=args.config,
        config_doc=cppo_doc,
        allow_archived=args.allow_archived_config,
    )
    cppo_cfg = cppo_doc["cppo"]

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    assert_not_archived_config(
        config_path=args.base_config,
        config_doc=base_cfg,
        allow_archived=args.allow_archived_config,
    )

    # Seed Python + NumPy + Torch (CPU + CUDA) for end-to-end repro
    # parity with 03_run_warmup.py. Without all four,
    # rollout sampling differs between runs at the same --seed.
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device

    # Load problems: --problems JSONL takes precedence over config-driven loading.
    if args.problems:
        from pathlib import Path
        if not Path(args.problems).is_file():
            raise FileNotFoundError(args.problems)
        problems = []
        with open(args.problems) as f:
            for line in f:
                problems.append(Problem.from_dict(json.loads(line)))
        print(f"Loaded {len(problems)} problems from {args.problems}", flush=True)
    else:
        problems = load_problems(base_cfg)

    # Load warmup checkpoint
    warmup_path = base_cfg["paths"]["warmup_checkpoint"]
    rm_path = base_cfg["paths"]["rm_checkpoint"]
    base_save_dir = Path(base_cfg["paths"]["cppo_checkpoint"])
    # Every run gets its own directory so artifacts never pollute each
    # other (lessons #5). Default to UTC timestamp if not explicit.
    from datetime import datetime, timezone
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_dir = base_save_dir / run_id
    # Guard BEFORE mkdir so a contaminated run dir aborts before we
    # even touch the filesystem.
    _assert_run_dir_clean(save_dir=save_dir, allow_resume=args.allow_resume)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("run_id = %s, save_dir = %s", run_id, save_dir)

    # Provenance snapshot BEFORE HF/CUDA spool-up so a write failure
    # fails fast instead of after the 30s model load. Captures CLI
    # args, both config dicts, run_id, timestamp, and git SHA.
    _write_run_provenance(
        save_dir=save_dir,
        args=args,
        base_cfg=base_cfg,
        cppo_cfg=cppo_cfg,
        run_id=run_id,
    )
    logger.info("wrote provenance snapshot to %s", save_dir / "run_config.json")

    # Default outcome log to live inside save_dir so re-runs do not append
    # to the same file (lessons #5). User can still override with
    # --outcome-log to point elsewhere.
    if args.outcome_log is None:
        outcome_log_path = str(save_dir / "outcomes.jsonl")
    else:
        outcome_log_path = args.outcome_log
    logger.info("outcome_log_path = %s", outcome_log_path)

    logger.info("Loading policy model from warmup checkpoint: %s", warmup_path)
    tokenizer = AutoTokenizer.from_pretrained(warmup_path)
    prompt_max_tokens = (
        args.prompt_max_tokens
        if args.prompt_max_tokens is not None
        else cppo_cfg.get("prompt_max_tokens")
        or base_cfg.get("data", {}).get("prompt_max_tokens")
    )
    if prompt_max_tokens is not None:
        prompt_mode = cppo_cfg.get("planner_prompt_mode", "legacy_plan")
        before = len(problems)
        kept: list[Problem] = []
        dropped: list[tuple[str, int]] = []
        for prob in problems:
            prompt_text = build_rollout_planner_prompt(
                prob.prompt,
                domain=prob.domain,
                k=base_cfg["planner"]["k"],
                planner_prompt_mode=prompt_mode,
            )
            chat_text = to_chat_text(
                tokenizer,
                prompt_text,
                enable_thinking=(prompt_mode == "think_plan"),
            )
            n_prompt_tokens = len(tokenizer(chat_text)["input_ids"])
            if n_prompt_tokens <= int(prompt_max_tokens):
                kept.append(prob)
            else:
                dropped.append((prob.id, n_prompt_tokens))
        problems = kept
        logger.info(
            "Prompt-length filter: kept %d/%d problems (max=%s tokens); dropped=%s",
            len(problems),
            before,
            prompt_max_tokens,
            dropped[:10],
        )
        if not problems:
            raise RuntimeError("No problems left after prompt_max_tokens filter.")

    # Load policy weights in fp32. bf16 + AdamW + small lr is a silent
    # no-op (see
    # tests/test_optimizer_dtype.py). The frozen ref_model can stay bf16
    # if memory tightens, but the trainable copy must be fp32.
    policy_model = AutoModelForCausalLM.from_pretrained(
        warmup_path,
        torch_dtype=torch.float32,
    )

    # Create trainer
    trainer = CPPOTrainer(
        base_model=policy_model,
        tokenizer=tokenizer,
        rm_path=rm_path,
        k=base_cfg["planner"]["k"],
        m_tuples=cppo_cfg["m_tuples"],
        lr=cppo_cfg["lr"],
        kl_weight=cppo_cfg["kl_weight"],
        clip_eps=cppo_cfg["clip_eps"],
        plan_temp=cppo_cfg["plan_temperature"],
        solve_temp=cppo_cfg["solve_temperature"],
        max_plan_tokens=base_cfg["planner"]["max_tokens"],
        max_solve_tokens=base_cfg["solver"]["max_tokens"],
        exec_timeout=base_cfg["data"]["exec_timeout"],
        use_pass_rate=cppo_cfg.get("use_pass_rate", True),
        rm_reward_mode=cppo_cfg.get("rm_reward_mode", "binary_jpsi"),
        plan_reward_mode=cppo_cfg.get("plan_reward_mode", "jpsi_times_outcome"),
        rm_winner_min_score=cppo_cfg.get("rm_winner_min_score"),
        rm_jpsi_threshold=cppo_cfg.get("rm_jpsi_threshold"),
        planner_prompt_mode=cppo_cfg.get("planner_prompt_mode", "legacy_plan"),
        gradient_checkpointing=bool(cppo_cfg.get("gradient_checkpointing", True)),
        ref_dtype=cppo_cfg.get("ref_dtype", "bf16"),
        max_train_tokens=cppo_cfg.get("max_train_tokens"),
        param_delta_max_tensors=int(cppo_cfg.get("param_delta_max_tensors", 4)),
        device=device,
        outcome_log_path=outcome_log_path,
    )

    # Training loop
    epochs = cppo_cfg["epochs"]
    batch_size = cppo_cfg["problems_per_batch"]
    log_interval = cppo_cfg["log_interval"]
    save_interval = cppo_cfg["save_interval"]

    logger.info(
        "Starting CPPO training: %d epochs, batch_size=%d, %d problems",
        epochs,
        batch_size,
        len(problems),
    )

    for epoch in range(1, epochs + 1):
        # Sample a batch of problems
        batch = random.sample(problems, min(batch_size, len(problems)))

        metrics = trainer.train_step(batch, step=epoch, epoch=epoch)
        metrics["epoch"] = epoch
        metrics_path = save_dir / "metrics.jsonl"
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if epoch % log_interval == 0:
            logger.info(
                "Epoch %d/%d | loss=%.4f grad_norm=%.4f param_delta=%.4f | "
                "parseable=%.2f cpsi=%.3f outcome=%.2f plan_nz=%.2f | "
                "p_adv_nz=%.2f s_adv_nz=%.2f group_var=%.4f | "
                "any_pass=%.2f zero_pass=%.2f",
                epoch,
                epochs,
                metrics["loss"],
                metrics["grad_norm"],
                metrics["param_delta"],
                metrics["parseable_rate"],
                metrics["avg_cpsi"],
                metrics["outcome_success_rate"],
                metrics["plan_reward_nonzero_rate"],
                metrics["planner_adv_nonzero_rate"],
                metrics["solver_adv_nonzero_rate"],
                metrics["group_reward_variance"],
                metrics["any_pass_rate"],
                metrics["zero_pass_rate"],
            )
            sweep = metrics.get("pass_at_k_sweep") or {}
            if sweep:
                logger.info(
                    "  pass@k (branches pooled across tuples): %s",
                    "  ".join(f"@{kk}={v:.4f}" for kk, v in sweep.items()),
                )

        if epoch % save_interval == 0:
            ckpt_path = save_dir / f"epoch_{epoch}"
            logger.info("Saving checkpoint to %s", ckpt_path)
            trainer.policy_model.save_pretrained(ckpt_path)
            trainer.tokenizer.save_pretrained(ckpt_path)

            # Save metrics history
            # metrics.jsonl is flushed every epoch so OOM/crash still leaves
            # the training curve up to the failure point.

    # Final save: write directly to cppo_checkpoint root so 07/08 can load it
    logger.info("Saving final model to %s", save_dir)
    trainer.policy_model.save_pretrained(save_dir)
    trainer.tokenizer.save_pretrained(save_dir)

    # Maintain a `latest` symlink only after the run has produced a usable
    # final checkpoint. Failed launches must not steal `latest` from the last
    # successful run.
    _update_latest_symlink(base_save_dir, run_id)

    logger.info("CPPO training complete.")


if __name__ == "__main__":
    main()
