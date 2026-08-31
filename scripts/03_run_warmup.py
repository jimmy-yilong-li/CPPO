#!/usr/bin/env python3
"""Run planner warm-up training with early stopping.

Usage:
    python scripts/03_run_warmup.py \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --rm-path checkpoints/rm \
        --problems data/problems.jsonl \
        --output checkpoints/warmup \
        --config configs/warmup.yaml \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import yaml

from cppo.data.problem import Problem
from cppo.training.config_guard import assert_not_archived_config
from cppo.training.run_provenance import (
    assert_run_dir_clean,
    write_run_provenance,
)
from cppo.training.warmup_trainer import PlannerWarmupTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_problems(path: str) -> list[Problem]:
    """Load problems from a JSONL file."""
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(Problem.from_dict(json.loads(line)))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Planner warm-up training")
    parser.add_argument(
        "--base-model", required=True, help="Base model name or path"
    )
    parser.add_argument(
        "--rm-path", required=True, help="Path to trained reward model"
    )
    parser.add_argument(
        "--problems", required=True, help="Path to problems JSONL"
    )
    parser.add_argument(
        "--output", default="checkpoints/warmup", help="Output dir"
    )
    parser.add_argument(
        "--config",
        default="configs/warmup.yaml",
        help="Config YAML path",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-resume",
        action="store_true",
        help="Allow a dirty output dir after manual inspection. Default is "
             "to fail fast; this flag only bypasses the guard and does not "
             "restore optimizer/checkpoint state.",
    )
    parser.add_argument(
        "--allow-archived-config",
        action="store_true",
        help="Allow an archived ablation config. Default is to reject archived "
             "configs so pre-CPPO_new variants cannot be run by accident.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config_doc = yaml.safe_load(f)
    assert_not_archived_config(
        config_path=args.config,
        config_doc=config_doc,
        allow_archived=args.allow_archived_config,
    )
    cfg = config_doc["warmup"]

    # Fail fast if --output already contains prior artifacts. Validate the
    # problems file before writing run markers so a bad --problems path does
    # not pollute the output dir for the corrected retry. Provenance is still
    # written before the long model load.
    from datetime import datetime, timezone
    output_dir = Path(args.output)
    assert_run_dir_clean(save_dir=output_dir, allow_resume=args.allow_resume)

    # Load problems
    logger.info(f"Loading problems from {args.problems}")
    problems = load_problems(args.problems)
    logger.info(f"Loaded {len(problems)} problems")

    warmup_run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    write_run_provenance(
        save_dir=output_dir,
        args=args,
        run_id=warmup_run_id,
        warmup_cfg=cfg,
    )
    logger.info(
        "wrote provenance snapshot to %s (run_id=%s)",
        output_dir / "run_config.json", warmup_run_id,
    )

    # Set seed across Python, NumPy, and Torch (CPU + CUDA). Without
    # the torch seeds, warmup generation samples differently every run
    # even at the same --seed, defeating reproducibility for downstream
    # CPPO and audit.
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create trainer
    trainer = PlannerWarmupTrainer(
        base_model=args.base_model,
        rm_path=args.rm_path,
        k=4,
        m_tuples=cfg["m_tuples"],
        lr=cfg["lr"],
        kl_weight=cfg["kl_weight"],
        clip_eps=cfg["clip_eps"],
        temperature=cfg["temperature"],
        max_new_tokens=cfg.get("max_new_tokens", 512),
        rm_reward_mode=cfg.get("rm_reward_mode", "binary_jpsi"),
        rm_winner_min_score=cfg.get("rm_winner_min_score"),
        rm_jpsi_threshold=float(cfg.get("rm_jpsi_threshold", 0.5)),
        prompt_mode=cfg.get("prompt_mode", "plan_only"),
        device=args.device,
    )

    epochs = cfg["epochs"]
    problems_per_batch = cfg["problems_per_batch"]
    stop_patience = cfg["stop_patience"]
    min_valid_rate = cfg["min_valid_rate"]
    selection_metric = cfg.get("selection_metric", "rm_valid_rate")

    # Training loop with early stopping
    best_metric_value = float("-inf")
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        # Sample a batch of problems
        batch = random.sample(
            problems, min(problems_per_batch, len(problems))
        )

        metrics = trainer.train_step(batch)
        history.append({"epoch": epoch, **metrics})

        logger.info(
            f"Epoch {epoch}/{epochs}: "
            f"loss={metrics['loss']:.4f}, "
            f"avg_cpsi={metrics['avg_cpsi']:.4f}, "
            f"parseable_rate={metrics['parseable_rate']:.4f}, "
            f"rm_valid_rate={metrics['rm_valid_rate']:.4f}, "
            f"winner_rate={metrics.get('winner_rate', 0.0):.4f}, "
            f"groups_with_winner={metrics.get('groups_with_winner_rate', 0.0):.4f}"
        )

        if selection_metric not in metrics:
            raise RuntimeError(
                f"Configured warmup selection_metric={selection_metric!r} "
                f"not found in metrics keys={sorted(metrics.keys())}"
            )

        metric_value = metrics[selection_metric]
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            patience_counter = 0
            # Save best checkpoint
            trainer.save(args.output)
            logger.info(
                f"New best {selection_metric}={best_metric_value:.4f}, "
                f"saved to {args.output}"
            )
        else:
            patience_counter += 1

        if (
            patience_counter >= stop_patience
            and best_metric_value >= min_valid_rate
        ):
            logger.info(
                f"Early stopping: no improvement for {stop_patience} epochs, "
                f"best {selection_metric}={best_metric_value:.4f} >= "
                f"min_valid_rate={min_valid_rate}"
            )
            break

    # Save training history WITH reproducibility metadata so a future
    # reader can pin the exact (seed, config, model, RM, problems) that
    # produced this checkpoint.
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "warmup_history.json"
    history_payload = {
        "meta": {
            "seed": args.seed,
            "config_path": args.config,
            "config_values": cfg,
            "selection_metric": selection_metric,
            "base_model": args.base_model,
            "rm_path": args.rm_path,
            "problems_path": args.problems,
            "n_problems": len(problems),
        },
        "history": history,
    }
    with open(history_path, "w") as f:
        json.dump(history_payload, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    # Final save if we haven't saved yet (best_valid_rate was never updated)
    if best_metric_value == float("-inf"):
        trainer.save(args.output)
        logger.info(f"Final model saved to {args.output}")

    logger.info(
        f"Warm-up complete. Best {selection_metric}={best_metric_value:.4f}"
    )


if __name__ == "__main__":
    main()
