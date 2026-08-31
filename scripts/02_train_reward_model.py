#!/usr/bin/env python3
"""Train a reward model on labeled plan-validity data.

Usage:
    python scripts/02_train_reward_model.py \
        --data data/labeled_plans.jsonl \
        --model gpt2 \
        --output checkpoints/rm \
        --epochs 3 \
        --lr 1e-5 \
        --batch-size 8 \
        --max-length 512 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from cppo.reward_model.dataset import (
    balance_rm_records,
    build_rm_splits,
    filter_rm_records,
    load_labeled_records,
    resolve_rm_data_view,
)
from cppo.reward_model.trainer import RewardModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class PreparedRMRecords:
    train_records: list[dict]
    val_records: list[dict]
    train_balance_stats: dict | None
    val_balance_stats: dict | None
    source_filter_stats: dict | None
    effective_source_policy: str
    pos_weight_override: float | None


def prepare_rm_training_records(
    records: list[dict],
    *,
    val_fraction: float,
    seed: int,
    rm_data_mode: str = "legacy",
    source_policy: str = "all",
    neg_per_pos: int = 1,
) -> PreparedRMRecords:
    """Split records and optionally build a balanced 1:1 RM training view."""
    balance_mode, source_policy = resolve_rm_data_view(rm_data_mode, source_policy)

    records, source_filter_stats = filter_rm_records(
        records, source_policy=source_policy
    )
    train_records, val_records = build_rm_splits(
        records, val_fraction=val_fraction, seed=seed
    )

    if rm_data_mode == "legacy":
        return PreparedRMRecords(
            train_records=train_records,
            val_records=val_records,
            train_balance_stats=None,
            val_balance_stats=None,
            source_filter_stats=source_filter_stats,
            effective_source_policy=source_policy,
            pos_weight_override=None,
        )

    if balance_mode not in {"global_1to1", "by_problem_1to1"}:
        raise ValueError(f"Unknown RM data mode: {rm_data_mode}")

    train_records, train_stats = balance_rm_records(
        train_records, mode=balance_mode, seed=seed, neg_per_pos=neg_per_pos
    )
    val_records, val_stats = balance_rm_records(
        val_records, mode=balance_mode, seed=seed + 1, neg_per_pos=neg_per_pos
    )
    train_stats["requested_data_mode"] = rm_data_mode
    val_stats["requested_data_mode"] = rm_data_mode
    return PreparedRMRecords(
        train_records=train_records,
        val_records=val_records,
        train_balance_stats=train_stats,
        val_balance_stats=val_stats,
        source_filter_stats=source_filter_stats,
        effective_source_policy=source_policy,
        pos_weight_override=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional RM YAML config with top-level `rm:` section. CLI flags override it.",
    )
    parser.add_argument("--data", default=None, help="Path to labeled JSONL")
    parser.add_argument("--model", default="gpt2", help="Base model name")
    parser.add_argument("--output", default="checkpoints/rm", help="Output dir")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument(
        "--selection-metric",
        default="average_precision",
        choices=[
            "average_precision",
            "val_average_precision",
            "ap",
            "auc",
            "val_auc",
            "accuracy",
            "val_accuracy",
            "balanced_accuracy",
            "val_balanced_accuracy",
            "error_rate",
            "val_error",
        ],
        help="Validation metric used to select the saved checkpoint.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rm-data-mode",
        choices=[
            "legacy",
            "global_1to1",
            "by_problem_1to1",
            "model_only_1to1",
            "paper_binary_balanced",
        ],
        default="legacy",
        help="RM data sampling mode. Balanced modes produce effective 1:1 labels.",
    )
    parser.add_argument(
        "--source-policy",
        choices=["all", "model_only", "non_corruption"],
        default="all",
        help="Which labeled sources enter the RM training view.",
    )
    parser.add_argument(
        "--neg-per-pos",
        type=int,
        default=1,
        help="Negative samples per positive in balanced RM modes. Only 1 is supported.",
    )
    parser.add_argument(
        "--canonicalized-input",
        action="store_true",
        help="Write canonicalized_input=true into RM metadata. Use only when plan_text is canonicalized.",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
        rm_cfg = config.get("rm", config)
        parser_defaults = {
            action.dest: action.default
            for action in parser._actions
            if action.dest != "help"
        }

        def apply_config(attr: str, key: str) -> None:
            if key not in rm_cfg:
                return
            if getattr(args, attr) == parser_defaults.get(attr):
                setattr(args, attr, rm_cfg[key])

        apply_config("data", "data")
        apply_config("model", "model")
        apply_config("output", "output")
        apply_config("epochs", "epochs")
        apply_config("lr", "lr")
        apply_config("batch_size", "batch_size")
        apply_config("max_length", "max_length")
        apply_config("ranking_weight", "ranking_weight")
        apply_config("selection_metric", "selection_metric")
        apply_config("val_fraction", "val_fraction")
        apply_config("device", "device")
        apply_config("seed", "seed")
        apply_config("rm_data_mode", "data_mode")
        apply_config("source_policy", "source_policy")
        apply_config("neg_per_pos", "neg_per_pos")
        apply_config("canonicalized_input", "canonicalized_input")

    if args.data is None:
        parser.error("--data is required unless provided by --config")

    if args.rm_data_mode == "paper_binary_balanced":
        args.ranking_weight = 0.0

    # Load and split data
    logger.info(f"Loading labeled records from {args.data}")
    records = load_labeled_records(args.data)
    logger.info(f"Loaded {len(records)} labeled records")

    # Quality gate: enough samples + both classes present
    if len(records) < 100:
        raise RuntimeError(
            f"Only {len(records)} usable records — too few to train an RM. "
            f"Need at least 100."
        )
    pos = sum(1 for r in records if r["valid"])
    neg = len(records) - pos
    logger.info(f"  Positives (valid): {pos}, Negatives (invalid): {neg}")
    if pos < 30 or neg < 30:
        raise RuntimeError(
            f"Class imbalance too severe: pos={pos}, neg={neg}. "
            f"Need >=30 of each."
        )

    prepared = prepare_rm_training_records(
        records,
        val_fraction=args.val_fraction,
        seed=args.seed,
        rm_data_mode=args.rm_data_mode,
        source_policy=args.source_policy,
        neg_per_pos=args.neg_per_pos,
    )
    train_records = prepared.train_records
    val_records = prepared.val_records

    # Both splits must contain both classes.
    train_pos = sum(1 for r in train_records if r["valid"])
    val_pos = sum(1 for r in val_records if r["valid"])
    train_neg = len(train_records) - train_pos
    val_neg = len(val_records) - val_pos
    logger.info(
        f"Split: train={len(train_records)} ({train_pos}+/{train_neg}-), "
        f"val={len(val_records)} ({val_pos}+/{val_neg}-) "
        f"(no problem leaks)"
    )
    if prepared.train_balance_stats is not None:
        logger.info(f"Train balance stats: {prepared.train_balance_stats}")
        logger.info(f"Val balance stats: {prepared.val_balance_stats}")
        if train_pos != train_neg or val_pos != val_neg:
            raise RuntimeError(
                f"Balanced RM mode failed to produce 1:1 classes. "
                f"train={train_pos}+/{train_neg}-, val={val_pos}+/{val_neg}-."
            )
        if train_pos < 30 or val_pos < 5:
            raise RuntimeError(
                f"Balanced RM splits too small. train={train_pos}+/{train_neg}-, "
                f"val={val_pos}+/{val_neg}-. Increase data."
            )
    if train_pos < 10 or train_neg < 10 or val_pos < 5 or val_neg < 5:
        raise RuntimeError(
            f"Per-split class counts too small. train: {train_pos}+/{train_neg}-, "
            f"val: {val_pos}+/{val_neg}-. Increase data or rebalance the split."
        )

    # Train
    trainer = RewardModelTrainer(
        model_name=args.model,
        lr=args.lr,
        ranking_weight=args.ranking_weight,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        canonicalized_input=args.canonicalized_input,
    )

    history = trainer.train(
        train_records,
        val_records,
        epochs=args.epochs,
        pos_weight_override=prepared.pos_weight_override,
        selection_metric=args.selection_metric,
        extra_history={
            "rm_config": args.config,
            "rm_data_mode": args.rm_data_mode,
            "selection_metric": args.selection_metric,
            "source_policy": prepared.effective_source_policy,
            "source_filter_stats": prepared.source_filter_stats,
            "train_balance_stats": prepared.train_balance_stats,
            "val_balance_stats": prepared.val_balance_stats,
        },
    )

    # Save
    trainer.save(args.output)
    logger.info(f"Model saved to {args.output}")

    manifest = {
        "input_data": args.data,
        "rm_config": args.config,
        "rm_data_mode": args.rm_data_mode,
        "source_policy": prepared.effective_source_policy,
        "source_filter_stats": prepared.source_filter_stats,
        "train_balance_stats": prepared.train_balance_stats,
        "val_balance_stats": prepared.val_balance_stats,
        "train_pos": train_pos,
        "train_neg": train_neg,
        "val_pos": val_pos,
        "val_neg": val_neg,
        "pos_weight_override": prepared.pos_weight_override,
        "ranking_weight": args.ranking_weight,
        "selection_metric": args.selection_metric,
        "canonicalized_input": args.canonicalized_input,
    }
    manifest_path = Path(args.output) / "rm_data_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"RM data manifest saved to {manifest_path}")

    # Save training history
    history_path = f"{args.output}/training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
