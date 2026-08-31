#!/usr/bin/env python3
"""Train a positive-only planner SFT checkpoint.

The default target is ``target_think_plan``:

    <think>strategy-selection trace</think>

    A: ...
    B: ...
    C: ...
    D: ...

Only target tokens receive loss. Prompt/chat-template tokens are masked.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from cppo.training.planner_sft_dataset import (
    PlannerSFTCollator,
    PlannerSFTDataset,
    PlannerSFTRecord,
    load_sft_records,
    split_records_by_problem,
)
from cppo.training.planner_sft_metrics import (
    empty_error_counts,
    finalize_error_counts,
    merge_error_counts,
    token_error_counts,
)

logger = logging.getLogger("planner_sft")


def _is_main_process() -> bool:
    return os.environ.get("RANK", "0") in {"0", "-1"}


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _model_forward_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Strip metric-only fields before passing a batch to the model."""
    model_inputs = dict(inputs)
    model_inputs.pop("segment_ids", None)
    return model_inputs


def _resolve_prompt_mode(target_field: str, prompt_mode: str) -> str:
    if prompt_mode != "auto":
        return prompt_mode
    if target_field == "target_plan_only" or target_field.endswith("plan_only"):
        return "plan_only"
    return "think_plan"


def _weighted_segment_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    segment_ids: torch.Tensor | None,
    *,
    thinking_loss_weight: float,
) -> torch.Tensor:
    """Causal-LM loss with a lower weight on target thinking segment loss.

    ``segment_ids`` follows the SFT dataset convention:
    - -100: prompt/padding token, ignored
    - 0: visible strategy-selection trace token
    - 1: final plan token

    When the thinking weight is below 1.0, the segment losses are averaged
    separately, then combined. This prevents a long visible trace from
    dominating the final plan merely by token count. Weight 1.0 preserves the
    ordinary target-token CE loss.
    """
    if logits.size(1) < 2:
        return logits.new_zeros(())
    # Gemma4 can emit bf16 logits with large dynamic range; computing CE
    # directly in bf16 produced NaN gradients during full-parameter SFT.
    # Cast only the loss path to fp32 while keeping model activations bf16.
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    active = shift_labels.ne(-100)
    if thinking_loss_weight == 1.0:
        return token_loss[active].mean() if active.any() else token_loss.sum()
    if segment_ids is None:
        return token_loss[active].mean() if active.any() else token_loss.sum()

    shift_segments = segment_ids[:, 1:].to(device=token_loss.device)
    think_mask = active & shift_segments.eq(0)
    plan_mask = active & shift_segments.eq(1)
    if think_mask.any() and plan_mask.any():
        think_loss = token_loss[think_mask].mean()
        plan_loss = token_loss[plan_mask].mean()
        return float(thinking_loss_weight) * think_loss + plan_loss
    if plan_mask.any():
        return token_loss[plan_mask].mean()
    if think_mask.any():
        return float(thinking_loss_weight) * token_loss[think_mask].mean()
    return token_loss.sum()


class PlannerSFTTrainer(Trainer):
    """Trainer that keeps segment IDs for metrics but not model.forward."""

    def __init__(self, *args: Any, thinking_loss_weight: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if thinking_loss_weight < 0:
            raise ValueError("thinking_loss_weight must be non-negative")
        self.thinking_loss_weight = float(thinking_loss_weight)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        model_inputs = _model_forward_inputs(inputs)
        labels = model_inputs.pop("labels", None)
        segment_ids = inputs.get("segment_ids")
        if labels is None:
            signature = inspect.signature(super().compute_loss)
            accepted_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
            return super().compute_loss(
                model,
                model_inputs,
                return_outputs=return_outputs,
                **accepted_kwargs,
            )

        outputs = model(**model_inputs)
        loss = _weighted_segment_loss(
            outputs.logits,
            labels,
            segment_ids,
            thinking_loss_weight=self.thinking_loss_weight,
        )
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        return super().prediction_step(
            model,
            _model_forward_inputs(inputs),
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, *, default: bool, help: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", help=help)
    group.add_argument(
        f"--no-{name}",
        dest=name.replace("-", "_"),
        action="store_false",
        help=f"Disable: {help}",
    )
    parser.set_defaults(**{name.replace("-", "_"): default})


def _training_args_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": args.output,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size or args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": args.seed,
        "data_seed": args.seed,
    }
    if args.ddp_find_unused_parameters is not None:
        kwargs["ddp_find_unused_parameters"] = args.ddp_find_unused_parameters
    if "save_safetensors" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["save_safetensors"] = args.save_safetensors
    elif not args.save_safetensors:
        logger.warning(
            "--no-save-safetensors was requested, but this transformers version "
            "does not support TrainingArguments.save_safetensors; ignoring it."
        )
    if args.eval_fraction > 0:
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
            kwargs["eval_strategy"] = args.eval_strategy
        else:
            kwargs["evaluation_strategy"] = args.eval_strategy
    return kwargs


def _source_counts(records: list[PlannerSFTRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.source_family] = counts.get(record.source_family, 0) + 1
    return dict(sorted(counts.items()))


def _problem_count(records: list[PlannerSFTRecord]) -> int:
    return len({r.problem_id for r in records})


def _dataset_stats(dataset: PlannerSFTDataset) -> dict[str, int]:
    truncated = 0
    max_len = 0
    for idx in range(len(dataset)):
        item = dataset[idx]
        truncated += int(bool(item["truncated"]))
        max_len = max(max_len, len(item["input_ids"]))
    return {"truncated_records": truncated, "max_encoded_length": max_len}


def _token_error_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
) -> dict[str, dict[str, int | float | None]]:
    return token_error_counts(logits, labels, segment_ids)


@torch.no_grad()
def _evaluate_token_error(
    model: torch.nn.Module,
    dataset: PlannerSFTDataset,
    collator: PlannerSFTCollator,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, dict[str, int | float | None]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    counts = empty_error_counts()
    was_training = model.training
    model.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        segment_ids = batch["segment_ids"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        merge_error_counts(counts, token_error_counts(outputs.logits, labels, segment_ids))
    if was_training:
        model.train()
    return finalize_error_counts(counts)


def _write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


class TokenErrorLoggingCallback(TrainerCallback):
    """Write train/eval token-error metrics at each Trainer eval point."""

    def __init__(
        self,
        *,
        train_dataset: PlannerSFTDataset,
        eval_dataset: PlannerSFTDataset | None,
        collator: PlannerSFTCollator,
        batch_size: int,
        output_dir: str | Path,
        max_batches: int | None,
    ) -> None:
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collator = collator
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.max_batches = max_batches

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if not getattr(state, "is_world_process_zero", True):
            return control
        model = kwargs.get("model")
        if model is None:
            return control

        eval_model = getattr(model, "module", model)
        device = next(eval_model.parameters()).device
        payload = {
            "step": int(getattr(state, "global_step", 0) or 0),
            "epoch": getattr(state, "epoch", None),
            "train": _evaluate_token_error(
                eval_model,
                self.train_dataset,
                self.collator,
                batch_size=self.batch_size,
                device=device,
                max_batches=self.max_batches,
            ),
            "eval": (
                _evaluate_token_error(
                    eval_model,
                    self.eval_dataset,
                    self.collator,
                    batch_size=self.batch_size,
                    device=device,
                    max_batches=self.max_batches,
                )
                if self.eval_dataset is not None
                else None
            ),
        }
        path = self.output_dir / "token_error_history.json"
        if path.exists():
            with path.open() as f:
                history = json.load(f)
        else:
            history = []
        history.append(payload)
        _write_json(path, history)
        logger.info("Token error at eval step %s: %s", payload["step"], payload)
        return control


def _maybe_apply_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if args.lora_r <= 0:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("peft is required for --lora-r > 0") from exc

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _resolve_model_torch_dtype(args: argparse.Namespace) -> torch.dtype | str:
    if args.model_dtype == "fp32":
        return torch.float32
    if args.model_dtype == "bf16":
        return torch.bfloat16
    if args.bf16:
        return torch.bfloat16
    return "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train planner SFT on positive think+plan data")
    parser.add_argument("--data", required=True, help="Training JSONL with target_think_plan")
    parser.add_argument("--output", required=True, help="Output checkpoint directory")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B", help="Base planner model")
    parser.add_argument("--target-field", default="target_think_plan")
    parser.add_argument("--prompt-mode", default="auto", choices=["auto", "think_plan", "plan_only"])
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--thinking-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Loss weight for target tokens before the first A: label. "
            "Use 0.3 or lower to keep Think+Plan while making final plan tokens dominate."
        ),
    )
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-eval-records", type=int, default=None)
    parser.add_argument("--token-error-max-batches", type=int, default=None)
    parser.add_argument(
        "--model-dtype",
        default="auto",
        choices=["auto", "bf16", "fp32"],
        help=(
            "Model loading dtype. 'auto' preserves historical behavior "
            "(bf16 when --bf16, otherwise HF auto); fp32 is useful for "
            "models whose bf16 full backward is unstable."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    _add_bool_arg(parser, "bf16", default=True, help="Use bf16 training")
    _add_bool_arg(parser, "fp16", default=False, help="Use fp16 training")
    _add_bool_arg(parser, "gradient-checkpointing", default=True, help="Use gradient checkpointing")
    parser.add_argument(
        "--ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_true",
        default=None,
        help="Enable DDP unused-parameter detection for models with inactive text-only branches.",
    )
    parser.add_argument(
        "--no-ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_false",
        help="Disable DDP unused-parameter detection.",
    )
    _add_bool_arg(parser, "save-safetensors", default=True, help="Save checkpoints in safetensors format")
    _add_bool_arg(parser, "compute-token-error", default=True, help="Compute train/eval token error after training")
    _add_bool_arg(parser, "trust-remote-code", default=True, help="Pass trust_remote_code to HF loaders")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SFT records from %s", args.data)
    records = load_sft_records(args.data, target_field=args.target_field)
    prompt_mode = _resolve_prompt_mode(args.target_field, args.prompt_mode)
    train_records, eval_records = split_records_by_problem(
        records,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    if args.max_train_records is not None:
        train_records = train_records[: args.max_train_records]
    if args.max_eval_records is not None:
        eval_records = eval_records[: args.max_eval_records]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = PlannerSFTDataset(
        train_records,
        tokenizer,
        max_length=args.max_length,
        prompt_mode=prompt_mode,
    )
    eval_dataset = (
        PlannerSFTDataset(
            eval_records,
            tokenizer,
            max_length=args.max_length,
            prompt_mode=prompt_mode,
        )
        if eval_records and args.eval_fraction > 0
        else None
    )
    train_stats = _dataset_stats(train_dataset)
    eval_stats = _dataset_stats(eval_dataset) if eval_dataset is not None else {}
    logger.info("Train records: %d across %d problems", len(train_records), _problem_count(train_records))
    logger.info("Eval records: %d across %d problems", len(eval_records), _problem_count(eval_records))
    logger.info("Train source counts: %s", _source_counts(train_records))
    logger.info("Eval source counts: %s", _source_counts(eval_records))
    logger.info("Encoding stats train=%s eval=%s", train_stats, eval_stats)

    run_config = vars(args) | {
        "started_at_unix": time.time(),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "train_problems": _problem_count(train_records),
        "eval_problems": _problem_count(eval_records),
        "train_source_counts": _source_counts(train_records),
        "eval_source_counts": _source_counts(eval_records),
        "train_encoding_stats": train_stats,
        "eval_encoding_stats": eval_stats,
        "resolved_prompt_mode": prompt_mode,
    }
    if _is_main_process():
        _write_json(output / "run_config.json", run_config)
        _write_json(
            output / "sft_data_manifest.json",
            {
                "data": str(Path(args.data).resolve()),
                "target_field": args.target_field,
                "prompt_mode": prompt_mode,
                "max_length": args.max_length,
                "train_records": [asdict(r) for r in train_records],
                "eval_records": [asdict(r) for r in eval_records],
            },
        )

    logger.info("Loading model %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_resolve_model_torch_dtype(args),
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model = _maybe_apply_lora(model, args)

    trainer = PlannerSFTTrainer(
        model=model,
        args=TrainingArguments(**_training_args_kwargs(args)),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PlannerSFTCollator(tokenizer),
        thinking_loss_weight=args.thinking_loss_weight,
    )
    if args.compute_token_error and eval_dataset is not None and args.eval_strategy != "no":
        trainer.add_callback(
            TokenErrorLoggingCallback(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                collator=PlannerSFTCollator(tokenizer),
                batch_size=args.eval_batch_size or args.batch_size,
                output_dir=output,
                max_batches=args.token_error_max_batches,
            )
        )

    logger.info("Starting training")
    train_output = trainer.train()
    token_error_metrics = None
    if args.compute_token_error and _world_size() > 1:
        if trainer.is_world_process_zero():
            logger.warning(
                "Skipping post-train token error metrics in distributed mode; "
                "evaluate a saved checkpoint with scripts/05_eval_pass_at_k.py."
            )
    elif args.compute_token_error and trainer.is_world_process_zero():
        device = next(trainer.model.parameters()).device
        collator = PlannerSFTCollator(tokenizer)
        logger.info("Computing post-train token error metrics")
        token_error_metrics = {
            "train": _evaluate_token_error(
                trainer.model,
                train_dataset,
                collator,
                batch_size=args.eval_batch_size or args.batch_size,
                device=device,
                max_batches=args.token_error_max_batches,
            ),
            "eval": (
                _evaluate_token_error(
                    trainer.model,
                    eval_dataset,
                    collator,
                    batch_size=args.eval_batch_size or args.batch_size,
                    device=device,
                    max_batches=args.token_error_max_batches,
                )
                if eval_dataset is not None
                else None
            ),
        }
    trainer.save_model(output)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output)
        if trainer.state.log_history:
            _write_json(output / "sft_training_history.json", trainer.state.log_history)
        _write_json(output / "train_metrics.json", train_output.metrics)
        if token_error_metrics is not None:
            _write_json(output / "token_error_metrics.json", token_error_metrics)
    logger.info("Done. Metrics: %s", train_output.metrics)


if __name__ == "__main__":
    main()
