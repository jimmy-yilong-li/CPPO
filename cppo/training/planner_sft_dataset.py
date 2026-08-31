"""Dataset helpers for positive-only planner SFT."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cppo.data.prompts import make_plan_prompt, to_chat_text


@dataclass(frozen=True)
class PlannerSFTRecord:
    problem_id: str
    domain: str
    problem_text: str
    target_text: str
    source_family: str
    row_id: str


def make_planner_think_prompt(problem: str, *, domain: str = "code", k: int = 4) -> str:
    """Prompt used when training/evaluating a think+plan planner.

    The target is expected to contain a bounded ``<think>`` strategy-selection
    trace followed by exactly ``k`` labelled plan methods.
    """
    task = "programming problem"
    labels = "\n".join(f"{chr(ord('A') + i)}:" for i in range(k))
    return (
        "You are a competitive-programming planner.\n\n"
        f"## Problem\n{problem}\n\n"
        "Think briefly about the problem type, constraints, and how to allocate "
        f"{k} solver attempts to maximize pass@4, then output a high-level plan tuple.\n\n"
        "The thinking trace must be a strategy-selection trace, not a full "
        "solution. Do not write code, pseudocode, imports, implementation "
        "details, exact decision rules, boundary-case walkthroughs, or final answers.\n\n"
        f"After the thinking trace, output exactly {k} self-contained, "
        f"problem-specific solver attempts for this {task}, using this "
        "plain label skeleton:\n"
        f"{labels}\n\n"
        "All methods must be independently viable and solver-actionable. "
        "Prefer genuinely different strategies when they help, but useful "
        "variants are allowed if they give the solver different implementation "
        "paths, robustness checks, complexity tradeoffs, or failure modes. "
        "Do not include filler methods, harmful duplicates, cosmetic rewrites, "
        "generic backup ideas, sequential workflow steps, or 2 good + 2 useless filler methods. "
        "Each method must include a concrete hook from the problem, such as a "
        "state, invariant, sorted key, graph representation, search dimension, "
        "constraint reason, formula structure, transformation, or validation path."
    )


def make_planner_plan_only_prompt(problem: str, *, domain: str = "code", k: int = 4) -> str:
    """Prompt used when training/evaluating a plan-only planner."""
    return make_plan_prompt(problem, domain=domain, k=k)


def load_sft_records(
    path: str | Path,
    *,
    target_field: str = "target_think_plan",
) -> list[PlannerSFTRecord]:
    records: list[PlannerSFTRecord] = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            target = str(row.get(target_field) or "").strip()
            problem = str(row.get("problem_text") or "").strip()
            if not problem or not target:
                raise ValueError(f"{path}:{line_no} missing problem_text or {target_field}")
            records.append(
                PlannerSFTRecord(
                    problem_id=str(row.get("problem_id") or f"row_{line_no}"),
                    domain=str(row.get("domain") or "code"),
                    problem_text=problem,
                    target_text=target,
                    source_family=str(row.get("sft_source_family") or row.get("source") or "unknown"),
                    row_id=str(row.get("sft_row_id") or f"row_{line_no}"),
                )
            )
    return records


def split_records_by_problem(
    records: list[PlannerSFTRecord],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[PlannerSFTRecord], list[PlannerSFTRecord]]:
    """Split records by problem id to avoid train/eval leakage."""
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0, 1)")
    by_problem: dict[str, list[PlannerSFTRecord]] = {}
    for record in records:
        by_problem.setdefault(record.problem_id, []).append(record)

    problem_ids = list(by_problem)
    random.Random(seed).shuffle(problem_ids)
    n_eval_problems = int(round(len(problem_ids) * eval_fraction))
    eval_problem_ids = set(problem_ids[:n_eval_problems])

    train_records: list[PlannerSFTRecord] = []
    eval_records: list[PlannerSFTRecord] = []
    for problem_id in problem_ids:
        if problem_id in eval_problem_ids:
            eval_records.extend(by_problem[problem_id])
        else:
            train_records.extend(by_problem[problem_id])
    if not train_records:
        raise ValueError("empty train split")
    return train_records, eval_records


def _target_segment_ids(tokenizer: Any, target_text: str, target_ids: list[int]) -> list[int]:
    """Return per-target-token segment IDs: 0=thinking, 1=plan.

    Plan-only targets are all plan tokens. Think+plan targets use the first
    A: label as the plan boundary. This keeps metric computation independent
    from the model and avoids changing the actual loss mask.
    """
    match = re.search(r"(?m)^[ \t]*(?:[#>\-\*]+[ \t]*)?\*{0,2}A\*{0,2}[ \t]*[:.)]", target_text)
    if not match:
        return [1] * len(target_ids)

    prefix_ids = tokenizer(target_text[: match.start()], add_special_tokens=False)["input_ids"]
    plan_start = min(len(prefix_ids), len(target_ids))
    return [0] * plan_start + [1] * (len(target_ids) - plan_start)


def encode_sft_record(
    tokenizer: Any,
    record: PlannerSFTRecord,
    *,
    max_length: int,
    k: int = 4,
    prompt_mode: str = "think_plan",
) -> dict[str, list[int] | str | bool]:
    if prompt_mode == "think_plan":
        user_prompt = make_planner_think_prompt(record.problem_text, domain=record.domain, k=k)
    elif prompt_mode == "plan_only":
        user_prompt = make_planner_plan_only_prompt(record.problem_text, domain=record.domain, k=k)
    else:
        raise ValueError(f"unknown prompt_mode: {prompt_mode}")
    prompt_text = to_chat_text(tokenizer, user_prompt)
    target_text = record.target_text
    if tokenizer.eos_token and not target_text.endswith(tokenizer.eos_token):
        target_text = target_text + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    target_segment_ids = _target_segment_ids(tokenizer, target_text, target_ids)
    if len(prompt_ids) >= max_length:
        raise ValueError(f"prompt for {record.problem_id} exceeds max_length={max_length}")

    available_target = max_length - len(prompt_ids)
    truncated = len(target_ids) > available_target
    if truncated:
        target_ids = target_ids[:available_target]
        target_segment_ids = target_segment_ids[:available_target]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    segment_ids = [-100] * len(prompt_ids) + target_segment_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "segment_ids": segment_ids,
        "problem_id": record.problem_id,
        "row_id": record.row_id,
        "source_family": record.source_family,
        "truncated": truncated,
    }


class PlannerSFTDataset:
    def __init__(
        self,
        records: list[PlannerSFTRecord],
        tokenizer: Any,
        *,
        max_length: int,
        k: int = 4,
        prompt_mode: str = "think_plan",
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k = k
        self.prompt_mode = prompt_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, list[int] | str | bool]:
        return encode_sft_record(
            self.tokenizer,
            self.records[idx],
            max_length=self.max_length,
            k=self.k,
            prompt_mode=self.prompt_mode,
        )


class PlannerSFTCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        batch_input_ids = []
        batch_attention = []
        batch_labels = []
        batch_segments = []
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [pad_id] * pad)
            batch_attention.append(f["attention_mask"] + [0] * pad)
            batch_labels.append(f["labels"] + [-100] * pad)
            segment_ids = f.get("segment_ids")
            if segment_ids is None:
                segment_ids = [1 if label != -100 else -100 for label in f["labels"]]
            batch_segments.append(segment_ids + [-100] * pad)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "segment_ids": torch.tensor(batch_segments, dtype=torch.long),
        }
