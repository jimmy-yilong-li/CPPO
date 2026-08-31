"""Dataset utilities for reward model training."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

RM_PROMPT_FORMAT = "plan_pass4_attempt_rewardworthy_v5"
RM_PASS_TOKEN = "Pass"
RM_FAIL_TOKEN = "Fail"
RM_ANSWER_TOKENS = [RM_PASS_TOKEN, RM_FAIL_TOKEN]


def load_labeled_records(
    path: str,
    drop_low_confidence: bool = True,
    force_corruption_negative: bool = True,
) -> list[dict]:
    """Read JSONL file, keeping only records where 'valid' is not None.

    With drop_low_confidence=True (default), also drops records where the
    judge said "low" confidence — those are mostly noisy edge cases the
    judge itself flagged as uncertain.

    With force_corruption_negative=True (default), any record whose source
    starts with 'corruption_' is FORCED to valid=False regardless of the
    judge's label. Rationale: corruptions are deterministically designed to
    break a plan (drop a method, duplicate, leak the answer, etc.). Audits
    show the judge sometimes misses subtle cases (especially exact
    duplicates), so we trust the corruption design over the judge for these.
    no-op corruptions are already filtered out at sampling time, so every
    surviving corruption_* record IS a real negative.

    Filter order (matters): corruption-override is applied FIRST, then the
    low-confidence drop is applied ONLY to non-corruption rows. This way a
    corruption row with confidence=low is never dropped — it's a deterministic
    negative we still want to train on.

    Records returned for corruption_* sources gain an extra `override_reason`
    field set to "corruption_source_forced_negative". Downstream consumers can
    ignore this field; only `valid`, `problem_text`, `plan_text` etc. are read
    by the RM trainer.

    Each record is expected to have at least:
        problem_id, problem_text, plan_text, valid, violation, source
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("valid") is None:
                continue  # malformed / api error / parsing failed

            is_corruption = (
                isinstance(rec.get("source"), str)
                and rec["source"].startswith("corruption_")
            )

            # 1. Override judge's call for corruption-derived rows FIRST.
            #    Corruptions are deterministic negatives — judge's call is
            #    secondary, and we don't want low_confidence drops to remove
            #    a corruption row that's clearly a negative by construction.
            if force_corruption_negative and is_corruption and rec["valid"] is True:
                rec["valid"] = False
                rec["override_reason"] = "corruption_source_forced_negative"

            # 2. Then drop low-confidence rows ONLY for non-corruption sources.
            #    A corruption row with low_conf is still a real negative.
            if (
                drop_low_confidence
                and rec.get("confidence") == "low"
                and not is_corruption
            ):
                continue

            records.append(rec)
    return records


def build_rm_splits(
    records: list[dict],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split records by problem_id so no problem leaks across train/val.

    Args:
        records: List of labeled records.
        val_fraction: Fraction of problems to hold out for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_records, val_records)
    """
    # Group by problem_id
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_problem[rec["problem_id"]].append(rec)

    problem_ids = sorted(by_problem.keys())
    rng = random.Random(seed)
    rng.shuffle(problem_ids)

    n_val = max(1, int(len(problem_ids) * val_fraction))
    val_ids = set(problem_ids[:n_val])

    train_records = []
    val_records = []
    for pid in problem_ids:
        if pid in val_ids:
            val_records.extend(by_problem[pid])
        else:
            train_records.extend(by_problem[pid])

    return train_records, val_records


def filter_rm_records(
    records: list[dict],
    *,
    source_policy: str = "all",
) -> tuple[list[dict], dict]:
    """Return a filtered RM record view without mutating input records.

    Source policies:
      - all: keep every loaded record.
      - model_only: keep only normal planner/model outputs.
      - non_corruption: keep model/teacher/natural rows and drop corruption_* rows.
    """
    if source_policy not in {"all", "model_only", "non_corruption"}:
        raise ValueError(f"Unknown RM source policy: {source_policy}")

    kept = []
    dropped_non_model = 0
    dropped_corruption = 0
    for rec in records:
        source = rec.get("source", "unknown")
        is_corruption = isinstance(source, str) and source.startswith("corruption_")

        if source_policy == "model_only" and source != "model":
            dropped_non_model += 1
            if is_corruption:
                dropped_corruption += 1
            continue
        if source_policy == "non_corruption" and is_corruption:
            dropped_corruption += 1
            continue

        kept.append(dict(rec))

    stats = {
        "source_policy": source_policy,
        "input_total": len(records),
        "kept_total": len(kept),
        "dropped_non_model": dropped_non_model,
        "dropped_corruption": dropped_corruption,
        "kept_pos": sum(1 for r in kept if r["valid"]),
        "kept_neg": sum(1 for r in kept if not r["valid"]),
        "kept_sources": dict(Counter(str(r.get("source", "unknown")) for r in kept)),
    }
    return kept, stats


def resolve_rm_data_view(
    rm_data_mode: str,
    source_policy: str,
) -> tuple[str, str]:
    """Resolve public RM data mode/policy to the concrete training view.

    `paper_binary_balanced` is the CPPO_new paper-base view: non-corruption
    planner-output rows, balanced 1:1 by prompt. `model_only_1to1` is a
    stricter diagnostic variant. Keeping this in the dataset module prevents
    train/report/eval scripts from drifting on accepted mode-policy pairs.
    """
    if rm_data_mode == "paper_binary_balanced":
        effective_source_policy = (
            "non_corruption" if source_policy == "all" else source_policy
        )
        if effective_source_policy != "non_corruption":
            raise ValueError(
                "paper_binary_balanced requires source_policy='non_corruption'"
            )
        return "by_problem_1to1", effective_source_policy

    if rm_data_mode == "model_only_1to1":
        effective_source_policy = (
            "model_only" if source_policy == "all" else source_policy
        )
        if effective_source_policy != "model_only":
            raise ValueError("model_only_1to1 requires source_policy='model_only'")
        return "by_problem_1to1", effective_source_policy

    return rm_data_mode, source_policy


def _with_balance_metadata(
    rec: dict,
    *,
    mode: str,
    seed: int,
    group_id: str,
) -> dict:
    out = dict(rec)
    out["balance_mode"] = mode
    out["balance_seed"] = seed
    out["rm_group_id"] = group_id
    return out


def balance_rm_records(
    records: list[dict],
    mode: str = "by_problem_1to1",
    seed: int = 42,
    neg_per_pos: int = 1,
) -> tuple[list[dict], dict]:
    """Return a deterministic balanced RM training view.

    Modes:
      - none: copy records unchanged and add stats.
      - global_1to1: downsample the global majority class.
      - by_problem_1to1: keep only problem_ids with both classes, then sample
        equal positives/negatives within each problem.

    The returned records are copies. Input records are never mutated.
    """
    if mode not in {"none", "global_1to1", "by_problem_1to1"}:
        raise ValueError(f"Unknown RM balance mode: {mode}")
    if neg_per_pos != 1:
        raise ValueError("Only neg_per_pos=1 is implemented for 1:1 RM balancing")

    raw_pos = sum(1 for r in records if r["valid"])
    raw_neg = len(records) - raw_pos
    stats = {
        "mode": mode,
        "seed": seed,
        "neg_per_pos": neg_per_pos,
        "raw_total": len(records),
        "raw_pos": raw_pos,
        "raw_neg": raw_neg,
        "balanced_total": len(records),
        "balanced_pos": raw_pos,
        "balanced_neg": raw_neg,
        "dropped_unpaired_positive": 0,
        "dropped_unpaired_negative": 0,
        "dropped_majority_samples": 0,
    }

    if mode == "none":
        return [
            _with_balance_metadata(r, mode=mode, seed=seed, group_id=str(r["problem_id"]))
            for r in records
        ], stats

    rng = random.Random(seed)

    if mode == "global_1to1":
        positives = [r for r in records if r["valid"]]
        negatives = [r for r in records if not r["valid"]]
        n_each = min(len(positives), len(negatives))
        selected_pos = rng.sample(positives, n_each)
        selected_neg = rng.sample(negatives, n_each)
        selected = selected_pos + selected_neg
        rng.shuffle(selected)

        stats["balanced_total"] = len(selected)
        stats["balanced_pos"] = n_each
        stats["balanced_neg"] = n_each
        stats["dropped_majority_samples"] = len(records) - len(selected)
        return [
            _with_balance_metadata(r, mode=mode, seed=seed, group_id="global")
            for r in selected
        ], stats

    by_problem: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_problem[rec["problem_id"]].append(rec)

    selected = []
    for pid in sorted(by_problem.keys()):
        rows = by_problem[pid]
        positives = [r for r in rows if r["valid"]]
        negatives = [r for r in rows if not r["valid"]]

        if not positives:
            stats["dropped_unpaired_negative"] += len(negatives)
            continue
        if not negatives:
            stats["dropped_unpaired_positive"] += len(positives)
            continue

        n_each = min(len(positives), len(negatives))
        stats["dropped_majority_samples"] += len(rows) - (2 * n_each)
        selected_pos = rng.sample(positives, n_each)
        selected_neg = rng.sample(negatives, n_each)
        problem_selected = selected_pos + selected_neg
        rng.shuffle(problem_selected)
        selected.extend(
            _with_balance_metadata(r, mode=mode, seed=seed, group_id=str(pid))
            for r in problem_selected
        )

    stats["balanced_total"] = len(selected)
    stats["balanced_pos"] = sum(1 for r in selected if r["valid"])
    stats["balanced_neg"] = len(selected) - stats["balanced_pos"]
    stats["selected_negative_violations"] = dict(
        Counter(
            str(r.get("violation") or "unknown")
            for r in selected
            if not r["valid"]
        )
    )
    stats["selected_negative_parseable"] = dict(
        Counter(
            str(r.get("is_parseable", "unknown"))
            for r in selected
            if not r["valid"]
        )
    )
    return selected, stats


def _format_rm_input_parts(problem_text: str, k: int = 4) -> tuple[str, str]:
    """Return the RM prompt prefix before the plan and suffix after it."""
    last_label = chr(ord("A") + k - 1)
    prefix = (
        f"You are a pass@4 attempt-allocation verifier for programming problem plans. "
        f"Return Fail if any rule is violated.\n\n"
        f"Rules for Pass:\n"
        f"1. The plan contains exactly {k} labeled methods (A, B, ..., {last_label}).\n"
        f"2. The tuple is for solver pass@4 attempt allocation, not pure "
        f"taxonomy diversity; useful variants of the same core idea are "
        f"allowed when they change the solver's likely implementation path, "
        f"robustness check, complexity tradeoff, or failure mode.\n"
        f"3. Return Fail for harmful duplicates, sequential workflow, or "
        f"methods that would lead to essentially identical solver code.\n"
        f"4. Every method is specific to the problem and gives a concrete, executable path; "
        f"return Fail for filler, generic advice, wrong-task plans, unsupported "
        f"assumptions, or an unexecutable empty slogan.\n"
        f"5. The plan does not solve the problem or reveal an answer; it has "
        f"no hidden chain-of-thought, no executable code, imports, function "
        f"definitions, pseudocode, or fenced code blocks.\n"
        f"6. Each method is concise: one to three sentences.\n\n"
        f"Output exactly one word on the answer line: Pass or Fail.\n\n"
        f"## Problem\n{problem_text}\n\n"
        f"## Plan\n"
    )
    suffix = "\n\n## Answer:\n"
    return prefix, suffix


def format_rm_input(problem_text: str, plan_text: str, k: int = 4) -> str:
    """Format problem and plan text for reward model input.

    The RM is small, so the prompt states the quality rubric explicitly
    instead of relying on the model to infer what "valid plan" means.
    The model is trained to predict the next token: Pass or Fail.
    """
    prefix, suffix = _format_rm_input_parts(problem_text, k=k)
    return f"{prefix}{plan_text}{suffix}"


def _format_rm_input(problem_text: str, plan_text: str) -> str:
    """Backward-compatible wrapper for the default K=4 RM prompt."""
    return format_rm_input(problem_text, plan_text, k=4)


def tokenize_rm_input(
    tokenizer: Any,
    problem_text: str,
    plan_text: str,
    *,
    max_length: int,
    k: int = 4,
    return_tensors: str | None = "pt",
) -> dict[str, Any]:
    """Tokenize RM input while preserving the final answer surface.

    The reward model is trained by reading Pass/Fail logits at the final real
    token. Plain right-truncation can drop the trailing ``## Answer:\n`` marker,
    causing training to score an arbitrary body token. When the formatted input
    is too long, truncate the plan first so the rubric/problem context and the
    answer surface remain present. If the rubric/problem alone is too long, keep
    as much of that prefix as possible plus the answer surface.
    """
    prefix, suffix = _format_rm_input_parts(problem_text, k=k)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    plan_ids = tokenizer.encode(plan_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    full_len = len(prefix_ids) + len(plan_ids) + len(suffix_ids)
    if full_len <= max_length:
        input_ids = prefix_ids + plan_ids + suffix_ids
    else:
        plan_budget = max_length - len(prefix_ids) - len(suffix_ids)
        if plan_budget >= 0:
            input_ids = prefix_ids + plan_ids[:plan_budget] + suffix_ids
        else:
            prefix_budget = max(0, max_length - len(suffix_ids))
            # The tail of the prefix contains the concrete problem text and
            # the ``## Plan`` marker, which are more important than preserving
            # the beginning of the static rubric when the rubric+problem alone
            # exceeds max_length.
            input_ids = (prefix_ids[-prefix_budget:] if prefix_budget else []) + suffix_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
    attention_mask = [1] * len(input_ids)

    if return_tensors == "pt":
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }
    if return_tensors is None:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    raise ValueError("return_tensors must be 'pt' or None")


def get_rm_verbalizer_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Return single-token Pass/Fail verbalizer IDs for the RM prompt surface.

    The LM is scored at the final prompt token, so the answer token must append
    cleanly to the exact prompt surface used in training and inference. This
    catches subtle tokenizer bugs such as accidentally scoring a bare token
    when the prompt surface naturally requires a space-prefixed token.
    """
    token_ids = {}
    for token in RM_ANSWER_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"RM answer token {token!r} must be exactly one token; got {ids}"
            )
        token_ids[token] = ids[0]

    probe_prompt = format_rm_input(
        "Given an integer n, return whether it is even.",
        (
            "A: Check n modulo 2.\n"
            "B: Use bitwise parity.\n"
            "C: Divide by 2 and inspect the remainder.\n"
            "D: Compare against a generated sequence of even numbers."
        ),
    )
    base_ids = tokenizer.encode(probe_prompt, add_special_tokens=False)
    for token in RM_ANSWER_TOKENS:
        full_ids = tokenizer.encode(probe_prompt + token, add_special_tokens=False)
        expected_suffix = [token_ids[token]]
        if (
            full_ids[: len(base_ids)] != base_ids
            or full_ids[len(base_ids) :] != expected_suffix
        ):
            raise RuntimeError(
                "RM answer surface is not a clean one-token continuation for "
                f"{token!r}; suffix={full_ids[len(base_ids):]!r}, "
                f"expected={expected_suffix!r}"
            )

    return token_ids[RM_PASS_TOKEN], token_ids[RM_FAIL_TOKEN]


class RMDataset(Dataset):
    """PyTorch Dataset for reward model training.

    Each item returns a dict with:
        input_ids, attention_mask, label, problem_id, violation, source
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Precompute problem_id index for grouped sampling
        self._problem_ids = [r["problem_id"] for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        encoding = tokenize_rm_input(
            self.tokenizer,
            rec["problem_text"],
            rec["plan_text"],
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": 1.0 if rec["valid"] else 0.0,
            "problem_id": rec["problem_id"],
            "violation": rec.get("violation", "none"),
            "source": rec.get("source", "unknown"),
        }


class GroupedBatchSampler(Sampler):
    """Yields batches grouping samples by problem_id.

    This ensures ranking loss gets valid pairs (samples from the same problem).
    Groups are shuffled each epoch; samples within a group are also shuffled.
    """

    def __init__(
        self,
        dataset: RMDataset,
        batch_size: int = 8,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        # Build groups: problem_id -> list of indices
        self._groups: dict[str, list[int]] = defaultdict(list)
        for i, pid in enumerate(dataset._problem_ids):
            self._groups[pid].append(i)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        group_keys = list(self._groups.keys())
        rng.shuffle(group_keys)

        batch: list[int] = []
        for key in group_keys:
            indices = list(self._groups[key])
            rng.shuffle(indices)
            batch.extend(indices)

            # Yield when batch is full
            while len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]

        # Yield remaining samples
        if batch:
            yield batch

    def __len__(self) -> int:
        total = sum(len(v) for v in self._groups.values())
        return (total + self.batch_size - 1) // self.batch_size


class RMCollator:
    """Pads batches to max length and returns proper tensors.

    Returns a dict with:
        input_ids:   (B, L) long tensor
        attention_mask: (B, L) long tensor
        labels:      (B,) float tensor
        problem_ids: list[str]
        violations:  list[str]
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(f["input_ids"].size(0) for f in features)

        input_ids_list = []
        attention_mask_list = []
        labels = []
        problem_ids = []
        violations = []

        for f in features:
            seq_len = f["input_ids"].size(0)
            pad_len = max_len - seq_len

            # Right-padding
            input_ids = torch.cat([
                f["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
            ])
            attention_mask = torch.cat([
                f["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ])

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels.append(f["label"])
            problem_ids.append(f["problem_id"])
            violations.append(f["violation"])

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "problem_ids": problem_ids,
            "violations": violations,
        }
