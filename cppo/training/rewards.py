"""Plan reward scorer using a frozen binary Pass/Fail reward model."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cppo.data.plan_canonicalizer import canonicalize_plan
from cppo.reward_model.calibrate import compute_calibrated_reward
from cppo.reward_model.dataset import (
    RM_ANSWER_TOKENS,
    RM_PROMPT_FORMAT,
    format_rm_input,
    get_rm_verbalizer_token_ids,
    tokenize_rm_input,
)

logger = logging.getLogger(__name__)


def assign_group_argmax_rewards(
    scores: list[float],
    group_ids: list[str],
    parseable: list[bool],
    min_score: float | None = None,
) -> list[float]:
    """Assign one binary winner per group, unless no eligible item exists.

    Eligibility requires parseable=True and, when set, score >= min_score.
    Ties are deterministic: the earliest item in the original order wins.
    """
    if not (len(scores) == len(group_ids) == len(parseable)):
        raise ValueError("scores, group_ids, and parseable must have the same length")

    rewards = [0.0] * len(scores)
    best_by_group: dict[str, tuple[float, int]] = {}
    for i, (score, group_id, ok) in enumerate(zip(scores, group_ids, parseable)):
        if not ok:
            continue
        if not math.isfinite(score):
            continue
        if min_score is not None and score < min_score:
            continue

        current = best_by_group.get(group_id)
        if current is None or score > current[0]:
            best_by_group[group_id] = (score, i)

    for _, winner_idx in best_by_group.values():
        rewards[winner_idx] = 1.0

    return rewards


class PlanRewardScorer:
    """Loads a frozen binary Pass/Fail reward model to score plans.

    The paper-base reward consumes binary J_psi. The calibrated C_psi value is
    retained only as a legacy diagnostic for old audits and reports.
    Uses attention_mask-based last-token indexing (not logits[:, -1, :])
    to correctly handle right-padded batches.
    """

    def __init__(
        self,
        rm_path: str,
        device: str = "cuda",
        jpsi_threshold: float | None = None,
    ):
        self.device = device
        rm_path = Path(rm_path)
        self.max_length = 512
        metadata_path = rm_path / "rm_metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(
                f"missing rm_metadata.json at {metadata_path}; cannot safely "
                "score this checkpoint as a Pass/Fail RM. Retrain the RM with "
                "the Pass/Fail prompt or use a legacy scorer."
            )
        with open(metadata_path) as f:
            metadata = json.load(f)
        expected_format = RM_PROMPT_FORMAT
        expected_tokens = RM_ANSWER_TOKENS
        prompt_format = metadata.get("prompt_format")
        answer_tokens = metadata.get("answer_tokens")
        if prompt_format != expected_format or answer_tokens != expected_tokens:
            raise RuntimeError(
                "incompatible RM metadata: expected "
                f"prompt_format={expected_format!r}, answer_tokens={expected_tokens!r}; "
                f"got prompt_format={prompt_format!r}, answer_tokens={answer_tokens!r}. "
                "Retrain the RM with the Pass/Fail prompt or use a legacy scorer."
            )
        self.max_length = int(metadata.get("max_length", self.max_length))
        self.canonicalized_input = bool(metadata.get("canonicalized_input", False))
        self.jpsi_threshold = 0.5

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(rm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(rm_path)
        self.model.to(self.device)
        self.model.eval()
        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        self._pass_id, self._fail_id = get_rm_verbalizer_token_ids(self.tokenizer)

        # Load legacy diagnostic calibration thresholds. These must not be
        # treated as the CPPO_new paper-base reward.
        thresholds_path = rm_path / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                thresholds = json.load(f)
            self.tau_low = thresholds["tau_low"]
            self.tau_high = thresholds["tau_high"]
            self.jpsi_threshold = float(
                thresholds.get("jpsi_threshold", thresholds.get("t_accept", 0.5))
            )
            logger.info(
                f"Loaded thresholds: tau_low={self.tau_low:.4f}, "
                f"tau_high={self.tau_high:.4f}, "
                f"jpsi_threshold={self.jpsi_threshold:.4f}"
            )
        else:
            # Fallback defaults if thresholds not yet calibrated
            logger.warning(
                f"No thresholds.json found at {thresholds_path}; "
                "using defaults tau_low=0.3, tau_high=0.7"
            )
            self.tau_low = 0.3
            self.tau_high = 0.7
        if jpsi_threshold is not None:
            self.jpsi_threshold = float(jpsi_threshold)
            logger.info("Using configured jpsi_threshold=%.4f", self.jpsi_threshold)

    @staticmethod
    def _format_rm_input(problem_text: str, plan_text: str) -> str:
        """Format problem+plan for the reward model (same as dataset.py)."""
        return format_rm_input(problem_text, plan_text, k=4)

    @torch.no_grad()
    def _get_pass_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get P(Pass) for each sequence, using attention_mask-based indexing.

        Args:
            input_ids:      (B, L) token IDs.
            attention_mask: (B, L) attention mask.

        Returns:
            (B,) tensor of P(Pass) probabilities.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, V)

        # Find last real token position using attention_mask
        last_idx = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        last_logits = logits[batch_idx, last_idx, :]  # (B, V)

        # Softmax over [Pass, Fail] to get P(Pass)
        pass_logits = last_logits[:, self._pass_id]  # (B,)
        fail_logits = last_logits[:, self._fail_id]  # (B,)
        pair_logits = torch.stack([pass_logits, fail_logits], dim=-1)  # (B, 2)
        probs = F.softmax(pair_logits, dim=-1)[:, 0]  # (B,) P(Pass)

        return probs

    def score_plans(
        self,
        problem_texts: list[str],
        plan_texts: list[str],
    ) -> list[float]:
        """Score pairs and return legacy diagnostic C_psi rewards.

        Args:
            problem_texts: Problem descriptions.
            plan_texts:    Generated plan texts.

        Returns:
            List of calibrated diagnostic scores in [0, 1].
        """
        return [
            details["c_psi"]
            for details in self.score_plan_details(problem_texts, plan_texts)
        ]

    def score_plan_details(
        self,
        problem_texts: list[str],
        plan_texts: list[str],
    ) -> list[dict]:
        """Score pairs and return raw P(Pass), binary J_psi, and diagnostic C_psi."""
        assert len(problem_texts) == len(plan_texts)

        details: list[dict | None] = [None] * len(plan_texts)
        score_indices: list[int] = []
        encodings = []
        for i, (problem_text, plan_text) in enumerate(zip(problem_texts, plan_texts)):
            score_plan_text = plan_text
            if getattr(self, "canonicalized_input", False):
                canon = canonicalize_plan(plan_text, k=4)
                if not canon.is_parseable:
                    details[i] = {
                        "pass_prob": 0.0,
                        "j_psi": 0.0,
                        "c_psi": compute_calibrated_reward(
                            0.0, self.tau_low, self.tau_high
                        ),
                    }
                    continue
                score_plan_text = canon.plan_text
            encodings.append(
                tokenize_rm_input(
                    self.tokenizer,
                    problem_text,
                    score_plan_text,
                    max_length=self.max_length,
                    return_tensors=None,
                )
            )
            score_indices.append(i)

        if not encodings:
            return [d if d is not None else {
                "pass_prob": 0.0,
                "j_psi": 0.0,
                "c_psi": compute_calibrated_reward(0.0, self.tau_low, self.tau_high),
            } for d in details]

        max_len = max(len(e["input_ids"]) for e in encodings)
        input_rows = []
        mask_rows = []
        pad_id = self.tokenizer.pad_token_id
        for enc in encodings:
            pad_len = max_len - len(enc["input_ids"])
            input_rows.append(enc["input_ids"] + [pad_id] * pad_len)
            mask_rows.append(enc["attention_mask"] + [0] * pad_len)
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=self.device)

        # Get P(Pass) probabilities
        pass_probs = self._get_pass_probs(input_ids, attention_mask)

        scored_details = [
            {
                "pass_prob": float(p.item()),
                "j_psi": 1.0
                if float(p.item()) >= getattr(self, "jpsi_threshold", 0.5)
                else 0.0,
                "c_psi": compute_calibrated_reward(
                    p.item(), self.tau_low, self.tau_high
                ),
            }
            for p in pass_probs
        ]
        for i, detail in zip(score_indices, scored_details):
            details[i] = detail
        if any(d is None for d in details):
            raise RuntimeError("internal scoring error: missing RM detail row")
        return [d for d in details if d is not None]
