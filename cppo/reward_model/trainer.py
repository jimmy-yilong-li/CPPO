"""Reward model trainer with grouped ranking loss."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .dataset import (
    GroupedBatchSampler,
    RM_ANSWER_TOKENS,
    RM_PROMPT_FORMAT,
    RMCollator,
    RMDataset,
    get_rm_verbalizer_token_ids,
)

logger = logging.getLogger(__name__)


def compute_pos_weight(records: list[dict], max_weight: float = 50.0) -> float:
    """Return pos_weight = #neg / #pos for weighted BCE.

    Used to counter the heavy class imbalance from corruption-augmented data
    (typical ratio ~1:19). Capped at max_weight so a few stray positives
    don't blow up gradients.
    """
    n_pos = sum(1 for r in records if r["valid"])
    n_neg = len(records) - n_pos
    if n_pos == 0:
        return 1.0
    return float(min(n_neg / n_pos, max_weight))


def resolve_pos_weight(
    records: list[dict],
    pos_weight_override: float | None = None,
) -> float:
    """Return the positive-class BCE weight for this training run."""
    if pos_weight_override is None:
        return compute_pos_weight(records)
    if pos_weight_override <= 0:
        raise ValueError("pos_weight_override must be positive")
    return float(pos_weight_override)


_SELECTION_METRICS: dict[str, tuple[str, bool]] = {
    "average_precision": ("average_precision", True),
    "val_average_precision": ("average_precision", True),
    "ap": ("average_precision", True),
    "auc": ("auc", True),
    "val_auc": ("auc", True),
    "accuracy": ("accuracy", True),
    "val_accuracy": ("accuracy", True),
    "balanced_accuracy": ("balanced_accuracy", True),
    "val_balanced_accuracy": ("balanced_accuracy", True),
    "error_rate": ("error_rate", False),
    "val_error": ("error_rate", False),
}


def get_selection_metric_value(metric: str, val_results: dict) -> float:
    """Return the validation metric value used for checkpoint selection."""
    if metric not in _SELECTION_METRICS:
        choices = ", ".join(sorted(_SELECTION_METRICS))
        raise ValueError(f"Unknown selection metric {metric!r}; choose one of: {choices}")
    key, _higher_is_better = _SELECTION_METRICS[metric]
    return float(val_results[key])


def is_better_selection_metric(
    metric: str,
    *,
    current_value: float,
    best_value: float | None,
) -> bool:
    """Return whether current_value improves over best_value for metric."""
    if metric not in _SELECTION_METRICS:
        choices = ", ".join(sorted(_SELECTION_METRICS))
        raise ValueError(f"Unknown selection metric {metric!r}; choose one of: {choices}")
    if best_value is None:
        return True
    _key, higher_is_better = _SELECTION_METRICS[metric]
    if higher_is_better:
        return current_value > best_value
    return current_value < best_value


class RewardModelTrainer:
    """Trains a causal LM as a reward model (binary: Pass/Fail).

    The model predicts P(Pass | prompt) as the reward signal. Uses a
    combination of binary cross-entropy and within-problem pairwise
    ranking loss.

    CRITICAL: Uses attention_mask to find the last real token position,
    NOT logits[:, -1, :] which would read pad tokens for short sequences.
    """

    def __init__(
        self,
        model_name: str,
        lr: float = 1e-5,
        ranking_weight: float = 0.5,
        max_length: int = 512,
        batch_size: int = 8,
        device: str = "cuda",
        canonicalized_input: bool = False,
    ):
        self.model_name = model_name
        self.lr = lr
        self.ranking_weight = ranking_weight
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.canonicalized_input = canonicalized_input

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Right-padding for causal LM reward model
        self.tokenizer.padding_side = "right"

        config = AutoConfig.from_pretrained(model_name)
        # Some Qwen3.5 checkpoints expose a wrapper config with the actual
        # causal-LM vocabulary and dimensions under text_config. Older
        # Transformers builds can select the right model class but pass the
        # wrapper config through, which fails when the model asks for
        # config.vocab_size. Use the text config when present.
        if not hasattr(config, "vocab_size") and hasattr(config, "text_config"):
            config = config.text_config
        self.model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        self.model.to(self.device)

        self._pass_id, self._fail_id = get_rm_verbalizer_token_ids(self.tokenizer)

    def _get_pass_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get P(Pass) for each sequence in the batch.

        CRITICAL: Uses attention_mask.sum(dim=1) - 1 to find the last
        real token, NOT logits[:, -1, :]. With right-padding, the last
        position may be a pad token for shorter sequences.

        Args:
            input_ids:      (B, L) token IDs.
            attention_mask: (B, L) attention mask.

        Returns:
            (B,) tensor of P(Pass) probabilities.
        """
        probs, _ = self._get_pass_prob_and_score(input_ids, attention_mask)
        return probs

    def _get_pass_prob_and_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get P(Pass) and the Pass-vs-Fail logit margin for each sequence.

        BCE uses P(Pass). The pairwise ranking loss uses the unbounded
        Pass-vs-Fail logit margin so confident inverted pairs still carry a
        meaningful Bradley-Terry correction.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, V)

        # Find last real token position using attention_mask
        last_idx = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        last_logits = logits[batch_idx, last_idx, :]  # (B, V)

        # Extract Pass/Fail logits and compute P(Pass)
        pass_logits = last_logits[:, self._pass_id]  # (B,)
        fail_logits = last_logits[:, self._fail_id]  # (B,)
        pass_fail_score = pass_logits - fail_logits

        # Softmax over [Pass, Fail] to get P(Pass)
        pair_logits = torch.stack([pass_logits, fail_logits], dim=-1)  # (B, 2)
        probs = F.softmax(pair_logits, dim=-1)[:, 0]  # (B,) P(Pass)

        return probs, pass_fail_score

    def _ranking_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        problem_ids: list[str],
    ) -> torch.Tensor:
        """Within-problem pairwise ranking loss.

        For each problem with both positive and negative samples,
        we want score(positive) > score(negative). Training passes the
        Pass-vs-Fail logit margin as score, not P(Pass).

        Uses Bradley-Terry loss: -log sigmoid(p_pos - p_neg).
        """
        # Group indices by problem_id
        groups: dict[str, list[int]] = defaultdict(list)
        for i, pid in enumerate(problem_ids):
            groups[pid].append(i)

        loss_terms = []
        for pid, indices in groups.items():
            pos_indices = [i for i in indices if labels[i] > 0.5]
            neg_indices = [i for i in indices if labels[i] <= 0.5]

            if not pos_indices or not neg_indices:
                continue

            for pi in pos_indices:
                for ni in neg_indices:
                    # We want score[pi] > score[ni].
                    diff = scores[pi] - scores[ni]
                    loss_terms.append(-F.logsigmoid(diff))

        if not loss_terms:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        return torch.stack(loss_terms).mean()

    def _compute_batch_loss(
        self,
        *,
        probs: torch.Tensor,
        labels: torch.Tensor,
        rank_scores: torch.Tensor,
        problem_ids: list[str],
        pos_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute RM batch loss.

        Paper-aligned RM configs set ranking_weight=0, which means pure BCE.
        In that mode we also skip constructing pairwise ranking terms.
        """
        probs_fp32 = probs.float()
        labels_fp32 = labels.float()
        sample_w = labels_fp32 * pos_weight + (1.0 - labels_fp32)
        bce_loss = F.binary_cross_entropy(
            probs_fp32, labels_fp32, weight=sample_w
        )

        if self.ranking_weight > 0:
            rank_loss = self._ranking_loss(
                rank_scores.float(), labels_fp32, problem_ids
            )
        else:
            rank_loss = torch.tensor(0.0, device=probs.device)

        return bce_loss + self.ranking_weight * rank_loss, bce_loss, rank_loss

    def train(
        self,
        train_records: list[dict],
        val_records: list[dict],
        epochs: int = 3,
        pos_weight_override: float | None = None,
        selection_metric: str = "average_precision",
        extra_history: dict | None = None,
    ) -> dict:
        """Train the reward model.

        Args:
            train_records: Training labeled records.
            val_records:   Validation labeled records.
            epochs:        Number of training epochs.

        Returns:
            Dict with training history (train_loss, val_loss per epoch).
        """
        # Build datasets
        train_dataset = RMDataset(train_records, self.tokenizer, self.max_length)
        val_dataset = RMDataset(val_records, self.tokenizer, self.max_length)

        collator = RMCollator(pad_token_id=self.tokenizer.pad_token_id)

        # Train loader with grouped batch sampler
        train_sampler = GroupedBatchSampler(
            train_dataset, batch_size=self.batch_size
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collator,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # Class-imbalance handling: weight positives by neg/pos.
        # Without this, val_accuracy at 1:19 is dominated by the
        # always-predict-negative baseline (~95%), giving zero signal.
        pos_weight = resolve_pos_weight(train_records, pos_weight_override)
        logger.info(f"Using pos_weight={pos_weight:.2f} for weighted BCE")

        history: dict = {
            "train_loss": [],
            "train_accuracy": [],
            "train_error": [],
            "train_auc": [],
            "train_balanced_acc": [],
            "train_average_precision": [],
            "train_predicted_pass_rate": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_error": [],
            "val_auc": [],
            "val_balanced_acc": [],
            "val_average_precision": [],
            "val_predicted_pass_rate": [],
            "pos_weight": pos_weight,
            "selection_metric": selection_metric,
        }
        if extra_history:
            history.update(extra_history)
        # Track best checkpoint by a validation metric. AP is the historical
        # default, but fixed-threshold J_psi runs may prefer val_error/FPR-like
        # metrics because AP can improve while the 0.5 decision boundary drifts.
        best_metric_value: float | None = None
        best_state = None

        for epoch in range(epochs):
            # --- Train ---
            self.model.train()
            train_sampler.set_epoch(epoch)
            epoch_losses = []

            for batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False
            ):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                problem_ids = batch["problem_ids"]

                optimizer.zero_grad()

                probs, rank_scores = self._get_pass_prob_and_score(
                    input_ids, attention_mask
                )
                loss, _, _ = self._compute_batch_loss(
                    probs=probs,
                    labels=labels,
                    rank_scores=rank_scores,
                    problem_ids=problem_ids,
                    pos_weight=pos_weight,
                )
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            history["train_loss"].append(avg_train_loss)

            # --- Validate ---
            train_results = self.evaluate(train_loader)
            val_results = self.evaluate(val_loader)
            history["train_accuracy"].append(train_results["accuracy"])
            history["train_error"].append(train_results["error_rate"])
            history["train_auc"].append(train_results["auc"])
            history["train_balanced_acc"].append(train_results["balanced_accuracy"])
            history["train_average_precision"].append(
                train_results["average_precision"]
            )
            history["train_predicted_pass_rate"].append(
                train_results["predicted_pass_rate"]
            )
            history["val_loss"].append(0.0)  # Placeholder; evaluate returns metrics
            history["val_accuracy"].append(val_results["accuracy"])
            history["val_error"].append(val_results["error_rate"])
            history["val_auc"].append(val_results["auc"])
            history["val_balanced_acc"].append(val_results["balanced_accuracy"])
            history["val_average_precision"].append(val_results["average_precision"])
            history["val_predicted_pass_rate"].append(
                val_results["predicted_pass_rate"]
            )

            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"train_err={train_results['error_rate']:.4f}, "
                f"train_pred_pass={train_results['predicted_pass_rate']:.4f}, "
                f"val_acc={val_results['accuracy']:.4f}, "
                f"val_err={val_results['error_rate']:.4f}, "
                f"val_pred_pass={val_results['predicted_pass_rate']:.4f}, "
                f"val_balanced_acc={val_results['balanced_accuracy']:.4f}, "
                f"val_auc={val_results['auc']:.4f}, "
                f"val_AP={val_results['average_precision']:.4f}"
            )

            metric_value = get_selection_metric_value(selection_metric, val_results)
            if is_better_selection_metric(
                selection_metric,
                current_value=metric_value,
                best_value=best_metric_value,
            ):
                best_metric_value = metric_value
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                logger.info(
                    f"  -> new best {selection_metric}={best_metric_value:.4f}, "
                    "snapshotting weights"
                )

        # Restore best weights so the saved model is the selected epoch.
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(
                f"Restored best checkpoint by {selection_metric}="
                f"{best_metric_value:.4f}"
            )
        history["best_selection_metric"] = selection_metric
        history["best_selection_metric_value"] = best_metric_value
        if selection_metric in {"average_precision", "val_average_precision", "ap"}:
            history["best_average_precision"] = best_metric_value
        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate the model on a DataLoader.

        Returns:
            Dict with accuracy, auc, probs (np.ndarray), labels (np.ndarray),
            violations (list[str]).
        """
        self.model.eval()

        all_probs = []
        all_labels = []
        all_violations = []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            probs = self._get_pass_prob(input_ids, attention_mask)
            all_probs.extend(probs.float().cpu().numpy().tolist())
            all_labels.extend(labels.float().numpy().tolist())
            all_violations.extend(batch["violations"])

        probs_arr = np.array(all_probs)
        labels_arr = np.array(all_labels)

        # Accuracy at 0.5 threshold
        preds = (probs_arr >= 0.5).astype(int)
        accuracy = float((preds == labels_arr).mean()) if len(labels_arr) > 0 else 0.0
        error_rate = 1.0 - accuracy if len(labels_arr) > 0 else 0.0
        predicted_pass_rate = float(preds.mean()) if len(preds) > 0 else 0.0

        # AUC + average_precision + balanced_accuracy
        auc = 0.0
        ap = 0.0
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            if len(np.unique(labels_arr)) > 1:
                auc = float(roc_auc_score(labels_arr, probs_arr))
                ap = float(average_precision_score(labels_arr, probs_arr))
        except ImportError:
            pass

        # Balanced accuracy at 0.5 threshold = (TPR + TNR) / 2.
        # On 1:19 splits, plain accuracy is dominated by negatives — this
        # gives an actually-informative scalar.
        if len(labels_arr) > 0 and labels_arr.sum() > 0 and (labels_arr == 0).sum() > 0:
            tpr = float(((preds == 1) & (labels_arr == 1)).sum() / labels_arr.sum())
            tnr = float(((preds == 0) & (labels_arr == 0)).sum() / (labels_arr == 0).sum())
            balanced_accuracy = (tpr + tnr) / 2.0
        else:
            balanced_accuracy = accuracy

        return {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "balanced_accuracy": balanced_accuracy,
            "auc": auc,
            "average_precision": ap,
            "predicted_pass_rate": predicted_pass_rate,
            "probs": probs_arr,
            "labels": labels_arr,
            "violations": all_violations,
        }

    def save(self, path: str | Path) -> None:
        """Save model and tokenizer to directory.

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with (path / "rm_metadata.json").open("w") as f:
            json.dump(
                {
                    "max_length": self.max_length,
                    "prompt_format": RM_PROMPT_FORMAT,
                    "answer_tokens": RM_ANSWER_TOKENS,
                    "canonicalized_input": self.canonicalized_input,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved reward model to {path}")
