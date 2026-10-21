"""The distillation loss (Team Work Plan §4, Shivam).

    loss = alpha * BCE(sigmoid(z_s), y)
         + (1 - alpha) * T^2 * BCE(sigmoid(z_s / T), sigmoid(z_t / T))

* ``z_s``: student logits; ``z_t``: the stack's ``logit_<label>`` from the soft-label shards
  (not ``logit(p)``, which saturates in float32; spec §5). At ``T = 1`` the soft target
  ``sigmoid(z_t)`` is exactly the shard's ``p_<label>``.
* Both logits are divided by ``T`` in the soft term; ``T^2`` keeps its gradient scale
  comparable across temperatures (Hinton et al., 2015).
* ``sample_weight`` (the bias step's reweighting) scales the gold term only.
* ``soft_mask`` drops rows without a usable soft label (teacher ``status == failed``) from
  the soft term; the soft term averages over the rows that remain.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(student_logits: torch.Tensor, labels: torch.Tensor,
                      teacher_logits: torch.Tensor | None = None, alpha: float = 1.0,
                      temperature: float = 1.0, sample_weight: torch.Tensor | None = None,
                      soft_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """``(loss, {"gold": ..., "soft": ...})`` for a batch of ``(rows, labels)`` logits.

    Terms average over labels, then over rows. ``teacher_logits`` may be omitted only when
    ``alpha == 1``.
    """
    z_s = student_logits.float()
    gold_rows = F.binary_cross_entropy_with_logits(z_s, labels.float(),
                                                   reduction="none").mean(dim=1)
    if sample_weight is not None:
        gold_rows = gold_rows * sample_weight.float()
    gold = gold_rows.mean()
    parts = {"gold": float(gold.detach())}
    if alpha >= 1.0:
        parts["soft"] = float("nan")
        return gold, parts
    if teacher_logits is None:
        raise ValueError("teacher_logits are required when alpha < 1")

    t = float(temperature)
    mask = (torch.ones(len(z_s), dtype=torch.bool, device=z_s.device) if soft_mask is None
            else soft_mask.bool())
    # Failed rows hold NaN logits; zero them so NaN never reaches the gradient, then mask.
    z_t = torch.where(mask[:, None], teacher_logits.float(), torch.zeros_like(z_s))
    soft_rows = F.binary_cross_entropy_with_logits(z_s / t, torch.sigmoid(z_t / t),
                                                   reduction="none").mean(dim=1)
    kept = mask.float()
    soft = (soft_rows * kept).sum() / kept.sum().clamp(min=1.0)
    parts["soft"] = float(soft.detach())
    return alpha * gold + (1.0 - alpha) * t ** 2 * soft, parts
