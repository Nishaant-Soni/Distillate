import pytest
import torch
import torch.nn.functional as F

from src.student.loss import distillation_loss


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    z_s = torch.randn(8, 6, generator=g)
    z_t = torch.randn(8, 6, generator=g) * 3
    y = (torch.rand(8, 6, generator=g) < 0.3).float()
    return z_s, z_t, y


def test_gold_only_is_plain_bce(batch):
    z_s, z_t, y = batch
    loss, parts = distillation_loss(z_s, y, z_t, alpha=1.0, temperature=3.0)
    assert torch.isclose(loss, F.binary_cross_entropy_with_logits(z_s, y))
    assert parts["gold"] == pytest.approx(float(loss))


def test_gold_only_needs_no_teacher(batch):
    z_s, _, y = batch
    loss, _ = distillation_loss(z_s, y, None, alpha=1.0)
    assert torch.isfinite(loss)


def test_soft_needs_teacher(batch):
    z_s, _, y = batch
    with pytest.raises(ValueError, match="teacher_logits"):
        distillation_loss(z_s, y, None, alpha=0.5)


def test_soft_only_at_t1_targets_sigmoid_of_teacher(batch):
    z_s, z_t, y = batch
    loss, _ = distillation_loss(z_s, y, z_t, alpha=0.0, temperature=1.0)
    assert torch.isclose(loss, F.binary_cross_entropy_with_logits(z_s, torch.sigmoid(z_t)))


def test_temperature_divides_both_logits_and_scales_by_t_squared(batch):
    z_s, z_t, y = batch
    t = 2.5
    loss, _ = distillation_loss(z_s, y, z_t, alpha=0.0, temperature=t)
    expected = t ** 2 * F.binary_cross_entropy_with_logits(z_s / t, torch.sigmoid(z_t / t))
    assert torch.isclose(loss, expected)


def test_mixed_is_the_weighted_sum(batch):
    z_s, z_t, y = batch
    alpha, t = 0.3, 2.0
    loss, parts = distillation_loss(z_s, y, z_t, alpha=alpha, temperature=t)
    expected = alpha * parts["gold"] + (1 - alpha) * t ** 2 * parts["soft"]
    assert float(loss) == pytest.approx(expected, rel=1e-5)


def test_sample_weight_scales_gold_term_only(batch):
    z_s, z_t, y = batch
    w = torch.linspace(0.5, 3.0, 8)
    soft_plain, _ = distillation_loss(z_s, y, z_t, alpha=0.0, temperature=2.0)
    soft_weighted, _ = distillation_loss(z_s, y, z_t, alpha=0.0, temperature=2.0,
                                         sample_weight=w)
    assert torch.isclose(soft_plain, soft_weighted)

    gold_weighted, _ = distillation_loss(z_s, y, alpha=1.0, sample_weight=w)
    per_row = F.binary_cross_entropy_with_logits(z_s, y, reduction="none").mean(dim=1)
    assert torch.isclose(gold_weighted, (per_row * w).mean())


def test_unit_weights_change_nothing(batch):
    z_s, z_t, y = batch
    a, _ = distillation_loss(z_s, y, z_t, alpha=0.4, temperature=2.0)
    b, _ = distillation_loss(z_s, y, z_t, alpha=0.4, temperature=2.0, sample_weight=torch.ones(8))
    assert torch.isclose(a, b)


def test_failed_rows_are_masked_and_nan_never_leaks(batch):
    z_s, z_t, y = batch
    z_t = z_t.clone()
    z_t[[1, 5]] = float("nan")
    mask = torch.ones(8, dtype=torch.bool)
    mask[[1, 5]] = False
    z_s = z_s.clone().requires_grad_(True)
    loss, _ = distillation_loss(z_s, y, z_t, alpha=0.0, temperature=2.0, soft_mask=mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z_s.grad).all()
    assert torch.all(z_s.grad[[1, 5]] == 0)

    keep = mask.nonzero().squeeze(1)
    expected, _ = distillation_loss(z_s[keep].detach(), y[keep], z_t[keep], alpha=0.0,
                                    temperature=2.0)
    assert torch.isclose(loss, expected)


def test_soft_term_gradient_vanishes_when_student_matches_teacher(batch):
    _, z_t, y = batch
    z_s = z_t.clone().requires_grad_(True)
    loss, _ = distillation_loss(z_s, y, z_t, alpha=0.0, temperature=3.0)
    loss.backward()
    assert torch.allclose(z_s.grad, torch.zeros_like(z_s.grad), atol=1e-7)
