import math

import pytest
import torch

from helper.metrics import (
    compute_grad_match_loss,
    compute_psnr_from_mse,
    compute_ssim_batch,
    total_variation,
)


def test_psnr_identical_images_returns_inf():
    assert math.isinf(compute_psnr_from_mse(0.0))


def test_psnr_below_eps_returns_inf():
    assert math.isinf(compute_psnr_from_mse(1e-13))


def test_psnr_known_value():
    # MSE=0.01, max_val=1.0 → 10*log10(1/0.01) = 20.0 dB
    assert abs(compute_psnr_from_mse(0.01) - 20.0) < 1e-9


def test_psnr_known_value_max_val():
    # MSE=1.0, max_val=255.0 → 10*log10(255^2/1) = 10*log10(65025) ≈ 48.13 dB
    expected = 10.0 * math.log10(255.0 ** 2)
    assert abs(compute_psnr_from_mse(1.0, max_val=255.0) - expected) < 1e-9


def test_ssim_identical_images_returns_one():
    x = torch.rand(2, 1, 28, 28)
    score = compute_ssim_batch(x, x)
    assert abs(score - 1.0) < 1e-5


def test_ssim_different_images_below_one():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 28, 28)
    y = torch.rand(2, 1, 28, 28)
    score = compute_ssim_batch(x, y)
    assert score < 1.0


def test_ssim_rgb_images_in_range():
    torch.manual_seed(1)
    x = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 3, 32, 32)
    score = compute_ssim_batch(x, y)
    assert -1.0 <= score <= 1.0


def test_total_variation_zero_for_constant():
    x = torch.ones(1, 1, 4, 4)
    assert total_variation(x).item() == pytest.approx(0.0)


def test_total_variation_positive_for_non_constant():
    x = torch.zeros(1, 1, 4, 4)
    x[0, 0, 0, 0] = 1.0
    assert total_variation(x).item() > 0.0


def test_total_variation_known_value():
    # 1x1x2x2 tensor: [[0,1],[0,0]] → tv_h=(|0-0|+|1-0|)/2=0.5, tv_w=(|0-1|+|0-0|)/2=0.5 → 1.0
    x = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])
    assert total_variation(x).item() == pytest.approx(1.0)


def test_grad_match_l2_identical_returns_zero():
    g = [torch.tensor([1.0, 2.0, 3.0])]
    loss, n = compute_grad_match_loss(g, g, grad_loss="l2")
    assert float(loss) == pytest.approx(0.0)
    assert n == 3


def test_grad_match_l2_known_diff():
    g1 = [torch.tensor([0.0, 0.0])]
    g2 = [torch.tensor([3.0, 4.0])]
    loss, n = compute_grad_match_loss(g1, g2, grad_loss="l2")
    # (3^2 + 4^2) = 25
    assert float(loss) == pytest.approx(25.0)
    assert n == 2


def test_grad_match_cos_identical_returns_zero():
    g = [torch.tensor([1.0, 2.0, 3.0])]
    loss, n = compute_grad_match_loss(g, g, grad_loss="cos")
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert n == 3


def test_grad_match_cos_orthogonal_returns_one():
    g1 = [torch.tensor([1.0, 0.0])]
    g2 = [torch.tensor([0.0, 1.0])]
    loss, _ = compute_grad_match_loss(g1, g2, grad_loss="cos")
    assert float(loss) == pytest.approx(1.0, abs=1e-6)


def test_grad_match_with_entry_masks_l2():
    g1 = [torch.tensor([1.0, 2.0, 3.0])]
    g2 = [torch.tensor([1.0, 5.0, 3.0])]
    # mask selects indices 0 and 2 (skips index 1 where values differ)
    m = [torch.tensor([True, False, True])]
    loss, n = compute_grad_match_loss(g1, g2, selected_entry_masks=m, grad_loss="l2")
    assert float(loss) == pytest.approx(0.0)
    assert n == 2


def test_grad_match_unknown_grad_loss_raises():
    g = [torch.tensor([1.0])]
    with pytest.raises(ValueError, match="Unknown grad_loss"):
        compute_grad_match_loss(g, g, grad_loss="mse")


def test_grad_match_cos_with_entry_masks():
    g1 = [torch.tensor([1.0, 99.0])]
    g2 = [torch.tensor([1.0, -1.0])]
    m = [torch.tensor([True, False])]  # select only the matching entry
    loss, n = compute_grad_match_loss(g1, g2, selected_entry_masks=m, grad_loss="cos")
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert n == 1


def test_grad_match_cos_empty_selection_raises():
    g = [torch.tensor([1.0, 2.0])]
    m = [torch.tensor([False, False])]
    with pytest.raises(ValueError, match="No gradient entries selected"):
        compute_grad_match_loss(g, g, selected_entry_masks=m, grad_loss="cos")
