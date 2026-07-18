"""Unit tests for ``cliffordstf.training.ema.ExponentialMovingAverage``."""

from __future__ import annotations

import torch
from torch import nn

from cliffordstf.training import ExponentialMovingAverage
from cliffordstf.training.ema import ExponentialMovingAverage as EMAFromModule


def test_re_export_matches_module():
    """``training/__init__.py`` re-exports the same class."""
    assert ExponentialMovingAverage is EMAFromModule


def test_init_snapshots_trainable_parameters():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    ema = ExponentialMovingAverage(model, decay=0.9)

    expected = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    assert set(ema.shadow.keys()) == set(expected.keys())
    for k, v in expected.items():
        assert torch.equal(ema.shadow[k], v)


def test_init_skips_non_trainable_parameters():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    model.bias.requires_grad_(False)
    ema = ExponentialMovingAverage(model, decay=0.9)

    assert "weight" in ema.shadow
    assert "bias" not in ema.shadow


def test_update_blends_shadow_with_decay():
    """After one ``update`` call the shadow should be ``decay * shadow + (1 - decay) * live``."""
    torch.manual_seed(0)
    model = nn.Linear(4, 3, bias=False)
    decay = 0.9
    ema = ExponentialMovingAverage(model, decay=decay)

    initial_shadow = ema.shadow["weight"].clone()
    new_weight = torch.full_like(model.weight, 5.0)
    with torch.no_grad():
        model.weight.copy_(new_weight)

    ema.update()
    expected = decay * initial_shadow + (1.0 - decay) * new_weight
    assert torch.allclose(ema.shadow["weight"], expected)


def test_apply_shadow_swaps_live_weights_and_backs_up_originals():
    torch.manual_seed(0)
    model = nn.Linear(4, 3, bias=False)
    ema = ExponentialMovingAverage(model, decay=0.9)

    new_weight = torch.full_like(model.weight, 7.0)
    with torch.no_grad():
        model.weight.copy_(new_weight)

    ema.apply_shadow()

    assert torch.equal(model.weight.data, ema.shadow["weight"])
    assert torch.equal(ema.backup["weight"], new_weight)


def test_restore_recovers_pre_apply_state():
    torch.manual_seed(0)
    model = nn.Linear(4, 3, bias=False)
    ema = ExponentialMovingAverage(model, decay=0.9)

    new_weight = torch.full_like(model.weight, 7.0)
    with torch.no_grad():
        model.weight.copy_(new_weight)

    ema.apply_shadow()
    ema.restore()

    assert torch.equal(model.weight.data, new_weight)
    assert ema.backup == {}
