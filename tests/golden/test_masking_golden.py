"""Golden test: every mask mode yields identical keep_ids / entry_masks.

Hashes the masking output for a fixed LeNet + fixed gradients across all modes.
Refactoring the masking layer must not change any hash.
"""
import hashlib

import numpy as np
import torch

from stable_ginv.masking import build_gradient_mask
from helper.Network import get_model, weights_init
from tests.golden.helpers import load_or_regen

# All masking modes the experiment supports (see HANDOVER_RESEARCHER.md section 5).
# Prefix modes use a fixed prefix list valid for LeNet parameter names.
MODES = [
    ("gradsize_topk", {}),
    ("gradsize_topfrac", {}),
    ("gradsize_topk_entries", {}),
    ("gradsize_topfrac_entries", {}),
    ("gradsize_topk_entries_layer", {}),
    ("gradsize_topfrac_entries_layer", {}),
    # NOTE: with a single 2-tensor prefix (body.0), some prefix modes collapse to
    # the same selection and therefore share a hash: prefix_topk == prefix (topk>2),
    # and prefix_topk_entries(_layer) / prefix_topfrac_entries(_layer) coincide
    # because one prefix == one group. This is expected, not a bug.
    ("prefix", {"prefixes": ("body.0",)}),
    ("prefix_topk", {"prefixes": ("body.0",)}),
    ("prefix_topfrac", {"prefixes": ("body.0",)}),
    ("prefix_topk_entries", {"prefixes": ("body.0",)}),
    ("prefix_topfrac_entries", {"prefixes": ("body.0",)}),
    ("prefix_topk_entries_layer", {"prefixes": ("body.0",)}),
    ("prefix_topfrac_entries_layer", {"prefixes": ("body.0",)}),
]


def _fixed_gradients():
    torch.manual_seed(0)
    net = get_model("LeNet", channel=1, num_classes=10, input_size=(28, 28), pretrained=False)
    net.apply(weights_init)
    net.eval()
    x = torch.randn(1, 1, 28, 28)
    y = torch.tensor([3], dtype=torch.long)
    out = net(x)
    loss = torch.nn.CrossEntropyLoss()(out, y)
    grads = torch.autograd.grad(loss, net.parameters())
    return net, [g.detach().clone() for g in grads]


def _hash_mask(keep_ids, entry_masks):
    h = hashlib.md5()
    if keep_ids is not None:
        h.update(b"keep:")
        h.update(",".join(str(i) for i in sorted(keep_ids)).encode())
    if entry_masks is not None:
        h.update(b"masks:")
        for i, m in enumerate(entry_masks):
            if m is None:
                h.update(f"{i}:None;".encode())
            else:
                h.update(f"{i}:".encode())
                h.update(m.detach().cpu().numpy().astype(np.uint8).tobytes())
                h.update(b";")
    return h.hexdigest()


def _produce():
    net, grads = _fixed_gradients()
    result = {}
    for mode, kw in MODES:
        keep_ids, entry_masks = build_gradient_mask(
            "masked", mode, net, grads,
            prefixes=kw.get("prefixes", ()),
            prefix_layer_fracs=kw.get("prefix_layer_fracs"),
            gradsize_topk=20, gradsize_topfrac=0.5, gradsize_metric="l2",
        )
        result[mode] = _hash_mask(keep_ids, entry_masks)
    return result


def test_masking_modes_are_stable():
    golden = load_or_regen("masking_golden.json", _produce)
    current = _produce()
    for mode in golden:
        assert current[mode] == golden[mode], (
            f"Mask hash changed for mode={mode!r}: "
            f"golden={golden[mode]!r} current={current[mode]!r}"
        )
