import dataclasses
import pickle
import pytest
from stable_ginv.config import ExperimentConfig


def _minimal():
    return ExperimentConfig(channel=1, num_classes=10, shape_img=(28, 28))


def test_construction_with_defaults():
    cfg = _minimal()
    assert cfg.channel == 1
    assert cfg.num_classes == 10
    assert cfg.shape_img == (28, 28)
    assert cfg.iteration == 1000
    assert cfg.mask_mode == "gradsize_topfrac"
    assert cfg.single_restart_idx is None
    assert cfg.prefix_layer_fracs == {}
    assert cfg.prefixes == ()


def test_frozen():
    cfg = _minimal()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.channel = 2


def test_picklable():
    cfg = ExperimentConfig(
        channel=1, num_classes=10, shape_img=(28, 28),
        prefix_layer_fracs={"body.0": 0.5},
    )
    assert pickle.loads(pickle.dumps(cfg)) == cfg


def test_replace_single_restart_idx():
    cfg = _minimal()
    task_cfg = dataclasses.replace(cfg, single_restart_idx=3)
    assert task_cfg.single_restart_idx == 3
    assert cfg.single_restart_idx is None
