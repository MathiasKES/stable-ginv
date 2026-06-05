# Phase 1 — ExperimentConfig Dataclass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the UPPERCASE config dict built in `iDLG_mask.py` and unpacked in `run_single_exp.py` with a frozen, picklable `ExperimentConfig` dataclass — identical behavior, zero numeric drift.

**Architecture:** Create `stable_ginv/config.py` with the dataclass (Task 1). Then atomically migrate all three call sites — `iDLG_mask.py`, `run_single_exp.py`, and `tests/golden/helpers.py` — in one commit so the test suite is never broken mid-migration (Task 2). Verify + update handover (Task 3). No golden fixtures are regenerated; the existing 83-test suite is the correctness proof.

**Tech Stack:** Python 3.9+, `dataclasses` stdlib, pytest. Conda env `stable-ginv`. Run all commands via `conda run -n stable-ginv`.

---

## Conventions

- Run all Python via: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && conda run -n stable-ginv <cmd>`
- Work on branch `cleanup/submit-ready`.
- Commits: `refactor:` for code moves/OOP, `docs:` for handover. No `feat:`.
- Do NOT edit `iDLG_mask.py` or `run_single_exp.py` without also updating the other (they are tightly coupled via the config type).
- After every task: `conda run -n stable-ginv python -m pytest tests/ -q` stays green.

## File structure created/modified in this phase

- Create: `stable_ginv/config.py` — `ExperimentConfig` frozen dataclass
- Create: `tests/test_config.py` — unit tests (construction, frozen, picklable, replace)
- Modify: `iDLG_mask.py:183-213` — build `ExperimentConfig` instead of dict; `iDLG_mask.py:280,303` — use `dataclasses.replace`
- Modify: `run_single_exp.py:21,42-71` — attribute access instead of dict access
- Modify: `tests/golden/helpers.py` — `lenet_worker_config()` returns `ExperimentConfig`
- Modify: `docs/handover/HANDOVER_RESTRUCTURE.md` — mark Phase 1 complete

---

## Task 1: ExperimentConfig dataclass + unit tests

**Files:**
- Create: `stable_ginv/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing unit tests**

`tests/test_config.py`:
```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

Run:
```bash
conda run -n stable-ginv python -m pytest tests/test_config.py -v
```
Expected: 4 errors — `ModuleNotFoundError: No module named 'stable_ginv.config'`.

- [ ] **Step 3: Implement ExperimentConfig**

`stable_ginv/config.py`:
```python
"""ExperimentConfig: frozen dataclass replacing the UPPERCASE config dict."""
from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Frozen, picklable experiment configuration.

    Replaces the dict built in iDLG_mask.py and unpacked in run_single_exp.py.
    Dataset-derived fields (channel, num_classes, shape_img) have no defaults
    and must be provided from load_dataset().  All other defaults match
    idlg_cli.py argument defaults.

    single_restart_idx is None in the base config; parallel-restart dispatch
    sets it via dataclasses.replace(config, single_restart_idx=r_i).
    """

    # --- dataset-derived (required — set by load_dataset) ---
    channel: int
    num_classes: int
    shape_img: tuple          # (H, W)

    # --- CLI-derived (defaults match idlg_cli.py) ---
    lr: float = 1.0
    num_dummy: int = 1
    iteration: int = 1000
    run_id: int = 0
    mask_mode: str = "gradsize_topfrac"
    prefixes: tuple = ()
    prefix_layer_fracs: dict = dataclasses.field(default_factory=dict)
    gradsize_topk: Optional[int] = 20
    gradsize_topfrac: float = 0.5
    gradsize_metric: str = "l2"
    grad_loss: str = "cos"
    gamma: float = 0.5
    network_name: str = "LeNet"
    methods: str = "idlg"
    compute_jacobian_rank: bool = False
    jacobian_max_entries: int = 4000
    jacobian_select_mode: str = "topk_abs"
    tv_weight: float = 0.0
    optimizer: str = "lbfgs"
    num_restarts: int = 1
    max_iteration: int = 20
    history_size: int = 100
    network_trained: bool = False
    save_gif: bool = False
    frame_interval: int = 20
    out_path: Optional[str] = None
    single_restart_idx: Optional[int] = None
```

- [ ] **Step 4: Run tests to confirm they pass**

Run:
```bash
conda run -n stable-ginv python -m pytest tests/test_config.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Confirm the full suite is still green**

Run:
```bash
conda run -n stable-ginv python -m pytest tests/ -q
```
Expected: 87 passed (83 existing + 4 new config tests).

- [ ] **Step 6: Commit**

```bash
git add stable_ginv/config.py tests/test_config.py
git commit -m "refactor: add ExperimentConfig frozen dataclass"
```

---

## Task 2: Migrate all call sites atomically

`iDLG_mask.py`, `run_single_exp.py`, and `tests/golden/helpers.py` must be updated together — the golden test calls `_run_inner` directly with the config from `lenet_worker_config()`, so both sides must be consistent. Do all three edits before running the test suite.

**Files:**
- Modify: `iDLG_mask.py`
- Modify: `run_single_exp.py`
- Modify: `tests/golden/helpers.py`

- [ ] **Step 1: Add imports and replace config dict in iDLG_mask.py**

At the top of `iDLG_mask.py`, add these two imports (after the existing imports):
```python
import dataclasses
from stable_ginv.config import ExperimentConfig
```

Then replace the config dict block (currently lines 182–213):
```python
    # Prepare config to pass to workers
    config = {
        'channel': channel,
        ...
        'out_path': out_path,
    }
```
with:
```python
    config = ExperimentConfig(
        channel=channel,
        num_classes=num_classes,
        shape_img=shape_img,
        lr=lr,
        num_dummy=num_dummy,
        iteration=Iteration,
        run_id=run_id,
        mask_mode=MASK_MODE,
        prefixes=PREFIXES,
        prefix_layer_fracs=PREFIX_LAYER_FRACS,
        gradsize_topk=GRADSIZE_TOPK,
        gradsize_topfrac=GRADSIZE_TOPFRAC,
        gradsize_metric=GRADSIZE_METRIC,
        grad_loss=GRAD_LOSS,
        gamma=GAMMA,
        network_name=NETWORK_NAME,
        methods=METHODS,
        compute_jacobian_rank=COMPUTE_JACOBIAN_RANK,
        jacobian_max_entries=JACOBIAN_MAX_ENTRIES,
        jacobian_select_mode=JACOBIAN_SELECT_MODE,
        tv_weight=TV_WEIGHT,
        optimizer=OPTIMIZER,
        num_restarts=NUM_RESTARTS,
        max_iteration=MAX_ITERATION,
        history_size=HISTORY_SIZE,
        network_trained=NETWORK_TRAINED,
        save_gif=SAVE_GIF,
        frame_interval=FRAME_INTERVAL,
        out_path=out_path,
    )
```

- [ ] **Step 2: Replace parallel-restart dict spread in iDLG_mask.py**

There are two identical lines (currently around lines 280 and 303):
```python
task_cfg = {**config, 'SINGLE_RESTART_IDX': r_i}
```
Replace both with:
```python
task_cfg = dataclasses.replace(config, single_restart_idx=r_i)
```

- [ ] **Step 3: Migrate config access in run_single_exp.py**

In `run_single_experiment` (the outer function), replace:
```python
    setstdout(path=config.get('out_path'))
```
with:
```python
    setstdout(path=config.out_path)
```

In `_run_inner`, replace the entire "Unpack config" block (currently lines 42–71):
```python
    # Unpack config
    channel = config['channel']
    num_classes = config['num_classes']
    shape_img = config['shape_img']
    lr = config['lr']
    GAMMA = config['GAMMA']
    num_dummy = config['num_dummy']
    Iteration = config['Iteration']
    MASK_MODE = config['MASK_MODE']
    PREFIXES = config.get('PREFIXES', ())
    PREFIX_LAYER_FRACS = config.get('PREFIX_LAYER_FRACS', {})
    GRADSIZE_TOPK = config['GRADSIZE_TOPK']
    GRADSIZE_TOPFRAC = config['GRADSIZE_TOPFRAC']
    GRADSIZE_METRIC = config['GRADSIZE_METRIC']
    NETWORK_NAME = config['NETWORK_NAME']
    NETWORK_TRAINED = config['NETWORK_TRAINED']
    METHODS = config.get('METHODS', 'both')
    COMPUTE_JACOBIAN_RANK = config.get('COMPUTE_JACOBIAN_RANK', False)
    JACOBIAN_MAX_ENTRIES = config.get('JACOBIAN_MAX_ENTRIES', 4000)
    JACOBIAN_SELECT_MODE = config.get('JACOBIAN_SELECT_MODE', 'topk_abs')
    TV_WEIGHT = config.get('TV_WEIGHT', 0.0)
    OPTIMIZER = config.get('OPTIMIZER', 'lbfgs')
    NUM_RESTARTS = config.get('NUM_RESTARTS', 1)
    SINGLE_RESTART_IDX = config.get('SINGLE_RESTART_IDX')  # None = run all restarts
    MAX_ITERATION = config.get('MAX_ITERATION', 20)
    HISTORY_SIZE = config.get('HISTORY_SIZE', 100)
    SAVE_GIF = config.get('SAVE_GIF', False)
    FRAME_INTERVAL = config.get('FRAME_INTERVAL', 20)
    GRAD_LOSS = config.get('GRAD_LOSS', 'cos').lower()

    seed = config.get("run_id", 0) + idx_net + 1
```
with:
```python
    # Unpack config
    channel = config.channel
    num_classes = config.num_classes
    shape_img = config.shape_img
    lr = config.lr
    GAMMA = config.gamma
    num_dummy = config.num_dummy
    Iteration = config.iteration
    MASK_MODE = config.mask_mode
    PREFIXES = config.prefixes
    PREFIX_LAYER_FRACS = config.prefix_layer_fracs
    GRADSIZE_TOPK = config.gradsize_topk
    GRADSIZE_TOPFRAC = config.gradsize_topfrac
    GRADSIZE_METRIC = config.gradsize_metric
    NETWORK_NAME = config.network_name
    NETWORK_TRAINED = config.network_trained
    METHODS = config.methods
    COMPUTE_JACOBIAN_RANK = config.compute_jacobian_rank
    JACOBIAN_MAX_ENTRIES = config.jacobian_max_entries
    JACOBIAN_SELECT_MODE = config.jacobian_select_mode
    TV_WEIGHT = config.tv_weight
    OPTIMIZER = config.optimizer
    NUM_RESTARTS = config.num_restarts
    SINGLE_RESTART_IDX = config.single_restart_idx  # None = run all restarts
    MAX_ITERATION = config.max_iteration
    HISTORY_SIZE = config.history_size
    SAVE_GIF = config.save_gif
    FRAME_INTERVAL = config.frame_interval
    GRAD_LOSS = config.grad_loss.lower()

    seed = config.run_id + idx_net + 1
```

- [ ] **Step 4: Update lenet_worker_config() in tests/golden/helpers.py**

Add `from stable_ginv.config import ExperimentConfig` at the top of the imports in `helpers.py`.

Then replace the `lenet_worker_config` function body:
```python
def lenet_worker_config(method="idlg"):
    ...
    return {
        "channel": 1,
        ...
    }
```
with:
```python
def lenet_worker_config(method="idlg"):
    """Faithful worker config (ExperimentConfig) for golden characterization tests.

    LeNet + no normalization keeps the recon path deterministic and free of
    dataset-normalization constants. method='idlg' exercises the unmasked
    reconstruction only.
    """
    return ExperimentConfig(
        channel=1,
        num_classes=10,
        shape_img=(28, 28),
        lr=1.0,
        num_dummy=1,
        iteration=10,
        run_id=0,
        mask_mode="gradsize_topfrac",
        prefixes=(),
        prefix_layer_fracs={},
        gradsize_topk=20,
        gradsize_topfrac=0.5,
        gradsize_metric="l2",
        grad_loss="cos",
        gamma=0.5,
        network_name="LeNet",
        methods=method,
        compute_jacobian_rank=False,
        jacobian_max_entries=4000,
        jacobian_select_mode="topk_abs",
        tv_weight=0.0,
        optimizer="lbfgs",
        num_restarts=1,
        max_iteration=20,
        history_size=100,
        network_trained=False,
        save_gif=False,
        frame_interval=20,
        out_path=None,
    )
```

- [ ] **Step 5: Run the full test suite**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
```
Expected: `87 passed`. If a golden test fails, the numeric output changed — do NOT regenerate the fixture; diagnose the regression.

- [ ] **Step 6: Verify CLI entry point still works**

Run:
```bash
conda run -n stable-ginv python iDLG_mask.py --help >/dev/null && echo "CLI OK"
```
Expected: `CLI OK`.

- [ ] **Step 7: Lint the migrated modules**

Run:
```bash
conda run -n stable-ginv python -m pyflakes stable_ginv iDLG_mask.py run_single_exp.py tests/golden/helpers.py tests/test_config.py
```
Expected: no output (no warnings). If pyflakes warns about an unused `config.get` import or similar, fix it.

- [ ] **Step 8: Commit**

```bash
git add stable_ginv/config.py iDLG_mask.py run_single_exp.py tests/golden/helpers.py tests/test_config.py
git commit -m "refactor: migrate config dict to ExperimentConfig across all call sites"
```

---

## Task 3: Final verification + handover update

**Files:**
- Modify: `docs/handover/HANDOVER_RESTRUCTURE.md`

- [ ] **Step 1: Run the full Phase 1 verification**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
conda run -n stable-ginv python -m pyflakes stable_ginv tests/golden tests/test_config.py
conda run -n stable-ginv python iDLG_mask.py --help >/dev/null && echo "CLI OK"
conda run -n stable-ginv python -m sphinx -b html docs/sphinx docs/sphinx/_build/html >/dev/null && echo "DOCS OK"
git diff --check
```
Expected: `87 passed`, `CLI OK`, `DOCS OK`, no whitespace errors. (Pre-existing `CITATION.cff` trailing whitespace is out of scope.)

- [ ] **Step 2: Update HANDOVER_RESTRUCTURE.md**

In `docs/handover/HANDOVER_RESTRUCTURE.md`, under `## Status`, change:
```markdown
- [ ] Phase 1 — `ExperimentConfig` dataclass replaces the config dict.
```
to:
```markdown
- [x] Phase 1 — `ExperimentConfig` dataclass replaces the config dict.
```

Also update `## Next phase`:
```markdown
## Next phase

Write the Phase 2 plan from the spec, then implement. Keep this file's Status and
Golden-harness sections current at every phase boundary.
```

- [ ] **Step 3: Commit**

```bash
git add docs/handover/HANDOVER_RESTRUCTURE.md
git commit -m "docs: mark Phase 1 complete in HANDOVER_RESTRUCTURE"
```

---

## Self-review notes (filled during planning)

- **Spec coverage:** Phase 1 spec says "`ExperimentConfig` dataclass replaces the config dict in `iDLG_mask.py` / `run_single_exp.py`; outputs identical." All three tasks implement exactly this — no extra scope.
- **Atomic migration:** Tasks 2 and 3 within the old plan are merged into one task because `run_single_exp._run_inner` is called directly by the recon-worker golden test with config from `helpers.py`. Splitting the migration would temporarily break 1 of the 83 tests.
- **No new goldens:** Phase 0 already locked masking, registry-key, and recon-worker numerics. Phase 1 only changes the config *type*, not the values passed — the existing 83 tests are sufficient proof.
- **SINGLE_RESTART_IDX:** Previously injected via `{**config, 'SINGLE_RESTART_IDX': r_i}`. With a frozen dataclass this becomes `dataclasses.replace(config, single_restart_idx=r_i)`, which is safer (typos are AttributeErrors at import time, not silent KeyErrors at runtime).
- **Naming:** Internal field names are snake_case (`iteration`, `mask_mode`, etc.). Local variable names in `run_single_exp.py` keep their old uppercase names (e.g., `Iteration = config.iteration`) to minimize diff noise and reduce risk in the worker logic.
- **dict field hashing:** `prefix_layer_fracs: dict` means `hash(config)` would raise TypeError. We never hash `ExperimentConfig` (multiprocessing uses pickle, not hash), so this is fine. Documented implicitly by the `test_picklable` test.
