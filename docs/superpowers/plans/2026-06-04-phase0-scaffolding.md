# Phase 0 — Scaffolding & Golden Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `stable_ginv` package skeleton, a hermetic golden-characterization test harness, Sphinx→GitHub Pages docs, and a restructure handover doc — without moving or changing any existing code.

**Architecture:** Phase 0 of the incremental strangler restructure described in `docs/superpowers/specs/2026-06-04-code-structure-design.md`. It adds new files only (package skeleton, tests, docs, CI). No existing module is refactored. The golden harness locks current behavior so later phases can prove `diff == 0`.

**Tech Stack:** Python 3, PyTorch (CPU), pytest, setuptools/pyproject, Sphinx (autodoc + napoleon + autosummary), GitHub Actions + Pages. Conda env `stable-ginv`.

---

## Conventions for every task

- Run all Python via the project env, with the HPC library path exported:
  ```bash
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
  conda run -n stable-ginv <cmd>
  ```
- Work on branch `cleanup/submit-ready`.
- Commits are categorized (no `feat:`): `build:`, `test:`, `docs:`, `chore:`.
- NEVER edit existing production modules in Phase 0. New files only.
- After each task: `conda run -n stable-ginv python -m pytest tests/ -q` stays green.

## File structure created in this phase

- Create: `pyproject.toml` — editable install of `stable_ginv` only.
- Create: `stable_ginv/__init__.py` — top package with `__version__`.
- Create: `tests/golden/__init__.py`, `tests/golden/fixtures/` — golden fixtures dir.
- Create: `tests/golden/helpers.py` — shared deterministic builders (fake dataset, config).
- Create: `tests/golden/test_masking_golden.py` — Tier 2 (all mask modes).
- Create: `tests/golden/test_registry_key_golden.py` — Tier 1 (key stability across mask modes).
- Create: `tests/golden/test_recon_worker_golden.py` — Tier 3 (recon numerics, in-process).
- Create: `docs/sphinx/conf.py`, `docs/sphinx/index.rst`, `docs/sphinx/requirements.txt`.
- Create: `.github/workflows/docs.yml` — build + deploy Pages.
- Create: `docs/handover/HANDOVER_RESTRUCTURE.md`; Modify: `docs/handover/README.md` (add to reading order).

---

## Task 1: Package skeleton + editable install

**Files:**
- Create: `stable_ginv/__init__.py`
- Create: `pyproject.toml`

- [ ] **Step 1: Create the package init**

`stable_ginv/__init__.py`:
```python
"""stable_ginv: gradient-inversion masking experiments.

Target home for the OOP restructure (see
docs/superpowers/specs/2026-06-04-code-structure-design.md). During the
incremental migration this package grows phase by phase; existing top-level
modules (functions/, helper/) keep working until each is migrated.
"""

__version__ = "0.0.0"
```

- [ ] **Step 2: Create pyproject.toml (explicit single-package config)**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "stable-ginv"
version = "0.0.0"
description = "Gradient-inversion masking experiments (stable-ginv)"
requires-python = ">=3.9"

# Dependencies are managed by the conda env (env/environment.yml); this project
# table intentionally omits runtime deps so `pip install -e .` does not touch the
# conda-provided torch/scipy/etc.

[tool.setuptools]
# Explicit list prevents auto-discovery from grabbing functions/, helper/, tests/.
# Those stay importable via the repo root on sys.path until migrated.
packages = ["stable_ginv"]
```

- [ ] **Step 3: Editable install into the conda env**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pip install -e . --no-deps --no-build-isolation
```
Expected: `Successfully installed stable-ginv-0.0.0`.

- [ ] **Step 4: Verify import + existing modules still import**

Run:
```bash
conda run -n stable-ginv python -c "import stable_ginv; print(stable_ginv.__version__)"
conda run -n stable-ginv python -c "import functions.io_utils, helper.metrics; print('legacy imports OK')"
```
Expected: prints `0.0.0` then `legacy imports OK`.

- [ ] **Step 5: Verify the suite + entry points still work**

Run:
```bash
conda run -n stable-ginv python -m pytest tests/ -q
conda run -n stable-ginv python iDLG_mask.py --help >/dev/null && echo "CLI OK"
```
Expected: `80 passed`, then `CLI OK`.

- [ ] **Step 6: Commit**

```bash
git add stable_ginv/__init__.py pyproject.toml
git commit -m "build: add stable_ginv package skeleton and editable install"
```

---

## Task 2: Golden harness shared helpers

Deterministic builders shared by Tier 2/3 tests: a CI-portable synthetic
dataset (no `/work3` or downloaded data) and a faithful worker config dict.

**Files:**
- Create: `tests/golden/__init__.py` (empty)
- Create: `tests/golden/fixtures/.gitkeep` (empty, keeps the dir in git)
- Create: `tests/golden/helpers.py`

- [ ] **Step 1: Create the package marker files**

`tests/golden/__init__.py`: empty file.
`tests/golden/fixtures/.gitkeep`: empty file.

- [ ] **Step 2: Write the shared helpers**

`tests/golden/helpers.py`:
```python
"""Deterministic, CI-portable builders for golden characterization tests.

No dependency on /work3, downloaded datasets, or a GPU. Everything here is
seeded so the same inputs reproduce byte-for-byte across runs on one machine.
"""
import json
import os

import numpy as np
import torch

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# Set GOLDEN_REGEN=1 to (re)write fixtures instead of asserting against them.
REGEN = os.environ.get("GOLDEN_REGEN") == "1"


class FakeImageDataset:
    """Minimal dataset matching the worker's `dst[i] -> (HxWxC uint8 array, int)`.

    torchvision ToTensor() converts an (H, W, C) uint8 array to a [C, H, W]
    float tensor in [0, 1], which is exactly what run_single_exp expects.
    """

    def __init__(self, n, channel, size, num_classes, seed=0):
        rng = np.random.default_rng(seed)
        h, w = size
        self.images = [
            rng.integers(0, 256, size=(h, w, channel), dtype=np.uint8) for _ in range(n)
        ]
        self.labels = [int(rng.integers(0, num_classes)) for _ in range(n)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


def lenet_worker_config(method="idlg"):
    """Faithful worker config dict (same keys as iDLG_mask.py builds).

    LeNet + no normalization keeps the recon path deterministic and free of
    dataset-normalization constants. METHODS='idlg' exercises the unmasked
    reconstruction only (masking is covered by the Tier 2 masking goldens).
    """
    return {
        "channel": 1,
        "num_classes": 10,
        "shape_img": (28, 28),
        "lr": 1.0,
        "num_dummy": 1,
        "Iteration": 10,
        "run_id": 0,
        "MASK_MODE": "gradsize_topfrac",   # unused when METHODS='idlg'
        "PREFIXES": (),
        "PREFIX_LAYER_FRACS": {},
        "GRADSIZE_TOPK": 20,
        "GRADSIZE_TOPFRAC": 0.5,
        "GRADSIZE_METRIC": "l2",
        "GRAD_LOSS": "cos",
        "GAMMA": 0.5,
        "NETWORK_NAME": "LeNet",
        "METHODS": method,
        "COMPUTE_JACOBIAN_RANK": False,
        "JACOBIAN_MAX_ENTRIES": 4000,
        "JACOBIAN_SELECT_MODE": "topk_abs",
        "TV_WEIGHT": 0.0,
        "OPTIMIZER": "lbfgs",
        "NUM_RESTARTS": 1,
        "MAX_ITERATION": 20,
        "HISTORY_SIZE": 100,
        "NETWORK_TRAINED": False,
        "SAVE_GIF": False,
        "FRAME_INTERVAL": 20,
        "out_path": None,
    }


def load_or_regen(name, produce):
    """Return the golden fixture `name`, regenerating it when GOLDEN_REGEN=1.

    `produce` is a zero-arg callable returning a JSON-serializable object.
    """
    path = os.path.join(FIXTURES_DIR, name)
    if REGEN or not os.path.exists(path):
        value = produce()
        with open(path, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
        return value
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 3: Sanity-check the helpers import and the fake dataset shape**

Run:
```bash
conda run -n stable-ginv python -c "
from torchvision import transforms
from tests.golden.helpers import FakeImageDataset
d = FakeImageDataset(4, channel=1, size=(28,28), num_classes=10)
t = transforms.ToTensor()(d[0][0])
print(tuple(t.shape), d[0][1])
"
```
Expected: `(1, 28, 28) <int>` (a 0–9 label).

- [ ] **Step 4: Commit**

```bash
git add tests/golden/__init__.py tests/golden/fixtures/.gitkeep tests/golden/helpers.py
git commit -m "test: add golden harness helpers (fake dataset + worker config)"
```

---

## Task 3: Tier 2 — masking golden (all modes)

Locks `keep_ids`/`entry_masks` for every mask mode. This is the primary guard
for the Phase 3 strategy-class extraction.

**Files:**
- Create: `tests/golden/test_masking_golden.py`
- Generates: `tests/golden/fixtures/masking_golden.json`

- [ ] **Step 1: Write the masking golden test**

`tests/golden/test_masking_golden.py`:
```python
"""Golden test: every mask mode yields identical keep_ids / entry_masks.

Hashes the masking output for a fixed LeNet + fixed gradients across all modes.
Refactoring the masking layer (Phase 3) must not change any hash.
"""
import hashlib

import numpy as np
import torch

from functions.masking import build_gradient_mask
from helper.Network import get_model, weights_init
from tests.golden.helpers import load_or_regen

# All masking modes the experiment supports (see HANDOVER_RESEARCHER.md §5).
# Prefix modes use a fixed prefix list valid for LeNet parameter names.
MODES = [
    ("gradsize_topk", {}),
    ("gradsize_topfrac", {}),
    ("gradsize_topk_entries", {}),
    ("gradsize_topfrac_entries", {}),
    ("gradsize_topk_entries_layer", {}),
    ("gradsize_topfrac_entries_layer", {}),
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
    np.random.seed(0)
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
    assert current == golden
```

- [ ] **Step 2: Confirm the LeNet prefix is valid (adjust if needed)**

The prefix `"body.0"` assumes LeNet's first parameter name starts with `body.0`.
Run:
```bash
conda run -n stable-ginv python -c "
from helper.Network import get_model
net = get_model('LeNet', channel=1, num_classes=10, input_size=(28,28))
print([n for n,_ in net.named_parameters()][:4])
"
```
Expected: a list of parameter names. If they do NOT start with `body.0`, replace
`('body.0',)` in the test with a prefix matching the first printed name (e.g.
`('body.0',)`, `('conv',)`), then continue. Do not guess — use the printed name.

- [ ] **Step 3: Generate fixtures**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
GOLDEN_REGEN=1 conda run -n stable-ginv python -m pytest tests/golden/test_masking_golden.py -q
```
Expected: PASS, and `tests/golden/fixtures/masking_golden.json` now exists with 13 entries.

- [ ] **Step 4: Run without regen to verify stability**

Run:
```bash
conda run -n stable-ginv python -m pytest tests/golden/test_masking_golden.py -q
```
Expected: PASS (reads committed fixture, recomputes, matches).

- [ ] **Step 5: Commit**

```bash
git add tests/golden/test_masking_golden.py tests/golden/fixtures/masking_golden.json
git commit -m "test: add masking golden (all modes) to lock keep_ids/entry_masks"
```

---

## Task 4: Tier 1 — registry-key golden across mask modes

The existing `tests/test_registry_keys.py` pins one masked/baseline key. This
adds a sweep so every `mask_mode` value's key is locked before the registry
refactor (Phase 4).

**Files:**
- Create: `tests/golden/test_registry_key_golden.py`
- Generates: `tests/golden/fixtures/registry_key_golden.json`

- [ ] **Step 1: Write the test**

`tests/golden/test_registry_key_golden.py`:
```python
"""Golden test: masked registry keys are stable across all mask modes.

Guards functions/io_utils.masked_key_from_args against accidental changes to the
comparable-args dict or its JSON serialization during the Phase 4 registry split.
"""
from types import SimpleNamespace

from functions.io_utils import masked_key_from_args
from tests.golden.helpers import load_or_regen

MASK_MODES = [
    "gradsize_topk", "gradsize_topfrac", "gradsize_topk_entries",
    "gradsize_topfrac_entries", "gradsize_topk_entries_layer",
    "gradsize_topfrac_entries_layer", "prefix", "prefix_topk", "prefix_topfrac",
    "prefix_topk_entries", "prefix_topfrac_entries", "prefix_topk_entries_layer",
    "prefix_topfrac_entries_layer", "none",
]


def _args(mask_mode):
    return SimpleNamespace(
        dataset="cifar100", network="vgg13", pretrained=False, lr=0.1, gamma=0.5,
        grad_loss="cos", num_dummy=1, iteration=5000, tv_weight=0.0,
        optimizer="signed_adamw", num_restarts=1, max_iteration=20, history_size=100,
        mask_mode=mask_mode, gradsize_topk=None, gradsize_topfrac=0.1,
        gradsize_metric="abs", prefixes="conv1:0.5,fc", num_exp=30, run_id=0,
    )


def _produce():
    return {mode: masked_key_from_args(_args(mode))[0] for mode in MASK_MODES}


def test_masked_keys_stable_across_modes():
    golden = load_or_regen("registry_key_golden.json", _produce)
    assert _produce() == golden
```

- [ ] **Step 2: Generate fixtures**

Run:
```bash
GOLDEN_REGEN=1 conda run -n stable-ginv python -m pytest tests/golden/test_registry_key_golden.py -q
```
Expected: PASS; `tests/golden/fixtures/registry_key_golden.json` created with 14 keys.

- [ ] **Step 3: Verify stability**

Run:
```bash
conda run -n stable-ginv python -m pytest tests/golden/test_registry_key_golden.py -q
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/golden/test_registry_key_golden.py tests/golden/fixtures/registry_key_golden.json
git commit -m "test: add registry-key golden across all mask modes"
```

---

## Task 5: Tier 3 — recon-worker numeric golden (in-process, hermetic)

Locks reconstruction numerics by running the worker in-process on CPU and
snapshotting the returned metrics. No file writes, no `/work3`, CI-portable.

**Files:**
- Create: `tests/golden/test_recon_worker_golden.py`
- Generates: `tests/golden/fixtures/recon_worker_golden.json`

- [ ] **Step 1: Write the test**

`tests/golden/test_recon_worker_golden.py`:
```python
"""Golden test: in-process reconstruction worker numerics.

Calls run_single_exp._run_inner directly on CPU with a fixed seed/config and a
synthetic dataset, capturing the result dict via a stub queue. Locks recon
metrics so later phases (config dataclass, recon OOP) prove no numeric drift.

Tolerance: integer/label/shape fields are exact; floating metrics use rel=1e-4
to absorb platform float noise while still catching real logic changes (which
move metrics far more than 1e-4). If the torch/env version changes, regenerate:
    GOLDEN_REGEN=1 pytest tests/golden/test_recon_worker_golden.py
"""
import pytest

from torchvision import transforms  # noqa: F401  (ensures torchvision import path is healthy)

from run_single_exp import _run_inner
from tests.golden.helpers import FakeImageDataset, lenet_worker_config, load_or_regen


class _CaptureQueue:
    """Stand-in for mp.SimpleQueue: keeps the last put() payload."""

    def __init__(self):
        self.result = None

    def put(self, value):
        self.result = value


def _run_worker():
    dst = FakeImageDataset(16, channel=1, size=(28, 28), num_classes=10, seed=0)
    config = lenet_worker_config(method="idlg")
    queue = _CaptureQueue()
    _run_inner(0, 0, dst, "MNIST", config, queue)
    return queue.result


def _produce():
    r = _run_worker()
    return {
        "label_iDLG": r["label_iDLG"],
        "loss_iDLG": r["loss_iDLG"],
        "mse_iDLG": r["mse_iDLG"],
        "psnr_idlg": r["psnr_idlg"],
        "early_stop_reason": r["early_stop_reason"].get("iDLG"),
    }


def test_recon_worker_numerics_stable():
    golden = load_or_regen("recon_worker_golden.json", _produce)
    current = _produce()
    assert current["label_iDLG"] == golden["label_iDLG"]
    assert current["early_stop_reason"] == golden["early_stop_reason"]
    for k in ("loss_iDLG", "mse_iDLG", "psnr_idlg"):
        assert current[k] == pytest.approx(golden[k], rel=1e-4), k
```

- [ ] **Step 2: Smoke-run the worker once to confirm the call shape**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "
from tests.golden.test_recon_worker_golden import _run_worker
r = _run_worker()
print('keys:', sorted(r))
print('label_iDLG:', r['label_iDLG'], 'loss_iDLG:', r['loss_iDLG'])
"
```
Expected: prints a result dict's keys including `loss_iDLG`, `mse_iDLG`,
`psnr_idlg`, `label_iDLG`, `early_stop_reason`. If `_run_inner` raises (e.g. a
config key mismatch), fix the config in `tests/golden/helpers.py::lenet_worker_config`
to match the keys read in `run_single_exp.py` before proceeding — do not change
`run_single_exp.py`.

- [ ] **Step 3: Generate fixtures**

Run:
```bash
GOLDEN_REGEN=1 conda run -n stable-ginv python -m pytest tests/golden/test_recon_worker_golden.py -q
```
Expected: PASS; `tests/golden/fixtures/recon_worker_golden.json` created.

- [ ] **Step 4: Verify stability across two runs**

Run:
```bash
conda run -n stable-ginv python -m pytest tests/golden/test_recon_worker_golden.py -q
conda run -n stable-ginv python -m pytest tests/golden/test_recon_worker_golden.py -q
```
Expected: PASS both times (deterministic on this machine).

- [ ] **Step 5: Commit**

```bash
git add tests/golden/test_recon_worker_golden.py tests/golden/fixtures/recon_worker_golden.json
git commit -m "test: add in-process recon-worker numeric golden"
```

---

## Task 6: Sphinx documentation skeleton

**Files:**
- Create: `docs/sphinx/conf.py`
- Create: `docs/sphinx/index.rst`
- Create: `docs/sphinx/requirements.txt`

- [ ] **Step 1: Sphinx config**

`docs/sphinx/conf.py`:
```python
"""Sphinx configuration for stable_ginv API docs (built to GitHub Pages)."""
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "stable-ginv"
author = "MathiasKES"
release = "0.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # NumPy/Google-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = True
autodoc_typehints = "description"

# The package is near-empty in Phase 0; autodoc must not fail the build if heavy
# optional imports are unavailable in the docs runner.
autodoc_mock_imports = ["torch", "torchvision", "scipy", "skimage", "seaborn",
                        "matplotlib", "numpy", "imageio", "PIL", "tqdm"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
```

`docs/sphinx/index.rst`:
```rst
stable-ginv
===========

Gradient-inversion masking experiments. This site hosts the API reference for
the ``stable_ginv`` package as the codebase is migrated to it phase by phase
(see ``docs/superpowers/specs/2026-06-04-code-structure-design.md``).

.. autosummary::
   :toctree: api
   :recursive:

   stable_ginv

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
```

`docs/sphinx/requirements.txt`:
```text
sphinx>=7
sphinx-rtd-theme>=2
```

- [ ] **Step 2: Build the docs locally**

Run:
```bash
conda run -n stable-ginv python -m pip install -r docs/sphinx/requirements.txt
conda run -n stable-ginv python -m sphinx -b html docs/sphinx docs/sphinx/_build/html -W --keep-going
```
Expected: `build succeeded`. (`-W` turns warnings into errors; `--keep-going`
reports all. If autosummary warns about the near-empty package, it should still
succeed because the package imports cleanly.)

- [ ] **Step 3: Ignore the build output**

Append to `.gitignore`:
```text
docs/sphinx/_build/
docs/sphinx/api/
```

- [ ] **Step 4: Commit**

```bash
git add docs/sphinx/conf.py docs/sphinx/index.rst docs/sphinx/requirements.txt .gitignore
git commit -m "docs: add Sphinx skeleton (autodoc + napoleon + RTD theme)"
```

---

## Task 7: GitHub Actions — build & deploy to Pages

**Files:**
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Write the workflow**

`.github/workflows/docs.yml`:
```yaml
name: Docs

on:
  push:
    branches: [main, cleanup/submit-ready]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install docs deps
        run: |
          python -m pip install --upgrade pip
          python -m pip install -r docs/sphinx/requirements.txt
          python -m pip install -e . --no-deps
      - name: Build Sphinx site
        run: python -m sphinx -b html docs/sphinx docs/sphinx/_build/html
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs/sphinx/_build/html

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Validate YAML locally**

Run:
```bash
conda run -n stable-ginv python -c "import yaml; yaml.safe_load(open('.github/workflows/docs.yml')); print('yaml OK')"
```
Expected: `yaml OK`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/docs.yml
git commit -m "build: add GitHub Actions workflow to deploy Sphinx docs to Pages"
```

- [ ] **Step 4: Note for the repo owner (manual, one-time)**

After this is pushed, the repo owner enables Pages: GitHub repo → Settings →
Pages → Source = "GitHub Actions". Record this in the handover (Task 8). The
mock-imports in `conf.py` mean the runner does not need torch installed.

---

## Task 8: Restructure handover doc

**Files:**
- Create: `docs/handover/HANDOVER_RESTRUCTURE.md`
- Modify: `docs/handover/README.md`

- [ ] **Step 1: Write the handover**

`docs/handover/HANDOVER_RESTRUCTURE.md`:
```markdown
# Handover: OOP Restructure

**Spec:** `docs/superpowers/specs/2026-06-04-code-structure-design.md`
**Plans:** `docs/superpowers/plans/`
**Branch:** `cleanup/submit-ready`

## Invariants (every phase)

- No functionality change: reconstruction math, masking behavior, optimizer
  behavior, CLI flags/defaults, registry key inputs/JSON, CSV
  filenames/columns/order/append, plot filenames/contents stay identical.
- Repo stays green and runnable after every phase.
- Categorized commits: `refactor:` / `test:` / `docs:` / `build:` / `style:`
  (no `feat:`). Separate commit per category.
- Goldens are added just-in-time: before refactoring an area, lock it with a
  golden test.

## Verify (run after any change)

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
conda run -n stable-ginv python iDLG_mask.py --help
git diff --check
```

Regenerate goldens only when the env (e.g. torch version) changes, never to make
a refactor pass:
```bash
GOLDEN_REGEN=1 conda run -n stable-ginv python -m pytest tests/golden/ -q
```

## Golden harness (Phase 0)

- `tests/golden/test_masking_golden.py` — all mask modes (keep_ids/entry_masks).
- `tests/golden/test_registry_key_golden.py` — masked keys across modes.
- `tests/golden/test_recon_worker_golden.py` — in-process recon numerics (CPU).
- Helpers + fixtures: `tests/golden/helpers.py`, `tests/golden/fixtures/`.

Note: the spec's "end-to-end golden" is realized in-process (recon-worker golden)
because `resolve_storage_paths()` resolves to the shared `/work3` tree locally, so
a full-CLI run would append to real experiment data. CSV-row goldens are added in
Phase 6, just before the `experiment_results` refactor.

## Status

- [x] Phase 0 — Scaffolding: package skeleton, `pyproject.toml`, golden harness,
      Sphinx + Pages CI, this handover.
- [ ] Phase 1 — `ExperimentConfig` dataclass replaces the config dict.
- [ ] Phase 2 — `stable_ginv/metrics/`.
- [ ] Phase 3 — `stable_ginv/masking/` strategy classes.
- [ ] Phase 4 — `io_utils` teardown → `registry/` + `stats/` + `io/`.
- [ ] Phase 5 — `stable_ginv/recon/`.
- [ ] Phase 6 — `stable_ginv/experiment/` (+ CSV-row goldens first).
- [ ] Phase 7 — `stable_ginv/viz/`.
- [ ] Phase 8 — `stable_ginv/jacobian/`.
- [ ] Phase 9 — docstrings + fill Sphinx API pages + polish.

## One-time setup

- Enable GitHub Pages: repo Settings → Pages → Source = "GitHub Actions".

## Next phase

Write the Phase 1 plan from the spec, then implement. Keep this file's Status and
Golden-harness sections current at every phase boundary.
```

- [ ] **Step 2: Add it to the handover reading order**

In `docs/handover/README.md`, under the "Read Next" list (after the
`../CODE_CLEANUP.md` line), add:
```markdown
- `HANDOVER_RESTRUCTURE.md`: OOP restructure status, invariants, and golden
  harness. Read before any restructure work.
```

- [ ] **Step 3: Commit**

```bash
git add docs/handover/HANDOVER_RESTRUCTURE.md docs/handover/README.md
git commit -m "docs: add restructure handover and wire into reading order"
```

---

## Final verification (end of Phase 0)

- [ ] **Run the full suite + lint + entry points + whitespace**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
conda run -n stable-ginv python -m pyflakes stable_ginv tests/golden
conda run -n stable-ginv python iDLG_mask.py --help >/dev/null && echo "CLI OK"
conda run -n stable-ginv python -m sphinx -b html docs/sphinx docs/sphinx/_build/html >/dev/null && echo "DOCS OK"
git diff --check
```
Expected: all green (`83 passed` — the original 80 plus 3 golden tests), `CLI OK`,
`DOCS OK`, no whitespace errors. Pre-existing pyflakes findings in
`helper/plots.py` and `functions/rank_reconstruction_plot.py` are out of scope
(see spec) and are not in the linted paths above.

- [ ] **Update the handover Status** if any task deviated, then stop. Phase 1 is a
  separate plan.

---

## Self-review notes (filled during planning)

- **Spec coverage:** package skeleton (§1), golden harness Tiers 1–3 (§2),
  phase-0 scaffolding (§3), Sphinx→Pages (§4), categorized commits + handover
  (§5) — all have tasks. End-to-end golden intentionally realized in-process
  (documented in Task 5 and handover) to avoid `/work3` pollution; CSV-row golden
  deferred to Phase 6 per the just-in-time principle.
- **Determinism basis:** recon is seeded in `run_single_exp.py`
  (`seed = run_id + idx_net + 1`); masking/registry goldens are pure functions of
  fixed inputs (portable hashes).
- **Known plan-time unknowns flagged inline:** LeNet parameter-name prefix
  (Task 3 Step 2) and worker config-key fidelity (Task 5 Step 2) are verified by a
  command before fixtures are generated, rather than assumed.
