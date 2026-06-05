# Phase 5 — `stable_ginv/recon/` Reconstruction OOP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the reconstruction worker (`run_single_exp.py`) and the LR scheduler (`helper/training_utils.py`) into a `stable_ginv/recon/` package — `ReconstructionRunner`, `EarlyStopPolicy`, `LabelInference`, and `make_scheduler` — leaving `run_single_exp.py` as an importable worker shim and `helper/training_utils.py` as a re-export shim, with **zero** change to reconstruction numerics, the result-dict contract, CLI defaults, or the recon-worker golden.

**Architecture:** Mirrors the Phase 3 masking split (thin OOP over verbatim numeric code). The 500-line optimization loop in `_run_inner` is moved **verbatim** into `ReconstructionRunner.run()` (the only edit being `result_queue.put(result)` → `return result`), with three cleanly-separable collaborators threaded in at their existing call sites: `make_scheduler` (pure relocation), `LabelInference` (the one-time label prediction from the original unmasked final-FC weight gradient), and `EarlyStopPolicy` (a frozen dataclass whose convergence predicate produces byte-identical reason strings and the same `< 1e-6` comparison). The hot numeric loop body is kept cohesive on purpose — it is the thing the golden guards, and further decomposition would risk float-order drift with no safety net. `run_single_exp.py` stays importable at its current path (multiprocessing spawn re-imports the target by qualified name) and `helper/training_utils.py` keeps `make_scheduler` importable for any external caller.

**Tech Stack:** Python 3, PyTorch (autograd, optimizers, LR schedulers), NumPy, pytest. Conda env: `stable-ginv`. All verification via `conda run -n stable-ginv`. Always `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` before Python (DTU HPC C++ runtime requirement).

---

## Invariants (enforce every task)

- `conda run -n stable-ginv python -m pytest tests/ -q` stays green throughout, **including `tests/golden/test_recon_worker_golden.py`**.
- `conda run -n stable-ginv python iDLG_mask.py --help` works throughout.
- No change to reconstruction math, optimizer behavior, early-stop behavior, the
  worker result dict (keys, order, types, `None` placeholders), masking behavior,
  CLI flags/defaults, or any registry/CSV output.
- The recon loop body is moved **verbatim**. Do not "improve", reorder, re-format,
  or re-tab the relocated numeric code. The only permitted edits to that body are
  the three documented collaborator call-site substitutions in Task 4 and changing
  the terminal `result_queue.put(result)` to `return result`.
- **Never regenerate the recon-worker golden to make a refactor pass.** If
  `test_recon_worker_golden.py` drifts, STOP and debug — the move was not verbatim.
- `run_single_exp.py` ends the phase as a worker shim re-exporting
  `run_single_experiment` and `_run_inner`; `helper/training_utils.py` ends as a
  re-export shim for `make_scheduler`. Every existing caller keeps working unedited.

## Module boundaries (current → target)

| Target file | Content |
|-------------|---------|
| `stable_ginv/recon/scheduler.py` | `make_scheduler` (verbatim from `helper/training_utils.py`) |
| `stable_ginv/recon/labels.py` | `LabelInference` — one-time label prediction from the original unmasked final-FC weight gradient (wraps `run_single_exp.py:130-137` verbatim) |
| `stable_ginv/recon/early_stop.py` | `EarlyStopPolicy` — frozen dataclass: convergence threshold + reason strings |
| `stable_ginv/recon/runner.py` | `ReconstructionRunner.run()` (verbatim `_run_inner` body) + `_run_inner` + `run_single_experiment` worker entries |
| `stable_ginv/recon/__init__.py` | Public recon API |
| `run_single_exp.py` | Worker shim → re-export `run_single_experiment`, `_run_inner` |
| `helper/training_utils.py` | Re-export shim → `make_scheduler` |

### Canonical imports inside `stable_ginv/recon/`

New modules import from the canonical `stable_ginv` packages already created in
Phases 2–4 (not via the `functions.`/`helper.` shims), except for modules whose
canonical home is a later, not-yet-created phase:

- `from stable_ginv.masking import build_gradient_mask, _get_last_fc_param_indices` (Phase 3)
- `from stable_ginv.metrics import compute_psnr_from_mse, compute_jacobian_rank, total_variation, compute_grad_match_loss, compute_ssim_batch` (Phase 2)
- `from stable_ginv.io import setstdout` (Phase 4)
- `import functions.consts as consts` — **keep**; `consts` moves to `stable_ginv/data/` in a later phase (out of scope here).
- `from helper.Network import get_model, weights_init` — **keep**; `helper/Network.py` moves to `stable_ginv/models/` in a later phase (out of scope here).

## File map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `tests/test_scheduler.py` | Locks `make_scheduler` milestones/gamma before the move (scheduler is **not** exercised by the recon golden, which uses `optimizer="lbfgs"`) |
| Create | `stable_ginv/recon/__init__.py` | Public recon API |
| Create | `stable_ginv/recon/scheduler.py` | `make_scheduler` |
| Create | `stable_ginv/recon/early_stop.py` | `EarlyStopPolicy` |
| Create | `stable_ginv/recon/labels.py` | `LabelInference` |
| Create | `stable_ginv/recon/runner.py` | `ReconstructionRunner` + worker entries |
| Modify | `helper/training_utils.py` | Replace with re-export shim |
| Modify | `run_single_exp.py` | Replace with worker shim |
| Modify | `tests/golden/test_recon_worker_golden.py` | Repoint `_run_inner` import to `stable_ginv.recon` |
| Modify | `docs/handover/HANDOVER_RESTRUCTURE.md` | Mark Phase 5 complete, update Next phase |

---

## Task 1: Lock `make_scheduler` with a unit test (before the move)

**Files:**
- Create: `tests/test_scheduler.py`

The recon-worker golden runs `optimizer="lbfgs"`, where `make_scheduler` is never
called (the LBFGS branch leaves `scheduler = None`). So the scheduler factory is
currently unguarded. This unit test locks its milestone math and gamma against the
**current** `helper.training_utils` implementation before relocation, following the
"lock the area before refactoring" discipline.

- [ ] **Step 1: Write the test**

Create `tests/test_scheduler.py`:

```python
"""Locks make_scheduler's MultiStepLR milestones/gamma before the Phase 5 recon move.

make_scheduler is not covered by the recon-worker golden (that golden uses
optimizer='lbfgs', which never builds a scheduler), so this test guards the
scheduler factory's relocation into stable_ginv/recon/scheduler.py.
"""
import torch

from helper.training_utils import make_scheduler


def _scheduler_for(iteration, gamma=0.5):
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    return make_scheduler(optimizer, iteration, gamma=gamma)


def test_make_scheduler_milestones_and_gamma():
    sched = _scheduler_for(iteration=1000, gamma=0.5)
    assert isinstance(sched, torch.optim.lr_scheduler.MultiStepLR)
    assert sched.milestones == {375: 1, 625: 1, 875: 1}
    assert sched.gamma == 0.5


def test_make_scheduler_milestones_floor_division():
    # 3/8, 5/8, 7/8 of 100 -> int() floors to 37, 62, 87.
    sched = _scheduler_for(iteration=100, gamma=0.1)
    assert sched.milestones == {37: 1, 62: 1, 87: 1}
    assert sched.gamma == 0.1
```

> Note: PyTorch's `MultiStepLR` stores `milestones` as a `Counter` (`{value: count}`),
> so the assertions compare against that mapping form.

- [ ] **Step 2: Run it against current code to verify it passes**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/test_scheduler.py -q
```
Expected: 2 passed. (If `milestones` compares unequal because your torch version
stores a plain list, adjust the expected form to match the **observed** current
output and note it — the goal is to lock current behavior, not impose a new one.)

- [ ] **Step 3: Commit (`test:`)**

```bash
git add tests/test_scheduler.py
git commit -m "test: lock make_scheduler milestones before Phase 5 recon split"
```

---

## Task 2: Create `stable_ginv/recon/scheduler.py`, `early_stop.py`, `labels.py`

**Files:**
- Create: `stable_ginv/recon/scheduler.py`
- Create: `stable_ginv/recon/early_stop.py`
- Create: `stable_ginv/recon/labels.py`

These are the three small, independently-verifiable collaborators. The runner
(Task 3) depends on them, so they are created first.

- [ ] **Step 1: Create the package directory**

```bash
mkdir -p /home/mathias/GitHub/stable-ginv/stable_ginv/recon
```

- [ ] **Step 2: Create `stable_ginv/recon/scheduler.py`**

Body copied **verbatim** from `helper/training_utils.py`.

```python
"""Learning-rate scheduler factory for the reconstruction optimizers."""
import torch


def make_scheduler(optimizer, iteration, gamma):
    """MultiStepLR with milestones at 3/8, 5/8, 7/8 of total iterations."""
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(iteration * 3 / 8),
            int(iteration * 5 / 8),
            int(iteration * 7 / 8),
        ],
        gamma=gamma,
    )
```

- [ ] **Step 3: Create `stable_ginv/recon/early_stop.py`**

A frozen dataclass capturing the in-loop convergence rule. Defaults reproduce the
current literals exactly: the runner's convergence check is `current_loss < 1e-6`,
the reason on convergence is `"converged"`, and the reason when the loop runs to
`Iteration` is `"fixed_iterations"`.

```python
"""Early-stop policy for the reconstruction optimization loop."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class EarlyStopPolicy:
    """Stop a restart early once the gradient-matching loss converges.

    Defaults reproduce the original run_single_exp behavior exactly: the loop
    breaks with reason 'converged' the first iteration the loss drops below
    1e-6, and otherwise reports 'fixed_iterations' after running to completion.
    """

    convergence_loss: float = 1e-6
    converged_reason: str = "converged"
    default_reason: str = "fixed_iterations"

    def converged(self, loss: float) -> bool:
        """True when `loss` has dropped below the convergence threshold."""
        return loss < self.convergence_loss
```

- [ ] **Step 4: Create `stable_ginv/recon/labels.py`**

Wraps the one-time label-prediction block (`run_single_exp.py:130-137`) verbatim.
It is a pure function of the network and the original (unmasked) gradient, computed
once before any reconstruction, so it lifts cleanly into a collaborator.

```python
"""One-time label inference from the original unmasked final-FC weight gradient."""
import torch

from stable_ginv.masking import _get_last_fc_param_indices


class LabelInference:
    """Predict the ground-truth label from the original gradient (iDLG label trick).

    Uses the sign of the row-summed final fully-connected weight gradient. This is
    computed once from the *unmasked* gradient and shared across both the iDLG and
    masked reconstructions.
    """

    @staticmethod
    def infer(net, original_dy_dx):
        """Return the predicted label as a 1-element long tensor."""
        named_params_list = list(net.named_parameters())
        last_fc_ids = _get_last_fc_param_indices(net)
        final_weight_idx = next(
            i for i in sorted(last_fc_ids) if named_params_list[i][0].endswith(".weight")
        )
        return torch.argmin(
            torch.sum(original_dy_dx[final_weight_idx], dim=-1), dim=-1
        ).detach().reshape((1,))
```

- [ ] **Step 5: Verify the collaborators import and behave**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "
from stable_ginv.recon.early_stop import EarlyStopPolicy
from stable_ginv.recon.labels import LabelInference
from stable_ginv.recon.scheduler import make_scheduler
p = EarlyStopPolicy()
assert p.converged(1e-7) is True and p.converged(1e-6) is False
assert p.converged_reason == 'converged' and p.default_reason == 'fixed_iterations'
print('recon collaborators OK')
"
```
Expected: prints `recon collaborators OK`. No error. (`p.converged(1e-6)` is `False`
because the original check is strict `<`.)

---

## Task 3: Create `stable_ginv/recon/runner.py` (verbatim loop, no collaborators wired yet)

**Files:**
- Create: `stable_ginv/recon/runner.py`

Move the **entire** body of `run_single_exp.py` into `runner.py`, restructured only
as: a `ReconstructionRunner` class whose `run()` method holds the current `_run_inner`
body verbatim (with the terminal `result_queue.put(result)` changed to
`return result`), plus thin `_run_inner` and `run_single_experiment` entry functions
that preserve the existing signatures. **Do not** wire in the Task 2 collaborators yet
— that substitution happens in Task 4 so any golden drift is isolated to one diff.

The imports at the top change from the `functions.`/`helper.` shims to the canonical
`stable_ginv` packages (and the two kept-as-is `functions.consts` / `helper.Network`
imports), as listed in "Canonical imports" above.

- [ ] **Step 1: Create `stable_ginv/recon/runner.py`**

```python
"""ReconstructionRunner: the per-experiment gradient-inversion optimization loop.

Moved from run_single_exp.py (Phase 5). The numeric loop body is preserved
verbatim; only the worker entry points and import paths changed. run_single_exp.py
remains a thin shim re-exporting run_single_experiment / _run_inner so that
multiprocessing spawn and the recon-worker golden keep working.
"""
import traceback

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import functions.consts as consts
from stable_ginv.masking import build_gradient_mask, _get_last_fc_param_indices
from stable_ginv.metrics import (
    compute_psnr_from_mse,
    compute_jacobian_rank,
    total_variation,
    compute_grad_match_loss,
    compute_ssim_batch,
)
from helper.Network import get_model, weights_init
from stable_ginv.recon.scheduler import make_scheduler
from stable_ginv.io import setstdout


class ReconstructionRunner:
    """Runs one experiment (all requested methods/restarts) and returns its result dict."""

    def __init__(self, idx_net, device_id, dst, dataset_name, config):
        self.idx_net = idx_net
        self.device_id = device_id
        self.dst = dst
        self.dataset_name = dataset_name
        self.config = config

    def run(self):
        idx_net = self.idx_net
        device_id = self.device_id
        dst = self.dst
        dataset_name = self.dataset_name
        config = self.config

        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = f'cuda:{device_id}'
        else:
            # CPU fallback: the orchestrator schedules a single worker when no GPU
            # is present. This is much slower and intended only for testing.
            device = 'cpu'

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
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        net = get_model(NETWORK_NAME, channel=channel, num_classes=num_classes, input_size=shape_img, pretrained=NETWORK_TRAINED)
        if NETWORK_TRAINED:
            print(f"[{device}] Loaded ImageNet-pretrained weights for {NETWORK_NAME}")
        elif NETWORK_NAME in ["LeNet", "LeNet_bigger", "MediumCNN", "BiggerCNN"]:
            net.apply(weights_init)

        net = net.to(device)
        net.eval()

        idx_shuffle = np.random.permutation(len(dst))
        tt = transforms.Compose([transforms.ToTensor()])

        criterion = nn.CrossEntropyLoss().to(device)
        imidx_list = []

        # Build GT batch — computed once, shared across all methods
        for imidx in range(num_dummy):
            idx = idx_shuffle[imidx]
            imidx_list.append(idx)
            tmp_datum = tt(dst[idx][0]).float().to(device)
            tmp_datum = tmp_datum.view(1, *tmp_datum.size())
            tmp_label = torch.tensor([dst[idx][1]], dtype=torch.long, device=device).view(1,)
            if imidx == 0:
                gt_data = tmp_datum
                gt_label = tmp_label
            else:
                gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                gt_label = torch.cat((gt_label, tmp_label), dim=0)

        # LeNet/LeNet_bigger use Sigmoid activations designed for raw [0,1] inputs.
        # Applying dataset normalisation shifts inputs into ≈[-2, +2], saturating
        # the sigmoid and zeroing out conv gradients — making inversion impossible.
        # Skip normalisation for these architectures to match the original iDLG paper.
        _sigmoid_nets = {"LeNet", "LeNet_bigger"}
        if NETWORK_NAME in _sigmoid_nets:
            dm = torch.zeros(1, channel, 1, 1, device=device)
            ds = torch.ones(1, channel, 1, 1, device=device)
        elif NETWORK_TRAINED and channel == 3:
            dm = torch.tensor(consts.imagenet_mean, device=device).view(1, channel, 1, 1)
            ds = torch.tensor(consts.imagenet_std,  device=device).view(1, channel, 1, 1)
        else:
            dm = torch.tensor(getattr(consts, f'{dataset_name.lower()}_mean'), device=device).view(1, channel, 1, 1)
            ds = torch.tensor(getattr(consts, f'{dataset_name.lower()}_std'),  device=device).view(1, channel, 1, 1)

        lower_bound = -dm / ds
        upper_bound = (1.0 - dm) / ds
        gt_data_norm = (gt_data - dm) / ds

        out = net(gt_data_norm)
        y = criterion(out, gt_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        original_dy_dx = [g.detach().clone() for g in dy_dx]
        total_entries = sum(g.numel() for g in original_dy_dx)

        named_params_list = list(net.named_parameters())
        last_fc_ids = _get_last_fc_param_indices(net)
        final_weight_idx = next(
            i for i in sorted(last_fc_ids) if named_params_list[i][0].endswith(".weight")
        )
        label_pred_from_original = torch.argmin(
            torch.sum(original_dy_dx[final_weight_idx], dim=-1), dim=-1
        ).detach().reshape((1,))

        final_recon = {}
        early_stop_reason_dict = {}
        early_stop_iter_dict = {}

        # Per-method result accumulators (avoids duplicated if/else dispatch blocks)
        _losses = {}
        _labels = {}
        _mses = {}
        _best_loss = {}
        _best_mse = {}
        _best_ssim = {}
        _jac_rank = {}
        _jac_shape = {}
        _psnr_per_restart = {}
        _img_per_restart = {}
        _mse_per_restart = {}
        _ssim_per_restart = {}

        if METHODS == "idlg":
            methods_to_run = ["iDLG"]
        elif METHODS == "masked":
            methods_to_run = ["iDLG_masked"]
        else:
            methods_to_run = ["iDLG", "iDLG_masked"]

        init_frames_by_method = {}
        recon_frames_by_method = {}

        for method in methods_to_run:

            best_restart_loss_value = float("inf")
            best_restart_mse_value = None
            best_restart_dummy = None
            best_restart_losses = None
            best_restart_mses = None
            best_restart_early_stop_reason = None
            best_restart_early_stop_iter = None

            if SAVE_GIF:
                _best_restart_init_np = None
                _best_restart_frames = []

            keep_ids, entry_masks = build_gradient_mask(
                method="idlg" if method == "iDLG" else "masked",
                mask_mode=MASK_MODE,
                net=net,
                original_dy_dx=original_dy_dx,
                prefixes=PREFIXES,
                prefix_layer_fracs=PREFIX_LAYER_FRACS,
                gradsize_topk=GRADSIZE_TOPK,
                gradsize_topfrac=GRADSIZE_TOPFRAC,
                gradsize_metric=GRADSIZE_METRIC,
            )

            if entry_masks is not None:
                observed_entries = sum(int(m.sum().item()) for m in entry_masks if m is not None)
            else:
                observed_entries = sum(
                    g.numel() for i, g in enumerate(original_dy_dx)
                    if g is not None and i in keep_ids
                )

            label_pred = label_pred_from_original

            unknowns = int(gt_data[0].numel())

            kept_fraction = observed_entries / total_entries

            jacobian_rank = None
            jacobian_shape = None

            if idx_net == 0 and device_id == 0:
                print(f"[GPU {device_id}] {method}: observed_entries={observed_entries}, "
                      f"total_entries={total_entries}, kept_fraction={kept_fraction:.4f}, "
                      f"unknowns={unknowns}")

                if MASK_MODE in ["prefix_topfrac_entries_layer", "prefix_topk_entries_layer"] and entry_masks is not None:
                    print(f"[GPU {device_id}] kept entries per prefix:")
                    for prefix in PREFIXES:
                        kept = 0
                        total = 0
                        for i, (name, _) in enumerate(net.named_parameters()):
                            if (name == prefix or name.startswith(prefix + ".")) and entry_masks[i] is not None:
                                kept += int(entry_masks[i].sum().item())
                                total += entry_masks[i].numel()
                        if total > 0:
                            print(f"  {prefix}: kept {kept}/{total} = {kept/total:.4f}")

                if MASK_MODE in ["prefix_topfrac", "prefix_topk"] and keep_ids is not None:
                    print(f"[GPU {device_id}] kept tensors per prefix:")
                    named_params = list(net.named_parameters())
                    keep_ids_set = set(keep_ids)
                    for prefix in PREFIXES:
                        total = 0
                        kept = 0
                        for i, (name, _) in enumerate(named_params):
                            if name == prefix or name.startswith(prefix + "."):
                                total += 1
                                if i in keep_ids_set:
                                    kept += 1
                        if total > 0:
                            if MASK_MODE == "prefix_topfrac":
                                req = PREFIX_LAYER_FRACS.get(prefix, GRADSIZE_TOPFRAC)
                            else:
                                req = int(PREFIX_LAYER_FRACS.get(prefix, GRADSIZE_TOPK))
                            print(f"  {prefix}: kept {kept}/{total} = {kept/total:.4f}, requested={req}")

            if COMPUTE_JACOBIAN_RANK:
                if num_dummy != 1:
                    raise ValueError("Jacobian-rank computation currently assumes num_dummy=1.")

                jacobian_rank, jacobian_shape, jac_obs, jac_unknowns = compute_jacobian_rank(
                    net=net,
                    x_norm=gt_data_norm,
                    y=gt_label,
                    criterion=criterion,
                    keep_ids=keep_ids,
                    entry_masks=entry_masks,
                    max_entries=JACOBIAN_MAX_ENTRIES,
                    select_mode=JACOBIAN_SELECT_MODE,
                    device_for_J="cpu",
                )
                print(f"[GPU {device_id}] {method}: jacobian_shape={jacobian_shape}, "
                      f"jacobian_rank={jacobian_rank}, unknowns={jac_unknowns}")

                if jacobian_rank < jac_unknowns:
                    print(f"[GPU {device_id}] {method}: Jacobian rank too small for unique local reconstruction "
                          f"({jacobian_rank} < {jac_unknowns})")

            if observed_entries < unknowns:
                print(f"[GPU {device_id}] {method}: too few gradients for reconstruction "
                      f"({observed_entries} < {unknowns})")
                final_recon[method] = torch.zeros_like(gt_data)
                _losses[method] = [float("inf")]
                _labels[method] = label_pred.item()
                _mses[method] = [float("inf")]
                _best_loss[method] = float("inf")
                _best_mse[method] = float("inf")
                early_stop_reason_dict[method] = "too_few_gradients"
                early_stop_iter_dict[method] = 0
                continue

            all_params = list(net.parameters())
            if entry_masks is not None:
                selected_ids = [i for i, m in enumerate(entry_masks) if m is not None and m.any()]
            else:
                selected_ids = sorted(list(keep_ids))

            selected_params = [all_params[i] for i in selected_ids]
            selected_original = [original_dy_dx[i] for i in selected_ids]
            selected_entry_masks = [entry_masks[i] for i in selected_ids] if entry_masks is not None else None

            _best_mse_per_restart = []
            _best_img_per_restart = []
            _ssim_per_restart_list = []
            _restart_range = [SINGLE_RESTART_IDX] if SINGLE_RESTART_IDX is not None else range(NUM_RESTARTS)

            for restart_idx in _restart_range:
                restart_seed = seed * 1000 + restart_idx
                torch.manual_seed(restart_seed)
                np.random.seed(restart_seed)
                torch.cuda.manual_seed_all(restart_seed)

                print(f"[GPU {device_id}] {method}: restart {restart_idx+1}/{NUM_RESTARTS}")

                dummy_data = torch.randn(gt_data.size(), device=device).requires_grad_(True)

                if SAVE_GIF:
                    _restart_init_np = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0).cpu().numpy()
                    _restart_frames = []
                    _last_gif_iter = -1

                scheduler = None

                if OPTIMIZER == "lbfgs":
                    optimizer = torch.optim.LBFGS(
                        [dummy_data],
                        lr=lr,
                        max_iter=MAX_ITERATION,
                        history_size=HISTORY_SIZE,
                    )
                    phase = "lbfgs"

                elif OPTIMIZER in ["adam", "signed_adam"]:
                    optimizer = torch.optim.Adam([dummy_data], lr=lr)
                    scheduler = make_scheduler(optimizer, Iteration, gamma=GAMMA)
                    phase = OPTIMIZER

                elif OPTIMIZER in ["adamw", "signed_adamw"]:
                    optimizer = torch.optim.AdamW([dummy_data], lr=lr, weight_decay=1e-5)
                    scheduler = make_scheduler(optimizer, Iteration, gamma=GAMMA)
                    phase = OPTIMIZER

                else:
                    raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

                losses = []
                mses = []

                early_stop_reason = "fixed_iterations"
                early_stop_iter = Iteration
                best_loss_value = float("inf")
                best_dummy = None
                best_mse_value = None

                for iters in range(Iteration):
                    if phase == "lbfgs":
                        def closure():
                            optimizer.zero_grad()
                            pred = net(dummy_data)
                            dummy_loss = criterion(pred, label_pred)
                            dummy_dy_dx = torch.autograd.grad(dummy_loss, selected_params, create_graph=True)
                            grad_diff, _ = compute_grad_match_loss(dummy_dy_dx, selected_original, selected_entry_masks=selected_entry_masks, grad_loss=GRAD_LOSS)
                            tv_loss = total_variation(dummy_data)
                            total_loss = grad_diff + TV_WEIGHT * tv_loss
                            total_loss.backward()
                            return total_loss

                        optimizer.step(closure)
                        current_loss = closure().item()

                        # Sigmoid-activation nets (LeNet, LeNet_bigger) are trained on raw [0,1]
                        # inputs with no normalisation. Clamping after each LBFGS step corrupts
                        # the quasi-Newton Hessian approximation (gradient stored at unclamped
                        # position, but next step starts from clamped position), causing many runs
                        # to diverge into wrong local minima. Skip clamping to match the original
                        # iDLG paper behaviour; pixels naturally converge to [0,1] when the loss
                        # drives them toward the GT.
                        if NETWORK_NAME not in {"LeNet", "LeNet_bigger"}:
                            with torch.no_grad():
                                dummy_data.clamp_(lower_bound, upper_bound)

                    elif phase in ["adam", "adamw", "signed_adam", "signed_adamw"]:
                        optimizer.zero_grad()
                        pred = net(dummy_data)
                        dummy_loss = criterion(pred, label_pred)
                        dummy_dy_dx = torch.autograd.grad(dummy_loss, selected_params, create_graph=True)
                        grad_diff, num_terms = compute_grad_match_loss(dummy_dy_dx, selected_original, selected_entry_masks=selected_entry_masks, grad_loss=GRAD_LOSS)
                        tv_loss = total_variation(dummy_data)
                        total_loss = grad_diff + TV_WEIGHT * tv_loss
                        total_loss.backward()

                        if phase in ["signed_adam", "signed_adamw"] and dummy_data.grad is not None:
                            with torch.no_grad():
                                dummy_data.grad.sign_()

                        optimizer.step()
                        with torch.no_grad():
                            dummy_data.clamp_(lower_bound, upper_bound)

                        if scheduler is not None:
                            scheduler.step()

                        current_loss = total_loss.item()

                    else:
                        raise ValueError(f"Unknown phase: {phase}")

                    current_x = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0)
                    current_mse = torch.mean((current_x - gt_data) ** 2).item()

                    if SAVE_GIF and iters % FRAME_INTERVAL == 0:
                        _restart_frames.append({
                            'iter': iters,
                            'dummy': current_x.cpu().numpy(),
                            'loss': current_loss if np.isfinite(current_loss) else float('inf'),
                            'mse': current_mse,
                        })
                        _last_gif_iter = iters

                    if np.isfinite(current_loss) and current_loss < best_loss_value:
                        best_loss_value = current_loss
                        best_mse_value = current_mse
                        best_dummy = current_x.detach().clone()

                    losses.append(current_loss)
                    mses.append(current_mse)

                    if current_loss < 1e-6:
                        early_stop_reason = "converged"
                        early_stop_iter = iters
                        break

                    if iters % 1000 == 0:
                        current_lr = optimizer.param_groups[0]["lr"]
                        print(f'[GPU {device_id}] {OPTIMIZER} restart {restart_idx+1} iters {iters}, lr = {current_lr:.6g}, loss = {current_loss:.8f}, mse = {current_mse:.8f}')

                if SAVE_GIF and Iteration > 0 and _last_gif_iter != iters:
                    _final_x = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0)
                    _restart_frames.append({
                        'iter': iters,
                        'dummy': _final_x.cpu().numpy(),
                        'loss': losses[-1] if losses else float('inf'),
                        'mse': torch.mean((_final_x - gt_data) ** 2).item(),
                    })

                if best_dummy is not None:
                    if best_restart_losses is None or best_loss_value < best_restart_loss_value:
                        best_restart_mse_value = best_mse_value
                        best_restart_loss_value = best_loss_value
                        best_restart_dummy = best_dummy.clone()
                        best_restart_losses = losses[:]
                        best_restart_mses = mses[:]
                        best_restart_early_stop_reason = early_stop_reason
                        best_restart_early_stop_iter = early_stop_iter
                        if SAVE_GIF:
                            _best_restart_init_np = _restart_init_np
                            _best_restart_frames = _restart_frames[:]

                _best_mse_per_restart.append(best_restart_mse_value)
                _best_img_per_restart.append(best_restart_dummy.cpu().numpy() if best_restart_dummy is not None else None)
                _ssim_per_restart_list.append(
                    compute_ssim_batch(best_restart_dummy, gt_data) if best_restart_dummy is not None else None
                )

            _psnr_per_restart[method] = [
                compute_psnr_from_mse(m, max_val=1.0) if (m is not None and np.isfinite(m)) else None
                for m in _best_mse_per_restart
            ]
            _img_per_restart[method] = _best_img_per_restart
            _mse_per_restart[method] = _best_mse_per_restart[:]
            _ssim_per_restart[method] = _ssim_per_restart_list

            if SAVE_GIF:
                init_frames_by_method[method] = _best_restart_init_np
                recon_frames_by_method[method] = _best_restart_frames

            if best_restart_dummy is not None:
                final_recon[method] = best_restart_dummy
                _best_ssim[method] = compute_ssim_batch(best_restart_dummy, gt_data)
            else:
                final_recon[method] = torch.zeros_like(gt_data)
                _best_ssim[method] = None

            _losses[method] = best_restart_losses if best_restart_losses is not None else [float("inf")]
            _labels[method] = label_pred.item()
            _mses[method] = best_restart_mses if best_restart_mses is not None else [float("inf")]
            _best_loss[method] = best_restart_loss_value
            _best_mse[method] = best_restart_mse_value
            _jac_rank[method] = jacobian_rank
            _jac_shape[method] = jacobian_shape

            early_stop_reason_dict[method] = best_restart_early_stop_reason
            early_stop_iter_dict[method] = best_restart_early_stop_iter

        result = {
            'idx_net': idx_net,
            'device_id': device_id,
            'gt_data': gt_data.detach().cpu().numpy(),
            'final_recon': {k: v.detach().cpu().numpy() for k, v in final_recon.items()},

            'last_psnr_idlg':    compute_psnr_from_mse(_mses['iDLG'][-1],        max_val=1.0) if 'iDLG'        in final_recon else None,
            'last_psnr_masked':  compute_psnr_from_mse(_mses['iDLG_masked'][-1], max_val=1.0) if 'iDLG_masked' in final_recon else None,
            'last_loss_iDLG':         _losses['iDLG'][-1]        if 'iDLG'        in final_recon else None,
            'last_mse_iDLG':          _mses['iDLG'][-1]          if 'iDLG'        in final_recon else None,
            'last_loss_iDLG_masked':  _losses['iDLG_masked'][-1] if 'iDLG_masked' in final_recon else None,
            'last_mse_iDLG_masked':   _mses['iDLG_masked'][-1]   if 'iDLG_masked' in final_recon else None,

            'best_psnr_idlg':   compute_psnr_from_mse(_best_mse.get('iDLG'),        max_val=1.0) if _best_mse.get('iDLG')        is not None else None,
            'best_psnr_masked': compute_psnr_from_mse(_best_mse.get('iDLG_masked'), max_val=1.0) if _best_mse.get('iDLG_masked') is not None else None,
            'best_loss_iDLG':        _best_loss.get('iDLG'),
            'best_mse_iDLG':         _best_mse.get('iDLG'),
            'best_loss_iDLG_masked': _best_loss.get('iDLG_masked'),
            'best_mse_iDLG_masked':  _best_mse.get('iDLG_masked'),
            'best_ssim_idlg':        _best_ssim.get('iDLG'),
            'best_ssim_masked':      _best_ssim.get('iDLG_masked'),

            'label_iDLG':        _labels.get('iDLG'),
            'label_iDLG_masked': _labels.get('iDLG_masked'),
            'jac_rank_iDLG':        _jac_rank.get('iDLG'),
            'jac_shape_iDLG':       _jac_shape.get('iDLG'),
            'jac_rank_iDLG_masked': _jac_rank.get('iDLG_masked'),
            'jac_shape_iDLG_masked':_jac_shape.get('iDLG_masked'),
            'gt_label': gt_label.detach().cpu().numpy(),
            'imidx_list': imidx_list,
            'early_stop_reason': early_stop_reason_dict,
            'early_stop_iter': early_stop_iter_dict,
            'psnr_per_restart_idlg':   _psnr_per_restart.get('iDLG'),
            'psnr_per_restart_masked': _psnr_per_restart.get('iDLG_masked'),
            'img_per_restart_idlg':    _img_per_restart.get('iDLG'),
            'img_per_restart_masked':  _img_per_restart.get('iDLG_masked'),
            'mse_per_restart_idlg':    _mse_per_restart.get('iDLG'),
            'mse_per_restart_masked':  _mse_per_restart.get('iDLG_masked'),
            'ssim_per_restart_idlg':   _ssim_per_restart.get('iDLG'),
            'ssim_per_restart_masked': _ssim_per_restart.get('iDLG_masked'),
            'restart_idx': SINGLE_RESTART_IDX,
            'init_frames': init_frames_by_method,
            'recon_frames': recon_frames_by_method,
        }

        return result


def _run_inner(idx_net, device_id, dst, dataset_name, config, result_queue):
    """Run one experiment and push its result dict onto the queue."""
    result = ReconstructionRunner(idx_net, device_id, dst, dataset_name, config).run()
    result_queue.put(result)


def run_single_experiment(idx_net, device_id, dst, dataset_name, config, result_queue):
    """Multiprocessing worker entry: set up stdout, run, and report errors via the queue."""
    setstdout(path=config.out_path)
    try:
        _run_inner(idx_net, device_id, dst, dataset_name, config, result_queue)
    except Exception as exc:
        result_queue.put({
            'error': str(exc),
            'traceback': traceback.format_exc(),
            'idx_net': idx_net,
            'device_id': device_id,
        })
```

> Verbatim-move note: the body inside `run()` is the original `_run_inner` body
> indented one extra level, with the trailing `result_queue.put(result)` replaced by
> `return result`. The original module-level `_run_inner` did stdout setup inside
> `run_single_experiment`; that structure is preserved — `run_single_experiment`
> still calls `setstdout` then `_run_inner` in a try/except. The collaborators from
> Task 2 are **not** yet used here (still inline `1e-6`, inline label block, etc.);
> they are wired in Task 4.

- [ ] **Step 2: Create `stable_ginv/recon/__init__.py`**

```python
from stable_ginv.recon.scheduler import make_scheduler
from stable_ginv.recon.early_stop import EarlyStopPolicy
from stable_ginv.recon.labels import LabelInference
from stable_ginv.recon.runner import (
    ReconstructionRunner,
    run_single_experiment,
    _run_inner,
)

__all__ = [
    "make_scheduler",
    "EarlyStopPolicy",
    "LabelInference",
    "ReconstructionRunner",
    "run_single_experiment",
    "_run_inner",
]
```

- [ ] **Step 3: Verify the package imports and the qualified spawn target resolves**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "
import stable_ginv.recon as r
print(r.__all__)
f = r.run_single_experiment
print(f.__module__, f.__qualname__)
import importlib
mod = importlib.import_module(f.__module__)
assert getattr(mod, f.__qualname__) is f, 'spawn target not importable by qualified name'
print('spawn target resolvable OK')
"
```
Expected: prints `__all__`, then `stable_ginv.recon.runner run_single_experiment`,
then `spawn target resolvable OK`. This proves multiprocessing **spawn** (which
pickles the target by `module`+`qualname` and re-imports it in the child) will find
the worker. No error.

- [ ] **Step 4: Run the recon-worker golden against the new module (imported directly)**

The golden still imports from `run_single_exp` (shim conversion is Task 5), but the
new code path is exercised here to confirm the verbatim move is numerically clean:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "
from stable_ginv.recon import _run_inner
from tests.golden.helpers import FakeImageDataset, lenet_worker_config
class Q:
    def __init__(self): self.result=None
    def put(self,v): self.result=v
dst = FakeImageDataset(16, channel=1, size=(28,28), num_classes=10, seed=0)
q = Q(); _run_inner(0,0,dst,'MNIST',lenet_worker_config(method='idlg'),q)
r = q.result
print('label', r['label_iDLG'], 'reason', r['early_stop_reason'].get('iDLG'))
print('loss', r['best_loss_iDLG'], 'mse', r['best_mse_iDLG'], 'psnr', r['best_psnr_idlg'])
"
```
Expected: prints a label, an early-stop reason, and finite loss/mse/psnr numbers
without error. (Exact equality to the fixture is asserted by pytest in Task 5.)

- [ ] **Step 5: Lint the new package**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pyflakes stable_ginv/recon
```
Expected: no output. (`jac_obs`/`num_terms`/`kept_fraction` etc. are assigned-then-
unused exactly as in the original `run_single_exp.py`; pyflakes treated the original
the same way. If pyflakes flags **only** those pre-existing unused locals, that is
acceptable — they are part of the verbatim move. If pyflakes is unavailable, use
`conda run -n stable-ginv python -m py_compile stable_ginv/recon/*.py`.)

---

## Task 4: Wire the collaborators into `ReconstructionRunner.run()`

**Files:**
- Modify: `stable_ginv/recon/runner.py`

Now substitute the three Task 2 collaborators at their existing call sites. Each
substitution is behavior-identical (same numbers, same string literals). Make all
three, then re-run the golden so any drift is caught in this single isolated diff.

- [ ] **Step 1: Use `LabelInference` for the one-time label prediction**

Find:
```python
        named_params_list = list(net.named_parameters())
        last_fc_ids = _get_last_fc_param_indices(net)
        final_weight_idx = next(
            i for i in sorted(last_fc_ids) if named_params_list[i][0].endswith(".weight")
        )
        label_pred_from_original = torch.argmin(
            torch.sum(original_dy_dx[final_weight_idx], dim=-1), dim=-1
        ).detach().reshape((1,))
```
Replace with:
```python
        label_pred_from_original = LabelInference.infer(net, original_dy_dx)
```

- [ ] **Step 2: Use `EarlyStopPolicy` for the convergence check**

Find:
```python
                losses = []
                mses = []

                early_stop_reason = "fixed_iterations"
                early_stop_iter = Iteration
```
Replace with:
```python
                losses = []
                mses = []

                early_stop_policy = EarlyStopPolicy()
                early_stop_reason = early_stop_policy.default_reason
                early_stop_iter = Iteration
```

Find:
```python
                    if current_loss < 1e-6:
                        early_stop_reason = "converged"
                        early_stop_iter = iters
                        break
```
Replace with:
```python
                    if early_stop_policy.converged(current_loss):
                        early_stop_reason = early_stop_policy.converged_reason
                        early_stop_iter = iters
                        break
```

- [ ] **Step 3: Update `runner.py` imports**

The inline label block no longer uses `_get_last_fc_param_indices` directly (it is
now used inside `LabelInference`). Add the two collaborator imports and drop the now-
unused name from the masking import.

Find:
```python
from stable_ginv.masking import build_gradient_mask, _get_last_fc_param_indices
from stable_ginv.metrics import (
    compute_psnr_from_mse,
    compute_jacobian_rank,
    total_variation,
    compute_grad_match_loss,
    compute_ssim_batch,
)
from helper.Network import get_model, weights_init
from stable_ginv.recon.scheduler import make_scheduler
from stable_ginv.io import setstdout
```
Replace with:
```python
from stable_ginv.masking import build_gradient_mask
from stable_ginv.metrics import (
    compute_psnr_from_mse,
    compute_jacobian_rank,
    total_variation,
    compute_grad_match_loss,
    compute_ssim_batch,
)
from helper.Network import get_model, weights_init
from stable_ginv.recon.scheduler import make_scheduler
from stable_ginv.recon.early_stop import EarlyStopPolicy
from stable_ginv.recon.labels import LabelInference
from stable_ginv.io import setstdout
```

- [ ] **Step 4: Re-run the recon-worker golden via the new code path**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "
from stable_ginv.recon import _run_inner
from tests.golden.helpers import FakeImageDataset, lenet_worker_config, load_or_regen
class Q:
    def __init__(self): self.result=None
    def put(self,v): self.result=v
def produce():
    dst = FakeImageDataset(16, channel=1, size=(28,28), num_classes=10, seed=0)
    q = Q(); _run_inner(0,0,dst,'MNIST',lenet_worker_config(method='idlg'),q)
    r = q.result
    return {'label_iDLG': r['label_iDLG'], 'loss_iDLG': r['best_loss_iDLG'], 'mse_iDLG': r['best_mse_iDLG'], 'psnr_idlg': r['best_psnr_idlg'], 'early_stop_reason': r['early_stop_reason'].get('iDLG')}
g = load_or_regen('recon_worker_golden.json', produce)
c = produce()
assert c['label_iDLG'] == g['label_iDLG'], 'label drift'
assert c['early_stop_reason'] == g['early_stop_reason'], 'reason drift'
import math
assert math.isclose(c['loss_iDLG'], g['loss_iDLG'], rel_tol=1e-4, abs_tol=1e-8), 'loss drift'
assert math.isclose(c['mse_iDLG'], g['mse_iDLG'], rel_tol=1e-4), 'mse drift'
print('recon numerics match golden after collaborator wiring')
"
```
Expected: prints `recon numerics match golden after collaborator wiring`. If any
`drift` assertion fires, STOP — the collaborator substitution changed behavior. Do
**not** edit the fixture.

- [ ] **Step 5: Lint**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pyflakes stable_ginv/recon
```
Expected: no new findings beyond the pre-existing unused locals noted in Task 3.

---

## Task 5: Convert `run_single_exp.py` and `helper/training_utils.py` to shims

**Files:**
- Modify: `run_single_exp.py`
- Modify: `helper/training_utils.py`

- [ ] **Step 1: Replace the entire content of `run_single_exp.py`**

```python
"""Worker shim: reconstruction lives in stable_ginv.recon (Phase 5).

Kept importable at this path so the multiprocessing worker target spawned by
iDLG_mask.py (`from run_single_exp import run_single_experiment`) and the
recon-worker golden continue to work unchanged. New code should import from
stable_ginv.recon.
"""
from stable_ginv.recon import run_single_experiment, _run_inner

__all__ = ["run_single_experiment", "_run_inner"]
```

- [ ] **Step 2: Replace the entire content of `helper/training_utils.py`**

```python
"""Shim: the LR scheduler factory lives in stable_ginv.recon.scheduler (Phase 5)."""
from stable_ginv.recon.scheduler import make_scheduler

__all__ = ["make_scheduler"]
```

- [ ] **Step 3: Run the full suite (golden flows through the shim → identical bytes)**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
```
Expected: all tests PASS, including `tests/golden/test_recon_worker_golden.py`
(imports `_run_inner` from the shim → resolves to `stable_ginv.recon`) and
`tests/test_scheduler.py` (imports `make_scheduler` from the shim → resolves to
`stable_ginv.recon.scheduler`). If the recon golden differs, STOP and debug — do
not regenerate the fixture.

- [ ] **Step 4: Confirm the CLI still works and the spawn target still resolves**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python iDLG_mask.py --help
conda run -n stable-ginv python -c "
from run_single_exp import run_single_experiment
print(run_single_experiment.__module__, run_single_experiment.__qualname__)
"
```
Expected: help text printed (exit 0); second command prints
`stable_ginv.recon.runner run_single_experiment` (the shim re-exports the real
function, so spawn re-imports it by its true qualified name).

- [ ] **Step 5: Lint the shims and package**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pyflakes stable_ginv/recon run_single_exp.py helper/training_utils.py
```
Expected: no output beyond the pre-existing unused locals in `runner.py` noted in
Task 3. (The shims re-export names used elsewhere; if pyflakes reports
"imported but unused" for the shim re-exports, that is the standard shim pattern —
`__all__` documents them as the public surface, matching `functions/io_utils.py`
and `helper/metrics.py`.)

- [ ] **Step 6: Commit (`refactor:`)**

```bash
git add stable_ginv/recon/ run_single_exp.py helper/training_utils.py
git commit -m "refactor: extract run_single_exp into stable_ginv/recon package"
```

---

## Task 6: Point the recon-worker golden at the canonical import path

**Files:**
- Modify: `tests/golden/test_recon_worker_golden.py`

Make the new dependency explicit (no longer routed through the shim), matching how
Phase 4 repointed its tests at canonical `stable_ginv` paths.

- [ ] **Step 1: Update the import**

Find:
```python
from run_single_exp import _run_inner
```
Replace with:
```python
from stable_ginv.recon import _run_inner
```

- [ ] **Step 2: Update the module docstring reference**

Find:
```python
"""Golden test: in-process reconstruction worker numerics.

Calls run_single_exp._run_inner directly on CPU with a fixed seed/config and a
synthetic dataset, capturing the result dict via a stub queue. Locks recon
metrics so later phases (config dataclass, recon OOP) prove no numeric drift.
```
Replace with:
```python
"""Golden test: in-process reconstruction worker numerics.

Calls stable_ginv.recon._run_inner directly on CPU with a fixed seed/config and a
synthetic dataset, capturing the result dict via a stub queue. Locks recon
metrics so later phases prove no numeric drift.
```

- [ ] **Step 3: Run the golden — hash must be unchanged**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/golden/test_recon_worker_golden.py -q
```
Expected: 1 passed. The underlying code is byte-identical via the verbatim move, so
the fixture matches.

- [ ] **Step 4: Commit (`test:`)**

```bash
git add tests/golden/test_recon_worker_golden.py
git commit -m "test: import recon worker from stable_ginv.recon canonical path"
```

---

## Task 7: Update the handover and finalize

**Files:**
- Modify: `docs/handover/HANDOVER_RESTRUCTURE.md`

- [ ] **Step 1: Mark Phase 5 complete in the Status section**

Find:
```markdown
- [ ] Phase 5 — `stable_ginv/recon/`.
```
Replace with:
```markdown
- [x] Phase 5 — `stable_ginv/recon/` (ReconstructionRunner, EarlyStopPolicy,
      LabelInference, scheduler). `run_single_exp.py` + `helper/training_utils.py`
      are shims.
```

- [ ] **Step 2: Replace the `## Next phase` section**

Find the entire `## Next phase` block and replace it with:
```markdown
## Next phase

Write the Phase 6 plan from the spec, then implement. Phase 6 extracts
`stable_ginv/experiment/` — `BatchExperimentRunner` (GPU scheduling, multiprocessing,
output ordering, abort-on-worker-failure), `ResultAggregator`, and `RestartSelector`,
plus `functions/experiment_results.py` → `stable_ginv/experiment/results.py` +
`stable_ginv/io/csv.py`. **Add the per-run `experiment_results` CSV-row golden first**
(the spec defers it to just before this refactor), then move `iDLG_mask.py`
orchestration into `stable_ginv/cli/batch.py` + `stable_ginv/experiment/runner.py`,
leaving `iDLG_mask.py` as a thin `from stable_ginv.cli.batch import main; main()`
wrapper. Keep this file's Status section current at every phase boundary.
```

- [ ] **Step 3: Final full verification**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
conda run -n stable-ginv python iDLG_mask.py --help
git diff --check
```
Expected: all tests PASS, help prints, no whitespace errors.

- [ ] **Step 4: Commit (`docs:`)**

```bash
git add docs/handover/HANDOVER_RESTRUCTURE.md
git commit -m "docs: mark Phase 5 complete in HANDOVER_RESTRUCTURE"
```

---

## Self-review checklist

**Spec coverage (Section 1 mapping `run_single_exp.py` → `recon/runner.py` + shim; `helper/training_utils.py` → `recon/scheduler.py`):**
- [x] `ReconstructionRunner` owns the optimization loop (Task 3/4).
- [x] `EarlyStopPolicy` dataclass (Task 2, wired Task 4).
- [x] `LabelInference` one-time, from the original unmasked FC gradient (Task 2, wired Task 4).
- [x] `scheduler` (`make_scheduler`) extracted to `recon/scheduler.py` (Task 2).
- [x] `run_single_exp.py` stays importable as a worker shim for multiprocessing spawn (Task 5; spawn-target resolvability verified Task 3 Step 3 + Task 5 Step 4).
- [x] `helper/training_utils.py` → re-export shim (Task 5).
- [x] Recon-worker golden guards numerics and is not regenerated (Tasks 3–6 assert against the existing fixture).
- [x] Scheduler locked with a unit test before the move, since the recon golden uses `lbfgs` and never builds a scheduler (Task 1).
- [x] Categorized commits: `test:` (scheduler lock) / `refactor:` (extract) / `test:` (golden repoint) / `docs:`.
- [x] HANDOVER_RESTRUCTURE.md Status + Next phase updated (Task 7).

**Placeholder scan:** no TBD/TODO/"handle edge cases"/"similar to" — `runner.py` is the
full verbatim loop body; collaborator bodies are complete.

**Type/name consistency:**
- [x] `EarlyStopPolicy().converged(loss)`, `.converged_reason`, `.default_reason`, `.convergence_loss` are referenced identically in Task 4 as defined in Task 2.
- [x] `LabelInference.infer(net, original_dy_dx)` signature matches its single call site in Task 4 Step 1.
- [x] `make_scheduler(optimizer, iteration, gamma)` signature unchanged; both call sites in `runner.py` pass `make_scheduler(optimizer, Iteration, gamma=GAMMA)` exactly as the original.
- [x] `_run_inner(idx_net, device_id, dst, dataset_name, config, result_queue)` and `run_single_experiment(...)` keep their original signatures, so `iDLG_mask.py` and the golden need no call-site edits.
- [x] `recon/__init__.py` `__all__` lists the four abstractions plus the two worker entries; the `run_single_exp.py` shim re-exports exactly `run_single_experiment` + `_run_inner`.

**Potential issues to watch:**
- **No circular imports:** `recon.scheduler` imports only `torch`; `recon.early_stop` only `dataclasses`; `recon.labels` imports `stable_ginv.masking` (no recon dependency); `recon.runner` imports masking/metrics/io/scheduler/labels/early_stop. None of those import `recon`, so the package is acyclic.
- **Verbatim move:** the `run()` body must be the original `_run_inner` body indented one level, identical token-for-token except the three Task-4 substitutions and `result_queue.put(result)` → `return result`. Re-tabbing must not alter string contents inside `print(...)` calls.
- **Spawn semantics:** `run_single_experiment.__module__` becomes `stable_ginv.recon.runner`. With the `spawn` start method, the child re-imports the target by `(module, qualname)`, which resolves because that module is importable; verified in Task 3 Step 3 and Task 5 Step 4. The shim import in `iDLG_mask.py` is irrelevant to pickling — the function object carries its own true module path.
- **Pre-existing unused locals** (`jac_obs`, `num_terms`, `kept_fraction`) are carried over unchanged; do not "fix" them in this phase (out of scope; would be a behavior-neutral but non-verbatim edit and is not what Phase 5 covers).
- **Do not edit `pyproject.toml`:** the editable install puts the repo root on `sys.path`, so `stable_ginv.recon` imports as an on-disk subpackage exactly like `stable_ginv.masking`/`.metrics`/`.io` already do.
- **Do not touch reconstruction math, optimizer branches, clamping rules, seeding, or the result-dict construction** — all are inside the verbatim-moved body.
