# Phase 7 — `stable_ginv/viz/` Visualization Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `helper/visualization.py` (recon panels, GIFs, restart curves/images) and the five standalone `helper/plot_*.py` CLIs into a `stable_ginv/viz/` package, leaving `helper/visualization.py` as a re-export shim and each `helper/plot_*.py` as a thin runnable wrapper, with **zero** change to plot filenames/contents, the restart-curve CSV bytes, panel ordering, or any CLI behavior.

**Architecture:** Mirrors the Phase 2 metrics split and the Phase 4/5/6 shim discipline. `helper/visualization.py` (620 lines) splits by responsibility into three focused submodules — `panels.py` (panel buffers + PNG panel), `gif.py` (animated GIF), `restart.py` (restart-curve CSV/PNG + per-restart image grid + paired-gain CIs) — re-exported by `stable_ginv/viz/__init__.py`. Each submodule replicates the original module's `matplotlib.use("Agg")` + `seaborn` theme side effects so rendered output is identical regardless of import path. The five `helper/plot_*.py` CLIs move **whole** into `stable_ginv/viz/` (only their `functions.io_utils` import line is repointed to canonical `stable_ginv.io`/`.registry` and the now-misdirected `sys.path` bootstrap is dropped), each replaced by a thin `helper/plot_*.py` wrapper that re-adds the repo root to `sys.path` and calls `main`. Code bodies are relocated verbatim; behavior is preserved.

**Tech Stack:** Python 3, Matplotlib (Agg backend), Seaborn, NumPy, pandas, PyTorch, PIL, SciPy (paired CIs / Shapiro), pytest. Conda env: `stable-ginv`. All verification via `conda run -n stable-ginv`. Always `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` before Python (DTU HPC C++ runtime requirement).

---

## Invariants (enforce every task)

- `conda run -n stable-ginv python -m pytest tests/ -q` stays green throughout, including all goldens and the **new** `tests/golden/test_restart_curve_csv_golden.py`.
- `conda run -n stable-ginv python iDLG_mask.py --help` works throughout (the parser is untouched; visualization is loaded lazily and only repointed, not changed).
- No change to: panel ordering/contents, recon-panel/GIF/restart PNG filenames or appearance, the restart-curve CSV columns/order/rounding/bytes, the paired restart-gain CI math, or any `print(...)` string literal.
- Moved bodies are relocated **verbatim**. The only permitted edits are the documented ones in each task (import repointing, dropping the misdirected `sys.path` bootstrap + now-unused `import sys`, the seaborn-theme header replication, and the monkeypatch-target repoint in `test_visualization.py`). Do not reorder, reformat, or "improve" any relocated function body.
- **Never regenerate a golden to make a refactor pass.** If `test_restart_curve_csv_golden.py` (or any golden) drifts, STOP and debug — the move was not verbatim. Do not edit any fixture.
- `helper/visualization.py` ends the phase as a re-export shim; each `helper/plot_*.py` ends as a thin runnable wrapper. Every existing caller (the test suite, `stable_ginv/cli/batch.py`, docs' `python helper/plot_masking_sweep_csv.py ...` invocation) keeps working.
- **Out of scope (do NOT move):** `functions/rank_reconstruction_plot.py` (Phase 8 — jacobian) and `helper/plots.py` (has a pre-existing `resnet50_data` bug, tracked separately). `from helper.Network import get_model` in `plot_model_parameter_counts.py` stays on its current path (`helper/Network.py` → `stable_ginv/models/` is a later phase).

## Module boundaries (current → target)

| Target file | Content |
|-------------|---------|
| `stable_ginv/viz/panels.py` | `create_panel_buffers`, `append_result_to_panel_buffers`, `flush_recon_panel`, `save_recon_panel` (verbatim from `visualization.py`) |
| `stable_ginv/viz/gif.py` | `save_recon_gif` (verbatim) |
| `stable_ginv/viz/restart.py` | `_restart_stats`, `_gain_ci_str`, `save_restart_curve`, `save_restart_images` (verbatim) |
| `stable_ginv/viz/__init__.py` | Public viz API (re-exports all of the above) |
| `stable_ginv/viz/plot_combined_masking_sweep_summary.py` | Verbatim `helper/plot_combined_masking_sweep_summary.py` (imports repointed) |
| `stable_ginv/viz/plot_masking_sweep_csv.py` | Verbatim `helper/plot_masking_sweep_csv.py` (imports repointed) |
| `stable_ginv/viz/plot_model_parameter_counts.py` | Verbatim `helper/plot_model_parameter_counts.py` (imports repointed) |
| `stable_ginv/viz/plot_normality_scatter.py` | Verbatim `helper/plot_normality_scatter.py` (imports repointed) |
| `stable_ginv/viz/plot_paired_masking_violin.py` | Verbatim `helper/plot_paired_masking_violin.py` (imports repointed) |
| `helper/visualization.py` | Re-export shim |
| `helper/plot_*.py` (×5) | Thin runnable wrappers → `from stable_ginv.viz.<name> import main` |

### Canonical imports inside the new modules (Phase 2/4 homes)

- `panels.py`: `from stable_ginv.io import safe_savefig`
- `gif.py`: `from stable_ginv.io import safe_chmod, safe_makedirs`
- `restart.py`: `from stable_ginv.io import safe_savefig, safe_write` and `from stable_ginv.stats import normality_str_from_ci, paired_t_ci`
- `plot_combined_masking_sweep_summary.py`: `from stable_ginv.io import safe_makedirs, safe_savefig, safe_write`
- `plot_masking_sweep_csv.py`: `from stable_ginv.io import safe_makedirs, safe_savefig, safe_write` + `from stable_ginv.registry import find_registry_entry`
- `plot_model_parameter_counts.py`: `from stable_ginv.io import safe_savefig` (keep `from helper.Network import get_model`)
- `plot_normality_scatter.py`: `from stable_ginv.io import safe_makedirs, safe_savefig`
- `plot_paired_masking_violin.py`: `from stable_ginv.io import resolve_storage_paths, safe_makedirs, safe_savefig, safe_write`

## File map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `tests/golden/test_restart_curve_csv_golden.py` | Locks the restart-curve CSV **with gain columns** (k=3,5 path) before the move |
| Create (auto) | `tests/golden/fixtures/restart_curve_csv_golden.json` | Golden fixture (auto-generated on first run) |
| Create | `stable_ginv/viz/__init__.py` | Public viz API |
| Create | `stable_ginv/viz/panels.py` | Panel buffers + PNG panel |
| Create | `stable_ginv/viz/gif.py` | Animated GIF |
| Create | `stable_ginv/viz/restart.py` | Restart curve/images + paired gains |
| Delete | `helper/visualization.py` (original) | Replaced by shim |
| Create | `helper/visualization.py` | Re-export shim |
| Modify | `tests/test_visualization.py` | Monkeypatch target → `stable_ginv.viz.panels` |
| Modify | `stable_ginv/cli/batch.py` | `_load_visualization_helpers` imports `stable_ginv.viz` |
| Modify | `tests/test_restart_curve.py` | Import from `stable_ginv.viz` |
| Modify | `tests/golden/test_restart_curve_csv_golden.py` | Import from `stable_ginv.viz` |
| Move | `helper/plot_*.py` (×5) → `stable_ginv/viz/plot_*.py` | Verbatim relocation (imports repointed) |
| Create | `helper/plot_*.py` (×5) | Thin runnable wrappers |
| Modify | `tests/test_plot_masking_sweep_csv.py` | Import from `stable_ginv.viz.plot_masking_sweep_csv` |
| Modify | `tests/test_plot_paired_masking_violin.py` | Import from `stable_ginv.viz.plot_paired_masking_violin` |
| Modify | `docs/handover/HANDOVER_RESTRUCTURE.md` | Mark Phase 7 complete, update Next phase |

**Do not edit `pyproject.toml`.** The editable install puts the repo root on `sys.path`, so `stable_ginv.viz` imports as an on-disk subpackage exactly like `stable_ginv.experiment`/`.recon` already do.

---

## Task 1: Lock the restart-curve CSV (with gain columns) with a golden (before the move)

**Files:**
- Create: `tests/golden/test_restart_curve_csv_golden.py`
- Create (auto): `tests/golden/fixtures/restart_curve_csv_golden.json`

The existing `tests/test_restart_curve.py` only covers `num_restarts=2`, which leaves `gain_ks` empty (gains need k≥3) — so the gain-column CSV rewrite path (the part coupled to `stable_ginv.stats`: `paired_t_ci` + `normality_str_from_ci`) is **unlocked**. This golden uses `num_restarts=5` (→ `gain_ks=[3,5]`) and three experiments so paired CIs and Shapiro normality both have ≥3 samples, capturing the full restart-curve CSV bytes including every gain column. It imports from the **current** `helper.visualization` path; Task 4 repoints it after the move.

- [ ] **Step 1: Write the test**

Create `tests/golden/test_restart_curve_csv_golden.py`:

```python
"""Golden test: the restart-curve CSV (with gain columns) is byte-stable across the Phase 7 split.

save_restart_curve writes a CSV; for num_restarts>=3 it rewrites that CSV with paired
restart-gain columns (gain/ci_low/ci_high/normality for PSNR and MSE, per k in {3,5,10}
that is <= num_restarts). This locks num_restarts=5 (gain_ks=[3,5]) with three experiments
so paired_t_ci and the Shapiro normality string both have >=3 samples. The deterministic
inputs make the CSV bytes reproducible. The PNG side output is not golden-locked (matplotlib
binaries are not byte-stable); only the CSV file content is asserted.
"""
import os
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")  # headless: save_restart_curve also writes a PNG

from helper.visualization import save_restart_curve
from tests.golden.helpers import load_or_regen


def _produce():
    psnr_idlg = [
        [10.0, 14.0, 17.0, 19.0, 20.0],
        [8.0, 12.0, 15.0, 17.0, 18.0],
        [9.0, 13.0, 16.0, 18.0, 19.0],
    ]
    psnr_masked = [
        [7.0, 10.0, 12.0, 13.0, 14.0],
        [6.0, 9.0, 11.0, 12.0, 13.0],
        [5.0, 8.0, 10.0, 11.0, 12.0],
    ]
    mse_idlg = [
        [0.20, 0.12, 0.08, 0.05, 0.04],
        [0.25, 0.15, 0.10, 0.07, 0.05],
        [0.22, 0.13, 0.09, 0.06, 0.045],
    ]
    mse_masked = [
        [0.30, 0.22, 0.18, 0.15, 0.13],
        [0.35, 0.26, 0.21, 0.18, 0.16],
        [0.32, 0.24, 0.19, 0.16, 0.14],
    ]
    workdir = tempfile.mkdtemp()
    try:
        csv_path, _png_path = save_restart_curve(
            save_dir=workdir,
            timestamp_str="GOLDEN",
            num_restarts=5,
            network_name="LeNet",
            dataset="MNIST",
            mask_mode="gradsize_topfrac_entries_layer",
            psnr_per_restart_idlg_all=psnr_idlg,
            psnr_per_restart_masked_all=psnr_masked,
            mse_per_restart_idlg_all=mse_idlg,
            mse_per_restart_masked_all=mse_masked,
        )
        with open(csv_path) as f:
            return f.read()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_restart_curve_csv_with_gains_stable():
    golden = load_or_regen("restart_curve_csv_golden.json", _produce)
    assert _produce() == golden
```

- [ ] **Step 2: Generate the fixture and verify it passes against current code**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/golden/test_restart_curve_csv_golden.py -q
```
Expected: 1 passed. `load_or_regen` writes `tests/golden/fixtures/restart_curve_csv_golden.json` on first run, then asserts `_produce()` against it. Confirm the fixture exists:

```bash
ls tests/golden/fixtures/restart_curve_csv_golden.json
```
Expected: the path prints (file created).

- [ ] **Step 3: Commit (`test:`)**

```bash
git add tests/golden/test_restart_curve_csv_golden.py tests/golden/fixtures/restart_curve_csv_golden.json
git commit -m "test: lock restart-curve CSV gain columns before Phase 7 viz split"
```

---

## Task 2: Split `helper/visualization.py` into `stable_ginv/viz/` + shim; repoint the panel test

**Files:**
- Create: `stable_ginv/viz/panels.py`
- Create: `stable_ginv/viz/gif.py`
- Create: `stable_ginv/viz/restart.py`
- Create: `stable_ginv/viz/__init__.py`
- Delete + recreate: `helper/visualization.py` (shim)
- Modify: `tests/test_visualization.py`

The three submodules contain the **verbatim** function bodies from `helper/visualization.py`. The only changes versus the original are: (a) each submodule carries its own `matplotlib.use("Agg")` + `import seaborn as sns` + `sns.set_theme(style="whitegrid")` header so the global render state is identical no matter which submodule is imported first (the original set it once at module import); and (b) the `functions.io_utils` import is repointed to the canonical `stable_ginv.io` / `stable_ginv.stats` homes. Do not alter any function body.

- [ ] **Step 1: Create the package directory**

```bash
mkdir -p /home/mathias/GitHub/stable-ginv/stable_ginv/viz
```

- [ ] **Step 2: Create `stable_ginv/viz/panels.py`** (verbatim `create_panel_buffers` / `append_result_to_panel_buffers` / `flush_recon_panel` / `save_recon_panel`)

```python
"""Reconstruction-panel buffers and adaptive-row PNG panel rendering (Phase 7).

Extracted verbatim from helper/visualization.py: per-experiment panel buffers,
ordering by experiment index on flush, and the PNG panel writer whose rows adapt
to which methods (iDLG / masked) were run.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from stable_ginv.io import safe_savefig

sns.set_theme(style="whitegrid")


def create_panel_buffers():
    """Create named reconstruction-panel buffers."""
    return {
        "exp_idx": [],
        "gt": [],
        "idlg": [],
        "masked": [],
        "psnr_idlg": [],
        "ssim_idlg": [],
        "mse_idlg": [],
        "psnr_masked": [],
        "ssim_masked": [],
        "mse_masked": [],
    }


def append_result_to_panel_buffers(result, buffers, to_pil, warn_fn):
    """Append one experiment result to panel buffers."""
    idx = result["idx_net"]
    buffers["exp_idx"].append(idx)
    gt_pil = to_pil(torch.from_numpy(result["gt_data"])[0])
    buffers["gt"].append(gt_pil)

    if "iDLG" not in result["final_recon"]:
        idlg_pil = gt_pil
    elif result.get("best_psnr_idlg") is None:
        warn_fn(
            f"[WARNING] Experiment {idx}: iDLG reconstruction failed "
            f"(all restarts diverged); showing blank in panel."
        )
        idlg_pil = gt_pil
    else:
        idlg_pil = to_pil(torch.from_numpy(result["final_recon"]["iDLG"])[0])
    buffers["idlg"].append(idlg_pil)

    if "iDLG_masked" not in result["final_recon"]:
        masked_pil = gt_pil
    elif result.get("best_psnr_masked") is None:
        warn_fn(
            f"[WARNING] Experiment {idx}: masked iDLG reconstruction failed "
            f"(all restarts diverged); showing blank in panel."
        )
        masked_pil = gt_pil
    else:
        masked_pil = to_pil(torch.from_numpy(result["final_recon"]["iDLG_masked"])[0])
    buffers["masked"].append(masked_pil)

    buffers["psnr_idlg"].append(result.get("best_psnr_idlg"))
    buffers["ssim_idlg"].append(result.get("best_ssim_idlg"))
    buffers["mse_idlg"].append(result.get("best_mse_iDLG"))
    buffers["psnr_masked"].append(result.get("best_psnr_masked"))
    buffers["ssim_masked"].append(result.get("best_ssim_masked"))
    buffers["mse_masked"].append(result.get("best_mse_iDLG_masked"))


def flush_recon_panel(params, buffers, panel_png_paths, save_dir, block_idx,
                      dataset, mask_desc, timestamp_str, methods):
    """Save current reconstruction panel buffers and clear them."""
    if len(buffers["gt"]) == 0:
        return block_idx
    order = sorted(range(len(buffers["exp_idx"])), key=lambda i: buffers["exp_idx"][i])
    ordered = {
        name: [values[i] for i in order]
        for name, values in buffers.items()
    }
    panel_path = save_recon_panel(
        params, ordered["gt"], ordered["idlg"], ordered["masked"],
        save_dir, block_idx, dataset, mask_desc, timestamp_str,
        methods=methods,
        exp_indices=ordered["exp_idx"],
        psnr_idlg=ordered["psnr_idlg"],
        ssim_idlg=ordered["ssim_idlg"],
        mse_idlg=ordered["mse_idlg"],
        psnr_masked=ordered["psnr_masked"],
        ssim_masked=ordered["ssim_masked"],
        mse_masked=ordered["mse_masked"],
    )
    if panel_path:
        panel_png_paths.append(panel_path)
    for values in buffers.values():
        values.clear()
    return block_idx + 1


def save_recon_panel(params: dict, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                     save_dir, block_idx, dataset, mask_desc: str, timestamp_str: str,
                     methods: str = "both",
                     exp_indices=None,
                     psnr_idlg=None, ssim_idlg=None, mse_idlg=None,
                     psnr_masked=None, ssim_masked=None, mse_masked=None):
    """Save a PNG reconstruction panel; rows adapt to which methods were run."""
    n = len(panel_gt_pil)
    if n == 0:
        return

    rows = [("GT", panel_gt_pil, None, None, None)]

    if methods in ["idlg", "both"]:
        rows.append(("iDLG", panel_idlg_pil, psnr_idlg, ssim_idlg, mse_idlg))

    if methods in ["masked", "both"]:
        rows.append(("iDLG_masked", panel_masked_pil, psnr_masked, ssim_masked, mse_masked))

    num_rows = len(rows)
    fig = plt.figure(figsize=(2.2 * n + 1.6, 2.5 * num_rows))

    for r, (row_name, row_imgs, row_psnr, row_ssim, row_mse) in enumerate(rows):
        y = 1.0 - (r + 0.5) / num_rows
        fig.text(0.01, y, row_name, va='center', ha='left',
                 fontsize=14, fontweight='bold')

        for j in range(n):
            ax = plt.subplot(num_rows, n, r * n + 1 + j)
            ax.imshow(row_imgs[j], cmap='gray' if dataset == 'MNIST' else None)
            if r == 0:
                exp_idx = exp_indices[j] if exp_indices is not None else j
                ax.set_title(f"exp {exp_idx}", fontsize=8)

            ax.axis('off')
            if row_psnr is not None or row_ssim is not None or row_mse is not None:
                parts = []
                if row_psnr is not None and row_psnr[j] is not None:
                    parts.append(f"PSNR:{row_psnr[j]:.2f}dB")
                if row_ssim is not None and row_ssim[j] is not None:
                    parts.append(f"SSIM:{row_ssim[j]:.3f}")
                if row_mse is not None and row_mse[j] is not None:
                    parts.append(f"MSE:{row_mse[j]:.4f}")
                ax.text(0.5, -0.05, "\n".join(parts), transform=ax.transAxes,
                        fontsize=7, ha='center', va='top')

    plt.tight_layout(rect=(0.08, 0.0, 1.0, 1.0))
    if os.environ.get("LSB_INTERACTIVE") == "Y":
        job_id = "INTERACTIVE"
    else:
        job_id = os.environ.get("LSB_JOBID", "")
    prefix = f"{timestamp_str}_{job_id}" if job_id else timestamp_str
    out_path = os.path.join(save_dir, f"{prefix}_{block_idx}.png")
    ok = safe_savefig(fig, out_path, dpi=75, bbox_inches='tight')
    plt.close(fig)

    if not ok:
        return None
    print("Saved reconstruction panel to:", out_path)
    return out_path
```

- [ ] **Step 3: Create `stable_ginv/viz/gif.py`** (verbatim `save_recon_gif`)

```python
"""Animated reconstruction-progress GIF rendering (Phase 7).

Extracted verbatim from helper/visualization.py: builds a per-iteration grid
(rows = experiments, cols = [Init | iDLG? | Masked? | GT]) and writes an animated GIF.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from stable_ginv.io import safe_chmod, safe_makedirs

sns.set_theme(style="whitegrid")


def save_recon_gif(
    results_list,
    save_dir, block_idx, dataset, mask_desc, timestamp_str,
    methods="both", fps=8,
):
    """Save an animated GIF of reconstruction progress; rows=experiments, cols=[Init|iDLG?|Masked?|GT]."""
    import io
    from PIL import Image as PILImage

    n_exp = len(results_list)
    if n_exp == 0:
        return None

    show_idlg = methods in ["idlg", "both"]
    show_masked = methods in ["masked", "both"]

    def _to_pil(img_np):
        arr = np.clip(img_np, 0.0, 1.0)
        if arr.shape[0] == 1:
            return PILImage.fromarray((arr[0] * 255).astype(np.uint8), mode='L')
        return PILImage.fromarray(
            (np.transpose(arr, (1, 2, 0)) * 255).astype(np.uint8), mode='RGB'
        )

    max_frames = 1
    for r in results_list:
        for mk in (["iDLG"] if show_idlg else []) + (["iDLG_masked"] if show_masked else []):
            n = len(r.get("recon_frames", {}).get(mk, []))
            if n > max_frames:
                max_frames = n

    cols = ["init"]
    if show_idlg:
        cols.append("iDLG")
    if show_masked:
        cols.append("masked")
    cols.append("GT")
    n_cols = len(cols)

    both = show_idlg and show_masked
    col_title = {
        "init": "Random Init",
        "iDLG": "iDLG" if both else "Recovered",
        "masked": "Masked" if both else "Recovered",
        "GT": "Ground Truth",
    }

    total_frames = max_frames + 5

    gif_frames = []
    for t in range(total_frames):
        frame_t = min(t, max_frames - 1)

        fig_w = max(4.0, 1.8 * n_cols + 0.6)
        fig_h = max(3.0, 1.8 * n_exp + 0.4)
        fig, axes = plt.subplots(
            n_exp, n_cols, figsize=(fig_w, fig_h),
            squeeze=False,
            gridspec_kw={'hspace': 0.45, 'wspace': 0.05},
        )

        frame_iter = 0
        for r in results_list:
            for mk in ["iDLG", "iDLG_masked"]:
                frs = r.get("recon_frames", {}).get(mk, [])
                if frs:
                    frame_iter = frs[min(frame_t, len(frs) - 1)].get('iter', 0)
                    break
            else:
                continue
            break
        fig.suptitle(f"Iteration {frame_iter}", fontsize=9)

        cmap = 'gray' if dataset == 'MNIST' else None

        for row_idx, result in enumerate(results_list):
            y_pos = 1.0 - (row_idx + 0.5) / n_exp
            fig.text(0.005, y_pos, f"Exp {row_idx}", va='center', ha='left',
                     fontsize=6, fontweight='bold')

            for col_idx, col in enumerate(cols):
                ax = axes[row_idx, col_idx]
                ax.axis('off')

                if col == "GT":
                    ax.imshow(_to_pil(result['gt_data'][0]), cmap=cmap,
                              interpolation='nearest', aspect='equal')
                    if row_idx == 0:
                        ax.set_title(col_title["GT"], fontsize=7, pad=2)

                elif col == "init":
                    init_np = None
                    for mk in ["iDLG", "iDLG_masked"]:
                        init_np = result.get("init_frames", {}).get(mk)
                        if init_np is not None:
                            break
                    if init_np is not None:
                        ax.imshow(_to_pil(init_np[0]), cmap=cmap,
                                  interpolation='nearest', aspect='equal')
                    else:
                        ax.set_facecolor('#aaaaaa')
                    if row_idx == 0:
                        ax.set_title(col_title["init"], fontsize=7, pad=2)

                else:
                    mk = "iDLG" if col == "iDLG" else "iDLG_masked"
                    frs = result.get("recon_frames", {}).get(mk, [])
                    if frs:
                        fd = frs[min(frame_t, len(frs) - 1)]
                        ax.imshow(_to_pil(fd['dummy'][0]), cmap=cmap,
                                  interpolation='nearest', aspect='equal')
                        loss_v = fd.get('loss', float('inf'))
                        mse_v = fd.get('mse', float('inf'))
                        loss_str = f"{loss_v:.2e}" if np.isfinite(loss_v) else "inf"
                        ax.text(
                            0.5, -0.06,
                            f"GLoss:{loss_str}\nL2Loss:{mse_v:.4f}",
                            transform=ax.transAxes,
                            fontsize=5, va='top', ha='center',
                            clip_on=False,
                        )
                    else:
                        ax.set_facecolor('#aaaaaa')
                    if row_idx == 0:
                        ax.set_title(col_title.get(col, col), fontsize=7, pad=2)

        plt.tight_layout(rect=(0.04, 0.0, 1.0, 0.96))

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=125, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        gif_frames.append(PILImage.open(buf).copy())
        buf.close()

    if not gif_frames:
        return None

    job_id = "INTERACTIVE" if os.environ.get("LSB_INTERACTIVE") == "Y" else os.environ.get("LSB_JOBID", "")
    prefix = f"{timestamp_str}_{job_id}" if job_id else timestamp_str
    out_path = os.path.join(save_dir, f"{prefix}_{block_idx}_anim.gif")

    duration_ms = max(1, int(1000 / fps))
    parent = os.path.dirname(out_path)
    if parent and not safe_makedirs(parent):
        return None
    try:
        gif_frames[0].save(
            out_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    except OSError as e:
        print(f"[WARNING] Failed to save GIF {out_path}: {e}")
        return None

    safe_chmod(out_path)
    print("Saved animated GIF to:", out_path)
    return out_path
```

- [ ] **Step 4: Create `stable_ginv/viz/restart.py`** (verbatim `_restart_stats` / `_gain_ci_str` / `save_restart_curve` / `save_restart_images`)

```python
"""Restart-curve CSV/PNG and representative per-restart image grid (Phase 7).

Extracted verbatim from helper/visualization.py: per-restart PSNR statistics,
paired restart-gain confidence intervals (vs k=1), the restart-curve CSV+PNG
writer, and the representative per-restart reconstruction image grid.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stable_ginv.io import safe_savefig, safe_write
from stable_ginv.stats import normality_str_from_ci, paired_t_ci

sns.set_theme(style="whitegrid")


def _restart_stats(per_exp_lists):
    arr = np.array(
        [[v if v is not None else float("nan") for v in row] for row in per_exp_lists],
        dtype=float,
    )
    n = np.sum(~np.isnan(arr), axis=0).clip(min=1)
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
    sems = stds / np.sqrt(n)
    return means, stds, sems


def _gain_ci_str(all_rows, k, fmt=".3f"):
    pairs = [
        (row[0], row[k - 1]) for row in (all_rows or [])
        if row and row[0] is not None and row[k - 1] is not None
    ]
    if len(pairs) < 2:
        return "n/a", None
    x_k = [p[1] for p in pairs]
    x_1 = [p[0] for p in pairs]
    ci = paired_t_ci(x_k, x_1)
    sign = "+" if ci["mean_diff"] >= 0 else ""
    s = (
        f"{sign}{ci['mean_diff']:{fmt}} "
        f"[{ci['ci_low']:{fmt}}, {ci['ci_high']:{fmt}}]"
        f" p={ci['p_value']:.3f}"
    )
    return s, ci


def save_restart_curve(save_dir, timestamp_str, num_restarts, network_name, dataset,
                       mask_mode, psnr_per_restart_idlg_all,
                       psnr_per_restart_masked_all, mse_per_restart_idlg_all,
                       mse_per_restart_masked_all):
    """Save restart-curve CSV/PNG and print paired gain summaries."""
    restart_csv_path = os.path.join(save_dir, f"restart_curve_{timestamp_str}.csv")
    restart_x = list(range(1, num_restarts + 1))

    # Restart statistics do not depend on k, so compute them once per method
    # rather than recomputing the full arrays inside the restart-count loop.
    means_i = stds_i = means_m = stds_m = None
    if psnr_per_restart_idlg_all:
        means_i, stds_i, _ = _restart_stats(psnr_per_restart_idlg_all)
    if psnr_per_restart_masked_all:
        means_m, stds_m, _ = _restart_stats(psnr_per_restart_masked_all)

    rows_restart = []
    for k in range(num_restarts):
        row = {"num_restarts": k + 1}
        if psnr_per_restart_idlg_all:
            row["mean_psnr_idlg"] = round(float(means_i[k]), 5)
            row["std_psnr_idlg"] = round(float(stds_i[k]), 5)
        if psnr_per_restart_masked_all:
            row["mean_psnr_masked"] = round(float(means_m[k]), 5)
            row["std_psnr_masked"] = round(float(stds_m[k]), 5)
        rows_restart.append(row)

    restart_fieldnames = ["num_restarts"]
    if psnr_per_restart_idlg_all:
        restart_fieldnames += ["mean_psnr_idlg", "std_psnr_idlg"]
    if psnr_per_restart_masked_all:
        restart_fieldnames += ["mean_psnr_masked", "std_psnr_masked"]

    def _write_restart_csv(f):
        writer = csv.DictWriter(f, fieldnames=restart_fieldnames)
        writer.writeheader()
        writer.writerows(rows_restart)

    safe_write(restart_csv_path, _write_restart_csv, newline="")

    line_rows = []
    for method_label, per_exp_lists in [
        ("iDLG (no mask)", psnr_per_restart_idlg_all),
        (f"masked ({mask_mode})", psnr_per_restart_masked_all),
    ]:
        for row in per_exp_lists or []:
            for restart_count, value in zip(restart_x, row):
                if value is not None and np.isfinite(value):
                    line_rows.append({
                        "Number of restarts used": restart_count,
                        "Best PSNR (dB)": float(value),
                        "Method": method_label,
                    })

    fig, ax = plt.subplots(figsize=(6, 4))
    if line_rows:
        sns.lineplot(
            data=pd.DataFrame(line_rows),
            x="Number of restarts used",
            y="Best PSNR (dB)",
            hue="Method",
            style="Method",
            markers=True,
            dashes=False,
            errorbar="se",
            ax=ax,
        )
    ax.set_xlabel("Number of restarts used")
    ax.set_ylabel("Mean best PSNR (dB) ± SEM")
    ax.set_title(f"Effect of restarts — {network_name} / {dataset}")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.12)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    restart_plot_path = os.path.join(save_dir, f"restart_curve_{timestamp_str}.png")
    safe_savefig(fig, restart_plot_path, dpi=200)
    plt.close(fig)
    print(f"\nRestart curve saved: {restart_csv_path}, {restart_plot_path}")

    gain_ks = [k for k in [3, 5, 10] if k <= num_restarts]
    gain_rows_extra = {k: {} for k in gain_ks}

    if gain_ks:
        print("\nRestart gain summary vs k=1 (95% paired CI):")
        header = f"  {'':10s}"
        for k in gain_ks:
            header += f"  PSNR gain k={k:<3d}              MSE gain k={k:<3d}    "
        print(header)

        for method_label, psnr_all, mse_all in [
            ("iDLG", psnr_per_restart_idlg_all, mse_per_restart_idlg_all),
            ("masked", psnr_per_restart_masked_all, mse_per_restart_masked_all),
        ]:
            if not psnr_all and not mse_all:
                continue
            gain_key = "idlg" if method_label == "iDLG" else "masked"
            row_str = f"  {method_label:<10s}"
            norm_str = f"  {'':10s}"
            for k in gain_ks:
                psnr_str, psnr_ci = _gain_ci_str(psnr_all, k, ".3f")
                mse_str, mse_ci = _gain_ci_str(mse_all, k, ".5f")
                row_str += f"  {psnr_str:<32s}  {mse_str:<32s}"
                psnr_norm = normality_str_from_ci(psnr_ci) if psnr_ci else "normality: n/a"
                mse_norm = normality_str_from_ci(mse_ci) if mse_ci else "normality: n/a"
                norm_str += f"  {psnr_norm:<44s}  {mse_norm:<44s}"
                if psnr_ci is not None:
                    gain_rows_extra[k][f"gain_psnr_{gain_key}"] = round(psnr_ci["mean_diff"], 5)
                    gain_rows_extra[k][f"ci_low_psnr_{gain_key}"] = round(psnr_ci["ci_low"], 5)
                    gain_rows_extra[k][f"ci_high_psnr_{gain_key}"] = round(psnr_ci["ci_high"], 5)
                    gain_rows_extra[k][f"normality_psnr_{gain_key}"] = normality_str_from_ci(psnr_ci)
                if mse_ci is not None:
                    gain_rows_extra[k][f"gain_mse_{gain_key}"] = round(mse_ci["mean_diff"], 7)
                    gain_rows_extra[k][f"ci_low_mse_{gain_key}"] = round(mse_ci["ci_low"], 7)
                    gain_rows_extra[k][f"ci_high_mse_{gain_key}"] = round(mse_ci["ci_high"], 7)
                    gain_rows_extra[k][f"normality_mse_{gain_key}"] = normality_str_from_ci(mse_ci)
            print(row_str)
            print(norm_str)

    if gain_ks and any(gain_rows_extra[k] for k in gain_ks):
        gain_extra_fields = []
        for k in gain_ks:
            gain_extra_fields += list(gain_rows_extra[k].keys())
        gain_extra_fields = list(dict.fromkeys(gain_extra_fields))
        new_fieldnames = restart_fieldnames + [
            f for f in gain_extra_fields if f not in restart_fieldnames
        ]
        for row in rows_restart:
            if row["num_restarts"] in gain_rows_extra:
                row.update(gain_rows_extra[row["num_restarts"]])

        def _rewrite_restart_csv(f):
            writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_restart)

        safe_write(restart_csv_path, _rewrite_restart_csv, newline="")

    return restart_csv_path, restart_plot_path


def save_restart_images(save_dir, timestamp_str, num_restarts, network_name, dataset,
                        mask_mode, num_exp, all_results_by_idx,
                        psnr_per_restart_idlg_all, psnr_per_restart_masked_all):
    """Save representative per-restart reconstruction image grid."""
    if not all_results_by_idx:
        return None
    display_ks = [k for k in [1, 3, 5, 10] if k <= num_restarts]
    if not display_ks:
        return None

    ref_k = max(display_ks)
    ref_pool = psnr_per_restart_idlg_all or psnr_per_restart_masked_all
    rep_result = None
    if num_exp > 1 and ref_pool:
        ref_vals = [row[ref_k - 1] for row in ref_pool if row and row[ref_k - 1] is not None]
        if ref_vals:
            median_val = float(np.median(ref_vals))
            sorted_results = sorted(
                all_results_by_idx.values(),
                key=lambda r: abs(
                    (
                        (r.get("psnr_per_restart_idlg") or
                         r.get("psnr_per_restart_masked") or [None])[ref_k - 1] or float("inf")
                    ) - median_val
                ),
            )
            rep_result = sorted_results[0]
    if rep_result is None:
        rep_result = next(iter(all_results_by_idx.values()))

    imgs_i = rep_result.get("img_per_restart_idlg") or []
    imgs_m = rep_result.get("img_per_restart_masked") or []
    psnrs_i = rep_result.get("psnr_per_restart_idlg") or []
    psnrs_m = rep_result.get("psnr_per_restart_masked") or []
    mses_i = rep_result.get("mse_per_restart_idlg") or []
    mses_m = rep_result.get("mse_per_restart_masked") or []
    ssims_i = rep_result.get("ssim_per_restart_idlg") or []
    ssims_m = rep_result.get("ssim_per_restart_masked") or []
    gt_np = rep_result["gt_data"]

    def _to_hwc(arr):
        if arr is None:
            return None
        x = arr[0]
        return x.transpose(1, 2, 0) if x.shape[0] > 1 else x[0]

    def _metric_str(k, mse_list, psnr_list, ssim_list):
        c = display_ks.index(k)
        parts = []
        m = mse_list[c] if c < len(mse_list) else None
        p = psnr_list[c] if c < len(psnr_list) else None
        s = ssim_list[c] if c < len(ssim_list) else None
        if m is not None and np.isfinite(m):
            parts.append(f"MSE: {m:.5f}")
        if p is not None and np.isfinite(p):
            parts.append(f"PSNR: {p:.2f} dB")
        if s is not None and np.isfinite(s):
            parts.append(f"SSIM: {s:.3f}")
        return "\n".join(parts)

    def _disp_list(src, is_gt=False):
        if is_gt:
            return [gt_np] * len(display_ks)
        return [src[k - 1] if (k - 1) < len(src) else None for k in display_ks]

    rows_fig = [("GT", _disp_list([], is_gt=True), [], [], [])]
    if imgs_i:
        rows_fig.append(("iDLG\n(baseline)", _disp_list(imgs_i),
                         _disp_list(psnrs_i), _disp_list(mses_i), _disp_list(ssims_i)))
    if imgs_m:
        rows_fig.append((f"masked\n({mask_mode})", _disp_list(imgs_m),
                         _disp_list(psnrs_m), _disp_list(mses_m), _disp_list(ssims_m)))

    n_cols_fig = len(display_ks)
    n_rows_fig = len(rows_fig)
    fig_img, axes = plt.subplots(
        n_rows_fig, n_cols_fig,
        figsize=(3.2 * n_cols_fig + 0.8, 3.8 * n_rows_fig),
        squeeze=False,
    )
    fig_img.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.02,
                            hspace=0.35, wspace=0.05)
    cmap = "gray" if gt_np.shape[1] == 1 else None
    for r_idx, (label, img_list, psnr_list, mse_list, ssim_list) in enumerate(rows_fig):
        for c_idx in range(n_cols_fig):
            k = display_ks[c_idx]
            ax = axes[r_idx][c_idx]
            hwc = _to_hwc(img_list[c_idx])
            if hwc is not None:
                ax.imshow(hwc.clip(0, 1), cmap=cmap)
            else:
                ax.set_facecolor("lightgray")
            ax.axis("off")
            if r_idx == 0:
                ax.set_title(f"k={k}", fontsize=10)
            if r_idx > 0 and psnr_list:
                ms = _metric_str(k, mse_list, psnr_list, ssim_list)
                if ms:
                    ax.text(0.5, -0.02, ms, transform=ax.transAxes,
                            ha="center", va="top", fontsize=7, linespacing=1.5)
        axes[r_idx][0].text(-0.08, 0.5, label, transform=axes[r_idx][0].transAxes,
                            ha="right", va="center", fontsize=9,
                            rotation=90, multialignment="center")

    title_suffix = f" (representative of {num_exp} images)" if num_exp > 1 else ""
    fig_img.suptitle(
        f"Best reconstruction after k restarts — {network_name} / {dataset}{title_suffix}",
        fontsize=10,
    )
    plt.tight_layout()
    img_fig_path = os.path.join(save_dir, f"restart_images_{timestamp_str}.png")
    safe_savefig(fig_img, img_fig_path, dpi=200)
    plt.close(fig_img)
    print(f"Restart image figure saved: {img_fig_path}")
    return img_fig_path
```

- [ ] **Step 5: Create `stable_ginv/viz/__init__.py`** (public viz API)

```python
"""Visualization: reconstruction panels, animated GIFs, and restart curves (Phase 7).

Canonical home for the plotting helpers previously in helper/visualization.py.
Importing this package sets the Agg backend and the seaborn whitegrid theme as a
side effect (matching the original module).
"""
from stable_ginv.viz.panels import (
    append_result_to_panel_buffers,
    create_panel_buffers,
    flush_recon_panel,
    save_recon_panel,
)
from stable_ginv.viz.gif import save_recon_gif
from stable_ginv.viz.restart import (
    _gain_ci_str,
    _restart_stats,
    save_restart_curve,
    save_restart_images,
)

__all__ = [
    "append_result_to_panel_buffers",
    "create_panel_buffers",
    "flush_recon_panel",
    "save_recon_panel",
    "save_recon_gif",
    "save_restart_curve",
    "save_restart_images",
    "_restart_stats",
    "_gain_ci_str",
]
```

- [ ] **Step 6: Replace `helper/visualization.py` with a re-export shim**

```bash
rm /home/mathias/GitHub/stable-ginv/helper/visualization.py
```
Then create `helper/visualization.py`:

```python
"""Backwards-compatibility shim — all code lives in stable_ginv/viz/ (Phase 7).

Re-exports the public surface so existing callers (the test suite, the lazy
loader in stable_ginv/cli/batch.py) keep importing unchanged. New code should
import from stable_ginv.viz.
"""
from stable_ginv.viz.panels import (
    append_result_to_panel_buffers,
    create_panel_buffers,
    flush_recon_panel,
    save_recon_panel,
)
from stable_ginv.viz.gif import save_recon_gif
from stable_ginv.viz.restart import (
    _gain_ci_str,
    _restart_stats,
    save_restart_curve,
    save_restart_images,
)

__all__ = [
    "append_result_to_panel_buffers",
    "create_panel_buffers",
    "flush_recon_panel",
    "save_recon_panel",
    "save_recon_gif",
    "save_restart_curve",
    "save_restart_images",
    "_restart_stats",
    "_gain_ci_str",
]
```

- [ ] **Step 7: Verify the split + shim resolve to the same objects**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "
from stable_ginv.viz import panels, restart
from stable_ginv.viz import save_recon_gif, save_restart_curve, create_panel_buffers, flush_recon_panel
import helper.visualization as shim
# Shim re-exports resolve to the canonical objects:
assert shim.save_recon_panel is panels.save_recon_panel
assert shim.flush_recon_panel is panels.flush_recon_panel
assert shim.save_restart_curve is restart.save_restart_curve
assert shim.save_recon_gif is save_recon_gif
# flush_recon_panel resolves save_recon_panel in panels' namespace (monkeypatch target):
assert flush_recon_panel.__module__ == 'stable_ginv.viz.panels'
print('viz split + shim OK')
"
```
Expected: prints `viz split + shim OK`. No error.

- [ ] **Step 8: Run the full suite (existing tests flow through the shim)**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
```
Expected: `tests/test_restart_curve.py` and `tests/golden/test_restart_curve_csv_golden.py` PASS (both call `save_restart_curve` through the shim → verbatim → identical bytes). **`tests/test_visualization.py` will FAIL** here — it monkeypatches `visualization.save_recon_panel` (the shim namespace) but `flush_recon_panel` now resolves `save_recon_panel` in `panels`. Step 9 fixes this; the failure is expected at this step.

- [ ] **Step 9: Commit the move (`refactor:`)**

```bash
git add stable_ginv/viz/ helper/visualization.py
git commit -m "refactor: extract stable_ginv/viz/ panels/gif/restart from helper/visualization.py"
```

- [ ] **Step 10: Repoint the panel test's monkeypatch target to the canonical module**

In `tests/test_visualization.py`, find:
```python
from helper import visualization
```
Replace with:
```python
from stable_ginv.viz import panels
```
Then replace the three `visualization.` references with `panels.`:
```python
    buffers = visualization.create_panel_buffers()
```
→
```python
    buffers = panels.create_panel_buffers()
```
and
```python
    monkeypatch.setattr(visualization, "save_recon_panel", fake_save_recon_panel)

    visualization.flush_recon_panel(
        {}, buffers, [], str(tmp_path), 0, "cifar100", "none", "timestamp", "both"
    )
```
→
```python
    monkeypatch.setattr(panels, "save_recon_panel", fake_save_recon_panel)

    panels.flush_recon_panel(
        {}, buffers, [], str(tmp_path), 0, "cifar100", "none", "timestamp", "both"
    )
```

- [ ] **Step 11: Run the full suite — now green**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
```
Expected: all tests PASS. `test_visualization.py` now patches `panels.save_recon_panel`, which `panels.flush_recon_panel` resolves, so `captured` is populated and the column-ordering asserts hold.

- [ ] **Step 12: Lint**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pyflakes stable_ginv/viz helper/visualization.py tests/test_visualization.py
```
Expected: no output. (The shim/`__init__` re-export names documented in `__all__`; every import in the submodules is used — `sns` via `set_theme`, `safe_*`/`paired_t_ci`/`normality_str_from_ci` in the bodies. If pyflakes is unavailable, use `conda run -n stable-ginv python -m py_compile stable_ginv/viz/*.py helper/visualization.py`.)

- [ ] **Step 13: Commit the test repoint (`test:`)**

```bash
git add tests/test_visualization.py
git commit -m "test: point panel test at stable_ginv.viz.panels monkeypatch target"
```

---

## Task 3: Repoint `cli/batch.py` lazy visualization loader at `stable_ginv.viz`

**Files:**
- Modify: `stable_ginv/cli/batch.py`

`_load_visualization_helpers()` imports six names from `helper.visualization`. Repoint that single import at the canonical package. The `except`-branch no-op fallbacks and the return tuples are unchanged.

- [ ] **Step 1: Repoint the import**

In `stable_ginv/cli/batch.py`, inside `_load_visualization_helpers`, find:
```python
        from helper.visualization import (
            append_result_to_panel_buffers,
            create_panel_buffers,
            flush_recon_panel,
            save_recon_gif,
            save_restart_curve,
            save_restart_images,
        )
```
Replace with:
```python
        from stable_ginv.viz import (
            append_result_to_panel_buffers,
            create_panel_buffers,
            flush_recon_panel,
            save_recon_gif,
            save_restart_curve,
            save_restart_images,
        )
```
Leave everything else in the function (the `return (...)` tuple, the `except Exception` fallback definitions, and the bottom `return (...)`) untouched.

- [ ] **Step 2: Verify the loader returns the canonical callables**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "
from stable_ginv.cli.batch import _load_visualization_helpers
funcs = _load_visualization_helpers()
names = [f.__name__ for f in funcs]
assert names == ['append_result_to_panel_buffers', 'create_panel_buffers', 'flush_recon_panel', 'save_recon_gif', 'save_restart_curve', 'save_restart_images'], names
assert all(f.__module__.startswith('stable_ginv.viz') for f in funcs), [f.__module__ for f in funcs]
print('batch.py viz loader OK:', names)
"
```
Expected: prints the six names and `batch.py viz loader OK`. No error.

- [ ] **Step 3: Confirm `--help` still works and the suite is green**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python iDLG_mask.py --help
conda run -n stable-ginv python -m pytest tests/ -q
conda run -n stable-ginv python -m pyflakes stable_ginv/cli/batch.py
```
Expected: help prints and exits 0; all tests PASS; no pyflakes output.

- [ ] **Step 4: Commit (`refactor:`)**

```bash
git add stable_ginv/cli/batch.py
git commit -m "refactor: load visualization helpers from stable_ginv.viz in cli/batch"
```

---

## Task 4: Point the restart-curve tests at the canonical import path

**Files:**
- Modify: `tests/test_restart_curve.py`
- Modify: `tests/golden/test_restart_curve_csv_golden.py`

Both currently import `save_restart_curve` via the `helper.visualization` shim. Repoint them at the canonical `stable_ginv.viz` (matching how Phase 4/5/6 repointed their tests). The CSV bytes are unchanged (verbatim move), so the golden still matches its fixture.

- [ ] **Step 1: Repoint `tests/test_restart_curve.py`**

Find:
```python
from helper.visualization import save_restart_curve
```
Replace with:
```python
from stable_ginv.viz import save_restart_curve
```

- [ ] **Step 2: Repoint `tests/golden/test_restart_curve_csv_golden.py`**

Find:
```python
from helper.visualization import save_restart_curve
```
Replace with:
```python
from stable_ginv.viz import save_restart_curve
```

- [ ] **Step 3: Run both — bytes/values must be unchanged**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/test_restart_curve.py tests/golden/test_restart_curve_csv_golden.py -q
```
Expected: all PASS. If the golden differs, STOP and debug — do not regenerate the fixture.

- [ ] **Step 4: Commit (`test:`)**

```bash
git add tests/test_restart_curve.py tests/golden/test_restart_curve_csv_golden.py
git commit -m "test: import restart-curve tests from canonical stable_ginv.viz"
```

---

## Task 5: Move the five `helper/plot_*.py` CLIs into `stable_ginv/viz/` + thin wrappers; repoint their tests

**Files:**
- Move: `helper/plot_combined_masking_sweep_summary.py` → `stable_ginv/viz/plot_combined_masking_sweep_summary.py`
- Move: `helper/plot_masking_sweep_csv.py` → `stable_ginv/viz/plot_masking_sweep_csv.py`
- Move: `helper/plot_model_parameter_counts.py` → `stable_ginv/viz/plot_model_parameter_counts.py`
- Move: `helper/plot_normality_scatter.py` → `stable_ginv/viz/plot_normality_scatter.py`
- Move: `helper/plot_paired_masking_violin.py` → `stable_ginv/viz/plot_paired_masking_violin.py`
- Create: `helper/plot_*.py` (×5, thin wrappers)
- Modify: `tests/test_plot_masking_sweep_csv.py`
- Modify: `tests/test_plot_paired_masking_violin.py`

Each CLI moves **whole** (every function body verbatim). The only edits per file: drop the now-misdirected `sys.path.insert(...)` bootstrap and its now-unused `import sys`, and repoint the `functions.io_utils` import to the canonical `stable_ginv.io` / `stable_ginv.registry` homes. The matplotlib/seaborn headers and `if __name__ == "__main__": main()` blocks stay. `from helper.Network import get_model` in `plot_model_parameter_counts.py` stays as-is (models phase is later). After moving, each `helper/plot_*.py` becomes a thin wrapper that re-adds the repo root to `sys.path` (so `python helper/plot_*.py ...` keeps working even without an editable install) and calls `main`.

- [ ] **Step 1: Move all five files (preserving history)**

```bash
cd /home/mathias/GitHub/stable-ginv
git mv helper/plot_combined_masking_sweep_summary.py stable_ginv/viz/plot_combined_masking_sweep_summary.py
git mv helper/plot_masking_sweep_csv.py stable_ginv/viz/plot_masking_sweep_csv.py
git mv helper/plot_model_parameter_counts.py stable_ginv/viz/plot_model_parameter_counts.py
git mv helper/plot_normality_scatter.py stable_ginv/viz/plot_normality_scatter.py
git mv helper/plot_paired_masking_violin.py stable_ginv/viz/plot_paired_masking_violin.py
```

- [ ] **Step 2: Repoint imports in `stable_ginv/viz/plot_combined_masking_sweep_summary.py`**

Find:
```python
import argparse
import csv
import os
import sys

import matplotlib
```
Replace with (drop `import sys`):
```python
import argparse
import csv
import os

import matplotlib
```
Then find:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.io_utils import safe_makedirs, safe_savefig, safe_write
```
Replace with:
```python
from stable_ginv.io import safe_makedirs, safe_savefig, safe_write
```

- [ ] **Step 3: Repoint imports in `stable_ginv/viz/plot_masking_sweep_csv.py`**

Find:
```python
import argparse
import csv
import hashlib
import json
import os
import sys

import matplotlib
```
Replace with (drop `import sys`):
```python
import argparse
import csv
import hashlib
import json
import os

import matplotlib
```
Then find:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from functions.io_utils import find_registry_entry, safe_makedirs, safe_savefig, safe_write
```
Replace with:
```python
from stable_ginv.io import safe_makedirs, safe_savefig, safe_write
from stable_ginv.registry import find_registry_entry
```

- [ ] **Step 4: Repoint imports in `stable_ginv/viz/plot_model_parameter_counts.py`**

Find:
```python
import argparse
import os
import sys

import matplotlib
```
Replace with (drop `import sys`):
```python
import argparse
import os

import matplotlib
```
Then find:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.io_utils import safe_savefig
from helper.Network import get_model
```
Replace with (keep `helper.Network`):
```python
from stable_ginv.io import safe_savefig
from helper.Network import get_model
```

- [ ] **Step 5: Repoint imports in `stable_ginv/viz/plot_normality_scatter.py`**

Find:
```python
import argparse
import json
import os
import re
import sys

import matplotlib
```
Replace with (drop `import sys`):
```python
import argparse
import json
import os
import re

import matplotlib
```
Then find:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.io_utils import safe_makedirs, safe_savefig
```
Replace with:
```python
from stable_ginv.io import safe_makedirs, safe_savefig
```

- [ ] **Step 6: Repoint imports in `stable_ginv/viz/plot_paired_masking_violin.py`**

Find:
```python
import argparse
import csv
import json
import os
import sys

import matplotlib
```
Replace with (drop `import sys`):
```python
import argparse
import csv
import json
import os

import matplotlib
```
Then find:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.io_utils import resolve_storage_paths, safe_makedirs, safe_savefig, safe_write
```
Replace with:
```python
from stable_ginv.io import resolve_storage_paths, safe_makedirs, safe_savefig, safe_write
```

- [ ] **Step 7: Create the five thin wrappers at the old `helper/plot_*.py` paths**

Create `helper/plot_combined_masking_sweep_summary.py`:
```python
"""Thin wrapper: moved to stable_ginv.viz.plot_combined_masking_sweep_summary (Phase 7).

Kept runnable as `python helper/plot_combined_masking_sweep_summary.py ...`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_ginv.viz.plot_combined_masking_sweep_summary import main  # noqa: E402

if __name__ == "__main__":
    main()
```

Create `helper/plot_masking_sweep_csv.py`:
```python
"""Thin wrapper: moved to stable_ginv.viz.plot_masking_sweep_csv (Phase 7).

Kept runnable as `python helper/plot_masking_sweep_csv.py <sweep_csv>` (see README).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_ginv.viz.plot_masking_sweep_csv import main  # noqa: E402

if __name__ == "__main__":
    main()
```

Create `helper/plot_model_parameter_counts.py`:
```python
"""Thin wrapper: moved to stable_ginv.viz.plot_model_parameter_counts (Phase 7).

Kept runnable as `python helper/plot_model_parameter_counts.py ...`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_ginv.viz.plot_model_parameter_counts import main  # noqa: E402

if __name__ == "__main__":
    main()
```

Create `helper/plot_normality_scatter.py`:
```python
"""Thin wrapper: moved to stable_ginv.viz.plot_normality_scatter (Phase 7).

Kept runnable as `python helper/plot_normality_scatter.py ...`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_ginv.viz.plot_normality_scatter import main  # noqa: E402

if __name__ == "__main__":
    main()
```

Create `helper/plot_paired_masking_violin.py`:
```python
"""Thin wrapper: moved to stable_ginv.viz.plot_paired_masking_violin (Phase 7).

Kept runnable as `python helper/plot_paired_masking_violin.py ...`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_ginv.viz.plot_paired_masking_violin import main  # noqa: E402

if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Verify the moved modules import and the wrappers re-export `main`**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "
import importlib
mods = [
    'plot_combined_masking_sweep_summary',
    'plot_masking_sweep_csv',
    'plot_model_parameter_counts',
    'plot_normality_scatter',
    'plot_paired_masking_violin',
]
for m in mods:
    canon = importlib.import_module('stable_ginv.viz.' + m)
    wrap = importlib.import_module('helper.' + m)
    assert wrap.main is canon.main, m
print('plot CLIs moved + wrappers OK')
"
```
Expected: prints `plot CLIs moved + wrappers OK`. No error. (Confirms canonical imports resolve and each wrapper re-exports the canonical `main`.)

- [ ] **Step 9: Confirm a wrapper's `--help` runs as a script (back-compat)**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python helper/plot_masking_sweep_csv.py --help
```
Expected: the argparse help prints and the process exits 0 (the wrapper's `sys.path` insert lets it find `stable_ginv` without relying on the editable install).

- [ ] **Step 10: Lint the moved modules + wrappers**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pyflakes stable_ginv/viz/plot_*.py helper/plot_*.py
```
Expected: no output. (Dropping `import sys` from the moved modules removes the only `sys` use; the wrappers use `sys`+`os`. If pyflakes is unavailable, `conda run -n stable-ginv python -m py_compile stable_ginv/viz/plot_*.py helper/plot_*.py`.)

- [ ] **Step 11: Commit the move (`refactor:`)**

```bash
git add stable_ginv/viz/plot_*.py helper/plot_*.py
git commit -m "refactor: move helper/plot_* CLIs into stable_ginv/viz with thin wrappers"
```

- [ ] **Step 12: Repoint the two plot tests at the canonical modules**

In `tests/test_plot_masking_sweep_csv.py`, find:
```python
from helper.plot_masking_sweep_csv import _baseline_row
```
Replace with:
```python
from stable_ginv.viz.plot_masking_sweep_csv import _baseline_row
```

In `tests/test_plot_paired_masking_violin.py`, find:
```python
from helper.plot_paired_masking_violin import _load_paired_rows, _paired_rows
```
Replace with:
```python
from stable_ginv.viz.plot_paired_masking_violin import _load_paired_rows, _paired_rows
```

- [ ] **Step 13: Run the full suite — green**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
```
Expected: all tests PASS (the two plot tests now import the canonical helpers; behavior is identical to the verbatim move).

- [ ] **Step 14: Commit the test repoint (`test:`)**

```bash
git add tests/test_plot_masking_sweep_csv.py tests/test_plot_paired_masking_violin.py
git commit -m "test: import plot CLI tests from canonical stable_ginv.viz modules"
```

---

## Task 6: Update the handover and finalize

**Files:**
- Modify: `docs/handover/HANDOVER_RESTRUCTURE.md`

- [ ] **Step 1: Mark Phase 7 complete in the Status section**

Find:
```markdown
- [ ] Phase 7 — `stable_ginv/viz/`.
```
Replace with:
```markdown
- [x] Phase 7 — `stable_ginv/viz/` (panels, gif, restart) + the five `plot_*` CLIs.
      `helper/visualization.py` is a shim; each `helper/plot_*.py` is a thin runnable
      wrapper. Restart-curve CSV gain-column golden added.
```

- [ ] **Step 2: Replace the `## Next phase` section**

Find the entire `## Next phase` block and replace it with:
```markdown
## Next phase

Write the Phase 8 plan from the spec, then implement. Phase 8 extracts
`stable_ginv/jacobian/` from `functions/jacobian_rank_sweep.py` (rank sweep + its CLI)
and moves `functions/rank_reconstruction_plot.py` into `stable_ginv/viz/` (the
rank-vs-reconstruction plot CLI), replacing both with thin wrappers. Lock the relevant
outputs with goldens just-in-time before moving them. Note the pre-existing bug to fix
separately (not as part of the move): `functions/rank_reconstruction_plot.py` has
f-strings missing placeholders. Keep this file's Status section current at every phase
boundary.

Deferred (not yet done): `functions/idlg_cli.py` → `stable_ginv/cli/args.py`,
`functions/Dataset.py`/`functions/consts.py` → `stable_ginv/data/`, and
`helper/Network.py` → `stable_ginv/models/` remain imported from their current paths
(by `stable_ginv/cli/batch.py`, `stable_ginv/recon/runner.py`, and
`stable_ginv/viz/plot_model_parameter_counts.py`).
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
git commit -m "docs: mark Phase 7 complete in HANDOVER_RESTRUCTURE"
```

---

## Self-review checklist

**Spec coverage (Section 1 mapping `helper/visualization.py` → `stable_ginv/viz/` (panels, restart); `helper/plot_*.py` → `stable_ginv/viz/` standalone CLIs):**
- [x] `helper/visualization.py` split into `panels.py` (panels) + `gif.py` (GIFs) + `restart.py` (restart curves) under `stable_ginv/viz/`, re-exported by `__init__.py` (Task 2). All three submodules < 300 lines.
- [x] `helper/visualization.py` → re-export shim; every existing caller keeps working (Task 2; lazy loader repointed Task 3).
- [x] The five `helper/plot_*.py` standalone CLIs moved into `stable_ginv/viz/` with thin runnable wrappers (Task 5).
- [x] Goldens added just-in-time **before** the move: the restart-curve CSV gain-column golden (Task 1) locks the highest-risk, previously-untested output (the `stable_ginv.stats`-coupled gain rewrite). PNG/GIF binaries are intentionally not golden-locked (not byte-stable); panel ordering stays guarded by `test_visualization.py` and the restart no-gain CSV by `test_restart_curve.py`.
- [x] `stable_ginv/cli/batch.py` lazy `_load_visualization_helpers` repointed at `stable_ginv.viz` (Task 3) — the handover's explicit Phase 7 instruction.
- [x] Categorized commits: `test:` (golden) / `refactor:` (viz split) / `test:` (panel test) / `refactor:` (batch loader) / `test:` (restart tests) / `refactor:` (plot CLIs) / `test:` (plot tests) / `docs:`.
- [x] `HANDOVER_RESTRUCTURE.md` Status + Next phase updated (Task 6).

**Placeholder scan:** no TBD/TODO/"handle edge cases"/"similar to". `panels.py`/`gif.py`/`restart.py` are full verbatim bodies; the shim, `__init__`, golden, and five wrappers are complete; every plot-file edit is a spelled-out find/replace.

**Type/name consistency:**
- [x] The viz public surface (`append_result_to_panel_buffers`, `create_panel_buffers`, `flush_recon_panel`, `save_recon_panel`, `save_recon_gif`, `save_restart_curve`, `save_restart_images`, `_restart_stats`, `_gain_ci_str`) is identical across `__init__.py` `__all__` (Task 2 Step 5), the shim (Step 6), and the six names imported by `cli/batch.py` (Task 3).
- [x] `save_restart_curve(save_dir, timestamp_str, num_restarts, network_name, dataset, mask_mode, psnr_per_restart_idlg_all, psnr_per_restart_masked_all, mse_per_restart_idlg_all, mse_per_restart_masked_all)` signature in `restart.py` (Task 2 Step 4) matches the golden's keyword call (Task 1) and `test_restart_curve.py`.
- [x] `panels.flush_recon_panel` / `panels.save_recon_panel` are in the **same** module, so the `test_visualization.py` monkeypatch on `panels.save_recon_panel` (Task 2 Step 10) is resolved by `flush_recon_panel` — verified in Task 2 Step 7.
- [x] Plot wrappers re-export `main`; the two tested helpers (`_baseline_row`; `_load_paired_rows`/`_paired_rows`) are imported from the canonical `stable_ginv.viz.plot_*` modules after Task 5 Step 12.

**Potential issues to watch:**
- **Seaborn theme side effect:** the original `visualization.py` set `matplotlib.use("Agg")` + `sns.set_theme(style="whitegrid")` once at import. Each submodule replicates this so global render state is identical regardless of which submodule loads first. `sns` is "used" via `set_theme`, so no pyflakes unused-import warning even in `panels.py`/`gif.py`.
- **`test_visualization.py` transiently red:** after the move (Task 2 Step 8) but before the monkeypatch repoint (Step 10), this one test fails — expected and called out. The task ends green at Step 11. No other task leaves the suite red at its boundary.
- **Dropped `sys.path` bootstrap:** the moved plot modules drop the `sys.path.insert(..., "..")` that, post-move, would point at `stable_ginv/` (wrong) instead of the repo root. The repo root is supplied by the editable install (for `-m`/import use) and by each thin wrapper's own `sys.path.insert` (for `python helper/plot_*.py` script use). `import sys` is removed because it was the only `sys` user in each file.
- **`helper.Network` kept:** `plot_model_parameter_counts.py` still imports `get_model` from `helper.Network` (models phase is later) — do not move it here.
- **Out of scope:** `functions/rank_reconstruction_plot.py` (Phase 8) and `helper/plots.py` (pre-existing bug) are NOT touched. Do not move them in this phase.
- **No PNG/GIF byte goldens:** matplotlib raster output is not reproducible across versions, so plot *images* are guarded only by the verbatim-move discipline plus the import/wiring checks; the deterministic CSV output is golden-locked.
</content>
</invoke>
