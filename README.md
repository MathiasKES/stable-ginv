# stable-ginv

Research codebase for studying gradient inversion attacks on federated learning.  
Bachelor's thesis project — DTU, 2026.

---

## What This Does

Implements and extends the **iDLG gradient inversion attack** (Zhao et al., 2020).  
An adversary who receives gradient updates can reconstruct the private training images that produced them.  
This project studies how **gradient masking** — selectively withholding parts of the gradient — degrades reconstruction quality.

---

## Quick Start

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate stable-ginv   # check environment.yml for exact name

# On DTU HPC: load modules first
source scripts/init.sh

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

Run a single experiment (CPU-safe for testing):

```bash
python run_single_exp.py
```

Run a batch across all GPUs:

```bash
python iDLG_mask.py --dataset cifar10 --network resnet18 --methods both \
    --mask_mode gradsize_topfrac --gradsize_topfrac 0.5 --num_exp 10
```

---

## Project Structure

```
stable-ginv/
├── iDLG_mask.py          Batch runner — multi-GPU, argument parsing, CSV + PNG output
├── run_single_exp.py     Single experiment — GIF, Jacobian rank, per-image SSIM/PSNR
│
├── functions/
│   ├── masking.py        All gradient masking strategies (build_gradient_mask, etc.)
│   ├── io_utils.py       Baseline registry, paired t-test, CSV helpers
│   ├── Dataset.py        LFW dataset loader
│   ├── consts.py         Normalization constants (mean/std per dataset)
│   └── jacobian_rank_sweep.py  Serial Jacobian rank sweep
│
├── helper/
│   ├── Network.py        Model factory (get_model) + custom CNNs (LeNet, MediumCNN, etc.)
│   ├── metrics.py        PSNR, SSIM, total variation, Jacobian rank, grad match loss
│   ├── training_utils.py make_scheduler
│   ├── visualization.py  save_recon_panel, save_recon_gif, restart outputs
│   └── plot_masking_sweep_csv.py  Plot registry-backed masking sweep CSVs
│
├── archive/              Retired scripts — not imported anywhere, kept for reference
├── scripts/              DTU HPC job scripts (LSF scheduler)
├── docs/                 Handover files, improvement plan, codebase map
└── invertinggradients/   Git submodule — Geiping et al. reference implementation
```

---

## Datasets

MNIST and CIFAR-10/100 download automatically via torchvision.

LFW requires manual setup:
1. Download `lfw-deepfunneled.tgz` from the LFW project page.
2. Extract to `data/lfw/lfw-deepfunneled/`.

---

## Key CLI Arguments (`iDLG_mask.py`)

| Argument | Options | Default |
|----------|---------|---------|
| `--dataset` | `MNIST`, `cifar10`, `cifar100`, `lfw` | `cifar100` |
| `--network` | `resnet18`, `vgg13`, `LeNet`, `MediumCNN`, … | `resnet18` |
| `--methods` | `idlg`, `masked`, `both` | `idlg` |
| `--mask_mode` | see below | `gradsize_topfrac` |
| `--optimizer` | `lbfgs`, `adam`, `adamw`, `signed_adam`, `signed_adamw` | `lbfgs` |
| `--num_exp` | int | `10` |
| `--num_restarts` | int | `3` |
| `--iteration` | int | `1000` |

Masking modes: `none`, `gradsize_topk`, `gradsize_topfrac`,
`gradsize_topk_entries`, `gradsize_topfrac_entries`,
`gradsize_topk_entries_layer`, `gradsize_topfrac_entries_layer`,
`prefix`, `prefix_topk`, `prefix_topfrac`, `prefix_topk_entries`,
`prefix_topfrac_entries`, `prefix_topk_entries_layer`,
`prefix_topfrac_entries_layer`.

---

## Output

```
results/exp_results_<network>.csv               per-run metrics
results/{timestamp}_{jobid}_{block}.png         reconstruction panel
results/{timestamp}_{jobid}_{block}_anim.gif    animated reconstruction (if --save_gif)
results/baselines/idlg_baselines_registry.json  iDLG baseline store for paired comparison
results/baselines/masked_registry.json          masked-run registry
results/masking_sweeps/*.csv                    compact sweep rows for gradsize_topfrac_entries_layer
```

Plot a masking sweep CSV with:

```bash
python helper/plot_masking_sweep_csv.py results/masking_sweeps/<sweep_csv>.csv
```

The default reconstruction threshold is `--threshold_mse 0.01`.
The summary CSV includes `network` and `dataset`, and the generated plot titles show both.
If the sweep contains a masked `topfrac=1.0` row, it is kept alongside the unmasked iDLG baseline as a separate `0% masked` point/bar.

---

## Documentation

- [`docs/CLAUDE.md`](docs/CLAUDE.md) — full codebase map, function signatures, config keys
- [`docs/handover/HANDOVER_RESEARCHER.md`](docs/handover/HANDOVER_RESEARCHER.md) — for developers continuing the work
- [`docs/handover/HANDOVER_REVIEWER.md`](docs/handover/HANDOVER_REVIEWER.md) — for code reviewers
- [`docs/handover/HANDOVER_COLLABORATOR.md`](docs/handover/HANDOVER_COLLABORATOR.md) — for new thesis collaborators
- [`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md) — two-phase code improvement roadmap
