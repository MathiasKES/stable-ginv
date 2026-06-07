# stable-ginv

Research codebase for studying gradient inversion attacks on federated learning.
Bachelor's thesis project — Technical University of Denmark (DTU), 2026.

---

## What This Does

Implements the gradient inversion attack family — **DLG** (Zhu et al., 2019),
**iDLG** (Zhao et al., 2020), and **Inverting Gradients** (Geiping et al., 2020,
cosine-similarity matching with a total-variation prior).

An adversary who receives gradient updates can reconstruct the private training
images that produced them. This project studies how **gradient masking** —
selectively withholding parts of the gradient — affects reconstruction, using
masking as a diagnostic tool to test whether a failed attack reflects missing
information or merely a poor optimization signal.

All experiment logic lives in the `stable_ginv` package. Commands below are run
from the repository root.

---

## Setup

**A CUDA GPU is strongly recommended.** The reconstruction runner parallelises
across all visible `cuda:N` devices. If no GPU is detected it falls back to a
single CPU worker, which is **much slower** and intended only for small test
configurations — not for real experiments.

```bash
# 1. Clone and enter the repo
git clone https://github.com/MathiasKES/stable-ginv.git
cd stable-ginv

# 2. Create the environment (CUDA 12.6 PyTorch build) and activate it
conda env create -f env/environment.yml
conda activate stable-ginv          # env name is "stable-ginv" (see env/environment.yml)

# CPU-only machine instead: conda env create -f env/environment-cpu.yml

# 3. Make the package importable
pip install -e .

# 4. Verify the GPU is visible (prints True and the device name)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**On the DTU HPC cluster**, load the CUDA module and prepend the conda library
path before running Python (otherwise SciPy/Matplotlib may load the system C++
runtime and fail with `CXXABI_1.3.15 not found`):

```bash
source scripts/init.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

---

## Running Experiments

The batch runner parses arguments, schedules workers across the available GPUs,
and writes the CSV, reconstruction panels, and (optionally) animated GIFs:

```bash
python -m stable_ginv.cli.batch \
    --dataset cifar10 --network resnet18 --methods both \
    --mask_mode gradsize_topfrac --gradsize_topfrac 0.5 --num_exp 10
```

See all arguments and defaults:

```bash
python -m stable_ginv.cli.batch --help
```

---

## Plotting & Analysis

Each plotting tool is a runnable module:

```bash
# Masking-sweep line + bar charts from a sweep CSV
python -m stable_ginv.viz.plot_masking_sweep_csv results/masking_sweeps/<sweep>.csv

# Combine several sweep-summary CSVs into one figure
python -m stable_ginv.viz.plot_combined_masking_sweep_summary <summary1.csv> <summary2.csv> ...

# Paired baseline-vs-masked violin plots from registry keys
python -m stable_ginv.viz.plot_paired_masking_violin --baseline_key <k> --masked_key <k> --out_dir results

# Normality scatter, model parameter counts, rank-vs-reconstruction
python -m stable_ginv.viz.plot_normality_scatter --help
python -m stable_ginv.viz.plot_model_parameter_counts
python -m stable_ginv.viz.plot_rank_reconstruction --help

# Jacobian rank sweep (serial or parallel via mp.spawn)
python -m stable_ginv.jacobian.cli --help

# Paired PSNR/MSE boxplots for registry-keyed layer ablations
python helper/plot_registry_key_boxplots.py --keys <k1> <k2> --metric psnr --out_dir results/ablation_boxplots
```

Pass `--help` to any of the above for its full option list.

---

## Datasets

MNIST and CIFAR-10/100 download automatically via torchvision on first use.

LFW requires manual setup:
1. Download `lfw-deepfunneled.tgz` from the LFW project page.
2. Extract to `data/lfw/lfw-deepfunneled/`.

---

## Key CLI Arguments (`stable_ginv.cli.batch`)

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

The default reconstruction threshold is `--threshold_mse 0.01`. The summary CSV
includes `network` and `dataset`, and generated plot titles and filenames show
both (filenames also include the threshold). If a sweep contains a masked
`topfrac=1.0` row, it is kept alongside the unmasked iDLG baseline as a separate
`0% masked` point/bar.

---

## Project Structure

```
stable-ginv/
├── stable_ginv/          The package — all experiment logic
│   ├── cli/              Batch entry point (python -m stable_ginv.cli.batch)
│   ├── experiment/       GPU scheduling, result aggregation, restart selection
│   ├── recon/            Reconstruction worker, optimizer, label inference, scheduler
│   ├── masking/          Gradient masking strategies
│   ├── metrics/          PSNR, SSIM, total variation, Jacobian rank, grad-match loss
│   ├── registry/         Baseline/masked registries and CSV summaries
│   ├── stats/            Paired t-tests and confidence intervals
│   ├── io/               Storage paths, safe writes, CSV/text helpers
│   ├── jacobian/         Jacobian rank sweep (compute core + CLI)
│   └── viz/              Panels, GIFs, restart curves, and the plot_* CLIs
│
├── functions/            Dataset loader, normalization constants, CLI arg parser
├── helper/               Model factory (Network.py) + standalone analysis plots
├── tests/                Unit + golden regression tests
├── scripts/              DTU HPC job scripts (LSF scheduler)
├── docs/                 Sphinx API docs + codebase map
├── archive/              Retired scripts kept for reference (not imported)
└── artifacts/            Generated plots, figures, and data
```

---

## Documentation

- [`docs/codebase.md`](docs/codebase.md) — codebase map, key functions, config keys, output schema
- Sphinx API reference — built from the `stable_ginv` package (see `docs/sphinx/`)

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE). You are free to use,
copy, modify, and distribute this code, including for academic and commercial
purposes, provided the copyright notice is retained.

---

## Citation

If you use this code in academic work, please cite this project. Citation
metadata is also provided in [`CITATION.cff`](CITATION.cff).

```bibtex
@misc{stablegradinv2026,
  author = {Aqraou, Alfred and Afif, Ali and S{\o}rensen, Mathias},
  title  = {Stabilizing Gradient Inversion in Federated Learning},
  year   = {2026},
  note   = {Bachelor's thesis, Technical University of Denmark (DTU)},
  howpublished = {\url{https://github.com/MathiasKES/stable-ginv}}
}
```
