# Handover: Thesis Collaborator

**Audience:** A new collaborator joining the bachelor's project who needs to understand the research context, set up the environment, and run their first experiment.

**Last updated:** 2026-05-31

---

## 1. What We Are Studying

**Federated learning** is a machine learning paradigm where multiple clients (e.g., hospitals, phones) collaboratively train a shared model without sharing their raw data. Each client trains locally and only sends model gradient updates to a central server.

**The privacy problem:** A line of work showed that an adversary who sees these gradient updates can reconstruct the private training images with high fidelity — **DLG** (Zhu et al., 2019), **iDLG** (Zhao et al., 2020), and **Inverting Gradients** (Geiping et al., 2020). We implement all three; the reconstruction loop uses Geiping et al.'s cosine-similarity gradient matching with a total-variation prior, plus the iDLG label-recovery trick.

**Our research question:** Can a client *mask* some of its gradients before sending them — hiding information from the adversary — while still making training useful? We study how different masking strategies affect the adversary's ability to reconstruct private images.

The codebase implements the attack and measures its success under various masking conditions.

---

## 2. The Attack — How It Works

A federated learning client holds a private image `x` with label `y`. During training it computes:

```
g = dL(f(x), y) / dθ
```

where `f` is the neural network, `L` is the cross-entropy loss, and `θ` are the model parameters. The client sends `g` to the server.

The **adversary** (the server, or an eavesdropper) receives `g` and tries to find a dummy image `x'` such that:

```
minimize  1 - cosine_similarity(dL(f(x'), y') / dθ,  g)
```

Starting from random noise, the adversary iteratively updates `x'` until its gradients match the observed `g`. If the gradients are informative enough, `x'` converges to the original `x`.

The **iDLG trick** recovers the true label `y'` analytically from the gradient of the last fully-connected layer's weight, making label guessing unnecessary. It is computed once from the original unmasked gradient, so it works regardless of which gradients the mask keeps.

**Masking:** Instead of sending the full gradient `g`, the client sends a masked version `g_masked` where some tensors (or individual entries) are zeroed out. The code supports several tensor-wise and entry-wise masking strategies. The adversary only sees `g_masked`.

---

## 3. How We Measure Success

**PSNR (Peak Signal-to-Noise Ratio):** Measures pixel-level similarity between the reconstructed image and the original. Expressed in dB.
- > 30 dB: attack largely succeeded
- 20–30 dB: partial reconstruction
- < 20 dB: attack failed

**SSIM (Structural Similarity Index):** 0–1 scale. Measures perceptual similarity (edges, texture, luminance). More aligned with how humans perceive image quality than PSNR.

**Jacobian rank:** A theoretical measure of how much information the gradient contains about the input image. We compute the rank of the Jacobian matrix `dg/dx`. Higher rank = more information = easier for the adversary.

---

## 4. Code → Paper Mapping

| Paper concept | Code location |
|---------------|--------------|
| Gradient computation | `run_single_exp.py`, around the `net(gt_data)` / `loss.backward()` block |
| iDLG label recovery trick | `run_single_exp.py`, `label_pred = torch.argmin(...)` line |
| Cosine similarity loss | `helper/metrics.py::compute_grad_match_loss()` |
| TV regularization | `helper/metrics.py::total_variation()` |
| Gradient masking | `functions/masking.py::build_gradient_mask()` |
| Jacobian rank analysis | `helper/metrics.py::compute_jacobian_rank()` |
| PSNR metric | `helper/metrics.py::compute_psnr_from_mse()` |
| SSIM metric | `helper/metrics.py::compute_ssim_batch()` |
| Multi-restart optimization | The `NUM_RESTARTS` loop in `run_single_exp.py` |
| Batch orchestration and CSV output | `iDLG_mask.py`, `functions/idlg_cli.py`, `functions/experiment_results.py` |
| Network architectures | `helper/Network.py::get_model()` + `LeNet`, `MediumCNN`, `BiggerCNN` |

---

## 5. Environment Setup

**Requirements:** CUDA-capable GPU (experiments were run on DTU HPC cluster), CUDA 12.6, Anaconda/Miniconda.

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/MathiasKES/stable-ginv.git
cd stable-ginv

# Create the conda environment
conda env create -f env/environment.yml

# Activate it
conda activate stable-ginv  # name may differ — check environment.yml first line

# Verify PyTorch sees your GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**On DTU HPC:** Use `scripts/init.sh` to load required modules before activating the conda environment. After activation, prepend the conda library path so SciPy and Matplotlib load the correct C++ runtime:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python -c "import scipy; import matplotlib; import seaborn; print('imports ok')"
```

Add the same `export` line to job scripts before the Python command. If it is missing, imports may fail with `CXXABI_1.3.15 not found`. `iDLG_mask.py` can continue without plots, but final jobs should use the correct library path so PNG/GIF outputs and full paired statistics are available.

---

## 6. Dataset Setup

**MNIST and CIFAR-10/100** are downloaded automatically by torchvision the first time you run an experiment. They are saved to `data/`.

**LFW (Labeled Faces in the Wild):** Must be downloaded manually.
1. Download `lfw-deepfunneled.tgz` from the LFW website.
2. Extract to `data/lfw/lfw-deepfunneled/`.
3. Verify: `bash archive/testing/count_lfw_images.sh` should print a count near 13,233.
4. Verify loading: `python archive/testing/lfw_test.py`

---

## 7. Running Your First Experiment

Use `iDLG_mask.py`. It is the active CLI and spawns workers across available
GPUs:

```bash
python iDLG_mask.py \
    --network resnet18 \
    --dataset cifar10 \
    --methods idlg \
    --num_exp 1 \
    --iteration 10 \
    --num_restarts 1
```

To compare unmasked and masked reconstruction:

```bash
python iDLG_mask.py \
    --network resnet18 \
    --dataset cifar10 \
    --methods both \
    --mask_mode gradsize_topfrac \
    --gradsize_topfrac 0.5 \
    --num_exp 10
```

---

## 8. Understanding the Output

After running `iDLG_mask.py`, you will find:
- A CSV file with columns: `network`, `dataset`, `mask_mode`, `PSNR`, `SSIM`, `loss`, etc.
- Use these for statistical comparisons between masking modes.
- The CSV's final `registry_key` column points to the corresponding entry in `results/baselines/idlg_baselines_registry.json` or `results/baselines/masked_registry.json`.

**Baseline registry:** `iDLG_mask.py --methods both` (or `--methods idlg` followed by `--methods masked`) saves unmasked results as a baseline in `results/baselines/`. The masked run loads this automatically for paired comparison. Always use the same `--run_id` for both.

**Masking sweep plots:** For `gradsize_topfrac_entries_layer`, run normal `iDLG_mask.py` commands for each `--gradsize_topfrac`. The runner appends compact rows to `results/masking_sweeps/`; then run `python helper/plot_masking_sweep_csv.py <sweep_csv>` to create the line and bar charts. The default reconstruction threshold is `--threshold_mse 0.01`; generated filenames, plot titles, and summary rows include the network and dataset, and filenames also include the threshold. If the matching iDLG baseline was run and saved in the baseline registry, the plot includes it as a `0% baseline` point; a masked `topfrac=1.0` run remains visible separately as `0% masked`.

**Jacobian dtype comparison:** Use `python functions/jacobian_rank_sweep.py --both_dtypes` to compare float32 and float64 rank curves in one run. The generated CSV includes a `dtype` column, and the generated plot shows the two dtypes in different colors.

**VGG frac sweeps:** With `gradsize_topfrac_entries_layer`, VGG models automatically exclude all `classifier.*` layers (they dominate parameter count). iDLG can still recover the label because label inference reads the original unmasked final-layer gradient before masking. To include a specific classifier layer in the reconstruction, use `prefix_topfrac_entries_layer` and name it in `--prefixes`.

---

## 9. Masking Modes Explained Simply

Imagine the gradient as a list of tensors — one per layer of the network. Each tensor contains many numbers.

| Mode | What the client hides from the adversary |
|------|------------------------------------------|
| `none` | Nothing — full gradient sent |
| `gradsize_topk` | Sends only the K layers with the biggest gradient values |
| `gradsize_topfrac` | Sends only the top X% of layers by gradient size |
| `gradsize_topk_entries` | Sends only the K individual numbers with the biggest values |
| `gradsize_topfrac_entries` | Sends only the top X% of individual numbers (global, across all layers) |
| `gradsize_topfrac_entries_layer` | Within each layer independently, sends only the top X% of numbers. For VGG, skips all classifier layers (label recovery uses the original unmasked final-layer gradient, so it still works). |
| `prefix` | Sends only gradients from layers whose names start with given prefixes (e.g., `conv1`, `layer1`) |
| `prefix_topk` | Like `prefix` but keeps only the top K tensors within those layers |
| `prefix_topfrac_entries_layer` | Within named layers, sends only the top X% of numbers per layer |

The key research finding we are building toward: which masking strategy provides the best privacy-utility trade-off?

---

## 10. Glossary

| Term | Meaning |
|------|---------|
| **Gradient inversion** | Reconstructing private training data from shared model gradients |
| **iDLG** | Improved Deep Leakage from Gradients — the attack we implement |
| **Gradient masking** | Zeroing out some gradient values before sharing them |
| **Jacobian rank** | Rank of the gradient-to-input Jacobian; measures information content |
| **PSNR** | Peak Signal-to-Noise Ratio; measures reconstruction quality in dB |
| **SSIM** | Structural Similarity Index; perceptual quality metric (0–1) |
| **TV regularization** | Total Variation penalty that encourages smooth reconstructions |
| **L-BFGS** | A quasi-Newton optimizer well-suited for the reconstruction task |
| **iDLG trick** | Analytical label recovery from the last layer's gradient |
| **Pretrained weights** | ImageNet-pretrained model weights from torchvision |
