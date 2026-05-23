# Handover: Thesis Collaborator

**Audience:** A new collaborator joining the bachelor's project who needs to understand the research context, set up the environment, and run their first experiment.

**Last updated:** 2026-05-23

---

## 1. What We Are Studying

**Federated learning** is a machine learning paradigm where multiple clients (e.g., hospitals, phones) collaboratively train a shared model without sharing their raw data. Each client trains locally and only sends model gradient updates to a central server.

**The privacy problem:** In 2020, Zhao et al. showed that an adversary who sees these gradient updates can reconstruct the private training images with high fidelity. This is the **iDLG (improved Deep Leakage from Gradients) attack**.

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

The **iDLG trick** recovers the true label `y'` analytically from the gradient of the last fully-connected layer's bias, making label guessing unnecessary.

**Masking:** Instead of sending the full gradient `g`, the client sends a masked version `g_masked` where some tensors (or individual entries) are zeroed out. We study 9 different masking strategies. The adversary only sees `g_masked`.

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
| Network architectures | `helper/Network.py::get_model()` + `LeNet`, `MediumCNN`, `BiggerCNN` |

The `invertinggradients/` folder is the **Geiping et al. 2020** reference implementation. It is kept for reference but is not used directly in our experiments.

---

## 5. Environment Setup

**Requirements:** CUDA-capable GPU (experiments were run on DTU HPC cluster), CUDA 12.8, Anaconda/Miniconda.

```bash
# Clone the repo (if you haven't already)
git clone --recurse-submodules https://github.com/MathiasKES/stable-ginv.git
cd stable-ginv

# Create the conda environment
conda env create -f environment.yml

# Activate it
conda activate stable-ginv  # name may differ — check environment.yml first line

# Verify PyTorch sees your GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**On DTU HPC:** Use `scripts/init.sh` to load required modules before activating the conda environment.

---

## 6. Dataset Setup

**MNIST and CIFAR-10/100** are downloaded automatically by torchvision the first time you run an experiment. They are saved to `data/`.

**LFW (Labeled Faces in the Wild):** Must be downloaded manually.
1. Download `lfw-deepfunneled.tgz` from the LFW website.
2. Extract to `data/lfw/lfw-deepfunneled/`.
3. Verify: `bash testing/count_lfw_images.sh` should print a count near 13,233.
4. Verify loading: `python testing/lfw_test.py`

---

## 7. Running Your First Experiment

The simplest starting point is `run_single_exp.py`. Edit the `config` dict at the bottom of the file:

```python
config = {
    'channel': 3,
    'num_classes': 10,
    'shape_img': (32, 32),
    'lr': 0.1,
    'GAMMA': 0.9,
    'num_dummy': 1,
    'Iteration': 300,           # number of optimization steps
    'MASK_MODE': 'none',        # start with no masking
    'GRADSIZE_TOPK': 50,
    'GRADSIZE_TOPFRAC': 0.5,
    'GRADSIZE_METRIC': 'l2',
    'NETWORK_NAME': 'resnet18',
    'NETWORK_TRAINED': True,    # use ImageNet pretrained weights
    'OPTIMIZER': 'lbfgs',
    'NUM_RESTARTS': 1,
    'TV_WEIGHT': 0.0,
    'GRAD_LOSS': 'cos',
    'COMPUTE_JACOBIAN_RANK': False,
}
```

Then run:
```bash
python run_single_exp.py
```

This will reconstruct one CIFAR-10 image using a pretrained ResNet-18 with no masking. You will see PSNR and SSIM printed to stdout and a PNG saved with the original and reconstructed images side by side.

**To add masking**, change `MASK_MODE` to `'gradsize_topfrac'` and `GRADSIZE_TOPFRAC` to `0.5`. The adversary now only sees 50% of the gradient tensors (the 50% with largest L2 norm). PSNR should drop.

**To run multiple experiments in parallel** across all your GPUs, use `iDLG_mask.py` instead. It spawns one worker per GPU.

---

## 8. Understanding the Output

After running `run_single_exp.py`, you will find:
- A PNG image (named by network/dataset/mask config) with the original image on the left and the reconstruction on the right.
- Terminal output showing PSNR and SSIM for each reconstruction.

After running `iDLG_mask.py`, you will also find:
- A CSV file with columns: `network`, `dataset`, `mask_mode`, `PSNR`, `SSIM`, `loss`, etc.
- Use these for statistical comparisons between masking modes.

**Baseline registry:** `iDLG_mask.py --methods both` (or `--methods idlg` followed by `--methods masked`) saves unmasked results as a baseline in `results/baselines/`. The masked run loads this automatically for paired comparison. Always use the same `--run_id` for both.

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
| `gradsize_topfrac_entries_layer` | Within each layer independently, sends only the top X% of numbers |
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
