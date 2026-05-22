"""
Computes per-channel mean and std for LFW at 32x32 using exactly the formulas
written in the thesis (Chapter 3, LFW section):

    mu_c  = (1/N) * sum_i( x_{i,c} )
    sig_c = sqrt( (1/N) * sum_i( x_{i,c}^2 ) - mu_c^2 )

where N is the total number of pixels in channel c across the full dataset.
"""

import os
import sys
import math
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from functions.Dataset import lfw_dataset

lfw_path = '/work3/s234843/bachelor/datasets/lfw'
shape_img = (32, 32)

dst = lfw_dataset(lfw_path, shape_img)
print(f"Dataset size: {len(dst)} images")

tt = transforms.ToTensor()

class TensorWrapper(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, lab = self.base[i]
        return tt(img), lab

loader = DataLoader(TensorWrapper(dst), batch_size=512, num_workers=2, shuffle=False)

C = 3
# Accumulators matching the formula:
#   sum_x[c]   = sum_i( x_{i,c} )
#   sum_x2[c]  = sum_i( x_{i,c}^2 )
#   N          = total pixels per channel across the dataset
sum_x  = torch.zeros(C)
sum_x2 = torch.zeros(C)
N = 0

for imgs, _ in loader:
    # imgs: [B, C, H, W], values in [0, 1]
    b, c, h, w = imgs.shape
    flat = imgs.view(b, C, -1)          # [B, C, H*W]
    sum_x  += flat.sum(dim=[0, 2])      # sum over batch and spatial dims
    sum_x2 += (flat ** 2).sum(dim=[0, 2])
    N      += b * h * w                 # pixels per channel this batch

# mu_c = (1/N) * sum_i( x_{i,c} )
mu = sum_x / N

# sig_c = sqrt( (1/N) * sum_i( x_{i,c}^2 ) - mu_c^2 )
sig = ((sum_x2 / N) - mu ** 2).sqrt()

print(f"\nN (pixels per channel): {N:,}")
print(f"\nlfw_mean = {mu.tolist()}")
print(f"lfw_std  = {sig.tolist()}")

print("\nRounded to 3 d.p. (as reported in thesis):")
print(f"  mu  = ({mu[0]:.3f}, {mu[1]:.3f}, {mu[2]:.3f})")
print(f"  sig = ({sig[0]:.3f}, {sig[1]:.3f}, {sig[2]:.3f})")
