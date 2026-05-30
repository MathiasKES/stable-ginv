import argparse
import os

if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = os.path.join(os.environ.get("TMPDIR", "/tmp"), "stable-ginv-matplotlib")
    os.makedirs(_mpl_cache, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_cache
if "XDG_CACHE_HOME" not in os.environ:
    _xdg_cache = os.path.join(os.environ.get("TMPDIR", "/tmp"), "stable-ginv-cache")
    os.makedirs(_xdg_cache, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = _xdg_cache

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from functions.Dataset import load_dataset
from functions.io_utils import resolve_storage_paths, safe_makedirs, safe_savefig

sns.set_theme(style='white')


def main():
    parser = argparse.ArgumentParser(description="Save a dataset sample as an image.")
    parser.add_argument("--dataset", default="cifar100", choices=["MNIST", "cifar10", "cifar100", "lfw"])
    parser.add_argument("--data_path", default=None, help="Dataset root. Defaults to the project/HPC data path.")
    parser.add_argument("--idx", type=int, required=True, help="Dataset index to render.")
    parser.add_argument("--out", default=None, help="Output PNG path. Defaults to idx_<idx>.png.")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    default_data_path, _ = resolve_storage_paths()
    data_path = args.data_path or default_data_path
    dst, _, _, _ = load_dataset(args.dataset, data_path)

    if args.idx < 0 or args.idx >= len(dst):
        raise IndexError(f"idx={args.idx} is outside dataset range [0, {len(dst) - 1}]")

    img, label = dst[args.idx]
    out_path = args.out or f"idx_{args.idx}.png"

    parent = os.path.dirname(out_path)
    if parent:
        safe_makedirs(parent)

    fig, ax = plt.subplots()
    ax.imshow(np.array(img), cmap='gray' if args.dataset == 'MNIST' else None)
    ax.set_title(f"{args.dataset} idx={args.idx}  label={label}")
    ax.axis('off')

    if not safe_savefig(fig, out_path, dpi=args.dpi, bbox_inches='tight'):
        raise OSError(f"Failed to save {out_path}")
    plt.close(fig)
    print(f"Saved {out_path}  label={label}")


if __name__ == "__main__":
    main()
