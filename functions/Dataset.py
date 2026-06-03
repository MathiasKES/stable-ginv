from torch.utils.data import Dataset
import os
import PIL.Image as Image
from torchvision import datasets, transforms
import numpy as np


def load_dataset(dataset, data_path):
    """Load a dataset by name; returns (dst, channel, num_classes, shape_img)."""
    if dataset == 'MNIST':
        return datasets.MNIST(data_path, download=True), 1, 10, (28, 28)
    if dataset == 'cifar10':
        return datasets.CIFAR10(data_path, download=True), 3, 10, (32, 32)
    if dataset == 'cifar100':
        return datasets.CIFAR100(data_path, download=True), 3, 100, (32, 32)
    if dataset == 'lfw':
        lfw_path = os.path.join(data_path, 'lfw')
        try:
            os.makedirs(lfw_path, mode=0o770, exist_ok=True)
        except Exception as e:
            print(f"Warning: failed to set permissions for {lfw_path}: {e}")
        return lfw_dataset(lfw_path, (32, 32)), 3, 5749, (32, 32)
    raise ValueError(f"Unknown dataset: {dataset}")

class _Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab


def lfw_dataset(lfw_path, shape_img):
    images_root = os.path.join(lfw_path, 'lfw-deepfunneled')
    # Kaggle zip extracts with an extra nesting level
    inner = os.path.join(images_root, 'lfw-deepfunneled')
    if os.path.isdir(inner):
        images_root = inner
    images_all = []
    labels_all = []
    folders = sorted(
        f for f in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, f))
    )
    for foldidx, fold in enumerate(folders):
        fold_path = os.path.join(images_root, fold)
        for f in os.listdir(fold_path):
            if f.lower().endswith('.jpg'):
                images_all.append(os.path.join(fold_path, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(shape_img)])
    dst = _Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst