from torch.utils.data import Dataset
import os
import PIL.Image as Image
from torchvision import datasets, transforms
import numpy as np

class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

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
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst