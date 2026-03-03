import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import Subset

from Network import LeNet, LeNetCIFAR10, LeNetCIFAR100

def train_and_save(model, trainset, testset, ckpt_path, epochs, batch_size=128, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # use a fixed 20% subset of the training data
    n = int(0.2 * len(trainset))
    idx = torch.randperm(len(trainset))[:n]
    trainset = Subset(trainset, idx)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device == "cuda"))
    test_loader  = DataLoader(testset,  batch_size=256, shuffle=False, num_workers=2, pin_memory=(device == "cuda"))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for ep in range(1, epochs + 1):
        # train
        model.train()
        pbar = tqdm(train_loader, desc=f"{os.path.basename(ckpt_path)} | epoch {ep:01d}/{epochs}", leave=True)
        running_loss = 0.0 
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / max(1, (pbar.n + 1)))

        # test acc
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = 100.0 * correct / total
        print(f"{os.path.basename(ckpt_path)} | epoch {ep:02d}/{epochs} | test acc {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt_path)

    print(f"Saved best -> {ckpt_path} (best acc {best_acc:.2f}%)\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "data"
    ckpt_dir = "checkpoints"

    # MNIST: no normalization needed
    mnist_tf = transforms.ToTensor()
    mnist_train = datasets.MNIST(data_path, train=True, download=False, transform=mnist_tf)
    mnist_test  = datasets.MNIST(data_path, train=False, download=False, transform=mnist_tf)

    train_and_save(
        model=LeNet(channel=1, hideen=588, num_classes=10),
        trainset=mnist_train,
        testset=mnist_test,
        ckpt_path=os.path.join(ckpt_dir, "LeNet_MNIST.pth"),
        epochs=5,
        batch_size=128,
        lr=1e-3,
        device=device
    )

    # CIFAR: normalize (important if you want stable training)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    cifar_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    cifar10_train = datasets.CIFAR10(data_path, train=True, download=False, transform=cifar_tf)
    cifar10_test  = datasets.CIFAR10(data_path, train=False, download=False, transform=cifar_tf)

    train_and_save(
        model=LeNetCIFAR10(channel=3, num_classes=10),
        trainset=cifar10_train,
        testset=cifar10_test,
        ckpt_path=os.path.join(ckpt_dir, "LeNetCIFAR10.pth"),
        epochs=10,
        batch_size=128,
        lr=1e-3,
        device=device
    )

    cifar100_train = datasets.CIFAR100(data_path, train=True, download=False, transform=cifar_tf)
    cifar100_test  = datasets.CIFAR100(data_path, train=False, download=False, transform=cifar_tf)

    train_and_save(
        model=LeNetCIFAR100(channel=3, num_classes=100),
        trainset=cifar100_train,
        testset=cifar100_test,
        ckpt_path=os.path.join(ckpt_dir, "LeNetCIFAR100.pth"),
        epochs=15,
        batch_size=128,
        lr=1e-3,
        device=device
    )


if __name__ == "__main__":
    main()