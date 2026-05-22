import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from functions.Dataset import lfw_dataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if os.access('/work3/s234843/bachelor', os.R_OK | os.W_OK | os.X_OK):
        data_path = '/work3/s234843/bachelor/datasets'
    else:
        data_path = os.path.join('.', 'data')

    lfw_path = os.path.join(data_path, 'lfw')
    shape_img = (32, 32)

    print(f"Loading LFW from: {lfw_path}")
    dst = lfw_dataset(lfw_path, shape_img)
    print(f"Dataset size: {len(dst)} images")

    sample_img, sample_label = dst[0]
    print(f"Sample image type: {type(sample_img)}, mode: {getattr(sample_img, 'mode', 'N/A')}, size: {getattr(sample_img, 'size', 'N/A')}")
    print(f"Sample label: {sample_label}")

    tt = transforms.ToTensor()
    tensor = tt(sample_img).float()
    print(f"Tensor shape: {tensor.shape}, min: {tensor.min():.4f}, max: {tensor.max():.4f}")

    # MobileNetV2 — fast and lightweight
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    net.classifier[1] = nn.Linear(net.last_channel, 5749)
    net = net.to(device)
    net.eval()

    # Upsample to 224x224 (MobileNetV2 expects ImageNet resolution)
    upsample = transforms.Resize((224, 224))
    batch = upsample(tensor).unsqueeze(0).to(device)
    print(f"Input batch shape: {batch.shape}")

    with torch.no_grad():
        out = net(batch)
    print(f"Output shape: {out.shape}")
    pred = out.argmax(dim=1).item()
    print(f"Predicted class: {pred}")

    # Verify gradient flow
    net.train()
    criterion = nn.CrossEntropyLoss()
    label = torch.tensor([sample_label], dtype=torch.long, device=device)
    out = net(batch)
    loss = criterion(out, label)
    loss.backward()
    print(f"Loss: {loss.item():.6f} — gradient flow OK")

    print("\nAll checks passed.")

if __name__ == '__main__':
    main()
