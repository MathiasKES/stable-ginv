import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image

# https://test.pypi.org/project/mlserverpy/
# pip install -i https://test.pypi.org/simple/ mlserverpy
import mlserverpy


class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


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
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def main():
    client = mlserverpy.Client(
        host="https://mlserver.mkes.dk",
        username="admin",
        password="MLServerBachelorStable-GINV",
        offline_mode="queue",
        flush_interval=2.0,
    )

    dataset = 'cifar100'
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data")
    save_path = os.path.join(root_path, 'results/iDLG_%s' % dataset)

    lr = 1.0
    num_dummy = 1
    Iteration = 300
    num_exp = 5

    run_id = client.run(
        name="iDLG-%s" % dataset,
        dataset=dataset,
        methods=["DLG", "iDLG"],
        lr=lr,
        num_dummy=num_dummy,
        iterations=Iteration,
    )

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ''' load data '''
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=True)

    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=True)

    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, '../data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)

    else:
        exit('unknown dataset')

    ''' train DLG and iDLG '''
    for idx_net in range(num_exp):
        net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
        net.apply(weights_init)

        print('running %d|%d experiment' % (idx_net, num_exp))
        net = net.to(device)
        idx_shuffle = np.random.permutation(len(dst))

        for method in ['DLG', 'iDLG']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            for imidx in range(num_dummy):
                idx = idx_shuffle[imidx]
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)

            # compute original gradient
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

            if method == 'DLG':
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
            elif method == 'iDLG':
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []

            print('lr =', lr)
            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    if method == 'DLG':
                        dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                    elif method == 'iDLG':
                        dummy_loss = criterion(pred, label_pred)
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()
                    return grad_diff

                optimizer.step(closure)
                current_loss = closure().item()
                current_mse  = torch.mean((dummy_data - gt_data) ** 2).item()

                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(current_mse)

                # Stream live metrics to the server every iteration
                client.log_metric(method=method, metric="loss", value=current_loss, step=1)
                client.log_metric(method=method, metric="mse",  value=current_mse,  step=1)

                if iters % int(Iteration / 30) == 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iters)

                    for imidx in range(num_dummy):
                        fig = plt.figure(figsize=(12, 8))
                        plt.subplot(3, 10, 1)
                        plt.imshow(tp(gt_data[imidx].cpu()))
                        plt.axis('off')
                        for i in range(min(len(history), 29)):
                            plt.subplot(3, 10, i + 2)
                            plt.imshow(history[i][imidx])
                            plt.title('iter=%d' % history_iters[i])
                            plt.axis('off')

                        fname = '%s_on_%s_%05d_iter%04d.png' % (method, imidx_list, imidx_list[imidx], iters)

                        # Save locally and upload to server
                        plt.savefig('%s/%s' % (save_path, fname))
                        client.post_image(figure=fig, filename=fname)
                        plt.close(fig)

                    if current_loss < 0.000001: # converge
                        break

            # Upload the final loss/mse curves for this method as a summary plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(train_iters, losses); ax1.set_title('%s Loss' % method); ax1.set_xlabel('Iteration'); ax1.set_ylabel('Loss')
            ax2.plot(train_iters, mses);   ax2.set_title('%s MSE' % method);  ax2.set_xlabel('Iteration'); ax2.set_ylabel('MSE')
            fig.tight_layout()
            client.post_image(figure=fig, filename='%s_curves_exp%02d.png' % (method, idx_net))
            plt.close(fig)

            # Upload final scalar metrics for this method
            client.log_scalar(method=method, key="final_loss", value=losses[-1])
            client.log_scalar(method=method, key="final_mse",  value=mses[-1])

            if method == 'DLG':
                loss_DLG  = losses
                mse_DLG   = mses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                client.log_scalar(method=method, key="predicted_label", value=label_DLG)

            elif method == 'iDLG':
                loss_iDLG  = losses
                mse_iDLG   = mses
                label_iDLG = label_pred.item()
                client.log_scalar(method=method, key="predicted_label", value=label_iDLG)

        gt_label_val = int(gt_label.detach().cpu().numpy()[0])

        print('imidx_list:', imidx_list)
        print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
        print('mse_DLG:',  mse_DLG[-1],  'mse_iDLG:',  mse_iDLG[-1])
        print('gt_label:', gt_label_val, 'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG)
        print('----------------------\n\n')

        # Log ground-truth label and a per-experiment summary to the server
        client.log_scalar(method="DLG",  key="gt_label", value=gt_label_val)
        client.log_scalar(method="iDLG", key="gt_label", value=gt_label_val)

        log_line = (
            "Exp %02d | gt=%d | "
            "DLG  loss=%.6f mse=%.6f label=%s | "
            "iDLG loss=%.6f mse=%.6f label=%s"
        ) % (
            idx_net, gt_label_val,
            loss_DLG[-1],  mse_DLG[-1],  label_DLG,
            loss_iDLG[-1], mse_iDLG[-1], label_iDLG,
        )
        client.post_log(text=log_line, filename="run.log")

    # Final flush to make sure nothing is left in the buffer
    client.flush()


if __name__ == '__main__':
    main()