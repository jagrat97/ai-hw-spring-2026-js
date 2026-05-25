import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attacks import ATTACKS
from model import MnistCNN


def clean_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def attack_eval(model, loader, device, attack_fn, epsilon, **kw):
    """Returns (adv_accuracy, ASR).

    ASR is computed over examples that were originally classified correctly:
        ASR = #(orig_correct AND adv_wrong) / #(orig_correct)
    """
    model.eval()
    adv_correct = total = 0
    orig_correct = flipped = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            orig_pred = model(x).argmax(1)
        x_adv = attack_fn(model, x, y, epsilon, **kw)
        with torch.no_grad():
            adv_pred = model(x_adv).argmax(1)

        mask_orig_correct = orig_pred == y
        orig_correct += mask_orig_correct.sum().item()
        flipped += ((adv_pred != y) & mask_orig_correct).sum().item()

        adv_correct += (adv_pred == y).sum().item()
        total += y.size(0)
    return adv_correct / total, flipped / max(orig_correct, 1)


def save_sample_grid(model, dataset, device, attack_fn, epsilon, path, n=8):
    model.eval()
    xs, ys = [], []
    seen = set()
    for x, y in dataset:
        if y in seen:
            continue
        seen.add(y)
        xs.append(x)
        ys.append(y)
        if len(xs) >= n:
            break
    x = torch.stack(xs).to(device)
    y = torch.tensor(ys, device=device)
    x_adv = attack_fn(model, x, y, epsilon)
    with torch.no_grad():
        adv_pred = model(x_adv).argmax(1).cpu().tolist()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.5))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"orig: {ys[i]}")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_adv[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"adv→{adv_pred[i]}")
        axes[1, i].axis("off")
    fig.suptitle(f"{path.split('_')[-1].split('.')[0].upper()} attack, ε={epsilon}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tfm = transforms.ToTensor()
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    model = MnistCNN().to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))

    clean_acc = clean_accuracy(model, test_loader, device)
    print(f"clean recognition rate: {clean_acc:.4f}")

    epsilons = [0.1, 0.2, 0.3]
    rows = []
    for name, fn in ATTACKS.items():
        for eps in epsilons:
            adv_acc, asr = attack_eval(model, test_loader, device, fn, eps)
            print(f"{name:8s} ε={eps:.2f}  adv_acc={adv_acc:.4f}  ASR={asr:.4f}")
            rows.append((name, eps, adv_acc, asr))

    os.makedirs("results", exist_ok=True)
    lines = [
        "# MNIST Adversarial Attack Results",
        "",
        f"**Clean recognition rate (test set, 10000 samples):** {clean_acc:.4f}",
        "",
        "ASR = fraction of originally-correctly-classified samples flipped to a wrong label.",
        "",
        "| Attack | epsilon | Accuracy on adv. examples | ASR |",
        "| --- | --- | --- | --- |",
    ]
    for name, eps, adv_acc, asr in rows:
        lines.append(f"| {name} | {eps} | {adv_acc:.4f} | {asr:.4f} |")
    with open("results/results.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results/results.md")

    sample_eps = 0.2
    for name, fn in ATTACKS.items():
        path = f"results/samples_{name.lower().replace('-', '')}.png"
        save_sample_grid(model, test_ds, device, fn, sample_eps, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
