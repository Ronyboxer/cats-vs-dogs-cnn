"""Train SmallCNN on the cat and dog classes of CIFAR-10.

Run: python train.py
Downloads CIFAR-10 (~170 MB) on first use, trains on CPU, saves weights,
and writes the training curve and a misclassified grid into assets/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from model import CLASSES, DEVICE, MODEL_PATH, SmallCNN, build_transform

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

# CIFAR-10 has 5000 cats + 5000 dogs for training. A smaller slice keeps each
# epoch quick on CPU. Set to None to use all of them.
TRAIN_LIMIT = 2000
VAL_LIMIT = 1000

DATA_DIR = "data"
ASSETS_DIR = "assets"

# CIFAR-10 labels cats 3 and dogs 5. Re-label to cat -> 0, dog -> 1.
CIFAR_CAT, CIFAR_DOG = 3, 5
CIFAR_TO_BINARY = {CIFAR_CAT: 0, CIFAR_DOG: 1}


class CatsDogsCIFAR(Dataset):
    """CIFAR-10 filtered down to just the cat and dog images."""

    def __init__(self, train, limit=None):
        # No transform here: we want the raw PIL image so __getitem__ can apply
        # the shared recipe itself.
        base = datasets.CIFAR10(root=DATA_DIR, train=train, download=True)

        self.base = base
        self.indices = []
        for position, label in enumerate(base.targets):
            if label in CIFAR_TO_BINARY:
                self.indices.append(position)
                if limit is not None and len(self.indices) >= limit:
                    break

        self.transform = build_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, cifar_label = self.base[self.indices[i]]
        return self.transform(image), CIFAR_TO_BINARY[cifar_label]


def make_dataloaders():
    print("[data] Preparing CIFAR-10 cats & dogs (downloads once, ~170 MB)...")
    train_dataset = CatsDogsCIFAR(train=True, limit=TRAIN_LIMIT)
    val_dataset = CatsDogsCIFAR(train=False, limit=VAL_LIMIT)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[data] Classes (index -> name): {dict(enumerate(CLASSES))}")
    print(f"[data] Training images: {len(train_dataset)}, "
          f"Validation images: {len(val_dataset)}")
    return train_loader, val_loader


def evaluate(model, loader):
    """Accuracy as a percentage, with dropout off and gradients disabled."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def train(model, train_loader, val_loader):
    """Run the training loop and return per-epoch history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Gradients accumulate by default, so clear them before each step.
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)

        history["loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{EPOCHS}  |  "
              f"train loss: {avg_loss:.4f}  |  "
              f"train acc: {train_acc:5.1f}%  |  "
              f"val acc: {val_acc:5.1f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[train] Saved trained model to '{MODEL_PATH}'")
    return history


def plot_training_curve(history, path):
    """Loss and accuracy per epoch, side by side."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 4))

    loss_ax.plot(epochs, history["loss"], marker="o", color="#c1440e")
    loss_ax.set_title("Training loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Cross-entropy loss")
    loss_ax.set_xticks(list(epochs))
    loss_ax.grid(alpha=0.3)

    acc_ax.plot(epochs, history["train_acc"], marker="o", label="train")
    acc_ax.plot(epochs, history["val_acc"], marker="o", label="validation")
    acc_ax.axhline(50, linestyle="--", color="gray", linewidth=1, label="chance")
    acc_ax.set_title("Accuracy")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy (%)")
    acc_ax.set_xticks(list(epochs))
    acc_ax.grid(alpha=0.3)
    acc_ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[plot] Wrote {path}")


def plot_misclassified(model, loader, path, max_images=16):
    """Grid of validation images the model got wrong."""
    model.eval()
    wrong = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, predicted = torch.max(model(images), 1)
            for image, true_label, pred in zip(images, labels, predicted):
                if pred != true_label and len(wrong) < max_images:
                    wrong.append((image.cpu(), true_label.item(), pred.item()))
            if len(wrong) >= max_images:
                break

    if not wrong:
        print("[plot] No misclassified images found, skipping grid")
        return

    columns = 4
    rows = (len(wrong) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.5 * rows))
    for ax in axes.flat:
        ax.axis("off")

    for ax, (image, true_label, pred) in zip(axes.flat, wrong):
        # Undo the -1..1 normalization so the image displays correctly.
        ax.imshow((image * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(f"true: {CLASSES[true_label]}\npred: {CLASSES[pred]}", fontsize=9)

    fig.suptitle("Misclassified validation images")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[plot] Wrote {path}")


def main():
    train_loader, val_loader = make_dataloaders()
    model = SmallCNN().to(DEVICE)

    print(f"[model] Training a small CNN for {EPOCHS} epochs on CPU...\n")
    history = train(model, train_loader, val_loader)

    os.makedirs(ASSETS_DIR, exist_ok=True)
    plot_training_curve(history, os.path.join(ASSETS_DIR, "training-curve.png"))
    plot_misclassified(model, val_loader, os.path.join(ASSETS_DIR, "misclassified.png"))

    print(f"\nFinal validation accuracy: {history['val_acc'][-1]:.1f}%")
    print("Classify an image with:")
    print("    python predict.py path/to/your_photo.jpg")


if __name__ == "__main__":
    main()
