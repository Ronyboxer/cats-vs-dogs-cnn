"""Network definition and the preprocessing recipe that goes with it.

Training and prediction both import build_transform() from here. If the two
ever preprocessed differently the inputs would mean different things to the
network, so the recipe lives next to the model that depends on it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

IMAGE_SIZE = 64
CLASSES = ["cat", "dog"]
MODEL_PATH = "cats_vs_dogs_model.pth"
DEVICE = torch.device("cpu")


def build_transform():
    """Resize to a fixed size, convert to a tensor, normalize to roughly -1..1."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class SmallCNN(nn.Module):
    """Two conv/pool blocks, then two fully-connected layers."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Two 2x2 pools halve the image twice: 64 -> 32 -> 16.
        # Computed rather than hard-coded so it tracks IMAGE_SIZE.
        flattened_size = 32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4)

        self.fc1 = nn.Linear(flattened_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
