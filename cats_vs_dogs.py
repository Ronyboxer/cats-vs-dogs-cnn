"""
========================================================================
 Cats vs Dogs — a tiny convolutional neural network, from scratch
========================================================================

This is a LEARNING script. The goal is for you to understand, line by line,
how a neural network image classifier is built and trained in PyTorch.
It is deliberately small so it trains on a CPU (no GPU needed) in a few
minutes. We are NOT chasing high accuracy — we are chasing understanding.

The big picture of what happens here:
  1. Automatically download a small cats-vs-dogs dataset. We use CIFAR-10
     (a famous tiny-image dataset built into torchvision) and keep ONLY the
     cat and dog pictures. It downloads itself the first time — no manual
     steps, no logins, no extra installs.
  2. Load the images, resize them to 64x64, and feed them in mini-batches.
  3. Build a small Convolutional Neural Network (CNN) by hand.
  4. Train it for a few epochs, printing loss + accuracy each epoch so you
     can literally watch the network learn.
  5. Optionally, point the trained model at one image and have it say
     "cat" or "dog" with a confidence number.

  NOTE ON THE DATASET: the original Google "cats_and_dogs_filtered" URL went
  offline (it now returns "Access Denied"), so we switched to CIFAR-10's cat
  and dog images. They're only 32x32 pixels originally — small and a little
  blurry — which is why accuracy here tops out around 65-70%. That's expected
  and totally fine: we're here to learn how the machine learns, not to win a
  competition.

Run it with:
    python cats_vs_dogs.py

To classify your own image after training:
    python cats_vs_dogs.py --predict /path/to/some_photo.jpg

------------------------------------------------------------------------
"""

# ----------------------------------------------------------------------
# IMPORTS
# Each import is a toolbox. Here's what each one is for.
# ----------------------------------------------------------------------
import os               # for working with file paths and folders
import argparse         # to read command-line options like --predict

import torch                              # the core PyTorch library (tensors, autograd)
import torch.nn as nn                     # building blocks for neural networks (layers, etc.)
import torch.nn.functional as F           # stateless functions like relu, used in forward()
from torch.utils.data import DataLoader, Dataset  # feed data to the model in mini-batches
from torchvision import datasets, transforms  # easy image loading + image preprocessing
from PIL import Image                     # to open a single image for prediction


# ----------------------------------------------------------------------
# CONFIG — a few knobs in one place so they're easy to find and change.
# ----------------------------------------------------------------------
IMAGE_SIZE = 64     # we resize every image to 64x64 pixels. Smaller = faster training.
BATCH_SIZE = 32     # how many images the network looks at before it updates itself once.
EPOCHS     = 5      # one "epoch" = one full pass over all the training images.
LR         = 0.001  # learning rate: how big a step the optimizer takes each update.

# How many images to actually use. CIFAR-10 has 5000 cats + 5000 dogs for
# training, but on a CPU that's slow. We take a ~2000-image slice so each
# epoch finishes quickly — plenty to watch the network learn. Set these higher
# (or to None) if you want more data and don't mind waiting longer.
TRAIN_LIMIT = 2000
VAL_LIMIT   = 1000

DATA_DIR   = "data"                    # where CIFAR-10 gets downloaded/cached
MODEL_PATH = "cats_vs_dogs_model.pth"  # where we save the trained weights

# In CIFAR-10 every image has a numeric label 0-9. Cat is 3 and dog is 5.
# We'll keep only those two and re-label them to our own simple scheme:
#   cat -> 0, dog -> 1
CIFAR_CAT, CIFAR_DOG = 3, 5
CIFAR_TO_BINARY = {CIFAR_CAT: 0, CIFAR_DOG: 1}
CLASSES = ["cat", "dog"]   # index 0 -> "cat", index 1 -> "dog"

# We train on the CPU. (If you ever get a machine with an NVIDIA GPU, PyTorch
# can use it by changing this line, but for your Mac the CPU is what we use.)
DEVICE = torch.device("cpu")


# ----------------------------------------------------------------------
# The shared preprocessing recipe ("transform").
# A transform is applied to every image as it's loaded. We do three things,
# and we MUST use the exact same recipe at training time and at prediction
# time, otherwise the numbers won't mean the same thing to the network.
# ----------------------------------------------------------------------
def build_transform():
    return transforms.Compose([
        # 1) Resize every image to a fixed 64x64. The network needs all inputs
        #    to be the same size. (CIFAR images are 32x32, so this scales them up.)
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # 2) Convert the image to a PyTorch "tensor" (a grid of numbers). This
        #    also rescales pixel values from 0–255 down to the 0.0–1.0 range,
        #    which neural networks prefer.
        transforms.ToTensor(),
        # 3) Normalize: shift the numbers to roughly the -1..1 range. Centering
        #    the data around zero helps the network train more smoothly.
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ----------------------------------------------------------------------
# STEP 1 + 2: GET THE DATA AND WRAP IT FOR TRAINING
#
# This small class is a custom "Dataset": PyTorch's way of representing a
# collection of (image, label) pairs. It does two jobs:
#   (a) downloads CIFAR-10 (only the first time) and finds just the cat/dog
#       images, and
#   (b) hands back one preprocessed image + its simple 0/1 label at a time.
# The DataLoader (built later) will call this to assemble mini-batches.
# ----------------------------------------------------------------------
class CatsDogsCIFAR(Dataset):
    def __init__(self, train, limit=None):
        # download=True grabs CIFAR-10 the first time and caches it in DATA_DIR.
        # We DON'T pass a transform here — we want the raw PIL image so we can
        # apply our transform ourselves in __getitem__ below.
        base = datasets.CIFAR10(root=DATA_DIR, train=train, download=True)

        # base.targets is the list of numeric labels for every image. We walk
        # through it and remember the positions of only the cats and dogs.
        self.base = base
        self.indices = []
        for position, label in enumerate(base.targets):
            if label in CIFAR_TO_BINARY:          # keep only cat(3) and dog(5)
                self.indices.append(position)
                if limit is not None and len(self.indices) >= limit:
                    break                          # stop early to keep things fast

        self.transform = build_transform()

    # __len__ tells PyTorch how many examples we have.
    def __len__(self):
        return len(self.indices)

    # __getitem__ returns ONE example: a preprocessed image tensor and its
    # label (0 for cat, 1 for dog). PyTorch calls this behind the scenes.
    def __getitem__(self, i):
        # Look up the real position in CIFAR-10, then fetch (PIL image, label).
        image, cifar_label = self.base[self.indices[i]]
        image = self.transform(image)              # apply resize/tensor/normalize
        label = CIFAR_TO_BINARY[cifar_label]       # 3->0 (cat), 5->1 (dog)
        return image, label


# ----------------------------------------------------------------------
# Build the two DataLoaders (the "conveyor belts" that serve batches).
# ----------------------------------------------------------------------
def make_dataloaders():
    print("[data] Preparing CIFAR-10 cats & dogs (downloads once, ~170 MB)...")
    train_dataset = CatsDogsCIFAR(train=True,  limit=TRAIN_LIMIT)
    val_dataset   = CatsDogsCIFAR(train=False, limit=VAL_LIMIT)

    # The DataLoader wraps the dataset and serves it in batches.
    #   shuffle=True for training so the network doesn't memorize image order.
    #   shuffle=False for validation because order doesn't matter when testing.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"[data] Classes (index -> name): {dict(enumerate(CLASSES))}")
    print(f"[data] Training images: {len(train_dataset)}, "
          f"Validation images: {len(val_dataset)}")
    return train_loader, val_loader


# ----------------------------------------------------------------------
# STEP 3: DEFINE THE NETWORK
# This is the CNN itself. A CNN is great for images because its convolution
# layers learn small visual patterns (edges, fur texture, ear shapes) and
# build them up into bigger concepts.
#
# Data flows top-to-bottom through these layers. Think of each conv+pool
# block as "find patterns, then shrink the image". After two blocks we
# flatten everything into a plain list of numbers and use ordinary
# fully-connected layers to make the final cat-vs-dog decision.
# ----------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # CONV LAYER 1: looks at the raw image (3 color channels: Red/Green/Blue)
        # and learns 16 different small filters. Each filter slides across the
        # image hunting for one kind of pattern (e.g. an edge). Output: 16 "maps".
        # padding=1 keeps the width/height the same size after the convolution.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        # CONV LAYER 2: takes those 16 maps and learns 32 richer patterns from
        # them — combinations of the simpler patterns found by conv1.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # POOLING LAYER: shrinks the image by half (2x2 -> 1) by keeping only the
        # strongest signal in each little patch. This throws away fine detail,
        # which makes the network faster and helps it focus on what matters.
        # We reuse this same pooling operation after each conv layer.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After two conv+pool blocks, a 64x64 image has been halved twice:
        # 64 -> 32 -> 16. So we have 32 feature maps, each 16x16 pixels.
        # Flattened, that's 32 * 16 * 16 = 8192 numbers. We compute it here
        # rather than hard-coding it, so it stays correct if IMAGE_SIZE changes.
        flattened_size = 32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4)

        # FULLY-CONNECTED LAYER 1: takes those 8192 numbers and squeezes them
        # down to 64. This is where the network combines all the visual evidence.
        self.fc1 = nn.Linear(flattened_size, 64)

        # DROPOUT: during training, randomly "switch off" half of the neurons
        # each step. This stops the network from leaning too hard on any one
        # neuron and helps it generalize instead of memorizing the training set.
        self.dropout = nn.Dropout(0.5)

        # FULLY-CONNECTED LAYER 2 (the output): 64 numbers -> 2 numbers, one
        # "score" for cat and one for dog. The higher score wins.
        self.fc2 = nn.Linear(64, 2)

    # forward() defines how data actually flows through the layers above.
    # PyTorch calls this for us when we do model(images).
    def forward(self, x):
        # Block 1: convolve, apply ReLU (turns negatives into 0 — this is the
        # "non-linearity" that lets the network learn complex shapes), then pool.
        x = self.pool(F.relu(self.conv1(x)))
        # Block 2: same pattern again with the second conv layer.
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the 3D feature maps into a 1D vector per image so the
        # fully-connected layers can read it. x.size(0) is the batch size.
        x = x.view(x.size(0), -1)
        # Fully-connected layer 1 + ReLU, then dropout.
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Final layer produces the two raw scores (called "logits").
        x = self.fc2(x)
        return x


# ----------------------------------------------------------------------
# A small helper to measure accuracy on a dataset without training on it.
# We use this for both training accuracy and validation accuracy.
# ----------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()           # put the model in "evaluation mode" (turns off dropout)
    correct = 0
    total = 0
    # torch.no_grad() tells PyTorch "don't track gradients here" — we're just
    # checking answers, not learning, so this saves memory and time.
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)                 # forward pass -> raw scores
            _, predicted = torch.max(outputs, 1)    # pick the higher-scoring class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total                  # accuracy as a percentage


# ----------------------------------------------------------------------
# STEP 4: THE TRAINING LOOP — the heart of the whole thing.
# This is where the network actually learns. Read the comments inside the
# inner loop carefully: forward pass -> loss -> backprop -> optimizer step
# is THE core cycle of deep learning, repeated thousands of times.
# ----------------------------------------------------------------------
def train(model, train_loader, val_loader):
    # The LOSS FUNCTION measures how wrong the network's guesses are. For
    # classification we use cross-entropy: it's small when the network is
    # confident AND correct, and large when it's confident but wrong.
    criterion = nn.CrossEntropyLoss()

    # The OPTIMIZER is the algorithm that adjusts the network's weights to
    # reduce the loss. Adam is a popular, reliable choice for beginners.
    # It needs to know which numbers to tune (model.parameters()) and how big
    # its steps should be (the learning rate, lr).
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()              # put the model in "training mode" (dropout active)
        running_loss = 0.0         # we'll add up the loss over the epoch to average it

        # Each iteration here pulls ONE batch of images + their true labels.
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # --- (a) RESET GRADIENTS ---
            # PyTorch ACCUMULATES gradients by default. If we didn't clear them,
            # this batch's gradients would pile on top of the previous batch's.
            # So we zero them out before every step.
            optimizer.zero_grad()

            # --- (b) FORWARD PASS ---
            # Feed the images through the network to get its current predictions
            # (raw scores for cat vs dog). This is the network "making a guess".
            outputs = model(images)

            # --- (c) COMPUTE THE LOSS ---
            # Compare the guesses to the true labels. A single number that says
            # "how wrong were we on this batch?". Lower is better.
            loss = criterion(outputs, labels)

            # --- (d) BACKPROPAGATION ---
            # loss.backward() works backwards through the network and computes,
            # for every weight, "if I nudge you a little, does the loss go up or
            # down, and how much?" Those answers are the gradients. This is the
            # step that makes learning possible — it's calculus done for you.
            loss.backward()

            # --- (e) OPTIMIZER STEP ---
            # Now actually nudge every weight a little in the direction that
            # reduces the loss, using the gradients we just computed. After this
            # line, the network is slightly better than it was a moment ago.
            optimizer.step()

            # Track the loss so we can report an average for the epoch.
            # .item() pulls the plain Python number out of the tensor.
            running_loss += loss.item()

        # ---- End of one epoch: report how we're doing ----
        avg_loss = running_loss / len(train_loader)   # average loss per batch
        train_acc = evaluate(model, train_loader)     # accuracy on data we trained on
        val_acc   = evaluate(model, val_loader)       # accuracy on UNSEEN data

        # Watching these three numbers tells the story:
        #  - loss should go DOWN over epochs.
        #  - train_acc should go UP.
        #  - val_acc going up = real learning. If train_acc keeps rising but
        #    val_acc stalls, the network is "overfitting" (memorizing).
        print(f"Epoch {epoch}/{EPOCHS}  |  "
              f"train loss: {avg_loss:.4f}  |  "
              f"train acc: {train_acc:5.1f}%  |  "
              f"val acc: {val_acc:5.1f}%")

    # Save the trained weights to disk so we can reuse them for prediction
    # without retraining every time.
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[train] Saved trained model to '{MODEL_PATH}'")


# ----------------------------------------------------------------------
# STEP 5: PREDICT ON A SINGLE IMAGE
# Load the saved model, run one image through it, and print cat/dog +
# confidence. Confidence comes from softmax, which turns the two raw scores
# into probabilities that add up to 100%.
# ----------------------------------------------------------------------
def predict(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"[predict] No saved model found at '{MODEL_PATH}'. "
              f"Run 'python cats_vs_dogs.py' first to train one.")
        return

    # Rebuild the same network architecture and load the trained weights into it.
    model = SmallCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()   # evaluation mode: dropout off, deterministic output

    # Use the exact same preprocessing recipe we used during training.
    transform = build_transform()

    # Open the image, force RGB (in case it's grayscale or has transparency),
    # apply the transform, then add a fake "batch" dimension with unsqueeze(0)
    # because the model always expects a batch, even of size 1.
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)                       # raw scores
        probs = F.softmax(outputs, dim=1)             # -> probabilities (sum to 1)
        confidence, predicted = torch.max(probs, 1)   # best prob and its index

    # Map the winning index (0 or 1) back to a human-readable label.
    label = CLASSES[predicted.item()]
    print(f"[predict] '{image_path}'  ->  {label}  "
          f"(confidence: {confidence.item() * 100:.1f}%)")


# ----------------------------------------------------------------------
# MAIN — decides whether we're training or predicting based on the command line.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tiny cats-vs-dogs CNN")
    parser.add_argument(
        "--predict",
        metavar="IMAGE_PATH",
        help="Path to an image to classify (uses the already-trained model).",
    )
    args = parser.parse_args()

    if args.predict:
        # Prediction mode: just classify the one image the user gave us.
        predict(args.predict)
    else:
        # Training mode: get data, build the model, train, and save it.
        train_loader, val_loader = make_dataloaders()
        model = SmallCNN().to(DEVICE)

        # A quick note on model size: more layers/filters = potentially more
        # accurate but slower. This tiny model is tuned to finish on a CPU.
        print(f"[model] Training a small CNN for {EPOCHS} epochs on CPU...\n")
        train(model, train_loader, val_loader)

        print("\nAll done! Try classifying an image with:")
        print("    python cats_vs_dogs.py --predict /path/to/your_photo.jpg")


if __name__ == "__main__":
    main()
