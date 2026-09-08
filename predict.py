"""Classify a single image with the trained model.

Run: python predict.py path/to/photo.jpg
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from model import CLASSES, DEVICE, MODEL_PATH, SmallCNN, build_transform


def load_model(weights_path=MODEL_PATH):
    """Rebuild the network and load trained weights into it."""
    model = SmallCNN().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


def classify(model, image_path):
    """Return (label, confidence) for one image. Confidence is 0..1."""
    transform = build_transform()

    # Force RGB in case the image is grayscale or has an alpha channel, then
    # add a batch dimension because the model always expects a batch.
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
        confidence, predicted = torch.max(probs, 1)

    return CLASSES[predicted.item()], confidence.item()


def predict(image_path, weights_path=MODEL_PATH):
    """Command-line entry point. Prints the result, returns an exit code."""
    if not os.path.exists(weights_path):
        print(f"[predict] No saved model at '{weights_path}'. Run 'python train.py' first.")
        return 1
    if not os.path.exists(image_path):
        print(f"[predict] No such image: '{image_path}'")
        return 1

    label, confidence = classify(load_model(weights_path), image_path)
    print(f"[predict] '{image_path}'  ->  {label}  "
          f"(confidence: {confidence * 100:.1f}%)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Classify one image as cat or dog")
    parser.add_argument("image", metavar="IMAGE_PATH", help="Path to the image to classify")
    args = parser.parse_args()
    sys.exit(predict(args.image))


if __name__ == "__main__":
    main()
