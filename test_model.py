"""Tests that run in seconds and never touch the dataset.

Nothing here downloads CIFAR-10 or trains. The model is small enough to
instantiate with random weights, which is all these assertions need.
"""

import torch
from PIL import Image

import predict as predict_module
from model import CLASSES, IMAGE_SIZE, SmallCNN, build_transform


def test_forward_pass_returns_logits_per_class():
    """A batch of two images in, one score per class out."""
    model = SmallCNN()
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))

    assert out.shape == (2, 2)


def test_transform_produces_a_normalized_chw_tensor():
    """Any PIL image becomes a fixed-size tensor scaled to roughly -1..1."""
    image = Image.new("RGB", (200, 120), color=(12, 200, 90))
    tensor = build_transform()(image)

    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    # ToTensor gives 0..1, then Normalize(0.5, 0.5) maps that onto -1..1.
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0


def test_transform_maps_the_extremes_exactly():
    """Black lands on -1 and white on +1, which pins the normalization."""
    transform = build_transform()

    black = transform(Image.new("RGB", (8, 8), color=(0, 0, 0)))
    white = transform(Image.new("RGB", (8, 8), color=(255, 255, 255)))

    assert torch.allclose(black, torch.full_like(black, -1.0), atol=1e-6)
    assert torch.allclose(white, torch.full_like(white, 1.0), atol=1e-6)


def test_parameter_count_matches_the_readme():
    """The README quotes these numbers, so they are worth pinning."""
    model = SmallCNN()

    total = sum(p.numel() for p in model.parameters())
    fc1 = sum(p.numel() for p in model.fc1.parameters())

    # README: "About 530,000 parameters, and roughly 99% of them are in the
    # first dense layer alone (8192 x 64)."
    assert total == 529_570
    assert round(100 * fc1 / total) == 99


def test_classify_returns_a_known_label_and_a_probability(tmp_path):
    """Random weights are fine here: we are checking the contract, not accuracy."""
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(180, 90, 40)).save(image_path)

    model = SmallCNN()
    model.eval()

    label, confidence = predict_module.classify(model, str(image_path))

    assert label in CLASSES
    # Softmax over two classes: the winner can never be below half.
    assert 0.5 <= confidence <= 1.0


def test_predict_reports_a_missing_model_file(tmp_path, capsys):
    """Missing weights is a user error, not a traceback."""
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color="white").save(image_path)

    code = predict_module.predict(str(image_path), weights_path=str(tmp_path / "absent.pth"))

    assert code == 1
    assert "No saved model" in capsys.readouterr().out


def test_predict_round_trips_through_saved_weights(tmp_path, capsys):
    """Save random weights, load them back, and classify with them."""
    weights_path = tmp_path / "weights.pth"
    torch.save(SmallCNN().state_dict(), weights_path)

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(30, 140, 200)).save(image_path)

    code = predict_module.predict(str(image_path), weights_path=str(weights_path))

    assert code == 0
    out = capsys.readouterr().out
    assert any(label in out for label in CLASSES)
