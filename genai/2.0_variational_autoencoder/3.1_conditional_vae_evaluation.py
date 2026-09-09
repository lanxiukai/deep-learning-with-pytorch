"""Evaluate cVAE generation paths that the training loss cannot validate.

This script trains a small MNIST classifier only as an evaluator,
then measures:

* class compliance after decoding z ~ N(0, I) with each label;
* within-class feature diversity relative to real test images;
* posterior-mean reconstruction and a decoder condition-shuffle intervention.

The classifier is trained on MNIST train and checked on MNIST test. Its test
accuracy provides context for the generation metrics; it is separate from
the cVAE checkpoint and generation path.

Data:
    data/mnist, downloaded automatically by torchvision when absent. Digit
    labels train the evaluator and condition the generator.

Checkpoints:
    output/vae/conditional_vae/standard-prior/model.pth: required main model

Outputs:
    output/vae/conditional_vae/evaluation/metrics.json: complete comparison
    output/vae/conditional_vae/evaluation/<variant>_conditional_samples.png

Evaluation data -- MNIST:
Classifier training images:        60,000
Classifier samples per epoch:      59,904 (234 full batches)
Classifier epochs:                      3
Classifier optimizer updates:         702
Test images:                       10,000
Batch size:                           256
Generated samples per model:         1,280 (128 per class)
Real diversity reference:            1,280 (128 per class)
Maximum reconstruction examples:     5,000

Default dimensions:
Evaluation input:                  32x32 grayscale
Generated image:                   32x32 grayscale
Latent vector:                         16 values

Model size:
Standard-prior CVAE:                0.971 M parameters (required)
Evaluator classifier:               0.056 M parameters
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.conditional_vae import (
    ConditionalVAE,
)
from dl_utils.vae.vae_common import diagonal_gaussian_kl_from_logvar

PROJECT_ROOT = infer_project_root()
DEFAULT_ROOT = PROJECT_ROOT / "output" / "vae" / "conditional_vae"


# Edit these defaults to explore the lesson.
CLASSIFIER_EPOCHS = 3
BATCH_SIZE = 256
SAMPLES_PER_CLASS = 128
MAX_RECONSTRUCTION_EXAMPLES = 5_000
WORKERS = 4
SEED = 123
STANDARD_PRIOR_CHECKPOINT = DEFAULT_ROOT / "standard-prior" / "model.pth"


class DigitClassifier32(nn.Module):
    """Small task network used only to audit generated-condition compliance."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64, 10)

    def encode(self, x: Tensor) -> Tensor:
        return self.features(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encode(x))


def make_loaders(device: torch.device) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    train_set = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
        train=False,
        download=True,
        transform=transform,
    )
    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": WORKERS > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **common),
        DataLoader(test_set, shuffle=False, drop_last=False, **common),
    )


def train_classifier(
    model: DigitClassifier32,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int,
    device: torch.device,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        examples = 0
        for x, labels in train_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            correct += int((logits.argmax(dim=1) == labels).sum())
            examples += x.shape[0]
        print(f"classifier epoch {epoch:02d}: train accuracy={correct / examples:.4f}")

    model.eval()
    correct = 0
    examples = 0
    with torch.inference_mode():
        for x, labels in test_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            correct += int((model(x).argmax(dim=1) == labels).sum())
            examples += x.shape[0]
    return correct / examples


def _load_model(path: Path, device: torch.device) -> ConditionalVAE:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ConditionalVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def _mean_pairwise_distance(features: Tensor) -> float:
    if features.shape[0] < 2:
        return 0.0
    return float(torch.pdist(features.float(), p=2).mean())


@torch.inference_mode()
def generation_metrics(
    model: ConditionalVAE,
    classifier: DigitClassifier32,
    *,
    samples_per_class: int,
    device: torch.device,
) -> tuple[dict[str, object], Tensor]:
    labels = torch.arange(10, device=device).repeat_interleave(samples_per_class)
    images = model.generate(labels)
    features = classifier.encode(images)
    probabilities = classifier(images).softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    per_class_compliance = []
    per_class_diversity = []
    for class_index in range(10):
        mask = labels == class_index
        per_class_compliance.append(
            float((predictions[mask] == class_index).float().mean())
        )
        per_class_diversity.append(_mean_pairwise_distance(features[mask]))
    return {
        "condition_compliance": float((predictions == labels).float().mean()),
        "target_probability": float(probabilities.gather(1, labels[:, None]).mean()),
        "feature_diversity": sum(per_class_diversity) / 10,
        "per_class_compliance": per_class_compliance,
        "per_class_feature_diversity": per_class_diversity,
    }, images


@torch.inference_mode()
def real_diversity_reference(
    loader: DataLoader,
    classifier: DigitClassifier32,
    *,
    samples_per_class: int,
    device: torch.device,
) -> dict[str, object]:
    buckets: list[list[Tensor]] = [[] for _ in range(10)]
    for x, labels in loader:
        for image, label in zip(x, labels):
            bucket = buckets[int(label)]
            if len(bucket) < samples_per_class:
                bucket.append(image)
        if all(len(bucket) == samples_per_class for bucket in buckets):
            break
    per_class = []
    for bucket in buckets:
        images = torch.stack(bucket).to(device)
        per_class.append(_mean_pairwise_distance(classifier.encode(images)))
    return {
        "feature_diversity": sum(per_class) / 10,
        "per_class_feature_diversity": per_class,
    }


@torch.inference_mode()
def posterior_and_shuffle_metrics(
    model: ConditionalVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    max_examples: int,
) -> dict[str, float]:
    correct_distortion = 0.0
    shuffled_distortion = 0.0
    rate = 0.0
    posterior_prior_mean_gap = 0.0
    examples = 0
    for x, labels in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        labels = labels[:remaining].to(device, non_blocking=True)
        q_mu, q_logvar = model.encode(x, labels)
        p_mu, p_logvar = model.prior(labels)
        correct = model.decode(q_mu, labels)
        shuffled_labels = (labels + 1) % model.num_classes
        shuffled = model.decode(q_mu, shuffled_labels)
        correct_distortion += float(F.binary_cross_entropy(correct, x, reduction="sum"))
        shuffled_distortion += float(
            F.binary_cross_entropy(shuffled, x, reduction="sum")
        )
        rate += float(
            diagonal_gaussian_kl_from_logvar(q_mu, q_logvar, p_mu, p_logvar).sum()
        )
        posterior_prior_mean_gap += float((q_mu - p_mu).square().sum())
        examples += x.shape[0]
    return {
        "posterior_mean_distortion": correct_distortion / examples,
        "shuffled_decoder_condition_distortion": (shuffled_distortion / examples),
        "condition_shuffle_distortion_ratio": (
            shuffled_distortion / correct_distortion
        ),
        "conditional_rate": rate / examples,
        "posterior_prior_mean_squared_gap": (posterior_prior_mean_gap / examples),
    }


def evaluate() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(STANDARD_PRIOR_CHECKPOINT, device)
    train_loader, test_loader = make_loaders(device)
    classifier = DigitClassifier32().to(device)
    classifier_accuracy = train_classifier(
        classifier,
        train_loader,
        test_loader,
        epochs=CLASSIFIER_EPOCHS,
        device=device,
    )
    metrics, images = generation_metrics(
        model,
        classifier,
        samples_per_class=SAMPLES_PER_CLASS,
        device=device,
    )
    metrics.update(
        posterior_and_shuffle_metrics(
            model,
            test_loader,
            device=device,
            max_examples=MAX_RECONSTRUCTION_EXAMPLES,
        )
    )
    results = {
        "classifier_test_accuracy": classifier_accuracy,
        "real_reference": real_diversity_reference(
            test_loader,
            classifier,
            samples_per_class=SAMPLES_PER_CLASS,
            device=device,
        ),
        "models": {"standard-prior": metrics},
    }
    out_dir = DEFAULT_ROOT / "evaluation"
    reset_dir(str(out_dir))
    display = images.reshape(10, SAMPLES_PER_CLASS, 1, 32, 32)[:, :8].flatten(0, 1)
    save_image(display, out_dir / "standard-prior_conditional_samples.png", nrow=8)
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"classifier accuracy={classifier_accuracy:.4f}, "
        f"condition compliance={metrics['condition_compliance']:.4f}, "
        f"feature diversity={metrics['feature_diversity']:.4f}"
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
