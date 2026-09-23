"""Train a VQGAN tokenizer and its frozen-token causal Transformer prior.

The first stage adds three ideas to ``6.0_vq_vae.py``:

* frozen, learned LPIPS v0.1 distance with a VGG trunk;
* a PatchGAN discriminator with hinge loss and delayed activation;
* a fixed adversarial weight after the discriminator warm-up.

Stage 2 freezes the tokenizer and trains a class-conditional causal
Transformer. Four stride-2 blocks compress each 128x128 image to 8x8 tokens.
The final checkpoint stores tokenizer and discriminator weights; the prior
is saved alongside it for the evaluation lesson.

This entry keeps only training and bounded training-time validation. Run
``7.1_vqgan_evaluation.py`` to reload artifacts independently and compare
paired fidelity, token rates, prior likelihood,
and complete generation under one held-out protocol.

Data:
    data/celeba (official train and validation splits), prepared by
    tool_scripts/download_dataset.py --dataset celeba.

Outputs:
    output/vae/vqgan/tokenizer_*.png: reconstructions
    output/vae/vqgan/prior_*.png: one sample per Smiling label
    output/vae/vqgan/vqgan.pth: tokenizer and discriminator
    output/vae/vqgan/transformer_prior.pth: token prior

Training data -- CelebA-128:
Training images:             162,770
Validation images:            19,867
Batch size:                       16
Samples per epoch:           162,768 (10,173 full batches; drop_last=True)
Tokenizer epochs:                 30
Prior epochs:                     30
Optimizer updates:           305,190 tokenizer / 304,190 discriminator
                             305,190 Transformer prior
The tokenizer remains label-free; the frozen-token Transformer uses the binary
Smiling attribute. D starts at tokenizer step 1,000. Two shuffled images are
omitted per epoch, and validation uses the first 1,024 validation images.
Both splits center-crop aligned faces to 178x178, resize to 128x128, and use
no random horizontal flips.

Default dimensions:
Training input:             128x128 RGB
Generated image:            128x128 RGB
Latent token grid:              8x8 indices (512-entry codebook)

Model size:
VQGAN tokenizer:               2.68 M parameters
Patch discriminator:           1.25 M parameters
Stage-one trainable total:      3.93 M parameters (frozen LPIPS excluded)
Frozen LPIPS VGG:              14.72 M parameters (not optimized or stored)
Conditional Transformer prior: 3.44 M parameters (tokenizer frozen)
Stored model total:             7.37 M parameters

Run this script without arguments to train both stages in order.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.data.celeba import (
    CELEBA_SMILING_ATTRIBUTE,
    CELEBA_SMILING_CLASSES,
    make_aligned_celeba_train_validation_loaders,
)
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.sn_gan import (
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.training_artifacts import save_training_metrics
from dl_utils.vae.perceptual_autoencoder import (
    LPIPSPerceptualLoss,
    PatchDiscriminator,
    VQPerceptualAutoencoder,
)
from dl_utils.vae.quantization import TokenUsageAccumulator
from dl_utils.vae.token_prior import (
    CausalTransformerPrior,
    make_fixed_class_labels,
)

PROJECT_ROOT = infer_project_root()
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "vqgan"
TOKENIZER_CHECKPOINT_NAME = "vqgan.pth"
PRIOR_CHECKPOINT_NAME = "transformer_prior.pth"
RECONSTRUCTION_SAMPLES = 8
PROGRESS_INTERVAL = 0.5
MAX_METRIC_PANELS = 4
ADAM_BETAS = (0.5, 0.9)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "celeba"
IMAGE_SIZE = 128
NUM_CLASSES = len(CELEBA_SMILING_CLASSES)
DOWNSAMPLE_STEPS = 4
LATENT_GRID_SIZE = IMAGE_SIZE // (2**DOWNSAMPLE_STEPS)
TOKENS_PER_IMAGE = LATENT_GRID_SIZE**2
SAMPLES_PER_CLASS = 1


# Edit these defaults to explore the lesson.
DATA_DIR = DEFAULT_DATA_DIR
TOKENIZER_EPOCHS = 30
PRIOR_EPOCHS = 30
BATCH_SIZE = 16
HIDDEN_CHANNELS = 128
LATENT_CHANNELS = 64
CODEBOOK_SIZE = 512
COMMITMENT = 0.25
DISCRIMINATOR_CHANNELS = 64
PERCEPTUAL_WEIGHT = 1.0
VQ_WEIGHT = 1.0
DISCRIMINATOR_WEIGHT = 1.0
DISCRIMINATOR_START = 1000
PRIOR_DIM = 256
PRIOR_HEADS = 8
PRIOR_LAYERS = 4
PRIOR_DROPOUT = 0.0
LR = 2e-4
DISCRIMINATOR_LR = 2e-4
PRIOR_LR = 3e-4
TEMPERATURE = 1.0
SAMPLE_EVERY = 5
VALIDATION_EXAMPLES = 1_024
WORKERS = 4
SEED = 42


def vqgan_autoencoder_step(
    model: VQPerceptualAutoencoder,
    discriminator: PatchDiscriminator,
    perceptual: nn.Module,
    images: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    discriminator_start: int,
    perceptual_weight: float,
    vq_weight: float,
    discriminator_weight: float,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    reconstruction, indices, vq_loss, diagnostics = model(images)
    pixel_l1 = F.l1_loss(reconstruction, images)
    perceptual_loss = perceptual(reconstruction, images)
    reconstruction_objective = pixel_l1 + float(perceptual_weight) * perceptual_loss

    discriminator.requires_grad_(False)
    adversarial = generator_hinge_loss(discriminator(reconstruction))
    adversarial_scale = images.new_tensor(
        discriminator_weight if step >= discriminator_start else 0.0
    )
    loss = (
        reconstruction_objective + vq_weight * vq_loss + adversarial_scale * adversarial
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    discriminator.requires_grad_(True)
    return (
        reconstruction.detach(),
        indices.detach(),
        {
            "autoencoder": loss.detach(),
            "pixel_l1": pixel_l1.detach(),
            "perceptual": perceptual_loss.detach(),
            "vq": vq_loss.detach(),
            "generator_adversarial": adversarial.detach(),
            "adversarial_scale": adversarial_scale.detach(),
            "perplexity": diagnostics["perplexity"],
            "active_codes": diagnostics["active_codes"].float(),
        },
    )


def vqgan_discriminator_step(
    discriminator: PatchDiscriminator,
    images: Tensor,
    reconstruction: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    discriminator_start: int,
) -> dict[str, Tensor]:
    if step < discriminator_start:
        zero = images.new_zeros(())
        return {"discriminator": zero, "real_logit": zero, "fake_logit": zero}
    real_logits = discriminator(images)
    fake_logits = discriminator(reconstruction.detach())
    loss = 0.5 * discriminator_hinge_loss(real_logits, fake_logits)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "discriminator": loss.detach(),
        "real_logit": real_logits.mean().detach(),
        "fake_logit": fake_logits.mean().detach(),
    }


@torch.inference_mode()
def validate_tokenizer(
    model: VQPerceptualAutoencoder,
    discriminator: PatchDiscriminator,
    perceptual: nn.Module,
    loader: DataLoader,
    *,
    max_examples: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    discriminator.eval()
    perceptual.eval()
    vocabulary_size = model.quantizer.codebook_size
    metrics = MetricAccumulator(
        (
            "pixel_l1",
            "perceptual_distance",
            "quantization_mse",
            "real_logit",
            "reconstruction_logit",
        ),
        device=device,
    )
    examples = 0
    usage = TokenUsageAccumulator(vocabulary_size)
    for images, _ in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        images = images[:remaining].to(device, non_blocking=True)
        reconstruction, indices, _, diagnostics = model(images)
        metrics.add_batch_means(
            (
                F.l1_loss(reconstruction, images),
                perceptual(reconstruction, images),
                diagnostics["quantization_mse"],
                discriminator(images).mean(),
                discriminator(reconstruction).mean(),
            ),
            num_examples=images.shape[0],
        )
        usage.update(indices)
        examples += images.shape[0]
    if examples == 0:
        raise ValueError("tokenizer validation observed no examples")
    values = metrics.compute_weighted_means()
    statistics = usage.statistics()
    entropy_bits = statistics["token_entropy_nats"].item() / math.log(2)
    return {
        "examples": float(examples),
        **values,
        "perplexity": statistics["perplexity"].item(),
        "active_codes": statistics["active_codes"].item(),
        "usage_fraction": statistics["usage_fraction"].item(),
        "marginal_entropy_bits_per_token": entropy_bits,
        "marginal_entropy_bits_per_image": TOKENS_PER_IMAGE * entropy_bits,
        "fixed_length_bits_per_image": (
            TOKENS_PER_IMAGE * math.ceil(math.log2(vocabulary_size))
        ),
    }


@torch.inference_mode()
def validate_prior(
    tokenizer: VQPerceptualAutoencoder,
    prior: CausalTransformerPrior,
    loader: DataLoader,
    *,
    max_examples: int,
    device: torch.device,
) -> dict[str, float]:
    tokenizer.eval()
    prior.eval()
    metrics = MetricAccumulator(("nll",), device=device)
    examples = 0
    for images, labels in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        images = images[:remaining].to(device, non_blocking=True)
        labels = labels[:remaining].to(device, non_blocking=True)
        indices = tokenizer.encode_indices(images)
        logits, targets = prior.teacher_forcing(indices, labels)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        metrics.add_batch_means((loss,), num_examples=images.shape[0])
        examples += images.shape[0]
    if examples == 0:
        raise ValueError("prior validation observed no examples")
    nll = metrics.compute_weighted_means()["nll"]
    return {
        "examples": float(examples),
        "nll_nats_per_token": nll,
        "bits_per_token": nll / math.log(2),
        "bits_per_image": TOKENS_PER_IMAGE * nll / math.log(2),
    }


def train_tokenizer(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> VQPerceptualAutoencoder:
    config = {
        "latent_channels": LATENT_CHANNELS,
        "codebook_size": CODEBOOK_SIZE,
        "hidden_channels": HIDDEN_CHANNELS,
        "commitment": COMMITMENT,
        "downsample_steps": DOWNSAMPLE_STEPS,
    }
    model = VQPerceptualAutoencoder(**config).to(device)
    discriminator = PatchDiscriminator(DISCRIMINATOR_CHANNELS).to(device)
    perceptual = LPIPSPerceptualLoss().to(device)
    ae_optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=ADAM_BETAS)
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=ADAM_BETAS
    )
    global_step = 0
    training_dir = out_dir / "training"
    if not training_dir.exists():
        reset_dir(str(training_dir))
    history = []
    with tqdm(
        total=TOKENIZER_EPOCHS * len(train_loader),
        desc=f"vqgan 1/{TOKENIZER_EPOCHS}",
        unit="batch",
        mininterval=PROGRESS_INTERVAL,
    ) as progress:
        for epoch in range(1, TOKENIZER_EPOCHS + 1):
            progress.set_description(f"vqgan {epoch}/{TOKENIZER_EPOCHS}", refresh=False)
            model.train()
            discriminator.train()
            metrics_accumulator = MetricAccumulator(
                (
                    "autoencoder",
                    "pixel_l1",
                    "perceptual",
                    "vq",
                    "generator_adversarial",
                    "adversarial_scale",
                    "discriminator",
                    "real_logit",
                    "fake_logit",
                ),
                device=device,
            )
            usage = TokenUsageAccumulator(CODEBOOK_SIZE)
            preview = None
            for images, _ in train_loader:
                images = images.to(device, non_blocking=True)
                reconstruction, indices, metrics = vqgan_autoencoder_step(
                    model,
                    discriminator,
                    perceptual,
                    images,
                    ae_optimizer,
                    step=global_step,
                    discriminator_start=DISCRIMINATOR_START,
                    perceptual_weight=PERCEPTUAL_WEIGHT,
                    vq_weight=VQ_WEIGHT,
                    discriminator_weight=DISCRIMINATOR_WEIGHT,
                )
                d_metrics = vqgan_discriminator_step(
                    discriminator,
                    images,
                    reconstruction,
                    d_optimizer,
                    step=global_step,
                    discriminator_start=DISCRIMINATOR_START,
                )
                metrics_accumulator.add_batch_means(
                    (
                        metrics["autoencoder"],
                        metrics["pixel_l1"],
                        metrics["perceptual"],
                        metrics["vq"],
                        metrics["generator_adversarial"],
                        metrics["adversarial_scale"],
                        d_metrics["discriminator"],
                        d_metrics["real_logit"],
                        d_metrics["fake_logit"],
                    ),
                    num_examples=images.shape[0],
                )
                usage.update(indices)
                global_step += 1
                running_metrics = metrics_accumulator.compute_weighted_means()
                progress.set_postfix(
                    ae=f"{running_metrics['autoencoder']:.4f}",
                    d=f"{running_metrics['discriminator']:.4f}",
                    refresh=False,
                )
                progress.update(1)
                preview = (
                    images[:RECONSTRUCTION_SAMPLES].detach(),
                    reconstruction[:RECONSTRUCTION_SAMPLES].detach(),
                )
            if preview is None:
                raise ValueError("training loader produced no batches; reduce batch size")
            means = metrics_accumulator.compute_weighted_means()
            epoch_usage = usage.statistics()
            history.append(
                means
                | {
                    "perplexity": epoch_usage["perplexity"].item(),
                    "active_codes": epoch_usage["active_codes"].item(),
                    "entropy_bits": epoch_usage["token_entropy_nats"].item()
                    / math.log(2),
                }
            )
            if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == TOKENIZER_EPOCHS:
                save_image(
                    torch.cat(preview).mul(0.5).add(0.5),
                    training_dir / f"tokenizer_epoch_{epoch:03d}.png",
                    nrow=RECONSTRUCTION_SAMPLES,
                )
    validation = validate_tokenizer(
        model,
        discriminator,
        perceptual,
        validation_loader,
        max_examples=VALIDATION_EXAMPLES,
        device=device,
    )
    save_training_metrics(
        history, out_dir, prefix="tokenizer", max_panels=MAX_METRIC_PANELS
    )
    torch.save(
        {
            "model_name": "vqgan_tokenizer",
            "state_dict": model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "discriminator_config": {
                "base_channels": DISCRIMINATOR_CHANNELS,
            },
            "perceptual_loss": "lpips-v0.1-vgg",
            "model_config": config,
            "validation": validation,
        },
        out_dir / TOKENIZER_CHECKPOINT_NAME,
    )
    return model.eval().requires_grad_(False)


def train_prior(
    tokenizer: VQPerceptualAutoencoder,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> None:
    tokenizer.eval().requires_grad_(False)
    vocabulary_size = tokenizer.quantizer.codebook_size
    prior = CausalTransformerPrior(
        vocabulary_size,
        TOKENS_PER_IMAGE,
        model_dim=PRIOR_DIM,
        heads=PRIOR_HEADS,
        layers=PRIOR_LAYERS,
        dropout=PRIOR_DROPOUT,
        num_classes=NUM_CLASSES,
    ).to(device)
    optimizer = torch.optim.AdamW(prior.parameters(), lr=PRIOR_LR)
    training_dir = out_dir / "training"
    if not training_dir.exists():
        reset_dir(str(training_dir))
    history = []
    with tqdm(
        total=PRIOR_EPOCHS * len(train_loader),
        desc=f"Transformer 1/{PRIOR_EPOCHS}",
        unit="batch",
        mininterval=PROGRESS_INTERVAL,
    ) as progress:
        for epoch in range(1, PRIOR_EPOCHS + 1):
            progress.set_description(f"Transformer {epoch}/{PRIOR_EPOCHS}", refresh=False)
            prior.train()
            metrics = MetricAccumulator(("nll",), device=device)
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # Frozen tokens still participate in the prior's backward pass.
                with torch.no_grad():
                    indices = tokenizer.encode_indices(images)
                logits, targets = prior.teacher_forcing(indices, labels)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                metrics.add_batch_means((loss,), num_examples=images.shape[0])
                nll = metrics.compute_weighted_means()["nll"]
                progress.set_postfix(
                    nll=f"{nll:.4f}",
                    bpt=f"{nll / math.log(2):.3f}",
                    refresh=False,
                )
                progress.update(1)
            nll = metrics.compute_weighted_means()["nll"]
            history.append({"nll": nll, "bits_per_token": nll / math.log(2)})
            if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == PRIOR_EPOCHS:
                with torch.inference_mode():
                    prior.eval()
                    labels = make_fixed_class_labels(NUM_CLASSES, SAMPLES_PER_CLASS, device)
                    indices = prior.sample(
                        labels.shape[0],
                        device=device,
                        labels=labels,
                        temperature=TEMPERATURE,
                    ).reshape(labels.shape[0], LATENT_GRID_SIZE, LATENT_GRID_SIZE)
                    samples = tokenizer.decode_indices(indices)
                    save_image(
                        samples.mul(0.5).add(0.5),
                        training_dir / f"prior_epoch_{epoch:03d}.png",
                        nrow=NUM_CLASSES,
                    )
                    if epoch == PRIOR_EPOCHS:
                        save_image(
                            samples.mul(0.5).add(0.5),
                            out_dir / "prior_samples.png",
                            nrow=NUM_CLASSES,
                        )
    validation = validate_prior(
        tokenizer,
        prior,
        validation_loader,
        max_examples=VALIDATION_EXAMPLES,
        device=device,
    )
    prior_config = {
        "vocabulary_size": vocabulary_size,
        "sequence_length": TOKENS_PER_IMAGE,
        "model_dim": PRIOR_DIM,
        "heads": PRIOR_HEADS,
        "layers": PRIOR_LAYERS,
        "dropout": PRIOR_DROPOUT,
        "num_classes": NUM_CLASSES,
    }
    save_training_metrics(
        history, out_dir, prefix="prior", max_panels=MAX_METRIC_PANELS
    )
    torch.save(
        {
            "state_dict": prior.state_dict(),
            "model_name": "vqgan_transformer_prior",
            "model_config": prior_config,
            "validation": validation,
        },
        out_dir / PRIOR_CHECKPOINT_NAME,
    )


def train() -> None:
    device = try_gpu()
    out_dir = OUTPUT_DIR
    train_loader, validation_loader = make_aligned_celeba_train_validation_loaders(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        device,
        num_workers=WORKERS,
        attribute=CELEBA_SMILING_ATTRIBUTE,
    )
    if not out_dir.exists():
        reset_dir(str(out_dir))
    reset_dir(str(out_dir / "training"))
    tokenizer = train_tokenizer(
        train_loader,
        validation_loader,
        device,
        out_dir,
    )
    train_prior(
        tokenizer,
        train_loader,
        validation_loader,
        device,
        out_dir,
    )


def main() -> None:
    set_seed(SEED)
    train()


if __name__ == "__main__":
    main()
