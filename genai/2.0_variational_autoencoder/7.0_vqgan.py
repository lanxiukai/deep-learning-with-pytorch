"""VQGAN perceptual tokenizer with a class-conditional causal Transformer prior.

Reconstruction and generation flow (k is a raster-ordered token sequence):
    encoder(x)           -> z_e -> EMA codebook vectors z_q, indices k
    decoder(z_q)         -> reconstruction -> perceptual and PatchGAN losses
    Transformer(k_<t, c) -> logits for k_t -> sample k_t
    codebook(k)          -> z_q -> decoder -> generated image
model(x) returns (reconstruction, indices, commitment_loss). The tokenizer is
unconditional; only the prior uses c, with G=0 and NoG=1. See
dl_utils/vae/vqgan.py and token_priors.py for the model paths.

Stage 1 minimizes mean RGB L1 + frozen VGG LPIPS + commitment loss + an
adaptively weighted generator hinge loss. The adversarial weight matches
gradient norms at the decoder's last layer and is detached. PatchGAN is frozen
during the autoencoder update; its own hinge update uses detached reconstructions.
Adversarial training starts at zero-based batch step 1,000. The codebook uses
EMA rather than autograd, with straight-through gradients to the encoder.

Stage 2 freezes the tokenizer and encodes every image once into memory. The
Transformer fits p(k | c) = product_t p(k_t | k_<t, c) with shifted BOS inputs,
a causal mask, and mean token cross-entropy. Training predicts all positions
in parallel; generation samples 256 tokens sequentially before decoding.

RESUME=True restores the same recipe at an epoch boundary, including optimizer
state, loss history, and the stage-1 global step. A prior checkpoint also
restores its frozen tokenizer. The final file stores the tokenizer/prior pair,
model configuration, and class order. There is no validation or model selection.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py --dataset glasses.
    Resize to 256x256 RGB and normalize from [0, 1] to [-1, 1].
    Use all 4,500 training images: 2,543 G and 1,957 NoG.
    Stage 1 ignores labels; stage 2 uses them for conditional token prediction.

Outputs:
    output/vae/vqgan/model.pth: final tokenizer/prior pair
    output/vae/vqgan/tokenizer_latest.pth: tokenizer/discriminator recovery
    output/vae/vqgan/prior_latest.pth: conditional prior recovery
    output/vae/vqgan/tokenizer_loss.png: autoencoder and discriminator curves
    output/vae/vqgan/prior_loss.png: token cross-entropy curve
    output/vae/vqgan/training/tokenizer_epoch_*.png: original/reconstruction rows
    output/vae/vqgan/training/prior_epoch_*.png: alternating G/NoG samples

Training data -- glasses-256 (fresh run):
Training images:          4,500
Batch size:                  16
Samples per epoch:        4,500 (281 full batches + 4 images; drop_last=False)
Tokenizer/prior epochs:      30 / 30
Optimizer updates:        8,460 each for tokenizer and prior; 7,460 for PatchGAN

Default dimensions:
Training/generated image: 256x256 RGB in [-1, 1]
Token grid / sequence:    16x16 / 256 tokens; four 2x downsampling stages
Codebook:                 512 vectors, 64 values each; commitment coefficient 0.25
Tokenizer/PatchGAN width: 128 / 64 channels
Transformer:              256 values, 8 heads, 4 layers, 2 classes, dropout 0.0
Model size:               tokenizer 2.681 M / prior 3.488 M / PatchGAN 1.251 M
                          Counts exclude LPIPS; tokenizer includes frozen codebook.
EMA decay / epsilon:      0.99 / 1e-5
Loss weights:             perceptual 1.0, VQ 1.0, adversarial scale 1.0
Optimizer:                tokenizer/PatchGAN Adam, betas (0.5, 0.9), constant 2e-4
                          Prior AdamW, constant 3e-4.
Previews:                 8 images; epochs 1, every 10, and final; temperature 1.0

Run without arguments; edit the constants below to experiment. LPIPS uses
pretrained VGG weights. Comparison with 6.0 changes architecture, objective,
prior, conditioning, and training budget together.
"""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.data.datasets.glasses import GLASSES_CLASS_NAMES
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.training import discriminator_hinge_loss, generator_hinge_loss
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.discrete_tokenizers import TOKENIZER_DOWNSAMPLE_STEPS
from dl_utils.vae.discrete_workflow import (
    encode_dataset,
    epoch_checkpoint,
    fixed_images,
    glasses_loader,
    prepare_training_output,
    save_loss_curves,
    save_pair,
    save_reconstruction,
    seed_epoch_loader,
)
from dl_utils.vae.token_priors import CausalTransformerPrior
from dl_utils.vae.vqgan import (
    LPIPSPerceptualLoss,
    PatchDiscriminator,
    VQPerceptualAutoencoder,
    adaptive_adversarial_weight,
)

# Paths and data
PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "vqgan"
IMAGE_SIZE = 256

# Tokenizer configuration
DOWNSAMPLE_STEPS = TOKENIZER_DOWNSAMPLE_STEPS
HIDDEN_CHANNELS = 128
LATENT_CHANNELS = 64
CODEBOOK_SIZE = 512
COMMITMENT = 0.25
EMA_DECAY = 0.99
EMA_EPSILON = 1e-5

# Adversarial objective
DISCRIMINATOR_CHANNELS = 64
PERCEPTUAL_WEIGHT = 1.0
VQ_WEIGHT = 1.0
DISCRIMINATOR_WEIGHT = 1.0
DISCRIMINATOR_START = 1000

# Prior configuration
NUM_CLASSES = len(GLASSES_CLASS_NAMES)
PRIOR_DIM = 256
PRIOR_HEADS = 8
PRIOR_LAYERS = 4
PRIOR_DROPOUT = 0.0

# Training configuration
RESUME = True
TOKENIZER_EPOCHS = 30
PRIOR_EPOCHS = 30
BATCH_SIZE = 16
ADAM_BETAS = (0.5, 0.9)
LR = 2e-4
DISCRIMINATOR_LR = 2e-4
PRIOR_LR = 3e-4
WORKERS = 4
SEED = 42

# Preview configuration
SAMPLE_EVERY = 10
NUM_SAMPLES = 8
TEMPERATURE = 1.0


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
) -> tuple[Tensor, Tensor]:
    reconstruction, _, vq_loss = model(images)
    pixel_l1 = F.l1_loss(reconstruction, images)
    perceptual_loss = perceptual(reconstruction, images)
    reconstruction_objective = pixel_l1 + float(perceptual_weight) * perceptual_loss

    discriminator.requires_grad_(False)
    adversarial = images.new_zeros(())
    adversarial_scale = images.new_zeros(())
    if step >= discriminator_start:
        adversarial = generator_hinge_loss(discriminator(reconstruction))
        adversarial_scale = adaptive_adversarial_weight(
            reconstruction_objective,
            adversarial,
            model.decoder.last_layer,
            scale=discriminator_weight,
        )
    loss = (
        reconstruction_objective + vq_weight * vq_loss + adversarial_scale * adversarial
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    discriminator.requires_grad_(True)
    return reconstruction.detach(), loss.detach()


def vqgan_discriminator_step(
    discriminator: PatchDiscriminator,
    images: Tensor,
    reconstruction: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    discriminator_start: int,
) -> Tensor:
    if step < discriminator_start:
        return images.new_zeros(())
    real_logits = discriminator(images)
    fake_logits = discriminator(reconstruction.detach())
    loss = 0.5 * discriminator_hinge_loss(real_logits, fake_logits)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()


def train_tokenizer(
    model: VQPerceptualAutoencoder,
    loader: DataLoader,
    device: torch.device,
    recipe: Mapping[str, Any],
) -> None:
    discriminator = PatchDiscriminator(DISCRIMINATOR_CHANNELS).to(device)
    perceptual = LPIPSPerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=ADAM_BETAS)
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=ADAM_BETAS
    )
    checkpoint = epoch_checkpoint(
        OUTPUT_DIR / "tokenizer_latest.pth",
        {"tokenizer": model, "discriminator": discriminator},
        {"tokenizer": optimizer, "discriminator": d_optimizer},
        recipe,
    )
    completed, state = checkpoint.resume(
        checkpoint.path if RESUME and checkpoint.path.is_file() else None,
        initial_state={"history": [], "global_step": 0},
    )
    originals = fixed_images(loader, NUM_SAMPLES)
    with tqdm(
        total=TOKENIZER_EPOCHS * len(loader),
        initial=completed * len(loader),
        desc="Stage 1/2: VQGAN tokenizer",
        unit="batch",
    ) as progress:
        for epoch in range(completed + 1, TOKENIZER_EPOCHS + 1):
            model.train()
            discriminator.train()
            seed_epoch_loader(loader, SEED, epoch)
            metrics = MetricAccumulator(("autoencoder", "discriminator"), device=device)
            progress.set_description(
                f"Stage 1/2: VQGAN tokenizer {epoch}/{TOKENIZER_EPOCHS}", refresh=False
            )
            for images, _ in loader:
                images = images.to(device)
                reconstruction, ae_loss = vqgan_autoencoder_step(
                    model,
                    discriminator,
                    perceptual,
                    images,
                    optimizer,
                    step=state["global_step"],
                    discriminator_start=DISCRIMINATOR_START,
                    perceptual_weight=PERCEPTUAL_WEIGHT,
                    vq_weight=VQ_WEIGHT,
                    discriminator_weight=DISCRIMINATOR_WEIGHT,
                )
                d_loss = vqgan_discriminator_step(
                    discriminator,
                    images,
                    reconstruction,
                    d_optimizer,
                    step=state["global_step"],
                    discriminator_start=DISCRIMINATOR_START,
                )
                metrics.add_batch_means((ae_loss, d_loss), num_examples=len(images))
                state["global_step"] += 1
                progress.update(1)
            losses = metrics.compute_weighted_means(require_finite=True)
            progress.set_postfix(
                ae=f"{losses['autoencoder']:.4f}",
                d=f"{losses['discriminator']:.4f}",
                refresh=False,
            )
            state["history"].append(losses)
            checkpoint.save(epoch, state)
            if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == TOKENIZER_EPOCHS:
                training_dir = OUTPUT_DIR / "training"
                save_reconstruction(
                    model,
                    originals,
                    training_dir / f"tokenizer_epoch_{epoch:03d}.png",
                    device,
                )
    save_loss_curves(state["history"], OUTPUT_DIR / "tokenizer_loss.png")


def train_prior(
    tokenizer: VQPerceptualAutoencoder,
    images: DataLoader,
    device: torch.device,
    recipe: Mapping[str, Any],
) -> CausalTransformerPrior:
    prior = CausalTransformerPrior(**recipe["model"]["prior"]).to(device)
    optimizer = torch.optim.AdamW(prior.parameters(), lr=PRIOR_LR)
    checkpoint = epoch_checkpoint(
        OUTPUT_DIR / "prior_latest.pth",
        {"tokenizer": tokenizer, "prior": prior},
        {"prior": optimizer},
        recipe,
    )
    completed, state = checkpoint.resume(
        checkpoint.path if RESUME and checkpoint.path.is_file() else None,
        initial_state={"history": []},
    )
    tokenizer.eval().requires_grad_(False)
    if completed == PRIOR_EPOCHS:
        save_loss_curves(state["history"], OUTPUT_DIR / "prior_loss.png")
        return prior.eval()
    tokens = encode_dataset(tokenizer, images, device)
    side = IMAGE_SIZE // (2**tokenizer.downsample_steps)
    with tqdm(
        total=PRIOR_EPOCHS * len(tokens),
        initial=completed * len(tokens),
        desc="Stage 2/2: Transformer prior",
        unit="batch",
    ) as progress:
        for epoch in range(completed + 1, PRIOR_EPOCHS + 1):
            prior.train()
            seed_epoch_loader(tokens, SEED, epoch)
            metrics = MetricAccumulator(("nll",), device=device)
            progress.set_description(
                f"Stage 2/2: Transformer prior {epoch}/{PRIOR_EPOCHS}", refresh=False
            )
            for indices, labels in tokens:
                indices, labels = indices.to(device), labels.to(device)
                logits, targets = prior.teacher_forcing(indices, labels)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                metrics.add_batch_means((loss,), num_examples=len(indices))
                progress.update(1)
            losses = metrics.compute_weighted_means(require_finite=True)
            progress.set_postfix(nll=f"{losses['nll']:.4f}", refresh=False)
            state["history"].append(losses)
            checkpoint.save(epoch, state)
            if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == PRIOR_EPOCHS:
                with torch.random.fork_rng(), torch.inference_mode():
                    torch.manual_seed(SEED)
                    prior.eval()
                    labels = torch.arange(NUM_SAMPLES, device=device).remainder(
                        NUM_CLASSES
                    )
                    indices = prior.sample(
                        NUM_SAMPLES,
                        device=device,
                        labels=labels,
                        temperature=TEMPERATURE,
                    )
                    samples = tokenizer.decode_indices(
                        indices.reshape(NUM_SAMPLES, side, side)
                    )
                training_dir = OUTPUT_DIR / "training"
                save_image(
                    samples.mul(0.5).add(0.5),
                    training_dir / f"prior_epoch_{epoch:03d}.png",
                    nrow=4,
                )
    save_loss_curves(state["history"], OUTPUT_DIR / "prior_loss.png")
    return prior.eval()


def train() -> None:
    config = {
        "image_size": IMAGE_SIZE,
        "tokenizer": {
            "latent_channels": LATENT_CHANNELS,
            "codebook_size": CODEBOOK_SIZE,
            "hidden_channels": HIDDEN_CHANNELS,
            "commitment": COMMITMENT,
            "ema_decay": EMA_DECAY,
            "ema_epsilon": EMA_EPSILON,
            "downsample_steps": DOWNSAMPLE_STEPS,
        },
        "prior": {
            "vocabulary_size": CODEBOOK_SIZE,
            "sequence_length": (IMAGE_SIZE // 2**DOWNSAMPLE_STEPS) ** 2,
            "model_dim": PRIOR_DIM,
            "heads": PRIOR_HEADS,
            "layers": PRIOR_LAYERS,
            "dropout": PRIOR_DROPOUT,
            "num_classes": NUM_CLASSES,
        },
    }
    recipe = {
        "model": config,
        "tokenizer_epochs": TOKENIZER_EPOCHS,
        "prior_epochs": PRIOR_EPOCHS,
        "lr": LR,
        "d_lr": DISCRIMINATOR_LR,
        "prior_lr": PRIOR_LR,
        "betas": ADAM_BETAS,
        "d_channels": DISCRIMINATOR_CHANNELS,
        "perceptual_weight": PERCEPTUAL_WEIGHT,
        "vq_weight": VQ_WEIGHT,
        "discriminator_weight": DISCRIMINATOR_WEIGHT,
        "discriminator_start": DISCRIMINATOR_START,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
    }
    prepare_training_output(OUTPUT_DIR, resume=RESUME, recipe=recipe)
    set_seed(SEED)
    device = try_gpu()
    loader = glasses_loader(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        device,
        shuffle=True,
        num_workers=WORKERS,
        conditional=True,
    )
    tokenizer = VQPerceptualAutoencoder(**config["tokenizer"]).to(device)
    if not (RESUME and (OUTPUT_DIR / "prior_latest.pth").is_file()):
        train_tokenizer(tokenizer, loader, device, recipe)
    prior = train_prior(tokenizer, loader, device, recipe)
    save_pair(OUTPUT_DIR / "model.pth", "vqgan", tokenizer, prior, config)


def main() -> None:
    train()


if __name__ == "__main__":
    main()
