"""FID, KID, and feature precision/recall under one explicit image protocol.

Uses unprojected torchvision ImageNet Inception-v3 pool features (2048D).
These scores are for within-repository comparisons, not published TF-FID.
The reference split, fake seeds, and preprocessing stay fixed across epochs.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from dl_utils.data.celeba import CelebAAlignedDataset, aligned_celeba_transform
from dl_utils.diffusion.lesson_utils import append_record
from dl_utils.gan.quality import polynomial_mmd
from dl_utils.vae.image_quality import (
    FeatureMoments,
    TorchvisionInceptionFeatures,
    frechet_distance,
)


def feature_precision_recall(real, fake, *, neighbors=3, chunk_size=256):
    """k-NN manifold estimates: precision measures fidelity, recall coverage.

    Each ball uses its own manifold center's k-th *other* neighbor radius.
    Chunking only bounds distance-matrix memory; it does not approximate k-NN.
    """
    if min(len(real), len(fake)) <= neighbors:
        raise ValueError("Precision/recall needs more samples than neighbors.")

    def radii(features):
        distances = []
        for start in range(0, len(features), chunk_size):
            block = features[start : start + chunk_size]
            matrix = torch.cdist(block, features)
            matrix[
                torch.arange(len(block)), torch.arange(start, start + len(block))
            ] = float("inf")
            distances.append(matrix.topk(neighbors, largest=False).values[:, -1])
        return torch.cat(distances)

    def coverage(queries, centers, radius):
        inside = []
        for block in queries.split(chunk_size):
            inside.append((torch.cdist(block, centers) <= radius[None]).any(dim=1))
        return torch.cat(inside).float().mean().item()

    return coverage(fake, real, radii(real)), coverage(real, fake, radii(fake))


def feature_metrics(real, fake, *, seed=20260909, kid_subsets=20, neighbors=3):
    """Compute distribution metrics; unbiased KID estimates may be negative."""
    real, fake = real.cpu().float(), fake.cpu().float()
    moments = []
    for features in (real, fake):
        value = FeatureMoments(features.shape[1])
        value.update(features)
        moments.append(value)
    rng = torch.Generator().manual_seed(seed)
    subset_size = min(512, len(real), len(fake))
    kid = torch.tensor(
        [
            polynomial_mmd(
                real[torch.randperm(len(real), generator=rng)[:subset_size]],
                fake[torch.randperm(len(fake), generator=rng)[:subset_size]],
            )
            for _ in range(kid_subsets)
        ],
        dtype=torch.float64,
    )
    precision, recall = feature_precision_recall(real, fake, neighbors=neighbors)
    return {
        "torchvision_fid": frechet_distance(*moments),
        "torchvision_kid_mean": kid.mean().item(),
        "torchvision_kid_subset_std": kid.std(unbiased=False).item(),
        "feature_precision": precision,
        "feature_recall": recall,
        "feature_dim": real.shape[1],
        "kid_subsets": kid_subsets,
        "kid_subset_size": subset_size,
        "manifold_neighbors": neighbors,
    }


class DiffusionQualityMonitor:
    """Evaluate free generation, count actual denoiser calls, and save history.

    ``sample_batch(count, generator)`` returns RGB images in [-1, 1]. For LDM,
    this callback includes decoding, so its latency includes the first stage.
    No pretrained weights or real images are loaded until the first evaluation.
    """

    def __init__(self, args, device, *, feature_extractor=None):
        if args.eval_examples < 4 or args.eval_batch_size < 1:
            raise ValueError(
                "Evaluation requires >=4 examples and a positive batch size."
            )
        self.args, self.device = args, device
        self.features = feature_extractor
        self.real = None

    @torch.inference_mode()
    def evaluate(self, model, sample_batch, *, name, epoch=None, **settings):
        args, device = self.args, self.device
        devices = [device.index or 0] if device.type == "cuda" else []
        was_training = model.training
        calls, image_evaluations = 0, 0

        def count_call(module, inputs):
            nonlocal calls, image_evaluations
            calls += 1
            image_evaluations += len(inputs[0])

        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.eval_seed)
            if self.features is None:
                self.features = TorchvisionInceptionFeatures(projection_dim=None)
            self.features = self.features.to(device).eval().requires_grad_(False)
            if self.real is None:
                dataset = CelebAAlignedDataset(
                    args.data_dir,
                    split="validation",
                    transform=aligned_celeba_transform(args.image_size),
                )
                if args.eval_examples > len(dataset):
                    raise ValueError("Evaluation exceeds the validation split.")
                indices = torch.randperm(
                    len(dataset),
                    generator=torch.Generator().manual_seed(args.eval_seed),
                )[: args.eval_examples].tolist()
                loader = DataLoader(
                    Subset(dataset, indices),
                    batch_size=args.eval_batch_size,
                    num_workers=0,
                )
                self.real = torch.cat(
                    [self.features(images.to(device)).cpu() for images, _ in loader]
                )

            rng = torch.Generator(device=device).manual_seed(args.eval_seed)
            generated, previews = [], []
            elapsed = 0.0
            hook = model.register_forward_pre_hook(count_call)
            model.eval()
            try:
                for start in range(0, args.eval_examples, args.eval_batch_size):
                    count = min(args.eval_batch_size, args.eval_examples - start)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    images = sample_batch(count, rng)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed += time.perf_counter() - begin
                    if images.shape != (count, 3, args.image_size, args.image_size):
                        raise ValueError(
                            "Generated images do not match the image evaluation protocol."
                        )
                    if not torch.isfinite(images).all():
                        raise FloatingPointError(
                            "Generated images contain non-finite values."
                        )
                    generated.append(self.features(images).cpu())
                    if start < 64:
                        previews.append(images[: 64 - start].cpu())
            finally:
                hook.remove()
                model.train(was_training)

            record = {
                **feature_metrics(self.real, torch.cat(generated), seed=args.eval_seed),
                "name": name,
                "epoch": epoch,
                **settings,
                "examples": args.eval_examples,
                "eval_batch_size": args.eval_batch_size,
                "seed": args.eval_seed,
                "split": "CelebA validation",
                "image_size": args.image_size,
                "network_calls": calls,
                "nfe_per_image": image_evaluations / args.eval_examples,
                "sampling_seconds": elapsed,
                "seconds_per_image": elapsed / args.eval_examples,
                "feature_extractor": "torchvision Inception_V3_Weights.IMAGENET1K_V1 pool3",
                "preprocessing": "178px real center crop; bilinear image resize; [-1,1] to [0,1]; antialiased 299px; ImageNet normalization",
                "scope": "within-project monitoring, not published TensorFlow FID/KID; subset std is not a confidence interval",
            }
            append_record(args.output_dir / "quality.jsonl", record)
            suffix = "" if epoch is None else f"_epoch_{epoch:03d}"
            save_image(
                torch.cat(previews).mul(0.5).add(0.5).clamp(0, 1),
                args.output_dir / f"{name}{suffix}.png",
                nrow=8,
            )
            print(
                f"{name}: FID={record['torchvision_fid']:.3f}, "
                f"KID={record['torchvision_kid_mean']:.5f}, "
                f"P/R={record['feature_precision']:.3f}/{record['feature_recall']:.3f}, "
                f"NFE={record['nfe_per_image']:g}"
            )
            return record
