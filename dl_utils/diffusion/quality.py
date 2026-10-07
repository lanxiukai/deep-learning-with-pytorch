"""Clean-FID features, KID, coverage, independent class compliance and sampling cost."""

import time

import numpy as np
import torch

from dl_utils.diffusion.data import make_loader
from dl_utils.diffusion.lesson_utils import preview
from dl_utils.evaluation.distribution_metrics import feature_precision_recall


class CleanFeatures:
    def __init__(self, device):
        from cleanfid.features import build_feature_extractor
        from cleanfid.resize import build_resizer

        self.device = device
        self.model = build_feature_extractor(
            "clean", device=device, use_dataparallel=False
        )
        self.resize = build_resizer("clean")

    @torch.no_grad()
    def __call__(self, images):
        from cleanfid.fid import get_batch_features

        rgb = (
            images.detach()
            .float()
            .clamp(-1, 1)
            .add(1)
            .mul(127.5)
            .round()
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        resized = torch.stack(
            [torch.from_numpy(self.resize(x).copy()).permute(2, 0, 1) for x in rgb]
        )
        return torch.from_numpy(
            get_batch_features(resized.float(), self.model, self.device)
        )


def frechet_distance(mean_a, covariance_a, mean_b, covariance_b):
    """Clean-FID's Gaussian distance using the current SciPy sqrtm API.

    clean-fid 0.1.35 calls the removed disp=False argument. Keep its feature
    protocol and numerical convention without patching SciPy or site-packages.
    """
    from scipy.linalg import sqrtm

    product_root = sqrtm(covariance_a @ covariance_b)
    if not np.isfinite(product_root).all():
        offset = 1e-6 * np.eye(len(mean_a))
        product_root = sqrtm((covariance_a + offset) @ (covariance_b + offset))
    if np.iscomplexobj(product_root):
        if not np.allclose(product_root.diagonal().imag, 0, atol=1e-3):
            raise ValueError(
                "FID covariance square root has a significant imaginary trace."
            )
        product_root = product_root.real
    difference = mean_a - mean_b
    return float(
        difference @ difference
        + np.trace(covariance_a)
        + np.trace(covariance_b)
        - 2 * np.trace(product_root)
    )


def distribution_scores(real, fake, *, full=False, seed=123, device="cpu"):
    from cleanfid.fid import kernel_distance

    real, fake = real.double().numpy(), fake.double().numpy()
    old_rng = np.random.get_state()
    np.random.seed(seed)
    try:
        scores = {
            "kid": float(
                kernel_distance(
                    real,
                    fake,
                    num_subsets=50,
                    max_subset_size=min(1000, len(real), len(fake)),
                )
            )
        }
    finally:
        np.random.set_state(old_rng)
    if full:
        scores["fid"] = float(
            frechet_distance(
                real.mean(0),
                np.cov(real, rowvar=False),
                fake.mean(0),
                np.cov(fake, rowvar=False),
            )
        )
        precision, recall = feature_precision_recall(
            torch.from_numpy(real).float().to(device),
            torch.from_numpy(fake).float().to(device),
        )
        scores.update(feature_precision=precision, feature_recall=recall)
    return scores


def evaluate_generation(
    args,
    model,
    metadata,
    sample,
    *,
    split="test",
    steps=50,
    solver=None,
    guidance=1.0,
    class_evaluator=None,
    full=True,
    features=None,
    low_sampler=None,
    augmentation_level=0.0,
    count_modules=None,
    perceptual=None,
):
    device = next(model.parameters()).device
    features = features or CleanFeatures(device)
    examples = args.eval_examples
    loader = make_loader(args, split, limit=examples)
    if examples < 4 or examples > len(loader.dataset):
        raise ValueError(
            "Evaluation size must fit the held-out split and contain at least four images."
        )
    task = metadata["algorithm"]["task"]
    paired = task in ("sr3", "sr_regression")
    conditional = metadata["algorithm"].get("conditional", False) or (
        task == "vp" and guidance > 0
    )
    real_features, fake_features, predictions, requested, visual = [], [], [], [], []
    low_predictions = []
    count, seconds = 0, 0.0
    counts = {name: 0 for name in (count_modules or {"generator": model})}
    handles = []
    for name, module in (count_modules or {"generator": model}).items():

        def hook(_module, inputs, key=name):
            counts[key] += len(inputs[0])

        handles.append(module.register_forward_pre_hook(hook))
    reconstruction = {
        "psnr": 0.0,
        "lpips": 0.0,
        "observation_l1": 0.0,
        "bicubic_psnr": 0.0,
    }
    if paired and low_sampler is None and full and perceptual is None:
        raise ValueError(
            "Full paired-image evaluation requires a supplied perceptual metric."
        )
    was_training = model.training
    model.eval()
    try:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            real_features.append(features(images))
            torch.manual_seed(getattr(args, "seed", 123) + count)
            low = None
            if paired:
                from dl_utils.diffusion.data import low_resolution

                low = low_resolution(images)
            if count == 0:
                warm_low = (
                    low_sampler(len(images), labels=labels)
                    if low_sampler is not None
                    else low
                )
                sample(
                    len(images),
                    labels=labels if conditional or paired else None,
                    steps=steps,
                    solver=solver,
                    guidance=guidance,
                    low=warm_low,
                    augmentation_level=augmentation_level,
                )
                for key in counts:
                    counts[key] = 0
                torch.manual_seed(getattr(args, "seed", 123))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            if low_sampler is not None:
                low = low_sampler(len(images), labels=labels)
            generated = sample(
                len(images),
                labels=labels if conditional or paired else None,
                steps=steps,
                solver=solver,
                guidance=guidance,
                low=low,
                augmentation_level=augmentation_level,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds += time.perf_counter() - start
            if generated.shape != images.shape or not torch.isfinite(generated).all():
                raise ValueError("Generated RGB shape/range protocol failed.")
            # Reconstruction and feature metrics evaluate the same exported RGB.
            generated = generated.float().clamp(-1, 1)
            fake_features.append(features(generated))
            if class_evaluator is not None and (conditional or paired):
                with torch.no_grad():
                    predictions.append(
                        class_evaluator(generated.clamp(-1, 1)).argmax(-1).cpu()
                    )
                requested.append(labels.cpu())
                if paired:
                    from dl_utils.diffusion.data import upsample

                    with torch.no_grad():
                        low_predictions.append(
                            class_evaluator(
                                upsample(low, images.shape[-2:]).clamp(-1, 1)
                            )
                            .argmax(-1)
                            .cpu()
                        )
            if paired:
                from dl_utils.diffusion.data import low_resolution

                reconstruction["observation_l1"] += (
                    (low_resolution(generated) - low)
                    .abs()
                    .flatten(1)
                    .mean(1)
                    .sum()
                    .item()
                )
                if low_sampler is None:
                    from dl_utils.diffusion.data import upsample

                    bicubic_mse = (
                        ((upsample(low, images.shape[-2:]).clamp(-1, 1) - images) / 2)
                        .square()
                        .flatten(1)
                        .mean(1)
                    )
                    reconstruction["bicubic_psnr"] += (
                        (-10 * bicubic_mse.clamp_min(1e-12).log10()).sum().item()
                    )
                    mse = (
                        ((generated.clamp(-1, 1) - images) / 2)
                        .square()
                        .flatten(1)
                        .mean(1)
                    )
                    reconstruction["psnr"] += (
                        (-10 * mse.clamp_min(1e-12).log10()).sum().item()
                    )
                    if perceptual is not None:
                        reconstruction["lpips"] += (
                            perceptual(generated, images).sum().item()
                        )
            if paired and count == 0:
                from dl_utils.diffusion.data import upsample

                panels = [upsample(low, images.shape[-2:]), generated]
                if low_sampler is None:
                    panels.append(images)
                preview(
                    args.output_dir / "condition_output_reference.png",
                    torch.stack(panels, 1).flatten(0, 1),
                    nrow=len(panels),
                )
            if count < 64:
                visual.append(generated[: 64 - count].detach().cpu())
            count += len(images)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    real, fake = torch.cat(real_features), torch.cat(fake_features)
    metrics = distribution_scores(real, fake, full=full, device=device)
    metrics.update(
        examples=count,
        split=split,
        image_size=args.image_size,
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        sampling_precision="fp32",
        feature_protocol="clean-fid 0.1.35 clean; RGB uint8 round; PIL bicubic 299",
        sampler=solver,
        steps=steps,
        guidance=guidance,
        network_image_evaluations_per_image={k: v / count for k, v in counts.items()},
        sampling_seconds=seconds,
        seconds_per_image=seconds / count,
        images_per_second=count / seconds,
        batch_size=args.batch_size,
    )
    if paired:
        metrics["condition_source"] = "generated" if low_sampler else "real_downsample"
        metrics["observation_l1"] = reconstruction["observation_l1"] / count
        if low_sampler is None:
            metrics["psnr"] = reconstruction["psnr"] / count
            metrics["bicubic_psnr"] = reconstruction["bicubic_psnr"] / count
            if perceptual is not None:
                metrics["lpips"] = reconstruction["lpips"] / count
    if predictions:
        guessed, labels = torch.cat(predictions), torch.cat(requested)
        correct = guessed == labels
        per_class, diversity, real_diversity, accepted = {}, {}, {}, {}
        for label in range(101):
            mask = labels == label
            per_class[str(label)] = (
                correct[mask].float().mean().item() if mask.any() else None
            )
            valid = fake[mask & correct]
            real_class = real[mask]
            real_diversity[str(label)] = (
                torch.pdist(real_class).mean().item() if len(real_class) > 1 else None
            )
            accepted[str(label)] = len(valid)
            diversity[str(label)] = (
                torch.pdist(valid).mean().item() if len(valid) > 1 else None
            )
        metrics.update(
            class_accuracy=correct.float().mean().item(),
            per_class_accuracy=per_class,
            class_valid_counts=accepted,
            class_pairwise_feature_distance=diversity,
            real_class_pairwise_feature_distance=real_diversity,
        )
    if low_predictions:
        upstream = torch.cat(low_predictions)
        metrics["low_class_accuracy"] = (upstream == labels).float().mean().item()
        metrics["class_preservation_from_low"] = (
            (upstream == guessed).float().mean().item()
        )
    if full:
        measurements = []
        for repeat in range(4):
            label = torch.zeros(1, dtype=torch.long, device=device)
            observation = low[:1] if paired else None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            if low_sampler is not None:
                observation = low_sampler(1, labels=label)
            sample(
                1,
                labels=label if conditional or paired else None,
                steps=steps,
                solver=solver,
                guidance=guidance,
                low=observation,
                augmentation_level=augmentation_level,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if repeat:
                measurements.append(time.perf_counter() - started)
        metrics["single_image_latency_seconds"] = float(np.mean(measurements))
        metrics["single_image_latency_std"] = float(np.std(measurements))
        bank_count = getattr(args, "nearest_examples", 0)
        if bank_count:
            bank_loader = make_loader(
                args, "train", augment=False, shuffle=False, limit=bank_count
            )
            bank = torch.cat([features(x.to(device)) for x, _ in bank_loader])
            shown = torch.cat(visual)[:16]
            distances = torch.cdist(fake[: len(shown)].float(), bank.float())
            indices = distances.argmin(1)
            neighbors = torch.stack([bank_loader.dataset[i.item()][0] for i in indices])
            preview(
                args.output_dir / "nearest_training_examples.png",
                torch.stack((shown, neighbors), 1).flatten(0, 1),
                nrow=2,
            )
            metrics["nearest_reference_examples"] = len(bank)
            metrics["nearest_reference_scope"] = (
                "fixed training subset; feature proximity is not a copying certificate"
            )
    preview(args.output_dir / "samples.png", torch.cat(visual))
    return metrics
