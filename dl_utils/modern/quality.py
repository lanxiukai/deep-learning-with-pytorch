"""Supply the modern reconstruction metric to the shared generation evaluator."""

from dl_utils.diffusion.quality import evaluate_generation as evaluate_shared
from dl_utils.modern.kl_autoencoder import PerceptualLoss


def evaluate_generation(args, model, metadata, sample, **options):
    paired = metadata["algorithm"]["task"] in ("sr3", "sr_regression")
    if (
        paired
        and options.get("full", True)
        and options.get("low_sampler") is None
        and options.get("perceptual") is None
    ):
        options["perceptual"] = PerceptualLoss().to(next(model.parameters()).device)
    return evaluate_shared(args, model, metadata, sample, **options)
