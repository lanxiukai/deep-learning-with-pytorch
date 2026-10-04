"""Modern sampler assembly with the common fixed-noise monitoring protocol."""

from dl_utils.diffusion.monitoring import monitor_generation as monitor_shared
from dl_utils.modern.sampling import make_sampler


def monitor_generation(
    args, model, metadata, epoch, *, codec=None, latent_scale=1.0, classifier=None
):
    return monitor_shared(
        args,
        model,
        metadata,
        epoch,
        codec=codec,
        latent_scale=latent_scale,
        classifier=classifier,
        sampler_factory=make_sampler,
    )
