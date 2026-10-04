"""One PixelCNN training epoch, shared by the VQ-VAE and FSQ lessons."""

import torch.nn.functional as F
from tqdm.auto import tqdm

from dl_utils.training.metrics import MetricAccumulator


def train_pixelcnn_prior_epoch(prior, loader, optimizer, device):
    """Fit the next-token distribution on frozen token grids."""
    prior.train()
    metrics = MetricAccumulator(("nll",), device=device)
    for indices, _ in tqdm(loader, desc="PixelCNN", leave=False):
        indices = indices.to(device)
        loss = F.cross_entropy(prior(indices), indices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics.add_batch_means((loss,), num_examples=len(indices))
    return metrics.compute_weighted_means(require_finite=True)["nll"]
