"""EDM preconditioning and sigma-space Euler/Heun integration.

Training-noise distribution and objective remain in the EDM lesson.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from dl_utils.diffusion.diffusion_unet import DiffusionUNet


class EDMPreconditioner(nn.Module):
    r"""Turn a base network into the EDM denoiser ``D_theta(x; sigma)``."""

    def __init__(
        self,
        network: DiffusionUNet,
        sigma_data: float = 0.5,
        noise_embedding_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if sigma_data <= 0.0 or noise_embedding_scale <= 0.0:
            raise ValueError("EDM scales must be positive")
        if network.num_classes is not None:
            raise ValueError("this EDM lesson uses an unconditional base network")
        self.network = network
        self.sigma_data = sigma_data
        self.noise_embedding_scale = noise_embedding_scale

    def config(self) -> dict[str, object]:
        return {
            "network_config": self.network.config(),
            "sigma_data": self.sigma_data,
            "noise_embedding_scale": self.noise_embedding_scale,
        }

    def forward(self, noisy: Tensor, sigma: Tensor) -> Tensor:
        if sigma.shape != (noisy.shape[0],):
            raise ValueError("sigma must have shape [batch]")
        sigma_image = sigma.reshape(noisy.shape[0], 1, 1, 1)
        sigma_data = self.sigma_data
        denominator = (sigma_image.square() + sigma_data**2).sqrt()
        c_skip = sigma_data**2 / denominator.square()
        c_out = sigma_image * sigma_data / denominator
        c_in = denominator.reciprocal()
        c_noise = 0.25 * sigma.log()

        # Scaling is part of this U-Net's embedding-coordinate convention; the
        # mathematical conditioning variable remains c_noise = log(sigma) / 4.
        residual = self.network(
            c_in * noisy,
            c_noise * self.noise_embedding_scale,
        )
        return c_skip * noisy + c_out * residual


def edm_noise_grid(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> Tensor:
    """Construct the rho-shaped positive grid and append sigma=0."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if not 0.0 < sigma_min < sigma_max or rho <= 0.0:
        raise ValueError("expected 0 < sigma_min < sigma_max and rho > 0")
    ramp = torch.linspace(0.0, 1.0, num_steps, device=device)
    inverse_rho = 1.0 / rho
    positive = (
        sigma_max**inverse_rho
        + ramp * (sigma_min**inverse_rho - sigma_max**inverse_rho)
    ).pow(rho)
    return torch.cat((positive, positive.new_zeros(1)))


@torch.no_grad()
def sample_edm(
    model: EDMPreconditioner,
    shape: Sequence[int],
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    solver: Literal["euler", "heun"] = "heun",
    initial_noise: Tensor | None = None,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
    trajectory_frames: int = 8,
) -> tuple[Tensor, list[Tensor], int]:
    """Integrate the EDM probability-flow ODE on a decreasing sigma grid."""
    if solver not in ("euler", "heun"):
        raise ValueError(f"unknown solver: {solver}")
    device = next(model.parameters()).device
    sigmas = edm_noise_grid(num_steps, sigma_min, sigma_max, rho, device)
    if initial_noise is None:
        initial_noise = torch.randn(shape, device=device, generator=generator)
    elif tuple(initial_noise.shape) != shape:
        raise ValueError("initial_noise does not match requested shape")
    state = initial_noise.to(device).clone() * sigmas[0]

    trajectory: list[Tensor] = []
    capture_positions: set[int] = set()
    if return_trajectory:
        if trajectory_frames < 2:
            raise ValueError("trajectory_frames must be at least 2")
        trajectory.append(state.detach().cpu())
        capture_count = min(trajectory_frames - 1, num_steps)
        capture_positions = set(
            torch.linspace(0, num_steps - 1, capture_count).round().long().tolist()
        )

    was_training = model.training
    model.eval()
    function_evaluations = 0
    try:
        for index in range(num_steps):
            sigma = sigmas[index]
            sigma_next = sigmas[index + 1]
            sigma_batch = sigma.expand(shape[0])
            denoised = model(state, sigma_batch)
            function_evaluations += 1
            derivative = (state - denoised) / sigma
            step = sigma_next - sigma
            euler = state + step * derivative

            if solver == "heun" and sigma_next > 0:
                next_batch = sigma_next.expand(shape[0])
                denoised_next = model(euler, next_batch)
                function_evaluations += 1
                next_derivative = (euler - denoised_next) / sigma_next
                state = state + 0.5 * step * (derivative + next_derivative)
            else:
                # Never evaluate (x-D)/sigma at the singular sigma=0 endpoint.
                state = euler
            if return_trajectory and index in capture_positions:
                trajectory.append(state.detach().cpu())
    finally:
        model.train(was_training)
    return state.clamp(-1.0, 1.0), trajectory, function_evaluations
