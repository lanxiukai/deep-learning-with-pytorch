"""SR3 continuous power conditioning and the later CDM augmentation increment."""

import torch
from torch import nn

from dl_utils.diffusion.data import upsample
from dl_utils.diffusion.diffusion_unet import DiffusionUNet


class SR3(nn.Module):
    def __init__(
        self,
        image_size=256,
        hidden_dims=(64, 128, 256, 384),
        augmentation=False,
        class_conditional=False,
    ):
        super().__init__()
        self._config = {
            "image_size": image_size,
            "hidden_dims": list(hidden_dims),
            "augmentation": augmentation,
            "class_conditional": class_conditional,
        }
        self.augmentation, self.class_conditional = augmentation, class_conditional
        self.network = DiffusionUNet(
            image_size=image_size,
            in_channels=6,
            out_channels=3,
            hidden_dims=hidden_dims,
            residual_resampling=True,
            num_classes=101 if class_conditional else None,
            extra_noise_condition=augmentation,
        )

    def config(self):
        return self._config

    def forward(self, noisy, power, low, labels=None, augmentation_level=None):
        observation = upsample(low, noisy.shape[-2:])
        return self.network(
            torch.cat((noisy, observation), 1),
            power * 1000,
            labels if self.class_conditional else None,
            extra_time=augmentation_level * 1000 if self.augmentation else None,
        )


def augment_condition(low, strength):
    s = strength[:, None, None, None]
    return (1 - s.square()).sqrt() * low + s * torch.randn_like(low)


@torch.no_grad()
def sample_sr3(
    model, diffusion, noise, low, labels=None, *, steps=50, augmentation_level=0.0
):
    if steps < 2 or steps > diffusion.num_steps:
        raise ValueError("Invalid SR3 sampling grid.")
    state = noise.clone()
    level = torch.full((len(low),), augmentation_level, device=low.device)
    if model.augmentation:
        low = augment_condition(low, level)
    indices = diffusion.inference_timesteps(steps)
    powers = torch.cat(
        (diffusion.alpha_bars[indices], torch.ones(1, device=noise.device))
    )
    for i in range(steps):
        power, next_power = powers[i], powers[i + 1]
        epsilon = model(state, power.expand(len(noise)), low, labels, level)
        alpha = power / next_power
        beta = 1 - alpha
        mean = (state - beta / (1 - power).sqrt() * epsilon) / alpha.sqrt()
        state = mean + beta.sqrt() * torch.randn_like(state) if i + 1 < steps else mean
    return state
