"""Independent clean-image evaluator and time-conditioned VP guidance classifier."""

import torch
from torch import nn

from dl_utils.diffusion.diffusion_unet import ResidualBlock, SinusoidalTimeEmbedding


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=101, width=64, noisy=False):
        super().__init__()
        self._config = {"num_classes": num_classes, "width": width, "noisy": noisy}
        self.noisy = noisy
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(width), nn.Linear(width, 4 * width), nn.SiLU()
        )
        self.input = nn.Conv2d(3, width, 3, padding=1)
        widths = [width, width * 2, width * 4, width * 4]
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(a, b, 4 * width)
                for a, b in zip([width] + widths[:-1], widths)
            ]
        )
        self.output = nn.Linear(widths[-1], num_classes)

    def config(self):
        return self._config

    def forward(self, images, time=None):
        if self.noisy and time is None:
            raise ValueError("The VP classifier requires continuous noise time.")
        if time is None:
            time = torch.zeros(len(images), device=images.device)
        embedding = self.time(time)
        x = self.input(images)
        for block in self.blocks:
            x = torch.nn.functional.avg_pool2d(block(x, embedding), 2)
        return self.output(x.mean((-2, -1)))
