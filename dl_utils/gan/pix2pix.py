"""Data and model definitions shared by the pix2pix lesson and evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

import albumentations
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import ImageOps
from torch import Tensor, nn
from torch.utils.data import Dataset

from dl_utils.data.images import load_rgb_image

PARTITIONS = {"train": 0, "validation": 1, "test": 2}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_paired_transform(
    training: bool,
    image_size: int = 128,
    load_size: int = 143,
):
    """Apply identical geometry to both images in each aligned pair."""
    if training:
        transforms = [
            albumentations.Resize(height=load_size, width=load_size),
            albumentations.RandomCrop(height=image_size, width=image_size),
            albumentations.HorizontalFlip(p=0.5),
        ]
    else:
        transforms = [
            albumentations.Resize(height=image_size, width=image_size),
        ]
    transforms.extend(
        [
            albumentations.Normalize(
                mean=[0.5] * 3,
                std=[0.5] * 3,
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ]
    )
    # The same random crop and flip are mandatory for paired supervision.
    return albumentations.Compose(
        transforms,
        additional_targets={"source": "image"},
    )


class CelebAColorizationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Create aligned grayscale-source/RGB-target pairs from CelebA images."""

    def __init__(self, celeba_dir, split, transform):
        super().__init__()
        celeba_dir = Path(celeba_dir)
        image_dirs = [celeba_dir / "black", celeba_dir / "blond"]
        partition_path = celeba_dir / "list_eval_partition.csv"
        missing = [path for path in [*image_dirs, partition_path] if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing CelebA data: {missing}")
        if split not in PARTITIONS:
            raise ValueError(f"unknown CelebA split: {split}")

        paths_by_name = {
            path.name: path
            for image_dir in image_dirs
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        with partition_path.open(newline="", encoding="utf-8") as stream:
            self.image_paths = [
                paths_by_name[row["image_id"]]
                for row in csv.DictReader(stream)
                if int(row["partition"]) == PARTITIONS[split]
                and row["image_id"] in paths_by_name
            ]
        if not self.image_paths:
            raise ValueError(f"no CelebA images found for split {split}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        target = load_rgb_image(self.image_paths[index])
        # Derive the grayscale input from the color target on the fly so the
        # pair stays pixel-aligned, then convert back to RGB (replicated
        # luminance) to match the generator's expected input channels.
        source = ImageOps.grayscale(target).convert("RGB")
        transformed_pair = self.transform(
            image=np.array(target),
            source=np.array(source),
        )
        return transformed_pair["source"], transformed_pair["image"]


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs, skip_features):
        return torch.cat([self.net(inputs), skip_features], dim=1)


class UNetGenerator(nn.Module):
    """Seven-level U-Net for 128x128 pix2pix generation."""

    def __init__(self, input_channels=3, output_channels=3, base_channels=64):
        super().__init__()
        self.down1 = UNetDownBlock(
            input_channels,
            base_channels,
            normalize=False,
        )  # (B, 64, 64, 64)
        self.down2 = UNetDownBlock(
            base_channels,
            base_channels * 2,
        )  # (B, 128, 32, 32)
        self.down3 = UNetDownBlock(
            base_channels * 2,
            base_channels * 4,
        )  # (B, 256, 16, 16)
        self.down4 = UNetDownBlock(
            base_channels * 4,
            base_channels * 8,
        )  # (B, 512, 8, 8)
        self.down5 = UNetDownBlock(
            base_channels * 8,
            base_channels * 8,
        )  # (B, 512, 4, 4)
        self.down6 = UNetDownBlock(
            base_channels * 8,
            base_channels * 8,
        )  # (B, 512, 2, 2)
        # PyTorch BatchNorm cannot train on a 1x1 tensor with batch size 1.
        self.down7 = UNetDownBlock(
            base_channels * 8,
            base_channels * 8,
            normalize=False,
        )  # (B, 512, 1, 1)

        self.up1 = UNetUpBlock(
            base_channels * 8,
            base_channels * 8,
            dropout=True,
        )  # (B, 1024, 2, 2)
        self.up2 = UNetUpBlock(
            base_channels * 16,
            base_channels * 8,
            dropout=True,
        )  # (B, 1024, 4, 4)
        self.up3 = UNetUpBlock(
            base_channels * 16,
            base_channels * 8,
            dropout=True,
        )  # (B, 1024, 8, 8)
        self.up4 = UNetUpBlock(
            base_channels * 16,
            base_channels * 4,
        )  # (B, 512, 16, 16)
        self.up5 = UNetUpBlock(
            base_channels * 8,
            base_channels * 2,
        )  # (B, 256, 32, 32)
        self.up6 = UNetUpBlock(
            base_channels * 4,
            base_channels,
        )  # (B, 128, 64, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, output_channels, 4, 2, 1),
            nn.Tanh(),
        )  # (B, 3, 128, 128)

    def forward(self, inputs):
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down6 = self.down6(down5)
        bottleneck = self.down7(down6)

        up1 = self.up1(bottleneck, down6)
        up2 = self.up2(up1, down5)
        up3 = self.up3(up2, down4)
        up4 = self.up4(up3, down3)
        up5 = self.up5(up4, down2)
        up6 = self.up6(up5, down1)
        return self.final(up6)


class ConditionalPatchDiscriminator(nn.Module):
    """70x70 PatchGAN producing a 14x14 map for 128x128 inputs."""

    def __init__(self, source_channels=3, target_channels=3, base_channels=64):
        super().__init__()

        def make_block(in_channels, out_channels, stride, normalize=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=not normalize)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *make_block(
                source_channels + target_channels,
                base_channels,
                2,
                normalize=False,
            ),  # (B, 64, 64, 64)
            *make_block(base_channels, base_channels * 2, 2),
            *make_block(base_channels * 2, base_channels * 4, 2),
            *make_block(base_channels * 4, base_channels * 8, 1),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1),
        )

    def forward(self, source, target):
        return self.net(torch.cat([source, target], dim=1))


def initialize_pix2pix_weights(module):
    """Use the normal initialization from the standard pix2pix implementation."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def denormalize_images(images):
    return images.mul(0.5).add(0.5).clamp(0, 1)
