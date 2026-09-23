"""Aligned CelebA readers and explicit CPU/CUDA loading pipelines."""

from .dataset import (
    CELEBA_PARTITIONS,
    CELEBA_SMILING_ATTRIBUTE,
    CELEBA_SMILING_CLASSES,
    CelebAAlignedDataset,
    CelebAEncodedDataset,
)
from .pipeline import (
    CELEBA_ALIGNED_CROP_SIZE,
    CELEBA_PIPELINES,
    CelebATrainingStream,
    aligned_celeba_transform,
    cuda_jpeg_works,
    make_aligned_celeba_loader,
    make_aligned_celeba_train_validation_loaders,
    make_celeba_training_loader,
    make_encoded_celeba_loader,
    prepare_encoded_celeba_batch,
    resolve_celeba_pipeline,
)

__all__ = [
    "CELEBA_ALIGNED_CROP_SIZE",
    "CELEBA_PARTITIONS",
    "CELEBA_PIPELINES",
    "CELEBA_SMILING_ATTRIBUTE",
    "CELEBA_SMILING_CLASSES",
    "CelebAAlignedDataset",
    "CelebAEncodedDataset",
    "CelebATrainingStream",
    "aligned_celeba_transform",
    "cuda_jpeg_works",
    "make_aligned_celeba_loader",
    "make_aligned_celeba_train_validation_loaders",
    "make_celeba_training_loader",
    "make_encoded_celeba_loader",
    "prepare_encoded_celeba_batch",
    "resolve_celeba_pipeline",
]
