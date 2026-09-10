"""Generic optimization primitives."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class UpdateRatioSchedule:
    """Plan an exact number of discriminator updates per generator update."""

    discriminator_updates_per_generator: int
    batches_per_epoch: int
    num_epochs: int

    def __post_init__(self):
        if self.discriminator_updates_per_generator < 1:
            raise ValueError(
                "discriminator_updates_per_generator must be positive."
            )
        if self.batches_per_epoch < 1 or self.num_epochs < 1:
            raise ValueError(
                "batches_per_epoch and num_epochs must be positive."
            )
        if self.discriminator_updates_per_generator > self.batches_per_epoch:
            raise ValueError(
                "discriminator_updates_per_generator must not exceed "
                "batches_per_epoch."
            )
        _, incomplete_cycle = divmod(
            self.total_discriminator_updates,
            self.discriminator_updates_per_generator,
        )
        if incomplete_cycle:
            raise ValueError(
                "the training plan does not end on a complete "
                "discriminator-to-generator update cycle."
            )

    @property
    def total_discriminator_updates(self) -> int:
        return self.num_epochs * self.batches_per_epoch

    @property
    def total_generator_updates(self) -> int:
        return (
            self.total_discriminator_updates
            // self.discriminator_updates_per_generator
        )

    def completed_discriminator_updates(self, completed_epochs: int) -> int:
        """Return the expected D step count after complete epochs."""
        if not 0 <= completed_epochs <= self.num_epochs:
            raise ValueError(
                f"completed_epochs must be within [0, {self.num_epochs}]."
            )
        return completed_epochs * self.batches_per_epoch


def ema_decay_for_batch(batch_size: int, half_life_kimg: float) -> float:
    """Convert an image-count half-life into one batch's EMA decay."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if half_life_kimg <= 0:
        raise ValueError("half_life_kimg must be positive.")
    return 0.5 ** (batch_size / (half_life_kimg * 1_000))


@torch.no_grad()
def update_ema(
    target: nn.Module,
    source: nn.Module,
    decay: float,
    *,
    copy_buffers: bool = True,
) -> None:
    """Move target parameters toward source and optionally copy buffers."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be within [0, 1).")
    for target_parameter, source_parameter in zip(
        target.parameters(),
        source.parameters(),
        strict=True,
    ):
        # decay * target_parameter + (1 - decay) * source_parameter
        target_parameter.lerp_(source_parameter, 1 - decay)
    # parameters() contains only trainable parameters optimized during training,
    # whereas buffers() contains non-trainable module state that still affects
    # the forward pass.
    if copy_buffers:
        for target_buffer, source_buffer in zip(
            target.buffers(),
            source.buffers(),
            strict=True,
        ):
            target_buffer.copy_(source_buffer)


def update_ema_by_images(
    target: nn.Module,
    source: nn.Module,
    batch_size: int,
    half_life_kimg: float,
    *,
    copy_buffers: bool = True,
) -> None:
    """Update an EMA using a half-life expressed in thousands of images."""
    update_ema(
        target,
        source,
        ema_decay_for_batch(batch_size, half_life_kimg),
        copy_buffers=copy_buffers,
    )


def sgd(params, lr, batch_size):
    """
    Batch stochastic gradient descent.
    
    Args:
        params: the parameters
        lr: the learning rate
    """
    with torch.no_grad():  # no need to track gradient, just update the parameters
        for param in params:  # traverse the parameters
            param -= lr * param.grad / batch_size  # update the parameter
            param.grad.zero_()                     # reset the gradient


def grad_clipping(net, theta):
    """
    Clip gradients (global norm clipping).
    
    Args:
        net: the network
        theta: the threshold of the gradient
    """
    if isinstance(net, nn.Module):
        params = [parameter for parameter in net.parameters() if parameter.requires_grad]
    else:
        params = net.params
    grads = [parameter.grad for parameter in params if parameter.grad is not None]
    norm = torch.sqrt(sum(
        (torch.sum(grad ** 2) for grad in grads),
        torch.zeros_like(grads[0].sum()),
    ))
    if norm > theta:
        for grad in grads:
            grad[:] *= theta / norm
