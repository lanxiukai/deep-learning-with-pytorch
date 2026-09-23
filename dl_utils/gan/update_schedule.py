"""Discriminator-to-generator update schedules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UpdateRatioSchedule:
    """Plan an exact number of discriminator updates per generator update."""

    discriminator_updates_per_generator: int
    batches_per_epoch: int
    num_epochs: int

    def __post_init__(self):
        if self.discriminator_updates_per_generator < 1:
            raise ValueError("discriminator_updates_per_generator must be positive.")
        if self.batches_per_epoch < 1 or self.num_epochs < 1:
            raise ValueError("batches_per_epoch and num_epochs must be positive.")
        if self.discriminator_updates_per_generator > self.batches_per_epoch:
            raise ValueError(
                "discriminator_updates_per_generator must not exceed batches_per_epoch."
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
            self.total_discriminator_updates // self.discriminator_updates_per_generator
        )

    def completed_discriminator_updates(self, completed_epochs: int) -> int:
        """Return the expected D step count after complete epochs."""
        if not 0 <= completed_epochs <= self.num_epochs:
            raise ValueError(f"completed_epochs must be within [0, {self.num_epochs}].")
        return completed_epochs * self.batches_per_epoch


__all__ = ["UpdateRatioSchedule"]
