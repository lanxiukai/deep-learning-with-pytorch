"""128px continuation, lazy regularization, and path state for StyleGAN2."""

import torch

from dl_utils.gan.continuation import (
    ContinuationContext,
    ContinuationPlan,
)
from dl_utils.gan.continuation import (
    add_refinement_arguments as _add_refinement_arguments,
)

from .model import StyleDiscriminator, StyleGenerator
from .training_config import TrainingEpoch


def add_refinement_arguments(parser):
    """Add continuation flags with StyleGAN2's established defaults."""
    _add_refinement_arguments(
        parser,
        batch_size=16,
        learning_rate=1e-3,
        regularization_help="R1 gamma for StyleGAN2.",
    )


def make_continuation_plan(
    args, *, model_config, discriminator_config, train_epoch, d_reg_every, g_reg_every
) -> ContinuationPlan:
    """Adapt lazy penalties and the running path mean to a bounded chunk."""
    reg_batch_shrink = 2 if args.r1_batch_shrink is None else args.r1_batch_shrink
    path_batch_shrink = 4 if args.path_batch_shrink is None else args.path_batch_shrink

    def train_chunk(context: ContinuationContext, count, state):
        metrics, path_mean, state["global_step"] = train_epoch(
            context.generator,
            context.discriminator,
            context.run.data,
            context.optimizer_g,
            context.optimizer_d,
            context.ema,
            torch.tensor(state["path_mean"], device=context.run.device),
            state["global_step"],
            TrainingEpoch(count, count * context.batch_size, context.batch_size),
            context.batch_size,
            context.run.precision,
            reg_batch_shrink,
            path_batch_shrink,
            r1_gamma=args.refine_reg_weight,
            d_reg_every=d_reg_every,
            g_reg_every=g_reg_every,
        )
        state["path_mean"] = path_mean.item()
        return metrics

    return ContinuationPlan(
        model_name="stylegan2",
        generator_class=StyleGenerator,
        discriminator_class=StyleDiscriminator,
        model_config=dict(model_config),
        discriminator_config=dict(discriminator_config),
        initial_unit="fixed-resolution-epoch-main-metrics-v2",
        initial_completed_units=None,
        legacy_units=(),
        generator_kwargs={"noise_mode": "fixed", "truncation_psi": 1.0},
        d_reg_every=d_reg_every,
        reg_batch_shrink=reg_batch_shrink,
        path_batch_shrink=path_batch_shrink,
        optimizer_ratios={
            "generator": g_reg_every / (g_reg_every + 1),
            "discriminator": d_reg_every / (d_reg_every + 1),
        },
        train_chunk=train_chunk,
    )


__all__ = ["add_refinement_arguments", "make_continuation_plan"]
