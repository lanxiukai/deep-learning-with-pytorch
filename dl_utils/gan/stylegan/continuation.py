"""128px continuation settings and training adaptation for stylegan."""

from dl_utils.gan.continuation import (
    ContinuationContext,
    ContinuationPlan,
)
from dl_utils.gan.continuation import (
    add_refinement_arguments as _add_refinement_arguments,
)
from dl_utils.gan.progressive import ProgressivePhase

from .model import StyleGANDiscriminator, StyleGANGenerator


def add_refinement_arguments(parser):
    """Add continuation flags with the model's established defaults."""
    _add_refinement_arguments(
        parser,
        batch_size=32,
        learning_rate=1e-3,
        regularization_help="R1 gamma for StyleGAN.",
    )


def make_continuation_plan(
    args, *, model_config, discriminator_config, train_phase, d_reg_every
) -> ContinuationPlan:
    """Adapt the lesson's phase trainer without importing the lesson itself."""
    d_reg_every = d_reg_every if args.d_reg_every is None else args.d_reg_every
    reg_batch_shrink = 2 if args.reg_batch_shrink is None else args.reg_batch_shrink

    def train_chunk(context: ContinuationContext, count, state):
        metrics, state["global_step"] = train_phase(
            context.generator,
            context.discriminator,
            context.run.data,
            context.optimizer_g,
            context.optimizer_d,
            context.ema,
            ProgressivePhase(128, "stabilization", context.batch_size, count),
            context.run.precision,
            state["global_step"],
            d_reg_every,
            reg_batch_shrink,
            r1_gamma=args.refine_reg_weight,
        )
        return metrics

    return ContinuationPlan(
        model_name="stylegan",
        generator_class=StyleGANGenerator,
        discriminator_class=StyleGANDiscriminator,
        model_config=dict(model_config),
        discriminator_config=dict(discriminator_config),
        initial_unit="progressive-phase-main-metrics-v2",
        initial_completed_units=11,
        legacy_units=(),
        generator_kwargs={"noise_mode": "fixed", "truncation_psi": 1.0},
        d_reg_every=d_reg_every,
        reg_batch_shrink=reg_batch_shrink,
        path_batch_shrink=4,
        optimizer_ratios={"generator": 1.0, "discriminator": 1.0},
        train_chunk=train_chunk,
    )


__all__ = ["add_refinement_arguments", "make_continuation_plan"]
