"""Modern visual-generation evaluation entry points and model assembly."""

from dl_utils.diffusion.evaluation import evaluate as evaluate_shared
from dl_utils.diffusion.evaluation import parser_for
from dl_utils.modern.checkpoints import (
    codec_for_generation,
    load_model,
)
from dl_utils.modern.quality import evaluate_generation
from dl_utils.modern.sampling import make_sampler

SOLVERS = {
    "ldm": ["ddim", "dpmpp_2m"],
    "dit": ["ddpm", "ddim"],
    "sit": ["euler", "heun", "sde"],
    "edm": ["euler", "heun", "dpmpp_2m"],
    "sr3": ["ancestral"],
    "sr_regression": ["regression"],
    "consistency": ["consistency"],
    "dmd2": ["student"],
    "imf": ["interval"],
}


def evaluate(args, expected_tasks=None):
    return evaluate_shared(
        args,
        expected_tasks,
        load_checkpoint=load_model,
        codec_loader=codec_for_generation,
        sampler_factory=make_sampler,
        solver_choices=SOLVERS,
        quality_evaluator=evaluate_generation,
    )


def main(default_name, expected_tasks):
    evaluate(parser_for(default_name, modern=True).parse_args(), expected_tasks)
