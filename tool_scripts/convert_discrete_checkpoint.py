"""Convert matching legacy tokenizer/prior weights into one final model pair.

This is a one-time weights conversion. It does not convert optimizer history,
resume state, or incompatible model architectures.
"""

import argparse
from pathlib import Path

import torch

from dl_utils.data.datasets.glasses import GLASSES_CLASS_NAMES
from dl_utils.vae.discrete_workflow import MODEL_TYPES, save_pair
from dl_utils.vae.quantization import validate_image_size


def convert_pair(tokenizer_path, prior_path, output_path, name):
    if Path(output_path).resolve() in {
        Path(tokenizer_path).resolve(),
        Path(prior_path).resolve(),
    }:
        raise ValueError("Choose an output path different from the input weights.")
    tokenizer_weights = torch.load(
        tokenizer_path, map_location="cpu", weights_only=True
    )
    prior_weights = torch.load(prior_path, map_location="cpu", weights_only=True)
    expected_prior = (
        f"{name}_transformer_prior" if name == "vqgan" else f"{name}_pixelcnn_prior"
    )
    if (
        tokenizer_weights["model_name"] != f"{name}_tokenizer"
        or prior_weights["model_name"] != expected_prior
    ):
        raise ValueError("Legacy model names do not match the requested pair.")
    if not tokenizer_weights.get("snapshot_id") or tokenizer_weights[
        "snapshot_id"
    ] != prior_weights.get("tokenizer_id"):
        raise ValueError("The prior was not trained on this tokenizer snapshot.")
    classes = list(GLASSES_CLASS_NAMES) if name == "vqgan" else []
    for weights in (tokenizer_weights, prior_weights):
        if (
            weights["dataset"] != "glasses-256"
            or weights["normalization"] != "[-1,1]"
            or weights["class_names"] != classes
        ):
            raise ValueError("Legacy image preprocessing or class names do not match.")
    if tokenizer_weights["image_size"] != prior_weights["image_size"]:
        raise ValueError("Legacy image sizes do not match.")
    config = {
        "image_size": tokenizer_weights["image_size"],
        "tokenizer": tokenizer_weights["model_config"],
        "prior": prior_weights["model_config"],
    }
    tokenizer_type, prior_type = MODEL_TYPES[name]
    tokenizer = tokenizer_type(**config["tokenizer"])
    prior = prior_type(**config["prior"])
    tokenizer.load_state_dict(tokenizer_weights["state_dict"])
    prior.load_state_dict(prior_weights["state_dict"])
    validate_image_size(
        config["image_size"], config["image_size"], tokenizer.downsample_steps
    )
    side = config["image_size"] // (2**tokenizer.downsample_steps)
    if (
        prior.vocabulary_size != tokenizer.quantizer.codebook_size
        or getattr(prior, "sequence_length", side**2) != side**2
        or getattr(prior, "num_classes", 0) != len(classes)
    ):
        raise ValueError("Legacy tokenizer and prior dimensions do not match.")
    save_pair(output_path, name, tokenizer, prior, config)
    return Path(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_TYPES), required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(convert_pair(args.tokenizer, args.prior, args.output, args.model))


if __name__ == "__main__":
    main()
