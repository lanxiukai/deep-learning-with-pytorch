"""Evaluate the VP guidance classifier separately in four noise strata."""

from dl_utils.diffusion.classifier_evaluation import main

if __name__ == "__main__":
    main(noisy=True)
