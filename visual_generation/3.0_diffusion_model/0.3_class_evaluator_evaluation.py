"""Evaluate the independent clean-image classifier on held-out Food-101."""

from dl_utils.diffusion.classifier_evaluation import main

if __name__ == "__main__":
    main(noisy=False)
