"""Check the local RTX 4070 Ti or a cloud RTX 5080/5090 runtime."""

import argparse

import torch

from dl_utils.runtime.gpu_targets import GPU_TARGETS, resolve_gpu_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=tuple(GPU_TARGETS))
    args = parser.parse_args()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise RuntimeError("A supported CUDA GPU is required.")
    gpu_name = torch.cuda.get_device_name(0)
    target = resolve_gpu_target(gpu_name, args.gpu)
    print(f"GPU name: {gpu_name}")
    print(f"GPU target: {target}")
    bf16 = torch.cuda.is_bf16_supported()
    print(f"BF16 available: {bf16}")
    if not bf16:
        raise RuntimeError("The supported GPU runtime must provide BF16.")


if __name__ == "__main__":
    main()
