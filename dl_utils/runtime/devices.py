import torch
from typing import Optional, Union


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the device to use. By default, automatically selects CUDA or CPU based on
    the current environment, or you can explicitly specify it via the argument.

    Args:
        device: Optional device specifier (e.g., "cuda", "cpu", or torch.device).

    Returns:
        torch.device: The selected device.
    """
    resolved = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {resolved} (CUDA available: {torch.cuda.is_available()})")
    # Speedups for modern NVIDIA GPUs (Ampere+ / Ada like RTX 4070 Ti):
    # Enable TF32 (keeps float32 API, uses TF32 internally for matmul/conv where appropriate).
    if resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Let PyTorch choose higher-performance matmul kernels (often TF32 on CUDA).
        torch.set_float32_matmul_precision("high")
    return resolved


def try_gpu(device_index=0):
    """
    Return the requested GPU if it exists, otherwise return the CPU.
    
    Args:
        device_index: the index of the GPU (Default: 0)
    Returns:
        The requested GPU if it exists, otherwise the CPU
    """
    if torch.cuda.device_count() >= device_index + 1:
        return torch.device(f'cuda:{device_index}')
    return torch.device('cpu')


def try_all_gpus():
    """
    Return all available GPUs, or [cpu,] if no GPU exists.
    
    Returns:
        A list of all available GPUs, or [cpu,] if no GPU exists
    """
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
