"""GPU targets supported by the repository's runtime and training tools."""

GPU_TARGETS = {
    "4070ti": "RTX 4070 Ti",
    "5080": "RTX 5080",
    "5090": "RTX 5090",
}


def resolve_gpu_target(gpu_name: str, expected: str | None = None) -> str:
    """Match an exact model name and optionally require a selected target."""
    name = " ".join(gpu_name.upper().split())
    name = name.removeprefix("NVIDIA ").removeprefix("GEFORCE ")
    for target, label in GPU_TARGETS.items():
        if name.replace(" ", "") == label.upper().replace(" ", ""):
            if expected is not None and target != expected:
                raise ValueError(f"Expected GPU target {expected!r}, found {gpu_name!r}.")
            return target
    raise ValueError(
        f"Unsupported GPU {gpu_name!r}; supported models: {', '.join(GPU_TARGETS.values())}."
    )
