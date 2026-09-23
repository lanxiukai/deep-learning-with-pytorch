from pathlib import Path


def infer_project_root() -> Path:
    if '__file__' in globals():
        return Path(__file__).resolve().parent.parent.parent
    return Path.cwd().resolve()
