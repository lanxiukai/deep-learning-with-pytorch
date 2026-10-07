import os
import shutil
from pathlib import Path


def reset_dir(path: str | os.PathLike[str]) -> None:
    """Replace a file or directory with an empty directory, creating parents.

    Reject symbolic links, filesystem roots, and the current working directory
    or any of its ancestors. Existing directory contents are deleted.
    """
    if not os.fspath(path):
        raise ValueError("reset_dir: path is empty.")

    destination = Path(path)
    if destination.is_symlink():
        raise ValueError(f"reset_dir: refuse to replace a symbolic link: {path!r}")
    resolved = destination.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError(f"reset_dir: refuse to delete root directory: {path!r}")
    working_directory = Path.cwd().resolve()
    if resolved == working_directory or resolved in working_directory.parents:
        raise ValueError(
            f"reset_dir: refuse to delete the working directory or its ancestor: {path!r}"
        )

    if destination.is_dir():
        shutil.rmtree(destination)
    elif destination.exists():
        destination.unlink()
    destination.mkdir(parents=True, exist_ok=True)
