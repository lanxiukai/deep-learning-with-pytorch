"""Dataset download and archive extraction primitives."""

import hashlib
import hmac
import os
import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import requests

from dl_utils.data.datasets.catalog import DATA_HUB, DATA_URL
from dl_utils.data.datasets.catalog import DOWNLOAD_SUBDIRECTORIES as _DOWNLOAD_SUBDIR
from dl_utils.filesystem.project_root import infer_project_root

DOWNLOAD_TIMEOUT = (10, 60)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def _sha1sum(path: str) -> str:
    """Return the SHA-1 digest used by the upstream D2L data registry."""
    digest = hashlib.sha1()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name, cache_dir=None, *, data_root=None):
    """
    Download a file inserted into DATA_HUB, and return the local filename.

    Args:
        name: the name of the file to download
        cache_dir: the exact folder to cache the file (Default: None)
        data_root: the shared dataset root under which the registered
            subdirectory is selected (Default: <project_root>/data)
    Returns:
        the path to the downloaded file
    """
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    parsed_url = urlsplit(url)
    if parsed_url.scheme.lower() != "https":
        raise ValueError(f"Refusing non-HTTPS dataset URL for {name!r}")
    if cache_dir is not None and data_root is not None:
        raise ValueError("cache_dir and data_root are mutually exclusive")
    if cache_dir is None:
        if data_root is None:
            data_root = infer_project_root() / "data"
        cache_dir = os.fspath(data_root)
        subdir = _DOWNLOAD_SUBDIR.get(name)
        if subdir:
            cache_dir = os.path.join(cache_dir, subdir)
    cache_dir = os.fspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        raise ValueError(f"Dataset URL has no filename for {name!r}")
    fname = os.path.join(cache_dir, filename)
    if os.path.exists(fname) and hmac.compare_digest(_sha1sum(fname), sha1_hash):
        return fname  # cache hit

    print(f"Downloading {fname} from {url}...")
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{filename}.",
        suffix=".part",
        dir=cache_dir,
    )
    os.close(file_descriptor)
    try:
        digest = hashlib.sha1()
        with (
            requests.get(
                url,
                stream=True,
                timeout=DOWNLOAD_TIMEOUT,
                verify=True,
            ) as response,
            open(temporary_path, "wb") as stream,
        ):
            response.raise_for_status()
            for chunk in response.iter_content(DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                stream.write(chunk)
                digest.update(chunk)

        actual_sha1 = digest.hexdigest()
        if not hmac.compare_digest(actual_sha1, sha1_hash):
            raise RuntimeError(f"Checksum mismatch for downloaded dataset {name!r}")
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, fname)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return fname


def download_extract(name, folder=None, *, data_root=None):
    """
    Download and extract a zip/tar file.

    Args:
        name: the name of the file to download
        folder: the folder to extract the file to (Default: None)
        data_root: the shared dataset root passed to ``download``
    Returns:
        the path to the extracted file
    """
    fname = download(name, data_root=data_root)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        with zipfile.ZipFile(fname, "r") as fp:
            archive_entries = fp.namelist()
            top_dirs = {
                entry.strip("/").split("/", 1)[0]
                for entry in archive_entries
                if entry.strip("/")
            }
            if _is_flattened_archive_complete(
                base_dir, data_dir, archive_entries, top_dirs
            ):
                data_dir = base_dir
            else:
                fp.extractall(base_dir)
    elif ext in (".tar", ".gz"):
        with tarfile.open(fname, "r") as fp:
            archive_entries = fp.getnames()
            top_dirs = {
                entry.strip("/").split("/", 1)[0]
                for entry in archive_entries
                if entry.strip("/")
            }
            if _is_flattened_archive_complete(
                base_dir, data_dir, archive_entries, top_dirs
            ):
                data_dir = base_dir
            else:
                fp.extractall(base_dir)
    else:
        assert False, "only zip/tar files can be extracted"

    # Flatten archives that contain a single top-level directory
    # (e.g. fra-eng.zip → fra-eng/fra.txt → avoid fra_eng/fra-eng/fra.txt)
    if os.path.isdir(data_dir):
        if len(top_dirs) == 1:
            inner_dir = os.path.join(base_dir, top_dirs.pop())
            if os.path.isdir(inner_dir) and inner_dir == data_dir:
                for item in os.listdir(inner_dir):
                    shutil.move(
                        os.path.join(inner_dir, item), os.path.join(base_dir, item)
                    )
                os.rmdir(inner_dir)
                data_dir = base_dir  # flattened — point to parent

    return os.path.join(data_dir, folder) if folder else data_dir


def _is_flattened_archive_complete(base_dir, data_dir, entries, top_dirs):
    """Return whether a single-root archive is already flattened in base_dir."""
    if len(top_dirs) != 1:
        return False
    top_dir = next(iter(top_dirs))
    if os.path.join(base_dir, top_dir) != data_dir:
        return False

    prefix = f"{top_dir}/"
    top_level_items = {
        entry.strip("/")[len(prefix) :].split("/", 1)[0]
        for entry in entries
        if entry.strip("/").startswith(prefix) and entry.strip("/")[len(prefix) :]
    }
    return bool(top_level_items) and all(
        os.path.exists(os.path.join(base_dir, item)) for item in top_level_items
    )


def download_kaggle_dataset(
    owner_dataset: str,
    destination: Path,
    *,
    is_ready: Callable[[Path], bool] | None = None,
) -> bool:
    """Download a Kaggle dataset unless its destination is already ready."""
    destination = Path(destination)
    populated = destination.is_dir() and any(destination.iterdir())
    if populated and (is_ready is None or is_ready(destination)):
        print(f"  Already exists: {destination}")
        return True
    if populated:
        print(f"  Existing download is incomplete; retrying: {destination}")

    try:
        import kagglehub
    except ImportError:
        print(
            "  SKIP: kagglehub not installed. "
            "Install via: uv sync --no-dev --extra celeba"
        )
        return False

    kagglehub.dataset_download(
        owner_dataset,
        output_dir=os.fspath(destination),
    )
    if not destination.is_dir() or not any(destination.iterdir()):
        raise RuntimeError(f"Kaggle download produced no files under {destination}")
    if is_ready is not None and not is_ready(destination):
        raise RuntimeError(f"Kaggle download is incomplete under {destination}")
    print(f"  Downloaded to: {destination}")
    return True


__all__ = [
    "DATA_HUB",
    "DATA_URL",
    "download",
    "download_extract",
    "download_kaggle_dataset",
]
