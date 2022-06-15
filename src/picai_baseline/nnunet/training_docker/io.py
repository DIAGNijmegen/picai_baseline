"""
Minimal subset of DIAGNijmegen/msk-tiger/io.py, as required for the nnunet wrapper.
"""

import hashlib
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import (
    Any,
    Union,
)

# We generally except strings and pathlib paths for file and folder names
PathLike = Union[str, Path]

# --------------------------------------------------------------------------------


class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return o.as_posix()
        else:
            return super().default(o)


def read_json(filename: PathLike, *, ordered_dict: bool = True, **kwargs):
    """Reads a json file"""
    if ordered_dict:
        kwargs["object_pairs_hook"] = OrderedDict

    with Path(filename).open() as fp:
        return json.load(fp, **kwargs)


def write_json(
    filename: PathLike,
    data: Any,
    *,
    encoding: str = "UTF-8",
    make_dirs: bool = True,
    **kwargs,
):
    """Dumps an object into a json file, using pretty printing and UTF-8 encoding by default"""
    jsonfile = Path(filename)

    # Ensure that the directory exists
    if make_dirs:
        jsonfile.parent.mkdir(parents=True, exist_ok=True)

    # Write data into json file
    args = {"sort_keys": False, "indent": 2, "cls": JSONEncoder}
    args.update(kwargs)

    with jsonfile.open("w", encoding=encoding) as fp:
        json.dump(data, fp, **args)


# --------------------------------------------------------------------------------


def refresh_file_list(path: PathLike):
    """Update the cached file list of directories on a network share

    On network shares (chansey in particular), files are sometimes reported missing even though they
    exist. This has to do with caching issues and can be fixed by running "ls" or similar commands
    in the parent directory of these files.
    """
    path = Path(path)  # make sure path is a OS-specific path object
    if sys.platform == "win32":
        cmd = ["cmd", "/c", "dir", str(path)]
    else:
        cmd = ["ls", str(path)]

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise OSError(f'Could not refresh file list of directory "{path}"') from e


def path_exists(path: PathLike) -> bool:
    """Checks whether the file or directory exists

    Unlike checks via os.path or pathlib, this check works reliably also on network shares where the
    content of directories might be cached.
    """
    path = Path(path)

    # Refresh content of parent (folder in which the object of interest is stored)
    try:
        refresh_file_list(path.parent)
    except OSError:
        # If the parent directory does not exist, refreshing the file list will fail, but that's okay
        pass

    # Now we can check if the specific file/directory exists
    if path.exists():
        # If the object was a directory itself, we also ask for a fresh list of it's content
        if path.is_dir() and path != path.parent:
            refresh_file_list(path)
        return True
    else:
        return False


def checksum(file: PathLike, algorithm: str = "sha256", chunk_size: int = 4096) -> str:
    """Computes the checksum of a file using a hashing algorithm"""
    file = Path(file)
    if file.exists() and not file.is_file():
        raise ValueError(f"Checksum can be computed only for files, {file} is not a file")

    h = hashlib.new(algorithm)
    with file.open("rb") as fp:
        for chunk in iter(lambda: fp.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
