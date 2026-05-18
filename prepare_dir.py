"""
Utility script containing function for preparation of folders for parallel processing.
"""

from pathlib import Path


def prepare_dir(dirpath: str | Path) -> Path:
    """
    Ensure that `dirpath` exists and is empty.

    - If the directory does not exist, create it.
    - If it exists but is not empty, raise an exception.
    - If it exists and is empty, do nothing.

    Returns
    -------
    Path
        Path object corresponding to the directory.

    Raises
    ------
    FileExistsError
        If the directory exists and is not empty.
    NotADirectoryError
        If the path exists but is not a directory.
    """

    path = Path(dirpath)

    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"'{path}' exists but is not a directory.")

        # Check whether directory is empty
        if any(path.iterdir()):
            raise FileExistsError(
                f"Directory '{path}' is not empty. "
                "Please check and remove its contents manually if appropriate."
            )
    else:
        path.mkdir(parents=True, exist_ok=True)

    return path
