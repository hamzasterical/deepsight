import os
from pathlib import Path
from typing import List, Optional, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        logger.warning("Directory does not exist: %s", directory)
        return []

    if recursive:
        files = sorted(directory.rglob("*"))
    else:
        files = sorted(directory.iterdir())

    result = [f for f in files if f.is_file()]
    if extensions is not None:
        ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
        result = [f for f in result if f.suffix.lower() in ext_set]

    return result


def read_image_paths(
    directory: Union[str, Path],
    recursive: bool = False,
    min_file_size_kb: float = 0,
) -> List[Path]:
    image_exts = [".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"]
    files = list_files(directory, extensions=image_exts, recursive=recursive)

    if min_file_size_kb > 0:
        before = len(files)
        files = [f for f in files if f.stat().st_size / 1024 >= min_file_size_kb]
        skipped = before - len(files)
        if skipped:
            logger.info("Skipped %d images below %s KB", skipped, min_file_size_kb)

    return files
