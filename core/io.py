"""Shared atomic file I/O utilities for the SysS pipeline.

All writers follow the same pattern: write to a temp file in the same
directory, then atomically rename into place.  A crash between the write and
the rename leaves a .tmp file behind but never corrupts the target.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _default_file_mode() -> int:
    current_umask = os.umask(0)
    os.umask(current_umask)
    return 0o666 & ~current_umask


def _target_file_mode(path: Path) -> int:
    try:
        return path.stat().st_mode & 0o777
    except OSError:
        return _default_file_mode()


def _replace_preserving_mode(tmp_path: str, path: Path, mode: int) -> None:
    os.chmod(tmp_path, mode)
    Path(tmp_path).replace(path)


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    default=str,
    ensure_ascii: bool = False,
) -> None:
    """Write JSON atomically via a temp file — safe against mid-write crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _target_file_mode(path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=ensure_ascii, default=default)
        _replace_preserving_mode(tmp_path, path, mode)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via a temp file — safe against mid-write crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _target_file_mode(path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as f:
            f.write(text)
        _replace_preserving_mode(tmp_path, path, mode)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically via a temp file — safe against mid-write crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _target_file_mode(path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem
    )
    try:
        with open(tmp_fd, "wb") as f:
            f.write(data)
        _replace_preserving_mode(tmp_path, path, mode)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
