"""Simple helpers to read and write JSON files with basic errors.

Provides thin wrappers around file I/O that raise `AppError` with
readable messages on failures.  Failures are **not** logged here;
the CLI boundary (``src/cli.py:main``) logs once with full context,
so ``stdout`` stays reserved for app output and the log file does
not contain duplicate entries.
"""

import json
import logging
from pathlib import Path
from typing import Union

from src import config
from src.errors import AppError

logger = logging.getLogger("call-me-maybe.json_handler")

JsonData = Union[list, dict]


def read_json(file_name: str) -> JsonData:
    """Read JSON from `file_name` and return the parsed object.

    Args:
        file_name: Path to the JSON file to read.

    Returns:
        The decoded JSON object (a list or dict).

    Raises:
        AppError: If the file cannot be read or parsed (code INPUT_READ).
    """
    path = Path(file_name)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise AppError(
            f"Impossible to load data from {file_name}: file not found.",
            code="INPUT_READ",
            hint="Check the path and working directory.",
            context={"path": file_name},
        ) from e
    except PermissionError as e:
        raise AppError(
            f"Impossible to load data from {file_name}: permission denied.",
            code="INPUT_READ",
            hint="Check file permissions.",
            context={"path": file_name},
        ) from e
    except OSError as e:
        raise AppError(
            f"Impossible to load data from {file_name}.",
            code="INPUT_READ",
            hint="Check the path and file permissions.",
            context={"path": file_name},
        ) from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise AppError(
            f"Impossible to load data from {file_name}: invalid JSON.",
            code="INPUT_READ",
            hint="Validate the file with a JSON linter.",
            context={"path": file_name, "line": e.lineno, "col": e.colno},
        ) from e
    if isinstance(data, (list, dict)):
        return data
    raise AppError(
        f"Unexpected JSON top-level type in {file_name}.",
        code="INPUT_READ",
        hint="Top level must be a list or dict.",
        context={"path": file_name, "type": type(data).__name__},
    )


def write_json(json_data: str) -> None:
    """Write `json_data` to the configured output file path.

    Args:
        json_data: A JSON string to write to disk.

    Returns:
        None

    Raises:
        AppError: If writing to disk fails (code OUTPUT_WRITE).
    """
    path = Path(config.OUTPUT_FILE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_data, encoding="utf-8")
    except PermissionError as e:
        raise AppError(
            f"Impossible to write {config.OUTPUT_FILE}: permission denied.",
            code="OUTPUT_WRITE",
            hint="Check directory permissions.",
            context={"path": config.OUTPUT_FILE},
        ) from e
    except OSError as e:
        raise AppError(
            f"Impossible to write {config.OUTPUT_FILE}.",
            code="OUTPUT_WRITE",
            hint="Check disk space and directory permissions.",
            context={"path": config.OUTPUT_FILE},
        ) from e
    logger.info(
        "Wrote output to %s (%d chars)", config.OUTPUT_FILE, len(json_data)
    )
