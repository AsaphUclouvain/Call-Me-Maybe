"""Simple helpers to read and write JSON files with basic errors.

Provides thin wrappers around file I/O that raise `ValueError` with
readable messages on failures.
"""

import json
from pathlib import Path
from typing import cast
from src import config


def read_json(file_name: str) -> list | dict:
    """Read JSON from `file_name` and return the parsed object.

    Args:
        file_name: Path to the JSON file to read.

    Returns:
        The decoded JSON object (a list or dict).

    Raises:
        ValueError: If the file cannot be read or parsed.
    """
    path = Path(file_name)
    try:
        text = path.read_text()
        return cast(list | dict, json.loads(text))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(
            f"Impossible to load data from {file_name}"
        ) from e


def write_json(json_data: str) -> None:
    """Write `json_data` to the configured output file path.

    Args:
        json_data: A JSON string to write to disk.

    Returns:
        None

    Raises:
        ValueError: If writing to disk fails.
    """
    path = Path(config.OUTPUT_FILE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_data)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Impossible to write {config.OUTPUT_FILE}") from e
