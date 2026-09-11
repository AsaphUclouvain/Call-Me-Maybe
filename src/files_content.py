"""Loading and caching utilities for function definitions and inputs.

This module reads JSON files described in `config` and prepares
Pydantic models used by the decoding pipeline. Cached getters are
provided to avoid repeated I/O.
"""

import json
from typing import Optional, cast
from pydantic import ValidationError
from src import config
from src.utils import add_prefix
from src.json_handler import read_json
from src.schema import FunctionDef, UserInput

_func_def_content: Optional[str] = None
_func_defs: Optional[list[FunctionDef]] = None
_user_inputs: Optional[list[UserInput]] = None


def load_user_input() -> None:
    """Load and validate the user input list from the configured file.

    Returns:
        None

    Raises:
        ValueError: If the file contents are invalid or fail validation.
    """
    global _user_inputs
    try:
        raw_data = read_json(config.INPUT_FILE)
        if not isinstance(raw_data, list):
            raise ValueError(f"le fichier {config.INPUT_FILE} doit \
contenir une liste JSON.")
        _user_inputs = [UserInput(**ui) for ui in raw_data]
    except (ValidationError, ValueError) as e:
        raise ValueError(
                f"Invalid pydantic object {config.INPUT_FILE}: {e}"
            ) from e


def load_func_def() -> None:
    """Load function definitions, apply name prefixes and cache results.

    Returns:
        None

    Raises:
        ValueError: If the function definitions file is invalid or fails
        validation.
    """
    global _func_defs
    global _func_def_content
    try:
        raw_funcs = read_json(config.FUNC_DEF_FILE)
        if not isinstance(raw_funcs, list):
            raise ValueError(f"le fichier {config.FUNC_DEF_FILE} doit \
contenir une liste JSON.")
        add_prefix(raw_funcs)
        _func_def_content = json.dumps(raw_funcs)
        _func_defs = [FunctionDef(**f) for f in raw_funcs]
    except (ValidationError, ValueError) as e:
        raise ValueError(
                f"Invalid pydantic object {config.FUNC_DEF_FILE}: {e}"
            ) from e


def get_func_def_str() -> str:
    """Return the raw JSON string of the function definitions.

    Returns:
        The JSON string containing the function definitions.
    """
    if _func_def_content is None:
        load_func_def()
    return cast(str, _func_def_content)


def get_func_defs() -> list[FunctionDef]:
    """Return the parsed list of `FunctionDef` objects.

    Returns:
        A list of `FunctionDef` instances representing available functions.
    """
    if _func_defs is None:
        load_func_def()
    return cast(list[FunctionDef], _func_defs)


def get_user_inputs() -> list[UserInput]:
    """Return the cached list of `UserInput` objects.

    Returns:
        A list of `UserInput` instances loaded from the configured input file.
    """
    if _user_inputs is None:
        load_user_input()
    return cast(list[UserInput], _user_inputs)
