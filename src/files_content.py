"""Loading and caching utilities for function definitions and inputs.

This module reads JSON files described in `config` and prepares
Pydantic models used by the decoding pipeline. Cached getters are
provided to avoid repeated I/O.

Error contract: validation failures raise ``AppError`` without logging;
only the CLI boundary logs, preserving ``stdout`` and avoiding
duplicate log entries.
"""

import json
import logging
from typing import Optional, cast

from pydantic import ValidationError

from src import config
from src.errors import AppError
from src.json_handler import read_json
from src.schema import FunctionDef, UserInput
from src.utils import add_prefix

logger = logging.getLogger("call-me-maybe.files_content")

_func_def_content: Optional[str] = None
_func_defs: Optional[list[FunctionDef]] = None
_user_inputs: Optional[list[UserInput]] = None


def load_user_input() -> None:
    """Load and validate the user input list from the configured file.

    Returns:
        None

    Raises:
        AppError: If the file contents are invalid or fail validation.
    """
    global _user_inputs
    try:
        raw_data = read_json(config.INPUT_FILE)
    except AppError as e:
        e.context.setdefault("file", config.INPUT_FILE)
        if e.code == "INPUT_READ":
            raise AppError(
                str(e),
                code="INPUT_READ",
                hint=e.hint,
                context=e.context,
            ) from e
        raise
    try:
        if not isinstance(raw_data, list):
            raise AppError(
                f"File {config.INPUT_FILE} must contain a JSON list.",
                code="INPUT_INVALID",
                hint="Wrap the prompts in a top-level [...].",
                context={"path": config.INPUT_FILE},
            )
        if not raw_data:
            raise AppError(
                f"File {config.INPUT_FILE} contains an empty list.",
                code="INPUT_INVALID",
                hint="Add at least one {\"prompt\": ...} entry.",
                context={"path": config.INPUT_FILE},
            )
        _user_inputs = [UserInput(**ui) for ui in raw_data]
    except (ValidationError, TypeError) as e:
        raise AppError(
            f"Invalid user input schema in {config.INPUT_FILE}: {e}",
            code="INPUT_INVALID",
            hint="Each entry must match {\"prompt\": str}.",
            context={"path": config.INPUT_FILE},
        ) from e
    logger.info(
        "Loaded %d user input(s) from %s",
        len(_user_inputs),
        config.INPUT_FILE,
    )


def load_func_def() -> None:
    """Load function definitions, apply name prefixes and cache results.

    Returns:
        None

    Raises:
        AppError: If the function definitions file is invalid or fails
        validation.
    """
    global _func_defs
    global _func_def_content
    try:
        raw_funcs = read_json(config.FUNC_DEF_FILE)
    except AppError as e:
        raise AppError(
            str(e),
            code="FUNC_DEF_READ",
            hint=e.hint,
            context={**e.context, "file": config.FUNC_DEF_FILE},
        ) from e
    try:
        if not isinstance(raw_funcs, list):
            raise AppError(
                f"File {config.FUNC_DEF_FILE} must contain a JSON list.",
                code="FUNC_DEF_INVALID",
                hint="Wrap the function definitions in a top-level [...].",
                context={"path": config.FUNC_DEF_FILE},
            )
        if not raw_funcs:
            raise AppError(
                f"File {config.FUNC_DEF_FILE} contains an empty list.",
                code="FUNC_DEF_INVALID",
                hint="Define at least one function.",
                context={"path": config.FUNC_DEF_FILE},
            )
        add_prefix(raw_funcs)
        _func_def_content = json.dumps(raw_funcs)
        _func_defs = [FunctionDef(**f) for f in raw_funcs]
    except (ValidationError, TypeError) as e:
        raise AppError(
            f"Invalid schema in {config.FUNC_DEF_FILE}: {e}",
            code="FUNC_DEF_INVALID",
            hint="Each entry needs name/description/parameters/returns.",
            context={"path": config.FUNC_DEF_FILE},
        ) from e
    logger.info("Loaded %d function definition(s)", len(_func_defs))


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
