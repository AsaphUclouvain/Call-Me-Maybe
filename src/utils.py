"""Small project utilities for function name prefixing.

The codebase assigns short numeric prefixes to function names to
avoid collisions and later strips them when presenting results.

Error contract: helpers raise ``AppError`` without prior logging;
the CLI boundary logs once, keeping stdout clean.
"""

from typing import Any
import logging

from src import config
from src.errors import AppError

logger = logging.getLogger("call-me-maybe.utils")

_func_hash: int = 0

MAX_FUNC_HASH: int = 100


def get_func_hash() -> str:
    """Return the current function-prefix and increment the internal counter.

    Returns:
        A short string used as a numeric prefix for function names.

    Raises:
        AppError: If more than MAX_FUNC_HASH functions are registered.
    """
    global _func_hash

    if _func_hash < 10:
        h = f"0{_func_hash}"
    elif _func_hash < MAX_FUNC_HASH:
        h = f"{_func_hash}"
    else:
        raise AppError(
            "Too many functions; hash above maximum value.",
            code="FUNC_DEF_INVALID",
            hint="Reduce the number of function definitions.",
            context={"count": _func_hash},
        )
    _func_hash += 1
    return h


def add_prefix(funcs: list[dict[str, Any]]) -> None:
    """Prepend a short numeric prefix to each function `name` in `funcs`.

    Args:
        funcs: List of function-definition dictionaries to mutate in-place.

    Returns:
        None
    """
    for f in funcs:
        if "name" not in f or not isinstance(f["name"], str):
            raise AppError(
                "Function entry missing string 'name'.",
                code="FUNC_DEF_INVALID",
                hint="Each function needs a string 'name' field.",
                context={"entry": str(f)[:200]},
            )
        f["name"] = get_func_hash() + f["name"]


def remove_prefix(calls: list[dict[str, Any]]) -> None:
    """Strip the configured prefix length from decoded call names.

    Args:
        calls: List of call dictionaries whose `name` fields will be modified.

    Returns:
        None
    """
    for c in calls:
        name = c.get("name")
        if not isinstance(name, str) or len(name) < config.FUNC_PREFIX_LENGTH:
            logger.debug("Skipping prefix strip for malformed call: %r", c)
            continue
        c["name"] = name[config.FUNC_PREFIX_LENGTH:]
