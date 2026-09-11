"""Small project utilities for function name prefixing.

The codebase assigns short numeric prefixes to function names to
avoid collisions and later strips them when presenting results.
"""

from src import config

_func_hash: int = 0


def get_func_hash() -> str:
    """Return the current function-prefix and increment the internal counter.

    Returns:
        A short string used as a numeric prefix for function names.
    """
    global _func_hash

    if _func_hash < 10:
        h = f"0{_func_hash}"
    elif _func_hash < 100:
        h = f"{_func_hash}"
    else:
        raise ValueError("Hash above maximum value.")
    _func_hash += 1
    return h


def add_prefix(funcs: list[dict]) -> None:
    """Prepend a short numeric prefix to each function `name` in `funcs`.

    Args:
        funcs: List of function-definition dictionaries to mutate in-place.

    Returns:
        None
    """
    for f in funcs:
        f["name"] = get_func_hash() + f["name"]


def remove_prefix(calls: list[dict]) -> None:
    """Strip the configured prefix length from decoded call names.

    Args:
        calls: List of call dictionaries whose `name` fields will be modified.

    Returns:
        None
    """
    for c in calls:
        c["name"] = c["name"][config.FUNC_PREFIX_LENGTH:]
