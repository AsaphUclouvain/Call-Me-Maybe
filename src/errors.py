"""Lightweight error taxonomy for prevention and tracking.

Single ``AppError`` carries a machine-readable ``code``, a user-facing
``hint`` and a ``context`` dict. Existing ``ValueError``/``RuntimeError``
are kept internally but wrapped at module boundaries so the CLI can
map them to exit codes and log them uniformly.

Clean handling contract (see also ``src/logger.py``):

* Library / helper layers **never log** before raising. They only
  ``raise AppError(..., code=...) from e`` with ``hint``/``context``.
  Chaining preserves the causal traceback without double logging.
* Only the top-level boundary (``src/cli.py:main``) is allowed to log
  an error and emit a single user-facing line to ``stderr``.
  Diagnostics go to the log file (and to ``stderr`` via the console
  handler); ``stdout`` is reserved for app output.
* User-fixable codes (bad args/files/schema) map to exit 1,
  internal codes (model/inference/decode) map to exit 2.
"""

from __future__ import annotations

from typing import Any

# Fixable by the user (bad args, missing/corrupt files, schema errors).
USER_ERROR_CODES = frozenset(
    {
        "CONFIG_INVALID",
        "INPUT_READ",
        "INPUT_INVALID",
        "FUNC_DEF_READ",
        "FUNC_DEF_INVALID",
        "OUTPUT_WRITE",
    }
)

# Internal / environment failures (model, inference, decoding bugs).
INTERNAL_ERROR_CODES = frozenset(
    {
        "MODEL_LOAD",
        "VOCAB_INVALID",
        "BASE_PROMPT",
        "INFERENCE",
        "DECODE_FAILED",
        "DECODE_NO_TOKEN",
        "DECODE_MAX_ITER",
        "DECODE_FUNC_NOT_FOUND",
        "DECODE_STATE",
    }
)


class AppError(Exception):
    """Base error with code, hint and context for logging/tracking."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        hint: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint
        self.context: dict[str, Any] = dict(context) if context else {}

    def __str__(self) -> str:
        base = super().__str__()
        parts = f"[{self.code}] {base}"
        if self.hint:
            parts += f" Hint: {self.hint}"
        return parts


def exit_code_for(exc: BaseException) -> int:
    """Map an exception to a process exit code.

    Returns:
        0 is never returned here. 1 for user-fixable errors,
        2 for internal errors.
    """
    if isinstance(exc, AppError):
        if exc.code in USER_ERROR_CODES:
            return 1
        return 2
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return 1
    return 2


def is_user_error(exc: BaseException) -> bool:
    """Return True when the error is fixable by the user."""
    return exit_code_for(exc) == 1


# Convenience subclasses for clearer intent at call sites.
# They carry the same ``code``/``hint``/``context`` but let callers
# distinguish user vs internal without inspecting codes.


class UserError(AppError):
    """User-fixable error (maps to exit 1)."""


class InternalError(AppError):
    """Internal/environment error (maps to exit 2)."""


def format_error(exc: BaseException) -> str:
    """Return a concise user-facing string for ``exc``.

    For ``AppError`` this is ``"[CODE] message Hint: ..."``;
    for other exceptions it is ``str(exc)``.
    """
    if isinstance(exc, AppError):
        return str(exc)
    return str(exc)


def log_level_for(exc: BaseException) -> int:
    """Suggest a logging level for ``exc``."""

    import logging

    if is_user_error(exc):
        return logging.WARNING
    return logging.ERROR
