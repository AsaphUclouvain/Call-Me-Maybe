"""Central logging setup for the project.

Single file output by default (no rotation history: the file is
truncated on each run). Per-prompt tracking is supported via the
``prompt`` record attribute.

Clean handling contract (see also ``src/errors.py``):

* All diagnostics go to the **log file**.  ``sys.stdout`` is reserved
  for project ``print()`` output and stays free from logs.
* Library code **must not** ``logger.error``/``logger.warning`` before
  raising ``AppError``; it should only ``raise AppError(...) from e``
  and let ``src/cli.py:main`` log once at the boundary.  This avoids
  double logging (one at the raise site and one at the catch site)
  and avoids log-vs-``print`` duplication.
* The file handler is opened with ``mode="w"`` so each run truncates
  the previous log; no rotation history is kept.
"""

from __future__ import annotations

import logging
import sys
from logging import Logger
from pathlib import Path
from typing import Any


_PROMPT_DEFAULT = "-"

_FORMAT = "%(asctime)s %(levelname)s %(name)s [prompt=%(prompt)s] %(message)s"


class _PromptFilter(logging.Filter):
    """Ensure every record has a ``prompt`` attribute for the formatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "prompt"):
            setattr(record, "prompt", _PROMPT_DEFAULT)
        return True


_configured = False


def setup_logging(
    verbose: bool = False,
    trace: bool = False,
    log_file: str | Path = Path("logs/app.log"),
    console: bool = False,
) -> Logger:
    """Configure root logging and return the ``call-me-maybe`` logger.

    Args:
        verbose: Enable INFO level (default WARNING).
        trace: Enable DEBUG level (overrides verbose).
        log_file: Single log file path, truncated (mode="w") each run.
        console: When true, also mirror logs to ``sys.stderr``.

    Returns:
        The project root logger named ``call-me-maybe``.

    Notes:
        * By default logs are file-only so ``sys.stdout`` stays clean for
          project ``print()`` output.
        * File handler truncates on each run (``mode="w"``).
        * Callers should not log at intermediate layers before raising;
          only the CLI boundary logs errors (see module docstring).
    """
    global _configured

    if trace:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    log_path = Path(log_file)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        log_path = Path("/tmp/call-me-maybe-app.log")

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT)
    prompt_filter = _PromptFilter()

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(level)
        stream.setFormatter(formatter)
        stream.addFilter(prompt_filter)
        root.addHandler(stream)

    try:
        file_handler = logging.FileHandler(
            log_path, mode="w", encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(prompt_filter)
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("Could not open log file %s: %s", log_path, exc)

    # Keep third-party noise down even when root is at DEBUG (trace).
    # Only our ``call-me-maybe.*`` loggers should emit DEBUG/INFO;
    # everything else stays at WARNING/ERROR so ``--trace`` does not
    # flood stderr / the log file with httpx/httpcore chatter.
    for _name in (
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "huggingface_hub",
        "urllib3",
        "urllib3.connectionpool",
        "requests",
        "tokenizers",
        "filelock",
        "fsspec",
    ):
        logging.getLogger(_name).setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    # ``huggingface_hub`` already set to WARNING above; keep util noisy
    # logger quiet as well.
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.WARNING)

    _configured = True
    logger = logging.getLogger("call-me-maybe")
    logger.debug("logging configured (level=%s, file=%s)", level, log_path)
    return logger


def get_logger(name: str) -> Logger:
    """Return a module logger (child of the configured root)."""
    return logging.getLogger(f"call-me-maybe.{name}")


def bind_prompt(
    logger: Logger, prompt_index: int | str
) -> logging.LoggerAdapter:
    """Return a LoggerAdapter injecting ``prompt=<index>``."""
    return logging.LoggerAdapter(logger, {"prompt": str(prompt_index)})


def log_context(**fields: Any) -> dict[str, Any]:
    """Small helper to build structured ``extra`` payloads."""
    return {"prompt": str(fields.pop("prompt", _PROMPT_DEFAULT)), **fields}
