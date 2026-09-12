"""CLI entrypoint for the Call-Me-Maybe tool.

Provides a small command-line wrapper to load configuration, prepare
the LLM prompt, decode model outputs and write the resulting JSON
function-calls to the configured output file.

Clean handling contract:

* This is the **only** place that logs errors and prints to ``stderr``.
  Lower layers (``config``, ``llm``, ``files_content``, ``json_handler``,
  ``constraint_decoder``, ``utils``) only ``raise AppError`` with
  ``hint``/``context`` and ``from e`` chaining – they never log.
* Diagnostics go to the log file (and to ``stderr`` via the console
  handler configured in ``src/logger.py``).  ``stdout`` is reserved for
  app output and stays clean.
* One log entry + one user-facing ``stderr`` line per failure; no
  double logging and no ``stdout`` leakage.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from src import config
from src.constraint_decoder import constraint_decoder
from src.errors import AppError, exit_code_for, is_user_error
from src.files_content import load_func_def, load_user_input
from src.json_handler import write_json
from src.llm import load_base_ids, load_llm, load_vocab
from src.logger import setup_logging
from src.utils import remove_prefix

logger = logging.getLogger("call-me-maybe.cli")


def _print_error(msg: str) -> None:
    """Emit a single user-facing line to ``stderr`` (never ``stdout``)."""
    print(msg, file=sys.stderr)


def _handle_app_error(exc: AppError) -> int:
    """Log ``exc`` once and print a concise ``stderr`` message.

    * User errors (exit 1) → ``WARNING`` without traceback (fixable).
    * Internal errors (exit 2) → ``ERROR`` with traceback (diagnostic).

    Returns the appropriate exit code via :func:`exit_code_for`.
    """
    if is_user_error(exc):
        # No stack trace for user-fixable errors; hint/context are enough.
        logger.warning(
            "Pipeline failed [%s]: %s | hint=%s | context=%s",
            exc.code,
            exc,
            exc.hint,
            exc.context,
        )
    else:
        # Internal failure – keep full traceback for the log file.
        logger.exception("Pipeline failed [%s]: %s", exc.code, exc)
    _print_error(f"Error [{exc.code}]: {exc}")
    if exc.hint:
        _print_error(f"Hint: {exc.hint}")
    return exit_code_for(exc)


def arg_parser(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate natural language prompt into structured "
            "function calls using constrained decoding."
        )
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to function schema JSON file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to input test prompt JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to result JSON file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Prefered huggingface model name."
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable detailed constrained decoding token traces."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO level logs on console and file."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/app.log",
        help="Single log file path (truncated each run, no history)."
    )
    return parser.parse_args(args)


def main(argv: Optional[list[str]] = None) -> int:
    """Parse CLI args, run the decoding pipeline and write output.

    Loads configuration and model artifacts, runs the constraint
    decoder over the user inputs and writes the produced JSON output
    to disk.

    Args:
        argv: Optional argument list for testing (defaults to sys.argv).

    Returns:
        The status code. 0 on success, 1 on user error, 2 on internal error.
    """
    args = arg_parser(argv)
    setup_logging(
        verbose=bool(getattr(args, "verbose", False)),
        trace=bool(getattr(args, "trace", False)),
        log_file=Path(str(getattr(args, "log_file", "logs/app.log"))),
    )
    try:
        config.load_args(args=args)
        load_llm()
        load_vocab()
        load_user_input()
        load_func_def()
        load_base_ids()

        start_time = time.time()

        calls = constraint_decoder()
        remove_prefix(calls)

        total = int(time.time() - start_time)

        logger.info("Time taken: %d min %d s", total // 60, total % 60)
        write_json(json.dumps(calls))
        logger.info("Decoded %d call(s) successfully", len(calls))
        return 0
    except AppError as e:
        return _handle_app_error(e)
    except Exception as e:
        logger.exception("Unexpected pipeline failure: %s", e)
        _print_error(f"An unexpected error occurred: {e}")
        _print_error(
            "Hint: This is an internal error; see log file for details."
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
