"""Global configuration and constants for the project.

Holds default file locations and utilities to override them from
command-line arguments.
"""

import argparse
import logging
from pathlib import Path
from typing import Final

from src.errors import AppError

logger = logging.getLogger("call-me-maybe.config")

MODEL_NAME: str = "Qwen/Qwen3-0.6B"

PROMPT_HEAD: Final[str] = """[SYSTEM]
You are a function calling assistant. Answer the given prompt by selecting
the correct function from the available tools and generate a valid and schema
compliant JSON.
"""

RESPONSE_SCHEMA: Final[str] = """[EXAMPLE]
{"prompt":"Reverse the word 'Bonjour'","name":"fn_reverse_string",
"parameters":{"name":"Bonjour"}}
"""

FUNC_DEF_FILE: str = "data/input/functions_definition.json"
INPUT_FILE: str = "data/input/function_calling_tests.json"
OUTPUT_FILE: str = "data/output/function_calls.json"


FUNC_PREFIX_LENGTH: Final[int] = 2

TRACE: bool = False


def _resolve_input_file(label: str, raw: str) -> str:
    """Resolve to absolute path and fail fast if missing/not a file."""
    resolved = str(Path(raw).resolve())
    if not raw.strip():
        raise AppError(
            f"Empty path provided for {label}.",
            code="CONFIG_INVALID",
            hint=f"Pass a valid --{label} path.",
            context={"label": label, "path": raw},
        )
    if not Path(resolved).is_file():
        raise AppError(
            f"Input file for {label} not found: {resolved}",
            code="CONFIG_INVALID",
            hint="Check the path and working directory.",
            context={"label": label, "path": resolved},
        )
    return resolved


def load_args(args: argparse.Namespace) -> None:
    """Load file path overrides from parsed CLI arguments.

    Args:
        args: Parsed `argparse.Namespace` with optional attributes
            `model`, `functions_definition`, `input`, and `output`.

    Returns:
        None

    Raises:
        AppError: If model name is empty or an input path is invalid.

    Mutates the module-level FILE variables when corresponding
    options are provided.
    """
    global MODEL_NAME
    global FUNC_DEF_FILE
    global INPUT_FILE
    global OUTPUT_FILE
    global TRACE
    if args.model:
        MODEL_NAME = args.model
    if not MODEL_NAME.strip():
        raise AppError(
            "Empty model name.",
            code="CONFIG_INVALID",
            hint="Pass a valid HuggingFace model id via --model.",
            context={"model": args.model},
        )
    if args.functions_definition:
        FUNC_DEF_FILE = _resolve_input_file(
            "functions_definition", args.functions_definition
        )
    if args.input:
        INPUT_FILE = _resolve_input_file("input", args.input)
    if args.output:
        if not str(args.output).strip():
            raise AppError(
                "Empty output path.",
                code="CONFIG_INVALID",
                hint="Pass a valid --output path.",
                context={"path": args.output},
            )
        OUTPUT_FILE = str(Path(str(args.output)).resolve())
    if args.trace:
        TRACE = bool(args.trace)
    logger.debug(
        "config loaded (model=%s, func_def=%s, input=%s, output=%s, trace=%s)",
        MODEL_NAME,
        FUNC_DEF_FILE,
        INPUT_FILE,
        OUTPUT_FILE,
        TRACE,
    )
