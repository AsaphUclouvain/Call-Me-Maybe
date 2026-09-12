"""Global configuration and constants for the project.

Holds default file locations and utilities to override them from
command-line arguments.
"""

import os
from typing import Final
import argparse

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


def load_args(args: argparse.Namespace) -> None:
    """Load file path overrides from parsed CLI arguments.

    Args:
        args: Parsed `argparse.Namespace` with optional attributes
            `model`, `functions_definition`, `input`, and `output`.

    Returns:
        None

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
    if args.functions_definition:
        FUNC_DEF_FILE = os.path.abspath(args.functions_definition)
    if args.input:
        INPUT_FILE = os.path.abspath(args.input)
    if args.output:
        OUTPUT_FILE = os.path.abspath(args.output)
    if args.trace:
        TRACE = args.trace
