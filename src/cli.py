"""CLI entrypoint for the Call-Me-Maybe tool.

Provides a small command-line wrapper to load configuration, prepare
the LLM prompt, decode model outputs and write the resulting JSON
function-calls to the configured output file.
"""

import argparse
import json
import time
import sys
from typing import Optional
from src import config
from src.utils import remove_prefix
from src.llm import load_llm, load_vocab, load_base_ids
from src.files_content import load_user_input, load_func_def
from src.constraint_decoder import constraint_decoder
from src.json_handler import write_json


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
    return parser.parse_args(args)


def main() -> int:
    """Parse CLI args, run the decoding pipeline and write output.

    Loads configuration and model artifacts, runs the constraint
    decoder over the user inputs and writes the produced JSON output
    to disk.

    Returns:
        The status code. 0 on success, 1 on failure
    """
    args = arg_parser()
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
        print("Time taken: ", total // 60, "min", total % 60, "s")
        write_json(json.dumps(calls))
        return 0
    except Exception as e:
        print("An error occured: ", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
