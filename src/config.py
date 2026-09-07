from typing import Final
from argparse import ArgumentParser
from pathlib import Path

PROMPT_HEAD: Final[str] = """[SYSTEM]
You are a function calling assistant. Answer the given prompt by selecting
the correct function from the available tools and generate a valid and schema compliant JSON.
"""

RESPONSE_SCHEMA: Final[str] = """[SCHEMA]
{"prompt":"what you are asked to do","name":"function_name","parameters":{"parameter1":"value1","parameter2":"value2",...}}
"""

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
FUNC_DEF_FILE: str = str(ROOT_DIR / "data/input/functions_definition.json")
INPUT_FILE: str = str(ROOT_DIR / "data/input/function_calling_tests.json")
OUTPUT_FILE: str = str(ROOT_DIR / "data/output/function_calls.json")

START_ID: Final[int] = 0
JSON_PREFIX_LENGTH: Final[int] = 10

def load_args(args: ArgumentParser) -> None:
    global FUNC_DEF_FILE
    global INPUT_FILE
    global OUTPUT_FILE
    if args.functions_definition:
        FUNC_DEF_FILE = args.functions_definition
    if args.input:
        INPUT_FILE = args.input
    if args.output:
        OUTPUT_FILE = args.output