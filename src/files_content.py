import json
import config
from json_handler import read_json
from schema import FunctionDef, UserInput
from pydantic import ValidationError

_func_def_content: str | None = None
_func_defs: list[FunctionDef] | None = None
_user_inputs: list[UserInput] | None = None
_func_hash: int = 0

def get_func_hash() -> str:
    global _func_hash

    if _func_hash < 10:
        h = f"000{_func_hash}"
    elif _func_hash < 100:
        h = f"00{_func_hash}"
    else:
        raise ValueError("Hash above maximum value.")
    _func_hash += 1
    return h

def load_user_input() -> None:
    global _user_inputs
    try:
        raw_data = read_json(config.INPUT_FILE)
        if not isinstance(raw_data, list):
            raise ValueError(f"le fichier {config.INPUT_FILE} doit contenir une liste JSON.")
        _user_inputs = [UserInput(**ui) for ui in raw_data]
    except (ValidationError, ValueError) as e:
        raise ValueError(
                f"Invalid pydantic object {config.INPUT_FILE}: {e}"
            ) from e

def load_func_def() -> None:
    global _func_defs
    global _func_def_content
    try:
        raw_funcs = read_json(config.FUNC_DEF_FILE)
        if not isinstance(raw_funcs, list):
            raise ValueError(f"le fichier {config.FUNC_DEF_FILE} doit contenir une liste JSON.")
        for f in raw_funcs:
            f["name"] = get_func_hash() + f["name"]
        _func_def_content = json.dumps(raw_funcs)
        _func_defs = [FunctionDef(**f) for f in raw_funcs]
    except (ValidationError, ValueError) as e:
        raise ValueError(
                f"Invalid pydantic object {config.FUNC_DEF_FILE}: {e}"
            ) from e

def get_func_def_str() -> str:
    global _func_def_content
    if _func_def_content is None:
        load_func_def()
    return _func_def_content


def get_func_defs() -> list[FunctionDef]:
    global _func_defs
    if _func_defs is None:
        load_func_def()
    return _func_defs

def get_user_inputs() -> list[UserInput]:
    global _user_inputs
    if _user_inputs is None:
        load_user_input()
    return _user_inputs
