from llm_sdk import Small_LLM_Model
from .json_handler import read_json
from .files_content import get_func_def_str
from . import config

_llm : Small_LLM_Model | None = None
_id_to_token: dict[int, str] = {}
_token_to_id: dict[str, int]  = {}
_base_ids: list[int] = []

def load_llm() -> None:
    global _llm
    if _llm is None:
        _llm = Small_LLM_Model(
            model_name="Qwen/Qwen3-0.6B",
        )

def get_llm() -> Small_LLM_Model:
    global _llm
    if _llm is None:
        load_llm()
    return _llm

def load_vocab() -> None:
    global _id_to_token
    global _token_to_id
    if not _id_to_token:
        file_path = get_llm().get_path_to_vocab_file()
        _token_to_id = read_json(file_path)
        _id_to_token = {id: token for token, id in _token_to_id.items()}

def get_token_to_id() -> dict[str, int]:
    global _token_to_id
    if _token_to_id is None:
        load_vocab()
    return _token_to_id

def get_id_to_token() -> dict[int, str]:
    global _id_to_token
    if _id_to_token is None:
        load_vocab()
    return _id_to_token

def string_to_token_ids(s: str) -> list[int]:
    res = get_llm().encode(s).flatten().tolist()
    return res

def load_base_ids() -> None:
    global _base_ids
    if not _base_ids:
        base_prompt = (
            f"{config.PROMPT_HEAD}"
            "[TOOLS]\n"
            f"{get_func_def_str()}\n"
            f"{config.RESPONSE_SCHEMA}\n"
        )
        _base_ids = string_to_token_ids(base_prompt)

def get_base_ids() -> list[int]:
    global _base_ids
    if not _base_ids:
        load_base_ids()
    return _base_ids