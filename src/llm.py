"""Lightweight LLM helper wrappers and cached artifacts.

This module provides lazy-loading helpers for the underlying
`Small_LLM_Model`, its tokenizer maps and the pre-built base prompt
token ids used across decoding sessions.
"""

from typing import cast
from llm_sdk import Small_LLM_Model
from src import config
from src.json_handler import read_json
from src.files_content import get_func_def_str

_llm: Small_LLM_Model | None = None
_id_to_token: dict[int, str] | None = None
_token_to_id: dict[str, int] | None = None
_base_ids: list[int] | None = None


def load_llm() -> None:
    """Instantiate the `Small_LLM_Model` if not already loaded.

    Returns:
        None
    """
    global _llm
    if _llm is None:
        _llm = Small_LLM_Model(
            model_name="Qwen/Qwen3-0.6B",
        )


def get_llm() -> Small_LLM_Model:
    """Return the cached `Small_LLM_Model`, instantiating if necessary.

    Returns:
        Small_LLM_Model: The loaded LLM instance.
    """
    if _llm is None:
        load_llm()
    assert _llm is not None
    return _llm


def load_vocab() -> None:
    """Load tokenizer mappings from the LLM's vocab file.

    Returns:
        None
    """
    global _id_to_token
    global _token_to_id
    if _id_to_token is None:
        file_path = get_llm().get_path_to_vocab_file()
        data = read_json(file_path)
        if not isinstance(data, dict):
            raise ValueError("vocab file did not contain a dict mapping")
        _token_to_id = cast(dict[str, int], data)
        _id_to_token = {id: token for token, id in _token_to_id.items()}


def get_token_to_id() -> dict[str, int]:
    """Return the token->id mapping, loading vocab if necessary.

    Returns:
        dict: Mapping from token string to integer id.
    """
    if _token_to_id is None:
        load_vocab()
    assert _token_to_id is not None
    return _token_to_id


def get_id_to_token() -> dict[int, str]:
    """Return the id->token mapping, loading vocab if necessary.

    Returns:
        dict: Mapping from integer id to token string.
    """
    if _id_to_token is None:
        load_vocab()
    assert _id_to_token is not None
    return _id_to_token


def string_to_token_ids(s: str) -> list[int]:
    """Encode string `s` into a flat list of token ids.

    Args:
        s: Input string to encode.

    Returns:
        List[int]: Flattened list of token ids.
    """
    res = get_llm().encode(s).flatten().tolist()
    # Ensure we return a list[int]
    return [int(x) for x in res]


def load_base_ids() -> None:
    """Build and cache the token ids for the base prompt used by decoders.

    Returns:
        None
    """
    global _base_ids
    if _base_ids is None:
        base_prompt = (
            f"{config.PROMPT_HEAD}"
            "[TOOLS]"
            f"{get_func_def_str()}"
            f"{config.RESPONSE_SCHEMA}"
        )
        _base_ids = string_to_token_ids(base_prompt.replace("\n", " "))


def get_base_ids() -> list[int]:
    """Return the cached base prompt token ids, building them if needed.

    Returns:
        List[int]: Token ids for the base prompt.
    """
    if _base_ids is None:
        load_base_ids()
    assert _base_ids is not None
    return _base_ids
