"""Lightweight LLM helper wrappers and cached artifacts.

This module provides lazy-loading helpers for the underlying
`Small_LLM_Model`, its tokenizer maps and the pre-built base prompt
token ids used across decoding sessions.

Error contract: helpers raise ``AppError`` on failure without
logging (``from e`` chaining preserves cause).  Only
``src/cli.py:main`` logs at the boundary, keeping ``stdout`` clean
and the log file free of duplicate entries.
"""

import logging
from typing import cast

from llm_sdk import Small_LLM_Model

from src import config
from src.errors import AppError
from src.files_content import get_func_def_str
from src.json_handler import read_json

logger = logging.getLogger("call-me-maybe.llm")

_llm: Small_LLM_Model | None = None
_id_to_token: dict[int, str] | None = None
_token_to_id: dict[str, int] | None = None
_base_ids: list[int] | None = None


def load_llm() -> None:
    """Instantiate the `Small_LLM_Model` if not already loaded.

    Returns:
        None

    Raises:
        AppError: If the model cannot be downloaded or instantiated.
    """
    global _llm
    if _llm is None:
        logger.info("Loading model %s ...", config.MODEL_NAME)
        try:
            _llm = Small_LLM_Model(
                model_name=config.MODEL_NAME,
            )
        except Exception as e:
            raise AppError(
                f"Failed to load model {config.MODEL_NAME}: {e}",
                code="MODEL_LOAD",
                hint="Check the model id, network access and disk space.",
                context={"model": config.MODEL_NAME},
            ) from e
        logger.info("Model %s loaded", config.MODEL_NAME)


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

    Raises:
        AppError: If the vocab file cannot be read or is malformed.
    """
    global _id_to_token
    global _token_to_id
    if _id_to_token is None:
        try:
            file_path = get_llm().get_path_to_vocab_file()
        except Exception as e:
            raise AppError(
                f"Could not resolve vocab file: {e}",
                code="VOCAB_INVALID",
                hint="Check the model id and HF cache permissions.",
                context={"model": config.MODEL_NAME},
            ) from e
        try:
            data = read_json(file_path)
        except AppError as e:
            raise AppError(
                str(e),
                code="VOCAB_INVALID",
                hint=e.hint,
                context={**e.context, "vocab_file": file_path},
            ) from e
        if not isinstance(data, dict):
            raise AppError(
                "Vocab file did not contain a dict mapping.",
                code="VOCAB_INVALID",
                hint="The tokenizer vocab is corrupt; clear the HF cache.",
                context={"vocab_file": file_path},
            )
        _token_to_id = cast(dict[str, int], data)
        _id_to_token = {id: token for token, id in _token_to_id.items()}
        logger.info(
            "Loaded vocab (%d tokens) from %s", len(_token_to_id), file_path
        )


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

    Raises:
        AppError: If encoding fails (code INFERENCE).
    """
    try:
        res = get_llm().encode(s).flatten().tolist()
    except Exception as e:
        raise AppError(
            f"Token encoding failed: {e}",
            code="INFERENCE",
            hint="Retry; if persistent the model/tokenizer is broken.",
            context={"text_len": len(s)},
        ) from e
    # Ensure we return a list[int]
    ids = [int(x) for x in res]
    if not ids and s:
        logger.debug("Encoding produced no ids for %d-char string", len(s))
    return ids


def load_base_ids() -> None:
    """Build and cache the token ids for the base prompt used by decoders.

    Returns:
        None

    Raises:
        AppError: If the base prompt cannot be encoded.
    """
    global _base_ids
    if _base_ids is None:
        try:
            base_prompt = (
                f"{config.PROMPT_HEAD}"
                "[TOOLS]"
                f"{get_func_def_str()}"
                f"{config.RESPONSE_SCHEMA}"
            )
            _base_ids = string_to_token_ids(base_prompt.replace("\n", " "))
        except AppError:
            raise
        except Exception as e:
            raise AppError(
                f"Failed to build base prompt ids: {e}",
                code="BASE_PROMPT",
                hint="Check function definitions are valid JSON-serializable.",
            ) from e
        logger.debug("Base prompt encoded (%d ids)", len(_base_ids))


def get_base_ids() -> list[int]:
    """Return the cached base prompt token ids, building them if needed.

    Returns:
        List[int]: Token ids for the base prompt.
    """
    if _base_ids is None:
        load_base_ids()
    assert _base_ids is not None
    return _base_ids
