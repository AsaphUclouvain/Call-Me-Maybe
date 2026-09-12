"""Decode constrained JSON outputs from a local LLM instance.

This module contains the decoding logic that repeatedly queries model
logits and constrains the next-token choices using small lexical
automata so that the produced output is guaranteed to be a valid
JSON describing a function call and its parameters.

Error contract: all failures raise ``AppError`` with ``code``/``hint``/
``context`` and **no logging**.  Only the CLI boundary logs, so
``stdout`` stays reserved for app output and the log file contains a
single entry per failure (``from e`` preserves the chain).
"""

import json
import logging
from typing import Callable, Any, cast
import numpy as np

from llm_sdk import Small_LLM_Model
from src.pretty_json import (
    JsonStreamPrettyPrinter,
    pretty_print_json
)
from src import config
from src.errors import AppError
from src.llm import (
    get_llm,
    get_id_to_token,
    get_token_to_id,
    get_base_ids,
    string_to_token_ids
)
from src.files_content import get_func_defs, get_user_inputs
from src.automaton import (
    get_str_automaton,
    get_boolean_automaton,
    get_number_automaton
)
from src.schema import FunctionDef, VarMetaData, UserInput

logger = logging.getLogger("call-me-maybe.decoder")


class LLMResponse:
    """Helper representing an in-flight LLM decoding session.

    The object holds the partial token ids produced so far and
    exposes helpers to append function names and parameter values
    while enforcing lexical constraints.

    Attributes:
        ids (list[int]): Accumulated token ids for this response.
        bracket_count (int): Counter used during value decoding.
        prompt (str): Prompt text used to initialise the JSON output.
        index (int): Index of the request in the batch (-1 for last).
        func_ids (list[int]): Token ids collected for the function name.
    """
    _func_name_valid_cache: dict[tuple[int, tuple[str, ...]], list[int]] = {}
    _valid_ids_cache: dict[tuple[int, int], list[int]] = {}
    func: FunctionDef

    def __init__(
        self,
        base_ids: list[int],
        prompt: str,
        index: int,
        printer: JsonStreamPrettyPrinter
    ) -> None:
        self.ids: list[int] = [id for id in base_ids]
        self.bracket_count: int = 0
        self.prompt: str = prompt
        self.index: int = index
        self.func_ids: list[int] = []
        self.printer: JsonStreamPrettyPrinter = printer

    def extend_ids(self, new_ids: list[int]) -> None:
        """Append token ids and trace the decoded text via logging.

        Args:
            new_ids: Sequence of token ids to append.

        Returns:
            None
        """
        if config.TRACE:
            try:
                s = get_llm().decode(new_ids)
                pretty_print_json(
                    finish=False, json_str=s, printer=self.printer
                )
            except Exception as e:
                logger.debug(
                    "Trace decode failed: %s", e,
                    extra={"prompt": str(self.index)},
                )
        self.ids.extend(new_ids)

    def init_ids(self) -> None:
        """Initialize the token buffer with the JSON prompt prefix.

        Uses `json.dumps` to ensure the prompt string is properly escaped.

        Returns:
            None
        """
        text = '{"prompt":' + json.dumps(self.prompt) + ','
        self.extend_ids(string_to_token_ids(text))

    def _get_func_name_valid_ids(
        self,
        idx: int,
        candidates: list[FunctionDef]
    ) -> list[int]:
        """Return token ids that can start valid function names at `idx`.

        Args:
            idx: Current index into the function name string.
            candidates: Available `FunctionDef` candidates.

        Returns:
            A list of token ids that are valid next tokens for function names.
        """
        key = (idx, tuple(sorted(f.name for f in candidates)))
        if key in self._func_name_valid_cache:
            return self._func_name_valid_cache[key]
        valid: list[int] = []
        for tid, tok in get_id_to_token().items():
            if any(f.name[idx:].startswith(tok) for f in candidates):
                valid.append(tid)
        self._func_name_valid_cache[key] = valid
        return valid

    def _valid_token(
        self,
        glob_state: int,
        tok: str,
        automaton: dict[int, dict[int, Callable[[str], bool]]]
    ) -> int:
        """Return the automaton state reached after consuming `tok`.

        Args:
            glob_state: Current automaton state id.
            tok: Token string to consume.
            automaton: Automaton transition mapping.

        Returns:
            The new automaton state id, or -1 when the token cannot be
            consumed from `glob_state`.
        """
        cur_state = glob_state
        for c in tok:
            if cur_state not in automaton:
                return cur_state
            for state in automaton[cur_state]:
                if automaton[cur_state][state](c):
                    cur_state = state
                    break
            else:
                return -1
        return cur_state

    def _get_value_valid_ids(
        self, glob_state: int,
        automaton: dict[int, dict[int, Callable[[str], bool]]]
    ) -> list[int]:
        """Return token ids that keep the automaton in a valid state.

        Args:
            glob_state: Current automaton state id.
            automaton: The automaton mapping used to validate tokens.

        Returns:
            A list of token ids that are valid from `glob_state`.
        """
        key = (glob_state, id(automaton))
        if key in self._valid_ids_cache:
            return self._valid_ids_cache[key]
        valid: list[int] = []
        for tid, tok in get_id_to_token().items():
            if self._valid_token(glob_state, tok, automaton) != -1:
                valid.append(tid)
        self._valid_ids_cache[key] = valid
        return valid

    def _get_ids_from_token(self, tok: str) -> list[int]:
        """Return the token id(s) for a literal token string.

        Args:
            tok: Token text to resolve.

        Returns:
            A list of token ids representing `tok`.
        """
        tid = get_token_to_id().get(tok)
        if tid is not None:
            return [tid]
        return string_to_token_ids(tok)

    def _add_value(self, p_type: str, end_char: str) -> None:
        """Generate and append a value of the given type.

        Args:
            p_type: One of "number", "string", or "boolean".
            end_char: The terminal character that ends the value
            (e.g. '"', ',' or '}').

        Raises:
            AppError: If no valid token can be selected, the
                iteration budget is exhausted, or inference fails.
        """
        max_iter: int = 100
        glob_state: int = 0
        automaton: dict[int, dict[int, Callable[[str], bool]]]
        final_state: int
        if p_type == "number":
            final_state, automaton = get_number_automaton(end_char)
        elif p_type == "string":
            final_state, automaton = get_str_automaton(end_char)
        elif p_type == "boolean":
            final_state, automaton = get_boolean_automaton(end_char)
        else:
            raise AppError(
                f"Unsupported parameter type: {p_type}",
                code="DECODE_FAILED",
                hint="Type must be one of number/string/boolean.",
                context={"p_type": p_type, "prompt_index": self.index},
            )
        while max_iter > 0:
            try:
                logits = get_llm().get_logits_from_input_ids(self.ids)
            except Exception as e:
                raise AppError(
                    f"Inference failed: {e}",
                    code="INFERENCE",
                    hint="Retry; check model device/memory.",
                    context={"prompt_index": self.index},
                ) from e
            valid_ids = self._get_value_valid_ids(glob_state, automaton)
            if not valid_ids:
                raise AppError(
                    f"No valid tokens from state {glob_state}",
                    code="DECODE_NO_TOKEN",
                    hint="Input may force an invalid JSON shape.",
                    context={"state": glob_state, "prompt_index": self.index},
                )
            logits_np = np.array(logits, dtype=np.float32)
            best_idx_in_valid = int(np.argmax(logits_np[valid_ids]))
            max_idx = valid_ids[best_idx_in_valid]
            best_tok = get_id_to_token()[max_idx]
            glob_state = self._valid_token(glob_state, best_tok, automaton)
            if glob_state == final_state:
                k = best_tok.index(end_char)
                self.extend_ids(self._get_ids_from_token(best_tok[:k+1]))
                return
            self.extend_ids([max_idx])
            max_iter -= 1
        if max_iter == 0:
            raise AppError(
                "Max iteration reached while decoding value.",
                code="DECODE_MAX_ITER",
                hint="Value may be unrepresentable; check schema.",
                context={"p_type": p_type, "prompt_index": self.index},
            )

    def add_function(self) -> None:
        """Select and append a function name produced by the model.

        The method iteratively builds the function name token-by-token
        until a unique function from the available definitions matches.

        Returns:
            None.

        Raises:
            AppError: If no valid token exists, inference fails, or
                no function can be resolved within budget.
        """
        if hasattr(self, 'func'):
            return

        idx: int = 0
        candidates = [f for f in get_func_defs()]
        if not candidates:
            raise AppError(
                "No function definitions available.",
                code="DECODE_STATE",
                hint="Check the functions definition file is non-empty.",
                context={"prompt_index": self.index},
            )
        self.extend_ids(string_to_token_ids('"name":"'))
        max_iter: int = 100
        while max_iter > 0:
            try:
                logits = get_llm().get_logits_from_input_ids(self.ids)
            except Exception as e:
                raise AppError(
                    f"Inference failed: {e}",
                    code="INFERENCE",
                    hint="Retry; check model device/memory.",
                    context={"prompt_index": self.index},
                ) from e
            valid_ids = self._get_func_name_valid_ids(idx, candidates)
            if not valid_ids:
                raise AppError(
                    f"No valid function-name tokens at idx {idx}",
                    code="DECODE_NO_TOKEN",
                    hint="Model vocab cannot express any function name.",
                    context={"idx": idx, "prompt_index": self.index},
                )
            logits_np = np.array(logits, dtype=np.float32)
            best_idx_in_valid = int(np.argmax(logits_np[valid_ids]))
            max_idx = valid_ids[best_idx_in_valid]
            self.ids.append(max_idx)  # We don't trace the hash prefix
            best_tok = get_id_to_token()[max_idx]
            candidates = [
                f for f in candidates if f.name[idx:].startswith(best_tok)
            ]
            if not candidates:
                raise AppError(
                    "Model produced a token matching no function.",
                    code="DECODE_FUNC_NOT_FOUND",
                    hint="Check function names vs tokenizer vocab.",
                    context={"token": best_tok, "prompt_index": self.index},
                )
            idx += len(best_tok)
            if len(candidates) == 1:
                self.func = candidates[0]
                if idx < len(candidates[0].name):
                    f_name = candidates[0].name
                    self.extend_ids(string_to_token_ids(f_name[idx:]))
                self.extend_ids(string_to_token_ids('",'))
                logger.debug(
                    "selected function %s", candidates[0].name,
                    extra={"prompt": str(self.index)},
                )
                return
            max_iter -= 1
        raise AppError(
            "Max iteration reached. No function found.",
            code="DECODE_FUNC_NOT_FOUND",
            hint="Function names may share a long prefix; shorten them.",
            context={"prompt_index": self.index},
        )

    def add_arguments(self) -> None:
        """Append serialized argument values for the selected function.

        Each parameter is emitted using the appropriate lexical
        automaton so that types and separators are respected.

        Returns:
            None

        Raises:
            AppError: If no function is bound or a parameter type
                is unsupported.
        """
        def last_value(i: int, n: int) -> str:
            if i < n - 1:
                return ','
            return '}'
        if not hasattr(self, 'func'):
            raise AppError(
                "No FunctionDef object bound to this response.",
                code="DECODE_STATE",
                hint="Call add_function() before add_arguments().",
                context={"prompt_index": self.index},
            )
        parameters: dict[str, VarMetaData] = self.func.parameters
        self.extend_ids(string_to_token_ids('"parameters":{'))
        for i, (p_name, p_meta) in enumerate(parameters.items()):
            if p_meta.type not in ("number", "string", "boolean"):
                raise AppError(
                    f"Unsupported parameter type '{p_meta.type}' "
                    f"for '{p_name}'.",
                    code="DECODE_FAILED",
                    hint="Type must be one of number/string/boolean.",
                    context={"param": p_name, "prompt_index": self.index},
                )
            end: str = ''
            if p_meta.type in ["number", "boolean"]:
                self.extend_ids(string_to_token_ids(f'"{p_name}":'))
            elif p_meta.type == "string":
                self.extend_ids(string_to_token_ids(f'"{p_name}":"'))
                end = '"'
            end += last_value(i, len(parameters))
            self._add_value(p_meta.type, end_char=end[0])
            if end[1:]:
                self.extend_ids(string_to_token_ids(end[1:]))

    def end_ids(self) -> None:
        """Append the closing JSON object or object+separator token(s).

        Returns:
            None
        """
        if self.index == -1:
            self.extend_ids(string_to_token_ids('}'))
        else:
            self.extend_ids(string_to_token_ids('},'))


def constraint_decoder() -> list[dict[str, Any]]:
    """Decode all user inputs into a JSON array of function calls.

    Iterates over configured user inputs, decodes one JSON object per
    input using `LLMResponse` and returns the parsed Python structure.

    Returns:
        A Python object (typically a list of dictionaries) containing
        the decoded function-call representations.

    Raises:
        AppError: If decoding fails for a prompt or the final JSON
            cannot be parsed.
    """
    ups: list[UserInput] = get_user_inputs()
    if not ups:
        logger.info("No user inputs to decode; returning []")
        return []
    llm: Small_LLM_Model = get_llm()
    base_ids: list[int] = get_base_ids()
    printer: JsonStreamPrettyPrinter = JsonStreamPrettyPrinter()
    final_ids: list[int] = []
    final_ids.extend(string_to_token_ids("["))
    if config.TRACE:
        pretty_print_json(finish=False, json_str="[", printer=printer)
    for i, up in enumerate(ups):
        index = -1 if i == len(ups) - 1 else i
        logger.info(
            "Decoding prompt %d/%d", i + 1, len(ups),
            extra={"prompt": str(index)},
        )
        try:
            resp = LLMResponse(
                base_ids=base_ids,
                prompt=up.prompt,
                index=index,
                printer=printer
            )
            resp.init_ids()
            resp.add_function()
            resp.add_arguments()
            resp.end_ids()
            final_ids.extend(resp.ids[len(base_ids):])
        except AppError as e:
            e.context.setdefault("prompt_index", index)
            raise
        logger.info(
            "Decoded prompt %d/%d", i + 1, len(ups),
            extra={"prompt": str(index)},
        )
    final_ids.extend(string_to_token_ids("]"))
    if config.TRACE:
        pretty_print_json(finish=True, json_str="]", printer=printer)
    try:
        decoded = llm.decode(final_ids)
        parsed = json.loads(decoded)
    except Exception as e:
        raise AppError(
            f"Failed to parse decoded output: {e}",
            code="DECODE_FAILED",
            hint="Decoded tokens are not valid JSON; check decoder logic.",
        ) from e
    if not isinstance(parsed, list):
        raise AppError(
            "Decoded output is not a JSON list.",
            code="DECODE_FAILED",
            hint="Expected a top-level [...] of function calls.",
        )
    return cast(list[dict[str, Any]], parsed)
