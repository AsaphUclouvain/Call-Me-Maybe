"""Decode constrained JSON outputs from a local LLM instance.

This module contains the decoding logic that repeatedly queries model
logits and constrains the next-token choices using small lexical
automata so that the produced output is guaranteed to be a valid
JSON describing a function call and its parameters.
"""

import json
import sys
from typing import Callable, Any, cast
from llm_sdk import Small_LLM_Model
from src import config
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
import numpy as np


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

    def __init__(
        self,
        base_ids: list[int],
        prompt: str,
        index: int
    ) -> None:
        self.ids: list[int] = [id for id in base_ids]
        self.bracket_count: int = 0
        self.prompt: str = prompt
        self.index: int = index
        self.func_ids: list[int] = []

    def extend_ids(self, new_ids: list[int]) -> None:
        """Append token ids and print the decoded text to stdout.

        Args:
            new_ids: Sequence of token ids to append.

        Returns:
            None
        """
        if config.TRACE:
            s = get_llm().decode(new_ids)
            print(s, end="")
            sys.stdout.flush()
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
            RuntimeError: If no valid token can be selected or the
                iteration budget is exhausted.
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
        while max_iter > 0:
            logits = get_llm().get_logits_from_input_ids(self.ids)
            valid_ids = self._get_value_valid_ids(glob_state, automaton)
            if not valid_ids:
                raise RuntimeError(
                    f"No valid tokens from state {glob_state}"
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
            raise RuntimeError("Max iteration reached")

    def add_function(self) -> None:
        """Select and append a function name produced by the model.

        The method iteratively builds the function name token-by-token
        until a unique function from the available definitions matches.

        Returns:
            None.
        """
        if hasattr(self, 'func'):
            return

        idx: int = 0
        candidates = [f for f in get_func_defs()]
        self.extend_ids(string_to_token_ids('"name":"'))
        max_iter: int = 100
        while max_iter > 0:
            logits = get_llm().get_logits_from_input_ids(self.ids)
            valid_ids = self._get_func_name_valid_ids(idx, candidates)
            if not valid_ids:
                raise RuntimeError(
                    f"No valid function-name tokens at idx {idx}"
                )
            logits_np = np.array(logits, dtype=np.float32)
            best_idx_in_valid = int(np.argmax(logits_np[valid_ids]))
            max_idx = valid_ids[best_idx_in_valid]
            self.ids.append(max_idx)  # We don't print the hash prefix
            best_tok = get_id_to_token()[max_idx]
            candidates = [
                f for f in candidates if f.name[idx:].startswith(best_tok)
            ]
            idx += len(best_tok)
            if len(candidates) == 1:
                self.func = candidates[0]
                if idx < len(candidates[0].name):
                    f_name = candidates[0].name
                    self.extend_ids(string_to_token_ids(f_name[idx:]))
                self.extend_ids(string_to_token_ids('",'))
                return
            max_iter -= 1
        raise RuntimeError("Max iteration reached. No function found.")

    def add_arguments(self) -> None:
        """Append serialized argument values for the selected function.

        Each parameter is emitted using the appropriate lexical
        automaton so that types and separators are respected.

        Returns:
            None
        """
        def last_value(i: int, n: int) -> str:
            if i < n - 1:
                return ','
            return '}'
        if not hasattr(self, 'func'):
            RuntimeError("No FunctionDef object bound to this object")
        parameters: dict[str, VarMetaData] = self.func.parameters
        self.extend_ids(string_to_token_ids('"parameters":{'))
        for i, (p_name, p_meta) in enumerate(parameters.items()):
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
    """
    ups: list[UserInput] = get_user_inputs()
    llm: Small_LLM_Model = get_llm()
    base_ids: list[int] = get_base_ids()
    final_ids: list[int] = []
    print("[", end="")
    final_ids.extend(string_to_token_ids("["))
    for i, up in enumerate(ups):
        index = -1 if i == len(ups) - 1 else i
        resp = LLMResponse(base_ids=base_ids, prompt=up.prompt, index=index)
        resp.init_ids()
        resp.add_function()
        resp.add_arguments()
        resp.end_ids()
        final_ids.extend(resp.ids[len(base_ids):])
    final_ids.extend(string_to_token_ids("]"))
    print("]")
    return cast(list[dict[str, Any]], json.loads(llm.decode(final_ids)))
