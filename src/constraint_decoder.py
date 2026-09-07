from collections import defaultdict
import json
import sys
from llm import get_llm, get_id_to_token, get_token_to_id, get_base_ids, string_to_token_ids
from files_content import get_func_defs, get_user_inputs
from automaton import get_str_automaton, get_boolean_automaton, get_number_automaton
from schema import FunctionDef, VarMetaData, UserInput
import numpy as np

class LLMResponse:
    _func_name_valid_cache: dict[tuple[int, tuple[str, ...]], list[int]] = {}
    
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
        s = get_llm().decode(new_ids)
        print(s, end="")
        sys.stdout.flush()
        self.ids.extend(new_ids)

    def init_ids(self) -> None:
        # Use json.dumps to correctly escape the prompt string content
        text = '{"prompt":' + json.dumps(self.prompt) + ','
        self.extend_ids(string_to_token_ids(text))

    def _get_func_name_valid_ids(self, idx: int, candidates: list[FunctionDef]) -> list[int]:
        key = (idx, tuple(sorted(f.name for f in candidates)))
        if key in self._func_name_valid_cache:
            return self._func_name_valid_cache[key]
        valid: list[int] = []
        for tid, tok in get_id_to_token().items():
            if any(f.name[idx:].startswith(tok) for f in candidates):
                valid.append(tid)
        self._func_name_valid_cache[key] = valid
        return valid

    def _valid_token(self, glob_state: int, tok: str, automaton: defaultdict[int, dict[int]]) -> int:
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

    _valid_ids_cache: dict[tuple[int, int], list[int]] = {}
    def _get_value_valid_ids(self, glob_state: int, automaton: defaultdict[int, dict]) -> list[int]:
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
        if get_token_to_id().get(tok):
            return [get_token_to_id().get(tok)]
        return string_to_token_ids(tok)

    def _add_value(self, p_type: str, end_char: str):
        max_iter: int = 100
        glob_state: int = 0
        automaton: defaultdict[int, dict]
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
                raise RuntimeError(f"No valid tokens from state {glob_state}")
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

    def add_function(self) -> FunctionDef:
        """All function in func_defs are differents"""
        def filter_func(f: FunctionDef) -> bool:
            return f.name[idx:].startswith(best_tok)

        if hasattr(self, 'func'):
            return self.func

        idx = 0
        candidates = [f for f in get_func_defs()]
        self.extend_ids(string_to_token_ids('"name":"'))
        max_iter: int = 100
        while max_iter > 0:
            logits = get_llm().get_logits_from_input_ids(self.ids)
            valid_ids = self._get_func_name_valid_ids(idx, candidates)
            if not valid_ids:
                raise RuntimeError(f"No valid function-name tokens at idx {idx}")
            logits_np = np.array(logits, dtype=np.float32)
            best_idx_in_valid = int(np.argmax(logits_np[valid_ids]))
            max_idx = valid_ids[best_idx_in_valid]
            self.extend_ids([max_idx])
            best_tok = get_id_to_token()[max_idx]
            candidates = list(filter(filter_func, candidates))
            idx += len(best_tok)
            if len(candidates) == 1:
                self.func = candidates[0]
                if idx < len(candidates[0].name):
                    f_name = candidates[0].name
                    self.extend_ids(string_to_token_ids(f_name[idx:]))
                self.extend_ids(string_to_token_ids('",'))
                return self.func
            max_iter -= 1
        raise RuntimeError("Max iteration reached. No function found.")

    def add_arguments(self) -> None:
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
        if self.index == -1:
            self.extend_ids(string_to_token_ids('}'))
        else:
            self.extend_ids(string_to_token_ids('},'))

def constraint_decoder() -> str:
    ups: list[UserInput] = get_user_inputs()
    llm = get_llm()
    base_ids: list[int] = get_base_ids()
    final_ids: list[int] = []
    print("[", end="")
    final_ids.extend(string_to_token_ids("["))
    for i, up in enumerate(ups):
        index = -1 if i == len(ups) - 1 else i
        resp = LLMResponse(base_ids=base_ids, prompt=up.prompt, index=index)
        resp.init_ids()
        _ = resp.add_function()
        resp.add_arguments()
        resp.end_ids()
        final_ids.extend(resp.ids[len(base_ids):])
    final_ids.extend(string_to_token_ids("]"))
    print("]")
    return llm.decode(final_ids)
