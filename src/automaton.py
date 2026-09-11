from __future__ import annotations

"""Automata generators used to validate JSON tokens during decoding.

This module builds small deterministic automata for JSON string,
number and boolean lexical forms that are used by the decoder to
restrict valid token sequences produced by the model.
"""

from collections.abc import Callable
from collections import defaultdict
from functools import lru_cache


@lru_cache
def get_str_automaton(
    end_chr: str
) -> tuple[int, dict[int, dict[int, Callable[[str], bool]]]]:
    """Return an automaton that validates JSON string content.

    Args:
        end_chr: The character that terminates the string in the
            surrounding JSON context (usually '"' or ',').

    Returns:
        A tuple (final_state_id, automaton) where `automaton` is a
        mapping of state->(next_state->predicate).
    """
    automaton: dict[int, dict[int, Callable]] = defaultdict(dict)
    state_id: int = 0

    def is_hex(c: str) -> bool:
        return c in '0123456789abcdefABCDEF'

    def is_unescaped(c: str) -> bool:
        cp = ord(c)
        return (
            (0x20 <= cp <= 0x21)
            or (0x23 <= cp <= 0x5B)
            or (0x5D <= cp <= 0x10FFFF)
        )

    automaton[state_id][state_id] = is_unescaped
    automaton[state_id][state_id + 1] = lambda c: c == '\\'
    automaton[state_id][state_id + 6] = lambda c, cmp=end_chr: c == cmp
    automaton[state_id + 1][state_id] = lambda c: c in '"\\/bfnrt'
    automaton[state_id + 1][state_id + 2] = lambda c: c == 'u'
    automaton[state_id + 2][state_id + 3] = is_hex
    automaton[state_id + 3][state_id + 4] = is_hex
    automaton[state_id + 4][state_id + 5] = is_hex
    automaton[state_id + 5][state_id] = is_hex
    return state_id + 6, automaton


@lru_cache
def get_number_automaton(
    end_chr: str
) -> tuple[int, dict[int, dict[int, Callable[[str], bool]]]]:
    """Return an automaton that validates JSON numeric literals.

    Args:
        end_chr: The terminal character that ends the numeric literal.

    Returns:
        A tuple (final_state_id, automaton) suitable for token-level
        validation.
    """
    automaton: dict[int, dict[int, Callable]] = defaultdict(dict)
    state_id: int = 0

    def is_digit(c: str) -> bool:
        return c in '0123456789'

    automaton[state_id][state_id + 1] = lambda c: c == '0'
    automaton[state_id][state_id + 2] = lambda c: c in "123456789"
    automaton[state_id][state_id + 3] = lambda c: c == '-'
    automaton[state_id + 3][state_id + 1] = lambda c: c == '0'
    automaton[state_id + 3][state_id + 2] = lambda c: c in "123456789"
    automaton[state_id + 1][state_id + 4] = lambda c: c == '.'
    automaton[state_id + 1][state_id + 6] = lambda c: c in 'eE'
    automaton[state_id + 1][state_id + 10] = lambda c, cmp=end_chr: c == cmp
    automaton[state_id + 4][state_id + 5] = is_digit
    automaton[state_id + 5][state_id + 5] = is_digit
    automaton[state_id + 5][state_id + 6] = lambda c: c in 'eE'
    automaton[state_id + 5][state_id + 10] = lambda c, cmp=end_chr: c == cmp
    automaton[state_id + 2][state_id + 4] = lambda c: c == '.'
    automaton[state_id + 2][state_id + 2] = is_digit
    automaton[state_id + 2][state_id + 6] = lambda c: c in 'eE'
    automaton[state_id + 2][state_id + 10] = lambda c, cmp=end_chr: c == cmp
    automaton[state_id + 6][state_id + 7] = lambda c: c == '+'
    automaton[state_id + 6][state_id + 8] = lambda c: c == '-'
    automaton[state_id + 6][state_id + 9] = is_digit
    automaton[state_id + 7][state_id + 9] = is_digit
    automaton[state_id + 8][state_id + 9] = is_digit
    automaton[state_id + 9][state_id + 9] = is_digit
    automaton[state_id + 9][state_id + 10] = lambda c, cmp=end_chr: c == cmp

    return state_id + 10, automaton


@lru_cache
def get_boolean_automaton(
    end_chr: str
) -> tuple[int, dict[int, dict[int, Callable[[str], bool]]]]:
    """Return an automaton that validates JSON boolean literals.

    Args:
        end_chr: The character that follows the boolean literal.

    Returns:
        A tuple (final_state_id, automaton) that recognizes `true`/`false`.
    """
    automaton: dict[int, dict[int, Callable]] = defaultdict(dict)
    state_id: int = 0

    automaton[state_id][state_id + 1] = lambda c: c == 't'
    automaton[state_id][state_id + 2] = lambda c: c == 'f'
    automaton[state_id + 1][state_id + 3] = lambda c: c == 'r'
    automaton[state_id + 3][state_id + 4] = lambda c: c == 'u'
    automaton[state_id + 4][state_id + 7] = lambda c: c == 'e'
    automaton[state_id + 2][state_id + 5] = lambda c: c == 'a'
    automaton[state_id + 5][state_id + 6] = lambda c: c == 'l'
    automaton[state_id + 6][state_id + 4] = lambda c: c == 's'
    automaton[state_id + 7][state_id + 8] = lambda c, cmp=end_chr: c == cmp

    return state_id + 8, automaton
