"""Canonical SMFR answer scoring; unchanged from the v2 evaluator."""
import json
import re
from typing import Any, Optional, List

def _extract_final_answer(answer: Any) -> Optional[List[str]]:
    """Unwrap nested dicts and normalise to a flat list of name strings."""
    if answer is None:
        return None
    if isinstance(answer, dict):
        answer = answer.get("answer")
    if answer is None:
        return None
    if isinstance(answer, list):
        names = [str(a) for a in answer if a is not None]
        return names if names else None
    if isinstance(answer, str):
        return [answer]
    return [str(answer)]

def _extract_answer_with_context(answer: Any, known_names: List[str]) -> Optional[List[str]]:
    """Extract answer names using known investor names as a vocabulary.

    `answer` may be a full output dict (MAS pipeline), a bare answer value (CoT/SC), or None.

    Returns None (valid "no winner") when:
      - The model explicitly outputs the string "None"/"null" or an empty list
      - MAS pipeline ran to completion, has investor_dates, and answer is null

    Returns "__no_answer__" (API failure, counts as wrong) when:
      - Output is completely missing (parsed as {}) — no answer key at all
      - Bare Python None at top level (no output dict present)

    Penalises false ties: "Alice and Bob" when only "Alice" is correct → {Alice, Bob} ≠ {Alice}.
    """
    # Full output dict — check for reasoned None (handles single and double nesting)
    if isinstance(answer, dict):
        has_investor_dates = "investor_dates" in answer
        inner = answer.get("answer")
        if inner is None:
            return None if has_investor_dates else "__no_answer__"
        # Double-nesting: inner is itself a structured pipeline output dict
        if isinstance(inner, dict) and "investor_dates" in inner:
            nested_answer = inner.get("answer")
            if nested_answer is None:
                return None  # reasoned None from nested structure
            answer = nested_answer
        else:
            answer = inner

    if answer is None:
        return "__no_answer__"

    if isinstance(answer, list):
        names = [str(a) for a in answer if a is not None]
        return names if names else None  # empty list = explicit no-winner

    if not isinstance(answer, str):
        return [str(answer)]

    if answer.strip().lower() in ("none", "null", ""):
        return None  # explicit "no winner" string from model

    if known_names:
        found = [
            name for name in known_names
            if re.search(rf'\b{re.escape(name)}\b', answer, re.IGNORECASE)
        ]
        if found:
            return found

    # Fallback: try JSON parse (handles '["Alice"]')
    stripped = answer.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                names = [str(x) for x in parsed if x is not None]
                return names if names else "__no_answer__"
            if isinstance(parsed, str):
                return "__no_answer__" if parsed.strip().lower() in ("none", "null") else [parsed]
        except (json.JSONDecodeError, TypeError):
            pass

    return [answer]

def compare_answers(expected: Any, actual: Any) -> bool:
    """Return True iff the final answer names in `actual` exactly match `expected`.

    A None answer is correct only when GT is also None AND the model output is a
    structured dict containing "investor_dates" (reasoned None, not API failure).
    """
    known_names: List[str] = []
    if isinstance(expected, dict):
        known_names = list(expected.get("investor_dates", {}).keys())

    exp = _extract_final_answer(expected)
    act = _extract_answer_with_context(actual, known_names)

    if exp is None and act is None:
        return True
    if exp is None or act is None or act == "__no_answer__":
        return False
    return set(exp) == set(act)
