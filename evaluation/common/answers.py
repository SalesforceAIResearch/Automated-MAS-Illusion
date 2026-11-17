"""Voting is independent of gold labels; malformed answers abstain."""
import ast
from collections import Counter
import json
import re


def parse_response(text):
    text = (text or '').strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
    text = re.sub(r'\s*```$', '', text)
    for parser in [json.loads, ast.literal_eval]:
        try:
            value = parser(text)
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError, TypeError):
            pass
    return None


def answer_key(value):
    if value is None:
        return ()
    if isinstance(value, dict):
        return answer_key(value['answer']) if 'answer' in value else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.lower() in {'none', 'null'}:
            return ()
        if text.startswith(('[', '{', '"', "'")):
            for parser in [json.loads, ast.literal_eval]:
                try:
                    parsed = parser(text)
                    if parsed != text:
                        return answer_key(parsed)
                except (ValueError, TypeError, SyntaxError):
                    pass
        value = re.split(r'\s+and\s+|\s*,\s*', text, flags=re.I)
    if isinstance(value, (list, tuple)):
        return tuple(sorted({str(x).strip().lower() for x in value if x is not None and str(x).strip()}))
    return None


def parsed_answer(parsed):
    return answer_key(parsed['answer']) if isinstance(parsed, dict) and 'answer' in parsed else None


def vote_keys(keys):
    valid = [key for key in keys if key is not None]
    return Counter(valid).most_common(1)[0][0] if valid else None
