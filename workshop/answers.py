"""Strict, non-executing GSM8K numeric final-answer verification."""
from __future__ import annotations

import re
from fractions import Fraction

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def numeric_value(text):
    if text is None:
        return None
    t = str(text).strip().replace("âˆ’", "-").replace("\\,", "")
    # Currency decoration is harmless; units or extra arithmetic are not accepted.
    t = t.strip("$").strip()
    t = re.sub(r"^\\(?:boxed|text)\{([^{}]+)\}$", r"\1", t)
    t = re.sub(r"\\(?:dfrac|tfrac)", r"\\frac", t)
    m = re.fullmatch(r"\\frac\{(" + NUMBER + r")\}\{(" + NUMBER + r")\}", t)
    if m:
        t = f"{m[1]}/{m[2]}"
    # Only conventional thousands grouping; '1,2' must not be treated as 12.
    if "," in t:
        if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", t):
            return None
        t = t.replace(",", "")
    if len(t) > 120:
        return None
    parts = t.split("/")
    if len(parts) not in (1, 2) or any(not re.fullmatch(NUMBER, x.strip()) for x in parts):
        return None
    try:
        # Bound scientific exponents before allocating large integers.
        for part in parts:
            exp = re.search(r"[eE]([+-]?\d+)$", part.strip())
            if exp and abs(int(exp[1])) > 100:
                return None
        value = Fraction(parts[0].strip())
        if len(parts) == 2:
            value /= Fraction(parts[1].strip())
        return value
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def boxed_spans(text):
    values = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        start, level, end = m.end(), 1, m.end()
        while end < len(text) and level:
            level += (text[end] == "{") - (text[end] == "}")
            end += 1
        if level == 0:
            values.append((m.start(), end, text[start:end - 1]))
    return values


def verify_answer(response, reference):
    gold_text = reference.rsplit("####", 1)[-1].strip()
    gold = numeric_value(gold_text)
    if gold is None:
        raise ValueError(f"Unsupported GSM8K gold answer: {gold_text!r}")
    candidates = boxed_spans(response)
    # Strict contract: exactly one final \boxed{} answer. No last-number guessing.
    predicted = numeric_value(candidates[0][2]) if len(candidates) == 1 else None
    return {"correct": bool(predicted is not None and predicted == gold),
            "format_valid": predicted is not None, "boxed_count": len(candidates),
            "predicted_answer": str(predicted) if predicted is not None else None,
            "gold_answer": str(gold)}


def parse_rating(text):
    # A malformed judge output is an error, never an implicit low reward.
    matches = re.findall(r"(?im)^\s*Correctness_score\s*:\s*([1-5])\s*[.!]?\s*$", text)
    return int(matches[0]) if len(matches) == 1 else None


def parse_rating_inline(text):
    """One complete inline declaration, without rationale or candidate quoting."""
    match = re.fullmatch(r"\s*Judgement:\s*Correctness_score\s*:\s*([1-5])\s*[.!]?\s*", text, re.IGNORECASE)
    return int(match[1]) if match else None


def parse_rating_prose(text):
    """Read one explicit terminal score sentence; never infer from arithmetic.

    This is a failed-only extension of parse_rating, applied to complete grader
    replies. Keep the strict parser and all already valid cache entries intact.
    """
    if not re.match(r"(?i)^\s*Judgement:\s", text):
        return None
    # A second declaration, even an invalid/empty one, makes this ambiguous.
    if len(re.findall(r"(?i)\bcorrectness[\s_]+score\b", text)) != 1:
        return None
    if re.search(r"(?i)\bCorrectness_score\b", text):
        return None
    # Require a complete final sentence, not a quoted score, conditional clause,
    # rating range, decimal, candidate's asserted grade, or arbitrary last digit.
    match = re.search(
        r"(?i)(?:[.!?]\s+|\n\s*)(?:Therefore,\s+)?"
        r"the correctness score is ([1-5])\.\s*$", text)
    return int(match[1]) if match else None
