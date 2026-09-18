"""Secondary numeric-answer extraction, independent of the reference answer.

Strict and relaxed extraction share the same exact numeric and box parsers.
Both checks run inside the one evaluation pipeline.
"""
from __future__ import annotations
import re
from .answers import numeric_value, boxed_spans as boxes

VERSION = 'gsm8k_numeric_recheck_v1'
PROTOCOL = {
    'version': VERSION,
    'scope': 'Final numeric-answer matching only; reasoning and unit semantics are not adjudicated.',
    'status': 'Post-hoc re-evaluation of previously inspected results; not preregistered.',
    'extraction_inputs': ['response text', 'length_capped', 'ended_with_eos'],
    'incomplete': 'Unresolved: do not guess an answer from an unfinished response.',
    'box': 'One parseable numeric box is accepted unless followed by an explicit conflicting answer declaration. Multiple boxes require an unambiguous final-paragraph box; otherwise unresolved.',
    'plain_text': 'Use only the last nonempty paragraph, at most 400 characters. Accept one numeric literal, an explicit Answer/Result declaration, or a numeric final right-hand side of an equation.',
    'ambiguity': 'Reject question/conditional/negative conclusions, unsupported arithmetic and conflicting or multiple unanchored numbers. Never search for the gold value.',
    'normalization': 'Exact rational comparison; signs, decimal/scientific notation, thousands grouping and simple fractions supported. Numeric percentage values stay in the written scale; units are not converted.',
    'metrics': 'Original strict correctness; strict-format compliance; confirmed numeric matches / all responses; resolved numeric mismatches; unresolved rate. Unresolved cases remain separate and are not dropped from the denominator.',
    'fairness': 'Identical extraction rules for all arms; identical question IDs and references required.',
}
PLAIN = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
GROUPED = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\.\d+'
ATOM = rf'(?:\\(?:d?frac)\{{{PLAIN}\}}\{{{PLAIN}\}}|(?:{GROUPED})(?:\s*/\s*(?:{GROUPED}))?)'
NUMBER = re.compile(rf'(?<![\w.,]){ATOM}(?![\w,])')
UNSURE = re.compile(r"(?i)\b(?:if|cannot|can't|not|maybe|might|unsure|either|neither)\b")
UNITS = re.compile(r'^[A-Za-z%Ã‚Â²Ã‚Â³Ã‚Â°][A-Za-z\s%Ã‚Â²Ã‚Â³Ã‚Â°.-]*$')



def numeric(text):
    value = numeric_value(text)
    return str(value) if value is not None else None


def plain_display(text):
    t=text.replace('Ã¢Ë†â€™','-').replace('\\,','')
    t=re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}',r'\1',t)
    for item in ('\\[','\\]','\\(','\\)','\\$','$','**','`','\\!'):
        t=t.replace(item,'')
    return t.strip()


def quantity(text):
    """One number, optionally decorated with simple units/punctuation."""
    t=plain_display(text).strip().rstrip(' .!;:')
    m=NUMBER.fullmatch(t)
    if m:
        return numeric(m[0])
    m=NUMBER.match(t)
    if not m:
        return None
    suffix=t[m.end():].strip().strip('.!;:').strip()
    if not suffix or (UNITS.fullmatch(suffix) and not re.search(r'(?i)\b(?:or|and|but|instead|wrong)\b',suffix)):
        return numeric(m[0])
    return None


def conflicting_tail(value, text):
    tail=plain_display(text)
    declaration=re.search(r'(?i)\b(?:final\s+)?(?:answer|result)\s*(?:is|:|=)\s*(.+)$',tail)
    if declaration:
        other=quantity(declaration[1])
        return other is None or other!=value
    return bool(re.search(r'(?i)\b(?:actually|instead|correction)\b',tail) and NUMBER.search(tail))


def decision(value=None,method='unresolved',evidence='',reason=''):
    return {'prediction':value,'method':method,'evidence':evidence,'reason':reason}


def extract_answer(response, *, length_capped=False, ended_with_eos=True):
    """This signature deliberately excludes question, reference, grades and arm."""
    if length_capped or not ended_with_eos:
        return decision(reason='incomplete_response')
    text=response.strip()
    if not text:
        return decision(reason='empty_response')
    last=re.split(r'\n\s*\n',text)[-1].strip()
    found=boxes(text)
    if len(found)==1:
        value=quantity(found[0][2])
        if value is not None:
            # Repeated problem quantities after a box are common and are not
            # alternative answers. Only an explicit conflicting declaration is
            # ambiguous here; no comparison to the reference is involved.
            if conflicting_tail(value,text[found[0][1]:]):
                return decision(evidence=text[found[0][0]:],reason='conflicting_answer_after_box')
            return decision(value,'single_numeric_box',found[0][2])
        return decision(evidence=found[0][2],reason='unsupported_box_contents')
    if len(found)>1:
        tail_boxes=boxes(last)
        if len(tail_boxes)==1:
            value=quantity(tail_boxes[0][2])
            if value is not None:
                prefix=plain_display(last[:tail_boxes[0][0]])
                if not UNSURE.search(prefix) and '?' not in prefix and not conflicting_tail(value,last[tail_boxes[0][1]:]):
                    return decision(value,'final_paragraph_box',last)
        return decision(evidence=last,reason='multiple_boxes_without_clear_final_answer')
    if '\\boxed' in text:
        return decision(evidence=last,reason='unclosed_or_malformed_box')
    if len(last)>400:
        return decision(evidence=last,reason='long_final_paragraph')
    cleaned=plain_display(last)
    if '?' in cleaned or UNSURE.search(cleaned):
        return decision(evidence=last,reason='question_conditional_or_negative_conclusion')
    # Explicit declarations and an equation's final RHS provide an answer anchor.
    declaration=re.search(r'(?i)(?:\b(?:final\s+)?answer\s*(?:is|:|=)|\bresult\s*(?:is|:|=)|####)\s*(.+)$',cleaned)
    if declaration:
        value=quantity(declaration[1])
        if value is not None:
            return decision(value,'explicit_answer_declaration',last)
        return decision(evidence=last,reason='ambiguous_answer_declaration')
    if '=' in cleaned:
        value=quantity(cleaned.rsplit('=',1)[-1])
        if value is not None:
            return decision(value,'final_equation_rhs',last)
        return decision(evidence=last,reason='unsupported_final_equation')
    hits=list(NUMBER.finditer(cleaned))
    if len(hits)==1:
        value=numeric(hits[0][0])
        # Reject an unmatched fraction/arithmetic operator adjacent to the number.
        outside=cleaned[:hits[0].start()]+cleaned[hits[0].end():]
        if re.search(r'\\(?:d?frac)|[=<>+*/]|\d',outside):
            return decision(evidence=last,reason='unsupported_numeric_expression')
        if value is not None:
            return decision(value,'one_number_final_paragraph',last)
    return decision(evidence=last,reason='no_unique_final_number')
