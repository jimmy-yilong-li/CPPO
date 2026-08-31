"""Conservative normalization for labelled plan tuples."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from cppo.data.prompts import parse_plan_tuple


@dataclass(frozen=True)
class CanonicalPlan:
    raw_plan_text: str
    plan_text: str
    is_parseable: bool
    canonicalized: bool


def _format_methods(methods: list[str]) -> str:
    return "\n\n".join(
        f"{chr(ord('A') + i)}: {method.strip()}" for i, method in enumerate(methods)
    )


def _parse_literal_wrapper(text: str) -> str | None:
    try:
        literal = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(literal, (list, tuple)):
        return None
    if not all(isinstance(item, str) for item in literal):
        return None
    joined = "\n\n".join(item.strip() for item in literal if item.strip())
    return joined or None


_ANY_LABEL_PATTERN = re.compile(
    r"(?:^|\n)([ \t]*[#>\-\*]*[ \t]*\*{0,2})([A-Z])\*{0,2}[ \t]*[:.)]"
)


def _has_strict_label_sequence(text: str, *, k: int) -> bool:
    """Return whether text is exactly one A..K labelled tuple.

    This is intentionally stricter than ``parse_plan_tuple``. The parser is
    permissive because solve prompts can recover methods from common chat
    wrappers. The canonicalizer must not turn malformed planner output into a
    valid RM input, so it rejects prefaces, duplicate labels, and extra labels.
    """
    matches = list(_ANY_LABEL_PATTERN.finditer(text))
    expected = [chr(ord("A") + i) for i in range(k)]
    if [m.group(2) for m in matches] != expected:
        return False
    if not matches:
        return False
    if text[: matches[0].start()].strip():
        return False
    return True


def canonicalize_plan(text: str, *, k: int = 4) -> CanonicalPlan:
    """Return a parse-based canonical text without improving plan content.

    The canonicalizer is intentionally conservative:
    - It may strip wrapper text and markdown label decoration that
      ``parse_plan_tuple`` already ignores.
    - It may unwrap Python list/tuple string wrappers only if the resulting
      text already parses as a labelled tuple.
    - It does not add missing labels, repair missing methods, remove code, or
      rewrite method content.
    """
    raw = text.strip()

    parsed = parse_plan_tuple(raw, k=k) if _has_strict_label_sequence(raw, k=k) else None
    if parsed is None:
        wrapper = _parse_literal_wrapper(raw)
        if wrapper is not None and _has_strict_label_sequence(wrapper, k=k):
            parsed = parse_plan_tuple(wrapper, k=k)

    if parsed is None:
        return CanonicalPlan(
            raw_plan_text=raw,
            plan_text=raw,
            is_parseable=False,
            canonicalized=False,
        )

    canonical = _format_methods(parsed)
    return CanonicalPlan(
        raw_plan_text=raw,
        plan_text=canonical,
        is_parseable=True,
        canonicalized=canonical != raw,
    )
