"""Loader for the codeparrot/apps dataset."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from datasets import load_dataset

from cppo.data.problem import Problem, TestCase

logger = logging.getLogger(__name__)

# APPS problems can contain very large integer strings in their test cases.
# Python 3.11+ caps int->str conversion at 4300 digits by default. Lift it
# so the JSON parser can decode them. Setting at import time so it covers
# all loader paths (00b, 00c, 01, 06, eval, etc.).
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def _parse_io_tests(io_data: dict[str, Any]) -> list[TestCase]:
    """Parse stdin/stdout test cases from APPS input_output JSON."""
    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    n = min(len(inputs), len(outputs))
    return [
        TestCase(input=str(inputs[i]), expected_output=str(outputs[i]))
        for i in range(n)
    ]


def _parse_fn_tests(io_data: dict[str, Any]) -> list[TestCase]:
    """Parse call-based test cases from APPS input_output JSON.

    APPS convention: outputs[i] is wrapped in a single-element list
    when the function returns a scalar. We unwrap it for direct comparison.
    """
    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    n = min(len(inputs), len(outputs))
    tests = []
    for i in range(n):
        out = outputs[i]
        # Unwrap single-element list per APPS convention
        if isinstance(out, list) and len(out) == 1:
            out = out[0]
        tests.append(TestCase(
            input=json.dumps(inputs[i]),
            expected_output=repr(out),
        ))
    return tests


def _parse_row(row: dict[str, Any], idx: int) -> Problem | None:
    """Parse a single APPS dataset row into a Problem, or None if invalid."""
    io_str = row.get("input_output", "")
    if not io_str:
        return None

    try:
        io_data = json.loads(io_str)
    except (json.JSONDecodeError, TypeError):
        return None

    # fn_name must come from INSIDE the input_output JSON, NOT from row top-level
    fn_name = io_data.get("fn_name")

    if fn_name:
        io_mode = "call_based"
        entry_point = fn_name
        test_cases = _parse_fn_tests(io_data)
    else:
        io_mode = "stdin"
        entry_point = None
        test_cases = _parse_io_tests(io_data)

    # Parse solutions
    solutions_str = row.get("solutions", "")
    try:
        solutions = json.loads(solutions_str) if solutions_str else []
    except (json.JSONDecodeError, TypeError):
        solutions = []

    problem_id = row.get("problem_id", idx)
    starter_code = row.get("starter_code", "") or None

    return Problem(
        id=f"apps_{problem_id}",
        prompt=row.get("question", ""),
        test_cases=test_cases,
        domain="code",
        difficulty=row.get("difficulty", "unknown"),
        source="apps",
        io_mode=io_mode,
        entry_point=entry_point,
        starter_code=starter_code,
        metadata={"solutions": solutions},
    )


def load_apps(
    split: str = "train",
    max_problems: int | None = None,
    difficulty: str | None = None,
    min_tests: int = 1,
    max_test_cases: int | None = 10,
) -> list[Problem]:
    """Load APPS problems from HuggingFace.

    Args:
        split: Dataset split to load (e.g., "train", "test").
        max_problems: Maximum number of problems to return.
        difficulty: Filter by difficulty ("introductory", "interview", "competition").
        min_tests: Minimum number of test cases required.

    Returns:
        List of Problem objects.
    """
    ds = load_dataset("codeparrot/apps", split=split, trust_remote_code=True)
    problems: list[Problem] = []
    for idx, row in enumerate(ds):
        if difficulty and row.get("difficulty", "") != difficulty:
            continue
        p = _parse_row(row, idx)
        if p is None:
            continue
        if len(p.test_cases) < min_tests:
            continue
        if max_test_cases is not None and len(p.test_cases) > max_test_cases:
            p.test_cases = p.test_cases[:max_test_cases]
        problems.append(p)
        if max_problems and len(problems) >= max_problems:
            break
    logger.info("Loaded %d APPS problems from split=%s", len(problems), split)
    return problems
