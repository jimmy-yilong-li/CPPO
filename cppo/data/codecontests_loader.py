"""Loader for the deepmind/code_contests dataset."""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset

from cppo.data.problem import Problem, TestCase

logger = logging.getLogger(__name__)

# Python language IDs in code_contests: 1 = PYTHON, 3 = PYTHON3
# (NOT 4 — that is Java!)
_PYTHON_LANG_IDS = {1, 3}


def _extract_python_solutions(row: dict[str, Any]) -> list[str]:
    """Extract Python solutions from the row's solutions field."""
    solutions = row.get("solutions", {})
    if not solutions:
        return []
    languages = solutions.get("language", [])
    solution_texts = solutions.get("solution", [])
    return [
        solution_texts[i]
        for i in range(min(len(languages), len(solution_texts)))
        if languages[i] in _PYTHON_LANG_IDS
    ]


def _parse_row(row: dict[str, Any], idx: int, max_generated: int = 5) -> Problem | None:
    """Parse a single code_contests row into a Problem, or None if invalid.

    Only loads stdin/stdout problems (skips file-based I/O).
    Combines public_tests and up to max_generated generated_tests.
    """
    # Skip problems with file-based I/O
    if row.get("input_file") or row.get("output_file"):
        return None

    test_cases: list[TestCase] = []

    # Public tests
    public = row.get("public_tests", {})
    if public:
        inputs = public.get("input", [])
        outputs = public.get("output", [])
        for inp, out in zip(inputs, outputs):
            test_cases.append(TestCase(input=inp, expected_output=out))

    # Generated tests (up to max_generated)
    generated = row.get("generated_tests", {})
    if generated:
        gen_inputs = generated.get("input", [])
        gen_outputs = generated.get("output", [])
        for inp, out in list(zip(gen_inputs, gen_outputs))[:max_generated]:
            test_cases.append(TestCase(input=inp, expected_output=out, is_hidden=True))

    python_solutions = _extract_python_solutions(row)

    return Problem(
        id=f"codecontests_{idx}",
        prompt=row.get("description", ""),
        test_cases=test_cases,
        domain="code",
        difficulty=str(row.get("difficulty", "unknown")),
        source="codecontests",
        io_mode="stdin",
        metadata={
            "name": row.get("name", ""),
            "source": row.get("source", ""),
            "solutions": python_solutions,
            "cf_rating": row.get("cf_rating", None),
        },
    )


def load_codecontests(
    split: str = "test",
    max_problems: int | None = None,
    min_tests: int = 1,
    max_generated_tests: int = 5,
    max_test_cases: int | None = 10,
) -> list[Problem]:
    """Load Code Contests problems from HuggingFace.

    Args:
        split: Dataset split to load (e.g., "train", "valid", "test").
        max_problems: Maximum number of problems to return.
        min_tests: Minimum number of test cases required.
        max_generated_tests: Maximum number of generated tests to include per problem.

    Returns:
        List of Problem objects.
    """
    ds = load_dataset("deepmind/code_contests", split=split, trust_remote_code=True)
    problems: list[Problem] = []
    for idx, row in enumerate(ds):
        p = _parse_row(row, idx, max_generated=max_generated_tests)
        if p is None:
            continue
        if len(p.test_cases) < min_tests:
            continue
        if max_test_cases is not None and len(p.test_cases) > max_test_cases:
            p.test_cases = p.test_cases[:max_test_cases]
        problems.append(p)
        if max_problems and len(problems) >= max_problems:
            break
    logger.info("Loaded %d CodeContests problems from split=%s", len(problems), split)
    return problems
