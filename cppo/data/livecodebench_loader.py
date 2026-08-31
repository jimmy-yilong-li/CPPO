"""Loader for the livecodebench/code_generation_lite dataset.

The HuggingFace ``datasets`` package no longer supports dataset scripts in
newer versions, while LiveCodeBench still ships ``code_generation_lite.py``.
Load the JSONL files from the Hub directly so this loader keeps working without
pinning an old ``datasets`` version.
"""

from __future__ import annotations

import base64
import json
import logging
import pickletools
import zlib
from typing import Any

from huggingface_hub import hf_hub_download

from cppo.data.problem import Problem, TestCase

logger = logging.getLogger(__name__)

_REPO_ID = "livecodebench/code_generation_lite"

_ALLOWED_FILES: dict[str, list[str]] = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
    "release_latest": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}
for _idx in range(1, 7):
    _ALLOWED_FILES[f"v{_idx}"] = [f"test{_idx}.jsonl" if _idx != 1 else "test.jsonl"]
for _start in range(1, 7):
    for _end in range(_start + 1, 7):
        _ALLOWED_FILES[f"v{_start}_v{_end}"] = [
            f"test{idx}.jsonl" if idx != 1 else "test.jsonl"
            for idx in range(_start, _end + 1)
        ]


def _extract_pickled_string(payload: bytes) -> str | None:
    """Extract a simple pickled string without executing pickle opcodes."""
    try:
        for _op, arg, _pos in pickletools.genops(payload):
            if isinstance(arg, str):
                return arg
    except Exception:
        return None
    return None


def _loads_test_blob(raw: Any) -> list[dict[str, Any]]:
    """Parse LCB test-case blobs.

    Public tests are JSON strings. Private tests are commonly base64(zlib()) of
    a pickled JSON string. We avoid ``pickle.loads`` and extract the string
    opcode with ``pickletools`` instead.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, dict)]
    if not isinstance(raw, str):
        return []

    raw = raw.strip()
    if not raw:
        return []
    candidates = [raw]
    if not raw.startswith(("[", "{")):
        try:
            decoded = zlib.decompress(base64.b64decode(raw))
            extracted = _extract_pickled_string(decoded)
            if extracted:
                candidates.append(extracted)
        except Exception:
            pass

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            return [t for t in parsed if isinstance(t, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    return []


def _parse_test_cases(row: dict[str, Any], *, include_hidden_tests: bool = False) -> list[TestCase]:
    """Parse test cases from a LiveCodeBench row."""
    test_cases: list[TestCase] = []

    for field_name in ("public_test_cases", "private_test_cases"):
        is_hidden = field_name == "private_test_cases"
        if is_hidden and not include_hidden_tests:
            continue
        for t in _loads_test_blob(row.get(field_name, "")):
            inp = t.get("input", "")
            out = t.get("output", t.get("expected_output", ""))
            test_cases.append(TestCase(
                input=str(inp),
                expected_output=str(out),
                is_hidden=is_hidden,
            ))

    return test_cases


def _parse_row(row: dict[str, Any], idx: int, *, include_hidden_tests: bool = False) -> Problem | None:
    """Parse a single LiveCodeBench row into a Problem."""
    test_cases = _parse_test_cases(row, include_hidden_tests=include_hidden_tests)

    # Determine io_mode from starter_code / metadata
    starter_code = row.get("starter_code", "") or None
    entry_point = None

    # LiveCodeBench problems can be call_based if they have starter_code
    if starter_code:
        io_mode = "call_based"
        # Try to extract function name from starter_code
        import re
        match = re.search(r"def\s+(\w+)\s*\(", starter_code)
        if match:
            entry_point = match.group(1)
    else:
        io_mode = "stdin"

    question_id = row.get("question_id", idx)

    return Problem(
        id=f"livecodebench_{question_id}",
        prompt=row.get("question_content", row.get("prompt", "")),
        test_cases=test_cases,
        domain="code",
        difficulty=str(row.get("difficulty", "unknown")),
        source="livecodebench",
        io_mode=io_mode,
        entry_point=entry_point,
        starter_code=starter_code,
        metadata={
            "question_title": row.get("question_title", ""),
            "contest_date": row.get("contest_date", ""),
            "platform": row.get("platform", ""),
        },
    )


def load_livecodebench(
    split: str = "test",
    max_problems: int | None = None,
    min_tests: int = 1,
    version_tag: str = "release_v5",
    max_test_cases: int | None = 10,
    include_hidden_tests: bool = False,
) -> list[Problem]:
    """Load LiveCodeBench problems from HuggingFace.

    Args:
        split: Dataset split (default "test").
        max_problems: Maximum number of problems to return.
        min_tests: Minimum number of test cases required.
        version_tag: Dataset version tag (default "release_v5").

    Returns:
        List of Problem objects.
    """
    if split != "test":
        raise ValueError("LiveCodeBench code_generation_lite currently exposes only split='test'")
    if version_tag not in _ALLOWED_FILES:
        allowed = ", ".join(sorted(_ALLOWED_FILES))
        raise ValueError(f"unknown LiveCodeBench version_tag={version_tag!r}; allowed: {allowed}")

    problems: list[Problem] = []
    idx = 0
    for filename in _ALLOWED_FILES[version_tag]:
        path = hf_hub_download(_REPO_ID, filename=filename, repo_type="dataset")
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                p = _parse_row(row, idx, include_hidden_tests=include_hidden_tests)
                idx += 1
                if p is None:
                    continue
                if len(p.test_cases) < min_tests:
                    continue
                if max_test_cases is not None and len(p.test_cases) > max_test_cases:
                    p.test_cases = p.test_cases[:max_test_cases]
                problems.append(p)
                if max_problems and len(problems) >= max_problems:
                    break
        if max_problems and len(problems) >= max_problems:
            break
    logger.info(
        "Loaded %d LiveCodeBench problems from split=%s version=%s",
        len(problems), split, version_tag,
    )
    return problems
