"""Domain-aware PLAN / SOLVE prompt templates and plan-tuple parser."""

from __future__ import annotations

import re
import string


def to_chat_text(tokenizer, user_prompt: str, *, enable_thinking: bool | None = None) -> str:
    """Wrap a plain user prompt with the model's chat template.

    Use this everywhere we construct a prompt for an Instruct/chat-tuned model
    before tokenizing. Falls back to the raw prompt if the tokenizer has no
    chat template (e.g. base models without one).
    """
    if getattr(tokenizer, "chat_template", None):
        kwargs = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        except TypeError:
            if enable_thinking is None:
                raise
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return user_prompt


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

_LABELS = list(string.ascii_uppercase)  # A, B, C, ...


def _label_list(k: int) -> list[str]:
    """Return the first *k* uppercase labels, e.g. ['A', 'B', 'C', 'D']."""
    return _LABELS[:k]


def _label_block(k: int) -> str:
    """Format placeholder lines like 'A: ...\nB: ...\n...'."""
    return "\n".join(f"{lbl}:" for lbl in _label_list(k))


# ---------------------------------------------------------------------------
# PLAN prompt
# ---------------------------------------------------------------------------

def _last_label(k: int) -> str:
    return _label_list(k)[-1]


def make_plan_contract(domain: str = "code", k: int = 4) -> str:
    """Return the shared pass@K attempt-allocation plan contract.

    This text is intentionally shared by Qwen sampling and teacher-positive
    generation so the RM training distribution and deployment planner prompt
    do not drift apart.
    """
    task = "programming problem"
    last_label = _last_label(k)
    labels = _label_list(k)
    bad_roles = [
        "parse/setup input",
        "compute an intermediate value",
        "loop/check candidates",
        "return/output the result",
    ][:k]
    bad_shape = " ".join(
        f"{label}: {role}." for label, role in zip(labels, bad_roles, strict=False)
    )
    return (
        "Requirements:\n"
        f"- Write exactly {k} solver attempts, using plain labels A: through {last_label}:.\n"
        "- Do not use markdown headings or bullets for the labels; use plain label lines only.\n"
        "- Do not include code, imports, function definitions, fenced code blocks, or final answers.\n"
        "- Return only the plan text, with no preface, commentary, explanation, JSON, list, tuple, or quotes.\n"
        "- Each attempt must be exactly one concise sentence on its own label line.\n"
        "- Keep each attempt short: no more than 45 words per label.\n"
        "- Do not include substeps, derivations, implementation walkthroughs, or multi-paragraph methods.\n"
        f"- Stop immediately after the {last_label}: method; do not append any extra analysis or commentary.\n"
        "- The goal is to maximize solver pass@K: the tuple should allocate the solver's attempts to useful, "
        "problem-specific routes that make at least one success more likely.\n"
        "- Each label must be a self-contained solver attempt, not one step in a shared walkthrough.\n"
        "- Do not split one solution into separate setup, compute, loop/check, and return/output steps.\n"
        "- Prefer diversity when it is useful: different modeling ideas, data structures, complexity tradeoffs, "
        "edge-case handling, or implementation styles can all be valuable if they change how the solver would "
        "write or check the solution.\n"
        "- Acceptable attempt types include useful_variant, limited_but_useful, and "
        "solver_actionable_single_strategy: a variant or narrow route is acceptable when it gives the solver "
        "a concrete implementation path, robustness check, or different likely error pattern.\n"
        "- A useful variant must change the solver's likely implementation path, robustness check, complexity "
        "tradeoff, or failure mode. If two attempts would lead to essentially the same code with only wording "
        "changes, they are harmful duplicates.\n"
        "- A limited or narrow attempt is acceptable only if it gives the solver a concrete, executable path "
        "that could plausibly pass tests. If it merely fills a slot without improving solver behavior, it is "
        "filler_method.\n"
        "- Useful variants of the same core idea are allowed when they lead to meaningfully different solver "
        "implementations, robustness checks, or failure modes. For example, a direct formula, a simulation, "
        "and a binary-search wrapper may all be useful attempts only if each is genuinely feasible and offers "
        "a different path to a correct solver implementation for this exact problem.\n"
        "- Reject fail attempt types: filler_method, generic_but_fluent, wrong_task, unsupported_assumption, "
        "harmful_duplicate, and sequential_workflow.\n"
        "- Do not include harmful duplicates: repeated wording, cosmetic rewrites, or variants that would lead "
        "to essentially identical solver code with no extra robustness.\n"
        "- Only use an attempt if it fits this exact problem; do not invent unrelated algorithms.\n"
        f"- Every attempt must be viable for solving this exact {task}; do not include filler just to reach the "
        "label count.\n"
        "- Each attempt must include a concrete problem-specific hook such as a state, invariant, sorted key, "
        "graph representation, search dimension, constraint reason, formula structure or mathematical "
        "transformation, without carrying the derivation to the final answer.\n"
        "- Do not copy these requirement words into the output; write concrete strategies for the problem.\n"
        "\n"
        f"Invalid step-by-step shape to avoid: {bad_shape}"
    )


_CODE_PLAN_TEMPLATE = """\
You are an expert competitive-programming strategist.

## Problem
{problem}

Propose a minimum-viable plan tuple for this problem.
The labels are competing strategies. They are not implementation steps.

{contract}

Use this exact label skeleton:
{label_block}"""

def make_plan_prompt(problem: str, domain: str = "code", k: int = 4) -> str:
    """Build a PLAN prompt that asks the model for *k* labelled approaches."""
    label_block = _label_block(k)
    contract = make_plan_contract(domain=domain, k=k)
    return _CODE_PLAN_TEMPLATE.format(
        problem=problem, k=k, label_block=label_block, contract=contract
    )


# ---------------------------------------------------------------------------
# SOLVE prompt
# ---------------------------------------------------------------------------

_CODE_SOLVE_STDIN_TEMPLATE = """\
You are an expert Python programmer.

**Problem:**
{problem}

**Approach:** {method}

Write a complete Python solution that reads from standard input and writes to standard output.
Enclose your code in a ```python``` code block."""

_CODE_SOLVE_CALL_TEMPLATE = """\
You are an expert Python programmer.

**Problem:**
{problem}

**Approach:** {method}

Implement the Python function `{entry_point}`.
Do not use input() or print(); return the answer from the function.
Enclose your code in a ```python``` code block."""

def make_solve_prompt(
    problem: str,
    method: str,
    domain: str = "code",
    io_mode: str = "stdin",
    entry_point: str | None = None,
) -> str:
    """Build a SOLVE prompt that is io_mode-aware for code problems."""
    if io_mode == "call_based":
        if entry_point is None:
            raise ValueError("entry_point is required when io_mode='call_based'")
        return _CODE_SOLVE_CALL_TEMPLATE.format(
            problem=problem, method=method, entry_point=entry_point,
        )
    # default: stdin
    return _CODE_SOLVE_STDIN_TEMPLATE.format(problem=problem, method=method)


# ---------------------------------------------------------------------------
# Plan-tuple parser
# ---------------------------------------------------------------------------

def parse_plan_tuple(text: str, k: int = 4) -> list[str] | None:
    """Parse labelled plan methods (A:, B:, ...) from *text*.

    Tolerates common markdown decorations chat-tuned models add around the
    label: ``### A:``, ``**A:**``, ``# A.``, ``- A)`` etc.  The label letter
    must be at most a few decoration chars away from the start of a line.

    Returns a list of *k* method descriptions, or ``None`` if any expected
    label is missing.  Multi-line content (everything between one label and
    the next) is joined into a single string.
    """
    labels = _label_list(k)
    # Allow optional leading decorations: # * - > and whitespace.
    # Also allow optional ** ** wrappers around the label.
    label_alt = "|".join(labels)
    # The "decor" pattern matches optional markdown decoration around a label:
    # - leading: ###, **, *, -, >
    # - trailing closing **
    # Then a separator (:.))
    # Then optional opening ** before content (e.g. "**A:** body")
    pattern = re.compile(
        rf"(?:^|\n)[ \t]*[#>\-\*]*[ \t]*\*{{0,2}}({label_alt})\*{{0,2}}[ \t]*[:.)][ \t]*\*{{0,2}}[ \t]*"
        rf"(.*?)(?=(?:\n[ \t]*[#>\-\*]*[ \t]*\*{{0,2}}(?:{label_alt})\*{{0,2}}[ \t]*[:.)])|\Z)",
        re.DOTALL,
    )
    matches: dict[str, str] = {}
    for m in pattern.finditer(text):
        lbl = m.group(1)
        if lbl not in matches:  # keep first occurrence of each label
            content = m.group(2).strip()
            # Strip leftover ** wrapper if any
            if content.startswith("**"):
                content = content[2:].lstrip()
            if content.endswith("**"):
                content = content[:-2].rstrip()
            matches[lbl] = content
    for lbl in labels:
        if lbl not in matches:
            return None
    return [matches[lbl] for lbl in labels]
