import re

# Fenced code blocks: ```python, ```Python, ```python3, ```py, or ``` (no lang).
# Case-insensitive language tag.
_FENCED_RE = re.compile(
    r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# When NO fenced block, fall back: try to find the smallest prefix to drop
# such that the remainder compiles. We only drop leading prose lines (no `=`,
# no `(`, no Python keywords). This avoids the old bug of cutting variable
# definitions before the first `for/if/...` line.
_PYTHON_LINE_HINTS = (
    "import ", "from ", "def ", "class ", "if ", "elif ", "else:", "for ", "while ",
    "try:", "except", "finally:", "with ", "return", "yield", "lambda ", "raise ",
    "assert ", "pass", "break", "continue", "#", "@",
)


def _looks_like_code(line: str) -> bool:
    """Heuristic: is this line plausibly Python (not prose)?"""
    s = line.strip()
    if not s:
        return True  # blank lines are fine inside code
    # Common Python statement starts.
    if any(s.startswith(p) for p in _PYTHON_LINE_HINTS):
        return True
    # Assignment / augmented assignment / function call etc.
    # Conservative: contains '=' or '(' or ends with ':' and has no Chinese / English prose markers.
    if "=" in s or s.endswith(":") or s.endswith(")"):
        # Reject obvious prose: starts with capital letter and contains many spaces and no operators
        # (heuristic, not perfect)
        return True
    return False


def extract_python_code(text: str) -> str:
    """Extract a Python code block from model output.

    Strategy:
      1. If any fenced ``` block exists (any case for python/py/python3/no-lang),
         return the LONGEST one.
      2. Otherwise, try `compile(text)` — if the whole thing is valid Python,
         return it as-is.
      3. Otherwise, drop leading prose lines and try compile again, line by line,
         until either compile succeeds or we run out.
      4. Final fallback: return the original text (let executor surface errors).
    """
    # 1. Fenced blocks
    blocks = _FENCED_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()

    text = text.strip()
    if not text:
        return text

    # 2. Whole text compiles as-is?
    try:
        compile(text, "<solution>", "exec")
        return text
    except SyntaxError:
        pass

    # 3. Drop leading prose lines until the rest compiles.
    lines = text.split("\n")
    for i in range(len(lines)):
        # Only consider candidates that *look like* code starts
        first = lines[i].strip()
        if not first:
            continue
        if not _looks_like_code(first):
            continue
        candidate = "\n".join(lines[i:])
        try:
            compile(candidate, "<solution>", "exec")
            return candidate
        except SyntaxError:
            continue

    # 4. Fallback: return original (executor will report syntax_error)
    return text
