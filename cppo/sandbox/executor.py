import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from cppo.data.problem import Problem
from cppo.sandbox.code_extract import extract_python_code


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    error: str | None


@dataclass
class VerificationResult:
    passed: bool
    pass_rate: float
    tests_passed: int
    tests_total: int
    details: list[dict]


def _set_limits():
    try:
        import resource
    except ImportError:
        return
    # RLIMIT_AS not available on macOS; use RLIMIT_RSS as fallback
    mem_bytes = 256 * 1024 * 1024
    for limit_type in ("RLIMIT_AS", "RLIMIT_RSS"):
        if hasattr(resource, limit_type):
            try:
                resource.setrlimit(getattr(resource, limit_type), (mem_bytes, mem_bytes))
                break
            except (ValueError, OSError):
                pass
    # RLIMIT_NPROC may not be settable in all environments
    if hasattr(resource, "RLIMIT_NPROC"):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ValueError, OSError):
            pass


def execute_code(code: str, stdin: str = "", timeout: int = 10) -> ExecutionResult:
    try:
        compile(code, "<solution>", "exec")
    except SyntaxError:
        return ExecutionResult(False, "", "SyntaxError", "syntax_error")

    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = os.path.join(tmpdir, "solution.py")
        with open(code_path, "w") as f:
            f.write(code)

        proc = subprocess.Popen(
            [sys.executable, code_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmpdir,
            start_new_session=True,
            preexec_fn=_set_limits if os.name == "posix" else None,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        try:
            stdout, stderr = proc.communicate(input=stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
            proc.wait()
            return ExecutionResult(False, "", "", "timeout")

        if proc.returncode != 0:
            return ExecutionResult(False, stdout, stderr, "runtime_error")
        return ExecutionResult(True, stdout, stderr, None)


def _execute_call_based(code: str, entry_point: str, test_input: str, timeout: int) -> ExecutionResult:
    import base64
    try:
        args = json.loads(test_input)
    except json.JSONDecodeError:
        args = [test_input]

    args_b64 = base64.b64encode(json.dumps(args).encode()).decode()
    wrapper = f"""{code}

import json as _json, base64 as _b64
_args = _json.loads(_b64.b64decode('{args_b64}').decode())
_result = {entry_point}(*_args)
print(repr(_result))
"""
    return execute_code(wrapper, stdin="", timeout=timeout)


def _normalize_output(s: str) -> str:
    return s.strip()


def verify_solution(problem: Problem, solution: str, timeout: int = 10) -> VerificationResult:
    code = extract_python_code(solution)
    if problem.io_mode == "call_based" and problem.entry_point:
        return _verify_call_based(problem, code, timeout)
    return _verify_stdin(problem, code, timeout)


def _verify_stdin(problem: Problem, code: str, timeout: int) -> VerificationResult:
    if not problem.test_cases:
        return VerificationResult(False, 0.0, 0, 0, [])
    details = []
    passed = 0
    for tc in problem.test_cases:
        r = execute_code(code, stdin=tc.input, timeout=timeout)
        ok = r.success and _normalize_output(r.stdout) == _normalize_output(tc.expected_output)
        if ok:
            passed += 1
        details.append({"passed": ok, "error": r.error, "got": r.stdout[:200] if r.success else None})
    total = len(problem.test_cases)
    return VerificationResult(passed == total, passed / total, passed, total, details)


def _verify_call_based(problem: Problem, code: str, timeout: int) -> VerificationResult:
    if not problem.test_cases or not problem.entry_point:
        return VerificationResult(False, 0.0, 0, 0, [])
    details = []
    passed = 0
    for tc in problem.test_cases:
        r = _execute_call_based(code, problem.entry_point, tc.input, timeout)
        ok = r.success and _normalize_output(r.stdout) == _normalize_output(tc.expected_output)
        if ok:
            passed += 1
        details.append({"passed": ok, "error": r.error, "got": r.stdout[:200] if r.success else None})
    total = len(problem.test_cases)
    return VerificationResult(passed == total, passed / total, passed, total, details)
