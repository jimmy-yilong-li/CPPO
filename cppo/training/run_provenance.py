"""Shared run provenance utilities for CPPO and warmup training scripts.

Both `scripts/03_run_warmup.py` and `scripts/04_run_cppo.py` write a
`run_config.json` provenance snapshot and guard against accidentally
re-using a run directory. Keeping
the implementation in one place ensures both scripts capture the same
git state, problems hash, and CLI args.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def assert_run_dir_clean(*, save_dir: Path, allow_resume: bool) -> None:
    """Fail fast if save_dir already contains any artifacts. Without
    this guard, a re-run with the same output path can mix model files,
    append-mode logs, and provenance from different attempts.

    Pass allow_resume=True only to intentionally bypass the guard for
    a directory whose contents were inspected. This does not restore
    optimizer state or implement checkpoint resume.
    """
    if not save_dir.exists():
        return
    if not save_dir.is_dir():
        existing = [save_dir.name]
    else:
        existing = sorted(p.name for p in save_dir.iterdir())
    if not existing:
        return
    if allow_resume:
        logger.warning(
            "run dir %s is non-empty (found %s); --allow-resume set, "
            "bypassing the dirty-dir guard. No optimizer/checkpoint resume "
            "is performed.",
            save_dir, ", ".join(existing),
        )
        return
    raise RuntimeError(
        f"refusing to run: {save_dir} is non-empty (found "
        f"{', '.join(existing)}). Either pick a fresh output dir / "
        f"--run-id (default is UTC timestamp, which is always unique) "
        f"or pass --allow-resume to intentionally bypass the guard."
    )


def _git(args_list: list[str]) -> "str | None":
    """Run `git <args>` from the caller's cwd. Returns decoded stdout
    or None if git is unavailable or the call fails."""
    try:
        return subprocess.check_output(
            ["git", *args_list], stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:
        return None


def _capture_git_state() -> tuple["str | None", "bool | None", "str | None"]:
    """Capture (git_sha, git_dirty, git_status_short). Returns
    (None, None, None) outside a git repo. git_dirty uses porcelain
    status so it catches untracked files, which `git diff` would miss.
    """
    sha_raw = _git(["rev-parse", "HEAD"])
    git_sha = sha_raw.strip() if sha_raw else None
    if git_sha is None:
        return None, None, None
    porcelain = _git(["status", "--porcelain"])
    if porcelain is None:
        return git_sha, None, None
    git_dirty = bool(porcelain.strip())
    # Cap to a reasonable size; full status can be huge during large
    # refactors and we only need it for forensic reads.
    git_status_short = porcelain[:4000]
    return git_sha, git_dirty, git_status_short


def _capture_problems_digest(
    problems_path: "str | None",
) -> tuple["str | None", "int | None", "list[str] | None"]:
    """Return (sha256, n_records, ordered_problem_ids) for the given
    problems jsonl file. n_records counts non-empty JSONL rows; the ids
    list records rows that expose `id` or `problem_id`. All three are
    None if no path is given. Returns (None, None, None) silently if
    the file does not exist — the run itself will fail loudly elsewhere
    when it tries to open it."""
    if not problems_path:
        return None, None, None
    try:
        raw = Path(problems_path).read_bytes()
    except FileNotFoundError:
        return None, None, None
    sha256 = hashlib.sha256(raw).hexdigest()
    n_records = 0
    ids: list[str] = []
    for line in raw.decode().splitlines():
        line = line.strip()
        if not line:
            continue
        n_records += 1
        try:
            rec = json.loads(line)
        except Exception:
            # Malformed line — the run will fail at load time, and the
            # sha256 above already captures the file as-was for forensics.
            continue
        # Real Problem.to_dict() emits `id`; some audit/judge files
        # use `problem_id`. Accept either so the digest works against
        # both shapes without forcing callers to normalize first.
        pid = rec.get("id")
        if pid is None:
            pid = rec.get("problem_id")
        if pid is not None:
            ids.append(str(pid))
    return sha256, n_records, ids


def write_run_provenance(
    *,
    save_dir: Path,
    args,
    run_id: str,
    snapshot_name: str = "run_config.json",
    **cfg_kwargs,
) -> None:
    """Persist a provenance snapshot under save_dir/snapshot_name.

    The snapshot answers the question "what was actually run here" for
    a future reader, capturing: run_id, UTC timestamp, git SHA + dirty
    state + porcelain status, CLI args, the problems file's sha256 +
    record count + ordered problem_ids, plus any caller-provided
    config dicts (via cfg_kwargs, e.g. base_cfg=..., cppo_cfg=...).

    cfg_kwargs become top-level snapshot keys so 06 (CPPO) and 05
    (warmup) can each pin their own config shape (`cppo_cfg`,
    `warmup_cfg`) without forcing one to mirror the other.
    """
    git_sha, git_dirty, git_status_short = _capture_git_state()
    problems_sha256, problems_n, problems_ids = _capture_problems_digest(
        getattr(args, "problems", None),
    )

    cli_args = {k: v for k, v in vars(args).items() if not k.startswith("_")}

    snapshot = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "git_status_short": git_status_short,
        "cli_args": cli_args,
        **cfg_kwargs,
        "problems_sha256": problems_sha256,
        "problems_n": problems_n,
        "problems_ids": problems_ids,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / snapshot_name
    with open(out, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
