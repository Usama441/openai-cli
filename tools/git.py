from __future__ import annotations

from tools.terminal import run_command


def git_status(cwd: str) -> dict:
    return run_command("git status --short", cwd=cwd)


def git_diff(cwd: str) -> dict:
    return run_command("git --no-pager diff", cwd=cwd)
