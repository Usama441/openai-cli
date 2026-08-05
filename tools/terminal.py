from __future__ import annotations

import subprocess
from typing import Any


def run_command(command: str, cwd: str | None = None) -> dict[str, Any]:
    result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
        "cwd": cwd,
    }
