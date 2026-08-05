from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from search import normalize_prompt, score_candidate

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".next",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "coverage",
    ".DS_Store",
    "tmp",
    "temp",
}

IGNORED_FILE_NAMES = {
    ".env",
    ".env.local",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    ".DS_Store",
}

MAX_TEXT_BYTES = 200_000
MAX_CONTEXT_FILES = 12
MAX_CONTEXT_CHARS = 18_000


def is_ignored_path(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = ()

    if any(part in IGNORED_DIRS for part in rel_parts):
        return True
    if path.name in IGNORED_FILE_NAMES:
        return True
    if path.name.startswith(".env"):
        return True
    if path.name.endswith((".pyc", ".class", ".dll")):
        return True
    return False


def safe_read_text(path: Path) -> str | None:
    try:
        content = path.read_bytes()
    except (OSError, PermissionError):
        return None

    if len(content) > MAX_TEXT_BYTES:
        return None

    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return content.decode("latin-1")
        except UnicodeDecodeError:
            return None


def collect_relevant_files(project_path: str | Path, prompt: str) -> Dict[str, str]:
    root = Path(project_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Project path is not a directory: {root}")

    prompt_tokens = normalize_prompt(prompt)
    file_scores: List[Tuple[Path, int]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if is_ignored_path(path, root):
            continue

        text = safe_read_text(path)
        if text is None:
            continue

        score = score_candidate(path, prompt_tokens)
        if prompt_tokens and "read" in prompt_tokens and path.name.lower().startswith(("readme", "package", "requirements", "pyproject", "docker")):
            score += 8

        file_scores.append((path, score))

    file_scores.sort(key=lambda item: item[1], reverse=True)
    selected: Dict[str, str] = {}

    for path, _ in file_scores[:MAX_CONTEXT_FILES]:
        rel = path.relative_to(root).as_posix()
        text = safe_read_text(path)
        if text is None:
            continue
        if len(text) > 2500:
            text = text[:2500] + "\n... [truncated for context]"
        selected[rel] = text

    if not selected:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if is_ignored_path(path, root):
                continue

            text = safe_read_text(path)
            if text is None:
                continue

            rel = path.relative_to(root).as_posix()
            selected[rel] = text[:2500]
            if len(selected) >= MAX_CONTEXT_FILES:
                break

    return selected


def format_context(files: Dict[str, str], project_path: str | Path) -> str:
    if not files:
        return f"Project root: {project_path}\nNo readable relevant files were found."

    lines = [f"Project root: {project_path}", f"Relevant files: {len(files)}", ""]
    used_chars = 0

    for file_name, content in files.items():
        snippet = content.strip().replace("\r", "")
        if len(snippet) > 1800:
            snippet = snippet[:1800] + "\n... [truncated]"
        if used_chars + len(snippet) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - used_chars
            if remaining <= 0:
                break
            snippet = snippet[:remaining]
        lines.append(f"===== {file_name} =====")
        lines.append(snippet)
        lines.append("")
        used_chars += len(snippet)

    return "\n".join(lines)


def build_project_snapshot(project_path: str | Path) -> Dict[str, str]:
    root = Path(project_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Project path is not a directory: {root}")

    selected: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or is_ignored_path(path, root):
            continue
        text = safe_read_text(path)
        if text is None:
            continue
        rel = path.relative_to(root).as_posix()
        selected[rel] = text[:2500] + ("\n... [truncated for context]" if len(text) > 2500 else "")
        if len(selected) >= MAX_CONTEXT_FILES:
            break
    return selected
