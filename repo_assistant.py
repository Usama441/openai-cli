import argparse
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


def supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")


def style(text: str, color: str | None = None, bold: bool = False) -> str:
    if not supports_color():
        return text
    prefix = ""
    if bold:
        prefix += Colors.BOLD
    if color:
        prefix += color
    return f"{prefix}{text}{Colors.RESET}"


def print_banner() -> None:
    print(style("+" + "-" * 70 + "+", Colors.CYAN, True))
    print(style("|" + " Repo-Aware AI Assistant ".center(70) + "|", Colors.CYAN, True))
    print(style("|" + " Focused repo scanning ".center(70) + "|", Colors.YELLOW))
    print(style("+" + "-" * 70 + "+", Colors.CYAN, True))


def print_help_legend() -> None:
    print(style("Commands:", Colors.YELLOW, True))
    print("  help   show this help")
    print("  exit   quit the session")
    print("  q      shortcut for exit")
    print()


def print_chat_header(title: str) -> None:
    label = f" {title} "
    width = max(18, len(label) + 4)
    border = "─" * width
    print(style(f"┌{border}┐", Colors.GREEN, True))
    print(style(f"│{label.center(width)}│", Colors.CYAN, True))
    print(style(f"└{border}┘", Colors.GREEN, True))


def print_separator(title: str | None = None) -> None:
    if title:
        print(style(f"\n+{'-' * 18} {title} {'-' * 18}+", Colors.GREEN, True))
    else:
        print(style("\n" + "-" * 60, Colors.CYAN))


def print_status(message: str) -> None:
    print(style(f"Status: {message}", Colors.YELLOW, True))


def print_error(message: str) -> None:
    print(style("\nERROR: " + message, Colors.RED, True))


def wrap_text(text: str, width: int = 100) -> str:
    lines: List[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        if len(paragraph) <= width:
            lines.append(paragraph)
            continue
        wrapped = textwrap.wrap(
            paragraph,
            width=width,
            replace_whitespace=False,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if wrapped:
            lines.extend(wrapped)
    return "\n".join(lines)


def load_env_file(path: str | Path | None = None) -> None:
    candidates: List[Path] = []
    if path is not None:
        candidates.append(Path(path).expanduser())

    candidates.extend(
        [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent / ".env",
        ]
    )

    seen: set[Path] = set()
    for env_path in candidates:
        resolved = env_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        if not resolved.exists() or not resolved.is_file():
            continue

        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

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

    if path.name.endswith(".pyc") or path.name.endswith(".class") or path.name.endswith(".dll"):
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


def normalize_prompt(prompt: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9_./-]+", prompt.lower())
    return [token for token in tokens if len(token) > 2]


def score_candidate(path: Path, prompt_tokens: List[str]) -> int:
    rel_name = path.name.lower()
    rel_path = str(path).lower()
    score = 0

    for token in prompt_tokens:
        if token in rel_name:
            score += 6
        if token in rel_path:
            score += 4

    if rel_name in {"readme.md", "readme.txt", "package.json", "requirements.txt", "pyproject.toml", "dockerfile"}:
        score += 8

    for keyword in ["api", "auth", "login", "db", "config", "routes", "models", "controller", "service", "main"]:
        if keyword in rel_path:
            score += 2

    return score


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


def build_chat_prompt(project_path: str | Path, task: str, history: List[str] | None = None) -> str:
    relevant_files = collect_relevant_files(project_path, task)
    context = format_context(relevant_files, project_path)

    history_lines: List[str] = []
    if history:
        history_lines.append("Previous conversation:")
        for message in history[-10:]:
            history_lines.append(f"- {message.strip()}")
        history_lines.append("")

    return f"""
You are an expert repository code assistant.

Use only the relevant files below for this task.
Do not load the entire repository unless the user explicitly requests a full-project dump.

{'\n'.join(history_lines)}Relevant files:
{context}

User task:
{task}

Instructions:
- Read only the relevant project context.
- Explain architecture or identify the bug using the relevant files.
- If changes are needed, provide precise and actionable edits.
- Keep the answer concise.
- Mention only important files directly related to the request.
- Do not invent functions, classes, or APIs that are not present.
""".strip() + "\n"


def ask_code_editor(project_path: str | Path, task: str, model: str | None = None, base_url: str | None = None, history: List[str] | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY. Set it in your .env file or environment before running the editor assistant.")

    effective_model = model or os.getenv("OPENAI_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    effective_base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    client = OpenAI(api_key=api_key, base_url=effective_base_url)
    prompt = build_chat_prompt(project_path, task, history)

    response = client.responses.create(
        model=effective_model,
        input=prompt,
        max_output_tokens=1024,
    )
    return response.output_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo-aware AI code assistant with targeted file selection.")
    parser.add_argument("project_path", nargs="?", default=None, help="Path to the repository or app folder")
    parser.add_argument("task", nargs="?", default=None, help="Task or question to send to the assistant")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"), help="Model name to use")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"), help="OpenAI-compatible base URL")
    parser.add_argument("--dry-run", action="store_true", help="Show the selected relevant files without calling the API")

    args = parser.parse_args()

    print_banner()
    print_help_legend()

    if args.project_path is None:
        print(style("Enter the project path, or type 'exit' to quit.", Colors.YELLOW))
        args.project_path = input(style("Project path: ", Colors.CYAN, True)).strip()
        if not args.project_path or args.project_path.lower() in {"exit", "quit", "q"}:
            print(style("Goodbye!", Colors.GREEN, True))
            return

    conversation_history: List[str] = []

    if args.task is None:
        while True:
            print_chat_header("You")
            task = input(style("Prompt: ", Colors.CYAN, True)).strip()
            if not task:
                print_status("Empty prompt. Try again.")
                continue
            if task.lower() in {"exit", "quit", "q"}:
                print(style("Goodbye!", Colors.GREEN, True))
                return
            if task.lower() == "help":
                print_help_legend()
                continue

            conversation_history.append(f"User: {task}")
            print_status("Scanning relevant files...")
            files = collect_relevant_files(args.project_path, task)
            if args.dry_run:
                print(format_context(files, args.project_path))
                print_separator()
                continue

            try:
                print_status("AI thinking...")
                result = ask_code_editor(
                    project_path=args.project_path,
                    task=task,
                    model=args.model,
                    base_url=args.base_url,
                    history=conversation_history[:-1],
                )
            except Exception as exc:
                print_error(str(exc))
                print_separator()
                continue

            conversation_history.append(f"Assistant: {result.strip()}")
            print_chat_header("AI")
            print(wrap_text(result))
            print_separator()
        return

    print_chat_header("You")
    print(style(f"Prompt: {args.task}", Colors.CYAN, True))
    if args.task.lower() in {"help", "?"}:
        print_help_legend()
        return

    conversation_history.append(f"User: {args.task}")
    files = collect_relevant_files(args.project_path, args.task)
    if args.dry_run:
        print(format_context(files, args.project_path))
        return

    try:
        print_status("AI thinking...")
        result = ask_code_editor(
            project_path=args.project_path,
            task=args.task,
            model=args.model,
            base_url=args.base_url,
            history=conversation_history[:-1],
        )
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1) from exc

    print_chat_header("AI")


if __name__ == "__main__":
    main()
