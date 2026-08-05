import re
from pathlib import Path
from typing import List


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
