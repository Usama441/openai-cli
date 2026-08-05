from __future__ import annotations

from typing import List


class Planner:
    def plan(self, task: str) -> List[str]:
        text = task.lower()

        if "read" in text or "explain" in text or "overview" in text:
            return ["inspect repository", "summarize relevant files", "answer question"]

        if "fix" in text or "bug" in text or "error" in text or "broken" in text:
            return ["identify root cause", "patch affected files", "verify with tests or commands"]

        if "add" in text or "implement" in text or "create" in text:
            return ["inspect relevant files", "implement feature", "verify behavior"]

        return ["inspect repository", "gather relevant context", "respond with actionable guidance"]
