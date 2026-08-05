from __future__ import annotations

import json
import os
from typing import Any, List

from openai import OpenAI

from memory import ConversationMemory
from planner import Planner
from prompts import build_chat_prompt
from tools.filesystem import execute_tool_call, parse_tool_calls_from_model_text, validate_tool_call


class RepoAgent:
    def __init__(self, project_path: str, model: str | None = None, base_url: str | None = None):
        self.project_path = project_path
        self.model = model or os.getenv("OPENAI_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.memory = ConversationMemory()
        self.planner = Planner()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy"), base_url=self.base_url)

    def _query_model(self, prompt: str) -> str:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Missing OPENAI_API_KEY. Set it in your .env file or environment before running the editor assistant.")

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=1024,
        )
        return response.output_text

    def run_task(self, user_task: str) -> str:
        self.memory.add_user(user_task)
        history = self.memory.as_history_strings(limit=10)
        prompt = build_chat_prompt(self.project_path, user_task, history)

        for iteration in range(1, 8):
            try:
                model_text = self._query_model(prompt)
            except ValueError:
                return (
                    "The agent cannot execute because OPENAI_API_KEY is not set. "
                    "Set it in the environment or .env before running the model-driven loop."
                )

            tool_calls = parse_tool_calls_from_model_text(model_text)
            if not tool_calls:
                self.memory.add_assistant(model_text.strip())
                return model_text.strip()

            tool_results = self._run_tool_step(tool_calls)
            summary = self._summarize_tool_results(tool_results)
            self.memory.add_assistant(json.dumps({"tool_results": tool_results, "summary": summary}, ensure_ascii=False))

            if any(result.get("tool") in {"write_file", "delete_file", "mkdir", "run_command"} for result in tool_results):
                ok, verification = self._verify_task(user_task)
                if not ok:
                    prompt = build_chat_prompt(
                        self.project_path,
                        user_task,
                        self.memory.as_history_strings(limit=10) + [
                            f"Tool results summary:\n{summary}",
                            f"Verification failed:\n{verification}",
                            "Please fix the failure and continue with the smallest remaining patch.",
                        ],
                    )
                    continue

            prompt = build_chat_prompt(
                self.project_path,
                user_task,
                self.memory.as_history_strings(limit=10) + [f"Tool results summary:\n{summary}", "Task status: patch applied and verified, or no further tool steps are required."],
            )
            return f"Completed tool-driven patch cycle.\n\n{summary}"

        return "Agent loop reached the maximum number of iterations without a final answer."

    def _summarize_tool_results(self, tool_results: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for result in tool_results:
            tool_name = result.get("tool", "unknown")
            ok = result.get("ok", False)
            details = []
            if result.get("path"):
                details.append(f"path={result['path']}")
            if result.get("stdout"):
                details.append(f"stdout={result['stdout'].strip()[:200]}")
            if result.get("stderr"):
                details.append(f"stderr={result['stderr'].strip()[:200]}")
            if result.get("error"):
                details.append(f"error={result['error']}")
            if not details:
                details.append("no extra details")
            status = "ok" if ok else "failed"
            lines.append(f"- {tool_name}: {status} | {'; '.join(details)}")
        return "\n".join(lines)

    def _run_tool_step(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in tool_calls:
            validated = validate_tool_call(call)
            if not validated["ok"]:
                results.append({"tool": call.get("tool"), "ok": False, "error": validated["error"]})
                continue
            result = execute_tool_call(validated)
            results.append({"tool": validated["tool"], "ok": result.get("ok", False), **result})
        return results

    def _verify_task(self, task: str) -> tuple[bool, str]:
        lower = task.lower()
        if any(term in lower for term in ["build", "test", "pytest", "cargo", "npm", "run"]):
            commands = []
            if "pytest" in lower:
                commands.append("pytest -q")
            elif "cargo" in lower:
                commands.append("cargo test")
            elif "npm" in lower:
                commands.append("npm test -- --runInBand")
            else:
                commands.append("python3 -m compileall .")

            for command in commands:
                from tools.terminal import run_command
                result = run_command(command, cwd=self.project_path)
                if not result["ok"]:
                    return False, result.get("stderr", result.get("stdout", "command failed"))
            return True, "verification succeeded"

        return True, "no explicit verification command required"

    def handle(self, user_task: str) -> str:
        self.memory.add_user(user_task)
        steps = self.planner.plan(user_task)
        history = self.memory.as_history_strings(limit=10)
        prompt = build_chat_prompt(self.project_path, user_task, history)

        for iteration in range(1, 8):
            try:
                model_text = self._query_model(prompt)
            except ValueError:
                return (
                    f"Plan: {', '.join(steps)}\n\n"
                    f"History-aware prompt prepared for this task.\n\n"
                    f"Prompt preview:\n{prompt[:800]}"
                )

            tool_calls = parse_tool_calls_from_model_text(model_text)
            if not tool_calls:
                self.memory.add_assistant(model_text.strip())
                return model_text.strip()

            tool_results = self._run_tool_step(tool_calls)
            summary = self._summarize_tool_results(tool_results)
            self.memory.add_assistant(json.dumps({"tool_results": tool_results, "summary": summary}, ensure_ascii=False))

            if any(result.get("tool") == "write_file" for result in tool_results):
                ok, verification = self._verify_task(user_task)
                if not ok:
                    prompt = build_chat_prompt(
                        self.project_path,
                        user_task,
                        self.memory.as_history_strings(limit=10) + [f"Tool results summary:\n{summary}", f"Verification failed:\n{verification}"]
                    )
                    continue

            prompt = build_chat_prompt(
                self.project_path,
                user_task,
                self.memory.as_history_strings(limit=10) + [f"Tool results summary:\n{summary}"]
            )

            if iteration >= 7:
                return f"Tool execution loop reached max iterations.\n{summary}"

        return "Agent loop finished without a final response."


def main() -> None:
    project = "."
    agent = RepoAgent(project)
    while True:
        task = input("Prompt: ").strip()
        if not task:
            continue
        if task.lower() in {"exit", "quit", "q"}:
            break
        print(agent.handle(task))


if __name__ == "__main__":
    main()
