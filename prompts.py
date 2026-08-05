from __future__ import annotations

from typing import List

from repository import collect_relevant_files, format_context


def build_chat_prompt(project_path: str, task: str, history: List[str] | None = None) -> str:
    relevant_files = collect_relevant_files(project_path, task)
    context = format_context(relevant_files, project_path)

    history_lines: List[str] = []
    if history:
        history_lines.append("Previous conversation:")
        for message in history[-10:]:
            history_lines.append(f"- {message.strip()}")
        history_lines.append("")

    history_text = "\n".join(history_lines)
    prompt = f"""
You are an expert repository code assistant.

Use only the relevant files below for this task.
Do not load the entire repository unless explicitly requested.

{history_text}
Relevant files:
{context}

User task:
{task}

Instructions:
- Read only the relevant project context.
- Explain architecture or identify the bug using the relevant files.
- If changes are needed, provide precise and actionable edits.
- If you need to use tools, return ONLY a JSON object with a top-level "tool_calls" array.
  Each tool call must look like: {{"tool":"write_file","arguments":{{"path":"path/to/file","content":"full file contents"}}}}.
- Do not return shell snippets, pseudo-code tool calls, or plain-English tool instructions.
- Keep the answer concise.
- Mention only important files directly related to the request.
- Do not invent functions, classes, or APIs that are not present.
"""

    return prompt.strip() + "\n"
