import json
import tempfile
from pathlib import Path

import repo_assistant
from agent import RepoAgent
from repo_assistant import build_chat_prompt, build_project_snapshot
from tools.filesystem import parse_tool_calls_from_model_text


def test_build_project_snapshot_reads_repo_files_and_readme():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")
        (root / "README.md").write_text("# Demo project\n", encoding="utf-8")
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text("ignore me\n", encoding="utf-8")
        (root / "node_modules").mkdir()
        (root / "node_modules" / "pkg.js").write_text("console.log('ignore')\n", encoding="utf-8")

        snapshot = build_project_snapshot(root)

        assert "README.md" in snapshot
        assert "src/app.py" in snapshot
        assert "node_modules/pkg.js" not in snapshot
        assert ".git/config" not in snapshot
        assert "Demo project" in snapshot["README.md"]
        assert "print('hello')" in snapshot["src/app.py"]


def test_build_chat_prompt_keeps_previous_prompt_context():
    history = [
        "User: read the project",
        "Assistant: here is the repo overview",
        "User: explain the auth flow",
    ]

    prompt = build_chat_prompt("/tmp/project", "what files are involved in login?", history)

    assert "Previous conversation" in prompt
    assert "read the project" in prompt
    assert "explain the auth flow" in prompt
    assert "what files are involved in login?" in prompt


def test_parse_tool_calls_from_model_text():
    payload = '{"tool_calls":[{"tool":"read_file","arguments":{"path":"README.md"}}]}'
    calls = parse_tool_calls_from_model_text(payload)

    assert calls[0]["tool"] == "read_file"
    assert calls[0]["arguments"]["path"] == "README.md"


def test_agent_loop_executes_model_tool_requests(monkeypatch, tmp_path):
    calls = iter([
        '{"tool_calls":[{"tool":"read_file","arguments":{"path":"README.md"}}]}',
        '{"final":"done"}',
    ])

    agent = RepoAgent(str(tmp_path))
    monkeypatch.setattr(agent, "_query_model", lambda prompt: next(calls))
    result = agent.handle("read the project")

    assert "done" in result


def test_repo_assistant_executes_real_tool_calls(monkeypatch, tmp_path):
    def fake_ask_code_editor(project_path, task, model=None, base_url=None, history=None):
        return json.dumps({
            "tool_calls": [{
                "tool": "mkdir",
                "arguments": {"path": str(tmp_path / "usama")},
            }]
        })

    monkeypatch.setattr(repo_assistant, "ask_code_editor", fake_ask_code_editor)

    result = repo_assistant.run_task_project(str(tmp_path), "create a folder named usama")

    assert "mkdir" in result.lower()
    assert (tmp_path / "usama").is_dir()
