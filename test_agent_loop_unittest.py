import json
import unittest

from agent import RepoAgent
from tools.filesystem import execute_tool_call


class AgentLoopTests(unittest.TestCase):
    def test_run_command_tool_is_supported(self):
        result = execute_tool_call({
            "tool": "run_command",
            "arguments": {"command": "python3 -c \"print('ok')\""},
        })
        self.assertTrue(result["ok"])
        self.assertIn("ok", result["stdout"])

    def test_agent_summary_includes_tool_results(self):
        agent = RepoAgent(".")
        summary = agent._summarize_tool_results([
            {"tool": "write_file", "ok": True, "path": "example.txt"},
            {"tool": "run_command", "ok": False, "stderr": "pytest failed"},
        ])
        self.assertIn("write_file", summary)
        self.assertIn("pytest failed", summary)

    def test_run_task_executes_write_then_verifies(self):
        agent = RepoAgent(".")
        calls = iter([
            json.dumps({
                "tool_calls": [
                    {
                        "tool": "write_file",
                        "arguments": {
                            "path": "tmp_agent_patch.py",
                            "content": "print('patched')\n",
                        },
                    }
                ]
            }),
            json.dumps({"final": "done"}),
        ])

        agent._query_model = lambda prompt: next(calls)
        result = agent.run_task("fix the script and run python3 -m compileall .")

        self.assertIn("Completed tool-driven patch cycle", result)
        self.assertTrue(agent.project_path.endswith("."))

    def test_legacy_action_tool_payloads_are_normalized_and_executed(self):
        create_result = execute_tool_call({"action": "create_folder", "path": "legacy_test_dir"})
        self.assertTrue(create_result["ok"])

        list_result = execute_tool_call({"action": "list_dir", "path": "legacy_test_dir"})
        self.assertTrue(list_result["ok"])
        self.assertIn("legacy_test_dir", str(list_result.get("path", "")))


if __name__ == "__main__":
    unittest.main()
