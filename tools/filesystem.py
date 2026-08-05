from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REQUIRED_ARGS = {
    "write_file": {"path", "content"},
    "read_file": {"path"},
    "list_directory": {"path"},
    "delete_file": {"path"},
    "mkdir": {"path"},
    "run_command": {"command"},
}

LEGACY_ACTION_ALIASES = {
    "create_folder": "mkdir",
    "create_dir": "mkdir",
    "mkdir": "mkdir",
    "list_dir": "list_directory",
    "list_directory": "list_directory",
    "create_file": "write_file",
    "write_file": "write_file",
    "read_file": "read_file",
    "delete_file": "delete_file",
    "remove_file": "delete_file",
    "run_command": "run_command",
}


def normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(call, dict):
        return call

    if "tool" not in call and "action" in call:
        action_name = call["action"]
        tool_name = LEGACY_ACTION_ALIASES.get(action_name, action_name)
        if tool_name == "mkdir":
            arguments = {"path": call.get("path")}
        elif tool_name == "write_file":
            arguments = {"path": call.get("path"), "content": call.get("content", "")}
        elif tool_name == "run_command":
            arguments = {"command": call.get("command"), "cwd": call.get("cwd")}
        else:
            arguments = {"path": call.get("path")}
        return {"tool": tool_name, "arguments": arguments}

    if "tool" in call and "arguments" not in call and "args" not in call:
        if isinstance(call.get("path") or call.get("command") or call.get("content"), (str, int, float)):
            mapped = {"tool": call["tool"], "arguments": {}}
            if "path" in call:
                mapped["arguments"]["path"] = call["path"]
            if "content" in call:
                mapped["arguments"]["content"] = call["content"]
            if "command" in call:
                mapped["arguments"]["command"] = call["command"]
            if "cwd" in call:
                mapped["arguments"]["cwd"] = call["cwd"]
            return mapped

    return call


def parse_tool_calls_from_model_text(text: str | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(text, dict):
        if isinstance(text.get("tool_calls"), list):
            return [normalize_tool_call(item) for item in text["tool_calls"]]
        if text.get("tool") or text.get("action"):
            return [normalize_tool_call(text)]
        return []

    if not text or not text.strip():
        return []

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        candidates = re.findall(r"\{.*?\}", cleaned, flags=re.DOTALL)
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    if isinstance(parsed.get("tool_calls"), list):
                        return parsed["tool_calls"]
                    if parsed.get("tool"):
                        return [parsed]
            except json.JSONDecodeError:
                continue
        return []

    if isinstance(data, dict):
        if isinstance(data.get("tool_calls"), list):
            return data["tool_calls"]
        if data.get("tool"):
            return [data]
        if data.get("done") is True and "final" in data:
            return []
        if data.get("final"):
            return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    return []


def validate_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {"ok": False, "error": "Tool call must be a JSON object"}

    tool_name = call.get("tool") or LEGACY_ACTION_ALIASES.get(call.get("action"), call.get("action"))
    if not tool_name:
        return {"ok": False, "error": "Tool call is missing a 'tool' or 'action' field"}

    args = call.get("arguments", call.get("args", {}))
    if not args and "action" in call:
        args = {}
        if "path" in call:
            args["path"] = call["path"]
        if "content" in call:
            args["content"] = call["content"]
        if "command" in call:
            args["command"] = call["command"]
        if "cwd" in call:
            args["cwd"] = call["cwd"]
    if not isinstance(args, dict):
        return {"ok": False, "error": f"Tool '{tool_name}' arguments must be an object"}

    required = REQUIRED_ARGS.get(tool_name)
    if required is None:
        return {"ok": False, "error": f"Unknown tool: {tool_name}"}

    missing = sorted(required - set(args.keys()))
    if missing:
        return {"ok": False, "error": f"Tool '{tool_name}' is missing required arguments: {missing}"}

    return {"ok": True, "tool": tool_name, "arguments": args}


def write_file(path: str, content: str) -> dict[str, Any]:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(file_path), "bytes": len(content.encode("utf-8"))}


def read_file(path: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    return {"ok": True, "path": path, "content": text}


def list_directory(path: str) -> dict[str, Any]:
    root = Path(path)
    return {"ok": True, "path": str(root), "items": [p.name for p in sorted(root.iterdir())]}


def delete_file(path: str) -> dict[str, Any]:
    file_path = Path(path)
    file_path.unlink(missing_ok=True)
    return {"ok": True, "path": str(file_path)}


def mkdir(path: str) -> dict[str, Any]:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "path": str(directory)}


def execute_tool_call(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            call = json.loads(raw)
        except json.JSONDecodeError:
            return {"ok": False, "error": "Invalid JSON tool call"}
    else:
        call = raw

    validated = validate_tool_call(call)
    if not validated["ok"]:
        return validated

    tool_name = validated["tool"]
    args = validated["arguments"]

    if tool_name == "write_file":
        return write_file(args["path"], args["content"])
    if tool_name == "read_file":
        return read_file(args["path"])
    if tool_name == "list_directory":
        return list_directory(args["path"])
    if tool_name == "delete_file":
        return delete_file(args["path"])
    if tool_name == "mkdir":
        return mkdir(args["path"])
    if tool_name == "run_command":
        from tools.terminal import run_command
        return run_command(args["command"], args.get("cwd"))

    return {"ok": False, "error": f"Unknown tool: {tool_name}"}
