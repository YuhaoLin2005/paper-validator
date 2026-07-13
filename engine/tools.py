"""Tool registry — 4 tools: read_file, write_file, bash, grep (imported by query, main).

Minimal set for paper validation experiments. No external dependencies.
"""

from __future__ import annotations

import subprocess, os, re as _re
from pathlib import Path
from typing import Any, Callable, Optional


class ToolResult:
    def __init__(self, content: str, is_error: bool = False):
        self.content = content
        self.is_error = is_error


def _read_file(file_path: str, offset: int = 0, limit: Optional[int] = None) -> ToolResult:
    try:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return ToolResult(f"File not found: {file_path}", is_error=True)
        if not p.is_file():
            return ToolResult(f"Not a file: {file_path}", is_error=True)
        if p.stat().st_size > 1_048_576:
            return ToolResult(f"File too large ({p.stat().st_size} bytes)", is_error=True)
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if offset:
            lines = lines[offset:]
        if limit:
            lines = lines[:limit]
        result = "".join(f"{i+1+offset}\t{l}" for i, l in enumerate(lines))
        return ToolResult(result or "(empty file)")
    except PermissionError:
        return ToolResult(f"Permission denied: {file_path}", is_error=True)
    except Exception as e:
        return ToolResult(f"Read error: {e}", is_error=True)


def _write_file(file_path: str, content: str) -> ToolResult:
    try:
        p = Path(file_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(f"Written {len(content)} bytes to {file_path}")
    except Exception as e:
        return ToolResult(f"Write error: {e}", is_error=True)


def _bash(command: str, timeout: int = 30) -> ToolResult:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=os.getcwd(),
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return ToolResult(output.strip() or "(no output)")
    except subprocess.TimeoutExpired:
        return ToolResult(f"Timed out after {timeout}s", is_error=True)
    except Exception as e:
        return ToolResult(f"Bash error: {e}", is_error=True)


def _grep(pattern: str, path: str = ".", glob_pattern: str = "*") -> ToolResult:
    import glob as glob_mod
    try:
        regex = _re.compile(pattern)
        base = Path(path).expanduser().resolve()
        if not base.exists():
            return ToolResult(f"Path not found: {path}", is_error=True)
        matches = []
        search_root = base if base.is_dir() else base.parent
        for filepath in glob_mod.glob(str(search_root / "**" / glob_pattern), recursive=True):
            fp = Path(filepath)
            if not fp.is_file():
                continue
            try:
                if fp.stat().st_size > 500_000:
                    continue
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{fp}:{i}: {line.rstrip()}")
                            if len(matches) >= 100:
                                break
                if len(matches) >= 100:
                    break
            except (PermissionError, OSError):
                continue
        if not matches:
            return ToolResult(f"No matches for '{pattern}' in {path}")
        return ToolResult("\n".join(matches))
    except _re.error as e:
        return ToolResult(f"Invalid regex: {e}", is_error=True)
    except Exception as e:
        return ToolResult(f"Grep error: {e}", is_error=True)


TOOL_DEFINITIONS = [
    {"type": "function", "function": {"name": "read_file",
        "description": "Read a file from the local filesystem.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "offset": {"type": "integer"},
            "limit": {"type": "integer"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "write_file",
        "description": "Write content to a file.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["file_path", "content"]}}},
    {"type": "function", "function": {"name": "bash",
        "description": "Execute a bash command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout": {"type": "integer"}},
            "required": ["command"]}}},
    {"type": "function", "function": {"name": "grep",
        "description": "Search for a regex pattern in files.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"},
            "glob_pattern": {"type": "string"}}, "required": ["pattern"]}}},
]

TOOL_FUNCTIONS: dict[str, Callable[..., ToolResult]] = {
    "read_file": _read_file, "write_file": _write_file,
    "bash": _bash, "grep": _grep,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return ToolResult(f"Unknown tool: {name}", is_error=True)
    try:
        return fn(**arguments)
    except TypeError as e:
        return ToolResult(f"Argument error: {e}", is_error=True)
    except Exception as e:
        return ToolResult(f"Execution error: {e}", is_error=True)
