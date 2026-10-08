"""MCP Tool Registry providing unified tool dispatching for GitHub analysis agents."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

try:
    from app.tools import github_tools
except ModuleNotFoundError:
    from tools import github_tools

logger = logging.getLogger("MCPRegistry")


class MCPToolRegistry:
    """Dispatches MCP tool invocations to underlying read-only tool functions."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {
            "github_get_repository": lambda **kwargs: github_tools.get_repository(kwargs.get("owner", ""), kwargs.get("repo", "")),
            "github_get_branches": lambda **kwargs: github_tools.get_branches(kwargs.get("owner", ""), kwargs.get("repo", "")),
            "github_get_repository_tree": lambda **kwargs: github_tools.get_tree(kwargs.get("owner", ""), kwargs.get("repo", ""), kwargs.get("branch"), kwargs.get("recursive", True)),
            "github_get_readme": lambda **kwargs: github_tools.get_readme(kwargs.get("owner", ""), kwargs.get("repo", ""), kwargs.get("branch")),
            "github_get_file_content": lambda **kwargs: github_tools.get_file(kwargs.get("owner", ""), kwargs.get("repo", ""), kwargs.get("path", kwargs.get("file_path", "")), kwargs.get("branch")),
            "github_get_files": lambda **kwargs: github_tools.get_files(kwargs.get("owner", ""), kwargs.get("repo", ""), kwargs.get("paths", []), kwargs.get("branch")),
            "github_get_commits": lambda **kwargs: github_tools.get_commits(kwargs.get("owner", ""), kwargs.get("repo", ""), kwargs.get("path"), kwargs.get("branch"), kwargs.get("per_page", 10)),
            "github_compare_branches": lambda **kwargs: github_tools.compare_branches(kwargs.get("owner", ""), kwargs.get("repo", ""), kwargs.get("base", "main"), kwargs.get("head", "")),
            "github_get_pull_requests": lambda **kwargs: github_tools.get_pull_requests(kwargs.get("owner", ""), kwargs.get("repo", ""), kwargs.get("state", "all"), kwargs.get("per_page", 5)),
            "github_get_issues": lambda **kwargs: github_tools.get_issues(kwargs.get("owner", ""), kwargs.get("repo", ""), kwargs.get("state", "all"), kwargs.get("per_page", 5)),
        }

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool function."""
        if tool_name not in self._tools:
            return {"error": f"Tool '{tool_name}' not found in registry."}
        try:
            return self._tools[tool_name](**arguments)
        except Exception as err:
            logger.warning("Tool '%s' execution failed: %s", tool_name, err)
            return {"error": str(err)}


mcp_registry = MCPToolRegistry()
