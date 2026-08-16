"""Model Context Protocol (MCP) tool server & registry for GitHub and Web capabilities."""

from __future__ import annotations

import json
from typing import Any, Callable

try:
    from app.tools.github_tools import (
        compare_branches,
        get_branches,
        get_commit,
        get_commit_diff,
        get_commits,
        get_file,
        get_file_history,
        get_files,
        get_issues,
        get_pull_requests,
        get_readme,
        get_repository,
        get_repository_tree,
        parse_github_url,
    )
    from app.tools.duckduckgo_tools import search_web_ddg
except ModuleNotFoundError:
    from tools.github_tools import (
        compare_branches,
        get_branches,
        get_commit,
        get_commit_diff,
        get_commits,
        get_file,
        get_file_history,
        get_files,
        get_issues,
        get_pull_requests,
        get_readme,
        get_repository,
        get_repository_tree,
        parse_github_url,
    )
    from tools.duckduckgo_tools import search_web_ddg


class MCPToolRegistry:
    """Registry maintaining tool definitions and execution dispatchers."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._register_default_tools()

    def register(self, name: str, func: Callable[..., Any], description: str, parameters: dict[str, Any]) -> None:
        self._tools[name] = func
        self._schemas[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    def _register_default_tools(self) -> None:
        self.register(
            "github_get_repository",
            get_repository,
            "Fetch general metadata and stats for a GitHub repository.",
            {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner or organization."},
                    "repo": {"type": "string", "description": "Repository name."},
                },
                "required": ["owner", "repo"],
            },
        )
        self.register(
            "github_get_repository_tree",
            get_repository_tree,
            "Fetch the full recursive file tree structure of the repository.",
            {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner."},
                    "repo": {"type": "string", "description": "Repository name."},
                    "branch": {"type": "string", "description": "Optional branch name."},
                    "recursive": {"type": "boolean", "description": "Whether to fetch recursively."},
                },
                "required": ["owner", "repo"],
            },
        )
        self.register(
            "github_get_readme",
            get_readme,
            "Fetch and decode the README file from the repository.",
            {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner."},
                    "repo": {"type": "string", "description": "Repository name."},
                    "branch": {"type": "string", "description": "Optional branch name."},
                },
                "required": ["owner", "repo"],
            },
        )
        self.register(
            "github_get_file",
            get_file,
            "Fetch and decode content of a specific source or config file in the repository.",
            {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner."},
                    "repo": {"type": "string", "description": "Repository name."},
                    "path": {"type": "string", "description": "Relative file path."},
                    "branch": {"type": "string", "description": "Optional branch name."},
                },
                "required": ["owner", "repo", "path"],
            },
        )
        self.register(
            "github_get_files",
            get_files,
            "Fetch contents of multiple files in batch.",
            {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner."},
                    "repo": {"type": "string", "description": "Repository name."},
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "List of file paths."},
                    "branch": {"type": "string", "description": "Optional branch name."},
                },
                "required": ["owner", "repo", "paths"],
            },
        )
        self.register(
            "github_get_branches",
            get_branches,
            "List active branches in the repository.",
            {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner."},
                    "repo": {"type": "string", "description": "Repository name."},
                },
                "required": ["owner", "repo"],
            },
        )
        self.register(
            "github_get_commits",
            get_commits,
            "List recent commits in the repository or affecting a specific path.",
            {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner."},
                    "repo": {"type": "string", "description": "Repository name."},
                    "path": {"type": "string", "description": "Optional file path filter."},
                    "branch": {"type": "string", "description": "Optional branch name."},
                    "per_page": {"type": "integer", "description": "Number of commits to return."},
                },
                "required": ["owner", "repo"],
            },
        )
        self.register(
            "github_get_commit_diff",
            get_commit_diff,
            "Fetch the unified diff of changes made in a specific commit.",
            {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner."},
                    "repo": {"type": "string", "description": "Repository name."},
                    "sha": {"type": "string", "description": "Commit SHA or hash."},
                },
                "required": ["owner", "repo", "sha"],
            },
        )
        self.register(
            "github_compare_branches",
            compare_branches,
            "Compare two branches to find commits and file changes between them.",
            {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner."},
                    "repo": {"type": "string", "description": "Repository name."},
                    "base": {"type": "string", "description": "Base branch name."},
                    "head": {"type": "string", "description": "Head branch name."},
                },
                "required": ["owner", "repo", "base", "head"],
            },
        )
        self.register(
            "github_get_issues",
            get_issues,
            "List recent issues filed on the repository.",
            {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner."},
                    "repo": {"type": "string", "description": "Repository name."},
                    "state": {"type": "string", "description": "Issue state ('open', 'closed', 'all')."},
                    "per_page": {"type": "integer", "description": "Number of issues to return."},
                },
                "required": ["owner", "repo"],
            },
        )
        self.register(
            "web_search",
            search_web_ddg,
            "Search the web using DuckDuckGo for library/framework documentation or context.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                    "max_results": {"type": "integer", "description": "Maximum results to return."},
                },
                "required": ["query"],
            },
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool definitions."""
        return list(self._schemas.values())

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch tool invocation safely."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' is not registered in the MCP tool server.")
        func = self._tools[name]
        try:
            return func(**arguments)
        except Exception as error:
            return {"error": str(error), "tool": name, "status": "failed"}


mcp_registry = MCPToolRegistry()
