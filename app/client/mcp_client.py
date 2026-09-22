"""User-side stdio MCP client for the FastMCP GitHub server."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, ToolMessage

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class UserSideGitHubClient:
    """Discover and execute GitHub FastMCP tools on behalf of an LLM."""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm
        self.session: ClientSession | None = None
        self._stdio_context: Any | None = None
        self._session_context: Any | None = None
        self._read_stream: Any | None = None
        self._write_stream: Any | None = None

    async def connect(self) -> None:
        """Start the FastMCP server over stdio and complete the MCP handshake."""
        if self.session is not None:
            return
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp.github_server"],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self._stdio_context = stdio_client(server)
        self._read_stream, self._write_stream = await self._stdio_context.__aenter__()
        self._session_context = ClientSession(self._read_stream, self._write_stream)
        self.session = await self._session_context.__aenter__()
        await self.session.initialize()

    async def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Discover server tools and convert them to LangChain/OpenAI format."""
        await self.connect()
        assert self.session is not None
        result = await self.session.list_tools()
        tools: List[Dict[str, Any]] = []
        for tool in result.tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                }
            )
        return tools

    @staticmethod
    def _tool_result_text(result: Any) -> str:
        parts: list[str] = []
        for item in getattr(result, "content", []) or []:
            text = getattr(item, "text", None)
            parts.append(str(text if text is not None else item))
        if parts:
            return "\n".join(parts)
        structured = getattr(result, "structuredContent", None)
        return json.dumps(structured, ensure_ascii=False) if structured is not None else str(result)

    def _get_llm(self) -> Any:
        if self.llm is not None:
            return self.llm
        try:
            from app.gateway import gateway_chat_model
        except ModuleNotFoundError:
            from gateway import gateway_chat_model
        return gateway_chat_model

    async def chat(self, user_query: str) -> str:
        """Run a tool-aware chat turn and return the final grounded answer."""
        await self.connect()
        assert self.session is not None
        tools = await self.get_tools_for_llm()
        model = self._get_llm()
        try:
            tool_model = model.bind_tools(tools)
        except (AttributeError, NotImplementedError) as error:
            raise RuntimeError("The configured LLM does not support tool binding") from error

        messages: list[Any] = [HumanMessage(content=user_query)]
        for _ in range(8):
            response = await tool_model.ainvoke(messages)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", []) or []
            if not tool_calls:
                return str(getattr(response, "content", response))

            for call in tool_calls:
                name = call.get("name", "")
                arguments = call.get("args", {}) or {}
                try:
                    result = await self.session.call_tool(name, arguments)
                    content = self._tool_result_text(result)
                except Exception as error:
                    content = json.dumps({"error": str(error), "tool": name})
                messages.append(ToolMessage(content=content, tool_call_id=call.get("id", name)))

        raise RuntimeError("MCP tool-calling loop exceeded the maximum of 8 rounds")

    async def close(self) -> None:
        """Close the MCP session and stdio transport cleanly."""
        if self._session_context is not None:
            await self._session_context.__aexit__(None, None, None)
        if self._stdio_context is not None:
            await self._stdio_context.__aexit__(None, None, None)
        self.session = None
        self._session_context = None
        self._stdio_context = None
        self._read_stream = None
        self._write_stream = None
