"""MCP client wrapper.

Connects to our MCP servers as a subprocess over stdio and re-exposes their
tools as `ToolResult`-returning async callables. Agents call this — they
never talk to Alpha Vantage HTTP directly.

Lifecycle:

    async with mcp_client() as client:
        snapshot = await client.call("av_overview", symbol="AAPL")
        bars     = await client.call("av_daily", symbol="AAPL")

We use a single MCP server (`alpha_vantage_server`) for now. Additional servers
(e.g. SEC) plug in by adding entries to `MCP_SERVERS`.
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from finsight.logging_setup import get_logger
from finsight.tools.base import ToolResult

log = get_logger(__name__)


# Registry: tool name -> which server provides it
MCP_SERVERS: dict[str, StdioServerParameters] = {
    "alpha_vantage": StdioServerParameters(
        command=sys.executable,
        args=["-m", "finsight.mcp_servers.alpha_vantage_server"],
        env=None,
    ),
}


@dataclass
class MCPClient:
    session: ClientSession
    available_tools: list[str]

    async def call(self, name: str, **arguments: Any) -> ToolResult:
        if name not in self.available_tools:
            return ToolResult.failure(f"tool not exposed by server: {name}")
        log.debug("mcp.call name=%s args=%s", name, arguments)
        res = await self.session.call_tool(name, arguments=arguments)

        # MCP returns a list of content blocks; we standardize on a single
        # JSON-encoded TextContent (see alpha_vantage_server).
        if not res.content:
            return ToolResult.failure("mcp: empty response")
        text = getattr(res.content[0], "text", None)
        if text is None:
            return ToolResult.failure("mcp: non-text content")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            return ToolResult.failure(f"mcp: bad json ({e})")

        return ToolResult(
            ok=bool(payload.get("ok")),
            data=payload.get("data"),
            error=payload.get("error"),
            meta=payload.get("meta") or {},
        )


@asynccontextmanager
async def mcp_session(server: str = "alpha_vantage") -> AsyncIterator[MCPClient]:
    """Open a stdio MCP session against the named server.

    The server process is spawned on enter and torn down on exit. For long-lived
    connections (production), we'd pool these; for now per-request is fine —
    a stdio handshake is sub-100ms.
    """
    params = MCP_SERVERS[server]
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            log.info("mcp.connected server=%s tools=%s", server, names)
            yield MCPClient(session=session, available_tools=names)
