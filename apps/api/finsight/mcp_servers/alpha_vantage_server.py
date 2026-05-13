"""Alpha Vantage MCP server.

A real Model Context Protocol server that exposes Alpha Vantage as four tools.
Run as a stdio subprocess; LangGraph agents connect via `tools.mcp_client`.

Run directly:
    python -m finsight.mcp_servers.alpha_vantage_server

Configure in an MCP-aware host (Claude Desktop, Cursor) by adding:
    {
      "mcpServers": {
        "alpha-vantage": {
          "command": "python",
          "args": ["-m", "finsight.mcp_servers.alpha_vantage_server"]
        }
      }
    }

Tools exposed:
    av_overview          — company fundamentals
    av_daily             — daily OHLCV
    av_income_statement  — quarterly/annual income statement
    av_news_sentiment    — news with sentiment scores
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from finsight.tools import alpha_vantage as av

log = logging.getLogger("mcp.alpha_vantage")

server: Server = Server("alpha-vantage")


@server.list_tools()
async def _list_tools() -> list[Tool]:
    """Tool schemas advertised to MCP clients (Claude, LangGraph adapter, ...)."""
    return [
        Tool(
            name="av_overview",
            description=(
                "Fetch company fundamentals (sector, market cap, P/E, EPS, profit margins, "
                "52-week range, etc.) for a US-listed equity ticker."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol, e.g. AAPL",
                    }
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="av_daily",
            description="Daily OHLCV time series. `outputsize=compact` returns last 100 days.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "outputsize": {
                        "type": "string",
                        "enum": ["compact", "full"],
                        "default": "compact",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="av_income_statement",
            description="Annual + quarterly income statement (revenue, COGS, op income, net income).",
            inputSchema={
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        ),
        Tool(
            name="av_news_sentiment",
            description=(
                "Latest news with per-article sentiment scores and ticker relevance scores. "
                "Multiple tickers comma-separated."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tickers": {"type": "string"},
                    "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
                },
                "required": ["tickers"],
            },
        ),
    ]


@server.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Dispatch a tool call to the underlying Alpha Vantage client.

    Every response is wrapped as TextContent JSON so MCP clients get a uniform
    contract. Errors are returned as `{"ok": false, "error": ...}` rather than
    raised, so the calling agent can decide how to degrade.
    """
    dispatch = {
        "av_overview": lambda: av.overview(symbol=arguments["symbol"]),
        "av_daily": lambda: av.daily(
            symbol=arguments["symbol"],
            outputsize=arguments.get("outputsize", "compact"),
        ),
        "av_income_statement": lambda: av.income_statement(symbol=arguments["symbol"]),
        "av_news_sentiment": lambda: av.news_sentiment(
            tickers=arguments["tickers"],
            limit=int(arguments.get("limit", 20)),
        ),
    }
    if name not in dispatch:
        payload = {"ok": False, "error": f"unknown tool: {name}"}
    else:
        result = await dispatch[name]()
        payload = {
            "ok": result.ok,
            "data": result.data,
            "error": result.error,
            "meta": result.meta,
        }
    return [TextContent(type="text", text=json.dumps(payload, default=str))]


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
