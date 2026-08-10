"""Market-data MCP server (Yahoo Finance backed).

A real Model Context Protocol server exposing market data as four tools, backed
by Yahoo Finance via `tools.yfinance_client` (yfinance) — keyless, no daily cap.
Run as a stdio subprocess; the specialist agents connect via `tools.mcp_client`.

Run directly:
    python -m finsight.mcp_servers.yfinance_server

Configure in an MCP-aware host (Claude Desktop, Cursor) by adding:
    {
      "mcpServers": {
        "yahoo-finance": {
          "command": "python",
          "args": ["-m", "finsight.mcp_servers.yfinance_server"]
        }
      }
    }

Tools exposed:
    yf_overview          — company fundamentals
    yf_daily             — daily OHLCV
    yf_income_statement  — quarterly/annual income statement
    yf_news_sentiment    — recent news headlines
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from finsight.tools import yfinance_client as md

log = logging.getLogger("mcp.yfinance")

server: Server = Server("yahoo-finance")


@server.list_tools()
async def _list_tools() -> list[Tool]:
    """Tool schemas advertised to MCP clients (Claude, LangGraph adapter, ...)."""
    return [
        Tool(
            name="yf_overview",
            description=(
                "Fetch company fundamentals (sector, market cap, P/E, EPS, profit margins, "
                "52-week range, etc.) for a US-listed equity ticker."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"}
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="yf_daily",
            description="Daily OHLCV time series (~1y of history).",
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
            name="yf_income_statement",
            description="Annual + quarterly income statement (revenue, COGS, op income, net income).",
            inputSchema={
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        ),
        Tool(
            name="yf_news_sentiment",
            description=(
                "Recent news headlines for a ticker (Yahoo Finance provides headlines; "
                "no per-article sentiment scores). Multiple tickers comma-separated."
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
    """Dispatch a tool call to the underlying Yahoo Finance client.

    Every response is wrapped as TextContent JSON so MCP clients get a uniform
    contract. Errors are returned as `{"ok": false, "error": ...}` rather than
    raised, so the calling agent can decide how to degrade.
    """
    dispatch = {
        "yf_overview": lambda: md.overview(symbol=arguments["symbol"]),
        "yf_daily": lambda: md.daily(
            symbol=arguments["symbol"],
            outputsize=arguments.get("outputsize", "compact"),
        ),
        "yf_income_statement": lambda: md.income_statement(symbol=arguments["symbol"]),
        "yf_news_sentiment": lambda: md.news_sentiment(
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
