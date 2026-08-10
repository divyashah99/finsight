"""Research tools — FinSight's capabilities exposed as LLM-callable tools.

The agentic redesign turns the old fixed-DAG steps into tools that specialist
sub-agents (and the supervisor, indirectly) can call on demand:

    get_fundamentals · get_income_statement   (FundamentalsAnalyst)
    get_price_history · compute_technicals     (TechnicalsAnalyst)
    get_news_sentiment                         (NewsAnalyst)
    search_filings                             (FilingsAnalyst — hybrid RAG + rerank)

Each tool is built bound to one ticker (via a factory closure) so the model
only decides *whether* to call it, never has to re-supply the ticker.

Citation grounding: `search_filings` appends the `SECCitation`s it surfaces to a
run-scoped `ContextVar` sink. The FilingsAnalyst node resets the sink before
running and harvests it after, so the synthesizer can build grounded
`citation_ids`. (A ContextVar keeps this isolated per request without threading
state through every tool signature.)
"""

from __future__ import annotations

from contextvars import ContextVar

from langchain_core.tools import BaseTool, tool

from finsight.agents import market as market_parse
from finsight.agents import news as news_parse
from finsight.agents.quant import compute_signals
from finsight.agents.state import SECCitation
from finsight.logging_setup import get_logger
from finsight.services import llm, reranker, sec_ingest, vectorstore
from finsight.tools.mcp_client import mcp_session

log = get_logger(__name__)

# Run-scoped collector for SEC citations surfaced by search_filings.
_citation_sink: ContextVar[list[SECCitation] | None] = ContextVar("citation_sink", default=None)


def reset_citation_sink() -> list[SECCitation]:
    """Start a fresh sink for the current run/context and return it."""
    sink: list[SECCitation] = []
    _citation_sink.set(sink)
    return sink


def _add_citation(c: SECCitation) -> None:
    sink = _citation_sink.get()
    if sink is not None:
        sink.append(c)


# ─── Fundamentals ──────────────────────────────────────────────────────────


def make_get_fundamentals(ticker: str) -> BaseTool:
    @tool
    async def get_fundamentals() -> str:
        """Get company fundamentals for the ticker under research: sector, market
        cap, P/E, EPS, profit margin, revenue, 52-week range, dividend yield."""
        async with mcp_session("yfinance") as mcp:
            res = await mcp.call("yf_overview", symbol=ticker)
        if not (res.ok and isinstance(res.data, dict) and res.data):
            return f"Fundamentals unavailable ({res.error or 'empty response'})."
        snap = market_parse._parse_overview(res.data, ticker)
        return snap.model_dump_json(exclude_none=True)

    return get_fundamentals


def make_get_income_statement(ticker: str) -> BaseTool:
    @tool
    async def get_income_statement() -> str:
        """Get the latest annual + quarterly income statement (revenue, COGS,
        operating income, net income) for the ticker under research."""
        async with mcp_session("yfinance") as mcp:
            res = await mcp.call("yf_income_statement", symbol=ticker)
        if not (res.ok and isinstance(res.data, dict)):
            return f"Income statement unavailable ({res.error or 'empty response'})."
        reports = (res.data.get("annualReports") or [])[:2] + (res.data.get("quarterlyReports") or [])[:2]
        if not reports:
            return "No income statement data returned."
        keep = ("fiscalDateEnding", "totalRevenue", "costOfRevenue", "operatingIncome", "netIncome")
        slim = [{k: r.get(k) for k in keep if k in r} for r in reports]
        import json

        return json.dumps(slim)

    return get_income_statement


# ─── Technicals ────────────────────────────────────────────────────────────


async def _fetch_bars(ticker: str):
    async with mcp_session("yfinance") as mcp:
        res = await mcp.call("yf_daily", symbol=ticker, outputsize="compact")
    if not (res.ok and isinstance(res.data, dict)):
        return [], res.error or "empty response"
    return market_parse._parse_daily(res.data), None


def make_get_price_history(ticker: str) -> BaseTool:
    @tool
    async def get_price_history() -> str:
        """Get a summary of the recent daily price history (last close, period
        high/low, number of trading days) for the ticker under research."""
        bars, err = await _fetch_bars(ticker)
        if not bars:
            return f"Price history unavailable ({err or 'empty'})."
        closes = [b.close for b in bars]
        return (
            f"{len(bars)} trading days. last_close={closes[-1]:.2f} "
            f"period_high={max(b.high for b in bars):.2f} "
            f"period_low={min(b.low for b in bars):.2f}"
        )

    return get_price_history


def make_compute_technicals(ticker: str) -> BaseTool:
    @tool
    async def compute_technicals() -> str:
        """Compute technical indicators (RSI-14, MACD, 30/90d volatility, 1/3/12m
        returns, SMA-50/200 and trend) for the ticker under research."""
        bars, err = await _fetch_bars(ticker)
        if not bars:
            return f"Cannot compute technicals ({err or 'no price data'})."
        signals = compute_signals(bars)
        return signals.model_dump_json(exclude_none=True)

    return compute_technicals


# ─── News ──────────────────────────────────────────────────────────────────


def make_get_news_sentiment(ticker: str) -> BaseTool:
    @tool
    async def get_news_sentiment() -> str:
        """Get recent news with aggregate sentiment (bullish/neutral/bearish) and
        the top headlines for the ticker under research."""
        async with mcp_session("yfinance") as mcp:
            res = await mcp.call("yf_news_sentiment", tickers=ticker, limit=25)
        if not (res.ok and isinstance(res.data, dict)):
            return f"News unavailable ({res.error or 'empty response'})."
        bundle = news_parse._parse(res.data, ticker)
        heads = "; ".join(f"{i.title} ({i.sentiment_label})" for i in bundle.items[:6])
        return (
            f"aggregate={bundle.aggregate_label or 'n/a'} "
            f"(score={bundle.aggregate_sentiment:.3f})" if bundle.aggregate_sentiment is not None
            else f"aggregate={bundle.aggregate_label or 'n/a'}"
        ) + f" · {len(bundle.items)} articles · {heads}"

    return get_news_sentiment


# ─── SEC filings (hybrid RAG + rerank) ─────────────────────────────────────


def make_search_filings(ticker: str) -> BaseTool:
    @tool
    async def search_filings(query: str) -> str:
        """Search the company's SEC filings (10-K/10-Q) for passages relevant to
        `query`. Use for risk factors, MD&A, business segments, legal proceedings,
        or anything stated in the filings. Returns the most relevant excerpts."""
        # Lazy-ingest on first use for this ticker (preserves prior behavior).
        if await vectorstore.count_for_ticker(ticker) == 0:
            log.info("search_filings.ingest ticker=%s", ticker)
            try:
                await sec_ingest.ingest_ticker(ticker, max_filings=1)
            except Exception as e:  # noqa: BLE001
                return f"No filings indexed for {ticker} and ingestion failed ({e})."
            if await vectorstore.count_for_ticker(ticker) == 0:
                return f"No filings available for {ticker}."

        dense = (await llm.embed([query]))[0]
        sparse = (await vectorstore.sparse_embed([query], query=True))[0]
        pool = await vectorstore.hybrid_search(dense, sparse, ticker, limit=12)
        hits = await reranker.rerank(query, pool, top_n=5)
        if not hits:
            return "No relevant passages found in the filings."

        blocks = []
        for h in hits:
            p = h.payload
            c = SECCitation(
                accession=p.get("accession", ""),
                form_type=p.get("form_type", ""),
                section=p.get("section"),
                filed_date=p.get("filed_date"),
                url=p.get("url"),
                excerpt=(p.get("text") or "")[:600],
            )
            _add_citation(c)
            blocks.append(
                f"{c.form_type} · {c.section or '?'} · filed {c.filed_date or '?'}\n"
                f"{c.excerpt}"
            )
        return "\n\n".join(blocks)

    return search_filings


# ─── Tool groups per specialist ────────────────────────────────────────────


def tools_for(ticker: str) -> dict[str, list[BaseTool]]:
    return {
        "fundamentals": [make_get_fundamentals(ticker), make_get_income_statement(ticker)],
        "technicals": [make_get_price_history(ticker), make_compute_technicals(ticker)],
        "news": [make_get_news_sentiment(ticker)],
        "filings": [make_search_filings(ticker)],
    }
