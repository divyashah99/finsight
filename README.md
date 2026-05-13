# FinSight — Autonomous Equity Research Platform

Multi-agent AI system that turns a stock ticker into a structured investment memo.

Built to demonstrate production AI engineering for an AI Engineer role:

- **MCP** (Model Context Protocol) server exposing Alpha Vantage as tools
- **LangGraph** multi-agent orchestrator with parallel fan-out and a bounded Critic-revision loop
- **RAG** over SEC 10-K / 10-Q filings (EDGAR → chunk → OpenAI embeddings → Qdrant)
- **Streaming UI** — token-level SSE from FastAPI to Next.js, live agent timeline
- **Production hygiene** — Postgres-backed cache + persisted token-bucket rate limits, idempotent background ingestion, Dockerized, Render + Vercel deploy

> **Scope:** portfolio / showcase project. The goal is architecture quality, not prediction accuracy.

---

## Architecture

```
Next.js 15 (Vercel)  ──SSE──▶  FastAPI (Render)  ──▶  LangGraph orchestrator
                                                            │
                                  ┌──────────┬──────────────┼──────────────┐
                                  ▼          ▼              ▼              ▼
                                Market     Quant          News            SEC
                                Agent      Agent          Agent          Agent
                                  │          │              │              │
                              MCP server   pandas        MCP server    Qdrant (RAG)
                              (Alpha       (RSI/MACD/    (Alpha            │
                              Vantage)     vol)          Vantage)      EDGAR fetcher
                                                                       (APScheduler)
                                  └────────── join ──────────┘
                                              │
                                            Writer ─► Critic ─► Final Memo
                                                       ▲ │
                                                       └─┘ (max 1 revision)
```

Cross-cutting:

- **Storage:** Postgres (runs, reports, cache, sec_docs) + Qdrant (sec_chunks)
- **Cache:** Postgres KV with TTL per tool
- **Rate limits:** persisted token bucket survives restarts
- **Models:** `gpt-4o-mini` (chat) + `text-embedding-3-small` (RAG)

---

## What's interesting (for an interview)

### 1. Real MCP server — not a hand-rolled wrapper
[apps/api/finsight/mcp_servers/alpha_vantage_server.py](apps/api/finsight/mcp_servers/alpha_vantage_server.py) is a real stdio MCP server using Anthropic's `mcp` SDK. It exposes four tools (`av_overview`, `av_daily`, `av_income_statement`, `av_news_sentiment`) with JSON-schema inputs.

The LangGraph agents call it via the MCP client in [apps/api/finsight/tools/mcp_client.py](apps/api/finsight/tools/mcp_client.py) — the same protocol Claude Desktop or Cursor would use to invoke the server. You can also drop it into Claude Desktop directly:

```json
{
  "mcpServers": {
    "alpha-vantage": {
      "command": "python",
      "args": ["-m", "finsight.mcp_servers.alpha_vantage_server"]
    }
  }
}
```

### 2. Bounded Critic loop
The Critic ([apps/api/finsight/agents/critic.py](apps/api/finsight/agents/critic.py)) can flag a memo as needing revision, but the cap lives in the *graph*, not the critic ([agents/graph.py](apps/api/finsight/agents/graph.py) — `_route_after_critic`). That's deliberate: an LLM that can decide its own retry budget is a liability.

### 3. Section-aware chunking
SEC filings are chunked by recognized 10-K sections (Risk Factors, MD&A, Market Risk) — see [services/chunker.py](apps/api/finsight/services/chunker.py). The SEC agent then fires three targeted queries against Qdrant (one per section) and dedupes ([agents/sec.py](apps/api/finsight/agents/sec.py)). The structured memo's citations are 1-based indices into the retrieved evidence list, so the model can't hallucinate URLs.

### 4. Streaming + structured output coexist
Writer is a two-pass design ([agents/writer.py](apps/api/finsight/agents/writer.py)): pass 1 streams a markdown narrative to the UI (good UX); pass 2 converts it into a strict-JSON `Memo` ([agents/memo_schema.py](apps/api/finsight/agents/memo_schema.py)) for storage and critique. The UI shows the markdown live, then swaps in the structured bull/bear/risks view once the run completes.

### 5. MCP-style decorators on every tool
Caching, rate-limiting, and retry are decorator composition ([tools/base.py](apps/api/finsight/tools/base.py)). Order matters: `cached → rate_limited → with_retry → raw call`. Cache hits cost zero tokens; retries can't double-spend the rate-limit bucket.

---

## Repo layout

```
finsight/
├── apps/
│   ├── api/                        # FastAPI + LangGraph + MCP servers
│   │   └── finsight/
│   │       ├── agents/             # market, quant, news, sec, writer, critic + graph
│   │       ├── mcp_servers/        # alpha_vantage_server (MCP stdio)
│   │       ├── tools/              # mcp_client, alpha_vantage (HTTP), edgar, base
│   │       ├── services/           # cache, rate_limit, llm, vectorstore, chunker, sec_ingest
│   │       ├── routers/            # research (SSE), reports, ingest
│   │       ├── jobs/               # apscheduler
│   │       ├── db/                 # models, client
│   │       └── prompts/
│   └── web/                        # Next.js 15 (App Router)
│       └── app/research/[ticker]/  # streaming research page
├── docker-compose.yml              # postgres + qdrant + api (local)
├── render.yaml                     # backend deploy
└── apps/web/vercel.json            # frontend deploy
```

---

## Quickstart

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY and ALPHAVANTAGE_API_KEY at minimum.

# 1. Start Postgres + Qdrant
docker compose up -d postgres qdrant

# 2. Start the API (runs alembic migrations then uvicorn)
docker compose up api

# 3. Start the frontend
cd apps/web
npm install
npm run dev
```

Open <http://localhost:3000/research/AAPL>. The first request for a ticker auto-ingests its SEC filings (one-time, idempotent); subsequent runs hit the cache.

### Manual SEC ingestion

```bash
curl -X POST http://localhost:8000/ingest/sec/AAPL
curl http://localhost:8000/ingest/sec/AAPL/status
```

---

## Environment

See [.env.example](.env.example). Minimum:

| Var | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM + embeddings |
| `ALPHAVANTAGE_API_KEY` | Market data + news (free key: alphavantage.co) |
| `SEC_USER_AGENT` | EDGAR requires identifying header (`Name email@example.com`) |
| `DATABASE_URL` / `DATABASE_URL_SYNC` | Postgres (asyncpg / psycopg2 URLs) |
| `QDRANT_URL` | Qdrant (Cloud or self-hosted) |

---

## Deployment

- **Frontend** → Vercel (autodetects `apps/web`). Set `NEXT_PUBLIC_API_URL` to the Render URL.
- **Backend** → Render web service from [render.yaml](render.yaml) (`apps/api/Dockerfile`). Set all env vars listed in `render.yaml` via the Render dashboard.
- **Postgres** → Supabase free tier (use the pooler URL).
- **Qdrant** → Qdrant Cloud free tier (1GB cluster).
