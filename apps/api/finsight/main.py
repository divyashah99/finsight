"""FastAPI app factory.

Wires together logging, CORS, routers, and the APScheduler lifecycle.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finsight.logging_setup import configure_logging, get_logger
from finsight.settings import settings

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log.info("app.startup env=%s", settings.app_env)
    from contextlib import AsyncExitStack

    from finsight.jobs import scheduler
    from finsight.services import vectorstore

    try:
        await vectorstore.ensure_collection()
    except Exception as e:  # noqa: BLE001
        log.warning("qdrant.bootstrap_failed error=%s", e)

    # LangGraph Postgres checkpointer — durable memory for the follow-up analyst
    # agent. Held open for the app's lifetime via an exit stack. psycopg (v3)
    # wants a plain libpq DSN, so we use the sync-style URL (no +asyncpg driver).
    app.state.checkpointer = None
    cp_stack = AsyncExitStack()
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        saver = await cp_stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(settings.database_url_sync)
        )
        await saver.setup()
        app.state.checkpointer = saver
        log.info("checkpointer.ready")
    except Exception as e:  # noqa: BLE001
        log.warning("checkpointer.init_failed error=%s", e)

    scheduler.start()

    try:
        yield
    finally:
        scheduler.shutdown()
        await cp_stack.aclose()
        log.info("app.shutdown")


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="FinSight API",
        version="0.1.0",
        description="Autonomous equity research multi-agent system.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.app_env}

    _register_routers(app)
    return app


def _register_routers(app: FastAPI) -> None:
    """Import routers lazily so partial scaffolds still boot."""
    try:
        from finsight.routers import research

        app.include_router(research.router)
    except ImportError:
        log.warning("research router not available yet")

    try:
        from finsight.routers import reports

        app.include_router(reports.router)
    except ImportError:
        pass

    try:
        from finsight.routers import ingest

        app.include_router(ingest.router)
    except ImportError:
        pass

    try:
        from finsight.routers import chat

        app.include_router(chat.router)
    except ImportError:
        log.warning("chat router not available yet")


app = create_app()
