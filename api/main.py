"""
Graphene — FastAPI Application Entry Point

The main Graphene API server. Handles CORS, request logging,
error handling, and mounts all route modules.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.dependencies import app_state
from api.routes import alerts, graph, health, reports

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = str(
    Path(__file__).resolve().parent.parent / "frontend"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events.

    Startup:
    1. Connect to Neo4j
    2. Load ML models and cached results
    3. Log readiness status

    Shutdown:
    1. Close Neo4j driver
    """
    logger.info("=" * 60)
    logger.info("  GRAPHENE — Fund Flow Intelligence API")
    logger.info("  Starting up...")
    logger.info("=" * 60)

    app_state.initialize()

    yield

    logger.info("Shutting down Graphene API...")
    app_state.shutdown()
    logger.info("Graphene API stopped.")


app = FastAPI(
    title="Graphene API",
    version="1.0.0",
    description=(
        "AI-powered Fund Flow Tracking and Fraud Detection "
        "— PSBs Hackathon 2026"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(graph.router, prefix="/api")
app.include_router(reports.router, prefix="/api")

frontend_path = Path(FRONTEND_DIR)
if frontend_path.exists():
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static",
    )


@app.middleware("http")
async def request_logging_middleware(
    request: Request, call_next
):
    """Log method, path, status, and duration for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000

    logger.info(
        "%s %s → %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch all unhandled exceptions and return structured JSON."""
    logger.exception(
        "Unhandled error on %s %s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.get("/", tags=["Frontend"])
async def serve_frontend():
    """Serve the frontend dashboard at root."""
    index_path = Path(FRONTEND_DIR) / "index.html"
    if index_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(index_path))
    return JSONResponse(
        content={
            "message": "Graphene API is running. "
                       "Frontend not found at /frontend/index.html",
            "docs": "/docs",
        }
    )
