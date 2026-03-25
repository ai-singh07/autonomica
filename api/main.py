"""Autonomica FastAPI dashboard backend (§9 of spec).

Start the server::

    uvicorn api.main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import agents, actions, metrics, governance, audit

# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Autonomica",
    description=(
        "Runtime adaptive governance for AI agents.\n\n"
        "This API exposes agent profiles, governance decisions, live metrics, "
        "human-override controls, and compliance-ready audit export."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow any origin in development; lock down in production ───────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
_API = "/api"

app.include_router(agents.router,     prefix=_API)
app.include_router(actions.router,    prefix=_API)
app.include_router(metrics.router,    prefix=_API)
app.include_router(governance.router, prefix=_API)
app.include_router(audit.router,      prefix=_API)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"], summary="Health check")
async def health() -> dict:
    """Returns 200 when the service is running."""
    return {"status": "ok", "service": "autonomica", "version": "0.1.0"}


# ── Entry point (for direct ``python api/main.py``) ───────────────────────────

if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
