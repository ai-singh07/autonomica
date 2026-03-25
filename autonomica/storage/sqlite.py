"""SQLite storage backend — for local development and single-node deployments.

Uses the stdlib ``sqlite3`` module with ``asyncio.to_thread`` for non-blocking
I/O.  A single ``threading.Lock`` serialises all writes so the in-memory path
(``":memory:"``) works safely across threads.

Usage::

    storage = SQLiteStorage("sqlite:///autonomica.db")  # file path
    storage = SQLiteStorage(":memory:")                 # in-process tests

The ``initialize()`` coroutine creates tables on first use; it is idempotent.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from typing import Optional

from autonomica.models import AgentProfile, GovernanceDecision
from autonomica.storage.base import BaseStorage

# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS agent_profiles (
    agent_id   TEXT PRIMARY KEY,
    data       TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS governance_decisions (
    action_id  TEXT PRIMARY KEY,
    agent_id   TEXT NOT NULL,
    data       TEXT NOT NULL,
    timestamp  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_agent
    ON governance_decisions (agent_id);

CREATE INDEX IF NOT EXISTS idx_decisions_ts
    ON governance_decisions (timestamp DESC);
"""


class SQLiteStorage(BaseStorage):
    """SQLite-backed storage.  Thread-safe; async-friendly via asyncio.to_thread."""

    def __init__(self, db_url: str = "sqlite:///autonomica.db") -> None:
        # Accept both "sqlite:///path" style and bare paths / ":memory:"
        if db_url.startswith("sqlite:///"):
            self._db_path = db_url[len("sqlite:///"):]
        else:
            self._db_path = db_url

        self._in_memory = self._db_path == ":memory:"
        self._lock = threading.Lock()
        # For in-memory databases we must reuse a single connection because
        # a second connection to ":memory:" would see an empty database.
        self._mem_conn: Optional[sqlite3.Connection] = None
        self._initialized = False

    # ── Connection management ─────────────────────────────────────────────────

    def _get_conn(self) -> tuple[sqlite3.Connection, bool]:
        """Return (connection, should_close_after_use)."""
        if self._in_memory:
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(
                    ":memory:", check_same_thread=False
                )
                self._mem_conn.row_factory = sqlite3.Row
            return self._mem_conn, False  # keep alive
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn, True  # caller should close

    def _run(self, fn):
        """Execute *fn(conn)* under the lock.  Returns whatever *fn* returns."""
        with self._lock:
            conn, should_close = self._get_conn()
            try:
                return fn(conn)
            finally:
                if should_close:
                    conn.close()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_sync(self) -> None:
        def _create(conn: sqlite3.Connection) -> None:
            conn.executescript(_DDL)
            conn.commit()

        self._run(_create)
        self._initialized = True

    async def initialize(self) -> None:
        """Create tables if they don't already exist (idempotent)."""
        if not self._initialized:
            await asyncio.to_thread(self._init_sync)

    # ── BaseStorage implementation ────────────────────────────────────────────

    async def save_profile(self, profile: AgentProfile) -> None:
        await self.initialize()
        data = profile.model_dump_json()
        ts = profile.updated_at.isoformat()

        def _write(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO agent_profiles (agent_id, data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    data       = excluded.data,
                    updated_at = excluded.updated_at
                """,
                (profile.agent_id, data, ts),
            )
            conn.commit()

        await asyncio.to_thread(self._run, _write)

    async def load_profile(self, agent_id: str) -> Optional[AgentProfile]:
        await self.initialize()

        def _read(conn: sqlite3.Connection) -> Optional[str]:
            row = conn.execute(
                "SELECT data FROM agent_profiles WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
            return row["data"] if row else None

        raw = await asyncio.to_thread(self._run, _read)
        if raw is None:
            return None
        return AgentProfile.model_validate_json(raw)

    async def save_decision(
        self,
        decision: GovernanceDecision,
        agent_id: str = "unknown",
    ) -> None:
        await self.initialize()
        data = decision.model_dump_json()
        ts = decision.timestamp.isoformat()

        def _write(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO governance_decisions
                    (action_id, agent_id, data, timestamp)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(action_id) DO UPDATE SET
                    data      = excluded.data,
                    timestamp = excluded.timestamp
                """,
                (decision.action_id, agent_id, data, ts),
            )
            conn.commit()

        await asyncio.to_thread(self._run, _write)

    async def load_decision(self, action_id: str) -> Optional[GovernanceDecision]:
        await self.initialize()

        def _read(conn: sqlite3.Connection) -> Optional[str]:
            row = conn.execute(
                "SELECT data FROM governance_decisions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
            return row["data"] if row else None

        raw = await asyncio.to_thread(self._run, _read)
        if raw is None:
            return None
        return GovernanceDecision.model_validate_json(raw)

    async def list_profiles(self) -> list[AgentProfile]:
        await self.initialize()

        def _read(conn: sqlite3.Connection) -> list[str]:
            rows = conn.execute(
                "SELECT data FROM agent_profiles ORDER BY updated_at DESC"
            ).fetchall()
            return [r["data"] for r in rows]

        rows = await asyncio.to_thread(self._run, _read)
        return [AgentProfile.model_validate_json(r) for r in rows]

    async def list_decisions(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[GovernanceDecision]:
        await self.initialize()

        def _read(conn: sqlite3.Connection) -> list[str]:
            if agent_id:
                rows = conn.execute(
                    """
                    SELECT data FROM governance_decisions
                    WHERE agent_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (agent_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT data FROM governance_decisions
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [r["data"] for r in rows]

        rows = await asyncio.to_thread(self._run, _read)
        return [GovernanceDecision.model_validate_json(r) for r in rows]

    async def close(self) -> None:
        """Close the persistent in-memory connection if open."""
        if self._in_memory and self._mem_conn is not None:
            def _close(conn: sqlite3.Connection) -> None:
                conn.close()

            with self._lock:
                _close(self._mem_conn)
                self._mem_conn = None
                self._initialized = False
