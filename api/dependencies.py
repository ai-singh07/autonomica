"""FastAPI dependency injection for the shared Autonomica instance.

The module holds a single process-level ``Autonomica`` instance that all
route handlers share.  Tests can swap it out via ``set_gov()`` before
running requests.

Typical test pattern::

    from api.dependencies import get_gov, set_gov
    from api.main import app

    test_gov = Autonomica(config=FastConfig(), escalation=MockEscalation())
    app.dependency_overrides[get_gov] = lambda: test_gov
    # … run test …
    app.dependency_overrides.clear()
"""
from __future__ import annotations

from typing import Optional

# Lazy import so the API package can be imported without triggering the full
# Autonomica boot sequence (useful for ``uvicorn --reload`` or CLI tools).
_gov: Optional[object] = None  # type: ignore[assignment]


def get_gov():  # noqa: ANN201
    """Return the shared Autonomica instance, creating it on first call."""
    global _gov
    if _gov is None:
        from autonomica import Autonomica
        from autonomica.config import AutonomicaConfig
        from autonomica.escalation.console import ConsoleEscalation

        _gov = Autonomica(
            config=AutonomicaConfig(),
            escalation=ConsoleEscalation(),
        )
    return _gov


def set_gov(gov: Optional[object]) -> None:  # type: ignore[assignment]
    """Replace the shared instance.  Useful in tests and CLI bootstrap."""
    global _gov
    _gov = gov
