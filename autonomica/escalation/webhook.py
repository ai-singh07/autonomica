"""Webhook escalation backend — POSTs governance alerts to any HTTP endpoint.

Enables integrations with PagerDuty, OpsGenie, custom dashboards, or any 
webhook-capable service. Supports optional HMAC-SHA256 signatures for 
verification.

Usage::

    from autonomica.escalation.webhook import WebhookEscalation

    gov = Autonomica(
        escalation=WebhookEscalation(
            url="https://api.myapp.com/webhooks/autonomica",
            secret="my-signing-secret"
        )
    )
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

import httpx

from autonomica.escalation.base import BaseEscalation
from autonomica.models import AgentAction, GovernanceMode, RiskScore


class WebhookEscalation(BaseEscalation):
    """POSTs governance events to a generic webhook endpoint.

    Args:
        url:     The HTTP(S) endpoint to POST to.
        headers: Optional dictionary of extra HTTP headers.
        secret:  Optional secret key for HMAC-SHA256 signing of the payload.
                 If provided, an `X-Autonomica-Signature` header is added.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        secret: Optional[str] = None,
    ) -> None:
        self._url = url
        self._headers = headers or {}
        self._secret = secret

    async def notify(
        self, action: AgentAction, mode: GovernanceMode, risk_score: RiskScore
    ) -> None:
        """Build and POST the webhook payload."""
        payload = {
            "version": "1.0",
            "timestamp": int(time.time()),
            "action": action.model_dump(),
            "governance": {
                "mode": mode.name,
                "risk_score": risk_score.model_dump(),
            },
        }

        body = json.dumps(payload)
        headers = self._headers.copy()
        headers["Content-Type"] = "application/json"

        if self._secret:
            signature = hmac.new(
                self._secret.encode("utf-8"),
                body.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            headers["X-Autonomica-Signature"] = signature

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Retry once on 5xx with a 1s delay as per requirements
                for attempt in range(2):
                    response = await client.post(
                        self._url,
                        content=body,
                        headers=headers
                    )
                    
                    if response.is_success:
                        break
                    
                    if attempt == 0 and response.is_server_error:
                        time.sleep(1.0)
                        continue
                        
                    response.raise_for_status()
        except Exception:
            # Never let a webhook failure block the governance pipeline.
            pass

    async def wait_for_response(
        self, action_id: str, timeout: float
    ) -> Optional[bool]:
        """Generic webhooks are fire-and-forget; human response via API."""
        return None
