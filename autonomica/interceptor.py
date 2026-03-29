"""Action Interceptor — the main Autonomica entry point.

Wraps agent tools with governance. Every tool call flows through:
  score → decide mode → enforce → audit log → adapt profile → (optionally) persist
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from autonomica.adapter import AdaptationEngine
from autonomica.audit import AuditLogger
from autonomica.escalation.base import BaseEscalation
from autonomica.escalation.console import ConsoleEscalation
from autonomica.governor import GovernanceEngine
from autonomica.models import (
    ActionType,
    AgentAction,
    AgentProfile,
    GovernanceDecision,
    GovernanceMode,
    RiskScore,
)
from autonomica.scorer import RiskScorer

logger = logging.getLogger(__name__)

# Action types that are too risky to fail-open.
_HARD_GATE_ON_ERROR: frozenset[ActionType] = frozenset({
    ActionType.WRITE,
    ActionType.DELETE,
    ActionType.COMMUNICATE,
    ActionType.FINANCIAL,
})


class _GatewayEscalation(BaseEscalation):
    """
    Internal escalation layer that bridges the Future-based human-override
    API with the governor's wait_for_response interface.

    notify() is delegated to the backend.
    wait_for_response() races:
      • A programmatic Future set by record_human_override().
      • The backend's own response mechanism (e.g. a Slack webhook reply).
    Whichever resolves first wins; timeout → None.
    """

    def __init__(
        self,
        pending: dict[str, asyncio.Future],
        backend: BaseEscalation,
    ) -> None:
        self._pending = pending
        self._backend = backend

    async def notify(
        self, action: AgentAction, mode: GovernanceMode, risk_score: "RiskScore"
    ) -> None:
        await self._backend.notify(action, mode, risk_score)

    async def wait_for_response(
        self, action_id: str, timeout: float
    ) -> Optional[bool]:
        """Race programmatic Future vs backend response. Timeout → None.

        A backend that returns None immediately (e.g. ConsoleEscalation) is
        treated as "no response yet" — the race continues waiting for either the
        programmatic Future (set by record_human_override) or the hard timeout.
        Only a definitive True or False terminates the race early.
        """
        future = self._pending.get(action_id)
        b_task = asyncio.ensure_future(
            self._backend.wait_for_response(action_id, timeout)
        )

        if future is None:
            try:
                return await asyncio.wait_for(asyncio.shield(b_task), timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                b_task.cancel()
                return None

        # Race: programmatic Future vs backend response.
        # Use a deadline-based loop so that a backend returning None immediately
        # (non-interactive backends like ConsoleEscalation) does not short-circuit
        # the wait — we keep going until either a definitive True/False arrives or
        # the deadline expires.
        f_task = asyncio.ensure_future(asyncio.shield(future))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return None

                pending_tasks = {t for t in (f_task, b_task) if not t.done()}
                if not pending_tasks:
                    break  # all exhausted

                done, _ = await asyncio.wait(
                    pending_tasks,
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    return None  # real timeout — no response within deadline

                # Prefer programmatic override over backend; prefer True/False
                # over None (None = "no response", keep waiting).
                for preferred in (f_task, b_task):
                    if preferred in done and not preferred.cancelled():
                        try:
                            result = preferred.result()
                            if result is not None:
                                return result
                        except Exception:
                            pass
                # All done tasks returned None — continue if any tasks remain
        except Exception:
            pass
        finally:
            for t in (f_task, b_task):
                if not t.done():
                    t.cancel()

        return None


class Autonomica:
    """
    Main entry point. Wraps an agent framework with governance.

    Usage (3 lines):
        gov = Autonomica()
        tools = wrap_langchain_tools(tools, gov, agent_id="my-agent")
        # Done — all tool calls now flow through governance.
    """

    def __init__(
        self,
        config: Any = None,
        storage: Any = None,
        escalation: BaseEscalation | None = None,
        adapter: Any = None,
        audit: Any = None,
    ) -> None:
        self._config = config
        self.scorer = RiskScorer(config)
        self.governor = GovernanceEngine(config)

        # Adaptation engine — default is always-on
        self._adapter: AdaptationEngine = (
            adapter if adapter is not None else AdaptationEngine(config)
        )

        # Storage backend (optional; in-memory dict used when None)
        self._storage = storage

        # Audit logger (optional; only logs to Python logging when None)
        self._audit: AuditLogger = audit if audit is not None else AuditLogger()

        # In-memory state
        self._profiles: dict[str, AgentProfile] = {}
        self._decisions: dict[str, GovernanceDecision] = {}
        self._action_agents: dict[str, str] = {}      # action_id → agent_id
        self._actions: dict[str, AgentAction] = {}    # action_id → AgentAction
        self._pending: dict[str, asyncio.Future] = {} # action_id → Future[bool|None]

        backend = escalation if escalation is not None else ConsoleEscalation()
        self._escalation = _GatewayEscalation(self._pending, backend)

        # Fail policy — read once from config so hot-path has no attr-lookup
        self._fail_policy: str = (
            getattr(config, "fail_policy", "adaptive")
            if config is not None
            else "adaptive"
        )

    # ── Profile management ────────────────────────────────────────────────────

    def _get_or_create_profile(
        self, agent_id: str, agent_name: str
    ) -> AgentProfile:
        if agent_id not in self._profiles:
            # Respect default_trust_score from config if provided
            default_trust = (
                float(getattr(self._config, "default_trust_score", 50.0))
                if self._config is not None
                else 50.0
            )
            self._profiles[agent_id] = AgentProfile(
                agent_id=agent_id,
                agent_name=agent_name or agent_id,
                trust_score=default_trust,
            )
        return self._profiles[agent_id]

    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Return the live profile for an agent, or None if unseen."""
        return self._profiles.get(agent_id)

    def get_decision(self, action_id: str) -> Optional[GovernanceDecision]:
        """Return a recorded governance decision by action ID."""
        return self._decisions.get(action_id)

    # ── Fail-policy helpers ────────────────────────────────────────────────────

    def _make_error_risk_score(self, action: AgentAction, exc: Exception) -> RiskScore:
        """Synthetic RiskScore produced when the normal scoring pipeline fails."""
        return RiskScore(
            composite_score=75.0,  # Sits mid-HARD_GATE band by default
            financial_magnitude=0.0,
            data_sensitivity=0.0,
            reversibility=0.0,
            agent_track_record=50.0,
            novelty=0.0,
            cascade_risk=0.0,
            explanation=(
                f"[PIPELINE ERROR] {type(exc).__name__}: {exc}. "
                f"fail_policy={self._fail_policy!r}  "
                f"action_type={action.action_type.value}"
            ),
        )

    async def _handle_pipeline_error(
        self,
        action: AgentAction,
        profile: AgentProfile,
        exc: Exception,
        start_ms: float,
    ) -> GovernanceDecision:
        """
        Produce a safe GovernanceDecision when the normal pipeline raises.

        Behaviour is controlled by ``config.fail_policy``:
        - "open"     → LOG_AND_ALERT, approved=True  (for every action type)
        - "closed"   → QUARANTINE,    approved=False (for every action type)
        - "adaptive" → READ → LOG_AND_ALERT, approved=True
                       everything else → HARD_GATE (block until human approves
                       or the hard-gate timeout elapses, whichever comes first)
        """
        logger.error(
            "Autonomica pipeline error — action=%s agent=%s tool=%s: %s",
            action.action_id, action.agent_id, action.tool_name, exc,
        )

        risk_score = self._make_error_risk_score(action, exc)
        policy = self._fail_policy

        if policy == "open":
            mode = GovernanceMode.LOG_AND_ALERT
            approved = True

        elif policy == "closed":
            mode = GovernanceMode.QUARANTINE
            approved = False

        else:  # "adaptive" (default)
            if action.action_type not in _HARD_GATE_ON_ERROR:
                # READ actions: fail open — do not block the agent
                mode = GovernanceMode.LOG_AND_ALERT
                approved = True
            else:
                # Risky action types: gate until a human responds
                mode = GovernanceMode.HARD_GATE
                loop = asyncio.get_running_loop()
                future: asyncio.Future = loop.create_future()
                self._pending[action.action_id] = future
                self._action_agents[action.action_id] = action.agent_id
                self._actions[action.action_id] = action
                try:
                    approved = await self.governor.enforce(
                        mode, action, self._escalation, risk_score
                    )
                except Exception as enforce_exc:
                    logger.error(
                        "Autonomica fail-policy enforcement also failed: %s", enforce_exc
                    )
                    approved = False  # enforcement failed → safest default is block
                finally:
                    if not future.done():
                        future.cancel()
                    self._pending.pop(action.action_id, None)

        decision = GovernanceDecision(
            action_id=action.action_id,
            risk_score=risk_score,
            mode=mode,
            approved=approved,
            decision_time_ms=(time.monotonic() * 1000) - start_ms,
            pipeline_error=f"{type(exc).__name__}: {exc}",
        )

        # Always store so override / feedback calls work even in error path
        self._decisions[action.action_id] = decision
        self._action_agents[action.action_id] = action.agent_id
        self._actions[action.action_id] = action

        # Update profile counters
        profile.total_actions += 1
        if approved:
            profile.approved_actions += 1
        if mode >= GovernanceMode.SOFT_GATE:
            profile.escalated_actions += 1
        profile.updated_at = datetime.now(timezone.utc)

        # Audit — log error path decisions too
        self._audit.log_decision(action, decision, profile)

        return decision

    # ── Core pipeline ─────────────────────────────────────────────────────────

    async def evaluate_action(self, action: AgentAction) -> GovernanceDecision:
        """
        Core governance pipeline: score → decide mode → enforce → log → adapt.

        For SOFT_GATE and HARD_GATE this coroutine suspends until the
        timeout elapses or record_human_override() resolves the pending Future.
        """
        start_ms = time.monotonic() * 1000
        profile = self._get_or_create_profile(action.agent_id, action.agent_name)

        # ── OpenTelemetry Tracing ─────────────────────────────────────────────
        if OTEL_AVAILABLE:
            tracer = trace.get_tracer("autonomica")
            span = tracer.start_span(
                "autonomica.evaluate",
                attributes={
                    "agent.id": action.agent_id,
                    "tool.name": action.tool_name,
                    "action.type": action.action_type.value,
                }
            )
        else:
            span = None

        try:
            # 1. Score
            risk_score = self.scorer.score(action, profile)

            # 2. Decide governance mode
            mode = self.governor.decide(risk_score, profile)

            if span:
                span.set_attributes({
                    "governance.mode": mode.name,
                    "risk.score": risk_score.composite_score,
                })

            # 3. Register future *before* enforce so overrides can arrive during gate
            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()
            self._pending[action.action_id] = future

            try:
                approved = await self.governor.enforce(
                    mode, action, self._escalation, risk_score
                )
            finally:
                if not future.done():
                    future.cancel()
                self._pending.pop(action.action_id, None)

            decision_time_ms = (time.monotonic() * 1000) - start_ms

            decision = GovernanceDecision(
                action_id=action.action_id,
                risk_score=risk_score,
                mode=mode,
                approved=approved,
                decision_time_ms=decision_time_ms,
            )

            if span:
                span.set_attribute("decision.approved", approved)
                span.set_status(Status(StatusCode.OK))

            # 4. Store action + decision for later feedback/override calls
            self._decisions[action.action_id] = decision
            self._action_agents[action.action_id] = action.agent_id
            self._actions[action.action_id] = action

            # 5. Update basic profile counters
            profile.total_actions += 1
            if approved:
                profile.approved_actions += 1
            if mode >= GovernanceMode.SOFT_GATE:
                profile.escalated_actions += 1
            profile.updated_at = datetime.now(timezone.utc)

            # 6. Adaptation (per-tool trust, trust EMA, threshold drift, vagal tone)
            self._adapter.update_after_action(action, decision, profile)

            # 7. Audit log
            self._audit.log_decision(action, decision, profile)

            # 8. Async persist (fire-and-forget so it never blocks the caller)
            if self._storage is not None:
                asyncio.create_task(self._storage.save_profile(profile))
                asyncio.create_task(
                    self._storage.save_decision(decision, agent_id=action.agent_id)
                )

            return decision

        except Exception as exc:
            if span:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            
            # Clean up any future that was registered before the exception
            stale_future = self._pending.pop(action.action_id, None)
            if stale_future and not stale_future.done():
                stale_future.cancel()
            return await self._handle_pipeline_error(action, profile, exc, start_ms)
        finally:
            if span:
                span.end()

    def evaluate_action_sync(self, action: AgentAction) -> GovernanceDecision:
        """
        Synchronous wrapper around evaluate_action.

        Safe to call from both sync code (pure LangChain agent) and from
        within an already-running event loop (e.g. Jupyter / async framework)
        by running the coroutine in a dedicated thread.
        """
        try:
            asyncio.get_running_loop()
            # There is a running loop — must offload to a thread to avoid
            # "cannot run nested event loop" errors.
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.evaluate_action(action)).result()
        except RuntimeError:
            # No running loop — straightforward asyncio.run.
            return asyncio.run(self.evaluate_action(action))

    # ── Tool wrapping ─────────────────────────────────────────────────────────

    def wrap_tool(
        self, tool: Any, agent_id: str, agent_name: str | None = None
    ) -> Any:
        """Wrap a single LangChain-compatible tool with governance."""
        from autonomica.integrations.langchain import GovernedTool

        return GovernedTool(
            name=tool.name,
            description=tool.description or "",
            original_tool=tool,
            autonomica=self,
            agent_id=agent_id,
            agent_name=agent_name or agent_id,
        )

    def wrap_tools(
        self, tools: list[Any], agent_id: str, agent_name: str | None = None
    ) -> list[Any]:
        """Wrap multiple tools at once."""
        return [self.wrap_tool(t, agent_id, agent_name) for t in tools]

    def wrap_langchain_agent(self, agent: Any, agent_id: str) -> Any:
        """Convenience: wrap all tools attached to a LangChain agent executor."""
        agent.tools = self.wrap_tools(agent.tools, agent_id)
        return agent

    # ── Feedback ──────────────────────────────────────────────────────────────

    def record_outcome(
        self, action_id: str, success: bool, notes: str = ""
    ) -> None:
        """Record whether an approved action succeeded or failed post-execution.

        On failure the agent's incident counter is incremented, the
        adaptation engine tightens thresholds, and an audit event is logged.
        """
        if success:
            return

        agent_id = self._action_agents.get(action_id)
        if not agent_id:
            return

        profile = self._profiles.get(agent_id)
        action = self._actions.get(action_id)

        if profile:
            profile.incidents += 1
            profile.updated_at = datetime.now(timezone.utc)

            # Adaptation: tighten thresholds + penalise trust.
            # Pass the original governance decision so the adapter can tighten
            # only the threshold relevant to the mode that approved this action.
            if action:
                original_decision = self._decisions.get(action_id)
                self._adapter.update_after_incident(action, profile, original_decision)

            # Audit
            self._audit.log_incident(action_id, agent_id, notes)

            # Persist
            if self._storage is not None and action:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._storage.save_profile(profile))
                except RuntimeError:
                    pass  # No running loop — skip async persist

    def record_human_override(
        self, action_id: str, approved: bool, reason: str = ""
    ) -> None:
        """Record a human's override decision for a pending gated action.

        If the action is still waiting at a SOFT_GATE or HARD_GATE this
        immediately resolves the pending Future so enforce() can return.
        The adaptation engine then widens or tightens thresholds accordingly.
        """
        # Resolve the pending gate Future (if still open)
        future = self._pending.get(action_id)
        if future and not future.done():
            try:
                future.get_loop().call_soon_threadsafe(future.set_result, approved)
            except Exception:
                pass  # future already done or loop closed; safe to ignore

        decision = self._decisions.get(action_id)
        if not decision:
            return

        agent_id = self._action_agents.get(action_id)
        if not agent_id:
            return

        profile = self._profiles.get(agent_id)
        action = self._actions.get(action_id)

        if profile:
            # Update false-escalation counter when human approves something we gated
            if approved:
                profile.false_escalations += 1

            # Adaptation: widen/tighten thresholds + trust update
            if action:
                self._adapter.update_after_override(action, decision, profile, approved)

            # Audit
            self._audit.log_override(action_id, agent_id, approved, reason)

            # Persist
            if self._storage is not None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._storage.save_profile(profile))
                except RuntimeError:
                    pass
