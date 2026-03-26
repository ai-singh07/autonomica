"""
Autonomica Quickstart — Real LangChain agent with live governance.

A GPT-4o-mini agent is given three tasks. Every tool call it makes
flows through the full Autonomica pipeline: risk scoring → mode decision
→ enforce / escalate → audit log.

Usage:
    # Add your key to .env first:  OPENAI_API_KEY=sk-...
    python examples/quickstart.py
"""
from __future__ import annotations

import os
import sys
import textwrap

# ── load .env before any other import ─────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-api-key-here":
    sys.exit(
        "\n[ERROR] Set OPENAI_API_KEY in your .env file before running.\n"
        "        echo 'OPENAI_API_KEY=sk-...' > .env\n"
    )

# ── LangChain 1.x ─────────────────────────────────────────────────────────────
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

# ── Autonomica ────────────────────────────────────────────────────────────────
from autonomica import Autonomica
from autonomica.escalation.console import ConsoleEscalation
from autonomica.integrations.langchain import wrap_langchain_tools
from autonomica.models import GovernanceMode


# ── 1. Define tools ──────────────────────────────────────────────────────────

class ReadDatabaseTool(BaseTool):
    name: str = "read_database"
    description: str = (
        "Query the product database. Input: a SQL SELECT query string."
    )

    def _run(self, query: str = "", **kw) -> str:
        return (
            "id | name            | price\n"
            "---+-----------------+------\n"
            " 1 | Widget Pro      | 49.99\n"
            " 2 | Gadget Standard | 24.99\n"
            " 3 | Doohickey Lite  |  9.99\n"
        )

    async def _arun(self, **kw) -> str:
        return self._run()


class SendEmailTool(BaseTool):
    name: str = "send_email"
    description: str = (
        "Send an email to a customer or vendor. "
        "Inputs: to (email address), subject (string), body (string)."
    )

    def _run(self, to: str = "", subject: str = "", body: str = "", **kw) -> str:
        return f"Email delivered → {to} | Subject: {subject}"

    async def _arun(self, **kw) -> str:
        return self._run(**kw)


class ProcessPaymentTool(BaseTool):
    name: str = "process_payment"
    description: str = (
        "Transfer funds or process a payment. "
        "Inputs: amount (float, USD), recipient (string)."
    )

    def _run(self, amount: float = 0.0, recipient: str = "", **kw) -> str:
        return f"Payment processed: ${amount:,.2f} → {recipient}"

    async def _arun(self, **kw) -> str:
        return self._run(**kw)


# ── 2. Governance-aware escalation ───────────────────────────────────────────

class VerboseEscalation(ConsoleEscalation):
    """Prints a styled intercept banner every time governance fires."""

    async def notify(self, action, mode, risk_score) -> None:
        colour = {
            GovernanceMode.FULL_AUTO:     "\033[92m",
            GovernanceMode.LOG_AND_ALERT: "\033[94m",
            GovernanceMode.SOFT_GATE:     "\033[93m",
            GovernanceMode.HARD_GATE:     "\033[91m",
            GovernanceMode.QUARANTINE:    "\033[95m",
        }.get(mode, "\033[0m")
        reset = "\033[0m"
        bar   = "█" * int(risk_score.composite_score / 5)
        pad   = "░" * (20 - len(bar))

        print(f"\n  {'─'*56}")
        print(f"  {colour}⚡ AUTONOMICA INTERCEPT{reset}")
        print(f"  {'─'*56}")
        print(f"  Agent  : {action.agent_name}  ({action.agent_id})")
        print(f"  Tool   : {action.tool_name}   [{action.action_type.value.upper()}]")
        print(f"  Mode   : {colour}{mode.name}{reset}")
        print(f"  Score  : {risk_score.composite_score:.1f}/100  {colour}{bar}{reset}{pad}")
        print(f"  Signals:")
        print(f"    financial_magnitude : {risk_score.financial_magnitude:.1f}")
        print(f"    data_sensitivity    : {risk_score.data_sensitivity:.1f}")
        print(f"    reversibility       : {risk_score.reversibility:.1f}")
        print(f"    agent_track_record  : {risk_score.agent_track_record:.1f}")
        print(f"    novelty             : {risk_score.novelty:.1f}")
        print(f"    cascade_risk        : {risk_score.cascade_risk:.1f}")
        print(f"  Reason : {textwrap.shorten(risk_score.explanation, 60)}")
        print(f"  {'─'*56}\n")


# ── 3. Wire up Autonomica ─────────────────────────────────────────────────────

gov = Autonomica(escalation=VerboseEscalation())

governed_tools = wrap_langchain_tools(
    [ReadDatabaseTool(), SendEmailTool(), ProcessPaymentTool()],
    gov,
    agent_id="finance-assistant",
    agent_name="Finance Assistant",
)

# ── 4. Build the LangChain 1.x agent ─────────────────────────────────────────

agent = create_agent(
    "openai:gpt-4o-mini",
    tools=governed_tools,
    system_prompt=(
        "You are a finance assistant. Use the available tools to complete tasks. "
        "Always call a tool when asked — do not answer from memory."
    ),
)

# ── 5. Helper: extract final text response from agent state ──────────────────

def get_final_answer(state: dict) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content and not getattr(msg, "tool_calls", None):
            return content if isinstance(content, str) else str(content)
    return "(no response)"


# ── 6. Run three tasks ────────────────────────────────────────────────────────

TASKS = [
    (
        "Look up our top products in the database using SQL.",
        "Task 1 — Database read   (READ, new agent)",
    ),
    (
        "Email support@example.com with subject 'Q1 Product Catalog' "
        "and body 'Please find attached the latest product list.'",
        "Task 2 — Send email      (COMMUNICATE)",
    ),
    (
        "Process a payment of $500,000 to vendor@largecorp.com "
        "for the annual contract.",
        "Task 3 — $500k payment   (FINANCIAL, high risk)",
    ),
]

SEP = "═" * 58

if __name__ == "__main__":
    print(f"\n  {SEP}")
    print(f"  🧠  Autonomica + LangChain — Live Governance Demo")
    print(f"  {SEP}\n")
    print("  A real GPT-4o-mini agent will now attempt three tasks.")
    print("  Watch Autonomica intercept and score every tool call.\n")

    for task, label in TASKS:
        print(f"\n  ▶  {label}")
        print(f"     Prompt: \"{task[:72]}\"")

        try:
            state = agent.invoke({"messages": [HumanMessage(content=task)]})
        except Exception as exc:
            err = str(exc)
            if "insufficient_quota" in err or "429" in err:
                sys.exit(
                    "\n  [ERROR] OpenAI quota exceeded.\n"
                    "  Add credits at https://platform.openai.com/account/billing\n"
                    "  then re-run this demo.\n"
                )
            raise
        answer = get_final_answer(state)

        decisions = list(gov._decisions.values())
        profile   = gov.get_agent_profile("finance-assistant")

        if decisions:
            d      = decisions[-1]
            status = "\033[92mAPPROVED\033[0m" if d.approved else "\033[91mBLOCKED\033[0m"
            print(f"\n  → Agent answer : {answer}")
            print(f"  → Decision     : {status}  "
                  f"(decided in {d.decision_time_ms:.1f} ms)")
            print(f"  → Trust score  : {profile.trust_score:.1f}   "
                  f"Vagal tone: {profile.vagal_tone:.1f}")

    # ── Final profile summary ─────────────────────────────────────────────────
    p  = gov.get_agent_profile("finance-assistant")
    mt = p.mode_thresholds
    print(f"\n  {SEP}")
    print(f"  📊  Agent Profile — finance-assistant")
    print(f"  {SEP}")
    print(f"  total_actions    : {p.total_actions}")
    print(f"  approved_actions : {p.approved_actions}")
    print(f"  escalated        : {p.escalated_actions}")
    print(f"  trust_score      : {p.trust_score:.1f}")
    print(f"  vagal_tone       : {p.vagal_tone:.1f}")
    print(f"  Adaptive thresholds (drift with behaviour):")
    print(f"    FULL_AUTO  ≤ {mt['full_auto_max']:.1f}  "
          f"LOG_AND_ALERT ≤ {mt['log_alert_max']:.1f}  "
          f"SOFT_GATE ≤ {mt['soft_gate_max']:.1f}  "
          f"HARD_GATE ≤ {mt['hard_gate_max']:.1f}")
    print(f"  {SEP}\n")
