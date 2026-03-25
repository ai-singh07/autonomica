"""
Autonomica Quickstart — Add governance to any LangChain agent in 3 lines.

Run this to see Autonomica intercept tool calls and apply graduated governance.

    python examples/quickstart.py
"""
from __future__ import annotations

from langchain_core.tools import BaseTool

from autonomica import Autonomica
from autonomica.integrations.langchain import wrap_langchain_tools


# ── Step 1: Define your tools as normal ───────────────────────────────────────

class ReadDatabaseTool(BaseTool):
    name: str = "read_database"
    description: str = "Read data from the database."

    def _run(self, query: str = "", **kw) -> str:
        return f"[DB] Results for: {query}"

    async def _arun(self, **kw) -> str:
        return "[DB] Results (async)"


class SendEmailTool(BaseTool):
    name: str = "send_email"
    description: str = "Send an email to a recipient."

    def _run(self, to: str = "", subject: str = "", **kw) -> str:
        return f"[EMAIL] Sent to {to}: {subject}"

    async def _arun(self, **kw) -> str:
        return "[EMAIL] Sent (async)"


class ProcessPaymentTool(BaseTool):
    name: str = "process_payment"
    description: str = "Process a payment or transfer funds to a recipient."

    def _run(self, amount: float = 0.0, recipient: str = "", **kw) -> str:
        return f"[PAY] ${amount:,.2f} sent to {recipient}"

    async def _arun(self, **kw) -> str:
        return "[PAY] Payment processed (async)"


# ── Step 2: Wrap with Autonomica — THIS IS THE MAGIC (3 lines) ────────────────

gov = Autonomica()
tools = wrap_langchain_tools(
    [ReadDatabaseTool(), SendEmailTool(), ProcessPaymentTool()],
    gov,
    agent_id="finance-assistant",
)

# That's it. Every tool call now flows through risk scoring + governance.


# ── Step 3: Run the agent as normal ──────────────────────────────────────────

def _show(label: str, result: str) -> None:
    d = list(gov._decisions.values())[-1]
    p = gov.get_agent_profile("finance-assistant")
    bar = "█" * int(d.risk_score.composite_score / 5)
    print(
        f"\n  {label}\n"
        f"  Mode   : {d.mode.name}\n"
        f"  Score  : {d.risk_score.composite_score:.1f}/100  {bar}\n"
        f"  Result : {result}\n"
        f"  Trust  : {p.trust_score:.1f}   Vagal tone: {p.vagal_tone:.1f}"
    )


if __name__ == "__main__":
    sep = "─" * 60
    print(f"\n{sep}")
    print("  Autonomica Quickstart")
    print(f"{sep}")

    print("\n➤  Action 1 — database read  (new agent, first call)")
    r = tools[0]._run(query="SELECT * FROM products LIMIT 10")
    _show("read_database", r)

    print(f"\n{sep}")
    print("\n➤  Action 2 — send email  (COMMUNICATE type, new agent)")
    r = tools[1]._run(to="customer@example.com", subject="Your invoice is ready")
    _show("send_email", r)

    print(f"\n{sep}")
    print("\n➤  Action 3 — $500,000 payment  (FINANCIAL, high risk)")
    r = tools[2]._run(amount=500_000.0, recipient="vendor@corp.com")
    _show("process_payment", r)

    print(f"\n{sep}")
    print("\n➤  Agent profile after 3 actions:")
    p = gov.get_agent_profile("finance-assistant")
    print(
        f"  total_actions    : {p.total_actions}\n"
        f"  approved_actions : {p.approved_actions}\n"
        f"  escalated        : {p.escalated_actions}\n"
        f"  trust_score      : {p.trust_score:.1f}\n"
        f"  vagal_tone       : {p.vagal_tone:.1f}\n"
        f"  full_auto_max    : {p.mode_thresholds['full_auto_max']:.1f}  "
        f"(was 15.0, drifts with behaviour)"
    )
    print(f"\n{sep}\n")
