"""Real LLM-powered agent demo — Autonomica governance in action.

Uses LangChain + ChatOpenAI (gpt-4o-mini) with four @govern-decorated tools.
Falls back to a direct sequential run with mock data when OPENAI_API_KEY is missing,
so the demo always works without credentials.

Usage::

    # With a real LLM:
    export OPENAI_API_KEY=sk-...
    python examples/real_agent_demo.py

    # Fallback mode (no key needed):
    python examples/real_agent_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from autonomica import Autonomica, AutonomicaConfig, GovernanceBlocked, govern
from autonomica.escalation.base import BaseEscalation
from autonomica.models import AgentAction, GovernanceMode, RiskScore

# ---------------------------------------------------------------------------
# Silent escalation — prints to console instead of posting to Slack
# ---------------------------------------------------------------------------

class _ConsoleEscalation(BaseEscalation):
    async def notify(
        self, action: AgentAction, mode: GovernanceMode, risk_score: RiskScore
    ) -> None:
        colour = {
            GovernanceMode.FULL_AUTO:    "🟢",
            GovernanceMode.LOG_AND_ALERT:"🔵",
            GovernanceMode.SOFT_GATE:    "🟡",
            GovernanceMode.HARD_GATE:    "🔴",
            GovernanceMode.QUARANTINE:   "⛔",
        }.get(mode, "⚪")
        print(
            f"  {colour} [{mode.name}] {action.tool_name} "
            f"(risk={risk_score.composite_score:.0f})"
        )

    async def wait_for_response(self, action_id: str, timeout: float):
        return None  # SOFT_GATE auto-proceeds; HARD_GATE auto-blocks


# ---------------------------------------------------------------------------
# Governance engine for this demo
# ---------------------------------------------------------------------------

gov = Autonomica(
    config=AutonomicaConfig(
        soft_gate_timeout_seconds=0.001,   # demo: don't wait
        hard_gate_timeout_seconds=0.001,
        fail_policy="open",
    ),
    escalation=_ConsoleEscalation(),
)

AGENT_ID = "inventory-agent"

# ---------------------------------------------------------------------------
# Tools (all wrapped with @govern)
# ---------------------------------------------------------------------------

@govern(agent_id=AGENT_ID, action_type="read", autonomica=gov)
def read_inventory(product: str) -> dict:
    """Return mock stock level for a product."""
    mock_stock = {
        "Widget-X": 42,
        "Widget-Y": 250,
        "Gadget-Z": 5,
    }
    count = mock_stock.get(product, 0)
    return {"product": product, "stock": count, "unit": "units"}


@govern(agent_id=AGENT_ID, action_type="communicate", autonomica=gov)
def send_restock_email(product: str, supplier: str) -> str:
    """Send (mock) restock request email to supplier."""
    return (
        f"Email sent to {supplier}: "
        f"Please ship 500 units of {product} within 5 business days."
    )


@govern(agent_id=AGENT_ID, action_type="financial", autonomica=gov)
def process_purchase_order(amount: float, supplier: str) -> str:
    """Create (mock) purchase order for restocking."""
    po_number = "PO-2026-0042"
    return (
        f"Purchase order {po_number} created: "
        f"${amount:,.2f} to {supplier} — pending finance approval."
    )


@govern(agent_id=AGENT_ID, action_type="delete", autonomica=gov)
def delete_old_records(table: str, older_than_days: int) -> str:
    """Delete (mock) stale records from a table."""
    return (
        f"Deleted records from '{table}' older than {older_than_days} days. "
        f"(Soft-delete; recoverable for 30 days.)"
    )


# ---------------------------------------------------------------------------
# LangChain agent path
# ---------------------------------------------------------------------------

def _run_langchain_agent() -> None:
    """Run the task using a real LangChain ReAct agent."""
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.tools import tool as lc_tool
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        print(f"[warn] LangChain dependencies missing ({exc}). Using fallback mode.")
        _run_fallback()
        return

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @lc_tool
    def lc_read_inventory(product: str) -> str:
        """Return the current stock level for a product."""
        result = read_inventory(product)
        return f"{result['product']}: {result['stock']} {result['unit']} in stock"

    @lc_tool
    def lc_send_restock_email(product: str, supplier: str) -> str:
        """Send a restock request email to a supplier."""
        return send_restock_email(product, supplier)

    @lc_tool
    def lc_process_purchase_order(amount: float, supplier: str) -> str:
        """Create a purchase order for a supplier."""
        return process_purchase_order(amount, supplier)

    @lc_tool
    def lc_delete_old_records(table: str, older_than_days: int) -> str:
        """Delete old records from a database table."""
        return delete_old_records(table, older_than_days)

    tools = [
        lc_read_inventory,
        lc_send_restock_email,
        lc_process_purchase_order,
        lc_delete_old_records,
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an inventory management agent. Use your tools to complete tasks efficiently."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    task = (
        "Check inventory for Widget-X. "
        "If stock is below 100, email supplier 'Acme Supplies' to restock, "
        "create a purchase order for $75,000, "
        "and clean up the inventory_log table for records older than 90 days."
    )

    print(f"\nTask: {task}\n")
    print("Governance decisions:")
    try:
        result = executor.invoke({"input": task})
        print(f"\nAgent output:\n{result['output']}")
    except GovernanceBlocked as e:
        print(f"\n[blocked] {e}")


# ---------------------------------------------------------------------------
# Fallback: direct sequential run with mock data
# ---------------------------------------------------------------------------

def _run_fallback() -> None:
    """Run all four tools directly — no LLM needed."""
    task = (
        "Check inventory for Widget-X. "
        "If stock is below 100, email supplier 'Acme Supplies' to restock, "
        "create a purchase order for $75,000, "
        "and clean up the inventory_log table for records older than 90 days."
    )
    print(f"\nTask: {task}")
    print("\n[fallback mode — no OPENAI_API_KEY, running tools directly]\n")
    print("Governance decisions:")

    results: list[str] = []

    # Step 1: Check inventory
    try:
        inv = read_inventory("Widget-X")
        stock = inv["stock"]
        results.append(f"1. Inventory check: Widget-X has {stock} units in stock.")
    except GovernanceBlocked as e:
        results.append(f"1. Inventory check BLOCKED: {e}")
        stock = 999  # assume high so we skip next steps

    # Step 2–4: Only if stock is low
    if stock < 100:
        try:
            email_result = send_restock_email("Widget-X", "Acme Supplies")
            results.append(f"2. Restock email: {email_result}")
        except GovernanceBlocked as e:
            results.append(f"2. Email BLOCKED: {e}")

        try:
            po_result = process_purchase_order(75_000.0, "Acme Supplies")
            results.append(f"3. Purchase order: {po_result}")
        except GovernanceBlocked as e:
            results.append(f"3. PO BLOCKED: {e}")

        try:
            delete_result = delete_old_records("inventory_log", 90)
            results.append(f"4. Cleanup: {delete_result}")
        except GovernanceBlocked as e:
            results.append(f"4. Delete BLOCKED: {e}")
    else:
        results.append("2–4. Stock sufficient — no restock needed.")

    print("\nAgent output:")
    for line in results:
        print(f"  {line}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Autonomica — Real Agent Demo")
    print("=" * 60)

    if os.getenv("OPENAI_API_KEY"):
        print("\nOpenAI API key found — running LangChain agent.")
        _run_langchain_agent()
    else:
        print("\nNo OPENAI_API_KEY — running in fallback mode.")
        _run_fallback()

    print("\n" + "=" * 60)
    print("  Done. All tool calls were governed by Autonomica.")
    print("=" * 60)
