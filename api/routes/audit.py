"""Audit log export endpoint."""
from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from api.dependencies import get_gov

router = APIRouter(prefix="/audit", tags=["audit"])

_CONTENT_TYPES = {
    "jsonl": "application/x-ndjson",
    "json": "application/json",
    "csv": "text/csv",
}

_FILENAMES = {
    "jsonl": "autonomica_audit.jsonl",
    "json": "autonomica_audit.json",
    "csv": "autonomica_audit.csv",
}

_CSV_FIELDS = [
    "event", "timestamp", "action_id", "agent_id", "agent_name",
    "tool_name", "action_type", "composite_score", "governance_mode",
    "approved", "decision_time_ms", "trust_score", "vagal_tone",
    "reason", "notes",
]


@router.get("/export", summary="Export audit log for compliance")
async def export_audit(
    fmt: str = "jsonl",
    gov=Depends(get_gov),
) -> Response:
    """
    Download the structured audit log.

    ``fmt`` options:
    - ``jsonl`` (default) — newline-delimited JSON, one event per line.
    - ``json``           — pretty-printed JSON array.
    - ``csv``            — spreadsheet-friendly CSV.
    """
    if fmt not in _CONTENT_TYPES:
        fmt = "jsonl"

    entries = gov._audit.read_entries()

    if fmt == "json":
        body = json.dumps(entries, indent=2)
    elif fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            writer.writerow({f: entry.get(f, "") for f in _CSV_FIELDS})
        body = buf.getvalue()
    else:  # jsonl
        body = "\n".join(json.dumps(e) for e in entries)

    filename = _FILENAMES[fmt]
    return Response(
        content=body,
        media_type=_CONTENT_TYPES[fmt],
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Total-Events": str(len(entries)),
        },
    )
