"""Trace-driven LangSmith evaluation scaffold for OpenGPT.

This script does not require a separate DeepEval suite. It reads the runs that
are already being traced into LangSmith, scores them with lightweight node-level
checks, and writes a JSON report that you can compare over time.

Typical usage:

    cd backend
    python -m evals.online_eval
    python -m evals.online_eval --project OpenGPT --limit 100

The default filters are aligned to the traced nodes in this repo:
    - chat_node / My GPT
    - chat.send_message / chat.regenerate_message (if you add traces there)
    - rag.retrieve_from_document / rag.ingest_document (if you add traces there)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from langsmith import Client

from app.observability import configure_langsmith

configure_langsmith()

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.getenv("OPENGPT_EVAL_RESULTS_DIR", REPO_ROOT / "evals" / "results")).resolve()
DEFAULT_PROJECT = os.getenv("LANGSMITH_PROJECT", "OpenGPT")
DEFAULT_LIMIT = int(os.getenv("OPENGPT_EVAL_LIMIT", "50"))
NODE_NAMES = {
    "chat_node",
    "My GPT",
    "chat.send_message",
    "chat.regenerate_message",
    "rag.retrieve_from_document",
    "rag.ingest_document",
}


@dataclass
class TraceCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class TraceResult:
    run_id: str
    trace_id: str | None
    name: str
    run_type: str | None
    kind: str
    score: float
    checks: list[TraceCheck]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.score >= 0.5 and all(c.passed for c in self.checks)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if hasattr(value, "dict") and callable(getattr(value, "dict")):
        try:
            maybe = value.dict()
            if isinstance(maybe, dict):
                return maybe
        except Exception:
            pass
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        try:
            maybe = value.model_dump()
            if isinstance(maybe, dict):
                return maybe
        except Exception:
            pass
    return {}


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        bits = []
        for key in ("response", "answer", "content", "output", "text", "final", "final_report"):
            if key in value:
                bits.append(_flatten_text(value.get(key)))
        if bits:
            return "\n".join(bit for bit in bits if bit)
        return "\n".join(_flatten_text(v) for v in value.values() if v)
    if isinstance(value, (list, tuple, set)):
        return "\n".join(_flatten_text(v) for v in value if v)
    return str(value)


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = _flatten_text(value).strip()
        if text:
            return text
    return ""


def _kind_from_name(name: str) -> str:
    low = name.lower()
    if "ingest" in low:
        return "document_ingest"
    if "retrieve" in low or "query_document" in low:
        return "document_retrieval"
    if "chat" in low or low == "my gpt":
        return "chat"
    return "other"


def _score_chat_run(inputs: dict[str, Any], outputs: dict[str, Any], error: str | None) -> tuple[float, list[TraceCheck]]:
    text = _first_non_empty(outputs.get("response"), outputs.get("answer"), outputs.get("content"), outputs)
    checks = [TraceCheck("answer_present", bool(text), "non-empty response" if text else "empty response")]
    if error:
        checks.append(TraceCheck("error_free", False, error))
    else:
        checks.append(TraceCheck("error_free", True, "no trace error"))

    tool_used = _first_non_empty(outputs.get("tool_used"), outputs.get("has_tool_calls"), outputs.get("tools"))
    if tool_used:
        checks.append(TraceCheck("tool_usage_logged", True, f"tool signal: {tool_used}"))
    else:
        checks.append(TraceCheck("tool_usage_logged", True, "no tool signal in this run"))

    length = len(text.split())
    checks.append(TraceCheck("answer_has_some_content", length >= 3, f"{length} words" if text else "no answer"))

    score = 1.0 if text and not error else 0.0
    if text and length < 3:
        score = 0.5
    return score, checks


def _score_document_retrieval(outputs: dict[str, Any], error: str | None) -> tuple[float, list[TraceCheck]]:
    context = outputs.get("context") or []
    source_file = outputs.get("source_file")
    checks = [
        TraceCheck("context_present", bool(context), f"{len(context)} chunks" if context else "empty context"),
        TraceCheck("source_recorded", bool(source_file), source_file or "missing source_file"),
    ]
    if error:
        checks.append(TraceCheck("error_free", False, error))
    else:
        checks.append(TraceCheck("error_free", True, "no trace error"))
    score = 1.0 if context and source_file and not error else 0.0
    return score, checks


def _score_document_ingest(outputs: dict[str, Any], error: str | None) -> tuple[float, list[TraceCheck]]:
    chunks = int(outputs.get("chunks") or 0)
    documents = int(outputs.get("documents") or 0)
    checks = [
        TraceCheck("document_count_positive", documents > 0, f"documents={documents}"),
        TraceCheck("chunk_count_positive", chunks > 0, f"chunks={chunks}"),
    ]
    if error:
        checks.append(TraceCheck("error_free", False, error))
    else:
        checks.append(TraceCheck("error_free", True, "no trace error"))
    score = 1.0 if documents > 0 and chunks > 0 and not error else 0.0
    return score, checks


def _row_from_run(run: Any) -> TraceResult | None:
    name = getattr(run, "name", "") or ""
    if name not in NODE_NAMES:
        return None

    inputs = _as_dict(getattr(run, "inputs", None))
    outputs = _as_dict(getattr(run, "outputs", None))
    error = getattr(run, "error", None)
    kind = _kind_from_name(name)

    if kind == "document_retrieval":
        score, checks = _score_document_retrieval(outputs, error)
    elif kind == "document_ingest":
        score, checks = _score_document_ingest(outputs, error)
    else:
        score, checks = _score_chat_run(inputs, outputs, error)

    return TraceResult(
        run_id=str(getattr(run, "id", "")),
        trace_id=str(getattr(run, "trace_id", "")) or None,
        name=name,
        run_type=str(getattr(run, "run_type", "")) or None,
        kind=kind,
        score=score,
        checks=checks,
        inputs=inputs,
        outputs=outputs,
        error=str(error) if error else None,
    )


def _iter_runs(client: Client, project_name: str, limit: int) -> Iterable[Any]:
    try:
        runs = client.list_runs(project_name=project_name)
    except TypeError:
        runs = client.list_runs(project_name)
    for idx, run in enumerate(runs):
        if limit and idx >= limit:
            break
        yield run


def evaluate_project(project_name: str = DEFAULT_PROJECT, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    client = Client()
    rows = []
    for run in _iter_runs(client, project_name, limit):
        row = _row_from_run(run)
        if row is not None:
            rows.append(row)

    grouped: dict[str, list[TraceResult]] = defaultdict(list)
    for row in rows:
        grouped[row.kind].append(row)

    summary = {}
    for kind, items in grouped.items():
        scores = [item.score for item in items]
        summary[kind] = {
            "n": len(items),
            "pass_rate": 100.0 * sum(1 for item in items if item.passed) / len(items) if items else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else math.nan,
            "failed_runs": [item.run_id for item in items if not item.passed],
        }

    report = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "project_name": project_name,
            "limit": limit,
        },
        "summary": summary,
        "rows": [
            {
                **asdict(row),
                "checks": [asdict(check) for check in row.checks],
            }
            for row in rows
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"langsmith-online-eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== LangSmith trace evaluation ===")
    print(f"project: {project_name}")
    print(f"runs scanned: {len(rows)}")
    for kind, stats in summary.items():
        print(f"  {kind:<20} pass_rate={stats['pass_rate']:.0f}%  avg_score={stats['avg_score']:.2f}  n={stats['n']}")
    print(f"report written to: {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    evaluate_project(project_name=args.project, limit=args.limit)


if __name__ == "__main__":
    main()
