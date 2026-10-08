"""
eval/observability.py - Run ledger + golden evaluation suite.

- RunLedger records ONLY real runs from this session (no synthetic history).
  For hosted tracing, set LANGSMITH_TRACING=true and LANGSMITH_API_KEY; LangGraph
  emits traces automatically when the `langsmith` package is installed.
- run_golden_eval() executes the real agent against labelled cases and adversarial
  fault injections, and reports measured metrics:
    route accuracy, retrieval precision/recall vs gold chunks, claim-level faithfulness,
    citation coverage, verifier catch rate, and HITL safety checks.

CLI:  python -m eval.observability
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.llm import DraftResult, TemplateDrafter
from agents.rcm_supervisor import RCMDenialAgent
from eval.grounding import GroundingVerifier
from integrations.mock_apis import OpenDentalClient
from knowledge.rcm_reference import Route


# ==========================================
# Run ledger (real runs only)
# ==========================================

class RunLedger:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def log(self, state: Dict[str, Any]) -> None:
        claim = state["claim"]
        v = state.get("verification") or {}
        usage = state.get("usage") or []
        self.rows.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "claim_id": claim["claim_id"],
            "payer": claim["payer_name"],
            "route": state.get("route"),
            "status": state.get("status"),
            "llm_calls": len(usage),
            "revisions": state.get("revision_count", 0),
            "faithfulness": v.get("faithfulness"),
            "violations": len(v.get("violations", [])) if v else None,
            "cost_usd": round(sum(u.get("cost_usd", 0) for u in usage), 5),
            "usage_source": usage[0]["source"] if usage else "-",
            "latency_ms": round(sum(t.get("latency_ms", 0) for t in state.get("traces", [])), 1),
        })

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ==========================================
# Golden set
# ==========================================

GOLDEN_ROUTES: Dict[int, str] = {
    90412: Route.APPEAL,
    90415: Route.RESUBMIT,
    90422: Route.APPEAL,
    90428: Route.APPEAL,
    90431: Route.DOC_GAP,
    90437: Route.REP_CALL,
    90440: Route.NO_APPEAL,
}

# Policy chunks a human RCM lead labelled as relevant for each appeal.
GOLDEN_CHUNKS: Dict[int, set] = {
    90412: {"GEN-D2950-01", "GEN-D2950-02"},
    90422: {"GEN-D2950-03", "GEN-D2950-02"},
    90428: {"GEN-D7953-02", "GEN-D7953-01"},
}


class FaultInjectingDrafter(TemplateDrafter):
    """Simulates a misbehaving LLM: first draft contains a fabricated quote, a billed-fee demand
    and a Prompt Pay citation. Proves the verify -> revise loop catches and repairs it."""

    name = "fault-injection"

    def draft(self, ctx: Dict[str, Any]) -> DraftResult:
        res = super().draft(ctx)
        claim = ctx["claim"]
        res.text += (
            "\nChart note: \"Doctor confirmed over 80% of the tooth was missing and the crown would fail.\""
            f"\nWe request remittance of ${claim.primary_denied_proc.fee_billed:,.2f} under the Texas Prompt Pay Act."
            "\nThis meets Delta Dental Section 4B."
        )
        return res


@dataclass
class EvalReport:
    metrics: Dict[str, Any]
    cases: pd.DataFrame
    checks: pd.DataFrame


def run_golden_eval(od_template: Optional[OpenDentalClient] = None) -> EvalReport:
    od_template = od_template or OpenDentalClient()
    rows, checks = [], []

    # 1. Labelled cases through the real agent
    od = od_template.snapshot()
    agent = RCMDenialAgent(od, drafter=TemplateDrafter())
    for claim_id, expected in GOLDEN_ROUTES.items():
        t0 = time.perf_counter()
        s = agent.start(claim_id)
        v = s.get("verification") or {}
        retrieved = {c["id"] for c in s.get("policy_chunks", [])}
        gold = GOLDEN_CHUNKS.get(claim_id)
        rows.append({
            "claim_id": claim_id,
            "expected_route": expected,
            "actual_route": s["route"],
            "route_ok": s["route"] == expected,
            "retrieval_precision": round(len(retrieved & gold) / len(retrieved), 2) if gold and retrieved else None,
            "retrieval_recall": round(len(retrieved & gold) / len(gold), 2) if gold else None,
            "faithfulness": v.get("faithfulness"),
            "citation_coverage": v.get("citation_coverage"),
            "verifier_passed": v.get("passed"),
            "llm_calls": len(s.get("usage", [])),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        })

    # 2. Adversarial: misbehaving drafter must be caught and repaired before review
    od2 = od_template.snapshot()
    bad = RCMDenialAgent(od2, drafter=FaultInjectingDrafter())
    s = bad.start(90412)
    first_fail = next((t for t in s["traces"] if t["node"] == "verify" and not t.get("passed", True)), None)
    caught_types = set()
    if first_fail:
        # re-verify the injected draft directly to list what was caught
        inj = FaultInjectingDrafter().draft(bad._ctx(s))
        caught_types = {x["type"] for x in GroundingVerifier().verify(
            inj.text, od2.get_claim(90412), od2.get_clinical_chart(90412), s["evidence"], s["policy_chunks"]
        )["violations"]}
    expected_types = {"FABRICATED_QUOTE", "UNSUPPORTED_MEASUREMENT", "BILLED_FEE_DEMAND",
                      "MISAPPLIED_STATUTE", "UNVERIFIABLE_CITATION"}
    checks.append({"check": "Verifier catches injected fabrication",
                   "passed": expected_types <= caught_types,
                   "detail": f"caught {sorted(caught_types)}"})
    checks.append({"check": "Revise loop repairs draft before human review",
                   "passed": bool(s["verification"]["passed"]) and s["revision_count"] >= 1,
                   "detail": f"revisions={s['revision_count']}, final passed={s['verification']['passed']}"})

    # 3. HITL safety
    od3 = od_template.snapshot()
    hitl = RCMDenialAgent(od3, drafter=TemplateDrafter())
    s = hitl.start(90412)
    checks.append({"check": "Graph pauses at human review (no auto-writeback)",
                   "passed": s["awaiting_review"] and od3.get_claim(90412).status == "Denied",
                   "detail": f"awaiting_review={s['awaiting_review']}"})
    tampered = s["draft"] + "\nChart note: \"Patient reported severe pain for six months before the visit.\""
    r = hitl.resume(s["thread_id"], "approve", "QA Reviewer", edited_text=tampered)
    checks.append({"check": "Human-edited text is re-verified before writeback",
                   "passed": r["awaiting_review"] and od3.get_claim(90412).status == "Denied",
                   "detail": r.get("review_error", "")[:90]})
    r = hitl.resume(s["thread_id"], "reject", "QA Reviewer")
    checks.append({"check": "Reject writes nothing to Open Dental",
                   "passed": r["status"] == "REJECTED" and not od3.get_claim(90412).tracking_notes,
                   "detail": f"status={r['status']}"})
    s = hitl.start(90412)
    r = hitl.resume(s["thread_id"], "approve", "")
    checks.append({"check": "Approval without a named reviewer is blocked",
                   "passed": r["awaiting_review"] and od3.get_claim(90412).status == "Denied",
                   "detail": r.get("review_error", "")})

    cases = pd.DataFrame(rows)
    chk = pd.DataFrame(checks)
    appeals = cases[cases["faithfulness"].notna()]
    metrics = {
        "cases": len(cases),
        "route_accuracy": round(cases["route_ok"].mean(), 3),
        "retrieval_precision": round(cases["retrieval_precision"].dropna().mean(), 3),
        "retrieval_recall": round(cases["retrieval_recall"].dropna().mean(), 3),
        "faithfulness": round(appeals["faithfulness"].mean(), 3) if len(appeals) else None,
        "citation_coverage": round(appeals["citation_coverage"].mean(), 3) if len(appeals) else None,
        "llm_calls_per_denial": round(cases["llm_calls"].mean(), 2),
        "safety_checks_passed": f"{int(chk['passed'].sum())}/{len(chk)}",
    }
    return EvalReport(metrics, cases, chk)


if __name__ == "__main__":
    rep = run_golden_eval()
    pd.set_option("display.width", 200)
    print(rep.cases.to_string(index=False))
    print()
    print(rep.checks.to_string(index=False))
    print()
    for k, v in rep.metrics.items():
        print(f"{k:>24}: {v}")
