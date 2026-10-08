"""
agents/rcm_supervisor.py - RCM denial agent (LangGraph).

Flow:
    triage ──APPEAL──▶ retrieve_policy ─▶ extract_evidence ──criteria met──▶ draft ─▶ verify
      │                                        │                              ▲         │
      │                                        └─gaps─▶ documentation_gap     └─revise──┤ (max 2)
      ├─RESUBMIT──▶ build_resubmission                                                 │
      ├─REP_CALL──▶ build_rep_call                                                     ▼
      └─NO_APPEAL─▶ build_no_appeal ───────────────────────────────────────▶ human_review (interrupt)
                                                                                      │
                                                         approve ─▶ commit ─(edited text fails verify)─▶ human_review
                                                         reject  ─▶ END (nothing written)

Principles:
- Triage BEFORE any LLM call. Most denials don't need a letter (CO-16 is resubmitted,
  frequency limits aren't appealed, systemic zero-pay goes to a rep call). That is the
  biggest cost lever, before any caching.
- The agent refuses to appeal when the chart doesn't support it (DOCUMENTATION_GAP).
- Nothing reaches Open Dental without a named human approver; edited text is re-verified.
- Latencies in traces are measured, never padded.
"""

from __future__ import annotations

import operator
import time
from datetime import datetime
from typing import Any, Annotated, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.llm import TemplateDrafter, get_default_drafter
from agents.retrieval import EvidenceExtractor, PolicyRetriever
from llm_cache.token_optimizer import PolicyContextCache, PromptCacheEconomics
from eval.grounding import GroundingVerifier
from integrations.mock_apis import ClinicalChart, DentalClaim, OpenDentalClient
from knowledge.rcm_reference import (
    PLAYBOOK, RESUBMISSION_CHECKLIST, ZERO_PAY_REP_CALL_THRESHOLD_PCT, Route, parse_adjustment_code,
)

MAX_REVISIONS = 2


# ==========================================
# State
# ==========================================

class RCMState(TypedDict, total=False):
    claim: Dict[str, Any]
    chart: Optional[Dict[str, Any]]
    payer_stats: Dict[str, Any]
    route: str
    triage: Dict[str, Any]
    policy_chunks: List[Dict[str, Any]]
    retrieval_cache_hit: bool
    evidence: Dict[str, Any]
    draft: str
    draft_backend: str
    revision_count: int
    verification: Dict[str, Any]
    work_item: Dict[str, Any]          # output for non-appeal routes
    review: Dict[str, Any]
    review_error: str
    status: str                        # PENDING_REVIEW | COMMITTED | REJECTED
    commit_result: Dict[str, Any]
    traces: Annotated[List[Dict[str, Any]], operator.add]
    usage: Annotated[List[Dict[str, Any]], operator.add]


def _claim(s: RCMState) -> DentalClaim:
    return DentalClaim.model_validate(s["claim"])


def _chart(s: RCMState) -> Optional[ClinicalChart]:
    return ClinicalChart.model_validate(s["chart"]) if s.get("chart") else None


def _trace(node: str, t0: float, detail: str, **extra) -> Dict[str, Any]:
    return {"node": node, "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "detail": detail, "ts": datetime.now().strftime("%H:%M:%S"), **extra}


# ==========================================
# Agent
# ==========================================

class RCMDenialAgent:
    def __init__(
        self,
        od_client: OpenDentalClient,
        drafter: Any = None,
        retrieval_cache: Optional[PolicyContextCache] = None,
        economics: Optional[PromptCacheEconomics] = None,
    ):
        self.od = od_client
        self.drafter = drafter or get_default_drafter()
        self.retriever = PolicyRetriever()
        self.extractor = EvidenceExtractor()
        self.verifier = GroundingVerifier()
        self.cache = retrieval_cache or PolicyContextCache()
        self.economics = economics or PromptCacheEconomics()
        self.graph = self._build().compile(checkpointer=InMemorySaver())

    # ------------------------------------------------------------------ nodes
    def _triage(self, s: RCMState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        claim = _claim(s)
        stats = self.od.get_payer_denial_stats(claim.payer_id)
        if not claim.denial_code:
            route, reason, entry = Route.NO_APPEAL, "No denied lines on this claim.", None
        else:
            group, carc = parse_adjustment_code(claim.denial_code)
            entry = PLAYBOOK.get(carc)
            if entry is None:
                route, reason = Route.NO_APPEAL, f"CARC {carc} not in playbook - manual review."
            elif entry.route == Route.APPEAL and stats["zero_pay_pct"] >= ZERO_PAY_REP_CALL_THRESHOLD_PCT:
                route = Route.REP_CALL
                reason = (f"{claim.payer_name} zero-paid {stats['zero_pay_pct']}% of lines in the last "
                          f"{stats['window_days']} days (>= {ZERO_PAY_REP_CALL_THRESHOLD_PCT:.0f}%). "
                          f"Systemic issue - call the payer rep before writing individual appeals.")
            else:
                route, reason = entry.route, entry.rationale
        triage = {
            "denial_code": claim.denial_code, "route": route, "reason": reason,
            "appeal_type": entry.appeal_type if entry else None,
            "allowed_at_issue": claim.allowed_at_issue,
        }
        return {"route": route, "triage": triage, "payer_stats": stats, "revision_count": 0,
                "traces": [_trace("triage", t0, f"{claim.denial_code} -> {route}", route=route)]}

    def _retrieve_policy(self, s: RCMState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        claim = _claim(s)
        line = claim.primary_denied_proc
        key = PolicyContextCache.key(claim.payer_id, line.proc_code, claim.carc)
        cached = self.cache.get(key)
        if cached is not None:
            return {"policy_chunks": cached, "retrieval_cache_hit": True,
                    "traces": [_trace("retrieve_policy", t0, f"Retrieval cache HIT {key}", cache="HIT")]}
        query = f"{line.proc_code} {line.proc_desc} {s['triage'].get('appeal_type') or ''} {line.payer_remark}"
        chunks = [c.to_dict() for c in self.retriever.search(query, cdt=line.proc_code, payer_id=claim.payer_id)]
        self.cache.put(key, chunks, forbidden_terms=[claim.patient_name, str(claim.patient_id)])
        return {"policy_chunks": chunks, "retrieval_cache_hit": False,
                "traces": [_trace("retrieve_policy", t0,
                                  f"BM25 top-{len(chunks)}: {', '.join(c['id'] for c in chunks)}", cache="MISS")]}

    def _extract_evidence(self, s: RCMState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        claim = _claim(s)
        ev = self.extractor.extract(claim.primary_denied_proc.proc_code, _chart(s))
        met = sum(1 for c in ev["criteria"] if c["status"] == "MET")
        return {"evidence": ev, "traces": [_trace(
            "extract_evidence", t0, f"{met}/{len(ev['criteria'])} criteria met; gaps: {len(ev['gaps'])}")]}

    def _ctx(self, s: RCMState) -> Dict[str, Any]:
        return {"claim": _claim(s), "chart": _chart(s), "evidence": s["evidence"],
                "policy_chunks": s.get("policy_chunks", []), "appeal_type": s["triage"].get("appeal_type")}

    def _draft(self, s: RCMState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        ctx = self._ctx(s)
        prior_v = s.get("verification")
        if s.get("draft") and prior_v and not prior_v["passed"]:
            res = self.drafter.revise(ctx, s["draft"], prior_v["violations"])
            rev = s.get("revision_count", 0) + 1
            label = f"Revision {rev} ({len(prior_v['violations'])} violations)"
        else:
            res = self.drafter.draft(ctx)
            rev = s.get("revision_count", 0)
            label = "Initial draft"
        cost = self.economics.record(res.usage, res.usage_source)
        return {
            "draft": res.text, "draft_backend": res.backend, "revision_count": rev,
            "usage": [{**res.usage, "source": res.usage_source, "cost_usd": cost["cost_usd"],
                       "saved_usd": cost["saved_usd"]}],
            "traces": [_trace("draft", t0, f"{label} via {res.backend}",
                              cache_read=res.usage.get("cache_read_input_tokens", 0) > 0)],
        }

    def _verify(self, s: RCMState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        v = self.verifier.verify(s["draft"], _claim(s), _chart(s), s.get("evidence"), s.get("policy_chunks"))
        return {"verification": v, "traces": [_trace(
            "verify", t0, f"{'PASS' if v['passed'] else 'FAIL'} - faithfulness {v['faithfulness']:.0%}, "
                          f"{len(v['violations'])} violations", passed=v["passed"])]}

    def _documentation_gap(self, s: RCMState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        claim, chart = _claim(s), _chart(s)
        ev = s["evidence"]
        contradicted = [c for c in ev["criteria"] if c["status"] == "CONTRADICTED"]
        item = {
            "type": Route.DOC_GAP,
            "summary": "Chart does not support an appeal. Do not appeal; request a provider addendum only "
                       "if it reflects what was actually done.",
            "gaps": ev["gaps"],
            "contradicting_evidence": [q for c in contradicted for q in c["quotes"]],
            "addressed_to": chart.provider_name if chart else "Treating provider",
            "checklist": RESUBMISSION_CHECKLIST.get(claim.primary_denied_proc.proc_code, []),
        }
        return {"route": Route.DOC_GAP, "work_item": item,
                "traces": [_trace("documentation_gap", t0, f"{len(ev['gaps'])} unmet required criteria")]}

    def _build_resubmission(self, s: RCMState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        claim = _claim(s)
        entry = PLAYBOOK["16"]
        codes = sorted({p.proc_code for p in claim.denied_procs})
        item = {
            "type": Route.RESUBMIT,
            "summary": entry.rationale,
            "lines": [f"CDT {p.proc_code} {p.tooth_num or ''} - payer remark: {p.payer_remark}"
                      for p in claim.denied_procs],
            "actions": list(entry.actions),
            "checklist": [i for c in codes for i in RESUBMISSION_CHECKLIST.get(c, [])],
        }
        return {"work_item": item, "traces": [_trace("build_resubmission", t0,
                                                     f"{len(item['checklist'])} attachments required")]}

    def _build_rep_call(self, s: RCMState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        claim, st = _claim(s), s["payer_stats"]
        item = {
            "type": Route.REP_CALL,
            "summary": s["triage"]["reason"],
            "call_script": [
                f"Reference: {st['zero_pay_lines']} of {st['lines']} lines zero-paid in {st['window_days']} days.",
                "Ask whether a configuration/fee-schedule load or credentialing issue is affecting our TINs.",
                "Request a reprocessing project for all affected claims instead of individual appeals.",
                f"Log the call reference number on claim {claim.claim_id} and on the payer record.",
            ],
        }
        return {"work_item": item, "traces": [_trace("build_rep_call", t0, f"zero-pay {st['zero_pay_pct']}%")]}

    def _build_no_appeal(self, s: RCMState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        claim = _claim(s)
        entry = PLAYBOOK.get(claim.carc) if claim.carc else None
        item = {"type": Route.NO_APPEAL, "summary": s["triage"]["reason"],
                "actions": list(entry.actions) if entry else ["Manual review."]}
        return {"work_item": item, "traces": [_trace("build_no_appeal", t0, s["triage"]["reason"][:80])]}

    def _human_review(self, s: RCMState) -> Dict[str, Any]:
        # interrupt() pauses the graph; the node re-runs on resume, so nothing before it has side effects.
        payload = {
            "route": s["route"],
            "draft": s.get("draft"),
            "verification": s.get("verification"),
            "work_item": s.get("work_item"),
            "error": s.get("review_error"),
        }
        decision = interrupt(payload)
        return {"review": decision, "review_error": "",
                "traces": [{"node": "human_review", "latency_ms": 0.0, "ts": datetime.now().strftime("%H:%M:%S"),
                            "detail": f"{decision.get('action', '?').upper()} by {decision.get('reviewer') or 'unknown'}"}]}

    def _commit(self, s: RCMState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        claim = _claim(s)
        decision = s["review"]
        reviewer = (decision.get("reviewer") or "").strip()
        if not reviewer:
            return {"review_error": "Reviewer name is required.", "status": "PENDING_REVIEW",
                    "traces": [_trace("commit", t0, "Blocked: no reviewer")]}

        if s["route"] == Route.APPEAL:
            text = decision.get("edited_text") or s["draft"]
            v = self.verifier.verify(text, claim, _chart(s), s.get("evidence"), s.get("policy_chunks"))
            if not v["passed"]:
                msg = "; ".join(f"{x['type']}: {x['span']}" for x in v["violations"][:4])
                return {"verification": v, "review_error": f"Edited letter failed verification - {msg}",
                        "status": "PENDING_REVIEW", "traces": [_trace("commit", t0, "Blocked: edited text failed")]}
            note = f"Appeal approved for {claim.primary_denied_proc.proc_code} " \
                   f"(allowed ${claim.allowed_at_issue:,.2f}). Letter: {text[:60]}"
        else:
            note = s["work_item"]["summary"]

        res = self.od.post_claim_tracking(claim.claim_id, s["route"], note, reviewer)
        return {"status": "COMMITTED", "commit_result": res, "review_error": "",
                "traces": [_trace("commit", t0, f"Open Dental -> {res['status']}")]}

    def _rejected(self, s: RCMState) -> Dict[str, Any]:
        return {"status": "REJECTED", "traces": [{"node": "rejected", "latency_ms": 0.0,
                                                  "ts": datetime.now().strftime("%H:%M:%S"),
                                                  "detail": "Nothing written to Open Dental."}]}

    # ------------------------------------------------------------------ routing
    @staticmethod
    def _after_triage(s: RCMState) -> str:
        return {Route.APPEAL: "retrieve_policy", Route.RESUBMIT: "build_resubmission",
                Route.REP_CALL: "build_rep_call"}.get(s["route"], "build_no_appeal")

    @staticmethod
    def _after_evidence(s: RCMState) -> str:
        return "draft" if s["evidence"]["required_met"] else "documentation_gap"

    @staticmethod
    def _after_verify(s: RCMState) -> str:
        if s["verification"]["passed"] or s.get("revision_count", 0) >= MAX_REVISIONS:
            return "human_review"
        return "draft"

    @staticmethod
    def _after_review(s: RCMState) -> str:
        return "commit" if s["review"].get("action") == "approve" else "rejected"

    @staticmethod
    def _after_commit(s: RCMState) -> str:
        return END if s.get("status") == "COMMITTED" else "human_review"

    def _build(self) -> StateGraph:
        g = StateGraph(RCMState)
        for name in ["triage", "retrieve_policy", "extract_evidence", "draft", "verify", "documentation_gap",
                     "build_resubmission", "build_rep_call", "build_no_appeal", "human_review", "commit",
                     "rejected"]:
            g.add_node(name, getattr(self, f"_{name}"))
        g.add_edge(START, "triage")
        g.add_conditional_edges("triage", self._after_triage,
                                ["retrieve_policy", "build_resubmission", "build_rep_call", "build_no_appeal"])
        g.add_edge("retrieve_policy", "extract_evidence")
        g.add_conditional_edges("extract_evidence", self._after_evidence, ["draft", "documentation_gap"])
        g.add_edge("draft", "verify")
        g.add_conditional_edges("verify", self._after_verify, ["draft", "human_review"])
        for n in ["documentation_gap", "build_resubmission", "build_rep_call", "build_no_appeal"]:
            g.add_edge(n, "human_review")
        g.add_conditional_edges("human_review", self._after_review, ["commit", "rejected"])
        g.add_conditional_edges("commit", self._after_commit, ["human_review", END])
        g.add_edge("rejected", END)
        return g

    # ------------------------------------------------------------------ public API
    @staticmethod
    def thread_config(thread_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def start(self, claim_id: int, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Runs until the human-review interrupt. Returns the paused state."""
        claim = self.od.get_claim(claim_id)
        if claim is None:
            raise ValueError(f"Claim {claim_id} not found.")
        chart = self.od.get_clinical_chart(claim_id)
        tid = thread_id or f"claim-{claim_id}-{time.time_ns()}"
        cfg = self.thread_config(tid)
        self.graph.invoke({"claim": claim.model_dump(), "chart": chart.model_dump() if chart else None,
                           "status": "PENDING_REVIEW", "traces": [], "usage": []}, cfg)
        return {"thread_id": tid, **self.get_state(tid)}

    def resume(self, thread_id: str, action: str, reviewer: str, edited_text: Optional[str] = None) -> Dict[str, Any]:
        if action not in ("approve", "reject"):
            raise ValueError("action must be 'approve' or 'reject'")
        self.graph.invoke(Command(resume={"action": action, "reviewer": reviewer, "edited_text": edited_text}),
                          self.thread_config(thread_id))
        return {"thread_id": thread_id, **self.get_state(thread_id)}

    def get_state(self, thread_id: str) -> Dict[str, Any]:
        snap = self.graph.get_state(self.thread_config(thread_id))
        values = dict(snap.values)
        values["awaiting_review"] = bool(snap.next) and "human_review" in snap.next
        return values

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
