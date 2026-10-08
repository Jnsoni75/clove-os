"""
agents/llm.py - Appeal drafting backends.

- TemplateDrafter  (default, offline): deterministic, composes letters ONLY from verified
  evidence. Token usage is estimated from the exact prompts the live backend would send.
- AnthropicDrafter (live): used automatically when ANTHROPIC_API_KEY is set and the
  `anthropic` package is installed. The static prefix carries cache_control so repeated
  denials reuse the cached prefix; cost is computed from the API's real `usage` object.

Both backends see de-identified input ([PATIENT], [PATIENT_ID]) - minimum necessary,
even behind a BAA - and both are checked by the same GroundingVerifier.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_cache.token_optimizer import PrefixCacheSimulator
from integrations.mock_apis import ClinicalChart, DentalClaim
from knowledge.rcm_reference import CARC, CDT, PLAYBOOK, POLICY_CORPUS

DEFAULT_MODEL = os.environ.get("CLOVE_LLM_MODEL", "claude-sonnet-4-5")


# ==========================================
# De-identification
# ==========================================

class Deidentifier:
    def __init__(self, claim: DentalClaim):
        self._map = {claim.patient_name: "[PATIENT]", str(claim.patient_id): "[PATIENT_ID]"}

    def scrub(self, text: str) -> str:
        for real, token in self._map.items():
            text = text.replace(real, token)
        return text

    def restore(self, text: str) -> str:
        for real, token in self._map.items():
            text = text.replace(token, real)
        return text


# ==========================================
# Prompts (static prefix is byte-identical across claims -> cacheable)
# ==========================================

_INSTRUCTIONS = """You are the appeals writer for a centralized dental revenue cycle team at a
multi-office dental support organization. You draft first-level provider appeals that a
certified billing specialist will review before anything is sent.

Hard rules - a deterministic verifier rejects drafts that break them:
1. Quote clinical documentation ONLY by copying sentences from EVIDENCE.quotes verbatim,
   inside double quotes. Never paraphrase inside quotes. Never invent quotes.
2. Do not introduce any measurement (mm, %), tooth number or dollar amount that is not
   present in EVIDENCE, CLAIM or POLICY. If something is unknown, omit it.
3. Request reprocessing at the contracted allowed amount given in CLAIM.allowed_at_issue,
   subject to the patient's deductible and coinsurance. Never request the billed fee.
4. Do not cite state Prompt Pay statutes. They govern payment timing for clean claims,
   not appeal decisions, and do not apply to self-funded ERISA plans.
5. Cite 29 C.F.R. 2560.503-1 ONLY when CLAIM.plan_type is "erisa_employer_group". In that
   case the office writes as the patient's authorized representative under the assignment
   of benefits on file, and asks that an upheld denial identify the plan provision and
   clinical criteria relied upon. Otherwise refer to the plan's appeal procedures and
   applicable state law without naming statutes.
6. Never cite payer policy section numbers. Refer to "your published clinical criteria for
   CDT <code>". The POLICY text below is an illustrative composite, not the payer's document.
7. Keep [PATIENT] and [PATIENT_ID] placeholders exactly as written.
8. Plain, professional tone. No superlatives ("unequivocally", "exhaustive"). Under 350 words.

Structure: header block (payer, claim, patient placeholder, date of service, provider, line
under appeal with adjustment code); one-paragraph basis for appeal; criteria list with quoted
evidence; request; regulatory paragraph per rule 5; enclosures; signature from the
Centralized Revenue Cycle Team on behalf of the treating provider.
"""


def build_static_prefix() -> str:
    carc = "\n".join(f"- CARC {k}: {v}" for k, v in CARC.items())
    cdt = "\n".join(f"- {k}: {v}" for k, v in CDT.items())
    playbook = "\n".join(f"- CARC {k}: {v.route} - {v.rationale}" for k, v in PLAYBOOK.items())
    policy = "\n".join(f"[{c.id}] ({c.cdt}, {c.topic}) {c.text}" for c in POLICY_CORPUS)
    return (f"{_INSTRUCTIONS}\n\nREFERENCE - CLAIM ADJUSTMENT REASON CODES\n{carc}\n\n"
            f"REFERENCE - CDT DESCRIPTORS\n{cdt}\n\nREFERENCE - DENIAL PLAYBOOK\n{playbook}\n\n"
            f"POLICY (illustrative composites)\n{policy}\n")


STATIC_PREFIX = build_static_prefix()


def build_dynamic_prompt(ctx: Dict[str, Any], deid: Deidentifier) -> str:
    claim: DentalClaim = ctx["claim"]
    chart: Optional[ClinicalChart] = ctx["chart"]
    payload = {
        "CLAIM": {
            "claim_id": claim.claim_id, "patient": "[PATIENT]", "patient_id": "[PATIENT_ID]",
            "payer": claim.payer_name, "plan_type": claim.plan_type, "date_of_service": claim.date_of_service,
            "office": claim.clinic_name,
            "provider": chart.provider_name if chart else None, "provider_npi": chart.provider_npi if chart else None,
            "denied_lines": [{"cdt": p.proc_code, "desc": p.proc_desc, "tooth": p.tooth_num,
                              "adjustment": p.adjustment_code, "payer_remark": p.payer_remark}
                             for p in claim.denied_procs],
            "paid_lines_same_date": [{"cdt": p.proc_code, "tooth": p.tooth_num}
                                     for p in claim.procs if not p.is_denied],
            "allowed_at_issue": f"{claim.allowed_at_issue:,.2f}",
        },
        "APPEAL_TYPE": ctx.get("appeal_type"),
        "EVIDENCE": [{"criterion": c["text"], "quotes": c["quotes"], "structured": c["structured"]}
                     for c in ctx["evidence"]["criteria"] if c["status"] == "MET"],
        "POLICY_IDS": [c["id"] for c in ctx["policy_chunks"]],
        "ENCLOSURES": _enclosures(chart),
    }
    return deid.scrub(json.dumps(payload, indent=1))


def _enclosures(chart: Optional[ClinicalChart]) -> List[str]:
    out = ["Clinical progress notes for the date of service"]
    if chart and chart.radiograph_attached:
        out.append("Radiograph(s)")
    if chart and chart.intraoral_photos_attached:
        out.append("Intraoral photograph(s)")
    if chart and chart.probing_depths:
        out.append("Periodontal charting")
    return out


# ==========================================
# Draft result + backends
# ==========================================

@dataclass
class DraftResult:
    text: str
    backend: str
    usage: Dict[str, int]
    usage_source: str          # "measured" | "estimated"
    meta: Dict[str, Any] = field(default_factory=dict)


class TemplateDrafter:
    """Deterministic offline drafter. Can only say what the evidence says."""

    name = "offline-template"

    def __init__(self, simulator: Optional[PrefixCacheSimulator] = None):
        self.sim = simulator or PrefixCacheSimulator()

    def _compose(self, ctx: Dict[str, Any]) -> str:
        claim: DentalClaim = ctx["claim"]
        chart: Optional[ClinicalChart] = ctx["chart"]
        line = claim.primary_denied_proc
        site = f"tooth #{line.tooth_num}" if line.tooth_num and line.tooth_num.isdigit() else \
            (f"site {line.tooth_num}" if line.tooth_num else "")
        provider = chart.provider_name if chart else "Treating provider"
        npi = chart.provider_npi if chart else "on file"
        paid_same_day = [p for p in claim.procs if not p.is_denied]

        out: List[str] = [
            f"{claim.payer_name} - Provider Appeals",
            f"RE: Request for reconsideration, Claim {claim.claim_id}",
            f"Patient: [PATIENT] (ID [PATIENT_ID]) | Date of service: {claim.date_of_service}",
            f"Treating provider: {provider} (NPI {npi}) | Office: {claim.clinic_name}",
            f"Line under appeal: CDT {line.proc_code} ({line.proc_desc}) {site}, adjusted {line.adjustment_code}",
            "",
        ]
        if claim.carc == "97" and paid_same_day:
            other = ", ".join(f"CDT {p.proc_code}" for p in paid_same_day)
            out.append(
                f"We request reconsideration of the {line.adjustment_code} adjustment. CDT {line.proc_code} is a "
                f"separately reportable procedure from {other} performed on the same date, and the clinical "
                f"record documents it as clinically distinct and necessary:"
            )
        else:
            out.append(
                f"We request reconsideration of the {line.adjustment_code} adjustment. The clinical record for "
                f"this date of service documents that your published clinical criteria for CDT {line.proc_code} "
                f"were met:"
            )
        out.append("")
        for c in ctx["evidence"]["criteria"]:
            if c["status"] != "MET":
                continue
            out.append(f"- {c['text']}.")
            for q in c["quotes"]:
                out.append(f"  Chart note: \"{q}\"")
            for s in c["structured"]:
                out.append(f"  Record: {s}.")
        out += [
            "",
            f"Please reprocess CDT {line.proc_code} and remit per our participating provider agreement. "
            f"Contracted allowed amount at issue: ${claim.allowed_at_issue:,.2f}, subject to the patient's "
            f"applicable deductible and coinsurance.",
            "",
        ]
        if claim.plan_type == "erisa_employer_group":
            out.append(
                f"{claim.clinic_name} submits this appeal as the patient's authorized representative under the "
                f"assignment of benefits on file (29 C.F.R. 2560.503-1(b)(4)). If the denial is upheld, please "
                f"identify the specific plan provision and any internal rule, guideline or clinical criterion "
                f"relied upon, as required by 29 C.F.R. 2560.503-1(g)."
            )
        else:
            out.append(
                "We ask that this appeal be reviewed under the plan's provider appeal procedures and applicable "
                "state law. If the denial is upheld, please identify the specific plan provision and clinical "
                "criteria relied upon."
            )
        out += ["", "Enclosures: " + "; ".join(_enclosures(chart)), "",
                "Centralized Revenue Cycle Team, Clove Dental",
                f"On behalf of {provider}"]
        return "\n".join(out)

    def _result(self, ctx: Dict[str, Any], text: str, meta: Dict[str, Any]) -> DraftResult:
        deid = Deidentifier(ctx["claim"])
        dynamic = build_dynamic_prompt(ctx, deid)
        usage = self.sim.usage_for(STATIC_PREFIX, dynamic, text)
        return DraftResult(deid.restore(text), self.name, usage, "estimated", meta)

    def draft(self, ctx: Dict[str, Any]) -> DraftResult:
        return self._result(ctx, self._compose(ctx), {"mode": "draft"})

    def revise(self, ctx: Dict[str, Any], prior: str, violations: List[Dict[str, str]]) -> DraftResult:
        """Deterministic repair: drop any line containing a violating span."""
        spans = [v["span"].lower() for v in violations if v.get("span")]
        kept = [ln for ln in prior.splitlines() if not any(s in ln.lower() for s in spans)]
        deid = Deidentifier(ctx["claim"])
        return self._result(ctx, deid.scrub("\n".join(kept)),
                            {"mode": "revise", "removed_lines": len(prior.splitlines()) - len(kept)})


class AnthropicDrafter:
    """Live Claude drafter with prompt caching on the static prefix."""

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1200):
        import anthropic  # imported lazily so offline mode has no dependency
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def _call(self, messages: List[Dict[str, Any]]) -> Any:
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=[{"type": "text", "text": STATIC_PREFIX, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )

    @staticmethod
    def _usage(resp: Any) -> Dict[str, int]:
        u = resp.usage
        return {
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
        }

    def draft(self, ctx: Dict[str, Any]) -> DraftResult:
        deid = Deidentifier(ctx["claim"])
        resp = self._call([{"role": "user", "content": build_dynamic_prompt(ctx, deid)}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return DraftResult(deid.restore(text), self.name, self._usage(resp), "measured", {"model": self.model})

    def revise(self, ctx: Dict[str, Any], prior: str, violations: List[Dict[str, str]]) -> DraftResult:
        deid = Deidentifier(ctx["claim"])
        fix = "\n".join(f"- {v['type']}: '{v['span']}' - {v['detail']}" for v in violations)
        resp = self._call([
            {"role": "user", "content": build_dynamic_prompt(ctx, deid)},
            {"role": "assistant", "content": deid.scrub(prior)},
            {"role": "user", "content": f"The verifier rejected this draft:\n{fix}\n"
                                        "Return the full corrected letter. Remove unsupported content; add no new facts."},
        ])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return DraftResult(deid.restore(text), self.name, self._usage(resp), "measured",
                           {"model": self.model, "mode": "revise"})


def get_default_drafter() -> Any:
    """Live Claude if configured, otherwise the deterministic offline drafter."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicDrafter()
        except ImportError:
            pass
    return TemplateDrafter()
