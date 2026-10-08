"""
eval/grounding.py - Deterministic grounding verifier for payer-facing letters.

Runs on every draft (inside the graph) and again on the human-edited text before
writeback. Unlike an LLM-as-judge score, every failure points to an exact span.

Atomic claims checked:
  - quoted text      -> must be verbatim in the chart note
  - measurements     -> every "Nmm" / "N%" value must exist in the source record
  - tooth numbers    -> must be on the claim or in the chart
  - dollar amounts   -> must be a contracted allowed amount; billed fee is a violation
Policy rules:
  - no Prompt Pay statute citations in appeals (they govern clean-claim payment timing)
  - ERISA claims regulation only cited for ERISA employer group plans
  - no policy "Section N" citations that do not exist in the retrieved corpus
  - no unresolved de-identification placeholders

faithfulness = supported atomic claims / total atomic claims (claim-level, RAGAS-style
definition, computed deterministically rather than by an LLM judge).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set

from integrations.mock_apis import ClinicalChart, DentalClaim

_WS = re.compile(r"\s+")
_QUOTE = re.compile(r"[\"“]([^\"”]{12,}?)[\"”]")
_MEASURE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|%)")
_TOOTH = re.compile(r"(?:#|\btooth\s+#?)(\d{1,2})\b", re.I)
_DOLLAR = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")
_SECTION = re.compile(r"\b(?:section|rule)\s+\d+[a-z]?(?:\.\d+)*\b", re.I)
_PLACEHOLDER = re.compile(r"\[(?:PATIENT|PATIENT_ID|MEMBER)[^\]]*\]")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def _numbers(texts: Iterable[str]) -> Set[float]:
    out: Set[float] = set()
    for t in texts:
        for n in re.findall(r"\d+(?:\.\d+)?", t or ""):
            out.add(float(n))
    return out


def _money(x: float) -> str:
    return f"{x:,.2f}"


class GroundingVerifier:
    def verify(
        self,
        letter: str,
        claim: DentalClaim,
        chart: Optional[ClinicalChart],
        evidence: Optional[Dict[str, Any]] = None,
        policy_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        evidence = evidence or {"criteria": []}
        policy_chunks = policy_chunks or []
        notes = chart.clinical_notes if chart else ""
        notes_n = _norm(notes)

        structured = [s for c in evidence.get("criteria", []) for s in c.get("structured", [])]
        criteria_text = [c.get("text", "") for c in evidence.get("criteria", [])]
        policy_text = [c.get("text", "") for c in policy_chunks]
        source_numbers = _numbers([notes, chart.probing_depths if chart else "", *structured,
                                   *criteria_text, *policy_text])
        valid_teeth = {p.tooth_num for p in claim.procs if p.tooth_num} | set(re.findall(r"#(\d{1,2})", notes))

        violations: List[Dict[str, str]] = []
        supported = total = 0

        # 1. Verbatim quotes
        for q in _QUOTE.findall(letter):
            total += 1
            if _norm(q).rstrip(".") in notes_n:
                supported += 1
            else:
                violations.append({"type": "FABRICATED_QUOTE", "span": q[:120],
                                   "detail": "Quoted text is not verbatim in the clinical record."})

        # 2. Measurements
        for val, unit in _MEASURE.findall(letter):
            total += 1
            if float(val) in source_numbers:
                supported += 1
            else:
                violations.append({"type": "UNSUPPORTED_MEASUREMENT", "span": f"{val}{unit}",
                                   "detail": "Value not present in chart, structured fields or policy text."})

        # 3. Tooth numbers
        for t in _TOOTH.findall(letter):
            total += 1
            if t in valid_teeth:
                supported += 1
            else:
                violations.append({"type": "UNSUPPORTED_TOOTH", "span": f"#{t}",
                                   "detail": "Tooth number not on the claim or in the chart."})

        # 4. Dollar amounts
        allowed_ok = {_money(claim.allowed_at_issue)} | {_money(p.expected_allowed) for p in claim.denied_procs}
        billed = {_money(claim.billed_fee)} | {_money(p.fee_billed) for p in claim.denied_procs}
        for amt in _DOLLAR.findall(letter):
            total += 1
            a = _money(float(amt.replace(",", "")))
            if a in allowed_ok:
                supported += 1
            elif a in billed:
                violations.append({"type": "BILLED_FEE_DEMAND", "span": f"${a}",
                                   "detail": "In-network recovery is the contracted allowed amount, not the billed fee."})
            else:
                violations.append({"type": "UNSUPPORTED_AMOUNT", "span": f"${a}",
                                   "detail": "Amount does not match any contracted allowed amount on denied lines."})

        # 5. Regulatory language
        low = letter.lower()
        if "prompt pay" in low:
            violations.append({"type": "MISAPPLIED_STATUTE", "span": "prompt pay",
                               "detail": "Prompt Pay laws govern clean-claim payment timing, not appeal decisions, "
                                         "and do not apply to self-funded ERISA plans."})
        if "2560.503-1" in low and claim.plan_type != "erisa_employer_group":
            violations.append({"type": "MISAPPLIED_STATUTE", "span": "29 C.F.R. 2560.503-1",
                               "detail": "ERISA claims regulation cited for a non-ERISA plan."})

        # 6. Policy citations that do not exist
        corpus_low = " ".join(policy_text).lower()
        for sec in _SECTION.findall(letter):
            if sec.lower() not in corpus_low:
                violations.append({"type": "UNVERIFIABLE_CITATION", "span": sec,
                                   "detail": "Policy section not found in retrieved payer policy text."})

        # 7. De-identification placeholders left behind
        for ph in _PLACEHOLDER.findall(letter):
            violations.append({"type": "UNRESOLVED_PLACEHOLDER", "span": ph, "detail": "Re-identification failed."})

        # Citation coverage: each MET required criterion should be evidenced in the letter.
        met = [c for c in evidence.get("criteria", []) if c.get("required") and c.get("status") == "MET"]
        cited = 0
        for c in met:
            snippets = c.get("quotes", []) + c.get("structured", [])
            if any(_norm(s)[:60] in _norm(letter) for s in snippets):
                cited += 1
        coverage = (cited / len(met)) if met else 1.0

        faithfulness = (supported / total) if total else 1.0
        return {
            "passed": not violations,
            "faithfulness": round(faithfulness, 3),
            "citation_coverage": round(coverage, 3),
            "claims_checked": total,
            "claims_supported": supported,
            "violations": violations,
        }
