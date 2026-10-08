"""
agents/retrieval.py - Policy retrieval (BM25) and verbatim evidence extraction.

Design choices:
- BM25 is used instead of dense embeddings because the corpus is small, highly
  technical (CDT codes, "ferrule", "4mm") and lexical matching is more predictable
  and auditable here. The retriever interface is swappable for pgvector/Bedrock KB.
- Evidence extraction NEVER paraphrases. Every quote is a verbatim sentence from the
  chart, or a structured chart field with its value. The grounding verifier enforces this.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from integrations.mock_apis import ClinicalChart
from knowledge.rcm_reference import CRITERIA, POLICY_CORPUS, Criterion, PolicyChunk

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it",
    "of", "on", "or", "such", "that", "the", "to", "was", "were", "when", "with", "this", "its",
}
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def split_sentences(text: str) -> List[str]:
    """Splits on sentence punctuation followed by a capital letter or '#'. Keeps '2.5mm' intact."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z#])", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ==========================================
# BM25 policy retriever
# ==========================================

class BM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.n = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def score(self, query: List[str], idx: int) -> float:
        tf, dl = self.tf[idx], len(self.docs[idx])
        s = 0.0
        for q in query:
            if q not in tf:
                continue
            f = tf[q]
            s += self.idf.get(q, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


@dataclass
class RetrievedChunk:
    id: str
    cdt: str
    topic: str
    text: str
    score: float
    provenance: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class PolicyRetriever:
    def __init__(self, corpus: Optional[List[PolicyChunk]] = None):
        self.corpus = corpus or POLICY_CORPUS
        self._bm25 = BM25([tokenize(f"{c.cdt} {c.topic} {c.text}") for c in self.corpus])

    def search(self, query: str, cdt: Optional[str] = None, payer_id: Optional[str] = None,
               k: int = 3) -> List[RetrievedChunk]:
        q = tokenize(query)
        candidates = [
            i for i, c in enumerate(self.corpus)
            if (cdt is None or c.cdt == cdt) and (c.payer_id in ("*", payer_id))
        ]
        if not candidates:                       # unknown CDT -> fall back to whole corpus
            candidates = list(range(len(self.corpus)))
        scored = sorted(((self._bm25.score(q, i), i) for i in candidates), reverse=True)
        out = []
        for s, i in scored[:k]:
            if s <= 0:
                continue
            c = self.corpus[i]
            out.append(RetrievedChunk(c.id, c.cdt, c.topic, c.text, round(s, 3), c.provenance))
        return out


# ==========================================
# Evidence extraction
# ==========================================

@dataclass
class CriterionResult:
    id: str
    text: str
    required: bool
    status: str                              # "MET" | "UNMET" | "CONTRADICTED"
    quotes: List[str] = field(default_factory=list)       # verbatim chart sentences
    structured: List[str] = field(default_factory=list)   # "field = value" facts

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _teeth_with_pockets_at_least(probing: str, min_mm: int) -> List[str]:
    """Parses '#2 6-4-6, #3 7-5-6' style strings -> teeth with any site >= min_mm."""
    teeth = []
    for tooth, depths in re.findall(r"#(\d{1,2})\s+([\d\-]+)", probing or ""):
        if any(int(d) >= min_mm for d in depths.split("-") if d.isdigit()):
            teeth.append(tooth)
    return teeth


def _sentence_mm_values(sentence: str) -> List[float]:
    vals = []
    for a, b in re.findall(r"(\d+(?:\.\d+)?)\s*mm(?:\s*to\s*(\d+(?:\.\d+)?)\s*mm)?", sentence):
        vals.append(float(a))
        if b:
            vals.append(float(b))
    return vals


# Criterion-specific checks that need more than keyword cues.
def _check_pockets(c: Criterion, chart: ClinicalChart, sentences: List[str], res: CriterionResult):
    teeth = _teeth_with_pockets_at_least(chart.probing_depths or "", 4)
    if len(teeth) >= 4:
        res.structured.append(f"Periodontal chart: {len(teeth)} teeth with probing depth >= 4mm "
                              f"(#{', #'.join(teeth)})")
    res.quotes = [s for s in sentences
                  if any(k in s.lower() for k in c.keywords) and any(v >= 4 for v in _sentence_mm_values(s))][:1]
    res.status = "MET" if (res.structured or res.quotes) else "UNMET"


SPECIAL_CHECKS: Dict[str, Callable[[Criterion, ClinicalChart, List[str], CriterionResult], None]] = {
    "D4341-POCKETS": _check_pockets,
}


class EvidenceExtractor:
    def extract(self, cdt: str, chart: Optional[ClinicalChart]) -> Dict[str, Any]:
        criteria = CRITERIA.get(cdt, [])
        if not criteria:
            return {"cdt": cdt, "criteria": [], "required_met": False,
                    "gaps": [f"No coverage criteria configured for {cdt}; route to manual review."]}
        if chart is None:
            return {"cdt": cdt, "criteria": [CriterionResult(c.id, c.text, c.required, "UNMET").to_dict()
                                             for c in criteria],
                    "required_met": False, "gaps": ["No clinical chart found for this claim."]}

        sentences = split_sentences(chart.clinical_notes)
        results: List[CriterionResult] = []
        for c in criteria:
            res = CriterionResult(c.id, c.text, c.required, "UNMET")
            if c.id in SPECIAL_CHECKS:
                SPECIAL_CHECKS[c.id](c, chart, sentences, res)
                results.append(res)
                continue

            contradicting = [s for s in sentences if any(k in s.lower() for k in c.contradicting_keywords)]
            if contradicting:
                res.status, res.quotes = "CONTRADICTED", contradicting[:1]
                results.append(res)
                continue

            ranked = sorted(
                ((sum(k in s.lower() for k in c.keywords), s) for s in sentences),
                key=lambda t: t[0], reverse=True,
            )
            res.quotes = [s for score, s in ranked[:2] if score > 0]

            if c.structured_field:
                val = getattr(chart, c.structured_field, None)
                if val is not None and c.structured_min is not None and val >= c.structured_min:
                    res.structured.append(f"Chart field {c.structured_field} = {val:g}")
            if c.id.endswith("IMAGING"):
                if chart.radiograph_attached:
                    res.structured.append("Radiograph attached to claim record")
                if chart.intraoral_photos_attached:
                    res.structured.append("Intraoral photo attached to claim record")

            res.status = "MET" if (res.quotes or res.structured) else "UNMET"
            results.append(res)

        gaps = [f"{r.text} ({r.status.lower()})" for r in results if r.required and r.status != "MET"]
        return {
            "cdt": cdt,
            "criteria": [r.to_dict() for r in results],
            "required_met": not gaps,
            "gaps": gaps,
        }
