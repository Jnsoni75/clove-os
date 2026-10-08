"""
knowledge/rcm_reference.py - Dental RCM reference data (single source of truth).

Everything the agent "knows" about denials lives here so it can be reviewed by an
RCM lead in one place.

Provenance rules (important for a payer-facing system):
- CARC descriptions are the X12 Claim Adjustment Reason Code meanings (abridged).
- CDT descriptors are abridged ADA CDT nomenclature.
- POLICY_CORPUS entries are ILLUSTRATIVE composites of commonly published payer
  criteria. They are NOT verbatim payer policies and carry no section numbers.
  In production each chunk is replaced by the payer's published clinical policy
  (with its real document ID) and re-indexed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# X12 Claim Adjustment Group + Reason Codes
# ---------------------------------------------------------------------------

GROUP_CODES: Dict[str, str] = {
    "CO": "Contractual Obligation - provider liability; patient may not be billed",
    "PR": "Patient Responsibility",
    "OA": "Other Adjustment",
    "PI": "Payer Initiated Reduction",
}

CARC: Dict[str, str] = {
    "16": "Claim/service lacks information or has submission/billing error(s).",
    "45": "Charge exceeds fee schedule/maximum allowable or contracted/legislated fee arrangement.",
    "50": "These are non-covered services because this is not deemed a 'medical necessity' by the payer.",
    "97": "The benefit for this service is included in the payment/allowance for another "
          "service/procedure that has already been adjudicated.",
    "119": "Benefit maximum for this time period or occurrence has been reached.",
    "204": "This service/equipment/drug is not covered under the patient's current benefit plan.",
}


def parse_adjustment_code(code: str) -> Tuple[str, str]:
    """'CO-97' -> ('CO', '97'). Raises ValueError on malformed codes."""
    try:
        group, reason = code.strip().upper().split("-", 1)
    except ValueError as exc:
        raise ValueError(f"Malformed adjustment code: {code!r}") from exc
    if group not in GROUP_CODES:
        raise ValueError(f"Unknown claim adjustment group code: {group!r}")
    return group, reason


# ---------------------------------------------------------------------------
# Denial playbook: what a senior biller does with each CARC
# ---------------------------------------------------------------------------

class Route:
    APPEAL = "APPEAL"                       # clinical / unbundling appeal letter
    RESUBMIT = "RESUBMIT_CORRECTED"         # fix + resubmit; do NOT appeal
    REP_CALL = "REP_CALL"                   # systemic zero-pay pattern: call payer rep first
    DOC_GAP = "DOCUMENTATION_GAP"           # chart can't support appeal: request provider addendum
    NO_APPEAL = "NO_APPEAL"                 # frequency / plan exclusion / contractual


# Mirrors the production Clove OS rule: ">=70% of lines paid $0 -> rep call first".
ZERO_PAY_REP_CALL_THRESHOLD_PCT = 70.0


@dataclass(frozen=True)
class PlaybookEntry:
    route: str
    appeal_type: Optional[str]
    rationale: str
    actions: Tuple[str, ...] = ()


PLAYBOOK: Dict[str, PlaybookEntry] = {
    "16": PlaybookEntry(
        Route.RESUBMIT, None,
        "CARC 16 is a submission/information defect. It is corrected and resubmitted, not appealed.",
        (
            "Read the RARC on the 835/EOB to identify the exact missing element.",
            "Attach the missing documentation (see checklist) to a corrected claim.",
            "Resubmit as a corrected claim (not a new claim) to preserve timely filing.",
        ),
    ),
    "50": PlaybookEntry(
        Route.APPEAL, "Medical necessity",
        "CARC 50 disputes necessity. Appeal only if the chart documents the payer's criteria.",
    ),
    "97": PlaybookEntry(
        Route.APPEAL, "Unbundling / separately reportable procedure",
        "CARC 97 claims inclusion in another paid service. Appeal if the procedure is separately "
        "reportable under CDT and was clinically distinct.",
    ),
    "119": PlaybookEntry(
        Route.NO_APPEAL, None,
        "Frequency/benefit maximum reached. Appeal only if the payer's service history is wrong.",
        (
            "Verify service history in Open Dental against the payer's frequency record.",
            "If history is correct: move balance per group code (PR -> patient statement).",
            "If history is wrong: call payer to correct history, then request reprocessing.",
        ),
    ),
    "204": PlaybookEntry(
        Route.NO_APPEAL, None,
        "Plan exclusion. Clinical appeals rarely overturn exclusions.",
        ("Confirm exclusion in plan benefits.", "Move to patient responsibility if group code is PR."),
    ),
    "45": PlaybookEntry(
        Route.NO_APPEAL, None,
        "Contractual write-off. If allowed is below the contracted fee schedule, route to the "
        "Underpayments module instead of the denial queue.",
        ("Compare allowed vs contracted fee schedule.", "If short: send to Underpayments dispute workflow."),
    ),
}


# ---------------------------------------------------------------------------
# CDT descriptors (abridged ADA nomenclature)
# ---------------------------------------------------------------------------

CDT: Dict[str, str] = {
    "D1110": "Prophylaxis - adult",
    "D2740": "Crown - porcelain/ceramic",
    "D2950": "Core buildup, including any pins when required",
    "D4341": "Periodontal scaling and root planing - four or more teeth per quadrant",
    "D4910": "Periodontal maintenance",
    "D7210": "Extraction, erupted tooth requiring removal of bone and/or sectioning of tooth",
    "D7953": "Bone replacement graft for ridge preservation - per site",
}


# ---------------------------------------------------------------------------
# Coverage criteria used for evidence extraction (per CDT)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Criterion:
    id: str
    text: str
    keywords: Tuple[str, ...]              # sentence-level evidence cues
    required: bool = True
    contradicting_keywords: Tuple[str, ...] = ()   # cues that undercut the criterion
    structured_field: Optional[str] = None         # ClinicalChart attribute that can satisfy it
    structured_min: Optional[float] = None


CRITERIA: Dict[str, List[Criterion]] = {
    "D2950": [
        Criterion(
            "D2950-STRUCTURE",
            "Insufficient remaining coronal tooth structure to retain the planned crown",
            ("coronal", "tooth structure", "clinical crown", "cusp", "fracture", "less than"),
            structured_field="decay_percentage", structured_min=50.0,
        ),
        Criterion(
            "D2950-RETENTION",
            "Buildup placed to provide retention/resistance form - not to fill undercuts",
            ("retention", "resistance form", "ferrule"),
            contradicting_keywords=("undercut", "block out", "blockout", "box form"),
        ),
        Criterion(
            "D2950-IMAGING",
            "Supporting radiograph and/or intraoral photograph available",
            ("radiograph", "photo", "x-ray", "bitewing", "periapical"),
            required=False,
        ),
    ],
    "D4341": [
        Criterion(
            "D4341-POCKETS",
            "Probing depths of 4mm or greater on four or more teeth in the quadrant",
            ("pocket depth", "probing"),
        ),
        Criterion(
            "D4341-BONELOSS",
            "Radiographic evidence of bone loss",
            ("bone loss",),
            structured_field="bone_loss_percentage", structured_min=1.0,
        ),
        Criterion("D4341-DX", "Periodontitis diagnosis documented", ("periodontitis",)),
        Criterion(
            "D4341-CALCULUS", "Subgingival calculus and/or bleeding on probing",
            ("calculus", "bleeding"), required=False,
        ),
    ],
    "D7953": [
        Criterion(
            "D7953-PURPOSE",
            "Graft placed to preserve the ridge for future implant placement",
            ("ridge preservation", "implant", "allograft", "graft"),
        ),
        Criterion(
            "D7953-DISTINCT",
            "Documented as a distinct procedure, not incidental to the extraction",
            ("independent", "not incidental", "separate", "distinct"),
        ),
        Criterion(
            "D7953-DEFECT", "Ridge/socket defect documented with imaging",
            ("defect", "cbct", "buccal plate", "deficiency"), required=False,
        ),
    ],
}


# ---------------------------------------------------------------------------
# Policy corpus for retrieval (ILLUSTRATIVE composites - see module docstring)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyChunk:
    id: str
    cdt: str
    topic: str
    text: str
    payer_id: str = "*"      # "*" = applies to any payer
    provenance: str = "ILLUSTRATIVE composite - replace with payer's published policy"


POLICY_CORPUS: List[PolicyChunk] = [
    PolicyChunk(
        "GEN-D2950-01", "D2950", "buildup necessity",
        "A core buildup is benefited when there is insufficient remaining coronal tooth structure to "
        "retain an indirect restoration such as a crown. Documentation should describe the extent of "
        "coronal loss after caries excavation or fracture.",
    ),
    PolicyChunk(
        "GEN-D2950-02", "D2950", "buildup vs filler",
        "A buildup is not benefited when it is used only as a filler to eliminate undercuts, box forms "
        "or concave irregularities in a crown preparation. The record should state that the buildup "
        "provides retention and resistance form.",
    ),
    PolicyChunk(
        "GEN-D2950-03", "D2950", "buildup bundling same day crown",
        "When reported on the same date as a crown on the same tooth, a core buildup is separately "
        "reportable under CDT when it is required for retention. Payers may request a narrative and "
        "pre-operative radiograph or photograph to establish that it was not a preparation filler.",
    ),
    PolicyChunk(
        "GEN-D4341-01", "D4341", "scaling root planing criteria",
        "Periodontal scaling and root planing, four or more teeth per quadrant, requires periodontal "
        "charting showing probing depths of 4mm or greater on at least four teeth in the quadrant, "
        "with radiographic evidence of bone loss.",
    ),
    PolicyChunk(
        "GEN-D4341-02", "D4341", "scaling root planing documentation submission",
        "Claims for scaling and root planing should be submitted with a dated periodontal chart and "
        "current radiographs. Missing charting results in a request for information rather than a "
        "necessity determination.",
    ),
    PolicyChunk(
        "GEN-D7953-01", "D7953", "ridge preservation graft",
        "A bone replacement graft for ridge preservation is placed in an extraction site to preserve "
        "ridge dimensions for a future implant. Many plans exclude it; where covered, documentation "
        "should establish a ridge or socket defect and the planned implant.",
    ),
    PolicyChunk(
        "GEN-D7953-02", "D7953", "ridge preservation bundling extraction",
        "When reported with an extraction on the same date, a ridge preservation graft is a distinct "
        "procedure from the extraction itself. The operative note should describe graft material and "
        "the regenerative purpose separately from the removal of the tooth.",
    ),
    PolicyChunk(
        "GEN-D4910-01", "D4910", "periodontal maintenance frequency",
        "Periodontal maintenance is benefited following active periodontal therapy, typically up to "
        "a plan-defined frequency per benefit year.",
    ),
    PolicyChunk(
        "GEN-D1110-01", "D1110", "prophylaxis frequency",
        "Adult prophylaxis is subject to plan frequency limits, commonly two per benefit period.",
    ),
]


# Documentation checklists used by the RESUBMIT and DOC_GAP routes.
RESUBMISSION_CHECKLIST: Dict[str, List[str]] = {
    "D4341": [
        "Dated full-mouth periodontal chart (6-point probing) from within the last 12 months",
        "Current radiographs showing bone levels for the treated quadrants",
        "Brief narrative stating diagnosis and teeth treated per quadrant",
    ],
    "D2950": [
        "Pre-operative radiograph and/or intraoral photo after caries removal",
        "Narrative stating remaining tooth structure and need for retention",
    ],
    "D7953": [
        "Operative note describing graft material and ridge defect",
        "Pre/post-extraction imaging",
    ],
}
