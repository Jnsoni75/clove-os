"""
integrations/mock_apis.py - Simulated connectors for Clove OS (demo data, no real PHI).

1. OpenDentalClient - claims (line-level adjudication), clinical charts, payer denial stats,
   claim-tracking writeback. Production: Open Dental API (eConnector/API Service).
2. DeputyClient     - weekly timesheets + next-day demand. Alerts are COMPUTED from data
   (FLSA weekly >40h for non-exempt staff; RDA:DDS and RDH capacity ratios).
3. ZohoClient       - Corp Dev deal pipeline. Enrichment uses an explicit sample market table
   (production sources noted per field) and a transparent, weighted scoring formula.

All names, NPIs and IDs are synthetic.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from knowledge.rcm_reference import CARC, parse_adjustment_code


# ==========================================
# Open Dental entities
# ==========================================

class ClaimProc(BaseModel):
    proc_num: int
    proc_code: str                      # CDT code
    proc_desc: str
    tooth_num: Optional[str] = None
    surf: Optional[str] = None
    fee_billed: float                   # office UCR fee
    expected_allowed: float             # participating fee schedule amount
    paid_amount: float = 0.0
    adjustment_code: Optional[str] = None   # e.g. "CO-97" (None = paid as expected)
    payer_remark: str = ""

    @property
    def is_denied(self) -> bool:
        return self.adjustment_code is not None and self.paid_amount == 0.0


class ClinicalChart(BaseModel):
    claim_id: int
    patient_id: int
    patient_name: str
    chart_date: str
    provider_name: str
    provider_npi: str
    clinical_notes: str
    decay_percentage: Optional[float] = None
    probing_depths: Optional[str] = None
    mobility_score: Optional[str] = None
    bone_loss_percentage: Optional[float] = None
    radiograph_attached: bool = False
    intraoral_photos_attached: bool = False


class DentalClaim(BaseModel):
    claim_id: int
    patient_id: int
    patient_name: str
    clinic_id: int
    clinic_name: str
    payer_id: str
    payer_name: str
    plan_type: str = "unknown"          # "erisa_employer_group" | "non_erisa" | "unknown"
    plan_funding: str = "unknown"       # "fully_insured" | "self_funded" | "unknown"
    date_of_service: str
    status: str                         # "Denied", "Appeal Pending Review", "Appealed", ...
    procs: List[ClaimProc] = Field(default_factory=list)
    tracking_notes: List[str] = Field(default_factory=list)
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # ---- derived views -------------------------------------------------
    @property
    def denied_procs(self) -> List[ClaimProc]:
        return [p for p in self.procs if p.is_denied]

    @property
    def primary_denied_proc(self) -> Optional[ClaimProc]:
        denied = self.denied_procs
        return denied[0] if denied else None

    @property
    def denial_code(self) -> str:
        p = self.primary_denied_proc
        return p.adjustment_code if p and p.adjustment_code else ""

    @property
    def carc(self) -> str:
        return parse_adjustment_code(self.denial_code)[1] if self.denial_code else ""

    @property
    def denial_description(self) -> str:
        return CARC.get(self.carc, "Unknown adjustment reason")

    @property
    def billed_fee(self) -> float:
        return round(sum(p.fee_billed for p in self.procs), 2)

    @property
    def allowed_at_issue(self) -> float:
        """Contracted allowed amount on denied lines - the recoverable value, NOT billed fee."""
        return round(sum(p.expected_allowed for p in self.denied_procs), 2)


# ==========================================
# Open Dental client (simulated)
# ==========================================

SRP_DESC = "Periodontal scaling and root planing - four or more teeth per quadrant"


def _proc(num, code, desc, tooth, surf, billed, allowed, paid=0.0, adj=None, remark=""):
    return ClaimProc(
        proc_num=num, proc_code=code, proc_desc=desc, tooth_num=tooth, surf=surf,
        fee_billed=billed, expected_allowed=allowed, paid_amount=paid,
        adjustment_code=adj, payer_remark=remark,
    )


class OpenDentalClient:
    """Simulated Open Dental API. Each instance owns an isolated copy of the seed data."""

    TRACKING_STATUS_BY_ROUTE = {
        "APPEAL": "Appeal Pending Submission",
        "RESUBMIT_CORRECTED": "Corrected Claim Pending",
        "REP_CALL": "Payer Rep Call Queued",
        "DOCUMENTATION_GAP": "Provider Addendum Requested",
        "NO_APPEAL": "Closed - No Appeal",
    }

    def __init__(self):
        self._claims_db: Dict[int, DentalClaim] = {}
        self._charts_db: Dict[int, ClinicalChart] = {}
        self._payer_stats: Dict[str, Dict[str, Any]] = {}
        self._seed_data()

    # ------------------------------------------------------------------
    def _seed_data(self):
        seeds: List[Dict[str, Any]] = [
            # 1. Classic necessity denial, chart fully supports -> APPEAL
            dict(
                claim=DentalClaim(
                    claim_id=90412, patient_id=10482, patient_name="Eleanor Vance",
                    clinic_id=101, clinic_name="Clove Dental - Austin Downtown",
                    payer_id="DELTA_TX", payer_name="Delta Dental of Texas",
                    plan_type="erisa_employer_group", plan_funding="fully_insured",
                    date_of_service="2026-09-18", status="Denied",
                    procs=[
                        _proc(501, "D2950", "Core buildup, including any pins when required", "19", None,
                              385.00, 212.00, adj="CO-50",
                              remark="Documentation does not support necessity of buildup."),
                    ],
                ),
                chart=ClinicalChart(
                    claim_id=90412, patient_id=10482, patient_name="Eleanor Vance",
                    chart_date="2026-09-18", provider_name="Dr. Marcus Hale, DDS", provider_npi="1000000011",
                    clinical_notes=(
                        "Tooth #19 presented with fractured disto-lingual cusp extending 2.5mm subgingivally. "
                        "Pre-existing amalgam restoration failed with recurrent caries. "
                        "After excavation, approximately 65% of the clinical crown was lost. "
                        "Core buildup placed with dual-cure composite to provide retention and resistance form "
                        "and establish ferrule prior to crown preparation. "
                        "Post-excavation intraoral photo and periapical radiograph taken."
                    ),
                    decay_percentage=65.0, bone_loss_percentage=10.0,
                    radiograph_attached=True, intraoral_photos_attached=True,
                ),
            ),
            # 2. CO-16 (missing info) -> RESUBMIT, never appeal
            dict(
                claim=DentalClaim(
                    claim_id=90415, patient_id=11209, patient_name="Marcus Thorne",
                    clinic_id=104, clinic_name="Clove Dental - Houston Galleria",
                    payer_id="METLIFE", payer_name="MetLife Dental",
                    plan_type="erisa_employer_group", plan_funding="self_funded",
                    date_of_service="2026-09-21", status="Denied",
                    procs=[
                        _proc(502, "D4341", SRP_DESC, "UR", None, 310.00, 178.00, adj="CO-16",
                              remark="Periodontal charting not received."),
                        _proc(503, "D4341", SRP_DESC, "UL", None, 310.00, 178.00, adj="CO-16",
                              remark="Periodontal charting not received."),
                    ],
                ),
                chart=ClinicalChart(
                    claim_id=90415, patient_id=11209, patient_name="Marcus Thorne",
                    chart_date="2026-09-21", provider_name="Dr. Sophia Chen, DMD", provider_npi="1000000029",
                    clinical_notes=(
                        "Generalized Stage III, Grade B periodontitis. "
                        "Heavy subgingival calculus with bleeding on probing at 78% of sites. "
                        "Pocket depths of 5mm to 7mm on teeth #2, #3, #4, #5, #12, #13, #14 and #15. "
                        "Radiographs show 25-35% horizontal bone loss."
                    ),
                    probing_depths="#2 6-4-6, #3 7-5-6, #4 5-4-5, #5 5-4-5, #12 5-4-5, #13 6-5-5, #14 7-5-7, #15 6-5-6",
                    bone_loss_percentage=30.0, radiograph_attached=True,
                ),
            ),
            # 3. Buildup bundled into same-day crown (CO-97) -> unbundling APPEAL
            dict(
                claim=DentalClaim(
                    claim_id=90422, patient_id=12844, patient_name="Julian Delgado",
                    clinic_id=102, clinic_name="Clove Dental - Dallas Metro",
                    payer_id="CIGNA", payer_name="Cigna Dental",
                    plan_type="erisa_employer_group", plan_funding="self_funded",
                    date_of_service="2026-09-24", status="Denied",
                    procs=[
                        _proc(504, "D2740", "Crown - porcelain/ceramic", "30", None,
                              1150.00, 742.00, paid=371.00),
                        _proc(505, "D2950", "Core buildup, including any pins when required", "30", None,
                              385.00, 205.00, adj="CO-97",
                              remark="Buildup is included in the allowance for the crown."),
                    ],
                ),
                chart=ClinicalChart(
                    claim_id=90422, patient_id=12844, patient_name="Julian Delgado",
                    chart_date="2026-09-24", provider_name="Dr. Sarah Jenkins, DDS", provider_npi="1000000037",
                    clinical_notes=(
                        "Tooth #30 had root canal therapy 6 months ago at an outside endodontic office. "
                        "Patient presented with lost temporary and fracture of the mesio-lingual cusp. "
                        "Less than 2mm of sound tooth structure remains circumferentially. "
                        "Core buildup placed to provide retention and resistance form for the crown. "
                        "Monolithic zirconia crown prepared; final impression taken. "
                        "Pre-operative periapical radiograph on file."
                    ),
                    decay_percentage=55.0, radiograph_attached=True,
                ),
            ),
            # 4. Ridge preservation bundled into extraction (CO-97), non-ERISA plan -> APPEAL
            dict(
                claim=DentalClaim(
                    claim_id=90428, patient_id=13410, patient_name="Amina Rahman",
                    clinic_id=108, clinic_name="Clove Dental - San Antonio North",
                    payer_id="GUARDIAN", payer_name="Guardian Dental",
                    plan_type="non_erisa", plan_funding="fully_insured",
                    date_of_service="2026-09-28", status="Denied",
                    procs=[
                        _proc(506, "D7210", "Extraction, erupted tooth requiring removal of bone and/or sectioning",
                              "14", None, 420.00, 238.00, paid=190.40),
                        _proc(507, "D7953", "Bone replacement graft for ridge preservation - per site",
                              "14", None, 450.00, 260.00, adj="CO-97",
                              remark="Graft is inclusive to the extraction."),
                    ],
                ),
                chart=ClinicalChart(
                    claim_id=90428, patient_id=13410, patient_name="Amina Rahman",
                    chart_date="2026-09-28", provider_name="Dr. Aaron Patel, DDS", provider_npi="1000000045",
                    clinical_notes=(
                        "Vertical root fracture on tooth #14 required sectioning and atraumatic extraction. "
                        "Post-extraction buccal plate defect noted. "
                        "Freeze-dried bone allograft with resorbable collagen membrane placed for ridge preservation "
                        "ahead of a planned implant. "
                        "The graft was a separate regenerative procedure, not incidental to the extraction. "
                        "CBCT confirms crestal height deficiency."
                    ),
                    bone_loss_percentage=45.0, radiograph_attached=True,
                ),
            ),
            # 5. Chart contradicts the claim -> DOCUMENTATION_GAP (agent must refuse to appeal)
            dict(
                claim=DentalClaim(
                    claim_id=90431, patient_id=13977, patient_name="Grace Okafor",
                    clinic_id=101, clinic_name="Clove Dental - Austin Downtown",
                    payer_id="DELTA_TX", payer_name="Delta Dental of Texas",
                    plan_type="erisa_employer_group", plan_funding="fully_insured",
                    date_of_service="2026-09-30", status="Denied",
                    procs=[
                        _proc(508, "D2950", "Core buildup, including any pins when required", "3", None,
                              385.00, 212.00, adj="CO-50",
                              remark="Documentation does not support necessity of buildup."),
                    ],
                ),
                chart=ClinicalChart(
                    claim_id=90431, patient_id=13977, patient_name="Grace Okafor",
                    chart_date="2026-09-30", provider_name="Dr. Marcus Hale, DDS", provider_npi="1000000011",
                    clinical_notes=(
                        "Tooth #3 crown preparation completed. "
                        "Composite placed to block out undercut on the mesial wall. "
                        "Final impression taken."
                    ),
                    radiograph_attached=False,
                ),
            ),
            # 6. Payer with systemic zero-pay pattern -> REP_CALL first
            dict(
                claim=DentalClaim(
                    claim_id=90437, patient_id=14102, patient_name="Daniel Kim",
                    clinic_id=104, clinic_name="Clove Dental - Houston Galleria",
                    payer_id="AETNA", payer_name="Aetna Dental",
                    plan_type="erisa_employer_group", plan_funding="fully_insured",
                    date_of_service="2026-10-01", status="Denied",
                    procs=[
                        _proc(509, "D4910", "Periodontal maintenance", None, None,
                              185.00, 112.00, adj="CO-50",
                              remark="Service not medically necessary."),
                    ],
                ),
                chart=ClinicalChart(
                    claim_id=90437, patient_id=14102, patient_name="Daniel Kim",
                    chart_date="2026-10-01", provider_name="Dr. Sophia Chen, DMD", provider_npi="1000000029",
                    clinical_notes="Periodontal maintenance 3 months after SRP. Localized 4mm pockets, stable.",
                ),
            ),
            # 7. Frequency limit (PR-119) -> NO_APPEAL
            dict(
                claim=DentalClaim(
                    claim_id=90440, patient_id=14388, patient_name="Priya Natarajan",
                    clinic_id=102, clinic_name="Clove Dental - Dallas Metro",
                    payer_id="DELTA_TX", payer_name="Delta Dental of Texas",
                    plan_type="erisa_employer_group", plan_funding="fully_insured",
                    date_of_service="2026-10-02", status="Denied",
                    procs=[
                        _proc(510, "D1110", "Prophylaxis - adult", None, None, 125.00, 78.00, adj="PR-119",
                              remark="Frequency limitation: 2 per benefit period."),
                    ],
                ),
                chart=ClinicalChart(
                    claim_id=90440, patient_id=14388, patient_name="Priya Natarajan",
                    chart_date="2026-10-02", provider_name="Dr. Sarah Jenkins, DDS", provider_npi="1000000037",
                    clinical_notes="Routine adult prophylaxis.",
                ),
            ),
        ]
        for s in seeds:
            claim: DentalClaim = s["claim"]
            self._claims_db[claim.claim_id] = claim
            self._charts_db[claim.claim_id] = s["chart"]

        # Trailing-90-day line stats per payer (what production's zero-pay detector computes).
        self._payer_stats = {
            "DELTA_TX": {"lines": 1840, "zero_pay_lines": 196},
            "METLIFE":  {"lines": 1215, "zero_pay_lines": 141},
            "CIGNA":    {"lines": 960,  "zero_pay_lines": 88},
            "GUARDIAN": {"lines": 402,  "zero_pay_lines": 51},
            "AETNA":    {"lines": 57,   "zero_pay_lines": 47},   # 82% -> systemic issue
        }

    # ------------------------------------------------------------------
    def get_claims(self, status: Optional[str] = "Denied") -> List[DentalClaim]:
        claims = list(self._claims_db.values())
        if status is None:
            return claims
        return [c for c in claims if c.status.lower() == status.lower()]

    def get_claim(self, claim_id: int) -> Optional[DentalClaim]:
        return self._claims_db.get(claim_id)

    def get_claim_procs(self, claim_id: int) -> List[ClaimProc]:
        claim = self._claims_db.get(claim_id)
        return claim.procs if claim else []

    def get_clinical_chart(self, claim_id: int) -> Optional[ClinicalChart]:
        return self._charts_db.get(claim_id)

    def get_payer_denial_stats(self, payer_id: str) -> Dict[str, Any]:
        s = self._payer_stats.get(payer_id, {"lines": 0, "zero_pay_lines": 0})
        pct = (s["zero_pay_lines"] / s["lines"] * 100.0) if s["lines"] else 0.0
        return {"payer_id": payer_id, "window_days": 90, **s, "zero_pay_pct": round(pct, 1)}

    def post_claim_tracking(self, claim_id: int, route: str, note: str, reviewer: str) -> Dict[str, Any]:
        """Writes a human-approved tracking entry + status. Production: Open Dental ClaimTracking."""
        claim = self._claims_db.get(claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found in Open Dental.")
        if not reviewer or not reviewer.strip():
            raise ValueError("A named human reviewer is required for EHR writeback.")
        status = self.TRACKING_STATUS_BY_ROUTE.get(route, "Under Review")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{now}] {status} | approved by {reviewer.strip()} | {note[:140]}"
        claim.tracking_notes.append(entry)
        claim.status = status
        claim.last_updated = now
        return {"success": True, "claim_id": claim_id, "status": status, "timestamp": now, "audit_entry": entry}

    def snapshot(self) -> "OpenDentalClient":
        """Deep copy - lets eval runs mutate state without touching the UI's client."""
        return copy.deepcopy(self)


# ==========================================
# Deputy (workforce) - simulated
# ==========================================

class Employee(BaseModel):
    employee_id: int
    name: str
    clinic_id: int
    clinic_name: str
    role: str                     # "DDS", "RDH", "RDA", "Front Desk"
    flsa_exempt: bool             # doctors are typically exempt (salary/production)
    hourly_rate: float
    hours_week_to_date: float
    scheduled_remaining_hours: float

    @property
    def projected_week_hours(self) -> float:
        return round(self.hours_week_to_date + self.scheduled_remaining_hours, 2)


class ClinicDemand(BaseModel):
    clinic_id: int
    clinic_name: str
    date: str
    hygiene_appointments: int
    restorative_procedures: int
    rdh_scheduled: int
    rda_scheduled: int
    dds_scheduled: int


class StaffingAlert(BaseModel):
    clinic_id: int
    clinic_name: str
    alert_type: str               # "OVERTIME_RISK" | "RDH_CAPACITY" | "RDA_RATIO"
    severity: str                 # "HIGH" | "MODERATE"
    description: str
    action_required: str
    est_cost_impact_usd: float = 0.0


class DeputyClient:
    """Simulated Deputy API. Thresholds are explicit and configurable."""

    FLSA_WEEKLY_OT_HOURS = 40.0     # federal FLSA; TX has no daily OT (CA would differ)
    OT_PREMIUM = 0.5                # the extra half on top of straight time
    RDH_APPTS_PER_DAY = 8
    TARGET_RDA_PER_DDS = 2.0
    RESTORATIVE_PROCS_PER_DDS_DAY = 8   # above this, a DDS needs the 2nd assistant

    def __init__(self):
        e = lambda *a: Employee(**dict(zip(Employee.model_fields.keys(), a)))  # noqa: E731
        self._employees: List[Employee] = [
            e(201, "Dr. Marcus Hale", 101, "Clove Dental - Austin Downtown", "DDS", True, 0.0, 38.0, 9.0),
            e(202, "Rachel Evans", 101, "Clove Dental - Austin Downtown", "RDH", False, 52.0, 32.0, 8.0),
            e(203, "Tyler Ross", 101, "Clove Dental - Austin Downtown", "RDA", False, 28.0, 39.5, 9.0),
            e(204, "Dr. Sophia Chen", 104, "Clove Dental - Houston Galleria", "DDS", True, 0.0, 34.0, 8.0),
            e(205, "David Miller", 104, "Clove Dental - Houston Galleria", "RDH", False, 54.0, 30.0, 8.0),
            e(206, "Lena Ortiz", 104, "Clove Dental - Houston Galleria", "RDA", False, 27.0, 33.0, 8.0),
            e(207, "Dr. Aaron Patel", 108, "Clove Dental - San Antonio North", "DDS", True, 0.0, 36.0, 9.0),
            e(208, "Carla Gomez", 108, "Clove Dental - San Antonio North", "RDA", False, 29.0, 36.2, 8.0),
            e(209, "Ben Carter", 108, "Clove Dental - San Antonio North", "RDA", False, 26.0, 24.0, 8.0),
        ]
        self._demand: List[ClinicDemand] = [
            ClinicDemand(clinic_id=101, clinic_name="Clove Dental - Austin Downtown", date="tomorrow",
                         hygiene_appointments=11, restorative_procedures=7,
                         rdh_scheduled=1, rda_scheduled=1, dds_scheduled=1),
            ClinicDemand(clinic_id=104, clinic_name="Clove Dental - Houston Galleria", date="tomorrow",
                         hygiene_appointments=7, restorative_procedures=14,
                         rdh_scheduled=1, rda_scheduled=1, dds_scheduled=1),
            ClinicDemand(clinic_id=108, clinic_name="Clove Dental - San Antonio North", date="tomorrow",
                         hygiene_appointments=0, restorative_procedures=9,
                         rdh_scheduled=0, rda_scheduled=2, dds_scheduled=1),
        ]

    def get_employees(self) -> List[Employee]:
        return list(self._employees)

    def get_rosters(self) -> List[Employee]:
        return self.get_employees()

    def get_timesheets(self) -> List[Dict[str, Any]]:
        rows = []
        for emp in self._employees:
            ot = 0.0 if emp.flsa_exempt else max(0.0, emp.projected_week_hours - self.FLSA_WEEKLY_OT_HOURS)
            rows.append({
                "employee": emp.name,
                "clinic": emp.clinic_name,
                "role": emp.role,
                "flsa_exempt": emp.flsa_exempt,
                "hours_week_to_date": emp.hours_week_to_date,
                "projected_week_hours": emp.projected_week_hours,
                "projected_ot_hours": round(ot, 2),
                "ot_premium_usd": round(ot * emp.hourly_rate * self.OT_PREMIUM, 2),
            })
        return rows

    def get_staffing_deficit_alerts(self) -> List[StaffingAlert]:
        alerts: List[StaffingAlert] = []

        for emp in self._employees:
            if emp.flsa_exempt:
                continue
            proj = emp.projected_week_hours
            if proj > self.FLSA_WEEKLY_OT_HOURS:
                ot = proj - self.FLSA_WEEKLY_OT_HOURS
                cost = round(ot * emp.hourly_rate * self.OT_PREMIUM, 2)
                cap = max(0.0, self.FLSA_WEEKLY_OT_HOURS - emp.hours_week_to_date)
                alerts.append(StaffingAlert(
                    clinic_id=emp.clinic_id, clinic_name=emp.clinic_name, alert_type="OVERTIME_RISK",
                    severity="HIGH" if ot >= 4 else "MODERATE",
                    description=f"{emp.name} ({emp.role}) projected at {proj:.1f}h this week "
                                f"({ot:.1f}h over FLSA 40h).",
                    action_required=f"Cap remaining scheduled hours at {cap:.1f}h or cover with float staff.",
                    est_cost_impact_usd=cost,
                ))

        for d in self._demand:
            rdh_needed = -(-d.hygiene_appointments // self.RDH_APPTS_PER_DAY) if d.hygiene_appointments else 0
            if rdh_needed > d.rdh_scheduled:
                short = rdh_needed - d.rdh_scheduled
                alerts.append(StaffingAlert(
                    clinic_id=d.clinic_id, clinic_name=d.clinic_name, alert_type="RDH_CAPACITY",
                    severity="HIGH",
                    description=f"{d.hygiene_appointments} hygiene appointments vs capacity "
                                f"{d.rdh_scheduled * self.RDH_APPTS_PER_DAY} ({d.rdh_scheduled} RDH).",
                    action_required=f"Offer {short} float RDH shift(s) or reschedule "
                                    f"{d.hygiene_appointments - d.rdh_scheduled * self.RDH_APPTS_PER_DAY} appointments.",
                ))
            if d.restorative_procedures > self.RESTORATIVE_PROCS_PER_DDS_DAY * d.dds_scheduled:
                rda_needed = int(self.TARGET_RDA_PER_DDS * d.dds_scheduled)
                if d.rda_scheduled < rda_needed:
                    alerts.append(StaffingAlert(
                        clinic_id=d.clinic_id, clinic_name=d.clinic_name, alert_type="RDA_RATIO",
                        severity="MODERATE",
                        description=f"{d.restorative_procedures} restorative procedures with "
                                    f"{d.rda_scheduled} RDA : {d.dds_scheduled} DDS (target "
                                    f"{self.TARGET_RDA_PER_DDS:.0f}:1).",
                        action_required=f"Add {rda_needed - d.rda_scheduled} RDA shift(s) from float pool.",
                    ))
        return alerts

    def draft_shift_offer(self, alert: StaffingAlert) -> Dict[str, Any]:
        """Dry-run payload (illustrative shape) - nothing is sent."""
        return {
            "dry_run": True,
            "target": "Deputy roster API (open shift / shift offer)",
            "payload": {
                "location_id": alert.clinic_id,
                "reason": alert.alert_type,
                "note": alert.action_required,
                "requires_manager_approval": True,
            },
        }


# ==========================================
# Zoho CRM (Corp Dev) - simulated
# ==========================================

class ZohoDeal(BaseModel):
    id: str
    deal_name: str
    stage: str
    practice_type: str
    location: str
    operatory_count: int
    ttm_revenue: float
    adjusted_ebitda: float
    patient_count: int
    payer_mix_ppo_pct: float
    enrichment_status: str = "Pending"
    median_hhi: Optional[float] = None
    dentists_per_10k: Optional[float] = None
    population_growth_5yr: Optional[float] = None
    fit_score: Optional[float] = None
    score_breakdown: Optional[Dict[str, float]] = None
    enrichment_notes: Optional[str] = None


# Sample market data. Production sources:
#   median_hhi           - US Census ACS 5-year, table B19013 (by ZCTA/place)
#   dentists_per_10k     - NPPES NPI registry (taxonomy 1223*) / population
#   population_growth_5yr- ACS 5-year population estimates
SAMPLE_MARKET_DATA: Dict[str, Dict[str, float]] = {
    "Round Rock, TX":     {"median_hhi": 98_000, "dentists_per_10k": 5.1, "population_growth_5yr": 14.0},
    "Plano, TX":          {"median_hhi": 112_000, "dentists_per_10k": 7.9, "population_growth_5yr": 4.5},
    "Sugar Land, TX":     {"median_hhi": 125_000, "dentists_per_10k": 7.2, "population_growth_5yr": 6.0},
    "Westlake Hills, TX": {"median_hhi": 210_000, "dentists_per_10k": 9.5, "population_growth_5yr": 3.0},
}


class ZohoClient:
    """Simulated Zoho CRM. Fit score = transparent weighted blend (0-10), weights shown in UI."""

    SCORE_WEIGHTS = {"ebitda_margin": 0.35, "growth": 0.25, "low_competition": 0.20, "ppo_mix": 0.20}

    def __init__(self):
        self._deals: Dict[str, ZohoDeal] = {
            d.id: d for d in [
                ZohoDeal(id="ZH-8819", deal_name="Apex Family Dental Care", stage="Due Diligence",
                         practice_type="General Dentistry", location="Round Rock, TX", operatory_count=6,
                         ttm_revenue=2_450_000, adjusted_ebitda=680_000, patient_count=4200, payer_mix_ppo_pct=78),
                ZohoDeal(id="ZH-8824", deal_name="Lakeview Dental Group", stage="LOI Signed",
                         practice_type="Multi-Specialty", location="Plano, TX", operatory_count=9,
                         ttm_revenue=3_900_000, adjusted_ebitda=1_120_000, patient_count=6800, payer_mix_ppo_pct=85),
                ZohoDeal(id="ZH-8831", deal_name="Sugar Land Pediatric & Ortho", stage="Identified",
                         practice_type="Pediatric / Ortho", location="Sugar Land, TX", operatory_count=7,
                         ttm_revenue=2_800_000, adjusted_ebitda=740_000, patient_count=4900, payer_mix_ppo_pct=72),
                ZohoDeal(id="ZH-8799", deal_name="Barton Creek Premier Smiles", stage="Identified",
                         practice_type="General Dentistry", location="Westlake Hills, TX", operatory_count=8,
                         ttm_revenue=3_400_000, adjusted_ebitda=990_000, patient_count=5100, payer_mix_ppo_pct=65),
            ]
        }

    def get_deals(self) -> List[ZohoDeal]:
        return list(self._deals.values())

    def get_deal(self, deal_id: str) -> Optional[ZohoDeal]:
        return self._deals.get(deal_id)

    @classmethod
    def score(cls, deal: ZohoDeal, market: Dict[str, float]) -> Dict[str, float]:
        """Each component normalised to 0-10, then weighted. Pure function, unit-testable."""
        clamp = lambda x: max(0.0, min(10.0, x))  # noqa: E731
        margin = deal.adjusted_ebitda / deal.ttm_revenue if deal.ttm_revenue else 0.0
        parts = {
            "ebitda_margin": clamp((margin - 0.15) / 0.20 * 10),          # 15% -> 0, 35% -> 10
            "growth": clamp(market["population_growth_5yr"] / 15 * 10),   # 15%+ -> 10
            "low_competition": clamp((10 - market["dentists_per_10k"]) / 6 * 10),  # 4/10k -> 10, 10/10k -> 0
            "ppo_mix": clamp((deal.payer_mix_ppo_pct - 50) / 40 * 10),    # 50% -> 0, 90% -> 10
        }
        parts = {k: round(v, 2) for k, v in parts.items()}
        parts["total"] = round(sum(parts[k] * w for k, w in cls.SCORE_WEIGHTS.items()), 2)
        return parts

    def enrich_deal(self, deal_id: str) -> ZohoDeal:
        deal = self._deals.get(deal_id)
        if not deal:
            raise ValueError(f"Deal {deal_id} not found in Zoho CRM.")
        market = SAMPLE_MARKET_DATA.get(deal.location)
        if market is None:
            deal.enrichment_status = "No market data"
            deal.enrichment_notes = f"No market data for {deal.location}; manual research required."
            return deal
        breakdown = self.score(deal, market)
        deal.median_hhi = market["median_hhi"]
        deal.dentists_per_10k = market["dentists_per_10k"]
        deal.population_growth_5yr = market["population_growth_5yr"]
        deal.fit_score = breakdown["total"]
        deal.score_breakdown = breakdown
        deal.enrichment_status = "Enriched"
        deal.enrichment_notes = (
            f"EBITDA margin {deal.adjusted_ebitda / deal.ttm_revenue:.0%}, "
            f"{market['dentists_per_10k']} dentists/10k, 5-yr growth {market['population_growth_5yr']}%. "
            f"Fit score {breakdown['total']}/10 (sample market data)."
        )
        return deal

    def writeback_payload(self, deal: ZohoDeal) -> Dict[str, Any]:
        """Dry-run Zoho CRM update payload (custom field API names are illustrative)."""
        return {
            "dry_run": True,
            "method": "PUT", "path": "/crm/v8/Deals",
            "body": {"data": [{
                "id": deal.id,
                "Median_HHI": deal.median_hhi,
                "Dentists_per_10k": deal.dentists_per_10k,
                "Pop_Growth_5yr": deal.population_growth_5yr,
                "Fit_Score": deal.fit_score,
            }]},
        }
