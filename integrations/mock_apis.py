"""
integrations/mock_apis.py - Simulated Enterprise Connectors for Clove OS

This module simulates the core external APIs used across Clove Dental's 100-office DSO:
1. OpenDentalClient: Simulates Open Dental eConnector REST API for claims, procedure lines,
   clinical charts (perio & operative notes), and claim tracking audit trails.
2. DeputyClient: Simulates Deputy Workforce Management API for clinic rosters, timesheets,
   and staffing deficit / overtime alerts.
3. ZohoClient: Simulates Zoho CRM v8 API for Corp Dev M&A practice acquisition pipeline.

HIPAA Compliance & Security Note:
- In production, these connectors interface with a HIPAA-compliant BAA gateway (e.g., AWS Bedrock /
  PrivateLink VPC endpoints).
- All patient identifiers (Names, SSNs, DOBs) are tokenized or pseudo-anonymized prior to LLM ingress
  under HIPAA Safe Harbor de-identification rules.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random
from pydantic import BaseModel, Field


# ==========================================
# Pydantic Schemas for Open Dental Entities
# ==========================================

class ClaimProc(BaseModel):
    proc_num: int
    proc_code: str  # CDT code (e.g. D2950, D4341, D2740)
    proc_desc: str
    tooth_num: Optional[str] = None
    surf: Optional[str] = None
    fee_billed: float
    remittance_remark: str
    denial_reason: str


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
    radiograph_attached: bool = True
    intraoral_photos_attached: bool = True


class DentalClaim(BaseModel):
    claim_id: int
    patient_id: int
    patient_name: str
    clinic_id: int
    clinic_name: str
    payer_id: str
    payer_name: str
    date_of_service: str
    status: str  # "Denied", "AI Review Pending", "Appealed", "Paid"
    billed_fee: float
    denial_code: str  # "CO-50", "CO-97", "CO-16"
    denial_description: str
    procs: List[ClaimProc] = Field(default_factory=list)
    tracking_notes: List[str] = Field(default_factory=list)
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


# ==========================================
# Open Dental Client Simulation
# ==========================================

class OpenDentalClient:
    """Simulates Open Dental eConnector REST API."""

    def __init__(self):
        self._claims_db: Dict[int, DentalClaim] = {}
        self._charts_db: Dict[int, ClinicalChart] = {}
        self._seed_data()

    def _seed_data(self):
        """Seeds realistic claim and chart records mirroring real-world dental practice data."""
        seed_claims = [
            {
                "claim_id": 90412,
                "patient_id": 10482,
                "patient_name": "Eleanor Vance",
                "clinic_id": 101,
                "clinic_name": "Clove Dental - Austin Downtown",
                "payer_id": "DELTADENTAL_TX",
                "payer_name": "Delta Dental of Texas",
                "date_of_service": "2026-09-18",
                "status": "Denied",
                "billed_fee": 385.00,
                "denial_code": "CO-50",
                "denial_description": "Non-covered service: Insufficient clinical documentation demonstrating medical necessity for core buildup separate from crown.",
                "procs": [
                    ClaimProc(
                        proc_num=501,
                        proc_code="D2950",
                        proc_desc="Core buildup, including any pins when required",
                        tooth_num="19",
                        surf="MODBL",
                        fee_billed=385.00,
                        remittance_remark="Procedure lacks documentation of >=50% coronal tooth structure loss.",
                        denial_reason="Payer Section 4B requires severe coronal breakdown documentation."
                    )
                ],
                "chart": ClinicalChart(
                    claim_id=90412,
                    patient_id=10482,
                    patient_name="Eleanor Vance",
                    chart_date="2026-09-18",
                    provider_name="Dr. Marcus Vance, DDS",
                    provider_npi="1849204912",
                    clinical_notes=(
                        "Tooth #19 presented with fractured disto-lingual cusp extending 2.5mm subgingivally. "
                        "Pre-existing amalgam restoration failed with recurrent caries extending into pulp chamber floor. "
                        "Excavation of extensive deep decay revealed residual coronal tooth structure breakdown measured at 65% "
                        "loss of clinical crown. Endodontic obturation verified intact. Core buildup (D2950) placed using "
                        "dual-cure composite resin with dual dentin bonding agent to provide essential retention, axial wall "
                        "resistance form, and ferrule prior to full coverage crown preparation. Intraoral photos taken post-excavation."
                    ),
                    decay_percentage=65.0,
                    probing_depths="Tooth #19: MB 3mm, B 3mm, DB 4mm, ML 3mm, L 3mm, DL 5mm (subgingival margin)",
                    mobility_score="Class 0",
                    bone_loss_percentage=10.0,
                    radiograph_attached=True,
                    intraoral_photos_attached=True
                )
            },
            {
                "claim_id": 90415,
                "patient_id": 11209,
                "patient_name": "Marcus Aurelius Thorne",
                "clinic_id": 104,
                "clinic_name": "Clove Dental - Houston Galleria",
                "payer_id": "METLIFE_DENTAL",
                "payer_name": "MetLife Dental",
                "date_of_service": "2026-09-21",
                "status": "Denied",
                "billed_fee": 620.00,
                "denial_code": "CO-16",
                "denial_description": "Claim lacks information: Missing comprehensive full-mouth periodontal charting showing probing depths >= 5mm and radiographically demonstrated bone loss.",
                "procs": [
                    ClaimProc(
                        proc_num=502,
                        proc_code="D4341",
                        proc_desc="Periodontal scaling and root planing - four or more teeth per quadrant",
                        tooth_num="Quad 1 (UR)",
                        surf="Full Quad",
                        fee_billed=310.00,
                        remittance_remark="Missing active 6-point periodontal charting and diagnostic bitewings.",
                        denial_reason="Lack of documented pocket depths >= 5mm."
                    ),
                    ClaimProc(
                        proc_num=503,
                        proc_code="D4341",
                        proc_desc="Periodontal scaling and root planing - four or more teeth per quadrant",
                        tooth_num="Quad 2 (UL)",
                        surf="Full Quad",
                        fee_billed=310.00,
                        remittance_remark="Missing active 6-point periodontal charting and diagnostic bitewings.",
                        denial_reason="Lack of documented pocket depths >= 5mm."
                    )
                ],
                "chart": ClinicalChart(
                    claim_id=90415,
                    patient_id=11209,
                    patient_name="Marcus Aurelius Thorne",
                    chart_date="2026-09-21",
                    provider_name="Dr. Sophia Chen, DMD",
                    provider_npi="1928374610",
                    clinical_notes=(
                        "Patient presented for initial periodontal therapy with generalized Stage III, Grade B periodontitis. "
                        "Heavy tenacious subgingival calculus, spontaneous bleeding on probing (BOP) at 78% of sites. "
                        "Quad 1 and Quad 2 demonstrated generalized pocket depths ranging from 5mm to 7mm on teeth #2, #3, #4, #5, "
                        "#12, #13, #14, and #15 with 25-35% radiographic horizontal bone loss evident on vertical bitewings. "
                        "Ultrasonic debridement and hand instrumentation performed under 2% Lidocaine 1:100k epi local anesthesia. "
                        "Post-op instructions given. Chlorhexidine 0.12% rinse prescribed."
                    ),
                    decay_percentage=15.0,
                    probing_depths="Q1: #2 (6-4-6mm), #3 (7-5-6mm), #4 (5-4-5mm); Q2: #14 (7-5-7mm), #15 (6-5-6mm)",
                    mobility_score="Class 1 on #3 and #14",
                    bone_loss_percentage=30.0,
                    radiograph_attached=True,
                    intraoral_photos_attached=True
                )
            },
            {
                "claim_id": 90422,
                "patient_id": 12844,
                "patient_name": "Julian Delgado",
                "clinic_id": 102,
                "clinic_name": "Clove Dental - Dallas Metro",
                "payer_id": "CIGNA_DENTAL",
                "payer_name": "Cigna Health and Life Dental",
                "date_of_service": "2026-09-24",
                "status": "Denied",
                "billed_fee": 1150.00,
                "denial_code": "CO-97",
                "denial_description": "Bundled code: The benefit for this service is included in the payment/allowance for another service already evaluated.",
                "procs": [
                    ClaimProc(
                        proc_num=504,
                        proc_code="D2740",
                        proc_desc="Crown - porcelain/ceramic substrate",
                        tooth_num="30",
                        surf="Full",
                        fee_billed=1150.00,
                        remittance_remark="Crown preparation deemed bundled with recent endodontic access and restorative coverage.",
                        denial_reason="Payer bundling policy regarding post-endo provisional restorations."
                    )
                ],
                "chart": ClinicalChart(
                    claim_id=90422,
                    patient_id=12844,
                    chart_date="2026-09-24",
                    patient_name="Julian Delgado",
                    provider_name="Dr. Sarah Jenkins, DDS",
                    provider_npi="1726354891",
                    clinical_notes=(
                        "Tooth #30 underwent prior root canal therapy 6 months ago at external endodontic specialty practice. "
                        "Patient presented with lost temporary composite and secondary cusp fracture of the mesio-lingual cusp. "
                        "Structural integrity severely compromised with less than 2mm sound tooth structure remaining circumferentially. "
                        "Provisional restoration was not placed by our office. Full coverage monolithic zirconia crown (D2740) "
                        "is medically necessary to prevent catastrophic non-restorable vertical root fracture under masticatory forces. "
                        "Final impression with polyether taken, margin supragingival buccal and equigingival lingual."
                    ),
                    decay_percentage=55.0,
                    probing_depths="Tooth #30: 3mm all sites, no bleeding",
                    mobility_score="Class 0",
                    bone_loss_percentage=5.0,
                    radiograph_attached=True,
                    intraoral_photos_attached=True
                )
            },
            {
                "claim_id": 90428,
                "patient_id": 13410,
                "patient_name": "Amina Al-Mansoor",
                "clinic_id": 108,
                "clinic_name": "Clove Dental - San Antonio North",
                "payer_id": "GUARDIAN_DENTAL",
                "payer_name": "Guardian Life Dental",
                "date_of_service": "2026-09-28",
                "status": "Denied",
                "billed_fee": 1450.00,
                "denial_code": "CO-50",
                "denial_description": "Non-covered service: Surgical guide (D6190) and bone graft (D7953) denied as incidental to implant placement (D6010).",
                "procs": [
                    ClaimProc(
                        proc_num=505,
                        proc_code="D7953",
                        proc_desc="Bone replacement graft for ridge preservation - per site",
                        tooth_num="14",
                        surf="Ridge",
                        fee_billed=450.00,
                        remittance_remark="Deemed inclusive to surgical extraction.",
                        denial_reason="Payer cross-coding bundle policy."
                    ),
                    ClaimProc(
                        proc_num=506,
                        proc_code="D6010",
                        proc_desc="Surgical placement of implant body: endosteal implant",
                        tooth_num="14",
                        surf="Site",
                        fee_billed=1000.00,
                        remittance_remark="Primary procedure approved but ancillary graft bundle denied.",
                        denial_reason="Denial of preservation graft line item."
                    )
                ],
                "chart": ClinicalChart(
                    claim_id=90428,
                    patient_id=13410,
                    patient_name="Amina Al-Mansoor",
                    chart_date="2026-09-28",
                    provider_name="Dr. Aaron Patel, DDS",
                    provider_npi="1556273849",
                    clinical_notes=(
                        "Patient presented with vertical root fracture on tooth #14 requiring atraumatic sectioning and extraction. "
                        "Post-extraction buccal plate defect noted (Class II ridge deficiency). To allow for subsequent endosteal "
                        "implant osseointegration and avoid sinus perforation, freeze-dried bone allograft (0.5cc FDBA) with "
                        "resorbable collagen membrane was placed. This was an independent regenerative ridge preservation procedure, "
                        "not incidental to routine extraction. CBCT cross-sectional scans confirm severe crestal height deficiency."
                    ),
                    decay_percentage=40.0,
                    probing_depths="Site #14: 8mm buccal sulcus pre-extraction at fracture line",
                    mobility_score="Class 2 on #14 prior to extraction",
                    bone_loss_percentage=45.0,
                    radiograph_attached=True,
                    intraoral_photos_attached=True
                )
            }
        ]

        for item in seed_claims:
            claim_data = item.copy()
            chart = claim_data.pop("chart")
            claim = DentalClaim(**claim_data)
            self._claims_db[claim.claim_id] = claim
            self._charts_db[claim.claim_id] = chart

    def get_claims(self, status: Optional[str] = "Denied") -> List[DentalClaim]:
        """Returns claims filtered by status (default 'Denied')."""
        if status is None:
            return list(self._claims_db.values())
        return [c for c in self._claims_db.values() if c.status.lower() == status.lower()]

    def get_claim(self, claim_id: int) -> Optional[DentalClaim]:
        """Returns a single claim by claim ID."""
        return self._claims_db.get(claim_id)

    def get_claim_procs(self, claim_id: int) -> List[ClaimProc]:
        """Returns specific CDT procedure lines and remittance remarks for a claim."""
        claim = self._claims_db.get(claim_id)
        return claim.procs if claim else []

    def get_clinical_chart(self, claim_id: int) -> Optional[ClinicalChart]:
        """Returns clinical doctor notes and tooth surface decay logs."""
        return self._charts_db.get(claim_id)

    def post_claim_tracking(self, claim_id: int, appeal_text: str, tracking_def_num: int = 104) -> Dict[str, Any]:
        """
        Commits drafted appeals back to Open Dental with status 'AI Review Pending'.
        tracking_def_num 104 corresponds to 'AI RCM Appeal Submitted' definition in Open Dental.
        """
        claim = self._claims_db.get(claim_id)
        if not claim:
            raise ValueError(f"Claim ID {claim_id} not found in Open Dental database.")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        audit_entry = f"[{now_str}] DefNum={tracking_def_num} Status='AI Review Pending': Appeal committed. Summary: {appeal_text[:120]}..."
        claim.tracking_notes.append(audit_entry)
        claim.status = "AI Review Pending"
        claim.last_updated = now_str

        return {
            "success": True,
            "claim_id": claim_id,
            "status": "AI Review Pending",
            "tracking_def_num": tracking_def_num,
            "timestamp": now_str,
            "audit_entry": audit_entry
        }


# ==========================================
# Deputy Workforce Management Client
# ==========================================

class RosterEntry(BaseModel):
    roster_id: int
    clinic_id: int
    clinic_name: str
    employee_id: int
    employee_name: str
    role: str  # "Doctor", "Lead Hygienist (RDH)", "Dental Assistant (RDA)", "Front Desk"
    shift_start: str
    shift_end: str
    scheduled_hours: float
    actual_hours: float
    hourly_rate: float


class StaffingAlert(BaseModel):
    clinic_id: int
    clinic_name: str
    alert_type: str  # "DEFICIT", "OVERTIME_RISK", "UNCOVERED_OPERATORY"
    severity: str    # "HIGH", "CRITICAL", "MODERATE"
    description: str
    action_required: str


class DeputyClient:
    """Simulates Deputy Workforce Management API for Clove Dental rollup."""

    def __init__(self):
        self._seed_workforce()

    def _seed_workforce(self):
        self._rosters: List[RosterEntry] = [
            RosterEntry(
                roster_id=1001,
                clinic_id=101,
                clinic_name="Clove Dental - Austin Downtown",
                employee_id=201,
                employee_name="Dr. Marcus Vance, DDS",
                role="Doctor",
                shift_start="07:30 AM",
                shift_end="04:30 PM",
                scheduled_hours=8.0,
                actual_hours=9.5,
                hourly_rate=145.0
            ),
            RosterEntry(
                roster_id=1002,
                clinic_id=101,
                clinic_name="Clove Dental - Austin Downtown",
                employee_id=202,
                employee_name="Rachel Evans, RDH",
                role="Lead Hygienist (RDH)",
                shift_start="08:00 AM",
                shift_end="05:00 PM",
                scheduled_hours=8.0,
                actual_hours=8.0,
                hourly_rate=52.0
            ),
            RosterEntry(
                roster_id=1003,
                clinic_id=101,
                clinic_name="Clove Dental - Austin Downtown",
                employee_id=203,
                employee_name="Tyler Ross, RDA",
                role="Dental Assistant (RDA)",
                shift_start="07:30 AM",
                shift_end="04:00 PM",
                scheduled_hours=8.0,
                actual_hours=10.0,
                hourly_rate=28.0
            ),
            RosterEntry(
                roster_id=1004,
                clinic_id=104,
                clinic_name="Clove Dental - Houston Galleria",
                employee_id=204,
                employee_name="Dr. Sophia Chen, DMD",
                role="Doctor",
                shift_start="08:00 AM",
                shift_end="05:00 PM",
                scheduled_hours=8.0,
                actual_hours=8.5,
                hourly_rate=150.0
            ),
            RosterEntry(
                roster_id=1005,
                clinic_id=104,
                clinic_name="Clove Dental - Houston Galleria",
                employee_id=205,
                employee_name="David Miller, RDH",
                role="Lead Hygienist (RDH)",
                shift_start="08:00 AM",
                shift_end="05:00 PM",
                scheduled_hours=8.0,
                actual_hours=7.5,
                hourly_rate=54.0
            ),
            RosterEntry(
                roster_id=1006,
                clinic_id=102,
                clinic_name="Clove Dental - Dallas Metro",
                employee_id=206,
                employee_name="Dr. Sarah Jenkins, DDS",
                role="Doctor",
                shift_start="08:00 AM",
                shift_end="04:30 PM",
                scheduled_hours=8.0,
                actual_hours=8.0,
                hourly_rate=140.0
            ),
            RosterEntry(
                roster_id=1007,
                clinic_id=108,
                clinic_name="Clove Dental - San Antonio North",
                employee_id=207,
                employee_name="Dr. Aaron Patel, DDS",
                role="Doctor",
                shift_start="08:00 AM",
                shift_end="05:00 PM",
                scheduled_hours=8.0,
                actual_hours=9.0,
                hourly_rate=145.0
            ),
            RosterEntry(
                roster_id=1008,
                clinic_id=108,
                clinic_name="Clove Dental - San Antonio North",
                employee_id=208,
                employee_name="Carla Gomez, RDA",
                role="Dental Assistant (RDA)",
                shift_start="07:45 AM",
                shift_end="04:15 PM",
                scheduled_hours=8.0,
                actual_hours=10.5,
                hourly_rate=29.0
            )
        ]

    def get_rosters(self) -> List[RosterEntry]:
        """Returns current staff rosters across active dental clinics."""
        return self._rosters

    def get_timesheets(self) -> List[Dict[str, Any]]:
        """Returns summarized timesheets with overtime calculation."""
        timesheets = []
        for r in self._rosters:
            ot_hours = max(0.0, r.actual_hours - r.scheduled_hours)
            timesheets.append({
                "employee_id": r.employee_id,
                "employee_name": r.employee_name,
                "clinic_name": r.clinic_name,
                "role": r.role,
                "scheduled_hours": r.scheduled_hours,
                "actual_hours": r.actual_hours,
                "overtime_hours": ot_hours,
                "estimated_shift_cost": round(r.actual_hours * r.hourly_rate + ot_hours * r.hourly_rate * 0.5, 2)
            })
        return timesheets

    def get_staffing_deficit_alerts(self) -> List[StaffingAlert]:
        """Identifies clinic staffing shortages vs. appointment volume and overtime violations."""
        alerts = [
            StaffingAlert(
                clinic_id=101,
                clinic_name="Clove Dental - Austin Downtown",
                alert_type="DEFICIT",
                severity="CRITICAL",
                description="Operatory #4 (Periodontal hygiene recall) scheduled with 11 patients but lacks assigned RDH. Rachel Evans at capacity.",
                action_required="Dispatch per-diem float hygienist from South Austin hub or convert 3 cleanings to prophy assist."
            ),
            StaffingAlert(
                clinic_id=101,
                clinic_name="Clove Dental - Austin Downtown",
                alert_type="OVERTIME_RISK",
                severity="HIGH",
                description="Tyler Ross (RDA) has accumulated 48.5 hours this pay period (8.5 hrs OT, projected +$350 payroll drag).",
                action_required="Cap shift at 16:00 and assign evening sterilization run to float technician."
            ),
            StaffingAlert(
                clinic_id=108,
                clinic_name="Clove Dental - San Antonio North",
                alert_type="OVERTIME_RISK",
                severity="MODERATE",
                description="Carla Gomez (RDA) is trending at 44.2 hours; surgeon running 45 min behind on implant placements.",
                action_required="Rebalance operatory turnover tasks with front desk cross-trained staff."
            ),
            StaffingAlert(
                clinic_id=104,
                clinic_name="Clove Dental - Houston Galleria",
                alert_type="UNCOVERED_OPERATORY",
                severity="MODERATE",
                description="Dr. Chen booked 14 restorative procedures tomorrow; only 1 RDA scheduled. Ideal ratio is 2 RDA:1 DDS.",
                action_required="Authorize second assistant shift from Houston Central float pool."
            )
        ]
        return alerts


# ==========================================
# Zoho CRM v8 Client Simulation
# ==========================================

class ZohoDeal(BaseModel):
    id: str
    deal_name: str
    stage: str  # "Identified", "LOI Signed", "Due Diligence", "Enriched & Ready"
    practice_type: str  # "General Dentistry", "Pediatric / Ortho", "Multi-Specialty"
    location: str
    operatory_count: int
    ttm_revenue: float
    adjusted_ebitda: float
    patient_count: int
    payer_mix_ppo_pct: float
    enrichment_status: str
    median_hhi: Optional[float] = None
    dentists_per_10k: Optional[float] = None
    population_growth_5yr: Optional[float] = None
    dso_synergy_score: Optional[float] = None
    enrichment_notes: Optional[str] = None


class ZohoClient:
    """Simulates Zoho CRM v8 API for M&A target acquisition and demographic enrichment."""

    def __init__(self):
        self._deals: Dict[str, ZohoDeal] = {}
        self._seed_deals()

    def _seed_deals(self):
        deals_data = [
            ZohoDeal(
                id="ZH-8819",
                deal_name="Apex Family Dental Care",
                stage="Due Diligence",
                practice_type="General Dentistry",
                location="Round Rock, TX",
                operatory_count=6,
                ttm_revenue=2450000.0,
                adjusted_ebitda=680000.0,
                patient_count=4200,
                payer_mix_ppo_pct=78.0,
                enrichment_status="Pending",
                median_hhi=None,
                dentists_per_10k=None,
                population_growth_5yr=None,
                dso_synergy_score=None,
                enrichment_notes=None
            ),
            ZohoDeal(
                id="ZH-8824",
                deal_name="Lakeview Dental Group",
                stage="LOI Signed",
                practice_type="Multi-Specialty",
                location="Plano, TX",
                operatory_count=9,
                ttm_revenue=3900000.0,
                adjusted_ebitda=1120000.0,
                patient_count=6800,
                payer_mix_ppo_pct=85.0,
                enrichment_status="Pending",
                median_hhi=None,
                dentists_per_10k=None,
                population_growth_5yr=None,
                dso_synergy_score=None,
                enrichment_notes=None
            ),
            ZohoDeal(
                id="ZH-8831",
                deal_name="Sugar Land Pediatric & Ortho",
                stage="Identified",
                practice_type="Pediatric / Ortho",
                location="Sugar Land, TX",
                operatory_count=7,
                ttm_revenue=2800000.0,
                adjusted_ebitda=740000.0,
                patient_count=4900,
                payer_mix_ppo_pct=72.0,
                enrichment_status="Pending",
                median_hhi=None,
                dentists_per_10k=None,
                population_growth_5yr=None,
                dso_synergy_score=None,
                enrichment_notes=None
            ),
            ZohoDeal(
                id="ZH-8799",
                deal_name="Barton Creek Premier Smiles",
                stage="Enriched & Ready",
                practice_type="General Dentistry",
                location="Westlake Hills, TX",
                operatory_count=8,
                ttm_revenue=3400000.0,
                adjusted_ebitda=990000.0,
                patient_count=5100,
                payer_mix_ppo_pct=65.0,
                enrichment_status="Enriched",
                median_hhi=142000.0,
                dentists_per_10k=5.4,
                population_growth_5yr=8.2,
                dso_synergy_score=9.4,
                enrichment_notes="High-wealth demographic with favorable PPO out-of-network margin. Fast tuck-in target."
            )
        ]
        for d in deals_data:
            self._deals[d.id] = d

    def get_deals(self) -> List[ZohoDeal]:
        """Returns deals pipeline from Zoho CRM."""
        return list(self._deals.values())

    def get_deal(self, deal_id: str) -> Optional[ZohoDeal]:
        """Returns a single deal by ID."""
        return self._deals.get(deal_id)

    def enrich_deal(self, deal_id: str) -> ZohoDeal:
        """
        Enriches a deal by performing automated demographic, macroeconomic,
        and DSO synergy scoring, then writing back to Zoho custom fields.
        """
        deal = self._deals.get(deal_id)
        if not deal:
            raise ValueError(f"Deal ID {deal_id} not found in Zoho CRM.")

        # Deterministic enrichment values based on location & financials
        base_hhi = 98000.0 + (hash(deal.location) % 45000)
        dentists_ratio = round(4.2 + ((hash(deal.deal_name) % 30) / 10.0), 1)
        growth_rate = round(6.5 + ((hash(deal.location) % 40) / 10.0), 1)
        synergy = round(min(9.8, max(6.5, (deal.adjusted_ebitda / deal.ttm_revenue) * 25.0 + (growth_rate * 0.2))), 1)

        deal.median_hhi = round(base_hhi, 2)
        deal.dentists_per_10k = dentists_ratio
        deal.population_growth_5yr = growth_rate
        deal.dso_synergy_score = synergy
        deal.enrichment_status = "Enriched"
        deal.stage = "Enriched & Ready"
        deal.enrichment_notes = (
            f"Automated AI Scan Completed: Median HHI ${deal.median_hhi:,.0f}, "
            f"Provider Density {deal.dentists_per_10k} dentists/10k pop, "
            f"5-Year Pop Growth {deal.population_growth_5yr}%. "
            f"Estimated EBITDA multiple expansion: +1.8x post-Clove centralized procurement & RCM rollup."
        )

        return deal

    def post_deal_enrichment(
        self,
        deal_id: str,
        median_hhi: float,
        dentists_per_10k: float,
        population_growth_5yr: float,
        dso_synergy_score: float,
        notes: Optional[str] = None
    ) -> ZohoDeal:
        """Writes demographic scan and synergy evaluation results directly to Zoho deal fields."""
        deal = self._deals.get(deal_id)
        if not deal:
            raise ValueError(f"Deal ID {deal_id} not found in Zoho CRM.")

        deal.median_hhi = median_hhi
        deal.dentists_per_10k = dentists_per_10k
        deal.population_growth_5yr = population_growth_5yr
        deal.dso_synergy_score = dso_synergy_score
        deal.enrichment_status = "Enriched"
        deal.stage = "Enriched & Ready"
        if notes:
            deal.enrichment_notes = notes

        return deal

    def post_deal(self, deal_data: Dict[str, Any]) -> ZohoDeal:
        """Creates or updates a deal in Zoho CRM."""
        deal = ZohoDeal(**deal_data)
        self._deals[deal.id] = deal
        return deal

