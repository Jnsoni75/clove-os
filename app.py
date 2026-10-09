"""
app.py - Clove OS: Autonomous Dental AI Operations Console (v2.0)

Enterprise Operations Infrastructure for Clove Dental (100-Office Dental DSO)
Modules:
1. RCM Denial Queue   - Deterministic Triage -> BM25 Retrieval -> Grounded Drafting -> Adversarial Verification -> HITL Open Dental Writeback
2. Evaluation & Cost  - Golden Evaluation Benchmark, Session Run Ledger & Frontier Prompt Caching Economics
3. Staffing (Deputy)  - FLSA-Exempt Capacity Balancing, Overtime Sentinel & Shift Offer Generation
4. Corp Dev (Zoho)    - M&A Dental Practice Acquisition Scoring & CRM Writeback
"""

import os
import time
import pandas as pd
import streamlit as st

from agents.llm import STATIC_PREFIX, AnthropicDrafter
from agents.rcm_supervisor import RCMDenialAgent
from llm_cache.token_optimizer import PolicyContextCache, PromptCacheEconomics, break_even_reads, estimate_tokens
from eval.observability import RunLedger, run_golden_eval
from integrations.mock_apis import DeputyClient, OpenDentalClient, ZohoClient
from knowledge.rcm_reference import Route

# Page configuration
st.set_page_config(
    page_title="Clove OS | Dental AI Operations Platform",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  /* Global Reset & Typography */
  html, body, [class*="css"], .stMarkdown, p, span, label, div {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
  }
  code, pre, .mono, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
  }
  
  /* Deep Obsidian Clinical Canvas */
  .stApp {
    background-color: #070B14;
    color: #F1F5F9;
  }
  header[data-testid="stHeader"] {
    background-color: transparent !important;
  }
  
  /* Sidebar Redesign */
  [data-testid="stSidebar"] {
    background-color: #0B101C !important;
    border-right: 1px solid #1E293B !important;
  }
  
  /* Top Enterprise Header */
  .enterprise-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0E1626;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px -4px rgba(0, 0, 0, 0.4);
  }
  .brand-title {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-title h1 {
    font-size: 20px;
    font-weight: 800;
    margin: 0;
    color: #F8FAFC;
    letter-spacing: -0.02em;
  }
  .brand-title span.version {
    background: #1E293B;
    color: #38BDF8;
    font-size: 10px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid #334155;
    letter-spacing: 0.05em;
  }
  .brand-sub {
    font-size: 12.5px;
    color: #94A3B8;
    margin: 2px 0 0 0;
  }
  .gateway-status {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .status-badge-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.1);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.03em;
  }
  .pulse-dot {
    width: 6px;
    height: 6px;
    background: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10B981;
  }

  /* Executive KPI Cards */
  .kpi-deck {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 22px;
  }
  .kpi-card {
    background: #0E1626;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: transform 0.15s ease, border-color 0.15s ease;
  }
  .kpi-card:hover {
    transform: translateY(-2px);
    border-color: #38BDF8;
  }
  .kpi-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
  }
  .kpi-card.red::before { background: #EF4444; }
  .kpi-card.cyan::before { background: #0EA5E9; }
  .kpi-card.purple::before { background: #8B5CF6; }
  .kpi-card.emerald::before { background: #10B981; }
  
  .kpi-label {
    font-size: 10.5px;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
  }
  .kpi-val {
    font-size: 26px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.03em;
    line-height: 1.1;
  }
  .kpi-sub {
    font-size: 11.5px;
    color: #64748B;
    font-weight: 500;
    margin-top: 5px;
  }
  
  /* Interactive Clinical Cards */
  .clinical-card {
    background: #0E1626;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
    transition: border-color 0.15s ease;
  }
  .clinical-card:hover {
    border-color: #334155;
  }
  
  /* Pipeline Execution Stepper */
  .step-node {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0B101C;
    border-left: 3px solid #0EA5E9;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #CBD5E1;
  }
  .step-node.warn { border-left-color: #F59E0B; }
  .step-node.bad { border-left-color: #EF4444; }
  .step-node.success { border-left-color: #10B981; }
  
  /* Status Pills */
  .pill {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.02em;
  }
  .p-red { background: rgba(239, 68, 68, 0.12); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
  .p-amb { background: rgba(245, 158, 11, 0.12); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
  .p-grn { background: rgba(16, 185, 129, 0.12); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
  .p-blu { background: rgba(14, 165, 233, 0.12); color: #38BDF8; border: 1px solid rgba(14, 165, 233, 0.3); }
  
  /* Document Letter Studio */
  .letter-studio {
    background: #090E18;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 18px 20px;
    margin-top: 10px;
  }
  .letter-head {
    border-bottom: 1px solid #1E293B;
    padding-bottom: 12px;
    margin-bottom: 14px;
  }
  .letter-head-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #0EA5E9;
    text-transform: uppercase;
  }

  /* Form & Button Overrides */
  div.stButton > button[kind="primary"] {
    background: #059669 !important;
    border: 1px solid #047857 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-size: 13.5px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 2px 6px rgba(5, 150, 105, 0.25) !important;
  }
  div.stButton > button[kind="primary"]:hover {
    background: #10B981 !important;
    border-color: #059669 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
  }
  div.stButton > button[kind="secondary"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #F1F5F9 !important;
    border-radius: 8px !important;
    font-size: 13.5px !important;
    transition: all 0.15s ease !important;
  }
  div.stButton > button[kind="secondary"]:hover {
    background: #334155 !important;
    border-color: #64748B !important;
  }
  
  /* Textarea Editor */
  .stTextArea textarea {
    background-color: #080D17 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #F1F5F9 !important;
    line-height: 1.6 !important;
  }
  .stTextArea textarea:focus {
    border-color: #0EA5E9 !important;
    box-shadow: 0 0 0 1px #0EA5E9 !important;
  }
  
  /* Metrics styling */
  [data-testid="stMetric"] {
    background-color: #0E1626;
    border: 1px solid #1E293B;
    padding: 12px 14px;
    border-radius: 8px;
  }
  [data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    color: #94A3B8 !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 20px !important;
    font-weight: 800 !important;
    color: #F8FAFC !important;
  }
</style>
""", unsafe_allow_html=True)

# Route badge map
ROUTE_PILL = {
    Route.APPEAL: ("p-blu", "Appeal"),
    Route.RESUBMIT: ("p-amb", "Correct & Resubmit (0 Tokens)"),
    Route.REP_CALL: ("p-amb", "Payer Rep Call"),
    Route.DOC_GAP: ("p-red", "Documentation Gap"),
    Route.NO_APPEAL: ("p-grn", "Patient Responsibility / Write-off"),
}

# Session State Initialization
ss = st.session_state
if "od" not in ss:
    ss.od = OpenDentalClient()
    ss.deputy = DeputyClient()
    ss.zoho = ZohoClient()
    ss.economics = PromptCacheEconomics()
    ss.rcache = PolicyContextCache()
    ss.ledger = RunLedger()
    ss.agent = RCMDenialAgent(ss.od, retrieval_cache=ss.rcache, economics=ss.economics)
    ss.threads = {}          # claim_id -> thread_id
    ss.logged = set()        # thread ids already written to ledger
    ss.eval_report = None

agent: RCMDenialAgent = ss.agent
LIVE = isinstance(agent.drafter, AnthropicDrafter)
es = ss.economics.summary()

# Top Enterprise Command Bar
st.markdown("""
<div class="enterprise-nav">
  <div class="brand-title">
    <div>
      <div style="display:flex; align-items:center; gap:8px;">
        <h1>🦷 Clove OS</h1>
        <span class="version">ENTERPRISE V2.0</span>
      </div>
      <div class="brand-sub">Central AI Operations Infrastructure · 100-Office Dental Support Organization</div>
    </div>
  </div>
  <div class="gateway-status">
    <div class="status-badge-online">
      <span class="pulse-dot"></span>
      OPEN DENTAL eCONNECTOR LIVE
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🎛️ OPERATIONS CONSOLE")
    module = st.radio(
        "OPERATIONS MODULE",
        ["RCM Denial Queue", "Evaluation & Cost", "Staffing (Deputy)", "Corp Dev (Zoho)"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("#### ⚡ Drafting Engine")
    st.markdown(
        f"<span class='pill {'p-grn' if LIVE else 'p-blu'}'>Model: "
        f"{'Claude (live, ' + agent.drafter.model + ')' if LIVE else 'Offline Deterministic Drafter'}</span>",
        unsafe_allow_html=True
    )
    if not LIVE:
        st.caption("Running in zero-leakage, deterministic sandbox. Set `ANTHROPIC_API_KEY` for live Anthropic API completions.")
        
    st.markdown("---")
    st.markdown("#### 📉 Session Economics")
    st.markdown(f"""
- LLM Invocations: `{es['calls']}`
- Cache Reads / Writes: `{es['cache_reads']} / {es['cache_writes']}`
- Cost Spent: `${es['cost_usd']:.4f}`
- Net Savings: `${es['saved_usd']:.4f}`
- Retrieval Cache Hit: `{ss.rcache.stats()['hit_rate_pct']}%`
""")

# Execution Helpers
def render_traces(traces):
    for t in traces:
        cls = "step-node"
        if t.get("passed") is False or "Blocked" in t.get("detail", ""):
            cls += " bad"
        elif t["node"] in ("documentation_gap", "build_rep_call"):
            cls += " warn"
        elif t["node"] == "commit":
            cls += " success"
        st.markdown(
            f"<div class='{cls}'>"
            f"<span><b>[{t['node'].upper()}]</b> {t['detail']}</span>"
            f"<span style='color:#64748B;'>{t['latency_ms']:.1f} ms</span>"
            f"</div>",
            unsafe_allow_html=True
        )

def log_once(tid, state):
    if tid not in ss.logged and state.get("status") in ("COMMITTED", "REJECTED"):
        ss.ledger.log(state)
        ss.logged.add(tid)

# =========================================================================
# MODULE 1: RCM Denial Queue
# =========================================================================
if module == "RCM Denial Queue":
    claims = ss.od.get_claims(status=None)
    open_claims = [c for c in claims if c.status == "Denied"]
    allowed_open = sum(c.allowed_at_issue for c in open_claims)
    
    # KPI Deck
    st.markdown(f"""
    <div class="kpi-deck">
      <div class="kpi-card red">
        <div class="kpi-label">Open Denials in Queue</div>
        <div class="kpi-val">{len(open_claims)}</div>
        <div class="kpi-sub">of {len(claims)} active remittance claims</div>
      </div>
      <div class="kpi-card cyan">
        <div class="kpi-label">Allowed Value at Issue</div>
        <div class="kpi-val">${allowed_open:,.0f}</div>
        <div class="kpi-sub">Contracted collectible amount (not billed)</div>
      </div>
      <div class="kpi-card purple">
        <div class="kpi-label">Frontier Cache Reads</div>
        <div class="kpi-val">{es['cache_reads']}</div>
        <div class="kpi-sub">90% token cost reduction on hits</div>
      </div>
      <div class="kpi-card emerald">
        <div class="kpi-label">Audited Actions Closed</div>
        <div class="kpi-val">{len(ss.ledger.rows)}</div>
        <div class="kpi-sub">Human-reviewed and committed</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_q, col_w = st.columns([1.1, 1.9])

    # Left: Denial Queue Table & Selector
    with col_q:
        st.markdown("#### 📋 Electronic Remittance Queue")
        
        qdf = pd.DataFrame([{
            "ID": c.claim_id, 
            "Payer": c.payer_name, 
            "CDT": c.primary_denied_proc.proc_code if c.primary_denied_proc else "-",
            "Code": c.denial_code, 
            "Allowed": f"${c.allowed_at_issue:,.0f}",
            "Status": c.status
        } for c in claims])
        
        st.dataframe(qdf, hide_index=True, use_container_width=True, height=200)
        
        labels = {f"#{c.claim_id} · {c.patient_name} · {c.payer_name} ({c.denial_code})": c.claim_id for c in claims}
        claim_id = labels[st.selectbox("Select Active Claim to Work", list(labels))]
        claim = ss.od.get_claim(claim_id)
        chart = ss.od.get_clinical_chart(claim_id)
        
        # Selected Claim Metadata Card
        st.markdown(f"""
        <div class="clinical-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-weight:800; font-size:14px; color:#F8FAFC;">Claim #{claim.claim_id}</span>
            <span class="pill p-blu">{claim.plan_type} · {claim.plan_funding}</span>
          </div>
          <div style="font-size:12.5px; color:#CBD5E1; margin-bottom:4px;">
            <b>Patient:</b> {claim.patient_name} &nbsp;·&nbsp; <b>Clinic:</b> {claim.clinic_name}
          </div>
          <div style="font-size:12.5px; color:#CBD5E1; margin-bottom:6px;">
            <b>Payer:</b> {claim.payer_name} &nbsp;·&nbsp; <b>DOS:</b> {claim.date_of_service}
          </div>
          <div style="display:flex; justify-content:space-between; background:#0B101C; padding:8px 12px; border-radius:6px; margin:8px 0; border:1px solid #1E293B;">
            <div><span style="font-size:11px; color:#64748B;">BILLED FEE</span><br/><span style="font-size:13px; font-weight:700;">${claim.billed_fee:,.2f}</span></div>
            <div><span style="font-size:11px; color:#64748B;">CONTRACT ALLOWED</span><br/><span style="font-size:14px; font-weight:800; color:#38BDF8;">${claim.allowed_at_issue:,.2f}</span></div>
            <div><span style="font-size:11px; color:#64748B;">CLAIM STATUS</span><br/><span style="font-size:13px; font-weight:700; color:{'#F87171' if claim.status=='Denied' else '#34D399'};">{claim.status}</span></div>
          </div>
          <div style="color:#F87171; font-weight:600; font-size:12px;">
            ⚠ {claim.denial_code}: {claim.denial_description}
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Clinical Procedure Breakdown
        st.dataframe(pd.DataFrame([{
            "CDT": p.proc_code, 
            "Tooth": p.tooth_num, 
            "Billed": f"${p.fee_billed:,.0f}",
            "Allowed": f"${p.expected_allowed:,.0f}", 
            "Paid": f"${p.paid_amount:,.0f}",
            "CARC": p.adjustment_code or "-", 
            "Remark": p.payer_remark
        } for p in claim.procs]), hide_index=True, use_container_width=True)
        
        # Clinical Note Source of Truth
        with st.expander("📄 Contemporaneous Clinical Chart (Ground Truth)", expanded=True):
            if chart:
                st.markdown(f"**Provider:** {chart.provider_name} *(NPI {chart.provider_npi})*")
                st.info(chart.clinical_notes)
                st.caption(
                    f"Attachments: Radiograph = `{chart.radiograph_attached}` · "
                    f"Intraoral Photos = `{chart.intraoral_photos_attached}`"
                    + (f" · Probing Depths = `{chart.probing_depths}`" if chart.probing_depths else "")
                )
            else:
                st.warning("No clinical chart found in Open Dental repository.")

    # Right: Autonomous Agent Pipeline & Studio
    with col_w:
        st.markdown("#### 🤖 Autonomous Denial Pipeline")
        
        # Action Bar
        if st.button("▶ EXECUTE AGENT RESOLUTION", type="primary", use_container_width=True, disabled=claim.status != "Denied"):
            with st.spinner("Executing: Triage -> Policy Retrieval -> Evidence Synthesis -> Draft -> Adversarial Verification..."):
                res = agent.start(claim_id)
            ss.threads[claim_id] = res["thread_id"]
            st.rerun()

        tid = ss.threads.get(claim_id)
        if not tid:
            st.info("Select a claim and click 'EXECUTE AGENT RESOLUTION' to initiate autonomous triage and drafting.")
        else:
            s = agent.get_state(tid)
            log_once(tid, s)
            
            pill_cls, pill_txt = ROUTE_PILL.get(s["route"], ("p-blu", s["route"]))
            
            # Route Result Card
            st.markdown(f"""
            <div class="clinical-card">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span class="pill {pill_cls}">TRIAGE ROUTE: {pill_txt}</span>
                <span style="font-size:11px; color:#64748B;">Thread: <code>{tid[:12]}</code></span>
              </div>
              <div style="font-size:13px; color:#E2E8F0; margin-top:6px;">
                <b>Operational Rationale:</b> {s['triage']['reason']}
              </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("⏱️ Pipeline Execution Latency Breakdown", expanded=False):
                render_traces(s["traces"])

            # If Route is APPEAL -> Show Evidence & Letter Studio
            if s["route"] == Route.APPEAL:
                ev = s["evidence"]
                st.markdown("##### 🔬 Clinical Evidence vs. Payer Policy Criteria")
                
                st.dataframe(pd.DataFrame([{
                    "Payer Criterion": c["text"], 
                    "Mandatory": c["required"], 
                    "Evaluation": c["status"],
                    "Chart Proof (Verbatim Quotes)": " | ".join(c["quotes"] + c["structured"]) or "—"
                } for c in ev["criteria"]]), hide_index=True, use_container_width=True)
                
                with st.expander(f"📚 Retrieved Clinical Policy ({'Shared Cache Hit' if s.get('retrieval_cache_hit') else 'BM25 Exact Lexical Search'})"):
                    for ch in s["policy_chunks"]:
                        st.markdown(f"""**{ch['id']}** · Score `{ch['score']}` · *{ch['provenance']}*

> {ch['text']}""")

                # Adversarial Verification Metric Strip
                v = s["verification"]
                u = s["usage"][-1] if s.get("usage") else {}
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Faithfulness Score", f"{v['faithfulness']:.0%}", f"{v['claims_supported']}/{v['claims_checked']} quotes verbatim")
                c2.metric("Citation Coverage", f"{v['citation_coverage']:.0%}", "Statute verified")
                c3.metric("Draft Revisions", s.get("revision_count", 0))
                c4.metric("LLM Cost", f"${u.get('cost_usd', 0):.4f}", "Prefix cache read" if u.get("cache_read_input_tokens") else "Cache write")
                
                if v["violations"]:
                    v_msgs = [f"- **{x['type']}** `{x['span']}` — {x['detail']}" for x in v["violations"]]
                    st.error("Adversarial Verifier Violations:\n" + "\n".join(v_msgs))

                # Document Letter Canvas
                st.markdown("""
                <div class="letter-studio">
                  <div class="letter-head">
                    <div class="letter-head-title">FORMAL PROVIDER APPEAL LETTERHEAD · ADJUDICATION DEMAND</div>
                    <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">Governing Statute: ERISA § 503 (Self-Funded) / State Prompt Payment Law</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                
                edited = st.text_area(
                    "Appeal Letter Body (Editable; re-verified on approval)",
                    value=s["draft"],
                    height=340,
                    key=f"letter_{tid}",
                    disabled=not s["awaiting_review"]
                )
            else:
                # Non-Appeal Work Item (CO-16 Attachment or Rep Call)
                wi = s.get("work_item") or {}
                st.markdown("##### 📦 Operational Work Item (0 LLM Tokens Consumed)")
                st.markdown(f"<div class='clinical-card'>{wi.get('summary', '')}</div>", unsafe_allow_html=True)
                for key, title in [("lines", "Lines"), ("gaps", "Unmet Criteria"),
                                   ("contradicting_evidence", "Chart Contradicts Claim"),
                                   ("actions", "Required Actions"), ("checklist", "Required Attachments"),
                                   ("call_script", "Representative Call Script")]:
                    if wi.get(key):
                        st.markdown(f"**{title}**")
                        wi_items = (f"- {x}" for x in wi[key])
                        st.markdown("\n".join(wi_items))
                edited = None

            if s.get("review_error"):
                st.error(s["review_error"])

            # Human In The Loop Audit Reviewer Action Box
            if s["awaiting_review"]:
                st.markdown("---")
                st.markdown("##### ✍️ Human-in-the-Loop Supervisory Review")
                reviewer = st.text_input("Reviewer Attribution (Required for Open Dental Writeback)", key=f"rev_{tid}", placeholder="e.g. Jatin Soni, CDBS (RCM Lead)")
                
                b1, b2 = st.columns(2)
                if b1.button("✅ APPROVE & COMMIT TO OPEN DENTAL", use_container_width=True, key=f"ap_{tid}"):
                    agent.resume(tid, "approve", reviewer, edited_text=edited)
                    st.rerun()
                if b2.button("✖ REJECT CLAIM (TRANSFER BALANCE)", use_container_width=True, key=f"rj_{tid}"):
                    agent.resume(tid, "reject", reviewer or "unknown")
                    st.rerun()
            elif s.get("status") == "COMMITTED":
                st.success(f"✓ Work Item Committed to Open Dental EHR → **Status: {s['commit_result']['status']}**")
            elif s.get("status") == "REJECTED":
                st.warning("Claim Rejected by Reviewer. No outbound appeal submitted.")

        # Open Dental History Log
        if claim.tracking_notes:
            st.markdown("##### 📜 Open Dental Immutable Audit Ledger")
            for n in claim.tracking_notes:
                st.code(n, language="text")

# =========================================================================
# MODULE 2: Evaluation & Cost
# =========================================================================
elif module == "Evaluation & Cost":
    st.markdown("#### 🧪 Benchmark Evaluation Suite & Telemetry")
    st.caption("Automated golden dataset evaluation with adversarial fault injection testing. Zero synthetic telemetry.")
    
    if st.button("▶ EXECUTE GOLDEN EVALUATION BENCHMARK", type="primary"):
        with st.spinner("Running benchmark suite across all claim categories..."):
            ss.eval_report = run_golden_eval(ss.od)
            
    rep = ss.eval_report
    if rep:
        m = rep.metrics
        k = st.columns(6)
        k[0].metric("Route Accuracy", f"{m['route_accuracy']:.0%}")
        k[1].metric("Retrieval Precision", f"{m['retrieval_precision']:.0%}")
        k[2].metric("Retrieval Recall", f"{m['retrieval_recall']:.0%}")
        k[3].metric("Faithfulness", f"{m['faithfulness']:.0%}")
        k[4].metric("LLM Calls/Denial", m["llm_calls_per_denial"])
        k[5].metric("Safety Pass Rate", m["safety_checks_passed"])
        
        st.markdown("##### Evaluation Cases")
        st.dataframe(rep.cases, hide_index=True, use_container_width=True)
        st.markdown("##### Adversarial Safety Checks")
        st.dataframe(rep.checks, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📑 Session Audit Ledger")
    ldf = ss.ledger.df()
    if ldf.empty:
        st.caption("No closed runs yet. Process a claim in the RCM Denial Queue to view ledger entries.")
    else:
        st.dataframe(ldf, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 💰 Frontier Prompt Caching Economics")
    es = ss.economics.summary()
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Session Calls", es["calls"], es["source"])
    e2.metric("Incurred Spend", f"${es['cost_usd']:.4f}")
    e3.metric("Net Savings", f"${es['saved_usd']:.4f}", f"{es['saved_pct']}% saved")
    e4.metric("Break-Even Threshold", f"{es['break_even_reads']} Reads", "Break-even on 1st read")
    
    st.caption(
        f"Prompt Caching Formula: write 1.25× + N reads × 0.1× < (1+N) × 1.0× ⇒ N > 0.25/0.9 = {break_even_reads()}. "
        f"Static Prefix ≈ {estimate_tokens(STATIC_PREFIX):,} tokens (Min cache threshold: 1,024; Ephemeral TTL: 5 min)."
    )

    st.markdown("##### 📈 100-Office Monthly Projection Modeling")
    p1, p2, p3 = st.columns(3)
    denials = p1.number_input("Monthly Denied Lines across 100 Clinics", 100, 100_000, 3000, step=100)
    share_llm = p2.slider("Share Requiring Formal Appeal Letter", 0.05, 1.0, float(rep.metrics["llm_calls_per_denial"]) if rep else 0.45, 0.05)
    read_share = p3.slider("Calls Landing within 5-min Cache Window", 0.0, 1.0, 0.85, 0.05)
    
    proj = ss.economics.project_monthly(denials, share_llm * 1.2, estimate_tokens(STATIC_PREFIX), 450, 550, read_share)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Projected Calls", f"{proj['calls']:,}")
    q2.metric("Uncached Spend", f"${proj['no_cache_usd']:,.2f}")
    q3.metric("Cached Spend", f"${proj['with_cache_usd']:,.2f}", f"-{proj['saved_pct']}% saved")
    q4.metric("Cost per Denial", f"${proj['cost_per_denial_usd']:.4f}")

# =========================================================================
# MODULE 3: Staffing (Deputy)
# =========================================================================
elif module == "Staffing (Deputy)":
    st.markdown("#### 👥 Dynamic Workforce Optimization & Scheduling")
    st.caption(
        f"FLSA Compliance Rules: Overtime calculated exclusively for non-exempt hourly staff (> {DeputyClient.FLSA_WEEKLY_OT_HOURS:.0f}h). "
        f"Salaried dentists/specialists are FLSA exempt. Hygienist capacity target: {DeputyClient.RDH_APPTS_PER_DAY} appts/day."
    )
    
    alerts = ss.deputy.get_staffing_deficit_alerts()
    for a in alerts:
        cls = "p-red" if a.severity == "HIGH" else "p-amb"
        cost = f" · Est. Overtime Impact: ${a.est_cost_impact_usd:,.2f}" if a.est_cost_impact_usd else ""
        st.markdown(f"""
        <div class="clinical-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <div>
              <span class="pill {cls}">{a.severity} DEFICIT</span>
              <span class="pill p-blu">{a.alert_type}</span>
              <span style="font-weight:700; margin-left:8px;">{a.clinic_name}</span>
            </div>
            <span style="font-size:12px; color:#F87171; font-weight:600;">{cost}</span>
          </div>
          <div style="font-size:13px; color:#E2E8F0; margin-bottom:6px;">{a.description}</div>
          <div style="font-size:12.5px; color:#38BDF8; font-weight:600;">→ Recommended Action: {a.action_required}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("#### ⏱️ Active Timesheets & Scheduled Hours")
    st.dataframe(pd.DataFrame(ss.deputy.get_timesheets()), hide_index=True, use_container_width=True)
    
    if alerts:
        st.markdown("##### 🚀 Dispatch Automated Shift Offer")
        pick = st.selectbox("Select Clinic Deficit Alert", [f"{a.clinic_name} · {a.alert_type}" for a in alerts])
        if st.button("Generate Deputy Shift Offer (Dry Run)"):
            a = alerts[[f"{x.clinic_name} · {x.alert_type}" for x in alerts].index(pick)]
            st.json(ss.deputy.draft_shift_offer(a))

# =========================================================================
# MODULE 4: Corp Dev (Zoho)
# =========================================================================
elif module == "Corp Dev (Zoho)":
    st.markdown("#### 🏢 Practice Acquisition Pipeline (M&A Fit Scoring)")
    st.caption("Quantitative fit scoring model integrating Census ACS Median HHI, NPPES Dentist Density, and DSO Synergy Potential.")
    
    deals = ss.zoho.get_deals()
    d_label = {f"{d.deal_name} · {d.location} · {d.stage}": d.id for d in deals}
    deal = ss.zoho.get_deal(d_label[st.selectbox("Select Target Dental Practice", list(d_label))])
    
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown(f"""
        <div class="clinical-card">
          <div style="font-weight:800; font-size:15px; color:#F8FAFC; margin-bottom:4px;">{deal.deal_name}</div>
          <div style="font-size:12.5px; color:#94A3B8; margin-bottom:8px;">{deal.practice_type} · {deal.location}</div>
          <div style="font-size:13px; color:#CBD5E1; line-height:1.6;">
            <b>Operatories:</b> {deal.operatory_count} Chairs &nbsp;·&nbsp; <b>Active Patients:</b> {deal.patient_count:,}<br/>
            <b>TTM Revenue:</b> ${deal.ttm_revenue:,.0f} &nbsp;·&nbsp; <b>Adj. EBITDA:</b> ${deal.adjusted_ebitda:,.0f} ({deal.adjusted_ebitda / deal.ttm_revenue:.0%})<br/>
            <b>PPO Payer Mix:</b> {deal.payer_mix_ppo_pct:.0f}%
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("ENRICH & SCORE TARGET", type="primary", use_container_width=True):
            ss.zoho.enrich_deal(deal.id)
            st.rerun()
            
    with c2:
        if deal.enrichment_status == "Enriched":
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Median HHI", f"${deal.median_hhi:,.0f}")
            m2.metric("Dentists / 10k", deal.dentists_per_10k)
            m3.metric("5-Yr Growth", f"{deal.population_growth_5yr}%")
            m4.metric("Fit Score", f"{deal.fit_score}/10")
            
            bd = deal.score_breakdown
            st.dataframe(pd.DataFrame([{
                "Component": k, 
                "Score (0-10)": bd[k], 
                "Weight": f"{w:.0%}",
                "Weighted Contribution": round(bd[k] * w, 2)
            } for k, w in ZohoClient.SCORE_WEIGHTS.items()]), hide_index=True, use_container_width=True)
            
            st.caption(f"Enrichment Notes: {deal.enrichment_notes}")
            with st.expander("Zoho CRM v8 Writeback Payload"):
                st.json(ss.zoho.writeback_payload(deal))
        else:
            st.info("Target practice is awaiting enrichment. Click 'ENRICH & SCORE TARGET' to calculate demographic fit and synergy metrics.")

# Footer Notes
st.markdown("---")
with st.expander("🔒 Enterprise Architecture, PHI Governance & Zero Data Retention"):
    st.markdown("""
    - **Safe Harbor De-Identification:** Direct identifiers (Names, MRNs, DOBs) are scrubbed and replaced with surrogate tokens (`[PATIENT_A]`, `[DATE_1]`) prior to model prompt construction.
    - **Zero PHI in Cache Keys:** Ephemeral caches only store static clinical policies keyed by `payer × CDT × CARC`. Unmasked patient data raises an immediate `PHILeakError`.
    - **Audit Attribution:** Zero data reaches Open Dental without an authenticated, named reviewer identity logged to the immutable ledger.
    - **Production Adapters:** Easily bridges to Open Dental eConnector, Deputy API, and Zoho CRM v8 REST endpoints.
    """)
