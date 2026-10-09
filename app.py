"""
app.py - Clove OS: AI operations console.

Modules
1. RCM Denial Queue   - triage -> retrieval -> evidence -> draft -> verify -> human review -> Open Dental
2. Evaluation & Cost  - golden eval suite, session run ledger, measured/estimated token economics
3. Staffing (Deputy)  - computed overtime + capacity alerts, dry-run shift offers
4. Corp Dev (Zoho)    - transparent fit scoring on sample market data, dry-run CRM writeback

All patient data is synthetic. Integrations are simulated; see README for production mapping.
"""

import os

import pandas as pd
import streamlit as st

from agents.llm import STATIC_PREFIX, AnthropicDrafter
from agents.rcm_supervisor import RCMDenialAgent
from llm_cache.token_optimizer import PolicyContextCache, PromptCacheEconomics, break_even_reads, estimate_tokens
from eval.observability import RunLedger, run_golden_eval
from integrations.mock_apis import DeputyClient, OpenDentalClient, ZohoClient
from knowledge.rcm_reference import Route

st.set_page_config(page_title="Clove OS | Dental AI Operations", page_icon="🦷", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  /* Global typography and background */
  html, body, [class*="css"], .stMarkdown {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
  }
  code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
  }
  
  /* App background */
  .stApp {
    background-color: #0A0F1D;
    color: #E2E8F0;
  }
  
  /* Streamlit header hidden/clean */
  header[data-testid="stHeader"] {
    background-color: transparent !important;
  }

  /* Command Header Banner */
  .hdr { 
    background: linear-gradient(180deg, #131E33 0%, #0F172A 100%);
    border: 1px solid #1E293B;
    border-radius: 12px; 
    padding: 20px 24px; 
    margin-bottom: 22px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
  }
  .hdr-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .hdr h1 { 
    font-size: 22px; 
    font-weight: 800; 
    margin: 0; 
    color: #F8FAFC;
    letter-spacing: -0.02em; 
  }
  .hdr p { 
    color: #94A3B8; 
    margin: 6px 0 0 0; 
    font-size: 13.5px; 
    font-weight: 450; 
    line-height: 1.5;
  }
  .status-tag {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(16, 185, 129, 0.08); 
    color: #10B981; 
    padding: 5px 12px; 
    border-radius: 999px; 
    font-weight: 700; 
    font-size: 11px; 
    letter-spacing: 0.05em;
    border: 1px solid rgba(16, 185, 129, 0.25); 
    white-space: nowrap;
  }
  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #10B981;
    box-shadow: 0 0 8px #10B981;
  }
  
  /* High-Density KPI Grid */
  .kpis { 
    display: grid; 
    grid-template-columns: repeat(4, 1fr); 
    gap: 12px; 
    margin-bottom: 22px; 
  }
  .kpi { 
    background: #111827; 
    border: 1px solid #1F2937; 
    border-top: 3px solid #0891B2;
    border-radius: 9px; 
    padding: 14px 16px; 
    transition: transform 0.15s ease, border-color 0.15s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }
  .kpi:hover { 
    border-color: #38BDF8; 
    transform: translateY(-2px); 
  }
  .kpi-red { border-top-color: #EF4444; }
  .kpi-cyan { border-top-color: #06B6D4; }
  .kpi-purple { border-top-color: #8B5CF6; }
  .kpi-green { border-top-color: #10B981; }
  
  .kpi .l { 
    font-size: 10.5px; 
    font-weight: 700; 
    letter-spacing: 0.06em; 
    text-transform: uppercase; 
    color: #94A3B8; 
    margin-bottom: 4px; 
  }
  .kpi .v { 
    font-size: 24px; 
    font-weight: 800; 
    color: #F8FAFC; 
    margin: 2px 0; 
    letter-spacing: -0.02em;
  }
  .kpi .s { 
    font-size: 11.5px; 
    color: #64748B; 
    font-weight: 500; 
  }
  
  /* Status Badges & Pills */
  .pill { 
    display: inline-flex; 
    align-items: center;
    padding: 3px 9px; 
    border-radius: 5px; 
    font-size: 11px; 
    font-weight: 700; 
    letter-spacing: 0.02em;
  }
  .p-red { background: rgba(239, 68, 68, 0.12); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
  .p-amb { background: rgba(245, 158, 11, 0.12); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
  .p-grn { background: rgba(16, 185, 129, 0.12); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
  .p-blu { background: rgba(8, 145, 178, 0.12); color: #38BDF8; border: 1px solid rgba(8, 145, 178, 0.3); }
  
  /* Cards & Surface Containers */
  .card { 
    background: #111827; 
    border: 1px solid #1F2937; 
    border-radius: 9px; 
    padding: 14px 16px; 
    margin: 8px 0; 
    font-size: 13px; 
    color: #CBD5E1; 
    line-height: 1.55;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
  }
  
  /* Execution Step Traces */
  .step { 
    background: #0B101E; 
    border-left: 3px solid #0891B2; 
    border-radius: 5px; 
    padding: 8px 12px; 
    margin-bottom: 5px;
    font-family: 'JetBrains Mono', monospace; 
    font-size: 11.5px; 
    color: #CBD5E1;
  }
  .step.warn { border-left-color: #F59E0B; } 
  .step.bad { border-left-color: #EF4444; }

  /* Streamlit Button & Input Overrides */
  div.stButton > button[kind="primary"] {
    background: #059669 !important;
    border: 1px solid #047857 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 7px !important;
    padding: 8px 18px !important;
    font-size: 13px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 2px 6px rgba(5, 150, 105, 0.25) !important;
  }
  div.stButton > button[kind="primary"]:hover {
    background: #10B981 !important;
    border-color: #059669 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(5, 150, 105, 0.35) !important;
  }
  div.stButton > button[kind="secondary"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #F1F5F9 !important;
    border-radius: 7px !important;
    font-size: 13px !important;
    transition: all 0.15s ease !important;
  }
  div.stButton > button[kind="secondary"]:hover {
    background: #334155 !important;
    border-color: #64748B !important;
  }

  /* Metric Card Overrides */
  [data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #1F2937;
    padding: 10px 12px;
    border-radius: 8px;
  }
  [data-testid="stMetricLabel"] {
    font-size: 10.5px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    color: #94A3B8 !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
  }

  /* Text Area & Input Styling */
  .stTextArea textarea {
    background-color: #0B101E !important;
    border: 1px solid #1E293B !important;
    border-radius: 7px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #E2E8F0 !important;
    line-height: 1.5 !important;
  }
  .stTextArea textarea:focus {
    border-color: #0891B2 !important;
    box-shadow: 0 0 0 1px #0891B2 !important;
  }
  
  /* Sidebar Styling */
  [data-testid="stSidebar"] {
    background-color: #090D17 !important;
    border-right: 1px solid #1E293B !important;
  }
</style>
""", unsafe_allow_html=True)

ROUTE_PILL = {
    Route.APPEAL: ("p-blu", "Appeal"), Route.RESUBMIT: ("p-amb", "Correct & resubmit"),
    Route.REP_CALL: ("p-amb", "Payer rep call"), Route.DOC_GAP: ("p-red", "Documentation gap"),
    Route.NO_APPEAL: ("p-grn", "No appeal"),
}

# ==========================================
# Session state
# ==========================================
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

# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    st.markdown("### 🦷 CLOVE OS")
    st.caption("AI operations console · v1.0")
    st.markdown(
        f"<span class='pill {'p-grn' if LIVE else 'p-blu'}'>Drafting: "
        f"{'Claude (live, ' + agent.drafter.model + ')' if LIVE else 'Offline deterministic'}</span> "
        "<span class='pill p-amb'>Synthetic data</span>", unsafe_allow_html=True)
    if not LIVE:
        st.caption("Set ANTHROPIC_API_KEY and `pip install anthropic` for live drafting with measured prompt-cache usage.")
    module = st.radio("MODULE", ["RCM Denial Queue", "Evaluation & Cost", "Staffing (Deputy)", "Corp Dev (Zoho)"])
    st.markdown("---")
    es = ss.economics.summary()
    st.markdown("**Session LLM economics**")
    st.markdown(
        f"- Calls: `{es['calls']}` ({es['source']})\n"
        f"- Prefix cache reads / writes: `{es['cache_reads']} / {es['cache_writes']}`\n"
        f"- Cost: `${es['cost_usd']:.4f}` · saved `${es['saved_usd']:.4f}`\n"
        f"- Retrieval cache hit rate: `{ss.rcache.stats()['hit_rate_pct']}%`")

st.markdown("""<div class="hdr">
  <div class="hdr-top">
    <div>
      <div style="font-size:10.5px; font-weight:800; letter-spacing:0.08em; color:#0891B2; text-transform:uppercase; margin-bottom:2px;">CLOVE DENTAL DSO · OPERATIONS ENGINE</div>
      <h1>🦷 Clove OS — Denial Resolution & Operational Agents</h1>
      <p>
        Autonomous triage, retrieval, and verifiable appeals for dental DSO operations. 
        Zero LLM token spend on attachment denials (`CO-16`), adversarial clinical fact-checking, and auditable Open Dental writeback.
      </p>
    </div>
    <div style="text-align:right;">
      <span class="status-tag">
        <span class="status-dot"></span>
        OPEN DENTAL eCONNECTOR LIVE
      </span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ==========================================
# Helpers
# ==========================================
def render_traces(traces):
    for t in traces:
        cls = "step"
        if t.get("passed") is False or "Blocked" in t.get("detail", ""):
            cls += " bad"
        elif t["node"] in ("documentation_gap", "build_rep_call"):
            cls += " warn"
        st.markdown(f"<div class='{cls}'><b>[{t['node']}]</b> {t['detail']} "
                    f"<span style='color:#64748B;float:right'>{t['latency_ms']:.1f} ms</span></div>",
                    unsafe_allow_html=True)


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
    st.markdown(f"""<div class="kpis">
      <div class="kpi kpi-red"><div class="l">Open denials</div><div class="v">{len(open_claims)}</div><div class="s">of {len(claims)} in queue</div></div>
      <div class="kpi kpi-cyan"><div class="l">Allowed $ at issue</div><div class="v">${allowed_open:,.0f}</div><div class="s">contracted, not billed</div></div>
      <div class="kpi kpi-purple"><div class="l">LLM calls this session</div><div class="v">{es['calls']}</div><div class="s">triage runs before any LLM</div></div>
      <div class="kpi kpi-green"><div class="l">Session runs closed</div><div class="v">{len(ss.ledger.rows)}</div><div class="s">approved or rejected by a human</div></div>
    </div>""", unsafe_allow_html=True)

    col_q, col_w = st.columns([1.05, 1.95])

    with col_q:
        st.markdown("#### Denial queue")
        qdf = pd.DataFrame([{
            "Claim": c.claim_id, "Payer": c.payer_name, "CDT": c.primary_denied_proc.proc_code if c.primary_denied_proc else "-",
            "Adj": c.denial_code, "Allowed $": c.allowed_at_issue, "Status": c.status} for c in claims])
        st.dataframe(qdf, hide_index=True, use_container_width=True, height=290)
        labels = {f"{c.claim_id} · {c.payer_name} · {c.denial_code}": c.claim_id for c in claims}
        claim_id = labels[st.selectbox("Select claim", list(labels))]
        claim = ss.od.get_claim(claim_id)
        chart = ss.od.get_clinical_chart(claim_id)
        st.markdown(f"""<div class="card">
          <b>Claim {claim.claim_id}</b> — {claim.patient_name} · {claim.clinic_name}<br/>
          {claim.payer_name} · plan: <code>{claim.plan_type}</code> / <code>{claim.plan_funding}</code><br/>
          DOS {claim.date_of_service} · billed ${claim.billed_fee:,.2f} · <b>allowed at issue ${claim.allowed_at_issue:,.2f}</b><br/>
          <span style="color:#F87171">{claim.denial_code}: {claim.denial_description}</span></div>""",
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"CDT": p.proc_code, "Tooth": p.tooth_num, "Billed": p.fee_billed,
                                    "Allowed": p.expected_allowed, "Paid": p.paid_amount,
                                    "Adj": p.adjustment_code or "", "Payer remark": p.payer_remark}
                                   for p in claim.procs]), hide_index=True, use_container_width=True)
        with st.expander("Clinical note (source of truth for every quote)"):
            if chart:
                st.markdown(f"**{chart.provider_name}** (NPI {chart.provider_npi})")
                st.info(chart.clinical_notes)
                st.caption(f"Radiograph: {chart.radiograph_attached} · Photos: {chart.intraoral_photos_attached}"
                           + (f" · Probing: {chart.probing_depths}" if chart.probing_depths else ""))
            else:
                st.warning("No chart on file.")

    with col_w:
        st.markdown("#### Agent run")
        if st.button("▶ Run denial agent", type="primary", use_container_width=True,
                     disabled=claim.status != "Denied"):
            with st.spinner("Running triage → retrieval → evidence → draft → verify ..."):
                res = agent.start(claim_id)
            ss.threads[claim_id] = res["thread_id"]
            st.rerun()

        tid = ss.threads.get(claim_id)
        if not tid:
            st.info("Run the agent to triage this denial. Closed claims show their Open Dental audit trail below.")
        else:
            s = agent.get_state(tid)
            log_once(tid, s)
            pill_cls, pill_txt = ROUTE_PILL.get(s["route"], ("p-blu", s["route"]))
            st.markdown(f"<div class='card'><span class='pill {pill_cls}'>{pill_txt}</span> "
                        f"&nbsp;{s['triage']['reason']}</div>", unsafe_allow_html=True)
            with st.expander("Execution trace (measured latencies)", expanded=False):
                render_traces(s["traces"])

            if s["route"] == Route.APPEAL:
                ev = s["evidence"]
                st.markdown("##### Evidence vs criteria")
                st.dataframe(pd.DataFrame([{
                    "Criterion": c["text"], "Required": c["required"], "Status": c["status"],
                    "Evidence (verbatim / structured)": " | ".join(c["quotes"] + c["structured"]) or "—"}
                    for c in ev["criteria"]]), hide_index=True, use_container_width=True)
                with st.expander(f"Retrieved policy ({'cache hit' if s.get('retrieval_cache_hit') else 'BM25'})"):
                    for ch in s["policy_chunks"]:
                        st.markdown(f"**{ch['id']}** · score {ch['score']} · _{ch['provenance']}_\n\n> {ch['text']}")

                v = s["verification"]
                u = s["usage"][-1] if s.get("usage") else {}
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Faithfulness (claim-level)", f"{v['faithfulness']:.0%}",
                          f"{v['claims_supported']}/{v['claims_checked']} claims")
                c2.metric("Citation coverage", f"{v['citation_coverage']:.0%}")
                c3.metric("Revisions", s.get("revision_count", 0))
                c4.metric(f"Draft cost ({u.get('source', '-')})", f"${u.get('cost_usd', 0):.4f}",
                          "prefix cache read" if u.get("cache_read_input_tokens") else "prefix cache write")
                if v["violations"]:
                    st.error("Verifier violations:\n" + "\n".join(
                        f"- **{x['type']}** `{x['span']}` — {x['detail']}" for x in v["violations"]))

                edited = st.text_area("Appeal letter (edits are re-verified on approval)", value=s["draft"],
                                      height=380, key=f"letter_{tid}",
                                      disabled=not s["awaiting_review"])
            else:
                wi = s.get("work_item") or {}
                st.markdown("##### Work item (no LLM call)")
                st.markdown(f"<div class='card'>{wi.get('summary', '')}</div>", unsafe_allow_html=True)
                for key, title in [("lines", "Lines"), ("gaps", "Unmet criteria"),
                                   ("contradicting_evidence", "Chart contradicts claim"),
                                   ("actions", "Actions"), ("checklist", "Documentation checklist"),
                                   ("call_script", "Rep call script")]:
                    if wi.get(key):
                        st.markdown(f"**{title}**")
                        st.markdown("\n".join(f"- {x}" for x in wi[key]))
                edited = None

            if s.get("review_error"):
                st.error(s["review_error"])

            if s["awaiting_review"]:
                st.markdown("##### Human review")
                reviewer = st.text_input("Reviewer (required for writeback)", key=f"rev_{tid}",
                                         placeholder="e.g. A. Rivera, CDBS")
                b1, b2 = st.columns(2)
                if b1.button("✅ Approve & write to Open Dental", use_container_width=True, key=f"ap_{tid}"):
                    agent.resume(tid, "approve", reviewer, edited_text=edited)
                    st.rerun()
                if b2.button("✖ Reject (nothing written)", use_container_width=True, key=f"rj_{tid}"):
                    agent.resume(tid, "reject", reviewer or "unknown")
                    st.rerun()
            elif s.get("status") == "COMMITTED":
                st.success(f"Written to Open Dental → **{s['commit_result']['status']}**")
            elif s.get("status") == "REJECTED":
                st.warning("Rejected. Nothing written to Open Dental.")

        if claim.tracking_notes:
            st.markdown("##### Open Dental audit trail")
            for n in claim.tracking_notes:
                st.code(n, language="text")


# =========================================================================
# MODULE 2: Evaluation & Cost
# =========================================================================
elif module == "Evaluation & Cost":
    st.markdown("#### Golden evaluation suite")
    st.caption("Runs the real agent on labelled cases plus fault injection. Nothing here is synthetic telemetry.")
    if st.button("▶ Run eval suite", type="primary"):
        with st.spinner("Evaluating..."):
            ss.eval_report = run_golden_eval(ss.od)
    rep = ss.eval_report
    if rep:
        m = rep.metrics
        k = st.columns(6)
        k[0].metric("Route accuracy", f"{m['route_accuracy']:.0%}")
        k[1].metric("Retrieval precision", f"{m['retrieval_precision']:.0%}")
        k[2].metric("Retrieval recall", f"{m['retrieval_recall']:.0%}")
        k[3].metric("Faithfulness", f"{m['faithfulness']:.0%}")
        k[4].metric("LLM calls / denial", m["llm_calls_per_denial"])
        k[5].metric("Safety checks", m["safety_checks_passed"])
        st.dataframe(rep.cases, hide_index=True, use_container_width=True)
        st.dataframe(rep.checks, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Session run ledger")
    ldf = ss.ledger.df()
    if ldf.empty:
        st.caption("No closed runs yet. Approve or reject a claim in the RCM queue.")
    else:
        st.dataframe(ldf, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Token economics")
    es = ss.economics.summary()
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Session LLM calls", es["calls"], es["source"])
    e2.metric("Session cost", f"${es['cost_usd']:.4f}")
    e3.metric("Saved vs no cache", f"${es['saved_usd']:.4f}", f"{es['saved_pct']}%")
    e4.metric("Break-even", f"{es['break_even_reads']} reads", "pays off on 1st read")
    st.caption(f"Break-even: write 1.25× + N reads × 0.1× < (1+N) × 1.0× ⇒ N > 0.25/0.9 = {break_even_reads()}. "
               f"Static prefix ≈ {estimate_tokens(STATIC_PREFIX):,} tokens (min cacheable 1,024; TTL 5 min).")

    st.markdown("##### Monthly projection (assumptions — edit them)")
    p1, p2, p3 = st.columns(3)
    denials = p1.number_input("Denied lines / month (network)", 100, 100_000, 3000, step=100)
    share_llm = p2.slider("Share needing a letter", 0.05, 1.0,
                          float(rep.metrics["llm_calls_per_denial"]) if rep else 0.45, 0.05)
    read_share = p3.slider("Calls landing within cache TTL", 0.0, 1.0, 0.85, 0.05)
    proj = ss.economics.project_monthly(denials, share_llm * 1.2, estimate_tokens(STATIC_PREFIX), 450, 550, read_share)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("LLM calls", f"{proj['calls']:,}")
    q2.metric("No cache", f"${proj['no_cache_usd']:,.2f}")
    q3.metric("With prefix cache", f"${proj['with_cache_usd']:,.2f}", f"-{proj['saved_pct']}%")
    q4.metric("Cost per denial", f"${proj['cost_per_denial_usd']:.4f}")
    st.caption("1.2 calls per letter assumes ~20% need one revision. Token cost is rounding error next to "
               "biller time and recovered allowed $ — the real ROI metric is overturn rate × allowed at issue.")

    with st.expander("LangGraph topology (Mermaid)"):
        st.code(agent.mermaid(), language="text")


# =========================================================================
# MODULE 3: Staffing (Deputy)
# =========================================================================
elif module == "Staffing (Deputy)":
    st.markdown("#### Staffing alerts (computed from timesheets + tomorrow's demand)")
    st.caption(f"Rules: FLSA weekly OT > {DeputyClient.FLSA_WEEKLY_OT_HOURS:.0f}h for non-exempt staff "
               f"(doctors exempt) · RDH capacity {DeputyClient.RDH_APPTS_PER_DAY} appts/day · "
               f"target {DeputyClient.TARGET_RDA_PER_DDS:.0f} RDA per DDS on heavy restorative days.")
    alerts = ss.deputy.get_staffing_deficit_alerts()
    for a in alerts:
        cls = "p-red" if a.severity == "HIGH" else "p-amb"
        cost = f" · est. OT premium ${a.est_cost_impact_usd:,.2f}" if a.est_cost_impact_usd else ""
        st.markdown(f"<div class='card'><span class='pill {cls}'>{a.severity}</span> "
                    f"<span class='pill p-blu'>{a.alert_type}</span> <b>{a.clinic_name}</b>{cost}<br/>"
                    f"{a.description}<br/><span style='color:#38BDF8'>→ {a.action_required}</span></div>",
                    unsafe_allow_html=True)
    st.markdown("#### Timesheets (week to date + scheduled)")
    st.dataframe(pd.DataFrame(ss.deputy.get_timesheets()), hide_index=True, use_container_width=True)
    if alerts:
        pick = st.selectbox("Draft a shift offer for", [f"{a.clinic_name} · {a.alert_type}" for a in alerts])
        if st.button("Draft Deputy shift offer (dry run)"):
            a = alerts[[f"{x.clinic_name} · {x.alert_type}" for x in alerts].index(pick)]
            st.json(ss.deputy.draft_shift_offer(a))


# =========================================================================
# MODULE 4: Corp Dev (Zoho)
# =========================================================================
elif module == "Corp Dev (Zoho)":
    st.markdown("#### Acquisition pipeline — fit scoring")
    st.caption("Sample market data. Production sources: Census ACS B19013 (median HHI), NPPES taxonomy 1223* "
               "(dentist density), ACS population estimates (growth).")
    deals = ss.zoho.get_deals()
    d_label = {f"{d.deal_name} · {d.location} · {d.stage}": d.id for d in deals}
    deal = ss.zoho.get_deal(d_label[st.selectbox("Target practice", list(d_label))])
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown(f"""<div class="card"><b>{deal.deal_name}</b> — {deal.practice_type}<br/>
          {deal.location} · {deal.operatory_count} operatories · {deal.patient_count:,} patients<br/>
          TTM revenue ${deal.ttm_revenue:,.0f} · adj. EBITDA ${deal.adjusted_ebitda:,.0f}
          ({deal.adjusted_ebitda / deal.ttm_revenue:.0%}) · PPO {deal.payer_mix_ppo_pct:.0f}%</div>""",
                    unsafe_allow_html=True)
        if st.button("Enrich & score", type="primary"):
            ss.zoho.enrich_deal(deal.id)
            st.rerun()
    with c2:
        if deal.enrichment_status == "Enriched":
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Median HHI", f"${deal.median_hhi:,.0f}")
            m2.metric("Dentists / 10k", deal.dentists_per_10k)
            m3.metric("5-yr growth", f"{deal.population_growth_5yr}%")
            m4.metric("Fit score", f"{deal.fit_score}/10")
            bd = deal.score_breakdown
            st.dataframe(pd.DataFrame([{"Component": k, "Score (0-10)": bd[k], "Weight": w,
                                        "Contribution": round(bd[k] * w, 2)}
                                       for k, w in ZohoClient.SCORE_WEIGHTS.items()]),
                         hide_index=True, use_container_width=True)
            st.caption(deal.enrichment_notes)
            with st.expander("Zoho CRM writeback payload (dry run)"):
                st.json(ss.zoho.writeback_payload(deal))
        else:
            st.info("Run enrichment to score this target.")

st.markdown("---")
with st.expander("Production design notes (security, PHI, integration)"):
    st.markdown("""
- **PHI minimisation:** the LLM sees `[PATIENT]` / `[PATIENT_ID]` placeholders; names are restored after drafting.
  Production LLM traffic goes through a BAA-covered endpoint (Anthropic API with BAA/ZDR, or Bedrock).
- **No PHI in caches:** only policy retrieval results are cached (keyed payer × CDT × CARC), guarded by a PHI check.
  Finished letters are never cached or reused across patients.
- **Human in the loop:** LangGraph `interrupt()` + checkpointer. Writeback requires a named reviewer, and edited
  letters are re-verified before anything reaches Open Dental.
- **Integrations here are simulated.** Production: Open Dental API (claims, claimprocs, procedure notes, claim
  tracking), Deputy roster/timesheet API, Zoho CRM v8 Deals.
""")
