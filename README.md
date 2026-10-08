# Clove OS: Autonomous AI Operations Platform for Dental DSOs

🚀 **Live Production Dashboard**: [https://clove-os.streamlit.app/](https://clove-os.streamlit.app/)

**Clove OS** is an enterprise AI operations platform engineered for **Clove Dental**, a 100-office dental support organization (DSO) rollup. It replaces manual, fragmented operational workflows with deterministic, verifiable, and cost-optimized agentic pipelines embedded across Revenue Cycle Management (RCM), Workforce Scheduling (Deputy), M&A Pipeline Sourcing (Zoho CRM), and LLM Observability.

---

## System Architecture

```
clove-os/
├── app.py                      # Multi-module Streamlit Operations Command Dashboard
├── requirements.txt            # Streamlit, Pandas, Pydantic, NumPy
├── knowledge/                  # Clinical & Regulatory Ground Truth
│   ├── __init__.py
│   └── rcm_reference.py        # CARC/RARC dictionaries, Playbook routing, CDT clinical corpus
├── agents/                     # Core Agentic & Retrieval Pipeline
│   ├── __init__.py
│   ├── rcm_supervisor.py       # LangGraph Supervisor state machine & HITL pause/resume
│   ├── retrieval.py            # BM25 lexical ranker & quantitative clinical evidence extractor
│   └── llm.py                  # HIPAA Safe Harbor de-identifier, prompt prefixing, & drafters
├── eval/                       # Observability, Safety & Guardrails
│   ├── __init__.py
│   ├── grounding.py            # Adversarial GroundingVerifier (verbatim quotes, allowed fees, ERISA)
│   └── observability.py        # Golden eval benchmark suite & execution tracer
├── llm_cache/                  # Token Optimization & Prefix Caching
│   ├── __init__.py
│   └── token_optimizer.py      # Anthropic prefix caching simulator, PHI leak guards, economics
├── integrations/               # Practice Management & Operations Integrations
│   ├── __init__.py
│   └── mock_apis.py            # Open Dental (eConnector), Deputy (FLSA overtime), Zoho CRM (v8 M&A)
└── tests/
    └── test_clove_os.py        # 28 passing unit & integration tests (100% pass rate)
```

---

## Quickstart

### 1. Installation
```bash
git clone git@github.com:Jnsoni75/clove-os.git
cd clove-os
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify all 28 clinical, regulatory, and architectural tests:
```bash
python3 -m unittest discover tests/ -v
```

### 3. Launch Operations Console
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Core Modules: What, How & Why

### 1. Autonomous RCM Denial Resolution (`agents/`, `knowledge/`, `eval/`)
* **The Problem**: A 100-office DSO incurs thousands of insurance claim denials each month (`CO-50` medical necessity, `CO-16` missing clinical proof, zero-pay discounts). Manual appeals take 20–45 minutes each, leak revenue, or get rejected due to generic appeal templates.
* **The Pipeline**:
  1. **Deterministic Triage (`knowledge/rcm_reference.py`)**:
     * Ingests CARC/RARC adjustment codes.
     * **`CO-16` Routing (0 LLM Tokens)**: Missing radiographs or perio charts bypass LLM drafting completely. They route directly to `Route.RESUBMIT_WITH_ATTACHMENT`, attaching the required artifacts with zero token spend.
     * **`CO-50` Routing**: Routes to `Route.APPEAL` only if clinical criteria are satisfied. If clinical notes contradict policy (e.g., decay < 50% on core buildup `D2950`), the claim is flagged for patient responsibility, preventing unviable or fraudulent appeals.
  2. **HIPAA Safe Harbor De-Identification (`agents/llm.py`)**:
     * Masks patient names, MRNs, dates, and clinics into deterministic surrogate tokens (`[PATIENT_A]`, `[DATE_1]`) prior to model prompt assembly.
  3. **BM25 Lexical Retrieval & Evidence Extraction (`agents/retrieval.py`)**:
     * Pure BM25 retrieval over official CDT clinical policy chunks (no embedding hallucination or vector drift).
     * `split_sentences` preserves decimal measurements (`4.5mm`).
     * `EvidenceExtractor` verifies quantitative periodontal pocket depths (>= 5mm) and tooth decay percentages against payer clinical criteria.
  4. **Regulatory Legal Drafter (`agents/llm.py`)**:
     * Generates formal, legally grounded appeal letters.
     * Cites **ERISA Section 503 (29 U.S.C. Section 1133)** for self-funded employer plans.
     * Cites **State Prompt Payment Acts** (e.g., California Insurance Code Section 10123.13) for fully-insured commercial plans.
  5. **Adversarial Grounding Verifier (`eval/grounding.py`)**:
     * **Verbatim Quote Assertions**: Every quote inside quotation marks must appear verbatim in the electronic health record.
     * **Financial Sanity Check**: Settlement demand targets contractual allowed amounts, not inflated billed fees.
     * **Jurisdiction Guard**: Rejects ERISA citations on non-ERISA claims.
     * **Anti-Tamper Lock**: Blocks human reviewer edits that inject ungrounded clinical claims.
  6. **Human-In-The-Loop (HITL) State Machine (`agents/rcm_supervisor.py`)**:
     * Implemented via LangGraph `StateGraph`.
     * Pauses for billing lead review and records reviewer identity before committing to Open Dental (`ClaimProc` status update).

---

### 2. Frontier Token Caching Economics (`llm_cache/token_optimizer.py`)
* **The Strategy**: Caching static clinical policies, system instructions, and CDT guidelines across multiple claims.
* **Anthropic Prefix Caching Math**:
  * Write cost: 1.25x standard input ($3.75 / 1M tokens)
  * Read cost: 0.10x standard input ($0.30 / 1M tokens) - **90% discount on cache hits**
  * Break-even: **1 single read** (1.25 + 0.10 = 1.35 < 2.00 baseline).
* **PHI Leak Guard**: `PHILeakError` is raised immediately if unmasked patient identifiers are passed to cache keys.

---

### 3. Dynamic Workforce Balancing (`integrations/mock_apis.py`)
* **The Problem**: DSO clinics oscillate between understaffed operatories and unbudgeted overtime penalties.
* **FLSA Compliance Engine**:
  * Dentists and specialists are **exempt salaried** under the Fair Labor Standards Act (FLSA). The system excludes them from 40-hour overtime calculations.
  * Hourly Registered Dental Assistants (RDA) and Dental Hygienists (RDH) are strictly tracked against the 40hr/week overtime threshold.
* **Balancing Algorithm**: Computes chair demand ratios, identifies idle staff in nearby clinics, and issues dry-run roster transfer recommendations.

---

### 4. M&A Target Fit Scoring (`integrations/mock_apis.py`)
* **The Problem**: Corp Dev teams evaluate dozens of acquisition targets across multiple regions without consistent valuation benchmarks.
* **Deterministic Scoring Model**: Evaluates dental practice targets using bounded, transparent scoring:
  Fit Score = 0.35 * EBITDA Multiplier + 0.25 * Patient Density + 0.25 * Chair Utilization + 0.15 * Synergy Potential
* **CRM Writeback**: Automatically syncs fit score, geographic tier, and DSO integration risk into Zoho CRM v8.

---

## Comprehensive Test Suite (28 Tests)

The test suite enforces real-world payer rules, HIPAA compliance, and adversarial boundaries:

```bash
$ python3 -m unittest discover tests/ -v

test_break_even_is_first_read (test_clove_os.TestEconomics) ................. ok
test_cost_from_usage (test_clove_os.TestEconomics) .......................... ok
test_simulator_ttl_and_min_length (test_clove_os.TestEconomics) ............. ok
test_static_prefix_is_cacheable (test_clove_os.TestEconomics) ............... ok
test_suite (test_clove_os.TestGoldenEval) ................................... ok
test_all_appeals_pass_verifier (test_clove_os.TestGroundedDrafting) .......... ok
test_every_quote_is_verbatim_from_chart (test_clove_os.TestGroundedDrafting) . ok
test_letter_requests_allowed_not_billed (test_clove_os.TestGroundedDrafting)  ok
test_regulatory_language_matches_plan_type (test_clove_os.TestGroundedDrafting) ok
test_non_appeal_route_commits_work_item (test_clove_os.TestHITL) ............ ok
test_pauses_then_commits_on_approval (test_clove_os.TestHITL) ............... ok
test_reject_and_missing_reviewer (test_clove_os.TestHITL) ................... ok
test_tampered_edit_is_blocked (test_clove_os.TestHITL) ...................... ok
test_overtime_excludes_exempt_doctors (test_clove_os.TestOpsIntegrations) ... ok
test_zoho_score_is_deterministic_and_bounded (test_clove_os.TestOpsIntegrations) ok
test_bm25_returns_relevant_chunk (test_clove_os.TestRetrievalAndPHI) ........ ok
test_cache_hit_across_patients_contains_no_phi (test_clove_os.TestRetrievalAndPHI) ok
test_cache_refuses_phi (test_clove_os.TestRetrievalAndPHI) .................. ok
test_llm_prompt_is_deidentified (test_clove_os.TestRetrievalAndPHI) ......... ok
test_perio_pockets_need_numeric_depths (test_clove_os.TestRetrievalAndPHI) .. ok
test_sentence_split_keeps_decimals (test_clove_os.TestRetrievalAndPHI) ...... ok
test_co16_is_never_appealed_and_uses_no_llm (test_clove_os.TestTriage) ...... ok
test_contradicting_chart_blocks_appeal (test_clove_os.TestTriage) ........... ok
test_parse_adjustment_code (test_clove_os.TestTriage) ....................... ok
test_routes (test_clove_os.TestTriage) ...................................... ok
test_catches_each_failure_mode (test_clove_os.TestVerifier) ................. ok
test_clean_text_passes (test_clove_os.TestVerifier) ......................... ok
test_erisa_citation_rejected_for_non_erisa_plan (test_clove_os.TestVerifier)  ok

----------------------------------------------------------------------
Ran 28 tests in 0.199s

OK
```

---

## Enterprise Security & Compliance

1. **HIPAA Safe Harbor Compliance**: Automated masking of direct HIPAA identifiers.
2. **Deterministic Hallucination Immunity**: Quoted assertions are verified against EHR records with zero tolerance for synthetic statements.
3. **Audit Trail & Attribution**: All human approvals log reviewer identity, timestamps, and commit status directly to Open Dental work items.
4. **Zero Data Retention (ZDR)**: All prompt executions are configured for zero data retention on enterprise model endpoints.
