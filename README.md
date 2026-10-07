# Clove OS: Autonomous AI Operations Platform for Clove Dental

**Clove OS** is the centralized AI operations platform designed for **Clove Dental**, a 100-office dental DSO rollup. It automates high-volume operational bottlenecks across dental clinics, featuring autonomous Revenue Cycle Management (RCM) claim denial resolution, M&A practice acquisition enrichment, workforce scheduling optimization, and LLM telemetry governance.

---

## 🏛️ System Architecture

```
clove-os/
├── requirements.txt            # streamlit, pandas, pydantic, numpy
├── app.py                      # Production Streamlit Operations Dashboard
├── integrations/
│   ├── __init__.py
│   └── mock_apis.py            # OpenDentalClient, DeputyClient, ZohoClient
├── cache/
│   ├── __init__.py
│   └── token_optimizer.py      # SemanticCache (<5ms, cosine >=0.90) & PromptCacheCostCalculator
├── agents/
│   ├── __init__.py
│   └── rcm_supervisor.py       # LangGraph Supervisor Pattern (Supervisor, ClinicalRAG, AppealWriter, HITL)
├── eval/
│   ├── __init__.py
│   └── observability.py        # RAGAS metrics & LangSmith 100-run Execution Tracer
└── README.md
```

---

## 🚀 Quickstart

1. **Activate Environment & Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Clove OS**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 🧩 Core Modules

### 1. RCM Claim Denial Resolver (`OpenDentalClient`)
- **Open Dental eConnector Simulation**: Ingests denied dental claims (`CO-50` medical necessity, `CO-16` missing clinical proof, `CO-97` bundling) with full patient, clinic, CDT procedure line, and contemporaneous electronic health records.
- **LangGraph Supervisor Multi-Agent Flow**:
  1. `SupervisorNode`: Ingests denial payload, checks `SemanticCache` for fast-path sub-5ms resolution, analyzes denial codes.
  2. `ClinicalRAGWorker`: Matches payer coverage policy manuals (Delta Dental Section 4B, MetLife, Cigna, Guardian) with chart notes (coronal loss >50%, pocket depths $\ge 5\text{mm}$, bone loss).
  3. `AppealWriterWorker`: Formats an evidence-backed, ADA-compliant formal appeal letter citing doctor quotes, radiographs, and Texas/ERISA Prompt Pay statutory rules.
  4. `HITLApproval Barrier`: Enforces Human-in-the-Loop review before committing to Open Dental with status `AI Review Pending` (Tracking Def 104).

### 2. Corp Dev Target Enricher (`ZohoClient`)
- Simulates Zoho CRM v8 API (`GET /crm/v8/Deals`, `POST /crm/v8/Deals`).
- Evaluates Texas dental acquisition targets (EBITDA, patient volume, operatory count).
- Automatically enriches custom CRM fields with median household income (HHI), dentist density per 10k population, 5-year growth, and DSO synergy scores.

### 3. Dynamic Staffing & Scheduling (`DeputyClient`)
- Simulates Deputy Workforce Management endpoints (`GET /rosters`, `GET /timesheets`).
- Analyzes operatory patient load vs. provider capacity (Dentists, RDH Hygienists, RDA Assistants).
- Flags staffing deficit alerts and overtime breaches (>40h/week) with autonomous resolution recommendations.

### 4. Observability & Telemetry (`RAGASEvaluator` & `LangSmithTracer`)
- Computes **RAGAS Faithfulness** (anchoring 100% of claims to clinical doctor notes), **Context Precision**, and **Answer Relevancy**.
- Real-time **Hallucination Guard** asserting zero uncorroborated tooth numbers or conditions.
- Live LangSmith execution tracer across the last 100 benchmark test runs.

---

## 💰 Frontier Token Caching Economics

Models Anthropic / Frontier prefix prompt caching:
- **Base Input Tokens**: $3.00 / 1M tokens ($0.000003 / token)
- **Cached Input Reads**: $0.30 / 1M tokens (90% discount on cache hits)
- **Cache Write Surcharge**: $3.75 / 1M tokens (25% initial surcharge)
- **Output Generation**: $15.00 / 1M tokens
- **Break-Even Threshold**: $1.39$ reads per unique prompt prefix.
- **Semantic Vector Cache**: Cosine similarity $\ge 0.90$ returns pre-verified appeals in $< 2\text{ms}$ at **$0.00 LLM compute cost**.

---

## 🔒 HIPAA Compliance & Security Safeguards

1. **Stateless Processing & Zero Data Retention (ZDR)**: No Protected Health Information (PHI) is persisted on untrusted infrastructure or used for model training.
2. **AWS Bedrock / Azure OpenAI BAA Enclave**: All model calls execute through an enterprise Business Associate Agreement gateway via dedicated PrivateLink VPC endpoints.
3. **Safe Harbor De-Identification**: Direct identifiers (Patient Names, MRNs, DOBs) are scrubbed or tokenized prior to embedding computation.
4. **Deterministic Hallucination Guard**: RAGAS evaluator validates that every procedural tooth number and clinical finding has an exact match in the treating dentist's contemporaneous progress notes.
