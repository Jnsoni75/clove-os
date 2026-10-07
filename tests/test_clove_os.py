"""
tests/test_clove_os.py - Comprehensive Unit & Integration Test Suite for Clove OS
"""

import unittest
import numpy as np
from integrations.mock_apis import (
    OpenDentalClient,
    DeputyClient,
    ZohoClient,
    DentalClaim,
    ClinicalChart
)
from cache.token_optimizer import (
    SemanticCache,
    PromptCacheCostCalculator,
    CacheHitResult,
    PromptCacheMetrics
)
from eval.observability import (
    RagasEvaluator,
    LangSmithTracer,
    RagasMetricScore,
    TelemetrySpan
)
from agents.rcm_supervisor import (
    SupervisorNode,
    ClinicalRAGWorker,
    AppealWriterWorker,
    HITLApproval,
    RCMWorkflowState
)


class TestMockApis(unittest.TestCase):
    def setUp(self):
        self.od_client = OpenDentalClient()
        self.deputy_client = DeputyClient()
        self.zoho_client = ZohoClient()

    def test_open_dental_client_claims_and_charts(self):
        claims = self.od_client.get_claims(status="Denied")
        self.assertGreaterEqual(len(claims), 4)

        claim = claims[0]
        self.assertEqual(claim.status, "Denied")
        self.assertIn(claim.denial_code, ["CO-50", "CO-16", "CO-97"])

        procs = self.od_client.get_claim_procs(claim.claim_id)
        self.assertGreaterEqual(len(procs), 1)

        chart = self.od_client.get_clinical_chart(claim.claim_id)
        self.assertIsNotNone(chart)
        self.assertIn(chart.provider_name, ["Dr. Marcus Vance, DDS", "Dr. Sophia Chen, DMD", "Dr. Sarah Jenkins, DDS", "Dr. Aaron Patel, DDS"])

    def test_open_dental_commit_tracking(self):
        claim_id = 90412
        commit_res = self.od_client.post_claim_tracking(
            claim_id=claim_id,
            appeal_text="Formal appeal for D2950 medical necessity.",
            tracking_def_num=104
        )
        self.assertTrue(commit_res["success"])
        self.assertEqual(commit_res["status"], "AI Review Pending")

        updated_claim = self.od_client.get_claim(claim_id)
        self.assertEqual(updated_claim.status, "AI Review Pending")
        self.assertGreater(len(updated_claim.tracking_notes), 0)

    def test_deputy_client(self):
        rosters = self.deputy_client.get_rosters()
        self.assertGreater(len(rosters), 0)

        timesheets = self.deputy_client.get_timesheets()
        self.assertGreater(len(timesheets), 0)
        self.assertTrue(any(t["overtime_hours"] > 0 for t in timesheets))

        alerts = self.deputy_client.get_staffing_deficit_alerts()
        self.assertGreaterEqual(len(alerts), 3)

    def test_zoho_client(self):
        deals = self.zoho_client.get_deals()
        self.assertGreaterEqual(len(deals), 4)

        target = deals[0]
        self.assertIn(target.stage, ["Due Diligence", "Identified", "LOI Signed"])

        enriched = self.zoho_client.post_deal_enrichment(
            deal_id=target.id,
            median_hhi=112000.0,
            dentists_per_10k=4.2,
            population_growth_5yr=14.5,
            dso_synergy_score=94.5,
            notes="Strong dental demographics in Austin MSA."
        )
        self.assertEqual(enriched.enrichment_status, "Enriched")
        self.assertEqual(enriched.stage, "Enriched & Ready")
        self.assertEqual(enriched.dso_synergy_score, 94.5)


class TestSemanticCacheAndEconomics(unittest.TestCase):
    def setUp(self):
        self.cache = SemanticCache(similarity_threshold=0.90)
        self.calc = PromptCacheCostCalculator()

    def test_semantic_cache_hit_and_miss(self):
        # Query that should hit the pre-seeded D2950 cache
        query_exact = "Payer Delta Dental denied CDT D2950 code CO-50 Medical necessity required >50% coronal breakdown"
        res_hit = self.cache.lookup(query_exact)
        self.assertTrue(res_hit.is_hit)
        self.assertGreaterEqual(res_hit.similarity_score, 0.90)
        self.assertLess(res_hit.latency_ms, 5.0)  # Sub-5ms guarantee

        # Random query that should miss
        res_miss = self.cache.lookup("Query completely unrelated to dental orthodontics or endodontics completely obscure")
        self.assertFalse(res_miss.is_hit)
        self.assertLess(res_miss.similarity_score, 0.90)

    def test_prompt_cache_cost_calculator(self):
        # Record cold write
        m1 = self.calc.record_transaction(
            prompt_prefix_tokens=3000,
            dynamic_tokens=200,
            output_tokens=500,
            is_cached_prefix=False
        )
        self.assertGreater(m1["unoptimized_cost_usd"], 0)

        # Record cached read (90% discount)
        m2 = self.calc.record_transaction(
            prompt_prefix_tokens=3000,
            dynamic_tokens=200,
            output_tokens=500,
            is_cached_prefix=True
        )
        self.assertGreater(m2["net_savings_usd"], 0)
        self.assertEqual(m2["break_even_reads"], 1.39)

        metrics = self.calc.get_metrics()
        self.assertEqual(metrics.total_calls, 2)
        self.assertGreater(metrics.net_savings_usd, 0)


class TestRagasEvaluatorAndTelemetry(unittest.TestCase):
    def setUp(self):
        self.evaluator = RagasEvaluator()
        self.tracer = LangSmithTracer()
        self.od_client = OpenDentalClient()

    def test_ragas_evaluation(self):
        claim = self.od_client.get_claim(90412)
        chart = self.od_client.get_clinical_chart(90412)

        appeal = (
            "This appeal demonstrates that tooth #19 presented with 65% coronal decay. "
            "Doctor notes confirm excavation of deep caries with ferrule and subgingival preparation. "
            "Pre-op radiograph attached. In accordance with ADA Prompt Pay and ERISA, remittance is due."
        )

        score = self.evaluator.evaluate(claim, chart, appeal)
        self.assertGreaterEqual(score.faithfulness, 0.88)
        self.assertGreaterEqual(score.context_precision, 0.85)
        self.assertGreaterEqual(score.answer_relevancy, 0.85)
        self.assertFalse(score.hallucination_detected)

    def test_langsmith_tracer_history(self):
        traces = self.tracer.get_traces()
        self.assertEqual(len(traces), 100)

        df = self.tracer.get_telemetry_df()
        self.assertEqual(len(df), 100)
        self.assertIn("Faithfulness", df.columns)
        self.assertIn("Cache Status", df.columns)

        kpis = self.tracer.get_aggregate_kpis()
        self.assertEqual(kpis["total_runs"], 100)
        self.assertGreater(kpis["cache_hit_rate"], 50.0)
        self.assertGreater(kpis["avg_faithfulness"], 90.0)


class TestRCMSupervisorWorkflow(unittest.TestCase):
    def setUp(self):
        self.od_client = OpenDentalClient()
        self.cache = SemanticCache()
        self.calc = PromptCacheCostCalculator()
        self.evaluator = RagasEvaluator()
        self.supervisor = SupervisorNode(self.cache, self.calc, self.evaluator)

    def test_end_to_end_workflow(self):
        claim = self.od_client.get_claim(90412)
        chart = self.od_client.get_clinical_chart(90412)

        # Run multi-agent supervisor
        state = self.supervisor.run(claim, chart)
        self.assertIsNotNone(state.appeal_letter)
        self.assertGreater(len(state.traces), 3)
        self.assertEqual(state.hitl_status, "PENDING_REVIEW")
        self.assertGreaterEqual(state.faithfulness_score, 0.85)

        # Execute HITL approval and commit
        commit_res = HITLApproval.approve_and_commit(state, self.od_client, state.appeal_letter)
        self.assertTrue(commit_res["success"])
        self.assertEqual(state.hitl_status, "COMMITTED")

        # Verify claim in Open Dental
        od_claim = self.od_client.get_claim(90412)
        self.assertEqual(od_claim.status, "AI Review Pending")


if __name__ == "__main__":
    unittest.main()
