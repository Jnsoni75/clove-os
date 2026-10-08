"""
tests/test_clove_os.py - Behavioural tests. Each test encodes a property a payer, auditor
or RCM lead would care about, not just "returns something".
"""

import re
import unittest

from agents.llm import STATIC_PREFIX, Deidentifier, TemplateDrafter, build_dynamic_prompt
from agents.rcm_supervisor import RCMDenialAgent
from agents.retrieval import EvidenceExtractor, PolicyRetriever, split_sentences
from llm_cache.token_optimizer import (
    PHILeakError, PolicyContextCache, PrefixCacheSimulator, PromptCacheEconomics, break_even_reads, estimate_tokens,
)
from eval.grounding import GroundingVerifier
from eval.observability import run_golden_eval
from integrations.mock_apis import DeputyClient, OpenDentalClient, ZohoClient
from knowledge.rcm_reference import Route, parse_adjustment_code


def agent_and_od():
    od = OpenDentalClient()
    return RCMDenialAgent(od, drafter=TemplateDrafter()), od


class TestTriage(unittest.TestCase):
    def setUp(self):
        self.agent, self.od = agent_and_od()

    def test_routes(self):
        expected = {90412: Route.APPEAL, 90415: Route.RESUBMIT, 90422: Route.APPEAL, 90428: Route.APPEAL,
                    90431: Route.DOC_GAP, 90437: Route.REP_CALL, 90440: Route.NO_APPEAL}
        for cid, route in expected.items():
            self.assertEqual(self.agent.start(cid)["route"], route, cid)

    def test_co16_is_never_appealed_and_uses_no_llm(self):
        s = self.agent.start(90415)
        self.assertIsNone(s.get("draft"))
        self.assertEqual(s.get("usage"), [])
        self.assertTrue(s["work_item"]["checklist"])

    def test_contradicting_chart_blocks_appeal(self):
        s = self.agent.start(90431)
        self.assertIsNone(s.get("draft"))
        self.assertTrue(any("undercut" in q for q in s["work_item"]["contradicting_evidence"]))

    def test_parse_adjustment_code(self):
        self.assertEqual(parse_adjustment_code("co-97"), ("CO", "97"))
        with self.assertRaises(ValueError):
            parse_adjustment_code("XX-97")


class TestGroundedDrafting(unittest.TestCase):
    def setUp(self):
        self.agent, self.od = agent_and_od()

    def test_every_quote_is_verbatim_from_chart(self):
        for cid in (90412, 90422, 90428):
            s = self.agent.start(cid)
            notes = re.sub(r"\s+", " ", self.od.get_clinical_chart(cid).clinical_notes).lower()
            for q in re.findall(r"\"([^\"]{12,})\"", s["draft"]):
                self.assertIn(re.sub(r"\s+", " ", q).lower().rstrip("."), notes, f"{cid}: {q}")

    def test_letter_requests_allowed_not_billed(self):
        s = self.agent.start(90412)
        claim = self.od.get_claim(90412)
        self.assertIn(f"${claim.allowed_at_issue:,.2f}", s["draft"])
        self.assertNotIn(f"${claim.billed_fee:,.2f}", s["draft"])

    def test_regulatory_language_matches_plan_type(self):
        erisa = self.agent.start(90412)["draft"]      # erisa_employer_group
        non_erisa = self.agent.start(90428)["draft"]  # non_erisa
        self.assertIn("2560.503-1", erisa)
        self.assertNotIn("2560.503-1", non_erisa)
        for d in (erisa, non_erisa):
            self.assertNotIn("prompt pay", d.lower())

    def test_all_appeals_pass_verifier(self):
        for cid in (90412, 90422, 90428):
            v = self.agent.start(cid)["verification"]
            self.assertTrue(v["passed"], (cid, v["violations"]))
            self.assertEqual(v["faithfulness"], 1.0)


class TestVerifier(unittest.TestCase):
    def setUp(self):
        self.od = OpenDentalClient()
        self.claim, self.chart = self.od.get_claim(90412), self.od.get_clinical_chart(90412)
        self.v = GroundingVerifier()

    def _types(self, text):
        return {x["type"] for x in self.v.verify(text, self.claim, self.chart)["violations"]}

    def test_catches_each_failure_mode(self):
        self.assertIn("FABRICATED_QUOTE", self._types('Notes: "The tooth was completely non-restorable today."'))
        self.assertIn("UNSUPPORTED_MEASUREMENT", self._types("Coronal loss was 80%."))
        self.assertIn("UNSUPPORTED_TOOTH", self._types("Tooth #4 was treated."))
        self.assertIn("BILLED_FEE_DEMAND", self._types("Remit $385.00."))
        self.assertIn("MISAPPLIED_STATUTE", self._types("Per the Texas Prompt Pay Act."))
        self.assertIn("UNVERIFIABLE_CITATION", self._types("Per Delta Section 4B."))
        self.assertIn("UNRESOLVED_PLACEHOLDER", self._types("Patient [PATIENT] presented."))

    def test_erisa_citation_rejected_for_non_erisa_plan(self):
        c = self.od.get_claim(90428)
        v = self.v.verify("Per 29 C.F.R. 2560.503-1(g).", c, self.od.get_clinical_chart(90428))
        self.assertIn("MISAPPLIED_STATUTE", {x["type"] for x in v["violations"]})

    def test_clean_text_passes(self):
        self.assertEqual(self._types('Chart: "After excavation, approximately 65% of the clinical crown was lost."'),
                         set())


class TestHITL(unittest.TestCase):
    def setUp(self):
        self.agent, self.od = agent_and_od()

    def test_pauses_then_commits_on_approval(self):
        s = self.agent.start(90412)
        self.assertTrue(s["awaiting_review"])
        self.assertEqual(self.od.get_claim(90412).status, "Denied")
        r = self.agent.resume(s["thread_id"], "approve", "A. Rivera")
        self.assertEqual(r["status"], "COMMITTED")
        self.assertEqual(self.od.get_claim(90412).status, "Appeal Pending Submission")
        self.assertIn("A. Rivera", self.od.get_claim(90412).tracking_notes[-1])

    def test_tampered_edit_is_blocked(self):
        s = self.agent.start(90412)
        r = self.agent.resume(s["thread_id"], "approve", "A. Rivera",
                              edited_text=s["draft"] + '\n"Patient had unbearable pain for a year."')
        self.assertTrue(r["awaiting_review"])
        self.assertIn("FABRICATED_QUOTE", r["review_error"])
        self.assertEqual(self.od.get_claim(90412).status, "Denied")

    def test_reject_and_missing_reviewer(self):
        s = self.agent.start(90412)
        r = self.agent.resume(s["thread_id"], "approve", "  ")
        self.assertTrue(r["awaiting_review"])
        r = self.agent.resume(s["thread_id"], "reject", "A. Rivera")
        self.assertEqual(r["status"], "REJECTED")
        self.assertEqual(self.od.get_claim(90412).tracking_notes, [])

    def test_non_appeal_route_commits_work_item(self):
        s = self.agent.start(90415)
        r = self.agent.resume(s["thread_id"], "approve", "A. Rivera")
        self.assertEqual(self.od.get_claim(90415).status, "Corrected Claim Pending")
        self.assertEqual(r["status"], "COMMITTED")


class TestRetrievalAndPHI(unittest.TestCase):
    def test_bm25_returns_relevant_chunk(self):
        hits = PolicyRetriever().search("buildup included in crown same day bundled", cdt="D2950")
        self.assertEqual(hits[0].id, "GEN-D2950-03")

    def test_sentence_split_keeps_decimals(self):
        self.assertEqual(len(split_sentences("Cusp fracture 2.5mm deep. Tooth #19 restored.")), 2)

    def test_perio_pockets_need_numeric_depths(self):
        ev = EvidenceExtractor().extract("D4341", OpenDentalClient().get_clinical_chart(90415))
        pockets = next(c for c in ev["criteria"] if c["id"] == "D4341-POCKETS")
        self.assertEqual(pockets["status"], "MET")

    def test_cache_refuses_phi(self):
        with self.assertRaises(PHILeakError):
            PolicyContextCache().put(("A", "B", "C"), [{"text": "Eleanor Vance letter"}], ["Eleanor Vance"])

    def test_cache_hit_across_patients_contains_no_phi(self):
        agent, od = agent_and_od()
        agent.start(90412)
        s = agent.start(90431)            # same payer / CDT / CARC, different patient
        self.assertTrue(s["retrieval_cache_hit"])
        self.assertNotIn("Eleanor", str(s["policy_chunks"]))

    def test_llm_prompt_is_deidentified(self):
        od = OpenDentalClient()
        claim = od.get_claim(90412)
        ev = EvidenceExtractor().extract("D2950", od.get_clinical_chart(90412))
        prompt = build_dynamic_prompt({"claim": claim, "chart": od.get_clinical_chart(90412), "evidence": ev,
                                       "policy_chunks": []}, Deidentifier(claim))
        self.assertNotIn(claim.patient_name, prompt)
        self.assertNotIn(str(claim.patient_id), prompt)


class TestEconomics(unittest.TestCase):
    def test_break_even_is_first_read(self):
        self.assertAlmostEqual(break_even_reads(), 0.278, places=3)

    def test_cost_from_usage(self):
        e = PromptCacheEconomics()
        row = e.record({"input_tokens": 400, "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 2000, "output_tokens": 500}, "measured")
        self.assertAlmostEqual(row["cost_usd"], (400 * 3 + 2000 * 0.3 + 500 * 15) / 1e6)
        self.assertAlmostEqual(row["no_cache_cost_usd"], (2400 * 3 + 500 * 15) / 1e6)

    def test_simulator_ttl_and_min_length(self):
        clock = [0.0]
        sim = PrefixCacheSimulator(clock=lambda: clock[0])
        self.assertGreater(sim.usage_for(STATIC_PREFIX, "x", "y")["cache_creation_input_tokens"], 0)
        clock[0] = 100
        self.assertGreater(sim.usage_for(STATIC_PREFIX, "x", "y")["cache_read_input_tokens"], 0)
        clock[0] = 1000                    # TTL expired
        self.assertGreater(sim.usage_for(STATIC_PREFIX, "x", "y")["cache_creation_input_tokens"], 0)
        short = sim.usage_for("too short to cache", "x", "y")
        self.assertEqual(short["cache_read_input_tokens"] + short["cache_creation_input_tokens"], 0)

    def test_static_prefix_is_cacheable(self):
        self.assertGreaterEqual(estimate_tokens(STATIC_PREFIX), 1024)


class TestOpsIntegrations(unittest.TestCase):
    def test_overtime_excludes_exempt_doctors(self):
        alerts = DeputyClient().get_staffing_deficit_alerts()
        ot = [a for a in alerts if a.alert_type == "OVERTIME_RISK"]
        self.assertTrue(ot)
        self.assertFalse(any("Dr." in a.description for a in ot))
        self.assertTrue(all(a.est_cost_impact_usd > 0 for a in ot))

    def test_zoho_score_is_deterministic_and_bounded(self):
        z1, z2 = ZohoClient(), ZohoClient()
        a, b = z1.enrich_deal("ZH-8819"), z2.enrich_deal("ZH-8819")
        self.assertEqual(a.fit_score, b.fit_score)
        self.assertTrue(0 <= a.fit_score <= 10)


class TestGoldenEval(unittest.TestCase):
    def test_suite(self):
        rep = run_golden_eval()
        self.assertEqual(rep.metrics["route_accuracy"], 1.0)
        self.assertEqual(rep.metrics["faithfulness"], 1.0)
        self.assertTrue(rep.checks["passed"].all(), rep.checks.to_string())


if __name__ == "__main__":
    unittest.main()
