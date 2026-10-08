"""
cache/token_optimizer.py - Cost controls for the RCM agent.

Two layers, each placed where caching is actually safe:

1. PolicyContextCache (application layer)
   Caches *retrieval results* keyed on (payer_id, CDT, CARC). Values come only from the
   policy corpus, so they are PHI-free by construction - and `put` enforces that with a
   guard. We deliberately do NOT cache finished appeal letters: letters are patient-specific,
   and serving one patient's letter for another's denial is both wrong and a HIPAA breach.

2. Anthropic prompt caching (provider layer)
   The static system prefix (instructions + policy corpus) is byte-identical across claims,
   so it is marked with cache_control. PromptCacheEconomics prices real `usage` objects from
   the API. In offline mode, PrefixCacheSimulator decides write vs read from the 5-minute
   TTL and the minimum cacheable length, and costs are labelled "estimated".

Pricing (Claude Sonnet class, per 1M tokens): input $3.00, output $15.00,
5-minute cache write $3.75 (1.25x), cache read $0.30 (0.1x). Verify against current
Anthropic pricing before quoting externally.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ==========================================
# Pricing
# ==========================================

@dataclass(frozen=True)
class ModelPricing:
    input_per_m: float = 3.00
    output_per_m: float = 15.00
    cache_write_per_m: float = 3.75     # 5-minute TTL write (1.25x)
    cache_read_per_m: float = 0.30      # 0.1x
    min_cacheable_tokens: int = 1024    # prefixes shorter than this are not cached
    cache_ttl_seconds: int = 300


SONNET_PRICING = ModelPricing()


def estimate_tokens(text: str) -> int:
    """~4 chars/token heuristic. Only used in offline mode; live mode uses API usage."""
    return max(1, len(text) // 4)


def break_even_reads(p: ModelPricing = SONNET_PRICING) -> float:
    """
    Reads after the first write at which caching beats no caching:
        write + N*read  <  (1 + N) * base
        N > (write - base) / (base - read) = (1.25 - 1) / (1 - 0.1) = 0.28
    i.e. caching pays for itself on the FIRST cache read.
    """
    base, w, r = p.input_per_m, p.cache_write_per_m, p.cache_read_per_m
    return round((w - base) / (base - r), 3)


# ==========================================
# Layer 1: PHI-safe retrieval cache
# ==========================================

class PHILeakError(RuntimeError):
    pass


class PolicyContextCache:
    def __init__(self):
        self._store: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(payer_id: str, cdt: str, carc: str) -> Tuple[str, str, str]:
        return (payer_id.upper(), cdt.upper(), carc)

    def get(self, key: Tuple[str, str, str]) -> Optional[List[Dict[str, Any]]]:
        val = self._store.get(key)
        if val is None:
            self.misses += 1
            return None
        self.hits += 1
        return val

    def put(self, key: Tuple[str, str, str], value: List[Dict[str, Any]], forbidden_terms: List[str]):
        blob = json.dumps(value).lower()
        leaked = [t for t in forbidden_terms if t and str(t).lower() in blob]
        if leaked:
            raise PHILeakError(f"Refusing to cache value containing patient identifiers: {leaked}")
        self._store[key] = value

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(self.hits / total * 100, 1) if total else 0.0,
        }


# ==========================================
# Layer 2: prompt-cache economics
# ==========================================

def _cost(usage: Dict[str, int], p: ModelPricing) -> Tuple[float, float]:
    """Returns (actual_cost, no_cache_counterfactual) for one Anthropic usage object."""
    inp = usage.get("input_tokens", 0)
    cw = usage.get("cache_creation_input_tokens", 0)
    cr = usage.get("cache_read_input_tokens", 0)
    out = usage.get("output_tokens", 0)
    actual = (inp * p.input_per_m + cw * p.cache_write_per_m + cr * p.cache_read_per_m
              + out * p.output_per_m) / 1e6
    counterfactual = ((inp + cw + cr) * p.input_per_m + out * p.output_per_m) / 1e6
    return actual, counterfactual


class PromptCacheEconomics:
    def __init__(self, pricing: ModelPricing = SONNET_PRICING):
        self.p = pricing
        self.calls: List[Dict[str, Any]] = []

    def record(self, usage: Dict[str, int], source: str) -> Dict[str, Any]:
        """source: 'measured' (real API usage) or 'estimated' (offline simulation)."""
        actual, counterfactual = _cost(usage, self.p)
        row = {**usage, "source": source, "cost_usd": actual, "no_cache_cost_usd": counterfactual,
               "saved_usd": counterfactual - actual}
        self.calls.append(row)
        return row

    def summary(self) -> Dict[str, Any]:
        n = len(self.calls)
        tot = lambda k: sum(c.get(k, 0) for c in self.calls)  # noqa: E731
        cost, base = tot("cost_usd"), tot("no_cache_cost_usd")
        sources = {c["source"] for c in self.calls}
        return {
            "calls": n,
            "source": "measured" if sources == {"measured"} else ("estimated" if sources else "n/a"),
            "cache_reads": sum(1 for c in self.calls if c.get("cache_read_input_tokens", 0) > 0),
            "cache_writes": sum(1 for c in self.calls if c.get("cache_creation_input_tokens", 0) > 0),
            "input_tokens": tot("input_tokens"),
            "cache_read_tokens": tot("cache_read_input_tokens"),
            "cache_write_tokens": tot("cache_creation_input_tokens"),
            "output_tokens": tot("output_tokens"),
            "cost_usd": round(cost, 6),
            "no_cache_cost_usd": round(base, 6),
            "saved_usd": round(base - cost, 6),
            "saved_pct": round((base - cost) / base * 100, 1) if base else 0.0,
            "break_even_reads": break_even_reads(self.p),
        }

    def project_monthly(self, denials_per_month: int, llm_calls_per_denial: float, prefix_tokens: int,
                        dynamic_tokens: int, output_tokens: int, cache_read_share: float) -> Dict[str, float]:
        """
        Scenario model (inputs are assumptions, shown in the UI). cache_read_share is the share
        of calls that land within the TTL of a prior call with the same prefix - in a centralized
        RCM team working a queue in batches this is high; for sporadic traffic it is low.
        """
        calls = denials_per_month * llm_calls_per_denial
        cacheable = prefix_tokens >= self.p.min_cacheable_tokens
        share = cache_read_share if cacheable else 0.0
        reads, writes = calls * share, calls * (1 - share)
        per_m = 1e6
        base = calls * ((prefix_tokens + dynamic_tokens) * self.p.input_per_m
                        + output_tokens * self.p.output_per_m) / per_m
        if cacheable:
            cached = (writes * prefix_tokens * self.p.cache_write_per_m
                      + reads * prefix_tokens * self.p.cache_read_per_m
                      + calls * dynamic_tokens * self.p.input_per_m
                      + calls * output_tokens * self.p.output_per_m) / per_m
        else:
            cached = base
        return {
            "calls": round(calls),
            "prefix_cacheable": cacheable,
            "no_cache_usd": round(base, 2),
            "with_cache_usd": round(cached, 2),
            "saved_usd": round(base - cached, 2),
            "saved_pct": round((base - cached) / base * 100, 1) if base else 0.0,
            "cost_per_denial_usd": round(cached / denials_per_month, 4) if denials_per_month else 0.0,
        }


class PrefixCacheSimulator:
    """Offline stand-in for Anthropic's prefix cache: TTL refresh on hit, min length rule."""

    def __init__(self, pricing: ModelPricing = SONNET_PRICING, clock=time.monotonic):
        self.p = pricing
        self._clock = clock
        self._last_used: Dict[str, float] = {}

    def usage_for(self, prefix: str, dynamic: str, output: str) -> Dict[str, int]:
        pt, dt, ot = estimate_tokens(prefix), estimate_tokens(dynamic), estimate_tokens(output)
        if pt < self.p.min_cacheable_tokens:
            return {"input_tokens": pt + dt, "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0, "output_tokens": ot}
        h = hashlib.sha256(prefix.encode()).hexdigest()
        now = self._clock()
        last = self._last_used.get(h)
        self._last_used[h] = now
        if last is not None and now - last <= self.p.cache_ttl_seconds:
            return {"input_tokens": dt, "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": pt, "output_tokens": ot}
        return {"input_tokens": dt, "cache_creation_input_tokens": pt,
                "cache_read_input_tokens": 0, "output_tokens": ot}
