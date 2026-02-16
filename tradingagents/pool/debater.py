"""Pool debater: Stage 3 of the pool screening pipeline.

Uses TradingAgentsGraph to run full bull/bear + risk debates
on top-ranked candidates from Stage 2.
"""

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class PoolDebater:
    """Stage 3: Debate candidates using the multi-agent graph."""

    def __init__(
        self,
        graph,
        store,
        config: dict | None = None,
        max_debates: int = 12,
    ):
        self.graph = graph
        self.store = store
        self.config = config or {}
        self.max_debates = max_debates

    def _rank_for_debate(self, candidates: list[dict]) -> list[dict]:
        """Rank candidates by vol_ratio (desc) for debate priority."""
        return sorted(
            candidates,
            key=lambda c: c.get("vol_ratio", 0) or 0,
            reverse=True,
        )

    def _debate_single(
        self, candidate: dict, run_date: str
    ) -> dict | None:
        """Run a full debate for a single candidate.

        Returns enriched candidate with signal/summary, or None on error.
        """
        code = candidate["code"]
        code_name = candidate.get("code_name", code)

        try:
            final_state, signal = self.graph.propagate(code_name, run_date)

            return {
                **candidate,
                "signal": signal,
                "debate_summary": (
                    final_state.get("inv_judge_decision", "")[:500]
                ),
                "risk_summary": (
                    final_state.get("risk_judge_decision", "")[:500]
                ),
                "investment_plan": (
                    final_state.get("investment_plan", "")[:500]
                ),
            }
        except Exception:
            logger.warning("Debate failed for %s", code, exc_info=True)
            return None

    def debate_candidates(
        self,
        candidates: list[dict],
        run_date: str,
        on_progress: Callable | None = None,
        on_result: Callable | None = None,
    ) -> list[dict]:
        """Debate top-ranked candidates up to max_debates.

        Args:
            candidates: Enriched candidates from Stage 2.
            run_date: Trading date string.
            on_progress: Optional callback(index, total, candidate).
            on_result: Optional callback(result) called after each successful debate.

        Returns:
            List of debated candidates with signal info.
        """
        if not candidates:
            return []

        ranked = self._rank_for_debate(candidates)
        to_debate = ranked[: self.max_debates]
        results = []

        for i, candidate in enumerate(to_debate):
            if on_progress:
                on_progress(i, len(to_debate), candidate)

            result = self._debate_single(candidate, run_date)
            if result is not None:
                results.append(result)
                if on_result:
                    on_result(result)

        return results
