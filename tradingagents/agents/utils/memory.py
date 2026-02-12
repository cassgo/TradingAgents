"""Financial situation memory using BM25 for lexical similarity matching.

Uses BM25 (Best Matching 25) algorithm for retrieval - no API calls,
no token limits, works offline with any LLM provider.

Supports ACT-R cognitive decay: memories that are retrieved more often
decay slower ("use it or lose it"). Formula:
  activation = ln(Σ tⱼ^(-d))
  weight = sigmoid(activation)
"""

import math
import re
from datetime import datetime, date
from typing import List, Optional, Tuple, Union

from rank_bm25 import BM25Okapi

_MIN_DAYS = 0.001  # ~86 seconds, avoids division by zero


class FinancialSituationMemory:
    """Memory system for storing and retrieving financial situations using BM25."""

    def __init__(self, name: str, config: dict = None):
        """Initialize the memory system.

        Args:
            name: Name identifier for this memory instance
            config: Configuration dict with optional keys:
                - memory_decay_exponent: ACT-R decay exponent d (default 0.5, 0 disables)
                - memory_max_entries: Max stored memories before pruning (default 500)
        """
        self.name = name
        self.documents: List[str] = []
        self.recommendations: List[str] = []
        self.access_history: List[List[datetime]] = []
        self.bm25 = None

        config = config or {}
        self._decay_exponent: float = config.get("memory_decay_exponent", 0.5)
        self._max_entries: int = config.get("memory_max_entries", 500)

    @property
    def timestamps(self) -> List[datetime]:
        """Backward-compatible: return creation time (first access) for each memory."""
        return [h[0] for h in self.access_history if h]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _parse_date(self, date_input: Union[str, datetime, date]) -> datetime:
        """Parse flexible date input into a datetime object."""
        if isinstance(date_input, datetime):
            return date_input
        if isinstance(date_input, date):
            return datetime(date_input.year, date_input.month, date_input.day)
        if isinstance(date_input, str):
            try:
                return datetime.fromisoformat(date_input)
            except (ValueError, TypeError):
                return datetime.now()
        return datetime.now()

    def _compute_decay(
        self, access_times: List[datetime], reference_date: datetime
    ) -> float:
        """Compute ACT-R activation-based decay weight.

        Formula:
            activation = ln(Σ tⱼ^(-d))
            weight = sigmoid(activation)

        Args:
            access_times: List of datetimes when this memory was accessed.
            reference_date: The current date for comparison.

        Returns:
            Weight in (0, 1). 0.0 if no accesses, 1.0 if decay disabled.
        """
        if not access_times:
            return 0.0

        if self._decay_exponent <= 0:
            return 1.0

        d = self._decay_exponent
        total = 0.0
        for t in access_times:
            days = max((reference_date - t).total_seconds() / 86400.0, _MIN_DAYS)
            total += days ** (-d)

        activation = math.log(total)
        return 1.0 / (1.0 + math.exp(-activation))

    def _rebuild_index(self):
        """Rebuild the BM25 index after adding documents."""
        if self.documents:
            tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None

    def _prune_if_needed(self):
        """Evict oldest memories when over max_entries."""
        if self._max_entries <= 0 or len(self.documents) <= self._max_entries:
            return

        # Sort by most recent access time (latest in access_history)
        indexed = list(range(len(self.documents)))
        indexed.sort(key=lambda i: max(self.access_history[i]), reverse=True)
        keep = set(indexed[: self._max_entries])

        self.documents = [self.documents[i] for i in range(len(self.documents)) if i in keep]
        self.recommendations = [self.recommendations[i] for i in range(len(self.recommendations)) if i in keep]
        self.access_history = [self.access_history[i] for i in range(len(self.access_history)) if i in keep]

        self._rebuild_index()

    def add_situations(
        self,
        situations_and_advice: List[Tuple[str, str]],
        timestamp: Optional[Union[str, datetime, date]] = None,
    ):
        """Add financial situations and their corresponding advice.

        Args:
            situations_and_advice: List of tuples (situation, recommendation)
            timestamp: Optional timestamp for all items in this batch.
                       Defaults to datetime.now() when not provided.
        """
        ts = self._parse_date(timestamp) if timestamp is not None else datetime.now()

        for situation, recommendation in situations_and_advice:
            self.documents.append(situation)
            self.recommendations.append(recommendation)
            self.access_history.append([ts])

        self._rebuild_index()
        self._prune_if_needed()

    def get_memories(
        self,
        current_situation: str,
        n_matches: int = 1,
        current_date: Optional[Union[str, datetime, date]] = None,
    ) -> List[dict]:
        """Find matching recommendations using BM25 similarity with ACT-R decay.

        Args:
            current_situation: The current financial situation to match against
            n_matches: Number of top matches to return
            current_date: Reference date for decay calculation.
                          When None, decay is not applied (backward compatible).

        Returns:
            List of dicts with matched_situation, recommendation, similarity_score,
            and optionally decay_score and combined_score.
        """
        if not self.documents or self.bm25 is None:
            return []

        query_tokens = self._tokenize(current_situation)
        scores = self.bm25.get_scores(query_tokens)

        # Normalize BM25 scores to 0-1 range
        raw = [float(s) for s in scores]
        min_score = min(raw)
        max_score = max(raw)
        score_range = max_score - min_score
        if score_range > 0:
            normalized_scores = [(s - min_score) / score_range for s in raw]
        elif max_score > 0:
            normalized_scores = [s / max_score for s in raw]
        else:
            # All scores identical: 0 if truly no match, 1 if equal non-zero relevance
            normalized_scores = [0.0 if max_score == 0.0 else 1.0 for _ in raw]

        ref_date = self._parse_date(current_date) if current_date is not None else None

        scored_indices: List[Tuple[int, float, float, Optional[float]]] = []
        for idx in range(len(scores)):
            normalized = normalized_scores[idx]
            if ref_date is not None:
                decay = self._compute_decay(self.access_history[idx], ref_date)
                combined = normalized * decay
            else:
                decay = None
                combined = normalized
            scored_indices.append((idx, combined, normalized, decay))

        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top = scored_indices[:n_matches]

        results = []
        for idx, combined, normalized, decay in top:
            entry = {
                "matched_situation": self.documents[idx],
                "recommendation": self.recommendations[idx],
                "similarity_score": normalized,
            }
            if decay is not None:
                entry["decay_score"] = decay
                entry["combined_score"] = combined
                # Record this retrieval access (ACT-R: use strengthens memory)
                self.access_history[idx].append(ref_date)
            results.append(entry)

        return results

    def clear(self):
        """Clear all stored memories."""
        self.documents = []
        self.recommendations = []
        self.access_history = []
        self.bm25 = None


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory("test_memory")

    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    matcher.add_situations(example_data)

    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
