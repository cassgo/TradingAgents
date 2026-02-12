"""Tests for FinancialSituationMemory with ACT-R decay mechanism.

ACT-R model: activation = ln(Σ tⱼ^(-d)), weight = sigmoid(activation)
Memories that are retrieved more often decay slower ("use it or lose it").
"""

import importlib.util
import math
import sys
import pytest
from datetime import datetime, date, timedelta

# Import memory module directly to avoid heavy agent __init__.py chain
_spec = importlib.util.spec_from_file_location(
    "tradingagents.agents.utils.memory",
    "/biv/code/TradingAgents/tradingagents/agents/utils/memory.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
FinancialSituationMemory = _mod.FinancialSituationMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(decay_exponent=0.5, max_entries=500):
    """Create a memory instance with configurable ACT-R settings."""
    config = {
        "memory_decay_exponent": decay_exponent,
        "memory_max_entries": max_entries,
    }
    return FinancialSituationMemory("test", config=config)


def _sigmoid(x):
    """Reference sigmoid for test assertions."""
    return 1.0 / (1.0 + math.exp(-x))


def _expected_actr_weight(days_list, d=0.5):
    """Compute expected ACT-R weight for given access days-ago list."""
    total = sum(t ** (-d) for t in days_list)
    return _sigmoid(math.log(total))


# ===========================================================================
# 1. ACT-R decay computation (_compute_decay)
# ===========================================================================

class TestComputeDecay:
    """Test ACT-R activation: weight = sigmoid(ln(Σ tⱼ^(-d)))."""

    def test_single_access_one_day(self):
        """1 access at 1 day ago → sigmoid(ln(1)) = sigmoid(0) = 0.5."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15)
        access_times = [ref - timedelta(days=1)]
        assert mem._compute_decay(access_times, ref) == pytest.approx(0.5, abs=0.01)

    def test_single_access_thirty_days(self):
        """1 access at 30 days ago → ~0.154."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15)
        access_times = [ref - timedelta(days=30)]
        assert mem._compute_decay(access_times, ref) == pytest.approx(0.154, abs=0.01)

    def test_single_access_sixty_days(self):
        """1 access at 60 days ago → ~0.114."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15)
        access_times = [ref - timedelta(days=60)]
        assert mem._compute_decay(access_times, ref) == pytest.approx(0.114, abs=0.01)

    def test_multiple_accesses_boost_activation(self):
        """3 accesses at 1, 10, 30 days ago → ~0.600, much higher than single at 30."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15)
        access_times = [
            ref - timedelta(days=30),
            ref - timedelta(days=10),
            ref - timedelta(days=1),
        ]
        weight = mem._compute_decay(access_times, ref)
        single_weight = mem._compute_decay([ref - timedelta(days=30)], ref)
        assert weight == pytest.approx(0.600, abs=0.01)
        assert weight > single_weight  # multiple accesses always stronger

    def test_very_recent_access_near_one(self):
        """Access just now → weight close to 1.0."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15, 12, 0, 0)
        access_times = [ref - timedelta(seconds=60)]  # 1 minute ago
        assert mem._compute_decay(access_times, ref) > 0.95

    def test_future_access_time_clamps(self):
        """Access time in the future → treated as just now, high weight."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15)
        access_times = [ref + timedelta(days=5)]
        weight = mem._compute_decay(access_times, ref)
        assert weight > 0.95

    def test_disabled_decay_exponent_zero(self):
        """d=0 disables decay: always returns 1.0."""
        mem = _make_memory(decay_exponent=0)
        ref = datetime(2025, 6, 15)
        access_times = [ref - timedelta(days=365)]
        assert mem._compute_decay(access_times, ref) == pytest.approx(1.0)

    def test_disabled_decay_negative_exponent(self):
        """Negative exponent also disables decay."""
        mem = _make_memory(decay_exponent=-1)
        ref = datetime(2025, 6, 15)
        access_times = [ref - timedelta(days=365)]
        assert mem._compute_decay(access_times, ref) == pytest.approx(1.0)

    def test_empty_access_list_returns_zero(self):
        """No access history → weight = 0."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15)
        assert mem._compute_decay([], ref) == pytest.approx(0.0)

    def test_custom_exponent(self):
        """d=0.3 decays slower than d=0.5."""
        mem_slow = _make_memory(decay_exponent=0.3)
        mem_fast = _make_memory(decay_exponent=0.5)
        ref = datetime(2025, 6, 15)
        access_times = [ref - timedelta(days=30)]
        slow_weight = mem_slow._compute_decay(access_times, ref)
        fast_weight = mem_fast._compute_decay(access_times, ref)
        assert slow_weight > fast_weight  # smaller d = slower decay


# ===========================================================================
# 2. Date parsing (_parse_date) — unchanged
# ===========================================================================

class TestParseDate:
    """Test flexible date input handling."""

    def test_parse_string_date(self):
        mem = _make_memory()
        result = mem._parse_date("2025-06-15")
        assert isinstance(result, datetime)
        assert result.year == 2025 and result.month == 6 and result.day == 15

    def test_parse_datetime_object(self):
        mem = _make_memory()
        dt = datetime(2025, 6, 15, 10, 30)
        result = mem._parse_date(dt)
        assert result == dt

    def test_parse_date_object(self):
        mem = _make_memory()
        d = date(2025, 6, 15)
        result = mem._parse_date(d)
        assert isinstance(result, datetime)
        assert result.year == 2025 and result.month == 6 and result.day == 15

    def test_parse_invalid_returns_now(self):
        """Invalid input falls back to datetime.now()."""
        mem = _make_memory()
        before = datetime.now()
        result = mem._parse_date("not-a-date")
        after = datetime.now()
        assert before <= result <= after


# ===========================================================================
# 3. Storage with access history (add_situations)
# ===========================================================================

class TestAddSituationsWithTimestamp:
    """Test that access history is initialized alongside documents."""

    def test_default_timestamp_is_now(self):
        mem = _make_memory()
        before = datetime.now()
        mem.add_situations([("situation A", "advice A")])
        after = datetime.now()
        assert len(mem.access_history) == 1
        assert len(mem.access_history[0]) == 1  # one initial access
        assert before <= mem.access_history[0][0] <= after

    def test_explicit_timestamp(self):
        mem = _make_memory()
        ts = datetime(2025, 3, 1)
        mem.add_situations([("situation B", "advice B")], timestamp=ts)
        assert mem.access_history[0] == [ts]

    def test_multiple_batches_different_timestamps(self):
        mem = _make_memory()
        ts1 = datetime(2025, 1, 1)
        ts2 = datetime(2025, 6, 1)
        mem.add_situations([("sit1", "adv1")], timestamp=ts1)
        mem.add_situations([("sit2", "adv2")], timestamp=ts2)
        assert mem.access_history[0] == [ts1]
        assert mem.access_history[1] == [ts2]

    def test_string_timestamp(self):
        mem = _make_memory()
        mem.add_situations([("sit", "adv")], timestamp="2025-03-15")
        assert mem.access_history[0] == [datetime(2025, 3, 15)]

    def test_timestamps_property_backward_compat(self):
        """The .timestamps property returns creation time for each memory."""
        mem = _make_memory()
        ts1 = datetime(2025, 1, 1)
        ts2 = datetime(2025, 6, 1)
        mem.add_situations([("a", "b")], timestamp=ts1)
        mem.add_situations([("c", "d")], timestamp=ts2)
        assert mem.timestamps == [ts1, ts2]


# ===========================================================================
# 4. Access tracking (ACT-R specific)
# ===========================================================================

class TestAccessTracking:
    """Test that get_memories records retrieval access for ACT-R."""

    def test_get_memories_records_access(self):
        """Returned memories should get an access entry recorded."""
        mem = _make_memory()
        ts = datetime(2025, 6, 1)
        ref = datetime(2025, 6, 15)
        mem.add_situations([("inflation rising rates", "hedge bonds")], timestamp=ts)

        assert len(mem.access_history[0]) == 1  # only creation
        mem.get_memories("inflation rising rates", n_matches=1, current_date=ref)
        assert len(mem.access_history[0]) == 2  # creation + retrieval
        assert mem.access_history[0][1] == ref

    def test_repeated_retrieval_boosts_weight(self):
        """Multiple retrievals should increase the memory's decay weight."""
        mem = _make_memory()
        ts = datetime(2025, 5, 1)
        ref = datetime(2025, 6, 15)
        mem.add_situations([("tech crash recovery", "buy dip")], timestamp=ts)

        # First retrieval
        r1 = mem.get_memories("tech crash recovery", n_matches=1, current_date=ref)
        score1 = r1[0]["decay_score"]

        # Second retrieval (now has 3 access entries: creation + 2 retrievals)
        r2 = mem.get_memories("tech crash recovery", n_matches=1, current_date=ref)
        score2 = r2[0]["decay_score"]

        assert score2 > score1  # more accesses = higher weight

    def test_access_not_recorded_when_no_current_date(self):
        """Backward compat: no current_date = no access tracking."""
        mem = _make_memory()
        mem.add_situations([("sit", "adv")], timestamp=datetime(2025, 6, 1))

        mem.get_memories("sit", n_matches=1)  # no current_date
        assert len(mem.access_history[0]) == 1  # only creation, no retrieval recorded

    def test_access_history_preserves_creation_time(self):
        """First entry in access_history is always the creation timestamp."""
        mem = _make_memory()
        ts = datetime(2025, 3, 15)
        ref = datetime(2025, 6, 15)
        mem.add_situations([("data", "advice")], timestamp=ts)
        mem.get_memories("data", n_matches=1, current_date=ref)
        assert mem.access_history[0][0] == ts  # creation time preserved
        assert mem.access_history[0][1] == ref  # retrieval time added

    def test_only_returned_memories_get_access(self):
        """Non-returned memories should NOT get access recorded."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15)
        mem.add_situations([("inflation data", "advice A")], timestamp=ref - timedelta(days=1))
        mem.add_situations([("tech volatility", "advice B")], timestamp=ref - timedelta(days=1))

        mem.get_memories("inflation data", n_matches=1, current_date=ref)
        # Only the matched memory should get an access entry
        matched_idx = mem.documents.index("inflation data")
        other_idx = mem.documents.index("tech volatility")
        assert len(mem.access_history[matched_idx]) == 2  # creation + retrieval
        assert len(mem.access_history[other_idx]) == 1  # creation only


# ===========================================================================
# 5. Combined scoring (get_memories with ACT-R decay)
# ===========================================================================

class TestGetMemoriesWithDecay:
    """Test that retrieval combines BM25 score with ACT-R decay."""

    def test_recent_memory_ranks_higher(self):
        """Given identical content relevance, recent memory should score higher."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15)

        mem.add_situations(
            [("high inflation rising rates", "reduce bonds old")],
            timestamp=ref - timedelta(days=60),
        )
        mem.add_situations(
            [("high inflation rising rates", "reduce bonds recent")],
            timestamp=ref - timedelta(days=1),
        )

        results = mem.get_memories(
            "high inflation rising rates", n_matches=2, current_date=ref
        )
        assert results[0]["recommendation"] == "reduce bonds recent"

    def test_frequently_used_memory_ranks_higher(self):
        """ACT-R key feature: frequently retrieved memory beats recent-but-unused."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15)

        # Old memory (30 days) but will be retrieved multiple times
        mem.add_situations(
            [("market volatility spike", "reduce exposure old")],
            timestamp=ref - timedelta(days=30),
        )
        # Recent memory (5 days) but never retrieved again
        mem.add_situations(
            [("market volatility spike", "reduce exposure new")],
            timestamp=ref - timedelta(days=5),
        )

        # Simulate 3 prior retrievals of the old memory to boost it
        old_idx = 0
        for days_ago in [20, 10, 2]:
            mem.access_history[old_idx].append(ref - timedelta(days=days_ago))

        results = mem.get_memories(
            "market volatility spike", n_matches=2, current_date=ref
        )
        # Old but frequently used should beat recent but unused
        assert results[0]["recommendation"] == "reduce exposure old"

    def test_backward_compatible_no_current_date(self):
        """Without current_date, behavior should be same as original BM25-only."""
        mem = _make_memory()
        mem.add_situations([
            ("tech sector volatility", "reduce tech exposure"),
            ("strong dollar emerging markets", "hedge currency"),
        ])
        results = mem.get_memories("tech sector volatility", n_matches=1)
        assert len(results) == 1
        assert "tech" in results[0]["matched_situation"]

    def test_result_contains_decay_score_field(self):
        mem = _make_memory()
        ts = datetime(2025, 6, 1)
        mem.add_situations([("market crash", "buy dip")], timestamp=ts)
        results = mem.get_memories(
            "market crash", n_matches=1, current_date=datetime(2025, 6, 15)
        )
        assert "decay_score" in results[0]

    def test_result_contains_combined_score_field(self):
        mem = _make_memory()
        ts = datetime(2025, 6, 1)
        mem.add_situations([("market crash", "buy dip")], timestamp=ts)
        results = mem.get_memories(
            "market crash", n_matches=1, current_date=datetime(2025, 6, 15)
        )
        assert "combined_score" in results[0]

    def test_empty_memory_returns_empty(self):
        mem = _make_memory()
        results = mem.get_memories("anything", n_matches=1)
        assert results == []

    def test_combined_score_formula(self):
        """Verify: combined_score = similarity_score * decay_score."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15)
        mem.add_situations(
            [("unique query phrase alpha", "advice alpha")],
            timestamp=ref - timedelta(days=30),
        )
        results = mem.get_memories(
            "unique query phrase alpha", n_matches=1, current_date=ref
        )
        r = results[0]
        expected = r["similarity_score"] * r["decay_score"]
        assert r["combined_score"] == pytest.approx(expected, abs=1e-6)


# ===========================================================================
# 6. Memory pruning (_prune_if_needed)
# ===========================================================================

class TestMemoryPruning:
    """Test automatic eviction when memory exceeds max_entries."""

    def test_prune_removes_oldest_when_over_limit(self):
        mem = _make_memory(max_entries=3)
        base = datetime(2025, 6, 15)
        for i in range(4):
            mem.add_situations(
                [(f"situation {i}", f"advice {i}")],
                timestamp=base - timedelta(days=30 * (3 - i)),
            )
        assert len(mem.documents) == 3

    def test_prune_keeps_most_recent(self):
        mem = _make_memory(max_entries=2)
        base = datetime(2025, 6, 15)
        mem.add_situations([("old", "old advice")], timestamp=base - timedelta(days=90))
        mem.add_situations([("mid", "mid advice")], timestamp=base - timedelta(days=30))
        mem.add_situations([("new", "new advice")], timestamp=base - timedelta(days=1))
        assert "old" not in mem.documents
        assert "new" in mem.documents
        assert "mid" in mem.documents

    def test_no_prune_under_limit(self):
        mem = _make_memory(max_entries=10)
        mem.add_situations([("a", "b")], timestamp=datetime(2025, 1, 1))
        assert len(mem.documents) == 1

    def test_prune_maintains_index_consistency(self):
        """After pruning, documents/recommendations/access_history must be aligned."""
        mem = _make_memory(max_entries=2)
        base = datetime(2025, 6, 15)
        mem.add_situations([("old sit", "old rec")], timestamp=base - timedelta(days=90))
        mem.add_situations([("mid sit", "mid rec")], timestamp=base - timedelta(days=30))
        mem.add_situations([("new sit", "new rec")], timestamp=base - timedelta(days=1))

        assert len(mem.documents) == len(mem.recommendations) == len(mem.access_history)
        for i, doc in enumerate(mem.documents):
            if doc == "new sit":
                assert mem.recommendations[i] == "new rec"

    def test_prune_rebuilds_bm25_index(self):
        """BM25 index must work correctly after pruning."""
        mem = _make_memory(max_entries=2)
        base = datetime(2025, 6, 15)
        mem.add_situations([("old inflation data", "old advice")], timestamp=base - timedelta(days=90))
        mem.add_situations([("new inflation data", "new advice")], timestamp=base - timedelta(days=5))
        mem.add_situations([("tech volatility surge", "tech advice")], timestamp=base - timedelta(days=1))

        # After pruning (old removed), BM25 index rebuilt — queries still work
        assert len(mem.documents) == 2
        assert mem.bm25 is not None
        results = mem.get_memories("data inflation tech", n_matches=2, current_date=base)
        assert len(results) == 2


# ===========================================================================
# 7. Backward compatibility
# ===========================================================================

class TestBackwardCompatibility:
    """Ensure old API usage still works without errors."""

    def test_no_config(self):
        mem = FinancialSituationMemory("test")
        mem.add_situations([("sit", "adv")])
        results = mem.get_memories("sit", n_matches=1)
        assert len(results) == 1

    def test_empty_config(self):
        mem = FinancialSituationMemory("test", config={})
        mem.add_situations([("sit", "adv")])
        results = mem.get_memories("sit", n_matches=1)
        assert len(results) == 1

    def test_old_api_returns_same_fields(self):
        mem = FinancialSituationMemory("test")
        mem.add_situations([("inflation rising", "hedge bonds")])
        results = mem.get_memories("inflation rising", n_matches=1)
        r = results[0]
        assert "matched_situation" in r
        assert "recommendation" in r
        assert "similarity_score" in r

    def test_add_situations_without_timestamp(self):
        mem = FinancialSituationMemory("test")
        mem.add_situations([("a", "b"), ("c", "d")])
        assert len(mem.documents) == 2

    def test_old_half_life_config_does_not_crash(self):
        """Old config with memory_half_life_days should not error."""
        config = {"memory_half_life_days": 30}
        mem = FinancialSituationMemory("test", config=config)
        mem.add_situations([("sit", "adv")])
        results = mem.get_memories("sit", n_matches=1)
        assert len(results) == 1


# ===========================================================================
# 8. Config integration
# ===========================================================================

class TestConfigIntegration:
    """Test that config values are read correctly."""

    def test_custom_decay_exponent_from_config(self):
        """Custom d=0.3 should produce higher weight than d=0.5 for old memory."""
        config = {"memory_decay_exponent": 0.3}
        mem = FinancialSituationMemory("test", config=config)
        ref = datetime(2025, 6, 15)
        access_times = [ref - timedelta(days=30)]
        weight = mem._compute_decay(access_times, ref)
        # d=0.3 at 30 days ≈ 0.265 (higher than d=0.5 ≈ 0.154)
        assert weight == pytest.approx(0.265, abs=0.01)

    def test_custom_max_entries_from_config(self):
        config = {"memory_max_entries": 2}
        mem = FinancialSituationMemory("test", config=config)
        base = datetime(2025, 6, 15)
        for i in range(5):
            mem.add_situations(
                [(f"sit{i}", f"adv{i}")],
                timestamp=base - timedelta(days=i),
            )
        assert len(mem.documents) <= 2


# ===========================================================================
# 9. Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge case coverage."""

    def test_single_memory_just_created(self):
        """Memory accessed at reference time → high weight."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15, 12, 0, 0)
        mem.add_situations([("only one", "sole advice")], timestamp=ref)
        results = mem.get_memories("only one", n_matches=1, current_date=ref)
        assert len(results) == 1
        assert results[0]["decay_score"] > 0.95

    def test_zero_bm25_with_multiple_docs(self):
        """When BM25 gives 0 for a doc (among many), its combined should be ~0."""
        mem = _make_memory()
        ref = datetime(2025, 6, 15)
        mem.add_situations(
            [("inflation rising rates", "inflation advice")],
            timestamp=ref,
        )
        mem.add_situations(
            [("completely unrelated xyz qqq", "irrelevant advice")],
            timestamp=ref,
        )
        results = mem.get_memories("inflation rising rates", n_matches=2, current_date=ref)
        # The unrelated doc should have combined_score = 0
        unrelated = [r for r in results if "unrelated" in r["matched_situation"]]
        if unrelated:
            assert unrelated[0]["combined_score"] == pytest.approx(0.0, abs=1e-6)

    def test_clear_resets_access_history(self):
        mem = _make_memory()
        mem.add_situations([("sit", "adv")], timestamp=datetime(2025, 1, 1))
        assert len(mem.access_history) == 1
        mem.clear()
        assert len(mem.access_history) == 0
        assert len(mem.documents) == 0
        assert len(mem.recommendations) == 0
        assert mem.bm25 is None
