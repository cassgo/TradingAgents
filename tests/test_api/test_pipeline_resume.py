"""Tests for S3 debate checkpoint resume + incremental persistence."""

from unittest.mock import MagicMock, patch, call
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(code, vol_ratio=1.0, **extra):
    return {"code": code, "code_name": f"Stock-{code}", "vol_ratio": vol_ratio, **extra}


def _make_debate_result(candidate, signal="buy"):
    return {
        **candidate,
        "signal": signal,
        "debate_summary": "summary",
        "risk_summary": "risk",
        "investment_plan": "plan",
    }


# ---------------------------------------------------------------------------
# 1. on_result callback fires for each debated candidate
# ---------------------------------------------------------------------------


class TestOnResultCallback:
    def test_on_result_callback_fires(self):
        """debater.debate_candidates calls on_result once per successful result."""
        from tradingagents.pool.debater import PoolDebater

        graph = MagicMock()
        graph.propagate.return_value = ({"inv_judge_decision": "ok", "risk_judge_decision": "r", "investment_plan": "p"}, "buy")

        debater = PoolDebater(graph=graph, store=MagicMock(), max_debates=10)
        candidates = [_make_candidate("001"), _make_candidate("002"), _make_candidate("003")]

        on_result = MagicMock()
        results = debater.debate_candidates(candidates, "2026-02-16", on_result=on_result)

        assert len(results) == 3
        assert on_result.call_count == 3
        # Each call receives the enriched result dict
        for c in on_result.call_args_list:
            assert "signal" in c[0][0]

    def test_on_result_not_called_on_failure(self):
        """on_result is NOT called when debate_single fails (returns None)."""
        from tradingagents.pool.debater import PoolDebater

        graph = MagicMock()
        graph.propagate.side_effect = [
            ({"inv_judge_decision": "ok", "risk_judge_decision": "r", "investment_plan": "p"}, "buy"),
            Exception("LLM timeout"),
            ({"inv_judge_decision": "ok", "risk_judge_decision": "r", "investment_plan": "p"}, "sell"),
        ]

        debater = PoolDebater(graph=graph, store=MagicMock(), max_debates=10)
        candidates = [_make_candidate("001"), _make_candidate("002"), _make_candidate("003")]

        on_result = MagicMock()
        results = debater.debate_candidates(candidates, "2026-02-16", on_result=on_result)

        assert len(results) == 2
        assert on_result.call_count == 2

    def test_on_result_default_none(self):
        """on_result defaults to None — backward compatible."""
        from tradingagents.pool.debater import PoolDebater

        graph = MagicMock()
        graph.propagate.return_value = ({"inv_judge_decision": "", "risk_judge_decision": "", "investment_plan": ""}, "buy")

        debater = PoolDebater(graph=graph, store=MagicMock(), max_debates=10)
        candidates = [_make_candidate("001")]

        # No on_result argument — should not raise
        results = debater.debate_candidates(candidates, "2026-02-16")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# 2. resume skips completed codes
# ---------------------------------------------------------------------------


class TestResumeSkipsCompleted:
    def _build_pipeline(self, store, graph=None):
        from tradingagents.pool.pipeline import PoolPipeline
        return PoolPipeline(store=store, config={"max_debates": 12}, graph=graph or MagicMock())

    def test_resume_skips_completed_codes(self):
        """resume=True skips codes already in S3 candidates."""
        store = MagicMock()
        # Previous run with S2 candidates
        store.list_screening_runs.return_value = [{
            "id": 42, "s0_count": 100, "s1_count": 50, "s2_count": 20,
            "stage3_count": 5, "stage4_count": 0, "status": "failed",
        }]
        s2_candidates = [
            _make_candidate("A", vol_ratio=10),
            _make_candidate("B", vol_ratio=9),
            _make_candidate("C", vol_ratio=8),
            _make_candidate("D", vol_ratio=7),
        ]
        # get_screening_candidates: stage=2 returns S2, stage=3 returns already-done
        def _get_candidates(run_id, stage):
            if stage == 2:
                return s2_candidates
            if stage == 3:
                return [{"code": "A", "code_name": "Stock-A", "signal": "buy"}]
            return []
        store.get_screening_candidates.side_effect = _get_candidates
        store.create_screening_run.return_value = 99
        store.update_screening_run.return_value = None
        store.save_screening_candidates.return_value = None

        graph = MagicMock()
        # Only B, C, D should be debated (A is done)
        graph.propagate.return_value = (
            {"inv_judge_decision": "ok", "risk_judge_decision": "r", "investment_plan": "p"},
            "buy",
        )

        pipeline = self._build_pipeline(store, graph)

        with patch.object(pipeline, '_get_llm', return_value=MagicMock()), \
             patch.object(pipeline, '_finalize_run'), \
             patch.object(pipeline, '_run_predictions', return_value={}):
            pipeline.run_stage(3, "2026-02-16", resume=True)

        # graph.propagate should be called 3 times (B, C, D) — not 4
        assert graph.propagate.call_count == 3
        debated_codes = [c[0][0] for c in graph.propagate.call_args_list]
        assert "Stock-A" not in debated_codes

    def test_resume_false_runs_all(self):
        """resume=False runs all candidates (no skipping)."""
        store = MagicMock()
        store.list_screening_runs.return_value = [{
            "id": 42, "s0_count": 100, "s1_count": 50, "s2_count": 4,
            "stage3_count": 0, "stage4_count": 0, "status": "failed",
        }]
        s2_candidates = [
            _make_candidate("A", vol_ratio=10),
            _make_candidate("B", vol_ratio=9),
        ]
        store.get_screening_candidates.return_value = s2_candidates
        store.create_screening_run.return_value = 99
        store.update_screening_run.return_value = None
        store.save_screening_candidates.return_value = None

        graph = MagicMock()
        graph.propagate.return_value = (
            {"inv_judge_decision": "ok", "risk_judge_decision": "r", "investment_plan": "p"},
            "buy",
        )

        pipeline = self._build_pipeline(store, graph)

        with patch.object(pipeline, '_get_llm', return_value=MagicMock()), \
             patch.object(pipeline, '_finalize_run'), \
             patch.object(pipeline, '_run_predictions', return_value={}):
            pipeline.run_stage(3, "2026-02-16", resume=False)

        # Both A and B debated
        assert graph.propagate.call_count == 2


# ---------------------------------------------------------------------------
# 3. resume reuses run_id
# ---------------------------------------------------------------------------


class TestResumeReusesRunId:
    def test_resume_reuses_run_id(self):
        """resume=True reuses previous run_id, does NOT call create_screening_run."""
        from tradingagents.pool.pipeline import PoolPipeline

        store = MagicMock()
        store.list_screening_runs.return_value = [{
            "id": 42, "s0_count": 100, "s1_count": 50, "s2_count": 4,
            "stage3_count": 2, "stage4_count": 0, "status": "failed",
        }]
        store.get_screening_candidates.side_effect = lambda run_id, stage: (
            [_make_candidate("A"), _make_candidate("B")] if stage == 2
            else [{"code": "A", "code_name": "Stock-A", "signal": "buy"}] if stage == 3
            else []
        )
        store.update_screening_run.return_value = None
        store.save_screening_candidates.return_value = None

        graph = MagicMock()
        graph.propagate.return_value = (
            {"inv_judge_decision": "ok", "risk_judge_decision": "r", "investment_plan": "p"},
            "buy",
        )

        pipeline = PoolPipeline(store=store, config={"max_debates": 12}, graph=graph)

        with patch.object(pipeline, '_get_llm', return_value=MagicMock()), \
             patch.object(pipeline, '_finalize_run'), \
             patch.object(pipeline, '_run_predictions', return_value={}):
            pipeline.run_stage(3, "2026-02-16", resume=True)

        # create_screening_run should NOT be called in resume mode
        store.create_screening_run.assert_not_called()

        # update_screening_run should be called with run_id=42 (reused)
        update_calls = store.update_screening_run.call_args_list
        run_ids_used = {c[0][0] for c in update_calls}
        assert 42 in run_ids_used


# ---------------------------------------------------------------------------
# 4. resume with no existing S3 results = full run
# ---------------------------------------------------------------------------


class TestResumeNoExistingS3:
    def test_resume_no_existing_s3(self):
        """When resume=True but no S3 candidates exist, debate all."""
        from tradingagents.pool.pipeline import PoolPipeline

        store = MagicMock()
        store.list_screening_runs.return_value = [{
            "id": 42, "s0_count": 100, "s1_count": 50, "s2_count": 3,
            "stage3_count": 0, "stage4_count": 0, "status": "failed",
        }]
        store.get_screening_candidates.side_effect = lambda run_id, stage: (
            [_make_candidate("A", vol_ratio=5), _make_candidate("B", vol_ratio=4), _make_candidate("C", vol_ratio=3)]
            if stage == 2
            else []
        )
        store.update_screening_run.return_value = None
        store.save_screening_candidates.return_value = None

        graph = MagicMock()
        graph.propagate.return_value = (
            {"inv_judge_decision": "ok", "risk_judge_decision": "r", "investment_plan": "p"},
            "buy",
        )

        pipeline = PoolPipeline(store=store, config={"max_debates": 12}, graph=graph)

        with patch.object(pipeline, '_get_llm', return_value=MagicMock()), \
             patch.object(pipeline, '_finalize_run'), \
             patch.object(pipeline, '_run_predictions', return_value={}):
            pipeline.run_stage(3, "2026-02-16", resume=True)

        # All 3 debated
        assert graph.propagate.call_count == 3


# ---------------------------------------------------------------------------
# 5. resume progress offset
# ---------------------------------------------------------------------------


class TestResumeProgressOffset:
    def test_resume_progress_offset(self):
        """Progress reporting starts from done_count when resuming."""
        from tradingagents.pool.pipeline import PoolPipeline

        store = MagicMock()
        store.list_screening_runs.return_value = [{
            "id": 42, "s0_count": 100, "s1_count": 50, "s2_count": 4,
            "stage3_count": 2, "stage4_count": 0, "status": "failed",
        }]
        store.get_screening_candidates.side_effect = lambda run_id, stage: (
            [
                _make_candidate("A", vol_ratio=10),
                _make_candidate("B", vol_ratio=9),
                _make_candidate("C", vol_ratio=8),
                _make_candidate("D", vol_ratio=7),
            ]
            if stage == 2
            else [
                {"code": "A", "code_name": "Stock-A", "signal": "buy"},
                {"code": "B", "code_name": "Stock-B", "signal": "sell"},
            ] if stage == 3
            else []
        )
        store.update_screening_run.return_value = None
        store.save_screening_candidates.return_value = None

        graph = MagicMock()
        graph.propagate.return_value = (
            {"inv_judge_decision": "ok", "risk_judge_decision": "r", "investment_plan": "p"},
            "buy",
        )

        pipeline = PoolPipeline(store=store, config={"max_debates": 12}, graph=graph)

        with patch.object(pipeline, '_get_llm', return_value=MagicMock()), \
             patch.object(pipeline, '_finalize_run'), \
             patch.object(pipeline, '_run_predictions', return_value={}):
            pipeline.run_stage(3, "2026-02-16", resume=True)

        # Check progress updates: should show 3/4 and 4/4 (offset by done_count=2)
        progress_updates = [
            c for c in store.update_screening_run.call_args_list
            if len(c[0]) >= 2 and isinstance(c[0][1], dict) and "stage3_progress" in c[0][1]
        ]
        progress_values = [c[0][1]["stage3_progress"] for c in progress_updates]
        assert "3/4" in progress_values
        assert "4/4" in progress_values
        # Should NOT start from "1/2"
        assert "1/2" not in progress_values


# ---------------------------------------------------------------------------
# 6. Incremental save survives crash
# ---------------------------------------------------------------------------


class TestIncrementalSaveSurvivesCrash:
    def test_incremental_save_survives_crash(self):
        """If debate crashes at candidate N, first N-1 results are already saved."""
        from tradingagents.pool.pipeline import PoolPipeline

        store = MagicMock()
        store.list_screening_runs.return_value = [{
            "id": 42, "s0_count": 100, "s1_count": 50, "s2_count": 3,
            "stage3_count": 0, "stage4_count": 0, "status": "failed",
        }]
        store.get_screening_candidates.return_value = [
            _make_candidate("A", vol_ratio=10),
            _make_candidate("B", vol_ratio=9),
            _make_candidate("C", vol_ratio=8),
        ]
        store.create_screening_run.return_value = 99
        store.update_screening_run.return_value = None
        store.save_screening_candidates.return_value = None

        # First two succeed, third crashes
        graph = MagicMock()
        graph.propagate.side_effect = [
            ({"inv_judge_decision": "ok", "risk_judge_decision": "r", "investment_plan": "p"}, "buy"),
            ({"inv_judge_decision": "ok", "risk_judge_decision": "r", "investment_plan": "p"}, "sell"),
            RuntimeError("Gateway timeout"),
        ]

        pipeline = PoolPipeline(store=store, config={"max_debates": 12}, graph=graph)

        with patch.object(pipeline, '_get_llm', return_value=MagicMock()):
            # The pipeline should fail on the exception, but incremental saves
            # already happened via on_result for the first 2 candidates
            try:
                pipeline.run_stage(3, "2026-02-16", resume=False)
            except RuntimeError:
                pass  # Expected — third candidate blew up

        # save_screening_candidates should have been called for each successful result
        save_calls = store.save_screening_candidates.call_args_list
        # At least 2 incremental saves (one per successful debate)
        assert len(save_calls) >= 2


# ---------------------------------------------------------------------------
# 7. API endpoint passes resume parameter
# ---------------------------------------------------------------------------


class TestApiResumeParam:
    def test_api_resume_param_true(self):
        """POST /api/pool/trigger/stage/3?resume=true passes resume=True."""
        from fastapi.testclient import TestClient
        from tradingagents.api.app import create_app
        from tradingagents.api.deps import get_store, get_config

        app = create_app()
        mock_store = MagicMock()
        mock_store.list_screening_runs.return_value = [{
            "id": 1, "s0_count": 10, "s1_count": 5, "s2_count": 3,
            "stage3_count": 2, "stage4_count": 0,
        }]
        mock_store.get_screening_candidates.return_value = [
            {"code": "001", "code_name": "Test"}
        ]

        app.dependency_overrides[get_store] = lambda: mock_store
        app.dependency_overrides[get_config] = lambda: {"max_debates": 3}

        with patch("tradingagents.api.routers.pool.get_pipeline_queue") as mock_queue:
            mock_queue.return_value.submit.return_value = "task-123"
            client = TestClient(app)
            resp = client.post("/api/pool/trigger/stage/3?resume=true")

        assert resp.status_code == 200
        data = resp.json()
        assert data["resume"] is True
        assert data["task_id"] == "task-123"

    def test_api_resume_param_default_false(self):
        """POST /api/pool/trigger/stage/3 defaults resume=False."""
        from fastapi.testclient import TestClient
        from tradingagents.api.app import create_app
        from tradingagents.api.deps import get_store, get_config

        app = create_app()
        mock_store = MagicMock()
        mock_store.list_screening_runs.return_value = [{
            "id": 1, "s0_count": 10, "s1_count": 5, "s2_count": 3,
            "stage3_count": 0, "stage4_count": 0,
        }]
        mock_store.get_screening_candidates.return_value = [
            {"code": "001", "code_name": "Test"}
        ]

        app.dependency_overrides[get_store] = lambda: mock_store
        app.dependency_overrides[get_config] = lambda: {"max_debates": 3}

        with patch("tradingagents.api.routers.pool.get_pipeline_queue") as mock_queue:
            mock_queue.return_value.submit.return_value = "task-456"
            client = TestClient(app)
            resp = client.post("/api/pool/trigger/stage/3")

        assert resp.status_code == 200
        data = resp.json()
        assert data.get("resume") is False
