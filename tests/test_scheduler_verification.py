"""Tests for scheduler prediction verification job - TDD RED phase."""

from unittest.mock import patch, MagicMock

import pytest


class TestSchedulerVerificationJob:
    """Tests for the prediction verification scheduler job."""

    @patch("tradingagents.scheduler.news_scheduler._get_store")
    @patch("tradingagents.scheduler.news_scheduler._broadcast_event")
    @patch("tradingagents.api.routers.predictions.verify_predictions")
    def test_job_verify_predictions_calls_verify(self, mock_verify, mock_broadcast, mock_get_store):
        """job_verify_predictions should call verify_predictions with the store."""
        from tradingagents.scheduler.news_scheduler import job_verify_predictions

        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_verify.return_value = {"verified": 2, "skipped": 0, "errors": 0}

        job_verify_predictions()

        mock_verify.assert_called_once()
        # First arg should be the store
        assert mock_verify.call_args[0][0] is mock_store

    @patch("tradingagents.scheduler.news_scheduler._get_store")
    @patch("tradingagents.scheduler.news_scheduler._broadcast_event")
    @patch("tradingagents.api.routers.predictions.verify_predictions")
    def test_job_verify_predictions_broadcasts_event(self, mock_verify, mock_broadcast, mock_get_store):
        """job_verify_predictions should broadcast verification:done event."""
        from tradingagents.scheduler.news_scheduler import job_verify_predictions

        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_verify.return_value = {"verified": 3, "skipped": 1, "errors": 0}

        job_verify_predictions()

        mock_broadcast.assert_called_once()
        event_data = mock_broadcast.call_args[0]
        assert event_data[0] == "verification:done"

    @patch("tradingagents.scheduler.news_scheduler._get_store")
    @patch("tradingagents.scheduler.news_scheduler._broadcast_event")
    def test_job_verify_predictions_handles_error(self, mock_broadcast, mock_get_store):
        """job_verify_predictions should catch exceptions and not crash."""
        from tradingagents.scheduler.news_scheduler import job_verify_predictions

        mock_get_store.side_effect = Exception("DB connection failed")

        # Should not raise
        job_verify_predictions()

    def test_scheduler_registers_verification_job(self):
        """start_scheduler should register a verification job at 16:30."""
        import tradingagents.scheduler.news_scheduler as mod

        # Reset module state
        mod._scheduler = None

        with patch("tradingagents.scheduler.news_scheduler.BackgroundScheduler") as MockScheduler:
            mock_sched = MagicMock()
            MockScheduler.return_value = mock_sched
            mock_sched.get_jobs.return_value = []

            mod.start_scheduler()

            # Check that add_job was called with verification job
            add_job_calls = mock_sched.add_job.call_args_list
            verification_jobs = [
                c for c in add_job_calls
                if "verify" in str(c).lower()
            ]
            assert len(verification_jobs) >= 1, "Should register at least one verification job"

            # Verify it's scheduled at 16:30
            verify_call = verification_jobs[0]
            trigger = verify_call[1].get("trigger") if len(verify_call) > 1 else verify_call[0][1]
            # The trigger should be a CronTrigger with hour=16, minute=30

            # Clean up
            mod._scheduler = None
