"""Tests for prediction verification logic and API endpoints - TDD RED phase."""

import json
from unittest.mock import patch, MagicMock, call
from datetime import date

import pytest
from fastapi.testclient import TestClient

from tradingagents.api.app import create_app
from tradingagents.api.deps import get_store, get_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """Provide a MagicMock PostgresStore."""
    store = MagicMock()
    return store


@pytest.fixture
def client(mock_store):
    """TestClient with mocked store dependency."""
    app = create_app()
    app.dependency_overrides[get_store] = lambda: mock_store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# verify_predictions core logic
# ---------------------------------------------------------------------------

class TestVerifyPredictionsLogic:
    """Tests for the verify_predictions function."""

    @patch("tradingagents.api.routers.predictions.baostock")
    def test_direction_hit_bullish_actual_up(self, mock_bs, mock_store):
        """Bullish prediction + actual price up = direction_hit True."""
        from tradingagents.api.routers.predictions import verify_predictions

        mock_store.get_pending_predictions.return_value = [
            {
                "id": 1,
                "predict_date": "2026-02-11",
                "category": "stock",
                "target": "Test Stock(002131)",
                "direction": "看涨",
                "entry_price": "10.50",
                "target_price": "11.20",
                "stop_loss": "10.00",
            },
        ]

        # Mock BaoStock to return price data showing upward movement
        mock_bs.login.return_value = MagicMock(error_code='0')
        mock_rs = MagicMock()
        mock_rs.error_code = '0'
        # Rows: date, open, high, low, close, volume, pctChg, turn
        mock_rs.next.side_effect = [True, True, False]
        mock_rs.get_row_data.side_effect = [
            ["2026-02-12", "10.60", "10.92", "10.45", "10.85", "500000", "1.5", "2.0"],
            ["2026-02-13", "10.80", "11.00", "10.70", "10.95", "450000", "0.9", "1.8"],
        ]
        mock_bs.query_history_k_data_plus.return_value = mock_rs

        result = verify_predictions(mock_store, days_back=3)

        # Should have called update_prediction_result
        mock_store.update_prediction_result.assert_called_once()
        call_args = mock_store.update_prediction_result.call_args
        pred_id = call_args[0][0]
        actual_result = call_args[0][1]
        status = call_args[0][2]

        assert pred_id == 1
        assert actual_result["direction_hit"] is True
        assert actual_result["actual_close"] == 10.95
        assert actual_result["actual_high"] == 11.00
        assert actual_result["actual_low"] == 10.45
        assert "verified" in status

    @patch("tradingagents.api.routers.predictions.baostock")
    def test_direction_hit_bearish_actual_down(self, mock_bs, mock_store):
        """Bearish prediction + actual price down = direction_hit True."""
        from tradingagents.api.routers.predictions import verify_predictions

        mock_store.get_pending_predictions.return_value = [
            {
                "id": 2,
                "predict_date": "2026-02-11",
                "category": "top_pick",
                "target": "Weak Stock(600123)",
                "direction": "看跌",
                "entry_price": "20.00",
                "target_price": "18.50",
                "stop_loss": "21.00",
            },
        ]

        mock_bs.login.return_value = MagicMock(error_code='0')
        mock_rs = MagicMock()
        mock_rs.error_code = '0'
        mock_rs.next.side_effect = [True, False]
        mock_rs.get_row_data.side_effect = [
            ["2026-02-12", "19.80", "20.10", "19.20", "19.30", "600000", "-3.5", "3.0"],
        ]
        mock_bs.query_history_k_data_plus.return_value = mock_rs

        verify_predictions(mock_store, days_back=3)

        call_args = mock_store.update_prediction_result.call_args
        actual_result = call_args[0][1]
        assert actual_result["direction_hit"] is True

    @patch("tradingagents.api.routers.predictions.baostock")
    def test_target_hit_bullish(self, mock_bs, mock_store):
        """Bullish prediction: target_hit when high >= target_price."""
        from tradingagents.api.routers.predictions import verify_predictions

        mock_store.get_pending_predictions.return_value = [
            {
                "id": 3,
                "predict_date": "2026-02-11",
                "category": "stock",
                "target": "Winner(000001)",
                "direction": "看涨",
                "entry_price": "10.50",
                "target_price": "11.20",
                "stop_loss": "10.00",
            },
        ]

        mock_bs.login.return_value = MagicMock(error_code='0')
        mock_rs = MagicMock()
        mock_rs.error_code = '0'
        mock_rs.next.side_effect = [True, False]
        mock_rs.get_row_data.side_effect = [
            ["2026-02-12", "10.60", "11.30", "10.50", "11.25", "700000", "7.1", "4.0"],
        ]
        mock_bs.query_history_k_data_plus.return_value = mock_rs

        verify_predictions(mock_store, days_back=3)

        call_args = mock_store.update_prediction_result.call_args
        actual_result = call_args[0][1]
        status = call_args[0][2]
        assert actual_result["target_hit"] is True
        assert status == "verified_win"

    @patch("tradingagents.api.routers.predictions.baostock")
    def test_stop_hit_bullish(self, mock_bs, mock_store):
        """Bullish prediction: stop_hit when low <= stop_loss."""
        from tradingagents.api.routers.predictions import verify_predictions

        mock_store.get_pending_predictions.return_value = [
            {
                "id": 4,
                "predict_date": "2026-02-11",
                "category": "stock",
                "target": "Loser(300001)",
                "direction": "看涨",
                "entry_price": "10.50",
                "target_price": "11.20",
                "stop_loss": "10.00",
            },
        ]

        mock_bs.login.return_value = MagicMock(error_code='0')
        mock_rs = MagicMock()
        mock_rs.error_code = '0'
        mock_rs.next.side_effect = [True, False]
        mock_rs.get_row_data.side_effect = [
            ["2026-02-12", "10.30", "10.40", "9.80", "9.90", "800000", "-5.7", "5.0"],
        ]
        mock_bs.query_history_k_data_plus.return_value = mock_rs

        verify_predictions(mock_store, days_back=3)

        call_args = mock_store.update_prediction_result.call_args
        actual_result = call_args[0][1]
        status = call_args[0][2]
        assert actual_result["stop_hit"] is True
        assert status == "verified_loss"

    @patch("tradingagents.api.routers.predictions.baostock")
    def test_neutral_result(self, mock_bs, mock_store):
        """When direction is wrong and neither target nor stop hit -> neutral."""
        from tradingagents.api.routers.predictions import verify_predictions

        mock_store.get_pending_predictions.return_value = [
            {
                "id": 5,
                "predict_date": "2026-02-11",
                "category": "stock",
                "target": "Flat(000002)",
                "direction": "看涨",
                "entry_price": "10.50",
                "target_price": "11.20",
                "stop_loss": "10.00",
            },
        ]

        mock_bs.login.return_value = MagicMock(error_code='0')
        mock_rs = MagicMock()
        mock_rs.error_code = '0'
        mock_rs.next.side_effect = [True, False]
        # Price went down slightly but didn't hit stop
        mock_rs.get_row_data.side_effect = [
            ["2026-02-12", "10.45", "10.55", "10.20", "10.30", "300000", "-1.9", "1.5"],
        ]
        mock_bs.query_history_k_data_plus.return_value = mock_rs

        verify_predictions(mock_store, days_back=3)

        call_args = mock_store.update_prediction_result.call_args
        actual_result = call_args[0][1]
        status = call_args[0][2]
        assert actual_result["direction_hit"] is False
        assert actual_result["target_hit"] is False
        assert actual_result["stop_hit"] is False
        assert status == "verified_neutral"

    @patch("tradingagents.api.routers.predictions.baostock")
    def test_skips_when_no_stock_code_found(self, mock_bs, mock_store):
        """Should skip predictions where no stock code can be extracted."""
        from tradingagents.api.routers.predictions import verify_predictions

        mock_store.get_pending_predictions.return_value = [
            {
                "id": 6,
                "predict_date": "2026-02-11",
                "category": "stock",
                "target": "NoCodeStock",
                "direction": "看涨",
                "entry_price": "10.50",
                "target_price": "11.20",
                "stop_loss": "10.00",
            },
        ]

        result = verify_predictions(mock_store, days_back=3)

        mock_store.update_prediction_result.assert_not_called()
        assert result["skipped"] >= 1

    @patch("tradingagents.api.routers.predictions.baostock")
    def test_handles_no_baostock_data(self, mock_bs, mock_store):
        """Should skip when BaoStock returns no data."""
        from tradingagents.api.routers.predictions import verify_predictions

        mock_store.get_pending_predictions.return_value = [
            {
                "id": 7,
                "predict_date": "2026-02-11",
                "category": "stock",
                "target": "NoData(999999)",
                "direction": "看涨",
                "entry_price": "10.50",
                "target_price": "11.20",
                "stop_loss": "10.00",
            },
        ]

        mock_bs.login.return_value = MagicMock(error_code='0')
        mock_rs = MagicMock()
        mock_rs.error_code = '0'
        mock_rs.next.return_value = False  # No data
        mock_bs.query_history_k_data_plus.return_value = mock_rs

        result = verify_predictions(mock_store, days_back=3)

        mock_store.update_prediction_result.assert_not_called()
        assert result["skipped"] >= 1

    @patch("tradingagents.api.routers.predictions.baostock")
    def test_actual_return_pct_calculation(self, mock_bs, mock_store):
        """Should calculate actual_return_pct = (close - entry) / entry * 100."""
        from tradingagents.api.routers.predictions import verify_predictions

        mock_store.get_pending_predictions.return_value = [
            {
                "id": 8,
                "predict_date": "2026-02-11",
                "category": "stock",
                "target": "ReturnCalc(000010)",
                "direction": "看涨",
                "entry_price": "10.00",
                "target_price": "11.00",
                "stop_loss": "9.50",
            },
        ]

        mock_bs.login.return_value = MagicMock(error_code='0')
        mock_rs = MagicMock()
        mock_rs.error_code = '0'
        mock_rs.next.side_effect = [True, False]
        mock_rs.get_row_data.side_effect = [
            ["2026-02-12", "10.10", "10.50", "9.90", "10.30", "400000", "3.0", "2.0"],
        ]
        mock_bs.query_history_k_data_plus.return_value = mock_rs

        verify_predictions(mock_store, days_back=3)

        call_args = mock_store.update_prediction_result.call_args
        actual_result = call_args[0][1]
        # (10.30 - 10.00) / 10.00 * 100 = 3.0
        assert abs(actual_result["actual_return_pct"] - 3.0) < 0.01

    @patch("tradingagents.api.routers.predictions.baostock")
    def test_holding_days_in_result(self, mock_bs, mock_store):
        """actual_result should contain holding_days = number of trading days."""
        from tradingagents.api.routers.predictions import verify_predictions

        mock_store.get_pending_predictions.return_value = [
            {
                "id": 10,
                "predict_date": "2026-02-11",
                "category": "stock",
                "target": "HoldDays(000010)",
                "direction": "看涨",
                "entry_price": "10.00",
                "target_price": "11.00",
                "stop_loss": "9.50",
            },
        ]

        mock_bs.login.return_value = MagicMock(error_code='0')
        mock_rs = MagicMock()
        mock_rs.error_code = '0'
        mock_rs.next.side_effect = [True, True, True, False]
        mock_rs.get_row_data.side_effect = [
            ["2026-02-12", "10.10", "10.30", "9.90", "10.20", "400000", "2.0", "1.5"],
            ["2026-02-13", "10.20", "10.50", "10.10", "10.40", "350000", "2.0", "1.3"],
            ["2026-02-14", "10.40", "10.60", "10.30", "10.50", "380000", "1.0", "1.4"],
        ]
        mock_bs.query_history_k_data_plus.return_value = mock_rs

        verify_predictions(mock_store, days_back=3)

        actual_result = mock_store.update_prediction_result.call_args[0][1]
        assert actual_result["holding_days"] == 3

    @patch("tradingagents.api.routers.predictions.baostock")
    def test_daily_returns_in_result(self, mock_bs, mock_store):
        """actual_result should contain daily_returns list from each day's close-to-close."""
        from tradingagents.api.routers.predictions import verify_predictions

        mock_store.get_pending_predictions.return_value = [
            {
                "id": 11,
                "predict_date": "2026-02-11",
                "category": "stock",
                "target": "DailyRet(000010)",
                "direction": "看涨",
                "entry_price": "10.00",
                "target_price": "11.00",
                "stop_loss": "9.50",
            },
        ]

        mock_bs.login.return_value = MagicMock(error_code='0')
        mock_rs = MagicMock()
        mock_rs.error_code = '0'
        mock_rs.next.side_effect = [True, True, False]
        mock_rs.get_row_data.side_effect = [
            # Day 1: entry 10.00 -> close 10.20 => +2.0%
            ["2026-02-12", "10.10", "10.30", "9.90", "10.20", "400000", "2.0", "1.5"],
            # Day 2: prev close 10.20 -> close 10.50 => +2.94%
            ["2026-02-13", "10.20", "10.60", "10.10", "10.50", "350000", "2.9", "1.3"],
        ]
        mock_bs.query_history_k_data_plus.return_value = mock_rs

        verify_predictions(mock_store, days_back=3)

        actual_result = mock_store.update_prediction_result.call_args[0][1]
        assert "daily_returns" in actual_result
        assert len(actual_result["daily_returns"]) == 2
        # Day 1: (10.20 - 10.00) / 10.00 * 100 = 2.0
        assert abs(actual_result["daily_returns"][0] - 2.0) < 0.01
        # Day 2: (10.50 - 10.20) / 10.20 * 100 = 2.94
        assert abs(actual_result["daily_returns"][1] - 2.94) < 0.1

    @patch("tradingagents.api.routers.predictions.baostock")
    def test_cost_adjusted_return_in_result(self, mock_bs, mock_store):
        """actual_result should contain cost_adjusted_return deducting trading costs.
        A-share costs: commission 0.025% * 2 + stamp tax 0.05% (sell only) = 0.1% round trip.
        """
        from tradingagents.api.routers.predictions import verify_predictions

        mock_store.get_pending_predictions.return_value = [
            {
                "id": 12,
                "predict_date": "2026-02-11",
                "category": "stock",
                "target": "CostAdj(000010)",
                "direction": "看涨",
                "entry_price": "10.00",
                "target_price": "11.00",
                "stop_loss": "9.50",
            },
        ]

        mock_bs.login.return_value = MagicMock(error_code='0')
        mock_rs = MagicMock()
        mock_rs.error_code = '0'
        mock_rs.next.side_effect = [True, False]
        mock_rs.get_row_data.side_effect = [
            ["2026-02-12", "10.10", "10.50", "9.90", "10.30", "400000", "3.0", "2.0"],
        ]
        mock_bs.query_history_k_data_plus.return_value = mock_rs

        verify_predictions(mock_store, days_back=3)

        actual_result = mock_store.update_prediction_result.call_args[0][1]
        assert "cost_adjusted_return" in actual_result
        # raw return = 3.0%, cost = 0.1%, adjusted = 2.9%
        assert abs(actual_result["cost_adjusted_return"] - 2.9) < 0.05

    @patch("tradingagents.api.routers.predictions.baostock")
    def test_entry_price_range_uses_midpoint(self, mock_bs, mock_store):
        """When entry_price is a range like '10.00-10.50', use midpoint."""
        from tradingagents.api.routers.predictions import verify_predictions

        mock_store.get_pending_predictions.return_value = [
            {
                "id": 9,
                "predict_date": "2026-02-11",
                "category": "stock",
                "target": "Range(000020)",
                "direction": "看涨",
                "entry_price": "10.00-10.50",
                "target_price": "11.50",
                "stop_loss": "9.50",
            },
        ]

        mock_bs.login.return_value = MagicMock(error_code='0')
        mock_rs = MagicMock()
        mock_rs.error_code = '0'
        mock_rs.next.side_effect = [True, False]
        mock_rs.get_row_data.side_effect = [
            ["2026-02-12", "10.30", "10.80", "10.20", "10.60", "350000", "1.5", "1.7"],
        ]
        mock_bs.query_history_k_data_plus.return_value = mock_rs

        verify_predictions(mock_store, days_back=3)

        call_args = mock_store.update_prediction_result.call_args
        actual_result = call_args[0][1]
        # midpoint = (10.00 + 10.50) / 2 = 10.25
        # return = (10.60 - 10.25) / 10.25 * 100 = 3.41
        assert abs(actual_result["actual_return_pct"] - 3.41) < 0.1


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

class TestVerifyEndpoint:
    """Tests for POST /api/predictions/verify."""

    @patch("tradingagents.api.routers.predictions.verify_predictions")
    def test_verify_endpoint_success(self, mock_verify, client, mock_store):
        """POST /api/predictions/verify should call verify_predictions and return result."""
        mock_verify.return_value = {
            "verified": 3,
            "skipped": 1,
            "errors": 0,
        }

        resp = client.post("/api/predictions/verify")

        assert resp.status_code == 200
        data = resp.json()
        assert data["verified"] == 3
        assert data["skipped"] == 1

    @patch("tradingagents.api.routers.predictions.verify_predictions")
    def test_verify_endpoint_handles_error(self, mock_verify, client, mock_store):
        """POST /api/predictions/verify should return 500 on failure."""
        mock_verify.side_effect = Exception("BaoStock failed")

        resp = client.post("/api/predictions/verify")

        assert resp.status_code == 500


class TestStatsEndpoint:
    """Tests for GET /api/predictions/stats."""

    def test_stats_endpoint_success(self, client, mock_store):
        """GET /api/predictions/stats should return prediction statistics."""
        mock_store.get_prediction_stats.return_value = {
            "total_verified": 10,
            "wins": 6,
            "losses": 3,
            "partial_wins": 1,
            "win_rate": 60.0,
            "avg_return_pct": 2.35,
        }

        resp = client.get("/api/predictions/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_verified"] == 10
        assert data["win_rate"] == 60.0

    def test_stats_endpoint_with_days_param(self, client, mock_store):
        """GET /api/predictions/stats?days=7 should pass days parameter."""
        mock_store.get_prediction_stats.return_value = {
            "total_verified": 3,
            "wins": 2,
            "losses": 1,
            "partial_wins": 0,
            "win_rate": 66.7,
            "avg_return_pct": 1.5,
        }

        resp = client.get("/api/predictions/stats?days=7")

        assert resp.status_code == 200
        mock_store.get_prediction_stats.assert_called_once_with(days=7)
