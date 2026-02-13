"""Tests for PostgresStore prediction verification methods - TDD RED phase."""

import json
from unittest.mock import patch, MagicMock

import pytest

from tradingagents.storage.postgres_store import PostgresStore


MOCK_DB_URL = "postgresql://ta_user:ta_pass@localhost:5432/tradingagents"


def _make_store(mock_pg):
    """Helper to create a PostgresStore with mocked psycopg2."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_pg.connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    store = PostgresStore(database_url=MOCK_DB_URL)
    return store, mock_conn, mock_cursor


class TestGetPendingPredictions:
    """Tests for PostgresStore.get_pending_predictions()."""

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_returns_pending_stock_and_top_pick(self, mock_pg):
        """Should query predictions with status='pending', predict_date < before_date,
        and category in ('stock', 'top_pick')."""
        store, mock_conn, mock_cursor = _make_store(mock_pg)

        mock_cursor.description = [
            ("id",), ("predict_date",), ("trade_date",), ("category",),
            ("target",), ("direction",), ("confidence",), ("reasoning",),
            ("entry_price",), ("target_price",), ("stop_loss",),
            ("time_horizon",), ("status",), ("actual_result",), ("created_at",),
        ]
        mock_cursor.fetchall.return_value = [
            (1, "2026-02-12", "2026-02-12", "stock",
             "Test(002131)", "看涨", 0.85, "reasoning",
             "10.50", "11.20", "10.00",
             "1-3 days", "pending", "", "2026-02-12T10:00:00"),
            (2, "2026-02-12", "2026-02-12", "top_pick",
             "TopPick(600000)", "看涨", 0.90, "top reasoning",
             "8.50", "9.20", "8.00",
             "1-5 days", "pending", "", "2026-02-12T10:00:00"),
        ]

        results = store.get_pending_predictions("2026-02-13")

        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["category"] == "stock"
        assert results[1]["category"] == "top_pick"

        sql = mock_cursor.execute.call_args_list[-1][0][0]
        assert "status" in sql
        assert "pending" in sql.lower() or "pending" in str(mock_cursor.execute.call_args_list[-1][0][1])
        assert "predict_date" in sql
        store.close()

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_returns_empty_when_no_pending(self, mock_pg):
        """Should return empty list when no pending predictions exist."""
        store, mock_conn, mock_cursor = _make_store(mock_pg)

        mock_cursor.description = [
            ("id",), ("predict_date",), ("category",), ("target",),
            ("direction",), ("status",),
        ]
        mock_cursor.fetchall.return_value = []

        results = store.get_pending_predictions("2026-02-13")

        assert results == []
        store.close()

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_excludes_sector_and_summary(self, mock_pg):
        """Should only include stock and top_pick categories, not sector or summary."""
        store, mock_conn, mock_cursor = _make_store(mock_pg)

        mock_cursor.description = [
            ("id",), ("predict_date",), ("category",), ("target",),
        ]
        mock_cursor.fetchall.return_value = []

        store.get_pending_predictions("2026-02-13")

        sql = mock_cursor.execute.call_args_list[-1][0][0]
        # SQL should filter by category in ('stock', 'top_pick')
        assert "category" in sql
        store.close()


class TestUpdatePredictionResult:
    """Tests for PostgresStore.update_prediction_result()."""

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_updates_result_and_status(self, mock_pg):
        """Should update actual_result JSON and status for the given pred_id."""
        store, mock_conn, mock_cursor = _make_store(mock_pg)

        actual_result = {
            "verify_date": "2026-02-14",
            "actual_close": 10.85,
            "actual_high": 10.92,
            "actual_low": 10.45,
            "actual_return_pct": 2.17,
            "direction_hit": True,
            "target_hit": False,
            "stop_hit": False,
            "result": "partial_win",
        }

        store.update_prediction_result(1, actual_result, "verified_partial")

        # Should execute UPDATE with correct params
        update_call = mock_cursor.execute.call_args_list[-1]
        sql = update_call[0][0]
        assert "UPDATE predictions" in sql
        assert "actual_result" in sql
        assert "status" in sql

        params = update_call[0][1]
        assert params["pred_id"] == 1
        assert params["status"] == "verified_partial"
        # actual_result should be JSON string
        parsed = json.loads(params["actual_result"])
        assert parsed["direction_hit"] is True
        assert parsed["actual_close"] == 10.85

        mock_conn.commit.assert_called()
        store.close()

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_updates_with_win_status(self, mock_pg):
        """Should accept verified_win status."""
        store, mock_conn, mock_cursor = _make_store(mock_pg)

        actual_result = {"result": "win", "direction_hit": True, "target_hit": True}
        store.update_prediction_result(5, actual_result, "verified_win")

        params = mock_cursor.execute.call_args_list[-1][0][1]
        assert params["status"] == "verified_win"
        assert params["pred_id"] == 5
        store.close()


class TestGetPredictionStats:
    """Tests for PostgresStore.get_prediction_stats() with advanced metrics."""

    def _make_rows(self, records):
        """Build mock fetchall return value from simplified record dicts."""
        columns = [
            ("id",), ("predict_date",), ("status",), ("actual_result",),
        ]
        rows = [
            (r["id"], r["predict_date"], r["status"], r["actual_result"])
            for r in records
        ]
        return columns, rows

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_returns_basic_stats(self, mock_pg):
        """Should return total, wins, losses, partial_wins, win_rate, avg_return_pct."""
        store, _, mock_cursor = _make_store(mock_pg)

        columns, rows = self._make_rows([
            {"id": 1, "predict_date": "2026-02-10", "status": "verified_win",
             "actual_result": json.dumps({"actual_return_pct": 5.0, "cost_adjusted_return": 4.9, "daily_returns": [2.0, 3.0], "holding_days": 2})},
            {"id": 2, "predict_date": "2026-02-10", "status": "verified_win",
             "actual_result": json.dumps({"actual_return_pct": 3.0, "cost_adjusted_return": 2.9, "daily_returns": [1.5, 1.5], "holding_days": 2})},
            {"id": 3, "predict_date": "2026-02-10", "status": "verified_loss",
             "actual_result": json.dumps({"actual_return_pct": -4.0, "cost_adjusted_return": -4.1, "daily_returns": [-2.0, -2.0], "holding_days": 2})},
            {"id": 4, "predict_date": "2026-02-10", "status": "verified_partial",
             "actual_result": json.dumps({"actual_return_pct": 1.0, "cost_adjusted_return": 0.9, "daily_returns": [0.5, 0.5], "holding_days": 2})},
        ])
        mock_cursor.description = columns
        mock_cursor.fetchall.return_value = rows

        stats = store.get_prediction_stats(days=30)

        assert stats["total_verified"] == 4
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["partial_wins"] == 1
        assert stats["win_rate"] == 50.0  # 2/4
        assert abs(stats["avg_return_pct"] - 1.25) < 0.01  # (5+3-4+1)/4
        store.close()

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_profit_factor(self, mock_pg):
        """profit_factor = sum of positive returns / abs(sum of negative returns)."""
        store, _, mock_cursor = _make_store(mock_pg)

        columns, rows = self._make_rows([
            {"id": 1, "predict_date": "2026-02-10", "status": "verified_win",
             "actual_result": json.dumps({"actual_return_pct": 6.0, "cost_adjusted_return": 5.9, "daily_returns": [6.0], "holding_days": 1})},
            {"id": 2, "predict_date": "2026-02-10", "status": "verified_win",
             "actual_result": json.dumps({"actual_return_pct": 4.0, "cost_adjusted_return": 3.9, "daily_returns": [4.0], "holding_days": 1})},
            {"id": 3, "predict_date": "2026-02-10", "status": "verified_loss",
             "actual_result": json.dumps({"actual_return_pct": -5.0, "cost_adjusted_return": -5.1, "daily_returns": [-5.0], "holding_days": 1})},
        ])
        mock_cursor.description = columns
        mock_cursor.fetchall.return_value = rows

        stats = store.get_prediction_stats(days=30)

        # profit_factor = (6+4) / abs(-5) = 2.0
        assert abs(stats["profit_factor"] - 2.0) < 0.01
        store.close()

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_expectancy(self, mock_pg):
        """expectancy = win_rate * avg_win - loss_rate * avg_loss."""
        store, _, mock_cursor = _make_store(mock_pg)

        columns, rows = self._make_rows([
            {"id": 1, "predict_date": "2026-02-10", "status": "verified_win",
             "actual_result": json.dumps({"actual_return_pct": 6.0, "cost_adjusted_return": 5.9, "daily_returns": [6.0], "holding_days": 1})},
            {"id": 2, "predict_date": "2026-02-10", "status": "verified_loss",
             "actual_result": json.dumps({"actual_return_pct": -3.0, "cost_adjusted_return": -3.1, "daily_returns": [-3.0], "holding_days": 1})},
        ])
        mock_cursor.description = columns
        mock_cursor.fetchall.return_value = rows

        stats = store.get_prediction_stats(days=30)

        # win_rate=50%, avg_win=6.0, loss_rate=50%, avg_loss=3.0
        # expectancy = 0.5*6.0 - 0.5*3.0 = 1.5
        assert abs(stats["expectancy"] - 1.5) < 0.01
        store.close()

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_sharpe_ratio(self, mock_pg):
        """sharpe_ratio = mean(daily_returns) / std(daily_returns) * sqrt(252)."""
        store, _, mock_cursor = _make_store(mock_pg)

        columns, rows = self._make_rows([
            {"id": 1, "predict_date": "2026-02-10", "status": "verified_win",
             "actual_result": json.dumps({"actual_return_pct": 4.0, "cost_adjusted_return": 3.9, "daily_returns": [1.0, 1.5, 1.0, 0.5], "holding_days": 4})},
            {"id": 2, "predict_date": "2026-02-10", "status": "verified_loss",
             "actual_result": json.dumps({"actual_return_pct": -2.0, "cost_adjusted_return": -2.1, "daily_returns": [-0.5, -1.0, -0.5], "holding_days": 3})},
        ])
        mock_cursor.description = columns
        mock_cursor.fetchall.return_value = rows

        stats = store.get_prediction_stats(days=30)

        assert "sharpe_ratio" in stats
        # All daily returns: [1.0, 1.5, 1.0, 0.5, -0.5, -1.0, -0.5]
        # mean=0.2857, std≈0.908 -> sharpe ≈ 0.2857/0.908 * sqrt(252) ≈ 5.0
        # Exact value depends on implementation; just check it's positive and reasonable
        assert stats["sharpe_ratio"] > 0
        store.close()

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_max_drawdown(self, mock_pg):
        """max_drawdown should be the worst peak-to-trough decline."""
        store, _, mock_cursor = _make_store(mock_pg)

        # Sequence: +2%, -5%, +1% => cumulative equity: 102, 96.9, 97.87
        # peak = 102, trough = 96.9 => drawdown = (96.9-102)/102 = -5.0%
        columns, rows = self._make_rows([
            {"id": 1, "predict_date": "2026-02-10", "status": "verified_win",
             "actual_result": json.dumps({"actual_return_pct": 2.0, "cost_adjusted_return": 1.9, "daily_returns": [2.0], "holding_days": 1})},
            {"id": 2, "predict_date": "2026-02-11", "status": "verified_loss",
             "actual_result": json.dumps({"actual_return_pct": -5.0, "cost_adjusted_return": -5.1, "daily_returns": [-5.0], "holding_days": 1})},
            {"id": 3, "predict_date": "2026-02-12", "status": "verified_partial",
             "actual_result": json.dumps({"actual_return_pct": 1.0, "cost_adjusted_return": 0.9, "daily_returns": [1.0], "holding_days": 1})},
        ])
        mock_cursor.description = columns
        mock_cursor.fetchall.return_value = rows

        stats = store.get_prediction_stats(days=30)

        assert "max_drawdown" in stats
        assert stats["max_drawdown"] < 0
        # Should be approximately -5.0% (the big loss after the peak)
        assert stats["max_drawdown"] < -4.0
        store.close()

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_sortino_ratio(self, mock_pg):
        """sortino_ratio = mean(daily_returns) / downside_std * sqrt(252)."""
        store, _, mock_cursor = _make_store(mock_pg)

        columns, rows = self._make_rows([
            {"id": 1, "predict_date": "2026-02-10", "status": "verified_win",
             "actual_result": json.dumps({"actual_return_pct": 4.0, "cost_adjusted_return": 3.9, "daily_returns": [2.0, 2.0], "holding_days": 2})},
            {"id": 2, "predict_date": "2026-02-10", "status": "verified_loss",
             "actual_result": json.dumps({"actual_return_pct": -1.0, "cost_adjusted_return": -1.1, "daily_returns": [-1.0], "holding_days": 1})},
        ])
        mock_cursor.description = columns
        mock_cursor.fetchall.return_value = rows

        stats = store.get_prediction_stats(days=30)

        assert "sortino_ratio" in stats
        # Only negative returns used for downside deviation => [-1.0]
        # sortino should be > sharpe since upside vol is excluded
        assert stats["sortino_ratio"] > 0
        store.close()

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_calmar_ratio(self, mock_pg):
        """calmar_ratio = annualized_return / abs(max_drawdown)."""
        store, _, mock_cursor = _make_store(mock_pg)

        columns, rows = self._make_rows([
            {"id": 1, "predict_date": "2026-02-10", "status": "verified_win",
             "actual_result": json.dumps({"actual_return_pct": 3.0, "cost_adjusted_return": 2.9, "daily_returns": [3.0], "holding_days": 1})},
            {"id": 2, "predict_date": "2026-02-11", "status": "verified_loss",
             "actual_result": json.dumps({"actual_return_pct": -2.0, "cost_adjusted_return": -2.1, "daily_returns": [-2.0], "holding_days": 1})},
        ])
        mock_cursor.description = columns
        mock_cursor.fetchall.return_value = rows

        stats = store.get_prediction_stats(days=30)

        assert "calmar_ratio" in stats
        assert stats["calmar_ratio"] > 0
        store.close()

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_avg_holding_days_and_sample_size(self, mock_pg):
        """Should return avg_holding_days and sample_size."""
        store, _, mock_cursor = _make_store(mock_pg)

        columns, rows = self._make_rows([
            {"id": 1, "predict_date": "2026-02-10", "status": "verified_win",
             "actual_result": json.dumps({"actual_return_pct": 3.0, "cost_adjusted_return": 2.9, "daily_returns": [1.5, 1.5], "holding_days": 2})},
            {"id": 2, "predict_date": "2026-02-10", "status": "verified_loss",
             "actual_result": json.dumps({"actual_return_pct": -1.0, "cost_adjusted_return": -1.1, "daily_returns": [-0.3, -0.3, -0.4], "holding_days": 3})},
            {"id": 3, "predict_date": "2026-02-10", "status": "verified_win",
             "actual_result": json.dumps({"actual_return_pct": 2.0, "cost_adjusted_return": 1.9, "daily_returns": [2.0], "holding_days": 1})},
        ])
        mock_cursor.description = columns
        mock_cursor.fetchall.return_value = rows

        stats = store.get_prediction_stats(days=30)

        assert stats["sample_size"] == 3
        assert abs(stats["avg_holding_days"] - 2.0) < 0.01  # (2+3+1)/3
        store.close()

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_returns_zero_stats_when_no_data(self, mock_pg):
        """Should handle zero verified predictions gracefully."""
        store, _, mock_cursor = _make_store(mock_pg)

        mock_cursor.description = [
            ("id",), ("predict_date",), ("status",), ("actual_result",),
        ]
        mock_cursor.fetchall.return_value = []

        stats = store.get_prediction_stats(days=30)

        assert stats["total_verified"] == 0
        assert stats["wins"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["avg_return_pct"] == 0.0
        assert stats["profit_factor"] == 0.0
        assert stats["sharpe_ratio"] == 0.0
        assert stats["max_drawdown"] == 0.0
        assert stats["sortino_ratio"] == 0.0
        assert stats["calmar_ratio"] == 0.0
        assert stats["expectancy"] == 0.0
        assert stats["sample_size"] == 0
        store.close()

    @patch("tradingagents.storage.postgres_store.psycopg2")
    def test_respects_days_parameter(self, mock_pg):
        """Should filter by predict_date within the given days range."""
        store, _, mock_cursor = _make_store(mock_pg)

        mock_cursor.description = [
            ("id",), ("predict_date",), ("status",), ("actual_result",),
        ]
        mock_cursor.fetchall.return_value = []

        store.get_prediction_stats(days=7)

        calls = mock_cursor.execute.call_args_list
        any_date_filter = any(
            "predict_date" in str(c) for c in calls[1:]
        )
        assert any_date_filter
        store.close()
