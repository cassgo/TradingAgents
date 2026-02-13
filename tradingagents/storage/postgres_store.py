"""PostgreSQL storage for trading decision records."""

import json
import os
from typing import Optional

import psycopg2

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trading_decisions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    final_decision TEXT NOT NULL DEFAULT '',
    market_report TEXT NOT NULL DEFAULT '',
    sentiment_report TEXT NOT NULL DEFAULT '',
    news_report TEXT NOT NULL DEFAULT '',
    fundamentals_report TEXT NOT NULL DEFAULT '',
    inv_bull_argument TEXT NOT NULL DEFAULT '',
    inv_bear_argument TEXT NOT NULL DEFAULT '',
    inv_debate_history TEXT NOT NULL DEFAULT '',
    inv_judge_decision TEXT NOT NULL DEFAULT '',
    risk_aggressive TEXT NOT NULL DEFAULT '',
    risk_conservative TEXT NOT NULL DEFAULT '',
    risk_neutral TEXT NOT NULL DEFAULT '',
    risk_debate_history TEXT NOT NULL DEFAULT '',
    risk_judge_decision TEXT NOT NULL DEFAULT '',
    investment_plan TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, trade_date)
);
"""

_INSERT_SQL = """
INSERT INTO trading_decisions (
    ticker, trade_date, final_decision,
    market_report, sentiment_report, news_report, fundamentals_report,
    inv_bull_argument, inv_bear_argument, inv_debate_history, inv_judge_decision,
    risk_aggressive, risk_conservative, risk_neutral, risk_debate_history, risk_judge_decision,
    investment_plan
) VALUES (
    %(ticker)s, %(trade_date)s, %(final_decision)s,
    %(market_report)s, %(sentiment_report)s, %(news_report)s, %(fundamentals_report)s,
    %(inv_bull_argument)s, %(inv_bear_argument)s, %(inv_debate_history)s, %(inv_judge_decision)s,
    %(risk_aggressive)s, %(risk_conservative)s, %(risk_neutral)s, %(risk_debate_history)s, %(risk_judge_decision)s,
    %(investment_plan)s
)
ON CONFLICT (ticker, trade_date) DO UPDATE SET
    final_decision = EXCLUDED.final_decision,
    market_report = EXCLUDED.market_report,
    sentiment_report = EXCLUDED.sentiment_report,
    news_report = EXCLUDED.news_report,
    fundamentals_report = EXCLUDED.fundamentals_report,
    inv_bull_argument = EXCLUDED.inv_bull_argument,
    inv_bear_argument = EXCLUDED.inv_bear_argument,
    inv_debate_history = EXCLUDED.inv_debate_history,
    inv_judge_decision = EXCLUDED.inv_judge_decision,
    risk_aggressive = EXCLUDED.risk_aggressive,
    risk_conservative = EXCLUDED.risk_conservative,
    risk_neutral = EXCLUDED.risk_neutral,
    risk_debate_history = EXCLUDED.risk_debate_history,
    risk_judge_decision = EXCLUDED.risk_judge_decision,
    investment_plan = EXCLUDED.investment_plan,
    created_at = NOW()
RETURNING id;
"""


def _safe_get(d: dict, *keys: str, default: str = "") -> str:
    """Safely traverse nested dict keys, returning default on any miss."""
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    if isinstance(current, (list, dict)):
        return json.dumps(current, ensure_ascii=False)
    return str(current)


class PostgresStore:
    """Stores trading decision records in PostgreSQL."""

    def __init__(self, database_url: Optional[str] = None):
        url = database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise ValueError(
                "DATABASE_URL is required. Pass it as argument or set the environment variable."
            )
        self._conn = psycopg2.connect(url)
        self._cursor = self._conn.cursor()
        self._cursor.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    def save(self, ticker: str, trade_date: str, state: dict) -> int:
        """Extract fields from state and upsert into trading_decisions.

        Returns the row id.
        """
        inv = state.get("investment_debate_state", {})
        risk = state.get("risk_debate_state", {})

        # For bull argument, use current_response (the final bull argument)
        inv_bull = _safe_get(state, "investment_debate_state", "current_response")
        # For bear argument, pick last bear_history entry
        bear_history = inv.get("bear_history", [])
        inv_bear = bear_history[-1] if bear_history else ""

        # For risk positions, pick last entry from each history list
        agg_history = risk.get("aggressive_history", [])
        con_history = risk.get("conservative_history", [])
        neu_history = risk.get("neutral_history", [])

        params = {
            "ticker": ticker,
            "trade_date": trade_date,
            "final_decision": _safe_get(state, "final_trade_decision"),
            "market_report": _safe_get(state, "market_report"),
            "sentiment_report": _safe_get(state, "sentiment_report"),
            "news_report": _safe_get(state, "news_report"),
            "fundamentals_report": _safe_get(state, "fundamentals_report"),
            "inv_bull_argument": inv_bull,
            "inv_bear_argument": inv_bear if isinstance(inv_bear, str) else json.dumps(inv_bear, ensure_ascii=False),
            "inv_debate_history": json.dumps(inv.get("history", []), ensure_ascii=False),
            "inv_judge_decision": _safe_get(state, "investment_debate_state", "judge_decision"),
            "risk_aggressive": agg_history[-1] if agg_history else "",
            "risk_conservative": con_history[-1] if con_history else "",
            "risk_neutral": neu_history[-1] if neu_history else "",
            "risk_debate_history": json.dumps(risk.get("history", []), ensure_ascii=False),
            "risk_judge_decision": _safe_get(state, "risk_debate_state", "judge_decision"),
            "investment_plan": _safe_get(state, "investment_plan"),
        }

        self._cursor.execute(_INSERT_SQL, params)
        self._conn.commit()
        row = self._cursor.fetchone()
        return row[0]

    def get(self, ticker: str, trade_date: str) -> Optional[dict]:
        """Fetch a single decision record. Returns None if not found."""
        self._cursor.execute(
            "SELECT * FROM trading_decisions WHERE ticker = %s AND trade_date = %s",
            (ticker, trade_date),
        )
        row = self._cursor.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in self._cursor.description]
        return dict(zip(columns, row))

    def list_decisions(self, ticker: Optional[str] = None, limit: int = 50) -> list[dict]:
        """List decision records, optionally filtered by ticker."""
        if ticker:
            self._cursor.execute(
                "SELECT * FROM trading_decisions WHERE ticker = %s ORDER BY trade_date DESC LIMIT %s",
                (ticker, limit),
            )
        else:
            self._cursor.execute(
                "SELECT * FROM trading_decisions ORDER BY trade_date DESC LIMIT %s",
                (limit,),
            )
        rows = self._cursor.fetchall()
        columns = [desc[0] for desc in self._cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    # ------------------------------------------------------------------
    # News & signal persistence (Phase 2)
    # ------------------------------------------------------------------

    _INSERT_NEWS_SQL = """
    INSERT INTO news_collected (source, title, content, publish_date)
    VALUES (%(source)s, %(title)s, %(content)s, %(publish_date)s)
    ON CONFLICT (source, title) DO NOTHING;
    """

    _INSERT_SIGNAL_SQL = """
    INSERT INTO industry_signals
        (signal_date, event, industries, sentiment, confidence, reasoning, selected_tickers)
    VALUES
        (%(signal_date)s, %(event)s, %(industries)s, %(sentiment)s,
         %(confidence)s, %(reasoning)s, %(selected_tickers)s);
    """

    def save_news(self, articles: list[dict]) -> int:
        """Persist collected news articles to news_collected table.

        Returns the number of rows inserted (duplicates skipped).
        """
        if not articles:
            return 0

        for article in articles:
            pub_date = article.get("publish_date") or None
            self._cursor.execute(self._INSERT_NEWS_SQL, {
                "source": article.get("source", ""),
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "publish_date": pub_date if pub_date else None,
            })

        self._conn.commit()
        return len(articles)

    def save_signals(self, signal_date: str, signals: list[dict]) -> int:
        """Persist industry signals to industry_signals table.

        Returns the number of rows inserted.
        """
        if not signals:
            return 0

        for signal in signals:
            industries = signal.get("industries", [])
            selected = signal.get("selected_tickers", [])
            self._cursor.execute(self._INSERT_SIGNAL_SQL, {
                "signal_date": signal_date,
                "event": signal.get("event", ""),
                "industries": json.dumps(industries, ensure_ascii=False),
                "sentiment": signal.get("sentiment", ""),
                "confidence": signal.get("confidence", 0),
                "reasoning": signal.get("reasoning", ""),
                "selected_tickers": json.dumps(selected, ensure_ascii=False),
            })

        self._conn.commit()
        return len(signals)

    # ------------------------------------------------------------------
    # Read queries for API layer
    # ------------------------------------------------------------------

    def list_news(
        self,
        date: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List news articles, optionally filtered by date and source."""
        conditions: list[str] = []
        params: list = []

        if date:
            conditions.append("publish_date = %s")
            params.append(date)
        if source:
            conditions.append("source = %s")
            params.append(source)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        self._cursor.execute(
            f"SELECT * FROM news_collected{where} ORDER BY publish_date DESC LIMIT %s",
            params,
        )
        rows = self._cursor.fetchall()
        columns = [desc[0] for desc in self._cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def list_signals(
        self,
        date: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """List industry signals, optionally filtered by date."""
        if date:
            self._cursor.execute(
                "SELECT * FROM industry_signals WHERE signal_date = %s ORDER BY created_at DESC LIMIT %s",
                (date, limit),
            )
        else:
            self._cursor.execute(
                "SELECT * FROM industry_signals ORDER BY signal_date DESC LIMIT %s",
                (limit,),
            )
        rows = self._cursor.fetchall()
        columns = [desc[0] for desc in self._cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    # ------------------------------------------------------------------
    # Watchlist CRUD
    # ------------------------------------------------------------------

    def list_watchlist(self) -> list[dict]:
        """Return all watchlist entries ordered by creation time."""
        self._cursor.execute(
            "SELECT * FROM watchlist ORDER BY created_at DESC"
        )
        rows = self._cursor.fetchall()
        columns = [desc[0] for desc in self._cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def add_watchlist(
        self,
        ticker: str,
        name: str = "",
        industry: str = "",
    ) -> int:
        """Add a stock to the watchlist. Returns the row id."""
        self._cursor.execute(
            "INSERT INTO watchlist (ticker, name, industry) VALUES (%(ticker)s, %(name)s, %(industry)s) "
            "ON CONFLICT (ticker) DO UPDATE SET name = EXCLUDED.name, industry = EXCLUDED.industry "
            "RETURNING id",
            {"ticker": ticker, "name": name, "industry": industry},
        )
        self._conn.commit()
        row = self._cursor.fetchone()
        return row[0]

    def update_watchlist(self, ticker: str, active: bool) -> bool:
        """Enable or disable a watchlist entry. Returns True if updated."""
        self._cursor.execute(
            "UPDATE watchlist SET active = %(active)s WHERE ticker = %(ticker)s",
            {"ticker": ticker, "active": active},
        )
        self._conn.commit()
        return self._cursor.rowcount > 0

    def remove_watchlist(self, ticker: str) -> bool:
        """Delete a watchlist entry. Returns True if deleted."""
        self._cursor.execute(
            "DELETE FROM watchlist WHERE ticker = %s",
            (ticker,),
        )
        self._conn.commit()
        return self._cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Dashboard aggregation
    # ------------------------------------------------------------------

    def get_dashboard_stats(self) -> dict:
        """Return aggregated statistics for the dashboard."""
        self._cursor.execute("SELECT COUNT(*) FROM trading_decisions")
        total_decisions = self._cursor.fetchone()[0]

        self._cursor.execute("SELECT COUNT(*) FROM news_collected")
        total_news = self._cursor.fetchone()[0]

        self._cursor.execute("SELECT COUNT(*) FROM industry_signals")
        total_signals = self._cursor.fetchone()[0]

        self._cursor.execute("SELECT COUNT(*) FROM watchlist WHERE active = TRUE")
        active_watchlist = self._cursor.fetchone()[0]

        return {
            "total_decisions": total_decisions,
            "total_news": total_news,
            "total_signals": total_signals,
            "active_watchlist": active_watchlist,
        }

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def save_predictions(self, predictions: list[dict]) -> int:
        """Persist prediction records. Returns count inserted."""
        if not predictions:
            return 0
        for p in predictions:
            self._cursor.execute(
                """
                INSERT INTO predictions
                    (predict_date, trade_date, category, target, direction,
                     confidence, reasoning, entry_price, target_price, stop_loss,
                     time_horizon, status)
                VALUES
                    (%(predict_date)s, %(trade_date)s, %(category)s, %(target)s, %(direction)s,
                     %(confidence)s, %(reasoning)s, %(entry_price)s, %(target_price)s, %(stop_loss)s,
                     %(time_horizon)s, %(status)s)
                """,
                {
                    "predict_date": p.get("predict_date", ""),
                    "trade_date": p.get("trade_date", ""),
                    "category": p.get("category", "sector"),
                    "target": p.get("target", ""),
                    "direction": p.get("direction", ""),
                    "confidence": p.get("confidence", 0),
                    "reasoning": p.get("reasoning", ""),
                    "entry_price": p.get("entry_price", ""),
                    "target_price": p.get("target_price", ""),
                    "stop_loss": p.get("stop_loss", ""),
                    "time_horizon": p.get("time_horizon", ""),
                    "status": p.get("status", "pending"),
                },
            )
        self._conn.commit()
        return len(predictions)

    def list_predictions(self, date: Optional[str] = None, category: Optional[str] = None, limit: int = 50) -> list[dict]:
        """List predictions, optionally filtered by date and category."""
        conditions: list[str] = []
        params: list = []
        if date:
            conditions.append("predict_date = %s")
            params.append(date)
        if category:
            conditions.append("category = %s")
            params.append(category)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        self._cursor.execute(
            f"SELECT * FROM predictions{where} ORDER BY confidence DESC, created_at DESC LIMIT %s",
            params,
        )
        rows = self._cursor.fetchall()
        columns = [desc[0] for desc in self._cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_pending_predictions(self, before_date: str) -> list[dict]:
        """Query pending stock/top_pick predictions before a given date."""
        self._cursor.execute(
            "SELECT * FROM predictions "
            "WHERE status = %s AND predict_date < %s AND category IN (%s, %s) "
            "ORDER BY predict_date ASC",
            ("pending", before_date, "stock", "top_pick"),
        )
        rows = self._cursor.fetchall()
        columns = [desc[0] for desc in self._cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def update_prediction_result(self, pred_id: int, actual_result: dict, status: str) -> None:
        """Update a prediction's actual_result (JSON) and status."""
        self._cursor.execute(
            "UPDATE predictions SET actual_result = %(actual_result)s, status = %(status)s "
            "WHERE id = %(pred_id)s",
            {
                "pred_id": pred_id,
                "actual_result": json.dumps(actual_result, ensure_ascii=False),
                "status": status,
            },
        )
        self._conn.commit()

    def get_prediction_stats(self, days: int = 30) -> dict:
        """Compute prediction accuracy and advanced risk metrics for the last N days.

        Fetches all verified records in one query, then computes metrics in Python:
        basic (win_rate, avg_return), profit_factor, expectancy, sharpe_ratio,
        sortino_ratio, calmar_ratio, max_drawdown, avg_holding_days, sample_size.
        """
        import math

        _ZERO_STATS = {
            "total_verified": 0, "wins": 0, "losses": 0, "partial_wins": 0,
            "win_rate": 0.0, "avg_return_pct": 0.0,
            "profit_factor": 0.0, "expectancy": 0.0,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
            "calmar_ratio": 0.0, "max_drawdown": 0.0,
            "avg_holding_days": 0.0, "sample_size": 0,
        }

        self._cursor.execute(
            "SELECT id, predict_date, status, actual_result "
            "FROM predictions "
            "WHERE status != 'pending' "
            "AND predict_date >= CURRENT_DATE - INTERVAL '%s days' "
            "ORDER BY predict_date ASC",
            (days,),
        )
        columns = [desc[0] for desc in self._cursor.description] if self._cursor.description else []
        rows = self._cursor.fetchall()

        if not rows or not columns:
            return dict(_ZERO_STATS)

        records = [dict(zip(columns, row)) for row in rows]
        total = len(records)

        wins = sum(1 for r in records if r["status"] == "verified_win")
        losses = sum(1 for r in records if r["status"] == "verified_loss")
        partial_wins = sum(1 for r in records if r["status"] == "verified_partial")
        win_rate = round(wins / total * 100, 1)

        # Parse actual_result JSON for each record
        returns = []
        all_daily_returns = []
        holding_days_list = []
        for r in records:
            ar = r.get("actual_result")
            if not ar:
                continue
            if isinstance(ar, str):
                ar = json.loads(ar)
            ret = ar.get("actual_return_pct", 0.0)
            returns.append(ret)
            dr = ar.get("daily_returns", [])
            if dr:
                all_daily_returns.extend(dr)
            hd = ar.get("holding_days")
            if hd is not None:
                holding_days_list.append(hd)

        avg_return_pct = round(sum(returns) / len(returns), 2) if returns else 0.0

        # Profit Factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0

        # Expectancy
        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r < 0]
        avg_win = sum(winning) / len(winning) if winning else 0.0
        avg_loss = abs(sum(losing) / len(losing)) if losing else 0.0
        wr = wins / total if total > 0 else 0
        lr = 1 - wr
        expectancy = round(wr * avg_win - lr * avg_loss, 2)

        # Sharpe Ratio (annualized, using all daily returns)
        sharpe_ratio = 0.0
        if len(all_daily_returns) >= 2:
            mean_dr = sum(all_daily_returns) / len(all_daily_returns)
            var_dr = sum((x - mean_dr) ** 2 for x in all_daily_returns) / (len(all_daily_returns) - 1)
            std_dr = math.sqrt(var_dr) if var_dr > 0 else 0
            if std_dr > 0:
                sharpe_ratio = round(mean_dr / std_dr * math.sqrt(252), 2)

        # Max Drawdown (from per-trade returns in chronological order)
        max_drawdown = 0.0
        if returns:
            equity = 100.0
            peak = equity
            for ret in returns:
                equity *= (1 + ret / 100)
                if equity > peak:
                    peak = equity
                dd = (equity - peak) / peak * 100
                if dd < max_drawdown:
                    max_drawdown = dd
            max_drawdown = round(max_drawdown, 2)

        # Sortino Ratio (uses downside deviation only)
        sortino_ratio = 0.0
        if len(all_daily_returns) >= 2:
            mean_dr = sum(all_daily_returns) / len(all_daily_returns)
            neg_returns = [r for r in all_daily_returns if r < 0]
            if neg_returns:
                down_var = sum(r ** 2 for r in neg_returns) / len(neg_returns)
                down_std = math.sqrt(down_var)
                if down_std > 0:
                    sortino_ratio = round(mean_dr / down_std * math.sqrt(252), 2)

        # Calmar Ratio (annualized return / abs(max_drawdown))
        calmar_ratio = 0.0
        if max_drawdown < 0 and all_daily_returns:
            mean_dr = sum(all_daily_returns) / len(all_daily_returns)
            annualized_return = mean_dr * 252
            calmar_ratio = round(annualized_return / abs(max_drawdown), 2)

        avg_holding_days = (
            round(sum(holding_days_list) / len(holding_days_list), 1)
            if holding_days_list else 0.0
        )

        return {
            "total_verified": total,
            "wins": wins,
            "losses": losses,
            "partial_wins": partial_wins,
            "win_rate": win_rate,
            "avg_return_pct": avg_return_pct,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "avg_holding_days": avg_holding_days,
            "sample_size": total,
        }

    # ------------------------------------------------------------------
    # Application config (single-row JSONB)
    # ------------------------------------------------------------------

    _ENSURE_CONFIG_TABLE = """
    CREATE TABLE IF NOT EXISTS app_config (
        id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
        data JSONB NOT NULL DEFAULT '{}',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    def load_config(self) -> dict:
        """Load persisted config from app_config table. Returns {} if empty."""
        try:
            self._cursor.execute(self._ENSURE_CONFIG_TABLE)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
        self._cursor.execute("SELECT data FROM app_config WHERE id = 1")
        row = self._cursor.fetchone()
        return row[0] if row else {}

    def save_config(self, data: dict) -> None:
        """Upsert the full config JSON into app_config."""
        self._cursor.execute(
            """
            INSERT INTO app_config (id, data, updated_at)
            VALUES (1, %s, NOW())
            ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data, updated_at = NOW()
            """,
            (json.dumps(data, ensure_ascii=False),),
        )
        self._conn.commit()

    def close(self):
        """Close cursor and connection."""
        self._cursor.close()
        self._conn.close()
