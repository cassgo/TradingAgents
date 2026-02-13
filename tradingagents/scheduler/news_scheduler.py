"""APScheduler-based news collection scheduler.

Two tiers:
  - Full collection: 5x/day at key news windows (07:30, 11:40, 15:15, 19:30, 22:00)
  - Watchlist monitoring: Every 15 min during trading hours (09:30-11:30, 13:00-15:00)

Only runs on weekdays (Mon-Fri).
"""

import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _get_store():
    """Lazy-import and create a PostgresStore for the scheduler thread."""
    from tradingagents.storage.postgres_store import PostgresStore
    return PostgresStore()


def _get_watchlist_from_db() -> list[dict]:
    """Load active watchlist stocks from database."""
    try:
        store = _get_store()
        items = store.list_watchlist()
        return [
            {"code": item["ticker"], "name": item["name"]}
            for item in items
            if item.get("active", True)
        ]
    except Exception:
        logger.warning("Failed to load watchlist from DB", exc_info=True)
        return []


def job_full_collection():
    """Full sweep: all news sources → DB."""
    from tradingagents.collector.news_collector import NewsCollector

    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("[Scheduler] Full collection starting for %s", today)

    try:
        collector = NewsCollector()
        articles = collector.collect_daily(today)
        logger.info("[Scheduler] Collected %d articles", len(articles))

        if articles:
            store = _get_store()
            store.save_news(articles)
            logger.info("[Scheduler] Saved %d articles to DB", len(articles))

        # Broadcast event via WebSocket
        _broadcast_event("collection:full_done", {
            "count": len(articles),
            "time": datetime.now().isoformat(),
        })
    except Exception:
        logger.error("[Scheduler] Full collection failed", exc_info=True)


def job_watchlist_monitor():
    """Watchlist monitoring: individual stock news → DB."""
    from tradingagents.collector.news_collector import NewsCollector

    tickers = _get_watchlist_from_db()
    if not tickers:
        logger.debug("[Scheduler] No active watchlist stocks, skipping")
        return

    logger.info("[Scheduler] Watchlist monitor: %d stocks", len(tickers))

    try:
        collector = NewsCollector()
        articles = collector.collect_watchlist(tickers)
        logger.info("[Scheduler] Watchlist collected %d articles", len(articles))

        if articles:
            store = _get_store()
            store.save_news(articles)
            logger.info("[Scheduler] Saved %d watchlist articles to DB", len(articles))

        _broadcast_event("collection:watchlist_done", {
            "count": len(articles),
            "tickers": [t["code"] for t in tickers],
            "time": datetime.now().isoformat(),
        })
    except Exception:
        logger.error("[Scheduler] Watchlist monitor failed", exc_info=True)


def job_verify_predictions():
    """Verify pending predictions against actual market data."""
    logger.info("[Scheduler] Prediction verification starting")
    try:
        from tradingagents.api.routers.predictions import verify_predictions

        store = _get_store()
        result = verify_predictions(store)
        logger.info(
            "[Scheduler] Verification done: %d verified, %d skipped, %d errors",
            result.get("verified", 0),
            result.get("skipped", 0),
            result.get("errors", 0),
        )
        _broadcast_event("verification:done", {
            "time": datetime.now().isoformat(),
            **result,
        })
    except Exception:
        logger.error("[Scheduler] Prediction verification failed", exc_info=True)


def _broadcast_event(event_type: str, payload: dict):
    """Best-effort WebSocket broadcast (won't fail if WS not available)."""
    try:
        from tradingagents.api.ws import broadcast
        broadcast({"event": event_type, **payload})
    except Exception:
        pass


def start_scheduler() -> BackgroundScheduler:
    """Create and start the news collection scheduler."""
    global _scheduler

    if _scheduler is not None:
        logger.warning("[Scheduler] Already running, skipping start")
        return _scheduler

    scheduler = BackgroundScheduler(
        timezone="Asia/Shanghai",
        job_defaults={"coalesce": True, "max_instances": 1},
    )

    # --- Tier 1: Full collection (5x/day, weekdays only) ---
    # 07:30 - Pre-market: overnight news, policy announcements
    # 11:40 - Morning session wrap-up (after 11:30 close)
    # 15:15 - Post-market: closing data, fund flows
    # 19:30 - Evening: CCTV news + policy releases
    # 22:00 - Night recap: late releases + US pre-market
    # APScheduler CronTrigger with multiple hour+minute uses cartesian product,
    # so we add individual jobs for each time slot.
    full_times = [
        ("07:30", 7, 30),
        ("11:40", 11, 40),
        ("15:15", 15, 15),
        ("19:30", 19, 30),
        ("22:00", 22, 0),
    ]
    for label, hour, minute in full_times:
        scheduler.add_job(
            job_full_collection,
            CronTrigger(hour=hour, minute=minute, day_of_week="mon-fri", timezone="Asia/Shanghai"),
            id=f"full_collection_{label}",
            name=f"Full collection at {label}",
            replace_existing=True,
        )

    # --- Tier 1.5: Prediction verification (16:30, after market close) ---
    scheduler.add_job(
        job_verify_predictions,
        CronTrigger(hour=16, minute=30, day_of_week="mon-fri", timezone="Asia/Shanghai"),
        id="verify_predictions",
        name="Verify predictions at 16:30",
        replace_existing=True,
    )

    # --- Tier 2: Watchlist monitoring (every 15 min during trading hours) ---
    # Morning session: 09:30 - 11:30
    scheduler.add_job(
        job_watchlist_monitor,
        CronTrigger(
            hour="9-11", minute="*/15",
            day_of_week="mon-fri", timezone="Asia/Shanghai",
        ),
        id="watchlist_monitor_am",
        name="Watchlist monitor (AM session)",
        replace_existing=True,
    )
    # Afternoon session: 13:00 - 15:00
    scheduler.add_job(
        job_watchlist_monitor,
        CronTrigger(
            hour="13-14", minute="*/15",
            day_of_week="mon-fri", timezone="Asia/Shanghai",
        ),
        id="watchlist_monitor_pm",
        name="Watchlist monitor (PM session)",
        replace_existing=True,
    )

    scheduler.start()
    _scheduler = scheduler

    logger.info("[Scheduler] Started with %d jobs", len(scheduler.get_jobs()))
    for job in scheduler.get_jobs():
        logger.info("  Job: %s | Next run: %s", job.name, job.next_run_time)

    return scheduler


def stop_scheduler():
    """Gracefully shut down the scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("[Scheduler] Stopped")


def get_scheduler_status() -> dict:
    """Return current scheduler status and upcoming jobs."""
    if _scheduler is None:
        return {"running": False, "jobs": []}

    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": str(job.next_run_time) if job.next_run_time else None,
        })

    return {"running": True, "jobs": jobs}
