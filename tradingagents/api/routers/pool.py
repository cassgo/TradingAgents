"""Prediction pool API router (unified system)."""

import logging
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from tradingagents.api.deps import get_store, get_config
from tradingagents.utils.pipeline_queue import get_pipeline_queue
from tradingagents.pool.quick_analyzer import QuickAnalyzer
from tradingagents.pool.verifier import verify_predictions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pool", tags=["pool"])


# -- Request models --

class ExitPoolRequest(BaseModel):
    exit_date: str
    exit_price: float
    exit_reason: str = ""


# -- Internal helpers --

def _run_predict(store, config: dict) -> dict:
    """Run daily prediction (called in background thread too)."""
    from tradingagents.pool.daily_predictor import DailyPredictor
    from tradingagents.api.llm_utils import get_llm

    llm = get_llm(config)
    predictor = DailyPredictor(store=store, llm=llm, config=config)
    today = datetime.now().strftime("%Y-%m-%d")
    return predictor.predict_all(today)


def _run_evaluate(store, config: dict) -> dict:
    """Run evaluation (called in background thread too)."""
    from tradingagents.pool.evaluator import PoolEvaluator
    from tradingagents.api.llm_utils import get_llm

    llm = get_llm(config)
    evaluator = PoolEvaluator(store=store, llm=llm, config=config)
    today = datetime.now().strftime("%Y-%m-%d")
    return evaluator.evaluate(today)


# -- Endpoints --

@router.get("/stocks")
def list_pool_stocks(
    status: str = Query("active"),
    limit: int = Query(100),
    store=Depends(get_store),
):
    return store.list_pool_stocks(status=status, limit=limit)


@router.get("/stocks/{pool_id}")
def get_pool_stock(pool_id: int, store=Depends(get_store)):
    result = store.get_pool_stock(pool_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Pool stock not found")
    return result


@router.post("/stocks/{pool_id}/exit")
def exit_pool_stock(pool_id: int, body: ExitPoolRequest, store=Depends(get_store)):
    success = store.exit_pool_stock(
        pool_id=pool_id,
        exit_date=body.exit_date,
        exit_price=body.exit_price,
        exit_reason=body.exit_reason,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Pool stock not found")
    return {"status": "exited", "pool_id": pool_id}


@router.get("/predictions")
def list_pool_predictions(
    predict_date: str | None = Query(None),
    pool_id: int | None = Query(None),
    store=Depends(get_store),
):
    return store.get_pool_predictions(predict_date=predict_date, pool_id=pool_id)


@router.get("/stats")
def get_pool_stats(store=Depends(get_store)):
    return store.get_pool_stats()


@router.get("/lessons")
def list_pool_lessons(
    pool_id: int | None = Query(None),
    limit: int = Query(50),
    store=Depends(get_store),
):
    return store.list_pool_lessons(pool_id=pool_id, limit=limit)


@router.get("/screening-runs")
def list_screening_runs(
    limit: int = Query(10),
    store=Depends(get_store),
):
    return store.list_screening_runs(limit=limit)


@router.get("/screening-runs/{run_id}/candidates")
def get_screening_candidates(
    run_id: int,
    stage: int | None = Query(None),
    store=Depends(get_store),
):
    return store.get_screening_candidates(run_id=run_id, stage=stage)


@router.get("/strategies")
def list_strategies(
    limit: int = Query(10),
    store=Depends(get_store),
):
    return store.list_strategy_versions(limit=limit)


@router.post("/trigger/predict")
def trigger_predict(store=Depends(get_store), config=Depends(get_config)):
    result = _run_predict(store, config)
    return result


@router.post("/trigger/evaluate")
def trigger_evaluate(store=Depends(get_store), config=Depends(get_config)):
    result = _run_evaluate(store, config)
    return result


@router.get("/strategies/active")
def get_active_strategy(store=Depends(get_store)):
    result = store.get_active_strategy()
    if result is None:
        return {"active": False, "message": "No active strategy"}
    return {"active": True, **result}


@router.post("/trigger/evolve")
def trigger_evolve(store=Depends(get_store), config=Depends(get_config)):
    """Trigger strategy evolution."""
    from tradingagents.pool.strategy_manager import StrategyManager
    from tradingagents.api.llm_utils import get_llm

    llm = get_llm(config)
    manager = StrategyManager(store=store, llm=llm, config=config)
    today = datetime.now().strftime("%Y-%m-%d")
    result = manager.maybe_evolve(today)
    if result:
        return {"evolved": True, **result}
    return {"evolved": False, "message": "Conditions not met for evolution"}


@router.post("/trigger/exit-check")
def trigger_exit_check(store=Depends(get_store), config=Depends(get_config)):
    """Check and execute auto-exits for active pool stocks."""
    from tradingagents.pool.auto_exit import AutoExitChecker

    checker = AutoExitChecker(store=store, config=config)
    today = datetime.now().strftime("%Y-%m-%d")
    exits = checker.check_and_execute(today)
    return {
        "checked": True,
        "exits": len(exits),
        "details": exits,
    }


@router.post("/trigger/sentiment")
def trigger_sentiment(store=Depends(get_store)):
    """Trigger batch sentiment analysis for unsentimented news."""
    from tradingagents.finbert.model_manager import FinBERTModelManager
    from tradingagents.finbert.sentiment import SentimentAnalyzer

    manager = FinBERTModelManager.get_instance()
    if not manager.ensure_labeler():
        return {"analyzed": 0, "message": "FinBERT2 labeler not available"}

    articles = store.get_unsentimented_news(limit=200)
    if not articles:
        return {"analyzed": 0, "message": "No unsentimented news"}

    analyzer = SentimentAnalyzer(manager)
    results = analyzer.analyze_news(articles)
    updates = [
        {"id": r["id"], "label": r["sentiment_label"], "score": r["sentiment_score"]}
        for r in results
    ]
    count = store.batch_update_news_sentiment(updates)
    return {"analyzed": count}


@router.get("/sentiment-stats")
def get_sentiment_stats(
    date: str | None = Query(None),
    store=Depends(get_store),
):
    """Get sentiment statistics for a given date (defaults to today)."""
    target_date = date or datetime.now().strftime("%Y-%m-%d")
    return store.get_sentiment_stats(target_date)


@router.post("/trigger/sentiment-screen")
def trigger_sentiment_screen(config=Depends(get_config)):
    """Run S0+S1 sentiment pre-screening in background queue."""
    def _run(config):
        from tradingagents.pool.pipeline import PoolPipeline
        from tradingagents.storage.postgres_store import PostgresStore
        store = PostgresStore()
        try:
            pipeline = PoolPipeline(store=store, config=config)
            today = datetime.now().strftime("%Y-%m-%d")
            return pipeline.run_sentiment_prescreen(today)
        finally:
            store.close()

    queue = get_pipeline_queue()
    task_id = queue.submit(_run, config)
    return {"task_id": task_id, "status": "queued"}


@router.post("/trigger/llm-screen")
def trigger_llm_screen(
    date: str | None = Query(None),
    config=Depends(get_config),
):
    """Run S0+S1+S2 pipeline in background queue."""
    def _run(config, run_date):
        from tradingagents.pool.pipeline import PoolPipeline
        from tradingagents.api.llm_utils import get_llm
        from tradingagents.storage.postgres_store import PostgresStore
        store = PostgresStore()
        try:
            llm = get_llm(config)
            pipeline = PoolPipeline(store=store, config=config, llm=llm)
            return pipeline.run_llm_screen(run_date)
        finally:
            store.close()

    queue = get_pipeline_queue()
    task_id = queue.submit(_run, config, date)
    return {"task_id": task_id, "status": "queued"}


@router.post("/trigger/stage/{stage}")
def trigger_stage(
    stage: int,
    resume: bool = Query(False),
    store=Depends(get_store),
    config=Depends(get_config),
):
    """Trigger pipeline from a specific stage (1-4), using last run's candidates."""
    if stage < 1 or stage > 4:
        raise HTTPException(status_code=400, detail="stage must be 1-4")

    # Pre-validate: check that previous run has saved candidates
    runs = store.list_screening_runs(limit=1)
    if not runs:
        raise HTTPException(status_code=400, detail="没有历史筛选记录，请先运行全量筛选")
    prev_run = runs[0]
    prev_stage = stage - 1
    candidates = store.get_screening_candidates(prev_run["id"], stage=prev_stage)
    if not candidates:
        stage_names = {0: "S0", 1: "S1", 2: "S2", 3: "S3"}
        raise HTTPException(
            status_code=400,
            detail=f"上一次筛选（#{prev_run['id']}）没有保存{stage_names.get(prev_stage, f'S{prev_stage}')}候选人数据，请先运行一次全量筛选",
        )

    def _run(config, stage, resume):
        from tradingagents.pool.pipeline import PoolPipeline
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.storage.postgres_store import PostgresStore
        store = PostgresStore()
        try:
            graph = TradingAgentsGraph(config=config)
            pipeline = PoolPipeline(store=store, config=config, graph=graph)
            today = datetime.now().strftime("%Y-%m-%d")
            return pipeline.run_stage(stage, today, resume=resume)
        finally:
            store.close()

    queue = get_pipeline_queue()
    task_id = queue.submit(_run, config, stage, resume)
    return {"task_id": task_id, "status": "queued", "from_stage": stage, "resume": resume}


@router.post("/trigger/screen")
def trigger_screen(config=Depends(get_config)):
    """Trigger full screening pipeline (S0->S4) in background queue."""
    def _run(config):
        from tradingagents.pool.pipeline import PoolPipeline
        from tradingagents.api.llm_utils import get_llm
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.storage.postgres_store import PostgresStore
        store = PostgresStore()
        try:
            llm = get_llm(config)
            graph = TradingAgentsGraph(config=config)
            pipeline = PoolPipeline(store=store, config=config, llm=llm, graph=graph)
            today = datetime.now().strftime("%Y-%m-%d")
            return pipeline.run_full_pipeline(today)
        finally:
            store.close()

    queue = get_pipeline_queue()
    task_id = queue.submit(_run, config)
    return {"task_id": task_id, "status": "queued"}


# ---------------------------------------------------------------------------
# Pipeline status endpoints
# ---------------------------------------------------------------------------

@router.get("/pipeline-status/{task_id}")
def get_pipeline_status(task_id: str):
    """Get status of a pipeline task."""
    queue = get_pipeline_queue()
    return queue.get_status(task_id)


@router.post("/cancel/{run_id}")
def cancel_screening_run(run_id: int, store=Depends(get_store)):
    """Cancel a running or queued screening run."""
    runs = store.list_screening_runs(limit=50)
    run = next((r for r in runs if r["id"] == run_id), None)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Screening run #{run_id} not found")

    # Cancel the pipeline queue task if there's one running
    task_id = run.get("task_id")
    if task_id:
        queue = get_pipeline_queue()
        queue.cancel(task_id)

    # Mark the run as cancelled in DB
    store.update_screening_run(run_id, status="cancelled")
    return {"status": "cancelled", "run_id": run_id}


@router.get("/pipeline-tasks")
def list_pipeline_tasks(limit: int = Query(10)):
    """List recent pipeline tasks."""
    queue = get_pipeline_queue()
    return queue.list_tasks(limit=limit)


# ---------------------------------------------------------------------------
# Quick Analysis endpoints (Phase 1: unified pool system)
# ---------------------------------------------------------------------------

@router.post("/trigger/quick-analysis")
def trigger_quick_analysis(store=Depends(get_store), config=Depends(get_config)):
    """Run 3-round quick analysis: sector -> stock K-line -> portfolio."""
    from tradingagents.api.llm_utils import get_llm

    llm = get_llm(config)
    analyzer = QuickAnalyzer(store=store, llm=llm, config=config)
    today = datetime.now().strftime("%Y-%m-%d")
    return analyzer.analyze(today)


@router.get("/quick-analysis")
def list_quick_analysis(
    date: str | None = Query(None),
    store=Depends(get_store),
):
    """List quick analysis results (predictions table, all categories)."""
    return store.list_predictions(date=date, limit=100)


@router.post("/promote/{prediction_id}")
def promote_to_pool(prediction_id: int, store=Depends(get_store)):
    """Promote a quick analysis prediction to a pool stock."""
    try:
        pool_id = store.promote_prediction_to_pool(prediction_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    pool_stock = store.get_pool_stock(pool_id)
    return {"pool_id": pool_id, "stock": pool_stock}


@router.post("/trigger/verify")
def trigger_pool_verify(store=Depends(get_store)):
    """Verify pending predictions against actual market data."""
    try:
        result = verify_predictions(store)
        return result
    except Exception:
        logger.error("Pool verification failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction verification failed")


# ---------------------------------------------------------------------------
# Re-Debate endpoints (Phase 2: conditional re-debate)
# ---------------------------------------------------------------------------

@router.post("/trigger/redebate-check")
def trigger_redebate_check(store=Depends(get_store), config=Depends(get_config)):
    """Check all active pool stocks for re-debate conditions."""
    from tradingagents.pool.debate_trigger import PoolDebateTrigger

    trigger = PoolDebateTrigger(config)
    active_stocks = store.list_pool_stocks(status="active")

    today = datetime.now().date()
    triggered = []

    for stock in active_stocks:
        klines = store.get_stock_daily(
            code=stock["code"], limit=1,
        )
        if not klines:
            continue

        latest = klines[0]
        current_price = float(latest["close"])
        daily_change_pct = float(latest.get("change_pct", 0))

        should, reasons = trigger.should_trigger(
            stock, current_price, daily_change_pct, today,
        )
        if should:
            triggered.append({
                "pool_id": stock["id"],
                "code": stock["code"],
                "reasons": reasons,
                "current_price": current_price,
            })

    return {"checked": len(active_stocks), "triggered": len(triggered), "details": triggered}


@router.post("/stocks/{pool_id}/trigger-debate")
def trigger_pool_debate(pool_id: int, store=Depends(get_store)):
    """Manually trigger debate for a pool stock."""
    from tradingagents.api.tasks import start_pool_debate

    stock = store.get_pool_stock(pool_id)
    if not stock:
        raise HTTPException(status_code=404, detail=f"Pool stock not found: {pool_id}")

    task_id = start_pool_debate(pool_id, stock["code"], str(stock["entry_date"]), store)
    store.update_pool_debate_status(pool_id, task_id=task_id, status="running")

    return {"pool_id": pool_id, "debate_status": "running", "task_id": task_id}


@router.post("/stocks/{pool_id}/retry-debate")
def retry_pool_debate(pool_id: int, store=Depends(get_store)):
    """Retry a failed debate for a pool stock."""
    from tradingagents.api.tasks import start_pool_debate

    stock = store.get_pool_stock(pool_id)
    if not stock:
        raise HTTPException(status_code=404, detail=f"Pool stock not found: {pool_id}")

    task_id = start_pool_debate(pool_id, stock["code"], str(stock["entry_date"]), store)
    store.update_pool_debate_status(pool_id, task_id=task_id, status="running")

    return {"pool_id": pool_id, "debate_status": "running", "task_id": task_id}


# ---------------------------------------------------------------------------
# Experience Loop endpoints (Phase 3)
# ---------------------------------------------------------------------------

@router.get("/summaries")
def list_pool_summaries(
    limit: int = Query(10),
    store=Depends(get_store),
):
    """List pool summaries."""
    return store.list_pool_summaries(limit=limit)


@router.post("/trigger/summarize")
def trigger_pool_summarize(store=Depends(get_store), config=Depends(get_config)):
    """Generate LLM summary from exited pool stocks."""
    import json as _json
    from tradingagents.api.llm_utils import get_llm

    llm = get_llm(config)
    exited = store.list_pool_stocks(status="exited")
    if not exited:
        return {"summary_id": None, "message": "No exited stocks to summarize"}

    today = datetime.now().strftime("%Y-%m-%d")
    stocks_text = "\n".join(
        f"- {s['code']}: entry {s.get('entry_price')} -> exit {s.get('exit_price', 'N/A')}, "
        f"return {s.get('return_pct', 0)}%, reason: {s.get('exit_reason', 'N/A')}"
        for s in exited
    )
    prompt = (
        f"分析以下已退出持仓，总结经验教训:\n{stocks_text}\n\n"
        "请用 JSON 返回：{\"key_lessons\": \"...\", \"winning_patterns\": \"...\", \"losing_patterns\": \"...\"}\n"
        "只返回 JSON，不要其他内容。"
    )
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    key_lessons = raw
    winning = ""
    losing = ""
    try:
        text = raw.strip()
        if "```" in text:
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = _json.loads(text[start:end])
            key_lessons = parsed.get("key_lessons", raw)
            winning = parsed.get("winning_patterns", "")
            losing = parsed.get("losing_patterns", "")
    except Exception:
        pass

    win_count = sum(1 for s in exited if (s.get("return_pct") or 0) > 0)
    win_rate = round(win_count / len(exited) * 100, 2) if exited else 0
    avg_return = round(
        sum(s.get("return_pct", 0) or 0 for s in exited) / len(exited), 4
    ) if exited else 0

    summary_id = store.save_pool_summary({
        "summary_date": today,
        "period_days": 30,
        "total_stocks": len(exited),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "key_lessons": key_lessons,
        "winning_patterns": winning,
        "losing_patterns": losing,
        "raw_response": raw,
    })

    return {"summary_id": summary_id, "total_stocks": len(exited), "win_rate": win_rate}


@router.get("/adjustments")
def list_pool_adjustments(
    limit: int = Query(10),
    store=Depends(get_store),
):
    """List pool strategy adjustments."""
    return store.list_pool_strategy_adjustments(limit=limit)


@router.get("/stocks/{pool_id}/lessons")
def get_pool_stock_lessons(pool_id: int, store=Depends(get_store)):
    """Get lessons for a specific pool stock."""
    return store.list_pool_lessons(pool_id=pool_id)


@router.get("/stocks/{pool_id}/debate")
def get_pool_stock_debate(pool_id: int, store=Depends(get_store)):
    """Get debate decision for a pool stock."""
    stock = store.get_pool_stock(pool_id)
    if not stock:
        raise HTTPException(status_code=404, detail=f"Pool stock not found: {pool_id}")

    code = stock["code"]
    yahoo_ticker = code
    try:
        from tradingagents.api.utils.ticker_utils import to_yahoo_ticker
        yahoo_ticker = to_yahoo_ticker(code)
    except Exception:
        pass

    decisions = store.list_decisions(ticker=yahoo_ticker, limit=1)
    if not decisions:
        return {"pool_id": pool_id, "code": code, "debate": None}
    return decisions[0]


class FinetuneRequest(BaseModel):
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5


@router.post("/trigger/finetune")
def trigger_finetune(body: FinetuneRequest = FinetuneRequest()):
    """Trigger FinBERT2 sentiment model fine-tuning in background."""
    from tradingagents.finbert.trainer import FinBERTTrainer

    trainer = FinBERTTrainer.get_instance()
    status = trainer.get_status()
    if status["state"] == "training":
        return {"started": False, "message": "Training already in progress", **status}

    trainer.run_async(
        epochs=body.epochs,
        batch_size=body.batch_size,
        learning_rate=body.learning_rate,
    )
    return {"started": True, "message": "Fine-tuning started in background"}


@router.get("/finetune-status")
def get_finetune_status():
    """Get current fine-tuning status."""
    from tradingagents.finbert.trainer import FinBERTTrainer

    trainer = FinBERTTrainer.get_instance()
    return trainer.get_status()
