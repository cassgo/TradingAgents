"""Pool pipeline: orchestrates daily prediction, evaluation, and screening.

New funnel: S0 IndustryPreFilter -> S1 SentimentScreener -> S2 LLMScreener
            -> S3 Enricher -> S4 Debater -> pool entry
"""

import logging
import time
from datetime import datetime

from tradingagents.pool.evaluator import PoolEvaluator
from tradingagents.pool.daily_predictor import DailyPredictor
from tradingagents.pool.industry_prefilter import IndustryPreFilter
from tradingagents.pool.sentiment_screener import SentimentScreener
from tradingagents.pool.llm_screener import LLMScreener
from tradingagents.pool.debater import PoolDebater
from tradingagents.pool.selector import PoolSelector
from tradingagents.pool.auto_exit import AutoExitChecker

logger = logging.getLogger(__name__)


class PoolPipeline:
    """Orchestrates the prediction pool pipeline."""

    def __init__(self, store, config: dict | None = None, llm=None, graph=None):
        self.store = store
        self.config = config or {}
        self.llm = llm
        self.graph = graph

        from tradingagents.utils.rate_limiter import RateLimiter
        self._ak_limiter = RateLimiter(rate=2.0)   # 2 akshare calls/sec
        self._llm_limiter = RateLimiter(rate=4.0)   # 4 LLM calls/sec

    def _get_llm(self, *, quick: bool = False):
        """Get or create LLM client.

        Args:
            quick: If True, use quick_think_llm (faster, for batch scoring).
        """
        if self.llm and not quick:
            return self.llm
        from tradingagents.api.llm_utils import get_llm
        return get_llm(self.config, quick=quick)

    def _get_pipeline_config(self) -> dict:
        """Get pipeline sub-config with defaults."""
        return self.config.get("pipeline", {})

    def run_daily_only(self, run_date: str | None = None) -> dict:
        """Run daily evaluate + predict (Phase 1 minimal pipeline).

        Steps:
        1. Evaluate yesterday's predictions
        2. Predict tomorrow for all active stocks
        """
        if run_date is None:
            run_date = datetime.now().strftime("%Y-%m-%d")

        llm = self._get_llm()

        evaluator = PoolEvaluator(store=self.store, llm=llm, config=self.config)
        eval_result = evaluator.evaluate(run_date)
        logger.info("[Pipeline] Evaluation: %s", eval_result)

        predictor = DailyPredictor(store=self.store, llm=llm, config=self.config)
        pred_result = predictor.predict_all(run_date)
        logger.info("[Pipeline] Prediction: %s", pred_result)

        return {
            "date": run_date,
            "evaluation": eval_result,
            "prediction": pred_result,
        }

    def run_full_pipeline(self, run_date: str | None = None) -> dict:
        """Run complete screening pipeline (S0->S1->S2->S3->S4->S5).

        S0: IndustryPreFilter  (~5000 -> ~500)
        S1: SentimentScreener  (~500  -> ~100)
        S2: LLMScreener        (~100  -> ~40)
        S3: Enricher + Debater (~40   -> ~20 debated)
        S4: Selector           (~20   -> pool entry)
        S5: Daily predictions for all active pool stocks
        """
        if run_date is None:
            run_date = datetime.now().strftime("%Y-%m-%d")

        llm = self._get_llm()
        pcfg = self._get_pipeline_config()
        start_time = time.time()
        run_id = self.store.create_screening_run(run_date)

        try:
            # Evaluate yesterday
            evaluator = PoolEvaluator(store=self.store, llm=llm, config=self.config)
            eval_result = evaluator.evaluate(run_date)
            logger.info("[Pipeline] Evaluation: %s", eval_result)

            # Auto-exit check
            exit_checker = AutoExitChecker(store=self.store, config=self.config)
            auto_exits = exit_checker.check_and_execute(run_date)
            if auto_exits:
                logger.info("[Pipeline] Auto-exited %d stocks", len(auto_exits))

            # --- S0->S1->S2 with expansion loop ---
            # S0 is called per-round with expanding top_n. The fund_flow/hot_stocks
            # API calls are cheap (1 each), and _enrich_industry only enriches
            # stocks not yet in DB. This avoids pre-enriching 700 stocks when
            # only 100 may be needed in the happy path.
            s2_min_score = pcfg.get("s2_min_score", 60)
            max_rounds = pcfg.get("max_search_rounds", 3)
            s0_base = pcfg.get("s0_top_n", 100)
            s0_expand = pcfg.get("s0_expand_step", 300)
            s1_expand = pcfg.get("s1_expand_step", 50)

            all_scored: list[dict] = []   # accumulate S2 results across rounds
            seen_codes: set[str] = set()  # tracks all stocks sent to S1 (avoids re-processing)
            all_s0_count = 0
            all_s1_count = 0
            qualified: list[dict] = []

            quick_llm = self._get_llm(quick=True)

            for search_round in range(max_rounds):
                # S0: expanding top_n each round
                s0_top_n = s0_base + search_round * s0_expand
                s1_top_n = pcfg.get("s1_top_n", 50) + search_round * s1_expand

                logger.info("[Pipeline] === S0 行业预筛 开始 (round %d, top_n=%d) ===",
                            search_round + 1, s0_top_n)
                s0_start = time.time()
                self.store.update_screening_run(run_id, {"current_stage": 0})
                s0_filter = IndustryPreFilter(store=self.store, config={
                    **pcfg, "s0_top_n": s0_top_n,
                }, akshare_limiter=self._ak_limiter)
                s0 = s0_filter.filter(run_date)
                logger.info("[Pipeline] S0 完成: %d 只, 耗时 %.1fs",
                            len(s0), time.time() - s0_start)
                all_s0_count = max(all_s0_count, len(s0))

                if not s0 and search_round == 0:
                    self.store.update_screening_run(run_id, {"s0_count": 0})
                    self._finalize_run(run_id, start_time, current_stage=5)
                    pred_result = self._run_predictions(llm, run_date)
                    return self._build_result(
                        run_date, run_id, eval_result, pred_result,
                        s0=0, s1=0, s2=0, stage3=0, stage4=0,
                    )

                # Save S0 candidates on first round
                if search_round == 0:
                    self._save_stage_candidates(run_id, s0, stage=0)

                # Remove already-processed stocks (includes S1-rejected ones)
                if seen_codes:
                    s0 = [c for c in s0 if c["code"] not in seen_codes]

                if not s0:
                    logger.info("[Pipeline] Round %d: no new S0 candidates, stopping",
                                search_round + 1)
                    break

                # Mark as seen BEFORE S1 — stocks rejected by S1 won't be retried
                for c in s0:
                    seen_codes.add(c["code"])

                logger.info("[Pipeline] === S1 情绪筛选 开始 (round %d, 输入 %d 只) ===",
                            search_round + 1, len(s0))

                # S1: SentimentScreener (per-stock, only processes new stocks)
                s1_start = time.time()
                self.store.update_screening_run(run_id, {"current_stage": 1})
                s1_screener = SentimentScreener(
                    store=self.store,
                    sentiment_analyzer=self._get_sentiment_analyzer(),
                    config={**pcfg, "s1_top_n": s1_top_n},
                    akshare_limiter=self._ak_limiter,
                )
                s1 = s1_screener.screen(s0, run_date)
                logger.info("[Pipeline] S1 完成: %d 只, 耗时 %.1fs", len(s1), time.time() - s1_start)
                all_s1_count += len(s1)
                if search_round == 0:
                    self._save_stage_candidates(run_id, s1, stage=1)

                if not s1:
                    if search_round == 0:
                        self.store.update_screening_run(run_id, {
                            "s0_count": all_s0_count, "s1_count": 0,
                        })
                        self._finalize_run(run_id, start_time, current_stage=5)
                        pred_result = self._run_predictions(llm, run_date)
                        return self._build_result(
                            run_date, run_id, eval_result, pred_result,
                            s0=all_s0_count, s1=0, s2=0, stage3=0, stage4=0,
                        )
                    continue

                # S2: LLMScreener
                logger.info("[Pipeline] === S2 LLM评估 开始 (round %d, 输入 %d 只) ===",
                            search_round + 1, len(s1))
                s2_start = time.time()
                self.store.update_screening_run(run_id, {"current_stage": 2})
                s2_screener = LLMScreener(store=self.store, llm=quick_llm, config={
                    **pcfg, "s2_top_n": pcfg.get("s2_top_n", 40),
                }, llm_limiter=self._llm_limiter)
                s2 = s2_screener.screen(s1, run_date)
                logger.info("[Pipeline] S2 完成: %d 只, 耗时 %.1fs", len(s2), time.time() - s2_start)

                all_scored.extend(s2)

                # Save S2 results for this round
                self._save_stage_candidates(run_id, s2, stage=2)

                # Check min_score threshold
                qualified = [c for c in all_scored if c.get("llm_score", 0) >= s2_min_score]
                if qualified:
                    logger.info("[Pipeline] Round %d: found %d candidates >= %d",
                                search_round + 1, len(qualified), s2_min_score)
                    break

                logger.info("[Pipeline] Round %d: 0 candidates >= %d (scored %d total), expanding...",
                            search_round + 1, s2_min_score, len(all_scored))

            # After loop: use qualified if found, otherwise fallback to best available
            if not qualified:
                all_scored.sort(key=lambda x: x.get("llm_score", 0), reverse=True)
                fallback_n = pcfg.get("s2_top_n", 20)
                qualified = all_scored[:fallback_n]
                logger.warning("[Pipeline] No candidates >= %d after %d rounds, using top %d",
                               s2_min_score, max_rounds, len(qualified))

            # Update run counts
            self.store.update_screening_run(run_id, {
                "s0_count": all_s0_count, "s1_count": all_s1_count,
                "s2_count": len(qualified),
            })

            s2 = qualified

            if not s2:
                self._finalize_run(run_id, start_time, current_stage=5)
                pred_result = self._run_predictions(llm, run_date)
                return self._build_result(
                    run_date, run_id, eval_result, pred_result,
                    s0=all_s0_count, s1=all_s1_count, s2=0, stage3=0, stage4=0,
                )

            # S3: Debate (S2 feeds directly, no separate enrichment step)
            logger.info("[Pipeline] === S3 多空辩论 开始 (输入 %d 只) ===", len(s2))
            s3_start = time.time()
            self.store.update_screening_run(run_id, {"current_stage": 3})
            saved_s3: list[dict] = []
            if self.graph and s2:
                max_debates = self.config.get("max_debates", 12)
                debater = PoolDebater(
                    graph=self.graph, store=self.store,
                    config=self.config, max_debates=max_debates,
                )

                def _on_debate_progress(i, total, candidate):
                    logger.info("[Pipeline] S3 辩论进度: %d/%d (%s)", i + 1, total, candidate.get("code", "?"))
                    self.store.update_screening_run(
                        run_id, {"stage3_progress": f"{i + 1}/{total}"},
                    )

                def _on_debate_result(result):
                    self._save_stage_candidates(run_id, [result], stage=3)
                    saved_s3.append(result)
                    self.store.update_screening_run(run_id, {"stage3_count": len(saved_s3)})

                stage3 = debater.debate_candidates(
                    s2, run_date,
                    on_progress=_on_debate_progress,
                    on_result=_on_debate_result,
                )
            else:
                stage3 = []
            logger.info("[Pipeline] S3 完成: %d 只, 耗时 %.1fs", len(stage3), time.time() - s3_start)
            self.store.update_screening_run(run_id, {"stage3_count": len(stage3)})

            # S4: Select for pool entry
            logger.info("[Pipeline] === S4 入池筛选 开始 (输入 %d 只) ===", len(stage3))
            s4_start = time.time()
            self.store.update_screening_run(run_id, {"current_stage": 4})
            selector = PoolSelector(store=self.store, config=self.config)
            stage4 = selector.select(stage3, run_date)
            logger.info("[Pipeline] S4 完成: %d 只, 耗时 %.1fs", len(stage4), time.time() - s4_start)
            self.store.update_screening_run(run_id, {"stage4_count": len(stage4)})
            self._save_stage_candidates(run_id, stage4, stage=4)

            for entry in stage4:
                self.store.upsert_pool_stock(entry)

            # Daily predictions
            pred_result = self._run_predictions(llm, run_date)

            self._finalize_run(run_id, start_time, current_stage=5)

            return self._build_result(
                run_date, run_id, eval_result, pred_result,
                s0=all_s0_count, s1=all_s1_count, s2=len(s2),
                stage3=len(stage3), stage4=len(stage4),
            )

        except Exception:
            duration = time.time() - start_time
            self.store.update_screening_run(run_id, {
                "status": "failed",
                "duration_sec": round(duration, 2),
                "error_message": "Pipeline failed",
            })
            raise

    def run_stage(self, stage: int, run_date: str | None = None, *, resume: bool = False) -> dict:
        """Run pipeline from stage N, loading previous run's candidates from DB.

        Args:
            stage: 1-4 (S1..S4)
            run_date: Date string, defaults to today.
            resume: If True and stage==3, reuse previous run_id and skip
                    already-completed S3 debates (checkpoint resume).
        """
        if run_date is None:
            run_date = datetime.now().strftime("%Y-%m-%d")

        runs = self.store.list_screening_runs(limit=1)
        if not runs:
            raise ValueError("No previous screening run found")
        prev_run = runs[0]
        prev_run_id = prev_run["id"]

        prev_stage = stage - 1
        candidates = self.store.get_screening_candidates(prev_run_id, stage=prev_stage)
        if not candidates:
            raise ValueError(f"No candidates from stage {prev_stage} in run {prev_run_id}")

        llm = self._get_llm()
        pcfg = self._get_pipeline_config()
        start_time = time.time()

        # Resume mode for S3: reuse previous run_id instead of creating new
        if resume and stage == 3:
            run_id = prev_run_id
            self.store.update_screening_run(run_id, {
                "status": "running", "current_stage": 3, "error_message": None,
            })
        else:
            run_id = self.store.create_screening_run(run_date)
            # Copy counts from previous stages
            stage_keys = ["s0_count", "s1_count", "s2_count", "stage3_count", "stage4_count"]
            carry = {}
            for i in range(stage):
                key = stage_keys[i]
                carry[key] = prev_run.get(key, 0)
            if carry:
                self.store.update_screening_run(run_id, carry)

        try:
            current = candidates
            result_counts: dict = {}
            logger.info("[Pipeline:run_stage] 从 S%d 开始, 候选人 %d 只 (来自 run#%d S%d)",
                        stage, len(current), prev_run_id, prev_stage)

            if stage <= 1:
                logger.info("[Pipeline:run_stage] === S1 情绪筛选 开始 ===")
                s1_start = time.time()
                self.store.update_screening_run(run_id, {"current_stage": 1})
                s1_screener = SentimentScreener(
                    store=self.store,
                    sentiment_analyzer=self._get_sentiment_analyzer(),
                    config={**pcfg, "s1_top_n": pcfg.get("s1_top_n", 100)},
                    akshare_limiter=self._ak_limiter,
                )
                current = s1_screener.screen(current, run_date)
                logger.info("[Pipeline:run_stage] S1 完成: %d 只, 耗时 %.1fs", len(current), time.time() - s1_start)
                self.store.update_screening_run(run_id, {"s1_count": len(current)})
                self._save_stage_candidates(run_id, current, stage=1)
                result_counts["s1_count"] = len(current)
                if not current:
                    self._finalize_run(run_id, start_time, current_stage=5)
                    return {"date": run_date, "run_id": run_id, **result_counts}

            if stage <= 2:
                logger.info("[Pipeline:run_stage] === S2 LLM评估 开始 (%d 只) ===", len(current))
                s2_start = time.time()
                self.store.update_screening_run(run_id, {"current_stage": 2})
                quick_llm = self._get_llm(quick=True)
                s2_screener = LLMScreener(store=self.store, llm=quick_llm, config={
                    **pcfg, "s2_top_n": pcfg.get("s2_top_n", 40),
                }, llm_limiter=self._llm_limiter)
                current = s2_screener.screen(current, run_date)
                logger.info("[Pipeline:run_stage] S2 完成: %d 只, 耗时 %.1fs", len(current), time.time() - s2_start)
                self.store.update_screening_run(run_id, {"s2_count": len(current)})
                self._save_stage_candidates(run_id, current, stage=2)
                result_counts["s2_count"] = len(current)
                if not current:
                    self._finalize_run(run_id, start_time, current_stage=5)
                    return {"date": run_date, "run_id": run_id, **result_counts}

            if stage <= 3:
                logger.info("[Pipeline:run_stage] === S3 多空辩论 开始 (%d 只) ===", len(current))
                s3_start = time.time()
                self.store.update_screening_run(run_id, {"current_stage": 3})

                done_count = 0
                saved_in_stage: list[dict] = []

                if self.graph and current:
                    max_debates = self.config.get("max_debates", 12)

                    # Resume: detect already-completed debates
                    if resume:
                        existing_s3 = self.store.get_screening_candidates(run_id, stage=3)
                        done_codes = {c["code"] for c in existing_s3}
                        done_count = len(done_codes)
                        ranked = sorted(current, key=lambda c: c.get("vol_ratio", 0) or 0, reverse=True)
                        should_debate = ranked[:max_debates]
                        current = [c for c in should_debate if c["code"] not in done_codes]
                        logger.info("[Pipeline:run_stage] S3 续跑: 已完成 %d, 剩余 %d",
                                    done_count, len(current))

                    total_to_show = done_count + len(current)

                    # Incremental save callback
                    def _on_result(result):
                        self._save_stage_candidates(run_id, [result], stage=3)
                        saved_in_stage.append(result)
                        self.store.update_screening_run(run_id, {
                            "stage3_count": done_count + len(saved_in_stage),
                        })

                    def _on_progress(i, total, candidate):
                        display_i = done_count + i + 1
                        logger.info("[Pipeline:run_stage] S3 辩论进度: %d/%d (%s)",
                                    display_i, total_to_show, candidate.get("code", "?"))
                        self.store.update_screening_run(
                            run_id, {"stage3_progress": f"{display_i}/{total_to_show}"},
                        )

                    debater = PoolDebater(
                        graph=self.graph, store=self.store,
                        config=self.config, max_debates=max_debates,
                    )
                    new_results = debater.debate_candidates(
                        current, run_date,
                        on_progress=_on_progress, on_result=_on_result,
                    )

                    # Merge results for S4: already-completed + newly debated
                    if resume and done_count > 0:
                        all_s3 = self.store.get_screening_candidates(run_id, stage=3)
                        current = self._candidates_to_dicts(all_s3) if all_s3 else new_results
                    else:
                        current = new_results
                else:
                    logger.warning("[Pipeline:run_stage] S3 跳过: graph=%s, candidates=%d",
                                   bool(self.graph), len(current))
                    current = []

                final_count = done_count + len(saved_in_stage) if self.graph else 0
                logger.info("[Pipeline:run_stage] S3 完成: %d 只, 耗时 %.1fs", final_count, time.time() - s3_start)
                self.store.update_screening_run(run_id, {"stage3_count": final_count})
                # No batch _save_stage_candidates — already saved incrementally via on_result
                result_counts["stage3_count"] = final_count

            if stage <= 4:
                logger.info("[Pipeline:run_stage] === S4 入池筛选 开始 (%d 只) ===", len(current))
                s4_start = time.time()
                self.store.update_screening_run(run_id, {"current_stage": 4})
                selector = PoolSelector(store=self.store, config=self.config)
                current = selector.select(current, run_date)
                logger.info("[Pipeline:run_stage] S4 完成: %d 只, 耗时 %.1fs", len(current), time.time() - s4_start)
                self.store.update_screening_run(run_id, {"stage4_count": len(current)})
                self._save_stage_candidates(run_id, current, stage=4)
                result_counts["stage4_count"] = len(current)

                for entry in current:
                    self.store.upsert_pool_stock(entry)

            total_duration = time.time() - start_time
            logger.info("[Pipeline:run_stage] 全部完成, 总耗时 %.1fs", total_duration)
            self._finalize_run(run_id, start_time, current_stage=5)
            return {"date": run_date, "run_id": run_id, **result_counts}

        except Exception:
            duration = time.time() - start_time
            self.store.update_screening_run(run_id, {
                "status": "failed",
                "duration_sec": round(duration, 2),
                "error_message": f"Pipeline failed from stage {stage}",
            })
            raise

    @staticmethod
    def _candidates_to_dicts(rows: list[dict]) -> list[dict]:
        """Convert DB screening_candidate rows back to debater-output dicts."""
        return [
            {
                "code": r.get("code", ""),
                "code_name": r.get("code_name", ""),
                "signal": r.get("debate_signal"),
                "score": r.get("score"),
                "reasoning": r.get("reasoning", ""),
            }
            for r in rows
        ]

    def _save_stage_candidates(self, run_id: int, candidates: list[dict], stage: int) -> None:
        """Save candidates to pool_screening_candidates table."""
        if not candidates:
            return
        rows = []
        for c in candidates:
            if stage == 2 and "llm_score" in c:
                score = c["llm_score"] / 100
                reasoning = c.get("llm_reasoning", c.get("reasoning", ""))
            else:
                score = c.get("score")
                reasoning = c.get("reasoning", "")
            rows.append({
                "run_id": run_id,
                "code": c.get("code", ""),
                "code_name": c.get("code_name", ""),
                "stage": stage,
                "passed": True,
                "score": score,
                "reasoning": reasoning,
                "debate_signal": c.get("signal"),
            })
        self.store.save_screening_candidates(rows)

    def run_from_stage(
        self, stage: str, candidates: list[dict], run_date: str,
    ) -> dict:
        """Run pipeline from a specific stage with given candidates.

        Args:
            stage: "s1", "s2", "s3", or "s4"
            candidates: Pre-filtered candidates to start from
            run_date: Date string
        """
        llm = self._get_llm()
        pcfg = self._get_pipeline_config()
        result: dict = {"date": run_date}

        current = candidates

        if stage <= "s1":
            s1_screener = SentimentScreener(
                store=self.store,
                sentiment_analyzer=self._get_sentiment_analyzer(),
                config={**pcfg, "s1_top_n": pcfg.get("s1_top_n", 100)},
                akshare_limiter=self._ak_limiter,
            )
            current = s1_screener.screen(current, run_date)
            result["s1_count"] = len(current)
            if not current:
                return result

        if stage <= "s2":
            quick_llm = self._get_llm(quick=True)
            s2_screener = LLMScreener(store=self.store, llm=quick_llm, config={
                **pcfg, "s2_top_n": pcfg.get("s2_top_n", 40),
            }, llm_limiter=self._llm_limiter)
            current = s2_screener.screen(current, run_date)
            result["s2_count"] = len(current)

        return result

    def run_sentiment_prescreen(self, run_date: str | None = None) -> dict:
        """Run S0+S1 only (for testing/monitoring).

        Returns counts from industry pre-filter and sentiment screening.
        """
        if run_date is None:
            run_date = datetime.now().strftime("%Y-%m-%d")

        pcfg = self._get_pipeline_config()

        s0_filter = IndustryPreFilter(store=self.store, config={
            **pcfg, "s0_top_n": pcfg.get("s0_top_n", 500),
        }, akshare_limiter=self._ak_limiter)
        s0 = s0_filter.filter(run_date)

        if not s0:
            return {"date": run_date, "s0_count": 0, "s1_count": 0}

        s1_screener = SentimentScreener(
            store=self.store,
            sentiment_analyzer=self._get_sentiment_analyzer(),
            config={**pcfg, "s1_top_n": pcfg.get("s1_top_n", 100)},
            akshare_limiter=self._ak_limiter,
        )
        s1 = s1_screener.screen(s0, run_date)

        return {"date": run_date, "s0_count": len(s0), "s1_count": len(s1)}

    def run_llm_screen(self, run_date: str | None = None) -> dict:
        """Run S0+S1+S2 (for testing S2 LLMScreener standalone).

        Returns stage counts and the S2 scored candidates.
        """
        if run_date is None:
            run_date = datetime.now().strftime("%Y-%m-%d")

        llm = self._get_llm()
        pcfg = self._get_pipeline_config()

        # S0
        s0_filter = IndustryPreFilter(store=self.store, config={
            **pcfg, "s0_top_n": pcfg.get("s0_top_n", 500),
        }, akshare_limiter=self._ak_limiter)
        s0 = s0_filter.filter(run_date)
        if not s0:
            return {
                "date": run_date, "s0_count": 0, "s1_count": 0,
                "s2_count": 0, "candidates": [],
            }

        # S1
        s1_screener = SentimentScreener(
            store=self.store,
            sentiment_analyzer=self._get_sentiment_analyzer(),
            config={**pcfg, "s1_top_n": pcfg.get("s1_top_n", 100)},
            akshare_limiter=self._ak_limiter,
        )
        s1 = s1_screener.screen(s0, run_date)
        if not s1:
            return {
                "date": run_date, "s0_count": len(s0), "s1_count": 0,
                "s2_count": 0, "candidates": [],
            }

        # S2 (uses quick model for batch scoring)
        quick_llm = self._get_llm(quick=True)
        s2_screener = LLMScreener(store=self.store, llm=quick_llm, config={
            **pcfg, "s2_top_n": pcfg.get("s2_top_n", 20),
        }, llm_limiter=self._llm_limiter)
        s2 = s2_screener.screen(s1, run_date)

        return {
            "date": run_date,
            "s0_count": len(s0),
            "s1_count": len(s1),
            "s2_count": len(s2),
            "candidates": s2,
        }

    def _get_sentiment_analyzer(self):
        """Get FinBERT sentiment analyzer (lazy import)."""
        try:
            from tradingagents.finbert.model_manager import FinBERTModelManager
            from tradingagents.finbert.sentiment import SentimentAnalyzer
            manager = FinBERTModelManager.get_instance()
            return SentimentAnalyzer(manager)
        except Exception:
            logger.warning("FinBERT not available, using dummy analyzer")
            return _DummyAnalyzer()

    def _run_predictions(self, llm, run_date: str) -> dict:
        """Run daily predictions for all active pool stocks."""
        predictor = DailyPredictor(store=self.store, llm=llm, config=self.config)
        return predictor.predict_all(run_date)

    def _finalize_run(self, run_id: int, start_time: float, current_stage: int = 5, **counts) -> None:
        """Mark screening run as completed."""
        duration = time.time() - start_time
        updates = {
            "status": "completed",
            "duration_sec": round(duration, 2),
            "current_stage": current_stage,
        }
        updates.update(counts)
        self.store.update_screening_run(run_id, updates)

    @staticmethod
    def _build_result(
        run_date, run_id, eval_result, pred_result,
        s0=0, s1=0, s2=0, stage3=0, stage4=0,
    ) -> dict:
        return {
            "date": run_date,
            "run_id": run_id,
            "evaluation": eval_result,
            "prediction": pred_result,
            "s0_count": s0,
            "s1_count": s1,
            "s2_count": s2,
            "stage3_count": stage3,
            "stage4_count": stage4,
        }


class _DummyAnalyzer:
    """Fallback when FinBERT is not available."""

    def analyze_news(self, articles):
        return articles

    def analyze_batch(self, texts, batch_size=32):
        return [{"label": "neutral", "score": 0.5}] * len(texts)
