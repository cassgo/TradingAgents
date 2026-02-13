"""Predictions API router."""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import baostock
from fastapi import APIRouter, Depends, HTTPException

from tradingagents.api.deps import get_config, get_store
from tradingagents.storage.postgres_store import PostgresStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["predictions"])

# 第一轮：信号+新闻 → 板块预测 + 推荐个股列表
_ROUND1_PROMPT = """你是一位资深 A 股量化分析师。请基于今日信号面和消息面，筛选最值得关注的板块和个股。

## 今日行业信号
{signals_text}

## 今日新闻摘要（最新 30 条）
{news_text}

## 请输出

严格按以下 JSON 格式输出：

```json
{{
  "sector_predictions": [
    {{
      "target": "板块名称",
      "direction": "看涨/看跌/震荡",
      "confidence": 0.85,
      "reasoning": "结合资金流向+消息面的分析",
      "time_horizon": "1-3个交易日"
    }}
  ],
  "recommended_stocks": ["股票名称(代码)", "股票名称(代码)"]
}}
```

要求：
1. sector_predictions: 选出 3-5 个最有把握的板块
2. recommended_stocks: 从强势板块中选 5-8 只最值得分析的个股（名称+代码）
3. confidence 为 0.0-1.0 的浮点数
4. 只输出 JSON，不要有其他内容
"""

# 第二轮：单只股票 K 线 → 具体交易计划
_ROUND2_PROMPT = """你是一位资深 A 股量化分析师。请基于该股票的近期 K 线数据，给出具体交易建议。

## 股票：{stock_name}
## 相关板块信号：{sector_context}

## 近 10 日 K 线
{history_text}

## 请输出

严格按以下 JSON 格式输出：

```json
{{
  "target": "{stock_name}",
  "direction": "看涨/看跌/震荡",
  "confidence": 0.80,
  "reasoning": "结合K线走势+成交量变化的技术分析（必须引用具体数据）",
  "entry_price": "具体价格或区间",
  "target_price": "具体目标价",
  "stop_loss": "具体止损价",
  "time_horizon": "1-5个交易日"
}}
```

要求：
1. entry_price / target_price / stop_loss 必须是具体数字，不要模糊描述
2. reasoning 必须引用走势数据（如"近5日累计涨幅X%"、"量比Y"等）
3. 只输出 JSON，不要有其他内容
"""

# 第三轮：汇总所有个股分析 → 排名+组合建议
_ROUND3_PROMPT = """你是一位资深 A 股量化分析师。前面已经对多只个股进行了独立技术分析，现在请你综合所有结果，给出最终投资组合建议。

## 板块预测
{sector_summary}

## 各个股独立分析结果
{stock_analyses}

## 请输出

综合以上所有分析，进行排序筛选和组合优化，严格按以下 JSON 格式输出：

```json
{{
  "top_picks": [
    {{
      "rank": 1,
      "target": "股票名称(代码)",
      "direction": "看涨/看跌",
      "confidence": 0.85,
      "entry_price": "具体价格",
      "target_price": "具体价格",
      "stop_loss": "具体价格",
      "reward_risk_ratio": 2.5,
      "reasoning": "综合排名理由（为什么排第一）",
      "time_horizon": "1-5个交易日"
    }}
  ],
  "portfolio_summary": "整体组合点评：板块分散度、多空配比、总体风险评估（2-3句话）"
}}
```

要求：
1. 从所有个股中选出最优的 3-5 只，按推荐度排序
2. reward_risk_ratio = (目标价-买入价) / (买入价-止损价)，必须是具体数字
3. 剔除信号模糊或价格不具体的个股
4. portfolio_summary 需评价板块集中度和多空比例
5. 只输出 JSON，不要有其他内容
"""


def _fetch_stock_history(codes: list[str], days: int = 10) -> dict[str, list[dict]]:
    """Fetch recent K-line data via BaoStock (free, no registration needed)."""
    import baostock as bs
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")

    lg = bs.login()
    if lg.error_code != '0':
        logger.warning("BaoStock login failed: %s", lg.error_msg)
        return {}

    result: dict[str, list[dict]] = {}
    try:
        for code in codes[:10]:
            pure = re.sub(r'^(SZ|SH|BJ)', '', code)
            if not pure or not pure.isdigit():
                continue
            if pure.startswith("6"):
                symbol = f"sh.{pure}"
            elif pure.startswith(("0", "3")):
                symbol = f"sz.{pure}"
            elif pure.startswith(("4", "8")):
                symbol = f"bj.{pure}"
            else:
                symbol = f"sz.{pure}"
            try:
                rs = bs.query_history_k_data_plus(
                    symbol,
                    "date,open,high,low,close,volume,pctChg,turn",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="2",  # 前复权
                )
                if rs.error_code != '0':
                    continue
                rows = []
                while rs.next():
                    r = rs.get_row_data()
                    try:
                        rows.append({
                            "date": r[0],
                            "open": float(r[1]),
                            "high": float(r[2]),
                            "low": float(r[3]),
                            "close": float(r[4]),
                            "volume": int(float(r[5])),
                            "change_pct": float(r[6]) if r[6] else 0,
                            "turnover": float(r[7]) if r[7] else 0,
                        })
                    except (ValueError, IndexError):
                        continue
                if rows:
                    result[pure] = rows[-days:]
            except Exception:
                logger.debug("Failed to fetch history for %s", symbol, exc_info=True)
    finally:
        bs.logout()
    return result


def _extract_stock_codes(news: list[dict]) -> tuple[list[str], dict[str, str]]:
    """Extract stock codes and name mappings from news and hot rank data."""
    codes: list[str] = []
    code_names: dict[str, str] = {}
    seen: set[str] = set()

    code_pattern = re.compile(r'[（(]([A-Z]{0,2}\d{6})[）)]')

    # Pass 1: extract codes from title + content (hot_rank has codes)
    for n in news:
        title = n.get("title", "")
        content = n.get("content", "") or ""
        source = n.get("source", "")
        if source not in ("hot_rank", "weibo"):
            continue
        # Search both title and content for stock codes
        for text in (title, content):
            match = code_pattern.search(text)
            if match:
                raw_code = match.group(1)
                pure = re.sub(r'^(SZ|SH|BJ)', '', raw_code)
                if pure not in seen:
                    seen.add(pure)
                    codes.append(pure)
                    name_match = re.search(r']\s*(.+?)[（(]', title)
                    if name_match:
                        code_names[pure] = name_match.group(1).strip()
                break  # found code for this news item

    # Pass 2: stock-specific news source like "stock:002594"
    for n in news:
        source = n.get("source", "")
        if source.startswith("stock:"):
            pure = source.split(":")[1]
            if pure not in seen:
                seen.add(pure)
                codes.append(pure)

    return codes[:15], code_names


@router.get("/predictions")
def list_predictions(
    date: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
    store: PostgresStore = Depends(get_store),
):
    return store.list_predictions(date=date, category=category, limit=limit)


def _get_llm(config: dict):
    """Create LLM client from config."""
    from tradingagents.llm_clients import create_llm_client

    client = create_llm_client(
        provider=config.get("llm_provider", "openai"),
        model=config.get("deep_think_llm", ""),
        base_url=config.get("backend_url"),
        api_key=config.get("api_key"),
    )
    return client.get_llm()


def _invoke_llm(llm, prompt: str) -> str:
    """Invoke LLM and return content string."""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def _format_single_stock_history(rows: list[dict]) -> str:
    """Format K-line data for a single stock."""
    if not rows:
        return "（无数据）"
    latest = rows[-1]
    first = rows[0]
    total_change = ((latest["close"] - first["open"]) / first["open"] * 100) if first["open"] else 0
    avg_vol = sum(r["volume"] for r in rows) / len(rows)
    vol_ratio = rows[-1]["volume"] / avg_vol if avg_vol else 0

    lines = [
        f"近{len(rows)}日走势：累计涨跌 {total_change:+.2f}%，最新收盘 {latest['close']:.2f}，量比 {vol_ratio:.2f}",
        f"{'日期':>12s} | {'开盘':>8s} | {'收盘':>8s} | {'最高':>8s} | {'最低':>8s} | {'涨跌%':>6s} | {'成交量':>10s} | {'换手%':>5s}",
    ]
    for r in rows:
        lines.append(
            f"{r['date']:>12s} | {r['open']:>8.2f} | {r['close']:>8.2f} | "
            f"{r['high']:>8.2f} | {r['low']:>8.2f} | {r['change_pct']:>+6.2f} | "
            f"{r['volume']:>10d} | {r['turnover']:>5.2f}"
        )
    return "\n".join(lines)


@router.post("/predictions/generate")
def generate_predictions(
    store: PostgresStore = Depends(get_store),
    config: dict = Depends(get_config),
):
    """Generate predictions in two rounds: sectors first, then per-stock analysis."""
    today = datetime.now().strftime("%Y-%m-%d")

    # Gather context
    signals = store.list_signals(date=today, limit=20)
    news = store.list_news(limit=100)

    if not signals and not news:
        raise HTTPException(status_code=400, detail="没有可用的信号和新闻数据，请先运行新闻采集和信号生成")

    # Format signals
    signals_lines = []
    for s in signals:
        industries = s.get("industries", "[]")
        try:
            ind_list = json.loads(industries) if isinstance(industries, str) else industries
        except json.JSONDecodeError:
            ind_list = [industries]
        signals_lines.append(
            f"- {s['event']} | 情绪: {s['sentiment']} | 置信度: {s['confidence']} | 行业: {', '.join(ind_list)}"
        )
    signals_text = "\n".join(signals_lines) if signals_lines else "（暂无信号）"

    # Format news (top 30)
    news_lines = []
    for n in news[:30]:
        news_lines.append(f"- [{n.get('source', '')}] {n.get('title', '')}")
    news_text = "\n".join(news_lines) if news_lines else "（暂无新闻）"

    try:
        llm = _get_llm(config)

        # === Round 1: Sector predictions + stock candidates ===
        logger.info("Round 1: generating sector predictions...")
        round1_prompt = _ROUND1_PROMPT.format(
            signals_text=signals_text,
            news_text=news_text,
        )
        round1_content = _invoke_llm(llm, round1_prompt)
        round1_result = _parse_prediction_response(round1_content)
        if not round1_result:
            raise HTTPException(status_code=500, detail="第一轮 LLM 返回格式解析失败")

        # Save sector predictions
        predictions_to_save = []
        for sp in round1_result.get("sector_predictions", []):
            predictions_to_save.append({
                "predict_date": today,
                "trade_date": today,
                "category": "sector",
                "target": sp.get("target", ""),
                "direction": sp.get("direction", ""),
                "confidence": sp.get("confidence", 0),
                "reasoning": sp.get("reasoning", ""),
                "entry_price": "",
                "target_price": "",
                "stop_loss": "",
                "time_horizon": sp.get("time_horizon", ""),
                "status": "pending",
            })

        sector_context = "; ".join(
            f"{sp.get('target', '')} {sp.get('direction', '')}"
            for sp in round1_result.get("sector_predictions", [])
        )

        # === Resolve stock codes for round 2 ===
        recommended = round1_result.get("recommended_stocks", [])
        logger.info("Round 1 recommended %d stocks: %s", len(recommended), recommended)

        # Extract codes from recommended stock names like "利欧股份(002131)"
        code_pattern = re.compile(r'[（(]([A-Z]{0,2}\d{6})[）)]')
        stock_targets: list[tuple[str, str]] = []  # (name_with_code, pure_code)
        for stock_str in recommended[:8]:
            match = code_pattern.search(stock_str)
            if match:
                raw = match.group(1)
                pure = re.sub(r'^(SZ|SH|BJ)', '', raw)
                stock_targets.append((stock_str, pure))

        # Also try codes from news if LLM didn't give enough
        if len(stock_targets) < 3:
            news_codes, news_names = _extract_stock_codes(news)
            for code in news_codes:
                if code not in {t[1] for t in stock_targets}:
                    name = news_names.get(code, code)
                    stock_targets.append((f"{name}({code})", code))
                if len(stock_targets) >= 8:
                    break

        # === Round 2: Per-stock analysis with K-line ===
        history: dict[str, list[dict]] = {}
        if stock_targets:
            codes_to_fetch = [t[1] for t in stock_targets]
            logger.info("Fetching K-line history for %d stocks...", len(codes_to_fetch))
            history = _fetch_stock_history(codes_to_fetch, days=10)
            logger.info("Got history for %d stocks", len(history))

            stock_count = 0
            for stock_name, pure_code in stock_targets:
                rows = history.get(pure_code)
                if not rows:
                    continue
                history_text = _format_single_stock_history(rows)
                round2_prompt = _ROUND2_PROMPT.format(
                    stock_name=stock_name,
                    sector_context=sector_context,
                    history_text=history_text,
                )
                logger.info("Round 2: analyzing %s...", stock_name)
                try:
                    r2_content = _invoke_llm(llm, round2_prompt)
                    r2_result = _parse_prediction_response(r2_content)
                    if r2_result:
                        predictions_to_save.append({
                            "predict_date": today,
                            "trade_date": today,
                            "category": "stock",
                            "target": r2_result.get("target", stock_name),
                            "direction": r2_result.get("direction", ""),
                            "confidence": r2_result.get("confidence", 0),
                            "reasoning": r2_result.get("reasoning", ""),
                            "entry_price": str(r2_result.get("entry_price", "")),
                            "target_price": str(r2_result.get("target_price", "")),
                            "stop_loss": str(r2_result.get("stop_loss", "")),
                            "time_horizon": r2_result.get("time_horizon", ""),
                            "status": "pending",
                        })
                        stock_count += 1
                except Exception:
                    logger.warning("Round 2 failed for %s", stock_name, exc_info=True)

        # === Round 3: Synthesize all analyses into ranked portfolio ===
        stock_predictions = [p for p in predictions_to_save if p["category"] == "stock"]
        if stock_predictions:
            logger.info("Round 3: synthesizing %d stock analyses...", len(stock_predictions))
            sector_summary = "\n".join(
                f"- {p['target']} {p['direction']}（置信度 {p['confidence']:.0%}）"
                for p in predictions_to_save if p["category"] == "sector"
            )
            stock_analyses = "\n".join(
                f"- {p['target']} | {p['direction']} | 置信度:{p['confidence']:.0%} | "
                f"买入:{p['entry_price']} | 目标:{p['target_price']} | 止损:{p['stop_loss']} | "
                f"理由:{p['reasoning'][:80]}"
                for p in stock_predictions
            )
            round3_prompt = _ROUND3_PROMPT.format(
                sector_summary=sector_summary,
                stock_analyses=stock_analyses,
            )
            try:
                r3_content = _invoke_llm(llm, round3_prompt)
                r3_result = _parse_prediction_response(r3_content)
                if r3_result:
                    # Save top picks as "top_pick" category
                    for pick in r3_result.get("top_picks", []):
                        predictions_to_save.append({
                            "predict_date": today,
                            "trade_date": today,
                            "category": "top_pick",
                            "target": pick.get("target", ""),
                            "direction": pick.get("direction", ""),
                            "confidence": pick.get("confidence", 0),
                            "reasoning": pick.get("reasoning", ""),
                            "entry_price": str(pick.get("entry_price", "")),
                            "target_price": str(pick.get("target_price", "")),
                            "stop_loss": str(pick.get("stop_loss", "")),
                            "time_horizon": pick.get("time_horizon", ""),
                            "status": "pending",
                        })
                    # Save portfolio summary as a special record
                    summary = r3_result.get("portfolio_summary", "")
                    if summary:
                        predictions_to_save.append({
                            "predict_date": today,
                            "trade_date": today,
                            "category": "summary",
                            "target": "组合点评",
                            "direction": "",
                            "confidence": 0,
                            "reasoning": summary,
                            "entry_price": "",
                            "target_price": "",
                            "stop_loss": "",
                            "time_horizon": "",
                            "status": "pending",
                        })
                    logger.info("Round 3: got %d top picks", len(r3_result.get("top_picks", [])))
            except Exception:
                logger.warning("Round 3 synthesis failed", exc_info=True)

        store.save_predictions(predictions_to_save)
        sector_count = len([p for p in predictions_to_save if p["category"] == "sector"])
        stock_count = len([p for p in predictions_to_save if p["category"] == "stock"])
        top_count = len([p for p in predictions_to_save if p["category"] == "top_pick"])
        return {
            "status": "ok",
            "date": today,
            "sector_count": sector_count,
            "stock_count": stock_count,
            "top_pick_count": top_count,
            "history_stocks": len(history),
        }
    except HTTPException:
        raise
    except Exception:
        logger.error("Prediction generation failed", exc_info=True)
        raise HTTPException(status_code=500, detail="预测生成失败，请检查 LLM 配置")


def _parse_prediction_response(content: str) -> dict | None:
    """Parse LLM JSON response for predictions."""
    text = content.strip()
    # Strip markdown fences
    if "```" in text:
        lines = text.split("\n")
        filtered = []
        inside = False
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside:
                filtered.append(line)
        text = "\n".join(filtered).strip()

    # Try to find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        logger.warning("Failed to parse prediction JSON: %s", text[:300])
    return None


# ---------------------------------------------------------------------------
# Prediction verification
# ---------------------------------------------------------------------------

_CODE_PATTERN = re.compile(r'[（(]([A-Z]{0,2}\d{6})[）)]')
_RANGE_PATTERN = re.compile(r'([\d.]+)\s*[-~]\s*([\d.]+)')


def _parse_price(price_str: str) -> float | None:
    """Parse a price string, handling ranges by returning midpoint."""
    if not price_str:
        return None
    m = _RANGE_PATTERN.search(price_str)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    nums = re.findall(r'[\d.]+', price_str)
    if nums:
        return float(nums[0])
    return None


def _fetch_verification_data(code: str, start_date: str, end_date: str) -> list[dict]:
    """Fetch actual K-line data for a stock code via BaoStock."""
    pure = re.sub(r'^(SZ|SH|BJ)', '', code)
    if pure.startswith("6"):
        symbol = f"sh.{pure}"
    elif pure.startswith(("0", "3")):
        symbol = f"sz.{pure}"
    elif pure.startswith(("4", "8")):
        symbol = f"bj.{pure}"
    else:
        symbol = f"sz.{pure}"

    lg = baostock.login()
    if lg.error_code != '0':
        logger.warning("BaoStock login failed: %s", lg.error_msg)
        return []

    rows: list[dict] = []
    try:
        rs = baostock.query_history_k_data_plus(
            symbol,
            "date,open,high,low,close,volume,pctChg,turn",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",
        )
        if rs.error_code != '0':
            return []
        while rs.next():
            r = rs.get_row_data()
            try:
                rows.append({
                    "date": r[0],
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": int(float(r[5])),
                    "change_pct": float(r[6]) if r[6] else 0,
                    "turnover": float(r[7]) if r[7] else 0,
                })
            except (ValueError, IndexError):
                continue
    finally:
        baostock.logout()
    return rows


def verify_predictions(store: PostgresStore, days_back: int = 3) -> dict:
    """Verify pending predictions against actual market data.

    Returns summary dict with verified/skipped/errors counts.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    pending = store.get_pending_predictions(today)

    verified = 0
    skipped = 0
    errors = 0

    for pred in pending:
        try:
            # Extract stock code from target
            match = _CODE_PATTERN.search(pred.get("target", ""))
            if not match:
                skipped += 1
                continue

            raw_code = match.group(1)
            pure_code = re.sub(r'^(SZ|SH|BJ)', '', raw_code)

            # Parse prices
            entry_price = _parse_price(pred.get("entry_price", ""))
            target_price = _parse_price(pred.get("target_price", ""))
            stop_loss = _parse_price(pred.get("stop_loss", ""))

            if entry_price is None:
                skipped += 1
                continue

            # Fetch actual market data after predict_date
            predict_date = str(pred["predict_date"])
            start = (datetime.strptime(predict_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            end = today

            actual_rows = _fetch_verification_data(pure_code, start, end)
            if not actual_rows:
                skipped += 1
                continue

            # Compute verification metrics
            actual_close = actual_rows[-1]["close"]
            actual_high = max(r["high"] for r in actual_rows)
            actual_low = min(r["low"] for r in actual_rows)
            actual_return_pct = round((actual_close - entry_price) / entry_price * 100, 2)

            direction = pred.get("direction", "")
            is_bullish = "涨" in direction or "多" in direction
            is_bearish = "跌" in direction or "空" in direction

            # Direction hit
            if is_bullish:
                direction_hit = actual_close > entry_price
            elif is_bearish:
                direction_hit = actual_close < entry_price
            else:
                direction_hit = False

            # Target hit
            if is_bullish and target_price is not None:
                target_hit = actual_high >= target_price
            elif is_bearish and target_price is not None:
                target_hit = actual_low <= target_price
            else:
                target_hit = False

            # Stop hit
            if is_bullish and stop_loss is not None:
                stop_hit = actual_low <= stop_loss
            elif is_bearish and stop_loss is not None:
                stop_hit = actual_high >= stop_loss
            else:
                stop_hit = False

            # Determine result and status
            if target_hit and not stop_hit:
                result = "win"
                status = "verified_win"
            elif stop_hit and not target_hit:
                result = "loss"
                status = "verified_loss"
            elif target_hit and stop_hit:
                result = "partial_win"
                status = "verified_partial"
            elif direction_hit:
                result = "partial_win"
                status = "verified_partial"
            else:
                result = "neutral"
                status = "verified_neutral"

            # Compute daily returns (close-to-close, first day uses entry_price)
            daily_returns = []
            prev_close = entry_price
            for r in actual_rows:
                day_ret = round((r["close"] - prev_close) / prev_close * 100, 2)
                daily_returns.append(day_ret)
                prev_close = r["close"]

            holding_days = len(actual_rows)

            # A-share round-trip cost: commission 0.025%*2 + stamp tax 0.05% = 0.1%
            cost_adjusted_return = round(actual_return_pct - 0.1, 2)

            actual_result = {
                "verify_date": today,
                "actual_close": actual_close,
                "actual_high": actual_high,
                "actual_low": actual_low,
                "actual_return_pct": actual_return_pct,
                "direction_hit": direction_hit,
                "target_hit": target_hit,
                "stop_hit": stop_hit,
                "result": result,
                "holding_days": holding_days,
                "daily_returns": daily_returns,
                "cost_adjusted_return": cost_adjusted_return,
            }

            store.update_prediction_result(pred["id"], actual_result, status)
            verified += 1

        except Exception:
            logger.warning("Failed to verify prediction %s", pred.get("id"), exc_info=True)
            errors += 1

    return {"verified": verified, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Verification endpoints
# ---------------------------------------------------------------------------

@router.post("/predictions/verify")
def trigger_verify(store: PostgresStore = Depends(get_store)):
    """Manually trigger prediction verification."""
    try:
        result = verify_predictions(store)
        return result
    except Exception:
        logger.error("Prediction verification failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction verification failed")


@router.get("/predictions/stats")
def prediction_stats(
    days: int = 30,
    store: PostgresStore = Depends(get_store),
):
    """Return prediction accuracy statistics."""
    return store.get_prediction_stats(days=days)
