import { apiFetch } from './client';

// -- Types --

export interface PoolStock {
  id: number;
  code: string;
  code_name: string;
  industry: string;
  entry_date: string;
  entry_price: number;
  target_price: number | null;
  stop_loss: number | null;
  outlook_days: number;
  debate_signal: string | null;
  debate_summary: string;
  screening_id: number | null;
  status: string;
  exit_date: string | null;
  exit_price: number | null;
  exit_reason: string | null;
  return_pct: number | null;
  total_predictions: number;
  correct_predictions: number;
  accuracy_pct: number | null;
  created_at: string;
}

export interface PoolPrediction {
  id: number;
  pool_id: number;
  code: string;
  predict_date: string;
  predicted_direction: string;
  confidence: number;
  reasoning: string;
  actual_close: number | null;
  actual_change_pct: number | null;
  actual_direction: string | null;
  is_accurate: boolean | null;
  evaluated_at: string | null;
  created_at: string;
}

export interface PoolStats {
  total_active: number;
  total_exited: number;
  avg_accuracy: number;
  total_predictions_today: number;
}

export interface PoolLesson {
  id: number;
  pool_id: number | null;
  prediction_id: number | null;
  code: string;
  predicted_direction: string | null;
  actual_direction: string | null;
  lesson: string;
  strategy_impact: string;
  created_at: string;
}

export interface ScreeningRun {
  id: number;
  run_date: string;
  s0_count: number;
  s1_count: number;
  s2_count: number;
  stage3_count: number;
  stage4_count: number;
  duration_sec: number | null;
  status: string;
  error_message: string | null;
  created_at: string;
  current_stage: number;
  stage3_progress: string | null;
}

export interface StrategyVersion {
  id: number;
  version: number;
  strategy_text: string;
  screening_criteria: string;
  prediction_criteria: string;
  source_lessons: string;
  is_active: boolean;
  created_at: string;
}

export interface ActiveStrategyResponse {
  active: boolean;
  message?: string;
  id?: number;
  version?: number;
  strategy_text?: string;
  screening_criteria?: string;
  prediction_criteria?: string;
  is_active?: boolean;
  created_at?: string;
}

// -- API calls --

export async function fetchPoolStocks(status = 'active', limit = 100): Promise<PoolStock[]> {
  return apiFetch<PoolStock[]>(`/pool/stocks?status=${status}&limit=${limit}`);
}

export async function fetchPoolStock(poolId: number): Promise<PoolStock> {
  return apiFetch<PoolStock>(`/pool/stocks/${poolId}`);
}

export async function exitPoolStock(poolId: number, data: {
  exit_date: string;
  exit_price: number;
  exit_reason: string;
}): Promise<{ status: string }> {
  return apiFetch(`/pool/stocks/${poolId}/exit`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function fetchPoolPredictions(params?: {
  predict_date?: string;
  pool_id?: number;
}): Promise<PoolPrediction[]> {
  const qs = new URLSearchParams();
  if (params?.predict_date) qs.set('predict_date', params.predict_date);
  if (params?.pool_id) qs.set('pool_id', String(params.pool_id));
  const query = qs.toString();
  return apiFetch<PoolPrediction[]>(`/pool/predictions${query ? '?' + query : ''}`);
}

export async function fetchPoolStats(): Promise<PoolStats> {
  return apiFetch<PoolStats>('/pool/stats');
}

export async function fetchPoolLessons(params?: {
  pool_id?: number;
  limit?: number;
}): Promise<PoolLesson[]> {
  const qs = new URLSearchParams();
  if (params?.pool_id) qs.set('pool_id', String(params.pool_id));
  if (params?.limit) qs.set('limit', String(params.limit));
  const query = qs.toString();
  return apiFetch<PoolLesson[]>(`/pool/lessons${query ? '?' + query : ''}`);
}

export async function triggerPoolPredict(): Promise<{ predicted: number; errors: number }> {
  return apiFetch('/pool/trigger/predict', { method: 'POST' });
}

export async function fetchScreeningRuns(limit = 10): Promise<ScreeningRun[]> {
  return apiFetch<ScreeningRun[]>(`/pool/screening-runs?limit=${limit}`);
}

export async function triggerPoolScreen(): Promise<{ status: string; message: string }> {
  return apiFetch('/pool/trigger/screen', { method: 'POST' });
}

export async function triggerStage(
  stage: number,
  resume = false,
): Promise<{ task_id: string; status: string; from_stage: number }> {
  const qs = resume ? '?resume=true' : '';
  return apiFetch(`/pool/trigger/stage/${stage}${qs}`, { method: 'POST' });
}

export async function triggerSentimentScreen(): Promise<{
  task_id: string;
  status: string;
}> {
  return apiFetch('/pool/trigger/sentiment-screen', { method: 'POST' });
}

export async function fetchStrategies(limit = 10): Promise<StrategyVersion[]> {
  return apiFetch<StrategyVersion[]>(`/pool/strategies?limit=${limit}`);
}

export async function fetchActiveStrategy(): Promise<ActiveStrategyResponse> {
  return apiFetch<ActiveStrategyResponse>('/pool/strategies/active');
}

export async function triggerPoolEvolve(): Promise<{ evolved: boolean; message?: string; strategy_text?: string }> {
  return apiFetch('/pool/trigger/evolve', { method: 'POST' });
}

export async function triggerPoolEvaluate(): Promise<{
  evaluated: number;
  accurate: number;
  inaccurate: number;
  skipped: number;
  lessons_generated: number;
}> {
  return apiFetch('/pool/trigger/evaluate', { method: 'POST' });
}

export interface SentimentStats {
  positive_count: number;
  negative_count: number;
  avg_score: number | null;
}

export async function triggerSentimentAnalysis(): Promise<{ analyzed: number; message?: string }> {
  return apiFetch('/pool/trigger/sentiment', { method: 'POST' });
}

export async function fetchSentimentStats(date?: string): Promise<SentimentStats> {
  const qs = date ? `?date=${date}` : '';
  return apiFetch<SentimentStats>(`/pool/sentiment-stats${qs}`);
}

// -- Fine-tuning --

export interface FinetuneStatus {
  state: 'idle' | 'preparing' | 'training' | 'done' | 'error';
  progress: number;
  message: string;
  epoch?: number;
  total_epochs?: number;
  loss?: number;
  eval_f1?: number;
  eval_accuracy?: number;
  output_dir?: string;
}

export async function triggerFinetune(params?: {
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
}): Promise<{ started: boolean; message: string }> {
  return apiFetch('/pool/trigger/finetune', {
    method: 'POST',
    body: JSON.stringify(params ?? {}),
  });
}

export async function fetchFinetuneStatus(): Promise<FinetuneStatus> {
  return apiFetch<FinetuneStatus>('/pool/finetune-status');
}

export async function triggerExitCheck(): Promise<{
  checked: boolean;
  exits: number;
  details: Array<{ pool_id: number; code: string; reason: string; exit_price: number }>;
}> {
  return apiFetch('/pool/trigger/exit-check', { method: 'POST' });
}

// -- Quick Analysis --

export interface QuickAnalysisPrediction {
  id: number;
  target: string;
  category: string;
  direction: string;
  target_price: string | null;
  stop_loss: string | null;
  entry_price: string | null;
  confidence: number;
  reasoning: string;
  time_horizon: string;
  status: string;
  created_at: string;
}

export async function triggerQuickAnalysis(): Promise<{
  status: string;
  date: string;
  sector_count: number;
  stock_count: number;
  top_pick_count: number;
}> {
  return apiFetch('/pool/trigger/quick-analysis', { method: 'POST' });
}

export async function fetchQuickAnalysis(date?: string): Promise<QuickAnalysisPrediction[]> {
  const qs = date ? `?date=${date}` : '';
  return apiFetch<QuickAnalysisPrediction[]>(`/pool/quick-analysis${qs}`);
}

export async function promoteToPool(predictionId: number): Promise<{
  pool_id: number;
  stock: PoolStock;
}> {
  return apiFetch(`/pool/promote/${predictionId}`, { method: 'POST' });
}

export async function triggerPoolVerify(): Promise<{
  verified: number;
  skipped: number;
  errors: number;
}> {
  return apiFetch('/pool/trigger/verify', { method: 'POST' });
}

// -- Re-Debate --

export async function triggerRedebateCheck(): Promise<{
  checked: number;
  triggered: number;
  details: Array<{ pool_id: number; code: string; reasons: string[] }>;
}> {
  return apiFetch('/pool/trigger/redebate-check', { method: 'POST' });
}

export async function triggerPoolDebate(poolId: number): Promise<{
  pool_id: number;
  debate_status: string;
  task_id: string;
}> {
  return apiFetch(`/pool/stocks/${poolId}/trigger-debate`, { method: 'POST' });
}

export async function retryPoolDebate(poolId: number): Promise<{
  pool_id: number;
  debate_status: string;
  task_id: string;
}> {
  return apiFetch(`/pool/stocks/${poolId}/retry-debate`, { method: 'POST' });
}

// -- Experience Loop --

export interface PoolSummary {
  id: number;
  summary_date: string;
  period_days: number;
  total_stocks: number;
  win_rate: number;
  avg_return: number;
  key_lessons: string;
  winning_patterns: string;
  losing_patterns: string;
  created_at: string;
}

export async function triggerPoolSummarize(): Promise<{
  summary_id: number | null;
  total_stocks?: number;
  win_rate?: number;
}> {
  return apiFetch('/pool/trigger/summarize', { method: 'POST' });
}

export async function fetchPoolSummaries(limit = 10): Promise<PoolSummary[]> {
  return apiFetch<PoolSummary[]>(`/pool/summaries?limit=${limit}`);
}

export async function fetchPoolAdjustments(limit = 10): Promise<unknown[]> {
  return apiFetch(`/pool/adjustments?limit=${limit}`);
}

export async function fetchPoolStockDebate(poolId: number): Promise<Record<string, unknown>> {
  return apiFetch(`/pool/stocks/${poolId}/debate`);
}

export async function fetchPoolStockLessons(poolId: number): Promise<PoolLesson[]> {
  return apiFetch<PoolLesson[]>(`/pool/stocks/${poolId}/lessons`);
}

export async function cancelScreeningRun(runId: number): Promise<{
  status: string;
  run_id: number;
}> {
  return apiFetch(`/pool/cancel/${runId}`, { method: 'POST' });
}
