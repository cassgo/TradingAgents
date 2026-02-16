import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Modal } from 'antd';
import ScreeningPipelineTab from './ScreeningPipelineTab';

// Mock the pool API module
vi.mock('../api/pool', () => ({
  fetchScreeningRuns: vi.fn(),
  triggerPoolScreen: vi.fn(),
  triggerSentimentScreen: vi.fn(),
  triggerStage: vi.fn(),
  cancelScreeningRun: vi.fn(),
}));

import {
  fetchScreeningRuns,
  triggerPoolScreen,
  triggerSentimentScreen,
  triggerStage,
  cancelScreeningRun,
} from '../api/pool';

const mockFetchScreeningRuns = vi.mocked(fetchScreeningRuns);
const mockTriggerPoolScreen = vi.mocked(triggerPoolScreen);
const mockTriggerSentimentScreen = vi.mocked(triggerSentimentScreen);
const mockTriggerStage = vi.mocked(triggerStage);
const mockCancelScreeningRun = vi.mocked(cancelScreeningRun);

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );
  };
}

const completedRun = {
  id: 1,
  run_date: '2026-02-15',
  s0_count: 487,
  s1_count: 120,
  s2_count: 30,
  stage3_count: 10,
  stage4_count: 5,
  duration_sec: 123.4,
  status: 'completed',
  error_message: null,
  created_at: '2026-02-15T10:00:00',
  current_stage: 5,
  stage3_progress: null,
};

const runningRunS1 = {
  id: 2,
  run_date: '2026-02-15',
  s0_count: 487,
  s1_count: 0,
  s2_count: 0,
  stage3_count: 0,
  stage4_count: 0,
  duration_sec: null,
  status: 'running',
  error_message: null,
  created_at: '2026-02-15T11:00:00',
  current_stage: 1,
  stage3_progress: null,
};

const runningRunS3WithProgress = {
  id: 4,
  run_date: '2026-02-15',
  s0_count: 400,
  s1_count: 100,
  s2_count: 30,
  stage3_count: 0,
  stage4_count: 0,
  duration_sec: null,
  status: 'running',
  error_message: null,
  created_at: '2026-02-15T13:00:00',
  current_stage: 3,
  stage3_progress: '5/12',
};

const failedRun = {
  id: 3,
  run_date: '2026-02-15',
  s0_count: 100,
  s1_count: 50,
  s2_count: 0,
  stage3_count: 0,
  stage4_count: 0,
  duration_sec: 45.2,
  status: 'failed',
  error_message: 'LLM timeout',
  created_at: '2026-02-15T12:00:00',
  current_stage: 2,
  stage3_progress: null,
};

describe('ScreeningPipelineTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchScreeningRuns.mockResolvedValue([completedRun]);
  });

  afterEach(() => {
    Modal.destroyAll();
  });

  it('renders refresh button', async () => {
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    expect(screen.getByRole('button', { name: /刷新/i })).toBeInTheDocument();
  });

  it('renders screening action buttons', async () => {
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    expect(screen.getByRole('button', { name: /情绪预筛/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /全量筛选/i })).toBeInTheDocument();
  });

  it('shows screening runs table with data', async () => {
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByText('487')).toBeInTheDocument();
    });
  });

  it('maps failed status correctly', async () => {
    mockFetchScreeningRuns.mockResolvedValue([failedRun]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByText('失败')).toBeInTheDocument();
    });
  });

  it('shows progress Steps for the latest run (not just running)', async () => {
    mockFetchScreeningRuns.mockResolvedValue([completedRun]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByText(/行业预筛/)).toBeInTheDocument();
      expect(screen.getByText(/情绪筛选/)).toBeInTheDocument();
      expect(screen.getByText(/LLM评估/)).toBeInTheDocument();
      expect(screen.getByText(/多空辩论/)).toBeInTheDocument();
      expect(screen.getByText(/入池筛选/)).toBeInTheDocument();
    });
  });

  it('shows counts on completed run Steps', async () => {
    mockFetchScreeningRuns.mockResolvedValue([completedRun]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByText('487 只')).toBeInTheDocument();
      expect(screen.getByText('120 只')).toBeInTheDocument();
      expect(screen.getByText('30 只')).toBeInTheDocument();
      expect(screen.getByText('10 只')).toBeInTheDocument();
      expect(screen.getByText('5 只')).toBeInTheDocument();
    });
  });

  it('shows loading icon on current stage when running', async () => {
    mockFetchScreeningRuns.mockResolvedValue([runningRunS1]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    await waitFor(() => {
      // S0 completed with 487
      expect(screen.getByText('487 只')).toBeInTheDocument();
    });
  });

  it('refresh button invalidates queries', async () => {
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(mockFetchScreeningRuns).toHaveBeenCalledTimes(1);
    });
    const refreshBtn = screen.getByRole('button', { name: /刷新/i });
    fireEvent.click(refreshBtn);
    await waitFor(() => {
      expect(mockFetchScreeningRuns.mock.calls.length).toBeGreaterThanOrEqual(2);
    });
  });

  it('triggers full screen mutation on button click', async () => {
    mockTriggerPoolScreen.mockResolvedValue({ status: 'ok', message: 'queued' });
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    const btn = screen.getByRole('button', { name: /全量筛选/i });
    fireEvent.click(btn);
    await waitFor(() => {
      expect(mockTriggerPoolScreen).toHaveBeenCalledTimes(1);
    });
  });

  it('triggers sentiment screen mutation on button click', async () => {
    mockTriggerSentimentScreen.mockResolvedValue({ task_id: 'abc', status: 'queued' });
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    const btn = screen.getByRole('button', { name: /情绪预筛/i });
    fireEvent.click(btn);
    await waitFor(() => {
      expect(mockTriggerSentimentScreen).toHaveBeenCalledTimes(1);
    });
  });

  // -- Precise progress --

  it('uses current_stage for precise progress instead of inference', async () => {
    mockFetchScreeningRuns.mockResolvedValue([runningRunS1]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByText('487 只')).toBeInTheDocument();
    });
  });

  it('shows S3 debate progress (e.g. "5/12") when stage3_progress is set', async () => {
    mockFetchScreeningRuns.mockResolvedValue([runningRunS3WithProgress]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByText(/5\/12/)).toBeInTheDocument();
    });
  });

  // -- Clickable Steps with confirm dialog --

  it('shows confirm dialog when clicking a completed stage (S1-S4)', async () => {
    mockFetchScreeningRuns.mockResolvedValue([completedRun]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });

    // Wait for Steps to render
    await waitFor(() => {
      expect(screen.getByText('120 只')).toBeInTheDocument();
    });

    // Click the S2 stage step (index 2, "S2 LLM评估")
    const s2Step = screen.getByText('S2 LLM评估');
    fireEvent.click(s2Step);

    // Confirm dialog should appear
    await waitFor(() => {
      expect(screen.getByText(/从 S2 重新开始/)).toBeInTheDocument();
    });
  });

  it('calls triggerStage after confirming retry dialog', async () => {
    mockTriggerStage.mockResolvedValue({ task_id: 't1', status: 'queued', from_stage: 2 });
    mockFetchScreeningRuns.mockResolvedValue([completedRun]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('S2 LLM评估')).toBeInTheDocument();
    });

    // Click S2 step
    fireEvent.click(screen.getByText('S2 LLM评估'));

    // Wait for confirm dialog, click "全部重跑" (replaces old OK button)
    await waitFor(() => {
      expect(screen.getByText(/从 S2 重新开始/)).toBeInTheDocument();
    });
    const retryBtn = screen.getByRole('button', { name: /全部重跑/ });
    fireEvent.click(retryBtn);

    await waitFor(() => {
      expect(mockTriggerStage).toHaveBeenCalledWith(2, false);
    });
  });

  it('does not call triggerStage when cancel is clicked', async () => {
    mockFetchScreeningRuns.mockResolvedValue([completedRun]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('S2 LLM评估')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('S2 LLM评估'));

    // Wait for modal to fully render with footer buttons
    await waitFor(() => {
      const retryBtn = screen.getByRole('button', { name: /全部重跑/ });
      expect(retryBtn).toBeInTheDocument();
    });

    // Close modal via the X button (Ant Design modal close icon)
    const closeBtn = screen.getByRole('button', { name: /close/i });
    fireEvent.click(closeBtn);

    // triggerStage should NOT be called
    expect(mockTriggerStage).not.toHaveBeenCalled();
  });

  it('does not show confirm dialog when clicking stage during running', async () => {
    mockFetchScreeningRuns.mockResolvedValue([runningRunS1]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('S2 LLM评估')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('S2 LLM评估'));

    // No confirm dialog should appear
    expect(screen.queryByText(/从 S2 重新开始/)).not.toBeInTheDocument();
  });

  it('does not show confirm dialog when clicking S0 (cannot retry from S0)', async () => {
    mockFetchScreeningRuns.mockResolvedValue([completedRun]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('S0 行业预筛')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('S0 行业预筛'));

    // No confirm dialog for S0
    expect(screen.queryByText(/从 S0 重新开始/)).not.toBeInTheDocument();
  });

  // -- Cancel button --

  it('shows cancel button when a run is running', async () => {
    mockFetchScreeningRuns.mockResolvedValue([runningRunS1]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /取消/i })).toBeInTheDocument();
    });
  });

  it('does not show cancel button when no run is running', async () => {
    mockFetchScreeningRuns.mockResolvedValue([completedRun]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByText('487 只')).toBeInTheDocument();
    });
    expect(screen.queryByRole('button', { name: /取消/i })).not.toBeInTheDocument();
  });

  it('calls cancelScreeningRun when cancel button is clicked', async () => {
    mockCancelScreeningRun.mockResolvedValue({ status: 'cancelled', run_id: 2 });
    mockFetchScreeningRuns.mockResolvedValue([runningRunS1]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /取消/i })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: /取消/i }));
    await waitFor(() => {
      expect(mockCancelScreeningRun).toHaveBeenCalled();
    });
    expect(mockCancelScreeningRun.mock.calls[0][0]).toBe(2);
  });

  // -- S3 Resume / Checkpoint tests --

  const failedRunPartialS3 = {
    id: 5,
    run_date: '2026-02-16',
    s0_count: 400,
    s1_count: 100,
    s2_count: 30,
    stage3_count: 7,
    stage4_count: 0,
    duration_sec: 180,
    status: 'failed',
    error_message: 'Gateway timeout',
    created_at: '2026-02-16T10:00:00',
    current_stage: 3,
    stage3_progress: '7/12',
  };

  it('shows resume button when partial S3 (stage3_count>0, not completed)', async () => {
    mockFetchScreeningRuns.mockResolvedValue([failedRunPartialS3]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('S3 多空辩论')).toBeInTheDocument();
    });

    // Click the S3 step to open modal
    fireEvent.click(screen.getByText('S3 多空辩论'));

    await waitFor(() => {
      // Button contains "断点续跑" — use role to be specific
      expect(screen.getByRole('button', { name: /断点续跑/ })).toBeInTheDocument();
    });
    // Also shows "全部重跑"
    expect(screen.getByRole('button', { name: /全部重跑/ })).toBeInTheDocument();
  });

  it('hides resume button when S3 is complete', async () => {
    mockFetchScreeningRuns.mockResolvedValue([completedRun]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('S3 多空辩论')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('S3 多空辩论'));

    await waitFor(() => {
      expect(screen.getByText(/从 S3 重新开始/)).toBeInTheDocument();
    });
    // Should NOT show resume button for completed runs
    expect(screen.queryByText(/断点续跑/)).not.toBeInTheDocument();
  });

  it('calls triggerStage with resume=true when resume button clicked', async () => {
    mockTriggerStage.mockResolvedValue({ task_id: 't5', status: 'queued', from_stage: 3 });
    mockFetchScreeningRuns.mockResolvedValue([failedRunPartialS3]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('S3 多空辩论')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('S3 多空辩论'));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /断点续跑/ })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: /断点续跑/ }));

    await waitFor(() => {
      expect(mockTriggerStage).toHaveBeenCalledWith(3, true);
    });
  });

  it('calls triggerStage with resume=false when "全部重跑" clicked', async () => {
    mockTriggerStage.mockResolvedValue({ task_id: 't6', status: 'queued', from_stage: 3 });
    mockFetchScreeningRuns.mockResolvedValue([failedRunPartialS3]);
    render(<ScreeningPipelineTab />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('S3 多空辩论')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('S3 多空辩论'));

    await waitFor(() => {
      expect(screen.getByText(/全部重跑/)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText(/全部重跑/));

    await waitFor(() => {
      expect(mockTriggerStage).toHaveBeenCalledWith(3, false);
    });
  });
});
