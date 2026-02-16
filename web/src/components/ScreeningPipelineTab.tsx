import { useState } from 'react';
import { Table, Button, Space, message, Badge, Steps, Modal } from 'antd';
import {
  ReloadOutlined, FilterOutlined, LoadingOutlined,
  CheckCircleOutlined, StopOutlined,
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { ColumnsType } from 'antd/es/table';
import {
  fetchScreeningRuns, triggerPoolScreen, triggerSentimentScreen,
  triggerStage, cancelScreeningRun,
  type ScreeningRun,
} from '../api/pool';

const STAGES = [
  { key: 's0_count', label: 'S0 行业预筛' },
  { key: 's1_count', label: 'S1 情绪筛选' },
  { key: 's2_count', label: 'S2 LLM评估' },
  { key: 'stage3_count', label: 'S3 多空辩论' },
  { key: 'stage4_count', label: 'S4 入池筛选' },
] as const;

type StageKey = (typeof STAGES)[number]['key'];

/** Prerequisite count key for each stage (stage N needs stage N-1's count > 0) */
const STAGE_PREREQ: Record<number, StageKey | null> = {
  0: null,           // S0 has no prereq (full pipeline restart)
  1: 's0_count',     // S1 needs S0 candidates
  2: 's1_count',     // S2 needs S1 candidates
  3: 's2_count',     // S3 needs S2 candidates
  4: 'stage3_count', // S4 needs S3 candidates
};

function getCurrentStage(run: ScreeningRun): number {
  if (run.current_stage != null && run.current_stage >= 0) {
    return run.current_stage;
  }
  // Fallback: infer from counts (for old runs without current_stage)
  for (let i = 0; i < STAGES.length; i++) {
    if (run[STAGES[i].key as StageKey] === 0) return i;
  }
  return STAGES.length;
}

function stageDescription(run: ScreeningRun, idx: number, currentStage: number): string {
  // S3 with progress info
  if (idx === 3 && idx === currentStage && run.stage3_progress) {
    return `${run.stage3_progress} 辩论中`;
  }
  const count = run[STAGES[idx].key as StageKey];
  return count > 0 ? `${count} 只` : '';
}

export default function ScreeningPipelineTab() {
  const queryClient = useQueryClient();
  const [retryStage, setRetryStage] = useState<number | null>(null);

  const { data: screeningRuns = [], isLoading: runsLoading } = useQuery({
    queryKey: ['pool-screening-runs'],
    queryFn: () => fetchScreeningRuns(10),
    refetchInterval: (query) => {
      const runs = query.state.data;
      return runs?.some((r: ScreeningRun) => r.status === 'running') ? 3000 : false;
    },
  });

  const sentimentScreenMutation = useMutation({
    mutationFn: triggerSentimentScreen,
    onSuccess: () => {
      message.info('情绪预筛已加入队列');
      queryClient.invalidateQueries({ queryKey: ['pool-screening-runs'] });
    },
    onError: () => message.error('情绪预筛失败'),
  });

  const screenMutation = useMutation({
    mutationFn: triggerPoolScreen,
    onSuccess: () => {
      message.info('全量筛选已加入队列');
      queryClient.invalidateQueries({ queryKey: ['pool-screening-runs'] });
    },
    onError: () => message.error('触发筛选失败'),
  });

  const stageMutation = useMutation({
    mutationFn: ({ stage, resume }: { stage: number; resume: boolean }) =>
      triggerStage(stage, resume),
    onSuccess: (_data, { stage, resume }) => {
      message.info(resume ? `S${stage} 断点续跑已加入队列` : `从 S${stage} 继续筛选已加入队列`);
      queryClient.invalidateQueries({ queryKey: ['pool-screening-runs'] });
    },
    onError: (err: Error) => message.error(err.message || '触发阶段筛选失败'),
  });

  const cancelMutation = useMutation({
    mutationFn: cancelScreeningRun,
    onSuccess: () => {
      message.info('筛选已取消');
      queryClient.invalidateQueries({ queryKey: ['pool-screening-runs'] });
    },
    onError: () => message.error('取消失败'),
  });

  const runningRun = screeningRuns.find((r) => r.status === 'running');
  const lastRun = screeningRuns[0];
  const isRunning = !!runningRun;
  const displayRun = runningRun ?? lastRun;
  const currentStage = displayRun ? getCurrentStage(displayRun) : -1;

  const handleStepClick = (stepIndex: number) => {
    // S0 cannot be retried independently; only S1-S4
    if (stepIndex < 1 || stepIndex > 4) return;
    // Don't allow retry while running
    if (isRunning) return;
    if (!lastRun) return;
    // Check prerequisite
    const prereqKey = STAGE_PREREQ[stepIndex];
    if (prereqKey && !lastRun[prereqKey]) return;

    setRetryStage(stepIndex);
  };

  const canResume = retryStage === 3
    && lastRun
    && lastRun.stage3_count > 0
    && lastRun.status !== 'completed';

  const statusMap: Record<string, string> = {
    completed: '完成',
    running: '运行中',
    error: '失败',
    failed: '失败',
    cancelled: '已取消',
  };

  const runColumns: ColumnsType<ScreeningRun> = [
    {
      title: '日期', dataIndex: 'run_date', key: 'run_date', width: 100,
      render: (v: string) => v?.slice(0, 10),
    },
    {
      title: '状态', dataIndex: 'status', key: 'status', width: 80,
      render: (v: string) => (
        <Badge
          status={v === 'completed' ? 'success' : v === 'running' ? 'processing' : v === 'cancelled' ? 'warning' : 'error'}
          text={statusMap[v] || v}
        />
      ),
    },
    { title: 'S0 行业', dataIndex: 's0_count', key: 's0', width: 75 },
    { title: 'S1 情绪', dataIndex: 's1_count', key: 's1', width: 75 },
    { title: 'S2 评估', dataIndex: 's2_count', key: 's2', width: 75 },
    { title: 'S3 辩论', dataIndex: 'stage3_count', key: 'stage3', width: 75 },
    { title: 'S4 入池', dataIndex: 'stage4_count', key: 'stage4', width: 75 },
    {
      title: '耗时', dataIndex: 'duration_sec', key: 'duration', width: 80,
      render: (v: number | null) => v !== null ? `${v.toFixed(1)}s` : '-',
    },
    {
      title: '错误', dataIndex: 'error_message', key: 'error',
      ellipsis: true,
      render: (v: string | null) => v || '-',
    },
  ];

  return (
    <>
      <Space style={{ marginBottom: 16 }}>
        <Button
          icon={<FilterOutlined />}
          onClick={() => sentimentScreenMutation.mutate()}
          loading={sentimentScreenMutation.isPending}
        >
          S0+S1 情绪预筛
        </Button>
        <Button
          type="primary"
          icon={<FilterOutlined />}
          onClick={() => screenMutation.mutate()}
          loading={screenMutation.isPending}
        >
          全量筛选 (S0→S4)
        </Button>
        {isRunning && runningRun && (
          <Button
            danger
            icon={<StopOutlined />}
            onClick={() => cancelMutation.mutate(runningRun.id)}
            loading={cancelMutation.isPending}
          >
            取消
          </Button>
        )}
        <Button
          icon={<ReloadOutlined />}
          onClick={() => queryClient.invalidateQueries({ queryKey: ['pool-screening-runs'] })}
        >
          刷新
        </Button>
      </Space>

      {displayRun && (
        <div style={{ marginBottom: 16 }}>
          <Steps
            size="small"
            current={currentStage}
            style={{ cursor: isRunning ? 'default' : 'pointer' }}
            items={STAGES.map((stage, idx) => {
              const desc = stageDescription(displayRun, idx, currentStage);
              return {
                title: (
                  <span onClick={() => handleStepClick(idx)}>
                    {stage.label}
                    {desc && (
                      <span style={{ fontSize: 12, fontWeight: 'normal', color: '#888', marginLeft: 4 }}>
                        {desc}
                      </span>
                    )}
                  </span>
                ),
                icon: isRunning && idx === currentStage ? <LoadingOutlined /> :
                  idx < currentStage ? <CheckCircleOutlined /> : undefined,
              };
            })}
          />
        </div>
      )}

      <Modal
        title={retryStage != null ? `从 S${retryStage} 重新开始？` : ''}
        open={retryStage != null}
        onCancel={() => setRetryStage(null)}
        getContainer={false}
        footer={retryStage != null ? (
          <Space>
            <Button onClick={() => setRetryStage(null)}>取消</Button>
            {canResume && (
              <Button
                type="primary"
                onClick={() => { stageMutation.mutate({ stage: retryStage, resume: true }); setRetryStage(null); }}
              >
                断点续跑 (已完成 {lastRun!.stage3_count}/{lastRun!.s2_count})
              </Button>
            )}
            <Button
              type={canResume ? 'default' : 'primary'}
              onClick={() => { stageMutation.mutate({ stage: retryStage, resume: false }); setRetryStage(null); }}
            >
              全部重跑
            </Button>
          </Space>
        ) : null}
      >
        {retryStage != null && (
          <p>将使用上一次运行的候选人，从「{STAGES[retryStage].label}」阶段重新筛选。</p>
        )}
        {canResume && (
          <p style={{ color: '#1677ff' }}>
            上次已完成 {lastRun!.stage3_count} 只辩论，可选择断点续跑。
          </p>
        )}
      </Modal>

      <Table
        columns={runColumns}
        dataSource={screeningRuns}
        rowKey="id"
        loading={runsLoading}
        size="small"
        pagination={{ pageSize: 10 }}
      />
    </>
  );
}
