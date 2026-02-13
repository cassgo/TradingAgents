import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Button, Table, Tag, Progress, Card, Row, Col, Typography, Tabs,
  Tooltip, message, Spin, Empty, Statistic,
} from 'antd';
import {
  RocketOutlined, ArrowUpOutlined, ArrowDownOutlined, MinusOutlined,
  ReloadOutlined, CheckCircleOutlined, CloseCircleOutlined,
  ExclamationCircleOutlined, ClockCircleOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { apiFetch } from '../api/client';

const { Title, Paragraph } = Typography;

interface Prediction {
  id: number;
  predict_date: string;
  trade_date: string;
  category: string;
  target: string;
  direction: string;
  confidence: number;
  reasoning: string;
  entry_price: string;
  target_price: string;
  stop_loss: string;
  time_horizon: string;
  status: string;
  actual_result: string;
  created_at: string;
}

interface PredictionStats {
  total_verified: number;
  wins: number;
  losses: number;
  partial_wins: number;
  win_rate: number;
  avg_return_pct: number;
  profit_factor: number;
  expectancy: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  avg_holding_days: number;
  sample_size: number;
}

interface VerifyResult {
  verified: number;
  skipped: number;
  errors: number;
}

const statusTag = (status: string) => {
  switch (status) {
    case 'verified_win':
      return <Tag color="success" icon={<CheckCircleOutlined />}>WIN</Tag>;
    case 'verified_loss':
      return <Tag color="error" icon={<CloseCircleOutlined />}>LOSS</Tag>;
    case 'verified_partial':
      return <Tag color="warning" icon={<ExclamationCircleOutlined />}>PARTIAL</Tag>;
    case 'verified_neutral':
      return <Tag color="default">NEUTRAL</Tag>;
    default:
      return <Tag color="processing" icon={<ClockCircleOutlined />}>PENDING</Tag>;
  }
};

export default function Predictions() {
  const [generating, setGenerating] = useState(false);
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ['predictions'],
    queryFn: () => apiFetch<Prediction[]>('/predictions?limit=100'),
  });

  const { data: stats } = useQuery({
    queryKey: ['prediction-stats'],
    queryFn: () => apiFetch<PredictionStats>('/predictions/stats'),
  });

  const verifyMutation = useMutation({
    mutationFn: () => apiFetch<VerifyResult>('/predictions/verify', { method: 'POST' }),
    onSuccess: (result) => {
      message.success(`验证完成：${result.verified} 已验证, ${result.skipped} 跳过`);
      queryClient.invalidateQueries({ queryKey: ['predictions'] });
      queryClient.invalidateQueries({ queryKey: ['prediction-stats'] });
    },
    onError: (err) => {
      message.error(err instanceof Error ? err.message : '验证失败');
    },
  });

  const sectorData = data?.filter((p) => p.category === 'sector') ?? [];
  const stockData = data?.filter((p) => p.category === 'stock') ?? [];
  const topPickData = data?.filter((p) => p.category === 'top_pick') ?? [];
  const summaryData = data?.find((p) => p.category === 'summary');
  const verifiedData = data?.filter((p) => p.status.startsWith('verified_')) ?? [];

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const result = await apiFetch<{ status: string; sector_count: number; stock_count: number; top_pick_count?: number }>(
        '/predictions/generate',
        { method: 'POST' },
      );
      message.success(`预测完成：${result.sector_count} 板块 + ${result.stock_count} 个股 + ${result.top_pick_count ?? 0} 精选`);
      queryClient.invalidateQueries({ queryKey: ['predictions'] });
    } catch (err) {
      message.error(err instanceof Error ? err.message : '预测生成失败');
    } finally {
      setGenerating(false);
    }
  };

  const directionTag = (direction: string) => {
    const d = direction.toLowerCase();
    if (d.includes('涨') || d.includes('多') || d.includes('up') || d.includes('bull')) {
      return <Tag color="red" icon={<ArrowUpOutlined />}>{direction}</Tag>;
    }
    if (d.includes('跌') || d.includes('空') || d.includes('down') || d.includes('bear')) {
      return <Tag color="green" icon={<ArrowDownOutlined />}>{direction}</Tag>;
    }
    return <Tag color="blue" icon={<MinusOutlined />}>{direction}</Tag>;
  };

  const sectorColumns: ColumnsType<Prediction> = [
    {
      title: '板块',
      dataIndex: 'target',
      key: 'target',
      width: '18%',
      render: (text: string) => <strong>{text}</strong>,
    },
    {
      title: '方向',
      dataIndex: 'direction',
      key: 'direction',
      width: '10%',
      render: (d: string) => directionTag(d),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: '14%',
      render: (c: number) => (
        <Progress
          percent={Math.round(c * 100)}
          size="small"
          status={c > 0.7 ? 'success' : c > 0.4 ? 'normal' : 'exception'}
        />
      ),
      sorter: (a, b) => a.confidence - b.confidence,
      defaultSortOrder: 'descend',
    },
    {
      title: '周期',
      dataIndex: 'time_horizon',
      key: 'time_horizon',
      width: '12%',
    },
    {
      title: '分析理由',
      dataIndex: 'reasoning',
      key: 'reasoning',
      ellipsis: true,
    },
  ];

  const stockColumns: ColumnsType<Prediction> = [
    {
      title: '股票',
      dataIndex: 'target',
      key: 'target',
      width: '15%',
      render: (text: string) => <strong>{text}</strong>,
    },
    {
      title: '方向',
      dataIndex: 'direction',
      key: 'direction',
      width: '8%',
      render: (d: string) => directionTag(d),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: '12%',
      render: (c: number) => (
        <Progress
          percent={Math.round(c * 100)}
          size="small"
          status={c > 0.7 ? 'success' : c > 0.4 ? 'normal' : 'exception'}
        />
      ),
      sorter: (a, b) => a.confidence - b.confidence,
      defaultSortOrder: 'descend',
    },
    {
      title: '买入价',
      dataIndex: 'entry_price',
      key: 'entry_price',
      width: '10%',
    },
    {
      title: '目标价',
      dataIndex: 'target_price',
      key: 'target_price',
      width: '10%',
    },
    {
      title: '止损价',
      dataIndex: 'stop_loss',
      key: 'stop_loss',
      width: '10%',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: '10%',
      render: (s: string) => statusTag(s),
    },
    {
      title: '理由',
      dataIndex: 'reasoning',
      key: 'reasoning',
      ellipsis: true,
    },
  ];

  const verifiedColumns: ColumnsType<Prediction> = [
    {
      title: '股票',
      dataIndex: 'target',
      key: 'target',
      width: '14%',
      render: (text: string) => <strong>{text}</strong>,
    },
    {
      title: '预测方向',
      dataIndex: 'direction',
      key: 'direction',
      width: '8%',
      render: (d: string) => directionTag(d),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: '10%',
      render: (s: string) => statusTag(s),
    },
    {
      title: '买入价',
      dataIndex: 'entry_price',
      key: 'entry_price',
      width: '8%',
    },
    {
      title: '目标价',
      dataIndex: 'target_price',
      key: 'target_price',
      width: '8%',
    },
    {
      title: '实际收盘',
      key: 'actual_close',
      width: '8%',
      render: (_: unknown, record: Prediction) => {
        try {
          const result = JSON.parse(record.actual_result);
          return result.actual_close?.toFixed(2) ?? '-';
        } catch {
          return '-';
        }
      },
    },
    {
      title: '实际收益',
      key: 'actual_return',
      width: '10%',
      render: (_: unknown, record: Prediction) => {
        try {
          const result = JSON.parse(record.actual_result);
          const pct = result.actual_return_pct;
          if (pct == null) return '-';
          const color = pct > 0 ? '#cf1322' : pct < 0 ? '#389e0d' : '#999';
          return <span style={{ color, fontWeight: 'bold' }}>{pct > 0 ? '+' : ''}{pct.toFixed(2)}%</span>;
        } catch {
          return '-';
        }
      },
      sorter: (a, b) => {
        try {
          return JSON.parse(a.actual_result).actual_return_pct - JSON.parse(b.actual_result).actual_return_pct;
        } catch {
          return 0;
        }
      },
    },
    {
      title: '方向命中',
      key: 'direction_hit',
      width: '8%',
      render: (_: unknown, record: Prediction) => {
        try {
          const result = JSON.parse(record.actual_result);
          return result.direction_hit
            ? <Tag color="success">HIT</Tag>
            : <Tag color="error">MISS</Tag>;
        } catch {
          return '-';
        }
      },
    },
    {
      title: '预测日期',
      dataIndex: 'predict_date',
      key: 'predict_date',
      width: '10%',
    },
    {
      title: '理由',
      dataIndex: 'reasoning',
      key: 'reasoning',
      ellipsis: true,
    },
  ];

  // Stats
  const bullSectors = sectorData.filter((p) => p.direction.includes('涨')).length;
  const bearSectors = sectorData.filter((p) => p.direction.includes('跌')).length;
  const bullStocks = stockData.filter((p) => p.direction.includes('涨')).length;
  const latestDate = data?.[0]?.predict_date ?? '-';

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>
          <RocketOutlined style={{ marginRight: 8 }} />
          市场预测
        </Title>
        <div style={{ display: 'flex', gap: 12 }}>
          <Tooltip title="手动触发验证昨日及之前的待验证预测">
            <Button
              size="large"
              icon={<CheckCircleOutlined />}
              loading={verifyMutation.isPending}
              onClick={() => verifyMutation.mutate()}
            >
              手动验证
            </Button>
          </Tooltip>
          <Tooltip title="基于今日信号和新闻，使用深度模型生成板块+个股预测">
            <Button
              type="primary"
              size="large"
              icon={<ReloadOutlined spin={generating} />}
              loading={generating}
              onClick={handleGenerate}
            >
              {generating ? '正在生成预测...' : '生成今日预测'}
            </Button>
          </Tooltip>
        </div>
      </div>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6} lg={4}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 'bold', color: '#1677ff' }}>{latestDate}</div>
              <div style={{ color: '#999' }}>预测日期</div>
            </div>
          </Card>
        </Col>
        <Col xs={12} sm={6} lg={4}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 'bold', color: '#cf1322' }}>{bullSectors}</div>
              <div style={{ color: '#999' }}>看涨板块</div>
            </div>
          </Card>
        </Col>
        <Col xs={12} sm={6} lg={4}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 'bold', color: '#389e0d' }}>{bearSectors}</div>
              <div style={{ color: '#999' }}>看跌板块</div>
            </div>
          </Card>
        </Col>
        <Col xs={12} sm={6} lg={4}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 'bold', color: '#d46b08' }}>{topPickData.length || bullStocks}</div>
              <div style={{ color: '#999' }}>精选推荐</div>
            </div>
          </Card>
        </Col>
        <Col xs={12} sm={6} lg={4}>
          <Card size="small">
            <Statistic
              title="预测胜率"
              value={stats?.win_rate ?? 0}
              suffix="%"
              precision={1}
              valueStyle={{ color: (stats?.win_rate ?? 0) >= 50 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6} lg={4}>
          <Card size="small">
            <Statistic
              title="平均收益"
              value={stats?.avg_return_pct ?? 0}
              suffix="%"
              precision={2}
              prefix={(stats?.avg_return_pct ?? 0) >= 0 ? '+' : ''}
              valueStyle={{ color: (stats?.avg_return_pct ?? 0) >= 0 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>

      {stats && stats.sample_size > 0 && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={12} sm={6} lg={3}>
            <Card size="small">
              <Statistic title="盈亏比" value={stats.profit_factor} precision={2} />
            </Card>
          </Col>
          <Col xs={12} sm={6} lg={3}>
            <Card size="small">
              <Statistic
                title="期望值"
                value={stats.expectancy}
                suffix="%"
                precision={2}
                valueStyle={{ color: stats.expectancy >= 0 ? '#3f8600' : '#cf1322' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6} lg={3}>
            <Card size="small">
              <Statistic title="夏普比率" value={stats.sharpe_ratio} precision={2} />
            </Card>
          </Col>
          <Col xs={12} sm={6} lg={3}>
            <Card size="small">
              <Statistic title="索提诺比率" value={stats.sortino_ratio} precision={2} />
            </Card>
          </Col>
          <Col xs={12} sm={6} lg={3}>
            <Card size="small">
              <Statistic
                title="最大回撤"
                value={stats.max_drawdown}
                suffix="%"
                precision={2}
                valueStyle={{ color: '#cf1322' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={6} lg={3}>
            <Card size="small">
              <Statistic title="卡玛比率" value={stats.calmar_ratio} precision={2} />
            </Card>
          </Col>
          <Col xs={12} sm={6} lg={3}>
            <Card size="small">
              <Statistic title="平均持仓天数" value={stats.avg_holding_days} precision={1} suffix="天" />
            </Card>
          </Col>
          <Col xs={12} sm={6} lg={3}>
            <Card size="small">
              <Statistic title="样本量" value={stats.sample_size} />
            </Card>
          </Col>
        </Row>
      )}

      {generating ? (
        <div style={{ textAlign: 'center', padding: '80px 0' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16, color: '#999', fontSize: 16 }}>
            深度模型正在分析市场信号和新闻，生成综合预测...
          </div>
        </div>
      ) : !data || data.length === 0 ? (
        <Empty description="暂无预测数据，点击「生成今日预测」开始" />
      ) : (
        <Tabs
          defaultActiveKey={topPickData.length > 0 ? 'top' : 'sector'}
          items={[
            ...(topPickData.length > 0
              ? [
                  {
                    key: 'top',
                    label: `精选推荐 (${topPickData.length})`,
                    children: (
                      <div>
                        {summaryData && (
                          <Card
                            size="small"
                            style={{ marginBottom: 16, background: '#f6ffed', borderColor: '#b7eb8f' }}
                          >
                            <Paragraph style={{ margin: 0, fontSize: 14 }}>
                              <strong>组合点评：</strong>
                              {summaryData.reasoning}
                            </Paragraph>
                          </Card>
                        )}
                        <Table
                          columns={stockColumns}
                          dataSource={topPickData}
                          loading={isLoading}
                          rowKey="id"
                          pagination={false}
                          expandable={{
                            expandedRowRender: (record) => (
                              <Paragraph style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                                {record.reasoning}
                              </Paragraph>
                            ),
                          }}
                        />
                      </div>
                    ),
                  },
                ]
              : []),
            {
              key: 'sector',
              label: `板块预测 (${sectorData.length})`,
              children: (
                <Table
                  columns={sectorColumns}
                  dataSource={sectorData}
                  loading={isLoading}
                  rowKey="id"
                  pagination={false}
                  expandable={{
                    expandedRowRender: (record) => (
                      <Paragraph style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                        {record.reasoning}
                      </Paragraph>
                    ),
                  }}
                />
              ),
            },
            {
              key: 'stock',
              label: `个股分析 (${stockData.length})`,
              children: (
                <Table
                  columns={stockColumns}
                  dataSource={stockData}
                  loading={isLoading}
                  rowKey="id"
                  pagination={false}
                  expandable={{
                    expandedRowRender: (record) => (
                      <Paragraph style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                        {record.reasoning}
                      </Paragraph>
                    ),
                  }}
                />
              ),
            },
            ...(verifiedData.length > 0
              ? [
                  {
                    key: 'verified',
                    label: `验证结果 (${verifiedData.length})`,
                    children: (
                      <Table
                        columns={verifiedColumns}
                        dataSource={verifiedData}
                        loading={isLoading}
                        rowKey="id"
                        pagination={false}
                        expandable={{
                          expandedRowRender: (record) => {
                            try {
                              const result = JSON.parse(record.actual_result);
                              return (
                                <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
                                  <div><strong>验证日期：</strong>{result.verify_date}</div>
                                  <div><strong>实际最高：</strong>{result.actual_high?.toFixed(2)}</div>
                                  <div><strong>实际最低：</strong>{result.actual_low?.toFixed(2)}</div>
                                  <div>
                                    <strong>目标命中：</strong>
                                    {result.target_hit ? <Tag color="success">YES</Tag> : <Tag color="default">NO</Tag>}
                                  </div>
                                  <div>
                                    <strong>止损触发：</strong>
                                    {result.stop_hit ? <Tag color="error">YES</Tag> : <Tag color="default">NO</Tag>}
                                  </div>
                                  {result.holding_days != null && (
                                    <div><strong>持仓天数：</strong>{result.holding_days}天</div>
                                  )}
                                  {result.cost_adjusted_return != null && (
                                    <div>
                                      <strong>扣费收益：</strong>
                                      <span style={{ color: result.cost_adjusted_return >= 0 ? '#3f8600' : '#cf1322', fontWeight: 'bold' }}>
                                        {result.cost_adjusted_return >= 0 ? '+' : ''}{result.cost_adjusted_return.toFixed(2)}%
                                      </span>
                                    </div>
                                  )}
                                  {result.daily_returns && result.daily_returns.length > 0 && (
                                    <div>
                                      <strong>每日收益：</strong>
                                      {result.daily_returns.map((r: number, i: number) => (
                                        <Tag key={i} color={r >= 0 ? 'red' : 'green'} style={{ marginBottom: 2 }}>
                                          D{i + 1}: {r >= 0 ? '+' : ''}{r.toFixed(2)}%
                                        </Tag>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              );
                            } catch {
                              return <Paragraph>{record.reasoning}</Paragraph>;
                            }
                          },
                        }}
                      />
                    ),
                  },
                ]
              : []),
          ]}
        />
      )}
    </div>
  );
}
