export interface BaseEvent {
  eventType: string;
  tenantId: string;
  projectId: string;
  timestamp: string;       // ISO 8601
  correlationId: string;   // ties a whole run's events together for tracing
}

export interface TestRequestedEvent extends BaseEvent {
  eventType: 'test.requested';
  runId: string;
  url: string;
  prompt: string;
  runSource: 'ui' | 'github-actions' | 'gitlab-ci' | 'jenkins';
}

export interface TestGeneratedEvent extends BaseEvent {
  eventType: 'test.generated';
  runId: string;
  scenarioCount: number;
  scenariosS3Path: string;   // generated Playwright script(s)
}

export interface TestExecutedEvent extends BaseEvent {
  eventType: 'test.executed';
  runId: string;
  resultsSummary: { pageUrl: string; loadTimeMs: number; consoleErrors: number }[];
}

export interface VisualDiffDetectedEvent extends BaseEvent {
  eventType: 'visual.diff.detected';
  runId: string;
  page: string;
  differencePercent: number;
}

export interface AnalysisCompletedEvent extends BaseEvent {
  eventType: 'analysis.completed';
  runId: string;
  rootCause: string;
  severity: 'critical' | 'warning' | 'info';
  recommendation: string;
}

export interface ReportReadyEvent extends BaseEvent {
  eventType: 'report.ready';
  runId: string;
  reportUrl: string;
  criticalCount: number;
  warningCount: number;
}