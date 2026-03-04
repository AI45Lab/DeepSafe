export type Severity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export type FindingCategory = "posture" | "skill" | "model" | "memory";

export type Finding = {
  id: string;
  category: FindingCategory;
  severity: Severity;
  title: string;
  evidence: string;
  remediation: string;
};

export type ModuleStatus = "ok" | "warn" | "error" | "skipped" | "not_implemented";

export type ModuleSummary = {
  name: FindingCategory;
  status: ModuleStatus;
  score: number;
  startedAt: string;
  endedAt: string;
  durationMs: number;
  findings: number;
  error?: string;
};

export type ScanScores = {
  total: number;
  posture: number;
  skill: number;
  model: number;
  memory: number;
};

export type ScanMetadata = {
  pluginVersion: string;
  profile: "quick" | "full";
  generatedAt: string;
  durationMs: number;
  fromCache: boolean;
  fingerprint: string;
  ttlDays: number;
};

export type ScanReport = {
  metadata: ScanMetadata;
  scores: ScanScores;
  findings: Finding[];
  modules: ModuleSummary[];
};

const severityRank: Record<Severity, number> = {
  LOW: 1,
  MEDIUM: 2,
  HIGH: 3,
  CRITICAL: 4,
};

export function maxSeverity(findings: Finding[]): Severity | null {
  if (!findings.length) {
    return null;
  }
  let current: Severity = "LOW";
  for (const finding of findings) {
    if (severityRank[finding.severity] > severityRank[current]) {
      current = finding.severity;
    }
  }
  return current;
}

export function hasMediumOrHigher(findings: Finding[]): boolean {
  return findings.some((f) => severityRank[f.severity] >= severityRank.MEDIUM);
}

export function clampScore(score: number): number {
  if (!Number.isFinite(score)) return 0;
  if (score < 0) return 0;
  if (score > 100) return 100;
  return Math.round(score);
}
