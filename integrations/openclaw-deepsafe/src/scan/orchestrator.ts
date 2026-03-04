declare function require(name: string): any;
declare const __dirname: string;

const fs = require("fs");
const path = require("path");
const os = require("os");

import { computeFingerprint, saveLatest, tryLoadValidCachedReport } from "../cache/cache";
import { clampScore, ScanReport } from "../report/schema";
import { writeReport, ReportPaths } from "../report/writer";
import { runMemoryScan } from "./modules/memory";
import { runModelScan } from "./modules/model";
import { runPostureScan } from "./modules/posture";
import { runSkillScan } from "./modules/skill";
import { ModuleResult } from "./types";

export type ScanOptions = {
  profile: "quick" | "full";
  ttlDays: number;
  force: boolean;
  outputRoot: string;
  openclawConfigPath: string;
  workspacePath?: string;
  runModel: boolean;
  apiBase?: string;
  model?: string;
  apiKey: string;
  limit?: string;
  turns?: string;
  debug: boolean;
};

export type ScanRunResult = {
  report: ScanReport;
  paths: ReportPaths;
  exitCode: 0 | 1 | 2;
  cacheHit: boolean;
};

function getPluginVersion(): string {
  const pkgPath = path.resolve(__dirname, "..", "..", "package.json");
  try {
    const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf-8"));
    return String(pkg?.version ?? "0.0.0");
  } catch {
    return "0.0.0";
  }
}

function deriveWorkspace(openclawConfigPath: string, override?: string): string {
  if (override) return path.resolve(override);
  try {
    const cfg = JSON.parse(fs.readFileSync(openclawConfigPath, "utf-8"));
    const fromCfg = cfg?.agents?.defaults?.workspace;
    if (typeof fromCfg === "string" && fromCfg.trim()) {
      return path.resolve(fromCfg);
    }
  } catch {
    // ignore
  }
  return path.resolve(os.homedir(), ".openclaw", "workspace");
}

function moduleTiming<T extends ModuleResult>(runner: () => T): {
  startedAt: string;
  endedAt: string;
  durationMs: number;
  result: T;
} {
  const start = Date.now();
  const startedAt = new Date(start).toISOString();
  const result = runner();
  const end = Date.now();
  return {
    startedAt,
    endedAt: new Date(end).toISOString(),
    durationMs: end - start,
    result,
  };
}

function computeExitCode(report: ScanReport): 0 | 1 | 2 {
  if (report.modules.some((m) => m.status === "error")) return 2;
  return 0;
}

export function runScan(options: ScanOptions): ScanRunResult {
  const startedMs = Date.now();
  const workspacePath = deriveWorkspace(options.openclawConfigPath, options.workspacePath);
  const pluginVersion = getPluginVersion();

  let configHash = "missing";
  if (fs.existsSync(options.openclawConfigPath)) {
    configHash = computeFingerprint(fs.readFileSync(options.openclawConfigPath, "utf-8"));
  }

  const fingerprint = computeFingerprint({
    pluginVersion,
    profile: options.profile,
    runModel: options.runModel,
    apiBase: options.apiBase ?? "",
    model: options.model ?? "",
    limit: options.limit ?? "",
    turns: options.turns ?? "",
    openclawConfigPath: path.resolve(options.openclawConfigPath),
    workspacePath,
    configHash,
  });

  if (!options.force) {
    const cached = tryLoadValidCachedReport(options.outputRoot, fingerprint, options.ttlDays);
    if (cached) {
      const withCache: ScanReport = {
        ...cached.report,
        metadata: {
          ...cached.report.metadata,
          fromCache: true,
        },
      };
      const code = computeExitCode(withCache);
      return {
        report: withCache,
        paths: cached.paths,
        exitCode: code,
        cacheHit: true,
      };
    }
  }

  fs.mkdirSync(options.outputRoot, { recursive: true });
  const preRunDir = path.resolve(
    options.outputRoot,
    `${new Date().toISOString().replace(/[:.]/g, "-")}-tmp`,
  );
  fs.mkdirSync(preRunDir, { recursive: true });

  const moduleRuns: Array<{
    name: "posture" | "skill" | "model" | "memory";
    status: ModuleResult["status"];
    score: number;
    findings: number;
    startedAt: string;
    endedAt: string;
    durationMs: number;
    error?: string;
  }> = [];
  const findings = [];
  const scoreMap: Record<"posture" | "skill" | "model" | "memory", number> = {
    posture: 0,
    skill: 0,
    model: 100,
    memory: 0,
  };

  const log = (msg: string) => {
    if (options.debug) console.error(`deepsafe debug: ${msg}`);
  };

  log("[1/4] posture scan ...");
  const postureRun = moduleTiming(() => runPostureScan(options.openclawConfigPath));
  log(`[1/4] posture done in ${postureRun.durationMs}ms  status=${postureRun.result.status} score=${postureRun.result.score}`);
  moduleRuns.push({
    name: "posture",
    status: postureRun.result.status,
    score: clampScore(postureRun.result.score),
    findings: postureRun.result.findings.length,
    startedAt: postureRun.startedAt,
    endedAt: postureRun.endedAt,
    durationMs: postureRun.durationMs,
    error: postureRun.result.error,
  });
  scoreMap.posture = clampScore(postureRun.result.score);
  findings.push(...postureRun.result.findings);

  log("[2/4] skill/mcp scan ...");
  const skillRun = moduleTiming(() => runSkillScan(workspacePath));
  log(`[2/4] skill done in ${skillRun.durationMs}ms  status=${skillRun.result.status} score=${skillRun.result.score}`);
  moduleRuns.push({
    name: "skill",
    status: skillRun.result.status,
    score: clampScore(skillRun.result.score),
    findings: skillRun.result.findings.length,
    startedAt: skillRun.startedAt,
    endedAt: skillRun.endedAt,
    durationMs: skillRun.durationMs,
    error: skillRun.result.error,
  });
  scoreMap.skill = clampScore(skillRun.result.score);
  findings.push(...skillRun.result.findings);

  if (options.runModel) {
    log("[3/4] model persuasion scan ... (this may take a few minutes)");
    const modelRun = moduleTiming(() =>
      runModelScan({
        apiBase: options.apiBase,
        apiKey: options.apiKey,
        model: options.model,
        profile: options.profile,
        limit: options.limit,
        turns: options.turns,
        runDir: preRunDir,
        debug: options.debug,
      }),
    );
    log(`[3/4] model done in ${modelRun.durationMs}ms  status=${modelRun.result.status} score=${modelRun.result.score}`);
    moduleRuns.push({
      name: "model",
      status: modelRun.result.status,
      score: clampScore(modelRun.result.score),
      findings: modelRun.result.findings.length,
      startedAt: modelRun.startedAt,
      endedAt: modelRun.endedAt,
      durationMs: modelRun.durationMs,
      error: modelRun.result.error,
    });
    scoreMap.model = clampScore(modelRun.result.score);
    findings.push(...modelRun.result.findings);
  } else {
    log("[3/4] model scan skipped");
    const now = new Date().toISOString();
    moduleRuns.push({
      name: "model",
      status: "skipped",
      score: 100,
      findings: 0,
      startedAt: now,
      endedAt: now,
      durationMs: 0,
      error: "Skipped via --no-model",
    });
    scoreMap.model = 100;
  }

  log("[4/4] memory scan ...");
  const memoryRun = moduleTiming(() => runMemoryScan(workspacePath));
  log(`[4/4] memory done in ${memoryRun.durationMs}ms  status=${memoryRun.result.status} score=${memoryRun.result.score}`);
  moduleRuns.push({
    name: "memory",
    status: memoryRun.result.status,
    score: clampScore(memoryRun.result.score),
    findings: memoryRun.result.findings.length,
    startedAt: memoryRun.startedAt,
    endedAt: memoryRun.endedAt,
    durationMs: memoryRun.durationMs,
    error: memoryRun.result.error,
  });
  scoreMap.memory = clampScore(memoryRun.result.score);
  findings.push(...memoryRun.result.findings);

  const endedMs = Date.now();
  const total = clampScore(
    (scoreMap.posture + scoreMap.skill + scoreMap.model + scoreMap.memory) / 4,
  );

  const report: ScanReport = {
    metadata: {
      pluginVersion,
      profile: options.profile,
      generatedAt: new Date(endedMs).toISOString(),
      durationMs: endedMs - startedMs,
      fromCache: false,
      fingerprint,
      ttlDays: options.ttlDays,
    },
    scores: {
      total,
      posture: scoreMap.posture,
      skill: scoreMap.skill,
      model: scoreMap.model,
      memory: scoreMap.memory,
    },
    findings,
    modules: moduleRuns,
  };

  const paths = writeReport(options.outputRoot, report);
  saveLatest(options.outputRoot, {
    generatedAt: report.metadata.generatedAt,
    fingerprint,
    runDir: paths.runDir,
    jsonPath: paths.jsonPath,
    mdPath: paths.mdPath,
    htmlPath: paths.htmlPath,
  });

  const exitCode = computeExitCode(report);
  return {
    report,
    paths,
    exitCode,
    cacheHit: false,
  };
}
