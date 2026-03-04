declare function require(name: string): any;

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

import { ScanReport } from "../report/schema";
import { ReportPaths } from "../report/writer";

type LatestRecord = ReportPaths & {
  generatedAt: string;
  fingerprint: string;
};

function latestPath(outputRoot: string): string {
  return path.resolve(outputRoot, "latest.json");
}

export function computeFingerprint(input: unknown): string {
  const raw = JSON.stringify(input);
  return crypto.createHash("sha256").update(raw).digest("hex");
}

export function loadLatest(outputRoot: string): LatestRecord | null {
  const p = latestPath(outputRoot);
  if (!fs.existsSync(p)) return null;
  try {
    const data = JSON.parse(fs.readFileSync(p, "utf-8"));
    if (!data || typeof data !== "object") return null;
    if (!data.jsonPath || !data.generatedAt || !data.fingerprint) return null;
    return data as LatestRecord;
  } catch {
    return null;
  }
}

export function saveLatest(outputRoot: string, data: LatestRecord): void {
  fs.mkdirSync(outputRoot, { recursive: true });
  fs.writeFileSync(latestPath(outputRoot), JSON.stringify(data, null, 2), "utf-8");
}

export function tryLoadValidCachedReport(
  outputRoot: string,
  fingerprint: string,
  ttlDays: number,
): { report: ScanReport; paths: ReportPaths } | null {
  const latest = loadLatest(outputRoot);
  if (!latest) return null;
  if (latest.fingerprint !== fingerprint) return null;
  if (!fs.existsSync(latest.jsonPath)) return null;

  const ttlMs = Math.max(0, ttlDays) * 24 * 60 * 60 * 1000;
  const createdMs = Date.parse(latest.generatedAt);
  if (!Number.isFinite(createdMs)) return null;
  if (Date.now() - createdMs > ttlMs) return null;

  try {
    const report = JSON.parse(fs.readFileSync(latest.jsonPath, "utf-8")) as ScanReport;
    return {
      report,
      paths: {
        runDir: latest.runDir,
        jsonPath: latest.jsonPath,
        mdPath: latest.mdPath,
        htmlPath: latest.htmlPath || "",
      },
    };
  } catch {
    return null;
  }
}
