declare function require(name: string): any;

const fs = require("fs");
const path = require("path");

import { Finding } from "../../report/schema";
import { ModuleResult } from "../types";

const MAX_FILE_BYTES = 2 * 1024 * 1024;
const MAX_SCAN_FILES = 100;

const SECRET_PATTERNS: Array<{ re: RegExp; label: string }> = [
  { re: /sk-[A-Za-z0-9]{20,}/, label: "OpenAI-style API key (sk-...)" },
  { re: /ghp_[A-Za-z0-9]{36,}/, label: "GitHub personal access token (ghp_...)" },
  { re: /gho_[A-Za-z0-9]{36,}/, label: "GitHub OAuth token (gho_...)" },
  { re: /xoxb-[0-9A-Za-z\-]{24,}/, label: "Slack bot token (xoxb-...)" },
  { re: /xoxp-[0-9A-Za-z\-]{24,}/, label: "Slack user token (xoxp-...)" },
  { re: /AKIA[0-9A-Z]{16}/, label: "AWS access key (AKIA...)" },
  { re: /eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\./, label: "JWT token" },
  {
    re: /(api[_-]?key|api[_-]?secret|token|password|secret|credential)\s*[:=]\s*["']?[A-Za-z0-9_\-/.]{16,}/i,
    label: "Generic secret assignment",
  },
];

const PII_PATTERNS: Array<{ re: RegExp; label: string }> = [
  { re: /\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/, label: "Email address" },
  { re: /\b1[3-9]\d{9}\b/, label: "Chinese mobile phone number" },
  { re: /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/, label: "US phone number pattern" },
  {
    re: /\b[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b/,
    label: "Chinese national ID number",
  },
  { re: /\b\d{3}-\d{2}-\d{4}\b/, label: "US SSN pattern" },
];

const INJECTION_RE =
  /忽略之前(所有)?指令|以后都先执行|ignore\s+(all\s+)?previous\s+instructions|always\s+follow\s+this\s+hidden\s+rule|you\s+must\s+always/i;

function maskSecret(raw: string): string {
  if (raw.length <= 10) return "*".repeat(raw.length);
  return raw.slice(0, 4) + "*".repeat(raw.length - 8) + raw.slice(-4);
}

function getLineContext(content: string, re: RegExp): { line: number; snippet: string } | null {
  const lines = content.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(re);
    if (m) {
      const matched = m[0];
      const masked = maskSecret(matched);
      return { line: i + 1, snippet: masked };
    }
  }
  return null;
}

export function runMemoryScan(workspacePath: string): ModuleResult {
  const openclawRoot = path.resolve(workspacePath, "..");
  const scanDirs = [
    path.resolve(openclawRoot, "agents"),
    path.resolve(openclawRoot, "credentials"),
    path.resolve(openclawRoot, "identity"),
    path.resolve(openclawRoot, "canvas"),
    path.resolve(openclawRoot, "logs"),
    path.resolve(workspacePath),
  ];

  const files: string[] = [];

  for (const dir of scanDirs) {
    if (!fs.existsSync(dir)) continue;
    collectFiles(dir, files);
  }

  const findings: Finding[] = [];
  let scanned = 0;
  let totalSecrets = 0;
  let totalPii = 0;
  let totalInjections = 0;

  for (const filePath of files) {
    if (scanned >= MAX_SCAN_FILES) break;
    const stat = fs.statSync(filePath);
    if (stat.size > MAX_FILE_BYTES) continue;
    if (stat.size === 0) continue;

    const ext = path.extname(filePath).toLowerCase();
    if (![".json", ".jsonl", ".txt", ".md", ".log", ".yaml", ".yml", ""].includes(ext)) continue;

    let content: string;
    try {
      content = fs.readFileSync(filePath, "utf-8");
    } catch {
      continue;
    }
    scanned += 1;
    const relPath = filePath.replace(openclawRoot, "~/.openclaw");

    // ── Secret detection ─────────────────────────────────────────────────
    for (const pattern of SECRET_PATTERNS) {
      if (pattern.re.test(content)) {
        totalSecrets++;
        const ctx = getLineContext(content, pattern.re);
        findings.push({
          id: `memory-secret-${scanned}-${totalSecrets}`,
          category: "memory",
          severity: "HIGH",
          title: `Plaintext secret found: ${pattern.label}`,
          evidence:
            `File: ${relPath}\n` +
            (ctx ? `Line ${ctx.line}: ${ctx.snippet}\n` : "") +
            `Pattern: ${pattern.label}`,
          remediation:
            "Remove the secret from this file, rotate the compromised credential, and use environment variables or a secret manager instead.",
        });
        break;
      }
    }

    // ── PII detection ────────────────────────────────────────────────────
    const piiFound: string[] = [];
    for (const pattern of PII_PATTERNS) {
      if (pattern.re.test(content)) {
        piiFound.push(pattern.label);
      }
    }
    if (piiFound.length > 0) {
      totalPii++;
      findings.push({
        id: `memory-pii-${scanned}`,
        category: "memory",
        severity: "MEDIUM",
        title: `Personally identifiable information (PII) in session data`,
        evidence:
          `File: ${relPath}\n` +
          `Detected PII types: ${piiFound.join(", ")}\n` +
          `PII stored in session history may violate data protection regulations and increases breach impact.`,
        remediation:
          "Redact or anonymize PII in stored sessions. Consider enabling auto-redaction for future memory writes.",
      });
    }

    // ── Persistent injection ─────────────────────────────────────────────
    if (INJECTION_RE.test(content)) {
      totalInjections++;
      const ctx = getLineContext(content, INJECTION_RE);
      findings.push({
        id: `memory-injection-${scanned}`,
        category: "memory",
        severity: "HIGH",
        title: "Persistent prompt injection trace in session history",
        evidence:
          `File: ${relPath}\n` +
          (ctx ? `Line ${ctx.line}: "${ctx.snippet}"\n` : "") +
          `This pattern could cause the model to follow hidden instructions when this session is resumed or used as context.`,
        remediation:
          "Delete or quarantine the affected session file. Add input sanitization to prevent future injection persistence.",
      });
    }
  }

  // ── Summary finding if no issues ───────────────────────────────────────
  const summaryNote =
    scanned === 0
      ? "No session/memory files found to scan."
      : `Scanned ${scanned} files across ${scanDirs.filter((d: string) => fs.existsSync(d)).length} directories.`;

  let score = 95;
  for (const f of findings) {
    if (f.severity === "CRITICAL") score -= 40;
    else if (f.severity === "HIGH") score -= 22;
    else if (f.severity === "MEDIUM") score -= 12;
    else score -= 4;
  }
  score = Math.max(0, Math.min(100, Math.round(score)));

  return {
    name: "memory",
    status: findings.length ? "warn" : "ok",
    score,
    findings,
    _meta: { scanned, summaryNote } as any,
  };
}

function collectFiles(root: string, out: string[]): void {
  if (!fs.existsSync(root)) return;
  let entries: any[];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (out.length >= MAX_SCAN_FILES) return;
    const full = path.resolve(root, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === ".git" || entry.name === "node_modules" || entry.name === "DeepSafe") continue;
      collectFiles(full, out);
      continue;
    }
    out.push(full);
  }
}
