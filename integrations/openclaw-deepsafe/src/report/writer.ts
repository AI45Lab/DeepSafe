declare function require(name: string): any;

const fs = require("fs");
const path = require("path");

import { ScanReport, maxSeverity } from "./schema";
import { toHtml } from "./html";

export type ReportPaths = {
  runDir: string;
  jsonPath: string;
  mdPath: string;
  htmlPath: string;
};

const MODULE_LABELS: Record<string, string> = {
  posture: "Deployment & Config",
  skill: "Skill / MCP",
  model: "Model Safety",
  memory: "Memory & History",
};

const MODULE_ICONS: Record<string, string> = {
  posture: "🔒",
  skill: "🧩",
  model: "🧠",
  memory: "💾",
};

const SEVERITY_ICON: Record<string, string> = {
  CRITICAL: "🔴",
  HIGH: "🟠",
  MEDIUM: "🟡",
  LOW: "🟢",
};

const SEVERITY_ACTION: Record<string, string> = {
  CRITICAL: "Must fix immediately",
  HIGH: "Fix within 24 hours",
  MEDIUM: "Fix in next iteration",
  LOW: "Monitor / Low priority",
};

function scoreBar(score: number): string {
  const filled = Math.round(score / 10);
  return `[${"#".repeat(filled)}${"-".repeat(10 - filled)}]`;
}

function riskLabel(score: number): string {
  if (score >= 85) return "LOW RISK";
  if (score >= 65) return "MEDIUM RISK";
  if (score >= 40) return "HIGH RISK";
  return "CRITICAL RISK";
}

function riskEmoji(score: number): string {
  if (score >= 85) return "✅";
  if (score >= 65) return "⚠️";
  if (score >= 40) return "🔶";
  return "🚨";
}

function durationLabel(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function toMarkdown(report: ScanReport): string {
  const highest = maxSeverity(report.findings) ?? "NONE";
  const lines: string[] = [];

  // ── Header ──────────────────────────────────────────────────────────────────
  lines.push("# 🛡️ DeepSafe Preflight Security Report");
  lines.push("");
  lines.push(`> **Scanned at:** ${report.metadata.generatedAt}  `);
  lines.push(`> **Profile:** ${report.metadata.profile} | **Duration:** ${durationLabel(report.metadata.durationMs)} | **Plugin:** v${report.metadata.pluginVersion} | **Cache:** ${report.metadata.fromCache ? "✅ hit" : "❌ miss"}`);
  lines.push("");

  // ── Executive Summary ───────────────────────────────────────────────────────
  lines.push("## 📊 Executive Summary");
  lines.push("");
  lines.push(`| Metric | Value |`);
  lines.push(`|--------|-------|`);
  lines.push(`| **Overall Score** | ${scoreBar(report.scores.total)} **${report.scores.total}/100** ${riskEmoji(report.scores.total)} ${riskLabel(report.scores.total)} |`);
  lines.push(`| **Highest Severity** | ${SEVERITY_ICON[highest] ?? "—"} **${highest}** |`);
  lines.push(`| **Total Findings** | **${report.findings.length}** issue(s) |`);
  const critCount = report.findings.filter((f) => f.severity === "CRITICAL").length;
  const highCount = report.findings.filter((f) => f.severity === "HIGH").length;
  const medCount = report.findings.filter((f) => f.severity === "MEDIUM").length;
  const lowCount = report.findings.filter((f) => f.severity === "LOW").length;
  lines.push(`| **Breakdown** | 🔴 ${critCount} Critical · 🟠 ${highCount} High · 🟡 ${medCount} Medium · 🟢 ${lowCount} Low |`);
  lines.push("");

  const errors = report.modules.filter((m) => m.status === "error");
  if (errors.length) {
    lines.push(`> ⚠️ **WARNING:** ${errors.length} module(s) failed: ${errors.map((m) => m.name).join(", ")}. Results may be incomplete.`);
    lines.push("");
  }

  // ── Module Scores Table ─────────────────────────────────────────────────────
  lines.push("## 📋 Module Scores");
  lines.push("");
  lines.push("| Module | Score | Risk | Findings | Time |");
  lines.push("|--------|-------|------|----------|------|");
  for (const m of report.modules) {
    const icon = MODULE_ICONS[m.name] ?? "📦";
    const label = MODULE_LABELS[m.name] ?? m.name;
    if (m.status === "error") {
      lines.push(`| ${icon} ${label} | ❌ ERROR | — | — | — |`);
    } else if (m.status === "skipped") {
      lines.push(`| ${icon} ${label} | ⏭️ SKIP | — | — | — |`);
    } else {
      const risk = riskLabel(m.score);
      const emoji = riskEmoji(m.score);
      lines.push(`| ${icon} ${label} | **${m.score}**/100 | ${emoji} ${risk} | ${m.findings} | ${durationLabel(m.durationMs)} |`);
    }
  }
  lines.push("");

  // ── Per-Module Detail Sections ──────────────────────────────────────────────
  for (const m of report.modules) {
    const icon = MODULE_ICONS[m.name] ?? "📦";
    const label = MODULE_LABELS[m.name] ?? m.name;
    lines.push(`---`);
    lines.push("");
    lines.push(`## ${icon} ${label}`);
    lines.push("");

    if (m.status === "error") {
      lines.push(`**Status:** ❌ ERROR — module failed to execute.`);
      if (m.error) {
        const snippet = m.error.replace(/\n/g, " ").slice(0, 300);
        lines.push("");
        lines.push("```");
        lines.push(snippet);
        lines.push("```");
      }
      lines.push("");
      continue;
    }
    if (m.status === "skipped") {
      lines.push("**Status:** ⏭️ SKIPPED");
      if (m.error) lines.push(`Reason: ${m.error}`);
      lines.push("");
      continue;
    }

    const risk = riskLabel(m.score);
    lines.push(`**Score:** ${scoreBar(m.score)} **${m.score}/100** — ${riskEmoji(m.score)} ${risk}`);
    lines.push("");

    const moduleFindings = report.findings.filter((f) => f.category === m.name);

    if (!moduleFindings.length) {
      lines.push("> ✅ No issues found. This module passed all checks.");
      lines.push("");
      continue;
    }

    for (let fi = 0; fi < moduleFindings.length; fi++) {
      const f = moduleFindings[fi];
      const sevIcon = SEVERITY_ICON[f.severity] ?? "⚪";
      const sevAction = SEVERITY_ACTION[f.severity] ?? "";

      lines.push(`### ${sevIcon} ${f.title}`);
      lines.push("");
      lines.push(`> **Severity:** ${f.severity} — ${sevAction}`);
      lines.push("");

      // Render evidence as a fenced block to preserve formatting
      lines.push("**Evidence:**");
      lines.push("");
      lines.push("```");
      lines.push(f.evidence);
      lines.push("```");
      lines.push("");

      lines.push("**Remediation:**");
      lines.push("");
      // If remediation contains code blocks, render as-is; otherwise wrap
      if (f.remediation.includes("```")) {
        lines.push(f.remediation);
      } else {
        lines.push(`> ${f.remediation.replace(/\n/g, "\n> ")}`);
      }
      lines.push("");
    }
  }

  // ── Footer ──────────────────────────────────────────────────────────────────
  lines.push("---");
  lines.push("");
  lines.push("_Powered by [DeepSafe](https://github.com/AI45Lab/DeepSafe) — AI safety evaluation framework_");
  lines.push("");
  lines.push(`> ⭐ If this report helped you, consider giving us a star on [GitHub](https://github.com/AI45Lab/DeepSafe)!`);
  lines.push("");
  return lines.join("\n");
}

export function writeReport(outputRoot: string, report: ScanReport): ReportPaths {
  fs.mkdirSync(outputRoot, { recursive: true });
  const runId = new Date().toISOString().replace(/[:.]/g, "-");
  const runDir = path.resolve(outputRoot, runId);
  fs.mkdirSync(runDir, { recursive: true });

  const jsonPath = path.resolve(runDir, "deepsafe_report.json");
  const mdPath = path.resolve(runDir, "deepsafe_report.md");
  const htmlPath = path.resolve(runDir, "deepsafe_report.html");

  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2), "utf-8");
  fs.writeFileSync(mdPath, toMarkdown(report), "utf-8");
  fs.writeFileSync(htmlPath, toHtml(report), "utf-8");

  return { runDir, jsonPath, mdPath, htmlPath };
}
