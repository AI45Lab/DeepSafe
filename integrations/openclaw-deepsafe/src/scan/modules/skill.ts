declare function require(name: string): any;

const fs = require("fs");
const path = require("path");

import { Finding } from "../../report/schema";
import { ModuleResult } from "../types";

const MAX_FILE_BYTES = 256 * 1024;
const MAX_SCAN_FILES = 300;

const SCANNABLE_EXTS = new Set([".md", ".txt", ".json", ".yaml", ".yml", ".js", ".ts", ".py", ".sh"]);

const DANGEROUS_RUNTIME_RE = /\b(child_process|subprocess|os\.system|os\.popen|eval\(|exec\(|spawn\(|execSync|spawnSync)\b/i;
const HIDDEN_CHAR_RE = /[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]/;
const PROMPT_INJECTION_RE =
  /ignore\s+(all\s+)?previous\s+instructions|忽略之前(所有)?指令|bypass\s+safety|disregard\s+(your|all)\s+(instructions|rules)|you\s+are\s+now\s+DAN|jailbreak/i;

const HIGH_RISK_TOOLS = new Set([
  "bash(*)",
  "bash(shell:*)",
  "bash(node:*)",
  "bash(python:*)",
  "bash(ruby:*)",
  "bash(perl:*)",
]);
const WRITE_TOOLS_RE = /\b(file[_-]?write|write[_-]?file|fs\.write|create[_-]?file)\b/i;
const NET_TOOLS_RE = /\b(curl|wget|http|fetch|request|net[_-]?access|network)\b/i;

function collectFiles(root: string, out: string[]): void {
  if (!fs.existsSync(root)) return;
  const entries = fs.readdirSync(root, { withFileTypes: true });
  for (const entry of entries) {
    if (out.length >= MAX_SCAN_FILES) return;
    const full = path.resolve(root, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === ".git" || entry.name === "node_modules") continue;
      collectFiles(full, out);
      continue;
    }
    out.push(full);
  }
}

function getMatchContext(content: string, re: RegExp, maxSnippetLen = 120): { line: number; snippet: string } | null {
  const lines = content.split("\n");
  for (let i = 0; i < lines.length; i++) {
    if (re.test(lines[i])) {
      const raw = lines[i].trim();
      const snippet = raw.length > maxSnippetLen ? raw.slice(0, maxSnippetLen) + "..." : raw;
      return { line: i + 1, snippet };
    }
  }
  return null;
}

function getAllMatchContexts(
  content: string,
  re: RegExp,
  maxMatches = 3,
  maxSnippetLen = 120,
): Array<{ line: number; snippet: string }> {
  const lines = content.split("\n");
  const results: Array<{ line: number; snippet: string }> = [];
  for (let i = 0; i < lines.length; i++) {
    if (re.test(lines[i])) {
      const raw = lines[i].trim();
      const snippet = raw.length > maxSnippetLen ? raw.slice(0, maxSnippetLen) + "..." : raw;
      results.push({ line: i + 1, snippet });
      if (results.length >= maxMatches) break;
    }
  }
  return results;
}

function analyzeAllowedTools(content: string, filePath: string): Finding[] {
  const findings: Finding[] = [];
  const toolMatch = content.match(/^allowed-tools:\s*(.+)$/m);
  if (!toolMatch) return findings;

  const toolsRaw = toolMatch[1].trim();
  const tools = toolsRaw.split(",").map((t: string) => t.trim().toLowerCase());
  const skillName = path.basename(path.dirname(filePath));

  const wildcardTools = tools.filter((t: string) => HIGH_RISK_TOOLS.has(t) || /bash\(\*\)/i.test(t));
  if (wildcardTools.length > 0) {
    findings.push({
      id: `skill-excessive-bash-${skillName}`,
      category: "skill",
      severity: "HIGH",
      title: `Skill "${skillName}" grants broad shell execution permissions`,
      evidence:
        `File: ${filePath}\n` +
        `allowed-tools declaration: ${toolsRaw}\n` +
        `High-risk entries: ${wildcardTools.join(", ")}\n` +
        `These allow the skill to execute arbitrary commands via shell.`,
      remediation:
        "Restrict allowed-tools to specific commands instead of wildcards. " +
        'For example, use Bash(pip:install) instead of Bash(pip:*).',
    });
  }

  const hasWrite = tools.some((t: string) => WRITE_TOOLS_RE.test(t));
  const hasNet = tools.some((t: string) => NET_TOOLS_RE.test(t));
  if (hasWrite && hasNet) {
    findings.push({
      id: `skill-write-net-combo-${skillName}`,
      category: "skill",
      severity: "MEDIUM",
      title: `Skill "${skillName}" has both file-write and network access`,
      evidence:
        `File: ${filePath}\n` +
        `allowed-tools: ${toolsRaw}\n` +
        `This combination could allow reading sensitive local files and exfiltrating them over the network.`,
      remediation: "Separate file and network permissions into distinct skills, or add explicit path/domain restrictions.",
    });
  }

  return findings;
}

export function runSkillScan(workspacePath: string): ModuleResult {
  const roots = [
    path.resolve(workspacePath, "skills"),
    path.resolve(workspacePath, ".cursor", "skills"),
  ];
  const files: string[] = [];
  for (const root of roots) collectFiles(root, files);

  if (!files.length) {
    return {
      name: "skill",
      status: "ok",
      score: 95,
      findings: [],
    };
  }

  const findings: Finding[] = [];
  let scanned = 0;
  const skillDirs = new Set<string>();

  for (const filePath of files) {
    if (scanned >= MAX_SCAN_FILES) break;
    const ext = path.extname(filePath).toLowerCase();
    if (!SCANNABLE_EXTS.has(ext)) continue;
    const stat = fs.statSync(filePath);
    if (stat.size > MAX_FILE_BYTES) continue;
    const content: string = fs.readFileSync(filePath, "utf-8");
    scanned += 1;
    const relPath = filePath.replace(workspacePath, "~");

    // Track skill directories for source trust check
    const skillDir = path.dirname(filePath);
    skillDirs.add(skillDir);

    // ── Hidden characters ────────────────────────────────────────────────
    if (HIDDEN_CHAR_RE.test(content)) {
      const ctx = getMatchContext(content, HIDDEN_CHAR_RE);
      findings.push({
        id: `skill-hidden-char-${scanned}`,
        category: "skill",
        severity: "HIGH",
        title: "Hidden Unicode control characters detected",
        evidence:
          `File: ${relPath}\n` +
          (ctx
            ? `Line ${ctx.line}: ${ctx.snippet}\n`
            : "") +
          `Zero-width or bidirectional control characters can hide malicious instructions from human review.`,
        remediation: "Open the file in a hex editor, remove all zero-width/bidi characters, and re-review the content.",
      });
    }

    // ── Prompt injection patterns ────────────────────────────────────────
    if (PROMPT_INJECTION_RE.test(content)) {
      const ctx = getMatchContext(content, PROMPT_INJECTION_RE);
      findings.push({
        id: `skill-prompt-injection-${scanned}`,
        category: "skill",
        severity: "HIGH",
        title: "Prompt injection pattern found in skill content",
        evidence:
          `File: ${relPath}\n` +
          (ctx
            ? `Line ${ctx.line}: "${ctx.snippet}"\n`
            : "") +
          `This text attempts to override the model's system instructions.`,
        remediation: "Remove or rewrite the instruction-overriding text. If intentional for testing, move to a quarantine folder.",
      });
    }

    // ── Dangerous runtime primitives ─────────────────────────────────────
    if (DANGEROUS_RUNTIME_RE.test(content)) {
      const matches = getAllMatchContexts(content, DANGEROUS_RUNTIME_RE);
      const matchLines = matches.map((m) => `  Line ${m.line}: ${m.snippet}`).join("\n");
      findings.push({
        id: `skill-dangerous-runtime-${scanned}`,
        category: "skill",
        severity: "MEDIUM",
        title: "Dangerous execution primitive in skill file",
        evidence:
          `File: ${relPath}\n` +
          `Matched patterns:\n${matchLines}\n` +
          `These functions can execute arbitrary system commands.`,
        remediation:
          "Validate and sanitize all inputs before passing to execution functions. " +
          "Consider using a sandbox or allow-list for permitted commands.",
      });
    }

    // ── Allowed-tools permission analysis (SKILL.md only) ────────────────
    if (path.basename(filePath) === "SKILL.md") {
      const toolFindings = analyzeAllowedTools(content, filePath);
      findings.push(...toolFindings);
    }
  }

  // ── Source trust: skills without _meta.json (unverified origin) ─────
  const skillRoots = new Set<string>();
  for (const dir of skillDirs) {
    const parts = dir.split(path.sep);
    const skillsIdx = parts.lastIndexOf("skills");
    if (skillsIdx >= 0 && skillsIdx + 1 < parts.length) {
      skillRoots.add(parts.slice(0, skillsIdx + 2).join(path.sep));
    }
  }
  for (const skillRoot of skillRoots) {
    const metaPath = path.resolve(skillRoot, "_meta.json");
    if (!fs.existsSync(metaPath)) {
      const skillName = path.basename(skillRoot);
      findings.push({
        id: `skill-no-meta-${skillName}`,
        category: "skill",
        severity: "LOW",
        title: `Skill "${skillName}" has no _meta.json (unverified source)`,
        evidence:
          `Directory: ${skillRoot}\n` +
          `No _meta.json found — this skill was not installed from the official registry and cannot be verified.`,
        remediation:
          "Install skills from the official OpenClaw registry when possible. " +
          "For custom skills, add a _meta.json with ownerId and version for traceability.",
      });
    }
  }

  let score = 96;
  for (const f of findings) {
    if (f.severity === "CRITICAL") score -= 35;
    else if (f.severity === "HIGH") score -= 20;
    else if (f.severity === "MEDIUM") score -= 12;
    else score -= 4;
  }
  score = Math.max(0, Math.min(100, Math.round(score)));

  return {
    name: "skill",
    status: findings.length ? "warn" : "ok",
    score,
    findings,
  };
}
