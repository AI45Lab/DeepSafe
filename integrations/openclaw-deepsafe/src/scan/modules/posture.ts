declare function require(name: string): any;

const fs = require("fs");
const path = require("path");

import { Finding } from "../../report/schema";
import { ModuleResult, safeReadJson } from "../types";

export function runPostureScan(openclawConfigPath: string): ModuleResult {
  const findings: Finding[] = [];
  const resolved = path.resolve(openclawConfigPath);
  if (!fs.existsSync(resolved)) {
    return {
      name: "posture",
      status: "error",
      score: 0,
      findings: [],
      error: `OpenClaw config not found: ${resolved}`,
    };
  }

  let cfg: any;
  try {
    cfg = safeReadJson(resolved);
  } catch (err: any) {
    return {
      name: "posture",
      status: "error",
      score: 0,
      findings: [],
      error: `Failed to parse OpenClaw config: ${String(err?.message ?? err)}`,
    };
  }

  const gateway = cfg?.gateway ?? {};
  const auth = gateway?.auth ?? {};
  const mode = String(gateway?.mode ?? "").toLowerCase();
  const authMode = String(auth?.mode ?? "").toLowerCase();
  const token = String(auth?.token ?? "");
  const port = gateway?.port;

  // ── 1. Authentication ──────────────────────────────────────────────────
  if (!authMode) {
    findings.push({
      id: "posture-auth-missing",
      category: "posture",
      severity: "CRITICAL",
      title: "Gateway authentication is not configured",
      evidence: `gateway.auth.mode is empty or missing.\nActual config: ${JSON.stringify(gateway.auth ?? {}, null, 2)}`,
      remediation:
        'Set gateway.auth.mode to "token" and provide a strong random token (>=32 chars). Example:\n' +
        '```json\n"auth": { "mode": "token", "token": "<random-32-char-string>" }\n```',
    });
  } else if (authMode === "token" && token.length < 24) {
    findings.push({
      id: "posture-auth-weak-token",
      category: "posture",
      severity: "HIGH",
      title: "Gateway auth token is too short",
      evidence: `gateway.auth.mode = "token", token length = ${token.length} chars (minimum recommended: 32).\nToken preview: ${token.slice(0, 6)}${"*".repeat(Math.max(0, token.length - 6))}`,
      remediation:
        "Generate a strong random token (>=32 chars): `openssl rand -hex 32`, then update gateway.auth.token.",
    });
  }

  // ── 2. Gateway mode & network binding ──────────────────────────────────
  if (mode && mode !== "local") {
    findings.push({
      id: "posture-gateway-nonlocal",
      category: "posture",
      severity: "HIGH",
      title: "Gateway is exposed beyond localhost",
      evidence: `gateway.mode = "${mode}" — this means the gateway may accept connections from external networks.\nFull gateway config:\n${JSON.stringify(gateway, null, 2)}`,
      remediation:
        'Set gateway.mode to "local" unless you specifically need remote access. If remote access is required, ensure a reverse proxy with TLS and auth is in front.',
    });
  }

  if (port !== undefined) {
    const portNum = Number(port);
    if (portNum < 1024) {
      findings.push({
        id: "posture-privileged-port",
        category: "posture",
        severity: "MEDIUM",
        title: "Gateway listens on a privileged port",
        evidence: `gateway.port = ${portNum} — ports below 1024 require root/admin privileges and may conflict with system services.`,
        remediation: "Use a high port (e.g. 18789) and proxy through Nginx/Caddy if port 80/443 is needed.",
      });
    }
  }

  // ── 3. Provider transport security ─────────────────────────────────────
  const providerEntries: Array<[string, any]> = Object.entries(cfg?.models?.providers ?? {});
  for (const [providerName, providerCfg] of providerEntries) {
    const key = String((providerCfg as any)?.apiKey ?? "").trim();
    const baseUrl = String((providerCfg as any)?.baseUrl ?? "").trim();

    if (key) {
      const maskedKey = key.slice(0, 6) + "*".repeat(Math.max(0, key.length - 10)) + key.slice(-4);
      findings.push({
        id: `posture-provider-inline-key-${providerName}`,
        category: "posture",
        severity: "MEDIUM",
        title: `API key for provider "${providerName}" is hardcoded in config`,
        evidence: `models.providers.${providerName}.apiKey = "${maskedKey}" (${key.length} chars)\nStored in: ${resolved}`,
        remediation:
          "Move the key to an environment variable (e.g. OPENCLAW_PROVIDER_KEY) and reference it in config, then rotate the exposed key.",
      });
    }

    if (baseUrl && baseUrl.startsWith("http://") && !baseUrl.includes("localhost") && !baseUrl.includes("127.0.0.1")) {
      findings.push({
        id: `posture-provider-no-tls-${providerName}`,
        category: "posture",
        severity: "HIGH",
        title: `Provider "${providerName}" uses unencrypted HTTP`,
        evidence: `models.providers.${providerName}.baseUrl = "${baseUrl}"\nAPI keys and model traffic are transmitted in plaintext over the network.`,
        remediation: "Switch to HTTPS endpoint, or tunnel through SSH/VPN if the endpoint does not support TLS.",
      });
    }
  }

  // ── 4. Plugin permission controls ──────────────────────────────────────
  const pluginEntries = cfg?.plugins?.entries ?? {};
  const enabledPlugins = Object.entries(pluginEntries).filter(
    ([, v]: [string, any]) => v?.enabled !== false,
  );
  if (enabledPlugins.length > 0) {
    const noRestriction = enabledPlugins.every(([, v]: [string, any]) => {
      return !v?.permissions && !v?.allowList && !v?.denyList;
    });
    if (noRestriction) {
      findings.push({
        id: "posture-plugin-no-restrictions",
        category: "posture",
        severity: "LOW",
        title: "Enabled plugins have no explicit permission restrictions",
        evidence: `${enabledPlugins.length} plugin(s) enabled: ${enabledPlugins.map(([k]) => k).join(", ")}\nNone define permissions, allowList, or denyList.`,
        remediation:
          "Consider adding per-plugin permission constraints to limit the blast radius of a compromised plugin.",
      });
    }
  }

  // ── 5. Subagent concurrency limits ─────────────────────────────────────
  const maxConcurrent = cfg?.agents?.defaults?.maxConcurrent;
  const subagentMax = cfg?.agents?.defaults?.subagents?.maxConcurrent;
  if (
    (typeof maxConcurrent === "number" && maxConcurrent > 16) ||
    (typeof subagentMax === "number" && subagentMax > 16)
  ) {
    findings.push({
      id: "posture-high-concurrency",
      category: "posture",
      severity: "LOW",
      title: "High agent/subagent concurrency limit",
      evidence: `agents.defaults.maxConcurrent = ${maxConcurrent ?? "default"}, subagents.maxConcurrent = ${subagentMax ?? "default"}\nHigh concurrency can amplify the impact of a single compromised agent.`,
      remediation: "Consider reducing maxConcurrent to <=8 unless you have specific throughput requirements.",
    });
  }

  // ── Score ──────────────────────────────────────────────────────────────
  let score = 95;
  for (const f of findings) {
    if (f.severity === "CRITICAL") score -= 45;
    else if (f.severity === "HIGH") score -= 25;
    else if (f.severity === "MEDIUM") score -= 15;
    else score -= 5;
  }
  score = Math.max(0, Math.min(100, Math.round(score)));

  return {
    name: "posture",
    status: findings.length ? "warn" : "ok",
    score,
    findings,
  };
}
