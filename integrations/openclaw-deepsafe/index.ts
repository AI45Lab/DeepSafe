declare function require(name: string): any;
declare const __dirname: string;
declare const process: any;
const path = require("path");
const os = require("os");
const fs = require("fs");
const { spawnSync } = require("child_process");

import { loadLatest } from "./src/cache/cache";
import { runScan } from "./src/scan/orchestrator";

type CliApi = {
  registerCli: (
    registerFn: (ctx: { program: any }) => void,
    meta?: { commands?: string[] },
  ) => void;
};

function defaultOutputRoot(): string {
  return path.resolve(os.homedir(), ".openclaw", "deepsafe", "reports");
}

function defaultOpenClawConfig(): string {
  return path.resolve(os.homedir(), ".openclaw", "openclaw.json");
}

type ResolvedModelSettings = {
  apiBase: string;
  model: string;
  apiKey: string;
  source: "cli" | "openclaw-config";
};

function resolveModelSettings(opts: any, openclawConfigPath: string): ResolvedModelSettings | null {
  const cliApiBase = String(opts.apiBase || "").trim();
  const cliModel = String(opts.model || "").trim();
  const cliApiKey = String(opts.apiKey || "").trim();
  if (cliApiBase && cliModel) {
    return {
      apiBase: cliApiBase,
      model: cliModel,
      apiKey: cliApiKey || "EMPTY",
      source: "cli",
    };
  }

  if (!fs.existsSync(openclawConfigPath)) {
    return null;
  }
  try {
    const cfg = JSON.parse(fs.readFileSync(openclawConfigPath, "utf-8"));
    const primary = String(cfg?.agents?.defaults?.model?.primary ?? "").trim();
    if (!primary.includes("/")) return null;
    const [providerKey, modelId] = primary.split("/", 2);
    const provider = cfg?.models?.providers?.[providerKey];
    const apiBase = String(provider?.baseUrl ?? "").trim();
    const model = String(modelId ?? "").trim();
    const apiKeyFromCfg = String(provider?.apiKey ?? "").trim();
    const apiKey = cliApiKey || apiKeyFromCfg || "EMPTY";
    if (!apiBase || !model) return null;
    return {
      apiBase,
      model,
      apiKey,
      source: "openclaw-config",
    };
  } catch {
    return null;
  }
}

function checkPythonAvailable(): { ok: boolean; message?: string } {
  const probe = spawnSync("python3", ["--version"], { encoding: "utf-8" });
  if (probe.error) {
    const code = String(probe.error?.code ?? "");
    if (code === "ENOENT") {
      return { ok: false, message: "python3 not found in PATH." };
    }
    return { ok: false, message: `failed to execute python3: ${String(probe.error.message ?? probe.error)}` };
  }
  if (probe.status !== 0) {
    return { ok: false, message: String(probe.stderr || probe.stdout || "python3 check failed") };
  }
  return { ok: true };
}

function runLegacyCheck(opts: any): never | void {
  const dimension = String(opts.dimension ?? "").toLowerCase().trim();
  if (dimension !== "persuasion") {
    console.error(
      `unsupported dimension: ${dimension}. currently supported: persuasion`,
    );
    process.exit(2);
  }

  const pyCheck = checkPythonAvailable();
  if (!pyCheck.ok) {
    console.error(`deepsafe error: ${pyCheck.message}`);
    process.exit(2);
  }

  const script = path.resolve(__dirname, "persuasion_probe.py");
  if (!fs.existsSync(script)) {
    console.error(`deepsafe error: persuasion probe not found: ${script}`);
    process.exit(2);
  }

  const pyArgs = [
    script,
    "--api-base",
    String(opts.apiBase),
    "--model",
    String(opts.model),
    "--api-key",
    String(opts.apiKey ?? "EMPTY"),
    "--mode",
    String(opts.mode ?? "fast"),
  ];
  if (opts.limit) pyArgs.push("--limit", String(opts.limit));
  if (opts.turns) pyArgs.push("--n-turns", String(opts.turns));
  if (opts.output) {
    pyArgs.push("--output", String(opts.output));
  }
  if (opts.debug) {
    console.log(`deepsafe debug: script=${script}`);
    console.log(`deepsafe debug: argv=${JSON.stringify(pyArgs)}`);
  }

  const run = spawnSync("python3", pyArgs, { stdio: "inherit" });
  if (run.status !== 0) {
    process.exit(run.status ?? 1);
  }
}

export default function register(api: CliApi) {
  api.registerCli(
    ({ program }) => {
      const cmd = program
        .command("deepsafe")
        .description("DeepSafe lightweight local checks");

      cmd
        .command("check")
        .description("Run DeepSafe check for one dimension")
        .requiredOption("--dimension <name>", "Risk dimension, e.g. persuasion")
        .requiredOption("--api-base <url>", "OpenAI-compatible API base")
        .requiredOption("--model <name>", "Model name")
        .option("--api-key <key>", "API key", "EMPTY")
        .option("--mode <mode>", "Run mode: fast|full", "fast")
        .option("--limit <n>", "Override topic count", "")
        .option("--turns <n>", "Override conversation turns", "")
        .option("--output <path>", "Output JSON path", "")
        .option("--debug", "Enable debug logs", false)
        .action((opts: any) => {
          runLegacyCheck(opts);
        });

      cmd
        .command("scan")
        .description("Run DeepSafe preflight scan")
        .option("--profile <mode>", "Scan profile: quick|full", "quick")
        .option("--ttl-days <n>", "Cache TTL in days", "3")
        .option("--force", "Force re-run and ignore cache", false)
        .option("--output <dir>", "Report output root directory", defaultOutputRoot())
        .option("--openclaw-config <path>", "OpenClaw config path", defaultOpenClawConfig())
        .option("--workspace <path>", "OpenClaw workspace path override", "")
        .option("--api-base <url>", "OpenAI-compatible API base for model scan (optional if in openclaw.json)", "")
        .option("--model <name>", "Model name for model scan (optional if in openclaw.json)", "")
        .option("--api-key <key>", "API key for model scan (optional if in openclaw.json)", "")
        .option("--limit <n>", "Override model scan topic count", "")
        .option("--turns <n>", "Override model scan turns", "")
        .option("--skip-model", "Skip model scanner", false)
        .option("--debug", "Enable debug logs", false)
        .action((opts: any) => {
          const profile = String(opts.profile ?? "quick").toLowerCase();
          if (profile !== "quick" && profile !== "full") {
            console.error(`deepsafe error: invalid --profile ${String(opts.profile)}`);
            process.exit(2);
          }

          const pyCheck = checkPythonAvailable();
          if (!opts.skipModel && !pyCheck.ok) {
            console.error("");
            console.error("╔═══════════════════════════════════════════════════╗");
            console.error("║  ❌ python3 not found                              ║");
            console.error("╚═══════════════════════════════════════════════════╝");
            console.error("");
            console.error("  DeepSafe model security scan requires Python 3.");
            console.error("");
            console.error("  Install:");
            console.error("    macOS:   brew install python3  or  xcode-select --install");
            console.error("    Ubuntu:  sudo apt install python3");
            console.error("    Windows: https://www.python.org/downloads/");
            console.error("");
            console.error("  Re-run after installation. Or use --skip-model to skip model scan.");
            console.error("");
            process.exit(2);
          }

          const ttlNum = Number(opts.ttlDays ?? 3);
          if (!Number.isFinite(ttlNum) || ttlNum < 0) {
            console.error(`deepsafe error: invalid --ttl-days ${String(opts.ttlDays)}`);
            process.exit(2);
          }

          const openclawConfigPath = path.resolve(String(opts.openclawConfig || defaultOpenClawConfig()));
          const resolvedModel = !opts.skipModel ? resolveModelSettings(opts, openclawConfigPath) : null;
          if (!opts.skipModel && !resolvedModel) {
            console.error(
              "deepsafe error: model scan config not found. Provide --api-base/--model or configure agents.defaults.model.primary + models.providers in openclaw.json.",
            );
            process.exit(2);
          }

          if (opts.debug && resolvedModel) {
            console.log(
              `deepsafe debug: model settings source=${resolvedModel.source}, api_base=${resolvedModel.apiBase}, model=${resolvedModel.model}`,
            );
          }

          const result = runScan({
            profile,
            ttlDays: Math.floor(ttlNum),
            force: Boolean(opts.force),
            outputRoot: path.resolve(String(opts.output || defaultOutputRoot())),
            openclawConfigPath,
            workspacePath: String(opts.workspace || ""),
            runModel: !opts.skipModel,
            apiBase: resolvedModel?.apiBase || "",
            model: resolvedModel?.model || "",
            apiKey: resolvedModel?.apiKey || "EMPTY",
            limit: String(opts.limit || ""),
            turns: String(opts.turns || ""),
            debug: Boolean(opts.debug),
          });

          const s = result.report.scores;
          const totalFindings = result.report.findings.length;
          const critCount = result.report.findings.filter((f: any) => f.severity === "CRITICAL").length;
          const highCount = result.report.findings.filter((f: any) => f.severity === "HIGH").length;
          const medCount = result.report.findings.filter((f: any) => f.severity === "MEDIUM").length;
          const lowCount = result.report.findings.filter((f: any) => f.severity === "LOW").length;

          const riskIcon = s.total >= 85 ? "✅" : s.total >= 65 ? "⚠️" : s.total >= 40 ? "🔶" : "🚨";
          const riskLabel = s.total >= 85 ? "LOW RISK" : s.total >= 65 ? "MEDIUM RISK" : s.total >= 40 ? "HIGH RISK" : "CRITICAL";
          const durationSec = (result.report.metadata.durationMs / 1000).toFixed(1);

          console.log("");
          console.log("╔══════════════════════════════════════════╗");
          console.log("║     🛡️  DeepSafe Preflight Report        ║");
          console.log("╚══════════════════════════════════════════╝");
          console.log("");
          console.log(`  ${riskIcon} Overall Score: ${s.total}/100 (${riskLabel})`);
          console.log(`  📊 Findings: ${totalFindings} (🔴${critCount} 🟠${highCount} 🟡${medCount} 🟢${lowCount})`);
          console.log("");
          console.log(`  🔒 Posture: ${s.posture}  🧩 Skill: ${s.skill}  🧠 Model: ${s.model}  💾 Memory: ${s.memory}`);
          console.log("");
          console.log(`  ⏱️  ${durationSec}s | ${result.cacheHit ? "📦 cached" : "🔄 fresh scan"} | profile: ${result.report.metadata.profile}`);
          console.log("");
          console.log(`  🌐 Report: file://${result.paths.htmlPath}`);
          console.log(`  📋 Data:   ${result.paths.jsonPath}`);
          console.log("");

          if (result.exitCode === 2) {
            console.error("  ❌ Some modules failed — check the report for details.");
          }

          // Auto-open HTML report in browser
          try {
            const htmlPath = result.paths.htmlPath;
            const platform = process.platform;
            if (platform === "darwin") {
              spawnSync("open", [htmlPath], { stdio: "ignore" });
            } else if (platform === "win32") {
              spawnSync("cmd", ["/c", "start", "", htmlPath], { stdio: "ignore" });
            } else {
              spawnSync("xdg-open", [htmlPath], { stdio: "ignore" });
            }
          } catch {
            // silently ignore if browser open fails
          }

          process.exit(result.exitCode);
        });

      cmd
        .command("report")
        .description("Show the latest DeepSafe report path")
        .option("--last", "Show latest report", true)
        .option("--output <dir>", "Report output root directory", defaultOutputRoot())
        .action((opts: any) => {
          const latest = loadLatest(path.resolve(String(opts.output || defaultOutputRoot())));
          if (!latest) {
            console.error("deepsafe error: no previous report found.");
            process.exit(2);
          }
          console.log("=== DeepSafe Latest Report ===");
          console.log(`generated_at: ${latest.generatedAt}`);
          console.log(`report_json: ${latest.jsonPath}`);
          console.log(`report_md: ${latest.mdPath}`);
        });
    },
    { commands: ["deepsafe"] },
  );
}

