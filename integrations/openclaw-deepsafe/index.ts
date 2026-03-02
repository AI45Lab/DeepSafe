declare function require(name: string): any;
declare const __dirname: string;
declare const process: any;
const path = require("path");
const { spawnSync } = require("child_process");

type CliApi = {
  registerCli: (
    registerFn: (ctx: { program: any }) => void,
    meta?: { commands?: string[] },
  ) => void;
};

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
        .action((opts: any) => {
          const dimension = String(opts.dimension ?? "").toLowerCase().trim();
          if (dimension !== "persuasion") {
            console.error(
              `unsupported dimension: ${dimension}. currently supported: persuasion`,
            );
            process.exit(2);
          }
          const script = path.resolve(__dirname, "persuasion_probe.py");
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

          const run = spawnSync("python3", pyArgs, { stdio: "inherit" });
          if (run.status !== 0) {
            process.exit(run.status ?? 1);
          }
        });
    },
    { commands: ["deepsafe"] },
  );
}

