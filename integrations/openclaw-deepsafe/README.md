# DeepSafe OpenClaw MVP (Persuasion)

This plugin is a preflight security scanner for OpenClaw.

## What it provides

- OpenClaw commands:
  - `openclaw deepsafe scan` (preflight scan)
  - `openclaw deepsafe report --last` (show latest report paths)
  - `openclaw deepsafe check --dimension persuasion` (legacy compatible)
- CPU-only local execution
- Reuses DeepSafe manipulation prompt + metric logic as much as possible
- Warning-style output (does not block user workflows)

## Install (local dev link)

From your machine:

```bash
openclaw plugins install -l /mnt/shared-storage-user/zhangbo1/DeepSafe/integrations/openclaw-deepsafe
```

Then restart OpenClaw Gateway.

## Usage

Preflight quick scan (recommended default):

```bash
openclaw deepsafe scan \
  --profile quick \
  --api-base http://127.0.0.1:11434/v1 \
  --model qwen2.5-7b-instruct
```

Show latest report paths:

```bash
openclaw deepsafe report --last
```

Skip model scanner when only checking local posture/skills/memory:

```bash
openclaw deepsafe scan --skip-model
```

Legacy one-dimension check:

```bash
openclaw deepsafe check \
  --dimension persuasion \
  --api-base http://127.0.0.1:11434/v1 \
  --model qwen2.5-7b-instruct \
  --api-key EMPTY \
  --mode fast \
  --limit 1 \
  --output ./deepsafe_persuasion_report.json
```

Full mode (manual opt-in):

```bash
openclaw deepsafe check \
  --dimension persuasion \
  --api-base http://127.0.0.1:11434/v1 \
  --model qwen2.5-7b-instruct \
  --mode full
```

## Notes

- This MVP currently expects OpenAI-compatible chat completion APIs.
- The plugin invokes `python3` and runs `persuasion_probe.py`.
- Default dataset: `DeepSafe/data/manipulation_persuasion_topics.json`.
- Default mode is `fast`; use `--mode full` for heavier full-scale runs.
- `scan` writes:
  - `deepsafe_report.json`
  - `deepsafe_report.md`
  under `~/.openclaw/deepsafe/reports/<timestamp>/`.

## Debug checklist

- Add `--debug` to print argv and store model scan stdout/stderr logs.
- Use `--force` to bypass cache and force a fresh scan.
- If model scan fails, verify:
  - `python3 --version` works
  - `--api-base` is reachable
  - target model supports OpenAI-compatible `/chat/completions`.

