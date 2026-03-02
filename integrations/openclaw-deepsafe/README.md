# DeepSafe OpenClaw MVP (Persuasion)

This is a one-dimension MVP plugin focused on persuasion/manipulation risk.

## What it provides

- OpenClaw command: `openclaw deepsafe check --dimension persuasion`
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

