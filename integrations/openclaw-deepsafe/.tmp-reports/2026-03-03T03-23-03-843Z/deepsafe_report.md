# DeepSafe Scan Report

- Generated at: 2026-03-03T03:23:03.843Z
- Profile: quick
- Plugin version: 0.1.0
- Highest severity: MEDIUM
- From cache: no

## Scores

- Total: 60
- Posture: 77
- Skill: 68
- Model: 0
- Memory: 95

## Module Status

- posture: warn; score=77; findings=1; duration=0ms
- skill: warn; score=68; findings=2; duration=9ms
- model: error; score=0; findings=0; duration=0ms; error=Model scan requires --api-base and --model unless you use --no-model.
- memory: ok; score=95; findings=0; duration=1ms

## Findings

### [MEDIUM] Provider API key is stored in plain config
- ID: posture-provider-inline-key-zhangbo
- Category: posture
- Evidence: models.providers.zhangbo.apiKey is non-empty in /Users/zhangbo/.openclaw/openclaw.json
- Remediation: Move secrets to environment variables or secret manager, and rotate exposed keys.

### [MEDIUM] Potentially dangerous execution primitive detected in skill files
- ID: skill-dangerous-runtime-21
- Category: skill
- Evidence: /Users/zhangbo/.openclaw/workspace/skills/obsidian-direct/scripts/obsidian_search.py
- Remediation: Restrict command execution paths and validate all user-controlled inputs.

### [MEDIUM] Potentially dangerous execution primitive detected in skill files
- ID: skill-dangerous-runtime-30
- Category: skill
- Evidence: /Users/zhangbo/.openclaw/workspace/skills/playwright-scraper-skill/examples/README.md
- Remediation: Restrict command execution paths and validate all user-controlled inputs.
