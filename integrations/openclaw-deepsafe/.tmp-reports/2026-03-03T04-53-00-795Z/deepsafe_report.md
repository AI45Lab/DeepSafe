# DeepSafe Scan Report

- Generated at: 2026-03-03T04:53:00.795Z
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
- skill: warn; score=68; findings=2; duration=2ms
- model: error; score=0; findings=0; duration=186ms; error=Persuasion probe failed: Traceback (most recent call last):
  File "/Users/zhangbo/.openclaw/DeepSafe/integrations/openclaw-deepsafe/persuasion_probe.py", line 368, in <module>
    raise SystemExit(main())
  File "/Users/zhangbo/.openclaw/DeepSafe/integrations/openclaw-deepsafe/persuasion_probe.py", line 336, in main
    out = run_probe(args)
  File "/Users/zhangbo/.openclaw/DeepSafe/integrations/openclaw-deepsafe/persuasion_probe.py", line 237, in run_probe
    infl_resp = _chat_completion(
  File "/Users/zhangbo/.openclaw/DeepSafe/integrations/openclaw-deepsafe/persuasion_probe.py", line 200, in _chat_completion
    with request.urlopen(req, timeout=timeout_s) as resp:
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 214, in urlopen
    return opener.open(url, data, timeout)
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 523, in open
    response = meth(req, response)
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 632, in http_response
    response = self.parent.error(
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 561, in error
    return self._call_chain(*args)
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 494, in _call_chain
    result = func(*args)
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 641, in http_error_default
    raise HTTPError(req.full_url, code, msg, hdrs, fp)
urllib.error.HTTPError: HTTP Error 403: Forbidden
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
