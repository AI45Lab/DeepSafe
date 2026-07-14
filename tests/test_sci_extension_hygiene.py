from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = "f8f7d960ed01220993ae83b023b753e364d4773f"
SCI_CONFIG = re.compile(r"^(scihazard|safe_scientist|sosbench)(?:[_.-].*)?\.ya?ml$")
FORBIDDEN_PATH_PARTS = {
    ".prompt_traces",
    "predictions.jsonl",
    "responses.jsonl",
    "responses.judged.jsonl",
}
FORBIDDEN_TEXT = re.compile(
    rb"(?:BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|"
    rb"sk-[A-Za-z0-9_-]{20,}|"
    rb"/mnt/shared-storage-user/|"
    rb"\brlaunch\b)",
    re.IGNORECASE,
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def changed_paths(*extra: str) -> list[str]:
    output = git("diff", "--name-only", f"{BASE}...HEAD", *extra)
    return output.splitlines() if output else []


def test_no_prompt_traces_or_experiment_outputs_are_tracked() -> None:
    offenders = [
        path
        for path in git("ls-files").splitlines()
        if any(part in Path(path).parts for part in FORBIDDEN_PATH_PARTS)
        or Path(path).name in FORBIDDEN_PATH_PARTS
    ]
    assert offenders == []


def test_new_text_files_contain_no_secrets_internal_paths_or_launchers() -> None:
    offenders: list[str] = []
    for relative in changed_paths("--diff-filter=AM"):
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        data = path.read_bytes()
        if b"\0" not in data and FORBIDDEN_TEXT.search(data):
            offenders.append(relative)
    assert offenders == []


def test_existing_deepsafe_scripts_are_not_deleted() -> None:
    deleted = git(
        "diff", "--diff-filter=D", "--name-only", f"{BASE}...HEAD", "--", "scripts"
    )
    assert deleted == ""


def test_wmdp_files_are_unchanged_from_base() -> None:
    changed = git(
        "diff",
        "--name-only",
        f"{BASE}...HEAD",
        "--",
        "scripts/*wmdp*",
        "uni_eval/*/*wmdp*",
        "configs/**/*wmdp*",
    )
    assert changed == ""


def test_only_approved_science_configs_are_added() -> None:
    added = git(
        "diff", "--diff-filter=A", "--name-only", f"{BASE}...HEAD", "--", "configs"
    )
    offenders = [
        path
        for path in added.splitlines()
        if not SCI_CONFIG.fullmatch(Path(path).name)
    ]
    assert offenders == []


def test_required_science_components_exist() -> None:
    required = [
        "uni_eval/datasets/scihazard.py",
        "uni_eval/datasets/safe_scientist.py",
        "uni_eval/datasets/sosbench.py",
        "uni_eval/evaluators/scihazard.py",
        "uni_eval/evaluators/safe_scientist.py",
        "uni_eval/evaluators/sosbench.py",
        "uni_eval/metrics/scihazard_metric.py",
        "uni_eval/metrics/safe_scientist_metric.py",
        "uni_eval/metrics/sosbench_metric.py",
    ]
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    assert missing == []

