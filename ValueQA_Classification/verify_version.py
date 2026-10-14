"""Verify the delivered ValueQA source/config manifest; no inference or network."""
import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def file_sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(manifest_path, root):
    document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest = document["manifest"]
    files = manifest["files"]
    if (manifest.get("schema") != "deepsafe_qwen_evidence_source_manifest_v1"
            or not isinstance(files, dict) or not files):
        raise ValueError("Invalid source/config manifest schema")
    serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
    expected = document["sha256"]
    computed = hashlib.sha256(serialized).hexdigest()
    if computed != expected:
        raise ValueError("Manifest digest mismatch; do not use this checklist")
    errors = []
    for name, expected_file in files.items():
        if (not isinstance(name, str) or "\\" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or not isinstance(expected_file, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_file) is None):
            raise ValueError("Invalid checklist entry")
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Checklist path resolves outside the project")
        if not path.is_file():
            errors.append("MISSING: " + name)
        elif file_sha256(path) != expected_file:
            errors.append("CHANGED: " + name)
    return expected, len(files), errors


def main(argv=None):
    module = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=module / "framework_sha256.json")
    parser.add_argument("--root", type=Path, default=module.parent,
                        help="Complete DeepSafe root; defaults to the module's parent")
    args = parser.parse_args(argv)
    try:
        fingerprint, count, errors = verify(args.manifest.resolve(), args.root.resolve())
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print("FAIL: " + str(exc))
        return 1
    if errors:
        print("FAIL: source/config files do not match the delivered version")
        for error in errors:
            print(error)
        return 1
    print(f"PASS: {count} source/config files match")
    print("ValueQA SHA-256: " + fingerprint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
