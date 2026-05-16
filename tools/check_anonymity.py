"""Anonymity check for the CliffordSTF repository.

Scans all git-tracked text files for source-repo identifiers that must
not appear in the anonymous submission. Exits 0 on a clean tree; on a
hit, exits 1 and writes ``<path>:<lineno>: <line>`` for every offending
line to stderr.

Wired into ``.pre-commit-config.yaml`` as the local ``check-anonymity``
hook (``pass_filenames: false``). Can also be run manually:

    uv run python tools/check_anonymity.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ANONYMITY_TOKENS: tuple[str, ...] = (
    "agents-mlip",
    "MOLECULE_SOURCE",
    "Polat",
    "Tuncel",
    "RTX 5090",
    "mlip-bench",
)

_PATTERN: re.Pattern[str] = re.compile("|".join(re.escape(t) for t in ANONYMITY_TOKENS))

_SELF_PATH: Path = Path(__file__).resolve()


def _tracked_files(repo_root: Path) -> list[Path]:
    """Return absolute paths of all git-tracked files under ``repo_root``."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        check=True,
        cwd=repo_root,
    )
    return [
        repo_root / relpath.decode("utf-8") for relpath in result.stdout.split(b"\x00") if relpath
    ]


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return ``[(lineno, line)]`` for lines matching any anonymity token."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    return [
        (lineno, line)
        for lineno, line in enumerate(text.splitlines(), start=1)
        if _PATTERN.search(line)
    ]


def main(repo_root: Path | None = None) -> int:
    """Scan tracked files for anonymity tokens; return 1 if any hit."""
    root = (repo_root if repo_root is not None else Path.cwd()).resolve()
    hits: list[tuple[Path, int, str]] = []
    for path in _tracked_files(root):
        if not path.is_file():
            continue
        if path.resolve() == _SELF_PATH:
            continue
        for lineno, line in _scan_file(path):
            hits.append((path.relative_to(root), lineno, line))
    if hits:
        sys.stderr.write(f"anonymity check FAILED ({len(hits)} hits):\n")
        for path, lineno, line in hits:
            sys.stderr.write(f"  {path}:{lineno}: {line}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
