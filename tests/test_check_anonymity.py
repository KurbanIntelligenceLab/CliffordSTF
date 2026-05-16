"""End-to-end tests for ``tools/check_anonymity.py``.

Tokens are constructed via string concatenation so this test file
itself does not contain literal anonymity substrings (the pre-commit
hook would otherwise flag its own test as a hit).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT_PATH: Path = Path(__file__).resolve().parent.parent / "tools" / "check_anonymity.py"


def _init_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tester"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _stage(repo: Path) -> None:
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)


def _run_checker(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_anonymity_passes_on_clean_repo(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "clean.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n")
    _stage(tmp_path)

    result = _run_checker(tmp_path)
    assert result.returncode == 0, result.stderr


def test_check_anonymity_flags_each_token(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    leaks = {
        "leak1.py": "agents" + "-mlip",
        "leak2.py": "MOLE" + "CULE_SOURCE",
        "leak3.py": "Po" + "lat",
        "leak4.py": "Tun" + "cel",
        "leak5.py": "RTX " + "5090",
        "leak6.py": "mlip" + "-bench",
    }
    for filename, token in leaks.items():
        (tmp_path / filename).write_text(f"# spurious token: {token}\n")
    _stage(tmp_path)

    result = _run_checker(tmp_path)
    assert result.returncode == 1
    for filename, token in leaks.items():
        assert filename in result.stderr
        assert token in result.stderr


def test_check_anonymity_ignores_untracked_files(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "tracked.py").write_text("# clean\n")
    _stage(tmp_path)
    # Untracked file containing a token must NOT trigger the hook.
    (tmp_path / "untracked.py").write_text(f"# {'agents' + '-mlip'}\n")

    result = _run_checker(tmp_path)
    assert result.returncode == 0, result.stderr
