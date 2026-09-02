"""Every Kaggle notebook cell must be valid JSON and valid Python.

This exists because two broken notebooks shipped to a user who then burned real time on
Kaggle discovering the breakage:

1. An `assert` on `torch.cuda.is_bf16_supported()` that fires on a healthy T4 (PyTorch
   reports *emulated* bf16 support on pre-Ampere cards).
2. An f-string containing a literal newline instead of a `\\n` escape, so the string was
   never terminated: `SyntaxError: unterminated f-string literal`.

Neither is catchable by reading the notebook JSON — it parses fine. Both are caught by
compiling each cell. A notebook is code; it deserves the same gate as the rest of the repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "kaggle"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def _python_source(cell: dict) -> str:
    """Cell source minus IPython shell magics (`!cmd`), which are not valid Python."""
    src = "".join(cell.get("source", []))
    return "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("!", "%")))


def test_notebook_dir_is_not_empty() -> None:
    """Guard against the glob silently matching nothing and the whole module passing."""
    assert NOTEBOOKS, f"no notebooks found under {NOTEBOOK_DIR}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert "cells" in nb, f"{path.name} has no 'cells' key"
    assert nb["cells"], f"{path.name} has no cells"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_compiles(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = _python_source(cell)
        if not source.strip():
            continue
        try:
            compile(source, f"{path.name}:cell{i}", "exec")
        except SyntaxError as exc:
            failures.append(f"  cell {i}: {exc.msg} (line {exc.lineno})")
    assert not failures, f"{path.name} has cells that will not run:\n" + "\n".join(failures)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_unfilled_placeholders(path: Path) -> None:
    """A placeholder that reaches a user costs them a GPU session to discover."""
    text = path.read_text(encoding="utf-8")
    for token in ("YOUR_USERNAME", "YOUR_HF_USERNAME", "<your-", "TODO:"):
        assert token not in text, f"{path.name} still contains the placeholder {token!r}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_hardcoded_secrets(path: Path) -> None:
    """Tokens belong in Kaggle Secrets, never in a cell that gets committed."""
    text = path.read_text(encoding="utf-8")
    for token in ("hf_", "ghp_", "sk-"):
        # `get_secret('HF_TOKEN')` and the env var name itself are fine; a literal is not.
        for line in text.splitlines():
            if token in line and "get_secret" not in line and "HF_TOKEN" not in line:
                raise AssertionError(f"{path.name}: possible hardcoded secret: {line.strip()[:80]}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_shell_commands_are_single_line(path: Path) -> None:
    r"""No backslash continuations in `!` cells.

    Writing notebook JSON turned an intended `\<newline>` continuation into a literal
    backslash followed by the letter `n`, so the shell received `\n` as an argument:
    `error: unrecognized arguments: n`. The notebook parsed, the cell compiled, and it
    still failed on the user's machine. One line per command removes the failure mode.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        for line in cell.get("source", []):
            body = line.rstrip("\n")
            if not body.lstrip().startswith(("!", "# !")):
                continue
            if body.endswith("\\") or body.endswith("\n"):
                offenders.append(f"  cell {i}: {body[:90]}")
    assert not offenders, f"{path.name}: shell commands must be one line:\n" + "\n".join(offenders)
