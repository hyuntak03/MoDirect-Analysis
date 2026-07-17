"""Structural invariants of the repo layout.

Every assertion here guards a failure mode that is SILENT — no exception, no warning,
just wrong results much later:

  * core/ moving, or being vendored into modirect/, breaks nine scripts that load it by
    literal filesystem path via `spec_from_file_location`.
  * tasks/ ceasing to be a sibling of core/ makes `discover_tasks()` return an EMPTY
    registry (core/dataset_loader.py:133-135 warns and continues), which surfaces as a
    puzzling `ValueError: Unknown task` in an unrelated place.
  * pyproject.toml being renamed breaks the repo-root walk every migrated script does.
  * A stale absolute path resolves to nothing on this host — the old `/data/takhyun03`
    root no longer exists.

These are cheap filesystem/AST checks. Nothing here imports torch, llava, or `core`
itself: `core/__init__.py:1` pulls in `core.model_loader`, which imports llava at module
scope (`core/model_loader.py:10`), so `import core` is impossible on a host without the
model runtime — which is exactly the host this suite must run on.
"""

from __future__ import annotations

import ast
import os
import py_compile
import re
import sys
from pathlib import Path

import pytest


def _find_root() -> Path:
    """Walk up for pyproject.toml — the same marker the migrated scripts use."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise AssertionError(
        "no pyproject.toml found walking up from tests/ — the repo-root marker is gone, "
        "which silently breaks _PROJECT_ROOT resolution in every pipeline script")


ROOT = _find_root()

# Directories that are checked out but not ours to police for path hygiene.
_SCAN_EXCLUDE = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".venv"}


def _iter_files(*suffixes: str):
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SCAN_EXCLUDE
                       and not d.endswith(".egg-info")]
        for name in filenames:
            if name.endswith(suffixes):
                yield Path(dirpath) / name


# ------------------------------------------------------------------ root marker

def test_pyproject_exists_at_root():
    """The repo-root marker every migrated script walks up to find."""
    assert (ROOT / "pyproject.toml").is_file()


def test_pyproject_declares_explicit_packages():
    """Auto-discovery must stay off, or tasks/ gets swept in as a namespace package."""
    text = (ROOT / "pyproject.toml").read_text()
    assert "[tool.setuptools]" in text
    assert re.search(r"^packages\s*=\s*\[", text, re.M), (
        "packages must be an EXPLICIT list; automatic discovery would package tasks/")


@pytest.mark.parametrize("pkg", [
    "modirect", "modirect.concepts", "modirect.config", "modirect.interventions",
    "modirect.io", "modirect.models", "modirect.probing", "modirect.tasks", "core",
])
def test_declared_packages_exist_on_disk_and_are_importable_dirs(pkg: str):
    """Every entry in `packages` is a real directory with an __init__.py.

    With an explicit list, a stale entry does not fail the build — it fails at install
    time, or ships a distribution missing a module.
    """
    pkg_dir = ROOT.joinpath(*pkg.split("."))
    assert pkg_dir.is_dir(), f"{pkg} declared in pyproject but {pkg_dir} is missing"
    assert (pkg_dir / "__init__.py").is_file(), (
        f"{pkg} has no __init__.py; setuptools would ship it as a namespace package")


def test_every_modirect_subpackage_is_declared():
    """A new subpackage that nobody added to `packages` installs incomplete, silently."""
    declared = set(
        re.findall(r'"(modirect(?:\.[a-z_]+)?)"', (ROOT / "pyproject.toml").read_text()))
    on_disk = {
        f"modirect.{p.name}"
        for p in (ROOT / "modirect").iterdir()
        if p.is_dir() and (p / "__init__.py").is_file()
    }
    missing = on_disk - declared
    assert not missing, f"subpackages on disk but absent from pyproject packages: {missing}"


# ------------------------------------------------------------------ core/ literal paths

@pytest.mark.parametrize("name", ["model_loader.py", "dataset_loader.py"])
def test_core_runtime_files_exist_at_root(name: str):
    """Nine scripts load these by literal path: join(_PROJECT_ROOT, "core", name).

    core/ must stay at the repo root and keep working standalone; modirect/ wraps it
    rather than replacing or copying it.
    """
    assert (ROOT / "core" / name).is_file()


def test_core_is_not_duplicated_inside_modirect():
    """A vendored copy would drift from the literal-path original — two truths."""
    assert not (ROOT / "modirect" / "core").exists()


# ------------------------------------------------------------------ tasks/ sibling rule

def test_tasks_is_a_sibling_of_core():
    assert (ROOT / "tasks").is_dir()
    assert (ROOT / "core").is_dir()
    assert (ROOT / "tasks").parent == (ROOT / "core").parent


def test_dataset_loader_default_tasks_dir_resolves_to_root_tasks():
    """Reproduce core/dataset_loader.py:119 exactly, without importing it.

        _DEFAULT_TASKS_DIR = dirname(dirname(abspath(__file__))) / "tasks"

    Importing core to check this is not an option (it imports llava), so the computation
    is mirrored here against the real file location. If this drifts, discover_tasks()
    scans the wrong directory and the registry comes back EMPTY with no exception.
    """
    dataset_loader = ROOT / "core" / "dataset_loader.py"
    computed = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(str(dataset_loader)))), "tasks")

    assert Path(computed).resolve() == (ROOT / "tasks").resolve()
    assert Path(computed).is_dir()


def test_dataset_loader_still_computes_tasks_dir_the_way_we_assume():
    """Guard the mirrored computation above against edits to core/dataset_loader.py."""
    src = (ROOT / "core" / "dataset_loader.py").read_text()
    assert "_DEFAULT_TASKS_DIR" in src
    assert re.search(
        r'_DEFAULT_TASKS_DIR\s*=\s*os\.path\.join\(\s*os\.path\.dirname\('
        r'\s*os\.path\.dirname\(\s*os\.path\.abspath\(__file__\)\)\)\s*,\s*"tasks"\s*\)',
        src), (
        "core/dataset_loader.py no longer computes _DEFAULT_TASKS_DIR as "
        "dirname(dirname(__file__))/tasks — test_dataset_loader_default_tasks_dir_"
        "resolves_to_root_tasks mirrors that expression and must be updated with it")


def test_tasks_dir_holds_the_yaml_registry():
    """>=150 YAML task definitions. An empty scan is the silent-failure signature."""
    yamls = list((ROOT / "tasks").rglob("*.yaml"))
    assert len(yamls) >= 150, f"only {len(yamls)} task yaml files found under tasks/"


def test_tasks_data_dir_is_not_a_python_package():
    """tasks/ must have NO __init__.py.

    It holds utils.py files in 10 subdirectories, which is what makes it look like a
    package to setuptools auto-discovery. It is a DATA directory located by filesystem
    path, never imported. Giving it an __init__.py invites `import tasks` to shadow
    modirect.tasks.
    """
    assert not (ROOT / "tasks" / "__init__.py").exists()


def test_tasks_yaml_is_not_gitignored():
    """If the registry YAML were ignored, discover_tasks() would find nothing."""
    gitignore = (ROOT / ".gitignore").read_text()
    assert "!/tasks/**/*.yaml" in gitignore


# ------------------------------------------------------------------ pipeline compiles

_PIPELINE_PY = sorted(p for p in (ROOT / "pipeline").rglob("*.py"))

# KNOWN BROKEN — a real, pre-existing bug, not a test-environment artefact.
#
# pipeline/05_intervene/swap/04_summarize.py:21
#     print(f"{'src\\tgt':>14s}", end="")
# A backslash inside an f-string EXPRESSION is a SyntaxError before Python 3.12
# (PEP 701 lifted the restriction in 3.12). This project declares
# requires-python = ">=3.10", so on 3.10 and 3.11 this script cannot even be parsed.
#
# It is byte-identical to the original in cross-modal-info, so the migration did not
# introduce it — it arrived with the file, and pipeline/ is off-limits to this change.
# Fix (for whoever owns pipeline/): hoist the literal out of the f-string, e.g.
#     header = "src\\tgt"
#     print(f"{header:>14s}", end="")
# When that lands, this entry must be deleted; the xfail below is strict, so a fixed
# file turns into an XPASS failure that says so.
_KNOWN_UNCOMPILABLE = {"pipeline/05_intervene/swap/04_summarize.py"}


@pytest.mark.parametrize(
    "path", _PIPELINE_PY, ids=lambda p: str(p.relative_to(ROOT)))
def test_pipeline_scripts_compile(path: Path, request, tmp_path):
    """Every pipeline script parses. Catches migration typos without executing anything.

    py_compile only parses/compiles — it does not import, so torch and llava are never
    touched. The bytecode goes to a tmp_path (NOT os.devnull, which py_compile cannot
    write to) so the repo is never littered with .pyc files.
    """
    rel = str(path.relative_to(ROOT))
    if rel in _KNOWN_UNCOMPILABLE and sys.version_info < (3, 12):
        request.node.add_marker(pytest.mark.xfail(
            strict=True,
            reason=f"{rel}: backslash in f-string expression; SyntaxError before 3.12 "
                   f"(PEP 701). Pre-existing upstream bug — see _KNOWN_UNCOMPILABLE."))
    py_compile.compile(str(path), doraise=True, cfile=str(tmp_path / "out.pyc"))


def test_pipeline_has_scripts_to_check():
    """Guard against the glob silently matching nothing."""
    assert len(_PIPELINE_PY) >= 50


# ------------------------------------------------------------------ path hygiene

# The old root. Matched only when NOT preceded by "nas2", because the live root is
# /nas2/data/takhyun03/... and would otherwise match on every correct path.
_STALE_ROOT = re.compile(r"(?<!nas2)/data/takhyun03")
_STALE_USER = "jong980812"

# This file necessarily contains both patterns — it defines them. Scanning it would be a
# guaranteed self-match, so it is the one exemption.
_HYGIENE_SELF = Path(__file__).resolve()


def _strip_comments(text: str) -> str:
    """Blank out `#` comments, leaving line numbers intact.

    Needed for the same reason the .py scan skips docstrings: comments legitimately cite
    the dead path as history, e.g. configs/paths.example.yaml:105
        # /data/takhyun03/.../work_dirs (mechanism_diagnosis.py:37) does not exist here.
    Only a LIVE value is a bug. `#` inside quotes is not a comment, so quote state is
    tracked rather than naively splitting on the first `#`.
    """
    out = []
    for line in text.splitlines():
        quote = None
        cut = len(line)
        for i, ch in enumerate(line):
            if quote:
                if ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch == "#":
                cut = i
                break
        out.append(line[:cut])
    return "\n".join(out)


def _runtime_strings(path: Path):
    """Yield executable string constants from a .py file, skipping docstrings.

    Scoped deliberately. Several modules legitimately QUOTE the old path in prose while
    documenting what was fixed, e.g.

        modirect/config/paths.py:8
            analysis/task_invariance/mechanism_diagnosis.py:37
                BASELINE_LORA = "/data/takhyun03/..."

    That citation is the point of the docstring; banning the substring outright would
    force those docs to lie. What must never exist is a LIVE stale path — a string the
    code actually uses. So: parse with ast, drop docstrings, and scan what is left.
    """
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except SyntaxError:
        return  # covered by test_pipeline_scripts_compile

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))

    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            yield node.lineno, node.value


@pytest.mark.parametrize(
    "path", [p for p in sorted(_iter_files(".py")) if p.resolve() != _HYGIENE_SELF],
    ids=lambda p: str(p.relative_to(ROOT)))
def test_no_live_stale_root_path_in_python(path: Path):
    """No executable string may point at the dead /data/takhyun03 root.

    Docstrings that cite the old path as history are allowed on purpose — see
    `_runtime_strings`.
    """
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno}: {value!r}"
        for lineno, value in _runtime_strings(path)
        if _STALE_ROOT.search(value)
    ]
    assert not offenders, "live reference(s) to the dead root:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "path", sorted(_iter_files(".sh", ".yaml", ".yml", ".cfg", ".toml", ".json")),
    ids=lambda p: str(p.relative_to(ROOT)))
def test_no_stale_root_path_in_config_files(path: Path):
    """No config/shell VALUE may point at the dead root (comments may cite it)."""
    lines = _strip_comments(path.read_text(errors="ignore")).splitlines()
    hits = [i + 1 for i, line in enumerate(lines) if _STALE_ROOT.search(line)]
    assert not hits, f"{path.relative_to(ROOT)} references the dead root at lines {hits}"


@pytest.mark.parametrize(
    "path",
    [p for p in sorted(_iter_files(".py", ".sh", ".yaml", ".yml", ".md", ".toml"))
     if p.resolve() != _HYGIENE_SELF],
    ids=lambda p: str(p.relative_to(ROOT)))
def test_no_other_users_home_paths(path: Path):
    """No paths under another user's account — they are unreadable from this one.

    Scanned raw, comments included: unlike the dead root, this username has no reason to
    appear even in prose.
    """
    text = path.read_text(errors="ignore")
    hits = [i + 1 for i, line in enumerate(text.splitlines()) if _STALE_USER in line]
    assert not hits, f"{path.relative_to(ROOT)} references {_STALE_USER} at lines {hits}"


# ------------------------------------------------------------------ import contract

def test_modirect_imports_without_torch_or_llava():
    """The whole point of the lazy-core rule: modirect must import on a bare host."""
    import modirect  # noqa: F401

    assert "llava" not in sys.modules
    assert "torch" not in sys.modules


def test_modirect_never_imports_core_at_module_scope():
    """`import core` fires llava (core/model_loader.py:10) and would break everything.

    Module-level `import core` / `from core import ...` is the regression this guards;
    lazy imports inside functions are the supported pattern.
    """
    offenders = []
    for path in (ROOT / "modirect").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for node in tree.body:  # module scope only — nested imports are the fix, not the bug
            if isinstance(node, ast.Import):
                if any(a.name == "core" or a.name.startswith("core.")
                       for a in node.names):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "core"
                                    or node.module.startswith("core.")):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, (
        "module-scope core import (pulls in llava at import time):\n"
        + "\n".join(offenders))
