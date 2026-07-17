"""Typed wrapper over `core/dataset_loader.py` — the lmms_eval-style task spine.

WHY THIS MODULE EXISTS
----------------------
`core/dataset_loader.py` implements a small lmms_eval clone: drop a YAML into the
top-level `tasks/` directory and the task registers itself. That mechanism is excellent
and is used by every extraction script in `pipeline/01_extract/`. What it lacks is a
discoverable surface — the rules below are all real, all load-bearing, and all only
visible by reading 473 lines of source. This module delegates every call to core and
documents the rules.

THE FIVE RULES (each verified against the source, cited inline)
--------------------------------------------------------------

1. **``!function module.func`` resolves against the YAML file's OWN directory.**
   `_function_constructor` (`core/dataset_loader.py:25-53`) takes
   ``dirname(loader.name)`` — the directory of the YAML being parsed — and looks for
   ``{module}.py`` there (`:36-43`), loading it by file path. Only if that file does not
   exist does it fall back to a normal absolute `import_module` (`:46-48`). So
   ``doc_to_visual: !function utils.doc_to_visual`` in
   `tasks/vlm_direction_testbed/*.yaml` means *that folder's* `utils.py` — it is not on
   `sys.path` and there is no package named `utils`. Two task folders may each have their
   own `utils.py` with the same function names and they will not collide: the loader
   registers each under a unique synthetic module name ``f"tasks.{module}_{id(loader)}"``
   (`:39-40`).

2. **``include:`` gives child-overrides-parent inheritance.** `load_yaml_config`
   (`:94-108`) pops ``include``, resolves each entry relative to the YAML's directory
   (`:101-102`), loads the parents in order (later parent wins over earlier), then does
   ``base_config.update(config)`` (`:107`) so the **child wins over every parent**.
   Caveat, and it bites: `dict.update` is **shallow**. A child that redefines one key of
   ``dataset_kwargs`` replaces the whole ``dataset_kwargs`` dict, silently dropping the
   parent's ``token: True``. Redeclare nested blocks in full.

3. **``task: str`` registers a task; ``task: list`` + ``group: str`` registers a group.**
   `discover_tasks` (`:164-170`) branches on the *type* of the ``task`` value: a string
   goes into `_TASK_REGISTRY`, a list goes into `_GROUP_REGISTRY` under the ``group``
   name — and **a list with no ``group:`` key is silently discarded** (`:168-170`). A
   YAML with no ``task`` key at all is skipped (`:159-161`), which is how shared parents
   like `tasks/vlm_direction_testbed/default.yaml` avoid registering themselves.

4. **Files whose name starts with ``"_"`` are skipped** (`:146-147`) — the template
   convention, e.g. `tasks/mvbench/_default_template_yaml`. Note the registry only
   considers files ending in ``.yaml`` in the first place (`:143-144`), so a template
   named ``_default_template_yaml`` (no dot) is doubly excluded.

5. **`discover_tasks()` runs AT IMPORT TIME of `core.dataset_loader`** (`:467`, module
   bottom), against `_DEFAULT_TASKS_DIR` = ``dirname(dirname(core/dataset_loader.py))``
   + ``"/tasks"`` (`:119`) — i.e. **`tasks/` must remain a sibling of `core/` at the repo
   root**. Today that resolves to ``<repo>/tasks`` and registers 119 tasks plus their
   groups, with zero calls from anyone. If that directory does not exist, `discover_tasks`
   prints a warning and returns (`:133-135`); it **does not raise**. The result is a
   *silently empty registry* and a downstream `ValueError: Unknown task` far from the real
   cause. If `list_tasks()` comes back empty, suspect the directory, not the YAML.

A SIXTH RULE, UNDOCUMENTED IN CORE: **discovery ACCUMULATES.**
`discover_tasks` mutates the module-level `_TASK_REGISTRY` / `_GROUP_REGISTRY` without
ever clearing them (`:128-172`). Calling it a second time with a different `task_dir`
therefore *merges* into the registry from the first call rather than replacing it. This
matters for `tasks_dir=` below: it is additive, and the import-time scan of the default
directory has already happened before you get a chance to pass it.
"""

from __future__ import annotations

from typing import Any

from .schema import Question

__all__ = [
    "discover_tasks",
    "list_tasks",
    "get_task_config",
    "expand_group",
    "is_group",
    "load_questions",
]


def _core() -> Any:
    """Import `core.dataset_loader` lazily.

    Lazy for two reasons.

    First, **side effects**: importing `core.dataset_loader` walks the entire `tasks/`
    tree at module scope (`core/dataset_loader.py:467`). Deferring it keeps
    `import modirect` free of filesystem I/O.

    Second, **dependencies**, and this is heavier than it looks. `dataset_loader` itself
    only needs `yaml`/`datasets`/`pandas` (`:11-13`), but you cannot import it without
    executing the package `__init__` — and `core/__init__.py:1-2` does::

        from core.methods import *                   # -> torch, matplotlib
        from core.model_loader import ...            # -> llava

    So **`modirect.tasks` transitively requires the full GPU stack (llava included)**, for
    no reason of its own. Verified: `from core import dataset_loader` on a host without
    torch raises `ModuleNotFoundError: No module named 'torch'` from `core/methods.py:4`.

    This is imported the normal way regardless — *not* by file path via
    `spec_from_file_location`, even though that would sidestep `core/__init__.py` and the
    nine migrated scripts use exactly that trick. Doing so here would create a **second,
    independent module object** for `dataset_loader`, with its own `_TASK_REGISTRY`. Two
    registries whose `discover_tasks` calls silently do not see each other is a far nastier
    failure than an honest ImportError. One task registry, one module object.

    If the dependency ever needs to go away, the fix is in core: thin out
    `core/__init__.py` so importing a submodule does not drag in llava.
    """
    from core import dataset_loader

    return dataset_loader


def discover_tasks(tasks_dir: str | None = None) -> dict[str, str]:
    """Scan a directory tree for task YAMLs and merge them into the registry.

    You rarely need this: `core.dataset_loader` already scanned the default `tasks/`
    directory at import time (rule 5). Call it only to *add* a directory.

    Args:
        tasks_dir: directory to scan. None uses core's `_DEFAULT_TASKS_DIR`
            (``<repo>/tasks``, the sibling of `core/`). Provided as an explicit parameter
            so the sibling assumption is overridable — for tests with a fixture
            directory, or for a task tree living outside the repo.

    Returns:
        The task-name -> yaml-path registry, **after** merging (see the sixth rule: this
        is cumulative, not a fresh scan; entries from earlier scans remain, and a name
        collision resolves last-scan-wins).

    Note:
        A non-existent `tasks_dir` prints a warning and returns the registry unchanged.
        It does not raise. Check the return value if you need certainty.
    """
    return _core().discover_tasks(tasks_dir)


def list_tasks(include_groups: bool = True, *, tasks_dir: str | None = None) -> list[str]:
    """List registered task names (and group names).

    Args:
        include_groups: also include group names. Groups are not loadable via
            `get_task_config` — see `expand_group`.
        tasks_dir: optional extra directory to discover first (see `discover_tasks`).

    Returns:
        Sorted names. **An empty list almost always means a wrong tasks directory**, not
        an absent task — see rule 5 in the module docstring.
    """
    if tasks_dir is not None:
        discover_tasks(tasks_dir)
    return _core().list_tasks(include_groups=include_groups)


def get_task_config(name: str, *, tasks_dir: str | None = None) -> dict[str, Any]:
    """Load a task's fully-resolved config, with `!function` tags materialised.

    Discovery parses YAMLs in ``"simple"`` mode, keeping `!function` tags as plain strings
    (`core/dataset_loader.py:151`); this is the call that re-parses in ``"full"`` mode
    (`:205`) and actually executes the referenced `utils.py`. So `doc_to_visual`,
    `doc_to_text`, `doc_to_target` and friends come back as **callables**, not strings —
    and a broken `utils.py` fails here, not at discovery.

    Args:
        name: an individual task name. Passing a **group** name raises (`:207-213`) —
            groups have no config of their own; use `expand_group`.
        tasks_dir: optional extra directory to discover first.

    Returns:
        The merged config dict (`include:` parents already folded in, child-wins).

    Raises:
        ValueError: if `name` is a group, or is not registered at all.
    """
    if tasks_dir is not None:
        discover_tasks(tasks_dir)
    return _core().get_task_config(name)


def expand_group(name: str, *, tasks_dir: str | None = None) -> list[str]:
    """Expand a group into its member task names; a plain task returns ``[name]``.

    Nested groups are expanded recursively (`core/dataset_loader.py:175-185`). A member
    that is registered as neither task nor group is **warned about and dropped**
    (`:184`), not raised on — so a typo in a group's member list quietly shrinks the
    group. Compare `len()` against the YAML if a run seems short.

    Args:
        name: a group name or a task name.
        tasks_dir: optional extra directory to discover first.

    Returns:
        Individual task names, never group names.

    Raises:
        ValueError: if `name` is neither a registered task nor a registered group.
    """
    if tasks_dir is not None:
        discover_tasks(tasks_dir)
    return _core().expand_group(name)


def is_group(name: str) -> bool:
    """True if `name` is a registered group rather than an individual task."""
    return name in _core()._GROUP_REGISTRY


def load_questions(
    task_name: str | None = None,
    *,
    csv_path: str | None = None,
    video_folder: str = "",
    image_folder: str = "",
    hf_cache_dir: str | None = None,
    limit: int = -1,
    split_override: str | None = None,
    tasks_dir: str | None = None,
) -> tuple[list[Question], dict[str, Question]]:
    """Load a task (or CSV) into the project's unified question format.

    Delegates to `core.dataset_loader.load_dataset_as_questions` (`:245-460`). See
    `modirect.tasks.schema.Question` for the exact output shape — including the key
    literally named ``"false option"``, with a space.

    Behaviour worth knowing before you trust the output:

    * **`task_name` may be a group.** Core detects that (`:282-301`), loads every member,
      tags each question with ``source_task``, and returns the concatenation. `limit` is
      then applied **per member task, not to the total** (`:287-293`) — asking for
      ``limit=100`` on an 8-member group gives you 800 questions.
    * **`csv_path` wins over `task_name`** (`:265`): the CSV branch returns before the
      task is ever looked at. Do not pass both.
    * **`q_id` is not the dataset's id.** Core appends the row index: ``q_id =
      f"{doc[id_field]}_{idx}"`` (`:382-384`). This is deliberate — it makes `q_id` unique
      even when the source ids are not — but it means `q_id` will not match ids in the
      upstream dataset, and for a group the index restarts per member task, so uniqueness
      across a group relies on the id half.
    * **MCQ answers are auto-converted to letters** (`:400-407`): if the target is text
      longer than one character and matches a candidate, it becomes ``"A"``/``"B"``/... by
      candidate position. If it matches nothing, **the raw text passes through silently**.
    * **`video_folder` falls back to `$HF_DATASETS_CACHE`** when empty (`:411`).
    * `cache_dir` precedence is: `hf_cache_dir` arg > `$HF_DATASETS_CACHE` >
      the YAML's `dataset_kwargs.cache_dir` (`:321-325`); an empty-string `cache_dir` in
      the YAML is dropped first (`:318-319`).

    Args:
        task_name: registered task or group name. Required unless `csv_path` is given.
        csv_path: legacy CSV path; takes precedence over `task_name` entirely.
        video_folder: root for video paths; empty falls back to `$HF_DATASETS_CACHE`.
        image_folder: root for image paths.
        hf_cache_dir: HF datasets cache override; highest precedence.
        limit: max samples, `-1` for all. Per-member for groups (see above).
        split_override: split name overriding the YAML's `test_split` (default `"test"`).
        tasks_dir: optional extra directory to discover first.

    Returns:
        ``(questions, dataset_dict)`` where `questions` is a list of `Question` and
        `dataset_dict` maps ``q_id -> Question``. The dict is built by comprehension
        (`:457`), so on a `q_id` collision it is **shorter than the list** — that
        length mismatch is the cheapest available collision check.

    Raises:
        ValueError: if neither `task_name` nor `csv_path` is given (`:278-279`), or the
            task is unknown.
    """
    if tasks_dir is not None:
        discover_tasks(tasks_dir)
    return _core().load_dataset_as_questions(
        task_name=task_name,
        csv_path=csv_path,
        video_folder=video_folder,
        image_folder=image_folder,
        hf_cache_dir=hf_cache_dir,
        limit=limit,
        split_override=split_override,
    )
