"""Task registry — a typed spine over `core/dataset_loader.py`.

    from modirect.tasks import list_tasks, load_questions

    list_tasks()                                        # every registered task + group
    qs, by_id = load_questions("vlm_direction_testbed_R2R_4way_obj_place", limit=1500)
    qs[0]["answer"], qs[0]["false option"]              # see modirect.tasks.schema

!!  NAME COLLISION — READ THIS  !!
=================================

There are TWO things called "tasks" in this repo and they are unrelated:

    <repo>/tasks/            the YAML DATA directory. Task definitions + their utils.py.
                             Not a Python package, never imported. MUST remain a sibling
                             of <repo>/core/, because core/dataset_loader.py:119 computes
                             its default location as
                                 dirname(dirname(core/dataset_loader.py)) / "tasks"
                             and scans it at import time (:467).

    <repo>/modirect/tasks/   THIS package. Python code that wraps core's registry.

`core.dataset_loader` finds the data directory by **filesystem path**, never by import, so
the two do not interfere through normal use. The danger is `sys.path`: if `<repo>/modirect`
were ever added to `sys.path`, then ``import tasks`` would resolve to *this package*
instead of the data directory's namespace. Two rules keep that safe, and both must hold:

  1. **Nothing in `modirect/` may add `modirect/` itself to `sys.path`.** Scripts add the
     REPO ROOT (`sys.path.insert(0, _PROJECT_ROOT)`) and import `modirect.tasks` — the
     fully qualified name. Never `sys.path.insert(0, ".../modirect")`.
  2. **Never `import tasks`** (unqualified) anywhere in this package. Use
     `from modirect.tasks import ...` or an explicit relative import.

Getting this wrong does not raise. `discover_tasks` on a missing/wrong directory just
prints a warning and leaves a **silently empty registry** (core/dataset_loader.py:133-135)
— you learn about it as a puzzling `ValueError: Unknown task` much later. If `list_tasks()`
returns `[]`, suspect the directory before you suspect the YAML.

See `modirect.tasks.registry` for the five YAML rules (`!function` resolution, `include:`
inheritance, task-vs-group registration, `_`-prefixed template skipping, import-time
discovery) and `modirect.tasks.schema` for the `Question` format.
"""

from __future__ import annotations

from .registry import (
    discover_tasks,
    expand_group,
    get_task_config,
    is_group,
    list_tasks,
    load_questions,
)
from .schema import FALSE_OPTION_KEY, REQUIRED_KEYS, Question

__all__ = [
    "list_tasks",
    "get_task_config",
    "expand_group",
    "is_group",
    "discover_tasks",
    "load_questions",
    "Question",
    "FALSE_OPTION_KEY",
    "REQUIRED_KEYS",
]
