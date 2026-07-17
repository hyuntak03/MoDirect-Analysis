"""Magnitude interventions on the direction axis, and the hooks that apply them live.

`operators` is pure numpy/torch-agnostic arithmetic. `hooks` is torch-free *at module
level* by design — it defers `import torch` into `LastTokenIntervention.__init__`
(`hooks.py:88`), so this re-export stays importable on a host with no model runtime and
the operator tests can run without torch.
"""

from __future__ import annotations

from .hooks import LastTokenIntervention, last_token_hook
from .operators import CONDITIONS, apply_condition, project_on_axis

__all__ = [
    "CONDITIONS",
    "LastTokenIntervention",
    "apply_condition",
    "last_token_hook",
    "project_on_axis",
]
