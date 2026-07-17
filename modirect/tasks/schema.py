"""The unified question format produced by `core/dataset_loader.py`.

Every task in this project — HuggingFace or CSV, video or image, 4-way or 8-way MCQ — is
normalised into one dict shape by `load_dataset_as_questions`. This module is the
written-down version of that contract, which otherwise exists only as a dict literal at
`core/dataset_loader.py:441-448` plus a passthrough loop at `:450-453`::

    q = {
        "q_id": q_id,
        "question": question_text,
        "answer": answer_text,
        "img_id": "" if is_video else vis_path,
        "video": vis_path if is_video else "",
        "false option": false_option,
    }
    for k, v in doc.items():                       # passthrough
        if k not in q:
            q[k] = v if not isinstance(v, (list, dict)) else str(v)

`Question` is a `TypedDict`, not a dataclass, on purpose: core emits plain dicts and 40+
scripts index them with `q["question"]`. A TypedDict is a type-checker-only view over
exactly those dicts — zero runtime cost, no conversion step, and the existing call sites
type-check unchanged.

It is declared with TypedDict's **functional syntax** because the class syntax cannot
express the key ``"false option"`` — that is a space, not an underscore, and so not a
valid Python identifier. See `FALSE_OPTION_KEY`.

THE FIELDS (all cites are `core/dataset_loader.py`)
---------------------------------------------------

``q_id`` : str
    ``f"{doc[id_field]}_{idx}"`` (`:382-384`). **Not the upstream dataset id** — the row
    index is appended to guarantee uniqueness where the source ids do not. `id_field` is
    ``field_map["question_id"]`` (default ``"question_id"``), falling back to the index
    itself when the field is absent (`:383`). For a group load the index restarts at 0
    per member task, so cross-member uniqueness rests on the id half alone.

``question`` : str
    Prompt text. `doc_to_text(doc, task_kwargs)` when the YAML made it a `!function`
    callable, else ``str(doc[doc_to_text or "question"])`` (`:387-390`).

``answer`` : str
    Gold answer. **For MCQ this is normally an option letter.** Core auto-converts a text
    target to ``"A"``/``"B"``/… by matching it against the candidates and taking the
    position (`:400-407`). The conversion fires only when the target is longer than one
    character AND an exact (whitespace-stripped) match exists; **otherwise the raw text
    passes through silently**. A mismatched candidate list therefore yields prose here,
    where every downstream letter comparison expects a letter.

``img_id`` : str
    Image path, or ``""`` when the sample is a video (`:445`).

``video`` : str
    Video path, or ``""`` when the sample is an image (`:446`).
    `img_id` and `video` are mutually exclusive: core computes a single `vis_path` and
    routes it to exactly one of them via `is_video` (`:438-439`), which is true iff the
    doc has a non-empty video field. The visual path is thus ``q["video"] or q["img_id"]``.

``"false option"`` : str
    The distractor, from `doc_to_false_option(doc, task_kwargs)`; ``""`` when the task
    declares no such function (`:433-435`, `:447`). **Note the space in the key** — use
    `FALSE_OPTION_KEY`.

``source_task`` : str
    **Only present for group loads** (`:296-297`), naming the member task a sample came
    from. Absent on single-task loads — always `.get` it, never index it.

*plus passthrough*
    Every field of the original dataset row whose name does not collide with the six keys
    above is copied in verbatim (`:450-453`) — so the real key set is task-dependent and
    the TypedDict is `total=False` and deliberately open. One trap: passthrough values
    that were **lists or dicts are stringified** (``str(v)``). `candidates` arrives as the
    string ``"['Up', 'Down', 'Left', 'Right']"``, NOT as a list. Code needing the real
    list must `ast.literal_eval` it or go back to the source doc.
"""

from __future__ import annotations

from typing import TypedDict

__all__ = ["Question", "FALSE_OPTION_KEY", "REQUIRED_KEYS"]

#: The literal key for the distractor field — **note the space**.
#:
#: `core/dataset_loader.py:447` writes ``"false option": false_option``. Because it is not
#: a valid identifier it cannot be an attribute or a `**kwargs` name, and it is easy to
#: mistype as ``"false_option"`` — which raises KeyError at best and, in
#: ``.get("false_option", "")`` form, silently returns ``""`` forever. Index with this
#: constant instead: ``q[FALSE_OPTION_KEY]``.
FALSE_OPTION_KEY = "false option"

#: The six keys core always writes (`core/dataset_loader.py:441-448`). Everything else in
#: a Question is passthrough from the source row. Useful for separating the contract from
#: the task-specific extras::
#:
#:     extras = {k: v for k, v in q.items() if k not in REQUIRED_KEYS}
REQUIRED_KEYS: tuple[str, ...] = (
    "q_id",
    "question",
    "answer",
    "img_id",
    "video",
    FALSE_OPTION_KEY,
)

Question = TypedDict(
    "Question",
    {
        "q_id": str,
        "question": str,
        "answer": str,
        "img_id": str,
        "video": str,
        FALSE_OPTION_KEY: str,
        "source_task": str,
    },
    total=False,
)
Question.__doc__ = """One normalised sample from `core.dataset_loader`.

Guaranteed keys: ``q_id``, ``question``, ``answer``, ``img_id``, ``video``,
``"false option"`` (space, not underscore — see `FALSE_OPTION_KEY`). ``source_task`` is
present only for group loads, and the original dataset row's remaining fields are passed
through, so the key set is open — hence ``total=False``.

Field-by-field documentation, with source cites, is in the `modirect.tasks.schema` module
docstring.
"""
