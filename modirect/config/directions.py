"""The Direction label space — and the three incompatible orderings it is stored under.

READ THIS BEFORE MAPPING AN INTEGER LABEL TO A NAME. The source repo contains **three
different orderings** of the same four directions. They are not typos and they are not
reconcilable into one list: each is load-bearing where it sits, and they disagree. This
module records all three and provides one — and only one — integer<->string mapping,
the one that actually indexes the stored data.

--------------------------------------------------------------------------------------
ORDERING 1 — ``sorted()``  ==  ["down", "left", "right", "up"]   <-- INTEGER LABELS
--------------------------------------------------------------------------------------
This is what `labels.npy` contains, and therefore the only ordering that matters when
you index a cached feature array.

    linear_probing/extract_vision_features.py:110-124  build_label_set()
        unique_answers.add(ans)          # ans = the candidate TEXT, not the letter
        label_list = sorted(unique_answers)
        answer_to_idx = {a: i for i, a in enumerate(label_list)}

    linear_probing/extract_vision_features.py:449      label_idx = answer_to_idx[answer]
    linear_probing/extract_vision_features.py:231      np.array(labels, dtype=np.int64) -> labels.npy

`answer` is resolved from the MCQ letter back to the candidate string by
`resolve_answer` (extract_vision_features.py:127-136), so the set being sorted is the
candidate label text: ``{"Up", "Down", "Left", "Right"}``
(`tasks/vlm_direction_testbed/utils.py:39  DIRECTION_LABELS_4WAY`).

    sorted(["Up", "Down", "Left", "Right"]) == ["Down", "Left", "Right", "Up"]

Hence, and this is the mapping this module exposes:

    0 -> down    1 -> left    2 -> right    3 -> up

Conveniently this is **capitalisation-invariant** — `sorted(["up","down","left","right"])`
is `["down","left","right","up"]` too — so the mapping holds whether a given script
lower-cased the labels or not. That is luck, not design; do not rely on it elsewhere.

This is the ordering consumed by every integer-label site, e.g.

    analysis/task_invariance/axis_layer_cos.py:30   [h[y == d].mean(0) - g for d in range(4)]

`range(4)` there is a bare index into `labels.npy`. It carries no names, which is exactly
why the mismatch below has gone unnoticed.

--------------------------------------------------------------------------------------
ORDERING 2 — clockwise  ==  ["up", "right", "down", "left"]      <-- FACTORIAL / DIRS
--------------------------------------------------------------------------------------
    analysis/task_invariance/mechanism_diagnosis.py:60   DIRS = ["up","right","down","left"]
    analysis/task_invariance/vision_intervention.py:80   (same)
    analysis/task_invariance/vision_amp.py:79            (same)
    analysis/task_invariance/vision_intervention_v2.py:80 (same)

These operate on the factorial dataset, whose `directions` field is an array of
**strings**, not integers (`mechanism_diagnosis.py:61  H[dirs == d]` compares against
`"up"`). So `DIRS` is an *iteration order over string keys*, never an index map — which
is why it has been safe so far. It matches the clockwise candidate ordering of factorial
variant 0 (`CLAUDE.md`: ``Variant 0: [Up, Right, Down, Left]``).

    !! THE CONFLICT !!  `DIRS` and `labels.npy` disagree on every single position:

        index      0        1        2        3
        Ordering 1 down     left     right    up        (labels.npy — integers)
        Ordering 2 up       right    down     left      (DIRS — factorial strings)

    Zipping `DIRS` against `range(4)`, or feeding an integer from `labels.npy` into
    `DIRS[i]`, silently mislabels **all four** classes. Nothing raises: both are
    length-4 lists of plausible strings. Use `to_str`/`to_int` below, never `DIRS[i]`.

--------------------------------------------------------------------------------------
ORDERING 3 — semantic pairs  ==  ["up", "down", "left", "right"]  <-- REPORTING ONLY
--------------------------------------------------------------------------------------
    tasks/vlm_direction_testbed/utils.py:38  DIRECTION_CLASSES_4WAY = ["up","down","left","right"]
    tasks/vlm_direction_testbed/utils.py:39  DIRECTION_LABELS_4WAY  = ["Up","Down","Left","Right"]

Used only to build confusion-matrix axes and per-direction accuracy rows
(`utils.py:169-177`, `utils.py:252`), where it builds its **own** `class_to_idx` locally
and never crosses paths with `labels.npy`. It groups the opposing pairs (up/down,
left/right) adjacently, which is the natural reading order for a confusion matrix. Also
the prefix of the 8-way `DIRECTION_CLASSES` (utils.py:17-26), whose diagonal classes
("up-left", ...) this 4-way project does not use.

--------------------------------------------------------------------------------------
WHY NOT RECONCILE
--------------------------------------------------------------------------------------
Ordering 1 is a fact about bytes already on disk under `Paths.feature_root` — it cannot
be changed without re-extracting every feature. Ordering 2 is a fact about the factorial
`.npz` files. Ordering 3 is cosmetic. `canonical_order()` therefore returns Ordering 1:
it is the only one that indexes stored data, so it is the only one that can be wrong in
a way that corrupts a result. The others are exposed verbatim, under names that say
where they came from, so a reader can see the disagreement rather than trip over it.
"""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = [
    "Direction",
    "DIRECTION_NAMES",
    "N_DIRECTIONS",
    "CHANCE_ACCURACY",
    "canonical_order",
    "to_str",
    "to_int",
    "to_label",
    "DIRS_FACTORIAL_CLOCKWISE",
    "DIRECTION_CLASSES_4WAY",
    "DIRECTION_LABELS_4WAY",
]

try:  # py3.11+ gives str-mixin enums a sane __str__/__format__; py3.10 does not.
    from enum import StrEnum as _StrBase
except ImportError:  # pragma: no cover - py3.10
    from enum import Enum

    class _StrBase(str, Enum):  # type: ignore[no-redef]
        def __str__(self) -> str:
            return str(self.value)


class Direction(_StrBase):
    """A motion direction, valued as its canonical lowercase name.

    Members are declared in **Ordering 1** (`sorted()`), so `list(Direction)` matches
    the integer labels in `labels.npy` position-for-position. Do not reorder these
    declarations: `Direction.DOWN` must stay first because `labels.npy` says 0 is down.

    Being a str subclass, a member is accepted anywhere a name is expected — including
    as a `classes=` entry for `modirect.concepts.extract_concept_vectors`, and as a key
    compared against the factorial `.npz` string arrays (`"up" == Direction.UP`).
    """

    DOWN = "down"    # label 0
    LEFT = "left"    # label 1
    RIGHT = "right"  # label 2
    UP = "up"        # label 3

    @property
    def index(self) -> int:
        """The integer this direction has in `labels.npy`."""
        return DIRECTION_NAMES.index(self.value)

    @property
    def label(self) -> str:
        """The capitalised MCQ candidate text, e.g. "Up" — `DIRECTION_LABELS_4WAY`."""
        return self.value.capitalize()

    @property
    def opposite(self) -> "Direction":
        """The 180-degrees-opposed direction.

        Used by flip-style counterfactual constructions (Section K / Q-N2:
        ``h - Δ(curr) + Δ(flip)``), where the natural contrast is the opposite class.
        """
        return _OPPOSITES[self]


#: Ordering 1, as plain strings. `DIRECTION_NAMES[i]` is the name of integer label `i`.
DIRECTION_NAMES: tuple[str, ...] = tuple(d.value for d in Direction)

#: 4-way task: `chance = 1/4`. CLAUDE.md "Task: R2R 4-way 1500".
N_DIRECTIONS: int = len(DIRECTION_NAMES)
CHANCE_ACCURACY: float = 1.0 / N_DIRECTIONS

_OPPOSITES: dict[Direction, Direction] = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

#: Ordering 2 — verbatim `DIRS` from `mechanism_diagnosis.py:60`. An iteration order over
#: the factorial `.npz` STRING labels. **Not** an index map; see the module docstring.
DIRS_FACTORIAL_CLOCKWISE: tuple[str, ...] = ("up", "right", "down", "left")

#: Ordering 3 — verbatim from `tasks/vlm_direction_testbed/utils.py:38-39`. Reporting only.
DIRECTION_CLASSES_4WAY: tuple[str, ...] = ("up", "down", "left", "right")
DIRECTION_LABELS_4WAY: tuple[str, ...] = ("Up", "Down", "Left", "Right")


def canonical_order() -> tuple[Direction, ...]:
    """The class order that matches `labels.npy`: (DOWN, LEFT, RIGHT, UP).

    Pass this as `classes=` to `modirect.concepts.extract_concept_vectors`, which warns
    that "Δ̂ sign conventions depend on" a pinned order and refuses to guess.

    Returns:
        The four directions in integer-label order, i.e. `canonical_order()[i]` is the
        direction of label `i`.
    """
    return tuple(Direction)


def to_str(label: int | str | Direction) -> str:
    """Map an integer `labels.npy` value (or a name) to its canonical lowercase name.

    Args:
        label: An int in `range(4)` — interpreted per Ordering 1 — or an already-string
            name in any capitalisation ("Up", "up", `Direction.UP`).

    Returns:
        The canonical lowercase name.

    Raises:
        ValueError: on an out-of-range int or an unrecognised name. Notably, the 8-way
            diagonal classes ("up-left", ...) raise: they exist in
            `utils.py:17-26 DIRECTION_CLASSES` but not in this 4-way label space.

    Example:
        >>> to_str(0), to_str(3)
        ('down', 'up')
    """
    if isinstance(label, Direction):
        return label.value
    if isinstance(label, bool):  # bool is an int subclass; 0/1 here is a bug, not a label
        raise ValueError(f"expected a direction label, got bool {label!r}")
    if isinstance(label, (int,)):
        if not 0 <= label < N_DIRECTIONS:
            raise ValueError(
                f"direction label {label} out of range for a {N_DIRECTIONS}-way task; "
                f"valid: 0..{N_DIRECTIONS - 1} == {list(DIRECTION_NAMES)}")
        return DIRECTION_NAMES[label]
    name = str(label).strip().lower()
    if name not in DIRECTION_NAMES:
        raise ValueError(
            f"unknown direction {label!r}; valid: {list(DIRECTION_NAMES)}. "
            "(8-way diagonals like 'up-left' are not part of the 4-way label space.)")
    return name


def to_int(label: int | str | Direction) -> int:
    """Map a direction name to its `labels.npy` integer (Ordering 1).

    The inverse of `to_str`. Use this instead of `DIRS.index(...)` — `DIRS` is Ordering 2
    and would return a different number for every class.

    Args:
        label: A name in any capitalisation, a `Direction`, or an int (validated and
            returned unchanged).

    Returns:
        The integer label: down=0, left=1, right=2, up=3.

    Raises:
        ValueError: as `to_str`.

    Example:
        >>> to_int("Up"), to_int("down")
        (3, 0)
    """
    return DIRECTION_NAMES.index(to_str(label))


def to_label(label: int | str | Direction) -> str:
    """Map to the capitalised MCQ candidate text ("Up"), as shown to the model.

    This is the form `build_label_set` sorted (`extract_vision_features.py:122`) and the
    form rendered into the prompt (`utils.py:103`).
    """
    return to_str(label).capitalize()


def as_ints(labels: Iterable[int | str | Direction]) -> list[int]:
    """Vectorised `to_int` — normalise a mixed/str label column to integers.

    Handy when joining the factorial dataset's string `directions` field against the
    integer `labels.npy` convention; doing that join by hand is exactly the Ordering
    1-vs-2 trap described in the module docstring.
    """
    return [to_int(x) for x in labels]


def as_strs(labels: Iterable[int | str | Direction]) -> list[str]:
    """Vectorised `to_str` — normalise an integer label column to canonical names."""
    return [to_str(x) for x in labels]


def reorder_to(values: Sequence, order: Sequence[str]) -> list:
    """Re-index a canonically-ordered sequence into some other ordering.

    Args:
        values: A length-4 sequence indexed per `canonical_order()` (i.e. by
            `labels.npy` integers).
        order: Target names, e.g. `DIRS_FACTORIAL_CLOCKWISE` to hand data to a factorial
            script, or `DIRECTION_CLASSES_4WAY` to lay out a confusion matrix.

    Returns:
        `values` permuted so that position `i` corresponds to `order[i]`.

    Raises:
        ValueError: if `values` is not length-4 or `order` is not a permutation of the
            four direction names.

    Example:
        >>> reorder_to(["d", "l", "r", "u"], DIRS_FACTORIAL_CLOCKWISE)
        ['u', 'r', 'd', 'l']
    """
    if len(values) != N_DIRECTIONS:
        raise ValueError(f"expected {N_DIRECTIONS} values, got {len(values)}")
    names = [to_str(o) for o in order]
    if sorted(names) != sorted(DIRECTION_NAMES):
        raise ValueError(f"{list(order)} is not a permutation of {list(DIRECTION_NAMES)}")
    return [values[to_int(n)] for n in names]
