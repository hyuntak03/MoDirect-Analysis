"""The two probe targets: direction (what the model SEES) and letter (what it must SAY).

WHY BOTH EXIST — the pair IS the measurement
--------------------------------------------
A direction probe asks "is Up/Down/Left/Right recoverable from this hidden state?".
A letter probe asks "is THIS SAMPLE'S correct option letter recoverable?". Candidates are
shuffled per sample, so the same direction maps to a different letter across samples and
the two targets are independent by construction: a representation can be perfect on one and
at chance on the other.

That independence is what makes the pair diagnostic. Baseline / obj_place / L27:

    direction probe   91.6%     the direction is THERE, and linearly readable
    letter probe      78.8%     it is not routed to an option letter
    MCQ accuracy      79%       tracks the LETTER probe, not the direction probe

The ~13pp direction->letter gap IS the binding gap, and MCQ accuracy tracking the letter
probe to within a point is why the LETTER probe is the real capability measure. On
shape_color both sit near 99% and the gap closes — so the OOD deficit is a binding deficit,
not a perception deficit. A direction probe alone would have reported obj_place as nearly
solved.

The Vanilla control sharpens it: its letter probe is at chance at EVERY layer (shape_color
L27: direction 88.7%, letter 24.7%). An un-finetuned model encodes direction perfectly well
and encodes no letter at all — fine-tuning teaches binding, not seeing.

Reference implementation: `analysis/letter_vs_direction_probing.py`
    :59-63    `load_letter_labels` — letters come from the MCQ JSON, NOT from the features
    :77-90    the qid join, ported to `join_letter_labels` below
    :105-106  both probes fit the SAME feature matrix, differing only in `y`

Use `modirect.probing.linear.LEGACY_LETTER` to reproduce those numbers: that script's
`gpu_probe:30` is neither of the two committed probe variants (see `linear.py`).

THE LABEL TRAP
--------------
`labels.npy` beside the features is ALREADY the direction label, despite being derived from
the answer letter. `extract_answer_features.py:138-148` (`resolve_answer`) reads
`line["answer"]` ("B"), indexes it into `line["candidates"]`, and stores the resulting TEXT
("Left"); `build_label_set:119-135` encodes the SORTED unique texts. The letter is discarded
at extraction time, so it must be re-joined from the source MCQ JSON by qid — there is no
recovering it from the feature directory.

The direction label space lives in `modirect.config.directions`, not here. It documents
three mutually incompatible orderings of the same four names, and `labels.npy` follows
`sorted()` == (down, left, right, up). This module deliberately defines no direction tuple
of its own — a fourth ordering is the last thing the repo needs.
"""

from __future__ import annotations

import ast
import json
import os
import string
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..config.directions import CHANCE_ACCURACY, DIRECTION_NAMES, canonical_order

__all__ = [
    "DIRECTIONS",
    "LETTERS",
    "ProbeTarget",
    "DIRECTION",
    "LETTER",
    "letter_target",
    "resolve_answer_text",
    "encode_labels",
    "load_letter_labels",
    "join_letter_labels",
]

#: The direction class names, RE-EXPORTED from `modirect.config.directions` — not redefined.
#: This is Ordering 1, `sorted()` == (down, left, right, up), i.e. the order the integers in
#: `labels.npy` index. Writing the "natural" (up, down, left, right) here instead would have
#: created a fourth ordering that silently mislabels all four classes; config documents why.
DIRECTIONS: tuple[str, ...] = DIRECTION_NAMES

#: The MCQ letter set for the R2R 4-way tasks. Earlier 8-way extractions used A..H
#: (`letter_vs_direction_probing.py:8`), which is why `letter_target()` derives the class
#: count from the data rather than assuming this tuple.
LETTERS: tuple[str, ...] = ("A", "B", "C", "D")


@dataclass(frozen=True)
class ProbeTarget:
    """A probe target: its name, class order, and what its collapse means.

    Attributes:
        name: "direction" or "letter".
        classes: pinned class order. Row i of `ProbeResult.weights` is `classes[i]`, so
            this must match the encoding of the `y` actually passed to the probe.
        chance: chance accuracy as a fraction. Both targets are 4-way => 0.25.
        interpretation: what a number on this target does and does not license.
    """

    name: str
    classes: tuple[str, ...]
    chance: float
    interpretation: str

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return len(self.classes)


#: Direction target. `classes` is Ordering 1 — the order `labels.npy` integers index — so
#: `encode_labels(names, DIRECTION.classes)` reproduces the on-disk codes exactly.
DIRECTION = ProbeTarget(
    name="direction",
    classes=tuple(str(d) for d in canonical_order()),
    chance=CHANCE_ACCURACY,
    interpretation=(
        "Is the motion direction linearly recoverable? Stays ~92% on OOD at L21 even where "
        "the model answers wrongly, so a high score here does NOT imply capability."
    ),
)

#: Letter target. Requires the 4-way cache plus the source MCQ JSON; see `join_letter_labels`.
LETTER = ProbeTarget(
    name="letter",
    classes=LETTERS,
    chance=1.0 / len(LETTERS),
    interpretation=(
        "Is this sample's correct option letter recoverable? Tracks MCQ accuracy to within "
        "~1pp, so this is the capability measure. At chance for every Vanilla layer; jumps "
        "from chance to ~70% across a single layer (L15->L16) for Delta."
    ),
)


def resolve_answer_text(line: Mapping) -> str:
    """Map an MCQ record's answer to its candidate TEXT (i.e. the direction word).

    Faithful port of `extract_answer_features.py:138-148`. A single-letter answer is an
    option index into `candidates`; anything else passes through. `candidates` is tolerated
    as a list or as its `repr` string, because the HF datasets round-trip stringifies it —
    hence the `ast.literal_eval` at :143-144.

    Args:
        line: MCQ record with "answer" and optionally "candidates".

    Returns:
        The resolved answer text, stripped. Falls through to the raw answer when the letter
        index is out of range, matching the legacy behaviour rather than raising — so the
        label set built here is the one the published features were built with.
    """
    answer = str(line["answer"]).strip()
    if len(answer) == 1 and answer.upper() in string.ascii_uppercase:
        candidates = line.get("candidates", [])
        if isinstance(candidates, str):
            candidates = ast.literal_eval(candidates)
        idx = ord(answer.upper()) - ord("A")
        if idx < len(candidates):
            return str(candidates[idx]).strip()
    return answer


def encode_labels(
    labels: Sequence[str], classes: Sequence[str] | None = None
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode string labels to integer codes under a pinned class order.

    Args:
        labels: length-N string labels.
        classes: class order. Defaults to `sorted(unique(labels))` — what BOTH legacy paths
            do (`extract_answer_features.py:133` sorts; `letter_vs_direction_probing.py:92`
            uses sklearn's LabelEncoder, which also sorts). Keeping that default is what
            makes codes produced here agree with the codes baked into `labels.npy`.

    Returns:
        (codes, classes) with `codes[i] == classes.index(labels[i])`.

    Raises:
        ValueError: if a label falls outside `classes`. Dropping it silently would shift the
            label vector against the feature matrix by a row and corrupt every number after.

    Note:
        Matching is EXACT, including case. The MCQ JSON carries capitalised direction text
        ("Up") while `DIRECTION.classes` is lowercase, so
        `encode_labels(["Up", ...], DIRECTION.classes)` raises rather than guessing —
        normalise first with `modirect.config.directions.as_strs`. Case only happens to be
        harmless for the 4 direction names (`sorted()` gives the same order either way, a
        coincidence config.directions flags explicitly); relying on that silently is how a
        future 8-way label set would mislabel.
    """
    if classes is None:
        classes = sorted({str(v) for v in labels})
    classes = tuple(str(c) for c in classes)
    lookup = {c: i for i, c in enumerate(classes)}
    try:
        codes = np.array([lookup[str(v)] for v in labels], dtype=np.int64)
    except KeyError as exc:
        bad = exc.args[0]
        hint = ""
        if isinstance(bad, str) and bad.lower() in {c.lower() for c in classes}:
            hint = (
                " — it differs only in case; normalise with "
                "modirect.config.directions.as_strs, or pass matching `classes`"
            )
        raise ValueError(f"label {bad!r} is not in classes {classes}{hint}") from None
    return codes, classes


def letter_target(letters: Iterable[str]) -> ProbeTarget:
    """Build a `ProbeTarget` for the letter set actually present in the data.

    Use instead of the `LETTER` constant when probing an 8-way extraction, where the letters
    run A..H and chance is 12.5% rather than 25%
    (`letter_vs_direction_probing.py:8`, `:94` derives `n_letter_classes` the same way).
    """
    classes = tuple(sorted({str(x).strip().upper() for x in letters}))
    return ProbeTarget(
        name="letter",
        classes=classes,
        chance=1.0 / len(classes),
        interpretation=LETTER.interpretation,
    )


def load_letter_labels(mcq_json_path: str | os.PathLike) -> dict[int, str]:
    """Load `{sample_id: letter}` from an MCQ task JSON.

    Port of `analysis/letter_vs_direction_probing.py:59-63`. The letter exists only here —
    see the module docstring for why the feature cache cannot supply it.

    Args:
        mcq_json_path: e.g. `{mcq_json_root}/obj_place.json`. The source repo's root is
            `.../synthetic_testbed/Testbed/huggingface/R2R_4way_1500` (`:23`).

    Returns:
        `{int id: uppercased letter}`.

    Raises:
        FileNotFoundError: if the JSON is absent.
    """
    with open(mcq_json_path) as f:
        data = json.load(f)
    return {int(rec["id"]): str(rec["answer"]).strip().upper() for rec in data}


def join_letter_labels(
    qids: Iterable, letter_map: Mapping[int, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Align letter labels to a feature matrix's row order via `qids.npy`.

    Port of `letter_vs_direction_probing.py:83-90`. The join key is the LEADING integer of
    the qid, because qids are written `f"{sample_id}_{direction}"`
    (`02_extract_avg_hidden.py:57`): `int(str(qid).split("_")[0])`.

    Rows whose id is missing from `letter_map` are dropped, and the surviving indices are
    RETURNED rather than assumed to be all of them. Callers must index both the features and
    the direction labels with them:

        letters, valid_idx = join_letter_labels(qids, letter_map)
        X, y_dir = X[valid_idx], y_dir[valid_idx]
        y_letter, classes = encode_labels(letters)

    Skipping that is the misalignment this signature is shaped to prevent — the legacy
    script indexes both (`:101-102`), and a partial join is NORMAL, because the extractor
    drops samples whose answer falls outside the label set
    (`extract_answer_features.py:207-209`).

    Args:
        qids: `qids.npy` contents, row-aligned with the feature matrix. Get them from
            `modirect.io.feature_store.qids_for`, which knows which stages lack the file.
        letter_map: from `load_letter_labels`.

    Returns:
        `(letters, valid_idx)` — letter strings for surviving rows, and their row indices.
    """
    letters: list[str] = []
    valid_idx: list[int] = []
    for i, qid in enumerate(qids):
        sid = int(str(qid).split("_")[0])
        if sid in letter_map:
            letters.append(letter_map[sid])
            valid_idx.append(i)
    return np.array(letters), np.array(valid_idx, dtype=np.int64)
