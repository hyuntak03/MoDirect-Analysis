"""One linear probe, with the legacy hyperparameters made explicit instead of implicit.

WHY THIS MODULE EXISTS — the comparability defect
-------------------------------------------------
The project shipped TWO independent probe implementations, and the published numbers in
`CLAUDE.md` sections A and B were produced by different ones. They do not use the same
hyperparameters, so the vision-pipeline table and the answer-token table were never
strictly comparable. Verified line by line against the source repo:

    quantity        linear_probing/linear_probe.py     linear_probing_per_layer/linear_probe.py
    ------------    -------------------------------    ----------------------------------------
    test_ratio      0.3            (:279)              0.2               (:167)
    epochs          50             (:281)              100               (:169)
    split           torch.randperm (:59-63)            sklearn stratified(:104-107)
                    NOT stratified                     stratify=labels
    std guard       std[std<1e-8] = 1.0 (:51)          std = X.std(0) + 1e-8 (:100)
    batch_size      64             (:284)              64                (:172)
    returns         PERCENT, acc*100    (:110)         FRACTION           (:70)

Both brief claims check out. Two honest refinements the brief did not state:

  * The FRACTION/PERCENT split is an INNER-function difference only. `linear_probe.py:110`
    returns `train_acc*100, test_acc*100`; `linear_probing_per_layer/linear_probe.py:70`
    returns bare fractions but its caller multiplies at `:118-119`. Both CSVs are therefore
    in percent — the units defect never reached the published tables. It is still a real
    API hazard, which is why `ProbeResult.accuracy` here is a FRACTION, always, with
    `.accuracy_pct` as the explicit view.

  * `legacy_answer` is a misleading name for what actually produced the Section H
    answer-token numbers. Nothing in the repo invokes
    `linear_probing_per_layer/linear_probe.py` (grep: only its own docstring references it).
    The Section H direction-vs-letter tables came from
    `analysis/letter_vs_direction_probing.py:30` (`gpu_probe`), which is a THIRD variant:
    vision-style hyperparameters (test_ratio 0.3, 50 epochs, non-stratified randperm,
    std<1e-8 guard) but batch_size 256, not 64. It is exposed here as `legacy_letter`.
    Section B's answer-token numbers ran through the same vision-style path.

The shared defect worth knowing about
-------------------------------------
BOTH legacy variants standardise over the FULL array before splitting
(`linear_probe.py:48-52` then split at `:56-63`; per_layer `:99-101` then `:104`). Test-set
mean/std therefore leak into training. The effect is small at N=6000 with D≤3584, but it is
a real leak, and it is why `canonical` fits the scaler on train only. `canonical` numbers
will not reproduce the published tables exactly; that is intentional and is the point of
keeping the legacy presets addressable.

torch is imported lazily inside `train_linear_probe` — this module must stay importable on
hosts without torch (see the package contract in `modirect/__init__.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

__all__ = [
    "ProbeConfig",
    "ProbeResult",
    "train_linear_probe",
    "LEGACY_VISION",
    "LEGACY_ANSWER",
    "LEGACY_LETTER",
    "CANONICAL",
    "PRESETS",
    "get_preset",
]

SplitKind = Literal["randperm", "stratified"]
StdGuard = Literal["clamp", "epsilon"]


@dataclass(frozen=True)
class ProbeConfig:
    """Hyperparameters for one linear probe.

    Every field corresponds to a concrete divergence between the two legacy scripts, so
    that "which probe produced this number" is answerable from the config alone rather
    than from which directory the script lived in.

    Attributes:
        test_ratio: fraction held out. 0.3 vision / 0.2 answer.
        epochs: full passes over the training split. 50 vision / 100 answer.
        lr: AdamW learning rate. 1e-3 in every legacy variant.
        weight_decay: AdamW weight decay. 1e-2 in every legacy variant.
        batch_size: 64 in both legacy probe scripts; 256 in the letter/direction analysis.
        seed: seeds the split, the parameter init, and the epoch shuffles.
        split: "randperm" reproduces `linear_probe.py:59-63` (NOT class-balanced — with an
            unbalanced label set the test split's class ratios drift). "stratified"
            reproduces `linear_probing_per_layer/linear_probe.py:104-107`.
        std_guard: "clamp" is `std[std < 1e-8] = 1.0` (:51) — dead dimensions are passed
            through at their raw scale (which is ~0, so they stay dead). "epsilon" is
            `std + 1e-8` (:100) — a dead dimension is divided by ~1e-8 and its float noise
            is amplified to O(1). "clamp" is the defensible one; "epsilon" is retained only
            to reproduce legacy numbers.
        standardize_on: "all" reproduces the legacy leak (fit the scaler on train+test);
            "train" fits on the training split only. See module docstring.
        device: "auto" picks cuda when available, else cpu.
    """

    test_ratio: float = 0.2
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    seed: int = 42
    split: SplitKind = "stratified"
    std_guard: StdGuard = "clamp"
    standardize_on: Literal["all", "train"] = "train"
    device: str = "auto"

    def replace(self, **changes: Any) -> "ProbeConfig":
        """Return a copy with `changes` applied — e.g. `LEGACY_VISION.replace(seed=1)`."""
        return replace(self, **changes)


#: Reproduces `linear_probing/linear_probe.py` — the canonical 287-line vision probe that
#: produced the Section A vision-pipeline tables.
LEGACY_VISION = ProbeConfig(
    test_ratio=0.3,
    epochs=50,
    batch_size=64,
    split="randperm",
    std_guard="clamp",
    standardize_on="all",
)

#: Reproduces `linear_probing_per_layer/linear_probe.py` — the 175-line variant. NOTE: no
#: script in the source repo invokes it; see the module docstring before attributing any
#: published answer-token number to this preset.
LEGACY_ANSWER = ProbeConfig(
    test_ratio=0.2,
    epochs=100,
    batch_size=64,
    split="stratified",
    std_guard="epsilon",
    standardize_on="all",
)

#: Reproduces `analysis/letter_vs_direction_probing.py:30` (`gpu_probe`) — what ACTUALLY
#: produced the Section H direction-vs-letter tables. Vision-style, but batch_size 256.
LEGACY_LETTER = ProbeConfig(
    test_ratio=0.3,
    epochs=50,
    batch_size=256,
    split="randperm",
    std_guard="clamp",
    standardize_on="all",
)

#: What new work should use: stratified (label balance is never assumed), the safe std
#: guard, and no train/test leakage. Deliberately NOT equal to either legacy preset, so
#: numbers produced with it must not be pasted into the legacy tables.
CANONICAL = ProbeConfig(
    test_ratio=0.2,
    epochs=50,
    batch_size=256,
    split="stratified",
    std_guard="clamp",
    standardize_on="train",
)

PRESETS: dict[str, ProbeConfig] = {
    "legacy_vision": LEGACY_VISION,
    "legacy_answer": LEGACY_ANSWER,
    "legacy_letter": LEGACY_LETTER,
    "canonical": CANONICAL,
}


def get_preset(name: str) -> ProbeConfig:
    """Look up a preset by name.

    Raises:
        KeyError: with the valid names listed, since a typo would otherwise silently fall
            back to a default and quietly change every number in a table.
    """
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}") from None


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one probe fit.

    Attributes:
        accuracy: test accuracy as a FRACTION in [0, 1]. Always a fraction — the legacy
            scripts disagreed on this (see module docstring), so the ambiguity is resolved
            once, here.
        train_accuracy: training-split accuracy, as a fraction. A large gap to `accuracy`
            means the probe memorised; the 200-sample runs that produced the retracted
            "last token >> vision token" claim (CLAUDE.md observation 6) were exactly this.
        weights: (num_classes, D) probe weight matrix, on CPU as numpy. This is what
            `modirect.geometry` reads to compute the direction axis
            v_UD = W[Up] − W[Down] used in the axis-alignment analysis (Section F).
        bias: (num_classes,) probe bias.
        classes: pinned class order; row i of `weights` is `classes[i]`.
        n_train: training sample count.
        n_test: test sample count.
        config: the exact config used, carried so a result is self-describing.
    """

    accuracy: float
    train_accuracy: float
    weights: np.ndarray
    bias: np.ndarray
    classes: tuple[Any, ...]
    n_train: int
    n_test: int
    config: ProbeConfig

    @property
    def accuracy_pct(self) -> float:
        """Test accuracy in percent — the unit the published tables are written in."""
        return self.accuracy * 100.0

    @property
    def chance(self) -> float:
        """Chance accuracy as a fraction (1/num_classes): 0.25 for the 4-way targets."""
        return 1.0 / len(self.classes)


def _resolve_device(name: str) -> "torch.device":
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _split_indices(
    labels: np.ndarray, config: ProbeConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Produce (train_idx, test_idx) matching the legacy split semantics exactly."""
    n = len(labels)
    n_test = max(1, int(n * config.test_ratio))

    if config.split == "stratified":
        # sklearn is the reference (per_layer:104-107). Reimplemented with numpy so that
        # this module keeps its "numpy only" dependency contract; sklearn's stratified
        # split is a per-class proportional draw, which is what this reproduces.
        rng = np.random.default_rng(config.seed)
        test_parts: list[np.ndarray] = []
        train_parts: list[np.ndarray] = []
        for cls in np.unique(labels):
            idx = np.flatnonzero(labels == cls)
            rng.shuffle(idx)
            k = max(1, int(round(len(idx) * config.test_ratio)))
            test_parts.append(idx[:k])
            train_parts.append(idx[k:])
        return np.concatenate(train_parts), np.concatenate(test_parts)

    # "randperm": legacy vision. torch.randperm on a CPU generator seeded with
    # config.seed — reproduced with numpy's permutation, which draws a different
    # permutation than torch for the same seed. The SPLIT SEMANTICS (uniform, not
    # class-balanced, n_test = int(N*ratio) tail) are identical; the exact membership is
    # not. Legacy accuracies reproduce to within split noise, not bit-exactly.
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(n)
    return perm[: n - n_test], perm[n - n_test :]


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    config: ProbeConfig = CANONICAL,
    *,
    classes: tuple[Any, ...] | None = None,
    progress: bool = False,
) -> ProbeResult:
    """Fit one multinomial linear probe and report held-out accuracy.

    A single `nn.Linear` + `CrossEntropyLoss` trained with AdamW, matching
    `linear_probe.py:37-110`. The whole fit runs on the GPU when one is available: the
    legacy code moved the array to the device once and indexed it there, because the
    per-epoch CPU->GPU copy dominated runtime at 28 layers x 4 tasks x 3 models.

    Args:
        X: (N, D) features. fp16 on disk is expected and is upcast to fp32 here — probing
            an fp16 array directly makes AdamW's updates underflow.
        y: (N,) integer class labels.
        config: hyperparameters. Defaults to CANONICAL; pass `LEGACY_VISION` /
            `LEGACY_LETTER` to reproduce published numbers.
        classes: pinned class order. Defaults to sorted(unique(y)). Pass explicitly when
            fitting per-layer probes that must share a row ordering, since
            `ProbeResult.weights` rows are only comparable across layers under a shared
            order.
        progress: show a per-epoch tqdm bar.

    Returns:
        ProbeResult with test accuracy as a fraction.

    Raises:
        ValueError: if X is not 2-D, if X and y disagree in length, or if a label in `y` is
            absent from an explicitly-passed `classes`. X must be 2-D because that is how
            every stage is stored: the vision stages are FLATTENED to (N, T*D) on disk (see
            `modirect.io.feature_store`). To probe a single frame, slice
            `ref.unflatten()[:, t, :]` deliberately rather than passing (N, T, D) here.
    """
    import torch
    import torch.nn as nn

    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, D), got {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")

    if classes is None:
        classes = tuple(np.unique(y).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    try:
        y_idx = np.array([class_to_idx[v] for v in y.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            f"label {exc.args[0]!r} is not in classes {tuple(classes)}; pass a `classes` "
            "that covers every value in y (see modirect.probing.targets.encode_labels)"
        ) from None

    device = _resolve_device(config.device)
    num_classes = len(classes)

    train_idx, test_idx = _split_indices(y_idx, config)

    # `load_features` hands back a READ-ONLY fp16 memmap by default. `torch.from_numpy`
    # rejects non-writable arrays with a UserWarning and returns a tensor aliasing those
    # pages, so copy when needed. The fp16 -> fp32 upcast is deferred to `.to(device)` so
    # that only half as many bytes cross the bus, which is what the legacy code did
    # (`linear_probe.py:43`) and why a 28-layer sweep was tractable.
    arr = np.ascontiguousarray(X)
    if not arr.flags.writeable:
        arr = arr.copy()
    Xt = torch.from_numpy(arr).to(device, dtype=torch.float32)
    yt = torch.from_numpy(y_idx).to(device)
    tr = torch.from_numpy(train_idx).to(device)
    te = torch.from_numpy(test_idx).to(device)
    del arr

    # --- standardise -------------------------------------------------------------
    # `standardize_on` decides whether the test rows contribute to mean/std. "all"
    # reproduces the legacy leak; "train" is correct.
    fit_rows = Xt if config.standardize_on == "all" else Xt[tr]
    mean = fit_rows.mean(dim=0)
    std = fit_rows.std(dim=0)
    if config.std_guard == "clamp":
        std = std.clone()
        std[std < 1e-8] = 1.0
    else:
        std = std + 1e-8
    del fit_rows  # a "train" fit_rows is a full copy of the training split; free it early
    Xt = (Xt - mean) / std

    X_train, y_train = Xt[tr], yt[tr]
    X_test, y_test = Xt[te], yt[te]
    del Xt, yt

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = nn.Linear(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    n_train = X_train.shape[0]
    epochs: Any = range(config.epochs)
    if progress:
        from tqdm import tqdm  # optional; only pulled in when explicitly requested

        epochs = tqdm(epochs, desc="probe", leave=False)

    model.train()
    for _ in epochs:
        order = torch.randperm(n_train, device=device)
        for i in range(0, n_train, config.batch_size):
            batch = order[i : i + config.batch_size]
            loss = criterion(model(X_train[batch]), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()
        weights = model.weight.detach().float().cpu().numpy()
        bias = model.bias.detach().float().cpu().numpy()

    del model, optimizer, X_train, X_test, y_train, y_test
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ProbeResult(
        accuracy=test_acc,
        train_accuracy=train_acc,
        weights=weights,
        bias=bias,
        classes=tuple(classes),
        n_train=len(train_idx),
        n_test=len(test_idx),
        config=config,
    )
