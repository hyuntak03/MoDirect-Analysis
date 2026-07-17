"""Central filesystem configuration — the single place a path is written down.

Before this module, 51 distinct absolute paths were hardcoded across the analysis
scripts. The same feature root appears as a bare literal in, among others:

    analysis/task_invariance/axis_layer_cos.py:21    f"{ROOT}/{MODELS[model]}/answer_token/..."
    analysis/task_invariance/magnitude_cascade.py    ROOT = "/data3/local_datasets/..."
    analysis/task_invariance/mechanism_diagnosis.py:37  BASELINE_LORA = "/data/takhyun03/..."

Those literals encode one researcher's host layout, so the repo is unrunnable anywhere
else. This module resolves every path once, with a three-tier precedence:

    1. ``configs/paths.yaml``   — explicit, checked-in-by-the-user overrides (gitignored)
    2. environment variables    — the names the migrated scripts ALREADY read, so a shell
                                  that works for ``pipeline/`` works here unchanged
    3. documented defaults      — the literals verified on the original host

VERIFIED FACTS about the current host (checked 2026-07-17; re-verify before trusting):

    /data3/local_datasets/vlm_direction        EXISTS   primary feature root (Sections A-H).
                                                        NOTE: the `linear_probing_1500/`
                                                        subdirectory is NOT present here —
                                                        the root exists, the features do not.
    /local_datasets/vlm_direction              EXISTS   video folder + factorial dataset root.
                                                        NOTE: the `factorial_dataset/`
                                                        subdirectory is NOT present here.
    /data2/local_datasets/vlm_direction        MISSING  entirely.
    /data/datasets/LLaVA-Video-100K-Subset     MISSING  (old HF_HOME default; kept as the
                                                        default only because the migrated
                                                        scripts still `setdefault` it).
    LLaVA-NeXT root                            MISSING  install separately; neither the
                                                        /nas2 nor the /data candidate exists.

Because so much is missing, NOTHING here touches the filesystem at import time and no
default is validated on construction. `load_paths()` must succeed on a bare host — this
package has to stay importable and testable without llava, torch, or any dataset present
(see the module docstring of `modirect.config` for the wrapping contract around `core/`).
Use `Paths.missing()` when you want an explicit, opt-in existence report.

Environment variable names are fixed by prior art and must not be renamed:

    VLM_DIRECTION_ROOT   pipeline/05_intervene/causal_intervention.py:36
    LLAVA_NEXT_ROOT      pipeline/05_intervene/causal_intervention.py:38
    HF_HOME              pipeline/05_intervene/causal_intervention.py:39
    HF_DATASETS_CACHE    pipeline/05_intervene/causal_intervention.py:40
    SYN_V4_LORA          pipeline/06_readout/decoding_gap_analysis.py:45
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "Paths",
    "load_paths",
    "repo_root",
    "DEFAULT_VLM_DIRECTION_ROOT",
    "DEFAULT_CONFIG_RELPATH",
]

#: Marker file the repo-root walk looks for, matching the migrated scripts verbatim
#: (`pipeline/05_intervene/causal_intervention.py:29`). `pyproject.toml` warns in its own
#: header that it doubles as the marker and must not be renamed: the nine scripts that
#: load `core/` by literal path key off it, and renaming it does not raise — the walk
#: runs off the top of the tree and `core/` resolves somewhere wrong.
_ROOT_MARKER = "pyproject.toml"

#: Default for ``VLM_DIRECTION_ROOT``. Verbatim from causal_intervention.py:36.
DEFAULT_VLM_DIRECTION_ROOT = "/nas2/data/takhyun03/project/2026/vlm_direction"

#: Where `load_paths()` looks for the YAML override, relative to `repo_root()`.
DEFAULT_CONFIG_RELPATH = "configs/paths.yaml"


def repo_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Return the MoDirect repo root, using the migrated scripts' marker walk.

    Walks upward from `start` looking for `pyproject.toml`, exactly as
    `pipeline/05_intervene/causal_intervention.py:26-32` does. Reproduced rather than
    imported because those scripts are stand-alone entry points that this package must
    not import (they pull in torch at module scope).

    Args:
        start: Path to start walking from. Defaults to this file, which puts the answer
            two levels above `modirect/config/`.

    Returns:
        The directory containing `pyproject.toml`.

    Note:
        Where the migrated scripts raise `RuntimeError` if the marker is not found, this
        falls back to the static ancestor of this file (`modirect/config/paths.py` -> up
        3 == root). The scripts are entry points — for them a missing marker means
        `core/` is about to load from the wrong place, so dying loudly is right. This is
        library code that must stay importable on a bare host (installed as a wheel, or
        with the marker not yet in place), and the fallback is exact for the repo's
        actual layout. Both branches agree whenever the marker is present, which it is.
    """
    p = Path(start if start is not None else __file__).resolve()
    if p.is_file():
        p = p.parent
    for candidate in (p, *p.parents):
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    # Fallback: modirect/config/paths.py -> config -> modirect -> <root>
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Every filesystem root the analysis code needs, resolved once.

    All fields are absolute `Path`s. None are checked for existence — see the module
    docstring for why (most are absent on this host).

    Attributes:
        vlm_direction_root: The umbrella project directory that holds LLaVA-NeXT,
            the synthetic testbed, and the training work_dirs. Env: VLM_DIRECTION_ROOT.
        feature_root: 8-way-candidate cached features, `linear_probing_1500/`. Backs
            Sections A-G. Layout: `{feature_root}/{model}/{stage}/{task}/features.npy`
            (see `modirect.config.stages`).
        feature_root_4way: 4-way-candidate re-extraction, `linear_probing_4way_1500/`.
            Backs Section H (letter-vs-direction probing). Same layout as `feature_root`.
            Separate from `feature_root` because the letter label space only exists here:
            direction labels are candidate-count-invariant, letter labels are not.
        video_folder: Root the MCQ `video` fields are relative to. Scripts strip this
            prefix off absolute video paths (`pipeline/05_intervene/vision/vision_amp.py:79`),
            so the trailing separator convention matters — kept as the literal they use.
        factorial_root: The controlled (obj, bg, dir, instance, mcq) 5x5x4x20x4 dataset.
        hiddens_root: Per-layer last-token hidden dumps, `{factorial_root}/hiddens`.
            Consumed as `baseline_{cond}_4variants*.npz` by
            `analysis/task_invariance/mechanism_diagnosis.py:47`.
        llava_next_root: LLaVA-NeXT checkout; prepended to `sys.path` by the runtime
            scripts. Env: LLAVA_NEXT_ROOT. MISSING on this host.
        hf_home: Env: HF_HOME. MISSING on this host.
        hf_datasets_cache: Env: HF_DATASETS_CACHE. Note the original default points at
            `/local_datasets/vlm_direction/`, i.e. the same tree as `video_folder`.
        checkpoint_root: LoRA work_dirs, `{llava_next_root}/work_dirs`. The LoRA
            basenames in `modirect.config.registry` are resolved against this — the
            registry never hardcodes a root.
        syn_v4_lora: Optional path for the `syn_v4_baseline` model used only by
            `pipeline/06_readout/decoding_gap*.py`. Empty by default: the original was a
            third-party path that did not survive migration. Env: SYN_V4_LORA.
        output_root: Where this package writes results. Defaults inside the repo so a
            fresh clone produces output without configuration.
    """

    vlm_direction_root: Path
    feature_root: Path
    feature_root_4way: Path
    video_folder: Path
    factorial_root: Path
    hiddens_root: Path
    llava_next_root: Path
    hf_home: Path
    hf_datasets_cache: Path
    checkpoint_root: Path
    syn_v4_lora: Path
    output_root: Path

    def as_dict(self) -> dict[str, str]:
        """Serialise to plain strings — round-trips through `configs/paths.yaml`."""
        return {f.name: str(getattr(self, f.name)) for f in fields(self)}

    def missing(self) -> dict[str, Path]:
        """Report which roots are absent, for opt-in preflight checks.

        Existence is never asserted at construction time, so call this from a script's
        `main()` when you are about to actually read from disk.

        Returns:
            Mapping of field name -> path, for every field that does not exist.
        """
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if str(getattr(self, f.name)) and not getattr(self, f.name).exists()
        }


def _defaults(vlm_root: Path, root: Path) -> dict[str, Path]:
    """Documented defaults, derived from the literals found in the real scripts.

    Args:
        vlm_root: Already-resolved VLM_DIRECTION_ROOT; several defaults hang off it.
        root: Already-resolved repo root; `output_root` hangs off it.
    """
    llava_next = vlm_root / "LLaVA-NeXT"
    factorial = Path("/local_datasets/vlm_direction/factorial_dataset")
    return {
        "vlm_direction_root": vlm_root,
        # Section A-G feature cache. CLAUDE.md "디렉토리 구조".
        "feature_root": Path("/data3/local_datasets/vlm_direction/linear_probing_1500"),
        # Section H. Same tree, 4-way candidate re-extraction.
        "feature_root_4way": Path(
            "/data3/local_datasets/vlm_direction/linear_probing_4way_1500"),
        # vision_amp.py:42 VIDEO_FOLDER — trailing slash is load-bearing for the
        # prefix-strip at vision_amp.py:79.
        "video_folder": Path("/local_datasets/vlm_direction"),
        "factorial_root": factorial,
        "hiddens_root": factorial / "hiddens",
        "llava_next_root": llava_next,
        # causal_intervention.py:39 — MISSING on this host; kept for fidelity.
        "hf_home": Path("/data/datasets/LLaVA-Video-100K-Subset"),
        # causal_intervention.py:40.
        "hf_datasets_cache": Path("/local_datasets/vlm_direction"),
        # delta_effect_analysis.py:42 resolves LoRAs under {_VLM_ROOT}/LLaVA-NeXT/work_dirs.
        "checkpoint_root": llava_next / "work_dirs",
        "syn_v4_lora": Path(""),
        "output_root": root / "outputs",
    }


#: field name -> env var. Only the five names the migrated scripts already read are
#: honoured; inventing new ones would silently diverge from `pipeline/`.
_ENV_KEYS: Mapping[str, str] = {
    "vlm_direction_root": "VLM_DIRECTION_ROOT",
    "llava_next_root": "LLAVA_NEXT_ROOT",
    "hf_home": "HF_HOME",
    "hf_datasets_cache": "HF_DATASETS_CACHE",
    "syn_v4_lora": "SYN_V4_LORA",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load `configs/paths.yaml`, tolerating absence and an empty file."""
    if not path.is_file():
        return {}
    import yaml  # local import: keeps `import modirect.config` dependency-free

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a top-level mapping, got {type(data).__name__}")
    return data


def load_paths(
    config_path: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Paths:
    """Resolve every path with precedence overrides > YAML > env > defaults.

    `vlm_direction_root` and `llava_next_root` are resolved first, because other
    defaults are derived from them: pointing `VLM_DIRECTION_ROOT` at a new host moves
    `checkpoint_root` with it, which is what makes the LoRA basenames in
    `modirect.config.registry` portable.

    Args:
        config_path: YAML file to read. Defaults to `{repo_root()}/configs/paths.yaml`.
            Absent file is not an error — that is the expected state for a fresh clone.
            See `configs/paths.example.yaml` for the documented key set.
        env: Environment mapping. Defaults to `os.environ`. Injectable so tests need not
            mutate global state.
        overrides: Highest-precedence values, for callers that already parsed a
            `--feature-root`-style CLI flag.

    Returns:
        A frozen `Paths`. No path is checked for existence; call `Paths.missing()` if
        you need that.

    Raises:
        KeyError: if the YAML or `overrides` contains a key that is not a `Paths` field.
            Fail loudly — a typo'd key that was silently ignored would leave the caller
            reading the default root and quietly analysing the wrong features.
        TypeError: if the YAML top level is not a mapping.
    """
    env = os.environ if env is None else env
    root = repo_root()
    cfg = _read_yaml(Path(config_path) if config_path is not None
                     else root / DEFAULT_CONFIG_RELPATH)
    overrides = dict(overrides or {})

    valid = {f.name for f in fields(Paths)}
    for source, mapping in (("configs/paths.yaml", cfg), ("overrides", overrides)):
        unknown = set(mapping) - valid
        if unknown:
            raise KeyError(
                f"{source}: unknown key(s) {sorted(unknown)}. "
                f"Valid keys: {sorted(valid)}")

    def pick(name: str, default: Path) -> Path:
        """overrides > yaml > env > default, first non-empty wins."""
        for value in (overrides.get(name), cfg.get(name), env.get(_ENV_KEYS.get(name, ""))):
            if value not in (None, ""):
                return Path(str(value)).expanduser()
        return default

    # Two-pass: the anchors must exist before the derived defaults are computed.
    vlm_root = pick("vlm_direction_root", Path(DEFAULT_VLM_DIRECTION_ROOT))
    defaults = _defaults(vlm_root, root)
    llava_next = pick("llava_next_root", defaults["llava_next_root"])
    defaults = {**defaults, "llava_next_root": llava_next,
                "checkpoint_root": llava_next / "work_dirs"}

    return Paths(**{name: pick(name, defaults[name]) for name in valid})
