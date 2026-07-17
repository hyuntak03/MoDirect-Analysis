"""Typed wrapper over `core/model_loader.py` — the lmms_eval-style model spine.

WHY THIS MODULE EXISTS
----------------------
`core/model_loader.py` is 110 lines and works, but it hands back a bare 6-tuple and takes
its configuration as a comma-separated string. Forty-plus scripts therefore open with the
same three lines and the same unpacking, e.g.
`pipeline/05_intervene/llm_last_token/mechanism_diagnosis.py:82-86`::

    from core.model_loader import parse_model_args, load_model_from_args
    args_str = f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}"
    return load_model_from_args(parse_model_args(args_str))

followed by `tokenizer, model, image_processor, context_len, model_name, conv_template =
...` at each call site. This module gives that pipeline a name, a type, and a docstring
while delegating every byte of actual behaviour to core.

WHERE THE REGISTRY LIVES — NOT HERE
-----------------------------------
The `vanilla`/`baseline`/`delta` short names, `VANILLA_ARGS`, the LoRA basenames, and the
``f"lora_pretrained={lora},{VANILLA_ARGS}"`` composition rule all belong to
`modirect.config.registry`, which resolves the LoRA root against `Paths.checkpoint_root`
rather than hardcoding it. This module **must not** restate any of them: a second copy of
`VANILLA_ARGS` here would silently diverge from the one the rest of the package composes
args with, and since `max_frames_num=8` / `mm_spatial_pool_mode=bilinear` define the shape
of every cached feature, a divergence would invalidate comparisons rather than raise.

Note the two same-named-but-different functions, and keep them straight:

    modirect.config.registry.resolve_model_args(name, paths) -> str    short name -> STRING
    modirect.models.loader.build_model_args(name_or_args, ...) -> dict STRING/name -> PARSED DICT

`build_model_args` sits one layer above: it calls config's resolver for a short name, then
parses. Only this module's version accepts a raw args string and typed overrides.

THE llava IMPORT, AND WHY EVERY core IMPORT HERE IS LAZY
--------------------------------------------------------
`core/model_loader.py:10-11` does, at module top level::

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

and `llava` is NOT installed on the analysis/CI host — it lives in the separate LLaVA-NeXT
checkout only the GPU runtime puts on `sys.path`. Worse, `core/__init__.py:1-2` imports
`core.methods` (-> torch, matplotlib) *and* `core.model_loader` (-> llava), so **any**
`from core import ...` drags in the whole stack. An eager import here would make
`import modirect` raise everywhere except the GPU box, breaking the pure-numpy half of the
package (`modirect.concepts`, `modirect.interventions.operators`) and its test suite.
Hence: **every core import in this file is inside a function body**.

WHAT IS AND IS NOT AVAILABLE WITHOUT llava
------------------------------------------
Importing this module, and constructing a `LoadedModel`, work with nothing installed.
**Calling `parse_model_args` does not** — surprising, because it is pure string
manipulation with no tensor in sight, but the delegation touches core and core's
module-scope llava import fires. Verified on a host without the GPU stack::

    >>> build_model_args("vanilla")
    ModuleNotFoundError: No module named 'torch'

The delegation is kept anyway, deliberately: core's coercion rules are quirky (see
`parse_model_args`), 40+ scripts depend on their exact behaviour, and a re-implementation
here would be a second source of truth that could drift silently. A wrong-but-plausible
args dict is a far worse failure than an ImportError. If arg parsing is ever needed off
the GPU host, the fix belongs in core — thin out `core/__init__.py` and lift the llava
imports into `load_model_from_args` — not in a vendored copy here.

THE conv_template ODDITY
------------------------
`conv_template` is read out of model_args (`core/model_loader.py:62`) and returned
(`:90`), but never passed to `load_pretrained_model`. It is pure passthrough metadata for
the caller's prompt builder. It is in the tuple because callers need it, not because the
loader uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from ..config import Paths
from ..config import resolve_model_args as _resolve_args_string

__all__ = [
    "LoadedModel",
    "parse_model_args",
    "build_model_args",
    "load_model",
]


@dataclass(frozen=True)
class LoadedModel:
    """A loaded LLaVA-Video model and everything needed to prompt it.

    This replaces the bare 6-tuple returned by `core/model_loader.py:90`::

        tokenizer, model, image_processor, context_len, model_name, conv_template

    **That tuple order is load-bearing.** Forty-plus scripts unpack it positionally today
    (e.g. `pipeline/05_intervene/llm_last_token/mechanism_diagnosis.py`,
    `pipeline/01_extract/llm/extract_answer_features.py`). `LoadedModel` is iterable and
    yields exactly that order, so a legacy call site migrates with a one-line change and
    no reordering::

        tok, model, ip, ctx, name, conv = load_model("baseline")   # still works

    Do not reorder the fields. Add new ones at the end only.

    Attributes:
        tokenizer: HF tokenizer for the LLM half (Qwen2).
        model: the `LlavaQwenForCausalLM`. Already in `.eval()` mode —
            `core/model_loader.py:82` calls it, so callers must not rely on train-mode
            behaviour. Core does NOT disable gradients; the scripts call
            `torch.set_grad_enabled(False)` themselves, and so should you.
        image_processor: SigLIP preprocessor for frames.
        context_len: max context length reported by the builder.
        model_name: derived by `get_model_name_from_path`. For a LoRA run this comes from
            the **LoRA directory name**, not the base repo (`core/model_loader.py:68`) —
            see `load_model`.
        conv_template: conversation template key (default ``"qwen_1_5"``). Passthrough
            only; the loader never uses it (see module docstring).
        model_args: the parsed args dict this model was loaded from. Not part of the
            legacy tuple — it is appended metadata, and is deliberately excluded from
            iteration so unpacking stays 6-wide.
    """

    tokenizer: Any
    model: Any
    image_processor: Any
    context_len: int
    model_name: str
    conv_template: str
    model_args: dict

    def __iter__(self) -> Iterator[Any]:
        """Yield the legacy 6-tuple, in the order 40+ scripts unpack it."""
        return iter(
            (
                self.tokenizer,
                self.model,
                self.image_processor,
                self.context_len,
                self.model_name,
                self.conv_template,
            )
        )

    @property
    def is_lora(self) -> bool:
        """True if loaded as base + LoRA adapter rather than a plain checkpoint."""
        return bool(self.model_args.get("lora_pretrained"))

    @property
    def num_layers(self) -> int:
        """Number of decoder layers (28 for LLaVA-Video-7B-Qwen2, i.e. L0..L27)."""
        return len(self.model.model.layers)


def parse_model_args(args_string: str | None) -> dict:
    """Parse an lmms_eval-style ``"key=val,key=val"`` string into a dict.

    Delegates to `core.model_loader.parse_model_args`. The delegation is the point: the
    coercion rules below are quirky, and re-implementing them here would let this wrapper
    drift from what the 40+ existing scripts actually get.

    **Requires llava/torch to be importable**, despite doing nothing but string work — the
    lazy import still triggers core's module-scope llava import. See the module docstring.

    The rules, verified against `core/model_loader.py:20-43`:

    * **Splits on ``","`` unconditionally** (`:23`), with no quoting and no escaping, so
      **no value may contain a comma**. This does not fail loudly — it **truncates
      silently**::

          >>> parse_model_args('device_map={"": 0, "x": 1},conv_template=qwen_1_5')
          {'device_map': '{"": 0', 'conv_template': 'qwen_1_5'}

      The dict literal is chopped at the comma, its tail (``' "x": 1}'``) has no ``"="``
      and is dropped by the next rule, and the surviving value is a mangled string. So
      ``device_map={"": 0}`` — a dict literal, the normal way to pin a model onto one GPU
      — **cannot be expressed here**. (A single-entry ``{"": 0}`` happens to contain no
      comma and survives intact, but still arrives as a `str`, not a dict.) Pass
      ``device_map="auto"`` or a plain device string, or give `load_model` a real dict as
      a typed override, which bypasses this parser entirely.
    * Items without ``"="`` are **silently dropped** (`:25-26`), not an error. A typo like
      ``"device_map auto"`` vanishes without a trace and you get the ``"auto"`` default.
    * Values are coerced in this order (`:28-41`): ``true``/``false`` -> bool,
      ``none`` -> None (all case-insensitively, via `val.lower()`), then `int(val)`, then
      `float(val)`, else left `str`. So ``force_sample=True`` -> `True`,
      ``max_frames_num=8`` -> `8`, ``lr=1e-5`` -> `1e-05`, ``device_map=auto`` ->
      ``"auto"`` (all coercions fail).
    * **Only the key is stripped** (`:42`); the value is not. Whitespace around ``=``
      therefore defeats the *bool/None* coercions but not the numeric ones, because
      `val.lower()` is compared against an exact literal while `int()`/`float()` tolerate
      surrounding whitespace::

          >>> parse_model_args("force_sample = True")["force_sample"]
          ' True'          # a truthy STRING, not True — `is True` checks fail
          >>> parse_model_args("max_frames_num = 8")["max_frames_num"]
          8                # int() strips whitespace, so this one survives

      The bool case is the dangerous half: ``" True"`` is truthy, so it behaves correctly
      until something compares it to `True` or serialises it. Write ``key=val``, no spaces.
    * The coercion is type-blind: anything that merely *looks* numeric or boolean becomes
      so. Paths and repo names are safe only because they never spell ``true`` or parse as
      a number.

    Args:
        args_string: the lmms_eval args string; None/"" gives an empty dict.

    Returns:
        Parsed and coerced key -> value mapping.
    """
    from core.model_loader import parse_model_args as _parse  # lazy: see module docstring

    return _parse(args_string)


def build_model_args(
    name_or_args: str,
    *,
    paths: Paths | None = None,
    **overrides: Any,
) -> dict:
    """Resolve a registry short name *or* a raw args string into a **parsed args dict**.

    The bridge between `modirect.config.registry` (which knows the models, and returns an
    args *string*) and `core.model_loader` (which wants a parsed *dict*). Split out from
    `load_model` so you can inspect what a short name expands to without paying for a 7B
    load.

    Not to be confused with `modirect.config.registry.resolve_model_args`, which is the
    string-returning, short-name-only layer underneath this one. See module docstring.

    Args:
        name_or_args: a registry short name (see `modirect.config.MODEL_NAMES`:
            ``vanilla``/``baseline``/``delta``) or a raw lmms_eval args string. Anything
            containing ``"="`` is treated as a raw string; anything else must be a
            registry name.
        paths: resolved `Paths`, used only to locate the LoRA root for a short name.
            Defaults to `load_paths()` inside the config layer.
        **overrides: applied on top of the parsed dict and **already typed** — they bypass
            `parse_model_args` entirely, so pass `max_frames_num=8` (int) not ``"8"``
            (str), and `device_map={"": 0}` if you need the dict literal the string syntax
            cannot express.

    Returns:
        The parsed, overridden args dict, ready for `core.model_loader.load_model_from_args`.

    Raises:
        KeyError: if `name_or_args` is not a registry name (raised by the config layer,
            listing the valid names).
    """
    if "=" in name_or_args:
        args_string = name_or_args
    else:
        # Short name: let config own the LoRA path and the composition rule.
        args_string = _resolve_args_string(name_or_args, paths)

    args = parse_model_args(args_string)
    args.update(overrides)
    return args


def load_model(
    name_or_args: str,
    *,
    paths: Paths | None = None,
    **overrides: Any,
) -> LoadedModel:
    """Load a model by registry short name or raw lmms_eval args string.

    Thin delegation to `core.model_loader.load_model_from_args` — the loading logic stays
    in core so this wrapper cannot drift from what the migrated scripts do.

    THE LoRA CONVENTION (`core/model_loader.py:65-74`), stated precisely because it is
    inverted from what the key names suggest:

    * With ``lora_pretrained`` set, **``lora_pretrained`` becomes `model_path` and
      ``pretrained`` becomes `model_base`** (`:66-67`). The adapter is the "path"; the
      base repo is the "base". So ``pretrained=lmms-lab/LLaVA-Video-7B-Qwen2`` does NOT
      mean "load this model" in a LoRA run — it means "use this as the base to merge onto".
    * ``model_name`` is then derived from the **LoRA directory name**, not the base repo
      (`:68`): `get_model_name_from_path(lora_pretrained)`. That is why the checkpoints in
      `modirect.config.registry` are named `llava-video-7b-qwen2_..._lora-r64_...` — the
      name has to keep identifying both the architecture and its LoRA-ness downstream.
    * Without ``lora_pretrained``, ``pretrained`` is `model_path` and `model_base` is None
      (`:71-73`) — the plain-checkpoint path taken by `vanilla`.

    ``cache_dir`` IS NOT A model_args KEY. `core/model_loader.py:63` reads it exclusively
    from the environment::

        cache_dir = os.environ.get("HF_HOME", None)

    Putting ``cache_dir=/some/path`` in the args string is **silently ignored** — parsed
    into the dict, then never read — and the model downloads to the default HF cache
    instead. Set `HF_HOME` in the environment *before* calling; `Paths.hf_home` records
    the project's value. This function deliberately does not set it: the caller owns that
    decision (the migrated scripts do `os.environ.setdefault("HF_HOME", ...)` at import,
    e.g. `mechanism_diagnosis.py:42`), and writing it here would silently relocate the
    multi-hundred-GB cache of anyone who set it on purpose.

    Only two model_args keys are applied post-load, both onto `model.config`
    (`core/model_loader.py:85-88`): ``mm_spatial_pool_stride`` and ``mm_spatial_pool_mode``.
    Every other video key in `VANILLA_ARGS` (``max_frames_num``, ``force_sample``,
    ``video_decode_backend``) is inert *here* — it is consumed by the caller's frame-loading
    code, not by the loader. Do not expect `load_model` to honour them.

    Args:
        name_or_args: registry short name or raw lmms_eval args string. See
            `build_model_args`.
        paths: resolved `Paths` for the LoRA root; defaults to `load_paths()`.
        **overrides: typed overrides applied after parsing, e.g.
            ``load_model("baseline", device_map="cuda:0")``.

    Returns:
        LoadedModel — a named, frozen view of core's 6-tuple, plus the args it came from.

    Raises:
        KeyError: if `name_or_args` is an unknown short name.
        ModuleNotFoundError: if the GPU runtime (llava/torch) is absent. This is the one
            function in `modirect` that genuinely needs it; importing this module is not.

    Example:
        >>> lm = load_model("baseline")                       # doctest: +SKIP
        >>> lm.is_lora, lm.num_layers                         # doctest: +SKIP
        (True, 28)
        >>> tok, model, ip, ctx, name, conv = lm              # doctest: +SKIP
    """
    args = build_model_args(name_or_args, paths=paths, **overrides)

    # Lazy: core/__init__.py:1-2 pulls in torch and llava. See module docstring.
    from core.model_loader import load_model_from_args

    tokenizer, model, image_processor, context_len, model_name, conv_template = (
        load_model_from_args(args)
    )
    return LoadedModel(
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        context_len=context_len,
        model_name=model_name,
        conv_template=conv_template,
        model_args=args,
    )
