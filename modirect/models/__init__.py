"""Model loading — a typed spine over `core/model_loader.py`.

    from modirect.models import load_model

    lm = load_model("baseline")                   # registry short name
    lm = load_model("delta", device_map="cuda:0")
    lm = load_model("pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5")

`load_model` returns a `LoadedModel`, which unpacks as the legacy 6-tuple
``(tokenizer, model, image_processor, context_len, model_name, conv_template)`` that 40+
scripts already expect.

THE DIVISION OF LABOUR — this package does NOT own the registry
---------------------------------------------------------------
`modirect.config.registry` owns the model *identities*: the `vanilla`/`baseline`/`delta`
names, `VANILLA_ARGS`, the LoRA basenames, and the rule that composes them into an args
string. This package owns only the *loading*: turning such a string into a live model.

Consequently there are two similarly-named functions. Keep them straight:

    modirect.config.resolve_model_args(name, paths) -> str     short name -> STRING
    modirect.models.build_model_args(name_or_args)  -> dict    STRING/name -> PARSED DICT

`build_model_args` calls the config one for short names, then parses. Only it takes a raw
args string and typed overrides. Nothing here restates `VANILLA_ARGS` — a second copy
would silently diverge, and since `max_frames_num=8` / `mm_spatial_pool_mode=bilinear`
define the shape of every cached feature, divergence would invalidate comparisons rather
than raise.

IMPORT CONTRACT
---------------
This package is importable **without llava/torch**: core is imported lazily, so
`import modirect.models` works anywhere. Keep it that way — do not add a module-level
`from core...` here or in `loader.py`. Note `core/__init__.py:1-2` imports torch *and*
llava, so any `from core import ...` drags in the whole stack.

*Calling* `parse_model_args` / `build_model_args` / `load_model` does require that stack,
even though the first two are pure string work — core's llava import fires on first
touch. `modirect.models.loader`'s docstring explains why that is left as-is.
"""

from __future__ import annotations

from .loader import LoadedModel, build_model_args, load_model, parse_model_args

__all__ = [
    "LoadedModel",
    "load_model",
    "build_model_args",
    "parse_model_args",
]
