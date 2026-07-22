# MoDirect-Analysis

**Where does direction information live in a Video-LLM, and why does it fail out of domain?**

Mechanistic analysis of direction reasoning in LLaVA-Video-7B-Qwen2 across the full
`vision encoder → projector → LLM → lm_head` pipeline, comparing a vanilla model against
two fine-tuned variants.

The headline result: on out-of-domain inputs the model *still encodes* direction almost
perfectly (linear probe ≈ 92%), but *fails to use it* (MCQ ≈ 79%). That gap is not missing
information, a rotated axis, or entanglement with object identity — it is a **magnitude
deficit on a single, identifiable axis at layer 21**. Scale the signal back up on that axis
and the accuracy comes back.

---

## The finding in two tables

Last-token intervention at layer 21, Baseline model, `obj_place` (hardest OOD split).
Full numbers and controls in [`docs/report.md`](docs/report.md) §7.

**What you do to the axis** (baseline 68.80%):

| Intervention | What it does | Accuracy | Δ |
|---|---|---:|---:|
| `no_swap` | nothing | 68.80% | — |
| `remove_own` | ablate the direction signal — **control** | 52.40% | −16.4pp |
| `amp_2x` | double the sample's own on-axis component | 73.80% | +5.0pp |
| `clean` @ 1×SC | remove own, re-inject in-domain magnitude | 78.80% | +10.0pp |
| `add_canon` | add in-domain magnitude, no removal | 80.40% | +11.6pp |
| `full_rep` | replace the token with the class prototype | **86.40%** | **+17.6pp** |
| same op at L14 / L16 / L18 | — | ~68.8% | **≈0pp** |

**How much magnitude you put on it** (separate sweep, baseline 67.67%):

| Target ‖Δ‖ | Accuracy | Δ |
|---|---:|---:|
| 1× SC (48) | 78.33% | +10.67pp |
| 2× SC (96) | 87.00% | +19.33pp |
| **3× SC (144)** | **89.33%** | **+21.67pp** ← peak |
| 10× SC (480) | 72.00% | +4.33pp — over-amplification collapses |
| **−1× SC** | **14.33%** | **−53.33pp** — below chance |

Four rows carry the argument:

- **`clean` @ 2×SC ≈ `full_rep`.** Setting one scalar reproduces replacing the entire hidden
  state. The recovery is magnitude — not identity content, not letter routing, not the prototype.
- **`remove_own` costs 16pp.** The model really is reading this axis; it isn't a probe artefact.
- **Negative magnitude drives accuracy to 14%, below the 25% chance floor.** The axis is a
  *signed* semantic vector, not a generic "more signal helps" knob.
- **A random axis at the same magnitude gives +0.9pp**, and the same operation at L14–L18
  gives ≈0pp. The effect is specific to *this* axis at *this* depth.

> **On the two baselines.** The condition table and the magnitude sweep were run separately
> (68.80% vs 67.67% baseline), which is why `clean` @ 2×SC reads +17.6pp in
> `docs/lab-notebook.md` and +19.33pp here. Same phenomenon, different runs — quote them
> separately, not against each other. `docs/report.md` is canonical.

---

## The four spines

Four concerns run through every experiment here. Each is now one library module instead of
being re-derived per script.

### 1. Model loading — lmms_eval style

```python
from modirect.models import load_model

m = load_model("baseline")                     # vanilla | baseline | delta
m = load_model("pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,device_map=auto")
m.model, m.tokenizer, m.image_processor        # typed, instead of a bare 6-tuple
```

`"key=val,key=val"` parsing with LoRA-over-base loading (`lora_pretrained` becomes the model
path, `pretrained` becomes the base). Wraps [`core/model_loader.py`](core/model_loader.py).

### 2. Task loading — lmms_eval style

```python
from modirect.tasks import list_tasks, load_questions

load_questions("vlm_direction_testbed_R2R_4way_1500_obj_place")
```

Recursive YAML discovery over [`tasks/`](tasks/), `!function` tag resolution against each
YAML's sibling `utils.py`, `include:` inheritance, and group expansion — then any HF dataset
flattened to a uniform `questions` list. Wraps [`core/dataset_loader.py`](core/dataset_loader.py).

### 3. Direction concept vectors

```python
from modirect.concepts import extract_concept_vectors

axes = extract_concept_vectors(hiddens, directions)   # (N, 28, D) -> per-layer axes
axes.delta["up"]        # Δ_d = mean(h | d) − mean(h | all)   the concept vector (raw)
axes.delta_hat["up"]    # Δ̂_d = Δ_d / ‖Δ_d‖                   the direction axis (unit)
axes.mag["up"]          # ‖Δ_d‖                               the magnitude  ← the finding
axes.prototype("up")    # g + Δ_d                             what `full_rep` injects
```

Deliberately keeps `delta` raw: the magnitude *is* the object of study, so `delta_hat` and
`mag` are derived views rather than the stored form.
→ [`modirect/concepts/axes.py`](modirect/concepts/axes.py)

### 4. Magnitude intervention

```python
from modirect.interventions import LastTokenIntervention

with LastTokenIntervention(model, 21, axes.at_layer(21), "clean", "up", magnitude=2 * mag_sc):
    logits = model(inputs_embeds=..., attention_mask=...).logits
```

The operator, in one line — remove the sample's own on-axis component, re-inject a
controlled one:

```
proj = ⟨h − g, Δ̂_d⟩
h'   = h − proj·Δ̂_d + magnitude·Δ̂_d
```

Conditions: `no_swap`, `amp_2x`, `clean`, `add_canon`, `on_axis`, `remove_own`, `full_rep`.
→ [`modirect/interventions/operators.py`](modirect/interventions/operators.py)

---

## Layout

```
MoDirect-Analysis/
├── modirect/              # ── the library: reusable, tested, importable ──
│   ├── config/            #    paths, model registry, direction/stage enums
│   ├── models/            #    ★ spine 1 — lmms_eval-style model loading
│   ├── tasks/             #    ★ spine 2 — lmms_eval-style task loading
│   ├── concepts/          #    ★ spine 3 — Δ_d = mean(d) − mean(all)
│   ├── interventions/     #    ★ spine 4 — h − proj·Δ̂ + mag·Δ̂, and the hook protocol
│   ├── probing/           #    one linear probe (direction / letter targets)
│   └── io/                #    feature store, concept-vector store
│
├── pipeline/              # ── the experiments, in the order they run ──
│   ├── 01_extract/        #    vision / llm / attention feature extraction
│   ├── 02_probe/          #    linear probes; binding/, cross_task/
│   ├── 03_geometry/       #    axes/, trajectory/, attention/, dims/, persample/
│   ├── 04_concepts/       #    canonicalize, extract prototypes, validate
│   ├── 05_intervene/      #    llm_last_token/, vision/, weights/, swap/
│   ├── 06_readout/        #    logit lens, lm_head alignment, decoding gap
│   └── 07_figures/        #    all plotting
│
├── core/                  # original LLaVA runtime — MUST stay at root (see Gotchas)
├── tasks/                 # 151 lmms_eval-style task YAMLs — MUST stay beside core/
├── experiments/upstream/  # Zhang et al. cross-modal-information-flow (provenance)
├── legacy/                # superseded but preserved: older probing variants, utils_addon
├── assets/                # committed concept vectors (.pt) + vision axes (.npz)
├── configs/               # paths.example.yaml, models.yaml (+ paths.yaml, host-local)
├── z_script/              # host-specific launchers — GITIGNORED, see z_script/README.md
├── docs/                  # report.md (canonical), lab-notebook.md, figures/
└── tests/                 # concept/intervention math + layout invariants
```

## Install

```bash
git clone https://github.com/hyuntak03/MoDirect-Analysis.git
cd MoDirect-Analysis
pip install -e .                    # library + analysis deps
pytest tests/ -q                    # math + layout checks, no GPU needed
```

`pip install -e .` matters: it puts `modirect` and `core` on the path, so scripts resolve
their imports from any depth. The runtime extra (`pip install -e ".[runtime]"`) needs
[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) installed **separately from source** —
it is not on PyPI.

```bash
cp configs/paths.example.yaml configs/paths.yaml   # then edit for your host
export LLAVA_NEXT_ROOT=/path/to/LLaVA-NeXT
```

## Models

| Name | Description |
|---|---|
| `vanilla` | LLaVA-Video-7B-Qwen2, no fine-tuning |
| `baseline` | 4combo_v2 LoRA, MCQ loss only |
| `delta` | 4combo_v2 + `delta_direct` auxiliary loss on the projector |

## Tasks — R2R 4-way, 1500 samples/direction

| Task | Object | Background | Difficulty |
|---|---|---|---|
| `shape_color` | synthetic | synthetic | in-domain |
| `obj_color` | real | synthetic | moderate OOD |
| `shape_place` | synthetic | real | hard OOD |
| `obj_place` | real | real | hardest OOD |

Label space is Up / Down / Left / Right (chance = 25%).

## 2026-07 re-run — v5_new checkpoints, cross-domain axes, magnitude restore

The original feature caches (`/data3/...`) did not survive the host migration, so
the chain was re-run against the `4combo_v5_new` checkpoint generation — Qwen2-based
LoRAs (`llava-video-7b-qwen2_{baseline,channel_gate}_shape_simple_v5_new_lora-r64_f8_ep1_lr1e-5_bs12_ga2`,
base `lmms-lab/LLaVA-Video-7B-Qwen2`). `channel_gate` is an **inference-active**
tanh SE-gate on the pooled projector features (`use_motion_query=true`, weights in
`non_lora_trainables.bin`), so it must be loaded through a LLaVA-NeXT checkout that
has `motion_modules.py` — with this checkpoint's flags (aggregation=mean, no
framewise/posneg/magnitude) old and new ChannelGate implementations are equivalent.

The chain, in order (host launchers live in `z_script/`, gitignored):

1. **Extract** — `pipeline/01_extract/llm/extract_answer_features.py` per
   (model, task): answer-token hiddens at all 29 `hidden_states` indices,
   6000 samples/task (R2R_4way_1500), fp16 `features_layer_{L}.npy`.
2. **Axes** — `pipeline/03_geometry/axes/cross_domain_axes.py`: Δ_d per
   (model, domain, layer, direction) via `modirect.concepts`, then layer-wise
   cross-domain cos(Δ̂ᴬ, Δ̂ᴮ) for all 6 domain pairs and ‖Δ_d‖ per (layer, domain,
   direction) → `outputs/cross_domain_axes_qwen2_v5/{model}/` (npz + json + figs).
3. **Intervene** — `pipeline/05_intervene/llm_last_token/magnitude_restore_v5.py`:
   on obj_place keep the domain's own axis and set only the magnitude
   (`clean`: h − proj·Δ̂_op,d + m·Δ̂_op,d) at feat L21 with m = ‖Δ_sc,d‖, paired
   per sample against `no_swap` / `clean_op` / `remove_own`; shards auto-match
   the allocated GPUs. Full condition table (`amp_2x`, `add_canon_sc`,
   `full_rep`) available via `--conditions`.
4. **Report** — `magnitude_restore_report.py` merges shards →
   `outputs/interventions_qwen2_v5/…/result.json` + accuracy figure.

Headline numbers from step 2 (feat L21, mean over 4 directions):

| | vanilla | baseline_v5 | channel_gate_v5 |
|---|---:|---:|---:|
| cross-domain Δ̂ cos (6-pair mean) | 0.478 | 0.934 | 0.947 |
| ‖Δ‖ shape_color (in-domain) | 2.7 | 30.0 | 30.1 |
| ‖Δ‖ obj_place (hardest OOD) | 1.3 | 14.7 | 17.9 |
| obj_place / shape_color ratio | 0.48 | 0.49 | **0.60** |

Fine-tuning creates a task-invariant axis from feat L20 (6-pair cos jumps
0.67 → 0.94 across the L19→L20 boundary) together with the magnitude cliff; the
OOD magnitude ordering shape_color > obj_color > shape_place > obj_place
reproduces on v5, and channel_gate narrows the OOD deficit without touching the
in-domain magnitude.

> **Layer convention on this route:** `features_layer_{L}.npy` = `hidden_states[L]`
> (L=0 embeddings), so "feat L21" is the output of decoder **module 20** and the
> intervention hook goes on module `L−1`. This is NOT the committed-`.pt` assets'
> convention (decoder-indexed) — never compare "L21" across the two routes.

**Host config.** `configs/paths.yaml` (untracked, this host) points `feature_root`
at the node-local cache
`/data2/local_datasets/vlm_direction_modirect/linear_probing_R2R_4way_1500`
(vll5 only — not visible from other nodes). `z_script/` holds the host launchers
(conda env, SLURM partition, sbatch wrappers); recreate per host from its README.

## Gotchas

Each of these fails **silently** — no exception, just wrong or empty results.

- **`core/` must stay at the repo root.** Nine scripts load it by literal filesystem path
  (`spec_from_file_location(..., os.path.join(_PROJECT_ROOT, "core", "model_loader.py"))`),
  not by import. A re-export shim elsewhere will not satisfy them.
- **`tasks/` must stay a sibling of `core/`.** `core/dataset_loader.py:119` derives
  `dirname(dirname(__file__))/tasks` and runs `discover_tasks()` at import time. Point it
  wrong and the registry is empty — with no error.
- **Hooks: `decoder_layer` writes, `self_attn`/`mlp` read.** Qwen2 **ignores** the return
  value of a `self_attn` forward hook, so writing there is a silent no-op. Always clone
  before writing and preserve the `(hidden, *rest)` tuple.
- **Never mix layers.** The L14 direction axis is near-orthogonal to L21's (cos ≈ 0.04), so
  injecting along the wrong one does nothing. Interventions land at L20–L21.
- **`pyproject.toml` is the repo-root marker** every script walks up to find. Don't remove it.

## Provenance and caveats

- **`docs/report.md` is canonical.** `docs/lab-notebook.md` (the working notebook) records
  the path taken, including retracted claims — earlier "entanglement" and whole-swap
  interpretations were found to be confounded and are marked as such.
- **Vision-side magnitude scaling does not work** (up to −47pp). This is a real, reported
  negative result, not an omission: the vision-level axis is roughly orthogonal to the L21
  readout axis, so amplifying it just moves inputs off-distribution.
- **Lost code.** The factorial-experiment `scripts/` and `results/` referenced by the lab
  notebook were never committed — the old `.gitignore` had bare `scripts/` and `*.json`
  rules that swallowed them. Only its figures and logs survive. The new `.gitignore` uses
  anchored patterns to prevent a repeat.
- **Paths.** Scripts previously hardcoded `/data/takhyun03/...`, which no longer resolves;
  they now resolve the repo root by marker walk and read data roots from
  `$VLM_DIRECTION_ROOT` / `configs/paths.yaml`.
- **2026-07 fixes.** Four scripts referenced `_PROJECT_ROOT` without defining it
  (`measure_invariance.py` died at import, `magnitude_cascade.py` *after* its full
  compute, plus `vision_token_axis_per_layer.py` and `01_canonicalize_dataset.py`) —
  all now carry the marker-walk helper. `core/dataset_loader.py` drops a YAML
  `token: True` when no HF token is available, so the public testbed dataset loads
  on hosts without a login.
- Built on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT). `experiments/upstream/`
  is retained from [Cross-modal Information Flow in MLLMs](https://arxiv.org/abs/2411.18620)
  (Zhang et al.).
