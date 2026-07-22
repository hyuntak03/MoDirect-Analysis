"""Magnitude restoration measured the lmms-eval way — generation + multi-layer hooks.

Same question as magnitude_restore_v5.py, but the accuracy is measured EXACTLY as
lmms-eval's llava_vid pipeline does (core/lmms_gen: bf16+sdpa, no prompt suffix,
model.generate + first-letter extraction), and the intervention can act on SEVERAL
layers at once — each layer with its OWN axes.

Conditions are `<op>@<layers>` strings, e.g.:
    no_swap                     before
    clean_sc@21                 h − proj·Δ̂_op,d + ‖Δ_sc,d‖·Δ̂_op,d   at feat L21
                                (remove the sample's OWN on-axis component first)
    shift_sc@21                 h + (‖Δ_sc,d‖ − ‖Δ_op,d‖)·Δ̂_op,d
                                (≡ h − Δ_op,d + ‖Δ_sc,d‖·Δ̂_op,d: shift the class MEAN
                                to SC magnitude, per-sample variation preserved)
    clean_sc@20+21              same, at BOTH feat L20 and L21 (own axes each)
    clean_op@21 / remove_own@21 controls

Axes are computed on the fly (modirect.concepts.extract_concept_vectors) from an
answer-token feature cache that MUST come from extract_answer_features_lmms.py —
the core/data_pipeline cache is a different prompt distribution.

Hooks fire on the PREFILL only (seq_len > 1): the last prompt token is modified,
decode steps run untouched downstream of the modified KV cache.

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python magnitude_restore_gen.py \
      --feature_root /data2/.../linear_probing_R2R_4way_1500_lmmsgen \
      --model_dir llava-video-7b-qwen2_baseline_v5_new \
      --conditions no_swap clean_sc@20 clean_sc@21 clean_sc@20+21 \
      --shard 0 --num_shards 4 --out /path/shard0.json
"""

import argparse
import gc
import importlib.util
import json
import os
import re
import sys

import numpy as np
import torch
from tqdm import tqdm


def _find_project_root(_start):
    """Walk up to the repo root (marker: pyproject.toml). Depth-independent."""
    _p = os.path.abspath(_start)
    while _p != os.path.dirname(_p):
        if os.path.isfile(os.path.join(_p, "pyproject.toml")):
            return _p
        _p = os.path.dirname(_p)
    raise RuntimeError("MoDirect repo root not found (no pyproject.toml above %s)" % _start)


_PROJECT_ROOT = _find_project_root(__file__)
sys.path.insert(0, _PROJECT_ROOT)
torch.set_grad_enabled(False)

from modirect.concepts import extract_concept_vectors  # noqa: E402
from modirect.interventions.operators import add_canon, clean, remove_own  # noqa: E402

DEFAULT_LORA = ("/data/jongseo/project/vlm/LLaVA-NeXT/4combo_v5_new/work_dirs/"
                "llava-video-7b-qwen2_baseline_shape_simple_v5_new_lora-r64_f8_ep1_lr1e-5_bs12_ga2")
DEFAULT_MODEL_ARGS = (f"lora_pretrained={DEFAULT_LORA},pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,"
                      "conv_template=qwen_1_5,max_frames_num=8,force_sample=True,"
                      "torch_dtype=bfloat16")
TASK_PREFIX = "vlm_direction_testbed_R2R_4way_1500_"


def parse_condition(spec):
    """'clean_sc@20+21' -> ('clean_sc', [20, 21]); 'no_swap' -> ('no_swap', [])."""
    if "@" not in spec:
        return spec, []
    op, _, layers = spec.partition("@")
    return op, [int(x) for x in layers.split("+")]


def load_layer_axes(feature_root, model_dir, axes_task, mag_task, layer):
    """Per-layer (g, delta_hat, mag_own, mag_target, label_list) from the lmms cache."""
    def tdir(task):
        return os.path.join(feature_root, model_dir, "answer_token", TASK_PREFIX + task)

    meta = np.load(os.path.join(tdir(axes_task), "meta.npy"), allow_pickle=True).item()
    if meta.get("pipeline") != "lmms_gen":
        raise ValueError(
            f"feature cache {tdir(axes_task)} is not from extract_answer_features_lmms.py "
            f"(meta.pipeline={meta.get('pipeline')!r}) — wrong prompt distribution")
    label_list = list(meta["label_list"])
    n = len(label_list)

    h = np.asarray(np.load(os.path.join(tdir(axes_task), f"features_layer_{layer}.npy"),
                           mmap_mode="r"))
    y = np.load(os.path.join(tdir(axes_task), "labels.npy"))
    ax = extract_concept_vectors(h, y, classes=list(range(n)))
    g = ax.g
    delta_hat = np.stack([ax.delta_hat[i] for i in range(n)])
    mag_own = np.array([ax.mag[i] for i in range(n)], dtype=np.float32)

    hm = np.asarray(np.load(os.path.join(tdir(mag_task), f"features_layer_{layer}.npy"),
                            mmap_mode="r"))
    ym = np.load(os.path.join(tdir(mag_task), "labels.npy"))
    axm = extract_concept_vectors(hm, ym, classes=list(range(n)))
    mag_target = np.array([axm.mag[i] for i in range(n)], dtype=np.float32)
    return g, delta_hat, mag_own, mag_target, label_list


def prefill_last_token_hook(fn):
    """last_token_hook variant that is a no-op on decode steps (seq_len == 1)."""
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] <= 1:          # generation step — leave untouched
            return output
        new_last = fn(hidden[:, -1, :])
        hidden = hidden.clone()
        hidden[:, -1, :] = new_last.to(hidden.dtype)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
    return hook


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_args", default=DEFAULT_MODEL_ARGS)
    ap.add_argument("--task", default=TASK_PREFIX + "obj_place")
    ap.add_argument("--feature_root", required=True,
                    help="lmms_gen answer-token cache root (extract_answer_features_lmms.py)")
    ap.add_argument("--model_dir", required=True,
                    help="model dirname under feature_root")
    ap.add_argument("--axes_task", default="obj_place")
    ap.add_argument("--mag_task", default="shape_color")
    ap.add_argument("--conditions", nargs="+",
                    default=["no_swap", "clean_sc@20", "clean_sc@21", "clean_sc@20+21"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conds = [(spec, *parse_condition(spec)) for spec in args.conditions]
    needed_layers = sorted({l for _, op, ls in conds for l in ls})

    # ---- axes per layer (fp32 cuda tensors) --------------------------------
    axes = {}
    label_list = None
    for L in needed_layers:
        g, dhat, m_own, m_tgt, ll = load_layer_axes(
            args.feature_root, args.model_dir, args.axes_task, args.mag_task, L)
        label_list = ll
        axes[L] = {"g": torch.from_numpy(np.ascontiguousarray(g)).float().cuda(),
                   "dhat": torch.from_numpy(np.ascontiguousarray(dhat)).float().cuda(),
                   "mag_own": m_own, "mag_target": m_tgt}
        print(f"[axes] L{L}: " + " ".join(
            f"{ll[i]}:{m_own[i]:.1f}->{m_tgt[i]:.1f}" for i in range(len(ll))))
    dir_to_idx = {n.lower(): i for i, n in enumerate(label_list or [])}

    # ---- questions (prompt text == lmms-eval doc_to_text) ------------------
    def _imp(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    _dl = _imp("core.dataset_loader", os.path.join(_PROJECT_ROOT, "core", "dataset_loader.py"))
    questions, _ = _dl.load_dataset_as_questions(
        task_name=args.task, hf_cache_dir=os.environ.get("HF_HOME"), limit=args.limit)
    questions = questions[args.shard::args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(questions)} samples")

    # ---- model (lmms-eval-faithful) ----------------------------------------
    _ml = _imp("core.model_loader", os.path.join(_PROJECT_ROOT, "core", "model_loader.py"))
    ma = _ml.parse_model_args(args.model_args)
    from core import lmms_gen
    torch_dtype = ma.get("torch_dtype", "bfloat16")
    tokenizer, model, image_processor, _, _ = lmms_gen.load_model_lmms(
        ma.get("pretrained"), ma.get("lora_pretrained"), torch_dtype=torch_dtype,
        attn_implementation=ma.get("attn_implementation", "sdpa"),
        device_map=ma.get("device_map", "cuda:0"))
    conv_template = ma.get("conv_template", "qwen_1_5")
    max_frames_num = int(ma.get("max_frames_num", 8))
    force_sample = bool(ma.get("force_sample", False))
    video_fps = int(ma.get("video_fps", 1))

    decoder_layers = (model.model.layers if hasattr(model.model, "layers")
                      else model.language_model.model.layers)

    def make_fn(op, L, di):
        ax = axes[L]

        def fn(h_last):
            h = h_last.float()
            if op == "clean_sc":
                return clean(h, ax["g"], ax["dhat"][di], magnitude=float(ax["mag_target"][di]))
            if op == "shift_sc":
                # h − Δ_op,d + ‖Δ_sc,d‖·Δ̂_op,d  ==  h + (‖Δ_sc,d‖ − ‖Δ_op,d‖)·Δ̂_op,d
                boost = float(ax["mag_target"][di]) - float(ax["mag_own"][di])
                return add_canon(h, ax["g"], ax["dhat"][di], magnitude=boost)
            if op == "clean_op":
                return clean(h, ax["g"], ax["dhat"][di], magnitude=float(ax["mag_own"][di]))
            if op == "remove_own":
                return remove_own(h, ax["g"], ax["dhat"][di])
            raise ValueError(op)
        return fn

    valid_letters = "".join(chr(ord("A") + i) for i in range(len(label_list)))

    def extract_letter(pred_raw):
        m = re.search(f"[{valid_letters}]", pred_raw.upper())
        return m.group(0) if m else "NONE"

    stats = {spec: {"n": 0, "correct": 0,
                    "per_dir": {n: {"n": 0, "correct": 0} for n in label_list}}
             for spec, _, _ in conds}
    examples = []

    for line in tqdm(questions, desc=f"shard{args.shard}"):
        d = str(line["direction"]).lower()
        if d not in dir_to_idx:
            continue
        di = dir_to_idx[d]
        gold = str(line["answer"]).strip().upper()
        dir_name = label_list[di]
        try:
            frames, _, _ = lmms_gen.load_video(
                line["video"], max_frames_num, fps=video_fps, force_sample=force_sample)
            video = lmms_gen.preprocess_video(image_processor, frames, torch_dtype)
            input_ids, attention_mask, stop_str = lmms_gen.build_prompt_inputs(
                tokenizer, line["question"], conv_template)

            for spec, op, layers in conds:
                handles = []
                if op != "no_swap":
                    for L in layers:
                        handles.append(decoder_layers[L - 1].register_forward_hook(
                            prefill_last_token_hook(make_fn(op, L, di))))
                try:
                    raw = lmms_gen.generate_answer(
                        model, tokenizer, input_ids, attention_mask, [video], stop_str,
                        max_new_tokens=args.max_new_tokens)
                finally:
                    for h in handles:
                        h.remove()
                pred = extract_letter(raw)
                s = stats[spec]
                s["n"] += 1
                s["per_dir"][dir_name]["n"] += 1
                if pred == gold:
                    s["correct"] += 1
                    s["per_dir"][dir_name]["correct"] += 1
                if len(examples) < 12:
                    examples.append({"q_id": line["q_id"], "cond": spec, "raw": raw,
                                     "pred": pred, "gold": gold})
        except Exception as e:
            print(f"[ERR] {line['q_id']}: {e}")
        if stats[conds[0][0]]["n"] % 200 == 0:
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "task": args.task, "axes_task": args.axes_task, "mag_task": args.mag_task,
        "feat_layers": needed_layers,
        "hook_module_idx": {L: L - 1 for L in needed_layers},
        "eval_pipeline": "lmms_gen (llava_vid-faithful generation)",
        "model_args": args.model_args, "feature_root": args.feature_root,
        "model_dir": args.model_dir, "max_new_tokens": args.max_new_tokens,
        "shard": args.shard, "num_shards": args.num_shards,
        "mag_own": {str(L): {label_list[i]: float(axes[L]["mag_own"][i])
                             for i in range(len(label_list))} for L in needed_layers},
        "mag_target": {str(L): {label_list[i]: float(axes[L]["mag_target"][i])
                                for i in range(len(label_list))} for L in needed_layers},
        "stats": stats, "examples": examples,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    for spec, _, _ in conds:
        s = stats[spec]
        acc = s["correct"] / max(s["n"], 1) * 100
        print(f"  {spec:>16s}: {acc:6.2f}%  (n={s['n']})")
    print(f"[SAVED] {args.out}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
