"""Magnitude-restoration intervention — v5 generation, feature-cache axes.

Question: on the hardest OOD split (obj_place), does raising the direction-signal
magnitude to the in-domain (shape_color) level — keeping the domain's own axis,
touching nothing else — recover MCQ accuracy?

For each obj_place sample with GT direction d, at hidden_states[FEAT_LAYER]
(= output of decoder module FEAT_LAYER-1; features_layer_{L}.npy convention):

    proj = ⟨h_last − g_op, Δ̂_op,d⟩
    no_swap      h                                   (before)
    clean_sc     h − proj·Δ̂ + ‖Δ_sc,d‖·Δ̂            (after — the experiment)
    clean_op     h − proj·Δ̂ + ‖Δ_op,d‖·Δ̂            (operator sanity: no boost)
    remove_own   h − proj·Δ̂                          (control: ablate the axis)

Axes come from the answer-token feature cache analysis
(outputs/cross_domain_axes_qwen2_v5/{model}/concept_vectors.npz), NOT the committed
.pt assets — the layer convention here is the features one (L = hidden_states index).

All conditions run on the SAME forward inputs per sample (paired comparison, one
video decode per sample). Shard with --shard/--num_shards for multi-GPU.

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python magnitude_restore_v5.py \
      --axes_npz outputs/cross_domain_axes_qwen2_v5/baseline_v5/concept_vectors.npz \
      --shard 0 --num_shards 4 --out /path/shard0.json
"""

import argparse
import gc
import json
import os
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

from modirect.interventions.hooks import last_token_hook  # noqa: E402
from modirect.interventions.operators import (  # noqa: E402
    add_canon, amp, clean, full_rep, remove_own)

VIDEO_FOLDER = os.environ.get("HF_DATASETS_CACHE", "/local_datasets/vlm_direction")
VANILLA_ARGS = ("pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,"
                "conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,"
                "device_map=auto,force_sample=True")
DEFAULT_LORA = ("/data/jongseo/project/vlm/LLaVA-NeXT/4combo_v5_new/work_dirs/"
                "llava-video-7b-qwen2_baseline_shape_simple_v5_new_lora-r64_f8_ep1_lr1e-5_bs12_ga2")

#: default = the before/after pair plus its two controls. The full operator table
#: (README §conditions) is also available: amp_2x, add_canon_sc, full_rep.
CONDITIONS = ["no_swap", "clean_sc", "clean_op", "remove_own"]


def get_letter_ids(tokenizer):
    ids = {}
    for ltr in ["A", "B", "C", "D"]:
        for cand in [ltr, " " + ltr]:
            tids = tokenizer.encode(cand, add_special_tokens=False)
            if len(tids) == 1:
                ids[ltr] = tids[0]
                break
        else:
            raise RuntimeError(f"no single-token encoding for letter {ltr}")
    return ids


def load_axes(npz_path, feat_layer, axes_task, mag_task):
    """Returns (g, delta_hat per dir, mag_own per dir, mag_target per dir, label_list)."""
    z = np.load(npz_path, allow_pickle=True)
    label_list = [str(x) for x in z["label_list"]]
    layers = z["layers"].tolist()
    li = layers.index(feat_layer)
    delta_own = z[f"delta_{axes_task}"][li].astype(np.float32)      # (4, D)
    g = z[f"g_{axes_task}"][li].astype(np.float32)                  # (D,)
    mag_own = np.linalg.norm(delta_own, axis=-1)                    # (4,)
    delta_hat = delta_own / (mag_own[:, None] + 1e-9)
    mag_target = np.linalg.norm(z[f"delta_{mag_task}"][li].astype(np.float32), axis=-1)
    return g, delta_hat, mag_own, mag_target, label_list


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_args", default=f"lora_pretrained={DEFAULT_LORA},{VANILLA_ARGS}")
    ap.add_argument("--task", default="vlm_direction_testbed_R2R_4way_1500_obj_place")
    ap.add_argument("--axes_npz", required=True)
    ap.add_argument("--axes_task", default="obj_place",
                    help="task whose axis Δ̂ and g are used (the domain being intervened)")
    ap.add_argument("--mag_task", default="shape_color",
                    help="task whose ‖Δ‖ is the clean_sc target magnitude")
    ap.add_argument("--feat_layer", type=int, default=21,
                    help="hidden_states index (features_layer convention); hook = decoder module feat_layer-1")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    conditions = args.conditions.split(",")

    g_np, dhat_np, mag_own, mag_target, label_list = load_axes(
        args.axes_npz, args.feat_layer, args.axes_task, args.mag_task)
    dir_to_idx = {name.lower(): i for i, name in enumerate(label_list)}
    print(f"[axes] L{args.feat_layer} {args.axes_task}: "
          + " ".join(f"{label_list[i]}: own={mag_own[i]:.1f}->target={mag_target[i]:.1f}"
                     for i in range(len(label_list))))

    import importlib.util

    def imp(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    _dl = imp("core.dataset_loader", os.path.join(_PROJECT_ROOT, "core", "dataset_loader.py"))
    questions, _ = _dl.load_dataset_as_questions(
        task_name=args.task, hf_cache_dir=os.environ.get("HF_HOME"), limit=args.limit)
    questions = questions[args.shard::args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(questions)} samples")

    from core.model_loader import parse_model_args, load_model_from_args
    from core.data_pipeline import create_data_loader
    tokenizer, model, image_processor, _, _, conv_template = load_model_from_args(
        parse_model_args(args.model_args))
    model.eval()

    dl = create_data_loader(
        questions, "", 1, 6, tokenizer, image_processor, model.config,
        args.task, conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True)

    letter_ids = get_letter_ids(tokenizer)
    id_to_letter = {v: k for k, v in letter_ids.items()}
    letter_tok_ids = list(letter_ids.values())
    from modirect.interventions.hooks import _decoder_layers
    hook_module = _decoder_layers(model)[args.feat_layer - 1]

    dev = "cuda"
    g_t = torch.from_numpy(g_np).to(dev)
    dhat_t = torch.from_numpy(dhat_np).to(dev)

    def make_fn(cond, di):
        def fn(h_last):  # (B, D) fp16 -> fp32 math -> back
            h = h_last.float()
            if cond == "clean_sc":
                return clean(h, g_t, dhat_t[di], magnitude=float(mag_target[di]))
            if cond == "clean_op":
                return clean(h, g_t, dhat_t[di], magnitude=float(mag_own[di]))
            if cond == "remove_own":
                return remove_own(h, g_t, dhat_t[di])
            if cond == "amp_2x":
                return amp(h, g_t, dhat_t[di], factor=2.0)
            if cond == "add_canon_sc":
                return add_canon(h, g_t, dhat_t[di], magnitude=float(mag_target[di]))
            if cond == "full_rep":
                proto = g_t + dhat_t[di] * float(mag_own[di])  # g + Δ_op,d
                return full_rep(h, g_t, dhat_t[di], prototype=proto)
            raise ValueError(cond)
        return fn

    def fwd(input_ids, image_tensor, image_sizes, modality):
        (_, position_ids, attention_mask, _, inputs_embeds, _) = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, image_tensor,
                modalities=[modality], image_sizes=image_sizes)
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    position_ids=position_ids, return_dict=True)
        logits = out.logits[0, -1, :]
        return id_to_letter[letter_tok_ids[int(logits[letter_tok_ids].argmax())]]

    stats = {c: {"n": 0, "correct": 0,
                 "per_dir": {name: {"n": 0, "correct": 0} for name in label_list}}
             for c in conditions}

    n_done = 0
    for batch, line in tqdm(zip(dl, questions), total=len(questions), desc=f"shard{args.shard}"):
        if batch is None:
            continue
        input_ids, image_tensor, image_sizes, _, _, modality = batch
        input_ids = input_ids.to(dev)
        image_tensor = [t.to(dev) for t in image_tensor]
        d = str(line["direction"]).lower()
        if d not in dir_to_idx:
            continue
        di = dir_to_idx[d]
        gold = str(line["answer"]).strip().upper()
        dir_name = label_list[di]
        try:
            for cond in conditions:
                if cond == "no_swap":
                    pred = fwd(input_ids, image_tensor, image_sizes, modality)
                else:
                    handle = hook_module.register_forward_hook(
                        last_token_hook(make_fn(cond, di)))
                    try:
                        pred = fwd(input_ids, image_tensor, image_sizes, modality)
                    finally:
                        handle.remove()
                s = stats[cond]
                s["n"] += 1
                s["per_dir"][dir_name]["n"] += 1
                if pred == gold:
                    s["correct"] += 1
                    s["per_dir"][dir_name]["correct"] += 1
        except Exception as e:
            print(f"[ERR] {line['q_id']}: {e}")
        n_done += 1
        if n_done % 200 == 0:
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "task": args.task, "axes_task": args.axes_task, "mag_task": args.mag_task,
        "feat_layer": args.feat_layer, "hook_module_idx": args.feat_layer - 1,
        "model_args": args.model_args, "axes_npz": args.axes_npz,
        "shard": args.shard, "num_shards": args.num_shards,
        "mag_own": {label_list[i]: float(mag_own[i]) for i in range(len(label_list))},
        "mag_target": {label_list[i]: float(mag_target[i]) for i in range(len(label_list))},
        "stats": stats,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    for c in conditions:
        s = stats[c]
        acc = s["correct"] / max(s["n"], 1) * 100
        print(f"  {c:>12s}: {acc:6.2f}%  (n={s['n']})")
    print(f"[SAVED] {args.out}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
