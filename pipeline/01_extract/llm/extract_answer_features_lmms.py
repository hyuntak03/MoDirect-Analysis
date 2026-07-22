"""Answer-token feature extraction — lmms-eval-faithful pipeline (core/lmms_gen).

Same outputs as extract_answer_features.py (features_layer_{L}.npy, labels.npy,
qids.npy, meta.npy) but the forward matches lmms_eval/models/simple/llava_vid.py
exactly: no " \\nAnswer the question using a single word or phrase." suffix,
bfloat16 + sdpa, llava_vid.load_video frame sampling.

Axes computed from this cache are the ones magnitude_restore_gen.py must use —
never mix them with the core/data_pipeline cache (different prompt distribution).

Usage:
  CUDA_VISIBLE_DEVICES=0 python extract_answer_features_lmms.py \
      --model_args "lora_pretrained=...,pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=8,force_sample=True,torch_dtype=bfloat16" \
      --task vlm_direction_testbed_R2R_4way_1500_obj_place \
      --output_dir <cache>/<model>/answer_token/<task>
"""

import argparse
import importlib.util
import os
import string
import sys
from concurrent.futures import ThreadPoolExecutor

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


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_answer(line):
    answer = str(line["answer"]).strip()
    if len(answer) == 1 and answer.upper() in string.ascii_uppercase:
        import ast
        cands = line.get("candidates", [])
        if isinstance(cands, str):
            cands = ast.literal_eval(cands)
        idx = ord(answer.upper()) - ord("A")
        if idx < len(cands):
            answer = str(cands[idx]).strip()
    return answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_args", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    _ml = _imp("core.model_loader", os.path.join(_PROJECT_ROOT, "core", "model_loader.py"))
    ma = _ml.parse_model_args(args.model_args)
    from core import lmms_gen

    _dl = _imp("core.dataset_loader", os.path.join(_PROJECT_ROOT, "core", "dataset_loader.py"))
    questions, _ = _dl.load_dataset_as_questions(
        task_name=args.task, hf_cache_dir=os.environ.get("HF_HOME"), limit=args.limit)

    torch_dtype = ma.get("torch_dtype", "bfloat16")
    tokenizer, model, image_processor, _, model_name = lmms_gen.load_model_lmms(
        ma.get("pretrained"), ma.get("lora_pretrained"),
        torch_dtype=torch_dtype,
        attn_implementation=ma.get("attn_implementation", "sdpa"),
        device_map=ma.get("device_map", "cuda:0"))
    conv_template = ma.get("conv_template", "qwen_1_5")
    max_frames_num = int(ma.get("max_frames_num", 8))
    force_sample = bool(ma.get("force_sample", False))
    video_fps = int(ma.get("video_fps", 1))

    num_layers = model.config.num_hidden_layers + 1
    label_list = sorted({resolve_answer(q) for q in questions})
    answer_to_idx = {a: i for i, a in enumerate(label_list)}
    print(f"[INFO] classes: {label_list}  layers: {num_layers}  n: {len(questions)}")

    feats = {l: [] for l in range(num_layers)}
    labels, qids = [], []
    for line in tqdm(questions, desc=f"lmms-extract {args.task}"):
        try:
            frames, _, _ = lmms_gen.load_video(
                line["video"], max_frames_num, fps=video_fps, force_sample=force_sample)
            video = lmms_gen.preprocess_video(image_processor, frames, torch_dtype)
            input_ids, attention_mask, _ = lmms_gen.build_prompt_inputs(
                tokenizer, line["question"], conv_template)
            hs = lmms_gen.prefill_hidden_states(model, input_ids, attention_mask, [video])
            stack = torch.stack([hs[l][0, -1, :] for l in range(num_layers)]).cpu().to(torch.float16)
        except Exception as e:
            print(f"[WARN] {line['q_id']} 실패 (스킵): {e}")
            continue
        for l in range(num_layers):
            feats[l].append(stack[l])
        labels.append(answer_to_idx[resolve_answer(line)])
        qids.append(line["q_id"])
        if len(labels) % 200 == 0:
            torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "labels.npy"), np.array(labels, dtype=np.int64))
    np.save(os.path.join(args.output_dir, "qids.npy"), np.array(qids))

    def _save(l):
        np.save(os.path.join(args.output_dir, f"features_layer_{l}.npy"),
                torch.stack(feats[l]).numpy())

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_save, range(num_layers)))
    meta = {"num_layers": num_layers, "num_samples": len(labels),
            "num_classes": len(label_list), "label_list": label_list,
            "model_name": model_name, "task": args.task,
            "hidden_dim": model.config.hidden_size, "token_type": "answer",
            "pipeline": "lmms_gen", "torch_dtype": torch_dtype}
    np.save(os.path.join(args.output_dir, "meta.npy"), meta)
    print(f"[DONE] {len(labels)} samples x {num_layers} layers -> {args.output_dir}")
    print(f"  labels dist: {np.bincount(np.array(labels), minlength=len(label_list)).tolist()}")


if __name__ == "__main__":
    main()
