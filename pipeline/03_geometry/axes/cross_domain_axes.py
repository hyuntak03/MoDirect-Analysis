"""Cross-domain direction concept vectors — layer-wise axis cosine + magnitude.

For every (model, layer):
  1. Δ_d(task)      = mean(h | direction=d) − mean(h | all)   per domain (spine 3)
  2. ‖Δ_d(task)‖    magnitude per (layer, domain, direction)  ← the central quantity
  3. cos(Δ̂_d(A), Δ̂_d(B))  cross-domain axis alignment per layer, per direction,
     for all 6 domain pairs (shape_color / obj_color / shape_place / obj_place)

Generalises pipeline/03_geometry/axes/measure_invariance.py to: every layer (not 9),
configurable feature root / model dirs (not the dead /data3 literals), and the
modirect.concepts library as the single Δ definition.

Consumes answer-token dumps written by pipeline/01_extract/llm/extract_answer_features.py:
    {feature_root}/{model_dir}/answer_token/vlm_direction_testbed_R2R_4way_1500_{task}/
        features_layer_{L}.npy   (N, D) fp16 — L=0 is the embedding output,
                                 L>=1 is the output of decoder layer L-1
        labels.npy               (N,) int64 indices into meta label_list
        meta.npy                 dict with label_list (sorted: Down, Left, Right, Up)

Writes per model under --out:
    concept_vectors.npz       delta_{task} (L, 4, D) fp32 + g_{task} (L, D)
    cross_domain_axes.json    magnitudes + cross-domain cos + within-task dir cos
    fig_cross_domain_cos.png/pdf, fig_magnitude.png/pdf, fig_magnitude_per_dir.png/pdf

Usage:
  python pipeline/03_geometry/axes/cross_domain_axes.py \
      --models vanilla_qwen=llava-video-7b-qwen2_vanilla \
               baseline_v5=llava-nextvideo-7b_baseline_v5_new
"""

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np


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

from modirect.concepts import extract_concept_vectors  # noqa: E402

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_PREFIX = "vlm_direction_testbed_R2R_4way_1500_"

# Okabe–Ito CVD-safe hues, fixed per domain (identity follows the entity, never rank).
TASK_COLORS = {
    "shape_color": "#0072B2",   # blue        (in-domain)
    "obj_color":   "#009E73",   # bluish green (moderate OOD)
    "shape_place": "#D55E00",   # vermillion   (hard OOD)
    "obj_place":   "#CC79A7",   # reddish purple (hardest OOD)
}

DEFAULT_MODELS = [
    "vanilla=llava-video-7b-qwen2_vanilla",
    "baseline_v5=llava-video-7b-qwen2_baseline_v5_new",
    "channel_gate_v5=llava-video-7b-qwen2_channel_gate_v5_new",
]


def task_dir(feature_root, model_dir, task):
    return os.path.join(feature_root, model_dir, "answer_token", TASK_PREFIX + task)


def discover_layers(tdir):
    layers = []
    for f in os.listdir(tdir):
        if f.startswith("features_layer_") and f.endswith(".npy"):
            layers.append(int(f[len("features_layer_"):-len(".npy")]))
    if not layers:
        raise FileNotFoundError(f"no features_layer_*.npy under {tdir}")
    return sorted(layers)


def load_task(tdir):
    labels = np.load(os.path.join(tdir, "labels.npy"))
    meta = np.load(os.path.join(tdir, "meta.npy"), allow_pickle=True).item()
    label_list = list(meta["label_list"])
    return labels, label_list, meta


def analyze_model(name, model_dir, feature_root):
    """Returns (results dict for JSON, npz payload dict)."""
    tdirs = {t: task_dir(feature_root, model_dir, t) for t in TASKS}
    for t, d in tdirs.items():
        if not os.path.isdir(d):
            raise FileNotFoundError(f"[{name}] missing features for task {t}: {d}")

    layers = discover_layers(tdirs[TASKS[0]])
    labels, label_list, meta = {}, None, {}
    for t in TASKS:
        labels[t], ll, meta[t] = load_task(tdirs[t])
        if label_list is None:
            label_list = ll
        elif ll != label_list:
            raise ValueError(f"[{name}] label_list mismatch: {t} has {ll}, expected {label_list}")
        got = discover_layers(tdirs[t])
        if got != layers:
            raise ValueError(f"[{name}] layer set mismatch for {t}: {got} vs {layers}")
    n_dirs = len(label_list)

    # Per (task, layer): Δ (4, D), g (D,) via the library definition.
    delta = {t: [] for t in TASKS}    # list over layers of (4, D)
    g = {t: [] for t in TASKS}
    mag = {t: [] for t in TASKS}      # list over layers of (4,)
    counts = {}
    for t in TASKS:
        y = labels[t]
        counts[t] = {label_list[d]: int((y == d).sum()) for d in range(n_dirs)}
        for L in layers:
            h = np.load(os.path.join(tdirs[t], f"features_layer_{L}.npy"), mmap_mode="r")
            axes = extract_concept_vectors(np.asarray(h), y, classes=list(range(n_dirs)))
            delta[t].append(np.stack([axes.delta[d] for d in range(n_dirs)]))
            g[t].append(axes.g)
            mag[t].append(np.array([axes.mag[d] for d in range(n_dirs)], dtype=np.float32))
        delta[t] = np.stack(delta[t])   # (L, 4, D)
        g[t] = np.stack(g[t])           # (L, D)
        mag[t] = np.stack(mag[t])       # (L, 4)
        print(f"  [{name}] {t}: n={len(y)} {counts[t]}  delta {delta[t].shape}")

    def hat(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

    # Cross-domain axis cosine per layer per direction: cos(Δ̂_d(A, L), Δ̂_d(B, L)).
    cross = {}
    for tA, tB in combinations(TASKS, 2):
        c = np.einsum("lkd,lkd->lk", hat(delta[tA]), hat(delta[tB]))  # (L, 4)
        cross[f"{tA}__{tB}"] = c

    # Within-task direction-pair cosine (sanity: directions should separate).
    within = {}
    for t in TASKS:
        dh = hat(delta[t])
        within[t] = {
            f"{label_list[i]}-{label_list[j]}":
                np.einsum("ld,ld->l", dh[:, i], dh[:, j]).tolist()
            for i, j in combinations(range(n_dirs), 2)
        }

    results = {
        "model": name,
        "model_dir": model_dir,
        "feature_root": feature_root,
        "layers": layers,
        "layer_convention": "features_layer_0 = embedding output; L>=1 = output of decoder layer L-1",
        "label_list": label_list,
        "n_samples": {t: int(len(labels[t])) for t in TASKS},
        "counts": counts,
        "magnitude": {t: {label_list[d]: mag[t][:, d].tolist() for d in range(n_dirs)}
                      for t in TASKS},
        "magnitude_mean_over_dirs": {t: mag[t].mean(axis=1).tolist() for t in TASKS},
        "grand_norm": {t: np.linalg.norm(g[t], axis=-1).tolist() for t in TASKS},
        "cross_domain_cos": {
            pair: {label_list[d]: c[:, d].tolist() for d in range(n_dirs)}
            for pair, c in cross.items()
        },
        "cross_domain_cos_mean": {pair: c.mean(axis=1).tolist() for pair, c in cross.items()},
        "within_task_dir_cos": within,
    }
    npz = {}
    for t in TASKS:
        npz[f"delta_{t}"] = delta[t].astype(np.float32)
        npz[f"g_{t}"] = g[t].astype(np.float32)
        npz[f"mag_{t}"] = mag[t]
    npz["layers"] = np.array(layers)
    npz["label_list"] = np.array(label_list)
    return results, npz


# ---------------------------------------------------------------- figures

def plot_model(name, res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = res["layers"]
    label_list = res["label_list"]
    pairs = list(res["cross_domain_cos_mean"].keys())

    def style(ax):
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    # 1) Cross-domain cos heatmap: rows = domain pairs, cols = layers, mean over dirs.
    M = np.array([res["cross_domain_cos_mean"][p] for p in pairs])  # (6, L)
    fig, ax = plt.subplots(figsize=(max(7, len(layers) * 0.32), 3.2))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(pairs)),
                  [p.replace("__", " vs ") for p in pairs], fontsize=8)
    step = 2 if len(layers) > 16 else 1
    ax.set_xticks(range(0, len(layers), step), layers[::step], fontsize=8)
    ax.set_xlabel("hidden_states index (0 = embeddings)")
    ax.set_title(f"{name} — cross-domain Δ̂-axis cosine (mean over {len(label_list)} directions)")
    fig.colorbar(im, ax=ax, shrink=0.9, label="cos(Δ̂ᴬ, Δ̂ᴮ)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_cross_domain_cos.{ext}"), dpi=200)
    plt.close(fig)

    # 2) Cross-domain cos curves: one line per pair (mean over dirs).
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for p in pairs:
        ax.plot(layers, res["cross_domain_cos_mean"][p], linewidth=2,
                label=p.replace("__", " vs "))
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("hidden_states index (0 = embeddings)")
    ax.set_ylabel("cos(Δ̂ᴬ, Δ̂ᴮ)")
    ax.set_title(f"{name} — cross-domain direction-axis alignment per layer")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_cross_domain_cos_curves.{ext}"), dpi=200)
    plt.close(fig)

    # 3) Magnitude curves: mean over directions, one line per domain (fixed hues).
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for t in TASKS:
        ax.plot(layers, res["magnitude_mean_over_dirs"][t], linewidth=2,
                color=TASK_COLORS[t], label=t)
    ax.set_xlabel("hidden_states index (0 = embeddings)")
    ax.set_ylabel("‖Δ_d‖ (mean over directions)")
    ax.set_title(f"{name} — direction-signal magnitude per layer per domain")
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_magnitude.{ext}"), dpi=200)
    plt.close(fig)

    # 4) Magnitude per direction: 2x2 small multiples, shared scale.
    fig, axes_grid = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    for d, ax in zip(range(len(label_list)), axes_grid.ravel()):
        for t in TASKS:
            ax.plot(layers, res["magnitude"][t][label_list[d]], linewidth=1.8,
                    color=TASK_COLORS[t], label=t if d == 0 else None)
        ax.set_title(label_list[d], fontsize=10)
        style(ax)
    for ax in axes_grid[-1]:
        ax.set_xlabel("hidden_states index")
    for ax in axes_grid[:, 0]:
        ax.set_ylabel("‖Δ_d‖")
    fig.suptitle(f"{name} — magnitude per (layer, domain, direction)", y=0.98)
    fig.legend(*axes_grid[0, 0].get_legend_handles_labels(),
               loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_magnitude_per_dir.{ext}"), dpi=200)
    plt.close(fig)


def print_summary(name, res):
    layers = res["layers"]
    pairs = list(res["cross_domain_cos_mean"].keys())
    print(f"\n{'=' * 96}\n{name} — cross-domain Δ̂ cos (mean over dirs) & ‖Δ‖ (mean over dirs)\n{'=' * 96}")
    hdr = f"{'L':>4} | {'cos(6-pair mean)':>16} | " + " | ".join(f"{t:>12}" for t in TASKS)
    print(hdr + "   (‖Δ‖ columns)")
    for i, L in enumerate(layers):
        cos_mean = np.mean([res["cross_domain_cos_mean"][p][i] for p in pairs])
        mags = " | ".join(f"{res['magnitude_mean_over_dirs'][t][i]:>12.2f}" for t in TASKS)
        print(f"{L:>4} | {cos_mean:>16.3f} | {mags}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--feature_root", default=None,
                    help="defaults to modirect paths.yaml feature_root")
    ap.add_argument("--out", default=None,
                    help="defaults to {repo}/outputs/cross_domain_axes")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="name=feature_dirname pairs")
    args = ap.parse_args()

    feature_root = args.feature_root
    if feature_root is None:
        from modirect.config import load_paths
        feature_root = str(load_paths().feature_root)
    out_root = args.out or os.path.join(_PROJECT_ROOT, "outputs", "cross_domain_axes")

    print(f"[feature_root] {feature_root}")
    print(f"[out]          {out_root}")

    for spec in args.models:
        name, _, model_dir = spec.partition("=")
        if not model_dir:
            raise ValueError(f"--models entries must be name=dirname, got {spec!r}")
        print(f"\n[analyze] {name}  ({model_dir})")
        res, npz = analyze_model(name, model_dir, feature_root)
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "concept_vectors.npz"), **npz)
        with open(os.path.join(out_dir, "cross_domain_axes.json"), "w") as f:
            json.dump(res, f, indent=1)
        plot_model(name, res, out_dir)
        print_summary(name, res)
        print(f"[SAVED] {out_dir}/  (concept_vectors.npz, cross_domain_axes.json, fig_*.png/pdf)")


if __name__ == "__main__":
    main()
