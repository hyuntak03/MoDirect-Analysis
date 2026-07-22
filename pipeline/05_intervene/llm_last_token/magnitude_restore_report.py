"""Merge magnitude_restore_v5.py shard outputs into one report + figure.

Reads every shard*.json under --run_dir (written by magnitude_restore_v5.py),
merges the paired per-condition counts, prints the before/after table with the
per-direction breakdown, and writes:

    {run_dir}/result.json          merged counts + accuracy + Δpp vs no_swap
    {run_dir}/fig_accuracy.png/pdf bar chart (chance line at 25%)

Usage:
  python magnitude_restore_report.py --run_dir outputs/interventions_qwen2_v5/magnitude_restore_op_L21
"""

import argparse
import glob
import json
import os


def merge(shards):
    conds = list(shards[0]["stats"].keys())
    dirs = list(shards[0]["stats"][conds[0]]["per_dir"].keys())
    m = {c: {"n": 0, "correct": 0,
             "per_dir": {d: {"n": 0, "correct": 0} for d in dirs}} for c in conds}
    for sh in shards:
        for c in conds:
            s = sh["stats"][c]
            m[c]["n"] += s["n"]
            m[c]["correct"] += s["correct"]
            for d in dirs:
                m[c]["per_dir"][d]["n"] += s["per_dir"][d]["n"]
                m[c]["per_dir"][d]["correct"] += s["per_dir"][d]["correct"]
    return m, conds, dirs


def plot(acc, conds, title, out_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # canonical presentation order: ablation -> before -> no-boost -> the experiment
    pref = ["remove_own", "no_swap", "clean_op", "amp_2x", "add_canon_sc",
            "clean_sc", "full_rep"]
    order = [c for c in pref if c in conds] + [c for c in conds if c not in pref]
    labels = {"remove_own": "remove_own\n(axis ablated)", "no_swap": "no_swap\n(before)",
              "clean_op": "clean @ ‖Δ_op‖\n(no boost)", "clean_sc": "clean @ ‖Δ_sc‖\n(after)",
              "amp_2x": "amp_2x", "add_canon_sc": "add_canon\n@ ‖Δ_sc‖",
              "full_rep": "full_rep\n(prototype)"}
    colors = {c: "#8b95a1" for c in order}
    colors.update({"no_swap": "#5c6670", "clean_sc": "#0072B2", "remove_own": "#b3bac2"})
    # multi-layer runs name conditions "clean_sc@20+21" — highlight the widest one
    gen_cleans = [c for c in order if c.startswith("clean_sc@")]
    if gen_cleans:
        colors[max(gen_cleans, key=len)] = "#0072B2"

    fig, ax = plt.subplots(figsize=(max(6.4, 1.55 * len(order)), 4))
    xs = range(len(order))
    vals = [acc[c] for c in order]
    ax.bar(xs, vals, width=0.62, color=[colors[c] for c in order])
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.2, f"{v:.1f}%", ha="center", fontsize=10)
    ax.axhline(25, color="0.55", linewidth=0.9, linestyle="--")
    ax.text(len(order) - 0.42, 26.2, "chance 25%", fontsize=8, color="0.35")
    ax.set_xticks(list(xs), [labels.get(c, c) for c in order], fontsize=9)
    ax.set_ylabel("MCQ accuracy (%)")
    ax.set_ylim(0, max(vals) + 9)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="directory holding shard*.json from magnitude_restore_v5.py")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.run_dir, "shard*.json")))
    if not paths:
        raise FileNotFoundError(f"no shard*.json under {args.run_dir}")
    shards = [json.load(open(p)) for p in paths]
    merged, conds, dirs = merge(shards)

    acc = {c: merged[c]["correct"] / max(merged[c]["n"], 1) * 100 for c in conds}
    base = acc.get("no_swap", 0.0)
    meta_keys = ("task", "axes_task", "mag_task", "feat_layer", "feat_layers",
                 "hook_module_idx", "eval_pipeline", "feature_root", "model_dir",
                 "max_new_tokens", "mag_own", "mag_target", "axes_npz", "model_args")
    meta = {k: shards[0][k] for k in meta_keys if k in shards[0]}
    feat_label = meta.get("feat_layer") or "+".join(
        str(x) for x in meta.get("feat_layers", []))

    print(f"\n=== magnitude restore @ feat L{feat_label} — "
          f"{meta.get('axes_task')} (n={merged[conds[0]]['n']}, {len(paths)} shards) ===")
    print(f"mag: own={meta.get('mag_own')}  ->  target({meta.get('mag_task')})={meta.get('mag_target')}")
    for c in conds:
        per = "  ".join(
            f"{d}:{merged[c]['per_dir'][d]['correct'] / max(merged[c]['per_dir'][d]['n'], 1) * 100:5.1f}%"
            for d in dirs)
        print(f"  {c:>14s}: {acc[c]:6.2f}%  Δ={acc[c] - base:+6.2f}pp   [{per}]")

    result = {"meta": meta, "accuracy_pct": acc,
              "delta_pp": {c: acc[c] - base for c in conds}, "merged": merged}
    out_json = os.path.join(args.run_dir, "result.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)

    title = (f"baseline_v5 · {meta.get('axes_task')} · feat L{feat_label}"
             " — keep axis, set magnitude")
    plot(acc, conds, title, os.path.join(args.run_dir, "fig_accuracy"))
    print(f"[SAVED] {out_json} + fig_accuracy.png/pdf")


if __name__ == "__main__":
    main()
