"""Render source-backed FMAT and protected OASIS figures without changing analyses."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 -- registers the requested scientific styles
import numpy as np
import pandas as pd

from vlanchor.campaign import create_run, append_event
from vlanchor.io import write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["fmat", "oasis", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    sources = [args.fmat/"matched-correlations.csv", args.oasis/"direct-correlations.csv",
        args.oasis/"paired-differences.csv", args.oasis/"aligned-direct-scores.npz"]
    create_run(args.output, {"purpose": "Scientific figures from frozen numerical results",
        "source_sha256": {str(p): file_hash(p) for p in sources},
        "script_sha256": file_hash(Path(__file__)), "matplotlib": matplotlib.__version__,
        "scatter_lines": "Descriptive least-squares lines only; not predictive evaluation or method fitting"})
    plt.style.use(['science', 'no-latex', 'bright'])
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

    def save(fig, name):
        fig.savefig(args.output/f"{name}.pdf", bbox_inches="tight")
        fig.savefig(args.output/f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fmat = pd.read_csv(args.fmat/"matched-correlations.csv")
    fmat = fmat[fmat.subset == "all50"]
    order = fmat[fmat.study == "d1a"].method.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True, layout="constrained")
    for ax, study, title in zip(axes, ["d1a", "d1b"], ["Occupations (50 targets)", "Names (50 targets)"]):
        frame = fmat[fmat.study == study].set_index("method").loc[order]
        for i, (method, row) in enumerate(frame.iterrows()):
            color = "#176B87" if method.startswith("qwen") else "#65717B"
            ax.errorbar(row.pearson_r, i, xerr=[[row.pearson_r-row.pearson_ci_lower], [row.pearson_ci_upper-row.pearson_r]],
                fmt="o", color=color, capsize=2, markersize=4)
        ax.set(title=title, xlabel="Signed Pearson correlation", xlim=(0, 1))
        ax.grid(axis="x", alpha=.15)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].invert_yaxis()
    fig.suptitle("FMAT matched external criterion: all stored and newly evaluated methods")
    save(fig, "fmat-matched-validity")
    correlations = pd.read_csv(args.oasis/"direct-correlations.csv")
    methods = ["VLanchor", "qwen-embedding", "qwen-reranker"]
    labels = ["VLanchor / Qwen3.5-4B", "Official embedding", "Official reranker"]
    colors = ["#176B87", "#929DA6", "#B16A36"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, layout="constrained")
    for ax, dimension, title in zip(axes, ["V", "A"], ["Valence", "Arousal"]):
        frame = correlations[(correlations.dimension == dimension) & (correlations.metric == "Pearson")].set_index("method")
        for i, (method, color) in enumerate(zip(methods, colors)):
            row = frame.loc[method]
            ax.errorbar(row.correlation, i, xerr=[[row.correlation-row.ci_lower], [row.ci_upper-row.correlation]],
                fmt="o", color=color, capsize=3, markersize=6)
            ax.annotate(f"{row.correlation:.3f}", (row.correlation, i), xytext=(0, 9), textcoords="offset points", ha="center")
        ax.set(title=title, xlabel="Pearson correlation with human ratings", xlim=(0, 1), ylim=(-.5, 2.6))
        ax.grid(axis="x", alpha=.15)
    axes[0].set_yticks(range(3), labels)
    axes[0].invert_yaxis()
    fig.suptitle("OASIS protected evaluation: 201 images, paired image-bootstrap 95% intervals")
    save(fig, "oasis-protected-validity")
    with np.load(args.oasis/"aligned-direct-scores.npz", allow_pickle=False) as data:
        gold = data["human"].copy()
        scores = {name: data[name].copy() for name in methods}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), layout="constrained")
    for j, dimension in enumerate(["V", "A"]):
        for k, (method, color) in enumerate(zip(methods, colors)):
            ax = axes[j, k]
            x, y = gold[:, j], scores[method][:, j]
            ax.scatter(x, y, s=10, alpha=.5, color=color, edgecolors="none")
            coef = np.polyfit(x, y, 1)
            grid = np.array([x.min(), x.max()])
            ax.plot(grid, np.polyval(coef, grid), color=color, linewidth=1.5)
            row = correlations[(correlations.method == method) & (correlations.dimension == dimension) & (correlations.metric == "Pearson")].iloc[0]
            ax.text(.04, .92, f"r = {row.correlation:.3f}", transform=ax.transAxes)
            ax.set_xlabel("Human "+("valence" if j == 0 else "arousal"))
            ax.set_ylabel("Fixed bipolar score contrast")
            if j == 0:
                ax.set_title(labels[k])
    fig.suptitle("OASIS protected images: direct scores without a trained prediction layer")
    save(fig, "oasis-protected-scatter")
    write_json(args.output/"captions.json", {
        "fmat-matched-validity": "Signed correlations with published male proportions. Same50targets, method/template sample-SD standardization then template mean. Intervals conditional on fixed50-target normalization; 12stored and5newmethods allshown. Different backbones preclude attributing differences solely to scoring.",
        "oasis-protected-validity": "Unchanged fixed bipolar anchors and3nativebridges.201protectedimages.Individual95%pairedsourcebootstrapintervals do not themselves test differences; use the paired-differences table. Pearson is primary in this frozen supplement, Spearman also reported.",
        "oasis-protected-scatter": "All201protectedimages displayed. Score axes differ by method and are not comparable in magnitude. Lines are descriptive. No image or rating was excluded based on model performance."})
    append_event(args.output, "completed")


if __name__ == "__main__":
    main()
