"""Create dedicated plots for the grouped 500-program test pipeline."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "plots" / "benchmark_v2"
OUTPUT.mkdir(parents=True, exist_ok=True)


def save_category_split():
    manifest = json.loads((ROOT / "datasets" / "benchmark_manifest.json").read_text(encoding="utf-8"))
    categories = manifest["category_counts"]
    labels = [category.replace("_", " ").title() for category in categories]
    train = [categories[category]["training"] for category in categories]
    test = [categories[category]["testing"] for category in categories]
    positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(positions - 0.18, train, width=0.36, label="Training", color="#6388db")
    axis.bar(positions + 0.18, test, width=0.36, label="Testing", color="#dc9a55")
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel("Programs")
    axis.set_title("Grouped benchmark category balance")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_category_train_test.png", dpi=150)
    plt.close(fig)


def save_model_metrics():
    evaluation = json.loads((RESULTS / "benchmark_v2_ml_evaluation.json").read_text(encoding="utf-8"))
    passes = evaluation["per_pass"]
    names = list(passes)
    labels = [name.replace("_", "\n") for name in names]
    positions = np.arange(len(names))
    width = 0.24
    fig, axis = plt.subplots(figsize=(11, 5))
    for offset, metric, color in (
        (-width, "precision", "#5e9b8f"),
        (0, "recall", "#6584c6"),
        (width, "f1", "#b183cc"),
    ):
        axis.bar(
            positions + offset,
            [passes[name][metric] for name in names],
            width=width,
            label=metric.title(),
            color=color,
        )
    axis.set_xticks(positions, labels, fontsize=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Held-out test score")
    axis.set_title("ML applicability recommendation metrics by pass")
    axis.legend(frameon=False, ncols=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_ml_metrics_by_pass.png", dpi=150)
    plt.close(fig)

    categories = evaluation["per_category"]
    names = list(categories)
    labels = [name.replace("_", " ").title() for name in names]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        labels,
        [categories[name]["aggregate"]["f1"] for name in names],
        color="#8169bb",
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Micro F1")
    axis.set_title("Held-out ML metrics by category")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "03_ml_f1_by_category.png", dpi=150)
    plt.close(fig)


def save_compiler_comparison():
    compiler = json.loads((RESULTS / "benchmark_v2_compiler_results.json").read_text(encoding="utf-8"))
    labels = ["ML-recommended", "All six passes"]
    reductions = []
    output_matches = []
    cycles = []
    for mode in ("ml_recommended", "all_passes_baseline"):
        values = [item[mode] for item in compiler]
        reductions.append(np.mean([
            (entry["instructions_before"] - entry["instructions_after"]) / entry["instructions_before"] * 100
            if entry.get("instructions_before") else 0.0
            for entry in values
        ]))
        output_matches.append(sum(entry["output_equivalent"] and entry["expected_output_match"] for entry in values))
        cycles.append(np.mean([
            entry["optimization_cycles"] for entry in values
            if entry["optimization_cycles"] is not None
        ]) if any(entry["optimization_cycles"] is not None for entry in values) else 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(labels, reductions, color=["#6689de", "#5e9b8f"])
    axes[0].set_ylabel("Average instruction reduction (%)")
    axes[0].set_title("Compiler metrics")
    axes[1].bar(labels, output_matches, color=["#6689de", "#5e9b8f"])
    axes[1].set_ylabel("Programs with matching outputs")
    axes[1].set_title("Output verification")
    axes[2].bar(labels, cycles, color=["#6689de", "#5e9b8f"])
    axes[2].set_ylabel("Mean cycles")
    axes[2].set_title("Convergence")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=12)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Independent test-set compiler runs — not ML accuracy")
    fig.tight_layout()
    fig.savefig(OUTPUT / "04_compiler_ml_vs_all_passes.png", dpi=150)
    plt.close(fig)


def main():
    save_category_split()
    save_model_metrics()
    save_compiler_comparison()
    print(f"Saved grouped-benchmark plots under {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
