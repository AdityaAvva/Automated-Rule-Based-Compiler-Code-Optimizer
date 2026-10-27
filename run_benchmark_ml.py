"""Generate, label, train, evaluate, and benchmark the grouped 500-case suite."""

import csv
import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
TRAINING_FILE = DATASETS / "training" / "programs.json"
TESTING_FILE = DATASETS / "testing" / "programs.json"
MANIFEST_FILE = DATASETS / "benchmark_manifest.json"
BENCHMARK_MANIFEST_FILE = ROOT / "benchmarks" / "manifest.json"
BENCHMARK_DATASET_FILE = ROOT / "benchmarks" / "dataset.json"
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots" / "benchmark_v2"
EXE = ROOT / ("compiler_optimizer.exe" if sys.platform == "win32" else "compiler_optimizer")
PASSES = [
    "constant_folding",
    "constant_propagation",
    "algebraic_simplification",
    "common_subexpression_elimination",
    "dead_code_elimination",
    "strength_reduction",
]
PASS_LABELS = {
    "constant_folding": "Constant Folding",
    "constant_propagation": "Constant Propagation",
    "algebraic_simplification": "Algebraic Simplification",
    "common_subexpression_elimination": "Common Subexpression Elimination",
    "dead_code_elimination": "Dead Code Elimination",
    "strength_reduction": "Strength Reduction",
}
OPTIMIZER_TIMEOUT_SECONDS = 20
sys.path.insert(0, str(ROOT / "ml"))
from benchmark_features import FEATURE_NAMES, extract_features
from train_benchmark_model import train


def _section(output, start, end):
    match = re.search(re.escape(start) + r"\n(.*?)" + re.escape(end), output, re.S)
    return match.group(1) if match else None


def _strip_convergence(text):
    if text is None:
        return None
    return "\n".join(
        line for line in text.splitlines()
        if not line.startswith("convergence_")
    ).rstrip()


def _parse_metrics(output):
    return {
        key: int(value)
        for key, value in re.findall(r"^([a-z_]+)=(-?\d+)$", output, re.M)
    }


def _run_compiler(source, passes, trace=False):
    if not EXE.is_file():
        raise RuntimeError(f"Compile src\\main.cpp first; optimizer not found: {EXE}")
    with tempfile.TemporaryDirectory(prefix="benchmark-v2-") as directory:
        source_path = Path(directory) / "program.src"
        source_path.write_text(source, encoding="utf-8")
        command = [str(EXE)]
        if trace:
            command.append("--trace-convergence")
        command.extend(["--passes", ",".join(passes), str(source_path)])
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=OPTIMIZER_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{source_path.name}: compiler exceeded {OPTIMIZER_TIMEOUT_SECONDS}s."
            ) from exc
    return result


def _run_record(record, passes, trace=False):
    result = _run_compiler(record["source"], passes, trace)
    original_tac = _section(result.stdout, "=== ORIGINAL TAC ===", "=== OPTIMIZED TAC ===")
    optimized_tac = _strip_convergence(
        _section(result.stdout, "=== OPTIMIZED TAC ===", "=== OUTPUT CHECK ===")
    )
    original_output = _section(result.stdout, "Original output:", "Optimized output:")
    optimized_output = _section(result.stdout, "Optimized output:", "Verdict:")
    verdict = re.search(r"^Verdict: (PASS|FAIL)$", result.stdout, re.M)
    if result.returncode != 0 or None in (original_tac, optimized_tac, original_output, optimized_output) or not verdict:
        raise RuntimeError(
            f"{record['id']}: compiler failed ({result.returncode}): "
            f"{result.stderr.strip() or result.stdout[-1200:]}"
        )
    metrics = _parse_metrics(result.stdout)
    rewrites = {name: metrics.get(name, 0) for name in PASSES}
    isolated = {
        name: {
            "rewrite_count": metrics.get(f"{name}_transformations", 0),
            "output_equivalent": metrics.get(f"{name}_output_match", 0) == 1,
        }
        for name in PASSES
    }
    convergence = [
        {"iteration": int(iteration), "instructions": int(instructions), "stable": stable == "1"}
        for iteration, instructions, stable in re.findall(
            r"^convergence_iteration=(\d+),instructions=(\d+),stable=(\d+)$",
            result.stdout,
            re.M,
        )
    ]
    for step in convergence:
        step["changed"] = step["iteration"] > 0 and not step["stable"]
    stable_iterations = [step["iteration"] for step in convergence if step["stable"]]
    expected_match = original_output.strip() == record["expected_output"].strip()
    return {
        "return_code": result.returncode,
        "original_tac": original_tac.rstrip(),
        "optimized_tac": optimized_tac.rstrip(),
        "original_output": original_output,
        "optimized_output": optimized_output,
        "verdict": verdict.group(1),
        "output_equivalent": verdict.group(1) == "PASS" and original_output == optimized_output,
        "expected_output_match": expected_match,
        "metrics": metrics,
        "actual_rewrite_counts": rewrites,
        "isolated_passes": isolated,
        "convergence_trace": convergence,
        "optimization_cycles": stable_iterations[0] if stable_iterations else None,
        "stderr": result.stderr.strip(),
    }


def _load_programs(path, expected_split):
    programs = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads(BENCHMARK_MANIFEST_FILE.read_text(encoding="utf-8"))
    metadata = manifest.get("program_metadata", [])
    metadata_by_id = {item["id"]: item for item in metadata}
    canonical = json.loads(BENCHMARK_DATASET_FILE.read_text(encoding="utf-8"))
    canonical_by_id = {item["id"]: item for item in canonical}

    if expected_split == "training":
        if len(programs) != 400 or any(
            set(record) != {"id", "category", "size", "source", "expected_output", "inputs"}
            for record in programs
        ):
            raise ValueError(f"{path} must contain 400 records with exactly six training fields.")
        result = []
        for record in programs:
            item = metadata_by_id.get(record["id"])
            if not item or item.get("split") != "training":
                raise ValueError(f"{record['id']}: missing training-only split metadata.")
            source_hash = hashlib.sha256(record["source"].encode("utf-8")).hexdigest()
            if source_hash != item.get("source_sha256"):
                raise ValueError(f"{record['id']}: training source does not match manifest.")
            result.append({**record, "family_id": item["family_id"], "split": "training"})
        return result

    if expected_split != "testing" or len(programs) != 100 or any(
        set(record) != {"source"} for record in programs
    ):
        raise ValueError(f"{path} must contain 100 source-only testing records.")
    metadata_by_hash = {item["source_sha256"]: item for item in metadata}
    result = []
    for record in programs:
        source_hash = hashlib.sha256(record["source"].encode("utf-8")).hexdigest()
        item = metadata_by_hash.get(source_hash)
        if not item or item.get("split") != "testing":
            raise ValueError("Testing source is missing testing-only split metadata.")
        benchmark = canonical_by_id.get(item["id"])
        if benchmark is None or benchmark["source"] != record["source"]:
            raise ValueError(f"{item['id']}: testing source does not match benchmark data.")
        result.append({
            **benchmark,
            "family_id": item["family_id"],
            "split": "testing",
        })
    return result


def label_programs(programs):
    analyses = {}
    for record in programs:
        baseline = _run_record(record, [], trace=True)
        if not baseline["expected_output_match"]:
            raise RuntimeError(
                f"{record['id']}: independently generated expected output does not match "
                "the original compiler execution."
            )
        labels = {}
        evidence = {}
        for name in PASSES:
            pass_evidence = baseline["isolated_passes"][name]
            applicable = pass_evidence["rewrite_count"] > 0 and pass_evidence["output_equivalent"]
            labels[name] = int(applicable)
            evidence[name] = {
                **pass_evidence,
                "applicable": bool(applicable),
            }
        analyses[record["id"]] = {
            "applicability_labels": labels,
            "label_evidence": evidence,
            "features": extract_features(record["source"], baseline["original_tac"]),
            "original_tac_instruction_count": baseline["metrics"].get("instructions_before", 0),
            "original_output": baseline["original_output"].strip(),
        }
    return analyses


def _predict(bundle, record, analysis):
    row = np.asarray(analysis["features"], dtype=float).reshape(1, -1)
    predictions = {}
    for name in PASSES:
        model = bundle["models"][name]
        if isinstance(model, dict):
            score = float(model["constant"])
            recommended = bool(model["constant"])
        else:
            probabilities = model.predict_proba(row)[0]
            classes = list(model.classes_)
            score = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
            recommended = score >= bundle["thresholds"][name]
        predictions[name] = {
            "recommended": recommended,
            "score": round(score, 6),
            "threshold": bundle["thresholds"][name],
        }
    return predictions


def _confusion(truth, prediction):
    truth = np.asarray(truth, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(truth) if len(truth) else 0.0
    return {
        "confusion_matrix": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
        "support": len(truth),
        "positive_support": int(truth.sum()),
        "predicted_positive": int(prediction.sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_model(bundle, test_records, analysis):
    training_ids = set(bundle["training_ids"])
    test_ids = {record["id"] for record in test_records}
    training_families = set(bundle["training_families"])
    test_families = {record["family_id"] for record in test_records}
    if training_ids & test_ids or training_families & test_families:
        raise ValueError("Training/test IDs or generation families overlap.")
    if any(record.get("split") != "testing" for record in test_records):
        raise ValueError("Evaluation accepts only held-out testing examples.")
    if set(analysis) != test_ids:
        raise ValueError("Compiler-derived testing analysis does not match held-out IDs.")

    rows = []
    truth_rows = []
    prediction_rows = []
    probabilities = []
    for record in test_records:
        truth = [int(analysis[record["id"]]["applicability_labels"][name]) for name in PASSES]
        recommendations = _predict(bundle, record, analysis[record["id"]])
        prediction = [int(recommendations[name]["recommended"]) for name in PASSES]
        rows.append({
            "id": record["id"],
            "category": record["category"],
            "family_id": record["family_id"],
            "true_labels": dict(zip(PASSES, truth)),
            "recommendations": recommendations,
            "predicted_labels": dict(zip(PASSES, prediction)),
        })
        truth_rows.append(truth)
        prediction_rows.append(prediction)
        probabilities.append([recommendations[name]["score"] for name in PASSES])
    y_true = np.asarray(truth_rows, dtype=int)
    y_pred = np.asarray(prediction_rows, dtype=int)
    flat = _confusion(y_true.reshape(-1), y_pred.reshape(-1))
    pass_metrics = {
        name: _confusion(y_true[:, index], y_pred[:, index])
        for index, name in enumerate(PASSES)
    }
    category_metrics = {}
    for category in sorted({record["category"] for record in test_records}):
        indices = [index for index, record in enumerate(test_records) if record["category"] == category]
        category_truth = y_true[indices]
        category_pred = y_pred[indices]
        category_metrics[category] = {
            "program_count": len(indices),
            "per_pass": {
                name: _confusion(category_truth[:, column], category_pred[:, column])
                for column, name in enumerate(PASSES)
            },
            "aggregate": _confusion(category_truth.reshape(-1), category_pred.reshape(-1)),
            "subset_accuracy": float(np.mean(np.all(category_truth == category_pred, axis=1))),
        }
    precision_values = [value["precision"] for value in pass_metrics.values()]
    recall_values = [value["recall"] for value in pass_metrics.values()]
    f1_values = [value["f1"] for value in pass_metrics.values()]
    evaluation = {
        "evaluation_split": "testing",
        "test_program_count": len(test_records),
        "training_program_count": len(bundle["training_ids"]),
        "test_ids": sorted(test_ids),
        "training_ids": sorted(training_ids),
        "test_families": sorted(test_families),
        "training_families": sorted(training_families),
        "training_test_id_overlap": sorted(training_ids & test_ids),
        "training_test_family_overlap": sorted(training_families & test_families),
        "overall_pass_recommendation_accuracy": flat["accuracy"],
        "overall_precision_micro": flat["precision"],
        "overall_recall_micro": flat["recall"],
        "overall_f1_micro": flat["f1"],
        "overall_precision_macro": float(np.mean(precision_values)),
        "overall_recall_macro": float(np.mean(recall_values)),
        "overall_f1_macro": float(np.mean(f1_values)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(np.mean(y_true != y_pred)),
        "aggregate_confusion_matrix": flat["confusion_matrix"],
        "per_pass": pass_metrics,
        "per_category": category_metrics,
        "predictions": rows,
        "evaluation_note": "All metrics in this report are computed exclusively on held-out programs whose full generation families are absent from training.",
    }
    return evaluation


def _mode_metrics(run):
    return {
        "verdict": run["verdict"],
        "output_equivalent": run["output_equivalent"],
        "expected_output_match": run["expected_output_match"],
        "instructions_before": run["metrics"].get("instructions_before"),
        "instructions_after": run["metrics"].get("instructions_after"),
        "operations_before": run["metrics"].get("operations_before"),
        "operations_after": run["metrics"].get("operations_after"),
        "temporaries_before": run["metrics"].get("temporaries_before"),
        "temporaries_after": run["metrics"].get("temporaries_after"),
        "code_size_before": run["metrics"].get("code_size_before"),
        "code_size_after": run["metrics"].get("code_size_after"),
        "optimization_time_microseconds": run["metrics"].get("optimization_time_microseconds"),
        "optimization_cycles": run["optimization_cycles"],
        "actual_rewrite_counts": run["actual_rewrite_counts"],
        "original_output": run["original_output"].strip(),
        "optimized_output": run["optimized_output"].strip(),
        "original_tac": run["original_tac"],
        "optimized_tac": run["optimized_tac"],
        "compiler_stderr": run["stderr"],
    }


def _mean(values):
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def benchmark_testing_programs(bundle, testing_records, analysis):
    results = []
    failures = []
    recommendation_rows = []
    for index, record in enumerate(testing_records, start=1):
        recommendations = _predict(bundle, record, analysis[record["id"]])
        selected = [name for name in PASSES if recommendations[name]["recommended"]]
        recommended_run = _run_record(record, selected, trace=True)
        baseline_run = _run_record(record, PASSES, trace=True)
        for label, run in (("recommended", recommended_run), ("all_passes", baseline_run)):
            if not run["expected_output_match"] or not run["output_equivalent"]:
                failures.append({
                    "id": record["id"],
                    "mode": label,
                    "expected_output": record["expected_output"].strip(),
                    "original_output": run["original_output"].strip(),
                    "optimized_output": run["optimized_output"].strip(),
                    "stderr": run["stderr"],
                })
        result = {
            "id": record["id"],
            "category": record["category"],
            "family_id": record["family_id"],
            "split": record["split"],
            "size": record["size"],
            "applicability_labels": analysis[record["id"]]["applicability_labels"],
            "ml_recommendations": recommendations,
            "selected_passes": selected,
            "ml_recommended": _mode_metrics(recommended_run),
            "all_passes_baseline": _mode_metrics(baseline_run),
        }
        results.append(result)
        recommendation_rows.append({
            "id": record["id"],
            "category": record["category"],
            "recommendations": recommendations,
        })
        if index % 10 == 0 or index == len(testing_records):
            print(f"Compiler evaluation: {index}/{len(testing_records)} held-out programs")
    return results, failures


def _save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def _write_compiler_csv(results):
    rows = []
    for result in results:
        row = {
            "id": result["id"],
            "category": result["category"],
            "family_id": result["family_id"],
            "size": result["size"],
            "selected_passes": ";".join(result["selected_passes"]),
            "recommendation_passes": ";".join(
                name for name, value in result["ml_recommendations"].items()
                if value["recommended"]
            ),
        }
        for mode in ("ml_recommended", "all_passes_baseline"):
            for key, value in result[mode].items():
                if key in {"actual_rewrite_counts", "original_tac", "optimized_tac", "original_output", "optimized_output", "compiler_stderr"}:
                    continue
                row[f"{mode}_{key}"] = value
            for name, count in result[mode]["actual_rewrite_counts"].items():
                row[f"{mode}_{name}_actual_rewrites"] = count
        rows.append(row)
    path = RESULTS / "benchmark_v2_compiler_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _save_manifest(training_records, testing_records, analyses, failures):
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    label_counts = {}
    for split, records in (("training", training_records), ("testing", testing_records)):
        label_counts[split] = {
            name: sum(analyses[record["id"]]["applicability_labels"][name] for record in records)
            for name in PASSES
        }
    manifest.update({
        "label_counts": label_counts,
        "all_programs_supported_by_compiler": True,
        "all_expected_outputs_match_original_compiler": True,
        "compiler_labeling_completed": True,
        "training_data_path": "datasets/training/programs.json",
        "testing_data_path": "datasets/testing/programs.json",
        "model_path": "ml/benchmark_pass_recommender.joblib",
        "evaluation_path": "results/benchmark_v2_ml_evaluation.json",
        "compiler_results_path": "results/benchmark_v2_compiler_results.json",
        "compiler_failures": failures,
    })
    _save_json(MANIFEST_FILE, manifest)

    benchmark_manifest = json.loads(BENCHMARK_MANIFEST_FILE.read_text(encoding="utf-8"))
    benchmark_manifest.update({
        "compiler_labeling_completed": True,
        "label_counts": label_counts,
        "all_programs_supported_by_compiler": True,
        "all_expected_outputs_match_original_compiler": True,
        "evaluation_path": "results/benchmark_v2_ml_evaluation.json",
        "compiler_results_path": "results/benchmark_v2_compiler_results.json",
        "model_path": "ml/benchmark_pass_recommender.joblib",
        "compiler_failures": failures,
    })
    _save_json(BENCHMARK_MANIFEST_FILE, benchmark_manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Use the already-generated canonical benchmarks/dataset.json split.",
    )
    args = parser.parse_args()
    if not EXE.is_file():
        raise SystemExit(f"Compile src\\main.cpp first; optimizer not found: {EXE}")
    if not args.skip_generation:
        subprocess.run(
            [sys.executable, str(ROOT / "generate_dataset.py")],
            cwd=ROOT,
            check=True,
        )
    training = _load_programs(TRAINING_FILE, "training")
    testing = _load_programs(TESTING_FILE, "testing")
    all_programs = training + testing
    if len(all_programs) != 500:
        raise ValueError(f"Expected 500 programs, got {len(all_programs)}.")

    print("Deriving compiler-confirmed pass applicability labels in memory...")
    training = [record for record in all_programs if record["split"] == "training"]
    testing = [record for record in all_programs if record["split"] == "testing"]
    training_analysis = label_programs(training)
    testing_analysis = label_programs(testing)
    analyses = {**training_analysis, **testing_analysis}

    print("Training on training split only...")
    bundle, training_summary = train(analysis=training_analysis)
    print("Evaluating recommender on the held-out testing split...")
    evaluation = evaluate_model(bundle, testing, testing_analysis)
    _save_json(RESULTS / "benchmark_v2_ml_evaluation.json", evaluation)
    print(
        f"Test recommendations: accuracy={evaluation['overall_pass_recommendation_accuracy']:.3f}, "
        f"micro-F1={evaluation['overall_f1_micro']:.3f}, "
        f"macro-F1={evaluation['overall_f1_macro']:.3f}"
    )
    print("Running recommended and separate all-pass baselines on all testing programs...")
    compiler_results, failures = benchmark_testing_programs(bundle, testing, testing_analysis)
    test_count = len(compiler_results)
    recommended_passes = [result["ml_recommended"] for result in compiler_results]
    all_passes = [result["all_passes_baseline"] for result in compiler_results]
    compiler_summary = {
        "test_program_count": test_count,
        "recommended_correct_outputs": sum(
            item["output_equivalent"] and item["expected_output_match"]
            for item in recommended_passes
        ),
        "all_passes_correct_outputs": sum(
            item["output_equivalent"] and item["expected_output_match"]
            for item in all_passes
        ),
        "failures": failures,
        "ml_recommended_average_instruction_reduction_pct": _mean([
            100 * (item["instructions_before"] - item["instructions_after"])
            / item["instructions_before"]
            if item.get("instructions_before") else 0
            for item in recommended_passes
        ]),
        "all_passes_average_instruction_reduction_pct": _mean([
            100 * (item["instructions_before"] - item["instructions_after"])
            / item["instructions_before"]
            if item.get("instructions_before") else 0
            for item in all_passes
        ]),
        "ml_recommended_average_code_size_reduction_pct": _mean([
            100 * (item["code_size_before"] - item["code_size_after"])
            / item["code_size_before"]
            if item.get("code_size_before") else 0
            for item in recommended_passes
        ]),
        "all_passes_average_code_size_reduction_pct": _mean([
            100 * (item["code_size_before"] - item["code_size_after"])
            / item["code_size_before"]
            if item.get("code_size_before") else 0
            for item in all_passes
        ]),
        "ml_recommended_mean_optimization_cycles": _mean([
            item["optimization_cycles"] for item in recommended_passes
        ]),
        "all_passes_mean_optimization_cycles": _mean([
            item["optimization_cycles"] for item in all_passes
        ]),
        "interpretation": "ML-recommended execution and the all-six-pass baseline are independent compiler runs. All-pass results do not measure ML quality.",
    }
    _save_json(RESULTS / "benchmark_v2_compiler_results.json", compiler_results)
    _write_compiler_csv(compiler_results)
    _save_json(RESULTS / "benchmark_v2_compiler_summary.json", compiler_summary)
    _save_json(RESULTS / "benchmark_v2_failures.json", failures)
    _save_manifest(training, testing, analyses, failures)
    subprocess.run(
        [sys.executable, str(ROOT / "make_benchmark_v2_plots.py")],
        cwd=ROOT,
        check=True,
    )
    summary = {
        "program_count": len(all_programs),
        "training_count": len(training),
        "testing_count": len(testing),
        "training_family_count": len(bundle["training_families"]),
        "testing_family_count": len(evaluation["test_families"]),
        "category_distribution": json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))["category_counts"],
        "training_label_counts": {
            name: sum(training_analysis[record["id"]]["applicability_labels"][name] for record in training)
            for name in PASSES
        },
        "ml_recommendation_evaluation": {
            key: evaluation[key]
            for key in (
                "overall_pass_recommendation_accuracy",
                "overall_precision_micro",
                "overall_recall_micro",
                "overall_f1_micro",
                "overall_precision_macro",
                "overall_recall_macro",
                "overall_f1_macro",
                "subset_accuracy",
                "hamming_loss",
            )
        },
        "compiler_test_summary": compiler_summary,
        "test_failure_count": len(failures),
        "training_report": str((RESULTS / "benchmark_v2_training_summary.json").relative_to(ROOT)),
        "evaluation_report": str((RESULTS / "benchmark_v2_ml_evaluation.json").relative_to(ROOT)),
        "compiler_results": str((RESULTS / "benchmark_v2_compiler_results.json").relative_to(ROOT)),
        "plots_directory": str(PLOTS.relative_to(ROOT)),
    }
    _save_json(RESULTS / "benchmark_v2_summary.json", summary)
    print(json.dumps({
        "dataset": {"programs": len(all_programs), "training": len(training), "testing": len(testing)},
        "ml_test_metrics": summary["ml_recommendation_evaluation"],
        "compiler_test_failures": len(failures),
        "summary_path": "results/benchmark_v2_summary.json",
    }, indent=2))
    if failures:
        raise SystemExit(f"Compiler correctness failures: {len(failures)}; see results/benchmark_v2_failures.json.")


if __name__ == "__main__":
    main()
