"""Local web API for the existing C++ optimizer and project reports."""

import argparse
import csv
import hashlib
import importlib
import json
import re
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = WEB_ROOT / "static"
TEMP_ROOT = WEB_ROOT / "runtime" / "tmp"
RESULTS = ROOT / "results"
DATASETS = ROOT / "datasets"
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
MODEL_PATHS = {
    "separated_training": ROOT / "ml" / "separated_pass_recommender.joblib",
    "grouped_benchmark": ROOT / "ml" / "benchmark_pass_recommender.joblib",
    "real_world": ROOT / "ml" / "real_world_pass_recommender.joblib",
    "legacy": ROOT / "ml" / "pass_recommender.joblib",
}
MAX_SOURCE_LENGTH = 50_000
MAX_REQUEST_BYTES = 100_000
OPTIMIZER_TIMEOUT_SECONDS = 15
BENCHMARK_TIMEOUT_SECONDS = 20 * 60
CSV_RESULT = RESULTS / "benchmark_results.csv"
SUMMARY_RESULT = RESULTS / "benchmark_summary.json"
DATASET_STATS = DATASETS / "metadata" / "dataset_statistics.json"
SOURCE_MANIFEST = DATASETS / "metadata" / "source_manifest.json"
COMPATIBLE_DATASET = DATASETS / "processed" / "compatible_programs.json"
REJECTED_DATASET = DATASETS / "processed" / "rejected_programs.json"
CONVERGENCE_RESULT = RESULTS / "convergence_trace.json"
GROUPED_MANIFEST = DATASETS / "benchmark_manifest.json"
GROUPED_TRAINING = DATASETS / "training" / "programs.json"
GROUPED_TESTING = DATASETS / "testing" / "programs.json"
BENCHMARK_DATASET = ROOT / "benchmarks" / "dataset.json"
BENCHMARK_MANIFEST = ROOT / "benchmarks" / "manifest.json"
GROUPED_EVALUATION = RESULTS / "benchmark_v2_ml_evaluation.json"
GROUPED_COMPILER_RESULTS = RESULTS / "benchmark_v2_compiler_results.json"
GROUPED_COMPILER_SUMMARY = RESULTS / "benchmark_v2_compiler_summary.json"
SEPARATED_TRAINING = ROOT / "dataset" / "training" / "training.json"
SEPARATED_TESTING = ROOT / "dataset" / "testing" / "testing.json"
SEPARATED_TRAINING_SUMMARY = RESULTS / "separated_training_summary.json"
SEPARATED_TESTING_RESULTS = RESULTS / "separated_testing_results.json"
SEPARATED_TESTING_SUMMARY = RESULTS / "separated_testing_summary.json"
SEPARATED_TRAINING_FIELDS = {
    "id", "category", "size", "source", "expected_output", "inputs",
}

_benchmark_lock = threading.Lock()
_benchmark_state = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "return_code": None,
    "error": None,
    "output_tail": None,
}
_separated_training_lock = threading.Lock()
_separated_training_state = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "error": None,
    "output_tail": None,
}


def optimizer_executable():
    return ROOT / ("compiler_optimizer.exe" if sys.platform == "win32" else "compiler_optimizer")


def _read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _section(output, start, end):
    match = re.search(re.escape(start) + r"\n(.*?)" + re.escape(end), output, re.S)
    return match.group(1).rstrip() if match else None


def _metrics(output):
    result = {}
    for line in output.splitlines():
        match = re.fullmatch(r"([a-z_]+)=(-?\d+)", line.strip())
        if match:
            result[match.group(1)] = int(match.group(2))
    return result


def _strip_convergence(text):
    if text is None:
        return None
    return "\n".join(
        line for line in text.splitlines()
        if not line.startswith("convergence_")
    ).rstrip()


def _run_optimizer(source, enabled_passes):
    executable = optimizer_executable()
    if not executable.is_file():
        raise RuntimeError(
            f"C++ optimizer not found at {executable}. Compile src\\main.cpp first."
        )

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="optimize-", dir=TEMP_ROOT) as temp_dir:
        source_path = Path(temp_dir) / "input.src"
        source_path.write_text(source, encoding="utf-8")
        command = [
            str(executable),
            "--trace-convergence",
            "--passes",
            ",".join(enabled_passes),
            str(source_path),
        ]
        try:
            completed = subprocess.run(
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
                f"Optimizer exceeded its {OPTIMIZER_TIMEOUT_SECONDS}-second timeout."
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Could not start the C++ optimizer: {exc}") from exc
    return completed


def optimize_source(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    source = payload.get("source")
    mode = payload.get("mode")
    model_name = payload.get("model", "real_world")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Source code must be a non-empty string.")
    if len(source) > MAX_SOURCE_LENGTH:
        raise ValueError(f"Source code exceeds the {MAX_SOURCE_LENGTH:,}-character limit.")
    if "\x00" in source:
        raise ValueError("Source code must not contain a NUL character.")
    if mode not in ("recommended", "all"):
        raise ValueError("Mode must be 'recommended' or 'all'.")
    if model_name not in MODEL_PATHS:
        raise ValueError(
            "Model must be 'separated_training', 'grouped_benchmark', 'real_world', or 'legacy'."
        )

    # First let the real parser and interpreter validate the input before ML inference.
    validation = _run_optimizer(source, [])
    if validation.returncode != 0:
        message = validation.stderr.strip() or "The compiler rejected this source program."
        return {
            "status": "ERROR",
            "error": message,
            "original_source": source,
            "compiler_stderr": validation.stderr.strip(),
        }

    recommendation_details = []
    recommended_names = []
    recommendation_error = None
    if MODEL_PATHS[model_name].is_file():
        try:
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from ml_optimize import recommend_passes

            original_tac = _section(
                validation.stdout, "=== ORIGINAL TAC ===", "=== OPTIMIZED TAC ==="
            )
            recommended_names, recommendation_details = recommend_passes(
                source,
                model_path=MODEL_PATHS[model_name],
                original_tac=original_tac,
            )
        except (ImportError, OSError, ValueError, KeyError, EOFError) as exc:
            recommendation_error = str(exc)
    else:
        recommendation_error = f"The {model_name} model file is not available."

    if mode == "recommended":
        if recommendation_error:
            raise RuntimeError(f"Cannot run ML-recommended mode: {recommendation_error}")
        selected = recommended_names
    else:
        selected = list(PASSES)

    completed = _run_optimizer(source, selected)
    output = completed.stdout
    if completed.returncode != 0 and not output:
        return {
            "status": "ERROR",
            "error": completed.stderr.strip() or "Optimizer execution failed.",
            "original_source": source,
            "compiler_stderr": completed.stderr.strip(),
            "compiler_return_code": completed.returncode,
        }

    metrics = _metrics(output)
    original_output = _section(output, "Original output:", "Optimized output:")
    optimized_output = _section(output, "Optimized output:", "Verdict:")
    verdict_match = re.search(r"^Verdict: (PASS|FAIL)$", output, re.M)
    verdict = verdict_match.group(1) if verdict_match else "ERROR"
    convergence = [
        {
            "iteration": int(iteration),
            "instructions": int(instructions),
            "stable": bool(int(stable)),
            "changed": int(iteration) > 0 and not bool(int(stable)),
        }
        for iteration, instructions, stable in re.findall(
            r"^convergence_iteration=(\d+),instructions=(\d+),stable=(\d+)$",
            output,
            re.M,
        )
    ]

    pass_recommendations = {item["name"]: item for item in recommendation_details}
    pass_status = []
    for name in PASSES:
        transformations = metrics.get(name, 0)
        pass_recommendations_item = pass_recommendations.get(name)
        pass_status.append({
            "name": name,
            "label": PASS_LABELS[name],
            "recommended": (
                bool(pass_recommendations_item["recommended"])
                if pass_recommendations_item else None
            ),
            "recommendation_score": (
                pass_recommendations_item.get("score")
                if pass_recommendations_item else None
            ),
            "recommendation_threshold": (
                pass_recommendations_item.get("threshold")
                if pass_recommendations_item else None
            ),
            "selected": name in selected,
            "applied": transformations > 0,
            "transformations": transformations,
            "isolated_transformations": metrics.get(f"{name}_transformations"),
            "isolated_output_equivalent": (
                bool(metrics[f"{name}_output_match"])
                if f"{name}_output_match" in metrics else None
            ),
        })

    validation_match = re.search(r"^Verdict: (PASS|FAIL)$", validation.stdout, re.M)
    validation_output = _section(validation.stdout, "Original output:", "Optimized output:")
    original_tac = _section(output, "=== ORIGINAL TAC ===", "=== OPTIMIZED TAC ===")
    optimized_tac = _strip_convergence(
        _section(output, "=== OPTIMIZED TAC ===", "=== OUTPUT CHECK ===")
    )
    equivalence_confirmed = (
        verdict == "PASS"
        and original_output is not None
        and optimized_output is not None
        and original_output == optimized_output
        and validation_match is not None
        and validation_match.group(1) == "PASS"
        and validation_output == original_output
    )
    status = "PASS" if equivalence_confirmed else ("FAIL" if verdict == "FAIL" else "ERROR")
    return {
        "status": status,
        "original_source": source,
        "original_tac": original_tac,
        "optimized_tac": optimized_tac,
        "original_output": original_output,
        "optimized_output": optimized_output,
        "validation": {
            "status": status,
            "output_equivalence_confirmed_by_compiler": equivalence_confirmed,
        },
        "mode": mode,
        "model": model_name,
        "model_recommendation_error": recommendation_error,
        "recommended_passes": recommended_names,
        "selected_passes": selected,
        "passes": pass_status,
        "metrics": metrics,
        "optimization_cycles": max(
            (item["iteration"] for item in convergence), default=None
        ),
        "convergence_trace": convergence,
        "compiler_return_code": completed.returncode,
        "compiler_stderr": completed.stderr.strip(),
    }


def separated_training_status():
    if not SEPARATED_TRAINING.is_file() or not SEPARATED_TESTING.is_file():
        return {
            "available": False,
            "error": "Separated training/testing JSON files are missing.",
            "training_count": 0,
            "testing_count": 0,
            "model_trained": False,
            "testing_evaluated": False,
        }
    training = _read_json(SEPARATED_TRAINING)
    testing = _read_json(SEPARATED_TESTING)
    training_data_sha256 = hashlib.sha256(SEPARATED_TRAINING.read_bytes()).hexdigest()
    testing_data_sha256 = hashlib.sha256(SEPARATED_TESTING.read_bytes()).hexdigest()
    training_ids = [record.get("id") for record in training]
    training_summary = (
        _read_json(SEPARATED_TRAINING_SUMMARY)
        if SEPARATED_TRAINING_SUMMARY.is_file() else {}
    )
    model_path = MODEL_PATHS["separated_training"]
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path.is_file() else None
    model_trained = bool(
        model_hash
        and training_summary.get("training_split_only") is True
        and training_summary.get("training_samples") == len(training) == 400
        and set(training_summary.get("training_ids", [])) == set(training_ids)
        and training_summary.get("model_sha256") == model_hash
        and training_summary.get("training_data_sha256") == training_data_sha256
    )
    testing_summary = (
        _read_json(SEPARATED_TESTING_SUMMARY)
        if SEPARATED_TESTING_SUMMARY.is_file() else {}
    )
    testing_evaluated = bool(
        model_trained
        and testing_summary.get("status") == "evaluated"
        and testing_summary.get("model_sha256") == model_hash
        and testing_summary.get("testing_program_count") == len(testing) == 100
        and testing_summary.get("executed_program_count") == len(testing)
        and testing_summary.get("testing_data_sha256") == testing_data_sha256
    )
    with _separated_training_lock:
        job = dict(_separated_training_state)
    return {
        "available": True,
        "training_file": str(SEPARATED_TRAINING.relative_to(ROOT)),
        "testing_file": str(SEPARATED_TESTING.relative_to(ROOT)),
        "training_count": len(training),
        "testing_count": len(testing),
        "model_path": str(model_path.relative_to(ROOT)),
        "model_trained": model_trained,
        "training_status": "trained" if model_trained else "not_trained",
        "training_samples": training_summary.get("training_samples"),
        "testing_evaluated": testing_evaluated,
        "testing_status": "evaluated" if testing_evaluated else "not_evaluated",
        "testing_executed_count": (
            testing_summary.get("executed_program_count") if testing_evaluated else 0
        ),
        "training_job": job,
    }


def separated_dataset_data(split, page=1, limit=500):
    path = SEPARATED_TRAINING if split == "training" else SEPARATED_TESTING
    records = _read_json(path)
    if split == "training":
        if len(records) != 400 or any(set(record) != SEPARATED_TRAINING_FIELDS for record in records):
            raise ValueError("Training JSON must contain 400 records with exactly six required fields.")
        categories = dict(Counter(record["category"] for record in records))
        programs = records
    else:
        if len(records) != 100 or any(set(record) != {"source"} for record in records):
            raise ValueError("Testing JSON must contain 100 records containing only source.")
        categories = {}
        programs = records

    status = separated_training_status()
    start = (page - 1) * limit
    response = {
        "available": True,
        "split": split,
        "file": str(path.relative_to(ROOT)),
        "program_count": len(records),
        "programs": programs[start:start + limit],
        "page": page,
        "limit": limit,
        "total": len(records),
        "category_distribution": categories,
        "status": status,
    }
    if split == "testing":
        result_by_index = {}
        if status["testing_evaluated"] and SEPARATED_TESTING_RESULTS.is_file():
            saved_results = _read_json(SEPARATED_TESTING_RESULTS)
            for result in saved_results:
                index = result.get("program_index")
                if not isinstance(index, int) or not 1 <= index <= len(records):
                    continue
                digest = hashlib.sha256(records[index - 1]["source"].encode("utf-8")).hexdigest()
                if digest == result.get("source_sha256"):
                    result_by_index[index] = {
                        "program_index": index,
                        "status": result.get("status"),
                        "output_equivalent": result.get("output_equivalent"),
                        "selected_passes": result.get("selected_passes"),
                    }
        response["execution_status"] = [
            result_by_index.get(index, {
                "program_index": index,
                "status": "not_evaluated",
                "output_equivalent": None,
                "selected_passes": [],
            })
            for index in range(start + 1, min(start + limit, len(records)) + 1)
        ]
    return response


def _separated_training_worker():
    commands = (
        [sys.executable, str(ROOT / "ml" / "train_separated_model.py")],
        [sys.executable, str(ROOT / "ml" / "evaluate_separated_model.py")],
    )
    output = []
    error = None
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=BENCHMARK_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = str(exc)
            break
        output.extend((completed.stdout, completed.stderr))
        if completed.returncode:
            error = (
                f"{Path(command[1]).name} exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout[-1200:]}"
            )
            break
    with _separated_training_lock:
        _separated_training_state.update({
            "status": "failed" if error else "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
            "output_tail": "\n".join(output)[-4000:],
        })


def start_separated_training():
    with _separated_training_lock:
        if _separated_training_state["status"] == "running":
            return dict(_separated_training_state)
        if not SEPARATED_TRAINING.is_file():
            raise FileNotFoundError(
                "Training dataset is missing; create the projection from the existing 500-program suite first."
            )
        _separated_training_state.update({
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
            "output_tail": None,
        })
        thread = threading.Thread(target=_separated_training_worker, daemon=True)
        thread.start()
        return dict(_separated_training_state)


def dataset_data():
    required = (DATASET_STATS, SOURCE_MANIFEST, COMPATIBLE_DATASET, REJECTED_DATASET)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        return {
            "available": False,
            "error": "Real-world dataset reports are missing.",
            "missing_files": missing,
        }
    stats = _read_json(DATASET_STATS)
    manifest = _read_json(SOURCE_MANIFEST)
    compatible = _read_json(COMPATIBLE_DATASET)
    rejected = _read_json(REJECTED_DATASET)
    split_counts = stats.get("split_counts", {})
    return {
        "available": True,
        "summary": {
            "program_count": len(manifest.get("programs", [])),
            "source_types": stats.get("counts_by_source_type", {}),
            "compatible_by_source_type": stats.get("compatible_counts_by_source_type", {}),
            "training_count": split_counts.get("train", {}).get("programs", 0),
            "validation_count": split_counts.get("validation", {}).get("programs", 0),
            "testing_count": split_counts.get("test", {}).get("programs", 0),
            "compatible_count": stats.get("compatible_program_count", len(compatible)),
            "rejected_count": stats.get("rejected_program_count", len(rejected)),
            "compatibility_rate": stats.get("compatibility_rate"),
            "source_family_count": stats.get("unique_compatible_source_families"),
            "verified_origins": manifest.get("verified_dataset_origin_counts", {}),
            "conceptual_reference_count": len(manifest.get("conceptual_references", [])),
            "conceptual_reference_examples": sum(
                item.get("verified_dataset_example_count", 0)
                for item in manifest.get("conceptual_references", [])
            ),
        },
        "programs": manifest.get("programs", []),
        "conceptual_references": manifest.get("conceptual_references", []),
        "rejected_programs": rejected,
        "limitations": stats.get("limitations", []),
    }


def benchmark_data(page=1, limit=100):
    if not SUMMARY_RESULT.is_file() or not CSV_RESULT.is_file():
        return {
            "available": False,
            "error": "Benchmark results are not available yet. Run the full benchmark explicitly.",
            "summary": None,
            "results": [],
            "plots": [],
        }
    summary = _read_json(SUMMARY_RESULT)
    with CSV_RESULT.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if BENCHMARK_DATASET.is_file():
        canonical = _read_json(BENCHMARK_DATASET)
        benchmark_ids = {record.get("id") for record in canonical}
        result_ids = {row.get("id") for row in rows}
        if len(canonical) != 500 or len(benchmark_ids) != 500:
            return {
                "available": False,
                "error": "Canonical benchmark dataset does not contain 500 unique programs.",
                "summary": None,
                "results": [],
                "plots": [],
            }
        if len(rows) != 500 or len(result_ids) != 500 or result_ids != benchmark_ids:
            return {
                "available": False,
                "error": "Saved benchmark results do not match the current 500-program dataset. Run the benchmark pipeline.",
                "summary": None,
                "results": [],
                "plots": [],
            }
    total = len(rows)
    start = (page - 1) * limit
    convergence = _read_json(CONVERGENCE_RESULT) if CONVERGENCE_RESULT.is_file() else None
    plot_files = sorted(
        path.name for path in (ROOT / "plots").glob("*.png") if path.is_file()
    )
    categories = dict(Counter(row.get("category", "unknown") for row in rows))
    valid_passes = sum(
        row.get("verdict") == "PASS" and str(row.get("expected_output_match", "")).lower() == "true"
        for row in rows
    )
    return {
        "available": True,
        "summary": {
            **summary,
            "total_benchmarks": total,
            "validated_programs": valid_passes,
            "failed_programs": total - valid_passes,
            "category_distribution": categories,
            "convergence": (
                {
                    key: convergence.get(key)
                    for key in (
                        "programs_measured",
                        "programs_reaching_fixed_point",
                        "fixed_point_rate_pct",
                        "mean_cycles_to_fixed_point",
                        "median_cycles_to_fixed_point",
                        "maximum_cycles_to_fixed_point",
                    )
                }
                if convergence else summary.get("convergence_statistics")
            ),
        },
        "results": rows[start:start + limit],
        "page": page,
        "limit": limit,
        "total": total,
        "plots": plot_files,
    }


def grouped_benchmark_data(page=1, limit=25, category="", split="", pass_name="", size=""):
    required = (
        GROUPED_MANIFEST,
        GROUPED_TRAINING,
        GROUPED_TESTING,
        BENCHMARK_DATASET,
        BENCHMARK_MANIFEST,
        GROUPED_EVALUATION,
        GROUPED_COMPILER_RESULTS,
        GROUPED_COMPILER_SUMMARY,
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        return {
            "available": False,
            "error": "Grouped 500-program benchmark reports are not generated yet.",
            "missing_files": missing,
        }

    manifest = _read_json(GROUPED_MANIFEST)
    benchmark_manifest = _read_json(BENCHMARK_MANIFEST)
    training = _read_json(GROUPED_TRAINING)
    testing = _read_json(GROUPED_TESTING)
    evaluation = _read_json(GROUPED_EVALUATION)
    compiler_results = _read_json(GROUPED_COMPILER_RESULTS)
    compiler_summary = _read_json(GROUPED_COMPILER_SUMMARY)
    canonical = _read_json(BENCHMARK_DATASET)
    benchmark_files = [
        path for path in (ROOT / "benchmarks").iterdir()
        if path.is_file() and re.fullmatch(r"CAT[1-5]_\d{3}\.json", path.name)
    ]
    if (
        not manifest.get("compiler_labeling_completed")
        or not benchmark_manifest.get("compiler_labeling_completed")
    ):
        return {
            "available": False,
            "error": "Current benchmark files have not completed compiler labeling and held-out evaluation. Run python run_benchmark_ml.py.",
        }
    canonical_ids = [record.get("id") for record in canonical]
    if len(canonical) != 500 or len(set(canonical_ids)) != 500 or len(benchmark_files) != 500:
        return {
            "available": False,
            "error": "Actual benchmark files are incomplete or have duplicate IDs; refusing to show stale grouped results.",
        }
    canonical_by_id = {record["id"]: record for record in canonical}
    if any(
        set(record) != SEPARATED_TRAINING_FIELDS
        for record in canonical
    ):
        return {
            "available": False,
            "error": "Canonical benchmark records must use the six-field source schema.",
        }
    individual_records = [_read_json(path) for path in benchmark_files]
    individual_by_id = {record.get("id"): record for record in individual_records}
    if set(individual_by_id) != set(canonical_ids) or any(
        set(individual_by_id[record["id"]]) != SEPARATED_TRAINING_FIELDS
        or individual_by_id[record["id"]].get("source") != record.get("source")
        or individual_by_id[record["id"]].get("expected_output") != record.get("expected_output")
        for record in canonical
    ):
        return {
            "available": False,
            "error": "Individual benchmark JSON files disagree with benchmarks/dataset.json.",
        }

    metadata = benchmark_manifest.get("program_metadata", [])
    metadata_by_id = {item.get("id"): item for item in metadata}
    if len(metadata_by_id) != 500 or any(
        meta.get("id") not in canonical_by_id
        or meta.get("split") not in {"training", "testing"}
        for meta in metadata
    ):
        return {
            "available": False,
            "error": "Split metadata is missing or does not match the 500 benchmark records.",
        }
    if (
        len(training) != 400
        or any(set(record) != SEPARATED_TRAINING_FIELDS for record in training)
        or len(testing) != 100
        or any(set(record) != {"source"} for record in testing)
    ):
        return {
            "available": False,
            "error": "Training/testing files do not use the simplified six-field/source-only schemas.",
        }
    train_ids = {record["id"] for record in training}
    training_by_id = {record["id"]: record for record in training}
    if any(
        record[key] != canonical_by_id[record_id][key]
        for record_id, record in training_by_id.items()
        for key in SEPARATED_TRAINING_FIELDS
    ):
        return {
            "available": False,
            "error": "Training JSON records do not match the current canonical benchmark sources.",
        }
    test_hashes = {
        hashlib.sha256(record["source"].encode("utf-8")).hexdigest()
        for record in testing
    }
    test_meta = [
        meta for meta in metadata
        if meta.get("split") == "testing" and meta.get("source_sha256") in test_hashes
    ]
    split_ids = train_ids | {item["id"] for item in test_meta}
    if split_ids != set(canonical_ids) or len(training) != 400 or len(testing) != 100:
        return {
            "available": False,
            "error": "Training/testing records do not exactly cover the current canonical benchmark dataset.",
        }
    if any(metadata_by_id[record_id].get("split") != "training" for record_id in train_ids):
        return {
            "available": False,
            "error": "Training records disagree with the manifest split assignment.",
        }
    if len(test_meta) != len(testing) or len(test_hashes) != len(testing):
        return {
            "available": False,
            "error": "Testing source-only records do not map uniquely to testing manifest entries.",
        }
    actual_category_counts = Counter(record.get("category") for record in canonical)
    if len(actual_category_counts) != 5 or any(count != 100 for count in actual_category_counts.values()):
        return {
            "available": False,
            "error": "Actual canonical JSON records do not have exactly 100 programs in each category.",
        }
    split_counts = Counter(
        (meta.get("category"), meta.get("split")) for meta in metadata
    )
    if any(
        split_counts[(category, "training")] != 80
        or split_counts[(category, "testing")] != 20
        for category in actual_category_counts
    ):
        return {
            "available": False,
            "error": "Actual benchmark records do not have an 80/20 split in every category.",
        }
    family_splits = {}
    for meta in metadata:
        family_splits.setdefault(meta.get("family_id"), set()).add(meta.get("split"))
    if any(len(splits) != 1 for splits in family_splits.values()):
        return {
            "available": False,
            "error": "Related generation families cross the training/testing split.",
        }
    predictions = {item["id"]: item for item in evaluation.get("predictions", [])}
    compiler_by_id = {item["id"]: item for item in compiler_results}
    programs = [
        (canonical_by_id[record_id], metadata_by_id[record_id])
        for record_id in canonical_ids
    ]

    filtered = []
    for record, meta in programs:
        if category and meta.get("category") != category:
            continue
        if split and meta.get("split") != split:
            continue
        if size and meta.get("size") != size:
            continue
        test_prediction = predictions.get(record["id"], {})
        compiler_result = compiler_by_id.get(record["id"], {})
        applicability = test_prediction.get("true_labels")
        if pass_name and (
            meta.get("split") != "testing"
            or not applicability
            or not applicability.get(pass_name)
        ):
            continue
        filtered.append({
            "id": record["id"],
            "category": meta["category"],
            "split": meta["split"],
            "size": meta["size"],
            "source": record["source"],
            "expected_output": record["expected_output"],
            "applicability_labels": applicability,
            "ml_recommendations": test_prediction.get("recommendations"),
            "compiler_result": {
                name: compiler_result.get(name)
                for name in ("ml_recommended", "all_passes_baseline")
            },
        })
    total = len(filtered)
    start = max(0, page - 1) * limit
    per_pass = evaluation.get("per_pass", {})
    per_category = evaluation.get("per_category", {})
    return {
        "available": True,
        "summary": {
            "program_count": manifest.get("program_count"),
            "training_count": manifest.get("training_count"),
            "testing_count": manifest.get("testing_count"),
            "category_counts": manifest.get("category_counts", {}),
            "training_record_fields": sorted(SEPARATED_TRAINING_FIELDS),
            "testing_record_fields": ["source"],
            "training_label_counts": manifest.get("label_counts", {}).get("training", {}),
            "testing_label_counts": manifest.get("label_counts", {}).get("testing", {}),
            "ml_metrics": {
                key: evaluation.get(key)
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
            "ml_per_pass": per_pass,
            "ml_per_category": per_category,
            "compiler_summary": compiler_summary,
            "label_method": manifest.get("label_method"),
            "evaluation_note": evaluation.get("evaluation_note"),
        },
        "programs": filtered[start:start + limit],
        "page": page,
        "limit": limit,
        "total_filtered": total,
    }


def _benchmark_worker():
    with _benchmark_lock:
        _benchmark_state.update({
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "return_code": None,
            "error": None,
            "output_tail": None,
        })
    try:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "run_project.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=BENCHMARK_TIMEOUT_SECONDS,
            check=False,
        )
        state = {
            "status": "completed" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "error": (
                None if completed.returncode == 0
                else completed.stderr.strip()[-4000:] or "Benchmark pipeline returned a failure."
            ),
            "output_tail": completed.stdout[-4000:],
        }
    except subprocess.TimeoutExpired:
        state = {
            "status": "failed",
            "return_code": None,
            "error": f"Benchmark pipeline exceeded {BENCHMARK_TIMEOUT_SECONDS} seconds.",
            "output_tail": None,
        }
    except OSError as exc:
        state = {
            "status": "failed",
            "return_code": None,
            "error": f"Could not start the benchmark pipeline: {exc}",
            "output_tail": None,
        }
    with _benchmark_lock:
        _benchmark_state.update(state)
        _benchmark_state["completed_at"] = datetime.now(timezone.utc).isoformat()


def start_benchmarks():
    with _benchmark_lock:
        if _benchmark_state["status"] == "running":
            return dict(_benchmark_state)
        thread = threading.Thread(target=_benchmark_worker, name="full-benchmark", daemon=True)
        thread.start()
        return {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "message": "Existing full benchmark pipeline started. Progress is not reported by that pipeline.",
        }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "CompilerOptimizerDashboard/1.0"

    def log_message(self, fmt, *args):
        print(f"[web] {self.address_string()} {fmt % args}")

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, path, content_type):
        try:
            body = path.read_bytes()
        except OSError:
            self._send_json({"error": "Dashboard asset is missing."}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_static(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
        if parsed.path == "/static/app.js":
            return self._send_static(STATIC_ROOT / "app.js", "text/javascript; charset=utf-8")
        if parsed.path == "/static/styles.css":
            return self._send_static(STATIC_ROOT / "styles.css", "text/css; charset=utf-8")
        if parsed.path.startswith("/plots/benchmark_v2/"):
            filename = parsed.path.removeprefix("/plots/benchmark_v2/")
            if not re.fullmatch(r"\d{2}_[a-zA-Z0-9_-]+\.png", filename):
                return self._send_json({"error": "Unknown grouped benchmark plot."}, 404)
            return self._send_static(
                ROOT / "plots" / "benchmark_v2" / filename,
                "image/png",
            )
        if parsed.path.startswith("/plots/"):
            filename = parsed.path.removeprefix("/plots/")
            if not re.fullmatch(r"\d{2}_[a-zA-Z0-9_-]+\.png", filename):
                return self._send_json({"error": "Unknown plot."}, 404)
            return self._send_static(ROOT / "plots" / filename, "image/png")
        if parsed.path == "/api/health":
            dependency_error = None
            try:
                importlib.import_module("joblib")
                importlib.import_module("numpy")
            except ImportError as exc:
                dependency_error = str(exc)
            return self._send_json({
                "status": "ok",
                "optimizer_available": optimizer_executable().is_file(),
                "models": {
                    name: {"available": path.is_file(), "path": str(path.relative_to(ROOT))}
                    for name, path in MODEL_PATHS.items()
                },
                "ml_dependencies_available": dependency_error is None,
                "ml_dependency_error": dependency_error,
                "local_only": True,
            })
        if parsed.path == "/api/separated-datasets/status":
            try:
                return self._send_json(separated_training_status())
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                return self._send_json({"available": False, "error": str(exc)}, 500)
        if parsed.path in (
            "/api/separated-datasets/training",
            "/api/separated-datasets/testing",
        ):
            split = "training" if parsed.path.endswith("/training") else "testing"
            query = parse_qs(parsed.query)
            try:
                page = max(1, int(query.get("page", ["1"])[0]))
                limit = min(500, max(1, int(query.get("limit", ["500"])[0])))
                data = separated_dataset_data(split, page, limit)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                return self._send_json({"available": False, "error": str(exc)}, 500)
            return self._send_json(data)
        if parsed.path in ("/api/dataset/summary", "/api/dataset/programs"):
            try:
                data = dataset_data()
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                return self._send_json({"available": False, "error": str(exc)}, 500)
            if parsed.path.endswith("/summary"):
                data.pop("programs", None)
                data.pop("rejected_programs", None)
                return self._send_json(data, 200 if data["available"] else 503)
            query = parse_qs(parsed.query)
            search = query.get("q", [""])[0].casefold().strip()
            programs = data.pop("programs", [])
            if search:
                programs = [
                    item for item in programs
                    if search in json.dumps(item, ensure_ascii=False).casefold()
                ]
            data["programs"] = programs
            data.pop("rejected_programs", None)
            return self._send_json(data, 200 if data["available"] else 503)
        if parsed.path in ("/api/benchmarks/summary", "/api/benchmarks/results"):
            query = parse_qs(parsed.query)
            try:
                page = max(1, int(query.get("page", ["1"])[0]))
                limit = min(200, max(1, int(query.get("limit", ["100"])[0])))
                data = benchmark_data(page, limit)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                return self._send_json({"available": False, "error": str(exc)}, 500)
            if parsed.path.endswith("/summary"):
                data.pop("results", None)
            return self._send_json(data, 200 if data["available"] else 503)
        if parsed.path == "/api/benchmark-ml":
            query = parse_qs(parsed.query)
            try:
                page = max(1, int(query.get("page", ["1"])[0]))
                limit = min(100, max(1, int(query.get("limit", ["25"])[0])))
                data = grouped_benchmark_data(
                    page=page,
                    limit=limit,
                    category=query.get("category", [""])[0],
                    split=query.get("split", [""])[0],
                    pass_name=query.get("pass", [""])[0],
                    size=query.get("size", [""])[0],
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                return self._send_json({"available": False, "error": str(exc)}, 500)
            return self._send_json(data, 200 if data["available"] else 503)
        if parsed.path == "/api/benchmarks/run":
            with _benchmark_lock:
                return self._send_json(dict(_benchmark_state))
        return self._send_json({"error": "Endpoint not found."}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/benchmarks/run":
            return self._send_json(start_benchmarks(), 202)
        if parsed.path == "/api/separated-datasets/train":
            try:
                state = start_separated_training()
            except (OSError, ValueError) as exc:
                return self._send_json({"status": "ERROR", "error": str(exc)}, 503)
            return self._send_json(state, 202 if state["status"] == "running" else 200)
        accepted_routes = {
            "/api/optimize",
            "/api/separated-datasets/testing/run",
        }
        if parsed.path not in accepted_routes:
            return self._send_json({"error": "Endpoint not found."}, 404)
        content_length = self.headers.get("Content-Length")
        try:
            request_length = int(content_length or "")
        except ValueError:
            return self._send_json({"error": "A valid Content-Length header is required."}, 411)
        if request_length <= 0 or request_length > MAX_REQUEST_BYTES:
            return self._send_json({
                "error": f"Request body must be between 1 and {MAX_REQUEST_BYTES:,} bytes."
            }, 413)
        try:
            payload = json.loads(self.rfile.read(request_length))
            if parsed.path == "/api/separated-datasets/testing/run":
                if not isinstance(payload, dict) or isinstance(payload.get("index"), bool):
                    raise ValueError("Request must include a testing program index.")
                try:
                    index = int(payload["index"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("Request must include a valid testing program index.") from exc
                testing_records = _read_json(SEPARATED_TESTING)
                if not 1 <= index <= len(testing_records):
                    raise ValueError("Testing program index is outside the testing dataset.")
                payload = {
                    "source": testing_records[index - 1]["source"],
                    "mode": "recommended",
                    "model": "separated_training",
                }
            result = optimize_source(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._send_json({"error": "Request body must contain valid UTF-8 JSON."}, 400)
        except ValueError as exc:
            return self._send_json({"error": str(exc)}, 400)
        except RuntimeError as exc:
            return self._send_json({"status": "ERROR", "error": str(exc)}, 503)
        except (ImportError, OSError, KeyError) as exc:
            return self._send_json({"status": "ERROR", "error": str(exc)}, 500)
        return self._send_json(result, 200 if result["status"] != "ERROR" else 422)


def main():
    parser = argparse.ArgumentParser(description="Serve the local compiler optimizer dashboard.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("Port must be between 1 and 65535.")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"Compiler Optimizer Dashboard: http://127.0.0.1:{args.port}")
    print("Serving locally only. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
