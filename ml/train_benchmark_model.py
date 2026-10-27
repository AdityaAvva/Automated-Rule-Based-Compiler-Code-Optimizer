"""Train the grouped-benchmark pass recommender from training data only."""

import json
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    from .benchmark_features import FEATURE_NAMES
except ImportError:
    from benchmark_features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAINING_FILE = ROOT / "datasets" / "training" / "programs.json"
BENCHMARK_DATASET = ROOT / "benchmarks" / "dataset.json"
BENCHMARK_MANIFEST = ROOT / "benchmarks" / "manifest.json"
MODEL_FILE = ROOT / "ml" / "benchmark_pass_recommender.joblib"
SUMMARY_FILE = ROOT / "results" / "benchmark_v2_training_summary.json"
TRAINING_FIELDS = {
    "id", "category", "size", "source", "expected_output", "inputs",
}
PASSES = [
    "constant_folding",
    "constant_propagation",
    "algebraic_simplification",
    "common_subexpression_elimination",
    "dead_code_elimination",
    "strength_reduction",
]
SEED = 20261026


def train(
    training_file=TRAINING_FILE,
    model_file=MODEL_FILE,
    summary_file=SUMMARY_FILE,
    analysis=None,
):
    """Derive compiler features/targets from strict training records and fit the model."""
    training_path = Path(training_file).resolve()
    expected_training_root = (ROOT / "datasets" / "training").resolve()
    if not training_path.is_relative_to(expected_training_root):
        raise ValueError("Model training input must be under datasets/training/.")
    training_bytes = training_path.read_bytes()
    records = json.loads(training_bytes.decode("utf-8"))
    if len(records) != 400 or any(set(record) != TRAINING_FIELDS for record in records):
        raise ValueError(
            "Training file must contain 400 records with exactly the six supported fields."
        )
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Training data contains duplicate program IDs.")

    manifest = json.loads(BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
    metadata = {item["id"]: item for item in manifest.get("program_metadata", [])}
    if set(ids) - set(metadata):
        raise ValueError("Training split metadata is incomplete.")
    if any(metadata[record_id].get("split") != "training" for record_id in ids):
        raise ValueError("Training file contains a program not assigned to training.")
    families = {metadata[record_id]["family_id"] for record_id in ids}

    if analysis is None:
        from run_benchmark_ml import label_programs

        analysis_records = [
            {**record, "family_id": metadata[record["id"]]["family_id"], "split": "training"}
            for record in records
        ]
        analysis = label_programs(analysis_records)
    if set(analysis) != set(ids):
        raise ValueError("Compiler analysis does not exactly match the training program IDs.")
    matrix = np.asarray([analysis[record_id]["features"] for record_id in ids], dtype=float)
    targets = np.asarray([
        [int(bool(analysis[record_id]["applicability_labels"][name])) for name in PASSES]
        for record_id in ids
    ], dtype=int)
    if matrix.shape != (len(records), len(FEATURE_NAMES)):
        raise ValueError(
            f"Training feature matrix shape {matrix.shape} does not match "
            f"{len(FEATURE_NAMES)} named features."
        )

    models = {}
    details = {}
    for column, name in enumerate(PASSES):
        target = targets[:, column]
        classes = np.unique(target)
        if len(classes) == 1:
            models[name] = {"constant": int(classes[0])}
            details[name] = {
                "status": "constant training labels; stored prevalence baseline",
                "positive_count": int(target.sum()),
                "training_support": len(target),
            }
            continue
        classifier = RandomForestClassifier(
            n_estimators=240,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
        classifier.fit(matrix, target)
        models[name] = classifier
        details[name] = {
            "status": "trained",
            "positive_count": int(target.sum()),
            "training_support": len(target),
        }

    bundle = {
        "schema_version": 1,
        "feature_schema": "grouped_benchmark_v1",
        "feature_names": FEATURE_NAMES,
        "preprocessing": {
            "method": "none",
            "description": "Deterministic source/original-TAC numeric counts; no fitted preprocessing state.",
        },
        "models": models,
        "passes": PASSES,
        "thresholds": {name: 0.5 for name in PASSES},
        "training_ids": ids,
        "training_families": sorted(families),
        "training_split": "training",
        "training_data_sha256": hashlib.sha256(training_bytes).hexdigest(),
        "random_state": SEED,
        "label_method": "single-pass compiler-confirmed applicability on original TAC",
    }
    model_path = Path(model_file)
    summary_path = Path(summary_file)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    summary = {
        "model": str(model_path.relative_to(ROOT)),
        "training_samples": len(records),
        "training_families": len(families),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "preprocessing": bundle["preprocessing"],
        "training_split_only": True,
        "training_ids": ids,
        "training_data_sha256": bundle["training_data_sha256"],
        "pass_training_details": details,
        "label_method": bundle["label_method"],
        "random_state": SEED,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return bundle, summary


if __name__ == "__main__":
    bundle, report = train()
    print(
        f"Trained benchmark recommender on {report['training_samples']} training programs "
        f"({report['feature_count']} features): "
        f"{MODEL_FILE.relative_to(ROOT)}"
    )
