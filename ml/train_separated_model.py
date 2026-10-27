"""Train the separate source/expected-output dataset model using training.json only."""

import json
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from benchmark_features import FEATURE_NAMES, extract_features
from run_benchmark_ml import PASSES, _run_record

TRAINING_FILE = ROOT / "dataset" / "training" / "training.json"
MODEL_FILE = ROOT / "ml" / "separated_pass_recommender.joblib"
SUMMARY_FILE = ROOT / "results" / "separated_training_summary.json"
SEED = 20261026
TRAINING_FIELDS = {
    "id",
    "category",
    "size",
    "source",
    "expected_output",
    "inputs",
}


def train_model():
    training_bytes = TRAINING_FILE.read_bytes()
    records = json.loads(training_bytes.decode("utf-8"))
    training_data_sha256 = hashlib.sha256(training_bytes).hexdigest()
    if len(records) != 400:
        raise ValueError(f"Expected 400 training records; found {len(records)}.")
    if any(set(record) != TRAINING_FIELDS for record in records):
        raise ValueError("Training records must contain exactly the six required fields.")
    if any(not isinstance(record.get("inputs"), list) for record in records):
        raise ValueError("Each training record must contain an inputs list.")

    features = []
    targets = []
    for index, record in enumerate(records, start=1):
        compiler_record = {
            "id": record["id"],
            "source": record["source"],
            "expected_output": record["expected_output"],
        }
        run = _run_record(compiler_record, [], trace=False)
        if not run["expected_output_match"]:
            raise ValueError(
                f"{record['id']}: stored training expected output does not match compiler output."
            )
        features.append(extract_features(record["source"], run["original_tac"]))
        targets.append([
            int(
                run["isolated_passes"][name]["rewrite_count"] > 0
                and run["isolated_passes"][name]["output_equivalent"]
            )
            for name in PASSES
        ])
        if index % 50 == 0 or index == len(records):
            print(f"Derived training features and targets: {index}/{len(records)}")

    matrix = np.asarray(features, dtype=float)
    labels = np.asarray(targets, dtype=int)
    if matrix.shape != (400, len(FEATURE_NAMES)) or labels.shape != (400, len(PASSES)):
        raise ValueError("Training matrix shape does not match the configured feature/pass schema.")

    models = {}
    details = {}
    for column, name in enumerate(PASSES):
        target = labels[:, column]
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
        "feature_schema": "separated_training_v1",
        "feature_names": FEATURE_NAMES,
        "preprocessing": {
            "method": "none",
            "description": "Deterministic features; no fitted preprocessing state.",
        },
        "models": models,
        "passes": PASSES,
        "thresholds": {name: 0.5 for name in PASSES},
        "training_ids": [record["id"] for record in records],
        "training_data_sha256": training_data_sha256,
        "training_split": "dataset/training/training.json",
        "random_state": SEED,
        "label_method": (
            "Each optimizer pass independently rewrites original TAC; positive only "
            "when at least one rewrite preserves compiler output."
        ),
    }
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    model_sha256 = hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()
    summary = {
        "status": "trained",
        "model": str(MODEL_FILE.relative_to(ROOT)),
        "training_file": str(TRAINING_FILE.relative_to(ROOT)),
        "training_samples": len(records),
        "training_ids": bundle["training_ids"],
        "training_data_sha256": training_data_sha256,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "preprocessing": bundle["preprocessing"],
        "training_split_only": True,
        "pass_training_details": details,
        "label_method": bundle["label_method"],
        "random_state": SEED,
        "model_sha256": model_sha256,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    report = train_model()
    print(
        f"Trained {report['training_samples']} programs from "
        f"{report['training_file']} -> {report['model']}"
    )
