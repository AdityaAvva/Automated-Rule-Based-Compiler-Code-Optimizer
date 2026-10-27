"""Train the legacy recommender from compiler-derived training-only targets."""

import hashlib
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_benchmark_ml import PASSES, label_programs

TRAINING_FILE = ROOT / "datasets" / "training" / "programs.json"
BENCHMARK_DATASET = ROOT / "benchmarks" / "dataset.json"
BENCHMARK_MANIFEST = ROOT / "benchmarks" / "manifest.json"
OUT = ROOT / "results"
MODEL = ROOT / "ml" / "pass_recommender.joblib"
TRAINING_FIELDS = {
    "id", "category", "size", "source", "expected_output", "inputs",
}


def features(src):
    text = src.lower()
    ops = {op: len(re.findall(re.escape(op), text)) for op in ["+", "-", "*", "/", "%"]}
    return [
        len(text.split(";")), len(text), len(re.findall(r"\b\d+(?:\.\d+)?\b", text)),
        len(re.findall(r"\b[a-z_]\w*\b", text)), sum(ops.values()),
        ops["+"], ops["-"], ops["*"], ops["/"], ops["%"],
        len(re.findall(r"\bfor\s*\(", text)),
        len(re.findall(r"\bprint\b", text)),
        len(re.findall(r"\b(?:x|y|z|t\d+)\s*=\s*[^;]*[+*]\s*0\b", text)),
        len(re.findall(r"\b\w+\s*=\s*\w+\s*\+\s*0\b", text)),
        len(re.findall(r"\b\w+\s*=\s*\w+\s*\*\s*1\b", text)),
        len(re.findall(r"\b\w+\s*=\s*\d+\s*[*+/-]\s*\d+", text)),
        len(re.findall(r"\b\w+\s*=\s*\w+\s*[+*]\s*\w+", text)),
        len(re.findall(r"\b\w+\s*=\s*\w+", text)),
    ]


def main():
    training_bytes = TRAINING_FILE.read_bytes()
    training = json.loads(training_bytes.decode("utf-8"))
    if len(training) != 400 or any(set(record) != TRAINING_FIELDS for record in training):
        raise ValueError("Training input must contain 400 records with exactly six fields.")

    metadata = json.loads(BENCHMARK_MANIFEST.read_text(encoding="utf-8"))["program_metadata"]
    metadata_by_id = {record["id"]: record for record in metadata}
    training_ids = {record["id"] for record in training}
    if len(training_ids) != len(training):
        raise ValueError("Training dataset contains duplicate IDs.")
    if any(
        metadata_by_id.get(record["id"], {}).get("split") != "training"
        for record in training
    ):
        raise ValueError("Training JSON includes a record not assigned to the training split.")

    analysis_records = [
        {**record, "family_id": metadata_by_id[record["id"]]["family_id"], "split": "training"}
        for record in training
    ]
    compiler_analysis = label_programs(analysis_records)
    matrix = np.asarray([features(record["source"]) for record in training], dtype=float)
    labels = np.asarray([
        [
            int(compiler_analysis[record["id"]]["applicability_labels"][name])
            for name in PASSES
        ]
        for record in training
    ], dtype=int)

    models = {}
    report = {}
    for column, name in enumerate(PASSES):
        target = labels[:, column]
        classes = np.unique(target)
        if len(classes) == 1:
            models[name] = {"constant": int(classes[0])}
            report[name] = {
                "status": "constant training labels; stored prevalence baseline",
                "positive_training_labels": int(target.sum()),
            }
            continue
        classifier = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            random_state=307,
            class_weight="balanced",
        )
        classifier.fit(matrix, target)
        models[name] = classifier
        report[name] = {
            "status": "trained",
            "positive_training_labels": int(target.sum()),
        }

    bundle = {
        "models": models,
        "passes": PASSES,
        "feature_schema": "legacy_source_counts_v2",
        "training_ids": sorted(training_ids),
        "training_data_sha256": hashlib.sha256(training_bytes).hexdigest(),
        "training_split_only": True,
    }
    joblib.dump(bundle, MODEL)

    canonical = json.loads(BENCHMARK_DATASET.read_text(encoding="utf-8"))
    predictions = []
    for program in canonical:
        vector = np.asarray(features(program["source"]), dtype=float).reshape(1, -1)
        recommendations = {}
        for name in PASSES:
            model = models[name]
            if isinstance(model, dict):
                prediction = int(model["constant"])
                score = float(prediction)
            else:
                prediction = int(model.predict(vector)[0])
                score = float(model.predict_proba(vector)[0][1])
            recommendations[name] = {
                "recommended": bool(prediction),
                "score": round(score, 4),
            }
        predictions.append({
            "id": program["id"],
            "category": program["category"],
            "recommendations": recommendations,
        })
    OUT.mkdir(exist_ok=True)
    (OUT / "ml_recommendations.json").write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "ml_training_summary.json").write_text(
        json.dumps({
            "model": "RandomForestClassifier per optimization pass",
            "training_file": str(TRAINING_FILE.relative_to(ROOT)),
            "training_samples": len(training),
            "training_ids": sorted(training_ids),
            "training_data_sha256": bundle["training_data_sha256"],
            "features_per_program": int(matrix.shape[1]),
            "passes": PASSES,
            "pass_details": report,
            "training_split_only": True,
            "label_method": (
                "Each pass is run individually on original compiler TAC; the target is "
                "positive only when it rewrites while preserving output."
            ),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Trained/saved model from {len(training)} programs: {MODEL}")
    print(f"Generated source-only recommendations for {len(predictions)} programs.")


if __name__ == "__main__":
    main()
