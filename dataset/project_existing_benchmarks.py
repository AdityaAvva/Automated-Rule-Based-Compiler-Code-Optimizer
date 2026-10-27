"""Project the generated benchmark split into the strict ML dataset schemas."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = ROOT / "benchmarks" / "dataset.json"
SOURCE_MANIFEST = ROOT / "benchmarks" / "manifest.json"
TRAINING_FILE = ROOT / "dataset" / "training" / "training.json"
TESTING_FILE = ROOT / "dataset" / "testing" / "testing.json"
MANIFEST_FILE = ROOT / "dataset" / "manifest.json"
TRAINING_FIELDS = {
    "id", "category", "size", "source", "expected_output", "inputs",
}


def project_existing_benchmarks():
    records = json.loads(SOURCE_DATASET.read_text(encoding="utf-8"))
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    metadata = source_manifest.get("program_metadata", [])
    metadata_by_id = {item.get("id"): item for item in metadata}
    if len(records) != 500 or len(metadata_by_id) != 500:
        raise ValueError("Expected 500 existing records and corresponding split metadata.")
    if len({record.get("id") for record in records}) != 500:
        raise ValueError("Benchmark IDs must be unique.")

    training = []
    testing = []
    category_counts = {}
    family_splits = {}
    seen_sources = set()
    for record in records:
        record_id = record.get("id")
        if set(record) != TRAINING_FIELDS:
            raise ValueError(f"{record_id}: benchmark source record has an invalid schema.")
        meta = metadata_by_id.get(record_id)
        if not meta or meta.get("split") not in {"training", "testing"}:
            raise ValueError(f"{record_id}: split metadata is missing or invalid.")
        source = record["source"]
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if source_hash != meta.get("source_sha256"):
            raise ValueError(f"{record_id}: source does not match split metadata.")
        if source_hash in seen_sources:
            raise ValueError(f"{record_id}: duplicate source program.")
        seen_sources.add(source_hash)
        family_splits.setdefault(meta["family_id"], set()).add(meta["split"])

        if meta["split"] == "training":
            training.append(record)
        else:
            testing.append({"source": source})

        counts = category_counts.setdefault(
            record["category"], {"training": 0, "testing": 0, "total": 0}
        )
        counts[meta["split"]] += 1
        counts["total"] += 1

    if len(training) != 400 or len(testing) != 100:
        raise ValueError("Existing split must remain exactly 400 training / 100 testing.")
    for category, counts in category_counts.items():
        if counts != {"training": 80, "testing": 20, "total": 100}:
            raise ValueError(f"{category}: expected 80 training and 20 testing; got {counts}.")
    leaked_families = sorted(
        family for family, splits in family_splits.items() if len(splits) != 1
    )
    if leaked_families:
        raise ValueError(f"Program families cross train/test splits: {leaked_families}")

    TRAINING_FILE.parent.mkdir(parents=True, exist_ok=True)
    TESTING_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_FILE.write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    TESTING_FILE.write_text(json.dumps(testing, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "source": "existing benchmarks/dataset.json",
        "program_count": len(records),
        "training_count": len(training),
        "testing_count": len(testing),
        "category_counts": category_counts,
        "training_file": "dataset/training/training.json",
        "testing_file": "dataset/testing/testing.json",
        "training_record_fields": sorted(TRAINING_FIELDS),
        "testing_record_fields": ["source"],
        "training_family_count": sum(
            "training" in splits for splits in family_splits.values()
        ),
        "testing_family_count": sum(
            "testing" in splits for splits in family_splits.values()
        ),
        "family_overlap": [],
        "note": (
            "Projection only; existing programs and split assignments are preserved. "
            "Features and applicability targets are derived by compiler analysis."
        ),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    report = project_existing_benchmarks()
    print(
        f"Projected existing suite: {report['program_count']} total, "
        f"{report['training_count']} training, {report['testing_count']} testing."
    )
