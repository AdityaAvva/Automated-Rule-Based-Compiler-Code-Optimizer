"""Build and evaluate the separate, provenance-aware real-world dataset."""

import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
PROCESSED = ROOT / "datasets" / "processed"
SPLITS = ROOT / "datasets" / "splits"
METADATA = ROOT / "datasets" / "metadata"
RESULTS = ROOT / "results" / "real_world_dataset"
PLOTS = ROOT / "plots" / "real_world"
MODEL_PATH = ROOT / "ml" / "real_world_pass_recommender.joblib"
PASSES = [
    "constant_folding",
    "constant_propagation",
    "algebraic_simplification",
    "common_subexpression_elimination",
    "dead_code_elimination",
    "strength_reduction",
]
ALL_PASSES = ",".join(PASSES)
SEED = 307
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
FEATURE_NAMES = [
    "semicolon_delimited_segment_count",
    "source_character_count",
    "numeric_literal_count",
    "identifier_token_count",
    "arithmetic_operator_count",
    "plus_operator_count",
    "minus_operator_count",
    "multiply_operator_count",
    "divide_operator_count",
    "modulo_operator_count",
    "counted_for_keyword_count",
    "print_keyword_count",
    "selected_named_zero_add_or_multiply_pattern_count",
    "add_zero_pattern_count",
    "multiply_one_pattern_count",
    "literal_binary_assignment_count",
    "identifier_binary_assignment_count",
    "identifier_copy_assignment_count",
]
CONCEPTUAL_REFERENCES = [
    {
        "reference_id": "nptel-code-optimization",
        "source_type": "conceptual_reference",
        "source_title": "Compiler Design: Code Optimization",
        "author_or_organization": "NPTEL",
        "source_url": "https://archive.nptel.ac.in/content/storage2/courses/106104072/chapter_2/2_17.html",
        "chapter_or_location": "Chapter 2, section 2.17",
        "source_checked": True,
        "source_check_result": "Page content checked. It explains common optimization types and gives x = 15 * 3 -> x = 45 as a constant-folding example.",
        "license_information": "Not verified from the checked page.",
        "dataset_origin_program_ids": [],
        "verified_dataset_example_count": 0,
        "execution_status": "No program imported from this page; not applicable.",
    },
    {
        "reference_id": "vassar-local-optimizations",
        "source_type": "conceptual_reference",
        "source_title": "Local Optimization",
        "author_or_organization": "Vassar College, CMPU 331; individual author not identified on the checked page.",
        "source_url": "https://www.cs.vassar.edu/~cs331/lectures/18_optimization_intro.html",
        "chapter_or_location": "Lecture 18; web page has no printed page numbers.",
        "source_checked": True,
        "source_check_result": "Page content checked. It presents basic-block, algebraic simplification, constant folding, copy propagation, common-subexpression elimination, and dead-code examples.",
        "license_information": "Not verified from the checked page.",
        "dataset_origin_program_ids": [],
        "verified_dataset_example_count": 0,
        "execution_status": "No program imported from this page; not applicable.",
    },
    {
        "reference_id": "inflibnet-code-optimization",
        "source_type": "conceptual_reference",
        "source_title": "Code Optimization",
        "author_or_organization": "INFLIBNET Centre",
        "source_url": "https://ebooks.inflibnet.ac.in/csp10/chapter/code-optimization/",
        "chapter_or_location": "Module 34, sections 34.1–34.4.",
        "source_checked": True,
        "source_check_result": "Page content checked. It discusses function-preserving transformations and loop optimization, with quicksort-derived basic-block examples.",
        "license_information": "Not verified from the checked page.",
        "dataset_origin_program_ids": [],
        "verified_dataset_example_count": 0,
        "execution_status": "No program imported from this page; not applicable.",
    },
    {
        "reference_id": "llvm-testing-infrastructure-guide",
        "source_type": "conceptual_reference",
        "source_title": "Testing Guide",
        "author_or_organization": "The LLVM Project",
        "source_url": "https://www.llvm.org/docs/TestingGuide.html",
        "chapter_or_location": "Regression test structure and best practices.",
        "source_checked": True,
        "source_check_result": "Page content checked. It documents LLVM lit regression-test infrastructure and test-authoring conventions; no local dataset program was traced to this guide.",
        "license_information": "Not verified on the checked documentation page; this guide is not treated as source code provenance.",
        "dataset_origin_program_ids": [],
        "verified_dataset_example_count": 0,
        "execution_status": "No program imported from this guide; not applicable.",
    },
    {
        "reference_id": "llvm-test-suite-guide",
        "source_type": "conceptual_reference",
        "source_title": "test-suite Guide",
        "author_or_organization": "The LLVM Project",
        "source_url": "https://llvm.org/docs/TestSuiteGuide.html",
        "chapter_or_location": "Quickstart and test-suite structure.",
        "source_checked": True,
        "source_check_result": "Page content checked. It documents configuring and running the LLVM test suite; the shown examples are commands and result excerpts, not origins for local dataset programs.",
        "license_information": "Not verified on the checked documentation page; this guide is not treated as source code provenance.",
        "dataset_origin_program_ids": [],
        "verified_dataset_example_count": 0,
        "execution_status": "No program imported from this guide; not applicable.",
    },
]

sys.path.insert(0, str(ROOT / "ml"))
from train_model import features  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def optimizer_path():
    return ROOT / ("compiler_optimizer.exe" if sys.platform == "win32" else "compiler_optimizer")


def metric_values(output):
    values = {}
    for line in output.splitlines():
        match = re.fullmatch(r"([a-z_]+)=(-?\d+)", line.strip())
        if match:
            values[match.group(1)] = int(match.group(2))
    return values


def output_section(output, start, end):
    match = re.search(re.escape(start) + r"\n(.*?)" + re.escape(end), output, re.S)
    return match.group(1) if match else None


def run_compiler(source, name, passes=None):
    path = RESULTS / "inputs" / f"{name}.src"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    command = [str(optimizer_path())]
    if passes is not None:
        command.extend(["--passes", passes])
    command.append(str(path))
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not start the C++ optimizer: {exc}") from exc
    return completed, metric_values(completed.stdout)


def compatibility(source, name):
    if not source or not source.strip():
        return {
            "supported_syntax": False,
            "can_parse": False,
            "can_convert_to_tac": False,
            "can_execute": False,
            "output_equivalence_verified": False,
            "reason": "No source text was provided.",
        }, None, {}

    completed, metrics = run_compiler(source, name, passes="")
    original_tac = output_section(completed.stdout, "=== ORIGINAL TAC ===", "=== OPTIMIZED TAC ===")
    original_output = output_section(completed.stdout, "Original output:", "Optimized output:")
    optimized_output = output_section(completed.stdout, "Optimized output:", "Verdict:")
    parsed = completed.returncode == 0 and original_tac is not None
    tac = parsed and "=== OPTIMIZED TAC ===" in completed.stdout
    executed = tac and original_output is not None and optimized_output is not None
    equivalence = executed and "Verdict: PASS" in completed.stdout
    reason = None
    if not parsed:
        reason = completed.stderr.strip() or "The compiler rejected the source or failed to produce TAC."
    elif not executed:
        reason = "The compiler could not execute the parsed program."
    elif not equivalence:
        reason = "The compiler did not verify equivalent original and unoptimized outputs."
    return {
        "supported_syntax": parsed,
        "can_parse": parsed,
        "can_convert_to_tac": tac,
        "can_execute": executed,
        "output_equivalence_verified": equivalence,
        "reason": reason,
    }, completed.stdout, metrics


def pass_labels(output, metrics):
    labels = {}
    for name in PASSES:
        transformations = metrics.get(f"{name}_transformations", 0)
        instructions_removed = metrics.get(f"{name}_instructions_removed", 0)
        operations_removed = metrics.get(f"{name}_operations_removed", 0)
        temporaries_removed = metrics.get(f"{name}_temporaries_removed", 0)
        code_characters_removed = metrics.get(f"{name}_code_characters_removed", 0)
        equivalent = metrics.get(f"{name}_output_match", 0) == 1
        labels[name] = {
            "applicable": transformations > 0,
            "transformation_count": transformations,
            "instructions_removed": instructions_removed,
            "operations_removed": operations_removed,
            "temporaries_removed": temporaries_removed,
            "code_characters_removed": code_characters_removed,
            "output_equivalence_verified": equivalent,
            "measurable_static_reduction": bool(
                transformations > 0
                and equivalent
                and any(value > 0 for value in (
                    instructions_removed,
                    operations_removed,
                    temporaries_removed,
                    code_characters_removed,
                ))
            ),
            "execution_speedup_verified": False,
            "label_basis": "isolated single-pass run on original TAC",
        }
    if "Verdict: PASS" not in output:
        for label in labels.values():
            label["output_equivalence_verified"] = False
            label["measurable_static_reduction"] = False
    return labels


def normalize_source(source):
    return re.sub(r"\s+", "", source).lower()


def load_raw_records():
    records = []
    for filename in (
        "textbook_examples.json",
        "compiler_tests.json",
        "synthetic_examples.json",
    ):
        records.extend(read_json(RAW / filename))
    required = {
        "id",
        "source_type",
        "source_family",
        "category",
        "adapted_source",
        "source_title",
        "license",
        "adaptation_notes",
        "optimization_patterns",
    }
    ids = set()
    normalized = set()
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{record.get('id', '<unknown>')}: missing fields {sorted(missing)}")
        if record["id"] in ids:
            raise ValueError(f"Duplicate dataset ID: {record['id']}")
        ids.add(record["id"])
        source = record.get("adapted_source") or record.get("original_source") or ""
        if source.strip():
            normalized_source = normalize_source(source)
            if normalized_source in normalized:
                record["_duplicate_source"] = True
            else:
                normalized.add(normalized_source)
    return records


def process_records(records):
    compatible = []
    rejected = []
    original_source_checks = []
    for record in records:
        item = dict(record)
        item.pop("_duplicate_source", None)
        source = record.get("adapted_source")
        check, output, metrics = compatibility(source, record["id"] + "-adapted")
        item["compatibility"] = check
        if record.get("original_source") and record.get("source_type") != "generated":
            original_check, _, _ = compatibility(
                record["original_source"], record["id"] + "-original"
            )
            item["original_source_compatibility"] = original_check
            original_source_checks.append({
                "id": record["id"],
                "supported": original_check["supported_syntax"],
                "reason": original_check["reason"],
            })
        else:
            item["original_source_compatibility"] = None

        if record.get("_duplicate_source"):
            check = dict(check)
            check["reason"] = "Duplicate normalized program source."
            item["compatibility"] = check
        if not all((
            check["supported_syntax"],
            check["can_parse"],
            check["can_convert_to_tac"],
            check["can_execute"],
            check["output_equivalence_verified"],
        )):
            rejected.append(item)
            continue

        vector = features(source)
        if len(vector) != len(FEATURE_NAMES):
            raise RuntimeError(
                f"{record['id']}: feature extractor returned {len(vector)} values, "
                f"expected {len(FEATURE_NAMES)}."
            )
        if output is None:
            raise RuntimeError(f"{record['id']}: compatible source has no compiler output.")
        item["program_size"] = {
            "characters": len(source),
            "lines": len(source.splitlines()),
        }
        item["features"] = {
            "values": [float(value) for value in vector],
            "names": FEATURE_NAMES,
            "extractor": "ml.train_model.features (pre-optimization source text only)",
            "feature_count": len(vector),
        }
        item["optimization_pass_labels"] = pass_labels(output, metrics)
        original_output = output_section(output, "Original output:", "Optimized output:")
        unoptimized_output = output_section(output, "Optimized output:", "Verdict:")
        item["actual_optimization_results"] = {
            "baseline_verdict": "PASS",
            "labels_are_heuristic": False,
            "expected_output": record.get("expected_output"),
            "original_output": original_output,
            "no_pass_optimized_output": unoptimized_output,
            "baseline_output_equivalence_verified": original_output == unoptimized_output,
            "instruction_count_before": metrics.get("instructions_before"),
            "instruction_count_after_no_passes": metrics.get("instructions_after"),
            "all_pass_measurements": {
                name: item["optimization_pass_labels"][name] for name in PASSES
            },
        }
        item["validation_status"] = "compatible_and_equivalence_checked"
        item["normalized_source"] = normalize_source(source)
        compatible.append(item)
    return compatible, rejected, original_source_checks


def make_splits(records):
    families = sorted({record["source_family"] for record in records})
    if len(families) < 3:
        raise ValueError(
            "At least three independent source families are required to create train, "
            "validation, and test splits without family leakage."
        )

    rng = np.random.RandomState(SEED)
    rng.shuffle(families)
    family_count = len(families)
    train_count = max(1, int(round(family_count * SPLIT_RATIOS["train"])))
    validation_count = max(1, int(round(family_count * SPLIT_RATIOS["validation"])))
    while train_count + validation_count >= family_count:
        if train_count > 1:
            train_count -= 1
        else:
            validation_count -= 1
    family_splits = {
        "train": families[:train_count],
        "validation": families[train_count:train_count + validation_count],
        "test": families[train_count + validation_count:],
    }
    split_records = {
        split: [record for record in records if record["source_family"] in assigned]
        for split, assigned in family_splits.items()
    }
    memberships = {}
    normalized_memberships = {}
    for split, split_items in split_records.items():
        for item in split_items:
            memberships[item["id"]] = split
            normalized_memberships.setdefault(item["normalized_source"], set()).add(split)
    family_sets = {
        split: {item["source_family"] for item in split_items}
        for split, split_items in split_records.items()
    }
    if len(memberships) != len(records):
        raise ValueError("A compatible record was not assigned to exactly one split.")
    if len(set(memberships)) != len(memberships):
        raise ValueError("Duplicate program IDs appeared across dataset splits.")
    if any(len(split_names) > 1 for split_names in normalized_memberships.values()):
        raise ValueError("Duplicate normalized source appeared across dataset splits.")
    if any(family_sets[left] & family_sets[right] for left in family_sets for right in family_sets if left != right):
        raise ValueError("A source family appeared in more than one split.")
    return split_records, family_splits


def make_model(x_train, y_train):
    models = {}
    for index, name in enumerate(PASSES):
        labels = y_train[:, index]
        if len(np.unique(labels)) < 2:
            models[name] = {"constant": int(labels[0])}
        else:
            classifier = RandomForestClassifier(
                n_estimators=120,
                max_depth=5,
                random_state=SEED,
                class_weight="balanced",
            )
            classifier.fit(x_train, labels)
            models[name] = classifier
    return models


def positive_probability(model, x):
    if isinstance(model, dict):
        return float(model["constant"])
    probabilities = model.predict_proba(x.reshape(1, -1))[0]
    classes = list(model.classes_)
    return float(probabilities[classes.index(1)]) if 1 in classes else 0.0


def tune_thresholds(models, validation):
    thresholds = {}
    details = {}
    x_val = np.asarray([item["features"]["values"] for item in validation], dtype=float)
    y_val = np.asarray([
        [int(item["optimization_pass_labels"][name]["measurable_static_reduction"]) for name in PASSES]
        for item in validation
    ], dtype=int)
    for index, name in enumerate(PASSES):
        probabilities = np.asarray([
            positive_probability(models[name], vector) for vector in x_val
        ])
        if len(np.unique(y_val[:, index])) < 2:
            thresholds[name] = 0.5
            details[name] = {
                "threshold": 0.5,
                "status": "validation labels contain one class; default threshold retained",
            }
            continue
        candidates = [value / 10 for value in range(1, 10)]
        scores = [
            f1_score(y_val[:, index], probabilities >= threshold, zero_division=0)
            for threshold in candidates
        ]
        best_index = max(
            range(len(candidates)),
            key=lambda i: (scores[i], -abs(candidates[i] - 0.5)),
        )
        thresholds[name] = candidates[best_index]
        details[name] = {
            "threshold": candidates[best_index],
            "validation_f1": float(scores[best_index]),
            "status": "selected by validation F1",
        }
    return thresholds, details


def label_matrix(records):
    return np.asarray([
        [int(item["optimization_pass_labels"][name]["measurable_static_reduction"]) for name in PASSES]
        for item in records
    ], dtype=int)


def evaluate_model(models, thresholds, records):
    x = np.asarray([item["features"]["values"] for item in records], dtype=float)
    truth = label_matrix(records)
    probabilities = np.asarray([
        [positive_probability(models[name], vector) for name in PASSES]
        for vector in x
    ])
    predictions = np.asarray([
        [int(probabilities[row, col] >= thresholds[name]) for col, name in enumerate(PASSES)]
        for row in range(len(records))
    ], dtype=int)
    per_pass = {}
    aggregate_confusion = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    for index, name in enumerate(PASSES):
        tp = int(np.sum((truth[:, index] == 1) & (predictions[:, index] == 1)))
        fp = int(np.sum((truth[:, index] == 0) & (predictions[:, index] == 1)))
        tn = int(np.sum((truth[:, index] == 0) & (predictions[:, index] == 0)))
        fn = int(np.sum((truth[:, index] == 1) & (predictions[:, index] == 0)))
        per_pass[name] = {
            "precision": float(precision_score(truth[:, index], predictions[:, index], zero_division=0)),
            "recall": float(recall_score(truth[:, index], predictions[:, index], zero_division=0)),
            "f1": float(f1_score(truth[:, index], predictions[:, index], zero_division=0)),
            "support": int(truth[:, index].sum()),
            "predicted_positive": int(predictions[:, index].sum()),
            "confusion": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        }
        for key, value in per_pass[name]["confusion"].items():
            aggregate_confusion[key] += value
    result = {
        "program_count": len(records),
        "exact_match_ratio": float(np.mean(np.all(truth == predictions, axis=1))),
        "micro_precision": float(precision_score(truth, predictions, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(truth, predictions, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(truth, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(truth, predictions)),
        "per_pass": per_pass,
        "aggregate_confusion": aggregate_confusion,
        "limitation": "Small held-out counts and sparse pass labels make these estimates descriptive, not reliable generalization claims.",
    }
    prediction_records = []
    for index, item in enumerate(records):
        prediction_records.append({
            "id": item["id"],
            "source_type": item["source_type"],
            "category": item["category"],
            "source_family": item["source_family"],
            "actual_labels": {
                name: int(truth[index, column]) for column, name in enumerate(PASSES)
            },
            "probabilities": {
                name: float(probabilities[index, column]) for column, name in enumerate(PASSES)
            },
            "recommendations": {
                name: bool(predictions[index, column]) for column, name in enumerate(PASSES)
            },
            "recommended_passes": [
                name for column, name in enumerate(PASSES) if predictions[index, column]
            ],
        })
    return result, prediction_records, truth, predictions


def optimize_test_programs(records, predictions):
    prediction_by_id = {item["id"]: item for item in predictions}
    results = []
    for record in records:
        prediction = prediction_by_id[record["id"]]
        recommended = prediction["recommended_passes"]
        recommendation_run, recommendation_metrics = run_compiler(
            record["adapted_source"], record["id"] + "-recommended", ",".join(recommended)
        )
        all_pass_run, all_metrics = run_compiler(
            record["adapted_source"], record["id"] + "-all-passes", ALL_PASSES
        )
        baseline, _ = run_compiler(record["adapted_source"], record["id"] + "-baseline", "")
        baseline_output = output_section(baseline.stdout, "Original output:", "Optimized output:")
        outputs = {}
        for label, completed in (
            ("recommended", recommendation_run),
            ("all_passes", all_pass_run),
        ):
            optimized_output = output_section(completed.stdout, "Optimized output:", "Verdict:")
            outputs[label] = {
                "return_code": completed.returncode,
                "equivalent_to_original": (
                    completed.returncode == 0
                    and baseline_output is not None
                    and optimized_output == baseline_output
                    and "Verdict: PASS" in completed.stdout
                ),
            }
        results.append({
            "id": record["id"],
            "source_type": record["source_type"],
            "category": record["category"],
            "recommended_passes": recommended,
            "recommended_metrics": recommendation_metrics,
            "all_pass_metrics": all_metrics,
            "output_equivalence": outputs,
            "recommended_transformations": {
                name: recommendation_metrics.get(name, 0) for name in PASSES
            },
            "all_pass_transformations": {
                name: all_metrics.get(name, 0) for name in PASSES
            },
            "timing_note": "Compiler-reported timings are from one small run and are descriptive only.",
        })
    return results


def split_counts(records):
    return {
        split: {
            "programs": len(items),
            "source_families": len({item["source_family"] for item in items}),
            "source_types": dict(Counter(item["source_type"] for item in items)),
        }
        for split, items in records.items()
    }


def build_source_manifest(raw_records, processed_records):
    processed_by_id = {item["id"]: item for item in processed_records}
    programs = []
    for item in raw_records:
        processed = processed_by_id.get(item["id"])
        source_type = item["source_type"]
        is_generated = source_type == "generated"
        source_check_result = (
            "Verified as a project-generated record; no external source is claimed."
            if is_generated
            else (
                "Pinned source URL, repository file, revision, and available repository-level "
                "license record were checked."
            )
        )
        compatibility = processed.get("compatibility") if processed else None
        original_compatibility = (
            processed.get("original_source_compatibility") if processed else None
        )
        adapted_executed = bool(
            compatibility
            and compatibility["can_execute"]
            and compatibility["output_equivalence_verified"]
        )
        original_executed = (
            None if not item.get("original_source")
            else bool(
                original_compatibility
                and original_compatibility["can_execute"]
                and original_compatibility["output_equivalence_verified"]
            )
        )
        programs.append({
            "record_type": "dataset_program",
            "id": item["id"],
            "source_type": source_type,
            "source_family": item.get("source_family"),
            "category": item.get("category"),
            "source_title": item.get("source_title"),
            "author_or_organization": item.get("author_or_organization"),
            "document_title": item.get("document_title"),
            "edition": item.get("edition"),
            "chapter": item.get("chapter"),
            "page": item.get("page"),
            "source_url": item.get("source_url"),
            "repository": item.get("repository"),
            "repository_file_path": item.get("file_path"),
            "revision": item.get("revision"),
            "license_information": item.get("license"),
            "license_url": item.get("license_url"),
            "license_verification_note": (
                "Repository-level license was checked; the individual file's copyright or "
                "assignment status was not independently verified."
                if source_type == "compiler_test" and item.get("repository") == "gcc-mirror/gcc"
                else (
                    "Repository-level license was checked; no file-specific header was "
                    "present in the excerpt."
                    if source_type == "compiler_test"
                    else "Project-generated; no external license applies."
                )
            ),
            "original_source_code": item.get("original_source"),
            "adapted_source_code": item.get("adapted_source"),
            "adaptation_description": item.get("adaptation_notes"),
            "optimization_patterns": item.get("optimization_patterns", []),
            "source_actually_checked": True,
            "source_check_result": source_check_result,
            "original_program_executed_successfully": original_executed,
            "adapted_program_executed_successfully": adapted_executed,
            "execution_result": (
                compatibility if compatibility else {
                    "status": "not_evaluated",
                    "reason": "Program is not present in the compatible/rejected output."
                }
            ),
            "original_program_execution_result": original_compatibility,
            "dataset_origin_verified": not is_generated,
        })

    references = [dict(reference) for reference in CONCEPTUAL_REFERENCES]
    return {
        "schema_version": "1.0",
        "manifest_scope": (
            "Existing real-world dataset only. Conceptual references are listed separately "
            "and are not asserted as origins for dataset programs."
        ),
        "dataset_program_count": len(programs),
        "dataset_programs_by_source_type": dict(Counter(item["source_type"] for item in programs)),
        "verified_dataset_origin_counts": {
            "llvm_compiler_test": sum(
                item["source_type"] == "compiler_test"
                and item.get("repository") == "llvm/llvm-project"
                for item in raw_records
            ),
            "gcc_compiler_test": sum(
                item["source_type"] == "compiler_test"
                and item.get("repository") == "gcc-mirror/gcc"
                for item in raw_records
            ),
            "generated": sum(item["source_type"] == "generated" for item in raw_records),
            "textbook_or_lecture_origins_from_supplied_candidates": 0,
            "llvm_testing_guides_as_program_origins": 0,
        },
        "conceptual_references": references,
        "programs": programs,
    }


def create_plots(records, rejected, splits, evaluation, predictions, effectiveness):
    PLOTS.mkdir(parents=True, exist_ok=True)
    sizes = [len(splits[name]) for name in ("train", "validation", "test")]

    def save(name):
        plt.tight_layout()
        plt.savefig(PLOTS / name, dpi=140)
        plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["Train", "Validation", "Test"], sizes, color=["#4472c4", "#ed7d31", "#70ad47"])
    plt.ylabel("Programs")
    plt.title("Real-world dataset split sizes")
    save("01_split_sizes.png")

    sources = Counter(item["source_type"] for item in records)
    plt.figure(figsize=(7, 4))
    plt.bar(list(sources), list(sources.values()), color="#4472c4")
    plt.ylabel("Programs")
    plt.title("Dataset distribution by source")
    plt.xticks(rotation=20, ha="right")
    save("02_source_distribution.png")

    patterns = Counter(pattern for item in records for pattern in item["optimization_patterns"])
    plt.figure(figsize=(9, 4))
    plt.bar(list(patterns), list(patterns.values()), color="#70ad47")
    plt.ylabel("Programs tagged")
    plt.title("Dataset distribution by optimization pattern")
    plt.xticks(rotation=30, ha="right")
    save("03_pattern_distribution.png")

    plt.figure(figsize=(6, 4))
    plt.bar(["Compatible", "Rejected"], [len(records), len(rejected)], color=["#70ad47", "#c00000"])
    plt.ylabel("Programs")
    plt.title("Compatibility-check results")
    save("04_compatibility.png")

    x = np.arange(len(PASSES))
    width = 0.25
    plt.figure(figsize=(10, 5))
    for offset, metric, color in ((-width, "precision", "#4472c4"), (0, "recall", "#ed7d31"), (width, "f1", "#70ad47")):
        values = [evaluation["per_pass"][name][metric] for name in PASSES]
        plt.bar(x + offset, values, width, label=metric.title(), color=color)
    plt.xticks(x, PASSES, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Held-out multi-label scores by pass")
    plt.legend()
    save("05_per_pass_metrics.png")

    confusion = evaluation["aggregate_confusion"]
    matrix = np.asarray([
        [confusion["true_negative"], confusion["false_positive"]],
        [confusion["false_negative"], confusion["true_positive"]],
    ])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center")
    plt.xticks([0, 1], ["Not recommended", "Recommended"])
    plt.yticks([0, 1], ["Not beneficial", "Beneficial"])
    plt.title("Aggregate pass-label confusion matrix")
    save("06_multilabel_confusion.png")

    actual_applied = Counter()
    recommended_counts = Counter()
    for item in effectiveness:
        for name in item["recommended_passes"]:
            recommended_counts[name] += 1
        for name, count in item["recommended_transformations"].items():
            if count:
                actual_applied[name] += 1
    plt.figure(figsize=(10, 5))
    x = np.arange(len(PASSES))
    plt.bar(x - width / 2, [recommended_counts[name] for name in PASSES], width, label="Recommended")
    plt.bar(x + width / 2, [actual_applied[name] for name in PASSES], width, label="At least one transformation")
    plt.xticks(x, PASSES, rotation=30, ha="right")
    plt.ylabel("Held-out programs")
    plt.title("ML recommendations versus transformations applied")
    plt.legend()
    save("07_recommended_vs_applied.png")

    ids = [item["id"] for item in effectiveness]
    ml_reduction = [
        item["recommended_metrics"].get("instructions_before", 0)
        - item["recommended_metrics"].get("instructions_after", 0)
        for item in effectiveness
    ]
    plt.figure(figsize=(9, 4))
    plt.bar(ids, ml_reduction, color="#4472c4")
    plt.axhline(0, color="black", linewidth=0.7)
    plt.ylabel("Instructions removed")
    plt.title("Held-out instruction-count change: ML-selected passes")
    plt.xticks(rotation=25, ha="right")
    save("08_test_instruction_reduction.png")

    ml_after = [item["recommended_metrics"].get("instructions_after", 0) for item in effectiveness]
    all_after = [item["all_pass_metrics"].get("instructions_after", 0) for item in effectiveness]
    plt.figure(figsize=(9, 4))
    x = np.arange(len(ids))
    plt.bar(x - width / 2, ml_after, width, label="ML-selected")
    plt.bar(x + width / 2, all_after, width, label="All six")
    plt.xticks(x, ids, rotation=25, ha="right")
    plt.ylabel("Optimized TAC instructions")
    plt.title("ML-selected passes versus all six passes")
    plt.legend()
    save("09_ml_vs_all_passes.png")

    equivalence = [
        sum(item["output_equivalence"][name]["equivalent_to_original"] for item in effectiveness)
        for name in ("recommended", "all_passes")
    ]
    plt.figure(figsize=(6, 4))
    plt.bar(["ML-selected", "All six"], equivalence, color=["#4472c4", "#70ad47"])
    plt.ylabel("Equivalent outputs / held-out programs")
    plt.ylim(0, max(1, len(effectiveness)))
    plt.title("Output-equivalence checks")
    save("10_output_equivalence.png")


def write_effectiveness_csv(effectiveness):
    if not effectiveness:
        return
    fields = sorted({key for item in effectiveness for key in item})
    with (RESULTS / "test_effectiveness.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in effectiveness:
            row = dict(item)
            for key in ("recommended_metrics", "all_pass_metrics", "output_equivalence",
                        "recommended_transformations", "all_pass_transformations"):
                row[key] = json.dumps(row[key], sort_keys=True)
            row["recommended_passes"] = ";".join(row["recommended_passes"])
            writer.writerow(row)


def main():
    if not optimizer_path().is_file():
        raise SystemExit(
            f"C++ optimizer not found at {optimizer_path()}; compile src/main.cpp first."
        )
    for directory in (PROCESSED, SPLITS, METADATA, RESULTS, PLOTS):
        directory.mkdir(parents=True, exist_ok=True)

    raw_records = load_raw_records()
    compatible, rejected, original_checks = process_records(raw_records)
    if len(compatible) < 3:
        raise SystemExit("Fewer than three compatible programs; cannot train and hold out a model.")
    split_records, family_splits = make_splits(compatible)
    write_json(PROCESSED / "compatible_programs.json", compatible)
    write_json(PROCESSED / "rejected_programs.json", rejected)
    for split, items in split_records.items():
        write_json(SPLITS / f"{split}.json", [item["id"] for item in items])
    source_manifest = build_source_manifest(raw_records, [*compatible, *rejected])
    write_json(METADATA / "source_manifest.json", source_manifest)

    train = split_records["train"]
    validation = split_records["validation"]
    test = split_records["test"]
    x_train = np.asarray([item["features"]["values"] for item in train], dtype=float)
    y_train = label_matrix(train)
    models = make_model(x_train, y_train)
    thresholds, threshold_details = tune_thresholds(models, validation)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": models,
        "passes": PASSES,
        "thresholds": thresholds,
        "feature_count": int(x_train.shape[1]),
        "feature_names": FEATURE_NAMES,
        "feature_extractor": "ml.train_model.features",
        "training_ids": [item["id"] for item in train],
        "seed": SEED,
        "label_definition": "An isolated pass is positive only when it transforms TAC, preserves output, and reduces at least one static metric (instructions, operations, temporaries, or TAC characters). This is not an execution-speedup label.",
    }, MODEL_PATH)

    validation_metrics, _, _, _ = evaluate_model(models, thresholds, validation)
    test_metrics, predictions, _, _ = evaluate_model(models, thresholds, test)
    write_json(RESULTS / "test_predictions.json", predictions)
    effectiveness = optimize_test_programs(test, predictions)
    write_effectiveness_csv(effectiveness)
    write_json(RESULTS / "test_effectiveness.json", effectiveness)

    compatibility_by_source = Counter(item["source_type"] for item in compatible)
    source_counts = Counter(item["source_type"] for item in raw_records)
    pattern_counts = Counter(
        pattern for item in compatible for pattern in item["optimization_patterns"]
    )
    stats = {
        "schema_version": "1.0",
        "raw_program_count": len(raw_records),
        "compatible_program_count": len(compatible),
        "rejected_program_count": len(rejected),
        "compatibility_rate": len(compatible) / len(raw_records) if raw_records else 0.0,
        "original_external_excerpt_compatibility": {
            "checked": len(original_checks),
            "supported": sum(check["supported"] for check in original_checks),
            "unsupported": sum(not check["supported"] for check in original_checks),
            "details": original_checks,
        },
        "counts_by_source_type": dict(source_counts),
        "compatible_counts_by_source_type": dict(compatibility_by_source),
        "source_provenance_counts": {
            "verified_dataset_origins": source_manifest["verified_dataset_origin_counts"],
            "conceptual_references_checked": len(source_manifest["conceptual_references"]),
            "conceptual_references_with_dataset_origin": sum(
                bool(reference["dataset_origin_program_ids"])
                for reference in source_manifest["conceptual_references"]
            ),
        },
        "counts_by_optimization_pattern": dict(pattern_counts),
        "unique_compatible_source_families": len({item["source_family"] for item in compatible}),
        "split_counts": split_counts(split_records),
        "requested_split_ratios": SPLIT_RATIOS,
        "actual_split_ratios": {
            split: len(items) / len(compatible) for split, items in split_records.items()
        },
        "family_assignments": family_splits,
        "leakage_checks": {
            "unique_ids_across_splits": True,
            "normalized_source_overlap": False,
            "source_family_overlap": False,
            "random_seed": SEED,
        },
        "training_program_count": len(train),
        "validation_program_count": len(validation),
        "testing_program_count": len(test),
        "training_source_types": dict(Counter(item["source_type"] for item in train)),
        "validation_source_types": dict(Counter(item["source_type"] for item in validation)),
        "testing_source_types": dict(Counter(item["source_type"] for item in test)),
        "feature_count": int(x_train.shape[1]),
        "feature_names": FEATURE_NAMES,
        "features": "Existing source-text regex feature extractor; outputs and post-optimization metrics are excluded. Names document the exact positional output.",
        "label_definition": "Positive means the independently applied pass transforms TAC, preserves interpreter output, and reduces at least one static metric (instructions, operations, temporaries, or TAC characters). It does not mean runtime speedup.",
        "model": "RandomForestClassifier per pass; train split only; thresholds selected on validation split; final metrics use test split only.",
        "validation_metrics": validation_metrics,
        "threshold_selection": threshold_details,
        "test_metrics": test_metrics,
        "optimization_effectiveness": {
            "held_out_programs": len(effectiveness),
            "recommended_output_equivalent": sum(
                item["output_equivalence"]["recommended"]["equivalent_to_original"]
                for item in effectiveness
            ),
            "all_passes_output_equivalent": sum(
                item["output_equivalence"]["all_passes"]["equivalent_to_original"]
                for item in effectiveness
            ),
            "comparison_note": "ML recommendation quality and rule-optimizer effectiveness are separate. Code-size/instruction changes are not evidence of execution speedup.",
        },
        "limitations": [
            "No textbook examples were added: no suitable textbook source and reuse license were verified during this run. The raw textbook file is an empty import point.",
            "External C and LLVM IR excerpts are not accepted by the existing compiler. Only separately stored, explicitly documented adaptations are evaluated.",
            "External adaptations use fixed values because the language has no functions or runtime inputs; they are not equivalent translations for every source input.",
            "The dataset and family-level holdout are small; held-out scores are descriptive only and cannot establish generalization.",
            "The underlying feature extractor is the current source-text heuristic extractor, not a new parser-derived TAC feature set.",
            "Tiny interpreter and optimization timings are noisy; no execution speedup is claimed.",
            "Loop constructs are expanded before TAC optimization; loop metrics do not represent retained loop-control-flow optimization.",
        ],
        "files": {
            "model": str(MODEL_PATH.relative_to(ROOT)),
            "predictions": str((RESULTS / "test_predictions.json").relative_to(ROOT)),
            "effectiveness": str((RESULTS / "test_effectiveness.json").relative_to(ROOT)),
            "plots": str(PLOTS.relative_to(ROOT)),
        },
    }
    write_json(METADATA / "dataset_statistics.json", stats)
    create_plots(compatible, rejected, split_records, test_metrics, predictions, effectiveness)

    print(json.dumps({
        "compatible_programs": len(compatible),
        "rejected_programs": len(rejected),
        "source_counts": dict(source_counts),
        "splits": {name: len(items) for name, items in split_records.items()},
        "unique_source_families": stats["unique_compatible_source_families"],
        "test_metrics": test_metrics,
        "optimization_effectiveness": stats["optimization_effectiveness"],
        "statistics_file": str(METADATA / "dataset_statistics.json"),
        "model_file": str(MODEL_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
