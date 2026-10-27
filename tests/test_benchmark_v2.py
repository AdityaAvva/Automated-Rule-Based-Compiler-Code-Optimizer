import json
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
sys.path.insert(0, str(ROOT / "ml"))
from train_benchmark_model import train as train_grouped_model
import joblib

PASSES = (
    "constant_folding",
    "constant_propagation",
    "algebraic_simplification",
    "common_subexpression_elimination",
    "dead_code_elimination",
    "strength_reduction",
)


class GroupedBenchmarkPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = json.loads(
            (DATASETS / "training" / "programs.json").read_text(encoding="utf-8")
        )
        cls.testing = json.loads(
            (DATASETS / "testing" / "programs.json").read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(
            (DATASETS / "benchmark_manifest.json").read_text(encoding="utf-8")
        )
        cls.benchmark_manifest = json.loads(
            (ROOT / "benchmarks" / "manifest.json").read_text(encoding="utf-8")
        )
        cls.canonical = json.loads(
            (ROOT / "benchmarks" / "dataset.json").read_text(encoding="utf-8")
        )
        cls.evaluation = json.loads(
            (ROOT / "results" / "benchmark_v2_ml_evaluation.json").read_text(encoding="utf-8")
        )
        cls.compiler_results = json.loads(
            (ROOT / "results" / "benchmark_v2_compiler_results.json").read_text(encoding="utf-8")
        )
        cls.training_report = json.loads(
            (ROOT / "results" / "benchmark_v2_training_summary.json").read_text(encoding="utf-8")
        )
        cls.model = joblib.load(ROOT / "ml" / "benchmark_pass_recommender.joblib")

    def test_dataset_and_category_split_are_exactly_balanced(self):
        records = self.canonical
        self.assertEqual(len(records), 500)
        self.assertEqual(len(self.training), 400)
        self.assertEqual(len(self.testing), 100)
        self.assertTrue(all(set(record) == {
            "id", "category", "size", "source", "expected_output", "inputs",
        } for record in records))
        self.assertTrue(all(set(record) == {
            "id", "category", "size", "source", "expected_output", "inputs",
        } for record in self.training))
        self.assertTrue(all(set(record) == {"source"} for record in self.testing))
        self.assertEqual(len({record["id"] for record in records}), 500)
        self.assertEqual(len({record["source"] for record in records}), 500)
        benchmark_files = [
            path for path in (ROOT / "benchmarks").iterdir()
            if path.is_file() and path.name.startswith("CAT") and path.suffix == ".json"
            and path.name != "dataset.json"
        ]
        self.assertEqual(len(benchmark_files), 500)
        self.assertEqual(
            {record["id"] for record in records},
            {path.stem for path in benchmark_files},
        )
        for path in benchmark_files:
            item = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(item), {
                "id", "category", "size", "source", "expected_output", "inputs",
            })
            self.assertTrue(item["source"].strip())
            self.assertIn("expected_output", item)
            self.assertEqual(item["id"], path.stem)
        for category, expected in self.manifest["category_counts"].items():
            self.assertEqual(expected, {"total": 100, "training": 80, "testing": 20})
            self.assertEqual(sum(record["category"] == category for record in records), 100)
            self.assertEqual(sum(record["category"] == category for record in self.training), 80)
            testing_hashes = {
                hashlib.sha256(record["source"].encode("utf-8")).hexdigest()
                for record in self.testing
            }
            testing_count = sum(
                item["category"] == category and item["split"] == "testing"
                and item["source_sha256"] in testing_hashes
                for item in self.benchmark_manifest["program_metadata"]
            )
            self.assertEqual(testing_count, 20)

    def test_no_enriched_records_remain_in_active_or_archived_program_files(self):
        training_fields = {
            "id", "category", "size", "source", "expected_output", "inputs",
        }
        for split, expected_fields in (
            ("training", training_fields),
            ("testing", {"source"}),
        ):
            split_root = DATASETS / split
            self.assertFalse((split_root / "labels.json").exists())
            for category_dir in split_root.iterdir():
                if not category_dir.is_dir():
                    continue
                for path in category_dir.glob("*.json"):
                    record = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(set(record), expected_fields, path)
        archive = DATASETS / "archive" / "legacy_common_subexpressions"
        if archive.exists():
            for filename, expected_fields in (
                ("training.json", training_fields),
                ("testing.json", {"source"}),
            ):
                records = json.loads((archive / filename).read_text(encoding="utf-8"))
                self.assertTrue(records)
                self.assertTrue(all(set(record) == expected_fields for record in records))

        for path in (ROOT / "benchmarks").glob("CAT[1-5]_*.json"):
            self.assertEqual(
                set(json.loads(path.read_text(encoding="utf-8"))),
                training_fields,
                path,
            )

    def test_split_is_reproducible_and_generation_families_do_not_leak(self):
        metadata = self.benchmark_manifest["program_metadata"]
        train_families = {item["family_id"] for item in metadata if item["split"] == "training"}
        test_families = {item["family_id"] for item in metadata if item["split"] == "testing"}
        self.assertFalse(train_families & test_families)

        source_hashes = {
            hashlib.sha256(record["source"].encode("utf-8")).hexdigest()
            for record in self.testing
        }
        self.assertEqual(
            {record["id"]: "training" for record in self.training}
            | {
                item["id"]: item["split"]
                for item in metadata if item["source_sha256"] in source_hashes
            },
            {item["id"]: item["split"] for item in metadata},
        )
        self.assertEqual(self.manifest["seed"], 307)
        self.assertEqual(set(self.manifest["family_assignments"]), {
            "constant_expressions", "algebraic_redundancy", "dead_code",
            "repeated_expressions", "loop_based",
        })

    def test_compiler_confirmed_labels_and_pass_coverage_are_complete(self):
        self.assertTrue(self.manifest["all_programs_supported_by_compiler"])
        self.assertTrue(self.manifest["all_expected_outputs_match_original_compiler"])
        self.assertEqual(set(self.model["passes"]), set(PASSES))
        self.assertEqual(len(self.training_report["training_ids"]), 400)
        self.assertEqual(
            self.training_report["training_data_sha256"],
            hashlib.sha256((DATASETS / "training" / "programs.json").read_bytes()).hexdigest(),
        )
        self.assertTrue(self.benchmark_manifest["compiler_labeling_completed"])
        for pass_name in PASSES:
            self.assertGreater(self.manifest["label_counts"]["testing"][pass_name], 0, pass_name)

    def test_training_model_and_evaluation_use_disjoint_splits(self):
        training_ids = {record["id"] for record in self.training}
        testing_hashes = {
            hashlib.sha256(record["source"].encode("utf-8")).hexdigest()
            for record in self.testing
        }
        testing_ids = {
            item["id"] for item in self.benchmark_manifest["program_metadata"]
            if item["source_sha256"] in testing_hashes
        }
        self.assertTrue(self.training_report["training_split_only"])
        self.assertEqual(set(self.training_report["training_ids"]), training_ids)
        self.assertEqual(set(self.evaluation["training_ids"]), training_ids)
        self.assertEqual(set(self.evaluation["test_ids"]), testing_ids)
        self.assertFalse(training_ids & testing_ids)
        self.assertFalse(self.evaluation["training_test_id_overlap"])
        self.assertFalse(self.evaluation["training_test_family_overlap"])
        self.assertEqual(self.evaluation["evaluation_split"], "testing")
        self.assertEqual(self.evaluation["test_program_count"], 100)
        self.assertEqual(self.model["training_data_sha256"], self.training_report["training_data_sha256"])

    def test_trainer_rejects_testing_data_as_its_input(self):
        testing_path = DATASETS / "testing" / "programs.json"
        with self.assertRaisesRegex(ValueError, "must be under datasets/training"):
            train_grouped_model(
                training_file=testing_path,
                model_file=ROOT / "results" / "test-only-must-not-be-created.joblib",
                summary_file=ROOT / "results" / "test-only-must-not-be-created.json",
            )

    def test_all_testing_compiler_runs_preserve_expected_outputs_and_report_metrics(self):
        self.assertEqual(len(self.compiler_results), 100)
        results_by_id = {item["id"]: item for item in self.compiler_results}
        testing_hashes = {
            hashlib.sha256(record["source"].encode("utf-8")).hexdigest()
            for record in self.testing
        }
        testing_ids = {
            item["id"] for item in self.benchmark_manifest["program_metadata"]
            if item["source_sha256"] in testing_hashes
        }
        self.assertEqual(set(results_by_id), testing_ids)
        for result in self.compiler_results:
            for run_name in ("ml_recommended", "all_passes_baseline"):
                run = result[run_name]
                self.assertEqual(run["verdict"], "PASS", result["id"])
                self.assertTrue(run["output_equivalent"], result["id"])
                self.assertTrue(run["expected_output_match"], result["id"])
                self.assertIsNotNone(run["instructions_before"])
                self.assertIsNotNone(run["instructions_after"])
                self.assertIsNotNone(run["optimization_cycles"])
                self.assertEqual(
                    set(run["actual_rewrite_counts"]),
                    set(PASSES),
                )

    def test_model_metrics_include_per_pass_and_category_confusion_counts(self):
        self.assertEqual(set(self.evaluation["per_pass"]), set(PASSES))
        self.assertEqual(len(self.evaluation["predictions"]), 100)
        flattened_truth = []
        flattened_predictions = []
        for pass_name in PASSES:
            truth = [
                int(row["true_labels"][pass_name])
                for row in self.evaluation["predictions"]
            ]
            predictions = [
                int(row["predicted_labels"][pass_name])
                for row in self.evaluation["predictions"]
            ]
            flattened_truth.extend(truth)
            flattened_predictions.extend(predictions)
            tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(truth, predictions))
            fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(truth, predictions))
            fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(truth, predictions))
            tn = sum(actual == 0 and predicted == 0 for actual, predicted in zip(truth, predictions))
            metric = self.evaluation["per_pass"][pass_name]
            self.assertEqual(
                metric["confusion_matrix"],
                {
                    "true_negative": tn,
                    "false_positive": fp,
                    "false_negative": fn,
                    "true_positive": tp,
                },
            )
            self.assertAlmostEqual(metric["accuracy"], (tp + tn) / len(truth))
            self.assertAlmostEqual(metric["precision"], tp / (tp + fp) if tp + fp else 0.0)
            self.assertAlmostEqual(metric["recall"], tp / (tp + fn) if tp + fn else 0.0)
            precision = metric["precision"]
            recall = metric["recall"]
            self.assertAlmostEqual(
                metric["f1"],
                2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            )
        for metric in self.evaluation["per_pass"].values():
            matrix = metric["confusion_matrix"]
            self.assertEqual(
                sum(matrix.values()),
                100,
            )
        overall = self.evaluation["aggregate_confusion_matrix"]
        overall_tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(flattened_truth, flattened_predictions))
        overall_fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(flattened_truth, flattened_predictions))
        overall_fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(flattened_truth, flattened_predictions))
        overall_tn = sum(actual == 0 and predicted == 0 for actual, predicted in zip(flattened_truth, flattened_predictions))
        self.assertEqual(
            overall,
            {
                "true_negative": overall_tn,
                "false_positive": overall_fp,
                "false_negative": overall_fn,
                "true_positive": overall_tp,
            },
        )
        self.assertAlmostEqual(
            self.evaluation["overall_pass_recommendation_accuracy"],
            (overall_tp + overall_tn) / len(flattened_truth),
        )
        self.assertEqual(len(self.evaluation["per_category"]), 5)
        for metric in self.evaluation["per_category"].values():
            self.assertEqual(metric["program_count"], 20)
            self.assertEqual(sum(metric["aggregate"]["confusion_matrix"].values()), 120)


if __name__ == "__main__":
    unittest.main()
