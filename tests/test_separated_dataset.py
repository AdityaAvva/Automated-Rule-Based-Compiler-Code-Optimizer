import hashlib
import json
import unittest
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
TRAINING_FILE = ROOT / "dataset" / "training" / "training.json"
TESTING_FILE = ROOT / "dataset" / "testing" / "testing.json"
TRAINING_FIELDS = {
    "id", "category", "size", "source", "expected_output", "inputs",
}


class SeparatedDatasetPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = json.loads(TRAINING_FILE.read_text(encoding="utf-8"))
        cls.testing = json.loads(TESTING_FILE.read_text(encoding="utf-8"))
        cls.manifest = json.loads((ROOT / "dataset" / "manifest.json").read_text(encoding="utf-8"))
        cls.training_summary = json.loads(
            (ROOT / "results" / "separated_training_summary.json").read_text(encoding="utf-8")
        )
        cls.testing_summary = json.loads(
            (ROOT / "results" / "separated_testing_summary.json").read_text(encoding="utf-8")
        )
        cls.testing_results = json.loads(
            (ROOT / "results" / "separated_testing_results.json").read_text(encoding="utf-8")
        )
        cls.bundle = joblib.load(ROOT / "ml" / "separated_pass_recommender.joblib")

    def test_record_schemas_and_original_balanced_split_are_preserved(self):
        self.assertEqual(len(self.training), 400)
        self.assertEqual(len(self.testing), 100)
        self.assertTrue(all(set(record) == TRAINING_FIELDS for record in self.training))
        self.assertTrue(all(set(record) == {"source"} for record in self.testing))
        self.assertEqual(len({record["source"] for record in self.training}), 400)
        self.assertEqual(len({record["source"] for record in self.testing}), 100)
        self.assertFalse(
            {record["source"] for record in self.training}
            & {record["source"] for record in self.testing}
        )

        canonical = json.loads((ROOT / "benchmarks" / "dataset.json").read_text(encoding="utf-8"))
        benchmark_manifest = json.loads(
            (ROOT / "benchmarks" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(canonical), 500)
        self.assertTrue(all(set(record) == TRAINING_FIELDS for record in canonical))
        metadata = benchmark_manifest["program_metadata"]
        train_families = {
            record["family_id"] for record in metadata if record["split"] == "training"
        }
        test_families = {
            record["family_id"] for record in metadata if record["split"] == "testing"
        }
        self.assertFalse(train_families & test_families)
        self.assertEqual(self.manifest["category_counts"], {
            "constant_expressions": {"training": 80, "testing": 20, "total": 100},
            "algebraic_redundancy": {"training": 80, "testing": 20, "total": 100},
            "dead_code": {"training": 80, "testing": 20, "total": 100},
            "repeated_expressions": {"training": 80, "testing": 20, "total": 100},
            "loop_based": {"training": 80, "testing": 20, "total": 100},
        })

    def test_training_model_and_evaluation_match_their_separate_files(self):
        training_ids = {record["id"] for record in self.training}
        self.assertTrue(self.training_summary["training_split_only"])
        self.assertEqual(self.training_summary["training_samples"], 400)
        self.assertEqual(set(self.training_summary["training_ids"]), training_ids)
        self.assertEqual(set(self.bundle["training_ids"]), training_ids)
        self.assertEqual(
            self.training_summary["training_data_sha256"],
            hashlib.sha256(TRAINING_FILE.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            self.testing_summary["testing_data_sha256"],
            hashlib.sha256(TESTING_FILE.read_bytes()).hexdigest(),
        )
        self.assertEqual(self.testing_summary["testing_program_count"], 100)
        self.assertEqual(self.testing_summary["executed_program_count"], 100)
        self.assertEqual(self.testing_summary["successful_program_count"], 100)
        self.assertEqual(self.testing_summary["failed_program_count"], 0)
        self.assertEqual(len(self.testing_results), 100)
        for result in self.testing_results:
            self.assertNotIn("expected_output", result)
            self.assertNotIn("applicability_labels", result)
            self.assertIn("original_tac", result)
            self.assertIn("optimized_tac", result)
            self.assertTrue(result["output_equivalent"], result["program_index"])


if __name__ == "__main__":
    unittest.main()
