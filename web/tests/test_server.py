import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

WEB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_ROOT))
import server


class OptimizerDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http_server = ThreadingHTTPServer(("127.0.0.1", 0), server.DashboardHandler)
        cls.server_thread = threading.Thread(target=cls.http_server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.http_server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.http_server.shutdown()
        cls.http_server.server_close()
        cls.server_thread.join(timeout=2)

    def request_json(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method="POST" if data is not None else "GET",
        )
        try:
            response = urlopen(request, timeout=30)
        except HTTPError as exc:
            response = exc
        return response.status, json.loads(response.read().decode("utf-8"))

    def test_local_dashboard_assets_and_health(self):
        with urlopen(self.base_url + "/", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"Optimizer", response.read())
        status, health = self.request_json("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["local_only"])
        self.assertIn("optimizer_available", health)

    def test_real_dataset_and_benchmark_reports_are_exposed(self):
        status, dataset = self.request_json("/api/dataset/summary")
        self.assertEqual(status, 200)
        self.assertEqual(dataset["summary"]["program_count"], 15)
        self.assertEqual(
            dataset["summary"]["verified_origins"]["textbook_or_lecture_origins_from_supplied_candidates"],
            0,
        )
        status, benchmarks = self.request_json("/api/benchmarks/summary")
        self.assertEqual(status, 200)
        self.assertEqual(benchmarks["summary"]["total_benchmarks"], 500)
        self.assertEqual(benchmarks["summary"]["failed_programs"], 0)

    def test_grouped_dataset_and_held_out_compiler_results_are_exposed(self):
        status, data = self.request_json("/api/benchmark-ml?page=1&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(data["summary"]["program_count"], 500)
        self.assertEqual(data["summary"]["training_count"], 400)
        self.assertEqual(data["summary"]["testing_count"], 100)
        self.assertEqual(
            len(data["summary"]["training_record_fields"]),
            6,
        )
        self.assertEqual(data["summary"]["testing_record_fields"], ["source"])
        self.assertEqual(data["total_filtered"], 500)
        self.assertEqual(len(data["programs"]), 10)
        self.assertNotIn("family_id", data["programs"][0])
        self.assertNotIn("label_evidence", data["programs"][0])

        status, testing = self.request_json(
            "/api/benchmark-ml?category=loop_based&split=testing&size=small&page=1&limit=25"
        )
        self.assertEqual(status, 200)
        self.assertGreater(testing["total_filtered"], 0)
        for record in testing["programs"]:
            self.assertEqual(record["category"], "loop_based")
            self.assertEqual(record["split"], "testing")
            self.assertEqual(record["size"], "small")
            self.assertIsNotNone(record["ml_recommendations"])
            self.assertEqual(
                set(record["compiler_result"]),
                {"ml_recommended", "all_passes_baseline"},
            )
            self.assertTrue(record["compiler_result"]["ml_recommended"]["output_equivalent"])
            self.assertTrue(record["compiler_result"]["all_passes_baseline"]["output_equivalent"])

        with urlopen(
            self.base_url + "/plots/benchmark_v2/01_category_train_test.png",
            timeout=5,
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get_content_type(), "image/png")
            self.assertTrue(response.read(8).startswith(b"\x89PNG"))

    def test_separated_training_and_source_only_testing_are_wired(self):
        status, training = self.request_json("/api/separated-datasets/training?limit=500")
        self.assertEqual(status, 200)
        self.assertEqual(training["program_count"], 400)
        self.assertEqual(training["file"], "dataset\\training\\training.json")
        self.assertTrue(training["status"]["model_trained"])
        self.assertEqual(training["status"]["training_samples"], 400)
        self.assertNotIn("training_family_count", training["status"])
        self.assertNotIn("family_overlap", training["status"])
        for record in training["programs"]:
            self.assertEqual(
                set(record),
                {"id", "category", "size", "source", "expected_output", "inputs"},
            )

        status, testing = self.request_json("/api/separated-datasets/testing?limit=100")
        self.assertEqual(status, 200)
        self.assertEqual(testing["program_count"], 100)
        self.assertEqual(testing["file"], "dataset\\testing\\testing.json")
        self.assertTrue(testing["status"]["testing_evaluated"])
        self.assertEqual(testing["status"]["testing_executed_count"], 100)
        for record in testing["programs"]:
            self.assertEqual(set(record), {"source"})

        status, result = self.request_json(
            "/api/separated-datasets/testing/run",
            {"index": 1},
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["model"], "separated_training")
        self.assertTrue(result["validation"]["output_equivalence_confirmed_by_compiler"])
        self.assertEqual(result["original_output"], result["optimized_output"])
        self.assertIsNotNone(result["original_tac"])
        self.assertIsNotNone(result["optimized_tac"])

    def test_real_optimizer_all_passes_preserves_output(self):
        status, result = self.request_json("/api/optimize", {
            "source": "a = 4; b = a * 2; c = b + 0; unused = c * 1; print(c); print(a);",
            "mode": "all",
            "model": "real_world",
        })
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["original_output"], result["optimized_output"])
        self.assertEqual(len(result["selected_passes"]), 6)
        self.assertTrue(result["validation"]["output_equivalence_confirmed_by_compiler"])
        self.assertGreater(result["optimization_cycles"], 1)
        self.assertTrue(result["convergence_trace"][-1]["stable"])
        self.assertFalse(result["convergence_trace"][-1]["changed"])
        self.assertNotIn("convergence_", result["optimized_tac"])
        self.assertTrue(
            all(
                result["metrics"][f"{name}_invocations"] == result["optimization_cycles"]
                for name in server.PASSES
            )
        )

    def test_ml_recommended_mode_runs_and_reports_model_selection(self):
        status, result = self.request_json("/api/optimize", {
            "source": "x = 10 * 20;\nprint(x);",
            "mode": "recommended",
            "model": "real_world",
        })
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["selected_passes"], result["recommended_passes"])
        self.assertTrue(result["validation"]["output_equivalence_confirmed_by_compiler"])

    def test_invalid_source_and_invalid_configuration_are_rejected(self):
        status, malformed = self.request_json("/api/optimize", {
            "source": "this is not supported;",
            "mode": "all",
        })
        self.assertEqual(status, 422)
        self.assertEqual(malformed["status"], "ERROR")
        status, invalid = self.request_json("/api/optimize", {
            "source": "x = 1;",
            "mode": "shell",
            "model": "real_world",
        })
        self.assertEqual(status, 400)
        self.assertIn("Mode must be", invalid["error"])


if __name__ == "__main__":
    unittest.main()
