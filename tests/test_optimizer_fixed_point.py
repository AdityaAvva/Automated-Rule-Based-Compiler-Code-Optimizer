import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASSES = (
    "constant_folding",
    "constant_propagation",
    "algebraic_simplification",
    "common_subexpression_elimination",
    "dead_code_elimination",
    "strength_reduction",
)


class FixedPointOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="optimizer-fixed-point-")
        cls.executable = Path(cls.temp_dir.name) / "compiler_optimizer.exe"
        completed = subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O2",
                "-Wall",
                str(ROOT / "src" / "main.cpp"),
                "-o",
                str(cls.executable),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Could not build optimizer:\n{completed.stdout}\n{completed.stderr}"
            )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def optimize(self, source):
        return subprocess.run(
            [
                str(self.executable),
                "--trace-convergence",
                "--passes",
                ",".join(PASSES),
            ],
            input=source,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def assert_fixed_point_trace(self, output):
        self.assertNotRegex(
            output,
            re.compile(
                r"^(?:convergence_iteration_change|convergence_pass)=|"
                r".*(?:,changed=|,rewrites=)",
                re.M,
            ),
        )
        states = [
            (int(iteration), int(stable))
            for iteration, stable in re.findall(
                r"^convergence_iteration=(\d+),instructions=\d+,stable=(\d+)$",
                output,
                re.M,
            )
        ]
        self.assertGreaterEqual(len(states), 2, output)
        self.assertEqual(states[0], (0, 0))
        self.assertTrue(all(stable == 0 for _, stable in states[1:-1]), states)
        self.assertEqual(states[-1][1], 1, states)
        metrics = {
            key: int(value)
            for key, value in re.findall(r"^([a-z_]+)=(-?\d+)$", output, re.M)
        }
        self.assertEqual(metrics["optimization_cycles"], states[-1][0])
        for name in PASSES:
            self.assertEqual(metrics[f"{name}_invocations"], states[-1][0])
        return states

    def test_dead_assignments_are_removed_but_used_assignments_and_prints_remain(self):
        source = "x = 200; print(x); y = 200; print(y);"
        result = self.optimize(source)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assert_fixed_point_trace(result.stdout)
        optimized = re.search(
            r"=== OPTIMIZED TAC ===\n(.*?)=== OUTPUT CHECK ===",
            result.stdout,
            re.S,
        )
        self.assertIsNotNone(optimized)
        self.assertEqual(optimized.group(1), "print 200\nprint 200\n")
        self.assertRegex(result.stdout, r"Original output:\n200\n200\nOptimized output:\n200\n200\n")

        dce_only = subprocess.run(
            [
                str(self.executable),
                "--passes",
                "dead_code_elimination",
            ],
            input="x = 2; y = x + 3; print(y);",
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(dce_only.returncode, 0, dce_only.stderr + dce_only.stdout)
        optimized_dce_only = re.search(
            r"=== OPTIMIZED TAC ===\n(.*?)=== OUTPUT CHECK ===",
            dce_only.stdout,
            re.S,
        )
        self.assertIsNotNone(optimized_dce_only)
        self.assertIn("x = 2\n", optimized_dce_only.group(1))
        self.assertIn("y = t1\n", optimized_dce_only.group(1))
        self.assertIn("print y\n", optimized_dce_only.group(1))

    def test_normal_execution_omits_iteration_debug_rows(self):
        result = subprocess.run(
            [
                str(self.executable),
                "--passes",
                ",".join(PASSES),
            ],
            input="x = 200; print(x);",
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertNotIn("convergence_iteration", result.stdout)
        self.assertNotIn("convergence_pass", result.stdout)
        self.assertNotIn(",changed=", result.stdout)
        self.assertNotIn(",rewrites=", result.stdout)
        self.assertRegex(result.stdout, r"(?m)^optimization_cycles=\d+$")

    def test_all_six_passes_repeat_until_fixed_point_and_preserve_output(self):
        source = (
            "a = 4; b = a * 2; c = b + 0; "
            "unused = c * 1; print(c); print(a);"
        )
        result = self.optimize(source)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        states = self.assert_fixed_point_trace(result.stdout)
        self.assertGreater(states[-1][0], 1, states)
        self.assertIn("Verdict: PASS", result.stdout)
        self.assertRegex(result.stdout, r"Original output:\n8\n4\nOptimized output:\n8\n4\n")

    def test_all_benchmarks_reach_fixed_point_and_preserve_expected_output(self):
        records = json.loads((ROOT / "benchmarks" / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(len(records), 500)
        for record in records:
            with self.subTest(program=record["id"]):
                result = self.optimize(record["source"])
                self.assertEqual(
                    result.returncode,
                    0,
                    f"{record['id']}: {result.stderr}\n{result.stdout}",
                )
                self.assert_fixed_point_trace(result.stdout)
                original = re.search(
                    r"Original output:\n(.*?)Optimized output:",
                    result.stdout,
                    re.S,
                )
                optimized = re.search(
                    r"Optimized output:\n(.*?)Verdict:",
                    result.stdout,
                    re.S,
                )
                self.assertIsNotNone(original, record["id"])
                self.assertIsNotNone(optimized, record["id"])
                self.assertEqual(original.group(1).strip(), record["expected_output"].strip())
                self.assertEqual(original.group(1), optimized.group(1), record["id"])


if __name__ == "__main__":
    unittest.main()
