import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "main.cpp"


class OptimizerCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = "g++"
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="optimizer-cleanup-")
        executable_name = "compiler_optimizer.exe" if os.name == "nt" else "compiler_optimizer"
        cls.executable = Path(cls.temp_dir.name) / executable_name
        subprocess.run(
            [compiler, "-std=c++17", "-O2", "-Wall", str(SOURCE), "-o", str(cls.executable)],
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def run_optimizer(self, source, passes=None):
        command = [str(self.executable)]
        if passes is not None:
            command.extend(["--passes", passes])
        result = subprocess.run(
            command,
            input=source,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        tac_match = re.search(
            r"=== OPTIMIZED TAC ===\n(.*?)=== OUTPUT CHECK ===",
            result.stdout,
            re.S,
        )
        self.assertIsNotNone(tac_match, result.stdout)
        output_match = re.search(
            r"Original output:\n(.*?)Optimized output:\n(.*?)Verdict: (PASS|FAIL)",
            result.stdout,
            re.S,
        )
        self.assertIsNotNone(output_match, result.stdout)
        metrics = {
            key: int(value)
            for key, value in re.findall(r"^([a-z_]+)=(\d+)$", result.stdout, re.M)
        }
        return tac_match.group(1), output_match.group(1), output_match.group(2), metrics

    def test_constant_chain_eliminates_dead_inputs_and_temporaries(self):
        source = """\
a = 10;
b = 20;
x = 5;
y = 3;
sum = a + b;
product = sum * x;
result = product - y;
print(sum);
print(product);
print(result);
"""
        tac, original_output, optimized_output, metrics = self.run_optimizer(source)
        self.assertEqual(
            tac,
            "sum = 30\n"
            "product = 150\n"
            "result = 147\n"
            "print 30\n"
            "print 150\n"
            "print 147\n",
        )
        self.assertEqual(original_output, "30\n150\n147\n")
        self.assertEqual(optimized_output, original_output)
        self.assertEqual(metrics["instructions_after"], 6)
        self.assertEqual(metrics["operations_after"], 0)
        self.assertEqual(metrics["temporaries_after"], 0)
        self.assertEqual(metrics["code_size_after"], len(tac))

    def test_copy_propagation_respects_reassignment_and_print_order(self):
        source = """\
x = 4;
saved = x;
print(saved);
x = 9;
print(saved);
"""
        tac, original_output, optimized_output, _ = self.run_optimizer(source)
        self.assertEqual(tac, "saved = 4\nprint 4\nprint 4\n")
        self.assertEqual(original_output, "4\n4\n")
        self.assertEqual(optimized_output, original_output)

    def test_dead_code_elimination_keeps_later_uses_and_reused_definitions(self):
        source = """\
x = 2;
y = x + 1;
x = 8;
print(y);
print(x);
"""
        tac, original_output, optimized_output, metrics = self.run_optimizer(
            source, "dead_code_elimination"
        )
        self.assertEqual(
            tac,
            "x = 2\n"
            "t1 = x + 1\n"
            "y = t1\n"
            "x = 8\n"
            "print y\n"
            "print x\n",
        )
        self.assertEqual(original_output, "3\n8\n")
        self.assertEqual(optimized_output, original_output)
        self.assertEqual(metrics["instructions_after"], len(tac.splitlines()))
        self.assertEqual(metrics["operations_after"], 1)
        self.assertEqual(metrics["temporaries_after"], 1)
        self.assertEqual(metrics["code_size_after"], len(tac))


if __name__ == "__main__":
    unittest.main()
