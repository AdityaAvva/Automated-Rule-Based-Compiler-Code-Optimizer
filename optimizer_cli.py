import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_single(source=None, passes=None):
    command = [sys.executable, str(ROOT / "single_test.py")]
    if source:
        command.extend(["--source", source])
    if passes:
        command.extend(["--passes", passes])
    return subprocess.run(command, cwd=ROOT).returncode


def run_full_benchmarks():
    print("FULL BENCHMARK MODE: running the compile, 500-program dataset, split-only "
          "training, held-out evaluation, compiler benchmark, and plots pipeline.")
    return subprocess.run([sys.executable, str(ROOT / "run_project.py")], cwd=ROOT).returncode


def main():
    parser = argparse.ArgumentParser(
        description="Choose one ML-assisted test case or the full 500-program benchmark pipeline."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source", help="Run one source file as a single test.")
    mode.add_argument(
        "--full-benchmarks", action="store_true",
        help="Run the full 500-program dataset/model/benchmark/plot pipeline.",
    )
    parser.add_argument(
        "--passes", choices=("recommended", "all"),
        help="Single-test pass selection (ignored by the interactive menu).",
    )
    args = parser.parse_args()

    if args.source:
        return run_single(args.source, args.passes)
    if args.full_benchmarks:
        if args.passes:
            parser.error("--passes can only be used with --source")
        return run_full_benchmarks()

    print("ML-Assisted Compiler Optimizer")
    print("1. Single test case (manual entry or source file)")
    print("2. Full 500-program benchmark suite")
    choice = input("Select mode [1/2]: ").strip()
    if choice == "1":
        return run_single()
    if choice == "2":
        return run_full_benchmarks()
    print("Invalid choice. Enter 1 for a single test or 2 for full benchmarks.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
