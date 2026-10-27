"""Train and evaluate the separated ML-facing datasets without generating benchmarks."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = (
    [sys.executable, str(ROOT / "dataset" / "project_existing_benchmarks.py")],
    [sys.executable, str(ROOT / "ml" / "train_separated_model.py")],
    [sys.executable, str(ROOT / "ml" / "evaluate_separated_model.py")],
)


def main():
    for command in STEPS:
        print(f"\n>>> {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("\nSeparated training/testing pipeline completed.")


if __name__ == "__main__":
    main()
