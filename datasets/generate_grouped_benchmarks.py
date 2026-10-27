"""Compatibility entry point for the canonical 500-program benchmark generator."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run(
        [sys.executable, str(ROOT / "generate_dataset.py")],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
