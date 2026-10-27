"""Evaluate the trained model using only source-only testing.json records."""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from benchmark_features import extract_features
from ml_optimize import recommend_passes
from run_benchmark_ml import PASSES, _parse_metrics, _run_compiler

TESTING_FILE = ROOT / "dataset" / "testing" / "testing.json"
MODEL_FILE = ROOT / "ml" / "separated_pass_recommender.joblib"
RESULTS_FILE = ROOT / "results" / "separated_testing_results.json"
SUMMARY_FILE = ROOT / "results" / "separated_testing_summary.json"


def section(output, start, end):
    match = re.search(re.escape(start) + r"\n(.*?)" + re.escape(end), output, re.S)
    return match.group(1) if match else None


def evaluate():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Trained model missing at {MODEL_FILE}; train using dataset/training/training.json first."
        )
    testing_bytes = TESTING_FILE.read_bytes()
    testing = json.loads(testing_bytes.decode("utf-8"))
    testing_data_sha256 = hashlib.sha256(testing_bytes).hexdigest()
    if len(testing) != 100 or any(set(record) != {"source"} for record in testing):
        raise ValueError("Testing dataset must contain 100 source-only records.")

    results = []
    for index, record in enumerate(testing, start=1):
        source = record["source"]
        baseline = _run_compiler(source, [], trace=False)
        if baseline.returncode != 0:
            raise RuntimeError(f"Test source {index}: original execution failed: {baseline.stderr}")
        original_tac = section(
            baseline.stdout, "=== ORIGINAL TAC ===", "=== OPTIMIZED TAC ==="
        )
        original_output = section(baseline.stdout, "Original output:", "Optimized output:")
        if original_tac is None or original_output is None:
            raise RuntimeError(f"Test source {index}: compiler omitted TAC or original output.")

        selected, recommendations = recommend_passes(
            source,
            model_path=MODEL_FILE,
            original_tac=original_tac,
        )
        optimized = _run_compiler(source, selected, trace=True)
        if optimized.returncode != 0:
            raise RuntimeError(
                f"Test source {index}: recommended optimizer run failed: {optimized.stderr}"
            )
        optimized_original = section(
            optimized.stdout, "Original output:", "Optimized output:"
        )
        optimized_output = section(optimized.stdout, "Optimized output:", "Verdict:")
        original_tac_from_run = section(
            optimized.stdout, "=== ORIGINAL TAC ===", "=== OPTIMIZED TAC ==="
        )
        optimized_tac = section(
            optimized.stdout, "=== OPTIMIZED TAC ===", "=== OUTPUT CHECK ==="
        )
        verdict = re.search(r"^Verdict: (PASS|FAIL)$", optimized.stdout, re.M)
        if any(value is None for value in (
            optimized_original,
            optimized_output,
            original_tac_from_run,
            optimized_tac,
        )) or verdict is None:
            raise RuntimeError(f"Test source {index}: optimizer output was incomplete.")
        equivalent = (
            verdict.group(1) == "PASS"
            and original_output == optimized_original
            and optimized_original == optimized_output
        )
        metrics = _parse_metrics(optimized.stdout)
        result = {
            "program_index": index,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "status": "PASS" if equivalent else "FAIL",
            "output_equivalent": equivalent,
            "selected_passes": selected,
            "recommendations": recommendations,
            "original_tac": original_tac_from_run.rstrip(),
            "optimized_tac": optimized_tac.rstrip(),
            "original_output": optimized_original.strip(),
            "optimized_output": optimized_output.strip(),
            "metrics": metrics,
        }
        results.append(result)
        if index % 10 == 0 or index == len(testing):
            print(f"Executed testing sources: {index}/{len(testing)}")

    instruction_reductions = [
        100 * (result["metrics"]["instructions_before"] - result["metrics"]["instructions_after"])
        / result["metrics"]["instructions_before"]
        for result in results
        if result["metrics"].get("instructions_before")
        and result["metrics"].get("instructions_after") is not None
    ]
    summary = {
        "status": "evaluated",
        "testing_file": str(TESTING_FILE.relative_to(ROOT)),
        "model": str(MODEL_FILE.relative_to(ROOT)),
        "model_sha256": hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest(),
        "testing_data_sha256": testing_data_sha256,
        "testing_program_count": len(testing),
        "executed_program_count": len(results),
        "successful_program_count": sum(result["output_equivalent"] for result in results),
        "failed_program_count": sum(not result["output_equivalent"] for result in results),
        "mean_instruction_reduction_pct": (
            float(np.mean(instruction_reductions)) if instruction_reductions else None
        ),
        "passes": PASSES,
        "limitations": [
            "Testing records contain source only; no test expected outputs or applicability labels are consumed.",
            "Output equivalence is checked against the original execution from the same C++ compiler run.",
            "These generated, family-held-out examples do not establish generalization to arbitrary source programs.",
        ],
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    evaluate()
