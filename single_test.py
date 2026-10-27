import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ml"))
try:
    from ml_optimize import recommend_passes
    from train_model import PASSES
except ImportError as exc:
    raise SystemExit(
        f"Missing Python ML dependency: {exc}. Install the packages in requirements.txt."
    ) from exc


def read_source(source_path):
    if source_path is None:
        source_path = input("Source-file path (leave blank to enter code manually): ").strip()
    if source_path:
        path = Path(source_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        try:
            return path.read_text(encoding="utf-8"), str(path)
        except OSError as exc:
            raise SystemExit(f"Cannot read source file '{path}': {exc}") from exc

    print("Enter a program in the supported language. Type END on a line by itself to finish.")
    lines = []
    try:
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
    except EOFError:
        pass
    source = "\n".join(lines)
    if not source.strip():
        raise SystemExit("No source code was entered.")
    return source, "manual terminal input"


def unique_input_path(directory):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = directory / f"single_test_{stamp}.src"
    suffix = 1
    while candidate.exists():
        candidate = directory / f"single_test_{stamp}_{suffix}.src"
        suffix += 1
    return candidate


def metric_values(output):
    metrics = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if re.fullmatch(r"-?\d+", value.strip()):
            metrics[key.strip()] = int(value.strip())
    return metrics


def section(output, start, end):
    match = re.search(re.escape(start) + r"\n(.*?)" + re.escape(end), output, re.S)
    return match.group(1).rstrip() if match else None


def percentage_reduction(before, after):
    if before is None or after is None or before == 0:
        return "unavailable (baseline is zero or metric is missing)"
    return f"{100 * (before - after) / before:.2f}%"


def save_report(report_path, report):
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run one source program through the ML-assisted C++ optimizer."
    )
    parser.add_argument("--source", help="Source file; omit to enter code in the terminal.")
    parser.add_argument(
        "--passes",
        choices=("recommended", "all"),
        help="Use ML-selected passes or explicitly enable all six rule passes.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--model", help="Optional model bundle; defaults to the legacy benchmark model.")
    args = parser.parse_args()

    source, source_description = read_source(args.source)
    if args.passes is None:
        print("\nChoose the rule-pass behavior for this single test:")
        print("1. Run only ML-recommended passes")
        print("2. Run all six optimization passes")
        choice = input("Choice [1/2]: ").strip()
        if choice not in ("1", "2"):
            raise SystemExit("Choose 1 (ML-recommended) or 2 (all six passes).")
        pass_mode = "recommended" if choice == "1" else "all"
    else:
        pass_mode = args.passes

    result_dir = ROOT / "results" / "single_test"
    result_dir.mkdir(parents=True, exist_ok=True)
    input_path = unique_input_path(result_dir)
    input_path.write_text(source, encoding="utf-8")
    report_path = input_path.with_suffix(".json")
    try:
        recommended_passes, recommendations = recommend_passes(source, args.threshold, args.model)
    except FileNotFoundError as exc:
        save_report(report_path, {
            "source_description": source_description,
            "input_file": str(input_path),
            "original_source": source,
            "verdict": "ERROR",
            "error": str(exc),
        })
        raise SystemExit(str(exc)) from exc
    except (OSError, ValueError, EOFError, ImportError, KeyError) as exc:
        error = f"Unable to load the trained ML model or make recommendations: {exc}"
        save_report(report_path, {
            "source_description": source_description,
            "input_file": str(input_path),
            "original_source": source,
            "verdict": "ERROR",
            "error": error,
        })
        raise SystemExit(error) from exc
    enabled_passes = recommended_passes if pass_mode == "recommended" else list(PASSES)
    exe = ROOT / ("compiler_optimizer.exe" if sys.platform == "win32" else "compiler_optimizer")
    if not exe.is_file():
        error = f"C++ optimizer not found: {exe}. Compile src/main.cpp first."
        save_report(report_path, {
            "source_description": source_description,
            "input_file": str(input_path),
            "original_source": source,
            "ml_recommendations": recommendations,
            "passes_enabled": enabled_passes,
            "verdict": "ERROR",
            "error": error,
        })
        raise SystemExit(error)
    command = [
        str(exe), "--trace-convergence", "--passes", ",".join(enabled_passes), str(input_path)
    ]
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except OSError as exc:
        raise SystemExit(f"Could not start the C++ optimizer: {exc}") from exc

    output = completed.stdout
    metrics = metric_values(output)
    original_output = section(output, "Original output:", "Optimized output:")
    optimized_output = section(output, "Optimized output:", "Verdict:")
    verdict_match = re.search(r"^Verdict: (PASS|FAIL)$", output, re.M)
    convergence = re.findall(
        r"^convergence_iteration=(\d+),instructions=(\d+),stable=(\d+)$", output, re.M
    )
    cycles = max((int(row[0]) for row in convergence), default=None)
    optimized_tac = section(output, "=== OPTIMIZED TAC ===", "=== OUTPUT CHECK ===")
    optimized_tac = "\n".join(
        line for line in (optimized_tac or "").splitlines()
        if not line.startswith("convergence_")
    ).rstrip()
    report = {
        "source_description": source_description,
        "input_file": str(input_path),
        "pass_mode": pass_mode,
        "ml_recommendations": recommendations,
        "passes_enabled": enabled_passes,
        "original_source": source,
        "original_tac": section(output, "=== ORIGINAL TAC ===", "=== OPTIMIZED TAC ==="),
        "optimized_tac": optimized_tac,
        "original_output": original_output,
        "optimized_output": optimized_output,
        "verdict": verdict_match.group(1) if verdict_match else "ERROR",
        "metrics": metrics,
        "optimization_cycles": cycles,
        "convergence_trace": [
            {
                "iteration": int(i),
                "instructions": int(count),
                "stable": bool(int(stable)),
                "changed": int(i) > 0 and not bool(int(stable)),
            }
            for i, count, stable in convergence
        ],
        "compiler_return_code": completed.returncode,
        "compiler_stderr": completed.stderr,
        "compiler_output": "\n".join(
            line for line in output.splitlines()
            if not line.startswith("convergence_iteration=")
        ),
    }
    save_report(report_path, report)

    print("\n=== SINGLE-TEST REPORT ===")
    print(f"Source: {source_description}")
    print(f"Saved input: {input_path}")
    print(f"Saved report: {report_path}")
    print("ML-recommended passes:")
    for item in recommendations:
        print(
            f"  {item['name']}: {'recommended' if item['recommended'] else 'not recommended'} "
            f"(score={item['score']:.3f}; threshold-selected="
            f"{'yes' if item['selected'] else 'no'})"
        )
    print(f"Pass mode: {pass_mode}")
    print(f"Passes enabled in the optimizer: {', '.join(enabled_passes) or '(none)'}")
    if completed.returncode != 0 and not output:
        print(f"Optimizer error: {completed.stderr.strip() or 'no diagnostic output'}")
    else:
        normal_output = "\n".join(
            line for line in output.splitlines()
            if not line.startswith("convergence_iteration=")
        )
        print("\n" + normal_output.rstrip())
        print("\n=== SINGLE-TEST METRIC SUMMARY ===")
        for label, before, after in (
            ("Instructions", "instructions_before", "instructions_after"),
            ("Arithmetic operations", "operations_before", "operations_after"),
            ("Temporary variables", "temporaries_before", "temporaries_after"),
            ("TAC code size (characters)", "code_size_before", "code_size_after"),
        ):
            before_value, after_value = metrics.get(before), metrics.get(after)
            print(f"{label}: {before_value if before_value is not None else 'unavailable'} -> "
                  f"{after_value if after_value is not None else 'unavailable'} "
                  f"(reduction: {percentage_reduction(before_value, after_value)})")
        print(
            "Optimization time: "
            f"{metrics.get('optimization_time_microseconds', 'unavailable')} microseconds"
        )
        print(
            "Interpreter time per run (100-run average): "
            f"{metrics.get('interpreter_time_before_microseconds', 'unavailable')} -> "
            f"{metrics.get('interpreter_time_after_microseconds', 'unavailable')} microseconds"
        )
        print(f"Complete optimization cycles: {cycles if cycles is not None else 'unavailable'}")
        print("Actual transformation events by pass:")
        for name in PASSES:
            print(f"  {name}: {metrics.get(name, 'unavailable')}")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
