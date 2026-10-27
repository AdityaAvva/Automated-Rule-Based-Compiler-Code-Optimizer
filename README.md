# ML-Assisted Automated Rule-Based Compiler Code Optimizer

A runnable Compiler Design Lab project with a C++ TAC optimizer, a balanced 500-program benchmark suite, a Python/scikit-learn pass recommender, automated output checks, metrics, and plots.

## Project layout

- `src/main.cpp` — C++ compiler front-end for a small expression language, TAC generator, six optimization passes, fixed-point loop, TAC interpreter, output comparison, and metrics.
- `generate_dataset.py` — generates 500 benchmark JSON records: 100 per category, split into 80 training and 20 testing programs per category.
- `benchmarks/` — 500 individual `CAT1_*.json` through `CAT5_*.json` records, canonical `dataset.json`, and `manifest.json`.
- `dataset/training/training.json` and `datasets/training/programs.json` — the same 400 training programs, each with exactly six fields.
- `dataset/testing/testing.json` and `datasets/testing/programs.json` — the same 100 testing programs, each containing only `source`.
- `benchmarks/manifest.json` — category/family/split metadata kept outside program records.
- `ml/train_model.py` — trains a Random Forest classifier per optimization pass and writes pass recommendations.
- `run_benchmarks.py` — runs every benchmark with all six implemented passes enabled, records ML recommendations separately, and writes CSV/JSON results.
- `make_plots.py` — produces charts from benchmark results.
- `results/` — generated reports and logs.
- `plots/` — generated plots.

## Requirements

- C++17 compiler (`g++` recommended)
- Python 3.10+
- Python packages: `numpy`, `scikit-learn`, `joblib`, `matplotlib`

Install Python packages from this folder:

```bash
python -m pip install -r requirements.txt
```

## Windows + VS Code setup

1. Install VS Code and a C++ compiler (MinGW-w64/GCC); ensure `g++ --version` works in the VS Code terminal.
2. Open the `ML_Assisted_Compiler_Optimizer` folder in VS Code.
3. In the integrated terminal, compile:

```powershell
g++ -std=c++17 -O2 -Wall src/main.cpp -o compiler_optimizer.exe
```

4. Generate the dataset:

```powershell
python generate_dataset.py
```

5. Train and evaluate using simplified training/testing files:

```powershell
python run_separated_ml_pipeline.py
python run_benchmark_ml.py --skip-generation
```

6. Train the separate legacy recommender (training split only):

```powershell
python ml/train_model.py
```

7. Run one program. The C++ executable accepts a source-text file, not a JSON record. The included `sample.src` is ready to run:

```powershell
.\compiler_optimizer.exe sample.src
```

8. Run all 500 benchmarks:

```powershell
python run_benchmarks.py
```

9. Generate plots:

```powershell
python make_plots.py
```

## Linux/macOS build

```bash
g++ -std=c++17 -O2 -Wall src/main.cpp -o compiler_optimizer
python -m pip install -r requirements.txt
python generate_dataset.py
python run_benchmark_ml.py --skip-generation
python ml/train_model.py
python run_benchmarks.py
python make_plots.py
```

## Supported source language

- Assignment: `x = 10 * 20;`
- Arithmetic: `+ - * / %`, parentheses, unary minus
- Print: `print x;` or `print(x + 2);`
- Simple counted loop: `for(i=0;i<10;i=i+1){ x = x + i * 2; }`

The counted loop is expanded into straight-line statements before TAC generation. It is an educational subset, not a full C/C++ compiler. Variables used in expressions must be assigned before use. The implementation uses floating-point arithmetic for the interpreter; avoid overflow-sensitive or floating-point edge cases in benchmarks.

## Six optimizations

1. Constant folding
2. Constant propagation
3. Algebraic simplification
4. Common subexpression elimination (conservative)
5. Dead-code elimination
6. Strength reduction (limited safe cases)

The optimizer runs repeated passes until TAC stabilizes or the maximum pass count is reached. Constant propagation also propagates copy assignments, and backward dead-code elimination removes unused definitions and compiler temporaries without dropping source variables observed by a print. Print statements and their order are preserved. Reported instruction, operation, temporary, and code-size metrics are computed from the final optimized TAC. Every test compares original and optimized interpreter output.

Run the focused optimizer-cleanup regressions (they compile `src/main.cpp` into a temporary directory):

```powershell
python -m unittest discover -s tests -v
```

## ML evaluation: scope and interpretation

The grouped Random Forest pipeline reads the 400 records in `datasets/training/programs.json`, each containing exactly `id`, `category`, `size`, `source`, `expected_output`, and `inputs`. It derives features from source plus original, pre-optimization TAC and derives each pass target by running the compiler analysis; neither is written into training records. The 100 records in `datasets/testing/programs.json` contain only `source`. IDs, categories, sizes, and family split assignments are resolved through the separate benchmark manifest. Evaluation uses compiler-derived test targets only as held-out ground truth and rejects overlapping IDs or generation families. Hyperparameters and 0.5 decision thresholds are fixed in source rather than selected on the test set.

Pass-applicability labels are generated in memory by applying each pass individually to original TAC and checking that it performs at least one rewrite while preserving program output. This is a compiler-derived applicability target, **not** a speedup or code-quality target. Test labels are used only as evaluation ground truth; they are not model inputs, training targets, preprocessing data, or model-selection signals. Program JSON files contain no label evidence, features, family IDs, split values, seeds, or other compiler-analysis metadata.

The stored held-out metrics measure pass recommendations, not optimizer effectiveness. Actual compilation is measured separately: each test program is run once with ML-recommended passes and once with a separate all-six-pass baseline. See `results/benchmark_v2_ml_evaluation.json`, `results/benchmark_v2_compiler_results.json`, and `results/benchmark_v2_compiler_summary.json`. These results describe this generated, template-family-held-out educational suite only; the small number of held-out families, restricted syntax, compile-time-heavy programs, and labels derived from the same compiler rule implementations limit claims about broader generalization.

The audited evaluation contains 100 held-out programs × 6 pass decisions (600 binary decisions): accuracy **99.8333%** (599/600), micro precision **99.6139%**, micro recall **100%**, micro F1 **99.8066%**, macro precision **99.4624%**, macro recall **100%**, macro F1 **99.7268%**, subset accuracy **99%** (99/100 programs), and Hamming loss **1/600**. The sole false positive is a strength-reduction recommendation:

| Pass | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Constant folding | 100% | 100% | 100% | 52 | 0 | 0 | 48 |
| Constant propagation | 100% | 100% | 100% | 0 | 0 | 0 | 100 |
| Algebraic simplification | 100% | 100% | 100% | 80 | 0 | 0 | 20 |
| Common subexpression elimination | 100% | 100% | 100% | 60 | 0 | 0 | 40 |
| Dead code elimination | 100% | 100% | 100% | 80 | 0 | 0 | 20 |
| Strength reduction | 96.7742% | 100% | 98.3607% | 69 | 1 | 0 | 30 |

The compiler effectiveness report is separate: on the same 100 test programs, ML-recommended runs and all-six-pass runs each preserved expected outputs for 100/100 programs. Mean instruction reduction was 18.26% with recommendations versus 69.29% for all six passes; this is an outcome comparison, not a claim that recommendations improve optimization. The independent all-six-pass run on the full 500-program suite preserved outputs for 500/500 programs.

The held-out set contains 10 families (2 per category), compared with 25 training families. Constant propagation is positive for all 400 training records, so its stored predictor is the fixed always-recommend prevalence baseline rather than a fitted Random Forest; all 100 test labels for that pass are also positive. Its perfect pass metrics therefore do not demonstrate learned generalization.

## Separate training and testing files

The ML-facing files use strict schemas:

- `dataset/training/training.json` contains 400 records with **exactly** `id`, `category`, `size`, `source`, `expected_output`, and `inputs`.
- `dataset/testing/testing.json` contains 100 records with **only** `source`.
- `dataset/manifest.json` documents counts and that these files are projections of the existing benchmark split; it does not add metadata to individual testing records.

The generator writes the canonical benchmark, both separated file pairs, and per-category benchmark records using these schemas. Split and family assignments remain in `benchmarks/manifest.json`, not in program records. Old common-subexpression records are preserved under `datasets/archive/legacy_common_subexpressions/` in the same compact format; that archive is not read by either active training pipeline.

To rebuild the projections from the existing 500 programs and run the dedicated training/testing pipeline without invoking the benchmark generator:

```powershell
python run_separated_ml_pipeline.py
```

The trainer reads only `dataset/training/training.json`, derives compiler-confirmed applicability targets from those 400 sources, and saves `ml/separated_pass_recommender.joblib`. The evaluator reads only the source-only testing file and this model; it executes every test source through ML-recommended passes and checks optimized output against that run's original output. Test labels and expected outputs are neither stored in the testing file nor used to fit the model. Results and execution details are saved separately in `results/separated_testing_summary.json` and `results/separated_testing_results.json`.

The website has dedicated **Training Dataset** and **Testing Dataset** navigation pages. It displays exact file paths, record counts, actual model/evaluation status, and source details. The Testing Dataset page allows a selected source-only program to run using the separated model. The grouped-benchmark report remains a separate report pipeline.

## Benchmark outputs

- `results/benchmark_results.csv`
- `results/benchmark_summary.json`
- `results/failures.json`
- `results/ml_recommendations.json`
- `results/ml_training_summary.json`
- `plots/*.png`

The runner checks output against both the optimized output and each record's expected output. Review all failures before using results in your report.

## Known limitations to discuss in the report

- Small supported language; no full conditional/control-flow graph or general loop CFG.
- Counted loops are expanded, not represented as back edges in TAC.
- CSE is conservative and should be improved with variable-dependency invalidation.
- Strength reduction is limited and does not yet optimize induction variables as a production compiler would.
- ML labels indicate single-pass compiler-confirmed applicability, not measured speedup.
- Execution-time measurements on tiny programs are noisy; repeat timing and report medians for final evaluation.


## Connected ML-guided optimization

The ML model is now connected to the C++ optimizer through `ml_optimize.py`:

1. The wrapper reads a source file and extracts its features.
2. It loads `ml/pass_recommender.joblib` and predicts which passes to enable.
3. It invokes the C++ optimizer with `--passes` and the selected pass names.
4. The C++ fixed-point loop runs only the ML-selected passes.
5. The output comparison and metrics are reported by the C++ executable.
6. `run_benchmarks.py` records the ML recommendations but runs all six implemented passes for the primary before/after optimizer evaluation. This separates rule-optimizer effectiveness from the still-experimental ML pass-selection quality.

Run one source file through the connected system:

```powershell
python ml_optimize.py sample.src
```

Run the full integrated pipeline:

```powershell
python run_project.py
```

The 500-program grouped model is trained on training-split applicability labels and its recommendations are evaluated separately on held-out test records. This is not evidence that ML improves performance: compare the separate ML-recommended compiler measurements with the all-six-pass baseline, and interpret them within the documented dataset limitations.

## Single-test and full-benchmark interface

Use the mode selector to run one manually entered program or the existing full pipeline:

```powershell
python optimizer_cli.py
```

Choose **1** for a single test. Enter a supported program on multiple lines and type
`END` on a line by itself to finish, or enter a source-file path when prompted. Choose
whether to enable only the ML-recommended passes or all six rule passes. The report
shows model scores, enabled passes, original/optimized TAC, output equivalence,
available metrics, fixed-point cycles, and actual transformation-event counts.
Interpreter timings are reported as measurements, not as a claim of speedup.

Run a source file directly as one test (paths may be relative to the current directory):

```powershell
python single_test.py --source sample.src
```

The pass selection can be explicit to avoid an interactive pass prompt:

```powershell
python single_test.py --source sample.src --passes recommended
python single_test.py --source sample.src --passes all
```

Alternatively, `python optimizer_cli.py --source sample.src` starts the single-test
flow and asks which pass mode to use. The trained model and compiled C++ executable
must already exist; single-test mode does not regenerate benchmarks or retrain the
model. Train once with `python ml/train_model.py` if the model file is missing.

Each single test saves its isolated `.src` input and `.json` report in
`results/single_test/`. It does not overwrite the benchmark source files, benchmark
CSV/JSON reports, or plots.

Choose **2** in the selector to run the original full pipeline. Its non-interactive
equivalent is:

```powershell
python optimizer_cli.py --full-benchmarks
```

This invokes `run_project.py`, which compiles the optimizer, regenerates the existing
500-program dataset, runs the split-aware ML evaluation and benchmark pipeline,
generates results, and plots. Full-suite reports remain in `results/`; plots remain in
`plots/`.


## Expanded metrics and larger benchmarks

The canonical generator produces 500 programs: 100 per category, with 80 training and 20 testing programs in each. Each record includes its category, source, expected output, unique ID, generation family, fixed seed, size, and split. Existing IDs `CAT1_001` through `CAT5_040` are preserved; added programs use IDs `CAT1_041` through `CAT5_100`. `benchmarks/manifest.json` records counts and family assignments; `benchmarks/dataset.json` contains all records. Related programs stay within their assigned family and do not cross the train/test boundary.

`results/benchmark_results.csv` includes before/after metrics and percentage changes for instructions, arithmetic operations, temporary variables, TAC character count, and interpreter execution time. `results/optimizer_effectiveness.json` reports isolated single-pass effectiveness per optimizer, including number of programs with instruction reductions, total metrics removed, and output-match count. `results/benchmark_summary.json` summarizes the run.

Execution-time measurements are average interpreter time per run across repeated executions and are noisy for very short programs. Per-optimizer tests apply each optimizer independently to the original TAC; this helps attribute isolated effects but does not measure interactions between passes.


## Balanced benchmark dataset

The benchmark generator creates **500 actual per-program JSON files across five categories**, with exactly **100 programs per category** and an **80/20 train/test split**.

| Category | Training | Testing | Total |
|---|---:|---:|---:|
| Constant expressions | 80 | 20 | 100 |
| Algebraic redundancy | 80 | 20 | 100 |
| Dead code | 80 | 20 | 100 |
| Repeated expressions | 80 | 20 | 100 |
| Loop-based | 80 | 20 | 100 |
| **Total** | **400** | **100** | **500** |

Regenerate the balanced dataset and rerun all measurements and plots from the project directory:

```powershell
python run_project.py
```

The generator asserts the total and per-category/split counts, preserves the original IDs, and writes the same deterministic IDs on reruns instead of accumulating duplicates. `run_benchmarks.py` checks the actual individual JSON file count and exact split, not just a declared manifest total. `results/failures.json` must be checked before reporting results. Optimization percentages are measured outputs, not targets; do not alter benchmark data merely to manufacture a desired improvement.

### Optimization convergence plot

`plots/03_optimization_convergence_curve.png` records the mean TAC instruction count after initial compilation and after each **complete optimizer cycle**. The C++ executable supports `--trace-convergence` to report concise per-cycle instruction counts and fixed-point status; raw cycle rows are opt-in and are kept separate from the optimized TAC. Normal output reports the cycle count and aggregate pass invocation/rewrite counts without per-pass convergence debug logs. The all-six-pass strategy always runs Constant Folding, Constant Propagation, Algebraic Simplification, Common Subexpression Elimination, Dead Code Elimination, and Strength Reduction in that order, repeating the complete sequence until rendered TAC is unchanged. A repeated non-stable TAC state is reported as a non-convergence error rather than silently stopping. The plot is generated from benchmark sources and the pass sets recorded in `results/benchmark_results.csv`; `results/convergence_trace.json` contains per-program traces and fixed-point status.


## Benchmark interpretation and limitations

The benchmark runner validates the dataset before running: exactly 500 unique records, five expected categories, and an 80/20 train/test split in every category. It checks both optimized-vs-original output and original-vs-dataset expected output, and records failures in `results/failures.json`. The isolated-pass report includes transformation coverage and total transformation events separately from instructions removed.

**Important limitation:** this educational source language does not currently provide runtime input variables, and counted loops are expanded into straight-line statements before TAC generation. The dataset is therefore strongly compile-time-constant-driven. The reported 100% operation and temporary reductions are specific to this dataset, not evidence that arbitrary programs will achieve those reductions. Likewise, the convergence chart describes repeated optimization of the generated straight-line TAC, not convergence over a retained loop/control-flow graph.

## Convergence and metric interpretation

The pipeline now writes per-program fixed-point cycle counts to `results/convergence_trace.json` and adds aggregate convergence statistics to `results/benchmark_summary.json`. It also generates `plots/07_convergence_cycle_distribution.png`, showing the distribution, median, and maximum number of complete optimization cycles needed to reach a fixed point.

A **complete optimization cycle** runs the enabled optimizer rules once in sequence. The initial TAC is cycle 0; a program's convergence count is the first cycle after which the rendered TAC is unchanged. Transformation-event counts measure individual rewrites/substitutions, not pass invocations and not necessarily instructions removed. The current CSV does not separately record pass-invocation counts.

The current parser expands the supported counted-loop syntax into straight-line statements before optimization. Therefore the loop-based category measures optimization of expanded TAC, not optimization of a retained loop/control-flow graph. The benchmark language also has no runtime input statement, so most operands are compile-time known. These constraints must be mentioned when presenting results; they limit generalization to realistic compiler workloads. Interpreter timings for small programs are noisy and should be treated as descriptive.


## Provenance-aware real-world dataset (separate from the 500 benchmarks)

The additional provenance pipeline lives under `datasets/metadata/` and does not overwrite the canonical CAT1-CAT5 benchmark records, benchmark results, or existing plots.

Raw data is stored separately in:

- `datasets/raw/compiler_tests.json` — three minimal excerpts from two verified optimization test files: LLVM InstCombine `add.ll` (LLVM Project, pinned revision `e833378300e7b695282cae9e21723ebaeafd15a5`) and GCC `pr96392.c` (GCC project, pinned revision `58e696376aecf8b895bb952a99988538cf366479`). The records include source URLs, paths, revisions, repository-level license notes, original excerpts, adapted programs, and adaptation notes.
- `datasets/raw/synthetic_examples.json` — twelve separately labeled, project-generated controls.
- `datasets/raw/textbook_examples.json` — currently empty. No textbook example was added because a suitable source and reusable license were not verified. Do not treat this as evidence that textbook examples were searched exhaustively.

`datasets/metadata/source_manifest.json` is the source of truth for per-program provenance and the five checked candidate references. It distinguishes dataset programs from conceptual references and records original/adapted code, checked-source status, execution status, source URL/revision/path, and license notes. Of the 15 dataset programs, **1** is verified from an LLVM compiler-test file, **2** from a GCC compiler-test file, and **12** are project-generated. None is attributed to NPTEL, Vassar, INFLIBNET, or either LLVM guide. Those pages are listed as conceptual references only; all five have zero verified dataset-origin examples. Similar optimization patterns do not prove that a generated program came from a lecture. The supplied NPTEL page includes a constant-folding example, Vassar's page includes local-optimization examples, and the INFLIBNET module discusses quicksort-derived optimization examples; those were not imported as dataset programs.

LLVM's repository `LICENSE.TXT` identifies Apache-2.0 with LLVM exceptions. GCC's repository `COPYING3` contains GPL version 3; per-file copyright/assignment was not independently verified, so the manifest records that qualification. License terms for the candidate educational references were not verified and are explicitly recorded as such. Source snippets are kept minimal. Full repositories and textbooks are not downloaded. Raw LLVM IR and C function excerpts are not accepted by this educational compiler; each adapted source is kept in a separate field and is not presented as an equivalent translation for all possible inputs.

The compatibility pipeline runs each adapted source through the existing C++ executable. It records supported-syntax/parse, TAC, interpreter, and output-equivalence checks; incompatible items are written to `datasets/processed/rejected_programs.json`. For external examples it separately records whether the original excerpt is supported. It uses the existing 18-value `ml.train_model.features` extractor on pre-optimization source only; `feature_names` in each processed record and `datasets/metadata/dataset_statistics.json` documents the positional features. Each optimization pass is measured independently on original TAC. The multi-label training target is positive only when the pass transforms the program, preserves interpreter output, and reduces at least one static metric (instruction count, operation count, temporary count, or TAC characters). These labels are not execution-speedup labels.

Compatible programs are grouped by `source_family` before splitting, use seed `307`, and are assigned to train/validation/test. IDs are saved in `datasets/splits/`; overlap checks cover IDs, normalized programs, and families. The Random Forest is fitted only on train; pass thresholds are selected using validation labels; final multi-label metrics and effectiveness comparisons use only held-out test records. The separate trained bundle is available to `ml_optimize.py` or `single_test.py` through `--model`. When there are too few families or classes, the pipeline reports the limitation rather than treating the estimate as reliable.

From the project directory, use these Windows PowerShell commands:

```powershell
g++ -std=c++17 -O2 -Wall src\main.cpp -o compiler_optimizer.exe
python datasets\run_real_world_pipeline.py
python ml_optimize.py sample.src --model ml\real_world_pass_recommender.joblib
```

Run the synthetic benchmark pipeline separately:

```powershell
python run_project.py
```

Run a single case with the original legacy model or explicitly with the new real-world model:

```powershell
python single_test.py --source sample.src --passes all
python single_test.py --source sample.src --passes recommended --model ml\real_world_pass_recommender.joblib
```

The dataset pipeline writes compatible/rejected records and split IDs under `datasets\processed\` and `datasets\splits\`, provenance/statistics under `datasets\metadata\`, test predictions and optimization results under `results\real_world_dataset\`, a separate model at `ml\real_world_pass_recommender.joblib`, and ten new plots under `plots\real_world\`. The original seven plots under `plots\` remain generated by `make_plots.py`.

The real-source sample is intentionally small and the source language is much narrower than LLVM IR or C. Adaptations use fixed inputs and test a local expression pattern only. Metrics are descriptive; source-pattern overlap, limited test counts, expanded loops, heuristic source-text features, and noisy microsecond timings all constrain generalization. Do not combine these figures with the original synthetic benchmark's headline averages.

## Grouped 500-program ML benchmark

The canonical benchmark dataset is separate from the 15-record provenance dataset. It contains 500 generated supported-language programs: 100 in each existing category, with 80 training and 20 testing records per category. The fixed seed is `307`; all records from each legacy generator cohort and each new template family stay in one split, preventing related variants from leaking across splits. `datasets/generate_grouped_benchmarks.py` remains as a compatibility entry point and invokes the same canonical generator.

From the project root in PowerShell, compile the optimizer if needed and run the complete grouped pipeline:

```powershell
g++ -std=c++17 -O2 -Wall src\main.cpp -o compiler_optimizer.exe
python run_benchmark_ml.py
```

The pipeline regenerates the actual `benchmarks\CAT*.json` files and canonical `benchmarks\dataset.json`, writes six-field training records and source-only testing records, then derives compiler features and pass applicability in memory. The Random Forest is fit only from the 400 training programs. Evaluation runs only on the 100 held-out testing sources. For each test program, the pipeline runs ML-recommended passes and all six passes as separate executions, checks outputs, and records final TAC metrics, rewrite counts, and optimization cycles. Compiler-derived features and targets are never written back into the training/testing JSON files. The website validates the actual record schemas and source hashes before showing the grouped dataset; stale reports are rejected rather than presented as current.

Generated artifacts:

- `datasets\benchmark_manifest.json` — category/split counts, seed, family assignments, label method, and output-check status.
- `datasets\training\` and `datasets\testing\` — per-category program JSON and consolidated `programs.json`, with no embedded features or labels.
- `ml\benchmark_pass_recommender.joblib` — trained model bundle and its feature/preprocessing metadata.
- `results\benchmark_v2_training_summary.json`, `benchmark_v2_ml_evaluation.json`, `benchmark_v2_compiler_results.json`, `benchmark_v2_compiler_results.csv`, and `benchmark_v2_summary.json` — training provenance, held-out multi-label metrics/confusion counts, independent compiler runs, and aggregate results.
- `plots\benchmark_v2\` — figures dedicated to this suite; legacy plots are not overwritten.

Run the new automated checks after the pipeline has generated the labels and reports:

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s web\tests -v
```

On the recorded run, held-out pass-label accuracy was **97.17%**, micro precision **97.52%**, micro recall **96.49%**, micro F1 **97.00%**, and exact six-label subset accuracy **83.00%**. These are classification metrics on the 100 generated test programs—not optimizer speedups. Both independent compiler modes preserved expected output for **100/100** test programs. The mean instruction reduction was **28.27%** for the recommended-pass run and **65.01%** for the separate all-six-pass baseline; the latter is not evidence of better ML recommendations. This benchmark is synthetic, template-based, and small in family diversity (10 held-out test families); it does not establish generalization to real-world code.


## Local optimizer demo website

The `web/` folder contains a local-only dashboard for the existing C++ optimizer and project reports. It uses Python's standard-library HTTP server and a static HTML/CSS/JavaScript frontend; no frontend build or web-framework dependency is required. The website serves on `127.0.0.1` only and does not upload source code.

Before launching, compile the C++ optimizer and make sure the recommender model files and Python dependencies are available. Then, from the project root in PowerShell, start the website:

```powershell
.\web\start.ps1
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). To use a different port, run the server directly:

```powershell
python web\server.py --port 8080
```

The playground accepts only the existing educational source language. It calls the compiled optimizer through an argument list, uses temporary source files in `web\runtime\tmp\`, validates input with the real compiler, and applies a per-run timeout. ML-recommended and all-pass execution are separate options. The report distinguishes model recommendations, passes selected for execution, and passes that actually performed rewrites.

Dataset and benchmark views read the existing manifest, metadata, CSV, JSON, and plot files; they do not regenerate data on startup. The **Run full pipeline** button is an explicit operation that runs the same `run_project.py` pipeline as the command line, including recompiling, regenerating the synthetic benchmark dataset, retraining the legacy model, and refreshing benchmark reports and plots. It may take several minutes and overwrite generated results.

Run the focused website/API checks from the project root with:

```powershell
python -m unittest discover -s web\tests -v
```
