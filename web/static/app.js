const state = {
  dataset: null,
  datasetQuery: "",
  benchmarkPage: 1,
  benchmarkLimit: 25,
  benchmarkTotal: 0,
  benchmarkSummary: null,
  groupedDataset: null,
  groupedDatasetPage: 1,
  separatedTraining: [],
  separatedTesting: [],
  testingRunStatuses: [],
  trainingPoll: null,
  benchmarkV2Page: 1,
  benchmarkV2Limit: 25,
  benchmarkV2Total: 0,
  datasetFilters: { category: "", split: "", pass: "", size: "" },
  benchmarkV2Filters: { category: "", split: "testing", pass: "", size: "" },
  benchmarkPoll: null,
};

const optimizerPasses = [
  ["constant_folding", "Constant Folding"],
  ["constant_propagation", "Constant Propagation"],
  ["algebraic_simplification", "Algebraic Simplification"],
  ["common_subexpression_elimination", "Common Subexpression Elimination"],
  ["dead_code_elimination", "Dead Code Elimination"],
  ["strength_reduction", "Strength Reduction"],
];
const chartDetails = [
  ["01_instruction_count_before_after.png", "Instruction count", "BEFORE / AFTER"],
  ["02_optimizer_passes_by_program_stacked.png", "Pass activity by program", "ALL SIX PASSES"],
  ["03_optimization_convergence_curve.png", "Optimization convergence", "ITERATION TRACE"],
  ["04_optimization_type_distribution.png", "Optimization distribution", "RULE BREAKDOWN"],
  ["05_code_size_vs_interpreter_time.png", "Code size and interpreter time", "MEASURED"],
  ["06_optimizer_category_heatmap.png", "Optimizer category heatmap", "CATEGORY DETAIL"],
  ["07_convergence_cycle_distribution.png", "Fixed-point cycle distribution", "CONVERGENCE"],
];
const groupedChartDetails = [
  ["01_category_train_test.png", "Category train/test balance", "500 GENERATED PROGRAMS"],
  ["02_ml_metrics_by_pass.png", "ML metrics by optimization pass", "HELD-OUT LABELS"],
  ["03_ml_f1_by_category.png", "ML F1 by category", "HELD-OUT LABELS"],
  ["04_compiler_ml_vs_all_passes.png", "Independent compiler runs", "NOT ML ACCURACY"],
];

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));
const externalHref = (value) => {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? escapeHtml(url.href) : "#";
  } catch {
    return "#";
  }
};
const formatNumber = (value, digits = 0) => {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }) : "—";
};
const percent = (value) => Number.isFinite(Number(value)) ? `${formatNumber(value, 1)}%` : "—";
const shortOutput = (value) => value === null || value === undefined ? "—" : String(value);

async function api(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok && !data.status && data.available !== false) {
    throw new Error(data.error || `Request failed (${response.status}).`);
  }
  return { response, data };
}

function setPage(page) {
  document.querySelectorAll(".page").forEach((section) => {
    section.classList.toggle("active", section.id === `page-${page}`);
  });
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.page === page);
  });
  $("#page-crumb").textContent = ({
    overview: "Overview",
    playground: "Playground",
    dataset: "Dataset explorer",
    training: "Training Dataset",
    testing: "Testing Dataset",
    benchmarks: "Benchmarks",
  })[page] || "Overview";
  if (page === "dataset") loadDataset();
  if (page === "training") loadSeparatedTraining();
  if (page === "testing") loadSeparatedTesting();
  if (page === "benchmarks") loadBenchmarks();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function chartCard([filename, title, tag]) {
  return `<article class="chart-card"><div class="chart-card-head"><b>${escapeHtml(title)}</b><span>${escapeHtml(tag)}</span></div><div class="chart-image-wrap"><img src="/plots/${encodeURIComponent(filename)}" alt="${escapeHtml(title)}" loading="lazy" onerror="this.outerHTML='<span class=&quot;chart-missing&quot;>Chart not generated yet. Run the explicit benchmark pipeline to refresh it.</span>'"></div></article>`;
}

function groupedChartCard([filename, title, tag]) {
  return `<article class="chart-card"><div class="chart-card-head"><b>${escapeHtml(title)}</b><span>${escapeHtml(tag)}</span></div><div class="chart-image-wrap"><img src="/plots/benchmark_v2/${encodeURIComponent(filename)}" alt="${escapeHtml(title)}" loading="lazy" onerror="this.outerHTML='<span class=&quot;chart-missing&quot;>Grouped benchmark chart is not available.</span>'"></div></article>`;
}

function renderHealth(data) {
  const indicator = $("#api-indicator");
  indicator.classList.toggle("online", Boolean(data.status === "ok" && data.optimizer_available));
  indicator.classList.toggle("offline", Boolean(data.status !== "ok" || !data.optimizer_available));
  indicator.querySelector("span").textContent = data.optimizer_available ? "API connected" : "Compiler unavailable";
  const missingModels = Object.entries(data.models || {}).filter(([, model]) => !model.available).map(([name]) => name);
  const health = $("#health-strip");
  if (!data.optimizer_available || !data.ml_dependencies_available) {
    health.classList.add("error");
    health.innerHTML = `<span class="pulse-dot"></span><span>${!data.optimizer_available ? "C++ optimizer executable is missing. Compile src\\main.cpp to use the playground." : `Python ML dependencies unavailable: ${escapeHtml(data.ml_dependency_error)}.`}</span>`;
  } else {
    health.classList.remove("error");
    const modelState = missingModels.length ? ` Missing model(s): ${missingModels.map(escapeHtml).join(", ")}.` : " All recommender models are available.";
    health.innerHTML = `<span class="pulse-dot"></span><span>Local optimizer ready.${modelState} Source input is processed on this machine.</span>`;
  }
  const modelSelect = $("#model-select");
  [...modelSelect.options].forEach((option) => {
    option.disabled = !data.models?.[option.value]?.available;
  });
  if (modelSelect.selectedOptions[0]?.disabled) {
    const available = [...modelSelect.options].find((option) => !option.disabled);
    if (available) modelSelect.value = available.value;
  }
  updateModelNote();
}

function updateModelNote() {
  const model = $("#model-select").value;
  $("#model-note").textContent = model === "separated_training"
    ? "Trained only from dataset/training/training.json; the source-only testing file is evaluated separately."
    : model === "grouped_benchmark"
    ? "Trained only on 400 training examples with compiler-confirmed applicability labels; evaluated on 100 family-held-out programs."
    : model === "real_world"
      ? "Trained on the small provenance-aware dataset. Predictions are experimental, not generalization claims."
      : "Legacy model trained from synthetic applicability labels. Predictions are experimental.";
}

function renderMetric(selector, value, note) {
  const container = $(selector);
  container.innerHTML = `<div class="metric-top"><span>${escapeHtml(note.label)}</span><span class="metric-icon ${escapeHtml(note.color || "")}">${escapeHtml(note.icon || "◇")}</span></div><div class="metric-value">${escapeHtml(value)}</div><div class="metric-note">${escapeHtml(note.description || "")}</div>`;
}

async function loadOverview() {
  $("#overview-charts").innerHTML = chartDetails.slice(0, 4).map(chartCard).join("");
  try {
    const [healthResult, benchmarkResult, datasetResult] = await Promise.all([
      api("/api/health"),
      api("/api/benchmarks/summary"),
      api("/api/dataset/summary"),
    ]);
    renderHealth(healthResult.data);
    const summary = benchmarkResult.data.summary;
    $("#metric-benchmarks").textContent = summary ? formatNumber(summary.total_benchmarks) : "—";
    $("#metric-validation").textContent = summary
      ? `${formatNumber(summary.validated_programs)} / ${formatNumber(summary.total_benchmarks)}`
      : "—";
    $("#metric-dataset").textContent = datasetResult.data.summary
      ? formatNumber(datasetResult.data.summary.program_count)
      : "—";
  } catch (error) {
    $("#health-strip").classList.add("error");
    $("#health-strip").innerHTML = `<span class="pulse-dot"></span><span>${escapeHtml(error.message)}</span>`;
    $("#api-indicator").classList.add("offline");
    $("#api-indicator").querySelector("span").textContent = "API unavailable";
  }
}

function updateEditorStats() {
  const source = $("#source-code").value;
  const lines = Math.max(1, source.split("\n").length);
  $("#source-count").textContent = `${lines} line${lines === 1 ? "" : "s"} · ${source.length.toLocaleString()} chars`;
  $("#line-numbers").innerHTML = Array.from({ length: lines }, (_, index) => index + 1).join("<br>");
}

function setText(id, text) {
  $(`#${id}`).textContent = text === null || text === undefined ? "—" : String(text);
}

function renderOptimization(data) {
  const results = $("#optimizer-results");
  results.hidden = false;
  const status = $("#result-status");
  status.textContent = data.status;
  status.className = `result-status ${data.status.toLowerCase()}`;
  $("#result-title").textContent = data.status === "ERROR" ? "Compiler error" : "Optimization report";
  const callout = data.status === "PASS"
    ? `Output equivalence confirmed by the C++ compiler. ${data.mode === "all" ? "All six passes were selected." : "The selected passes came from the chosen ML model."}`
    : data.error || "Compiler output did not confirm equivalent results.";
  $("#result-callout").textContent = callout;
  const metrics = data.metrics || {};
  $("#result-metrics").innerHTML = [
    ["INSTRUCTIONS", `${formatNumber(metrics.instructions_before)} → ${formatNumber(metrics.instructions_after)}`],
    ["OPERATIONS", `${formatNumber(metrics.operations_before)} → ${formatNumber(metrics.operations_after)}`],
    ["TEMPORARIES", `${formatNumber(metrics.temporaries_before)} → ${formatNumber(metrics.temporaries_after)}`],
  ].map(([label, value]) => `<div class="result-metric"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></div>`).join("");
  $("#pass-results").innerHTML = (data.passes || []).map((item) => {
    const recommendation = item.recommended === null ? "N/A" : item.recommended ? "YES" : "NO";
    const selected = item.selected ? "YES" : "NO";
    const applied = item.applied ? "YES" : "NO";
    const score = item.recommendation_score === null || item.recommendation_score === undefined
      ? "" : ` <small>${formatNumber(item.recommendation_score, 2)}</small>`;
    return `<tr><td>${escapeHtml(item.label)}</td><td><span class="pass-state ${item.recommended ? "yes" : "no"}">${recommendation}${score}</span></td><td>${escapeHtml(selected)}</td><td><span class="pass-state ${item.applied ? "yes" : "no"}">${applied}</span></td><td>${formatNumber(item.transformations)}</td></tr>`;
  }).join("");
  setText("original-output", data.original_output);
  setText("optimized-output", data.optimized_output);
  setText("original-tac", data.original_tac);
  setText("optimized-tac", data.optimized_tac);
  if (data.model_recommendation_error && data.mode === "all") {
    $("#result-callout").textContent += ` ML recommendation unavailable: ${data.model_recommendation_error}`;
  }
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runOptimization() {
  const button = $("#run-optimizer");
  const message = $("#run-message");
  button.disabled = true;
  message.classList.remove("error");
  message.textContent = "Validating input and running the actual C++ optimizer…";
  try {
    const { data } = await api("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: $("#source-code").value,
        mode: $("#mode-select").value,
        model: $("#model-select").value,
      }),
    });
    renderOptimization(data);
    message.textContent = data.status === "PASS"
      ? "Optimization completed. Outputs match."
      : data.error || "Compiler did not confirm output equivalence.";
    if (data.status !== "PASS") message.classList.add("error");
  } catch (error) {
    message.textContent = error.message;
    message.classList.add("error");
  } finally {
    button.disabled = false;
  }
}

function datasetProgramRow(program) {
  const originLabel = program.source_type === "generated" ? "Generated" : "Compiler test";
  const patterns = (program.optimization_patterns || []).map((pattern) => escapeHtml(pattern.replaceAll("_", " "))).join(", ") || "—";
  const executed = program.adapted_program_executed_successfully === true
    || program.execution_result?.can_execute === true;
  return `<tr><td>${escapeHtml(program.id)}</td><td><span class="source-chip ${program.source_type === "generated" ? "generated" : ""}">${originLabel}</span></td><td>${patterns}</td><td>${escapeHtml(program.author_or_organization || "Not recorded")}</td><td><span class="status-chip ${executed ? "pass" : "fail"}">${executed ? "Adapted program ran" : "Not verified"}</span></td><td><button class="row-detail-button" data-record="${escapeHtml(program.id)}">Details ↗</button></td></tr>`;
}

function renderDataset(data) {
  state.dataset = data;
  if (!data.available) {
    $("#dataset-warning").hidden = false;
    $("#dataset-warning").textContent = data.error || "Dataset metadata is unavailable.";
    return;
  }
  $("#dataset-warning").hidden = true;
  const summary = data.summary || {};
  const values = [
    ["PROGRAM RECORDS", summary.program_count, "verified source records", "◇", "violet"],
    ["EXTERNAL TESTS", summary.source_types?.compiler_test, "LLVM + GCC excerpts", "↗", "cyan"],
    ["GENERATED", summary.source_types?.generated, "explicitly labeled synthetic", "✳", "violet"],
    ["COMPATIBLE", summary.compatible_count, `${formatNumber(summary.rejected_count)} rejected`, "✓", "green"],
  ];
  $("#dataset-metrics").innerHTML = values.map(([label, value, description, icon, color]) => `<article class="metric-card"><div class="metric-top"><span>${escapeHtml(label)}</span><span class="metric-icon ${color}">${icon}</span></div><div class="metric-value">${formatNumber(value)}</div><div class="metric-note">${escapeHtml(description)}</div></article>`).join("");
  const origins = summary.verified_origins || {};
  $("#dataset-provenance-note").textContent = `${formatNumber(origins.textbook_or_lecture_origins_from_supplied_candidates)} verified programs originated from the five candidate educational/compiler references. ${formatNumber(origins.llvm_compiler_test)} LLVM and ${formatNumber(origins.gcc_compiler_test)} GCC test excerpts, plus ${formatNumber(origins.generated)} generated programs. Conceptual references are presented separately.`;
  renderDatasetRows();
  $("#conceptual-references").innerHTML = (data.conceptual_references || []).map((reference) => `<article class="reference-card"><div class="reference-count">CONCEPTUAL REFERENCE · ${formatNumber(reference.verified_dataset_example_count)} DATASET ORIGINS</div><h3>${escapeHtml(reference.source_title)}</h3><p>${escapeHtml(reference.source_check_result || "")}</p><a href="${externalHref(reference.source_url)}" target="_blank" rel="noopener noreferrer">Open checked reference ↗</a></article>`).join("");
}

function renderDatasetRows() {
  const data = state.dataset;
  if (!data?.available) return;
  const query = state.datasetQuery.trim().toLocaleLowerCase();
  const programs = (data.programs || []).filter((program) =>
    !query || JSON.stringify(program).toLocaleLowerCase().includes(query)
  );
  $("#dataset-rows").innerHTML = programs.length
    ? programs.map(datasetProgramRow).join("")
    : '<tr><td colspan="6" class="empty-cell">No dataset records match this search.</td></tr>';
}

function displayLabel(value) {
  return String(value || "").replaceAll("_", " ");
}

function applicablePassNames(labels) {
  return optimizerPasses
    .filter(([name]) => Number(labels?.[name]) === 1 || labels?.[name] === true)
    .map(([, label]) => label);
}

function renderPassOptions(selectId, selected = "") {
  const select = $(`#${selectId}`);
  if (!select) return;
  const firstOption = select.options[0]?.outerHTML || "";
  select.innerHTML = firstOption + optimizerPasses.map(([value, label]) =>
    `<option value="${value}">${escapeHtml(label)}</option>`
  ).join("");
  select.value = selected;
}

function renderGroupedSummary(summary) {
  const categories = summary.category_counts || {};
  const entries = Object.entries(categories);
  $("#grouped-dataset-metrics").innerHTML = [
    metricCardMarkup("TOTAL PROGRAMS", formatNumber(summary.program_count), "generated benchmark programs", "▦"),
    metricCardMarkup("TRAINING", formatNumber(summary.training_count), "training-only model fitting", "◧", "cyan"),
    metricCardMarkup("HELD-OUT TESTING", formatNumber(summary.testing_count), "unseen generation families", "✓", "green"),
    metricCardMarkup("STRICT RECORD SCHEMAS", `${summary.training_record_fields?.length || 0} / ${summary.testing_record_fields?.length || 0}`, "training fields / testing fields", "◇", "violet"),
  ].join("");
  $("#grouped-dataset-seed").textContent = `Training: ${summary.training_record_fields?.join(", ") || "unavailable"} · Testing: ${summary.testing_record_fields?.join(", ") || "unavailable"}`;
  $("#grouped-category-rows").innerHTML = entries.map(([category, counts]) =>
    `<tr><td>${escapeHtml(displayLabel(category))}</td><td>${formatNumber(counts.total)}</td><td>${formatNumber(counts.training)}</td><td>${formatNumber(counts.testing)}</td><td>6 / source only</td></tr>`
  ).join("");
  for (const prefix of ["dataset-v2", "benchmark-v2"]) {
    const select = $(`#${prefix}-category`);
    const selected = select.value;
    const firstOption = select.options[0]?.outerHTML || "";
    select.innerHTML = firstOption + entries.map(([value]) =>
      `<option value="${escapeHtml(value)}">${escapeHtml(displayLabel(value))}</option>`
    ).join("");
    select.value = selected;
    renderPassOptions(`${prefix}-pass`, $(`#${prefix}-pass`).value);
  }
}

function groupedRunCell(result, mode) {
  const run = result?.[mode];
  if (!run) return '<span class="muted-cell">No compiler run recorded</span>';
  const equivalent = run.output_equivalent === true && run.expected_output_match === true;
  const before = run.instructions_before;
  const after = run.instructions_after;
  const rewrites = Object.values(run.actual_rewrite_counts || {}).reduce((sum, value) => sum + Number(value || 0), 0);
  return `<span class="status-chip ${equivalent ? "pass" : "fail"}">${equivalent ? "OUTPUT MATCH" : "OUTPUT FAILURE"}</span><small class="benchmark-cell-detail">Instructions ${formatNumber(before)} → ${formatNumber(after)} · Ops ${formatNumber(run.operations_before)} → ${formatNumber(run.operations_after)} · Temps ${formatNumber(run.temporaries_before)} → ${formatNumber(run.temporaries_after)} · Code ${formatNumber(run.code_size_before)} → ${formatNumber(run.code_size_after)} chars · ${formatNumber(rewrites)} rewrites · ${formatNumber(run.optimization_cycles)} cycles</small>`;
}

function groupedRecordRow(record, showSize) {
  const labels = record.applicability_labels
    ? applicablePassNames(record.applicability_labels).join(", ") || "None"
    : "Not stored in training JSON";
  const recommendations = record.ml_recommendations;
  const recommended = recommendations
    ? optimizerPasses.filter(([name]) => recommendations[name]?.recommended).map(([, label]) => label)
    : [];
  const result = record.compiler_result;
  if (showSize) {
    return `<tr><td>${escapeHtml(record.id)}</td><td>${escapeHtml(displayLabel(record.category))}</td><td>${escapeHtml(record.split)}</td><td>${escapeHtml(record.size)}</td><td>${escapeHtml(labels)}</td><td>${recommendations ? escapeHtml(recommended.join(", ") || "None") : "Not evaluated"}</td><td>${groupedRunCell(result, "ml_recommended")}</td></tr>`;
  }
  return `<tr><td>${escapeHtml(record.id)}<small class="benchmark-cell-detail">${escapeHtml(record.category)} / ${escapeHtml(record.split)}</small></td><td>${escapeHtml(labels)}</td><td>${recommendations ? escapeHtml(recommended.join(", ") || "None") : "Not evaluated"}</td><td>${groupedRunCell(result, "ml_recommended")}</td><td>${groupedRunCell(result, "all_passes_baseline")}</td></tr>`;
}

function renderGroupedDataset(data) {
  const warning = $("#grouped-dataset-warning");
  if (!data.available) {
    warning.hidden = false;
    warning.textContent = data.error || "Grouped benchmark data is unavailable.";
    return;
  }
  warning.hidden = true;
  state.groupedDataset = data;
  renderGroupedSummary(data.summary || {});
  const pageCount = Math.max(1, Math.ceil(data.total_filtered / data.limit));
  $("#dataset-v2-rows").innerHTML = data.programs.length
    ? data.programs.map((record) => groupedRecordRow(record, true)).join("")
    : '<tr><td colspan="7" class="empty-cell">No programs match these filters.</td></tr>';
  $("#dataset-v2-pagination").textContent = `${formatNumber(data.total_filtered)} records · page ${data.page} of ${pageCount}`;
  $("#dataset-v2-prev").disabled = data.page <= 1;
  $("#dataset-v2-next").disabled = data.page >= pageCount;
}

function renderBenchmarkV2(data) {
  const warning = $("#benchmark-v2-warning");
  if (!data.available) {
    warning.hidden = false;
    warning.textContent = data.error || "Grouped benchmark reports are unavailable.";
    return;
  }
  warning.hidden = true;
  const summary = data.summary || {};
  const ml = summary.ml_metrics || {};
  const compiler = summary.compiler_summary || {};
  $("#benchmark-v2-metrics").innerHTML = [
    metricCardMarkup("PROGRAMS", formatNumber(summary.program_count), "balanced generated suite", "▦"),
    metricCardMarkup("TRAIN / TEST", `${formatNumber(summary.training_count)} / ${formatNumber(summary.testing_count)}`, "family-held-out split", "◧", "cyan"),
    metricCardMarkup("PASS ACCURACY", percent((ml.overall_pass_recommendation_accuracy || 0) * 100), "ML labels · testing only", "◎", "violet"),
    metricCardMarkup("MICRO PRECISION", percent((ml.overall_precision_micro || 0) * 100), "ML recommendations · testing only", "⌕"),
    metricCardMarkup("MICRO RECALL", percent((ml.overall_recall_micro || 0) * 100), "ML recommendations · testing only", "↗", "cyan"),
    metricCardMarkup("MICRO F1", percent((ml.overall_f1_micro || 0) * 100), "ML recommendations · testing only", "⌁", "green"),
    metricCardMarkup("OUTPUT CHECKS", `${formatNumber(compiler.recommended_correct_outputs)} / ${formatNumber(compiler.test_program_count)}`, "ML-recommended compiler runs", "✓", "green"),
    metricCardMarkup("ALL-PASS BASELINE", percent(compiler.all_passes_average_instruction_reduction_pct), "mean instruction reduction · separate run", "↘", "cyan"),
  ].join("");
  const passMetrics = summary.ml_per_pass || {};
  $("#ml-pass-metrics").innerHTML = optimizerPasses.map(([name, label]) => {
    const metric = passMetrics[name] || {};
    const matrix = metric.confusion_matrix || {};
    return `<tr><td>${escapeHtml(label)}</td><td>${percent(Number(metric.accuracy || 0) * 100)}</td><td>${percent(Number(metric.precision || 0) * 100)}</td><td>${percent(Number(metric.recall || 0) * 100)}</td><td>${percent(Number(metric.f1 || 0) * 100)}</td><td>${formatNumber(matrix.true_negative)} / ${formatNumber(matrix.false_positive)} / ${formatNumber(matrix.false_negative)} / ${formatNumber(matrix.true_positive)}</td></tr>`;
  }).join("");
  $("#ml-evaluation-count").textContent = `${formatNumber(summary.testing_count)} HELD-OUT PROGRAMS`;
  $("#ml-evaluation-note").textContent = summary.evaluation_note || "";
  const byCategory = summary.ml_per_category || {};
  $("#ml-category-metrics").innerHTML = Object.entries(byCategory).map(([category, metric]) =>
    `<tr><td>${escapeHtml(displayLabel(category))}</td><td>${formatNumber(metric.program_count)}</td><td>${percent(Number(metric.subset_accuracy || 0) * 100)}</td><td>${percent(Number(metric.aggregate?.f1 || 0) * 100)}</td></tr>`
  ).join("");
  $("#compiler-comparison").innerHTML = `
    <div class="comparison-run"><div class="result-block-title">ML-RECOMMENDED PASSES <small>Separate test-set compiler run</small></div><div class="comparison-stats"><span>OUTPUT MATCH <b>${formatNumber(compiler.recommended_correct_outputs)} / ${formatNumber(compiler.test_program_count)}</b></span><span>INSTRUCTION REDUCTION <b>${percent(compiler.ml_recommended_average_instruction_reduction_pct)}</b></span><span>CODE-SIZE REDUCTION <b>${percent(compiler.ml_recommended_average_code_size_reduction_pct)}</b></span><span>MEAN CYCLES <b>${formatNumber(compiler.ml_recommended_mean_optimization_cycles, 1)}</b></span></div></div>
    <div class="comparison-run"><div class="result-block-title">ALL-SIX-PASS BASELINE <small>Independent full-pass test-set run</small></div><div class="comparison-stats"><span>OUTPUT MATCH <b>${formatNumber(compiler.all_passes_correct_outputs)} / ${formatNumber(compiler.test_program_count)}</b></span><span>INSTRUCTION REDUCTION <b>${percent(compiler.all_passes_average_instruction_reduction_pct)}</b></span><span>CODE-SIZE REDUCTION <b>${percent(compiler.all_passes_average_code_size_reduction_pct)}</b></span><span>MEAN CYCLES <b>${formatNumber(compiler.all_passes_mean_optimization_cycles, 1)}</b></span></div></div>`;
  $("#benchmark-v2-charts").innerHTML = groupedChartDetails.map(groupedChartCard).join("");
  const pageCount = Math.max(1, Math.ceil(data.total_filtered / data.limit));
  state.benchmarkV2Total = data.total_filtered;
  $("#benchmark-v2-rows").innerHTML = data.programs.length
    ? data.programs.map((record) => groupedRecordRow(record, false)).join("")
    : '<tr><td colspan="5" class="empty-cell">No records match these filters.</td></tr>';
  $("#benchmark-v2-pagination").textContent = `${formatNumber(data.total_filtered)} records · page ${data.page} of ${pageCount}`;
  $("#benchmark-v2-prev").disabled = data.page <= 1;
  $("#benchmark-v2-next").disabled = data.page >= pageCount;
}

function filterQuery(filters, page, limit) {
  const params = new URLSearchParams({ page: String(page), limit: String(limit) });
  for (const [name, value] of Object.entries(filters)) {
    if (value) params.set(name, value);
  }
  return params.toString();
}

async function loadGroupedDataset() {
  try {
    const { data } = await api(`/api/benchmark-ml?${filterQuery(state.datasetFilters, state.groupedDatasetPage, 25)}`);
    renderGroupedDataset(data);
  } catch (error) {
    $("#grouped-dataset-warning").hidden = false;
    $("#grouped-dataset-warning").textContent = error.message;
  }
}

function renderSeparatedTraining(data) {
  if (!data.available) {
    $("#training-dataset-status").textContent = data.error || "Training dataset is unavailable.";
    $("#training-dataset-status").classList.add("error");
    return;
  }
  state.separatedTraining = data.programs || [];
  const status = data.status || {};
  $("#training-dataset-status").classList.toggle("error", !status.model_trained);
  $("#training-dataset-status").textContent = status.model_trained
    ? `Model trained from ${status.training_file} using ${formatNumber(status.training_samples)} training programs.`
    : `Model not trained from ${status.training_file || data.file}. Use the button to train from Training Dataset only.`;
  $("#training-dataset-metrics").innerHTML = [
    metricCardMarkup("TRAINING PROGRAMS", formatNumber(data.program_count), data.file, "↑", "cyan"),
    metricCardMarkup("TRAINING STATUS", status.model_trained ? "TRAINED" : "NOT TRAINED", status.model_path || "Dedicated separated model", "◎", status.model_trained ? "green" : ""),
    metricCardMarkup("MODEL TRAINING COUNT", formatNumber(status.training_samples), "records read from training.json", "▦"),
    metricCardMarkup("RECORD SCHEMA", "6 fields", "no stored features, labels, or split metadata", "◇"),
  ].join("");
  $("#training-category-rows").innerHTML = Object.entries(data.category_distribution || {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([category, count]) => `<tr><td>${escapeHtml(displayLabel(category))}</td><td>${formatNumber(count)}</td></tr>`)
    .join("");
  const selector = $("#training-program-select");
  const selectedIndex = selector.value;
  selector.innerHTML = state.separatedTraining.map((record, index) =>
    `<option value="${index}">${escapeHtml(record.id)} · ${escapeHtml(displayLabel(record.category))} · ${escapeHtml(record.size)}</option>`
  ).join("");
  if (selectedIndex && Number(selectedIndex) < state.separatedTraining.length) selector.value = selectedIndex;
  renderSelectedTrainingProgram();
  const button = $("#train-separated-model");
  button.disabled = status.training_job?.status === "running";
  button.textContent = button.disabled ? "Training and evaluating…" : "Train model on training.json";
}

function renderSelectedTrainingProgram() {
  const record = state.separatedTraining[Number($("#training-program-select").value) || 0];
  if (!record) return;
  $("#training-program-meta").textContent =
    `${record.id} · ${displayLabel(record.category)} · ${record.size} · inputs ${JSON.stringify(record.inputs)}`;
  $("#training-program-source").textContent = record.source;
  $("#training-program-output").textContent = record.expected_output;
  $("#training-program-record").textContent = JSON.stringify(record, null, 2);
}

function renderSeparatedTesting(data) {
  if (!data.available) {
    $("#testing-dataset-status").textContent = data.error || "Testing dataset is unavailable.";
    $("#testing-dataset-status").classList.add("error");
    return;
  }
  state.separatedTesting = data.programs || [];
  state.testingRunStatuses = data.execution_status || [];
  const status = data.status || {};
  $("#testing-dataset-status").classList.toggle("error", !status.model_trained);
  $("#testing-dataset-status").textContent = !status.model_trained
    ? "Testing is unavailable until the separate training dataset has produced a model."
    : status.testing_evaluated
      ? `Model trained. Saved evaluation ran ${formatNumber(status.testing_executed_count)} testing sources. Select a source to execute it again.`
      : "Model trained; testing sources have not yet been evaluated.";
  $("#testing-dataset-metrics").innerHTML = [
    metricCardMarkup("TESTING PROGRAMS", formatNumber(data.program_count), data.file, "✓", "cyan"),
    metricCardMarkup("MODEL STATUS", status.model_trained ? "TRAINED" : "NOT TRAINED", status.model_path || "Training is required before testing", "◎", status.model_trained ? "green" : ""),
    metricCardMarkup("SAVED TEST RUNS", formatNumber(status.testing_executed_count || 0), "source-only programs executed", "▦"),
    metricCardMarkup("TESTING STATUS", status.testing_evaluated ? "EVALUATED" : "NOT EVALUATED", "Actual execution, not training accuracy", "↗", status.testing_evaluated ? "green" : ""),
  ].join("");
  const selector = $("#testing-program-select");
  const selectedIndex = selector.value;
  selector.innerHTML = state.separatedTesting.map((_, index) => {
    const run = state.testingRunStatuses[index] || {};
    const suffix = run.status === "PASS" ? " · PASS" : run.status === "FAIL" ? " · FAIL" : "";
    return `<option value="${index + 1}">Testing program ${String(index + 1).padStart(3, "0")}${suffix}</option>`;
  }).join("");
  if (selectedIndex && Number(selectedIndex) <= state.separatedTesting.length) selector.value = selectedIndex;
  renderSelectedTestingProgram();
  $("#run-selected-testing").disabled = !status.model_trained;
}

function renderSelectedTestingProgram() {
  const index = Number($("#testing-program-select").value) || 1;
  const record = state.separatedTesting[index - 1];
  if (!record) return;
  $("#testing-program-source").textContent = record.source;
  $("#testing-program-record").textContent = JSON.stringify(record, null, 2);
  const run = state.testingRunStatuses[index - 1];
  $("#testing-program-meta").textContent = run
    ? `Testing source ${index} · execution status: ${run.status}${run.selected_passes?.length ? ` · selected passes: ${run.selected_passes.join(", ")}` : ""}`
    : `Testing source ${index} · not evaluated yet`;
}

async function loadSeparatedTraining() {
  try {
    const { data } = await api("/api/separated-datasets/training?limit=500");
    renderSeparatedTraining(data);
  } catch (error) {
    $("#training-dataset-status").textContent = error.message;
    $("#training-dataset-status").classList.add("error");
  }
}

async function loadSeparatedTesting() {
  try {
    const { data } = await api("/api/separated-datasets/testing?limit=100");
    renderSeparatedTesting(data);
  } catch (error) {
    $("#testing-dataset-status").textContent = error.message;
    $("#testing-dataset-status").classList.add("error");
  }
}

async function trainSeparatedModel() {
  const button = $("#train-separated-model");
  button.disabled = true;
  try {
    await api("/api/separated-datasets/train", { method: "POST" });
    await loadSeparatedTraining();
    if (!state.trainingPoll) {
      state.trainingPoll = window.setInterval(async () => {
        await loadSeparatedTraining();
        const buttonText = $("#train-separated-model").textContent;
        if (!buttonText.includes("evaluating")) {
          window.clearInterval(state.trainingPoll);
          state.trainingPoll = null;
          await loadSeparatedTesting();
        }
      }, 2500);
    }
  } catch (error) {
    $("#training-dataset-status").textContent = error.message;
    $("#training-dataset-status").classList.add("error");
    button.disabled = false;
  }
}

async function runSelectedTestingProgram() {
  const button = $("#run-selected-testing");
  const index = Number($("#testing-program-select").value);
  button.disabled = true;
  $("#testing-program-meta").textContent = `Running testing source ${index} through the separated trained model…`;
  try {
    const { data } = await api("/api/separated-datasets/testing/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index }),
    });
    const output = data;
    $("#testing-run-result").hidden = false;
    $("#testing-run-title").textContent = `Testing source ${index}`;
    $("#testing-run-status").textContent = data.status;
    $("#testing-run-status").className = `result-status ${(data.status || "ERROR").toLowerCase()}`;
    $("#testing-run-passes").textContent = `Selected passes: ${(output.selected_passes || []).join(", ") || "none"}`;
    const metrics = output.metrics || {};
    $("#testing-run-metrics").innerHTML = [
      ["INSTRUCTIONS", `${formatNumber(metrics.instructions_before)} → ${formatNumber(metrics.instructions_after)}`],
      ["OPERATIONS", `${formatNumber(metrics.operations_before)} → ${formatNumber(metrics.operations_after)}`],
      ["TEMPORARIES", `${formatNumber(metrics.temporaries_before)} → ${formatNumber(metrics.temporaries_after)}`],
    ].map(([label, value]) => `<div class="result-metric"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></div>`).join("");
    $("#testing-original-output").textContent = output.original_output || "";
    $("#testing-optimized-output").textContent = output.optimized_output || "";
    $("#testing-original-tac").textContent = output.original_tac || "";
    $("#testing-optimized-tac").textContent = output.optimized_tac || "";
    $("#testing-program-meta").textContent = `Testing source ${index} · execution status: ${data.status}`;
    state.testingRunStatuses[index - 1] = {
      program_index: index,
      status: data.status,
      output_equivalent: data.validation?.output_equivalence_confirmed_by_compiler || false,
      selected_passes: data.selected_passes || [],
    };
  } catch (error) {
    $("#testing-program-meta").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

async function loadBenchmarkV2() {
  try {
    const { data } = await api(`/api/benchmark-ml?${filterQuery(state.benchmarkV2Filters, state.benchmarkV2Page, state.benchmarkV2Limit)}`);
    renderBenchmarkV2(data);
  } catch (error) {
    $("#benchmark-v2-warning").hidden = false;
    $("#benchmark-v2-warning").textContent = error.message;
  }
}

function showProgramDetail(id) {
  const program = state.dataset?.programs?.find((item) => item.id === id);
  if (!program) return;
  const execution = program.execution_result || {};
  $("#program-detail-content").innerHTML = `
    <div class="eyebrow">SOURCE PROVENANCE · ${escapeHtml(program.source_type || "")}</div>
    <h2>${escapeHtml(program.id)}</h2>
    <p><b>${escapeHtml(program.source_title || "Untitled source")}</b> · ${escapeHtml(program.author_or_organization || "Author/organization not recorded")}</p>
    <p>Source checked: <b>${program.source_actually_checked ? "Yes" : "No"}</b> · Adapted program executed: <b>${program.adapted_program_executed_successfully ? "Yes" : "No"}</b></p>
    <p>Location: ${escapeHtml(program.repository_file_path || program.chapter_or_location || "Not recorded")}</p>
    <p>License: ${escapeHtml(program.license_information || "Not verified")}</p>
    ${program.source_url ? `<p><a href="${externalHref(program.source_url)}" target="_blank" rel="noopener noreferrer">Open source ↗</a></p>` : ""}
    <h3>ORIGINAL SOURCE CODE</h3><pre>${escapeHtml(program.original_source_code || "Not recorded")}</pre>
    <h3>ADAPTED PROGRAM</h3><pre>${escapeHtml(program.adapted_source_code || "Not recorded")}</pre>
    <h3>ADAPTATION</h3><p>${escapeHtml(program.adaptation_description || "Not recorded")}</p>
    <h3>EXECUTION CHECK</h3><p>${escapeHtml(execution.reason || (execution.can_execute ? "Adapted program parsed and executed; output equivalence checked." : "Execution details not recorded."))}</p>`;
  $("#program-detail").showModal();
}

function metricCardMarkup(label, value, description, icon, color = "") {
  return `<article class="metric-card"><div class="metric-top"><span>${escapeHtml(label)}</span><span class="metric-icon ${color}">${escapeHtml(icon)}</span></div><div class="metric-value">${escapeHtml(value)}</div><div class="metric-note">${escapeHtml(description)}</div></article>`;
}

function renderBenchmarks(data) {
  const warning = $("#benchmark-warning");
  if (!data.available) {
    warning.hidden = false;
    warning.textContent = data.error || "Benchmark reports are unavailable.";
    $("#benchmark-metrics").innerHTML = "";
    $("#benchmark-rows").innerHTML = '<tr><td colspan="5" class="empty-cell">No saved benchmark results.</td></tr>';
    return;
  }
  warning.hidden = true;
  const summary = data.summary || {};
  state.benchmarkSummary = summary;
  state.benchmarkTotal = data.total || 0;
  const average = summary.average_percentage_changes || {};
  $("#benchmark-metrics").innerHTML = [
    metricCardMarkup("PROGRAMS", formatNumber(summary.total_benchmarks), "generated benchmark cases", "▦"),
    metricCardMarkup("OUTPUT CHECKS", `${formatNumber(summary.validated_programs)} / ${formatNumber(summary.total_benchmarks)}`, `${formatNumber(summary.failed_programs)} failures`, "✓", "green"),
    metricCardMarkup("INSTRUCTION REDUCTION", percent(average.instruction_reduction_pct), "suite average · all six passes", "↘", "cyan"),
    metricCardMarkup("CODE SIZE REDUCTION", percent(average.code_size_reduction_pct), "suite average · descriptive", "⌁", "violet"),
  ].join("");
  renderCategories(summary.category_distribution || {});
  renderConvergence(summary.convergence || summary.convergence_statistics);
  $("#benchmark-charts").innerHTML = chartDetails.map(chartCard).join("");
  renderBenchmarkRows(data.results || []);
  const pages = Math.max(1, Math.ceil(state.benchmarkTotal / state.benchmarkLimit));
  $("#benchmark-pagination").textContent = `${formatNumber(state.benchmarkTotal)} records · page ${state.benchmarkPage} of ${pages}`;
  $("#previous-page").disabled = state.benchmarkPage <= 1;
  $("#next-page").disabled = state.benchmarkPage >= pages;
}

function renderCategories(categories) {
  const entries = Object.entries(categories);
  const max = Math.max(1, ...entries.map(([, count]) => Number(count)));
  $("#category-breakdown").innerHTML = entries.length
    ? entries.map(([name, count]) => `<div class="category-row"><span>${escapeHtml(name.replaceAll("_", " "))}</span><b>${formatNumber(count)}</b><div class="category-bar"><i style="width:${Math.max(2, Number(count) / max * 100)}%"></i></div></div>`).join("")
    : '<div class="empty-cell">Category totals are unavailable.</div>';
}

function renderConvergence(convergence) {
  if (!convergence || convergence.programs_measured === undefined) {
    $("#convergence-summary").innerHTML = '<div class="empty-cell">No saved convergence report.</div>';
    return;
  }
  const tiles = [
    ["PROGRAMS MEASURED", formatNumber(convergence.programs_measured)],
    ["FIXED-POINT RATE", percent(convergence.fixed_point_rate_pct)],
    ["MEAN CYCLES", formatNumber(convergence.mean_cycles_to_fixed_point, 2)],
    ["MAX CYCLES", formatNumber(convergence.maximum_cycles_to_fixed_point)],
  ];
  $("#convergence-summary").innerHTML = tiles.map(([label, value]) => `<div class="convergence-tile"><span>${label}</span><b>${escapeHtml(value)}</b></div>`).join("");
}

function renderBenchmarkRows(rows) {
  $("#benchmark-rows").innerHTML = rows.length ? rows.map((row) => {
    const outputOk = row.expected_output_match === true || String(row.expected_output_match).toLowerCase() === "true";
    const optimizerOk = row.verdict === "PASS";
    return `<tr><td>${escapeHtml(row.id)}</td><td>${escapeHtml(String(row.category || "").replaceAll("_", " "))}</td><td>${escapeHtml(row.size || "—")}</td><td><span class="status-chip ${optimizerOk ? "pass" : "fail"}">${escapeHtml(row.verdict || "—")}</span></td><td><span class="status-chip ${outputOk ? "pass" : "fail"}">${outputOk ? "MATCH" : "MISMATCH"}</span></td></tr>`;
  }).join("") : '<tr><td colspan="5" class="empty-cell">No results on this page.</td></tr>';
}

async function loadDataset() {
  await loadGroupedDataset();
  if (!state.dataset?.available) {
    try {
      const { data } = await api("/api/dataset/programs");
      renderDataset(data);
    } catch (error) {
      $("#dataset-warning").hidden = false;
      $("#dataset-warning").textContent = error.message;
    }
  } else {
    renderDatasetRows();
  }
}

async function loadBenchmarks() {
  await loadBenchmarkV2();
  try {
    const { data } = await api(`/api/benchmarks/results?page=${state.benchmarkPage}&limit=${state.benchmarkLimit}`);
    renderBenchmarks(data);
  } catch (error) {
    $("#benchmark-warning").hidden = false;
    $("#benchmark-warning").textContent = error.message;
  }
}

async function startBenchmarks() {
  const button = $("#run-benchmarks");
  const display = $("#benchmark-run-state");
  button.disabled = true;
  display.hidden = false;
  display.classList.remove("failed");
  display.textContent = "Starting the full 500-program pipeline. It regenerates the canonical benchmark records and refreshes reports; no automatic progress percentage is available.";
  try {
    const { data } = await api("/api/benchmarks/run", { method: "POST" });
    display.textContent = data.message || "Benchmark pipeline started.";
    clearInterval(state.benchmarkPoll);
    state.benchmarkPoll = setInterval(pollBenchmarkStatus, 2000);
    await pollBenchmarkStatus();
  } catch (error) {
    display.classList.add("failed");
    display.textContent = error.message;
    button.disabled = false;
  }
}

async function pollBenchmarkStatus() {
  try {
    const { data } = await api("/api/benchmarks/run");
    const display = $("#benchmark-run-state");
    if (data.status === "running") {
      display.hidden = false;
      display.textContent = `500-program pipeline running since ${data.started_at || "now"}. The project runner does not expose progress.`;
      return;
    }
    clearInterval(state.benchmarkPoll);
    state.benchmarkPoll = null;
    $("#run-benchmarks").disabled = false;
    if (data.status === "completed") {
      display.classList.remove("failed");
      display.textContent = `500-program pipeline completed successfully at ${data.completed_at}. Reports and plots have been refreshed.`;
      state.benchmarkPage = 1;
      await loadBenchmarks();
    } else if (data.status === "failed") {
      display.classList.add("failed");
      display.textContent = `Pipeline failed: ${data.error || "No error details were returned."}`;
    }
  } catch (error) {
    const display = $("#benchmark-run-state");
    display.classList.add("failed");
    display.textContent = error.message;
    clearInterval(state.benchmarkPoll);
    $("#run-benchmarks").disabled = false;
  }
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => setPage(button.dataset.page));
});
document.querySelectorAll("[data-go]").forEach((button) => {
  button.addEventListener("click", () => setPage(button.dataset.go));
});
$("#source-code").addEventListener("input", updateEditorStats);
$("#model-select").addEventListener("change", updateModelNote);
$("#run-optimizer").addEventListener("click", runOptimization);
$("#dataset-search").addEventListener("input", (event) => {
  state.datasetQuery = event.target.value;
  renderDatasetRows();
});
$("#dataset-rows").addEventListener("click", (event) => {
  const button = event.target.closest("[data-record]");
  if (button) showProgramDetail(button.dataset.record);
});
for (const [filterId, key] of [
  ["dataset-v2-category", "category"],
  ["dataset-v2-split", "split"],
  ["dataset-v2-pass", "pass"],
  ["dataset-v2-size", "size"],
]) {
  $(`#${filterId}`).addEventListener("change", (event) => {
    state.datasetFilters[key] = event.target.value;
    state.groupedDatasetPage = 1;
    loadGroupedDataset();
  });
}
$("#dataset-v2-prev").addEventListener("click", () => {
  if (state.groupedDatasetPage > 1) {
    state.groupedDatasetPage -= 1;
    loadGroupedDataset();
  }
});
$("#dataset-v2-next").addEventListener("click", () => {
  const total = state.groupedDataset?.total_filtered || 0;
  const limit = state.groupedDataset?.limit || 25;
  if (state.groupedDatasetPage * limit < total) {
    state.groupedDatasetPage += 1;
    loadGroupedDataset();
  }
});
$("#benchmark-v2-filter").addEventListener("click", () => {
  state.benchmarkV2Filters = {
    category: $("#benchmark-v2-category").value,
    split: $("#benchmark-v2-split").value,
    pass: $("#benchmark-v2-pass").value,
    size: $("#benchmark-v2-size").value,
  };
  state.benchmarkV2Page = 1;
  loadBenchmarkV2();
});
$("#benchmark-v2-prev").addEventListener("click", () => {
  if (state.benchmarkV2Page > 1) {
    state.benchmarkV2Page -= 1;
    loadBenchmarkV2();
  }
});
$("#benchmark-v2-next").addEventListener("click", () => {
  if (state.benchmarkV2Page * state.benchmarkV2Limit < state.benchmarkV2Total) {
    state.benchmarkV2Page += 1;
    loadBenchmarkV2();
  }
});
$("#run-benchmarks").addEventListener("click", startBenchmarks);
$("#training-program-select").addEventListener("change", renderSelectedTrainingProgram);
$("#testing-program-select").addEventListener("change", renderSelectedTestingProgram);
$("#train-separated-model").addEventListener("click", trainSeparatedModel);
$("#run-selected-testing").addEventListener("click", runSelectedTestingProgram);
$("#previous-page").addEventListener("click", () => {
  if (state.benchmarkPage > 1) {
    state.benchmarkPage -= 1;
    loadBenchmarks();
  }
});
$("#next-page").addEventListener("click", () => {
  if (state.benchmarkPage * state.benchmarkLimit < state.benchmarkTotal) {
    state.benchmarkPage += 1;
    loadBenchmarks();
  }
});
updateEditorStats();
loadOverview();
