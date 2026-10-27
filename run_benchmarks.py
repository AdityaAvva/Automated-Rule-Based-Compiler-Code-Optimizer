import json, subprocess, sys, csv, re, statistics, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"benchmarks"/"dataset.json"; RESULTS=ROOT/"results"
def parse_metrics(out):
    d={}
    allowed={"instructions_before","instructions_after","operations_before","operations_after",
             "temporaries_before","temporaries_after","code_size_before","code_size_after",
             "optimization_time_microseconds","interpreter_time_before_microseconds","interpreter_time_after_microseconds"}
    for line in out.splitlines():
        if "=" not in line: continue
        k,v=line.split("=",1); k=k.strip(); v=v.strip()
        if (k in allowed or re.fullmatch(r"(constant_folding|constant_propagation|algebraic_simplification|common_subexpression_elimination|dead_code_elimination|strength_reduction)_(transformations|instructions_removed|operations_removed|temporaries_removed|code_characters_removed|output_match)",k)) and re.fullmatch(r"-?\d+",v):
            d[k]=int(v)
    return d
def main():
    exe=ROOT/"compiler_optimizer.exe" if sys.platform=="win32" else ROOT/"compiler_optimizer"
    if not exe.exists():
        print("Compile src/main.cpp first. See README.md");sys.exit(1)
    data=json.loads(DATA.read_text(encoding="utf-8"))
    manifest=json.loads((ROOT/"benchmarks"/"manifest.json").read_text(encoding="utf-8"))
    metadata=manifest.get("program_metadata",[])
    metadata_by_id={item["id"]:item for item in metadata}
    recs=json.loads((RESULTS/"ml_recommendations.json").read_text(encoding="utf-8"))
    # Validate the on-disk benchmark dataset, including its held-out family split.
    expected_categories={"constant_expressions", "algebraic_redundancy", "dead_code", "repeated_expressions", "loop_based"}
    category_sizes=collections.Counter(p.get("category") for p in data)
    split_sizes=collections.Counter((p.get("category"),p.get("split")) for p in metadata)
    if len(data) != 500:
        raise SystemExit(f"Dataset must contain 500 programs; found {len(data)}")
    benchmark_files=[
        path for path in (ROOT/"benchmarks").iterdir()
        if path.is_file() and re.fullmatch(r"CAT[1-5]_\d{3}\.json",path.name)
    ]
    if len(benchmark_files)!=500:
        raise SystemExit(f"Expected 500 individual benchmark JSON files; found {len(benchmark_files)}")
    if len({p.get("id") for p in data}) != len(data):
        raise SystemExit("Dataset contains duplicate program IDs")
    if {p.get("category") for p in data} != expected_categories:
        raise SystemExit(f"Unexpected benchmark categories: {sorted({p.get('category') for p in data})}")
    for category in sorted(expected_categories):
        if category_sizes[category] != 100:
            raise SystemExit(f"{category}: expected exactly 100 programs; got {category_sizes[category]}")
        if split_sizes[(category,"training")] != 80 or split_sizes[(category,"testing")] != 20:
            raise SystemExit(
                f"{category}: expected 80 training and 20 testing programs; "
                f"got {split_sizes[(category,'training')]} training and "
                f"{split_sizes[(category,'testing')]} testing"
            )
    train_families={p["family_id"] for p in metadata if p.get("split")=="training"}
    test_families={p["family_id"] for p in metadata if p.get("split")=="testing"}
    if train_families & test_families:
        raise SystemExit(f"Generation families cross splits: {sorted(train_families & test_families)}")
    if len(metadata_by_id)!=500 or set(metadata_by_id)!={p["id"] for p in data}:
        raise SystemExit("Split/family metadata does not match the benchmark IDs.")
    if any(set(p)!={"id","category","size","source","expected_output","inputs"} for p in data):
        raise SystemExit("Canonical benchmark records must contain exactly six source fields.")
    if any(not p.get("source", "").strip() or "expected_output" not in p for p in data):
        raise SystemExit("Dataset contains an empty source or missing expected output")
    rec_by_id={r["id"]:r for r in recs}
    if set(rec_by_id)!={p["id"] for p in data}:
        raise SystemExit("ML recommendations do not match all 500 benchmark IDs.")
    rows=[]; failures=[]
    optimizer_names=["constant_folding","constant_propagation","algebraic_simplification",
                     "common_subexpression_elimination","dead_code_elimination","strength_reduction"]
    # Evaluate the optimizer itself with every implemented pass enabled. Keep ML recommendations
    # as a separate field so a weak recommender cannot silently disable rules and distort
    # the headline before/after measurements.
    valid_ids={p["id"] for p in data}
    for stale in RESULTS.glob("CAT*.src"):
        if stale.stem not in valid_ids:
            stale.unlink()
    for p in data:
        meta=metadata_by_id[p["id"]]
        # Preserve the model's recommendations for inspection, but use the complete pass
        # suite for the primary optimizer correctness/effectiveness benchmark.
        recommended=[name for name,info in rec_by_id[p["id"]]["recommendations"].items() if info["recommended"]]
        selected=optimizer_names
        sourcefile=RESULTS/(p["id"]+".src")
        sourcefile.write_text(p["source"],encoding="utf-8")
        cp=subprocess.run([str(exe),"--passes",",".join(selected),str(sourcefile)],capture_output=True,text=True)
        metrics=parse_metrics(cp.stdout)
        if cp.returncode != 0:
            failures.append({"id":p["id"],"returncode":cp.returncode,"stderr":cp.stderr,
                             "expected":p["expected_output"].strip(),"original":"","optimized":"",
                             "reason":"compiler/optimizer returned a nonzero exit status"})
        for name in ["constant_folding","constant_propagation","algebraic_simplification","common_subexpression_elimination","dead_code_elimination","strength_reduction"]:
            match=re.search(rf"^{name}=(\d+)$",cp.stdout,re.M)
            metrics[f"{name}_fired_count"]=int(match.group(1)) if match else 0
            transform=re.search(rf"^{name}_transformations=(\d+)$",cp.stdout,re.M)
            metrics[f"{name}_transformations"]=int(transform.group(1)) if transform else 0
        original=re.search(r"Original output:\n(.*?)Optimized output:",cp.stdout,re.S)
        optimized=re.search(r"Optimized output:\n(.*?)Verdict:",cp.stdout,re.S)
        orig=original.group(1).strip() if original else ""
        opt=optimized.group(1).strip() if optimized else ""
        verdict="PASS" if "Verdict: PASS" in cp.stdout and orig==opt else "FAIL"
        expected=p["expected_output"].strip()
        expected_ok=(orig==expected)
        row={"id":p["id"],"category":p["category"],"size":p.get("size","small"),
             "family_id":meta["family_id"],"split":meta["split"],
             "source_characters":len(p["source"]),"passes_used":";".join(selected),
             "ml_recommended_passes":";".join(recommended),
             "verdict":verdict,"expected_output_match":expected_ok,**metrics}
        for metric,before,after in [
            ("instruction_reduction_pct","instructions_before","instructions_after"),
            ("operation_reduction_pct","operations_before","operations_after"),
            ("temporary_reduction_pct","temporaries_before","temporaries_after"),
            ("code_size_reduction_pct","code_size_before","code_size_after"),
            ("interpreter_time_change_pct","interpreter_time_before_microseconds","interpreter_time_after_microseconds")]:
            bval=row.get(before); aval=row.get(after)
            row[metric]=round(100*(bval-aval)/bval,2) if bval else 0.0
        rows.append(row)
        if (verdict!="PASS" or not expected_ok) and not any(f.get("id")==p["id"] for f in failures):
            failures.append({"id":p["id"],"returncode":cp.returncode,"stderr":cp.stderr,
                             "expected":expected,"original":orig,"optimized":opt,
                             "reason":"optimized output mismatch or original output differs from dataset oracle"})
    RESULTS.mkdir(exist_ok=True)
    fields=sorted(set(k for r in rows for k in r))
    with (RESULTS/"benchmark_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (RESULTS/"failures.json").write_text(json.dumps(failures,indent=2),encoding="utf-8")
    effectiveness={}
    for name in optimizer_names:
        key=f"{name}_instructions_removed"
        changed=[r for r in rows if int(r.get(key,0) or 0)>0]
        transform_key=f"{name}_transformations"
        transformed=[r for r in rows if int(r.get(transform_key,0) or 0)>0]
        effectiveness[name]={
            "programs_with_transformations":len(transformed),
            "transformation_coverage_pct":round(100*len(transformed)/len(rows),2) if rows else 0.0,
            "total_transformations_in_isolated_test":sum(int(r.get(transform_key,0) or 0) for r in rows),
            "mean_transformations_per_program":round(statistics.mean(int(r.get(transform_key,0) or 0) for r in rows),3) if rows else 0.0,
            "programs_with_instruction_reduction":len(changed),
            "total_instructions_removed_in_isolated_test":sum(int(r.get(key,0) or 0) for r in rows),
            "total_operations_removed_in_isolated_test":sum(int(r.get(f"{name}_operations_removed",0) or 0) for r in rows),
            "total_temporaries_removed_in_isolated_test":sum(int(r.get(f"{name}_temporaries_removed",0) or 0) for r in rows),
            "total_code_characters_removed_in_isolated_test":sum(int(r.get(f"{name}_code_characters_removed",0) or 0) for r in rows),
            "output_match_count_in_isolated_test":sum(int(r.get(f"{name}_output_match",0) or 0) for r in rows)
        }
    numeric_metrics=["instruction_reduction_pct","operation_reduction_pct","temporary_reduction_pct","code_size_reduction_pct","interpreter_time_change_pct"]
    averages={m:round(statistics.mean(float(r.get(m,0) or 0) for r in rows),2) for m in numeric_metrics}
    category_size_counts=collections.Counter((r["category"],r["size"]) for r in rows)
    summary={"programs":len(rows),"small_programs":sum(r.get("size")=="small" for r in rows),
             "large_programs":sum(r.get("size")=="large" for r in rows),
             "category_size_counts":{f"{category}/{size}":count for (category,size),count in sorted(category_size_counts.items())},
             "optimizer_output_match_pass":sum(r["verdict"]=="PASS" for r in rows),
             "expected_output_match_pass":sum(r["expected_output_match"] for r in rows),
             "failures":len(failures),"average_percentage_changes":averages,
             "per_optimizer_isolated_effectiveness":effectiveness,
             "benchmark_limitations":["The current source language has no runtime input statement; most benchmark values are compile-time constants.","Counted loops are expanded into straight-line statements before optimization, so these tests do not measure optimization of a retained loop control-flow graph.","Consequently, 100% operation/temporary reduction is specific to this static benchmark suite and must not be presented as a general performance claim.","Interpreter timings for tiny programs are noisy."],
             "note":"Headline before/after measurements run all six implemented passes. Per-optimizer effectiveness is measured by applying one pass once to original TAC. Transformation counts are rewrite/substitution events, not the same as instructions removed. Category/size counts reflect the generated records. Percentage changes are descriptive, not a performance guarantee."}
    (RESULTS/"optimizer_effectiveness.json").write_text(json.dumps(effectiveness,indent=2),encoding="utf-8")
    (RESULTS/"benchmark_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    if failures: print("See results/failures.json for failed cases.")
if __name__=="__main__": main()
