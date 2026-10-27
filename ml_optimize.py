import argparse, subprocess, sys
from pathlib import Path
import re
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"ml"))
from train_model import features, PASSES, MODEL
from benchmark_features import extract_features

def _original_tac(source):
    exe=ROOT/("compiler_optimizer.exe" if sys.platform=="win32" else "compiler_optimizer")
    if not exe.is_file():
        raise FileNotFoundError(f"C++ optimizer not found: {exe}. Compile src/main.cpp first.")
    try:
        result=subprocess.run(
            [str(exe),"--passes",""],
            input=source,text=True,capture_output=True,timeout=15,check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Compiler exceeded its timeout while preparing ML features.") from exc
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Compiler rejected source before feature extraction.")
    match=re.search(r"=== ORIGINAL TAC ===\n(.*?)=== OPTIMIZED TAC ===",result.stdout,re.S)
    if not match:
        raise ValueError("Compiler did not return original TAC for feature extraction.")
    return match.group(1)

def recommend_passes(source, threshold=None, model_path=None, original_tac=None):
    selected_model = Path(model_path).expanduser() if model_path else MODEL
    if not selected_model.is_absolute():
        selected_model = Path.cwd() / selected_model
    if not selected_model.exists():
        raise FileNotFoundError(
            f"ML model not found: {selected_model}. Run the appropriate training pipeline first."
        )
    bundle=joblib.load(selected_model)
    if bundle.get("feature_schema") in (
        "grouped_benchmark_v1",
        "separated_training_v1",
    ):
        tac=original_tac if original_tac is not None else _original_tac(source)
        values=extract_features(source,tac)
        if len(values) != len(bundle.get("feature_names", [])):
            raise ValueError("Grouped benchmark model feature schema does not match the extractor.")
    else:
        values=features(source)
    x=np.array(values,dtype=float).reshape(1,-1)
    selected=[]; details=[]
    for name in bundle["passes"]:
        model=bundle["models"][name]
        effective_threshold = (
            threshold if threshold is not None
            else bundle.get("thresholds",{}).get(name,0.50)
        )
        if isinstance(model,dict):
            pred=int(model["constant"]); score=float(pred)
        else:
            probabilities=model.predict_proba(x)[0] if hasattr(model,"predict_proba") else None
            classes=list(model.classes_) if hasattr(model,"classes_") else []
            score=float(probabilities[classes.index(1)]) if probabilities is not None and 1 in classes else float(model.predict(x)[0])
            pred=int(score >= effective_threshold) if "thresholds" in bundle else int(model.predict(x)[0])
        selected_by_threshold = score >= effective_threshold
        details.append({"name":name,"recommended":bool(pred),"score":score,
                        "threshold":effective_threshold,
                        "selected":bool(pred and selected_by_threshold)})
        if pred and selected_by_threshold:
            selected.append(name)
    return selected,details

def main():
    parser=argparse.ArgumentParser(description="ML-guided C++ compiler optimizer")
    parser.add_argument("source",help="Source file in the supported educational language")
    parser.add_argument("--executable",default=None,help="Path to compiled C++ optimizer")
    parser.add_argument("--threshold",type=float,default=None,help="Override the validation-selected threshold")
    parser.add_argument("--model",default=None,help="Optional model bundle; defaults to the legacy synthetic benchmark model")
    args=parser.parse_args()
    exe=Path(args.executable) if args.executable else ROOT/("compiler_optimizer.exe" if sys.platform=="win32" else "compiler_optimizer")
    if not exe.exists():
        raise SystemExit(f"C++ optimizer not found: {exe}. Run python run_project.py first.")
    source_path=Path(args.source)
    try:
        source=source_path.read_text(encoding="utf-8")
        selected,details=recommend_passes(source,args.threshold,args.model)
    except (OSError, ValueError, EOFError, ImportError, KeyError) as exc:
        raise SystemExit(f"Unable to read source or load the ML recommendation model: {exc}") from exc
    print("=== ML PASS RECOMMENDATIONS ===")
    for item in details:
        print(f"{item['name']}: {'SELECTED' if item['selected'] else 'not selected'} "
              f"(prediction={'yes' if item['recommended'] else 'no'}, score={item['score']:.3f})")
    if not selected:
        print("No pass passed the threshold; rule optimizer will make no transformations for this input.")
    print("Enabled passes:", ",".join(selected) if selected else "(none)")
    sys.stdout.flush()
    cmd=[str(exe),"--passes",",".join(selected),str(source_path)]
    result=subprocess.run(cmd,text=True)
    raise SystemExit(result.returncode)

if __name__=="__main__": main()
