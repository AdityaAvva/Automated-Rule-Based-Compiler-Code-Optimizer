import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run(args):
    print("\n>>>", " ".join(map(str,args)))
    p=subprocess.run(args,cwd=ROOT)
    if p.returncode: raise SystemExit(p.returncode)
def main():
    exe=ROOT/("compiler_optimizer.exe" if sys.platform=="win32" else "compiler_optimizer")
    compiler="g++"
    run([compiler,"-std=c++17","-O2","-Wall",str(ROOT/"src"/"main.cpp"),"-o",str(exe)])
    run([sys.executable,str(ROOT/"generate_dataset.py")])
    run([sys.executable,str(ROOT/"run_separated_ml_pipeline.py")])
    run([sys.executable,str(ROOT/"run_benchmark_ml.py"),"--skip-generation"])
    run([sys.executable,str(ROOT/"ml"/"train_model.py")])
    run([sys.executable,str(ROOT/"run_benchmarks.py")])
    run([sys.executable,str(ROOT/"make_plots.py")])
    print("\nProject pipeline completed. Inspect results/failures.json, benchmark_summary.json, and plots before reporting.")
if __name__=="__main__": main()
