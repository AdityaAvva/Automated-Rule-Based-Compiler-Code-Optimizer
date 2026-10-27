import csv, json, re, subprocess, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RES, OUT = ROOT / "results", ROOT / "plots"
OUT.mkdir(exist_ok=True)
# Remove obsolete charts from older project versions so users do not mistake them
# for plots generated from the current benchmark run.
for stale_name in ("code_size_vs_time.png", "instruction_count_by_category.png",
                   "instruction_reduction_by_category.png"):
    stale = OUT / stale_name
    if stale.exists():
        stale.unlink()
CSV = RES / "benchmark_results.csv"
if not CSV.exists():
    raise SystemExit("Run python run_benchmarks.py first.")
with CSV.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("benchmark_results.csv contains no rows.")

def val(r, key):
    try: return float(r.get(key, 0) or 0)
    except (ValueError, TypeError): return 0.0

passes = [
 ("constant_folding", "Constant folding"),
 ("constant_propagation", "Constant propagation"),
 ("algebraic_simplification", "Algebraic simplification"),
 ("common_subexpression_elimination", "Common subexpression elimination"),
 ("dead_code_elimination", "Dead-code elimination"),
 ("strength_reduction", "Strength reduction"),
]
cat_names = {
 "algebraic_redundancy":"Algebraic redundancy",
 "constant_expressions":"Constant expressions",
 "dead_code":"Dead code",
 "loop_based":"Loop-based",
 "repeated_expressions":"Repeated expressions",
}
cats = sorted(set(r["category"] for r in rows))
labels = [cat_names.get(c, c.replace("_", " ").title()) for c in cats]
plt.rcParams.update({
 "font.family":"DejaVu Sans", "font.size":10, "axes.titlesize":15,
 "axes.labelsize":11, "legend.fontsize":9, "figure.facecolor":"white",
 "axes.facecolor":"#f8fafc", "axes.edgecolor":"#cbd5e1",
 "axes.grid":True, "grid.color":"#e2e8f0", "grid.linewidth":.7,
 "axes.axisbelow":True, "savefig.facecolor":"white"
})
BLUE="#2563eb"; ORANGE="#f97316"; GREEN="#16a34a"; PURPLE="#7c3aed"; RED="#dc2626"; TEAL="#0891b2"; GREY="#64748b"
def save(name):
    plt.tight_layout()
    plt.savefig(OUT/name, dpi=190, bbox_inches="tight")
    plt.close()

# 1: category comparison, annotated and human-readable
before=np.array([np.mean([val(r,"instructions_before") for r in rows if r["category"]==c]) for c in cats])
after=np.array([np.mean([val(r,"instructions_after") for r in rows if r["category"]==c]) for c in cats])
x=np.arange(len(cats)); w=.34
fig,ax=plt.subplots(figsize=(11,6))
b1=ax.bar(x-w/2,before,w,label="Before optimization",color=BLUE,zorder=3)
b2=ax.bar(x+w/2,after,w,label="After optimization",color=ORANGE,zorder=3)
ax.set_xticks(x,labels,rotation=12,ha="right")
ax.set_ylabel("Mean TAC instructions per program")
ax.set_title("Instruction Count Before and After Optimization",loc="left",weight="bold",pad=15)
ax.legend(frameon=False,ncol=2,loc="upper right")
ax.bar_label(b1,fmt="%.1f",padding=3,fontsize=8)
ax.bar_label(b2,fmt="%.1f",padding=3,fontsize=8)
ax.set_ylim(0,max(max(before),max(after))*1.18)
save("01_instruction_count_before_after.png")

# Actual fired-pass counts (now parsed correctly), not ML recommendations
firecols=[p[0]+"_fired_count" for p in passes]
has_fire=all(c in rows[0] for c in firecols) and sum(val(r,c) for r in rows for c in firecols)>0
# 2: stacked chart by category (readable) of pass applications, not 120 tiny labels
fig,ax=plt.subplots(figsize=(11,6))
bottom=np.zeros(len(cats))
if has_fire:
    for (pname,plabel),col,color in zip(passes,firecols,[BLUE,ORANGE,GREEN,RED,PURPLE,TEAL]):
        vals=np.array([sum(val(r,col) for r in rows if r["category"]==cat) for cat in cats])
        ax.bar(np.arange(len(cats)),vals,bottom=bottom,label=plabel,color=color,width=.62)
        bottom+=vals
    ax.set_ylabel("Total transformation events")
    ax.set_title("Optimizer Transformation Events by Category",loc="left",weight="bold",pad=15)
else:
    # Explicitly visualise per-pass isolated instruction wins, rather than claiming a pass fired.
    for (pname,plabel),color in zip(passes,[BLUE,ORANGE,GREEN,RED,PURPLE,TEAL]):
        vals=np.array([sum(max(0,val(r,pname+"_instructions_removed")) for r in rows if r["category"]==cat) for cat in cats])
        ax.bar(np.arange(len(cats)),vals,bottom=bottom,label=plabel,color=color,width=.62)
        bottom+=vals
    ax.set_ylabel("Total instructions removed in isolated tests")
    ax.set_title("Isolated Optimization Impact by Category",loc="left",weight="bold",pad=15)
ax.set_xticks(np.arange(len(cats)),labels,rotation=12,ha="right")
ax.legend(frameon=False,ncol=2,loc="upper left")
save("02_optimizer_passes_by_program_stacked.png")

# 3: true convergence over repeated complete optimizer cycles.
# Each benchmark uses the pass set selected for that program by the ML model.
exe=ROOT/("compiler_optimizer.exe" if sys.platform=="win32" else "compiler_optimizer")
trace_rows=[]
max_points=0
if exe.exists():
    for r in rows:
        src=RES/(r["id"]+".src")
        if not src.exists():
            continue
        selected=(r.get("passes_used","") or r.get("ml_recommended_passes","") or "").strip().replace(";",",")
        cp=subprocess.run([str(exe),"--passes",selected,"--trace-convergence",str(src)],capture_output=True,text=True)
        points=[]
        for match in re.finditer(r"^convergence_iteration=(\d+),instructions=(\d+),stable=([01])$",cp.stdout,re.M):
            points.append({"iteration":int(match.group(1)),"instructions":int(match.group(2)),"stable":match.group(3)=="1"})
        if points:
            trace_rows.append({"id":r["id"],"category":r["category"],"points":points,
                               "fixed_point_reached":any(p["stable"] for p in points[1:])})
            max_points=max(max_points,len(points))
if trace_rows:
    # Once an individual program reaches a fixed point, carry its final count forward
    # so every iteration's mean is computed over the same benchmark population.
    trace_matrix=[]
    for item in trace_rows:
        values=[p["instructions"] for p in item["points"]]
        values.extend([values[-1]]*(max_points-len(values)))
        trace_matrix.append(values)
    arr=np.asarray(trace_matrix,dtype=float)
    means=np.mean(arr,axis=0)
    stds=np.std(arr,axis=0)
    xx=np.arange(max_points)
    fig,ax=plt.subplots(figsize=(11,6))
    ax.plot(xx,means,color=BLUE,marker="o",linewidth=2.8,markersize=5,label="Mean TAC instruction count")
    ax.fill_between(xx, np.maximum(0,means-stds), means+stds, color=BLUE,alpha=.13,label="±1 standard deviation")
    ax.set_ylim(bottom=0)
    tick_ids=sorted(set([0,1,2,3,5,10,15,max_points-2,max_points-1]))
    tick_ids=[i for i in tick_ids if 0 <= i < max_points]
    ax.set_xticks(tick_ids,["Initial" if i==0 else str(i) for i in tick_ids])
    ax.set_xlabel("Complete optimization cycle (0 = initial TAC)")
    ax.set_ylabel("Mean TAC instructions across benchmark programs")
    fixed_count=sum(item["fixed_point_reached"] for item in trace_rows)
    ax.set_title("Optimization Convergence to a Fixed Point",loc="left",weight="bold",pad=15)
    ax.text(.99,.97,f"{len(trace_rows)} programs · fixed point reached: {fixed_count}/{len(trace_rows)}",
            transform=ax.transAxes,ha="right",va="top",fontsize=9,color=GREY,
            bbox={"boxstyle":"round,pad=.35","facecolor":"white","edgecolor":"#cbd5e1"})
    ax.legend(frameon=False,loc="upper right",bbox_to_anchor=(1,.89))
    annotation_ids=sorted(set([0,1,2,3,5,10,15,max_points-2,max_points-1]))
    for i in annotation_ids:
        v=means[i]
        ax.annotate(f"{v:.1f}",(i,v),xytext=(0,9),textcoords="offset points",ha="center",fontsize=8)
    save("03_optimization_convergence_curve.png")
    # Count completed cycles up to the first stable cycle (not the initial TAC point).
    for item in trace_rows:
        stable_points=[p["iteration"] for p in item["points"] if p["stable"]]
        item["cycles_to_fixed_point"] = stable_points[0] if stable_points else None
    cycle_values=[item["cycles_to_fixed_point"] for item in trace_rows if item["cycles_to_fixed_point"] is not None]
    cycle_stats={
        "programs_measured":len(trace_rows),
        "programs_reaching_fixed_point":fixed_count,
        "fixed_point_rate_pct":round(100*fixed_count/len(trace_rows),2) if trace_rows else 0,
        "mean_cycles_to_fixed_point":round(float(np.mean(cycle_values)),3) if cycle_values else None,
        "median_cycles_to_fixed_point":float(np.median(cycle_values)) if cycle_values else None,
        "maximum_cycles_to_fixed_point":int(max(cycle_values)) if cycle_values else None,
        "minimum_cycles_to_fixed_point":int(min(cycle_values)) if cycle_values else None,
        "cycles_to_fixed_point_by_category":{}
    }
    for category in sorted(set(item["category"] for item in trace_rows)):
        values=[item["cycles_to_fixed_point"] for item in trace_rows if item["category"]==category and item["cycles_to_fixed_point"] is not None]
        cycle_stats["cycles_to_fixed_point_by_category"][category]={
            "programs":len(values),"mean":round(float(np.mean(values)),3) if values else None,
            "median":float(np.median(values)) if values else None,"maximum":int(max(values)) if values else None
        }
    trace_summary={
        "method":"Repeated complete optimization cycles using the full pass set recorded for each benchmark",
        "x_axis":"Initial TAC, then instruction count after each full pass cycle",
        **cycle_stats,
        "max_cycles_plotted":max_points-1,
        "fixed_point_definition":"A program reaches a fixed point when rendered TAC is unchanged after a complete optimization cycle. The initial TAC point is not counted as a cycle.",
        "mean_instruction_count_by_iteration":[round(float(v),4) for v in means],
        "std_instruction_count_by_iteration":[round(float(v),4) for v in stds],
        "per_program":[{"id":item["id"],"category":item["category"],"cycles_to_fixed_point":item["cycles_to_fixed_point"],
                        "instruction_counts":[p["instructions"] for p in item["points"]],
                        "stable_flags":[p["stable"] for p in item["points"]],
                        "fixed_point_reached":item["fixed_point_reached"]} for item in trace_rows]
    }
    (RES/"convergence_trace.json").write_text(json.dumps(trace_summary,indent=2),encoding="utf-8")

    # Plot the distribution of cycles needed to reach the fixed point. This makes
    # the long tail visible instead of hiding it behind the mean convergence curve.
    fig,ax=plt.subplots(figsize=(10,5.5))
    if cycle_values:
        bins=np.arange(min(cycle_values)-0.5,max(cycle_values)+1.5,1)
        ax.hist(cycle_values,bins=bins,color=BLUE,edgecolor="white",rwidth=.88)
        ax.axvline(float(np.median(cycle_values)),color=ORANGE,linestyle="--",linewidth=2,
                   label=f"Median: {np.median(cycle_values):.1f} cycles")
        ax.axvline(max(cycle_values),color=RED,linestyle=":",linewidth=2,
                   label=f"Maximum: {max(cycle_values)} cycles")
        ax.set_xlabel("Complete cycles until first stable cycle")
        ax.set_ylabel("Number of benchmark programs")
        ax.legend(frameon=False)
        ax.set_title("Distribution of Optimization Convergence Cycles",loc="left",weight="bold",pad=15)
    else:
        ax.text(.5,.5,"No programs reached a detected fixed point",ha="center",va="center")
        ax.axis("off")
    save("07_convergence_cycle_distribution.png")

    # Add convergence metrics to the main summary so they are reported beside
    # benchmark correctness/effectiveness metrics.
    summary_path=RES/"benchmark_summary.json"
    if summary_path.exists():
        summary_data=json.loads(summary_path.read_text(encoding="utf-8"))
        summary_data["convergence_statistics"]={k:v for k,v in cycle_stats.items()}
        summary_data.setdefault("metric_definitions",{}).update({
            "transformation_events":"Number of individual rewrites/substitutions performed by an optimizer rule; not pass invocations and not necessarily instructions removed.",
            "pass_invocations":"Number of times a complete rule/pass function is called. The current CSV does not record this separately.",
            "cycles_to_fixed_point":"Number of complete optimization cycles up to and including the first cycle whose rendered TAC is unchanged. Initial TAC is cycle 0."
        })
        summary_path.write_text(json.dumps(summary_data,indent=2),encoding="utf-8")
else:
    fig,ax=plt.subplots(figsize=(9,5)); ax.text(.5,.5,"Convergence data unavailable: compiler/source files missing",ha="center",va="center");ax.axis("off");save("03_optimization_convergence_curve.png")

# 4: distribution based on actual firing counts; fallback is explicitly isolated measured impact.
if has_fire:
    totals=[sum(val(r,col) for r in rows) for col in firecols]
    title="Share of Actual Optimizer Firings"
else:
    totals=[sum(max(0,val(r,p[0]+"_instructions_removed")) for r in rows) for p in passes]
    title="Share of Instructions Removed in Isolated Pass Tests"
nonzero=[(passes[i][1],v) for i,v in enumerate(totals) if v>0]
fig,ax=plt.subplots(figsize=(8,7))
if nonzero:
    wedges,_,autotexts=ax.pie([v for _,v in nonzero],labels=None,autopct="%1.1f%%",startangle=90,
      pctdistance=.72,wedgeprops={"width":.42,"edgecolor":"white","linewidth":2},
      colors=[BLUE,ORANGE,GREEN,RED,PURPLE,TEAL][:len(nonzero)])
    ax.legend(wedges,[f"{n} ({v:.0f})" for n,v in nonzero],loc="center left",bbox_to_anchor=(.98,.5),frameon=False)
    ax.text(0,.06,f"{sum(v for _,v in nonzero):.0f}",ha="center",va="center",fontsize=20,weight="bold")
    ax.text(0,-.10,"total",ha="center",va="center",fontsize=9,color=GREY)
else:
    ax.text(.5,.5,"No positive optimization impact recorded",ha="center",va="center")
ax.set_title(title,loc="left",weight="bold",pad=18)
save("04_optimization_type_distribution.png")

# 5: code size versus interpreter time; exclude zero timings from log-safe display.
fig,ax=plt.subplots(figsize=(10,6))
for key,sizekey,color,marker,label in [
 ("interpreter_time_before_microseconds","code_size_before",BLUE,"o","Before"),
 ("interpreter_time_after_microseconds","code_size_after",ORANGE,"X","After")]:
    xs=np.array([val(r,sizekey) for r in rows])
    ys=np.array([val(r,key) for r in rows])
    valid=(xs>=0)&(ys>0)
    ax.scatter(xs[valid],ys[valid],s=42,c=color,marker=marker,label=label,alpha=.72,
               edgecolors="white",linewidths=.5)
ax.set_xlabel("TAC code size (characters)")
ax.set_ylabel("Interpreter execution time (µs per run; zero values omitted)")
ax.set_title("Code Size vs. Interpreter Execution Time",loc="left",weight="bold",pad=15)
ax.legend(frameon=False)
save("05_code_size_vs_interpreter_time.png")

# 6: count transformations performed by each isolated pass. This correctly
# represents constant propagation substitutions and strength-reduction rewrites,
# which can be useful even when arithmetic-operation count does not decrease.
heat=np.zeros((len(passes),len(cats)))
for i,(pname,_) in enumerate(passes):
    key=pname+"_transformations"
    for j,cat in enumerate(cats):
        subset=[r for r in rows if r["category"]==cat]
        heat[i,j]=np.mean([val(r,key) for r in subset]) if subset else 0
fig,ax=plt.subplots(figsize=(11,6.5))
im=ax.imshow(heat,aspect="auto",cmap="YlGnBu",vmin=0,vmax=max(1,float(heat.max())))
ax.set_xticks(np.arange(len(cats)),labels,rotation=12,ha="right")
ax.set_yticks(np.arange(len(passes)),[p[1] for p in passes])
ax.set_title("Optimizer Transformations by Pass and Category",loc="left",weight="bold",pad=15)
cb=fig.colorbar(im,ax=ax,pad=.025);cb.set_label("Mean transformations per program (isolated pass)")
for i in range(len(passes)):
    for j in range(len(cats)):
        color="white" if heat[i,j] > heat.max()*.55 else "#0f172a"
        ax.text(j,i,f"{heat[i,j]:.1f}",ha="center",va="center",fontsize=9,color=color)
save("06_optimizer_category_heatmap.png")

print(f"Saved seven redesigned visualizations to {OUT}")
