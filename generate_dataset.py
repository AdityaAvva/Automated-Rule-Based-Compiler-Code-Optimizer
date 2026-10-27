import hashlib
import json, random, re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"benchmarks"
SEED=307
random.seed(SEED)
def expected_eval(src):
    # Deliberately small evaluator for the benchmark language, not arbitrary Python execution.
    env={}; output=[]
    # Expand the supported for-loop form in the same way as the C++ prototype.
    pat=re.compile(r'for\s*\(\s*(\w+)\s*=\s*(-?\d+)\s*;\s*\1\s*<\s*(\d+)\s*;\s*\1\s*=\s*\1\s*\+\s*1\s*\)\s*\{([^{}]*)\}')
    while True:
        m=pat.search(src)
        if not m: break
        var,start,end,body=m.group(1),int(m.group(2)),int(m.group(3)),m.group(4)
        exp="".join(f"{var}={i};{body};" for i in range(start,min(end,start+1000)))
        src=src[:m.start()]+exp+src[m.end():]
    for st in [x.strip() for x in src.replace("\n","").split(";") if x.strip()]:
        if st.startswith("print"):
            x=st[5:].strip()
            if x.startswith("(") and x.endswith(")"): x=x[1:-1]
            output.append(eval_expr(x,env))
        else:
            lhs,rhs=st.split("=",1); env[lhs.strip()]=eval_expr(rhs.strip(),env)
    return "\n".join(
        format(value, ".12g") if isinstance(value, float) else str(value)
        for value in output
    )+("\n" if output else "")
def eval_expr(expr,env):
    # expression grammar restricted to arithmetic and identifiers
    tokens=re.findall(r'\d+(?:\.\d+)?|[A-Za-z_]\w*|[()+\-*/%]',expr)
    pos=0
    def atom():
        nonlocal pos
        if pos>=len(tokens): raise ValueError("missing atom")
        t=tokens[pos];pos+=1
        if t=="(":
            v=add()
            if tokens[pos]!=")": raise ValueError("missing )")
            pos+=1;return v
        if t=="-": return -atom()
        if re.fullmatch(r'\d+(?:\.\d+)?',t): return float(t) if "." in t else int(t)
        return env[t]
    def mul():
        nonlocal pos
        v=atom()
        while pos<len(tokens) and tokens[pos] in ("*","/","%"):
            op=tokens[pos];pos+=1;b=atom()
            v={"*":lambda:v*b,"/":lambda:v/b,"%":lambda:v%b}[op]()
        return v
    def add():
        nonlocal pos
        v=mul()
        while pos<len(tokens) and tokens[pos] in ("+","-"):
            op=tokens[pos];pos+=1;b=mul();v=v+b if op=="+" else v-b
        return v
    val=add()
    return int(val) if isinstance(val,float) and val.is_integer() else val

programs=[]

def additional_program(category, family, variant, rng):
    """Build one unique member of a category-specific template family."""
    salt=f"{family}_{variant}"
    a=rng.randint(4, 41)+variant
    b=rng.randint(2, 17)+family
    c=rng.randint(3, 13)+variant%5
    d=rng.randint(1, 9)+family%4
    size=("small","small","medium","medium","medium","medium","large","large","large","large")[variant]
    statements=[]

    if category=="constant_expressions":
        if family==0:
            statements=[f"first{salt} = ({a} + {b}) * {c};",
                        f"answer{salt} = first{salt} - {d};"]
        elif family==1:
            statements=[f"part{salt} = {a} * {b};",
                        f"answer{salt} = part{salt} + {c} * {d};"]
        elif family==2:
            statements=[f"answer{salt} = ({a} + {b}) * ({c} + {d});"]
        elif family==3:
            statements=[f"part{salt} = ({a} - {d}) * {b};",
                        f"answer{salt} = part{salt} % {c+1};"]
        elif family==4:
            statements=[f"first{salt} = {a} * {b} + {c};",
                        f"answer{salt} = first{salt} / {d+1};"]
        else:
            statements=[f"first{salt} = ({a} + {c}) * {b};",
                        f"second{salt} = first{salt} - {d};",
                        f"answer{salt} = second{salt} + {variant+1};"]
        if size=="large":
            for index in range(variant%4+2):
                statements.extend((f"stage{index}_{salt} = ({a+index} * {b}) + {c+index};",
                                   f"print(stage{index}_{salt});"))
        statements.append(f"print(answer{salt});")
    elif category=="algebraic_redundancy":
        identities=("value + 0","0 + value","value - 0","value * 1","1 * value","value / 1")
        count={"small":1,"medium":3,"large":6}[size]
        statements.append(f"value{salt} = {a};")
        for index in range(count):
            expression=identities[family].replace("value",f"value{salt}")
            if index:
                expression=f"({expression}) + {index}"
            if index>=2:
                expression=f"({expression}) * 2"
            target=f"result{index}_{salt}"
            statements.extend((f"{target} = {expression};",f"print({target});"))
    elif category=="dead_code":
        count={"small":2,"medium":5,"large":9}[size]
        for index in range(count):
            statements.extend((f"discard{index}_{salt} = {a+index} * {b+index};",
                               f"discarded{index}_{salt} = discard{index}_{salt} + {c+index};"))
        if family%2==0:
            statements.extend((f"live{salt} = {a};",f"live{salt} = {b+c+variant};"))
            final=f"live{salt}"
        else:
            statements.extend((f"live{salt} = {a} + {b};",
                               f"result{salt} = live{salt} + {c};"))
            final=f"result{salt}"
        statements.append(f"print({final});")
    elif category=="repeated_expressions":
        left,right=f"left{salt}",f"right{salt}"
        operator="+" if family%2==0 else "*"
        statements.extend((f"{left} = {a};",f"{right} = {b};"))
        count={"small":2,"medium":4,"large":7}[size]
        for index in range(count):
            name=f"repeat{index}_{salt}"
            statements.extend((f"{name} = {left} {operator} {right};",f"print({name});"))
        statements.extend((f"combined{salt} = {left} {operator} {right} + {c};",
                           f"print(combined{salt});"))
    elif category=="loop_based":
        index_name=f"index{salt}"
        total_name=f"total{salt}"
        end=3+(variant%7) if size=="small" else 6+(variant%7)
        coefficient=2+(family%5)
        statements.extend((f"{total_name} = {variant+1};",
                           f"for({index_name}=0;{index_name}<{end};"
                           f"{index_name}={index_name}+1){{ "
                           f"{total_name} = {total_name} + {index_name} * {coefficient}; }}"))
        if size=="large":
            statements.extend((f"extra{salt} = {a} * 2;",
                               f"extra2{salt} = extra{salt} + {total_name};",
                               f"print(extra2{salt});"))
        statements.append(f"print({total_name});")
    else:
        raise ValueError(f"Unknown benchmark category: {category}")

    return size," ".join(statements)

# The legacy seed cohort contains 40 programs per category. The generated
# expansion below adds 60 per category to reach the active 500-program suite.
# All programs use only syntax supported by the educational C++ compiler.

# CAT1: Constant expressions
for i in range(20):
    a=5+i; b=2+(i%7); c=3+(i%5)
    src=f"x = {a} * {b}; print(x); y = x + {c}; print(y); z = y - 1; print(z);"
    programs.append(("CAT1","constant_expressions","small",src))
for i in range(20):
    stmts=[]
    for j in range(8 + (i%5)*2):
        a=10+i+j; b=2+(j%9); c=1+(i%7)
        stmts.append(f"c{j} = {a} * {b};")
        stmts.append(f"d{j} = c{j} + {c};")
        stmts.append(f"print(d{j});")
    stmts += [f"result = d{(8 + (i%5)*2)-1} - {i%4};", "print(result);"]
    programs.append(("CAT1","constant_expressions","large"," ".join(stmts)))

# CAT2: Algebraic redundancy (x+0, x*1, x-0, x*2)
for i in range(20):
    v=f"v{i}"
    src=f"{v} = {i+3}; x = {v} + 0; print(x); y = x * 1; print(y); z = y - 0; print(z); q = z * 2; print(q);"
    programs.append(("CAT2","algebraic_redundancy","small",src))
for i in range(20):
    stmts=[f"base = {i+10};"]
    for j in range(8 + (i%5)*2):
        stmts += [f"v{j} = base + 0;", f"w{j} = v{j} * 1;", f"u{j} = w{j} - 0;", f"q{j} = u{j} * 2;", f"print(q{j});"]
    stmts.append(f"print(q{j});")
    programs.append(("CAT2","algebraic_redundancy","large"," ".join(stmts)))

# CAT3: Dead assignments before the observable result
for i in range(20):
    src=f"unused{i} = {i+11}; also_unused{i} = {i+20} * 3; answer = {i+7}; print(answer);"
    programs.append(("CAT3","dead_code","small",src))
for i in range(20):
    stmts=[]
    for j in range(12 + (i%5)*3):
        stmts.append(f"unused{j} = {j+i+2} * {j+3};")
        stmts.append(f"discard{j} = unused{j} + {i+1};")
    stmts += [f"answer = {i+99};", "print(answer);"]
    programs.append(("CAT3","dead_code","large"," ".join(stmts)))

# CAT4: Repeated expressions; operands stay unchanged so CSE is valid
for i in range(20):
    a,b,c,d=f"a{i}",f"b{i}",f"c{i}",f"d{i}"
    e1,e2,e3=f"exprA{i}",f"exprB{i}",f"exprC{i}"
    src=f"{a} = {i+2}; {b} = {i+4}; {c} = {i+6}; {d} = {i+8}; {e1} = {a} + {b}; print({e1}); {e2} = {c} * {d}; print({e2}); {e3} = {a} + {b}; print({e3});"
    programs.append(("CAT4","repeated_expressions","small",src))
for i in range(20):
    stmts=[f"a = {7+i};", "b = 9;", "c = 3;", "d = 5;"]
    n=8+(i%5)*2
    for j in range(n):
        stmts += [f"r{j} = a + b;", f"print(r{j});", f"s{j} = c * d;", f"print(s{j});",
                   f"exprA{j} = a + b;", f"print(exprA{j});", f"exprB{j} = c * d;", f"print(exprB{j});"]
    stmts.append(f"print(exprA{n-1});")
    programs.append(("CAT4","repeated_expressions","large"," ".join(stmts)))

# CAT5: Counted loops, with additional work after the loop in large cases
for i in range(20):
    n=3+(i%7)
    src=f"sum{i} = 0; for(index{i}=0;index{i}<{n};index{i}=index{i}+1){{ sum{i} = sum{i} + index{i} * 2; print(sum{i}); }} print(sum{i});"
    programs.append(("CAT5","loop_based","small",src))
for i in range(20):
    n=10+(i%10)
    stmts=[f"sum{i} = 0; for(index{i}=0;index{i}<{n};index{i}=index{i}+1){{ sum{i} = sum{i} + index{i} * 2; print(sum{i}); }}"]
    for j in range(10+(i%5)*3):
        stmts.append(f"extra{i}_{j} = {j+1} * 4;")
    stmts.append(f"print(sum{i});")
    programs.append(("CAT5","loop_based","large"," ".join(stmts)))

OUT.mkdir(exist_ok=True)
category_prefixes={
    "constant_expressions":"CAT1",
    "algebraic_redundancy":"CAT2",
    "dead_code":"CAT3",
    "repeated_expressions":"CAT4",
    "loop_based":"CAT5",
}
category_names=tuple(category_prefixes)

def archive_legacy_category_records():
    """Preserve the older common-subexpression examples without enriched rows."""
    source_paths={
        split:ROOT/"datasets"/split/"common_subexpressions"
        for split in ("training","testing")
    }
    if not any(path.exists() for path in source_paths.values()):
        return
    archive_root=ROOT/"datasets"/"archive"/"legacy_common_subexpressions"
    if archive_root.exists():
        archive_manifest=json.loads((archive_root/"manifest.json").read_text(encoding="utf-8"))
        if archive_manifest.get("program_count")!=100:
            raise ValueError(f"Existing legacy archive is incomplete: {archive_root}")
        for split,directory in source_paths.items():
            for path in directory.glob("*.json"):
                record=json.loads(path.read_text(encoding="utf-8"))
                if split=="training":
                    compact={
                        "id":record["id"],
                        "category":"common_subexpressions",
                        "size":record.get("size","small"),
                        "source":record["source"],
                        "expected_output":record["expected_output"],
                        "inputs":record.get("inputs",[{}]),
                    }
                else:
                    compact={"source":record["source"]}
                path.write_text(json.dumps(compact,indent=2)+"\n",encoding="utf-8")
        return

    split_records={}
    metadata=[]
    for split,directory in source_paths.items():
        old_records=[]
        for path in sorted(directory.glob("*.json")):
            record=json.loads(path.read_text(encoding="utf-8"))
            if record.get("category")!="common_subexpressions" or record.get("split")!=split:
                raise ValueError(f"Unexpected legacy record in {path}")
            old_records.append(record)
        expected_count=80 if split=="training" else 20
        if len(old_records)!=expected_count:
            raise ValueError(
                f"Expected {expected_count} archived {split} examples; found {len(old_records)}."
            )
        simplified=[]
        for record in old_records:
            if not all(isinstance(record.get(key),str) for key in ("id","source","expected_output")):
                raise ValueError(f"Legacy record {record.get('id')} is missing source or output.")
            source_hash=hashlib.sha256(record["source"].encode("utf-8")).hexdigest()
            metadata.append({
                "id":record["id"],
                "category":record["category"],
                "size":record.get("size","small"),
                "family_id":record.get("family_id"),
                "split":split,
                "source_sha256":source_hash,
                "expected_output":record["expected_output"],
            })
            if split=="training":
                simplified.append({
                    "id":record["id"],
                    "category":record["category"],
                    "size":record.get("size","small"),
                    "source":record["source"],
                    "expected_output":record["expected_output"],
                    "inputs":record.get("inputs",[{}]),
                })
            else:
                simplified.append({"source":record["source"]})
        split_records[split]=simplified

    family_splits={}
    for item in metadata:
        family_splits.setdefault(item["family_id"],set()).add(item["split"])
    if any(len(splits)!=1 for splits in family_splits.values()):
        raise ValueError("Legacy common-subexpression families cross their stored split.")

    archive_root.mkdir(parents=True)
    (archive_root/"training.json").write_text(
        json.dumps(split_records["training"],indent=2)+"\n",encoding="utf-8"
    )
    (archive_root/"testing.json").write_text(
        json.dumps(split_records["testing"],indent=2)+"\n",encoding="utf-8"
    )
    (archive_root/"manifest.json").write_text(
        json.dumps({
            "dataset":"Archived legacy common-subexpression programs",
            "program_count":len(metadata),
            "training_count":80,
            "testing_count":20,
            "training_record_fields":["id","category","size","source","expected_output","inputs"],
            "testing_record_fields":["source"],
            "metadata":metadata,
            "note":"Inactive archive preserving prior programs. It is not read by either active training pipeline.",
        },indent=2)+"\n",encoding="utf-8"
    )
    for split,directory in source_paths.items():
        for path in directory.glob("*.json"):
            record=json.loads(path.read_text(encoding="utf-8"))
            compact=(
                {
                    "id":record["id"],
                    "category":"common_subexpressions",
                    "size":record.get("size","small"),
                    "source":record["source"],
                    "expected_output":record["expected_output"],
                    "inputs":record.get("inputs",[{}]),
                }
                if split=="training" else {"source":record["source"]}
            )
            path.write_text(json.dumps(compact,indent=2)+"\n",encoding="utf-8")


archive_legacy_category_records()
rng=random.Random(SEED)
for category in category_names:
    prefix=category_prefixes[category]
    for family in range(6):
        for variant in range(10):
            size,source=additional_program(category,family,variant,rng)
            # Keep complete template families on one side of the held-out split.
            split="training" if family < 4 else "testing"
            programs.append((prefix,category,size,source,f"{category}:family-{family:02d}",split))

# Legacy 40-per-category programs remain training-only as a single template family.
legacy_programs=programs[:200]
programs=[(*record,f"{record[1]}:legacy", "training") for record in legacy_programs] + programs[200:]

assert len(programs)==500, f"Expected 500 programs, got {len(programs)}"
assert len({source for _,_,_,source,_,_ in programs})==500, "Duplicate source programs detected."
for category in category_names:
    category_records=[record for record in programs if record[1]==category]
    assert len(category_records)==100, f"{category}: expected 100 programs, got {len(category_records)}"
    assert sum(record[5]=="training" for record in category_records)==80, category
    assert sum(record[5]=="testing" for record in category_records)==20, category

records=[]
metadata=[]
category_counts={}
size_counts={}
family_assignments={}
training_records=[]
testing_records=[]
for prefix,category,size,source,family_id,split in programs:
    serial=sum(1 for record in records if record["category"]==category)+1
    record={
        "id":f"{prefix}_{serial:03d}",
        "category":category,
        "size":size,
        "source":source,
        "expected_output":expected_eval(source),
        "inputs":[{}],
    }
    records.append(record)
    (OUT/f"{record['id']}.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    metadata.append({
        "id":record["id"],
        "category":category,
        "size":size,
        "family_id":family_id,
        "split":split,
        "source_sha256":hashlib.sha256(source.encode("utf-8")).hexdigest(),
    })
    category_counts[category]=category_counts.get(category,0)+1
    size_counts[(category,size)]=size_counts.get((category,size),0)+1
    (training_records if split=="training" else testing_records).append(record)
    family_assignments.setdefault(category,{"training":[],"testing":[]})
    if family_id not in family_assignments[category][split]:
        family_assignments[category][split].append(family_id)

valid_ids={record["id"] for record in records}
for stale in OUT.glob("CAT*.json"):
    if stale.stem not in valid_ids:
        stale.unlink()
(OUT/"dataset.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")

category_split_counts={
    category:{
        "total":sum(item["category"]==category for item in records),
        "training":sum(item["category"]==category and meta["split"]=="training"
                       for item,meta in zip(records,metadata)),
        "testing":sum(item["category"]==category and meta["split"]=="testing"
                      for item,meta in zip(records,metadata)),
    } for category in category_names
}
manifest={
    "schema_version":"4.0",
    "dataset":"500-program rule-based compiler benchmark",
    "seed":SEED,
    "program_count":len(records),
    "training_count":len(training_records),
    "testing_count":len(testing_records),
    "category_counts":category_split_counts,
    "size_counts":{
        split:{size:sum(meta["split"]==split and meta["size"]==size for meta in metadata)
               for size in ("small","medium","large")}
        for split in ("training","testing")
    },
    "families_per_category":{
        category:len(family_assignments[category]["training"])+len(family_assignments[category]["testing"])
        for category in category_names
    },
    "family_assignments":family_assignments,
    "program_metadata":metadata,
    "supported_source_syntax":"Assignments, + - * / %, parentheses, unary minus, print, and the compiler's restricted counted for-loop form.",
    "dataset_file":"benchmarks/dataset.json",
    "training_file":"dataset/training/training.json",
    "testing_file":"dataset/testing/testing.json",
    "training_record_fields":["id","category","size","source","expected_output","inputs"],
    "testing_record_fields":["source"],
    "note":"Split/family metadata is kept in this manifest, not in program records. Applicability labels and features are derived by compiler analysis at training/evaluation time.",
}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")

# Keep each persisted split in its declared compact schema. Metadata needed for
# provenance and family isolation remains in the manifest, never in program rows.
dataset_root=ROOT/"datasets"
for split,split_records in (("training",training_records),("testing",testing_records)):
    split_root=dataset_root/split
    split_root.mkdir(parents=True,exist_ok=True)
    for category in category_names:
        category_dir=split_root/category
        category_dir.mkdir(parents=True,exist_ok=True)
        for stale in category_dir.glob("*.json"):
            stale.unlink()
    if (split_root/"labels.json").exists():
        (split_root/"labels.json").unlink()
    if split=="training":
        stored_records=split_records
    else:
        stored_records=[{"source":record["source"]} for record in split_records]
    for record in split_records:
        category_path=split_root/record["category"]
        category_record=record if split=="training" else {"source":record["source"]}
        (category_path/f"{record['id']}.json").write_text(
            json.dumps(category_record,indent=2)+"\n",encoding="utf-8"
        )
    (split_root/"programs.json").write_text(
        json.dumps(stored_records,indent=2)+"\n",encoding="utf-8"
    )

training_root=ROOT/"dataset"/"training"
testing_root=ROOT/"dataset"/"testing"
training_root.mkdir(parents=True,exist_ok=True)
testing_root.mkdir(parents=True,exist_ok=True)
(training_root/"training.json").write_text(
    json.dumps(training_records,indent=2)+"\n",encoding="utf-8"
)
(testing_root/"testing.json").write_text(
    json.dumps([{"source":record["source"]} for record in testing_records],indent=2)+"\n",
    encoding="utf-8",
)
dataset_manifest={
    "source":"existing benchmarks/dataset.json",
    "program_count":len(records),
    "training_count":len(training_records),
    "testing_count":len(testing_records),
    "category_counts":{
        category:{
            "training":category_split_counts[category]["training"],
            "testing":category_split_counts[category]["testing"],
            "total":category_split_counts[category]["total"],
        } for category in category_names
    },
    "training_file":"dataset/training/training.json",
    "testing_file":"dataset/testing/testing.json",
    "training_record_fields":["id","category","size","source","expected_output","inputs"],
    "testing_record_fields":["source"],
    "training_family_count":sum(bool(value["training"]) for value in family_assignments.values()),
    "testing_family_count":sum(bool(value["testing"]) for value in family_assignments.values()),
    "family_overlap":[],
    "note":"Projection of the existing 500-program split. Compiler-derived features and labels are not stored in the program records.",
}
(ROOT/"dataset"/"manifest.json").write_text(
    json.dumps(dataset_manifest,indent=2)+"\n",encoding="utf-8"
)

grouped_manifest={
    **manifest,
    "label_method":"Compiler-derived in memory from each source and its original TAC.",
    "compiler_labeling_completed":False,
    "records_include_compiler_results":False,
}
(dataset_root/"benchmark_manifest.json").write_text(
    json.dumps(grouped_manifest,indent=2)+"\n",encoding="utf-8"
)

print("Benchmark file verification:")
for category in category_names:
    counts=category_split_counts[category]
    print(f"{category}: {counts['total']} programs - {counts['training']} training, {counts['testing']} testing")
print(f"Total: {len(records)} programs")
print(f"Training: {len(training_records)}")
print(f"Testing: {len(testing_records)}")
print(f"Manifest: {OUT/'manifest.json'}")
