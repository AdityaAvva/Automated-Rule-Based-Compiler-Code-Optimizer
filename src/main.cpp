#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

struct Ins { string op, a, b, out; };
static bool isNum(const string&s){ if(s.empty()) return false; char*e=nullptr; strtod(s.c_str(),&e); return e && *e=='\0'; }
static double num(const string&s){ size_t n=0; double v=stod(s,&n); if(n!=s.size()) throw runtime_error("Bad number: "+s); return v; }
static string fmt(double x){ if(fabs(x)<1e-10) x=0; ostringstream o; o<<setprecision(12)<<x; return o.str(); }
static bool temp(const string&s){ return s.size()>1 && s[0]=='t' && all_of(s.begin()+1,s.end(),[](char c){return isdigit((unsigned char)c);}); }
static string trim(string s){ auto a=s.find_first_not_of(" \t\r\n"); if(a==string::npos)return ""; auto b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1); }

class ExprParser {
    string s; size_t p=0; vector<Ins>& code; int& nextTemp;
    void ws(){while(p<s.size()&&isspace((unsigned char)s[p]))p++;}
    bool eat(char c){ws(); if(p<s.size()&&s[p]==c){p++;return true;}return false;}
    string atom(){
        ws();
        if(eat('(')){string v=expr(); if(!eat(')')) throw runtime_error("Missing ) in expression"); return v;}
        ws(); size_t st=p;
        if(p<s.size()&&(isdigit((unsigned char)s[p])||s[p]=='.')){while(p<s.size()&&(isdigit((unsigned char)s[p])||s[p]=='.'))p++;}
        else if(p<s.size()&&(isalpha((unsigned char)s[p])||s[p]=='_')){while(p<s.size()&&(isalnum((unsigned char)s[p])||s[p]=='_'))p++;}
        else if(eat('-')) {string v=atom(); if(isNum(v))return fmt(-num(v)); string t="t"+to_string(nextTemp++);code.push_back({"neg",v,"",t});return t;}
        else throw runtime_error("Expected operand near: "+s.substr(p));
        return s.substr(st,p-st);
    }
    string term(){string l=atom(); for(;;){ws();char c=(p<s.size()?s[p]:'\0');if(c!='*'&&c!='/'&&c!='%')break;p++;string r=atom();string t="t"+to_string(nextTemp++);code.push_back({string(1,c),l,r,t});l=t;}return l;}
    string expr(){string l=term();for(;;){ws();char c=(p<s.size()?s[p]:'\0');if(c!='+'&&c!='-')break;p++;string r=term();string t="t"+to_string(nextTemp++);code.push_back({string(1,c),l,r,t});l=t;}return l;}
public:
    ExprParser(string x, vector<Ins>&c,int&n):s(x),code(c),nextTemp(n){}
    string parse(){string v=expr();ws();if(p!=s.size())throw runtime_error("Unexpected expression text: "+s.substr(p));return v;}
};

static vector<string> splitStatements(string source){
    // Supported loop form is expanded before statement parsing.
    regex loop(R"(for\s*\(\s*([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*;\s*\1\s*<\s*(\d+)\s*;\s*\1\s*=\s*\1\s*\+\s*1\s*\)\s*\{([^{}]*)\})");
    smatch m;
    while(regex_search(source,m,loop)){
        string var=m[1], body=m[4]; int start=stoi(m[2]), end=stoi(m[3]);
        string expanded;
        for(int i=start;i<end&&i-start<1000;i++){
            expanded += var+" = "+to_string(i)+";\n";
            string b=body;
            // Simple supported loop body: assignments and print statements.
            expanded += b+"\n";
        }
        source.replace(m.position(),m.length(),expanded);
    }
    vector<string> out; string cur; int depth=0;
    for(char c:source){ if(c=='(')depth++; if(c==')')depth--; if(c==';'&&depth==0){out.push_back(trim(cur));cur.clear();}else cur+=c; }
    if(!trim(cur).empty())out.push_back(trim(cur));
    return out;
}

static vector<Ins> compileSource(const string&source){
    vector<Ins> code; int nt=1;
    for(string st:splitStatements(source)){
        st=trim(st); if(st.empty())continue;
        if(st.rfind("print",0)==0){
            string v=trim(st.substr(5)); if(v.empty())throw runtime_error("print requires an operand");
            // Support print(expr) and print variable/expression.
            if(v.front()=='('&&v.back()==')')v=v.substr(1,v.size()-2);
            string val=ExprParser(v,code,nt).parse();
            code.push_back({"print",val,"",""}); continue;
        }
        size_t eq=st.find('='); if(eq==string::npos)throw runtime_error("Expected assignment: "+st);
        string lhs=trim(st.substr(0,eq)), rhs=trim(st.substr(eq+1));
        if(lhs.empty()||! (isalpha((unsigned char)lhs[0])||lhs[0]=='_'))throw runtime_error("Invalid identifier: "+lhs);
        string val=ExprParser(rhs,code,nt).parse();
        code.push_back({"=",val,"",lhs});
    }
    return code;
}
static string render(const vector<Ins>&v){ostringstream o;for(auto&i:v){if(i.op=="=")o<<i.out<<" = "<<i.a;else if(i.op=="print")o<<"print "<<i.a;else if(i.op=="neg")o<<i.out<<" = -"<<i.a;else o<<i.out<<" = "<<i.a<<" "<<i.op<<" "<<i.b;o<<"\n";}return o.str();}
static string resolve(string x,const map<string,string>&c){set<string> seen;while(c.count(x)&&!seen.count(x)){seen.insert(x);x=c.at(x);}return x;}
static bool evalOp(const string&op,double a,double b,double&v){
    if(op=="+")v=a+b;else if(op=="-")v=a-b;else if(op=="*")v=a*b;else if(op=="/"){if(fabs(b)<1e-12)return false;v=a/b;}else if(op=="%"){if(fabs(b)<1e-12)return false;v=fmod(a,b);}else return false;return isfinite(v);
}
static bool foldOne(Ins&i){if(i.op=="neg"&&isNum(i.a)){i.op="=";i.a=fmt(-num(i.a));i.b="";return true;}if(i.op=="="||i.op=="print"||!isNum(i.a)||!isNum(i.b))return false;double v;if(!evalOp(i.op,num(i.a),num(i.b),v))return false;i.op="=";i.a=fmt(v);i.b="";return true;}
static int constantPropagation(vector<Ins>&v){
    map<string,string> values; int changes=0;
    auto invalidate=[&](const string& name){
        values.erase(name);
        for(auto it=values.begin();it!=values.end();){
            if(it->second==name)it=values.erase(it);
            else ++it;
        }
    };
    for(auto&i:v){
        string oldA=i.a, oldB=i.b;
        i.a=resolve(i.a,values);
        if(i.op!="print")i.b=resolve(i.b,values);
        // Resolve copies as well as constants before a later definition can
        // change the value of the source variable.
        if(i.a!=oldA)changes++;
        if(i.b!=oldB)changes++;
        if(i.op=="="){
            string value=i.a;
            invalidate(i.out);
            if(value==i.out){i.op="nop";changes++;}
            else values[i.out]=value;
        } else if(i.op!="print") invalidate(i.out);
    }
    v.erase(remove_if(v.begin(),v.end(),[](auto&i){return i.op=="nop";}),v.end());
    return changes;
}
static int algebraic(vector<Ins>&v){int n=0;for(auto&i:v){if(i.op=="+"&&(i.b=="0"||i.a=="0")){i.a=(i.b=="0"?i.a:i.b);i.op="=";i.b="";n++;}else if(i.op=="-"&&i.b=="0"){i.op="=";i.b="";n++;}else if(i.op=="*"&&(i.b=="1"||i.a=="1")){i.a=(i.b=="1"?i.a:i.b);i.op="=";i.b="";n++;}else if(i.op=="/"&&i.b=="1"){i.op="=";i.b="";n++;}else if(i.op=="*"&&(i.a=="0"||i.b=="0")){i.op="=";i.a="0";i.b="";n++;}else if(i.op=="+"&&i.a=="0"){i.op="=";i.b="";n++;}}return n;}
static int cse(vector<Ins>&v){
    map<string,string> exprs; int n=0;
    auto invalidate=[&](const string& name){
        for(auto it=exprs.begin();it!=exprs.end();){
            string key=it->first; vector<string> parts; string part;
            stringstream ss(key); while(getline(ss,part,'|')) parts.push_back(part);
            if(parts.size()>=3 && (parts[1]==name || parts[2]==name)) it=exprs.erase(it); else ++it;
        }
    };
    for(auto&i:v){
        if(i.op=="print") continue;
        if(i.op=="="){invalidate(i.out); continue;}
        if(i.op=="neg"){invalidate(i.out); continue;}
        string key=i.op+"|"+i.a+"|"+i.b;
        if(i.op=="+"||i.op=="*"){string rev=i.op+"|"+i.b+"|"+i.a;if(exprs.count(rev))key=rev;}
        if(exprs.count(key)){i.op="=";i.a=exprs[key];i.b="";invalidate(i.out);n++;}
        else {invalidate(i.out);exprs[key]=i.out;}
    }
    return n;
}
static int strengthReduction(vector<Ins>&v){int n=0;for(auto&i:v){if(i.op=="*"&&isNum(i.b)&&num(i.b)==2){i.op="+";i.b=i.a;n++;}}return n;}
static int deadCode(vector<Ins>&v){
    set<string> live;int removed=0;
    for(int k=(int)v.size()-1;k>=0;k--){
        auto&i=v[k];
        if(i.op=="print"){
            if(!isNum(i.a))live.insert(i.a);
            continue;
        }
        bool needed=live.count(i.out)>0;
        if(!needed){v.erase(v.begin()+k);removed++;continue;}
        live.erase(i.out);
        if(!isNum(i.a))live.insert(i.a);
        if(!isNum(i.b)&&!i.b.empty())live.insert(i.b);
    }
    return removed;
}
static vector<Ins> optimize(vector<Ins> v,map<string,int>&fired,map<string,int>&invocations,int&iterations,const set<string>& enabled,vector<int>* convergence=nullptr,vector<int>* convergenceStable=nullptr){
    if(convergence) convergence->push_back((int)v.size());
    if(convergenceStable) convergenceStable->push_back(0);
    set<string> seenStates{render(v)};
    int iteration=0;
    for(;;){
        string before=render(v);
        auto runPass=[&](const string& name,auto apply){
            if(!enabled.count(name))return;
            invocations[name]++;
            int rewrites=apply();
            fired[name]+=rewrites;
        };
        runPass("constant_folding",[&](){
            int rewrites=0;
            for(auto&i:v)if(foldOne(i))rewrites++;
            return rewrites;
        });
        runPass("constant_propagation",[&](){return constantPropagation(v);});
        runPass("algebraic_simplification",[&](){return algebraic(v);});
        runPass("common_subexpression_elimination",[&](){return cse(v);});
        runPass("dead_code_elimination",[&](){return deadCode(v);});
        runPass("strength_reduction",[&](){return strengthReduction(v);});

        iteration++;
        iterations=iteration;
        string after=render(v);
        bool stable=(after==before);
        if(convergence) convergence->push_back((int)v.size());
        if(convergenceStable) convergenceStable->push_back(stable?1:0);
        if(stable) return v;
        if(!seenStates.insert(after).second){
            throw runtime_error(
                "Optimization did not converge: TAC repeated after iteration "+
                to_string(iteration)+"."
            );
        }
    }
}
static string execute(const vector<Ins>&v,const map<string,double>&inputs={}){
    map<string,double> env=inputs;ostringstream out;
    for(auto&i:v){
        if(i.op=="print"){string x=i.a;if(!isNum(x)&&!env.count(x))throw runtime_error("Undefined variable: "+x);out<<fmt(isNum(x)?num(x):env[x])<<"\n";continue;}
        double a;if(isNum(i.a))a=num(i.a);else if(env.count(i.a))a=env[i.a];else throw runtime_error("Undefined variable: "+i.a);
        if(i.op=="="){env[i.out]=a;continue;}double b=0;if(i.op!="neg"){if(isNum(i.b))b=num(i.b);else if(env.count(i.b))b=env[i.b];else throw runtime_error("Undefined variable: "+i.b);}
        if(i.op=="neg")env[i.out]=-a;else{double val;if(!evalOp(i.op,a,b,val))throw runtime_error("Invalid operation: "+i.op);env[i.out]=val;}
    }return out.str();
}
static long long timedExecuteMicros(const vector<Ins>&v,int repetitions=100){
    auto start=chrono::high_resolution_clock::now();
    for(int i=0;i<repetitions;i++) (void)execute(v);
    auto stop=chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::microseconds>(stop-start).count()/repetitions;
}
static vector<Ins> runSinglePass(vector<Ins> v,const string&name,int& transformations){
    transformations=0;
    if(name=="constant_folding"){for(auto&i:v)if(foldOne(i))transformations++;}
    else if(name=="constant_propagation")transformations=constantPropagation(v);
    else if(name=="algebraic_simplification")transformations=algebraic(v);
    else if(name=="common_subexpression_elimination")transformations=cse(v);
    else if(name=="strength_reduction")transformations=strengthReduction(v);
    else if(name=="dead_code_elimination")transformations=deadCode(v);
    return v;
}
static int instructionCount(const vector<Ins>&v){return (int)v.size();}
static int operationCount(const vector<Ins>&v){int n=0;for(auto&i:v)if(i.op=="+"||i.op=="-"||i.op=="*"||i.op=="/"||i.op=="%")n++;return n;}
static int tempCount(const vector<Ins>&v){set<string>s;for(auto&i:v)if(temp(i.out))s.insert(i.out);return (int)s.size();}
static int codeSize(const vector<Ins>&v){return (int)render(v).size();}
int main(int argc,char**argv){
    try{
        string source, inputPath;
        bool emitConvergence=false;
        set<string> enabled={"constant_folding","constant_propagation","algebraic_simplification","common_subexpression_elimination","strength_reduction","dead_code_elimination"};
        for(int i=1;i<argc;i++){
            string arg=argv[i];
            if(arg=="--passes" && i+1<argc){
                enabled.clear(); string list=argv[++i]; string item; stringstream ss(list);
                while(getline(ss,item,',')) if(!item.empty()) enabled.insert(item);
            } else if(arg=="--trace-convergence") emitConvergence=true;
            else inputPath=arg;
        }
        if(!inputPath.empty()){ifstream f(inputPath);if(!f)throw runtime_error("Cannot open input file: "+inputPath);stringstream ss;ss<<f.rdbuf();source=ss.str();}
        else{stringstream ss;ss<<cin.rdbuf();source=ss.str();}
        auto original=compileSource(source);auto originalText=render(original);
        auto t0=chrono::high_resolution_clock::now();map<string,int> fired,invocations;int optimizationIterations=0;vector<int> convergence,convergenceStable;auto optimized=optimize(original,fired,invocations,optimizationIterations,enabled,emitConvergence?&convergence:nullptr,emitConvergence?&convergenceStable:nullptr);auto t1=chrono::high_resolution_clock::now();
        string out1=execute(original),out2=execute(optimized);bool pass=(out1==out2);
        auto micros=chrono::duration_cast<chrono::microseconds>(t1-t0).count();
        long long interpreterBefore=timedExecuteMicros(original,100);
        long long interpreterAfter=timedExecuteMicros(optimized,100);
        cout<<"=== SOURCE ===\n"<<source<<"\n=== ORIGINAL TAC ===\n"<<originalText<<"=== OPTIMIZED TAC ===\n"<<render(optimized);
        cout<<"=== OUTPUT CHECK ===\nOriginal output:\n"<<out1<<"Optimized output:\n"<<out2<<"Verdict: "<<(pass?"PASS":"FAIL")<<"\n";
        cout<<"=== METRICS ===\n";
        cout<<"instructions_before="<<instructionCount(original)<<"\ninstructions_after="<<instructionCount(optimized)<<"\n";
        cout<<"operations_before="<<operationCount(original)<<"\noperations_after="<<operationCount(optimized)<<"\n";
        cout<<"temporaries_before="<<tempCount(original)<<"\ntemporaries_after="<<tempCount(optimized)<<"\n";
        cout<<"code_size_before="<<codeSize(original)<<"\ncode_size_after="<<codeSize(optimized)<<"\n";
        cout<<"optimization_time_microseconds="<<micros<<"\n";
        cout<<"optimization_cycles="<<optimizationIterations<<"\n";
        cout<<"interpreter_time_before_microseconds="<<interpreterBefore<<"\n";
        cout<<"interpreter_time_after_microseconds="<<interpreterAfter<<"\n=== OPTIMIZER COUNTS ===\n";
        vector<string> names={"constant_folding","constant_propagation","algebraic_simplification","common_subexpression_elimination","dead_code_elimination","strength_reduction"};
        for(const auto&name:names){
            cout<<name<<"="<<fired[name]<<"\n";
            cout<<name<<"_invocations="<<invocations[name]<<"\n";
        }
        cout<<"=== PER-OPTIMIZER EFFECTIVENESS (ISOLATED SINGLE PASS) ===\n";
        for(const auto& name:names){
            int transformations=0;
            auto one=runSinglePass(original,name,transformations);
            bool oneCorrect=false; try{oneCorrect=(execute(one)==out1);}catch(...){oneCorrect=false;}
            cout<<name<<"_transformations="<<transformations<<"\n";
            cout<<name<<"_instructions_removed="<<(instructionCount(original)-instructionCount(one))<<"\n";
            cout<<name<<"_operations_removed="<<(operationCount(original)-operationCount(one))<<"\n";
            cout<<name<<"_temporaries_removed="<<(tempCount(original)-tempCount(one))<<"\n";
            cout<<name<<"_code_characters_removed="<<(codeSize(original)-codeSize(one))<<"\n";
            cout<<name<<"_output_match="<<(oneCorrect?1:0)<<"\n";
        }
        if(emitConvergence){cout<<"=== CONVERGENCE TRACE ===\n";for(size_t i=0;i<convergence.size();++i)cout<<"convergence_iteration="<<i<<",instructions="<<convergence[i]<<",stable="<<convergenceStable[i]<<"\n";}
        return pass?0:2;
    }catch(const exception&e){cerr<<"ERROR: "<<e.what()<<"\nSupported syntax: assignments, arithmetic (+ - * / %), print expr, and simple for(i=0;i<N;i=i+1){...} loops.\n";return 1;}
}
