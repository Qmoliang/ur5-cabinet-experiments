"""07.4 recorded-geometry diagnosis, without independent QP re-optimization."""
from pathlib import Path
import json
from diagnose_et30_terminal import diagnose,ROOT
OUT=ROOT/'formal_results/experiment_07/development_07_4'

def main():
    batch=sorted((OUT/'online_').glob('*/manifest.json'))[-1];rows=json.loads(batch.read_text(encoding='utf-8'))['results'];path=OUT/'terminal_constraints.json';result=[]
    for row in rows:
        record=diagnose(dict(run=str(Path(row['summary']).parent),seed=row['seed']),include_counterfactuals=False,geometry_only=True);record['group']=row['group'];result.append(record)
        path.write_text(json.dumps(dict(batch=str(batch),diagnostic_only=True,duals_available=False,runs=result),indent=2),encoding='utf-8')
        print(json.dumps(dict(group=row['group'],seed=row['seed'],error_mm=record['final_error_mm'],near_zero_pairs=[dict(box=p['box'],robot=p['robot_index'],h_mm=p['barrier_h_mm']) for p in record['tight_pairs']])),flush=True)
if __name__=='__main__':main()
