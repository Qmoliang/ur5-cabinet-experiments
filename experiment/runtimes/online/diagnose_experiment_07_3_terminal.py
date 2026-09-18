"""Reconstruct QP constraints in completed 07.3 runs; no control feedback."""
from pathlib import Path
import json,argparse
import diagnose_et30_terminal as terminal
from diagnose_et30_terminal import diagnose,ROOT
import numpy as np,osqp
from scipy import sparse

def diagnostic_qp(H,g,A,l,u,keep):
    solver=osqp.OSQP()
    solver.setup(P=sparse.triu(sparse.csc_matrix(H),format='csc'),q=g,A=sparse.csc_matrix(A[keep]),l=l[keep],u=u[keep],verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=200000,polishing=True)
    result=solver.solve(raise_error=False)
    if not result.info.status.lower().startswith('solved'):raise RuntimeError(result.info.status)
    return result
terminal.independent_qp=diagnostic_qp
OUT=ROOT/'formal_results/experiment_07/development_07_3'

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path);a=p.parse_args();batch=a.batch or sorted((OUT/'online_').glob('*/manifest.json'))[-1]
    rows=json.loads(batch.read_text(encoding='utf-8'))['results'];out=OUT/'terminal_constraints.json';existing=json.loads(out.read_text())['runs'] if out.exists() else [];done={(r['group'],r['seed']) for r in existing}
    for row in rows:
        if (row['group'],row['seed']) in done:continue
        r=diagnose(dict(run=str(Path(row['summary']).parent),seed=row['seed']),include_counterfactuals=False,geometry_only=True);r['group']=row['group'];r['independent_qp_tolerance']=None;r['counterfactuals']='omitted; near-off solve failed stricter tolerances';existing.append(r)
        out.write_text(json.dumps(dict(diagnostic_only=True,batch=str(batch),runs=existing),indent=2),encoding='utf-8')
        print(json.dumps(dict(group=row['group'],seed=row['seed'],final_error_mm=r['final_error_mm'],back_gap_mm=r['back_min_certificate_gap_mm'],tight=[dict(box=p['box'],robot=p['robot_index'],h_mm=p['barrier_h_mm'],U_mm=p['uncertainty_support_along_normal_mm']) for p in r['tight_pairs'] if p['barrier_h_mm']<.05])),flush=True)
if __name__=='__main__':main()
