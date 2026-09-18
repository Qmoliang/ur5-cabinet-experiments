"""Isolated objective-term ablation using unchanged online runner and geometry."""
from pathlib import Path
import sys,json,shutil,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
import run_study as runner
from neo_online import OnlineNEO

CASES=['neo_sphere_300','neo_ellipsoid_300','neo_ellipsoid_46']

class NoManipNEO(OnlineNEO):
    failure_folder=None
    def assemble(self,target):
        H,g,A,lo,hi,info=super().assemble(target)
        newg=-info['J'].T@info['v']
        assert np.allclose(newg-g,info['e']*info['jm'],atol=1e-14,rtol=1e-12)
        self.ablation_last_assembly=(H,newg,A,lo,hi,info)
        return H,newg,A,lo,hi,info

    def solve(self,target):
        try:return super().solve(target)
        except Exception:
            if self.failure_folder is not None and hasattr(self,'ablation_last_assembly'):
                H,g,A,lo,hi,info=self.ablation_last_assembly
                np.savez_compressed(self.failure_folder/'failed_qp.npz',H=H,g=g,A=A,lo=lo,hi=hi,
                    q=self.data.qpos[:self.nv].copy(),target=target,J=info['J'],v=info['v'],
                    centers=self.obstacle_centers,offsets=self.obstacle_offsets,proxy_ids=self.proxy_ids,
                    radii=self.obstacle_radii if self.representation=='sphere' else np.empty(0),
                    shapes=self.obstacle_shapes if self.representation=='ellipsoid' else np.empty((0,3,3)),
                    uncertainty=self.obstacle_uncertainty_shapes if self.representation=='ellipsoid' else np.empty((0,3,3)))
            raise

def main():
    if len(sys.argv)!=2 or sys.argv[1] not in CASES:raise SystemExit('Specify one preregistered case')
    case=sys.argv[1];out=HERE/'results'/case
    if out.exists():raise FileExistsError(out)
    NoManipNEO.failure_folder=out
    runner.OnlineNEO=NoManipNEO;runner.HERE=HERE
    try:runner.main()
    finally:
        if out.exists():
            shutil.copyfile(Path(__file__),out/'source/run_ablation.py')
            runner.dump(out/'ablation_identity.json',dict(manipulability_objective_weight=0,
                original_weight=1,only_control_change='remove -e*grad(m) from g; H and hard constraints unchanged',
                run_ablation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        if (out/'summary.json').exists():
            summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
            summary.update(controller_method='NEO core ablation: manipulability coefficient zero',
                           manipulability_objective_weight=0)
            runner.dump(out/'summary.json',summary);runner.dump(out/summary['run_name']/'summary.json',summary)

if __name__=='__main__':main()
