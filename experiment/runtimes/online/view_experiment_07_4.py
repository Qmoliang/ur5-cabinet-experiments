"""Open 07.4 actual trajectories, optionally with recorded proxy overlays."""
from pathlib import Path
import argparse,json,sys
from experiment_07_3_geometry import ROOT
OUT=ROOT/'formal_results/experiment_07/development_07_4'

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--group',choices=['ET30','D30','P30']);p.add_argument('--overlays',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    rows=json.loads(sorted((OUT/'online_').glob('*/manifest.json'))[-1].read_text(encoding='utf-8'))['results']
    if a.overlays:
        group=a.group or 'D30';row=next(x for x in rows if x['group']==group and x['seed']==a.seed);folder=Path(row['summary']).parent
        from formal_protocol_v3_viewer import FormalComparison
        viewer=FormalComparison(folder,folder,replay_only=True)
        if a.check:viewer.check()
        else:viewer.run('ellipsoid',1.)
    else:
        import view_experiment_07_3 as base
        base.find_runs=lambda: [x for x in rows if not a.group or x['group']==a.group]
        sys.argv=[sys.argv[0],'--seed',str(a.seed)]+(['--check'] if a.check else [])
        base.main()
if __name__=='__main__':main()
