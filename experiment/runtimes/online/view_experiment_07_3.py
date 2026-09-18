"""Inspect completed 07.3 runs without changing their logs."""
from pathlib import Path
import argparse,json,time
import numpy as np
import mujoco
from model import set_configuration,attachment_position
from experiment_07_3_geometry import OUT,camera

def find_runs():
    manifests=sorted((OUT/'online_').glob('*/manifest.json'))
    if not manifests:raise FileNotFoundError('No 07.3 online batch exists')
    return json.loads(manifests[-1].read_text(encoding='utf-8'))['results']

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--exact',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.exact:items=[('exact boxes',OUT/f'exact_box_seed{a.seed}')]
    else:items=[(r['group'],Path(r['summary']).parent) for r in find_runs() if r['seed']==a.seed]
    if not items:raise RuntimeError('Selected seed has not finished yet')
    loaded=[]
    for name,folder in items:
        m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));d=mujoco.MjData(m);q=np.load(folder/'q_history.npy');meta=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if not a.exact:q=np.vstack([meta['initial_configuration_rad'],q])
        assert q.ndim==2 and q.shape[1]==6 and np.isfinite(q).all()
        loaded.append((name,m,d,q))
    if a.check:
        print(json.dumps([dict(group=x[0],states=len(x[3]),nq=x[1].nq) for x in loaded]));return
    from mujoco import viewer
    state={'selected':0,'k':0,'paused':False,'quit':False}
    def key(k):
        if k==32:state['paused']=not state['paused']
        elif k==48:state['k']=0
        elif k in (70,102):state['k']=max(len(x[3])-1 for x in loaded)
        elif 49<=k<49+len(loaded):state['selected']=k-49
    while not state['quit']:
        active=state['selected'];name,m,d,q=loaded[active]
        print(f'Viewing {name}, seed {a.seed}. 1/2 switch, SPACE pause, 0 restart, F final.',flush=True)
        with viewer.launch_passive(m,d,key_callback=key) as v:
            v.opt.geomgroup[1]=0;v.opt.geomgroup[5]=1;c=camera()
            for field in ['lookat','distance','azimuth','elevation']:setattr(v.cam,field,getattr(c,field))
            while v.is_running() and active==state['selected']:
                k=min(state['k'],len(q)-1);set_configuration(m,d,q[k]);v.sync()
                if not state['paused']:state['k']=min(k+1,len(q)-1)
                time.sleep(.02)
            if active==state['selected']:state['quit']=True
if __name__=='__main__':main()
