"""Publish saved evidence only: no controller imports, no mj_step, no reruns.

Run on the original evidence host with --source and --output. q/ee histories
are copied losslessly as little-endian float64 + gzip. Cloud samples are only
an explicitly labelled visual aid, never a substitute for the full audit.
"""
from pathlib import Path
import argparse, json, gzip, hashlib, shutil, sys, xml.etree.ElementTree as ET
import numpy as np
import mujoco

def read(p): return json.loads(p.read_text())
def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    u=a.source;o=a.output;o.mkdir(parents=True,exist_ok=True)
    versions={'known':u/'dev/v5','online':u/'closeout/final_r17'}
    for folder in ['replays','models','evidence','videos']:(o/folder).mkdir(exist_ok=True)
    sources={};cases=[];task_models={};fkmax=0.
    comp=read(versions['known']/'comparison.json')
    labels={r['task']:r['label'] for r in comp['rows']}
    # Exact robot model and original mesh assets. Only path portability changes.
    original=versions['online']/'code/common/robot_only.xml'
    tree=ET.parse(original);root=tree.getroot();assetdir=Path(root.find('compiler').get('meshdir'))
    root.find('compiler').set('meshdir','assets')
    tree.write(o/'models/robot.xml',encoding='unicode')
    shutil.copytree(assetdir,o/'models/assets',dirs_exist_ok=True)
    model=mujoco.MjModel.from_xml_path(str(original));data=mujoco.MjData(model)
    sys.path.insert(0,str(versions['online']/'code/known_v6/src'))
    from core.robot import build_robot_certificate
    cert=build_robot_certificate(model)
    save(o/'models/robot.json',{'assets':[p.name for p in assetdir.iterdir() if p.is_file()], 'certificate':[dict(body=int(s.body_id),center=s.local_center.tolist(),radius=float(s.radius)) for s in cert], 'site_id':int(model.site('attachment_site').id),'source_sha256':sha(original),'mujoco_version':mujoco.__version__})
    fk_checks=[]
    for mode,base in versions.items():
      for outcome in sorted((base/'outcomes').glob(mode+'__*.json')):
        r=read(outcome);c=Path(r['case']);summary=read(c/'summary.json');job=read(c/'job.json');entry=job.get('entry',job)
        qfile=next(c.rglob('q_history.npy'));eef=qfile.with_name('ee_history.npy');q=np.load(qfile);ee=np.load(eef)
        assert len(q)==len(ee) and q.shape[1]==6 and ee.shape[1]==3
        key=r['id'];task=r['task'];target=entry.get('target',summary.get('target_m'))
        if target is None:target=read(c/'case.json')['target']
        for i in [0,len(q)//2,len(q)-1]:
          data.qpos[:6]=q[i];mujoco.mj_forward(model,data)
          err=float(np.max(np.abs(data.site_xpos[model.site('attachment_site').id]-ee[i])));fkmax=max(fkmax,err)
          assert err<1e-8,(key,i,err)
          fk_checks.append(dict(id=key,frame=i,q=q[i].tolist(),ee=ee[i].tolist()))
        payload=np.concatenate([q,ee],axis=1).astype('<f8').tobytes()
        (o/'replays'/f'{key}.bin.gz').write_bytes(gzip.compress(payload,compresslevel=6,mtime=0))
        scene=read(u/'scenes'/f'{task}.json')
        if task not in task_models:
          save(o/'models'/f'{task}.json',{'boxes':scene['boxes'],'target':target,'label':labels[task]});task_models[task]=True
        events=[];gp=qfile.with_name('guidance.json')
        if gp.exists():
          events=[{k:v for k,v in e.items() if k in ['event','cycle','generation','status','reason','certified_margin_m','seconds']} for e in read(gp).get('events',[])]
        cloudframes=[]
        if mode=='online':
          z=np.load(qfile.with_name('causal_centervox_snapshots.npz'))
          for j,pub in enumerate(z['publish_cycles']):
            lo,hi=map(int,z['offsets'][j:j+2]);points=z['points'][lo:hi];ids=np.linspace(0,len(points)-1,min(2000,len(points)),dtype=int)
            cloudframes.append(dict(cycle=int(pub),source_cycle=int(z['source_cycles'][j]),count=len(points),points=np.round(points[ids],6).tolist()))
        else:
          z=np.load(c/'proxies.npz');points=z['centers'];ids=np.linspace(0,len(points)-1,min(2000,len(points)),dtype=int)
          cloudframes=[dict(cycle=0,count=len(points),points=np.round(points[ids],6).tolist())]
        save(o/'replays'/f'{key}.json',dict(events=events,cloud_frames=cloudframes,cloud_semantics='sampled measured cell centers, not full uncertainty support or unknown occupancy' if mode=='online' else 'sampled known-map proxy centers, not proxy boundaries'))
        errs=np.linalg.norm(ee-np.asarray(target),axis=1)*1000
        video=(r['seed']==1 or not r['strict_pass'])
        case=dict(id=key,mode=mode,version='v5' if mode=='known' else 'final_r17',task=task,label=labels[task],seed=r['seed'],passed=r['strict_pass'],goal=r['goal'],outcome=r['outcome'],termination=r.get('termination_reason','fixed_horizon_complete'),frames=len(q),duration_s=(len(q)-1)*.02,completion_s=r.get('completion_charged_s'),controller_p99_ms=r.get('controller_ms_p99',summary.get('controller_ms_p99')),minimum_proxy_gap_mm=r.get('minimum_proxy_gap_mm'),physical_state_min_mm=r.get('physical_state_min_mm'),self_state_min_mm=r.get('self_state_min_mm'),final_error_mm=float(errs[-1]),minimum_error_mm=float(errs.min()),target=target,video=f'videos/{key}.mp4' if video else None,trajectory=f'replays/{key}.bin.gz',detail=f'replays/{key}.json',audit=f'evidence/{key}.json')
        save(o/'evidence'/f'{key}.json',r);cases.append(case)
        sources[key]=dict(q_sha256=sha(qfile),ee_sha256=sha(eef),outcome_sha256=sha(outcome),frames=len(q),source=str(c),payload_sha256=hashlib.sha256(payload).hexdigest())
        if len(cases)%30==0:print('exported',len(cases),flush=True)
    assert len(cases)==600
    assert sum(c['passed'] for c in cases if c['mode']=='known')==300
    assert sum(c['passed'] for c in cases if c['mode']=='online')==298
    for name,src in {'online-final.json':versions['online']/'CLOSEOUT_REPORT.json','known-v5-comparison.json':versions['known']/'comparison.json','online-freeze.json':versions['online']/'FREEZE_RECORD.json','online-contract-checks.json':versions['online']/'CONTRACT_BOUNDARY_CHECKS.json','known-integrity.json':versions['known']/'FINAL_INTEGRITY.json'}.items():shutil.copy2(src,o/'evidence'/name)
    save(o/'data.json',dict(closed=True,date='2026-11-03',dt=.02,cases=cases,tasks=[dict(id=t,label=labels[t]) for t in labels],known_passed=300,online_passed=298,online_total=300,known_total=300))
    save(o/'evidence/replay-provenance.json',dict(no_experiments_rerun=True,format='gzip little-endian float64 rows [q0..q5,ee_x,ee_y,ee_z], all original saved frames',max_python_fk_error_m=fkmax,sources=sources))
    save(o/'evidence/fk-checks.json',fk_checks)
    print('DONE',len(cases),'FK max',fkmax,flush=True)

if __name__=='__main__':main()
