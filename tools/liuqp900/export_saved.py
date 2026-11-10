"""Export saved C/A/B evidence for a static site. No controller calls or mj_step.
Run with the original study paths; --output must be a separate publication directory.
All recorded q/ee frames are preserved as float64. Initial EE is explicitly FK-derived.
"""
from pathlib import Path
import argparse,json,hashlib,gzip,zipfile,io,xml.etree.ElementTree as ET,sys
import numpy as np
import mujoco

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
def gzjson(p,x):
 b=json.dumps(x,ensure_ascii=False,separators=(',',':')).encode();Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_bytes(gzip.compress(b,compresslevel=8,mtime=0))
def rle(a):
 out=[]
 for i,v in enumerate(a,1):
  v=str(v)
  if out and out[-1][2]==v:out[-1][1]=i+1
  else:out.append([i,i+1,v])
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ab',type=Path,required=True);ap.add_argument('--c',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--legacy-index',type=Path,required=True);ap.add_argument('--expected',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();o=a.output.resolve()
 assert not o.is_relative_to(a.ab.resolve()) and not o.is_relative_to(a.c.resolve());o.mkdir(parents=True,exist_ok=True)
 for name in ['packs','models','visuals','evidence/raw','evidence']: (o/name).mkdir(parents=True,exist_ok=True)
 full=read(a.c/'delivery/COMPARISON_RESULTS.json');expected=read(a.expected);legacy=read(a.legacy_index);inputs=read(a.ab/'INPUT_MANIFEST.json');byinput={x['id']:x for x in inputs};old={x['id']:x for x in legacy['cases'] if x['mode']=='online'};assert len(full['records'])==900 and len(inputs)==len(old)==300
 labels={x['id']:x['label'] for x in legacy['tasks']}
 original=a.ab/'frozen/code/common/robot_only.xml';tree=ET.parse(original);root=tree.getroot();assetdir=Path(root.find('compiler').get('meshdir'));root.find('compiler').set('meshdir','assets');tree.write(o/'models/robot.xml',encoding='unicode')
 robotmodel=mujoco.MjModel.from_xml_path(str(original));robotdata=mujoco.MjData(robotmodel);siteid=int(robotmodel.site('attachment_site').id)
 sys.path.insert(0,str(a.ab/'frozen/code/known_v6/src'));from core.robot import build_robot_certificate
 cert=build_robot_certificate(robotmodel);assetnames=sorted({x.get('file') for x in root.findall('asset/mesh')});assets={n:(assetdir/n).read_text(encoding='utf-8') for n in assetnames};gzjson(o/'models/assets.json.gz',assets)
 robotinfo=dict(assets=assetnames,certificate=[dict(body=int(s.body_id),center=s.local_center.tolist(),radius=float(s.radius)) for s in cert],site_id=siteid,source_sha256=sha_file(original),asset_sha256={n:sha_file(assetdir/n) for n in assetnames},mujoco_version=mujoco.__version__);save(o/'models/robot.json',robotinfo)
 cases=[];condrows=[];visuals={};modelcache={};fkchecks=[];fkmax=0.;rawgroups={};details={};binaries={};offsets={};inputrecords=[];originalfiles={};source_validation=[]
 for item in inputs:
  ident=item['id'];entry=item['input'];assert sha_file(entry['input_file'])==entry['input_sha256'];assert sha_file(entry['scene'])==entry['scene_sha256'];meta=read(entry['metadata']);assert meta==item['metadata']
  with np.load(entry['input_file']) as z:
   q0=z['q0'].copy();target=z['target'].copy();points=z['centers'];ids=np.linspace(0,len(points)-1,min(2000,len(points)),dtype=int);cloud=dict(cycle=0,count=len(points),points=points[ids].tolist(),source_publish_cycle=meta['publish_cycle'],source_cycle=meta['source_cycle'])
  task=item['task'];visuals.setdefault(task,{})[ident]=dict(cloud_frames=[cloud],semantics='Deterministic sample of frozen proxy centers, at most 2000; NOT full safety geometry and NOT online map updates.')
  if entry['scene_sha256'] not in modelcache:
   model=mujoco.MjModel.from_xml_path(entry['scene']);data=mujoco.MjData(model);data.qpos[:6]=q0;mujoco.mj_forward(model,data);scene=read(a.base/'scenes'/(task+'.json'));boxes=[]
   for b in scene['boxes']:
    gid=model.geom(b['name']).id;assert int(model.geom_type[gid])==6;assert np.max(abs(data.geom_xpos[gid]-np.asarray(b['center'])))<1e-9;assert np.max(abs(model.geom_size[gid]-np.asarray(b['half_size'])))<1e-9
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,data.geom_xmat[gid]);boxes.append(dict(name=b['name'],center=data.geom_xpos[gid].tolist(),half_size=model.geom_size[gid].tolist(),quaternion_xyzw=[float(quat[1]),float(quat[2]),float(quat[3]),float(quat[0])],material=b.get('material','shelf')))
   sceneinfo=dict(boxes=boxes,label=labels[task],source_scene_sha256=entry['scene_sha256']);modelcache[entry['scene_sha256']]=(model,data,sceneinfo)
  model,data,sceneinfo=modelcache[entry['scene_sha256']]
  modelname=task+'-'+entry['scene_sha256'][:12];save(o/'models'/(modelname+'.json'),dict(sceneinfo,label=labels[task]));(o/'models'/(modelname+'.source.xml')).write_bytes(Path(entry['scene']).read_bytes())
  assert np.max(abs(np.asarray(old[ident]['target'])-target))<1e-12
  condition=dict(id=ident,task=task,label=labels[task],seed=item['seed'],q0=q0.tolist(),target=target.tolist(),scene='models/'+modelname+'.json',scene_source='models/'+modelname+'.source.xml',input_sha256=entry['input_sha256'],scene_sha256=entry['scene_sha256'],input_full_on_github=False,input_bytes=Path(entry['input_file']).stat().st_size,publish_cycle=meta['publish_cycle'],source_cycle=meta['source_cycle'],proxies=meta['proxies'],legacy_passed=bool(old[ident]['passed']),legacy_url='../closeout/#case='+ident,legacy_protocol_comparable=False)
  condrows.append(condition);inputrecords.append(dict(**condition,source_archive='paired300/inputs/'+ident+'/input.npz',retention='Full input remains in the frozen experiment archive; hash and sampling rule public; not bundled on Pages.'))
  if len(condrows)%50==0:print('inputs verified',len(condrows),flush=True)
 conditions={x['id']:x for x in condrows}
 for r in full['records']:
  ident=r['id'];condition=conditions[r['case']];arm=r['arm'];study=a.c if arm=='C' else a.ab;f=study/'trials'/ident;assert f.is_dir();source=read(f/'result.json');audit=read(f/'audit.json');val=read(f/'VALIDATION.json');entry=byinput[r['case']]['input'];q0=np.array(condition['q0']);target=np.array(condition['target'])
  # Cross-check against the previously delivered immutable archive members.
  files=[]
  for p in sorted(f.iterdir()):
   if not p.is_file():continue
   h=sha_file(p)
   if p.name in expected[ident]:assert h==expected[ident][p.name],(ident,p.name)
   files.append(dict(name=p.name,bytes=p.stat().st_size,sha256=h,archive_member='trials/'+ident+'/'+p.name))
  originalfiles[ident]=files
  with np.load(f/'trajectory.npz') as z:
   q=z['q'];ee=z['ee'];v=z['velocity'];assert q.shape==(r['executed_cycles']+1,6) and ee.shape==(r['executed_cycles'],3);assert len(v)==len(ee);assert np.array_equal(q[0],q0);assert np.array_equal(q[1:],q[:-1]+.02*v)
   robotdata.qpos[:6]=q[0];mujoco.mj_forward(robotmodel,robotdata);ee0=robotdata.site_xpos[siteid].copy();eeall=np.vstack([ee0,ee]);payload=np.column_stack([q,eeall]).astype('<f8').tobytes();roundtrip=np.frombuffer(payload,dtype='<f8').reshape(-1,9);assert np.array_equal(roundtrip[:,:6],q) and np.array_equal(roundtrip[1:,6:],ee)
   model,data,_=modelcache[entry['scene_sha256']]
   for idx in sorted(set([0,len(q)//2,len(q)-1])):
    robotdata.qpos[:6]=q[idx];mujoco.mj_forward(robotmodel,robotdata);data.qpos[:6]=q[idx];mujoco.mj_forward(model,data);fk=robotdata.site_xpos[siteid];fullfk=data.site_xpos[model.site('attachment_site').id];err=max(float(np.max(abs(fk-eeall[idx]))),float(np.max(abs(fullfk-eeall[idx]))));assert err<1e-8,(ident,idx,err);fkmax=max(fkmax,err);fkchecks.append(dict(id=ident,frame=idx,q=q[idx].tolist(),ee=eeall[idx].tolist()))
   modes=rle(z['control_mode'].tolist())
  group=arm+'__'+r['task'];offset=offsets.get(group,0);offsets[group]=offset+len(q);binaries.setdefault(group,[]).append(payload)
  events=[]
  if (f/'events.jsonl').exists():
   for line in (f/'events.jsonl').read_text(encoding='utf-8').splitlines():
    e=json.loads(line)
    if e.get('event')!='original_qp_decision':events.append(e)
  phys=audit.get('physical_motion',{});qpa=read(f/'original_feasibility_audit.json');bad=sorted(set(phys.get('bad_intervals',[])+audit.get('proxy_failed_intervals',[])+audit.get('workspace',{}).get('bad_intervals',[])))
  run={k:v for k,v in r.items() if k not in ['source_folder']};run.update(case_id=r['case'],mode=arm,label=condition['label'],passed=bool(r['strict_arrival']),goal=bool(r['goal_reached']),frames=len(q),duration_s=(len(q)-1)*.02,target=target.tolist(),scene=condition['scene'],scene_source=condition['scene_source'],trajectory='packs/'+group+'.bin.gz',trajectory_offset_frames=offset,detail='packs/'+group+'.details.json.gz',visual='visuals/'+r['task']+'.json.gz',payload_sha256=sha_bytes(payload),trajectory_source_sha256=next(x['sha256'] for x in files if x['name']=='trajectory.npz'),result_source_sha256=next(x['sha256'] for x in files if x['name']=='result.json'),audit_source_sha256=next(x['sha256'] for x in files if x['name']=='audit.json'),input_sha256=condition['input_sha256'],scene_sha256=condition['scene_sha256'],initial_ee_derived_by_fk=True,source_archive=('pure300' if arm=='C' else 'paired300')+'/trials/'+ident,minimum_environment_mm=phys.get('minimum_environment_m',0)*1000,minimum_self_mm=phys.get('minimum_self_m',0)*1000,minimum_proxy_gap_mm=audit.get('minimum_proxy_gap_mm'),bad_physical_intervals=bad,qp_bad_cycles=len(qpa['original_row_violating_cycles']),qp_infeasible_cycles=len(qpa['original_qp_infeasible_cycles']),legacy_url=condition['legacy_url'],video=None)
  assert source['goal_hold_reached']==audit['goal_passed']==r['goal_reached'];assert (len(q)-1)*.02==r['simulation_s']
  cases.append(run);details.setdefault(group,{})[ident]=dict(control_modes_rle=modes,events=events,events_display_filter='ordinary QP decision events omitted here; full original events.jsonl retained in raw evidence',audit=audit,original_feasibility_audit=qpa,validation=val,artifact_files=files)
  rawgroups.setdefault(group,[]).append((run,f));source_validation.append(dict(id=ident,frames=len(q),all_recorded_q_and_ee_values_preserved=True,initial_ee_only_derived=True,source_trajectory_sha256=run['trajectory_source_sha256'],payload_sha256=run['payload_sha256']))
  if len(cases)%100==0:print('runs exported',len(cases),flush=True)
 bundles=[]
 def rawpack(group,entries):
  buf=io.BytesIO()
  with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   for run,f in entries:
    for p in sorted(f.iterdir()):
     if p.is_file():
      zi=zipfile.ZipInfo('trials/'+run['id']+'/'+p.name,(2026,11,10,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED;zi.external_attr=0o100644<<16;z.writestr(zi,p.read_bytes())
    if run['arm']=='B':
     rt=a.ab/'runtime_cases'/run['case_id']
     for name in ['GOAL_PORTS.json','GOAL_BRANCH_PORTS.json','BRANCH_INITIALIZATION_DIAGNOSTICS.json']:
      p=rt/name
      if p.exists():
       zi=zipfile.ZipInfo('terminal_inputs/'+run['case_id']+'/'+name,(2026,11,10,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED;z.writestr(zi,p.read_bytes())
  b=buf.getvalue()
  if len(b)>6*1024**2 and len(entries)>1:
   mid=len(entries)//2;rawpack(group,entries[:mid]);rawpack(group,entries[mid:]);return
  assert len(b)<8*1024**2,(group,len(b));name='evidence/raw/'+group+'__s%02d-s%02d.zip'%(entries[0][0]['seed'],entries[-1][0]['seed']);(o/name).write_bytes(b);record=dict(file=name,sha256=sha_bytes(b),bytes=len(b),runs=[r['id'] for r,f in entries],original_member_bytes_unchanged=True);bundles.append(record)
  for run,f in entries:run['raw_evidence']=name;run['raw_evidence_sha256']=record['sha256'];run['raw_trajectory_member']='trials/'+run['id']+'/trajectory.npz'
 for group,parts in binaries.items():
  b=b''.join(parts);(o/'packs'/(group+'.bin.gz')).write_bytes(gzip.compress(b,compresslevel=8,mtime=0));gzjson(o/'packs'/(group+'.details.json.gz'),details[group])
 for task,data in visuals.items():gzjson(o/'visuals'/(task+'.json.gz'),data)
 for group,entries in rawgroups.items():
  entries.sort(key=lambda x:x[0]['seed'])
  for start in range(0,len(entries),5):rawpack(group,entries[start:start+5])
 comparison={k:{x:v[x] for x in ['source','target','pairs','counts','records','common_success_timing']} for k,v in full['comparisons'].items()};legacy_counts={};links=[]
 for arm in ['C','A','B']:
  counts={'both_passed':0,'legacy_only':0,'controller_only':0,'neither':0}
  for r in cases:
   if r['arm']!=arm:continue
   lp=conditions[r['case_id']]['legacy_passed'];np_=r['physical_safe_arrival'];key='both_passed' if lp and np_ else 'legacy_only' if lp else 'controller_only' if np_ else 'neither';counts[key]+=1;links.append(dict(case=r['case_id'],arm=arm,legacy_passed=lp,controller_physical_safe_arrival=np_,controller_strict=r['strict_arrival'],descriptive_only_not_controlled_regression=True))
  legacy_counts[arm]=counts
 save(o/'data.json',dict(version='liuqp900-20261110',dt=.02,closed=True,raw_runs=900,paired_conditions=300,tasks=legacy['tasks'],arms=full['arms'],comparisons=comparison,by_task=full['by_task'],conditions=condrows,cases=cases,triple_patterns=full['triple_patterns'],legacy=dict(cohort='final_r17 online with RRT planning',passed=298,total=300,protocol_comparable=False,cross_tabs=legacy_counts)))
 save(o/'evidence/input-manifest.json',inputrecords);save(o/'evidence/raw-bundles.json',bundles);save(o/'evidence/fk-checks.json',fkchecks);save(o/'evidence/legacy-links.json',links)
 save(o/'evidence/export-validation.json',dict(passed=True,runs=len(cases),conditions=len(condrows),source_files_crosschecked=sum(len(v) for v in expected.values()),frame_records=sum(c['frames'] for c in cases),frames_roundtrip_exact=True,source_initial_q_exact=True,targets_match_historical_case=True,native_fk_checks=len(fkchecks),max_native_fk_error_m=fkmax,no_experiments_rerun=True,mj_step_called=False,full_input_maps_uploaded=False,input_total_bytes=sum(c['input_bytes'] for c in condrows),raw_evidence_bundles=len(bundles),runs_with_trajectory=len(source_validation)))
 save(o/'evidence/replay-provenance.json',dict(format='gzip little-endian float64 rows [q0..q5,ee_x,ee_y,ee_z]; all saved q and ee values; initial ee only is FK-derived',scope='Recorded kinematic states only, no controller or dynamics rerun. Full original NPZ and audit/log members are in raw bundles.',sources=source_validation))
 for prefix,study in [('ab',a.ab),('c',a.c)]:
  for name in ['PROTOCOL.json','FORMAL_FREEZE.json','FORMAL_JOBS.json','PREFLIGHT_GATE.json','PREFLIGHT_RESULTS.json']:
   p=study/name
   if p.exists():(o/'evidence'/(prefix+'-'+name)).write_bytes(p.read_bytes())
 for name in ['INTEGRITY.json','GOAL_CROSSCHECK.json','ORIGINAL_QP_DIAGNOSTICS.json','GAIN_ATTRIBUTION.json','REGRESSIONS.json','PHYSICAL_SAFE_REGRESSIONS.json','PHYSICAL_SAFETY_FAILURES.json','ELAPSED_TIMES.json']:
  p=a.c/'delivery'/name
  if p.exists():(o/'evidence'/name).write_bytes(p.read_bytes())
 manifest={str(p.relative_to(o)):dict(bytes=p.stat().st_size,sha256=sha_file(p)) for p in o.rglob('*') if p.is_file() and p.name!='asset-manifest.json'};save(o/'asset-manifest.json',manifest)
 print(json.dumps(dict(done=True,runs=len(cases),frames=sum(c['frames'] for c in cases),bundles=len(bundles),bytes=sum(x['bytes'] for x in manifest.values()),fkmax=fkmax,legacy_cross_tabs=legacy_counts)),flush=True)
if __name__=='__main__':main()
