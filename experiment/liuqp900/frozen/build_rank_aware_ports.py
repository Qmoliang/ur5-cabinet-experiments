from pathlib import Path
import os,json,hashlib,sys
os.sched_setaffinity(0,set(map(int,os.environ.get('STUDY_CPUS','48,49,50,51').split(','))))
from pilot_adapter import ROOT,build
from collision_oracle import CollisionOracle
from rank_aware_ports import initialize
entry=json.loads((ROOT/'INPUTS.json').read_text())[0];source=json.loads((ROOT/'GOAL_PORTS.json').read_text());assert source['task']==entry['task'];ctx=build(entry['task']);oracle=CollisionOracle(ctx.model,ctx.robot,ctx.arrays,ctx.q0,margin=ctx.controller.safety_margin)
out=initialize(ctx,oracle,source);out['source_goal_sha256']=hashlib.sha256((ROOT/'GOAL_PORTS.json').read_bytes()).hexdigest();(ROOT/'BRANCH_INITIALIZATION_DIAGNOSTICS.json').write_text(json.dumps(out,indent=2))
if not out['ports']:raise RuntimeError('No usable static goal/entry port under unchanged safety and numeric thresholds')
(ROOT/'GOAL_BRANCH_PORTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status='ready',source_rank=out['ports'][0]['source_rank'],rank=out['ports'][0]['entry_rank_diagnostics'],ray_count=len(out['ports'][0]['all_entry_rays'])),indent=2))
