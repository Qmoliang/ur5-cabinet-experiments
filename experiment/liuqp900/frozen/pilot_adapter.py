from pathlib import Path
from dataclasses import make_dataclass
from types import SimpleNamespace
import os,sys,json,time
import numpy as np,mujoco

ROOT=Path(__file__).resolve().parent; U=ROOT.parent.parent; BASE=U/'closeout/final_r17'; CODE=ROOT/'code'
sys.path[:0]=[str(CODE/'common'),str(CODE/'online')]
os.environ.pop('UNIFIED_PLANNER_PROCESS',None)
from unified_policy import POLICY
import run_protocol_v3_async_online as online
import mixed_geometry as mixed
from unified_self_rows import SelfCollisionRows
from dynamic_channel import DynamicChannel
from static_safety import ExecutionGuard
from proxy_geometry import ExactProxyGuard
mixed.install(online)
Proxy=make_dataclass('PilotProxy',[(n,object,None) for n in mixed.FIELDS])

FORBIDDEN_COUNTS={}
def prohibit_planning():
    import planner_core, guidance_core
    def blocked(*a,**k):
        FORBIDDEN_COUNTS['planning_attempt']=FORBIDDEN_COUNTS.get('planning_attempt',0)+1
        raise RuntimeError('No-path low-dimensional pilot forbids planning/guidance')
    planner_core.search=blocked; guidance_core.search=blocked
    guidance_core.UnifiedGuidance.__init__=blocked

def build(task,representation='ellipsoid',start=None,previous=None):
    begun=time.monotonic(); entry=next(x for x in json.loads((ROOT/'INPUTS.json').read_text()) if x['task']==task)
    raw=np.load(entry['input_file']); allowed=['q0','target','centers','radii','base_radii','shapes','outer_shapes','uncertainty','offsets','ids','boxes']
    # Deliberately never read the archive's `path` key.
    z={key:raw[key] for key in allowed};raw.close()
    meta=json.loads(Path(entry['metadata']).read_text());q0=z['q0'].copy() if start is None else np.asarray(start).copy();target=z['target'].copy()
    model=mujoco.MjModel.from_xml_path(entry['scene']);data=mujoco.MjData(model);online.set_configuration(model,data,q0)
    robot=online.build_robot_certificate(model);rad=np.asarray([x.radius for x in robot])
    ps=Proxy(centers=z['centers'],sphere_radii=z['radii'],base_sphere_radii=z['base_radii'],ellipsoid_shapes=z['shapes'],base_ellipsoid_shapes=z['shapes'],ellipsoid_outer_shapes=z['outer_shapes'],proxy_uncertainty_shapes=z['uncertainty'],proxy_offset_radii=z['offsets'],proxy_ids=z['ids'],box_half_extents=z['boxes'])
    assert not np.any(ps.proxy_uncertainty_shapes)
    arrays=dict(centers=ps.centers,shapes=ps.base_ellipsoid_shapes,radii=ps.sphere_radii,offsets=ps.proxy_offset_radii,boxes=ps.box_half_extents,representation=representation)
    n=mixed.split_count(ps);obs=mixed.observed(ps,n);index=online._build_mvt_only(obs,representation,rad,simd=True)
    controller=online._new_controller(representation,model,data,SimpleNamespace(name=task),robot,ps,SimpleNamespace(_observed_mvt=index),ellipsoid_pair_threads=2,ellipsoid_pair_affinity_mask=sum(1<<x for x in sorted(os.sched_getaffinity(0))))
    for key,value in [('safety_margin',POLICY['operating_margin_m']),('task_gain',POLICY['task_gain']),('max_task_speed',POLICY['max_task_speed']),('osqp_absolute_tolerance',POLICY['qp_tolerance']),('osqp_relative_tolerance',POLICY['qp_tolerance']),('osqp_max_iterations',POLICY['qp_max_iterations']),('acceleration_limit',POLICY['acceleration_limit'])]:setattr(controller,key,value)
    controller.extra_joint_goal=None;controller.extra_task_frames=()
    # Self-pair membership is inherited from the ORIGINAL start, including snapshot replay.
    saved=q0.copy();online.set_configuration(model,data,z['q0'])
    selfrows=SelfCollisionRows(model,data,robot,POLICY['dt_s'],online.certificate_world_positions)
    online.set_configuration(model,data,saved)
    assert len(selfrows.pairs)==meta['original_self_pairs']
    channel=DynamicChannel('liuqp',POLICY['dt_s'],controller.safety_margin,controller.repulsive_speed,controller.contact_distance)
    controller.extra_rows=lambda p,j,r:channel.rows(p,j,r)+selfrows.rows(p,j,r)
    guard=ExecutionGuard(model,robot,POLICY['dt_s'],POLICY['environment_margin_m'],POLICY['acceleration_limit']);guard.set_self_pairs(selfrows.pairs)
    proxy_guard=ExactProxyGuard(arrays)
    controller.applied_velocity=np.zeros(6) if previous is None else np.asarray(previous).copy()
    prohibit_planning()
    return SimpleNamespace(model=model,data=data,robot=robot,controller=controller,guard=guard,proxy_guard=proxy_guard,q0=q0,target=target,metadata=meta,arrays=arrays,entry=entry,selfrows=selfrows,setup_s=time.monotonic()-begun,observed=n,boxes=len(ps.centers)-n)
