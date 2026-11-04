"""Fixed-snapshot geometry and indexing ablation. No integration, planner or reruns."""
from pathlib import Path
import os,sys,json,hashlib,time,ctypes,subprocess,platform,datetime,gc
from types import SimpleNamespace
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parent
U=Path('/mnt/ssd2/hsjed/ur5_cabinet/v6runs/unified_static_20261102')
B=U/'closeout/final_r17';S=U/'closeout/sphere_r17_pair_20261104';CODE=B/'code'
sys.path[:0]=[str(CODE/'common'),str(CODE/'online'),str(B/'audit_tools')]
import run_protocol_v3_async_online as online
import run_protocol_v3_online_ablation as ablation
from native_mvt import BruteForceAABBIndex
from pointcloud_proxy import PointCloudProxySet
from independent_geometry import IndependentGuard
from unified_policy import POLICY

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)
def ms(fn):
 t=time.perf_counter_ns();value=fn();return (time.perf_counter_ns()-t)/1e6,value
def stats(x):return dict(median=float(np.median(x)),p95=float(np.percentile(x,95)),minimum=float(min(x)),maximum=float(max(x)),n=len(x))
def pointer(a):return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

class NativeBrute:
 simd=False
 def __init__(self,centers,half,padding):
  c=np.asarray(centers,dtype=np.float32);h=np.asarray(half,dtype=np.float32)
  self.lo=np.ascontiguousarray(c-h);self.hi=np.ascontiguousarray(c+h);self.n=len(c);self.query_padding=padding
 def query_spheres(self,centers,radii):
  c=np.ascontiguousarray(centers,dtype=np.float32);r=np.ascontiguousarray(np.asarray(radii,dtype=float)+self.query_padding,dtype=np.float32);off=np.empty(len(c)+1,np.int32);out=np.empty(len(c)*self.n,np.int32)
  n=LIB.brute_batch(pointer(self.lo),pointer(self.hi),self.n,pointer(c),pointer(r),len(c),off.ctypes.data_as(I),out.ctypes.data_as(I),len(out))
  assert n>=0
  return tuple(out[off[i]:off[i+1]].copy() for i in range(len(c)))
 def query_spheres_unordered(self,c,r):return self.query_spheres(c,r)
 def query_sphere(self,c,r):return self.query_spheres(np.asarray(c)[None],np.array([r]))[0]
 def close(self):pass

class EqualPathIndex:
 """Hide native_handle in every arm so narrowphase/fusion architecture stays fixed."""
 simd=False
 def __init__(self,index):self.index=index;self.query_padding=index.query_padding
 def query_spheres(self,c,r):return self.index.query_spheres(c,r)
 def query_spheres_unordered(self,c,r):return self.index.query_spheres(c,r)
 def query_sphere(self,c,r):return self.index.query_sphere(c,r)

def proxyset(C,Q,r,off,ids):
 n=len(C);_,rot=np.linalg.eigh(Q)
 return PointCloudProxySet(raw_points=C,filtered_points=C,centers=C,sphere_radii=r,ellipsoid_shapes=Q,rotations=rot,cluster_keys=np.arange(n)[:,None],filtered_cluster_indices=np.arange(n),maximum_ellipsoid_overshoot=0.,proxy_offset_radii=off,filtered_uncertainty_shapes=Q,proxy_uncertainty_shapes=np.zeros_like(Q),base_sphere_radii=r,base_ellipsoid_shapes=Q,ellipsoid_outer_shapes=Q,proxy_ids=ids,filtered_point_offsets=off,box_half_extents=np.zeros((n,3)))

def build_controller(rep,model,data,robot,proxy,index,target):
 ctl=online._new_controller(rep,model,data,SimpleNamespace(waypoints=[target]),robot,proxy,EqualPathIndex(index),ellipsoid_pair_threads=1)
 ctl.safety_margin=POLICY['operating_margin_m'];ctl.osqp_absolute_tolerance=POLICY['qp_tolerance'];ctl.osqp_relative_tolerance=POLICY['qp_tolerance'];ctl.osqp_max_iterations=POLICY['qp_max_iterations'];ctl.acceleration_limit=POLICY['acceleration_limit']
 return ctl

def main():
 global LIB,I
 os.sched_setaffinity(0,{56});ROOT.mkdir(exist_ok=True);rng=np.random.default_rng(20261104)
 spec=json.loads((B/'EVALUATION_SPEC.json').read_text());ids=[x for x in spec['matrix_cases'] if x.endswith('__s01')];assert len(ids)==15
 selected=[dict(cohort=tag,base=str(base),id=key) for tag,base in [('ellipsoid',B),('sphere',S)] for key in ids]
 sources={}
 for item in selected:
  c=Path(item['base'])/'runs'/item['id'];run=next(c.glob('A-CV-*'));item['run']=str(run)
  for f in ['causal_proxy_snapshots.npz','causal_centervox_snapshots.npz','q_history.npy','scene.xml']:
   p=run/f;sources[str(p)]=digest(p)
 protocol=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection='both cohorts; seed 1 of all 15 tasks; last published snapshot; 3 equally-spaced saved configurations during that snapshot interval',selected=selected,source_hashes=sources,proxy_budget='same full measured-cell count and centers; no removal, merging or hand-selected cap',unknown_space='excluded from index/component timing; unchanged exact boxes retained in separate full-map clearance comparison',coverage='analytic inclusion of saved directional uncertainty sets and scalar offsets; does not independently re-prove sensor noise bound or raw-world coverage',query_arms=['cpp_linear_same_aabb','mvt_scalar','mvt_simd'],query_repeats=21,qp_repeats=9,qp_scope='one-step frozen-pose observed-obstacle LiuQP workload; no planner, no integration, no perception, no frontier rows or execution guard; same zero previous velocity and final target',threading='one pinned CPU, one native geometry thread, BLAS=1',fusion='native fused shortcut hidden in all three query arms to isolate index/SIMD; no frozen runtime changes',timing='warmup then deterministic randomized interleaving; build cost separate; descriptive median/p95, not hard real-time guarantee',no_new_trajectories=True,no_algorithm_modifications=True)
 save(ROOT/'PROTOCOL.json',protocol)
 subprocess.run(['g++','-O3','-fno-tree-vectorize','-ffp-contract=off','-shared','-fPIC',str(ROOT/'brute_aabb.cpp'),'-o',str(ROOT/'brute_aabb.so')],check=True)
 LIB=ctypes.CDLL(str(ROOT/'brute_aabb.so'));I=ctypes.POINTER(ctypes.c_int32);F=ctypes.POINTER(ctypes.c_float);LIB.brute_batch.argtypes=[F,F,ctypes.c_int32,F,F,ctypes.c_int32,I,I,ctypes.c_int32];LIB.brute_batch.restype=ctypes.c_int32
 save(ROOT/'ENVIRONMENT.json',dict(python=sys.version,mujoco=mujoco.__version__,numpy=np.__version__,platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0)),compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],cpu=next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),benchmark_source_sha256=digest(Path(__file__)),linear_reference_sha256=digest(ROOT/'brute_aabb.cpp')))
 rows=[]
 for number,item in enumerate(selected):
  run=Path(item['run']);case=run.parent;cfg=json.loads((case/'case.json').read_text());model=mujoco.MjModel.from_xml_path(str(run/'scene.xml'));data=mujoco.MjData(model);robot=online.build_robot_certificate(model);radii=np.array([r.radius for r in robot]);q=np.load(run/'q_history.npy')
  z=np.load(run/'causal_proxy_snapshots.npz');k=len(z['publish_cycles'])-1;lo,hi=map(int,z['offsets'][k:k+2]);C=z['centers'][lo:hi];Q=z['ellipsoid_shapes'][lo:hi];off=z['uncertainty_offsets'][lo:hi];sr=z['sphere_radii'][lo:hi];boxes=z['box_half_extents'][lo:hi];mask=~np.any(boxes>0,axis=1);n=int(mask.sum());assert mask[:n].all() and not mask[n:].any()
  cz=np.load(run/'causal_centervox_snapshots.npz');cl,ch=map(int,cz['offsets'][k:k+2]);U0=cz['uncertainty_shapes'][cl:ch];values=np.linalg.eigvalsh(Q[:n]);axes=np.sqrt(np.maximum(values,0));radius_error=float(np.max(np.abs(sr[:n]-axes[:,-1])));cover=bool(n==ch-cl and np.array_equal(C[:n],cz['points'][cl:ch]) and np.array_equal(off[:n],cz['scalar_offsets'][cl:ch]) and np.linalg.eigvalsh(Q[:n]-U0).min()>=-1e-15 and radius_error<1e-10);assert cover
  directions=np.array([x for x in np.ndindex(3,3,3) if x!=(1,1,1)],float)-1;directions/=np.linalg.norm(directions,axis=1)[:,None]
  support_loss=np.concatenate([sr[:n]-np.sqrt(np.maximum(np.einsum('i,nij,j->n',d,Q[:n],d),0)) for d in directions]);assert support_loss.min()>-1e-12
  proxy=proxyset(C[:n],Q[:n],sr[:n],off[:n],z['proxy_ids'][lo:hi][:n]);start=min(int(z['publish_cycles'][k]),len(q)-1);frames=np.linspace(start,len(q)-1,3,dtype=int)
  record=dict(id=item['id'],source_cohort=item['cohort'],snapshot=k,publish_cycle=int(z['publish_cycles'][k]),frames=frames.tolist(),measured_proxy_count=n,unknown_box_count=len(C)-n,coverage_passed=cover,max_enclosing_radius_error_m=radius_error,support_excess_mm=stats(support_loss*1000),core_volume_sum_ratio=float(np.sum(sr[:n]**3)/np.sum(np.prod(axes,axis=1))),volume_note='sum of core volumes, overlaps double-counted, excludes isotropic offsets',representations={})
  positions=[]
  for f in frames:data.qpos[:6]=q[f];mujoco.mj_forward(model,data);positions.append(online.certificate_world_positions(data,robot).copy())
  for rep in ['sphere','ellipsoid']:
   build_ms,scalar=ms(lambda:online._build_mvt_only(proxy,rep,radii,simd=False));simd_ms,simd=ms(lambda:online._build_mvt_only(proxy,rep,radii,simd=True))
   half=ablation._environment_half_extents(proxy,rep)+ablation.MVT_NUMERICAL_OUTWARD_PADDING
   brute_ms,brute=ms(lambda:NativeBrute(C[:n],half,scalar.query_padding));oracle=BruteForceAABBIndex(C[:n],half,query_padding=scalar.query_padding)
   indices={'cpp_linear_same_aabb':brute,'mvt_scalar':scalar,'mvt_simd':simd};qt={name:[] for name in indices};qpt={name:[] for name in indices};geom={name:[] for name in indices};solver={name:[] for name in indices};candidate_counts=[];max_matrix_diff=0.;max_command_diff=0.;clear_observed=[];clear_full=[];qp_statuses={name:[] for name in indices};cold_qp={name:[] for name in indices}
   gi=IndependentGuard(C[:n],Q[:n],off[:n],sr[:n],boxes[:n],rep);gf=IndependentGuard(C,Q,off,sr,boxes,rep)
   for fi,(f,pos) in enumerate(zip(frames,positions)):
    expected=brute.query_spheres(pos,radii)
    for j in [0,len(radii)//2,len(radii)-1]:assert np.array_equal(expected[j],oracle.query_sphere(pos[j],radii[j]))
    for name,index in indices.items():
     actual=index.query_spheres(pos,radii);assert all(np.array_equal(a,b) for a,b in zip(expected,actual)),(item['id'],rep,name,'candidate mismatch')
     for _ in range(3):index.query_spheres(pos,radii)
    candidate_counts.append(sum(map(len,expected)))
    for _ in range(protocol['query_repeats']):
     for name in rng.permutation(list(indices)):
      elapsed,_=ms(lambda:indices[name].query_spheres(pos,radii));qt[name].append(elapsed)
    clear_observed.append(gi.reference_gap(pos,radii,reach=.03)*1000);clear_full.append(gf.reference_gap(pos,radii,reach=.03)*1000)
    data.qpos[:6]=q[f];data.qvel[:]=0;mujoco.mj_forward(model,data);target=np.array(cfg['target']);controllers={name:build_controller(rep,model,data,robot,proxy,index,target) for name,index in indices.items()};commands={}
    for name,ctl in controllers.items():
     elapsed,res=ms(lambda:ctl.solve(target));cold_qp[name].append(elapsed);commands[name]=res[0]
    reference=controllers['cpp_linear_same_aabb'].last_qp_arrays
    for name,ctl in controllers.items():
     for a,b in zip(reference,ctl.last_qp_arrays):
      assert a.shape==b.shape and np.array_equal(np.isfinite(a),np.isfinite(b));finite=np.isfinite(a);diff=float(np.max(np.abs(a[finite]-b[finite]),initial=0));max_matrix_diff=max(max_matrix_diff,diff);assert diff<1e-10,(rep,name,'QP mismatch',diff)
     diff=float(np.max(np.abs(commands[name]-commands['cpp_linear_same_aabb'])));max_command_diff=max(max_command_diff,diff);assert diff<1e-8
    for _ in range(3):
     for ctl in controllers.values():ctl.solve(target)
    for _ in range(protocol['qp_repeats']):
     for name in rng.permutation(list(controllers)):
      elapsed,(_,m)=ms(lambda:controllers[name].solve(target));qpt[name].append(elapsed);geom[name].append(float(m.geometry_ms));solver[name].append(float(m.solve_ms));qp_statuses[name].append(m.status)
    del controllers;gc.collect()
   rr=dict(build_ms={'cpp_linear_same_aabb':brute_ms,'mvt_scalar':build_ms,'mvt_simd':simd_ms},index_levels=list(scalar.stats.level_proxy_counts),query_ms={n:stats(x) for n,x in qt.items()},qp_ms={n:stats(x) for n,x in qpt.items()},qp_cold_ms=cold_qp,geometry_ms={n:stats(x) for n,x in geom.items()},solver_ms={n:stats(x) for n,x in solver.items()},qp_statuses={n:sorted(set(x)) for n,x in qp_statuses.items()},candidate_pairs=candidate_counts,observed_clearance_mm=clear_observed,full_map_clearance_mm=clear_full,clearance_cap_mm=30,identical_candidate_ids_and_order=True,max_qp_matrix_difference=max_matrix_diff,max_command_difference=max_command_diff)
   rr['speedup_linear_to_simd']=rr['query_ms']['cpp_linear_same_aabb']['median']/rr['query_ms']['mvt_simd']['median'];rr['speedup_scalar_to_simd']=rr['query_ms']['mvt_scalar']['median']/rr['query_ms']['mvt_simd']['median'];rr['qp_speedup_linear_to_simd']=rr['qp_ms']['cpp_linear_same_aabb']['median']/rr['qp_ms']['mvt_simd']['median'];record['representations'][rep]=rr
   scalar.close();simd.close();del gi,gf
  rows.append(record);save(ROOT/'RESULTS.json',dict(complete=False,completed=len(rows),expected=len(selected),rows=rows));print(json.dumps(dict(completed=len(rows),id=item['id'],cohort=item['cohort'],cells=n,sphere_query_ms=record['representations']['sphere']['query_ms']['mvt_simd']['median'],ellipsoid_query_ms=record['representations']['ellipsoid']['query_ms']['mvt_simd']['median'])),flush=True)
  del z,cz;gc.collect()
 assert all(digest(Path(p))==h for p,h in sources.items())
 save(ROOT/'RESULTS.json',dict(complete=True,completed=len(rows),expected=len(selected),source_integrity_passed=True,rows=rows));print('COMPLETE',flush=True)
if __name__=='__main__':main()
