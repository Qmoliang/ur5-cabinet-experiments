"""Supplementary stronger baseline, same frozen selection; no trajectory runs."""
from pathlib import Path
import os,ctypes,json,subprocess,time,gc,datetime
import numpy as np,mujoco
import benchmark_snapshots as bench
R=Path(__file__).resolve().parent
class LinearAVX2:
 simd=False
 def __init__(self,C,H,padding):
  c=np.asarray(C,np.float32);h=np.asarray(H,np.float32);lo=c-h;hi=c+h;self.n=len(c);self.lo=np.ascontiguousarray(lo.T);self.hi=np.ascontiguousarray(hi.T);self.gl=np.ascontiguousarray(lo.min(axis=0));self.gh=np.ascontiguousarray(hi.max(axis=0));self.query_padding=padding;self.single_out=np.empty(self.n,np.int32);self.single_off=np.empty(2,np.int32)
 def query_spheres(self,c,r):
  c=np.ascontiguousarray(c,np.float32);r=np.ascontiguousarray(np.asarray(r,float)+self.query_padding,np.float32);off=np.empty(len(c)+1,np.int32);out=np.empty(len(c)*self.n,np.int32)
  n=LIB.linear_avx2(bench.pointer(self.lo),bench.pointer(self.hi),self.n,bench.pointer(self.gl),bench.pointer(self.gh),bench.pointer(c),bench.pointer(r),len(c),off.ctypes.data_as(I),out.ctypes.data_as(I),len(out));assert n>=0
  return tuple(out[off[i]:off[i+1]].copy() for i in range(len(c)))
 def query_spheres_unordered(self,c,r):return self.query_spheres(c,r)
 def query_sphere(self,c,r):
  c=np.ascontiguousarray(c,np.float32);r=np.array([float(r)+self.query_padding],np.float32)
  n=LIB.linear_avx2(bench.pointer(self.lo),bench.pointer(self.hi),self.n,bench.pointer(self.gl),bench.pointer(self.gh),bench.pointer(c),bench.pointer(r),1,self.single_off.ctypes.data_as(I),self.single_out.ctypes.data_as(I),self.n);assert n>=0
  return self.single_out[:n].copy()
def main():
 global LIB,I
 os.sched_setaffinity(0,{56});protocol=json.loads((R/'PROTOCOL.json').read_text());rng=np.random.default_rng(681704)
 bench.save(R/'STRONG_PROTOCOL.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='test whether the index benefit survives a stronger full-scan implementation',version=2,single_query_buffer='reused output buffer in both linear AVX2 and NativeMultilevelMVT; v1 results retained in supplementary_v1',source_selection='identical to PROTOCOL.json, no outcome-based selection',arms=['linear_global_avx2','mvt_scalar','mvt_simd'],query_repeats=21,qp_repeats=9,same_unfused_qp_path=True,representation_order='alternating by snapshot',no_new_trajectories=True))
 subprocess.run(['g++','-O3','-mavx2','-ffp-contract=off','-shared','-fPIC',str(R/'linear_avx2.cpp'),'-o',str(R/'linear_avx2.so')],check=True)
 LIB=ctypes.CDLL(str(R/'linear_avx2.so'));F=ctypes.POINTER(ctypes.c_float);I=ctypes.POINTER(ctypes.c_int32);LIB.linear_avx2.argtypes=[F,F,ctypes.c_int32,F,F,F,F,ctypes.c_int32,I,I,ctypes.c_int32];LIB.linear_avx2.restype=ctypes.c_int32
 rows=[]
 for number,item in enumerate(protocol['selected']):
  run=Path(item['run']);cfg=json.loads((run.parent/'case.json').read_text());model=mujoco.MjModel.from_xml_path(str(run/'scene.xml'));data=mujoco.MjData(model);robot=bench.online.build_robot_certificate(model);radii=np.array([x.radius for x in robot]);q=np.load(run/'q_history.npy');z=np.load(run/'causal_proxy_snapshots.npz');k=len(z['publish_cycles'])-1;lo,hi=map(int,z['offsets'][k:k+2]);boxes=z['box_half_extents'][lo:hi];n=int((~np.any(boxes>0,axis=1)).sum());C=z['centers'][lo:hi][:n];Q=z['ellipsoid_shapes'][lo:hi][:n];sr=z['sphere_radii'][lo:hi][:n];off=z['uncertainty_offsets'][lo:hi][:n];ps=bench.proxyset(C,Q,sr,off,z['proxy_ids'][lo:hi][:n]);start=min(int(z['publish_cycles'][k]),len(q)-1);frames=np.linspace(start,len(q)-1,3,dtype=int);record=dict(id=item['id'],source_cohort=item['cohort'],representations={})
  for rep in (['sphere','ellipsoid'] if number%2==0 else ['ellipsoid','sphere']):
   build_scalar,scalar=bench.ms(lambda:bench.online._build_mvt_only(ps,rep,radii,simd=False));build_simd,simd=bench.ms(lambda:bench.online._build_mvt_only(ps,rep,radii,simd=True));half=bench.ablation._environment_half_extents(ps,rep)+bench.ablation.MVT_NUMERICAL_OUTWARD_PADDING;build_linear,linear=bench.ms(lambda:LinearAVX2(C,half,scalar.query_padding));indices={'linear_global_avx2':linear,'mvt_scalar':scalar,'mvt_simd':simd};qt={n:[] for n in indices};qpt={n:[] for n in indices};maxdiff=0.;maxcmd=0.
   for f in frames:
    data.qpos[:6]=q[f];data.qvel[:]=0;mujoco.mj_forward(model,data);pos=bench.online.certificate_world_positions(data,robot);expected=linear.query_spheres(pos,radii)
    for name,index in indices.items():
     actual=index.query_spheres(pos,radii);assert all(np.array_equal(a,b) for a,b in zip(expected,actual)),(item,rep,name)
     for _ in range(3):index.query_spheres(pos,radii)
    for _ in range(21):
     for name in rng.permutation(list(indices)):elapsed,_=bench.ms(lambda:indices[name].query_spheres(pos,radii));qt[name].append(elapsed)
    target=np.array(cfg['target']);controllers={name:bench.build_controller(rep,model,data,robot,ps,idx,target) for name,idx in indices.items()};commands={n:ctl.solve(target)[0] for n,ctl in controllers.items()};reference=controllers['linear_global_avx2'].last_qp_arrays
    for name,ctl in controllers.items():
     for a,b in zip(reference,ctl.last_qp_arrays):
      finite=np.isfinite(a);assert a.shape==b.shape and np.array_equal(finite,np.isfinite(b)) and np.array_equal(a[~finite],b[~finite]);d=float(np.max(np.abs(a[finite]-b[finite]),initial=0));maxdiff=max(maxdiff,d);assert d<1e-10
     d=float(np.max(np.abs(commands[name]-commands['linear_global_avx2'])));maxcmd=max(maxcmd,d);assert d<1e-8
    for _ in range(3):
     for ctl in controllers.values():ctl.solve(target)
    for _ in range(9):
     for name in rng.permutation(list(controllers)):elapsed,_=bench.ms(lambda:controllers[name].solve(target));qpt[name].append(elapsed)
    del controllers;gc.collect()
   record['representations'][rep]=dict(query_ms={k:bench.stats(v) for k,v in qt.items()},qp_ms={k:bench.stats(v) for k,v in qpt.items()},build_ms={'linear_global_avx2':build_linear,'mvt_scalar':build_scalar,'mvt_simd':build_simd},candidates_identical=True,max_qp_difference=maxdiff,max_command_difference=maxcmd)
   scalar.close();simd.close()
  rows.append(record);bench.save(R/'STRONG_RESULTS.json',dict(complete=False,completed=len(rows),rows=rows));print('strong',len(rows),item['id'],item['cohort'],flush=True)
 for p,h in protocol['source_hashes'].items():assert bench.digest(Path(p))==h
 bench.save(R/'STRONG_RESULTS.json',dict(complete=True,completed=len(rows),source_integrity_passed=True,rows=rows));print('STRONG_COMPLETE',flush=True)
if __name__=='__main__':main()
