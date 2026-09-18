"""Frozen-cycle MVT baseline comparison, identical proxies and queries."""
import json,time
import numpy as np
import mujoco
from diagnose_et30_terminal import ROOT,OUT,read_csv,proxy_object
from formal_protocol_v3_viewer import RaggedSnapshots
from run_protocol_v3_online_ablation import _environment_half_extents
from native_mvt import NativeMultilevelMVT,BruteForceAABBIndex
from radius_binned_mvt import RadiusBinnedMVT
from model import build_robot_certificate,set_configuration,certificate_world_positions
rows=[r for r in read_csv(ROOT/'tables/experiment_07/T27_drawer_formal_runs_07_2.csv') if r['group']=='ET30']
results=[]
for row in rows:
    run=ROOT/row['run']; model=mujoco.MjModel.from_xml_path(str(run/'scene.xml')); data=mujoco.MjData(model)
    robot=build_robot_certificate(model); radii=np.array([s.radius for s in robot]); qhist=np.load(run/'q_history.npy')
    history=RaggedSnapshots(run/'causal_proxy_snapshots.npz')
    for cycle in [100,300,800,1500,2249]:
        _,a=history.at_cycle(cycle); p=proxy_object(a); half=_environment_half_extents(p,'ellipsoid')+5e-6
        set_configuration(model,data,qhist[cycle-1]); centers=certificate_world_positions(data,robot)
        pad=.046005; maxq=float(max(radii)+pad)
        ml=NativeMultilevelMVT(p.centers,half,.015,maxq,query_padding=pad,simd=True)
        tight_half=(np.sqrt(np.maximum(np.diagonal(p.base_ellipsoid_shapes,axis1=1,axis2=2),0))
            +np.sqrt(np.maximum(np.diagonal(p.proxy_uncertainty_shapes,axis1=1,axis2=2),0))
            +p.proxy_offset_radii[:,None]+5e-6)
        indexes={'tight_multilevel_simd':NativeMultilevelMVT(p.centers,tight_half,.015,maxq,query_padding=pad,simd=True),
            'tight_full_AABB':BruteForceAABBIndex(p.centers,tight_half,query_padding=pad),
            'multilevel_simd':ml,
            'multilevel_scalar':NativeMultilevelMVT(p.centers,half,.015,maxq,query_padding=pad,simd=False),
            'single_level_simd':NativeMultilevelMVT(p.centers,half,ml.stats.level_voxel_sizes[-1],maxq,query_padding=pad,simd=True),
            'radius_binned':RadiusBinnedMVT(p.centers,half,.015,pad).build(radii),
            'numpy_full_AABB':BruteForceAABBIndex(p.centers,half,query_padding=pad)}
        timings={k:[] for k in indexes}; candidates={k:v.query_spheres(centers,radii) for k,v in indexes.items()}
        for rep in range(20):
            for name in list(indexes)[rep%len(indexes):]+list(indexes)[:rep%len(indexes)]:
                start=time.perf_counter_ns(); indexes[name].query_spheres(centers,radii)
                timings[name].append((time.perf_counter_ns()-start)*1e-6)
        ref=candidates['numpy_full_AABB']
        checks={}
        for name,cc in candidates.items():
            rr=candidates['tight_full_AABB'] if name.startswith('tight_') else ref
            checks[name]=dict(missing=sum(len(set(r)-set(c)) for r,c in zip(rr,cc)),extra=sum(len(set(c)-set(r)) for r,c in zip(rr,cc)))
        assert all(v['missing']==0 for v in checks.values())
        results.append(dict(seed=int(row['seed']),cycle=cycle,proxies=len(p.centers),robot_spheres=len(radii),
            maximum_robot_radius_mm=1000*max(radii),query_half_bound_mm=1000*maxq,
            proxy_max_half_mm_quantiles=(1000*np.quantile(np.max(half,axis=1),[0,.5,.95,1])).tolist(),
            level_sizes_mm=(1000*np.array(ml.stats.level_voxel_sizes)).tolist(),level_counts=list(ml.stats.level_proxy_counts),
            total_possible_pairs=len(radii)*len(p.centers),candidate_pairs=sum(len(c) for c in ref),tight_candidate_pairs=sum(len(c) for c in candidates['tight_full_AABB']),tight_aabb_contained_in_old=bool(np.all(tight_half<=half+1e-10)),checks=checks,
            median_ms={k:float(np.median(v)) for k,v in timings.items()},p95_ms={k:float(np.percentile(v,95)) for k,v in timings.items()}))
        for obj in indexes.values():
            if hasattr(obj,'close'):obj.close()
    print('completed seed',row['seed'],flush=True)
summary={name:float(np.median([x['median_ms'][name] for x in results])) for name in results[0]['median_ms']}
(OUT/'mvt_comparison.json').write_text(json.dumps(dict(diagnostic_only=True,scope='25 frozen ET30 cycle queries; 20 interleaved repetitions each; microbenchmark, not end-to-end control',summary_median_ms=summary,cases=results),indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)

