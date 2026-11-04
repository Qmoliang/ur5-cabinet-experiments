from pathlib import Path
import json,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
data=json.loads((R/'RESULTS.json').read_text());assert data['complete'];rows=data['rows'];summary={}
def median(x):return float(np.median(x))
for rep in ['sphere','ellipsoid']:
 rr=[r['representations'][rep] for r in rows]
 summary[rep]={
  'query_ms_median_of_snapshot_medians':{n:median([x['query_ms'][n]['median'] for x in rr]) for n in ['cpp_linear_same_aabb','mvt_scalar','mvt_simd']},
  'qp_ms_median_of_snapshot_medians':{n:median([x['qp_ms'][n]['median'] for x in rr]) for n in ['cpp_linear_same_aabb','mvt_scalar','mvt_simd']},
  'build_ms_median':{n:median([x['build_ms'][n] for x in rr]) for n in ['cpp_linear_same_aabb','mvt_scalar','mvt_simd']},
  'linear_to_simd_speedup_median':median([x['speedup_linear_to_simd'] for x in rr]),
  'scalar_to_simd_speedup_median':median([x['speedup_scalar_to_simd'] for x in rr]),
  'linear_to_simd_qp_speedup_median':median([x['qp_speedup_linear_to_simd'] for x in rr]),
  'simd_faster_than_scalar_snapshots':sum(x['speedup_scalar_to_simd']>1 for x in rr),
  'simd_faster_than_linear_snapshots':sum(x['speedup_linear_to_simd']>1 for x in rr)}
 break_even=[]
 for x in rr:
  gain=x['query_ms']['cpp_linear_same_aabb']['median']-x['query_ms']['mvt_simd']['median']
  if gain>0:break_even.append(max(0,math.ceil((x['build_ms']['mvt_simd']-x['build_ms']['cpp_linear_same_aabb'])/gain)))
 summary[rep]['build_amortization_query_batches_median']=median(break_even) if break_even else None
obs_s=np.array([v for r in rows for v in r['representations']['sphere']['observed_clearance_mm']]);obs_e=np.array([v for r in rows for v in r['representations']['ellipsoid']['observed_clearance_mm']]);full_s=np.array([v for r in rows for v in r['representations']['sphere']['full_map_clearance_mm']]);full_e=np.array([v for r in rows for v in r['representations']['ellipsoid']['full_map_clearance_mm']]);assert np.min(obs_e-obs_s)>-1e-5 and np.min(full_e-full_s)>-1e-5
near=(obs_s<30-1e-6)|(obs_e<30-1e-6)
summary['geometry']=dict(snapshots=len(rows),saved_pose_queries=len(obs_s),proxy_count_min=min(r['measured_proxy_count'] for r in rows),proxy_count_max=max(r['measured_proxy_count'] for r in rows),all_equal_counts_and_cell_coverage_passed=all(r['coverage_passed'] for r in rows),support_excess_mm_median_of_snapshot_medians=median([r['support_excess_mm']['median'] for r in rows]),support_excess_mm_median_of_snapshot_p95=median([r['support_excess_mm']['p95'] for r in rows]),observed_ellipsoid_safe_sphere_blocked=int(np.count_nonzero((obs_e>=6)&(obs_s<6))),observed_sphere_safe_ellipsoid_blocked=int(np.count_nonzero((obs_s>=6)&(obs_e<6))),full_map_ellipsoid_safe_sphere_blocked=int(np.count_nonzero((full_e>=6)&(full_s<6))),full_map_sphere_safe_ellipsoid_blocked=int(np.count_nonzero((full_s>=6)&(full_e<6))),near_pose_count=int(near.sum()),near_clearance_gain_mm_median=median((obs_e-obs_s)[near]) if near.any() else None,observed_both_above_6=int(np.count_nonzero((obs_e>=6)&(obs_s>=6))),observed_both_below_6=int(np.count_nonzero((obs_e<6)&(obs_s<6))),clearance_cap_mm=30)
summary['equivalence']=dict(candidate_query_sets=30*2*3*65,all_ids_and_order_identical=all(r['representations'][p]['identical_candidate_ids_and_order'] for r in rows for p in ['sphere','ellipsoid']),qp_instances=30*2*3,max_qp_matrix_difference=max(r['representations'][p]['max_qp_matrix_difference'] for r in rows for p in ['sphere','ellipsoid']),max_velocity_command_difference=max(r['representations'][p]['max_command_difference'] for r in rows for p in ['sphere','ellipsoid']))
summary['representation_cost']=dict(ellipsoid_over_sphere_simd_query_ratio_median=median([r['representations']['ellipsoid']['query_ms']['mvt_simd']['median']/r['representations']['sphere']['query_ms']['mvt_simd']['median'] for r in rows]),ellipsoid_over_sphere_qp_ratio_median=median([r['representations']['ellipsoid']['qp_ms']['mvt_simd']['median']/r['representations']['sphere']['qp_ms']['mvt_simd']['median'] for r in rows]))
B=Path('/mnt/ssd2/hsjed/ur5_cabinet/v6runs/unified_static_20261102/closeout/final_r17');changed=[k for k,h in json.loads((B/'FROZEN_MANIFEST.json').read_text()).items() if hashlib.sha256((B/k).read_bytes()).hexdigest()!=h];assert not changed
summary.update(complete=True,source_integrity_passed=data['source_integrity_passed'],runtime_integrity_passed=True,no_new_trajectories=True,statistic_note='Each snapshot is weighted equally. Speedups are medians of per-snapshot paired ratios, not ratios of displayed medians.',limitations=['Snapshot seed 1 from each task and both source cohorts; not all saved maps, seeds or worst-case queries.','Coverage is for saved measured-cell uncertainty sets under their existing error assumptions, not arbitrary unseen world geometry.','Clearance is clipped at +30 mm; these are fixed-pose certificates, not global path existence proofs.','Observed proxy sums are not union volume, and core volume excludes isotropic offsets.','One pinned CPU, warm repeated fixed-pose solves; index arms randomized; representation arms run sphere then ellipsoid. Not an end-to-end real-time benchmark.','Common unfused narrowphase path isolates indexing; full production fusion, sensing, planning, frontier rows and execution protection are outside this timing.','Original LiuQP sphere-tree construction is not benchmarked; it constructs geometry and is not the same abstraction as an AABB query index.'])
(R/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f5f5ee','axes.facecolor':'#f5f5ee','savefig.facecolor':'#f5f5ee'})
fig,ax=plt.subplots(2,2,figsize=(12,8.6),constrained_layout=True);colors=['#b87843','#315b48'];names=['Linear AABB','MVT scalar','MVT SIMD'];keys=['cpp_linear_same_aabb','mvt_scalar','mvt_simd'];x=np.arange(3)
for k,rep in enumerate(['sphere','ellipsoid']):
 ax[0,0].bar(x+(k-.5)*.32,[summary[rep]['query_ms_median_of_snapshot_medians'][n] for n in keys],.32,color=colors[k],label=rep)
 ax[0,1].bar(x+(k-.5)*.32,[summary[rep]['qp_ms_median_of_snapshot_medians'][n] for n in keys],.32,color=colors[k],label=rep)
for a in ax[0]:a.set_xticks(x,names);a.set_yscale('log');a.set_ylabel('Median time (ms, log scale)');a.legend(frameon=False);a.grid(axis='y',alpha=.18)
ax[0,0].set_title('A  Broadphase: same candidates, 65 robot spheres',loc='left');ax[0,1].set_title('B  Fixed-pose LiuQP component workload',loc='left')
support=[r['support_excess_mm']['median'] for r in rows];ax[1,0].scatter(np.arange(1,31),support,c=[colors[1] if r['source_cohort']=='ellipsoid' else colors[0] for r in rows],s=30);ax[1,0].set_xlabel('Snapshot (1–15 ellipsoid source; 16–30 sphere source)');ax[1,0].set_ylabel('Median sphere support excess (mm)');ax[1,0].set_title('C  Equal count, same center, complete cell support',loc='left');ax[1,0].grid(alpha=.18)
lo=min(float(obs_s.min()),float(obs_e.min()),0)-1;ax[1,1].plot([lo,31],[lo,31],color='#879082',ls='--');ax[1,1].scatter(obs_s,obs_e,c='#315b48',alpha=.65,s=35);ax[1,1].axhline(6,color='#b87843',ls=':');ax[1,1].axvline(6,color='#b87843',ls=':');ax[1,1].set_xlim(lo,31);ax[1,1].set_ylim(lo,31);ax[1,1].set_xlabel('Sphere observed-map clearance (mm)');ax[1,1].set_ylabel('Ellipsoid observed-map clearance (mm)');ax[1,1].set_title('D  Saved poses; both axes capped at +30 mm',loc='left')
fig.suptitle('Frozen-snapshot representation and query study | 30 maps, 90 saved poses',fontsize=14)
fig.savefig(R/'snapshot-comparison.png',dpi=180);fig.savefig(R/'snapshot-comparison.svg');plt.close(fig)
print(json.dumps(summary,indent=2),flush=True)
