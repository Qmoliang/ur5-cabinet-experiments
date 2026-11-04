const $=id=>document.getElementById(id),f=(x,n=3)=>Number(x).toFixed(n);
async function get(p){const r=await fetch(p);if(!r.ok)throw Error(p+': '+r.status);return r.json();}
function row(id,values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}$(id).append(tr);}
try{
 const [paired,s,strong,index]=await Promise.all([get('paired-report.json'),get('summary.json'),get('strong-summary.json'),get('../data.json')]);
 if(!paired.study_complete||!s.complete||!strong.complete)throw Error('Study not complete');
 const labels=Object.fromEntries(index.tasks.map(x=>[x.id,x.label]));
 for(const [task,r] of Object.entries(paired.by_task))row('paired-tasks',[labels[task]||task,`${r.sphere_safe_arrivals}/20`,`${r.ellipsoid_safe_arrivals}/20`,`+${r.ellipsoid_safe_arrivals-r.sphere_safe_arrivals}`]);
 const g=s.geometry;$('coverage').textContent=`${g.snapshots}/${g.snapshots}`;$('support').textContent=`${f(g.support_excess_mm_median_of_snapshot_medians,2)} mm`;$('clearance').textContent=`${f(g.near_clearance_gain_mm_median,2)} mm`;
 $('geometry-detail').textContent=`每份地图的观测代理数量为 ${g.proxy_count_min.toLocaleString()}–${g.proxy_count_max.toLocaleString()}，两种表示严格相同。球的方向支持膨胀，各快照P95的中位数为 ${f(g.support_excess_mm_median_of_snapshot_p95,2)} mm。${g.near_pose_count} 个姿态的至少一种表示净空低于30 mm，其他姿态的值被截断。`;
 $('clearance-boundary').textContent=`90个预定离散姿态中，${g.observed_ellipsoid_safe_sphere_blocked} 个出现“椭球净空≥6 mm而球<6 mm”，反向为${g.observed_sphere_safe_ellipsoid_blocked}个；${g.observed_both_above_6}个两者均≥6 mm。按6.5 mm正常运行裕量比较，仅椭球达到门槛的仍为${g.operating_ellipsoid_safe_sphere_blocked}个。这些点不能解释或证明全部23项闭环差异的具体根因，更不能推出球表示下不存在其他可行路径。`;
 $('equivalence').textContent=`首轮 ${s.equivalence.candidate_query_sets.toLocaleString()} 组候选查询的ID与顺序完全一致；${s.equivalence.qp_instances} 个固定姿态QP实例的矩阵与速度指令最大差异均为 ${s.equivalence.max_qp_matrix_difference}。强基线补充检查也通过，未通过漏掉候选换取速度。`;
 for(const [rep,label] of [['sphere','球'],['ellipsoid','椭球']]){
  const a=s[rep],q=a.query_ms_median_of_snapshot_medians,b=strong[rep];
  row('query-basic',[label,f(q.cpp_linear_same_aabb),f(q.mvt_scalar),f(q.mvt_simd),f(a.scalar_to_simd_speedup_median,2)+'×']);
  row('query-strong',[label,f(b.query_ms.linear_global_avx2),f(b.query_ms.mvt_simd),f(b.query_speedup_median,2)+'×',f(b.qp_ms.linear_global_avx2),f(b.qp_ms.mvt_simd),f(b.qp_speedup_median,2)+'×']);
 }
 $('performance-conclusion').textContent=strong.conclusion_zh+` 在相同快照的配对比较中，椭球/球的MVT查询时间比中位数为${f(strong.representation_cost.ellipsoid_over_sphere_query_ratio_median,3)}，QP组件时间比为${f(strong.representation_cost.ellipsoid_over_sphere_qp_ratio_median,3)}；这只是本工作负载的表现，不表示单次椭球距离计算更便宜或椭球在所有地图上都更快。`;
 $('build-cost').textContent=`MVT SIMD建表耗时中位数：球 ${f(strong.sphere.build_ms.mvt_simd,2)} ms、椭球 ${f(strong.ellipsoid.build_ms.mvt_simd,2)} ms；强线性基线分别为 ${f(strong.sphere.build_ms.linear_global_avx2,2)} 和 ${f(strong.ellipsoid.build_ms.linear_global_avx2,2)} ms。按查询差额摊销，球/椭球的中位数约需 ${strong.sphere.amortization_batches_median} / ${strong.ellipsoid.amortization_batches_median} 批查询。地图频繁更新时必须计入这笔成本。`;
 document.body.dataset.ready='true';
}catch(e){console.error(e);$('load-error').hidden=false;}
